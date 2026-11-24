#!/usr/bin/env python3
"""Directional transfer: train on {30s, 1min, 2min}, test on {5min}.

Strictly harder than LOIO for the 5min instance class: the model never
sees ANY 5min log during training. Both the 15min logs and all 5min logs
are excluded from training (15min is excluded from test too).

5 model configs (K=20 = all 20 features, no Gini subset):
  - forest_dNone_K20  (deployed)
  - forest_d16_K20
  - forest_d8_K20
  - tree_d4_K20
  - tree_dNone_K20

3 baselines:
  - random         (top-1 = 1/9, top-3 = 3/9, regret = max(ei) - mean(ei))
  - best_avg_pair  (per-pair train mean ei_pct argmax, constant on test)
  - best_avg_top3  (per-pair train mean ei_pct argsort top-3, constant on test)

State exclusion: max(ei_pct over 9 USED) > 0. State-weighted aggregation.

Output:
  cross_pair_ranking_grid_2026-05-25/tables/directional_transfer_5min.csv
  cross_pair_ranking_grid_2026-05-25/DIRECTIONAL_TRANSFER_REPORT.md
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
PRIOR_SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"
sys.path.insert(0, str(PRIOR_SCRIPTS))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
xpr = _load("xpr", PRIOR_SCRIPTS / "01_cross_pair_ranking.py")

OUT_CSV = REPO / "cross_pair_ranking_grid_2026-05-25" / "tables" / "directional_transfer_5min.csv"
OUT_MD  = REPO / "cross_pair_ranking_grid_2026-05-25" / "DIRECTIONAL_TRANSFER_REPORT.md"

TRAIN_SIZES  = {"30_s", "01_min", "02_min"}
TEST_SIZE    = "05_min"
EXCLUDE_SIZE = "15min"

CONFIGS = [
    {"tag": "forest_dNone_K20", "type": "forest", "max_depth": None, "K": 20},
    {"tag": "forest_d16_K20",   "type": "forest", "max_depth": 16,   "K": 20},
    {"tag": "forest_d8_K20",    "type": "forest", "max_depth": 8,    "K": 20},
    {"tag": "tree_d4_K20",      "type": "tree",   "max_depth": 4,    "K": 20},
    {"tag": "tree_dNone_K20",   "type": "tree",   "max_depth": None, "K": 20},
]

TEST_BUCKETS = ["05_min_d03", "05_min_d05", "05_min_d10"]


def make_clf(cfg):
    if cfg["type"] == "forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=cfg["max_depth"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return DecisionTreeClassifier(
        max_depth=cfg["max_depth"], class_weight="balanced", random_state=42,
    )


def fit_and_predict(df, train_mask, test_mask, cfg):
    """Returns P (n_test, 9) — per-pair P(class=1) on test rows.
    K=20 → no feature subset; aggregated_top_k_indices(20) returns range(20)."""
    feat_idx = xpr.aggregated_top_k_indices(cfg["K"])
    X = df[xpr.FEATURE_NAMES_20].values
    X_tr = X[train_mask][:, feat_idx]
    X_te = X[test_mask][:, feat_idx]
    P = np.zeros((test_mask.sum(), 9), dtype=float)
    for j, p in enumerate(xpr.USED_PAIRS):
        y_tr = df[p].values[train_mask].astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            P[:, j] = float(y_tr.mean()); continue
        clf = make_clf(cfg)
        clf.fit(X_tr, y_tr)
        P[:, j] = clf.predict_proba(X_te)[:, 1]
    return P


def model_metrics(P, ei_test, valid_mask):
    """Per-state ranking metrics over valid test states from a (n_test, 9) proba matrix."""
    if not valid_mask.any(): return None
    P_v = P[valid_mask]; E_v = ei_test[valid_mask]
    true_best = E_v.argmax(axis=1)
    pred_rank = np.argsort(-P_v, axis=1)
    pred_top1 = pred_rank[:, 0]
    in_top3 = np.array([true_best[i] in pred_rank[i, :3] for i in range(len(true_best))])
    arange = np.arange(len(E_v))
    abs_reg = E_v[arange, true_best] - E_v[arange, pred_top1]
    rel_reg = abs_reg / E_v[arange, true_best] * 100.0
    return {
        "n_states": int(valid_mask.sum()),
        "top1": float((pred_top1 == true_best).mean()),
        "top3": float(in_top3.mean()),
        "abs_reg": float(abs_reg.mean()),
        "rel_reg": float(rel_reg.mean()),
    }


def random_metrics(ei_test, valid_mask):
    if not valid_mask.any(): return None
    E_v = ei_test[valid_mask]
    max_ei = E_v.max(axis=1); mean_ei = E_v.mean(axis=1)
    return {
        "n_states": int(valid_mask.sum()),
        "top1": 1/9, "top3": 3/9,
        "abs_reg": float((max_ei - mean_ei).mean()),
        "rel_reg": float(((max_ei - mean_ei) / max_ei * 100).mean()),
    }


def const_pair_metrics(ei_test, valid_mask, pair_idx, chosen_set=None):
    """Constant single-pair (chosen_set=None → only top-1) or constant top-3
    (chosen_set provided) predictor metrics on test."""
    if not valid_mask.any(): return None
    E_v = ei_test[valid_mask]
    arange = np.arange(len(E_v))
    true_best = E_v.argmax(axis=1)
    top1_hit = float((true_best == pair_idx).mean())
    if chosen_set is None:
        top3 = None
    else:
        top3 = float(np.array([tb in chosen_set for tb in true_best]).mean())
    abs_reg = (E_v[arange, true_best] - E_v[:, pair_idx]).mean()
    rel_reg = ((E_v[arange, true_best] - E_v[:, pair_idx]) / E_v[arange, true_best] * 100.0).mean()
    return {
        "n_states": int(valid_mask.sum()),
        "top1": top1_hit, "top3": top3,
        "abs_reg": float(abs_reg), "rel_reg": float(rel_reg),
    }


def main():
    print("Loading data...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")

    train_mask = df["size"].isin(TRAIN_SIZES).values
    test_mask  = (df["size"] == TEST_SIZE).values
    excl_mask  = (df["size"] == EXCLUDE_SIZE).values
    print(f"  train rows ({sorted(TRAIN_SIZES)}): {train_mask.sum()}")
    print(f"  test  rows ({TEST_SIZE}):              {test_mask.sum()}")
    print(f"  excluded   ({EXCLUDE_SIZE}):           {excl_mask.sum()}")
    assert (train_mask & test_mask).sum() == 0, "train/test overlap!"
    assert (train_mask & excl_mask).sum() == 0, "train/exclude overlap!"
    assert (test_mask  & excl_mask).sum() == 0, "test/exclude overlap!"

    ei = df[xpr.USED_EI].values
    ei_train = ei[train_mask]
    ei_test  = ei[test_mask]
    valid_test_all = ei_test.max(axis=1) > 0
    test_inst = df["instance"].values[test_mask]
    print(f"  valid test states (max_ei>0): {int(valid_test_all.sum())}/{len(valid_test_all)}")

    # --- Train-side per-pair mean ei_pct -> chosen pair(s) for baselines ---
    pair_means = ei_train.mean(axis=0)
    pair_order = np.argsort(-pair_means)
    chosen_top1 = int(pair_order[0])
    chosen_top3 = [int(i) for i in pair_order[:3]]
    chosen_top3_set = set(chosen_top3)
    print(f"\n  best_avg_pair  chosen = {xpr.USED_PAIRS[chosen_top1]} (idx {chosen_top1})")
    print(f"  best_avg_top3  chosen = "
          f"[{', '.join(xpr.USED_PAIRS[i] for i in chosen_top3)}]  (idx {chosen_top3})")

    rows = []

    # =========================================
    # Aggregate (all 3 5_min buckets pooled)
    # =========================================
    print(f"\nFitting 5 configs on {train_mask.sum()} train rows...")
    config_proba = {}    # tag -> (n_test, 9)
    for cfg in CONFIGS:
        P = fit_and_predict(df, train_mask, test_mask, cfg)
        config_proba[cfg["tag"]] = P
        m = model_metrics(P, ei_test, valid_test_all)
        rows.append({"scope": "aggregate", "model": cfg["tag"], "instance": "ALL", **m})
        print(f"  {cfg['tag']:22s} top1={m['top1']:.3f} top3={m['top3']:.3f} "
              f"abs={m['abs_reg']:.6f} rel={m['rel_reg']:.2f}%")

    rnd = random_metrics(ei_test, valid_test_all)
    rows.append({"scope": "aggregate", "model": "random", "instance": "ALL", **rnd})
    bap = const_pair_metrics(ei_test, valid_test_all, chosen_top1, chosen_set=None)
    rows.append({"scope": "aggregate", "model": "best_avg_pair", "instance": "ALL", **bap})
    bat3 = const_pair_metrics(ei_test, valid_test_all, chosen_top1, chosen_set=chosen_top3_set)
    rows.append({"scope": "aggregate", "model": "best_avg_top3", "instance": "ALL", **bat3})

    # =========================================
    # Per-bucket breakdown (3 5_min buckets)
    # =========================================
    for bucket in TEST_BUCKETS:
        bucket_mask = (test_inst == bucket)
        if not bucket_mask.any():
            print(f"  WARN: bucket {bucket} has 0 test rows"); continue
        ei_bucket = ei_test[bucket_mask]
        valid_bucket = ei_bucket.max(axis=1) > 0

        rnd_b = random_metrics(ei_bucket, valid_bucket)
        rows.append({"scope": "per_instance", "model": "random", "instance": bucket, **rnd_b})

        bap_b = const_pair_metrics(ei_bucket, valid_bucket, chosen_top1, chosen_set=None)
        rows.append({"scope": "per_instance", "model": "best_avg_pair", "instance": bucket, **bap_b})

        bat3_b = const_pair_metrics(ei_bucket, valid_bucket, chosen_top1, chosen_set=chosen_top3_set)
        rows.append({"scope": "per_instance", "model": "best_avg_top3", "instance": bucket, **bat3_b})

        for cfg in CONFIGS:
            P = config_proba[cfg["tag"]]
            P_b = P[bucket_mask]
            m_b = model_metrics(P_b, ei_bucket, valid_bucket)
            rows.append({"scope": "per_instance", "model": cfg["tag"], "instance": bucket, **m_b})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out_df)} rows)")

    # =========================================
    # Summary MD
    # =========================================
    write_summary(out_df, chosen_top1, chosen_top3, train_mask.sum(), test_mask.sum(),
                  excl_mask.sum(), int(valid_test_all.sum()))


def write_summary(df_out, chosen_top1, chosen_top3, n_tr, n_te, n_ex, n_valid):
    agg = df_out[df_out.scope == "aggregate"].copy()
    per = df_out[df_out.scope == "per_instance"].copy()

    # Order: baselines first, models after
    order = ["random", "best_avg_pair", "best_avg_top3"] + [c["tag"] for c in CONFIGS]
    agg = agg.set_index("model").loc[order].reset_index()

    # Determine winners for each metric (best across ALL rows)
    bap_top1 = float(agg[agg.model == "best_avg_pair"].iloc[0]["top1"])
    bat3_top3 = float(agg[agg.model == "best_avg_top3"].iloc[0]["top3"])
    bap_abs  = float(agg[agg.model == "best_avg_pair"].iloc[0]["abs_reg"])
    bap_rel  = float(agg[agg.model == "best_avg_pair"].iloc[0]["rel_reg"])

    L = []
    L.append("# Directional Transfer — train on {30s, 1min, 2min}, test on {5min}")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- **Train:** all states from `{{30_s, 01_min, 02_min}} × {{d03, d05, d10}}` = 9 buckets, **{n_tr} rows** total.")
    L.append(f"- **Test:** all states from `05_min × {{d03, d05, d10}}` = 3 buckets, **{n_te} rows** total ({n_valid} valid after `max(ei_pct) > 0` filter).")
    L.append(f"- **Excluded entirely:** all `15min` instances (**{n_ex} rows**).")
    L.append("- **Train/test/exclude masks verified disjoint.**")
    L.append("- **Pipeline:** same as `cross_pair_ranking_grid_2026-05-25/01_grid.py` — `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)` or `DecisionTreeClassifier(class_weight='balanced', random_state=42)`. K=20 means all 20 features (no Gini subset).")
    L.append("- **State exclusion:** `max(ei_pct over 9 USED) > 0`, same as the rest of the analysis.")
    L.append("- **Baseline selection (computed on TRAIN only):**")
    L.append(f"  - `best_avg_pair` picks **`{xpr.USED_PAIRS[chosen_top1]}`** (idx {chosen_top1}) by train-mean `ei_pct`.")
    L.append(f"  - `best_avg_top3` picks **[{', '.join('`' + xpr.USED_PAIRS[i] + '`' for i in chosen_top3)}]** (idx {chosen_top3}).")
    L.append("")

    L.append("## 1. Aggregate over all 3 `05_min` test buckets (state-weighted)")
    L.append("")
    L.append("| Model | n states | Top-1 | Top-3 | Abs regret | Rel regret (%) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        t3 = f"{r['top3']:.3f}" if pd.notna(r['top3']) else "n/a"
        L.append(f"| `{r['model']}` | {int(r['n_states'])} | "
                 f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.6f} | {r['rel_reg']:.2f} |")
    L.append("")

    # =========================================
    # Per-instance breakdown
    # =========================================
    L.append("## 2. Per-bucket breakdown")
    L.append("")
    for bucket in TEST_BUCKETS:
        L.append(f"### {bucket}")
        L.append("")
        L.append("| Model | n states | Top-1 | Top-3 | Abs regret | Rel regret (%) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        sub = per[per.instance == bucket].copy()
        sub = sub.set_index("model").loc[order].reset_index()
        for _, r in sub.iterrows():
            t3 = f"{r['top3']:.3f}" if pd.notna(r['top3']) else "n/a"
            L.append(f"| `{r['model']}` | {int(r['n_states'])} | "
                     f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.6f} | {r['rel_reg']:.2f} |")
        L.append("")

    # =========================================
    # Headline: any model beat the baselines?
    # =========================================
    L.append("## 3. Does any trained model beat the constant baselines?")
    L.append("")
    L.append("Checking each of the 5 trained models against the strongest baseline on each metric:")
    L.append("")
    L.append("| Model | beats `best_avg_pair` on top-1? | beats `best_avg_top3` on top-3? | beats `best_avg_pair` on abs regret? | beats `best_avg_pair` on rel regret? |")
    L.append("|---|:---:|:---:|:---:|:---:|")
    for cfg in CONFIGS:
        r = agg[agg.model == cfg["tag"]].iloc[0]
        m_top1 = float(r["top1"]); m_top3 = float(r["top3"])
        m_abs  = float(r["abs_reg"]); m_rel = float(r["rel_reg"])
        beat_top1 = "✓" if m_top1 > bap_top1  else f"✗  ({m_top1:.3f} vs {bap_top1:.3f})"
        beat_top3 = "✓" if m_top3 > bat3_top3 else f"✗  ({m_top3:.3f} vs {bat3_top3:.3f})"
        beat_abs  = "✓" if m_abs  < bap_abs   else f"✗  ({m_abs:.6f} vs {bap_abs:.6f})"
        beat_rel  = "✓" if m_rel  < bap_rel   else f"✗  ({m_rel:.2f} vs {bap_rel:.2f})"
        L.append(f"| `{cfg['tag']}` | {beat_top1} | {beat_top3} | {beat_abs} | {beat_rel} |")
    L.append("")

    # Count winners
    winners_any = 0
    for cfg in CONFIGS:
        r = agg[agg.model == cfg["tag"]].iloc[0]
        if (float(r["top1"]) > bap_top1 or float(r["top3"]) > bat3_top3 or
            float(r["abs_reg"]) < bap_abs or float(r["rel_reg"]) < bap_rel):
            winners_any += 1
    L.append(f"**Models that beat the corresponding best baseline on at least one metric: {winners_any}/{len(CONFIGS)}.**")
    L.append("")
    if winners_any == 0:
        L.append("Every trained model loses to the constant `best_avg_pair` / `best_avg_top3` baselines on every measured metric. The trained models have learned features that don't generalise to a problem-size class they never saw at train time, while the constant 'always pick the empirically-best pair from training' baseline transfers trivially.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append(f"- Per-row CSV: `{OUT_CSV.relative_to(REPO)}`")
    L.append("")

    OUT_MD.write_text("\n".join(L))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
