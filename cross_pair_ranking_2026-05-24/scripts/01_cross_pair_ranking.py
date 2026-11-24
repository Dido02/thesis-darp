#!/usr/bin/env python3
"""Cross-pair ranking — bench-proxy for the deployed softmax sampler.

For each (state, model):
  - Get the 9 per-pair predict_proba outputs.
  - Get ground-truth ei_pct for the 9 pairs from labels_t0.3_with_ei.parquet.
  - Skip states where max(ei_pct over 9 pairs) == 0 (no improvement possible).
  - Compute top-1 / top-3 hit rate and abs/rel regret vs the true-best pair.

Models (4):
  - forest_d_none_K20   (deployed: depth=None, all 20 features)
  - forest_d8_K20       (depth=8, all 20)
  - forest_d8_K16       (depth=8, top-16 by aggregate Gini from deployed)
  - tree_d4_K4          (single tree, depth=4, top-4 features)

Protocols (3):
  - single_75_25  — exact reproduction of the deployed model's per-log split
                    (random_state=42, train/test logs from deployed
                     training_report.json)
  - 5fold         — stratified by (size, delay) round-robin
  - LOIO          — leave one (size, delay) bucket out at a time

Baselines:
  - random        — top1 = 1/9, top3 = 3/9, expected regret = ei[best] - mean(ei[9])
  - best-avg-pair — the single pair with highest mean ei_pct in the training set;
                    its regret on the test set is the "non-learning floor".

Outputs (tables/ + a markdown report):
  - aggregate_per_protocol.csv  : per (protocol, model) top1/top3/abs/rel regret
  - per_instance_loio.csv       : per LOIO fold (size_delay) breakdown
  - per_instance_5fold.csv      : per (size, delay) bucket under 5-fold
  - side_by_side.csv            : per (model, instance) 5fold-top1 vs LOIO-top1 vs Δ
  - CROSS_PAIR_RANKING_REPORT.md
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
CACHE = REPO / "ml_train" / "cache_v3_full"
DEPLOYED_DIR = REPO / "ml_train" / "models_v3_full_20f_t07"
OUT = REPO / "cross_pair_ranking_2026-05-24"
TBL = OUT / "tables"
TBL.mkdir(parents=True, exist_ok=True)

USED_PAIRS_TUP = [(d, r) for d in (0, 1, 2) for r in (0, 1, 2)]
USED_PAIRS = [f"pair_{d}_{r}" for d, r in USED_PAIRS_TUP]      # binary label cols
USED_EI    = [f"ei_pct_{d}_{r}" for d, r in USED_PAIRS_TUP]    # ei_pct cols

FEATURE_NAMES_20 = [
    "num_active_requests", "radius_avg_tt", "frac_committed", "avg_route_load",
    "temperature_normalized", "ride_time_utilization_avg",
    "ride_time_utilization_max", "near_violation_count", "route_slack_avg",
    "route_slack_std", "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "neighborhood_size_frac", "cost_vs_initial_frac",
    "cost_vs_incumbent_frac", "iteration_norm", "unassigned_count",
    "committed_travel_time_frac", "horizon_elapsed_frac",
]
RE_LOG = re.compile(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+")


# ============================================================
# Feature subsets — exact replica of new_models/00_train_for_bench.py
# ============================================================
def aggregated_top_k_indices(K: int) -> list[int]:
    if K >= 20: return list(range(20))
    imps = np.zeros(20)
    for d, r in USED_PAIRS_TUP:
        clf = joblib.load(DEPLOYED_DIR / f"rf_pair_{d}_{r}.joblib")
        imps += clf.feature_importances_
    imps /= 9
    return sorted(np.argsort(-imps)[:K].tolist())


# ============================================================
# Model factory
# ============================================================
def make_clf(cfg):
    if cfg["type"] == "forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=cfg["max_depth"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return DecisionTreeClassifier(
        max_depth=cfg["max_depth"], class_weight="balanced", random_state=42,
    )


MODEL_CONFIGS = [
    {"tag": "forest_d_none_K20", "type": "forest", "max_depth": None, "K": 20},
    {"tag": "forest_d8_K20",     "type": "forest", "max_depth": 8,    "K": 20},
    {"tag": "forest_d8_K16",     "type": "forest", "max_depth": 8,    "K": 16},
    {"tag": "tree_d4_K4",        "type": "tree",   "max_depth": 4,    "K": 4 },
]


# ============================================================
# Data loading + (size, delay) parsing
# ============================================================
def load_data():
    X_raw = pd.read_parquet(CACHE / "features.parquet")
    Y_bin = pd.read_parquet(CACHE / "labels_t0.7.parquet")
    Y_ei  = pd.read_parquet(CACHE / "labels_t0.3_with_ei.parquet")
    # ei_pct is threshold-independent; t03 file just happens to have it.
    df = X_raw.merge(Y_bin, on=["log_file", "iter"], how="inner")
    ei = Y_ei[["log_file", "iter"] + USED_EI]
    df = df.merge(ei, on=["log_file", "iter"], how="inner")
    sizes  = df["log_file"].map(lambda n: (RE_LOG.match(n).group("size")  if RE_LOG.match(n) else None))
    delays = df["log_file"].map(lambda n: (RE_LOG.match(n).group("delay") if RE_LOG.match(n) else None))
    df["size"], df["delay"] = sizes.values, delays.values
    df["instance"] = df["size"] + "_d" + df["delay"]
    return df


# ============================================================
# Protocol fold generation
# ============================================================
def deployed_split(df):
    report = json.loads((DEPLOYED_DIR / "training_report.json").read_text())
    tr = set(report["split"]["train_logs"]); te = set(report["split"]["test_logs"])
    train = df["log_file"].isin(tr).values
    test  = df["log_file"].isin(te).values
    yield {"name": "deployed_75_25", "train_mask": train, "test_mask": test,
           "test_instances": df.loc[test, "instance"].values}


def fivefold_splits(df, n_folds=5, seed=42):
    """Stratified by (size, delay), round-robin assignment within each stratum."""
    rng = np.random.RandomState(seed)
    log_to_fold = {}
    for inst, sub in df.groupby("instance"):
        logs = sorted(sub["log_file"].unique())
        rng.shuffle(logs)
        for i, lf in enumerate(logs):
            log_to_fold[lf] = i % n_folds
    for fold in range(n_folds):
        test_logs  = {lf for lf, f in log_to_fold.items() if f == fold}
        train_logs = {lf for lf, f in log_to_fold.items() if f != fold}
        test_mask  = df["log_file"].isin(test_logs).values
        train_mask = df["log_file"].isin(train_logs).values
        yield {"name": f"5fold_f{fold}", "train_mask": train_mask, "test_mask": test_mask,
               "test_instances": df.loc[test_mask, "instance"].values}


def loio_splits(df):
    """One fold per (size, delay) instance bucket."""
    for inst in sorted(df["instance"].unique()):
        test_mask  = (df["instance"] == inst).values
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        yield {"name": f"loio_{inst}", "train_mask": train_mask, "test_mask": test_mask,
               "test_instances": df.loc[test_mask, "instance"].values}


# ============================================================
# Fit 9 per-pair models + predict_proba on test
# Returns a (n_test, 9) matrix of P(y=1) per pair.
# ============================================================
def fit_and_predict(df, train_mask, test_mask, cfg):
    feat_idx = aggregated_top_k_indices(cfg["K"])
    X = df[FEATURE_NAMES_20].values
    Xtr, Xte = X[train_mask][:, feat_idx], X[test_mask][:, feat_idx]
    P = np.zeros((test_mask.sum(), 9), dtype=float)
    train_prevalence = np.zeros(9)
    for j, p in enumerate(USED_PAIRS):
        y = df[p].values[train_mask].astype(int)
        train_prevalence[j] = y.mean()
        if y.sum() == 0 or y.sum() == len(y):
            P[:, j] = train_prevalence[j]
            continue
        clf = make_clf(cfg)
        clf.fit(Xtr, y)
        P[:, j] = clf.predict_proba(Xte)[:, 1]
    return P, train_prevalence


# ============================================================
# Metrics from (proba, ei) per state, with mask for states-with-improvement
# ============================================================
def compute_state_metrics(proba_mat, ei_mat):
    """Returns per-state arrays for valid (max-ei > 0) states only."""
    max_ei = ei_mat.max(axis=1)
    valid = max_ei > 0
    if not valid.any():
        return None
    P = proba_mat[valid]; E = ei_mat[valid]
    true_best = E.argmax(axis=1)
    pred_rank = np.argsort(-P, axis=1)              # cols sorted by descending proba
    pred_top1 = pred_rank[:, 0]
    in_top3   = np.array([true_best[i] in pred_rank[i, :3]
                          for i in range(len(true_best))])
    abs_reg   = E[np.arange(len(E)), true_best] - E[np.arange(len(E)), pred_top1]
    rel_reg   = abs_reg / E[np.arange(len(E)), true_best] * 100
    return {
        "n": int(valid.sum()),
        "top1_hit": (pred_top1 == true_best).astype(float),
        "top3_hit": in_top3.astype(float),
        "abs_reg": abs_reg,
        "rel_reg": rel_reg,
        "valid_mask": valid,
    }


def baseline_random(ei_mat):
    max_ei = ei_mat.max(axis=1)
    valid = max_ei > 0
    E = ei_mat[valid]
    if len(E) == 0:
        return {"n": 0, "top1": np.nan, "top3": np.nan, "abs_reg": np.nan, "rel_reg": np.nan}
    expected_abs_reg = (E.max(axis=1) - E.mean(axis=1)).mean()
    expected_rel_reg = ((E.max(axis=1) - E.mean(axis=1)) / E.max(axis=1) * 100).mean()
    return {"n": int(valid.sum()), "top1": 1/9, "top3": 3/9,
            "abs_reg": float(expected_abs_reg), "rel_reg": float(expected_rel_reg)}


def baseline_best_avg_pair(ei_train, ei_test):
    """Pick the pair with highest mean ei_pct on TRAIN; evaluate on TEST."""
    if ei_train.size == 0 or ei_test.size == 0:
        return {"n": 0, "top1": np.nan, "top3": np.nan, "abs_reg": np.nan, "rel_reg": np.nan, "pair_idx": -1}
    pair_idx = int(ei_train.mean(axis=0).argmax())
    max_ei = ei_test.max(axis=1)
    valid = max_ei > 0
    E = ei_test[valid]
    if len(E) == 0:
        return {"n": 0, "top1": np.nan, "top3": np.nan, "abs_reg": np.nan, "rel_reg": np.nan, "pair_idx": pair_idx}
    true_best = E.argmax(axis=1)
    top1_hit = (pair_idx == true_best).astype(float).mean()
    # top3 is degenerate (always same single pair) — report N/A
    abs_reg = (E[np.arange(len(E)), true_best] - E[:, pair_idx]).mean()
    rel_reg = ((E[np.arange(len(E)), true_best] - E[:, pair_idx]) /
               E[np.arange(len(E)), true_best] * 100).mean()
    return {"n": int(valid.sum()), "top1": float(top1_hit), "top3": np.nan,
            "abs_reg": float(abs_reg), "rel_reg": float(rel_reg), "pair_idx": pair_idx}


# ============================================================
# Per-protocol evaluation
# ============================================================
def evaluate_protocol(df, fold_iter_fn, protocol_label):
    """Iterate folds; for each, fit each model + compute per-state metrics.
    Returns a long DataFrame: one row per (fold, model, state)."""
    rows = []
    folds = list(fold_iter_fn(df))
    print(f"  [{protocol_label}] {len(folds)} fold(s)")
    for fi, fold in enumerate(folds):
        tr, te = fold["train_mask"], fold["test_mask"]
        ei_train = df.loc[tr, USED_EI].values
        ei_test  = df.loc[te, USED_EI].values
        test_inst = fold["test_instances"]

        # Baselines (don't depend on model)
        rnd = baseline_random(ei_test)
        bap = baseline_best_avg_pair(ei_train, ei_test)

        for cfg in MODEL_CONFIGS:
            print(f"    fold {fi+1}/{len(folds)} ({fold['name']}) — {cfg['tag']}", flush=True)
            P, prev = fit_and_predict(df, tr, te, cfg)
            metrics = compute_state_metrics(P, ei_test)
            if metrics is None:
                continue
            valid = metrics["valid_mask"]
            valid_inst = test_inst[valid]
            for k in range(metrics["n"]):
                rows.append({
                    "protocol": protocol_label, "fold": fold["name"],
                    "model": cfg["tag"], "instance": valid_inst[k],
                    "top1_hit": metrics["top1_hit"][k],
                    "top3_hit": metrics["top3_hit"][k],
                    "abs_reg":  metrics["abs_reg"][k],
                    "rel_reg":  metrics["rel_reg"][k],
                })

        # Baselines as long rows (one per fold; model="random" / "best_avg_pair")
        for k_lbl, k_val in [("random", rnd), ("best_avg_pair", bap)]:
            rows.append({
                "protocol": protocol_label, "fold": fold["name"],
                "model": k_lbl, "instance": "_AGG_",
                "top1_hit": k_val["top1"], "top3_hit": k_val["top3"],
                "abs_reg":  k_val["abs_reg"], "rel_reg": k_val["rel_reg"],
            })
    return pd.DataFrame(rows)


# ============================================================
# Aggregation + tables
# ============================================================
def agg_per_protocol_model(df_state):
    """Mean metrics across all valid states per (protocol, model)."""
    out = []
    for (proto, model), sub in df_state.groupby(["protocol", "model"]):
        if model in ("random", "best_avg_pair"):
            out.append({
                "protocol": proto, "model": model,
                "n_states": int(sub.notna().sum()["top1_hit"]),
                "top1": float(sub["top1_hit"].mean()) if pd.notna(sub["top1_hit"]).any() else np.nan,
                "top3": float(sub["top3_hit"].mean()) if pd.notna(sub["top3_hit"]).any() else np.nan,
                "abs_reg": float(sub["abs_reg"].mean()),
                "rel_reg": float(sub["rel_reg"].mean()),
            })
        else:
            sub = sub[sub.instance != "_AGG_"]
            out.append({
                "protocol": proto, "model": model, "n_states": int(len(sub)),
                "top1": float(sub["top1_hit"].mean()),
                "top3": float(sub["top3_hit"].mean()),
                "abs_reg": float(sub["abs_reg"].mean()),
                "rel_reg": float(sub["rel_reg"].mean()),
            })
    return pd.DataFrame(out)


def agg_per_instance(df_state, protocol):
    """Per-instance aggregation under one protocol."""
    sub = df_state[(df_state.protocol == protocol) & (df_state.instance != "_AGG_")]
    out = []
    for (model, instance), grp in sub.groupby(["model", "instance"]):
        out.append({
            "protocol": protocol, "model": model, "instance": instance,
            "n_states": len(grp),
            "top1": float(grp["top1_hit"].mean()),
            "top3": float(grp["top3_hit"].mean()),
            "abs_reg": float(grp["abs_reg"].mean()),
            "rel_reg": float(grp["rel_reg"].mean()),
        })
    return pd.DataFrame(out)


def side_by_side(per_inst_5fold, per_inst_loio):
    a = per_inst_5fold[["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    a.columns = ["model", "instance", "fold5_top1", "fold5_top3", "fold5_abs", "fold5_rel"]
    b = per_inst_loio[["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    b.columns = ["model", "instance", "loio_top1", "loio_top3", "loio_abs", "loio_rel"]
    cmp = a.merge(b, on=["model", "instance"], how="outer")
    cmp["delta_top1"] = cmp["loio_top1"] - cmp["fold5_top1"]
    cmp["delta_top3"] = cmp["loio_top3"] - cmp["fold5_top3"]
    cmp["delta_abs"]  = cmp["loio_abs"]  - cmp["fold5_abs"]
    return cmp


# ============================================================
# Markdown writer
# ============================================================
def write_md(agg, per_inst_5f, per_inst_lo, side, out_path):
    L = []
    L.append("# Cross-pair Ranking — bench-proxy for the deployed softmax sampler")
    L.append("")
    L.append("**Setup:** For each (state, model) we compute 9 per-pair `predict_proba`, join with ground-truth `ei_pct_{d}_{r}` over the 9 USED pairs, skip states with `max(ei_pct) = 0`, and measure top-1, top-3, mean absolute regret, and mean relative regret. Models trained per (destroy, repair) pair (binary t07 label).")
    L.append("")
    L.append("Random baseline: top-1 = 1/9 = 11.1%, top-3 = 3/9 = 33.3%; abs regret = `mean(max(ei) − mean(ei))`. `best_avg_pair` baseline: pick the single pair with highest train-set mean `ei_pct`, evaluate as constant predictor on test.")
    L.append("")

    L.append("## 1. Aggregate per protocol per model")
    L.append("")
    L.append("| protocol | model | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    order = ["deployed_75_25", "5fold", "LOIO"]
    model_order = ["random", "best_avg_pair",
                   "forest_d_none_K20", "forest_d8_K20", "forest_d8_K16", "tree_d4_K4"]
    for p in order:
        for m in model_order:
            row = agg[(agg.protocol == p) & (agg.model == m)]
            if len(row) == 0: continue
            r = row.iloc[0]
            t1 = f"{r['top1']:.3f}" if pd.notna(r["top1"]) else "n/a"
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| {p} | `{m}` | {r['n_states']} | {t1} | {t3} | "
                     f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")
    L.append("**How to read:** higher top-K and lower regret = better. Random ≈ 0.111 / 0.333 and a fixed positive abs regret. The deployed sampler effectively samples by softmax-T=0.10 from the model's proba vector — its bench behavior should sit between top-1 and top-3 in expectation.")
    L.append("")

    L.append("## 2. Per-instance breakdown — LOIO protocol")
    L.append("")
    L.append("Each row = (model, held-out instance bucket).")
    L.append("")
    L.append("| model | instance | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in [c["tag"] for c in MODEL_CONFIGS]:
        sub = per_inst_lo[per_inst_lo.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            L.append(f"| `{m}` | {r['instance']} | {r['n_states']} | {r['top1']:.3f} | "
                     f"{r['top3']:.3f} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 3. Per-instance breakdown — 5-fold protocol")
    L.append("")
    L.append("| model | instance | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in [c["tag"] for c in MODEL_CONFIGS]:
        sub = per_inst_5f[per_inst_5f.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            L.append(f"| `{m}` | {r['instance']} | {r['n_states']} | {r['top1']:.3f} | "
                     f"{r['top3']:.3f} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 4. Side-by-side — 5-fold vs LOIO per (model, instance)")
    L.append("")
    L.append("Positive `Δ top-1` = LOIO retained or improved skill; negative = transfer collapse on that instance.")
    L.append("")
    L.append("| model | instance | 5fold top-1 | LOIO top-1 | Δ top-1 | 5fold abs reg | LOIO abs reg | Δ abs reg |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in [c["tag"] for c in MODEL_CONFIGS]:
        sub = side[side.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            def f(v): return "n/a" if pd.isna(v) else f"{v:.3f}"
            def fa(v): return "n/a" if pd.isna(v) else f"{v:.4f}"
            L.append(f"| `{m}` | {r['instance']} | {f(r['fold5_top1'])} | {f(r['loio_top1'])} | "
                     f"{f(r['delta_top1'])} | {fa(r['fold5_abs'])} | {fa(r['loio_abs'])} | "
                     f"{fa(r['delta_abs'])} |")
    L.append("")

    L.append("## 5. Reading guide")
    L.append("")
    L.append("- **§1 deployed_75_25 row of `forest_d_none_K20`** is the closest analogue to the deployed selector's training-time evaluation. Compare its top-1/top-3 to the random baseline to see the headline ranking lift.")
    L.append("- **§1 LOIO row** is the transfer-test lift — by how much does ranking quality survive when no log from the held-out (size, delay) class was in training.")
    L.append("- **§4 `Δ top-1` column** shows the per-instance LOIO − 5-fold gap. Negative on every row = uniform transfer collapse; mixed signs = some instances retain skill, others don't.")
    out_path.write_text("\n".join(L))


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    df = load_data()
    print(f"  joined rows: {len(df)}")
    print(f"  instances:   {sorted(df.instance.unique())}")
    print(f"  rows per instance:")
    print(df.groupby("instance").size().to_string())

    print("\n--- Protocol: single 75/25 split ---")
    state_75 = evaluate_protocol(df, deployed_split, "deployed_75_25")
    print("\n--- Protocol: 5-fold CV ---")
    state_5f = evaluate_protocol(df, lambda d: fivefold_splits(d), "5fold")
    print("\n--- Protocol: LOIO ---")
    state_lo = evaluate_protocol(df, loio_splits, "LOIO")

    state_all = pd.concat([state_75, state_5f, state_lo], ignore_index=True)
    state_all.to_csv(TBL / "state_level_predictions.csv", index=False)

    print("\nAggregating...")
    agg = agg_per_protocol_model(state_all)
    agg.to_csv(TBL / "aggregate_per_protocol.csv", index=False)

    per_inst_lo = agg_per_instance(state_all, "LOIO")
    per_inst_lo.to_csv(TBL / "per_instance_loio.csv", index=False)

    per_inst_5f = agg_per_instance(state_all, "5fold")
    per_inst_5f.to_csv(TBL / "per_instance_5fold.csv", index=False)

    side = side_by_side(per_inst_5f, per_inst_lo)
    side.to_csv(TBL / "side_by_side.csv", index=False)

    print("\nAggregate table:")
    print(agg.to_string(index=False))

    write_md(agg, per_inst_5f, per_inst_lo, side, OUT / "CROSS_PAIR_RANKING_REPORT.md")
    print(f"\nWrote {OUT / 'CROSS_PAIR_RANKING_REPORT.md'}")
    print(f"     + 4 CSVs in {TBL}/")


if __name__ == "__main__":
    main()
