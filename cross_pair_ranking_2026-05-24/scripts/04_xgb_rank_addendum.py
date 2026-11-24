#!/usr/bin/env python3
"""XGBoost ranker addendum (rank:ndcg).

Adds one more configuration to the cross-pair ranking analysis:
xgboost.XGBRanker trained with rank:ndcg, treating each state as a query
group of 9 candidate (destroy, repair) pairs.

Model:
  XGBRanker(objective='rank:ndcg', eval_metric='ndcg@1',
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, tree_method='hist')

Per-state feature construction:
  For each valid state (max(ei_pct over 9 USED) > 0) we emit 9 rows, one
  per USED pair. Each row's feature vector is the state's 20 features
  (replicated identically across the 9 pair rows) concatenated with a
  9-dim one-hot indicator of which pair this row represents -> 29 dims.

Relevance label (per row):
  relevance = round(ei_pct * 10000) clipped to [0, 31].
  ei_pct is bounded ~0.04 in this dataset; *10000 maps that to ~400,
  so the clip at 31 is in practice rarely binding (~p99 of ei_pct is
  ~0.006 -> relevance 60, but most states have a sharper bound). This
  is a deliberate compression onto a small integer grade space which
  rank:ndcg's gain function expects.

Group field:
  All states have group size = 9 (one query per state, 9 candidates).

State filter: train and test both restricted to max(ei_pct) > 0.

Append-only: adds new rows for model="xgb_rank" to:
  state_level_predictions.csv
  aggregate_per_protocol.csv
  per_instance_loio.csv
  per_instance_5fold.csv
  side_by_side.csv

Existing rows for other models are NOT modified — the script reads the
existing CSV, drops any prior xgb_rank rows for idempotency, computes the
new xgb_rank rows, and concats. Forest/tree/logistic/baseline numbers are
unchanged.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError:
    sys.exit("xgboost not installed. Run: "
             "/Users/didodimitrov/Documents/thesis-darp/.venv/bin/pip install xgboost")

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"
sys.path.insert(0, str(SCRIPTS))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
xpr = _load("xpr", SCRIPTS / "01_cross_pair_ranking.py")

OUT = REPO / "cross_pair_ranking_2026-05-24"
TBL = OUT / "tables"
MODEL_TAG = "xgb_rank"

REL_SCALE = 10000      # multiplier on ei_pct
REL_CLIP  = 31         # max integer relevance grade


# ============================================================
# Build the (n_valid * 9, 29) feature matrix + relevance + groups
# ============================================================
def build_query_data(X_state, ei_state):
    """X_state: (n_states, 20).  ei_state: (n_states, 9).
    Returns Xq, y, group_sizes for valid states only."""
    valid = ei_state.max(axis=1) > 0
    Xs = X_state[valid]; Es = ei_state[valid]
    n_v = len(Xs)
    if n_v == 0:
        return None, None, None, None
    rep_features = np.repeat(Xs, 9, axis=0)                                # (n_v*9, 20)
    one_hot = np.tile(np.eye(9, dtype=float), (n_v, 1))                    # (n_v*9, 9)
    Xq = np.concatenate([rep_features, one_hot], axis=1)                   # (n_v*9, 29)
    y = np.clip(np.round(Es.flatten() * REL_SCALE), 0, REL_CLIP).astype(int)  # (n_v*9,)
    group_sizes = np.full(n_v, 9, dtype=int)
    return Xq, y, group_sizes, valid


# ============================================================
# Fit + predict for one fold
# ============================================================
def fit_and_predict_xgb(df, train_mask, test_mask):
    X = df[xpr.FEATURE_NAMES_20].values
    E = df[xpr.USED_EI].values
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]

    Xq_tr, y_tr, g_tr, _ = build_query_data(X[tr_idx], E[tr_idx])
    Xq_te, _, _, valid_te = build_query_data(X[te_idx], E[te_idx])
    if Xq_tr is None or Xq_te is None:
        return None

    ranker = XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@1",
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, tree_method="hist",
        verbosity=0,
    )
    ranker.fit(Xq_tr, y_tr, group=g_tr)

    scores = ranker.predict(Xq_te)         # (n_v*9,)
    n_v = len(scores) // 9
    score_mat = scores.reshape(n_v, 9)     # (n_v, 9): rows = states, cols = pair index
    te_idx_v = te_idx[valid_te]
    return score_mat, te_idx_v


def evaluate_one_fold(df, fold, protocol_label):
    out = fit_and_predict_xgb(df, fold["train_mask"], fold["test_mask"])
    if out is None: return []
    score_mat, te_idx_v = out
    E_valid = df[xpr.USED_EI].values[te_idx_v]
    inst = df["instance"].values[te_idx_v]
    true_best = E_valid.argmax(axis=1)
    pred_rank = np.argsort(-score_mat, axis=1)
    pred_top1 = pred_rank[:, 0]
    in_top3 = np.array([true_best[i] in pred_rank[i, :3] for i in range(len(true_best))])
    arange = np.arange(len(E_valid))
    abs_reg = E_valid[arange, true_best] - E_valid[arange, pred_top1]
    rel_reg = abs_reg / E_valid[arange, true_best] * 100.0
    return [{
        "protocol": protocol_label, "fold": fold["name"],
        "model": MODEL_TAG, "instance": inst[k],
        "top1_hit": float(pred_top1[k] == true_best[k]),
        "top3_hit": float(in_top3[k]),
        "abs_reg": float(abs_reg[k]),
        "rel_reg": float(rel_reg[k]),
    } for k in range(len(E_valid))]


# ============================================================
# Append-only aggregation helpers (only touch xgb_rank rows)
# ============================================================
def agg_one_model(state_df, model_tag):
    sub = state_df[state_df.model == model_tag]
    out = []
    for proto, grp in sub.groupby("protocol"):
        t3 = float(grp["top3_hit"].mean()) if grp["top3_hit"].notna().any() else np.nan
        out.append({"protocol": proto, "model": model_tag, "n_states": int(len(grp)),
                    "top1": float(grp["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(grp["abs_reg"].mean()),
                    "rel_reg": float(grp["rel_reg"].mean())})
    return pd.DataFrame(out)

def per_instance_one_model(state_df, model_tag, protocol):
    sub = state_df[(state_df.model == model_tag) & (state_df.protocol == protocol)]
    out = []
    for inst, g in sub.groupby("instance"):
        t3 = float(g["top3_hit"].mean()) if g["top3_hit"].notna().any() else np.nan
        out.append({"protocol": protocol, "model": model_tag, "instance": inst,
                    "n_states": len(g),
                    "top1": float(g["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(g["abs_reg"].mean()),
                    "rel_reg": float(g["rel_reg"].mean())})
    return pd.DataFrame(out)

def side_by_side_one_model(p_5f, p_lo, model_tag):
    a = p_5f[p_5f.model == model_tag][["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    a.columns = ["model", "instance", "fold5_top1", "fold5_top3", "fold5_abs", "fold5_rel"]
    b = p_lo[p_lo.model == model_tag][["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    b.columns = ["model", "instance", "loio_top1", "loio_top3", "loio_abs", "loio_rel"]
    cmp = a.merge(b, on=["model", "instance"], how="outer")
    cmp["delta_top1"] = cmp["loio_top1"] - cmp["fold5_top1"]
    cmp["delta_top3"] = cmp["loio_top3"] - cmp["fold5_top3"]
    cmp["delta_abs"]  = cmp["loio_abs"]  - cmp["fold5_abs"]
    return cmp


def append_only(csv_path: Path, new_rows: pd.DataFrame, model_tag: str):
    """Read existing csv, drop any rows for model_tag (idempotent re-runs),
    concat new rows, write back. Order: existing rows preserved, new rows
    appended at the bottom."""
    existing = pd.read_csv(csv_path)
    kept = existing[existing["model"] != model_tag]
    out = pd.concat([kept, new_rows], ignore_index=True)
    out.to_csv(csv_path, index=False)
    return len(kept), len(new_rows)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + reproducing folds (no retraining of existing models)...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")

    # Drop existing xgb_rank rows from state_level_predictions.csv (idempotent)
    state_csv = TBL / "state_level_predictions.csv"
    existing_state = pd.read_csv(state_csv)
    print(f"  loaded state_level_predictions.csv: {len(existing_state)} rows")
    if MODEL_TAG in existing_state["model"].unique():
        print(f"  dropping previous {MODEL_TAG} rows for clean re-fit")
        existing_state = existing_state[existing_state["model"] != MODEL_TAG]

    # Generate xgb_rank rows for all 3 protocols
    new_rows = []
    print("\n[deployed_75_25] 1 fold")
    t0 = time.time()
    for fold in xpr.deployed_split(df):
        rs = evaluate_one_fold(df, fold, "deployed_75_25")
        print(f"  {fold['name']}: {len(rs)} valid test states  ({time.time()-t0:.1f}s)")
        new_rows.extend(rs)

    print("\n[5fold] 5 folds")
    for fold in xpr.fivefold_splits(df):
        rs = evaluate_one_fold(df, fold, "5fold")
        print(f"  {fold['name']}: {len(rs)} valid test states  ({time.time()-t0:.1f}s)")
        new_rows.extend(rs)

    print("\n[LOIO] 14 folds")
    for fold in xpr.loio_splits(df):
        rs = evaluate_one_fold(df, fold, "LOIO")
        print(f"  {fold['name']}: {len(rs)} valid test states  ({time.time()-t0:.1f}s)")
        new_rows.extend(rs)

    print(f"\nGenerated {len(new_rows)} new {MODEL_TAG} per-state rows  "
          f"(total elapsed {time.time()-t0:.1f}s)")
    new_state_df = pd.DataFrame(new_rows)
    combined_state = pd.concat([existing_state, new_state_df], ignore_index=True)
    combined_state.to_csv(state_csv, index=False)
    print(f"  state_level_predictions.csv: {len(existing_state)} kept + "
          f"{len(new_state_df)} new = {len(combined_state)} rows")

    # ----- Aggregate only the xgb_rank rows, append to the four CSVs -----
    agg_new = agg_one_model(combined_state, MODEL_TAG)
    p_lo_new = per_instance_one_model(combined_state, MODEL_TAG, "LOIO")
    p_5f_new = per_instance_one_model(combined_state, MODEL_TAG, "5fold")
    side_new = side_by_side_one_model(p_5f_new, p_lo_new, MODEL_TAG)

    a_keep, a_new = append_only(TBL / "aggregate_per_protocol.csv", agg_new, MODEL_TAG)
    print(f"\naggregate_per_protocol.csv: kept {a_keep}, appended {a_new}")
    l_keep, l_new = append_only(TBL / "per_instance_loio.csv", p_lo_new, MODEL_TAG)
    print(f"per_instance_loio.csv: kept {l_keep}, appended {l_new}")
    f_keep, f_new = append_only(TBL / "per_instance_5fold.csv", p_5f_new, MODEL_TAG)
    print(f"per_instance_5fold.csv: kept {f_keep}, appended {f_new}")
    s_keep, s_new = append_only(TBL / "side_by_side.csv", side_new, MODEL_TAG)
    print(f"side_by_side.csv: kept {s_keep}, appended {s_new}")

    # ----- Comparison vs best_avg_pair and forest_dNone_K8 -----
    full_agg = pd.read_csv(TBL / "aggregate_per_protocol.csv")
    print("\nXGB_rank aggregate (state-weighted):")
    print(agg_new.sort_values("protocol").to_string(index=False))

    # ----- Write addendum MD -----
    L = []
    L.append("# XGBoost Ranker Addendum — rank:ndcg over (state, pair) query groups")
    L.append("")
    L.append("**Date:** 2026-05-24")
    L.append("")
    L.append("## Model")
    L.append("")
    L.append("- `xgboost.XGBRanker(objective='rank:ndcg', eval_metric='ndcg@1', n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist')`")
    L.append("- Each state is one query group containing all 9 USED (destroy, repair) candidate pairs.")
    L.append("- Feature vector per (state, pair) row: the state's 20 features (replicated identically across the 9 rows) concatenated with a 9-dim one-hot indicator of which pair the row represents. Input dim = 29.")
    L.append(f"- Relevance label: `round(ei_pct * {REL_SCALE})` clipped to `[0, {REL_CLIP}]`. ei_pct's empirical p99 in this dataset is ~0.006, so most relevance grades fall in [0, 6]; the clip at {REL_CLIP} is rarely binding.")
    L.append("- Group field per state = 9 (each state is one query of 9 candidates).")
    L.append("- State filter: train and test both restricted to `max(ei_pct over 9 USED) > 0`. Same exclusion as every other configuration in this folder.")
    L.append("")
    L.append("Why this configuration: the existing forests/trees/logistic are pointwise classifiers (binary t07 or multinomial argmax-ei_pct). `XGBRanker` with `rank:ndcg` is a learning-to-rank model that directly optimises the per-query ranking quality, which is the bench-relevant objective (the deployed selector cares about the ordering of the 9 pairs at each state, not about absolute probabilities).")
    L.append("")

    L.append("## Aggregate results (state-weighted)")
    L.append("")
    L.append("| protocol | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for proto in ["deployed_75_25", "5fold", "LOIO"]:
        r = agg_new[agg_new.protocol == proto]
        if len(r) == 0: continue
        r = r.iloc[0]
        L.append(f"| {proto} | {r['n_states']} | {r['top1']:.3f} | {r['top3']:.3f} | "
                 f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## Comparison vs `best_avg_pair` and `forest_dNone_K8`")
    L.append("")
    L.append("`forest_dNone_K8` is the best forest on LOIO top-1 from the full-grid analysis (`cross_pair_ranking_grid_2026-05-25/`). `best_avg_pair` is the constant-predictor baseline (always pair_1_1 = worst+regret2). `random` included for reference.")
    L.append("")
    L.append("| protocol | model | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|")
    for proto in ["deployed_75_25", "5fold", "LOIO"]:
        for m in [MODEL_TAG, "forest_dNone_K8", "forest_d_none_K20", "best_avg_pair", "random"]:
            r = full_agg[(full_agg.protocol == proto) & (full_agg.model == m)]
            if len(r) == 0: continue
            r = r.iloc[0]
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| {proto} | `{m}` | {r['top1']:.3f} | {t3} | "
                     f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
        L.append("")

    L.append("(Note: `forest_dNone_K8` is from the full-grid analysis in `cross_pair_ranking_grid_2026-05-25/`. If it's not present in this folder's `aggregate_per_protocol.csv` the row will be missing here.)")
    L.append("")

    L.append("## Per-instance LOIO breakdown")
    L.append("")
    L.append("| instance | n | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in p_lo_new.sort_values("instance").iterrows():
        L.append(f"| {r['instance']} | {r['n_states']} | {r['top1']:.3f} | "
                 f"{r['top3']:.3f} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 5-fold vs LOIO transfer")
    L.append("")
    L.append("| instance | 5-fold top-1 | LOIO top-1 | Δ top-1 | 5-fold abs | LOIO abs | Δ abs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    def fmt(v): return "n/a" if pd.isna(v) else f"{v:.3f}"
    def fmta(v): return "n/a" if pd.isna(v) else f"{v:.4f}"
    for _, r in side_new.sort_values("instance").iterrows():
        L.append(f"| {r['instance']} | {fmt(r['fold5_top1'])} | {fmt(r['loio_top1'])} | "
                 f"{fmt(r['delta_top1'])} | {fmta(r['fold5_abs'])} | {fmta(r['loio_abs'])} | "
                 f"{fmta(r['delta_abs'])} |")
    L.append("")

    (OUT / "XGB_RANK_ADDENDUM.md").write_text("\n".join(L))
    print(f"\nWrote {OUT / 'XGB_RANK_ADDENDUM.md'}")


if __name__ == "__main__":
    main()
