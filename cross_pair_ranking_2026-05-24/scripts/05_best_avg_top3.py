#!/usr/bin/env python3
"""best_avg_top3 baseline addendum.

At each state, predict the THREE pairs with the highest mean ei_pct on the
training portion of each fold, in descending order. The selection is
constant across all test states within a fold but chosen per-fold (just
like best_avg_pair).

Metrics:
  top-1   = same definition as best_avg_pair, identical numeric value
            because top-1 element is the top-mean-ei_pct pair.
  top-3   = whether actual best is in the predicted top-3 (constant pair
            triple per fold).
  abs_reg = regret of the top-1 prediction (identical to best_avg_pair).
  rel_reg = same.

All state-weighted, same valid-state mask as everywhere else in this
folder.

Append-only:
  state_level_predictions.csv     <- per-state best_avg_top3 rows
  aggregate_per_protocol.csv      <- 3 rows
  per_instance_loio.csv           <- 14 rows
  CROSS_PAIR_RANKING_REPORT.md    <- regenerated with new baseline added

Audit:
  best_avg_top3_choices.csv       <- chosen top-3 pair indices/names per fold
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd

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
MODEL_TAG = "best_avg_top3"


# ============================================================
# Per-state metric: top-3 constant predictor
# ============================================================
def baseline_best_avg_top3_per_state(ei_train, ei_test_valid):
    """Return per-state arrays for a top-3 constant predictor whose
    selection is the train-mean-ei_pct argsort top-3. Top-1 is the first
    of the three; top-3 considers all three.
    """
    if ei_train.size == 0 or ei_test_valid.size == 0:
        empty = np.array([], dtype=float)
        return {"top3_indices": [-1, -1, -1],
                "top1_hit": empty, "top3_hit": empty,
                "abs_reg": empty, "rel_reg": empty}
    pair_means = ei_train.mean(axis=0)
    top3_idx = np.argsort(-pair_means)[:3]              # descending order
    pair_top1 = int(top3_idx[0])
    top3_set = set(int(i) for i in top3_idx)
    E = ei_test_valid
    arange = np.arange(len(E))
    true_best = E.argmax(axis=1)
    top1_hit = (true_best == pair_top1).astype(float)
    top3_hit = np.array([tb in top3_set for tb in true_best], dtype=float)
    abs_reg = E[arange, true_best] - E[:, pair_top1]
    rel_reg = abs_reg / E[arange, true_best] * 100.0
    return {"top3_indices": [int(i) for i in top3_idx],
            "top1_hit": top1_hit, "top3_hit": top3_hit,
            "abs_reg": abs_reg, "rel_reg": rel_reg}


# ============================================================
# Iterate folds, emit per-state rows + audit-log the chosen triples
# ============================================================
def emit_rows(df, fold_iter_fn, protocol_label, chosen_triples):
    rows = []
    folds = list(fold_iter_fn(df))
    print(f"  [{protocol_label}] {len(folds)} folds")
    for fold in folds:
        tr, te = fold["train_mask"], fold["test_mask"]
        ei_train = df.loc[tr, xpr.USED_EI].values
        ei_test  = df.loc[te, xpr.USED_EI].values
        test_inst = fold["test_instances"]
        max_ei = ei_test.max(axis=1)
        valid = max_ei > 0
        if not valid.any(): continue
        E_valid = ei_test[valid]
        valid_inst = test_inst[valid]
        res = baseline_best_avg_top3_per_state(ei_train, E_valid)
        chosen_triples.append({
            "protocol": protocol_label, "fold": fold["name"],
            "top1_idx": res["top3_indices"][0],
            "top2_idx": res["top3_indices"][1],
            "top3_idx": res["top3_indices"][2],
            "top1_name": xpr.USED_PAIRS[res["top3_indices"][0]] if res["top3_indices"][0] >= 0 else "n/a",
            "top2_name": xpr.USED_PAIRS[res["top3_indices"][1]] if res["top3_indices"][1] >= 0 else "n/a",
            "top3_name": xpr.USED_PAIRS[res["top3_indices"][2]] if res["top3_indices"][2] >= 0 else "n/a",
            "n_train_rows": int(tr.sum()), "n_test_states": int(valid.sum()),
        })
        for k in range(int(valid.sum())):
            rows.append({
                "protocol": protocol_label, "fold": fold["name"],
                "model": MODEL_TAG, "instance": valid_inst[k],
                "top1_hit": res["top1_hit"][k], "top3_hit": res["top3_hit"][k],
                "abs_reg":  res["abs_reg"][k],  "rel_reg":  res["rel_reg"][k],
            })
    return rows


# ============================================================
# Append-only helper
# ============================================================
def append_only(csv_path: Path, new_rows: pd.DataFrame, model_tag: str):
    existing = pd.read_csv(csv_path)
    kept = existing[existing["model"] != model_tag]
    out = pd.concat([kept, new_rows], ignore_index=True)
    out.to_csv(csv_path, index=False)
    return len(kept), len(new_rows)


# ============================================================
# Aggregations (only for best_avg_top3 rows)
# ============================================================
def agg_for_model(state_df, model_tag):
    sub = state_df[state_df.model == model_tag]
    out = []
    for proto, grp in sub.groupby("protocol"):
        t3 = float(grp["top3_hit"].mean()) if grp["top3_hit"].notna().any() else np.nan
        out.append({"protocol": proto, "model": model_tag, "n_states": int(len(grp)),
                    "top1": float(grp["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(grp["abs_reg"].mean()),
                    "rel_reg": float(grp["rel_reg"].mean())})
    return pd.DataFrame(out)

def per_instance_for_model(state_df, model_tag, protocol):
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


# ============================================================
# Regenerate CROSS_PAIR_RANKING_REPORT.md with best_avg_top3 included
# (preserves existing models that were in the report; adds best_avg_top3
#  right after best_avg_pair so they sit together).
# ============================================================
def regenerate_report(agg_csv: Path, p_lo_csv: Path, p_5f_csv: Path, side_csv: Path,
                      out_md: Path):
    agg = pd.read_csv(agg_csv)
    p_lo = pd.read_csv(p_lo_csv)
    p_5f = pd.read_csv(p_5f_csv)
    side = pd.read_csv(side_csv)

    # Same model_order as the original 01_cross_pair_ranking.py write_md(),
    # but with best_avg_top3 inserted right after best_avg_pair.
    model_order = ["random", "best_avg_pair", "best_avg_top3",
                   "forest_d_none_K20", "forest_d8_K20", "forest_d8_K16",
                   "tree_d4_K4"]
    proto_order = ["deployed_75_25", "5fold", "LOIO"]

    L = []
    L.append("# Cross-pair Ranking — bench-proxy for the deployed softmax sampler")
    L.append("")
    L.append("**Setup:** For each (state, model) we compute 9 per-pair `predict_proba`, join with ground-truth `ei_pct_{d}_{r}` over the 9 USED pairs, skip states with `max(ei_pct) = 0`, and measure top-1, top-3, mean absolute regret, and mean relative regret. Models trained per (destroy, repair) pair (binary t07 label).")
    L.append("")
    L.append("**Baselines:** `random` (top-1 = 1/9, top-3 = 3/9, per-state regret from `max(ei) − mean(ei)`); `best_avg_pair` (per-fold single-pair constant predictor by train-mean `ei_pct` argmax, top-3 undefined); `best_avg_top3` (per-fold top-3 constant predictor by train-mean `ei_pct`; top-1 identical to `best_avg_pair`; top-3 well-defined; abs/rel regret use the top-1 pair, identical to `best_avg_pair`).")
    L.append("")

    L.append("## 1. Aggregate per protocol per model")
    L.append("")
    L.append("| protocol | model | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for p in proto_order:
        for m in model_order:
            row = agg[(agg.protocol == p) & (agg.model == m)]
            if len(row) == 0: continue
            r = row.iloc[0]
            t1 = f"{r['top1']:.3f}" if pd.notna(r["top1"]) else "n/a"
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| {p} | `{m}` | {int(r['n_states'])} | {t1} | {t3} | "
                     f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")
    L.append("**How to read:** higher top-K and lower regret = better. Random ≈ 0.111 / 0.333 and a fixed positive abs regret. `best_avg_pair` and `best_avg_top3` share top-1 and abs/rel regret by construction (same primary pair); they differ on top-3 (`best_avg_pair` is N/A; `best_avg_top3` is the per-fold top-3 hit rate). The deployed sampler effectively samples by softmax-T=0.10 from the model's proba vector — its bench behavior should sit between top-1 and top-3 in expectation.")
    L.append("")

    L.append("## 2. Per-instance breakdown — LOIO protocol")
    L.append("")
    L.append("Each row = (model, held-out instance bucket).")
    L.append("")
    L.append("| model | instance | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in model_order:
        sub = p_lo[p_lo.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| `{m}` | {r['instance']} | {int(r['n_states'])} | "
                     f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 3. Per-instance breakdown — 5-fold protocol")
    L.append("")
    L.append("| model | instance | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in model_order:
        sub = p_5f[p_5f.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| `{m}` | {r['instance']} | {int(r['n_states'])} | "
                     f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 4. Side-by-side — 5-fold vs LOIO per (model, instance)")
    L.append("")
    L.append("Positive `Δ top-1` = LOIO retained or improved skill; negative = transfer collapse on that instance.")
    L.append("")
    L.append("| model | instance | 5fold top-1 | LOIO top-1 | Δ top-1 | 5fold abs reg | LOIO abs reg | Δ abs reg |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in model_order:
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
    L.append("- **`best_avg_top3` top-3 vs `forest_d_none_K20` top-3** isolates whether the model's ranking signal beats a constant-pair-triple baseline. If forest top-3 ≈ best_avg_top3 top-3, the model isn't doing meaningful per-state discrimination beyond \"the same three worst-destroy pairs are always best\".")
    L.append("")
    L.append("Side analyses in this folder (not included above): `LOGISTIC_ADDENDUM.md` for a multi-class logistic regression baseline; `XGB_RANK_ADDENDUM.md` for an XGBoost learning-to-rank model.")

    out_md.write_text("\n".join(L))


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + reproducing folds (no retraining of any models)...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")

    # 1. Drop prior best_avg_top3 from state-level (idempotent)
    state_csv = TBL / "state_level_predictions.csv"
    existing = pd.read_csv(state_csv)
    print(f"  loaded existing state_level_predictions.csv: {len(existing)} rows")
    if MODEL_TAG in existing["model"].unique():
        print(f"  dropping previous {MODEL_TAG} rows for clean recompute")
        existing = existing[existing["model"] != MODEL_TAG]

    # 2. Generate per-state rows for all three protocols
    chosen_triples = []
    new_rows = []
    new_rows += emit_rows(df, xpr.deployed_split, "deployed_75_25", chosen_triples)
    new_rows += emit_rows(df, lambda d: xpr.fivefold_splits(d), "5fold", chosen_triples)
    new_rows += emit_rows(df, xpr.loio_splits, "LOIO", chosen_triples)
    new_state_df = pd.DataFrame(new_rows)
    print(f"\nGenerated {len(new_state_df)} new {MODEL_TAG} per-state rows")

    # 3. Audit: chosen triples per fold
    triples_df = pd.DataFrame(chosen_triples)
    triples_csv = TBL / "best_avg_top3_choices.csv"
    triples_df.to_csv(triples_csv, index=False)
    print(f"\nChosen top-3 pair indices per fold (audit) -> {triples_csv}:")
    for r in chosen_triples:
        print(f"  {r['protocol']:15s} {r['fold']:30s} "
              f"-> [{r['top1_name']}, {r['top2_name']}, {r['top3_name']}]  "
              f"(n_test_states={r['n_test_states']})")

    # Check whether all folds picked the same triple
    unique_triples = triples_df[["top1_name", "top2_name", "top3_name"]].drop_duplicates()
    print(f"\n  Unique top-3 triples across all {len(triples_df)} folds: {len(unique_triples)}")
    if len(unique_triples) == 1:
        t = unique_triples.iloc[0]
        print(f"  Every fold picked: [{t['top1_name']}, {t['top2_name']}, {t['top3_name']}]")

    # 4. Append to state_level_predictions.csv
    combined_state = pd.concat([existing, new_state_df], ignore_index=True)
    combined_state.to_csv(state_csv, index=False)
    print(f"\nstate_level_predictions.csv: {len(existing)} kept + "
          f"{len(new_state_df)} new = {len(combined_state)} rows")

    # 5. Aggregate just for best_avg_top3, append to aggregate_per_protocol.csv
    agg_new = agg_for_model(combined_state, MODEL_TAG)
    a_keep, a_new = append_only(TBL / "aggregate_per_protocol.csv", agg_new, MODEL_TAG)
    print(f"aggregate_per_protocol.csv: kept {a_keep}, appended {a_new}")
    print(f"\nbest_avg_top3 aggregate (state-weighted):")
    print(agg_new.sort_values("protocol").to_string(index=False))

    # 6. Per-instance LOIO just for best_avg_top3, append
    p_lo_new = per_instance_for_model(combined_state, MODEL_TAG, "LOIO")
    l_keep, l_new = append_only(TBL / "per_instance_loio.csv", p_lo_new, MODEL_TAG)
    print(f"per_instance_loio.csv: kept {l_keep}, appended {l_new}")

    # 7. Regenerate CROSS_PAIR_RANKING_REPORT.md including best_avg_top3
    regenerate_report(
        TBL / "aggregate_per_protocol.csv",
        TBL / "per_instance_loio.csv",
        TBL / "per_instance_5fold.csv",
        TBL / "side_by_side.csv",
        OUT / "CROSS_PAIR_RANKING_REPORT.md",
    )
    print(f"\nRegenerated {OUT / 'CROSS_PAIR_RANKING_REPORT.md'} (with best_avg_top3 alongside best_avg_pair)")


if __name__ == "__main__":
    main()
