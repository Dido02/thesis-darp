#!/usr/bin/env python3
"""Aggregation fix: make baselines (random, best_avg_pair) state-weighted
to match the ML models.

Previously baselines emitted ONE aggregate row per fold (`instance="_AGG_"`),
so `agg_per_protocol_model` averaged across folds — fold-weighted. ML models
emitted one row per state, so their aggregation was state-weighted. With
unequal fold sizes (LOIO especially: 30_s_d03 = 1290 states vs 15min_d05 =
258 states), this is inconsistent.

This patch:
  - Keeps the ML model per-state predictions intact (no retraining).
  - Recomputes baselines per-state per fold from labels_t0.3_with_ei.parquet.
  - For best_avg_pair: per-fold pair selection is preserved (chosen by mean
    ei_pct on TRAIN); the resulting per-state metrics are then pooled across
    all (fold, state) into a single state-weighted aggregate.
  - For random: per-state metrics are emitted (top-1 = 1/9, top-3 = 3/9
    constants; abs/rel regret = (max-mean)/... per state). Pooled aggregation.
  - Rewrites all four CSVs and the markdown report. The 75/25 protocol has
    a single fold so its baselines are unaffected numerically.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Reuse helpers + constants from the original script.
import importlib.util
spec = importlib.util.spec_from_file_location("xpr",
    SCRIPTS / "01_cross_pair_ranking.py")
xpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xpr)

OUT = REPO / "cross_pair_ranking_2026-05-24"
TBL = OUT / "tables"


# ============================================================
# Per-state baselines (the actual fix)
# ============================================================
def baseline_random_per_state(ei_test_valid):
    """Per-state expected metrics for the random baseline.
    top1/top3 are constant theoretical expectations (1/9 and 3/9).
    abs/rel regret are realized per-state quantities.
    """
    E = ei_test_valid
    if len(E) == 0:
        empty = np.array([], dtype=float)
        return {"top1_hit": empty, "top3_hit": empty,
                "abs_reg": empty, "rel_reg": empty}
    best = E.max(axis=1)
    mean_ = E.mean(axis=1)
    return {
        "top1_hit": np.full(len(E), 1.0 / 9.0, dtype=float),
        "top3_hit": np.full(len(E), 3.0 / 9.0, dtype=float),
        "abs_reg":  best - mean_,
        "rel_reg":  (best - mean_) / best * 100.0,
    }


def baseline_best_avg_pair_per_state(ei_train, ei_test_valid):
    """Pick the pair with highest mean ei_pct on TRAIN; evaluate per-state.

    Per-fold pair selection preserved. Per-state metrics emitted so that
    pooled aggregation across (fold, state) is state-weighted.

    Top-3 is undefined for a single-pair predictor (only one pair in the
    ranking); we report NaN per row so .mean() correctly excludes it.
    """
    if ei_train.size == 0 or ei_test_valid.size == 0:
        empty = np.array([], dtype=float)
        return {"pair_idx": -1, "top1_hit": empty, "top3_hit": empty,
                "abs_reg": empty, "rel_reg": empty}
    pair_idx = int(ei_train.mean(axis=0).argmax())
    E = ei_test_valid
    arange = np.arange(len(E))
    true_best = E.argmax(axis=1)
    top1_hit = (true_best == pair_idx).astype(float)
    top3_hit = np.full(len(E), np.nan)   # N/A for a single-pair predictor
    abs_reg = E[arange, true_best] - E[:, pair_idx]
    rel_reg = abs_reg / E[arange, true_best] * 100.0
    return {"pair_idx": pair_idx, "top1_hit": top1_hit, "top3_hit": top3_hit,
            "abs_reg": abs_reg, "rel_reg": rel_reg}


# ============================================================
# Per-fold baseline row generation
# ============================================================
def emit_baseline_rows(df, fold_iter_fn, protocol_label, bap_chosen_pairs):
    """Iterate folds; for each, compute per-state baseline rows."""
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
        valid_inst = test_inst[valid]
        E_valid = ei_test[valid]
        n = int(valid.sum())

        rnd = baseline_random_per_state(E_valid)
        bap = baseline_best_avg_pair_per_state(ei_train, E_valid)
        bap_chosen_pairs.append({
            "protocol": protocol_label, "fold": fold["name"],
            "pair_idx": bap["pair_idx"],
            "pair_name": xpr.USED_PAIRS[bap["pair_idx"]] if bap["pair_idx"] >= 0 else "n/a",
            "n_train_rows": int(tr.sum()), "n_test_states": n,
        })

        for k in range(n):
            rows.append({
                "protocol": protocol_label, "fold": fold["name"],
                "model": "random", "instance": valid_inst[k],
                "top1_hit": rnd["top1_hit"][k], "top3_hit": rnd["top3_hit"][k],
                "abs_reg":  rnd["abs_reg"][k],  "rel_reg":  rnd["rel_reg"][k],
            })
            rows.append({
                "protocol": protocol_label, "fold": fold["name"],
                "model": "best_avg_pair", "instance": valid_inst[k],
                "top1_hit": bap["top1_hit"][k], "top3_hit": bap["top3_hit"][k],
                "abs_reg":  bap["abs_reg"][k],  "rel_reg":  bap["rel_reg"][k],
            })
    return rows


# ============================================================
# Patched aggregator (now uniform pooled-by-state for everyone)
# ============================================================
def agg_per_protocol_model_pooled(df_state):
    out = []
    for (proto, model), sub in df_state.groupby(["protocol", "model"]):
        # Top-3 may be NaN for best_avg_pair; .mean() skips NaN by default.
        t3 = float(sub["top3_hit"].mean()) if sub["top3_hit"].notna().any() else np.nan
        out.append({
            "protocol": proto, "model": model, "n_states": int(len(sub)),
            "top1": float(sub["top1_hit"].mean()),
            "top3": t3,
            "abs_reg": float(sub["abs_reg"].mean()),
            "rel_reg": float(sub["rel_reg"].mean()),
        })
    return pd.DataFrame(out)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + reproducing folds (no retraining)...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")

    # 1. Compute per-state baseline rows for all three protocols.
    bap_chosen_pairs = []
    base_rows = []
    base_rows += emit_baseline_rows(df, xpr.deployed_split, "deployed_75_25", bap_chosen_pairs)
    base_rows += emit_baseline_rows(df, lambda d: xpr.fivefold_splits(d), "5fold", bap_chosen_pairs)
    base_rows += emit_baseline_rows(df, xpr.loio_splits, "LOIO", bap_chosen_pairs)
    baseline_df = pd.DataFrame(base_rows)
    print(f"  generated {len(baseline_df)} baseline per-state rows")
    print(f"  best_avg_pair chosen per fold:")
    pd.DataFrame(bap_chosen_pairs).to_csv(TBL / "best_avg_pair_choices.csv", index=False)
    for r in bap_chosen_pairs:
        print(f"    {r['protocol']:15s} {r['fold']:30s} -> {r['pair_name']} (n_test_states={r['n_test_states']})")

    # 2. Load existing model per-state predictions, drop old baseline rows.
    existing = pd.read_csv(TBL / "state_level_predictions.csv")
    print(f"\n  loaded existing state_level_predictions.csv: {len(existing)} rows")
    model_rows = existing[~existing["model"].isin(["random", "best_avg_pair"])]
    print(f"  model rows kept: {len(model_rows)}; dropped baseline rows: "
          f"{len(existing) - len(model_rows)}")

    # 3. Concat + write patched state_level_predictions.csv.
    new_state = pd.concat([model_rows, baseline_df], ignore_index=True)
    new_state.to_csv(TBL / "state_level_predictions.csv", index=False)
    print(f"  wrote patched state_level_predictions.csv: {len(new_state)} rows "
          f"({len(model_rows)} model + {len(baseline_df)} baseline)")

    # 4. Re-aggregate everything with the consistent pooled-by-state rule.
    print("\nRe-aggregating...")
    agg = agg_per_protocol_model_pooled(new_state)
    agg.to_csv(TBL / "aggregate_per_protocol.csv", index=False)

    per_inst_lo = xpr.agg_per_instance(new_state, "LOIO")
    per_inst_lo.to_csv(TBL / "per_instance_loio.csv", index=False)

    per_inst_5f = xpr.agg_per_instance(new_state, "5fold")
    per_inst_5f.to_csv(TBL / "per_instance_5fold.csv", index=False)

    side = xpr.side_by_side(per_inst_5f, per_inst_lo)
    side.to_csv(TBL / "side_by_side.csv", index=False)

    print("\nPatched aggregate table:")
    print(agg.sort_values(["protocol", "model"]).to_string(index=False))

    # 5. Rewrite the report.
    xpr.write_md(agg, per_inst_5f, per_inst_lo, side, OUT / "CROSS_PAIR_RANKING_REPORT.md")
    print(f"\nWrote {OUT / 'CROSS_PAIR_RANKING_REPORT.md'}")
    print(f"     + {TBL / 'best_avg_pair_choices.csv'} (new)")


if __name__ == "__main__":
    main()
