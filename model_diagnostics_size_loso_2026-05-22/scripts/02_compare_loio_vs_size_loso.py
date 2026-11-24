#!/usr/bin/env python3
"""Side-by-side: LOIO (held out 1 of 3 5_min folds, others in training)
vs size-LOSO (held out ALL 5_min, trained on smaller sizes only).

LOIO: 3 folds matching the size-LOSO test set per pair, target t07,
      model forest_baseline_d_none_K20.
size-LOSO: from tables/size_loso_per_pair.csv produced by 01_size_loso_train.py.

Prints a side-by-side table per pair and a one-row aggregate.
MCC computed from TP/FP/TN/FN since the LOIO dump didn't store MCC.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LOIO_JSON = REPO / "model_diagnostics_robust_2026-05-19" / "tables" / "loio_raw_results.json"
SIZE_CSV  = REPO / "model_diagnostics_size_loso_2026-05-22" / "tables" / "size_loso_per_pair.csv"
USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]


def mcc_from_cm(tp, fp, tn, fn):
    n = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if n == 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(n)


def main():
    data = json.load(open(LOIO_JSON))
    folds_5min = [f for f in data["folds_meta"] if f["instance"].startswith("05_min")]
    fold_ids = {f["fold"] for f in folds_5min}
    print(f"5_min folds in LOIO: {sorted(fold_ids)}  "
          f"instances: {[f['instance'] for f in folds_5min]}")

    # Filter LOIO results. Two aggregations: mean-of-per-fold and pooled-CM.
    rows = []
    for r in data["results"]:
        if r["target"] != "t07": continue
        if r["model_label"] != "forest_baseline_d_none_K20": continue
        if r["fold"] not in fold_ids: continue
        for pair in USED_PAIRS:
            pm = r["pair_metrics"].get(pair, {})
            tp = pm.get("test_tp", 0); fp = pm.get("test_fp", 0)
            tn = pm.get("test_tn", 0); fn = pm.get("test_fn", 0)
            rows.append({
                "fold": r["fold"], "instance": r["instance"], "pair": pair,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "mcc": mcc_from_cm(tp, fp, tn, fn),
                "f1": pm.get("test_f1"), "f1_lift": pm.get("test_f1_lift"),
                "roc_auc": pm.get("test_roc_auc"), "pr_auc": pm.get("test_pr_auc"),
                "brier": pm.get("test_brier"), "ece": pm.get("test_ece_10bin"),
                "prev_test": pm.get("test_prevalence"),
                "n": pm.get("test_n", 0), "n_positive": pm.get("test_n_positive", 0),
            })
    loio = pd.DataFrame(rows)
    print(f"LOIO rows (t07, baseline, 5_min folds × 9 pairs): {len(loio)}")

    # Mean-of-per-fold aggregation (original)
    loio_mean = loio.groupby("pair").agg({
        "mcc": "mean", "f1": "mean", "f1_lift": "mean",
        "roc_auc": "mean", "pr_auc": "mean", "brier": "mean", "ece": "mean",
        "prev_test": "mean",
    }).reset_index().rename(columns={"mcc": "mcc_mean", "f1": "f1_mean"})

    # Pooled-CM aggregation: sum TP/FP/TN/FN across the 3 5_min folds,
    # compute MCC/F1 from the pooled CM. Apples-to-apples with size-LOSO.
    pool_rows = []
    for pair in USED_PAIRS:
        sub = loio[loio.pair == pair]
        tp = int(sub.tp.sum()); fp = int(sub.fp.sum())
        tn = int(sub.tn.sum()); fn = int(sub.fn.sum())
        n_total = tp + fp + tn + fn
        n_pos = tp + fn
        prev = n_pos / n_total if n_total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        pool_rows.append({
            "pair": pair,
            "mcc_pooled": mcc_from_cm(tp, fp, tn, fn),
            "f1_pooled": f1,
            "tp_pool": tp, "fp_pool": fp, "tn_pool": tn, "fn_pool": fn,
            "n_pool": n_total, "prev_pool": prev,
            "precision_pool": precision, "recall_pool": recall,
        })
    loio_pool = pd.DataFrame(pool_rows)
    loio_agg = loio_mean.merge(loio_pool, on="pair")

    size_df = pd.read_csv(SIZE_CSV)[
        ["pair", "mcc", "f1", "f1_lift", "roc_auc", "pr_auc", "brier", "ece",
         "prev_train", "prev_test"]
    ]

    # Merge and present
    cmp = pd.merge(
        loio_agg.add_suffix("_loio"),
        size_df.add_suffix("_size"),
        left_on="pair_loio", right_on="pair_size",
    )

    print("\n" + "=" * 130)
    print("PER-PAIR COMPARISON: LOIO (5_min in train) vs size-LOSO (5_min never seen)")
    print("LOIO_pooled = MCC computed from summed CM across 3 5_min folds")
    print("LOIO_mean   = mean of per-fold MCC across 3 5_min folds")
    print("=" * 130)
    print(f"\n{'pair':<10} {'LOIO_mean':>10} {'LOIO_pool':>10} {'size_LOSO':>10} "
          f"{'Δ (size-pool)':>14} {'LOIO_pool_F1':>12} {'size_F1':>9} "
          f"{'prev_pool':>10} {'prev_size':>10} {'n_pool':>8}")
    print("-" * 130)
    for _, row in cmp.iterrows():
        d_mcc = row.mcc_size - row.mcc_pooled_loio
        print(f"{row.pair_size:<10} {row.mcc_mean_loio:>10.3f} {row.mcc_pooled_loio:>10.3f} "
              f"{row.mcc_size:>10.3f} {d_mcc:>+14.3f} {row.f1_pooled_loio:>12.3f} "
              f"{row.f1_size:>9.3f} {row.prev_pool_loio:>10.3f} {row.prev_test_size:>10.3f} "
              f"{int(row.n_pool_loio):>8d}")

    print(f"\n{'AGGREGATE (mean over 9 pairs):':<25}")
    print(f"  LOIO mean-of-folds MCC: {cmp.mcc_mean_loio.mean():.3f}")
    print(f"  LOIO pooled-CM MCC:     {cmp.mcc_pooled_loio.mean():.3f}")
    print(f"  size-LOSO MCC:          {cmp.mcc_size.mean():.3f}")
    print(f"  Δ (size − pooled):      {(cmp.mcc_size - cmp.mcc_pooled_loio).mean():+.3f}")
    print(f"  LOIO pooled F1:    {cmp.f1_pooled_loio.mean():.3f}")
    print(f"  size-LOSO F1:      {cmp.f1_size.mean():.3f}")

    out_csv = SIZE_CSV.parent / "loio_vs_size_loso_comparison.csv"
    cmp.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
