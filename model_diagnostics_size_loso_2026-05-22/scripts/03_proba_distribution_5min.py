#!/usr/bin/env python3
"""Distribution of predicted probabilities on the 5_min test set.

Re-fits the 9 size-LOSO forests (train on {30s,1m,2m,15min}, test on 5_min)
and reports the proba distribution per pair, split by true class.

For each pair:
  - mean / median / std / min / max of predict_proba on positives and on negatives
  - separation = mean(proba|y=1) − mean(proba|y=0)
  - decile bins of proba with TPR per bin (so we can see WHERE the positives sit)
  - ASCII histogram (20 bins) overlaid for the two classes

Also: confusion matrix at threshold 0.5 to show why shaw pairs hit MCC=0
      (they likely never cross 0.5).

Outputs:
  tables/proba_distribution.csv
  tables/proba_decile_breakdown.csv
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
OUT = REPO / "model_diagnostics_size_loso_2026-05-22"
TBL = OUT / "tables"

USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]
FEATURE_NAMES = [
    "num_active_requests", "radius_avg_tt", "frac_committed", "avg_route_load",
    "temperature_normalized", "ride_time_utilization_avg",
    "ride_time_utilization_max", "near_violation_count", "route_slack_avg",
    "route_slack_std", "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "neighborhood_size_frac", "cost_vs_initial_frac",
    "cost_vs_incumbent_frac", "iteration_norm", "unassigned_count",
    "committed_travel_time_frac", "horizon_elapsed_frac",
]
TRAIN_SIZES = {"30_s", "01_min", "02_min", "15min"}
TEST_SIZE   = "05_min"
RE_LOG = re.compile(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min")


def ascii_hist(p_pos, p_neg, n_bins=20, width=40):
    """Two-class overlay: '+' for positives, '·' for negatives. 20 bins on [0,1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    counts_pos, _ = np.histogram(p_pos, bins=edges)
    counts_neg, _ = np.histogram(p_neg, bins=edges)
    max_count = max(counts_pos.max() if len(counts_pos) else 0,
                    counts_neg.max() if len(counts_neg) else 0, 1)
    lines = []
    for i in range(n_bins):
        lo = edges[i]; hi = edges[i + 1]
        n_pos_bar = round(width * counts_pos[i] / max_count)
        n_neg_bar = round(width * counts_neg[i] / max_count)
        bar = "+" * n_pos_bar + "·" * n_neg_bar
        lines.append(f"    [{lo:.2f}-{hi:.2f}]  pos={counts_pos[i]:>5d}  neg={counts_neg[i]:>5d}  {bar}")
    return "\n".join(lines)


def main():
    X_raw = pd.read_parquet(CACHE / "features.parquet")
    Y_raw = pd.read_parquet(CACHE / "labels_t0.7.parquet")
    merged = X_raw.merge(Y_raw, on=["log_file", "iter"], how="inner")

    sizes = merged["log_file"].map(lambda n: (RE_LOG.match(n).group("size") if RE_LOG.match(n) else None))
    train_mask = sizes.isin(TRAIN_SIZES)
    test_mask  = (sizes == TEST_SIZE)

    X_train = merged.loc[train_mask, FEATURE_NAMES].values
    X_test  = merged.loc[test_mask,  FEATURE_NAMES].values
    print(f"train rows: {train_mask.sum()},  test rows: {test_mask.sum()}\n")

    dist_rows = []; dec_rows = []

    for pair in USED_PAIRS:
        y_train = merged.loc[train_mask, pair].values.astype(int)
        y_test  = merged.loc[test_mask,  pair].values.astype(int)
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"\n{pair}: degenerate train, skipping"); continue

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        p_pos = proba[y_test == 1]
        p_neg = proba[y_test == 0]
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())

        print(f"\n{'=' * 78}")
        print(f"{pair}  (prev_train={y_train.mean():.3f}  prev_test={y_test.mean():.3f}  n_test={len(y_test)})")
        print(f"{'=' * 78}")
        print(f"  proba on POSITIVES (n={len(p_pos)}):  "
              f"mean={p_pos.mean():.3f}  median={np.median(p_pos):.3f}  "
              f"std={p_pos.std():.3f}  min={p_pos.min():.3f}  max={p_pos.max():.3f}")
        print(f"  proba on NEGATIVES (n={len(p_neg)}):  "
              f"mean={p_neg.mean():.3f}  median={np.median(p_neg):.3f}  "
              f"std={p_neg.std():.3f}  min={p_neg.min():.3f}  max={p_neg.max():.3f}")
        print(f"  separation (mean_pos − mean_neg): {p_pos.mean() - p_neg.mean():+.3f}")
        print(f"  share of test proba ≥ 0.5: {(proba >= 0.5).mean():.3f}  "
              f"share ≥ 0.3: {(proba >= 0.3).mean():.3f}  "
              f"share ≥ 0.1: {(proba >= 0.1).mean():.3f}")
        print(f"  confusion matrix @ 0.5: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"\n  histogram (+ = positives, · = negatives, 20 bins on [0,1]):")
        print(ascii_hist(p_pos, p_neg))

        # Decile breakdown: per quantile of proba, what's the TPR?
        deciles = np.linspace(0, 1, 11)
        print(f"\n  decile breakdown (rows = bottom→top 10% of predicted proba):")
        print(f"    {'decile':<8} {'p_min':>7} {'p_max':>7} {'n':>6} {'n_pos':>6} {'TPR':>6}")
        n = len(proba)
        sorted_idx = np.argsort(proba)
        for d in range(10):
            lo = d * n // 10; hi = (d + 1) * n // 10
            idx = sorted_idx[lo:hi]
            n_d = len(idx); n_pos_d = int(y_test[idx].sum())
            tpr = n_pos_d / n_d if n_d else 0
            p_min = proba[idx].min() if n_d else 0
            p_max = proba[idx].max() if n_d else 0
            print(f"    d{d+1:<7d} {p_min:>7.3f} {p_max:>7.3f} {n_d:>6d} {n_pos_d:>6d} {tpr:>6.3f}")
            dec_rows.append({"pair": pair, "decile": d + 1, "p_min": p_min, "p_max": p_max,
                             "n": n_d, "n_pos": n_pos_d, "tpr": tpr})

        dist_rows.append({
            "pair": pair, "n_test": len(y_test), "prev_test": float(y_test.mean()),
            "prev_train": float(y_train.mean()),
            "mean_pos": float(p_pos.mean()), "mean_neg": float(p_neg.mean()),
            "separation": float(p_pos.mean() - p_neg.mean()),
            "max_proba": float(proba.max()), "min_proba": float(proba.min()),
            "share_ge_0.5": float((proba >= 0.5).mean()),
            "share_ge_0.3": float((proba >= 0.3).mean()),
            "share_ge_0.1": float((proba >= 0.1).mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    pd.DataFrame(dist_rows).to_csv(TBL / "proba_distribution.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(TBL / "proba_decile_breakdown.csv", index=False)
    print(f"\nWritten: {TBL / 'proba_distribution.csv'}")
    print(f"         {TBL / 'proba_decile_breakdown.csv'}")


if __name__ == "__main__":
    main()
