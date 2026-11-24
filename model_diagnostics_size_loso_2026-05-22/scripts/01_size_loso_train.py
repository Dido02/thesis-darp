#!/usr/bin/env python3
"""Size-LOSO (Leave-One-Size-Out) diagnostic.

Train: all logs with size in {30_s, 01_min, 02_min, 15min}
Test:  all logs with size = 05_min
Target: t07 (labels_t0.7.parquet)
Model: deployed config (RandomForestClassifier, n_estimators=100,
       max_depth=None, K=20 features, class_weight='balanced',
       random_state=42)

For each of the 9 USED pairs we report:
  MCC, F1, F1-lift, precision, recall, ROC-AUC, PR-AUC, Brier, ECE,
  balanced accuracy, prevalence_train, prevalence_test, n_train, n_test.

Plus state-level top-K joint accuracy using `improve_freq_d_r` as the
"true best" pair per state.

Output:
  tables/size_loso_per_pair.csv   — full metric bundle
  tables/size_loso_topk.csv       — state-level top-1/top-3 accuracy
  prints to stdout: a one-screen summary

Compare these numbers against:
  model_diagnostics_robust_2026-05-19/tables/loio_raw_results.json (the
  LOIO version that DID see other 5_min folds in training). The size-LOSO
  numbers should be WORSE — that's the size-transfer penalty.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
OUT = REPO / "model_diagnostics_size_loso_2026-05-22"
TBL = OUT / "tables"
TBL.mkdir(parents=True, exist_ok=True)

USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]
USED_IMPROVE = [p.replace("pair_", "improve_freq_") for p in USED_PAIRS]
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
RE_LOG = re.compile(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+")


def compute_ece(y_true, y_proba, n_bins=10):
    y_true = np.asarray(y_true, float); y_proba = np.asarray(y_proba, float)
    if len(y_true) == 0: return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba <= hi if i == n_bins - 1 else y_proba < hi)
        nb = mask.sum()
        if nb == 0: continue
        ece += (nb / n) * abs(y_proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def metrics_bundle(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n = len(y_true); pos = int(y_true.sum())
    prev = pos / n if n else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_a1 = (2 * prev / (1 + prev)) if prev > 0 else 0.0
    out = {
        "mcc":        float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "f1":         f1,
        "f1_lift":    f1 - f1_a1,
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": (float(recall_score(y_true, y_pred, zero_division=0)) + specificity) / 2,
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"]  = float(average_precision_score(y_true, y_proba))
        out["brier"]   = float(brier_score_loss(y_true, y_proba))
        out["ece"]     = compute_ece(y_true, y_proba)
    else:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = out["ece"] = None
    out["n"] = n; out["n_positive"] = pos; out["prevalence"] = prev
    return out


def parse_size(log: str):
    m = RE_LOG.match(log)
    return m.group("size") if m else None


def main():
    print("Loading cache_v3_full ...")
    X_raw = pd.read_parquet(CACHE / "features.parquet")
    Y_raw = pd.read_parquet(CACHE / "labels_t0.7.parquet")
    print(f"  raw X={len(X_raw)}, raw Y={len(Y_raw)} — inner-merging on (log_file, iter)")

    # Align on (log_file, iter). Y has slightly fewer rows than X (some states
    # were dropped during label generation). Inner-merge to keep only rows
    # present in both.
    merged = X_raw.merge(Y_raw, on=["log_file", "iter"], how="inner",
                         suffixes=("", "_y"))
    print(f"  merged rows: {len(merged)}")
    X = merged[FEATURE_NAMES + ["log_file", "iter"]]
    Y = merged[["log_file", "iter"] + USED_PAIRS + USED_IMPROVE]

    log_files = merged["log_file"]
    sizes = log_files.map(parse_size)

    train_mask = sizes.isin(TRAIN_SIZES)
    test_mask  = (sizes == TEST_SIZE)
    print(f"  train rows: {train_mask.sum()}  (sizes: {sorted(TRAIN_SIZES)})")
    print(f"  test  rows: {test_mask.sum()}  (size: {TEST_SIZE})")

    X_train = X.loc[train_mask, FEATURE_NAMES].values
    X_test  = X.loc[test_mask,  FEATURE_NAMES].values
    log_test = log_files[test_mask].values

    rows = []
    proba_test_all = np.zeros((test_mask.sum(), len(USED_PAIRS)), dtype=float)

    for j, pair in enumerate(USED_PAIRS):
        y_train = Y.loc[train_mask, pair].values.astype(int)
        y_test  = Y.loc[test_mask,  pair].values.astype(int)
        prev_train = float(y_train.mean()); prev_test = float(y_test.mean())

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"  {pair}: degenerate train (prev={prev_train}); skipping")
            continue

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        proba_test_all[:, j] = y_proba

        m = metrics_bundle(y_test, y_pred, y_proba)
        m.update({"pair": pair, "prev_train": prev_train, "prev_test": prev_test,
                  "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})
        rows.append(m)
        print(f"  {pair}: MCC={m['mcc']:+.3f}  F1-lift={m['f1_lift']:+.3f}  ROC={m['roc_auc']:.3f}  Brier={m['brier']:.3f}  prev_train={prev_train:.3f} prev_test={prev_test:.3f}")

    per_pair = pd.DataFrame(rows)
    per_pair.to_csv(TBL / "size_loso_per_pair.csv", index=False)

    # --- state-level top-K joint ---
    # "true best" pair = argmax over 9 USED improve_freq cols at this state.
    # If no positive improve_freq exists, mark the state as 'no signal' and skip.
    print("\nState-level top-K (using improve_freq as 'true best' signal) ...")
    Y_freq = Y.loc[test_mask, USED_IMPROVE].values
    has_signal = Y_freq.max(axis=1) > 0
    true_best = np.argmax(Y_freq, axis=1)
    pred_rank = np.argsort(-proba_test_all, axis=1)

    n_eval = int(has_signal.sum())
    top1 = float((pred_rank[has_signal, 0] == true_best[has_signal]).mean())
    top3 = float(np.array(
        [true_best[i] in pred_rank[i, :3] for i in np.where(has_signal)[0]]
    ).mean())
    cost_strict = float(Y_freq[has_signal, true_best[has_signal]].sum()
                        - Y_freq[has_signal, pred_rank[has_signal, 0]].sum())
    cost_strict_mean = cost_strict / n_eval

    print(f"  states with signal: {n_eval}/{len(has_signal)}")
    print(f"  top-1 acc: {top1:.3f}  (random baseline = 1/9 = {1/9:.3f})")
    print(f"  top-3 acc: {top3:.3f}  (random baseline = 3/9 = {3/9:.3f})")
    print(f"  cost regret (mean improve_freq foregone): {cost_strict_mean:.5f}")

    topk = pd.DataFrame([{
        "n_test": int(test_mask.sum()), "n_eval": n_eval,
        "top1_acc": top1, "top3_acc": top3,
        "random_top1": 1 / 9, "random_top3": 3 / 9,
        "cost_regret_mean": cost_strict_mean,
    }])
    topk.to_csv(TBL / "size_loso_topk.csv", index=False)

    # --- one-screen summary ---
    print("\n" + "=" * 70)
    print("SIZE-LOSO SUMMARY")
    print("=" * 70)
    print(f"\n  Train sizes: {sorted(TRAIN_SIZES)}  ({train_mask.sum()} rows)")
    print(f"  Test size:   {TEST_SIZE}             ({test_mask.sum()} rows)")
    print(f"\n  Per-pair metrics (MCC = chance-corrected, range [-1, +1]):")
    print(per_pair[["pair", "mcc", "f1", "f1_lift", "roc_auc", "pr_auc",
                   "brier", "ece", "prev_train", "prev_test"]]
          .round(4).to_string(index=False))
    print(f"\n  Aggregate MCC across 9 pairs: mean={per_pair['mcc'].mean():.3f}  "
          f"median={per_pair['mcc'].median():.3f}  "
          f"min={per_pair['mcc'].min():.3f}  max={per_pair['mcc'].max():.3f}")
    print(f"  Aggregate F1-lift across 9 pairs: mean={per_pair['f1_lift'].mean():+.3f}  "
          f"median={per_pair['f1_lift'].median():+.3f}")
    print(f"  Aggregate ROC-AUC across 9 pairs: mean={per_pair['roc_auc'].mean():.3f}  "
          f"median={per_pair['roc_auc'].median():.3f}")
    print(f"\n  State-level (9-pair joint):")
    print(f"    top-1: {top1:.3f}   (random = 0.111)")
    print(f"    top-3: {top3:.3f}   (random = 0.333)")
    print(f"\n  Written: {TBL / 'size_loso_per_pair.csv'}")
    print(f"           {TBL / 'size_loso_topk.csv'}")


if __name__ == "__main__":
    main()
