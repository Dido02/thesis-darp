"""Shared helpers for the robust-model investigation.

Kept self-contained so we don't import from the sibling cv folder.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES_PQ = CACHE / "features.parquet"
OUT_DIR = REPO / "model_diagnostics_robust_2026-05-19"
TBL = OUT_DIR / "tables"
FIG = OUT_DIR / "figures"
FOLD_MODELS = OUT_DIR / "fold_models"

USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]
USED_EI_COLS = [p.replace("pair_", "ei_pct_") for p in USED_PAIRS]
TARGETS = {
    "t02": "labels_t0.2.parquet",
    "t03": "labels_t0.3.parquet",
    "t05": "labels_t0.5.parquet",
    "t07": "labels_t0.7.parquet",
}
MODEL_CONFIGS = [
    ("forest_baseline_d_none_K20", None),
    ("forest_grid_d8_K20", 8),
]

FEATURE_NAMES_20 = [
    "num_active_requests", "radius_avg_tt", "frac_committed", "avg_route_load",
    "temperature_normalized", "ride_time_utilization_avg",
    "ride_time_utilization_max", "near_violation_count", "route_slack_avg",
    "route_slack_std", "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "neighborhood_size_frac", "cost_vs_initial_frac",
    "cost_vs_incumbent_frac", "iteration_norm", "unassigned_count",
    "committed_travel_time_frac", "horizon_elapsed_frac",
]


def parse_log(name: str):
    """Return (size, delay) bucket key. None for unparseable.

    Matches: manhattan_30_s_delay03_min_s1024_t50e100_dynamic.jsonl
             manhattan_01_min_delay05_min_s789_t50e50_dynamic.jsonl
             manhattan_15min_03min_s42_t...
    """
    m = re.match(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+", name)
    if not m:
        return None
    return (m.group("size"), m.group("delay"))


def make_loio_folds(log_files: list[str]):
    """One fold per (size, delay) instance. Deterministic by sorted key."""
    strata = defaultdict(list)
    for n in log_files:
        key = parse_log(n) or ("other", "other")
        strata[key].append(n)
    all_logs_set = set(log_files)
    splits = []
    for i, key in enumerate(sorted(strata)):
        test_logs = sorted(strata[key])
        train_logs = sorted(all_logs_set - set(test_logs))
        instance = f"{key[0]}_d{key[1]}"
        splits.append({
            "fold": i, "instance": instance, "key": key,
            "test_logs": test_logs, "train_logs": train_logs,
        })
    return splits


def compute_baselines(y_test) -> dict:
    n = len(y_test); pos = int(y_test.sum())
    prev = pos / n if n else 0.0
    a0 = (n - pos) / n if n else 0.0
    a1 = pos / n if n else 0.0
    f1_a0 = 0.0
    f1_a1 = (2 * prev / (1 + prev)) if prev > 0 else 0.0
    return {
        "n": int(n), "n_positive": pos, "prevalence": prev,
        "always0_accuracy": a0, "always1_accuracy": a1,
        "best_baseline_accuracy": max(a0, a1),
        "always0_f1": f1_a0, "always1_f1": f1_a1,
        "best_baseline_f1": max(f1_a0, f1_a1),
    }


def compute_ece(y_true, y_proba, n_bins: int = 10) -> float:
    """Expected calibration error with empty-bin handling.

    Drop empty bins from the sum; weight by bin population.
    Returns 0.0 if y_true is empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        mean_conf = float(y_proba[mask].mean())
        mean_true = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(mean_conf - mean_true)
    return float(ece)


def compute_clf_metrics(clf, X, y, *, proba=None) -> dict:
    """Full metric bundle. Pass `proba` to skip predict_proba (e.g. when
    we've already cached probas during phase 2 dumps).
    """
    if len(y) == 0:
        return {"empty": True}
    if proba is None:
        y_pred = clf.predict(X)
        try:
            proba = clf.predict_proba(X)[:, 1]
        except Exception:
            proba = y_pred.astype(float)
    else:
        proba = np.asarray(proba)
        y_pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    bacc = (rec + specificity) / 2
    base = compute_baselines(y)
    out = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": specificity, "f1": f1, "balanced_accuracy": bacc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y, proba))
            out["pr_auc"] = float(average_precision_score(y, proba))
            out["brier"] = float(brier_score_loss(y, proba))
            out["ece_10bin"] = compute_ece(y, proba, n_bins=10)
        except Exception:
            out["roc_auc"] = None; out["pr_auc"] = None
            out["brier"] = None; out["ece_10bin"] = None
    else:
        out["roc_auc"] = None; out["pr_auc"] = None
        out["brier"] = None; out["ece_10bin"] = None
    out["f1_lift"] = f1 - base["best_baseline_f1"]
    out["accuracy_lift"] = acc - base["best_baseline_accuracy"]
    out["precision_lift"] = prec - base["prevalence"]
    out["balanced_accuracy_lift"] = bacc - 0.5
    out["roc_auc_lift"] = (out["roc_auc"] - 0.5) if out["roc_auc"] is not None else None
    out.update(base)
    return out
