"""Shared helpers for the t07 grid sweep."""
from __future__ import annotations
import math, re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES_PQ = CACHE / "features.parquet"
LABELS_PQ = CACHE / "labels_t0.7.parquet"
OUT_DIR = REPO / "model_diagnostics_t07_sweep_2026-05-20"
TBL = OUT_DIR / "tables"
FIG = OUT_DIR / "figures"

USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]

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
    m = re.match(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+", name)
    return (m.group("size"), m.group("delay")) if m else None


def make_stratified_folds(log_files, k=5, seed=42):
    """Stratified-by-(size,delay) round-robin within strata across K folds.
    Mirrors model_diagnostics_cv_2026-05-19/scripts/01_cv_train.py.
    """
    strata = defaultdict(list)
    for n in log_files:
        key = parse_log(n) or ("other", "other")
        strata[key].append(n)
    rng = np.random.RandomState(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    next_fold = 0
    for key in sorted(strata):
        logs = list(strata[key])
        rng.shuffle(logs)
        for log in logs:
            folds[next_fold % k].append(log)
            next_fold += 1
    splits = []
    for i in range(k):
        test_set = set(folds[i])
        train_set = set()
        for j in range(k):
            if j != i:
                train_set.update(folds[j])
        splits.append({"fold": i, "test_logs": sorted(test_set),
                       "train_logs": sorted(train_set)})
    return splits


def mcc_cm(tp, fp, fn, tn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def aggregated_top_k(K: int) -> list[str]:
    """Top-K features by aggregate Gini importance across the 9 USED pairs
    in the deployed forest. Matches the CV pipeline's feature-subset method.
    """
    ref_dir = REPO / "ml_train" / "models_v3_full_20f_t07"
    if K >= len(FEATURE_NAMES_20):
        return FEATURE_NAMES_20
    imps = np.zeros(len(FEATURE_NAMES_20))
    for pair in USED_PAIRS:
        clf = joblib.load(ref_dir / f"rf_{pair}.joblib")
        imps += clf.feature_importances_
    imps /= len(USED_PAIRS)
    order = np.argsort(-imps)
    return [FEATURE_NAMES_20[i] for i in order[:K]]
