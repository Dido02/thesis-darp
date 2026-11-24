#!/usr/bin/env python3
"""K-fold stratified-by-(size,delay) cross-validation for a subset of models.

For each (target, model_config) and each of K folds we:
  - Train on the fold's train logs
  - Evaluate on the fold's test logs
  - Capture per-pair metrics + lifts + baselines
  - Capture structural artifacts: feature_importances_, root-split feature/threshold,
    top-3 BFS splits (trees), tree depth, n_leaves

Output: cv_raw_results.json (everything aggregated for downstream analysis).

Usage:
    python 01_cv_train.py                          # default subset (t07 + t05 + regr)
    python 01_cv_train.py --targets t07            # restrict
    python 01_cv_train.py --K 5                    # 5-fold (default)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml_train"))

from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES_PQ = CACHE / "features.parquet"
OUT_DIR = REPO / "model_diagnostics_cv_2026-05-19"
TBL = OUT_DIR / "tables"
TBL.mkdir(parents=True, exist_ok=True)

# USED 9 pairs: destroy in {0,1,2} x repair in {0,1,2}
USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]

# Targets: only classifiers in this initial run (the user asked about lifts
# which are classifier-flavoured; regr would be a separate metric set).
TARGETS = {
    "t07": {"labels": "labels_t0.7.parquet"},
    "t05": {"labels": "labels_t0.5.parquet"},
    "t03": {"labels": "labels_t0.3.parquet"},
    "t02": {"labels": "labels_t0.2.parquet"},
}

# Model subset per target. (label, model_type, depth, K)
# K = features used. K=20 → all 20; K<20 → top-K from v3_full_20f_<t> forest aggregate
# importance across USED pairs.
SUBSET = {
    "t07": [
        ("forest_baseline_d_none_K20",  "forest", None, 20),
        ("forest_grid_d8_K20",          "forest", 8,    20),
        ("tree_grid_d4_K12",            "tree",   4,    12),
        ("tree_grid_d8_K8",             "tree",   8,    8),
    ],
    "t05": [
        ("forest_baseline_d_none_K20",  "forest", None, 20),
        ("forest_grid_d8_K8",           "forest", 8,    8),
    ],
    "t03": [
        ("forest_baseline_d_none_K20",  "forest", None, 20),
        ("forest_grid_d8_K20",          "forest", 8,    20),
    ],
    "t02": [
        ("forest_baseline_d_none_K20",  "forest", None, 20),
        ("forest_grid_d4_K20",          "forest", 4,    20),
    ],
}


def parse_log(name: str):
    """Return (size, delay) bucket key for stratification. None for unparseable."""
    # manhattan_30_s_delay03_min_s1024_t50e100_dynamic.jsonl
    m = re.match(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+", name)
    if not m:
        return None
    return (m.group("size"), m.group("delay"))


def make_stratified_folds(log_files: list[str], k: int = 5, seed: int = 42):
    """Stratified by (size, delay). Each stratum's logs are distributed
    round-robin across the K folds so every fold gets the same (size, delay)
    coverage as much as possible.
    """
    strata = defaultdict(list)
    for n in log_files:
        key = parse_log(n) or ("other", "other")
        strata[key].append(n)

    rng = np.random.RandomState(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    # Global round-robin: keep load balanced across folds even when each stratum
    # is small. Without this, strata of size 1 all pile into fold 0.
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


def aggregated_top_k(reference_dir: Path, K: int, used_pairs: list[str]) -> list[str]:
    """Top-K features by mean Gini importance across the USED pairs."""
    meta = json.loads((reference_dir / "meta.json").read_text())
    feat_names = meta["feature_names"]
    if K >= len(feat_names):
        return feat_names
    imps = np.zeros(len(feat_names))
    for pair in used_pairs:
        # Reference dir was deleted post-cleanup if it lives under model_diagnostics_*/
        # but ml_train/models_v3_full_20f_*/ are intact.
        clf_path = reference_dir / f"rf_{pair}.joblib"
        if not clf_path.exists():
            raise FileNotFoundError(f"Missing reference: {clf_path}")
        clf = joblib.load(clf_path)
        imps += clf.feature_importances_
    imps /= len(used_pairs)
    order = np.argsort(-imps)
    return [feat_names[i] for i in order[:K]]


def compute_baselines(y_test) -> dict:
    n = len(y_test); pos = int(y_test.sum())
    prev = pos / n if n else 0.0
    a0 = (n - pos) / n if n else 0.0
    a1 = pos / n if n else 0.0
    f1_a0 = 0.0  # no TP
    f1_a1 = (2 * prev / (1 + prev)) if prev > 0 else 0.0
    return {
        "n": int(n), "n_positive": pos, "prevalence": prev,
        "always0_accuracy": a0, "always1_accuracy": a1,
        "best_baseline_accuracy": max(a0, a1),
        "always0_f1": f1_a0, "always1_f1": f1_a1,
        "best_baseline_f1": max(f1_a0, f1_a1),
    }


def compute_clf_metrics(clf, X_test, y_test) -> dict:
    if len(y_test) == 0:
        return {"empty": True}
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    bacc = (rec + specificity) / 2

    base = compute_baselines(y_test)

    out = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": specificity, "f1": f1, "balanced_accuracy": bacc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_test)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            out["pr_auc"] = float(average_precision_score(y_test, y_proba))
            out["brier"] = float(brier_score_loss(y_test, y_proba))
        except Exception:
            out["roc_auc"] = None; out["pr_auc"] = None; out["brier"] = None
    else:
        out["roc_auc"] = None; out["pr_auc"] = None; out["brier"] = None
    try:
        out["mcc"] = float(matthews_corrcoef(y_test, y_pred))
    except Exception:
        out["mcc"] = None

    # Lifts
    out["f1_lift"] = f1 - base["best_baseline_f1"]
    out["accuracy_lift"] = acc - base["best_baseline_accuracy"]
    out["precision_lift"] = prec - base["prevalence"]
    out["balanced_accuracy_lift"] = bacc - 0.5
    if out["roc_auc"] is not None:
        out["roc_auc_lift"] = out["roc_auc"] - 0.5
    else:
        out["roc_auc_lift"] = None

    out.update(base)
    return out


def tree_structural_artifacts(tree, feature_names: list[str]) -> dict:
    """Capture root split, top-3 BFS splits, depth, n_leaves, feature_importances."""
    t = tree.tree_
    importances = tree.feature_importances_.tolist()

    bfs_splits = []
    queue = [0]
    while queue and len(bfs_splits) < 3:
        node = queue.pop(0)
        if t.children_left[node] == -1:
            continue
        f_idx = int(t.feature[node])
        thr = float(t.threshold[node])
        bfs_splits.append({
            "feature_idx": f_idx,
            "feature_name": feature_names[f_idx],
            "threshold": thr,
            "samples_at_node": int(t.n_node_samples[node]),
        })
        queue.append(int(t.children_left[node]))
        queue.append(int(t.children_right[node]))

    return {
        "kind": "tree",
        "max_depth": int(tree.get_depth()),
        "n_leaves": int(t.n_leaves),
        "node_count": int(t.node_count),
        "feature_importances": importances,
        "top_3_bfs_splits": bfs_splits,
    }


def forest_structural_artifacts(forest, feature_names: list[str]) -> dict:
    """Forest: feature_importances_ + per-estimator structure aggregates."""
    importances = forest.feature_importances_.tolist()
    depths = [t.get_depth() for t in forest.estimators_]
    leaves = [t.tree_.n_leaves for t in forest.estimators_]
    # Aggregate root split: which feature is most often picked as root across the 100 trees
    root_features = []
    for t in forest.estimators_:
        if t.tree_.feature[0] >= 0:
            root_features.append(int(t.tree_.feature[0]))
    from collections import Counter
    root_counter = Counter(root_features)
    most_common = root_counter.most_common(3)
    return {
        "kind": "forest",
        "median_depth": float(np.median(depths)),
        "median_n_leaves": float(np.median(leaves)),
        "feature_importances": importances,
        "top_3_root_features": [
            {"feature_idx": fi, "feature_name": feature_names[fi],
             "n_estimators_with_this_root": cnt}
            for fi, cnt in most_common
        ],
    }


def train_and_eval(model_type: str, depth, X_train, y_train, X_test, y_test,
                   feature_names, pair_name: str, seed: int):
    if model_type == "tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed,
                                     max_depth=depth)
    else:
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                     random_state=seed, n_jobs=-1, max_depth=depth)
    clf.fit(X_train, y_train)
    metrics = compute_clf_metrics(clf, X_test, y_test)
    train_metrics = compute_clf_metrics(clf, X_train, y_train)
    metrics = {f"test_{k}": v for k, v in metrics.items()}
    metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
    if model_type == "tree":
        struct = tree_structural_artifacts(clf, feature_names)
    else:
        struct = forest_structural_artifacts(clf, feature_names)
    return metrics, struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(SUBSET))
    ap.add_argument("--K", type=int, default=5, help="Number of CV folds.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading features.parquet...")
    features_df = pd.read_parquet(FEATURES_PQ)
    print(f"  features: {features_df.shape}")

    # Build folds (shared across all targets so we report on the same splits)
    log_files = sorted(features_df["log_file"].unique())
    folds = make_stratified_folds(log_files, k=args.K, seed=args.seed)
    print(f"Folds (k={args.K}, stratified by size×delay):")
    for f in folds:
        print(f"  fold {f['fold']}: train={len(f['train_logs'])} test={len(f['test_logs'])}")

    # Pre-resolve top-K feature subsets per target (using the ORIGINAL v3 forest)
    feat_subsets = {}
    for target in args.targets:
        if target not in SUBSET: continue
        ref_dir = REPO / "ml_train" / f"models_v3_full_20f_{target}"
        for label, _, _, K in SUBSET[target]:
            if (target, K) in feat_subsets: continue
            feat_subsets[(target, K)] = aggregated_top_k(ref_dir, K, USED_PAIRS)

    results = {
        "config": {"K": args.K, "seed": args.seed,
                   "used_pairs": USED_PAIRS,
                   "n_logs": len(log_files)},
        "folds_meta": [{"fold": f["fold"],
                        "n_train_logs": len(f["train_logs"]),
                        "n_test_logs": len(f["test_logs"]),
                        "test_logs": f["test_logs"]} for f in folds],
        "feat_subsets": {f"{t}_K{K}": names for (t, K), names in feat_subsets.items()},
        "results": [],  # list of {target, model_label, fold, pair_metrics, pair_structural}
    }

    t_start = time.monotonic()
    for target in args.targets:
        if target not in SUBSET:
            print(f"[skip] {target}: not in SUBSET")
            continue
        tgt = TARGETS[target]
        labels_pq = CACHE / tgt["labels"]
        labels_df = pd.read_parquet(labels_pq)
        df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
        pair_cols = sorted(c for c in labels_df.columns if c.startswith("pair_"))

        for model_label, model_type, depth, K in SUBSET[target]:
            feat_names = feat_subsets[(target, K)]
            for fold in folds:
                t0 = time.monotonic()
                train_mask = df["log_file"].isin(fold["train_logs"]).values
                test_mask = df["log_file"].isin(fold["test_logs"]).values
                X_all = df[feat_names].values
                X_train = X_all[train_mask]
                X_test = X_all[test_mask]

                fold_pair_metrics = {}
                fold_pair_structural = {}
                for pair in pair_cols:
                    y = df[pair].values
                    y_train = y[train_mask]
                    y_test = y[test_mask]
                    if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
                        # Degenerate within this fold's train — skip
                        fold_pair_metrics[pair] = {"degenerate_train": True,
                                                    "test_n": int(test_mask.sum())}
                        continue
                    metrics, struct = train_and_eval(
                        model_type, depth, X_train, y_train, X_test, y_test,
                        feat_names, pair, args.seed)
                    fold_pair_metrics[pair] = metrics
                    fold_pair_structural[pair] = struct
                wall = time.monotonic() - t0
                print(f"  {target} / {model_label} / fold {fold['fold']} ({wall:.1f}s)")

                results["results"].append({
                    "target": target,
                    "model_label": model_label,
                    "model_type": model_type,
                    "depth": "none" if depth is None else int(depth),
                    "K": K,
                    "fold": fold["fold"],
                    "n_features": len(feat_names),
                    "feature_names": feat_names,
                    "pair_metrics": fold_pair_metrics,
                    "pair_structural": fold_pair_structural,
                })

    out_path = TBL / "cv_raw_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nTotal wall: {time.monotonic() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
