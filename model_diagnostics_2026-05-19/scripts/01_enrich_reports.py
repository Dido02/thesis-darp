#!/usr/bin/env python3
"""Phase A — enrich each existing training_report.json with train-set metrics,
ROC/PR/Brier (classifiers), RMSE (regressors), and tree-structure stats.

Reads from ml_train/models_v{3,6}_full_*/ (read-only), writes one sidecar
JSON per directory to model_diagnostics_2026-05-19/enriched_reports/. The
original training_report.json files are never modified.

For each model dir we:
  1. Load meta.json -> feature_names, model_kind, target_metric, log_transform.
  2. Load training_report.json -> per-pair test metrics + train/test log split.
  3. Pick the correct labels parquet based on dir name (classifier threshold)
     or model_kind (regressor uses _t0.3_with_ei).
  4. Load features.parquet, mask by train/test logs.
  5. For each pair: load joblib, predict on TRAIN and TEST -> richer metrics.
  6. Extract tree-structure stats (depth, n_leaves, leaf sample distribution,
     unique features used, per-feature split count).
     Forest variants get min/median/max aggregates across the 100 trees.
  7. Write enriched_reports/<dir_name>.json.

Usage:
    python 01_enrich_reports.py
    python 01_enrich_reports.py --target-dir <single_dir>     # one dir only
    python 01_enrich_reports.py --root path/to/parents...      # custom search root
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES_PARQUET = CACHE / "features.parquet"
DEFAULT_OUT_DIR = REPO / "model_diagnostics_2026-05-19" / "enriched_reports"
OUT_DIR = DEFAULT_OUT_DIR   # mutated in main() when --out-dir is passed

# Map classifier threshold suffix -> labels parquet (binary pair_* columns)
_CLF_LABELS = {
    "_t02": "labels_t0.2.parquet",
    "_t03": "labels_t0.3.parquet",
    "_t05": "labels_t0.5.parquet",
    "_t07": "labels_t0.7.parquet",
}
# Regressors are trained on labels_t0.3_with_ei.parquet (has improve_freq_* and ei_pct_*).
_REGR_LABELS = "labels_t0.3_with_ei.parquet"


def resolve_labels_path(dir_name: str, model_kind: str) -> Path:
    if model_kind == "regressor":
        return CACHE / _REGR_LABELS
    for suffix, lbl in _CLF_LABELS.items():
        # Match suffix anywhere as a token boundary: _t07 / _t07_ / _t07_tree
        if (dir_name.endswith(suffix)
                or dir_name.endswith(suffix + "_tree")
                or (suffix + "_") in dir_name):
            return CACHE / lbl
    raise ValueError(f"Cannot infer classifier threshold from dir name: {dir_name}")


def tree_structure_stats(tree) -> dict:
    """Per-tree structure stats from a sklearn DecisionTree* (or estimator inside a forest)."""
    t = tree.tree_
    is_leaf = t.children_left == -1
    leaf_samples = t.n_node_samples[is_leaf]
    feature_use_count = np.zeros(t.n_features, dtype=int)
    for f in t.feature[~is_leaf]:
        feature_use_count[f] += 1
    return {
        "max_depth": int(tree.get_depth()),
        "node_count": int(t.node_count),
        "n_leaves": int(is_leaf.sum()),
        "leaf_samples": {
            "min": int(leaf_samples.min()),
            "p25": float(np.percentile(leaf_samples, 25)),
            "median": float(np.median(leaf_samples)),
            "p75": float(np.percentile(leaf_samples, 75)),
            "p90": float(np.percentile(leaf_samples, 90)),
            "max": int(leaf_samples.max()),
            "frac_lt_5": float(np.mean(leaf_samples < 5)),
        },
        "n_unique_features_used": int((feature_use_count > 0).sum()),
        "feature_use_count": feature_use_count.tolist(),
    }


def forest_structure_stats(forest) -> dict:
    """Forest aggregate: min/median/max over the 100 estimator trees."""
    per_tree = [tree_structure_stats(t) for t in forest.estimators_]
    if not per_tree:
        return {}
    sum_use = np.zeros(forest.n_features_in_, dtype=int)
    for s in per_tree:
        sum_use += np.array(s["feature_use_count"])
    depths = [s["max_depth"] for s in per_tree]
    leaves = [s["n_leaves"] for s in per_tree]
    nodes = [s["node_count"] for s in per_tree]
    feats_used = [s["n_unique_features_used"] for s in per_tree]
    return {
        "n_estimators": len(per_tree),
        "max_depth": {
            "min": int(np.min(depths)),
            "median": float(np.median(depths)),
            "max": int(np.max(depths)),
        },
        "n_leaves": {
            "min": int(np.min(leaves)),
            "median": float(np.median(leaves)),
            "max": int(np.max(leaves)),
        },
        "node_count": {
            "min": int(np.min(nodes)),
            "median": float(np.median(nodes)),
            "max": int(np.max(nodes)),
        },
        "n_unique_features_used": {
            "min": int(np.min(feats_used)),
            "median": float(np.median(feats_used)),
            "max": int(np.max(feats_used)),
        },
        "feature_use_count_total": sum_use.tolist(),
    }


def _classifier_metrics(clf, X, y, prefix: str, feature_use_in_clf: bool) -> dict:
    """Compute classifier metrics on (X, y). Prefix is 'train' or 'test'."""
    y_pred = clf.predict(X)
    # ROC/PR/Brier need probabilities; some pathological trees emit prob=1.0 for both classes.
    try:
        y_proba = clf.predict_proba(X)[:, 1]
    except (AttributeError, IndexError):
        y_proba = y_pred.astype(float)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y, y_pred, zero_division=0))
    y_ones = np.ones_like(y)
    y_zeros = np.zeros_like(y)
    always1_f1 = float(f1_score(y, y_ones, zero_division=0))
    always0_f1 = float(f1_score(y, y_zeros, zero_division=0))
    # ROC-AUC requires both classes present in y
    roc_auc = float("nan")
    pr_auc = float("nan")
    brier = float("nan")
    if len(np.unique(y)) > 1:
        try:
            roc_auc = float(roc_auc_score(y, y_proba))
            pr_auc = float(average_precision_score(y, y_proba))
            brier = float(brier_score_loss(y, y_proba))
        except Exception:
            pass
    try:
        mcc = float(matthews_corrcoef(y, y_pred))
    except Exception:
        mcc = float("nan")
    return {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_n_positive": int(int(np.sum(y))),
        f"{prefix}_accuracy": float(accuracy_score(y, y_pred)),
        f"{prefix}_precision": float(precision_score(y, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, y_pred, zero_division=0)),
        f"{prefix}_specificity": specificity,
        f"{prefix}_f1": f1,
        f"{prefix}_always1_f1": always1_f1,
        f"{prefix}_always0_f1": always0_f1,
        f"{prefix}_f1_lift": f1 - max(always1_f1, always0_f1),
        f"{prefix}_mcc": mcc,
        f"{prefix}_roc_auc": roc_auc,
        f"{prefix}_pr_auc": pr_auc,
        f"{prefix}_brier": brier,
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
    }


def _regressor_metrics(reg, X, y, prefix: str, log_transform: bool,
                      transform_params: dict | None) -> dict:
    """Compute regressor metrics. If log_transform, invert before metricizing."""
    y_pred = reg.predict(X)
    if log_transform and transform_params is not None:
        log_min = transform_params["log_min"]
        log_max = transform_params["log_max"]
        span = max(log_max - log_min, 1e-12)
        y_pred = np.exp(y_pred * span + log_min) - 1e-6
        y_pred = np.clip(y_pred, 0.0, None)
    resid = y - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    baseline = float(y.mean())  # baseline = constant mean on this split
    baseline_mae = float(np.mean(np.abs(y - baseline)))
    return {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_baseline_mae": baseline_mae,
        f"{prefix}_mae_lift": baseline_mae - mae,
        f"{prefix}_label_mean": float(y.mean()),
        f"{prefix}_label_std": float(y.std()),
        f"{prefix}_residual_std": float(resid.std()),
    }


def enrich_one_dir(model_dir: Path, features_df: pd.DataFrame) -> dict:
    """Compute the enriched report for a single model directory."""
    meta = json.loads((model_dir / "meta.json").read_text())
    report = json.loads((model_dir / "training_report.json").read_text())

    feature_names = meta["feature_names"]
    model_kind = meta["model_kind"]
    target_metric = meta.get("target_metric", "improve_freq")
    log_transform = meta.get("log_transform_label", False)
    label_transforms = meta.get("label_transforms_per_pair", {})
    model_type = meta.get("model_type", "forest")

    labels_path = resolve_labels_path(model_dir.name, model_kind)
    labels_df = pd.read_parquet(labels_path)
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")

    train_logs = set(report["split"]["train_logs"])
    test_logs = set(report["split"]["test_logs"])
    train_mask = df["log_file"].isin(train_logs)
    test_mask = df["log_file"].isin(test_logs)

    X_full = df[feature_names].values
    X_train = X_full[train_mask.values]
    X_test = X_full[test_mask.values]

    label_cols = sorted(c for c in labels_df.columns if c.startswith("pair_"))
    target_prefix = "ei_pct_" if target_metric == "ei_pct" else "improve_freq_"

    enriched_pairs: list[dict] = []
    for pair_col in label_cols:
        joblib_path = model_dir / f"rf_{pair_col}.joblib"
        if not joblib_path.exists():
            continue
        clf = joblib.load(joblib_path)

        if model_kind == "classifier":
            y_full = df[pair_col].values
            y_train = y_full[train_mask.values]
            y_test = y_full[test_mask.values]
            metrics = {"pair_name": pair_col}
            metrics.update(_classifier_metrics(clf, X_train, y_train, "train", True))
            metrics.update(_classifier_metrics(clf, X_test, y_test, "test", True))
        else:
            target_col = pair_col.replace("pair_", target_prefix)
            y_full = df[target_col].values
            y_train = y_full[train_mask.values]
            y_test = y_full[test_mask.values]
            tparams = label_transforms.get(pair_col) if log_transform else None
            metrics = {"pair_name": pair_col}
            metrics.update(_regressor_metrics(clf, X_train, y_train, "train",
                                              log_transform, tparams))
            metrics.update(_regressor_metrics(clf, X_test, y_test, "test",
                                              log_transform, tparams))

        # Tree-structure stats
        if model_type == "tree":
            metrics["tree_structure"] = tree_structure_stats(clf)
        else:
            metrics["forest_structure"] = forest_structure_stats(clf)

        enriched_pairs.append(metrics)

    return {
        "model_dir": str(model_dir.relative_to(REPO)),
        "model_kind": model_kind,
        "model_type": model_type,
        "target_metric": target_metric,
        "log_transform_label": log_transform,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "batches": meta.get("batches", []),
        "labels_parquet": str(labels_path.relative_to(REPO)),
        "n_train_logs": len(train_logs),
        "n_test_logs": len(test_logs),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "pairs": enriched_pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="Parent dir to scan for models_*/ subdirs. Repeatable. "
                         "Defaults to ml_train/, model_diagnostics_2026-05-19/depth_sweep_models/, "
                         "and model_diagnostics_2026-05-19/feat_ablation_models/.")
    ap.add_argument("--target-dir", default=None,
                    help="Run on a single model dir (path relative to repo or absolute).")
    ap.add_argument("--pattern", default="models_v*_full_*",
                    help="Glob within each root for model dirs (default: models_v*_full_*).")
    ap.add_argument("--out-dir", default=None,
                    help="Override the output enriched_reports/ directory.")
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
        if not OUT_DIR.is_absolute():
            OUT_DIR = (REPO / OUT_DIR).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    if args.target_dir is not None:
        target = Path(args.target_dir)
        if not target.is_absolute():
            target = REPO / target
        model_dirs = [target]
    else:
        roots = args.root or [
            REPO / "ml_train",
            REPO / "model_diagnostics_2026-05-19" / "depth_sweep_models",
            REPO / "model_diagnostics_2026-05-19" / "feat_ablation_models",
        ]
        model_dirs = []
        for r in roots:
            r = Path(r)
            if not r.is_absolute():
                r = (REPO / r).resolve()
            if not r.exists():
                continue
            for d in sorted(r.iterdir()):
                if not d.is_dir():
                    continue
                # accept either the canonical models_v*_full_* pattern or any
                # subdir under depth_sweep_models / feat_ablation_models that
                # has a training_report.json.
                if (d / "training_report.json").exists() and (d / "meta.json").exists():
                    model_dirs.append(d)

    print(f"Loading features.parquet from {FEATURES_PARQUET}")
    features_df = pd.read_parquet(FEATURES_PARQUET)
    print(f"  features: {features_df.shape}")

    print(f"\nEnriching {len(model_dirs)} model directories...\n")
    summary = []
    for i, d in enumerate(model_dirs, 1):
        try:
            enriched = enrich_one_dir(d, features_df)
        except Exception as e:
            print(f"  [{i:>3}/{len(model_dirs)}] {d.name}: FAILED ({e})")
            continue
        out_path = OUT_DIR / f"{d.name}.json"
        with open(out_path, "w") as f:
            json.dump(enriched, f, indent=2)
        # Quick eyeball summary line
        if enriched["pairs"]:
            sample = enriched["pairs"][0]
            if enriched["model_kind"] == "classifier":
                train_f1 = sample.get("train_f1", float("nan"))
                test_f1 = sample.get("test_f1", float("nan"))
                gap = train_f1 - test_f1
                tag = f"f1_train={train_f1:.3f} f1_test={test_f1:.3f} gap={gap:+.3f}"
            else:
                train_mae_lift = sample.get("train_mae_lift", float("nan"))
                test_mae_lift = sample.get("test_mae_lift", float("nan"))
                tag = (f"mae_lift_train={train_mae_lift:+.4f} "
                       f"mae_lift_test={test_mae_lift:+.4f}")
            print(f"  [{i:>3}/{len(model_dirs)}] {d.name:<48}  pair_0_0: {tag}")
        summary.append(d.name)

    print(f"\nWrote {len(summary)} enriched reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
