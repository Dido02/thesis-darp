#!/usr/bin/env python3
"""Leave-One-Instance-Out training across the 9 USED pairs.

For each (target, model_config, LOIO_fold, pair):
  - Train on the (size, delay) train instances' rows
  - Evaluate on the held-out instance's rows
  - Save fitted forest + per-state probabilities (for phase 3 join)
  - Capture forest structural artifacts

Outputs:
  tables/loio_raw_results.json
  fold_models/{target}/{model_label}/fold{idx:02d}_{instance}/forest_{pair}.joblib
  fold_models/{target}/{model_label}/fold{idx:02d}_{instance}/state_proba.parquet

Usage:
  python 01_loio_train.py
  python 01_loio_train.py --targets t03 t07
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FOLD_MODELS, MODEL_CONFIGS, OUT_DIR, TARGETS,
    TBL, USED_PAIRS, compute_clf_metrics, make_loio_folds,
)


def forest_structural_artifacts(forest, feature_names: list[str]) -> dict:
    importances = forest.feature_importances_.tolist()
    depths = [t.get_depth() for t in forest.estimators_]
    leaves = [t.tree_.n_leaves for t in forest.estimators_]
    root_features = []
    for t in forest.estimators_:
        if t.tree_.feature[0] >= 0:
            root_features.append(int(t.tree_.feature[0]))
    counter = Counter(root_features)
    most_common = counter.most_common(3)
    return {
        "median_depth": float(np.median(depths)),
        "median_n_leaves": float(np.median(leaves)),
        "feature_importances": importances,
        "top_3_root_features": [
            {"feature_idx": fi, "feature_name": feature_names[fi],
             "n_estimators_with_this_root": cnt}
            for fi, cnt in most_common
        ],
    }


def train_one_pair(X_train, y_train, X_test, y_test, max_depth):
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=max_depth,
    )
    clf.fit(X_train, y_train)
    test_metrics = compute_clf_metrics(clf, X_test, y_test)
    train_metrics = compute_clf_metrics(clf, X_train, y_train)
    test_proba = clf.predict_proba(X_test)[:, 1]
    return clf, test_metrics, train_metrics, test_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    args = ap.parse_args()

    TBL.mkdir(parents=True, exist_ok=True)
    FOLD_MODELS.mkdir(parents=True, exist_ok=True)

    print(f"Loading features.parquet...")
    features_df = pd.read_parquet(CACHE / "features.parquet")
    print(f"  features: {features_df.shape}")

    log_files = sorted(features_df["log_file"].unique())
    folds = make_loio_folds(log_files)
    print(f"LOIO: {len(folds)} folds (one per (size, delay) instance)")
    for f in folds:
        print(f"  fold {f['fold']:02d} {f['instance']:<14} "
              f"train_logs={len(f['train_logs'])} test_logs={len(f['test_logs'])}")

    results = {
        "config": {
            "protocol": "leave-one-instance-out",
            "used_pairs": USED_PAIRS,
            "feature_names": FEATURE_NAMES_20,
            "model_configs": [{"label": lbl, "max_depth": d}
                              for lbl, d in MODEL_CONFIGS],
            "targets": args.targets,
            "n_logs": len(log_files),
            "n_folds": len(folds),
        },
        "folds_meta": [{
            "fold": f["fold"], "instance": f["instance"],
            "n_train_logs": len(f["train_logs"]),
            "n_test_logs": len(f["test_logs"]),
            "test_logs": f["test_logs"],
        } for f in folds],
        "results": [],
    }

    t_start = time.monotonic()
    for target in args.targets:
        labels_df = pd.read_parquet(CACHE / TARGETS[target])
        df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
        X_all = df[FEATURE_NAMES_20].values
        log_arr = df["log_file"].values
        iter_arr = df["iter"].values

        for model_label, max_depth in MODEL_CONFIGS:
            for fold in folds:
                t0 = time.monotonic()
                train_mask = np.isin(log_arr, fold["train_logs"])
                test_mask = np.isin(log_arr, fold["test_logs"])

                if test_mask.sum() == 0:
                    # Test logs exist in features but were dropped by labels filtering
                    # (epoch_1 / uninformative). Record + skip.
                    print(f"  [skip] {target} / {model_label} / fold "
                          f"{fold['fold']:02d} {fold['instance']:<14} "
                          f"— 0 test rows after labels merge")
                    results["results"].append({
                        "target": target, "model_label": model_label,
                        "max_depth": "none" if max_depth is None else int(max_depth),
                        "fold": fold["fold"], "instance": fold["instance"],
                        "n_train_logs": len(fold["train_logs"]),
                        "n_test_logs": len(fold["test_logs"]),
                        "degenerate_test_fold": True,
                        "pair_metrics": {}, "pair_structural": {},
                    })
                    continue
                if train_mask.sum() == 0:
                    print(f"  [skip] {target} / {model_label} / fold "
                          f"{fold['fold']:02d} {fold['instance']:<14} "
                          f"— 0 train rows after labels merge")
                    results["results"].append({
                        "target": target, "model_label": model_label,
                        "max_depth": "none" if max_depth is None else int(max_depth),
                        "fold": fold["fold"], "instance": fold["instance"],
                        "n_train_logs": len(fold["train_logs"]),
                        "n_test_logs": len(fold["test_logs"]),
                        "degenerate_train_fold": True,
                        "pair_metrics": {}, "pair_structural": {},
                    })
                    continue

                X_train = X_all[train_mask]
                X_test = X_all[test_mask]
                test_log_files = log_arr[test_mask]
                test_iters = iter_arr[test_mask]

                fold_dir = (FOLD_MODELS / target / model_label /
                            f"fold{fold['fold']:02d}_{fold['instance']}")
                fold_dir.mkdir(parents=True, exist_ok=True)

                fold_pair_metrics = {}
                fold_pair_structural = {}
                proba_dict = {
                    "log_file": test_log_files,
                    "iter": test_iters,
                }
                for pair in USED_PAIRS:
                    y = df[pair].values
                    y_train = y[train_mask]
                    y_test = y[test_mask]
                    if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
                        # Use train prevalence as a baseline proba
                        prev = float(y_train.mean()) if len(y_train) else 0.0
                        fold_pair_metrics[pair] = {
                            "degenerate_train": True,
                            "train_prevalence": prev,
                            "test_n": int(test_mask.sum()),
                            "test_n_positive": int(y_test.sum()),
                        }
                        proba_dict[f"proba_{pair}"] = np.full(
                            test_mask.sum(), prev, dtype=np.float32)
                        continue
                    clf, test_m, train_m, test_proba = train_one_pair(
                        X_train, y_train, X_test, y_test, max_depth)
                    metrics = {f"test_{k}": v for k, v in test_m.items()}
                    metrics.update({f"train_{k}": v for k, v in train_m.items()})
                    fold_pair_metrics[pair] = metrics
                    fold_pair_structural[pair] = forest_structural_artifacts(
                        clf, FEATURE_NAMES_20)
                    joblib.dump(clf, fold_dir / f"forest_{pair}.joblib",
                                compress=3)
                    proba_dict[f"proba_{pair}"] = test_proba.astype(np.float32)

                # Dump per-state probas
                pd.DataFrame(proba_dict).to_parquet(
                    fold_dir / "state_proba.parquet", index=False)

                wall = time.monotonic() - t0
                print(f"  {target} / {model_label} / fold {fold['fold']:02d} "
                      f"{fold['instance']:<14} ({wall:.1f}s)")

                results["results"].append({
                    "target": target,
                    "model_label": model_label,
                    "max_depth": "none" if max_depth is None else int(max_depth),
                    "fold": fold["fold"],
                    "instance": fold["instance"],
                    "n_train_logs": len(fold["train_logs"]),
                    "n_test_logs": len(fold["test_logs"]),
                    "pair_metrics": fold_pair_metrics,
                    "pair_structural": fold_pair_structural,
                })

    out_path = TBL / "loio_raw_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nTotal wall: {time.monotonic() - t_start:.1f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
