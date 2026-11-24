#!/usr/bin/env python3
"""LOIO evaluation of the best single tree (depth=4, K=4) at t07.

Runs 14 LOIO folds (one per (size, delay) instance with labelled rows),
trains a depth=4 K=4 tree per USED pair per fold, captures full train/test
metrics + per-pair structural artifacts.

Output: tables/loio_tree_d4_K4_raw.json
"""
from __future__ import annotations
import json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURES_PQ, LABELS_PQ, TBL, USED_PAIRS, aggregated_top_k, parse_log,
    mcc_cm,
)


def make_loio_folds(log_files):
    strata = defaultdict(list)
    for n in log_files:
        key = parse_log(n) or ("other", "other")
        strata[key].append(n)
    all_set = set(log_files)
    splits = []
    for i, key in enumerate(sorted(strata)):
        test_logs = sorted(strata[key])
        train_logs = sorted(all_set - set(test_logs))
        splits.append({"fold": i, "instance": f"{key[0]}_d{key[1]}",
                       "test_logs": test_logs, "train_logs": train_logs})
    return splits


def metric_bundle(clf, X, y):
    if len(y) == 0:
        return {"empty": True}
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    out = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "specificity": spec,
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "mcc": float(mcc_cm(tp, fp, fn, tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)), "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
    }
    if len(np.unique(y)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y, proba))
            out["pr_auc"] = float(average_precision_score(y, proba))
            out["brier"] = float(brier_score_loss(y, proba))
        except Exception:
            out["roc_auc"] = out["pr_auc"] = out["brier"] = None
    else:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = None
    return out


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PQ).merge(
        pd.read_parquet(LABELS_PQ), on=["log_file", "iter"], how="inner")
    feat_names = aggregated_top_k(4)
    print(f"Top-4 features: {feat_names}")

    log_files = sorted(df["log_file"].unique())
    folds = make_loio_folds(log_files)
    print(f"LOIO: {len(folds)} folds")

    X_all = df[feat_names].values
    log_arr = df["log_file"].values
    results = {
        "config": {"tag": "tree_d4_K4", "model_type": "tree", "max_depth": 4,
                   "K": 4, "n_folds": len(folds), "feature_names": feat_names},
        "folds_meta": [{"fold": f["fold"], "instance": f["instance"],
                        "n_train_logs": len(f["train_logs"]),
                        "n_test_logs": len(f["test_logs"])} for f in folds],
        "results": [],
    }
    t_start = time.monotonic()
    for fold in folds:
        train_mask = np.isin(log_arr, fold["train_logs"])
        test_mask = np.isin(log_arr, fold["test_logs"])
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            print(f"  [skip] fold {fold['fold']:02d} {fold['instance']:<14} 0 train/test rows")
            results["results"].append({"fold": fold["fold"], "instance": fold["instance"],
                                       "degenerate_test_fold": True, "pair_metrics": {}})
            continue
        X_train = X_all[train_mask]; X_test = X_all[test_mask]
        fold_pair = {}
        for pair in USED_PAIRS:
            y = df[pair].values
            y_train, y_test = y[train_mask], y[test_mask]
            if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
                fold_pair[pair] = {"degenerate_train": True,
                                   "test_n": int(test_mask.sum()),
                                   "test_n_positive": int(y_test.sum())}
                continue
            clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
            clf.fit(X_train, y_train)
            m_train = metric_bundle(clf, X_train, y_train)
            m_test = metric_bundle(clf, X_test, y_test)
            fold_pair[pair] = {**{f"train_{k}": v for k, v in m_train.items()},
                               **{f"test_{k}": v for k, v in m_test.items()}}
        print(f"  fold {fold['fold']:02d} {fold['instance']:<14}  "
              f"({time.monotonic() - t_start:.1f}s elapsed)")
        results["results"].append({
            "fold": fold["fold"], "instance": fold["instance"],
            "n_train_logs": len(fold["train_logs"]),
            "n_test_logs": len(fold["test_logs"]),
            "pair_metrics": fold_pair,
        })
    out_path = TBL / "loio_tree_d4_K4_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nTotal wall: {time.monotonic() - t_start:.0f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
