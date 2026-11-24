#!/usr/bin/env python3
"""Full t07 grid sweep under 5-fold CV.

Forests: n_estimators in {50, 100, 200}, depth in {4, 8, 16, None}, K in {4, 8, 16, 20}
Trees:   depth in {4, 8, 16, None}, K in {4, 8, 16, 20}

Total configs: 3*4*4 + 4*4 = 64.
Per config: 5 folds * 9 USED pairs = 45 fits → 2,880 total.

Each fit captures full train/test metrics:
  accuracy, precision, recall, specificity, f1, MCC, ROC-AUC, PR-AUC, Brier,
  TP/FP/FN/TN.

Output: tables/t07_sweep_raw.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FEATURES_PQ, LABELS_PQ, OUT_DIR, TBL,
    USED_PAIRS, aggregated_top_k, make_stratified_folds, mcc_cm,
)

# Grid
FOREST_NE = [50, 100, 200]
DEPTHS    = [4, 8, 16, None]
KS        = [4, 8, 16, 20]


def build_configs():
    configs = []
    for ne in FOREST_NE:
        for d in DEPTHS:
            for K in KS:
                configs.append({
                    "tag": f"forest_n{ne}_d{'none' if d is None else d}_K{K}",
                    "type": "forest", "n_estimators": ne, "depth": d, "K": K,
                })
    for d in DEPTHS:
        for K in KS:
            configs.append({
                "tag": f"tree_d{'none' if d is None else d}_K{K}",
                "type": "tree", "n_estimators": 1, "depth": d, "K": K,
            })
    return configs


def fit_model(cfg, X_train, y_train):
    if cfg["type"] == "forest":
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["depth"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        clf = DecisionTreeClassifier(
            max_depth=cfg["depth"], class_weight="balanced", random_state=42,
        )
    clf.fit(X_train, y_train)
    return clf


def metric_bundle(clf, X, y) -> dict:
    if len(y) == 0:
        return {"empty": True}
    y_pred = clf.predict(X)
    try:
        proba = clf.predict_proba(X)[:, 1]
    except Exception:
        proba = y_pred.astype(float)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
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
            out["roc_auc"] = None; out["pr_auc"] = None; out["brier"] = None
    else:
        out["roc_auc"] = None; out["pr_auc"] = None; out["brier"] = None
    return out


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    print("Loading parquets...")
    features_df = pd.read_parquet(FEATURES_PQ)
    labels_df = pd.read_parquet(LABELS_PQ)
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
    log_files = sorted(df["log_file"].unique())
    folds = make_stratified_folds(log_files, k=5, seed=42)
    print(f"  {len(log_files)} logs, {len(df)} labelled states, 5 folds")
    print()

    configs = build_configs()
    print(f"Configs to train: {len(configs)}")

    # Pre-compute feature subsets per K (shared across configs and folds)
    feat_subsets = {K: aggregated_top_k(K) for K in KS}
    for K in KS:
        print(f"  K={K}: features = {feat_subsets[K][:K] if K <= 6 else (feat_subsets[K][:3] + ['...'])}")
    print()

    log_arr = df["log_file"].values
    results = {
        "config": {
            "protocol": "5-fold stratified-by-(size,delay) round-robin",
            "n_folds": 5, "seed": 42,
            "configs_tested": [c["tag"] for c in configs],
            "n_logs": len(log_files), "n_states": len(df),
            "used_pairs": USED_PAIRS,
        },
        "feature_subsets": feat_subsets,
        "folds": [{"fold": f["fold"], "n_train_logs": len(f["train_logs"]),
                   "n_test_logs": len(f["test_logs"]),
                   "test_logs": f["test_logs"]} for f in folds],
        "results": [],
    }

    t_start = time.monotonic()
    n_done = 0; n_total = len(configs) * 5
    for cfg in configs:
        K = cfg["K"]
        feat_names = feat_subsets[K]
        X_all = df[feat_names].values
        for fold in folds:
            t0 = time.monotonic()
            train_mask = np.isin(log_arr, fold["train_logs"])
            test_mask = np.isin(log_arr, fold["test_logs"])
            X_train = X_all[train_mask]; X_test = X_all[test_mask]

            fold_pair_metrics = {}
            for pair in USED_PAIRS:
                y = df[pair].values
                y_train = y[train_mask]; y_test = y[test_mask]
                if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
                    fold_pair_metrics[pair] = {
                        "degenerate_train": True,
                        "test_n": int(test_mask.sum()),
                        "test_n_positive": int(y_test.sum()),
                    }
                    continue
                clf = fit_model(cfg, X_train, y_train)
                m_train = metric_bundle(clf, X_train, y_train)
                m_test = metric_bundle(clf, X_test, y_test)
                fold_pair_metrics[pair] = {
                    **{f"train_{k}": v for k, v in m_train.items()},
                    **{f"test_{k}": v for k, v in m_test.items()},
                }
            wall = time.monotonic() - t0
            n_done += 1
            elapsed = time.monotonic() - t_start
            print(f"  [{n_done:>3}/{n_total}] {cfg['tag']:<28} fold {fold['fold']} "
                  f"({wall:.1f}s, elapsed {elapsed:.0f}s)")
            results["results"].append({
                "tag": cfg["tag"], "type": cfg["type"],
                "n_estimators": cfg["n_estimators"], "depth": cfg["depth"],
                "K": cfg["K"], "fold": fold["fold"],
                "n_features": len(feat_names),
                "pair_metrics": fold_pair_metrics,
            })

    out_path = TBL / "t07_sweep_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nTotal wall: {time.monotonic() - t_start:.0f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
