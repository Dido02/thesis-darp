"""Step B — deep-dive on pair_3_3 (rtcritical destroy + slack repair) regression.

In the 5-feature baseline this pair had +0.36 F1 lift (best of all 20).
In the 21-feature model it has -0.192 F1 lift. Diagnose what changed.

Replicates the per-log split from training.py exactly (random_state=42,
test_size=0.25). For pair_3_3:
- positive rate train vs test
- confusion matrix on the test set
- per-feature mean/std on positive vs negative samples (test set)
- top features by separation score (|mean_pos - mean_neg| / pooled_std)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

FEATURES_PATH = Path("cache_v2_dynamic/features.parquet")
LABELS_PATH = Path("cache_v2_dynamic/labels.parquet")
PAIR = "pair_3_3"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def main():
    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
    print(f"Joined: {len(df)} samples")

    feature_names = [c for c in features_df.columns if c not in ("log_file", "iter")]
    print(f"Features ({len(feature_names)}): {feature_names}\n")

    # Replicate per-log split EXACTLY (training.py:148-158)
    log_files = df["log_file"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = rng.permutation(log_files)
    n_train_logs = int(len(shuffled) * (1 - TEST_SIZE))
    train_logs = set(shuffled[:n_train_logs])
    train_mask = df["log_file"].isin(train_logs)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    print(f"Train: {len(train_logs)} logs, {len(train_idx)} rows")
    print(f"Test:  {len(log_files) - len(train_logs)} logs, {len(test_idx)} rows\n")

    if PAIR not in df.columns:
        print(f"ERROR: {PAIR} not in labels parquet. Available pair cols:")
        for c in df.columns:
            if c.startswith("pair_"):
                print(f"  {c}")
        return

    y_full = df[PAIR].values
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    # ---- Class balance ----
    print("=" * 80)
    print(f"PAIR {PAIR} class balance")
    print("=" * 80)
    print(f"  train: {int(y_train.sum())}/{len(y_train)} positive "
          f"({y_train.mean():.4f})")
    print(f"  test:  {int(y_test.sum())}/{len(y_test)} positive "
          f"({y_test.mean():.4f})\n")

    # ---- Re-train pair_3_3 with same params, capture predictions ----
    X = df[feature_names].values
    X_train = X[train_idx]
    X_test = X[test_idx]
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("=" * 80)
    print("Confusion matrix on TEST set (rows = true, cols = pred)")
    print("=" * 80)
    print(f"        pred=0   pred=1")
    print(f"true=0  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"true=1  {cm[1,0]:6d}   {cm[1,1]:6d}")
    print(f"\n  always-1 baseline F1: {f1_score(y_test, np.ones_like(y_test), zero_division=0):.4f}")
    print(f"  always-0 baseline F1: {f1_score(y_test, np.zeros_like(y_test), zero_division=0):.4f}")
    print(f"  RF F1               : {f1_score(y_test, y_pred, zero_division=0):.4f}\n")

    # ---- Feature separation on TEST set ----
    print("=" * 80)
    print(f"Feature separation on TEST set (sorted by |mean_pos - mean_neg| / pooled_std)")
    print("=" * 80)
    test_pos_mask = y_test == 1
    test_neg_mask = y_test == 0
    print(f"  n_pos = {test_pos_mask.sum()},  n_neg = {test_neg_mask.sum()}\n")

    rows = []
    for i, name in enumerate(feature_names):
        col = X_test[:, i]
        mp, sp = col[test_pos_mask].mean(), col[test_pos_mask].std()
        mn, sn = col[test_neg_mask].mean(), col[test_neg_mask].std()
        pooled_std = np.sqrt((sp**2 + sn**2) / 2.0)
        sep = abs(mp - mn) / pooled_std if pooled_std > 1e-12 else 0.0
        rows.append((name, mp, sp, mn, sn, sep))
    rows.sort(key=lambda r: -r[5])
    print(f"  {'feature':32s} {'mean_pos':>10s} {'std_pos':>10s} "
          f"{'mean_neg':>10s} {'std_neg':>10s} {'separation':>10s}")
    print("  " + "-" * 88)
    for name, mp, sp, mn, sn, sep in rows:
        print(f"  {name:32s} {mp:10.4f} {sp:10.4f} {mn:10.4f} {sn:10.4f} {sep:10.4f}")

    # ---- Per-pair feature importances from this RF ----
    print("\n" + "=" * 80)
    print(f"Per-pair feature importances for {PAIR} (sorted desc)")
    print("=" * 80)
    imps = list(zip(feature_names, clf.feature_importances_))
    imps.sort(key=lambda x: -x[1])
    for name, imp in imps:
        print(f"  {name:32s}  {imp:.4f}")

    # ---- Per-test-log positive rate ----
    print("\n" + "=" * 80)
    print(f"Per-test-log positive rate for {PAIR}")
    print("=" * 80)
    test_logs = df.loc[test_idx, "log_file"].values
    test_log_set = sorted(set(test_logs))
    for lg in test_log_set:
        mask = test_logs == lg
        n = int(mask.sum())
        n_pos = int(y_test[mask].sum())
        print(f"  {lg:60s} {n_pos:4d}/{n:4d} ({n_pos/n if n else 0:.3f})")


if __name__ == "__main__":
    main()
