#!/usr/bin/env python3
"""Learning curves: train on {20, 40, 60, 80, 100}% of training logs, test on
held-out LOIO instance. One curve per (target, model_label, pair).

If curves plateau by 80% -> data-saturated, more instances won't help.
If still rising at 100% -> more diverse instances would help.

Outputs:
  tables/learning_curve.csv
  figures/learning_curve_{target}_{model_label}_f1_lift.png   (per pair)
  figures/learning_curve_{target}_{model_label}_brier.png
  figures/learning_curve_aggregate.png  (2x4 grid)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FIG, MODEL_CONFIGS, TARGETS, TBL, USED_PAIRS,
    compute_baselines, make_loio_folds, parse_log,
)

FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]


def stratified_log_sample(train_logs: list[str], frac: float, rng: np.random.RandomState):
    """Sample `frac` of train_logs, stratified by (size, delay) round-robin.
    Guarantees at least 1 log per stratum if frac > 0.
    """
    strata = defaultdict(list)
    for n in train_logs:
        key = parse_log(n) or ("other", "other")
        strata[key].append(n)
    for key in strata:
        rng.shuffle(strata[key])
    if frac >= 1.0:
        return list(train_logs)
    n_total = len(train_logs)
    n_target = max(1, int(round(frac * n_total)))
    selected: list[str] = []
    pointers = {k: 0 for k in strata}
    keys_cycle = sorted(strata.keys())
    # Round-robin: take one log from each non-exhausted stratum until we hit n_target
    while len(selected) < n_target:
        progress = False
        for k in keys_cycle:
            if pointers[k] < len(strata[k]):
                selected.append(strata[k][pointers[k]])
                pointers[k] += 1
                progress = True
                if len(selected) >= n_target:
                    break
        if not progress:
            break
    return selected


def train_and_score(X_train, y_train, X_test, y_test, max_depth):
    if len(X_test) == 0 or len(X_train) == 0:
        return {"degenerate_train": True, "f1_lift": None, "brier": None,
                "prevalence": 0.0}
    if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
        return {"degenerate_train": True, "f1_lift": None, "brier": None,
                "prevalence": float(y_test.mean()) if len(y_test) else 0.0}
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=max_depth,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    base = compute_baselines(y_test)
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    f1_lift = f1 - base["best_baseline_f1"]
    try:
        brier = float(brier_score_loss(y_test, proba))
    except Exception:
        brier = None
    return {"degenerate_train": False, "f1_lift": f1_lift, "brier": brier,
            "prevalence": base["prevalence"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    args = ap.parse_args()

    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print(f"Loading features.parquet...")
    features_df = pd.read_parquet(CACHE / "features.parquet")
    log_files = sorted(features_df["log_file"].unique())
    folds = make_loio_folds(log_files)

    rows = []
    t_start = time.monotonic()
    for target in args.targets:
        labels_df = pd.read_parquet(CACHE / TARGETS[target])
        df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
        X_all = df[FEATURE_NAMES_20].values
        log_arr = df["log_file"].values

        for model_label, max_depth in MODEL_CONFIGS:
            for fold in folds:
                test_mask = np.isin(log_arr, fold["test_logs"])
                X_test = X_all[test_mask]
                rng = np.random.RandomState(42 + fold["fold"])
                for frac in FRACTIONS:
                    sampled = stratified_log_sample(fold["train_logs"], frac, rng)
                    train_mask = np.isin(log_arr, sampled)
                    X_train = X_all[train_mask]
                    n_train_rows = int(train_mask.sum())
                    n_train_logs = len(sampled)
                    for pair in USED_PAIRS:
                        y = df[pair].values
                        y_train = y[train_mask]
                        y_test = y[test_mask]
                        score = train_and_score(X_train, y_train, X_test,
                                                y_test, max_depth)
                        rows.append({
                            "target": target, "model_label": model_label,
                            "pair": pair, "fold": fold["fold"],
                            "instance": fold["instance"],
                            "fraction": frac,
                            "n_train_logs": n_train_logs,
                            "n_train_rows": n_train_rows,
                            "n_test_rows": int(test_mask.sum()),
                            **score,
                        })
                print(f"  {target}/{model_label} fold={fold['fold']:02d} "
                      f"{fold['instance']:<14} done ({(time.monotonic()-t_start):.0f}s elapsed)")

    out_csv = TBL / "learning_curve.csv"
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nWrote {out_csv} ({len(rows)} rows) in {time.monotonic()-t_start:.0f}s")

    # ---------------------------------------------------------------------
    # Per-(target, model_label) line plot: pair on y, fraction on x, F1-lift
    # ---------------------------------------------------------------------
    by_pair = {}
    for r in rows:
        if r["f1_lift"] is None:
            continue
        k = (r["target"], r["model_label"], r["pair"])
        by_pair.setdefault(k, []).append(r)

    # Build aggregates: per (target, model_label) - {pair: {frac: [vals over folds]}}
    for metric in ("f1_lift", "brier"):
        nested = {}
        for k, rs in by_pair.items():
            target, model_label, pair = k
            inner = nested.setdefault((target, model_label), {})
            inner_pair = inner.setdefault(pair, defaultdict(list))
            for r in rs:
                v = r[metric]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    inner_pair[r["fraction"]].append(v)

        for (target, model_label), per_pair in nested.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for pair, frac_map in sorted(per_pair.items()):
                xs = sorted(frac_map.keys())
                means = [np.mean(frac_map[x]) for x in xs]
                stds = [np.std(frac_map[x], ddof=1) if len(frac_map[x]) > 1 else 0
                        for x in xs]
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                            label=pair, linewidth=1.2)
            ax.set_xlabel("Fraction of training logs used")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{target} / {model_label} — learning curve ({metric})")
            ax.legend(fontsize=7, ncol=2, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = FIG / f"learning_curve_{target}_{model_label}_{metric}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)

    # ---------------------------------------------------------------------
    # Aggregate 2x4 grid: rows=metric, cols=target. Average over pairs+configs.
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    targets_ordered = ["t02", "t03", "t05", "t07"]
    metrics_ordered = ["f1_lift", "brier"]
    for ri, metric in enumerate(metrics_ordered):
        for ci, target in enumerate(targets_ordered):
            ax = axes[ri][ci]
            # Pool over (model_label, pair, fold) for this (target, fraction)
            frac_map = defaultdict(list)
            for r in rows:
                if r["target"] != target:
                    continue
                v = r[metric]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                frac_map[r["fraction"]].append(v)
            xs = sorted(frac_map.keys())
            means = [np.mean(frac_map[x]) for x in xs]
            stds = [np.std(frac_map[x], ddof=1) if len(frac_map[x]) > 1 else 0
                    for x in xs]
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                        color="C0", linewidth=1.5)
            ax.set_title(f"{target} — {metric}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.set_ylabel(metric)
            if ri == 1:
                ax.set_xlabel("fraction of train logs")
    fig.suptitle("Learning curves — averaged across pairs, configs, folds",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_agg = FIG / "learning_curve_aggregate.png"
    fig.savefig(out_agg, dpi=110)
    plt.close(fig)
    print(f"Wrote aggregate plot {out_agg}")


if __name__ == "__main__":
    main()
