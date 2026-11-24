#!/usr/bin/env python3
"""Intra-log temporal split diagnostic.

Picks one log per (size, delay) instance (lexicographic first). For each log:
  - Sort rows by `iter`, split bottom 70% -> train, top 30% -> test.
  - Train forest_baseline_d_none_K20 on pair_0_0 at t03.
  - Record test F1-lift.

Compares against the same log's LOIO contribution from phase 2 (looked up via
loio_raw_results.json). Expected: intra-log F1-lift is inflated because the
feature set contains monotonic-in-iteration columns.

Outputs:
  tables/intra_log_diag.csv
  figures/intra_log_vs_loio.png
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FIG, TARGETS, TBL,
    compute_baselines, parse_log,
)

TARGET = "t03"
PAIR = "pair_0_0"


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading parquets...")
    features_df = pd.read_parquet(CACHE / "features.parquet")
    labels_df = pd.read_parquet(CACHE / TARGETS[TARGET])
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")

    # One log per instance (lexicographic first)
    by_instance = defaultdict(list)
    for lf in sorted(df["log_file"].unique()):
        key = parse_log(lf) or ("other", "other")
        by_instance[key].append(lf)
    chosen = {k: v[0] for k, v in by_instance.items()}

    # Try to look up LOIO F1-lift for pair_0_0 from raw results
    loio_path = TBL / "loio_raw_results.json"
    loio_by_instance = {}
    if loio_path.exists():
        loio = json.loads(loio_path.read_text())
        for entry in loio["results"]:
            if (entry["target"] != TARGET or
                entry["model_label"] != "forest_baseline_d_none_K20"):
                continue
            pm = entry["pair_metrics"].get(PAIR, {})
            f1l = pm.get("test_f1_lift") if isinstance(pm, dict) else None
            loio_by_instance[entry["instance"]] = f1l
    else:
        print("[warn] loio_raw_results.json not found — LOIO column will be empty")

    rows = []
    for key, log in sorted(chosen.items()):
        instance = f"{key[0]}_d{key[1]}"
        sub = df[df["log_file"] == log].sort_values("iter").reset_index(drop=True)
        n = len(sub)
        if n < 20:
            print(f"[skip] {log}: only {n} rows")
            continue
        cut = int(round(0.7 * n))
        train = sub.iloc[:cut]
        test = sub.iloc[cut:]
        X_train = train[FEATURE_NAMES_20].values
        X_test = test[FEATURE_NAMES_20].values
        y_train = train[PAIR].values
        y_test = test[PAIR].values
        if y_train.sum() < 2 or len(np.unique(y_train)) < 2 or len(y_test) == 0:
            f1_lift = None
            note = "degenerate_train_or_empty_test"
        else:
            clf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=42, n_jobs=-1, max_depth=None,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            base = compute_baselines(y_test)
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            f1_lift = f1 - base["best_baseline_f1"]
            note = ""
        rows.append({
            "instance": instance,
            "log_file": log,
            "n_total": n,
            "n_train_intra": cut,
            "n_test_intra": n - cut,
            "intra_log_f1_lift": f1_lift,
            "loio_f1_lift_pair_0_0": loio_by_instance.get(instance),
            "note": note,
        })
        print(f"  {instance:<14} intra={f1_lift if f1_lift is None else f'{f1_lift:+.3f}'}"
              f"  loio={loio_by_instance.get(instance)}")

    out_csv = TBL / "intra_log_diag.csv"
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.5))
    ypos = np.arange(len(rows))
    intra_vals = [r["intra_log_f1_lift"] if r["intra_log_f1_lift"] is not None
                  else 0 for r in rows]
    loio_vals = [r["loio_f1_lift_pair_0_0"]
                 if r["loio_f1_lift_pair_0_0"] is not None else 0 for r in rows]
    ax.barh(ypos - 0.18, intra_vals, height=0.36, label="intra-log 70/30",
            color="C3")
    ax.barh(ypos + 0.18, loio_vals, height=0.36, label="LOIO (cross-instance)",
            color="C0")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["instance"] for r in rows], fontsize=9)
    ax.set_xlabel("test F1-lift")
    ax.set_title(f"Intra-log vs LOIO F1-lift  ({TARGET}, {PAIR},\n"
                 f"forest_baseline_d_none_K20)\n"
                 f"Higher intra-log lift = temporal leakage from iteration-progress features")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png = FIG / "intra_log_vs_loio.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
