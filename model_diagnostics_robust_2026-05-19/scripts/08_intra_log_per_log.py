#!/usr/bin/env python3
"""Per-log intra-trajectory training.

For each (target, model_config, log, feature_set, pair):
  - Sort log rows by iter, split bottom 70% -> train, top 30% -> test
  - Train one forest per pair, evaluate on the held-out top 30%
  - Capture F1-lift, Brier, ROC-AUC, PR-AUC, prevalence

Then aggregate across the 47 logs.

Feature sets:
  - all_20:      all 20 features (matches LOIO)
  - no_temporal: drops iteration-progress features
                 {iteration_norm, temperature_normalized, horizon_elapsed_frac,
                  frac_committed, committed_travel_time_frac,
                  cost_vs_initial_frac, cost_vs_incumbent_frac}
                 If F1-lift drops sharply, that's the temporal leak signal.

Outputs:
  tables/intra_log_per_log.csv         (long: one row per log)
  tables/intra_log_summary.csv         (per target/model/pair/feature_set)
  tables/intra_log_vs_loio.csv         (head-to-head with LOIO)
  figures/intra_log_aggregate.png      (bar chart at t03 baseline across pairs)
  figures/intra_log_target_grid.png    (2x4 grid across targets)
  figures/intra_log_feat_ablation.png  (with vs without temporal features)
"""
from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FIG, MODEL_CONFIGS, TARGETS, TBL, USED_PAIRS,
    compute_baselines, parse_log,
)

TEMPORAL_FEATURES = {
    "iteration_norm", "temperature_normalized", "horizon_elapsed_frac",
    "frac_committed", "committed_travel_time_frac",
    "cost_vs_initial_frac", "cost_vs_incumbent_frac",
}
FEATURE_SETS = {
    "all_20": FEATURE_NAMES_20,
    "no_temporal": [f for f in FEATURE_NAMES_20 if f not in TEMPORAL_FEATURES],
}
SPLIT_FRAC = 0.70


def train_and_eval(X_train, y_train, X_test, y_test, max_depth):
    if len(X_train) < 5 or len(X_test) < 1:
        return {"degenerate": True, "reason": "n_rows<5"}
    if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
        return {"degenerate": True, "reason": "degenerate_train_y"}
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=max_depth,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    base = compute_baselines(y_test)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    out = {
        "degenerate": False,
        "f1": f1,
        "f1_lift": f1 - base["best_baseline_f1"],
        "prevalence": base["prevalence"],
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
    }
    try:
        out["brier"] = float(brier_score_loss(y_test, proba))
    except Exception:
        out["brier"] = None
    if len(np.unique(y_test)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
            out["pr_auc"] = float(average_precision_score(y_test, proba))
        except Exception:
            out["roc_auc"] = None; out["pr_auc"] = None
    else:
        out["roc_auc"] = None; out["pr_auc"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    args = ap.parse_args()

    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    features_df = pd.read_parquet(CACHE / "features.parquet")

    rows = []
    t_start = time.monotonic()
    for target in args.targets:
        labels_df = pd.read_parquet(CACHE / TARGETS[target])
        df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
        logs = sorted(df["log_file"].unique())
        print(f"\n[{target}] {len(logs)} logs after labels merge")

        for log in logs:
            sub = df[df["log_file"] == log].sort_values("iter").reset_index(drop=True)
            n_total = len(sub)
            cut = int(round(SPLIT_FRAC * n_total))
            if cut < 5 or n_total - cut < 1:
                continue
            instance = parse_log(log)
            instance_str = f"{instance[0]}_d{instance[1]}" if instance else "other"
            train_part = sub.iloc[:cut]
            test_part = sub.iloc[cut:]
            for feat_set_name, feat_list in FEATURE_SETS.items():
                X_train = train_part[feat_list].values
                X_test = test_part[feat_list].values
                for model_label, max_depth in MODEL_CONFIGS:
                    for pair in USED_PAIRS:
                        y_train = train_part[pair].values
                        y_test = test_part[pair].values
                        score = train_and_eval(X_train, y_train, X_test,
                                               y_test, max_depth)
                        rows.append({
                            "target": target,
                            "model_label": model_label,
                            "feature_set": feat_set_name,
                            "n_features": len(feat_list),
                            "log_file": log,
                            "instance": instance_str,
                            "pair": pair,
                            "n_total": n_total,
                            "n_train": cut,
                            "n_test": n_total - cut,
                            **score,
                        })
        elapsed = time.monotonic() - t_start
        print(f"  done {target} ({elapsed:.0f}s elapsed)")

    # Long CSV — use UNION of keys across all rows (degenerate rows have
    # different keys than non-degenerate ones)
    out_long = TBL / "intra_log_per_log.csv"
    with open(out_long, "w", newline="") as f:
        if rows:
            all_keys = []
            seen = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k); all_keys.append(k)
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nWrote {out_long} ({len(rows)} rows) in {time.monotonic()-t_start:.0f}s")

    # ---------------------------------------------------------------------
    # Summary: per (target, model_label, feature_set, pair) — mean across logs
    # ---------------------------------------------------------------------
    summary = defaultdict(list)
    for r in rows:
        if r.get("degenerate"):
            continue
        k = (r["target"], r["model_label"], r["feature_set"], r["pair"])
        summary[k].append(r)
    sum_rows = []
    for k, rs in sorted(summary.items()):
        target, ml, fs, pair = k

        def col(name):
            return [r[name] for r in rs if isinstance(r[name], (int, float))
                    and not (isinstance(r[name], float) and np.isnan(r[name]))]
        f1l = col("f1_lift")
        brier = col("brier")
        roc = col("roc_auc")
        pr = col("pr_auc")
        sum_rows.append({
            "target": target, "model_label": ml, "feature_set": fs, "pair": pair,
            "n_logs_used": len(rs),
            "f1_lift_mean": float(np.mean(f1l)) if f1l else None,
            "f1_lift_median": float(np.median(f1l)) if f1l else None,
            "f1_lift_std": float(np.std(f1l, ddof=1)) if len(f1l) > 1 else 0.0,
            "f1_lift_p10": float(np.percentile(f1l, 10)) if f1l else None,
            "f1_lift_p90": float(np.percentile(f1l, 90)) if f1l else None,
            "brier_mean": float(np.mean(brier)) if brier else None,
            "roc_auc_mean": float(np.mean(roc)) if roc else None,
            "pr_auc_mean": float(np.mean(pr)) if pr else None,
        })
    out_sum = TBL / "intra_log_summary.csv"
    with open(out_sum, "w", newline="") as f:
        if sum_rows:
            w = csv.DictWriter(f, fieldnames=list(sum_rows[0]))
            w.writeheader()
            for r in sum_rows:
                w.writerow(r)
    print(f"Wrote {out_sum} ({len(sum_rows)} rows)")

    # ---------------------------------------------------------------------
    # Head-to-head with LOIO
    # ---------------------------------------------------------------------
    loio_path = TBL / "loio_raw_results.json"
    if loio_path.exists():
        loio = json.loads(loio_path.read_text())
        loio_idx = defaultdict(list)
        for entry in loio["results"]:
            for pair, pm in entry["pair_metrics"].items():
                if not isinstance(pm, dict) or pm.get("degenerate_train"):
                    continue
                if pm.get("test_f1_lift") is not None:
                    loio_idx[(entry["target"], entry["model_label"], pair)].append(
                        float(pm["test_f1_lift"]))

        h2h_rows = []
        for r in sum_rows:
            k = (r["target"], r["model_label"], r["pair"])
            loio_vals = loio_idx.get(k, [])
            h2h_rows.append({
                "target": r["target"], "model_label": r["model_label"],
                "feature_set": r["feature_set"], "pair": r["pair"],
                "intra_log_f1_lift_mean": r["f1_lift_mean"],
                "loio_f1_lift_mean": (float(np.mean(loio_vals))
                                      if loio_vals else None),
                "intra_minus_loio": (
                    (r["f1_lift_mean"] - float(np.mean(loio_vals)))
                    if (r["f1_lift_mean"] is not None and loio_vals) else None),
            })
        out_h2h = TBL / "intra_log_vs_loio.csv"
        with open(out_h2h, "w", newline="") as f:
            if h2h_rows:
                w = csv.DictWriter(f, fieldnames=list(h2h_rows[0]))
                w.writeheader()
                for r in h2h_rows:
                    w.writerow(r)
        print(f"Wrote {out_h2h} ({len(h2h_rows)} rows)")
    else:
        print(f"[skip] intra_log_vs_loio (no loio_raw_results.json)")

    # ---------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------
    # 1) Aggregate bar at t03 / forest_baseline: per pair, two bars (all_20 vs no_temporal)
    target = "t03"; ml = "forest_baseline_d_none_K20"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(USED_PAIRS))
    all_means = []; all_stds = []
    no_means = []; no_stds = []
    for p in USED_PAIRS:
        rs_all = [r for r in rows if r["target"] == target and r["model_label"] == ml
                  and r["feature_set"] == "all_20" and r["pair"] == p
                  and not r.get("degenerate")]
        rs_no = [r for r in rows if r["target"] == target and r["model_label"] == ml
                 and r["feature_set"] == "no_temporal" and r["pair"] == p
                 and not r.get("degenerate")]
        all_means.append(np.mean([r["f1_lift"] for r in rs_all]) if rs_all else 0)
        all_stds.append(np.std([r["f1_lift"] for r in rs_all], ddof=1) if len(rs_all) > 1 else 0)
        no_means.append(np.mean([r["f1_lift"] for r in rs_no]) if rs_no else 0)
        no_stds.append(np.std([r["f1_lift"] for r in rs_no], ddof=1) if len(rs_no) > 1 else 0)
    ax.bar(x - 0.2, all_means, width=0.4, yerr=all_stds, color="C3",
           capsize=2, label="all 20 features")
    ax.bar(x + 0.2, no_means, width=0.4, yerr=no_stds, color="C2",
           capsize=2, label="no temporal features")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("pair_", "") for p in USED_PAIRS],
                      rotation=45, fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("test F1-lift (mean across logs)")
    ax.set_title(f"Intra-log per-log training  ({target} / {ml})\n"
                 f"with vs without iteration/temperature/cost-ratio features")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "intra_log_aggregate.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"Wrote {out}")

    # 2) Target grid: 2x4, rows=feature_set, cols=target. Averaged over pairs.
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    targets_ordered = ["t02", "t03", "t05", "t07"]
    for ri, fs in enumerate(["all_20", "no_temporal"]):
        for ci, target_g in enumerate(targets_ordered):
            ax = axes[ri][ci]
            xs = np.arange(len(USED_PAIRS))
            means = []; stds = []
            for p in USED_PAIRS:
                rs = [r for r in rows if r["target"] == target_g
                      and r["model_label"] == ml and r["feature_set"] == fs
                      and r["pair"] == p and not r.get("degenerate")]
                if rs:
                    means.append(np.mean([r["f1_lift"] for r in rs]))
                    stds.append(np.std([r["f1_lift"] for r in rs], ddof=1)
                                if len(rs) > 1 else 0)
                else:
                    means.append(0); stds.append(0)
            color = "C3" if fs == "all_20" else "C2"
            ax.bar(xs, means, yerr=stds, color=color, capsize=2)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_title(f"{target_g} ({fs})", fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels([p.replace("pair_", "") for p in USED_PAIRS],
                              rotation=45, fontsize=7)
            if ci == 0:
                ax.set_ylabel("intra-log F1-lift")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Intra-log per-log F1-lift across targets, with vs without temporal features  ({ml})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "intra_log_target_grid.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"Wrote {out}")

    # 3) Feature-ablation: at each target, mean f1_lift across all pairs+logs,
    #    for all_20 vs no_temporal vs (if loio available) LOIO baseline
    loio_means_target = {}
    if loio_path.exists():
        for k, vs in loio_idx.items():
            t, mlx, p = k
            if mlx == ml:
                loio_means_target.setdefault(t, []).extend(vs)
        loio_means_target = {t: float(np.mean(vs))
                             for t, vs in loio_means_target.items() if vs}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(targets_ordered))
    all_vals = []; no_vals = []; loio_vals_list = []
    for t in targets_ordered:
        rs_all = [r["f1_lift"] for r in rows if r["target"] == t
                  and r["model_label"] == ml and r["feature_set"] == "all_20"
                  and not r.get("degenerate")]
        rs_no = [r["f1_lift"] for r in rows if r["target"] == t
                 and r["model_label"] == ml and r["feature_set"] == "no_temporal"
                 and not r.get("degenerate")]
        all_vals.append(np.mean(rs_all) if rs_all else 0)
        no_vals.append(np.mean(rs_no) if rs_no else 0)
        loio_vals_list.append(loio_means_target.get(t, 0))
    width = 0.25
    ax.bar(xs - width, all_vals, width=width, color="C3", label="intra-log all_20")
    ax.bar(xs, no_vals, width=width, color="C2", label="intra-log no_temporal")
    ax.bar(xs + width, loio_vals_list, width=width, color="C0",
           label="LOIO all_20 (reference)")
    ax.set_xticks(xs)
    ax.set_xticklabels(targets_ordered)
    ax.set_xlabel("target")
    ax.set_ylabel("mean test F1-lift")
    ax.set_title(f"Temporal-feature leakage diagnostic  ({ml})\n"
                 f"gap between red and green = temporal-leak contribution to F1-lift")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "intra_log_feat_ablation.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
