#!/usr/bin/env python3
"""Cross-protocol summary: LOIO vs 5-fold + bigger-picture + overfitting.

Reads:
  tables/loio_raw_results.json                   (this folder)
  tables/loio_calibration_summary.csv
  tables/loio_topk_state.csv
  tables/learning_curve.csv
  ../model_diagnostics_cv_2026-05-19/tables/cv_raw_results.json  (read-only)

Writes:
  tables/loio_vs_5fold_paired.csv
  tables/bigger_picture.csv
  tables/overfitting_diagnosis.csv
  figures/headline_loio_vs_5fold.png
  figures/calibration_grid.png

Optional:
  --purge-fold-models --yes  -> shutil.rmtree(fold_models)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    FIG, FOLD_MODELS, OUT_DIR, REPO, TBL, USED_PAIRS,
)


def safe_wilcoxon(loio_vals, cv5_vals):
    try:
        from scipy.stats import wilcoxon
        if len(loio_vals) < 3 or len(cv5_vals) < 3:
            return None
        m = min(len(loio_vals), len(cv5_vals))
        a = loio_vals[:m]; b = cv5_vals[:m]
        if all(x == y for x, y in zip(a, b)):
            return 1.0
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge-fold-models", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="Confirm the fold_models/ purge.")
    args = ap.parse_args()

    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading raw inputs...")
    loio = json.loads((TBL / "loio_raw_results.json").read_text())
    cv5_path = (REPO / "model_diagnostics_cv_2026-05-19" /
                "tables" / "cv_raw_results.json")
    if not cv5_path.exists():
        cv5_path = (REPO / "model_diagnostics_cv_2026-05-19" /
                    "cv_raw_results.json")
    cv5 = json.loads(cv5_path.read_text()) if cv5_path.exists() else None

    cal_sum_path = TBL / "loio_calibration_summary.csv"
    topk_path = TBL / "loio_topk_state.csv"
    lc_path = TBL / "learning_curve.csv"

    cal_df = pd.read_csv(cal_sum_path) if cal_sum_path.exists() else pd.DataFrame()
    topk_df = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    lc_df = pd.read_csv(lc_path) if lc_path.exists() else pd.DataFrame()

    # ---------------------------------------------------------------------
    # 1. LOIO vs 5-fold paired table
    # ---------------------------------------------------------------------
    print("Building LOIO vs 5-fold table...")
    # Index LOIO test F1-lift: per (target, model_label, pair) -> list across folds
    loio_idx = defaultdict(list)
    loio_train_idx = defaultdict(list)
    loio_brier_idx = defaultdict(list)
    loio_ece_idx = defaultdict(list)
    for entry in loio["results"]:
        for pair, pm in entry["pair_metrics"].items():
            if not isinstance(pm, dict) or pm.get("degenerate_train"):
                continue
            k = (entry["target"], entry["model_label"], pair)
            if pm.get("test_f1_lift") is not None:
                loio_idx[k].append(float(pm["test_f1_lift"]))
            if pm.get("train_f1_lift") is not None:
                loio_train_idx[k].append(float(pm["train_f1_lift"]))
            if pm.get("test_brier") is not None:
                loio_brier_idx[k].append(float(pm["test_brier"]))
            if pm.get("test_ece_10bin") is not None:
                loio_ece_idx[k].append(float(pm["test_ece_10bin"]))

    cv5_idx = defaultdict(list)
    cv5_train_idx = defaultdict(list)
    cv5_brier_idx = defaultdict(list)
    if cv5 is not None:
        for entry in cv5["results"]:
            for pair, pm in entry["pair_metrics"].items():
                if not isinstance(pm, dict) or pm.get("degenerate_train"):
                    continue
                k = (entry["target"], entry["model_label"], pair)
                if pm.get("test_f1_lift") is not None:
                    cv5_idx[k].append(float(pm["test_f1_lift"]))
                if pm.get("train_f1_lift") is not None:
                    cv5_train_idx[k].append(float(pm["train_f1_lift"]))
                if pm.get("test_brier") is not None:
                    cv5_brier_idx[k].append(float(pm["test_brier"]))

    paired_rows = []
    for k in sorted(loio_idx):
        target, ml, pair = k
        loio_vals = loio_idx[k]
        cv5_vals = cv5_idx.get(k, [])
        row = {
            "target": target, "model_label": ml, "pair": pair,
            "loio_n_folds": len(loio_vals),
            "loio_f1_lift_mean": float(np.mean(loio_vals)) if loio_vals else None,
            "loio_f1_lift_median": float(np.median(loio_vals)) if loio_vals else None,
            "loio_f1_lift_iqr": (float(np.percentile(loio_vals, 75) -
                                       np.percentile(loio_vals, 25))
                                 if loio_vals else None),
            "loio_f1_lift_p10": float(np.percentile(loio_vals, 10)) if loio_vals else None,
            "loio_f1_lift_p90": float(np.percentile(loio_vals, 90)) if loio_vals else None,
            "loio_f1_lift_std": (float(np.std(loio_vals, ddof=1))
                                  if len(loio_vals) > 1 else 0.0),
            "cv5_n_folds": len(cv5_vals),
            "cv5_f1_lift_mean": float(np.mean(cv5_vals)) if cv5_vals else None,
            "cv5_f1_lift_median": float(np.median(cv5_vals)) if cv5_vals else None,
            "cv5_f1_lift_std": (float(np.std(cv5_vals, ddof=1))
                                 if len(cv5_vals) > 1 else 0.0),
            "delta_loio_minus_cv5": (
                (float(np.mean(loio_vals) - np.mean(cv5_vals))
                 if (loio_vals and cv5_vals) else None)
            ),
        }
        # Per-fold Wilcoxon not appropriate (folds aren't matched 1-to-1); we
        # compute the across-pair paired test in a separate aggregation below.
        paired_rows.append(row)

    out_paired = TBL / "loio_vs_5fold_paired.csv"
    with open(out_paired, "w", newline="") as f:
        if paired_rows:
            w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
            w.writeheader()
            for r in paired_rows:
                w.writerow(r)
    print(f"Wrote {out_paired} ({len(paired_rows)} rows)")

    # Across-pair Wilcoxon: pair the (target, model_label, pair) means
    # (n=4*2*9=72 pairs at most) — gives one global p-value per (target, ml).
    print("Computing per-(target, model_label) paired Wilcoxon over pairs...")
    wilcoxon_rows = []
    by_tm = defaultdict(list)
    for r in paired_rows:
        if r["loio_f1_lift_mean"] is None or r["cv5_f1_lift_mean"] is None:
            continue
        by_tm[(r["target"], r["model_label"])].append(
            (r["loio_f1_lift_mean"], r["cv5_f1_lift_mean"]))
    for (target, ml), pairs in sorted(by_tm.items()):
        if len(pairs) < 3:
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        p_val = safe_wilcoxon(a, b)
        wilcoxon_rows.append({
            "target": target, "model_label": ml,
            "n_pairs_compared": len(pairs),
            "mean_delta": float(np.mean([ai - bi for ai, bi in pairs])),
            "median_delta": float(np.median([ai - bi for ai, bi in pairs])),
            "wilcoxon_p": p_val,
        })
    out_wx = TBL / "loio_vs_5fold_wilcoxon.csv"
    with open(out_wx, "w", newline="") as f:
        if wilcoxon_rows:
            w = csv.DictWriter(f, fieldnames=list(wilcoxon_rows[0]))
            w.writeheader()
            for r in wilcoxon_rows:
                w.writerow(r)
    print(f"Wrote {out_wx} ({len(wilcoxon_rows)} rows)")

    # ---------------------------------------------------------------------
    # 2. Bigger picture table
    # ---------------------------------------------------------------------
    print("Building bigger-picture table...")
    cal_map = {}
    if not cal_df.empty:
        for _, r in cal_df.iterrows():
            cal_map[(r["target"], r["model_label"], r["pair"])] = {
                "brier_mean": r.get("brier_mean"),
                "ece_mean": r.get("ece_10bin_mean"),
                "roc_auc_mean": r.get("roc_auc_mean"),
                "pr_auc_mean": r.get("pr_auc_mean"),
            }
    topk_map = {}
    if not topk_df.empty:
        for (target, ml), rs in topk_df.groupby(["target", "model_label"]):
            topk_map[(target, ml)] = {
                "top1_acc_mean": float(rs["top1_acc"].mean()),
                "top3_acc_mean": float(rs["top3_acc"].mean()),
                "cost_strict_mean": float(rs["cost_strict_mean"].mean()),
                "cost_regret_mean": float(rs["cost_regret_mean"].mean()),
            }

    bp_rows = []
    for k in sorted(loio_idx):
        target, ml, pair = k
        loio_vals = loio_idx[k]
        cal = cal_map.get(k, {})
        topk = topk_map.get((target, ml), {})
        bp_rows.append({
            "target": target, "model_label": ml, "pair": pair,
            "loio_f1_lift_mean": float(np.mean(loio_vals)) if loio_vals else None,
            "loio_pr_auc_mean": cal.get("pr_auc_mean"),
            "loio_roc_auc_mean": cal.get("roc_auc_mean"),
            "loio_brier_mean": cal.get("brier_mean"),
            "loio_ece_mean": cal.get("ece_mean"),
            "top1_acc_mean": topk.get("top1_acc_mean"),
            "top3_acc_mean": topk.get("top3_acc_mean"),
            "cost_strict_mean": topk.get("cost_strict_mean"),
            "cost_regret_mean": topk.get("cost_regret_mean"),
            "random_top1_baseline": 1.0 / 9,
            "random_top3_baseline": 3.0 / 9,
        })

    out_bp = TBL / "bigger_picture.csv"
    with open(out_bp, "w", newline="") as f:
        if bp_rows:
            w = csv.DictWriter(f, fieldnames=list(bp_rows[0]))
            w.writeheader()
            for r in bp_rows:
                w.writerow(r)
    print(f"Wrote {out_bp} ({len(bp_rows)} rows)")

    # ---------------------------------------------------------------------
    # 3. Overfitting diagnosis
    # ---------------------------------------------------------------------
    print("Building overfitting diagnosis...")
    of_rows = []
    for k in sorted(loio_idx):
        target, ml, pair = k
        loio_test = loio_idx[k]
        loio_train = loio_train_idx.get(k, [])
        # Learning curve slope between 80% and 100%
        slope = None
        if not lc_df.empty:
            sub = lc_df[(lc_df["target"] == target) &
                        (lc_df["model_label"] == ml) &
                        (lc_df["pair"] == pair)]
            if len(sub) > 0 and sub["f1_lift"].notna().any():
                f80 = sub[sub["fraction"] == 0.80]["f1_lift"]
                f100 = sub[sub["fraction"] == 1.00]["f1_lift"]
                if f80.notna().any() and f100.notna().any():
                    slope = float(f100.mean() - f80.mean())
        of_rows.append({
            "target": target, "model_label": ml, "pair": pair,
            "loio_test_f1_lift_mean": float(np.mean(loio_test)) if loio_test else None,
            "loio_train_f1_lift_mean": float(np.mean(loio_train)) if loio_train else None,
            "gap_train_minus_test_f1_lift": (
                float(np.mean(loio_train) - np.mean(loio_test))
                if (loio_train and loio_test) else None),
            "learning_curve_slope_80_to_100": slope,
            "n_loio_folds": len(loio_test),
        })

    out_of = TBL / "overfitting_diagnosis.csv"
    with open(out_of, "w", newline="") as f:
        if of_rows:
            w = csv.DictWriter(f, fieldnames=list(of_rows[0]))
            w.writeheader()
            for r in of_rows:
                w.writerow(r)
    print(f"Wrote {out_of} ({len(of_rows)} rows)")

    # ---------------------------------------------------------------------
    # 4. Headline figure: LOIO vs 5-fold per target
    # ---------------------------------------------------------------------
    print("Rendering headline LOIO vs 5-fold figure...")
    targets_ordered = ["t02", "t03", "t05", "t07"]
    # Restrict to baseline model for the figure (simpler)
    ml_for_plot = "forest_baseline_d_none_K20"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ci, target in enumerate(targets_ordered):
        ax = axes[ci]
        pairs = USED_PAIRS
        x = np.arange(len(pairs))
        loio_means = []
        cv5_means = []
        loio_stds = []
        cv5_stds = []
        for p in pairs:
            k = (target, ml_for_plot, p)
            lv = loio_idx.get(k, [])
            cv5v = cv5_idx.get(k, [])
            loio_means.append(np.mean(lv) if lv else 0)
            loio_stds.append(np.std(lv, ddof=1) if len(lv) > 1 else 0)
            cv5_means.append(np.mean(cv5v) if cv5v else 0)
            cv5_stds.append(np.std(cv5v, ddof=1) if len(cv5v) > 1 else 0)
        ax.bar(x - 0.2, loio_means, width=0.4, yerr=loio_stds,
               color="C3", capsize=2, label="LOIO")
        ax.bar(x + 0.2, cv5_means, width=0.4, yerr=cv5_stds,
               color="C0", capsize=2, label="5-fold CV")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("pair_", "") for p in pairs],
                          rotation=45, fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(target)
        if ci == 0:
            ax.set_ylabel("test F1-lift")
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"LOIO vs 5-fold CV F1-lift  ({ml_for_plot})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "headline_loio_vs_5fold.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------------------------------------------------------------------
    # 5. Calibration grid (3x3 reliability for the 9 pairs on t03 baseline)
    # ---------------------------------------------------------------------
    print("Rendering calibration grid...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    target = "t03"; ml = "forest_baseline_d_none_K20"
    for idx, pair in enumerate(USED_PAIRS):
        ax = axes[idx // 3][idx % 3]
        # Try to embed the saved reliability PNG by re-loading from disk
        rel_path = FIG / f"reliability_{target}_{ml}_{pair}.png"
        if rel_path.exists():
            try:
                img = plt.imread(rel_path)
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
            except Exception:
                ax.text(0.5, 0.5, pair, ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"{pair}\n(no reliability png)",
                    ha="center", va="center")
        ax.set_title(pair, fontsize=9)
    fig.suptitle(f"Calibration reliability grid — {target} / {ml}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "calibration_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------------------------------------------------------------------
    # Optional purge
    # ---------------------------------------------------------------------
    if args.purge_fold_models:
        if not args.yes:
            print("\n[purge requested] add --yes to confirm rm of fold_models/")
        else:
            if FOLD_MODELS.exists():
                print(f"\nPurging {FOLD_MODELS} ...")
                shutil.rmtree(FOLD_MODELS)
                print("  done.")

    # Summary stdout
    print("\n=== Summary ===")
    print(f"tables/ contents:")
    for p in sorted(TBL.glob("*.csv")):
        print(f"  {p.name}")
    print(f"figures/ contents (count): {len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
