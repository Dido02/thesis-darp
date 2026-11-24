#!/usr/bin/env python3
"""Calibration (Brier, ECE), PR/ROC-AUC, and reliability diagrams.

Reads loio_raw_results.json + per-fold state_proba.parquet.
No retraining; pure post-processing on cached probas.

Outputs:
  tables/loio_calibration.csv          (long: per fold per pair)
  tables/loio_calibration_summary.csv  (per (target, model_label, pair))
  figures/reliability_{target}_{model_label}_{pair}.png  (averaged across folds)
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FIG, FOLD_MODELS, MODEL_CONFIGS, TARGETS, TBL, USED_PAIRS,
    compute_ece,
)
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score,
)


def reliability_bins(y_true, y_proba, n_bins: int = 10):
    """Return per-bin (mean_proba, fraction_positive, count)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            out.append((float((lo + hi) / 2), float("nan"), 0))
        else:
            out.append((float(y_proba[mask].mean()),
                        float(y_true[mask].mean()), n_bin))
    return out


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    raw = json.loads((TBL / "loio_raw_results.json").read_text())
    folds = {f["fold"]: f for f in raw["folds_meta"]}

    # Long per-fold table
    long_rows = []
    # Accumulator for averaged reliability per (target, model_label, pair)
    # key -> list of (proba_array, label_array)
    bucket = {}

    for entry in raw["results"]:
        target = entry["target"]
        model_label = entry["model_label"]
        fold_idx = entry["fold"]
        instance = entry["instance"]
        fold_meta = folds[fold_idx]
        labels_df = pd.read_parquet(CACHE / TARGETS[target])
        # restrict to test logs
        labels_df = labels_df[labels_df["log_file"].isin(fold_meta["test_logs"])]
        fold_dir = (FOLD_MODELS / target / model_label /
                    f"fold{fold_idx:02d}_{instance}")
        proba_path = fold_dir / "state_proba.parquet"
        if not proba_path.exists():
            print(f"[skip] missing {proba_path}")
            continue
        proba_df = pd.read_parquet(proba_path)
        merged = proba_df.merge(labels_df, on=["log_file", "iter"], how="inner")

        for pair in USED_PAIRS:
            proba_col = f"proba_{pair}"
            if proba_col not in merged.columns:
                continue
            y = merged[pair].values
            p = merged[proba_col].values
            if len(y) == 0:
                continue
            uniq = np.unique(y)
            if len(uniq) > 1:
                brier = float(brier_score_loss(y, p))
                roc = float(roc_auc_score(y, p))
                pr = float(average_precision_score(y, p))
            else:
                brier = float(((p - y) ** 2).mean())
                roc = float("nan")
                pr = float("nan")
            ece = compute_ece(y, p, n_bins=10)
            long_rows.append({
                "target": target, "model_label": model_label,
                "fold": fold_idx, "instance": instance, "pair": pair,
                "n_test": int(len(y)),
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "brier": brier, "ece_10bin": ece,
                "roc_auc": roc, "pr_auc": pr,
            })
            key = (target, model_label, pair)
            bucket.setdefault(key, []).append((p, y))

    # Write long CSV
    out_long = TBL / "loio_calibration.csv"
    with open(out_long, "w", newline="") as f:
        if long_rows:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
            w.writeheader()
            for r in long_rows:
                w.writerow(r)
    print(f"Wrote {out_long} ({len(long_rows)} rows)")

    # Summary CSV: aggregated across folds per (target, model_label, pair)
    summary_rows = []
    by_key = {}
    for r in long_rows:
        k = (r["target"], r["model_label"], r["pair"])
        by_key.setdefault(k, []).append(r)
    for k, rs in sorted(by_key.items()):
        def col(name):
            vs = [r[name] for r in rs if isinstance(r[name], (int, float))
                  and not (isinstance(r[name], float) and np.isnan(r[name]))]
            return vs

        def stats(vs):
            if not vs:
                return {"n": 0, "mean": "", "median": "", "std": "", "min": "", "max": ""}
            return {
                "n": len(vs), "mean": float(np.mean(vs)),
                "median": float(np.median(vs)),
                "std": float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
                "min": float(min(vs)), "max": float(max(vs)),
            }
        row = {"target": k[0], "model_label": k[1], "pair": k[2]}
        for name in ("brier", "ece_10bin", "roc_auc", "pr_auc"):
            s = stats(col(name))
            row[f"{name}_mean"] = s["mean"]
            row[f"{name}_median"] = s["median"]
            row[f"{name}_std"] = s["std"]
        row["n_folds"] = len(rs)
        summary_rows.append(row)

    out_sum = TBL / "loio_calibration_summary.csv"
    with open(out_sum, "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"Wrote {out_sum} ({len(summary_rows)} rows)")

    # Reliability diagrams, one per (target, model_label, pair),
    # averaged across folds
    print(f"Rendering reliability diagrams for {len(bucket)} (target, model, pair) tuples...")
    for (target, model_label, pair), batches in bucket.items():
        # Pool all (proba, y) across folds for this key
        p_all = np.concatenate([b[0] for b in batches])
        y_all = np.concatenate([b[1] for b in batches])
        bins = reliability_bins(y_all, p_all, n_bins=10)
        means = [b[0] for b in bins]
        fracs = [b[1] for b in bins]
        counts = [b[2] for b in bins]
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1,
                label="perfect calibration")
        valid = [i for i, c in enumerate(counts) if c > 0]
        if valid:
            x = [means[i] for i in valid]
            y = [fracs[i] for i in valid]
            sizes = [max(20, 200 * counts[i] / max(counts)) for i in valid]
            ax.scatter(x, y, s=sizes, alpha=0.6, edgecolor="k",
                       linewidth=0.5)
            ax.plot(x, y, color="C0", linewidth=1.2)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive fraction")
        ax.set_title(f"{target} / {model_label} / {pair}\n"
                     f"(pooled over {len(batches)} LOIO folds, n={len(y_all)})")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = FIG / f"reliability_{target}_{model_label}_{pair}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
    print(f"Wrote {len(bucket)} reliability diagrams to {FIG}")


if __name__ == "__main__":
    main()
