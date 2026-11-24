#!/usr/bin/env python3
"""State-level top-K selection accuracy + cost-weighted error.

At each test state we have 9 probabilities (one per USED pair) from the LOIO
fold model. Ground truth = argmax(ei_pct_{d,r}) — the pair that would have
produced the biggest objective improvement at that state. We exclude states
with all-zero ei_pct (no improvement was achievable).

Metrics:
  - top1_acc: prob argmax == true_best_pair
  - top3_acc: true_best_pair in top-3 by prob
  - cost_strict:  ei_pct[true_best] if top1 wrong, else 0
  - cost_regret:  ei_pct[true_best] - ei_pct[predicted_top1]  (>= 0)

Random baselines:  top1 = 1/9, top3 = 3/9.

Outputs:
  tables/loio_topk_state.csv   (long: per fold per (target, model_label))
  tables/loio_topk_summary.csv (per (target, model_label))
  figures/topk_by_instance_{target}_{model_label}.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FIG, FOLD_MODELS, MODEL_CONFIGS, TARGETS, TBL, USED_EI_COLS,
    USED_PAIRS,
)


def compute_fold_topk(proba_mat: np.ndarray, ei_mat: np.ndarray):
    """proba_mat, ei_mat: (n_states, 9). Returns dict of aggregated metrics."""
    n_total = proba_mat.shape[0]
    # Exclude states with all-zero ei_pct
    keep = ei_mat.max(axis=1) > 0
    n_excluded = int((~keep).sum())
    if keep.sum() == 0:
        return None
    P = proba_mat[keep]
    E = ei_mat[keep]
    pred_top1 = P.argmax(axis=1)
    true_best = E.argmax(axis=1)
    top1_hit = (pred_top1 == true_best).astype(int)
    # Top-3: argpartition over -P
    pred_top3_idx = np.argpartition(-P, kth=3, axis=1)[:, :3]
    top3_hit = np.zeros(len(P), dtype=int)
    for i in range(len(P)):
        if true_best[i] in pred_top3_idx[i]:
            top3_hit[i] = 1
    # Cost-weighted
    rows = np.arange(len(P))
    ei_true = E[rows, true_best]
    ei_pred = E[rows, pred_top1]
    cost_strict = np.where(top1_hit == 1, 0.0, ei_true)
    cost_regret = ei_true - ei_pred  # >= 0 always
    return {
        "n_states_scored": int(len(P)),
        "n_states_excluded_zero_ei": n_excluded,
        "n_states_total": int(n_total),
        "top1_acc": float(top1_hit.mean()),
        "top3_acc": float(top3_hit.mean()),
        "cost_strict_mean": float(cost_strict.mean()),
        "cost_regret_mean": float(cost_regret.mean()),
        "ei_true_mean": float(ei_true.mean()),
    }


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    raw = json.loads((TBL / "loio_raw_results.json").read_text())
    folds = {f["fold"]: f for f in raw["folds_meta"]}

    long_rows = []
    for entry in raw["results"]:
        target = entry["target"]
        model_label = entry["model_label"]
        fold_idx = entry["fold"]
        instance = entry["instance"]
        fold_meta = folds[fold_idx]
        labels_df = pd.read_parquet(CACHE / TARGETS[target])
        labels_df = labels_df[labels_df["log_file"].isin(fold_meta["test_logs"])]
        fold_dir = (FOLD_MODELS / target / model_label /
                    f"fold{fold_idx:02d}_{instance}")
        proba_path = fold_dir / "state_proba.parquet"
        if not proba_path.exists():
            print(f"[skip] missing {proba_path}")
            continue
        proba_df = pd.read_parquet(proba_path)
        merged = proba_df.merge(labels_df, on=["log_file", "iter"], how="inner")

        proba_cols = [f"proba_{p}" for p in USED_PAIRS]
        if not all(c in merged.columns for c in proba_cols):
            print(f"[skip] missing proba cols in fold {fold_idx} {model_label}")
            continue
        if not all(c in merged.columns for c in USED_EI_COLS):
            print(f"[skip] missing ei_pct cols in fold {fold_idx} {target}")
            continue

        proba_mat = merged[proba_cols].values
        ei_mat = merged[USED_EI_COLS].values
        result = compute_fold_topk(proba_mat, ei_mat)
        if result is None:
            print(f"[skip] all states zero-ei: fold {fold_idx} {target}")
            continue
        long_rows.append({
            "target": target, "model_label": model_label,
            "fold": fold_idx, "instance": instance,
            "n_test_logs": len(fold_meta["test_logs"]),
            **result,
        })

    # Long CSV
    out_long = TBL / "loio_topk_state.csv"
    with open(out_long, "w", newline="") as f:
        if long_rows:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
            w.writeheader()
            for r in long_rows:
                w.writerow(r)
    print(f"Wrote {out_long} ({len(long_rows)} rows)")

    # Summary per (target, model_label)
    summary_rows = []
    by_key = {}
    for r in long_rows:
        by_key.setdefault((r["target"], r["model_label"]), []).append(r)
    for (target, model_label), rs in sorted(by_key.items()):
        row = {
            "target": target, "model_label": model_label,
            "n_folds": len(rs),
            "random_top1_baseline": 1.0 / 9,
            "random_top3_baseline": 3.0 / 9,
        }
        for name in ("top1_acc", "top3_acc", "cost_strict_mean",
                     "cost_regret_mean", "ei_true_mean"):
            vs = [r[name] for r in rs]
            row[f"{name}_mean"] = float(np.mean(vs))
            row[f"{name}_median"] = float(np.median(vs))
            row[f"{name}_p10"] = float(np.percentile(vs, 10))
            row[f"{name}_p90"] = float(np.percentile(vs, 90))
            row[f"{name}_std"] = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
        summary_rows.append(row)

    out_sum = TBL / "loio_topk_summary.csv"
    with open(out_sum, "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
    print(f"Wrote {out_sum} ({len(summary_rows)} rows)")

    # Per-instance bar charts
    print("Rendering per-instance top-K bars...")
    for (target, model_label), rs in by_key.items():
        rs_sorted = sorted(rs, key=lambda r: r["instance"])
        instances = [r["instance"] for r in rs_sorted]
        top1 = [r["top1_acc"] for r in rs_sorted]
        top3 = [r["top3_acc"] for r in rs_sorted]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(instances) + 1.5))
        ypos = np.arange(len(instances))
        ax.barh(ypos - 0.18, top1, height=0.36, label="top-1", color="C0")
        ax.barh(ypos + 0.18, top3, height=0.36, label="top-3", color="C1")
        ax.axvline(1.0 / 9, color="C0", linestyle="--", linewidth=1,
                   label="random top-1 = 1/9")
        ax.axvline(3.0 / 9, color="C1", linestyle="--", linewidth=1,
                   label="random top-3 = 3/9")
        mean_top1 = float(np.mean(top1))
        mean_top3 = float(np.mean(top3))
        ax.axvline(mean_top1, color="C0", linestyle=":", linewidth=1.5,
                   label=f"mean top-1 = {mean_top1:.3f}")
        ax.axvline(mean_top3, color="C1", linestyle=":", linewidth=1.5,
                   label=f"mean top-3 = {mean_top3:.3f}")
        ax.set_yticks(ypos)
        ax.set_yticklabels(instances, fontsize=8)
        ax.set_xlabel("Selection accuracy")
        ax.set_xlim(0, 1)
        ax.set_title(f"{target} / {model_label}\nstate-level top-K selection (per LOIO fold)")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        out = FIG / f"topk_by_instance_{target}_{model_label}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
    print(f"Wrote {len(by_key)} per-instance bar charts")


if __name__ == "__main__":
    main()
