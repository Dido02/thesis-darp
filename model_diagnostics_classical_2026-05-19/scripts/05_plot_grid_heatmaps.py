#!/usr/bin/env python3
"""Plot depth x K heatmaps (4-panel: test mean, test std, test min, train-test gap)
for tree and forest grids on the USED 9 non-degenerate pairs.

Inputs:
  tables/lift_distribution_used.csv  (after running 03_lift_distribution.py)

Outputs:
  figures/grid_heatmap_<target>_<tree|forest>.png   (4-panel)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TBL = REPO / "model_diagnostics_classical_2026-05-19" / "tables"
FIG = REPO / "model_diagnostics_classical_2026-05-19" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load():
    return list(csv.DictReader(open(TBL / "lift_distribution_used.csv")))


def stat(rows, variant, split, metric, what):
    cand = [r for r in rows if r["variant"] == variant and r["split"] == split and r["metric"] == metric]
    if not cand:
        return float("nan")
    try:
        return float(cand[0][what])
    except (ValueError, KeyError):
        return float("nan")


def plot_grid(rows, target, model_type, lift_metric: str):
    depths = ["2", "4", "8", "16", "none"]
    Ks = [3, 5, 8, 12, 20]
    test_mean = np.full((len(depths), len(Ks)), np.nan)
    test_std = np.full((len(depths), len(Ks)), np.nan)
    test_min = np.full((len(depths), len(Ks)), np.nan)
    gap = np.full((len(depths), len(Ks)), np.nan)
    for i, d in enumerate(depths):
        for j, K in enumerate(Ks):
            v = f"grid_{target}_{model_type}_depth{d}_K{K}"
            tm = stat(rows, v, "test", lift_metric, "mean")
            ts = stat(rows, v, "test", lift_metric, "std")
            mn = stat(rows, v, "test", lift_metric, "min")
            tr = stat(rows, v, "train", lift_metric, "mean")
            test_mean[i, j] = tm
            test_std[i, j] = ts
            test_min[i, j] = mn
            gap[i, j] = (tr - tm) if not (np.isnan(tr) or np.isnan(tm)) else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    title_metric = "F1-lift" if lift_metric == "f1_lift" else "MAE-lift"
    fig.suptitle(
        f"Grid heatmaps — {target} {model_type} (USED 9, n=9 non-degenerate per cell)",
        fontsize=13, weight="bold")

    panels = [
        ("test mean (higher = better)", test_mean, "viridis", None),
        ("test std (lower = more consistent)", test_std, "magma_r", None),
        ("test min (worst-pair; higher = safer)", test_min, "RdYlGn", None),
        ("train-test gap (lower = less overfit)", gap, "RdYlGn_r", None),
    ]
    for ax, (title, mat, cmap, _) in zip(axes.flat, panels):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels([f"K={k}" for k in Ks])
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"d={d}" for d in depths])
        for i in range(len(depths)):
            for j in range(len(Ks)):
                if not np.isnan(mat[i, j]):
                    txt = f"{mat[i,j]:+.3f}" if lift_metric == "f1_lift" else f"{mat[i,j]:+.4f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
        ax.set_title(f"{title} ({title_metric})", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG / f"grid_heatmap_{target}_{model_type}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    rows = load()
    print(f"Loaded {len(rows)} distribution rows")
    for target, lift in (("t02", "f1_lift"), ("t03", "f1_lift"), ("t05", "f1_lift"),
                         ("t07", "f1_lift"), ("regr", "mae_lift")):
        for model_type in ("tree", "forest"):
            plot_grid(rows, target, model_type, lift)


if __name__ == "__main__":
    main()
