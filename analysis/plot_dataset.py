"""Generate the same 5 plots as data_exploration.ipynb but parameterized on
input cache + output figures dirs. Read-only.

Usage:
    python analysis/plot_dataset.py <cache_dir> <figures_dir> <label>
        cache_dir   : contains features.parquet + labels.parquet
        figures_dir : where to write *.png
        label       : short string used in plot titles (e.g. "v2-only ≤02min")
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Make analysis/helpers and ml_train/darpml importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ml_train"))

from helpers import (
    BATCH_COLORS, DESTROY_COLORS, DESTROY_OPS, REPAIR_OPS,
    IMPROVEMENT_FREQ_THRESHOLD,
    feature_batch_map, pair_freq_columns, pair_label_columns,
    pair_label_to_name, parse_tier,
)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    cache_dir = Path(sys.argv[1])
    figures_dir = Path(sys.argv[2])
    label = sys.argv[3]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache:   {cache_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Label:   {label}")

    features_df = pd.read_parquet(cache_dir / "features.parquet")
    labels_df = pd.read_parquet(cache_dir / "labels.parquet")
    labels_df["tier"] = labels_df["log_file"].apply(parse_tier)
    features_df["tier"] = features_df["log_file"].apply(parse_tier)

    print(f"\nLoaded {len(features_df):,} feature rows, {len(labels_df):,} labeled rows, "
          f"{labels_df['log_file'].nunique()} unique logs.")
    print("Per-tier states (labels):")
    print(labels_df["tier"].value_counts().sort_index().to_string())

    # ----------------------- Section 2 -----------------------
    pair_cols = pair_label_columns(labels_df)
    pos_rates = labels_df[pair_cols].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bar_labels, heights, colors = [], [], []
    for c in pair_cols:
        d, r = pair_label_to_name(c)
        bar_labels.append(f"{d}+{r}")
        heights.append(pos_rates[c])
        colors.append(DESTROY_COLORS[d])
    x = np.arange(len(bar_labels))
    ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.05, color="grey", lw=0.8, ls=":")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Positive label rate")
    ax.set_title(f"[{label}] Per-pair positive label rate (label = improvement_freq >= 0.3)")
    ax.set_ylim(0, 1.0)
    legend = [Patch(facecolor=DESTROY_COLORS[d], label=d) for d in DESTROY_OPS]
    ax.legend(handles=legend, loc="upper left", fontsize=8, title="destroy op")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_per_pair_positive_rate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------- Section 3 -----------------------
    fig, axes = plt.subplots(5, 4, figsize=(13, 14), sharex=True, sharey=False)
    for di, d in enumerate(DESTROY_OPS):
        for ri, r in enumerate(REPAIR_OPS):
            col = f"improve_freq_{di}_{ri}"
            ax = axes[di, ri]
            if col not in labels_df.columns:
                ax.set_visible(False)
                continue
            vals = labels_df[col].values
            ax.hist(vals, bins=21, range=(0, 1.0),
                    color=DESTROY_COLORS[d], edgecolor="black", linewidth=0.4)
            ax.axvline(IMPROVEMENT_FREQ_THRESHOLD, color="red", lw=1.2)
            pos_rate = (vals >= IMPROVEMENT_FREQ_THRESHOLD).mean()
            ax.set_title(f"{d}+{r}\npos_rate = {pos_rate:.2%}", fontsize=9)
            ax.set_xlim(0, 1.0)
            if di == 4: ax.set_xlabel("improvement_freq")
            if ri == 0: ax.set_ylabel("# states")
    fig.suptitle(
        f"[{label}] improve_freq distribution per pair  (binarization threshold = 0.3, red line)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "03_improvement_freq_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------- Section 4 -----------------------
    labels_df["improvement_count"] = labels_df[pair_cols].sum(axis=1)
    ic = labels_df["improvement_count"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(ic, bins=np.arange(-0.5, 21.5, 1.0),
             color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("# pairs labeled positive at this state (out of 20)")
    ax1.set_ylabel("# states")
    ax1.set_title(f"[{label}] State-level improvement-options")
    ax1.axvline(ic.mean(), color="red", ls="--", lw=1.2, label=f"mean={ic.mean():.1f}")
    ax1.axvline(ic.median(), color="orange", ls="--", lw=1.2, label=f"median={ic.median():.0f}")
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(0, 21))
    tier_order = ["30s", "01min", "02min", "05min", "15min"]
    for t in tier_order:
        sub = labels_df.loc[labels_df["tier"] == t, "improvement_count"]
        if len(sub) == 0: continue
        ax2.hist(sub, bins=np.arange(-0.5, 21.5, 1.0),
                 alpha=0.4, label=f"{t} (n={len(sub):,})", density=True)
    ax2.set_xlabel("# pairs labeled positive")
    ax2.set_ylabel("density")
    ax2.set_title("Stratified by instance tier")
    ax2.legend(fontsize=8)
    ax2.set_xticks(range(0, 21, 2))
    fig.tight_layout()
    fig.savefig(figures_dir / "04_improvement_options_distribution.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------- Section 5a (feature distributions) -----------------------
    batch_map = feature_batch_map()
    feat_cols = [c for c in features_df.columns if c in batch_map]
    batch_order = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]
    feat_cols_sorted = sorted(
        feat_cols,
        key=lambda c: (batch_order.index(batch_map[c]) if batch_map[c] in batch_order else 99, c),
    )
    n = len(feat_cols_sorted)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(feat_cols_sorted):
        ax = axes[i]
        b = batch_map[c]
        ax.hist(features_df[c].values, bins=40, color=BATCH_COLORS.get(b, "grey"),
                edgecolor="black", linewidth=0.3)
        ax.set_title(f"{c}\n(batch: {b})", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    legend = [Patch(facecolor=BATCH_COLORS[b], label=b) for b in batch_order]
    fig.legend(handles=legend, loc="upper center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"[{label}] Feature distributions (color = batch)", fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(figures_dir / "05a_feature_distributions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ----------------------- Section 5b (correlation) -----------------------
    corr = features_df[feat_cols_sorted].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(feat_cols_sorted)))
    ax.set_yticks(range(len(feat_cols_sorted)))
    ax.set_xticklabels(feat_cols_sorted, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(feat_cols_sorted, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson r")
    ax.set_title(f"[{label}] Feature correlation matrix")
    last_b = None
    for i, c in enumerate(feat_cols_sorted):
        b = batch_map[c]
        if b != last_b and last_b is not None:
            ax.axhline(i - 0.5, color="black", lw=0.7)
            ax.axvline(i - 0.5, color="black", lw=0.7)
        last_b = b
    fig.tight_layout()
    fig.savefig(figures_dir / "05b_feature_correlation_heatmap.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved 5 figures to {figures_dir}")


if __name__ == "__main__":
    main()
