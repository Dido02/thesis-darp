#!/usr/bin/env python3
"""Figures 1, 2, 3, 5 — chapter + appendix heatmaps for the grid sweep.

Common settings:
  Depth values: {4, 8, 16, None} on the x-axis
  K values:     {4, 8, 16, 20}   on the y-axis (top-K features by Gini)
  Cell annotations: 3 sig figs for accuracy/MCC, 2 decimals for regret %
  Color scheme: viridis (higher=better) for top-K/MCC;
                Reds (higher=worse, intuitive warning hue) for regret.
  300 DPI, sans-serif, white background, tight layout.

Read-only on cross_pair_ranking_grid_2026-05-25/tables/ranking_grid_aggregate.csv.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
TBL = REPO / "cross_pair_ranking_grid_2026-05-25" / "tables"
FIG = REPO / "cross_pair_ranking_grid_2026-05-25" / "figures"

DEPTHS = [4, 8, 16, None]
KS     = [4, 8, 16, 20]
DEPTH_LABELS = ["None" if d is None else str(d) for d in DEPTHS]

plt.rcParams["font.family"]      = "sans-serif"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["figure.facecolor"] = "white"


# ============================================================
# Matrix builder + heatmap helpers
# ============================================================
def build_matrix(agg, protocol, kind, metric):
    """Build a (len(KS), len(DEPTHS)) matrix of `metric` values for rows
    matching protocol and starts-with `kind`_d (forest_d... or tree_d...)."""
    mat = np.full((len(KS), len(DEPTHS)), np.nan)
    for _, r in agg.iterrows():
        if r["protocol"] != protocol: continue
        if not r["model"].startswith(kind + "_d"): continue
        parts = r["model"].split("_")    # ['forest','dNone','K20'] or ['tree','d4','K4']
        if len(parts) != 3: continue
        try:
            d_str = parts[1][1:]
            K = int(parts[2][1:])
            d = None if d_str == "None" else int(d_str)
        except (ValueError, IndexError):
            continue
        if d not in DEPTHS or K not in KS: continue
        i = KS.index(K); j = DEPTHS.index(d)
        v = r[metric]
        if pd.notna(v):
            mat[i][j] = v
    return mat


def draw_heatmap(ax, mat, title, *, cmap, fmt_fn, vmin=None, vmax=None):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_yticks(range(len(KS)))
    ax.set_yticklabels([str(k) for k in KS])
    ax.set_xlabel("depth")
    ax.set_ylabel("K (top-K features by Gini)")
    for i in range(len(KS)):
        for j in range(len(DEPTHS)):
            v = mat[i][j]
            if not np.isnan(v):
                ax.text(j, i, fmt_fn(v), ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title(title, fontsize=10)
    return im


# Annotation formatters per spec
fmt_3sig = lambda v: f"{v:.3f}"     # 3 sig figs for accuracy/MCC values in [0, 1]
fmt_2pct = lambda v: f"{v:.2f}"     # 2 decimals for regret % values


# ============================================================
# Figure 1 — forest_grid_5fold (1x3, independent colorbars)
# ============================================================
def fig1_forest_5fold(agg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    m1 = build_matrix(agg, "5fold", "forest", "top1")
    m2 = build_matrix(agg, "5fold", "forest", "top3")
    m3 = build_matrix(agg, "5fold", "forest", "rel_reg")
    im1 = draw_heatmap(axes[0], m1, "Forest top-1, 5-fold",
                       cmap="viridis", fmt_fn=fmt_3sig)
    plt.colorbar(im1, ax=axes[0])
    im2 = draw_heatmap(axes[1], m2, "Forest top-3, 5-fold",
                       cmap="viridis", fmt_fn=fmt_3sig)
    plt.colorbar(im2, ax=axes[1])
    im3 = draw_heatmap(axes[2], m3, "Forest rel regret (%), 5-fold",
                       cmap="Reds", fmt_fn=fmt_2pct)
    plt.colorbar(im3, ax=axes[2])
    fig.tight_layout()
    out_png = FIG / "forest_grid_5fold.png"
    out_pdf = FIG / "forest_grid_5fold.pdf"
    fig.savefig(out_png, dpi=300); fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


# ============================================================
# Figure 2 — forests_vs_trees_5fold (1x2, SHARED colorbar)
# ============================================================
def fig2_forests_vs_trees_5fold(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m_f = build_matrix(agg, "5fold", "forest", "top1")
    m_t = build_matrix(agg, "5fold", "tree",   "top1")
    # Shared scale spanning both matrices for fair visual comparison
    vals = np.concatenate([m_f[~np.isnan(m_f)], m_t[~np.isnan(m_t)]])
    vmin, vmax = float(vals.min()), float(vals.max())
    draw_heatmap(axes[0], m_f, "Forest top-1, 5-fold",
                 cmap="viridis", fmt_fn=fmt_3sig, vmin=vmin, vmax=vmax)
    im2 = draw_heatmap(axes[1], m_t, "Tree top-1, 5-fold",
                       cmap="viridis", fmt_fn=fmt_3sig, vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=list(axes), shrink=0.8, location="right",
                 label=f"top-1 (shared, [{vmin:.3f}, {vmax:.3f}])")
    out_png = FIG / "forests_vs_trees_5fold.png"
    out_pdf = FIG / "forests_vs_trees_5fold.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


# ============================================================
# Figure 3 — protocol_comparison_forest_top1 (1x2, SHARED colorbar)
# ============================================================
def fig3_protocol_comparison_forest_top1(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m_75 = build_matrix(agg, "deployed_75_25", "forest", "top1")
    m_5f = build_matrix(agg, "5fold",          "forest", "top1")
    vals = np.concatenate([m_75[~np.isnan(m_75)], m_5f[~np.isnan(m_5f)]])
    vmin, vmax = float(vals.min()), float(vals.max())
    draw_heatmap(axes[0], m_75, "Forest top-1, 75/25 split",
                 cmap="viridis", fmt_fn=fmt_3sig, vmin=vmin, vmax=vmax)
    im2 = draw_heatmap(axes[1], m_5f, "Forest top-1, 5-fold",
                       cmap="viridis", fmt_fn=fmt_3sig, vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=list(axes), shrink=0.8, location="right",
                 label=f"top-1 (shared, [{vmin:.3f}, {vmax:.3f}])")
    out_png = FIG / "protocol_comparison_forest_top1.png"
    out_pdf = FIG / "protocol_comparison_forest_top1.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


# ============================================================
# Figure 5 — tree_grid_appendix (2x3, independent colorbars)
# ============================================================
def fig5_tree_grid_appendix(agg):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    proto_label = {"deployed_75_25": "75/25 split", "5fold": "5-fold"}
    for row, proto in enumerate(["deployed_75_25", "5fold"]):
        m_t1 = build_matrix(agg, proto, "tree", "top1")
        m_t3 = build_matrix(agg, proto, "tree", "top3")
        m_rr = build_matrix(agg, proto, "tree", "rel_reg")
        im1 = draw_heatmap(axes[row, 0], m_t1, f"Tree top-1, {proto_label[proto]}",
                           cmap="viridis", fmt_fn=fmt_3sig)
        plt.colorbar(im1, ax=axes[row, 0])
        im2 = draw_heatmap(axes[row, 1], m_t3, f"Tree top-3, {proto_label[proto]}",
                           cmap="viridis", fmt_fn=fmt_3sig)
        plt.colorbar(im2, ax=axes[row, 1])
        im3 = draw_heatmap(axes[row, 2], m_rr, f"Tree rel regret (%), {proto_label[proto]}",
                           cmap="Reds", fmt_fn=fmt_2pct)
        plt.colorbar(im3, ax=axes[row, 2])
    fig.tight_layout()
    out_png = FIG / "tree_grid_appendix.png"
    out_pdf = FIG / "tree_grid_appendix.pdf"
    fig.savefig(out_png, dpi=300); fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


# ============================================================
# Main
# ============================================================
def main():
    agg = pd.read_csv(TBL / "ranking_grid_aggregate.csv")
    print(f"Loaded {len(agg)} rows × {len(agg.columns)} cols from ranking_grid_aggregate.csv")
    print(f"  unique models: {agg['model'].nunique()}, "
          f"protocols: {sorted(agg['protocol'].unique())}")

    print("\nFigure 1 — forest_grid_5fold (1x3, independent colorbars)...")
    p1png, p1pdf = fig1_forest_5fold(agg)
    print(f"  {p1png}\n  {p1pdf}")

    print("\nFigure 2 — forests_vs_trees_5fold (1x2, SHARED colorbar)...")
    p2png, p2pdf = fig2_forests_vs_trees_5fold(agg)
    print(f"  {p2png}\n  {p2pdf}")

    print("\nFigure 3 — protocol_comparison_forest_top1 (1x2, SHARED colorbar)...")
    p3png, p3pdf = fig3_protocol_comparison_forest_top1(agg)
    print(f"  {p3png}\n  {p3pdf}")

    print("\nFigure 5 — tree_grid_appendix (2x3, independent colorbars)...")
    p5png, p5pdf = fig5_tree_grid_appendix(agg)
    print(f"  {p5png}\n  {p5pdf}")


if __name__ == "__main__":
    main()
