"""Feature importance + F1 lift visualizer for the three candidate models
(5f / 13f / 20f). Loads training_report.json for summary metrics and the
per-pair .joblib pickles for per-pair feature_importances_.

Usage:
    python analysis/plot_model_importance.py \
        <model_dir_5f> <model_dir_13f> <model_dir_20f> \
        <figures_dir> <label>

Example (after training on v2 <=02min):
    python analysis/plot_model_importance.py \
        ml_train/models_v2_le02_5f \
        ml_train/models_v2_le02_13f \
        ml_train/models_v2_le02_20f \
        analysis/figures_models_v2_le02 \
        "v2-only, <=02min"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from helpers import (
    BATCH_COLORS, DESTROY_OPS, REPAIR_OPS,
    feature_batch_map, pair_label_to_name,
)


def load_candidate(model_dir: Path):
    """Return (report_dict, per_pair_importance_matrix, pair_names, feat_names)."""
    report = json.loads((model_dir / "training_report.json").read_text())
    meta = json.loads((model_dir / "meta.json").read_text())
    feat_names = meta["feature_names"]

    pair_cols = sorted(m["pair_name"] for m in report["pair_metrics"])
    mat = np.zeros((len(pair_cols), len(feat_names)))
    for i, pair in enumerate(pair_cols):
        clf = joblib.load(model_dir / f"rf_{pair}.joblib")
        mat[i, :] = clf.feature_importances_
    return report, mat, pair_cols, feat_names


def pair_nice_name(pair_col: str) -> str:
    d, r = pair_label_to_name(pair_col)
    return f"{d}+{r}"


def plot_aggregate_bars(candidates, figures_dir: Path, label: str):
    """Bar chart of mean feature importance per candidate."""
    fig, axes = plt.subplots(1, len(candidates), figsize=(5.5 * len(candidates), 4.5),
                             sharey=False)
    if len(candidates) == 1:
        axes = [axes]
    batch_map = feature_batch_map()
    for ax, (cand_label, model_dir, report, mat, pair_cols, feat_names) in zip(axes, candidates):
        mean_imp = mat.mean(axis=0)
        order = np.argsort(mean_imp)[::-1]
        names_sorted = [feat_names[i] for i in order]
        imp_sorted = mean_imp[order]
        colors = [BATCH_COLORS.get(batch_map.get(n, ""), "grey") for n in names_sorted]
        y = np.arange(len(names_sorted))
        ax.barh(y, imp_sorted, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(names_sorted, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("mean importance (across 20 pairs)")
        ax.set_title(f"{cand_label}: {len(feat_names)} features")
    fig.suptitle(f"[{label}] Aggregate feature importance per candidate", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_aggregate_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_pair_heatmaps(candidates, figures_dir: Path, label: str):
    """One heatmap per candidate: rows=20 pairs, cols=features, colored by importance."""
    for cand_label, model_dir, report, mat, pair_cols, feat_names in candidates:
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(feat_names) + 3), 7))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels(feat_names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(pair_cols)))
        ax.set_yticklabels([pair_nice_name(p) for p in pair_cols], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="RF feature importance")
        ax.set_title(f"[{label}] {cand_label}: per-pair feature importance "
                     f"({len(pair_cols)} pairs × {len(feat_names)} features)")
        fig.tight_layout()
        fig.savefig(figures_dir / f"m2_per_pair_heatmap_{cand_label}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_f1_lift_comparison(candidates, figures_dir: Path, label: str):
    """Grouped bar chart: F1 lift per pair, 3 bars per pair (5f/13f/20f)."""
    pair_order = sorted({m["pair_name"]
                          for _, _, r, _, _, _ in candidates
                          for m in r["pair_metrics"]})
    n_pairs = len(pair_order)
    n_cand = len(candidates)
    width = 0.8 / n_cand

    fig, (ax_lift, ax_pos) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    x = np.arange(n_pairs)
    cand_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    pos_rates = None
    for ci, (cand_label, _, report, _, _, _) in enumerate(candidates):
        metrics_map = {m["pair_name"]: m for m in report["pair_metrics"]}
        lifts = [metrics_map[p]["f1_lift"] for p in pair_order]
        ax_lift.bar(x + (ci - (n_cand - 1) / 2) * width, lifts, width,
                    label=cand_label, color=cand_colors[ci % len(cand_colors)],
                    edgecolor="black", linewidth=0.3)
        if ci == 0:
            pos_rates = [metrics_map[p]["n_positive_test"] / max(metrics_map[p]["n_test"], 1)
                         for p in pair_order]

    ax_lift.axhline(0, color="black", lw=0.6)
    ax_lift.axhline(0.01, color="green", lw=0.6, ls=":", label="+0.01 threshold")
    ax_lift.axhline(-0.01, color="red", lw=0.6, ls=":")
    ax_lift.set_ylabel("F1 lift over best constant baseline")
    ax_lift.set_title(f"[{label}] Per-pair F1 lift across candidates")
    ax_lift.legend(loc="upper right", fontsize=8)
    ax_lift.grid(axis="y", alpha=0.3)

    bar_colors = ["#cccccc" if (0.05 <= pr <= 0.95) else "#e06666" for pr in pos_rates]
    ax_pos.bar(x, pos_rates, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax_pos.axhline(0.05, color="grey", ls=":", lw=0.8)
    ax_pos.axhline(0.95, color="grey", ls=":", lw=0.8)
    ax_pos.set_ylabel("test pos rate")
    ax_pos.set_xticks(x)
    ax_pos.set_xticklabels([pair_nice_name(p) for p in pair_order],
                            rotation=60, ha="right", fontsize=8)
    ax_pos.set_ylim(0, 1.0)
    ax_pos.set_title("Per-pair positive-label rate on test set "
                     "(red = degenerate, outside [0.05, 0.95])")

    fig.tight_layout()
    fig.savefig(figures_dir / "m3_f1_lift_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(candidates, figures_dir: Path, label: str):
    """Text table summarizing the 3 candidates."""
    lines = [f"=== [{label}] Candidate summary ===", ""]
    lines.append(f"{'candidate':>10s} {'#feat':>6s} {'#pairs':>7s} "
                 f"{'non_degen':>10s} {'median_lift':>12s} "
                 f"{'+lift':>6s} {'~0':>4s} {'-lift':>6s}")
    for cand_label, model_dir, report, mat, pair_cols, feat_names in candidates:
        ls = report["lift_summary"]
        lines.append(f"{cand_label:>10s} {len(feat_names):>6d} "
                     f"{report['n_pairs']:>7d} "
                     f"{ls['n_non_degenerate']:>10d} "
                     f"{ls['median_f1_lift']:>+12.4f} "
                     f"{ls['n_positive_lift']:>6d} "
                     f"{ls['n_zero_lift']:>4d} "
                     f"{ls['n_negative_lift']:>6d}")
    text = "\n".join(lines)
    print("\n" + text)
    (figures_dir / "m0_summary.txt").write_text(text + "\n")


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    dirs = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])]
    figures_dir = Path(sys.argv[4])
    label = sys.argv[5]
    figures_dir.mkdir(parents=True, exist_ok=True)

    cand_labels = ["5f", "13f", "20f"]
    candidates = []
    for cl, d in zip(cand_labels, dirs):
        print(f"Loading {cl} from {d} ...")
        report, mat, pairs, feats = load_candidate(d)
        candidates.append((cl, d, report, mat, pairs, feats))

    plot_summary_table(candidates, figures_dir, label)
    plot_aggregate_bars(candidates, figures_dir, label)
    plot_per_pair_heatmaps(candidates, figures_dir, label)
    plot_f1_lift_comparison(candidates, figures_dir, label)

    print(f"\nSaved figures to {figures_dir}")


if __name__ == "__main__":
    main()
