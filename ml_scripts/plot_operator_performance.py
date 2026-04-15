"""Analyze operator performance: heatmaps from trial data (no state records needed)."""
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from parse_logs import parse_log
from output_utils import get_output_path


def main(log_path: str):
    data = parse_log(log_path)
    trials = data["trials"]
    states = data["states"]
    config = data["config"]
    d_names = config.destroy_ops
    r_names = config.repair_ops
    nd, nr = len(d_names), len(r_names)

    # If we have state records, use them for selection/acceptance stats
    has_states = len(states) > 0

    # --- From trials: average min-delta and mean-delta per combo ---
    mean_deltas = defaultdict(list)
    min_deltas = defaultdict(list)
    zero_frac = defaultdict(list)

    for tr in trials:
        for (di, ri), deltas in tr.deltas.items():
            mean_deltas[(di, ri)].append(np.mean(deltas))
            min_deltas[(di, ri)].append(min(deltas))
            zero_frac[(di, ri)].append(sum(1 for d in deltas if d == 0) / len(deltas))

    avg_mean = np.full((nd, nr), np.nan)
    avg_min = np.full((nd, nr), np.nan)
    avg_zero = np.full((nd, nr), np.nan)

    for di in range(nd):
        for ri in range(nr):
            key = (di, ri)
            if key in mean_deltas:
                avg_mean[di, ri] = np.mean(mean_deltas[key])
                avg_min[di, ri] = np.mean(min_deltas[key])
                avg_zero[di, ri] = np.mean(zero_frac[key])

    if has_states:
        # Selection count and acceptance from states
        count = np.zeros((nd, nr))
        accept_count = np.zeros((nd, nr))
        for s in states:
            count[s.d_idx, s.r_idx] += 1
            if s.accept:
                accept_count[s.d_idx, s.r_idx] += 1
        accept_rate = np.divide(accept_count, count, where=count > 0, out=np.full_like(count, np.nan))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        panels = [
            (axes[0, 0], count, "Selection Count", "YlOrRd", None),
            (axes[0, 1], avg_mean, "Avg Trial Mean Delta", "RdYlGn_r", None),
            (axes[1, 0], accept_rate, "Acceptance Rate (states)", "YlGn", (0, 1)),
            (axes[1, 1], avg_min, "Avg Trial Best Delta", "RdYlGn_r", None),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        panels = [
            (axes[0], avg_mean, "Avg Trial Mean Delta", "RdYlGn_r", None),
            (axes[1], avg_min, "Avg Trial Best Delta", "RdYlGn_r", None),
            (axes[2], avg_zero, "Avg Zero-Delta Fraction", "YlOrRd", (0, 1)),
        ]

    for ax, data_mat, title, cmap, vlim in panels:
        kwargs = {"cmap": cmap, "aspect": "auto"}
        if vlim:
            kwargs["vmin"], kwargs["vmax"] = vlim
        im = ax.imshow(data_mat, **kwargs)
        ax.set_xticks(range(nr)); ax.set_xticklabels(r_names, rotation=45, ha="right")
        ax.set_yticks(range(nd)); ax.set_yticklabels(d_names)
        for i in range(nd):
            for j in range(nr):
                v = data_mat[i, j]
                if not np.isnan(v):
                    fmt = f"{int(v)}" if v > 10 else f"{v:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Operator Performance - {config.instance.split('/')[-1]}", fontsize=14)
    plt.tight_layout()
    out = get_output_path(log_path, "_operators.png", "plots")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_operator_performance.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
