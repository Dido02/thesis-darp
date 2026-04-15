"""Visualize trial delta distributions per operator combo using box plots."""
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from parse_logs import parse_log
from output_utils import get_output_path


def main(log_path: str):
    data = parse_log(log_path)
    trials = data["trials"]
    config = data["config"]
    d_names = config.destroy_ops
    r_names = config.repair_ops
    nd, nr = len(d_names), len(r_names)

    if not trials:
        print("No trial records found.")
        return

    # Collect all deltas per (d,r) combo
    all_deltas = defaultdict(list)
    for tr in trials:
        for (di, ri), deltas in tr.deltas.items():
            all_deltas[(di, ri)].extend(deltas)

    fig, axes = plt.subplots(nd, 1, figsize=(14, 3 * nd), sharex=True)
    if nd == 1:
        axes = [axes]

    for di in range(nd):
        ax = axes[di]
        box_data = []
        labels = []
        for ri in range(nr):
            key = (di, ri)
            if key in all_deltas:
                vals = all_deltas[key]
                non_zero = [v for v in vals if v != 0]
                box_data.append(non_zero if non_zero else [0])
                labels.append(r_names[ri])

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
            colors = plt.cm.Set2(np.linspace(0, 1, nr))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_ylabel("Delta")
        ax.set_title(f"Destroy: {d_names[di]}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Repair operator")
    fig.suptitle(f"Trial Delta Distributions (non-zero) - {config.instance.split('/')[-1]}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = get_output_path(log_path, "_trial_dist.png", "plots")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trial_distributions.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
