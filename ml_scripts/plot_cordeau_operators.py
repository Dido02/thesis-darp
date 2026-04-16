"""Plot operator performance from Cordeau static results (results_operators/).

Produces heatmaps similar to the dynamic trial analysis but from actual
solution quality data: avg objective, best objective, gap to BKS.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(results_dir: str, iterations: int = 25000):
    results_dir = Path(results_dir)
    raw = pd.read_csv(results_dir / "all_raw_results.csv")
    summary = pd.read_csv(results_dir / f"summary_iter{iterations}.csv")

    # Only single-pair configs
    pairs = raw[(raw["config"].str.startswith("pair_")) & (raw["iterations"] == iterations)].copy()

    d_names = ["random", "worst", "shaw", "rtcritical", "lowslack"]
    r_names = ["greedy", "regret2", "regret3", "slack"]
    nd, nr = len(d_names), len(r_names)

    # Build matrices
    avg_obj = np.full((nd, nr), np.nan)
    best_obj = np.full((nd, nr), np.nan)
    avg_gap = np.full((nd, nr), np.nan)
    best_gap = np.full((nd, nr), np.nan)
    win_count = np.zeros((nd, nr))

    for di, d in enumerate(d_names):
        for ri, r in enumerate(r_names):
            subset = pairs[(pairs["destroy"] == d) & (pairs["repair"] == r)]
            if subset.empty:
                continue
            avg_obj[di, ri] = subset["obj"].mean()
            best_obj[di, ri] = subset["obj"].min()

            # Gaps from summary
            config_name = f"pair_{d}+{r}"
            s = summary[summary["config"] == config_name]
            if not s.empty:
                avg_gap[di, ri] = s["gap_avg_%"].mean()
                best_gap[di, ri] = s["gap_best_%"].mean()

            # Win count: how many (instance, seed) combos does this pair give the best obj?
            for (inst, seed), grp in pairs.groupby(["instance", "seed"]):
                best_for_combo = grp["obj"].min()
                pair_obj = subset[(subset["instance"] == inst) & (subset["seed"] == seed)]["obj"]
                if not pair_obj.empty and pair_obj.iloc[0] <= best_for_combo + 0.01:
                    win_count[di, ri] += 1

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    panels = [
        (axes[0, 0], avg_obj, "Avg Objective (lower=better)", "RdYlGn_r", ".0f"),
        (axes[0, 1], best_obj, "Best Objective (lower=better)", "RdYlGn_r", ".0f"),
        (axes[1, 0], avg_gap, "Avg Gap to BKS % (lower=better)", "RdYlGn_r", ".2f"),
        (axes[1, 1], win_count, "Win Count (higher=better)", "YlGn", ".0f"),
    ]

    for ax, data, title, cmap, fmt in panels:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(nr))
        ax.set_xticklabels(r_names, rotation=45, ha="right")
        ax.set_yticks(range(nd))
        ax.set_yticklabels(d_names)
        for i in range(nd):
            for j in range(nr):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Operator Pair Performance — Cordeau pr01-pr20 ({iterations} iterations, 5 seeds)",
                 fontsize=14)
    plt.tight_layout()

    out = results_dir / f"operator_heatmaps_iter{iterations}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()

    # --- Per-instance ranking plot ---
    fig2, ax2 = plt.subplots(figsize=(16, 8))

    # For each instance, rank the pairs by avg obj
    instances = sorted(pairs["instance"].unique())
    pair_labels = [f"{d}+{r}" for d in d_names for r in r_names]
    rank_matrix = np.full((len(pair_labels), len(instances)), np.nan)

    for ii, inst in enumerate(instances):
        inst_data = pairs[pairs["instance"] == inst]
        pair_avgs = []
        for di, d in enumerate(d_names):
            for ri, r in enumerate(r_names):
                subset = inst_data[(inst_data["destroy"] == d) & (inst_data["repair"] == r)]
                avg = subset["obj"].mean() if not subset.empty else np.inf
                pair_avgs.append(avg)
        # Rank (1 = best)
        sorted_idx = np.argsort(pair_avgs)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(sorted_idx) + 1)
        rank_matrix[:, ii] = ranks

    # Show only top 10 pairs by average rank
    avg_ranks = np.nanmean(rank_matrix, axis=1)
    top_idx = np.argsort(avg_ranks)[:10]

    for idx in top_idx:
        ax2.plot(instances, rank_matrix[idx], "o-", label=f"{pair_labels[idx]} (avg rank {avg_ranks[idx]:.1f})",
                 markersize=4)

    ax2.set_ylabel("Rank (1 = best)")
    ax2.set_xlabel("Instance")
    ax2.set_title(f"Top 10 Operator Pairs by Rank — Cordeau ({iterations} iters)")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    out2 = results_dir / f"operator_rankings_iter{iterations}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()

    # --- ALNS configs vs best single pair ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    multi_configs = ["A_standard", "B1_std+rtcritical", "B2_std+lowslack",
                     "B3_std+rtcrit+lowslack", "C_std+slack_repair", "D_all"]
    multi = raw[(raw["config"].isin(multi_configs)) & (raw["iterations"] == iterations)]

    # Best single pair per instance
    best_pair_per_inst = pairs.groupby("instance")["obj"].min().reset_index()
    best_pair_per_inst.columns = ["instance", "best_pair_obj"]

    configs_to_plot = multi_configs
    x = np.arange(len(instances))
    width = 0.12

    for ci, config in enumerate(configs_to_plot):
        c_data = multi[multi["config"] == config]
        avgs = []
        for inst in instances:
            inst_sub = c_data[c_data["instance"] == inst]
            avgs.append(inst_sub["obj"].mean() if not inst_sub.empty else np.nan)
        ax3.bar(x + ci * width, avgs, width, label=config, alpha=0.8)

    # Overlay best single pair as line
    best_pair_vals = [best_pair_per_inst[best_pair_per_inst["instance"] == inst]["best_pair_obj"].iloc[0]
                      for inst in instances]
    ax3.plot(x + len(configs_to_plot) * width / 2, best_pair_vals, "k--o",
             label="Best single pair", markersize=4, linewidth=1.5)

    ax3.set_xticks(x + len(configs_to_plot) * width / 2)
    ax3.set_xticklabels(instances, rotation=45, ha="right")
    ax3.set_ylabel("Avg Objective")
    ax3.set_title(f"ALNS Configs vs Best Single Pair — Cordeau ({iterations} iters)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out3 = results_dir / f"alns_vs_pairs_iter{iterations}.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")

    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_operators"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    main(results_dir, iterations)
