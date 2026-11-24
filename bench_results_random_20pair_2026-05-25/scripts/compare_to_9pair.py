#!/usr/bin/env python3
"""Compare random+20-pair vs random+9-pair, paired by (instance, delay, budget, seed).

20-pair source: bench_results_random_20pair_2026-05-25/results.csv (this folder).
 9-pair source: bench_results_final/results.csv  (selector == 'random').

Per delay, paired Δ% = (obj_20 - obj_9) / obj_9 * 100 over the 80 matched
rows (4 instances × 4 budgets × 5 seeds). We report mean, median, n, and a
paired Wilcoxon signed-rank p-value.

Negative Δ% means the 20-pair pool improves on the 9-pair pool.

Output: bench_results_random_20pair_2026-05-25/comparison_to_9pair.md
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

REPO   = Path(__file__).resolve().parent.parent.parent
CSV_20 = REPO / "bench_results_random_20pair_2026-05-25" / "results.csv"
CSV_9  = REPO / "bench_results_final" / "results.csv"
OUT_MD = REPO / "bench_results_random_20pair_2026-05-25" / "comparison_to_9pair.md"

KEY = ["instance_base", "delay", "budget", "seed"]


def main():
    df20 = pd.read_csv(CSV_20)
    df9_full = pd.read_csv(CSV_9)
    df9 = df9_full[df9_full.selector == "random"].copy()

    print(f"20-pair rows: {len(df20)}  ({df20.obj.notna().sum()} non-null obj)")
    print(f" 9-pair rows: {len(df9)}  ({df9.obj.notna().sum()} non-null obj)")

    paired = df20.merge(df9[KEY + ["obj"]], on=KEY, suffixes=("_20", "_9"))
    paired = paired.dropna(subset=["obj_20", "obj_9"])
    paired["delta_pct"] = (paired.obj_20 - paired.obj_9) / paired.obj_9 * 100.0
    print(f"Paired rows: {len(paired)}")

    L = []
    L.append("# Random selector: 20-pair vs 9-pair pool")
    L.append("")
    L.append("Paired by `(instance_base, delay, budget, seed)`. "
             "`Δ% = (obj_20 − obj_9) / obj_9 × 100`. Negative means the "
             "20-pair pool (including the three novel operators) improves on "
             "the 9-pair pool. Wilcoxon is two-sided signed-rank on the "
             "paired Δs.")
    L.append("")
    L.append("- 20-pair pool: destroys `random,worst,shaw,rtcritical,lowslack` "
             "× repairs `greedy,regret2,regret3,slack`")
    L.append("-  9-pair pool: destroys `random,worst,shaw` × repairs "
             "`greedy,regret2,regret3`")
    L.append("- Grid per delay: 4 instances × 4 budgets × 5 seeds = 80 paired runs")
    L.append("")
    L.append("| delay | n | mean Δ% | median Δ% | wilcoxon p (two-sided) | wins (Δ<0) |")
    L.append("|---:|---:|---:|---:|---:|---:|")

    rows = []
    for d in sorted(paired.delay.unique()):
        g = paired[paired.delay == d]
        mean = g.delta_pct.mean()
        median = g.delta_pct.median()
        n_wins = int((g.delta_pct < 0).sum())
        if len(g) > 0 and (g.delta_pct.abs() > 0).any():
            stat, pval = wilcoxon(g.delta_pct.values)
        else:
            pval = float("nan")
        rows.append((d, len(g), mean, median, pval, n_wins))
        L.append(f"| d{d} | {len(g)} | {mean:+.3f}% | {median:+.3f}% | "
                 f"{pval:.4f} | {n_wins}/{len(g)} |")

    # Pooled (all delays)
    mean = paired.delta_pct.mean()
    median = paired.delta_pct.median()
    n_wins = int((paired.delta_pct < 0).sum())
    stat, pval = wilcoxon(paired.delta_pct.values)
    L.append(f"| **pooled** | {len(paired)} | {mean:+.3f}% | {median:+.3f}% | "
             f"{pval:.4f} | {n_wins}/{len(paired)} |")

    L.append("")
    L.append("## Per (delay, instance) breakdown")
    L.append("")
    L.append("| delay | instance | n | mean Δ% | median Δ% |")
    L.append("|---:|---|---:|---:|---:|")
    for (d, inst), g in paired.groupby(["delay", "instance_base"]):
        L.append(f"| d{d} | `{inst}` | {len(g)} | "
                 f"{g.delta_pct.mean():+.3f}% | {g.delta_pct.median():+.3f}% |")
    L.append("")

    OUT_MD.write_text("\n".join(L))
    print(f"\nWrote {OUT_MD}")
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
