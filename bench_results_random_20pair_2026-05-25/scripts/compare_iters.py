#!/usr/bin/env python3
"""Compare ALNS iteration counts between 9-pair and 20-pair random benches.

Same pairing key as compare_to_9pair.py. Reports per-delay and per-budget
mean iters for each pool, plus paired Δ (iters_20 − iters_9) and the ratio.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

REPO   = Path(__file__).resolve().parent.parent.parent
CSV_20 = REPO / "bench_results_random_20pair_2026-05-25" / "results.csv"
CSV_9  = REPO / "bench_results_final" / "results.csv"
OUT_MD = REPO / "bench_results_random_20pair_2026-05-25" / "iters_comparison.md"

KEY = ["instance_base", "delay", "budget", "seed"]


def main():
    df20 = pd.read_csv(CSV_20)
    df9 = pd.read_csv(CSV_9)
    df9 = df9[df9.selector == "random"].copy()

    paired = df20.merge(df9[KEY + ["iters"]], on=KEY, suffixes=("_20", "_9"))
    paired = paired.dropna(subset=["iters_20", "iters_9"])
    paired["delta"] = paired.iters_20 - paired.iters_9
    paired["ratio"] = paired.iters_20 / paired.iters_9
    print(f"Paired rows: {len(paired)}")

    L = []
    L.append("# ALNS iteration counts: 20-pair vs 9-pair (random selector)")
    L.append("")
    L.append("Paired by `(instance_base, delay, budget, seed)`. `iters` is "
             "the total ALNS iteration count reported by the solver (summed "
             "across all epochs in the rolling-horizon dynamic run).")
    L.append("")
    L.append("## Per-delay summary")
    L.append("")
    L.append("| delay | n | mean iters 9 | mean iters 20 | mean Δ | mean ratio 20/9 | median ratio |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(paired.delay.unique()):
        g = paired[paired.delay == d]
        L.append(f"| d{d} | {len(g)} | {g.iters_9.mean():,.0f} | "
                 f"{g.iters_20.mean():,.0f} | {g.delta.mean():+,.0f} | "
                 f"{g.ratio.mean():.3f} | {g.ratio.median():.3f} |")
    L.append(f"| **pooled** | {len(paired)} | {paired.iters_9.mean():,.0f} | "
             f"{paired.iters_20.mean():,.0f} | {paired.delta.mean():+,.0f} | "
             f"{paired.ratio.mean():.3f} | {paired.ratio.median():.3f} |")
    L.append("")

    L.append("## Per-budget summary (pooled across delays and instances)")
    L.append("")
    L.append("| budget (ms) | n | mean iters 9 | mean iters 20 | mean ratio 20/9 |")
    L.append("|---:|---:|---:|---:|---:|")
    for b in sorted(paired.budget.unique()):
        g = paired[paired.budget == b]
        L.append(f"| {b} | {len(g)} | {g.iters_9.mean():,.0f} | "
                 f"{g.iters_20.mean():,.0f} | {g.ratio.mean():.3f} |")
    L.append("")

    L.append("## Per-(delay, instance) ratio")
    L.append("")
    L.append("| delay | instance | n | mean iters 9 | mean iters 20 | mean ratio 20/9 |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for (d, inst), g in paired.groupby(["delay", "instance_base"]):
        L.append(f"| d{d} | `{inst}` | {len(g)} | {g.iters_9.mean():,.0f} | "
                 f"{g.iters_20.mean():,.0f} | {g.ratio.mean():.3f} |")
    L.append("")

    L.append("## Paired Wilcoxon on iteration delta")
    L.append("")
    L.append("Null hypothesis: the paired distribution of `iters_20 − iters_9` "
             "is centered at zero. Negative Δ means the 20-pair pool runs *fewer* "
             "iterations at the same wall-clock budget (consistent with some "
             "operator pairs being slower per call).")
    L.append("")
    L.append("| delay | n | mean Δ | wilcoxon p (two-sided) |")
    L.append("|---:|---:|---:|---:|")
    for d in sorted(paired.delay.unique()):
        g = paired[paired.delay == d]
        stat, pval = wilcoxon(g.delta.values)
        L.append(f"| d{d} | {len(g)} | {g.delta.mean():+,.0f} | {pval:.4f} |")
    stat, pval = wilcoxon(paired.delta.values)
    L.append(f"| **pooled** | {len(paired)} | {paired.delta.mean():+,.0f} | "
             f"{pval:.4f} |")
    L.append("")

    OUT_MD.write_text("\n".join(L))
    print(f"\nWrote {OUT_MD}\n")
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
