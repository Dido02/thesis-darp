#!/usr/bin/env python3
"""Paired Δ% vs roulette per delay for ml_20f_t05__T0.10 and ml_20f_t07__T0.10.

Reads bench_results_final/results.csv. For each (ML model, delay) cell:
  - Pair each of the 80 (instance, budget, seed) ML runs against the
    roulette run at the same key.
  - per_pair_pct = (obj_ML - obj_roulette) / obj_roulette * 100
  - Report mean per_pair_pct + paired Wilcoxon p on (obj_ML - obj_roulette).

Output: bench_results_final/t05_t07_vs_roulette_aggregate.csv
Selector identifiers use the canonical CSV form: 'ml_20f_t05__T0.10' /
'ml_20f_t07__T0.10' (double underscore + dot, as in results.csv).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
CSV_IN  = REPO / "bench_results_final" / "results.csv"
CSV_OUT = REPO / "bench_results_final" / "t05_t07_vs_roulette_aggregate.csv"

MODELS = ["ml_20f_t05__T0.10", "ml_20f_t07__T0.10"]
DELAYS = [3, 5, 10]
KEY = ["instance_base", "delay", "budget", "seed"]


def main():
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} rows from {CSV_IN.name}")

    rows = []
    for model in MODELS:
        for delay in DELAYS:
            ml = df[(df.selector == model)   & (df.delay == delay)].copy()
            ro = df[(df.selector == "roulette") & (df.delay == delay)].copy()
            paired = ml.merge(
                ro[KEY + ["obj"]],
                on=KEY, suffixes=("_ml", "_ro"),
            )
            n = len(paired)
            assert n == 80, f"Expected 80 paired observations, got {n} for {model} d{delay}"
            diff = paired["obj_ml"].values - paired["obj_ro"].values
            pct  = (paired["obj_ml"].values - paired["obj_ro"].values) / paired["obj_ro"].values * 100.0
            mean_pct = float(pct.mean())
            try:
                p_val = float(wilcoxon(paired["obj_ml"].values, paired["obj_ro"].values,
                                       alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                p_val = 1.0   # only fires if all diffs are 0
            rows.append({
                "model": model, "delay": delay, "n": n,
                "mean_delta_pct_vs_roulette": mean_pct,
                "wilcoxon_p": p_val,
            })
            sig = "✓" if p_val < 0.05 else ""
            print(f"  {model:24s} delay={delay:2d}  n={n}  "
                  f"mean Δ%={mean_pct:+.4f}%  p={p_val:.4f} {sig}")

    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT}  ({len(out)} rows)")

    # ----- Print a copy-paste-ready markdown table -----
    print("\nMarkdown table:\n")
    print("| model | delay | n | mean Δ% vs roulette | Wilcoxon p-value |")
    print("|---|---:|---:|---:|---:|")
    for _, r in out.iterrows():
        print(f"| `{r['model']}` | d{int(r['delay']):02d} | {int(r['n'])} | "
              f"{r['mean_delta_pct_vs_roulette']:+.4f}% | {r['wilcoxon_p']:.4f} |")


if __name__ == "__main__":
    main()
