#!/usr/bin/env python3
"""Per-(delay, instance, budget) cell mean Δ% for the deployed ML selector
against random and roulette baselines.

Source: bench_results_final/results.csv (contains the deployed
ml_20f_t07__T0.10 alongside random and roulette).

Per-cell Δ% = mean over 5 seeds of (obj_ml − obj_baseline) / obj_baseline × 100,
where each pair is matched on (instance_base, delay, budget, seed).
Grid: 4 instances × 3 delays × 4 budgets = 48 cells per baseline.

Output: analysis/per_cell_gap_summary.md
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
CSV_IN = REPO / "bench_results_final" / "results.csv"
OUT_MD = REPO / "analysis" / "per_cell_gap_summary.md"

ML_TAG = "ml_20f_t07__T0.10"
KEY = ["instance_base", "delay", "budget", "seed"]


def cells(ml, base):
    paired = ml.merge(base[KEY + ["obj"]], on=KEY, suffixes=("_ml", "_base"))
    paired["delta_pct"] = (paired["obj_ml"] - paired["obj_base"]) / paired["obj_base"] * 100.0
    g = paired.groupby(["delay", "instance_base", "budget"], as_index=False).agg(
        mean_delta_pct=("delta_pct", "mean"),
        n_seeds=("seed", "count"),
    )
    return g


def summarize(g, label):
    best = g.loc[g.mean_delta_pct.idxmin()]
    worst = g.loc[g.mean_delta_pct.idxmax()]
    n_wins = int((g.mean_delta_pct < 0).sum())
    n_total = len(g)
    return {
        "label": label,
        "best_delay": int(best.delay),
        "best_instance": best.instance_base,
        "best_budget": int(best.budget),
        "best_delta_pct": float(best.mean_delta_pct),
        "worst_delay": int(worst.delay),
        "worst_instance": worst.instance_base,
        "worst_budget": int(worst.budget),
        "worst_delta_pct": float(worst.mean_delta_pct),
        "n_wins": n_wins,
        "n_total": n_total,
    }


def main():
    df = pd.read_csv(CSV_IN)
    ml  = df[df.selector == ML_TAG]
    rnd = df[df.selector == "random"]
    rou = df[df.selector == "roulette"]
    print(f"Loaded {len(df)} rows;  ml={len(ml)}, random={len(rnd)}, roulette={len(rou)}")
    assert len(ml) == 240 and len(rnd) == 240 and len(rou) == 240, "expected 240 per selector"

    cells_rnd = cells(ml, rnd)
    cells_rou = cells(ml, rou)
    s_rnd = summarize(cells_rnd, "random")
    s_rou = summarize(cells_rou, "roulette")

    L = []
    L.append("# Per-cell Δ% for the deployed ML selector (forest d=None, K=20, t=0.7)")
    L.append("")
    L.append(f"**Source:** `bench_results_final/results.csv` ({ML_TAG} alongside both baselines).")
    L.append(f"**Per-cell Δ%:** mean over 5 seeds of (obj_ml − obj_baseline) / obj_baseline × 100, paired by (instance, delay, budget, seed).")
    L.append(f"**Grid:** 4 instances × 3 delays × 4 budgets = 48 cells per baseline.")
    L.append("")

    for s in (s_rnd, s_rou):
        L.append(f"## ML vs `{s['label']}`")
        L.append("")
        L.append(f"- **Best cell (largest ML win):** delay {s['best_delay']:02d}, instance `{s['best_instance']}`, budget {s['best_budget']} → mean Δ% = **{s['best_delta_pct']:+.3f}%**")
        L.append(f"- **Worst cell (largest ML loss):** delay {s['worst_delay']:02d}, instance `{s['worst_instance']}`, budget {s['worst_budget']} → mean Δ% = **{s['worst_delta_pct']:+.3f}%**")
        L.append(f"- **Cells where ML wins (Δ% < 0):** {s['n_wins']} / {s['n_total']}")
        L.append("")

    OUT_MD.write_text("\n".join(L))
    print(f"\nWrote {OUT_MD}")

    # Print to stdout too for the response
    print()
    print("=" * 60)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
