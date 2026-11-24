#!/usr/bin/env python3
"""Best per-cell mean Δ% for each ML configuration vs random and roulette.

For each of the 6 ML configs:
  - 2 deployed-cohort configs from bench_results_final/results.csv
    (ml_20f_t05__T0.10 = forest d=None K=20 t=0.5,
     ml_20f_t07__T0.10 = forest d=None K=20 t=0.7)
  - 4 new-model configs from bench_results_new_models_2026-05-20/results.csv
    (forest_d8_K20, forest_d8_K16, tree_d4_K4, tree_d8_K16 — all t=0.7)

Both benches share the same evening grid (4 instances × 3 delays × 4 budgets ×
5 seeds = 240 runs per selector); baselines (random, roulette) are pulled from
bench_results_final/results.csv and paired by (instance_base, delay, budget,
seed). Per-cell Δ% is the seed-mean of (obj_ml − obj_base) / obj_base × 100.

Output: analysis/best_cell_per_model.md
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
CSV_FINAL = REPO / "bench_results_final" / "results.csv"
CSV_NEW   = REPO / "bench_results_new_models_2026-05-20" / "results.csv"
OUT_MD    = REPO / "analysis" / "best_cell_per_model.md"

# (CSV path, selector tag in that CSV, human label for the table)
ML_CONFIGS = [
    (CSV_FINAL, "ml_20f_t05__T0.10",       "forest d=None K=20 t=0.5"),
    (CSV_FINAL, "ml_20f_t07__T0.10",       "forest d=None K=20 t=0.7"),
    (CSV_NEW,   "ml_forest_d8_K20__T0.10", "forest d=8 K=20 t=0.7"),
    (CSV_NEW,   "ml_forest_d8_K16__T0.10", "forest d=8 K=16 t=0.7"),
    (CSV_NEW,   "ml_tree_d4_K4__T0.10",    "tree d=4 K=4 t=0.7"),
    (CSV_NEW,   "ml_tree_d8_K16__T0.10",   "tree d=8 K=16 t=0.7"),
]
KEY = ["instance_base", "delay", "budget", "seed"]


def per_cell_delta(ml_df, base_df):
    paired = ml_df.merge(base_df[KEY + ["obj"]], on=KEY, suffixes=("_ml", "_base"))
    paired["delta_pct"] = (paired["obj_ml"] - paired["obj_base"]) / paired["obj_base"] * 100.0
    g = paired.groupby(["delay", "instance_base", "budget"], as_index=False).agg(
        mean_delta_pct=("delta_pct", "mean"),
        n=("seed", "count"),
    )
    return g


def best_row(g):
    return g.loc[g.mean_delta_pct.idxmin()]


def main():
    final = pd.read_csv(CSV_FINAL)
    new   = pd.read_csv(CSV_NEW)
    rnd = final[final.selector == "random"]
    rou = final[final.selector == "roulette"]
    assert len(rnd) == 240 and len(rou) == 240, "expected 240 baseline rows per selector"

    rows = []
    for csv_path, sel, label in ML_CONFIGS:
        src = final if csv_path == CSV_FINAL else new
        ml = src[src.selector == sel]
        if len(ml) == 0:
            print(f"MISSING: {sel} not in {csv_path.name}")
            continue
        assert len(ml) == 240, f"expected 240 rows for {sel}, got {len(ml)}"
        cells_rnd = per_cell_delta(ml, rnd)
        cells_rou = per_cell_delta(ml, rou)
        b_rnd = best_row(cells_rnd)
        b_rou = best_row(cells_rou)
        rows.append({
            "config": label,
            "best_cell_rnd": f"(d{int(b_rnd.delay):02d}, {b_rnd.instance_base}, b={int(b_rnd.budget)})",
            "best_delta_rnd": float(b_rnd.mean_delta_pct),
            "best_cell_rou": f"(d{int(b_rou.delay):02d}, {b_rou.instance_base}, b={int(b_rou.budget)})",
            "best_delta_rou": float(b_rou.mean_delta_pct),
        })
        print(f"  {label:30s}  "
              f"best vs random: {rows[-1]['best_cell_rnd']} {rows[-1]['best_delta_rnd']:+.3f}%   "
              f"best vs roulette: {rows[-1]['best_cell_rou']} {rows[-1]['best_delta_rou']:+.3f}%")

    L = []
    L.append("# Best per-cell mean Δ% for each ML configuration")
    L.append("")
    L.append("**Cell:** (delay, instance, budget). Δ% = mean over 5 seeds of "
             "`(obj_ml − obj_base) / obj_base × 100`, paired by (instance, delay, budget, seed).")
    L.append("**Grid per ML config:** 4 instances × 3 delays × 4 budgets = **48 cells**.")
    L.append("**Sources:** `ml_20f_t05/t07` from `bench_results_final/results.csv`; "
             "`forest_d8_*` and `tree_*` from `bench_results_new_models_2026-05-20/results.csv`. "
             "Baselines (`random`, `roulette`) from `bench_results_final/results.csv` in both cases.")
    L.append("")
    L.append("| ML config | best cell vs random | best Δ% vs random | best cell vs roulette | best Δ% vs roulette |")
    L.append("|---|---|---:|---|---:|")
    for r in rows:
        L.append(f"| `{r['config']}` | {r['best_cell_rnd']} | "
                 f"{r['best_delta_rnd']:+.3f}% | {r['best_cell_rou']} | "
                 f"{r['best_delta_rou']:+.3f}% |")
    L.append("")

    OUT_MD.write_text("\n".join(L))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
