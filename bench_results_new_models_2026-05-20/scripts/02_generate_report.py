#!/usr/bin/env python3
"""Generate BENCH_NEW_MODELS_2026-05-20.md from completed runs only.

Reads:
  bench_results_new_models_2026-05-20/results.csv  (4 new ML configs)
  bench_results_final/results.csv                   (random, roulette, deployed t07)

Only cells with all 5 seeds completed are included in statistical tests;
incomplete cells are listed in an appendix.

Writes:
  bench_results_new_models_2026-05-20/BENCH_NEW_MODELS_2026-05-20.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from collections import defaultdict

NEW_DIR   = Path("/Users/didodimitrov/Documents/thesis-darp/bench_results_new_models_2026-05-20")
FINAL_DIR = Path("/Users/didodimitrov/Documents/thesis-darp/bench_results_final")
OUT       = NEW_DIR / "BENCH_NEW_MODELS_2026-05-20.md"

# Selectors of interest
NEW_SELECTORS      = ["ml_forest_d8_K20__T0.10", "ml_forest_d8_K16__T0.10",
                      "ml_tree_d4_K4__T0.10",    "ml_tree_d8_K16__T0.10"]
BASELINE_SELECTORS = ["random", "roulette", "ml_20f_t07__T0.10"]
ALL_SELECTORS      = BASELINE_SELECTORS + NEW_SELECTORS

INSTANCES = ["30_s", "01_min", "02_min", "05_min"]
DELAYS    = [3, 5, 10]
BUDGETS   = [200, 1000, 5000, 10000]
SEEDS     = [11, 22, 33, 44, 55]


def load_data():
    new = pd.read_csv(NEW_DIR / "results.csv")
    fin = pd.read_csv(FINAL_DIR / "results.csv")
    # Normalise delay typing
    new["delay"] = new["delay"].astype(int)
    fin["delay"] = fin["delay"].astype(int)
    # Restrict baselines from final
    fin = fin[fin.selector.isin(BASELINE_SELECTORS)]
    return pd.concat([new, fin[new.columns.tolist()]], ignore_index=True)


def cell_completeness(df):
    """For each (instance, delay, selector, budget): how many seeds completed."""
    g = df.groupby(["instance_base", "delay", "selector", "budget"])["seed"].nunique()
    return g.reset_index().rename(columns={"seed": "n_seeds"})


def paired_wilcoxon(df, selector_a, selector_b, delay):
    """Paired test over (instance, budget, seed) common to both selectors."""
    sub = df[(df.delay == delay) & (df.selector.isin([selector_a, selector_b]))]
    piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                          columns="selector", values="obj").dropna(subset=[selector_a, selector_b])
    if len(piv) < 3:
        return {"n": len(piv), "mean_diff": None, "p": None}
    diffs = piv[selector_a].values - piv[selector_b].values
    if (diffs == 0).all():
        return {"n": len(piv), "mean_diff": 0.0, "p": 1.0}
    try:
        p = float(wilcoxon(piv[selector_a].values, piv[selector_b].values,
                           alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        p = None
    return {"n": int(len(piv)), "mean_diff": float(np.mean(diffs)),
            "mean_a": float(np.mean(piv[selector_a])),
            "mean_b": float(np.mean(piv[selector_b])),
            "p": p}


def cell_delta(df, selector, baseline, delay, instance, budget):
    """Per-cell mean Δ% vs baseline, averaged over seeds present in both."""
    sub = df[(df.delay == delay) & (df.instance_base == instance) &
             (df.budget == budget) & (df.selector.isin([selector, baseline]))]
    if sub.empty:
        return None, None, 0
    piv = sub.pivot_table(index="seed", columns="selector", values="obj")
    if selector not in piv.columns or baseline not in piv.columns:
        return None, None, 0
    piv = piv.dropna(subset=[selector, baseline])
    if len(piv) == 0:
        return None, None, 0
    delta_pct = ((piv[selector] - piv[baseline]) / piv[baseline] * 100).values
    return float(np.mean(delta_pct)), float(np.std(delta_pct, ddof=1) if len(delta_pct) > 1 else 0), len(piv)


def section_header(out, title, level=2):
    out.append("#" * level + " " + title)
    out.append("")


def main():
    df = load_data()
    completeness = cell_completeness(df)
    print(df.groupby("selector")["obj"].count().to_string())

    out = []
    out.append("# Bench Report — new t07 model configurations")
    out.append("")
    out.append(f"**Date:** 2026-05-20")
    out.append(f"**Selectors compared:** {len(ALL_SELECTORS)} "
               f"({len(BASELINE_SELECTORS)} baselines from `bench_results_final/`, "
               f"{len(NEW_SELECTORS)} new from this sweep)")
    out.append("")

    # 1. TL;DR
    section_header(out, "1. TL;DR")
    counts = df.groupby("selector").size()
    out.append("Summary of completed runs per selector "
               "(target: 4 instances × 3 delays × 4 budgets × 5 seeds = 240):")
    out.append("")
    out.append("| selector | runs completed | status |")
    out.append("|---|---:|---|")
    for s in ALL_SELECTORS:
        n = counts.get(s, 0)
        status = "complete" if n >= 240 else ("partial" if n > 0 else "missing")
        out.append(f"| `{s}` | {n} / 240 | {status} |")
    out.append("")

    # 2. Methodology
    section_header(out, "2. Methodology")
    out.append("- Same C++ solver binary as BENCH_FINAL_2026-05-18 (BUILD_ID inherited).")
    out.append("- Grid: 4 instances `{30_s, 01_min, 02_min, 05_min}` × 3 delays `{03, 05, 10 min}` × 4 budgets `{200, 1000, 5000, 10000 ms}` × 5 seeds `{11, 22, 33, 44, 55}`.")
    out.append("- Fair budget via `--exclude-selection-from-budget`.")
    out.append("- Per-iter logging via `--log-epoch-iters`.")
    out.append("- ML inference uses softmax with temperature $T = 0.10$.")
    out.append("- Random and roulette baselines + the deployed `ml_20f_t07__T0.10` are pulled from `bench_results_final/results.csv` (unchanged).")
    out.append("")
    out.append("**Partial completion note:** the new-model bench was interrupted before all 960 runs finished. Statistical tests only include cells with all 5 seeds for both selectors being compared.")
    out.append("")

    # 3. Per-delay Wilcoxon vs random
    for baseline in ("random", "roulette", "ml_20f_t07__T0.10"):
        section_header(out, f"3. Paired Wilcoxon — new configs vs `{baseline}`")
        out.append(f"Paired Wilcoxon over `(instance, budget, seed)` cells where both selectors have data.")
        out.append("")
        out.append(f"| selector | delay | $n$ paired | mean obj diff vs {baseline} | $p$-value |")
        out.append("|---|---:|---:|---:|---:|")
        for sel in NEW_SELECTORS:
            for d in DELAYS:
                r = paired_wilcoxon(df, sel, baseline, d)
                p_str = f"{r['p']:.4f}" if r['p'] is not None else "n/a"
                diff_str = (f"{r['mean_diff']:+.0f}" if r['mean_diff'] is not None else "n/a")
                marker = " ✓" if (r['p'] is not None and r['p'] < 0.05 and r['mean_diff'] < 0) else ""
                out.append(f"| `{sel}` | {d:02d} | {r['n']} | {diff_str} | {p_str}{marker} |")
        out.append("")
        out.append("(Negative mean diff = new selector wins on average; ✓ = $p<0.05$ AND mean diff favours the new selector. No Holm correction applied here.)")
        out.append("")

    # 4. Cross-delay mean Δ% per (selector, delay)
    section_header(out, "4. Cross-delay Δ% summary (mean ± std)")
    out.append("Each cell aggregates over `(instance, budget, seed)` triples completed for both selectors. Δ% = (new − baseline) / baseline, paired by seed.")
    out.append("")
    for baseline in ("random", "roulette", "ml_20f_t07__T0.10"):
        out.append(f"**vs `{baseline}`**")
        out.append("")
        out.append("| selector | delay 03 | delay 05 | delay 10 |")
        out.append("|---|---:|---:|---:|")
        for sel in NEW_SELECTORS:
            cells = []
            for d in DELAYS:
                sub = df[(df.delay == d) & (df.selector.isin([sel, baseline]))]
                piv = sub.pivot_table(index=["instance_base","budget","seed"],
                                      columns="selector", values="obj").dropna(subset=[sel, baseline])
                if len(piv) == 0:
                    cells.append("n/a")
                else:
                    delta_pct = (piv[sel] - piv[baseline]) / piv[baseline] * 100
                    cells.append(f"{delta_pct.mean():+.3f}% ± {delta_pct.std(ddof=1):.3f}% (n={len(piv)})")
            out.append(f"| `{sel}` | " + " | ".join(cells) + " |")
        out.append("")

    # 5. Per-delay Δ% vs random per (instance, budget)
    section_header(out, "5. Per-cell Δ% vs `random` (negative = new selector wins)")
    for d in DELAYS:
        out.append(f"### delay = {d:02d} min")
        out.append("")
        out.append("| instance | budget | " + " | ".join(f"`{s}`" for s in NEW_SELECTORS) + " |")
        out.append("|---|---:|" + "---:|" * len(NEW_SELECTORS))
        for inst in INSTANCES:
            for b in BUDGETS:
                cells = []
                for sel in NEW_SELECTORS:
                    delta, std, n = cell_delta(df, sel, "random", d, inst, b)
                    if delta is None:
                        cells.append("n/a")
                    else:
                        marker = " ✓" if delta < 0 else ""
                        cells.append(f"{delta:+.3f}% (n={n}){marker}")
                out.append(f"| {inst} | {b} | " + " | ".join(cells) + " |")
        out.append("")

    # 6. Per-selector win count vs random
    section_header(out, "6. Win-rate summary (cells where new selector beats `random` on mean Δ%)")
    out.append("")
    out.append("| selector | overall | by delay (03/05/10) | by instance (30s/01/02/05) | by budget (200/1k/5k/10k) |")
    out.append("|---|---:|---|---|---|")
    for sel in NEW_SELECTORS:
        # Build per-cell wins
        per_cell = []
        for d in DELAYS:
            for inst in INSTANCES:
                for b in BUDGETS:
                    delta, _, n = cell_delta(df, sel, "random", d, inst, b)
                    if delta is None or n == 0: continue
                    per_cell.append({"delay": d, "instance": inst, "budget": b, "win": delta < 0})
        total = len(per_cell); wins = sum(p["win"] for p in per_cell)
        delay_str = " / ".join(f"{sum(1 for p in per_cell if p['delay'] == d and p['win'])}/{sum(1 for p in per_cell if p['delay'] == d)}" for d in DELAYS)
        inst_str  = " / ".join(f"{sum(1 for p in per_cell if p['instance'] == i and p['win'])}/{sum(1 for p in per_cell if p['instance'] == i)}" for i in INSTANCES)
        bud_str   = " / ".join(f"{sum(1 for p in per_cell if p['budget'] == b and p['win'])}/{sum(1 for p in per_cell if p['budget'] == b)}" for b in BUDGETS)
        out.append(f"| `{sel}` | {wins}/{total} ({100*wins/max(total,1):.0f}%) | {delay_str} | {inst_str} | {bud_str} |")
    out.append("")

    # 7. Iter parity (sanity)
    section_header(out, "7. Iter parity (sanity check)")
    out.append("Mean ALNS iters per cell, ratio (new selector / `random`). Values near 1.00 mean the fair-budget flag is honoured.")
    out.append("")
    out.append("| selector | delay 03 ratio | delay 05 ratio | delay 10 ratio |")
    out.append("|---|---:|---:|---:|")
    for sel in NEW_SELECTORS:
        cells = []
        for d in DELAYS:
            sub = df[(df.delay == d) & (df.selector.isin([sel, "random"]))]
            piv = sub.pivot_table(index=["instance_base","budget","seed"],
                                  columns="selector", values="iters").dropna(subset=[sel, "random"])
            if len(piv) == 0:
                cells.append("n/a")
            else:
                ratio = (piv["random"] / piv[sel]).mean()
                cells.append(f"{ratio:.3f}×")
        out.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    out.append("")

    # 8. Incomplete cells appendix
    section_header(out, "8. Incomplete cells (appendix)")
    out.append("Cells where fewer than 5 seeds completed for at least one selector.")
    out.append("")
    incomplete = completeness[completeness.n_seeds < 5]
    if len(incomplete) == 0:
        out.append("None — all reported cells have 5 seeds.")
    else:
        out.append("| instance | delay | selector | budget | seeds completed |")
        out.append("|---|---:|---|---:|---:|")
        for _, r in incomplete.iterrows():
            out.append(f"| {r['instance_base']} | {r['delay']} | `{r['selector']}` | {r['budget']} | {r['n_seeds']} |")
    out.append("")

    OUT.write_text("\n".join(out))
    print(f"\nWrote {OUT} ({len(out)} lines)")


if __name__ == "__main__":
    main()
