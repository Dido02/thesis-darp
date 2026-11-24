#!/usr/bin/env python3
"""Generate BENCH_MORNING_2026-05-21.md.

Compares 4 new ML models + deployed t07 against random/roulette on the
07-00 morning instances. Then cross-references against the 18-00 evening
bench (bench_results_final/) to test whether the delay-5/10-win pattern
persists.
"""
from __future__ import annotations
import re, csv
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT   = Path("/Users/didodimitrov/Documents/thesis-darp")
MORN   = ROOT / "bench_results_morning_2026-05-21"
FINAL  = ROOT / "bench_results_final"
OUT    = MORN / "BENCH_MORNING_2026-05-21.md"

_RE_NAME = re.compile(r"dynamic__manhattan_morning_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_TT = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_IT = re.compile(r"Total ALNS iterations:\s+(\d+)")

ML_SELECTORS = ["ml_20f_t07__T0.10",
                "ml_forest_d8_K20__T0.10",
                "ml_forest_d8_K16__T0.10",
                "ml_tree_d4_K4__T0.10",
                "ml_tree_d8_K16__T0.10"]
BASELINES = ["random", "roulette"]
ALL_SELECTORS = BASELINES + ML_SELECTORS

INSTANCES = ["30_s", "01_min", "02_min", "05_min"]
DELAYS    = [3, 5, 10]
BUDGETS   = [200, 1000, 5000, 10000]
SEEDS     = [11, 22, 33, 44, 55]


def parse_morning():
    rows = []
    for p in sorted((MORN / "raw_logs").glob("*.log")):
        m = _RE_NAME.match(p.name)
        if not m: continue
        text = p.read_text()
        mt = _RE_TT.search(text); mi = _RE_IT.search(text)
        if not mt: continue
        rows.append({"instance_base": m.group(1), "delay": int(m.group(2)),
                     "selector": m.group(3), "budget": int(m.group(4)),
                     "seed": int(m.group(5)), "obj": float(mt.group(1)),
                     "iters": int(mi.group(1)) if mi else None})
    return pd.DataFrame(rows)


def load_evening():
    df = pd.read_csv(FINAL / "results.csv")
    df["delay"] = df["delay"].astype(int)
    return df[df.selector.isin(ALL_SELECTORS)]


def paired_wilcoxon(df, a, b, delay):
    sub = df[(df.delay == delay) & (df.selector.isin([a, b]))]
    piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                          columns="selector", values="obj")
    if a not in piv.columns or b not in piv.columns:
        return None
    piv = piv.dropna(subset=[a, b])
    if len(piv) < 3: return None
    diffs = piv[a].values - piv[b].values
    if (diffs == 0).all():
        return {"n": int(len(piv)), "mean_diff": 0.0, "p": 1.0}
    try:
        p = float(wilcoxon(piv[a].values, piv[b].values,
                           alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        p = None
    return {"n": int(len(piv)), "mean_diff": float(np.mean(diffs)),
            "mean_a": float(piv[a].mean()), "mean_b": float(piv[b].mean()),
            "p": p}


def delta_pct(df, sel, baseline, delay):
    sub = df[(df.delay == delay) & (df.selector.isin([sel, baseline]))]
    piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                          columns="selector", values="obj")
    if sel not in piv.columns or baseline not in piv.columns:
        return None
    piv = piv.dropna(subset=[sel, baseline])
    if len(piv) == 0: return None
    d = (piv[sel] - piv[baseline]) / piv[baseline] * 100
    return {"mean": float(d.mean()), "std": float(d.std(ddof=1)) if len(d) > 1 else 0,
            "n": len(d), "wins": int((d < 0).sum())}


def header(L, title, level=2):
    L.append("#" * level + " " + title); L.append("")


def main():
    morn = parse_morning()
    even = load_evening()

    print(f"Morning rows: {len(morn)}, evening rows: {len(even)}")
    print(morn.groupby("selector").size().to_string())

    L = []
    L.append("# Bench Report — 07-00 morning Manhattan instances")
    L.append("")
    L.append("**Date:** 2026-05-21")
    L.append(f"**Total runs:** {len(morn)}")
    L.append("")

    # 1. Completion + scope
    header(L, "1. Scope and completion")
    L.append("Grid: 4 instances `{morning_30_s, morning_01_min, morning_02_min, morning_05_min}` × 3 delays `{03, 05, 10}` × 7 selectors × 4 budgets `{200, 1000, 5000, 10000 ms}` × 5 seeds `{11, 22, 33, 44, 55}` = 1680 runs.")
    L.append("")
    counts = morn.groupby("selector").size()
    L.append("| selector | runs |")
    L.append("|---|---:|")
    for s in ALL_SELECTORS:
        L.append(f"| `{s}` | {counts.get(s, 0)} / 240 |")
    L.append("")
    L.append("The morning instances are subsamples of the Fiedler 07-00 16-hour Manhattan source: same vehicle count and same time-of-day duration as the 18-00 instances, but morning request densities (~50% of evening at 7 AM). The new ML configurations and the deployed `ml_20f_t07__T0.10` were all trained on 18-00 evening data and have NEVER seen 07-00 morning data during training.")
    L.append("")

    # 2. Per-delay Wilcoxon for each ML config vs random and roulette
    for baseline in ("random", "roulette", "ml_20f_t07__T0.10"):
        if baseline == "ml_20f_t07__T0.10":
            header(L, f"2c. Wilcoxon — new configs vs deployed `ml_20f_t07__T0.10`")
            sel_pool = [s for s in ML_SELECTORS if s != baseline]
        else:
            n = "a" if baseline == "random" else "b"
            header(L, f"2{n}. Wilcoxon — ML configs vs `{baseline}`")
            sel_pool = ML_SELECTORS
        L.append("Paired Wilcoxon over `(instance, budget, seed)` cells.")
        L.append("")
        L.append(f"| selector | delay 3 | delay 5 | delay 10 |")
        L.append("|---|---|---|---|")
        for sel in sel_pool:
            cells = []
            for d in DELAYS:
                r = paired_wilcoxon(morn, sel, baseline, d)
                if r is None:
                    cells.append("n/a")
                else:
                    p_str = f"{r['p']:.4f}" if r['p'] is not None else "n/a"
                    mark = " ✓" if (r['p'] is not None and r['p'] < 0.05 and r['mean_diff'] < 0) else ""
                    cells.append(f"Δ={r['mean_diff']:+.0f}, p={p_str}{mark}")
            L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
        L.append("")
        L.append("(Δ = mean obj difference. Negative = selector wins. ✓ = p<0.05 AND selector wins.)")
        L.append("")

    # 3. Cross-delay Δ% summary (mirrors BENCH_FINAL §5)
    header(L, "3. Cross-delay Δ% summary (mean ± std)")
    L.append("Each cell aggregates over (instance, budget, seed). Paired by seed.")
    L.append("")
    for baseline in ("random", "roulette"):
        L.append(f"**vs `{baseline}`**")
        L.append("")
        L.append("| selector | delay 3 | delay 5 | delay 10 |")
        L.append("|---|---|---|---|")
        for sel in ML_SELECTORS:
            cells = []
            for d in DELAYS:
                r = delta_pct(morn, sel, baseline, d)
                if r is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{r['mean']:+.3f}% ± {r['std']:.3f}% (n={r['n']})")
            L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
        L.append("")

    # 4. Headline — does the delay-5/10 win pattern from evening persist on morning?
    header(L, "4. Does the evening pattern persist on morning? (cross-bench)")
    L.append("Compares each selector's mean Δ% vs `random` between evening (`bench_results_final/`) and morning (this bench), per delay.")
    L.append("")
    L.append("| selector | delay | evening Δ% vs random | morning Δ% vs random | trend |")
    L.append("|---|---:|---:|---:|---|")
    for sel in ML_SELECTORS:
        for d in DELAYS:
            ev = delta_pct(even, sel, "random", d)
            mn = delta_pct(morn, sel, "random", d)
            ev_str = f"{ev['mean']:+.3f}%" if ev else "n/a"
            mn_str = f"{mn['mean']:+.3f}%" if mn else "n/a"
            trend = ""
            if ev and mn:
                if ev['mean'] < 0 and mn['mean'] < 0:
                    trend = "✓ both win"
                elif ev['mean'] >= 0 and mn['mean'] >= 0:
                    trend = "✗ both lose"
                elif ev['mean'] < 0 and mn['mean'] >= 0:
                    trend = "wins evening, loses morning"
                else:
                    trend = "loses evening, wins morning"
            L.append(f"| `{sel}` | d{d:02d} | {ev_str} | {mn_str} | {trend} |")
    L.append("")

    # 5. Per-instance win-rate (morning only)
    header(L, "5. Per-instance win-rate vs `random` (morning)")
    L.append("Wins = cells where mean Δ% < 0, averaged over 5 seeds per (delay, budget) cell.")
    L.append("")
    L.append("| selector | 30_s | 01_min | 02_min | 05_min | overall |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for sel in ML_SELECTORS:
        per_inst = {}
        total = 0; total_wins = 0
        for inst in INSTANCES:
            wins = 0; n = 0
            for d in DELAYS:
                for b in BUDGETS:
                    sub = morn[(morn.delay == d) & (morn.instance_base == inst) &
                               (morn.budget == b) & (morn.selector.isin([sel, "random"]))]
                    piv = sub.pivot_table(index="seed", columns="selector", values="obj")
                    if sel not in piv.columns or "random" not in piv.columns: continue
                    piv = piv.dropna(subset=[sel, "random"])
                    if len(piv) == 0: continue
                    mean_delta = ((piv[sel] - piv["random"]) / piv["random"]).mean()
                    n += 1
                    if mean_delta < 0: wins += 1
            per_inst[inst] = (wins, n); total_wins += wins; total += n
        cells = " | ".join(f"{w}/{n}" for inst, (w, n) in per_inst.items() if inst in INSTANCES)
        L.append(f"| `{sel}` | {cells} | {total_wins}/{total} ({100*total_wins/max(total,1):.0f}%) |")
    L.append("")

    # 6. Quick observations
    header(L, "6. Observations")
    L.append("- See §4 (cross-bench): if a selector wins delay 5/10 on evening AND morning, the pattern persists out-of-cohort.")
    L.append("- The 4 new ML configurations were carried into this bench based on prior LOIO + 5-fold analysis. Their performance here is on instances not seen in any training stage.")
    L.append("- Morning request density is ~50% of evening at the same time-of-day duration — fewer requests per cell, more vehicle slack. Cells may saturate easier (less headroom for any selector to improve).")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"\nWrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
