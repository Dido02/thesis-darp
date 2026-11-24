#!/usr/bin/env python3
"""Generate BENCH_SINGLE_PAIR_2026-05-21.md.

Compares 4 single-(destroy, repair) ALNS configurations against the
9-pair classical baselines (random, roulette) from bench_results_final/.

Goal: justify operator diversity in ALNS by showing each single-pair
variant is worse than the full 9-pair pool.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
SP   = ROOT / "bench_results_single_pair_2026-05-21"
FIN  = ROOT / "bench_results_final"
OUT  = SP / "BENCH_SINGLE_PAIR_2026-05-21.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_TT = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_IT = re.compile(r"Total ALNS iterations:\s+(\d+)")

SINGLE_PAIRS = ["op_random_greedy", "op_worst_greedy", "op_shaw_greedy", "op_worst_regret2"]
BASELINES    = ["random", "roulette"]   # both use the full 9-pair pool

INSTANCES = ["30_s", "01_min", "02_min", "05_min"]
DELAYS    = [3, 5, 10]
BUDGETS   = [200, 1000, 5000, 10000]
SEEDS     = [11, 22, 33, 44, 55]


def parse_logs():
    rows = []
    for p in sorted((SP / "raw_logs").glob("*.log")):
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


def load_baselines():
    df = pd.read_csv(FIN / "results.csv")
    df["delay"] = df["delay"].astype(int)
    return df[df.selector.isin(BASELINES)]


def paired_wilcoxon(df, a, b, delay):
    sub = df[(df.delay == delay) & (df.selector.isin([a, b]))]
    piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                          columns="selector", values="obj")
    if a not in piv.columns or b not in piv.columns:
        return None
    piv = piv.dropna(subset=[a, b])
    if len(piv) < 3: return None
    diffs = piv[a].values - piv[b].values
    if (diffs == 0).all(): return {"n": int(len(piv)), "mean_diff": 0.0, "p": 1.0}
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


def header(L, t, level=2):
    L.append("#" * level + " " + t); L.append("")


def main():
    sp = parse_logs()
    bl = load_baselines()
    df = pd.concat([sp, bl[sp.columns.tolist()]], ignore_index=True)

    print(f"Single-pair rows: {len(sp)}, baseline rows: {len(bl)}")
    print(sp.groupby("selector").size().to_string())

    L = []
    L.append("# Bench Report — single-(destroy, repair) ablation")
    L.append("")
    L.append("**Date:** 2026-05-21")
    L.append(f"**Single-pair runs:** {len(sp)}")
    L.append(f"**Baseline runs (from `bench_results_final/`):** {len(bl)}")
    L.append("")

    # 1. Scope
    header(L, "1. Scope")
    L.append("Tests whether ALNS lift requires operator diversity. Each single-pair config runs ALNS with exactly one destroy and one repair (1-pair pool). Baselines are the classical 9-pair pool (3 destroys × 3 repairs) with `random` or `roulette` selection from `bench_results_final/`.")
    L.append("")
    L.append("**Configs in this bench (1-pair pool each):**")
    L.append("")
    L.append("| tag | destroy | repair |")
    L.append("|---|---|---|")
    for c in SINGLE_PAIRS:
        _, d, r = c.split("_", 2)
        L.append(f"| `{c}` | {d} | {r} |")
    L.append("")
    L.append("**Baselines (9-pair pool from `bench_results_final/`):** `random`, `roulette`. Both use the same `destroy=random,worst,shaw; repair=greedy,regret2,regret3` pool.")
    L.append("")
    L.append("**Grid (matches BENCH_FINAL):** 4 instances × 3 delays × 4 budgets × 5 seeds = 240 cells per single-pair = 960 runs.")
    L.append("")

    # 2. Wilcoxon per single-pair vs 9-pair baselines
    for baseline in BASELINES:
        header(L, f"2. Wilcoxon — single-pair vs 9-pair `{baseline}`")
        L.append(f"Paired Wilcoxon over `(instance, budget, seed)` triples per delay.")
        L.append("")
        L.append("| single-pair config | delay 3 | delay 5 | delay 10 |")
        L.append("|---|---|---|---|")
        for sel in SINGLE_PAIRS:
            cells = []
            for d in DELAYS:
                r = paired_wilcoxon(df, sel, baseline, d)
                if r is None:
                    cells.append("n/a")
                else:
                    p_str = f"{r['p']:.4f}" if r['p'] is not None else "n/a"
                    win = " ✓" if (r['p'] is not None and r['p'] < 0.05 and r['mean_diff'] < 0) else ""
                    lose = " ✗" if (r['p'] is not None and r['p'] < 0.05 and r['mean_diff'] > 0) else ""
                    cells.append(f"Δ={r['mean_diff']:+.0f}, p={p_str}{win}{lose}")
            L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
        L.append("")
        L.append("(Δ = mean obj difference. ✓ = single-pair wins (rare/never expected). ✗ = single-pair loses significantly — the diversity-matters result we want to see.)")
        L.append("")

    # 3. Cross-delay Δ% summary (mean ± std)
    header(L, "3. Cross-delay Δ% summary (mean ± std)")
    L.append("Positive Δ% = single-pair is WORSE than baseline (the diversity penalty).")
    L.append("")
    for baseline in BASELINES:
        L.append(f"**vs 9-pair `{baseline}`**")
        L.append("")
        L.append("| single-pair config | delay 3 | delay 5 | delay 10 |")
        L.append("|---|---|---|---|")
        for sel in SINGLE_PAIRS:
            cells = []
            for d in DELAYS:
                r = delta_pct(df, sel, baseline, d)
                if r is None:
                    cells.append("n/a")
                else:
                    sign = "↑ worse" if r['mean'] > 0 else "↓ better"
                    cells.append(f"{r['mean']:+.3f}% ± {r['std']:.3f}% ({sign})")
            L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
        L.append("")

    # 4. Per-instance breakdown vs 9-pair random
    header(L, "4. Per-instance Δ% vs 9-pair `random` (averaged over budgets and seeds)")
    L.append("")
    for sel in SINGLE_PAIRS:
        L.append(f"**`{sel}`**")
        L.append("")
        L.append("| instance | delay 3 | delay 5 | delay 10 |")
        L.append("|---|---|---|---|")
        for inst in INSTANCES:
            cells = []
            for d in DELAYS:
                sub = df[(df.delay == d) & (df.instance_base == inst) &
                         (df.selector.isin([sel, "random"]))]
                piv = sub.pivot_table(index=["budget", "seed"],
                                      columns="selector", values="obj")
                if sel not in piv.columns or "random" not in piv.columns:
                    cells.append("n/a"); continue
                piv = piv.dropna(subset=[sel, "random"])
                if len(piv) == 0:
                    cells.append("n/a"); continue
                d_pct = (piv[sel] - piv["random"]) / piv["random"] * 100
                cells.append(f"{d_pct.mean():+.3f}% (n={len(piv)})")
            L.append(f"| {inst} | " + " | ".join(cells) + " |")
        L.append("")

    # 5. Per-pair comparison — which destroy/repair is least bad alone?
    header(L, "5. Which single-pair is least bad?")
    L.append("Mean Δ% vs 9-pair `random` across all 240 cells per single-pair. Smaller (closer to 0%) = closer to the 9-pair baseline = less penalised by losing diversity.")
    L.append("")
    L.append("| config | mean Δ% | std | wins / 240 cells |")
    L.append("|---|---:|---:|---:|")
    rank = []
    for sel in SINGLE_PAIRS:
        sub = df[df.selector.isin([sel, "random"])]
        piv = sub.pivot_table(index=["instance_base", "delay", "budget", "seed"],
                              columns="selector", values="obj")
        if sel not in piv.columns: continue
        piv = piv.dropna(subset=[sel, "random"])
        d = (piv[sel] - piv["random"]) / piv["random"] * 100
        wins = int((d < 0).sum())
        rank.append((sel, d.mean(), d.std(ddof=1), wins, len(d)))
    rank.sort(key=lambda x: x[1])
    for sel, m, s, w, n in rank:
        L.append(f"| `{sel}` | {m:+.3f}% | {s:.3f} | {w} / {n} |")
    L.append("")
    L.append("Interpretation:")
    L.append("- If all 4 rows show a clearly **positive** mean Δ% → ALNS needs the 9-pair pool, single pairs are systematically worse.")
    L.append("- If any row is **near zero or negative** → that single pair is competitive with the 9-pair pool, weakening the diversity argument.")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"\nWrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
