#!/usr/bin/env python3
"""Search for any (model, sub-grid) where improve_rate beats random/roulette.

Scans the full BENCH_FINAL + new_models grid:
  - 7 ML configs × 2 baselines × 4 budgets × 3 delays × 4 instances
For each (selector, slice), computes paired Δ vs baseline + Wilcoxon p.
Reports the cells where Δ > 0.

Slices tested:
  (a) per-budget (delays + instances aggregated)
  (b) per-(budget, delay)
  (c) per-(budget, instance)
  (d) per-(delay, instance) — averaged over budgets
  (e) per-(budget, delay, instance) — finest granularity (n=5 seeds)
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
FIN = ROOT / "bench_results_final" / "raw_logs"
NEW = ROOT / "bench_results_new_models_2026-05-20" / "raw_logs"
OUT = ROOT / "bench_results_final" / "IMPROVE_RATE_WINNERS_SEARCH.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

ML_SELECTORS = [
    "ml_20f_t07__T0.10", "ml_20f_t05__T0.10", "ml_20f_ei_log__T0.10",
    "ml_forest_d8_K20__T0.10", "ml_forest_d8_K16__T0.10",
    "ml_tree_d4_K4__T0.10", "ml_tree_d8_K16__T0.10",
]
BASELINES = ["random", "roulette"]
BUDGETS = [200, 1000, 5000, 10000]
DELAYS = [3, 5, 10]
INSTANCES = ["30_s", "01_min", "02_min", "05_min"]


def parse_log(p: Path):
    m = _RE_NAME.match(p.name)
    if not m: return None
    it = im = bst = sa = 0
    for ln in p.read_text().splitlines():
        em = _RE_EPOCH.search(ln)
        if em:
            a, b, c, d = map(int, em.groups())
            it += a; im += b; bst += c; sa += d
    if it == 0: return None
    return {
        "instance_base": m.group(1), "delay": int(m.group(2)),
        "selector": m.group(3), "budget": int(m.group(4)), "seed": int(m.group(5)),
        "iters": it, "improve_rate": im / it,
    }


def paired_test(df, sel, baseline, filters: dict):
    sub = df[df.selector.isin([sel, baseline])]
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    # Pivot on whatever isn't in the filter
    free_axes = [a for a in ["instance_base", "delay", "budget", "seed"]
                 if a not in filters]
    piv = sub.pivot_table(index=free_axes, columns="selector",
                          values="improve_rate").dropna()
    if sel not in piv.columns or baseline not in piv.columns or len(piv) < 3:
        return None
    diffs = piv[sel].values - piv[baseline].values
    mean_d = float(diffs.mean())
    if (diffs == 0).all():
        return {"n": len(piv), "delta_pp": mean_d * 100, "p": 1.0}
    try:
        p = float(wilcoxon(piv[sel].values, piv[baseline].values,
                           alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        p = None
    return {"n": len(piv), "delta_pp": mean_d * 100, "p": p}


def main():
    rows = [r for p in sorted(FIN.glob("*.log")) if (r := parse_log(p))]
    rows += [r for p in sorted(NEW.glob("*.log")) if (r := parse_log(p))]
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} logs ({df.selector.nunique()} selectors)")

    L = []
    L.append("# Improve-rate winners — exhaustive search for any positive Δ vs random/roulette")
    L.append("")
    L.append("**Goal:** find any (model, sub-grid) where the ML selector's improve_rate beats a baseline.")
    L.append("")
    L.append("Scans 5 granularities (budget; (budget,delay); (budget,instance); (delay,instance); (budget,delay,instance)) for each of 7 ML configs vs `random` and `roulette`. Reports every cell where Δ > 0, sorted descending.")
    L.append("")

    # ---- 5 granularities ----
    granularities = [
        ("per budget", lambda: [{"budget": b} for b in BUDGETS]),
        ("per (budget, delay)", lambda: [{"budget": b, "delay": d} for b in BUDGETS for d in DELAYS]),
        ("per (budget, instance)", lambda: [{"budget": b, "instance_base": i} for b in BUDGETS for i in INSTANCES]),
        ("per (delay, instance)", lambda: [{"delay": d, "instance_base": i} for d in DELAYS for i in INSTANCES]),
        ("per (budget, delay, instance) [n=5 seeds]",
            lambda: [{"budget": b, "delay": d, "instance_base": i}
                     for b in BUDGETS for d in DELAYS for i in INSTANCES]),
    ]

    for label, gen in granularities:
        L.append(f"## Granularity: {label}")
        L.append("")
        wins = []
        for sel in ML_SELECTORS:
            for baseline in BASELINES:
                for filt in gen():
                    r = paired_test(df, sel, baseline, filt)
                    if r is None or r["delta_pp"] <= 0:
                        continue
                    wins.append({
                        "selector": sel, "baseline": baseline,
                        "filt": ", ".join(f"{k}={v}" for k, v in filt.items()),
                        "delta_pp": r["delta_pp"], "p": r["p"], "n": r["n"],
                        "holm_sig": (r["p"] is not None and r["p"] < 0.05),
                    })
        if not wins:
            L.append("_None — no positive Δ found at this granularity._"); L.append("")
            continue
        wins.sort(key=lambda x: x["delta_pp"], reverse=True)
        L.append(f"Found **{len(wins)}** positive cells (sig: **{sum(1 for w in wins if w['holm_sig'])}**). Top 15:")
        L.append("")
        L.append("| selector | vs | sub-grid | Δ (pp) | n | p | sig? |")
        L.append("|---|---|---|---:|---:|---:|:---:|")
        for w in wins[:15]:
            sig = "✓" if w["holm_sig"] else ""
            p_str = f"{w['p']:.3f}" if w["p"] is not None else "n/a"
            L.append(f"| `{w['selector']}` | `{w['baseline']}` | {w['filt']} | "
                     f"{w['delta_pp']:+.3f} | {w['n']} | {p_str} | {sig} |")
        L.append("")

    # ---- summary: any sig wins? ----
    L.append("## Summary: are there ANY p<0.05 positive cells for improve_rate?")
    L.append("")
    L.append("If the answer is 'no' across all 5 granularities, then no model in the current logs shows the v6 +improve-rate signal — at any sub-grid.")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
