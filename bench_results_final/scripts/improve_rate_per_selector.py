#!/usr/bin/env python3
"""Improvement rate per selector — evening bench (BENCH_FINAL).

Reads bench_results_final/raw_logs/*.log, sums improve/iters per run,
compares 5 selectors (random, roulette, ml_20f_t07, ml_20f_t05,
ml_20f_ei_log).

Key question: do ML selectors make MORE improving moves than random and
roulette on in-cohort evening data?
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
LOGS = ROOT / "bench_results_final" / "raw_logs"
OUT  = ROOT / "bench_results_final" / "IMPROVE_RATE_VS_RANDOM_ROULETTE.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

SELECTORS_ORDER = [
    "random", "roulette",
    "ml_20f_t07__T0.10",
    "ml_20f_t05__T0.10",
    "ml_20f_ei_log__T0.10",
]
DELAYS = [3, 5, 10]
INSTANCES = ["30_s", "01_min", "02_min", "05_min"]
BUDGETS = [200, 1000, 5000, 10000]


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
        "iters": it, "improve": im, "best": bst, "sa": sa,
        "improve_rate": im / it,
        "best_rate": bst / it,
        "worsen_accept_rate": sa / max(it - im, 1),
    }


def main():
    rows = []
    for p in sorted(LOGS.glob("*.log")):
        r = parse_log(p)
        if r: rows.append(r)
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} logs")
    print(df.groupby("selector").size().to_string())

    L = []
    L.append("# Improvement Rate — evening bench (BENCH_FINAL)")
    L.append("")
    L.append("**Bench:** `bench_results_final` (18-00 evening Manhattan)")
    L.append("")
    L.append("**Metric:** improving iterations / total iterations (summed across all epochs of a run).")
    L.append("")
    L.append("Companion to `bench_results_morning_2026-05-21/IMPROVE_RATE_VS_RANDOM_ROULETTE.md`. On morning data ML showed −0.01 anti-skill; here we test whether the same selectors achieve **positive** improve-rate lift in-cohort.")
    L.append("")

    L.append("## 1. Overall improve rate per selector (averaged across all cells)")
    L.append("")
    L.append("| selector | mean improve rate | std | n logs |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        sub = df[df.selector == sel]
        if len(sub) == 0:
            L.append(f"| `{sel}` | n/a | n/a | 0 |"); continue
        L.append(f"| `{sel}` | {sub.improve_rate.mean():.4f} | {sub.improve_rate.std(ddof=1):.4f} | {len(sub)} |")
    L.append("")

    L.append("## 2. Mean improve rate per delay")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        cells = []
        for d in DELAYS:
            sub = df[(df.selector == sel) & (df.delay == d)]
            cells.append("n/a" if len(sub) == 0 else f"{sub.improve_rate.mean():.4f}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Paired Δ vs `random` per delay")
    L.append("")
    L.append("Positive Δ = selector improves more often than random.")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        if sel == "random": continue
        cells = []
        for d in DELAYS:
            sub = df[(df.delay == d) & (df.selector.isin([sel, "random"]))]
            piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                                   columns="selector", values="improve_rate")
            if sel not in piv.columns or "random" not in piv.columns:
                cells.append("n/a"); continue
            piv = piv.dropna(subset=[sel, "random"])
            if len(piv) == 0: cells.append("n/a"); continue
            delta = piv[sel] - piv["random"]
            sign = "✓" if delta.mean() > 0 else "✗"
            cells.append(f"{delta.mean():+.4f} {sign} (n={len(piv)})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 4. Paired Δ vs `roulette` per delay")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        if sel == "roulette": continue
        cells = []
        for d in DELAYS:
            sub = df[(df.delay == d) & (df.selector.isin([sel, "roulette"]))]
            piv = sub.pivot_table(index=["instance_base", "budget", "seed"],
                                   columns="selector", values="improve_rate")
            if sel not in piv.columns or "roulette" not in piv.columns:
                cells.append("n/a"); continue
            piv = piv.dropna(subset=[sel, "roulette"])
            if len(piv) == 0: cells.append("n/a"); continue
            delta = piv[sel] - piv["roulette"]
            sign = "✓" if delta.mean() > 0 else "✗"
            cells.append(f"{delta.mean():+.4f} {sign} (n={len(piv)})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 5. Best-found rate per selector (best updates / total iters)")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        cells = []
        for d in DELAYS:
            sub = df[(df.selector == sel) & (df.delay == d)]
            cells.append("n/a" if len(sub) == 0 else f"{sub.best_rate.mean():.5f}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 6. Per-instance Δ vs random on delay 5 (most-informative cell)")
    L.append("")
    L.append("Δ = mean improve rate (selector − random), averaged across budgets and seeds.")
    L.append("")
    L.append("| instance | " + " | ".join(f"`{s}`" for s in SELECTORS_ORDER if s != "random") + " |")
    L.append("|---|" + "|".join("---:" for _ in SELECTORS_ORDER if _ != "random") + "|")
    for inst in INSTANCES:
        cells = []
        for sel in SELECTORS_ORDER:
            if sel == "random": continue
            sub = df[(df.selector.isin([sel, "random"])) & (df.delay == 5) & (df.instance_base == inst)]
            piv = sub.pivot_table(index=["budget", "seed"], columns="selector", values="improve_rate")
            if sel not in piv.columns or "random" not in piv.columns:
                cells.append("n/a"); continue
            piv = piv.dropna(subset=[sel, "random"])
            if len(piv) == 0: cells.append("n/a"); continue
            d = piv[sel] - piv["random"]
            cells.append(f"{d.mean():+.4f}")
        L.append(f"| {inst} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 7. Verdict")
    L.append("")
    ml_sels = [s for s in SELECTORS_ORDER if s.startswith("ml_")]
    ml_mean = df[df.selector.isin(ml_sels)].improve_rate.mean()
    rnd_mean = df[df.selector == "random"].improve_rate.mean()
    rou_mean = df[df.selector == "roulette"].improve_rate.mean()
    L.append(f"- Mean improve rate across all ML configs: **{ml_mean:.4f}**")
    L.append(f"- Mean improve rate for `random`:  **{rnd_mean:.4f}**")
    L.append(f"- Mean improve rate for `roulette`: **{rou_mean:.4f}**")
    if ml_mean > max(rnd_mean, rou_mean):
        L.append(f"- ✓ ML selectors improve **more often** than both baselines (gap vs random: {ml_mean - rnd_mean:+.4f}).")
    elif ml_mean < min(rnd_mean, rou_mean):
        L.append(f"- ✗ ML selectors improve **less often** than both baselines (gap vs random: {ml_mean - rnd_mean:+.4f}) — anti-skill, same shape as morning.")
    else:
        L.append(f"- ≈ ML selectors between random ({rnd_mean:.4f}) and roulette ({rou_mean:.4f}) — no clear advantage on improve-rate.")
    L.append("")
    L.append("Compare to morning result (`bench_results_morning_2026-05-21/IMPROVE_RATE_VS_RANDOM_ROULETTE.md` §7): morning ML=0.0689, random=0.0790, roulette=0.0782 (ML loses by ~0.01 on morning).")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
