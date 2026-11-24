#!/usr/bin/env python3
"""Compare improvement rate (improving iters / total iters) across selectors
on the morning bench. Tests whether ML selectors actually pick operators
that improve the current solution MORE OFTEN than random/roulette.

Reads existing logs only. No bench runs.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
LOGS = ROOT / "bench_results_morning_2026-05-21" / "raw_logs"
OUT  = ROOT / "bench_results_morning_2026-05-21" / "IMPROVE_RATE_VS_RANDOM_ROULETTE.md"

_RE_NAME = re.compile(r"dynamic__manhattan_morning_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

SELECTORS_ORDER = [
    "random", "roulette",
    "ml_20f_t07__T0.10",
    "ml_forest_d8_K20__T0.10",
    "ml_forest_d8_K16__T0.10",
    "ml_tree_d4_K4__T0.10",
    "ml_tree_d8_K16__T0.10",
]
DELAYS = [3, 5, 10]
INSTANCES = ["30_s", "01_min", "02_min", "05_min"]
BUDGETS = [200, 1000, 5000, 10000]


def parse_log(p: Path):
    m = _RE_NAME.match(p.name)
    if not m: return None
    total_iters = 0; total_improve = 0; total_best = 0; total_sa = 0; n_epochs = 0
    for line in p.read_text().splitlines():
        em = _RE_EPOCH.search(line)
        if em:
            it, im, bst, sa = map(int, em.groups())
            total_iters += it; total_improve += im
            total_best += bst; total_sa += sa
            n_epochs += 1
    if total_iters == 0:
        return None
    return {
        "instance_base": m.group(1), "delay": int(m.group(2)),
        "selector": m.group(3), "budget": int(m.group(4)),
        "seed": int(m.group(5)),
        "iters": total_iters, "improve": total_improve,
        "best": total_best, "sa": total_sa,
        "n_epochs": n_epochs,
        "improve_rate": total_improve / total_iters,
        "best_rate": total_best / total_iters,
        "sa_rate": total_sa / total_iters,
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
    L.append("# Improvement Rate — ML selectors vs random vs roulette")
    L.append("")
    L.append("**Bench:** `bench_results_morning_2026-05-21` (07-00 morning Manhattan)")
    L.append("")
    L.append("**Metric:** improving iterations / total iterations (summed across all epochs of a run).")
    L.append("")
    L.append("**Interpretation:**")
    L.append("- If ML > random/roulette → ML selects operators that improve the current solution more often.")
    L.append("- If ML ≈ random → ML provides no operator-quality advantage.")
    L.append("- If ML < random → ML picks operators that fail to improve more often (anti-skill).")
    L.append("")

    # §1 — overall per selector
    L.append("## 1. Overall improve rate per selector (averaged across all cells)")
    L.append("")
    L.append("| selector | mean improve rate | std | n logs |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        sub = df[df.selector == sel]
        if len(sub) == 0:
            L.append(f"| `{sel}` | n/a | n/a | 0 |")
            continue
        L.append(f"| `{sel}` | {sub.improve_rate.mean():.4f} | {sub.improve_rate.std(ddof=1):.4f} | {len(sub)} |")
    L.append("")

    # §2 — per delay
    L.append("## 2. Mean improve rate per delay")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        cells = []
        for d in DELAYS:
            sub = df[(df.selector == sel) & (df.delay == d)]
            if len(sub) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{sub.improve_rate.mean():.4f}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §3 — paired Δ vs random (per delay, averaged over instance/budget/seed)
    L.append("## 3. Paired Δ vs `random` per delay")
    L.append("")
    L.append("Positive Δ = selector improves more often than random (more operator skill).")
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
            if len(piv) == 0:
                cells.append("n/a"); continue
            delta = piv[sel] - piv["random"]
            sign = "✓" if delta.mean() > 0 else "✗"
            cells.append(f"{delta.mean():+.4f} {sign} (n={len(piv)})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §4 — paired Δ vs roulette
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
            if len(piv) == 0:
                cells.append("n/a"); continue
            delta = piv[sel] - piv["roulette"]
            sign = "✓" if delta.mean() > 0 else "✗"
            cells.append(f"{delta.mean():+.4f} {sign} (n={len(piv)})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §5 — best-found-rate (the "best" counter is the headline)
    L.append("## 5. Best-found rate per selector (best updates / total iters)")
    L.append("")
    L.append("This is the rate at which the selector finds NEW best-known solutions. The smallest absolute number but the most bench-relevant.")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in SELECTORS_ORDER:
        cells = []
        for d in DELAYS:
            sub = df[(df.selector == sel) & (df.delay == d)]
            if len(sub) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{sub.best_rate.mean():.5f}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §6 — per instance breakdown (only on the most-relevant: ml_20f_t07 vs random vs roulette)
    L.append("## 6. Improve rate per instance, delay 5 (most informative cell)")
    L.append("")
    L.append("| instance | random | roulette | ml_20f_t07 | ml_forest_d8_K20 | ml_forest_d8_K16 | ml_tree_d4_K4 | ml_tree_d8_K16 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for inst in INSTANCES:
        cells = []
        for sel in SELECTORS_ORDER:
            sub = df[(df.selector == sel) & (df.delay == 5) & (df.instance_base == inst)]
            if len(sub) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{sub.improve_rate.mean():.4f}")
        L.append(f"| {inst} | " + " | ".join(cells) + " |")
    L.append("")

    # §7 — summary verdict
    L.append("## 7. Verdict")
    L.append("")
    # Compute headline: avg ML improve rate vs random
    ml_mean = df[df.selector.isin([s for s in SELECTORS_ORDER if s.startswith("ml_")])].improve_rate.mean()
    rnd_mean = df[df.selector == "random"].improve_rate.mean()
    rou_mean = df[df.selector == "roulette"].improve_rate.mean()
    L.append(f"- Mean improve rate across all ML configs: **{ml_mean:.4f}**")
    L.append(f"- Mean improve rate for `random`: **{rnd_mean:.4f}**")
    L.append(f"- Mean improve rate for `roulette`: **{rou_mean:.4f}**")
    if ml_mean > max(rnd_mean, rou_mean):
        L.append(f"- ✓ ML selectors improve more often than both classical baselines (gap vs random: {ml_mean - rnd_mean:+.4f}).")
    elif ml_mean < min(rnd_mean, rou_mean):
        L.append(f"- ✗ ML selectors improve LESS often than both classical baselines (gap vs random: {ml_mean - rnd_mean:+.4f}). Anti-skill.")
    else:
        L.append(f"- ≈ ML selectors are between random ({rnd_mean:.4f}) and roulette ({rou_mean:.4f}) — no clear advantage on improve-rate metric.")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
