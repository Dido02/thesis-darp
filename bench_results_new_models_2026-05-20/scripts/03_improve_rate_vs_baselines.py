#!/usr/bin/env python3
"""Improvement rate — 4 new ML configs vs random/roulette baselines.

New_models bench has only ML configs (no random/roulette). The new ML
runs were on evening 18-00 instances, so we pull the random/roulette
baselines from bench_results_final/raw_logs (same evening instances).

Joins by (instance, delay, budget, seed) — exact paired comparison.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
NEW  = ROOT / "bench_results_new_models_2026-05-20" / "raw_logs"
FIN  = ROOT / "bench_results_final" / "raw_logs"
OUT  = ROOT / "bench_results_new_models_2026-05-20" / "IMPROVE_RATE_VS_RANDOM_ROULETTE.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

NEW_ML = [
    "ml_forest_d8_K20__T0.10",
    "ml_forest_d8_K16__T0.10",
    "ml_tree_d4_K4__T0.10",
    "ml_tree_d8_K16__T0.10",
]
BASELINES = ["random", "roulette"]
DEPLOYED  = "ml_20f_t07__T0.10"
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
        "iters": it, "improve": im, "best": bst, "sa": sa,
        "improve_rate": im / it, "best_rate": bst / it,
        "worsen_accept_rate": sa / max(it - im, 1),
    }


def main():
    rows_new = [r for p in sorted(NEW.glob("*.log")) if (r := parse_log(p))]
    rows_fin = [r for p in sorted(FIN.glob("*.log"))
                if (r := parse_log(p)) and r["selector"] in BASELINES + [DEPLOYED]]
    df = pd.DataFrame(rows_new + rows_fin)
    print(f"new_models rows: {len(rows_new)}, final-baseline rows: {len(rows_fin)}")
    print(df.groupby("selector").size().to_string())

    all_sels = BASELINES + [DEPLOYED] + NEW_ML

    L = []
    L.append("# Improvement Rate — new ML configs vs random/roulette baselines")
    L.append("")
    L.append("**Bench:** 4 new ML configs (`bench_results_new_models_2026-05-20`) on 18-00 evening Manhattan instances. Baselines (`random`, `roulette`) and deployed (`ml_20f_t07__T0.10`) pulled from `bench_results_final/raw_logs` (same evening instances, same grid).")
    L.append("")
    L.append("**Metric:** improving iterations / total iterations.")
    L.append("")

    L.append("## 1. Overall improve rate per selector")
    L.append("")
    L.append("| selector | mean improve rate | std | n logs |")
    L.append("|---|---:|---:|---:|")
    for sel in all_sels:
        sub = df[df.selector == sel]
        if len(sub) == 0:
            L.append(f"| `{sel}` | n/a | n/a | 0 |"); continue
        L.append(f"| `{sel}` | {sub.improve_rate.mean():.4f} | {sub.improve_rate.std(ddof=1):.4f} | {len(sub)} |")
    L.append("")

    L.append("## 2. Mean improve rate per delay")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in all_sels:
        cells = []
        for d in DELAYS:
            sub = df[(df.selector == sel) & (df.delay == d)]
            cells.append("n/a" if len(sub) == 0 else f"{sub.improve_rate.mean():.4f}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Paired Δ vs `random` per delay (positive = selector improves more often)")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    for sel in [DEPLOYED] + NEW_ML:
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
    for sel in [DEPLOYED] + NEW_ML:
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

    L.append("## 5. Verdict")
    L.append("")
    ml_pool = [DEPLOYED] + NEW_ML
    ml_mean = df[df.selector.isin(ml_pool)].improve_rate.mean()
    rnd_mean = df[df.selector == "random"].improve_rate.mean()
    rou_mean = df[df.selector == "roulette"].improve_rate.mean()
    L.append(f"- Mean improve rate across deployed + 4 new ML configs: **{ml_mean:.4f}**")
    L.append(f"- Mean improve rate for `random`:   **{rnd_mean:.4f}**")
    L.append(f"- Mean improve rate for `roulette`: **{rou_mean:.4f}**")
    if ml_mean > max(rnd_mean, rou_mean):
        L.append(f"- ✓ ML selectors improve **more often** than both baselines on evening (gap vs random: {ml_mean - rnd_mean:+.4f}).")
    elif ml_mean < min(rnd_mean, rou_mean):
        L.append(f"- ✗ ML selectors improve **less often** than both baselines on evening (gap vs random: {ml_mean - rnd_mean:+.4f}).")
    else:
        L.append(f"- ≈ ML between random ({rnd_mean:.4f}) and roulette ({rou_mean:.4f}).")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
