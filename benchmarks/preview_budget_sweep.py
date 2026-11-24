#!/usr/bin/env python3
"""Mid-run preview of the budget sweep — plots one (mode, instance) cell from
raw_logs/ without waiting for the full sweep to write results.csv.

Usage:
    python benchmarks/preview_budget_sweep.py                         # dynamic / 01_min
    python benchmarks/preview_budget_sweep.py --instance manhattan_02_min_delay03_min
    python benchmarks/preview_budget_sweep.py --mode static --instance manhattan_30_s_delay03_min
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "bench_results_budget_sweep" / "raw_logs"

OBJ_RE = {
    "dynamic": re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)"),
    "static":  re.compile(r"Final objective:\s+(-?\d+(?:\.\d+)?)"),
}

STYLE = {
    "random":            {"c": "#888888", "mk": "s", "dodge": 0.88, "label": "random"},
    "roulette":          {"c": "#1f77b4", "mk": "o", "dodge": 0.95, "label": "roulette"},
    "roulette_persist":  {"c": "#9467bd", "mk": "P", "dodge": 1.02, "label": "roulette (persist)"},
    "ml_20f_t07__T0.10": {"c": "#d62728", "mk": "^", "dodge": 1.10, "label": "ml/20f_t07 T=0.10"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="manhattan_01_min_delay03_min")
    ap.add_argument("--mode", choices=("dynamic", "static"), default="dynamic")
    args = ap.parse_args()

    name_re = re.compile(
        rf"^{args.mode}__{re.escape(args.instance)}__(.+)__b(\d+)__s(\d+)\.log$"
    )
    obj_re = OBJ_RE[args.mode]

    data: dict[tuple[str, int], list[float]] = {}
    n_files = 0
    for p in sorted(LOGS.glob(f"{args.mode}__{args.instance}__*.log")):
        m = name_re.match(p.name)
        if not m:
            continue
        n_files += 1
        sel, budget = m.group(1), int(m.group(2))
        mo = obj_re.search(p.read_text())
        if not mo:
            continue
        data.setdefault((sel, budget), []).append(float(mo.group(1)))

    n_parsed = sum(len(v) for v in data.values())
    print(f"{args.mode} / {args.instance}")
    print(f"Files scanned: {n_files}, successfully parsed: {n_parsed}\n")
    for (sel, b), objs in sorted(data.items()):
        sd = stdev(objs) if len(objs) > 1 else 0
        print(f"  {sel:<22} b={b:>5}  mean={mean(objs):.1f}  std={sd:.1f}  (n={len(objs)})")

    if not data:
        raise SystemExit("\nNo runs parsed — nothing to plot yet.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_lo, all_hi = float("inf"), 0.0
    for sel in ("random", "roulette", "roulette_persist", "ml_20f_t07__T0.10"):
        pts = sorted([(b, data[(s, b)]) for (s, b) in data if s == sel])
        if not pts:
            continue
        x_raw = [b for b, _ in pts]
        x = [b * STYLE[sel]["dodge"] for b in x_raw]
        mu = [mean(objs) for _, objs in pts]
        sd = [stdev(objs) if len(objs) > 1 else 0 for _, objs in pts]
        n_per = [len(objs) for _, objs in pts]
        label_n = f"n={'/'.join(str(n) for n in n_per)}"
        ax.errorbar(
            x, mu, yerr=sd,
            marker=STYLE[sel]["mk"], markersize=8, capsize=4,
            color=STYLE[sel]["c"], linewidth=1.8, elinewidth=1.4,
            label=f"{STYLE[sel]['label']}  ({label_n})",
        )
        all_lo = min(all_lo, *[m - s for m, s in zip(mu, sd)])
        all_hi = max(all_hi, *[m + s for m, s in zip(mu, sd)])

    pad = (all_hi - all_lo) * 0.10
    ax.set_ylim(all_lo - pad, all_hi + pad)
    ax.set_xscale("log")
    ticks = sorted({b for _, b in data})
    ax.set_xticks(ticks)
    unit = "ms" if args.mode == "dynamic" else "iters"
    ax.set_xticklabels([f"{t} {unit}" for t in ticks])
    ax.set_xlabel(f"{'epoch budget' if args.mode == 'dynamic' else 'iterations'} (log scale)")
    ax.set_ylabel("final objective: total travel  (lower = better)"
                  if args.mode == "dynamic"
                  else "final objective  (lower = better)")
    ax.set_title(f"Budget sweep PREVIEW  -  {args.mode} / {args.instance}  (5 seeds, fair budget)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout()

    out = LOGS.parent / f"preview_{args.mode}_{args.instance.replace('manhattan_', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
