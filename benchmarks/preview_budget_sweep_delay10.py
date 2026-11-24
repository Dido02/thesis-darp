#!/usr/bin/env python3
"""Mid-run preview for the delay10 budget sweep — plots one instance with all
5 selectors (random, roulette, 3 ML variants).

Usage:
    python benchmarks/preview_budget_sweep_delay10.py                                # default 01_min
    python benchmarks/preview_budget_sweep_delay10.py --instance manhattan_05_min_delay10_min
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "bench_results_budget_sweep_delay10" / "raw_logs"

OBJ_RE  = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")

# Match: dynamic__<instance>__<selector>__b<N>__s<N>.log
# selector can contain __ (e.g. ml_20f_t07__T0.10) so we anchor on the budget
# token at the end and capture greedily up to that.
NAME_TMPL = r"^dynamic__{inst}__(.+)__b(\d+)__s(\d+)\.log$"

STYLE = {
    "random":              {"c": "#888888", "mk": "s", "dodge": 0.74, "label": "random"},
    "roulette":            {"c": "#1f77b4", "mk": "o", "dodge": 0.80, "label": "roulette"},
    "roulette_persist":    {"c": "#9467bd", "mk": "P", "dodge": 0.86, "label": "roulette (persist)"},
    "ml_20f_t05":          {"c": "#2ca02c", "mk": "D", "dodge": 0.92, "label": "ml/20f_t05"},
    "ml_20f_t05__T0.10":   {"c": "#2ca02c", "mk": "P", "dodge": 0.99, "label": "ml/20f_t05 T=0.10"},
    "ml_20f_ei_log":       {"c": "#ff7f0e", "mk": "v", "dodge": 1.06, "label": "ml/20f_ei_log"},
    "ml_20f_ei_log__T0.10":{"c": "#ff7f0e", "mk": "*", "dodge": 1.13, "label": "ml/20f_ei_log T=0.10"},
    "ml_20f_t07":          {"c": "#d62728", "mk": "^", "dodge": 1.20, "label": "ml/20f_t07"},
    "ml_20f_t07__T0.10":   {"c": "#d62728", "mk": "X", "dodge": 1.28, "label": "ml/20f_t07 T=0.10"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="manhattan_01_min_delay10_min")
    args = ap.parse_args()

    name_re = re.compile(NAME_TMPL.format(inst=re.escape(args.instance)))

    data: dict[tuple[str, int], list[float]] = {}
    n_files = 0
    for p in sorted(LOGS.glob(f"dynamic__{args.instance}__*.log")):
        m = name_re.match(p.name)
        if not m:
            continue
        n_files += 1
        sel, budget = m.group(1), int(m.group(2))
        mo = OBJ_RE.search(p.read_text())
        if not mo:
            continue
        data.setdefault((sel, budget), []).append(float(mo.group(1)))

    n_parsed = sum(len(v) for v in data.values())
    print(f"delay10 dynamic / {args.instance}")
    print(f"Files scanned: {n_files}, successfully parsed: {n_parsed}\n")
    for (sel, b), objs in sorted(data.items()):
        sd = stdev(objs) if len(objs) > 1 else 0
        print(f"  {sel:<22} b={b:>5}  mean={mean(objs):.1f}  std={sd:.1f}  (n={len(objs)})")

    if not data:
        raise SystemExit("\nNo runs parsed — nothing to plot yet.")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lo, all_hi = float("inf"), 0.0
    for sel in STYLE:
        pts = sorted([(b, data[(s, b)]) for (s, b) in data if s == sel])
        if not pts:
            continue
        x_raw = [b for b, _ in pts]
        x     = [b * STYLE[sel]["dodge"] for b in x_raw]
        mu    = [mean(objs) for _, objs in pts]
        sd    = [stdev(objs) if len(objs) > 1 else 0 for _, objs in pts]
        n_per = [len(objs) for _, objs in pts]
        ax.errorbar(
            x, mu, yerr=sd,
            marker=STYLE[sel]["mk"], markersize=8, capsize=4,
            color=STYLE[sel]["c"], linewidth=1.6, elinewidth=1.2,
            label=f"{STYLE[sel]['label']}  (n={'/'.join(str(n) for n in n_per)})",
        )
        all_lo = min(all_lo, *[m - s for m, s in zip(mu, sd)])
        all_hi = max(all_hi, *[m + s for m, s in zip(mu, sd)])

    pad = (all_hi - all_lo) * 0.10
    ax.set_ylim(all_lo - pad, all_hi + pad)
    ax.set_xscale("log")
    ticks = sorted({b for _, b in data})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t} ms" for t in ticks])
    ax.set_xlabel("epoch budget (log scale)")
    ax.set_ylabel("final objective: total travel  (lower = better)")
    ax.set_title(f"Budget sweep PREVIEW  -  delay10 dynamic / {args.instance}  (5 seeds, fair budget)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()

    out = LOGS.parent / f"preview_dynamic_{args.instance.replace('manhattan_', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
