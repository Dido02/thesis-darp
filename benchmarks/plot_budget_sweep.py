#!/usr/bin/env python3
"""Plot final-obj vs budget curves from launch_budget_sweep.py results.

For each instance, draws one chart with 3 selector lines (random, roulette, ml),
y = final obj (lower = better), x = budget (log-scale). Error bars show seed std.

Usage:
    python benchmarks/plot_budget_sweep.py                              # both modes
    python benchmarks/plot_budget_sweep.py --mode dynamic                # dynamic only
    python benchmarks/plot_budget_sweep.py --results-dir bench_results_budget_sweep
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent


SELECTOR_COLORS = {
    "random":            "#888888",
    "roulette":          "#1f77b4",
    "ml":                "#d62728",
}


def load(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["budget"] = int(r["budget"])
        r["seed"]   = int(r["seed"])
        r["obj"]    = float(r["obj"]) if r["obj"] else None
    return [r for r in rows if r["obj"] is not None]


def plot_mode(rows: list[dict], mode: str, out_path: Path) -> None:
    instances = sorted({r["instance"] for r in rows})
    selectors = ["random", "roulette", "ml"]
    budgets   = sorted({r["budget"] for r in rows})

    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 4.5), sharey=False)
    if len(instances) == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        for sel in selectors:
            label = next((r["selector_label"] for r in rows
                          if r["selector"] == sel and r["instance"] == inst), sel)
            x, mu, sd = [], [], []
            for b in budgets:
                objs = [r["obj"] for r in rows
                        if r["mode"] == mode and r["instance"] == inst
                        and r["selector"] == sel and r["budget"] == b]
                if not objs:
                    continue
                x.append(b)
                mu.append(mean(objs))
                sd.append(stdev(objs) if len(objs) > 1 else 0.0)
            if not x:
                continue
            ax.errorbar(x, mu, yerr=sd, marker="o", capsize=3,
                        color=SELECTOR_COLORS[sel], label=label, linewidth=1.5)

        ax.set_xscale("log")
        ax.set_xlabel(f"budget ({'ms' if mode == 'dynamic' else 'iterations'})")
        ax.set_ylabel("final objective (lower is better)")
        ax.set_title(inst.replace("manhattan_", "").replace("_delay03_min", " (delay03)"))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Budget sweep — {mode} mode ({len(rows)} runs, 5 seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "bench_results_budget_sweep"))
    ap.add_argument("--mode", choices=("both", "dynamic", "static"), default="both")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    csv_path = rd / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"No results.csv at {csv_path}. Run launch_budget_sweep.py first.")

    rows = load(csv_path)
    print(f"Loaded {len(rows)} successful runs from {csv_path}")

    for mode in ("dynamic", "static"):
        if args.mode not in ("both", mode):
            continue
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            print(f"  no rows for mode={mode}, skipping")
            continue
        plot_mode(mode_rows, mode, rd / f"budget_sweep_{mode}.png")


if __name__ == "__main__":
    main()
