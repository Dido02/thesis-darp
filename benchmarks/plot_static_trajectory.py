"""Plot best-obj trajectory across iterations, parsed from static-bench stdout.log.

Reads `iter N: current=X best=Y temp=Z` lines from each run's stdout.log,
aggregates `best` across seeds per (instance, selector), and plots one subplot
per instance (selectors as colored lines).

Usage:
    python benchmarks/plot_static_trajectory.py \\
        --results-dir bench_results_v5_static \\
        --iters 5000 \\
        --out plots/static_trajectory_5000.png
"""
from __future__ import annotations

import argparse
import re
import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


_PAT = re.compile(
    r"^\s*iter\s+(\d+):\s+current=(\d+(?:\.\d+)?),\s+best=(\d+(?:\.\d+)?),\s+temp=([\d.eE+-]+)",
    re.M,
)
_PAT_INIT = re.compile(r"^Initial solution: obj=(\d+(?:\.\d+)?)", re.M)
_PAT_FINAL = re.compile(r"^Final objective:\s+(\d+(?:\.\d+)?)", re.M)


SELECTOR_COLORS = {
    "random":               "#888888",
    "roulette":             "#000000",
    "20f_regr":             "#1f77b4",
    "20f_regr__T0.10":      "#aec7e8",
    "20f_t07":              "#d62728",
    "20f_t07__T0.10":       "#ff9896",
    "20f_ei_log":           "#2ca02c",
}
SELECTOR_LINESTYLES = {
    "random":               (":", 1.5),
    "roulette":             ("-",  2.0),
    "20f_regr":             ("-",  1.5),
    "20f_regr__T0.10":      ("--", 1.5),
    "20f_t07":              ("-",  1.5),
    "20f_t07__T0.10":       ("--", 1.5),
    "20f_ei_log":           ("-.", 1.5),
}


def parse_log(path: Path) -> tuple[float, list[tuple[int, float]], float]:
    """Return (init_obj, [(iter, best_obj), ...], final_obj)."""
    text = path.read_text()
    init_m = _PAT_INIT.search(text)
    init_obj = float(init_m.group(1)) if init_m else 0.0
    final_m = _PAT_FINAL.search(text)
    final_obj = float(final_m.group(1)) if final_m else 0.0
    pts = [(int(m.group(1)), float(m.group(3))) for m in _PAT.finditer(text)]
    return init_obj, pts, final_obj


def selector_label_for(stdout_path: Path, results_dir: Path) -> str:
    parts = stdout_path.resolve().relative_to(results_dir.resolve()).parts
    runs_idx = parts.index("runs")
    return "/".join(parts[:runs_idx]).replace("baselines/", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bench_results_v5_static")
    ap.add_argument("--iters", type=int, default=5000,
                    help="Iter count to load (default 5000)")
    ap.add_argument("--out", default="plots/static_trajectory_5000.png")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # (instance, selector_label) -> {iter -> [best_obj across seeds]}
    points = defaultdict(lambda: defaultdict(list))
    initial_per_inst = defaultdict(list)
    final_per_cell = defaultdict(list)

    pattern = str(results_dir / f"**/runs/*iters{args.iters}*/stdout.log")
    for log_path in glob.glob(pattern, recursive=True):
        log = Path(log_path)
        sel_label = selector_label_for(log, results_dir)
        # parse instance from run dir name
        run_dir = log.parent.name
        instance = run_dir.split("__")[0]
        init_obj, pts, final_obj = parse_log(log)
        initial_per_inst[instance].append(init_obj)
        final_per_cell[(instance, sel_label)].append(final_obj)
        for it, best in pts:
            points[(instance, sel_label)][it].append(best)

    instances = sorted({inst for (inst, _) in points})
    selectors = sorted({sel for (_, sel) in points})
    print(f"Loaded {len(points)} (instance, selector) trajectories "
          f"({len(instances)} instances × {len(selectors)} selectors)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for ax, inst in zip(axes, instances):
        inst_short = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")

        # Plot initial obj as a dashed horizontal reference
        if initial_per_inst[inst]:
            init_mean = mean(initial_per_inst[inst])
            ax.axhline(y=init_mean, linestyle=":", color="gray", alpha=0.5,
                       label=f"initial ({init_mean:,.0f})")

        for sel in selectors:
            if (inst, sel) not in points: continue
            iters_dict = points[(inst, sel)]
            xs = sorted(iters_dict.keys())
            ys = [mean(iters_dict[x]) for x in xs]
            sds = [stdev(iters_dict[x]) if len(iters_dict[x]) > 1 else 0.0 for x in xs]
            color = SELECTOR_COLORS.get(sel, "purple")
            ls, lw = SELECTOR_LINESTYLES.get(sel, ("-", 1.5))
            # final-obj annotation at end-of-run
            final_mean = mean(final_per_cell.get((inst, sel), [ys[-1]] if ys else [0]))
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                    label=f"{sel} (final={final_mean:,.0f})", alpha=0.85)
            ax.fill_between(xs,
                            [y - s for y, s in zip(ys, sds)],
                            [y + s for y, s in zip(ys, sds)],
                            color=color, alpha=0.12, linewidth=0)

        ax.set_title(f"{inst_short}", fontsize=12, fontweight="bold")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best obj (mean over 3 seeds)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=1)

    fig.suptitle(
        f"Best-obj trajectory — static mode @ {args.iters} iters "
        f"(7 selectors, 3 seeds averaged)",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Also produce a "Δ vs random" plot — same axes, but plotting ML/roulette - random
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes2 = axes2.flatten()
    for ax, inst in zip(axes2, instances):
        inst_short = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")
        rand_dict = points.get((inst, "random"), {})
        if not rand_dict:
            ax.set_title(f"{inst_short} — no random baseline"); continue
        rand_iters = sorted(rand_dict.keys())
        rand_means = {x: mean(rand_dict[x]) for x in rand_iters}

        for sel in selectors:
            if sel == "random" or (inst, sel) not in points: continue
            iters_dict = points[(inst, sel)]
            xs, ys, sds = [], [], []
            for x in sorted(iters_dict.keys()):
                if x not in rand_dict or rand_means[x] <= 0:
                    continue
                # Per-seed paired deltas → mean and std of that paired sample
                rand_seeds = rand_dict[x]    # list of best-obj across seeds for random
                sel_seeds = iters_dict[x]
                # Pair by index (seed order is preserved by glob sort)
                k = min(len(rand_seeds), len(sel_seeds))
                if k < 1: continue
                pdeltas = [100 * (sel_seeds[i] - rand_seeds[i]) / rand_seeds[i]
                           for i in range(k) if rand_seeds[i] > 0]
                if not pdeltas: continue
                xs.append(x)
                ys.append(mean(pdeltas))
                sds.append(stdev(pdeltas) if len(pdeltas) > 1 else 0.0)
            color = SELECTOR_COLORS.get(sel, "purple")
            ls, lw = SELECTOR_LINESTYLES.get(sel, ("-", 1.5))
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                    label=sel, alpha=0.85)
            ax.fill_between(xs,
                            [y - s for y, s in zip(ys, sds)],
                            [y + s for y, s in zip(ys, sds)],
                            color=color, alpha=0.12, linewidth=0)
        ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
        ax.set_title(f"{inst_short}", fontsize=12, fontweight="bold")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Δ% vs random (negative = better)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)

    fig2.suptitle(
        f"Best-obj Δ% vs random — static mode @ {args.iters} iters",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig2.tight_layout()
    delta_path = out_path.parent / out_path.name.replace(".png", "_delta.png")
    fig2.savefig(delta_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {delta_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
