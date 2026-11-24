"""Plot per-epoch Δ% of one or more ML selectors vs random + roulette.

For each (instance, budget, seed, epoch_id), reads obj_after from epoch_metrics.csv
under bench_results_v5_fast(_epoch60)/{fair,unfair}/. Averages across seeds, computes:
    Δ% vs random   = 100 * (sel.obj_after - random.obj_after) / random.obj_after
    Δ% vs roulette = same with roulette as denom

For each (budget, fairness) cell produces TWO figures (4 subplots each, one per
instance): one with all ML configs plotted vs random, one vs roulette.
Negative = ML is better.

Usage:
    python benchmarks/plot_dynamic_delta_per_epoch.py \\
        --results-dir bench_results_v5_fast_epoch60 \\
        --fairness fair \\
        --ml-configs ml/20f_t07__T0.10 ml/20f_t07 ml/20f_regr__T0.10 \\
                     ml/20f_regr ml/20f_ei_log
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent

INSTANCES = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]

CONFIG_COLORS = {
    "ml/20f_t07__T0.10":   "#d62728",
    "ml/20f_t07":          "#ff9896",
    "ml/20f_regr__T0.10":  "#1f77b4",
    "ml/20f_regr":         "#aec7e8",
    "ml/20f_ei_log":       "#2ca02c",
}
CONFIG_MARKERS = {
    "ml/20f_t07__T0.10":   "o",
    "ml/20f_t07":          "s",
    "ml/20f_regr__T0.10":  "D",
    "ml/20f_regr":         "v",
    "ml/20f_ei_log":       "^",
}


def load_obj_after_per_epoch(results_dir: Path, instance: str, budget: int,
                              fairness: str, sel: str) -> dict[int, list[float]]:
    """Return {epoch_id: [obj_after across seeds]} for one (sel, instance, budget) cell."""
    if sel.startswith("ml/"):
        model_short = sel.replace("ml/", "")
        bucket = f"{fairness}/models_v3_full_{model_short}"
        run_sel = "ml"
    else:
        bucket = f"baselines_{fairness}"
        run_sel = sel

    by_epoch: dict[int, list[float]] = defaultdict(list)
    pattern = str(results_dir / bucket / "runs" /
                  f"{instance}__{run_sel}__classical__budget{budget}__seed*")
    for run_dir in sorted(glob.glob(pattern)):
        csv_path = Path(run_dir) / "epoch_metrics.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                eid = int(row.get("epoch_id") or 0)
                obj = float(row.get("obj_after") or 0)
                by_epoch[eid].append(obj)
    return by_epoch


def make_plot(results_dir: Path, fairness: str, budget: int,
              ml_configs: list[str], comparator: str, out_dir: Path) -> Path:
    """Plot per-instance per-epoch Δ% for each ML config vs `comparator` (random or roulette)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes_flat = axes.flatten()

    # Pre-load comparator
    comp_data = {
        inst: load_obj_after_per_epoch(results_dir, inst, budget, fairness, comparator)
        for inst in INSTANCES
    }

    for ax, inst in zip(axes_flat, INSTANCES):
        comp = comp_data[inst]
        ax.axhline(0.0, color="#888", lw=0.8, ls="--")

        for cfg in ml_configs:
            ml = load_obj_after_per_epoch(results_dir, inst, budget, fairness, cfg)
            epochs = sorted(set(comp) & set(ml))
            if not epochs:
                continue
            mean_d, std_d = [], []
            for e in epochs:
                # Pair seeds positionally (load_obj_after_per_epoch sorts by run_dir,
                # so seed1/seed2/seed3 align across selectors).
                paired = list(zip(ml[e], comp[e]))
                per_seed = [100.0 * (m - c) / c for (m, c) in paired if c]
                if not per_seed:
                    mean_d.append(0.0); std_d.append(0.0); continue
                mean_d.append(mean(per_seed))
                std_d.append(stdev(per_seed) if len(per_seed) >= 2 else 0.0)
            color = CONFIG_COLORS.get(cfg, None)
            ax.plot(epochs, mean_d,
                    marker=CONFIG_MARKERS.get(cfg, "o"),
                    color=color,
                    linestyle="-", linewidth=1.4,
                    label=cfg.replace("ml/", ""))
            lower = [m - s for m, s in zip(mean_d, std_d)]
            upper = [m + s for m, s in zip(mean_d, std_d)]
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.15, linewidth=0)

        n_epochs = len(comp)
        ax.set_title(f"{inst.replace('manhattan_', '').replace('_delay03_min', '/03min')}  "
                     f"({n_epochs} epochs)")
        ax.set_xlabel("Epoch ID")
        ax.set_ylabel(f"Δ% vs {comparator} (negative = ML better)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Per-epoch Δ% of ML configs vs {comparator}  |  "
                 f"{results_dir.name}, {fairness.upper()}, budget={budget}ms",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"dynamic_delta_per_epoch__{results_dir.name}__"
                          f"{fairness}__bud{budget}__vs_{comparator}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bench_results_v5_fast_epoch60")
    ap.add_argument("--fairness", choices=["fair", "unfair"], default="fair")
    ap.add_argument("--ml-configs", nargs="+",
                    default=["ml/20f_t07__T0.10", "ml/20f_t07",
                             "ml/20f_regr__T0.10", "ml/20f_regr",
                             "ml/20f_ei_log"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--comparators", nargs="+", default=["random", "roulette"])
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    rd = (REPO_ROOT / args.results_dir).resolve()
    od = (REPO_ROOT / args.out_dir).resolve()

    written = []
    for B in args.budgets:
        for comp in args.comparators:
            p = make_plot(rd, args.fairness, B, args.ml_configs, comp, od)
            written.append(p)
    for p in written:
        print(f"wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
