"""Plot the cumulative ALNS trajectory across an entire dynamic run.

Concatenates iter checkpoints from every epoch onto a single x-axis so the
full simulation's best-obj trajectory is visible. Epoch boundaries are
annotated as vertical dashed lines + text labels along the top axis.

Each selector has its own cumulative-iter coordinate (selectors with cheaper
per-iter cost run more iters in the same wall-clock budget, so their cumulative
iter at epoch boundary N differs from ML's). Boundaries shown for each
selector that's plotted; the most-prominent x-tick labels use the median
cumulative iter across plotted selectors at each boundary.

Usage:
    python benchmarks/plot_dynamic_trajectory_concat.py \\
        --results-dir bench_results_v6_30s --fairness fair --budget 5000 \\
        --seeds 1 2 \\
        --instances manhattan_05_min_delay03_min \\
        --selectors random roulette ml/20f_regr ml/20f_t07 ml/20f_t07__T0.10
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent

LINE_RE = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+iter\s+(\d+)\s+@\s+\d+ms:\s+"
    r"current=\d+(?:\.\d+)?,\s+best=(\d+(?:\.\d+)?),",
    re.M,
)

SELECTOR_COLORS = {
    "random":             "#888888",
    "roulette":           "#000000",
    "ml/20f_regr":        "#1f77b4",
    "ml/20f_regr__T0.10": "#aec7e8",
    "ml/20f_t07":         "#d62728",
    "ml/20f_t07__T0.10":  "#ff9896",
    "ml/20f_ei_log":      "#2ca02c",
}


def selector_dir(results_dir: Path, fairness: str, sel: str) -> Path:
    if sel.startswith("ml/"):
        short = sel.replace("ml/", "")
        return results_dir / fairness / f"models_v3_full_{short}" / "runs"
    return results_dir / f"baselines_{fairness}" / "runs"


def parse_run(results_dir: Path, fairness: str, sel: str, instance: str,
              budget: int, seed: int):
    """Return list of (epoch_id, iter, best) for the run, in stream order."""
    sel_tag = "ml" if sel.startswith("ml/") else sel
    p = (selector_dir(results_dir, fairness, sel) /
         f"{instance}__{sel_tag}__classical__budget{budget}__seed{seed}" /
         "stdout.log")
    if not p.exists():
        raise FileNotFoundError(p)
    rows = []
    for m in LINE_RE.finditer(p.read_text()):
        rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return rows


def cumulative(rows: list[tuple[int, int, float]]):
    """Convert (epoch, iter, best) → (cum_iter, best) plus per-epoch end positions.

    cum_iter at epoch e, iter K = (sum of max-iter in epochs 1..e-1) + K.
    Returns (xs, ys, epoch_end_cums) where epoch_end_cums maps epoch_id ->
    cumulative iter at the last logged checkpoint of that epoch.
    """
    by_epoch = defaultdict(list)
    for (e, it, best) in rows:
        by_epoch[e].append((it, best))
    epoch_ids = sorted(by_epoch)
    offset = 0
    xs, ys = [], []
    epoch_end_cum = {}
    for e in epoch_ids:
        sorted_iters = sorted(by_epoch[e], key=lambda r: r[0])
        for (it, best) in sorted_iters:
            xs.append(offset + it)
            ys.append(best)
        last_iter = sorted_iters[-1][0]
        epoch_end_cum[e] = offset + last_iter
        offset += last_iter
    return xs, ys, epoch_end_cum


def make_plot(results_dir: Path, fairness: str, budget: int, seeds: list[int],
              instances: list[str], selectors: list[str], out_dir: Path):
    """For each seed, render one PNG: 4 subplots (one per instance), lines per selector."""
    written = []
    for seed in seeds:
        n = len(instances)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.5 * rows),
                                 squeeze=False)
        for idx, inst in enumerate(instances):
            ax = axes[idx // cols][idx % cols]
            # Collect parsed runs for boundary aggregation
            sel_data = {}
            for sel in selectors:
                try:
                    parsed = parse_run(results_dir, fairness, sel, inst, budget, seed)
                except FileNotFoundError as e:
                    print(f"missing: {sel} {inst} budget={budget} seed={seed}: {e}")
                    continue
                xs, ys, epoch_end_cum = cumulative(parsed)
                sel_data[sel] = (xs, ys, epoch_end_cum)

            # Plot each selector's trajectory on its own cumulative-iter x
            for sel, (xs, ys, _) in sel_data.items():
                color = SELECTOR_COLORS.get(sel, None)
                ax.plot(xs, ys, "-", color=color, linewidth=1.4, label=sel)

            # Compute median cumulative iter at each epoch boundary across selectors
            all_epochs = sorted({e for (_, _, eec) in sel_data.values() for e in eec})
            med_boundary = {}
            for e in all_epochs:
                vals = [sel_data[s][2][e] for s in sel_data if e in sel_data[s][2]]
                if vals:
                    med_boundary[e] = sorted(vals)[len(vals) // 2]
            # Vertical dashed lines at median boundaries
            for e, x in med_boundary.items():
                ax.axvline(x, color="#cccccc", lw=0.8, ls="--", zorder=0)
            # Epoch labels along top
            ymax = max((max(ys) for (xs, ys, _) in sel_data.values() if ys),
                       default=0)
            ymin = min((min(ys) for (xs, ys, _) in sel_data.values() if ys),
                       default=0)
            if ymax > ymin:
                top_y = ymax + (ymax - ymin) * 0.04
                prev_x = 0
                for e in all_epochs:
                    x = med_boundary.get(e)
                    if x is None:
                        continue
                    # show label centered between prev boundary and this one
                    label_x = (prev_x + x) / 2
                    ax.text(label_x, top_y, f"E{e}", fontsize=7,
                            color="#666666", ha="center", va="bottom")
                    prev_x = x

            short_inst = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")
            ax.set_title(f"{short_inst}  (seed={seed})")
            ax.set_xlabel("cumulative iter (across all epochs)")
            ax.set_ylabel("best obj")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

        # hide unused subplots
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].set_visible(False)

        fig.suptitle(f"Cumulative within-epoch trajectory  |  "
                     f"{results_dir.name}, {fairness.upper()}, "
                     f"budget={budget}ms, seed={seed}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (f"dynamic_trajectory_concat__{results_dir.name}__"
                              f"{fairness}__bud{budget}__seed{seed}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


DEFAULT_INSTANCES = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bench_results_v6_30s")
    ap.add_argument("--fairness", choices=["fair", "unfair"], default="fair")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--selectors", nargs="+",
                    default=["random", "roulette",
                             "ml/20f_regr", "ml/20f_t07", "ml/20f_t07__T0.10"])
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    rd = (REPO_ROOT / args.results_dir).resolve()
    od = (REPO_ROOT / args.out_dir).resolve()
    written = make_plot(rd, args.fairness, args.budget, args.seeds,
                        args.instances, args.selectors, od)
    for p in written:
        print(f"wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
