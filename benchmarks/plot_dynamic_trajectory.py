"""Plot per-epoch within-epoch ALNS trajectories from dynamic-mode runs.

Parses the `[Epoch N] iter K @ <ms>ms: current=C, best=B, temp=T` lines emitted
by `--log-epoch-iters` (added to `solve_warm` so each dynamic-mode epoch dumps
its own trajectory). Compare multiple selectors on the same instance/budget,
averaging across seeds; produces an iter-aligned and a wall-aligned PNG.

ALIGNMENT-SCHEME METHODOLOGY NOTE
---------------------------------
- iter-aligned ("ml at iter K vs random at iter K"): NOT an apples-to-apples
  temperature comparison. Per-iter wall-cost differs across selectors (ML
  inference adds ms per iter; roulette is near-zero), so iter K of selector A
  and iter K of selector B sit at different wall-clock points and therefore
  different SA-cooling temperatures.
- wall-aligned ("ml at 200ms vs random at 200ms"): the budget-fair comparison.
  Matches the SA-cooling alignment under `--exclude-selection-from-budget`.
Both views are emitted; default the analysis story to wall-fraction.

Usage A — bench-dir mode (mirrors plot_static_trajectory.py layout):
    python benchmarks/plot_dynamic_trajectory.py \\
        --results-dir bench_results_v5_fast_epoch60 \\
        --fairness fair \\
        --instance manhattan_30_s_delay03_min \\
        --budget 1000 \\
        --selectors ml/20f_t07__T0.10 random roulette

Usage B — explicit-paths mode (handy for ad-hoc verification runs):
    python benchmarks/plot_dynamic_trajectory.py \\
        --logs random:/tmp/dyn_random_seed1.log \\
               random:/tmp/dyn_random_seed2.log \\
               ml/20f_t07__T0.10:/tmp/dyn_ml_seed1.log \\
        --label-tag verification
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent

LINE_RE = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+iter\s+(\d+)\s+@\s+(\d+)ms:\s+"
    r"current=(\d+(?:\.\d+)?),\s+best=(\d+(?:\.\d+)?),\s+temp=([\d.eE+\-]+)",
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


def parse_log(path: Path) -> dict[int, list[tuple[int, int, float, float]]]:
    """Return {epoch_id: [(iter, ms_elapsed, current, best), ...]}."""
    text = path.read_text()
    by_epoch: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
    for m in LINE_RE.finditer(text):
        eid = int(m.group(1))
        it = int(m.group(2))
        ms = int(m.group(3))
        cur = float(m.group(4))
        best = float(m.group(5))
        by_epoch[eid].append((it, ms, cur, best))
    return by_epoch


def discover_logs_bench_mode(results_dir: Path, fairness: str, instance: str,
                             budget: int, selectors: list[str]) -> dict[str, list[Path]]:
    """Locate stdout.log files under bench_results_*/{fairness}/.../runs/."""
    out: dict[str, list[Path]] = {}
    for sel in selectors:
        if sel.startswith("ml/"):
            model_short = sel.replace("ml/", "")
            bucket = f"{fairness}/models_v3_full_{model_short}"
            run_sel = "ml"
        else:
            bucket = f"baselines_{fairness}"
            run_sel = sel
        pattern = str(results_dir / bucket / "runs" /
                      f"{instance}__{run_sel}__classical__budget{budget}__seed*" /
                      "stdout.log")
        out[sel] = sorted(Path(p) for p in glob.glob(pattern))
    return out


def aggregate(logs_by_sel: dict[str, list[Path]]) -> dict[str, dict[int, list[list[tuple[int, int, float]]]]]:
    """Return {selector: {epoch_id: [per-seed [(iter, ms, best), ...]]}}."""
    out: dict[str, dict[int, list]] = {}
    for sel, paths in logs_by_sel.items():
        per_epoch: dict[int, list] = defaultdict(list)
        for p in paths:
            parsed = parse_log(p)
            for eid, rows in parsed.items():
                per_epoch[eid].append([(it, ms, best) for (it, ms, _cur, best) in rows])
        out[sel] = per_epoch
    return out


def average_across_seeds(seed_curves: list[list[tuple[int, int, float]]],
                          xkey: str) -> tuple[list[float], list[float], list[float]]:
    """Average best across seeds at common xkey ('iter' or 'ms') points.

    Each seed has its own iter/ms grid. We use each seed's iter values (the
    100/200/.../K_max grid) as the x-axis, intersected across seeds. For ms
    we use a coarse linear grid over the common min/max range and lerp each
    seed's curve onto it, then average.
    """
    if xkey == "iter":
        # iters are guaranteed to be 100, 200, ..., so the intersection is just
        # the iters present in all seed curves.
        seed_dicts = [{it: best for (it, _ms, best) in s} for s in seed_curves]
        common_iters = sorted(set.intersection(*[set(d) for d in seed_dicts])) if seed_dicts else []
        if not common_iters:
            return [], [], []
        xs = common_iters
        means, stds = [], []
        for it in xs:
            vals = [d[it] for d in seed_dicts]
            means.append(mean(vals))
            stds.append(stdev(vals) if len(vals) >= 2 else 0.0)
        return [float(x) for x in xs], means, stds
    else:  # xkey == "ms" — interpolate
        if not seed_curves or any(len(s) == 0 for s in seed_curves):
            return [], [], []
        ms_min = max(min(ms for (_it, ms, _b) in s) for s in seed_curves)
        ms_max = min(max(ms for (_it, ms, _b) in s) for s in seed_curves)
        if ms_max <= ms_min:
            return [], [], []
        n_grid = 50
        grid = [ms_min + (ms_max - ms_min) * i / (n_grid - 1) for i in range(n_grid)]
        means, stds = [], []
        for x in grid:
            vals = []
            for s in seed_curves:
                # piecewise linear interp on (ms, best)
                pts = sorted(s, key=lambda r: r[1])
                # find bracket
                if x <= pts[0][1]:
                    vals.append(pts[0][2]); continue
                if x >= pts[-1][1]:
                    vals.append(pts[-1][2]); continue
                for i in range(len(pts) - 1):
                    a_ms = pts[i][1]; b_ms = pts[i+1][1]
                    if a_ms <= x <= b_ms:
                        a_b = pts[i][2]; b_b = pts[i+1][2]
                        if b_ms == a_ms:
                            vals.append(a_b)
                        else:
                            vals.append(a_b + (b_b - a_b) * (x - a_ms) / (b_ms - a_ms))
                        break
            means.append(mean(vals) if vals else 0.0)
            stds.append(stdev(vals) if len(vals) >= 2 else 0.0)
        return grid, means, stds


def make_plot(data: dict[str, dict[int, list]], xkey: str, title_tag: str,
              out_path: Path) -> Path:
    """One subplot per epoch_id; lines per selector."""
    all_epochs = sorted({eid for sel_data in data.values() for eid in sel_data})
    if not all_epochs:
        raise SystemExit("no epochs found in any log")
    n = len(all_epochs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for idx, eid in enumerate(all_epochs):
        ax = axes[idx // cols][idx % cols]
        for sel, sel_data in data.items():
            seed_curves = sel_data.get(eid, [])
            if not seed_curves:
                continue
            xs, means, stds = average_across_seeds(seed_curves, xkey)
            if not xs:
                continue
            color = SELECTOR_COLORS.get(sel, None)
            ax.plot(xs, means, "-", color=color, linewidth=1.6,
                    label=f"{sel} (n={len(seed_curves)})")
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)
        ax.set_title(f"Epoch {eid}")
        ax.set_xlabel("iter within epoch" if xkey == "iter" else "wall-clock ms within epoch")
        ax.set_ylabel("best obj (mean ± std across seeds)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # hide unused subplots
    for k in range(len(all_epochs), rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    fig.suptitle(f"Per-epoch within-epoch trajectories — {title_tag} — {xkey}-aligned",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    # bench-dir mode
    ap.add_argument("--results-dir")
    ap.add_argument("--fairness", choices=["fair", "unfair"], default="fair")
    ap.add_argument("--instance")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--selectors", nargs="+")
    # explicit-paths mode
    ap.add_argument("--logs", nargs="+",
                    help="One or more `<selector>:<path>` pairs (one per seed log).")
    ap.add_argument("--label-tag", default="custom",
                    help="Free-form tag used in output filenames when --logs is given.")
    # output
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    if args.logs:
        logs_by_sel: dict[str, list[Path]] = defaultdict(list)
        for spec in args.logs:
            sel, _, path = spec.partition(":")
            if not sel or not path:
                raise SystemExit(f"--logs entry malformed: {spec}")
            logs_by_sel[sel].append(Path(path))
        title_tag = args.label_tag
        out_stem = f"dynamic_trajectory__{args.label_tag}"
    else:
        if not all([args.results_dir, args.instance, args.budget, args.selectors]):
            raise SystemExit("bench-dir mode requires --results-dir --instance --budget --selectors")
        rd = Path(args.results_dir)
        if not rd.is_absolute():
            rd = REPO_ROOT / rd
        logs_by_sel = discover_logs_bench_mode(rd.resolve(), args.fairness,
                                                args.instance, args.budget, args.selectors)
        title_tag = f"{args.instance}, {args.fairness.upper()}, budget={args.budget}ms"
        out_stem = f"dynamic_trajectory__{rd.name}__{args.fairness}__{args.instance}__bud{args.budget}"

    # Sanity check logs
    found_any = False
    for sel, paths in logs_by_sel.items():
        if not paths:
            print(f"warn: no logs found for selector '{sel}'")
        else:
            found_any = True
    if not found_any:
        return 1

    data = aggregate(logs_by_sel)
    out_dir = REPO_ROOT / args.out_dir
    iter_path = make_plot(data, "iter", title_tag, out_dir / f"{out_stem}__iter.png")
    wall_path = make_plot(data, "ms", title_tag, out_dir / f"{out_stem}__wall.png")
    print(f"wrote: {iter_path}")
    print(f"wrote: {wall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
