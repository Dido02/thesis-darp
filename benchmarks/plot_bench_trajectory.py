"""Plot bench trajectories: objective and iter counts vs simulation time.

Compares one baseline selector against one ML config on a single instance/budget,
averaging over seeds. Reads epoch_metrics.csv files written by run_dynamic_bench.

Usage:
    python benchmarks/plot_bench_trajectory.py \\
        --results-dir bench_results_v4_fast \\
        --instance manhattan_05_min_delay03_min \\
        --budget 5000 \\
        --fairness fair \\
        --baseline random \\
        --config ml/20f_ei_log \\
        --out-dir plots
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_epochs_for(results_dir: Path, instance: str, budget: int,
                    fairness: str, sel: str) -> list[list[dict]]:
    """Return list of per-run epoch lists for the (instance, budget, fairness, sel) cell.
    Each item is a sorted list of epoch dicts for one seed."""
    if sel.startswith("ml/"):
        model_short = sel.replace("ml/", "")
        bucket = f"{fairness}/models_v3_full_{model_short}"
        run_sel = "ml"
    else:
        bucket = f"baselines_{fairness}"
        run_sel = sel

    out = []
    pattern = str(results_dir / bucket / "runs" / f"{instance}__{run_sel}__classical__budget{budget}__seed*")
    for run_dir in sorted(glob.glob(pattern)):
        csv_path = Path(run_dir) / "epoch_metrics.csv"
        if not csv_path.exists():
            continue
        rows = []
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        "epoch_id": int(row.get("epoch_id") or 0),
                        "sim_time": float(row.get("sim_time") or 0),
                        "obj_before": float(row.get("obj_before") or 0),
                        "obj_after": float(row.get("obj_after") or 0),
                        "alns_iterations": int(row.get("alns_iterations") or 0),
                        "iters_improving": int(row.get("iters_improving") or 0),
                        "iters_new_best": int(row.get("iters_new_best") or 0),
                        "iters_sa_accepted": int(row.get("iters_sa_accepted") or 0),
                    })
        except Exception:
            continue
        rows.sort(key=lambda r: r["epoch_id"])
        if rows:
            out.append(rows)
    return out


def aggregate_by_epoch(per_seed_rows: list[list[dict]],
                       metric: str) -> dict[int, dict]:
    """Collapse per-seed lists into per-epoch (mean, min, max) of `metric`."""
    by_ep = defaultdict(list)
    sim_time_by_ep = {}
    for seed_rows in per_seed_rows:
        for r in seed_rows:
            by_ep[r["epoch_id"]].append(r[metric])
            sim_time_by_ep[r["epoch_id"]] = r["sim_time"]
    out = {}
    for ep, vals in by_ep.items():
        out[ep] = {
            "mean": mean(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
            "sim_time": sim_time_by_ep[ep],
        }
    return out


def plot_objective(baseline_rows, config_rows, baseline_label, config_label,
                   instance, budget, fairness, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for rows, label, color in [
        (baseline_rows, baseline_label, "C0"),
        (config_rows, config_label, "C1"),
    ]:
        agg = aggregate_by_epoch(rows, "obj_after")
        if not agg:
            continue
        eps = sorted(agg.keys())
        xs = [agg[e]["sim_time"] for e in eps]
        means = [agg[e]["mean"] for e in eps]
        mins = [agg[e]["min"] for e in eps]
        maxs = [agg[e]["max"] for e in eps]
        ax.plot(xs, means, "-o", color=color, label=f"{label} (mean of {agg[eps[0]]['n']} seeds)", lw=2)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15, label=f"{label} (min-max band)")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Objective (cost, after epoch)")
    ax.set_title(f"Objective trajectory — {instance}, budget={budget}ms, {fairness.upper()}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_pct_improving(baseline_rows, config_rows, baseline_label, config_label,
                        instance, budget, fairness, out_path: Path) -> None:
    """One panel: % improving per epoch (improving / total iters × 100)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for rows, label, color in [
        (baseline_rows, baseline_label, "C0"),
        (config_rows, config_label, "C1"),
    ]:
        # Compute pct = imp / total per (seed, epoch), then aggregate.
        # We can't just aggregate (mean imp / mean total) — that biases when seeds differ.
        # Instead: compute pct per seed-epoch, then mean/min/max across seeds.
        pct_by_ep = defaultdict(list)
        sim_time_by_ep = {}
        for seed_rows in rows:
            for r in seed_rows:
                if r["alns_iterations"] > 0:
                    pct = 100.0 * r["iters_improving"] / r["alns_iterations"]
                    pct_by_ep[r["epoch_id"]].append(pct)
                    sim_time_by_ep[r["epoch_id"]] = r["sim_time"]
        if not pct_by_ep:
            continue
        eps = sorted(pct_by_ep.keys())
        xs = [sim_time_by_ep[e] for e in eps]
        means = [mean(pct_by_ep[e]) for e in eps]
        mins = [min(pct_by_ep[e]) for e in eps]
        maxs = [max(pct_by_ep[e]) for e in eps]
        ax.plot(xs, means, "-o", color=color, label=f"{label}", lw=2)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("% improving iterations (per epoch)")
    ax.set_title(f"Improving rate per epoch — {instance}, budget={budget}ms, {fairness.upper()}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_pct_new_best(baseline_rows, config_rows, baseline_label, config_label,
                      instance, budget, fairness, out_path: Path) -> None:
    """% new-best per epoch (iters_new_best / total iters × 100)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for rows, label, color in [
        (baseline_rows, baseline_label, "C0"),
        (config_rows, config_label, "C1"),
    ]:
        pct_by_ep = defaultdict(list)
        sim_time_by_ep = {}
        for seed_rows in rows:
            for r in seed_rows:
                if r["alns_iterations"] > 0:
                    pct = 100.0 * r["iters_new_best"] / r["alns_iterations"]
                    pct_by_ep[r["epoch_id"]].append(pct)
                    sim_time_by_ep[r["epoch_id"]] = r["sim_time"]
        if not pct_by_ep:
            continue
        eps = sorted(pct_by_ep.keys())
        xs = [sim_time_by_ep[e] for e in eps]
        means = [mean(pct_by_ep[e]) for e in eps]
        mins = [min(pct_by_ep[e]) for e in eps]
        maxs = [max(pct_by_ep[e]) for e in eps]
        ax.plot(xs, means, "-o", color=color, label=f"{label}", lw=2)
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("% new-best iterations (per epoch)")
    ax.set_title(f"New-best rate per epoch — {instance}, budget={budget}ms, {fairness.upper()}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_iterations(baseline_rows, config_rows, baseline_label, config_label,
                    instance, budget, fairness, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_total, ax_imp = axes

    for rows, label, color in [
        (baseline_rows, baseline_label, "C0"),
        (config_rows, config_label, "C1"),
    ]:
        for metric, ax in [("alns_iterations", ax_total), ("iters_improving", ax_imp)]:
            agg = aggregate_by_epoch(rows, metric)
            if not agg:
                continue
            eps = sorted(agg.keys())
            xs = [agg[e]["sim_time"] for e in eps]
            means = [agg[e]["mean"] for e in eps]
            mins = [agg[e]["min"] for e in eps]
            maxs = [agg[e]["max"] for e in eps]
            ax.plot(xs, means, "-o", color=color, label=f"{label}", lw=2)
            ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)

    ax_total.set_ylabel("Total iterations per epoch")
    ax_total.set_title(f"ALNS work per epoch — {instance}, budget={budget}ms, {fairness.upper()}")
    ax_total.legend(loc="best")
    ax_total.grid(True, alpha=0.3)

    ax_imp.set_ylabel("Improving iterations per epoch")
    ax_imp.set_xlabel("Simulation time (s)")
    ax_imp.legend(loc="best")
    ax_imp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="bench_results_v4_fast",
                    help="Bench results root (default: bench_results_v4_fast)")
    ap.add_argument("--instance", default="manhattan_05_min_delay03_min",
                    help="Instance basename (e.g. manhattan_05_min_delay03_min)")
    ap.add_argument("--budget", type=int, default=5000,
                    help="Epoch budget in ms (default: 5000)")
    ap.add_argument("--fairness", default="fair", choices=["fair", "unfair"],
                    help="Fair or unfair budget (default: fair)")
    ap.add_argument("--baseline", default="random",
                    help="Baseline selector: random or roulette (default: random)")
    ap.add_argument("--config", default="ml/20f_ei_log",
                    help="ML config to compare. Format: ml/<model_short>, "
                         "e.g. ml/20f_ei_log (default: ml/20f_ei_log)")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory for output PNGs (default: plots)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.baseline} runs ...")
    baseline_rows = load_epochs_for(results_dir, args.instance, args.budget,
                                    args.fairness, args.baseline)
    print(f"  {len(baseline_rows)} seed runs")
    print(f"Loading {args.config} runs ...")
    config_rows = load_epochs_for(results_dir, args.instance, args.budget,
                                  args.fairness, args.config)
    print(f"  {len(config_rows)} seed runs")

    if not baseline_rows or not config_rows:
        print(f"\nERROR: missing data. baseline={len(baseline_rows)} runs, "
              f"config={len(config_rows)} runs. Check --instance / --budget / --fairness / "
              f"--config args.")
        return 1

    suffix = f"{args.instance}__bud{args.budget}__{args.fairness}__{args.baseline}_vs_{args.config.replace('/', '_')}"

    print("\nGenerating plots ...")
    plot_objective(baseline_rows, config_rows, args.baseline, args.config,
                   args.instance, args.budget, args.fairness,
                   out_dir / f"obj_{suffix}.png")
    plot_iterations(baseline_rows, config_rows, args.baseline, args.config,
                    args.instance, args.budget, args.fairness,
                    out_dir / f"iters_{suffix}.png")
    plot_pct_improving(baseline_rows, config_rows, args.baseline, args.config,
                       args.instance, args.budget, args.fairness,
                       out_dir / f"pct_imp_{suffix}.png")
    plot_pct_new_best(baseline_rows, config_rows, args.baseline, args.config,
                      args.instance, args.budget, args.fairness,
                      out_dir / f"pct_best_{suffix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
