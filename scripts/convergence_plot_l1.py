#!/usr/bin/env python3
"""Generate convergence plots for ALNS on selected Cordeau instances.

Usage: python3 convergence_plot.py [--instances pr03,pr07,pr10] [--iterations 25000] [--seeds 42,123,456]

Runs the solver (parallelized), parses iteration-level logs, and produces convergence plots.
Requires matplotlib: pip install matplotlib
"""

import subprocess
import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
from concurrent.futures import ProcessPoolExecutor, as_completed

SOLVER = "./cpp/build/darp_solver"
DATA_DIR = "./data/cordeau"
FEASIBILITY = "o1"
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)

BKS = {
    'pr01': 190.02, 'pr02': 301.34, 'pr03': 532.00, 'pr04': 570.25, 'pr05': 625.64,
    'pr06': 783.78, 'pr07': 291.71, 'pr08': 487.84, 'pr09': 653.94, 'pr10': 845.47,
    'pr11': 164.46, 'pr12': 295.66, 'pr13': 484.83, 'pr14': 529.33, 'pr15': 573.56,
    'pr16': 725.22, 'pr17': 248.21, 'pr18': 458.73, 'pr19': 592.23, 'pr20': 783.81
}

CONFIGS = {
    'A_standard': {
        'destroy': 'random,worst,shaw',
        'repair': 'greedy,regret2,regret3',
        'label': 'A: Standard',
        'color': '#1f77b4',
    },
    'D_all': {
        'destroy': 'random,worst,shaw,rtcritical,lowslack',
        'repair': 'greedy,regret2,regret3,slack',
        'label': 'D: All operators',
        'color': '#d62728',
    },
}


def parse_iteration_log(output):
    """Parse iteration-level best/current/temp from solver output."""
    entries = []
    for m in re.finditer(
        r"iter (\d+): current=([\d.]+), best=([\d.]+), temp=([\d.]+)", output
    ):
        entries.append({
            'iter': int(m.group(1)),
            'current': float(m.group(2)),
            'best': float(m.group(3)),
            'temp': float(m.group(4)),
        })
    return entries


def run_and_parse(instance, iterations, destroy, repair, seed):
    """Run solver and return iteration log."""
    instance_path = os.path.join(DATA_DIR, f"{instance}.txt")
    cmd = [
        SOLVER,
        "--instance", instance_path,
        "--iterations", str(iterations),
        "--destroy", destroy,
        "--repair", repair,
        "--selection", "roulette",
        "--feasibility", FEASIBILITY,
        "--seed", str(seed),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return parse_iteration_log(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {instance} seed={seed}")
        return []


def run_all_jobs(instance, iterations, seeds):
    """Run all config x seed combinations in parallel. Returns {config_name: {seed: entries}}."""
    results = {}
    futures = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for config_name, config in CONFIGS.items():
            for seed in seeds:
                f = executor.submit(
                    run_and_parse, instance, iterations,
                    config['destroy'], config['repair'], seed
                )
                futures[f] = (config_name, seed)

        for future in as_completed(futures):
            config_name, seed = futures[future]
            entries = future.result()
            if config_name not in results:
                results[config_name] = {}
            results[config_name][seed] = entries
            print(f"  Done: {config_name}, seed={seed} ({len(entries)} entries)")

    return results


def plot_convergence(instance, iterations, seeds, output_dir, results):
    """Generate convergence plot for one instance from pre-collected results."""
    bks_val = BKS.get(instance, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{instance} — Convergence ({iterations // 1000}K iterations, '
                 f'seeds: {",".join(str(s) for s in seeds)})', fontsize=14)

    for config_name, config in CONFIGS.items():
        all_best = {}

        for seed in seeds:
            entries = results.get(config_name, {}).get(seed, [])
            for e in entries:
                it = e['iter']
                if it not in all_best:
                    all_best[it] = []
                all_best[it].append(e['best'])

        if not all_best:
            continue

        iters_sorted = sorted(all_best.keys())
        mean_best = [sum(all_best[i]) / len(all_best[i]) for i in iters_sorted]
        min_best = [min(all_best[i]) for i in iters_sorted]
        max_best = [max(all_best[i]) for i in iters_sorted]

        # Left plot: absolute objective
        ax1.plot(iters_sorted, mean_best, label=config['label'],
                 color=config['color'], linewidth=1.5)
        ax1.fill_between(iters_sorted, min_best, max_best,
                         alpha=0.15, color=config['color'])

        # Right plot: gap to BKS (%)
        if bks_val:
            mean_gap = [(v - bks_val) / bks_val * 100 for v in mean_best]
            min_gap = [(v - bks_val) / bks_val * 100 for v in min_best]
            max_gap = [(v - bks_val) / bks_val * 100 for v in max_best]
            ax2.plot(iters_sorted, mean_gap, label=config['label'],
                     color=config['color'], linewidth=1.5)
            ax2.fill_between(iters_sorted, min_gap, max_gap,
                             alpha=0.15, color=config['color'])

    # Format left plot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Objective')
    ax1.set_title('Best Objective over Iterations')
    if bks_val:
        ax1.axhline(y=bks_val, color='green', linestyle='--', alpha=0.5, label=f'BKS={bks_val}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format right plot
    if bks_val:
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gap to BKS (%)')
        ax2.set_title('Gap to Best Known Solution')
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    seeds_str = "_".join(str(s) for s in seeds)
    out_path = os.path.join(output_dir, f"convergence_{instance}_seeds_{seeds_str}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ALNS convergence plots')
    parser.add_argument('--instances', default='pr03,pr07,pr10',
                        help='Comma-separated instance names (default: pr03,pr07,pr10)')
    parser.add_argument('--iterations', type=int, default=25000,
                        help='Number of iterations (default: 25000)')
    parser.add_argument('--seeds', default='42,123,456',
                        help='Comma-separated seeds (default: 42,123,456)')
    parser.add_argument('--output', default='results_convergence',
                        help='Output directory (default: results_convergence)')
    args = parser.parse_args()

    instances = args.instances.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.output, exist_ok=True)

    print(f"Instances: {instances}")
    print(f"Iterations: {args.iterations}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Output: {args.output}")
    print()

    for inst in instances:
        print(f"\n=== {inst} ===")
        print(f"  Running {len(CONFIGS) * len(seeds)} jobs in parallel...")
        results = run_all_jobs(inst, args.iterations, seeds)
        plot_convergence(inst, args.iterations, seeds, args.output, results)

    print(f"\nDone — plots in {args.output}/")


if __name__ == "__main__":
    main()