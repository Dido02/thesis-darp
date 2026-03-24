#!/usr/bin/env python3
"""Comprehensive operator evaluation — parallelized, with incremental saves and operator stats."""

import subprocess
import re
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

SOLVER = "./cpp/build/darp_solver"
DATA_DIR = "./data/cordeau"
INSTANCES = [f"pr{i:02d}" for i in range(1, 21)]
FEASIBILITY = "o1"
SEEDS = [42, 123, 456, 789, 1024]
ITERATION_COUNTS = [1000, 5000, 25000]
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def load_cordeau_values():
    values = {}
    for inst in INSTANCES:
        res_path = os.path.join(DATA_DIR, f"{inst}.res")
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                first_line = f.readline().strip()
                try:
                    values[inst] = float(first_line)
                except ValueError:
                    print(f"  WARNING: Could not parse {res_path}: '{first_line}'")
                    values[inst] = 0
        else:
            print(f"  WARNING: {res_path} not found")
            values[inst] = 0
    return values


CORDEAU = load_cordeau_values()

# ── Operator sets ──
DESTROYS = ["random", "worst", "shaw", "rtcritical", "lowslack"]
REPAIRS = ["greedy", "regret2", "regret3", "slack"]

# ── Configurations (26 total) ──
CONFIGS = {}

# 1) All pairwise destroy x repair
for d in DESTROYS:
    for r in REPAIRS:
        CONFIGS[f"pair_{d}+{r}"] = {"destroy": d, "repair": r}

# 2) A: standard only
CONFIGS["A_standard"] = {
    "destroy": "random,worst,shaw",
    "repair": "greedy,regret2,regret3",
}

# 3) B variants: custom destroy additions
CONFIGS["B1_std+rtcritical"] = {
    "destroy": "random,worst,shaw,rtcritical",
    "repair": "greedy,regret2,regret3",
}
CONFIGS["B2_std+lowslack"] = {
    "destroy": "random,worst,shaw,lowslack",
    "repair": "greedy,regret2,regret3",
}
CONFIGS["B3_std+rtcrit+lowslack"] = {
    "destroy": "random,worst,shaw,rtcritical,lowslack",
    "repair": "greedy,regret2,regret3",
}

# 4) C: standard destroy + extended repair (with slack)
CONFIGS["C_std+slack_repair"] = {
    "destroy": "random,worst,shaw",
    "repair": "greedy,regret2,regret3,slack",
}

# 5) D: all operators
CONFIGS["D_all"] = {
    "destroy": "random,worst,shaw,rtcritical,lowslack",
    "repair": "greedy,regret2,regret3,slack",
}

RESULTS_DIR = "results_operators"


def parse_operator_stats(output):
    """Parse operator stats and weights from solver output."""
    ops = []

    # Parse: destroy/random: used=1000, improved=44
    for m in re.finditer(r"(destroy|repair)/(\w+): used=(\d+), improved=(\d+)", output):
        ops.append({
            "op_type": m.group(1),
            "op_name": m.group(2),
            "used": int(m.group(3)),
            "improved": int(m.group(4)),
            "weight": None,
        })

    # Parse: destroy/random: weight=2.42329
    for m in re.finditer(r"(destroy|repair)/(\w+): weight=([\d.]+)", output):
        op_type, op_name, weight = m.group(1), m.group(2), float(m.group(3))
        for op in ops:
            if op["op_type"] == op_type and op["op_name"] == op_name:
                op["weight"] = weight
                break

    return ops


def run_solver(instance_path, iterations, destroy, repair, seed):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout

        obj_match = re.search(r"^Final objective:\s+([\d.]+)", output, re.MULTILINE)
        time_match = re.search(r"^Solve time:\s+([\d.]+)s", output, re.MULTILINE)
        feasible_match = re.search(r"^Feasible:\s+(\w+)", output, re.MULTILINE)
        unassigned_match = re.search(r"^Unassigned:\s+(\d+)", output, re.MULTILINE)

        op_stats = parse_operator_stats(output)

        return {
            "obj": float(obj_match.group(1)) if obj_match else None,
            "time": float(time_match.group(1)) if time_match else None,
            "feasible": feasible_match.group(1) if feasible_match else "?",
            "unassigned": int(unassigned_match.group(1)) if unassigned_match else -1,
            "op_stats": op_stats,
        }
    except subprocess.TimeoutExpired:
        return {"obj": None, "time": None, "feasible": "timeout", "unassigned": -1, "op_stats": []}
    except Exception as e:
        return {"obj": None, "time": None, "feasible": str(e), "unassigned": -1, "op_stats": []}


def run_config_parallel(config, iter_count, seeds, instances, data_dir):
    """Submit all (instance × seed) jobs for one config to the pool."""
    jobs = []
    for inst in instances:
        instance_path = os.path.join(data_dir, f"{inst}.txt")
        for seed in seeds:
            jobs.append((inst, seed, instance_path))

    results = {}  # (inst, seed) -> solver result
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for inst, seed, instance_path in jobs:
            f = executor.submit(
                run_solver, instance_path, iter_count,
                config["destroy"], config["repair"], seed
            )
            futures[f] = (inst, seed)

        for future in as_completed(futures):
            inst, seed = futures[future]
            results[(inst, seed)] = future.result()

    return results


def append_raw_csv(path, rows):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "iterations", "config", "destroy", "repair",
            "instance", "seed", "obj", "time", "feasible", "unassigned"
        ])
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def append_operator_csv(path, rows):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "iterations", "config", "instance", "seed",
            "op_type", "op_name", "used", "improved", "weight"
        ])
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def save_wide_table(iter_count, iter_results):
    path = os.path.join(RESULTS_DIR, f"wide_iter{iter_count}.csv")
    configs_list = list(iter_results.keys())

    header = ["Instance", "Cordeau"]
    for c in configs_list:
        header.extend([f"{c}_best", f"{c}_avg", f"{c}_time"])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        avg_gap_best = {c: [] for c in configs_list}
        avg_gap_avg = {c: [] for c in configs_list}

        for inst in INSTANCES:
            bks = CORDEAU.get(inst, 0)
            row = [inst, f"{bks:.2f}"]
            for c in configs_list:
                stats = iter_results.get(c, {}).get(inst)
                if stats:
                    row.extend([
                        f"{stats['best']:.2f}",
                        f"{stats['avg']:.2f}",
                        f"{stats['avg_time']:.1f}",
                    ])
                    if bks:
                        avg_gap_best[c].append((stats["best"] - bks) / bks * 100)
                        avg_gap_avg[c].append((stats["avg"] - bks) / bks * 100)
                else:
                    row.extend(["", "", ""])
            w.writerow(row)

        row = ["Avg gap", ""]
        for c in configs_list:
            if avg_gap_best[c]:
                gb = sum(avg_gap_best[c]) / len(avg_gap_best[c])
                ga = sum(avg_gap_avg[c]) / len(avg_gap_avg[c])
                row.extend([f"{gb:.2f}%", f"{ga:.2f}%", ""])
            else:
                row.extend(["", "", ""])
        w.writerow(row)

    print(f"  Wide table saved: {path}")


def save_summary_table(iter_count, iter_results):
    path = os.path.join(RESULTS_DIR, f"summary_iter{iter_count}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "destroy", "repair", "instance", "cordeau",
                     "best", "avg", "worst", "avg_time",
                     "gap_best_%", "gap_avg_%"])

        for config_name, instances in iter_results.items():
            config = CONFIGS[config_name]
            for inst in INSTANCES:
                stats = instances.get(inst)
                bks = CORDEAU.get(inst, 0)
                if stats:
                    gap_best = ((stats["best"] - bks) / bks * 100) if bks else 0
                    gap_avg = ((stats["avg"] - bks) / bks * 100) if bks else 0
                    w.writerow([
                        config_name, config["destroy"], config["repair"], inst, bks,
                        f"{stats['best']:.2f}", f"{stats['avg']:.2f}",
                        f"{stats['worst']:.2f}", f"{stats['avg_time']:.1f}",
                        f"{gap_best:.2f}", f"{gap_avg:.2f}",
                    ])
                else:
                    w.writerow([config_name, config["destroy"], config["repair"],
                                inst, bks, "", "", "", "", "", ""])

    print(f"  Summary table saved: {path}")


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw_csv_path = os.path.join(RESULTS_DIR, "all_raw_results.csv")
    op_csv_path = os.path.join(RESULTS_DIR, "operator_stats.csv")
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
    if os.path.exists(op_csv_path):
        os.remove(op_csv_path)

    # ── Verify ──
    print("=" * 50)
    print("  Cordeau reference values (from .res files)")
    print("=" * 50)
    for inst in INSTANCES:
        val = CORDEAU.get(inst, 0)
        print(f"  {inst}: {val:.2f}" if val else f"  {inst}: MISSING")
    print("=" * 50)
    print(f"\n  Configs: {len(CONFIGS)}")
    print(f"  Instances: {len(INSTANCES)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Iterations: {ITERATION_COUNTS}")
    print(f"  Total runs: {len(CONFIGS) * len(INSTANCES) * len(SEEDS) * len(ITERATION_COUNTS)}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  Parallel workers: {MAX_WORKERS}")
    print()
    input("  Press Enter to start, or Ctrl+C to abort...")
    print()

    total_configs = len(CONFIGS)

    for iter_count in ITERATION_COUNTS:
        print(f"\n{'='*90}")
        print(f"  ITERATION COUNT: {iter_count}")
        print(f"{'='*90}")

        iter_results = {}

        for ci, (config_name, config) in enumerate(CONFIGS.items()):
            iter_results[config_name] = {}

            print(f"\n  [{ci+1}/{total_configs}] {config_name}  "
                  f"(destroy={config['destroy']} repair={config['repair']})")
            print(f"  Running 20 instances × 5 seeds = 100 jobs on {MAX_WORKERS} workers...")

            # Run all (instance × seed) in parallel
            all_results = run_config_parallel(config, iter_count, SEEDS, INSTANCES, DATA_DIR)

            # Collect and display results per instance
            config_raw_rows = []
            config_op_rows = []

            for inst in INSTANCES:
                objs = []
                times = []
                for seed in SEEDS:
                    r = all_results[(inst, seed)]

                    config_raw_rows.append({
                        "iterations": iter_count,
                        "config": config_name,
                        "destroy": config["destroy"],
                        "repair": config["repair"],
                        "instance": inst,
                        "seed": seed,
                        "obj": r["obj"],
                        "time": r["time"],
                        "feasible": r["feasible"],
                        "unassigned": r["unassigned"],
                    })

                    for op in r["op_stats"]:
                        config_op_rows.append({
                            "iterations": iter_count,
                            "config": config_name,
                            "instance": inst,
                            "seed": seed,
                            "op_type": op["op_type"],
                            "op_name": op["op_name"],
                            "used": op["used"],
                            "improved": op["improved"],
                            "weight": f"{op['weight']:.4f}" if op["weight"] is not None else "",
                        })

                    if r["obj"] is not None:
                        objs.append(r["obj"])
                        times.append(r["time"])

                if objs:
                    stats = {
                        "best": min(objs),
                        "avg": sum(objs) / len(objs),
                        "worst": max(objs),
                        "avg_time": sum(times) / len(times),
                    }
                    iter_results[config_name][inst] = stats
                    bks = CORDEAU.get(inst, 0)
                    gap = ((stats["best"] - bks) / bks * 100) if bks else 0
                    print(f"    {inst}  best={stats['best']:.2f}  avg={stats['avg']:.2f}  "
                          f"gap={gap:.2f}%  t={stats['avg_time']:.1f}s")
                else:
                    iter_results[config_name][inst] = None
                    print(f"    {inst}  FAILED")

            # ── Save after every config ──
            append_raw_csv(raw_csv_path, config_raw_rows)
            append_operator_csv(op_csv_path, config_op_rows)
            save_wide_table(iter_count, iter_results)
            save_summary_table(iter_count, iter_results)
            print(f"  >> {config_name} saved ({len(config_raw_rows)} runs, "
                  f"{len(config_op_rows)} op stats)")

        print(f"\n  Iteration tier {iter_count} complete.")

    print(f"\n{'='*90}")
    print(f"  ALL DONE — results in {RESULTS_DIR}/")
    print(f"{'='*90}")


if __name__ == "__main__":
    run_all()