#!/usr/bin/env python3
"""
Benchmark runner for DARP ALNS solver.
Tests operator combinations at different iteration counts.
Each run saves an individual JSON file — safe to interrupt anytime.
Run order is shuffled for diverse early results.
Skips already-completed runs on restart.
"""

import subprocess
import json
import re
import os
import time
import random
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

SOLVER_PATH = "cpp/build/darp_solver"
INSTANCE_DIR = "data/cordeau"
RESULTS_DIR = "results2/runs"

INSTANCES = ["pr01", "pr03", "pr10"]
FEASIBILITY_MODES = ["o1", "slack"]
SELECTION = "roulette"
TIME_LIMIT = 20
ITERATION_COUNTS = [200, 1000, 5000]
SEED = 42

ALL_DESTROY = ["random", "worst", "shaw", "rtcritical", "lowslack"]
ALL_REPAIR = ["greedy", "regret2", "regret3", "slack"]


# ============================================================================
# Test configurations
# ============================================================================

def build_configs():
    configs = []

    # Group 1: Every destroy-repair pair (5 x 4 = 20)
    for d in ALL_DESTROY:
        for r in ALL_REPAIR:
            configs.append((d, r, f"pair_{d}+{r}"))

    # Group 2: All operators baseline
    all_d = ",".join(ALL_DESTROY)
    all_r = ",".join(ALL_REPAIR)
    configs.append((all_d, all_r, "all_ops"))

    # Group 3: All destroy, each repair alone
    for r in ALL_REPAIR:
        configs.append((all_d, r, f"allD_{r}"))

    # Group 4: Each destroy alone, all repair
    for d in ALL_DESTROY:
        configs.append((d, all_r, f"{d}_allR"))

    # Group 5: All destroy, standard repair combos
    configs.append((all_d, "greedy,regret2,regret3", "allD_G+R2+R3"))
    configs.append((all_d, "greedy,regret2", "allD_G+R2"))

    # Group 6: Key destroy combos with best repair
    best_r = "greedy,regret2,regret3"
    configs.append(("random,shaw", best_r, "R+S_G+R2+R3"))
    configs.append(("random,worst,shaw", best_r, "R+W+S_G+R2+R3"))
    configs.append(("random,shaw,rtcritical", best_r, "R+S+RT_G+R2+R3"))

    return configs


# ============================================================================
# Parser
# ============================================================================

def parse_output(output):
    result = {}

    m = re.search(r"Final objective:\s+([\d.]+)", output)
    result["final_obj"] = float(m.group(1)) if m else None

    m = re.search(r"Best solution:\s+obj=([\d.]+)\s+\(([\d.]+)s\)", output)
    if m:
        result["best_obj"] = float(m.group(1))
        result["solve_time"] = float(m.group(2))

    m = re.search(r"Time limit reached at iter (\d+)", output)
    if m:
        result["iterations_completed"] = int(m.group(1))
        result["time_limited"] = True
    else:
        iters = re.findall(r"iter (\d+):", output)
        result["iterations_completed"] = int(iters[-1]) if iters else 0
        result["time_limited"] = False

    m = re.search(r"SA calibrated: ~([\d.]+) iter/s, estimated (\d+)", output)
    if m:
        result["iter_per_sec"] = float(m.group(1))
        result["estimated_iters"] = int(m.group(2))

    m = re.search(r"Unassigned:\s+(\d+)", output)
    result["unassigned"] = int(m.group(1)) if m else None

    m = re.search(r"Feasible:\s+(yes|no)", output)
    result["feasible"] = m.group(1) if m else None

    m = re.search(r"Repairing (\d+) request", output)
    result["repair_count"] = int(m.group(1)) if m else 0

    op_stats = {}
    for m in re.finditer(r"(destroy|repair)/(\w+): used=(\d+), improved=(\d+)", output):
        key = f"{m.group(1)}/{m.group(2)}"
        op_stats[key] = {"used": int(m.group(3)), "improved": int(m.group(4))}
    result["op_stats"] = op_stats

    weights = {}
    for m in re.finditer(r"(destroy|repair)/(\w+): weight=([\d.]+)", output):
        key = f"{m.group(1)}/{m.group(2)}"
        weights[key] = float(m.group(3))
    result["weights"] = weights

    return result


# ============================================================================
# Runner
# ============================================================================

def run_id(instance, feasibility, label, iterations):
    return f"{instance}_{feasibility}_{label}_i{iterations}"


def run_solver(instance, feasibility, destroy, repair, selection, seed, iterations):
    instance_path = os.path.join(INSTANCE_DIR, f"{instance}.txt")
    cmd = [
        SOLVER_PATH,
        "--instance", instance_path,
        "--iterations", str(iterations),
        "--time-limit", str(TIME_LIMIT),
        "--destroy", destroy,
        "--repair", repair,
        "--selection", selection,
        "--feasibility", feasibility,
        "--seed", str(seed),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIME_LIMIT + 60,
        )
        return parse_output(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"final_obj": None, "error": "timeout"}
    except Exception as e:
        return {"final_obj": None, "error": str(e)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = build_configs()

    # Build all runs
    all_runs = []
    for instance in INSTANCES:
        for feasibility in FEASIBILITY_MODES:
            for destroy, repair, label in configs:
                for iterations in ITERATION_COUNTS:
                    all_runs.append({
                        "instance": instance,
                        "feasibility": feasibility,
                        "destroy": destroy,
                        "repair": repair,
                        "label": label,
                        "iterations": iterations,
                    })

    # Shuffle for diverse early results
    random.seed(42)
    random.shuffle(all_runs)

    # Skip completed
    completed = set()
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            completed.add(fname.replace(".json", ""))

    remaining = [r for r in all_runs
                 if run_id(r["instance"], r["feasibility"], r["label"], r["iterations"])
                 not in completed]

    total = len(all_runs)
    done = total - len(remaining)

    print(f"=== DARP ALNS Benchmark ===")
    print(f"Total runs: {total}")
    print(f"Already completed: {done}")
    print(f"Remaining: {len(remaining)}")
    print(f"Iteration counts: {ITERATION_COUNTS}")
    print(f"Time limit: {TIME_LIMIT}s (safety)")
    print(f"Seed: {SEED}")
    print(f"Estimated max time: {len(remaining) * TIME_LIMIT / 60:.0f} minutes")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    for i, run in enumerate(remaining):
        rid = run_id(run["instance"], run["feasibility"], run["label"], run["iterations"])
        progress = f"[{done + i + 1}/{total}]"
        print(f"{progress} {run['instance']} {run['feasibility']} "
              f"{run['label']} iter={run['iterations']} ...", end=" ", flush=True)

        t0 = time.time()
        result = run_solver(
            run["instance"], run["feasibility"],
            run["destroy"], run["repair"],
            SELECTION, SEED, run["iterations"],
        )
        elapsed = time.time() - t0

        obj = result.get("final_obj", "ERR")
        iters = result.get("iterations_completed", "?")
        limited = " [TIME-LIMITED]" if result.get("time_limited") else ""
        print(f"obj={obj} iters={iters}{limited} ({elapsed:.1f}s)")

        save_data = {
            "run_id": rid,
            "instance": run["instance"],
            "feasibility": run["feasibility"],
            "destroy": run["destroy"],
            "repair": run["repair"],
            "selection": SELECTION,
            "seed": SEED,
            "label": run["label"],
            "max_iterations": run["iterations"],
            "time_limit": TIME_LIMIT,
            "final_obj": result.get("final_obj"),
            "best_obj": result.get("best_obj"),
            "solve_time": result.get("solve_time"),
            "iterations_completed": result.get("iterations_completed"),
            "time_limited": result.get("time_limited"),
            "iter_per_sec": result.get("iter_per_sec"),
            "unassigned": result.get("unassigned"),
            "feasible": result.get("feasible"),
            "repair_count": result.get("repair_count"),
            "op_stats": result.get("op_stats", {}),
            "weights": result.get("weights", {}),
            "timestamp": datetime.now().isoformat(),
        }

        json_path = os.path.join(RESULTS_DIR, f"{rid}.json")
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)

    print(f"\nDone! {total} runs completed.")


if __name__ == "__main__":
    main()