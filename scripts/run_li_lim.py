#!/usr/bin/env python3
"""
Run ALNS solver on all Li & Lim PDPTW instances.
Saves per-instance JSON with objective, routes, and vehicle count.
Safe to interrupt and restart.
"""

import subprocess
import json
import re
import os
import time
from pathlib import Path

SOLVER_PATH = "cpp/build/darp_solver"
INSTANCE_DIR = "data/li_lim"
RESULTS_DIR = "results_li_lim"

ITERATIONS = 1000
SEED = 42
FEASIBILITY = "o1"
SELECTION = "roulette"
DESTROY = "random,worst,shaw"
REPAIR = "greedy,regret2,regret3"


def parse_output(output):
    result = {}

    m = re.search(r"Final objective:\s+([\d.]+)", output)
    result["objective"] = float(m.group(1)) if m else None

    m = re.search(r"Unassigned:\s+(\d+)", output)
    result["unassigned"] = int(m.group(1)) if m else None

    m = re.search(r"Feasible:\s+(yes|no)", output)
    result["feasible"] = m.group(1) == "yes" if m else None

    m = re.search(r"Best solution:\s+obj=[\d.]+\s+\(([\d.]+)s\)", output)
    result["solve_time"] = float(m.group(1)) if m else None

    # Parse routes, skip empty ones (depot-only)
    routes = []
    for m in re.finditer(r"R\d+: \[(.+?)\]", output):
        nodes = [int(x) for x in m.group(1).split(", ")]
        if len(nodes) > 2:  # skip empty routes [0, depot_end]
            routes.append(nodes)
    result["routes"] = routes
    result["num_vehicles"] = len(routes)

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    instances = sorted(Path(INSTANCE_DIR).glob("*.txt"))
    if not instances:
        print(f"No instances found in {INSTANCE_DIR}")
        return

    completed = {f.stem for f in Path(RESULTS_DIR).glob("*.json")}
    remaining = [p for p in instances if p.stem not in completed]

    total = len(instances)
    done = total - len(remaining)

    print(f"=== Li & Lim PDPTW Benchmark ===")
    print(f"Instances: {total}, completed: {done}, remaining: {len(remaining)}")
    print(f"Config: {ITERATIONS} iters, destroy=[{DESTROY}], repair=[{REPAIR}]")
    print(f"Feasibility: {FEASIBILITY}, seed: {SEED}")
    print()

    for i, inst_path in enumerate(remaining):
        name = inst_path.stem
        progress = f"[{done + i + 1}/{total}]"
        print(f"{progress} {name} ...", end=" ", flush=True)

        cmd = [
            SOLVER_PATH,
            "--instance", str(inst_path),
            "--iterations", str(ITERATIONS),
            "--destroy", DESTROY,
            "--repair", REPAIR,
            "--selection", SELECTION,
            "--feasibility", FEASIBILITY,
            "--seed", str(SEED),
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            result = parse_output(proc.stdout)
            if proc.returncode != 0:
                result["error"] = proc.stderr.strip()
        except subprocess.TimeoutExpired:
            result = {"objective": None, "error": "timeout"}
        except Exception as e:
            result = {"objective": None, "error": str(e)}
        elapsed = time.time() - t0

        obj = result.get("objective", "ERR")
        nv = result.get("num_vehicles", "?")
        print(f"obj={obj}, vehicles={nv} ({elapsed:.1f}s)")

        save_data = {
            "instance": name,
            "objective": result.get("objective"),
            "num_vehicles": result.get("num_vehicles"),
            "feasible": result.get("feasible"),
            "unassigned": result.get("unassigned"),
            "solve_time": result.get("solve_time"),
            "routes": result.get("routes", []),
        }

        with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
            json.dump(save_data, f, indent=2)

    # Summary
    lines = []
    header = f"{'Instance':<12} {'Obj':>10} {'Vehicles':>8} {'Feasible':>8}"
    sep = "-" * 40
    lines.append(header)
    lines.append(sep)
    for inst_path in instances:
        name = inst_path.stem
        json_path = os.path.join(RESULTS_DIR, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            obj = d.get("objective", 0) or 0
            nv = d.get("num_vehicles", "?")
            feas = "yes" if d.get("feasible") else "no"
            lines.append(f"{name:<12} {obj:>10.2f} {nv:>8} {feas:>8}")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
