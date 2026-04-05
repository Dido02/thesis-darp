#!/usr/bin/env python3
"""
End-to-end pipeline: convert Fiedler instance → solve → convert solution → validate.

Usage:
    python scripts/run_fiedler.py \
        --area Manhattan \
        --duration 01_min \
        --delay 10_min \
        --time-limit 30 \
        --seed 42

    python scripts/run_fiedler.py \
        --area Manhattan \
        --duration 05_min \
        --delay 05_min \
        --iterations 5000 \
        --time-limit 60
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run full Fiedler DARP pipeline")
    parser.add_argument("--area", required=True, help="Area name (Manhattan, DC, Chicago, NYC)")
    parser.add_argument("--duration", required=True, help="Duration folder name (e.g., 01_min, 05_min, 30_min, 02_h, 16_h)")
    parser.add_argument("--delay", required=True, help="Max delay folder name (e.g., 03_min, 05_min, 10_min)")
    parser.add_argument("--start", default="18-00", help="Start time folder (default: 18-00)")
    parser.add_argument("--iterations", type=int, default=1000, help="ALNS iterations (default: 1000)")
    parser.add_argument("--time-limit", type=float, default=30, help="Time limit in seconds (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--feasibility", default="o1", help="Feasibility mode: o1, full, slack (default: o1)")
    parser.add_argument("--penalty", type=float, default=10000, help="Unassigned request penalty (default: 1000)")
    parser.add_argument("--init-sort", default="slack", help="Initial solution sort: slack, time, random, none (default: slack)")
    parser.add_argument("--skip-validate", action="store_true", help="Skip solution validation")
    args = parser.parse_args()

    # Paths
    instance_dir = ROOT / f"data/fiedler/Instances/{args.area}/instances/start_{args.start}/duration_{args.duration}/max_delay_{args.delay}"
    dm_path = ROOT / f"data/fiedler/Instances/{args.area}/dm.h5"
    solver = ROOT / "cpp/build/darp_solver"

    tag = f"{args.area.lower()}_{args.duration}_delay{args.delay}"
    out_dir = ROOT / "data/fiedler_converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = out_dir / f"{tag}.txt"
    routes = out_dir / f"{tag}_result_routes.txt"
    solution = out_dir / f"{tag}_result_solution.json"

    # Validate inputs exist
    if not instance_dir.exists():
        print(f"ERROR: Instance directory not found: {instance_dir}")
        sys.exit(1)
    if not dm_path.exists():
        print(f"ERROR: Distance matrix not found: {dm_path}")
        sys.exit(1)
    if not solver.exists():
        print(f"ERROR: Solver not found: {solver}")
        print("  Run: cd cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8")
        sys.exit(1)

    # Fix config.yaml trips.di → requests.csv if needed
    config_path = instance_dir / "config.yaml"
    config_text = config_path.read_text()
    if "trips.di" in config_text:
        config_path.write_text(config_text.replace("filepath: ./trips.di", "filepath: ./requests.csv"))
        print(f"Fixed config.yaml: trips.di → requests.csv")

    # Step 1: Convert instance
    print(f"\n{'='*60}")
    print(f"STEP 1: Convert instance")
    print(f"{'='*60}")
    ret = subprocess.run([
        sys.executable, str(ROOT / "scripts/convert_fiedler.py"),
        "--instance", str(instance_dir),
        "--dm", str(dm_path),
        "--output", str(converted),
    ])
    if ret.returncode != 0:
        print("ERROR: Conversion failed")
        sys.exit(1)

    # Step 2: Solve
    print(f"\n{'='*60}")
    print(f"STEP 2: Solve (iterations={args.iterations}, time_limit={args.time_limit}s, seed={args.seed}, feasibility={args.feasibility}, penalty={args.penalty}, init_sort={args.init_sort})")
    print(f"{'='*60}")
    ret = subprocess.run([
        str(solver),
        "--instance", str(converted),
        "--iterations", str(args.iterations),
        "--feasibility", args.feasibility,
        "--time-limit", str(args.time_limit),
        "--seed", str(args.seed),
        "--penalty", str(args.penalty),
        "--init-sort", args.init_sort,
        "--output-routes", str(routes),
    ])
    if ret.returncode != 0:
        print("ERROR: Solver failed")
        sys.exit(1)

    # Step 3: Convert solution
    print(f"\n{'='*60}")
    print(f"STEP 3: Convert solution to Fiedler JSON")
    print(f"{'='*60}")
    ret = subprocess.run([
        sys.executable, str(ROOT / "scripts/convert_solution_fiedler.py"),
        "--routes", str(routes),
        "--instance", str(instance_dir),
        "--output", str(solution),
    ])
    if ret.returncode != 0:
        print("ERROR: Solution conversion failed")
        sys.exit(1)

    # Step 4: Validate
    if not args.skip_validate:
        print(f"\n{'='*60}")
        print(f"STEP 4: Validate solution")
        print(f"{'='*60}")
        ret = subprocess.run([
            sys.executable, "-m", "darpinstances.solution_checker",
            str(solution.resolve()),
            "-i", str(config_path.resolve()),
        ])
        if ret.returncode != 0:
            print("WARNING: Validation failed or checker error")
        else:
            print("Validation passed!")

    # Step 5: Compare with their baselines
    print(f"\n{'='*60}")
    print(f"STEP 5: Compare with baselines")
    print(f"{'='*60}")

    # Load our solution
    with open(solution) as f:
        our_sol = json.load(f)
    our_cost = our_sol["cost"]
    our_vehicles = len(our_sol["plans"])
    our_dropped = len(our_sol["dropped_requests"])

    # Parse our solve time from solver output
    our_time = args.time_limit  # approximate (actual is close to limit)

    # Load their baselines
    results_base = ROOT / f"data/fiedler/Results/{args.area}/start_{args.start}/duration_{args.duration}/max_delay_{args.delay}"

    rows = []
    rows.append({
        "method": "Our ALNS",
        "cost_s": our_cost,
        "cost_min": round(our_cost / 60),
        "vehicles": our_vehicles,
        "dropped": our_dropped,
        "time_ms": int(our_time * 1000),
        "time_readable": f"{our_time:.0f}s",
    })

    for method in ["ih", "vga"]:
        method_dir = results_base / method
        sol_file = method_dir / "config.yaml-solution.json"
        perf_file = method_dir / "config.yaml-performance.json"

        if not sol_file.exists():
            continue

        with open(sol_file) as f:
            their_sol = json.load(f)
        their_cost = their_sol["cost"]
        their_vehicles = len(their_sol["plans"])
        their_dropped = len(their_sol.get("dropped_requests", []))

        their_time_ms = 0
        if perf_file.exists():
            with open(perf_file) as f:
                perf = json.load(f)
            their_time_ms = perf.get("total_time", 0)

        if their_time_ms < 1000:
            time_str = f"{their_time_ms}ms"
        elif their_time_ms < 60000:
            time_str = f"{their_time_ms/1000:.1f}s"
        elif their_time_ms < 3600000:
            time_str = f"{their_time_ms/60000:.1f}min"
        else:
            time_str = f"{their_time_ms/3600000:.1f}h"

        rows.append({
            "method": f"Their {method.upper()}",
            "cost_s": their_cost,
            "cost_min": round(their_cost / 60),
            "vehicles": their_vehicles,
            "dropped": their_dropped,
            "time_ms": their_time_ms,
            "time_readable": time_str,
        })

    # Sort by cost
    rows.sort(key=lambda r: r["cost_s"])

    # Print table
    ih_cost = next((r["cost_s"] for r in rows if "IH" in r["method"]), None)
    print(f"\n{'Method':<16} {'Cost (s)':>10} {'Cost (min)':>10} {'Vehicles':>9} {'Dropped':>8} {'Time':>10} {'vs IH':>8}")
    print("-" * 75)
    for r in rows:
        vs_ih = ""
        if ih_cost and ih_cost > 0:
            pct = (r["cost_s"] / ih_cost - 1) * 100
            vs_ih = f"{pct:+.1f}%"
        print(f"{r['method']:<16} {r['cost_s']:>10} {r['cost_min']:>10} {r['vehicles']:>9} {r['dropped']:>8} {r['time_readable']:>10} {vs_ih:>8}")

    # Save CSV
    csv_path = out_dir / f"{tag}_result_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cost_s", "cost_min", "vehicles", "dropped", "time_ms", "time_readable"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Instance:    {converted}")
    print(f"  Routes:      {routes}")
    print(f"  Solution:    {solution}")
    print(f"  Comparison:  {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
