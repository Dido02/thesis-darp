#!/usr/bin/env python3
"""
Analyze benchmark results.
Reads individual JSON files from results/runs/ and generates:
  - results/benchmark.csv (all data)
  - results/summary.csv (best per instance/feasibility/iterations)
  - Console summary tables
"""

import json
import os
import csv
from collections import defaultdict

RESULTS_DIR = "results2/runs"
OUTPUT_DIR = "results2"


def load_results():
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            results.append(json.load(f))
    return results


def write_full_csv(results):
    if not results:
        return
    path = os.path.join(OUTPUT_DIR, "benchmark.csv")
    fields = [
        "instance", "feasibility", "destroy", "repair", "selection",
        "seed", "label", "max_iterations", "final_obj", "best_obj",
        "solve_time", "iterations_completed", "time_limited",
        "iter_per_sec", "unassigned", "feasible", "repair_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Full CSV: {path} ({len(results)} rows)")


def print_tables(results):
    if not results:
        print("No results found.")
        return

    instances = sorted(set(r["instance"] for r in results))
    iter_counts = sorted(set(r.get("max_iterations", 0) for r in results))

    for inst in instances:
        print(f"\n{'='*90}")
        print(f"  Instance: {inst}")
        print(f"{'='*90}")

        for iters in iter_counts:
            for feas in ["o1", "slack"]:
                rows = [r for r in results
                        if r["instance"] == inst
                        and r["feasibility"] == feas
                        and r.get("max_iterations") == iters
                        and r.get("final_obj") is not None]

                if not rows:
                    continue

                rows.sort(key=lambda r: r["final_obj"])

                tl_count = sum(1 for r in rows if r.get("time_limited"))

                print(f"\n  {feas} | {iters} iterations "
                      f"({tl_count}/{len(rows)} hit time limit)")
                print(f"  {'Label':<30} {'Final Obj':>10} {'Best Obj':>10} "
                      f"{'Iters':>7} {'Time':>7} {'Repair':>7} {'TL':>4}")
                print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")

                for r in rows[:15]:
                    tl = "Y" if r.get("time_limited") else ""
                    obj = r.get("final_obj", 0)
                    best = r.get("best_obj", 0)
                    it = r.get("iterations_completed", 0)
                    t = r.get("solve_time", 0)
                    rep = r.get("repair_count", 0)
                    print(f"  {r['label']:<30} {obj:>10.2f} {best:>10.2f} "
                          f"{it:>7d} {t:>7.1f} {rep:>7d} {tl:>4}")

    # Best per instance per feasibility per iteration count
    print(f"\n{'='*90}")
    print(f"  BEST CONFIGURATIONS")
    print(f"{'='*90}")
    for inst in instances:
        for iters in iter_counts:
            for feas in ["o1", "slack"]:
                rows = [r for r in results
                        if r["instance"] == inst
                        and r["feasibility"] == feas
                        and r.get("max_iterations") == iters
                        and r.get("final_obj") is not None]
                if rows:
                    best = min(rows, key=lambda r: r["final_obj"])
                    tl = " [TL]" if best.get("time_limited") else ""
                    print(f"  {inst} {feas} i={iters}: "
                          f"{best['final_obj']:.2f}{tl} — {best['label']} "
                          f"({best['destroy']} / {best['repair']})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading results from {RESULTS_DIR}...")
    results = load_results()
    print(f"Loaded {len(results)} results.\n")

    if not results:
        print("No results found. Run benchmark.py first.")
        return

    write_full_csv(results)
    print_tables(results)


if __name__ == "__main__":
    main()