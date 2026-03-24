#!/usr/bin/env python3
"""Show per-instance results for a specific operator pair.

Usage: python3 pair_detail.py <destroy> <repair> [iter_count]
Example: python3 pair_detail.py lowslack regret2 25000

Defaults to 25000 iterations. Reads summary_iter{N}.csv from current directory.
"""
import csv
import sys

if len(sys.argv) < 3:
    print("Usage: python3 pair_detail.py <destroy> <repair> [iter_count]")
    print("Example: python3 pair_detail.py lowslack regret2 25000")
    sys.exit(1)

destroy = sys.argv[1]
repair = sys.argv[2]
iter_count = sys.argv[3] if len(sys.argv) > 3 else "25000"

config = f"pair_{destroy}+{repair}"
csv_path = f"summary_iter{iter_count}.csv"

bks = {
    'pr01': 190.02, 'pr02': 301.34, 'pr03': 532.00, 'pr04': 570.25, 'pr05': 625.64,
    'pr06': 783.78, 'pr07': 291.71, 'pr08': 487.84, 'pr09': 653.94, 'pr10': 845.47,
    'pr11': 164.46, 'pr12': 295.66, 'pr13': 484.83, 'pr14': 529.33, 'pr15': 573.56,
    'pr16': 725.22, 'pr17': 248.21, 'pr18': 458.73, 'pr19': 592.23, 'pr20': 783.81
}

rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

config_rows = [r for r in rows if r['config'] == config]
if not config_rows:
    print(f"Config '{config}' not found in {csv_path}")
    sys.exit(1)

instances = [f'pr{i:02d}' for i in range(1, 21)]

print(f"### {destroy} + {repair} — {int(iter_count)//1000}K iterations, per instance")
print()
print("| Instance | BKS | Best | Avg | Best Gap | Avg Gap |")
print("|---|---:|---:|---:|---:|---:|")

total_bg = []
total_ag = []
for inst in instances:
    r = [row for row in config_rows if row['instance'] == inst]
    if not r:
        print(f"| {inst} | — | — | — | — | — |")
        continue
    r = r[0]
    b = bks[inst]
    best = float(r['best'])
    avg = float(r['avg'])
    bg = (best - b) / b * 100
    ag = (avg - b) / b * 100
    total_bg.append(bg)
    total_ag.append(ag)
    print(f"| {inst} | {b:.2f} | {best:.2f} | {avg:.2f} | {bg:.2f}% | {ag:.2f}% |")

print(f"| **Avg** | | | | **{sum(total_bg)/len(total_bg):.2f}%** | **{sum(total_ag)/len(total_ag):.2f}%** |")