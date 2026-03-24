#!/usr/bin/env python3
"""Generate destroy × repair pair results tables for all iteration counts.

Usage: python3 pair_table_l1.py [results_dir]

Defaults to current directory. Looks for summary_iter1000.csv, summary_iter5000.csv, summary_iter25000.csv.
"""
import csv
import sys
import os

results_dir = sys.argv[1] if len(sys.argv) > 1 else "."

bks = {
    'pr01': 190.02, 'pr02': 301.34, 'pr03': 532.00, 'pr04': 570.25, 'pr05': 625.64,
    'pr06': 783.78, 'pr07': 291.71, 'pr08': 487.84, 'pr09': 653.94, 'pr10': 845.47,
    'pr11': 164.46, 'pr12': 295.66, 'pr13': 484.83, 'pr14': 529.33, 'pr15': 573.56,
    'pr16': 725.22, 'pr17': 248.21, 'pr18': 458.73, 'pr19': 592.23, 'pr20': 783.81
}

destroys = ['random', 'worst', 'shaw', 'rtcritical', 'lowslack']
repairs = ['greedy', 'regret2', 'regret3', 'slack']

iter_counts = [1000, 5000, 25000]

for ic in iter_counts:
    csv_path = os.path.join(results_dir, f"summary_iter{ic}.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {ic} iterations — {csv_path} not found")
        continue

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    pair_gaps = {}
    for d in destroys:
        for r in repairs:
            config = f"pair_{d}+{r}"
            config_rows = [row for row in rows if row['config'] == config]
            if not config_rows:
                pair_gaps[(d, r)] = (None, None)
                continue
            best_gaps = []
            avg_gaps = []
            for row in config_rows:
                inst = row['instance']
                b = bks[inst]
                best_gaps.append((float(row['best']) - b) / b * 100)
                avg_gaps.append((float(row['avg']) - b) / b * 100)
            pair_gaps[(d, r)] = (
                sum(best_gaps) / len(best_gaps),
                sum(avg_gaps) / len(avg_gaps)
            )

    header = "| Destroy \\ Repair | " + " | ".join(repairs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(repairs)) + "|"

    print(f"### Average Gap (%) — {ic // 1000}K iterations, 5 runs, 20 Cordeau instances")
    print()
    print(header)
    print(sep)
    for d in destroys:
        cells = []
        for r in repairs:
            bg, ag = pair_gaps[(d, r)]
            cells.append(f"{ag:.2f}" if ag is not None else "—")
        print(f"| **{d}** | " + " | ".join(cells) + " |")

    print()
    print(f"### Best Gap (%) — {ic // 1000}K iterations, 5 runs, 20 Cordeau instances")
    print()
    print(header)
    print(sep)
    for d in destroys:
        cells = []
        for r in repairs:
            bg, ag = pair_gaps[(d, r)]
            cells.append(f"{bg:.2f}" if bg is not None else "—")
        print(f"| **{d}** | " + " | ".join(cells) + " |")

    print()
    print("---")
    print()