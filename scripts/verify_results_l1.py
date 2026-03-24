"""
Verification script for Level 1 results tables.
Run this to confirm all numbers in the thesis match the raw data.

Usage: python3 verify_level1.py

Requires: summary_iter1000.csv, summary_iter5000.csv, summary_iter25000.csv,
          operator_stats.csv in same directory.
"""
import csv
import statistics

bks = {
    'pr01': 190.02, 'pr02': 301.34, 'pr03': 532.00, 'pr04': 570.25, 'pr05': 625.64,
    'pr06': 783.78, 'pr07': 291.71, 'pr08': 487.84, 'pr09': 653.94, 'pr10': 845.47,
    'pr11': 164.46, 'pr12': 295.66, 'pr13': 484.83, 'pr14': 529.33, 'pr15': 573.56,
    'pr16': 725.22, 'pr17': 248.21, 'pr18': 458.73, 'pr19': 592.23, 'pr20': 783.81
}

portfolios = ['A_standard', 'B1_std+rtcritical', 'B2_std+lowslack',
              'B3_std+rtcrit+lowslack', 'C_std+slack_repair', 'D_all']

def load_summary(iter_count):
    rows = []
    with open(f'summary_iter{iter_count}.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def compute_gaps(rows, config):
    p_rows = [r for r in rows if r['config'] == config]
    best_gaps = []
    avg_gaps = []
    for r in p_rows:
        b = bks[r['instance']]
        best_gaps.append((float(r['best']) - b) / b * 100)
        avg_gaps.append((float(r['avg']) - b) / b * 100)
    return sum(best_gaps)/len(best_gaps), sum(avg_gaps)/len(avg_gaps)

print("TABLE 1: Portfolio comparison")
print(f"{'Config':<8} | {'1K best':>8} {'1K avg':>8} | {'5K best':>8} {'5K avg':>8} | {'25K best':>8} {'25K avg':>8}")
for p in portfolios:
    vals = []
    for ic in ['1000', '5000', '25000']:
        rows = load_summary(ic)
        bg, ag = compute_gaps(rows, p)
        vals.extend([bg, ag])
    label = p.split('_')[0] if '_' in p else p[0]
    print(f"{p:<30} | {vals[0]:>7.2f}% {vals[1]:>7.2f}% | {vals[2]:>7.2f}% {vals[3]:>7.2f}% | {vals[4]:>7.2f}% {vals[5]:>7.2f}%")

print("\nTABLE 2: Per-instance A vs D at 25K")
rows = load_summary('25000')
instances = [f'pr{i:02d}' for i in range(1, 21)]
for inst in instances:
    a = [r for r in rows if r['config'] == 'A_standard' and r['instance'] == inst][0]
    d = [r for r in rows if r['config'] == 'D_all' and r['instance'] == inst][0]
    b = bks[inst]
    a_bg = (float(a['best'])-b)/b*100; a_ag = (float(a['avg'])-b)/b*100
    d_bg = (float(d['best'])-b)/b*100; d_ag = (float(d['avg'])-b)/b*100
    print(f"  {inst}: A best={a_bg:.2f}% avg={a_ag:.2f}%  D best={d_bg:.2f}% avg={d_ag:.2f}%")

print("\nTABLE 3: Operator usage in D_all at 25K")
op_rows = []
with open('operator_stats.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        op_rows.append(row)

for op_type in ['destroy', 'repair']:
    subset = [r for r in op_rows if r['config']=='D_all' and r['iterations']=='25000' and r['op_type']==op_type]
    ops = {}
    for r in subset:
        name = r['op_name']
        if name not in ops: ops[name] = {'used':[], 'weight':[]}
        ops[name]['used'].append(int(r['used']))
        ops[name]['weight'].append(float(r['weight']))
    print(f"\n  {op_type}:")
    for name in sorted(ops.keys()):
        avg_u = sum(ops[name]['used'])/len(ops[name]['used'])
        avg_w = sum(ops[name]['weight'])/len(ops[name]['weight'])
        print(f"    {name:<15} usage={avg_u/25000*100:.1f}%  weight={avg_w:.2f}")