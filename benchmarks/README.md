# Dynamic Fiedler Benchmark

Sweeps (instance × selection × operator-config × budget × seed) on dynamic Fiedler Manhattan instances. Captures per-run artifacts, validates every solution, and emits a flat `summary.csv` + baseline comparison.

The primary question it answers: **does ML-guided operator selection beat roulette and random on dynamic Manhattan?**

---

## Prerequisites

```bash
# 1. C++ solver built with ML enabled
cd cpp && mkdir -p build && cd build && cmake -DENABLE_ML_SELECTION=ON .. && make -j8

# 2. ONNX models exported + manifest written
cd ml_train && python export_onnx.py

# 3. Python package for validation (one-time)
pip install darpinstances
```

---

## Quick smoke test (≈2 min)

```bash
python benchmarks/run_dynamic_bench.py \
    --max-duration 30s \
    --delays 03min \
    --selections roulette,ml,random \
    --configs full \
    --budgets 2000 \
    --seeds 42 \
    --skip-validation
```

Produces 3 runs (one per selection method) under `bench_results/runs/`. Each run dir contains `stdout.log`, `routes.txt`, `solution.json`, `summary.json`, `epoch_metrics.csv`, `validation.txt`.

---

## Default matrix (≈5 min)

```bash
python benchmarks/run_dynamic_bench.py
```

With defaults:
- `--max-duration 02min` → 9 instances (30s, 01min, 02min × 3 delays)
- `--selections random,roulette,ml` → 3
- `--configs full` → 1
- `--budgets 5000` → 1
- `--seeds 42,123,456` → 3

= **81 runs**. Validation mandatory. Runs in ~5 min with 4 workers.

---

## Common variations

```bash
# Budget ablation (81 × 3 = 243 runs)
python benchmarks/run_dynamic_bench.py --budgets 1000,5000,15000

# Config ablation (81 × 3 = 243 runs)
python benchmarks/run_dynamic_bench.py --configs classical,full,novel-focused

# Full factorial at 02min tier (729 runs, ~40 min)
python benchmarks/run_dynamic_bench.py \
    --budgets 1000,5000,15000 \
    --configs classical,full,novel-focused

# Include 05min instances (heavy — auto-sequential, multi-hour)
python benchmarks/run_dynamic_bench.py --max-duration 05min

# Only ML vs roulette, one budget
python benchmarks/run_dynamic_bench.py --selections ml,roulette

# Both start times (if you've converted 07-00 instances)
python benchmarks/run_dynamic_bench.py --start both

# Dry-run to preview
python benchmarks/run_dynamic_bench.py --dry-run
```

---

## Post-processing

Separate scripts so analysis is independent of the runner.

```bash
# 1. Walk runs/, emit bench_results/summary.csv
python benchmarks/aggregate_results.py

# 2. Build baselines.csv (from data/fiedler/Results/) + comparison.csv (joined)
python benchmarks/compare_to_baselines.py
```

`comparison.csv` is the table you read in pandas:

```python
import pandas as pd
df = pd.read_csv("bench_results/comparison.csv")

# Per (instance, seed), which selection is cheapest?
df.sort_values(["instance", "seed", "cost"]).groupby(
    ["instance", "seed"]
).head(1)[["instance", "seed", "selection", "cost", "pct_vs_ih"]]

# Mean ± std pct_vs_ih by selection
df.groupby("selection")["pct_vs_ih"].agg(["mean", "std"])
```

---

## Output layout

```
bench_results/
├── runs/
│   └── <run_id>/                    # e.g. manhattan_01_min_delay03_min__ml__full__budget5000__seed42/
│       ├── stdout.log                # raw solver output (for debugging)
│       ├── routes.txt                # C++ route schedules
│       ├── solution.json             # Fiedler solution format
│       ├── summary.json              # structured metrics (primary per-run artifact)
│       ├── epoch_metrics.csv         # one row per ALNS epoch
│       ├── validation.txt            # solution_checker output
│       └── conversion.log            # convert_solution_fiedler.py output
├── skipped_runs.csv                 # combos skipped (e.g. ML on 15min+ instances)
├── summary.csv                       # flat, one row per run (from aggregate_results.py)
├── baselines.csv                     # IH/VGA baselines per (instance, method)
└── comparison.csv                    # summary.csv × baselines.csv join with pct_vs_ih, etc.
```

---

## Run IDs and re-runs

Run ID format: `<instance>__<selection>__<config>__budget<ms>__seed<n>`

**By default, re-running the same command skips completed runs** (detects the existing `summary.json`). Use `--force` to force a re-run — this creates a NEW timestamped directory (`<run_id>__YYYYMMDD-HHMMSS`) so old results are preserved. The aggregator picks up all of them; filter by `timestamp` column in `summary.csv` to distinguish.

---

## Operator configs

| Config | Destroy | Repair |
|--------|---------|--------|
| `classical` | random, worst, shaw | greedy, regret2, regret3 |
| `full` (default) | random, worst, shaw, rtcritical, lowslack | greedy, regret2, regret3, slack |
| `novel-focused` | random, rtcritical, lowslack | greedy, slack |

---

## Instance filters

**`--max-duration TIER`** includes all instances with duration ≤ TIER.

| TIER | Instances included |
|------|-------------------|
| `30s` | 30s × 3 delays = 3 |
| `01min` | 30s + 01min × 3 delays = 6 |
| `02min` (default) | ... + 02min = 9 |
| `05min` | + 05min = 12 (auto-sequential) |
| `15min` | + 15min = 15 (auto-sequential, ML runs skipped — see below) |
| `30min` | + 30min = 18 (auto-sequential, ML runs skipped) |

**`--start`**: `07-00`, `18-00` (default), or `both`.

**`--delays`**: comma-separated subset of `03min,05min,10min`.

**`--instances` override**: comma-separated explicit basenames, bypasses all filters.

---

## ML + large-instance skip rule

The ML models were trained only on instances ≤ 05min (the Python pipeline skips instances with > 10,000 nodes, which covers 15min and 30min).

For any `(15min|30min, ml)` combination, the runner:

1. Logs the combo to `bench_results/skipped_runs.csv` with a clear reason.
2. Does not dispatch the run.

`random` and `roulette` still run on 15min / 30min.

Override with `--force-ml-oversize` if you explicitly want to test extrapolation behavior.

---

## Execution modes

- **Parallel (default)**: `--workers` defaults to `min(4, cpu_count // 2)`.
- **Auto-sequential** for heavy instances (05min+): memory pressure forces `workers=1`. Override with `--force-parallel`.
- **Explicit single worker**: `--workers 1`.

---

## Full flag reference

```
Instance selection:
  --max-duration STR        30s | 01min | 02min | 05min | 15min | 30min
  --start STR               07-00 | 18-00 | both
  --delays STR[,STR...]     03min,05min,10min
  --instances STR[,STR...]  Override: explicit basenames

Run configuration:
  --selections STR[,STR...] random | roulette | ml
  --configs STR[,STR...]    classical | full | novel-focused
  --budgets INT[,INT...]    Epoch budgets in ms
  --seeds INT[,INT...]      Random seeds
  --epoch-length FLOAT      Epoch length in seconds (default: 30)

Execution:
  --workers INT             Parallel workers
  --force-parallel          Allow parallelism on heavy instances
  --out-dir PATH            Results root (default: bench_results/)
  --force                   Re-run even if completed (new timestamped dir)
  --skip-validation         Skip solution_checker
  --force-ml-oversize       Allow ML on 15min/30min instances
  --ml-model-dir PATH       Directory with feature_manifest.json + .onnx files
  --solver PATH             Path to darp_solver binary
  --converted-dir PATH      Directory with converted Fiedler .txt files
  --dry-run                 Print matrix, don't execute
```
