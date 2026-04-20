# Running the DARP Solver — Quick Reference

All commands assume you're in `cpp/build/`. Instance paths are relative from there.

---

## Build

```bash
# With ML selection (requires ONNX models exported)
cd cpp && rm -rf build && mkdir build && cd build
cmake -DENABLE_ML_SELECTION=ON -DCMAKE_BUILD_TYPE=Release .. && make -j8

# Without ML (faster build, no ONNX dependency)
cmake -DENABLE_ML_SELECTION=OFF -DCMAKE_BUILD_TYPE=Release .. && make -j8

# Incremental rebuild (after code changes, from build/)
make -j8
```

---

## Static Mode (Cordeau instances)

Single ALNS run, no rolling horizon. Good for quick tests and Cordeau benchmarks.

```bash
# Basic run
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42

# With time limit instead of iteration count
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 1000000 --time-limit 30 --seed 42

# ML-guided selection
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42 \
  --selection ml --ml-model-dir ../../ml_train/models/onnx/

# Compare all 3 selection methods
for sel in random roulette ml; do
  ARG=""
  [ "$sel" = "ml" ] && ARG="--ml-model-dir ../../ml_train/models/onnx/"
  echo "=== $sel ==="
  ./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42 \
    --selection $sel $ARG 2>&1 | grep "Final objective"
done
```

### Cordeau instances available

| Instance | Requests | Vehicles | BKS |
|----------|----------|----------|-----|
| pr01 | 24 | 3 | 190.02 |
| pr02 | 24 | 2 | 301.34 |
| pr03 | 36 | 3 | 532.00 |
| pr04 | 36 | 2 | 570.25 |
| pr05 | 48 | 4 | 625.24 |
| pr06 | 48 | 3 | 798.41 |
| pr07 | 72 | 6 | 291.71 |
| pr08 | 72 | 4 | 487.72 |
| pr09 | 96 | 8 | 658.50 |
| pr10 | 96 | 5 | 854.56 |

```bash
# Run all small Cordeau instances
for inst in pr01 pr02 pr03 pr04 pr05; do
  echo "=== $inst ==="
  ./darp_solver --instance ../../data/cordeau/$inst.txt --iterations 10000 --seed 42 \
    2>&1 | grep "Final objective"
done
```

---

## Dynamic Mode (Fiedler Manhattan instances)

Rolling horizon with epoch-based re-optimization. This is the realistic setting.

### Key parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--dynamic` | (required) | Enable rolling horizon mode |
| `--epoch-length 30` | 30s sim time | How often ALNS re-optimizes |
| `--epoch-budget-ms 5000` | 5000ms | Wall-clock ALNS budget per epoch |
| `--dynamic-stats` | off | Print per-epoch breakdown |
| `--selection` | roulette | `random`, `roulette`, or `ml` |
| `--penalty 10000` | 1000 | Unassigned penalty (use 10000 for Fiedler) |

### Manhattan instances available

| Instance | Vehicles | Requests | Load time | Memory |
|----------|----------|----------|-----------|--------|
| `manhattan_30_s_delay03_min` | 124 | 140 | ~0.1s | tiny |
| `manhattan_30_s_delay05_min` | 124 | 140 | ~0.1s | tiny |
| `manhattan_30_s_delay10_min` | 124 | 140 | ~0.1s | tiny |
| `manhattan_01_min_delay03_min` | 200 | 286 | ~1s | ~0.5GB |
| `manhattan_01_min_delay05_min` | 200 | 286 | ~1s | ~0.5GB |
| `manhattan_01_min_delay10_min` | 200 | 286 | ~1s | ~0.5GB |
| `manhattan_02_min_delay03_min` | 352 | 605 | ~3s | ~2GB |
| `manhattan_02_min_delay05_min` | 352 | 605 | ~3s | ~2GB |
| `manhattan_02_min_delay10_min` | 352 | 605 | ~3s | ~2GB |
| `manhattan_05_min_delay03_min` | 781 | 1658 | ~15s | ~5GB |
| `manhattan_05_min_delay05_min` | 781 | 1658 | ~15s | ~5GB |
| `manhattan_05_min_delay10_min` | 781 | 1658 | ~15s | ~5GB |
| `manhattan_15min_03min` | 1672 | 5113 | ~30s | ~3GB |
| `manhattan_15min_05min` | 1672 | 5113 | ~30s | ~3GB |
| `manhattan_15min_10min` | 1672 | 5113 | ~30s | ~3GB |

Delay variants: `03_min` = tight time windows (some rejections expected), `10_min` = relaxed (near 100% service).

### Quick test (30s instance, ~15s total)

```bash
./darp_solver --instance ../../data/fiedler_converted/manhattan_30_s_delay03_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 2000 --dynamic-stats
```

### Standard run (01-min instance, ~40s total)

```bash
./darp_solver --instance ../../data/fiedler_converted/manhattan_01_min_delay03_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 --dynamic-stats
```

### ML-guided run

```bash
./darp_solver --instance ../../data/fiedler_converted/manhattan_01_min_delay03_min.txt \
  --seed 42 --selection ml --ml-model-dir ../../ml_train/models/onnx/ --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 --dynamic-stats
```

### Compare all 3 selectors on one instance

```bash
INST=../../data/fiedler_converted/manhattan_01_min_delay03_min.txt
for sel in random roulette ml; do
  ARG=""
  [ "$sel" = "ml" ] && ARG="--ml-model-dir ../../ml_train/models/onnx/"
  echo "=== $sel ==="
  ./darp_solver --instance $INST --seed 42 --selection $sel $ARG --penalty 10000 \
    --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
    2>&1 | grep -E "Served|Rejected|Total travel"
done
```

### Heavy instance (05-min, ~3min per run, ~5GB RAM)

```bash
./darp_solver --instance ../../data/fiedler_converted/manhattan_05_min_delay05_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 --dynamic-stats
```

### Very heavy (15-min, ~10min per run)

```bash
./darp_solver --instance ../../data/fiedler_converted/manhattan_15min_10min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 --dynamic-stats
```

Note: ML models were trained on ≤05min instances. Running `--selection ml` on 15min/30min is extrapolation.

---

## Varying Epoch Budget

Budget controls quality vs speed tradeoff. More budget = more ALNS iterations = better solutions.

```bash
INST=../../data/fiedler_converted/manhattan_01_min_delay03_min.txt
for budget in 1000 5000 15000; do
  echo "=== budget ${budget}ms ==="
  ./darp_solver --instance $INST --seed 42 --selection roulette --penalty 10000 \
    --dynamic --epoch-length 30 --epoch-budget-ms $budget \
    2>&1 | grep -E "Served|Total travel|Wall-clock"
done
```

---

## Operator Subsets

Test specific destroy/repair combos.

```bash
# Classical only (Ropke-Pisinger operators)
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42 \
  --destroy random,worst,shaw --repair greedy,regret2,regret3

# Our novel operators only
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42 \
  --destroy rtcritical,lowslack --repair slack

# ML with operator subset (loads only matching models)
./darp_solver --instance ../../data/cordeau/pr01.txt --iterations 5000 --seed 42 \
  --destroy random,worst --repair greedy,regret2 \
  --selection ml --ml-model-dir ../../ml_train/models/onnx/
```

### Available operators

| Destroy | Description |
|---------|-------------|
| `random` | Remove random requests |
| `worst` | Remove highest-cost requests |
| `shaw` | Remove related (nearby) requests |
| `rtcritical` | Remove ride-time-critical requests (novel) |
| `lowslack` | Remove requests from low-slack routes (novel) |

| Repair | Description |
|--------|-------------|
| `greedy` | Insert cheapest request first |
| `regret2` | 2-regret insertion |
| `regret3` | 3-regret insertion |
| `slack` | Slack-based insertion (novel) |

---

## Benchmark Runner

Automated sweep of (instance × selection × config × budget × seed).

```bash
cd /Users/didodimitrov/Documents/thesis-darp

# Dry-run: see what would be executed
python benchmarks/run_dynamic_bench.py --dry-run

# Quick smoke test (3 runs, ~1 min)
python benchmarks/run_dynamic_bench.py \
  --max-duration 30s --delays 03min --budgets 2000 --seeds 42 \
  --skip-validation

# Default matrix (81 runs, ~5 min)
python benchmarks/run_dynamic_bench.py

# Budget ablation (243 runs, ~15 min)
python benchmarks/run_dynamic_bench.py --budgets 1000,5000,15000

# Config ablation (243 runs, ~15 min)
python benchmarks/run_dynamic_bench.py --configs classical,full,novel-focused

# After runs finish:
python benchmarks/aggregate_results.py
python benchmarks/compare_to_baselines.py
# Results in bench_results/summary.csv and bench_results/comparison.csv
```

---

## Logging (for ML training data generation)

```bash
# Log trials only (for ML training)
./darp_solver --instance ../../data/fiedler_converted/manhattan_01_min_delay03_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
  --log-trials --m-trials 50 --m-trials-every 50 --log-dir ../../trail_logs

# Log everything (state + trials, big files)
./darp_solver --instance ../../data/fiedler_converted/manhattan_01_min_delay03_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
  --log-state --log-trials --m-trials 50 --m-trials-every 50 --log-dir ../../trail_logs
```

---

## Solution Validation

Validates the Fiedler solution against the original instance using their checker.

```bash
# 1. Solve with route output
./darp_solver --instance ../../data/fiedler_converted/manhattan_01_min_delay03_min.txt \
  --seed 42 --selection roulette --penalty 10000 \
  --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
  --output-routes /tmp/routes.txt

# 2. Convert to Fiedler JSON
python ../../scripts/convert_solution_fiedler.py \
  --routes /tmp/routes.txt \
  --instance ../../data/fiedler/Instances/Manhattan/instances/start_18-00/duration_01_min/max_delay_03_min \
  --output /tmp/solution.json

# 3. Validate
python -m darpinstances.solution_checker /tmp/solution.json \
  -i ../../data/fiedler/Instances/Manhattan/instances/start_18-00/duration_01_min/max_delay_03_min/config.yaml
```

---

## All CLI Flags

```
./darp_solver --help

Core:
  --instance <path>         Instance file (required)
  --iterations N            Max ALNS iterations (default: 1000)
  --seed S                  Random seed (default: 42)
  --time-limit SECS         Wall-clock limit (0 = use iterations)
  --penalty P               Unassigned penalty (default: 1000, use 10000 for Fiedler)

Operators:
  --destroy OPS             Comma-separated (default: random,worst,shaw,rtcritical,lowslack)
  --repair OPS              Comma-separated (default: greedy,regret2,regret3,slack)
  --selection METHOD        random | roulette | ml (default: roulette)
  --ml-model-dir DIR        ONNX model directory (required for --selection ml)

SA tuning:
  --sa-start TEMP           SA start temperature
  --sa-end TEMP             SA end temperature
  --feasibility MODE        o1 | full | slack (default: o1)
  --init-sort MODE          slack | time | random | none (default: slack)

Dynamic mode:
  --dynamic                 Enable rolling horizon
  --epoch-length SECS       Epoch length in sim seconds (default: 30)
  --epoch-budget-ms MS      ALNS wall-clock budget per epoch (default: 5000)
  --epoch-requests N        Trigger epoch after N new requests (overrides --epoch-length)
  --top-n-routes N          Limit repair to N nearest routes (0 = all)
  --dynamic-stats           Print per-epoch breakdown

Logging:
  --log-state               Log solver state every iteration
  --log-trials              Log M-trials counterfactual data
  --m-trials N              Trials per operator pair (default: 5)
  --m-trials-every N        Log trials every N iterations (default: 50)
  --log-dir DIR             Log output directory (default: logs/)

Output:
  --output-routes FILE      Write route schedules for solution conversion
```
