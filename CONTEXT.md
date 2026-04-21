# Project Context Summary

## What This Project Is

**Thesis**: "ML-Guided ALNS for the Dynamic Dial-a-Ride Problem"

A C++ ALNS (Adaptive Large Neighborhood Search) solver for DARP (Dial-a-Ride Problem) with three novel contributions:
1. **Three novel operators**: ride-time critical removal, low-slack route removal, slack-based insertion
2. **ML-guided operator selection**: trained Random Forest classifiers predict which destroy-repair pair is "good" at each solver state, replacing the standard roulette-wheel adaptive selection
3. **Dynamic rolling-horizon framework**: epoch-based re-optimization for online ridesharing scenarios on real Manhattan road networks

---

## Repository Structure

```
thesis-darp/
├── cpp/                          # C++ ALNS solver (the main codebase)
│   ├── src/
│   │   ├── solver.cpp            # Static + warm-start ALNS loop
│   │   ├── dynamic_solver.cpp    # Rolling-horizon dynamic DARP
│   │   ├── operators/            # 6 destroy + 6 repair operators
│   │   ├── feasibility/          # O(1) feasibility check (Gschwind & Drexl 2016)
│   │   ├── ml/                   # ML integration layer
│   │   │   ├── feature_extractor.cpp   # 20 features with REGISTER_FEATURE_V2 macro
│   │   │   ├── ml_selector.cpp         # ONNX inference + weighted pair sampling
│   │   │   └── manifest.cpp            # JSON manifest parser
│   │   └── logging/              # JSONL trial/state logger for training data
│   ├── include/                  # Headers mirror src/ structure
│   ├── tests/
│   │   ├── test_feature_parity.cpp     # Python↔C++ feature value agreement
│   │   └── test_o1_vs_full.cpp         # O(1) feasibility agreement test
│   └── CMakeLists.txt            # ENABLE_ML_SELECTION option, FetchContent for onnxruntime + nlohmann/json
│
├── ml_train/                     # Python ML training pipeline
│   ├── darpml/
│   │   ├── features.py           # 20 features with @register_feature decorator
│   │   ├── labels.py             # Improvement-frequency labeling (threshold=0.3)
│   │   ├── training.py           # 20 RF classifiers, per-log-file split, baseline metrics
│   │   ├── loader.py             # JSONL log parser with epoch_id tracking
│   │   ├── instance.py           # Fiedler/Cordeau instance loader
│   │   ├── reconstruction.py     # Objective validation
│   │   └── inference.py          # Python-side prediction
│   ├── export_onnx.py            # sklearn → ONNX conversion + feature_manifest.json
│   ├── export_parity_states.py   # Generate C++/Python feature comparison fixtures
│   ├── diagnostics.py            # Label imbalance / leakage analysis
│   ├── labeling_sweep.py         # Compare 5 labeling rules across thresholds
│   ├── models/                   # Trained .joblib + models/onnx/*.onnx (gitignored)
│   └── cache/                    # features.parquet, labels.parquet (gitignored)
│
├── benchmarks/                   # Benchmark runner + analysis
│   ├── run_dynamic_bench.py      # Sweep (instance × selection × config × budget × seed)
│   ├── aggregate_results.py      # → summary.csv
│   ├── compare_to_baselines.py   # → baselines.csv + comparison.csv (vs IH/VGA)
│   └── parse_solver_output.py    # Stdout → structured metrics + epoch CSV
│
├── scripts/                      # One-off utilities
│   ├── run_fiedler.py            # Full pipeline: convert → solve → validate
│   ├── convert_fiedler.py        # Fiedler CSV+HDF5 → extended Cordeau .txt
│   └── convert_solution_fiedler.py  # C++ routes → Fiedler solution JSON
│
├── data/                         # Instance files (gitignored)
│   ├── cordeau/                  # pr01-pr20 standard benchmarks
│   ├── fiedler_converted/        # Manhattan instances (30s to 30min, 3 delays each)
│   └── fiedler/                  # Raw Fiedler data + IH/VGA baseline results
│
├── trail_logs/                   # JSONL logs from dynamic runs (training data source)
├── bench_results/                # Benchmark output (gitignored)
├── RUNNING.md                    # Copy-paste commands for all run configurations
├── CLAUDE.md                     # Instructions for AI assistants
└── CONTEXT.md                    # This file
```

---

## C++ Solver Architecture

### Selection methods (`--selection` flag)

| Method | How it picks operators | Weight updates |
|--------|----------------------|----------------|
| `random` | Uniform random destroy + repair independently | None |
| `roulette` | Adaptive weights updated every 100 iters based on improvement scores | Yes (standard ALNS) |
| `ml` | 20 ONNX RF models predict P(good) per pair; sample proportional to probability | None |

### Operator registry

`OperatorRegistry` owns all selection logic. Key method: `select_pair(rng, sol*, inst*, alns_state*)` dispatches on `SelectionMethod`. For ML mode, delegates to `MLSelector::select_pair()` which runs ONNX inference.

### Dynamic solver flow (per epoch)

```
release_requests(sim_clock)
  → greedy_insert_pending()          # fast feasible insertion of new requests
  → commit_picked_up_requests()      # freeze in-progress pickups
  → reject_expired_requests()        # drop requests past their deadline
  → [epoch trigger check]
  → run_epoch_alns(epoch_id)         # warm-start ALNS with time budget
      → rebuild unassigned from pending
      → solve_warm() with epoch budget
      → check_request_uniqueness()   # invariant: no duplicate requests
      → rebuild pending from ALNS unassigned
advance_clock()
```

### Node numbering

- 0 = origin depot
- 1..n = pickups (request i has pickup node i)
- n+1..2n = deliveries (request i has delivery node n+i)
- 2n+1 = destination depot
- 2n+2..2n+1+K = per-vehicle start nodes (Fiedler instances only)

**Important**: The Fiedler `.txt` header is `K n_nodes T Q L` where `n_nodes` is total pickup+delivery count, so `num_requests = n_nodes / 2`.

### Build

```bash
cd cpp && mkdir -p build && cd build
cmake -DENABLE_ML_SELECTION=ON -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

Dependencies fetched automatically via CMake FetchContent:
- onnxruntime 1.19.2 (prebuilt, only when ENABLE_ML_SELECTION=ON)
- nlohmann/json 3.11.3 (header-only, always)

---

## ML Pipeline

### Training data

31 JSONL logs in `trail_logs/` from dynamic Manhattan runs (30s through 05min instances, 3 seeds each). Each log contains state snapshots + 50-trial counterfactual data per 20 operator pairs at regular intervals.

### Labeling rule

**Improvement frequency** (not Bongiovanni's SA-acceptance rule — that produced trivially imbalanced labels due to high SA temperature):
- For each pair, compute fraction of 50 trials where `delta < -1e-6`
- Label = 1 if `improvement_fraction >= 0.3`
- Epoch 1 filtered out (degenerate warm-up with ~5 requests)

This gives mean positive rate ~0.40 with 9/20 pairs in the balanced [0.3, 0.7] range.

### Feature batches

20 features across 5 named, toggleable batches. Each feature is registered with three pieces of metadata: `batch` (string tag for subset selection), `mode` (`"static"`, `"dynamic"`, or `"both"` — restricts which solver mode the feature is meaningful in), and `tolerance` (per-feature parity threshold; 0.0 for sums/ratios/counts, 1e-12 for variance/std-dev). The registry is the single source of truth — `export_onnx.py` reads it, `export_parity_states.py` reads it, and the parity fixture's per-feature tolerances flow from it.

| Batch | Count | Mode coverage | Purpose |
|-------|-------|---------------|---------|
| `core` | 5 | 4 both, 1 dynamic | Smoke-test set across Bongiovanni's 5 categories: demand size, demand spread, vehicle state, routing solution, ALNS state |
| `tightness` | 5 | both | Surface how close to ride-time / time-window violations the solution is. Targets `rtcritical` and `lowslack` operators. |
| `worst_support` | 3 | both | Dispersion of per-request and per-route cost contributions. Discriminative signal for `worst_destroy`. |
| `lns_state` | 5 | both | Adaptation of Bongiovanni 2022's top-importance LNS state descriptors plus request-bank count. |
| `rolling_horizon` | 2 | dynamic-only | Dynamic-DARP state the static-mode model can't observe. Basis for the static-vs-dynamic ablation. Originally 3 features; `flexibility_horizon` was removed after diagnostic — see THESIS_NOTES.md. |

Full per-feature listing (with batch, mode, tolerance, and Bongiovanni attribution) lives at the top of [cpp/src/ml/feature_extractor.cpp](cpp/src/ml/feature_extractor.cpp) and in the [ml_train/darpml/features.py](ml_train/darpml/features.py) module docstring — single source of truth, kept in sync.

Both Python (`@register_feature` decorator) and C++ (`REGISTER_FEATURE_V2` macro) implement identical computations. Parity verified at the per-feature tolerance on five trajectory + synthetic fixture scenarios in every change — see "Verification" below.

### Mode axis (static vs dynamic)

Features tagged `"dynamic"` are stripped from static-mode model manifests by the registry's `get_features(batches, mode)` filter. Features tagged `"both"` apply to either model. The CLI rejects `--mode static --batches rolling_horizon` with a clear error.

The mode tag governs **model applicability**, not per-state computation. Features that fork on dynamic-vs-static behavior at runtime (e.g. `neighborhood_size_frac`'s q-sampling formula) detect dynamic state per-call via `bool(state.committed)` / `!sol.committed.empty()`, mirroring the C++ destroy operators' per-state check.

### Canonical training commands

Two model variants ship from the same code:

**Dynamic** (all 5 batches, 20 features):
```bash
cd ml_train
python -m darpml extract-features --logs ../trail_logs/ --data ../data/ \
  --out cache/features.parquet --mode dynamic \
  --batches core,tightness,worst_support,lns_state,rolling_horizon
python -m darpml compute-labels --logs ../trail_logs/ \
  --out cache/labels.parquet --mode dynamic --rule improvement_freq --threshold 0.3
python -m darpml train --features cache/features.parquet --labels cache/labels.parquet \
  --out models/ --mode dynamic \
  --batches core,tightness,worst_support,lns_state,rolling_horizon
python export_onnx.py
```

**Static** (4 batches, no `rolling_horizon`, no `frac_committed`; 17 features):
```bash
python -m darpml extract-features --logs <static_logs>/ --data ../data/ \
  --out cache/features_static.parquet --mode static \
  --batches core,tightness,worst_support,lns_state
python -m darpml compute-labels --logs <static_logs>/ \
  --out cache/labels_static.parquet --mode static
python -m darpml train --features cache/features_static.parquet \
  --labels cache/labels_static.parquet --out models_static/ --mode static \
  --batches core,tightness,worst_support,lns_state
python export_onnx.py --models-dir models_static --out-dir models_static/onnx
```

Static logs need to be generated separately (no current dataset; out of scope for the feature pipeline build-out). Once they exist, the same end-to-end pipeline applies.

### Feature attribution (Bongiovanni 2022 vs novel)

Adapted from Bongiovanni et al. 2022's 46-feature catalogue:

- `radius_avg_tt` ← Bongiovanni #3 ("Radius avg" — adapted from mean Euclidean distance to mean travel-time distance, since our Fiedler `Node` structs don't carry populated x/y coordinates and travel times come from a precomputed road-network matrix)
- `avg_route_load` ← Bongiovanni #26
- `neighborhood_size_frac` ← Bongiovanni #41
- `cost_vs_initial_frac` ← Bongiovanni #42
- `cost_vs_incumbent_frac` ← Bongiovanni #43
- `iteration_norm` ← Bongiovanni #46
- `horizon_elapsed_frac` ← Bongiovanni #2 ("planning horizon elapsed [%]")

The remaining 13 features are novel to this work. The `tightness` batch (5 features) and `worst_support` batch (3 features) are entirely novel and target our novel destroy operators (`rtcritical`, `lowslack`, `slack` repair). `frac_committed` has no Bongiovanni counterpart — their static e-ADARP formulation has no committed-set concept.

### Verification

Each change to a feature, helper, or fixture runs through five (or six in dynamic+static-mode work) parity scenarios:

1. **Trajectory small** — 10 states from a 01_min Manhattan log, all batches.
2. **Trajectory large** — 40 states from a 02_min Manhattan log, all batches.
3. **Synthetic A (dynamic)** — 10 hand-crafted Cordeau pr01 states exercising empty-set, single-sample, threshold-boundary, and rolling_horizon edge branches the trajectory fixtures don't naturally hit.
4. **Synthetic B (static-strip)** — same 10 states, regenerated with `--mode static`, no rolling_horizon. Validates the static-mode code path before real static logs exist.
5. **Bad-manifest** — deliberately-broken manifest with wrong batch, wrong mode, and missing-feature entries. Test must exit 1 with all 3 diagnostics.
6. **Legacy** — existing 5-feature manifest at `models/onnx/feature_manifest.json`. Validates backwards-compat (manifest parser legacy path).

Each fixture record carries `feature_tolerances`, `route_schedules`, and `objective_py` so the parity test simultaneously verifies feature values, the per-route scheduler, and `DARPSolution::objective()` against `compute_objective`. All currently pass at exact-zero error except where a feature explicitly declares 1e-12 tolerance (variance/std-dev features only).

### Known numerical / semantic notes

Four findings from the Part 1–5 build-out worth flagging for future work:

1. **`current_q` 1-iteration lag.** `ALNSState.current_q` is the q from the *previous* iteration's destroy at runtime (q for the current iter isn't sampled until destroy runs). Parity tests feed q from the same iteration's logged value, so training and runtime see slightly different q values for the same-iteration features. Documented in [cpp/include/ml/alns_state.h](cpp/include/ml/alns_state.h). If `neighborhood_size_frac` ranks high in feature importance after the first lns_state-trained model, the lag is tolerable; if low, worth revisiting.

2. **Synthetic fixture is the FMA-bug detector.** Trajectory fixtures use Fiedler instances which load travel-time matrices from precomputed files (bit-identical across Python/C++). Last-bit divergences from accumulation order or compiler FMA fusion only show up on Cordeau (computed `sqrt(dx² + dy²)`) where the synthetic fixture lives. This caught two real bugs (Parts 3 and 5 below) before they could contaminate trained models. Always run synthetic alongside trajectory.

3. **Cordeau parser uses `math.hypot` to match clang FMA.** On arm64, clang fuses `dx*dx + dy*dy` inside `std::sqrt(...)` into a single FMA instruction (one rounding instead of three), giving correctly-rounded distances. Numpy's `sqrt(dx**2 + dy**2)` doesn't FMA — produced 1-ULP divergence on small active sets. Fixed by switching the Python Cordeau parser to per-pair `math.hypot(dx, dy)` ([instance.py:215-225](ml_train/darpml/instance.py#L215-L225)). Fiedler instances unaffected (file-loaded tt).

4. **Trajectory fixtures sample one epoch at a time.** Each Manhattan log's M-trial extraction yields states from a single ALNS epoch. Cross-epoch coverage of edge cases (empty active set, single non-empty route, unusual q values) requires the synthetic fixture. To stress test more naturally, generate logs with smaller `--m-trials-every` or aggregate multiple logs.

### Training results

20 RF classifiers (one per destroy-repair pair), `class_weight='balanced'`, 100 trees, per-log-file train/test split (23 train / 8 test logs, `random_state=42`) to prevent within-trajectory leakage.

**Run metadata** (apply to all numbers below): features parquet `cache_v2_dynamic/features.parquet` extracted from all 31 trail_logs/ files (manhattan 30s/01_min/02_min/05_min × 3 delays × ≤3 seeds), labels via `improvement_freq` rule with `threshold=0.3`, epoch-1 filtered. **Going forward, every metric reported in this file should carry a one-line run-metadata reference (parquet path + log set + date) so we can reproduce the number months later.**

Three candidates trained from the same `cache_v2_dynamic` data via [train_candidates.py](ml_train/train_candidates.py):

| candidate | batches | #features | median F1 lift (non-degenerate) |
|-----------|---------|-----------|---------------------------------|
| `5f`  | `core` | 5 | +0.034 |
| `13f` | `core,tightness,worst_support` | 13 | **+0.053** |
| `20f` | all 5 batches | 20 | +0.029–+0.047 (RF parallel-scheduling variance) |

The 13-feature configuration is the offline-F1 peak. Adding `lns_state` + `rolling_horizon` (the 20f config) is roughly tied with 13f within RandomForest `n_jobs=-1` non-determinism; the online benchmark is the deciding test.

Top feature importances on the 20f model: `cost_contribution_std` (10.3%), `route_slack_std` (9.0%), `ride_time_utilization_avg` (8.8%) — all from the `tightness` and `worst_support` batches that target our novel operators. The four Bongiovanni-adapted `lns_state` features rank in the middle band (#4-#16). The Bongiovanni-adapted `horizon_elapsed_frac` carries genuine signal (importance 2.8%) but ranks low on this dataset.

**Note on the legacy "+0.086" headline**: an earlier run reported +0.086 median lift on 5 features. That number is not reproducible — it was extracted from a parquet that no longer exists, on an unknown log subset. The current reproducible 5-feature reference is +0.034. Don't chase the old number.

### Pair-level dynamics worth noting

- The `worst_support` batch flips `pair_1_3` (worst+slack) from −0.33 to +0.50 lift while pulling `pair_3_3` (rtcritical+slack) from +0.15 to −0.21. The cost-dispersion features support `worst_destroy` precisely while shifting RF split capacity away from `rtcritical`'s decision boundary on the test set's outlier log (`manhattan_05_min_delay03_min_s42`, 100% positive rate for pair_3_3 vs ~0.4% on the other 7 test logs).
- The pair_3_3 collapse is a data-split artifact + RF conservatism interaction, not a feature problem (precision stays at 1.0; recall drops). Stratified train/test splitting and/or `max_features` tuning are open methodology tasks; the online benchmark may not reflect this offline-metric quirk.

### ONNX export

`ml_train/export_onnx.py` converts 20 joblib models to ONNX (opset 21) and emits `feature_manifest.json` — the single source of truth for feature order, operator names, and model version. C++ reads the manifest at startup and validates all features exist in the C++ registry.

### Feature versioning contract

Adding a new feature requires:
1. Add `@register_feature` in Python `features.py`
2. Add `REGISTER_FEATURE_V2` in C++ `feature_extractor.cpp` with explicit `(batch, mode, tolerance)`
3. Retrain + re-export ONNX
4. Regenerate parity fixture

NO changes needed to: MLSelector, CMake, operator_registry, manifest parser, parity test code, or any header with a hardcoded feature count.

---

## Benchmark Results (latest, 81 runs)

### Setup

- 9 Manhattan instances (30s, 01min, 02min × 3 delays)
- 3 selection methods (random, roulette, ml)
- 1 config (full: 5 destroy × 4 repair = 20 pairs)
- 1 budget (5000ms per epoch)
- 3 seeds (42, 123, 456)
- 30s epoch windows, dynamic mode

### Results

| Instance | Random | Roulette | ML | Winner |
|----------|--------|----------|------|--------|
| 30s_delay03 | 68358±183 | 67942±365 | **67916±170** | ML |
| 30s_delay05 | 58546±724 | 58438±439 | **58007±240** | ML |
| 30s_delay10 | **48303±176** | 48975±246 | 49141±324 | random |
| 01min_delay03 | **127669±675** | 128308±798 | 128559±1453 | random |
| 01min_delay05 | 108335±1311 | **107842±196** | 107889±887 | roulette |
| 01min_delay10 | 90200±736 | **89477±515** | 90088±574 | roulette |
| 02min_delay03 | 237185±2540 | **236778±1735** | 238089±1090 | roulette |
| 02min_delay05 | 202325±974 | 201651±1506 | **200493±1429** | ML |
| 02min_delay10 | 173748±2202 | **172886±928** | 172924±381 | roulette |

**Instance wins**: Roulette 4/9, ML 3/9, Random 2/9

**Overall mean vs IH baseline**: all three at −12% (we beat the insertion heuristic by 12% on average)

**ML vs Roulette paired**: ML wins 14/27 runs, roulette 13/27. Mean gap +0.06% — statistically indistinguishable.

**VGA comparison**: we're ~4-14% worse than VGA (their method uses significantly more compute).

### Interpretation

The benchmark above was run on the **5-feature** ML model (the only model that existed at the time). With 5 features, ML-guided selection is statistically tied with roulette.

The 21-feature pipeline was subsequently built (since reduced to 20 after dropping `flexibility_horizon` for structural degeneracy on Manhattan logs). Three new candidate models exist (`models_v2_dyn_5f`, `_13f`, `_20f`) but **the benchmark has not yet been re-run on them** — that is the immediate next step. Offline F1 lift suggests the 13-feature model (core+tightness+worst_support) is the strongest candidate; the online benchmark is the deciding test.

The 81-run table above is preserved as the reproducible 5-feature baseline. Replace it with the 5f-vs-13f-vs-20f comparison once the benchmark completes.

---

## Recent Bug Fixes

### Duplicate-request bug in dynamic solver (fixed)

**Symptom**: Same request picked up twice and delivered twice in one vehicle's route. Caught by external `darpinstances.solution_checker`.

**Root cause**: `greedy_insert_pending()` in `dynamic_solver.cpp` inserted requests into routes but did NOT remove them from `current_solution_.unassigned`. At the next epoch, ALNS saw the request in both routes AND unassigned, and re-inserted it.

**Fix** (3 layers):
1. `greedy_insert_pending` now erases from `unassigned` after insertion (root cause)
2. `run_epoch_alns` defensively rebuilds `unassigned = pending_` before each epoch
3. `check_request_uniqueness()` runs after every epoch and throws on corrupt state

### Epoch-1 waste (fixed)

**Symptom**: First epoch had only ~5 requests (those arriving exactly at sim start time), wasting 5s of ALNS budget.

**Fix**: `MIN_REQUESTS_FOR_EPOCH = 10` — skip ALNS if fewer than 10 requests known. Greedy pre-insertion still handles initial requests. First real ALNS epoch fires when enough requests accumulate.

---

## Key Configuration Knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--epoch-length` | 30s | How often ALNS re-optimizes (sim time) |
| `--epoch-budget-ms` | 5000 | Wall-clock ALNS budget per epoch |
| `--penalty` | 1000 (use 10000 for Fiedler) | Cost of leaving a request unassigned |
| `--selection` | roulette | `random`, `roulette`, or `ml` |
| `--destroy` | all 5 | Comma-separated operator subset |
| `--repair` | all 4 | Comma-separated operator subset |
| `--feasibility` | o1 | `o1` (Gschwind-Drexl), `full`, or `slack` |

---

## Instance Landscape

| Duration | Vehicles | Requests | Approx run time (5s budget) |
|----------|----------|----------|-----------------------------|
| 30s | 124 | 140 | ~35s |
| 01min | 200 | 286 | ~50s |
| 02min | 352 | 605 | ~55s |
| 05min | 781 | 1658 | ~4min |
| 15min | 1672 | 5113 | ~10min |
| 30min | 1993 | 10362 | ~30min+ (14GB RAM) |

Each duration has 3 delay variants: 03min (tight), 05min (moderate), 10min (relaxed). Tighter delay = more rejections possible.

ML models were trained on ≤05min data. Running ML on 15min/30min is extrapolation (benchmark runner auto-skips these by default).

---

## What's Next

1. **Benchmark the three candidate models** — three ONNX bundles already exist (`models_v2_dyn_{5f,13f,20f}/onnx/`), each from `cache_v2_dynamic` (20-feature parquet, 31 trail logs, `random_state=42`). Run the benchmark harness with `--ml-model-dir` pointing at each in turn, plus `random` and `roulette` controls; report per-instance means and paired comparisons. Offline F1 lift puts the 13f model on top (+0.053 median lift); the online benchmark is the deciding test.
2. **Static-vs-dynamic ablation** — generate static training logs (no current dataset), train a static model with `--mode static --batches core,tightness,worst_support,lns_state` (17 features, no rolling_horizon), benchmark on dynamic instances. The synthetic fixture B already validates the static-mode code path end-to-end; the missing piece is real static logs.
3. **Budget/config ablation** — run the full factorial (`--budgets 1000,5000,15000 --configs classical,full,novel-focused`) to understand interaction effects.
4. **Larger instances** — generate training data from 05min instances, retrain, test on 15min.
5. **Write thesis chapter** — infrastructure + methodology + per-feature attribution are all reportable; pull the four findings from [THESIS_NOTES.md](THESIS_NOTES.md) into the methodology/reproducibility chapter.
