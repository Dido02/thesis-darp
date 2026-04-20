# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

ALNS solver for the Dial-a-Ride Problem (DARP). Two implementations:
- **C++** (`cpp/`) — Performance-critical with O(1) feasibility checking. Currently being debugged.
- **Python** (`src/`) — Working baseline using N-Wouda ALNS library.

## Current Focus: C++ O(1) Feasibility Check Debugging

Implementing constant-time feasibility test from Gschwind & Drexl (2016). Status:
- 0.97% false rejection rate (too conservative) — 57 of 5906 feasible insertions missed
- 0.88% false acceptance rate (too permissive) — 52 of 5901 O(1) accepts are infeasible
- 331184 total checks on pr03, 99.97% agreement with ALAP ground truth

IMPORTANT: Always read docs/algorithms.md before modifying feasibility code. It documents the index mapping (paper 1-indexed → code 0-indexed) and travel time convention (paper includes service time, code does not).

## C++ Build & Test

```bash
cd cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

./test_o1_vs_full ../../data/cordeau/pr03.txt
./test_o1_diagnostic ../../data/cordeau/pr03.txt
./darp_solver ../../data/cordeau/pr03.txt
```

## C++ Key Files

- `cpp/src/feasibility/o1_feasibility.cpp` — O(1) check (Algorithms 2 & 3) — MAIN DEBUG TARGET
- `cpp/include/feasibility/o1_feasibility.h` — RouteAuxData struct, declarations
- `cpp/src/feasibility/full_optimizer.cpp` — Full feasibility check
- `cpp/include/feasibility.h` — Ground truth feasibility
- `cpp/include/instance.h` — DARPInstance, Node
- `cpp/include/solution.h` — DARPSolution
- `cpp/src/initial_solution.cpp` — Greedy initial solution
- `cpp/tests/test_o1_vs_full.cpp` — Agreement test (O(1) vs ALAP)
- `cpp/tests/test_o1_diagnostic.cpp` — Per-function diagnostic with tracing

## Reference Materials

- `docs/algorithms.md` — Extracted algorithms, Proposition 1, special cases, index mappings
- `docs/LM-2016-08.pdf` — Gschwind & Drexl 2016 (O(1) check paper)
- `docs/tang.pdf` — Tang et al. 2010 (AEAP scheduling)

## Key Conventions

- Node numbering: 0=origin depot, 1..n=pickups, n+1..2n=deliveries, 2n+1=destination depot
- check_insertion_o1(aux, req, k, l, inst): insert pickup before position k, delivery before position l
- Paper is 1-indexed (insert after v_k), code is 0-indexed (insert before position k)
- Paper's t_{ij} INCLUDES service time at i; code uses tt[][] = pure travel + explicit service_duration
- Known BKS for pr03: 532.00

## Rules

- NEVER change algorithm logic without citing which paper line/equation justifies it
- Always verify changes by running test_o1_vs_full on pr03
- After editing o1_feasibility.cpp, rebuild and test before moving on

## Agents

Subagents in .claude/agents/:
- **paper-auditor** — Compares paper algorithms against C++ code. Read-only.
- **test-runner** — Builds and runs tests/solver, reports structured results.
- **failure-analyzer** — Traces code with specific inputs to diagnose WHY a case fails.

## Python Solver

```bash
python main.py data/cordeau/pr01.txt --iterations 10000 --seed 42
python -m pytest src/tests/
```

Dependencies: `pip install alns numpy matplotlib` (Python 3.12, venv in `.venv/`)

## Thesis Context

"ML-Guided ALNS for the Dynamic Dial-a-Ride Problem": standard ALNS + three novel operators (ride-time critical removal, low-slack route removal, slack-based insertion) + ML-guided operator selection. Evaluation on Manhattan NYC instances, sanity check on Cordeau.

## ML feature pipeline rules

**Feature versioning contract.** Every `REGISTER_FEATURE_V2` call must declare `(name, batch, mode, tolerance, fn)` explicitly. There are no defaults — the backwards-compat `REGISTER_FEATURE` shim was deliberately removed so that adding a feature forces a conscious decision about which batch it joins, which solver mode(s) it applies to, and what numerical tolerance it can support against the C++ reference. The Python `@register_feature(name, *, batch, mode, tolerance)` decorator mirrors this — all three kwargs are required.

Adding a feature requires the same metadata in both languages. Mismatch is caught at startup by `validate_manifest_against_registry` (called from MLSelector ctor and from `test_feature_parity`).

**Parity discipline.** Never relax a feature's declared tolerance to silence a parity failure. Per-feature tolerances live in the registry and ride through to the parity test via the fixture's `feature_tolerances` map; widening one to mask drift hides the real bug. When a 0.0-tolerance feature fails at ~1e-15 or ~1e-16, investigate the upstream cause first:

- **Accumulation order**: numpy's vectorized `sum()` may iterate elements in a different order than C++'s explicit loop. Match the loop structure exactly (outer i / inner j; naive sum-of-squared-deviations).
- **FMA fusion**: arm64 clang fuses `dx*dx + dy*dy` into a single FMA inside `std::sqrt(...)`. Numpy doesn't. Use `math.hypot` on the Python side to match.
- **Stale solver state**: `DARPSolution`'s default ctor pre-fills `routes` with K depot+vstart+depot empties and seeds `unassigned` with `1..n`. Test code that *appends* fixture data to that default state ends up with 2K routes and a stale unassigned set. Always `clear()` before populating.

The 1e-12 budget exists *only* for variance/std-dev features where two-pass naive accumulation can produce a last-bit diff between numpy and C++ even with matched algorithms. Sums, ratios, max/min, counts must hold at exact zero.

**How to add a batch.** Five steps; if any one fails, the others are pointless:

1. **Python**: add features in `ml_train/darpml/features.py` with `@register_feature(name, batch="<batch>", mode="<mode>", tolerance=<num>)`. If they need helper functions (route slack, removal cost, ride times), put the helpers in `ml_train/darpml/reconstruction.py` as bit-exact mirrors of the C++ side.
2. **C++**: add features in `cpp/src/ml/feature_extractor.cpp` with `REGISTER_FEATURE_V2(...)`. If a helper needs to be lifted out of an operator translation unit (worst_destroy's `removal_cost`, slack_insertion's `compute_route_slack`/`compute_route_cost`), declare it in `cpp/include/operators/<name>_helpers.h` and drop the `static` qualifier. **Do not add new .cpp files** — that requires CMakeLists.txt edits which are out of scope; keep the implementation in the existing operator .cpp.
3. **Synthetic fixture**: add states in `ml_train/build_synthetic_fixture.py` that exercise every new code branch (empty inputs, single-element inputs, threshold boundaries, degenerate denominators). Run once and inspect the printed feature values to confirm each branch fired.
4. **Regenerate** all three working fixtures (small trajectory, large trajectory, synthetic) with the new `--batches` list. The build scripts emit a sibling manifest at `validation/[synthetic/]manifest/feature_manifest.json` matching the active feature set.
5. **Verify** all six parity scenarios:
   - trajectory small: `./test_feature_parity ../../ml_train/validation/parity_states_small.jsonl ../../ml_train/validation/manifest/`
   - trajectory large: same path with `_large.jsonl`
   - synthetic A (dynamic): `./test_feature_parity ../../ml_train/validation/synthetic/synthetic_edge_cases.jsonl ../../ml_train/validation/synthetic/manifest/`
   - synthetic B (static-strip, only when adding non-`dynamic` features): same with `synthetic_static/` paths
   - bad-manifest: `./test_feature_parity ../../ml_train/validation/parity_states_large.jsonl ../../ml_train/validation/bad_manifest/` (must exit 1 with three diagnostics)
   - legacy: `./test_feature_parity ../../ml_train/validation/parity_states_large.jsonl ../../ml_train/models/onnx/` (validates backwards-compat)

   All happy-path scenarios must show `SCHED PASS`, `OBJ PASS`, and `PASS: all N features match`. The bad-manifest scenario must continue to fire all three diagnostics.

If something fails: stop, diagnose the root cause (don't widen tolerance), fix, re-run all six. The synthetic fixture is the most valuable detector — Cordeau-based with hand-crafted edge states, it catches FMA-class precision bugs that the file-loaded Fiedler trajectory fixtures miss.

See [CONTEXT.md](CONTEXT.md) "Feature batches" section for the full feature catalogue, attribution, and known numerical/semantic notes; and [THESIS_NOTES.md](THESIS_NOTES.md) for findings worth pulling into the methodology chapter.