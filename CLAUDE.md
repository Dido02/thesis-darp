# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

ALNS solver for the Dial-a-Ride Problem (DARP). Two implementations:
- **C++** (`cpp/`) — Performance-critical with O(1) feasibility checking. Currently being debugged.
- **Python** (`src/`) — Working baseline using N-Wouda ALNS library.

## Current Focus: C++ O(1) Feasibility Check Debugging

Implementing constant-time feasibility test from Gschwind & Drexl (2016). Status:
- 62% false rejection rate (too conservative) — main bug
- 0.4% false acceptance rate (too permissive) — secondary bug
- compute_aeap produces infeasible schedules on some routes (tau > latest)
- Preprocessing data (tau_bar, tau_bar_given, gamma, wait) verified correct by brute force

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