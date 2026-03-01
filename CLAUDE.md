# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive Large Neighborhood Search (ALNS) solver for the Dial-a-Ride Problem (DARP). Academic thesis project implementing pickup-and-delivery vehicle routing with time windows, ride time constraints, and capacity limits. Uses the N-Wouda ALNS Python library.

## Commands

```bash
# Run solver on a Cordeau instance
python main.py data/cordeau/pr01.txt --iterations 10000 --seed 42

# Run solver on a Li & Lim instance
python main.py data/li/pdp_100/lc101.txt --format li_lim

# Initial solution only (skip ALNS)
python main.py data/cordeau/pr01.txt --no-alns

# Analyze instance characteristics
python analyze_instance.py data/cordeau/pr01.txt --plot

# Run with performance profiling
python run_with_timing.py data/cordeau/pr01.txt --iterations 100 --plot
```

Dependencies: `pip install alns numpy matplotlib` (Python 3.12, venv in `.venv/`)

## Architecture

### Data Flow
`Instance file → Parser → DARPInstance → Initial solution (greedy) → ALNS destroy/repair loop → Final DARPSolution`

### Core Modules (src/)

- **instance.py** — Data model: `Node`, `Request`, `Vehicle`, `DARPInstance`. DARPInstance precomputes distance/time matrices on construction.
- **solution.py** — `Route`, `RouteStop`, `DARPSolution`. Solution implements `objective()` for ALNS (total distance + unassigned penalty). Supports insert/remove of requests and deep copy for destroy-repair cycles.
- **feasibility.py** — Two constraint-checking implementations:
  - Simple forward pass (pedagogical, O(n) per route)
  - Cordeau's 8-step algorithm with forward time slack (optimized for insertion feasibility)
  - Checks: capacity, time windows, pickup-before-delivery precedence, max ride time, max route duration
- **alns_solver.py** — ALNS configuration (`ALNSConfig` dataclass), `solve_darp()` using Simulated Annealing acceptance + Roulette Wheel operator selection, `solve_darp_fallback()` for use without ALNS library, `create_initial_solution()` greedy construction.
- **operators/destroy.py** — Random, worst (cost-based), Shaw (relatedness-based), route removal
- **operators/repair.py** — Greedy, regret-k, random insertion

### Instance Parsers (src/parsers/)

- **cordeau.py** — Cordeau format: header `K n T Q L`, nodes ordered as depot/pickups/deliveries. Creates virtual end depot.
- **li_lim.py** — Li & Lim PDPTW format: header `K Q S`, nodes have explicit pickup/delivery index fields. Different indexing scheme mapped to unified DARPInstance.

### Key Design Patterns

- Operators are wrapped as ALNS-compatible callback functions via factory methods
- Every route modification triggers feasibility validation (forward pass + constraint checks)
- ALNS copies the solution before each destroy-repair cycle; accepts/rejects via SA criterion

## Thesis Research Context

This implements the methodology from the thesis "ML-Guided ALNS for the Dynamic Dial-a-Ride Problem".

### Core Approach
- Build ALNS with standard operators (random removal, worst removal, Shaw removal, greedy insertion, regret-2, regret-3)
- Add three novel DARP-specific operators (see below)
- Train offline ML model (Random Forest or Gradient Boosting) to replace roulette-wheel operator selection
- Dynamic policy: attempt greedy insertion first, trigger ML-ALNS only if insertion fails

### Three Novel Operators
- **Ride-time critical removal**: remove requests with highest ratio of actual ride time to max allowed ride time
- **Low-slack route removal**: remove requests from routes with lowest total schedule slack
- **Slack-based insertion**: score positions by cost increase minus alpha * remaining slack after insertion

### Evaluation
- Dataset: Manhattan subset of NYC ridesharing instances (Fiedler & Mrkos 2023) (https://github.com/aicenter/Ridesharing_DARP_instances)
- Sanity check only: Cordeau instances
- Baselines: Insertion Heuristic, VGA, standard ALNS with roulette-wheel

### Baselines and Comparison
We evaluate in three levels. Level 1 (static, operator quality): compare all destroy-repair combinations (standard + new operators) on Cordeau instances to assess whether our new operators improve over standard ones. Level 2 (static, selection method): compare ML-guided selection vs. roulette-wheel vs. random selection vs. fixed-best-combo, measuring convergence speed and final solution quality. Level 3 (dynamic, full comparison): rolling-horizon experiments on NYC Manhattan instances, comparing against the Insertion Heuristic and VGA baselines provided in the dataset, plus our standard ALNS (same time budget). 