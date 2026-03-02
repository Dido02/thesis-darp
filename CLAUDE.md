# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive Large Neighborhood Search (ALNS) solver for the Dial-a-Ride Problem (DARP). Academic thesis project implementing pickup-and-delivery vehicle routing with time windows, ride time constraints, and capacity limits. Uses the N-Wouda ALNS Python library.

## Commands

```bash
# Run solver on a Cordeau instance
python main.py data/cordeau/pr01.txt --iterations 10000 --seed 42

# Run solver on a Li & Lim instance
python main.py data/cordeau/pr01.txt --format li_lim

# Initial solution only (skip ALNS)
python main.py data/cordeau/pr01.txt --no-alns

# Analyze instance characteristics
python analyze_instance.py data/cordeau/pr01.txt --plot

# Run tests
python -m pytest src/tests/

# Run a single test file
python -m pytest src/tests/test_solution.py

# Run a single test
python -m pytest src/tests/test_solution.py::test_known_solution_feasibility

# Quick solver demo
python scripts/run_solver.py

# Download Cordeau benchmark instances
python scripts/download_cordeau.py
```

Dependencies: `pip install alns numpy matplotlib` (Python 3.12, venv in `.venv/`)

## Architecture

### Data Flow
`Instance file → Parser → DARPInstance → Initial solution (greedy) → ALNS destroy/repair loop → Final DARPSolution`

### Core Modules (src/)

- **instance.py** — Data model: `Node`, `DARPInstance`. Node numbering: 0=origin depot, 1..n=pickups, n+1..2n=deliveries, 2n+1=destination depot. DARPInstance precomputes distance/time matrix on construction.
- **solution.py** — `RouteSchedule`, `DARPSolution`. Routes are `list[list[int]]` (node IDs per vehicle). Solution implements `objective()` (total distance + 1000 penalty per unassigned request). Scheduling uses Cordeau's 3-step algorithm: forward pass → depot delay optimization → iterative pickup delay. Feasibility checks: time windows, capacity, precedence, ride time, route duration.
- **initial_solution.py** — `build_initial_solution()`: greedy sequential cheapest-feasible insertion of all requests.
- **solver.py** — `solve()` orchestrates ALNS: greedy construction → destroy/repair loop with Simulated Annealing acceptance + Roulette Wheel operator selection. Returns `SolverResult` with best/initial solutions and timing.
- **operators/destroy/random_removal.py** — Removes 10-40% of assigned requests randomly.
- **operators/repair/greedy_insertion.py** — Reinserts unassigned requests via iterative cheapest feasible insertion.

### Instance Parsers (src/parsers/)

- **cordeau.py** — Cordeau format: header `K n T Q L`, nodes ordered as depot/pickups/deliveries. Creates virtual end depot. Computes Euclidean distance matrix.
- **li_lim.py** — Li & Lim PDPTW format: header `K Q S`, explicit pickup/delivery index fields. Different indexing mapped to unified DARPInstance.

### Key Design Patterns

- Operators receive `(DARPSolution, rng)` and return a new solution (ALNS library compatible)
- Every route modification triggers feasibility validation (scheduling + constraint checks)
- ALNS copies the solution before each destroy-repair cycle; accepts/rejects via SA criterion
- Routes are lazily scheduled on-demand (not cached)
- All `__init__.py` files are empty; use direct imports (e.g., `from src.parsers.cordeau import parse_cordeau`)

### Test Data

Cordeau pr01.txt (3 vehicles, 24 requests) is the primary test instance. Known optimal cost ≈ 190.02. Tests validate against published optimal routes.

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