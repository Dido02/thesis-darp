"""ALNS solver for DARP using the Wouda & Lan library."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import time

import numpy as np
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from .instance import DARPInstance
from .solution import DARPSolution
from .initial_solution import build_initial_solution
from .operators.destroy.random_removal import random_removal
from .operators.repair.greedy_insertion import greedy_insertion


@dataclass
class SolverResult:
    """Container for solver output and diagnostics."""

    best: DARPSolution
    initial: DARPSolution
    alns_result: object  # alns.Result
    iteration_log: list[dict] = field(default_factory=list)
    construction_time: float = 0.0
    solve_time: float = 0.0


def solve(
    instance: DARPInstance,
    max_iterations: int = 1000,
    start_temperature: float = 100.0,
    end_temperature: float = 0.01,
    decay: float = 0.8,
    seed: int = 42,
    verbose: bool = True,
) -> SolverResult:
    """Run ALNS on a DARP instance.

    Returns a SolverResult with the best solution, initial solution,
    the library's Result object (for built-in plots), and per-iteration logs.
    """
    rng = np.random.default_rng(seed)

    # Build initial solution
    t0 = time.perf_counter()
    init_sol = build_initial_solution(instance)
    construction_time = time.perf_counter() - t0

    if verbose:
        print(f"Initial solution: obj={init_sol.objective():.2f}, "
              f"unassigned={len(init_sol.unassigned)} "
              f"({construction_time:.3f}s)")

    # Set up ALNS
    alns = ALNS(rng)

    # Register operators
    alns.add_destroy_operator(random_removal, name="random_removal")
    alns.add_repair_operator(greedy_insertion, name="greedy_insertion")

    # Operator selection
    select = RouletteWheel(
        scores=[5, 2, 1, 0.5],
        decay=decay,
        num_destroy=1,
        num_repair=1,
    )

    # Acceptance criterion
    sa_step = (end_temperature / start_temperature) ** (1.0 / max_iterations)
    accept = SimulatedAnnealing(
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        step=sa_step,
        method="exponential",
    )

    # Stopping criterion
    stop = MaxIterations(max_iterations)

    # Run ALNS
    t0 = time.perf_counter()
    result = alns.iterate(init_sol, select, accept, stop)
    solve_time = time.perf_counter() - t0

    best = result.best_state
    if verbose:
        print(f"Best solution:    obj={best.objective():.2f} ({solve_time:.3f}s)")

    return SolverResult(
        best=best,
        initial=init_sol,
        alns_result=result,
        construction_time=construction_time,
        solve_time=solve_time,
    )