"""
ALNS Solver for DARP

This module wraps the N-Wouda ALNS library to solve DARP instances.

Usage:
    from src.alns_solver import solve_darp

    solution = solve_darp(instance, iterations=10000, seed=42)

The N-Wouda ALNS library: https://github.com/N-Wouda/ALNS
Install with: pip install alns
"""

import random
import time
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass

import numpy as np

try:
    from alns import ALNS
    from alns.accept import SimulatedAnnealing, HillClimbing, RecordToRecordTravel
    from alns.select import RouletteWheel, SegmentedRouletteWheel
    from alns.stop import MaxIterations, MaxRuntime, NoImprovement

    ALNS_AVAILABLE = True
except ImportError:
    ALNS_AVAILABLE = False
    print("Warning: ALNS library not installed. Install with: pip install alns")

from .instance import DARPInstance
from .solution import DARPSolution, create_empty_solution
from .feasibility import check_solution_feasibility, update_solution_schedules
from .operators import (
    create_random_removal,
    create_worst_removal,
    create_shaw_removal,
    create_greedy_repair,
    create_regret_repair,
)


@dataclass
class ALNSConfig:
    """
    Configuration for ALNS solver.

    Attributes:
        iterations: Maximum number of ALNS iterations
        seed: Random seed for reproducibility

        # Operator configuration
        destroy_pct_min: Minimum percentage of requests to remove (0-1)
        destroy_pct_max: Maximum percentage of requests to remove (0-1)

        # Simulated Annealing parameters
        sa_start_temp: Starting temperature
        sa_end_temp: Ending temperature
        sa_step: Temperature reduction per iteration

        # Weight adjustment parameters
        w_best: Weight score for finding new best solution
        w_better: Weight score for finding better solution
        w_accept: Weight score for accepted worse solution
        w_reject: Weight score for rejected solution

        # Segment length for weight updates
        segment_length: Number of iterations per segment
    """
    iterations: int = 10000
    seed: int = 42

    # Destroy configuration
    destroy_pct_min: float = 0.1
    destroy_pct_max: float = 0.4

    # Simulated Annealing
    sa_start_temp: float = 100.0
    sa_end_temp: float = 1.0
    sa_step: float = 0.9995

    # Weight adjustment (Ropke & Pisinger scheme)
    w_best: float = 33.0
    w_better: float = 9.0
    w_accept: float = 13.0
    w_reject: float = 0.0

    # Segment length
    segment_length: int = 100

    # Verbosity
    verbose: bool = True
    log_interval: int = 1000


@dataclass
class ALNSResult:
    """
    Result of ALNS optimization.
    """
    solution: DARPSolution
    objective: float
    iterations: int
    runtime: float
    feasible: bool
    history: List[float]  # Objective value history


def create_initial_solution(instance: DARPInstance, rng: random.Random) -> DARPSolution:
    """
    Create an initial solution using greedy insertion.

    Args:
        instance: The DARP instance
        rng: Random number generator

    Returns:
        Initial feasible (or partially feasible) solution
    """
    from .operators.repair import greedy_insertion

    solution = create_empty_solution(instance)
    greedy_insertion(solution, rng)
    update_solution_schedules(instance, solution)

    return solution


def solve_darp(
        instance: DARPInstance,
        config: Optional[ALNSConfig] = None,
        initial_solution: Optional[DARPSolution] = None
) -> ALNSResult:
    """
    Solve a DARP instance using ALNS.

    Args:
        instance: The DARP instance to solve
        config: ALNS configuration (uses defaults if None)
        initial_solution: Starting solution (creates one if None)

    Returns:
        ALNSResult with best solution found
    """
    if not ALNS_AVAILABLE:
        raise ImportError("ALNS library not installed. Install with: pip install alns")

    if config is None:
        config = ALNSConfig()

    # Set random seed
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    # Create initial solution
    if initial_solution is None:
        if config.verbose:
            print("Creating initial solution...")
        initial_solution = create_initial_solution(instance, rng)

    if config.verbose:
        print(f"Initial objective: {initial_solution.objective():.2f}")
        print(f"Unassigned requests: {len(initial_solution.unassigned)}")

    # Compute number of requests to remove based on percentage
    n_requests = instance.num_requests
    min_remove = max(1, int(n_requests * config.destroy_pct_min))
    max_remove = max(min_remove + 1, int(n_requests * config.destroy_pct_max))

    # Create ALNS instance
    alns = ALNS(np_rng)

    # Register destroy operators with different removal sizes
    for num_remove in range(min_remove, max_remove + 1, max(1, (max_remove - min_remove) // 3)):
        alns.add_destroy_operator(create_random_removal(num_remove))
        alns.add_destroy_operator(create_worst_removal(num_remove))
        alns.add_destroy_operator(create_shaw_removal(num_remove))

    # Register repair operators
    alns.add_repair_operator(create_greedy_repair())
    alns.add_repair_operator(create_regret_repair(k=2))
    alns.add_repair_operator(create_regret_repair(k=3))

    # Configure acceptance criterion (Simulated Annealing)
    accept = SimulatedAnnealing(
        start_temperature=config.sa_start_temp,
        end_temperature=config.sa_end_temp,
        step=config.sa_step
    )

    # Configure operator selection (Roulette Wheel with weight adjustment)
    select = RouletteWheel(
        scores=[config.w_best, config.w_better, config.w_accept, config.w_reject],
        decay=0.8,
        num_destroy=len([op for op in alns.destroy_operators]),
        num_repair=len([op for op in alns.repair_operators])
    )

    # Configure stopping criterion
    stop = MaxIterations(config.iterations)

    # Track history
    history = []
    best_obj = initial_solution.objective()

    # Callback for logging
    def callback(state, rnd, best, current):
        nonlocal best_obj
        obj = current.objective()
        history.append(obj)

        if obj < best_obj:
            best_obj = obj

        if config.verbose and len(history) % config.log_interval == 0:
            print(f"Iteration {len(history)}: current={obj:.2f}, best={best_obj:.2f}")

    # Run ALNS
    start_time = time.time()

    if config.verbose:
        print(f"\nStarting ALNS with {config.iterations} iterations...")
        print(f"Destroy operators: {len(alns.destroy_operators)}")
        print(f"Repair operators: {len(alns.repair_operators)}")
        print()

    result = alns.iterate(
        initial_solution,
        select,
        accept,
        stop,
        callback=callback
    )

    runtime = time.time() - start_time

    # Get best solution
    best_solution = result.best_state

    # Check feasibility
    feasible, violations = check_solution_feasibility(instance, best_solution)

    if config.verbose:
        print()
        print(f"=== ALNS Complete ===")
        print(f"Runtime: {runtime:.2f}s")
        print(f"Final objective: {best_solution.objective():.2f}")
        print(
            f"Improvement: {(initial_solution.objective() - best_solution.objective()) / initial_solution.objective() * 100:.1f}%")
        print(f"Feasible: {feasible}")
        if not feasible:
            print(f"Violations: {violations}")
        print(f"Unassigned: {len(best_solution.unassigned)}")

    return ALNSResult(
        solution=best_solution,
        objective=best_solution.objective(),
        iterations=len(history),
        runtime=runtime,
        feasible=feasible,
        history=history
    )


def solve_darp_simple(
        instance: DARPInstance,
        iterations: int = 10000,
        seed: int = 42,
        verbose: bool = True
) -> DARPSolution:
    """
    Simplified interface to solve DARP.

    Args:
        instance: The DARP instance
        iterations: Number of ALNS iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        Best solution found
    """
    config = ALNSConfig(
        iterations=iterations,
        seed=seed,
        verbose=verbose
    )
    result = solve_darp(instance, config)
    return result.solution


# =============================================================================
# FALLBACK SOLVER (if ALNS library not available)
# =============================================================================

def solve_darp_fallback(
        instance: DARPInstance,
        iterations: int = 1000,
        seed: int = 42,
        verbose: bool = True
) -> DARPSolution:
    """
    Simple fallback solver without ALNS library.

    Uses basic destroy-repair loop with random acceptance.

    Args:
        instance: The DARP instance
        iterations: Number of iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        Best solution found
    """
    from .operators.destroy import random_removal, worst_removal
    from .operators.repair import greedy_insertion

    rng = random.Random(seed)

    # Create initial solution
    solution = create_empty_solution(instance)
    greedy_insertion(solution, rng)
    update_solution_schedules(instance, solution)

    best_solution = solution.copy()
    best_obj = best_solution.objective()

    if verbose:
        print(f"Initial objective: {best_obj:.2f}")

    # Simple destroy-repair loop
    for i in range(iterations):
        # Copy current solution
        current = solution.copy()

        # Destroy: remove some requests
        num_remove = rng.randint(1, max(1, instance.num_requests // 4))
        if rng.random() < 0.5:
            random_removal(current, num_remove, rng)
        else:
            worst_removal(current, num_remove, rng)

        # Repair: insert removed requests
        greedy_insertion(current, rng)
        update_solution_schedules(instance, current)

        current_obj = current.objective()

        # Accept if better, or with probability based on temperature
        temp = 100 * (0.995 ** i)
        delta = current_obj - solution.objective()

        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
            solution = current

            if current_obj < best_obj:
                best_solution = current.copy()
                best_obj = current_obj

                if verbose and (i + 1) % 100 == 0:
                    print(f"Iteration {i + 1}: new best = {best_obj:.2f}")

    if verbose:
        print(f"Final objective: {best_obj:.2f}")

    return best_solution


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("ALNS Solver module")
    print(f"ALNS library available: {ALNS_AVAILABLE}")

    if not ALNS_AVAILABLE:
        print("\nTo install ALNS library:")
        print("  pip install alns")