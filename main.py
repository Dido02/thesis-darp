"""
thesis-darp: Main entry point

Run ALNS on DARP instances.

Usage:
    python main.py data/cordeau/pr01.txt
    python main.py data/cordeau/pr01.txt --iterations 5000 --seed 123
"""

import argparse
from pathlib import Path
import sys

from src.parsers import parse_cordeau
from src.solution import create_empty_solution
from src.feasibility import check_solution_feasibility, update_solution_schedules


def main():
    parser = argparse.ArgumentParser(
        description="ALNS solver for the Dial-a-Ride Problem"
    )
    parser.add_argument(
        "instance",
        type=str,
        help="Path to instance file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["cordeau", "li_lim"],
        default="cordeau",
        help="Instance format (default: cordeau)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of ALNS iterations (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-alns",
        action="store_true",
        help="Skip ALNS, just create initial solution"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (minimal output)"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Load instance
    if verbose:
        print(f"Loading instance: {args.instance}")

    if args.format == "cordeau":
        instance = parse_cordeau(args.instance)
    else:
        raise NotImplementedError(f"Parser for {args.format} not yet implemented")

    if verbose:
        print(instance.summary())
        print()

    # Create initial solution using greedy insertion
    if verbose:
        print("Creating initial solution with greedy insertion...")

    import random
    from src.operators.repair import greedy_insertion

    rng = random.Random(args.seed)
    solution = create_empty_solution(instance)
    greedy_insertion(solution, rng)
    update_solution_schedules(instance, solution)

    if verbose:
        print(f"Initial solution objective: {solution.objective():.2f}")
        print(f"Assigned requests: {len(solution.get_assigned_requests())}/{instance.num_requests}")
        print(f"Unassigned requests: {len(solution.unassigned)}")
        print()

    # Check feasibility
    feasible, violations = check_solution_feasibility(instance, solution)

    if verbose:
        print(f"Initial solution feasible: {feasible}")
        if not feasible:
            print(f"Violations: {violations[:5]}...")  # Show first 5
        print()

    if args.no_alns:
        if verbose:
            print("Skipping ALNS (--no-alns flag)")
        return

    # Run ALNS
    try:
        from src.alns_solver import solve_darp, ALNSConfig, ALNS_AVAILABLE

        if not ALNS_AVAILABLE:
            print("ALNS library not installed. Using fallback solver...")
            from src.alns_solver import solve_darp_fallback
            best_solution = solve_darp_fallback(
                instance,
                iterations=args.iterations,
                seed=args.seed,
                verbose=verbose
            )
        else:
            config = ALNSConfig(
                iterations=args.iterations,
                seed=args.seed,
                verbose=verbose
            )

            result = solve_darp(instance, config, initial_solution=solution)
            best_solution = result.solution

        # Print final solution
        if verbose:
            print()
            print("=== Final Solution ===")
            print(best_solution.summary())

    except ImportError as e:
        print(f"Error importing ALNS solver: {e}")
        print("\nTo install required packages:")
        print("  pip install alns numpy")
        sys.exit(1)


if __name__ == "__main__":
    main()