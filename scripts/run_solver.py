"""Run ALNS solver on a Cordeau instance."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau
from src.solver import solve


def main():
    parser = argparse.ArgumentParser(description="Run ALNS solver on a DARP instance")
    parser.add_argument(
        "--instance",
        default="data/cordeau/pr01.txt",
        help="Path to instance file (default: data/cordeau/pr01.txt)",
    )
    parser.add_argument(
        "--feasibility",
        choices=["slack", "cordeau", "full"],
        default="slack",
        help="Feasibility checker mode (default: slack)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of ALNS iterations (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    instance_path = Path(args.instance)
    if not instance_path.is_absolute():
        instance_path = PROJECT_ROOT / instance_path

    instance = parse_cordeau(instance_path)
    result = solve(
        instance,
        max_iterations=args.iterations,
        seed=args.seed,
        feasibility_mode=args.feasibility,
    )

    print(f"\n=== INITIAL SOLUTION ===")
    result.initial.print_solution()

    print(f"\n=== BEST SOLUTION (mode={args.feasibility}) ===")
    best = result.best
    best.print_solution()

    feasible, reason = best.is_feasible()
    print(f"\nFeasible: {feasible}" + (f" ({reason})" if not feasible else ""))
    print(f"Unassigned: {len(best.unassigned)}")
    print(f"Construction: {result.construction_time:.3f}s")
    print(f"ALNS solve:   {result.solve_time:.1f}s")


if __name__ == "__main__":
    main()
