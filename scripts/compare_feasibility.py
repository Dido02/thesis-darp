"""Compare Python feasibility checkers on pr03 initial solution routes."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau
from src.initial_solution import build_initial_solution
from src.feasibility.slack_corrected import check as check_slack
from src.feasibility.cordeau import check as check_cordeau
from src.feasibility.full_optimizer import check as check_full


def main():
    inst = parse_cordeau(PROJECT_ROOT / "data/cordeau/pr03.txt")
    sol = build_initial_solution(inst)

    print(f"Instance: pr03  (n={inst.num_requests}, K={inst.num_vehicles})")
    print(f"Initial solution: obj={sol.objective():.6f}, unassigned={len(sol.unassigned)}")
    print()

    for k, route in enumerate(sol.routes[:3]):
        print(f"Route {k} (size={len(route)}): {route}")
        for name, checker in [("slack_corrected", check_slack),
                              ("cordeau", check_cordeau),
                              ("full_optimizer", check_full)]:
            feasible, reason = checker(route, inst)
            tag = "FEASIBLE" if feasible else f"INFEASIBLE: {reason}"
            print(f"  {name:20s} -> {tag}")
        print()


if __name__ == "__main__":
    main()
