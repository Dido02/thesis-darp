"""Tests for initial_solution.py"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau
from src.initial_solution import build_initial_solution

DATA_DIR = PROJECT_ROOT / "data" / "cordeau"


def test_all_requests_assigned():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)

    assert len(sol.unassigned) == 0, f"Unassigned requests: {sol.unassigned}"
    print("  [PASS] all_requests_assigned")


def test_feasibility():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)

    feasible, reason = sol.is_feasible()
    assert feasible, f"Initial solution should be feasible: {reason}"
    print("  [PASS] feasibility")


def test_objective_reasonable():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)

    obj = sol.objective()
    # Known optimal is ~190. Greedy should be worse but not absurd.
    assert obj > 0, "Objective should be positive"
    assert obj < 1000, f"Objective {obj:.2f} seems too high"
    print(f"  [PASS] objective_reasonable (obj={obj:.2f}, known best=190.02)")


def test_route_structure():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)
    n = inst.num_requests
    depot_d = 2 * n + 1

    for k, route in enumerate(sol.routes):
        # Each route starts at depot 0 and ends at depot 2n+1
        assert route[0] == 0, f"Route {k} doesn't start at depot"
        assert route[-1] == depot_d, f"Route {k} doesn't end at depot"

        # Each pickup appears before its delivery
        seen_pickups = set()
        for node_id in route:
            if 1 <= node_id <= n:
                seen_pickups.add(node_id)
            elif n + 1 <= node_id <= 2 * n:
                req = node_id - n
                assert req in seen_pickups, f"Route {k}: delivery {node_id} before pickup {req}"

    print("  [PASS] route_structure")


if __name__ == "__main__":
    print("Testing initial solution:")
    test_all_requests_assigned()
    test_feasibility()
    test_objective_reasonable()
    test_route_structure()
    print("All tests passed.")