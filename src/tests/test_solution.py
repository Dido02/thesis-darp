"""Tests for solution.py"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.instance import DARPInstance, Node
from src.parsers.cordeau import parse_cordeau
from src.solution import DARPSolution

DATA_DIR = PROJECT_ROOT / "data" / "cordeau"


def build_known_solution(inst: DARPInstance) -> DARPSolution:
    """Build the known Cordeau pr01 solution (cost=190.02).

    Route 1: 0 10 11 35 34 0
    Route 2: 0 14 22 3 27 46 38 12 24 48 6 36 15 18 30 21 39 42 45 0
    Route 3: 0 9 17 33 8 20 1 41 7 31 44 32 2 25 5 13 29 26 16 4 28 37 19 23 40 47 43 0
    """
    sol = DARPSolution(inst)
    n = inst.num_requests  # 24
    depot_d = 2 * n + 1  # 49

    sol.routes[0] = [0, 10, 11, 35, 34, depot_d]
    sol.routes[1] = [0, 14, 22, 3, 27, 46, 38, 12, 24, 48, 6, 36, 15, 18, 30, 21, 39, 42, 45, depot_d]
    sol.routes[2] = [0, 9, 17, 33, 8, 20, 1, 41, 7, 31, 44, 32, 2, 25, 5, 13, 29, 26, 16, 4, 28, 37, 19, 23, 40, 47, 43, depot_d]
    sol.unassigned = set()
    return sol


def test_empty_solution():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = DARPSolution(inst)

    # All routes are depot -> depot
    assert len(sol.routes) == 3
    for route in sol.routes:
        assert route == [0, 49]

    # All requests unassigned
    assert sol.unassigned == set(range(1, 25))

    # Objective includes unassigned penalty: 24 requests × 1000
    assert sol.objective() == 24 * sol.UNASSIGNED_PENALTY

    print("  [PASS] empty_solution")


def test_copy():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_known_solution(inst)
    sol2 = sol.copy()

    # Same content
    assert sol.routes == sol2.routes
    assert sol.unassigned == sol2.unassigned

    # Independent: modifying copy doesn't affect original
    sol2.routes[0].append(99)
    assert 99 not in sol.routes[0]

    sol2.unassigned.add(99)
    assert 99 not in sol.unassigned

    # Same instance reference
    assert sol.instance is sol2.instance

    print("  [PASS] copy")


def test_schedule():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_known_solution(inst)

    # Test route 0 (short route): 0 -> 10 -> 11 -> 35 -> 34 -> 49
    sched = sol.compute_route_schedule(sol.routes[0])

    # Origin depot starts at time 0 (or earliest)
    assert sched.begin_service[0] >= 0.0

    # Load should go: 0, +1, +1, -1, -1, 0
    assert sched.load[0] == 0  # depot
    assert sched.load[1] == 1  # pickup 10
    assert sched.load[2] == 2  # pickup 11
    assert sched.load[3] == 1  # delivery 35 (= request 11)
    assert sched.load[4] == 0  # delivery 34 (= request 10)
    assert sched.load[5] == 0  # dest depot

    # Ride times should exist for requests 10 and 11
    assert 10 in sched.ride_time
    assert 11 in sched.ride_time

    # Ride times must be non-negative
    assert sched.ride_time[10] >= 0
    assert sched.ride_time[11] >= 0

    # Times should be monotonically non-decreasing
    for i in range(1, len(sched.begin_service)):
        assert sched.begin_service[i] >= sched.begin_service[i - 1] - 1e-6

    print("  [PASS] schedule")


def test_known_solution_feasibility():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_known_solution(inst)

    feasible, reason = sol.is_feasible()
    assert feasible, f"Known solution should be feasible but: {reason}"

    print("  [PASS] known_solution_feasibility")


def test_known_solution_objective():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_known_solution(inst)

    obj = sol.objective()
    # Known solution cost is 190.02 (total travel distance)
    assert abs(obj - 190.02) < 0.5, f"Objective {obj:.2f} should be ~190.02"

    print("  [PASS] known_solution_objective")


def test_infeasible_capacity():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = DARPSolution(inst)
    n = inst.num_requests
    depot_d = 2 * n + 1

    # Stuff all pickups into one route without deliveries interleaved
    # This should violate capacity (Q=6) quickly
    route = [0] + list(range(1, n + 1)) + list(range(n + 1, 2 * n + 1)) + [depot_d]
    sol.routes[0] = route
    sol.routes[1] = [0, depot_d]
    sol.routes[2] = [0, depot_d]
    sol.unassigned = set()

    feasible, reason = sol.is_feasible()
    assert not feasible, "Should be infeasible (capacity)"
    assert "Capacity" in reason or "capacity" in reason.lower()

    print("  [PASS] infeasible_capacity")


def test_infeasible_precedence():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = DARPSolution(inst)
    n = inst.num_requests
    depot_d = 2 * n + 1

    # Put delivery before pickup for request 1
    sol.routes[0] = [0, 25, 1, depot_d]  # delivery 25 (req 1) before pickup 1
    sol.unassigned = set(range(2, n + 1))

    feasible, reason = sol.check_route_feasibility(sol.routes[0])
    assert not feasible, "Should be infeasible (delivery before pickup)"

    print("  [PASS] infeasible_precedence")


def test_request_to_route():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_known_solution(inst)

    mapping = sol.request_to_route()

    # All 24 requests should be mapped
    assert len(mapping) == 24

    # Requests 10, 11 should be on route 0
    assert mapping[10] == 0
    assert mapping[11] == 0

    # Request 14 should be on route 1
    assert mapping[14] == 1

    # Request 9 should be on route 2
    assert mapping[9] == 2

    print("  [PASS] request_to_route")


if __name__ == "__main__":
    print("Testing solution:")
    test_empty_solution()
    test_copy()
    test_schedule()
    test_known_solution_feasibility()
    test_known_solution_objective()
    test_infeasible_capacity()
    test_infeasible_precedence()
    test_request_to_route()
    print("All tests passed.")