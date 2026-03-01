"""Tests for ALNS operators."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau
from src.initial_solution import build_initial_solution
from src.operators.destroy.random_removal import random_removal
from src.operators.repair.greedy_insertion import greedy_insertion

DATA_DIR = PROJECT_ROOT / "data" / "cordeau"


def test_random_removal():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)
    rng = np.random.default_rng(42)

    destroyed = random_removal(sol, rng)

    # Some requests should be removed
    assert len(destroyed.unassigned) > 0, "Should have removed some requests"

    # Original should be unchanged
    assert len(sol.unassigned) == 0, "Original should be untouched"

    # Removed requests should not appear in any route
    n = inst.num_requests
    for req in destroyed.unassigned:
        pickup_id = req
        delivery_id = req + n
        for route in destroyed.routes:
            assert pickup_id not in route, f"Pickup {pickup_id} still in route"
            assert delivery_id not in route, f"Delivery {delivery_id} still in route"

    # Remaining routes should still be feasible
    for k, route in enumerate(destroyed.routes):
        feasible, reason = destroyed.check_route_feasibility(route)
        assert feasible, f"Route {k} infeasible after removal: {reason}"

    print(f"  [PASS] random_removal (removed {len(destroyed.unassigned)} requests)")


def test_greedy_insertion():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)
    rng = np.random.default_rng(42)

    # Destroy then repair
    destroyed = random_removal(sol, rng)
    removed_count = len(destroyed.unassigned)
    repaired = greedy_insertion(destroyed, rng)

    # All requests should be reassigned
    assert len(repaired.unassigned) == 0, f"Unassigned after repair: {repaired.unassigned}"

    # Solution should be feasible
    feasible, reason = repaired.is_feasible()
    assert feasible, f"Repaired solution infeasible: {reason}"

    print(f"  [PASS] greedy_insertion (reinserted {removed_count} requests)")


def test_destroy_repair_cycle():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")
    sol = build_initial_solution(inst)
    rng = np.random.default_rng(123)

    original_obj = sol.objective()

    # Run 10 destroy-repair cycles
    for i in range(10):
        destroyed = random_removal(sol, rng)
        repaired = greedy_insertion(destroyed, rng)
        assert len(repaired.unassigned) == 0

        feasible, reason = repaired.is_feasible()
        assert feasible, f"Cycle {i}: infeasible: {reason}"

    print(f"  [PASS] destroy_repair_cycle (10 cycles, obj: {original_obj:.2f} -> {repaired.objective():.2f})")


if __name__ == "__main__":
    print("Testing operators:")
    test_random_removal()
    test_greedy_insertion()
    test_destroy_repair_cycle()
    print("All tests passed.")