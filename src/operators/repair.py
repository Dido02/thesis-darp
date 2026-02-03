"""
Repair operators for ALNS.

These operators insert unassigned requests back into the solution.

Operators implemented:
1. Greedy insertion - Insert at cheapest feasible position
2. Regret-k insertion - Prioritize requests with high "regret"
"""

import random
from typing import List, Optional, Tuple, Callable

from ..solution import DARPSolution, RouteStop
from ..instance import DARPInstance
from ..feasibility import (
    check_insertion_feasibility,
    compute_insertion_cost,
    check_route_feasibility,
    update_route_schedule
)


def greedy_insertion(
        solution: DARPSolution,
        rng: random.Random = None
) -> DARPSolution:
    """
    Insert all unassigned requests using greedy insertion.

    For each unassigned request, find the cheapest feasible insertion
    and perform it. Requests are processed sorted by time window
    to encourage feasible interleaving.

    Args:
        solution: Current solution (will be modified)
        rng: Random number generator (used to shuffle request order)

    Returns:
        Modified solution
    """
    if rng is None:
        rng = random.Random()

    instance = solution.instance

    # Sort requests by pickup earliest time (helps with time window feasibility)
    unassigned = list(solution.unassigned)
    unassigned.sort(key=lambda r: instance.nodes[r].earliest)

    # Add some randomization while keeping rough time order
    # Shuffle within small windows
    window_size = max(3, len(unassigned) // 8)
    for i in range(0, len(unassigned), window_size):
        end = min(i + window_size, len(unassigned))
        window = unassigned[i:end]
        rng.shuffle(window)
        unassigned[i:end] = window

    for req_id in list(unassigned):  # Use list() since we modify during iteration
        if req_id not in solution.unassigned:
            continue  # Already inserted

        best = find_best_insertion_position(solution, req_id)

        if best is not None:
            route_idx, pickup_pos, delivery_pos, _ = best
            solution.insert_request(req_id, route_idx, pickup_pos, delivery_pos)
            # Update schedule for the modified route
            update_route_schedule(instance, solution.routes[route_idx])

    return solution


def regret_insertion(
        solution: DARPSolution,
        regret_k: int = 2,
        rng: random.Random = None
) -> DARPSolution:
    """
    Insert requests using regret-k insertion.

    Prioritizes requests where the difference between the best
    and k-th best insertion is largest (high "regret" if not
    inserted now).

    The intuition: if a request has only one good insertion spot,
    we should insert it now before that spot is taken.

    Args:
        solution: Current solution (will be modified)
        regret_k: Number of best insertions to consider (typically 2 or 3)
        rng: Random number generator

    Returns:
        Modified solution
    """
    if rng is None:
        rng = random.Random()

    instance = solution.instance

    while solution.unassigned:
        # Compute regret for each unassigned request
        regrets = []

        for req_id in solution.unassigned:
            insertions = find_k_best_insertions(solution, req_id, regret_k)

            if not insertions:
                # No feasible insertion - will remain unassigned
                regrets.append((req_id, float('-inf'), None))
                continue

            # Regret = difference between best and k-th best (or last available)
            best_cost = insertions[0][3]

            if len(insertions) >= regret_k:
                kth_cost = insertions[regret_k - 1][3]
            else:
                # If fewer than k insertions, use the worst available
                kth_cost = insertions[-1][3]

            regret = kth_cost - best_cost
            regrets.append((req_id, regret, insertions[0]))

        # Sort by regret (descending) - highest regret first
        regrets.sort(key=lambda x: x[1], reverse=True)

        # Insert request with highest regret
        if regrets and regrets[0][2] is not None:
            req_id = regrets[0][0]
            route_idx, pickup_pos, delivery_pos, _ = regrets[0][2]
            solution.insert_request(req_id, route_idx, pickup_pos, delivery_pos)
            # Update schedule
            update_route_schedule(instance, solution.routes[route_idx])
        else:
            # No feasible insertions for any request - stop
            break

    return solution


def find_best_insertion_position(
        solution: DARPSolution,
        request_id: int
) -> Optional[Tuple[int, int, int, float]]:
    """
    Find the best feasible insertion position for a request.

    Tries all routes and all (pickup_pos, delivery_pos) combinations.

    Args:
        solution: Current solution
        request_id: Request to insert

    Returns:
        Tuple of (route_idx, pickup_pos, delivery_pos, cost) or None if no feasible insertion
    """
    best = None
    best_cost = float('inf')

    for route_idx, route in enumerate(solution.routes):
        n_stops = len(route.stops)

        # Try all pickup positions (not at position 0 which is depot start,
        # and not at the last position which is depot end)
        for pickup_pos in range(1, n_stops):
            # Delivery must come after pickup
            for delivery_pos in range(pickup_pos + 1, n_stops + 1):
                # Check feasibility
                result = check_insertion_feasibility(
                    solution, request_id, route_idx, pickup_pos, delivery_pos
                )

                if result.is_feasible:
                    cost = compute_insertion_cost(
                        solution, request_id, route_idx, pickup_pos, delivery_pos
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best = (route_idx, pickup_pos, delivery_pos, cost)

    return best


def find_k_best_insertions(
        solution: DARPSolution,
        request_id: int,
        k: int
) -> List[Tuple[int, int, int, float]]:
    """
    Find the k best feasible insertion positions for a request.

    Args:
        solution: Current solution
        request_id: Request to insert
        k: Number of best insertions to find

    Returns:
        List of (route_idx, pickup_pos, delivery_pos, cost) tuples, sorted by cost (ascending)
    """
    insertions = []

    for route_idx, route in enumerate(solution.routes):
        n_stops = len(route.stops)

        for pickup_pos in range(1, n_stops):
            for delivery_pos in range(pickup_pos + 1, n_stops + 1):
                result = check_insertion_feasibility(
                    solution, request_id, route_idx, pickup_pos, delivery_pos
                )

                if result.is_feasible:
                    cost = compute_insertion_cost(
                        solution, request_id, route_idx, pickup_pos, delivery_pos
                    )
                    insertions.append((route_idx, pickup_pos, delivery_pos, cost))

    # Sort by cost and return top k
    insertions.sort(key=lambda x: x[3])
    return insertions[:k]


def random_insertion(
        solution: DARPSolution,
        rng: random.Random = None
) -> DARPSolution:
    """
    Insert requests at random feasible positions.

    For each unassigned request, find all feasible insertions
    and pick one randomly.

    Args:
        solution: Current solution (will be modified)
        rng: Random number generator

    Returns:
        Modified solution
    """
    if rng is None:
        rng = random.Random()

    instance = solution.instance
    unassigned = list(solution.unassigned)
    rng.shuffle(unassigned)

    for req_id in unassigned:
        if req_id not in solution.unassigned:
            continue

        # Find all feasible insertions
        feasible = []
        for route_idx, route in enumerate(solution.routes):
            n_stops = len(route.stops)
            for pickup_pos in range(1, n_stops):
                for delivery_pos in range(pickup_pos + 1, n_stops + 1):
                    result = check_insertion_feasibility(
                        solution, req_id, route_idx, pickup_pos, delivery_pos
                    )
                    if result.is_feasible:
                        feasible.append((route_idx, pickup_pos, delivery_pos))

        if feasible:
            route_idx, pickup_pos, delivery_pos = rng.choice(feasible)
            solution.insert_request(req_id, route_idx, pickup_pos, delivery_pos)
            update_route_schedule(instance, solution.routes[route_idx])

    return solution


# =============================================================================
# ALNS-COMPATIBLE WRAPPER FUNCTIONS
# =============================================================================

def create_greedy_repair() -> Callable:
    """
    Create a greedy repair operator for ALNS.

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        return greedy_insertion(solution, rng)

    operator.__name__ = "greedy_repair"
    return operator


def create_regret_repair(k: int = 2) -> Callable:
    """
    Create a regret-k repair operator for ALNS.

    Args:
        k: Number of best insertions to consider for regret calculation

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        return regret_insertion(solution, k, rng)

    operator.__name__ = f"regret_{k}_repair"
    return operator


def create_random_repair() -> Callable:
    """
    Create a random repair operator for ALNS.

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        return random_insertion(solution, rng)

    operator.__name__ = "random_repair"
    return operator