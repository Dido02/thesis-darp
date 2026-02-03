"""
Destroy operators for ALNS.

These operators remove requests from the current solution
to create a partial solution that will be repaired.

Operators implemented:
1. Random removal - Remove random requests
2. Worst removal - Remove requests with highest cost contribution
3. Shaw removal - Remove related requests (spatial/temporal similarity)
"""

import random
from typing import List, Set, Callable
import numpy as np
from numpy.f2py.auxfuncs import replace

from ..solution import DARPSolution
from ..instance import DARPInstance


def random_removal(
        solution: DARPSolution,
        num_remove: int,
        rng: random.Random = None
) -> List[int]:
    """
    Remove random requests from the solution.

    Args:
        solution: Current solution (will be modified)
        num_remove: Number of requests to remove
        rng: Random number generator

    Returns:
        List of removed request IDs
    """
    if rng is None:
        rng = random.Random()

    # Get assigned requests
    assigned = list(solution.get_assigned_requests())

    if len(assigned) == 0:
        return []

    # Select random requests to remove
    num_remove = min(num_remove, len(assigned))
    # to_remove = rng.sample(assigned, num_remove)
    to_remove = rng.choice(assigned, size=num_remove, replace=False).tolist()

    # Remove them
    for req_id in to_remove:
        solution.remove_request(req_id)

    return to_remove


def worst_removal(
        solution: DARPSolution,
        num_remove: int,
        rng: random.Random = None,
        determinism: float = 3.0
) -> List[int]:
    """
    Remove requests that contribute most to the cost.

    Uses randomized selection biased toward worst requests.
    The determinism parameter controls how biased the selection is:
    - Higher determinism = more likely to pick the absolute worst
    - Lower determinism = more random selection

    Args:
        solution: Current solution (will be modified)
        num_remove: Number of requests to remove
        rng: Random number generator
        determinism: Higher = more deterministic (always pick worst)

    Returns:
        List of removed request IDs
    """
    if rng is None:
        rng = random.Random()

    instance = solution.instance
    assigned = list(solution.get_assigned_requests())

    if len(assigned) == 0:
        return []

    # Compute cost contribution of each request
    # Cost = distance added by this request's pickup and delivery
    costs = []

    for req_id in assigned:
        route_idx = solution.find_request_route(req_id)
        route = solution.routes[route_idx]
        stops = route.stops

        pickup_node = req_id
        delivery_node = instance.num_requests + req_id

        # Find positions
        pickup_pos = None
        delivery_pos = None
        for i, stop in enumerate(stops):
            if stop.node_id == pickup_node:
                pickup_pos = i
            elif stop.node_id == delivery_node:
                delivery_pos = i

        if pickup_pos is None or delivery_pos is None:
            costs.append((req_id, 0))
            continue

        # Compute detour cost caused by this request
        # This is the extra distance compared to skipping the request
        cost = 0.0

        # Cost around pickup:
        # Current: prev -> pickup -> next
        # Without: prev -> next
        if pickup_pos > 0 and pickup_pos < len(stops) - 1:
            prev_node = stops[pickup_pos - 1].node_id
            next_node = stops[pickup_pos + 1].node_id

            cost += instance.distance(prev_node, pickup_node)
            cost += instance.distance(pickup_node, next_node)
            cost -= instance.distance(prev_node, next_node)

        # Cost around delivery (skip if adjacent to pickup)
        if delivery_pos > 0 and delivery_pos < len(stops) - 1:
            prev_node = stops[delivery_pos - 1].node_id
            next_node = stops[delivery_pos + 1].node_id

            # Only count if prev is not the pickup (already counted)
            if prev_node != pickup_node:
                cost += instance.distance(prev_node, delivery_node)
                cost += instance.distance(delivery_node, next_node)
                cost -= instance.distance(prev_node, next_node)

        costs.append((req_id, cost))

    # Sort by cost (descending = worst first)
    costs.sort(key=lambda x: x[1], reverse=True)

    # Remove with randomization
    removed = []
    num_remove = min(num_remove, len(costs))

    while len(removed) < num_remove and costs:
        # Pick with bias toward worst
        # y = x^p where p = determinism, x uniform [0,1)
        # Higher p = more likely to pick index 0 (worst)
        idx = int(len(costs) * (rng.random() ** determinism))
        idx = min(idx, len(costs) - 1)

        req_id = costs[idx][0]
        solution.remove_request(req_id)
        removed.append(req_id)
        costs.pop(idx)

    return removed


def shaw_removal(
        solution: DARPSolution,
        num_remove: int,
        rng: random.Random = None,
        determinism: float = 3.0,
        weights: tuple = (1.0, 1.0, 1.0)
) -> List[int]:
    """
    Remove related requests (Shaw removal).

    Relatedness is based on:
    - Spatial proximity (pickup and delivery locations)
    - Temporal proximity (time windows)
    - Load similarity

    The idea is that removing related requests creates a larger
    "neighborhood" that can be efficiently repaired.

    Args:
        solution: Current solution (will be modified)
        num_remove: Number of requests to remove
        rng: Random number generator
        determinism: Higher = more deterministic
        weights: (spatial, temporal, load) weights for relatedness

    Returns:
        List of removed request IDs
    """
    if rng is None:
        rng = random.Random()

    instance = solution.instance
    assigned = list(solution.get_assigned_requests())

    if len(assigned) == 0:
        return []

    num_remove = min(num_remove, len(assigned))

    # Start with a random request
    seed_req = rng.choice(assigned)
    removed = [seed_req]
    solution.remove_request(seed_req)
    remaining = [r for r in assigned if r != seed_req]

    w_spatial, w_temporal, w_load = weights

    while len(removed) < num_remove and remaining:
        # Compute relatedness to already removed requests
        relatedness = []

        for req_id in remaining:
            # Average relatedness to all removed requests
            total_rel = 0.0

            for rem_req in removed:
                # Get node info
                p1 = instance.nodes[req_id]  # pickup of candidate
                p2 = instance.nodes[rem_req]  # pickup of removed
                d1 = instance.nodes[instance.num_requests + req_id]  # delivery of candidate
                d2 = instance.nodes[instance.num_requests + rem_req]  # delivery of removed

                # Spatial: distance between pickups + deliveries
                pickup_dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
                delivery_dist = np.sqrt((d1.x - d2.x) ** 2 + (d1.y - d2.y) ** 2)
                spatial = pickup_dist + delivery_dist

                # Temporal: difference in time windows
                time_diff = (abs(p1.earliest - p2.earliest) +
                             abs(d1.earliest - d2.earliest))

                # Load: difference in demand
                load_diff = abs(p1.load - p2.load)

                # Combine (lower = more related)
                rel = w_spatial * spatial + w_temporal * time_diff + w_load * load_diff
                total_rel += rel

            avg_rel = total_rel / len(removed)
            relatedness.append((req_id, avg_rel))

        # Sort by relatedness (ascending = most related first)
        relatedness.sort(key=lambda x: x[1])

        # Pick with bias toward most related
        idx = int(len(relatedness) * (rng.random() ** determinism))
        idx = min(idx, len(relatedness) - 1)

        req_id = relatedness[idx][0]
        solution.remove_request(req_id)
        removed.append(req_id)
        remaining.remove(req_id)

    return removed


def route_removal(
        solution: DARPSolution,
        num_routes: int = 1,
        rng: random.Random = None
) -> List[int]:
    """
    Remove all requests from one or more routes.

    Useful for diversification - allows complete route restructuring.

    Args:
        solution: Current solution (will be modified)
        num_routes: Number of routes to empty
        rng: Random number generator

    Returns:
        List of removed request IDs
    """
    if rng is None:
        rng = random.Random()

    # Find non-empty routes
    non_empty = [i for i, r in enumerate(solution.routes) if not r.is_empty]

    if len(non_empty) == 0:
        return []

    # Select routes to empty
    num_routes = min(num_routes, len(non_empty))
    # routes_to_empty = rng.sample(non_empty, num_routes)
    routes_to_empty = rng.choice(non_empty, size=num_routes, replace=False).tolist()

    removed = []
    for route_idx in routes_to_empty:
        route = solution.routes[route_idx]
        for req_id in list(route.requests_served):
            solution.remove_request(req_id)
            removed.append(req_id)

    return removed


# =============================================================================
# ALNS-COMPATIBLE WRAPPER FUNCTIONS
# =============================================================================

def create_random_removal(num_remove: int) -> Callable:
    """
    Create a random removal operator for ALNS.

    Args:
        num_remove: Number of requests to remove

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        random_removal(solution, num_remove, rng)
        return solution

    operator.__name__ = f"random_removal_{num_remove}"
    return operator


def create_worst_removal(num_remove: int, determinism: float = 3.0) -> Callable:
    """
    Create a worst removal operator for ALNS.

    Args:
        num_remove: Number of requests to remove
        determinism: Higher = more deterministic selection

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        worst_removal(solution, num_remove, rng, determinism)
        return solution

    operator.__name__ = f"worst_removal_{num_remove}"
    return operator


def create_shaw_removal(num_remove: int, determinism: float = 3.0) -> Callable:
    """
    Create a Shaw removal operator for ALNS.

    Args:
        num_remove: Number of requests to remove
        determinism: Higher = more deterministic selection

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        shaw_removal(solution, num_remove, rng, determinism)
        return solution

    operator.__name__ = f"shaw_removal_{num_remove}"
    return operator


def create_route_removal(num_routes: int = 1) -> Callable:
    """
    Create a route removal operator for ALNS.

    Args:
        num_routes: Number of routes to empty

    Returns:
        ALNS-compatible operator function
    """
    def operator(solution: DARPSolution, rng: random.Random, **kwargs) -> DARPSolution:
        route_removal(solution, num_routes, rng)
        return solution

    operator.__name__ = f"route_removal_{num_routes}"
    return operator