"""Greedy construction heuristic for DARP initial solutions."""

from __future__ import annotations

from .instance import DARPInstance
from .solution import DARPSolution


def build_initial_solution(instance: DARPInstance) -> DARPSolution:
    """Build an initial solution by greedy insertion.

    For each unassigned request, find the cheapest feasible
    (route, pickup_pos, delivery_pos) and insert it.
    """
    sol = DARPSolution(instance)
    n = instance.num_requests
    tt = instance.travel_time

    for req in range(1, n + 1):
        pickup_id = req
        delivery_id = req + n

        best_cost = float("inf")
        best_route_idx = None
        best_p_pos = None
        best_d_pos = None

        for k, route in enumerate(sol.routes):
            # Try all (pickup_pos, delivery_pos) combinations
            for p_pos in range(1, len(route)):
                for d_pos in range(p_pos + 1, len(route) + 1):
                    # Build candidate route
                    candidate = list(route)
                    candidate.insert(p_pos, pickup_id)
                    candidate.insert(d_pos, delivery_id)

                    # Insertion cost = new route cost - old route cost
                    cost = _route_cost(candidate, tt) - _route_cost(route, tt)

                    if cost >= best_cost:
                        continue

                    # Check feasibility
                    feasible, _ = sol.check_route_feasibility(candidate)
                    if feasible:
                        best_cost = cost
                        best_route_idx = k
                        best_p_pos = p_pos
                        best_d_pos = d_pos

        if best_route_idx is not None:
            sol.routes[best_route_idx].insert(best_p_pos, pickup_id)
            sol.routes[best_route_idx].insert(best_d_pos, delivery_id)
            sol.unassigned.discard(req)

    return sol


def _route_cost(route: list[int], tt) -> float:
    """Total travel time for a route."""
    return sum(tt[route[i], route[i + 1]] for i in range(len(route) - 1))