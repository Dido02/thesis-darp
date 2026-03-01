"""Greedy insertion repair operator for ALNS."""

from __future__ import annotations

import numpy as np

from ...solution import DARPSolution


def greedy_insertion(sol: DARPSolution, rng: np.random.Generator) -> DARPSolution:
    """Reinsert all unassigned requests at their cheapest feasible position.

    Iteratively picks the request with the cheapest feasible insertion
    and inserts it, until all requests are inserted or no feasible
    insertion remains.
    """
    repaired = sol.copy()
    tt = repaired.instance.travel_time

    while repaired.unassigned:
        best_cost = float("inf")
        best_req = None
        best_route_idx = None
        best_p_pos = None
        best_d_pos = None

        for req in list(repaired.unassigned):
            pickup_id = req
            delivery_id = req + repaired.instance.num_requests

            for k, route in enumerate(repaired.routes):
                for p_pos in range(1, len(route)):
                    for d_pos in range(p_pos + 1, len(route) + 1):
                        candidate = list(route)
                        candidate.insert(p_pos, pickup_id)
                        candidate.insert(d_pos, delivery_id)

                        cost = _route_cost(candidate, tt) - _route_cost(route, tt)

                        if cost >= best_cost:
                            continue

                        feasible, _ = repaired.check_route_feasibility(candidate)
                        if feasible:
                            best_cost = cost
                            best_req = req
                            best_route_idx = k
                            best_p_pos = p_pos
                            best_d_pos = d_pos

        if best_req is None:
            break  # No feasible insertion for any remaining request

        repaired.routes[best_route_idx].insert(best_p_pos, best_req)
        repaired.routes[best_route_idx].insert(best_d_pos, best_req + repaired.instance.num_requests)
        repaired.unassigned.discard(best_req)

    return repaired


def _route_cost(route: list[int], tt) -> float:
    """Total travel time for a route."""
    return sum(tt[route[i], route[i + 1]] for i in range(len(route) - 1))