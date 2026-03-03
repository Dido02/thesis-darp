"""Greedy insertion repair operator for ALNS."""

from __future__ import annotations

import numpy as np

from ...solution import DARPSolution


def greedy_insertion(sol: DARPSolution, rng: np.random.Generator) -> DARPSolution:
    """Reinsert all unassigned requests at their cheapest feasible position.

    Iteratively picks the request with the cheapest feasible insertion
    and inserts it, until all requests are inserted or no feasible
    insertion remains.

    Optimizations over naive approach:
    - Precomputed forward-pass schedule (departure times, loads) per route
    - Pickup reachability filter (O(1) per p_pos)
    - Capacity filter (O(segment) per candidate)
    - O(1) cost delta instead of two O(n) route cost computations
    - Cost pruning before feasibility check
    - Fast O(n) feasibility check (single forward pass, no ride-time optimizer)
    """
    repaired = sol.copy()
    inst = repaired.instance
    tt = inst.travel_time
    nodes = inst.nodes
    n = inst.num_requests
    Q = inst.vehicle_capacity

    while repaired.unassigned:
        best_cost = float("inf")
        best_req = None
        best_route_idx = None
        best_p_pos = None
        best_d_pos = None

        # Precompute forward schedules for all routes
        route_caches = []
        for route in repaired.routes:
            depart, load = _forward_pass(route, nodes, tt)
            route_caches.append((depart, load))

        for req in list(repaired.unassigned):
            pickup_id = req
            delivery_id = req + n

            for k, route in enumerate(repaired.routes):
                depart_cache, load_cache = route_caches[k]
                rlen = len(route)

                for p_pos in range(1, rlen):
                    # Reachability filter: can we reach pickup in time?
                    arrival_at_pickup = depart_cache[p_pos - 1] + tt[route[p_pos - 1], pickup_id]
                    if arrival_at_pickup > nodes[pickup_id].latest + 1e-6:
                        continue

                    for d_pos in range(p_pos + 1, rlen + 1):
                        # Capacity filter: check load between pickup and delivery
                        cap_ok = True
                        for j in range(p_pos - 1, d_pos - 1):
                            if load_cache[j] + 1 > Q:
                                cap_ok = False
                                break
                        if not cap_ok:
                            continue

                        # O(1) cost delta
                        if d_pos == p_pos + 1:
                            # Adjacent: remove r[p-1]->r[p], add r[p-1]->P->D->r[p]
                            delta = (tt[route[p_pos - 1], pickup_id]
                                     + tt[pickup_id, delivery_id]
                                     + tt[delivery_id, route[p_pos]]
                                     - tt[route[p_pos - 1], route[p_pos]])
                        else:
                            # Non-adjacent: two edge replacements
                            delta = (tt[route[p_pos - 1], pickup_id]
                                     + tt[pickup_id, route[p_pos]]
                                     - tt[route[p_pos - 1], route[p_pos]]
                                     + tt[route[d_pos - 2], delivery_id]
                                     + tt[delivery_id, route[d_pos - 1]]
                                     - tt[route[d_pos - 2], route[d_pos - 1]])

                        # Cost pruning
                        if delta >= best_cost:
                            continue

                        # Fast feasibility check (O(n) forward pass)
                        candidate = list(route)
                        candidate.insert(p_pos, pickup_id)
                        candidate.insert(d_pos, delivery_id)
                        feasible, _ = repaired.check_route_feasibility(candidate)
                        if feasible:
                            best_cost = delta
                            best_req = req
                            best_route_idx = k
                            best_p_pos = p_pos
                            best_d_pos = d_pos

        if best_req is None:
            break  # No feasible insertion for any remaining request

        repaired.routes[best_route_idx].insert(best_p_pos, best_req)
        repaired.routes[best_route_idx].insert(best_d_pos, best_req + n)
        repaired.unassigned.discard(best_req)

    return repaired


def _forward_pass(route, nodes, tt):
    """Compute departure times and loads via a single forward pass."""
    size = len(route)
    depart = [0.0] * size
    load = [0] * size

    node0 = nodes[route[0]]
    depart[0] = node0.earliest + node0.service_duration
    load[0] = 0

    for i in range(1, size):
        prev_id = route[i - 1]
        curr_id = route[i]
        node = nodes[curr_id]
        arrival = depart[i - 1] + tt[prev_id, curr_id]
        begin = max(arrival, node.earliest)
        depart[i] = begin + node.service_duration
        load[i] = load[i - 1] + node.demand

    return depart, load
