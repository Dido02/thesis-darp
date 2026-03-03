"""Pure AEAP forward-pass feasibility checker (most conservative).

Single forward pass with inline constraint checks. No depot delay optimization,
no slack correction. Ride times and route duration are computed on the raw AEAP
schedule, making this the most conservative (strictest) checker.

This will reject some routes that are actually feasible after optimization,
but is the fastest mode and useful as a baseline.
"""

from __future__ import annotations

from ..instance import DARPInstance


def check(route: list[int], instance: DARPInstance) -> tuple[bool, str]:
    """Pure AEAP forward-pass feasibility check.

    Single forward pass computing earliest feasible times, with inline checks
    for time windows, capacity, precedence, ride time, and route duration.
    No depot delay, no slack correction.
    """
    nodes = instance.nodes
    n = instance.num_requests
    tt = instance.travel_time
    Q = instance.vehicle_capacity

    size = len(route)
    if size <= 2:
        return True, ""

    # --- AEAP forward pass ---
    begin = [0.0] * size
    depart = [0.0] * size
    load = [0] * size

    node0 = nodes[route[0]]
    begin[0] = node0.earliest
    depart[0] = begin[0] + node0.service_duration

    for i in range(1, size):
        prev_id = route[i - 1]
        curr_id = route[i]
        node = nodes[curr_id]

        arrival_i = depart[i - 1] + tt[prev_id, curr_id]
        begin[i] = max(arrival_i, node.earliest)
        depart[i] = begin[i] + node.service_duration
        load[i] = load[i - 1] + node.demand

        # Time window check
        if begin[i] > node.latest + 1e-6:
            return False, (
                f"Time window violated at node {curr_id}: "
                f"begin={begin[i]:.2f} > latest={node.latest:.2f}"
            )

        # Capacity check
        if load[i] < 0 or load[i] > Q:
            return False, (
                f"Capacity violated at node {curr_id}: "
                f"load={load[i]}, capacity={Q}"
            )

    # --- Precedence + ride time on raw AEAP schedule ---
    pickup_pos = {}
    for i in range(size):
        node_id = route[i]
        if 1 <= node_id <= n:
            pickup_pos[node_id] = i
        elif n + 1 <= node_id <= 2 * n:
            req = node_id - n
            if req not in pickup_pos:
                return False, f"Precedence violated: delivery {node_id} before pickup {req}"
            p = pickup_pos[req]
            ride_time = begin[i] - depart[p]
            if ride_time > instance.max_ride_time + 1e-6:
                return False, (
                    f"Ride time violated for request {req}: "
                    f"ride_time={ride_time:.2f} > max={instance.max_ride_time:.2f}"
                )

    # --- Route duration on AEAP schedule (no depot delay) ---
    route_dur = begin[-1] - depart[0]
    if route_dur > instance.max_route_duration + 1e-6:
        return False, (
            f"Route duration violated: "
            f"duration={route_dur:.2f} > max={instance.max_route_duration:.2f}"
        )

    return True, ""
