"""Slack-corrected feasibility checker (Steps 1+2 + per-pickup correction).

Fast O(n) check that uses the AEAP forward pass with depot delay optimization,
then applies per-pickup slack correction for ride times. This is the default
mode used during ALNS search — slightly optimistic but very fast.
"""

from __future__ import annotations

from ..instance import DARPInstance


def check(route: list[int], instance: DARPInstance) -> tuple[bool, str]:
    """Fast O(n) feasibility check using Steps 1+2 plus slack correction.

    1. AEAP forward pass with inline time-window/capacity checks.
    2. Compute forward time slack, apply depot delay (Step 2), re-forward.
    3. On the Step 2 schedule, compute per-pickup slack correction for
       ride times:  best_rt = rt_step2 - min(slack_step2[p], wait(p,d))
    4. Check route duration directly on Step 2 schedule.
    """
    nodes = instance.nodes
    n = instance.num_requests
    tt = instance.travel_time
    Q = instance.vehicle_capacity

    size = len(route)
    if size <= 2:
        return True, ""

    # --- Step 1: AEAP forward pass ---
    arrival = [0.0] * size
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

        arrival[i] = depart[i - 1] + tt[prev_id, curr_id]
        begin[i] = max(arrival[i], node.earliest)
        depart[i] = begin[i] + node.service_duration
        load[i] = load[i - 1] + node.demand

        # Time window (AEAP is earliest possible; if late here, always late)
        if begin[i] > node.latest + 1e-6:
            return False, (
                f"Time window violated at node {curr_id}: "
                f"begin={begin[i]:.2f} > latest={node.latest:.2f}"
            )

        # Capacity (timing-independent)
        if load[i] < 0 or load[i] > Q:
            return False, (
                f"Capacity violated at node {curr_id}: "
                f"load={load[i]}, capacity={Q}"
            )

    # --- Compute forward time slack on AEAP (backward pass) ---
    slack_val = nodes[route[-1]].latest - begin[-1]
    for i in range(size - 2, -1, -1):
        wait_next = begin[i + 1] - arrival[i + 1]
        slack_val = min(
            nodes[route[i]].latest - begin[i],
            slack_val + wait_next,
        )

    # --- Step 2: Apply depot delay, re-run forward pass ---
    if slack_val > 1e-6:
        begin[0] += slack_val
        depart[0] = begin[0] + node0.service_duration
        for i in range(1, size):
            prev_id = route[i - 1]
            curr_id = route[i]
            node = nodes[curr_id]
            arrival[i] = depart[i - 1] + tt[prev_id, curr_id]
            begin[i] = max(arrival[i], node.earliest)
            depart[i] = begin[i] + node.service_duration

    # --- Compute slack + cumulative wait on Step 2 schedule ---
    slack = [0.0] * size
    slack[-1] = nodes[route[-1]].latest - begin[-1]
    for i in range(size - 2, -1, -1):
        wait_next = begin[i + 1] - arrival[i + 1]
        slack[i] = min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next,
        )

    cum_wait = [0.0] * size
    for i in range(1, size):
        cum_wait[i] = cum_wait[i - 1] + (begin[i] - arrival[i])

    # --- Ride time + precedence with per-pickup correction ---
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
            w_pd = cum_wait[i] - cum_wait[p]
            best_ride_time = ride_time - min(slack[p], w_pd)
            if best_ride_time > instance.max_ride_time + 1e-6:
                return False, (
                    f"Ride time violated for request {req}: "
                    f"ride_time={best_ride_time:.2f} > max={instance.max_ride_time:.2f}"
                )

    # --- Route duration on Step 2 schedule ---
    route_dur = begin[-1] - depart[0]
    if route_dur > instance.max_route_duration + 1e-6:
        return False, (
            f"Route duration violated: "
            f"duration={route_dur:.2f} > max={instance.max_route_duration:.2f}"
        )

    return True, ""
