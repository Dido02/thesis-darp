"""Full 3-step optimizer feasibility checker.

Runs the complete Cordeau (2006) scheduling algorithm:
  Step 1: AEAP forward pass
  Step 2: Depot delay via forward time slack
  Step 3: Iterative per-pickup delay (multiple passes until convergence)

Then checks all constraints on the optimized schedule. This is the most
accurate but slowest checker — produces the same schedule as
DARPSolution.compute_route_schedule().
"""

from __future__ import annotations

from ..instance import DARPInstance


def check(route: list[int], instance: DARPInstance) -> tuple[bool, str]:
    """Full 3-step optimizer feasibility check.

    Computes the fully optimized schedule and checks all constraints.
    Equivalent to compute_route_schedule() + is_feasible() constraint checks.
    """
    nodes = instance.nodes
    n = instance.num_requests
    tt = instance.travel_time
    Q = instance.vehicle_capacity

    size = len(route)
    if size <= 2:
        return True, ""

    # --- Step 1: Forward pass (earliest feasible times) ---
    arrival = [0.0] * size
    begin = [0.0] * size
    depart = [0.0] * size
    load = [0] * size

    begin[0] = nodes[route[0]].earliest
    depart[0] = begin[0] + nodes[route[0]].service_duration

    for i in range(1, size):
        prev, curr = route[i - 1], route[i]
        node = nodes[curr]
        arrival[i] = depart[i - 1] + tt[prev, curr]
        begin[i] = max(arrival[i], node.earliest)
        depart[i] = begin[i] + node.service_duration
        load[i] = load[i - 1] + node.demand

    # --- Step 2: Optimize ride times by delaying departures ---
    # Compute forward time slack
    slack = [0.0] * size
    slack[-1] = nodes[route[-1]].latest - begin[-1]
    for i in range(size - 2, -1, -1):
        wait_next = begin[i + 1] - arrival[i + 1]
        slack[i] = min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next,
        )

    # Apply: delay depot departure by slack[0]
    if slack[0] > 1e-6:
        shift = slack[0]
        begin[0] += shift
        depart[0] = begin[0] + nodes[route[0]].service_duration

        for i in range(1, size):
            prev, curr = route[i - 1], route[i]
            node = nodes[curr]
            arrival[i] = depart[i - 1] + tt[prev, curr]
            begin[i] = max(arrival[i], node.earliest)
            depart[i] = begin[i] + node.service_duration

    # --- Step 3: Further reduce ride times by adding waiting at pickups ---
    for _pass in range(size):
        improved = False
        for i in range(size):
            node_id = route[i]
            if not (1 <= node_id <= n):
                continue

            req = node_id
            del_id = req + n
            del_pos = None
            for j in range(i + 1, size):
                if route[j] == del_id:
                    del_pos = j
                    break
            if del_pos is None:
                continue

            current_rt = begin[del_pos] - depart[i]
            if current_rt <= instance.max_ride_time + 1e-6:
                continue

            needed = current_rt - instance.max_ride_time

            # Compute forward time slack at position i
            fi = nodes[route[size - 1]].latest - begin[size - 1]
            for j in range(size - 2, i - 1, -1):
                wait_next = begin[j + 1] - arrival[j + 1]
                fi = min(
                    nodes[route[j]].latest - begin[j],
                    fi + wait_next,
                )

            delay = min(needed, fi)
            if delay < 1e-6:
                continue

            improved = True
            begin[i] += delay
            depart[i] = begin[i] + nodes[route[i]].service_duration
            for j in range(i + 1, size):
                prev_j, curr_j = route[j - 1], route[j]
                arrival[j] = depart[j - 1] + tt[prev_j, curr_j]
                begin[j] = max(arrival[j], nodes[curr_j].earliest)
                depart[j] = begin[j] + nodes[curr_j].service_duration

        if not improved:
            break

    # --- Constraint checks on optimized schedule ---

    # Time windows
    for i in range(size):
        node = nodes[route[i]]
        if begin[i] > node.latest + 1e-6:
            return False, (
                f"Time window violated at node {route[i]}: "
                f"begin={begin[i]:.2f} > latest={node.latest:.2f}"
            )

    # Capacity
    for i in range(size):
        if load[i] < 0 or load[i] > Q:
            return False, (
                f"Capacity violated at node {route[i]}: "
                f"load={load[i]}, capacity={Q}"
            )

    # Precedence + ride time
    pickup_depart = {}
    for i in range(size):
        node_id = route[i]
        if 1 <= node_id <= n:
            pickup_depart[node_id] = depart[i]
        elif n + 1 <= node_id <= 2 * n:
            req = node_id - n
            if req not in pickup_depart:
                return False, f"Precedence violated: delivery {node_id} before pickup {req}"
            ride_time = begin[i] - pickup_depart[req]
            if ride_time > instance.max_ride_time + 1e-6:
                return False, (
                    f"Ride time violated for request {req}: "
                    f"ride_time={ride_time:.2f} > max={instance.max_ride_time:.2f}"
                )

    # Route duration
    route_dur = begin[-1] - depart[0]
    if route_dur > instance.max_route_duration + 1e-6:
        return False, (
            f"Route duration violated: "
            f"duration={route_dur:.2f} > max={instance.max_route_duration:.2f}"
        )

    return True, ""
