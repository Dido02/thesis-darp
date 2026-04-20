"""Objective recomputation, ride-time scheduling, and slack/cost helpers.

Scheduling and slack functions are bit-exact mirrors of their C++ siblings:
- compute_route_schedule -> cpp/src/solution.cpp::DARPSolution::compute_route_schedule
- compute_route_slack    -> cpp/src/operators/repair/slack_insertion.cpp::compute_route_slack
- compute_route_cost     -> cpp/src/operators/repair/slack_insertion.cpp::compute_route_cost

Mirroring the C++ exactly is required for max_abs_err = 0 parity on features
that consume ride-times or slack. We use Python floats (no numpy ops in the
inner loops) so accumulation order matches C++ doubles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .instance import DARPInstance, InstanceTooLargeError, load_instance
from .loader import LogHeader, SolverState, StateTrial, load_log, resolve_instance_path

# Mirrors EPS in cpp/src/solution.cpp and slack_insertion.cpp.
EPS = 1e-6


# ---------------------------------------------------------------------------
# Per-route schedule (Tang AEAP). Bit-exact mirror of the C++ implementation.
# Returns a dict: request_id -> actual ride time (delivery.begin - pickup.depart).
# Empty routes return {}.
# ---------------------------------------------------------------------------
def compute_route_schedule(route: list[int], instance: DARPInstance) -> dict[int, float]:
    nodes = instance.nodes
    tt = instance.travel_time
    n = instance.num_requests
    size = len(route)

    if size <= instance.empty_route_size:
        return {}

    # Step 1: forward pass — earliest feasible begin/depart times.
    arrival = [0.0] * size
    begin = [0.0] * size
    depart = [0.0] * size

    arrival[0] = 0.0
    begin[0] = float(nodes[route[0]].earliest)
    depart[0] = begin[0] + float(nodes[route[0]].service_duration)

    for i in range(1, size):
        prev, curr = route[i - 1], route[i]
        node = nodes[curr]
        arrival[i] = depart[i - 1] + float(tt[prev][curr])
        begin[i] = max(arrival[i], float(node.earliest))
        depart[i] = begin[i] + float(node.service_duration)

    # Step 2: shift the entire schedule by the depot-position slack to reduce
    # ride times where possible without violating any time window.
    slack = [0.0] * size
    slack[size - 1] = float(nodes[route[size - 1]].latest) - begin[size - 1]
    for i in range(size - 2, -1, -1):
        wait_next = begin[i + 1] - arrival[i + 1]
        slack[i] = min(float(nodes[route[i]].latest) - begin[i],
                       slack[i + 1] + wait_next)

    if slack[0] > EPS:
        shift = slack[0]
        begin[0] += shift
        depart[0] = begin[0] + float(nodes[route[0]].service_duration)
        for i in range(1, size):
            prev, curr = route[i - 1], route[i]
            node = nodes[curr]
            arrival[i] = depart[i - 1] + float(tt[prev][curr])
            begin[i] = max(arrival[i], float(node.earliest))
            depart[i] = begin[i] + float(node.service_duration)

    # Step 3: iteratively delay individual pickup departures to shorten the
    # ride time of the corresponding request, when forward time slack allows.
    for _pass in range(size):
        improved = False
        for i in range(size):
            node_id = route[i]
            if not (1 <= node_id <= n):
                continue
            req = node_id
            del_id = req + n
            del_pos = -1
            for j in range(i + 1, size):
                if route[j] == del_id:
                    del_pos = j
                    break
            if del_pos == -1:
                continue

            current_rt = begin[del_pos] - depart[i]
            max_rt = instance.max_ride_time_for(req)
            if current_rt <= max_rt + EPS:
                continue
            needed = current_rt - max_rt

            # Forward time slack at position i.
            fi = float(nodes[route[size - 1]].latest) - begin[size - 1]
            for j in range(size - 2, i - 1, -1):
                wait_next = begin[j + 1] - arrival[j + 1]
                fi = min(float(nodes[route[j]].latest) - begin[j],
                         fi + wait_next)

            delay = min(needed, fi)
            if delay < EPS:
                continue

            improved = True
            begin[i] += delay
            depart[i] = begin[i] + float(nodes[route[i]].service_duration)
            for j in range(i + 1, size):
                prev_j, curr_j = route[j - 1], route[j]
                node_j = nodes[curr_j]
                arrival[j] = depart[j - 1] + float(tt[prev_j][curr_j])
                begin[j] = max(arrival[j], float(node_j.earliest))
                depart[j] = begin[j] + float(node_j.service_duration)
        if not improved:
            break

    # Compute ride times.
    ride_time: dict[int, float] = {}
    pickup_depart: dict[int, float] = {}
    for i in range(size):
        node_id = route[i]
        if 1 <= node_id <= n:
            pickup_depart[node_id] = depart[i]
        elif n + 1 <= node_id <= 2 * n:
            req = node_id - n
            if req in pickup_depart:
                ride_time[req] = begin[i] - pickup_depart[req]
    return ride_time


def compute_ride_times(routes: list[list[int]],
                        instance: DARPInstance) -> dict[int, float]:
    """Merge per-route ride-time dicts. Requests not currently in any route
    are absent from the result. Mirrors how C++ feature code aggregates by
    iterating routes and reading RouteSchedule::ride_time."""
    out: dict[int, float] = {}
    for route in routes:
        out.update(compute_route_schedule(route, instance))
    return out


# ---------------------------------------------------------------------------
# Per-route forward time slack. Bit-exact mirror of the C++ helper in
# cpp/src/operators/repair/slack_insertion.cpp.
# ---------------------------------------------------------------------------
def compute_route_slack(route: list[int], instance: DARPInstance) -> float:
    nodes = instance.nodes
    tt = instance.travel_time
    size = len(route)
    if size <= instance.empty_route_size:
        return 0.0

    # Forward pass: tau[i] = max(arrival, earliest)
    tau = [0.0] * size
    tau[0] = float(nodes[route[0]].earliest)
    depart = tau[0] + float(nodes[route[0]].service_duration)
    for i in range(1, size):
        arrival = depart + float(tt[route[i - 1]][route[i]])
        tau[i] = max(arrival, float(nodes[route[i]].earliest))
        depart = tau[i] + float(nodes[route[i]].service_duration)

    # Per-position arrival_arr (mirrors the second pass in C++ exactly).
    arrival_arr = [0.0] * size
    depart = tau[0] + float(nodes[route[0]].service_duration)
    for i in range(1, size):
        arrival_arr[i] = depart + float(tt[route[i - 1]][route[i]])
        depart = tau[i] + float(nodes[route[i]].service_duration)

    # Backward recurrence.
    slack = [0.0] * size
    slack[size - 1] = float(nodes[route[size - 1]].latest) - tau[size - 1]
    for i in range(size - 2, -1, -1):
        wait_next = tau[i + 1] - arrival_arr[i + 1]
        slack[i] = min(float(nodes[route[i]].latest) - tau[i],
                       slack[i + 1] + wait_next)

    total = 0.0
    for i in range(size):
        total += slack[i]
    return total


def compute_route_cost(route: list[int], instance: DARPInstance) -> float:
    """Sum of edge travel times along a route. Mirrors C++ helper in
    slack_insertion.cpp."""
    cost = 0.0
    tt = instance.travel_time
    for i in range(len(route) - 1):
        cost += float(tt[route[i]][route[i + 1]])
    return cost


def removal_cost(routes: list[list[int]], instance: DARPInstance,
                  req: int, route_idx: int) -> float:
    """Cost saved by removing a request (pickup+delivery) from a known route.

    Bit-exact mirror of cpp/src/operators/destroy/worst_removal.cpp::removal_cost.
    Preconditions: `req` is in 1..n; `routes[route_idx]` contains BOTH the
    pickup (node id == req) AND its delivery (node id == req + n). Returns 0.0
    if the pickup is missing (matches C++).
    """
    tt = instance.travel_time
    n = instance.num_requests
    pickup_id = req
    delivery_id = req + n
    route = routes[route_idx]

    p_pos, d_pos = -1, -1
    for i in range(len(route)):
        if route[i] == pickup_id:
            p_pos = i
        if route[i] == delivery_id:
            d_pos = i
    if p_pos < 0:
        return 0.0

    cost_with_p = float(tt[route[p_pos - 1]][pickup_id]) \
                + float(tt[pickup_id][route[p_pos + 1]])
    cost_without_p = float(tt[route[p_pos - 1]][route[p_pos + 1]])

    d_adj = d_pos - 1 if d_pos > p_pos else d_pos
    route_no_p = [route[i] for i in range(len(route)) if i != p_pos]

    cost_with_d = float(tt[route_no_p[d_adj - 1]][delivery_id]) \
                + float(tt[delivery_id][route_no_p[d_adj + 1]])
    cost_without_d = float(tt[route_no_p[d_adj - 1]][route_no_p[d_adj + 1]])

    return (cost_with_p - cost_without_p) + (cost_with_d - cost_without_d)


def compute_objective(
    routes: list[list[int]],
    unassigned: list[int],
    travel_time: np.ndarray,
    unassigned_penalty: float,
) -> float:
    """Recompute objective from routes and unassigned set.

    Mirrors cpp/src/solution.cpp::objective() lines 282-294 exactly:
        obj = sum(travel_time[route[i]][route[i+1]]) + len(unassigned) * penalty
    """
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += travel_time[route[i], route[i + 1]]
    total += len(unassigned) * unassigned_penalty
    return total


@dataclass
class ValidationReport:
    log_path: str
    instance_path: str
    total_states: int
    passed: int
    failed: int
    max_abs_error: float
    mismatches: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def validate_state(
    state: SolverState,
    instance: DARPInstance,
    unassigned_penalty: float,
    tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Validate one state record. Returns (passed, abs_error)."""
    recomputed = compute_objective(
        state.routes, state.unassigned,
        instance.travel_time, unassigned_penalty,
    )
    error = abs(recomputed - state.obj_before)
    return error < tolerance, error


def validate_log(
    log_path: str | Path,
    data_root: str | Path,
    tolerance: float = 0.05,
    max_mismatches: int = 10,
) -> ValidationReport:
    """Validate all state records in a log against recomputed objectives."""
    log_path = Path(log_path)
    header, state_trials = load_log(log_path)

    # Resolve and load instance
    instance_path = resolve_instance_path(header.instance_path, data_root)
    instance = load_instance(instance_path)

    report = ValidationReport(
        log_path=str(log_path),
        instance_path=str(instance_path),
        total_states=len(state_trials),
        passed=0,
        failed=0,
        max_abs_error=0.0,
    )

    for st in state_trials:
        passed, error = validate_state(
            st.state, instance, header.unassigned_penalty, tolerance,
        )
        report.max_abs_error = max(report.max_abs_error, error)

        if passed:
            report.passed += 1
        else:
            report.failed += 1
            if len(report.mismatches) < max_mismatches:
                recomputed = compute_objective(
                    st.state.routes, st.state.unassigned,
                    instance.travel_time, header.unassigned_penalty,
                )
                report.mismatches.append({
                    "iter": st.state.iter,
                    "logged": st.state.obj_before,
                    "recomputed": recomputed,
                    "error": error,
                    "n_unassigned": len(st.state.unassigned),
                })

    return report
