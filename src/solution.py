"""DARP solution representation, scheduling, and feasibility checking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instance import DARPInstance


@dataclass
class RouteSchedule:
    """Computed schedule for a single route."""

    arrival_time: list[float]
    begin_service: list[float]
    departure_time: list[float]
    load: list[int]
    ride_time: dict[int, float]  # request_id -> ride time


class DARPSolution:
    """A solution to a DARP instance.

    Routes are lists of node IDs: [0, ..., 2n+1] (depot to depot).
    Empty routes are [0, 2n+1].
    """

    def __init__(self, instance: DARPInstance):
        self.instance = instance
        n = instance.num_requests
        depot_o = 0
        depot_d = 2 * n + 1

        # One route per vehicle, initially empty (depot -> depot)
        self.routes: list[list[int]] = [
            [depot_o, depot_d] for _ in range(instance.num_vehicles)
        ]
        self.unassigned: set[int] = set(range(1, n + 1))

    def copy(self) -> DARPSolution:
        """Cheap copy for ALNS iterations."""
        new = DARPSolution.__new__(DARPSolution)
        new.instance = self.instance  # shared, read-only
        new.routes = [route[:] for route in self.routes]
        new.unassigned = set(self.unassigned)
        return new

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def compute_route_schedule(self, route: list[int]) -> RouteSchedule:
        """Forward pass with wait-time optimization to reduce ride times.

        Step 1: Forward pass with earliest feasible times.
        Step 2: Push begin_service forward at each node as much as possible
                without violating time windows, to reduce passenger ride times.
                This follows Cordeau (2006) Section 4.
        """
        inst = self.instance
        nodes = inst.nodes
        tt = inst.travel_time
        n = inst.num_requests

        size = len(route)
        if size <= 2:
            # Empty route: depot -> depot
            return RouteSchedule(
                arrival_time=[0.0, 0.0],
                begin_service=[0.0, 0.0],
                departure_time=[0.0, 0.0],
                load=[0, 0],
                ride_time={},
            )

        # --- Step 1: Forward pass (earliest feasible times) ---
        arrival = [0.0] * size
        begin = [0.0] * size
        depart = [0.0] * size
        load = [0] * size

        arrival[0] = 0.0
        begin[0] = nodes[route[0]].earliest
        depart[0] = begin[0] + nodes[route[0]].service_duration
        load[0] = 0

        for i in range(1, size):
            prev, curr = route[i - 1], route[i]
            node = nodes[curr]
            arrival[i] = depart[i - 1] + tt[prev, curr]
            begin[i] = max(arrival[i], node.earliest)
            depart[i] = begin[i] + node.service_duration
            load[i] = load[i - 1] + node.demand

        # --- Step 2: Optimize ride times by delaying departures ---
        # For each pickup-delivery pair, compute how much we could delay
        # the pickup departure without violating any downstream time windows.
        # Then apply the minimum feasible delay at the depot to push
        # everything forward.
        #
        # We use the forward time slack F_i from Cordeau (2006):
        #   F_i = min over j>=i of (latest_j - begin_service_j)
        # This tells us how much we can delay node i without violating
        # any time window at i or later.

        # Compute forward time slack
        slack = [0.0] * size
        slack[-1] = nodes[route[-1]].latest - begin[-1]
        for i in range(size - 2, -1, -1):
            wait_next = begin[i + 1] - arrival[i + 1]  # waiting at i+1
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
        # For each pickup, delay begin_service to reduce ride time.
        # Multiple passes needed since delaying one pickup affects downstream.
        for _pass in range(size):  # at most size passes
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
                if current_rt <= inst.max_ride_time + 1e-6:
                    continue

                needed = current_rt - inst.max_ride_time

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

        # Compute ride times
        ride_time = {}
        pickup_depart = {}
        for i, node_id in enumerate(route):
            if 1 <= node_id <= n:
                pickup_depart[node_id] = depart[i]
            elif n + 1 <= node_id <= 2 * n:
                req = node_id - n
                if req in pickup_depart:
                    ride_time[req] = begin[i] - pickup_depart[req]

        return RouteSchedule(
            arrival_time=arrival,
            begin_service=begin,
            departure_time=depart,
            load=load,
            ride_time=ride_time,
        )

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def check_route_feasibility(self, route: list[int]) -> tuple[bool, str]:
        """Check all constraints for a single route. Returns (feasible, reason)."""
        inst = self.instance
        nodes = inst.nodes
        n = inst.num_requests
        sched = self.compute_route_schedule(route)

        for i, node_id in enumerate(route):
            node = nodes[node_id]

            # Time window
            if sched.begin_service[i] > node.latest + 1e-6:
                return False, (
                    f"Time window violated at node {node_id}: "
                    f"begin={sched.begin_service[i]:.2f} > latest={node.latest:.2f}"
                )

            # Capacity
            if sched.load[i] < 0 or sched.load[i] > inst.vehicle_capacity:
                return False, (
                    f"Capacity violated at node {node_id}: "
                    f"load={sched.load[i]}, capacity={inst.vehicle_capacity}"
                )

        # Ride time
        for req, rt in sched.ride_time.items():
            if rt > inst.max_ride_time + 1e-6:
                return False, (
                    f"Ride time violated for request {req}: "
                    f"ride_time={rt:.2f} > max={inst.max_ride_time:.2f}"
                )

        # Route duration
        route_dur = sched.begin_service[-1] - sched.departure_time[0]
        if route_dur > inst.max_route_duration + 1e-6:
            return False, (
                f"Route duration violated: "
                f"duration={route_dur:.2f} > max={inst.max_route_duration:.2f}"
            )

        # Precedence: pickup must appear before delivery
        seen_pickups = set()
        for node_id in route:
            if 1 <= node_id <= n:
                seen_pickups.add(node_id)
            elif n + 1 <= node_id <= 2 * n:
                req = node_id - n
                if req not in seen_pickups:
                    return False, f"Precedence violated: delivery {node_id} before pickup {req}"

        return True, ""

    def is_feasible(self) -> tuple[bool, str]:
        """Check feasibility of entire solution."""
        # All requests must be assigned
        if self.unassigned:
            return False, f"Unassigned requests: {self.unassigned}"

        # Each request appears exactly once
        seen = set()
        for route in self.routes:
            for node_id in route:
                n = self.instance.num_requests
                if 1 <= node_id <= n:
                    if node_id in seen:
                        return False, f"Request {node_id} appears in multiple routes"
                    seen.add(node_id)

        if seen != set(range(1, self.instance.num_requests + 1)):
            missing = set(range(1, self.instance.num_requests + 1)) - seen
            return False, f"Missing requests: {missing}"

        # Check each route
        for k, route in enumerate(self.routes):
            feasible, reason = self.check_route_feasibility(route)
            if not feasible:
                return False, f"Route {k}: {reason}"

        return True, ""

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    # Penalty per unassigned request. Large enough that ALNS always
    # prefers serving a request over skipping it.
    UNASSIGNED_PENALTY = 1000.0

    def objective(self) -> float:
        """Total travel time + penalty for unassigned requests."""
        tt = self.instance.travel_time
        total = 0.0
        for route in self.routes:
            for i in range(len(route) - 1):
                total += tt[route[i], route[i + 1]]
        total += len(self.unassigned) * self.UNASSIGNED_PENALTY
        return total

    def travel_cost(self) -> float:
        """Pure travel time without penalty (for reporting)."""
        tt = self.instance.travel_time
        total = 0.0
        for route in self.routes:
            for i in range(len(route) - 1):
                total += tt[route[i], route[i + 1]]
        return total

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def request_to_route(self) -> dict[int, int]:
        """Map request_id -> route index."""
        n = self.instance.num_requests
        mapping = {}
        for k, route in enumerate(self.routes):
            for node_id in route:
                if 1 <= node_id <= n:
                    mapping[node_id] = k
        return mapping

    def print_solution(self) -> str:
        """Print solution in Cordeau format and return as string.

        Format per route:
          Route header: k D:<distance> Q:<max_load> W:<total_wait> T:<avg_ride_time>
          Per node: id (w:<wait> a:<begin_service>; t:<ride_time>; q:<load>)

        Footer: total duration, total waiting, total transit.
        """
        inst = self.instance
        n = inst.num_requests
        tt = inst.travel_time
        lines = []

        lines.append(f"{self.travel_cost():.2f}")
        if self.unassigned:
            lines.append(f"  (penalty: {len(self.unassigned)} unassigned × "
                         f"{self.UNASSIGNED_PENALTY} = {len(self.unassigned) * self.UNASSIGNED_PENALTY:.0f}, "
                         f"total obj: {self.objective():.2f})")

        total_duration = 0.0
        total_waiting = 0.0
        total_transit = 0.0
        total_passengers = 0

        for k, route in enumerate(self.routes):
            sched = self.compute_route_schedule(route)

            # Route distance (D = route duration: last arrival - first departure)
            route_dist = sched.begin_service[-1] - sched.departure_time[0]

            # Max load on route
            max_load = max(sched.load) if sched.load else 0

            # Total waiting on route (excluding depot departure delay)
            route_wait = 0.0
            for i in range(1, len(route)):  # skip origin depot
                wait = sched.begin_service[i] - sched.arrival_time[i]
                route_wait += max(0.0, wait)

            # Average ride time on route
            if sched.ride_time:
                avg_rt = sum(sched.ride_time.values()) / len(sched.ride_time)
            else:
                avg_rt = 0.0

            # Build node details
            # Track ride times at delivery nodes: need to know which node is
            # a delivery and what the ride time is at that point
            # For pickups: t=0, for deliveries: t=ride_time of that request
            # For depots: t=0
            node_strs = []
            for i, node_id in enumerate(route):
                wait = max(0.0, sched.begin_service[i] - sched.arrival_time[i])
                begin = sched.begin_service[i]
                load = sched.load[i]

                # Ride time display
                if n + 1 <= node_id <= 2 * n:
                    # Delivery node
                    req = node_id - n
                    rt = sched.ride_time.get(req, 0.0)
                else:
                    rt = 0.0

                if i == 0:
                    # Origin depot: show as "0 (b:<begin>; t:0.00; q:0.00)"
                    node_strs.append(f"0 (b:{begin:.2f}; t:0.00; q:0.00)")
                elif i == len(route) - 1:
                    # Destination depot
                    node_strs.append(f"0 (w:{wait:.2f} a:{begin:.2f}; t:0.00; q:0.00)")
                else:
                    node_strs.append(
                        f"{node_id} (w:{wait:.2f} a:{begin:.2f}; t:{rt:.2f}; q:{load:.2f})"
                    )

            header = (
                f"{k + 1} D: {route_dist:>8.2f} Q: {max_load:>6.2f} "
                f"W: {route_wait:>9.2f} T: {avg_rt:>8.2f}"
            )
            lines.append(f"{header}  {' '.join(node_strs)}")

            total_duration += route_dist
            total_waiting += route_wait
            total_transit += sum(sched.ride_time.values())
            total_passengers += len(sched.ride_time)

        lines.append("")
        lines.append(f"Total duration:  {total_duration:>10.2f}")
        lines.append(
            f"Total waiting time:  {total_waiting:>10.2f}; "
            f"average:    {total_waiting / max(1, total_passengers):>8.2f}"
        )
        lines.append(
            f"Total transit time:  {total_transit:>10.2f}; "
            f"average:    {total_transit / max(1, total_passengers):>8.2f}"
        )

        output = "\n".join(lines)
        print(output)
        return output

    def __repr__(self) -> str:
        n_assigned = self.instance.num_requests - len(self.unassigned)
        return (
            f"DARPSolution(assigned={n_assigned}/{self.instance.num_requests}, "
            f"objective={self.objective():.2f})"
        )