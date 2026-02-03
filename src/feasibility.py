"""
DARP Feasibility Checking

This module provides two implementations:
1. Simple version - Easy to understand, O(n²) for insertion checks
2. Cordeau's 8-step algorithm - Optimized with forward time slack, O(n) per route

Based on:
- Cordeau & Laporte (2003) "A tabu search heuristic for the static multi-vehicle DARP"
- Cordeau & Laporte (2007) "The dial-a-ride problem: models and algorithms"

Key Constraints for DARP:
1. Capacity: Load ≤ Q at all times
2. Time Windows: arrival_i ≤ latest_i for all nodes
3. Precedence: Pickup i before delivery n+i on same route
4. Ride Time: For user i, (arrival at delivery) - (departure from pickup) ≤ L
5. Route Duration: Total route time ≤ T
"""

from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import numpy as np

from .instance import DARPInstance
from .solution import DARPSolution, Route, RouteStop


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeasibilityResult:
    """
    Result of a feasibility check.

    Used by operators to check insertion feasibility.
    """
    is_feasible: bool
    reason: str = ""

    def __bool__(self):
        return self.is_feasible


@dataclass
class NodeTiming:
    """
    Timing information for a node in a route.

    This stores all the computed values needed for Cordeau's algorithm.
    """
    node_id: int

    # Basic timing
    arrival: float = 0.0          # A_i: arrival time
    start_service: float = 0.0    # B_i: beginning of service
    departure: float = 0.0        # D_i: departure time (B_i + service_time)
    wait_time: float = 0.0        # W_i: waiting time (B_i - A_i)

    # Load
    load_after: float = 0.0       # Q_i: load after visiting this node

    # Ride time (for deliveries)
    ride_time: float = 0.0        # L_i: ride time for delivered passenger

    # Forward time slack (Cordeau's key insight)
    forward_slack: float = 0.0    # F_i: how much we can push the schedule forward


# =============================================================================
# PART 1: SIMPLE VERSION (Easy to understand)
# =============================================================================

def check_route_feasibility_simple(
    instance: DARPInstance,
    route: Route,
    return_reason: bool = False
) -> Union[bool, Tuple[bool, str]]:
    """
    Simple feasibility check for a route.

    This version recomputes the entire schedule from scratch and checks
    all constraints. Easy to understand but O(n) per check, so O(n²)
    when checking all insertion positions.

    Args:
        instance: The DARP instance
        route: The route to check
        return_reason: If True, return (feasible, reason) tuple

    Returns:
        True if feasible, False otherwise
        If return_reason=True, returns (bool, str) with explanation
    """
    nodes = route.get_node_sequence()

    if len(nodes) < 2:
        return (True, "Empty route") if return_reason else True

    # === Check 1: Precedence ===
    # For each request, pickup must appear before delivery
    pickup_seen = set()
    for node_id in nodes:
        if instance.is_pickup(node_id):
            pickup_seen.add(node_id)
        elif instance.is_delivery(node_id):
            pickup_node = instance.get_paired_node(node_id)
            if pickup_node not in pickup_seen:
                reason = f"Precedence violated: delivery {node_id} before pickup {pickup_node}"
                return (False, reason) if return_reason else False

    # === Compute Schedule (Forward Pass) ===
    # We compute arrival time, start of service, and departure time at each node

    current_time = instance.nodes[nodes[0]].earliest  # Start at depot earliest
    current_load = 0

    # Track departure times from pickups (needed for ride time check)
    pickup_departure: Dict[int, float] = {}

    schedule = []  # List of (arrival, start_service, departure, load)

    for i, node_id in enumerate(nodes):
        node = instance.nodes[node_id]

        # Travel time from previous node
        if i > 0:
            prev_node = nodes[i - 1]
            travel = instance.travel_time(prev_node, node_id)
            arrival = current_time + travel
        else:
            arrival = current_time

        # === Check 2: Time Window ===
        if arrival > node.latest:
            reason = f"Time window violated at node {node_id}: arrival {arrival:.2f} > latest {node.latest:.2f}"
            return (False, reason) if return_reason else False

        # Start of service (wait if early)
        start_service = max(arrival, node.earliest)

        # Departure time
        departure = start_service + node.service_time

        # === Check 3: Capacity ===
        current_load += node.load  # +load for pickup, -load for delivery

        if current_load > instance.vehicles[0].capacity:
            reason = f"Capacity violated at node {node_id}: load {current_load} > capacity {instance.vehicles[0].capacity}"
            return (False, reason) if return_reason else False

        if current_load < 0:
            reason = f"Negative load at node {node_id}: {current_load}"
            return (False, reason) if return_reason else False

        # Track pickup departures for ride time calculation
        if instance.is_pickup(node_id):
            pickup_departure[node_id] = departure

        # === Check 4: Ride Time ===
        if instance.is_delivery(node_id):
            pickup_node = instance.get_paired_node(node_id)
            if pickup_node in pickup_departure:
                ride_time = arrival - pickup_departure[pickup_node]
                if ride_time > instance.max_ride_time:
                    reason = f"Ride time violated for request {instance.get_request_id(node_id)}: {ride_time:.2f} > {instance.max_ride_time}"
                    return (False, reason) if return_reason else False

        schedule.append((arrival, start_service, departure, current_load))
        current_time = departure

    # === Check 5: Route Duration ===
    if len(schedule) >= 2:
        route_duration = schedule[-1][0] - schedule[0][2]  # arrival at end - departure from start
        max_duration = instance.vehicles[0].max_route_duration
        if route_duration > max_duration:
            reason = f"Route duration violated: {route_duration:.2f} > {max_duration}"
            return (False, reason) if return_reason else False

    return (True, "Feasible") if return_reason else True


def compute_schedule_simple(
    instance: DARPInstance,
    route: Route
) -> List[RouteStop]:
    """
    Compute the schedule for a route (arrival times, service times, etc.)

    This updates the RouteStop objects with timing information.
    Uses earliest possible times (no intentional waiting beyond time windows).

    Args:
        instance: The DARP instance
        route: The route to schedule

    Returns:
        List of RouteStop objects with updated timing
    """
    nodes = route.get_node_sequence()

    if len(nodes) == 0:
        return []

    current_time = instance.nodes[nodes[0]].earliest
    current_load = 0
    pickup_departure: Dict[int, float] = {}

    updated_stops = []

    for i, node_id in enumerate(nodes):
        node = instance.nodes[node_id]

        # Travel from previous node
        if i > 0:
            prev_node = nodes[i - 1]
            travel = instance.travel_time(prev_node, node_id)
            arrival = current_time + travel
        else:
            arrival = current_time

        # Start service (wait if early)
        start_service = max(arrival, node.earliest)

        # Departure
        departure = start_service + node.service_time

        # Update load
        current_load += node.load

        # Track pickup departure for ride time
        if instance.is_pickup(node_id):
            pickup_departure[node_id] = departure

        # Compute ride time for deliveries
        ride_time = None
        if instance.is_delivery(node_id):
            pickup_node = instance.get_paired_node(node_id)
            if pickup_node in pickup_departure:
                ride_time = arrival - pickup_departure[pickup_node]

        stop = RouteStop(
            node_id=node_id,
            arrival_time=arrival,
            start_service=start_service,
            departure_time=departure,
            load_after=current_load,
            ride_time=ride_time
        )
        updated_stops.append(stop)

        current_time = departure

    return updated_stops


def update_route_schedule(instance: DARPInstance, route: Route) -> None:
    """
    Update a route's stops with computed schedule in-place.

    Args:
        instance: The DARP instance
        route: The route to update (modified in-place)
    """
    route.stops = compute_schedule_simple(instance, route)


# =============================================================================
# PART 2: CORDEAU'S 8-STEP ALGORITHM (Optimized with Forward Time Slack)
# =============================================================================

def compute_forward_time_slack(
    instance: DARPInstance,
    nodes: List[int],
    timings: List[NodeTiming]
) -> None:
    """
    Compute forward time slack for each node (Cordeau's algorithm).

    The forward time slack F_i represents the maximum amount of time that
    the departure from node i can be delayed without violating:
    - Time window constraints at subsequent nodes
    - Ride time constraints for passengers currently on board

    This is computed BACKWARDS from the last node.

    Formula (from Cordeau 2003):
        F_i = min(F_{i+1} + W_{i+1}, slack_tw_i, slack_rt_i)

    where:
        - W_{i+1} = wait time at next node
        - slack_tw_i = l_i - B_i (time window slack)
        - slack_rt_i = L - ride_time (ride time slack for users being delivered after i)
    """
    n = len(nodes)
    if n == 0:
        return

    # Initialize F for last node (depot end)
    # At the last node, slack is just the time window slack
    last = timings[n - 1]
    last_node = instance.nodes[nodes[n - 1]]
    last.forward_slack = last_node.latest - last.start_service

    # Track which users are on board (picked up but not yet delivered)
    # We process backwards, so we "un-deliver" and "un-pickup"
    users_on_board: Dict[int, int] = {}  # request_id -> delivery_index

    # First pass: identify delivery indices
    delivery_index: Dict[int, int] = {}  # request_id -> index in route where delivered
    for i, node_id in enumerate(nodes):
        if instance.is_delivery(node_id):
            req_id = instance.get_request_id(node_id)
            delivery_index[req_id] = i

    # Backward pass to compute forward slack
    for i in range(n - 2, -1, -1):
        node_id = nodes[i]
        node = instance.nodes[node_id]
        timing = timings[i]
        next_timing = timings[i + 1]

        # Base slack from next node (can push forward if next node has wait time)
        slack = next_timing.forward_slack + next_timing.wait_time

        # Constraint 1: Time window at current node
        tw_slack = node.latest - timing.start_service
        slack = min(slack, tw_slack)

        # Constraint 2: Ride time for users currently on board
        # If we're at a pickup, the user just got on
        # Their ride time constraint limits how much we can delay
        if instance.is_pickup(node_id):
            req_id = instance.get_request_id(node_id)
            if req_id in delivery_index:
                del_idx = delivery_index[req_id]
                # Current ride time
                current_ride = timings[del_idx].arrival - timing.departure
                # Slack in ride time
                rt_slack = instance.max_ride_time - current_ride
                slack = min(slack, rt_slack)

        timing.forward_slack = max(0, slack)


def compute_route_timing_cordeau(
    instance: DARPInstance,
    route: Route
) -> Tuple[List[NodeTiming], bool, str]:
    """
    Compute full timing information for a route using Cordeau's algorithm.

    This implements the 8-step evaluation from Cordeau (2003):
    1. Initialize at depot
    2. Forward pass: compute arrival, service start, departure, wait, load
    3. Check capacity constraints
    4. Check time window constraints
    5. Check precedence constraints
    6. Compute ride times
    7. Check ride time constraints
    8. Compute forward time slack

    Args:
        instance: The DARP instance
        route: The route to evaluate

    Returns:
        Tuple of:
        - List of NodeTiming objects
        - bool: True if feasible
        - str: Reason if infeasible
    """
    nodes = route.get_node_sequence()
    n = len(nodes)

    if n == 0:
        return [], True, "Empty route"

    timings: List[NodeTiming] = []

    # === Step 1: Initialize ===
    current_time = instance.nodes[nodes[0]].earliest
    current_load = 0

    # Track pickup departures for ride time calculation
    pickup_departure: Dict[int, float] = {}  # pickup_node -> departure_time
    pickup_index: Dict[int, int] = {}  # pickup_node -> index in route

    # === Step 2-5: Forward Pass ===
    for i, node_id in enumerate(nodes):
        node = instance.nodes[node_id]

        # === Precedence check (Step 5) ===
        if instance.is_delivery(node_id):
            pickup_node = instance.get_paired_node(node_id)
            if pickup_node not in pickup_departure:
                return timings, False, f"Precedence: delivery {node_id} before pickup {pickup_node}"

        # Travel time
        if i > 0:
            prev_node = nodes[i - 1]
            travel = instance.travel_time(prev_node, node_id)
            arrival = current_time + travel
        else:
            arrival = current_time

        # === Time window check (Step 4) ===
        if arrival > node.latest:
            return timings, False, f"Time window at node {node_id}: arrival {arrival:.2f} > latest {node.latest}"

        # Service timing
        start_service = max(arrival, node.earliest)
        wait_time = start_service - arrival
        departure = start_service + node.service_time

        # === Capacity check (Step 3) ===
        current_load += node.load

        if current_load > instance.vehicles[0].capacity:
            return timings, False, f"Capacity at node {node_id}: {current_load} > {instance.vehicles[0].capacity}"

        if current_load < 0:
            return timings, False, f"Negative load at node {node_id}"

        # Track pickups
        if instance.is_pickup(node_id):
            pickup_departure[node_id] = departure
            pickup_index[node_id] = i

        # === Ride time calculation (Step 6) ===
        ride_time = 0.0
        if instance.is_delivery(node_id):
            pickup_node = instance.get_paired_node(node_id)
            ride_time = arrival - pickup_departure[pickup_node]

            # === Ride time check (Step 7) ===
            if ride_time > instance.max_ride_time:
                req_id = instance.get_request_id(node_id)
                return timings, False, f"Ride time for request {req_id}: {ride_time:.2f} > {instance.max_ride_time}"

        timing = NodeTiming(
            node_id=node_id,
            arrival=arrival,
            start_service=start_service,
            departure=departure,
            wait_time=wait_time,
            load_after=current_load,
            ride_time=ride_time,
            forward_slack=0.0  # Computed in step 8
        )
        timings.append(timing)

        current_time = departure

    # === Route Duration Check ===
    if len(timings) >= 2:
        route_duration = timings[-1].arrival - timings[0].departure
        max_duration = instance.vehicles[0].max_route_duration
        if route_duration > max_duration:
            return timings, False, f"Route duration: {route_duration:.2f} > {max_duration}"

    # === Step 8: Compute Forward Time Slack ===
    compute_forward_time_slack(instance, nodes, timings)

    return timings, True, "Feasible"


def check_route_feasibility_cordeau(
    instance: DARPInstance,
    route: Route,
    return_reason: bool = False
) -> Union[bool, Tuple[bool, str]]:
    """
    Check route feasibility using Cordeau's algorithm.

    Args:
        instance: The DARP instance
        route: The route to check
        return_reason: If True, return (feasible, reason) tuple

    Returns:
        True if feasible, False otherwise
    """
    _, feasible, reason = compute_route_timing_cordeau(instance, route)
    return (feasible, reason) if return_reason else feasible


# =============================================================================
# PART 3: WRAPPER FUNCTIONS (For operators compatibility)
# =============================================================================

def check_route_feasibility(
    route: Route,
    instance: DARPInstance,
    update_schedule: bool = False,
    return_reason: bool = False
) -> Union[bool, Tuple[bool, str], FeasibilityResult]:
    """
    Check route feasibility - wrapper function for operators.

    This is the main function called by operators.

    Args:
        route: The route to check
        instance: The DARP instance
        update_schedule: If True, update route stops with timing info
        return_reason: If True, return (feasible, reason) tuple

    Returns:
        bool, (bool, str), or FeasibilityResult depending on flags
    """
    feasible, reason = check_route_feasibility_simple(instance, route, return_reason=True)

    if update_schedule and feasible:
        update_route_schedule(instance, route)

    if return_reason:
        return (feasible, reason)
    return feasible


def check_insertion_feasibility(
    solution: DARPSolution,
    request_id: int,
    route_idx: int,
    pickup_pos: int,
    delivery_pos: int
) -> FeasibilityResult:
    """
    Check if inserting a request at given positions is feasible.

    This is the main function called by repair operators.

    Args:
        solution: Current solution
        request_id: Request to insert
        route_idx: Index of route to insert into
        pickup_pos: Position for pickup
        delivery_pos: Position for delivery (must be > pickup_pos)

    Returns:
        FeasibilityResult with is_feasible and reason
    """
    instance = solution.instance

    # Basic validation
    if delivery_pos <= pickup_pos:
        return FeasibilityResult(False, "delivery_pos must be > pickup_pos")

    # Create a copy of the route
    route_copy = solution.routes[route_idx].copy()

    # Get node IDs
    pickup_node = request_id
    delivery_node = instance.num_requests + request_id

    # Insert nodes
    pickup_stop = RouteStop(node_id=pickup_node)
    delivery_stop = RouteStop(node_id=delivery_node)

    route_copy.stops.insert(pickup_pos, pickup_stop)
    route_copy.stops.insert(delivery_pos, delivery_stop)

    # Check feasibility
    feasible, reason = check_route_feasibility_simple(instance, route_copy, return_reason=True)

    return FeasibilityResult(feasible, reason)


def compute_insertion_cost(
    solution: DARPSolution,
    request_id: int,
    route_idx: int,
    pickup_pos: int,
    delivery_pos: int
) -> float:
    """
    Compute the cost increase of inserting a request.

    Cost = increase in total distance

    Args:
        solution: Current solution
        request_id: Request to insert
        route_idx: Route to insert into
        pickup_pos: Position for pickup
        delivery_pos: Position for delivery

    Returns:
        Cost increase (float), or inf if infeasible
    """
    instance = solution.instance
    route = solution.routes[route_idx]
    nodes = route.get_node_sequence()

    pickup_node = request_id
    delivery_node = instance.num_requests + request_id

    # Current distance
    old_distance = solution.route_distance(route)

    # Create modified node sequence
    new_nodes = nodes.copy()
    new_nodes.insert(pickup_pos, pickup_node)
    new_nodes.insert(delivery_pos, delivery_node)

    # Compute new distance
    new_distance = 0.0
    for i in range(len(new_nodes) - 1):
        new_distance += instance.distance(new_nodes[i], new_nodes[i + 1])

    return new_distance - old_distance


def find_best_insertion(
    instance: DARPInstance,
    solution: DARPSolution,
    request_id: int
) -> Optional[Tuple[int, int, int, float]]:
    """
    Find the best feasible insertion position for a request.

    Tries all routes and all position combinations.

    Args:
        instance: The DARP instance
        solution: Current solution
        request_id: Request to insert

    Returns:
        Tuple of (route_idx, pickup_pos, delivery_pos, cost) or None if no feasible insertion
    """
    best_insertion = None
    best_cost = float('inf')

    for route_idx, route in enumerate(solution.routes):
        num_stops = len(route.stops)

        # Try all pickup positions (between depot_start and depot_end)
        for pickup_pos in range(1, num_stops):
            # Delivery must come after pickup
            for delivery_pos in range(pickup_pos + 1, num_stops + 1):
                result = check_insertion_feasibility(
                    solution, request_id, route_idx, pickup_pos, delivery_pos
                )

                if result.is_feasible:
                    cost = compute_insertion_cost(
                        solution, request_id, route_idx, pickup_pos, delivery_pos
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_insertion = (route_idx, pickup_pos, delivery_pos, cost)

    return best_insertion


# =============================================================================
# PART 4: SOLUTION-LEVEL FEASIBILITY
# =============================================================================

def check_solution_feasibility(
    instance: DARPInstance,
    solution: DARPSolution,
    use_cordeau: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if an entire solution is feasible.

    Args:
        instance: The DARP instance
        solution: The solution to check
        use_cordeau: If True, use Cordeau's algorithm; otherwise simple version

    Returns:
        Tuple of (feasible, list of violation reasons)
    """
    violations = []

    for i, route in enumerate(solution.routes):
        if use_cordeau:
            feasible, reason = check_route_feasibility_cordeau(instance, route, return_reason=True)
        else:
            feasible, reason = check_route_feasibility_simple(instance, route, return_reason=True)

        if not feasible:
            violations.append(f"Route {i}: {reason}")

    # Check that each request is served exactly once (or unassigned)
    request_count: Dict[int, int] = {r: 0 for r in range(1, instance.num_requests + 1)}

    for route in solution.routes:
        for stop in route.stops:
            if instance.is_pickup(stop.node_id):
                req_id = instance.get_request_id(stop.node_id)
                request_count[req_id] += 1

    for req_id, count in request_count.items():
        if count > 1:
            violations.append(f"Request {req_id} served {count} times")
        elif count == 0 and req_id not in solution.unassigned:
            violations.append(f"Request {req_id} neither served nor marked unassigned")

    return len(violations) == 0, violations


def update_solution_schedules(instance: DARPInstance, solution: DARPSolution) -> None:
    """
    Update all route schedules in a solution.

    Call this after making modifications to ensure timing is up-to-date.

    Args:
        instance: The DARP instance
        solution: The solution to update (modified in-place)
    """
    for route in solution.routes:
        update_route_schedule(instance, route)


# =============================================================================
# PART 5: UTILITY FUNCTIONS
# =============================================================================

def print_route_schedule(
    instance: DARPInstance,
    route: Route,
    use_cordeau: bool = True
) -> None:
    """
    Print a detailed schedule for a route.

    Useful for debugging.
    """
    nodes = route.get_node_sequence()

    if use_cordeau:
        timings, feasible, reason = compute_route_timing_cordeau(instance, route)
        print(f"Route feasibility: {feasible} ({reason})")
        print()
        print(f"{'Node':>6} {'Type':>8} {'Arrival':>10} {'Service':>10} {'Depart':>10} {'Wait':>8} {'Load':>6} {'Ride':>8} {'Slack':>8}")
        print("-" * 90)

        for timing in timings:
            node_id = timing.node_id
            if instance.is_depot(node_id):
                node_type = "depot"
            elif instance.is_pickup(node_id):
                node_type = f"P{instance.get_request_id(node_id)}"
            else:
                node_type = f"D{instance.get_request_id(node_id)}"

            ride_str = f"{timing.ride_time:.2f}" if timing.ride_time > 0 else "-"

            print(f"{node_id:>6} {node_type:>8} {timing.arrival:>10.2f} {timing.start_service:>10.2f} "
                  f"{timing.departure:>10.2f} {timing.wait_time:>8.2f} {int(timing.load_after):>6} "
                  f"{ride_str:>8} {timing.forward_slack:>8.2f}")
    else:
        stops = compute_schedule_simple(instance, route)
        feasible, reason = check_route_feasibility_simple(instance, route, return_reason=True)
        print(f"Route feasibility: {feasible} ({reason})")
        print()
        print(f"{'Node':>6} {'Type':>8} {'Arrival':>10} {'Service':>10} {'Depart':>10} {'Load':>6} {'Ride':>8}")
        print("-" * 70)

        for stop in stops:
            node_id = stop.node_id
            if instance.is_depot(node_id):
                node_type = "depot"
            elif instance.is_pickup(node_id):
                node_type = f"P{instance.get_request_id(node_id)}"
            else:
                node_type = f"D{instance.get_request_id(node_id)}"

            ride_str = f"{stop.ride_time:.2f}" if stop.ride_time else "-"

            print(f"{node_id:>6} {node_type:>8} {stop.arrival_time:>10.2f} {stop.start_service:>10.2f} "
                  f"{stop.departure_time:>10.2f} {int(stop.load_after):>6} {ride_str:>8}")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Feasibility module loaded.")
    print("Use: from src.feasibility import check_route_feasibility, check_insertion_feasibility")