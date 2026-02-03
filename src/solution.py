"""
DARP Solution Representation

This module defines the solution representation for DARP,
compatible with the N-Wouda ALNS library.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
from copy import deepcopy
import numpy as np

from .instance import DARPInstance


@dataclass
class RouteStop:
    """
    A stop in a vehicle route.

    Attributes:
        node_id: Index of the node being visited
        arrival_time: Time of arrival at the node
        start_service: Time when service starts (may wait if early)
        departure_time: Time of departure (start_service + service_time)
        load_after: Vehicle load after this stop
        ride_time: For delivery nodes, the ride time of the delivered passenger
    """
    node_id: int
    arrival_time: float = 0.0
    start_service: float = 0.0
    departure_time: float = 0.0
    load_after: int = 0
    ride_time: Optional[float] = None  # Only for delivery nodes


@dataclass
class Route:
    """
    A vehicle route.

    A route is a sequence of stops starting and ending at the depot.
    The route maintains timing and load information for feasibility checking.

    Attributes:
        vehicle_id: ID of the vehicle serving this route
        stops: Ordered list of stops (including depot start/end)
        requests_served: Set of request IDs served by this route
    """
    vehicle_id: int
    stops: List[RouteStop] = field(default_factory=list)
    requests_served: Set[int] = field(default_factory=set)

    @property
    def num_stops(self) -> int:
        """Number of stops (excluding depots)."""
        return max(0, len(self.stops) - 2)

    @property
    def is_empty(self) -> bool:
        """Check if route has no customer stops."""
        return self.num_stops == 0

    @property
    def duration(self) -> float:
        """Total route duration."""
        if len(self.stops) < 2:
            return 0.0
        return self.stops[-1].arrival_time - self.stops[0].departure_time

    @property
    def total_distance(self) -> float:
        """
        Note: This requires the instance to compute.
        Use DARPSolution.route_distance() instead.
        """
        raise NotImplementedError("Use DARPSolution.route_distance()")

    def copy(self) -> 'Route':
        """Create a deep copy of this route."""
        return Route(
            vehicle_id=self.vehicle_id,
            stops=[RouteStop(
                node_id=s.node_id,
                arrival_time=s.arrival_time,
                start_service=s.start_service,
                departure_time=s.departure_time,
                load_after=s.load_after,
                ride_time=s.ride_time
            ) for s in self.stops],
            requests_served=self.requests_served.copy()
        )

    def get_node_sequence(self) -> List[int]:
        """Get the sequence of node IDs in this route."""
        return [stop.node_id for stop in self.stops]

    def __str__(self) -> str:
        nodes = self.get_node_sequence()
        return f"Route(v={self.vehicle_id}, stops={nodes}, requests={self.requests_served})"


class DARPSolution:
    """
    A complete solution to a DARP instance.

    This class is designed to be compatible with the N-Wouda ALNS library,
    implementing the required interface for the State class.

    Attributes:
        instance: The DARP instance being solved
        routes: List of vehicle routes
        unassigned: Set of request IDs not yet assigned to any route
    """

    def __init__(self, instance: DARPInstance, routes: Optional[List[Route]] = None):
        """
        Initialize a DARP solution.

        Args:
            instance: The DARP instance
            routes: Optional list of routes (creates empty routes if None)
        """
        self.instance = instance
        self.unassigned: Set[int] = set()

        if routes is None:
            # Create empty routes for each vehicle
            self.routes = []
            for v in range(instance.num_vehicles):
                route = Route(vehicle_id=v)
                # Initialize with depot start and end
                depot_start = RouteStop(
                    node_id=0,
                    arrival_time=instance.nodes[0].earliest,
                    start_service=instance.nodes[0].earliest,
                    departure_time=instance.nodes[0].earliest,
                    load_after=0
                )
                depot_end = RouteStop(
                    node_id=instance.num_nodes - 1,
                    arrival_time=instance.nodes[0].earliest,
                    start_service=instance.nodes[0].earliest,
                    departure_time=instance.nodes[0].earliest,
                    load_after=0
                )
                route.stops = [depot_start, depot_end]
                self.routes.append(route)
            # All requests are initially unassigned
            self.unassigned = set(range(1, instance.num_requests + 1))
        else:
            self.routes = routes
            # Compute unassigned requests
            assigned = set()
            for route in self.routes:
                assigned.update(route.requests_served)
            self.unassigned = set(range(1, instance.num_requests + 1)) - assigned

    def objective(self) -> float:
        """
        Compute the objective value (total distance/cost).

        This is the main method required by ALNS.
        Lower is better.
        """
        total_cost = 0.0

        # Sum of route distances
        for route in self.routes:
            total_cost += self.route_distance(route)

        # Penalty for unassigned requests (large penalty)
        total_cost += len(self.unassigned) * 1000.0

        return total_cost

    def route_distance(self, route: Route) -> float:
        """Compute the total distance of a route."""
        if len(route.stops) < 2:
            return 0.0

        total = 0.0
        for i in range(len(route.stops) - 1):
            from_node = route.stops[i].node_id
            to_node = route.stops[i + 1].node_id
            total += self.instance.distance(from_node, to_node)

        return total

    def copy(self) -> 'DARPSolution':
        """
        Create a deep copy of this solution.
        Required by ALNS for the destroy-repair cycle.
        """
        new_routes = [route.copy() for route in self.routes]
        new_solution = DARPSolution(self.instance, new_routes)
        new_solution.unassigned = self.unassigned.copy()
        return new_solution

    def get_assigned_requests(self) -> Set[int]:
        """Get the set of all assigned request IDs."""
        assigned = set()
        for route in self.routes:
            assigned.update(route.requests_served)
        return assigned

    def find_request_route(self, request_id: int) -> Optional[int]:
        """
        Find which route serves a request.

        Returns:
            Route index or None if request is unassigned
        """
        for i, route in enumerate(self.routes):
            if request_id in route.requests_served:
                return i
        return None

    def remove_request(self, request_id: int) -> bool:
        """
        Remove a request from its current route.

        Args:
            request_id: ID of the request to remove

        Returns:
            True if request was removed, False if not found
        """
        route_idx = self.find_request_route(request_id)

        if route_idx is None:
            return False

        route = self.routes[route_idx]

        # Get pickup and delivery node IDs
        pickup_node = request_id
        delivery_node = self.instance.num_requests + request_id

        # Remove both nodes from the route
        route.stops = [s for s in route.stops
                       if s.node_id not in (pickup_node, delivery_node)]
        route.requests_served.remove(request_id)

        # Add to unassigned
        self.unassigned.add(request_id)

        return True

    def insert_request(
            self,
            request_id: int,
            route_idx: int,
            pickup_pos: int,
            delivery_pos: int
    ) -> bool:
        """
        Insert a request into a route at specified positions.

        Args:
            request_id: ID of the request to insert
            route_idx: Index of the route to insert into
            pickup_pos: Position for pickup in the CURRENT route
            delivery_pos: Position for delivery in the CURRENT route (must be > pickup_pos)

        Returns:
            True if insertion was successful

        Note:
            - Positions are indices into the current stops list
            - delivery_pos must be > pickup_pos
            - This does NOT check feasibility. Use feasibility module for that.

        Example:
            Route is [depot, A, B, depot_end] (indices 0,1,2,3)
            insert_request(req, route, pickup_pos=1, delivery_pos=3)
            Result: [depot, pickup, A, B, delivery, depot_end]

            For empty route [depot, depot_end] (indices 0,1):
            insert_request(req, route, pickup_pos=1, delivery_pos=2)
            Result: [depot, pickup, delivery, depot_end]
        """
        if request_id not in self.unassigned:
            return False

        if delivery_pos <= pickup_pos:
            raise ValueError("delivery_pos must be > pickup_pos")

        route = self.routes[route_idx]

        # Get node info
        pickup_node = request_id
        delivery_node = self.instance.num_requests + request_id

        # Create stops (timing will be updated by schedule computation)
        pickup_stop = RouteStop(node_id=pickup_node)
        delivery_stop = RouteStop(node_id=delivery_node)

        # Insert in correct order:
        # 1. Insert pickup at pickup_pos
        # 2. After pickup insertion, all indices >= pickup_pos have shifted by 1
        #    So to insert delivery at what was originally delivery_pos,
        #    we need to insert at delivery_pos (not +1, because the shift accounts for pickup)
        route.stops.insert(pickup_pos, pickup_stop)
        route.stops.insert(delivery_pos, delivery_stop)

        # Update bookkeeping
        route.requests_served.add(request_id)
        self.unassigned.remove(request_id)

        return True

    def get_route_nodes(self, route_idx: int) -> List[int]:
        """Get the node sequence for a route."""
        return self.routes[route_idx].get_node_sequence()

    def is_complete(self) -> bool:
        """Check if all requests are assigned."""
        return len(self.unassigned) == 0

    def __str__(self) -> str:
        lines = [f"DARPSolution(objective={self.objective():.2f})"]
        for i, route in enumerate(self.routes):
            if not route.is_empty:
                lines.append(f"  {route}")
        if self.unassigned:
            lines.append(f"  Unassigned: {self.unassigned}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a detailed summary of the solution."""
        lines = [
            f"=== DARP Solution for {self.instance.name} ===",
            f"Total cost: {self.objective():.2f}",
            f"Assigned requests: {len(self.get_assigned_requests())}/{self.instance.num_requests}",
            f"Unassigned: {len(self.unassigned)}",
            ""
        ]

        for i, route in enumerate(self.routes):
            dist = self.route_distance(route)
            lines.append(
                f"Route {i} (vehicle {route.vehicle_id}): "
                f"{route.num_stops} stops, "
                f"distance={dist:.2f}, "
                f"requests={route.requests_served}"
            )

            # Show node sequence
            nodes = route.get_node_sequence()
            lines.append(f"  Sequence: {' -> '.join(map(str, nodes))}")

        return "\n".join(lines)


def create_empty_solution(instance: DARPInstance) -> DARPSolution:
    """
    Create an empty solution with all requests unassigned.

    This is the starting point for construction heuristics.
    """
    return DARPSolution(instance)