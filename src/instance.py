"""
DARP Instance Data Model

This module defines the core data structures for representing DARP instances.
Supports both Cordeau and Li & Lim formats through a unified representation.

Based on the formulation from Cordeau & Laporte (2003, 2007).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class Request:
    """
    A transportation request (pickup-delivery pair).

    Attributes:
        id: Unique request identifier (1-indexed as per Cordeau)
        pickup_node: Index of pickup node in the node list
        delivery_node: Index of delivery node in the node list
        load: Number of passengers (positive for pickup)
    """
    id: int
    pickup_node: int
    delivery_node: int
    load: int = 1


@dataclass
class Node:
    """
    A location in the DARP network.

    Node types:
        - Depot (start): index 0
        - Pickups: indices 1 to n
        - Deliveries: indices n+1 to 2n
        - Depot (end): index 2n+1

    Attributes:
        id: Node index
        x: X coordinate
        y: Y coordinate
        service_time: Time required to service this node
        load: Demand at node (+q for pickup, -q for delivery, 0 for depot)
        earliest: Earliest allowed start of service
        latest: Latest allowed start of service
        is_depot: True if this is a depot node
        is_pickup: True if this is a pickup node
        is_delivery: True if this is a delivery node
        paired_node: Index of the paired pickup/delivery node (None for depot)
    """
    id: int
    x: float
    y: float
    service_time: float
    load: int
    earliest: float
    latest: float
    is_depot: bool = False
    is_pickup: bool = False
    is_delivery: bool = False
    paired_node: Optional[int] = None


@dataclass
class Vehicle:
    """
    A vehicle in the DARP fleet.

    Attributes:
        id: Vehicle identifier
        capacity: Maximum number of passengers
        max_route_duration: Maximum allowed route duration (T)
        start_depot: Index of starting depot node
        end_depot: Index of ending depot node
    """
    id: int
    capacity: int
    max_route_duration: float
    start_depot: int = 0
    end_depot: Optional[int] = None  # Set to 2n+1 after instance is loaded


@dataclass
class DARPInstance:
    """
    Complete DARP instance.

    This is the unified data model that can be loaded from different formats
    (Cordeau, Li & Lim, NYC Taxi, etc.)

    Attributes:
        name: Instance name (e.g., "pr01", "a2-16")
        num_requests: Number of transportation requests (n)
        num_vehicles: Number of vehicles (K)
        nodes: List of all nodes [depot, pickups, deliveries, depot_end]
        vehicles: List of vehicles
        requests: List of requests
        max_ride_time: Maximum ride time for any user (L)
        distance_matrix: Precomputed distances between nodes
        travel_time_matrix: Precomputed travel times between nodes
    """
    name: str
    num_requests: int
    num_vehicles: int
    nodes: List[Node]
    vehicles: List[Vehicle]
    requests: List[Request]
    max_ride_time: float
    distance_matrix: Optional[np.ndarray] = None
    travel_time_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute distance and travel time matrices if not provided."""
        if self.distance_matrix is None:
            self._compute_distance_matrix()
        if self.travel_time_matrix is None:
            # For standard DARP, travel time equals distance
            self.travel_time_matrix = self.distance_matrix.copy()

    def _compute_distance_matrix(self):
        """Compute Euclidean distance matrix between all nodes."""
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.nodes[i].x - self.nodes[j].x
                    dy = self.nodes[i].y - self.nodes[j].y
                    self.distance_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (2n + 2 for DARP)."""
        return len(self.nodes)

    @property
    def depot_start(self) -> Node:
        """Starting depot node."""
        return self.nodes[0]

    @property
    def depot_end(self) -> Node:
        """Ending depot node."""
        return self.nodes[-1]

    @property
    def pickup_nodes(self) -> List[Node]:
        """All pickup nodes (indices 1 to n)."""
        return self.nodes[1:self.num_requests + 1]

    @property
    def delivery_nodes(self) -> List[Node]:
        """All delivery nodes (indices n+1 to 2n)."""
        return self.nodes[self.num_requests + 1:-1]

    def get_pickup_node(self, request_id: int) -> Node:
        """Get pickup node for a request (request_id is 1-indexed)."""
        return self.nodes[request_id]

    def get_delivery_node(self, request_id: int) -> Node:
        """Get delivery node for a request (request_id is 1-indexed)."""
        return self.nodes[self.num_requests + request_id]

    def get_request(self, request_id: int) -> Request:
        """Get request by ID (1-indexed)."""
        return self.requests[request_id - 1]

    def distance(self, i: int, j: int) -> float:
        """Get distance between nodes i and j."""
        return self.distance_matrix[i, j]

    def travel_time(self, i: int, j: int) -> float:
        """Get travel time between nodes i and j."""
        return self.travel_time_matrix[i, j]

    def direct_ride_time(self, request_id: int) -> float:
        """
        Get the direct travel time from pickup to delivery for a request.
        This is the minimum possible ride time for the request.
        """
        pickup = request_id
        delivery = self.num_requests + request_id
        return self.travel_time(pickup, delivery)

    def is_pickup(self, node_id: int) -> bool:
        """Check if node is a pickup node."""
        return 1 <= node_id <= self.num_requests

    def is_delivery(self, node_id: int) -> bool:
        """Check if node is a delivery node."""
        return self.num_requests + 1 <= node_id <= 2 * self.num_requests

    def is_depot(self, node_id: int) -> bool:
        """Check if node is a depot node."""
        return node_id == 0 or node_id == len(self.nodes) - 1

    def get_paired_node(self, node_id: int) -> int:
        """
        Get the paired pickup/delivery node.

        For pickup i: returns delivery n+i
        For delivery n+i: returns pickup i
        """
        if self.is_pickup(node_id):
            return node_id + self.num_requests
        elif self.is_delivery(node_id):
            return node_id - self.num_requests
        else:
            raise ValueError(f"Node {node_id} is a depot, has no paired node")

    def get_request_id(self, node_id: int) -> int:
        """Get the request ID for a pickup or delivery node."""
        if self.is_pickup(node_id):
            return node_id
        elif self.is_delivery(node_id):
            return node_id - self.num_requests
        else:
            raise ValueError(f"Node {node_id} is a depot, has no request ID")

    def __str__(self) -> str:
        return (
            f"DARPInstance(name={self.name}, "
            f"requests={self.num_requests}, "
            f"vehicles={self.num_vehicles}, "
            f"capacity={self.vehicles[0].capacity if self.vehicles else 'N/A'}, "
            f"max_ride_time={self.max_ride_time})"
        )

    def summary(self) -> str:
        """Return a detailed summary of the instance."""
        lines = [
            f"=== DARP Instance: {self.name} ===",
            f"Requests: {self.num_requests}",
            f"Vehicles: {self.num_vehicles}",
            f"Vehicle capacity: {self.vehicles[0].capacity if self.vehicles else 'N/A'}",
            f"Max route duration: {self.vehicles[0].max_route_duration if self.vehicles else 'N/A'}",
            f"Max ride time: {self.max_ride_time}",
            f"Total nodes: {self.num_nodes}",
            "",
            "Time window ranges:",
            f"  Depot: [{self.depot_start.earliest}, {self.depot_start.latest}]",
        ]

        # Sample a few requests
        for i in range(min(50, self.num_requests)):
            req = self.requests[i]
            p = self.nodes[req.pickup_node]
            d = self.nodes[req.delivery_node]
            lines.append(
                f"  Request {req.id}: "
                f"P[{p.earliest:.0f},{p.latest:.0f}] -> "
                f"D[{d.earliest:.0f},{d.latest:.0f}]"
            )

        # if self.num_requests > 3:
        #     lines.append(f"  ... and {self.num_requests - 3} more requests")

        return "\n".join(lines)