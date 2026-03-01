"""DARP instance data structures.

Designed to support both:
- Cordeau instances: Euclidean distances, depot-based, explicit time windows
- Fiedler/NYC instances: precomputed road-network travel times, no coordinates
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    """A node in the DARP graph."""

    id: int
    service_duration: float
    demand: int  # +1 pickup, -1 delivery, 0 depot
    earliest: float
    latest: float

    @property
    def is_pickup(self) -> bool:
        return self.demand > 0

    @property
    def is_delivery(self) -> bool:
        return self.demand < 0

    @property
    def is_depot(self) -> bool:
        return self.demand == 0


@dataclass
class DARPInstance:
    """A DARP instance.

    Node numbering (used for both Cordeau and Fiedler):
        0         = origin depot
        1..n      = pickups
        n+1..2n   = deliveries
        2n+1      = destination depot

    Request i has pickup node i, delivery node n+i (1-indexed).

    The travel_time matrix is the solver's primary input.
    Cordeau: travel_time == Euclidean distance (computed by parser).
    Fiedler: travel_time from precomputed road-network HDF5.
    """

    nodes: list[Node]
    num_vehicles: int
    num_requests: int
    vehicle_capacity: int
    max_ride_time: float
    max_route_duration: float
    travel_time: np.ndarray = field(repr=False)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def pickup_node(self, request: int) -> Node:
        """Get pickup node for request i (1-indexed)."""
        return self.nodes[request]

    def delivery_node(self, request: int) -> Node:
        """Get delivery node for request i (1-indexed)."""
        return self.nodes[self.num_requests + request]

    @property
    def origin_depot(self) -> Node:
        return self.nodes[0]

    @property
    def dest_depot(self) -> Node:
        return self.nodes[-1]