"""Parser for Cordeau DARP instances (tabu/ directory format).

Source: http://neumann.hec.ca/chairedistributique/data/darp/tabu/
Note: the general readme says format does NOT apply to darp/ directory.
DARP instances use their own format:

    Line 1: K n T Q L
        K = num vehicles
        n = num customer nodes (pickups + deliveries, so num_requests = n/2)
        T = max route duration
        Q = vehicle capacity
        L = max ride time
    Lines 2..n+1: i x y d q e l
        i = node id
        x, y = coordinates (used only to compute Euclidean distance matrix)
        d = service duration
        q = demand (+1 pickup, -1 delivery, 0 depot)
        e = earliest start of service
        l = latest start of service
    Node 0 = depot (first data line)
    Nodes 1..n/2 = pickups
    Nodes n/2+1..n = deliveries
    Destination depot (node n+1) is implicit, same location as node 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..instance import DARPInstance, Node


def _build_euclidean_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """Compute Euclidean distance matrix from coordinate pairs."""
    c = np.array(coords)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def parse_cordeau(filepath: str | Path) -> DARPInstance:
    """Parse a Cordeau-format DARP instance file."""
    filepath = Path(filepath)

    with open(filepath) as f:
        lines = f.read().strip().splitlines()

    # Header: K n T Q L
    # n = total non-depot nodes (pickups + deliveries), so num_requests = n // 2
    header = lines[0].split()
    num_vehicles = int(header[0])
    n = int(header[1])  # total non-depot nodes
    max_route_duration = float(header[2])
    vehicle_capacity = int(header[3])
    max_ride_time = float(header[4])
    num_requests = n // 2

    # Parse nodes and coordinates
    nodes = []
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        node = Node(
            id=int(parts[0]),
            service_duration=float(parts[3]),
            demand=int(parts[4]),
            earliest=float(parts[5]),
            latest=float(parts[6]),
        )
        nodes.append(node)
        coords.append((float(parts[1]), float(parts[2])))

    # Add destination depot (2n+1) as copy of origin depot (0)
    origin = nodes[0]
    dest_depot = Node(
        id=2 * num_requests + 1,
        service_duration=0,
        demand=0,
        earliest=origin.earliest,
        latest=origin.latest,
    )
    nodes.append(dest_depot)
    coords.append(coords[0])  # same location as origin depot

    # Sanity checks
    expected = 2 * num_requests + 2  # depot_o + n pickups + n deliveries + depot_d
    assert len(nodes) == expected, (
        f"Expected {expected} nodes, got {len(nodes)}"
    )

    # For Cordeau, travel_time == Euclidean distance
    travel_time = _build_euclidean_matrix(coords)

    return DARPInstance(
        nodes=nodes,
        num_vehicles=num_vehicles,
        num_requests=num_requests,
        vehicle_capacity=vehicle_capacity,
        max_ride_time=max_ride_time,
        max_route_duration=max_route_duration,
        travel_time=travel_time,
    )