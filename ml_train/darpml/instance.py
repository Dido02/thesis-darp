"""Load DARP instance files (Fiedler-converted and Cordeau formats)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class NodeInfo:
    id: int
    x: float
    y: float
    service_duration: float
    demand: int
    earliest: float
    latest: float


@dataclass
class DARPInstance:
    num_vehicles: int
    num_requests: int
    vehicle_capacity: int
    max_ride_time: float
    max_route_duration: float
    nodes: list[NodeInfo] = field(repr=False)
    travel_time: np.ndarray = field(repr=False)
    vehicle_start_nodes: list[int] = field(default_factory=list)
    per_request_ride_times: list[float] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def has_vehicle_starts(self) -> bool:
        return len(self.vehicle_start_nodes) > 0

    @property
    def route_start_offset(self) -> int:
        return 2 if self.has_vehicle_starts else 1

    @property
    def empty_route_size(self) -> int:
        return 3 if self.has_vehicle_starts else 2

    @property
    def dest_depot(self) -> int:
        return 2 * self.num_requests + 1

    def max_ride_time_for(self, request: int) -> float:
        """Per-request max ride time L. Falls back to global max_ride_time when
        per_request_ride_times is unset (Cordeau format). Mirrors C++
        DARPInstance::max_ride_time_for in include/instance.h."""
        if self.per_request_ride_times:
            return self.per_request_ride_times[request]
        return self.max_ride_time


# Module-level cache: resolved path -> DARPInstance
_INSTANCE_CACHE: dict[str, DARPInstance] = {}

MAX_NODES_FOR_FULL_LOAD = 10_000


def load_instance(path: str | Path) -> DARPInstance:
    """Load a DARP instance, auto-detecting format. Cached per absolute path."""
    p = Path(path).resolve()
    key = str(p)
    if key in _INSTANCE_CACHE:
        return _INSTANCE_CACHE[key]

    if not p.exists():
        raise FileNotFoundError(f"Instance file not found: {p}")

    fmt = _detect_format(p)
    if fmt == "fiedler":
        inst = _parse_fiedler(p)
    else:
        inst = _parse_cordeau(p)

    _INSTANCE_CACHE[key] = inst
    return inst


def _detect_format(path: Path) -> str:
    """Detect 'fiedler' vs 'cordeau' by scanning for section markers."""
    with open(path) as f:
        for line in f:
            if "TRAVEL_TIMES" in line or "VEHICLE_START" in line or "PER_REQUEST" in line:
                return "fiedler"
    return "cordeau"


def _parse_fiedler(path: Path) -> DARPInstance:
    """Parse Fiedler-converted format with TRAVEL_TIMES section.

    Mirrors cpp/src/parsers/fiedler.cpp exactly.
    """
    with open(path) as f:
        # Header: K n_nodes T Q L
        header = f.readline().split()
        K = int(header[0])
        n_nodes = int(header[1])
        T = float(header[2])
        Q = int(header[3])
        L = float(header[4])
        num_requests = n_nodes // 2

        # Read node lines until section marker
        nodes: list[NodeInfo] = []
        line = ""
        for line in f:
            if "PER_REQUEST" in line or "VEHICLE_START" in line or "TRAVEL_TIMES" in line:
                break
            parts = line.split()
            if len(parts) < 7:
                continue
            nodes.append(NodeInfo(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                service_duration=float(parts[3]),
                demand=int(parts[4]),
                earliest=float(parts[5]),
                latest=float(parts[6]),
            ))

        total = len(nodes)

        if total > MAX_NODES_FOR_FULL_LOAD:
            raise InstanceTooLargeError(
                f"Instance has {total} nodes (>{MAX_NODES_FOR_FULL_LOAD}). "
                f"Travel time matrix would be {total**2 * 8 / 1e9:.1f} GB. Skipping."
            )

        # Parse remaining sections
        per_request_rt: list[float] = []
        vehicle_start_nodes: list[int] = []
        travel_time = np.zeros((total, total), dtype=np.float64)

        # Process the line that broke the node loop, then continue
        while True:
            if "PER_REQUEST_RIDE_TIMES" in line:
                rt_line = f.readline()
                parts = rt_line.split()
                per_request_rt = [L]  # index 0 unused
                for i in range(num_requests):
                    per_request_rt.append(float(parts[i]))

            elif "VEHICLE_START_NODES" in line:
                vs_line = f.readline()
                vehicle_start_nodes = [int(x) for x in vs_line.split()]

            elif "TRAVEL_TIMES" in line:
                for i in range(total):
                    row_line = f.readline()
                    travel_time[i] = np.fromstring(row_line, sep=" ", dtype=np.float64)

            line = f.readline()
            if not line:
                break

    return DARPInstance(
        num_vehicles=K,
        num_requests=num_requests,
        vehicle_capacity=Q,
        max_ride_time=L,
        max_route_duration=T,
        nodes=nodes,
        travel_time=travel_time,
        vehicle_start_nodes=vehicle_start_nodes,
        per_request_ride_times=per_request_rt,
    )


def _parse_cordeau(path: Path) -> DARPInstance:
    """Parse Cordeau format, compute Euclidean distance matrix.

    Mirrors cpp/src/parsers/cordeau.cpp.
    """
    with open(path) as f:
        header = f.readline().split()
        K = int(header[0])
        n_nodes = int(header[1])
        T = float(header[2])
        Q = int(header[3])
        L = float(header[4])
        num_requests = n_nodes // 2

        nodes: list[NodeInfo] = []
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            nodes.append(NodeInfo(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                service_duration=float(parts[3]),
                demand=int(parts[4]),
                earliest=float(parts[5]),
                latest=float(parts[6]),
            ))

    # Add destination depot (2n+1) as copy of origin depot (0)
    origin = nodes[0]
    nodes.append(NodeInfo(
        id=2 * num_requests + 1,
        x=origin.x, y=origin.y,
        service_duration=0.0, demand=0,
        earliest=origin.earliest, latest=origin.latest,
    ))

    total = len(nodes)
    assert total == 2 * num_requests + 2, f"Expected {2*num_requests+2} nodes, got {total}"

    # Euclidean distance matrix. We use math.hypot per-pair instead of
    # np.sqrt(dx**2 + dy**2) because clang on arm64 fuses dx*dx + dy*dy into a
    # single FMA inside std::sqrt(), giving a different last bit than the
    # non-FMA numpy path. math.hypot(dx, dy) computes the correctly-rounded
    # Euclidean distance, which matches the C++ Cordeau parser's FMA-fused
    # result bit-for-bit. Fiedler instances load tt from a file so they're
    # unaffected; this matters only for Cordeau-derived features.
    travel_time = np.zeros((total, total), dtype=np.float64)
    for i in range(total):
        ix, iy = nodes[i].x, nodes[i].y
        for j in range(total):
            travel_time[i, j] = math.hypot(ix - nodes[j].x, iy - nodes[j].y)

    return DARPInstance(
        num_vehicles=K,
        num_requests=num_requests,
        vehicle_capacity=Q,
        max_ride_time=L,
        max_route_duration=T,
        nodes=nodes,
        travel_time=travel_time,
    )


class InstanceTooLargeError(Exception):
    """Raised when an instance is too large to load the full travel time matrix."""
