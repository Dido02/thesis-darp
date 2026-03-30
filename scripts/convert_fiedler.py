#!/usr/bin/env python3
"""
Convert Fiedler & Mrkos ridesharing DARP instances to extended Cordeau format.

Usage:
    python scripts/convert_fiedler.py \
        --instance data/fiedler/Instances/Manhattan/instances/start_18-00/duration_01_min/max_delay_10_min \
        --dm data/fiedler/Instances/Manhattan/dm.h5 \
        --output data/fiedler_converted/manhattan_01min_delay10.txt

The output file uses an extended Cordeau format with a TRAVEL_TIMES section
appended after the node data, since coordinates are meaningless for road
network-based travel times.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def load_requests(instance_dir: Path) -> list[dict]:
    """Load requests from CSV. Returns list of {time_s, origin, dest, direct_tt}."""
    requests_file = instance_dir / "requests.csv"
    requests = []
    with open(requests_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            requests.append({
                'time_s': int(parts[0]) / 1000,  # ms -> seconds
                'origin': int(parts[1]),
                'dest': int(parts[2]),
                'direct_tt': int(parts[3]),
            })
    return requests


def load_vehicles(instance_dir: Path) -> list[dict]:
    """Load vehicles from CSV. Returns list of {position, capacity}."""
    vehicles_file = instance_dir / "vehicles.csv"
    vehicles = []
    with open(vehicles_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts[0].isdigit():
                continue  # skip header if present
            vehicles.append({
                'position': int(parts[0]),
                'capacity': int(parts[1]),
            })
    return vehicles


def load_config(instance_dir: Path) -> dict:
    """Load config.yaml and extract key parameters."""
    config_file = instance_dir / "config.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def load_distance_matrix(dm_path: Path) -> np.ndarray:
    """Load the HDF5 distance matrix."""
    with h5py.File(dm_path, 'r') as f:
        key = list(f.keys())[0]
        dm = f[key][:]
    return dm.astype(np.float64)


def convert(instance_dir: Path, dm_path: Path, output_path: Path):
    """Convert a Fiedler instance to extended Cordeau format."""
    requests = load_requests(instance_dir)
    vehicles = load_vehicles(instance_dir)
    config = load_config(instance_dir)

    n = len(requests)  # number of requests
    K = len(vehicles)
    Q = config.get('vehicles', {}).get('vehicle_capacity', 4)
    delta = config.get('max_prolongation', 600)  # seconds

    print(f"Instance: {instance_dir.name}")
    print(f"  Requests: {n}, Vehicles: {K}, Capacity: {Q}, Delta: {delta}s")

    # Load distance matrix
    print(f"  Loading distance matrix from {dm_path}...")
    dm = load_distance_matrix(dm_path)
    print(f"  DM shape: {dm.shape}")

    # Collect all unique road network node IDs used in this instance
    road_nodes = set()
    for r in requests:
        road_nodes.add(r['origin'])
        road_nodes.add(r['dest'])
    for v in vehicles:
        road_nodes.add(v['position'])
    road_nodes = sorted(road_nodes)
    print(f"  Unique road network nodes used: {len(road_nodes)}")

    # Build mapping: road_node_id -> index in our submatrix (for internal use)
    road_to_sub = {node: idx for idx, node in enumerate(road_nodes)}

    # Extract submatrix from full distance matrix
    indices = np.array(road_nodes)
    sub_dm = dm[np.ix_(indices, indices)]

    # Node numbering:
    # 0 = virtual origin depot
    # 1..n = pickups
    # n+1..2n = deliveries
    # 2n+1 = virtual destination depot
    # 2n+2..2n+1+K = vehicle start positions
    total_nodes = 2 * n + 2 + K

    # Map each internal node to its road network node
    internal_to_road = {}
    for i, r in enumerate(requests):
        internal_to_road[i + 1] = r['origin']        # pickup i+1
        internal_to_road[n + i + 1] = r['dest']      # delivery n+i+1
    for k, v in enumerate(vehicles):
        internal_to_road[2 * n + 2 + k] = v['position']  # vehicle start k

    # Build the travel time matrix
    tt = np.zeros((total_nodes, total_nodes), dtype=np.float64)

    depot_origin = 0
    depot_dest = 2 * n + 1
    vstart_first = 2 * n + 2
    vstart_last = 2 * n + 1 + K

    for i in range(total_nodes):
        for j in range(total_nodes):
            if i == depot_origin:
                # From virtual origin depot: zero to vehicle starts, large to others
                if vstart_first <= j <= vstart_last:
                    tt[i][j] = 0.0
                elif j == depot_dest:
                    tt[i][j] = 0.0
                else:
                    tt[i][j] = 999999.0
            elif j == depot_dest:
                # To virtual dest depot: always zero (no return trip)
                tt[i][j] = 0.0
            elif j == depot_origin or i == depot_dest:
                # To origin depot or from dest depot: shouldn't happen
                tt[i][j] = 999999.0
            else:
                # Real travel time between any two real nodes
                ri = internal_to_road[i]
                rj = internal_to_road[j]
                si = road_to_sub[ri]
                sj = road_to_sub[rj]
                tt[i][j] = sub_dm[si][sj]

    # Compute time windows and other node data
    global_L = max(r['direct_tt'] + delta for r in requests)

    min_time = min(r['time_s'] for r in requests)
    max_time = max(r['time_s'] for r in requests)
    T = (max_time - min_time) + global_L + delta + 3600  # generous buffer

    depot_earliest = min_time
    depot_latest = max_time + global_L + delta + 3600

    print(f"  Time range: {min_time:.0f}s - {max_time:.0f}s")
    print(f"  Global max ride time (L): {global_L:.0f}s")
    print(f"  Max route duration (T): {T:.0f}s")
    print(f"  Vehicle start nodes: {2*n+2}..{2*n+1+K}")

    # Write output in extended Cordeau format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Header: K  2n  T  Q  L
        f.write(f"{K} {2*n} {T:.1f} {Q} {global_L:.1f}\n")

        # Node 0: origin depot
        f.write(f"  0  0.000  0.000  0  0  {depot_earliest:.0f}  {depot_latest:.0f}\n")

        # Nodes 1..n: pickups
        for i, r in enumerate(requests):
            node_id = i + 1
            e = r['time_s']
            l = r['time_s'] + delta
            f.write(f"  {node_id}  0.000  0.000  0  1  {e:.0f}  {l:.0f}\n")

        # Nodes n+1..2n: deliveries
        for i, r in enumerate(requests):
            node_id = n + i + 1
            e = r['time_s'] + r['direct_tt']
            l = r['time_s'] + r['direct_tt'] + delta
            f.write(f"  {node_id}  0.000  0.000  0  -1  {e:.0f}  {l:.0f}\n")

        # Node 2n+1: destination depot
        f.write(f"  {2*n+1}  0.000  0.000  0  0  {depot_earliest:.0f}  {depot_latest:.0f}\n")

        # Nodes 2n+2..2n+1+K: vehicle start positions
        for k in range(K):
            node_id = 2 * n + 2 + k
            f.write(f"  {node_id}  0.000  0.000  0  0  {depot_earliest:.0f}  {depot_latest:.0f}\n")

        # Per-request max ride times (request 1..n)
        f.write("PER_REQUEST_RIDE_TIMES\n")
        ride_times = [r['direct_tt'] + delta for r in requests]
        f.write(" ".join(f"{rt:.1f}" for rt in ride_times) + "\n")

        # Vehicle start node IDs
        f.write("VEHICLE_START_NODES\n")
        vstart_ids = [2 * n + 2 + k for k in range(K)]
        f.write(" ".join(str(v) for v in vstart_ids) + "\n")

        # Travel time matrix
        f.write("TRAVEL_TIMES\n")
        for i in range(total_nodes):
            row = " ".join(f"{tt[i][j]:.1f}" for j in range(total_nodes))
            f.write(row + "\n")

    print(f"  Written to {output_path}")
    print(f"  Total nodes: {total_nodes} (2*{n}+2+{K})")

    # Verify against CSV direct travel times
    mismatches = 0
    for i, r in enumerate(requests):
        pickup_id = i + 1
        delivery_id = n + i + 1
        expected = r['direct_tt']
        actual = tt[pickup_id][delivery_id]
        if abs(expected - actual) > 0.5:
            mismatches += 1
            if mismatches <= 5:
                print(f"  WARNING: Request {i} direct_tt mismatch: CSV={expected}, DM={actual}")
    if mismatches > 0:
        print(f"  Total direct_tt mismatches: {mismatches}/{n}")
    else:
        print(f"  All {n} direct travel times verified against distance matrix.")


def main():
    parser = argparse.ArgumentParser(description="Convert Fiedler DARP instances to extended Cordeau format")
    parser.add_argument("--instance", required=True, help="Path to Fiedler instance directory")
    parser.add_argument("--dm", required=True, help="Path to HDF5 distance matrix (dm.h5)")
    parser.add_argument("--output", required=True, help="Output file path (.txt)")
    args = parser.parse_args()

    instance_dir = Path(args.instance)
    dm_path = Path(args.dm)
    output_path = Path(args.output)

    if not instance_dir.exists():
        print(f"Error: Instance directory not found: {instance_dir}", file=sys.stderr)
        sys.exit(1)
    if not dm_path.exists():
        print(f"Error: Distance matrix not found: {dm_path}", file=sys.stderr)
        sys.exit(1)

    convert(instance_dir, dm_path, output_path)


if __name__ == "__main__":
    main()
