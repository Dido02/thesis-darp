#!/usr/bin/env python3
"""
Convert solver route schedule output to Fiedler solution JSON format.

Usage:
    python scripts/convert_solution_fiedler.py \
        --routes data/fiedler_converted/manhattan_01min_delay10_routes.txt \
        --instance data/fiedler/Instances/Manhattan/instances/start_18-00/duration_01_min/max_delay_10_min \
        --output data/fiedler_converted/manhattan_01min_delay10_solution.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml


def load_requests(instance_dir: Path) -> list[dict]:
    requests_file = instance_dir / "requests.csv"
    requests = []
    with open(requests_file) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            requests.append({
                'time_s': int(parts[0]) / 1000,
                'origin': int(parts[1]),
                'dest': int(parts[2]),
                'direct_tt': int(parts[3]),
            })
    return requests


def load_vehicles(instance_dir: Path) -> list[dict]:
    vehicles_file = instance_dir / "vehicles.csv"
    vehicles = []
    with open(vehicles_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts[0].isdigit():
                continue
            vehicles.append({
                'position': int(parts[0]),
                'capacity': int(parts[1]),
            })
    return vehicles


def load_config(instance_dir: Path) -> dict:
    config_file = instance_dir / "config.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


def parse_route_schedules(routes_path: Path):
    """Parse the C++ solver route schedule output."""
    with open(routes_path) as f:
        lines = f.readlines()

    header = {}
    routes = []
    current_route = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == "SOLUTION":
            continue
        elif line.startswith("objective "):
            header['objective'] = float(line.split()[1])
        elif line.startswith("travel_cost "):
            header['travel_cost'] = float(line.split()[1])
        elif line.startswith("num_routes "):
            header['num_routes'] = int(line.split()[1])
        elif line.startswith("num_requests "):
            header['num_requests'] = int(line.split()[1])
        elif line.startswith("unassigned "):
            header['unassigned'] = int(line.split()[1])
        elif line.startswith("ROUTE "):
            if current_route is not None:
                routes.append(current_route)
            current_route = {'index': int(line.split()[1]), 'stops': []}
        elif line.startswith("size "):
            current_route['size'] = int(line.split()[1])
        else:
            parts = line.split()
            if len(parts) == 5:
                current_route['stops'].append({
                    'node_id': int(parts[0]),
                    'arrival': float(parts[1]),
                    'begin_service': float(parts[2]),
                    'departure': float(parts[3]),
                    'load': int(parts[4]),
                })

    if current_route is not None:
        routes.append(current_route)

    return header, routes


def convert(routes_path: Path, instance_dir: Path, output_path: Path):
    requests = load_requests(instance_dir)
    vehicles = load_vehicles(instance_dir)
    config = load_config(instance_dir)
    header, routes = parse_route_schedules(routes_path)

    n = len(requests)
    K = len(vehicles)
    delta = config.get('max_prolongation', 600)
    capacity = config.get('vehicles', {}).get('vehicle_capacity', 4)

    # Build node ID → road network mapping
    # Internal: 0=depot, 1..n=pickups, n+1..2n=deliveries, 2n+1=dest depot, 2n+2..2n+1+K=vehicle starts
    internal_to_road = {}
    for i, r in enumerate(requests):
        internal_to_road[i + 1] = r['origin']
        internal_to_road[n + i + 1] = r['dest']
    for k, v in enumerate(vehicles):
        internal_to_road[2 * n + 2 + k] = v['position']

    # Build solution JSON
    total_cost = 0
    plans = []
    served_requests = set()

    for route_data in routes:
        k = route_data['index']
        stops = route_data['stops']

        # Skip empty routes (depot-vstart-depot or depot-depot)
        customer_stops = [s for s in stops
                          if 1 <= s['node_id'] <= 2 * n]
        if not customer_stops:
            continue

        # Build actions list
        actions = []
        for stop in stops:
            nid = stop['node_id']

            # Skip depot nodes and vehicle start nodes
            if nid == 0 or nid == 2 * n + 1:
                continue
            if nid >= 2 * n + 2:  # vehicle start node
                continue

            if 1 <= nid <= n:
                # Pickup
                req_idx = nid - 1  # 0-indexed request
                r = requests[req_idx]
                served_requests.add(req_idx)
                action_id = req_idx * 2
                actions.append({
                    "arrival_time": int(round(stop['arrival'])),
                    "departure_time": int(round(stop['departure'])),
                    "action": {
                        "id": action_id,
                        "request_index": req_idx,
                        "type": "pickup",
                        "position": {"index": r['origin']},
                        "min_time": int(r['time_s']),
                        "max_time": int(r['time_s'] + delta),
                        "service_duration": 0
                    }
                })
            elif n + 1 <= nid <= 2 * n:
                # Delivery
                req_idx = nid - n - 1  # 0-indexed request
                r = requests[req_idx]
                action_id = req_idx * 2 + 1
                actions.append({
                    "arrival_time": int(round(stop['arrival'])),
                    "departure_time": int(round(stop['departure'])),
                    "action": {
                        "id": action_id,
                        "request_index": req_idx,
                        "type": "drop_off",
                        "position": {"index": r['dest']},
                        "min_time": int(r['time_s'] + r['direct_tt']),
                        "max_time": int(r['time_s'] + r['direct_tt'] + delta),
                        "service_duration": 0
                    }
                })

        if not actions:
            continue

        # Compute plan cost: travel time from vehicle start through all actions
        # Find vehicle start stop (node 2n+2+k)
        vstart_stop = None
        for stop in stops:
            if stop['node_id'] == 2 * n + 2 + k:
                vstart_stop = stop
                break

        departure_time = int(round(vstart_stop['departure'])) if vstart_stop else int(round(actions[0]['arrival_time']))
        arrival_time = int(round(actions[-1]['departure_time']))
        plan_cost = arrival_time - departure_time

        # Subtract waiting times to get pure travel cost
        wait_time = 0
        for act in actions:
            wait_time += act['departure_time'] - act['arrival_time']
        plan_cost -= wait_time

        total_cost += plan_cost

        plans.append({
            "cost": plan_cost,
            "vehicle": {
                "index": k,
                "init_position": {"index": vehicles[k]['position']},
                "capacity": capacity
            },
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "actions": actions
        })

    # Dropped requests
    dropped = []
    for i in range(n):
        if i not in served_requests:
            r = requests[i]
            dropped.append({
                "index": i,
                "pickup": {
                    "id": i * 2,
                    "request_index": i,
                    "type": "pickup",
                    "position": {"index": r['origin']},
                    "min_time": int(r['time_s']),
                    "max_time": int(r['time_s'] + delta),
                    "service_duration": 0
                },
                "drop_off": {
                    "id": i * 2 + 1,
                    "request_index": i,
                    "type": "drop_off",
                    "position": {"index": r['dest']},
                    "min_time": int(r['time_s'] + r['direct_tt']),
                    "max_time": int(r['time_s'] + r['direct_tt'] + delta),
                    "service_duration": 0
                },
                "min_travel_time": r['direct_tt']
            })

    solution = {
        "cost": total_cost,
        "cost_minutes": math.ceil(total_cost / 60),
        "plans": plans,
        "dropped_requests": dropped
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(solution, f, indent=4)

    print(f"Solution written to {output_path}")
    print(f"  Cost: {total_cost}s ({math.ceil(total_cost / 60)} min)")
    print(f"  Active vehicles: {len(plans)}")
    print(f"  Served requests: {len(served_requests)}/{n}")
    print(f"  Dropped requests: {len(dropped)}")


def main():
    parser = argparse.ArgumentParser(description="Convert solver routes to Fiedler solution JSON")
    parser.add_argument("--routes", required=True, help="Path to solver route schedule file")
    parser.add_argument("--instance", required=True, help="Path to Fiedler instance directory")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    convert(Path(args.routes), Path(args.instance), Path(args.output))


if __name__ == "__main__":
    main()
