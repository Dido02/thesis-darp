"""Tests for instance.py and parsers/cordeau.py"""

import sys
from pathlib import Path

# Project root = thesis-darp/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.instance import DARPInstance, Node
from src.parsers.cordeau import parse_cordeau

DATA_DIR = PROJECT_ROOT / "data" / "cordeau"


def test_parse_pr01():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")

    # Header values
    assert inst.num_vehicles == 3, f"Expected 3 vehicles, got {inst.num_vehicles}"
    assert inst.num_requests == 24, f"Expected 24 requests, got {inst.num_requests}"
    assert inst.max_route_duration == 480
    assert inst.vehicle_capacity == 6
    assert inst.max_ride_time == 90

    # Node count: 0 + 24 pickups + 24 deliveries + dest_depot = 50
    assert inst.num_nodes == 50, f"Expected 50 nodes, got {inst.num_nodes}"

    # Depot checks
    assert inst.origin_depot.id == 0
    assert inst.origin_depot.is_depot
    assert inst.dest_depot.id == 49  # 2*24+1
    assert inst.dest_depot.is_depot

    # Pickup/delivery pairing
    for i in range(1, 25):
        p = inst.pickup_node(i)
        d = inst.delivery_node(i)
        assert p.is_pickup, f"Node {p.id} should be pickup"
        assert d.is_delivery, f"Node {d.id} should be delivery"
        assert p.id == i
        assert d.id == 24 + i

    # Spot check node 1 (first pickup)
    n1 = inst.nodes[1]
    assert n1.id == 1
    assert n1.service_duration == 10
    assert n1.demand == 1
    assert n1.earliest == 0
    assert n1.latest == 1440

    # Spot check node 13 (pickup with tight time window)
    n13 = inst.nodes[13]
    assert n13.earliest == 325
    assert n13.latest == 358
    assert n13.demand == 1

    # Spot check node 25 (first delivery)
    n25 = inst.nodes[25]
    assert n25.demand == -1
    assert n25.earliest == 258
    assert n25.latest == 287

    print("  [PASS] parse_pr01")


def test_distance_matrix():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")

    # Matrix shape
    assert inst.travel_time.shape == (50, 50)

    # Diagonal is zero
    for i in range(50):
        assert inst.travel_time[i, i] == 0.0

    # Symmetric
    for i in range(50):
        for j in range(i + 1, 50):
            assert abs(inst.travel_time[i, j] - inst.travel_time[j, i]) < 1e-10

    # All non-negative
    assert (inst.travel_time >= 0).all()

    # Depot to dest_depot should be 0 (same location)
    assert inst.travel_time[0, 49] == 0.0

    # Triangle inequality spot check
    assert inst.travel_time[1, 3] <= inst.travel_time[1, 2] + inst.travel_time[2, 3] + 1e-10

    print("  [PASS] distance_matrix")


def test_node_types():
    inst = parse_cordeau(DATA_DIR / "pr01.txt")

    # Node 0: depot
    assert inst.nodes[0].is_depot and not inst.nodes[0].is_pickup
    # Nodes 1-24: pickups
    for i in range(1, 25):
        assert inst.nodes[i].is_pickup
    # Nodes 25-48: deliveries
    for i in range(25, 49):
        assert inst.nodes[i].is_delivery
    # Node 49: dest depot
    assert inst.nodes[49].is_depot

    print("  [PASS] node_types")


if __name__ == "__main__":
    print("Testing instance + cordeau parser:")
    test_parse_pr01()
    test_distance_matrix()
    test_node_types()
    print("All tests passed.")