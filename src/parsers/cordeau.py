"""
Parser for Cordeau DARP instance format.

Cordeau format (from Cordeau & Laporte 2003):

Line 1: K n T Q L
    K = number of vehicles
    n = number of NODES (not requests!) = 2 * num_requests
    T = maximum route duration (per vehicle)
    Q = vehicle capacity
    L = maximum ride time (per user)

Lines 2 to n+2: node data (n+1 nodes: depot + n customer nodes)
    Each line: id x y service_time load earliest latest

    Node ordering in file:
    - Node 0: depot
    - Nodes 1 to n/2: pickup nodes (load > 0)
    - Nodes n/2+1 to n: delivery nodes (load < 0)

    Note: The file does NOT include an explicit end depot.
    We create a virtual end depot (copy of node 0) in our internal representation.

    Internal representation after parsing:
    - Node 0: depot (start)
    - Nodes 1 to num_requests: pickup nodes
    - Nodes num_requests+1 to 2*num_requests: delivery nodes
    - Node 2*num_requests+1: depot (end) - virtual copy of node 0

Example (pr01 - first few lines):
    3 48 480 6 90
    0  -1.044  2.000  0  0    0 1440
    1  -2.973  6.414 10  1    0 1440
    ...
    25 -5.476  1.437 10 -1  258  287   <- first delivery node
    ...
"""

from pathlib import Path
from typing import Union

from ..instance import DARPInstance, Node, Request, Vehicle


def parse_cordeau(filepath: Union[str, Path]) -> DARPInstance:
    """
    Parse a Cordeau format DARP instance file.

    Args:
        filepath: Path to the instance file

    Returns:
        DARPInstance object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Instance file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse first line: K n T Q L
    first_line = lines[0].strip().split()

    if len(first_line) < 5:
        raise ValueError(
            f"Invalid first line format. Expected 'K n T Q L', got: {lines[0]}"
        )

    num_vehicles = int(first_line[0])       # K
    n_nodes = int(first_line[1])            # n = 2 * num_requests (NOT num_requests!)
    max_route_duration = float(first_line[2])  # T
    vehicle_capacity = int(first_line[3])   # Q
    max_ride_time = float(first_line[4])    # L

    # n is the number of customer nodes (pickups + deliveries)
    # so num_requests = n / 2
    if n_nodes % 2 != 0:
        raise ValueError(f"n must be even (n = 2 * num_requests), got n={n_nodes}")

    num_requests = n_nodes // 2

    # File contains: 1 depot + n customer nodes = n + 1 lines of node data
    # We will create: depot + n customer nodes + virtual end depot = n + 2 nodes
    nodes_in_file = n_nodes + 1  # depot (0) + customer nodes (1 to n)

    # Parse node data from file
    nodes = []

    for i in range(nodes_in_file):
        line_idx = i + 1  # Skip first line (header)

        if line_idx >= len(lines):
            raise ValueError(
                f"Expected {nodes_in_file} node lines, but file ended at line {line_idx}"
            )

        line = lines[line_idx].strip()

        if not line:
            raise ValueError(f"Empty line at position {line_idx + 1}")

        # Parse: id x y service_time load earliest latest
        parts = line.split()

        if len(parts) < 7:
            raise ValueError(
                f"Invalid node line {line_idx + 1}. Expected 7 values, got {len(parts)}: {line}"
            )

        node_id_in_file = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        service_time = float(parts[3])
        load = int(parts[4])
        earliest = float(parts[5])
        latest = float(parts[6])

        # Determine node type based on position
        is_depot = (i == 0)
        is_pickup = (1 <= i <= num_requests)
        is_delivery = (num_requests + 1 <= i <= n_nodes)

        # Determine paired node
        paired_node = None
        if is_pickup:
            paired_node = i + num_requests  # Delivery is num_requests positions ahead
        elif is_delivery:
            paired_node = i - num_requests  # Pickup is num_requests positions behind

        node = Node(
            id=i,  # Use our sequential index (0 to n)
            x=x,
            y=y,
            service_time=service_time,
            load=load,
            earliest=earliest,
            latest=latest,
            is_depot=is_depot,
            is_pickup=is_pickup,
            is_delivery=is_delivery,
            paired_node=paired_node
        )
        nodes.append(node)

    # Create virtual end depot (copy of start depot with new id)
    start_depot = nodes[0]
    end_depot_id = n_nodes + 1  # = 2 * num_requests + 1

    end_depot = Node(
        id=end_depot_id,
        x=start_depot.x,
        y=start_depot.y,
        service_time=start_depot.service_time,
        load=0,
        earliest=start_depot.earliest,
        latest=start_depot.latest,
        is_depot=True,
        is_pickup=False,
        is_delivery=False,
        paired_node=None
    )
    nodes.append(end_depot)

    # Create vehicles (homogeneous fleet)
    vehicles = []
    for k in range(num_vehicles):
        vehicle = Vehicle(
            id=k,
            capacity=vehicle_capacity,
            max_route_duration=max_route_duration,
            start_depot=0,
            end_depot=end_depot_id
        )
        vehicles.append(vehicle)

    # Create requests
    requests = []
    for i in range(1, num_requests + 1):
        pickup_node = i
        delivery_node = i + num_requests
        load = abs(nodes[pickup_node].load)  # Use absolute value (pickup has +, delivery has -)

        request = Request(
            id=i,
            pickup_node=pickup_node,
            delivery_node=delivery_node,
            load=load
        )
        requests.append(request)

    # Extract instance name from filename
    name = filepath.stem

    # Create instance
    instance = DARPInstance(
        name=name,
        num_requests=num_requests,
        num_vehicles=num_vehicles,
        nodes=nodes,
        vehicles=vehicles,
        requests=requests,
        max_ride_time=max_ride_time
    )

    return instance


def download_cordeau_instance(instance_name: str, output_dir: Union[str, Path] = "data/cordeau") -> Path:
    """
    Download a Cordeau instance from the official repository.

    Args:
        instance_name: Name of instance (e.g., "pr01", "a2-16")
        output_dir: Directory to save the file

    Returns:
        Path to downloaded file
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine URL based on instance name
    if instance_name.startswith("pr"):
        # Tabu search instances (pr01-pr20)
        url = f"http://neumann.hec.ca/chairedistributique/data/darp/tabu/{instance_name}"
    elif instance_name.startswith("a") or instance_name.startswith("b"):
        # Branch-and-cut instances (a2-16, b2-16, etc.)
        url = f"http://neumann.hec.ca/chairedistributique/data/darp/branch-and-cut/{instance_name}"
    else:
        raise ValueError(f"Unknown instance type: {instance_name}")

    output_path = output_dir / f"{instance_name}.txt"

    if output_path.exists():
        print(f"Instance already exists: {output_path}")
        return output_path

    print(f"Downloading {instance_name} from {url}...")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {instance_name}: {e}")


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        instance = parse_cordeau(filepath)
        print(instance.summary())
    else:
        print("Usage: python -m src.parsers.cordeau <instance_file>")
        print("\nExample: python -m src.parsers.cordeau data/cordeau/pr01.txt")