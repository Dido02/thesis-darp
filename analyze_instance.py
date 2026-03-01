"""
Cordeau DARP Instance Analyzer

Analyzes and visualizes DARP benchmark instances.

Usage:
    python analyze_instance.py data/cordeau/pr01.txt
    python analyze_instance.py data/cordeau/pr01.txt --plot
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.parsers import parse_cordeau, parse_li_lim


def analyze_instance(instance):
    """Compute statistics about a DARP instance."""

    print("=" * 60)
    print(f"INSTANCE: {instance.name}")
    print("=" * 60)

    # Basic info
    print("\n[1] BASIC PARAMETERS")
    print(f"    Requests:           {instance.num_requests}")
    print(f"    Vehicles:           {instance.num_vehicles}")
    print(f"    Vehicle capacity:   {instance.vehicles[0].capacity}")
    print(f"    Max route duration: {instance.vehicles[0].max_route_duration}")
    print(f"    Max ride time:      {instance.max_ride_time}")
    print(f"    Requests/vehicle:   {instance.num_requests / instance.num_vehicles:.1f}")

    # Node info
    print("\n[2] NODES")
    print(f"    Total nodes:        {instance.num_nodes}")
    print(f"    Depot:              node 0 and node {instance.num_nodes - 1}")
    print(f"    Pickups:            nodes 1 to {instance.num_requests}")
    print(f"    Deliveries:         nodes {instance.num_requests + 1} to {2 * instance.num_requests}")

    # Geographic spread
    x_coords = [n.x for n in instance.nodes]
    y_coords = [n.y for n in instance.nodes]

    print("\n[3] GEOGRAPHIC AREA")
    print(f"    X range:            [{min(x_coords):.2f}, {max(x_coords):.2f}]")
    print(f"    Y range:            [{min(y_coords):.2f}, {max(y_coords):.2f}]")
    print(f"    Depot location:     ({instance.nodes[0].x:.2f}, {instance.nodes[0].y:.2f})")

    # Distance analysis
    direct_distances = []
    for req in instance.requests:
        d = instance.distance(req.pickup_node, req.delivery_node)
        direct_distances.append(d)

    print("\n[4] DISTANCES (pickup to delivery)")
    print(f"    Min:                {min(direct_distances):.2f}")
    print(f"    Max:                {max(direct_distances):.2f}")
    print(f"    Average:            {np.mean(direct_distances):.2f}")

    # Time window analysis
    pickup_flexible = 0
    pickup_tight = 0
    delivery_flexible = 0
    delivery_tight = 0

    pickup_widths = []
    delivery_widths = []

    for req in instance.requests:
        p = instance.nodes[req.pickup_node]
        d = instance.nodes[req.delivery_node]

        p_width = p.latest - p.earliest
        d_width = d.latest - d.earliest

        pickup_widths.append(p_width)
        delivery_widths.append(d_width)

        if p_width > 1000:
            pickup_flexible += 1
        else:
            pickup_tight += 1

        if d_width > 1000:
            delivery_flexible += 1
        else:
            delivery_tight += 1

    print("\n[5] TIME WINDOWS")
    print(f"    Pickup flexible:    {pickup_flexible} (wide window [0,1440])")
    print(f"    Pickup tight:       {pickup_tight} (narrow window)")
    print(f"    Delivery flexible:  {delivery_flexible}")
    print(f"    Delivery tight:     {delivery_tight}")

    if pickup_tight > 0:
        tight_pickup_widths = [w for w in pickup_widths if w < 1000]
        print(f"    Tight pickup avg:   {np.mean(tight_pickup_widths):.1f} time units")

    if delivery_tight > 0:
        tight_delivery_widths = [w for w in delivery_widths if w < 1000]
        print(f"    Tight delivery avg: {np.mean(tight_delivery_widths):.1f} time units")

    # Service time
    service_times = set(n.service_time for n in instance.nodes if not n.is_depot)
    print("\n[6] SERVICE TIME")
    print(f"    Service time:       {list(service_times)[0]} per stop")

    # Constraint tightness
    print("\n[7] CONSTRAINT ANALYSIS")

    # Can a vehicle serve all requests alone?
    max_possible_per_vehicle = instance.vehicles[0].max_route_duration / (2 * list(service_times)[0] + np.mean(direct_distances) * 2)
    print(f"    Rough max requests/vehicle (time): ~{max_possible_per_vehicle:.0f}")

    # Ride time constraint tightness
    min_ride_time = min(direct_distances)  # Just travel, no detour
    ride_time_slack = instance.max_ride_time - min_ride_time
    print(f"    Min possible ride time: {min_ride_time:.2f}")
    print(f"    Max allowed ride time:  {instance.max_ride_time}")
    print(f"    Ride time slack:        {ride_time_slack:.2f}")

    # Requests summary
    print("\n[8] REQUEST DETAILS")
    print(f"    {'Req':>4} {'Pickup':>8} {'Delivery':>8} {'P_TW':>12} {'D_TW':>12} {'Dist':>8}")
    print("    " + "-" * 56)

    for req in instance.requests[:10]:  # Show first 10
        p = instance.nodes[req.pickup_node]
        d = instance.nodes[req.delivery_node]
        dist = instance.distance(req.pickup_node, req.delivery_node)

        p_tw = f"[{p.earliest:.0f},{p.latest:.0f}]"
        d_tw = f"[{d.earliest:.0f},{d.latest:.0f}]"

        print(f"    {req.id:>4} {req.pickup_node:>8} {req.delivery_node:>8} {p_tw:>12} {d_tw:>12} {dist:>8.2f}")

    if instance.num_requests > 10:
        print(f"    ... and {instance.num_requests - 10} more requests")

    print()
    return instance


def plot_instance(instance):
    """Create visualization of the instance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Node locations
    ax1 = axes[0]

    # Depot
    depot = instance.nodes[0]
    ax1.scatter([depot.x], [depot.y], c='red', s=200, marker='s', label='Depot', zorder=5)

    # Pickups
    pickup_x = [instance.nodes[i].x for i in range(1, instance.num_requests + 1)]
    pickup_y = [instance.nodes[i].y for i in range(1, instance.num_requests + 1)]
    ax1.scatter(pickup_x, pickup_y, c='blue', s=50, marker='^', label='Pickups', alpha=0.7)

    # Deliveries
    delivery_x = [instance.nodes[i].x for i in range(instance.num_requests + 1, 2 * instance.num_requests + 1)]
    delivery_y = [instance.nodes[i].y for i in range(instance.num_requests + 1, 2 * instance.num_requests + 1)]
    ax1.scatter(delivery_x, delivery_y, c='green', s=50, marker='v', label='Deliveries', alpha=0.7)

    # Draw lines connecting pickup-delivery pairs
    for req in instance.requests:
        p = instance.nodes[req.pickup_node]
        d = instance.nodes[req.delivery_node]
        ax1.plot([p.x, d.x], [p.y, d.y], 'gray', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title(f'{instance.name}: Node Locations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time windows - show pickup and delivery pairs connected
    ax2 = axes[1]

    # Sort requests by pickup earliest time (more intuitive)
    sorted_reqs = sorted(instance.requests, key=lambda r: instance.nodes[r.pickup_node].earliest)

    # Find time range for proper scaling
    all_times = []
    for req in instance.requests:
        p = instance.nodes[req.pickup_node]
        d = instance.nodes[req.delivery_node]
        all_times.extend([p.earliest, p.latest, d.earliest, d.latest])

    max_time = max(all_times)

    for i, req in enumerate(sorted_reqs):
        p = instance.nodes[req.pickup_node]
        d = instance.nodes[req.delivery_node]

        # Pickup time window (blue)
        p_width = p.latest - p.earliest
        ax2.barh(i, p_width, left=p.earliest, height=0.6, color='blue', alpha=0.6, label='Pickup' if i == 0 else '')

        # Delivery time window (green)
        d_width = d.latest - d.earliest
        ax2.barh(i, d_width, left=d.earliest, height=0.6, color='green', alpha=0.6, label='Delivery' if i == 0 else '')

        # Connect pickup end to delivery start with a thin line
        ax2.plot([p.latest, d.earliest], [i, i], 'gray', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Request (sorted by pickup time)')
    ax2.set_title(f'{instance.name}: Time Windows')
    ax2.set_xlim(0, max_time * 1.05)  # Add 5% margin
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # Save plot
    output_path = f"{instance.name}_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze DARP instance")
    parser.add_argument("instance", help="Path to instance file")
    parser.add_argument("--format", type=str, choices=["cordeau", "li_lim"],
                        default="cordeau", help="Instance format (default: cordeau)")
    parser.add_argument("--plot", action="store_true", help="Generate visualization")

    args = parser.parse_args()

    # Parse based on format
    if args.format == "cordeau":
        instance = parse_cordeau(args.instance)
    else:
        instance = parse_li_lim(args.instance)

    analyze_instance(instance)

    if args.plot:
        plot_instance(instance)


if __name__ == "__main__":
    main()