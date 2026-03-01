"""Visualize a DARP instance: node locations + time windows."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau


def visualize_instance(filepath: str | Path, save_path: str | Path | None = None):
    """Plot node locations and time windows for a Cordeau DARP instance."""

    # We need coords for plotting, so re-parse raw file
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.read().strip().splitlines()

    header = lines[0].split()
    n = int(header[1])
    num_requests = n // 2

    # Parse coords and node data
    ids, xs, ys, demands, earliests, latests = [], [], [], [], [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        ids.append(int(parts[0]))
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))
        demands.append(int(parts[4]))
        earliests.append(float(parts[5]))
        latests.append(float(parts[6]))

    xs, ys = np.array(xs), np.array(ys)
    demands = np.array(demands)
    earliests, latests = np.array(earliests), np.array(latests)

    # Indices
    depot_idx = 0
    pickup_idx = list(range(1, num_requests + 1))
    delivery_idx = list(range(num_requests + 1, n + 1))

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    instance_name = filepath.stem

    # === Left: Node Locations ===
    # Draw pickup-delivery links
    for i in range(num_requests):
        pi, di = pickup_idx[i], delivery_idx[i]
        ax1.plot([xs[pi], xs[di]], [ys[pi], ys[di]],
                 color='#cccccc', linewidth=0.6, zorder=1)

    # Depot
    ax1.scatter(xs[depot_idx], ys[depot_idx],
                c='red', s=200, marker='s', zorder=5, edgecolors='darkred', linewidths=1.5)

    # Pickups
    ax1.scatter(xs[pickup_idx], ys[pickup_idx],
                c='#4169E1', s=60, marker='^', zorder=3, edgecolors='#2a4494', linewidths=0.5,
                label='Pickups')

    # Deliveries
    ax1.scatter(xs[delivery_idx], ys[delivery_idx],
                c='#2E8B57', s=60, marker='v', zorder=3, edgecolors='#1e5c3a', linewidths=0.5,
                label='Deliveries')

    # Legend
    depot_patch = mpatches.Patch(color='red', label='Depot')
    pickup_patch = mpatches.Patch(color='#4169E1', label='Pickups')
    delivery_patch = mpatches.Patch(color='#2E8B57', label='Deliveries')
    ax1.legend(handles=[depot_patch, pickup_patch, delivery_patch],
               loc='upper right', fontsize=10, framealpha=0.9)

    ax1.set_xlabel('X coordinate', fontsize=11)
    ax1.set_ylabel('Y coordinate', fontsize=11)
    ax1.set_title(f'{instance_name}: Node Locations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # === Right: Time Windows ===
    # Sort requests by pickup earliest time
    pickup_earliest = earliests[pickup_idx]
    sort_order = np.argsort(pickup_earliest)

    bar_height = 0.7
    for plot_pos, req_idx in enumerate(sort_order):
        pi = pickup_idx[req_idx]
        di = delivery_idx[req_idx]

        pe, pl = earliests[pi], latests[pi]
        de, dl = earliests[di], latests[di]

        # Pickup time window bar
        ax2.barh(plot_pos, pl - pe, left=pe, height=bar_height,
                 color='#6A8FD6', alpha=0.7, edgecolor='none')

        # Delivery time window bar
        ax2.barh(plot_pos, dl - de, left=de, height=bar_height,
                 color='#5CB87B', alpha=0.7, edgecolor='none')

    pickup_patch2 = mpatches.Patch(color='#6A8FD6', alpha=0.7, label='Pickup')
    delivery_patch2 = mpatches.Patch(color='#5CB87B', alpha=0.7, label='Delivery')
    ax2.legend(handles=[pickup_patch2, delivery_patch2],
               loc='upper right', fontsize=10, framealpha=0.9)

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Request (sorted by pickup time)', fontsize=11)
    ax2.set_title(f'{instance_name}: Time Windows', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(num_requests))
    ax2.set_yticklabels(range(num_requests), fontsize=8)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "cordeau"
    instance_file = data_dir / "pr01.txt"
    save_file = PROJECT_ROOT / "data" / "pr01_viz.png"
    visualize_instance(instance_file, save_path=save_file)