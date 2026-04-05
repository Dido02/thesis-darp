#!/usr/bin/env python3
"""Plot pickup and delivery locations for a Fiedler instance on a scatter plot."""

import pandas as pd
import matplotlib.pyplot as plt
import sys

instance_dir = sys.argv[1] if len(sys.argv) > 1 else \
    "data/fiedler/Instances/Manhattan/instances/start_18-00/duration_01_min/max_delay_05_min"

# Derive map dir from instance dir
# e.g. .../Manhattan/instances/... -> .../Manhattan/map/nodes.csv
parts = instance_dir.split("/")
city_idx = parts.index("instances") - 1 if "instances" in parts else None
if city_idx is None:
    # Try to find map relative to instance
    map_dir = "/".join(parts[:parts.index("Instances") + 2]) + "/map"
else:
    map_dir = "/".join(parts[:city_idx + 1]) + "/map"

nodes = pd.read_csv(f"{map_dir}/nodes.csv", sep="\t")
requests = pd.read_csv(f"{instance_dir}/requests.csv", sep="\t")

# Merge to get coordinates
origins = requests.merge(nodes, left_on="origin", right_on="id", how="left")
dests = requests.merge(nodes, left_on="dest", right_on="id", how="left")

fig, ax = plt.subplots(figsize=(10, 10))

# Plot all nodes lightly in background
ax.scatter(nodes["x"], nodes["y"], c="lightgray", s=1, alpha=0.3, label="road network")

# Plot origins and destinations
ax.scatter(origins["x"], origins["y"], c="green", s=20, alpha=0.7, label=f"pickups ({len(origins)})")
ax.scatter(dests["x"], dests["y"], c="red", s=20, alpha=0.7, label=f"deliveries ({len(dests)})")

# Draw lines from origin to destination
for _, row in requests.iterrows():
    o = nodes[nodes["id"] == row["origin"]].iloc[0]
    d = nodes[nodes["id"] == row["dest"]].iloc[0]
    ax.plot([o["x"], d["x"]], [o["y"], d["y"]], c="blue", alpha=0.15, linewidth=0.5)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Requests ({instance_dir.split('Instances/')[1]})")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("requests_map.png", dpi=150)
plt.show()
print(f"Saved to requests_map.png")
