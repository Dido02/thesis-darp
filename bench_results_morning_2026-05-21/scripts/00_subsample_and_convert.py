#!/usr/bin/env python3
"""Subsample 07-00 (16h) Manhattan source into duration-matched instances,
then convert to extended-Cordeau via scripts/convert_fiedler.py.

Sizes mirror the existing 18-00 grid (30_s, 01_min, 02_min, 05_min, 15_min).
For each (size, delay) combination we:
  1. Read 07-00 / duration_16_h / max_delay_<delay>_min/{requests, vehicles, config}.
  2. Filter requests to the first <duration_seconds> seconds.
  3. Take the first N vehicles from 07-00's fleet, where N matches the existing
     18-00 vehicle count for the same duration (keeps the bench cells directly
     comparable on a per-cell basis: same fleet size, different request set).
  4. Materialise the subsampled instance under a fresh subdirectory:
     data/fiedler/Instances/Manhattan/instances/start_07-00/duration_<dur>/max_delay_<delay>_min/
  5. Run scripts/convert_fiedler.py on it.
  6. Output file:
     data/fiedler_converted/manhattan_morning_<size>_delay<delay>_min.txt

Naming convention chosen to avoid collision with existing
manhattan_<size>_delay<delay>_min.txt (the 18-00 evening instances).

Existing fiedler_converted/ files are NOT touched.
"""
from __future__ import annotations
import shutil
import subprocess
import yaml
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "data" / "fiedler" / "Instances" / "Manhattan"
SOURCE_07   = SOURCE_ROOT / "instances" / "start_07-00" / "duration_16_h"
DM_PATH     = SOURCE_ROOT / "dm.h5"
OUT_DIR     = REPO / "data" / "fiedler_converted"
CONVERTER   = REPO / "scripts" / "convert_fiedler.py"

# (size_label, duration_seconds, n_vehicles_to_match_evening)
SIZE_GRID = [
    ("30_s",   30,    124),
    ("01_min", 60,    200),
    ("02_min", 120,   352),
    ("05_min", 300,   781),
    ("15_min", 900,   1672),
]
DELAYS = ["03", "05", "10"]


_DM_CACHE = None
def get_dm():
    """Load the distance matrix once and cache."""
    global _DM_CACHE
    if _DM_CACHE is None:
        print(f"  Loading {DM_PATH.name} (cached for all subsequent conversions)...")
        with h5py.File(DM_PATH, "r") as f:
            key = list(f.keys())[0]
            _DM_CACHE = f[key][:].astype(np.float64)
        print(f"    DM shape: {_DM_CACHE.shape}")
    return _DM_CACHE


def subsample_one(size_label: str, duration_s: int, n_vehicles: int, delay: str):
    src = SOURCE_07 / f"max_delay_{delay}_min"
    if not src.exists():
        print(f"  [skip] source missing: {src}")
        return None

    dst_root = SOURCE_ROOT / "instances" / "start_07-00" / f"duration_{size_label}"
    dst = dst_root / f"max_delay_{delay}_min"
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Subsample requests.csv by time window. The source 07-00 file has
    #    3 columns (time_ms, origin, dest). The converter expects 4 columns
    #    (time_ms, origin, dest, min_travel_time). Compute the 4th from the
    #    distance matrix.
    dm = get_dm()
    requests_src = src / "requests.csv"
    requests_dst = dst / "requests.csv"
    kept = 0
    with open(requests_src) as f_in, open(requests_dst, "w") as f_out:
        header_in = f_in.readline().strip().split("\t")
        # Always emit the 4-column header matching 18-00 format
        f_out.write("time_ms\torigin\tdest\tmin_travel_time\n")
        first_time = None
        cutoff_ms = None
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            t = int(parts[0]); o = int(parts[1]); d = int(parts[2])
            if first_time is None:
                first_time = t
                cutoff_ms = first_time + duration_s * 1000
            if t >= cutoff_ms:
                break
            # 4th column: prefer existing value if present, else compute from dm
            if len(parts) >= 4 and parts[3].strip():
                mtt = parts[3]
            else:
                mtt = str(int(round(float(dm[o, d]))))
            f_out.write(f"{t}\t{o}\t{d}\t{mtt}\n")
            kept += 1

    # 2. Subset vehicles.csv (first N)
    vehicles_src = src / "vehicles.csv"
    vehicles_dst = dst / "vehicles.csv"
    with open(vehicles_src) as f_in, open(vehicles_dst, "w") as f_out:
        # Detect whether the first line is a header (non-numeric first field)
        first_line = f_in.readline()
        if first_line.strip() and not first_line.split("\t")[0].strip().lstrip("-").isdigit():
            f_out.write(first_line)
        else:
            f_in.seek(0)
        n_written = 0
        for line in f_in:
            if n_written >= n_vehicles:
                break
            f_out.write(line); n_written += 1

    # 3. Copy config.yaml verbatim (config doesn't encode duration explicitly)
    shutil.copy(src / "config.yaml", dst / "config.yaml")

    print(f"  {size_label:<7} d{delay}: kept {kept} requests, {n_written} vehicles -> {dst}")

    # 4. Run convert_fiedler.py on the new dir
    output_file = OUT_DIR / f"manhattan_morning_{size_label}_delay{delay}_min.txt"
    cmd = [
        "python3", str(CONVERTER),
        "--instance", str(dst),
        "--dm", str(DM_PATH),
        "--output", str(output_file),
    ]
    print(f"    converting -> {output_file.name}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    ERROR converting: {res.stderr.strip()[:300]}")
        return None
    return output_file


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Subsampling 07-00 instances. Existing files in {OUT_DIR} are not touched.")
    print()
    written = []
    for size_label, duration_s, n_vehicles in SIZE_GRID:
        print(f"== {size_label} ({duration_s} s, {n_vehicles} vehicles) ==")
        for delay in DELAYS:
            out = subsample_one(size_label, duration_s, n_vehicles, delay)
            if out and out.exists():
                written.append(out.name)
        print()

    print(f"Wrote {len(written)} converted instance files (all prefixed 'manhattan_morning_'):")
    for n in written:
        print(f"  {n}")
    if len(written) < len(SIZE_GRID) * len(DELAYS):
        print(f"\n[!] expected {len(SIZE_GRID) * len(DELAYS)} files, got {len(written)} — check output above for errors.")


if __name__ == "__main__":
    main()
