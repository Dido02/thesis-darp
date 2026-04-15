"""Extract ML training features from ALNS trial logs.

For each trial record, produces per-combo features from the operator landscape.
Works with trials-only logs (no state records needed).

Filters out uninformative iterations where all combos return zero delta
(e.g., early epochs with few assigned requests).

Output: CSV with features and labels.
"""
import sys
import csv
import numpy as np
from collections import defaultdict
from parse_logs import parse_log
from output_utils import get_output_path


def main(log_path: str, min_nonzero_combos: int = 3):
    """Extract features.

    Args:
        log_path: Path to JSONL log.
        min_nonzero_combos: Skip iterations where fewer than this many combos
                            have a non-zero best delta. Filters out boring states
                            where most routes are empty.
    """
    data = parse_log(log_path)
    trials = data["trials"]
    config = data["config"]
    d_names = config.destroy_ops
    r_names = config.repair_ops
    nd, nr = len(d_names), len(r_names)

    if not trials:
        print("No trial records found.")
        return

    # Rolling stats window
    WINDOW = 10
    rolling = defaultdict(lambda: [])

    out_path = get_output_path(log_path, "_features.csv", "summary")
    n_written = 0
    n_skipped = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        cols = [
            "iter", "obj_before", "temperature",
            "n_active_routes", "n_unassigned", "total_stops",
            "max_route_len", "n_vehicles", "utilization",
        ]
        # Per-combo trial stats
        for di in range(nd):
            for ri in range(nr):
                tag = f"{d_names[di]}_{r_names[ri]}"
                cols.extend([
                    f"mean_{tag}", f"min_{tag}", f"median_{tag}",
                    f"zero_frac_{tag}", f"rolling_min_{tag}",
                ])
        # Labels
        cols.extend(["best_d_idx", "best_r_idx", "best_min_delta"])
        writer.writerow(cols)

        for tr in trials:
            # First pass: check if this iteration is informative
            nonzero_combos = 0
            combo_mins = {}
            for di in range(nd):
                for ri in range(nr):
                    deltas = tr.deltas.get((di, ri), [0])
                    mi = min(deltas)
                    combo_mins[(di, ri)] = mi
                    if mi != 0:
                        nonzero_combos += 1

            if nonzero_combos < min_nonzero_combos:
                n_skipped += 1
                # Still update rolling stats so they stay warm
                for (di, ri), deltas in tr.deltas.items():
                    rolling[(di, ri)].append(min(deltas))
                    if len(rolling[(di, ri)]) > WINDOW:
                        rolling[(di, ri)].pop(0)
                continue

            # Build feature row
            utilization = tr.n_active_routes / tr.n_vehicles if tr.n_vehicles > 0 else 0
            row = [
                tr.iter, tr.obj_before, tr.temperature,
                tr.n_active_routes, tr.n_unassigned, tr.total_stops,
                tr.max_route_len, tr.n_vehicles, f"{utilization:.4f}",
            ]

            best_d, best_r, best_min = -1, -1, float("inf")
            for di in range(nd):
                for ri in range(nr):
                    deltas = tr.deltas.get((di, ri), [0])
                    mn = np.mean(deltas)
                    mi = min(deltas)
                    md = np.median(deltas)
                    zf = sum(1 for d in deltas if d == 0) / len(deltas)

                    # Rolling best
                    rv = rolling[(di, ri)]
                    roll_min = np.mean(rv) if rv else 0.0

                    row.extend([mn, mi, md, zf, roll_min])

                    if mi < best_min:
                        best_min = mi
                        best_d, best_r = di, ri

            row.extend([best_d, best_r, best_min])
            writer.writerow(row)
            n_written += 1

            # Update rolling stats
            for (di, ri), deltas in tr.deltas.items():
                rolling[(di, ri)].append(min(deltas))
                if len(rolling[(di, ri)]) > WINDOW:
                    rolling[(di, ri)].pop(0)

    print(f"Saved: {out_path}")
    print(f"  Written: {n_written}, Skipped: {n_skipped} "
          f"(needed >= {min_nonzero_combos} non-zero combos)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <log.jsonl> [min_nonzero_combos=3]")
        sys.exit(1)
    log_path = sys.argv[1]
    min_nz = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(log_path, min_nz)
