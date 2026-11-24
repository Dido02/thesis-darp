"""Walk bench_results/runs/*/summary.json → flat summary.csv.

Re-runnable; independent of the runner. Picks up every run directory (including
timestamped re-runs from --force) and emits one CSV row per run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "bench_results"


def main():
    ap = argparse.ArgumentParser(description="Aggregate bench_results/runs/ into summary.csv")
    ap.add_argument("--results-dir", default=str(DEFAULT_OUT_DIR),
                    help="Root with runs/ and where summary.csv is written")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <results-dir>/summary.csv)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs_dir = results_dir / "runs"
    out_path = Path(args.out) if args.out else results_dir / "summary.csv"

    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        try:
            with open(summary_path) as f:
                row = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  WARN: cannot parse {summary_path}: {e}", file=sys.stderr)
            continue
        # Flatten any nested dicts (we don't expect any except validation_message
        # which is already a string, but be defensive).
        flat = {k: (v if not isinstance(v, (dict, list)) else json.dumps(v))
                for k, v in row.items()}
        rows.append(flat)

    if not rows:
        print(f"No summary.json files found under {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Union of all keys, stable ordering
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})

    print(f"Aggregated {len(rows)} runs -> {out_path}")
    print(f"Columns: {len(all_keys)}")


if __name__ == "__main__":
    main()
