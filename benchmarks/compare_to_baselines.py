"""Build baselines.csv from data/fiedler/Results/ and comparison.csv (join).

Two-step pipeline:

1. baselines.csv  — one row per (instance, baseline_method).
                    Pulled from data/fiedler/Results/<Area>/start_<start>/
                    duration_<dur>/max_delay_<dly>/{ih,vga}/config.yaml-*.json.

2. comparison.csv — the join of summary.csv (our runs) with baselines.csv.
                    One row per run with pct_vs_ih, pct_vs_vga columns broadcast
                    from the seed-independent baselines.

Both files are written under the bench_results directory and can be regenerated
freely.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "bench_results"
FIEDLER_RESULTS_ROOT = REPO_ROOT / "data" / "fiedler" / "Results"
BASELINE_METHODS = ["ih", "vga"]


# ---------------------------------------------------------------------------
# Duration tier <-> Fiedler folder name
# ---------------------------------------------------------------------------
def _fiedler_tier(canonical: str) -> str:
    """'01min' -> '01_min', '30s' -> '30_s', '15min' -> '15_min', etc."""
    if canonical == "30s":
        return "30_s"
    return canonical.replace("min", "_min")


def _fiedler_result_dir(area: str, start: str, duration: str, delay: str,
                        method: str) -> Path:
    """Build expected path under data/fiedler/Results/."""
    return (FIEDLER_RESULTS_ROOT / area.capitalize()
            / f"start_{start}"
            / f"duration_{_fiedler_tier(duration)}"
            / f"max_delay_{_fiedler_tier(delay)}"
            / method)


# ---------------------------------------------------------------------------
# Step 1: baselines.csv
# ---------------------------------------------------------------------------
def build_baselines(out_dir: Path, summary_csv: Path) -> Path:
    """Emit baselines.csv with one row per (instance, start, method)."""
    # Read unique (instance, duration, delay, start) combos from summary.csv
    combos: set[tuple[str, str, str, str]] = set()
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            combos.add((
                row["instance"],
                row["instance_duration"],
                row["instance_delay"],
                row["start"],
            ))

    out_path = out_dir / "baselines.csv"
    cols = ["instance", "start", "method", "cost", "cost_min",
            "vehicles_used", "dropped", "time_ms", "available"]
    rows = []

    for (instance, duration, delay, start) in sorted(combos):
        for method in BASELINE_METHODS:
            method_dir = _fiedler_result_dir("Manhattan", start, duration, delay, method)
            sol_path = method_dir / "config.yaml-solution.json"
            perf_path = method_dir / "config.yaml-performance.json"

            row = {"instance": instance, "start": start, "method": method,
                   "cost": "", "cost_min": "", "vehicles_used": "",
                   "dropped": "", "time_ms": "", "available": False}

            if sol_path.exists():
                try:
                    with open(sol_path) as f:
                        sol = json.load(f)
                    cost = sol.get("cost", None)
                    row.update({
                        "cost": cost,
                        "cost_min": (cost / 60.0) if cost is not None else "",
                        "vehicles_used": len(sol.get("plans", [])),
                        "dropped": len(sol.get("dropped_requests", [])),
                        "available": True,
                    })
                except (OSError, json.JSONDecodeError):
                    pass

            if perf_path.exists():
                try:
                    with open(perf_path) as f:
                        perf = json.load(f)
                    row["time_ms"] = perf.get("total_time", "")
                except (OSError, json.JSONDecodeError):
                    pass

            rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    available = sum(1 for r in rows if r["available"])
    print(f"baselines: {len(rows)} rows written ({available} available)"
          f" -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 2: comparison.csv (join)
# ---------------------------------------------------------------------------
def build_comparison(out_dir: Path, summary_csv: Path, baselines_csv: Path) -> Path:
    """Emit comparison.csv joining summary.csv × baselines.csv."""
    # Index baselines by (instance, start)
    baseline_idx: dict[tuple[str, str], dict[str, dict]] = {}
    with open(baselines_csv) as f:
        for row in csv.DictReader(f):
            key = (row["instance"], row["start"])
            baseline_idx.setdefault(key, {})[row["method"]] = row

    def _f(v) -> float | None:
        if v in ("", None, "None"):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    out_path = out_dir / "comparison.csv"
    cols = [
        "instance", "instance_duration", "instance_delay", "start", "seed",
        "selection", "config_name", "epoch_budget_ms",
        "cost", "cost_minutes", "vehicles_used", "service_rate",
        "validation_passed", "total_wall_clock_s",
        "model_version",
        "ih_cost", "vga_cost",
        "ih_time_ms", "vga_time_ms",
        "pct_vs_ih", "pct_vs_vga",
        "speedup_vs_ih", "speedup_vs_vga",
    ]
    rows_out = []

    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            bl = baseline_idx.get((row["instance"], row["start"]), {})
            ih = bl.get("ih", {})
            vga = bl.get("vga", {})

            our_cost = _f(row.get("cost"))
            our_wall_s = _f(row.get("total_wall_clock_s"))

            ih_cost = _f(ih.get("cost"))
            vga_cost = _f(vga.get("cost"))
            ih_time_ms = _f(ih.get("time_ms"))
            vga_time_ms = _f(vga.get("time_ms"))

            def _pct(ours, theirs):
                if ours is None or theirs is None or theirs == 0:
                    return ""
                return (ours - theirs) / theirs * 100.0

            def _speedup(ours_s, theirs_ms):
                if ours_s is None or theirs_ms is None or ours_s == 0:
                    return ""
                return theirs_ms / (ours_s * 1000.0)

            out = {c: "" for c in cols}
            # Carry over from summary
            for c in ("instance", "instance_duration", "instance_delay", "start",
                      "seed", "selection", "config_name", "epoch_budget_ms",
                      "cost", "cost_minutes", "vehicles_used", "service_rate",
                      "validation_passed", "total_wall_clock_s", "model_version"):
                out[c] = row.get(c, "")
            out["ih_cost"] = ih_cost if ih_cost is not None else ""
            out["vga_cost"] = vga_cost if vga_cost is not None else ""
            out["ih_time_ms"] = ih_time_ms if ih_time_ms is not None else ""
            out["vga_time_ms"] = vga_time_ms if vga_time_ms is not None else ""
            out["pct_vs_ih"] = _pct(our_cost, ih_cost)
            out["pct_vs_vga"] = _pct(our_cost, vga_cost)
            out["speedup_vs_ih"] = _speedup(our_wall_s, ih_time_ms)
            out["speedup_vs_vga"] = _speedup(our_wall_s, vga_time_ms)
            rows_out.append(out)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"comparison: {len(rows_out)} rows written -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build baselines.csv + comparison.csv")
    ap.add_argument("--results-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--summary-csv", default=None,
                    help="Path to summary.csv (default: <results-dir>/summary.csv)")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    summary_csv = (Path(args.summary_csv) if args.summary_csv
                   else out_dir / "summary.csv")
    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found. "
              f"Run aggregate_results.py first.", file=sys.stderr)
        sys.exit(1)

    baselines_csv = build_baselines(out_dir, summary_csv)
    build_comparison(out_dir, summary_csv, baselines_csv)


if __name__ == "__main__":
    main()
