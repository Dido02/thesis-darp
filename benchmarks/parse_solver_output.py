"""Parse darp_solver --dynamic stdout into structured metrics.

Public API:
    parse_solver_stdout(text: str) -> tuple[dict, list[dict]]
        -> (summary_metrics, epoch_rows)

Summary metrics include everything not keyed by epoch (instance size, totals,
ML model version if present). Epoch rows are one dict per [Epoch N] block,
suitable for a CSV with one row per epoch.

This module is standalone: no dependencies beyond the Python stdlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_INSTANCE_HEADER = re.compile(
    r"^\s*n=(\d+),\s*K=(\d+),\s*Q=(\d+),\s*L=(\S+),\s*T=(\S+)")

_RE_ML_SELECTOR = re.compile(
    r"ML selector loaded:\s*model_version=(\S+),\s*(\d+)\s*active pairs,\s*(\d+)\s*features")

_RE_EPOCH_HEADER = re.compile(
    r"^\[Epoch\s+(\d+)\]\s*t=(\S+)\s*\|\s*"
    r"known=(\d+)\s*committed=(\d+)\s*uncommitted=(\d+)\s*pending=(\d+)\s*rejected=(\d+)")

_RE_EPOCH_ITERS = re.compile(
    r"\|\s*iters=(\d+)"
    r"(?:\s*\(improve=(\d+)\s+best=(\d+)\s+sa=(\d+)\))?"  # optional: present in newer logs
    r"\s*obj=(\S+?)->(\S+?)\s*\([^)]*\)\s*\|\s*"
    r"destroy=(\S+?)ms\s*repair=(\S+?)ms\s*other=(\S+?)ms\s*total=(\S+?)ms")

_RE_EPOCH_STATS = re.compile(
    r"\|\s*routes/req=(\d+)/\d+\s*feas_checks=(\d+)/iter\s*"
    r"aux=(\S+?)ms\s*greedy=(\S+?)ms"
    r"(?:\s*selection=(\S+?)ms)?")  # selection=...ms is optional (older logs lack it)

_RE_REPAIR_OPS = re.compile(
    r"\|\s*repair_ops:\s*(.+)$")

_RE_DESTROY_OPS = re.compile(
    r"\|\s*destroy_ops:\s*(.+)$")

_RE_OP_ENTRY = re.compile(r"(\w+)=(\S+?)ms\((\d+)\)")

_RE_SUMMARY_SERVED = re.compile(r"Served:\s*(\d+)")
_RE_SUMMARY_REJECTED = re.compile(r"Rejected:\s*(\d+)")
_RE_SUMMARY_TOTAL_TRAVEL = re.compile(r"Total travel:\s*(\S+)")
_RE_SUMMARY_EPOCHS = re.compile(r"^Epochs:\s*(\d+)", re.MULTILINE)
_RE_SUMMARY_WALL_CLOCK = re.compile(r"Wall-clock:\s*(\S+?)s")
_RE_EPOCH_BUDGET = re.compile(r"Epoch budget:\s*(\d+)ms")
_RE_EPOCH_LENGTH = re.compile(r"Epoch mode:\s*time-based\s*\((\S+?)s\)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _float(s: str) -> float:
    """Parse a float, tolerating '+123', '-0', etc."""
    return float(s)


def _parse_op_list(line_tail: str) -> dict[str, dict[str, float]]:
    """Parse '| repair_ops: greedy=462ms(5388) regret2=476ms(5478) ...' tail.

    Returns {op_name: {'time_ms': float, 'count': int}}.
    """
    out: dict[str, dict[str, float]] = {}
    for m in _RE_OP_ENTRY.finditer(line_tail):
        op = m.group(1)
        out[op] = {
            "time_ms": float(m.group(2)),
            "count": int(m.group(3)),
        }
    return out


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def parse_solver_stdout(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract structured metrics from darp_solver --dynamic output.

    Returns (summary_dict, epoch_rows).
    summary_dict includes instance meta, ML metadata (if present), totals.
    epoch_rows: one dict per [Epoch N] block.
    """
    summary: dict[str, Any] = {
        "instance_num_requests": None,
        "instance_num_vehicles": None,
        "instance_capacity": None,
        "instance_max_ride_time": None,
        "instance_max_route_duration": None,
        "epoch_length_s": None,
        "epoch_budget_ms": None,
        "model_version": None,
        "n_active_pairs_loaded": None,
        "n_features_loaded": None,
        "total_served": None,
        "total_rejected": None,
        "total_travel_time": None,
        "num_epochs": None,
        "total_wall_clock_s": None,
    }

    epoch_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Single-pass: split lines, find epoch blocks + one-off markers.
    # ------------------------------------------------------------------
    lines = text.splitlines()

    # Instance header & simulation meta (one-time markers anywhere early).
    for line in lines:
        if summary["instance_num_requests"] is None:
            m = _RE_INSTANCE_HEADER.match(line)
            if m:
                summary["instance_num_requests"] = int(m.group(1))
                summary["instance_num_vehicles"] = int(m.group(2))
                summary["instance_capacity"] = int(m.group(3))
                summary["instance_max_ride_time"] = _float(m.group(4))
                summary["instance_max_route_duration"] = _float(m.group(5))
                continue

        if summary["model_version"] is None:
            m = _RE_ML_SELECTOR.search(line)
            if m:
                summary["model_version"] = m.group(1)
                summary["n_active_pairs_loaded"] = int(m.group(2))
                summary["n_features_loaded"] = int(m.group(3))
                continue

        if summary["epoch_budget_ms"] is None:
            m = _RE_EPOCH_BUDGET.search(line)
            if m:
                summary["epoch_budget_ms"] = int(m.group(1))
                continue

        if summary["epoch_length_s"] is None:
            m = _RE_EPOCH_LENGTH.search(line)
            if m:
                summary["epoch_length_s"] = _float(m.group(1))
                continue

    # Epoch blocks: each one spans 5 non-blank lines starting with [Epoch N].
    # We parse them sequentially.
    i = 0
    while i < len(lines):
        line = lines[i]
        m_hdr = _RE_EPOCH_HEADER.match(line)
        if not m_hdr:
            i += 1
            continue

        row: dict[str, Any] = {
            "epoch_id": int(m_hdr.group(1)),
            "sim_time": _float(m_hdr.group(2)),
            "requests_known": int(m_hdr.group(3)),
            "requests_committed": int(m_hdr.group(4)),
            "requests_uncommitted": int(m_hdr.group(5)),
            "requests_pending": int(m_hdr.group(6)),
            "requests_rejected": int(m_hdr.group(7)),
        }

        # Next line: iters/obj/timing
        if i + 1 < len(lines):
            m2 = _RE_EPOCH_ITERS.search(lines[i + 1])
            if m2:
                row["alns_iterations"] = int(m2.group(1))
                if m2.group(2) is not None:
                    row["iters_improving"] = int(m2.group(2))
                    row["iters_new_best"] = int(m2.group(3))
                    row["iters_sa_accepted"] = int(m2.group(4))
                row["obj_before"] = _float(m2.group(5))
                row["obj_after"] = _float(m2.group(6))
                row["destroy_time_ms"] = _float(m2.group(7))
                row["repair_time_ms"] = _float(m2.group(8))
                row["other_time_ms"] = _float(m2.group(9))
                row["wall_clock_ms"] = _float(m2.group(10))

        # --dynamic-stats lines are optional. Try the next 3 lines.
        for j in range(i + 2, min(i + 5, len(lines))):
            m_stats = _RE_EPOCH_STATS.search(lines[j])
            if m_stats:
                row["avg_routes_per_request"] = int(m_stats.group(1))
                row["avg_feas_checks"] = int(m_stats.group(2))
                row["aux_precompute_ms"] = _float(m_stats.group(3))
                row["greedy_preinsert_ms"] = _float(m_stats.group(4))
                if m_stats.group(5) is not None:
                    row["selection_time_ms"] = _float(m_stats.group(5))

            m_rep = _RE_REPAIR_OPS.search(lines[j])
            if m_rep and "repair_ops" in lines[j]:
                ops = _parse_op_list(m_rep.group(1))
                row["repair_ops"] = ops

            m_des = _RE_DESTROY_OPS.search(lines[j])
            if m_des and "destroy_ops" in lines[j]:
                ops = _parse_op_list(m_des.group(1))
                row["destroy_ops"] = ops

        epoch_rows.append(row)
        i += 1

    # ------------------------------------------------------------------
    # Final summary section
    # ------------------------------------------------------------------
    for m, key, cast in [
        (_RE_SUMMARY_SERVED.search(text), "total_served", int),
        (_RE_SUMMARY_REJECTED.search(text), "total_rejected", int),
        (_RE_SUMMARY_TOTAL_TRAVEL.search(text), "total_travel_time", _float),
        (_RE_SUMMARY_EPOCHS.search(text), "num_epochs", int),
        (_RE_SUMMARY_WALL_CLOCK.search(text), "total_wall_clock_s", _float),
    ]:
        if m:
            summary[key] = cast(m.group(1))

    # Derived aggregates from epoch rows
    if epoch_rows:
        summary["total_alns_iterations"] = sum(
            r.get("alns_iterations", 0) or 0 for r in epoch_rows)
        summary["total_iters_improving"] = sum(
            r.get("iters_improving", 0) or 0 for r in epoch_rows)
        summary["total_iters_new_best"] = sum(
            r.get("iters_new_best", 0) or 0 for r in epoch_rows)
        summary["total_iters_sa_accepted"] = sum(
            r.get("iters_sa_accepted", 0) or 0 for r in epoch_rows)
        summary["total_destroy_time_s"] = sum(
            r.get("destroy_time_ms", 0.0) or 0.0 for r in epoch_rows) / 1000.0
        summary["total_repair_time_s"] = sum(
            r.get("repair_time_ms", 0.0) or 0.0 for r in epoch_rows) / 1000.0
        summary["total_other_time_s"] = sum(
            r.get("other_time_ms", 0.0) or 0.0 for r in epoch_rows) / 1000.0

        first = epoch_rows[0]
        last = epoch_rows[-1]
        summary["obj_first_epoch_before"] = first.get("obj_before")
        summary["obj_last_epoch_after"] = last.get("obj_after")
    else:
        summary["total_alns_iterations"] = 0
        summary["total_iters_improving"] = 0
        summary["total_iters_new_best"] = 0
        summary["total_iters_sa_accepted"] = 0
        summary["total_destroy_time_s"] = 0.0
        summary["total_repair_time_s"] = 0.0
        summary["total_other_time_s"] = 0.0
        summary["obj_first_epoch_before"] = None
        summary["obj_last_epoch_after"] = None

    # Service rate
    if summary["total_served"] is not None:
        denom = summary["total_served"] + (summary["total_rejected"] or 0)
        summary["service_rate"] = (
            summary["total_served"] / denom if denom > 0 else None)
    else:
        summary["service_rate"] = None

    return summary, epoch_rows


# ---------------------------------------------------------------------------
# CLI: parse a file or stdin, print JSON
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Parse darp_solver stdout to JSON")
    ap.add_argument("input", nargs="?", default="-",
                    help="Path to stdout log file, or - for stdin (default)")
    args = ap.parse_args()

    text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    summary, epochs = parse_solver_stdout(text)

    print(json.dumps({"summary": summary, "epochs": epochs}, indent=2,
                     default=str))
