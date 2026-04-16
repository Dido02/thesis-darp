"""Objective recomputation and validation against logged values."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .instance import DARPInstance, InstanceTooLargeError, load_instance
from .loader import LogHeader, SolverState, StateTrial, load_log, resolve_instance_path


def compute_objective(
    routes: list[list[int]],
    unassigned: list[int],
    travel_time: np.ndarray,
    unassigned_penalty: float,
) -> float:
    """Recompute objective from routes and unassigned set.

    Mirrors cpp/src/solution.cpp::objective() lines 282-294 exactly:
        obj = sum(travel_time[route[i]][route[i+1]]) + len(unassigned) * penalty
    """
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += travel_time[route[i], route[i + 1]]
    total += len(unassigned) * unassigned_penalty
    return total


@dataclass
class ValidationReport:
    log_path: str
    instance_path: str
    total_states: int
    passed: int
    failed: int
    max_abs_error: float
    mismatches: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def validate_state(
    state: SolverState,
    instance: DARPInstance,
    unassigned_penalty: float,
    tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Validate one state record. Returns (passed, abs_error)."""
    recomputed = compute_objective(
        state.routes, state.unassigned,
        instance.travel_time, unassigned_penalty,
    )
    error = abs(recomputed - state.obj_before)
    return error < tolerance, error


def validate_log(
    log_path: str | Path,
    data_root: str | Path,
    tolerance: float = 0.05,
    max_mismatches: int = 10,
) -> ValidationReport:
    """Validate all state records in a log against recomputed objectives."""
    log_path = Path(log_path)
    header, state_trials = load_log(log_path)

    # Resolve and load instance
    instance_path = resolve_instance_path(header.instance_path, data_root)
    instance = load_instance(instance_path)

    report = ValidationReport(
        log_path=str(log_path),
        instance_path=str(instance_path),
        total_states=len(state_trials),
        passed=0,
        failed=0,
        max_abs_error=0.0,
    )

    for st in state_trials:
        passed, error = validate_state(
            st.state, instance, header.unassigned_penalty, tolerance,
        )
        report.max_abs_error = max(report.max_abs_error, error)

        if passed:
            report.passed += 1
        else:
            report.failed += 1
            if len(report.mismatches) < max_mismatches:
                recomputed = compute_objective(
                    st.state.routes, st.state.unassigned,
                    instance.travel_time, header.unassigned_penalty,
                )
                report.mismatches.append({
                    "iter": st.state.iter,
                    "logged": st.state.obj_before,
                    "recomputed": recomputed,
                    "error": error,
                    "n_unassigned": len(st.state.unassigned),
                })

    return report
