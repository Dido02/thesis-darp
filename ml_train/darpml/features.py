"""Feature registry and extraction for ML operator selection."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .instance import DARPInstance
from .loader import LogHeader, SolverState

FeatureFunc = Callable[[SolverState, DARPInstance, LogHeader], float]

_REGISTRY: dict[str, FeatureFunc] = {}


def register_feature(name: str):
    """Decorator to register a feature function."""
    def decorator(func: FeatureFunc) -> FeatureFunc:
        _REGISTRY[name] = func
        return func
    return decorator


def get_feature_names() -> list[str]:
    """Return sorted list of registered feature names."""
    return sorted(_REGISTRY.keys())


def extract_all(
    state: SolverState,
    instance: DARPInstance,
    header: LogHeader,
) -> dict[str, float]:
    """Extract all registered features for one state."""
    return {name: func(state, instance, header) for name, func in _REGISTRY.items()}


def _get_active_pickups(state: SolverState, n: int) -> list[int]:
    """Get pickup node IDs of in-route-not-committed requests."""
    committed = set(state.committed)
    active = []
    for route in state.routes:
        for node_id in route:
            if 1 <= node_id <= n and node_id not in committed:
                active.append(node_id)
    return active


# ---------------------------------------------------------------------------
# Smoke-test features (5 total, one per Bongiovanni category)
# ---------------------------------------------------------------------------

@register_feature("num_active_requests")
def _num_active_requests(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """Count of requests in routes but not committed. Category: demand size."""
    return float(len(_get_active_pickups(state, instance.num_requests)))


@register_feature("radius_avg_tt")
def _radius_avg_tt(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """Mean pairwise travel time between active pickups. Category: demand spread.

    Uses the instance travel-time matrix. For active pickup nodes, extracts the
    submatrix and computes mean of off-diagonal entries.
    """
    pickups = _get_active_pickups(state, instance.num_requests)
    if len(pickups) <= 1:
        return 0.0
    idx = np.array(pickups)
    sub = instance.travel_time[np.ix_(idx, idx)]
    # Mean of off-diagonal elements
    n = len(pickups)
    total = sub.sum() - np.trace(sub)
    return float(total / (n * (n - 1)))


@register_feature("frac_committed")
def _frac_committed(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """|committed| / (|committed| + |in-route-not-committed|). Category: vehicle state."""
    n_committed = len(state.committed)
    n_active = len(_get_active_pickups(state, instance.num_requests))
    denom = n_committed + n_active
    return n_committed / denom if denom > 0 else 0.0


@register_feature("avg_route_load")
def _avg_route_load(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """Mean requests per non-empty route. Category: routing solution."""
    n = instance.num_requests
    empty_size = instance.empty_route_size
    loads = []
    for route in state.routes:
        if len(route) > empty_size:
            req_count = sum(1 for nid in route if 1 <= nid <= n)
            loads.append(req_count)
    return float(np.mean(loads)) if loads else 0.0


@register_feature("temperature_normalized")
def _temperature_normalized(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """temperature / sa_start_temp, in [0, 1]. Category: ALNS state."""
    if header.sa_start_temp <= 0:
        return 0.0
    return state.temperature / header.sa_start_temp
