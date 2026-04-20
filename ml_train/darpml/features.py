"""Feature registry and extraction for ML operator selection.

Each feature carries three pieces of metadata, declared at registration:
- batch:     named group (e.g. "core", "tightness"). Used to toggle subsets.
- mode:      "static", "dynamic", or "both". Restricts which solver mode the
             feature is meaningful in. "both" matches either.
- tolerance: parity tolerance threshold (max abs error vs C++) for this feature.
             Convention: 0.0 for sums/counts/ratios/max/min; 1e-12 for variance
             and other multi-pass accumulations where numpy/C++ may differ in
             the last bit. The parity fixture emits this per-feature so the
             C++ test reads it from a single source of truth (the registry).

Adding a feature requires the same metadata in C++ via REGISTER_FEATURE_V2 in
cpp/src/ml/feature_extractor.cpp.

21 features across 5 batches:

  feature                          batch            mode       tol     source
  -------------------------------- ---------------- ---------- ------- ----------
  num_active_requests              core             both       0.0     novel
  radius_avg_tt                    core             both       0.0     Bongiovanni #3
  frac_committed                   core             dynamic    0.0     novel
  avg_route_load                   core             both       0.0     Bongiovanni #26
  temperature_normalized           core             both       0.0     novel
  ride_time_utilization_avg        tightness        both       0.0     novel
  ride_time_utilization_max        tightness        both       0.0     novel
  near_violation_count             tightness        both       0.0     novel
  route_slack_avg                  tightness        both       0.0     novel
  route_slack_std                  tightness        both       1e-12   novel
  cost_contribution_std            worst_support    both       1e-12   novel
  cost_contribution_max_ratio      worst_support    both       0.0     novel
  route_cost_std                   worst_support    both       1e-12   novel
  neighborhood_size_frac           lns_state        both       0.0     Bongiovanni #41
  cost_vs_initial_frac             lns_state        both       0.0     Bongiovanni #42
  cost_vs_incumbent_frac           lns_state        both       0.0     Bongiovanni #43
  iteration_norm                   lns_state        both       0.0     Bongiovanni #46
  unassigned_count                 lns_state        both       0.0     novel
  committed_travel_time_frac       rolling_horizon  dynamic    0.0     novel
  horizon_elapsed_frac             rolling_horizon  dynamic    0.0     Bongiovanni #2
  flexibility_horizon              rolling_horizon  dynamic    0.0     novel

Bongiovanni numbering follows their 2022 paper's 46-feature catalogue. `mode`
semantics: features tagged `dynamic` are stripped from static-mode model
manifests; `both` features apply to either. Per-state dynamic detection in
features that fork on it (e.g. neighborhood_size_frac) uses
`bool(state.committed)`, NOT the registry mode tag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .instance import DARPInstance
from .loader import LogHeader, SolverState

FeatureFunc = Callable[[SolverState, DARPInstance, LogHeader], float]
FeatureMode = Literal["static", "dynamic", "both"]


@dataclass(frozen=True)
class FeatureEntry:
    name: str
    func: FeatureFunc
    batch: str
    mode: FeatureMode
    tolerance: float


_REGISTRY: dict[str, FeatureEntry] = {}


def register_feature(
    name: str,
    *,
    batch: str,
    mode: FeatureMode,
    tolerance: float,
):
    """Decorator to register a feature. batch/mode/tolerance are all required."""
    if mode not in ("static", "dynamic", "both"):
        raise ValueError(f"register_feature {name}: mode must be one of "
                         f"'static','dynamic','both', got {mode!r}")
    if tolerance < 0:
        raise ValueError(f"register_feature {name}: tolerance must be >= 0")

    def decorator(func: FeatureFunc) -> FeatureFunc:
        if name in _REGISTRY:
            raise ValueError(f"Feature {name!r} already registered")
        _REGISTRY[name] = FeatureEntry(
            name=name, func=func, batch=batch, mode=mode, tolerance=tolerance,
        )
        return func
    return decorator


def _mode_matches(feature_mode: FeatureMode, requested_mode: FeatureMode) -> bool:
    """A feature is included in `requested_mode` if its mode == requested or 'both'."""
    if feature_mode == "both":
        return True
    return feature_mode == requested_mode


def get_features(
    batches: list[str] | None,
    mode: FeatureMode,
) -> list[FeatureEntry]:
    """Return registered features filtered by batches and mode.

    - batches=None matches all batches.
    - mode="static" matches features whose mode is "static" or "both".
    - mode="dynamic" matches features whose mode is "dynamic" or "both".

    Order: registry insertion order (deterministic, matches training-time order).
    """
    if mode not in ("static", "dynamic"):
        raise ValueError(f"get_features: mode must be 'static' or 'dynamic', got {mode!r}")
    batch_set = set(batches) if batches is not None else None
    out: list[FeatureEntry] = []
    for entry in _REGISTRY.values():
        if batch_set is not None and entry.batch not in batch_set:
            continue
        if not _mode_matches(entry.mode, mode):
            continue
        out.append(entry)
    return out


def get_feature_names(
    batches: list[str] | None = None,
    mode: FeatureMode | None = None,
) -> list[str]:
    """Return registered feature names in registry-insertion order.

    With no args, returns ALL registered names (legacy behavior, used by
    extract_all default and by callers that want everything). With batches/mode,
    returns only the filtered subset.
    """
    if batches is None and mode is None:
        return list(_REGISTRY.keys())
    if mode is None:
        raise ValueError("get_feature_names: pass mode when filtering by batches")
    return [e.name for e in get_features(batches, mode)]


def extract_all(
    state: SolverState,
    instance: DARPInstance,
    header: LogHeader,
    batches: list[str] | None = None,
    mode: FeatureMode | None = None,
) -> dict[str, float]:
    """Extract features for one state.

    With no batches/mode: extract every registered feature (legacy behavior).
    With batches+mode: extract only the filtered subset.
    """
    if batches is None and mode is None:
        return {name: e.func(state, instance, header) for name, e in _REGISTRY.items()}
    if mode is None:
        raise ValueError("extract_all: pass mode when filtering by batches")
    return {e.name: e.func(state, instance, header) for e in get_features(batches, mode)}


def get_registered_batches() -> list[str]:
    """Return the set of distinct batch names currently registered, in insertion order."""
    seen: dict[str, None] = {}
    for e in _REGISTRY.values():
        seen.setdefault(e.batch, None)
    return list(seen.keys())


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
# Batch: core (5 features). Smoke-test set, one per Bongiovanni category.
# ---------------------------------------------------------------------------

@register_feature("num_active_requests", batch="core", mode="both", tolerance=0.0)
def _num_active_requests(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """Count of requests in routes but not committed. Category: demand size."""
    return float(len(_get_active_pickups(state, instance.num_requests)))


@register_feature("radius_avg_tt", batch="core", mode="both", tolerance=0.0)
def _radius_avg_tt(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """Mean pairwise travel time between active pickups. Category: demand spread.

    Implementation matches the C++ feat_radius_avg_tt accumulation order
    exactly (outer i, inner j; trace accumulated in the outer loop) so the
    last-bit floating-point result agrees with C++. Using numpy's sub.sum()
    here produced a ~1e-15 divergence on small active sets because numpy's
    pairwise summation visits elements in a different order.
    """
    pickups = _get_active_pickups(state, instance.num_requests)
    n = len(pickups)
    if n <= 1:
        return 0.0
    tt = instance.travel_time
    total = 0.0
    trace = 0.0
    for i in range(n):
        ni = pickups[i]
        trace += float(tt[ni][ni])
        for j in range(n):
            total += float(tt[ni][pickups[j]])
    off_diag = total - trace
    return off_diag / (float(n) * (n - 1))


@register_feature("frac_committed", batch="core", mode="dynamic", tolerance=0.0)
def _frac_committed(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """|committed| / (|committed| + |in-route-not-committed|). Category: vehicle state.

    Dynamic-only because `committed` is meaningfully populated only in rolling-horizon mode;
    in pure static mode the committed set is empty by construction.
    """
    n_committed = len(state.committed)
    n_active = len(_get_active_pickups(state, instance.num_requests))
    denom = n_committed + n_active
    return n_committed / denom if denom > 0 else 0.0


@register_feature("avg_route_load", batch="core", mode="both", tolerance=0.0)
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


@register_feature("temperature_normalized", batch="core", mode="both", tolerance=0.0)
def _temperature_normalized(state: SolverState, instance: DARPInstance, header: LogHeader) -> float:
    """temperature / sa_start_temp, in [0, 1]. Category: ALNS state."""
    if header.sa_start_temp <= 0:
        return 0.0
    return state.temperature / header.sa_start_temp


# ---------------------------------------------------------------------------
# Batch: tightness (5 features). Targets rtcritical + lowslack operators by
# surfacing how close to ride-time and time-window violations the current
# solution is.
# ---------------------------------------------------------------------------

# Threshold for the "near violation" count. A request whose ride time uses
# this fraction of its allowed L is considered close to violating.
NEAR_VIOLATION_THRESHOLD = 0.9


def _ride_time_utilizations(state: SolverState, instance: DARPInstance) -> list[float]:
    """Per active-pickup request: actual_ride_time / max_ride_time_for(req).

    Iterates routes in the same order as C++. Active = appears in some route
    AND not in `committed`. Requests with no recorded ride time (e.g., the
    pickup is in route but the delivery is missing — shouldn't happen on a
    valid solution) are skipped to mirror C++'s map lookup behavior.
    """
    from .reconstruction import compute_route_schedule
    committed = set(state.committed)
    n = instance.num_requests
    out: list[float] = []
    for route in state.routes:
        ride = compute_route_schedule(route, instance)
        # Mirror C++ traversal order: walk pickup positions in this route.
        for node_id in route:
            if not (1 <= node_id <= n):
                continue
            if node_id in committed:
                continue
            if node_id not in ride:
                continue
            L = instance.max_ride_time_for(node_id)
            if L <= 0.0:
                continue
            out.append(ride[node_id] / L)
    return out


@register_feature("ride_time_utilization_avg",
                  batch="tightness", mode="both", tolerance=0.0)
def _ride_time_utilization_avg(state: SolverState, instance: DARPInstance,
                                header: LogHeader) -> float:
    """Mean of actual_ride_time / L over active requests. 0.0 if none."""
    utils = _ride_time_utilizations(state, instance)
    if not utils:
        return 0.0
    total = 0.0
    for u in utils:
        total += u
    return total / len(utils)


@register_feature("ride_time_utilization_max",
                  batch="tightness", mode="both", tolerance=0.0)
def _ride_time_utilization_max(state: SolverState, instance: DARPInstance,
                                header: LogHeader) -> float:
    """Max of actual_ride_time / L over active requests. 0.0 if none."""
    utils = _ride_time_utilizations(state, instance)
    if not utils:
        return 0.0
    m = utils[0]
    for u in utils[1:]:
        if u > m:
            m = u
    return m


@register_feature("near_violation_count",
                  batch="tightness", mode="both", tolerance=0.0)
def _near_violation_count(state: SolverState, instance: DARPInstance,
                           header: LogHeader) -> float:
    """Count of active requests with utilization >= NEAR_VIOLATION_THRESHOLD."""
    utils = _ride_time_utilizations(state, instance)
    count = 0
    for u in utils:
        if u >= NEAR_VIOLATION_THRESHOLD:
            count += 1
    return float(count)


def _per_route_slacks(state: SolverState, instance: DARPInstance) -> list[float]:
    """Slack per non-empty route, in route iteration order."""
    from .reconstruction import compute_route_slack
    out: list[float] = []
    for route in state.routes:
        if len(route) > instance.empty_route_size:
            out.append(compute_route_slack(route, instance))
    return out


@register_feature("route_slack_avg",
                  batch="tightness", mode="both", tolerance=0.0)
def _route_slack_avg(state: SolverState, instance: DARPInstance,
                      header: LogHeader) -> float:
    """Mean total forward time slack across non-empty routes. 0.0 if none."""
    slacks = _per_route_slacks(state, instance)
    if not slacks:
        return 0.0
    total = 0.0
    for s in slacks:
        total += s
    return total / len(slacks)


@register_feature("route_slack_std",
                  batch="tightness", mode="both", tolerance=1e-12)
def _route_slack_std(state: SolverState, instance: DARPInstance,
                      header: LogHeader) -> float:
    """Population std dev of per-route slacks (ddof=0). 0.0 if <= 1 route.

    Two-pass naive accumulation (sum-then-mean, then sum-of-squared-deviations)
    so the C++ side can mirror with the same operation order.
    """
    slacks = _per_route_slacks(state, instance)
    if len(slacks) <= 1:
        return 0.0
    total = 0.0
    for s in slacks:
        total += s
    mean = total / len(slacks)
    sq = 0.0
    for s in slacks:
        d = s - mean
        sq += d * d
    return (sq / len(slacks)) ** 0.5


# ---------------------------------------------------------------------------
# Batch: worst_support (3 features). Targets the worst_destroy operator by
# surfacing the dispersion of per-request and per-route cost contributions.
# ---------------------------------------------------------------------------

def _per_active_request_costs(state: SolverState,
                                instance: DARPInstance) -> list[float]:
    """removal_cost for each active request in route iteration order.

    Active = pickup in some route AND not committed AND delivery in same route.
    The "delivery in same route" guard mirrors the C++ removal_cost contract
    (it would index out-of-bounds if d_pos < 0). On valid solutions this is
    always satisfied for active requests; the explicit guard documents intent.
    """
    from .reconstruction import removal_cost
    committed = set(state.committed)
    n = instance.num_requests
    out: list[float] = []
    for ri, route in enumerate(state.routes):
        for node_id in route:
            if not (1 <= node_id <= n):
                continue
            if node_id in committed:
                continue
            # Skip requests whose delivery isn't in the same route.
            if (node_id + n) not in route:
                continue
            out.append(removal_cost(state.routes, instance, node_id, ri))
    return out


def _per_route_costs(state: SolverState, instance: DARPInstance) -> list[float]:
    """compute_route_cost over non-empty routes in iteration order."""
    from .reconstruction import compute_route_cost
    out: list[float] = []
    for route in state.routes:
        if len(route) > instance.empty_route_size:
            out.append(compute_route_cost(route, instance))
    return out


@register_feature("cost_contribution_std",
                  batch="worst_support", mode="both", tolerance=1e-12)
def _cost_contribution_std(state: SolverState, instance: DARPInstance,
                            header: LogHeader) -> float:
    """Population std dev of per-active-request removal costs (ddof=0).
    0.0 if 0 or 1 active request.
    """
    costs = _per_active_request_costs(state, instance)
    if len(costs) <= 1:
        return 0.0
    total = 0.0
    for c in costs:
        total += c
    mean = total / len(costs)
    sq = 0.0
    for c in costs:
        d = c - mean
        sq += d * d
    return (sq / len(costs)) ** 0.5


@register_feature("cost_contribution_max_ratio",
                  batch="worst_support", mode="both", tolerance=0.0)
def _cost_contribution_max_ratio(state: SolverState, instance: DARPInstance,
                                   header: LogHeader) -> float:
    """max / mean of per-active-request removal costs.

    Edge cases (matched in C++):
    - empty active set => 0.0
    - mean == 0.0 (all contributions are zero, possible on degenerate routes)
      => 1.0 to avoid division-by-zero and to reflect "no dispersion"
    - by triangle inequality each contribution is >= 0; if any is positive
      the mean is positive too.
    """
    costs = _per_active_request_costs(state, instance)
    if not costs:
        return 0.0
    total = 0.0
    m = costs[0]
    for c in costs:
        total += c
        if c > m:
            m = c
    mean = total / len(costs)
    if mean == 0.0:
        return 1.0
    return m / mean


@register_feature("route_cost_std",
                  batch="worst_support", mode="both", tolerance=1e-12)
def _route_cost_std(state: SolverState, instance: DARPInstance,
                     header: LogHeader) -> float:
    """Population std dev of per-route travel costs (non-empty routes, ddof=0).
    0.0 if <= 1 non-empty route.
    """
    costs = _per_route_costs(state, instance)
    if len(costs) <= 1:
        return 0.0
    total = 0.0
    for c in costs:
        total += c
    mean = total / len(costs)
    sq = 0.0
    for c in costs:
        d = c - mean
        sq += d * d
    return (sq / len(costs)) ** 0.5


# ---------------------------------------------------------------------------
# Batch: lns_state (5 features). Adapts Bongiovanni et al. 2022's
# top-importance LNS state descriptors (their #41/#42/#43/#46) plus
# request-bank count.
# ---------------------------------------------------------------------------

# Constants must match the C++ destroy operators' q-sampling formula in
# cpp/src/operators/destroy/{random,worst,shaw,ride_time_critical,low_slack_route}.cpp.
# These values are identical across all 5 destroy ops; if the solver formula
# changes, update both sides simultaneously.
LO_PCT = 0.10                # lower bound on q as a fraction of N
HI_PCT_STATIC = 0.40         # upper bound when committed is empty (static mode)
HI_PCT_DYNAMIC = 0.25        # upper bound when committed is non-empty


@register_feature("neighborhood_size_frac",
                  batch="lns_state", mode="both", tolerance=0.0)
def _neighborhood_size_frac(state: SolverState, instance: DARPInstance,
                              header: LogHeader) -> float:
    """q / q_max where q_max mirrors the destroy operators' upper bound.

    `q_max = min(hi - 1, N)` to match `std::min(dist(rng), N)` in C++ —
    without the clamp, q / q_max can be < 1 even when q hit its true ceiling
    (the case where hi - 1 > N).

    Dynamic detection mirrors the per-state `!committed.empty()` check in
    each destroy operator's call site (NOT the model's mode tag).
    """
    n = len(_get_active_pickups(state, instance.num_requests))
    if n == 0:
        return 0.0
    dynamic = bool(state.committed)
    hi_pct = HI_PCT_DYNAMIC if dynamic else HI_PCT_STATIC
    lo = max(1, int(n * LO_PCT))
    hi = max(lo + 1, int(n * hi_pct) + 1)
    q_max = min(hi - 1, n)
    if q_max <= 0:
        return 0.0
    return state.q / q_max


@register_feature("cost_vs_initial_frac",
                  batch="lns_state", mode="both", tolerance=0.0)
def _cost_vs_initial_frac(state: SolverState, instance: DARPInstance,
                            header: LogHeader) -> float:
    """(obj_before - obj_initial) / obj_initial. Positive => worse than epoch start.
    No clipping; sign carries information. 0.0 if obj_initial == 0.
    """
    if state.obj_initial == 0.0:
        return 0.0
    return (state.obj_before - state.obj_initial) / state.obj_initial


@register_feature("cost_vs_incumbent_frac",
                  batch="lns_state", mode="both", tolerance=0.0)
def _cost_vs_incumbent_frac(state: SolverState, instance: DARPInstance,
                              header: LogHeader) -> float:
    """(obj_before - obj_best) / obj_best. SA accepts worsening moves so
    obj_before can exceed obj_best — value is genuinely signed, no clipping.
    0.0 if obj_best == 0.
    """
    if state.obj_best == 0.0:
        return 0.0
    return (state.obj_before - state.obj_best) / state.obj_best


@register_feature("iteration_norm",
                  batch="lns_state", mode="both", tolerance=0.0)
def _iteration_norm(state: SolverState, instance: DARPInstance,
                     header: LogHeader) -> float:
    """iter / max_iterations clipped to [0, 1]. 0.0 if max_iterations == 0."""
    if header.max_iterations == 0:
        return 0.0
    raw = state.iter / header.max_iterations
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


@register_feature("unassigned_count",
                  batch="lns_state", mode="both", tolerance=0.0)
def _unassigned_count(state: SolverState, instance: DARPInstance,
                       header: LogHeader) -> float:
    """len(unassigned) as float. No clipping."""
    return float(len(state.unassigned))


# ---------------------------------------------------------------------------
# Batch: rolling_horizon (3 features, dynamic-only). Surfaces dynamic-DARP
# state that the static-mode model can't observe — the basis for the
# static-vs-dynamic ML ablation in the thesis.
# ---------------------------------------------------------------------------

# First-pass tuning constant for flexibility_horizon. epoch_length * 10 gives
# ~5min of horizon at the standard 30s epoch — enough to span the typical
# pickup-window slack. May need retuning after first training results if the
# distribution turns out to be all-near-0 or all-saturated.
HORIZON_CONSTANT_MULT = 10


@register_feature("committed_travel_time_frac",
                  batch="rolling_horizon", mode="dynamic", tolerance=0.0)
def _committed_travel_time_frac(state: SolverState, instance: DARPInstance,
                                  header: LogHeader) -> float:
    """Fraction of total travel time already locked in by committed prefixes.

    Numerator:   sum over routes r of TT[route[i], route[i+1]] for i in
                 0..min_insert_pos[r] - 2 (the prefix that's locked in by
                 already-departed vehicles).
    Denominator: sum over routes r of TT along all edges.
    Degrades to 0.0 if denominator is 0 (all-empty solution).
    """
    tt = instance.travel_time
    committed_total = 0.0
    overall_total = 0.0
    # min_insert_pos may be empty for non-dynamic states; len() match guards us.
    min_insert_pos = state.min_insert_pos
    for ri, route in enumerate(state.routes):
        # Total along all edges
        for i in range(len(route) - 1):
            overall_total += float(tt[route[i]][route[i + 1]])
        # Committed prefix: edges from 0 up to (but not including) min_insert_pos[r] - 1
        if ri < len(min_insert_pos):
            limit = min_insert_pos[ri] - 1
            for i in range(limit):
                committed_total += float(tt[route[i]][route[i + 1]])
    if overall_total == 0.0:
        return 0.0
    return committed_total / overall_total


@register_feature("horizon_elapsed_frac",
                  batch="rolling_horizon", mode="dynamic", tolerance=0.0)
def _horizon_elapsed_frac(state: SolverState, instance: DARPInstance,
                            header: LogHeader) -> float:
    """Planning-horizon elapsed fraction.

    horizon_elapsed_frac = (sim_clock - sim_start) / (sim_end - sim_start)
    where sim_start = inst.nodes[0].earliest (operational-day start, may be
    non-zero on Fiedler instances) and sim_end = max over requests 1..n of
    inst.nodes[req].latest. Clipped to [0, 1].

    Mirrors Bongiovanni et al. 2022 feature #2 ("planning horizon elapsed [%]").
    Replaces the originally-planned `time_to_next_epoch_norm` which was
    degenerate because sim_clock is frozen during ALNS iterations.
    """
    sim_start = float(instance.nodes[0].earliest)
    sim_end = 0.0
    for req in range(1, instance.num_requests + 1):
        latest = float(instance.nodes[req].latest)
        if latest > sim_end:
            sim_end = latest
    span = sim_end - sim_start
    if span == 0.0:
        return 0.0
    raw = (state.sim_clock - sim_start) / span
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


@register_feature("flexibility_horizon",
                  batch="rolling_horizon", mode="dynamic", tolerance=0.0)
def _flexibility_horizon(state: SolverState, instance: DARPInstance,
                            header: LogHeader) -> float:
    """Mean over active requests of (pickup_tw_start - sim_clock) / HORIZON_CONSTANT.

    Negative values (pickup window already opened) clip to 0.0 — the feature
    represents *remaining* horizon, so already-opened windows contribute zero.
    HORIZON_CONSTANT = epoch_length * HORIZON_CONSTANT_MULT (10 by default);
    a first-pass choice that may need retuning after first training results
    if the distribution is pathological.
    """
    horizon_constant = header.epoch_length * HORIZON_CONSTANT_MULT
    if horizon_constant == 0.0:
        return 0.0
    pickups = _get_active_pickups(state, instance.num_requests)
    if not pickups:
        return 0.0
    total = 0.0
    for nid in pickups:
        raw = float(instance.nodes[nid].earliest) - state.sim_clock
        if raw < 0.0:
            raw = 0.0
        total += raw / horizon_constant
    return total / len(pickups)
