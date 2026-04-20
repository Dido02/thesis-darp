"""Construct a small hand-crafted fixture that exercises edge-case branches the
trajectory-derived fixtures don't naturally hit.

Output: ml_train/validation/synthetic/synthetic_edge_cases.jsonl
        ml_train/validation/synthetic/manifest/feature_manifest.json
        ml_train/validation/synthetic/README.md

Each state targets specific code paths in the feature/scheduler implementation.
The build script computes truth values by running the Python features on each
state, so the fixture is always self-consistent. The C++ side is verified to
match these values bit-for-bit by test_feature_parity.

Run:
    cd ml_train && python build_synthetic_fixture.py \
        --instance ../data/cordeau/pr01.txt \
        --batches core,tightness --mode dynamic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from darpml.features import _REGISTRY, extract_all, get_feature_names
from darpml.instance import load_instance
from darpml.loader import LogHeader, SolverState
from darpml.reconstruction import compute_objective, compute_route_schedule


def make_header(instance_path: str, max_iterations: int = 1000,
                  epoch_length: float = 30.0) -> LogHeader:
    """Synthetic log header. sa_start_temp tracks the per-state temperature so
    temperature_normalized always lands on a known value. max_iterations and
    epoch_length are overridable for edge-case tests."""
    return LogHeader(
        instance_path=instance_path,
        seed=0,
        sa_start_temp=100.0,
        sa_end_temp=1.0,
        sa_step=0.99,
        destroy_ops=["random", "worst", "shaw", "rtcritical", "lowslack"],
        repair_ops=["greedy", "regret2", "regret3", "slack"],
        unassigned_penalty=1000.0,
        num_requests=24,
        num_vehicles=3,
        m_trials=0,
        m_trials_every=0,
        dynamic=True,
        max_iterations=max_iterations,
        epoch_length=epoch_length,
    )


def make_state(
    iter_id: int,
    routes: list[list[int]],
    committed: list[int],
    unassigned: list[int],
    temperature: float,
    q: int = 0,
    obj_best: float = 0.0,
    obj_initial: float = 0.0,
    sim_clock: float = 0.0,
    min_insert_pos: list[int] | None = None,
) -> SolverState:
    return SolverState(
        iter=iter_id,
        d_idx=0, r_idx=0, d_name="", r_name="",
        obj_before=0.0,  # filled in below from compute_objective
        obj_candidate=0.0,
        obj_best=obj_best,
        delta=0.0,
        temperature=temperature,
        accept=False,
        q=q,
        routes=routes,
        unassigned=unassigned,
        committed=committed,
        min_insert_pos=min_insert_pos if min_insert_pos is not None
                        else [0] * len(routes),
        epoch_id=2,
        obj_initial=obj_initial,
        sim_clock=sim_clock,
    )


# Cordeau pr01 layout: depot=0, pickups 1..24, deliveries 25..48, dest_depot=49.
# K=3 vehicles, no vehicle_start_nodes => empty_route_size=2, empty layout [0, 49].
DEPOT = 0
DEST = 49


def empty_route() -> list[int]:
    return [DEPOT, DEST]


def single_request_route(pickup: int) -> list[int]:
    return [DEPOT, pickup, pickup + 24, DEST]


def build_states() -> list[tuple[str, str, SolverState]]:
    """Return a list of (state_label, description, SolverState).

    state_label is a short tag used in error messages; description is for the README.
    """
    states: list[tuple[str, str, SolverState]] = []

    # ---------- State 0: ALL ROUTES EMPTY ----------
    # Every route is the depot+dest_depot pair; no pickups, no deliveries.
    # Exercises:
    #   - tightness empty-active branch (ride_time_utilization_avg/max -> 0.0,
    #     near_violation_count -> 0)
    #   - tightness empty-non-empty-routes branch (route_slack_avg/std -> 0.0)
    #   - core: num_active_requests=0, radius_avg_tt=0 (n<=1 active), frac_committed=0/0=0,
    #     avg_route_load=0 (no non-empty routes)
    states.append((
        "all_routes_empty",
        "Every vehicle has only [depot, dest_depot]. Tests every empty-set branch.",
        make_state(
            iter_id=0,
            routes=[empty_route(), empty_route(), empty_route()],
            committed=[],
            unassigned=list(range(1, 25)),  # all 24 requests in the bank
            temperature=50.0,  # temperature_normalized = 0.5
        ),
    ))

    # ---------- State 1: SINGLE NON-EMPTY ROUTE WITH ONE REQUEST ----------
    # Tests route_slack_std == 0.0 single-sample branch (only one non-empty
    # route, len(slacks) == 1).
    states.append((
        "single_route_one_request",
        "One non-empty route carrying request 1, two empty routes. Tests "
        "route_slack_std single-sample (len==1 -> 0.0) and ride_time_utilization "
        "single-element max == avg.",
        make_state(
            iter_id=1,
            routes=[
                single_request_route(1),
                empty_route(),
                empty_route(),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r != 1],
            temperature=25.0,  # temperature_normalized = 0.25
        ),
    ))

    # ---------- State 2: ZERO ACTIVE REQUESTS BUT NON-EMPTY ROUTES ----------
    # All requests in routes are also in `committed`. Tests the active-set-empty
    # branch while routes are non-empty (route_slack_* still produce real values).
    states.append((
        "all_committed",
        "Two routes carry one request each; both are in `committed`. Active set "
        "is empty, but route_slack_* still iterate non-empty routes.",
        make_state(
            iter_id=2,
            routes=[
                single_request_route(1),
                single_request_route(2),
                empty_route(),
            ],
            committed=[1, 2],
            unassigned=[r for r in range(3, 25)],
            temperature=10.0,  # temperature_normalized = 0.1
        ),
    ))

    # ---------- State 3: SEVERAL ACTIVE REQUESTS ----------
    # Multiple single-request routes -> several utilization values to exercise
    # the threshold comparison (>= NEAR_VIOLATION_THRESHOLD). Pick requests with
    # different deliveries so ride times differ. The build script reports the
    # actual utilization values; what matters is that the strict-comparison
    # behavior matches between Python and C++.
    states.append((
        "threshold_spread",
        "Three single-request routes with different requests. Utilizations vary, "
        "exercising the >= 0.9 threshold comparison in near_violation_count and "
        "the population-stddev path with k=3 samples.",
        make_state(
            iter_id=3,
            routes=[
                single_request_route(1),
                single_request_route(7),
                single_request_route(13),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r not in (1, 7, 13)],
            temperature=75.0,
        ),
    ))

    # ---------- State 4: HEAVY ROUTE (interleaved pickups/deliveries) ----------
    # One vehicle does several requests in sequence. Tests the non-trivial
    # scheduler path (Step 3 iterative ride-time minimization is more likely to
    # do work) and gives a state with multiple ride-time-bearing requests AND
    # multiple non-empty routes. obj_best/obj_initial set so cost_vs_* features
    # produce non-zero, well-behaved values.
    states.append((
        "interleaved_heavy_route",
        "One route doing 3 requests interleaved (pickup, pickup, deliver, "
        "pickup, deliver, deliver) plus one single-request route. Exercises "
        "Step 3 of the scheduler more thoroughly than single-pickup routes. "
        "obj_best below obj_before triggers the SA-worsening case for "
        "cost_vs_incumbent_frac (positive ratio).",
        make_state(
            iter_id=400,           # iteration_norm = 0.4 (max=1000)
            routes=[
                [DEPOT, 1, 2, 25, 3, 26, 27, DEST],
                single_request_route(20),
                empty_route(),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r not in (1, 2, 3, 20)],
            temperature=99.0,
            q=2,                   # arbitrary, exercises q / q_max ratio
            obj_best=20000.0,      # below obj_before to trigger SA-worsening branch
            obj_initial=22000.0,   # below obj_before (positive cost_vs_initial_frac)
        ),
    ))

    # ---------- State 5: obj_initial == 0 degrade branch ----------
    # Tests cost_vs_initial_frac fallback when obj_initial is zero (pre-epoch
    # state or unset). cost_vs_incumbent_frac still produces a real value.
    states.append((
        "obj_initial_zero",
        "Same routes as state 4 but obj_initial = 0. Tests cost_vs_initial_frac "
        "degrade-to-0 path. cost_vs_incumbent_frac still computes from obj_best.",
        make_state(
            iter_id=100,
            routes=[
                [DEPOT, 1, 2, 25, 3, 26, 27, DEST],
                single_request_route(20),
                empty_route(),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r not in (1, 2, 3, 20)],
            temperature=50.0,
            q=1,
            obj_best=80.0,         # well below obj_before
            obj_initial=0.0,       # triggers the degrade branch
        ),
    ))

    # ---------- State 6: max_iterations == 0 degrade branch ----------
    # Tests iteration_norm fallback when max_iterations is zero (e.g., a
    # time-limited run before SA calibration sets a real estimate). Uses a
    # custom header so this state lives within its own iteration budget.
    states.append((
        "max_iterations_zero",
        "Standard interleaved route but log header has max_iterations=0. "
        "Tests iteration_norm degrade-to-0 path. Other lns_state features "
        "are unaffected.",
        make_state(
            iter_id=50,
            routes=[
                [DEPOT, 1, 2, 25, 3, 26, 27, DEST],
                single_request_route(20),
                empty_route(),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r not in (1, 2, 3, 20)],
            temperature=70.0,
            q=1,
            obj_best=80.0,
            obj_initial=82.0,
        ),
    ))

    # ---------- State 7: obj_before below obj_best (SA-already-improved) ----------
    # When the previous iteration's accept improved obj_best, the next
    # iteration's obj_before == obj_best. This exercises the cost_vs_incumbent
    # zero-numerator (not zero-denominator) path: ratio = 0 cleanly.
    states.append((
        "obj_before_equals_obj_best",
        "obj_before == obj_best (just improved last accept). "
        "cost_vs_incumbent_frac should be exactly 0.0 (zero numerator).",
        make_state(
            iter_id=200,
            routes=[
                single_request_route(5),
                single_request_route(10),
                empty_route(),
            ],
            committed=[],
            unassigned=[r for r in range(1, 25) if r not in (5, 10)],
            temperature=40.0,
            q=1,
            obj_best=0.0,          # set below to obj_before
            obj_initial=2050.0,    # ratio still meaningful
        ),
    ))

    # ---------- State 8: rolling_horizon mid-horizon with committed prefix ----------
    # Tests:
    # - horizon_elapsed_frac: sim_clock at the midpoint of [0, 1440] -> 0.5
    # - committed_travel_time_frac: route 0 has min_insert_pos=3, so the
    #   first 2 edges (depot->1, 1->25) are part of the committed prefix.
    #   committed_total = TT[0][1] + TT[1][25]. overall_total includes all edges.
    # - flexibility_horizon: 2 active requests; their pickup tw starts are at 0
    #   on Cordeau, so (0 - 720) clips to 0.0 -> contribution = 0.
    states.append((
        "rolling_mid_horizon",
        "sim_clock at midpoint of operational day; route 0 has 2 committed "
        "prefix edges. Tests horizon_elapsed_frac (= 0.5), committed_travel_time_frac "
        "(prefix-sum > 0), and flexibility_horizon (clip-to-0 for past-window).",
        make_state(
            iter_id=300,
            routes=[
                [DEPOT, 1, 25, 2, 26, DEST],   # 5 edges total
                single_request_route(7),         # 3 edges
                empty_route(),                   # 1 edge
            ],
            committed=[],                        # active set NOT empty
            unassigned=[r for r in range(1, 25) if r not in (1, 2, 7)],
            temperature=20.0,
            q=1,
            obj_best=300.0,
            obj_initial=350.0,
            sim_clock=720.0,                     # midpoint of [0, 1440]
            min_insert_pos=[3, 0, 0],            # route 0: 2 committed edges
        ),
    ))

    # ---------- State 9: sim_clock past sim_end (clip to 1.0 + neg-clip in flexibility) ----------
    # Tests:
    # - horizon_elapsed_frac upper-clip path (sim_clock > sim_end -> 1.0)
    # - flexibility_horizon: all pickup windows already opened; each
    #   contribution clips to 0 -> mean = 0
    # - committed_travel_time_frac: zero denominator path (all-empty solution).
    states.append((
        "sim_clock_past_horizon",
        "sim_clock past the operational-day end with all routes empty. "
        "Tests horizon_elapsed_frac upper-clip (-> 1.0), flexibility_horizon "
        "all-clipped path (every contribution -> 0), and "
        "committed_travel_time_frac zero-denominator path.",
        make_state(
            iter_id=999,
            routes=[empty_route(), empty_route(), empty_route()],
            committed=[],
            unassigned=list(range(1, 25)),
            temperature=1.0,
            q=0,
            obj_best=24000.0,
            obj_initial=24000.0,
            sim_clock=2000.0,                    # past sim_end (1440)
        ),
    ))

    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True,
                    help="Path to a small instance (Cordeau pr01 recommended)")
    ap.add_argument("--batches", default="core,tightness")
    ap.add_argument("--mode", default="dynamic", choices=["static", "dynamic"])
    ap.add_argument("--out-dir", default="validation/synthetic",
                    help="Output directory (will hold .jsonl, manifest/, README.md)")
    args = ap.parse_args()

    instance_path = Path(args.instance).resolve()
    instance = load_instance(instance_path)
    assert instance.num_requests == 24, \
        f"This fixture is hand-tuned for pr01 (24 requests); got {instance.num_requests}"

    batches = [b.strip() for b in args.batches.split(",") if b.strip()]
    feature_names = get_feature_names(batches=batches, mode=args.mode)
    feature_tolerances = {n: _REGISTRY[n].tolerance for n in feature_names}
    print(f"Mode: {args.mode}, Batches: {batches}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest").mkdir(exist_ok=True)

    header = make_header(str(instance_path))

    states = build_states()
    print(f"\nBuilding {len(states)} synthetic states from {instance_path.name}\n")

    fixture_path = out_dir / "synthetic_edge_cases.jsonl"
    descriptions: list[tuple[str, str, dict]] = []  # (label, description, summary)

    with open(fixture_path, "w") as f:
        for sid, (label, description, st) in enumerate(states):
            # Compute obj_before from routes so cost_vs_* features have a
            # real comparand. State-specific overrides (e.g. state 7's
            # obj_best == obj_before, state 6's max_iterations == 0) are
            # applied here.
            obj_py = compute_objective(
                st.routes, st.unassigned,
                instance.travel_time, header.unassigned_penalty,
            )
            st.obj_before = obj_py
            if label == "obj_before_equals_obj_best":
                st.obj_best = obj_py
            # State 6 needs a header with max_iterations=0 to exercise the
            # iteration_norm degrade branch; other states use the default header.
            state_header = (
                make_header(str(instance_path), max_iterations=0)
                if label == "max_iterations_zero"
                else header
            )

            feats = extract_all(st, instance, state_header,
                                batches=batches, mode=args.mode)
            route_schedules = [
                {str(req): rt for req, rt in compute_route_schedule(r, instance).items()}
                for r in st.routes
            ]
            rec = {
                "state_id": sid,
                "state_label": label,
                "description": description,
                "instance_path": str(instance_path),
                "epoch_id": st.epoch_id,
                "iter": st.iter,
                "temperature": st.temperature,
                "sa_start_temp": header.sa_start_temp,
                "unassigned_penalty": header.unassigned_penalty,
                "max_iterations": state_header.max_iterations,
                "epoch_length": state_header.epoch_length,
                "q": st.q,
                "obj_before": st.obj_before,
                "obj_best": st.obj_best,
                "obj_initial": st.obj_initial,
                "sim_clock": st.sim_clock,
                "routes": st.routes,
                "unassigned": list(st.unassigned),
                "committed": list(st.committed),
                "min_insert_pos": list(st.min_insert_pos),
                "features": feats,
                "feature_tolerances": feature_tolerances,
                "route_schedules": route_schedules,
                "objective_py": obj_py,
                "mode": args.mode,
                "batches": batches,
            }
            f.write(json.dumps(rec) + "\n")

            # Capture summary for README + console.
            n_active = int(feats.get("num_active_requests", 0))
            n_nonempty_routes = sum(1 for r in st.routes
                                     if len(r) > instance.empty_route_size)
            print(f"State {sid} [{label}]:")
            print(f"  description: {description}")
            print(f"  routes={len(st.routes)} (non-empty={n_nonempty_routes}), "
                  f"active={n_active}, committed={len(st.committed)}")
            for fn in feature_names:
                print(f"    {fn:32s} = {feats[fn]:.6f}")
            print()
            descriptions.append((label, description, dict(
                state_id=sid,
                routes_count=len(st.routes),
                non_empty_routes=n_nonempty_routes,
                active=n_active,
                committed=len(st.committed),
                features=feats,
            )))

    # Write a sibling manifest matching the feature set.
    manifest = {
        "mode": args.mode,
        "batches": batches,
        "features": [
            {"name": n,
             "batch": _REGISTRY[n].batch,
             "mode": _REGISTRY[n].mode,
             "tolerance": _REGISTRY[n].tolerance}
            for n in feature_names
        ],
        "feature_order": feature_names,
        "n_features": len(feature_names),
        "model_version": "synthetic-edge-cases",
        "destroy_ops": ["random", "worst", "shaw", "rtcritical", "lowslack"],
        "repair_ops": ["greedy", "regret2", "regret3", "slack"],
    }
    with open(out_dir / "manifest" / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write README capturing intent, contents, and the test-bug caveat.
    readme = [
        "# Synthetic edge-case parity fixture",
        "",
        "Hand-crafted states that exercise empty-set, single-sample, and",
        "threshold-boundary branches the trajectory-derived parity fixtures",
        "don't naturally hit. Each state's `features` dict is computed by the",
        "Python implementation; the C++ side is verified to match bit-for-bit",
        "by `test_feature_parity` against the sibling manifest in `manifest/`.",
        "",
        "Generated by: `ml_train/build_synthetic_fixture.py`",
        f"Instance: `{instance_path.name}` (Cordeau pr01: 24 requests, 3 vehicles, L={instance.max_ride_time})",
        f"Mode: `{args.mode}`  Batches: `{','.join(batches)}`",
        f"Features ({len(feature_names)}): {', '.join(feature_names)}",
        "",
        "## States",
        "",
    ]
    for label, description, summary in descriptions:
        readme.append(f"### State {summary['state_id']} — `{label}`")
        readme.append("")
        readme.append(description)
        readme.append("")
        readme.append(
            f"- routes: {summary['routes_count']} "
            f"(non-empty: {summary['non_empty_routes']})"
        )
        readme.append(f"- active requests: {summary['active']}")
        readme.append(f"- committed requests: {summary['committed']}")
        readme.append("")
        readme.append("Feature values:")
        readme.append("")
        readme.append("```")
        for fn in feature_names:
            readme.append(f"{fn:32s} = {summary['features'][fn]}")
        readme.append("```")
        readme.append("")

    readme.extend([
        "## Test-harness caveat",
        "",
        "`test_feature_parity.cpp` MUST clear `sol.routes` and `sol.unassigned`",
        "before populating from the fixture (see [test_feature_parity.cpp:113-115]",
        "(../../../cpp/tests/test_feature_parity.cpp#L113-L115)). DARPSolution's",
        "default ctor pre-fills `routes` with K depot+vstart+depot empties and",
        "seeds `unassigned` with `1..n`. If the test appends instead of replacing,",
        "the synthetic 'all_routes_empty' state would paradoxically show K extra",
        "default empty routes, and committed-only / single-route states would",
        "have stray default routes interleaved. The trajectory-derived fixtures",
        "happen to mask this (empty routes are no-ops for features) but the",
        "synthetic fixture would expose it immediately.",
        "",
        "## How to regenerate",
        "",
        "```",
        "cd ml_train",
        "python build_synthetic_fixture.py --instance ../data/cordeau/pr01.txt \\",
        f"  --batches {','.join(batches)} --mode {args.mode}",
        "```",
        "",
        "## Verification",
        "",
        "Run alongside the trajectory fixture in every Part:",
        "",
        "```",
        "cd cpp/build",
        f"./test_feature_parity ../../{fixture_path} \\",
        f"  ../../{out_dir}/manifest/",
        "```",
        "",
        "Expected: `SCHED PASS` and `PASS: all N features match` at",
        "`max_abs_err=0` (or 1e-12 for std-dev features).",
    ])
    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(readme) + "\n")

    print(f"Wrote {len(states)} states to {fixture_path}")
    print(f"Wrote manifest to {out_dir / 'manifest' / 'feature_manifest.json'}")
    print(f"Wrote README to {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
