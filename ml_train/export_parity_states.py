"""Export parity fixture: states + Python-computed features for C++ verification.

Reads a JSONL solver log, picks N diverse states (spread across epochs/iterations),
and writes a fixture file with enough info to reconstruct the state in C++ and
compare feature values.

Each fixture record carries the per-feature tolerance from the registry, so the
C++ test reads tolerance from a single source of truth (the registry decorator).

Usage:
    python export_parity_states.py \
        --log ../trail_logs/manhattan_01_min_delay03_min_s456_t50e50_dynamic.jsonl \
        --data ../data/ \
        --n-states 10 \
        --mode dynamic \
        --out validation/parity_states.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darpml.features import _REGISTRY, extract_all, get_feature_names
from darpml.instance import InstanceTooLargeError, load_instance
from darpml.loader import load_log, resolve_instance_path
from darpml.reconstruction import compute_objective, compute_route_schedule


def pick_diverse_states(state_trials, n: int, seed: int = 0) -> list:
    """Select n states spanning different epochs + iterations.

    Stratified sampling: group by epoch_id, take proportional count per epoch,
    spread across iter range within each group. Falls back to uniform spread
    if too few records.
    """
    if len(state_trials) <= n:
        return list(state_trials)

    rng = np.random.RandomState(seed)
    # Group by epoch
    by_epoch: dict[int, list] = {}
    for st in state_trials:
        by_epoch.setdefault(st.state.epoch_id, []).append(st)

    # Skip epoch 1 (warmup, as in training)
    usable_epochs = [eid for eid in by_epoch if eid != 1]
    if not usable_epochs:
        usable_epochs = sorted(by_epoch.keys())

    # Allocate states proportionally
    total_usable = sum(len(by_epoch[e]) for e in usable_epochs)
    picks = []
    for eid in usable_epochs:
        group = by_epoch[eid]
        # At least 1 from each epoch, up to proportion
        k = max(1, int(round(n * len(group) / total_usable)))
        if len(group) <= k:
            picks.extend(group)
        else:
            idx = rng.choice(len(group), size=k, replace=False)
            picks.extend([group[i] for i in sorted(idx)])

    # Trim or pad to exactly n
    if len(picks) > n:
        # Stride to keep diversity
        step = len(picks) / n
        picks = [picks[int(i * step)] for i in range(n)]
    return picks


def _parse_batches(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Export parity fixture")
    ap.add_argument("--log", required=True, help="JSONL log to sample from")
    ap.add_argument("--data", default="../data/", help="Data root for instance files")
    ap.add_argument("--n-states", type=int, default=10)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", required=True, choices=["static", "dynamic"],
                    help="Solver mode (filters which features get included)")
    ap.add_argument("--batches", default=None,
                    help="Comma-separated batch names (default: all registered)")
    args = ap.parse_args()

    batches = _parse_batches(args.batches)
    if args.mode == "static" and batches and "rolling_horizon" in batches:
        sys.exit("ERROR: --mode static is incompatible with rolling_horizon batch.")

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(batches=batches, mode=args.mode)
    if not feature_names:
        sys.exit(f"ERROR: no features matched batches={batches} mode={args.mode}")

    feature_tolerances = {n: _REGISTRY[n].tolerance for n in feature_names}

    # Load log
    header, state_trials = load_log(log_path)
    print(f"Loaded {len(state_trials)} state-trial pairs from {log_path.name}")

    # Load instance (needed for feature computation)
    instance_path = resolve_instance_path(header.instance_path, args.data)
    try:
        instance = load_instance(instance_path)
    except InstanceTooLargeError as e:
        print(f"ERROR: {e}")
        return 1

    # Pick diverse states
    picks = pick_diverse_states(state_trials, args.n_states, seed=args.seed)
    print(f"Selected {len(picks)} states (epochs: "
          f"{sorted(set(p.state.epoch_id for p in picks))})")

    print(f"Mode: {args.mode}, Batches: {batches if batches else 'all'}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Write fixture
    with open(out_path, "w") as f:
        for sid, st in enumerate(picks):
            feats = extract_all(st.state, instance, header,
                                batches=batches, mode=args.mode)
            for name in feature_names:
                assert name in feats, f"Feature {name} missing"

            # Per-route ride-time dicts from the Python scheduler. Carried in the
            # fixture so the C++ parity test can verify the scheduler directly
            # (not just via the ride_time_utilization_* features, which mask out
            # committed requests). Keys are stringified for JSON; the C++ test
            # parses them back to ints.
            route_schedules = [
                {str(req): rt for req, rt in compute_route_schedule(r, instance).items()}
                for r in st.state.routes
            ]

            # Single objective computation, fed to BOTH sides via the fixture.
            # Stored as objective_py; C++ test_feature_parity calls sol.objective()
            # independently and compares (the OBJ PASS audit). Features that need
            # obj_before read it from sol.objective() (C++) / state.obj_before
            # (Python) — no second compute path that could drift.
            obj_py = compute_objective(
                st.state.routes, st.state.unassigned,
                instance.travel_time, header.unassigned_penalty,
            )

            rec = {
                "state_id": sid,
                "instance_path": str(instance_path.resolve()),
                "epoch_id": st.state.epoch_id,
                "iter": st.state.iter,
                "temperature": st.state.temperature,
                "sa_start_temp": header.sa_start_temp,
                "unassigned_penalty": header.unassigned_penalty,
                "max_iterations": header.max_iterations,
                "epoch_length": header.epoch_length,
                "q": st.state.q,
                "obj_before": st.state.obj_before,
                "obj_best": st.state.obj_best,
                "obj_initial": st.state.obj_initial,
                "sim_clock": st.state.sim_clock,
                "routes": st.state.routes,
                "unassigned": list(st.state.unassigned),
                "committed": list(st.state.committed),
                "min_insert_pos": list(st.state.min_insert_pos),
                "features": feats,
                "feature_tolerances": feature_tolerances,
                "route_schedules": route_schedules,
                "objective_py": obj_py,
                "mode": args.mode,
                "batches": batches if batches else [],
            }
            f.write(json.dumps(rec) + "\n")

    # Write a sibling manifest matching the feature set so the C++ parity test
    # has a manifest that lines up with the fixture's `features` dict. Lives
    # in <out_path.parent>/manifest/feature_manifest.json so the test can be
    # invoked with that directory as its model_dir argument.
    manifest_dir = out_path.parent / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "mode": args.mode,
        "batches": batches if batches else [],
        "features": [
            {"name": n,
             "batch": _REGISTRY[n].batch,
             "mode": _REGISTRY[n].mode,
             "tolerance": _REGISTRY[n].tolerance}
            for n in feature_names
        ],
        "feature_order": feature_names,
        "n_features": len(feature_names),
        "model_version": f"parity-fixture-{args.mode}-{'-'.join(batches) if batches else 'all'}",
        # Operators are unused by the parity test but the manifest schema requires them.
        "destroy_ops": ["random", "worst", "shaw", "rtcritical", "lowslack"],
        "repair_ops": ["greedy", "regret2", "regret3", "slack"],
    }
    manifest_path = manifest_dir / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(fm, f, indent=2)
    print(f"Wrote sibling manifest to {manifest_path}")

    print(f"Wrote {len(picks)} records to {out_path}")


if __name__ == "__main__":
    main()
