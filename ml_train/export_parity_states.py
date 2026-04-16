"""Export parity fixture: states + Python-computed features for C++ verification.

Reads a JSONL solver log, picks N diverse states (spread across epochs/iterations),
and writes a fixture file with enough info to reconstruct the state in C++ and
compare feature values.

Usage:
    python export_parity_states.py \
        --log ../trail_logs/manhattan_01_min_delay03_min_s456_t50e50_dynamic.jsonl \
        --data ../data/ \
        --n-states 10 \
        --out validation/parity_states.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darpml.features import extract_all, get_feature_names
from darpml.instance import InstanceTooLargeError, load_instance
from darpml.loader import load_log, resolve_instance_path


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


def main():
    ap = argparse.ArgumentParser(description="Export parity fixture")
    ap.add_argument("--log", required=True, help="JSONL log to sample from")
    ap.add_argument("--data", default="../data/", help="Data root for instance files")
    ap.add_argument("--n-states", type=int, default=10)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    feature_names = get_feature_names()

    # Write fixture
    with open(out_path, "w") as f:
        for sid, st in enumerate(picks):
            feats = extract_all(st.state, instance, header)
            # Sanity: ensure all registered features are present
            for name in feature_names:
                assert name in feats, f"Feature {name} missing"

            rec = {
                "state_id": sid,
                "instance_path": str(instance_path.resolve()),
                "epoch_id": st.state.epoch_id,
                "iter": st.state.iter,
                "temperature": st.state.temperature,
                "sa_start_temp": header.sa_start_temp,
                "routes": st.state.routes,
                "unassigned": list(st.state.unassigned),
                "committed": list(st.state.committed),
                "min_insert_pos": list(st.state.min_insert_pos),
                "features": feats,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(picks)} records to {out_path}")
    print(f"Features computed: {feature_names}")


if __name__ == "__main__":
    main()
