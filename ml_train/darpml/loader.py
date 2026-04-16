"""JSONL log parsing and state-trial pairing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class LogHeader:
    instance_path: str
    seed: int
    sa_start_temp: float
    sa_end_temp: float
    sa_step: float
    destroy_ops: list[str]
    repair_ops: list[str]
    unassigned_penalty: float
    num_requests: int
    num_vehicles: int
    m_trials: int
    m_trials_every: int
    dynamic: bool


@dataclass
class SolverState:
    iter: int
    d_idx: int
    r_idx: int
    d_name: str
    r_name: str
    obj_before: float
    obj_candidate: float
    obj_best: float
    delta: float
    temperature: float
    accept: bool
    q: int
    routes: list[list[int]]
    unassigned: list[int]
    committed: list[int]
    min_insert_pos: list[int]
    epoch_id: int = 0  # 0 for non-dynamic logs


@dataclass
class TrialPairResult:
    d_idx: int
    r_idx: int
    deltas: list[float]


@dataclass
class TrialsRecord:
    iter: int
    obj_before: float
    temperature: float
    results: list[TrialPairResult]


@dataclass
class StateTrial:
    """Paired state + trials at the same iteration."""
    state: SolverState
    trials: TrialsRecord


def stream_log(path: str | Path) -> Iterator[tuple[str, dict]]:
    """Yield (record_type, raw_dict) for each line. Streaming, no full-file load."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["type"], rec


def _parse_header(rec: dict) -> LogHeader:
    return LogHeader(
        instance_path=rec.get("instance", ""),
        seed=rec.get("seed", 0),
        sa_start_temp=rec.get("sa_start_temp", 0.0),
        sa_end_temp=rec.get("sa_end_temp", 0.0),
        sa_step=rec.get("sa_step", 1.0),
        destroy_ops=rec.get("destroy_ops", []),
        repair_ops=rec.get("repair_ops", []),
        unassigned_penalty=rec.get("unassigned_penalty", 0.0),
        num_requests=rec.get("num_requests", 0),
        num_vehicles=rec.get("num_vehicles", 0),
        m_trials=rec.get("m_trials", 0),
        m_trials_every=rec.get("m_trials_every", 0),
        dynamic=rec.get("dynamic", False),
    )


def _parse_state(rec: dict) -> SolverState:
    return SolverState(
        iter=rec["iter"],
        d_idx=rec["d_idx"],
        r_idx=rec["r_idx"],
        d_name=rec["d_name"],
        r_name=rec["r_name"],
        obj_before=rec["obj_before"],
        obj_candidate=rec["obj_candidate"],
        obj_best=rec["obj_best"],
        delta=rec["delta"],
        temperature=rec["temperature"],
        accept=rec["accept"],
        q=rec.get("q", 0),
        routes=rec["routes"],
        unassigned=rec.get("unassigned", []),
        committed=rec.get("committed", []),
        min_insert_pos=rec.get("min_insert_pos", []),
    )


def _parse_trials(rec: dict) -> TrialsRecord:
    results = []
    for entry in rec["results"]:
        results.append(TrialPairResult(
            d_idx=entry["d_idx"],
            r_idx=entry["r_idx"],
            deltas=entry["deltas"],
        ))
    return TrialsRecord(
        iter=rec["iter"],
        obj_before=rec["obj_before"],
        temperature=rec["temperature"],
        results=results,
    )


def load_log(path: str | Path) -> tuple[LogHeader, list[StateTrial]]:
    """Parse a JSONL log, pairing state records with their trial records.

    Returns (header, list_of_state_trial_pairs).
    Only iterations that have BOTH a state and trials record are included.
    """
    header: LogHeader | None = None
    states_by_iter: dict[int, SolverState] = {}
    trials_by_iter: dict[int, TrialsRecord] = {}
    current_epoch = 0

    for rtype, rec in stream_log(path):
        if rtype == "header":
            header = _parse_header(rec)
        elif rtype == "epoch_start":
            current_epoch = rec["epoch_id"]
        elif rtype == "state":
            st = _parse_state(rec)
            st.epoch_id = current_epoch
            states_by_iter[st.iter] = st
        elif rtype == "trials":
            tr = _parse_trials(rec)
            trials_by_iter[tr.iter] = tr

    if header is None:
        raise ValueError(f"No header record found in {path}")

    # Pair state + trials by iteration number
    paired: list[StateTrial] = []
    for it in sorted(states_by_iter.keys()):
        if it in trials_by_iter:
            paired.append(StateTrial(
                state=states_by_iter[it],
                trials=trials_by_iter[it],
            ))

    return header, paired


def resolve_instance_path(header_path: str, data_root: str | Path) -> Path:
    """Resolve instance path from log header against data root directory.

    Tries:
    1. {data_root}/fiedler_converted/{basename}
    2. {data_root}/cordeau/{basename}
    3. Raw relative path resolution
    """
    data_root = Path(data_root)
    basename = Path(header_path).name

    # Try fiedler_converted first (most common)
    candidate = data_root / "fiedler_converted" / basename
    if candidate.exists():
        return candidate

    # Try cordeau
    candidate = data_root / "cordeau" / basename
    if candidate.exists():
        return candidate

    # Try raw path relative to data_root parent
    candidate = (data_root / header_path).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot resolve instance '{basename}' under {data_root}. "
        f"Searched fiedler_converted/ and cordeau/."
    )
