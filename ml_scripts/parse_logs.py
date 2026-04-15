"""Shared log parser for ALNS JSONL logs."""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RunConfig:
    instance: str = ""
    seed: int = 0
    sa_start_temp: float = 0
    sa_end_temp: float = 0
    destroy_ops: List[str] = field(default_factory=list)
    repair_ops: List[str] = field(default_factory=list)
    num_requests: int = 0
    num_vehicles: int = 0
    unassigned_penalty: float = 0
    dynamic: bool = False
    m_trials: int = 0


@dataclass
class IterState:
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
    q: int  # SA quality score
    n_unassigned: int = 0
    n_routes_active: int = 0


@dataclass
class TrialResult:
    iter: int
    obj_before: float
    temperature: float
    # deltas[d_idx * n_repair + r_idx] -> list of deltas across m_trials
    deltas: Dict[tuple, List[float]] = field(default_factory=dict)
    # State context (from new logging)
    n_active_routes: int = 0
    n_unassigned: int = 0
    total_stops: int = 0
    max_route_len: int = 0
    n_vehicles: int = 0


def parse_log(path: str) -> Dict[str, Any]:
    """Parse a JSONL log file into structured data."""
    config = RunConfig()
    states: List[IterState] = []
    trials: List[TrialResult] = []
    weights: List[Dict] = []
    epochs: List[Dict] = []

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            t = rec["type"]

            if t == "header":
                config.instance = rec.get("instance", "")
                config.seed = rec.get("seed", 0)
                config.sa_start_temp = rec.get("sa_start_temp", 0)
                config.sa_end_temp = rec.get("sa_end_temp", 0)
                config.destroy_ops = rec.get("destroy_ops", [])
                config.repair_ops = rec.get("repair_ops", [])
                config.num_requests = rec.get("num_requests", 0)
                config.num_vehicles = rec.get("num_vehicles", 0)
                config.unassigned_penalty = rec.get("unassigned_penalty", 0)
                config.dynamic = rec.get("dynamic", False)
                config.m_trials = rec.get("m_trials", 0)

            elif t == "state":
                routes = rec.get("routes", [])
                active = sum(1 for r in routes if len(r) > 2)
                s = IterState(
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
                    n_unassigned=len(rec.get("unassigned", [])),
                    n_routes_active=active,
                )
                states.append(s)

            elif t == "trials":
                tr = TrialResult(
                    iter=rec["iter"],
                    obj_before=rec["obj_before"],
                    temperature=rec["temperature"],
                    n_active_routes=rec.get("n_active_routes", 0),
                    n_unassigned=rec.get("n_unassigned", 0),
                    total_stops=rec.get("total_stops", 0),
                    max_route_len=rec.get("max_route_len", 0),
                    n_vehicles=rec.get("n_vehicles", 0),
                )
                for entry in rec["results"]:
                    key = (entry["d_idx"], entry["r_idx"])
                    tr.deltas[key] = entry["deltas"]
                trials.append(tr)

            elif t == "weights":
                weights.append(rec)

            elif t == "epoch_start":
                epochs.append(rec)

    return {
        "config": config,
        "states": states,
        "trials": trials,
        "weights": weights,
        "epochs": epochs,
    }
