#!/usr/bin/env python3
"""Bench launcher for the 07-00 (morning) Manhattan instances.

Tests whether the deployed t07 forest, the new t07 variants, and the
baselines generalise to a different time-of-day cohort that we have
NOT trained on.

Selectors (7):
  - random
  - roulette
  - ml_20f_t07__T0.10               (the deployed model)
  - ml_forest_d8_K20__T0.10         (depth-cap forest, K=20)
  - ml_forest_d8_K16__T0.10         (depth-cap forest, K=16)
  - ml_tree_d4_K4__T0.10            (best single tree)
  - ml_tree_d8_K16__T0.10           (second tree variant)

Grid:
  5 instances {morning_30_s, morning_01_min, morning_02_min, morning_05_min, morning_15_min}
  x 3 delays {03, 05, 10}
  x 7 selectors
  x 4 budgets {200, 1000, 5000, 10000}
  x 5 seeds {11, 22, 33, 44, 55}
  = 2100 runs

Same fair-budget flags as BENCH_FINAL_2026-05-18:
  --exclude-selection-from-budget --log-epoch-iters
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

REPO   = Path(__file__).resolve().parents[2]
SOLVER = REPO / "cpp" / "build" / "darp_solver"
DATA   = REPO / "data" / "fiedler_converted"

BUILD_ID = str(int(SOLVER.stat().st_mtime)) if SOLVER.exists() else "missing"

INSTANCE_BASES = ["morning_30_s", "morning_01_min", "morning_02_min",
                  "morning_05_min", "morning_15_min"]
DELAYS         = ["03", "05", "10"]
BUDGETS_MS     = [200, 1000, 5000, 10000]
SEEDS          = [11, 22, 33, 44, 55]

ML_VARIANTS = [
    ("ml_20f_t07__T0.10",        "models_v3_full_20f_t07",      0.10),
    ("ml_forest_d8_K20__T0.10",  "models_t07_forest_d8_K20",    0.10),
    ("ml_forest_d8_K16__T0.10",  "models_t07_forest_d8_K16",    0.10),
    ("ml_tree_d4_K4__T0.10",     "models_t07_tree_d4_K4",       0.10),
    ("ml_tree_d8_K16__T0.10",    "models_t07_tree_d8_K16",      0.10),
]
SELECTORS = ["random", "roulette"] + [v[0] for v in ML_VARIANTS]
ML_LOOKUP = {label: (dir_name, softmax) for label, dir_name, softmax in ML_VARIANTS}

DESTROY = "random,worst,shaw"
REPAIR  = "greedy,regret2,regret3"

_RE_TT = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_IT = re.compile(r"Total ALNS iterations:\s+(\d+)")
_RE_BID = re.compile(r"^# BUILD_ID=(\S+)")


@dataclass
class Run:
    instance_base: str
    delay: str
    selector: str
    budget: int
    seed: int

    @property
    def instance_file(self) -> str:
        return f"manhattan_{self.instance_base}_delay{self.delay}_min"


def build_cmd(r: Run) -> list[str]:
    cmd = [
        str(SOLVER),
        "--instance", str(DATA / f"{r.instance_file}.txt"),
        "--seed", str(r.seed),
        "--penalty", "10000",
        "--destroy", DESTROY,
        "--repair", REPAIR,
        "--dynamic",
        "--epoch-length", "30",
        "--epoch-budget-ms", str(r.budget),
        "--exclude-selection-from-budget",
        "--log-epoch-iters",
    ]
    if r.selector == "random":
        cmd += ["--selection", "random"]
    elif r.selector == "roulette":
        cmd += ["--selection", "roulette"]
    else:
        dir_name, softmax = ML_LOOKUP[r.selector]
        cmd += ["--selection", "ml",
                "--ml-model-dir", str(REPO / "ml_train" / dir_name / "onnx")]
        if softmax is not None:
            cmd += ["--ml-softmax-temp", str(softmax)]
    return cmd


def parse(text: str) -> dict:
    mt = _RE_TT.search(text); mi = _RE_IT.search(text)
    return {"obj": float(mt.group(1)) if mt else None,
            "iters": int(mi.group(1)) if mi else None}


def cached_log_is_valid(log_path: Path) -> bool:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False
    with open(log_path) as f:
        first = f.readline()
    m = _RE_BID.match(first)
    if m is None:
        return True
    return m.group(1) == BUILD_ID


def run_one(r: Run, raw_dir: Path, timeout_s: int) -> dict:
    run_id = (f"dynamic__{r.instance_file}__{r.selector}"
              f"__b{r.budget}__s{r.seed}")
    log_path = raw_dir / f"{run_id}.log"

    if cached_log_is_valid(log_path):
        text = log_path.read_text()
        parsed = parse(text)
        return {**asdict(r), **parsed, "wall_s": None, "exit_code": 0,
                "run_id": run_id, "cached": True}

    cmd = build_cmd(r)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              check=False, timeout=timeout_s)
        text = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        text = f"TIMEOUT after {timeout_s}s\n"
        rc = -2
    except Exception as e:
        text = f"FAIL: {e}\n"
        rc = -1
    wall = time.monotonic() - t0

    log_path.write_text(f"# BUILD_ID={BUILD_ID}\n{text}")
    parsed = parse(text)
    return {**asdict(r), **parsed, "wall_s": round(wall, 2), "exit_code": rc,
            "run_id": run_id, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=str(REPO / "bench_results_morning_2026-05-21"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--delays", nargs="*", default=DELAYS)
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS_MS)
    ap.add_argument("--instances", nargs="*", default=INSTANCE_BASES)
    ap.add_argument("--selectors", nargs="*", default=SELECTORS)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_logs"; raw.mkdir(exist_ok=True)

    runs: list[Run] = []
    for inst in args.instances:
        for delay in args.delays:
            for sel in args.selectors:
                for b in args.budgets:
                    for s in SEEDS:
                        runs.append(Run(inst, delay, sel, b, s))

    print(f"Morning bench: {len(runs)} runs across {args.workers} workers")
    print(f"Output: {out}")
    print(f"BUILD_ID={BUILD_ID}")
    print(f"Selectors: {args.selectors}")
    print()

    t_start = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, r, raw, args.timeout): r for r in runs}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            tag = "cache" if res.get("cached") else f"{res.get('wall_s', '?')}s"
            elapsed = time.monotonic() - t_start
            print(f"[{i}/{len(runs)}] {res['run_id']} -> obj={res['obj']} "
                  f"iters={res['iters']} ({tag}, elapsed {elapsed:.0f}s)")

    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nTotal wall: {time.monotonic() - t_start:.0f}s")
    print(f"Wrote {csv_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
