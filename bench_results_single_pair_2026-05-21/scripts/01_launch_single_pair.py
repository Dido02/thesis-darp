#!/usr/bin/env python3
"""Single-(destroy, repair) ablation bench on the 18-00 instances.

For each of 4 single-pair configurations, run ALNS with ONLY that
destroy+repair pair available. With a single pair, the selector choice
is irrelevant (any selector picks the only available pair every time),
so we use --selection random for all runs.

Goal: show that operator diversity matters by comparing each single-pair
ALNS against the 9-pair ALNS in bench_results_final/ (which had
random/roulette baselines using all 9 pairs).

Configs:
  1. op_random_greedy    — random destroy + greedy repair
  2. op_worst_greedy     — worst destroy + greedy repair
  3. op_shaw_greedy      — shaw destroy + greedy repair
  4. op_worst_regret2    — worst destroy + regret2 repair

Grid (matches BENCH_FINAL):
  4 instances {30_s, 01_min, 02_min, 05_min}
  x 3 delays {03, 05, 10}
  x 4 configs (above)
  x 4 budgets {200, 1000, 5000, 10000}
  x 5 seeds {11, 22, 33, 44, 55}
  = 960 runs

Same fair-budget flags as the other benches.
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

INSTANCE_BASES = ["30_s", "01_min", "02_min", "05_min"]
DELAYS         = ["03", "05", "10"]
BUDGETS_MS     = [200, 1000, 5000, 10000]
SEEDS          = [11, 22, 33, 44, 55]

# (selector_tag, destroy_name, repair_name)
SINGLE_PAIR_CONFIGS = [
    ("op_random_greedy",  "random", "greedy"),
    ("op_worst_greedy",   "worst",  "greedy"),
    ("op_shaw_greedy",    "shaw",   "greedy"),
    ("op_worst_regret2",  "worst",  "regret2"),
]
SELECTORS = [c[0] for c in SINGLE_PAIR_CONFIGS]
CONFIG_LOOKUP = {tag: (d, r) for tag, d, r in SINGLE_PAIR_CONFIGS}

_RE_TT  = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_IT  = re.compile(r"Total ALNS iterations:\s+(\d+)")
_RE_BID = re.compile(r"^# BUILD_ID=(\S+)")


@dataclass
class Run:
    instance_base: str
    delay: str
    selector: str          # one of the op_* tags above
    budget: int
    seed: int

    @property
    def instance_file(self) -> str:
        return f"manhattan_{self.instance_base}_delay{self.delay}_min"


def build_cmd(r: Run) -> list[str]:
    destroy, repair = CONFIG_LOOKUP[r.selector]
    return [
        str(SOLVER),
        "--instance", str(DATA / f"{r.instance_file}.txt"),
        "--seed", str(r.seed),
        "--penalty", "10000",
        "--destroy", destroy,
        "--repair",  repair,
        "--dynamic",
        "--epoch-length", "30",
        "--epoch-budget-ms", str(r.budget),
        "--exclude-selection-from-budget",
        "--log-epoch-iters",
        "--selection", "random",   # only 1 pair available; selector is moot
    ]


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
                    default=str(REPO / "bench_results_single_pair_2026-05-21"))
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

    print(f"Single-pair ablation: {len(runs)} runs across {args.workers} workers")
    print(f"Output: {out}")
    print(f"BUILD_ID={BUILD_ID}")
    print(f"Configs: {args.selectors}")
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
