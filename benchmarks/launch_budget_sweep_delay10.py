#!/usr/bin/env python3
"""Budget sweep on the **delay10** Manhattan instances (no other bench has hit
these — training data covered all delays but eval was always delay03).

Same shape as launch_budget_sweep.py but:
  - delay10 instances instead of delay03
  - 3 base ML models x {proportional, T=0.10} = 6 ML variants:
      * 20f_t05            — best final-obj ML in v6 forest fair (-0.028% vs random)
      * 20f_ei_log         — best per-iter new-best lift in v6 (+0.67pp Holm-sig)
      * 20f_t07            — canonical classifier from delay03 sweep
    Each gets a proportional (no softmax) and a T=0.10 row so the softmax
    effect is directly visible across budgets.
  - --log-epoch-iters ON for every run (within-epoch trajectories captured)
  - dynamic mode only
  - separate out dir (bench_results_budget_sweep_delay10) — no clobber

Grid: 4 instances x 8 selectors x 4 budgets x 5 seeds = 640 runs.

Usage:
    python benchmarks/launch_budget_sweep_delay10.py            # 4 workers default
    python benchmarks/launch_budget_sweep_delay10.py --workers 6
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

REPO  = Path(__file__).resolve().parent.parent
SOLVER = REPO / "cpp" / "build" / "darp_solver"
DATA   = REPO / "data" / "fiedler_converted"

INSTANCES = [
    "manhattan_30_s_delay10_min",
    "manhattan_01_min_delay10_min",
    "manhattan_02_min_delay10_min",
    "manhattan_05_min_delay10_min",
]

# (label, dir_name, softmax_temp)
# Each base model gets a proportional (no softmax) and a T=0.10 variant so we
# can see softmax effect head-to-head. T=0.10 was the strongest softmax setting
# in v6 forest fair (T=0.05 always under-performed both).
ML_VARIANTS = [
    ("ml_20f_t05",            "models_v3_full_20f_t05",    None),   # best ML final-obj in v6 forest fair
    ("ml_20f_t05__T0.10",     "models_v3_full_20f_t05",    0.10),   # softmax variant of the best
    ("ml_20f_ei_log",         "models_v3_full_20f_ei_log", None),   # best per-iter new-best lift
    ("ml_20f_ei_log__T0.10",  "models_v3_full_20f_ei_log", 0.10),   # second-best ML final-obj in v6 forest fair
    ("ml_20f_t07",            "models_v3_full_20f_t07",    None),   # canonical classifier (proportional)
    ("ml_20f_t07__T0.10",     "models_v3_full_20f_t07",    0.10),   # apples-to-apples with delay03 sweep
]

# Order: baselines first, ML variants second (so the queue runs baselines
# while the more expensive ML jobs queue up).
# roulette_persist = roulette with --persist-roulette-weights (adaptive weights
# carry across dynamic-mode epochs). Same wall cost as roulette.
SELECTORS = ["random", "roulette", "roulette_persist"] + [v[0] for v in ML_VARIANTS]
ML_LOOKUP = {label: (dir_name, softmax) for label, dir_name, softmax in ML_VARIANTS}

BUDGETS_MS = [200, 1000, 5000, 25000]
SEEDS = [11, 22, 33, 44, 55]

DESTROY = "random,worst,shaw"
REPAIR  = "greedy,regret2,regret3"

_RE_TOTAL_TRAVEL = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_ITERS_DYN    = re.compile(r"Total ALNS iterations:\s+(\d+)")


@dataclass
class Run:
    instance: str
    selector: str   # one of SELECTORS
    budget: int
    seed: int


def build_cmd(r: Run) -> list[str]:
    cmd = [
        str(SOLVER),
        "--instance", str(DATA / f"{r.instance}.txt"),
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
    if r.selector == "roulette_persist":
        cmd += ["--selection", "roulette", "--persist-roulette-weights"]
    elif r.selector in ("random", "roulette"):
        cmd += ["--selection", r.selector]
    else:
        dir_name, softmax = ML_LOOKUP[r.selector]
        cmd += ["--selection", "ml",
                "--ml-model-dir", str(REPO / "ml_train" / dir_name / "onnx")]
        if softmax is not None:
            cmd += ["--ml-softmax-temp", str(softmax)]
    return cmd


def parse(text: str) -> dict:
    mt = _RE_TOTAL_TRAVEL.search(text)
    mi = _RE_ITERS_DYN.search(text)
    return {
        "obj":   float(mt.group(1)) if mt else None,
        "iters": int(mi.group(1))   if mi else None,
    }


def run_one(r: Run, raw_dir: Path, timeout_s: int) -> dict:
    run_id = f"dynamic__{r.instance}__{r.selector}__b{r.budget}__s{r.seed}"
    log_path = raw_dir / f"{run_id}.log"
    if log_path.exists() and log_path.stat().st_size > 0:
        text = log_path.read_text()
        parsed = parse(text)
        return {**asdict(r), **parsed, "wall_s": None, "exit_code": 0,
                "run_id": run_id, "cached": True}

    cmd = build_cmd(r)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
        text = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        text = f"TIMEOUT after {timeout_s}s\n"
        rc = -2
    except Exception as e:
        text = f"FAIL: {e}\n"
        rc = -1
    wall = time.monotonic() - t0
    log_path.write_text(text)
    parsed = parse(text)
    return {**asdict(r), **parsed, "wall_s": round(wall, 2), "exit_code": rc,
            "run_id": run_id, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "bench_results_budget_sweep_delay10"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400,
                    help="Per-run timeout in seconds (default 2400 = 40 min). "
                         "Heavy ML (ei_log, 24 MB ONNX) at b=25000ms can take ~25 min on the "
                         "larger instances; 900s was too tight and produced spurious TIMEOUTs.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_logs"; raw.mkdir(exist_ok=True)

    runs: list[Run] = []
    for inst in INSTANCES:
        for sel in SELECTORS:
            for b in BUDGETS_MS:
                for s in SEEDS:
                    runs.append(Run(inst, sel, b, s))

    print(f"Total runs:    {len(runs)}")
    print(f"Workers:       {args.workers}")
    print(f"Output dir:    {out}")
    print(f"Selectors:     {SELECTORS}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, r, raw, args.timeout): r for r in runs}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            results.append(res)
            tag = " [cached]" if res.get("cached") else ""
            obj = f"{res['obj']:.1f}" if res['obj'] is not None else "FAIL"
            print(f"  [{i:>3}/{len(runs)}] "
                  f"{res['instance'].replace('manhattan_','').replace('_delay10_min',''):>9} | "
                  f"{res['selector']:<22} | b={res['budget']:>5} | s={res['seed']:>2} | "
                  f"obj={obj:>10}{tag}")

    fields = ["instance", "selector", "budget", "seed",
              "obj", "iters", "wall_s", "exit_code", "run_id", "cached"]
    csv_path = out / "results.csv"
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {csv_path} ({len(results)} rows)")
    print(f"Failures: {sum(1 for r in results if r['obj'] is None)}")


if __name__ == "__main__":
    main()
