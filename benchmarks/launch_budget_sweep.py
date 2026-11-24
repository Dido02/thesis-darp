#!/usr/bin/env python3
"""Budget sweep — measure final-obj vs budget for 3 selectors on 4 instances.

For each (mode, instance, selector, budget, seed) tuple runs darp_solver once and
records the final objective + iteration count. Output is a single flat CSV that
plot_budget_sweep.py turns into final-obj-vs-budget curves.

Grid:
  mode      ∈ {dynamic, static}
  instance  ∈ 4 Manhattan delay03 instances (same as v5/v6 bench)
  selector  ∈ {random, roulette, ml/<best classifier>}
  budget    ∈ {200ms, 1000ms, 5000ms, 25000ms} dynamic
             {200,   1000,  5000,   25000   } iters (static)
  seed      ∈ {11, 22, 33, 44, 55}  (disjoint from bench {1,2,3} & training {100+})

Total: 2 × 4 × 3 × 4 × 5 = 480 runs.

Usage:
    python benchmarks/launch_budget_sweep.py                  # both modes, 4 workers
    python benchmarks/launch_budget_sweep.py --mode dynamic   # only dynamic
    python benchmarks/launch_budget_sweep.py --workers 6      # more parallelism
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

INSTANCES_DYNAMIC = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]
# 05_min skipped for static — 25k iters takes ~30 min/run on this instance,
# which would dominate wall time. Dynamic still includes it (wall-bounded).
INSTANCES_STATIC = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
]

# ML choice: 20f_t07 forest with softmax T=0.10. Canonical classifier winner
# across v5/v6 benches. Forest (not tree) because we want to see the budget x
# inference-cost trade-off — forest's inference penalty is more visible at low
# budgets and gets amortized at high budgets.
ML_MODEL_DIR    = REPO / "ml_train" / "models_v3_full_20f_t07"
ML_SOFTMAX_TEMP = 0.10
ML_LABEL        = "ml/20f_t07__T0.10"

# roulette_persist = same roulette wheel but with --persist-roulette-weights,
# so adaptive weights carry across dynamic-mode epochs instead of resetting to
# 1.0 each epoch. Dynamic-only effect; in static mode it's identical to plain
# roulette (single ALNS call, no epoch boundaries).
SELECTORS = ["random", "roulette", "roulette_persist", "ml"]
DYNAMIC_BUDGETS_MS = [200, 1000, 5000, 25000]
STATIC_ITERATIONS  = [200, 1000, 5000, 25000]
SEEDS = [11, 22, 33, 44, 55]

DESTROY = "random,worst,shaw"
REPAIR  = "greedy,regret2,regret3"

_RE_TOTAL_TRAVEL = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_FINAL_OBJ    = re.compile(r"Final objective:\s+(-?\d+(?:\.\d+)?)")
_RE_ITERS_DYN    = re.compile(r"Total ALNS iterations:\s+(\d+)")
_RE_ITERS_STATIC = re.compile(r"Total iterations:\s+(\d+)")


@dataclass
class Run:
    mode: str
    instance: str
    selector: str
    budget: int
    seed: int


def selector_label(sel: str) -> str:
    if sel == "ml":
        return ML_LABEL
    return sel  # "random", "roulette", "roulette_persist"


def build_cmd(r: Run) -> list[str]:
    # roulette_persist runs the solver with --selection roulette plus the
    # --persist-roulette-weights flag so adaptive weights carry across epochs.
    cli_sel = "roulette" if r.selector == "roulette_persist" else r.selector
    cmd = [
        str(SOLVER),
        "--instance", str(DATA / f"{r.instance}.txt"),
        "--seed", str(r.seed),
        "--selection", cli_sel,
        "--penalty", "10000",
        "--destroy", DESTROY,
        "--repair", REPAIR,
    ]
    if r.mode == "dynamic":
        cmd += [
            "--dynamic",
            "--epoch-length", "30",
            "--epoch-budget-ms", str(r.budget),
            "--exclude-selection-from-budget",
        ]
        if r.selector == "roulette_persist":
            cmd.append("--persist-roulette-weights")
    else:
        cmd += ["--iterations", str(r.budget)]
    if r.selector == "ml":
        cmd += [
            "--ml-model-dir", str(ML_MODEL_DIR / "onnx"),
            "--ml-softmax-temp", str(ML_SOFTMAX_TEMP),
        ]
    return cmd


def parse(text: str, mode: str) -> dict:
    if mode == "dynamic":
        m = _RE_TOTAL_TRAVEL.search(text)
        obj = float(m.group(1)) if m else None
        m = _RE_ITERS_DYN.search(text)
        iters = int(m.group(1)) if m else None
    else:
        m = _RE_FINAL_OBJ.search(text)
        obj = float(m.group(1)) if m else None
        m = _RE_ITERS_STATIC.search(text)
        iters = int(m.group(1)) if m else None
    return {"obj": obj, "iters": iters}


def run_one(r: Run, raw_dir: Path, timeout_s: int) -> dict:
    run_id = f"{r.mode}__{r.instance}__{selector_label(r.selector).replace('/', '_')}__b{r.budget}__s{r.seed}"
    log_path = raw_dir / f"{run_id}.log"
    if log_path.exists() and log_path.stat().st_size > 0:
        # Resume: skip if log already there, just re-parse
        text = log_path.read_text()
        parsed = parse(text, r.mode)
        return {**asdict(r), "selector_label": selector_label(r.selector),
                **parsed, "wall_s": None, "exit_code": 0, "run_id": run_id, "cached": True}

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
    parsed = parse(text, r.mode)
    return {**asdict(r), "selector_label": selector_label(r.selector),
            **parsed, "wall_s": round(wall, 2), "exit_code": rc,
            "run_id": run_id, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "bench_results_budget_sweep"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=("both", "dynamic", "static"), default="both")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Per-run timeout in seconds (default 900 = 15 min).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_logs"; raw.mkdir(exist_ok=True)

    runs: list[Run] = []
    pairs = [("dynamic", DYNAMIC_BUDGETS_MS), ("static", STATIC_ITERATIONS)]
    for mode, budgets in pairs:
        if args.mode != "both" and args.mode != mode:
            continue
        insts = INSTANCES_DYNAMIC if mode == "dynamic" else INSTANCES_STATIC
        for inst in insts:
            for sel in SELECTORS:
                # roulette_persist is dynamic-only (no epoch boundary in static
                # → flag is a no-op → results would just duplicate roulette).
                if sel == "roulette_persist" and mode != "dynamic":
                    continue
                for b in budgets:
                    for s in SEEDS:
                        runs.append(Run(mode, inst, sel, b, s))

    print(f"Total runs to execute: {len(runs)}")
    print(f"Workers:               {args.workers}")
    print(f"Output dir:            {out}")
    print(f"ML model dir:          {ML_MODEL_DIR}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, r, raw, args.timeout): r for r in runs}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            results.append(res)
            cached_tag = " [cached]" if res.get("cached") else ""
            obj_str = f"{res['obj']:.1f}" if res['obj'] is not None else "FAIL"
            print(f"  [{i:>3}/{len(runs)}] {res['mode']:>7} | "
                  f"{res['instance'].replace('manhattan_','').replace('_delay03_min',''):>9} | "
                  f"{res['selector_label']:<22} | b={res['budget']:>5} | s={res['seed']:>2} | "
                  f"obj={obj_str:>10} | iters={res['iters']}{cached_tag}")

    # Write flat CSV
    fields = ["mode", "instance", "selector", "selector_label",
              "budget", "seed", "obj", "iters", "wall_s", "exit_code", "run_id", "cached"]
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
