#!/usr/bin/env python3
"""Final fresh bench — delay-stratified, single binary, 1440 runs.

Grid (all dynamic, fair budget, --log-epoch-iters on):
  4 instances (30s, 01_min, 02_min, 05_min)
  x 3 delays (03, 05, 10)
  x 6 selectors (random, roulette, roulette_persist, 3 ML models with softmax T=0.10)
  x 4 budgets (200, 1000, 5000, 10000 ms)
  x 5 seeds (11, 22, 33, 44, 55)
  = 1440 runs

ML picks (each tests a distinct hypothesis):
  - 20f_t05 T=0.10  : 8/8 cells beat random on delay10 fresh data (the surprise winner)
  - 20f_t07 T=0.10  : highest training F1-lift (+0.272), 7/8 on delay10
  - 20f_ei_log T=0.10: regressor (Holm-sig per-iter lift in v6); tests whether
                       per-iter quality converts to final-obj win under clean conditions

Roulette persistence (--persist-roulette-weights) is treated as a first-class
baseline so the report can directly answer "does carrying weights across epochs
explain ML's lift, or does ML add something more?".

BUILD_ID stamping: every log gets a `# BUILD_ID=<solver mtime>` header line.
Resume validates the stamp against the current binary and re-runs on mismatch.
Logs without a stamp (legacy) are still accepted so existing benches don't get
invalidated unnecessarily.

Usage:
    python benchmarks/launch_final_bench.py                  # 4 workers, all 1440 runs
    python benchmarks/launch_final_bench.py --workers 6
    python benchmarks/launch_final_bench.py --delays 10      # only delay10 tier
    python benchmarks/launch_final_bench.py --budgets 5000 25000   # only some budgets
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

REPO   = Path(__file__).resolve().parent.parent
SOLVER = REPO / "cpp" / "build" / "darp_solver"
DATA   = REPO / "data" / "fiedler_converted"

# BUILD_ID derived from the solver binary's mtime so every relaunch can detect
# a mid-run rebuild and refuse to mix binaries. Read once at import-time so the
# value is consistent across all worker processes for this submission.
BUILD_ID = str(int(SOLVER.stat().st_mtime)) if SOLVER.exists() else "missing"

INSTANCE_BASES = ["30_s", "01_min", "02_min", "05_min"]
DELAYS         = ["03", "05", "10"]
BUDGETS_MS     = [200, 1000, 5000, 10000]
SEEDS          = [11, 22, 33, 44, 55]

# (label, dir_name, softmax_temp)
ML_VARIANTS = [
    ("ml_20f_t05__T0.10",     "models_v3_full_20f_t05",    0.10),
    ("ml_20f_t07__T0.10",     "models_v3_full_20f_t07",    0.10),
    ("ml_20f_ei_log__T0.10",  "models_v3_full_20f_ei_log", 0.10),
]

# Headline grid — baselines first so the queue drains the cheap rows early,
# then ML rows queue while the slow tails finish.
SELECTORS = ["random", "roulette", "roulette_persist"] + [v[0] for v in ML_VARIANTS]
ML_LOOKUP = {label: (dir_name, softmax) for label, dir_name, softmax in ML_VARIANTS}

DESTROY = "random,worst,shaw"
REPAIR  = "greedy,regret2,regret3"

_RE_TOTAL_TRAVEL = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")
_RE_ITERS_DYN    = re.compile(r"Total ALNS iterations:\s+(\d+)")
_RE_BUILD_ID     = re.compile(r"^# BUILD_ID=(\S+)")


@dataclass
class Run:
    instance_base: str   # e.g. "30_s"
    delay: str           # "03" / "05" / "10"
    selector: str        # one of SELECTORS
    budget: int          # ms
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


def cached_log_is_valid(log_path: Path) -> bool:
    """Resume rule: cached log counts as fresh iff (a) it exists and is non-empty,
    AND (b) its BUILD_ID header matches the current binary OR the log has no
    BUILD_ID header (legacy from before this script — accept rather than wipe).
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False
    with open(log_path) as f:
        first_line = f.readline()
    m = _RE_BUILD_ID.match(first_line)
    if m is None:
        # Legacy log without stamp — trust by default.
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

    # Stamp the BUILD_ID so future relaunches can detect a mid-run rebuild.
    log_path.write_text(f"# BUILD_ID={BUILD_ID}\n{text}")
    parsed = parse(text)
    return {**asdict(r), **parsed, "wall_s": round(wall, 2), "exit_code": rc,
            "run_id": run_id, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "bench_results_final"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400,
                    help="Per-run timeout in seconds (default 2400 = 40 min).")
    ap.add_argument("--delays", nargs="*", default=DELAYS,
                    help="Restrict to these delays (e.g. --delays 10).")
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS_MS,
                    help="Restrict to these budget tiers (ms).")
    ap.add_argument("--instances", nargs="*", default=INSTANCE_BASES,
                    help="Restrict to these instance bases.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_logs"; raw.mkdir(exist_ok=True)

    runs: list[Run] = []
    for inst in args.instances:
        for delay in args.delays:
            for sel in SELECTORS:
                for b in args.budgets:
                    for s in SEEDS:
                        runs.append(Run(inst, delay, sel, b, s))

    print(f"Total runs:     {len(runs)}")
    print(f"Workers:        {args.workers}")
    print(f"Output dir:     {out}")
    print(f"Solver binary:  {SOLVER} (BUILD_ID={BUILD_ID})")
    print(f"Selectors:      {SELECTORS}")
    print(f"Delays:         {args.delays}")
    print(f"Budgets (ms):   {args.budgets}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, r, raw, args.timeout): r for r in runs}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            results.append(res)
            tag = " [cached]" if res.get("cached") else ""
            obj = f"{res['obj']:.1f}" if res['obj'] is not None else "FAIL"
            print(f"  [{i:>4}/{len(runs)}]  "
                  f"d{res['delay']} | "
                  f"{res['instance_base']:>6} | "
                  f"{res['selector']:<22} | "
                  f"b={res['budget']:>5} | "
                  f"s={res['seed']:>2} | "
                  f"obj={obj:>10}{tag}")

    fields = ["instance_base", "delay", "selector", "budget", "seed",
              "obj", "iters", "wall_s", "exit_code", "run_id", "cached"]
    csv_path = out / "results.csv"
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {csv_path} ({len(results)} rows)")
    n_fail = sum(1 for r in results if r['obj'] is None)
    print(f"Failures: {n_fail}")


if __name__ == "__main__":
    main()
