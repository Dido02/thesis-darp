#!/usr/bin/env python3
"""Phase E (user-run) — full tree bench mirroring the final forest bench.

Forks benchmarks/launch_final_bench.py with tree selectors only. Writes to
model_diagnostics_2026-05-19/bench_results_final_trees/ so the existing
bench_results_final/ (forests + baselines) is left untouched.

Grid: 4 instances * 3 delays * 3 tree selectors * 4 budgets * 5 seeds = 720 runs.
Tree selectors:
  - ml_tree_20f_t05__T0.10  -> models_v6_full_20f_t05_tree (existing)
  - ml_tree_20f_t07__T0.10  -> models_v6_full_20f_t07_tree (existing)
  - ml_tree_20f_ei_log__T0.10 -> models_v6_full_20f_ei_log_tree (existing)

NOTE: this uses the *existing* v6 single trees (max_depth=None), not the new
depth-4 or top-K variants from the deep dive — those would need their ONNX
artefacts copied/regenerated into the canonical models_v6_full_*_tree shape
that the C++ solver loads. Recommended follow-up if Phase E confirms the
training-side picture.

BUILD_ID stamping + resume rule are inherited from the final bench.

Usage:
    python 06_launch_tree_bench.py
    python 06_launch_tree_bench.py --workers 6
    python 06_launch_tree_bench.py --delays 10           # only delay10 tier
    python 06_launch_tree_bench.py --budgets 5000 10000  # only some budgets
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

# (label, dir_name, softmax_temp)
ML_VARIANTS = [
    ("ml_tree_20f_t05__T0.10",     "models_v6_full_20f_t05_tree",    0.10),
    ("ml_tree_20f_t07__T0.10",     "models_v6_full_20f_t07_tree",    0.10),
    ("ml_tree_20f_ei_log__T0.10",  "models_v6_full_20f_ei_log_tree", 0.10),
]

SELECTORS = [v[0] for v in ML_VARIANTS]
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
    selector: str
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
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False
    with open(log_path) as f:
        first_line = f.readline()
    m = _RE_BUILD_ID.match(first_line)
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
    default_out = REPO / "model_diagnostics_2026-05-19" / "bench_results_final_trees"
    ap.add_argument("--out-dir", default=str(default_out))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400,
                    help="Per-run timeout in seconds (default 2400 = 40 min).")
    ap.add_argument("--delays", nargs="*", default=DELAYS)
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS_MS)
    ap.add_argument("--instances", nargs="*", default=INSTANCE_BASES)
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
                  f"{res['selector']:<28} | "
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
