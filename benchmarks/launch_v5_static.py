"""Fast static-mode bench launcher (v5-static).

Runs the C++ solver in *static* mode (no `--dynamic`) on the same Manhattan
instances as v5-fast, sweeping fixed iteration counts.

Differences from launch_v5_fast.py:
  - No --dynamic / no time-budget. Uses --iterations N instead.
  - No fair/unfair distinction (selection-time is part of the iteration loop;
    fairness flag has no meaning here).
  - Per-iter ALNS counters (improving / new-best / sa-accepted) parsed from
    stdout — no epoch_metrics.csv since static has only one ALNS call.
  - Self-contained runner (does not go through run_dynamic_bench.execute_run)
    because Fiedler solution conversion + validation is only needed for
    dynamic-mode benches.

Grid: 4 instances × 3 iter counts × 3 seeds × 7 selectors = 252 runs.
Selector rationale (see analysis/feature_importance_static.md):
  - random, roulette                    : baselines
  - ml/20f_regr, 20f_regr__T0.10        : best static-transfer (7.4% dyn weight)
  - ml/20f_t07,  20f_t07__T0.10         : low-dyn classifier (9.2%)
  - ml/20f_ei_log                       : control — predicted to regress (21.3%)

Output layout:
    bench_results_v5_static/
    ├── launch.log
    ├── baselines/runs/<run_id>/summary.json
    ├── 20f_regr/runs/<run_id>/summary.json
    ├── 20f_regr__T0.10/runs/<run_id>/summary.json
    ├── 20f_t07/runs/<run_id>/summary.json
    ├── 20f_t07__T0.10/runs/<run_id>/summary.json
    └── 20f_ei_log/runs/<run_id>/summary.json

Each summary.json mimics the dynamic-bench schema where applicable so the
existing report generator can ingest these results (filter by --budget=N
where the "budget" field contains the iteration count for static runs).
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOLVER = REPO_ROOT / "cpp" / "build" / "darp_solver"
ML_TRAIN = REPO_ROOT / "ml_train"
INSTANCES_DIR = REPO_ROOT / "data" / "fiedler_converted"


WANTED_INSTANCES = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]
SEEDS = [1, 2, 3]
ITERS = [200, 1000, 5000]
N_WORKERS = 4
DESTROY = "random,worst,shaw"
REPAIR = "greedy,regret2,regret3"
CONFIG_NAME = "classical"
PENALTY = 10000

# (selector_label, c++ selection arg, ml_model_dir or None, softmax_temp or None)
# Selector_label is also the on-disk subdirectory under <out>/<label>/runs.
SELECTORS: list[tuple[str, str, Optional[Path], Optional[float]]] = [
    ("baselines/random",   "random",   None, None),
    ("baselines/roulette", "roulette", None, None),
    ("20f_regr",           "ml", ML_TRAIN / "models_v3_full_20f_regr"   / "onnx", None),
    ("20f_regr__T0.10",    "ml", ML_TRAIN / "models_v3_full_20f_regr"   / "onnx", 0.10),
    ("20f_t07",            "ml", ML_TRAIN / "models_v3_full_20f_t07"    / "onnx", None),
    ("20f_t07__T0.10",     "ml", ML_TRAIN / "models_v3_full_20f_t07"    / "onnx", 0.10),
    ("20f_ei_log",         "ml", ML_TRAIN / "models_v3_full_20f_ei_log" / "onnx", None),
]


# ---------------------------------------------------------------------------
# Run definition
# ---------------------------------------------------------------------------
@dataclass
class StaticRunSpec:
    instance_basename: str
    instance_path: Path
    selector_label: str            # e.g. "20f_regr__T0.10" or "baselines/random"
    selection_arg: str             # "random" | "roulette" | "ml"
    ml_model_dir: Optional[Path]
    ml_softmax_temp: Optional[float]
    iterations: int
    seed: int

    @property
    def run_id(self) -> str:
        return (f"{self.instance_basename}__{self.selection_arg}__"
                f"{CONFIG_NAME}__iters{self.iterations}__seed{self.seed}")


def build_cmd(spec: StaticRunSpec, solver: Path) -> list[str]:
    cmd = [
        str(solver),
        "--instance", str(spec.instance_path),
        "--iterations", str(spec.iterations),
        "--seed", str(spec.seed),
        "--selection", spec.selection_arg,
        "--destroy", DESTROY,
        "--repair", REPAIR,
        "--penalty", str(PENALTY),
    ]
    if spec.selection_arg == "ml":
        if spec.ml_model_dir is None:
            raise ValueError("ML selection requires ml_model_dir")
        cmd += ["--ml-model-dir", str(spec.ml_model_dir)]
        if spec.ml_softmax_temp is not None:
            cmd += ["--ml-softmax-temp", str(spec.ml_softmax_temp)]
    return cmd


# Patterns for parsing static-solver stdout
_PAT_OBJ = re.compile(r"^Final objective:\s+(\d+(?:\.\d+)?)", re.M)
_PAT_TRAVEL = re.compile(r"^Travel cost:\s+(\d+(?:\.\d+)?)", re.M)
_PAT_UNASSIGNED = re.compile(r"^Unassigned:\s+(\d+)", re.M)
_PAT_SOLVE_TIME = re.compile(r"^Solve time:\s+([\d.]+)s", re.M)
_PAT_TOTAL_ITERS = re.compile(r"^Total iterations:\s+(\d+)", re.M)
_PAT_IMPROVING = re.compile(r"improving:\s+(\d+)", re.M)
_PAT_NEW_BEST = re.compile(r"new global best:\s+(\d+)", re.M)
_PAT_SA_ACCEPTED = re.compile(r"SA-accepted:\s+(\d+)", re.M)
_PAT_FEASIBLE = re.compile(r"^Feasible:\s+(\w+)", re.M)
_PAT_INIT_OBJ = re.compile(r"Initial solution: obj=(\d+(?:\.\d+)?)", re.M)


def _grab(pat: re.Pattern, text: str, default=None):
    m = pat.search(text)
    return m.group(1) if m else default


def parse_static_stdout(text: str) -> dict[str, Any]:
    return {
        "cost": float(_grab(_PAT_OBJ, text, 0)),
        "total_travel_time": float(_grab(_PAT_TRAVEL, text, 0)),
        "unassigned": int(_grab(_PAT_UNASSIGNED, text, 0)),
        "solve_time_s": float(_grab(_PAT_SOLVE_TIME, text, 0)),
        "total_alns_iterations": int(_grab(_PAT_TOTAL_ITERS, text, 0)),
        "total_iters_improving": int(_grab(_PAT_IMPROVING, text, 0)),
        "total_iters_new_best": int(_grab(_PAT_NEW_BEST, text, 0)),
        "total_iters_sa_accepted": int(_grab(_PAT_SA_ACCEPTED, text, 0)),
        "feasible": _grab(_PAT_FEASIBLE, text, "unknown") == "yes",
        "initial_obj": float(_grab(_PAT_INIT_OBJ, text, 0)),
    }


def execute_static_run(spec: StaticRunSpec, runs_dir: Path, solver: Path,
                       force: bool = False) -> dict[str, Any]:
    """Run a single static-mode invocation. Returns summary dict."""
    run_dir = runs_dir / spec.run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        return {"run_id": spec.run_id, "skipped_reason": "already completed"}
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(spec, solver)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        text = proc.stdout + proc.stderr
        rc = proc.returncode
    except Exception as e:
        text = f"[runner] failed to launch solver: {e}\n"
        rc = -1
    wall_clock_s = time.monotonic() - t0
    text += f"\n[runner] wall-clock: {wall_clock_s:.3f}s\n"
    (run_dir / "stdout.log").write_text(text)

    parsed = parse_static_stdout(text) if rc == 0 else {}
    summary = {
        "run_id": spec.run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "instance": spec.instance_basename,
        "selection": spec.selection_arg,
        "selector_label": spec.selector_label,
        "config_name": CONFIG_NAME,
        "destroy": DESTROY,
        "repair": REPAIR,
        "iterations": spec.iterations,
        # report generator filters by epoch_budget_ms — store iters there
        # so existing aggregation code still works as a "budget" axis.
        "epoch_budget_ms": spec.iterations,
        "seed": spec.seed,
        "ml_model_dir": str(spec.ml_model_dir) if spec.ml_model_dir else None,
        "ml_softmax_temp": spec.ml_softmax_temp,
        "wall_clock_s": wall_clock_s,
        "solver_exit_code": rc,
        "solver_crashed": rc != 0,
        **parsed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Tee (mirror stdout to launch.log)
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try: st.write(s)
            except (OSError, ValueError): pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try: st.flush()
            except (OSError, ValueError): pass

    def isatty(self):
        return False


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _format_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def _banner(title: str, level: int = 1) -> None:
    char = "=" if level == 1 else "-"
    print(char * 60)
    print(f" {title}")
    print(char * 60)


# ---------------------------------------------------------------------------
# Spec construction + execution
# ---------------------------------------------------------------------------
def discover_instance_path(basename: str) -> Path:
    p = INSTANCES_DIR / f"{basename}.txt"
    if not p.exists():
        raise SystemExit(f"ERROR: instance file not found: {p}")
    return p


def build_specs(instances: list[str], iters: list[int], seeds: list[int]
                ) -> list[StaticRunSpec]:
    out: list[StaticRunSpec] = []
    for inst in instances:
        inst_path = discover_instance_path(inst)
        for label, sel_arg, model_dir, softmax_t in SELECTORS:
            for n_iter, seed in itertools.product(iters, seeds):
                out.append(StaticRunSpec(
                    instance_basename=inst,
                    instance_path=inst_path,
                    selector_label=label,
                    selection_arg=sel_arg,
                    ml_model_dir=model_dir,
                    ml_softmax_temp=softmax_t,
                    iterations=n_iter,
                    seed=seed,
                ))
    return out


def run_specs_parallel(specs: list[StaticRunSpec], out_dir: Path, solver: Path,
                       n_workers: int, force: bool) -> dict[str, Any]:
    completed = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    total = len(specs)
    t0 = time.monotonic()

    def runs_dir_for(spec: StaticRunSpec) -> Path:
        return out_dir / spec.selector_label / "runs"

    def _record(spec: StaticRunSpec, summary: Optional[dict], err: Optional[str]) -> None:
        nonlocal completed, skipped, failed
        if err is not None:
            failed += 1
            failures.append((spec.run_id, err))
            tag = "FAIL"
        elif summary and summary.get("skipped_reason"):
            skipped += 1
            tag = "SKIP"
        elif summary and summary.get("solver_crashed"):
            failed += 1
            failures.append((spec.run_id, f"solver exit={summary.get('solver_exit_code')}"))
            tag = "FAIL"
        else:
            completed += 1
            tag = "OK"
        done = completed + skipped + failed
        cost_str = ""
        if summary and summary.get("cost"):
            cost_str = f" cost={summary['cost']:,.0f}"
        print(f"  [{done:3d}/{total}] {tag:4s}  [{spec.selector_label}] {spec.run_id}{cost_str}")

    if n_workers <= 1:
        for spec in specs:
            try:
                summary = execute_static_run(spec, runs_dir_for(spec), solver, force=force)
            except Exception:
                _record(spec, None, traceback.format_exc())
                continue
            _record(spec, summary, None)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(execute_static_run, spec, runs_dir_for(spec), solver, force): spec
                    for spec in specs}
            for fut in as_completed(futs):
                spec = futs[fut]
                try:
                    summary = fut.result()
                except Exception:
                    _record(spec, None, traceback.format_exc())
                    continue
                _record(spec, summary, None)

    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "elapsed": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="bench_results_v5_static",
                    help="Results root (default: bench_results_v5_static)")
    ap.add_argument("--iters", type=int, nargs="+", default=ITERS,
                    help=f"Iteration counts to sweep (default: {ITERS})")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help=f"Seeds (default: {SEEDS})")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help=f"Parallel workers (default: {N_WORKERS})")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if summary.json already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print matrix + estimated wall, do not execute")
    args = ap.parse_args()

    if not DEFAULT_SOLVER.exists():
        print(f"ERROR: solver not found at {DEFAULT_SOLVER}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    specs = build_specs(WANTED_INSTANCES, args.iters, args.seeds)

    # Wall estimate: very rough — assume 12s/run at 200 iters, 60s @ 1000, 300s @ 5000
    iter_cost = {200: 12, 1000: 60, 5000: 300}
    avg_run_s = sum(iter_cost.get(i, max(i / 50, 5)) for i in args.iters) / len(args.iters)
    est_serial = len(specs) * avg_run_s
    est_parallel = est_serial / max(1, args.workers)

    log_file = None
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / "launch.log", "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_file)

    _banner("launch_v5_static")
    print(f" started:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" out dir:  {out_dir}")
    print(f" config:   {CONFIG_NAME}  destroy={DESTROY}  repair={REPAIR}")
    print(f" iters:    {args.iters}")
    print(f" seeds:    {args.seeds}")
    print(f" workers:  {args.workers}")
    print(f" instances ({len(WANTED_INSTANCES)}):")
    for inst in WANTED_INSTANCES:
        print(f"   - {inst}")
    print(f" selectors ({len(SELECTORS)}):")
    for label, sel_arg, model_dir, softmax_t in SELECTORS:
        sftx = f", softmax={softmax_t}" if softmax_t is not None else ""
        mdir = model_dir.name if model_dir else "—"
        print(f"   - {label:<22} ({sel_arg}, model={mdir}{sftx})")
    print()
    print(f" Total runs:                     {len(specs)}")
    print(f" Wall estimate (serial):         ~{_format_elapsed(est_serial)}")
    print(f" Wall estimate ({args.workers} workers):     ~{_format_elapsed(est_parallel)}")
    print()

    if args.dry_run:
        _banner("DRY RUN — first 30 run_ids", level=2)
        for spec in specs[:30]:
            print(f"  {spec.selector_label}/runs/{spec.run_id}")
        if len(specs) > 30:
            print(f"  ... and {len(specs) - 30} more")
        print("\n--dry-run: not executing.")
        return 0

    result = run_specs_parallel(specs, out_dir, DEFAULT_SOLVER, args.workers, args.force)

    _banner(
        f"DONE — {result['completed']} ok, {result['skipped']} skipped, "
        f"{result['failed']} failed, total wall {_format_elapsed(result['elapsed'])}"
    )
    if result["failures"]:
        print(" Failures:")
        for run_id, _ in result["failures"]:
            print(f"   - {run_id}")
        with open(out_dir / "failures.log", "a") as f:
            for run_id, tb in result["failures"]:
                f.write(f"=== {run_id} ===\n{tb}\n\n")

    if log_file is not None:
        log_file.flush()
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
