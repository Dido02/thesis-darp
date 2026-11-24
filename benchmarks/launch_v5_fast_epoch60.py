"""Fast multi-model bench launcher (v5-fast, 60s epoch length).

Identical to launch_v5_fast.py EXCEPT:
  - EPOCH_LENGTH_S = 60.0 (vs 30.0 in v5-fast)
  - BUDGETS = [1000, 5000] (covers both budgets — 5000ms partly offsets the
    fewer-epochs compute loss, 1000ms isolates the per-epoch SA-depth effect)
  - Default out-dir: bench_results_v5_fast_epoch60

Rationale: with 30s epochs, the 5min instance hits 15 epochs and SA gets
reset every epoch. Doubling epoch length to 60s halves the number of epochs
(e.g. 5min → ~7-8 epochs), so each SA run gets to cool deeper inside its
own schedule before reset. Running two budgets lets us decouple the
SA-depth effect (1000ms) from the per-epoch compute scaling (5000ms ≈
matched-compute against v5's 1000ms × 30s grid for the 5min cell).

Grid: 4 instances × 2 budgets × 3 seeds × {random, roulette, ml × 6
v3_full_20f_* models × 3 softmax variants} × {fair, unfair} = 960 runs.

Differences from v4-fast:
  - Single 1000ms budget (drops the 5000ms cell).
  - 20f models only, excluding models_v3_full_20f_ei (keeps ei_log).
  - Each ML model expands into 3 softmax variants (default, T=0.05, T=0.10).
  - Each ML run writes ml_probs.jsonl alongside its summary.json
    (consumed by benchmarks/plot_ml_probabilities.py).

Two phases:
  1. Baselines (random + roulette) into <out>/baselines_{fair,unfair}/runs/.
  2. Per-model × softmax-variant ML loop into
     <out>/{fair,unfair}/<model_name><variant_suffix>/runs/.

Each phase logs a banner + per-(model, variant) elapsed wall. All output is
teed to <out>/launch.log. Exit code is non-zero iff any individual run raised.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# Add benchmarks/ to sys.path so both this process AND any spawned worker
# subprocesses can import run_dynamic_bench when re-executing the module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dynamic_bench import (  # noqa: E402  (after sys.path tweak)
    Instance,
    OPERATOR_CONFIGS,
    RunSpec,
    discover_instances,
    execute_run,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOLVER = REPO_ROOT / "cpp" / "build" / "darp_solver"
ML_MODEL_GLOB = "models_v3_full_20f_*"
EXCLUDED_MODELS = {"models_v3_full_20f_ei"}

WANTED_INSTANCES = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]
SEEDS = [1, 2, 3]
BUDGETS = [1000, 5000]
N_WORKERS = 4
CONFIG_NAME = "classical"
EPOCH_LENGTH_S = 60.0

# (suffix appended to model_name, softmax_temp passed to solver)
# "" with None = baseline proportional sampler (existing behaviour).
SOFTMAX_VARIANTS: list[tuple[str, Optional[float]]] = [
    ("",        None),
    ("__T0.05", 0.05),
    ("__T0.10", 0.10),
]


# ---------------------------------------------------------------------------
# Tee: mirror writes to both stdout and a log file.
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except (OSError, ValueError):
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
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
# Discovery helpers
# ---------------------------------------------------------------------------
def discover_ml_dirs(repo_root: Path) -> list[tuple[str, Path]]:
    """Return [(model_name, onnx_dir)] for every 20f_* model except those in
    EXCLUDED_MODELS. Each must have a feature_manifest.json under onnx/."""
    base = repo_root / "ml_train"
    found: list[tuple[str, Path]] = []
    for manifest in sorted(base.glob(f"{ML_MODEL_GLOB}/onnx/feature_manifest.json")):
        onnx_dir = manifest.parent
        model_dir = onnx_dir.parent
        if model_dir.name in EXCLUDED_MODELS:
            continue
        found.append((model_dir.name, onnx_dir))
    return found


def pick_instances(repo_root: Path, wanted: list[str]) -> list[Instance]:
    converted_dir = repo_root / "data" / "fiedler_converted"
    all_inst = discover_instances(converted_dir)
    by_name = {i.basename: i for i in all_inst}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise SystemExit(f"ERROR: instances not found in {converted_dir}: {missing}")
    return [by_name[w] for w in wanted]


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------
def build_specs(
    instances: list[Instance],
    selections: list[str],
    seeds: list[int],
    budgets: list[int],
    fairness: bool,
    ml_model_dir: Optional[Path],
    solver: Path,
    softmax_temp: Optional[float] = None,
    log_probs: bool = False,
) -> list[RunSpec]:
    cfg = OPERATOR_CONFIGS[CONFIG_NAME]
    out: list[RunSpec] = []
    for inst, sel, budget, seed in itertools.product(instances, selections, budgets, seeds):
        out.append(RunSpec(
            instance=inst,
            selection=sel,
            config_name=CONFIG_NAME,
            destroy=cfg["destroy"],
            repair=cfg["repair"],
            budget_ms=budget,
            seed=seed,
            epoch_length_s=EPOCH_LENGTH_S,
            ml_model_dir=ml_model_dir if sel == "ml" else None,
            solver=solver,
            skip_validation=False,
            exclude_selection_from_budget=fairness,
            ml_softmax_temp=softmax_temp if sel == "ml" else None,
            log_ml_probs=(sel == "ml" and log_probs),
        ))
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_specs(
    specs: list[RunSpec],
    runs_dir: Path,
    n_workers: int,
) -> dict[str, Any]:
    """Execute specs (parallel via ProcessPoolExecutor or serial if workers=1).

    Per-spec failures are caught, logged to the failures list, and do NOT halt
    the sweep. Return aggregate counts + failure list.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    total = len(specs)
    t0 = time.monotonic()

    def _record(spec: RunSpec, summary: Optional[dict], err: Optional[str]) -> None:
        nonlocal completed, skipped, failed
        if err is not None:
            failed += 1
            failures.append((spec.run_id, err))
            tag = "FAIL"
        elif summary and summary.get("skipped_reason"):
            skipped += 1
            tag = "SKIP"
        else:
            completed += 1
            tag = "OK"
        done = completed + skipped + failed
        print(f"  [{done:3d}/{total}] {tag:4s}  {spec.run_id}")

    if n_workers <= 1:
        for spec in specs:
            try:
                summary = execute_run(spec, runs_dir, force=False)
            except Exception:
                _record(spec, None, traceback.format_exc())
                continue
            _record(spec, summary, None)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(execute_run, spec, runs_dir, False): spec for spec in specs}
            for fut in as_completed(futs):
                spec = futs[fut]
                try:
                    summary = fut.result()
                except Exception:
                    _record(spec, None, traceback.format_exc())
                    continue
                _record(spec, summary, None)

    elapsed = time.monotonic() - t0
    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="bench_results_v5_fast_epoch60",
                    help="Results root (default: bench_results_v5_fast_epoch60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print matrix + estimated wall, do not execute")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if not DEFAULT_SOLVER.exists():
        print(f"ERROR: solver not found at {DEFAULT_SOLVER}", file=sys.stderr)
        print("  Build: cd cpp/build && cmake -DENABLE_ML_SELECTION=ON .. && make",
              file=sys.stderr)
        return 1

    instances = pick_instances(REPO_ROOT, WANTED_INSTANCES)
    ml_dirs = discover_ml_dirs(REPO_ROOT)
    if not ml_dirs:
        print(f"ERROR: no ML models found under ml_train/{ML_MODEL_GLOB}/onnx/ "
              f"(after excluding {EXCLUDED_MODELS})", file=sys.stderr)
        return 1

    # Run-count breakdown
    n_baseline_per_fairness = len(instances) * 2 * len(BUDGETS) * len(SEEDS)  # 2 baselines
    n_ml_per_variant_per_fairness = len(instances) * 1 * len(BUDGETS) * len(SEEDS)
    n_baseline_total = n_baseline_per_fairness * 2
    n_ml_total = n_ml_per_variant_per_fairness * 2 * len(ml_dirs) * len(SOFTMAX_VARIANTS)
    total = n_baseline_total + n_ml_total

    # Wall estimate (avg over the 4 instances chosen here; budget = 1s only)
    avg_epochs = (1 + 2 + 4 + 10) / 4   # 30s, 1min, 2min, 5min
    avg_budget_s = sum(BUDGETS) / len(BUDGETS) / 1000.0
    overhead_s = 1.5
    avg_run_s = avg_epochs * (avg_budget_s + overhead_s)
    est_serial_s = total * avg_run_s
    est_parallel_s = est_serial_s / N_WORKERS

    # Tee stdout to launch.log (skip for dry-run since no out-dir activity expected)
    log_file = None
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / "launch.log", "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_file)

    _banner("launch_v5_fast")
    print(f" started:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f" out dir:  {out_dir}")
    print(f" config:   {CONFIG_NAME}")
    print(f" budgets:  {BUDGETS} ms")
    print(f" seeds:    {SEEDS}")
    print(f" workers:  {N_WORKERS}")
    print(f" instances ({len(instances)}):")
    for inst in instances:
        print(f"   - {inst.basename} [{inst.duration}, {inst.delay}]")
    print(f" ML model dirs ({len(ml_dirs)}):")
    for name, _ in ml_dirs:
        print(f"   - {name}")
    print(f" softmax variants ({len(SOFTMAX_VARIANTS)}):")
    for suffix, t in SOFTMAX_VARIANTS:
        label = "default proportional" if t is None else f"softmax T={t}"
        print(f"   - suffix='{suffix}'  ({label})")
    print()
    print(f" Total runs:                     {total}")
    print(f"   baselines (fair+unfair):        {n_baseline_total}")
    print(f"   ml × {len(ml_dirs)} dirs × {len(SOFTMAX_VARIANTS)} variants × 2 fairness: {n_ml_total}")
    print(f" Wall estimate (serial):         ~{_format_elapsed(est_serial_s)}")
    print(f" Wall estimate ({N_WORKERS} workers):     ~{_format_elapsed(est_parallel_s)}")
    print()

    if args.dry_run:
        _banner("DRY RUN — run_id list", level=2)
        # Phase 1 specs
        print("Phase 1 — baselines:")
        for fair in (False, True):
            label = "fair" if fair else "unfair"
            specs = build_specs(
                instances=instances,
                selections=["random", "roulette"],
                seeds=SEEDS, budgets=BUDGETS, fairness=fair,
                ml_model_dir=None, solver=DEFAULT_SOLVER,
            )
            for s in specs:
                print(f"  baselines_{label}/runs/{s.run_id}")
        # Phase 2 specs
        print()
        print("Phase 2 — ML × softmax variants:")
        for model_name, onnx_dir in ml_dirs:
            for suffix, softmax_t in SOFTMAX_VARIANTS:
                full_name = f"{model_name}{suffix}"
                for fair in (False, True):
                    label = "fair" if fair else "unfair"
                    specs = build_specs(
                        instances=instances,
                        selections=["ml"],
                        seeds=SEEDS, budgets=BUDGETS, fairness=fair,
                        ml_model_dir=onnx_dir, solver=DEFAULT_SOLVER,
                        softmax_temp=softmax_t, log_probs=True,
                    )
                    for s in specs:
                        print(f"  {label}/{full_name}/runs/{s.run_id}")
        print()
        print("--dry-run: not executing.")
        return 0

    # ------------------------------------------------------------------
    # Phase 1 — baselines (random, roulette) for both fairness modes
    # ------------------------------------------------------------------
    overall_t0 = time.monotonic()
    overall_completed = 0
    overall_skipped = 0
    overall_failed = 0
    all_failures: list[tuple[str, str]] = []

    for fair in (False, True):
        label = "fair" if fair else "unfair"
        _banner(f"PHASE 1/2  baselines ({label})  started {_now()}")
        specs = build_specs(
            instances=instances,
            selections=["random", "roulette"],
            seeds=SEEDS, budgets=BUDGETS, fairness=fair,
            ml_model_dir=None, solver=DEFAULT_SOLVER,
        )
        runs_dir = out_dir / f"baselines_{label}" / "runs"
        result = run_specs(specs, runs_dir, N_WORKERS)
        overall_completed += result["completed"]
        overall_skipped += result["skipped"]
        overall_failed += result["failed"]
        all_failures += result["failures"]
        print(f"  -> {result['completed']} ok, {result['skipped']} skipped, "
              f"{result['failed']} failed in {_format_elapsed(result['elapsed'])}")
        print()

    # ------------------------------------------------------------------
    # Phase 2 — ML × softmax-variant loop, each (model, variant, fairness)
    # in its own runs/ dir.
    # ------------------------------------------------------------------
    n_variants = len(ml_dirs) * len(SOFTMAX_VARIANTS)
    variant_idx = 0
    for model_name, onnx_dir in ml_dirs:
        for suffix, softmax_t in SOFTMAX_VARIANTS:
            variant_idx += 1
            full_name = f"{model_name}{suffix}"
            for fair in (False, True):
                label = "fair" if fair else "unfair"
                _banner(
                    f"PHASE 2/2  [{variant_idx}/{n_variants}] {full_name} "
                    f"({label})  started {_now()}",
                    level=2,
                )
                specs = build_specs(
                    instances=instances,
                    selections=["ml"],
                    seeds=SEEDS, budgets=BUDGETS, fairness=fair,
                    ml_model_dir=onnx_dir, solver=DEFAULT_SOLVER,
                    softmax_temp=softmax_t, log_probs=True,
                )
                runs_dir = out_dir / label / full_name / "runs"
                result = run_specs(specs, runs_dir, N_WORKERS)
                overall_completed += result["completed"]
                overall_skipped += result["skipped"]
                overall_failed += result["failed"]
                all_failures += result["failures"]
                print(f"  -> {result['completed']} ok, {result['skipped']} skipped, "
                      f"{result['failed']} failed in {_format_elapsed(result['elapsed'])}")
                print()

    overall_elapsed = time.monotonic() - overall_t0
    _banner(
        f"DONE — {overall_completed} ok, {overall_skipped} skipped, "
        f"{overall_failed} failed, total wall {_format_elapsed(overall_elapsed)}"
    )
    if all_failures:
        print(" Failures:")
        for run_id, _ in all_failures:
            print(f"   - {run_id}")
        # Persist full tracebacks to a sidecar file.
        with open(out_dir / "failures.log", "a") as f:
            for run_id, tb in all_failures:
                f.write(f"=== {run_id} ===\n{tb}\n\n")
        print(f" Full tracebacks: {out_dir / 'failures.log'}")
    print()
    print("To aggregate per-bucket results:")
    print(f"  for d in {out_dir}/baselines_*/runs {out_dir}/*/*/runs; do")
    print(f"    python benchmarks/aggregate_results.py --results-dir \"$(dirname $d)\"")
    print(f"  done")

    if log_file is not None:
        log_file.flush()

    return 1 if overall_failed else 0


if __name__ == "__main__":
    sys.exit(main())
