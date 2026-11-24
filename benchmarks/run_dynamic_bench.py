"""Dynamic Fiedler benchmark runner — ML vs Roulette vs Random.

Sweeps (instance × selection × operator-config × budget × seed) on dynamic
Fiedler Manhattan instances, captures rich metrics, validates solutions,
writes per-run artifacts + a flat summary.csv.

See benchmarks/README.md for usage.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Local module — same directory
from parse_solver_output import parse_solver_stdout


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONVERTED_DIR = REPO_ROOT / "data" / "fiedler_converted"
DEFAULT_SOLVER = REPO_ROOT / "cpp" / "build" / "darp_solver"
DEFAULT_ML_MODEL_DIR = REPO_ROOT / "ml_train" / "models" / "onnx"
DEFAULT_OUT_DIR = REPO_ROOT / "bench_results"
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "convert_solution_fiedler.py"
FIEDLER_INSTANCE_ROOT = REPO_ROOT / "data" / "fiedler" / "Instances"

DURATION_TIERS = ["30s", "01min", "02min", "05min", "15min", "30min"]
DURATION_RANK = {d: i for i, d in enumerate(DURATION_TIERS)}

OPERATOR_CONFIGS = {
    "classical": {
        "destroy": "random,worst,shaw",
        "repair":  "greedy,regret2,regret3",
    },
    "classical-plus-rt": {
        # Classical + the one novel destroy that survives the data-exploration
        # filter (rtcritical). Drops lowslack (degenerate <5% pos rate on v2)
        # and slack repair (weak signal on small instances). 4 x 3 = 12 pairs.
        "destroy": "random,worst,shaw,rtcritical",
        "repair":  "greedy,regret2,regret3",
    },
    "full": {
        "destroy": "random,worst,shaw,rtcritical,lowslack",
        "repair":  "greedy,regret2,regret3,slack",
    },
    "novel-focused": {
        "destroy": "random,rtcritical,lowslack",
        "repair":  "greedy,slack",
    },
}

# Instance sizes that should run sequentially (memory pressure).
HEAVY_DURATIONS = {"05min", "15min", "30min"}

# ML models were trained only on ≤05min. Anything larger is out of distribution.
ML_OVERSIZE_DURATIONS = {"15min", "30min"}


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    """One converted Fiedler instance with its canonical tags."""
    basename: str                 # filename without .txt (used as run_id component)
    file_path: Path
    area: str                     # "manhattan"
    duration: str                 # normalized: "30s" / "01min" / ... / "30min"
    delay: str                    # normalized: "03min" / "05min" / "10min"
    start: str                    # "18-00" (default) or "07-00" if explicit in name


_RE_INSTANCE_NAME = re.compile(
    # Match both forms:
    #   manhattan_01_min_delay03_min  (underscored, default start=18-00)
    #   manhattan_01min_03min         (short, default start=18-00)
    # Optionally: manhattan_07-00_01_min_delay03_min (future: with start tag)
    r"^(?P<area>manhattan)"
    r"(?:_(?P<start>\d{2}-\d{2}))?"
    r"_(?P<duration_raw>30_s|30s|\d{2}_?min)"
    r"_(?:delay)?(?P<delay_raw>\d{2}_?min)$",
    re.IGNORECASE,
)


def _normalize_tier(raw: str) -> str:
    """Map '30_s', '30s', '01_min', '01min' → canonical form in DURATION_TIERS."""
    s = raw.lower().replace("_", "")
    # Now s is "30s", "01min", "15min", etc.
    return s


def discover_instances(converted_dir: Path) -> list[Instance]:
    """Scan the converted-instances directory, return all valid Manhattans."""
    instances: list[Instance] = []
    for path in sorted(converted_dir.glob("manhattan_*.txt")):
        if "_result" in path.name or "_routes" in path.name:
            continue
        stem = path.stem
        m = _RE_INSTANCE_NAME.match(stem)
        if not m:
            continue
        duration = _normalize_tier(m.group("duration_raw"))
        delay = _normalize_tier(m.group("delay_raw"))
        start = m.group("start") or "18-00"

        # Only keep canonical tiers
        if duration not in DURATION_RANK or delay not in {"03min", "05min", "10min"}:
            continue

        instances.append(Instance(
            basename=stem,
            file_path=path,
            area="manhattan",
            duration=duration,
            delay=delay,
            start=start,
        ))
    return instances


def filter_instances(
    all_instances: list[Instance],
    max_duration: str,
    start: str,          # "07-00" | "18-00" | "both"
    delays: list[str],
) -> list[Instance]:
    """Apply --max-duration / --start / --delays filters.

    Preserves only the *first* Instance seen per (duration, delay, start) so the
    short-form duplicates (e.g. manhattan_01min_03min vs manhattan_01_min_delay03_min)
    don't both end up in the matrix.
    """
    if max_duration not in DURATION_RANK:
        raise ValueError(f"--max-duration must be one of {DURATION_TIERS}, got {max_duration!r}")
    max_rank = DURATION_RANK[max_duration]

    valid_starts = {"07-00", "18-00"} if start == "both" else {start}

    seen: dict[tuple[str, str, str], Instance] = {}
    for inst in all_instances:
        if DURATION_RANK[inst.duration] > max_rank:
            continue
        if inst.start not in valid_starts:
            continue
        if inst.delay not in delays:
            continue
        key = (inst.duration, inst.delay, inst.start)
        if key not in seen:
            seen[key] = inst
    return sorted(seen.values(),
                  key=lambda i: (DURATION_RANK[i.duration], i.delay, i.start))


# ---------------------------------------------------------------------------
# Fiedler Instance config path lookup (for validation + baselines)
# ---------------------------------------------------------------------------
def fiedler_instance_config(inst: Instance) -> Optional[Path]:
    """Return path to the Fiedler config.yaml for `inst`, or None if not found."""
    # Fiedler uses underscored forms: '01_min', '03_min', etc.
    def _with_underscore(tier: str) -> str:
        # "01min" -> "01_min", "30s" -> "30_s", "30min" -> "30_min"
        if tier == "30s":
            return "30_s"
        # insert underscore before "min"
        return tier.replace("min", "_min")

    dur = _with_underscore(inst.duration)
    dly = _with_underscore(inst.delay)

    config = (FIEDLER_INSTANCE_ROOT / inst.area.capitalize()
              / "instances" / f"start_{inst.start}"
              / f"duration_{dur}" / f"max_delay_{dly}" / "config.yaml")
    return config if config.exists() else None


# ---------------------------------------------------------------------------
# Run definition + single-run execution
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    instance: Instance
    selection: str            # "random" | "roulette" | "ml"
    config_name: str          # "classical" | "full" | "novel-focused"
    destroy: str
    repair: str
    budget_ms: int
    seed: int
    epoch_length_s: float
    ml_model_dir: Optional[Path]
    solver: Path
    skip_validation: bool
    exclude_selection_from_budget: bool = False
    ml_softmax_temp: Optional[float] = None
    log_ml_probs: bool = False
    log_epoch_iters: bool = False

    @property
    def run_id(self) -> str:
        return (f"{self.instance.basename}__{self.selection}__"
                f"{self.config_name}__budget{self.budget_ms}__seed{self.seed}")


def _build_solver_cmd(spec: RunSpec, routes_out: Path) -> list[str]:
    cmd = [
        str(spec.solver),
        "--instance", str(spec.instance.file_path),
        "--dynamic",
        "--epoch-length", str(spec.epoch_length_s),
        "--epoch-budget-ms", str(spec.budget_ms),
        "--dynamic-stats",
        "--seed", str(spec.seed),
        "--selection", spec.selection,
        "--destroy", spec.destroy,
        "--repair", spec.repair,
        "--output-routes", str(routes_out),
    ]
    if spec.exclude_selection_from_budget:
        cmd.append("--exclude-selection-from-budget")
    if spec.log_epoch_iters:
        cmd.append("--log-epoch-iters")
    if spec.selection == "ml":
        if spec.ml_model_dir is None:
            raise ValueError("ML selection requires ml_model_dir")
        cmd += ["--ml-model-dir", str(spec.ml_model_dir)]
        if spec.ml_softmax_temp is not None:
            cmd += ["--ml-softmax-temp", str(spec.ml_softmax_temp)]
        if spec.log_ml_probs:
            cmd += ["--log-ml-probs", str(routes_out.parent / "ml_probs.jsonl")]
    return cmd


def _run_solver(spec: RunSpec, run_dir: Path) -> tuple[int, str]:
    """Execute the C++ solver. Returns (exit_code, stdout+stderr text).

    stdout is also written to run_dir/stdout.log.
    """
    routes_out = run_dir / "routes.txt"
    cmd = _build_solver_cmd(spec, routes_out)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout + proc.stderr
        rc = proc.returncode
    except Exception as e:
        text = f"[runner] failed to launch solver: {e}\n"
        rc = -1
    elapsed = time.monotonic() - t0

    text += f"\n[runner] wall-clock: {elapsed:.3f}s\n"
    (run_dir / "stdout.log").write_text(text)
    return rc, text


def _convert_solution(routes_path: Path, inst: Instance, sol_out: Path) -> tuple[int, str]:
    """Invoke scripts/convert_solution_fiedler.py. Returns (rc, output)."""
    instance_dir = fiedler_instance_config(inst)
    if instance_dir is None:
        return 2, f"Fiedler instance dir not found for {inst.basename}\n"

    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        "--routes", str(routes_path),
        "--instance", str(instance_dir.parent),
        "--output", str(sol_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout + proc.stderr


def _validate_solution(solution_path: Path, inst: Instance) -> tuple[bool, str]:
    """Invoke darpinstances.solution_checker. Returns (passed, message)."""
    config = fiedler_instance_config(inst)
    if config is None:
        return False, f"Fiedler config not found for {inst.basename}"
    cmd = [
        sys.executable, "-m", "darpinstances.solution_checker",
        str(solution_path.resolve()),
        "-i", str(config.resolve()),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=300)
    except subprocess.TimeoutExpired:
        return False, "solution_checker timed out (300s)"
    return proc.returncode == 0, proc.stdout + proc.stderr


def _extract_fiedler_metrics(sol_path: Path) -> dict[str, Any]:
    """Extract cost/vehicles/dropped from Fiedler solution.json."""
    try:
        with open(sol_path) as f:
            sol = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"cost": None, "vehicles_used": None, "dropped": None,
                "error": str(e)}
    return {
        "cost": sol.get("cost"),
        "cost_minutes": (sol["cost"] / 60.0) if sol.get("cost") else None,
        "vehicles_used": len(sol.get("plans", [])),
        "dropped": len(sol.get("dropped_requests", [])),
    }


# ---------------------------------------------------------------------------
# Per-run entry point (used by both sequential + parallel paths)
# ---------------------------------------------------------------------------
def execute_run(spec: RunSpec, runs_dir: Path, force: bool) -> dict[str, Any]:
    """Execute one RunSpec end-to-end. Returns the summary dict."""
    run_id = spec.run_id
    base_dir = runs_dir / run_id

    # --force policy: create a NEW timestamped dir, don't touch the old one.
    if base_dir.exists() and not force:
        existing = base_dir / "summary.json"
        if existing.exists():
            return {"run_id": run_id, "skipped_reason": "already completed",
                    "summary_path": str(existing)}
        # Directory exists but incomplete — overwrite in-place.
        run_dir = base_dir
    elif base_dir.exists() and force:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = runs_dir / f"{run_id}__{ts}"
    else:
        run_dir = base_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run solver
    rc, stdout_text = _run_solver(spec, run_dir)

    # 2. Parse stdout
    try:
        parsed_summary, epoch_rows = parse_solver_stdout(stdout_text)
    except Exception as e:
        parsed_summary, epoch_rows = {}, []
        (run_dir / "parser_error.txt").write_text(str(e))

    # 3. Convert solution to Fiedler JSON
    routes_path = run_dir / "routes.txt"
    solution_path = run_dir / "solution.json"
    conv_rc, conv_out = (1, "skipped — solver did not finish")
    fiedler_metrics: dict[str, Any] = {}
    if rc == 0 and routes_path.exists():
        conv_rc, conv_out = _convert_solution(routes_path, spec.instance, solution_path)
        (run_dir / "conversion.log").write_text(conv_out)
        if conv_rc == 0 and solution_path.exists():
            fiedler_metrics = _extract_fiedler_metrics(solution_path)

    # 4. Validate (unless skipped)
    validation_passed = None
    validation_msg = ""
    if rc == 0 and conv_rc == 0 and not spec.skip_validation:
        validation_passed, validation_msg = _validate_solution(solution_path, spec.instance)
        (run_dir / "validation.txt").write_text(validation_msg)
    elif spec.skip_validation:
        validation_passed = None
        validation_msg = "skipped"
        (run_dir / "validation.txt").write_text("skipped\n")

    # 4b. For ML runs, capture model metadata from the manifest directly.
    #     (In dynamic mode the solver runs with verbose=false and doesn't print
    #     the "ML selector loaded" banner, so we can't parse it from stdout.)
    ml_meta: dict[str, Any] = {}
    if spec.selection == "ml" and spec.ml_model_dir is not None:
        try:
            with open(spec.ml_model_dir / "feature_manifest.json") as f:
                manifest = json.load(f)
            ml_meta = {
                "model_version": manifest.get("model_version"),
                "n_features_loaded": manifest.get("n_features"),
                # n_active_pairs depends on operator subset; compute here.
                "n_active_pairs_loaded": (
                    len(spec.destroy.split(",")) * len(spec.repair.split(","))
                ),
            }
        except (OSError, json.JSONDecodeError) as e:
            ml_meta = {"ml_manifest_error": str(e)}

    # 5. Assemble summary.json
    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),

        # Config
        "instance": spec.instance.basename,
        "instance_duration": spec.instance.duration,
        "instance_delay": spec.instance.delay,
        "start": spec.instance.start,
        "selection": spec.selection,
        "config_name": spec.config_name,
        "destroy": spec.destroy,
        "repair": spec.repair,
        "epoch_length_s": spec.epoch_length_s,
        "epoch_budget_ms": spec.budget_ms,
        "seed": spec.seed,

        # Solver + Fiedler metrics
        **parsed_summary,
        **fiedler_metrics,
        **ml_meta,

        # Validation
        "validation_passed": validation_passed,
        "validation_message": (validation_msg[:1000]
                               if validation_msg else ""),

        # Status
        "solver_exit_code": rc,
        "solver_crashed": rc != 0,
        "conversion_exit_code": conv_rc,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # 6. Epoch metrics CSV
    if epoch_rows:
        _write_epoch_csv(run_dir / "epoch_metrics.csv", epoch_rows)

    return summary


def _write_epoch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Flatten per-op dicts into `{repair,destroy}_{op}_time_ms` and `_count` columns."""
    flat_rows = []
    all_keys: set[str] = set()
    for r in rows:
        flat: dict[str, Any] = {}
        for k, v in r.items():
            if k in ("repair_ops", "destroy_ops"):
                prefix = k.split("_")[0]
                for op_name, stats in (v or {}).items():
                    flat[f"{prefix}_{op_name}_time_ms"] = stats.get("time_ms")
                    flat[f"{prefix}_{op_name}_count"] = stats.get("count")
            else:
                flat[k] = v
        flat_rows.append(flat)
        all_keys.update(flat.keys())

    keys = sorted(all_keys)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: r.get(k) for k in keys})


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def build_matrix(
    instances: list[Instance],
    selections: list[str],
    config_names: list[str],
    budgets: list[int],
    seeds: list[int],
    epoch_length_s: float,
    ml_model_dir: Optional[Path],
    solver: Path,
    skip_validation: bool,
    exclude_selection_from_budget: bool = False,
    log_epoch_iters: bool = False,
) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    """Cartesian product → list of RunSpec. Also returns a list of skipped combos."""
    specs: list[RunSpec] = []
    skips: list[dict[str, Any]] = []

    for inst in instances:
        for sel in selections:
            for cfg_name in config_names:
                cfg = OPERATOR_CONFIGS[cfg_name]
                for budget in budgets:
                    for seed in seeds:
                        # ML + oversize skip
                        if sel == "ml" and inst.duration in ML_OVERSIZE_DURATIONS:
                            skips.append({
                                "instance": inst.basename,
                                "selection": sel,
                                "config": cfg_name,
                                "budget_ms": budget,
                                "seed": seed,
                                "reason": ("ML models not trained on instances "
                                           f"larger than 05min; skipping {inst.duration}"),
                            })
                            continue
                        specs.append(RunSpec(
                            instance=inst,
                            selection=sel,
                            config_name=cfg_name,
                            destroy=cfg["destroy"],
                            repair=cfg["repair"],
                            budget_ms=budget,
                            seed=seed,
                            epoch_length_s=epoch_length_s,
                            ml_model_dir=ml_model_dir if sel == "ml" else None,
                            solver=solver,
                            skip_validation=skip_validation,
                            exclude_selection_from_budget=exclude_selection_from_budget,
                            log_epoch_iters=log_epoch_iters,
                        ))
    return specs, skips


# ---------------------------------------------------------------------------
# CLI + dispatch
# ---------------------------------------------------------------------------
def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in _parse_csv_list(s)]


def main():
    ap = argparse.ArgumentParser(
        description="Dynamic Fiedler benchmark runner (ML vs roulette vs random)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Instance selection
    ap.add_argument("--max-duration", default="02min",
                    choices=DURATION_TIERS,
                    help="Include all durations <= this tier (default: 02min)")
    ap.add_argument("--start", default="18-00",
                    choices=["07-00", "18-00", "both"],
                    help="Start time filter (default: 18-00)")
    ap.add_argument("--delays", default="03min,05min,10min",
                    help="Comma-separated delay values (default: all 3)")
    ap.add_argument("--instances", default=None,
                    help="Override: comma-separated explicit instance basenames")

    # Run configuration
    ap.add_argument("--selections", default="random,roulette,ml",
                    help="Comma-separated: random, roulette, ml (default: all 3)")
    ap.add_argument("--configs", default="full",
                    help="Comma-separated: classical, full, novel-focused "
                         "(default: full)")
    ap.add_argument("--budgets", default="5000",
                    help="Comma-separated epoch budgets in ms (default: 5000)")
    ap.add_argument("--seeds", default="42,123,456",
                    help="Comma-separated seeds (default: 42,123,456)")
    ap.add_argument("--epoch-length", type=float, default=30.0,
                    help="Epoch length in seconds (default: 30)")

    # Execution
    ap.add_argument("--workers", type=int, default=None,
                    help=f"Parallel workers (default: min(4, cpu_count//2))")
    ap.add_argument("--force-parallel", action="store_true",
                    help="Allow parallelism on heavy instances (>=05min)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Results root (default: bench_results/)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if already completed (creates new timestamped dir)")
    ap.add_argument("--skip-validation", action="store_true",
                    help="Skip solution_checker step")
    ap.add_argument("--force-ml-oversize", action="store_true",
                    help="Allow ML runs on 15min/30min instances (out of distribution)")
    ap.add_argument("--exclude-selection-from-budget", action="store_true",
                    help="Subtract operator-selection wall time (ML inference, etc) "
                         "from the epoch budget so ALNS gets the same compute "
                         "regardless of selection strategy.")
    ap.add_argument("--log-epoch-iters", action="store_true",
                    help="Pass --log-epoch-iters to the solver so each epoch's "
                         "ALNS call emits per-100-iter `[Epoch N] iter K @ <ms>ms: "
                         "current=..., best=..., temp=...` lines into stdout.log. "
                         "Time spent printing is excluded from the SA budget.")
    ap.add_argument("--ml-model-dir", default=str(DEFAULT_ML_MODEL_DIR),
                    help="Directory with feature_manifest.json + .onnx files")
    ap.add_argument("--solver", default=str(DEFAULT_SOLVER),
                    help="Path to darp_solver binary")
    ap.add_argument("--converted-dir", default=str(DEFAULT_CONVERTED_DIR),
                    help="Directory with converted Fiedler .txt files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print matrix; don't execute")

    args = ap.parse_args()

    # Parse list flags
    delays = _parse_csv_list(args.delays)
    selections = _parse_csv_list(args.selections)
    configs = _parse_csv_list(args.configs)
    budgets = _parse_csv_ints(args.budgets)
    seeds = _parse_csv_ints(args.seeds)

    for s in selections:
        if s not in ("random", "roulette", "ml"):
            print(f"ERROR: unknown selection {s!r}", file=sys.stderr)
            sys.exit(1)
    for c in configs:
        if c not in OPERATOR_CONFIGS:
            print(f"ERROR: unknown config {c!r}. Available: "
                  f"{list(OPERATOR_CONFIGS)}", file=sys.stderr)
            sys.exit(1)

    solver = Path(args.solver)
    if not solver.exists():
        print(f"ERROR: solver not found at {solver}", file=sys.stderr)
        print("  Build: cd cpp/build && cmake -DENABLE_ML_SELECTION=ON .. && make",
              file=sys.stderr)
        sys.exit(1)

    ml_model_dir = Path(args.ml_model_dir)
    if "ml" in selections and not (ml_model_dir / "feature_manifest.json").exists():
        print(f"ERROR: ML model dir invalid (missing feature_manifest.json): "
              f"{ml_model_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover + filter instances
    all_instances = discover_instances(Path(args.converted_dir))
    if args.instances:
        wanted = set(_parse_csv_list(args.instances))
        instances = [i for i in all_instances if i.basename in wanted]
        missing = wanted - {i.basename for i in instances}
        if missing:
            print(f"ERROR: instances not found: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
    else:
        instances = filter_instances(all_instances, args.max_duration, args.start, delays)

    if not instances:
        print("ERROR: no instances matched filters.", file=sys.stderr)
        sys.exit(1)

    # Build matrix
    specs, skips = build_matrix(
        instances=instances,
        selections=selections,
        config_names=configs,
        budgets=budgets,
        seeds=seeds,
        epoch_length_s=args.epoch_length,
        ml_model_dir=ml_model_dir,
        solver=solver,
        skip_validation=args.skip_validation,
        exclude_selection_from_budget=args.exclude_selection_from_budget,
        log_epoch_iters=args.log_epoch_iters,
    )

    # Apply --force-ml-oversize: re-admit those combos
    if args.force_ml_oversize and skips:
        for skip in skips[:]:
            if "ML models not trained" in skip["reason"]:
                # Reconstruct a spec for it
                inst = next(i for i in instances if i.basename == skip["instance"])
                cfg = OPERATOR_CONFIGS[skip["config"]]
                specs.append(RunSpec(
                    instance=inst,
                    selection=skip["selection"],
                    config_name=skip["config"],
                    destroy=cfg["destroy"],
                    repair=cfg["repair"],
                    budget_ms=skip["budget_ms"],
                    seed=skip["seed"],
                    epoch_length_s=args.epoch_length,
                    ml_model_dir=ml_model_dir,
                    solver=solver,
                    skip_validation=args.skip_validation,
                    exclude_selection_from_budget=args.exclude_selection_from_budget,
                    log_epoch_iters=args.log_epoch_iters,
                ))
                skips.remove(skip)

    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Persist skipped combos
    if skips:
        with open(out_dir / "skipped_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(skips[0].keys()))
            w.writeheader()
            w.writerows(skips)

    # Reporting
    print(f"=== Dynamic Fiedler Benchmark ===")
    print(f"Instances ({len(instances)}):")
    for inst in instances:
        print(f"  - {inst.basename} [{inst.duration}, {inst.delay}, start={inst.start}]")
    print(f"Selections: {selections}")
    print(f"Configs:    {configs}")
    print(f"Budgets:    {budgets} ms")
    print(f"Seeds:      {seeds}")
    print(f"Epoch:      {args.epoch_length}s")
    print(f"")
    print(f"Matrix size: {len(specs)} runs  (+{len(skips)} skipped)")
    if skips:
        print(f"  Skipped combos recorded to {out_dir / 'skipped_runs.csv'}")

    if args.dry_run:
        print(f"\n--dry-run: not executing.")
        return 0

    # Decide workers
    has_heavy = any(s.instance.duration in HEAVY_DURATIONS for s in specs)
    if has_heavy and not args.force_parallel:
        n_workers = 1
        print(f"\nHeavy instances present (05min+); forcing sequential. "
              f"Use --force-parallel to override.")
    elif args.workers is not None:
        n_workers = max(1, args.workers)
    else:
        n_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    print(f"Workers: {n_workers}")
    print(f"Out dir: {out_dir}")
    print()

    # Execute
    t0 = time.monotonic()
    completed = 0

    if n_workers == 1:
        for spec in specs:
            summary = execute_run(spec, runs_dir, args.force)
            completed += 1
            _print_progress(completed, len(specs), spec.run_id, summary)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(execute_run, spec, runs_dir, args.force): spec
                for spec in specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    summary = fut.result()
                except Exception as e:
                    summary = {"run_id": spec.run_id, "error": str(e)}
                completed += 1
                _print_progress(completed, len(specs), spec.run_id, summary)

    elapsed = time.monotonic() - t0
    print(f"\n=== Done ===")
    print(f"Completed {completed}/{len(specs)} in {elapsed:.1f}s")
    print(f"Results: {out_dir}")
    print(f"Next: python benchmarks/aggregate_results.py --results-dir {out_dir}")
    return 0


def _print_progress(done: int, total: int, run_id: str, summary: dict[str, Any]) -> None:
    status = "OK"
    if summary.get("skipped_reason"):
        status = "SKIP"
    elif summary.get("solver_crashed"):
        status = "CRASH"
    elif summary.get("validation_passed") is False:
        status = "INVALID"
    cost = summary.get("cost")
    cost_str = f"cost={cost:.0f}" if cost else "cost=n/a"
    print(f"  [{done}/{total}] {status}  {run_id}  {cost_str}")


if __name__ == "__main__":
    sys.exit(main() or 0)
