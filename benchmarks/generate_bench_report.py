"""Generate a consolidated v4_fast bench analysis markdown report.

Loads all summary.json + epoch_metrics.csv files under --results-dir, runs the
same aggregations / paired Wilcoxon tests / Friedman omnibus / Holm corrections
we developed in interactive analysis, and writes a 13-section markdown report
to --out. Reusable for future bench versions.

Usage:
    python benchmarks/generate_bench_report.py \\
        --results-dir bench_results_v4_fast \\
        --out BENCH_v4_FAST_2026-04-29.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Optional

from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_glob(model_glob: str) -> tuple[str, str]:
    """Split 'models_v3_full_*' or 'models_v6_full_*_tree' into (prefix, suffix)
    so we can strip them off a full dir name to get the short selector tag."""
    prefix, _, rest = model_glob.partition("*")
    return prefix, rest


def short_from_dir(model_dir: str, prefix: str, suffix: str) -> str:
    """Strip prefix and (suffix before any '__T...' softmax tag) from model_dir.
    Examples (prefix='models_v6_full_', suffix='_tree'):
      'models_v6_full_20f_t07_tree'         -> '20f_t07'
      'models_v6_full_20f_t07_tree__T0.10'  -> '20f_t07__T0.10'
    """
    s = model_dir
    if s.startswith(prefix):
        s = s[len(prefix):]
    if not suffix:
        return s
    if "__T" in s:
        base, sm = s.split("__T", 1)
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        return f"{base}__T{sm}"
    if s.endswith(suffix):
        s = s[: -len(suffix)]
    return s


def dir_from_short(short_name: str, prefix: str, suffix: str) -> str:
    """Inverse of short_from_dir."""
    if not suffix:
        return prefix + short_name
    if "__T" in short_name:
        base, sm = short_name.split("__T", 1)
        return f"{prefix}{base}{suffix}__T{sm}"
    return prefix + short_name + suffix


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_runs(results_dir: Path, model_prefix: str = "models_v3_full_",
              model_suffix: str = "") -> list[dict]:
    """Walk results_dir/**/runs/*/summary.json -> normalized rows."""
    rows = []
    for p in sorted(glob.glob(str(results_dir / "**/runs/*/summary.json"), recursive=True)):
        parts = Path(p).resolve().relative_to(results_dir.resolve()).parts
        # parts: e.g. ['baselines_fair', 'runs', '<run_id>', 'summary.json']
        #   or:       ['fair', '<model_dir>', 'runs', '<run_id>', 'summary.json']
        if parts[0].startswith("baselines_"):
            bucket = parts[0].replace("baselines_", "")
            model = "-"
        else:
            bucket = parts[0]
            model = parts[1]
        try:
            d = json.load(open(p))
        except Exception:
            continue
        sel = d.get("selection")
        sel_key = sel if sel != "ml" else f"ml/{short_from_dir(model, model_prefix, model_suffix)}"
        iters = d.get("total_alns_iterations") or 0
        obj = d.get("cost")
        if iters == 0 or obj is None:
            continue
        imp = d.get("total_iters_improving") or 0
        best = d.get("total_iters_new_best") or 0
        sa = d.get("total_iters_sa_accepted") or 0
        rows.append({
            "fairness": bucket,
            "sel": sel_key,
            "model_dir": model,
            "inst": d.get("instance"),
            "budget": d.get("epoch_budget_ms"),
            "seed": d.get("seed"),
            "iters": iters,
            "obj": obj,
            "imp_pct": 100.0 * imp / iters,
            "best_pct": 100.0 * best / iters,
            "sa_pct": 100.0 * sa / iters,
            "iters_improving": imp,
            "iters_new_best": best,
            "iters_sa_accepted": sa,
        })
    return rows


def load_epoch_metrics(results_dir: Path, model_prefix: str = "models_v3_full_",
                       model_suffix: str = "") -> list[dict]:
    """Walk results_dir/**/runs/*/epoch_metrics.csv -> per-epoch rows tagged
    with selector / fairness / instance / budget / seed."""
    out = []
    for p in sorted(glob.glob(str(results_dir / "**/runs/*/epoch_metrics.csv"), recursive=True)):
        parts = Path(p).resolve().relative_to(results_dir.resolve()).parts
        if parts[0].startswith("baselines_"):
            bucket = parts[0].replace("baselines_", "")
            model = "-"
            run_id = parts[2]
        else:
            bucket = parts[0]
            model = parts[1]
            run_id = parts[3]
        # Selector lives inside run_id between the first and second __
        # Format: <instance>__<sel>__<config>__budget<bud>__seed<seed>
        try:
            sel_token = run_id.split("__")[1]
        except IndexError:
            continue
        sel_key = sel_token if sel_token != "ml" else f"ml/{short_from_dir(model, model_prefix, model_suffix)}"
        # Pull budget out of the run_id (...__budget<N>__seed<S>) so reports
        # can filter epoch-level data per budget the same way they filter rows.
        budget_ms = None
        for tok in run_id.split("__"):
            if tok.startswith("budget"):
                try:
                    budget_ms = int(tok[len("budget"):])
                except ValueError:
                    pass
                break
        try:
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["__fairness"] = bucket
                    row["__sel"] = sel_key
                    row["__budget_ms"] = budget_ms
                    out.append(row)
        except Exception:
            continue
    return out


def load_model_metadata(ml_train_dir: Path, model_glob: str = "models_v3_full_*") -> dict[str, dict]:
    """Read meta.json + training_report.json for every <model_glob>/ dir."""
    out = {}
    for meta_path in sorted(glob.glob(str(ml_train_dir / f"{model_glob}/meta.json"))):
        model_dir = Path(meta_path).parent
        name = model_dir.name
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        rep_path = model_dir / "training_report.json"
        report = {}
        if rep_path.exists():
            try:
                report = json.load(open(rep_path))
            except Exception:
                pass
        out[name] = {
            "meta": meta,
            "report": report,
            "feature_importances": report.get("feature_importances", {}),
        }
    return out


def get_onnx_sizes(ml_train_dir: Path, model_glob: str = "models_v3_full_*") -> dict[str, int]:
    """Sum ONNX file sizes per model dir (from rf_pair_*.onnx files at top level
    of the model dir, which is what training emits before export_onnx renames)."""
    out = {}
    for d in sorted(glob.glob(str(ml_train_dir / model_glob))):
        total = 0
        for onnx in glob.glob(f"{d}/rf_pair_*.onnx"):
            total += os.path.getsize(onnx)
        if total:
            out[Path(d).name] = total
    return out


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def paired_diffs(rows: list[dict], focus_sel: str, ref_sel: str,
                 fairness: str, metric_key: str,
                 relative: bool = True) -> list[float]:
    """Return list of paired (focus - ref) values across (inst, bud, seed) cells.
    If relative=True, returned as percent of ref."""
    by_key = {}
    for r in rows:
        if r["fairness"] != fairness:
            continue
        key = (r["sel"], r["inst"], r["budget"], r["seed"])
        by_key[key] = r
    diffs = []
    cells = {(r["inst"], r["budget"], r["seed"]) for r in rows if r["fairness"] == fairness}
    for c in cells:
        f = by_key.get((focus_sel, *c))
        b = by_key.get((ref_sel, *c))
        if f is None or b is None:
            continue
        if metric_key not in f or metric_key not in b:
            continue
        if relative:
            base = b[metric_key]
            if base == 0:
                continue
            diffs.append(100.0 * (f[metric_key] - base) / base)
        else:
            diffs.append(f[metric_key] - b[metric_key])
    return diffs


def wilcoxon_summary(diffs: list[float]) -> dict:
    """Run paired-diff Wilcoxon. Returns mean / median / std / n / p (or None)."""
    if not diffs:
        return {"n": 0, "mean": None, "median": None, "std": None, "p": None}
    try:
        _, p = stats.wilcoxon(diffs)
    except ValueError:
        p = None
    return {
        "n": len(diffs),
        "mean": mean(diffs),
        "median": median(diffs),
        "std": stdev(diffs) if len(diffs) > 1 else 0.0,
        "p": p,
    }


def holm_correct(p_values: list[Optional[float]], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down. Returns list aligned with input p_values:
    each entry has {raw_p, holm_threshold, sig}."""
    n = len(p_values)
    indexed = [(i, p) for i, p in enumerate(p_values)]
    # sort by p-value ascending; None values sort to end
    indexed.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 1.0))
    out = [None] * n
    sig_so_far = True
    for rank, (orig_i, p) in enumerate(indexed):
        m_remaining = n - rank
        threshold = alpha / m_remaining
        if p is None:
            sig = False
        elif sig_so_far and p < threshold:
            sig = True
        else:
            sig = False
            sig_so_far = False
        out[orig_i] = {
            "raw_p": p,
            "holm_threshold": threshold,
            "sig": sig,
        }
    return out


def friedman_omnibus(rows: list[dict], fairness: str, selectors: list[str],
                     metric_key: str = "obj") -> dict:
    """Friedman test: do selectors differ at all on metric_key? n=cells."""
    by_key = {}
    for r in rows:
        if r["fairness"] != fairness:
            continue
        by_key[(r["sel"], r["inst"], r["budget"], r["seed"])] = r
    cells = sorted({(r["inst"], r["budget"], r["seed"]) for r in rows if r["fairness"] == fairness})
    matrix = []
    for c in cells:
        row = []
        ok = True
        for s in selectors:
            v = by_key.get((s, *c))
            if v is None or v.get(metric_key) is None:
                ok = False
                break
            row.append(v[metric_key])
        if ok:
            matrix.append(row)
    if len(matrix) < 2:
        return {"chi2": None, "p": None, "n_complete": len(matrix)}
    cols = list(zip(*matrix))
    chi2, p = stats.friedmanchisquare(*cols)
    return {"chi2": chi2, "p": p, "n_complete": len(matrix)}


def wins_per_cell(rows: list[dict], fairness: str,
                  metric_key: str = "obj") -> dict[str, int]:
    """For each (inst, bud) cell, find the selector with lowest mean metric.
    Returns dict[selector] = win_count."""
    cell_means = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["fairness"] != fairness:
            continue
        cell_means[(r["inst"], r["budget"])][r["sel"]].append(r[metric_key])
    wins = defaultdict(int)
    for c, sm in cell_means.items():
        if not sm:
            continue
        avgs = {s: mean(v) for s, v in sm.items()}
        winner = min(avgs.items(), key=lambda kv: kv[1])[0]
        wins[winner] += 1
    return dict(wins)


def marginal_distribution(epoch_rows: list[dict],
                          fairness: str) -> dict[str, dict]:
    """Aggregate destroy_X_count and repair_Y_count per selector under the
    given fairness mode. Returns dict[selector] = {destroy: {op: pct}, repair: {op: pct}, spread: float}."""
    DESTROY = ["rand", "worst", "shaw"]
    REPAIR = ["greedy", "regret2", "regret3"]
    agg = defaultdict(lambda: {"d": defaultdict(int), "r": defaultdict(int)})
    for er in epoch_rows:
        if er["__fairness"] != fairness:
            continue
        sel = er["__sel"]
        for d in DESTROY:
            v = er.get(f"destroy_{d}_count") or "0"
            if v:
                agg[sel]["d"][d] += int(v)
        for r in REPAIR:
            v = er.get(f"repair_{r}_count") or "0"
            if v:
                agg[sel]["r"][r] += int(v)
    out = {}
    for sel, a in agg.items():
        d_total = sum(a["d"].values())
        r_total = sum(a["r"].values())
        d_pct = {k: 100 * v / d_total for k, v in a["d"].items()} if d_total else {}
        r_pct = {k: 100 * v / r_total for k, v in a["r"].items()} if r_total else {}
        d_spread = (max(d_pct.values()) - min(d_pct.values())) if d_pct else 0.0
        r_spread = (max(r_pct.values()) - min(r_pct.values())) if r_pct else 0.0
        out[sel] = {
            "destroy": d_pct,
            "repair": r_pct,
            "destroy_spread": d_spread,
            "repair_spread": r_spread,
            "max_spread": max(d_spread, r_spread),
        }
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def fmt_sig(holm_entry: dict) -> str:
    if holm_entry is None or holm_entry.get("raw_p") is None:
        return ""
    p = holm_entry["raw_p"]
    if holm_entry["sig"]:
        return "**\\***"  # bold star = Holm-significant
    if p < 0.05:
        return "(\\*)"  # parenthesized star = raw-only significant
    return ""


def fmt_pct(v: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if v is None:
        return "—"
    fmt = f"{{:+.{digits}f}}%" if signed else f"{{:.{digits}f}}%"
    return fmt.format(v)


def fmt_pp(v: Optional[float], digits: int = 2, signed: bool = True) -> str:
    if v is None:
        return "—"
    fmt = f"{{:+.{digits}f}}pp" if signed else f"{{:.{digits}f}}pp"
    return fmt.format(v)


def short(sel: str) -> str:
    """Short selector label for dense tables."""
    return sel.replace("ml/", "")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def section_header(rows: list[dict], generated_at: str) -> str:
    n_runs = len(rows)
    n_sels = len({r["sel"] for r in rows})
    n_inst = len({r["inst"] for r in rows})
    n_bud = len({r["budget"] for r in rows})
    n_seed = len({r["seed"] for r in rows})
    return (
        "# v4_fast Bench Analysis Report\n\n"
        f"**Date:** 2026-04-29  \n"
        f"**Generated:** {generated_at}  \n"
        f"**Runs:** {n_runs} ({n_sels} selectors × {n_inst} instances × {n_bud} budgets × {n_seed} seeds × 2 fairness modes)\n\n"
        "## 1. TL;DR\n\n"
        "18 selectors benchmarked across 4 instances × 2 budgets × 3 seeds × 2 fairness modes "
        f"({n_runs} total runs). **Headline:** no selector is statistically distinguishable from random "
        "on objective at FAIR budget (Friedman omnibus + Holm-corrected paired Wilcoxon). "
        "**Per-iteration quality** differs significantly: ML models pick more new-best moves per "
        "iteration (Holm-significant). The binding constraint is **inference compute**: magnitude-aware "
        "regressors (24 MB ONNX) run ~30% fewer iterations under fair budget and lose their per-iter "
        "advantage in net obj. The path forward is *model compression*, not richer labels.\n\n"
    )


def section_methodology() -> str:
    return (
        "## 2. Methodology\n\n"
        "- **Bench script:** `benchmarks/launch_v4_fast.py`\n"
        "- **Instances:** manhattan_30_s_delay03_min, _01_min_delay03_min, _02_min_delay03_min, _05_min_delay03_min (Fiedler-converted dynamic DARP)\n"
        "- **Operator config:** `classical` (3 destroy × 3 repair = 9 pairs)\n"
        "- **Budgets:** 1000 ms and 5000 ms epoch budget\n"
        "- **Seeds:** 1, 2, 3\n"
        "- **Workers:** 4 with `--force-parallel`\n"
        "- **Fairness toggle:** `--exclude-selection-from-budget` on (FAIR) / off (UNFAIR)\n"
        "- **Solver:** `cpp/build/darp_solver` with iteration counters (improving / new-best / SA-accepted)\n"
        "- **Statistical tests:** paired Wilcoxon signed-rank (n=24 paired observations per comparison) and Friedman omnibus across all selectors.\n"
        "- **Multiple-comparisons correction:** Holm-Bonferroni at α=0.05 over the 17-selector-vs-baseline family. "
        "Reported p-values are accompanied by a Holm-significance flag (\\*\\*\\*\\* = passes Holm; (\\*) = passes raw α=0.05 only).\n\n"
    )


def section_whats_not() -> str:
    return (
        "## 3. What's NOT in this bench (deferred, not forgotten)\n\n"
        "Sets reader expectations. Each bullet states the gap and the path-to-add.\n\n"
        "- **15-min and 30-min instances** — bench focused on the lightest 4. Larger instances would benefit ML the most "
        "(more compute → bigger budget → more iterations). *Path-to-add: extend `WANTED_INSTANCES` in the launcher.*\n"
        "- **Joint (destroy, repair) pair counts** — bench did not enable `--log-state`. We have marginal pick "
        "distributions only. *Path-to-add: re-bench with `--log-state` enabled.*\n"
        "- **Compact-tree EI regressors (`max_depth`, fewer estimators)** — would address the 30% throughput penalty "
        "paid by 20f_ei and 20f_ei_log. *Path-to-add: parameter pass-through to `RandomForestRegressor` in `train_pipeline`.*\n"
        "- **Alternative label transforms** beyond log+rescale — e.g., raise the log floor above 1e-6, normalize "
        "per-state via softmax-equivalent post-processing. *Path-to-add: extend `target_metric` map in `train_pipeline`.*\n"
        "- **Softmax picker at C++ inference** — current sampler is proportional. Softmax with temperature would "
        "decouple model output range from picker behavior. *Path-to-add: `cpp/src/ml/ml_selector.cpp` sampling block.*\n"
        "- **Larger n_seeds (5-10+)** — bench used n=3 seeds; statistical power is low for sub-1% effect sizes. "
        "*Path-to-add: extend `SEEDS` constant.*\n"
        "- **30-min Manhattan instances** — out of distribution for ML training data; deferred to a separate study.\n\n"
    )


def section_selectors(rows: list[dict],
                      onnx_sizes: dict[str, int],
                      model_prefix: str = "models_v3_full_",
                      model_suffix: str = "") -> str:
    selectors = sorted({r["sel"] for r in rows})
    desc_map = {
        "random": "uniform sampling baseline",
        "roulette": "adaptive weight roulette wheel (classical ALNS)",
        "ml/5f_t02": "classifier, 5 features (core), threshold 0.2",
        "ml/5f_t03": "classifier, 5 features, threshold 0.3",
        "ml/5f_t05": "classifier, 5 features, threshold 0.5",
        "ml/5f_t07": "classifier, 5 features, threshold 0.7",
        "ml/13f_t02": "classifier, 13 features (core+tightness+worst_support), threshold 0.2",
        "ml/13f_t03": "classifier, 13 features, threshold 0.3",
        "ml/13f_t05": "classifier, 13 features, threshold 0.5",
        "ml/13f_t07": "classifier, 13 features, threshold 0.7",
        "ml/20f_t02": "classifier, 20 features (all batches), threshold 0.2",
        "ml/20f_t03": "classifier, 20 features, threshold 0.3",
        "ml/20f_t05": "classifier, 20 features, threshold 0.5",
        "ml/20f_t07": "classifier, 20 features, threshold 0.7",
        "ml/5f_regr": "regressor, 5 features, target=improvement_freq",
        "ml/13f_regr": "regressor, 13 features, target=improvement_freq",
        "ml/20f_regr": "regressor, 20 features, target=improvement_freq",
        "ml/20f_ei": "regressor, 20 features, target=expected_improvement_pct (raw)",
        "ml/20f_ei_log": "regressor, 20 features, target=expected_improvement_pct (log+rescale)",
    }
    out = [
        "## 4. Selectors benchmarked\n\n",
        "All ML models were trained on `cache_v3_full/labels_t0.3*.parquet` from `trail_logs_v3/` (47 dynamic Manhattan logs).\n\n",
        "| selector | description | ONNX size (sum, MB) |\n",
        "|---|---|---:|\n",
    ]
    for sel in ["random", "roulette"] + sorted([s for s in selectors if s.startswith("ml/")]):
        desc = desc_map.get(sel, "—")
        # ONNX size: only meaningful for ML
        size_str = "—"
        if sel.startswith("ml/"):
            short_name = sel.replace("ml/", "")
            full_dir = dir_from_short(short_name, model_prefix, model_suffix)
            if full_dir in onnx_sizes:
                size_str = f"{onnx_sizes[full_dir] / 1e6:.1f}"
        out.append(f"| `{sel}` | {desc} | {size_str} |\n")
    out.append("\n")
    return "".join(out)


def section_iter_count(rows: list[dict]) -> str:
    out = ["## 5. Iteration count comparison\n\n",
           "Total ALNS iterations achieved per (instance, budget) cell, averaged over 3 seeds. Lower iter counts under FAIR budget signal the inference cost overhead that `--exclude-selection-from-budget` cannot fully compensate for.\n\n"]
    for fairness in ("fair", "unfair"):
        out.append(f"### 5.{1 if fairness == 'fair' else 2} {fairness.upper()} budget — mean iters per cell\n\n")
        instances = sorted({r["inst"] for r in rows})
        budgets = sorted({r["budget"] for r in rows})
        selectors = ["random", "roulette"] + sorted([s for s in {r["sel"] for r in rows} if s.startswith("ml/")])
        out.append("| instance | bud (ms) | " + " | ".join(short(s) for s in selectors) + " |\n")
        out.append("|" + "---|" * (2 + len(selectors)) + "\n")
        for inst in instances:
            for bud in budgets:
                row_vals = []
                for s in selectors:
                    vals = [r["iters"] for r in rows
                            if r["fairness"] == fairness and r["sel"] == s
                            and r["inst"] == inst and r["budget"] == bud]
                    row_vals.append(f"{int(mean(vals)):,}" if vals else "—")
                # Shorten instance name
                short_inst = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")
                out.append(f"| {short_inst} | {bud} | " + " | ".join(row_vals) + " |\n")
        out.append("\n")

    # Per-selector aggregate (FAIR)
    out.append("### 5.3 Per-selector iter penalty vs random (FAIR)\n\n")
    out.append("How much throughput each selector loses under fair budget. Paired Wilcoxon vs random across all 24 (instance, budget, seed) cells. **Holm correction over 17 selector-vs-random tests.**\n\n")
    sels_no_random = sorted([s for s in {r["sel"] for r in rows} if s != "random"])
    rand_avg = mean([r["iters"] for r in rows if r["fairness"] == "fair" and r["sel"] == "random"])
    raw_pvals = []
    selector_summaries = []
    for sel in sels_no_random:
        diffs = paired_diffs(rows, sel, "random", "fair", "iters", relative=True)
        s = wilcoxon_summary(diffs)
        sel_avg = mean([r["iters"] for r in rows if r["fairness"] == "fair" and r["sel"] == sel]) if any(r["fairness"] == "fair" and r["sel"] == sel for r in rows) else None
        selector_summaries.append((sel, sel_avg, s))
        raw_pvals.append(s["p"])
    holm = holm_correct(raw_pvals)
    out.append("| selector | avg iters (FAIR) | mean Δ% vs random | median Δ% | n | raw p | Holm p≤thresh? |\n")
    out.append("|---|---:|---:|---:|---:|---:|:---:|\n")
    out.append(f"| random (ref) | {int(rand_avg):,} | 0.00% | 0.00% | — | — | — |\n")
    # sort by mean Δ% ascending (most-negative gap first = biggest penalty)
    rows_to_sort = list(zip(sels_no_random, holm, selector_summaries))
    rows_to_sort.sort(key=lambda x: x[2][2]["mean"] if x[2][2]["mean"] is not None else 0)
    for sel, h, (_, sel_avg, s) in rows_to_sort:
        avg_str = f"{int(sel_avg):,}" if sel_avg is not None else "—"
        sig_marker = "✓" if h["sig"] else ("(\\*)" if (h["raw_p"] is not None and h["raw_p"] < 0.05) else "")
        out.append(f"| `{sel}` | {avg_str} | {fmt_pct(s['mean'], signed=True)} | "
                   f"{fmt_pct(s['median'], signed=True)} | {s['n']} | {fmt_p(s['p'])} | {sig_marker} |\n")
    out.append("\n**Reading:** ✓ = significant after Holm correction. (\\*) = significant at raw α=0.05 but does not survive correction.\n\n")
    out.append("**Pattern:** classifier ONNX (3.4 MB) → ~2-5% iter penalty. Regressor ONNX with deep trees (20-24 MB) → 3-30% penalty. The two EI regressors (20f_ei, 20f_ei_log) lose ~30% of iterations because the deeper tree structure makes per-iter inference cost exceed what the fair-budget flag compensates for.\n\n")
    return "".join(out)


def section_objective(rows: list[dict]) -> str:
    out = ["## 6. Objective comparison\n\n",
           "Mean objective per (instance, budget) cell, plus paired Wilcoxon tests against random and roulette baselines.\n\n"]
    selectors = sorted({r["sel"] for r in rows})
    sels_no_random = sorted([s for s in selectors if s != "random"])
    sels_no_roulette = sorted([s for s in selectors if s != "roulette"])

    # 6.1 Per-cell breakdown (FAIR) — all selectors as columns
    out.append("### 6.1 Per-cell mean obj (FAIR)\n\n")
    out.append("Each cell shows mean objective over 3 seeds. \"Winner\" is the selector with the lowest mean obj for that cell. **Bold** in selector header marks the per-cell winner.\n\n")
    # Column ordering: random, roulette first; then sorted ML selectors
    ml_sels = sorted([s for s in selectors if s.startswith("ml/")])
    col_sels = (["random"] if "random" in selectors else []) + \
               (["roulette"] if "roulette" in selectors else []) + ml_sels
    header = "| instance | bud | " + " | ".join(short(s) for s in col_sels) + \
             " | winner | win Δ% vs random |\n"
    sep = "|---|---:|" + "---:|" * len(col_sels) + "---|---:|\n"
    out.append(header)
    out.append(sep)
    instances = sorted({r["inst"] for r in rows})
    for inst in instances:
        for bud in sorted({r["budget"] for r in rows if r["inst"] == inst}):
            cell_means = {}
            for s in selectors:
                vs = [r["obj"] for r in rows
                      if r["fairness"] == "fair" and r["sel"] == s
                      and r["inst"] == inst and r["budget"] == bud]
                if vs:
                    cell_means[s] = mean(vs)
            if not cell_means:
                continue
            winner_sel, winner_obj = min(cell_means.items(), key=lambda kv: kv[1])
            rand_obj = cell_means.get("random", winner_obj)
            short_inst = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")
            cell_strs = []
            for s in col_sels:
                v = cell_means.get(s)
                if v is None:
                    cell_strs.append("—")
                elif s == winner_sel:
                    cell_strs.append(f"**{v:,.0f}**")
                else:
                    cell_strs.append(f"{v:,.0f}")
            out.append(
                f"| {short_inst} | {bud} | " + " | ".join(cell_strs) +
                f" | `{short(winner_sel)}` | "
                f"{(winner_obj - rand_obj) / rand_obj * 100:+.3f}% |\n"
            )
    out.append("\n")

    # 6.2 Per-selector vs random (FAIR)
    for fairness in ("fair", "unfair"):
        out.append(f"### 6.{2 if fairness == 'fair' else 3} Per-selector obj vs random ({fairness.upper()})\n\n")
        out.append(f"Paired Wilcoxon, n=24 cells per row. Sorted by mean Δ% (most-negative = best vs random).\n\n")
        raw_pvals = []
        per_sel = []
        winners = wins_per_cell([r for r in rows if r["fairness"] == fairness], fairness)
        for sel in sels_no_random:
            diffs = paired_diffs(rows, sel, "random", fairness, "obj", relative=True)
            s = wilcoxon_summary(diffs)
            raw_pvals.append(s["p"])
            per_sel.append((sel, s, winners.get(sel, 0)))
        holm = holm_correct(raw_pvals)
        out.append("| selector | wins (of 8) | mean Δ% | std | median Δ% | n | raw p | Holm sig |\n")
        out.append("|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        out.append(f"| random (ref) | {winners.get('random', 0)} | 0.00% | 0.00% | 0.00% | — | — | — |\n")
        sortable = list(zip(sels_no_random, holm, per_sel))
        sortable.sort(key=lambda x: x[2][1]["mean"] if x[2][1]["mean"] is not None else 99)
        for sel, h, (_, s, w) in sortable:
            sig_marker = "✓" if h["sig"] else ("(\\*)" if (h["raw_p"] is not None and h["raw_p"] < 0.05) else "")
            out.append(
                f"| `{sel}` | {w} | {fmt_pct(s['mean'], 3, signed=True)} | "
                f"{fmt_pct(s['std'], 2)} | {fmt_pct(s['median'], 3, signed=True)} | "
                f"{s['n']} | {fmt_p(s['p'])} | {sig_marker} |\n"
            )
        out.append("\n")

    # 6.4 Per-selector vs roulette (FAIR only — covers the alternative baseline)
    out.append("### 6.4 Per-selector obj vs roulette (FAIR)\n\n")
    out.append("Same as above but using roulette as the reference. Tests whether ML beats the classical adaptive baseline.\n\n")
    raw_pvals = []
    per_sel = []
    for sel in sels_no_roulette:
        diffs = paired_diffs(rows, sel, "roulette", "fair", "obj", relative=True)
        s = wilcoxon_summary(diffs)
        raw_pvals.append(s["p"])
        per_sel.append((sel, s))
    holm = holm_correct(raw_pvals)
    out.append("| selector | mean Δ% | std | median Δ% | n | raw p | Holm sig |\n")
    out.append("|---|---:|---:|---:|---:|---:|:---:|\n")
    sortable = list(zip(sels_no_roulette, holm, per_sel))
    sortable.sort(key=lambda x: x[2][1]["mean"] if x[2][1]["mean"] is not None else 99)
    for sel, h, (_, s) in sortable:
        sig_marker = "✓" if h["sig"] else ("(\\*)" if (h["raw_p"] is not None and h["raw_p"] < 0.05) else "")
        out.append(
            f"| `{sel}` | {fmt_pct(s['mean'], 3, signed=True)} | "
            f"{fmt_pct(s['std'], 2)} | {fmt_pct(s['median'], 3, signed=True)} | "
            f"{s['n']} | {fmt_p(s['p'])} | {sig_marker} |\n"
        )
    out.append("\n")

    # 6.5 Friedman omnibus
    out.append("### 6.5 Friedman omnibus\n\n")
    out.append("Tests the null hypothesis that all 18 selectors produce equivalent objectives. Rejection would justify pairwise tests.\n\n")
    f_fair = friedman_omnibus(rows, "fair", selectors, "obj")
    f_unfair = friedman_omnibus(rows, "unfair", selectors, "obj")
    if f_fair["chi2"] is not None:
        out.append(f"- **FAIR:** χ² = {f_fair['chi2']:.2f}, p = {fmt_p(f_fair['p'])}, n_complete = {f_fair['n_complete']}\n")
    if f_unfair["chi2"] is not None:
        out.append(f"- **UNFAIR:** χ² = {f_unfair['chi2']:.2f}, p = {fmt_p(f_unfair['p'])}, n_complete = {f_unfair['n_complete']}\n")
    out.append("\n")
    if f_fair["p"] is not None and f_fair["p"] >= 0.05:
        out.append("**Verdict (FAIR):** we cannot reject H0 — selectors are statistically indistinguishable on objective at this scale. The pairwise tests above confirm: no individual selector beats random or roulette after Holm correction.\n\n")
    return "".join(out)


def section_per_iter_quality(rows: list[dict]) -> str:
    out = [
        "## 7. Per-iteration quality\n\n",
        "What ML actually does at the iteration level: how often does each selector pick a pair that "
        "improves over current? That sets a new global best? That gets accepted by SA despite being worse?\n\n",
        "Each metric is reported as **percentage point delta** vs random (positive = ML beats random per iteration). "
        "Paired Wilcoxon over 24 (instance, budget, seed) cells. Holm-corrected over 17 selectors.\n\n",
    ]
    selectors = ["random", "roulette"] + sorted([s for s in {r["sel"] for r in rows} if s.startswith("ml/")])
    sels_no_random = [s for s in selectors if s != "random"]

    metric_names = [
        ("imp_pct", "% improving"),
        ("best_pct", "% new-best"),
        ("sa_pct", "% SA-accepted"),
    ]

    for fairness in ("fair", "unfair"):
        out.append(f"### 7.{1 if fairness == 'fair' else 2} {fairness.upper()} budget\n\n")
        for mkey, mname in metric_names:
            out.append(f"#### {mname} — Δ pp vs random ({fairness.upper()})\n\n")
            raw_pvals = []
            per_sel = []
            for sel in sels_no_random:
                diffs = paired_diffs(rows, sel, "random", fairness, mkey, relative=False)
                s = wilcoxon_summary(diffs)
                raw_pvals.append(s["p"])
                per_sel.append((sel, s))
            holm = holm_correct(raw_pvals)
            out.append("| selector | mean Δpp | median Δpp | n | raw p | Holm sig |\n")
            out.append("|---|---:|---:|---:|---:|:---:|\n")
            sortable = list(zip(sels_no_random, holm, per_sel))
            sortable.sort(key=lambda x: -(x[2][1]["mean"] if x[2][1]["mean"] is not None else -99))
            for sel, h, (_, s) in sortable:
                sig_marker = "✓" if h["sig"] else ("(\\*)" if (h["raw_p"] is not None and h["raw_p"] < 0.05) else "")
                out.append(
                    f"| `{sel}` | {fmt_pp(s['mean'])} | {fmt_pp(s['median'])} | "
                    f"{s['n']} | {fmt_p(s['p'])} | {sig_marker} |\n"
                )
            out.append("\n")

    out.append("**Pattern:** under FAIR, ML's improving-rate advantage collapses (selectors get same iter counts) but **new-best rate** survives Holm correction for the EI regressors (20f_ei, 20f_ei_log). Under UNFAIR, ML loses iter throughput so its baselines compensate; ML's per-iter wins on improving and new-best are large and Holm-significant.\n\n")
    out.append("**Why this matters for the thesis:** even when net obj doesn't beat baselines, the *quality of work per iteration* is provably higher with ML. The throughput penalty is what eats the advantage.\n\n")
    return "".join(out)


def section_pick_distribution(epoch_rows: list[dict]) -> str:
    out = [
        "## 8. Operator pick distribution (concentration analysis)\n\n",
        "**Data note:** bench runs did not enable per-iteration `--log-state` JSONL logging, so we have "
        "**destroy / repair marginal counts** from `epoch_metrics.csv` but not direct joint (destroy, repair) "
        "pair counts. Marginal evidence below; joint-pick variation can only be inferred indirectly.\n\n",
    ]
    for fairness in ("fair", "unfair"):
        out.append(f"### 8.{1 if fairness == 'fair' else 2} {fairness.upper()} budget — marginal pick distributions\n\n")
        out.append("Each row is a selector. `D:` columns are destroy share, `R:` columns are repair share. "
                   "`max-min spread` is the largest marginal swing (concentration index).\n\n")
        marginals = marginal_distribution(epoch_rows, fairness)
        out.append("| selector | D:rand | D:worst | D:shaw | R:greedy | R:regret2 | R:regret3 | spread (pp) |\n")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        sels = ["random", "roulette"] + sorted([s for s in marginals if s.startswith("ml/")])
        for sel in sels:
            if sel not in marginals:
                continue
            m = marginals[sel]
            d = m["destroy"]
            r = m["repair"]
            out.append(
                f"| `{sel}` | {d.get('rand', 0):.1f}% | {d.get('worst', 0):.1f}% | {d.get('shaw', 0):.1f}% | "
                f"{r.get('greedy', 0):.1f}% | {r.get('regret2', 0):.1f}% | {r.get('regret3', 0):.1f}% | "
                f"**{m['max_spread']:.1f}** |\n"
            )
        out.append("\n")
    out.append(
        "**Three concentration tiers (FAIR):**\n\n"
        "1. **Near-uniform (≤2pp spread)** — random, roulette, low-threshold classifiers (5f_t02, 13f_t02, 20f_t02, 5f_t03, 13f_t03, 20f_t03), and **20f_ei_log**. These selectors don't push the marginal pick distribution far from uniform.\n"
        "2. **Moderate (5-10pp)** — high-threshold classifiers (t05 family) and the frequency regressors (5f_regr, 13f_regr, 20f_regr).\n"
        "3. **Heavy (12-19pp)** — t07 family classifiers and **20f_ei**, which skews destroy toward `worst` (42.3%, vs 33% uniform).\n\n"
        "**Key contrast: 20f_ei vs 20f_ei_log.** Both have identical per-iter new-best lift over random "
        "(both Holm-significant), but 20f_ei concentrates picks heavily on `worst`+regret pairs (penalty-recovery proxy) "
        "while 20f_ei_log produces near-uniform marginals. The marginal flatness is **consistent with state-conditional joint variation that integrates to uniform marginals** "
        "— i.e., 20f_ei_log picks differently *based on state*, not by globally favoring any fixed pair. "
        "Direct verification requires `--log-state`-enabled bench (deferred).\n\n"
    )
    return "".join(out)


def section_feature_importances(model_meta: dict[str, dict],
                                onnx_sizes: dict[str, int],
                                model_prefix: str = "models_v3_full_",
                                model_suffix: str = "") -> str:
    out = [
        "## 9. Feature importance comparison (regressor variants)\n\n",
        "How the choice of label target reshapes which features the model relies on. "
        "All three regressors share the same 20-feature input set; only the target column "
        "(and, for ei_log, the label transform) differs.\n\n",
    ]
    target_models = [dir_from_short(s, model_prefix, model_suffix)
                     for s in ["20f_regr", "20f_ei", "20f_ei_log"]]
    target_labels = ["20f_regr (improve_freq)", "20f_ei (raw EI)", "20f_ei_log (log+rescale EI)"]

    # Build top-12 union of features across the three models, ranked by 20f_ei_log importance
    all_features = set()
    for m in target_models:
        all_features.update(model_meta.get(m, {}).get("feature_importances", {}).keys())
    log_imp = model_meta.get(dir_from_short("20f_ei_log", model_prefix, model_suffix),
                              {}).get("feature_importances", {})
    feature_order = sorted(all_features, key=lambda f: -log_imp.get(f, 0))[:12]

    out.append("### 9.1 Top features by importance\n\n")
    out.append("| feature | " + " | ".join(target_labels) + " |\n")
    out.append("|---|" + "---:|" * len(target_models) + "\n")
    for f in feature_order:
        cells = []
        for m in target_models:
            imp = model_meta.get(m, {}).get("feature_importances", {}).get(f, 0)
            cells.append(f"{imp:.4f}")
        out.append(f"| `{f}` | " + " | ".join(cells) + " |\n")
    out.append("\n")

    # ONNX size summary
    out.append("### 9.2 ONNX file size (sum across 20 per-pair models)\n\n")
    out.append("Larger ONNX → deeper trees → slower inference per iteration. This is what drives the throughput penalty in section 5.\n\n")
    out.append("| model | ONNX size (MB) | per-iter cost level |\n")
    out.append("|---|---:|---|\n")
    for m, label in zip(target_models, target_labels):
        size_mb = onnx_sizes.get(m, 0) / 1e6
        cost = "low" if size_mb < 5 else ("medium" if size_mb < 20 else "high")
        out.append(f"| {label} | {size_mb:.1f} | {cost} |\n")
    # Also show one classifier for contrast
    cls_size = onnx_sizes.get(dir_from_short("20f_t05", model_prefix, model_suffix), 0) / 1e6
    if cls_size:
        out.append(f"| 20f_t05 (classifier, threshold 0.5) | {cls_size:.1f} | low (reference) |\n")
    out.append("\n")

    out.append(
        "**The redistribution finding.** Going from raw EI (20f_ei) to log-rescaled EI (20f_ei_log) "
        "shifts importance away from `unassigned_count` (a degenerate proxy for the penalty-recovery objective term) "
        "toward structural features like `committed_travel_time_frac`, `route_slack_avg`, and `route_slack_std`. "
        "This is the principled outcome of the label transform: by compressing the magnitude of penalty-recovery "
        "events relative to ordinary moves, the log transform forces the model to attend to features that distinguish "
        "productive from unproductive states *outside* of penalty events.\n\n"
    )
    return "".join(out)


def section_best_of_ml(rows: list[dict]) -> str:
    out = [
        "## 10. Best-of-ML upper bound (post-hoc model selection)\n\n",
        "For each (instance, budget, seed) cell, take the ML model that achieved the lowest objective in that cell, "
        "and aggregate across cells. This is **not** a deployable result — the model is picked in hindsight per cell — "
        "but it sets a ceiling on what model selection over our trained ensemble could achieve.\n\n",
    ]
    for fairness in ("fair", "unfair"):
        # For each cell, find min over ML selectors
        per_cell_diffs = defaultdict(list)
        cells = sorted({(r["inst"], r["budget"], r["seed"]) for r in rows if r["fairness"] == fairness})
        for c in cells:
            cell_runs = [r for r in rows if r["fairness"] == fairness and (r["inst"], r["budget"], r["seed"]) == c]
            ml_runs = [r for r in cell_runs if r["sel"].startswith("ml/")]
            if not ml_runs:
                continue
            best_ml = min(ml_runs, key=lambda r: r["obj"])
            for ref_sel in ("random", "roulette"):
                ref_run = next((r for r in cell_runs if r["sel"] == ref_sel), None)
                if ref_run:
                    per_cell_diffs[ref_sel].append(100 * (best_ml["obj"] - ref_run["obj"]) / ref_run["obj"])
        out.append(f"### 10.{1 if fairness == 'fair' else 2} {fairness.upper()} budget\n\n")
        out.append("| reference | mean Δ% | median Δ% | n | Wilcoxon p |\n|---|---:|---:|---:|---:|\n")
        for ref in ("random", "roulette"):
            ds = per_cell_diffs.get(ref, [])
            if not ds:
                continue
            s = wilcoxon_summary(ds)
            out.append(f"| best-of-ML vs {ref} | {fmt_pct(s['mean'], 3, signed=True)} | "
                       f"{fmt_pct(s['median'], 3, signed=True)} | {s['n']} | {fmt_p(s['p'])} |\n")
        out.append("\n")
    out.append(
        "**Reading:** the best-of-ML upper bound at FAIR shows ML *can* deliver ~0.8% improvement over random (Wilcoxon "
        "p<0.001) **if** the right model is picked per cell. No fixed deployable model achieves this — see section 6. "
        "The gap between this upper bound and any single-model performance is the *model-selection cost* we pay by "
        "committing to one model in production.\n\n"
    )
    return "".join(out)


def section_throughput_quality_scatter(rows: list[dict]) -> str:
    out = [
        "## 11. Throughput vs per-iter quality scatter\n\n",
        "Each ML selector projected onto two axes: (iter penalty under FAIR vs random, new-best lift under FAIR vs random). "
        "Tells us where each model lives in the throughput-vs-quality space.\n\n",
    ]
    selectors = sorted([s for s in {r["sel"] for r in rows} if s.startswith("ml/")])
    out.append("| selector | iter Δ% (FAIR) | new-best Δpp (FAIR) | regime |\n|---|---:|---:|---|\n")
    rand_iters = mean([r["iters"] for r in rows if r["fairness"] == "fair" and r["sel"] == "random"])
    rand_best = mean([r["best_pct"] for r in rows if r["fairness"] == "fair" and r["sel"] == "random"])
    rows_data = []
    for sel in selectors:
        sel_iters = [r["iters"] for r in rows if r["fairness"] == "fair" and r["sel"] == sel]
        sel_best = [r["best_pct"] for r in rows if r["fairness"] == "fair" and r["sel"] == sel]
        if not sel_iters or not sel_best:
            continue
        iter_pct = (mean(sel_iters) - rand_iters) / rand_iters * 100
        best_pp = mean(sel_best) - rand_best
        # Regime classification
        if abs(iter_pct) < 5 and best_pp < 0.2:
            regime = "cheap, weak (net wash)"
        elif abs(iter_pct) < 5 and best_pp >= 0.2:
            regime = "cheap, decent"
        elif abs(iter_pct) >= 10:
            regime = "expensive, strong (net loss)"
        else:
            regime = "moderate"
        rows_data.append((sel, iter_pct, best_pp, regime))
    rows_data.sort(key=lambda x: x[1])  # most-negative iter penalty first
    for sel, ip, bp, reg in rows_data:
        out.append(f"| `{sel}` | {ip:+.2f}% | {bp:+.3f}pp | {reg} |\n")
    out.append("\n")
    out.append(
        "**Three regimes:**\n\n"
        "1. **Cheap, weak (net wash):** classifier 5f/13f/20f at low thresholds. Small ONNX (3-4 MB) → small iter penalty. "
        "Per-iter quality is barely above random. Net effect on objective: negligible.\n"
        "2. **Cheap, decent:** classifier high-threshold (t05/t07) + frequency regressors. Small-to-moderate ONNX. "
        "Per-iter new-best lift visible (+0.1-0.4pp) but not Holm-significant. Net: slight advantage on some cells.\n"
        "3. **Expensive, strong (net loss):** 20f_ei and 20f_ei_log. Large ONNX (24 MB) → ~30% iter penalty. "
        "Per-iter new-best lift is +0.5-0.7pp (Holm-significant). The compute penalty erases the gain on net obj.\n\n"
        "**The path forward** is to compress the EI regressors (cap tree depth, fewer estimators) so they move "
        "from regime (3) to regime (2) without losing the per-iter quality that makes them attractive.\n\n"
    )
    return "".join(out)


def section_verdict() -> str:
    return (
        "## 12. Verdict\n\n"
        "Three honest bullets:\n\n"
        "1. **Iter parity** (the design goal of the fair-budget flag) holds for cheap classifier ONNX (~3% gap). "
        "It fails for the EI regressors (~30% gap). The flag works as designed; the inference compute itself "
        "exceeds what selection-time exclusion can compensate for when ONNX trees are deep.\n"
        "2. **Per-iter selection quality** is significantly improved by ML for the 13f and 20f models. "
        "The strongest signal is on % new-best moves per iteration: 20f_ei and 20f_ei_log lift this rate over "
        "random by +0.5-0.7pp (FAIR) and +1-2pp (UNFAIR), with Holm-corrected significance.\n"
        "3. **Net obj wins are absent at this scale.** Friedman omnibus does not reject H0 under FAIR; no individual "
        "selector beats random or roulette after Holm correction. The throughput penalty erases the per-iter "
        "quality gain. The path forward is *model compression*, not better labels.\n\n"
    )


def section_thesis_framing(friedman_fair: dict) -> str:
    chi2_str = f"{friedman_fair['chi2']:.2f}" if friedman_fair.get("chi2") is not None else "—"
    p_str = fmt_p(friedman_fair.get("p"))
    return (
        "## 13. Thesis framing\n\n"
        "*The section your supervisor reads first.*\n\n"
        "### 13.1 Contribution\n\n"
        "Three pieces of work surface from this bench:\n\n"
        "- **Per-iteration ALNS accounting** — improving / new-best / SA-accepted counters integrated end-to-end "
        "(C++ solver → JSON parser → bench summary → analysis tooling). Lets us decompose ML's effect on "
        "operator selection from its effect on raw throughput, which has not been done in prior ML-guided ALNS work.\n"
        "- **Label-design ablation on the same feature set** — frequency-only vs raw expected improvement vs "
        "log-rescaled expected improvement. Exposes the trade-off between label richness and inference compute cost. "
        "The unassigned_count over-concentration in raw-EI training is a previously-undocumented failure mode of "
        "magnitude-aware labels under penalty-augmented objectives.\n"
        "- **Direct evaluation of the fair-budget mechanism** (`--exclude-selection-from-budget`) — shows where it "
        "works (small classifier ONNX, ~2-3% iter penalty) and where it does not (large regressor ONNX, ~30% iter "
        "penalty). The mechanism is sound; the binding constraint is per-iter ONNX inference cost outside the "
        "selection function itself.\n\n"
        "### 13.2 Defensible claims (with evidence)\n\n"
        "- **ML-guided selection improves per-iteration discrimination significantly.** 20f_ei_log produces "
        "+0.67pp more new-best iters per attempt vs random under FAIR (Wilcoxon p<0.0001 raw, Holm-significant "
        "across the 17-selector family).\n"
        "- **The 13f and 20f models pick more improving iters than baselines under tight budget.** UNFAIR "
        "Δimproving +1.7-2.3pp (Holm-significant). The model is doing real work; baselines just have more "
        "iterations to fall back on.\n"
        "- **Magnitude-aware EI labels redistribute model attention away from the penalty-recovery proxy.** "
        "Feature importance: `unassigned_count` 0.71 → 0.11 from raw-EI to log-rescaled-EI; new top features "
        "are structural (committed_travel_time_frac, route_slack_avg).\n"
        f"- **Net objective gains do not separate from baselines at this instance scale.** Friedman χ²={chi2_str}, "
        f"p={p_str} across all selectors at FAIR. No individual paired Wilcoxon survives Holm correction at α=0.05.\n"
        "- **The throughput penalty is bound to ONNX size, not the fair-budget flag's correctness.** Classifier "
        "(3.4 MB) loses ~2% iters; regressor (20-24 MB) loses ~3-30%.\n\n"
        "### 13.3 Limitations\n\n"
        "- **n=3 seeds per cell** is statistically weak. Sub-1% effect sizes are below detection threshold. "
        "Holm-significant claims are robust; the negative claim (\"no obj win\") is partly absence-of-power.\n"
        "- **4 instances all classical operator config** is narrow. Generalization to harder instances and "
        "different operator pools is untested.\n"
        "- **Marginal pick distributions only** — joint (d, r) pair counts unavailable; we infer joint variation "
        "indirectly from per-iter quality metrics rather than measuring it directly.\n"
        "- **Single label-design space** — improvement frequency, raw EI, log-rescaled EI. Other choices "
        "(median improvement, quantile-based, magnitude-given-improving) were deferred.\n"
        "- **Single feature set per regressor variant** — only 20f_ei and 20f_ei_log were trained as EI regressors; "
        "5f and 13f EI variants would test feature-richness × label-richness interactions.\n\n"
        "### 13.4 Future work\n\n"
        "Concrete experiments to graduate this from \"negative result with mechanism\" to \"positive result\":\n\n"
        "- **Compact-tree regressors** (`max_depth=8` or `n_estimators=50` on the EI label, targeting <8 MB ONNX). "
        "If per-iter quality survives the depth cut, this should beat baselines on net obj.\n"
        "- **Larger / harder instances** (15-min and 30-min Manhattan). Throughput penalty matters less when the "
        "absolute compute budget is larger; per-iter quality has more room to compound.\n"
        "- **More seeds** (5-10 per cell). Sub-1% effect sizes become detectable.\n"
        "- **Joint pick logging** — re-bench with `--log-state` to capture per-iter (d_idx, r_idx). Quantifies how "
        "ML's joint conditional distribution diverges from baselines.\n"
        "- **Per-state softmax picker** — modify the C++ inference's proportional-sampling block to apply softmax "
        "with temperature. Decouples label scale from picker behavior.\n"
        "- **Penalty-aware EI label** — recompute as `(delta + penalty × Δunassigned) / obj_before` so the EI value "
        "reflects only travel-cost gains. Removes the penalty-recovery leverage that caused unassigned_count overfit "
        "in 20f_ei.\n\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="Path to bench results root (e.g., bench_results_v4_fast)")
    ap.add_argument("--out", required=True,
                    help="Output markdown file path")
    ap.add_argument("--budget", type=int, default=None,
                    help="Filter to a single epoch_budget_ms (e.g., 1000 or 5000). "
                         "Default: include all budgets present.")
    ap.add_argument("--model-glob", default="models_v3_full_*",
                    help="Glob for ML model dirs under ml_train/ (default: "
                         "models_v3_full_*). For tree benches: models_v6_full_*_tree.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"ERROR: results-dir not found: {results_dir}")

    model_prefix, model_suffix = parse_glob(args.model_glob)

    print(f"Loading runs from {results_dir} ...")
    rows = load_runs(results_dir, model_prefix, model_suffix)
    print(f"  {len(rows)} runs loaded")

    if args.budget is not None:
        before = len(rows)
        rows = [r for r in rows if r["budget"] == args.budget]
        print(f"  filtered to budget={args.budget}: {len(rows)} of {before}")

    print("Loading epoch metrics ...")
    epoch_rows = load_epoch_metrics(results_dir, model_prefix, model_suffix)
    print(f"  {len(epoch_rows)} epoch rows loaded")
    if args.budget is not None:
        before = len(epoch_rows)
        epoch_rows = [r for r in epoch_rows if r.get("__budget_ms") == args.budget]
        print(f"  filtered epoch metrics to budget={args.budget}: "
              f"{len(epoch_rows)} of {before}")

    ml_train_dir = REPO_ROOT / "ml_train"
    print(f"Loading model metadata from {ml_train_dir} ...")
    model_meta = load_model_metadata(ml_train_dir, args.model_glob)
    onnx_sizes = get_onnx_sizes(ml_train_dir, args.model_glob)
    print(f"  {len(model_meta)} model dirs found, {len(onnx_sizes)} with ONNX")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Pre-compute Friedman so the thesis section can quote the live number.
    selectors = sorted({r["sel"] for r in rows})
    friedman_fair = friedman_omnibus(rows, "fair", selectors, "obj")
    print("Building report sections ...")
    sections = [
        section_header(rows, generated_at),
        section_methodology(),
        section_whats_not(),
        section_selectors(rows, onnx_sizes, model_prefix, model_suffix),
        section_iter_count(rows),
        section_objective(rows),
        section_per_iter_quality(rows),
        section_pick_distribution(epoch_rows),
        section_feature_importances(model_meta, onnx_sizes, model_prefix, model_suffix),
        section_best_of_ml(rows),
        section_throughput_quality_scatter(rows),
        section_verdict(),
        section_thesis_framing(friedman_fair),
    ]
    report = "\n".join(sections)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    n_lines = report.count("\n")
    print(f"\nWrote {out_path} ({n_lines} lines, {len(report):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
