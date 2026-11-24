"""Within-epoch dominance % and mean Δ% tables for dynamic-mode benches.

Mirrors BENCH_v5_STATIC_2026-05-05.md Sections 8/9 but slices each epoch's
internal iter trajectory (recorded by --log-epoch-iters) into the same
five phases: first 10/20/50%, last 50/20%.

Pooling. Each (instance × seed × epoch × checkpoint) tuple is one observation
when ALL selectors under comparison have logged that exact (epoch, iter). The
phase fraction is `checkpoint_iter / common_max_iter_in_epoch` where the
common max is the min across selectors. Pooling across all epochs gives one
combined view.

ALIGNMENT NOTE. We compare strict obj at the same iter point ("iter K of ml
vs iter K of random"). Per-iter wall-cost differs across selectors so iter K
sits at different SA-cooling temperatures across selectors — the comparison
is iter-aligned, not temperature-aligned. This matches static Section 8/9's
methodology (static budgets are iter-based, so iter K does mean same
temperature there; here it doesn't, so the bias is small but worth noting).

Usage:
    python benchmarks/dynamic_within_epoch_dominance.py \\
        --results-dir bench_results_v6_30s \\
        --fairness fair \\
        --budget 1000 \\
        --out thesis_results/dynamic_within_epoch_dominance_v6_bud1000.md
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parent.parent

LINE_RE = re.compile(
    r"^\[Epoch\s+(\d+)\]\s+iter\s+(\d+)\s+@\s+\d+ms:\s+"
    r"current=\d+(?:\.\d+)?,\s+best=(\d+(?:\.\d+)?),",
    re.M,
)

INSTANCES = [
    "manhattan_30_s_delay03_min",
    "manhattan_01_min_delay03_min",
    "manhattan_02_min_delay03_min",
    "manhattan_05_min_delay03_min",
]

ML_SELECTORS = ["20f_t07__T0.10", "20f_t07", "20f_regr__T0.10", "20f_regr", "20f_ei_log"]


def selector_dir(results_dir: Path, fairness: str, sel: str,
                  ml_dir_template: str = "models_v3_full_{sel}") -> Path:
    if sel in ("random", "roulette"):
        return results_dir / f"baselines_{fairness}" / "runs"
    # Split off softmax suffix (e.g. "__T0.10") so it stays at the END of the dir,
    # after any template-applied suffix like "_tree".
    if "__T" in sel:
        base, softmax = sel.split("__T", 1)
        softmax = "__T" + softmax
    else:
        base, softmax = sel, ""
    return results_dir / fairness / (ml_dir_template.format(sel=base) + softmax) / "runs"


def load_run(results_dir: Path, fairness: str, sel: str, instance: str,
             budget: int, seed: int,
             ml_dir_template: str = "models_v3_full_{sel}") -> dict[int, dict[int, float]]:
    """Return {epoch_id: {iter: best_obj}} for one run."""
    sel_tag = sel if sel in ("random", "roulette") else "ml"
    run_dir = (selector_dir(results_dir, fairness, sel, ml_dir_template) /
               f"{instance}__{sel_tag}__classical__budget{budget}__seed{seed}")
    p = run_dir / "stdout.log"
    if not p.exists():
        raise FileNotFoundError(p)
    by_epoch: dict[int, dict[int, float]] = defaultdict(dict)
    for m in LINE_RE.finditer(p.read_text()):
        eid = int(m.group(1))
        it = int(m.group(2))
        best = float(m.group(3))
        by_epoch[eid][it] = best
    return by_epoch


def phase_predicate(phase: str, iter_val: int, max_iter: int) -> bool:
    frac = iter_val / max_iter if max_iter > 0 else 0.0
    if phase == "first10": return frac <= 0.10 + 1e-9
    if phase == "first20": return frac <= 0.20 + 1e-9
    if phase == "first50": return frac <= 0.50 + 1e-9
    if phase == "last50":  return frac >= 0.50 - 1e-9
    if phase == "last20":  return frac >= 0.80 - 1e-9
    raise ValueError(phase)


PHASES = [("first10", "first 10%"), ("first20", "first 20%"),
          ("first50", "first 50%"), ("last50",  "last 50%"),
          ("last20",  "last 20%")]


def collect(results_dir: Path, fairness: str, budget: int, seeds: list[int],
            epochs_filter: list[int] | None = None,
            ml_dir_template: str = "models_v3_full_{sel}"):
    """Returns nested dict: data[selector][(inst, seed, epoch, iter)] = best_obj.

    epochs_filter: if given, only include checkpoints from these epoch ids.
    """
    needed = ["random", "roulette"] + [f"ml/{s}" for s in ML_SELECTORS]
    out: dict[str, dict[tuple, float]] = {sel: {} for sel in needed}
    for sel in needed:
        sel_short = sel.replace("ml/", "")
        for inst in INSTANCES:
            for s in seeds:
                try:
                    parsed = load_run(results_dir, fairness, sel_short, inst, budget, s,
                                       ml_dir_template)
                except FileNotFoundError as e:
                    print(f"missing log: {sel} {inst} {budget} seed{s}: {e}")
                    return None
                for eid, iters in parsed.items():
                    if epochs_filter is not None and eid not in epochs_filter:
                        continue
                    for it, best in iters.items():
                        out[sel][(inst, s, eid, it)] = best
    return out


def aligned_tuples(data: dict[str, dict[tuple, float]],
                   selectors_required: list[str]):
    """Yield (inst, seed, epoch, iter, max_iter_in_epoch_aligned) for each tuple
    where all `selectors_required` have a logged value at that key.

    `max_iter_in_epoch_aligned` = min over `selectors_required` of (max iter
    each selector logged in that epoch). Ensures phase fractions are aligned.
    """
    common_keys = set.intersection(*[set(data[s]) for s in selectors_required])
    # Compute aligned max-iter per (inst, seed, epoch) cell
    max_iter: dict[tuple, int] = {}
    cell_groups: dict[tuple, set[int]] = defaultdict(set)
    for (inst, seed, epoch, it) in common_keys:
        cell_groups[(inst, seed, epoch)].add(it)
    for cell, iters in cell_groups.items():
        max_iter[cell] = max(iters)
    for (inst, seed, epoch, it) in sorted(common_keys):
        yield (inst, seed, epoch, it, max_iter[(inst, seed, epoch)])


def dominance_stats(data, ml_sel: str, comparators: list[str]):
    """For each phase, count (wins, total) where ml_sel < ALL comparators."""
    needed = [ml_sel] + comparators
    counts = {key: [0, 0] for (key, _) in PHASES}  # [wins, total]
    for inst, seed, epoch, it, max_it in aligned_tuples(data, needed):
        ml_v = data[ml_sel][(inst, seed, epoch, it)]
        wins = all(ml_v < data[c][(inst, seed, epoch, it)] for c in comparators)
        for (key, _) in PHASES:
            if phase_predicate(key, it, max_it):
                counts[key][1] += 1
                if wins:
                    counts[key][0] += 1
    return counts


def delta_stats(data, ml_sel: str, comparator: str):
    """For each phase, list of paired Δ% values (sel - comp) / comp * 100."""
    needed = [ml_sel, comparator]
    deltas = {key: [] for (key, _) in PHASES}
    for inst, seed, epoch, it, max_it in aligned_tuples(data, needed):
        m = data[ml_sel][(inst, seed, epoch, it)]
        c = data[comparator][(inst, seed, epoch, it)]
        if c <= 0:
            continue
        d = 100.0 * (m - c) / c
        for (key, _) in PHASES:
            if phase_predicate(key, it, max_it):
                deltas[key].append(d)
    return deltas


def fmt_dom(counts):
    cells = []
    for (key, _) in PHASES:
        w, t = counts[key]
        if t == 0:
            cells.append("— (0/0)")
        else:
            cells.append(f"{100.0 * w / t:.1f}% ({w}/{t})")
    return cells


def fmt_delta(deltas):
    cells = []
    for (key, _) in PHASES:
        ds = deltas[key]
        if not ds:
            cells.append("—")
        else:
            cells.append(f"{mean(ds):+.3f}%")
    return cells


def write_report(results_dir: Path, fairness: str, budget: int, seeds: list[int],
                 out_path: Path, epochs_filter: list[int] | None = None,
                 ml_dir_template: str = "models_v3_full_{sel}"):
    epoch_label = ("all epochs" if epochs_filter is None
                   else f"epoch{'s' if len(epochs_filter) > 1 else ''} "
                        f"{','.join(str(e) for e in epochs_filter)}")
    print(f"Loading {results_dir.name} / {fairness} / budget={budget} / seeds={seeds}"
          f" / {epoch_label} ...")
    data = collect(results_dir, fairness, budget, seeds, epochs_filter, ml_dir_template)
    if data is None:
        return 1
    n_random = len(data["random"])
    n_roulette = len(data["roulette"])
    n_ml = {sel: len(data[f"ml/{sel}"]) for sel in ML_SELECTORS}
    print(f"  random checkpoints: {n_random}")
    print(f"  roulette checkpoints: {n_roulette}")
    for sel, n in n_ml.items():
        print(f"  ml/{sel} checkpoints: {n}")

    out: list[str] = []
    out.append(f"# Within-epoch dominance / Δ% — {results_dir.name} ({epoch_label})\n\n")
    out.append(f"**Bench:** {results_dir.name} / fairness={fairness} / budget={budget}ms"
               f" / seeds={seeds} / {epoch_label}\n\n")
    out.append(
        "Mirrors [BENCH_v5_STATIC_2026-05-05.md §8/§9](BENCH_v5_STATIC_2026-05-05.md) but slicing "
        "each *within-epoch* iter trajectory recorded by `--log-epoch-iters`. Each "
        "(instance × seed × epoch × checkpoint) tuple is one observation, pooled across "
        "all epochs. Phase fraction = `iter / max_iter_in_epoch`, where `max_iter_in_epoch` "
        "is aligned across selectors (min over selectors of each epoch's max logged iter).\n\n"
    )
    out.append(
        "**Alignment caveat:** at fixed wall-clock budget, ML hits fewer iters than random "
        "in the same time, so 'iter K of ML' and 'iter K of random' sit at different "
        "SA-cooling temperatures. The static report's iter-budget format made this exact, "
        "here it's a small bias to keep in mind.\n\n"
    )

    # --- Section 1: Dominance % ---
    out.append("## 1. Dominance — % of (instance × seed × epoch × checkpoint) where the selector beat both random and roulette\n\n")
    out.append("| selector | " + " | ".join(label for (_, label) in PHASES) + " |\n")
    out.append("|---|" + "|".join(["---:"] * len(PHASES)) + "|\n")
    for sel in ML_SELECTORS:
        cells = fmt_dom(dominance_stats(data, f"ml/{sel}", ["random", "roulette"]))
        out.append(f"| `ml/{sel}` | " + " | ".join(cells) + " |\n")
    # roulette vs random
    cells = fmt_dom(dominance_stats(data, "roulette", ["random"]))
    out.append(f"| `roulette vs random` | " + " | ".join(cells) + " |\n\n")

    # --- Section 2: Mean Δ% vs random ---
    out.append("## 2. Mean Δ% vs random\n\n")
    out.append("Negative = selector beats random by that %. Δ% = 100 × (sel.best − random.best) / random.best at the same (epoch, iter).\n\n")
    out.append("| selector | " + " | ".join(label for (_, label) in PHASES) + " |\n")
    out.append("|---|" + "|".join(["---:"] * len(PHASES)) + "|\n")
    for sel in ML_SELECTORS:
        cells = fmt_delta(delta_stats(data, f"ml/{sel}", "random"))
        out.append(f"| `ml/{sel}` | " + " | ".join(cells) + " |\n")
    cells = fmt_delta(delta_stats(data, "roulette", "random"))
    out.append(f"| `roulette` | " + " | ".join(cells) + " |\n\n")

    # --- Section 3: Mean Δ% vs roulette ---
    out.append("## 3. Mean Δ% vs roulette\n\n")
    out.append("Same metric using roulette as the reference.\n\n")
    out.append("| selector | " + " | ".join(label for (_, label) in PHASES) + " |\n")
    out.append("|---|" + "|".join(["---:"] * len(PHASES)) + "|\n")
    for sel in ML_SELECTORS:
        cells = fmt_delta(delta_stats(data, f"ml/{sel}", "roulette"))
        out.append(f"| `ml/{sel}` | " + " | ".join(cells) + " |\n")
    cells = fmt_delta(delta_stats(data, "random", "roulette"))
    out.append(f"| `random` | " + " | ".join(cells) + " |\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out))
    print(f"\nWrote {out_path} ({sum(len(s) for s in out)} chars)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bench_results_v6_30s")
    ap.add_argument("--fairness", choices=["fair", "unfair"], default="fair")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="Filter to these epoch ids (e.g. --epochs 1). "
                         "Default: include all epochs.")
    ap.add_argument("--out-dir", default="thesis_results")
    ap.add_argument("--ml-dir-template", default="models_v3_full_{sel}",
                    help="Template for ML model dir names; use {sel} placeholder for "
                         "the short selector name. Default: models_v3_full_{sel}. "
                         "For tree benches: models_v6_full_{sel}_tree.")
    ap.add_argument("--tag", default="",
                    help="Optional tag appended to output filename (e.g. 'tree').")
    args = ap.parse_args()

    rd = (REPO_ROOT / args.results_dir).resolve()
    od = (REPO_ROOT / args.out_dir).resolve()

    epochs_filter = args.epochs if args.epochs else None
    epoch_tag = ("" if epochs_filter is None
                 else "_ep" + "-".join(str(e) for e in epochs_filter))

    tag_suffix = f"_{args.tag}" if args.tag else ""
    rc = 0
    for B in args.budgets:
        out = od / (f"dynamic_within_epoch_dominance_{rd.name}_"
                    f"{args.fairness}_bud{B}{epoch_tag}{tag_suffix}.md")
        rc |= write_report(rd, args.fairness, B, args.seeds, out, epochs_filter,
                            args.ml_dir_template)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
