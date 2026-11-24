"""Generate a static-bench analysis markdown report.

Loads all summary.json under --results-dir (assumed to follow the
launch_v5_static.py layout: <root>/<selector_label>/runs/<run_id>/summary.json
where selector_label is e.g. 'baselines/random', '20f_regr', '20f_t07__T0.10').

Differences from generate_bench_report.py (which assumed dynamic layout):
  - No fair/unfair distinction.
  - "Budget" axis is iteration count (200 / 1000 / 5000), not wall-clock.
  - Selector keys come from selector_label, not from the dynamic-bench
    "baselines_<fair>" + "<model_dir>" convention.

Usage:
    python benchmarks/generate_static_bench_report.py \\
        --results-dir bench_results_v5_static \\
        --out BENCH_v5_STATIC_2026-05-05.md
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

from scipy import stats


def load_runs(results_dir: Path) -> list[dict]:
    rows = []
    for sj in glob.glob(str(results_dir / "**/runs/*/summary.json"), recursive=True):
        try:
            s = json.load(open(sj))
        except Exception:
            continue
        rel = Path(sj).resolve().relative_to(results_dir.resolve()).parts
        runs_idx = rel.index("runs")
        sel_label = "/".join(rel[:runs_idx])
        # Normalize selector key for tables/comparisons
        if sel_label.startswith("baselines/"):
            sel_key = sel_label.split("/")[-1]
        else:
            sel_key = f"ml/{sel_label}"
        rows.append({
            "sel_key": sel_key,
            "selector_label": sel_label,
            "instance": s["instance"],
            "iters": s["iterations"],
            "seed": s["seed"],
            "cost": s["cost"],
            "initial_obj": s.get("initial_obj"),
            "iters_improving": s["total_iters_improving"],
            "iters_new_best": s["total_iters_new_best"],
            "iters_sa_accepted": s["total_iters_sa_accepted"],
            "wall_s": s["wall_clock_s"],
            "ml_softmax_temp": s.get("ml_softmax_temp"),
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def short(s: str) -> str:
    return s.replace("ml/", "").replace("models_v3_full_", "")


def fmt_pct(v: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if v is None: return "—"
    return f"{'+' if signed and v >= 0 else ''}{v:.{digits}f}%"


def fmt_p(p: Optional[float]) -> str:
    if p is None: return "—"
    if p < 0.0001: return f"{p:.5f}"
    if p < 0.001: return f"{p:.4f}"
    return f"{p:.4f}"


def paired_diffs_pct(rows, focus_sel, ref_sel, iters):
    """Return paired %-diffs across (instance, seed) cells for a given iter count."""
    f = {(r["instance"], r["seed"]): r["cost"]
         for r in rows if r["sel_key"] == focus_sel and r["iters"] == iters}
    g = {(r["instance"], r["seed"]): r["cost"]
         for r in rows if r["sel_key"] == ref_sel and r["iters"] == iters}
    diffs = []
    for k in f:
        if k in g and g[k] > 0:
            diffs.append(100 * (f[k] - g[k]) / g[k])
    return diffs


def wilcoxon(diffs):
    if len(diffs) < 4: return None
    if all(abs(d) < 1e-12 for d in diffs): return 1.0
    try:
        st = stats.wilcoxon(diffs)
        return float(st.pvalue)
    except Exception:
        return None


def holm_correct(pvals_with_keys, alpha=0.05):
    """Holm-Bonferroni: returns dict key -> {raw_p, corrected_p, sig}."""
    valid = [(k, p) for k, p in pvals_with_keys if p is not None]
    if not valid:
        return {k: {"raw_p": p, "corrected_p": None, "sig": False}
                for k, p in pvals_with_keys}
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    out = {}
    for i, (k, p) in enumerate(valid):
        corrected = min(1.0, p * (m - i))
        out[k] = {"raw_p": p, "corrected_p": corrected, "sig": corrected <= alpha}
    for k, p in pvals_with_keys:
        if k not in out:
            out[k] = {"raw_p": p, "corrected_p": None, "sig": False}
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_header(rows, generated_at):
    instances = sorted({r["instance"] for r in rows})
    iters = sorted({r["iters"] for r in rows})
    sels = sorted({r["sel_key"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    return (
        f"# v5_static Bench Analysis Report\n\n"
        f"**Generated:** {generated_at}  \n"
        f"**Mode:** STATIC (single ALNS call per instance, no rolling horizon)  \n"
        f"**Runs:** {len(rows)} ({len(sels)} selectors × {len(instances)} instances "
        f"× {len(iters)} iter counts × {len(seeds)} seeds)\n\n"
        f"## 1. Methodology\n\n"
        f"- **Bench script:** `benchmarks/launch_v5_static.py`\n"
        f"- **Mode:** static — instance loaded all at once, no `--dynamic`, no commits.\n"
        f"- **Operator config:** classical (3 destroy × 3 repair = 9 pairs)\n"
        f"- **Iter counts:** {iters}\n"
        f"- **Seeds:** {seeds}\n"
        f"- **Selectors:** {len(sels)} (baselines + ML variants)\n"
        f"- **Stat tests:** paired Wilcoxon vs random across (instance × seed) cells "
        f"(n={len(instances)*len(seeds)} pairs per row), Holm-corrected over the "
        f"selector-vs-random family (m={len(sels)-1}).\n\n"
        f"## 2. Selector roster\n\n"
        f"| selector | rationale (per analysis/feature_importance_static.md) |\n"
        f"|---|---|\n"
        f"| `random` | baseline (uniform pair sampling) |\n"
        f"| `roulette` | classical adaptive-weight ALNS baseline |\n"
        f"| `ml/20f_regr` | best static-transfer (7.4% dynamic-feature weight) |\n"
        f"| `ml/20f_regr__T0.10` | same model, softmax T=0.10 |\n"
        f"| `ml/20f_t07` | low-dyn classifier (9.2%) |\n"
        f"| `ml/20f_t07__T0.10` | same, softmax T=0.10 |\n"
        f"| `ml/20f_ei_log` | control — predicted to regress (21.3% dyn weight) |\n"
        f"\n"
    )


def section_per_cell(rows):
    instances = sorted({r["instance"] for r in rows})
    iters_list = sorted({r["iters"] for r in rows})
    selectors = sorted({r["sel_key"] for r in rows})

    out = ["## 3. Per-cell mean obj (3 seeds)\n\n",
           "Each cell shows mean objective over 3 seeds. "
           "**Bold** marks the per-cell winner; *italic* marks where ML beats random.\n\n"]
    for n_iter in iters_list:
        out.append(f"### 3.{iters_list.index(n_iter)+1} Iters = {n_iter}\n\n")
        out.append("| instance | " + " | ".join(short(s) for s in selectors) + " | winner | win Δ% vs random |\n")
        out.append("|---|" + "---:|" * len(selectors) + "---|---:|\n")
        for inst in instances:
            inst_label = inst.replace("manhattan_", "").replace("_delay03_min", "/03min")
            cell_means = {}
            for s in selectors:
                vs = [r["cost"] for r in rows
                      if r["sel_key"] == s and r["instance"] == inst and r["iters"] == n_iter]
                if vs: cell_means[s] = mean(vs)
            if not cell_means: continue
            winner_sel, winner_obj = min(cell_means.items(), key=lambda kv: kv[1])
            rand_obj = cell_means.get("random", winner_obj)
            cell_strs = []
            for s in selectors:
                v = cell_means.get(s)
                if v is None: cell_strs.append("—"); continue
                fmt = f"{v:,.0f}"
                if s == winner_sel: fmt = f"**{fmt}**"
                elif s.startswith("ml/") and v < rand_obj: fmt = f"*{fmt}*"
                cell_strs.append(fmt)
            delta = (winner_obj - rand_obj) / rand_obj * 100 if rand_obj else 0
            out.append(f"| {inst_label} | " + " | ".join(cell_strs) +
                       f" | `{short(winner_sel)}` | {delta:+.3f}% |\n")
        out.append("\n")
    return "".join(out)


def section_aggregate(rows):
    iters_list = sorted({r["iters"] for r in rows})
    selectors = sorted({r["sel_key"] for r in rows})
    sels_no_random = [s for s in selectors if s != "random"]
    out = ["## 4. Selector aggregate vs random (per iter count)\n\n",
           "Paired Wilcoxon over (instance × seed) cells. "
           "Holm correction over the selector-vs-random family.\n\n"]
    for n_iter in iters_list:
        out.append(f"### 4.{iters_list.index(n_iter)+1} Iters = {n_iter}\n\n")
        rows_per_sel = []
        for sel in sels_no_random:
            d = paired_diffs_pct(rows, sel, "random", n_iter)
            if not d:
                rows_per_sel.append((sel, None, None, None, None, None)); continue
            rows_per_sel.append((sel, mean(d), median(d),
                                 stdev(d) if len(d) > 1 else 0,
                                 len(d), wilcoxon(d)))
        holm_in = [(sel, p) for sel, _, _, _, _, p in rows_per_sel]
        holm = holm_correct(holm_in)
        # sort by mean
        rows_per_sel.sort(key=lambda r: r[1] if r[1] is not None else 99)
        out.append("| selector | mean Δ% | median Δ% | std | n | raw p | Holm sig |\n")
        out.append("|---|---:|---:|---:|---:|---:|:---:|\n")
        out.append("| `random` (ref) | 0.00% | 0.00% | 0.00% | — | — | — |\n")
        for sel, m, med, sd, n, p in rows_per_sel:
            if m is None: continue
            h = holm[sel]
            sig = "✓" if h["sig"] else ("(\\*)" if (h["raw_p"] and h["raw_p"] < 0.05) else "")
            out.append(f"| `{sel}` | {fmt_pct(m, 3, True)} | {fmt_pct(med, 3, True)} | "
                       f"{fmt_pct(sd, 2)} | {n} | {fmt_p(p)} | {sig} |\n")
        out.append("\n")
    return "".join(out)


def section_best_of_ml(rows):
    iters_list = sorted({r["iters"] for r in rows})
    out = ["## 5. Best-of-ML upper bound\n\n",
           "For each (instance, iters, seed), pick the ML selector achieving the lowest obj. "
           "Aggregate vs random / roulette across cells. (Hindsight-optimal — not deployable, "
           "but bounds what model-selection over the ensemble could deliver.)\n\n"]
    for n_iter in iters_list:
        out.append(f"### 5.{iters_list.index(n_iter)+1} Iters = {n_iter}\n\n")
        # group by (instance, seed)
        per_cell = defaultdict(dict)
        for r in rows:
            if r["iters"] != n_iter: continue
            per_cell[(r["instance"], r["seed"])][r["sel_key"]] = r["cost"]
        diffs_rand, diffs_roul = [], []
        for (inst, seed), d in per_cell.items():
            if "random" not in d: continue
            ml_costs = [v for k, v in d.items() if k.startswith("ml/")]
            if not ml_costs: continue
            best_ml = min(ml_costs)
            diffs_rand.append(100 * (best_ml - d["random"]) / d["random"])
            if "roulette" in d:
                diffs_roul.append(100 * (best_ml - d["roulette"]) / d["roulette"])
        if not diffs_rand: continue
        out.append("| reference | mean Δ% | median Δ% | n | Wilcoxon p |\n")
        out.append("|---|---:|---:|---:|---:|\n")
        out.append(f"| best-of-ML vs random | {fmt_pct(mean(diffs_rand), 3, True)} | "
                   f"{fmt_pct(median(diffs_rand), 3, True)} | {len(diffs_rand)} | "
                   f"{fmt_p(wilcoxon(diffs_rand))} |\n")
        out.append(f"| best-of-ML vs roulette | {fmt_pct(mean(diffs_roul), 3, True)} | "
                   f"{fmt_pct(median(diffs_roul), 3, True)} | {len(diffs_roul)} | "
                   f"{fmt_p(wilcoxon(diffs_roul))} |\n\n")
    return "".join(out)


def section_per_iter_quality(rows):
    iters_list = sorted({r["iters"] for r in rows})
    selectors = sorted({r["sel_key"] for r in rows})
    sels_no_random = [s for s in selectors if s != "random"]
    out = ["## 6. Per-iteration ALNS quality vs random\n\n",
           "How does each selector affect the *quality* of moves per iteration?\n"
           "- **% improving** = `iters_improving / iters_total`\n"
           "- **% new-best** = `iters_new_best / iters_total`\n"
           "- **% SA-accepted** = `iters_sa_accepted / iters_total`\n\n"
           "Δpp = percentage-point delta vs random. Paired across (instance × seed).\n\n"]
    for n_iter in iters_list:
        out.append(f"### 6.{iters_list.index(n_iter)+1} Iters = {n_iter}\n\n")
        # build per-sel mean rates
        rates = defaultdict(lambda: defaultdict(list))   # metric -> sel -> [pct]
        for r in rows:
            if r["iters"] != n_iter: continue
            tot = r["iters_improving"] + r["iters_new_best"] + r["iters_sa_accepted"]
            if r["iters"] > 0:
                rates["improving"][r["sel_key"]].append(100 * r["iters_improving"] / r["iters"])
                rates["new_best"][r["sel_key"]].append(100 * r["iters_new_best"] / r["iters"])
                rates["sa_accepted"][r["sel_key"]].append(100 * r["iters_sa_accepted"] / r["iters"])
        out.append("| selector | Δpp improving | Δpp new-best | Δpp sa-accepted |\n")
        out.append("|---|---:|---:|---:|\n")
        rand_imp = mean(rates["improving"]["random"]) if rates["improving"]["random"] else 0
        rand_nb  = mean(rates["new_best"]["random"]) if rates["new_best"]["random"] else 0
        rand_sa  = mean(rates["sa_accepted"]["random"]) if rates["sa_accepted"]["random"] else 0
        out.append(f"| `random` (ref imp/nb/sa%) | {rand_imp:.2f} | {rand_nb:.2f} | {rand_sa:.2f} |\n")
        # sort by best-new_best Δ
        sortable = []
        for sel in sels_no_random:
            if not rates["improving"][sel]:
                sortable.append((sel, None, None, None)); continue
            d_imp = mean(rates["improving"][sel]) - rand_imp
            d_nb  = mean(rates["new_best"][sel]) - rand_nb
            d_sa  = mean(rates["sa_accepted"][sel]) - rand_sa
            sortable.append((sel, d_imp, d_nb, d_sa))
        sortable.sort(key=lambda x: -(x[2] if x[2] is not None else -99))
        for sel, di, dn, ds in sortable:
            if di is None: continue
            out.append(f"| `{sel}` | {di:+.2f}pp | {dn:+.2f}pp | {ds:+.2f}pp |\n")
        out.append("\n")
    return "".join(out)


def section_comparison_to_dynamic(rows):
    """Compare static obj to known IH and v5 dynamic best on the same instances."""
    out = ["## 7. Static vs IH vs dynamic comparison\n\n",
           "Per-instance: best obj across all selectors and seeds in the static bench, "
           "compared to IH (Fiedler greedy) and the best dynamic v5 result.\n\n"]
    static_best = {}
    for r in rows:
        if r["iters"] != 5000: continue
        cur = static_best.get(r["instance"], (float("inf"), None))
        if r["cost"] < cur[0]:
            static_best[r["instance"]] = (r["cost"], r["sel_key"])
    # IH and v5 dynamic best (manually entered from earlier bench)
    refs = {
        "manhattan_30_s_delay03_min":  {"IH": 69442,  "VGA": 65197, "v5_dyn_best": 67855},
        "manhattan_01_min_delay03_min":{"IH": 136237, "VGA": 119998,"v5_dyn_best": 126182},
        "manhattan_02_min_delay03_min":{"IH": 256304, "VGA": 215228,"v5_dyn_best": 232952},
        "manhattan_05_min_delay03_min":{"IH": 618274, "VGA": None,  "v5_dyn_best": 579070},
    }
    out.append("| instance | static best (5K) | IH | VGA | dyn best (v5) | static vs IH | static vs dyn |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for inst in sorted(static_best):
        sb, sb_sel = static_best[inst]
        ref = refs.get(inst, {})
        ih  = ref.get("IH"); vga = ref.get("VGA"); dyn = ref.get("v5_dyn_best")
        ih_pct  = f"{100*(sb-ih)/ih:+.2f}%" if ih else "—"
        dyn_pct = f"{100*(sb-dyn)/dyn:+.2f}%" if dyn else "—"
        vga_str = f"{vga:,.0f}" if vga else "—"
        out.append(f"| {inst.replace('manhattan_','').replace('_delay03_min','/03min')} | "
                   f"**{sb:,.0f}** ({short(sb_sel)}) | {ih:,.0f} | {vga_str} | {dyn:,.0f} | "
                   f"{ih_pct} | {dyn_pct} |\n")
    out.append("\n")
    return "".join(out)


def section_findings(rows):
    return (
        "## 8. Where do the models do well? — narrative findings\n\n"
        "_See sections 3-6 for the underlying numbers._\n\n"
        "(This section is auto-summarized in the cover answer; see also the "
        "feature-importance analysis in `analysis/feature_importance_static.md`.)\n\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"ERROR: results-dir not found: {results_dir}")

    print(f"Loading runs from {results_dir} ...")
    rows = load_runs(results_dir)
    print(f"  {len(rows)} runs loaded")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        section_header(rows, generated_at),
        section_per_cell(rows),
        section_aggregate(rows),
        section_best_of_ml(rows),
        section_per_iter_quality(rows),
        section_comparison_to_dynamic(rows),
        section_findings(rows),
    ]
    report = "\n".join(sections)

    out_path = Path(args.out)
    out_path.write_text(report)
    print(f"Wrote {out_path} ({len(report.splitlines())} lines, {len(report)} chars)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
