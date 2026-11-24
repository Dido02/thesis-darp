#!/usr/bin/env python3
"""All three v6 per-iter signals broken down per budget.

v6_30s_2026-05-10 §13 claimed three per-iter metrics where ML beats random:
  (1) improve_rate  = improving iters / total iters       (v6: +1-3pp ✗ no longer)
  (2) sa_accept_rate = SA-accepted (worse) iters / total iters  (v6: +3-5pp)
  (3) best_rate     = new-best iters / total iters        (v6: +0.5-0.7pp ✓ holds)

Reports Δ vs random AND Δ vs roulette per (selector, budget), with paired
Wilcoxon p-values. All 7 ML configs evaluated:
  - deployed:   ml_20f_t07__T0.10
  - evening:    ml_20f_t05__T0.10, ml_20f_ei_log__T0.10
  - new_models: ml_forest_d8_K20, ml_forest_d8_K16, ml_tree_d4_K4, ml_tree_d8_K16
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
FIN = ROOT / "bench_results_final" / "raw_logs"
NEW = ROOT / "bench_results_new_models_2026-05-20" / "raw_logs"
OUT = ROOT / "bench_results_final" / "PER_ITER_SIGNALS_PER_BUDGET.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

DEPLOYED = "ml_20f_t07__T0.10"
ML_SELECTORS = [
    DEPLOYED, "ml_20f_t05__T0.10", "ml_20f_ei_log__T0.10",
    "ml_forest_d8_K20__T0.10", "ml_forest_d8_K16__T0.10",
    "ml_tree_d4_K4__T0.10", "ml_tree_d8_K16__T0.10",
]
BASELINES = ["random", "roulette"]
BUDGETS = [200, 1000, 5000, 10000]
DELAYS = [3, 5, 10]
METRICS = ["improve_rate", "sa_accept_rate", "best_rate"]
METRIC_LABEL = {
    "improve_rate":   "Δ improve_rate (improving iters / total)",
    "sa_accept_rate": "Δ sa_accept_rate (sa-accepted worse iters / total)",
    "best_rate":      "Δ best_rate (new-best iters / total)",
}
V6_CLAIM = {
    "improve_rate":   "+1-3pp Holm-significant",
    "sa_accept_rate": "+3-5pp Holm-significant",
    "best_rate":      "+0.5-0.7pp Holm-significant",
}


def parse_log(p: Path):
    m = _RE_NAME.match(p.name)
    if not m: return None
    it = im = bst = sa = 0
    for ln in p.read_text().splitlines():
        em = _RE_EPOCH.search(ln)
        if em:
            a, b, c, d = map(int, em.groups())
            it += a; im += b; bst += c; sa += d
    if it == 0: return None
    return {
        "instance_base": m.group(1), "delay": int(m.group(2)),
        "selector": m.group(3), "budget": int(m.group(4)), "seed": int(m.group(5)),
        "iters": it, "improve": im, "best": bst, "sa": sa,
        "improve_rate":   im / it,
        "sa_accept_rate": sa / it,
        "best_rate":      bst / it,
    }


def wilcoxon_paired(df, sel, baseline, metric, *, budget=None):
    sub = df[df.selector.isin([sel, baseline])]
    if budget is not None: sub = sub[sub.budget == budget]
    idx = ["instance_base", "delay", "seed"]
    if budget is None: idx.append("budget")
    piv = sub.pivot_table(index=idx, columns="selector", values=metric).dropna()
    if sel not in piv.columns or baseline not in piv.columns or len(piv) < 3:
        return None
    diffs = piv[sel].values - piv[baseline].values
    mean_d = float(diffs.mean())
    if (diffs == 0).all():
        return {"n": len(piv), "delta": mean_d, "p": 1.0}
    try:
        p = float(wilcoxon(piv[sel].values, piv[baseline].values,
                           alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        p = None
    return {"n": len(piv), "delta": mean_d, "p": p}


def fmt_cell(r):
    if r is None: return "n/a"
    sign = "✓" if (r["p"] is not None and r["p"] < 0.05 and r["delta"] > 0) else (
        "✗" if (r["p"] is not None and r["p"] < 0.05 and r["delta"] < 0) else "·")
    p_str = f"{r['p']:.3f}" if r["p"] is not None else "n/a"
    return f"{r['delta']*100:+.2f}pp {sign} p={p_str}"


def render_per_budget(df, metric, baseline, selectors):
    L = ["| selector | " + " | ".join(f"b={b}" for b in BUDGETS) + " |",
         "|---|" + "|".join("---:" for _ in BUDGETS) + "|"]
    for sel in selectors:
        cells = [fmt_cell(wilcoxon_paired(df, sel, baseline, metric, budget=b))
                 for b in BUDGETS]
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    return L


def main():
    rows = [r for p in sorted(FIN.glob("*.log")) if (r := parse_log(p))]
    rows += [r for p in sorted(NEW.glob("*.log")) if (r := parse_log(p))]
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} logs")
    print(df.groupby("selector").size().to_string())

    L = []
    L.append("# Per-iter signals per budget — three v6 metrics reconciled")
    L.append("")
    L.append("**Selectors compared:** 7 ML configs (deployed `ml_20f_t07__T0.10` + 2 evening siblings + 4 new) vs `random` and `roulette` baselines.")
    L.append("**Test:** paired Wilcoxon at each budget, over (instance, delay, seed) cells (n=60 per cell).")
    L.append("**Convention:** ✓ = positive Δ at p<0.05; ✗ = negative Δ at p<0.05; · = not significant. Cells in percentage points (pp).")
    L.append("")
    L.append("Each section is one of the three per-iter metrics v6 claimed ML beats baselines on. The v6 claim is at the top of each section for direct comparison.")
    L.append("")

    sect = 1
    for metric in METRICS:
        L.append(f"## {sect}. {METRIC_LABEL[metric]}")
        L.append("")
        L.append(f"**v6 claim:** {V6_CLAIM[metric]}")
        L.append("")
        L.append(f"### {sect}.a vs `random`")
        L.append("")
        L.extend(render_per_budget(df, metric, "random", ML_SELECTORS))
        L.append("")
        L.append(f"### {sect}.b vs `roulette`")
        L.append("")
        L.extend(render_per_budget(df, metric, "roulette", ML_SELECTORS))
        L.append("")
        sect += 1

    # §4: matrix for deployed only — all three metrics side by side, per budget × delay
    L.append(f"## 4. Deployed `{DEPLOYED}` only — all three metrics, per (delay, budget)")
    L.append("")
    L.append("n per cell = 20 = 4 instances × 5 seeds. Paired Wilcoxon vs `random`.")
    L.append("")
    for metric in METRICS:
        L.append(f"**{METRIC_LABEL[metric]}**")
        L.append("")
        L.append("| delay \\ budget | " + " | ".join(f"b={b}" for b in BUDGETS) + " |")
        L.append("|---|" + "|".join("---:" for _ in BUDGETS) + "|")
        for d in DELAYS:
            cells = []
            for b in BUDGETS:
                sub = df[(df.selector.isin([DEPLOYED, "random"])) & (df.delay == d) & (df.budget == b)]
                piv = sub.pivot_table(index=["instance_base", "seed"],
                                       columns="selector", values=metric).dropna()
                if DEPLOYED not in piv.columns or "random" not in piv.columns or len(piv) < 3:
                    cells.append("n/a"); continue
                diffs = piv[DEPLOYED].values - piv["random"].values
                mean_d = float(diffs.mean())
                try:
                    p = float(wilcoxon(piv[DEPLOYED].values, piv["random"].values,
                                       alternative="two-sided", zero_method="wilcox").pvalue)
                except Exception:
                    p = None
                cells.append(fmt_cell({"n": len(piv), "delta": mean_d, "p": p}))
            L.append(f"| d{d:02d} | " + " | ".join(cells) + " |")
        L.append("")

    # §5: summary verdict
    L.append("## 5. Verdict on the v6 three-metric claim")
    L.append("")
    L.append("Looking at §1.a — §3.a deployed row, per budget:")
    L.append("")
    for metric in METRICS:
        L.append(f"### {METRIC_LABEL[metric]}")
        for b in BUDGETS:
            r = wilcoxon_paired(df, DEPLOYED, "random", metric, budget=b)
            L.append(f"- b={b}: {fmt_cell(r)}")
        L.append("")
    L.append("**Interpretation rule:** if v6's claim is reproducible in this code/data, we should see ✓ marks. ✗ marks mean the claim flipped in the broader BENCH_FINAL grid.")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
