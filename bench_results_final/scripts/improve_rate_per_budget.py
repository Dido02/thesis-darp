#!/usr/bin/env python3
"""Improvement-rate analysis broken down BY BUDGET.

Cross-references BENCH_v6_30s_2026-05-10's claim that ML beats random by
+1-3pp on improve rate. v6 tested budgets {1000, 5000} on delay 3 only.
This script computes per-budget Δ over the broader BENCH_FINAL grid (4
instances × 3 delays × 4 budgets × 5 seeds) so we can see at which
budgets the v6 claim still holds.

Reads bench_results_final/raw_logs (evening) AND
bench_results_new_models_2026-05-20/raw_logs (4 new configs on same
evening instances). Joins baselines on (instance, delay, budget, seed).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
FIN = ROOT / "bench_results_final" / "raw_logs"
NEW = ROOT / "bench_results_new_models_2026-05-20" / "raw_logs"
OUT = ROOT / "bench_results_final" / "IMPROVE_RATE_PER_BUDGET.md"

_RE_NAME = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")

DEPLOYED = "ml_20f_t07__T0.10"
EVENING_ML = ["ml_20f_t07__T0.10", "ml_20f_t05__T0.10", "ml_20f_ei_log__T0.10"]
NEW_ML     = ["ml_forest_d8_K20__T0.10", "ml_forest_d8_K16__T0.10",
              "ml_tree_d4_K4__T0.10", "ml_tree_d8_K16__T0.10"]
BASELINES = ["random", "roulette"]
ALL_ML    = EVENING_ML + NEW_ML

DELAYS  = [3, 5, 10]
BUDGETS = [200, 1000, 5000, 10000]
INSTANCES = ["30_s", "01_min", "02_min", "05_min"]


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
        "improve_rate": im / it, "best_rate": bst / it,
    }


def pair_wilcoxon(df, sel, baseline, *, delay=None, budget=None):
    """Paired Wilcoxon over (instance, [delay,] [budget,] seed) cells."""
    sub = df[df.selector.isin([sel, baseline])]
    if delay is not None: sub = sub[sub.delay == delay]
    if budget is not None: sub = sub[sub.budget == budget]
    idx = ["instance_base", "seed"]
    if delay is None: idx.append("delay")
    if budget is None: idx.append("budget")
    piv = sub.pivot_table(index=idx, columns="selector", values="improve_rate").dropna()
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


def render_per_budget_table(df, selectors, baseline, *, restrict_v6=False):
    """Rows = selectors, columns = budgets (delays aggregated)."""
    L = ["| selector | " + " | ".join(f"b={b}" for b in BUDGETS) + " |",
         "|---|" + "|".join("---:" for _ in BUDGETS) + "|"]
    for sel in selectors:
        cells = []
        for b in BUDGETS:
            sub = df[(df.selector.isin([sel, baseline])) & (df.budget == b)]
            if restrict_v6:
                sub = sub[(sub.delay == 3) & (sub.seed.isin([11, 22, 33]))]
            r = pair_wilcoxon(sub, sel, baseline, budget=b)
            if r is None:
                cells.append("n/a")
            else:
                sign = "✓" if r["delta"] > 0 and r["p"] is not None and r["p"] < 0.05 else (
                    "✗" if r["delta"] < 0 and r["p"] is not None and r["p"] < 0.05 else "·"
                )
                p_str = f"{r['p']:.3f}" if r['p'] is not None else "n/a"
                cells.append(f"{r['delta']*100:+.2f}pp {sign} p={p_str}")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    return L


def main():
    rows_fin = [r for p in sorted(FIN.glob("*.log")) if (r := parse_log(p))]
    rows_new = [r for p in sorted(NEW.glob("*.log")) if (r := parse_log(p))]
    df = pd.DataFrame(rows_fin + rows_new)
    print(f"Parsed {len(rows_fin)} evening + {len(rows_new)} new_models logs")
    print(df.groupby("selector").size().to_string())

    L = []
    L.append("# Improvement Rate per Budget — reconciling with BENCH_v6 claim")
    L.append("")
    L.append("**Question:** BENCH_v6_30s_2026-05-10 claimed ML wins improve-rate by +1-3pp Holm-significant. Recent BENCH_FINAL aggregate shows -1pp. Does the v6 claim still hold *at the budgets v6 actually tested* (1000ms, 5000ms, delay 3 only)?")
    L.append("")
    L.append("**Metric:** Δ improve_rate (selector − baseline), paired Wilcoxon per cell.")
    L.append("**Cells use percentage points (pp).** ✓ = positive Δ at p<0.05; ✗ = negative Δ at p<0.05; · = not significant.")
    L.append("")
    L.append("**Bench scope (this analysis):**")
    L.append("- Evening BENCH_FINAL ML configs: deployed `ml_20f_t07__T0.10`, plus `ml_20f_t05__T0.10`, `ml_20f_ei_log__T0.10`")
    L.append("- New-models ML configs (same evening instances): 4 forest/tree variants")
    L.append("- Baselines: `random`, `roulette` (from BENCH_FINAL, n=240 each)")
    L.append("- Grid: 4 instances × 3 delays × 4 budgets × 5 seeds = 240 cells per selector")
    L.append("")

    # §1
    L.append("## 1. Δ vs `random` per budget (all delays aggregated)")
    L.append("")
    L.append("n per cell = 60 = 4 instances × 3 delays × 5 seeds. Wilcoxon paired by (instance, delay, seed).")
    L.append("")
    L.extend(render_per_budget_table(df, ALL_ML, "random"))
    L.append("")

    # §2
    L.append("## 2. Δ vs `roulette` per budget (all delays aggregated)")
    L.append("")
    L.extend(render_per_budget_table(df, ALL_ML, "roulette"))
    L.append("")

    # §3: deployed model, full per-(budget, delay) heatmap
    L.append(f"## 3. Per-(budget, delay) Δ vs `random` for deployed `{DEPLOYED}`")
    L.append("")
    L.append("n per cell = 20 = 4 instances × 5 seeds.")
    L.append("")
    L.append("| delay \\ budget | " + " | ".join(f"b={b}" for b in BUDGETS) + " |")
    L.append("|---|" + "|".join("---:" for _ in BUDGETS) + "|")
    for d in DELAYS:
        cells = []
        for b in BUDGETS:
            sub = df[(df.selector.isin([DEPLOYED, "random"])) & (df.delay == d) & (df.budget == b)]
            r = pair_wilcoxon(sub, DEPLOYED, "random", delay=d, budget=b)
            if r is None:
                cells.append("n/a")
            else:
                sign = "✓" if r["delta"] > 0 and r["p"] is not None and r["p"] < 0.05 else (
                    "✗" if r["delta"] < 0 and r["p"] is not None and r["p"] < 0.05 else "·")
                p_str = f"{r['p']:.3f}" if r['p'] is not None else "n/a"
                cells.append(f"{r['delta']*100:+.2f}pp {sign} p={p_str}")
        L.append(f"| d{d:02d} | " + " | ".join(cells) + " |")
    L.append("")

    # §4: same for tree_d4_K4 (the closest-to-positive new config)
    SEL_TREE = "ml_tree_d4_K4__T0.10"
    L.append(f"## 4. Per-(budget, delay) Δ vs `random` for new `{SEL_TREE}`")
    L.append("")
    L.append("| delay \\ budget | " + " | ".join(f"b={b}" for b in BUDGETS) + " |")
    L.append("|---|" + "|".join("---:" for _ in BUDGETS) + "|")
    for d in DELAYS:
        cells = []
        for b in BUDGETS:
            sub = df[(df.selector.isin([SEL_TREE, "random"])) & (df.delay == d) & (df.budget == b)]
            r = pair_wilcoxon(sub, SEL_TREE, "random", delay=d, budget=b)
            if r is None:
                cells.append("n/a")
            else:
                sign = "✓" if r["delta"] > 0 and r["p"] is not None and r["p"] < 0.05 else (
                    "✗" if r["delta"] < 0 and r["p"] is not None and r["p"] < 0.05 else "·")
                p_str = f"{r['p']:.3f}" if r['p'] is not None else "n/a"
                cells.append(f"{r['delta']*100:+.2f}pp {sign} p={p_str}")
        L.append(f"| d{d:02d} | " + " | ".join(cells) + " |")
    L.append("")

    # §5: v6 sub-grid reconciliation
    L.append("## 5. v6 sub-grid reconciliation — delay 3 only, budgets 1000/5000, seeds 11/22/33")
    L.append("")
    L.append("This restricts to the EXACT cells v6 tested (4 instances × delay 3 × b∈{1000,5000} × 3 seeds = 24 cells per Wilcoxon).")
    L.append("If the v6 claim is reproducible in this code at this scope, we should see + signs here.")
    L.append("")
    L.append("| selector | b=1000 | b=5000 |")
    L.append("|---|---:|---:|")
    for sel in ALL_ML:
        cells = []
        for b in [1000, 5000]:
            sub = df[(df.selector.isin([sel, "random"])) & (df.delay == 3) &
                     (df.budget == b) & (df.seed.isin([11, 22, 33]))]
            r = pair_wilcoxon(sub, sel, "random", delay=3, budget=b)
            if r is None:
                cells.append("n/a")
            else:
                sign = "✓" if r["delta"] > 0 and r["p"] is not None and r["p"] < 0.05 else (
                    "✗" if r["delta"] < 0 and r["p"] is not None and r["p"] < 0.05 else "·")
                p_str = f"{r['p']:.3f}" if r['p'] is not None else "n/a"
                cells.append(f"{r['delta']*100:+.2f}pp {sign} p={p_str} (n={r['n']})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §6: best_rate per budget — the SECONDARY signal we found earlier
    L.append(f"## 6. Best-found rate Δ vs `random` per budget (deployed `{DEPLOYED}` only)")
    L.append("")
    L.append("This is the metric where the deployed model DOES beat random (§5 of `bench_results_final/IMPROVE_RATE_VS_RANDOM_ROULETTE.md`). Per-budget breakdown to see if the lift is concentrated at specific budgets.")
    L.append("")
    L.append("| delay \\ budget | " + " | ".join(f"b={b}" for b in BUDGETS) + " |")
    L.append("|---|" + "|".join("---:" for _ in BUDGETS) + "|")
    for d in DELAYS:
        cells = []
        for b in BUDGETS:
            sub = df[(df.selector.isin([DEPLOYED, "random"])) & (df.delay == d) & (df.budget == b)]
            piv = sub.pivot_table(index=["instance_base", "seed"],
                                   columns="selector", values="best_rate").dropna()
            if DEPLOYED not in piv.columns or "random" not in piv.columns or len(piv) < 3:
                cells.append("n/a"); continue
            diffs = piv[DEPLOYED].values - piv["random"].values
            mean_d = float(diffs.mean())
            try:
                p = float(wilcoxon(piv[DEPLOYED].values, piv["random"].values,
                                   alternative="two-sided", zero_method="wilcox").pvalue)
            except Exception:
                p = None
            sign = "✓" if mean_d > 0 and p is not None and p < 0.05 else (
                "✗" if mean_d < 0 and p is not None and p < 0.05 else "·")
            p_str = f"{p:.3f}" if p is not None else "n/a"
            cells.append(f"{mean_d*100:+.3f}pp {sign} p={p_str}")
        L.append(f"| d{d:02d} | " + " | ".join(cells) + " |")
    L.append("")

    # §7: take-home
    L.append("## 7. Take-home")
    L.append("")
    L.append("Read §1 across the row of `ml_20f_t07__T0.10`: each cell shows the Δ at a different budget. Where the cell is + sign and ✓, the v6 claim still holds at that budget. Where it's − sign, the v6 claim does NOT extend to that budget under the broader grid.")
    L.append("")
    L.append("§3 shows the same data broken down by delay too — useful if the budget effect interacts with delay.")
    L.append("")
    L.append("§5 is the strict v6 sub-grid (delay 3, b∈{1000,5000}, 3 seeds). If we don't see + signs here, the v6 claim is not reproducible with the current logs even on its original grid.")
    L.append("")
    L.append("§6 shows best-found rate, where the model DOES win. If those + signs are concentrated at the budgets where improve-rate loses, the trade-off is direct: the model trades simple-improving moves for new-best moves.")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
