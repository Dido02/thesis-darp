#!/usr/bin/env python3
"""Per-cell preview plots for the single-pair ablation bench.

Mirrors benchmarks/preview_final_bench.py but with:
  - 4 single-pair configs from this bench
  - + 9-pair `random` and `roulette` baselines from bench_results_final/

Generates 12 PNGs (4 instances × 3 delays), one per (delay, instance) cell.
Each chart: x = epoch budget (log scale), y = final objective (lower = better),
6 lines with errorbars over 5 seeds.

Usage:
    python3 03_plot_per_cell.py                 # all 12 cells
    python3 03_plot_per_cell.py --delay 10      # just delay 10 cells (4 plots)
    python3 03_plot_per_cell.py --instance 01_min   # just 01_min cells (3 plots)
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO       = Path(__file__).resolve().parents[2]
SP_LOGS    = REPO / "bench_results_single_pair_2026-05-21" / "raw_logs"
FINAL_LOGS = REPO / "bench_results_final" / "raw_logs"
OUT_DIR    = REPO / "bench_results_single_pair_2026-05-21"

OBJ_RE = re.compile(r"Total travel:\s+(-?\d+(?:\.\d+)?)")

# 6 selectors total: 4 single-pair from this bench + 2 9-pair from final
STYLE = {
    # 9-pair baselines (from bench_results_final/)
    "random":              {"c": "#000000", "mk": "s", "dodge": 0.84, "label": "9-pair random",   "ls": "-",  "lw": 2.0},
    "roulette":            {"c": "#1f77b4", "mk": "o", "dodge": 0.91, "label": "9-pair roulette", "ls": "-",  "lw": 2.0},
    # 4 single-pair (from this bench)
    "op_random_greedy":    {"c": "#ff7f0e", "mk": "v", "dodge": 0.98, "label": "1-pair random+greedy",  "ls": "--", "lw": 1.4},
    "op_worst_greedy":     {"c": "#2ca02c", "mk": "^", "dodge": 1.04, "label": "1-pair worst+greedy",   "ls": "--", "lw": 1.4},
    "op_shaw_greedy":      {"c": "#9467bd", "mk": "D", "dodge": 1.11, "label": "1-pair shaw+greedy",    "ls": "--", "lw": 1.4},
    "op_worst_regret2":    {"c": "#d62728", "mk": "P", "dodge": 1.18, "label": "1-pair worst+regret2",  "ls": "--", "lw": 1.4},
}

ALL_DELAYS    = ["03", "05", "10"]
ALL_INSTANCES = ["30_s", "01_min", "02_min", "05_min"]


def collect_for_cell(delay: str, instance: str) -> dict[tuple[str, int], list[float]]:
    """Returns {(selector, budget) -> [obj values across seeds]} for this cell."""
    instance_full = f"manhattan_{instance}_delay{delay}_min"
    name_re = re.compile(rf"^dynamic__{re.escape(instance_full)}__(.+)__b(\d+)__s(\d+)\.log$")
    data: dict[tuple[str, int], list[float]] = {}
    for log_root in (SP_LOGS, FINAL_LOGS):
        for p in sorted(log_root.glob(f"dynamic__{instance_full}__*.log")):
            m = name_re.match(p.name)
            if not m: continue
            sel, budget = m.group(1), int(m.group(2))
            if sel not in STYLE: continue   # ignore other ML selectors we don't want
            mo = OBJ_RE.search(p.read_text())
            if not mo: continue
            data.setdefault((sel, budget), []).append(float(mo.group(1)))
    return data


def plot_cell(delay: str, instance: str) -> Path | None:
    data = collect_for_cell(delay, instance)
    if not data:
        print(f"  [skip] d{delay}/{instance}: no data")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lo, all_hi = float("inf"), 0.0
    legend_lines = []
    for sel, style in STYLE.items():
        pts = sorted([(b, data[(s, b)]) for (s, b) in data if s == sel])
        if not pts: continue
        x_raw = [b for b, _ in pts]
        x = [b * style["dodge"] for b in x_raw]
        mu = [mean(o) for _, o in pts]
        sd = [stdev(o) if len(o) > 1 else 0 for _, o in pts]
        n_per = [len(o) for _, o in pts]
        ax.errorbar(x, mu, yerr=sd,
                    marker=style["mk"], markersize=8, capsize=4,
                    color=style["c"], linewidth=style["lw"],
                    linestyle=style["ls"], elinewidth=1.2,
                    label=f"{style['label']}  (n={'/'.join(str(n) for n in n_per)})")
        all_lo = min(all_lo, *[m - s for m, s in zip(mu, sd)])
        all_hi = max(all_hi, *[m + s for m, s in zip(mu, sd)])

    if all_lo == float("inf"):
        plt.close(fig)
        return None
    pad = (all_hi - all_lo) * 0.10
    ax.set_ylim(all_lo - pad, all_hi + pad)
    ax.set_xscale("log")
    ticks = sorted({b for _, b in data})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t} ms" for t in ticks])
    ax.set_xlabel("epoch budget (log scale)")
    ax.set_ylabel("final objective: total travel  (lower = better)")
    ax.set_title(f"Single-pair vs 9-pair baselines — delay {delay} / manhattan_{instance}  "
                 f"(5 seeds, fair budget)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"preview_single_pair_d{delay}_{instance}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", choices=ALL_DELAYS, default=None,
                    help="If set, only render this delay (otherwise all 3).")
    ap.add_argument("--instance", choices=ALL_INSTANCES, default=None,
                    help="If set, only render this instance (otherwise all 4).")
    args = ap.parse_args()

    delays    = [args.delay]    if args.delay    else ALL_DELAYS
    instances = [args.instance] if args.instance else ALL_INSTANCES

    print(f"Rendering {len(delays) * len(instances)} cells...")
    written = []
    for d in delays:
        for inst in instances:
            out = plot_cell(d, inst)
            if out:
                print(f"  wrote {out.name}")
                written.append(out)
    print(f"\nDone — {len(written)} PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
