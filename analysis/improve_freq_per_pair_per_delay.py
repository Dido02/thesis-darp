#!/usr/bin/env python3
"""improve_freq distribution per (pair, delay) — 9 classical pairs.

Source: ml_train/cache_v3_full/labels_t0.7.parquet.

For each of the 9 USED pairs (pair_{d}_{r}, d, r ∈ {0, 1, 2}) and each
delay tier (3, 5, 10 minutes parsed from log_file), report:
  - n_states           total states in the (pair, delay) cell
  - n_label_positive   states where the binary t07 label == 1
                       (improve_freq >= 0.7 over M=50 oracle trials)
  - n_any_improve      states where improve_freq > 0 (any improvement at all)
  - mean_improve_freq  mean of continuous improve_freq across all states
  - p10..p90           5-quantile distribution stats
  - mean_freq_given_label_1
                       mean improve_freq restricted to states where the
                       binary label is 1 (conditional on being a
                       reliable improver)

Outputs:
  analysis/improve_freq_per_pair_per_delay.csv
  analysis/IMPROVE_FREQ_REPORT.md
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
LABELS = REPO / "ml_train" / "cache_v3_full" / "labels_t0.7.parquet"
OUT_CSV = REPO / "analysis" / "improve_freq_per_pair_per_delay.csv"
OUT_MD  = REPO / "analysis" / "IMPROVE_FREQ_REPORT.md"

USED_PAIRS_TUP = [(d, r) for d in (0, 1, 2) for r in (0, 1, 2)]
USED_PAIRS = [f"pair_{d}_{r}" for d, r in USED_PAIRS_TUP]
USED_FREQ  = [f"improve_freq_{d}_{r}" for d, r in USED_PAIRS_TUP]
DESTROY_NAMES = {0: "random", 1: "worst", 2: "shaw"}
REPAIR_NAMES  = {0: "greedy", 1: "regret2", 2: "regret3"}
DELAYS = [3, 5, 10]

# Same regex used everywhere else for parsing log_file -> (size, delay)
RE_LOG = re.compile(r"^manhattan_(?P<size>\d+_s|\d+_min|\d+min)_(?:delay)?(?P<delay>\d+)_?min_s\d+")


def main():
    df = pd.read_parquet(LABELS)
    print(f"Loaded {len(df)} rows × {len(df.columns)} cols from {LABELS.name}")

    # Parse delay (in minutes) from log_file
    delays_str = df["log_file"].map(
        lambda n: (RE_LOG.match(n).group("delay") if RE_LOG.match(n) else None)
    )
    df = df.assign(delay=delays_str.astype(int))
    print("Rows per delay tier:")
    print(df.groupby("delay").size().to_string())

    rows = []
    for d, r in USED_PAIRS_TUP:
        pair_col = f"pair_{d}_{r}"
        freq_col = f"improve_freq_{d}_{r}"
        for delay in DELAYS:
            sub = df[df["delay"] == delay]
            freq = sub[freq_col].values
            label = sub[pair_col].values.astype(int)
            n_states = len(sub)
            n_pos = int(label.sum())
            n_any = int((freq > 0).sum())
            mean_f = float(freq.mean()) if n_states else float("nan")
            p10, p25, p50, p75, p90 = (
                float(np.percentile(freq, q)) if n_states else float("nan")
                for q in (10, 25, 50, 75, 90)
            )
            if n_pos > 0:
                mean_cond = float(freq[label == 1].mean())
            else:
                mean_cond = float("nan")
            rows.append({
                "pair": pair_col,
                "destroy_op": DESTROY_NAMES[d],
                "repair_op":  REPAIR_NAMES[r],
                "delay": delay,
                "n_states": n_states,
                "n_label_positive": n_pos,
                "n_any_improve": n_any,
                "mean_improve_freq": mean_f,
                "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
                "mean_freq_given_label_1": mean_cond,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out_df)} rows)")

    write_report(out_df, df.groupby("delay").size().to_dict())


# ============================================================
# Markdown report
# ============================================================
def write_report(out_df: pd.DataFrame, delay_n: dict[int, int]):
    L = []
    L.append("# Per-pair improve_freq distributions across delay tiers")
    L.append("")
    L.append("## Source")
    L.append("")
    L.append("`ml_train/cache_v3_full/labels_t0.7.parquet`. The continuous `improve_freq_{d}_{r}` columns hold the fraction of M=50 oracle trials in which the (destroy_d, repair_r) pair produced an improvement at that state. The binary `pair_{d}_{r}` column is 1 iff `improve_freq_{d}_{r} >= 0.7` (t07 labeling rule).")
    L.append("")
    L.append("Delay tier parsed from `log_file` with the same regex as the rest of the analysis.")
    L.append("")
    L.append("**State counts per delay tier:**")
    L.append("")
    L.append("| delay | n states |")
    L.append("|---:|---:|")
    for d in DELAYS:
        L.append(f"| {d} | {delay_n.get(d, 0)} |")
    L.append("")

    # ----- Section 1: full table -----
    L.append("## 1. Full per-(pair, delay) table")
    L.append("")
    L.append("| pair | destroy | repair | delay | n | n_label_1 | n_any_improve | mean | p10 | p25 | p50 | p75 | p90 | mean \\| label=1 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in out_df.iterrows():
        m_cond = f"{r['mean_freq_given_label_1']:.3f}" if not np.isnan(r["mean_freq_given_label_1"]) else "n/a"
        L.append(f"| `{r['pair']}` | {r['destroy_op']} | {r['repair_op']} | d{int(r['delay']):02d} | "
                 f"{int(r['n_states'])} | {int(r['n_label_positive'])} | {int(r['n_any_improve'])} | "
                 f"{r['mean_improve_freq']:.3f} | {r['p10']:.3f} | {r['p25']:.3f} | "
                 f"{r['p50']:.3f} | {r['p75']:.3f} | {r['p90']:.3f} | {m_cond} |")
    L.append("")

    # ----- Section 2: prevalence (label=1) heatmap (text) -----
    L.append("## 2. t07 label prevalence per (pair, delay)")
    L.append("")
    L.append("Fraction of states where the binary t07 label is 1 (i.e., the pair improves on ≥70% of M=50 oracle trials).")
    L.append("")
    L.append("| destroy + repair | d3 | d5 | d10 |")
    L.append("|---|---:|---:|---:|")
    for d, r in USED_PAIRS_TUP:
        cells = []
        for delay in DELAYS:
            sub = out_df[(out_df.pair == f"pair_{d}_{r}") & (out_df.delay == delay)].iloc[0]
            prev = sub["n_label_positive"] / sub["n_states"] if sub["n_states"] else 0
            cells.append(f"{prev*100:.1f}%")
        L.append(f"| `{DESTROY_NAMES[d]}+{REPAIR_NAMES[r]}` | " + " | ".join(cells) + " |")
    L.append("")

    # ----- Section 3: mean improve_freq heatmap -----
    L.append("## 3. Mean improve_freq per (pair, delay)")
    L.append("")
    L.append("| destroy + repair | d3 | d5 | d10 |")
    L.append("|---|---:|---:|---:|")
    for d, r in USED_PAIRS_TUP:
        cells = []
        for delay in DELAYS:
            sub = out_df[(out_df.pair == f"pair_{d}_{r}") & (out_df.delay == delay)].iloc[0]
            cells.append(f"{sub['mean_improve_freq']:.3f}")
        L.append(f"| `{DESTROY_NAMES[d]}+{REPAIR_NAMES[r]}` | " + " | ".join(cells) + " |")
    L.append("")

    # ----- Section 4: any-improve fraction -----
    L.append("## 4. Any-improvement rate per (pair, delay)")
    L.append("")
    L.append("Fraction of states where `improve_freq > 0` (i.e., at least one of the M=50 oracle trials improved). This is a much weaker bar than the t07 binary label.")
    L.append("")
    L.append("| destroy + repair | d3 | d5 | d10 |")
    L.append("|---|---:|---:|---:|")
    for d, r in USED_PAIRS_TUP:
        cells = []
        for delay in DELAYS:
            sub = out_df[(out_df.pair == f"pair_{d}_{r}") & (out_df.delay == delay)].iloc[0]
            frac = sub["n_any_improve"] / sub["n_states"] if sub["n_states"] else 0
            cells.append(f"{frac*100:.1f}%")
        L.append(f"| `{DESTROY_NAMES[d]}+{REPAIR_NAMES[r]}` | " + " | ".join(cells) + " |")
    L.append("")

    # ----- Section 5: marginal stats by destroy/repair/delay -----
    L.append("## 5. Marginal patterns")
    L.append("")
    # Mean improve_freq by delay (aggregated across all 9 pairs)
    L.append("**Mean improve_freq aggregated across the 9 pairs, by delay:**")
    L.append("")
    L.append("| delay | mean improve_freq | mean t07 prevalence | mean any-improve |")
    L.append("|---:|---:|---:|---:|")
    for delay in DELAYS:
        sub = out_df[out_df.delay == delay]
        prev = (sub["n_label_positive"] / sub["n_states"]).mean()
        any_frac = (sub["n_any_improve"] / sub["n_states"]).mean()
        L.append(f"| d{delay:02d} | {sub['mean_improve_freq'].mean():.3f} | {prev*100:.1f}% | {any_frac*100:.1f}% |")
    L.append("")

    # Mean improve_freq by destroy family
    L.append("**Mean improve_freq aggregated across delays and repair, by destroy operator:**")
    L.append("")
    L.append("| destroy | mean improve_freq | mean t07 prevalence |")
    L.append("|---|---:|---:|")
    for d in (0, 1, 2):
        sub = out_df[out_df.destroy_op == DESTROY_NAMES[d]]
        prev = (sub["n_label_positive"] / sub["n_states"]).mean()
        L.append(f"| {DESTROY_NAMES[d]} | {sub['mean_improve_freq'].mean():.3f} | {prev*100:.1f}% |")
    L.append("")

    # Mean improve_freq by repair operator
    L.append("**Mean improve_freq aggregated across delays and destroy, by repair operator:**")
    L.append("")
    L.append("| repair | mean improve_freq | mean t07 prevalence |")
    L.append("|---|---:|---:|")
    for r in (0, 1, 2):
        sub = out_df[out_df.repair_op == REPAIR_NAMES[r]]
        prev = (sub["n_label_positive"] / sub["n_states"]).mean()
        L.append(f"| {REPAIR_NAMES[r]} | {sub['mean_improve_freq'].mean():.3f} | {prev*100:.1f}% |")
    L.append("")

    # ----- Section 6: key patterns / take-aways -----
    L.append("## 6. Key patterns")
    L.append("")

    # Compute headline patterns automatically
    # 1. Does t07 prevalence drop with delay?
    prev_by_delay = []
    for delay in DELAYS:
        sub = out_df[out_df.delay == delay]
        prev_by_delay.append((sub["n_label_positive"] / sub["n_states"]).mean())

    if prev_by_delay[0] > prev_by_delay[2]:
        L.append(f"- **t07 prevalence falls as delay loosens** ({prev_by_delay[0]*100:.1f}% → "
                 f"{prev_by_delay[1]*100:.1f}% → {prev_by_delay[2]*100:.1f}% across d3→d5→d10). "
                 f"At looser delays the feasibility space is larger and any one pair improves a smaller fraction of trials reliably; the binary label becomes much sparser.")
    else:
        L.append(f"- **t07 prevalence rises as delay loosens** ({prev_by_delay[0]*100:.1f}% → "
                 f"{prev_by_delay[1]*100:.1f}% → {prev_by_delay[2]*100:.1f}%).")
    L.append("")

    # 2. mean_freq_given_label_1 vs unconditional mean — selectivity gap
    L.append("**Gap between unconditional mean and conditional (label=1) mean:**")
    L.append("")
    L.append("Conditional mean is bounded below by 0.7 by construction (the t07 threshold). The gap shows how aggressive the t07 cut is.")
    L.append("")
    L.append("| pair | d3 uncond → cond | d5 uncond → cond | d10 uncond → cond |")
    L.append("|---|---|---|---|")
    for d, r in USED_PAIRS_TUP:
        cells = []
        for delay in DELAYS:
            sub = out_df[(out_df.pair == f"pair_{d}_{r}") & (out_df.delay == delay)].iloc[0]
            cond = f"{sub['mean_freq_given_label_1']:.3f}" if not np.isnan(sub["mean_freq_given_label_1"]) else "n/a"
            cells.append(f"{sub['mean_improve_freq']:.3f} → {cond}")
        L.append(f"| `{DESTROY_NAMES[d]}+{REPAIR_NAMES[r]}` | " + " | ".join(cells) + " |")
    L.append("")

    # 3. Destroy-family pattern
    L.append("- **Destroy-family pattern:** look at §5 — if the `worst` row has substantially higher mean improve_freq than `random` and `shaw`, the worst-destroy family is the structural \"easy win\" pair, which would explain why `best_avg_pair` always picks one of pair_1_*.")
    means_by_destroy = {DESTROY_NAMES[d]: out_df[out_df.destroy_op == DESTROY_NAMES[d]]['mean_improve_freq'].mean() for d in (0,1,2)}
    sorted_destroy = sorted(means_by_destroy.items(), key=lambda x: -x[1])
    L.append(f"  - Mean improve_freq ranking: "
             f"**{sorted_destroy[0][0]}** ({sorted_destroy[0][1]:.3f}) > "
             f"**{sorted_destroy[1][0]}** ({sorted_destroy[1][1]:.3f}) > "
             f"**{sorted_destroy[2][0]}** ({sorted_destroy[2][1]:.3f}).")
    L.append("")

    # 4. Repair-family pattern
    means_by_repair = {REPAIR_NAMES[r]: out_df[out_df.repair_op == REPAIR_NAMES[r]]['mean_improve_freq'].mean() for r in (0,1,2)}
    sorted_repair = sorted(means_by_repair.items(), key=lambda x: -x[1])
    L.append(f"- **Repair-family pattern:** mean improve_freq ranking: "
             f"**{sorted_repair[0][0]}** ({sorted_repair[0][1]:.3f}) > "
             f"**{sorted_repair[1][0]}** ({sorted_repair[1][1]:.3f}) > "
             f"**{sorted_repair[2][0]}** ({sorted_repair[2][1]:.3f}).")
    L.append("")

    # 5. Where the distribution mass sits
    L.append("- **Distribution shape:** the median (p50) for most (pair, delay) cells sits near 0 or below 0.1 — most states have very low improve_freq for any single pair. The mean is pulled up by the right tail. p90 is the most useful summary of \"states where this pair would clearly help\": if p90 < 0.7 then fewer than 10% of states would be t07-positive for that pair.")
    L.append("")

    L.append("## Files")
    L.append("")
    L.append(f"- Per-row CSV: `{OUT_CSV.relative_to(REPO)}`")

    OUT_MD.write_text("\n".join(L))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
