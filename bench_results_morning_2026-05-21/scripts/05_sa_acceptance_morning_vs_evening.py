#!/usr/bin/env python3
"""Why does SA accept so many worsening moves? Compute SA acceptance rate
per selector on morning and evening benches, and compare.

For each log we sum across epochs:
- iters: total ALNS iterations
- improve: improving moves (auto-accepted)
- best:    subset of improve that became new global best
- sa:      worsening moves accepted via SA criterion

Derived:
- accept_rate     = (improve + sa) / iters
- worsen_accept   = sa / (iters - improve)   # SA acceptance among worsening
- improve_rate    = improve / iters
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
MORN = ROOT / "bench_results_morning_2026-05-21" / "raw_logs"
EVEN = ROOT / "bench_results_final" / "raw_logs"
OUT  = ROOT / "bench_results_morning_2026-05-21" / "SA_ACCEPTANCE_MORNING_VS_EVENING.md"

_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")
_RE_MORN = re.compile(r"dynamic__manhattan_morning_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EVEN = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")

DELAYS = [3, 5, 10]
INSTANCES = ["30_s", "01_min", "02_min", "05_min"]

# Selectors common to both cohorts
SELECTORS = ["random", "roulette", "ml_20f_t07__T0.10"]


def parse_logs(root: Path, name_re: re.Pattern, cohort_label: str):
    rows = []
    for p in sorted(root.glob("*.log")):
        m = name_re.match(p.name)
        if not m: continue
        sel = m.group(3)
        if sel not in SELECTORS: continue
        it = im = bst = sa = 0
        for ln in p.read_text().splitlines():
            em = _RE_EPOCH.search(ln)
            if em:
                a, b, c, d = map(int, em.groups())
                it += a; im += b; bst += c; sa += d
        if it == 0: continue
        rows.append({
            "cohort": cohort_label,
            "instance_base": m.group(1), "delay": int(m.group(2)),
            "selector": sel, "budget": int(m.group(4)), "seed": int(m.group(5)),
            "iters": it, "improve": im, "best": bst, "sa": sa,
            "improve_rate": im / it,
            "accept_rate": (im + sa) / it,
            "worsen_accept_rate": sa / max(it - im, 1),
        })
    return pd.DataFrame(rows)


def main():
    morn = parse_logs(MORN, _RE_MORN, "morning")
    even = parse_logs(EVEN, _RE_EVEN, "evening")
    print(f"Morning rows: {len(morn)}")
    print(f"Evening rows: {len(even)}")

    L = []
    L.append("# SA acceptance rate — morning vs evening")
    L.append("")
    L.append("**Question:** Why does SA accept so many worsening moves in the morning bench? Was this also the case in the evening bench?")
    L.append("")
    L.append("**Metrics:**")
    L.append("- `improve_rate` = improving iters / total iters")
    L.append("- `worsen_accept_rate` = sa-accepted iters / (iters − improving iters)")
    L.append("- `accept_rate` = (improving + sa-accepted) / total iters")
    L.append("")
    L.append("Higher `worsen_accept_rate` ⇒ SA is hotter (accepting more bad moves) ⇒ less greedy exploration.")
    L.append("")

    # §1 overall per cohort
    L.append("## 1. Overall acceptance rates per cohort")
    L.append("")
    L.append("| cohort | selector | improve | worsen accept | overall accept | n logs |")
    L.append("|---|---|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        for sel in SELECTORS:
            sub = df[df.selector == sel]
            if len(sub) == 0:
                L.append(f"| {label} | `{sel}` | n/a | n/a | n/a | 0 |")
                continue
            L.append(f"| {label} | `{sel}` | {sub.improve_rate.mean():.4f} | {sub.worsen_accept_rate.mean():.4f} | {sub.accept_rate.mean():.4f} | {len(sub)} |")
    L.append("")

    # §2 per delay
    L.append("## 2. Worsen-accept rate per delay")
    L.append("")
    L.append("Higher = SA hotter. If morning > evening, SA is accepting more bad moves on the morning cohort.")
    L.append("")
    L.append("| cohort | selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        for sel in SELECTORS:
            cells = []
            for d in DELAYS:
                sub = df[(df.selector == sel) & (df.delay == d)]
                cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
            L.append(f"| {label} | `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §3 per instance — most interesting: 30_s where there's little ride-time budget
    L.append("## 3. Worsen-accept rate per instance × delay (`ml_20f_t07__T0.10`)")
    L.append("")
    L.append("Compares the deployed model only, instance-by-instance.")
    L.append("")
    L.append("| cohort | instance | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        for inst in INSTANCES:
            cells = []
            for d in DELAYS:
                sub = df[(df.selector == "ml_20f_t07__T0.10") & (df.delay == d) & (df.instance_base == inst)]
                cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
            L.append(f"| {label} | {inst} | " + " | ".join(cells) + " |")
    L.append("")

    # §4 paired delta morning - evening per (selector, delay)
    L.append("## 4. Δ (morning − evening) in worsen-accept rate, paired by (instance, delay, budget, seed)")
    L.append("")
    L.append("Positive ⇒ morning SA accepts more bad moves than evening.")
    L.append("")
    L.append("| selector | delay 3 | delay 5 | delay 10 |")
    L.append("|---|---:|---:|---:|")
    merged_keys = ["instance_base", "delay", "budget", "seed", "selector"]
    m2 = morn.set_index(merged_keys)["worsen_accept_rate"]
    e2 = even.set_index(merged_keys)["worsen_accept_rate"]
    joined = pd.DataFrame({"morn": m2, "even": e2}).dropna()
    joined["delta"] = joined["morn"] - joined["even"]
    joined = joined.reset_index()
    for sel in SELECTORS:
        cells = []
        for d in DELAYS:
            sub = joined[(joined.selector == sel) & (joined.delay == d)]
            if len(sub) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{sub.delta.mean():+.4f} (n={len(sub)})")
        L.append(f"| `{sel}` | " + " | ".join(cells) + " |")
    L.append("")

    # §5 summary
    L.append("## 5. Interpretation")
    L.append("")
    L.append("The SA acceptance criterion is `exp(-Δobj/T)` where Δobj is the proposed worsening and T is the normalised temperature.")
    L.append("Three things can make SA accept more worsening moves:")
    L.append("1. **Higher T** — the schedule starts hotter.")
    L.append("2. **Smaller worsening sizes** — the typical Δobj is small relative to current obj, so `exp(-Δobj/T)` ≈ 1.")
    L.append("3. **The optimiser keeps proposing tiny diffs** — bad operators don't break much.")
    L.append("")
    L.append("If morning ≈ evening in worsen-accept rate, the SA acceptance is structural to the cooling schedule (T schedule + scale).")
    L.append("If morning > evening, morning has either smaller typical worsenings (fewer requests = smaller Δobj scale) or the schedule is misaligned with the lower-density instances.")
    L.append("")
    L.append("Note: with worsen-accept rate ~0.95, the operator selector influences only ~5% of accepted moves directly via the 'improve' channel. The other 95% are exploration moves that SA waves through regardless of which destroy/repair fired. This caps the maximum lift any operator-selection learner can deliver on the final objective.")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
