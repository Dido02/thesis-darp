#!/usr/bin/env python3
"""Investigate WHEN SA is accepting moves:
  (a) Does the SA worsen-accept rate cool across epochs of one run?
  (b) Does a 50x longer budget (10000ms vs 200ms) give SA enough time
      to cool within an epoch, lowering worsen-accept?

Compares morning vs evening (both ml_20f_t07__T0.10) to test whether the
behaviour is structural or cohort-specific.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/didodimitrov/Documents/thesis-darp")
MORN = ROOT / "bench_results_morning_2026-05-21" / "raw_logs"
EVEN = ROOT / "bench_results_final" / "raw_logs"
OUT  = ROOT / "bench_results_morning_2026-05-21" / "SA_PER_EPOCH_AND_BUDGET.md"

_RE_EPOCH = re.compile(r"iters=(\d+) \(improve=(\d+) best=(\d+) sa=(\d+)\)")
_RE_MORN = re.compile(r"dynamic__manhattan_morning_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")
_RE_EVEN = re.compile(r"dynamic__manhattan_(\d+_s|\d+_min)_delay(\d+)_min__(\S+?)__b(\d+)__s(\d+)\.log")

SELECTOR = "ml_20f_t07__T0.10"
BUDGETS = [200, 1000, 5000, 10000]


def parse_epochs(root: Path, name_re: re.Pattern, cohort: str):
    rows = []
    for p in sorted(root.glob("*.log")):
        m = name_re.match(p.name)
        if not m: continue
        if m.group(3) != SELECTOR: continue
        budget = int(m.group(4))
        epoch_idx = 0
        for line in p.read_text().splitlines():
            em = _RE_EPOCH.search(line)
            if not em: continue
            it, im, bst, sa = map(int, em.groups())
            if it == 0 or it == im: continue
            rows.append({
                "cohort": cohort,
                "instance_base": m.group(1), "delay": int(m.group(2)),
                "budget": budget, "seed": int(m.group(5)),
                "epoch": epoch_idx,
                "iters": it, "improve": im, "best": bst, "sa": sa,
                "improve_rate": im / it,
                "worsen_accept_rate": sa / (it - im),
            })
            epoch_idx += 1
    return pd.DataFrame(rows)


def main():
    morn = parse_epochs(MORN, _RE_MORN, "morning")
    even = parse_epochs(EVEN, _RE_EVEN, "evening")
    print(f"morning epoch-rows: {len(morn)}")
    print(f"evening epoch-rows: {len(even)}")

    L = []
    L.append("# SA acceptance — across epochs and across budgets")
    L.append("")
    L.append("Selector: `ml_20f_t07__T0.10`. All numbers are mean `worsen_accept_rate = sa / (iters − improve)`.")
    L.append("")
    L.append("If the dynamic-ALNS temperature schedule resets each epoch and cools within an epoch:")
    L.append("- across epochs of one run: rate should be roughly similar (each epoch starts hot again).")
    L.append("- across budgets: longer budget per epoch ⇒ more iterations of cooling ⇒ lower average rate.")
    L.append("")
    L.append("If neither pattern shows up: temperature is too hot for the worsening magnitudes on these instances.")
    L.append("")

    # §1 per-epoch index — does it cool across the 7 dynamic epochs of a run?
    L.append("## 1. Worsen-accept rate vs epoch index (averaged across all instances/seeds/budgets)")
    L.append("")
    L.append("| cohort | epoch 0 | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | epoch 6 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        cells = []
        for e in range(7):
            sub = df[df.epoch == e]
            cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Cooling across epochs: a value that drops monotonically over the row says the SA temperature carries state from one epoch to the next. A roughly flat row says temperature resets each epoch.")
    L.append("")

    # §2 per-budget — does longer budget per epoch give SA time to cool?
    L.append("## 2. Worsen-accept rate vs budget (avg across all epochs/instances/seeds)")
    L.append("")
    L.append("| cohort | b=200 | b=1000 | b=5000 | b=10000 |")
    L.append("|---|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        cells = []
        for b in BUDGETS:
            sub = df[df.budget == b]
            cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    # §3 same but iter counts so we see what 'more time' means in iters
    L.append("## 3. Mean iters per epoch by budget")
    L.append("")
    L.append("| cohort | b=200 | b=1000 | b=5000 | b=10000 |")
    L.append("|---|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        cells = []
        for b in BUDGETS:
            sub = df[df.budget == b]
            cells.append("n/a" if len(sub) == 0 else f"{sub.iters.mean():.0f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("This is the headline: if b=10000 gives 50× more iters per epoch than b=200 yet the SA rate barely drops, then the per-iter cooling step is too small to ever bring T down to a useful level.")
    L.append("")

    # §4 by (budget, delay) for instance 05_min (the largest, where Δobj should be biggest)
    L.append("## 4. Worsen-accept rate on `05_min` (largest, biggest Δobj) per (budget, delay)")
    L.append("")
    L.append("| cohort | delay | b=200 | b=1000 | b=5000 | b=10000 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        for d in [3, 5, 10]:
            cells = []
            for b in BUDGETS:
                sub = df[(df.budget == b) & (df.delay == d) & (df.instance_base == "05_min")]
                cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
            L.append(f"| {label} | d{d:02d} | " + " | ".join(cells) + " |")
    L.append("")

    # §5 by (budget, delay) for instance 30_s (smallest, smallest Δobj)
    L.append("## 5. Worsen-accept rate on `30_s` (smallest, smallest Δobj) per (budget, delay)")
    L.append("")
    L.append("| cohort | delay | b=200 | b=1000 | b=5000 | b=10000 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for df, label in [(morn, "morning"), (even, "evening")]:
        for d in [3, 5, 10]:
            cells = []
            for b in BUDGETS:
                sub = df[(df.budget == b) & (df.delay == d) & (df.instance_base == "30_s")]
                cells.append("n/a" if len(sub) == 0 else f"{sub.worsen_accept_rate.mean():.4f}")
            L.append(f"| {label} | d{d:02d} | " + " | ".join(cells) + " |")
    L.append("")

    # §6 verdict
    L.append("## 6. Verdict")
    L.append("")
    flat_thresh = 0.02   # if range across epochs < 0.02 ⇒ flat
    morn_by_epoch = [morn[morn.epoch == e].worsen_accept_rate.mean() for e in range(7)]
    even_by_epoch = [even[even.epoch == e].worsen_accept_rate.mean() for e in range(7)]
    morn_range = max(morn_by_epoch) - min(morn_by_epoch)
    even_range = max(even_by_epoch) - min(even_by_epoch)
    L.append(f"- Range across the 7 epochs (morning): **{morn_range:.4f}**")
    L.append(f"- Range across the 7 epochs (evening): **{even_range:.4f}**")
    L.append(f"- A range < {flat_thresh:.2f} means epochs are essentially indistinguishable in SA behaviour (temperature fully resets each epoch and cools to the same point).")
    L.append("")
    morn_b200 = morn[morn.budget == 200].worsen_accept_rate.mean()
    morn_b10k = morn[morn.budget == 10000].worsen_accept_rate.mean()
    even_b200 = even[even.budget == 200].worsen_accept_rate.mean()
    even_b10k = even[even.budget == 10000].worsen_accept_rate.mean()
    L.append(f"- Δ across budgets (morning, b200 − b10000): **{morn_b200 - morn_b10k:+.4f}**")
    L.append(f"- Δ across budgets (evening, b200 − b10000): **{even_b200 - even_b10k:+.4f}**")
    L.append(f"- A Δ near 0 means a 50× longer per-epoch budget barely changes the SA acceptance rate — i.e. within-epoch cooling is weak.")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
