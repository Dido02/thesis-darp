"""Exploratory labeling sweep: compare 5 labeling rules across thresholds.

Reads JSONL logs directly, filters epoch 1, computes labels in-memory,
and produces labeling_sweep_report.md with per-rule/threshold statistics
and SA regime analysis.

Usage:
    cd ml_train && python labeling_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data structures (minimal, independent of darpml to keep this self-contained)
# ---------------------------------------------------------------------------
@dataclass
class TrialPair:
    d_idx: int
    r_idx: int
    deltas: list[float]


@dataclass
class StateTrialRecord:
    log_file: str
    epoch_id: int
    iter_in_epoch: int
    obj_before: float
    temperature: float
    trial_pairs: list[TrialPair]


# ---------------------------------------------------------------------------
# JSONL streaming loader with epoch tracking
# ---------------------------------------------------------------------------
def load_all_state_trials(logs_dir: Path) -> list[StateTrialRecord]:
    """Load all state-trial pairs from all JSONL logs, tracking epoch_id."""
    records: list[StateTrialRecord] = []
    log_files = sorted(logs_dir.glob("*.jsonl"))

    for lf in log_files:
        current_epoch = 0
        states_by_iter: dict[int, dict] = {}
        trials_by_iter: dict[int, dict] = {}
        epoch_by_iter: dict[int, int] = {}

        with open(lf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rtype = rec["type"]

                if rtype == "epoch_start":
                    current_epoch = rec["epoch_id"]
                elif rtype == "state":
                    it = rec["iter"]
                    states_by_iter[it] = rec
                    epoch_by_iter[it] = current_epoch
                elif rtype == "trials":
                    it = rec["iter"]
                    trials_by_iter[it] = rec

        # Pair state + trials
        for it in sorted(states_by_iter.keys()):
            if it not in trials_by_iter:
                continue
            tr = trials_by_iter[it]
            pairs = []
            for entry in tr.get("results", []):
                pairs.append(TrialPair(
                    d_idx=entry["d_idx"],
                    r_idx=entry["r_idx"],
                    deltas=entry["deltas"],
                ))
            if not pairs:
                continue
            records.append(StateTrialRecord(
                log_file=lf.name,
                epoch_id=epoch_by_iter.get(it, 0),
                iter_in_epoch=it,
                obj_before=tr["obj_before"],
                temperature=tr["temperature"],
                trial_pairs=pairs,
            ))

    return records


# ---------------------------------------------------------------------------
# SA acceptance probability
# ---------------------------------------------------------------------------
def sa_prob(delta: float, T: float) -> float:
    if delta <= 0:
        return 1.0
    if T < 1e-10:
        return 0.0
    return math.exp(-delta / T)


# ---------------------------------------------------------------------------
# Labeling rules
# ---------------------------------------------------------------------------
EPS = 1e-6


def rule1_bongiovanni(rec: StateTrialRecord, threshold: float) -> list[int]:
    """SA acceptance rate within X of best."""
    T = rec.temperature
    rates = {}
    for p in rec.trial_pairs:
        probs = [sa_prob(d, T) for d in p.deltas]
        rates[(p.d_idx, p.r_idx)] = sum(probs) / len(probs)
    best = max(rates.values())
    if best < EPS:
        return [0] * len(rec.trial_pairs)
    cutoff = threshold * best
    labels = []
    for p in rec.trial_pairs:
        labels.append(1 if rates[(p.d_idx, p.r_idx)] >= cutoff else 0)
    return labels


def rule2_improvement_freq(rec: StateTrialRecord, threshold: float) -> list[int]:
    """Fraction of trials with delta < -eps, label=1 if >= threshold."""
    labels = []
    for p in rec.trial_pairs:
        n_improve = sum(1 for d in p.deltas if d < -EPS)
        frac = n_improve / len(p.deltas) if p.deltas else 0.0
        labels.append(1 if frac >= threshold else 0)
    return labels


def rule3_improvement_freq_relative(rec: StateTrialRecord, threshold: float) -> list[int]:
    """Improvement fraction within X of best improvement fraction."""
    fracs = {}
    for p in rec.trial_pairs:
        n_improve = sum(1 for d in p.deltas if d < -EPS)
        fracs[(p.d_idx, p.r_idx)] = n_improve / len(p.deltas) if p.deltas else 0.0
    best_frac = max(fracs.values())
    if best_frac < EPS:
        return [0] * len(rec.trial_pairs)
    cutoff = threshold * best_frac
    labels = []
    for p in rec.trial_pairs:
        labels.append(1 if fracs[(p.d_idx, p.r_idx)] >= cutoff else 0)
    return labels


def rule4_mean_delta(rec: StateTrialRecord, threshold: float) -> list[int]:
    """Mean delta below threshold × obj_before."""
    obj = rec.obj_before
    cutoff = threshold * obj  # threshold is negative, so cutoff is negative
    labels = []
    for p in rec.trial_pairs:
        mean_d = sum(p.deltas) / len(p.deltas) if p.deltas else 0.0
        labels.append(1 if mean_d < cutoff else 0)
    return labels


def rule5_combined(rec: StateTrialRecord, threshold: float) -> list[int]:
    """Fraction of trials where delta < -eps AND SA prob > 0.5."""
    T = rec.temperature
    labels = []
    for p in rec.trial_pairs:
        n_both = sum(1 for d in p.deltas if d < -EPS and sa_prob(d, T) > 0.5)
        frac = n_both / len(p.deltas) if p.deltas else 0.0
        labels.append(1 if frac >= threshold else 0)
    return labels


RULES = {
    "R1_bongiovanni": (rule1_bongiovanni, [0.9, 0.7, 0.5, 0.3]),
    "R2_improve_freq": (rule2_improvement_freq, [0.5, 0.3, 0.2, 0.1]),
    "R3_improve_rel": (rule3_improvement_freq_relative, [0.9, 0.7, 0.5, 0.3]),
    "R4_mean_delta": (rule4_mean_delta, [-0.001, -0.005, -0.01, -0.05]),
    "R5_combined": (rule5_combined, [0.5, 0.3, 0.2]),
}


# ---------------------------------------------------------------------------
# SA regime classification
# ---------------------------------------------------------------------------
def classify_sa_regime(rec: StateTrialRecord) -> str:
    """Classify state into SA regime based on mean pair acceptance rate."""
    T = rec.temperature
    rates = []
    for p in rec.trial_pairs:
        probs = [sa_prob(d, T) for d in p.deltas]
        rates.append(sum(probs) / len(probs))
    mean_rate = sum(rates) / len(rates)
    if mean_rate > 0.75:
        return "high-T"
    elif mean_rate < 0.10:
        return "low-T"
    else:
        return "discriminative"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    logs_dir = Path("../trail_logs")
    if not logs_dir.exists():
        logs_dir = Path("trail_logs")
    if not logs_dir.exists():
        print("ERROR: Cannot find trail_logs/ directory")
        sys.exit(1)

    print("Loading all state-trial records...")
    all_records = load_all_state_trials(logs_dir)
    total_before = len(all_records)
    print(f"  Total records: {total_before}")

    # Count epoch 1 states
    epoch1_count = sum(1 for r in all_records if r.epoch_id == 1)
    records = [r for r in all_records if r.epoch_id != 1]
    total_after = len(records)
    print(f"  Filtered epoch 1: {epoch1_count} removed, {total_after} remaining")

    # Also filter uninformative (all deltas == 0)
    def is_uninformative(rec):
        return all(d == 0.0 for p in rec.trial_pairs for d in p.deltas)

    uninformative = sum(1 for r in records if is_uninformative(r))
    records = [r for r in records if not is_uninformative(r)]
    final_count = len(records)
    print(f"  Filtered uninformative: {uninformative} removed, {final_count} remaining")

    # Determine n_destroy, n_repair from first record
    d_indices = set()
    r_indices = set()
    for p in records[0].trial_pairs:
        d_indices.add(p.d_idx)
        r_indices.add(p.r_idx)
    n_d = max(d_indices) + 1
    n_r = max(r_indices) + 1
    n_pairs = n_d * n_r
    d_names = ["random", "worst", "shaw", "rtcritical", "lowslack"][:n_d]
    r_names = ["greedy", "regret2", "regret3", "slack"][:n_r]
    print(f"  Operators: {n_d} destroy × {n_r} repair = {n_pairs} pairs")

    # -----------------------------------------------------------------------
    # SA Regime Analysis
    # -----------------------------------------------------------------------
    print("\nAnalyzing SA regimes...")
    regimes = [classify_sa_regime(r) for r in records]

    regime_counts = defaultdict(int)
    for rg in regimes:
        regime_counts[rg] += 1

    # By epoch
    epoch_regime = defaultdict(lambda: defaultdict(int))
    for rec, rg in zip(records, regimes):
        epoch_regime[rec.epoch_id][rg] += 1

    # By within-epoch tertile
    # First, find max iter per (log_file, epoch_id) to compute tertiles
    max_iter_per_epoch = defaultdict(int)
    for rec in records:
        key = (rec.log_file, rec.epoch_id)
        max_iter_per_epoch[key] = max(max_iter_per_epoch[key], rec.iter_in_epoch)

    tertile_regime = defaultdict(lambda: defaultdict(int))
    for rec, rg in zip(records, regimes):
        key = (rec.log_file, rec.epoch_id)
        max_it = max_iter_per_epoch[key]
        if max_it == 0:
            tertile = "early"
        else:
            frac = rec.iter_in_epoch / max_it
            if frac < 0.333:
                tertile = "early"
            elif frac < 0.667:
                tertile = "middle"
            else:
                tertile = "late"
        tertile_regime[tertile][rg] += 1

    # -----------------------------------------------------------------------
    # Labeling sweep
    # -----------------------------------------------------------------------
    print("\nRunning labeling sweep...")

    # Results: dict of (rule_name, threshold) -> per_pair_pos_rates
    sweep_results: dict[tuple[str, float], np.ndarray] = {}

    for rule_name, (rule_fn, thresholds) in RULES.items():
        for thresh in thresholds:
            # Accumulate per-pair label counts
            pair_pos = np.zeros(n_pairs)
            pair_total = np.zeros(n_pairs)

            for rec in records:
                labels = rule_fn(rec, thresh)
                for i, lab in enumerate(labels):
                    pair_pos[i] += lab
                    pair_total[i] += 1

            pos_rates = pair_pos / np.maximum(pair_total, 1)
            sweep_results[(rule_name, thresh)] = pos_rates
            print(f"  {rule_name} thresh={thresh}: mean_pos={pos_rates.mean():.3f}, "
                  f"degenerate={sum((pos_rates < 0.05) | (pos_rates > 0.95))}, "
                  f"balanced={sum((pos_rates >= 0.3) & (pos_rates <= 0.7))}")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    print("\nWriting report...")
    out = Path("labeling_sweep_report.md")

    lines = []
    w = lines.append

    w("# Labeling Sweep Report")
    w("")
    w("## 1. Filtering")
    w("")
    w(f"- **Total states (all logs):** {total_before}")
    w(f"- **Epoch 1 removed:** {epoch1_count}")
    w(f"- **Uninformative removed (all deltas=0):** {uninformative}")
    w(f"- **States remaining:** {final_count}")
    w(f"- **Log files:** {len(set(r.log_file for r in records))}")
    w(f"- **Operators:** {n_d} destroy × {n_r} repair = {n_pairs} pairs")
    w("")

    # SA regime
    w("## 2. SA Regime Distribution")
    w("")
    w("SA regime is determined by the mean acceptance rate across all 20 pairs:")
    w("- **high-T** (>0.75): temperature so high that most moves are accepted regardless of quality")
    w("- **low-T** (<0.10): temperature so low that only strict improvements are accepted")
    w("- **discriminative** (0.10–0.75): acceptance rates vary meaningfully between pairs")
    w("")

    w("### 2a. Overall")
    w("")
    w("| Regime | Count | Fraction |")
    w("|--------|------:|--------:|")
    for rg in ["high-T", "discriminative", "low-T"]:
        cnt = regime_counts[rg]
        w(f"| {rg} | {cnt} | {cnt/final_count:.3f} |")
    w("")

    w("### 2b. By Epoch")
    w("")
    epoch_ids = sorted(epoch_regime.keys())
    w("| Epoch | high-T | discriminative | low-T | Total |")
    w("|------:|-------:|---------------:|------:|------:|")
    for eid in epoch_ids:
        total_ep = sum(epoch_regime[eid].values())
        ht = epoch_regime[eid]["high-T"]
        disc = epoch_regime[eid]["discriminative"]
        lt = epoch_regime[eid]["low-T"]
        w(f"| {eid} | {ht} ({ht/total_ep:.0%}) | {disc} ({disc/total_ep:.0%}) | {lt} ({lt/total_ep:.0%}) | {total_ep} |")
    w("")

    w("### 2c. By Within-Epoch Tertile")
    w("")
    w("| Tertile | high-T | discriminative | low-T | Total |")
    w("|---------|-------:|---------------:|------:|------:|")
    for tert in ["early", "middle", "late"]:
        total_t = sum(tertile_regime[tert].values())
        if total_t == 0:
            continue
        ht = tertile_regime[tert]["high-T"]
        disc = tertile_regime[tert]["discriminative"]
        lt = tertile_regime[tert]["low-T"]
        w(f"| {tert} | {ht} ({ht/total_t:.0%}) | {disc} ({disc/total_t:.0%}) | {lt} ({lt/total_t:.0%}) | {total_t} |")
    w("")

    # Summary table
    w("## 3. Summary Table")
    w("")
    w("| Rule | Threshold | Mean Pos Rate | Std Pos Rate | Degenerate (<5% or >95%) | Balanced (30-70%) |")
    w("|------|-----------|--------------|-------------|--------------------------|-------------------|")
    for (rule_name, thresh), pos_rates in sorted(sweep_results.items()):
        mean_pr = pos_rates.mean()
        std_pr = pos_rates.std()
        n_degen = int(np.sum((pos_rates < 0.05) | (pos_rates > 0.95)))
        n_bal = int(np.sum((pos_rates >= 0.3) & (pos_rates <= 0.7)))
        w(f"| {rule_name} | {thresh} | {mean_pr:.3f} | {std_pr:.3f} | {n_degen} | {n_bal} |")
    w("")

    # Per-pair detail tables
    w("## 4. Per-Pair Positive Rates (5×4 Grids)")
    w("")
    for (rule_name, thresh), pos_rates in sorted(sweep_results.items()):
        w(f"### {rule_name}, threshold={thresh}")
        w(f"Mean={pos_rates.mean():.3f}, Degenerate={int(np.sum((pos_rates < 0.05) | (pos_rates > 0.95)))}, "
          f"Balanced={int(np.sum((pos_rates >= 0.3) & (pos_rates <= 0.7)))}")
        w("")
        # Header row
        header = "| destroy \\ repair |"
        sep = "|------------------|"
        for rn in r_names:
            header += f" {rn} |"
            sep += "------:|"
        w(header)
        w(sep)
        for di in range(n_d):
            row = f"| {d_names[di]} |"
            for ri in range(n_r):
                idx = di * n_r + ri
                pr = pos_rates[idx]
                flag = ""
                if pr < 0.05 or pr > 0.95:
                    flag = " *"
                elif 0.3 <= pr <= 0.7:
                    flag = " **"
                row += f" {pr:.3f}{flag} |"
            w(row)
        w("")
        w("\\* = degenerate, \\*\\* = balanced")
        w("")

    # Recommendation
    w("## 5. Recommendation")
    w("")

    # Find best by each criterion
    # Criterion A: mean pos rate closest to 0.5
    by_balance = sorted(sweep_results.items(), key=lambda x: abs(x[1].mean() - 0.5))
    # Criterion B: fewest degenerate
    by_degen = sorted(sweep_results.items(),
                      key=lambda x: (np.sum((x[1] < 0.05) | (x[1] > 0.95)), abs(x[1].mean() - 0.5)))
    # Criterion C: most balanced pairs
    by_n_balanced = sorted(sweep_results.items(),
                           key=lambda x: -np.sum((x[1] >= 0.3) & (x[1] <= 0.7)))

    w("### Criterion A: Mean positive rate closest to 0.5")
    w("")
    for i in range(min(3, len(by_balance))):
        key, pr = by_balance[i]
        n_degen = int(np.sum((pr < 0.05) | (pr > 0.95)))
        n_bal = int(np.sum((pr >= 0.3) & (pr <= 0.7)))
        w(f"{i+1}. **{key[0]} @ {key[1]}**: mean={pr.mean():.3f}, degenerate={n_degen}, balanced={n_bal}")
    w("")

    w("### Criterion B: Fewest degenerate pairs")
    w("")
    for i in range(min(3, len(by_degen))):
        key, pr = by_degen[i]
        n_degen = int(np.sum((pr < 0.05) | (pr > 0.95)))
        n_bal = int(np.sum((pr >= 0.3) & (pr <= 0.7)))
        w(f"{i+1}. **{key[0]} @ {key[1]}**: degenerate={n_degen}, mean={pr.mean():.3f}, balanced={n_bal}")
    w("")

    w("### Criterion C: Most balanced pairs (pos rate in [0.3, 0.7])")
    w("")
    for i in range(min(3, len(by_n_balanced))):
        key, pr = by_n_balanced[i]
        n_degen = int(np.sum((pr < 0.05) | (pr > 0.95)))
        n_bal = int(np.sum((pr >= 0.3) & (pr <= 0.7)))
        w(f"{i+1}. **{key[0]} @ {key[1]}**: balanced={n_bal}, degenerate={n_degen}, mean={pr.mean():.3f}")
    w("")

    # Top 3 overall candidates (union of above, deduplicated)
    seen = set()
    top3 = []
    for ranking in [by_balance, by_degen, by_n_balanced]:
        for key, pr in ranking:
            if key not in seen:
                seen.add(key)
                top3.append((key, pr))
            if len(top3) >= 6:
                break
        if len(top3) >= 6:
            break

    w("### Top Candidates (union across criteria)")
    w("")
    w("| Rank | Rule | Threshold | Mean Pos | Degenerate | Balanced | Trade-off |")
    w("|------|------|-----------|---------|-----------|---------|-----------|")
    for i, (key, pr) in enumerate(top3[:6]):
        n_degen = int(np.sum((pr < 0.05) | (pr > 0.95)))
        n_bal = int(np.sum((pr >= 0.3) & (pr <= 0.7)))
        mean_pr = pr.mean()
        if mean_pr < 0.3:
            tradeoff = "sparse positives"
        elif mean_pr > 0.7:
            tradeoff = "too many positives"
        elif n_degen > 5:
            tradeoff = "many degenerate pairs"
        elif n_bal > 10:
            tradeoff = "excellent balance"
        else:
            tradeoff = "moderate balance"
        w(f"| {i+1} | {key[0]} | {key[1]} | {mean_pr:.3f} | {n_degen} | {n_bal} | {tradeoff} |")
    w("")

    w("### Caveat")
    w("")
    w("Label balance is **necessary but not sufficient** for learnable signal. "
      "The next step after picking a rule is to retrain with per-log-file split "
      "and see if any actual signal emerges above trivial baselines.")

    out.write_text("\n".join(lines))
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
