"""Label computation from M-trials data.

Supports two labeling rules:
- improvement_freq: fraction of trials where delta < -IMPROVEMENT_EPS >= threshold
- sa_acceptance: SA acceptance rate within threshold of best (Bongiovanni et al. 2022)
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .loader import TrialsRecord, StateTrial

# Delta must be strictly below -IMPROVEMENT_EPS to count as an improvement
IMPROVEMENT_EPS = 1e-6


# ---------------------------------------------------------------------------
# SA acceptance probability (kept for sa_acceptance rule)
# ---------------------------------------------------------------------------
def sa_acceptance_prob(delta: float, temperature: float) -> float:
    """Compute SA acceptance probability for a single move.

    p = 1.0              if delta <= 0  (improvement or neutral)
    p = exp(-delta / T)  if delta > 0   (worsening)
    """
    if delta <= 0:
        return 1.0
    if temperature < 1e-10:
        return 0.0
    return math.exp(-delta / temperature)


def compute_acceptance_rates(trials: TrialsRecord) -> dict[tuple[int, int], float]:
    """Compute expected SA acceptance rate for each operator pair."""
    T = trials.temperature
    rates = {}
    for pair in trials.results:
        probs = [sa_acceptance_prob(d, T) for d in pair.deltas]
        rates[(pair.d_idx, pair.r_idx)] = sum(probs) / len(probs) if probs else 0.0
    return rates


def compute_sa_labels(
    rates: dict[tuple[int, int], float],
    threshold_fraction: float = 0.9,
) -> dict[tuple[int, int], int]:
    """Binary labels from SA acceptance rates (Bongiovanni rule).

    label[d,r] = 1 if rate[d,r] >= threshold_fraction * best_rate, else 0.
    """
    if not rates:
        return {}
    best_rate = max(rates.values())
    if best_rate < 1e-10:
        return {k: 0 for k in rates}
    threshold = threshold_fraction * best_rate
    return {k: int(v >= threshold) for k, v in rates.items()}


# ---------------------------------------------------------------------------
# Improvement frequency rule (new default)
# ---------------------------------------------------------------------------
def compute_improvement_freqs(trials: TrialsRecord) -> dict[tuple[int, int], float]:
    """Fraction of M trials where delta < -IMPROVEMENT_EPS for each pair."""
    freqs = {}
    for pair in trials.results:
        n_improve = sum(1 for d in pair.deltas if d < -IMPROVEMENT_EPS)
        freqs[(pair.d_idx, pair.r_idx)] = n_improve / len(pair.deltas) if pair.deltas else 0.0
    return freqs


def compute_improvement_freq_labels(
    freqs: dict[tuple[int, int], float],
    threshold: float = 0.3,
) -> dict[tuple[int, int], int]:
    """Label=1 if improvement_freq >= threshold."""
    return {k: int(v >= threshold) for k, v in freqs.items()}


# ---------------------------------------------------------------------------
# Uninformative state detection
# ---------------------------------------------------------------------------
def is_uninformative(trials: TrialsRecord) -> bool:
    """Check if all deltas are zero (nothing to learn from this state)."""
    for pair in trials.results:
        for d in pair.deltas:
            if d != 0.0:
                return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_labels_for_log(
    state_trials: list[StateTrial],
    n_destroy: int,
    n_repair: int,
    rule: str = "improvement_freq",
    threshold: float = 0.3,
    filter_epoch_1: bool = True,
) -> tuple[list[dict], dict]:
    """Compute labels for all state-trial pairs from one log.

    Args:
        state_trials: paired state + trials records
        n_destroy: number of destroy operators
        n_repair: number of repair operators
        rule: "improvement_freq" or "sa_acceptance"
        threshold: threshold parameter for the chosen rule
        filter_epoch_1: if True, skip states from epoch_id == 1

    Returns:
        (rows, stats) where each row has: iter, epoch_id, pair_D_R (binary),
        improve_freq_D_R (float). stats has filtering counts.
    """
    rows: list[dict] = []
    n_epoch1 = 0
    n_uninformative = 0

    for st in state_trials:
        # Filter epoch 1
        if filter_epoch_1 and st.state.epoch_id == 1:
            n_epoch1 += 1
            continue

        # Filter uninformative
        if is_uninformative(st.trials):
            n_uninformative += 1
            continue

        # Compute improvement frequencies (always, for diagnostics columns)
        freqs = compute_improvement_freqs(st.trials)

        # Compute binary labels based on chosen rule
        if rule == "improvement_freq":
            labels = compute_improvement_freq_labels(freqs, threshold)
        elif rule == "sa_acceptance":
            rates = compute_acceptance_rates(st.trials)
            nonzero = sum(1 for r in rates.values() if r > 1e-10)
            if nonzero < 3:
                n_uninformative += 1
                continue
            labels = compute_sa_labels(rates, threshold)
        else:
            raise ValueError(f"Unknown labeling rule: {rule}")

        # Build row
        row: dict = {
            "iter": st.state.iter,
            "epoch_id": st.state.epoch_id,
        }
        for di in range(n_destroy):
            for ri in range(n_repair):
                key = (di, ri)
                row[f"pair_{di}_{ri}"] = labels.get(key, 0)
                row[f"improve_freq_{di}_{ri}"] = freqs.get(key, 0.0)
        rows.append(row)

    stats = {
        "n_epoch1_filtered": n_epoch1,
        "n_uninformative_filtered": n_uninformative,
        "n_labeled": len(rows),
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Legacy API (kept for backwards compatibility with existing tests)
# ---------------------------------------------------------------------------
def compute_binary_labels(
    rates: dict[tuple[int, int], float],
    threshold_fraction: float = 0.9,
) -> dict[tuple[int, int], int]:
    """Legacy wrapper for compute_sa_labels."""
    return compute_sa_labels(rates, threshold_fraction)


def compute_labels_for_pair(
    state_trials: list[StateTrial],
    n_destroy: int,
    n_repair: int,
    threshold_fraction: float = 0.9,
    min_nonzero_pairs: int = 3,
) -> list[dict]:
    """Legacy API — uses SA acceptance rule, no epoch filtering."""
    rows, _ = compute_labels_for_log(
        state_trials, n_destroy, n_repair,
        rule="sa_acceptance",
        threshold=threshold_fraction,
        filter_epoch_1=False,
    )
    return rows
