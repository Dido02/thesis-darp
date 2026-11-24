#!/usr/bin/env python3
"""Validation script for the 'why ML wins on d05/d10 but not d03' argument.

For each delay tier (03 / 05 / 10), this re-derives every number used in the
thesis claim, citing the exact file and operation each came from.

  STEP 1: label prevalence per delay  (from labels_t0.7.parquet)
  STEP 2: forest predictions per delay (from features.parquet + ONNX inference)
  STEP 3: confidence-region counts at multiple thresholds (not just p90)
  STEP 4: softmax(p/0.1) concentration per state — how peaky is the actual sampler?
  STEP 5: bench outcome cross-check  (from BENCH_FINAL §4.3 win counts)

Run:
  /Users/didodimitrov/Documents/thesis-darp/.venv/bin/python \\
    analysis/validate_delay_signal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "ml_train" / "cache_v3_full" / "labels_t0.7.parquet"
FEATURES = REPO / "ml_train" / "cache_v3_full" / "features.parquet"
MODEL_DIR = REPO / "ml_train" / "models_v3_full_20f_t07" / "onnx"

CLASSICAL = [
    ("random", "greedy"), ("random", "regret2"), ("random", "regret3"),
    ("worst",  "greedy"), ("worst",  "regret2"), ("worst",  "regret3"),
    ("shaw",   "greedy"), ("shaw",   "regret2"), ("shaw",   "regret3"),
]

# Map (destroy, repair) -> column name in labels_t0.7.parquet
# Order in parquet: pair_0_0..0_2 = random, pair_1_0..1_2 = worst,
#                   pair_2_0..2_2 = shaw   (destroy indices 0..2)
# repair indices 0=greedy, 1=regret2, 2=regret3.
PAIR_TO_COL = {
    ("random", "greedy"):  "pair_0_0",
    ("random", "regret2"): "pair_0_1",
    ("random", "regret3"): "pair_0_2",
    ("worst",  "greedy"):  "pair_1_0",
    ("worst",  "regret2"): "pair_1_1",
    ("worst",  "regret3"): "pair_1_2",
    ("shaw",   "greedy"):  "pair_2_0",
    ("shaw",   "regret2"): "pair_2_1",
    ("shaw",   "regret3"): "pair_2_2",
}


def step1_prevalence():
    print("=" * 78)
    print("STEP 1 — label prevalence per delay tier (the 'is the problem degenerate?' check)")
    print("=" * 78)
    print(f"Source: {LABELS.relative_to(REPO)}")
    print("Operation: for each delay tier, mean of pair_d_r binary labels across the 9 classical pairs")
    print()

    lab = pd.read_parquet(LABELS)
    lab["delay"] = lab["log_file"].str.extract(r"delay(\d+)_min")[0]
    delays = sorted(lab["delay"].dropna().unique())

    print(f"  Total labelled states: {len(lab):,}")
    print(f"  States per delay: {dict(lab['delay'].value_counts().sort_index())}")
    print()

    # Per-pair per-delay positive rates
    print(f"  Per-pair positive rate (= P(label = 1)):")
    print(f'  {"pair":<22}' + ''.join(f'  d{d:<6}' for d in delays))
    for (d_name, r_name) in CLASSICAL:
        col = PAIR_TO_COL[(d_name, r_name)]
        line = f'  {d_name:>7} + {r_name:<8} '
        for delay in delays:
            sub = lab[lab.delay == delay]
            rate = sub[col].mean()
            line += f'  {rate:>6.3f}'
        print(line)
    print()
    print(f"  Mean over the 9 pairs (the 'rollup' prevalence):")
    for delay in delays:
        sub = lab[lab.delay == delay]
        rates = [sub[PAIR_TO_COL[(d, r)]].mean() for (d, r) in CLASSICAL]
        print(f"    d{delay}: mean = {np.mean(rates):.3f}  "
              f"min = {min(rates):.3f}  max = {max(rates):.3f}  spread = {max(rates) - min(rates):.3f}")
    print()


def load_features_and_run_forests():
    """Return (df_features, predictions_by_pair, delays)."""
    df = pd.read_parquet(FEATURES)
    df["delay"] = df["log_file"].str.extract(r"delay(\d+)_min")[0]

    with open(MODEL_DIR / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_order = [f["name"] for f in manifest["features"]]
    X = df[feature_order].to_numpy(dtype=np.float32)

    preds = {}
    for (d_name, r_name) in CLASSICAL:
        sess = ort.InferenceSession(
            str(MODEL_DIR / f"{d_name}_{r_name}.onnx"),
            providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]
        prob_idx = out_names.index("probabilities") if "probabilities" in out_names else 1
        out = sess.run([out_names[prob_idx]], {in_name: X})[0]
        if isinstance(out, list):
            p = np.array([float(d[1]) for d in out])
        else:
            arr = np.asarray(out)
            p = arr[:, 1].astype(np.float64) if arr.ndim == 2 else arr.flatten().astype(np.float64)
        preds[(d_name, r_name)] = p

    delays = sorted(df["delay"].dropna().unique())
    return df, preds, delays


def step2_predictions(df, preds, delays):
    print("=" * 78)
    print("STEP 2 — t07 forest prediction percentiles per delay")
    print("=" * 78)
    print(f"Source: {FEATURES.relative_to(REPO)} (8,992 cached states) +")
    print(f"        {MODEL_DIR.relative_to(REPO)}/<pair>.onnx (9 forests)")
    print("Operation: for each forest × each delay tier, compute p50, p90, p99 of P(class=1) over the in-tier states")
    print()

    for pct, label in [(50, "p50 (median)"), (90, "p90"), (99, "p99")]:
        print(f"  {label} — value below which {pct}% of states' predictions lie:")
        print(f'  {"pair":<22}' + ''.join(f'  d{d:<6}' for d in delays))
        for (d_name, r_name) in CLASSICAL:
            line = f'  {d_name:>7} + {r_name:<8} '
            for delay in delays:
                mask = (df["delay"] == delay).to_numpy()
                p = preds[(d_name, r_name)][mask]
                line += f'  {np.percentile(p, pct):>6.3f}'
            print(line)
        print()


def step3_confidence_counts(df, preds, delays):
    print("=" * 78)
    print("STEP 3 — confidence-region counts at MULTIPLE thresholds (not just p90)")
    print("=" * 78)
    print("Question: 'for how many pairs is the forest confident in at least X% of states?'")
    print("Operation: for each threshold T_prob, count states where P > T_prob; divide by n_states.")
    print("           Then ask: how many of the 9 pairs have ≥10% of states with P > T_prob?")
    print()
    THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
    FRAC = 0.10  # at least 10% of states in that region
    for delay in delays:
        mask = (df["delay"] == delay).to_numpy()
        n = int(mask.sum())
        print(f"  delay {delay}  (n = {n} states)")
        for T_prob in THRESHOLDS:
            n_confident_pairs = 0
            pairs_with_region = []
            for (d_name, r_name) in CLASSICAL:
                p = preds[(d_name, r_name)][mask]
                frac_above = float(np.mean(p > T_prob))
                if frac_above >= FRAC:
                    n_confident_pairs += 1
                    pairs_with_region.append(f"{d_name[0]}+{r_name[:3]}({frac_above*100:.0f}%)")
            print(f"    P > {T_prob}:  {n_confident_pairs}/9 pairs have ≥10% of states in the region    "
                  f"[{', '.join(pairs_with_region) if pairs_with_region else 'none'}]")
        print()


def step4_softmax_concentration(df, preds, delays):
    print("=" * 78)
    print("STEP 4 — softmax(p / 0.1) concentration: how peaky is the actual sampler per state?")
    print("=" * 78)
    print("Per state, build the 9-pair prediction vector → apply softmax(p / T) with T=0.10 →")
    print("compute (a) top-1 probability, (b) Shannon entropy, (c) effective number of pairs = 2^entropy.")
    print("Random selector: top-1 = 1/9 ≈ 0.111, entropy = log2(9) ≈ 3.17, effective pairs = 9.")
    print()

    T = 0.10
    print(f'  delay   |  mean top1  median top1  |  mean entropy   |  mean eff. pairs  |  frac states "near-uniform" (eff>=6)')
    for delay in delays:
        mask = (df["delay"] == delay).to_numpy()
        n = int(mask.sum())
        P = np.stack([preds[k][mask] for k in CLASSICAL], axis=1)  # shape [n, 9]
        # softmax with temperature
        P_T = P / T
        P_T -= P_T.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(P_T)
        w = e / e.sum(axis=1, keepdims=True)  # shape [n, 9]
        top1 = w.max(axis=1)
        ent = -np.sum(w * np.log2(np.maximum(w, 1e-30)), axis=1)
        eff = 2.0 ** ent
        print(f'  d{delay:<6} |  {top1.mean():>7.3f}    {np.median(top1):>9.3f}   |  {ent.mean():>10.3f}     |  {eff.mean():>10.3f}        |  {(eff >= 6).mean()*100:>5.1f}% of states')
    print()
    print("  Reading:")
    print("    'effective pairs' ≈ how many pairs the softmax effectively spreads probability over.")
    print("    Low (e.g. 1.5) = sampler basically deterministic on one pair (= strong signal).")
    print("    High (e.g. 6-9) = sampler near-uniform across many pairs (= behaves like random).")


def step5_bench_crosscheck():
    print("=" * 78)
    print("STEP 5 — bench-outcome cross-check (independent of the prediction analysis)")
    print("=" * 78)
    print(f"Source: BENCH_FINAL_2026-05-18.md §4.3, §4.5, §4.10")
    print("These tables report Δ% vs random per cell, with ✓ marking ML wins.")
    print("I counted the ✓ marks per delay for the 20f_t07/T10 column.")
    print()
    print('  delay   | t07/T10 wins / 16   | paired Wilcoxon p   | Holm-sig over 9 tests')
    print('  d03     |  6 / 16             | n.s. (~0.45)        | no')
    print('  d05     | 11 / 16             | 0.003               | yes')
    print('  d10     | 11 / 16             | 0.006               | yes')
    print()
    print("  (Win = ML mean obj < random mean obj for that (instance, budget) cell, 5-seed averaged.)")


def main():
    step1_prevalence()
    df, preds, delays = load_features_and_run_forests()
    step2_predictions(df, preds, delays)
    step3_confidence_counts(df, preds, delays)
    step4_softmax_concentration(df, preds, delays)
    step5_bench_crosscheck()


if __name__ == "__main__":
    main()
