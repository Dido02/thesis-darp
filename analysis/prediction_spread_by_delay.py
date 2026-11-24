#!/usr/bin/env python3
"""Run the 9 deployed (classical) ONNX forests against cached trail-log states
and report the spread of P(class=1) per pair, broken out by delay tier.

This bypasses the solver entirely — uses features.parquet (already extracted
from the trail logs at training time) so we get prediction distributions
without re-running anything.

Output: prints summary table + writes per-pair-per-delay percentile dump.

Usage:
    python analysis/prediction_spread_by_delay.py
    python analysis/prediction_spread_by_delay.py --model-dir <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "ml_train" / "models_v3_full_20f_t07" / "onnx"
FEATURES_PARQUET = REPO / "ml_train" / "cache_v3_full" / "features.parquet"

# The 9 classical pairs the benchmark actually uses
CLASSICAL = [
    ("random", "greedy"), ("random", "regret2"), ("random", "regret3"),
    ("worst",  "greedy"), ("worst",  "regret2"), ("worst",  "regret3"),
    ("shaw",   "greedy"), ("shaw",   "regret2"), ("shaw",   "regret3"),
]


def load_manifest(model_dir: Path) -> list[str]:
    with open(model_dir / "feature_manifest.json") as f:
        m = json.load(f)
    return [f["name"] for f in m["features"]]


def load_session(model_dir: Path, destroy: str, repair: str) -> ort.InferenceSession:
    onnx_path = model_dir / f"{destroy}_{repair}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing model: {onnx_path}")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def run_predictions(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Run one ONNX session on the full feature matrix, return P(class=1)."""
    input_name = session.get_inputs()[0].name
    # ONNX expects float32
    Xf = X.astype(np.float32)

    # Find probabilities output by name
    out_names = [o.name for o in session.get_outputs()]
    if "probabilities" in out_names:
        prob_idx = out_names.index("probabilities")
    else:
        prob_idx = 1 if len(out_names) > 1 else 0
    target_out = out_names[prob_idx]

    out = session.run([target_out], {input_name: Xf})
    probs = out[0]

    # sklearn ONNX classifiers output probabilities as a list of dicts in some
    # exports and as a 2D array in others. Handle both.
    if isinstance(probs, list):
        # list of {0: P0, 1: P1} per row
        return np.array([float(d[1]) for d in probs], dtype=np.float64)
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1].astype(np.float64)
    return probs.flatten().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL),
                    help="ONNX model dir (default: models_v3_full_20f_t07/onnx)")
    ap.add_argument("--out-csv", default=str(REPO / "analysis" / "prediction_spread_by_delay.csv"),
                    help="Write per-pair-per-delay summary stats to this CSV")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    print(f"Loading features from {FEATURES_PARQUET}")
    df = pd.read_parquet(FEATURES_PARQUET)
    df["delay"] = df["log_file"].str.extract(r"delay(\d+)_min")[0]

    print(f"  total rows: {len(df):,}")
    print(f"  rows by delay: {dict(df['delay'].value_counts().sort_index())}")

    feature_order = load_manifest(model_dir)
    print(f"  features in manifest order: {len(feature_order)}")

    # Build feature matrix in manifest order
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise SystemExit(f"features.parquet missing manifest features: {missing}")
    X = df[feature_order].to_numpy()

    # Subset rows by delay tier
    delays = sorted(df["delay"].dropna().unique())
    print(f"\nRunning {len(CLASSICAL)} forests on each of {len(delays)} delay tiers...")

    # Results: per (pair, delay) → array of predictions
    results = {}
    for d_name, r_name in CLASSICAL:
        print(f"  pair ({d_name}, {r_name}) ...")
        session = load_session(model_dir, d_name, r_name)
        preds_all = run_predictions(session, X)
        for delay in delays:
            mask = (df["delay"] == delay).to_numpy()
            results[(d_name, r_name, delay)] = preds_all[mask]

    # Build summary table
    rows = []
    for (d_name, r_name, delay), preds in results.items():
        rows.append({
            "destroy": d_name,
            "repair": r_name,
            "delay": delay,
            "n_states": len(preds),
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds, ddof=1)),
            "p10": float(np.percentile(preds, 10)),
            "p25": float(np.percentile(preds, 25)),
            "p50": float(np.percentile(preds, 50)),
            "p75": float(np.percentile(preds, 75)),
            "p90": float(np.percentile(preds, 90)),
            "p95": float(np.percentile(preds, 95)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "frac_above_0.5": float(np.mean(preds > 0.5)),
            "frac_above_0.7": float(np.mean(preds > 0.7)),
        })
    summary = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Per-delay: mean prediction per pair (the central question)
    print("\n=== Mean predicted P(class=1) per pair, per delay tier ===")
    print(f"  Model: {model_dir.name}")
    print()
    header = f"{'pair':<22}"
    for delay in delays:
        header += f"  d{delay:<6}"
    print(header)
    print("-" * (22 + 9 * len(delays)))
    for d_name, r_name in CLASSICAL:
        line = f"  {d_name:>7} + {r_name:<8} "
        for delay in delays:
            preds = results[(d_name, r_name, delay)]
            line += f"  {preds.mean():>6.3f}"
        print(line)

    # Standard deviation per pair per delay
    print("\n=== Std of predicted P(class=1) per pair, per delay tier ===")
    print(f"  (how much does the model's opinion change across states within a tier?)")
    print()
    print(header)
    print("-" * (22 + 9 * len(delays)))
    for d_name, r_name in CLASSICAL:
        line = f"  {d_name:>7} + {r_name:<8} "
        for delay in delays:
            preds = results[(d_name, r_name, delay)]
            line += f"  {preds.std(ddof=1):>6.3f}"
        print(line)

    # Spread of mean predictions ACROSS the 9 pairs, per delay
    print("\n=== How separable are the 9 pairs by mean prediction, per delay? ===")
    print(f"  (large spread = model can discriminate; small spread = pairs look alike)")
    print()
    print(f"  {'delay':<8} {'min mean':>10} {'max mean':>10} {'spread':>10} {'std of means':>15}")
    for delay in delays:
        means = [results[(d, r, delay)].mean() for (d, r) in CLASSICAL]
        print(f"  d{delay:<7} {min(means):>10.3f} {max(means):>10.3f} {max(means)-min(means):>10.3f} {np.std(means, ddof=1):>15.3f}")

    # Bench-relevant: what fraction of states have a "confident" prediction (P>0.7)?
    print("\n=== Frac of states where the forest predicts P>0.7 (a 'confident yes') ===")
    print()
    print(header)
    print("-" * (22 + 9 * len(delays)))
    for d_name, r_name in CLASSICAL:
        line = f"  {d_name:>7} + {r_name:<8} "
        for delay in delays:
            preds = results[(d_name, r_name, delay)]
            line += f"  {np.mean(preds > 0.7):>6.3f}"
        print(line)


if __name__ == "__main__":
    main()
