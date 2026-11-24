"""Step A — feature sanity diagnostic on cache_v2_dynamic/features.parquet.

Reports:
- df.describe() per feature
- nunique per feature (flag nunique==1)
- fraction-of-zeros per feature (flag >95%)
- NaN/inf counts (any nonzero is a bug)
- Deep-dive on the 5 suspiciously-low-importance features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("cache_v2_dynamic/features.parquet")
DEEP_DIVE = [
    "flexibility_horizon",
    "near_violation_count",
    "neighborhood_size_frac",
    "unassigned_count",
    "frac_committed",
]


def main():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("log_file", "iter")]
    print(f"Loaded {len(df)} rows × {len(feature_cols)} features\n")

    # Sidecar: confirms which batches/mode the parquet came from.
    sidecar = FEATURES_PATH.with_suffix(FEATURES_PATH.suffix + ".meta.json")
    if sidecar.exists():
        with open(sidecar) as f:
            meta = json.load(f)
        print(f"Sidecar: mode={meta.get('mode')}, batches={meta.get('batches')}")
        print(f"Feature count match: {len(meta.get('feature_names', [])) == len(feature_cols)}\n")

    # ---- describe ----
    print("=" * 90)
    print("PER-FEATURE describe()")
    print("=" * 90)
    desc = df[feature_cols].describe().T
    print(desc.to_string())
    print()

    # ---- nunique ----
    print("=" * 90)
    print("PER-FEATURE nunique (sorted ascending — constant features at top)")
    print("=" * 90)
    nu = df[feature_cols].nunique().sort_values()
    for name, n in nu.items():
        flag = " <-- CONSTANT" if n == 1 else ""
        print(f"  {name:32s}  {n:8d} unique values{flag}")
    print()

    # ---- fraction-of-zeros ----
    print("=" * 90)
    print("PER-FEATURE zero-fraction (sorted descending — mostly-zero at top)")
    print("=" * 90)
    zfrac = ((df[feature_cols] == 0).sum() / len(df)).sort_values(ascending=False)
    for name, frac in zfrac.items():
        flag = " <-- >95% ZERO" if frac > 0.95 else (" <-- >50% zero" if frac > 0.5 else "")
        print(f"  {name:32s}  {frac:8.4f} ({int(frac * len(df)):6d}/{len(df)}){flag}")
    print()

    # ---- NaN/inf ----
    print("=" * 90)
    print("NaN / inf check (any nonzero is a bug)")
    print("=" * 90)
    nan_total = int(df[feature_cols].isna().sum().sum())
    arr = df[feature_cols].to_numpy()
    inf_total = int((~np.isfinite(arr)).sum())
    print(f"  total NaN cells: {nan_total}")
    print(f"  total non-finite cells: {inf_total}")
    if nan_total > 0:
        print("  per-column NaN counts:")
        for name, n in df[feature_cols].isna().sum().items():
            if n > 0:
                print(f"    {name}: {n}")
    print()

    # ---- deep-dive ----
    print("=" * 90)
    print("DEEP DIVE on suspicious features")
    print("=" * 90)
    for name in DEEP_DIVE:
        s = df[name]
        n_zero = int((s == 0).sum())
        n_uniq = int(s.nunique())
        # Quantiles
        q = s.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]).to_dict()
        print(f"\n  {name}:")
        print(f"    n_unique  = {n_uniq}")
        print(f"    n_zero    = {n_zero} ({n_zero / len(s):.2%})")
        print(f"    min       = {q[0.0]:.6f}")
        print(f"    1%ile     = {q[0.01]:.6f}")
        print(f"    25%ile    = {q[0.25]:.6f}")
        print(f"    median    = {q[0.5]:.6f}")
        print(f"    75%ile    = {q[0.75]:.6f}")
        print(f"    99%ile    = {q[0.99]:.6f}")
        print(f"    max       = {q[1.0]:.6f}")
        print(f"    mean      = {s.mean():.6f}")
        print(f"    std       = {s.std():.6f}")
        # Per-log-file distribution: how many logs have this feature == constant?
        if "log_file" in df.columns:
            per_log_unique = df.groupby("log_file")[name].nunique()
            n_logs_const = int((per_log_unique == 1).sum())
            print(f"    logs with constant value (within log): "
                  f"{n_logs_const}/{len(per_log_unique)}")
            # Top-3 most common values
            vc = s.value_counts().head(3)
            print(f"    top-3 values: ", end="")
            print(", ".join(f"{v:.4f}({c} = {c / len(s):.1%})" for v, c in vc.items()))


if __name__ == "__main__":
    main()
