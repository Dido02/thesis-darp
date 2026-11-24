#!/usr/bin/env python3
"""Phase classical-B.5 — train top-K trees on USED 9 pairs only, two ways:

  (A) AGGREGATED ranking: pick top-K features by MEAN Gini importance across
      the 9 USED pair models in the 20f forest. One feature set per K.
  (B) UNION ranking:      for each USED pair, pick that pair's top-K by Gini.
      Take the UNION of those K-element sets across the 9 USED pairs. The
      resulting feature set may be larger than K (size in [K, min(20, 9K)]).
      One union feature set per K.

For each (target, mode, K), we train a single-tree variant of all 20 pair
models on the chosen feature subset and emit:
  feat_ablation_models_classical/topK{K}_{agg|union}_{t07|regr}_classical_tree/

Also emits tables/topK_classical_union_features.csv listing which features
are in each union set.

Targets: t07 (classifier, threshold-0.7) and regr (regressor, improve_freq).

USED 9 pairs:  destroy in {random, worst, shaw} x repair in {greedy, regret2, regret3}
              = pair_0_0, pair_0_1, pair_0_2, pair_1_0, pair_1_1, pair_1_2,
                pair_2_0, pair_2_1, pair_2_2

Usage:
    python 01_train_classical_topK.py                  # all targets, all K
    python 01_train_classical_topK.py --K 8            # only K=8
    python 01_train_classical_topK.py --targets t07    # only t07
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml_train"))

from darpml.training import train_pipeline   # noqa: E402

CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES = CACHE / "features.parquet"
CLASSICAL_ROOT = REPO / "model_diagnostics_classical_2026-05-19"
OUT_ROOT = CLASSICAL_ROOT / "feat_ablation_models_classical"
TBL = CLASSICAL_ROOT / "tables"

ALL_BATCHES = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]

# USED pairs: destroy in {0,1,2}, repair in {0,1,2} = 9 of 20
USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]

# Each target -> (reference forest dir, model_kind, target_metric, log_transform, labels_filename)
TARGETS = {
    "t02": {
        "reference": "models_v3_full_20f_t02",
        "model_kind": "classifier",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.2.parquet",
    },
    "t03": {
        "reference": "models_v3_full_20f_t03",
        "model_kind": "classifier",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.3.parquet",
    },
    "t05": {
        "reference": "models_v3_full_20f_t05",
        "model_kind": "classifier",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.5.parquet",
    },
    "t07": {
        "reference": "models_v3_full_20f_t07",
        "model_kind": "classifier",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.7.parquet",
    },
    "regr": {
        "reference": "models_v3_full_20f_regr",
        "model_kind": "regressor",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.3_with_ei.parquet",
    },
}


def per_pair_top_k(reference_dir: Path, K: int, used_pairs: list[str]) -> dict[str, list[str]]:
    """For each used pair, return its top-K feature names (by Gini, descending)."""
    out = {}
    for pair in used_pairs:
        clf = joblib.load(reference_dir / f"rf_{pair}.joblib")
        meta = json.loads((reference_dir / "meta.json").read_text())
        feat_names = meta["feature_names"]
        order = np.argsort(-clf.feature_importances_)
        out[pair] = [feat_names[i] for i in order[:K]]
    return out


def aggregated_top_k(reference_dir: Path, K: int, used_pairs: list[str]) -> list[str]:
    """Top-K by mean Gini importance across the used pairs (descending)."""
    meta = json.loads((reference_dir / "meta.json").read_text())
    feat_names = meta["feature_names"]
    imps = np.zeros(len(feat_names))
    for pair in used_pairs:
        clf = joblib.load(reference_dir / f"rf_{pair}.joblib")
        imps += clf.feature_importances_
    imps /= len(used_pairs)
    order = np.argsort(-imps)
    return [feat_names[i] for i in order[:K]]


def union_top_k(per_pair: dict[str, list[str]]) -> list[str]:
    """Sort the union by frequency-in-top-K (descending), then alphabetically."""
    from collections import Counter
    cnt = Counter()
    for pair_list in per_pair.values():
        cnt.update(pair_list)
    return sorted(cnt, key=lambda f: (-cnt[f], f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    ap.add_argument("--K", nargs="*", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TBL.mkdir(parents=True, exist_ok=True)

    print(f"USED 9 pairs: {USED_PAIRS}")
    print(f"Targets: {args.targets}")
    print(f"K values: {args.K}")
    print()

    union_rows = []
    aggregate_rows = []

    t_total = time.monotonic()
    summary = []
    for target_label in args.targets:
        tgt = TARGETS[target_label]
        reference_dir = REPO / "ml_train" / tgt["reference"]
        labels_path = CACHE / tgt["labels"]
        if not reference_dir.exists() or not labels_path.exists():
            print(f"  [skip] {target_label}: missing reference or labels")
            continue

        for K in args.K:
            # ------------- AGGREGATED -------------
            agg_features = aggregated_top_k(reference_dir, K, USED_PAIRS)
            agg_tag = f"topK{K}_agg_{target_label}_classical_tree"
            agg_dir = OUT_ROOT / agg_tag
            agg_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {agg_tag} ---")
            print(f"  features ({len(agg_features)}): {agg_features}")
            t0 = time.monotonic()
            report = train_pipeline(
                features_path=FEATURES,
                labels_path=labels_path,
                output_dir=agg_dir,
                n_estimators=100,
                feature_names=agg_features,
                mode="dynamic",
                batches=ALL_BATCHES,
                model_kind=tgt["model_kind"],
                target_metric=tgt["target_metric"],
                log_transform_label=tgt["log_transform_label"],
                model_type="tree",
                tree_max_depth=None,
                random_state=args.seed,
            )
            meta_path = agg_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(K=K, mode_classical="aggregated", used_pairs=USED_PAIRS,
                        reference_dir=str(reference_dir.relative_to(REPO)))
            meta_path.write_text(json.dumps(meta, indent=2))

            wall = time.monotonic() - t0
            summary.append((agg_tag, K, "agg", len(agg_features),
                            report.median_f1_lift,
                            report.n_pairs - report.n_degenerate_pairs, wall))
            aggregate_rows.append({
                "target": target_label, "K": K, "n_features": len(agg_features),
                "features": ", ".join(agg_features),
                "median_lift_train_test": "see enriched JSON",
            })

            # ------------- UNION -------------
            per_pair = per_pair_top_k(reference_dir, K, USED_PAIRS)
            union_features = union_top_k(per_pair)
            union_tag = f"topK{K}_union_{target_label}_classical_tree"
            union_dir = OUT_ROOT / union_tag
            union_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- {union_tag} ---")
            print(f"  per-pair top-{K} (USED 9):")
            for p in USED_PAIRS:
                print(f"    {p}: {per_pair[p]}")
            print(f"  UNION ({len(union_features)} features): {union_features}")
            t0 = time.monotonic()
            report = train_pipeline(
                features_path=FEATURES,
                labels_path=labels_path,
                output_dir=union_dir,
                n_estimators=100,
                feature_names=union_features,
                mode="dynamic",
                batches=ALL_BATCHES,
                model_kind=tgt["model_kind"],
                target_metric=tgt["target_metric"],
                log_transform_label=tgt["log_transform_label"],
                model_type="tree",
                tree_max_depth=None,
                random_state=args.seed,
            )
            meta_path = union_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(K=K, mode_classical="union", used_pairs=USED_PAIRS,
                        reference_dir=str(reference_dir.relative_to(REPO)),
                        per_pair_top_k=per_pair)
            meta_path.write_text(json.dumps(meta, indent=2))

            wall = time.monotonic() - t0
            summary.append((union_tag, K, "union", len(union_features),
                            report.median_f1_lift,
                            report.n_pairs - report.n_degenerate_pairs, wall))
            for rank, feat in enumerate(union_features, 1):
                # Count how many pairs include this feature in their top-K
                in_n_pairs = sum(1 for plist in per_pair.values() if feat in plist)
                union_rows.append({
                    "target": target_label, "K": K, "rank": rank,
                    "feature": feat, "in_n_used_pairs": in_n_pairs,
                    "total_union_size": len(union_features),
                })

    # Write union features table
    with open(TBL / "topK_classical_union_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "K", "rank", "feature",
                                          "in_n_used_pairs", "total_union_size"])
        w.writeheader()
        for r in union_rows:
            w.writerow(r)
    # Write aggregate features table
    with open(TBL / "topK_classical_aggregated_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "K", "n_features", "features",
                                          "median_lift_train_test"])
        w.writeheader()
        for r in aggregate_rows:
            w.writerow(r)

    print(f"\n{'=' * 78}")
    print(f"Classical top-K training complete in {time.monotonic() - t_total:.1f}s")
    print(f"{'=' * 78}")
    print(f"{'tag':>40s} {'K':>3s} {'mode':>6s} {'#feat':>6s} {'med_lift':>10s} {'non_deg':>8s} {'wall_s':>7s}")
    for tag, K, mode, n_feat, ml, nd, w in summary:
        print(f"{tag:>40s} {K:3d} {mode:>6s} {n_feat:6d} {ml:10.4f} {nd:8d} {w:7.1f}")


if __name__ == "__main__":
    main()
