#!/usr/bin/env python3
"""Phase B.5 — feature-combination ablation.

For each target (t07 classifier, regr regressor), pick the top-K features by
mean Gini importance from the **20-feature forest** (highest-quality reference
for "which features matter"), then train a single-tree variant restricted to
just those features. K in {3, 5, 8, 12}.

Outputs to model_diagnostics_2026-05-19/feat_ablation_models/:
  topK{3,5,8,12}_{t07,regr}_tree/   (8 new model dirs)

Each dir gets the same {rf_pair_*.joblib, training_report.json, meta.json}
layout as ml_train/models_v6_*, so the Phase A enrichment script handles them.

We use the forest's importance ranking (not the tree's) as the reference
because the forest is less noisy — single-tree feature ranking is unstable on
small samples. The ablation answers: "is the model essentially using only the
top-K features even when given 20?"

Usage:
    python 04_train_feat_ablation.py
    python 04_train_feat_ablation.py --K 3 8        # subset of K values
    python 04_train_feat_ablation.py --targets t07  # subset of targets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml_train"))

from darpml.training import train_pipeline   # noqa: E402

CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES = CACHE / "features.parquet"
OUT_ROOT = REPO / "model_diagnostics_2026-05-19" / "feat_ablation_models"

ALL_BATCHES = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]

# Each target maps to (reference forest dir for importance ranking,
# model_kind, target_metric, log_transform, labels_filename).
TARGETS = {
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


def top_k_features_from_forest(reference_dir: Path, k: int) -> list[str]:
    """Return the top-k feature names by mean Gini importance, in importance order."""
    report = json.loads((reference_dir / "training_report.json").read_text())
    imps = report["feature_importances"]  # dict name -> importance
    ranked = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    ap.add_argument("--K", nargs="*", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")
    print(f"Targets: {args.targets}")
    print(f"K values: {args.K}")
    print()

    t_total = time.monotonic()
    summary = []
    for target_label in args.targets:
        tgt = TARGETS[target_label]
        reference_dir = REPO / "ml_train" / tgt["reference"]
        labels_path = CACHE / tgt["labels"]
        if not reference_dir.exists():
            print(f"  [skip] {target_label}: reference dir missing ({reference_dir})")
            continue
        if not labels_path.exists():
            print(f"  [skip] {target_label}: labels missing ({labels_path})")
            continue

        for K in args.K:
            feat_names = top_k_features_from_forest(reference_dir, K)
            tag = f"topK{K}_{target_label}_tree"
            out_dir = OUT_ROOT / tag
            print(f"\n{'#' * 78}")
            print(f"# {tag}  ({tgt['model_kind']}, top-{K} features from "
                  f"{tgt['reference']})")
            print(f"#   features used: {feat_names}")
            print(f"{'#' * 78}\n")
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            report = train_pipeline(
                features_path=FEATURES,
                labels_path=labels_path,
                output_dir=out_dir,
                n_estimators=100,
                feature_names=feat_names,
                mode="dynamic",
                batches=ALL_BATCHES,
                model_kind=tgt["model_kind"],
                target_metric=tgt["target_metric"],
                log_transform_label=tgt["log_transform_label"],
                model_type="tree",
                tree_max_depth=None,                # unrestricted; the feature count is the lever
                random_state=args.seed,
            )
            wall = time.monotonic() - t0
            # Stash the top-K feature list inside meta.json for later reference
            meta_path = out_dir / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["topK_K"] = K
            meta["topK_reference"] = tgt["reference"]
            meta["topK_features_ordered"] = feat_names
            meta_path.write_text(json.dumps(meta, indent=2))

            summary.append((tag, K, report.median_f1_lift,
                            report.n_pairs - report.n_degenerate_pairs, wall))
            print(f"  -> wall={wall:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"Feature-ablation training complete in {time.monotonic() - t_total:.1f}s")
    print(f"{'=' * 78}")
    print(f"{'tag':>32s} {'K':>3s} {'median_lift':>11s} {'non_degen':>10s} {'wall_s':>7s}")
    for tag, K, ml, nd, w in summary:
        print(f"{tag:>32s} {K:3d} {ml:11.4f} {nd:10d} {w:7.1f}")


if __name__ == "__main__":
    main()
