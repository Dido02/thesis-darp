#!/usr/bin/env python3
"""Full depth × top-K grid for trees AND forests on the classical (USED-9) ranking.

For each target in {t07, regr}:
  - 5 depths × 5 K = 25 cells
  - tree variant + forest variant per cell = 50 model dirs per target
  - 100 model dirs total (2 targets)

K is the top-K-AGGREGATED features by mean Gini importance over the 9 USED pairs
of the 20f forest reference. K=20 means "all features" (no restriction).
depth=None means "no depth cap".

Output: model_diagnostics_classical_2026-05-19/grid_models/<target>/<model_type>/
        depth{d}_K{k}/{rf_pair_*.joblib, ...}

Naming so the enrichment script can resolve the labels path:
  tree:   grid_<target>_tree_depth{d}_K{k}/
  forest: grid_<target>_forest_depth{d}_K{k}/

Usage:
    python 04_train_full_grid.py                       # full grid t07 + regr
    python 04_train_full_grid.py --targets t07         # t07 only
    python 04_train_full_grid.py --depths 4 8 none --K 5 8 20  # subset
    python 04_train_full_grid.py --skip-forest         # tree grid only (faster)
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
OUT_ROOT = REPO / "model_diagnostics_classical_2026-05-19" / "grid_models"
TBL = REPO / "model_diagnostics_classical_2026-05-19" / "tables"
TBL.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

ALL_BATCHES = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]
USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]

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


def _depth_label(d):
    return "none" if d is None else str(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--depths", nargs="*", default=["2", "4", "8", "16", "none"])
    ap.add_argument("--K", nargs="*", type=int, default=[3, 5, 8, 12, 20])
    ap.add_argument("--skip-tree", action="store_true")
    ap.add_argument("--skip-forest", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    depths = [None if d == "none" else int(d) for d in args.depths]

    print(f"Targets: {args.targets}")
    print(f"Depths:  {depths}")
    print(f"K values:{args.K}")
    print(f"Tree:    {not args.skip_tree}")
    print(f"Forest:  {not args.skip_forest}")
    print()

    summary = []
    t_total = time.monotonic()

    for target_label in args.targets:
        tgt = TARGETS[target_label]
        reference_dir = REPO / "ml_train" / tgt["reference"]
        labels_path = CACHE / tgt["labels"]
        if not reference_dir.exists() or not labels_path.exists():
            print(f"[skip] {target_label}: missing reference or labels")
            continue
        if reference_dir.name == "models_v3_full_20f_t02":
            n_features_full = 20  # for safety
        # pre-resolve features per K once
        feat_for_K = {K: aggregated_top_k(reference_dir, K, USED_PAIRS) for K in args.K}

        for K in args.K:
            features = feat_for_K[K]
            for depth in depths:
                d_lbl = _depth_label(depth)
                modes = []
                if not args.skip_tree:
                    modes.append("tree")
                if not args.skip_forest:
                    modes.append("forest")
                for model_type in modes:
                    tag = f"grid_{target_label}_{model_type}_depth{d_lbl}_K{K}"
                    out_dir = OUT_ROOT / tag
                    out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n--- {tag} ({len(features)} feat, depth={depth}, type={model_type}) ---")
                    t0 = time.monotonic()
                    report = train_pipeline(
                        features_path=FEATURES,
                        labels_path=labels_path,
                        output_dir=out_dir,
                        n_estimators=100,                  # ignored if model_type=tree
                        feature_names=features,
                        mode="dynamic",
                        batches=ALL_BATCHES,
                        model_kind=tgt["model_kind"],
                        target_metric=tgt["target_metric"],
                        log_transform_label=tgt["log_transform_label"],
                        model_type=model_type,
                        tree_max_depth=depth,              # also applied via max_depth for RF
                        random_state=args.seed,
                    )
                    # Stash grid coords in meta
                    meta_path = out_dir / "meta.json"
                    meta = json.loads(meta_path.read_text())
                    meta.update(grid_target=target_label, grid_K=K, grid_depth=d_lbl,
                                grid_model_type=model_type, grid_used_pairs=USED_PAIRS,
                                topK_features_ordered=features)
                    meta_path.write_text(json.dumps(meta, indent=2))
                    wall = time.monotonic() - t0
                    summary.append({
                        "target": target_label, "model_type": model_type,
                        "depth": d_lbl, "K": K, "n_features": len(features),
                        "wall_s": round(wall, 2), "tag": tag,
                    })
                    print(f"  -> wall={wall:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"Grid training complete in {time.monotonic() - t_total:.1f}s")
    print(f"{'=' * 78}")
    print(f"{'tag':>48s} {'wall_s':>7s}")
    for r in summary:
        print(f"{r['tag']:>48s} {r['wall_s']:>7.1f}")

    # Write a manifest CSV
    with open(TBL / "grid_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "model_type", "depth", "K",
                                          "n_features", "wall_s", "tag"])
        w.writeheader()
        for r in summary:
            w.writerow(r)


if __name__ == "__main__":
    main()
