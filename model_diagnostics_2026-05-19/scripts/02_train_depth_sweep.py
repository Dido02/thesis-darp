#!/usr/bin/env python3
"""Phase A.5 — train depth-controlled single trees for 3 headline configs.

Configs trained:
  - 20f_t07        (best classifier — uses labels_t0.7.parquet)
  - 20f_regr       (best regressor — uses labels_t0.3_with_ei.parquet)
  - 20f_ei_log     (collapsed regressor — uses labels_t0.3_with_ei.parquet)

Depths trained: 2, 4, 8, 16, None (unrestricted, self-consistent baseline).

Total: 3 configs * 5 depths = 15 new model directories under
model_diagnostics_2026-05-19/depth_sweep_models/. Each dir gets the same
{rf_pair_*.joblib, training_report.json, meta.json} layout as ml_train/models_v6_*
so the Phase A enrichment script can be re-run against it.

Re-uses ml_train/darpml.training.train_pipeline so this stays consistent with
how the existing v6 tree models were trained.

Usage:
    python 02_train_depth_sweep.py
    python 02_train_depth_sweep.py --depths 2 4         # subset
    python 02_train_depth_sweep.py --configs 20f_t07    # subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml_train"))

from darpml.training import train_pipeline   # noqa: E402
from darpml.features import _REGISTRY        # noqa: E402

CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES = CACHE / "features.parquet"
OUT_ROOT = REPO / "model_diagnostics_2026-05-19" / "depth_sweep_models"

ALL_BATCHES = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]


def feature_names_for_batches(batches: list[str], mode: str = "dynamic") -> list[str]:
    bset = set(batches)
    return [
        name for name, entry in _REGISTRY.items()
        if entry.batch in bset and (entry.mode == "both" or entry.mode == mode)
    ]


# (config_label, model_kind, target_metric, log_transform, labels_filename)
CONFIGS = {
    "20f_t07": {
        "model_kind": "classifier",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.7.parquet",
    },
    "20f_regr": {
        "model_kind": "regressor",
        "target_metric": "improve_freq",
        "log_transform_label": False,
        "labels": "labels_t0.3_with_ei.parquet",
    },
    "20f_ei_log": {
        "model_kind": "regressor",
        "target_metric": "ei_pct",
        "log_transform_label": True,
        "labels": "labels_t0.3_with_ei.parquet",
    },
}


def _depth_label(d):
    return "none" if d is None else str(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS),
                    choices=list(CONFIGS),
                    help="Subset of CONFIGS to train.")
    ap.add_argument("--depths", nargs="*", type=str, default=["2", "4", "8", "16", "none"],
                    help="Depths to train (use 'none' for unrestricted).")
    ap.add_argument("--seed", type=int, default=42,
                    help="random_state passed to train_pipeline (default 42, "
                         "matches existing v6 tree models).")
    args = ap.parse_args()

    depths = [None if d == "none" else int(d) for d in args.depths]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")
    print(f"Configs: {args.configs}")
    print(f"Depths: {depths}")
    print()

    t_total = time.monotonic()
    summary = []
    for cfg_label in args.configs:
        cfg = CONFIGS[cfg_label]
        feat_names = feature_names_for_batches(ALL_BATCHES)
        labels_path = CACHE / cfg["labels"]
        if not labels_path.exists():
            print(f"  [skip] {cfg_label}: labels missing ({labels_path})")
            continue
        for depth in depths:
            tag = f"depth{_depth_label(depth)}_{cfg_label}_tree"
            out_dir = OUT_ROOT / tag
            t0 = time.monotonic()
            print(f"\n{'#' * 78}")
            print(f"# {tag}  ({cfg['model_kind']}, target={cfg['target_metric']}, "
                  f"depth={depth}, n_feat={len(feat_names)})")
            print(f"{'#' * 78}\n")
            out_dir.mkdir(parents=True, exist_ok=True)
            report = train_pipeline(
                features_path=FEATURES,
                labels_path=labels_path,
                output_dir=out_dir,
                n_estimators=100,                        # ignored under model_type=tree
                feature_names=feat_names,
                mode="dynamic",
                batches=ALL_BATCHES,
                model_kind=cfg["model_kind"],
                target_metric=cfg["target_metric"],
                log_transform_label=cfg["log_transform_label"],
                model_type="tree",
                tree_max_depth=depth,
                random_state=args.seed,
            )
            wall = time.monotonic() - t0
            summary.append((tag, depth,
                            report.median_f1_lift if cfg["model_kind"] == "classifier"
                            else report.median_f1_lift,  # holds mae_lift for regressors
                            report.n_pairs - report.n_degenerate_pairs,
                            wall))
            print(f"  -> wall={wall:.1f}s")

    print(f"\n{'=' * 78}")
    print(f"Depth sweep complete in {time.monotonic() - t_total:.1f}s")
    print(f"{'=' * 78}")
    print(f"{'tag':>40s} {'depth':>6s} {'median_lift':>11s} {'non_degen':>10s} {'wall_s':>7s}")
    for tag, d, ml, nd, w in summary:
        dl = _depth_label(d)
        print(f"{tag:>40s} {dl:>6s} {ml:11.4f} {nd:10d} {w:7.1f}")


if __name__ == "__main__":
    main()
