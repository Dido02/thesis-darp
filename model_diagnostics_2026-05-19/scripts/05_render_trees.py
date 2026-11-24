#!/usr/bin/env python3
"""Phase C — render the curated set of trees with annotations.

For each (model_dir, pair_name) in CURATED, emit to figures/trees/:
  <name>.txt            — sklearn export_text ASCII
  <name>.dot            — graphviz dot source (renderable later with `dot -Tpng`)
  <name>.png            — matplotlib plot_tree render (always works)
  <name>_top3.png       — same but truncated to max_depth=3 for readability
  <name>.svg            — dtreeviz render (only if system `dot` is on PATH)
  <name>.yaml           — sidecar with depth, n_leaves, top splits, leaf coverage

Curated set: 4 groups (≈20 trees):
  Group 1 — anatomy of headline variants:
    20f_t07_tree pair_0_0, pair_4_3
    20f_regr_tree pair_0_0, pair_4_3
    20f_ei_log_tree pair_0_0
  Group 2 — feature-count effect (same pair across 5f/13f/20f):
    {5f,13f,20f}_t07_tree pair_0_0
    {5f,13f,20f}_regr_tree pair_0_0
  Group 3 — depth effect:
    depth2_20f_t07_tree pair_0_0
    depth4_20f_t07_tree pair_0_0
    depth8_20f_t07_tree pair_0_0
    depth16_20f_t07_tree pair_0_0
  Group 4 — forest vs single tree (estimator_0 of the 100-tree forest):
    models_v3_full_20f_t07/rf_pair_0_0.joblib.estimators_[0]
    models_v3_full_20f_regr/rf_pair_0_0.joblib.estimators_[0]

Usage:
    python 05_render_trees.py
    python 05_render_trees.py --skip-dtreeviz  # skip the SVG step
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    export_graphviz, export_text, plot_tree,
)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
FEATURES = CACHE / "features.parquet"
FIG = REPO / "model_diagnostics_2026-05-19" / "figures" / "trees"
FIG.mkdir(parents=True, exist_ok=True)

# Each entry: (display_name, model_dir relative to REPO, pair_name, labels_parquet, target_metric, forest_estimator_idx_or_None)
CURATED = [
    # Group 1 — headline variants
    ("20f_t07_tree__pair_0_0",
     "ml_train/models_v6_full_20f_t07_tree", "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("20f_t07_tree__pair_4_3",
     "ml_train/models_v6_full_20f_t07_tree", "pair_4_3", "labels_t0.7.parquet", "improve_freq", None),
    ("20f_regr_tree__pair_0_0",
     "ml_train/models_v6_full_20f_regr_tree", "pair_0_0", "labels_t0.3_with_ei.parquet", "improve_freq", None),
    ("20f_regr_tree__pair_4_3",
     "ml_train/models_v6_full_20f_regr_tree", "pair_4_3", "labels_t0.3_with_ei.parquet", "improve_freq", None),
    ("20f_ei_log_tree__pair_0_0",
     "ml_train/models_v6_full_20f_ei_log_tree", "pair_0_0", "labels_t0.3_with_ei.parquet", "ei_pct", None),
    # Group 2 — feature-count effect
    ("5f_t07_tree__pair_0_0",
     "ml_train/models_v6_full_5f_t07_tree", "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("13f_t07_tree__pair_0_0",
     "ml_train/models_v6_full_13f_t07_tree", "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("5f_regr_tree__pair_0_0",
     "ml_train/models_v6_full_5f_regr_tree", "pair_0_0", "labels_t0.3_with_ei.parquet", "improve_freq", None),
    ("13f_regr_tree__pair_0_0",
     "ml_train/models_v6_full_13f_regr_tree", "pair_0_0", "labels_t0.3_with_ei.parquet", "improve_freq", None),
    # Group 3 — depth-controlled trees
    ("depth2_20f_t07_tree__pair_0_0",
     "model_diagnostics_2026-05-19/depth_sweep_models/depth2_20f_t07_tree",
     "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("depth4_20f_t07_tree__pair_0_0",
     "model_diagnostics_2026-05-19/depth_sweep_models/depth4_20f_t07_tree",
     "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("depth8_20f_t07_tree__pair_0_0",
     "model_diagnostics_2026-05-19/depth_sweep_models/depth8_20f_t07_tree",
     "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    ("depth16_20f_t07_tree__pair_0_0",
     "model_diagnostics_2026-05-19/depth_sweep_models/depth16_20f_t07_tree",
     "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
    # Group 4 — forest estimator vs single tree
    ("forest20f_t07__pair_0_0__estimator0",
     "ml_train/models_v3_full_20f_t07", "pair_0_0", "labels_t0.7.parquet", "improve_freq", 0),
    ("forest20f_regr__pair_0_0__estimator0",
     "ml_train/models_v3_full_20f_regr", "pair_0_0", "labels_t0.3_with_ei.parquet", "improve_freq", 0),
    # Bonus: top-K ablation winner for t07
    ("topK8_t07_tree__pair_0_0",
     "model_diagnostics_2026-05-19/feat_ablation_models/topK8_t07_tree",
     "pair_0_0", "labels_t0.7.parquet", "improve_freq", None),
]


def safe_dot_available() -> bool:
    return shutil.which("dot") is not None


def load_tree_for(model_dir: Path, pair: str, estimator_idx):
    bundle = joblib.load(model_dir / f"rf_{pair}.joblib")
    if estimator_idx is not None:
        return bundle.estimators_[estimator_idx]
    return bundle


def split_path_info(tree, feature_names: list[str], top_n: int = 3) -> list[dict]:
    """Top N root-most decision nodes (BFS) with feature + threshold + impurity drop."""
    t = tree.tree_
    queue = [0]
    out = []
    while queue and len(out) < top_n:
        node = queue.pop(0)
        if t.children_left[node] == -1:
            continue
        left = t.children_left[node]
        right = t.children_right[node]
        nL, nR = t.n_node_samples[left], t.n_node_samples[right]
        nN = t.n_node_samples[node]
        # weighted child impurity
        wc = (nL * t.impurity[left] + nR * t.impurity[right]) / max(nN, 1)
        gain = t.impurity[node] - wc
        out.append({
            "feature": feature_names[t.feature[node]],
            "threshold": float(t.threshold[node]),
            "samples_at_node": int(nN),
            "impurity_drop": float(gain),
        })
        queue += [left, right]
    return out


def leaf_coverage(tree, X_test) -> list[dict]:
    """Per-leaf: % of test rows landing there, predicted class/value, sample count."""
    leaf_ids = tree.apply(X_test)
    counts = Counter(leaf_ids.tolist())
    total = sum(counts.values())
    t = tree.tree_
    out = []
    for leaf_id, cnt in counts.most_common():
        # predicted value/class
        value = t.value[leaf_id]
        if value.shape[-1] > 1:   # classifier
            klass = int(np.argmax(value[0]))
            pred_str = f"class={klass} (probs={value[0].tolist()})"
        else:
            pred_str = f"value={float(value[0, 0]):.4f}"
        out.append({
            "leaf_id": int(leaf_id),
            "test_rows": int(cnt),
            "test_share": cnt / total,
            "prediction": pred_str,
            "samples_train": int(t.n_node_samples[leaf_id]),
        })
    return out


def emit_yaml(path: Path, info: dict) -> None:
    """Minimal YAML emitter — no PyYAML dependency."""
    lines = []
    for k, v in info.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                if isinstance(item, dict):
                    first = True
                    for kk, vv in item.items():
                        prefix = "  - " if first else "    "
                        lines.append(f"{prefix}{kk}: {vv}")
                        first = False
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def render_one(display_name: str, model_dir: Path, pair: str, labels_file: str,
               target_metric: str, estimator_idx, use_dtreeviz: bool) -> dict:
    print(f"  {display_name}")
    meta = json.loads((model_dir / "meta.json").read_text())
    feature_names = meta["feature_names"]
    model_kind = meta["model_kind"]
    log_transform = meta.get("log_transform_label", False)
    label_transforms = meta.get("label_transforms_per_pair", {})

    tree = load_tree_for(model_dir, pair, estimator_idx)

    # ASCII text (sklearn's export_text rejects max_depth=None, so use a big int)
    txt_path = FIG / f"{display_name}.txt"
    txt = export_text(tree, feature_names=feature_names,
                      max_depth=max(tree.get_depth() + 1, 10))
    txt_path.write_text(txt)

    # Matplotlib full tree (deep trees might be huge — cap the figure size)
    full_depth = tree.get_depth()
    fig_w = max(12, min(60, tree.tree_.n_leaves * 0.5))
    fig_h = max(6, min(40, full_depth * 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(tree, feature_names=feature_names, max_depth=None,
              filled=True, rounded=True, fontsize=6,
              class_names=(["neg", "pos"] if isinstance(tree, DecisionTreeClassifier) else None),
              ax=ax)
    ax.set_title(f"{display_name}  (depth={full_depth}, leaves={tree.tree_.n_leaves})")
    fig.tight_layout()
    fig.savefig(FIG / f"{display_name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Matplotlib truncated to depth=3 (always readable)
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(tree, feature_names=feature_names, max_depth=3,
              filled=True, rounded=True, fontsize=8,
              class_names=(["neg", "pos"] if isinstance(tree, DecisionTreeClassifier) else None),
              ax=ax)
    ax.set_title(f"{display_name}  (top 3 depth shown out of {full_depth})")
    fig.tight_layout()
    fig.savefig(FIG / f"{display_name}_top3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # graphviz dot source (renderable later)
    dot_path = FIG / f"{display_name}.dot"
    export_graphviz(
        tree, out_file=str(dot_path), feature_names=feature_names,
        filled=True, rounded=True,
        class_names=(["neg", "pos"] if isinstance(tree, DecisionTreeClassifier) else None),
        impurity=False, proportion=True,
    )

    # dtreeviz SVG if `dot` is available
    svg_path = FIG / f"{display_name}.svg"
    dtreeviz_used = False
    if use_dtreeviz and safe_dot_available():
        try:
            import dtreeviz
            # We need X_train and y_train for dtreeviz. Pull from the cache.
            features_df = pd.read_parquet(FEATURES)
            labels_df = pd.read_parquet(CACHE / labels_file)
            df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
            report = json.loads((model_dir / "training_report.json").read_text())
            train_logs = set(report["split"]["train_logs"])
            mask = df["log_file"].isin(train_logs)
            X = df[feature_names].values[mask.values]
            if model_kind == "classifier":
                y = df[pair].values[mask.values]
                target_lbl = pair
            else:
                target_col = pair.replace(
                    "pair_", "ei_pct_" if target_metric == "ei_pct" else "improve_freq_")
                y = df[target_col].values[mask.values]
                target_lbl = target_col
            viz = dtreeviz.model(
                tree, X_train=X, y_train=y,
                feature_names=feature_names,
                target_name=target_lbl,
                class_names=["neg", "pos"] if model_kind == "classifier" else None,
            )
            viz.view(depth_range_to_display=(0, min(full_depth, 4))).save(str(svg_path))
            dtreeviz_used = True
        except Exception as e:
            warnings.warn(f"dtreeviz failed for {display_name}: {e}")

    # YAML sidecar — depth, n_leaves, top-3 splits, top-5 leaves by test coverage
    features_df = pd.read_parquet(FEATURES)
    labels_df = pd.read_parquet(CACHE / labels_file)
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
    report = json.loads((model_dir / "training_report.json").read_text())
    test_logs = set(report["split"]["test_logs"])
    mask = df["log_file"].isin(test_logs)
    X_test = df[feature_names].values[mask.values]
    splits = split_path_info(tree, feature_names, top_n=5)
    leaves = leaf_coverage(tree, X_test)
    top5_share = sum(l["test_share"] for l in leaves[:5])
    info = {
        "display_name": display_name,
        "model_dir": str(model_dir.relative_to(REPO)),
        "pair": pair,
        "model_kind": model_kind,
        "estimator_idx": estimator_idx if estimator_idx is not None else "(single tree)",
        "depth": full_depth,
        "n_leaves": int(tree.tree_.n_leaves),
        "node_count": int(tree.tree_.node_count),
        "n_features_used": int((np.bincount(tree.tree_.feature[tree.tree_.feature >= 0],
                                            minlength=len(feature_names)) > 0).sum()),
        "top_5_splits_bfs": splits,
        "top_5_leaves_by_test_coverage": leaves[:5],
        "top5_leaf_test_coverage_share": round(top5_share, 4),
        "dtreeviz_svg_emitted": dtreeviz_used,
    }
    emit_yaml(FIG / f"{display_name}.yaml", info)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dtreeviz", action="store_true",
                    help="Skip dtreeviz SVG step even if `dot` is on PATH.")
    args = ap.parse_args()

    has_dot = safe_dot_available()
    print(f"dot binary available: {has_dot}")
    print(f"Rendering {len(CURATED)} trees to {FIG}\n")

    for display_name, model_dir_rel, pair, labels, target_metric, est_idx in CURATED:
        model_dir = REPO / model_dir_rel
        if not (model_dir / f"rf_{pair}.joblib").exists():
            print(f"  [skip] {display_name}: model not found")
            continue
        try:
            render_one(display_name, model_dir, pair, labels, target_metric, est_idx,
                       use_dtreeviz=not args.skip_dtreeviz)
        except Exception as e:
            print(f"  [FAIL] {display_name}: {e}")

    print(f"\nDone. Outputs in {FIG}")
    if not has_dot:
        print(
            "\nNOTE: system `dot` binary not on PATH. Install with `brew install graphviz`\n"
            "      to enable dtreeviz SVG renders. The .dot files are still written and\n"
            "      can be rendered later with `dot -Tpng <file>.dot -o <file>.png`."
        )


if __name__ == "__main__":
    main()
