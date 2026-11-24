#!/usr/bin/env python3
"""Train 4 new t07 model configurations for bench evaluation, exporting ONNX.

Configs:
  1. forest n=100, d=8, K=20  -> ml_train/models_t07_forest_d8_K20/
  2. forest n=100, d=8, K=16  -> ml_train/models_t07_forest_d8_K16/
  3. tree d=4, K=4            -> ml_train/models_t07_tree_d4_K4/
  4. tree d=8, K=16           -> ml_train/models_t07_tree_d8_K16/

Uses the deployed model's exact 75/25 log split (33 train / 12 test) so the
trained models can be compared apples-to-apples against the deployed
ml/20f_t07__T0.10 selector.

For K<20 configs, the sklearn Pipeline embeds a ColumnTransformer that
slices the 20-dim feature vector down to the top-K features by aggregate
Gini importance in the deployed forest. The resulting ONNX takes the full
20-dim input (matching what the C++ ML selector feeds at runtime) but
internally drops the unused columns before the classifier.

Output for each config:
  ml_train/models_t07_<tag>/
    rf_pair_{d}_{r}.joblib
    rf_pair_{d}_{r}.onnx
    onnx/rf_pair_{d}_{r}.onnx   (mirror — the C++ points at this dir)
    meta.json
    training_report.json
"""
from __future__ import annotations
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "ml_train" / "cache_v3_full"
DEPLOYED_DIR = REPO / "ml_train" / "models_v3_full_20f_t07"

# Canonical 20 feature names (must match C++ ml selector's feature order)
FEATURE_NAMES_20 = [
    "num_active_requests", "radius_avg_tt", "frac_committed", "avg_route_load",
    "temperature_normalized", "ride_time_utilization_avg",
    "ride_time_utilization_max", "near_violation_count", "route_slack_avg",
    "route_slack_std", "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "neighborhood_size_frac", "cost_vs_initial_frac",
    "cost_vs_incumbent_frac", "iteration_norm", "unassigned_count",
    "committed_travel_time_frac", "horizon_elapsed_frac",
]
# All 20 pair columns the meta.json must list (5 destroy x 4 repair).
# The C++ ml selector references ONNX files by operator name, not pair index.
DESTROY_NAMES = ["random", "worst", "shaw", "lowslack", "rtcritical"]
REPAIR_NAMES  = ["greedy", "regret2", "regret3", "slack"]
ALL_PAIRS = [f"pair_{d}_{r}" for d in range(5) for r in range(4)]
USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]
DEGENERATE_PAIRS = sorted(set(ALL_PAIRS) - set(USED_PAIRS))


def pair_to_onnx_name(pair: str) -> str:
    """Convert pair_d_r to <destroy_name>_<repair_name>.onnx."""
    _, d, r = pair.split("_")
    return f"{DESTROY_NAMES[int(d)]}_{REPAIR_NAMES[int(r)]}.onnx"

CONFIGS = [
    {"tag": "forest_d8_K20",  "type": "forest", "n_estimators": 100, "max_depth": 8,  "K": 20},
    {"tag": "forest_d8_K16",  "type": "forest", "n_estimators": 100, "max_depth": 8,  "K": 16},
    {"tag": "tree_d4_K4",     "type": "tree",   "max_depth": 4,                       "K": 4},
    {"tag": "tree_d8_K16",    "type": "tree",   "max_depth": 8,                       "K": 16},
]


def aggregated_top_k_indices(K: int) -> list[int]:
    """Return indices (into FEATURE_NAMES_20) of top-K features by aggregate
    Gini importance across the 9 USED pairs in the deployed forest."""
    if K >= len(FEATURE_NAMES_20):
        return list(range(len(FEATURE_NAMES_20)))
    imps = np.zeros(len(FEATURE_NAMES_20))
    for pair in USED_PAIRS:
        clf = joblib.load(DEPLOYED_DIR / f"rf_{pair}.joblib")
        imps += clf.feature_importances_
    imps /= len(USED_PAIRS)
    order = np.argsort(-imps)
    return sorted(order[:K].tolist())


def build_estimator(cfg) -> object:
    if cfg["type"] == "forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        return DecisionTreeClassifier(
            max_depth=cfg["max_depth"], class_weight="balanced", random_state=42,
        )


def make_pipeline(cfg, top_k_idx):
    if cfg["K"] >= 20:
        # No column selection needed; ColumnTransformer would round-trip
        # the input through a passthrough — fine for ONNX but adds a no-op
        # node. Use bare estimator for cleanliness.
        return build_estimator(cfg)
    ct = ColumnTransformer(
        [("select", "passthrough", top_k_idx)], remainder="drop"
    )
    return Pipeline([("select_features", ct), ("clf", build_estimator(cfg))])


def degenerate_clf(prevalence: float):
    """Return a tiny tree that predicts the majority class everywhere.
    Used to populate pair_3_*, pair_4_* slots that the C++ expects but
    we don't actually use in the bench (the C++ only consumes pair_0..2 x 0..2)."""
    # We need a fitted estimator that takes 20 inputs and outputs probas
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    X_fake = np.zeros((4, 20))
    X_fake[2:] = 1
    y_fake = np.array([0, 0, 1, 1])
    clf.fit(X_fake, y_fake)
    return clf


def export_onnx(estimator, out_path: Path):
    # Default skl2onnx wraps predict_proba in a "zipmap" that produces a
    # sequence<map<int64, float>>. The C++ ml selector expects a plain
    # tensor[batch, 2]. Disable zipmap and rename outputs to match the
    # deployed format ("label" + "probabilities").
    onnx_model = convert_sklearn(
        estimator,
        initial_types=[("input", FloatTensorType([None, 20]))],
        target_opset=15,
        options={id(estimator): {"zipmap": False, "output_class_labels": False}},
    )
    # Rename outputs to match the deployed naming convention.
    for o in onnx_model.graph.output:
        if o.name in ("output_label", "label"):
            o.name = "label"
        elif o.name in ("output_probability", "probabilities"):
            o.name = "probabilities"
    # Patch any internal nodes that reference the old names.
    for node in onnx_model.graph.node:
        for i, n in enumerate(node.output):
            if n == "output_label": node.output[i] = "label"
            elif n == "output_probability": node.output[i] = "probabilities"
        for i, n in enumerate(node.input):
            if n == "output_label": node.input[i] = "label"
            elif n == "output_probability": node.input[i] = "probabilities"
    out_path.write_bytes(onnx_model.SerializeToString())


def main():
    # Load data + deployed split
    features = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels_t0.7.parquet")
    df = features.merge(labels, on=["log_file", "iter"], how="inner")
    report = json.loads((DEPLOYED_DIR / "training_report.json").read_text())
    train_logs = set(report["split"]["train_logs"])
    test_logs = set(report["split"]["test_logs"])
    train_mask = df["log_file"].isin(train_logs).values
    print(f"Deployed split: {len(train_logs)} train logs / {len(test_logs)} test logs")
    print(f"  {train_mask.sum()} train rows / {(~train_mask).sum()} test rows")

    X_all = df[FEATURE_NAMES_20].values
    X_train = X_all[train_mask]

    deployed_manifest = DEPLOYED_DIR / "onnx" / "feature_manifest.json"
    for cfg in CONFIGS:
        out_dir = REPO / "ml_train" / f"models_t07_{cfg['tag']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_subdir = out_dir / "onnx"
        onnx_subdir.mkdir(exist_ok=True)
        # The C++ ml selector reads feature_manifest.json from the onnx/ dir
        # to know which features to feed in which order. Copy from deployed.
        shutil.copy(deployed_manifest, onnx_subdir / "feature_manifest.json")
        top_k_idx = aggregated_top_k_indices(cfg["K"])
        top_k_names = [FEATURE_NAMES_20[i] for i in top_k_idx]
        print(f"\n=== {cfg['tag']} ===")
        print(f"  K={cfg['K']} (indices {top_k_idx})")
        per_pair_metrics = []
        for pair in ALL_PAIRS:
            onnx_name = pair_to_onnx_name(pair)  # e.g. "random_greedy.onnx"
            if pair not in USED_PAIRS:
                # Degenerate slot — keep meta.json full but the C++ never queries these
                clf = degenerate_clf(0.0)
                joblib.dump(clf, out_dir / f"rf_{pair}.joblib")
                export_onnx(clf, onnx_subdir / onnx_name)
                continue
            y = df[pair].values
            y_train = y[train_mask]
            y_test = y[~train_mask]
            if y_train.sum() < 2:
                clf = degenerate_clf(float(y_train.mean()))
            else:
                clf = make_pipeline(cfg, top_k_idx)
                clf.fit(X_train, y_train)
            joblib.dump(clf, out_dir / f"rf_{pair}.joblib")
            export_onnx(clf, onnx_subdir / onnx_name)
            # Quick test metrics
            y_pred = clf.predict(X_all[~train_mask])
            from sklearn.metrics import f1_score, accuracy_score
            per_pair_metrics.append({
                "pair_name": pair,
                "n_train": int(train_mask.sum()),
                "n_test": int((~train_mask).sum()),
                "n_positive_train": int(y_train.sum()),
                "n_positive_test": int(y_test.sum()),
                "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
            })
            print(f"  {pair}: test_f1={per_pair_metrics[-1]['test_f1']:.3f}")

        # Write meta.json mirroring the deployed format
        meta = {
            "feature_names": FEATURE_NAMES_20,
            "label_columns": ALL_PAIRS,
            "mode": "classifier",
            "batches": "full20",
            "model_kind": "classifier",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (out_dir / "training_report.json").write_text(json.dumps({
            "model_kind": "classifier",
            "config": cfg,
            "feature_names_used": top_k_names,
            "n_samples": int(len(df)),
            "n_features": 20,
            "n_pairs": len(ALL_PAIRS),
            "split": {"method": "per_log_file (deployed split)",
                      "n_train_logs": len(train_logs),
                      "n_test_logs": len(test_logs),
                      "n_train_rows": int(train_mask.sum()),
                      "n_test_rows": int((~train_mask).sum())},
            "pair_metrics": per_pair_metrics,
        }, indent=1))
        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
