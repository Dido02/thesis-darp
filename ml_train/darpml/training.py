"""Random Forest training, evaluation, and model saving."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


@dataclass
class ClassifierMetrics:
    pair_name: str
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    n_train: int
    n_test: int
    n_positive_train: int
    n_positive_test: int
    always1_f1: float
    always0_f1: float
    f1_lift: float


@dataclass
class RegressorMetrics:
    pair_name: str
    mae: float
    r2: float
    baseline_mae: float
    mae_lift: float           # baseline_mae - mae; positive = beats mean predictor
    n_train: int
    n_test: int
    label_mean_train: float
    label_std_train: float
    label_mean_test: float
    label_std_test: float


@dataclass
class TrainingReport:
    pair_metrics: list                 # list[ClassifierMetrics] or list[RegressorMetrics]
    feature_importances: dict[str, float]
    feature_names: list[str]
    n_samples: int
    n_features: int
    n_pairs: int
    n_train_logs: int
    n_test_logs: int
    n_train_rows: int
    n_test_rows: int
    train_log_files: list[str]
    test_log_files: list[str]
    median_f1_lift: float              # 0.0 for regressor
    n_positive_lift: int
    n_zero_lift: int
    n_negative_lift: int
    n_degenerate_pairs: int
    model_kind: str = "classifier"


def train_single_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pair_name: str,
    n_estimators: int = 100,
    random_state: int = 42,
    model_type: str = "forest",
    tree_max_depth: Optional[int] = None,
) -> tuple[Union[RandomForestClassifier, DecisionTreeClassifier], ClassifierMetrics]:
    """Train and evaluate one classifier for a single operator pair.

    model_type='forest' (default): RandomForestClassifier with n_estimators trees.
    model_type='tree': single DecisionTreeClassifier (no ensembling). For tree
        mode, tree_max_depth optionally caps depth (None = unrestricted).
    """
    # Check for degenerate labels
    unique_train = np.unique(y_train)
    if len(unique_train) < 2:
        warnings.warn(
            f"Pair {pair_name}: only class {unique_train[0]} in training data. "
            f"Classifier will be degenerate."
        )

    if model_type == "tree":
        clf = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            max_depth=tree_max_depth,
        )
    elif model_type == "forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            max_depth=tree_max_depth,
        )
    else:
        raise ValueError(f"model_type must be 'forest' or 'tree', got {model_type!r}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute metrics (handle edge cases for single-class)
    tn, fp, fn, tp = 0, 0, 0, 0
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Baseline metrics
    y_ones = np.ones_like(y_test)
    y_zeros = np.zeros_like(y_test)
    always1_f1 = float(f1_score(y_test, y_ones, zero_division=0))
    always0_f1 = float(f1_score(y_test, y_zeros, zero_division=0))
    rf_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    lift = rf_f1 - max(always1_f1, always0_f1)

    metrics = ClassifierMetrics(
        pair_name=pair_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        specificity=specificity,
        f1=rf_f1,
        n_train=len(y_train),
        n_test=len(y_test),
        n_positive_train=int(y_train.sum()),
        n_positive_test=int(y_test.sum()),
        always1_f1=always1_f1,
        always0_f1=always0_f1,
        f1_lift=lift,
    )
    return clf, metrics


def train_single_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pair_name: str,
    n_estimators: int = 100,
    random_state: int = 42,
    model_type: str = "forest",
    tree_max_depth: Optional[int] = None,
) -> tuple[Union[RandomForestRegressor, DecisionTreeRegressor], RegressorMetrics]:
    """Train and evaluate one regressor for a single operator pair.

    model_type='forest' (default): RandomForestRegressor with n_estimators trees.
    model_type='tree': single DecisionTreeRegressor. tree_max_depth optionally
        caps depth (None = unrestricted).

    Target: continuous improve_freq value in [0, 1] (the M-trial improvement
    rate). Reports MAE, R², and MAE-lift over a constant mean predictor.
    """
    label_std_train = float(np.std(y_train))
    if label_std_train < 1e-6:
        warnings.warn(
            f"Pair {pair_name}: degenerate continuous label "
            f"(std={label_std_train:.3e}). Regressor will be near-constant."
        )

    if model_type == "tree":
        reg = DecisionTreeRegressor(
            random_state=random_state,
            max_depth=tree_max_depth,
        )
    elif model_type == "forest":
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=tree_max_depth,
        )
    else:
        raise ValueError(f"model_type must be 'forest' or 'tree', got {model_type!r}")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae = float(np.mean(np.abs(y_test - y_pred)))

    # R² (manual, handles constant test label)
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Baseline: predict the train-set mean for every test point
    y_baseline = np.full_like(y_test, fill_value=float(y_train.mean()), dtype=np.float64)
    baseline_mae = float(np.mean(np.abs(y_test - y_baseline)))
    mae_lift = baseline_mae - mae  # positive = regressor beats mean predictor

    metrics = RegressorMetrics(
        pair_name=pair_name,
        mae=mae,
        r2=r2,
        baseline_mae=baseline_mae,
        mae_lift=mae_lift,
        n_train=len(y_train),
        n_test=len(y_test),
        label_mean_train=float(y_train.mean()),
        label_std_train=label_std_train,
        label_mean_test=float(y_test.mean()),
        label_std_test=float(np.std(y_test)),
    )
    return reg, metrics


def train_pipeline(
    features_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    n_estimators: int = 100,
    test_size: float = 0.25,
    random_state: int = 42,
    feature_names: list[str] | None = None,
    mode: str | None = None,
    batches: list[str] | None = None,
    model_kind: str = "classifier",
    target_metric: str = "improve_freq",
    log_transform_label: bool = False,
    model_type: str = "forest",
    tree_max_depth: Optional[int] = None,
) -> TrainingReport:
    """Full training pipeline: load data, train 20 classifiers, save models.

    Uses per-log-file train/test split to prevent within-trajectory leakage.

    feature_names, when provided, restricts training to that explicit subset
    (validated against the parquet columns). mode/batches get recorded in
    meta.json so export_onnx can emit a new-format manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)

    # Join on (log_file, iter)
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
    print(f"Joined dataset: {len(df)} samples")

    # Identify feature and label columns
    parquet_features = [c for c in features_df.columns if c not in ("log_file", "iter")]
    if feature_names is None:
        feature_names = parquet_features
    else:
        missing = [n for n in feature_names if n not in parquet_features]
        if missing:
            raise ValueError(
                f"train_pipeline: feature_names contains columns not in parquet: {missing}"
            )
    if model_kind not in ("classifier", "regressor"):
        raise ValueError(f"model_kind must be 'classifier' or 'regressor', got {model_kind!r}")
    if model_type not in ("forest", "tree"):
        raise ValueError(f"model_type must be 'forest' or 'tree', got {model_type!r}")

    # Map target_metric to its label-column prefix. Only used for regressor;
    # classifier ignores this and reads the binary `pair_{d}_{r}` columns.
    target_prefix_map = {
        "improve_freq": "improve_freq_",  # frequency-only (legacy default)
        "ei_pct":       "ei_pct_",        # expected improvement % (magnitude-aware)
    }
    if target_metric not in target_prefix_map:
        raise ValueError(
            f"target_metric must be one of {list(target_prefix_map)}, got {target_metric!r}"
        )

    label_cols = sorted(c for c in labels_df.columns if c.startswith("pair_"))
    if model_kind == "regressor":
        # Verify the continuous columns we'll actually train on are present.
        prefix = target_prefix_map[target_metric]
        missing = [c.replace("pair_", prefix) for c in label_cols
                   if c.replace("pair_", prefix) not in labels_df.columns]
        if missing:
            raise ValueError(
                f"model_kind='regressor' with target_metric={target_metric!r} "
                f"requires {prefix}* columns; missing: {missing}"
            )

    X = df[feature_names].values
    n_samples, n_features = X.shape

    # Per-log-file split
    log_files = df["log_file"].unique()
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(log_files)
    n_train_logs = int(len(shuffled) * (1 - test_size))
    train_logs = set(shuffled[:n_train_logs])
    test_logs = set(shuffled[n_train_logs:])

    assert len(train_logs) >= 1 and len(test_logs) >= 1, \
        f"Split degeneracy: {len(train_logs)} train, {len(test_logs)} test logs"

    train_mask = df["log_file"].isin(train_logs)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    print(f"Per-log split: {len(train_logs)} train logs ({len(train_idx)} rows), "
          f"{len(test_logs)} test logs ({len(test_idx)} rows)")

    X_train, X_test = X[train_idx], X[test_idx]

    # Train one model per pair (classifier or regressor depending on model_kind)
    all_metrics: list = []
    all_importances = np.zeros(n_features)
    models: dict = {}

    target_prefix = target_prefix_map[target_metric]

    # When True, regressor labels are log-transformed and min-max scaled per
    # pair (fit on TRAIN ONLY) before training. The trained ONNX therefore
    # emits a scaled-log-EI score; relative ordering across pairs is preserved
    # so the C++ proportional sampler still works without any code change.
    # Metrics in the report are computed back in ORIGINAL space (after
    # inversion) so they're directly comparable to the non-log regressor.
    log_transform_active = (
        log_transform_label
        and model_kind == "regressor"
        and target_metric == "ei_pct"
    )
    label_transforms_per_pair: dict[str, dict[str, float]] = {}

    for col in label_cols:
        if model_kind == "regressor":
            target_col = col.replace("pair_", target_prefix)
        else:
            target_col = col
        y = df[target_col].values
        y_train, y_test = y[train_idx], y[test_idx]

        if log_transform_active:
            # log(label + 1e-6), then min-max scale using TRAIN-ONLY stats.
            log_y_train = np.log(y_train + 1e-6)
            log_y_test = np.log(y_test + 1e-6)
            log_min = float(log_y_train.min())
            log_max = float(log_y_train.max())
            span = max(log_max - log_min, 1e-12)
            y_train_used = (log_y_train - log_min) / span
            y_test_used = (log_y_test - log_min) / span
            label_transforms_per_pair[col] = {
                "log_min": log_min,
                "log_max": log_max,
            }
        else:
            y_train_used = y_train
            y_test_used = y_test

        if model_kind == "regressor":
            clf, metrics = train_single_regressor(
                X_train, y_train_used, X_test, y_test_used,
                pair_name=col,
                n_estimators=n_estimators,
                random_state=random_state,
                model_type=model_type,
                tree_max_depth=tree_max_depth,
            )
            if log_transform_active:
                # Re-compute MAE / R² / mae_lift / label-mean in ORIGINAL space.
                y_pred_t = clf.predict(X_test)
                # Inverse: y_orig = exp(y_t * span + log_min) - 1e-6, clipped to non-negative
                y_pred_orig = np.exp(y_pred_t * span + log_min) - 1e-6
                y_pred_orig = np.clip(y_pred_orig, 0.0, None)
                mae_orig = float(np.mean(np.abs(y_test - y_pred_orig)))
                ss_res = float(np.sum((y_test - y_pred_orig) ** 2))
                ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                r2_orig = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                y_baseline = np.full_like(
                    y_test, fill_value=float(y_train.mean()), dtype=np.float64
                )
                baseline_mae_orig = float(np.mean(np.abs(y_test - y_baseline)))
                metrics.mae = mae_orig
                metrics.r2 = r2_orig
                metrics.baseline_mae = baseline_mae_orig
                metrics.mae_lift = baseline_mae_orig - mae_orig
                metrics.label_mean_train = float(y_train.mean())
                metrics.label_std_train = float(np.std(y_train))
                metrics.label_mean_test = float(y_test.mean())
                metrics.label_std_test = float(np.std(y_test))
        else:
            clf, metrics = train_single_classifier(
                X_train, y_train_used, X_test, y_test_used,
                pair_name=col,
                n_estimators=n_estimators,
                random_state=random_state,
                model_type=model_type,
                tree_max_depth=tree_max_depth,
            )
        all_metrics.append(metrics)
        all_importances += clf.feature_importances_
        models[col] = clf

        if model_kind == "regressor":
            print(f"  {col}: mae={metrics.mae:.4f} r2={metrics.r2:+.3f} "
                  f"baseline_mae={metrics.baseline_mae:.4f} "
                  f"mae_lift={metrics.mae_lift:+.4f} "
                  f"(label_std_train={metrics.label_std_train:.3f})")
        else:
            print(f"  {col}: acc={metrics.accuracy:.3f} prec={metrics.precision:.3f} "
                  f"rec={metrics.recall:.3f} f1={metrics.f1:.3f} "
                  f"a1={metrics.always1_f1:.3f} lift={metrics.f1_lift:+.3f} "
                  f"(+train={metrics.n_positive_train}/{metrics.n_train})")

    # Average feature importances across all classifiers
    all_importances /= len(label_cols)
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names, all_importances)
    }

    # Compute lift summary (excluding degenerate pairs).
    # Classifier degenerate = pos rate <0.05 or >0.95 on test set.
    # Regressor degenerate = train-label std < 1e-3 (constant target).
    non_degen_lifts = []
    n_degenerate = 0
    for m in all_metrics:
        if model_kind == "regressor":
            if m.label_std_train < 1e-3:
                n_degenerate += 1
            else:
                non_degen_lifts.append(m.mae_lift)
        else:
            pos_rate = m.n_positive_test / m.n_test if m.n_test > 0 else 0.0
            if pos_rate < 0.05 or pos_rate > 0.95:
                n_degenerate += 1
            else:
                non_degen_lifts.append(m.f1_lift)

    median_lift = float(np.median(non_degen_lifts)) if non_degen_lifts else 0.0
    if model_kind == "regressor":
        # Lifts here are MAE-lifts; "positive" means regressor beats baseline.
        n_positive = sum(1 for l in non_degen_lifts if l > 0.001)
        n_zero = sum(1 for l in non_degen_lifts if abs(l) <= 0.001)
        n_negative = sum(1 for l in non_degen_lifts if l < -0.001)
    else:
        n_positive = sum(1 for l in non_degen_lifts if l > 0.01)
        n_zero = sum(1 for l in non_degen_lifts if abs(l) <= 0.01)
        n_negative = sum(1 for l in non_degen_lifts if l < -0.01)

    # Save models
    for col, clf in models.items():
        model_path = output_dir / f"rf_{col}.joblib"
        joblib.dump(clf, model_path)

    # Try ONNX export (optional)
    _try_onnx_export(models, feature_names, output_dir)

    # Save feature names + mode/batches for downstream ONNX export.
    # label_columns stays in pair_{d}_{r} form regardless of model_kind so
    # export_onnx.py's index parsing works unchanged; the regressor was
    # trained on the parallel improve_freq_{d}_{r} columns but the model
    # files are still keyed by the binary-pair names.
    meta = {
        "feature_names": feature_names,
        "label_columns": label_cols,
        "mode": mode,
        "batches": batches if batches is not None else [],
        "model_kind": model_kind,
        "target_metric": target_metric,  # which label column the regressor was trained on
        "log_transform_label": log_transform_active,
        # Per-pair (log_min, log_max) so the C++ side could invert the
        # scaled-log output to original-space EI if it ever wanted to.
        "label_transforms_per_pair": label_transforms_per_pair,
        # Estimator type ("forest" = RandomForest n_estimators trees;
        # "tree" = single DecisionTree). C++ inference is type-agnostic;
        # this field is purely for downstream tooling / reproducibility.
        "model_type": model_type,
        "tree_max_depth": tree_max_depth,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Build report
    report = TrainingReport(
        pair_metrics=all_metrics,
        feature_importances=feature_importances,
        feature_names=feature_names,
        n_samples=n_samples,
        n_features=n_features,
        n_pairs=len(label_cols),
        n_train_logs=len(train_logs),
        n_test_logs=len(test_logs),
        n_train_rows=len(train_idx),
        n_test_rows=len(test_idx),
        train_log_files=sorted(train_logs),
        test_log_files=sorted(test_logs),
        median_f1_lift=median_lift,
        n_positive_lift=n_positive,
        n_zero_lift=n_zero,
        n_negative_lift=n_negative,
        n_degenerate_pairs=n_degenerate,
        model_kind=model_kind,
    )

    # Save report (machine-readable JSON). Use "median_f1_lift" key for
    # classifier and "median_mae_lift" for regressor so consumers can
    # disambiguate without reading model_kind separately.
    lift_metric_name = "median_mae_lift" if model_kind == "regressor" else "median_f1_lift"
    report_dict = {
        "model_kind": model_kind,
        "n_samples": report.n_samples,
        "n_features": report.n_features,
        "n_pairs": report.n_pairs,
        "split": {
            "method": "per_log_file",
            "n_train_logs": report.n_train_logs,
            "n_test_logs": report.n_test_logs,
            "n_train_rows": report.n_train_rows,
            "n_test_rows": report.n_test_rows,
            "train_logs": report.train_log_files,
            "test_logs": report.test_log_files,
        },
        "lift_summary": {
            "n_degenerate_excluded": report.n_degenerate_pairs,
            "n_non_degenerate": report.n_pairs - report.n_degenerate_pairs,
            lift_metric_name: report.median_f1_lift,
            "n_positive_lift": report.n_positive_lift,
            "n_zero_lift": report.n_zero_lift,
            "n_negative_lift": report.n_negative_lift,
        },
        "feature_importances": report.feature_importances,
        "pair_metrics": [asdict(m) for m in report.pair_metrics],
    }
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nSaved {len(models)} models to {output_dir}")

    return report


def _try_onnx_export(
    models: dict[str, RandomForestClassifier],
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Try to export models to ONNX. Silently skip if skl2onnx not installed."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("  (skl2onnx not installed — skipping ONNX export)")
        return

    n_features = len(feature_names)
    initial_type = [("features", FloatTensorType([None, n_features]))]

    for col, clf in models.items():
        try:
            onnx_model = convert_sklearn(clf, initial_types=initial_type)
            onnx_path = output_dir / f"rf_{col}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except Exception as e:
            warnings.warn(f"ONNX export failed for {col}: {e}")
