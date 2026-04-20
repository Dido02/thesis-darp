"""Random Forest training, evaluation, and model saving."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
class TrainingReport:
    pair_metrics: list[ClassifierMetrics]
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
    median_f1_lift: float
    n_positive_lift: int
    n_zero_lift: int
    n_negative_lift: int
    n_degenerate_pairs: int


def train_single_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pair_name: str,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, ClassifierMetrics]:
    """Train and evaluate one RF classifier for a single operator pair."""
    # Check for degenerate labels
    unique_train = np.unique(y_train)
    if len(unique_train) < 2:
        warnings.warn(
            f"Pair {pair_name}: only class {unique_train[0]} in training data. "
            f"Classifier will be degenerate."
        )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
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
    label_cols = [c for c in labels_df.columns if c.startswith("pair_")]

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

    # Train one classifier per pair
    all_metrics: list[ClassifierMetrics] = []
    all_importances = np.zeros(n_features)
    models: dict[str, RandomForestClassifier] = {}

    for col in sorted(label_cols):
        y = df[col].values
        y_train, y_test = y[train_idx], y[test_idx]

        clf, metrics = train_single_classifier(
            X_train, y_train, X_test, y_test,
            pair_name=col,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        all_metrics.append(metrics)
        all_importances += clf.feature_importances_
        models[col] = clf

        print(f"  {col}: acc={metrics.accuracy:.3f} prec={metrics.precision:.3f} "
              f"rec={metrics.recall:.3f} f1={metrics.f1:.3f} "
              f"a1={metrics.always1_f1:.3f} lift={metrics.f1_lift:+.3f} "
              f"(+train={metrics.n_positive_train}/{metrics.n_train})")

    # Average feature importances across all classifiers
    all_importances /= len(label_cols)
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names, all_importances)
    }

    # Compute lift summary (excluding degenerate pairs)
    # Degenerate = positive rate < 0.05 or > 0.95 on test set
    non_degen_lifts = []
    n_degenerate = 0
    for m in all_metrics:
        pos_rate = m.n_positive_test / m.n_test if m.n_test > 0 else 0.0
        if pos_rate < 0.05 or pos_rate > 0.95:
            n_degenerate += 1
        else:
            non_degen_lifts.append(m.f1_lift)

    median_lift = float(np.median(non_degen_lifts)) if non_degen_lifts else 0.0
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
    meta = {
        "feature_names": feature_names,
        "label_columns": sorted(label_cols),
        "mode": mode,
        "batches": batches if batches is not None else [],
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
    )

    # Save report (machine-readable JSON)
    report_dict = {
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
            "median_f1_lift": report.median_f1_lift,
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
