"""Load trained models and predict operator selection."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def load_models(
    model_dir: str | Path,
) -> tuple[dict[str, RandomForestClassifier], list[str]]:
    """Load all trained RF models and feature names from a model directory.

    Returns (models_dict, feature_names) where models_dict maps
    pair column name (e.g. 'pair_0_0') to the classifier.
    """
    model_dir = Path(model_dir)

    meta_path = model_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    label_columns = meta["label_columns"]

    models = {}
    for col in label_columns:
        model_path = model_dir / f"rf_{col}.joblib"
        if model_path.exists():
            models[col] = joblib.load(model_path)

    return models, feature_names


def predict(
    models: dict[str, RandomForestClassifier],
    features: dict[str, float],
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Predict P(good) for each operator pair.

    Returns list of (pair_name, probability) sorted by probability descending.
    """
    X = np.array([[features[name] for name in feature_names]])

    results = []
    for pair_name, clf in models.items():
        proba = clf.predict_proba(X)
        # P(good) = probability of class 1
        p_good = proba[0, 1] if proba.shape[1] > 1 else float(clf.predict(X)[0])
        results.append((pair_name, p_good))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
