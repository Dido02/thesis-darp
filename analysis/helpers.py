"""Helper utilities for the data exploration notebook.

Just enough to keep the notebook itself focused on plots and analysis rather
than path manipulation. Read-only — never writes to ml_train/ or cpp/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Repo paths derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_TRAIN_DIR = REPO_ROOT / "ml_train"
CACHE_DIR = ML_TRAIN_DIR / "cache_v2_dynamic"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Make `darpml` importable from the notebook so we can pull batch metadata.
if str(ML_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAIN_DIR))


# Operator names — must match the order in cpp/src/ml/manifest.cpp /
# DESTROY_OPS, REPAIR_OPS in ml_train/export_onnx.py.
DESTROY_OPS = ["random", "worst", "shaw", "rtcritical", "lowslack"]
REPAIR_OPS = ["greedy", "regret2", "regret3", "slack"]

# Per-destroy color palette so the 4 repair variants of one destroy share a hue.
DESTROY_COLORS = {
    "random":     "#1f77b4",
    "worst":      "#ff7f0e",
    "shaw":       "#2ca02c",
    "rtcritical": "#d62728",
    "lowslack":   "#9467bd",
}

# Batch palette, used for feature plots.
BATCH_COLORS = {
    "core":            "#1f77b4",
    "tightness":       "#ff7f0e",
    "worst_support":   "#2ca02c",
    "lns_state":       "#d62728",
    "rolling_horizon": "#9467bd",
}

# Reference: the labeling threshold used in
# ml_train/darpml/labels.py::compute_improvement_freq_labels.
IMPROVEMENT_FREQ_THRESHOLD = 0.3


def load_features_labels() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached features + labels parquets from ml_train/cache_v2_dynamic/."""
    features = pd.read_parquet(CACHE_DIR / "features.parquet")
    labels = pd.read_parquet(CACHE_DIR / "labels.parquet")
    return features, labels


def pair_label_columns(df: pd.DataFrame) -> list[str]:
    """All `pair_<d>_<r>` (binary label) columns, in registry index order."""
    cols = [c for c in df.columns if c.startswith("pair_")]
    return sorted(cols, key=_pair_index_key)


def pair_freq_columns(df: pd.DataFrame) -> list[str]:
    """All `improve_freq_<d>_<r>` (continuous) columns, in registry index order."""
    cols = [c for c in df.columns if c.startswith("improve_freq_")]
    return sorted(cols, key=_freq_index_key)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All feature columns (everything except join keys)."""
    return [c for c in df.columns if c not in ("log_file", "iter")]


def _pair_index_key(col: str):
    _, d, r = col.split("_")
    return int(d), int(r)


def _freq_index_key(col: str):
    parts = col.split("_")
    return int(parts[2]), int(parts[3])


def pair_label_to_name(col: str) -> Tuple[str, str]:
    """`pair_3_0` -> ('rtcritical', 'greedy')."""
    _, d, r = col.split("_")
    return DESTROY_OPS[int(d)], REPAIR_OPS[int(r)]


def freq_col_to_name(col: str) -> Tuple[str, str]:
    """`improve_freq_3_0` -> ('rtcritical', 'greedy')."""
    parts = col.split("_")
    return DESTROY_OPS[int(parts[2])], REPAIR_OPS[int(parts[3])]


def parse_tier(log_filename: str) -> str:
    """`manhattan_30_s_delay03_min_s42_t50e100_dynamic.jsonl` -> '30s'.
    `manhattan_15min_03min_s42_t50e10_dynamic.jsonl` -> '15min'.
    """
    if "_30_s_" in log_filename:
        return "30s"
    if "_01_min_" in log_filename:
        return "01min"
    if "_02_min_" in log_filename:
        return "02min"
    if "_05_min_" in log_filename:
        return "05min"
    if "_15min_" in log_filename:
        return "15min"
    return "unknown"


def feature_batch_map() -> Dict[str, str]:
    """Returns {feature_name: batch} from the registry."""
    from darpml.features import _REGISTRY
    return {name: entry.batch for name, entry in _REGISTRY.items()}


def feature_mode_map() -> Dict[str, str]:
    from darpml.features import _REGISTRY
    return {name: entry.mode for name, entry in _REGISTRY.items()}


def save_fig(fig, name: str) -> Path:
    """Save fig to analysis/figures/<name>.png and return the path."""
    p = FIGURES_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    return p
