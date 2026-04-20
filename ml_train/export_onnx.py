"""Export trained RF models to ONNX and emit feature_manifest.json.

Reads 20 joblib classifiers from ml_train/models/ and writes:
- ml_train/models/onnx/{d_name}_{r_name}.onnx (20 files)
- ml_train/models/onnx/feature_manifest.json

The manifest is the single source of truth binding Python features, model files,
and operator indices together. C++ reads it at solver startup.

Usage:
    pip install -e '.[onnx]'
    python export_onnx.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


# The operator lists must match what was in effect when trials were logged
# (verified against JSONL header.destroy_ops / header.repair_ops).
DESTROY_OPS = ["random", "worst", "shaw", "rtcritical", "lowslack"]
REPAIR_OPS = ["greedy", "regret2", "regret3", "slack"]


def get_git_short_hash(repo_root: Path) -> str:
    """Get short git commit hash; empty string if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def make_version_string(repo_root: Path) -> str:
    """Build model_version string: UTC timestamp + git short hash."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash = get_git_short_hash(repo_root)
    return f"{ts}-{git_hash}" if git_hash else ts


def main():
    ap = argparse.ArgumentParser(description="Export RF models to ONNX")
    ap.add_argument("--models-dir", default="models",
                    help="Input directory with rf_pair_*.joblib + meta.json (default: models)")
    ap.add_argument("--out-dir", default="models/onnx",
                    help="Output directory for .onnx files + manifest (default: models/onnx)")
    args = ap.parse_args()

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("ERROR: skl2onnx not installed. Run: pip install -e '.[onnx]'",
              file=sys.stderr)
        sys.exit(1)

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load meta.json — produced by training.py. Carries feature_names (in the
    # exact training-time column order), label_columns, and the mode + batches
    # that were used to extract those features.
    meta_path = models_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    label_columns = meta["label_columns"]
    mode = meta.get("mode")
    batches = meta.get("batches", [])

    # Pull each feature's batch + mode + tolerance straight from the registry —
    # single source of truth, can't drift from training because both reference
    # the same _REGISTRY.
    from darpml.features import _REGISTRY  # noqa: PLC0415
    feature_meta = []
    for name in feature_names:
        if name not in _REGISTRY:
            sys.exit(f"ERROR: feature {name!r} listed in meta.json is not "
                     f"registered. Did the registry change between training "
                     f"and export?")
        e = _REGISTRY[name]
        feature_meta.append({
            "name": name,
            "batch": e.batch,
            "mode": e.mode,
            "tolerance": e.tolerance,
        })

    # Verify we have 20 models in pair_{d}_{r} form
    n_d = len(DESTROY_OPS)
    n_r = len(REPAIR_OPS)
    expected = n_d * n_r
    assert len(label_columns) == expected, \
        f"Expected {expected} label columns, got {len(label_columns)}"

    print(f"Exporting {len(label_columns)} models")
    print(f"  Features (alphabetical): {feature_names}")
    print(f"  n_features: {len(feature_names)}")
    print(f"  Destroy ops: {DESTROY_OPS}")
    print(f"  Repair ops: {REPAIR_OPS}")

    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]

    # Convert each model; name files by operator pair name (not index)
    for col in label_columns:
        # Parse pair_{d}_{r}
        parts = col.split("_")
        d_idx = int(parts[1])
        r_idx = int(parts[2])
        d_name = DESTROY_OPS[d_idx]
        r_name = REPAIR_OPS[r_idx]

        joblib_path = models_dir / f"rf_{col}.joblib"
        if not joblib_path.exists():
            print(f"  SKIP {col}: {joblib_path} not found")
            continue
        clf = joblib.load(joblib_path)

        # Flat output (no ZipMap dict) so C++ gets a tensor.
        # target_opset pinned to 21 to match onnxruntime 1.19.2 support.
        onnx_model = convert_sklearn(
            clf, initial_types=initial_type,
            options={id(clf): {"zipmap": False}},
            target_opset=21,
        )
        onnx_path = out_dir / f"{d_name}_{r_name}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  {col} -> {onnx_path.name}")

    # Write manifest. Emits NEW format (mode + batches + features as objects)
    # AND keeps the legacy feature_order (bare strings) so any older C++ build
    # using load_manifest's legacy path keeps working.
    # CRITICAL: order in `features` and `feature_order` must match feature_names
    # exactly — that's the order the RandomForests were trained with (DataFrame
    # column order = registry insertion order at training time).
    if mode is None:
        sys.exit("ERROR: meta.json has no 'mode' — retrain with the latest "
                 "darpml CLI which writes mode/batches into meta.json.")
    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "mode": mode,
        "batches": batches,
        "features": feature_meta,
        "feature_order": list(feature_names),  # legacy compat
        "n_features": len(feature_names),
        "model_version": make_version_string(repo_root),
        "destroy_ops": DESTROY_OPS,
        "repair_ops": REPAIR_OPS,
    }
    manifest_path = out_dir / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Model version: {manifest['model_version']}")
    print(f"Manifest mode: {mode}, batches: {batches if batches else 'all'}")


if __name__ == "__main__":
    main()
