#pragma once

#include "ml/feature_extractor.h"

#include <string>
#include <vector>

// Per-feature manifest entry.
struct ManifestFeature {
    std::string name;
    std::string batch;       // empty when loaded from a legacy manifest
    FeatureMode mode;        // Both when loaded from a legacy manifest (i.e. don't enforce)
    double tolerance;        // 0.0 default if not specified in manifest
};

// Manifest binding Python-side feature/model decisions to C++ runtime.
// Source of truth: ml_train/models/onnx/feature_manifest.json
//
// Two formats are accepted:
// - LEGACY: only `feature_order` (list of strings) is present. We load names in
//   the order given and DO NOT validate batch/mode tags against the registry.
//   This preserves the original training-time order so existing ONNX models
//   keep working without retraining.
// - NEW: `mode`, `batches`, and `features` (list of objects) are present. Each
//   feature must exist in the registry with matching batch and mode tags.
struct FeatureManifest {
    std::vector<ManifestFeature> features;   // ordered as ONNX expects
    int n_features = 0;
    std::string model_version;
    std::vector<std::string> destroy_ops;
    std::vector<std::string> repair_ops;

    // Set by parser. New-format manifests carry tag metadata; legacy ones don't.
    bool is_legacy = false;

    // Set from manifest's "mode" / "batches" keys (NEW format only). For legacy
    // manifests, these stay at defaults (Both / empty) and aren't used.
    FeatureMode mode = FeatureMode::Both;
    std::vector<std::string> batches;

    // Convenience: ordered list of names (matches the order of `features`).
    std::vector<std::string> feature_names() const;
};

// Load and parse the manifest JSON. Throws std::runtime_error on any failure
// (file missing, invalid JSON, missing required fields, inconsistent counts).
FeatureManifest load_manifest(const std::string& path);
