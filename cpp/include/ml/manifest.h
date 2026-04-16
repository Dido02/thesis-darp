#pragma once

#include <string>
#include <vector>

// Manifest binding Python-side feature/model decisions to C++ runtime.
// Source of truth: ml_train/models/onnx/feature_manifest.json
struct FeatureManifest {
    std::vector<std::string> feature_order;  // alphabetical, defines ONNX input tensor order
    int n_features = 0;
    std::string model_version;               // for logging / debugging
    std::vector<std::string> destroy_ops;    // destroy op names in training pair index order
    std::vector<std::string> repair_ops;     // repair op names in training pair index order
};

// Load and parse the manifest JSON. Throws std::runtime_error on any failure
// (file missing, invalid JSON, missing fields, inconsistent n_features).
FeatureManifest load_manifest(const std::string& path);
