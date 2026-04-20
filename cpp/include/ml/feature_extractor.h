#pragma once

#include "instance.h"
#include "ml/alns_state.h"
#include "solution.h"

#include <string>
#include <unordered_map>
#include <vector>

struct FeatureManifest;  // fwd-decl for validate_manifest_against_registry

// Feature function signature: pure function of (solution, instance, ALNS state) -> double.
// Return double (not float) so we preserve full precision; the ONNX tensor-build
// step casts to float32 at its boundary.
using FeatureFunc = double (*)(const DARPSolution&, const DARPInstance&, const ALNSState&);

enum class FeatureMode { Static, Dynamic, Both };

FeatureMode parse_feature_mode(const std::string& s);
const char* feature_mode_name(FeatureMode m);

struct FeatureEntry {
    std::string name;
    FeatureFunc fn;
    std::string batch;
    FeatureMode mode;
    double tolerance;   // parity threshold; 0.0 for sums/ratios, 1e-12 for variance
};

// Process-wide feature registry. Populated at static-init via REGISTER_FEATURE_V2.
// Insertion order is preserved so the C++ ordering can mirror Python's
// registry-insertion order (which is what training uses).
class FeatureRegistry {
public:
    static FeatureRegistry& instance();

    void add(FeatureEntry entry);

    // Names of all registered features in insertion order.
    std::vector<std::string> names() const;

    // Lookup by name. Returns nullptr if not registered.
    const FeatureEntry* find(const std::string& name) const;

    // True if a feature with this name is registered.
    bool has(const std::string& name) const;

    // Run all registered features on the given state, return name-keyed map.
    std::unordered_map<std::string, double> extract_all(
        const DARPSolution& sol,
        const DARPInstance& inst,
        const ALNSState& st) const;

    // Run a specific named subset (in the order given). Throws std::runtime_error
    // if any name is not registered.
    std::vector<double> extract_by_names(
        const std::vector<std::string>& names,
        const DARPSolution& sol,
        const DARPInstance& inst,
        const ALNSState& st) const;

    // Filtered subset: features whose batch is in `batches` (or no filter if
    // batches is empty) AND whose mode matches `mode` ("Both" matches Static
    // or Dynamic). Returned in insertion order.
    std::vector<const FeatureEntry*> get_filtered(
        const std::vector<std::string>& batches,
        FeatureMode mode) const;

private:
    std::unordered_map<std::string, size_t> name_to_index_;
    std::vector<FeatureEntry> entries_;
};

// Result of validating a manifest against the registry.
// `tag_mismatches` is only populated when the manifest is in NEW format
// (legacy manifests skip the tag check, per spec).
struct ManifestValidation {
    std::vector<std::string> missing_features;  // names not in registry
    std::vector<std::string> tag_mismatches;    // pre-formatted "name (manifest:... registry:...)"
    bool ok() const { return missing_features.empty() && tag_mismatches.empty(); }
};

ManifestValidation validate_manifest_against_registry(
    const FeatureManifest& manifest,
    const class FeatureRegistry& registry);

// Convenience free function — delegates to the singleton.
inline std::unordered_map<std::string, double> extract_all_features(
    const DARPSolution& sol,
    const DARPInstance& inst,
    const ALNSState& st) {
    return FeatureRegistry::instance().extract_all(sol, inst, st);
}

// Register a feature at static-init. The unique struct name prevents collisions.
// MODE_STR must be one of "static", "dynamic", "both". batch/mode/tolerance are
// declared explicitly at every call site — no defaults, no silent fallbacks.
#define REGISTER_FEATURE_V2(NAME, BATCH, MODE_STR, TOLERANCE, FN)            \
    namespace {                                                              \
    struct FN##_registrar {                                                  \
        FN##_registrar() {                                                   \
            FeatureRegistry::instance().add(FeatureEntry{                    \
                NAME, FN, BATCH, parse_feature_mode(MODE_STR), TOLERANCE});  \
        }                                                                    \
    };                                                                       \
    static FN##_registrar FN##_registrar_instance;                           \
    }
