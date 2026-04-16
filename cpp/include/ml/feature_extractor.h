#pragma once

#include "instance.h"
#include "ml/alns_state.h"
#include "solution.h"

#include <string>
#include <unordered_map>
#include <vector>

// Feature function signature: pure function of (solution, instance, ALNS state) -> double.
// Return double (not float) so we preserve full precision; the ONNX tensor-build
// step casts to float32 at its boundary.
using FeatureFunc = double (*)(const DARPSolution&, const DARPInstance&, const ALNSState&);

// Process-wide feature registry. Populated at static-init via REGISTER_FEATURE.
class FeatureRegistry {
public:
    static FeatureRegistry& instance();

    void add(const std::string& name, FeatureFunc fn);

    // Names of all registered features (not sorted).
    std::vector<std::string> names() const;

    // Run all registered features on the given state, return name-keyed map.
    std::unordered_map<std::string, double> extract_all(
        const DARPSolution& sol,
        const DARPInstance& inst,
        const ALNSState& st) const;

    // Check if a feature is registered.
    bool has(const std::string& name) const;

private:
    std::unordered_map<std::string, FeatureFunc> fns_;
};

// Convenience free function — delegates to the singleton.
inline std::unordered_map<std::string, double> extract_all_features(
    const DARPSolution& sol,
    const DARPInstance& inst,
    const ALNSState& st) {
    return FeatureRegistry::instance().extract_all(sol, inst, st);
}

// Register a feature at static-init. The unique struct name prevents collisions.
#define REGISTER_FEATURE(NAME, FN)                                  \
    namespace {                                                     \
    struct FN##_registrar {                                         \
        FN##_registrar() {                                          \
            FeatureRegistry::instance().add(NAME, FN);              \
        }                                                           \
    };                                                              \
    static FN##_registrar FN##_registrar_instance;                  \
    }
