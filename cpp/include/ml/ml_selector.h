#pragma once

#include "instance.h"
#include "ml/alns_state.h"
#include "ml/manifest.h"
#include "solution.h"

#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Forward-declare Ort types so callers don't need onnxruntime headers.
namespace Ort {
class Env;
class Session;
class MemoryInfo;
}  // namespace Ort

// ML-driven per-pair operator selector.
// Loads one ONNX session per active (destroy, repair) pair, then samples a pair
// each iteration by running all sessions and weighting by P(good).
class MLSelector {
public:
    // model_dir: directory containing feature_manifest.json + {d}_{r}.onnx files.
    // active_destroy / active_repair: names of operators currently registered in
    //   the solver. Sessions loaded only for the intersection with the manifest.
    // registry_destroy_index / registry_repair_index: map operator name -> current
    //   registry index, used to translate selected (d,r) name pair back into
    //   registry-facing indices.
    MLSelector(
        const std::string& model_dir,
        const std::vector<std::string>& active_destroy,
        const std::vector<std::string>& active_repair,
        const std::map<std::string, int>& registry_destroy_index,
        const std::map<std::string, int>& registry_repair_index,
        const std::string& prob_log_path = "",
        std::optional<float> softmax_temp = std::nullopt);

    ~MLSelector();

    // Run 20 (or fewer, per subset) inferences; sample a pair; return registry indices.
    std::pair<int, int> select_pair(
        const DARPSolution& sol,
        const DARPInstance& inst,
        const ALNSState& st,
        std::mt19937& rng);

    // Introspection
    const std::string& model_version() const { return manifest_.model_version; }
    int n_active_pairs() const { return static_cast<int>(pair_order_.size()); }
    int n_features() const { return manifest_.n_features; }

private:
    FeatureManifest manifest_;

    // Session + index maps keyed by (d_name, r_name). unique_ptr so we can
    // forward-declare Ort::Session.
    using PairKey = std::pair<std::string, std::string>;
    std::map<PairKey, std::unique_ptr<Ort::Session>> sessions_;
    std::map<PairKey, std::pair<int, int>> pair_to_reg_idx_;

    // Stable ordering of keys for sampling (from map iteration).
    std::vector<PairKey> pair_order_;

    // Ort env is owned here so sessions can reference it.
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    // Optional per-iteration probability logging. When prob_log_path is non-empty,
    // we open a JSONL file in the ctor and write one record per select_pair() call.
    // call_idx_ is a cumulative counter across epochs (st.iter resets per epoch
    // because solve_warm() is called fresh each epoch).
    std::ofstream prob_log_;
    long long call_idx_ = 0;

    // Optional softmax-with-temperature picker. When set, replaces the default
    // proportional sampler in select_pair(): weights = exp((p - max_p) / T).
    // T<1 sharpens contrast; T>1 flattens; default (nullopt) keeps the existing
    // proportional sampler so bench results without the flag are byte-identical.
    std::optional<float> softmax_temp_;

    // Reusable scratch buffer for softmax weights. Sized once in the ctor so
    // select_pair() avoids per-call heap allocation (which polluted caches and
    // ate ~2-4% of iterations even under --exclude-selection-from-budget).
    std::vector<double> softmax_weights_;
};
