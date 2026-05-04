#include "ml/ml_selector.h"

#include "ml/feature_extractor.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Ctor: load manifest, validate features, intersect active ops, load sessions.
// ---------------------------------------------------------------------------
MLSelector::MLSelector(
    const std::string& model_dir,
    const std::vector<std::string>& active_destroy,
    const std::vector<std::string>& active_repair,
    const std::map<std::string, int>& registry_destroy_index,
    const std::map<std::string, int>& registry_repair_index,
    const std::string& prob_log_path,
    std::optional<float> softmax_temp) {

    softmax_temp_ = softmax_temp;

    fs::path root(model_dir);
    manifest_ = load_manifest((root / "feature_manifest.json").string());

    // 1. Validate manifest features against the registry. LEGACY manifests
    //    only check existence; NEW manifests additionally enforce batch/mode tags.
    const auto& reg = FeatureRegistry::instance();
    auto validation = validate_manifest_against_registry(manifest_, reg);
    if (!validation.missing_features.empty()) {
        std::ostringstream oss;
        oss << "ML manifest requires features not registered in C++: [";
        for (size_t i = 0; i < validation.missing_features.size(); ++i) {
            if (i) oss << ", ";
            oss << validation.missing_features[i];
        }
        oss << "]. Add them to cpp/src/ml/feature_extractor.cpp with REGISTER_FEATURE_V2.";
        throw std::runtime_error(oss.str());
    }
    if (!validation.tag_mismatches.empty()) {
        std::ostringstream oss;
        oss << "ML manifest features have batch/mode tags that don't match the C++ registry:\n";
        for (const auto& s : validation.tag_mismatches) oss << "  " << s << "\n";
        throw std::runtime_error(oss.str());
    }

    // 2. Compute intersection of active operators with manifest.
    //    Every active pair must have a corresponding model file.
    std::set<std::string> manifest_destroy(
        manifest_.destroy_ops.begin(), manifest_.destroy_ops.end());
    std::set<std::string> manifest_repair(
        manifest_.repair_ops.begin(), manifest_.repair_ops.end());

    std::vector<PairKey> missing_pairs;
    std::vector<PairKey> to_load;
    for (const auto& d_name : active_destroy) {
        for (const auto& r_name : active_repair) {
            bool in_manifest = manifest_destroy.count(d_name) && manifest_repair.count(r_name);
            fs::path onnx_path = root / (d_name + "_" + r_name + ".onnx");
            if (!in_manifest || !fs::exists(onnx_path)) {
                missing_pairs.emplace_back(d_name, r_name);
            } else {
                to_load.emplace_back(d_name, r_name);
            }
        }
    }
    if (!missing_pairs.empty()) {
        std::ostringstream oss;
        oss << "ML selection requires models for all active operator pairs.\n"
            << "Missing: [";
        for (size_t i = 0; i < missing_pairs.size(); ++i) {
            if (i) oss << ", ";
            oss << "(\"" << missing_pairs[i].first << "\",\""
                << missing_pairs[i].second << "\")";
        }
        oss << "]\nEither retrain with these operators or exclude them from "
            << "--destroy / --repair.";
        throw std::runtime_error(oss.str());
    }

    // 3. Initialize Ort environment + memory info.
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "darp_ml_selector");
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // 4. Load all sessions; build pair_to_reg_idx_ concurrently.
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    for (const auto& key : to_load) {
        fs::path onnx_path = root / (key.first + "_" + key.second + ".onnx");
        sessions_[key] = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), so);

        auto d_it = registry_destroy_index.find(key.first);
        auto r_it = registry_repair_index.find(key.second);
        if (d_it == registry_destroy_index.end() || r_it == registry_repair_index.end()) {
            throw std::runtime_error(
                "Internal error: operator " + key.first + "/" + key.second
                + " missing from registry index map");
        }
        pair_to_reg_idx_[key] = {d_it->second, r_it->second};
    }

    // Stable iteration order for sampling.
    for (const auto& [key, _] : sessions_) pair_order_.push_back(key);

    // Optional probability log: open file + write one header line listing
    // every active (destroy, repair) pair so the plot script can label columns.
    // Append mode because solve_warm() (and therefore this ctor) is called
    // fresh per epoch in dynamic mode — we want to accumulate records across
    // epochs into a single file. The header gets re-emitted each epoch (same
    // pair_order); the plot script dedupes.
    if (!prob_log_path.empty()) {
        prob_log_.open(prob_log_path, std::ios::app);
        if (!prob_log_) {
            throw std::runtime_error(
                "Failed to open --log-ml-probs file: " + prob_log_path);
        }
        prob_log_ << "{\"type\":\"header\",\"n_pairs\":" << pair_order_.size()
                  << ",\"pair_order\":[";
        for (size_t i = 0; i < pair_order_.size(); ++i) {
            if (i) prob_log_ << ",";
            prob_log_ << "[\"" << pair_order_[i].first << "\",\""
                      << pair_order_[i].second << "\"]";
        }
        prob_log_ << "]}\n";
    }
}

MLSelector::~MLSelector() = default;


// ---------------------------------------------------------------------------
// Run one session, return P(class=1) for the single input row.
// Our ONNX graphs emit two outputs: "label" (int64) and "probabilities" (float, shape=[1,2]).
// We read "probabilities" by name.
// ---------------------------------------------------------------------------
static float run_session_p_class1(
    Ort::Session& session,
    const Ort::MemoryInfo& memory_info,
    std::vector<float>& input_values) {

    const int64_t shape[2] = {1, static_cast<int64_t>(input_values.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(), shape, 2);

    // Input/output names: fetch dynamically to be robust across skl2onnx versions.
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, alloc);
    const char* input_names[] = {input_name_ptr.get()};

    // Find the output named "probabilities"; fall back to index 1.
    size_t n_out = session.GetOutputCount();
    int prob_idx = -1;
    std::vector<Ort::AllocatedStringPtr> out_names_owned;
    std::vector<const char*> out_names;
    out_names_owned.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        out_names_owned.push_back(session.GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_names_owned.back().get());
        if (std::string(out_names.back()) == "probabilities") prob_idx = static_cast<int>(i);
    }
    if (prob_idx < 0) {
        // Fallback: sklearn usually emits label as output 0, probabilities as output 1.
        prob_idx = (n_out > 1) ? 1 : 0;
    }

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        out_names.data(), out_names.size());

    // outputs[prob_idx] is a [1, 2] float tensor.
    const Ort::Value& prob = outputs[prob_idx];
    const float* data = prob.GetTensorData<float>();
    auto info = prob.GetTensorTypeAndShapeInfo();
    auto shape_vec = info.GetShape();

    // Guard: binary classifier has shape [1, 2] → class-1 prob at index 1.
    if (shape_vec.size() == 2 && shape_vec[1] >= 2) {
        return data[1];
    }
    // Degenerate classifier (single-class): treat output as "the class we saw".
    // If it's class 0, P(good) = 0; if it's class 1, P(good) = 1.
    return data[0];
}


// ---------------------------------------------------------------------------
// select_pair: extract features, build tensor, run all sessions, sample.
// ---------------------------------------------------------------------------
std::pair<int, int> MLSelector::select_pair(
    const DARPSolution& sol,
    const DARPInstance& inst,
    const ALNSState& st,
    std::mt19937& rng) {

    // 1. Extract only the manifest's feature subset, in the manifest's declared
    //    order. ctor validated names exist in the registry.
    auto values = FeatureRegistry::instance().extract_by_names(
        manifest_.feature_names(), sol, inst, st);

    // Cast double -> float at the ONNX tensor boundary. Features are computed in
    // double to match Python/numpy precision; ONNX graphs use float32 inputs.
    std::vector<float> input(manifest_.n_features);
    for (size_t i = 0; i < values.size(); ++i) {
        input[i] = static_cast<float>(values[i]);
    }

    // 2. Run inference for each loaded session in stable order.
    std::vector<float> p_good;
    p_good.reserve(pair_order_.size());
    for (const auto& key : pair_order_) {
        auto& session = *sessions_.at(key);
        p_good.push_back(run_session_p_class1(session, *memory_info_, input));
    }

    // 3. Sample. Two paths: (a) softmax-with-temperature when softmax_temp_ is
    //    set; (b) the existing proportional sampler when not. The default path
    //    (softmax_temp_ unset) is byte-identical to prior versions: same
    //    constants, same loop order, same RNG draw count.
    size_t chosen;

    if (softmax_temp_.has_value()) {
        // Softmax: weights[k] = exp((p_good[k] - max_p) / T).
        // Subtracting max_p stabilises against overflow (the resulting
        // distribution is unchanged because the subtraction cancels in the
        // ratio). T<1 sharpens contrast; T>1 flattens.
        const float T = *softmax_temp_;
        const float max_p = *std::max_element(p_good.begin(), p_good.end());
        std::vector<double> weights(p_good.size());
        double sum = 0.0;
        for (size_t i = 0; i < p_good.size(); ++i) {
            weights[i] = std::exp(static_cast<double>(p_good[i] - max_p) / T);
            sum += weights[i];
        }
        double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
        double cum = 0.0;
        chosen = weights.size() - 1;
        for (size_t k = 0; k < weights.size(); ++k) {
            cum += weights[k];
            if (r <= cum) { chosen = k; break; }
        }
    } else {
        // Existing proportional sampler — verbatim from prior versions.
        constexpr float MIN_P = 1e-6f;
        constexpr double MIN_SUM = 0.01;

        double sum = 0.0;
        for (float p : p_good) sum += std::max(p, MIN_P);

        if (sum < MIN_SUM) {
            // Degenerate — uniform over loaded pairs.
            chosen = std::uniform_int_distribution<size_t>(
                0, pair_order_.size() - 1)(rng);
        } else {
            double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
            double cum = 0.0;
            chosen = pair_order_.size() - 1;
            for (size_t k = 0; k < p_good.size(); ++k) {
                cum += std::max(p_good[k], MIN_P);
                if (r <= cum) { chosen = k; break; }
            }
        }
    }

    // Optional: log probabilities + chosen index for offline visualization.
    // Written AFTER sampling so the "picked" field is included. call_idx_ is
    // cumulative across epochs (solve_warm restarts st.iter to 0 each epoch).
    if (prob_log_.is_open()) {
        prob_log_ << "{\"call_idx\":" << call_idx_
                  << ",\"epoch_iter\":" << st.iter
                  << ",\"sim_clock\":" << st.sim_clock
                  << ",\"probs\":[";
        for (size_t i = 0; i < p_good.size(); ++i) {
            if (i) prob_log_ << ",";
            prob_log_ << p_good[i];
        }
        prob_log_ << "],\"picked\":" << chosen << "}\n";
        ++call_idx_;
    }

    return pair_to_reg_idx_.at(pair_order_[chosen]);
}
