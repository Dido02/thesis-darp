// Parity test: reads parity_states.jsonl + feature_manifest.json, then for each
// state reconstructs a DARPSolution in C++, computes all manifest features, and
// compares them against the Python-computed values stored in the fixture.
//
// Usage:
//   test_feature_parity <parity_states.jsonl> <onnx_model_dir>
//
// The ONNX model dir is only used to load feature_manifest.json. The test does
// NOT require onnxruntime (the feature_extractor is a core library component).

#include "instance.h"
#include "ml/alns_state.h"
#include "ml/feature_extractor.h"
#include "ml/manifest.h"
#include "parsers/cordeau.h"
#include "parsers/fiedler.h"
#include "parsers/li_lim.h"
#include "solution.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr double TOLERANCE = 1e-6;


static DARPInstance load_any_instance(const std::string& path) {
    bool is_fiedler = path.find("fiedler") != std::string::npos;
    bool is_li_lim = path.find("li_lim") != std::string::npos;
    if (is_fiedler) return parse_fiedler(path);
    if (is_li_lim) return parse_li_lim(path);
    return parse_cordeau(path);
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <parity_states.jsonl> <onnx_model_dir>\n";
        return 1;
    }
    const std::string fixture_path = argv[1];
    const std::string model_dir = argv[2];

    // Load manifest (source of truth for which features to compare)
    FeatureManifest manifest;
    try {
        manifest = load_manifest((fs::path(model_dir) / "feature_manifest.json").string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load manifest: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Manifest features (n=" << manifest.n_features << "): ";
    for (const auto& n : manifest.feature_order) std::cout << n << " ";
    std::cout << "\n";

    // Validate all manifest features are registered in C++
    const auto& reg = FeatureRegistry::instance();
    for (const auto& name : manifest.feature_order) {
        if (!reg.has(name)) {
            std::cerr << "ERROR: C++ feature registry missing: " << name << "\n";
            return 1;
        }
    }

    std::ifstream f(fixture_path);
    if (!f) {
        std::cerr << "Cannot open fixture: " << fixture_path << "\n";
        return 1;
    }

    // Cache instances by path to avoid re-parsing.
    std::string cached_instance_path;
    DARPInstance cached_instance{};

    int total_states = 0;
    int failed_states = 0;
    double max_abs_err = 0.0;
    std::string max_err_feature;
    int max_err_state = -1;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json rec = json::parse(line);

        int state_id = rec["state_id"];
        std::string instance_path = rec["instance_path"];

        if (instance_path != cached_instance_path) {
            cached_instance = load_any_instance(instance_path);
            cached_instance_path = instance_path;
        }

        // Reconstruct DARPSolution
        DARPSolution sol(cached_instance);
        for (const auto& r : rec["routes"]) {
            sol.routes.push_back(r.get<std::vector<int>>());
        }
        for (int u : rec["unassigned"]) sol.unassigned.insert(u);
        for (int c : rec["committed"]) sol.committed.insert(c);
        for (int m : rec["min_insert_pos"]) sol.min_insert_pos.push_back(m);

        ALNSState st{
            rec["temperature"].get<double>(),
            rec["sa_start_temp"].get<double>(),
        };

        auto cpp_feats = extract_all_features(sol, cached_instance, st);

        // Compare each manifest feature
        bool state_ok = true;
        for (const auto& name : manifest.feature_order) {
            if (!rec["features"].contains(name)) {
                std::cerr << "  state " << state_id << ": fixture missing feature "
                          << name << "\n";
                state_ok = false;
                continue;
            }
            double py_val = rec["features"][name].get<double>();
            double cpp_val = cpp_feats.at(name);
            double err = std::abs(py_val - cpp_val);

            if (err > max_abs_err) {
                max_abs_err = err;
                max_err_feature = name;
                max_err_state = state_id;
            }
            if (err > TOLERANCE) {
                std::cerr << "  FAIL state=" << state_id << " feature=" << name
                          << " python=" << py_val << " cpp=" << cpp_val
                          << " err=" << err << "\n";
                state_ok = false;
            }
        }
        if (!state_ok) ++failed_states;
        ++total_states;
    }

    std::cout << "\n";
    if (failed_states == 0) {
        std::cout << "PASS: all " << manifest.feature_order.size()
                  << " features match on " << total_states << "/" << total_states
                  << " states (max_abs_err=" << max_abs_err
                  << " in feature=" << max_err_feature
                  << " state=" << max_err_state << ")\n";
        return 0;
    } else {
        std::cout << "FAIL: " << failed_states << "/" << total_states
                  << " states had mismatches (max_abs_err=" << max_abs_err
                  << " in feature=" << max_err_feature
                  << " state=" << max_err_state << ")\n";
        return 1;
    }
}
