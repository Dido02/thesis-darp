// Parity test: reads parity_states.jsonl + feature_manifest.json, then for each
// state reconstructs a DARPSolution in C++, computes all manifest features, and
// compares them against the Python-computed values stored in the fixture.
//
// Per-feature tolerance: each fixture record's `feature_tolerances` map declares
// the max-abs-error allowed per feature (registry-owned: 0.0 for sums/ratios,
// 1e-12 for variance/std-dev). Falls back to 1e-12 if not declared.
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

// Fallback tolerance when a fixture record doesn't declare one for a feature.
constexpr double FALLBACK_TOLERANCE = 1e-12;


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

    auto names = manifest.feature_names();
    std::cout << "Manifest features (n=" << manifest.n_features
              << ", " << (manifest.is_legacy ? "legacy" : "new")
              << " format): ";
    for (const auto& n : names) std::cout << n << " ";
    std::cout << "\n";

    // Validate all manifest features against the registry (existence + tags).
    const auto& reg = FeatureRegistry::instance();
    auto validation = validate_manifest_against_registry(manifest, reg);
    if (!validation.ok()) {
        if (!validation.missing_features.empty()) {
            std::cerr << "ERROR: C++ registry missing features: ";
            for (const auto& n : validation.missing_features) std::cerr << n << " ";
            std::cerr << "\n";
        }
        if (!validation.tag_mismatches.empty()) {
            std::cerr << "ERROR: manifest tag mismatches:\n";
            for (const auto& s : validation.tag_mismatches) std::cerr << "  " << s << "\n";
        }
        return 1;
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

    // Direct scheduler-parity tracking. Tolerance is exact zero — the scheduler
    // is deterministic and the Python port mirrors it line-for-line.
    constexpr double SCHED_TOLERANCE = 0.0;
    int sched_states_checked = 0;
    int sched_states_failed = 0;
    int sched_routes_checked = 0;
    int sched_requests_checked = 0;
    double sched_max_err = 0.0;

    // Objective-parity tracking. C++ objective() and Python compute_objective
    // both iterate routes outer / edges inner. Should match bit-for-bit on
    // Fiedler (file-loaded tt) and on Cordeau (after Part 3 fma fix).
    constexpr double OBJ_TOLERANCE = 0.0;
    int obj_states_checked = 0;
    int obj_states_failed = 0;
    double obj_max_err = 0.0;

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

        // Reconstruct DARPSolution. The default ctor pre-fills `routes` with
        // K depot+vstart+depot empties and inserts 1..n into `unassigned` —
        // we replace both with the fixture state so indices line up exactly.
        DARPSolution sol(cached_instance);
        sol.routes.clear();
        sol.unassigned.clear();
        for (const auto& r : rec["routes"]) {
            sol.routes.push_back(r.get<std::vector<int>>());
        }
        for (int u : rec["unassigned"]) sol.unassigned.insert(u);
        for (int c : rec["committed"]) sol.committed.insert(c);
        for (int m : rec["min_insert_pos"]) sol.min_insert_pos.push_back(m);

        ALNSState st{
            rec["temperature"].get<double>(),
            rec["sa_start_temp"].get<double>(),
            rec.value("q", 0),
            rec.value("iter", 0),
            rec.value("max_iterations", 0),
            rec.value("obj_initial", 0.0),
            rec.value("obj_best", 0.0),
            rec.value("sim_clock", 0.0),
            rec.value("epoch_length", 0.0),
        };

        auto cpp_feats = extract_all_features(sol, cached_instance, st);

        // Compare each manifest feature using its per-feature tolerance.
        const bool has_tol_map = rec.contains("feature_tolerances");
        bool state_ok = true;
        for (const auto& name : names) {
            if (!rec["features"].contains(name)) {
                std::cerr << "  state " << state_id << ": fixture missing feature "
                          << name << "\n";
                state_ok = false;
                continue;
            }
            double py_val = rec["features"][name].get<double>();
            double cpp_val = cpp_feats.at(name);
            double err = std::abs(py_val - cpp_val);

            double tol = FALLBACK_TOLERANCE;
            if (has_tol_map && rec["feature_tolerances"].contains(name)) {
                tol = rec["feature_tolerances"][name].get<double>();
            }

            if (err > max_abs_err) {
                max_abs_err = err;
                max_err_feature = name;
                max_err_state = state_id;
            }
            if (err > tol) {
                std::cerr << "  FAIL state=" << state_id << " feature=" << name
                          << " python=" << py_val << " cpp=" << cpp_val
                          << " err=" << err << " tol=" << tol << "\n";
                state_ok = false;
            }
        }
        if (!state_ok) ++failed_states;
        ++total_states;

        // ---- Direct scheduler parity ----
        // For each route, recompute the schedule in C++ and compare its
        // ride_time map against the Python-computed map carried in the fixture.
        // Verifies the scheduler across ALL requests (not just non-committed,
        // which is all the ride_time_utilization_* features see).
        if (rec.contains("route_schedules")) {
            ++sched_states_checked;
            bool sched_state_ok = true;
            const auto& py_schedules = rec["route_schedules"];
            if (py_schedules.size() != sol.routes.size()) {
                std::cerr << "  state " << state_id
                          << ": route_schedules count mismatch (py="
                          << py_schedules.size() << " cpp=" << sol.routes.size() << ")\n";
                sched_state_ok = false;
            }
            for (size_t r = 0; r < sol.routes.size() && r < py_schedules.size(); ++r) {
                ++sched_routes_checked;
                auto cpp_sched = sol.compute_route_schedule(sol.routes[r]);
                const auto& py_dict = py_schedules[r];
                if (cpp_sched.ride_time.size() != py_dict.size()) {
                    std::cerr << "  state " << state_id << " route " << r
                              << ": ride_time count mismatch (py=" << py_dict.size()
                              << " cpp=" << cpp_sched.ride_time.size() << ")\n";
                    sched_state_ok = false;
                    continue;
                }
                for (auto it = py_dict.begin(); it != py_dict.end(); ++it) {
                    int req = std::stoi(it.key());
                    double py_rt = it.value().get<double>();
                    auto cit = cpp_sched.ride_time.find(req);
                    if (cit == cpp_sched.ride_time.end()) {
                        std::cerr << "  state " << state_id << " route " << r
                                  << ": req " << req << " missing in C++ schedule\n";
                        sched_state_ok = false;
                        continue;
                    }
                    ++sched_requests_checked;
                    double err = std::abs(py_rt - cit->second);
                    if (err > sched_max_err) sched_max_err = err;
                    if (err > SCHED_TOLERANCE) {
                        std::cerr << "  SCHED FAIL state=" << state_id << " route=" << r
                                  << " req=" << req
                                  << " py=" << py_rt << " cpp=" << cit->second
                                  << " err=" << err << "\n";
                        sched_state_ok = false;
                    }
                }
            }
            if (!sched_state_ok) ++sched_states_failed;
        }

        // ---- Objective parity ----
        // Compares C++ DARPSolution::objective() to fixture's "objective_py".
        // Fixture only carries it if the build script computed it.
        if (rec.contains("objective_py")) {
            ++obj_states_checked;
            double py_obj = rec["objective_py"].get<double>();
            // Use the fixture's penalty (synthetic uses default 1000;
            // trajectory uses header.unassigned_penalty).
            double penalty = rec.value("unassigned_penalty", 1000.0);
            sol.unassigned_penalty = penalty;
            sol.cached_obj_ = -1.0;  // force recompute
            double cpp_obj = sol.objective();
            double err = std::abs(py_obj - cpp_obj);
            if (err > obj_max_err) obj_max_err = err;
            if (err > OBJ_TOLERANCE) {
                std::cerr << "  OBJ FAIL state=" << state_id
                          << " python=" << py_obj << " cpp=" << cpp_obj
                          << " err=" << err << "\n";
                ++obj_states_failed;
            }
        }
    }

    std::cout << "\n";
    bool feats_ok = (failed_states == 0);
    bool sched_ok = (sched_states_failed == 0);
    bool obj_ok = (obj_states_failed == 0);

    if (sched_states_checked > 0) {
        if (sched_ok) {
            std::cout << "SCHED PASS: " << sched_states_checked << " states, "
                      << sched_routes_checked << " routes, "
                      << sched_requests_checked << " requests "
                      << "(max_abs_err=" << sched_max_err << ")\n";
        } else {
            std::cout << "SCHED FAIL: " << sched_states_failed << "/"
                      << sched_states_checked << " states had mismatches "
                      << "(max_abs_err=" << sched_max_err << ")\n";
        }
    } else {
        std::cout << "SCHED SKIP: fixture has no route_schedules field\n";
    }

    if (obj_states_checked > 0) {
        if (obj_ok) {
            std::cout << "OBJ PASS: " << obj_states_checked << " states "
                      << "(max_abs_err=" << obj_max_err << ")\n";
        } else {
            std::cout << "OBJ FAIL: " << obj_states_failed << "/"
                      << obj_states_checked << " states had mismatches "
                      << "(max_abs_err=" << obj_max_err << ")\n";
        }
    }

    if (feats_ok && sched_ok && obj_ok) {
        std::cout << "PASS: all " << names.size()
                  << " features match on " << total_states << "/" << total_states
                  << " states (max_abs_err=" << max_abs_err
                  << " in feature=" << max_err_feature
                  << " state=" << max_err_state << ")\n";
        return 0;
    } else {
        if (!feats_ok) {
            std::cout << "FAIL: " << failed_states << "/" << total_states
                      << " states had feature mismatches (max_abs_err=" << max_abs_err
                      << " in feature=" << max_err_feature
                      << " state=" << max_err_state << ")\n";
        }
        return 1;
    }
}
