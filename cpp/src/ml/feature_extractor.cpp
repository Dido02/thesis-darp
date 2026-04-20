// =============================================================================
// ML feature implementations. Each feature is registered with REGISTER_FEATURE_V2
// declaring (name, batch, mode, tolerance, fn). Mirror in
// ml_train/darpml/features.py. Adding a feature requires both sides + parity
// fixture regeneration; see CLAUDE.md "How to add a batch".
//
// 20 features across 5 batches:
//
//   feature                          batch            mode       tol     source
//   -------------------------------- ---------------- ---------- ------- ----------
//   num_active_requests              core             both       0.0     novel
//   radius_avg_tt                    core             both       0.0     Bongiovanni #3
//   frac_committed                   core             dynamic    0.0     novel
//   avg_route_load                   core             both       0.0     Bongiovanni #26
//   temperature_normalized           core             both       0.0     novel
//   ride_time_utilization_avg        tightness        both       0.0     novel
//   ride_time_utilization_max        tightness        both       0.0     novel
//   near_violation_count             tightness        both       0.0     novel
//   route_slack_avg                  tightness        both       0.0     novel
//   route_slack_std                  tightness        both       1e-12   novel
//   cost_contribution_std            worst_support    both       1e-12   novel
//   cost_contribution_max_ratio      worst_support    both       0.0     novel
//   route_cost_std                   worst_support    both       1e-12   novel
//   neighborhood_size_frac           lns_state        both       0.0     Bongiovanni #41
//   cost_vs_initial_frac             lns_state        both       0.0     Bongiovanni #42
//   cost_vs_incumbent_frac           lns_state        both       0.0     Bongiovanni #43
//   iteration_norm                   lns_state        both       0.0     Bongiovanni #46
//   unassigned_count                 lns_state        both       0.0     novel
//   committed_travel_time_frac       rolling_horizon  dynamic    0.0     novel
//   horizon_elapsed_frac             rolling_horizon  dynamic    0.0     Bongiovanni #2
//
// `flexibility_horizon` was removed after diagnostic — see THESIS_NOTES.md.
//
// Bongiovanni numbering follows their 2022 paper's 46-feature catalogue.
// `mode` semantics: features tagged `dynamic` are stripped from static-mode
// model manifests; `both` features apply to either. Per-state dynamic detection
// in features that fork on it (e.g. neighborhood_size_frac) uses
// `!sol.committed.empty()`, NOT the registry mode tag.
// =============================================================================

#include "ml/feature_extractor.h"

#include "ml/manifest.h"
#include "operators/cost_helpers.h"
#include "operators/slack_helpers.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// FeatureMode helpers
// ---------------------------------------------------------------------------
FeatureMode parse_feature_mode(const std::string& s) {
    if (s == "static") return FeatureMode::Static;
    if (s == "dynamic") return FeatureMode::Dynamic;
    if (s == "both") return FeatureMode::Both;
    throw std::runtime_error("Invalid feature mode: " + s
        + " (expected static|dynamic|both)");
}

const char* feature_mode_name(FeatureMode m) {
    switch (m) {
        case FeatureMode::Static:  return "static";
        case FeatureMode::Dynamic: return "dynamic";
        case FeatureMode::Both:    return "both";
    }
    return "unknown";
}

// True if a feature whose declared mode is `feat_mode` should be included
// when caller requests `requested`. "Both" matches either requested side.
static bool mode_matches(FeatureMode feat_mode, FeatureMode requested) {
    if (feat_mode == FeatureMode::Both) return true;
    return feat_mode == requested;
}

// ---------------------------------------------------------------------------
// Registry implementation
// ---------------------------------------------------------------------------
FeatureRegistry& FeatureRegistry::instance() {
    static FeatureRegistry r;
    return r;
}

void FeatureRegistry::add(FeatureEntry entry) {
    if (name_to_index_.count(entry.name)) {
        throw std::runtime_error("Feature already registered: " + entry.name);
    }
    name_to_index_[entry.name] = entries_.size();
    entries_.push_back(std::move(entry));
}

std::vector<std::string> FeatureRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& e : entries_) out.push_back(e.name);
    return out;
}

const FeatureEntry* FeatureRegistry::find(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) return nullptr;
    return &entries_[it->second];
}

bool FeatureRegistry::has(const std::string& name) const {
    return name_to_index_.find(name) != name_to_index_.end();
}

std::unordered_map<std::string, double> FeatureRegistry::extract_all(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState& st) const {
    std::unordered_map<std::string, double> result;
    result.reserve(entries_.size());
    for (const auto& e : entries_) {
        result.emplace(e.name, e.fn(sol, inst, st));
    }
    return result;
}

std::vector<double> FeatureRegistry::extract_by_names(
    const std::vector<std::string>& names,
    const DARPSolution& sol,
    const DARPInstance& inst,
    const ALNSState& st) const {
    std::vector<double> out;
    out.reserve(names.size());
    for (const auto& n : names) {
        const FeatureEntry* e = find(n);
        if (!e) {
            throw std::runtime_error("extract_by_names: feature not registered: " + n);
        }
        out.push_back(e->fn(sol, inst, st));
    }
    return out;
}

ManifestValidation validate_manifest_against_registry(
    const FeatureManifest& manifest, const FeatureRegistry& registry) {
    ManifestValidation v;
    for (const auto& mf : manifest.features) {
        const FeatureEntry* entry = registry.find(mf.name);
        if (!entry) {
            v.missing_features.push_back(mf.name);
            continue;
        }
        if (manifest.is_legacy) continue;  // tags unknown on legacy path
        if (entry->batch != mf.batch || entry->mode != mf.mode) {
            std::ostringstream oss;
            oss << mf.name
                << " (manifest: batch=" << mf.batch
                << ",mode=" << feature_mode_name(mf.mode)
                << "; registry: batch=" << entry->batch
                << ",mode=" << feature_mode_name(entry->mode) << ")";
            v.tag_mismatches.push_back(oss.str());
        }
    }
    return v;
}


std::vector<const FeatureEntry*> FeatureRegistry::get_filtered(
    const std::vector<std::string>& batches,
    FeatureMode mode) const {
    std::vector<const FeatureEntry*> out;
    out.reserve(entries_.size());
    const bool all_batches = batches.empty();
    for (const auto& e : entries_) {
        if (!all_batches) {
            bool found = false;
            for (const auto& b : batches) {
                if (b == e.batch) { found = true; break; }
            }
            if (!found) continue;
        }
        if (!mode_matches(e.mode, mode)) continue;
        out.push_back(&e);
    }
    return out;
}


// ---------------------------------------------------------------------------
// Shared helper — must mirror Python's _get_active_pickups exactly.
// Returns pickup node IDs (1..n) of requests that are in some route but not
// committed. Order matches Python (route iteration order).
// ---------------------------------------------------------------------------
static std::vector<int> get_active_pickups(const DARPSolution& sol, int n) {
    std::vector<int> active;
    for (const auto& route : sol.routes) {
        for (int node_id : route) {
            if (node_id >= 1 && node_id <= n && sol.committed.count(node_id) == 0) {
                active.push_back(node_id);
            }
        }
    }
    return active;
}


// ===========================================================================
// Batch: core (5 features)
// ===========================================================================

// num_active_requests — count of active pickups. (demand size)
static double feat_num_active_requests(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    return static_cast<double>(get_active_pickups(sol, inst.num_requests).size());
}
REGISTER_FEATURE_V2("num_active_requests", "core", "both", 0.0, feat_num_active_requests)


// radius_avg_tt — mean pairwise travel time between active pickups (off-diagonal mean
// of submatrix tt[pickups, pickups]). 0.0 if <= 1 active pickup. (demand spread)
static double feat_radius_avg_tt(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto pickups = get_active_pickups(sol, inst.num_requests);
    const int n = static_cast<int>(pickups.size());
    if (n <= 1) return 0.0;

    // Mirror Python's `sub.sum() - np.trace(sub)` accumulation order:
    // outer i, inner j; trace accumulated in the outer loop.
    double total = 0.0;
    double trace = 0.0;
    for (int i = 0; i < n; ++i) {
        int ni = pickups[i];
        trace += inst.travel_time[ni][ni];
        for (int j = 0; j < n; ++j) {
            total += inst.travel_time[ni][pickups[j]];
        }
    }
    double off_diag = total - trace;
    return off_diag / (static_cast<double>(n) * (n - 1));
}
REGISTER_FEATURE_V2("radius_avg_tt", "core", "both", 0.0, feat_radius_avg_tt)


// frac_committed — |committed| / (|committed| + |active|). 0.0 if denom=0.
// (vehicle state) Dynamic-only; static mode has empty committed by construction.
static double feat_frac_committed(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    const int n_committed = static_cast<int>(sol.committed.size());
    const int n_active = static_cast<int>(get_active_pickups(sol, inst.num_requests).size());
    const int denom = n_committed + n_active;
    if (denom == 0) return 0.0;
    return static_cast<double>(n_committed) / denom;
}
REGISTER_FEATURE_V2("frac_committed", "core", "dynamic", 0.0, feat_frac_committed)


// avg_route_load — mean requests per non-empty route. (routing solution)
static double feat_avg_route_load(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    const int n = inst.num_requests;
    const int empty_size = inst.empty_route_size();
    double total = 0.0;
    int count = 0;
    for (const auto& route : sol.routes) {
        if (static_cast<int>(route.size()) > empty_size) {
            int req_count = 0;
            for (int nid : route) {
                if (nid >= 1 && nid <= n) ++req_count;
            }
            total += req_count;
            ++count;
        }
    }
    if (count == 0) return 0.0;
    return total / count;
}
REGISTER_FEATURE_V2("avg_route_load", "core", "both", 0.0, feat_avg_route_load)


// temperature_normalized — temperature / sa_start_temp, in [0,1]. (ALNS state)
static double feat_temperature_normalized(
    const DARPSolution&, const DARPInstance&, const ALNSState& st) {
    if (st.sa_start_temp <= 0.0) return 0.0;
    return st.temperature / st.sa_start_temp;
}
REGISTER_FEATURE_V2("temperature_normalized", "core", "both", 0.0, feat_temperature_normalized)


// ===========================================================================
// Batch: tightness (5 features)
// Surfaces how close to ride-time / time-window violations the solution is.
// Targets rtcritical + lowslack operators.
// ===========================================================================

// Threshold for "near violation" — must match Python NEAR_VIOLATION_THRESHOLD.
static constexpr double NEAR_VIOLATION_THRESHOLD = 0.9;

// Walk routes in order; collect actual_ride_time / L for each active pickup.
// Mirrors Python _ride_time_utilizations exactly (including iteration order).
static std::vector<double> ride_time_utilizations(
    const DARPSolution& sol, const DARPInstance& inst) {
    const int n = inst.num_requests;
    std::vector<double> out;
    for (const auto& route : sol.routes) {
        auto sched = sol.compute_route_schedule(route);
        for (int node_id : route) {
            if (!(node_id >= 1 && node_id <= n)) continue;
            if (sol.committed.count(node_id)) continue;
            auto it = sched.ride_time.find(node_id);
            if (it == sched.ride_time.end()) continue;
            double L = inst.max_ride_time_for(node_id);
            if (L <= 0.0) continue;
            out.push_back(it->second / L);
        }
    }
    return out;
}

// ride_time_utilization_avg
static double feat_ride_time_utilization_avg(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto utils = ride_time_utilizations(sol, inst);
    if (utils.empty()) return 0.0;
    double total = 0.0;
    for (double u : utils) total += u;
    return total / static_cast<double>(utils.size());
}
REGISTER_FEATURE_V2("ride_time_utilization_avg", "tightness", "both", 0.0,
                     feat_ride_time_utilization_avg)

// ride_time_utilization_max
static double feat_ride_time_utilization_max(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto utils = ride_time_utilizations(sol, inst);
    if (utils.empty()) return 0.0;
    double m = utils[0];
    for (size_t i = 1; i < utils.size(); ++i) {
        if (utils[i] > m) m = utils[i];
    }
    return m;
}
REGISTER_FEATURE_V2("ride_time_utilization_max", "tightness", "both", 0.0,
                     feat_ride_time_utilization_max)

// near_violation_count
static double feat_near_violation_count(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto utils = ride_time_utilizations(sol, inst);
    int count = 0;
    for (double u : utils) {
        if (u >= NEAR_VIOLATION_THRESHOLD) ++count;
    }
    return static_cast<double>(count);
}
REGISTER_FEATURE_V2("near_violation_count", "tightness", "both", 0.0,
                     feat_near_violation_count)

// Per-route slack for non-empty routes, in route iteration order.
// Mirrors Python _per_route_slacks.
static std::vector<double> per_route_slacks(
    const DARPSolution& sol, const DARPInstance& inst) {
    std::vector<double> out;
    const int empty_size = inst.empty_route_size();
    for (const auto& route : sol.routes) {
        if (static_cast<int>(route.size()) > empty_size) {
            out.push_back(compute_route_slack(route, inst));
        }
    }
    return out;
}

// route_slack_avg
static double feat_route_slack_avg(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto slacks = per_route_slacks(sol, inst);
    if (slacks.empty()) return 0.0;
    double total = 0.0;
    for (double s : slacks) total += s;
    return total / static_cast<double>(slacks.size());
}
REGISTER_FEATURE_V2("route_slack_avg", "tightness", "both", 0.0,
                     feat_route_slack_avg)

// route_slack_std — population std dev (ddof=0). Two-pass naive accumulation
// in the same order as Python. 0.0 if <= 1 non-empty route.
static double feat_route_slack_std(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto slacks = per_route_slacks(sol, inst);
    const size_t k = slacks.size();
    if (k <= 1) return 0.0;
    double total = 0.0;
    for (double s : slacks) total += s;
    double mean = total / static_cast<double>(k);
    double sq = 0.0;
    for (double s : slacks) {
        double d = s - mean;
        sq += d * d;
    }
    return std::sqrt(sq / static_cast<double>(k));
}
REGISTER_FEATURE_V2("route_slack_std", "tightness", "both", 1e-12,
                     feat_route_slack_std)


// ===========================================================================
// Batch: worst_support (3 features)
// Surfaces dispersion of per-request and per-route cost contributions to give
// the worst_destroy operator more discriminative state signal.
// ===========================================================================

// removal_cost for each active request, in route iteration order.
// Active = pickup in some route AND not committed AND delivery in same route.
// Mirrors Python _per_active_request_costs exactly.
static std::vector<double> per_active_request_costs(
    const DARPSolution& sol, const DARPInstance& inst) {
    const int n = inst.num_requests;
    std::vector<double> out;
    for (int ri = 0; ri < static_cast<int>(sol.routes.size()); ++ri) {
        const auto& route = sol.routes[ri];
        for (int node_id : route) {
            if (!(node_id >= 1 && node_id <= n)) continue;
            if (sol.committed.count(node_id)) continue;
            // Skip if delivery isn't in this route (matches Python guard).
            int del_id = node_id + n;
            bool del_present = false;
            for (int x : route) {
                if (x == del_id) { del_present = true; break; }
            }
            if (!del_present) continue;
            out.push_back(removal_cost(sol, node_id, ri));
        }
    }
    return out;
}

// compute_route_cost over non-empty routes, in iteration order.
static std::vector<double> per_route_costs(
    const DARPSolution& sol, const DARPInstance& inst) {
    std::vector<double> out;
    const int empty_size = inst.empty_route_size();
    for (const auto& route : sol.routes) {
        if (static_cast<int>(route.size()) > empty_size) {
            out.push_back(compute_route_cost(route, inst));
        }
    }
    return out;
}

// cost_contribution_std — population std dev (ddof=0) of per-active-request
// removal costs. 0.0 if <= 1 active.
static double feat_cost_contribution_std(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto costs = per_active_request_costs(sol, inst);
    const size_t k = costs.size();
    if (k <= 1) return 0.0;
    double total = 0.0;
    for (double c : costs) total += c;
    double mean = total / static_cast<double>(k);
    double sq = 0.0;
    for (double c : costs) {
        double d = c - mean;
        sq += d * d;
    }
    return std::sqrt(sq / static_cast<double>(k));
}
REGISTER_FEATURE_V2("cost_contribution_std", "worst_support", "both", 1e-12,
                     feat_cost_contribution_std)

// cost_contribution_max_ratio — max / mean. Empty => 0.0; mean == 0 => 1.0.
static double feat_cost_contribution_max_ratio(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto costs = per_active_request_costs(sol, inst);
    if (costs.empty()) return 0.0;
    double total = 0.0;
    double m = costs[0];
    for (double c : costs) {
        total += c;
        if (c > m) m = c;
    }
    double mean = total / static_cast<double>(costs.size());
    if (mean == 0.0) return 1.0;
    return m / mean;
}
REGISTER_FEATURE_V2("cost_contribution_max_ratio", "worst_support", "both", 0.0,
                     feat_cost_contribution_max_ratio)

// route_cost_std — population std dev (ddof=0) of per-route travel costs over
// non-empty routes. 0.0 if <= 1 non-empty route.
static double feat_route_cost_std(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto costs = per_route_costs(sol, inst);
    const size_t k = costs.size();
    if (k <= 1) return 0.0;
    double total = 0.0;
    for (double c : costs) total += c;
    double mean = total / static_cast<double>(k);
    double sq = 0.0;
    for (double c : costs) {
        double d = c - mean;
        sq += d * d;
    }
    return std::sqrt(sq / static_cast<double>(k));
}
REGISTER_FEATURE_V2("route_cost_std", "worst_support", "both", 1e-12,
                     feat_route_cost_std)


// ===========================================================================
// Batch: lns_state (5 features)
// Bongiovanni et al. 2022 top-importance LNS state descriptors plus
// request-bank count.
// ===========================================================================

// q-sampling constants — must match all destroy operators in
// cpp/src/operators/destroy/{random,worst,shaw,ride_time_critical,low_slack_route}.cpp.
// All five operators share the same lo/hi formula.
static constexpr double LO_PCT = 0.10;
static constexpr double HI_PCT_STATIC = 0.40;
static constexpr double HI_PCT_DYNAMIC = 0.25;

// neighborhood_size_frac — q / q_max. q_max mirrors destroy ops' upper bound:
//   q_max = min(hi - 1, N) where hi follows the per-state dynamic detection
//   (committed.empty() => static rule, else dynamic rule). 0.0 if N == 0.
static double feat_neighborhood_size_frac(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState& st) {
    int n = static_cast<int>(get_active_pickups(sol, inst.num_requests).size());
    if (n == 0) return 0.0;
    bool dynamic = !sol.committed.empty();
    double hi_pct = dynamic ? HI_PCT_DYNAMIC : HI_PCT_STATIC;
    int lo = std::max(1, static_cast<int>(n * LO_PCT));
    int hi = std::max(lo + 1, static_cast<int>(n * hi_pct) + 1);
    int q_max = std::min(hi - 1, n);  // mirror std::min(dist(rng), N) clamp
    if (q_max <= 0) return 0.0;
    return static_cast<double>(st.current_q) / static_cast<double>(q_max);
}
REGISTER_FEATURE_V2("neighborhood_size_frac", "lns_state", "both", 0.0,
                     feat_neighborhood_size_frac)

// cost_vs_initial_frac — (obj_before - obj_initial) / obj_initial.
// obj_before is read via sol.objective() (matches state.obj_before logged at
// iter start; OBJ PASS audit confirms bit-equality). 0.0 if obj_initial == 0.
static double feat_cost_vs_initial_frac(
    const DARPSolution& sol, const DARPInstance&, const ALNSState& st) {
    if (st.obj_initial == 0.0) return 0.0;
    return (sol.objective() - st.obj_initial) / st.obj_initial;
}
REGISTER_FEATURE_V2("cost_vs_initial_frac", "lns_state", "both", 0.0,
                     feat_cost_vs_initial_frac)

// cost_vs_incumbent_frac — (obj_before - obj_best) / obj_best. SA can accept
// worsening moves so obj_before may exceed obj_best — feature is signed,
// no clipping. 0.0 if obj_best == 0.
static double feat_cost_vs_incumbent_frac(
    const DARPSolution& sol, const DARPInstance&, const ALNSState& st) {
    if (st.obj_best == 0.0) return 0.0;
    return (sol.objective() - st.obj_best) / st.obj_best;
}
REGISTER_FEATURE_V2("cost_vs_incumbent_frac", "lns_state", "both", 0.0,
                     feat_cost_vs_incumbent_frac)

// iteration_norm — iter / max_iterations clipped to [0, 1]. 0.0 if max == 0.
static double feat_iteration_norm(
    const DARPSolution&, const DARPInstance&, const ALNSState& st) {
    if (st.max_iterations == 0) return 0.0;
    double raw = static_cast<double>(st.iter)
               / static_cast<double>(st.max_iterations);
    if (raw < 0.0) return 0.0;
    if (raw > 1.0) return 1.0;
    return raw;
}
REGISTER_FEATURE_V2("iteration_norm", "lns_state", "both", 0.0,
                     feat_iteration_norm)

// unassigned_count — |unassigned| as double. No clipping.
static double feat_unassigned_count(
    const DARPSolution& sol, const DARPInstance&, const ALNSState&) {
    return static_cast<double>(sol.unassigned.size());
}
REGISTER_FEATURE_V2("unassigned_count", "lns_state", "both", 0.0,
                     feat_unassigned_count)


// ===========================================================================
// Batch: rolling_horizon (3 features, dynamic-only)
// State the static-mode model cannot observe — basis for the static-vs-dynamic
// ML ablation in the thesis.
// ===========================================================================

// committed_travel_time_frac — TT along committed prefixes / TT total.
// Numerator iterates each route's edges 0..min_insert_pos[r]-2; denominator
// iterates all edges. Mirrors Python exactly. 0.0 if denominator is 0.
static double feat_committed_travel_time_frac(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    const auto& tt = inst.travel_time;
    double committed_total = 0.0;
    double overall_total = 0.0;
    const auto& mip = sol.min_insert_pos;
    for (size_t ri = 0; ri < sol.routes.size(); ++ri) {
        const auto& route = sol.routes[ri];
        for (size_t i = 0; i + 1 < route.size(); ++i) {
            overall_total += tt[route[i]][route[i + 1]];
        }
        if (ri < mip.size()) {
            int limit = mip[ri] - 1;
            for (int i = 0; i < limit; ++i) {
                committed_total += tt[route[i]][route[i + 1]];
            }
        }
    }
    if (overall_total == 0.0) return 0.0;
    return committed_total / overall_total;
}
REGISTER_FEATURE_V2("committed_travel_time_frac", "rolling_horizon", "dynamic",
                     0.0, feat_committed_travel_time_frac)

// horizon_elapsed_frac — (sim_clock - sim_start) / (sim_end - sim_start).
// sim_start = nodes[0].earliest; sim_end = max over requests 1..n of
// nodes[req].latest. Clipped to [0, 1]. Mirrors Bongiovanni et al. 2022 #2.
static double feat_horizon_elapsed_frac(
    const DARPSolution&, const DARPInstance& inst, const ALNSState& st) {
    double sim_start = inst.nodes[0].earliest;
    double sim_end = 0.0;
    int n = inst.num_requests;
    for (int req = 1; req <= n; ++req) {
        if (inst.nodes[req].latest > sim_end) sim_end = inst.nodes[req].latest;
    }
    double span = sim_end - sim_start;
    if (span == 0.0) return 0.0;
    double raw = (st.sim_clock - sim_start) / span;
    if (raw < 0.0) return 0.0;
    if (raw > 1.0) return 1.0;
    return raw;
}
REGISTER_FEATURE_V2("horizon_elapsed_frac", "rolling_horizon", "dynamic",
                     0.0, feat_horizon_elapsed_frac)

// flexibility_horizon was removed after the first 21-feature training run.
// Diagnostic on cache_v2_dynamic showed it was structurally degenerate:
// constant 0 across all 13,762 sampled rows because Manhattan dynamic logs
// only fire epochs (MIN_REQUESTS_FOR_EPOCH gating) AFTER all active pickups'
// time windows have already opened. Documented under "Resolved methodology
// findings" in THESIS_NOTES.md.
