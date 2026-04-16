#include "ml/feature_extractor.h"

#include <vector>

// ---------------------------------------------------------------------------
// Registry implementation
// ---------------------------------------------------------------------------
FeatureRegistry& FeatureRegistry::instance() {
    static FeatureRegistry r;
    return r;
}

void FeatureRegistry::add(const std::string& name, FeatureFunc fn) {
    fns_[name] = fn;
}

std::vector<std::string> FeatureRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(fns_.size());
    for (const auto& [k, _] : fns_) out.push_back(k);
    return out;
}

bool FeatureRegistry::has(const std::string& name) const {
    return fns_.find(name) != fns_.end();
}

std::unordered_map<std::string, double> FeatureRegistry::extract_all(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState& st) const {
    std::unordered_map<std::string, double> result;
    result.reserve(fns_.size());
    for (const auto& [name, fn] : fns_) {
        result.emplace(name, fn(sol, inst, st));
    }
    return result;
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


// ---------------------------------------------------------------------------
// Feature: num_active_requests
// Count of requests in routes but not committed. Category: demand size.
// ---------------------------------------------------------------------------
static double feat_num_active_requests(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    return static_cast<double>(get_active_pickups(sol, inst.num_requests).size());
}
REGISTER_FEATURE("num_active_requests", feat_num_active_requests)


// ---------------------------------------------------------------------------
// Feature: radius_avg_tt
// Mean pairwise travel time between active pickups (off-diagonal mean of the
// submatrix tt[pickups, pickups]). 0.0 if <= 1 active pickup.
// ---------------------------------------------------------------------------
static double feat_radius_avg_tt(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    auto pickups = get_active_pickups(sol, inst.num_requests);
    const int n = static_cast<int>(pickups.size());
    if (n <= 1) return 0.0;

    // Sum of off-diagonal entries (diagonal is 0 per convention, but subtract trace explicitly
    // to match Python's `sub.sum() - np.trace(sub)` semantics exactly).
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
REGISTER_FEATURE("radius_avg_tt", feat_radius_avg_tt)


// ---------------------------------------------------------------------------
// Feature: frac_committed
// |committed| / (|committed| + |in-route-not-committed|). 0.0 if denom=0.
// ---------------------------------------------------------------------------
static double feat_frac_committed(
    const DARPSolution& sol, const DARPInstance& inst, const ALNSState&) {
    const int n_committed = static_cast<int>(sol.committed.size());
    const int n_active = static_cast<int>(get_active_pickups(sol, inst.num_requests).size());
    const int denom = n_committed + n_active;
    if (denom == 0) return 0.0;
    return static_cast<double>(n_committed) / denom;
}
REGISTER_FEATURE("frac_committed", feat_frac_committed)


// ---------------------------------------------------------------------------
// Feature: avg_route_load
// Mean requests per non-empty route. A route is "non-empty" if len > empty_size.
// ---------------------------------------------------------------------------
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
REGISTER_FEATURE("avg_route_load", feat_avg_route_load)


// ---------------------------------------------------------------------------
// Feature: temperature_normalized
// temperature / sa_start_temp, in [0, 1]. 0.0 if sa_start_temp <= 0.
// ---------------------------------------------------------------------------
static double feat_temperature_normalized(
    const DARPSolution&, const DARPInstance&, const ALNSState& st) {
    if (st.sa_start_temp <= 0.0) return 0.0;
    return st.temperature / st.sa_start_temp;
}
REGISTER_FEATURE("temperature_normalized", feat_temperature_normalized)
