#include "operators/destroy/worst_removal.h"
#include "operators/cost_helpers.h"
#include "operators/repair/repair_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

static constexpr double EPS = 1e-6;

// Randomization degree (from paper: p=3 for worst removal)
static constexpr int RANDOMIZATION_P = 3;

// ---------------------------------------------------------------------------
// Helper: build request-to-route index
// ---------------------------------------------------------------------------
static std::vector<int> build_req_to_route(const DARPSolution& sol) {
    int n = sol.instance->num_requests;
    std::vector<int> req_route(n + 1, -1);
    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        for (int nid : sol.routes[r]) {
            if (nid >= 1 && nid <= n) {
                req_route[nid] = r;
            }
        }
    }
    return req_route;
}

// ---------------------------------------------------------------------------
// Helper: compute cost saved by removing request from a known route.
// Declared in operators/cost_helpers.h so feature extractors share it.
// ---------------------------------------------------------------------------
double removal_cost(const DARPSolution& sol, int req, int route_idx) {
    const auto& tt = sol.instance->travel_time;
    int n = sol.instance->num_requests;
    int pickup_id = req;
    int delivery_id = req + n;
    const auto& route = sol.routes[route_idx];

    int p_pos = -1, d_pos = -1;
    for (int i = 0; i < static_cast<int>(route.size()); i++) {
        if (route[i] == pickup_id) p_pos = i;
        if (route[i] == delivery_id) d_pos = i;
    }
    if (p_pos < 0) return 0.0;

    double cost_with_p = tt[route[p_pos - 1]][pickup_id]
                       + tt[pickup_id][route[p_pos + 1]];
    double cost_without_p = tt[route[p_pos - 1]][route[p_pos + 1]];

    int d_adj = (d_pos > p_pos) ? d_pos - 1 : d_pos;

    std::vector<int> route_no_p;
    for (int i = 0; i < static_cast<int>(route.size()); i++) {
        if (i != p_pos) route_no_p.push_back(route[i]);
    }

    double cost_with_d = tt[route_no_p[d_adj - 1]][delivery_id]
                       + tt[delivery_id][route_no_p[d_adj + 1]];
    double cost_without_d = tt[route_no_p[d_adj - 1]][route_no_p[d_adj + 1]];

    return (cost_with_p - cost_without_p) + (cost_with_d - cost_without_d);
}

// ---------------------------------------------------------------------------
// Helper: remove a request from a known route
// ---------------------------------------------------------------------------
static void remove_request(DARPSolution& sol, int req, int route_idx) {
    int pickup_id = req;
    int delivery_id = req + sol.instance->num_requests;
    auto& route = sol.routes[route_idx];

    auto it_p = std::find(route.begin(), route.end(), pickup_id);
    if (it_p != route.end()) route.erase(it_p);
    auto it_d = std::find(route.begin(), route.end(), delivery_id);
    if (it_d != route.end()) route.erase(it_d);
    sol.unassigned.insert(req);
}

// ---------------------------------------------------------------------------
// Worst removal
// ---------------------------------------------------------------------------
DARPSolution worst_removal(const DARPSolution& sol, std::mt19937& rng) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    DARPSolution destroyed = sol.copy();
    int n = destroyed.instance->num_requests;

    // Collect assigned requests
    std::vector<int> assigned;
    for (int i = 1; i <= n; i++) {
        if (!destroyed.unassigned.count(i) && !destroyed.committed.count(i)) {
            assigned.push_back(i);
        }
    }
    if (assigned.empty()) return destroyed;

    // Number to remove: 10-40% (static), 5-15% (dynamic)
    bool dynamic = !destroyed.committed.empty();
    double lo_pct = dynamic ? 0.10 : 0.10;
    double hi_pct = dynamic ? 0.25 : 0.40;
    int lo = std::max(1, static_cast<int>(assigned.size() * lo_pct));
    int hi = std::max(lo + 1, static_cast<int>(assigned.size() * hi_pct) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), static_cast<int>(assigned.size()));

    // Build req_route index and compute removal costs once
    auto req_route = build_req_to_route(destroyed);

    std::vector<std::pair<double, int>> costs;
    for (int req : assigned) {
        if (req_route[req] >= 0) {
            double c = removal_cost(destroyed, req, req_route[req]);
            costs.push_back({c, req});
        }
    }

    // Sort descending by cost (highest cost = worst = removed first)
    std::sort(costs.begin(), costs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int removed = 0; removed < q && !costs.empty(); removed++) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int idx = static_cast<int>(
            std::floor(std::pow(u(rng), RANDOMIZATION_P) * costs.size()));
        idx = std::min(idx, static_cast<int>(costs.size()) - 1);

        int req = costs[idx].second;
        if (req_route[req] >= 0) {
            remove_request(destroyed, req, req_route[req]);
        }
        costs.erase(costs.begin() + idx);
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    DynamicStats::destroy_worst_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    DynamicStats::destroy_worst_count++;
    return destroyed;
}
