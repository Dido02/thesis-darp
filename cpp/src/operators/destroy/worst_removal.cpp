#include "operators/destroy/worst_removal.h"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr double EPS = 1e-6;

// Randomization degree (from paper: p=3 for worst removal)
static constexpr int RANDOMIZATION_P = 3;

// ---------------------------------------------------------------------------
// Helper: compute cost saved by removing request req from its route
// ---------------------------------------------------------------------------
static double removal_cost(const DARPSolution& sol, int req) {
    const auto& tt = sol.instance->travel_time;
    int n = sol.instance->num_requests;
    int pickup_id = req;
    int delivery_id = req + n;

    for (const auto& route : sol.routes) {
        int p_pos = -1, d_pos = -1;
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (route[i] == pickup_id) p_pos = i;
            if (route[i] == delivery_id) d_pos = i;
        }
        if (p_pos < 0) continue;

        // Cost of arcs touching pickup: prev->pickup->next
        double cost_with_p = tt[route[p_pos - 1]][pickup_id]
                           + tt[pickup_id][route[p_pos + 1]];
        double cost_without_p = tt[route[p_pos - 1]][route[p_pos + 1]];

        // After removing pickup, delivery shifts by -1
        int d_adj = (d_pos > p_pos) ? d_pos - 1 : d_pos;

        // Build route without pickup to get correct indices for delivery
        std::vector<int> route_no_p;
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (i != p_pos) route_no_p.push_back(route[i]);
        }

        // Cost of arcs touching delivery in route_no_p
        double cost_with_d = tt[route_no_p[d_adj - 1]][delivery_id]
                           + tt[delivery_id][route_no_p[d_adj + 1]];
        double cost_without_d = tt[route_no_p[d_adj - 1]][route_no_p[d_adj + 1]];

        return (cost_with_p - cost_without_p) + (cost_with_d - cost_without_d);
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Helper: remove a request (pickup + delivery) from its route
// ---------------------------------------------------------------------------
static void remove_request(DARPSolution& sol, int req) {
    int pickup_id = req;
    int delivery_id = req + sol.instance->num_requests;

    for (auto& route : sol.routes) {
        auto it_p = std::find(route.begin(), route.end(), pickup_id);
        if (it_p != route.end()) {
            route.erase(it_p);
            auto it_d = std::find(route.begin(), route.end(), delivery_id);
            if (it_d != route.end()) {
                route.erase(it_d);
            }
            sol.unassigned.insert(req);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Worst removal
// ---------------------------------------------------------------------------
DARPSolution worst_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    int n = destroyed.instance->num_requests;

    // Collect assigned requests
    std::vector<int> assigned;
    for (int i = 1; i <= n; i++) {
        if (!destroyed.unassigned.count(i)) {
            assigned.push_back(i);
        }
    }
    if (assigned.empty()) return destroyed;

    // Number to remove: 10-40% (same as random removal)
    int lo = std::max(1, static_cast<int>(assigned.size() * 0.10));
    int hi = std::max(lo + 1, static_cast<int>(assigned.size() * 0.40) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), static_cast<int>(assigned.size()));

    for (int removed = 0; removed < q; removed++) {
        // Recompute removal costs for remaining assigned requests
        // (dynamic version: costs change after each removal)
        std::vector<std::pair<double, int>> costs;
        for (int req : assigned) {
            if (!destroyed.unassigned.count(req)) {
                double c = removal_cost(destroyed, req);
                costs.push_back({c, req});
            }
        }
        if (costs.empty()) break;

        // Sort descending by cost saved
        std::sort(costs.begin(), costs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Randomized selection: index = floor(random^p * size)
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int idx = static_cast<int>(
            std::floor(std::pow(u(rng), RANDOMIZATION_P) * costs.size()));
        idx = std::min(idx, static_cast<int>(costs.size()) - 1);

        remove_request(destroyed, costs[idx].second);
    }

    return destroyed;
}