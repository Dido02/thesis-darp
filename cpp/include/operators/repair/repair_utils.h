#pragma once

#include "instance.h"
#include "solution.h"

#include <algorithm>
#include <atomic>
#include <vector>

// ---------------------------------------------------------------------------
// Dynamic stats counters (reset per epoch, read after ALNS)
// ---------------------------------------------------------------------------
struct DynamicStats {
    static long long routes_checked;
    static long long request_evals;  // number of (request, route-list) evaluations
    static long long feas_checks;
    static int repair_calls;
    static double aux_precompute_ms;

    static void reset() {
        routes_checked = 0;
        request_evals = 0;
        feas_checks = 0;
        repair_calls = 0;
        aux_precompute_ms = 0.0;
    }
};

// ---------------------------------------------------------------------------
// Build candidate route list for a request: top-N routes by travel time
// from route's current position to pickup. Returns route indices sorted
// by travel time (nearest first).
//
// If top_n <= 0 or min_insert_pos is empty, returns all valid route indices.
// ---------------------------------------------------------------------------
inline std::vector<int> build_candidate_routes(
    const DARPSolution& sol,
    int pickup_id,
    int top_n) {

    const auto& inst = *sol.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int K = static_cast<int>(sol.routes.size());

    // Collect (travel_time_to_pickup, route_index) for reachable routes
    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(K);

    for (int k = 0; k < K; k++) {
        const auto& route = sol.routes[k];
        int rlen = static_cast<int>(route.size());

        int min_pos = inst.route_start_offset();
        if (!sol.min_insert_pos.empty())
            min_pos = std::max(min_pos, sol.min_insert_pos[k]);
        if (min_pos >= rlen) continue;

        // Reachability: earliest departure from current position + travel to pickup
        int prev_node = route[min_pos - 1];
        double depart_at_prev = nodes[prev_node].earliest + nodes[prev_node].service_duration;
        double travel = tt[prev_node][pickup_id];
        if (depart_at_prev + travel > nodes[pickup_id].latest + 1e-6)
            continue;

        candidates.push_back({travel, k});
    }

    // If no limit or fewer candidates than limit, return all
    if (top_n <= 0 || static_cast<int>(candidates.size()) <= top_n) {
        std::vector<int> result;
        result.reserve(candidates.size());
        for (auto& [t, k] : candidates) result.push_back(k);
        return result;
    }

    // Partial sort to get top N nearest
    std::partial_sort(candidates.begin(),
                      candidates.begin() + top_n,
                      candidates.end());

    std::vector<int> result;
    result.reserve(top_n);
    for (int i = 0; i < top_n; i++) {
        result.push_back(candidates[i].second);
    }
    return result;
}
