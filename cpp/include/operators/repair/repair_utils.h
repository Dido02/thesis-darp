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

    // Per-repair-operator timing (ms)
    static double repair_greedy_ms;
    static double repair_regret2_ms;
    static double repair_regret3_ms;
    static double repair_regret4_ms;
    static double repair_regretM_ms;
    static double repair_slack_ms;
    static int repair_greedy_count;
    static int repair_regret2_count;
    static int repair_regret3_count;
    static int repair_regret4_count;
    static int repair_regretM_count;
    static int repair_slack_count;

    // Per-destroy-operator timing (ms)
    static double destroy_random_ms;
    static double destroy_worst_ms;
    static double destroy_shaw_ms;
    static double destroy_rtcritical_ms;
    static double destroy_lowslack_ms;
    static double destroy_entireroute_ms;
    static int destroy_random_count;
    static int destroy_worst_count;
    static int destroy_shaw_count;
    static int destroy_rtcritical_count;
    static int destroy_lowslack_count;
    static int destroy_entireroute_count;

    static void reset() {
        routes_checked = 0;
        request_evals = 0;
        feas_checks = 0;
        repair_calls = 0;
        aux_precompute_ms = 0.0;
        repair_greedy_ms = 0.0;
        repair_regret2_ms = 0.0;
        repair_regret3_ms = 0.0;
        repair_regret4_ms = 0.0;
        repair_regretM_ms = 0.0;
        repair_slack_ms = 0.0;
        repair_greedy_count = 0;
        repair_regret2_count = 0;
        repair_regret3_count = 0;
        repair_regret4_count = 0;
        repair_regretM_count = 0;
        repair_slack_count = 0;
        destroy_random_ms = 0.0;
        destroy_worst_ms = 0.0;
        destroy_shaw_ms = 0.0;
        destroy_rtcritical_ms = 0.0;
        destroy_lowslack_ms = 0.0;
        destroy_entireroute_ms = 0.0;
        destroy_random_count = 0;
        destroy_worst_count = 0;
        destroy_shaw_count = 0;
        destroy_rtcritical_count = 0;
        destroy_lowslack_count = 0;
        destroy_entireroute_count = 0;
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

        // Use cached real departure time if available (dynamic solver populates it
        // via compute_min_insert_positions). Fall back to optimistic estimate
        // for callers that don't populate the cache (static solver, tests).
        double depart_at_prev;
        int prev_node;
        if (!sol.route_ready_time.empty()) {
            depart_at_prev = sol.route_ready_time[k];
            prev_node = sol.route_ready_node[k];
        } else {
            prev_node = route[min_pos - 1];
            depart_at_prev = nodes[prev_node].earliest + nodes[prev_node].service_duration;
        }

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
