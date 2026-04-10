#include "operators/repair/slack_insertion.h"
#include "operators/repair/repair_utils.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

static constexpr double EPS = 1e-6;

// Alpha controls cost vs slack tradeoff (0 = greedy, 1 = max slack preservation)
static constexpr double ALPHA = 0.1;

// ---------------------------------------------------------------------------
// Helper: compute total forward time slack for a route
// ---------------------------------------------------------------------------
static double compute_route_slack(const std::vector<int>& route,
                                   const DARPInstance& inst) {
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int size = static_cast<int>(route.size());
    if (size <= inst.empty_route_size()) return 0.0;

    // Forward pass
    std::vector<double> tau(size);
    tau[0] = nodes[route[0]].earliest;
    double depart = tau[0] + nodes[route[0]].service_duration;
    for (int i = 1; i < size; i++) {
        double arrival = depart + tt[route[i - 1]][route[i]];
        tau[i] = std::max(arrival, nodes[route[i]].earliest);
        depart = tau[i] + nodes[route[i]].service_duration;
    }

    // Forward time slack at position 0
    std::vector<double> arrival_arr(size, 0.0);
    depart = tau[0] + nodes[route[0]].service_duration;
    for (int i = 1; i < size; i++) {
        arrival_arr[i] = depart + tt[route[i - 1]][route[i]];
        depart = tau[i] + nodes[route[i]].service_duration;
    }

    std::vector<double> slack(size);
    slack[size - 1] = nodes[route[size - 1]].latest - tau[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = tau[i + 1] - arrival_arr[i + 1];
        slack[i] = std::min(nodes[route[i]].latest - tau[i], slack[i + 1] + wait_next);
    }

    // Return total forward time slack (sum over all positions)
    double total = 0.0;
    for (int i = 0; i < size; i++) {
        total += slack[i];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Helper: compute total travel cost (sum of edge travel times) for a route
// ---------------------------------------------------------------------------
static double compute_route_cost(const std::vector<int>& route,
                                  const DARPInstance& inst) {
    double cost = 0.0;
    for (int i = 0; i + 1 < static_cast<int>(route.size()); i++) {
        cost += inst.travel_time[route[i]][route[i + 1]];
    }
    return cost;
}

// ---------------------------------------------------------------------------
// Slack-based insertion — O(1) path
// ---------------------------------------------------------------------------
static DARPSolution slack_insertion_o1(const DARPSolution& sol, std::mt19937& rng) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;

    using Clock = std::chrono::high_resolution_clock;

    // Lazy aux precomputation
    std::vector<RouteAuxData> aux_cache(repaired.routes.size());
    std::vector<double> baseline_slack(repaired.routes.size(), 0.0);
    std::vector<double> baseline_cost(repaired.routes.size(), 0.0);
    std::vector<bool> aux_computed(repaired.routes.size(), false);

    // Best-insertion cache: indexed by request id [0..n]
    struct SlackCache {
        double score;
        int route_idx;
        int p_pos;
        int d_pos;
        bool valid;
    };

    std::vector<SlackCache> slack_cache(n + 1);
    for (int i = 0; i <= n; i++) {
        slack_cache[i].valid = false;
        slack_cache[i].score = std::numeric_limits<double>::infinity();
        slack_cache[i].route_idx = -1;
        slack_cache[i].p_pos = -1;
        slack_cache[i].d_pos = -1;
    }

    // Lambda: scan all candidate routes for a single request, return best score/position
    auto scan_request = [&](int req) -> SlackCache {
        SlackCache result;
        result.score = std::numeric_limits<double>::infinity();
        result.route_idx = -1;
        result.p_pos = -1;
        result.d_pos = -1;
        result.valid = true;

        int pickup_id = req;
        int delivery_id = req + n;

        auto candidates = build_candidate_routes(repaired, pickup_id, repaired.top_n_routes);
        DynamicStats::routes_checked += static_cast<int>(candidates.size());
        DynamicStats::request_evals++;

        for (int r : candidates) {
            const auto& route = repaired.routes[r];
            int rlen = static_cast<int>(route.size());

            int min_pos = inst.route_start_offset();
            if (!repaired.min_insert_pos.empty())
                min_pos = std::max(min_pos, repaired.min_insert_pos[r]);
            if (min_pos >= rlen) continue;

            if (!aux_computed[r]) {
                auto t0 = Clock::now();
                aux_cache[r] = precompute_route_aux(route, inst);
                baseline_slack[r] = compute_route_slack(route, inst);
                baseline_cost[r] = compute_route_cost(route, inst);
                aux_computed[r] = true;
                auto t1 = Clock::now();
                DynamicStats::aux_precompute_ms +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            const auto& aux = aux_cache[r];

            for (int k_pos = min_pos; k_pos < rlen; k_pos++) {
                for (int l_pos = k_pos; l_pos < rlen; l_pos++) {
                    double delta;
                    if (k_pos == l_pos) {
                        delta = tt[route[k_pos - 1]][pickup_id]
                              + tt[pickup_id][delivery_id]
                              + tt[delivery_id][route[k_pos]]
                              - tt[route[k_pos - 1]][route[k_pos]];
                    } else {
                        delta = tt[route[k_pos - 1]][pickup_id]
                              + tt[pickup_id][route[k_pos]]
                              - tt[route[k_pos - 1]][route[k_pos]]
                              + tt[route[l_pos - 1]][delivery_id]
                              + tt[delivery_id][route[l_pos]]
                              - tt[route[l_pos - 1]][route[l_pos]];
                    }
                    double norm_delta_lb = (baseline_cost[r] > EPS)
                        ? delta / baseline_cost[r] : delta;
                    if ((1.0 - ALPHA) * norm_delta_lb >= result.score) continue;

                    DynamicStats::feas_checks++;
                    if (!check_insertion_o1(aux, req, k_pos, l_pos, inst))
                        continue;

                    std::vector<int> candidate = route;
                    candidate.insert(candidate.begin() + k_pos, pickup_id);
                    candidate.insert(candidate.begin() + l_pos + 1, delivery_id);

                    double new_slack = compute_route_slack(candidate, inst);
                    double slack_loss = baseline_slack[r] - new_slack;

                    double norm_delta = (baseline_cost[r] > EPS)
                        ? delta / baseline_cost[r] : delta;
                    double norm_slack = (baseline_slack[r] > EPS)
                        ? slack_loss / baseline_slack[r] : slack_loss;
                    double score = (1.0 - ALPHA) * norm_delta + ALPHA * norm_slack;

                    if (score < result.score) {
                        auto [feas, reason] = repaired.check_route_feasibility(candidate);
                        if (!feas) continue;

                        result.score = score;
                        result.route_idx = r;
                        result.p_pos = k_pos;
                        result.d_pos = l_pos + 1;
                    }
                }
            }
        }

        return result;
    };

    while (!repaired.unassigned.empty()) {
        // Step 1: Refresh cache for any invalidated entries
        for (int req : repaired.unassigned) {
            if (!slack_cache[req].valid) {
                slack_cache[req] = scan_request(req);
            }
        }

        // Step 2: Find lowest-score request
        double best_score = std::numeric_limits<double>::infinity();
        int best_req = -1;
        for (int req : repaired.unassigned) {
            if (slack_cache[req].valid &&
                slack_cache[req].route_idx >= 0 &&
                slack_cache[req].score < best_score) {
                best_score = slack_cache[req].score;
                best_req = req;
            }
        }

        if (best_req < 0) break;

        const SlackCache b = slack_cache[best_req];
        int k_star = b.route_idx;

        // Step 3: Safety re-validation before commit
        if (!check_insertion_o1(aux_cache[k_star], best_req, b.p_pos, b.d_pos - 1, inst)) {
            slack_cache[best_req].valid = false;
            continue;
        }

        // Step 4: Commit the insertion
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.p_pos, best_req);
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.d_pos, best_req + n);
        repaired.unassigned.erase(best_req);
        slack_cache[best_req].valid = false;

        // Step 5: Recompute aux, slack, and cost for the modified route
        aux_cache[k_star] = precompute_route_aux(repaired.routes[k_star], inst);
        baseline_slack[k_star] = compute_route_slack(repaired.routes[k_star], inst);
        baseline_cost[k_star] = compute_route_cost(repaired.routes[k_star], inst);

        // Step 6: Invalidate cache entries that pointed to the modified route
        for (int req : repaired.unassigned) {
            if (slack_cache[req].valid && slack_cache[req].route_idx == k_star) {
                slack_cache[req].valid = false;
            }
        }
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Slack-based insertion — classic path
// ---------------------------------------------------------------------------
static DARPSolution slack_insertion_classic(const DARPSolution& sol, std::mt19937& rng) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    int Q = inst.vehicle_capacity;

    // Precompute baseline slack and cost
    std::vector<double> baseline_slack(repaired.routes.size());
    std::vector<double> baseline_cost(repaired.routes.size());
    for (int r = 0; r < static_cast<int>(repaired.routes.size()); r++) {
        baseline_slack[r] = compute_route_slack(repaired.routes[r], inst);
        baseline_cost[r] = compute_route_cost(repaired.routes[r], inst);
    }

    while (!repaired.unassigned.empty()) {
        double best_score = std::numeric_limits<double>::infinity();
        int best_req = -1;
        int best_route = -1;
        int best_p = -1;
        int best_d = -1;

        for (int req : repaired.unassigned) {
            int pickup_id = req;
            int delivery_id = req + n;

            auto candidates = build_candidate_routes(repaired, pickup_id, repaired.top_n_routes);
            DynamicStats::routes_checked += static_cast<int>(candidates.size());
            DynamicStats::request_evals++;

            for (int r : candidates) {
                const auto& route = repaired.routes[r];
                int rlen = static_cast<int>(route.size());

                int min_pos_c = inst.route_start_offset();
                if (!repaired.min_insert_pos.empty())
                    min_pos_c = std::max(min_pos_c, repaired.min_insert_pos[r]);
                if (min_pos_c >= rlen) continue;

                // Forward pass for filters
                std::vector<double> depart_cache(rlen);
                std::vector<int> load_cache(rlen);
                depart_cache[0] = nodes[route[0]].earliest + nodes[route[0]].service_duration;
                load_cache[0] = 0;
                for (int i = 1; i < rlen; i++) {
                    double arr = depart_cache[i - 1] + tt[route[i - 1]][route[i]];
                    double begin = std::max(arr, nodes[route[i]].earliest);
                    depart_cache[i] = begin + nodes[route[i]].service_duration;
                    load_cache[i] = load_cache[i - 1] + nodes[route[i]].demand;
                }

                for (int p_pos = min_pos_c; p_pos < rlen; p_pos++) {
                    double arr_pickup = depart_cache[p_pos - 1] + tt[route[p_pos - 1]][pickup_id];
                    if (arr_pickup > nodes[pickup_id].latest + EPS) continue;

                    for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                        bool cap_ok = true;
                        for (int j = p_pos - 1; j < d_pos - 1; j++) {
                            if (load_cache[j] + nodes[pickup_id].demand > Q) { cap_ok = false; break; }
                        }
                        if (!cap_ok) continue;

                        double delta;
                        if (d_pos == p_pos + 1) {
                            delta = tt[route[p_pos - 1]][pickup_id]
                                  + tt[pickup_id][delivery_id]
                                  + tt[delivery_id][route[p_pos]]
                                  - tt[route[p_pos - 1]][route[p_pos]];
                        } else {
                            delta = tt[route[p_pos - 1]][pickup_id]
                                  + tt[pickup_id][route[p_pos]]
                                  - tt[route[p_pos - 1]][route[p_pos]]
                                  + tt[route[d_pos - 2]][delivery_id]
                                  + tt[delivery_id][route[d_pos - 1]]
                                  - tt[route[d_pos - 2]][route[d_pos - 1]];
                        }

                        std::vector<int> candidate = route;
                        candidate.insert(candidate.begin() + p_pos, pickup_id);
                        candidate.insert(candidate.begin() + d_pos, delivery_id);

                        auto [feasible, reason] = repaired.check_route_feasibility(candidate);
                        if (!feasible) continue;

                        double new_slack = compute_route_slack(candidate, inst);
                        double slack_loss = baseline_slack[r] - new_slack;

                        double norm_delta = (baseline_cost[r] > EPS)
                            ? delta / baseline_cost[r] : delta;
                        double norm_slack = (baseline_slack[r] > EPS)
                            ? slack_loss / baseline_slack[r] : slack_loss;
                        double score = (1.0 - ALPHA) * norm_delta + ALPHA * norm_slack;

                        if (score < best_score) {
                            best_score = score;
                            best_req = req;
                            best_route = r;
                            best_p = p_pos;
                            best_d = d_pos;
                        }
                    }
                }
            }
        }

        if (best_req < 0) break;

        repaired.routes[best_route].insert(
            repaired.routes[best_route].begin() + best_p, best_req);
        repaired.routes[best_route].insert(
            repaired.routes[best_route].begin() + best_d, best_req + n);
        repaired.unassigned.erase(best_req);

        baseline_slack[best_route] = compute_route_slack(repaired.routes[best_route], inst);
        baseline_cost[best_route] = compute_route_cost(repaired.routes[best_route], inst);
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
DARPSolution slack_insertion(const DARPSolution& sol, std::mt19937& rng) {
    if (sol.feasibility_mode == "o1") {
        return slack_insertion_o1(sol, rng);
    } else {
        return slack_insertion_classic(sol, rng);
    }
}