#include "operators/repair/greedy_insertion.h"
#include "operators/repair/repair_utils.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Forward pass helper for greedy insertion (used by slack/full modes)
// ---------------------------------------------------------------------------
static void forward_pass(
    const std::vector<int>& route,
    const std::vector<Node>& nodes,
    const std::vector<std::vector<double>>& tt,
    std::vector<double>& depart,
    std::vector<int>& load) {

    int size = static_cast<int>(route.size());
    depart.resize(size);
    load.resize(size);

    const auto& node0 = nodes[route[0]];
    depart[0] = node0.earliest + node0.service_duration;
    load[0] = 0;

    for (int i = 1; i < size; i++) {
        int prev_id = route[i - 1];
        int curr_id = route[i];
        const auto& node = nodes[curr_id];
        double arrival = depart[i - 1] + tt[prev_id][curr_id];
        double begin = std::max(arrival, node.earliest);
        depart[i] = begin + node.service_duration;
        load[i] = load[i - 1] + node.demand;
    }
}

// ---------------------------------------------------------------------------
// Greedy insertion — O(1) feasibility path
// ---------------------------------------------------------------------------
static DARPSolution greedy_insertion_o1(const DARPSolution& sol, std::mt19937& rng) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;

    using Clock = std::chrono::high_resolution_clock;

    // Lazy aux precomputation: compute on first access per route
    std::vector<RouteAuxData> aux_data(repaired.routes.size());
    std::vector<bool> aux_computed(repaired.routes.size(), false);

    DynamicStats::repair_calls++;

    // Best-insertion cache: indexed by request id [0..n]
    struct BestInsertion {
        double cost;
        int route_idx;
        int p_pos;
        int d_pos;
        bool valid;
    };

    std::vector<BestInsertion> best_cache(n + 1);
    for (int i = 0; i <= n; i++) {
        best_cache[i].valid = false;
        best_cache[i].cost = std::numeric_limits<double>::infinity();
        best_cache[i].route_idx = -1;
        best_cache[i].p_pos = -1;
        best_cache[i].d_pos = -1;
    }

    // Lambda: scan all candidate routes for a single request, return best insertion
    auto scan_request = [&](int req) -> BestInsertion {
        BestInsertion result;
        result.cost = std::numeric_limits<double>::infinity();
        result.route_idx = -1;
        result.p_pos = -1;
        result.d_pos = -1;
        result.valid = true;

        int pickup_id = req;
        int delivery_id = req + n;

        auto candidates = build_candidate_routes(repaired, pickup_id, repaired.top_n_routes);
        DynamicStats::routes_checked += static_cast<int>(candidates.size());
        DynamicStats::request_evals++;

        for (int k : candidates) {
            const auto& route = repaired.routes[k];
            int rlen = static_cast<int>(route.size());

            int min_pos = inst.route_start_offset();
            if (!repaired.min_insert_pos.empty())
                min_pos = std::max(min_pos, repaired.min_insert_pos[k]);
            if (min_pos >= rlen) continue;

            if (!aux_computed[k]) {
                auto t0 = Clock::now();
                aux_data[k] = precompute_route_aux(route, inst);
                aux_computed[k] = true;
                auto t1 = Clock::now();
                DynamicStats::aux_precompute_ms +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            const auto& aux = aux_data[k];

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
                    if (delta >= result.cost) continue;

                    DynamicStats::feas_checks++;
                    if (!check_insertion_o1(aux, req, k_pos, l_pos, inst))
                        continue;

                    result.cost = delta;
                    result.route_idx = k;
                    result.p_pos = k_pos;
                    result.d_pos = l_pos + 1;
                }
            }
        }

        return result;
    };

    while (!repaired.unassigned.empty()) {
        // Step 1: Refresh cache for any invalidated entries
        for (int req : repaired.unassigned) {
            if (!best_cache[req].valid) {
                best_cache[req] = scan_request(req);
            }
        }

        // Step 2: Find the globally cheapest valid feasible entry
        double best_cost = std::numeric_limits<double>::infinity();
        int best_req = -1;
        for (int req : repaired.unassigned) {
            if (best_cache[req].valid &&
                best_cache[req].route_idx >= 0 &&
                best_cache[req].cost < best_cost) {
                best_cost = best_cache[req].cost;
                best_req = req;
            }
        }

        if (best_req < 0) break;

        const BestInsertion b = best_cache[best_req];
        int k_star = b.route_idx;

        // Step 3: Safety re-validation before commit
        if (!check_insertion_o1(aux_data[k_star], best_req, b.p_pos, b.d_pos - 1, inst)) {
            best_cache[best_req].valid = false;
            continue;
        }

        // Step 4: Commit the insertion
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.p_pos, best_req);
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.d_pos, best_req + n);
        repaired.unassigned.erase(best_req);
        best_cache[best_req].valid = false;

        // Step 5: Recompute aux for the modified route
        {
            auto t0 = Clock::now();
            aux_data[k_star] = precompute_route_aux(repaired.routes[k_star], inst);
            auto t1 = Clock::now();
            DynamicStats::aux_precompute_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Step 6: Invalidate cache entries that pointed to the modified route
        for (int req : repaired.unassigned) {
            if (best_cache[req].valid && best_cache[req].route_idx == k_star) {
                best_cache[req].valid = false;
            }
        }
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Greedy insertion — slack/full feasibility path (original implementation)
// ---------------------------------------------------------------------------
static DARPSolution greedy_insertion_classic(const DARPSolution& sol, std::mt19937& rng) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    int Q = inst.vehicle_capacity;

    while (!repaired.unassigned.empty()) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_req = -1;
        int best_route_idx = -1;
        int best_p_pos = -1;
        int best_d_pos = -1;

        // Precompute forward schedules for all routes
        struct RouteCache {
            std::vector<double> depart;
            std::vector<int> load;
        };
        std::vector<RouteCache> route_caches(repaired.routes.size());
        for (int k = 0; k < static_cast<int>(repaired.routes.size()); k++) {
            forward_pass(repaired.routes[k], nodes, tt,
                         route_caches[k].depart, route_caches[k].load);
        }

        for (int req : repaired.unassigned) {
            int pickup_id = req;
            int delivery_id = req + n;

            auto candidates = build_candidate_routes(repaired, pickup_id, repaired.top_n_routes);
            DynamicStats::routes_checked += static_cast<int>(candidates.size());
            DynamicStats::request_evals++;

            for (int k : candidates) {
                const auto& route = repaired.routes[k];
                int rlen = static_cast<int>(route.size());

                int min_pos_c = inst.route_start_offset();
                if (!repaired.min_insert_pos.empty())
                    min_pos_c = std::max(min_pos_c, repaired.min_insert_pos[k]);
                if (min_pos_c >= rlen) continue;

                const auto& depart_cache = route_caches[k].depart;
                const auto& load_cache = route_caches[k].load;

                for (int p_pos = min_pos_c; p_pos < rlen; p_pos++) {
                    // Reachability filter
                    double arrival_at_pickup = depart_cache[p_pos - 1] +
                        tt[route[p_pos - 1]][pickup_id];
                    if (arrival_at_pickup > nodes[pickup_id].latest + EPS) {
                        continue;
                    }

                    for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                        // Capacity filter
                        int pickup_demand = nodes[pickup_id].demand;
                        bool cap_ok = true;
                        for (int j = p_pos - 1; j < d_pos - 1; j++) {
                            if (load_cache[j] + pickup_demand > Q) {
                                cap_ok = false;
                                break;
                            }
                        }
                        if (!cap_ok) continue;

                        // O(1) cost delta
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

                        if (delta >= best_cost) continue;

                        // Build candidate and check feasibility
                        std::vector<int> candidate = route;
                        candidate.insert(candidate.begin() + p_pos, pickup_id);
                        candidate.insert(candidate.begin() + d_pos, delivery_id);

                        auto [feasible, reason] = repaired.check_route_feasibility(candidate);
                        if (feasible) {
                            best_cost = delta;
                            best_req = req;
                            best_route_idx = k;
                            best_p_pos = p_pos;
                            best_d_pos = d_pos;
                        }
                    }
                }
            }
        }

        if (best_req < 0) break;

        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_p_pos, best_req);
        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_d_pos, best_req + n);
        repaired.unassigned.erase(best_req);
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Greedy insertion — dispatcher
// ---------------------------------------------------------------------------
DARPSolution greedy_insertion(const DARPSolution& sol, std::mt19937& rng) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    DARPSolution result = (sol.feasibility_mode == "o1")
        ? greedy_insertion_o1(sol, rng)
        : greedy_insertion_classic(sol, rng);
    auto _t1 = std::chrono::high_resolution_clock::now();
    DynamicStats::repair_greedy_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    DynamicStats::repair_greedy_count++;
    return result;
}