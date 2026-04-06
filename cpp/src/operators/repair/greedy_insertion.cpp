#include "operators/repair/greedy_insertion.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
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

    // Lazy aux precomputation: compute on first access per route
    std::vector<RouteAuxData> aux_data(repaired.routes.size());
    std::vector<bool> aux_computed(repaired.routes.size(), false);

    while (!repaired.unassigned.empty()) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_req = -1;
        int best_route_idx = -1;
        int best_p_pos = -1;
        int best_d_pos = -1;

        for (int req : repaired.unassigned) {
            int pickup_id = req;
            int delivery_id = req + n;

            for (int k = 0; k < static_cast<int>(repaired.routes.size()); k++) {
                const auto& route = repaired.routes[k];
                int rlen = static_cast<int>(route.size());

                int min_pos = inst.route_start_offset();
                if (!repaired.min_insert_pos.empty())
                    min_pos = std::max(min_pos, repaired.min_insert_pos[k]);
                // Skip routes with no insertable positions
                if (min_pos >= rlen) continue;

                // Reachability filter
                double depart_at_prev = nodes[route[min_pos - 1]].earliest
                                      + nodes[route[min_pos - 1]].service_duration;
                if (depart_at_prev + tt[route[min_pos - 1]][pickup_id] > nodes[pickup_id].latest + EPS)
                    continue;

                // Lazy aux precomputation
                if (!aux_computed[k]) {
                    aux_data[k] = precompute_route_aux(route, inst);
                    aux_computed[k] = true;
                }
                const auto& aux = aux_data[k];

                for (int k_pos = min_pos; k_pos < rlen; k_pos++) {
                    for (int l_pos = k_pos; l_pos < rlen; l_pos++) {
                        // O(1) feasibility check
                        if (!check_insertion_o1(aux, req, k_pos, l_pos, inst))
                            continue;

                        // Compute cost delta
                        double delta;
                        if (k_pos == l_pos) {
                            // Consecutive: route[k_pos-1] -> pickup -> delivery -> route[k_pos]
                            delta = tt[route[k_pos - 1]][pickup_id]
                                  + tt[pickup_id][delivery_id]
                                  + tt[delivery_id][route[k_pos]]
                                  - tt[route[k_pos - 1]][route[k_pos]];
                        } else {
                            // General: pickup between k_pos-1 and k_pos,
                            //          delivery between l_pos-1 and l_pos
                            delta = tt[route[k_pos - 1]][pickup_id]
                                  + tt[pickup_id][route[k_pos]]
                                  - tt[route[k_pos - 1]][route[k_pos]]
                                  + tt[route[l_pos - 1]][delivery_id]
                                  + tt[delivery_id][route[l_pos]]
                                  - tt[route[l_pos - 1]][route[l_pos]];
                        }

                        if (delta < best_cost) {
                            best_cost = delta;
                            best_req = req;
                            best_route_idx = k;
                            best_p_pos = k_pos;
                            // Convert to post-pickup-insertion position for
                            // the actual insertion step
                            best_d_pos = l_pos + 1;
                        }
                    }
                }
            }
        }

        if (best_req < 0) break;  // No feasible insertion found

        // Perform the insertion
        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_p_pos, best_req);
        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_d_pos, best_req + n);
        repaired.unassigned.erase(best_req);

        // Recompute aux data only for the modified route
        aux_data[best_route_idx] = precompute_route_aux(
            repaired.routes[best_route_idx], inst);
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

            for (int k = 0; k < static_cast<int>(repaired.routes.size()); k++) {
                const auto& route = repaired.routes[k];
                int rlen = static_cast<int>(route.size());

                int min_pos_c = inst.route_start_offset();
                if (!repaired.min_insert_pos.empty())
                    min_pos_c = std::max(min_pos_c, repaired.min_insert_pos[k]);
                if (min_pos_c >= rlen) continue;

                const auto& depart_cache = route_caches[k].depart;
                const auto& load_cache = route_caches[k].load;

                // Reachability filter
                if (depart_cache[min_pos_c - 1] + tt[route[min_pos_c - 1]][pickup_id]
                    > nodes[pickup_id].latest + EPS)
                    continue;

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
    if (sol.feasibility_mode == "o1") {
        return greedy_insertion_o1(sol, rng);
    } else {
        return greedy_insertion_classic(sol, rng);
    }
}