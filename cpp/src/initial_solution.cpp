#include "initial_solution.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <limits>
#include <vector>

static constexpr double EPS = 1e-6;

// Forward pass: compute departure times and loads
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

DARPSolution build_initial_solution(const DARPInstance& inst,
                                     const std::string& feasibility_mode) {
    DARPSolution sol(inst);
    sol.feasibility_mode = feasibility_mode;
    int n = inst.num_requests;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int Q = inst.vehicle_capacity;

    bool use_o1 = (feasibility_mode == "o1");

    // Sort requests by time window tightness (tightest first)
    // Constrained requests get inserted first while routes have room
    std::vector<int> request_order(n);
    for (int i = 0; i < n; i++) request_order[i] = i + 1;
    std::sort(request_order.begin(), request_order.end(),
        [&nodes](int a, int b) {
            double slack_a = nodes[a].latest - nodes[a].earliest;
            double slack_b = nodes[b].latest - nodes[b].earliest;
            return slack_a < slack_b;
        });

    // Aux data cache for O(1) mode
    std::vector<RouteAuxData> aux_cache;
    if (use_o1) {
        aux_cache.resize(sol.routes.size());
        for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
            aux_cache[k] = precompute_route_aux(sol.routes[k], inst);
        }
    }

    for (int req : request_order) {
        int pickup_id = req;
        int delivery_id = req + n;

        double best_cost = std::numeric_limits<double>::infinity();
        int best_route_idx = -1;
        int best_p_pos = -1;
        int best_d_pos = -1;

        if (use_o1) {
            // --- O(1) feasibility path ---
            for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
                const auto& route = sol.routes[k];
                const auto& aux = aux_cache[k];
                int rlen = static_cast<int>(route.size());

                for (int k_pos = inst.route_start_offset(); k_pos < rlen; k_pos++) {
                    for (int l_pos = k_pos; l_pos < rlen; l_pos++) {
                        if (!check_insertion_o1(aux, req, k_pos, l_pos, inst))
                            continue;

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

                        if (delta < best_cost) {
                            best_cost = delta;
                            best_route_idx = k;
                            best_p_pos = k_pos;
                            best_d_pos = l_pos + 1;
                        }
                    }
                }
            }
        } else {
            // --- Classic feasibility path ---
            struct RouteCache {
                std::vector<double> depart;
                std::vector<int> load;
            };
            std::vector<RouteCache> route_caches(sol.routes.size());
            for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
                forward_pass(sol.routes[k], nodes, tt,
                             route_caches[k].depart, route_caches[k].load);
            }

            for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
                const auto& route = sol.routes[k];
                const auto& depart_cache = route_caches[k].depart;
                const auto& load_cache = route_caches[k].load;
                int rlen = static_cast<int>(route.size());

                for (int p_pos = inst.route_start_offset(); p_pos < rlen; p_pos++) {
                    double arrival_at_pickup = depart_cache[p_pos - 1] +
                        tt[route[p_pos - 1]][pickup_id];
                    if (arrival_at_pickup > nodes[pickup_id].latest + EPS)
                        continue;

                    for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                        bool cap_ok = true;
                        for (int j = p_pos - 1; j < d_pos - 1; j++) {
                            if (load_cache[j] + 1 > Q) {
                                cap_ok = false;
                                break;
                            }
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

                        if (delta >= best_cost) continue;

                        std::vector<int> candidate = route;
                        candidate.insert(candidate.begin() + p_pos, pickup_id);
                        candidate.insert(candidate.begin() + d_pos, delivery_id);

                        auto [feasible, reason] = sol.check_route_feasibility(candidate);
                        if (feasible) {
                            best_cost = delta;
                            best_route_idx = k;
                            best_p_pos = p_pos;
                            best_d_pos = d_pos;
                        }
                    }
                }
            }
        }

        if (best_route_idx >= 0) {
            sol.routes[best_route_idx].insert(
                sol.routes[best_route_idx].begin() + best_p_pos, pickup_id);
            sol.routes[best_route_idx].insert(
                sol.routes[best_route_idx].begin() + best_d_pos, delivery_id);
            sol.unassigned.erase(req);

            // Update aux data for modified route
            if (use_o1) {
                aux_cache[best_route_idx] = precompute_route_aux(
                    sol.routes[best_route_idx], inst);
            }
        }
    }

    return sol;
}