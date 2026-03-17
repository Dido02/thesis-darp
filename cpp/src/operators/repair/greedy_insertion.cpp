#include "operators/repair/greedy_insertion.h"
#include "feasibility.h"

#include <algorithm>
#include <limits>
#include <vector>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Forward pass helper for greedy insertion
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
// Greedy insertion
// ---------------------------------------------------------------------------
DARPSolution greedy_insertion(const DARPSolution& sol, std::mt19937& rng) {
    (void)rng;  // not used, but matches ALNS operator signature
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
                const auto& depart_cache = route_caches[k].depart;
                const auto& load_cache = route_caches[k].load;
                int rlen = static_cast<int>(route.size());

                for (int p_pos = 1; p_pos < rlen; p_pos++) {
                    // Reachability filter
                    double arrival_at_pickup = depart_cache[p_pos - 1] +
                        tt[route[p_pos - 1]][pickup_id];
                    if (arrival_at_pickup > nodes[pickup_id].latest + EPS) {
                        continue;
                    }

                    for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                        // Capacity filter
                        bool cap_ok = true;
                        for (int j = p_pos - 1; j < d_pos - 1; j++) {
                            if (load_cache[j] + 1 > Q) {
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

        if (best_req < 0) break;  // No feasible insertion

        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_p_pos, best_req);
        repaired.routes[best_route_idx].insert(
            repaired.routes[best_route_idx].begin() + best_d_pos, best_req + n);
        repaired.unassigned.erase(best_req);
    }

    return repaired;
}
