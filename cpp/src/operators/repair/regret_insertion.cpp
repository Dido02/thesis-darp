#include "operators/repair/regret_insertion.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <limits>
#include <vector>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Insertion option for tracking best positions
// ---------------------------------------------------------------------------
struct InsertionOption {
    double cost;
    int route_idx;
    int p_pos;
    int d_pos;  // position after pickup insertion (for actual insert)
};

// ---------------------------------------------------------------------------
// Find k best insertion options for a request — O(1) feasibility path
// ---------------------------------------------------------------------------
static std::vector<InsertionOption> find_best_insertions_o1(
    const DARPSolution& sol,
    const std::vector<RouteAuxData>& aux_cache,
    int req, int k_best) {

    const auto& inst = *sol.instance;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int pickup_id = req;
    int delivery_id = req + n;

    // We only need the top k_best, so maintain a sorted small list
    std::vector<InsertionOption> best;
    best.reserve(k_best + 1);

    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        const auto& route = sol.routes[r];
        const auto& aux = aux_cache[r];
        int rlen = static_cast<int>(route.size());

        for (int k_pos = 1; k_pos < rlen; k_pos++) {
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

                // Only keep if it could be in top k_best
                if (static_cast<int>(best.size()) < k_best || delta < best.back().cost) {
                    best.push_back({delta, r, k_pos, l_pos + 1});
                    std::sort(best.begin(), best.end(),
                              [](const auto& a, const auto& b) { return a.cost < b.cost; });
                    if (static_cast<int>(best.size()) > k_best) {
                        best.pop_back();
                    }
                }
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Find k best insertion options for a request — classic feasibility path
// ---------------------------------------------------------------------------
static std::vector<InsertionOption> find_best_insertions_classic(
    const DARPSolution& sol, int req, int k_best) {

    const auto& inst = *sol.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    int Q = inst.vehicle_capacity;
    int pickup_id = req;
    int delivery_id = req + n;

    std::vector<InsertionOption> best;
    best.reserve(k_best + 1);

    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        const auto& route = sol.routes[r];
        int rlen = static_cast<int>(route.size());

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

        for (int p_pos = 1; p_pos < rlen; p_pos++) {
            double arr_pickup = depart_cache[p_pos - 1] + tt[route[p_pos - 1]][pickup_id];
            if (arr_pickup > nodes[pickup_id].latest + EPS) continue;

            for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                bool cap_ok = true;
                for (int j = p_pos - 1; j < d_pos - 1; j++) {
                    if (load_cache[j] + 1 > Q) { cap_ok = false; break; }
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

                // Skip if can't be in top k_best
                if (static_cast<int>(best.size()) >= k_best && delta >= best.back().cost)
                    continue;

                std::vector<int> candidate = route;
                candidate.insert(candidate.begin() + p_pos, pickup_id);
                candidate.insert(candidate.begin() + d_pos, delivery_id);

                auto [feasible, reason] = sol.check_route_feasibility(candidate);
                if (feasible) {
                    best.push_back({delta, r, p_pos, d_pos});
                    std::sort(best.begin(), best.end(),
                              [](const auto& a, const auto& b) { return a.cost < b.cost; });
                    if (static_cast<int>(best.size()) > k_best) {
                        best.pop_back();
                    }
                }
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Generic regret-k insertion — O(1) path with cached aux data
// ---------------------------------------------------------------------------
static DARPSolution regret_k_o1(const DARPSolution& sol, std::mt19937& rng, int k) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    int n = inst.num_requests;

    // Precompute aux data for all routes
    std::vector<RouteAuxData> aux_cache(repaired.routes.size());
    for (int r = 0; r < static_cast<int>(repaired.routes.size()); r++) {
        aux_cache[r] = precompute_route_aux(repaired.routes[r], inst);
    }

    while (!repaired.unassigned.empty()) {
        double best_regret = -std::numeric_limits<double>::infinity();
        int best_req = -1;
        InsertionOption best_option{};
        bool found_any = false;

        for (int req : repaired.unassigned) {
            auto options = find_best_insertions_o1(repaired, aux_cache, req, k);
            if (options.empty()) continue;

            double regret = 0.0;
            for (int j = 1; j < k; j++) {
                if (j < static_cast<int>(options.size())) {
                    regret += options[j].cost - options[0].cost;
                } else {
                    regret += 1000.0;
                }
            }

            if (regret > best_regret || !found_any) {
                best_regret = regret;
                best_req = req;
                best_option = options[0];
                found_any = true;
            }
        }

        if (!found_any) break;

        repaired.routes[best_option.route_idx].insert(
            repaired.routes[best_option.route_idx].begin() + best_option.p_pos,
            best_req);
        repaired.routes[best_option.route_idx].insert(
            repaired.routes[best_option.route_idx].begin() + best_option.d_pos,
            best_req + n);
        repaired.unassigned.erase(best_req);

        // Recompute aux only for modified route
        aux_cache[best_option.route_idx] = precompute_route_aux(
            repaired.routes[best_option.route_idx], inst);
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Generic regret-k insertion — classic path
// ---------------------------------------------------------------------------
static DARPSolution regret_k_classic(const DARPSolution& sol, std::mt19937& rng, int k) {
    (void)rng;
    DARPSolution repaired = sol.copy();
    const auto& inst = *repaired.instance;
    int n = inst.num_requests;

    while (!repaired.unassigned.empty()) {
        double best_regret = -std::numeric_limits<double>::infinity();
        int best_req = -1;
        InsertionOption best_option{};
        bool found_any = false;

        for (int req : repaired.unassigned) {
            auto options = find_best_insertions_classic(repaired, req, k);
            if (options.empty()) continue;

            double regret = 0.0;
            for (int j = 1; j < k; j++) {
                if (j < static_cast<int>(options.size())) {
                    regret += options[j].cost - options[0].cost;
                } else {
                    regret += 1000.0;
                }
            }

            if (regret > best_regret || !found_any) {
                best_regret = regret;
                best_req = req;
                best_option = options[0];
                found_any = true;
            }
        }

        if (!found_any) break;

        repaired.routes[best_option.route_idx].insert(
            repaired.routes[best_option.route_idx].begin() + best_option.p_pos,
            best_req);
        repaired.routes[best_option.route_idx].insert(
            repaired.routes[best_option.route_idx].begin() + best_option.d_pos,
            best_req + n);
        repaired.unassigned.erase(best_req);
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// Public API — dispatchers
// ---------------------------------------------------------------------------
DARPSolution regret2_insertion(const DARPSolution& sol, std::mt19937& rng) {
    if (sol.feasibility_mode == "o1") {
        return regret_k_o1(sol, rng, 2);
    } else {
        return regret_k_classic(sol, rng, 2);
    }
}

DARPSolution regret3_insertion(const DARPSolution& sol, std::mt19937& rng) {
    if (sol.feasibility_mode == "o1") {
        return regret_k_o1(sol, rng, 3);
    } else {
        return regret_k_classic(sol, rng, 3);
    }
}

DARPSolution regret4_insertion(const DARPSolution& sol, std::mt19937& rng) {
    if (sol.feasibility_mode == "o1") {
        return regret_k_o1(sol, rng, 4);
    } else {
        return regret_k_classic(sol, rng, 4);
    }
}

DARPSolution regretM_insertion(const DARPSolution& sol, std::mt19937& rng) {
    int M = static_cast<int>(sol.routes.size());
    if (sol.feasibility_mode == "o1") {
        return regret_k_o1(sol, rng, M);
    } else {
        return regret_k_classic(sol, rng, M);
    }
}