#include "operators/repair/regret_insertion.h"
#include "operators/repair/repair_utils.h"
#include "feasibility.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <chrono>
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
    std::vector<RouteAuxData>& aux_cache,
    std::vector<bool>& aux_computed,
    int req, int k_best) {

    const auto& inst = *sol.instance;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int pickup_id = req;
    int delivery_id = req + n;

    // We only need the top k_best, so maintain a sorted small list
    std::vector<InsertionOption> best;
    best.reserve(k_best + 1);

    using Clock = std::chrono::high_resolution_clock;

    auto candidates = build_candidate_routes(sol, pickup_id, sol.top_n_routes);
    DynamicStats::routes_checked += static_cast<int>(candidates.size());
    DynamicStats::request_evals++;

    for (int r : candidates) {
        const auto& route = sol.routes[r];
        int rlen = static_cast<int>(route.size());

        int min_pos = inst.route_start_offset();
        if (!sol.min_insert_pos.empty())
            min_pos = std::max(min_pos, sol.min_insert_pos[r]);
        if (min_pos >= rlen) continue;

        // Lazy aux precomputation
        if (!aux_computed[r]) {
            auto t0 = Clock::now();
            aux_cache[r] = precompute_route_aux(route, inst);
            aux_computed[r] = true;
            auto t1 = Clock::now();
            DynamicStats::aux_precompute_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        const auto& aux = aux_cache[r];

        for (int k_pos = min_pos; k_pos < rlen; k_pos++) {
            for (int l_pos = k_pos; l_pos < rlen; l_pos++) {
                // Cost-bound pruning: compute cheap delta first
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

                // Skip if can't be in top k_best
                if (static_cast<int>(best.size()) >= k_best && delta >= best.back().cost)
                    continue;

                DynamicStats::feas_checks++;
                if (!check_insertion_o1(aux, req, k_pos, l_pos, inst))
                    continue;

                // Insert into top k_best
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

    auto candidates = build_candidate_routes(sol, pickup_id, sol.top_n_routes);
    DynamicStats::routes_checked += static_cast<int>(candidates.size());
    DynamicStats::request_evals++;

    for (int r : candidates) {
        const auto& route = sol.routes[r];
        int rlen = static_cast<int>(route.size());

        int min_pos_c = inst.route_start_offset();
        if (!sol.min_insert_pos.empty())
            min_pos_c = std::max(min_pos_c, sol.min_insert_pos[r]);
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

    // Lazy aux precomputation: compute on first access per route
    std::vector<RouteAuxData> aux_cache(repaired.routes.size());
    std::vector<bool> aux_computed(repaired.routes.size(), false);

    // Best-insertion cache: indexed by request id [0..n]
    struct RegretCache {
        std::vector<InsertionOption> options;
        double regret;
        bool valid;
    };

    std::vector<RegretCache> regret_cache(n + 1);
    for (int i = 0; i <= n; i++) {
        regret_cache[i].valid = false;
        regret_cache[i].regret = -std::numeric_limits<double>::infinity();
    }

    while (!repaired.unassigned.empty()) {
        // Step 1: Refresh cache for any invalidated entries
        for (int req : repaired.unassigned) {
            if (!regret_cache[req].valid) {
                auto options = find_best_insertions_o1(
                    repaired, aux_cache, aux_computed, req, k);

                double regret = 0.0;
                for (int j = 1; j < k; j++) {
                    if (j < static_cast<int>(options.size())) {
                        regret += options[j].cost - options[0].cost;
                    } else {
                        regret += 1000.0;
                    }
                }

                regret_cache[req].options = std::move(options);
                regret_cache[req].regret = regret;
                regret_cache[req].valid = true;
            }
        }

        // Step 2: Find highest-regret request
        double best_regret = -std::numeric_limits<double>::infinity();
        int best_req = -1;
        for (int req : repaired.unassigned) {
            if (regret_cache[req].valid &&
                !regret_cache[req].options.empty() &&
                regret_cache[req].regret > best_regret) {
                best_regret = regret_cache[req].regret;
                best_req = req;
            }
        }

        if (best_req < 0) break;

        const InsertionOption b = regret_cache[best_req].options[0];
        int k_star = b.route_idx;

        // Step 3: Safety re-validation before commit
        if (!check_insertion_o1(aux_cache[k_star], best_req, b.p_pos, b.d_pos - 1, inst)) {
            regret_cache[best_req].valid = false;
            continue;
        }

        // Step 4: Commit the insertion
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.p_pos, best_req);
        repaired.routes[k_star].insert(
            repaired.routes[k_star].begin() + b.d_pos, best_req + n);
        repaired.unassigned.erase(best_req);
        regret_cache[best_req].valid = false;

        // Step 5: Recompute aux for the modified route
        aux_cache[k_star] = precompute_route_aux(repaired.routes[k_star], inst);

        // Step 6: Invalidate cache entries where ANY option points to k_star
        for (int req : repaired.unassigned) {
            if (regret_cache[req].valid) {
                for (const auto& opt : regret_cache[req].options) {
                    if (opt.route_idx == k_star) {
                        regret_cache[req].valid = false;
                        break;
                    }
                }
            }
        }
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