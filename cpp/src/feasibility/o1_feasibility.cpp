#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// compute_aeap — Steps 1+2+3: forward pass, depot delay by forward time
// slack, and per-pickup ride-time reduction (Tang et al. 2010).
// Produces a feasible AEAP schedule that respects ride-time constraints.
// ---------------------------------------------------------------------------
void compute_aeap(const std::vector<int>& route, const DARPInstance& inst,
                  std::vector<double>& tau, std::vector<double>& depart,
                  std::vector<int>& load) {
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int size = static_cast<int>(route.size());

    tau.resize(size);
    depart.resize(size);
    load.resize(size);

    if (size <= 2) {
        for (int i = 0; i < size; i++) {
            tau[i] = nodes[route[i]].earliest;
            depart[i] = tau[i] + nodes[route[i]].service_duration;
            load[i] = 0;
        }
        return;
    }

    // --- Step 1: Forward pass ---
    std::vector<double> arrival(size, 0.0);
    tau[0] = nodes[route[0]].earliest;
    depart[0] = tau[0] + nodes[route[0]].service_duration;
    load[0] = 0;

    for (int i = 1; i < size; i++) {
        int prev = route[i - 1], curr = route[i];
        const auto& node = nodes[curr];
        arrival[i] = depart[i - 1] + tt[prev][curr];
        tau[i] = std::max(arrival[i], node.earliest);
        depart[i] = tau[i] + node.service_duration;
        load[i] = load[i - 1] + node.demand;
    }

    // --- Step 2: Delay depot departure by forward time slack ---
    std::vector<double> slack(size, 0.0);
    slack[size - 1] = nodes[route[size - 1]].latest - tau[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = tau[i + 1] - arrival[i + 1];
        slack[i] = std::min(nodes[route[i]].latest - tau[i], slack[i + 1] + wait_next);
    }

    if (slack[0] > EPS) {
        tau[0] += slack[0];
        depart[0] = tau[0] + nodes[route[0]].service_duration;
        for (int i = 1; i < size; i++) {
            int prev = route[i - 1], curr = route[i];
            const auto& node = nodes[curr];
            arrival[i] = depart[i - 1] + tt[prev][curr];
            tau[i] = std::max(arrival[i], node.earliest);
            depart[i] = tau[i] + node.service_duration;
        }
    }

    // --- Step 3: Per-pickup ride-time reduction (Tang et al. 2010) ---
    // For each pickup in route order, if the ride time exceeds L,
    // delay the pickup by absorbing downstream waiting time.
    // Iterate until no more violations (a single pass can fail when
    // delaying one pickup consumes waiting time needed by another).
    int n_req = inst.num_requests;
    // Build position map for deliveries
    std::unordered_map<int, int> del_pos_map;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= n_req + 1 && nid <= 2 * n_req)
            del_pos_map[nid] = i;
    }

    for (int pass = 0; pass < size; pass++) {
        bool any_fixed = false;
        for (int i = 0; i < size; i++) {
            int nid = route[i];
            if (nid < 1 || nid > n_req) continue;  // not a pickup

            int del_id = nid + n_req;
            auto dit = del_pos_map.find(del_id);
            if (dit == del_pos_map.end()) continue;
            int d = dit->second;

            double ride = tau[d] - depart[i];
            if (ride <= inst.max_ride_time + EPS) continue;

            double excess = ride - inst.max_ride_time;

            // Compute available waiting time between positions i+1 and d
            double wait_available = 0.0;
            for (int j = i + 1; j <= d; j++) {
                double arr_j = depart[j - 1] + tt[route[j - 1]][route[j]];
                wait_available += tau[j] - arr_j;
            }

            double delay = std::min(excess, wait_available);
            if (delay < EPS) continue;

            // Also respect the pickup's own time window
            delay = std::min(delay, nodes[nid].latest - tau[i]);
            if (delay < EPS) continue;

            // Apply delay and propagate forward
            tau[i] += delay;
            depart[i] = tau[i] + nodes[nid].service_duration;
            for (int j = i + 1; j < size; j++) {
                int prev_j = route[j - 1], curr_j = route[j];
                arrival[j] = depart[j - 1] + tt[prev_j][curr_j];
                tau[j] = std::max(arrival[j], nodes[curr_j].earliest);
                depart[j] = tau[j] + nodes[curr_j].service_duration;
            }
            any_fixed = true;
        }
        if (!any_fixed) break;
    }
}

// ---------------------------------------------------------------------------
// compute_alap — Latest feasible begin-service times (backward pass with
// ride-time and route-duration tightening).
//
// This is an iterative fixed-point computation equivalent to running the
// "reverse Tang et al." algorithm. Converges in at most r iterations.
//
// pickup_pairs/delivery_pairs: pre-built arrays of (pickup_pos, delivery_pos)
// for each request present in the route. Built once in precompute_route_aux.
// ---------------------------------------------------------------------------
static void compute_alap_internal(const std::vector<int>& route, const DARPInstance& inst,
                  const std::vector<double>& tau, const std::vector<double>& depart,
                  std::vector<double>& tau_bar,
                  const std::vector<std::pair<int,int>>& pd_pairs) {
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int size = static_cast<int>(route.size());

    tau_bar.resize(size);

    if (size <= 2) {
        for (int i = 0; i < size; i++)
            tau_bar[i] = nodes[route[i]].latest;
        return;
    }

    // Iterative tightening loop
    for (int iter = 0; iter < size + 2; iter++) {
        bool changed = false;

        // Backward pass
        double new_val = nodes[route[size - 1]].latest;
        if (iter == 0) tau_bar[size - 1] = new_val;
        for (int i = size - 2; i >= 0; i--) {
            double limit = tau_bar[i + 1] - tt[route[i]][route[i + 1]]
                           - nodes[route[i]].service_duration;
            new_val = std::min(nodes[route[i]].latest, limit);
            if (iter == 0) {
                tau_bar[i] = new_val;
            } else if (new_val < tau_bar[i] - EPS) {
                tau_bar[i] = new_val;
                changed = true;
            }
        }

        // Ride-time tightening
        for (const auto& [p, d] : pd_pairs) {
            double s_p = nodes[route[p]].service_duration;
            double ride = tau_bar[d] - (tau_bar[p] + s_p);
            if (ride > inst.max_ride_time + EPS) {
                double new_tb = tau_bar[p] + s_p + inst.max_ride_time;
                if (new_tb < tau_bar[d] - EPS) {
                    tau_bar[d] = new_tb;
                    changed = true;
                }
            }
        }

        // Route duration tightening
        {
            double s0 = nodes[route[0]].service_duration;
            double dur = tau_bar[size - 1] - (tau_bar[0] + s0);
            if (dur > inst.max_route_duration + EPS) {
                double new_end = tau_bar[0] + s0 + inst.max_route_duration;
                if (new_end < tau_bar[size - 1] - EPS) {
                    tau_bar[size - 1] = new_end;
                    changed = true;
                }
            }
        }

        if (iter > 0 && !changed) break;
    }

    // Clamp: tau_bar[i] >= tau[i]
    for (int i = 0; i < size; i++)
        tau_bar[i] = std::max(tau_bar[i], tau[i]);
}

// Public wrapper that builds pd_pairs (used by check_alap_feasible)
void compute_alap(const std::vector<int>& route, const DARPInstance& inst,
                  const std::vector<double>& tau, const std::vector<double>& depart,
                  std::vector<double>& tau_bar) {
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());
    std::vector<int> pickup_pos(n + 1, -1);
    std::vector<std::pair<int,int>> pd_pairs;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) {
            int req = nid - n;
            if (pickup_pos[req] >= 0)
                pd_pairs.emplace_back(pickup_pos[req], i);
        }
    }
    compute_alap_internal(route, inst, tau, depart, tau_bar, pd_pairs);
}

// ---------------------------------------------------------------------------
// compute_alap_constrained — Fix position fix_pos to tau[fix_pos], propagate.
// Reuses pre-built pd_pairs and a caller-provided scratch vector.
// ---------------------------------------------------------------------------
static void compute_alap_constrained_fast(
    const std::vector<int>& route, const DARPInstance& inst,
    const std::vector<double>& tau,
    const std::vector<double>& tau_bar,
    const std::vector<std::pair<int,int>>& pd_pairs,
    int fix_pos, std::vector<double>& tau_bar_c) {

    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int size = static_cast<int>(route.size());

    // Reset to tau_bar, then fix
    for (int i = 0; i < size; i++) tau_bar_c[i] = tau_bar[i];
    tau_bar_c[fix_pos] = tau[fix_pos];

    for (int iter = 0; iter < size + 2; iter++) {
        bool changed = false;

        // Full backward pass
        for (int i = size - 2; i >= 0; i--) {
            double limit = tau_bar_c[i + 1] - tt[route[i]][route[i + 1]]
                           - nodes[route[i]].service_duration;
            double new_val = std::min(tau_bar_c[i], limit);
            if (new_val < tau_bar_c[i] - EPS) {
                tau_bar_c[i] = new_val;
                changed = true;
            }
        }

        // Ride-time tightening
        for (const auto& [p, d] : pd_pairs) {
            double s_p = nodes[route[p]].service_duration;
            double ride = tau_bar_c[d] - (tau_bar_c[p] + s_p);
            if (ride > inst.max_ride_time + EPS) {
                double new_tb = tau_bar_c[p] + s_p + inst.max_ride_time;
                if (new_tb < tau_bar_c[d] - EPS) {
                    tau_bar_c[d] = new_tb;
                    changed = true;
                }
            }
        }

        // Route duration
        {
            double s0 = nodes[route[0]].service_duration;
            double dur = tau_bar_c[size - 1] - (tau_bar_c[0] + s0);
            if (dur > inst.max_route_duration + EPS) {
                double new_end = tau_bar_c[0] + s0 + inst.max_route_duration;
                if (new_end < tau_bar_c[size - 1] - EPS) {
                    tau_bar_c[size - 1] = new_end;
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }

    for (int i = 0; i < size; i++)
        tau_bar_c[i] = std::max(tau_bar_c[i], tau[i]);
}

// Keep old signature for check_alap_feasible (called rarely)
void compute_alap_constrained(const std::vector<int>& route, const DARPInstance& inst,
                              const std::vector<double>& tau, const std::vector<double>& depart,
                              const std::vector<double>& tau_bar,
                              int fix_pos, std::vector<double>& tau_bar_c) {
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());
    std::vector<int> pickup_pos(n + 1, -1);
    std::vector<std::pair<int,int>> pd_pairs;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) {
            int req = nid - n;
            if (pickup_pos[req] >= 0)
                pd_pairs.emplace_back(pickup_pos[req], i);
        }
    }
    tau_bar_c.resize(size);
    compute_alap_constrained_fast(route, inst, tau, tau_bar, pd_pairs, fix_pos, tau_bar_c);
}

// ---------------------------------------------------------------------------
// precompute_route_aux — Master preprocessing, O(r^3) total.
// Optimized: builds pickup/delivery maps once, reuses scratch vectors.
// ---------------------------------------------------------------------------
RouteAuxData precompute_route_aux(const std::vector<int>& route, const DARPInstance& inst) {
    RouteAuxData aux;
    int size = static_cast<int>(route.size());
    int n = inst.num_requests;
    aux.route = route;

    compute_aeap(route, inst, aux.tau, aux.depart, aux.load);

    // Build pickup/delivery pair list once
    std::vector<int> pickup_pos(n + 1, -1);
    std::vector<std::pair<int,int>> pd_pairs;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) {
            int req = nid - n;
            if (pickup_pos[req] >= 0)
                pd_pairs.emplace_back(pickup_pos[req], i);
        }
    }

    compute_alap_internal(route, inst, aux.tau, aux.depart, aux.tau_bar, pd_pairs);

    // Constrained ALAP for each position — reuse scratch vector
    aux.tau_bar_given.resize(size, std::vector<double>(size, 0.0));
    std::vector<double> tau_bar_c(size);
    for (int k = 0; k < size; k++) {
        compute_alap_constrained_fast(route, inst, aux.tau, aux.tau_bar, pd_pairs, k, tau_bar_c);
        for (int i = 0; i < size; i++)
            aux.tau_bar_given[i][k] = tau_bar_c[i];
    }

    // gamma[i][j] = cumulative travel time from i to j along route
    aux.gamma.resize(size, std::vector<double>(size, 0.0));
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++)
            aux.gamma[i][j] = aux.gamma[i][j - 1] + inst.travel_time[route[j - 1]][route[j]];
    }

    // wait[i][j] = cumulative waiting time
    aux.wait.resize(size, std::vector<double>(size, 0.0));
    {
        std::vector<double> ps(size + 1, 0.0);
        for (int i = 0; i < size; i++)
            ps[i + 1] = ps[i] + inst.nodes[route[i]].service_duration;
        for (int i = 0; i < size; i++)
            for (int j = i + 1; j < size; j++)
                aux.wait[i][j] = aux.tau[j] - aux.tau[i] - aux.gamma[i][j] - (ps[j] - ps[i]);
    }

    // max_load[i][j] = max(load[i..j])
    aux.max_load.resize(size, std::vector<int>(size, 0));
    for (int i = 0; i < size; i++) {
        aux.max_load[i][i] = aux.load[i];
        for (int j = i + 1; j < size; j++)
            aux.max_load[i][j] = std::max(aux.max_load[i][j - 1], aux.load[j]);
    }

    return aux;
}

// ---------------------------------------------------------------------------
// Helpers — Proposition 1(ii) and (iii) from Gschwind & Drexl (2016).
//
// tau_bar_at(aux, i, t_k, k) implements:
//   τ̄_i(t_k) = min{ τ̄_i(τ̄_k),  τ̄_i(τ_k) + (t_k - τ_k) }
//            = min{ τ̄_i,          tau_bar_given[i][k] + (t_k - τ_k) }
// ---------------------------------------------------------------------------
static inline double tau_bar_at(const RouteAuxData& aux, int i, double t_k, int k) {
    return std::min(aux.tau_bar[i], aux.tau_bar_given[i][k] + (t_k - aux.tau[k]));
}

static inline double tau_bar_at2(const RouteAuxData& aux, int i,
                                 double t_k, int k, double t_l, int l) {
    return std::min(tau_bar_at(aux, i, t_k, k), tau_bar_at(aux, i, t_l, l));
}

// ---------------------------------------------------------------------------
// check_insertion_o1 — O(1) feasibility check following Algorithms 2 & 3
// from Gschwind & Drexl (2016).
//
// Inserts pickup of request req before position k, delivery before position l.
// New route: route[0..k-1], pickup, route[k..l-1], delivery, route[l..end]
//
// Paper is 1-indexed (v1..vr), pickup inserted after vk. Our code is
// 0-indexed, pickup inserted before position k. So:
//   Paper's vk = route[k-1], vk+1 = route[k], tau_k = aux.tau[k-1], etc.
//
// Convention: paper's t_{ij} includes service time at i.
//   paper's t(v_k, i+) = s_{v_k} + travel(v_k, i+)
//   code uses tt[][] = pure travel time, and adds service_duration explicitly.
// ---------------------------------------------------------------------------
bool check_insertion_o1(const RouteAuxData& aux, int req, int k, int l,
                        const DARPInstance& inst) {
    const auto& route = aux.route;
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int Q = inst.vehicle_capacity;

    int pickup_id = req;
    int delivery_id = req + n;
    const auto& pn = nodes[pickup_id];
    const auto& dn = nodes[delivery_id];
    double s_p = pn.service_duration;
    double s_d = dn.service_duration;

    // --- Capacity check ---
    if (k <= l - 1) {
        if (aux.max_load[k][l - 1] + 1 > Q) return false;
    }

    // ===================================================================
    // Consecutive case: k == l
    // New route: route[0..k-1], pickup, delivery, route[k..end]
    //
    // Paper (page 9): Like Algorithm 2 but insert i- between Lines 2 & 3,
    // then replace i+ with i- in Lines 3-5. Line 6 checks tau_bar bound.
    // Ride time of request i is automatically satisfied if time-window
    // tightening was applied in preprocessing.
    // ===================================================================
    if (k == l) {
        // Alg2 Line 1: earliest service at pickup
        double tau_p = std::max(pn.earliest,
                                aux.depart[k - 1] + tt[route[k - 1]][pickup_id]);
        // Alg2 Line 2: TW check on pickup
        if (tau_p > pn.latest + EPS) return false;

        // Insert delivery (between Lines 2 and 3)
        double dep_p = tau_p + s_p;
        double tau_d = std::max(dn.earliest, dep_p + tt[pickup_id][delivery_id]);
        if (tau_d > dn.latest + EPS) return false;

        // Check ride time (in case TW tightening was not done)
        double ride = tau_d - dep_p;
        if (ride > inst.max_ride_time + EPS) return false;

        // Alg2 Line 3 (with i- replacing i+): earliest service at vk+1
        double dep_d = tau_d + s_d;
        double tau_k_new = std::max(static_cast<double>(aux.tau[k]),
                                     dep_d + tt[delivery_id][route[k]]);

        // Alg2 Line 4 (with i- replacing i+): maximize tau_d given tau_k_new
        double max_tau_d = std::min(dn.latest,
                                     tau_k_new - tt[delivery_id][route[k]] - s_d);

        // Alg2 Line 5 (with i- replacing i+): maximize tau_km1 given max_tau_d
        // Chain backward: vk -> i+ -> i- -> vk+1
        // max_tau_km1 = min(tau_bar[k-1], max_tau_d - t(vk, i-))
        // where t(vk, i-) = s_vk + travel(vk, i+) + s_i+ + travel(i+, i-)
        // But we go through the chain: max_tau_d -> max_tau_p -> max_tau_km1
        double max_tau_p = std::min(pn.latest,
                                     max_tau_d - tt[pickup_id][delivery_id] - s_p);
        double max_tau_km1 = std::min(aux.tau_bar[k - 1],
                                       max_tau_p - tt[route[k - 1]][pickup_id]
                                       - nodes[route[k - 1]].service_duration);

        // Alg2 Line 6: bound check
        if (tau_k_new > tau_bar_at(aux, k, max_tau_km1, k - 1) + EPS) return false;

        return true;
    }

    // ===================================================================
    // General case: k < l
    // Phase 1: Algorithm 2 — pickup insertion
    // Phase 2: Algorithm 3 — delivery insertion
    // ===================================================================

    // --- Phase 1: Algorithm 2 ---

    // Line 1: earliest service at pickup
    double tau_p = std::max(pn.earliest,
                            aux.depart[k - 1] + tt[route[k - 1]][pickup_id]);
    // Line 2: TW check
    if (tau_p > pn.latest + EPS) return false;

    // Line 3: earliest service at successor route[k]
    double dep_p = tau_p + s_p;
    double tau_k_new = std::max(static_cast<double>(aux.tau[k]),
                                 dep_p + tt[pickup_id][route[k]]);

    // Line 4: maximize pickup time given tau_k_new
    double max_tau_p = std::min(pn.latest,
                                 tau_k_new - tt[pickup_id][route[k]] - s_p);

    // Line 5: maximize vk time given max_tau_p
    double max_tau_km1 = std::min(aux.tau_bar[k - 1],
                                   max_tau_p - tt[route[k - 1]][pickup_id]
                                   - nodes[route[k - 1]].service_duration);

    // Line 6: bound check at route[k]
    double tb_k = tau_bar_at(aux, k, max_tau_km1, k - 1);
    if (tau_k_new > tb_k + EPS) return false;

    // --- Phase 2: Algorithm 3 ---

    // Line 1: propagate delay from route[k] to route[l-1]
    double delay_k = tau_k_new - aux.tau[k];

    double tau_lm1;
    if (k == l - 1) {
        // k+1 == l: vl is route[k], service time already known
        tau_lm1 = tau_k_new;
    } else {
        tau_lm1 = aux.tau[l - 1] + std::max(0.0, delay_k - aux.wait[k][l - 1]);
    }
    double dep_lm1 = tau_lm1 + nodes[route[l - 1]].service_duration;

    // Line 2: earliest service at delivery
    double tau_d = std::max(dn.earliest, dep_lm1 + tt[route[l - 1]][delivery_id]);
    // Line 3: TW check
    if (tau_d > dn.latest + EPS) return false;

    // Line 4: earliest service at successor route[l]
    double dep_d = tau_d + s_d;
    double tau_l_shifted = aux.tau[l] + std::max(0.0, delay_k - aux.wait[k][l]);
    double tau_l_new = std::max(tau_l_shifted, dep_d + tt[delivery_id][route[l]]);

    // Line 5: maximize tau at route[l-1]
    double max_tau_lm1 = std::min(
        tau_bar_at(aux, l - 1, max_tau_km1, k - 1),
        tau_d - tt[route[l - 1]][delivery_id] - nodes[route[l - 1]].service_duration);

    // Line 6: bound check at route[l]
    double tb_l = tau_bar_at2(aux, l, max_tau_km1, k - 1, max_tau_lm1, l - 1);
    double delta_l = tau_l_new - tb_l;

    // Lines 7-14: delta_l repair
    if (delta_l > EPS) {
        // Compute w_{k+1,i-} on the new route (or w_{i+,i-} for k+1==l).
        // Paper page 10: for k+1==l, use w_{i+,i-} instead of w_{k+1,i-}.
        double w_to_delivery;
        if (k == l - 1) {
            // FIX: w_{i+,i-} = waiting between i+ and v_{k+1} + waiting between v_l and i-
            // After Alg3 Line 5 increases tau'_l, there may be waiting between i+ and v_{k+1}.
            double wait_ip_to_vk1 = std::max(0.0, tau_lm1 - (max_tau_p + s_p + tt[pickup_id][route[k]]));
            double wait_vl_to_im  = std::max(0.0, tau_d - (dep_lm1 + tt[route[l - 1]][delivery_id]));
            w_to_delivery = wait_ip_to_vk1 + wait_vl_to_im;
        } else {
            double wait_k_to_lm1 = aux.wait[k][l - 1];
            double wait_at_d = tau_d - (dep_lm1 + tt[route[l - 1]][delivery_id]);
            w_to_delivery = std::max(0.0, wait_k_to_lm1 - delay_k) + std::max(0.0, wait_at_d);
        }

        // Line 8: check repair feasibility
        if (max_tau_km1 + delta_l > aux.tau_bar[k - 1] + EPS) return false;
        if (max_tau_p + delta_l > pn.latest + EPS) return false;
        if (delta_l > w_to_delivery + EPS) return false;

        // Lines 9-11: increase service times
        max_tau_km1 += delta_l;
        max_tau_p += delta_l;
        tau_k_new += delta_l;

        // Line 12: re-check at route[k]
        // For k+1 == l: use max instead of simple addition (paper page 10)
        if (k == l - 1) {
            tau_k_new = std::max(tau_k_new,
                                  max_tau_p + s_p + tt[pickup_id][route[k]]);
        }

        tb_k = tau_bar_at(aux, k, max_tau_km1, k - 1);
        if (tau_k_new > tb_k + EPS) return false;

        // Line 13: recompute with new times
        delay_k = tau_k_new - aux.tau[k];
        if (k == l - 1) {
            tau_lm1 = tau_k_new;
        } else {
            tau_lm1 = aux.tau[l - 1] + std::max(0.0, delay_k - aux.wait[k][l - 1]);
        }
        dep_lm1 = tau_lm1 + nodes[route[l - 1]].service_duration;

        tau_d = std::max(dn.earliest, dep_lm1 + tt[route[l - 1]][delivery_id]);
        if (tau_d > dn.latest + EPS) return false;
        dep_d = tau_d + s_d;

        max_tau_lm1 = std::min(
            tau_bar_at(aux, l - 1, max_tau_km1, k - 1),
            tau_d - tt[route[l - 1]][delivery_id] - nodes[route[l - 1]].service_duration);

        // Line 14: re-check at route[l]
        tau_l_shifted = aux.tau[l] + std::max(0.0, delay_k - aux.wait[k][l]);
        tau_l_new = std::max(tau_l_shifted, dep_d + tt[delivery_id][route[l]]);

        tb_l = tau_bar_at2(aux, l, max_tau_km1, k - 1, max_tau_lm1, l - 1);
        if (tau_l_new > tb_l + EPS) return false;
    }

    // --- Lines 15-21: ride-time check and repair ---
    // After delta_l repair, the paper's τ'_{i+} was increased (Line 10).
    // Update tau_p to reflect that the pickup can no longer be served at
    // the original earliest time — the delta_l repair committed to a later time.
    // Use max_tau_p as the effective pickup time (it was increased by delta_l if repair occurred).
    // This matches the paper where τ'_{i+} is a single variable overwritten by Line 4 and Line 10.
    tau_p = max_tau_p;
    double ride = tau_d - (tau_p + s_p);

    if (ride > inst.max_ride_time + EPS) {
        // Line 16: needed delay
        double delta_i = ride - inst.max_ride_time;

        // Recompute w for ride-time repair (same logic as above)
        double w_check;
        if (k == l - 1) {
            // FIX: w_{i+,i-} after any previous repairs
            double wait_ip_to_vk1 = std::max(0.0, tau_lm1 - (tau_p + s_p + tt[pickup_id][route[k]]));
            double wait_vl_to_im  = std::max(0.0, tau_d - (dep_lm1 + tt[route[l - 1]][delivery_id]));
            w_check = wait_ip_to_vk1 + wait_vl_to_im;
        } else {
            double residual_wait = std::max(0.0, aux.wait[k][l - 1] - delay_k);
            double wait_at_d = std::max(0.0, tau_d - (dep_lm1 + tt[route[l - 1]][delivery_id]));
            w_check = residual_wait + wait_at_d;
        }

        // Line 17: check repair feasibility
        if (tau_p + delta_i > pn.latest + EPS) return false;
        if (delta_i > w_check + EPS) return false;

        // Line 18: increase pickup time
        tau_p += delta_i;

        // Line 19: increase tau_k_new
        tau_k_new += delta_i;
        // For k+1 == l: use max (paper page 10)
        if (k == l - 1) {
            tau_k_new = std::max(tau_k_new,
                                  tau_p + s_p + tt[pickup_id][route[k]]);
        }

        // Line 20: maximize vk given new pickup time
        max_tau_km1 = std::min(aux.tau_bar[k - 1],
                                tau_p - tt[route[k - 1]][pickup_id]
                                - nodes[route[k - 1]].service_duration);

        // Line 21: re-check at route[k]
        tb_k = tau_bar_at(aux, k, max_tau_km1, k - 1);
        if (tau_k_new > tb_k + EPS) return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// check_alap_feasible — ALAP-based feasibility: does a feasible schedule exist?
// Uses Steps 1+2 AEAP + ALAP. If tau[i] <= tau_bar[i] for all i, a feasible
// schedule exists (the ALAP encodes ride-time + route-duration constraints).
// ---------------------------------------------------------------------------
std::pair<bool, std::string> check_alap_feasible(
    const std::vector<int>& route, const DARPInstance& inst) {

    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    int Q = inst.vehicle_capacity;
    int size = static_cast<int>(route.size());

    if (size <= 2) return {true, ""};

    std::vector<double> tau, depart;
    std::vector<int> load;
    compute_aeap(route, inst, tau, depart, load);

    // Capacity
    for (int i = 0; i < size; i++) {
        if (load[i] < 0 || load[i] > Q) {
            return {false, "Capacity violated at node " + std::to_string(route[i])};
        }
    }

    // Time windows (AEAP must not exceed latest)
    for (int i = 0; i < size; i++) {
        if (tau[i] > nodes[route[i]].latest + EPS) {
            return {false, "Time window violated at node " + std::to_string(route[i])};
        }
    }

    // Precedence
    std::unordered_map<int, bool> seen_pickup;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) seen_pickup[nid] = true;
        else if (nid >= n + 1 && nid <= 2 * n) {
            if (!seen_pickup.count(nid - n))
                return {false, "Precedence violated"};
        }
    }

    // Compute unclamped ALAP to detect infeasibility.
    // (compute_alap clamps tau_bar[i] >= tau[i] for interpolation use,
    // but we need the unclamped values to detect infeasibility.)
    std::vector<double> tau_bar_raw(size);

    // Initialize from time window upper bounds
    tau_bar_raw[size - 1] = nodes[route[size - 1]].latest;
    for (int i = size - 2; i >= 0; i--) {
        double limit = tau_bar_raw[i + 1] - inst.travel_time[route[i]][route[i + 1]]
                       - nodes[route[i]].service_duration;
        tau_bar_raw[i] = std::min(nodes[route[i]].latest, limit);
    }

    // Build pickup/delivery maps for ride-time tightening
    std::unordered_map<int, int> pickup_pos2, delivery_pos2;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos2[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) delivery_pos2[nid - n] = i;
    }

    // Iterative tightening (same as compute_alap but without final clamp)
    for (int iter = 0; iter < size + 2; iter++) {
        bool changed = false;

        // Backward pass
        for (int i = size - 2; i >= 0; i--) {
            double limit = tau_bar_raw[i + 1] - inst.travel_time[route[i]][route[i + 1]]
                           - nodes[route[i]].service_duration;
            double new_val = std::min(tau_bar_raw[i], limit);
            if (new_val < tau_bar_raw[i] - EPS) {
                tau_bar_raw[i] = new_val;
                changed = true;
            }
        }

        // Ride-time tightening
        for (auto& [req, p] : pickup_pos2) {
            auto dit = delivery_pos2.find(req);
            if (dit == delivery_pos2.end()) continue;
            int d = dit->second;
            double s_p = nodes[route[p]].service_duration;
            double ride = tau_bar_raw[d] - (tau_bar_raw[p] + s_p);
            if (ride > inst.max_ride_time + EPS) {
                double new_tb = tau_bar_raw[p] + s_p + inst.max_ride_time;
                if (new_tb < tau_bar_raw[d] - EPS) {
                    tau_bar_raw[d] = new_tb;
                    changed = true;
                }
            }
        }

        // Route duration tightening
        {
            double s0 = nodes[route[0]].service_duration;
            double dur = tau_bar_raw[size - 1] - (tau_bar_raw[0] + s0);
            if (dur > inst.max_route_duration + EPS) {
                double new_end = tau_bar_raw[0] + s0 + inst.max_route_duration;
                if (new_end < tau_bar_raw[size - 1] - EPS) {
                    tau_bar_raw[size - 1] = new_end;
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }

    // Check tau[i] <= tau_bar_raw[i] (unclamped — detects true infeasibility)
    for (int i = 0; i < size; i++) {
        if (tau[i] > tau_bar_raw[i] + EPS) {
            return {false, "ALAP infeasible at pos " + std::to_string(i) +
                    " node " + std::to_string(route[i]) +
                    ": tau=" + std::to_string(tau[i]) +
                    " > tau_bar=" + std::to_string(tau_bar_raw[i])};
        }
    }

    return {true, ""};
}