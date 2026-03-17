#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// compute_aeap — Steps 1+2 only (forward pass + depot delay).
// No Step 3 per-pickup delays. The ALAP handles ride-time constraints.
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
}

// ---------------------------------------------------------------------------
// compute_alap — Latest feasible begin-service times (backward pass with
// ride-time and route-duration tightening).
// ---------------------------------------------------------------------------
void compute_alap(const std::vector<int>& route, const DARPInstance& inst,
                  const std::vector<double>& tau, const std::vector<double>& depart,
                  std::vector<double>& tau_bar) {
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());

    tau_bar.resize(size);

    if (size <= 2) {
        for (int i = 0; i < size; i++)
            tau_bar[i] = nodes[route[i]].latest;
        return;
    }

    // Build pickup/delivery position maps
    std::unordered_map<int, int> pickup_pos, delivery_pos;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) delivery_pos[nid - n] = i;
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
        for (auto& [req, p] : pickup_pos) {
            auto dit = delivery_pos.find(req);
            if (dit == delivery_pos.end()) continue;
            int d = dit->second;
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

// ---------------------------------------------------------------------------
// compute_alap_constrained — Fix position fix_pos to tau[fix_pos], propagate.
// ---------------------------------------------------------------------------
void compute_alap_constrained(const std::vector<int>& route, const DARPInstance& inst,
                              const std::vector<double>& tau, const std::vector<double>& depart,
                              const std::vector<double>& tau_bar,
                              int fix_pos, std::vector<double>& tau_bar_c) {
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());

    tau_bar_c = tau_bar;
    tau_bar_c[fix_pos] = tau[fix_pos];

    std::unordered_map<int, int> pickup_pos, delivery_pos;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) delivery_pos[nid - n] = i;
    }

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
        for (auto& [req, p] : pickup_pos) {
            auto dit = delivery_pos.find(req);
            if (dit == delivery_pos.end()) continue;
            int d = dit->second;
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

// ---------------------------------------------------------------------------
// precompute_route_aux — Master preprocessing, O(r^3) total.
// ---------------------------------------------------------------------------
RouteAuxData precompute_route_aux(const std::vector<int>& route, const DARPInstance& inst) {
    RouteAuxData aux;
    int size = static_cast<int>(route.size());
    aux.route = route;

    compute_aeap(route, inst, aux.tau, aux.depart, aux.load);
    compute_alap(route, inst, aux.tau, aux.depart, aux.tau_bar);

    // Constrained ALAP for each position
    aux.tau_bar_given.resize(size, std::vector<double>(size, 0.0));
    for (int k = 0; k < size; k++) {
        std::vector<double> tau_bar_c;
        compute_alap_constrained(route, inst, aux.tau, aux.depart, aux.tau_bar, k, tau_bar_c);
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
// Helpers
// ---------------------------------------------------------------------------
static inline double tau_bar_at(const RouteAuxData& aux, int i, double t_k, int k) {
    return std::min(aux.tau_bar[i], aux.tau_bar_given[i][k] + (t_k - aux.tau[k]));
}

static inline double tau_bar_at2(const RouteAuxData& aux, int i,
                                 double t_k, int k, double t_l, int l) {
    return std::min(tau_bar_at(aux, i, t_k, k), tau_bar_at(aux, i, t_l, l));
}

// ---------------------------------------------------------------------------
// check_insertion_o1 — O(1) feasibility check
//
// New route: route[0..k-1], pickup, route[k..l-1], delivery, route[l..end]
// k >= 1, l >= k, l <= size-1
//
// The tau_bar checks at positions k and l implicitly cover:
// - Time windows at all affected positions
// - Ride times of all existing requests
// - Route duration
// We only explicitly check: capacity, new request TWs, new request ride time.
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

    // --- Capacity ---
    if (k <= l - 1) {
        if (aux.max_load[k][l - 1] + 1 > Q) return false;
    }

    // === Consecutive case: k == l ===
    if (k == l) {
        double tau_p = std::max(pn.earliest, aux.depart[k - 1] + tt[route[k - 1]][pickup_id]);
        if (tau_p > pn.latest + EPS) return false;

        double dep_p = tau_p + pn.service_duration;
        double tau_d = std::max(dn.earliest, dep_p + tt[pickup_id][delivery_id]);
        if (tau_d > dn.latest + EPS) return false;

        double ride = tau_d - dep_p;
        if (ride > inst.max_ride_time + EPS) return false;

        double dep_d = tau_d + dn.service_duration;
        double tau_k = std::max(aux.tau[k], dep_d + tt[delivery_id][route[k]]);

        if (tau_k > tau_bar_at(aux, k, aux.tau[k - 1], k - 1) + EPS) return false;
        return true;
    }

    // === General case: k < l ===

    // --- Pickup insertion (before position k) ---
    double tau_p = std::max(pn.earliest, aux.depart[k - 1] + tt[route[k - 1]][pickup_id]);
    if (tau_p > pn.latest + EPS) return false;

    double dep_p = tau_p + pn.service_duration;
    double tau_k = std::max(aux.tau[k], dep_p + tt[pickup_id][route[k]]);

    double tb_k = tau_bar_at(aux, k, aux.tau[k - 1], k - 1);
    if (tau_k > tb_k + EPS) return false;

    // --- Propagate delay from k to l-1 ---
    double delay_k = tau_k - aux.tau[k];

    double tau_lm1;
    if (k == l - 1) {
        tau_lm1 = tau_k;
    } else {
        tau_lm1 = aux.tau[l - 1] + std::max(0.0, delay_k - aux.wait[k][l - 1]);
    }
    double dep_lm1 = tau_lm1 + nodes[route[l - 1]].service_duration;

    // --- Delivery insertion (before position l) ---
    double tau_d = std::max(dn.earliest, dep_lm1 + tt[route[l - 1]][delivery_id]);
    if (tau_d > dn.latest + EPS) return false;

    double dep_d = tau_d + dn.service_duration;

    // Shifted time at position l (from pickup delay only)
    double tau_l_shifted = aux.tau[l] + std::max(0.0, delay_k - aux.wait[k][l]);
    // After delivery insertion
    double tau_l = std::max(tau_l_shifted, dep_d + tt[delivery_id][route[l]]);

    double tb_l = tau_bar_at2(aux, l, aux.tau[k - 1], k - 1, tau_lm1, l - 1);
    if (tau_l > tb_l + EPS) return false;

    // --- New request ride time ---
    double ride = tau_d - dep_p;
    if (ride > inst.max_ride_time + EPS) {
        // Repair: delay pickup to reduce ride time
        double needed_dep_p = tau_d - inst.max_ride_time;
        double needed_tau_p = needed_dep_p - pn.service_duration;
        if (needed_tau_p > pn.latest + EPS) return false;
        if (needed_tau_p <= tau_p + EPS) return false;  // can't improve

        tau_p = needed_tau_p;
        dep_p = tau_p + pn.service_duration;

        // Recheck pickup -> position k
        tau_k = std::max(aux.tau[k], dep_p + tt[pickup_id][route[k]]);
        if (tau_k > tb_k + EPS) return false;

        // Recompute delay propagation
        delay_k = tau_k - aux.tau[k];
        if (k == l - 1) {
            tau_lm1 = tau_k;
        } else {
            tau_lm1 = aux.tau[l - 1] + std::max(0.0, delay_k - aux.wait[k][l - 1]);
        }
        dep_lm1 = tau_lm1 + nodes[route[l - 1]].service_duration;

        tau_d = std::max(dn.earliest, dep_lm1 + tt[route[l - 1]][delivery_id]);
        if (tau_d > dn.latest + EPS) return false;
        dep_d = tau_d + dn.service_duration;

        tau_l_shifted = aux.tau[l] + std::max(0.0, delay_k - aux.wait[k][l]);
        tau_l = std::max(tau_l_shifted, dep_d + tt[delivery_id][route[l]]);

        tb_l = tau_bar_at2(aux, l, aux.tau[k - 1], k - 1, tau_lm1, l - 1);
        if (tau_l > tb_l + EPS) return false;

        ride = tau_d - dep_p;
        if (ride > inst.max_ride_time + EPS) return false;
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

    // ALAP
    std::vector<double> tau_bar;
    compute_alap(route, inst, tau, depart, tau_bar);

    // Check tau[i] <= tau_bar[i] (a feasible schedule exists)
    for (int i = 0; i < size; i++) {
        if (tau[i] > tau_bar[i] + EPS) {
            return {false, "ALAP infeasible at pos " + std::to_string(i) +
                    " node " + std::to_string(route[i]) +
                    ": tau=" + std::to_string(tau[i]) +
                    " > tau_bar=" + std::to_string(tau_bar[i])};
        }
    }

    return {true, ""};
}
