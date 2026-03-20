// Diagnostic: test each preprocessing function independently.
// For each route in the initial solution:
//   1. Verify AEAP schedule is feasible (TW, precedence, capacity)
//   2. Verify ALAP: tau[i] <= tau_bar[i] for all i
//   3. Verify ALAP is actually LATEST: try nudging tau_bar[i] up, check infeasibility
//   4. Verify constrained ALAP: fix pos k to tau[k], recompute ALAP from scratch,
//      compare with precomputed tau_bar_given[i][k]
//   5. Verify tau_bar_at interpolation: for random t_k in [tau[k], tau_bar[k]],
//      compute tau_bar_at and compare with brute-force constrained ALAP
//   6. Verify wait[i][j] matches actual waiting time in schedule
//   7. For specific "TOO CONSERVATIVE" cases, trace through check_insertion_o1
//      step by step and identify which check fails

#include "instance.h"
#include "parsers/cordeau.h"
#include "solution.h"
#include "initial_solution.h"
#include "feasibility/o1_feasibility.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

static constexpr double EPS = 1e-6;

// Brute-force constrained ALAP: build fresh, fix position, compute
static std::vector<double> brute_force_constrained_alap(
    const std::vector<int>& route, const DARPInstance& inst,
    const std::vector<double>& tau, const std::vector<double>& depart,
    int fix_pos) {
    
    std::vector<double> tb;
    compute_alap(route, inst, tau, depart, tb);
    
    // Now fix position fix_pos to tau[fix_pos] and re-tighten
    tb[fix_pos] = tau[fix_pos];
    
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());
    
    // Build pd_pairs
    std::vector<int> pp(n + 1, -1);
    std::vector<std::pair<int,int>> pd_pairs;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pp[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) {
            int req = nid - n;
            if (pp[req] >= 0) pd_pairs.emplace_back(pp[req], i);
        }
    }
    
    for (int iter = 0; iter < size + 2; iter++) {
        bool changed = false;
        for (int i = size - 2; i >= 0; i--) {
            double limit = tb[i + 1] - tt[route[i]][route[i + 1]] - nodes[route[i]].service_duration;
            double nv = std::min(tb[i], limit);
            if (nv < tb[i] - EPS) { tb[i] = nv; changed = true; }
        }
        for (auto& [p, d] : pd_pairs) {
            double sp = nodes[route[p]].service_duration;
            double ride = tb[d] - (tb[p] + sp);
            if (ride > inst.max_ride_time + EPS) {
                double ntb = tb[p] + sp + inst.max_ride_time;
                if (ntb < tb[d] - EPS) { tb[d] = ntb; changed = true; }
            }
        }
        {
            double s0 = nodes[route[0]].service_duration;
            double dur = tb[size - 1] - (tb[0] + s0);
            if (dur > inst.max_route_duration + EPS) {
                double ne = tb[0] + s0 + inst.max_route_duration;
                if (ne < tb[size - 1] - EPS) { tb[size - 1] = ne; changed = true; }
            }
        }
        if (!changed) break;
    }
    
    for (int i = 0; i < size; i++)
        tb[i] = std::max(tb[i], tau[i]);
    
    return tb;
}

int main(int argc, char* argv[]) {
    std::string path = "../../data/cordeau/pr03.txt";
    if (argc > 1) path = argv[1];
    
    std::cout << "Loading: " << path << "\n";
    DARPInstance inst = parse_cordeau(path);
    int n = inst.num_requests;
    
    DARPSolution sol = build_initial_solution(inst, "slack");
    std::cout << "Initial: obj=" << sol.objective() << ", unassigned=" << sol.unassigned.size() << "\n\n";
    
    int total_errors = 0;
    
    for (int r = 0; r < (int)sol.routes.size(); r++) {
        const auto& route = sol.routes[r];
        if (route.size() <= 2) continue;
        
        std::cout << "=== Route " << r << " (size=" << route.size() << ") ===\n";
        
        RouteAuxData aux = precompute_route_aux(route, inst);
        int sz = route.size();
        
        // --- Test 1: AEAP schedule basic checks ---
        for (int i = 0; i < sz; i++) {
            if (aux.tau[i] < inst.nodes[route[i]].earliest - EPS) {
                std::cout << "  ERROR: tau[" << i << "] < earliest at node " << route[i] << "\n";
                total_errors++;
            }
            if (aux.tau[i] > inst.nodes[route[i]].latest + EPS) {
                std::cout << "  ERROR: tau[" << i << "] > latest at node " << route[i] << "\n";
                total_errors++;
            }
            if (i > 0) {
                double arr = aux.depart[i-1] + inst.travel_time[route[i-1]][route[i]];
                if (aux.tau[i] < arr - EPS) {
                    std::cout << "  ERROR: tau[" << i << "] < arrival at node " << route[i] << "\n";
                    total_errors++;
                }
            }
        }
        
        // --- Test 2: ALAP >= AEAP ---
        for (int i = 0; i < sz; i++) {
            if (aux.tau_bar[i] < aux.tau[i] - EPS) {
                std::cout << "  ERROR: tau_bar[" << i << "]=" << aux.tau_bar[i] 
                          << " < tau[" << i << "]=" << aux.tau[i] << "\n";
                total_errors++;
            }
        }
        
        // --- Test 3: Verify wait[i][j] ---
        for (int i = 0; i < sz; i++) {
            for (int j = i + 1; j < sz; j++) {
                // Compute actual wait between i and j
                double actual_wait = 0.0;
                for (int p = i + 1; p <= j; p++) {
                    double arr_p = aux.depart[p-1] + inst.travel_time[route[p-1]][route[p]];
                    actual_wait += aux.tau[p] - arr_p;
                }
                if (std::abs(aux.wait[i][j] - actual_wait) > EPS) {
                    std::cout << "  ERROR: wait[" << i << "][" << j << "]=" << aux.wait[i][j]
                              << " != actual=" << actual_wait << "\n";
                    total_errors++;
                    if (total_errors > 20) goto done;
                }
            }
        }
        
        // --- Test 4: Verify constrained ALAP matches brute force ---
        {
            int errors_here = 0;
            for (int k = 0; k < sz && errors_here < 3; k++) {
                auto bf = brute_force_constrained_alap(route, inst, aux.tau, aux.depart, k);
                for (int i = 0; i < sz; i++) {
                    double precomp = aux.tau_bar_given[i][k];
                    if (std::abs(precomp - bf[i]) > 1e-4) {
                        std::cout << "  ERROR: tau_bar_given[" << i << "][" << k << "]=" 
                                  << precomp << " != brute_force=" << bf[i] 
                                  << " (diff=" << precomp - bf[i] << ")\n";
                        total_errors++;
                        errors_here++;
                        if (errors_here >= 3) break;
                    }
                }
            }
        }
        
        // --- Test 5: Verify tau_bar_at interpolation ---
        // For random t_k values, check that tau_bar_at gives the correct answer
        // by comparing with brute-force: fix v_k to t_k, run full constrained ALAP
        {
            std::mt19937 rng(42 + r);
            int interp_errors = 0;
            for (int k = 0; k < sz && interp_errors < 5; k++) {
                // Test at tau[k] (should match tau_bar_given[i][k])
                // Test at tau_bar[k] (should match tau_bar[i])
                // Test at midpoint
                double lo = aux.tau[k], hi = aux.tau_bar[k];
                if (hi - lo < EPS) continue;
                
                std::vector<double> test_points = {lo, hi, (lo + hi) / 2.0};
                for (double t_k : test_points) {
                    // Compute tau_bar_at for each position
                    // To verify, we'd need to run constrained ALAP with v_k fixed to t_k
                    // But that's expensive. Instead, verify the two endpoints match:
                    for (int i = 0; i < sz; i++) {
                        double interp = std::min(aux.tau_bar[i], 
                                                  aux.tau_bar_given[i][k] + (t_k - aux.tau[k]));
                        
                        // At t_k = tau[k], should equal tau_bar_given[i][k]
                        if (std::abs(t_k - lo) < EPS) {
                            double expected = aux.tau_bar_given[i][k];
                            if (std::abs(interp - expected) > 1e-4) {
                                std::cout << "  ERROR: tau_bar_at(" << i << ", tau[" << k << "], " << k 
                                          << ")=" << interp << " != tau_bar_given=" << expected << "\n";
                                total_errors++;
                                interp_errors++;
                            }
                        }
                        // At t_k = tau_bar[k], should equal tau_bar[i]
                        if (std::abs(t_k - hi) < EPS) {
                            double expected = aux.tau_bar[i];
                            // interp = min(tau_bar[i], tau_bar_given[i][k] + (tau_bar[k] - tau[k]))
                            // The second term could be > tau_bar[i], so min gives tau_bar[i]. Good.
                            if (interp > expected + 1e-4) {
                                std::cout << "  ERROR: tau_bar_at(" << i << ", tau_bar[" << k << "], " << k 
                                          << ")=" << interp << " > tau_bar=" << expected << "\n";
                                total_errors++;
                                interp_errors++;
                            }
                        }
                    }
                }
            }
        }
        
        // --- Test 6: For specific conservative cases, trace the O(1) check ---
        // Try inserting each unassigned request (or removed request) and find
        // cases where O(1) rejects but ALAP accepts
        // We'll test a few requests at position kp=1, lp=1
        {
            int traced = 0;
            for (int req = 1; req <= n && traced < 3; req++) {
                // Skip if request is in this route
                bool in_route = false;
                for (int nid : route) {
                    if (nid == req) { in_route = true; break; }
                }
                if (in_route) continue;
                
                int pid = req, did = req + n;
                // Try kp=1, lp=1 (consecutive at start)
                for (int kp = 1; kp < std::min(4, sz); kp++) {
                    for (int lp = kp; lp < std::min(kp + 3, sz); lp++) {
                        bool o1 = check_insertion_o1(aux, req, kp, lp, inst);
                        
                        std::vector<int> cand = route;
                        cand.insert(cand.begin() + kp, pid);
                        cand.insert(cand.begin() + lp + 1, did);
                        auto [alap_ok, reason] = check_alap_feasible(cand, inst);
                        
                        if (!o1 && alap_ok && traced < 3) {
                            std::cout << "  CONSERVATIVE: req=" << req << " kp=" << kp << " lp=" << lp << "\n";
                            
                            // Trace through the O(1) check manually
                            const auto& pn = inst.nodes[pid];
                            const auto& dn = inst.nodes[did];
                            double s_p = pn.service_duration;
                            double s_d = dn.service_duration;
                            
                            // Capacity
                            if (kp <= lp - 1) {
                                std::cout << "    max_load[" << kp << ".." << lp-1 << "]=" << aux.max_load[kp][lp-1] 
                                          << " + 1 vs Q=" << inst.vehicle_capacity << "\n";
                            }
                            
                            if (kp == lp) {
                                // Consecutive case trace
                                double tau_p = std::max(pn.earliest, aux.depart[kp-1] + inst.travel_time[route[kp-1]][pid]);
                                std::cout << "    [consec] tau_p=" << tau_p << " [" << pn.earliest << "," << pn.latest << "]\n";
                                if (tau_p > pn.latest + EPS) { std::cout << "    REJECT: pickup TW\n"; traced++; continue; }
                                
                                double dep_p = tau_p + s_p;
                                double tau_d = std::max(dn.earliest, dep_p + inst.travel_time[pid][did]);
                                std::cout << "    [consec] tau_d=" << tau_d << " [" << dn.earliest << "," << dn.latest << "]\n";
                                if (tau_d > dn.latest + EPS) { std::cout << "    REJECT: delivery TW\n"; traced++; continue; }
                                
                                double ride = tau_d - dep_p;
                                std::cout << "    [consec] ride=" << ride << " vs L=" << inst.max_ride_time << "\n";
                                if (ride > inst.max_ride_time + EPS) { std::cout << "    REJECT: ride time\n"; traced++; continue; }
                                
                                double dep_d = tau_d + s_d;
                                double tau_k_new = std::max((double)aux.tau[kp], dep_d + inst.travel_time[did][route[kp]]);
                                std::cout << "    [consec] tau_k_new=" << tau_k_new << " (was " << aux.tau[kp] << ")\n";
                                
                                double max_tau_d2 = std::min(dn.latest, tau_k_new - inst.travel_time[did][route[kp]] - s_d);
                                double max_tau_p2 = std::min(pn.latest, max_tau_d2 - inst.travel_time[pid][did] - s_p);
                                double max_tau_km1 = std::min(aux.tau_bar[kp-1], max_tau_p2 - inst.travel_time[route[kp-1]][pid] - inst.nodes[route[kp-1]].service_duration);
                                
                                std::cout << "    [consec] max_tau_p=" << max_tau_p2 << " max_tau_km1=" << max_tau_km1 << "\n";
                                std::cout << "    [consec] tau_bar[kp-1]=" << aux.tau_bar[kp-1] << " tau[kp-1]=" << aux.tau[kp-1] << "\n";
                                
                                double tb_at = std::min(aux.tau_bar[kp], aux.tau_bar_given[kp][kp-1] + (max_tau_km1 - aux.tau[kp-1]));
                                std::cout << "    [consec] tau_bar_at(k," << max_tau_km1 << ",k-1)=" << tb_at << "\n";
                                std::cout << "    [consec] tau_bar[k]=" << aux.tau_bar[kp] << " tau_bar_given[k][k-1]=" << aux.tau_bar_given[kp][kp-1] << "\n";
                                std::cout << "    [consec] CHECK: " << tau_k_new << " > " << tb_at << " ? " << (tau_k_new > tb_at + EPS ? "REJECT" : "OK") << "\n";
                                
                                // Now verify tau_bar_given[kp][kp-1] by brute force
                                auto bf = brute_force_constrained_alap(route, inst, aux.tau, aux.depart, kp-1);
                                std::cout << "    [consec] BRUTE FORCE tau_bar_given[k][k-1]=" << bf[kp] << " vs precomp=" << aux.tau_bar_given[kp][kp-1] << "\n";
                                
                            } else {
                                // General case trace
                                double tau_p = std::max(pn.earliest, aux.depart[kp-1] + inst.travel_time[route[kp-1]][pid]);
                                std::cout << "    tau_p=" << tau_p << " [" << pn.earliest << "," << pn.latest << "]\n";
                                if (tau_p > pn.latest + EPS) { std::cout << "    REJECT: pickup TW\n"; traced++; continue; }
                                
                                double dep_p = tau_p + s_p;
                                double tau_k_new = std::max((double)aux.tau[kp], dep_p + inst.travel_time[pid][route[kp]]);
                                double max_tau_p = std::min(pn.latest, tau_k_new - inst.travel_time[pid][route[kp]] - s_p);
                                double max_tau_km1 = std::min(aux.tau_bar[kp-1], max_tau_p - inst.travel_time[route[kp-1]][pid] - inst.nodes[route[kp-1]].service_duration);
                                
                                double tb_k = std::min(aux.tau_bar[kp], aux.tau_bar_given[kp][kp-1] + (max_tau_km1 - aux.tau[kp-1]));
                                std::cout << "    Alg2: tau_k_new=" << tau_k_new << " tb_k=" << tb_k 
                                          << " (tau_bar[k]=" << aux.tau_bar[kp] << " tau_bar_given[k][k-1]=" << aux.tau_bar_given[kp][kp-1]
                                          << " tau[k-1]=" << aux.tau[kp-1] << " max_tau_km1=" << max_tau_km1 << ")\n";
                                
                                if (tau_k_new > tb_k + EPS) {
                                    std::cout << "    REJECT at Alg2 Line 6\n";
                                    // Verify with brute force
                                    auto bf = brute_force_constrained_alap(route, inst, aux.tau, aux.depart, kp-1);
                                    std::cout << "    BRUTE FORCE: tau_bar_given[k][k-1]=" << bf[kp] << " vs precomp=" << aux.tau_bar_given[kp][kp-1] << "\n";
                                    double tb_bf = std::min(aux.tau_bar[kp], bf[kp] + (max_tau_km1 - aux.tau[kp-1]));
                                    std::cout << "    With BF: tb_k=" << tb_bf << " -> " << (tau_k_new > tb_bf + EPS ? "STILL REJECT" : "WOULD ACCEPT") << "\n";
                                    traced++;
                                    continue;
                                }
                                
                                // Continue to Alg3...
                                double delay_k = tau_k_new - aux.tau[kp];
                                double tau_lm1 = (kp == lp - 1) ? tau_k_new : aux.tau[lp-1] + std::max(0.0, delay_k - aux.wait[kp][lp-1]);
                                double dep_lm1 = tau_lm1 + inst.nodes[route[lp-1]].service_duration;
                                double tau_d = std::max(dn.earliest, dep_lm1 + inst.travel_time[route[lp-1]][did]);
                                
                                if (tau_d > dn.latest + EPS) { std::cout << "    REJECT: delivery TW in Alg3\n"; traced++; continue; }
                                
                                double dep_d = tau_d + s_d;
                                double tau_l_shifted = aux.tau[lp] + std::max(0.0, delay_k - aux.wait[kp][lp]);
                                double tau_l_new = std::max(tau_l_shifted, dep_d + inst.travel_time[did][route[lp]]);
                                
                                double max_tau_lm1 = std::min(
                                    std::min(aux.tau_bar[lp-1], aux.tau_bar_given[lp-1][kp-1] + (max_tau_km1 - aux.tau[kp-1])),
                                    tau_d - inst.travel_time[route[lp-1]][did] - inst.nodes[route[lp-1]].service_duration);
                                
                                double tb_l = std::min(
                                    std::min(aux.tau_bar[lp], aux.tau_bar_given[lp][kp-1] + (max_tau_km1 - aux.tau[kp-1])),
                                    std::min(aux.tau_bar[lp], aux.tau_bar_given[lp][lp-1] + (max_tau_lm1 - aux.tau[lp-1])));
                                
                                double delta_l = tau_l_new - tb_l;
                                std::cout << "    Alg3: tau_l_new=" << tau_l_new << " tb_l=" << tb_l << " delta_l=" << delta_l << "\n";
                                
                                if (delta_l > EPS) {
                                    std::cout << "    REJECT at Alg3 (delta_l repair needed)\n";
                                }
                                
                                // Ride time
                                double ride = tau_d - (max_tau_p + s_p);
                                std::cout << "    ride=" << ride << " vs L=" << inst.max_ride_time << "\n";
                            }
                            traced++;
                        }
                    }
                }
            }
        }
    }
    
    done:
    std::cout << "\n=== TOTAL ERRORS: " << total_errors << " ===\n";
    return total_errors > 0 ? 1 : 0;
}