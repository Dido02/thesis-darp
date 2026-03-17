#include "instance.h"
#include "feasibility.h"
#include "parsers/cordeau.h"
#include "solution.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Test 1: AEAP consistency — verify compute_aeap produces valid Steps 1+2 schedule
// (tau[i] >= earliest, tau[i] >= arrival, loads correct, depot delayed by slack)
// ---------------------------------------------------------------------------
static bool test_aeap_consistency(const DARPInstance& inst,
                                  const std::vector<std::vector<int>>& routes) {
    std::cout << "\n--- Test 1: AEAP consistency (Steps 1+2) ---\n";
    bool ok = true;

    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        const auto& route = routes[k];
        int sz = static_cast<int>(route.size());
        if (sz <= 2) continue;

        std::vector<double> tau, depart;
        std::vector<int> load;
        compute_aeap(route, inst, tau, depart, load);

        // Verify forward consistency
        for (int i = 0; i < sz; i++) {
            // Time window lower bound
            if (tau[i] < inst.nodes[route[i]].earliest - EPS) {
                std::cerr << "  FAIL Route " << k << " pos " << i
                          << ": tau=" << tau[i] << " < earliest="
                          << inst.nodes[route[i]].earliest << "\n";
                ok = false;
            }
            // Time window upper bound
            if (tau[i] > inst.nodes[route[i]].latest + EPS) {
                std::cerr << "  FAIL Route " << k << " pos " << i
                          << ": tau=" << tau[i] << " > latest="
                          << inst.nodes[route[i]].latest << "\n";
                ok = false;
            }
            // Departure
            double exp_dep = tau[i] + inst.nodes[route[i]].service_duration;
            if (std::abs(depart[i] - exp_dep) > EPS) {
                std::cerr << "  FAIL Route " << k << " pos " << i
                          << ": depart=" << depart[i] << " expected=" << exp_dep << "\n";
                ok = false;
            }
            // Arrival consistency (for i > 0)
            if (i > 0) {
                double arr = depart[i - 1] + inst.travel_time[route[i - 1]][route[i]];
                if (tau[i] < arr - EPS) {
                    std::cerr << "  FAIL Route " << k << " pos " << i
                              << ": tau=" << tau[i] << " < arrival=" << arr << "\n";
                    ok = false;
                }
            }
        }
        // Load consistency
        if (load[0] != 0) {
            std::cerr << "  FAIL Route " << k << ": load[0]=" << load[0] << "\n";
            ok = false;
        }
        for (int i = 1; i < sz; i++) {
            if (load[i] != load[i - 1] + inst.nodes[route[i]].demand) {
                std::cerr << "  FAIL Route " << k << " pos " << i << ": load mismatch\n";
                ok = false;
            }
        }
    }

    std::cout << (ok ? "  PASSED" : "  FAILED") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: ALAP invariants — tau[i] <= tau_bar[i] <= latest[route[i]]
// ---------------------------------------------------------------------------
static bool test_alap_invariants(const DARPInstance& inst,
                                  const std::vector<std::vector<int>>& routes) {
    std::cout << "\n--- Test 2: ALAP invariants ---\n";
    bool ok = true;

    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        const auto& route = routes[k];
        if (static_cast<int>(route.size()) <= 2) continue;

        std::vector<double> tau, depart;
        std::vector<int> load;
        compute_aeap(route, inst, tau, depart, load);

        std::vector<double> tau_bar;
        compute_alap(route, inst, tau, depart, tau_bar);

        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (tau[i] > tau_bar[i] + EPS) {
                std::cerr << "  FAIL Route " << k << " pos " << i
                          << ": tau=" << tau[i] << " > tau_bar=" << tau_bar[i] << "\n";
                ok = false;
            }
            if (tau_bar[i] > inst.nodes[route[i]].latest + EPS) {
                std::cerr << "  FAIL Route " << k << " pos " << i
                          << ": tau_bar=" << tau_bar[i]
                          << " > latest=" << inst.nodes[route[i]].latest << "\n";
                ok = false;
            }
        }
    }

    std::cout << (ok ? "  PASSED" : "  FAILED") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: Exhaustive correctness — compare O(1) check vs check_alap_feasible
// (ALAP-based ground truth using the same Steps 1+2 + ALAP framework)
// ---------------------------------------------------------------------------
static bool test_exhaustive_correctness(const DARPInstance& inst,
                                         const std::vector<std::vector<int>>& routes) {
    std::cout << "\n--- Test 3: Exhaustive correctness ---\n";
    bool ok = true;
    int total_checks = 0;
    int agreements = 0;
    int false_positives = 0;  // O(1) says feasible, full says infeasible
    int false_negatives = 0;  // O(1) says infeasible, full says feasible
    int n = inst.num_requests;
    int routes_tested = 0;

    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        const auto& route = routes[k];
        if (static_cast<int>(route.size()) < 6) continue;
        if (routes_tested >= 5) break;
        routes_tested++;

        std::cout << "  Testing route " << k << " (size=" << route.size() << ")...\n";

        // Collect requests in this route
        std::vector<int> requests_in_route;
        for (int nid : route) {
            if (nid >= 1 && nid <= n) {
                requests_in_route.push_back(nid);
            }
        }

        // Try removing each request and reinserting
        for (int req : requests_in_route) {
            int pickup_id = req;
            int delivery_id = req + n;

            // Build reduced route (remove pickup and delivery)
            std::vector<int> reduced;
            for (int nid : route) {
                if (nid != pickup_id && nid != delivery_id) {
                    reduced.push_back(nid);
                }
            }

            int rlen = static_cast<int>(reduced.size());
            if (rlen < 3) continue;

            // Precompute aux data
            RouteAuxData aux = precompute_route_aux(reduced, inst);

            // Try all (p_pos, d_pos) pairs
            for (int p_pos = 1; p_pos < rlen; p_pos++) {
                for (int d_pos = p_pos; d_pos < rlen; d_pos++) {
                    // Build candidate route

                    auto [reduced_ok, reduced_reason] = check_alap_feasible(reduced, inst);
                    if (!reduced_ok) continue;
                    std::vector<int> candidate = reduced;
                    candidate.insert(candidate.begin() + p_pos, pickup_id);
                    candidate.insert(candidate.begin() + d_pos + 1, delivery_id);

                    // Ground truth: check_alap_feasible
                    auto [full_ok, full_reason] = check_alap_feasible(candidate, inst);

                    // O(1) check
                    bool o1_ok = check_insertion_o1(aux, req, p_pos, d_pos, inst);

                    total_checks++;
                    if (o1_ok == full_ok) {
                        agreements++;
                    } else if (o1_ok && !full_ok) {
                        false_positives++;
                        std::cerr << "  FALSE POSITIVE: route " << k
                                  << ", req " << req
                                  << ", k=" << p_pos << ", l=" << d_pos
                                  << "\n    full_reason: " << full_reason
                                  << "\n    reduced route:";
                        for (int nid : reduced) std::cerr << " " << nid;
                        std::cerr << "\n    candidate:";
                        for (int nid : candidate) std::cerr << " " << nid;
                        std::cerr << "\n";
                        ok = false;
                    } else {
                        false_negatives++;
                        if (false_negatives <= 5) {
                            std::cerr << "  FALSE NEGATIVE: route " << k
                                      << ", req " << req
                                      << ", k=" << p_pos << ", l=" << d_pos
                                      << "\n    reduced route:";
                            for (int nid : reduced) std::cerr << " " << nid;
                            std::cerr << "\n    candidate:";
                            for (int nid : candidate) std::cerr << " " << nid;
                            std::cerr << "\n";
                        }
                    }
                }
            }
        }
    }

    std::cout << "  Total checks: " << total_checks << "\n";
    std::cout << "  Agreements: " << agreements << "\n";
    std::cout << "  False positives: " << false_positives << "\n";
    std::cout << "  False negatives: " << false_negatives << "\n";
    double agreement_rate = total_checks > 0
        ? 100.0 * agreements / total_checks : 0.0;
    std::cout << "  Agreement rate: " << agreement_rate << "%\n";

    if (false_positives > 0) {
        std::cerr << "  FAILED: " << false_positives << " false positives!\n";
        ok = false;
    }

    if (false_negatives > 0) {
        double fn_rate = 100.0 * false_negatives / total_checks;
        std::cerr << "  FAILED: " << false_negatives << " false negatives ("
                  << fn_rate << "%)\n";
        ok = false;
    }

    std::cout << (ok ? "  PASSED (exact match)" : "  FAILED") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Test 4: Performance benchmark
// ---------------------------------------------------------------------------
static void test_benchmark(const DARPInstance& inst,
                            const std::vector<std::vector<int>>& routes) {
    std::cout << "\n--- Test 4: Performance benchmark ---\n";
    int n = inst.num_requests;

    // Find a route with >= 6 nodes
    int route_idx = -1;
    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        if (static_cast<int>(routes[k].size()) >= 8) {
            route_idx = k;
            break;
        }
    }
    if (route_idx < 0) {
        std::cout << "  No suitable route found for benchmark\n";
        return;
    }

    const auto& route = routes[route_idx];
    std::cout << "  Route " << route_idx << " (size=" << route.size() << ")\n";

    // Pick first request in route to remove
    int req = -1;
    for (int nid : route) {
        if (nid >= 1 && nid <= n) { req = nid; break; }
    }
    if (req < 0) return;

    int pickup_id = req;
    int delivery_id = req + n;
    std::vector<int> reduced;
    for (int nid : route) {
        if (nid != pickup_id && nid != delivery_id) {
            reduced.push_back(nid);
        }
    }
    int rlen = static_cast<int>(reduced.size());

    // Benchmark precomputation
    const int PRECOMP_ITERS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    RouteAuxData aux;
    for (int i = 0; i < PRECOMP_ITERS; i++) {
        aux = precompute_route_aux(reduced, inst);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double precomp_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / PRECOMP_ITERS;

    // Count total (k,l) pairs
    int num_pairs = 0;
    for (int p = 1; p < rlen; p++) {
        for (int d = p; d < rlen; d++) {
            num_pairs++;
        }
    }

    // Benchmark O(1) checks
    const int CHECK_ITERS = 10000;
    t0 = std::chrono::high_resolution_clock::now();
    int feasible_count = 0;
    for (int iter = 0; iter < CHECK_ITERS; iter++) {
        for (int p = 1; p < rlen; p++) {
            for (int d = p; d < rlen; d++) {
                if (check_insertion_o1(aux, req, p, d, inst))
                    feasible_count++;
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double o1_total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double o1_per_pair = o1_total_us / (CHECK_ITERS * num_pairs);

    // Benchmark check_slack_corrected (what greedy insertion currently uses)
    const int SLACK_ITERS = 1000;
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < SLACK_ITERS; iter++) {
        for (int p = 1; p < rlen; p++) {
            for (int d = p; d < rlen; d++) {
                std::vector<int> candidate = reduced;
                candidate.insert(candidate.begin() + p, pickup_id);
                candidate.insert(candidate.begin() + d + 1, delivery_id);
                check_slack_corrected(candidate, inst);
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double slack_total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double slack_per_pair = slack_total_us / (SLACK_ITERS * num_pairs);

    std::cout << "  Precomputation: " << precomp_us << " us\n";
    std::cout << "  O(1) per pair: " << o1_per_pair << " us\n";
    std::cout << "  Slack-corrected per pair: " << slack_per_pair << " us\n";
    std::cout << "  Speedup per pair (vs slack): " << slack_per_pair / o1_per_pair << "x\n";
    std::cout << "  Pairs per route: " << num_pairs << "\n";
    std::cout << "  O(1) total (precomp + all pairs): "
              << precomp_us + num_pairs * o1_per_pair << " us\n";
    std::cout << "  Slack total (all pairs): "
              << num_pairs * slack_per_pair << " us\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== Test O(1) Feasibility ===\n";

    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");

    // Use the same test routes as test_feasibility.cpp
    std::vector<std::vector<int>> test_routes = {
        {0, 23, 2, 95, 1, 74, 73, 52, 26, 124, 28, 19, 100, 98, 91, 17, 22, 25, 24, 94, 89, 97, 96, 3, 75, 145},
        {0, 5, 4, 18, 77, 9, 20, 90, 76, 92, 71, 81, 143, 62, 44, 134, 116, 7, 8, 80, 53, 125, 79, 145},
        {0, 33, 11, 14, 6, 78, 105, 83, 21, 86, 93, 12, 10, 13, 82, 85, 84, 27, 99, 38, 15, 16, 87, 88, 110, 145},
    };

    bool ok = true;
    ok = test_aeap_consistency(inst, test_routes) && ok;
    ok = test_alap_invariants(inst, test_routes) && ok;
    ok = test_exhaustive_correctness(inst, test_routes) && ok;
    test_benchmark(inst, test_routes);

    if (ok) {
        std::cout << "\nAll O(1) feasibility tests PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSome O(1) feasibility tests FAILED\n";
        return 1;
    }
}
