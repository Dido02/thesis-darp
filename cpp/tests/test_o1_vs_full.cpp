#include "instance.h"
#include "parsers/cordeau.h"
#include "solution.h"
#include "initial_solution.h"
#include "feasibility/o1_feasibility.h"
#include "feasibility.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <set>

static constexpr double EPS = 1e-6;

int main(int argc, char* argv[]) {
    std::string instance_path = "../../data/cordeau/pr03.txt";
    if (argc > 1) instance_path = argv[1];

    std::cout << "Loading: " << instance_path << "\n";
    DARPInstance inst = parse_cordeau(instance_path);
    int n = inst.num_requests;
    const auto& tt = inst.travel_time;

    // Build initial solution with slack (good starting point)
    DARPSolution sol = build_initial_solution(inst, "slack");
    sol.feasibility_mode = "o1";
    std::cout << "Initial: obj=" << sol.objective()
              << ", unassigned=" << sol.unassigned.size() << "\n\n";

    // Simulate several destroy-repair cycles
    std::mt19937 rng(42);
    int total_checks = 0;
    int o1_accept_alap_accept = 0;
    int o1_accept_alap_reject = 0;  // O(1) too permissive
    int o1_reject_alap_accept = 0;  // O(1) too conservative
    int o1_reject_alap_reject = 0;

    for (int cycle = 0; cycle < 20; cycle++) {
        // Random removal of 20% requests
        std::vector<int> assigned;
        for (int i = 1; i <= n; i++) {
            if (!sol.unassigned.count(i)) assigned.push_back(i);
        }
        int q = std::max(1, static_cast<int>(assigned.size() * 0.2));
        std::shuffle(assigned.begin(), assigned.end(), rng);

        DARPSolution destroyed = sol.copy();
        for (int i = 0; i < q; i++) {
            int req = assigned[i];
            int pid = req, did = req + n;
            for (auto& route : destroyed.routes) {
                auto it = std::find(route.begin(), route.end(), pid);
                if (it != route.end()) {
                    route.erase(it);
                    auto it2 = std::find(route.begin(), route.end(), did);
                    if (it2 != route.end()) route.erase(it2);
                    destroyed.unassigned.insert(req);
                    break;
                }
            }
        }

        // Precompute aux for all routes
        std::vector<RouteAuxData> aux(destroyed.routes.size());
        for (int k = 0; k < static_cast<int>(destroyed.routes.size()); k++) {
            aux[k] = precompute_route_aux(destroyed.routes[k], inst);
        }

        // Try inserting each unassigned request at every position
        int cycle_checks = 0;
        int cycle_o1_permissive = 0;
        int cycle_o1_conservative = 0;

        for (int req : destroyed.unassigned) {
            int pid = req, did = req + n;

            for (int k = 0; k < static_cast<int>(destroyed.routes.size()); k++) {
                const auto& route = destroyed.routes[k];
                int rlen = static_cast<int>(route.size());

                for (int kp = 1; kp < rlen; kp++) {
                    for (int lp = kp; lp < rlen; lp++) {
                        // O(1) check
                        bool o1_ok = check_insertion_o1(aux[k], req, kp, lp, inst);

                        // Ground truth: build candidate, check with ALAP
                        std::vector<int> candidate = route;
                        candidate.insert(candidate.begin() + kp, pid);
                        candidate.insert(candidate.begin() + lp + 1, did);
                        auto [alap_ok, reason] = check_alap_feasible(candidate, inst);

                        total_checks++;
                        cycle_checks++;

                        if (o1_ok && alap_ok) {
                            o1_accept_alap_accept++;
                        } else if (o1_ok && !alap_ok) {
                            o1_accept_alap_reject++;
                            cycle_o1_permissive++;
                            if (cycle_o1_permissive <= 3) {
                                std::cout << "  TOO PERMISSIVE: req=" << req
                                          << " route=" << k << " kp=" << kp
                                          << " lp=" << lp << " reason=" << reason << "\n";
                            }
                        } else if (!o1_ok && alap_ok) {
                            o1_reject_alap_accept++;
                            cycle_o1_conservative++;
                            if (cycle_o1_conservative <= 3) {
                                std::cout << "  TOO CONSERVATIVE: req=" << req
                                          << " route=" << k << " kp=" << kp
                                          << " lp=" << lp << "\n";
                            }
                        } else {
                            o1_reject_alap_reject++;
                        }
                    }
                }
            }
        }

        std::cout << "Cycle " << cycle << ": " << cycle_checks << " checks, "
                  << cycle_o1_permissive << " permissive, "
                  << cycle_o1_conservative << " conservative\n";

        // Repair using O(1) so next cycle has a different solution
        for (int req : std::set<int>(destroyed.unassigned)) {
            int pid = req, did = req + n;
            double best_cost = 1e18;
            int best_k = -1, best_kp = -1, best_lp = -1;

            for (int k = 0; k < static_cast<int>(destroyed.routes.size()); k++) {
                const auto& route = destroyed.routes[k];
                int rlen = static_cast<int>(route.size());
                for (int kp = 1; kp < rlen; kp++) {
                    for (int lp = kp; lp < rlen; lp++) {
                        if (!check_insertion_o1(aux[k], req, kp, lp, inst)) continue;
                        double delta;
                        if (kp == lp) {
                            delta = tt[route[kp-1]][pid] + tt[pid][did] + tt[did][route[kp]]
                                  - tt[route[kp-1]][route[kp]];
                        } else {
                            delta = tt[route[kp-1]][pid] + tt[pid][route[kp]] - tt[route[kp-1]][route[kp]]
                                  + tt[route[lp-1]][did] + tt[did][route[lp]] - tt[route[lp-1]][route[lp]];
                        }
                        if (delta < best_cost) {
                            best_cost = delta; best_k = k; best_kp = kp; best_lp = lp;
                        }
                    }
                }
            }
            if (best_k >= 0) {
                destroyed.routes[best_k].insert(destroyed.routes[best_k].begin() + best_kp, pid);
                destroyed.routes[best_k].insert(destroyed.routes[best_k].begin() + best_lp + 1, did);
                destroyed.unassigned.erase(req);
                aux[best_k] = precompute_route_aux(destroyed.routes[best_k], inst);
            }
        }
        sol = destroyed;
    }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total insertion checks:      " << total_checks << "\n";
    std::cout << "Both accept (agree):         " << o1_accept_alap_accept << "\n";
    std::cout << "Both reject (agree):         " << o1_reject_alap_reject << "\n";
    std::cout << "O(1) accepts, ALAP rejects:  " << o1_accept_alap_reject << " (too permissive)\n";
    std::cout << "O(1) rejects, ALAP accepts:  " << o1_reject_alap_accept << " (too conservative)\n";

    double total_feasible = o1_accept_alap_accept + o1_reject_alap_accept;
    if (total_feasible > 0) {
        double miss_rate = 100.0 * o1_reject_alap_accept / total_feasible;
        std::cout << "\nFalse rejection rate: " << miss_rate << "% of feasible insertions missed\n";
    }
    double total_o1_accept = o1_accept_alap_accept + o1_accept_alap_reject;
    if (total_o1_accept > 0) {
        double false_accept = 100.0 * o1_accept_alap_reject / total_o1_accept;
        std::cout << "False acceptance rate: " << false_accept << "% of O(1) accepts are infeasible\n";
    }

    return 0;
}
