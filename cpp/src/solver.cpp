#include "solver.h"
#include "initial_solution.h"
#include "operators.h"
#include "feasibility.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Post-ALNS repair: remove requests with ride-time violations, reinsert
// ---------------------------------------------------------------------------
static DARPSolution repair_ride_times(const DARPSolution& sol, bool verbose) {
    const auto& inst = *sol.instance;
    int n = inst.num_requests;
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;

    // Find requests with ride-time or route-duration violations
    std::set<int> to_remove;
    for (const auto& route : sol.routes) {
        auto sched = sol.compute_route_schedule(route);
        for (const auto& [req, rt] : sched.ride_time) {
            if (rt > inst.max_ride_time + EPS) {
                to_remove.insert(req);
            }
        }
        double route_dur = sched.begin_service.back() - sched.departure_time[0];
        if (route_dur > inst.max_route_duration + EPS) {
            for (int nid : route) {
                if (nid >= 1 && nid <= n) {
                    to_remove.insert(nid);
                }
            }
        }
    }

    if (to_remove.empty()) return sol.copy();

    if (verbose) {
        std::cout << "  Repairing " << to_remove.size()
                  << " request(s) with ride-time violations...\n";
    }

    DARPSolution repaired = sol.copy();

    // Remove violating requests
    for (int req : to_remove) {
        int pickup_id = req;
        int delivery_id = req + n;
        for (auto& route : repaired.routes) {
            auto it_p = std::find(route.begin(), route.end(), pickup_id);
            if (it_p != route.end()) {
                route.erase(it_p);
                auto it_d = std::find(route.begin(), route.end(), delivery_id);
                if (it_d != route.end()) {
                    route.erase(it_d);
                }
                repaired.unassigned.insert(req);
                break;
            }
        }
    }

    // Reinsert using full optimizer feasibility check
    std::vector<int> to_reinsert;
    for (int req : to_remove) {
        if (repaired.unassigned.count(req)) {
            to_reinsert.push_back(req);
        }
    }
    std::sort(to_reinsert.begin(), to_reinsert.end());

    for (int req : to_reinsert) {
        int pickup_id = req;
        int delivery_id = req + n;
        double best_cost = std::numeric_limits<double>::infinity();
        int best_k = -1;
        int best_p = -1;
        int best_d = -1;

        for (int k = 0; k < static_cast<int>(repaired.routes.size()); k++) {
            const auto& route = repaired.routes[k];
            int rlen = static_cast<int>(route.size());

            for (int p_pos = 1; p_pos < rlen; p_pos++) {
                for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                    std::vector<int> candidate = route;
                    candidate.insert(candidate.begin() + p_pos, pickup_id);
                    candidate.insert(candidate.begin() + d_pos, delivery_id);

                    // Compute cost delta
                    double new_cost = 0.0;
                    for (int i = 0; i < static_cast<int>(candidate.size()) - 1; i++) {
                        new_cost += tt[candidate[i]][candidate[i + 1]];
                    }
                    double old_cost = 0.0;
                    for (int i = 0; i < rlen - 1; i++) {
                        old_cost += tt[route[i]][route[i + 1]];
                    }
                    double cost = new_cost - old_cost;

                    if (cost >= best_cost) continue;

                    // Full optimizer check
                    auto sched = repaired.compute_route_schedule(candidate);
                    bool ok = true;
                    for (int j = 0; j < static_cast<int>(candidate.size()); j++) {
                        if (sched.begin_service[j] > nodes[candidate[j]].latest + EPS) {
                            ok = false;
                            break;
                        }
                        if (sched.load[j] < 0 || sched.load[j] > inst.vehicle_capacity) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        for (const auto& [r, rt] : sched.ride_time) {
                            if (rt > inst.max_ride_time + EPS) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (ok) {
                        double rd = sched.begin_service.back() - sched.departure_time[0];
                        if (rd > inst.max_route_duration + EPS) {
                            ok = false;
                        }
                    }
                    if (ok) {
                        best_cost = cost;
                        best_k = k;
                        best_p = p_pos;
                        best_d = d_pos;
                    }
                }
            }
        }

        if (best_k >= 0) {
            repaired.routes[best_k].insert(
                repaired.routes[best_k].begin() + best_p, pickup_id);
            repaired.routes[best_k].insert(
                repaired.routes[best_k].begin() + best_d, delivery_id);
            repaired.unassigned.erase(req);
        }
    }

    if (verbose) {
        std::set<int> still;
        for (int req : to_remove) {
            if (repaired.unassigned.count(req)) {
                still.insert(req);
            }
        }
        if (!still.empty()) {
            std::cout << "  Could not reinsert: {";
            bool first = true;
            for (int r : still) {
                if (!first) std::cout << ", ";
                std::cout << r;
                first = false;
            }
            std::cout << "}\n";
        }
    }

    return repaired;
}

// ---------------------------------------------------------------------------
// ALNS Solver
// ---------------------------------------------------------------------------
SolverResult solve(
    const DARPInstance& instance,
    int max_iterations,
    double start_temperature,
    double end_temperature,
    int seed,
    bool verbose,
    const std::string& feasibility_mode) {

    using Clock = std::chrono::high_resolution_clock;

    std::mt19937 rng(seed);

    // Build initial solution
    auto t0 = Clock::now();
    DARPSolution init_sol = build_initial_solution(instance);
    auto t1 = Clock::now();
    double construction_time = std::chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        std::cout << "Initial solution: obj=" << init_sol.objective()
                  << ", unassigned=" << init_sol.unassigned.size()
                  << " (" << construction_time << "s)\n";
    }

    // Switch feasibility mode for ALNS (initial always uses slack)
    if (feasibility_mode != "slack") {
        init_sol.feasibility_mode = feasibility_mode;
    }

    // SA parameters
    double sa_step = std::pow(end_temperature / start_temperature,
                              1.0 / max_iterations);
    double temperature = start_temperature;

    // Track best
    DARPSolution best = init_sol.copy();
    double best_obj = best.objective();
    DARPSolution current = init_sol.copy();
    double current_obj = current.objective();

    auto solve_t0 = Clock::now();

    for (int iter = 0; iter < max_iterations; iter++) {
        // Destroy
        DARPSolution destroyed = random_removal(current, rng);

        // Repair
        DARPSolution candidate = greedy_insertion(destroyed, rng);
        double candidate_obj = candidate.objective();

        // SA acceptance
        double delta = candidate_obj - current_obj;
        bool accept = false;
        if (delta < -EPS) {
            accept = true;
        } else {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < std::exp(-delta / temperature)) {
                accept = true;
            }
        }

        if (accept) {
            current = std::move(candidate);
            current_obj = current.objective();
        }

        if (current_obj < best_obj - EPS) {
            best = current.copy();
            best_obj = current_obj;
        }

        temperature *= sa_step;

        if (verbose && (iter + 1) % 10 == 0) {
            std::cout << "  iter " << (iter + 1)
                      << ": current=" << current_obj
                      << ", best=" << best_obj
                      << ", temp=" << temperature << "\n";
        }
    }

    auto solve_t1 = Clock::now();
    double solve_time = std::chrono::duration<double>(solve_t1 - solve_t0).count();

    if (verbose) {
        std::cout << "Best solution:    obj=" << best.objective()
                  << " (" << solve_time << "s)\n";
    }

    // Post-ALNS feasibility repair (skip if mode=full, already exact)
    DARPSolution repaired = (feasibility_mode == "full")
        ? best.copy()
        : repair_ride_times(best, verbose);

    return SolverResult{
        std::move(repaired),
        std::move(init_sol),
        construction_time,
        solve_time
    };
}
