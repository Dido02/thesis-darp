#include "solver.h"
#include "initial_solution.h"
#include "operators.h"
#include "feasibility.h"
#include "logging/alns_logger.h"
#include "ml/alns_state.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

static constexpr double EPS = 1e-6;

// ---------------------------------------------------------------------------
// Post-ALNS repair: remove requests with ride-time violations, reinsert
// ---------------------------------------------------------------------------
static DARPSolution repair_ride_times(const DARPSolution& sol, bool verbose) {
    const auto& inst = *sol.instance;
    int n = inst.num_requests;
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;

    std::set<int> to_remove;
    for (const auto& route : sol.routes) {
        auto sched = sol.compute_route_schedule(route);
        for (const auto& [req, rt] : sched.ride_time) {
            if (rt > inst.max_ride_time_for(req) + EPS) {
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
        int best_k = -1, best_p = -1, best_d = -1;

        for (int k = 0; k < static_cast<int>(repaired.routes.size()); k++) {
            const auto& route = repaired.routes[k];
            int rlen = static_cast<int>(route.size());

            for (int p_pos = inst.route_start_offset(); p_pos < rlen; p_pos++) {
                for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
                    std::vector<int> candidate = route;
                    candidate.insert(candidate.begin() + p_pos, pickup_id);
                    candidate.insert(candidate.begin() + d_pos, delivery_id);

                    double new_cost = 0.0;
                    for (int i = 0; i < static_cast<int>(candidate.size()) - 1; i++)
                        new_cost += tt[candidate[i]][candidate[i + 1]];
                    double old_cost = 0.0;
                    for (int i = 0; i < rlen - 1; i++)
                        old_cost += tt[route[i]][route[i + 1]];
                    double cost = new_cost - old_cost;

                    if (cost >= best_cost) continue;

                    auto sched = repaired.compute_route_schedule(candidate);
                    bool ok = true;
                    for (int j = 0; j < static_cast<int>(candidate.size()); j++) {
                        if (sched.begin_service[j] > nodes[candidate[j]].latest + EPS) { ok = false; break; }
                        if (sched.load[j] < 0 || sched.load[j] > inst.vehicle_capacity) { ok = false; break; }
                    }
                    if (ok) {
                        for (const auto& [r, rt] : sched.ride_time) {
                            if (rt > inst.max_ride_time_for(r) + EPS) { ok = false; break; }
                        }
                    }
                    if (ok) {
                        double rd = sched.begin_service.back() - sched.departure_time[0];
                        if (rd > inst.max_route_duration + EPS) ok = false;
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

    return repaired;
}

// ---------------------------------------------------------------------------
// Helper: parse comma-separated operator names
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_operator_list(const std::string& s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) result.push_back(token);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Build operator registry from config strings
// ---------------------------------------------------------------------------
static OperatorRegistry build_registry(
    const std::string& destroy_config,
    const std::string& repair_config,
    const std::string& selection_config) {

    OperatorRegistry registry;

    auto destroy_names = parse_operator_list(destroy_config);
    for (const auto& name : destroy_names) {
        if (name == "random") {
            registry.add_destroy("random", random_removal);
        } else if (name == "worst") {
            registry.add_destroy("worst", worst_removal);
        } else if (name == "shaw") {
            registry.add_destroy("shaw", shaw_removal);
        } else if (name == "rtcritical") {
            registry.add_destroy("rtcritical", ride_time_critical_removal);
        } else if (name == "lowslack") {
            registry.add_destroy("lowslack", low_slack_route_removal);
        } else if (name == "entireroute") {
            registry.add_destroy("entireroute", entire_route_removal);
        } else {
            std::cerr << "Unknown destroy operator: " << name << "\n";
        }
    }

    auto repair_names = parse_operator_list(repair_config);
    for (const auto& name : repair_names) {
        if (name == "greedy") {
            registry.add_repair("greedy", greedy_insertion);
        } else if (name == "regret2") {
            registry.add_repair("regret2", regret2_insertion);
        } else if (name == "regret3") {
            registry.add_repair("regret3", regret3_insertion);
        } else if (name == "slack") {
            registry.add_repair("slack", slack_insertion);
        } else if (name == "regret4") {
            registry.add_repair("regret4", regret4_insertion);
        } else if (name == "regretM") {
            registry.add_repair("regretM", regretM_insertion);
        } else {
            std::cerr << "Unknown repair operator: " << name << "\n";
        }
    }

    // Fallback defaults
    if (registry.num_destroy() == 0) {
        registry.add_destroy("random", random_removal);
    }
    if (registry.num_repair() == 0) {
        registry.add_repair("greedy", greedy_insertion);
    }

    if (selection_config == "random") {
        registry.set_selection_method(SelectionMethod::RANDOM);
    } else if (selection_config == "roulette") {
        registry.set_selection_method(SelectionMethod::ROULETTE_WHEEL);
    } else if (selection_config == "ml") {
        registry.set_selection_method(SelectionMethod::ML_GUIDED);
    } else {
        registry.set_selection_method(SelectionMethod::RANDOM);
    }

    return registry;
}

// ---------------------------------------------------------------------------
// Attach MLSelector after build_registry when --selection ml is requested.
// Scoped here because MLSelector construction needs the registry's current
// operator names and indices. No-op for other selection methods.
// ---------------------------------------------------------------------------
#ifdef DARP_ENABLE_ML
#include "ml/ml_selector.h"
#endif

static void attach_ml_selector_if_needed(
    OperatorRegistry& registry,
    const std::string& selection_config,
    const std::string& ml_model_dir,
    bool verbose,
    const std::string& log_ml_probs_path = "") {

    if (selection_config != "ml") return;

#ifndef DARP_ENABLE_ML
    (void)registry; (void)ml_model_dir; (void)verbose;
    (void)log_ml_probs_path;
    throw std::runtime_error(
        "--selection ml: this binary was built without ML support. "
        "Rebuild with -DENABLE_ML_SELECTION=ON.");
#else
    if (ml_model_dir.empty()) {
        throw std::runtime_error(
            "--selection ml requires --ml-model-dir <path> pointing to a directory "
            "with feature_manifest.json and {d}_{r}.onnx files.");
    }

    // Active operators + name→index maps (registry is source of truth).
    std::vector<std::string> active_destroy, active_repair;
    std::map<std::string, int> d_to_idx, r_to_idx;
    for (int i = 0; i < registry.num_destroy(); ++i) {
        active_destroy.push_back(registry.destroy_name(i));
        d_to_idx[registry.destroy_name(i)] = i;
    }
    for (int i = 0; i < registry.num_repair(); ++i) {
        active_repair.push_back(registry.repair_name(i));
        r_to_idx[registry.repair_name(i)] = i;
    }

    auto ml = std::make_unique<MLSelector>(
        ml_model_dir, active_destroy, active_repair, d_to_idx, r_to_idx,
        log_ml_probs_path);

    if (verbose) {
        std::cout << "ML selector loaded: model_version=" << ml->model_version()
                  << ", " << ml->n_active_pairs() << " active pairs, "
                  << ml->n_features() << " features\n";
    }
    registry.set_ml_selector(std::move(ml));
#endif
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
    const std::string& feasibility_mode,
    const std::string& destroy_config,
    const std::string& repair_config,
    const std::string& selection_config,
    double time_limit,
    double unassigned_penalty,
    const std::string& init_sort,
    ALNSLogger* logger,
    const std::string& ml_model_dir,
    bool exclude_selection_from_budget,
    const std::string& log_ml_probs_path) {

    using Clock = std::chrono::high_resolution_clock;

    std::mt19937 rng(seed);

    // Build operator registry
    OperatorRegistry registry = build_registry(destroy_config, repair_config, selection_config);
    attach_ml_selector_if_needed(registry, selection_config, ml_model_dir, verbose,
                                 log_ml_probs_path);

    if (verbose) {
        std::cout << "Operators: destroy=[";
        for (int i = 0; i < registry.num_destroy(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << registry.destroy_name(i);
        }
        std::cout << "] repair=[";
        for (int i = 0; i < registry.num_repair(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << registry.repair_name(i);
        }
        std::cout << "] selection=" << selection_config << "\n";
    }

    // Build initial solution
    auto t0 = Clock::now();
    DARPSolution init_sol = build_initial_solution(instance, "o1", init_sort, seed);
    init_sol.feasibility_mode = feasibility_mode;
    init_sol.unassigned_penalty = unassigned_penalty;
    auto t1 = Clock::now();
    double construction_time = std::chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        std::cout << "Initial solution: obj=" << init_sol.objective()
                  << ", unassigned=" << init_sol.unassigned.size()
                  << " (" << construction_time << "s)\n";
    }

    // if (feasibility_mode != "slack") {
    //     init_sol.feasibility_mode = feasibility_mode;
    // }

    // SA parameters — calibrated per Gschwind & Drexl (2016)
    double sa_start_temp = 0.025 * init_sol.travel_cost();
    double sa_end_temp = 0.065 * sa_start_temp;

    double temperature = sa_start_temp;
    double sa_step;
    if (time_limit > 0.0) {
        sa_step = 1.0;  // no cooling until calibrated
    } else {
        sa_step = std::pow(sa_end_temp / sa_start_temp,
                          1.0 / max_iterations);
    }

    if (verbose) {
        std::cout << "  SA: start_temp=" << sa_start_temp
                  << ", end_temp=" << sa_end_temp
                  << ", travel_cost=" << init_sol.travel_cost()
                  << ", objective=" << init_sol.objective() << "\n";
    }

    DARPSolution best = init_sol.copy();
    double best_obj = best.objective();
    DARPSolution current = init_sol.copy();
    double current_obj = current.objective();
    const double obj_initial = current_obj;  // run-start objective for ML features
    int last_q = 0;                          // tracks q across iterations for ALNSState

    // Operator usage stats
    std::vector<int> destroy_count(registry.num_destroy(), 0);
    std::vector<int> repair_count(registry.num_repair(), 0);
    std::vector<int> destroy_improve(registry.num_destroy(), 0);
    std::vector<int> repair_improve(registry.num_repair(), 0);

    auto solve_t0 = Clock::now();
    bool sa_calibrated = false;

    double total_destroy_time = 0.0;
    double total_repair_time = 0.0;
    double total_other_time = 0.0;
    auto prev_iter_end = solve_t0;

    // Logging: header + initial weights + snapshot buffers
    if (logger) {
        logger->write_header(registry, max_iterations, sa_start_temp, sa_end_temp, sa_step,
                             feasibility_mode, unassigned_penalty, selection_config,
                             instance.num_requests, instance.num_vehicles);
        logger->log_weights(0, registry);
    }
    std::vector<std::vector<int>> pre_routes;
    std::set<int> pre_unassigned;
    std::set<int> pre_committed;
    std::vector<int> pre_min_insert_pos;
    double total_logging_time = 0.0;
    double total_selection_time = 0.0;  // always tracked so callers can report
                                        // it; only excluded from the budget if
                                        // exclude_selection_from_budget is set.

    int final_iter = 0;
    int iters_improving = 0;
    int iters_new_best = 0;
    int iters_sa_accepted = 0;
    for (int iter = 0; iter < max_iterations; iter++) {
        final_iter = iter + 1;

        if (time_limit > 0.0) {
            auto now = Clock::now();
            double elapsed = std::chrono::duration<double>(now - solve_t0).count()
                           - total_logging_time
                           - (exclude_selection_from_budget ? total_selection_time : 0.0);
            if (elapsed >= time_limit) {
                if (verbose) {
                    std::cout << "  Time limit reached at iter " << iter
                              << " (" << elapsed << "s)\n";
                }
                break;
            }

            // Calibrate cooling after warm-up iterations
            if (!sa_calibrated && iter == 10 && elapsed > 0.0) {
                double iters_per_sec = 10.0 / elapsed;
                int time_based_total = static_cast<int>(iters_per_sec * time_limit);
                int estimated_total = std::min(time_based_total, max_iterations);
                estimated_total = std::max(estimated_total, 100);
                sa_step = std::pow(sa_end_temp / sa_start_temp,
                                  1.0 / estimated_total);
                temperature = sa_start_temp;
                sa_calibrated = true;
                if (verbose) {
                    std::cout << "  SA calibrated: ~" << iters_per_sec
                              << " iter/s, estimated " << estimated_total
                              << " total, step=" << sa_step << "\n";
                }
                if (logger) {
                    logger->log_sa_calibrated(iter, sa_step, estimated_total);
                }
            }
        }
        // Select operators (unified dispatch: RANDOM / ROULETTE / ML_GUIDED).
        // last_q is the q from the PREVIOUS iter's destroy (0 at iter 0); ML
        // features see this slightly-lagged value because the next iter's q
        // isn't known until destroy runs. Static solve passes 0/0 for the
        // rolling_horizon fields (mode='dynamic' features won't read them).
        ALNSState alns_state{
            temperature, sa_start_temp,
            last_q, iter, max_iterations,
            obj_initial, best_obj,
            0.0, 0.0,
        };
        auto t_sel0 = Clock::now();
        auto [d_idx, r_idx] = registry.select_pair(rng, &current, &instance, &alns_state);
        total_selection_time += std::chrono::duration<double>(Clock::now() - t_sel0).count();
        const auto& destroy = registry.get_destroy(d_idx);
        const auto& repair  = registry.get_repair(r_idx);

        destroy_count[d_idx]++;
        repair_count[r_idx]++;

        // Hook B: pre-destroy snapshot for logging
        double temp_before = temperature;
        double best_obj_before = best_obj;
        if (logger) {
            pre_routes = current.routes;
            pre_unassigned = current.unassigned;
            pre_committed = current.committed;
            pre_min_insert_pos = current.min_insert_pos;
        }

        // Destroy + Repair
        auto t_destroy = Clock::now();
        DARPSolution destroyed = destroy(current, rng);

        // Hook C: compute q (always — ML features need it carried into the
        // next iteration's ALNSState, regardless of whether logging is on).
        int q = static_cast<int>(destroyed.unassigned.size())
              - static_cast<int>(current.unassigned.size());
        last_q = q;

        auto t_repair = Clock::now();
        DARPSolution candidate = repair(destroyed, rng);
        auto t_done = Clock::now();

        total_destroy_time += std::chrono::duration<double>(t_repair - t_destroy).count();
        total_repair_time += std::chrono::duration<double>(t_done - t_repair).count();
        total_other_time += std::chrono::duration<double>(t_destroy - prev_iter_end).count();

        double candidate_obj = candidate.objective();

        // SA acceptance
        double current_obj_before_accept = current_obj;
        double delta = candidate_obj - current_obj;
        bool accept = false;
        if (delta < -EPS) {
            accept = true;
            iters_improving++;
        } else {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < std::exp(-delta / temperature)) {
                accept = true;
                iters_sa_accepted++;
            }
        }

        // Score for weight update
        double score = 0.0;
        if (candidate_obj < best_obj - EPS) {
            score = 20.0;  // new global best
        } else if (delta < -EPS) {
            score = 10.0;  // improving
        } else if (accept) {
            score = 5.0;   // accepted worse
        }

        if (accept) {
            current = std::move(candidate);
            current_obj = current.objective();
        }

        if (current_obj < best_obj - EPS) {
            best = current.copy();
            best_obj = current_obj;
            destroy_improve[d_idx]++;
            repair_improve[r_idx]++;
            iters_new_best++;
        }

        registry.update_weights(d_idx, r_idx, score);
        temperature *= sa_step;

        // Hook D: log state (every iter or only at trial iters) + trials
        if (logger) {
            auto t_log0 = Clock::now();
            bool is_trial_iter = logger->should_run_trials(iter);

            // Log state: every iter if --log-state, or only at trial iters if --log-trials
            if (logger->is_logging_state() || is_trial_iter) {
                logger->log_iteration(iter, d_idx, r_idx,
                    registry.destroy_name(d_idx), registry.repair_name(r_idx),
                    current_obj_before_accept, candidate_obj, best_obj_before,
                    delta, temp_before, accept, q,
                    pre_routes, pre_unassigned, pre_committed, pre_min_insert_pos);
            }
            if (logger->is_logging_state() && (iter + 1) % 100 == 0) {
                logger->log_weights(iter + 1, registry);
            }
            // Hook E: M-trials
            if (is_trial_iter) {
                DARPSolution trial_state(instance);
                trial_state.routes = pre_routes;
                trial_state.unassigned = pre_unassigned;
                trial_state.committed = pre_committed;
                trial_state.min_insert_pos = pre_min_insert_pos;
                trial_state.feasibility_mode = feasibility_mode;
                trial_state.unassigned_penalty = unassigned_penalty;

                uint64_t trial_seed = static_cast<uint64_t>(seed) * 1000003ULL
                                    + static_cast<uint64_t>(iter);
                logger->run_and_log_trials(iter, trial_state, current_obj_before_accept,
                                           temp_before, registry, trial_seed);
            }
            auto t_log1 = Clock::now();
            total_logging_time += std::chrono::duration<double>(t_log1 - t_log0).count();
        }

        if (verbose && (iter + 1) % 100 == 0) {
            std::cout << "  iter " << (iter + 1)
                      << ": current=" << current_obj
                      << ", best=" << best_obj
                      << ", temp=" << temperature << "\n";
        }

        prev_iter_end = Clock::now();
    }

    auto solve_t1 = Clock::now();
    double solve_time = std::chrono::duration<double>(solve_t1 - solve_t0).count();

    if (verbose) {
        std::cout << "Best solution:    obj=" << best.objective()
                  << " (" << solve_time << "s)\n";

        // Print operator stats
        std::cout << "Operator stats:\n";
        for (int i = 0; i < registry.num_destroy(); i++) {
            std::cout << "  destroy/" << registry.destroy_name(i)
                      << ": used=" << destroy_count[i]
                      << ", improved=" << destroy_improve[i] << "\n";
        }
        for (int i = 0; i < registry.num_repair(); i++) {
            std::cout << "  repair/" << registry.repair_name(i)
                      << ": used=" << repair_count[i]
                      << ", improved=" << repair_improve[i] << "\n";
        }

        // Print final weights (for roulette wheel)
        if (registry.get_selection_method() == SelectionMethod::ROULETTE_WHEEL) {
            std::cout << "Final weights:\n";
            for (int i = 0; i < registry.num_destroy(); i++) {
                std::cout << "  destroy/" << registry.destroy_name(i)
                          << ": weight=" << registry.get_destroy_weights()[i] << "\n";
            }
            for (int i = 0; i < registry.num_repair(); i++) {
                std::cout << "  repair/" << registry.repair_name(i)
                          << ": weight=" << registry.get_repair_weights()[i] << "\n";
            }
        }

        std::cout << "Time breakdown:\n"
                << "  destroy: " << total_destroy_time << "s\n"
                << "  repair:  " << total_repair_time << "s\n"
                << "  other:   " << total_other_time << "s\n";
    }

    DARPSolution repaired = (feasibility_mode == "full")
        ? best.copy()
        : repair_ride_times(best, verbose);

    SolverResult result{
        std::move(repaired),
        std::move(init_sol),
        construction_time,
        solve_time,
        final_iter
    };
    result.iters_improving = iters_improving;
    result.iters_new_best = iters_new_best;
    result.iters_sa_accepted = iters_sa_accepted;
    result.total_destroy_time = total_destroy_time;
    result.total_repair_time = total_repair_time;
    result.total_other_time = total_other_time;
    result.total_selection_time = total_selection_time;
    return result;
}

// ---------------------------------------------------------------------------
// Warm-start ALNS: use provided initial solution, skip construction
// ---------------------------------------------------------------------------
SolverResult solve_warm(
    const DARPInstance& instance,
    const DARPSolution& initial_solution,
    int max_iterations,
    double start_temperature,
    double end_temperature,
    int seed,
    bool verbose,
    const std::string& feasibility_mode,
    const std::string& destroy_config,
    const std::string& repair_config,
    const std::string& selection_config,
    double time_limit,
    double unassigned_penalty,
    ALNSLogger* logger,
    const std::string& ml_model_dir,
    double sim_clock,
    double epoch_length,
    bool exclude_selection_from_budget,
    const std::string& log_ml_probs_path) {

    using Clock = std::chrono::high_resolution_clock;

    std::mt19937 rng(seed);

    OperatorRegistry registry = build_registry(destroy_config, repair_config, selection_config);
    attach_ml_selector_if_needed(registry, selection_config, ml_model_dir, verbose,
                                 log_ml_probs_path);

    DARPSolution init_sol = initial_solution.copy();
    init_sol.feasibility_mode = feasibility_mode;
    init_sol.unassigned_penalty = unassigned_penalty;

    // SA parameters — calibrated from initial solution cost
    double travel = init_sol.travel_cost();
    double sa_start_temp = (travel > 1e-6) ? 0.025 * travel : start_temperature;
    double sa_end_temp = 0.065 * sa_start_temp;

    double temperature = sa_start_temp;
    double sa_step;
    if (time_limit > 0.0) {
        sa_step = 1.0;
    } else {
        sa_step = std::pow(sa_end_temp / sa_start_temp, 1.0 / max_iterations);
    }

    DARPSolution best = init_sol.copy();
    double best_obj = best.objective();
    DARPSolution current = init_sol.copy();
    double current_obj = current.objective();
    const double obj_initial = current_obj;  // epoch-start objective for ML features
    int last_q = 0;                          // tracks q across iterations for ALNSState

    std::vector<int> destroy_count(registry.num_destroy(), 0);
    std::vector<int> repair_count(registry.num_repair(), 0);

    auto solve_t0 = Clock::now();
    bool sa_calibrated = false;
    int final_iter = 0;
    int iters_improving = 0;
    int iters_new_best = 0;
    int iters_sa_accepted = 0;
    double w_total_destroy_time = 0.0;
    double w_total_repair_time = 0.0;

    // Logging: header + initial weights + snapshot buffers
    if (logger) {
        logger->write_header(registry, max_iterations, sa_start_temp, sa_end_temp, sa_step,
                             feasibility_mode, unassigned_penalty, selection_config,
                             instance.num_requests, instance.num_vehicles);
        logger->log_weights(0, registry);
    }
    std::vector<std::vector<int>> pre_routes;
    std::set<int> pre_unassigned;
    std::set<int> pre_committed;
    std::vector<int> pre_min_insert_pos;
    double total_logging_time = 0.0;    // excludes logging/trial time from budget
    double total_selection_time = 0.0;  // always tracked; only excluded from the
                                        // budget if exclude_selection_from_budget
                                        // is set. ML inference dominates this.

    for (int iter = 0; iter < max_iterations; iter++) {
        final_iter = iter + 1;

        if (time_limit > 0.0) {
            auto now = Clock::now();
            double elapsed = std::chrono::duration<double>(now - solve_t0).count()
                           - total_logging_time
                           - (exclude_selection_from_budget ? total_selection_time : 0.0);
            if (elapsed >= time_limit) break;

            if (!sa_calibrated && iter == 10 && elapsed > 0.0) {
                double iters_per_sec = 10.0 / elapsed;
                int estimated_total = std::min(
                    static_cast<int>(iters_per_sec * time_limit), max_iterations);
                estimated_total = std::max(estimated_total, 100);
                sa_step = std::pow(sa_end_temp / sa_start_temp, 1.0 / estimated_total);
                temperature = sa_start_temp;
                sa_calibrated = true;
                if (logger) {
                    logger->log_sa_calibrated(iter, sa_step, estimated_total);
                }
            }
        }

        // Select operators (unified dispatch: RANDOM / ROULETTE / ML_GUIDED).
        // last_q is the q from the PREVIOUS iter's destroy (0 at iter 0).
        // sim_clock and epoch_length come from the dynamic_solver caller
        // (frozen at this epoch's trigger time; 0/0 if the caller is static).
        ALNSState alns_state{
            temperature, sa_start_temp,
            last_q, iter, max_iterations,
            obj_initial, best_obj,
            sim_clock, epoch_length,
        };
        auto t_sel0 = Clock::now();
        auto [d_idx, r_idx] = registry.select_pair(rng, &current, &instance, &alns_state);
        total_selection_time += std::chrono::duration<double>(Clock::now() - t_sel0).count();
        const auto& destroy = registry.get_destroy(d_idx);
        const auto& repair  = registry.get_repair(r_idx);
        destroy_count[d_idx]++;
        repair_count[r_idx]++;

        // Hook B: pre-destroy snapshot for logging
        double temp_before = temperature;
        double best_obj_before = best_obj;
        if (logger) {
            pre_routes = current.routes;
            pre_unassigned = current.unassigned;
            pre_committed = current.committed;
            pre_min_insert_pos = current.min_insert_pos;
        }

        auto t_d0 = Clock::now();
        DARPSolution destroyed = destroy(current, rng);

        // Hook C: compute q (always — needed by next iter's ALNSState).
        int q = static_cast<int>(destroyed.unassigned.size())
              - static_cast<int>(current.unassigned.size());
        last_q = q;

        auto t_d1 = Clock::now();
        DARPSolution candidate = repair(destroyed, rng);
        auto t_d2 = Clock::now();
        w_total_destroy_time += std::chrono::duration<double>(t_d1 - t_d0).count();
        w_total_repair_time += std::chrono::duration<double>(t_d2 - t_d1).count();
        double candidate_obj = candidate.objective();

        double current_obj_before_accept = current_obj;
        double delta = candidate_obj - current_obj;
        bool accept = false;
        if (delta < -1e-6) {
            accept = true;
            iters_improving++;
        } else {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < std::exp(-delta / temperature)) {
                accept = true;
                iters_sa_accepted++;
            }
        }

        double score = 0.0;
        if (candidate_obj < best_obj - 1e-6) score = 20.0;
        else if (delta < -1e-6) score = 10.0;
        else if (accept) score = 5.0;

        if (accept) {
            current = std::move(candidate);
            current_obj = current.objective();
        }
        if (current_obj < best_obj - 1e-6) {
            best = current.copy();
            best_obj = current_obj;
            iters_new_best++;
        }

        registry.update_weights(d_idx, r_idx, score);
        temperature *= sa_step;

        // Hook D: log state (every iter or only at trial iters) + trials
        if (logger) {
            auto t_log0 = Clock::now();
            bool is_trial_iter = logger->should_run_trials(iter);

            // Log state: every iter if --log-state, or only at trial iters if --log-trials
            if (logger->is_logging_state() || is_trial_iter) {
                logger->log_iteration(iter, d_idx, r_idx,
                    registry.destroy_name(d_idx), registry.repair_name(r_idx),
                    current_obj_before_accept, candidate_obj, best_obj_before,
                    delta, temp_before, accept, q,
                    pre_routes, pre_unassigned, pre_committed, pre_min_insert_pos);
            }
            if (logger->is_logging_state() && (iter + 1) % 100 == 0) {
                logger->log_weights(iter + 1, registry);
            }
            // Hook E: M-trials
            if (is_trial_iter) {
                DARPSolution trial_state(instance);
                trial_state.routes = pre_routes;
                trial_state.unassigned = pre_unassigned;
                trial_state.committed = pre_committed;
                trial_state.min_insert_pos = pre_min_insert_pos;
                trial_state.feasibility_mode = feasibility_mode;
                trial_state.unassigned_penalty = unassigned_penalty;

                uint64_t trial_seed = static_cast<uint64_t>(seed) * 1000003ULL
                                    + static_cast<uint64_t>(iter);
                logger->run_and_log_trials(iter, trial_state, current_obj_before_accept,
                                           temp_before, registry, trial_seed);
            }
            auto t_log1 = Clock::now();
            total_logging_time += std::chrono::duration<double>(t_log1 - t_log0).count();
        }
    }

    auto solve_t1 = Clock::now();
    double solve_time = std::chrono::duration<double>(solve_t1 - solve_t0).count();

    // Post-ALNS ride-time repair (same as static solver)
    DARPSolution repaired = (feasibility_mode == "full")
        ? best.copy()
        : repair_ride_times(best, verbose);

    SolverResult result{
        std::move(repaired),
        std::move(init_sol),
        0.0,  // no construction time for warm start
        solve_time,
        final_iter
    };
    result.iters_improving = iters_improving;
    result.iters_new_best = iters_new_best;
    result.iters_sa_accepted = iters_sa_accepted;
    result.total_destroy_time = w_total_destroy_time;
    result.total_repair_time = w_total_repair_time;
    result.total_other_time = solve_time - w_total_destroy_time - w_total_repair_time;
    result.total_selection_time = total_selection_time;
    return result;
}