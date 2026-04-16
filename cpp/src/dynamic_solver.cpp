#include "dynamic_solver.h"
#include "solver.h"
#include "feasibility/o1_feasibility.h"
#include "operators/repair/repair_utils.h"
#include "logging/alns_logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

static constexpr double EPS = 1e-6;

DynamicSolver::DynamicSolver(const DARPInstance& instance,
                             const EpochConfig& epoch_config,
                             const SolverParams& solver_params,
                             bool verbose)
    : instance_(instance),
      epoch_config_(epoch_config),
      solver_params_(solver_params),
      verbose_(verbose),
      current_solution_(instance),
      sim_clock_(0.0),
      new_since_last_epoch_(0) {

    // Initialize current solution with empty routes (all requests unassigned)
    // We'll clear unassigned since requests start as unreleased
    current_solution_.unassigned.clear();
    current_solution_.feasibility_mode = solver_params_.feasibility_mode;
    current_solution_.unassigned_penalty = solver_params_.unassigned_penalty;
    current_solution_.top_n_routes = solver_params_.top_n_routes;

    // Build unreleased list sorted by release time (= pickup earliest time)
    for (int req = 1; req <= instance_.num_requests; req++) {
        double release_time = instance_.nodes[req].earliest;
        unreleased_.push_back({release_time, req});
    }
    std::sort(unreleased_.begin(), unreleased_.end());
}

// ---------------------------------------------------------------------------
// Release requests whose release time <= up_to_time
// ---------------------------------------------------------------------------
void DynamicSolver::release_requests(double up_to_time) {
    while (!unreleased_.empty() && unreleased_.front().first <= up_to_time + EPS) {
        int req = unreleased_.front().second;
        unreleased_.erase(unreleased_.begin());
        pending_.insert(req);
        new_since_last_epoch_++;
    }
}

// ---------------------------------------------------------------------------
// Greedy pre-insertion of pending requests between epochs
// ---------------------------------------------------------------------------
void DynamicSolver::greedy_insert_pending() {
    if (pending_.empty()) return;

    const auto& tt = instance_.travel_time;
    const auto& nodes = instance_.nodes;
    int n = instance_.num_requests;

    // Sort pending by deadline urgency (tightest first)
    std::vector<std::pair<double, int>> urgent;
    for (int req : pending_) {
        urgent.push_back({nodes[req].latest, req});
    }
    std::sort(urgent.begin(), urgent.end());

    // Compute min insertion positions
    compute_min_insert_positions();

    if (current_solution_.feasibility_mode == "o1") {
        // O(1) feasibility path — lazy aux precomputation
        std::vector<RouteAuxData> aux_cache(current_solution_.routes.size());
        std::vector<bool> aux_computed(current_solution_.routes.size(), false);

        for (auto& [deadline, req] : urgent) {
            if (!pending_.count(req)) continue;

            int pickup_id = req;
            int delivery_id = req + n;
            double best_cost = std::numeric_limits<double>::infinity();
            int best_route = -1, best_p = -1, best_d = -1;

            // Build candidate route list (top-N nearest)
            auto candidates = build_candidate_routes(
                current_solution_, pickup_id, current_solution_.top_n_routes);

            for (int k : candidates) {
                const auto& route = current_solution_.routes[k];
                int rlen = static_cast<int>(route.size());

                int min_pos = instance_.route_start_offset();
                if (!current_solution_.min_insert_pos.empty())
                    min_pos = std::max(min_pos, current_solution_.min_insert_pos[k]);
                if (min_pos >= rlen) continue;

                // Lazy aux precomputation
                if (!aux_computed[k]) {
                    aux_cache[k] = precompute_route_aux(route, instance_);
                    aux_computed[k] = true;
                }
                const auto& aux = aux_cache[k];

                for (int k_pos = min_pos; k_pos < rlen; k_pos++) {
                    for (int l_pos = k_pos; l_pos < rlen; l_pos++) {
                        if (!check_insertion_o1(aux, req, k_pos, l_pos, instance_))
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

                        if (delta < best_cost) {
                            best_cost = delta;
                            best_route = k;
                            best_p = k_pos;
                            best_d = l_pos + 1;
                        }
                    }
                }
            }

            if (best_route >= 0) {
                current_solution_.routes[best_route].insert(
                    current_solution_.routes[best_route].begin() + best_p, req);
                current_solution_.routes[best_route].insert(
                    current_solution_.routes[best_route].begin() + best_d, req + n);
                pending_.erase(req);
                current_solution_.cached_obj_ = -1.0;

                // Recompute aux data and min_insert_pos for modified route
                aux_cache[best_route] = precompute_route_aux(
                    current_solution_.routes[best_route], instance_);
                // Update min_insert_pos for this route
                if (!current_solution_.min_insert_pos.empty()) {
                    const auto& route = current_solution_.routes[best_route];
                    auto sched = current_solution_.compute_route_schedule(route);
                    int last_visited = 0;
                    for (int j = 0; j < static_cast<int>(route.size()); j++) {
                        if (sched.begin_service[j] <= sim_clock_ + EPS) last_visited = j;
                    }
                    current_solution_.min_insert_pos[best_route] = last_visited + 1;
                }
            }
        }
    } else {
        // Classic feasibility path
        for (auto& [deadline, req] : urgent) {
            if (!pending_.count(req)) continue;

            int pickup_id = req;
            int delivery_id = req + n;
            double best_cost = std::numeric_limits<double>::infinity();
            int best_route = -1, best_p = -1, best_d = -1;

            for (int k = 0; k < static_cast<int>(current_solution_.routes.size()); k++) {
                const auto& route = current_solution_.routes[k];
                int rlen = static_cast<int>(route.size());

                int min_pos = instance_.route_start_offset();
                if (!current_solution_.min_insert_pos.empty())
                    min_pos = std::max(min_pos, current_solution_.min_insert_pos[k]);

                for (int p_pos = min_pos; p_pos < rlen; p_pos++) {
                    for (int d_pos = p_pos + 1; d_pos <= rlen; d_pos++) {
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

                        std::vector<int> candidate = route;
                        candidate.insert(candidate.begin() + p_pos, pickup_id);
                        candidate.insert(candidate.begin() + d_pos, delivery_id);

                        auto [feasible, reason] = current_solution_.check_route_feasibility(candidate);
                        if (feasible) {
                            best_cost = delta;
                            best_route = k;
                            best_p = p_pos;
                            best_d = d_pos;
                        }
                    }
                }
            }

            if (best_route >= 0) {
                current_solution_.routes[best_route].insert(
                    current_solution_.routes[best_route].begin() + best_p, req);
                current_solution_.routes[best_route].insert(
                    current_solution_.routes[best_route].begin() + best_d, req + n);
                pending_.erase(req);
                current_solution_.cached_obj_ = -1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Commit requests whose scheduled pickup time <= sim_clock
// ---------------------------------------------------------------------------
void DynamicSolver::commit_picked_up_requests() {
    int n = instance_.num_requests;

    for (const auto& route : current_solution_.routes) {
        if (static_cast<int>(route.size()) <= instance_.empty_route_size()) continue;

        auto sched = current_solution_.compute_route_schedule(route);

        for (int j = 0; j < static_cast<int>(route.size()); j++) {
            int nid = route[j];
            // Only commit via pickup nodes
            if (nid >= 1 && nid <= n && !committed_.count(nid)) {
                if (sched.begin_service[j] <= sim_clock_ + EPS) {
                    committed_.insert(nid);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reject pending requests whose latest pickup time has passed
// ---------------------------------------------------------------------------
void DynamicSolver::reject_expired_requests() {
    std::vector<int> to_reject;
    for (int req : pending_) {
        if (instance_.nodes[req].latest < sim_clock_ - EPS) {
            to_reject.push_back(req);
        }
    }
    for (int req : to_reject) {
        pending_.erase(req);
        rejected_.insert(req);
        if (verbose_) {
            std::cout << "  [REJECT] request " << req
                      << " (deadline=" << instance_.nodes[req].latest
                      << ", t=" << sim_clock_ << ")\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Compute min insertion position per route based on sim_clock
// ---------------------------------------------------------------------------
void DynamicSolver::compute_min_insert_positions() {
    int K = static_cast<int>(current_solution_.routes.size());
    current_solution_.min_insert_pos.resize(K);
    current_solution_.route_ready_time.resize(K);
    current_solution_.route_ready_node.resize(K);

    for (int k = 0; k < K; k++) {
        const auto& route = current_solution_.routes[k];
        int rlen = static_cast<int>(route.size());

        // Empty routes (depot-[vstart]-depot) are always fully available
        if (rlen <= instance_.empty_route_size()) {
            int start_offset = instance_.route_start_offset();
            current_solution_.min_insert_pos[k] = start_offset;
            int depot_node = route[start_offset - 1];
            current_solution_.route_ready_node[k] = depot_node;
            current_solution_.route_ready_time[k] =
                instance_.nodes[depot_node].earliest +
                instance_.nodes[depot_node].service_duration;
            continue;
        }

        auto sched = current_solution_.compute_route_schedule(route);

        // Find last visited stop, but exclude the destination depot —
        // a vehicle that has served all its requests and returned to depot
        // is still available for new insertions before the depot.
        int last_visited = 0;
        int dest_depot = 2 * instance_.num_requests + 1;
        for (int j = 0; j < rlen; j++) {
            if (route[j] == dest_depot) continue;  // don't count dest depot as visited
            if (route[j] == 0) continue;            // don't count origin depot
            if (sched.begin_service[j] <= sim_clock_ + EPS) {
                last_visited = j;
            }
        }
        // Insert only after the last visited non-depot position
        int min_pos = std::max(instance_.route_start_offset(), last_visited + 1);
        current_solution_.min_insert_pos[k] = min_pos;

        // Cache the actual departure time and node at the position
        // immediately before the insertion point.
        int prev_idx = min_pos - 1;
        current_solution_.route_ready_node[k] = route[prev_idx];
        current_solution_.route_ready_time[k] = sched.departure_time[prev_idx];
    }
}

// ---------------------------------------------------------------------------
// Run one epoch of ALNS optimization
// ---------------------------------------------------------------------------
EpochLog DynamicSolver::run_epoch_alns(int epoch_id) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // Prepare solution for ALNS
    compute_min_insert_positions();
    DARPSolution epoch_sol = current_solution_.copy();

    // Add pending requests to unassigned pool
    for (int req : pending_) {
        epoch_sol.unassigned.insert(req);
    }
    // Set committed set — include unreleased requests so destroy operators
    // don't waste time trying to remove requests that aren't in any route
    epoch_sol.committed = committed_;
    for (const auto& [release_time, req] : unreleased_) {
        epoch_sol.committed.insert(req);
    }

    // Reset stats counters for this epoch
    DynamicStats::reset();

    // Capture pre-ALNS objective
    double obj_before = epoch_sol.travel_cost();

    // Emit epoch_start record if logging
    if (solver_params_.logger) {
        solver_params_.logger->log_epoch_start(
            epoch_id, sim_clock_,
            epoch_sol.routes, epoch_sol.unassigned,
            epoch_sol.committed, epoch_sol.min_insert_pos,
            pending_, rejected_, obj_before);
    }

    // Run ALNS with time budget
    double time_budget = epoch_config_.epoch_budget_ms / 1000.0;
    SolverResult result = solve_warm(
        instance_, epoch_sol,
        solver_params_.max_iterations,
        solver_params_.sa_start,
        solver_params_.sa_end,
        solver_params_.seed + epoch_id,  // vary seed per epoch
        false,  // not verbose inside ALNS
        solver_params_.feasibility_mode,
        solver_params_.destroy_config,
        solver_params_.repair_config,
        solver_params_.selection_config,
        time_budget,
        solver_params_.unassigned_penalty,
        solver_params_.logger,
        solver_params_.ml_model_dir);

    // Update current solution from ALNS result
    current_solution_ = std::move(result.best);
    // Preserve committed and min_insert_pos
    current_solution_.committed = committed_;
    compute_min_insert_positions();

    // Update pending from ALNS unassigned (anything ALNS couldn't place)
    pending_.clear();
    for (int req : current_solution_.unassigned) {
        // Only keep as pending if not yet rejected
        if (!rejected_.count(req)) {
            pending_.insert(req);
        }
    }

    auto t1 = Clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Count stats
    int total_known = instance_.num_requests - static_cast<int>(unreleased_.size());
    int total_assigned = total_known - static_cast<int>(pending_.size())
                       - static_cast<int>(rejected_.size());

    EpochLog log;
    log.epoch_id = epoch_id;
    log.sim_time = sim_clock_;
    log.requests_known = total_known;
    log.requests_assigned = total_assigned;
    log.requests_committed = static_cast<int>(committed_.size());
    log.requests_uncommitted = total_assigned - static_cast<int>(committed_.size());
    log.requests_pending = static_cast<int>(pending_.size());
    log.requests_rejected = static_cast<int>(rejected_.size());
    log.alns_iterations = result.iterations_completed;
    log.obj_before = obj_before;
    log.obj_after = current_solution_.travel_cost();
    log.wall_clock_ms = wall_ms;
    log.destroy_time_ms = result.total_destroy_time * 1000.0;
    log.repair_time_ms = result.total_repair_time * 1000.0;
    log.other_time_ms = wall_ms - log.destroy_time_ms - log.repair_time_ms;

    // Performance stats
    int iters = std::max(1, result.iterations_completed);
    log.avg_routes_per_request = DynamicStats::request_evals > 0
        ? static_cast<int>(DynamicStats::routes_checked / DynamicStats::request_evals) : 0;
    log.avg_feas_checks = static_cast<int>(DynamicStats::feas_checks / iters);
    log.aux_precompute_ms = DynamicStats::aux_precompute_ms;
    log.greedy_preinsert_ms = greedy_preinsert_ms_;
    greedy_preinsert_ms_ = 0.0;  // reset for next epoch

    return log;
}

// ---------------------------------------------------------------------------
// Main simulation loop
// ---------------------------------------------------------------------------
DynamicResult DynamicSolver::run() {
    using Clock = std::chrono::high_resolution_clock;
    auto wall_t0 = Clock::now();

    // Determine simulation time range
    // Start: vehicle start time (depot earliest)
    sim_clock_ = instance_.nodes[0].earliest;

    // End: latest request deadline + some buffer
    double end_time = 0.0;
    for (int req = 1; req <= instance_.num_requests; req++) {
        end_time = std::max(end_time, instance_.nodes[req].latest);
    }

    if (verbose_) {
        std::cout << "=== Dynamic DARP Simulation ===\n"
                  << "  Requests: " << instance_.num_requests << "\n"
                  << "  Vehicles: " << instance_.num_vehicles << "\n"
                  << "  Sim start: " << sim_clock_ << "\n"
                  << "  Sim end:   " << end_time << "\n"
                  << "  Epoch mode: "
                  << (epoch_config_.trigger == EpochConfig::TIME_BASED
                      ? "time-based (" + std::to_string(epoch_config_.epoch_length) + "s)"
                      : "request-count (" + std::to_string(epoch_config_.epoch_request_count) + ")")
                  << "\n"
                  << "  Epoch budget: " << epoch_config_.epoch_budget_ms << "ms\n\n";
    }

    std::vector<EpochLog> epoch_logs;
    int epoch_id = 0;
    double next_epoch_time = sim_clock_ + epoch_config_.epoch_length;

    while (sim_clock_ <= end_time + EPS || !pending_.empty()) {
        // 1. Release all requests at current time (tie-breaking: releases first)
        release_requests(sim_clock_);

        // 2. Greedy pre-insertion of pending requests
        {
            auto gt0 = Clock::now();
            greedy_insert_pending();
            auto gt1 = Clock::now();
            greedy_preinsert_ms_ += std::chrono::duration<double, std::milli>(gt1 - gt0).count();
        }

        // 3. Commit picked-up requests
        commit_picked_up_requests();

        // 4. Reject expired requests
        reject_expired_requests();

        // 5. Check epoch trigger
        bool trigger_epoch = false;
        if (epoch_config_.trigger == EpochConfig::TIME_BASED) {
            trigger_epoch = (sim_clock_ >= next_epoch_time - EPS);
        } else {
            trigger_epoch = (new_since_last_epoch_ >= epoch_config_.epoch_request_count);
        }

        // Only trigger if there's something to optimize
        int total_known = instance_.num_requests - static_cast<int>(unreleased_.size());
        int total_assigned = total_known - static_cast<int>(pending_.size())
                           - static_cast<int>(rejected_.size());
        int assigned_uncommitted = total_assigned - static_cast<int>(committed_.size());
        bool has_work = !pending_.empty() || assigned_uncommitted > 0;

        if (trigger_epoch && has_work) {
            epoch_id++;
            EpochLog log = run_epoch_alns(epoch_id);
            epoch_logs.push_back(log);
            new_since_last_epoch_ = 0;

            if (verbose_) {
                std::cout << "[Epoch " << log.epoch_id << "] t=" << log.sim_time
                          << " | known=" << log.requests_known
                          << " committed=" << log.requests_committed
                          << " uncommitted=" << log.requests_uncommitted
                          << " pending=" << log.requests_pending
                          << " rejected=" << log.requests_rejected
                          << "\n | iters=" << log.alns_iterations
                          << " obj=" << log.obj_before << "->" << log.obj_after
                          << " (" << (log.obj_before > 1e-6 ? -100.0*(log.obj_after - log.obj_before)/log.obj_before : 0.0) << "%)"
                          << " | destroy=" << static_cast<int>(log.destroy_time_ms)
                          << "ms repair=" << static_cast<int>(log.repair_time_ms)
                          << "ms other=" << static_cast<int>(log.other_time_ms)
                          << "ms total=" << static_cast<int>(log.wall_clock_ms) << "ms";
                if (solver_params_.dynamic_stats) {
                    std::cout << " \n| routes/req=" << log.avg_routes_per_request
                              << "/" << static_cast<int>(current_solution_.routes.size())
                              << " feas_checks=" << log.avg_feas_checks << "/iter"
                              << " aux=" << static_cast<int>(log.aux_precompute_ms) << "ms"
                              << " greedy=" << static_cast<int>(log.greedy_preinsert_ms) << "ms";
                    // Per-repair-operator timing
                    std::cout << "\n| repair_ops:";
                    if (DynamicStats::repair_greedy_count > 0)
                        std::cout << " greedy=" << static_cast<int>(DynamicStats::repair_greedy_ms) << "ms(" << DynamicStats::repair_greedy_count << ")";
                    if (DynamicStats::repair_regret2_count > 0)
                        std::cout << " regret2=" << static_cast<int>(DynamicStats::repair_regret2_ms) << "ms(" << DynamicStats::repair_regret2_count << ")";
                    if (DynamicStats::repair_regret3_count > 0)
                        std::cout << " regret3=" << static_cast<int>(DynamicStats::repair_regret3_ms) << "ms(" << DynamicStats::repair_regret3_count << ")";
                    if (DynamicStats::repair_regret4_count > 0)
                        std::cout << " regret4=" << static_cast<int>(DynamicStats::repair_regret4_ms) << "ms(" << DynamicStats::repair_regret4_count << ")";
                    if (DynamicStats::repair_regretM_count > 0)
                        std::cout << " regretM=" << static_cast<int>(DynamicStats::repair_regretM_ms) << "ms(" << DynamicStats::repair_regretM_count << ")";
                    if (DynamicStats::repair_slack_count > 0)
                        std::cout << " slack=" << static_cast<int>(DynamicStats::repair_slack_ms) << "ms(" << DynamicStats::repair_slack_count << ")";
                    // Per-destroy-operator timing
                    std::cout << "\n| destroy_ops:";
                    if (DynamicStats::destroy_random_count > 0)
                        std::cout << " rand=" << static_cast<int>(DynamicStats::destroy_random_ms) << "ms(" << DynamicStats::destroy_random_count << ")";
                    if (DynamicStats::destroy_worst_count > 0)
                        std::cout << " worst=" << static_cast<int>(DynamicStats::destroy_worst_ms) << "ms(" << DynamicStats::destroy_worst_count << ")";
                    if (DynamicStats::destroy_shaw_count > 0)
                        std::cout << " shaw=" << static_cast<int>(DynamicStats::destroy_shaw_ms) << "ms(" << DynamicStats::destroy_shaw_count << ")";
                    if (DynamicStats::destroy_rtcritical_count > 0)
                        std::cout << " rtcrit=" << static_cast<int>(DynamicStats::destroy_rtcritical_ms) << "ms(" << DynamicStats::destroy_rtcritical_count << ")";
                    if (DynamicStats::destroy_lowslack_count > 0)
                        std::cout << " lowslack=" << static_cast<int>(DynamicStats::destroy_lowslack_ms) << "ms(" << DynamicStats::destroy_lowslack_count << ")";
                    if (DynamicStats::destroy_entireroute_count > 0)
                        std::cout << " entire=" << static_cast<int>(DynamicStats::destroy_entireroute_ms) << "ms(" << DynamicStats::destroy_entireroute_count << ")";
                    std::cout << "\n";
                }
                std::cout << "\n";
            }

            if (epoch_config_.trigger == EpochConfig::TIME_BASED) {
                next_epoch_time = sim_clock_ + epoch_config_.epoch_length;
            }
        } else if (trigger_epoch) {
            // Reset epoch trigger even if no work
            new_since_last_epoch_ = 0;
            if (epoch_config_.trigger == EpochConfig::TIME_BASED) {
                next_epoch_time = sim_clock_ + epoch_config_.epoch_length;
            }
        }

        // 6. Advance clock to next event
        double next_time = std::numeric_limits<double>::infinity();

        // Next epoch boundary
        if (epoch_config_.trigger == EpochConfig::TIME_BASED) {
            next_time = std::min(next_time, next_epoch_time);
        }

        // Next request release
        if (!unreleased_.empty()) {
            next_time = std::min(next_time, unreleased_.front().first);
        }

        // If no more events, we're done
        if (next_time == std::numeric_limits<double>::infinity() || next_time > end_time + EPS) {
            if (pending_.empty()) break;
            // Still have pending — advance to reject them
            sim_clock_ = end_time + 1.0;
            reject_expired_requests();
            break;
        }

        if (next_time <= sim_clock_ + EPS) {
            // Avoid infinite loop — advance by minimum step
            sim_clock_ += epoch_config_.epoch_length;
        } else {
            sim_clock_ = next_time;
        }
    }

    auto wall_t1 = Clock::now();
    double total_wall = std::chrono::duration<double>(wall_t1 - wall_t0).count();

    // Compute final stats
    int total_served = 0;
    for (int req = 1; req <= instance_.num_requests; req++) {
        if (!rejected_.count(req) && !pending_.count(req)) {
            total_served++;
        }
    }
    int total_rejected = static_cast<int>(rejected_.size()) + static_cast<int>(pending_.size());
    double total_travel = current_solution_.travel_cost();
    double service_rate = (instance_.num_requests > 0)
        ? 100.0 * total_served / instance_.num_requests : 0.0;

    if (verbose_) {
        std::cout << "\n=== Dynamic DARP Summary ===\n"
                  << "Total requests:  " << instance_.num_requests << "\n"
                  << "Served:          " << total_served
                  << " (" << service_rate << "%)\n"
                  << "Rejected:        " << total_rejected << "\n"
                  << "Total travel:    " << total_travel << "\n"
                  << "Epochs:          " << epoch_id << "\n"
                  << "Wall-clock:      " << total_wall << "s\n";
    }

    return DynamicResult{
        current_solution_.copy(),
        std::move(epoch_logs),
        total_served,
        total_rejected,
        total_travel,
        service_rate
    };
}
