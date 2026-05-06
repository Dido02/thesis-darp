#pragma once

#include "instance.h"
#include "solution.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

class ALNSLogger;  // forward declaration

struct EpochConfig {
    enum TriggerMode { TIME_BASED, REQUEST_COUNT };
    TriggerMode trigger = TIME_BASED;
    double epoch_length = 30.0;       // seconds of sim time (TIME_BASED)
    int epoch_request_count = 50;     // request count trigger (REQUEST_COUNT)
    double epoch_budget_ms = 5000.0;  // wall-clock ms per ALNS epoch
};

// Commit / min-insert-position policy for the dynamic simulator.
// Legacy (default): a pickup commits when its eager-schedule begin_service has
//   passed sim_clock_. This is the historical behavior — keep all existing
//   bench results reproducible.
// Lazy: a pickup commits only when its latest-feasible begin time has passed
//   sim_clock_. Matches the dynamic-DARP literature (Cordeau & Laporte 2003,
//   Schilde 2011, Bongiovanni 2022). Defers commits while time-window slack
//   remains, leaving more material for ALNS to optimize per epoch.
enum class SchedulingPolicy { Legacy, Lazy };

struct EpochLog {
    int epoch_id;
    double sim_time;
    int requests_known;
    int requests_assigned;
    int requests_committed;
    int requests_uncommitted;  // assigned - committed: what ALNS can rearrange
    int requests_pending;
    int requests_rejected;
    int alns_iterations;
    int iters_improving = 0;     // delta < -EPS within the epoch
    int iters_new_best = 0;      // new global best within the epoch (subset of iters_improving)
    int iters_sa_accepted = 0;   // SA-accepted worsening within the epoch
    double obj_before;   // objective at start of epoch (before ALNS)
    double obj_after;    // objective at end of epoch (after ALNS)
    double wall_clock_ms;
    double destroy_time_ms;
    double repair_time_ms;
    double other_time_ms;  // overhead (precompute, copy, SA, etc.)
    // Stats (only when --dynamic-stats is enabled)
    int avg_routes_per_request = 0;
    int avg_feas_checks = 0;
    double aux_precompute_ms = 0.0;
    double greedy_preinsert_ms = 0.0;
    double selection_time_ms = 0.0;  // wall time spent in select_pair (ML inference dominates)
};

struct DynamicResult {
    DARPSolution final_solution;
    std::vector<EpochLog> epoch_logs;
    int total_served;
    int total_rejected;
    double total_travel_time;
    double service_rate;
};

// Parameters forwarded to the static ALNS solver at each epoch
struct SolverParams {
    int max_iterations = 100000;
    double sa_start = 100.0;
    double sa_end = 0.01;
    int seed = 42;
    std::string feasibility_mode = "o1";
    std::string destroy_config = "random,worst,shaw,rtcritical,lowslack";
    std::string repair_config = "greedy,regret2,regret3,slack";
    std::string selection_config = "roulette";
    double unassigned_penalty = 1000.0;
    int top_n_routes = 0;  // limit repair to N nearest routes (0 = no limit)
    bool dynamic_stats = false;  // print per-epoch performance stats
    bool exclude_selection_from_budget = false;  // subtract operator-selection
                                                  // wall time (e.g. ML ONNX
                                                  // inference) from epoch budget
                                                  // so ALNS gets the same
                                                  // compute regardless of
                                                  // selection strategy.
    ALNSLogger* logger = nullptr;  // non-owning, nullable
    std::string ml_model_dir;  // empty = no ML; required when selection_config == "ml"
    std::string log_ml_probs_path;  // empty = no logging; if set, MLSelector writes
                                    // per-iteration probability JSONL to this path.
    std::optional<float> ml_softmax_temp;  // empty = default proportional sampler;
                                           // if set, MLSelector uses softmax(p / T)
                                           // for picking. T<1 sharpens contrast.
};

class DynamicSolver {
public:
    DynamicSolver(const DARPInstance& instance,
                  const EpochConfig& epoch_config,
                  const SolverParams& solver_params,
                  bool verbose = true,
                  SchedulingPolicy policy = SchedulingPolicy::Legacy);

    DynamicResult run();

private:
    const DARPInstance& instance_;
    EpochConfig epoch_config_;
    SolverParams solver_params_;
    bool verbose_;
    SchedulingPolicy policy_;

    // Request pools
    std::vector<std::pair<double, int>> unreleased_;  // (release_time, request_id), sorted
    std::set<int> pending_;          // released but unassigned
    std::set<int> committed_;        // picked up, frozen in routes
    std::set<int> rejected_;         // expired unassigned

    DARPSolution current_solution_;
    double sim_clock_;
    int new_since_last_epoch_;       // for REQUEST_COUNT trigger
    double greedy_preinsert_ms_ = 0.0;  // accumulated greedy pre-insertion time per epoch

    void release_requests(double up_to_time);
    void greedy_insert_pending();
    void commit_picked_up_requests();
    void reject_expired_requests();
    void compute_min_insert_positions();
    EpochLog run_epoch_alns(int epoch_id);
};
