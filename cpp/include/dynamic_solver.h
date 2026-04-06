#pragma once

#include "instance.h"
#include "solution.h"

#include <set>
#include <string>
#include <vector>

struct EpochConfig {
    enum TriggerMode { TIME_BASED, REQUEST_COUNT };
    TriggerMode trigger = TIME_BASED;
    double epoch_length = 30.0;       // seconds of sim time (TIME_BASED)
    int epoch_request_count = 50;     // request count trigger (REQUEST_COUNT)
    double epoch_budget_ms = 5000.0;  // wall-clock ms per ALNS epoch
};

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
    double obj_before;   // objective at start of epoch (before ALNS)
    double obj_after;    // objective at end of epoch (after ALNS)
    double wall_clock_ms;
    double destroy_time_ms;
    double repair_time_ms;
    double other_time_ms;  // overhead (precompute, copy, SA, etc.)
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
};

class DynamicSolver {
public:
    DynamicSolver(const DARPInstance& instance,
                  const EpochConfig& epoch_config,
                  const SolverParams& solver_params,
                  bool verbose = true);

    DynamicResult run();

private:
    const DARPInstance& instance_;
    EpochConfig epoch_config_;
    SolverParams solver_params_;
    bool verbose_;

    // Request pools
    std::vector<std::pair<double, int>> unreleased_;  // (release_time, request_id), sorted
    std::set<int> pending_;          // released but unassigned
    std::set<int> committed_;        // picked up, frozen in routes
    std::set<int> rejected_;         // expired unassigned

    DARPSolution current_solution_;
    double sim_clock_;
    int new_since_last_epoch_;       // for REQUEST_COUNT trigger

    void release_requests(double up_to_time);
    void greedy_insert_pending();
    void commit_picked_up_requests();
    void reject_expired_requests();
    void compute_min_insert_positions();
    EpochLog run_epoch_alns(int epoch_id);
};
