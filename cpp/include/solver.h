#pragma once

#include "instance.h"
#include "solution.h"
#include <optional>
#include <string>

class ALNSLogger;  // forward declaration

struct SolverResult {
    DARPSolution best;
    DARPSolution initial;
    double construction_time;
    double solve_time;
    int iterations_completed = 0;
    int iters_improving = 0;     // delta < -EPS (strict improvement over current)
    int iters_new_best = 0;      // candidate set a new global best (subset of iters_improving)
    int iters_sa_accepted = 0;   // delta >= -EPS but accepted by SA exp roll
    double total_destroy_time = 0.0;
    double total_repair_time = 0.0;
    double total_other_time = 0.0;
    double total_selection_time = 0.0;  // wall time spent in registry.select_pair
                                        // (ML inference dominates; roulette ≈ 0)
};

SolverResult solve(
    const DARPInstance& instance,
    int max_iterations = 1000,
    double start_temperature = 100.0,
    double end_temperature = 0.01,
    int seed = 42,
    bool verbose = true,
    const std::string& feasibility_mode = "o1",
    const std::string& destroy_config = "random",
    const std::string& repair_config = "greedy",
    const std::string& selection_config = "random",
    double time_limit = 0.0,
    double unassigned_penalty = 1000.0,
    const std::string& init_sort = "slack",
    ALNSLogger* logger = nullptr,
    const std::string& ml_model_dir = "",
    bool exclude_selection_from_budget = false,
    const std::string& log_ml_probs_path = "",
    std::optional<float> ml_softmax_temp = std::nullopt);

// Warm-start ALNS: skip initial solution construction, use provided solution.
// sim_clock and epoch_length are passed through to ALNSState so rolling_horizon
// ML features can read them. Both default to 0.0 (irrelevant for static runs).
SolverResult solve_warm(
    const DARPInstance& instance,
    const DARPSolution& initial_solution,
    int max_iterations = 1000,
    double start_temperature = 100.0,
    double end_temperature = 0.01,
    int seed = 42,
    bool verbose = false,
    const std::string& feasibility_mode = "o1",
    const std::string& destroy_config = "random",
    const std::string& repair_config = "greedy",
    const std::string& selection_config = "random",
    double time_limit = 0.0,
    double unassigned_penalty = 1000.0,
    ALNSLogger* logger = nullptr,
    const std::string& ml_model_dir = "",
    double sim_clock = 0.0,
    double epoch_length = 0.0,
    bool exclude_selection_from_budget = false,
    const std::string& log_ml_probs_path = "",
    std::optional<float> ml_softmax_temp = std::nullopt);
