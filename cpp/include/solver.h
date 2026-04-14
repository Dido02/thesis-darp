#pragma once

#include "instance.h"
#include "solution.h"
#include <string>

class ALNSLogger;  // forward declaration

struct SolverResult {
    DARPSolution best;
    DARPSolution initial;
    double construction_time;
    double solve_time;
    int iterations_completed = 0;
    double total_destroy_time = 0.0;
    double total_repair_time = 0.0;
    double total_other_time = 0.0;
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
    ALNSLogger* logger = nullptr);

// Warm-start ALNS: skip initial solution construction, use provided solution
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
    ALNSLogger* logger = nullptr);
