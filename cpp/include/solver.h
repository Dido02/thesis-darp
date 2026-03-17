#pragma once

#include "instance.h"
#include "solution.h"

#include <string>

struct SolverResult {
    DARPSolution best;
    DARPSolution initial;
    double construction_time;
    double solve_time;
};

SolverResult solve(
    const DARPInstance& instance,
    int max_iterations = 1000,
    double start_temperature = 100.0,
    double end_temperature = 0.01,
    int seed = 42,
    bool verbose = true,
    const std::string& feasibility_mode = "slack");
