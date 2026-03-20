#pragma once

#include "solution.h"
#include <random>

// Repair: insert request with highest regret-k value
// Regret-k = cost(k-th best position) - cost(best position)
DARPSolution regret2_insertion(const DARPSolution& sol, std::mt19937& rng);
DARPSolution regret3_insertion(const DARPSolution& sol, std::mt19937& rng);
DARPSolution regret4_insertion(const DARPSolution& sol, std::mt19937& rng);
DARPSolution regretM_insertion(const DARPSolution& sol, std::mt19937& rng);