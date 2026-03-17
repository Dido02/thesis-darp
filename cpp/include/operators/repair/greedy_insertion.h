#pragma once

#include "solution.h"
#include <random>

// Repair: reinsert all unassigned requests at cheapest feasible position
DARPSolution greedy_insertion(const DARPSolution& sol, std::mt19937& rng);
