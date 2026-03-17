#pragma once

#include "solution.h"
#include <random>

// Destroy: remove requests whose removal saves the most cost
// Uses randomization parameter p (default 3) to bias toward worst-placed
DARPSolution worst_removal(const DARPSolution& sol, std::mt19937& rng);