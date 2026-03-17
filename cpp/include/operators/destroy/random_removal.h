#pragma once

#include "solution.h"
#include <random>

// Destroy: remove 10-40% of assigned requests randomly
DARPSolution random_removal(const DARPSolution& sol, std::mt19937& rng);
