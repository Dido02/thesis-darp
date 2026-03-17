#pragma once

#include "solution.h"
#include <random>

// Destroy: remove requests that are similar to each other (Shaw 1997)
// Uses relatedness based on distance, time, and load
DARPSolution shaw_removal(const DARPSolution& sol, std::mt19937& rng);