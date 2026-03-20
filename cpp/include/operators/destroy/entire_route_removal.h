#pragma once

#include "solution.h"
#include <random>

// Destroy: remove all requests from 1-2 random routes
DARPSolution entire_route_removal(const DARPSolution& sol, std::mt19937& rng);