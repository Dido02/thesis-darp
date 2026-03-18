#pragma once

#include "solution.h"
#include <random>

// Destroy: remove requests with highest ride-time utilization ratio
// (actual_ride_time / max_ride_time). These bottleneck requests
// constrain their routes the most.
DARPSolution ride_time_critical_removal(const DARPSolution& sol, std::mt19937& rng);