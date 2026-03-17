#pragma once

#include "instance.h"
#include <string>
#include <utility>
#include <vector>

// Full 3-step optimizer feasibility checker
// Steps 1+2+3 (iterative per-pickup delay) + all constraint checks.
// Most accurate but slowest.
std::pair<bool, std::string> check_full_optimizer(
    const std::vector<int>& route, const DARPInstance& inst);
