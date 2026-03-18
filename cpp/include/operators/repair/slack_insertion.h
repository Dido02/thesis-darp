#pragma once

#include "solution.h"
#include <random>

// Repair: insert at positions that balance cost with remaining schedule
// flexibility (forward time slack). Controlled by parameter alpha:
//   score = (1 - alpha) * normalized_cost + alpha * normalized_slack_loss
// alpha = 0 is pure greedy, alpha = 1 is pure slack preservation.
DARPSolution slack_insertion(const DARPSolution& sol, std::mt19937& rng);