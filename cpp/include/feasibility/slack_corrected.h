#pragma once

#include "instance.h"
#include <string>
#include <utility>
#include <vector>

// Slack-corrected feasibility checker (Steps 1+2 + per-pickup slack correction)
// Fast O(n) check. Default during ALNS search.
std::pair<bool, std::string> check_slack_corrected(
    const std::vector<int>& route, const DARPInstance& inst);
