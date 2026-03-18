#pragma once

#include "solution.h"
#include <random>

// Destroy: remove requests from routes with lowest total slack.
// Rigid routes cannot absorb future requests — breaking them apart
// and redistributing creates more flexible routes.
DARPSolution low_slack_route_removal(const DARPSolution& sol, std::mt19937& rng);