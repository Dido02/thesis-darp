#pragma once

#include "instance.h"

#include <vector>

// Total forward time slack of a route (sum of per-position slacks).
// Mirrors the helper that originated inside slack_insertion.cpp; lifted to a
// shared header so feature extractors and operators compute slack identically.
//
// Algorithm: forward pass to compute begin-service times, then a backward
// recurrence for slack[i] = min(latest[i] - tau[i], slack[i+1] + wait_next).
// Returns 0.0 for empty routes (size <= empty_route_size()).
double compute_route_slack(const std::vector<int>& route, const DARPInstance& inst);

// Sum of edge travel times along a route.
double compute_route_cost(const std::vector<int>& route, const DARPInstance& inst);
