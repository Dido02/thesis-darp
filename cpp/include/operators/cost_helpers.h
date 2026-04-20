#pragma once

#include "solution.h"

// Cost saved by removing a request (pickup+delivery pair) from a known route.
// Mirrors the helper that originated inside worst_removal.cpp; lifted to a
// shared header so feature extractors and operators compute contributions
// identically.
//
// Preconditions: `req` is the request id (1..n), `route_idx` indexes a route
// that contains BOTH the pickup (node id == req) AND its delivery
// (node id == req + n). Returns 0.0 if the pickup is missing.
double removal_cost(const DARPSolution& sol, int req, int route_idx);
