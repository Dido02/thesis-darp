#pragma once

#include "instance.h"
#include <string>
#include <vector>

// Auxiliary data for O(1) insertion feasibility checks (Gschwind & Drexl 2019)
struct RouteAuxData {
    std::vector<int> route;
    std::vector<double> tau;          // AEAP begin-service (Step 1 only: true earliest)
    std::vector<double> tau_bar;      // ALAP begin-service (latest feasible)
    std::vector<double> depart;       // AEAP departure times
    std::vector<int> load;            // load at each position
    // tau_bar_given[i][k] = latest feasible time at pos i, given pos k fixed to tau[k]
    std::vector<std::vector<double>> tau_bar_given;  // r x r
    std::vector<std::vector<double>> gamma;          // cumulative travel time i->j
    std::vector<std::vector<double>> wait;           // cumulative waiting time i->j
    std::vector<std::vector<int>> max_load;          // max(load[i..j]) for O(1) capacity
};

// Run the AEAP scheduler, return {tau, depart, load}. No constraint checking.
// skip_step2=true: Steps 1+3 only (skip depot delay) — for O(1) precomputation.
//   Preserves the tau-to-tau_bar gap needed for Proposition 1 interpolation.
// skip_step2=false (default): full Steps 1+2+3.
void compute_aeap(const std::vector<int>& route, const DARPInstance& inst,
                  std::vector<double>& tau, std::vector<double>& depart,
                  std::vector<int>& load, bool skip_step2 = false);

// Compute latest feasible begin-service times (ALAP schedule).
void compute_alap(const std::vector<int>& route, const DARPInstance& inst,
                  const std::vector<double>& tau, const std::vector<double>& depart,
                  std::vector<double>& tau_bar);

// Compute constrained ALAP: fix position fix_pos to tau[fix_pos], propagate backward.
void compute_alap_constrained(const std::vector<int>& route, const DARPInstance& inst,
                              const std::vector<double>& tau, const std::vector<double>& depart,
                              const std::vector<double>& tau_bar,
                              int fix_pos, std::vector<double>& tau_bar_c);

// Precompute all auxiliary data for a route. O(r^3) total.
RouteAuxData precompute_route_aux(const std::vector<int>& route, const DARPInstance& inst);

// O(1) feasibility check for inserting request req at positions k (pickup) and l (delivery).
// k and l are positions in the ORIGINAL route (before insertion).
// Pickup inserted before route[k], delivery inserted before route[l].
// Requires k >= 1, l >= k+1, l <= route.size()-1.
bool check_insertion_o1(const RouteAuxData& aux, int req, int k, int l,
                        const DARPInstance& inst);

// ALAP-based feasibility check: determines whether a feasible schedule EXISTS
// for the given route, using Steps 1+2 AEAP + ALAP with ride-time/route-duration.
// This is the correct ground truth for the O(1) check.
std::pair<bool, std::string> check_alap_feasible(
    const std::vector<int>& route, const DARPInstance& inst);
