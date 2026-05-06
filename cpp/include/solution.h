#pragma once

#include "instance.h"

#include <map>
#include <set>
#include <string>
#include <vector>

struct RouteSchedule {
    std::vector<double> arrival_time;
    std::vector<double> begin_service;
    std::vector<double> departure_time;
    std::vector<int> load;
    std::map<int, double> ride_time;  // request_id -> ride time
};

class DARPSolution {
public:
    double unassigned_penalty = 1000.0;

    const DARPInstance* instance;  // non-owning pointer, always valid
    std::vector<std::vector<int>> routes;
    std::set<int> unassigned;
    std::set<int> committed;              // frozen requests (dynamic DARP)
    std::vector<int> min_insert_pos;      // per-route minimum insertion position (dynamic DARP)
    std::vector<double> route_ready_time; // actual departure time at min_insert_pos[k] - 1 (dynamic DARP)
    std::vector<int> route_ready_node;    // node id at min_insert_pos[k] - 1 (dynamic DARP)
    int top_n_routes = 0;                 // limit repair to N nearest routes per request (0 = no limit)
    std::string feasibility_mode = "o1";  // "slack" or "full"
    mutable double cached_obj_ = -1.0;

    explicit DARPSolution(const DARPInstance& inst);

    DARPSolution copy() const;

    // Scheduling
    RouteSchedule compute_route_schedule(const std::vector<int>& route) const;

    // ALAP backward-sweep schedule used by the lazy commit policy in dynamic
    // mode. Returns latest_begin[j] = the latest sim time at which the vehicle
    // can START service at route[j] without violating any downstream node's
    // latest time-window. compute_route_schedule is unchanged; this is a
    // separate, lighter computation used only by commit/min-insert decisions.
    // For routes that are infeasible under any schedule, individual entries
    // may be < node.earliest (callers treat that as "must commit now").
    std::vector<double>
    compute_latest_begin_times(const std::vector<int>& route) const;

    // Feasibility (delegates to slack_corrected by default)
    std::pair<bool, std::string> check_route_feasibility(
        const std::vector<int>& route) const;

    // Full feasibility check on entire solution
    std::pair<bool, std::string> is_feasible() const;

    // Lightweight check: no request appears twice in routes, and no request
    // is simultaneously in routes AND unassigned. Safe for dynamic mode
    // (does not require all n requests to be accounted for).
    std::pair<bool, std::string> check_request_uniqueness() const;

    // Objective
    double objective() const;
    double travel_cost() const;

    // Utilities
    std::map<int, int> request_to_route() const;
};
