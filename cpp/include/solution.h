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
    static constexpr double UNASSIGNED_PENALTY = 1000.0;

    const DARPInstance* instance;  // non-owning pointer, always valid
    std::vector<std::vector<int>> routes;
    std::set<int> unassigned;
    std::string feasibility_mode = "slack";  // "slack" or "full"
    mutable double cached_obj_ = -1.0;

    explicit DARPSolution(const DARPInstance& inst);

    DARPSolution copy() const;

    // Scheduling
    RouteSchedule compute_route_schedule(const std::vector<int>& route) const;

    // Feasibility (delegates to slack_corrected by default)
    std::pair<bool, std::string> check_route_feasibility(
        const std::vector<int>& route) const;

    // Full feasibility check on entire solution
    std::pair<bool, std::string> is_feasible() const;

    // Objective
    double objective() const;
    double travel_cost() const;

    // Utilities
    std::map<int, int> request_to_route() const;
};
