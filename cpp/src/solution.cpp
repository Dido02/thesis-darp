#include "solution.h"
#include "feasibility.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

static constexpr double EPS = 1e-6;

DARPSolution::DARPSolution(const DARPInstance& inst)
    : instance(&inst) {
    int n = inst.num_requests;
    int depot_d = 2 * n + 1;

    routes.resize(inst.num_vehicles);
    for (int k = 0; k < inst.num_vehicles; k++) {
        if (!inst.vehicle_start_nodes.empty())
            routes[k] = {0, inst.vehicle_start_nodes[k], depot_d};
        else
            routes[k] = {0, depot_d};
    }
    for (int i = 1; i <= n; i++) {
        unassigned.insert(i);
    }
}

DARPSolution DARPSolution::copy() const {
    DARPSolution sol(*instance);
    sol.routes.clear();
    sol.routes.reserve(routes.size());
    for (const auto& r : routes) {
        sol.routes.push_back(r);
    }
    sol.unassigned = unassigned;
    sol.committed = committed;
    sol.min_insert_pos = min_insert_pos;
    sol.route_ready_time = route_ready_time;
    sol.route_ready_node = route_ready_node;
    sol.top_n_routes = top_n_routes;
    sol.feasibility_mode = feasibility_mode;
    sol.unassigned_penalty = unassigned_penalty;
    return sol;
}

RouteSchedule DARPSolution::compute_route_schedule(const std::vector<int>& route) const {
    const auto& inst = *instance;
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;
    int size = static_cast<int>(route.size());

    if (size <= inst.empty_route_size()) {
        RouteSchedule empty;
        empty.arrival_time.resize(size, 0.0);
        empty.begin_service.resize(size, 0.0);
        empty.departure_time.resize(size, 0.0);
        empty.load.resize(size, 0);
        return empty;
    }

    // --- Step 1: Forward pass (earliest feasible times) ---
    std::vector<double> arrival(size, 0.0);
    std::vector<double> begin(size, 0.0);
    std::vector<double> depart(size, 0.0);
    std::vector<int> load(size, 0);

    arrival[0] = 0.0;
    begin[0] = nodes[route[0]].earliest;
    depart[0] = begin[0] + nodes[route[0]].service_duration;
    load[0] = 0;

    for (int i = 1; i < size; i++) {
        int prev = route[i - 1], curr = route[i];
        const auto& node = nodes[curr];
        arrival[i] = depart[i - 1] + tt[prev][curr];
        begin[i] = std::max(arrival[i], node.earliest);
        depart[i] = begin[i] + node.service_duration;
        load[i] = load[i - 1] + node.demand;
    }

    // --- Step 2: Optimize ride times by delaying departures ---
    std::vector<double> slack(size, 0.0);
    slack[size - 1] = nodes[route[size - 1]].latest - begin[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = begin[i + 1] - arrival[i + 1];
        slack[i] = std::min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next);
    }

    if (slack[0] > EPS) {
        double shift = slack[0];
        begin[0] += shift;
        depart[0] = begin[0] + nodes[route[0]].service_duration;

        for (int i = 1; i < size; i++) {
            int prev = route[i - 1], curr = route[i];
            const auto& node = nodes[curr];
            arrival[i] = depart[i - 1] + tt[prev][curr];
            begin[i] = std::max(arrival[i], node.earliest);
            depart[i] = begin[i] + node.service_duration;
        }
    }

    // --- Step 3: Further reduce ride times by adding waiting at pickups ---
    for (int pass = 0; pass < size; pass++) {
        bool improved = false;
        for (int i = 0; i < size; i++) {
            int node_id = route[i];
            if (!(node_id >= 1 && node_id <= n)) continue;

            int req = node_id;
            int del_id = req + n;
            int del_pos = -1;
            for (int j = i + 1; j < size; j++) {
                if (route[j] == del_id) {
                    del_pos = j;
                    break;
                }
            }
            if (del_pos == -1) continue;

            double current_rt = begin[del_pos] - depart[i];
            if (current_rt <= inst.max_ride_time_for(req) + EPS) continue;

            double needed = current_rt - inst.max_ride_time_for(req);

            // Compute forward time slack at position i
            double fi = nodes[route[size - 1]].latest - begin[size - 1];
            for (int j = size - 2; j >= i; j--) {
                double wait_next = begin[j + 1] - arrival[j + 1];
                fi = std::min(
                    nodes[route[j]].latest - begin[j],
                    fi + wait_next);
            }

            double delay = std::min(needed, fi);
            if (delay < EPS) continue;

            improved = true;
            begin[i] += delay;
            depart[i] = begin[i] + nodes[route[i]].service_duration;
            for (int j = i + 1; j < size; j++) {
                int prev_j = route[j - 1], curr_j = route[j];
                arrival[j] = depart[j - 1] + tt[prev_j][curr_j];
                begin[j] = std::max(arrival[j], nodes[curr_j].earliest);
                depart[j] = begin[j] + nodes[curr_j].service_duration;
            }
        }
        if (!improved) break;
    }

    // Compute ride times
    std::map<int, double> ride_time;
    std::map<int, double> pickup_depart;
    for (int i = 0; i < size; i++) {
        int node_id = route[i];
        if (node_id >= 1 && node_id <= n) {
            pickup_depart[node_id] = depart[i];
        } else if (node_id >= n + 1 && node_id <= 2 * n) {
            int req = node_id - n;
            auto it = pickup_depart.find(req);
            if (it != pickup_depart.end()) {
                ride_time[req] = begin[i] - it->second;
            }
        }
    }

    return RouteSchedule{
        std::move(arrival),
        std::move(begin),
        std::move(depart),
        std::move(load),
        std::move(ride_time)
    };
}

std::pair<bool, std::string> DARPSolution::check_route_feasibility(
    const std::vector<int>& route) const {
    if (feasibility_mode == "full") {
        return check_full_optimizer(route, *instance);
    }
    return check_slack_corrected(route, *instance);
}

std::pair<bool, std::string> DARPSolution::is_feasible() const {
    const auto& inst = *instance;
    int n = inst.num_requests;

    if (!unassigned.empty()) {
        std::ostringstream oss;
        oss << "Unassigned requests: {";
        bool first = true;
        for (int r : unassigned) {
            if (!first) oss << ", ";
            oss << r;
            first = false;
        }
        oss << "}";
        return {false, oss.str()};
    }

    // Each request appears exactly once
    std::set<int> seen;
    for (const auto& route : routes) {
        for (int node_id : route) {
            if (node_id >= 1 && node_id <= n) {
                if (seen.count(node_id)) {
                    return {false, "Request " + std::to_string(node_id) +
                        " appears in multiple routes"};
                }
                seen.insert(node_id);
            }
        }
    }

    std::set<int> expected;
    for (int i = 1; i <= n; i++) expected.insert(i);
    if (seen != expected) {
        return {false, "Missing requests"};
    }

    // Check each route
    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        const auto& route = routes[k];
        auto sched = compute_route_schedule(route);

        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            const auto& node = inst.nodes[route[i]];
            if (sched.begin_service[i] > node.latest + EPS) {
                std::ostringstream oss;
                oss << "Route " << k << ": Time window violated at node " << route[i]
                    << ": begin=" << sched.begin_service[i]
                    << " > latest=" << node.latest;
                return {false, oss.str()};
            }
            if (sched.load[i] < 0 || sched.load[i] > inst.vehicle_capacity) {
                std::ostringstream oss;
                oss << "Route " << k << ": Capacity violated at node " << route[i];
                return {false, oss.str()};
            }
        }

        for (const auto& [req, rt] : sched.ride_time) {
            if (rt > inst.max_ride_time_for(req) + EPS) {
                std::ostringstream oss;
                oss << "Route " << k << ": Ride time violated for request " << req
                    << ": ride_time=" << rt << " > max=" << inst.max_ride_time_for(req);
                return {false, oss.str()};
            }
        }

        double route_dur = sched.begin_service.back() - sched.departure_time[0];
        if (route_dur > inst.max_route_duration + EPS) {
            std::ostringstream oss;
            oss << "Route " << k << ": Route duration violated: duration=" << route_dur
                << " > max=" << inst.max_route_duration;
            return {false, oss.str()};
        }

        // Precedence check
        std::set<int> seen_pickups;
        for (int node_id : route) {
            if (node_id >= 1 && node_id <= n) {
                seen_pickups.insert(node_id);
            } else if (node_id >= n + 1 && node_id <= 2 * n) {
                int req = node_id - n;
                if (!seen_pickups.count(req)) {
                    return {false, "Route " + std::to_string(k) +
                        ": Precedence violated: delivery " +
                        std::to_string(node_id) + " before pickup " +
                        std::to_string(req)};
                }
            }
        }
    }

    return {true, ""};
}

double DARPSolution::objective() const {
    if (cached_obj_ >= 0.0) return cached_obj_;
    const auto& tt = instance->travel_time;
    double total = 0.0;
    for (const auto& route : routes) {
        for (int i = 0; i < static_cast<int>(route.size()) - 1; i++) {
            total += tt[route[i]][route[i + 1]];
        }
    }
    total += static_cast<double>(unassigned.size()) * unassigned_penalty;
    cached_obj_ = total;
    return total;
}

double DARPSolution::travel_cost() const {
    const auto& tt = instance->travel_time;
    double total = 0.0;
    for (const auto& route : routes) {
        for (int i = 0; i < static_cast<int>(route.size()) - 1; i++) {
            total += tt[route[i]][route[i + 1]];
        }
    }
    return total;
}

std::pair<bool, std::string> DARPSolution::check_request_uniqueness() const {
    int n = instance->num_requests;
    std::set<int> seen_pickups, seen_deliveries;
    for (size_t k = 0; k < routes.size(); ++k) {
        for (int nid : routes[k]) {
            if (nid >= 1 && nid <= n) {
                if (!seen_pickups.insert(nid).second) {
                    return {false, "Pickup " + std::to_string(nid)
                        + " appears twice (route " + std::to_string(k) + ")"};
                }
            } else if (nid >= n + 1 && nid <= 2 * n) {
                if (!seen_deliveries.insert(nid).second) {
                    return {false, "Delivery " + std::to_string(nid)
                        + " appears twice (route " + std::to_string(k) + ")"};
                }
            }
        }
    }
    // Request must not be simultaneously in a route AND in unassigned
    for (int req : unassigned) {
        if (seen_pickups.count(req)) {
            return {false, "Request " + std::to_string(req)
                + " is in route AND unassigned"};
        }
    }
    return {true, ""};
}

std::map<int, int> DARPSolution::request_to_route() const {
    int n = instance->num_requests;
    std::map<int, int> mapping;
    for (int k = 0; k < static_cast<int>(routes.size()); k++) {
        for (int node_id : routes[k]) {
            if (node_id >= 1 && node_id <= n) {
                mapping[node_id] = k;
            }
        }
    }
    return mapping;
}
