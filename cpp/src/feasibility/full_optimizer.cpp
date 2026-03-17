#include "feasibility/full_optimizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

static constexpr double EPS = 1e-6;

std::pair<bool, std::string> check_full_optimizer(
    const std::vector<int>& route, const DARPInstance& inst) {

    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    const auto& tt = inst.travel_time;
    int Q = inst.vehicle_capacity;

    int size = static_cast<int>(route.size());
    if (size <= 2) return {true, ""};

    // --- Step 1: Forward pass (earliest feasible times) ---
    std::vector<double> arrival(size, 0.0);
    std::vector<double> begin(size, 0.0);
    std::vector<double> depart(size, 0.0);
    std::vector<int> load(size, 0);

    begin[0] = nodes[route[0]].earliest;
    depart[0] = begin[0] + nodes[route[0]].service_duration;

    for (int i = 1; i < size; i++) {
        int prev = route[i - 1], curr = route[i];
        const auto& node = nodes[curr];
        arrival[i] = depart[i - 1] + tt[prev][curr];
        begin[i] = std::max(arrival[i], node.earliest);
        depart[i] = begin[i] + node.service_duration;
        load[i] = load[i - 1] + node.demand;
    }

    // --- Step 2: Optimize ride times by delaying departures ---
    // Compute forward time slack
    std::vector<double> slack(size, 0.0);
    slack[size - 1] = nodes[route[size - 1]].latest - begin[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = begin[i + 1] - arrival[i + 1];
        slack[i] = std::min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next);
    }

    // Apply: delay depot departure by slack[0]
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
            if (current_rt <= inst.max_ride_time + EPS) continue;

            double needed = current_rt - inst.max_ride_time;

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

    // --- Constraint checks on optimized schedule ---

    // Time windows
    for (int i = 0; i < size; i++) {
        const auto& node = nodes[route[i]];
        if (begin[i] > node.latest + EPS) {
            std::ostringstream oss;
            oss << "Time window violated at node " << route[i]
                << ": begin=" << begin[i] << " > latest=" << node.latest;
            return {false, oss.str()};
        }
    }

    // Capacity
    for (int i = 0; i < size; i++) {
        if (load[i] < 0 || load[i] > Q) {
            std::ostringstream oss;
            oss << "Capacity violated at node " << route[i]
                << ": load=" << load[i] << ", capacity=" << Q;
            return {false, oss.str()};
        }
    }

    // Precedence + ride time
    std::unordered_map<int, double> pickup_depart;
    for (int i = 0; i < size; i++) {
        int node_id = route[i];
        if (node_id >= 1 && node_id <= n) {
            pickup_depart[node_id] = depart[i];
        } else if (node_id >= n + 1 && node_id <= 2 * n) {
            int req = node_id - n;
            auto it = pickup_depart.find(req);
            if (it == pickup_depart.end()) {
                return {false, "Precedence violated: delivery " +
                    std::to_string(node_id) + " before pickup " +
                    std::to_string(req)};
            }
            double ride_time = begin[i] - it->second;
            if (ride_time > inst.max_ride_time + EPS) {
                std::ostringstream oss;
                oss << "Ride time violated for request " << req
                    << ": ride_time=" << ride_time
                    << " > max=" << inst.max_ride_time;
                return {false, oss.str()};
            }
        }
    }

    // Route duration
    double route_dur = begin[size - 1] - depart[0];
    if (route_dur > inst.max_route_duration + EPS) {
        std::ostringstream oss;
        oss << "Route duration violated: duration=" << route_dur
            << " > max=" << inst.max_route_duration;
        return {false, oss.str()};
    }

    return {true, ""};
}
