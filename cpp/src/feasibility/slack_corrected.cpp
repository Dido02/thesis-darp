#include "feasibility/slack_corrected.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

static constexpr double EPS = 1e-6;

std::pair<bool, std::string> check_slack_corrected(
    const std::vector<int>& route, const DARPInstance& inst) {

    const auto& nodes = inst.nodes;
    int n = inst.num_requests;
    const auto& tt = inst.travel_time;
    int Q = inst.vehicle_capacity;

    int size = static_cast<int>(route.size());
    if (size <= 2) return {true, ""};

    // --- Step 1: AEAP forward pass ---
    std::vector<double> arrival(size, 0.0);
    std::vector<double> begin(size, 0.0);
    std::vector<double> depart(size, 0.0);
    std::vector<int> load(size, 0);

    begin[0] = nodes[route[0]].earliest;
    depart[0] = begin[0] + nodes[route[0]].service_duration;

    for (int i = 1; i < size; i++) {
        int prev_id = route[i - 1];
        int curr_id = route[i];
        const auto& node = nodes[curr_id];

        arrival[i] = depart[i - 1] + tt[prev_id][curr_id];
        begin[i] = std::max(arrival[i], node.earliest);
        depart[i] = begin[i] + node.service_duration;
        load[i] = load[i - 1] + node.demand;

        // Time window
        if (begin[i] > node.latest + EPS) {
            std::ostringstream oss;
            oss << "Time window violated at node " << curr_id
                << ": begin=" << begin[i] << " > latest=" << node.latest;
            return {false, oss.str()};
        }

        // Capacity
        if (load[i] < 0 || load[i] > Q) {
            std::ostringstream oss;
            oss << "Capacity violated at node " << curr_id
                << ": load=" << load[i] << ", capacity=" << Q;
            return {false, oss.str()};
        }
    }

    // --- Compute forward time slack on AEAP (backward pass) ---
    double slack_val = nodes[route[size - 1]].latest - begin[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = begin[i + 1] - arrival[i + 1];
        slack_val = std::min(
            nodes[route[i]].latest - begin[i],
            slack_val + wait_next);
    }

    // --- Step 2: Apply depot delay, re-run forward pass ---
    if (slack_val > EPS) {
        begin[0] += slack_val;
        depart[0] = begin[0] + nodes[route[0]].service_duration;
        for (int i = 1; i < size; i++) {
            int prev_id = route[i - 1];
            int curr_id = route[i];
            const auto& node = nodes[curr_id];
            arrival[i] = depart[i - 1] + tt[prev_id][curr_id];
            begin[i] = std::max(arrival[i], node.earliest);
            depart[i] = begin[i] + node.service_duration;
        }
    }

    // --- Compute slack + cumulative wait on Step 2 schedule ---
    std::vector<double> slack(size, 0.0);
    slack[size - 1] = nodes[route[size - 1]].latest - begin[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = begin[i + 1] - arrival[i + 1];
        slack[i] = std::min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next);
    }

    std::vector<double> cum_wait(size, 0.0);
    for (int i = 1; i < size; i++) {
        cum_wait[i] = cum_wait[i - 1] + (begin[i] - arrival[i]);
    }

    // --- Ride time + precedence with per-pickup correction ---
    std::unordered_map<int, int> pickup_pos;
    for (int i = 0; i < size; i++) {
        int node_id = route[i];
        if (node_id >= 1 && node_id <= n) {
            pickup_pos[node_id] = i;
        } else if (node_id >= n + 1 && node_id <= 2 * n) {
            int req = node_id - n;
            auto it = pickup_pos.find(req);
            if (it == pickup_pos.end()) {
                return {false, "Precedence violated: delivery " +
                    std::to_string(node_id) + " before pickup " +
                    std::to_string(req)};
            }
            int p = it->second;
            double ride_time = begin[i] - depart[p];
            double w_pd = cum_wait[i] - cum_wait[p];
            double best_ride_time = ride_time - std::min(slack[p], w_pd);
            if (best_ride_time > inst.max_ride_time + EPS) {
                std::ostringstream oss;
                oss << "Ride time violated for request " << req
                    << ": ride_time=" << best_ride_time
                    << " > max=" << inst.max_ride_time;
                return {false, oss.str()};
            }
        }
    }

    // --- Route duration on Step 2 schedule ---
    double route_dur = begin[size - 1] - depart[0];
    if (route_dur > inst.max_route_duration + EPS) {
        std::ostringstream oss;
        oss << "Route duration violated: duration=" << route_dur
            << " > max=" << inst.max_route_duration;
        return {false, oss.str()};
    }

    return {true, ""};
}
