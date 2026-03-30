#include "feasibility/full_optimizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

static constexpr double EPS = 1e-6;
static constexpr double INF = 1e18;

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

    // --- Step 2: Delay depot departure by forward time slack ---
    // (Standard PDPTW FTS for Step 2 — no ride-time term needed here
    //  because we're only delaying the depot, which helps ride times)
    std::vector<double> slack(size, 0.0);
    slack[size - 1] = nodes[route[size - 1]].latest - begin[size - 1];
    for (int i = size - 2; i >= 0; i--) {
        double wait_next = begin[i + 1] - arrival[i + 1];
        slack[i] = std::min(
            nodes[route[i]].latest - begin[i],
            slack[i + 1] + wait_next);
    }

    if (slack[0] > EPS) {
        begin[0] += slack[0];
        depart[0] = begin[0] + nodes[route[0]].service_duration;

        for (int i = 1; i < size; i++) {
            int prev = route[i - 1], curr = route[i];
            const auto& node = nodes[curr];
            arrival[i] = depart[i - 1] + tt[prev][curr];
            begin[i] = std::max(arrival[i], node.earliest);
            depart[i] = begin[i] + node.service_duration;
        }
    }

    // --- Build pickup/delivery position maps ---
    // pickup_pos[req] = position of pickup node req in route
    // delivery_pos[req] = position of delivery node req+n in route
    std::unordered_map<int, int> pickup_pos, delivery_pos;
    for (int i = 0; i < size; i++) {
        int nid = route[i];
        if (nid >= 1 && nid <= n) pickup_pos[nid] = i;
        else if (nid >= n + 1 && nid <= 2 * n) delivery_pos[nid - n] = i;
    }

    // --- Helper: compute DARP-adapted FTS at position i ---
    // Equation (5) from Gschwind & Drexl (2016):
    //   F_i = min over j=i..r-1 of { w_{i,j} + min(b_{vj} - tau_j, S_{ij}) }
    // where S_{ij} = L - (tau_j - tau_{j+})  if v_j is a delivery and j+ < i
    //             = +inf otherwise
    // Also includes route duration via the dummy request (depot -> end depot).
    auto compute_darp_fts = [&](int i) -> double {
        double cumwait = 0.0;
        double fts = INF;
        for (int j = i; j < size; j++) {
            if (j > i) {
                double wait_j = begin[j] - arrival[j];
                cumwait += wait_j;
            }

            // Time window slack at j
            double tw_slack = nodes[route[j]].latest - begin[j];

            // Ride time slack S_{ij}: check if node at j is a delivery
            // whose pickup is before position i
            double s_ij = INF;
            int nid = route[j];
            if (nid >= n + 1 && nid <= 2 * n) {
                // This is a delivery node for request (nid - n)
                int req = nid - n;
                auto pit = pickup_pos.find(req);
                if (pit != pickup_pos.end() && pit->second < i) {
                    // Pickup is before position i, so delaying i affects ride time
                    double current_ride = begin[j] - depart[pit->second];
                    s_ij = inst.max_ride_time_for(req) - current_ride;
                }
            }
            // Also handle route duration as dummy request:
            // end depot (position size-1) is "delivery" of depot "request"
            // pickup at position 0. If i > 0, route duration constrains.
            if (j == size - 1 && i > 0) {
                double current_dur = begin[size - 1] - depart[0];
                double dur_slack = inst.max_route_duration - current_dur;
                s_ij = std::min(s_ij, dur_slack);
            }

            double bound = cumwait + std::min(tw_slack, s_ij);
            fts = std::min(fts, bound);
        }
        return std::max(0.0, fts);
    };

    // --- Step 3: Reduce ride times by delaying pickups ---
    // Uses the DARP-adapted FTS to avoid breaking other requests' ride times
    for (int pass = 0; pass < size; pass++) {
        bool improved = false;
        for (int i = 0; i < size; i++) {
            int node_id = route[i];
            if (!(node_id >= 1 && node_id <= n)) continue;

            int req = node_id;
            auto dit = delivery_pos.find(req);
            if (dit == delivery_pos.end()) continue;
            int del_pos = dit->second;

            double current_rt = begin[del_pos] - depart[i];
            if (current_rt <= inst.max_ride_time_for(req) + EPS) continue;

            double needed = current_rt - inst.max_ride_time_for(req);

            // Compute DARP-adapted FTS at position i
            double fi = compute_darp_fts(i);

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
    for (int i = 0; i < size; i++) {
        int node_id = route[i];
        if (node_id >= n + 1 && node_id <= 2 * n) {
            int req = node_id - n;
            auto pit = pickup_pos.find(req);
            if (pit == pickup_pos.end()) {
                return {false, "Precedence violated: delivery " +
                    std::to_string(node_id) + " before pickup " +
                    std::to_string(req)};
            }
            double ride_time = begin[i] - depart[pit->second];
            if (ride_time > inst.max_ride_time_for(req) + EPS) {
                std::ostringstream oss;
                oss << "Ride time violated for request " << req
                    << ": ride_time=" << ride_time
                    << " > max=" << inst.max_ride_time_for(req);
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