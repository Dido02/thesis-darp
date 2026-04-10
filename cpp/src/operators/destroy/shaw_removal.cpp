#include "operators/destroy/shaw_removal.h"
#include "operators/repair/repair_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

// Shaw removal parameters (from Ropke & Pisinger 2006 / Gschwind & Drexl)
static constexpr double WEIGHT_DISTANCE = 9.0;
static constexpr double WEIGHT_TIME = 9.0;
static constexpr double WEIGHT_LOAD = 6.0;
static constexpr int RANDOMIZATION_P = 9;

// ---------------------------------------------------------------------------
// Helper: build request-to-route index
// ---------------------------------------------------------------------------
static std::vector<int> build_req_to_route(const DARPSolution& sol) {
    int n = sol.instance->num_requests;
    std::vector<int> req_route(n + 1, -1);
    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        for (int nid : sol.routes[r]) {
            if (nid >= 1 && nid <= n) req_route[nid] = r;
        }
    }
    return req_route;
}

// ---------------------------------------------------------------------------
// Helper: remove a request from a known route
// ---------------------------------------------------------------------------
static void remove_request(DARPSolution& sol, int req, int route_idx) {
    int pickup_id = req;
    int delivery_id = req + sol.instance->num_requests;
    auto& route = sol.routes[route_idx];

    auto it_p = std::find(route.begin(), route.end(), pickup_id);
    if (it_p != route.end()) route.erase(it_p);
    auto it_d = std::find(route.begin(), route.end(), delivery_id);
    if (it_d != route.end()) route.erase(it_d);
    sol.unassigned.insert(req);
}

// ---------------------------------------------------------------------------
// Helper: find the route index and pickup position for a request
// ---------------------------------------------------------------------------
struct RequestInfo {
    int route_idx;
    int pickup_pos;
    double pickup_time;   // departure time at pickup (from forward pass)
    int load_at_pickup;
};

static RequestInfo get_request_info(const DARPSolution& sol, int req,
                                    const std::vector<int>& req_route) {
    const auto& inst = *sol.instance;
    const auto& tt = inst.travel_time;
    const auto& nodes = inst.nodes;
    int pickup_id = req;

    int k = req_route[req];
    if (k < 0) return {-1, -1, 0.0, 0};

    const auto& route = sol.routes[k];
    double depart = nodes[route[0]].earliest + nodes[route[0]].service_duration;
    int load = 0;

    for (int i = 1; i < static_cast<int>(route.size()); i++) {
        double arrival = depart + tt[route[i - 1]][route[i]];
        double begin = std::max(arrival, nodes[route[i]].earliest);
        load += nodes[route[i]].demand;

        if (route[i] == pickup_id) {
            return {k, i, begin, load};
        }

        depart = begin + nodes[route[i]].service_duration;
    }
    return {-1, -1, 0.0, 0};
}

// ---------------------------------------------------------------------------
// Shaw removal
// ---------------------------------------------------------------------------
DARPSolution shaw_removal(const DARPSolution& sol, std::mt19937& rng) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    DARPSolution destroyed = sol.copy();
    const auto& inst = *destroyed.instance;
    const auto& nodes = inst.nodes;
    int n = inst.num_requests;

    // Collect assigned requests
    std::vector<int> assigned;
    for (int i = 1; i <= n; i++) {
        if (!destroyed.unassigned.count(i) && !destroyed.committed.count(i)) {
            assigned.push_back(i);
        }
    }
    if (assigned.empty()) return destroyed;

    // Number to remove: 10-40% (static), 5-15% (dynamic)
    bool dynamic = !destroyed.committed.empty();
    double lo_pct = dynamic ? 0.10 : 0.10;
    double hi_pct = dynamic ? 0.25 : 0.40;
    int lo = std::max(1, static_cast<int>(assigned.size() * lo_pct));
    int hi = std::max(lo + 1, static_cast<int>(assigned.size() * hi_pct) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), static_cast<int>(assigned.size()));

    // Normalize factors for relatedness (use instance max values)
    double max_dist = inst.max_travel_time();
    double max_time = inst.max_route_duration;
    int max_load = inst.vehicle_capacity;
    if (max_dist < 1e-9) max_dist = 1.0;
    if (max_time < 1e-9) max_time = 1.0;

    // Build request-to-route index once for efficient lookups
    auto req_route = build_req_to_route(destroyed);

    // Pick a random seed request
    std::uniform_int_distribution<int> seed_dist(0, static_cast<int>(assigned.size()) - 1);
    int seed_req = assigned[seed_dist(rng)];

    // Get info for seed request
    RequestInfo seed_info = get_request_info(destroyed, seed_req, req_route);

    // Compute relatedness to seed for all other assigned requests
    std::vector<std::pair<double, int>> relatedness;
    for (int req : assigned) {
        if (req == seed_req) continue;
        if (destroyed.unassigned.count(req)) continue;

        RequestInfo info = get_request_info(destroyed, req, req_route);
        if (info.route_idx < 0) continue;

        int pickup_i = seed_req;
        int pickup_j = req;

        // Distance between pickups
        double dist_val = inst.travel_time[pickup_i][pickup_j] / max_dist;

        // Time difference
        double time_val = std::abs(seed_info.pickup_time - info.pickup_time) / max_time;

        // Load difference
        double load_val = static_cast<double>(std::abs(seed_info.load_at_pickup - info.load_at_pickup))
                        / max_load;

        double rel = WEIGHT_DISTANCE * dist_val
                   + WEIGHT_TIME * time_val
                   + WEIGHT_LOAD * load_val;

        relatedness.push_back({rel, req});
    }

    // Sort by relatedness (ascending = most similar first)
    std::sort(relatedness.begin(), relatedness.end());

    // Remove seed request first
    if (req_route[seed_req] >= 0)
        remove_request(destroyed, seed_req, req_route[seed_req]);
    int removed = 1;

    // Remove most similar requests with randomization
    while (removed < q && !relatedness.empty()) {
        // Randomized selection: index = floor(random^p * size)
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int idx = static_cast<int>(
            std::floor(std::pow(u(rng), RANDOMIZATION_P) * relatedness.size()));
        idx = std::min(idx, static_cast<int>(relatedness.size()) - 1);

        int req = relatedness[idx].second;
        if (!destroyed.unassigned.count(req) && req_route[req] >= 0) {
            remove_request(destroyed, req, req_route[req]);
            removed++;
        }

        // Remove from list
        relatedness.erase(relatedness.begin() + idx);
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    DynamicStats::destroy_shaw_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    DynamicStats::destroy_shaw_count++;
    return destroyed;
}