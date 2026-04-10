#include "operators/destroy/ride_time_critical.h"
#include "operators/repair/repair_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

static constexpr double EPS = 1e-6;
static constexpr int RANDOMIZATION_P = 3;

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

DARPSolution ride_time_critical_removal(const DARPSolution& sol, std::mt19937& rng) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    DARPSolution destroyed = sol.copy();
    const auto& inst = *destroyed.instance;
    int n = inst.num_requests;

    // Compute ride-time ratio for each assigned request
    std::vector<std::pair<double, int>> ratios;

    for (const auto& route : destroyed.routes) {
        auto sched = destroyed.compute_route_schedule(route);
        for (const auto& [req, rt] : sched.ride_time) {
            if (!destroyed.unassigned.count(req) && !destroyed.committed.count(req)) {
                double ratio = rt / inst.max_ride_time_for(req);
                ratios.push_back({ratio, req});
            }
        }
    }

    if (ratios.empty()) return destroyed;

    std::sort(ratios.begin(), ratios.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Number to remove: 10-40% (static), 5-15% (dynamic)
    int total_assigned = static_cast<int>(ratios.size());
    bool dynamic = !destroyed.committed.empty();
    double lo_pct = dynamic ? 0.10 : 0.10;
    double hi_pct = dynamic ? 0.25 : 0.40;
    int lo = std::max(1, static_cast<int>(total_assigned * lo_pct));
    int hi = std::max(lo + 1, static_cast<int>(total_assigned * hi_pct) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), total_assigned);

    // Build req_route index once and pick q requests in a single pass
    auto req_route = build_req_to_route(destroyed);

    for (int removed = 0; removed < q && !ratios.empty(); removed++) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int idx = static_cast<int>(
            std::floor(std::pow(u(rng), RANDOMIZATION_P) * ratios.size()));
        idx = std::min(idx, static_cast<int>(ratios.size()) - 1);

        int req = ratios[idx].second;
        if (req_route[req] >= 0) {
            remove_request(destroyed, req, req_route[req]);
        }
        ratios.erase(ratios.begin() + idx);
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    DynamicStats::destroy_rtcritical_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    DynamicStats::destroy_rtcritical_count++;
    return destroyed;
}
