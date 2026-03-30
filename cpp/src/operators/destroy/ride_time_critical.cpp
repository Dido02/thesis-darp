#include "operators/destroy/ride_time_critical.h"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr double EPS = 1e-6;
static constexpr int RANDOMIZATION_P = 3;

static void remove_request(DARPSolution& sol, int req) {
    int pickup_id = req;
    int delivery_id = req + sol.instance->num_requests;
    for (auto& route : sol.routes) {
        auto it_p = std::find(route.begin(), route.end(), pickup_id);
        if (it_p != route.end()) {
            route.erase(it_p);
            auto it_d = std::find(route.begin(), route.end(), delivery_id);
            if (it_d != route.end()) route.erase(it_d);
            sol.unassigned.insert(req);
            return;
        }
    }
}

DARPSolution ride_time_critical_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    const auto& inst = *destroyed.instance;
    int n = inst.num_requests;

    // Compute ride-time ratio for each assigned request
    std::vector<std::pair<double, int>> ratios;  // (ratio, req)

    for (const auto& route : destroyed.routes) {
        auto sched = destroyed.compute_route_schedule(route);
        for (const auto& [req, rt] : sched.ride_time) {
            if (!destroyed.unassigned.count(req)) {
                double ratio = rt / inst.max_ride_time_for(req);
                ratios.push_back({ratio, req});
            }
        }
    }

    if (ratios.empty()) return destroyed;

    // Sort descending by ratio (highest utilization first)
    std::sort(ratios.begin(), ratios.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Number to remove: 10-40%
    int total_assigned = static_cast<int>(ratios.size());
    int lo = std::max(1, static_cast<int>(total_assigned * 0.10));
    int hi = std::max(lo + 1, static_cast<int>(total_assigned * 0.40) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), total_assigned);

    for (int removed = 0; removed < q; removed++) {
        // Recompute ratios for remaining requests
        std::vector<std::pair<double, int>> current_ratios;
        for (const auto& route : destroyed.routes) {
            auto sched = destroyed.compute_route_schedule(route);
            for (const auto& [req, rt] : sched.ride_time) {
                if (!destroyed.unassigned.count(req)) {
                    current_ratios.push_back({rt / inst.max_ride_time_for(req), req});
                }
            }
        }
        if (current_ratios.empty()) break;

        std::sort(current_ratios.begin(), current_ratios.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Randomized selection biased toward highest ratio
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int idx = static_cast<int>(
            std::floor(std::pow(u(rng), RANDOMIZATION_P) * current_ratios.size()));
        idx = std::min(idx, static_cast<int>(current_ratios.size()) - 1);

        remove_request(destroyed, current_ratios[idx].second);
    }

    return destroyed;
}