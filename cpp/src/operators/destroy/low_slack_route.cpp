#include "operators/destroy/low_slack_route.h"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr double EPS = 1e-6;

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

DARPSolution low_slack_route_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    const auto& inst = *destroyed.instance;
    const auto& nodes = inst.nodes;
    const auto& tt = inst.travel_time;
    int n = inst.num_requests;

    // Compute total slack for each route
    struct RouteSlack {
        int route_idx;
        double total_slack;
        int num_requests;
    };
    std::vector<RouteSlack> route_slacks;

    for (int k = 0; k < static_cast<int>(destroyed.routes.size()); k++) {
        const auto& route = destroyed.routes[k];
        int rlen = static_cast<int>(route.size());
        if (rlen <= 2) continue;

        // Forward pass
        std::vector<double> tau(rlen);
        tau[0] = nodes[route[0]].earliest;
        double depart = tau[0] + nodes[route[0]].service_duration;
        for (int i = 1; i < rlen; i++) {
            double arrival = depart + tt[route[i - 1]][route[i]];
            tau[i] = std::max(arrival, nodes[route[i]].earliest);
            depart = tau[i] + nodes[route[i]].service_duration;
        }

        // Compute forward time slack via backward recursion
        std::vector<double> arrival_arr(rlen, 0.0);
        depart = tau[0] + nodes[route[0]].service_duration;
        for (int i = 1; i < rlen; i++) {
            arrival_arr[i] = depart + tt[route[i - 1]][route[i]];
            depart = tau[i] + nodes[route[i]].service_duration;
        }

        std::vector<double> slack(rlen);
        slack[rlen - 1] = nodes[route[rlen - 1]].latest - tau[rlen - 1];
        for (int i = rlen - 2; i >= 0; i--) {
            double wait_next = tau[i + 1] - arrival_arr[i + 1];
            slack[i] = std::min(nodes[route[i]].latest - tau[i], slack[i + 1] + wait_next);
        }

        // Total forward time slack
        double total_slack = 0.0;
        for (int i = 0; i < rlen; i++) {
            total_slack += slack[i];
        }

        // Count requests in this route
        int req_count = 0;
        for (int nid : route) {
            if (nid >= 1 && nid <= n) req_count++;
        }

        if (req_count > 0) {
            route_slacks.push_back({k, total_slack / req_count, req_count});
        }
    }

    if (route_slacks.empty()) return destroyed;

    // Sort by slack per request (ascending = tightest first)
    std::sort(route_slacks.begin(), route_slacks.end(),
              [](const auto& a, const auto& b) { return a.total_slack < b.total_slack; });

    // Pick 1-2 of the tightest routes
    int num_routes_to_destroy = std::min(
        static_cast<int>(route_slacks.size()),
        std::uniform_int_distribution<int>(1, 2)(rng));

    // Number to remove total: 10-40% of all assigned
    int total_assigned = 0;
    for (int i = 1; i <= n; i++) {
        if (!destroyed.unassigned.count(i)) total_assigned++;
    }
    int lo = std::max(1, static_cast<int>(total_assigned * 0.10));
    int hi = std::max(lo + 1, static_cast<int>(total_assigned * 0.40) + 1);
    std::uniform_int_distribution<int> q_dist(lo, hi - 1);
    int q = std::min(q_dist(rng), total_assigned);

    int removed = 0;
    for (int ri = 0; ri < num_routes_to_destroy && removed < q; ri++) {
        int route_idx = route_slacks[ri].route_idx;
        const auto& route = destroyed.routes[route_idx];

        // Collect requests in this route
        std::vector<int> reqs_in_route;
        for (int nid : route) {
            if (nid >= 1 && nid <= n && !destroyed.unassigned.count(nid)) {
                reqs_in_route.push_back(nid);
            }
        }

        // Remove some or all requests from this route
        // Shuffle and remove up to remaining quota
        std::shuffle(reqs_in_route.begin(), reqs_in_route.end(), rng);
        int to_remove = std::min(static_cast<int>(reqs_in_route.size()), q - removed);

        for (int i = 0; i < to_remove; i++) {
            remove_request(destroyed, reqs_in_route[i]);
            removed++;
        }
    }

    return destroyed;
}