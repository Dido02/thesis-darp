#include "operators/destroy/random_removal.h"
#include "operators/repair/repair_utils.h"

#include <algorithm>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: build request-to-route index (vector indexed by request ID, O(1) lookup)
// ---------------------------------------------------------------------------
static std::vector<int> build_req_to_route(const DARPSolution& sol) {
    int n = sol.instance->num_requests;
    std::vector<int> req_route(n + 1, -1);  // req_route[req] = route index
    for (int r = 0; r < static_cast<int>(sol.routes.size()); r++) {
        for (int nid : sol.routes[r]) {
            if (nid >= 1 && nid <= n) {
                req_route[nid] = r;
            }
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
// Random removal
// ---------------------------------------------------------------------------
DARPSolution random_removal(const DARPSolution& sol, std::mt19937& rng) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    DARPSolution destroyed = sol.copy();
    int n = destroyed.instance->num_requests;

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

    // Build request-to-route index once
    auto req_route = build_req_to_route(destroyed);

    // Fisher-Yates partial shuffle to select q random elements
    for (int i = 0; i < q; i++) {
        std::uniform_int_distribution<int> swap_dist(i, static_cast<int>(assigned.size()) - 1);
        int j = swap_dist(rng);
        std::swap(assigned[i], assigned[j]);
    }

    for (int i = 0; i < q; i++) {
        int req = assigned[i];
        if (req_route[req] >= 0) {
            remove_request(destroyed, req, req_route[req]);
        }
    }

    auto _t1 = std::chrono::high_resolution_clock::now();
    DynamicStats::destroy_random_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    DynamicStats::destroy_random_count++;
    return destroyed;
}
