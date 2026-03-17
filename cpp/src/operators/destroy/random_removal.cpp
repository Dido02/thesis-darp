#include "operators/destroy/random_removal.h"

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: remove a request (pickup + delivery) from its route
// ---------------------------------------------------------------------------
static void remove_request(DARPSolution& sol, int req) {
    int pickup_id = req;
    int delivery_id = req + sol.instance->num_requests;

    for (auto& route : sol.routes) {
        auto it_p = std::find(route.begin(), route.end(), pickup_id);
        if (it_p != route.end()) {
            route.erase(it_p);
            auto it_d = std::find(route.begin(), route.end(), delivery_id);
            if (it_d != route.end()) {
                route.erase(it_d);
            }
            sol.unassigned.insert(req);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Random removal
// ---------------------------------------------------------------------------
DARPSolution random_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    int n = destroyed.instance->num_requests;

    // Collect assigned requests
    std::vector<int> assigned;
    for (int i = 1; i <= n; i++) {
        if (!destroyed.unassigned.count(i)) {
            assigned.push_back(i);
        }
    }
    if (assigned.empty()) return destroyed;

    // Number to remove: 10-40%
    int lo = std::max(1, static_cast<int>(assigned.size() * 0.10));
    int hi = std::max(lo + 1, static_cast<int>(assigned.size() * 0.40) + 1);
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    int q = std::min(dist(rng), static_cast<int>(assigned.size()));

    // Fisher-Yates partial shuffle to select q random elements
    for (int i = 0; i < q; i++) {
        std::uniform_int_distribution<int> swap_dist(i, static_cast<int>(assigned.size()) - 1);
        int j = swap_dist(rng);
        std::swap(assigned[i], assigned[j]);
    }

    for (int i = 0; i < q; i++) {
        remove_request(destroyed, assigned[i]);
    }

    return destroyed;
}
