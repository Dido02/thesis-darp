#include "operators/destroy/entire_route_removal.h"

#include <algorithm>
#include <vector>

DARPSolution entire_route_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    int n = destroyed.instance->num_requests;

    // Find non-empty routes (more than just depot-depot)
    std::vector<int> non_empty;
    for (int r = 0; r < static_cast<int>(destroyed.routes.size()); r++) {
        if (destroyed.routes[r].size() > 2) {
            non_empty.push_back(r);
        }
    }
    if (non_empty.empty()) return destroyed;

    // Remove 1 or 2 routes
    int num_to_remove = (non_empty.size() >= 2) ?
        std::uniform_int_distribution<int>(1, 2)(rng) : 1;

    // Shuffle and pick
    for (int i = 0; i < num_to_remove && i < static_cast<int>(non_empty.size()); i++) {
        std::uniform_int_distribution<int> dist(i, static_cast<int>(non_empty.size()) - 1);
        std::swap(non_empty[i], non_empty[dist(rng)]);
    }

    for (int i = 0; i < num_to_remove; i++) {
        int r = non_empty[i];
        auto& route = destroyed.routes[r];
        // Collect all requests in this route
        std::vector<int> reqs;
        for (int nid : route) {
            if (nid >= 1 && nid <= n) {
                reqs.push_back(nid);
            }
        }
        // Clear route to depot-depot
        route = {0, 2 * n + 1};
        // Mark as unassigned
        for (int req : reqs) {
            destroyed.unassigned.insert(req);
        }
    }

    return destroyed;
}