#include "operators/destroy/entire_route_removal.h"

#include <algorithm>
#include <vector>

DARPSolution entire_route_removal(const DARPSolution& sol, std::mt19937& rng) {
    DARPSolution destroyed = sol.copy();
    const auto& inst = *destroyed.instance;
    int n = inst.num_requests;

    // Find non-empty routes (more than just depot-depot or depot-vstart-depot)
    std::vector<int> non_empty;
    int min_route = inst.empty_route_size();
    for (int r = 0; r < static_cast<int>(destroyed.routes.size()); r++) {
        if (static_cast<int>(destroyed.routes[r].size()) > min_route) {
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
        // Collect non-committed requests to remove
        std::vector<int> reqs_to_remove;
        for (int nid : route) {
            if (nid >= 1 && nid <= n && !destroyed.committed.count(nid)) {
                reqs_to_remove.push_back(nid);
            }
        }
        // Rebuild route keeping depots, vehicle start, and committed nodes
        std::vector<int> new_route;
        for (int nid : route) {
            bool is_pickup = (nid >= 1 && nid <= n);
            bool is_delivery = (nid >= n + 1 && nid <= 2 * n);
            int req = is_pickup ? nid : (is_delivery ? nid - n : 0);
            bool is_committed_node = (is_pickup || is_delivery) && destroyed.committed.count(req);
            if (!is_pickup && !is_delivery) {
                // depot or vehicle start — always keep
                new_route.push_back(nid);
            } else if (is_committed_node) {
                new_route.push_back(nid);
            }
            // else: non-committed request node — remove
        }
        route = new_route;
        // Mark removed requests as unassigned
        for (int req : reqs_to_remove) {
            destroyed.unassigned.insert(req);
        }
    }

    return destroyed;
}