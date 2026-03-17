/**
 * Compare C++ feasibility checkers on pr03 initial solution routes.
 * Mirrors scripts/compare_feasibility.py for cross-language verification.
 */

#include "instance.h"
#include "initial_solution.h"
#include "parsers/cordeau.h"
#include "feasibility/slack_corrected.h"
#include "feasibility/full_optimizer.h"

#include <iostream>

int main() {
    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");
    DARPSolution sol = build_initial_solution(inst);

    std::cout << "Instance: pr03  (n=" << inst.num_requests
              << ", K=" << inst.num_vehicles << ")\n";
    std::cout << "Initial solution: obj=" << sol.objective()
              << ", unassigned=" << sol.unassigned.size() << "\n\n";

    int num_routes = std::min(3, static_cast<int>(sol.routes.size()));
    for (int k = 0; k < num_routes; k++) {
        const auto& route = sol.routes[k];
        std::cout << "Route " << k << " (size=" << route.size() << "): [";
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << route[i];
        }
        std::cout << "]\n";

        // slack_corrected
        {
            auto [feasible, reason] = check_slack_corrected(route, inst);
            std::cout << "  slack_corrected       -> "
                      << (feasible ? "FEASIBLE" : "INFEASIBLE: " + reason) << "\n";
        }

        // full_optimizer
        {
            auto [feasible, reason] = check_full_optimizer(route, inst);
            std::cout << "  full_optimizer        -> "
                      << (feasible ? "FEASIBLE" : "INFEASIBLE: " + reason) << "\n";
        }

        std::cout << "\n";
    }

    return 0;
}
