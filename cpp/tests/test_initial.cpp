#include "instance.h"
#include "initial_solution.h"
#include "parsers/cordeau.h"

#include <cmath>
#include <iostream>

int main() {
    std::cout << "=== Test Initial Solution ===\n";

    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");

    std::cout << "Building initial solution...\n";
    DARPSolution sol = build_initial_solution(inst);

    int unassigned = static_cast<int>(sol.unassigned.size());
    double obj = sol.objective();
    double travel = sol.travel_cost();

    std::cout << "  Unassigned: " << unassigned << "\n";
    std::cout << "  Objective:  " << obj << "\n";
    std::cout << "  Travel:     " << travel << "\n";

    // Python reference: Unassigned=0, Objective≈852.07
    bool ok = true;

    if (unassigned != 0) {
        std::cerr << "FAIL: Expected 0 unassigned, got " << unassigned << "\n";
        ok = false;
    }

    double expected_obj = 852.066319;
    if (std::abs(obj - expected_obj) > 0.01) {
        std::cerr << "FAIL: Expected objective ~" << expected_obj
                  << ", got " << obj << " (diff=" << std::abs(obj - expected_obj) << ")\n";
        ok = false;
    }

    // Print routes
    for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
        const auto& route = sol.routes[k];
        std::cout << "  R" << k << ": [";
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << route[i];
        }
        std::cout << "]\n";
    }

    if (ok) {
        std::cout << "\nAll initial solution tests PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSome initial solution tests FAILED\n";
        return 1;
    }
}
