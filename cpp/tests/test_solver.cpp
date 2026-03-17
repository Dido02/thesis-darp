#include "instance.h"
#include "feasibility.h"
#include "parsers/cordeau.h"
#include "solver.h"

#include <chrono>
#include <iostream>

int main() {
    std::cout << "=== Test Solver ===\n";

    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");

    auto t0 = std::chrono::high_resolution_clock::now();
    SolverResult result = solve(inst, 200, 100.0, 0.01, 42, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t1 - t0).count();

    double obj = result.best.objective();
    int unassigned = static_cast<int>(result.best.unassigned.size());

    std::cout << "\n--- Verification ---\n";
    std::cout << "  Objective:   " << obj << "\n";
    std::cout << "  Unassigned:  " << unassigned << "\n";
    std::cout << "  Total time:  " << total_time << "s\n";

    bool ok = true;

    // Check objective in reasonable range
    if (obj < 400 || obj > 1200) {
        std::cerr << "FAIL: Objective " << obj << " outside expected range [400, 1200]\n";
        ok = false;
    }

    // Check unassigned
    if (unassigned != 0) {
        std::cerr << "FAIL: Expected 0 unassigned, got " << unassigned << "\n";
        ok = false;
    }

    // Check all routes pass full_optimizer
    for (int k = 0; k < static_cast<int>(result.best.routes.size()); k++) {
        auto [feasible, reason] = check_full_optimizer(result.best.routes[k], inst);
        if (!feasible) {
            std::cerr << "FAIL: Route " << k << " infeasible (full): " << reason << "\n";
            ok = false;
        }
    }

    // Check time < 5s (though 200 iterations may vary)
    if (total_time > 10.0) {
        std::cerr << "WARN: Total time " << total_time << "s > 10s\n";
    }

    if (ok) {
        std::cout << "\nAll solver tests PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSome solver tests FAILED\n";
        return 1;
    }
}
