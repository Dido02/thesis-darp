#include "instance.h"
#include "feasibility.h"
#include "parsers/cordeau.h"

#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Test Feasibility ===\n";

    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");

    bool ok = true;

    // Routes from Python initial solution reference
    std::vector<std::vector<int>> test_routes = {
        {0, 23, 2, 95, 1, 74, 73, 52, 26, 124, 28, 19, 100, 98, 91, 17, 22, 25, 24, 94, 89, 97, 96, 3, 75, 145},
        {0, 5, 4, 18, 77, 9, 20, 90, 76, 92, 71, 81, 143, 62, 44, 134, 116, 7, 8, 80, 53, 125, 79, 145},
        {0, 33, 11, 14, 6, 78, 105, 83, 21, 86, 93, 12, 10, 13, 82, 85, 84, 27, 99, 38, 15, 16, 87, 88, 110, 145},
    };

    // Expected Python results:
    // Route 0: slack=true,  full=true
    // Route 1: slack=true,  full=false (ride time req 4)
    // Route 2: slack=true,  full=true
    struct Expected {
        bool slack_feasible;
        bool full_feasible;
    };
    Expected expected[] = {
        {true, true},
        {true, false},
        {true, true},
    };

    for (int k = 0; k < 3; k++) {
        std::cout << "\nRoute " << k << " (size=" << test_routes[k].size() << "):\n";

        auto [sf, sr] = check_slack_corrected(test_routes[k], inst);
        std::cout << "  slack: feasible=" << (sf ? "true" : "false")
                  << ", reason=\"" << sr << "\"\n";

        auto [ff, fr] = check_full_optimizer(test_routes[k], inst);
        std::cout << "  full:  feasible=" << (ff ? "true" : "false")
                  << ", reason=\"" << fr << "\"\n";

        if (sf != expected[k].slack_feasible) {
            std::cerr << "  FAIL: slack feasibility mismatch for route " << k
                      << ": got " << sf << ", expected " << expected[k].slack_feasible << "\n";
            ok = false;
        }
        if (ff != expected[k].full_feasible) {
            std::cerr << "  FAIL: full feasibility mismatch for route " << k
                      << ": got " << ff << ", expected " << expected[k].full_feasible << "\n";
            ok = false;
        }
    }

    // Test empty route
    {
        std::vector<int> empty_route = {0, 145};
        auto [sf, sr] = check_slack_corrected(empty_route, inst);
        auto [ff, fr] = check_full_optimizer(empty_route, inst);
        if (!sf || !ff) {
            std::cerr << "  FAIL: empty route should be feasible\n";
            ok = false;
        } else {
            std::cout << "\nEmpty route: both feasible OK\n";
        }
    }

    if (ok) {
        std::cout << "\nAll feasibility tests PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSome feasibility tests FAILED\n";
        return 1;
    }
}
