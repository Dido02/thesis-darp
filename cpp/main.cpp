#include "instance.h"
#include "parsers/cordeau.h"
#include "solver.h"

#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --instance <path> [--iterations N] [--seed S]"
              << " [--feasibility slack|full]\n";
}

int main(int argc, char* argv[]) {
    std::string instance_path;
    int iterations = 1000;
    int seed = 42;
    std::string feasibility_mode = "slack";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--instance" && i + 1 < argc) {
            instance_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--feasibility" && i + 1 < argc) {
            feasibility_mode = argv[++i];
            if (feasibility_mode != "slack" && feasibility_mode != "full") {
                std::cerr << "Unknown feasibility mode: " << feasibility_mode
                          << " (valid: slack, full)\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (instance_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Loading instance: " << instance_path << "\n";
    DARPInstance inst = parse_cordeau(instance_path);
    std::cout << "  n=" << inst.num_requests
              << ", K=" << inst.num_vehicles
              << ", Q=" << inst.vehicle_capacity
              << ", L=" << inst.max_ride_time
              << ", T=" << inst.max_route_duration << "\n";

    std::cout << "Feasibility mode: " << feasibility_mode << "\n";
    SolverResult result = solve(inst, iterations, 100.0, 0.01, seed, true, feasibility_mode);

    std::cout << "\nFinal objective: " << result.best.objective() << "\n";
    std::cout << "Travel cost:     " << result.best.travel_cost() << "\n";
    std::cout << "Unassigned:      " << result.best.unassigned.size() << "\n";
    std::cout << "Construction:    " << result.construction_time << "s\n";
    std::cout << "Solve time:      " << result.solve_time << "s\n";

    // Feasibility check
    auto [feasible, reason] = result.best.is_feasible();
    std::cout << "Feasible:        " << (feasible ? "yes" : "no") << "\n";
    if (!feasible) {
        std::cout << "Reason:          " << reason << "\n";
    }

    // Print routes
    std::cout << "\nRoutes:\n";
    for (int k = 0; k < static_cast<int>(result.best.routes.size()); k++) {
        const auto& route = result.best.routes[k];
        std::cout << "  R" << k << ": [";
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << route[i];
        }
        std::cout << "]\n";
    }

    return 0;
}
