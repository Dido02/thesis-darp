#include "instance.h"
#include "parsers/cordeau.h"
#include "solver.h"

#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --instance <path> [options]\n"
              << "\nOptions:\n"
              << "  --iterations N       Number of ALNS iterations (default: 1000)\n"
              << "  --seed S             Random seed (default: 42)\n"
              << "  --feasibility MODE   Feasibility check: slack, full, o1 (default: slack)\n"
              << "  --destroy OPS        Comma-separated destroy operators (default: random,worst)\n"
              << "                       Available: random, worst\n"
              << "  --repair OPS         Comma-separated repair operators (default: greedy)\n"
              << "                       Available: greedy\n"
              << "  --selection METHOD   Operator selection: random, roulette (default: random)\n"
              << "  --sa-start TEMP      SA start temperature (default: 100)\n"
              << "  --sa-end TEMP        SA end temperature (default: 0.01)\n";
}

int main(int argc, char* argv[]) {
    std::string instance_path;
    int iterations = 1000;
    int seed = 42;
    std::string feasibility_mode = "o1";
    std::string destroy_config = "random,worst,shaw,rtcritical,lowslack";
    std::string repair_config = "greedy,regret2,regret3";
    std::string selection_config = "roulette";
    double sa_start = 100.0;
    double sa_end = 0.01;

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
            if (feasibility_mode != "slack" && feasibility_mode != "full" && feasibility_mode != "o1") {
                std::cerr << "Unknown feasibility mode: " << feasibility_mode
                          << " (valid: slack, full, o1)\n";
                return 1;
            }
        } else if (arg == "--destroy" && i + 1 < argc) {
            destroy_config = argv[++i];
        } else if (arg == "--repair" && i + 1 < argc) {
            repair_config = argv[++i];
        } else if (arg == "--selection" && i + 1 < argc) {
            selection_config = argv[++i];
        } else if (arg == "--sa-start" && i + 1 < argc) {
            sa_start = std::stod(argv[++i]);
        } else if (arg == "--sa-end" && i + 1 < argc) {
            sa_end = std::stod(argv[++i]);
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
    SolverResult result = solve(inst, iterations, sa_start, sa_end, seed, true,
                                feasibility_mode, destroy_config, repair_config,
                                selection_config);

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