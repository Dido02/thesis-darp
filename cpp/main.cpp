#include "instance.h"
#include "parsers/cordeau.h"
#include "parsers/fiedler.h"
#include "parsers/li_lim.h"
#include "solver.h"
#include "dynamic_solver.h"
#include "logging/alns_logger.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --instance <path> [options]\n"
              << "\nOptions:\n"
              << "  --iterations N       Number of ALNS iterations (default: 1000)\n"
              << "  --seed S             Random seed (default: 42)\n"
              << "  --feasibility MODE   Feasibility check: slack, full, o1 (default: o1)\n"
              << "  --destroy OPS        Comma-separated destroy operators\n"
              << "  --repair OPS         Comma-separated repair operators\n"
              << "  --selection METHOD   Operator selection: random, roulette, ml (default: roulette)\n"
              << "  --ml-model-dir DIR   Directory with feature_manifest.json + .onnx files (required for --selection ml)\n"
              << "  --sa-start TEMP      SA start temperature (default: 100)\n"
              << "  --sa-end TEMP        SA end temperature (default: 0.01)\n"
              << "  --time-limit SECS    Stop after this many seconds (default: 0 = use iterations)\n"
              << "  --penalty P          Unassigned request penalty (default: 1000)\n"
              << "  --init-sort MODE     Initial solution sort: slack, time, random, none (default: slack)\n"
              << "  --output-routes FILE Write route schedules to file for solution conversion\n"
              << "\nDynamic DARP options:\n"
              << "  --dynamic            Enable rolling horizon dynamic mode\n"
              << "  --epoch-length SECS  Epoch length in sim seconds (default: 30, time-based trigger)\n"
              << "  --epoch-requests N   Epoch trigger after N new requests (overrides --epoch-length)\n"
              << "  --epoch-budget-ms MS ALNS wall-clock budget per epoch in ms (default: 5000)\n"
              << "  --top-n-routes N   Limit repair to N nearest routes per request (default: 0 = all)\n"
              << "  --dynamic-stats    Print per-epoch performance stats\n";
}

// Write route schedules to file (shared by static and dynamic paths)
static void write_route_schedules(const DARPSolution& sol, const DARPInstance& inst,
                                   const std::string& path) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        std::cerr << "Cannot open output file: " << path << "\n";
        return;
    }
    fout << "SOLUTION\n";
    fout << "objective " << sol.objective() << "\n";
    fout << "travel_cost " << sol.travel_cost() << "\n";
    fout << "num_routes " << sol.routes.size() << "\n";
    fout << "num_requests " << inst.num_requests << "\n";
    fout << "unassigned " << sol.unassigned.size() << "\n";

    for (int k = 0; k < static_cast<int>(sol.routes.size()); k++) {
        const auto& route = sol.routes[k];
        auto sched = sol.compute_route_schedule(route);
        fout << "ROUTE " << k << "\n";
        fout << "size " << route.size() << "\n";
        for (int i = 0; i < static_cast<int>(route.size()); i++) {
            fout << route[i]
                 << " " << sched.arrival_time[i]
                 << " " << sched.begin_service[i]
                 << " " << sched.departure_time[i]
                 << " " << sched.load[i] << "\n";
        }
    }
    fout.close();
    std::cout << "\nRoute schedules written to: " << path << "\n";
}

int main(int argc, char* argv[]) {
    std::string instance_path;
    int iterations = 1000;
    int seed = 42;
    std::string feasibility_mode = "o1";
    std::string destroy_config = "random,worst,shaw,rtcritical,lowslack";
    std::string repair_config = "greedy,regret2,regret3,slack";
    std::string selection_config = "roulette";
    double time_limit = 0.0;
    double sa_start = 100.0;
    double sa_end = 0.01;
    double penalty = 1000.0;
    std::string init_sort = "slack";
    std::string output_routes_path;
    std::string ml_model_dir;

    // Dynamic mode options
    bool dynamic_mode = false;
    double epoch_length = 30.0;
    int epoch_requests = -1;  // -1 = not set, use time-based
    double epoch_budget_ms = 5000.0;
    int top_n_routes = 0;     // 0 = no limit
    bool dynamic_stats = false;

    // Logging options
    bool log_state = false;
    bool log_trials = false;
    int m_trials = 5;
    int m_trials_every = 50;
    std::string log_dir = "logs";

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
        } else if (arg == "--ml-model-dir" && i + 1 < argc) {
            ml_model_dir = argv[++i];
        } else if (arg == "--sa-start" && i + 1 < argc) {
            sa_start = std::stod(argv[++i]);
        } else if (arg == "--sa-end" && i + 1 < argc) {
            sa_end = std::stod(argv[++i]);
        } else if (arg == "--time-limit" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--penalty" && i + 1 < argc) {
            penalty = std::stod(argv[++i]);
        } else if (arg == "--init-sort" && i + 1 < argc) {
            init_sort = argv[++i];
        } else if (arg == "--output-routes" && i + 1 < argc) {
            output_routes_path = argv[++i];
        } else if (arg == "--dynamic") {
            dynamic_mode = true;
        } else if (arg == "--epoch-length" && i + 1 < argc) {
            epoch_length = std::stod(argv[++i]);
        } else if (arg == "--epoch-requests" && i + 1 < argc) {
            epoch_requests = std::stoi(argv[++i]);
        } else if (arg == "--epoch-budget-ms" && i + 1 < argc) {
            epoch_budget_ms = std::stod(argv[++i]);
        } else if (arg == "--top-n-routes" && i + 1 < argc) {
            top_n_routes = std::stoi(argv[++i]);
        } else if (arg == "--dynamic-stats") {
            dynamic_stats = true;
        } else if (arg == "--log-state") {
            log_state = true;
        } else if (arg == "--log-trials") {
            log_trials = true;
        } else if (arg == "--m-trials" && i + 1 < argc) {
            m_trials = std::stoi(argv[++i]);
        } else if (arg == "--m-trials-every" && i + 1 < argc) {
            m_trials_every = std::stoi(argv[++i]);
        } else if (arg == "--log-dir" && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (instance_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Early validation for --selection ml
    if (selection_config == "ml") {
#ifndef DARP_ENABLE_ML
        std::cerr << "Error: --selection ml is not available: this binary was built "
                  << "without ML support. Rebuild with -DENABLE_ML_SELECTION=ON.\n";
        return 1;
#else
        if (ml_model_dir.empty()) {
            std::cerr << "Error: --selection ml requires --ml-model-dir <path> "
                      << "pointing to a directory with feature_manifest.json "
                      << "and {d}_{r}.onnx files.\n";
            return 1;
        }
#endif
    }

    std::cout << "Loading instance: " << instance_path << "\n";
    bool is_li_lim = instance_path.find("li_lim") != std::string::npos;
    bool is_fiedler = instance_path.find(".fiedler") != std::string::npos
                   || instance_path.find("fiedler") != std::string::npos;
    DARPInstance inst = is_fiedler ? parse_fiedler(instance_path)
                      : is_li_lim ? parse_li_lim(instance_path)
                                  : parse_cordeau(instance_path);
    std::cout << "  n=" << inst.num_requests
              << ", K=" << inst.num_vehicles
              << ", Q=" << inst.vehicle_capacity
              << ", L=" << inst.max_ride_time
              << ", T=" << inst.max_route_duration << "\n";

    // Warm lazy cache to prevent contamination from logging code paths
    inst.max_travel_time();

    // --log-trials no longer implies --log-state (can log trials independently)

    // Build CLI args string for logging
    std::string cli_args;
    for (int i = 0; i < argc; i++) {
        if (i > 0) cli_args += ' ';
        cli_args += argv[i];
    }

    // Construct logger (or nullptr if disabled)
    std::unique_ptr<ALNSLogger> logger_ptr;
    if (log_state || log_trials) {
        ALNSLogConfig lc;
        lc.log_state = log_state;
        lc.log_trials = log_trials;
        lc.m_trials = m_trials;
        lc.m_trials_every = m_trials_every;
        lc.output_dir = log_dir;
        lc.instance_path = instance_path;
        lc.cli_args = cli_args;
        lc.seed = seed;
        lc.dynamic = dynamic_mode;

        // Build filename
        std::string stem = std::filesystem::path(instance_path).stem().string();
        std::string mode_suffix;
        if (log_trials && log_state) {
            mode_suffix = "_t" + std::to_string(m_trials) + "e" + std::to_string(m_trials_every) + "_full";
        } else if (log_trials) {
            mode_suffix = "_t" + std::to_string(m_trials) + "e" + std::to_string(m_trials_every);
        } else {
            mode_suffix = "_state";
        }
        std::string filename;
        if (dynamic_mode) {
            filename = stem + "_s" + std::to_string(seed) + mode_suffix + "_dynamic.jsonl";
        } else {
            filename = stem + "_s" + std::to_string(seed) + "_i" + std::to_string(iterations) + mode_suffix + ".jsonl";
        }
        logger_ptr = std::make_unique<ALNSLogger>(lc, filename);
    }
    ALNSLogger* logger = logger_ptr.get();

    try {
    if (dynamic_mode) {
        // ---- Dynamic DARP mode ----
        EpochConfig ec;
        if (epoch_requests > 0) {
            ec.trigger = EpochConfig::REQUEST_COUNT;
            ec.epoch_request_count = epoch_requests;
        } else {
            ec.trigger = EpochConfig::TIME_BASED;
            ec.epoch_length = epoch_length;
        }
        ec.epoch_budget_ms = epoch_budget_ms;

        SolverParams sp;
        // In dynamic mode, time budget controls epoch length — use high iteration cap
        sp.max_iterations = 1000000;
        sp.sa_start = sa_start;
        sp.sa_end = sa_end;
        sp.seed = seed;
        sp.feasibility_mode = feasibility_mode;
        sp.destroy_config = destroy_config;
        sp.repair_config = repair_config;
        sp.selection_config = selection_config;
        sp.unassigned_penalty = penalty;
        sp.top_n_routes = top_n_routes;
        sp.dynamic_stats = dynamic_stats;
        sp.logger = logger;
        sp.ml_model_dir = ml_model_dir;

        DynamicSolver dynamic_solver(inst, ec, sp, true);
        DynamicResult result = dynamic_solver.run();

        // Write route schedules if requested
        if (!output_routes_path.empty()) {
            write_route_schedules(result.final_solution, inst, output_routes_path);
        }

    } else {
        // ---- Static DARP mode (unchanged) ----
        std::cout << "Feasibility mode: " << feasibility_mode << "\n";
        SolverResult result = solve(inst, iterations, sa_start, sa_end, seed, true,
                                    feasibility_mode, destroy_config, repair_config,
                                    selection_config, time_limit, penalty, init_sort,
                                    logger, ml_model_dir);

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

        // Write route schedules to file if requested
        if (!output_routes_path.empty()) {
            write_route_schedules(result.best, inst, output_routes_path);
        }
    }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
