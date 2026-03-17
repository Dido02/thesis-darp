#include "instance.h"
#include "parsers/cordeau.h"

#include <cmath>
#include <iostream>

int main() {
    std::cout << "=== Test Parser ===\n";

    DARPInstance inst = parse_cordeau("../../data/cordeau/pr03.txt");

    // Verify header
    bool ok = true;

    auto check = [&](const char* name, auto got, auto expected) {
        if (got != expected) {
            std::cerr << "FAIL: " << name << " = " << got
                      << ", expected " << expected << "\n";
            ok = false;
        } else {
            std::cout << "  " << name << " = " << got << " OK\n";
        }
    };

    check("num_requests", inst.num_requests, 72);
    check("num_vehicles", inst.num_vehicles, 7);
    check("vehicle_capacity", inst.vehicle_capacity, 6);
    check("max_ride_time", inst.max_ride_time, 90.0);
    check("max_route_duration", inst.max_route_duration, 480.0);
    check("num_nodes", inst.num_nodes(), 146);

    // Verify first 5 nodes
    struct Expected {
        int id, demand;
        double earliest, latest, service;
    };
    Expected exp[] = {
        {0,  0, 0.0, 1440.0, 0.0},
        {1,  1, 0.0, 1440.0, 10.0},
        {2,  1, 0.0, 1440.0, 10.0},
        {3,  1, 0.0, 1440.0, 10.0},
        {4,  1, 0.0, 1440.0, 10.0},
    };
    for (int i = 0; i < 5; i++) {
        const auto& nd = inst.nodes[i];
        std::cout << "  Node " << nd.id << ": demand=" << nd.demand
                  << ", tw=[" << nd.earliest << "," << nd.latest
                  << "], service=" << nd.service_duration << "\n";
        check(("node[" + std::to_string(i) + "].id").c_str(), nd.id, exp[i].id);
        check(("node[" + std::to_string(i) + "].demand").c_str(), nd.demand, exp[i].demand);
    }

    // Verify distances
    auto check_dist = [&](const char* name, double got, double expected) {
        if (std::abs(got - expected) > 1e-4) {
            std::cerr << "FAIL: " << name << " = " << got
                      << ", expected " << expected << "\n";
            ok = false;
        } else {
            std::cout << "  " << name << " = " << got << " OK\n";
        }
    };

    check_dist("tt[0][1]", inst.travel_time[0][1], 4.483985);
    check_dist("tt[1][73]", inst.travel_time[1][73], 2.544259);

    // Verify dest depot
    const auto& dest = inst.dest_depot();
    check("dest_depot.id", dest.id, 145);
    check("dest_depot.demand", dest.demand, 0);

    if (ok) {
        std::cout << "\nAll parser tests PASSED\n";
        return 0;
    } else {
        std::cerr << "\nSome parser tests FAILED\n";
        return 1;
    }
}
