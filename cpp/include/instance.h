#pragma once

#include <vector>

struct Node {
    int id;
    double x, y;
    int demand;           // +1 pickup, -1 delivery, 0 depot
    double earliest;
    double latest;
    double service_duration;

    bool is_pickup() const { return demand > 0; }
    bool is_delivery() const { return demand < 0; }
    bool is_depot() const { return demand == 0; }
};

struct DARPInstance {
    std::vector<Node> nodes;
    int num_requests;
    int num_vehicles;
    int vehicle_capacity;
    double max_ride_time;
    double max_route_duration;
    std::vector<std::vector<double>> travel_time;  // (2n+2) x (2n+2)
    std::vector<double> max_ride_time_per_request;  // indexed 1..n, empty = use global
    std::vector<int> vehicle_start_nodes;  // size K node IDs for per-vehicle starts, empty = shared depot

    int num_nodes() const { return static_cast<int>(nodes.size()); }

    // First valid insertion position in a route (after depot and optional vehicle start)
    int route_start_offset() const { return vehicle_start_nodes.empty() ? 1 : 2; }

    // Minimum route size for an "empty" route (depot-depot or depot-vstart-depot)
    int empty_route_size() const { return vehicle_start_nodes.empty() ? 2 : 3; }

    // Per-request max ride time; falls back to global if vector is empty
    double max_ride_time_for(int request) const {
        if (!max_ride_time_per_request.empty())
            return max_ride_time_per_request[request];
        return max_ride_time;
    }

    const Node& pickup_node(int request) const { return nodes[request]; }
    const Node& delivery_node(int request) const { return nodes[num_requests + request]; }
    const Node& origin_depot() const { return nodes[0]; }
    const Node& dest_depot() const { return nodes.back(); }
};
