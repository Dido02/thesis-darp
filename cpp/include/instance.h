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

    int num_nodes() const { return static_cast<int>(nodes.size()); }

    const Node& pickup_node(int request) const { return nodes[request]; }
    const Node& delivery_node(int request) const { return nodes[num_requests + request]; }
    const Node& origin_depot() const { return nodes[0]; }
    const Node& dest_depot() const { return nodes.back(); }
};
