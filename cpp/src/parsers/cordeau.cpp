#include "parsers/cordeau.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

DARPInstance parse_cordeau(const std::string& filepath) {
    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // Header: K n T Q L
    int K, n_nodes;
    double T;
    int Q;
    double L;
    fin >> K >> n_nodes >> T >> Q >> L;
    int num_requests = n_nodes / 2;

    // Parse node lines: i x y d q e l
    std::vector<Node> nodes;
    std::vector<std::pair<double, double>> coords;

    std::string line;
    std::getline(fin, line);  // consume rest of header line

    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        int id;
        double x, y, d;
        int q_val;
        double e, l_val;

        if (!(iss >> id >> x >> y >> d >> q_val >> e >> l_val)) {
            continue;  // skip malformed/empty lines
        }

        Node node;
        node.id = id;
        node.x = x;
        node.y = y;
        node.service_duration = d;
        node.demand = q_val;
        node.earliest = e;
        node.latest = l_val;

        nodes.push_back(node);
        coords.push_back({x, y});
    }

    // Add destination depot (2n+1) as copy of origin depot (0)
    Node dest_depot;
    dest_depot.id = 2 * num_requests + 1;
    dest_depot.x = nodes[0].x;
    dest_depot.y = nodes[0].y;
    dest_depot.service_duration = 0;
    dest_depot.demand = 0;
    dest_depot.earliest = nodes[0].earliest;
    dest_depot.latest = nodes[0].latest;

    nodes.push_back(dest_depot);
    coords.push_back(coords[0]);

    int total = 2 * num_requests + 2;
    if (static_cast<int>(nodes.size()) != total) {
        throw std::runtime_error(
            "Expected " + std::to_string(total) + " nodes, got " +
            std::to_string(nodes.size()));
    }

    // Compute Euclidean distance matrix
    std::vector<std::vector<double>> travel_time(total, std::vector<double>(total, 0.0));
    for (int i = 0; i < total; i++) {
        for (int j = i + 1; j < total; j++) {
            double dx = coords[i].first - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            double dist = std::sqrt(dx * dx + dy * dy);
            travel_time[i][j] = dist;
            travel_time[j][i] = dist;
        }
    }

    DARPInstance inst;
    inst.nodes = std::move(nodes);
    inst.num_requests = num_requests;
    inst.num_vehicles = K;
    inst.vehicle_capacity = Q;
    inst.max_ride_time = L;
    inst.max_route_duration = T;
    inst.travel_time = std::move(travel_time);

    return inst;
}
