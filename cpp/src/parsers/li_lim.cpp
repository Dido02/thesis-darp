#include "parsers/li_lim.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

DARPInstance parse_li_lim(const std::string& filepath, double max_ride_time) {
    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // Header: K Q speed
    // (speed is typically 1 and unused — travel time = Euclidean distance)
    int K, Q;
    double speed;
    fin >> K >> Q >> speed;

    // --- First pass: read all raw nodes ---
    struct RawNode {
        int id;
        double x, y;
        int demand;
        double earliest, latest;
        double service_time;
        int pickup_idx;    // >0 means "my pickup sibling is node pickup_idx" (I am a delivery)
        int delivery_idx;  // >0 means "my delivery sibling is node delivery_idx" (I am a pickup)
    };

    std::vector<RawNode> raw_nodes;
    std::string line;
    std::getline(fin, line); // consume rest of header line

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        RawNode rn;
        if (!(iss >> rn.id >> rn.x >> rn.y >> rn.demand
                  >> rn.earliest >> rn.latest >> rn.service_time
                  >> rn.pickup_idx >> rn.delivery_idx)) {
            continue; // skip malformed lines
        }
        raw_nodes.push_back(rn);
    }

    if (raw_nodes.empty()) {
        throw std::runtime_error("No nodes parsed from file: " + filepath);
    }

    // --- Identify depot and requests ---
    // Node 0 in the file is the depot.
    // A node is a pickup  if delivery_idx > 0 (it tells you where to deliver)
    // A node is a delivery if pickup_idx  > 0 (it tells you where pickup was)
    // Both 0 → depot

    // Collect pickup-delivery pairs: pickup_file_id → delivery_file_id
    std::vector<std::pair<int, int>> requests; // (pickup_file_id, delivery_file_id)
    std::unordered_map<int, bool> seen_as_pickup;

    for (const auto& rn : raw_nodes) {
        if (rn.id == 0) continue; // skip depot
        if (rn.delivery_idx > 0) {
            // This node is a pickup, its delivery is at delivery_idx
            requests.push_back({rn.id, rn.delivery_idx});
            seen_as_pickup[rn.id] = true;
        }
    }

    int num_requests = static_cast<int>(requests.size());

    // --- Build mapping: file_id → internal index ---
    // Internal layout: 0 = origin depot, 1..n = pickups, n+1..2n = deliveries, 2n+1 = dest depot
    // Request r (0-based) has pickup at index r+1, delivery at index num_requests+r+1

    std::unordered_map<int, int> file_id_to_internal;
    file_id_to_internal[0] = 0; // origin depot

    for (int r = 0; r < num_requests; r++) {
        int pickup_file_id  = requests[r].first;
        int delivery_file_id = requests[r].second;
        file_id_to_internal[pickup_file_id]  = r + 1;                    // pickup slot
        file_id_to_internal[delivery_file_id] = num_requests + r + 1;    // delivery slot
    }

    // --- Build raw_nodes lookup by file id ---
    std::unordered_map<int, const RawNode*> raw_by_id;
    for (const auto& rn : raw_nodes) {
        raw_by_id[rn.id] = &rn;
    }

    // --- Construct DARPInstance nodes ---
    int total = 2 * num_requests + 2; // +2 for origin and destination depots
    std::vector<Node> nodes(total);
    std::vector<std::pair<double, double>> coords(total);

    auto set_node = [&](int internal_idx, const RawNode& rn) {
        Node& n = nodes[internal_idx];
        n.id = internal_idx;
        n.x = rn.x;
        n.y = rn.y;
        // Use the sign of demand from the file, or override if needed
        n.demand = rn.demand;
        n.earliest = rn.earliest;
        n.latest = rn.latest;
        n.service_duration = rn.service_time;
        coords[internal_idx] = {rn.x, rn.y};
    };

    // Origin depot (index 0)
    const RawNode& depot = *raw_by_id.at(0);
    set_node(0, depot);
    nodes[0].demand = 0;

    // Pickups and deliveries
    for (int r = 0; r < num_requests; r++) {
        int pickup_file_id  = requests[r].first;
        int delivery_file_id = requests[r].second;

        int pickup_internal  = r + 1;
        int delivery_internal = num_requests + r + 1;

        set_node(pickup_internal, *raw_by_id.at(pickup_file_id));
        set_node(delivery_internal, *raw_by_id.at(delivery_file_id));

        // Ensure correct demand signs regardless of file encoding
        // Li & Lim uses load quantities; for DARP we use +/- to indicate pickup/delivery
        nodes[pickup_internal].demand = std::abs(nodes[pickup_internal].demand);
        nodes[delivery_internal].demand = -std::abs(nodes[delivery_internal].demand);
    }

    // Destination depot (index 2n+1) — copy of origin depot
    int dest_idx = 2 * num_requests + 1;
    nodes[dest_idx].id = dest_idx;
    nodes[dest_idx].x = depot.x;
    nodes[dest_idx].y = depot.y;
    nodes[dest_idx].demand = 0;
    nodes[dest_idx].earliest = depot.earliest;
    nodes[dest_idx].latest = depot.latest;
    nodes[dest_idx].service_duration = 0;
    coords[dest_idx] = {depot.x, depot.y};

    // --- Compute Euclidean distance matrix ---
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

    // --- Assemble instance ---
    DARPInstance inst;
    inst.nodes = std::move(nodes);
    inst.num_requests = num_requests;
    inst.num_vehicles = K;
    inst.vehicle_capacity = Q;
    inst.max_ride_time = max_ride_time;              // Li & Lim doesn't specify this; caller provides
    inst.max_route_duration = depot.latest;           // planning horizon from depot time window
    inst.travel_time = std::move(travel_time);

    return inst;
}