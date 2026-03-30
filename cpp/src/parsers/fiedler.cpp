#include "parsers/fiedler.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

DARPInstance parse_fiedler(const std::string& filepath) {
    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    // Header: K n T Q L (same as Cordeau)
    int K, n_nodes;
    double T;
    int Q;
    double L;
    fin >> K >> n_nodes >> T >> Q >> L;
    int num_requests = n_nodes / 2;

    // Count total nodes: 2n+2 base nodes + K vehicle start nodes (if present)
    // We'll read all node lines first, then determine from sections
    std::vector<Node> nodes;
    std::string line;
    std::getline(fin, line);  // consume rest of header line

    // Read all node lines until we hit a section marker
    while (std::getline(fin, line)) {
        if (line.find("PER_REQUEST") != std::string::npos ||
            line.find("VEHICLE_START") != std::string::npos ||
            line.find("TRAVEL_TIMES") != std::string::npos) {
            break;
        }

        std::istringstream iss(line);
        int id;
        double x, y, d;
        int q_val;
        double e, l_val;

        if (!(iss >> id >> x >> y >> d >> q_val >> e >> l_val)) {
            continue;  // skip empty lines
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
    }

    int total = static_cast<int>(nodes.size());

    // Parse remaining sections
    std::vector<double> per_request_rt;
    std::vector<int> vehicle_start_nodes;
    std::vector<std::vector<double>> travel_time(total, std::vector<double>(total, 0.0));

    // Process the line that broke the node-reading loop, then continue
    do {
        if (line.find("PER_REQUEST_RIDE_TIMES") != std::string::npos) {
            if (!std::getline(fin, line)) break;
            std::istringstream iss(line);
            per_request_rt.resize(num_requests + 1, L);
            for (int i = 1; i <= num_requests; i++) {
                if (!(iss >> per_request_rt[i])) {
                    throw std::runtime_error("Failed to parse per-request ride time " + std::to_string(i));
                }
            }
        } else if (line.find("VEHICLE_START_NODES") != std::string::npos) {
            if (!std::getline(fin, line)) break;
            std::istringstream iss(line);
            int node_id;
            while (iss >> node_id) {
                vehicle_start_nodes.push_back(node_id);
            }
        } else if (line.find("TRAVEL_TIMES") != std::string::npos) {
            for (int i = 0; i < total; i++) {
                if (!std::getline(fin, line)) {
                    throw std::runtime_error("Unexpected end of file reading travel time row " + std::to_string(i));
                }
                std::istringstream iss(line);
                for (int j = 0; j < total; j++) {
                    if (!(iss >> travel_time[i][j])) {
                        throw std::runtime_error(
                            "Failed to parse travel time at [" + std::to_string(i) + "][" + std::to_string(j) + "]");
                    }
                }
            }
        }
    } while (std::getline(fin, line));

    DARPInstance inst;
    inst.nodes = std::move(nodes);
    inst.num_requests = num_requests;
    inst.num_vehicles = K;
    inst.vehicle_capacity = Q;
    inst.max_ride_time = L;
    inst.max_route_duration = T;
    inst.travel_time = std::move(travel_time);
    inst.max_ride_time_per_request = std::move(per_request_rt);
    inst.vehicle_start_nodes = std::move(vehicle_start_nodes);

    return inst;
}
