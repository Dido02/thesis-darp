#include "logging/alns_logger.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <random>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

ALNSLogger::ALNSLogger(const ALNSLogConfig& config, const std::string& filename)
    : config_(config) {
    std::filesystem::create_directories(config_.output_dir);
    std::string path = config_.output_dir + "/" + filename;
    file_.open(path, std::ios::out | std::ios::trunc);
}

ALNSLogger::~ALNSLogger() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

// ---------------------------------------------------------------------------
// JSON formatting helpers
// ---------------------------------------------------------------------------

std::string ALNSLogger::fmt4(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

std::string ALNSLogger::fmt6(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

std::string ALNSLogger::json_int_array(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) s += ',';
        s += std::to_string(v[i]);
    }
    s += ']';
    return s;
}

std::string ALNSLogger::json_int_set(const std::set<int>& st) {
    std::string s = "[";
    bool first = true;
    for (int v : st) {
        if (!first) s += ',';
        s += std::to_string(v);
        first = false;
    }
    s += ']';
    return s;
}

std::string ALNSLogger::json_double_array4(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) s += ',';
        s += fmt4(v[i]);
    }
    s += ']';
    return s;
}

std::string ALNSLogger::json_double_array6(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) s += ',';
        s += fmt6(v[i]);
    }
    s += ']';
    return s;
}

std::string ALNSLogger::json_routes(const std::vector<std::vector<int>>& routes) {
    std::string s = "[";
    for (size_t r = 0; r < routes.size(); r++) {
        if (r > 0) s += ',';
        s += json_int_array(routes[r]);
    }
    s += ']';
    return s;
}

std::string ALNSLogger::json_string_array(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) s += ',';
        s += "\"" + v[i] + "\"";
    }
    s += ']';
    return s;
}

std::string ALNSLogger::escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 10);
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

void ALNSLogger::write_line(const std::string& json) {
    file_ << json << '\n';
    file_.flush();
}

// ---------------------------------------------------------------------------
// Header record
// ---------------------------------------------------------------------------

void ALNSLogger::write_header(const OperatorRegistry& registry,
                               int max_iterations,
                               double sa_start_temp, double sa_end_temp, double sa_step,
                               const std::string& feasibility_mode,
                               double unassigned_penalty,
                               const std::string& selection_config,
                               int num_requests, int num_vehicles) {
    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&tt));

    // Operator names
    std::vector<std::string> d_names, r_names;
    for (int i = 0; i < registry.num_destroy(); i++)
        d_names.push_back(registry.destroy_name(i));
    for (int i = 0; i < registry.num_repair(); i++)
        r_names.push_back(registry.repair_name(i));

    std::string log_mode = "state";
    if (config_.log_trials) log_mode = "state+trials";

    std::string json = "{\"type\":\"header\"";
    json += ",\"cli_args\":\"" + escape_json_string(config_.cli_args) + "\"";
    json += ",\"instance\":\"" + escape_json_string(config_.instance_path) + "\"";
    json += ",\"seed\":" + std::to_string(config_.seed);
    json += ",\"max_iterations\":" + std::to_string(max_iterations);
    json += ",\"sa_start_temp\":" + fmt6(sa_start_temp);
    json += ",\"sa_end_temp\":" + fmt6(sa_end_temp);
    json += ",\"sa_step\":" + fmt6(sa_step);
    json += ",\"destroy_ops\":" + json_string_array(d_names);
    json += ",\"repair_ops\":" + json_string_array(r_names);
    json += ",\"selection\":\"" + selection_config + "\"";
    json += ",\"feasibility_mode\":\"" + feasibility_mode + "\"";
    json += ",\"unassigned_penalty\":" + fmt4(unassigned_penalty);
    json += ",\"num_requests\":" + std::to_string(num_requests);
    json += ",\"num_vehicles\":" + std::to_string(num_vehicles);
    json += ",\"log_mode\":\"" + log_mode + "\"";
    json += ",\"m_trials\":" + std::to_string(config_.m_trials);
    json += ",\"m_trials_every\":" + std::to_string(config_.m_trials_every);
    json += ",\"dynamic\":" + std::string(config_.dynamic ? "true" : "false");
    json += ",\"timestamp\":\"" + std::string(ts) + "\"";
    json += "}";

    write_line(json);
}

// ---------------------------------------------------------------------------
// State record (every iteration)
// ---------------------------------------------------------------------------

void ALNSLogger::log_iteration(int iter, int d_idx, int r_idx,
                                const std::string& d_name, const std::string& r_name,
                                double obj_before, double obj_candidate, double obj_best,
                                double delta, double temperature, bool accept, int q,
                                const std::vector<std::vector<int>>& routes,
                                const std::set<int>& unassigned,
                                const std::set<int>& committed,
                                const std::vector<int>& min_insert_pos) {
    if (!config_.log_state) return;

    std::string json = "{\"type\":\"state\"";
    json += ",\"iter\":" + std::to_string(iter);
    json += ",\"d_idx\":" + std::to_string(d_idx);
    json += ",\"r_idx\":" + std::to_string(r_idx);
    json += ",\"d_name\":\"" + d_name + "\"";
    json += ",\"r_name\":\"" + r_name + "\"";
    json += ",\"obj_before\":" + fmt4(obj_before);
    json += ",\"obj_candidate\":" + fmt4(obj_candidate);
    json += ",\"obj_best\":" + fmt4(obj_best);
    json += ",\"delta\":" + fmt4(delta);
    json += ",\"temperature\":" + fmt6(temperature);
    json += ",\"accept\":" + std::string(accept ? "true" : "false");
    json += ",\"q\":" + std::to_string(q);
    json += ",\"routes\":" + json_routes(routes);
    json += ",\"unassigned\":" + json_int_set(unassigned);
    json += ",\"committed\":" + json_int_set(committed);
    json += ",\"min_insert_pos\":" + json_int_array(min_insert_pos);
    json += ",\"pending\":[]";
    json += ",\"rejected\":[]";
    json += "}";

    write_line(json);
}

// ---------------------------------------------------------------------------
// Weights record
// ---------------------------------------------------------------------------

void ALNSLogger::log_weights(int iter, const OperatorRegistry& registry) {
    if (!config_.log_state) return;

    std::string json = "{\"type\":\"weights\"";
    json += ",\"iter\":" + std::to_string(iter);
    json += ",\"d_weights\":" + json_double_array6(registry.get_destroy_weights());
    json += ",\"r_weights\":" + json_double_array6(registry.get_repair_weights());
    json += "}";

    write_line(json);
}

// ---------------------------------------------------------------------------
// SA calibration record
// ---------------------------------------------------------------------------

void ALNSLogger::log_sa_calibrated(int iter, double sa_step, int estimated_total) {
    if (!config_.log_state) return;

    std::string json = "{\"type\":\"sa_calibrated\"";
    json += ",\"iter\":" + std::to_string(iter);
    json += ",\"sa_step\":" + fmt6(sa_step);
    json += ",\"estimated_total\":" + std::to_string(estimated_total);
    json += "}";

    write_line(json);
}

// ---------------------------------------------------------------------------
// Epoch start record (dynamic mode)
// ---------------------------------------------------------------------------

void ALNSLogger::log_epoch_start(int epoch_id, double sim_clock,
                                  const std::vector<std::vector<int>>& routes,
                                  const std::set<int>& unassigned,
                                  const std::set<int>& committed,
                                  const std::vector<int>& min_insert_pos,
                                  const std::set<int>& pending,
                                  const std::set<int>& rejected,
                                  double obj) {
    std::string json = "{\"type\":\"epoch_start\"";
    json += ",\"epoch_id\":" + std::to_string(epoch_id);
    json += ",\"sim_clock\":" + fmt4(sim_clock);
    json += ",\"routes\":" + json_routes(routes);
    json += ",\"unassigned\":" + json_int_set(unassigned);
    json += ",\"committed\":" + json_int_set(committed);
    json += ",\"min_insert_pos\":" + json_int_array(min_insert_pos);
    json += ",\"pending\":" + json_int_set(pending);
    json += ",\"rejected\":" + json_int_set(rejected);
    json += ",\"obj\":" + fmt4(obj);
    json += "}";

    write_line(json);
}

// ---------------------------------------------------------------------------
// Trials
// ---------------------------------------------------------------------------

bool ALNSLogger::should_run_trials(int iter) const {
    return config_.log_trials &&
           (config_.m_trials_every == 0 || iter % config_.m_trials_every == 0);
}

void ALNSLogger::run_and_log_trials(int iter,
                                     const DARPSolution& current,
                                     double current_obj, double temperature,
                                     const OperatorRegistry& registry,
                                     uint64_t trial_base_seed) {
    int nd = registry.num_destroy();
    int nr = registry.num_repair();
    int M = config_.m_trials;

    std::string json = "{\"type\":\"trials\"";
    json += ",\"iter\":" + std::to_string(iter);
    json += ",\"obj_before\":" + fmt4(current_obj);
    json += ",\"temperature\":" + fmt6(temperature);
    json += ",\"results\":[";

    bool first_pair = true;
    for (int di = 0; di < nd; di++) {
        for (int ri = 0; ri < nr; ri++) {
            if (!first_pair) json += ',';
            first_pair = false;

            json += "{\"d_idx\":" + std::to_string(di);
            json += ",\"r_idx\":" + std::to_string(ri);
            json += ",\"deltas\":[";

            const DestroyOp& destroy_op = registry.get_destroy(di);
            const RepairOp& repair_op = registry.get_repair(ri);

            for (int m = 0; m < M; m++) {
                if (m > 0) json += ',';

                uint64_t pseed = trial_base_seed * 100003ULL
                    + static_cast<uint64_t>(di * nr + ri) * 1009ULL
                    + static_cast<uint64_t>(m);
                std::mt19937 trial_rng(
                    static_cast<uint32_t>(pseed & 0xFFFFFFFFULL));

                DARPSolution trial_copy = current.copy();
                DARPSolution destroyed = destroy_op(trial_copy, trial_rng);
                DARPSolution candidate = repair_op(destroyed, trial_rng);
                double trial_delta = candidate.objective() - current_obj;

                json += fmt4(trial_delta);
            }

            json += "]}";
        }
    }

    json += "]}";
    write_line(json);
}
