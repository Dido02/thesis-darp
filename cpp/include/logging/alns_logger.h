#pragma once

#include "solution.h"
#include "operators/operator_registry.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <vector>

struct ALNSLogConfig {
    bool log_state = false;
    bool log_trials = false;
    int m_trials = 5;
    int m_trials_every = 50;
    std::string output_dir = "logs";
    std::string instance_path;
    std::string cli_args;
    int seed = 42;
    bool dynamic = false;
};

class ALNSLogger {
public:
    explicit ALNSLogger(const ALNSLogConfig& config, const std::string& filename);
    ~ALNSLogger();

    void write_header(const OperatorRegistry& registry,
                      int max_iterations,
                      double sa_start_temp, double sa_end_temp, double sa_step,
                      const std::string& feasibility_mode,
                      double unassigned_penalty,
                      const std::string& selection_config,
                      int num_requests, int num_vehicles);

    void log_iteration(int iter, int d_idx, int r_idx,
                       const std::string& d_name, const std::string& r_name,
                       double obj_before, double obj_candidate, double obj_best,
                       double delta, double temperature, bool accept, int q,
                       const std::vector<std::vector<int>>& routes,
                       const std::set<int>& unassigned,
                       const std::set<int>& committed,
                       const std::vector<int>& min_insert_pos);

    void log_weights(int iter, const OperatorRegistry& registry);

    void log_sa_calibrated(int iter, double sa_step, int estimated_total);

    void log_epoch_start(int epoch_id, double sim_clock,
                         const std::vector<std::vector<int>>& routes,
                         const std::set<int>& unassigned,
                         const std::set<int>& committed,
                         const std::vector<int>& min_insert_pos,
                         const std::set<int>& pending,
                         const std::set<int>& rejected,
                         double obj);

    void run_and_log_trials(int iter,
                            const DARPSolution& current,
                            double current_obj, double temperature,
                            const OperatorRegistry& registry,
                            uint64_t trial_base_seed);

    bool is_enabled() const { return config_.log_state || config_.log_trials; }
    bool is_logging_state() const { return config_.log_state; }
    bool should_run_trials(int iter) const;

private:
    ALNSLogConfig config_;
    std::ofstream file_;

    void write_line(const std::string& json);

    // JSON helpers
    static std::string fmt4(double v);
    static std::string fmt6(double v);
    static std::string json_int_array(const std::vector<int>& v);
    static std::string json_int_set(const std::set<int>& s);
    static std::string json_double_array4(const std::vector<double>& v);
    static std::string json_double_array6(const std::vector<double>& v);
    static std::string json_routes(const std::vector<std::vector<int>>& routes);
    static std::string json_string_array(const std::vector<std::string>& v);
    static std::string escape_json_string(const std::string& s);
};
