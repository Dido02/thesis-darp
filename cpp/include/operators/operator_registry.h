#pragma once

#include "operators/operator_types.h"
#include <random>
#include <string>
#include <utility>
#include <vector>

class OperatorRegistry {
public:
    // Register operators
    void add_destroy(const std::string& name, DestroyOp op);
    void add_repair(const std::string& name, RepairOp op);

    // Select an operator, returns (index, reference to function)
    std::pair<int, DestroyOp&> select_destroy(std::mt19937& rng);
    std::pair<int, RepairOp&> select_repair(std::mt19937& rng);

    // Update weights after an iteration (for roulette wheel)
    void update_weights(int destroy_idx, int repair_idx, double score);

    // Configuration
    void set_selection_method(SelectionMethod method);
    SelectionMethod get_selection_method() const { return method_; }

    // Info
    int num_destroy() const { return static_cast<int>(destroy_ops_.size()); }
    int num_repair() const { return static_cast<int>(repair_ops_.size()); }
    const std::string& destroy_name(int idx) const { return destroy_ops_[idx].first; }
    const std::string& repair_name(int idx) const { return repair_ops_[idx].first; }

    const std::vector<double>& get_destroy_weights() const { return destroy_weights_; }
    const std::vector<double>& get_repair_weights() const { return repair_weights_; }

private:
    std::vector<std::pair<std::string, DestroyOp>> destroy_ops_;
    std::vector<std::pair<std::string, RepairOp>> repair_ops_;
    std::vector<double> destroy_weights_;
    std::vector<double> repair_weights_;
    SelectionMethod method_ = SelectionMethod::RANDOM;

    // Roulette wheel helper
    int roulette_select(const std::vector<double>& weights, std::mt19937& rng);

    std::vector<double> destroy_scores_;
    std::vector<double> repair_scores_;
    std::vector<int> destroy_uses_;
    std::vector<int> repair_uses_;
    int segment_iter_ = 0;
    int segment_length_ = 100;
};