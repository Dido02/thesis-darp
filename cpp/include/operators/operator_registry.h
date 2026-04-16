#pragma once

#include "operators/operator_types.h"
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Forward declarations so we can hold an MLSelector pointer without pulling
// onnxruntime headers into every translation unit that uses the registry.
class DARPSolution;
class DARPInstance;
class MLSelector;
struct ALNSState;

class OperatorRegistry {
public:
    OperatorRegistry();
    ~OperatorRegistry();

    // Move-only (holds std::unique_ptr<MLSelector>). Copy is deleted.
    OperatorRegistry(const OperatorRegistry&) = delete;
    OperatorRegistry& operator=(const OperatorRegistry&) = delete;
    OperatorRegistry(OperatorRegistry&&) noexcept;
    OperatorRegistry& operator=(OperatorRegistry&&) noexcept;

    // Register operators
    void add_destroy(const std::string& name, DestroyOp op);
    void add_repair(const std::string& name, RepairOp op);

    // Individual selectors (used for RANDOM and ROULETTE_WHEEL).
    std::pair<int, DestroyOp&> select_destroy(std::mt19937& rng);
    std::pair<int, RepairOp&> select_repair(std::mt19937& rng);

    // Unified selector that dispatches on the configured SelectionMethod.
    // For RANDOM / ROULETTE_WHEEL: independently selects destroy and repair
    //   (sol/inst/st may be null).
    // For ML_GUIDED: requires non-null sol/inst/st and delegates to MLSelector;
    //   throws if no MLSelector has been attached.
    std::pair<int, int> select_pair(
        std::mt19937& rng,
        const DARPSolution* sol = nullptr,
        const DARPInstance* inst = nullptr,
        const ALNSState* alns_state = nullptr);

    // Update weights after an iteration (for roulette wheel)
    void update_weights(int destroy_idx, int repair_idx, double score);

    // Configuration
    void set_selection_method(SelectionMethod method);
    SelectionMethod get_selection_method() const { return method_; }

    // Attach an ML selector (ownership transferred). Must be called after all
    // operators have been registered so subset intersection is accurate.
    void set_ml_selector(std::unique_ptr<MLSelector> selector);
    bool has_ml_selector() const { return ml_selector_ != nullptr; }
    const MLSelector* get_ml_selector() const { return ml_selector_.get(); }

    // Info
    int num_destroy() const { return static_cast<int>(destroy_ops_.size()); }
    int num_repair() const { return static_cast<int>(repair_ops_.size()); }
    const std::string& destroy_name(int idx) const { return destroy_ops_[idx].first; }
    const std::string& repair_name(int idx) const { return repair_ops_[idx].first; }

    const std::vector<double>& get_destroy_weights() const { return destroy_weights_; }
    const std::vector<double>& get_repair_weights() const { return repair_weights_; }
    const DestroyOp& get_destroy(int idx) const { return destroy_ops_[idx].second; }
    const RepairOp& get_repair(int idx) const { return repair_ops_[idx].second; }

private:
    std::vector<std::pair<std::string, DestroyOp>> destroy_ops_;
    std::vector<std::pair<std::string, RepairOp>> repair_ops_;
    std::vector<double> destroy_weights_;
    std::vector<double> repair_weights_;
    SelectionMethod method_ = SelectionMethod::RANDOM;

    std::unique_ptr<MLSelector> ml_selector_;

    // Roulette wheel helper
    int roulette_select(const std::vector<double>& weights, std::mt19937& rng);

    std::vector<double> destroy_scores_;
    std::vector<double> repair_scores_;
    std::vector<int> destroy_uses_;
    std::vector<int> repair_uses_;
    int segment_iter_ = 0;
    int segment_length_ = 100;
};