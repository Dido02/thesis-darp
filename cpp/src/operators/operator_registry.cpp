#include "operators/operator_registry.h"

#include <algorithm>
#include <numeric>

void OperatorRegistry::add_destroy(const std::string& name, DestroyOp op) {
    destroy_ops_.emplace_back(name, std::move(op));
    destroy_weights_.push_back(1.0);
    destroy_scores_.push_back(0.0);
    destroy_uses_.push_back(0);
}

void OperatorRegistry::add_repair(const std::string& name, RepairOp op) {
    repair_ops_.emplace_back(name, std::move(op));
    repair_weights_.push_back(1.0);
    repair_scores_.push_back(0.0);
    repair_uses_.push_back(0);
}

void OperatorRegistry::set_selection_method(SelectionMethod method) {
    method_ = method;
}

std::pair<int, DestroyOp&> OperatorRegistry::select_destroy(std::mt19937& rng) {
    int idx = 0;
    if (destroy_ops_.size() == 1) {
        idx = 0;
    } else if (method_ == SelectionMethod::RANDOM) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(destroy_ops_.size()) - 1);
        idx = dist(rng);
    } else if (method_ == SelectionMethod::ROULETTE_WHEEL) {
        idx = roulette_select(destroy_weights_, rng);
    }
    return {idx, destroy_ops_[idx].second};
}

std::pair<int, RepairOp&> OperatorRegistry::select_repair(std::mt19937& rng) {
    int idx = 0;
    if (repair_ops_.size() == 1) {
        idx = 0;
    } else if (method_ == SelectionMethod::RANDOM) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(repair_ops_.size()) - 1);
        idx = dist(rng);
    } else if (method_ == SelectionMethod::ROULETTE_WHEEL) {
        idx = roulette_select(repair_weights_, rng);
    }
    return {idx, repair_ops_[idx].second};
}

void OperatorRegistry::update_weights(int destroy_idx, int repair_idx, double score) {
    if (method_ != SelectionMethod::ROULETTE_WHEEL) return;

    // Accumulate scores (called every iteration)
    destroy_scores_[destroy_idx] += score;
    repair_scores_[repair_idx] += score;
    destroy_uses_[destroy_idx]++;
    repair_uses_[repair_idx]++;
    segment_iter_++;

    // Update weights every segment_length iterations
    if (segment_iter_ >= segment_length_) {
        double reaction_factor = 0.1;
        for (int i = 0; i < static_cast<int>(destroy_weights_.size()); i++) {
            if (destroy_uses_[i] > 0) {
                double avg_score = destroy_scores_[i] / destroy_uses_[i];
                destroy_weights_[i] = destroy_weights_[i] * (1.0 - reaction_factor)
                                    + avg_score * reaction_factor;
            }
            destroy_scores_[i] = 0.0;
            destroy_uses_[i] = 0;
        }
        for (int i = 0; i < static_cast<int>(repair_weights_.size()); i++) {
            if (repair_uses_[i] > 0) {
                double avg_score = repair_scores_[i] / repair_uses_[i];
                repair_weights_[i] = repair_weights_[i] * (1.0 - reaction_factor)
                                    + avg_score * reaction_factor;
            }
            repair_scores_[i] = 0.0;
            repair_uses_[i] = 0;
        }

        // Floor to prevent zero weights
        double min_weight = 0.1;
        for (auto& w : destroy_weights_) w = std::max(w, min_weight);
        for (auto& w : repair_weights_) w = std::max(w, min_weight);

        segment_iter_ = 0;
    }
}

int OperatorRegistry::roulette_select(const std::vector<double>& weights, std::mt19937& rng) {
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    double cumsum = 0.0;
    for (int i = 0; i < static_cast<int>(weights.size()); i++) {
        cumsum += weights[i];
        if (r <= cumsum) return i;
    }
    return static_cast<int>(weights.size()) - 1;
}