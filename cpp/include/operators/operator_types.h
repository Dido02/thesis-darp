#pragma once

#include "solution.h"
#include <functional>
#include <random>
#include <string>

// Function signatures for destroy and repair operators
using DestroyOp = std::function<DARPSolution(const DARPSolution&, std::mt19937&)>;
using RepairOp = std::function<DARPSolution(const DARPSolution&, std::mt19937&)>;

// Selection method for choosing operators each iteration
enum class SelectionMethod {
    RANDOM,           // uniform random among registered operators
    ROULETTE_WHEEL,   // adaptive weight adjustment (standard ALNS)
    ML_GUIDED         // ML model predicts best operator pair
};