#pragma once

#include "instance.h"
#include "solution.h"

DARPSolution build_initial_solution(const DARPInstance& inst,
                                     const std::string& feasibility_mode = "o1");
