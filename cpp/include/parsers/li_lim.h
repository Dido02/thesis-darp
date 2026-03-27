#pragma once

#include "instance.h"
#include <string>

DARPInstance parse_li_lim(const std::string& filepath, double max_ride_time = 1e9);