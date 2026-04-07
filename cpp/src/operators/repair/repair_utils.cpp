#include "operators/repair/repair_utils.h"

long long DynamicStats::routes_checked = 0;
long long DynamicStats::request_evals = 0;
long long DynamicStats::feas_checks = 0;
int DynamicStats::repair_calls = 0;
double DynamicStats::aux_precompute_ms = 0.0;
