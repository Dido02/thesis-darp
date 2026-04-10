#include "operators/repair/repair_utils.h"

long long DynamicStats::routes_checked = 0;
long long DynamicStats::request_evals = 0;
long long DynamicStats::feas_checks = 0;
int DynamicStats::repair_calls = 0;
double DynamicStats::aux_precompute_ms = 0.0;

double DynamicStats::repair_greedy_ms = 0.0;
double DynamicStats::repair_regret2_ms = 0.0;
double DynamicStats::repair_regret3_ms = 0.0;
double DynamicStats::repair_regret4_ms = 0.0;
double DynamicStats::repair_regretM_ms = 0.0;
double DynamicStats::repair_slack_ms = 0.0;
int DynamicStats::repair_greedy_count = 0;
int DynamicStats::repair_regret2_count = 0;
int DynamicStats::repair_regret3_count = 0;
int DynamicStats::repair_regret4_count = 0;
int DynamicStats::repair_regretM_count = 0;
int DynamicStats::repair_slack_count = 0;

double DynamicStats::destroy_random_ms = 0.0;
double DynamicStats::destroy_worst_ms = 0.0;
double DynamicStats::destroy_shaw_ms = 0.0;
double DynamicStats::destroy_rtcritical_ms = 0.0;
double DynamicStats::destroy_lowslack_ms = 0.0;
double DynamicStats::destroy_entireroute_ms = 0.0;
int DynamicStats::destroy_random_count = 0;
int DynamicStats::destroy_worst_count = 0;
int DynamicStats::destroy_shaw_count = 0;
int DynamicStats::destroy_rtcritical_count = 0;
int DynamicStats::destroy_lowslack_count = 0;
int DynamicStats::destroy_entireroute_count = 0;
