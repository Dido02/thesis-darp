#pragma once

// Lightweight state snapshot passed to feature functions and MLSelector.
// Holds just what features need that isn't already on the solution/instance.
//
// Field semantics for `current_q`: the size of the most recent destroy
// operator's perturbation (= |destroyed.unassigned| - |current.unassigned|
// at the previous iteration). Initialized to 0 at iter 0. At feature-eval
// time on iter N, this is q[N-1]. The parity test sets this field from
// fixture state.q, so training/parity sees q[N] but runtime sees q[N-1] —
// a 1-iteration shift that's tolerable since q changes slowly across iters.
struct ALNSState {
    double temperature = 0.0;       // current SA temperature
    double sa_start_temp = 0.0;     // initial SA temperature (for normalization)
    int current_q = 0;              // size of most recent destroy perturbation
    int iter = 0;                   // current iteration counter (0-indexed)
    int max_iterations = 0;         // total iterations planned for this run
    double obj_initial = 0.0;       // objective at run/epoch start
    double obj_best = 0.0;          // best objective seen so far in this run
    double sim_clock = 0.0;         // simulation clock at this epoch's start (dynamic mode)
    double epoch_length = 0.0;      // epoch period in sim time units (dynamic mode)
};
