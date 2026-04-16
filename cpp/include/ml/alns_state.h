#pragma once

// Lightweight state snapshot passed to feature functions and MLSelector.
// Holds just what features need that isn't already on the solution/instance.
struct ALNSState {
    double temperature;      // current SA temperature
    double sa_start_temp;    // initial SA temperature (for normalization features)
};
