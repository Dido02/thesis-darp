# Labeling Sweep Report

## 1. Filtering

- **Total states (all logs):** 13736
- **Epoch 1 removed:** 1194
- **Uninformative removed (all deltas=0):** 0
- **States remaining:** 12542
- **Log files:** 31
- **Operators:** 5 destroy × 4 repair = 20 pairs

## 2. SA Regime Distribution

SA regime is determined by the mean acceptance rate across all 20 pairs:
- **high-T** (>0.75): temperature so high that most moves are accepted regardless of quality
- **low-T** (<0.10): temperature so low that only strict improvements are accepted
- **discriminative** (0.10–0.75): acceptance rates vary meaningfully between pairs

### 2a. Overall

| Regime | Count | Fraction |
|--------|------:|--------:|
| high-T | 12235 | 0.976 |
| discriminative | 307 | 0.024 |
| low-T | 0 | 0.000 |

### 2b. By Epoch

| Epoch | high-T | discriminative | low-T | Total |
|------:|-------:|---------------:|------:|------:|
| 2 | 19 (6%) | 287 (94%) | 0 (0%) | 306 |
| 3 | 43 (80%) | 11 (20%) | 0 (0%) | 54 |
| 7 | 644 (99%) | 7 (1%) | 0 (0%) | 651 |
| 8 | 874 (100%) | 0 (0%) | 0 (0%) | 874 |
| 10 | 1772 (100%) | 0 (0%) | 0 (0%) | 1772 |
| 11 | 596 (100%) | 0 (0%) | 0 (0%) | 596 |
| 12 | 1280 (100%) | 0 (0%) | 0 (0%) | 1280 |
| 14 | 1679 (100%) | 0 (0%) | 0 (0%) | 1679 |
| 16 | 1785 (100%) | 0 (0%) | 0 (0%) | 1785 |
| 17 | 361 (100%) | 1 (0%) | 0 (0%) | 362 |
| 21 | 607 (100%) | 1 (0%) | 0 (0%) | 608 |
| 22 | 1106 (100%) | 0 (0%) | 0 (0%) | 1106 |
| 24 | 1469 (100%) | 0 (0%) | 0 (0%) | 1469 |

### 2c. By Within-Epoch Tertile

| Tertile | high-T | discriminative | low-T | Total |
|---------|-------:|---------------:|------:|------:|
| early | 4044 (100%) | 0 (0%) | 0 (0%) | 4044 |
| middle | 4045 (100%) | 1 (0%) | 0 (0%) | 4046 |
| late | 4146 (93%) | 306 (7%) | 0 (0%) | 4452 |

## 3. Summary Table

| Rule | Threshold | Mean Pos Rate | Std Pos Rate | Degenerate (<5% or >95%) | Balanced (30-70%) |
|------|-----------|--------------|-------------|--------------------------|-------------------|
| R1_bongiovanni | 0.3 | 0.952 | 0.082 | 15 | 0 |
| R1_bongiovanni | 0.5 | 0.915 | 0.154 | 15 | 4 |
| R1_bongiovanni | 0.7 | 0.875 | 0.230 | 15 | 4 |
| R1_bongiovanni | 0.9 | 0.817 | 0.331 | 15 | 0 |
| R2_improve_freq | 0.1 | 0.572 | 0.353 | 4 | 2 |
| R2_improve_freq | 0.2 | 0.466 | 0.313 | 4 | 6 |
| R2_improve_freq | 0.3 | 0.398 | 0.280 | 6 | 9 |
| R2_improve_freq | 0.5 | 0.308 | 0.229 | 6 | 12 |
| R3_improve_rel | 0.3 | 0.517 | 0.368 | 5 | 0 |
| R3_improve_rel | 0.5 | 0.431 | 0.325 | 6 | 6 |
| R3_improve_rel | 0.7 | 0.311 | 0.265 | 6 | 9 |
| R3_improve_rel | 0.9 | 0.161 | 0.183 | 8 | 3 |
| R4_mean_delta | -0.05 | 0.000 | 0.000 | 20 | 0 |
| R4_mean_delta | -0.01 | 0.018 | 0.014 | 20 | 0 |
| R4_mean_delta | -0.005 | 0.024 | 0.022 | 17 | 0 |
| R4_mean_delta | -0.001 | 0.201 | 0.179 | 8 | 5 |
| R5_combined | 0.2 | 0.466 | 0.313 | 4 | 6 |
| R5_combined | 0.3 | 0.398 | 0.280 | 6 | 9 |
| R5_combined | 0.5 | 0.308 | 0.229 | 6 | 12 |

## 4. Per-Pair Positive Rates (5×4 Grids)

### R1_bongiovanni, threshold=0.3
Mean=0.952, Degenerate=15, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 1.000 * | 1.000 * | 1.000 * | 0.781 |
| worst | 1.000 * | 1.000 * | 1.000 * | 0.792 |
| shaw | 1.000 * | 1.000 * | 1.000 * | 0.777 |
| rtcritical | 1.000 * | 1.000 * | 1.000 * | 0.815 |
| lowslack | 0.979 * | 0.976 * | 0.979 * | 0.939 |

\* = degenerate, \*\* = balanced

### R1_bongiovanni, threshold=0.5
Mean=0.915, Degenerate=15, Balanced=4

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 1.000 * | 1.000 * | 1.000 * | 0.583 ** |
| worst | 0.999 * | 1.000 * | 1.000 * | 0.638 ** |
| shaw | 1.000 * | 1.000 * | 1.000 * | 0.582 ** |
| rtcritical | 1.000 * | 1.000 * | 1.000 * | 0.639 ** |
| lowslack | 0.978 * | 0.976 * | 0.978 * | 0.926 |

\* = degenerate, \*\* = balanced

### R1_bongiovanni, threshold=0.7
Mean=0.875, Degenerate=15, Balanced=4

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 1.000 * | 1.000 * | 1.000 * | 0.391 ** |
| worst | 0.993 * | 1.000 * | 1.000 * | 0.447 ** |
| shaw | 1.000 * | 1.000 * | 1.000 * | 0.390 ** |
| rtcritical | 1.000 * | 1.000 * | 1.000 * | 0.440 ** |
| lowslack | 0.976 * | 0.974 * | 0.976 * | 0.906 |

\* = degenerate, \*\* = balanced

### R1_bongiovanni, threshold=0.9
Mean=0.817, Degenerate=15, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.990 * | 1.000 * | 1.000 * | 0.119 |
| worst | 0.967 * | 1.000 * | 1.000 * | 0.216 |
| shaw | 0.987 * | 1.000 * | 1.000 * | 0.110 |
| rtcritical | 0.974 * | 0.998 * | 1.000 * | 0.192 |
| lowslack | 0.973 * | 0.974 * | 0.975 * | 0.870 |

\* = degenerate, \*\* = balanced

### R2_improve_freq, threshold=0.1
Mean=0.572, Degenerate=4, Balanced=2

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.897 | 0.901 | 0.899 | 0.278 |
| worst | 0.868 | 0.872 | 0.873 | 0.358 ** |
| shaw | 0.866 | 0.870 | 0.873 | 0.192 |
| rtcritical | 0.754 | 0.761 | 0.762 | 0.308 ** |
| lowslack | 0.036 * | 0.036 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R2_improve_freq, threshold=0.2
Mean=0.466, Degenerate=4, Balanced=6

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.728 | 0.738 | 0.734 | 0.119 |
| worst | 0.801 | 0.811 | 0.808 | 0.288 |
| shaw | 0.681 ** | 0.691 ** | 0.690 ** | 0.067 |
| rtcritical | 0.624 ** | 0.628 ** | 0.629 ** | 0.171 |
| lowslack | 0.036 * | 0.036 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R2_improve_freq, threshold=0.3
Mean=0.398, Degenerate=6, Balanced=9

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.617 ** | 0.629 ** | 0.625 ** | 0.039 * |
| worst | 0.723 | 0.736 | 0.740 | 0.237 |
| shaw | 0.568 ** | 0.589 ** | 0.589 ** | 0.023 * |
| rtcritical | 0.531 ** | 0.539 ** | 0.540 ** | 0.134 |
| lowslack | 0.036 * | 0.035 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R2_improve_freq, threshold=0.5
Mean=0.308, Degenerate=6, Balanced=12

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.475 ** | 0.487 ** | 0.488 ** | 0.002 * |
| worst | 0.585 ** | 0.631 ** | 0.632 ** | 0.159 |
| shaw | 0.419 ** | 0.449 ** | 0.446 ** | 0.002 * |
| rtcritical | 0.383 ** | 0.416 ** | 0.418 ** | 0.108 |
| lowslack | 0.020 * | 0.018 * | 0.021 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R3_improve_rel, threshold=0.3
Mean=0.517, Degenerate=5, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.845 | 0.850 | 0.846 | 0.051 |
| worst | 0.872 | 0.884 | 0.884 | 0.271 |
| shaw | 0.798 | 0.812 | 0.810 | 0.029 * |
| rtcritical | 0.703 | 0.716 | 0.716 | 0.141 |
| lowslack | 0.036 * | 0.036 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R3_improve_rel, threshold=0.5
Mean=0.431, Degenerate=6, Balanced=6

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.718 | 0.729 | 0.727 | 0.003 * |
| worst | 0.814 | 0.854 | 0.856 | 0.181 |
| shaw | 0.613 ** | 0.644 ** | 0.644 ** | 0.002 * |
| rtcritical | 0.518 ** | 0.544 ** | 0.547 ** | 0.110 |
| lowslack | 0.035 * | 0.035 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R3_improve_rel, threshold=0.7
Mean=0.311, Degenerate=6, Balanced=9

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.475 ** | 0.530 ** | 0.530 ** | 0.000 * |
| worst | 0.719 | 0.780 | 0.777 | 0.076 |
| shaw | 0.330 ** | 0.394 ** | 0.401 ** | 0.000 * |
| rtcritical | 0.317 ** | 0.374 ** | 0.371 ** | 0.050 |
| lowslack | 0.030 * | 0.029 * | 0.030 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R3_improve_rel, threshold=0.9
Mean=0.161, Degenerate=8, Balanced=3

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.140 | 0.181 | 0.188 | 0.000 * |
| worst | 0.483 ** | 0.584 ** | 0.581 ** | 0.007 * |
| shaw | 0.077 | 0.142 | 0.144 | 0.000 * |
| rtcritical | 0.170 | 0.233 | 0.231 | 0.002 * |
| lowslack | 0.016 * | 0.015 * | 0.017 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R4_mean_delta, threshold=-0.05
Mean=0.000, Degenerate=20, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.000 * | 0.000 * | 0.000 * | 0.000 * |
| worst | 0.000 * | 0.000 * | 0.000 * | 0.000 * |
| shaw | 0.000 * | 0.000 * | 0.000 * | 0.000 * |
| rtcritical | 0.000 * | 0.000 * | 0.000 * | 0.000 * |
| lowslack | 0.000 * | 0.000 * | 0.000 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R4_mean_delta, threshold=-0.01
Mean=0.018, Degenerate=20, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.028 * | 0.029 * | 0.029 * | 0.000 * |
| worst | 0.032 * | 0.033 * | 0.033 * | 0.002 * |
| shaw | 0.028 * | 0.028 * | 0.028 * | 0.000 * |
| rtcritical | 0.025 * | 0.027 * | 0.028 * | 0.000 * |
| lowslack | 0.000 * | 0.000 * | 0.000 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R4_mean_delta, threshold=-0.005
Mean=0.024, Degenerate=17, Balanced=0

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.031 * | 0.031 * | 0.031 * | 0.000 * |
| worst | 0.061 | 0.064 | 0.065 | 0.006 * |
| shaw | 0.030 * | 0.030 * | 0.031 * | 0.000 * |
| rtcritical | 0.034 * | 0.034 * | 0.034 * | 0.000 * |
| lowslack | 0.000 * | 0.000 * | 0.000 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R4_mean_delta, threshold=-0.001
Mean=0.201, Degenerate=8, Balanced=5

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.227 | 0.277 | 0.278 | 0.000 * |
| worst | 0.442 ** | 0.521 ** | 0.521 ** | 0.034 * |
| shaw | 0.218 | 0.266 | 0.264 | 0.000 * |
| rtcritical | 0.273 | 0.342 ** | 0.344 ** | 0.003 * |
| lowslack | 0.000 * | 0.000 * | 0.000 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R5_combined, threshold=0.2
Mean=0.466, Degenerate=4, Balanced=6

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.728 | 0.738 | 0.734 | 0.119 |
| worst | 0.801 | 0.811 | 0.808 | 0.288 |
| shaw | 0.681 ** | 0.691 ** | 0.690 ** | 0.067 |
| rtcritical | 0.624 ** | 0.628 ** | 0.629 ** | 0.171 |
| lowslack | 0.036 * | 0.036 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R5_combined, threshold=0.3
Mean=0.398, Degenerate=6, Balanced=9

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.617 ** | 0.629 ** | 0.625 ** | 0.039 * |
| worst | 0.723 | 0.736 | 0.740 | 0.237 |
| shaw | 0.568 ** | 0.589 ** | 0.589 ** | 0.023 * |
| rtcritical | 0.531 ** | 0.539 ** | 0.540 ** | 0.134 |
| lowslack | 0.036 * | 0.035 * | 0.036 * | 0.000 * |

\* = degenerate, \*\* = balanced

### R5_combined, threshold=0.5
Mean=0.308, Degenerate=6, Balanced=12

| destroy \ repair | greedy | regret2 | regret3 | slack |
|------------------|------:|------:|------:|------:|
| random | 0.475 ** | 0.487 ** | 0.488 ** | 0.002 * |
| worst | 0.585 ** | 0.631 ** | 0.632 ** | 0.159 |
| shaw | 0.419 ** | 0.449 ** | 0.446 ** | 0.002 * |
| rtcritical | 0.383 ** | 0.416 ** | 0.418 ** | 0.108 |
| lowslack | 0.020 * | 0.018 * | 0.021 * | 0.000 * |

\* = degenerate, \*\* = balanced

## 5. Recommendation

### Criterion A: Mean positive rate closest to 0.5

1. **R3_improve_rel @ 0.3**: mean=0.517, degenerate=5, balanced=0
2. **R2_improve_freq @ 0.2**: mean=0.466, degenerate=4, balanced=6
3. **R5_combined @ 0.2**: mean=0.466, degenerate=4, balanced=6

### Criterion B: Fewest degenerate pairs

1. **R2_improve_freq @ 0.2**: degenerate=4, mean=0.466, balanced=6
2. **R5_combined @ 0.2**: degenerate=4, mean=0.466, balanced=6
3. **R2_improve_freq @ 0.1**: degenerate=4, mean=0.572, balanced=2

### Criterion C: Most balanced pairs (pos rate in [0.3, 0.7])

1. **R2_improve_freq @ 0.5**: balanced=12, degenerate=6, mean=0.308
2. **R5_combined @ 0.5**: balanced=12, degenerate=6, mean=0.308
3. **R2_improve_freq @ 0.3**: balanced=9, degenerate=6, mean=0.398

### Top Candidates (union across criteria)

| Rank | Rule | Threshold | Mean Pos | Degenerate | Balanced | Trade-off |
|------|------|-----------|---------|-----------|---------|-----------|
| 1 | R3_improve_rel | 0.3 | 0.517 | 5 | 0 | moderate balance |
| 2 | R2_improve_freq | 0.2 | 0.466 | 4 | 6 | moderate balance |
| 3 | R5_combined | 0.2 | 0.466 | 4 | 6 | moderate balance |
| 4 | R3_improve_rel | 0.5 | 0.431 | 6 | 6 | many degenerate pairs |
| 5 | R2_improve_freq | 0.1 | 0.572 | 4 | 2 | moderate balance |
| 6 | R2_improve_freq | 0.3 | 0.398 | 6 | 9 | many degenerate pairs |

### Caveat

Label balance is **necessary but not sufficient** for learnable signal. The next step after picking a rule is to retrain with per-log-file split and see if any actual signal emerges above trivial baselines.