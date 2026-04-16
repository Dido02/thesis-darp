# ML Pipeline Diagnostic Report
## Headline Finding
- 15/20 pairs have positive rate >90% (heavily imbalanced toward class 1).
- Average F1 drop from Split A (per-row) to Split B (per-log): +0.104.
- 12/20 classifiers predict class 1 for >90% of test samples under Split B.
- For the heavily-imbalanced pairs, the always-1 baseline already achieves F1 > 0.97.
- The high F1 scores are **primarily driven by trivial label imbalance**, not learned signal.
  Most operator pairs are labeled 'good' >90% of the time because the 0.9× threshold
  combined with similar acceptance rates across pairs makes almost everything 'good'.
## Summary Table
| Pair | Pos Rate | RF F1 (A) | RF F1 (B) | F1 no temp (B) | Always-1 F1 | Verdict |
|------|----------|-----------|-----------|-----------------|-------------|----------|
| pair_0_0 | 0.962 | 0.992 | 0.983 | 0.982 | 0.980 | degenerate |
| pair_0_1 | 0.991 | 0.998 | 0.992 | 0.993 | 0.995 | degenerate |
| pair_0_2 | 0.994 | 0.999 | 0.995 | 0.991 | 0.997 | degenerate |
| pair_0_3 | 0.051 | 0.606 | 0.130 | 0.081 | 0.096 | leakage suspected |
| pair_1_0 | 0.948 | 0.992 | 0.971 | 0.962 | 0.973 | trivially imbalanced |
| pair_1_1 | 0.996 | 0.998 | 0.997 | 0.994 | 0.998 | degenerate |
| pair_1_2 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | degenerate |
| pair_1_3 | 0.097 | 0.634 | 0.280 | 0.172 | 0.178 | leakage suspected |
| pair_2_0 | 0.955 | 0.989 | 0.975 | 0.955 | 0.977 | degenerate |
| pair_2_1 | 0.994 | 0.999 | 0.995 | 0.995 | 0.997 | degenerate |
| pair_2_2 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | degenerate |
| pair_2_3 | 0.049 | 0.577 | 0.076 | 0.065 | 0.093 | leakage suspected |
| pair_3_0 | 0.936 | 0.990 | 0.967 | 0.960 | 0.967 | trivially imbalanced |
| pair_3_1 | 0.980 | 0.996 | 0.989 | 0.977 | 0.990 | degenerate |
| pair_3_2 | 0.988 | 0.998 | 0.996 | 0.992 | 0.994 | degenerate |
| pair_3_3 | 0.075 | 0.617 | 0.135 | 0.159 | 0.140 | leakage suspected |
| pair_4_0 | 0.926 | 0.993 | 0.966 | 0.942 | 0.962 | trivially imbalanced |
| pair_4_1 | 0.934 | 0.993 | 0.978 | 0.972 | 0.966 | trivially imbalanced |
| pair_4_2 | 0.938 | 0.994 | 0.979 | 0.967 | 0.968 | trivially imbalanced |
| pair_4_3 | 0.694 | 0.980 | 0.857 | 0.826 | 0.819 | marginal signal |

## Diagnostic 1: Label Distribution
- 11 pairs are degenerate (pos_rate <5% or >95%).
- 15 pairs have pos_rate > 90%.
- The 0.9× threshold in label computation, combined with similar acceptance rates
  across operator pairs in the M-trials data, causes most pairs to be labeled 'good'
  for most iterations. The classification task is therefore trivial for these pairs.
## Diagnostic 2: Trivial Baseline F1
- pair_0_0: always-1 F1 = 0.980, pos_rate = 0.962
- pair_0_1: always-1 F1 = 0.995, pos_rate = 0.991
- pair_0_2: always-1 F1 = 0.997, pos_rate = 0.994
- pair_1_0: always-1 F1 = 0.973, pos_rate = 0.948
- pair_1_1: always-1 F1 = 0.998, pos_rate = 0.996
- pair_1_2: always-1 F1 = 0.999, pos_rate = 0.999
- pair_2_0: always-1 F1 = 0.977, pos_rate = 0.955
- pair_2_1: always-1 F1 = 0.997, pos_rate = 0.994
- pair_2_2: always-1 F1 = 0.999, pos_rate = 0.998
- pair_3_0: always-1 F1 = 0.967, pos_rate = 0.936
- pair_3_1: always-1 F1 = 0.990, pos_rate = 0.980
- pair_3_2: always-1 F1 = 0.994, pos_rate = 0.988
- pair_4_0: always-1 F1 = 0.962, pos_rate = 0.926
- pair_4_1: always-1 F1 = 0.966, pos_rate = 0.934
- pair_4_2: always-1 F1 = 0.968, pos_rate = 0.938
## Diagnostic 3: Split A vs Split B
- Production training uses per-row split (Split A): potential within-log leakage.
- Average F1 delta (A - B): +0.104
- pair_0_3: F1_A=0.606 → F1_B=0.130 (delta=+0.477)
- pair_1_3: F1_A=0.634 → F1_B=0.280 (delta=+0.353)
- pair_2_3: F1_A=0.577 → F1_B=0.076 (delta=+0.501)
- pair_3_3: F1_A=0.617 → F1_B=0.135 (delta=+0.482)
- pair_4_3: F1_A=0.980 → F1_B=0.857 (delta=+0.124)
## Diagnostic 4: Drop temperature_normalized
- pair_0_3: F1_B drops by +0.049 without temperature
- pair_1_3: F1_B drops by +0.108 without temperature
- pair_4_3: F1_B drops by +0.030 without temperature
## Diagnostic 5: Drop temperature + num_active_requests
- pair_0_0: 3-feat F1_B=0.982 vs full F1_B=0.983 vs always-1=0.980
- pair_0_1: 3-feat F1_B=0.988 vs full F1_B=0.992 vs always-1=0.995
- pair_0_2: 3-feat F1_B=0.991 vs full F1_B=0.995 vs always-1=0.997
- pair_0_3: 3-feat F1_B=0.039 vs full F1_B=0.130 vs always-1=0.096
- pair_1_0: 3-feat F1_B=0.962 vs full F1_B=0.971 vs always-1=0.973
- pair_1_1: 3-feat F1_B=0.994 vs full F1_B=0.997 vs always-1=0.998
- pair_1_2: 3-feat F1_B=1.000 vs full F1_B=1.000 vs always-1=0.999
- pair_1_3: 3-feat F1_B=0.154 vs full F1_B=0.280 vs always-1=0.178
- pair_2_0: 3-feat F1_B=0.958 vs full F1_B=0.975 vs always-1=0.977
- pair_2_1: 3-feat F1_B=0.995 vs full F1_B=0.995 vs always-1=0.997
- pair_2_2: 3-feat F1_B=0.999 vs full F1_B=0.999 vs always-1=0.999
- pair_2_3: 3-feat F1_B=0.036 vs full F1_B=0.076 vs always-1=0.093
- pair_3_0: 3-feat F1_B=0.971 vs full F1_B=0.967 vs always-1=0.967
- pair_3_1: 3-feat F1_B=0.978 vs full F1_B=0.989 vs always-1=0.990
- pair_3_2: 3-feat F1_B=0.987 vs full F1_B=0.996 vs always-1=0.994
- pair_3_3: 3-feat F1_B=0.221 vs full F1_B=0.135 vs always-1=0.140
- pair_4_0: 3-feat F1_B=0.968 vs full F1_B=0.966 vs always-1=0.962
- pair_4_1: 3-feat F1_B=0.971 vs full F1_B=0.978 vs always-1=0.966
- pair_4_2: 3-feat F1_B=0.972 vs full F1_B=0.979 vs always-1=0.968
- pair_4_3: 3-feat F1_B=0.825 vs full F1_B=0.857 vs always-1=0.819
## Diagnostic 6: Per-log-file F1 Variance (Split B)
- pair_0_0: mean=0.989 std=0.020 min=0.947 max=1.000
- pair_0_1: mean=0.995 std=0.014 min=0.959 max=1.000
- pair_0_2: mean=0.997 std=0.008 min=0.977 max=1.000
- pair_0_3: mean=0.124 std=0.191 min=0.000 max=0.556
- pair_1_0: mean=0.982 std=0.032 min=0.918 max=1.000
- pair_1_1: mean=0.998 std=0.005 min=0.985 max=1.000
- pair_1_2: mean=1.000 std=0.000 min=1.000 max=1.000
- pair_1_3: mean=0.196 std=0.266 min=0.000 max=0.700
- pair_2_0: mean=0.985 std=0.027 min=0.931 max=1.000
- pair_2_1: mean=0.997 std=0.008 min=0.976 max=1.000
- pair_2_2: mean=1.000 std=0.001 min=0.996 max=1.000
- pair_2_3: mean=0.072 std=0.103 min=0.000 max=0.286
- pair_3_0: mean=0.978 std=0.037 min=0.907 max=1.000
- pair_3_1: mean=0.992 std=0.017 min=0.948 max=1.000
- pair_3_2: mean=0.997 std=0.007 min=0.980 max=1.000
- pair_3_3: mean=0.102 std=0.187 min=0.000 max=0.531
- pair_4_0: mean=0.977 std=0.040 min=0.901 max=1.000
- pair_4_1: mean=0.986 std=0.030 min=0.909 max=1.000
- pair_4_2: mean=0.987 std=0.025 min=0.926 max=1.000
- pair_4_3: mean=0.652 std=0.352 min=0.091 max=1.000
## Diagnostic 7: Confusion Matrices (Split B)
- pair_0_0: TN=112 FP=87 FN=43 TP=3675 (pred_1_rate=0.96)
- pair_0_1: TN=0 FP=63 FN=0 TP=3854 (pred_1_rate=1.00)
- pair_0_2: TN=0 FP=36 FN=0 TP=3881 (pred_1_rate=1.00)
- pair_0_3: TN=3715 FP=59 FN=129 TP=14 (pred_1_rate=0.02)
- pair_1_0: TN=66 FP=151 FN=69 TP=3631 (pred_1_rate=0.97)
- pair_1_1: TN=0 FP=24 FN=0 TP=3893 (pred_1_rate=1.00)
- pair_1_2: TN=0 FP=0 FN=0 TP=3917 (pred_1_rate=1.00)
- pair_1_3: TN=3494 FP=126 FN=228 TP=69 (pred_1_rate=0.05)
- pair_2_0: TN=118 FP=121 FN=67 TP=3611 (pred_1_rate=0.95)
- pair_2_1: TN=0 FP=38 FN=0 TP=3879 (pred_1_rate=1.00)
- pair_2_2: TN=0 FP=6 FN=0 TP=3911 (pred_1_rate=1.00)
- pair_2_3: TN=3714 FP=53 FN=142 TP=8 (pred_1_rate=0.02)
- pair_3_0: TN=275 FP=43 FN=187 TP=3412 (pred_1_rate=0.88)
- pair_3_1: TN=67 FP=36 FN=51 TP=3763 (pred_1_rate=0.97)
- pair_3_2: TN=44 FP=32 FN=2 TP=3839 (pred_1_rate=0.99)
- pair_3_3: TN=3613 FP=20 FN=262 TP=22 (pred_1_rate=0.01)
- pair_4_0: TN=340 FP=108 FN=124 TP=3345 (pred_1_rate=0.88)
- pair_4_1: TN=310 FP=50 FN=107 TP=3450 (pred_1_rate=0.89)
- pair_4_2: TN=277 FP=65 FN=84 TP=3491 (pred_1_rate=0.91)
- pair_4_3: TN=1426 FP=405 FN=219 TP=1867 (pred_1_rate=0.58)
## Recommendations
1. **The 0.9× threshold creates trivially imbalanced labels.** Consider lowering to 0.7× or 0.5× to create a harder classification task with more balanced classes.
2. **Switch to per-log-file split** for honest evaluation — the current per-row split allows within-trajectory leakage.
3. **Add more diverse features** — 5 features on heavily imbalanced labels doesn't test whether the model can actually discriminate between operator pairs.
4. **Focus evaluation on the slack-repair pairs** (pair_*_3) — these are the only pairs with meaningful class balance and where operator selection actually matters.
5. **Generate data from more diverse instances** — current data is all Manhattan dynamic; adding Cordeau static instances would test generalization.
