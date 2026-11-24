#!/usr/bin/env python3
"""Per-pair Gini feature importance for the deployed d=None, K=20 forest.

The 9 USED pair-classifiers at ml_train/models_v3_full_20f_t07/ ARE the
grid sweep's forest_dNone_K20 cell under the deployed_75_25 protocol —
same seed, same hyperparams, same data, verified earlier. Load the
joblibs directly, extract feature_importances_ per pair, normalise each
row to sum to 1.0 (defensive — sklearn already guarantees this).

Outputs:
  cross_pair_ranking_grid_2026-05-25/tables/per_pair_feature_importance.csv
  cross_pair_ranking_grid_2026-05-25/PER_PAIR_FEATURE_IMPORTANCE_REPORT.md
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
DEPLOYED_DIR = REPO / "ml_train" / "models_v3_full_20f_t07"
OUT_CSV = REPO / "cross_pair_ranking_grid_2026-05-25" / "tables" / "per_pair_feature_importance.csv"
OUT_MD  = REPO / "cross_pair_ranking_grid_2026-05-25" / "PER_PAIR_FEATURE_IMPORTANCE_REPORT.md"

FEATURE_NAMES = [
    "num_active_requests", "radius_avg_tt", "frac_committed", "avg_route_load",
    "temperature_normalized", "ride_time_utilization_avg",
    "ride_time_utilization_max", "near_violation_count", "route_slack_avg",
    "route_slack_std", "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "neighborhood_size_frac", "cost_vs_initial_frac",
    "cost_vs_incumbent_frac", "iteration_norm", "unassigned_count",
    "committed_travel_time_frac", "horizon_elapsed_frac",
]
USED_PAIRS_TUP = [(d, r) for d in (0, 1, 2) for r in (0, 1, 2)]
DESTROY_NAMES = {0: "random", 1: "worst", 2: "shaw"}
REPAIR_NAMES  = {0: "greedy", 1: "regret2", 2: "regret3"}

# Hypothesis-driven feature groupings for §4 analysis
COST_FEATURES = [
    "cost_contribution_std", "cost_contribution_max_ratio",
    "route_cost_std", "cost_vs_initial_frac", "cost_vs_incumbent_frac",
]
SPATIAL_TEMPORAL_FEATURES = [
    "radius_avg_tt", "route_slack_avg", "route_slack_std",
    "ride_time_utilization_avg", "ride_time_utilization_max",
]
TEMPORAL_PROGRESS_FEATURES = [
    "temperature_normalized", "iteration_norm", "horizon_elapsed_frac",
    "frac_committed",
]


def main():
    print("Loading 9 per-pair joblibs from deployed forest...")
    importance_mat = np.zeros((9, 20), dtype=float)
    rows = []
    for i, (d, r) in enumerate(USED_PAIRS_TUP):
        path = DEPLOYED_DIR / f"rf_pair_{d}_{r}.joblib"
        clf = joblib.load(path)
        imps = clf.feature_importances_
        # Defensive renormalisation (sklearn already guarantees sum=1)
        imps = imps / imps.sum()
        importance_mat[i] = imps
        row = {
            "pair_id":   f"pair_{d}_{r}",
            "destroy_op": DESTROY_NAMES[d],
            "repair_op":  REPAIR_NAMES[r],
        }
        for fn, v in zip(FEATURE_NAMES, imps):
            row[fn] = float(v)
        rows.append(row)
        print(f"  pair_{d}_{r}  ({DESTROY_NAMES[d]:6s}+{REPAIR_NAMES[r]:8s})  "
              f"sum={imps.sum():.6f}  top1={FEATURE_NAMES[int(imps.argmax())]} ({imps.max():.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows × {len(df.columns)} cols)")

    write_report(importance_mat)


def write_report(M: np.ndarray):
    L = []
    L.append("# Per-pair Gini feature importance — deployed forest (d=None, K=20)")
    L.append("")
    L.append("## Source")
    L.append("")
    L.append("9 binary `RandomForestClassifier` per-pair models loaded directly from `ml_train/models_v3_full_20f_t07/rf_pair_*.joblib` — these ARE the grid sweep's `forest_dNone_K20` cell under the `deployed_75_25` protocol (`n_estimators=100`, `max_depth=None`, `class_weight='balanced'`, `random_state=42`, all 20 features, 75/25 log-file split with `random_state=42`).")
    L.append("")
    L.append("Importances are sklearn's mean decrease in Gini impurity, normalised to sum to 1 per pair-classifier. Loaded from disk; no retraining.")
    L.append("")

    # ----- §1 Top-5 per pair -----
    top5_per_pair: dict[str, list[tuple[str, float]]] = {}
    for i, (d, r) in enumerate(USED_PAIRS_TUP):
        pair_name = f"pair_{d}_{r}"
        imps = M[i]
        top5_idx = np.argsort(-imps)[:5]
        top5_per_pair[pair_name] = [(FEATURE_NAMES[j], float(imps[j])) for j in top5_idx]

    L.append("## 1. Top-5 features by importance per pair")
    L.append("")
    for i, (d, r) in enumerate(USED_PAIRS_TUP):
        pair_name = f"pair_{d}_{r}"
        L.append(f"### `{pair_name}` — {DESTROY_NAMES[d]} + {REPAIR_NAMES[r]}")
        L.append("")
        L.append("| rank | feature | importance |")
        L.append("|---:|---|---:|")
        for rank, (fn, imp) in enumerate(top5_per_pair[pair_name], 1):
            L.append(f"| {rank} | `{fn}` | {imp:.4f} |")
        L.append("")

    # ----- §2 Universal features -----
    top5_sets = [set(fn for fn, _ in v) for v in top5_per_pair.values()]
    universal = set.intersection(*top5_sets)
    L.append("## 2. Universal features (top-5 in all 9 pairs)")
    L.append("")
    if universal:
        L.append("| feature | mean importance across 9 pairs | min / max across pairs |")
        L.append("|---|---:|---:|")
        for fn in sorted(universal, key=lambda f: -M[:, FEATURE_NAMES.index(f)].mean()):
            j = FEATURE_NAMES.index(fn)
            col = M[:, j]
            L.append(f"| `{fn}` | {col.mean():.4f} | {col.min():.4f} / {col.max():.4f} |")
    else:
        L.append("_No feature appears in the top-5 of all 9 pairs._")
    L.append("")

    # ----- §3 Pair-specific features -----
    L.append("## 3. Pair-specific features (top-5 in only 1 or 2 pairs)")
    L.append("")
    feat_pair_count: dict[str, int] = {fn: sum(1 for s in top5_sets if fn in s) for fn in FEATURE_NAMES}
    pair_specific = {fn: c for fn, c in feat_pair_count.items() if 1 <= c <= 2}
    if pair_specific:
        L.append("| feature | # pairs with this in top-5 | which pairs |")
        L.append("|---|---:|---|")
        # Sort by descending count, then by mean importance among those pairs
        order = sorted(
            pair_specific.keys(),
            key=lambda f: (-feat_pair_count[f], -M[:, FEATURE_NAMES.index(f)].mean()),
        )
        for fn in order:
            which = [pn for pn, top5 in top5_per_pair.items()
                     if any(fn == f for f, _ in top5)]
            L.append(f"| `{fn}` | {feat_pair_count[fn]} | {', '.join('`'+w+'`' for w in which)} |")
    else:
        L.append("_No features are pair-specific (every feature in top-5 of ≥3 pairs or none)._")
    L.append("")

    # ----- §4 Operator-semantics analysis -----
    L.append("## 4. Operator-semantics analysis")
    L.append("")
    L.append("**Hypothesis-driven feature groups:**")
    L.append(f"- **Cost features:** `{'`, `'.join(COST_FEATURES)}`")
    L.append(f"- **Spatial/temporal-structure features:** `{'`, `'.join(SPATIAL_TEMPORAL_FEATURES)}`")
    L.append(f"- **Progress/temperature features:** `{'`, `'.join(TEMPORAL_PROGRESS_FEATURES)}`")
    L.append("")

    def group_mean_for_rows(group: list[str], row_idxs: list[int]) -> float:
        col_idxs = [FEATURE_NAMES.index(f) for f in group]
        return float(M[np.ix_(row_idxs, col_idxs)].mean())

    # Row indices per destroy family
    rd = [0, 1, 2]   # destroy 0 = random
    wd = [3, 4, 5]   # destroy 1 = worst
    sd = [6, 7, 8]   # destroy 2 = shaw

    L.append("**Mean importance per feature group, by destroy-operator family:**")
    L.append("")
    L.append("| destroy family | cost group | spatial/temporal group | progress group |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| random `pair_0_*` | {group_mean_for_rows(COST_FEATURES, rd):.4f} | "
             f"{group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, rd):.4f} | "
             f"{group_mean_for_rows(TEMPORAL_PROGRESS_FEATURES, rd):.4f} |")
    L.append(f"| worst  `pair_1_*` | {group_mean_for_rows(COST_FEATURES, wd):.4f} | "
             f"{group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, wd):.4f} | "
             f"{group_mean_for_rows(TEMPORAL_PROGRESS_FEATURES, wd):.4f} |")
    L.append(f"| shaw   `pair_2_*` | {group_mean_for_rows(COST_FEATURES, sd):.4f} | "
             f"{group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, sd):.4f} | "
             f"{group_mean_for_rows(TEMPORAL_PROGRESS_FEATURES, sd):.4f} |")
    L.append("")

    # Top-5 features per destroy family (averaged across the 3 repair pairs)
    L.append("**Top-5 features per destroy family (averaging importance across the 3 repair pairs):**")
    L.append("")
    L.append("| rank | random destroy | worst destroy | shaw destroy |")
    L.append("|---:|---|---|---|")
    def top5_for_family(row_idxs):
        means = M[row_idxs].mean(axis=0)
        idx = np.argsort(-means)[:5]
        return [(FEATURE_NAMES[i], float(means[i])) for i in idx]
    rd5 = top5_for_family(rd); wd5 = top5_for_family(wd); sd5 = top5_for_family(sd)
    for k in range(5):
        L.append(f"| {k+1} | "
                 f"`{rd5[k][0]}` ({rd5[k][1]:.3f}) | "
                 f"`{wd5[k][0]}` ({wd5[k][1]:.3f}) | "
                 f"`{sd5[k][0]}` ({sd5[k][1]:.3f}) |")
    L.append("")

    # Autodetect findings (hypothesis verdicts)
    cost_random = group_mean_for_rows(COST_FEATURES, rd)
    cost_worst  = group_mean_for_rows(COST_FEATURES, wd)
    cost_shaw   = group_mean_for_rows(COST_FEATURES, sd)
    spatial_random = group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, rd)
    spatial_worst  = group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, wd)
    spatial_shaw   = group_mean_for_rows(SPATIAL_TEMPORAL_FEATURES, sd)

    L.append("**Hypothesis verdicts:**")
    L.append("")
    # H1: worst-destroy classifiers weight cost-features more than random-destroy
    if cost_worst > cost_random:
        L.append(f"- **H1** (worst-destroy weights cost-features > random-destroy): **supported** "
                 f"({cost_worst:.4f} vs {cost_random:.4f}, +{cost_worst - cost_random:+.4f}). "
                 f"Aligns with `worst_removal` selecting requests by cost contribution.")
    else:
        L.append(f"- **H1** (worst-destroy weights cost-features > random-destroy): **not supported** "
                 f"({cost_worst:.4f} vs {cost_random:.4f}, {cost_worst - cost_random:+.4f}). "
                 f"The cost-feature lift expected from `worst_removal` semantics does not show up in average Gini importance.")

    # H2: shaw-destroy classifiers weight spatial/temporal features more than random-destroy
    if spatial_shaw > spatial_random:
        L.append(f"- **H2** (shaw-destroy weights spatial/temporal features > random-destroy): **supported** "
                 f"({spatial_shaw:.4f} vs {spatial_random:.4f}, +{spatial_shaw - spatial_random:+.4f}). "
                 f"Aligns with `shaw_removal` selecting by spatial/temporal relatedness.")
    else:
        L.append(f"- **H2** (shaw-destroy weights spatial/temporal features > random-destroy): **not supported** "
                 f"({spatial_shaw:.4f} vs {spatial_random:.4f}, {spatial_shaw - spatial_random:+.4f}).")
    L.append("")
    L.append("Caveat: these are per-pair *binary* classifiers (target = t07 label of whether THAT pair was the chosen one at that state). The importance signal reflects which features discriminate \"this pair was picked\" from \"this pair was not picked\", not which features discriminate \"this pair would be good\" from \"would be bad\". The deployed selector's behaviour is the SOFTMAX over the 9 per-pair `P(class=1)` outputs, so feature importance is one step removed from the joint ranking the bench actually exercises.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append(f"- Per-row CSV: `{OUT_CSV.relative_to(REPO)}`")

    OUT_MD.write_text("\n".join(L))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
