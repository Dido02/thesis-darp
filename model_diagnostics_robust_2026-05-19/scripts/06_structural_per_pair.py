#!/usr/bin/env python3
"""Per-pair structural deep-dive.

For each of the 9 USED pairs on t03:
  - Fit one full-data RandomForestClassifier(n_estimators=100, max_depth=None,
    class_weight='balanced', random_state=42).
  - Pick the largest-by-node-count estimator as the representative tree.
  - Extract top-3 BFS splits (feature, threshold, samples, Gini drop).
  - Render top-3-depth tree PNG via sklearn.tree.plot_tree.
  - Forest-level top-5 features by importance.

Outputs:
  tables/structural_top3_splits.csv
  tables/structural_top_features.csv
  tables/structural_prose_hints.txt
  figures/tree_{pair}.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURE_NAMES_20, FIG, TARGETS, TBL, USED_PAIRS,
)

TARGET = "t03"

# Prose-hint matrices (used to seed the per-pair narrative in the report)
SHAW_FEATURES = {"radius_avg_tt", "cost_contribution_std",
                 "near_violation_count", "route_slack_std",
                 "ride_time_utilization_avg", "ride_time_utilization_max"}
COST_FEATURES = {"cost_contribution_max_ratio", "route_cost_std",
                 "cost_vs_initial_frac", "cost_vs_incumbent_frac",
                 "cost_contribution_std"}
TEMPORAL_FEATURES = {"iteration_norm", "temperature_normalized",
                     "horizon_elapsed_frac", "frac_committed",
                     "committed_travel_time_frac"}
LOAD_FEATURES = {"unassigned_count", "num_active_requests", "avg_route_load",
                 "neighborhood_size_frac"}

PAIR_DOMAIN = {
    "pair_0_0": ("random", "greedy"),
    "pair_0_1": ("random", "regret2"),
    "pair_0_2": ("random", "regret3"),
    "pair_1_0": ("worst",  "greedy"),
    "pair_1_1": ("worst",  "regret2"),
    "pair_1_2": ("worst",  "regret3"),
    "pair_2_0": ("shaw",   "greedy"),
    "pair_2_1": ("shaw",   "regret2"),
    "pair_2_2": ("shaw",   "regret3"),
}


def bfs_top3(tree, feature_names: list[str]) -> list[dict]:
    """Top-3 BFS splits with feature, threshold, samples, gini_drop."""
    t = tree.tree_
    queue = [0]
    out = []
    while queue and len(out) < 3:
        node = queue.pop(0)
        if t.children_left[node] == -1:
            continue
        f_idx = int(t.feature[node])
        thr = float(t.threshold[node])
        n_node = int(t.n_node_samples[node])
        L = int(t.children_left[node])
        R = int(t.children_right[node])
        n_L = int(t.n_node_samples[L])
        n_R = int(t.n_node_samples[R])
        imp_node = float(t.impurity[node])
        imp_L = float(t.impurity[L])
        imp_R = float(t.impurity[R])
        gini_drop = imp_node - (n_L * imp_L + n_R * imp_R) / max(1, n_node)
        out.append({
            "feature_idx": f_idx,
            "feature_name": feature_names[f_idx],
            "threshold": thr,
            "samples_at_node": n_node,
            "gini_drop": gini_drop,
            "impurity_at_node": imp_node,
            "n_left": n_L, "n_right": n_R,
        })
        queue.append(L); queue.append(R)
    return out


def categorize_features(feature_names: set[str]) -> dict:
    return {
        "shaw_relatedness": sorted(feature_names & SHAW_FEATURES),
        "cost_driven": sorted(feature_names & COST_FEATURES),
        "temporal_progress": sorted(feature_names & TEMPORAL_FEATURES),
        "load_state": sorted(feature_names & LOAD_FEATURES),
        "other": sorted(feature_names - SHAW_FEATURES - COST_FEATURES
                        - TEMPORAL_FEATURES - LOAD_FEATURES),
    }


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading parquets...")
    features_df = pd.read_parquet(CACHE / "features.parquet")
    labels_df = pd.read_parquet(CACHE / TARGETS[TARGET])
    df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")
    X = df[FEATURE_NAMES_20].values

    splits_rows = []
    feats_rows = []
    prose_lines = [
        f"# Per-pair structural evidence (target={TARGET}, forest full-data fit)",
        f"# pair: (destroy, repair)",
        "",
    ]

    for pair in USED_PAIRS:
        destroy, repair = PAIR_DOMAIN[pair]
        y = df[pair].values
        prev = float(y.mean())
        n_pos = int(y.sum())
        if y.sum() < 2 or len(np.unique(y)) < 2:
            prose_lines.append(f"## {pair} ({destroy}+{repair}) — DEGENERATE "
                               f"(n_positive={n_pos}/{len(y)}={prev:.3f})")
            prose_lines.append("")
            continue
        print(f"  fitting {pair} ({destroy}+{repair})...")
        clf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1, max_depth=None,
        )
        clf.fit(X, y)
        importances = clf.feature_importances_
        order = np.argsort(-importances)
        top5 = [(FEATURE_NAMES_20[i], float(importances[i])) for i in order[:5]]
        for rank, (fn, imp) in enumerate(top5, start=1):
            feats_rows.append({"pair": pair, "rank": rank,
                               "feature": fn, "importance": imp})

        # Pick largest-by-node-count estimator
        sizes = [t.tree_.node_count for t in clf.estimators_]
        rep_idx = int(np.argmax(sizes))
        rep_tree = clf.estimators_[rep_idx]
        rep_size = sizes[rep_idx]
        rep_depth = rep_tree.get_depth()
        rep_leaves = rep_tree.tree_.n_leaves
        top3 = bfs_top3(rep_tree, FEATURE_NAMES_20)
        for i, s in enumerate(top3):
            splits_rows.append({
                "pair": pair, "destroy": destroy, "repair": repair,
                "bfs_idx": i,
                **s,
            })

        # Render top-3-depth tree
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_tree(rep_tree, max_depth=3, filled=True,
                  feature_names=FEATURE_NAMES_20,
                  class_names=["negative", "positive"],
                  rounded=True, fontsize=8, ax=ax)
        ax.set_title(f"{pair} ({destroy}+{repair}) — representative tree "
                     f"(estimator {rep_idx}/100, depth={rep_depth}, "
                     f"leaves={rep_leaves}, nodes={rep_size}) — top-3 depth view")
        fig.tight_layout()
        out = FIG / f"tree_{pair}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)

        # Prose hints
        prose_lines.append(f"## {pair} ({destroy}+{repair})")
        prose_lines.append(f"  prevalence: {prev:.3f}  ({n_pos}/{len(y)} positive)")
        prose_lines.append(f"  representative tree: estimator {rep_idx}, "
                           f"depth={rep_depth}, leaves={rep_leaves}, nodes={rep_size}")
        prose_lines.append(f"  forest top-5 features (Gini importance):")
        for rank, (fn, imp) in enumerate(top5, start=1):
            prose_lines.append(f"    {rank}. {fn} ({imp:.3f})")
        prose_lines.append(f"  top-3 BFS splits in representative tree:")
        for i, s in enumerate(top3):
            prose_lines.append(
                f"    {i}: {s['feature_name']:<32} "
                f"<= {s['threshold']:+.4f}   "
                f"n={s['samples_at_node']}  "
                f"Gini drop={s['gini_drop']:.4f}"
            )
        # Feature category breakdown of top-5
        cats = categorize_features({fn for fn, _ in top5})
        cat_summary = []
        for label, lst in cats.items():
            if lst:
                cat_summary.append(f"{label}({len(lst)})")
        prose_lines.append(f"  top-5 categories: {', '.join(cat_summary)}")
        # Per-pair domain check
        if destroy == "shaw":
            n_shaw = sum(1 for fn, _ in top5 if fn in SHAW_FEATURES)
            prose_lines.append(f"  domain check: {n_shaw}/5 top features are "
                               f"Shaw-relatedness — {'plausible' if n_shaw >= 2 else 'unexpected'}")
        elif destroy == "worst":
            n_cost = sum(1 for fn, _ in top5 if fn in COST_FEATURES)
            prose_lines.append(f"  domain check: {n_cost}/5 top features are "
                               f"cost-driven — {'plausible' if n_cost >= 2 else 'unexpected'}")
        elif destroy == "random":
            n_temp = sum(1 for fn, _ in top5 if fn in TEMPORAL_FEATURES)
            prose_lines.append(f"  domain check: {n_temp}/5 top features are "
                               f"temporal-progress — random+greedy is expected to lean "
                               f"on temperature/iteration if it works")
        prose_lines.append("")

    # Write tables
    out_splits = TBL / "structural_top3_splits.csv"
    with open(out_splits, "w", newline="") as f:
        if splits_rows:
            w = csv.DictWriter(f, fieldnames=list(splits_rows[0]))
            w.writeheader()
            for r in splits_rows:
                w.writerow(r)
    print(f"Wrote {out_splits} ({len(splits_rows)} rows)")

    out_feats = TBL / "structural_top_features.csv"
    with open(out_feats, "w", newline="") as f:
        if feats_rows:
            w = csv.DictWriter(f, fieldnames=list(feats_rows[0]))
            w.writeheader()
            for r in feats_rows:
                w.writerow(r)
    print(f"Wrote {out_feats} ({len(feats_rows)} rows)")

    out_prose = TBL / "structural_prose_hints.txt"
    out_prose.write_text("\n".join(prose_lines))
    print(f"Wrote {out_prose}")


if __name__ == "__main__":
    main()
