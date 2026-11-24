#!/usr/bin/env python3
"""Render full views of the best single tree (depth=4, K=4) for the 9 USED pairs.

The tree at depth 4 is small enough to display in full without max_depth
truncation. Renders one PNG per pair plus a console-friendly text dump.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CACHE, FEATURES_PQ, LABELS_PQ, FIG, USED_PAIRS, aggregated_top_k,
)

NAMES = {
    "pair_0_0": "random+greedy", "pair_0_1": "random+regret2", "pair_0_2": "random+regret3",
    "pair_1_0": "worst+greedy",  "pair_1_1": "worst+regret2",  "pair_1_2": "worst+regret3",
    "pair_2_0": "shaw+greedy",   "pair_2_1": "shaw+regret2",   "pair_2_2": "shaw+regret3",
}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PQ).merge(
        pd.read_parquet(LABELS_PQ), on=["log_file", "iter"], how="inner")
    feat_names = aggregated_top_k(4)
    print(f"Top-4 features used: {feat_names}")
    X = df[feat_names].values
    for pair in USED_PAIRS:
        y = df[pair].values
        if y.sum() < 2 or len(np.unique(y)) < 2:
            print(f"  {pair} ({NAMES[pair]}): degenerate; skipping")
            continue
        clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                     random_state=42)
        clf.fit(X, y)
        prev = float(y.mean())
        depth = clf.get_depth(); n_leaves = clf.tree_.n_leaves
        fig, ax = plt.subplots(figsize=(13, 7))
        plot_tree(clf, max_depth=4, filled=True, feature_names=feat_names,
                  class_names=["no improve", "improve"], rounded=True,
                  fontsize=8, ax=ax)
        ax.set_title(f"{pair} ({NAMES[pair]}) — tree d=4 K=4 full view  "
                     f"(prevalence {prev:.2f}, depth={depth}, leaves={n_leaves})",
                     fontsize=10)
        fig.tight_layout()
        out = FIG / f"best_tree_{pair}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  {pair} ({NAMES[pair]}): leaves={n_leaves}, depth={depth} -> {out.name}")
        # ASCII dump for grep-friendly review
        ascii_dump = export_text(clf, feature_names=feat_names, max_depth=4,
                                  show_weights=True)
        (FIG / f"best_tree_{pair}.txt").write_text(ascii_dump)


if __name__ == "__main__":
    main()
