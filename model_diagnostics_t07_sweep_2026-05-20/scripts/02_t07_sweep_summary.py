#!/usr/bin/env python3
"""Aggregate t07_sweep_raw.json into per-config summary tables and heatmaps.

Outputs:
  tables/t07_sweep_summary.csv         — per-config mean ± std (train/test metrics, gap)
  tables/t07_sweep_best_per_n_est.csv  — best forest cell per n_estimators
  figures/heatmap_forest_n{50,100,200}_test_mcc.png
  figures/heatmap_forest_n{50,100,200}_gap_mcc.png
  figures/heatmap_tree_test_mcc.png
  figures/heatmap_tree_gap_mcc.png
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FIG, TBL, USED_PAIRS

DEPTHS = [4, 8, 16, "none"]
KS     = [4, 8, 16, 20]
FOREST_NE = [50, 100, 200]


def main():
    TBL.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    raw = json.load(open(TBL / "t07_sweep_raw.json"))
    # Index: (tag, fold) -> list of pair metric dicts
    per_fold_pairs = defaultdict(list)
    config_meta = {}
    for e in raw["results"]:
        config_meta[e["tag"]] = (e["type"], e["n_estimators"], e["depth"], e["K"])
        for pair in USED_PAIRS:
            pm = e["pair_metrics"].get(pair, {})
            if not isinstance(pm, dict) or pm.get("degenerate_train"):
                continue
            per_fold_pairs[(e["tag"], e["fold"])].append(pm)

    # Build per-config rows: mean ± std across the 5 folds, where each
    # per-fold value is itself the mean across the 9 USED pairs.
    METRICS = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc", "brier"]
    summary_rows = []
    by_tag = defaultdict(list)
    for (tag, fold), pms in per_fold_pairs.items():
        by_tag[tag].append(pms)

    for tag in sorted(by_tag):
        type_, ne, depth, K = config_meta[tag]
        per_fold = by_tag[tag]
        row = {
            "tag": tag, "type": type_, "n_estimators": ne,
            "depth": "none" if depth is None else int(depth), "K": int(K),
            "n_folds": len(per_fold),
        }
        for split in ("train", "test"):
            for m in METRICS:
                vals = []
                for pair_dicts in per_fold:
                    fold_means = [pm.get(f"{split}_{m}") for pm in pair_dicts]
                    fold_means = [v for v in fold_means if isinstance(v, (int, float))]
                    if fold_means:
                        vals.append(float(np.mean(fold_means)))
                if vals:
                    row[f"{split}_{m}_mean"] = float(np.mean(vals))
                    row[f"{split}_{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f"{split}_{m}_mean"] = None; row[f"{split}_{m}_std"] = None
        # gap on MCC
        if row["train_mcc_mean"] is not None and row["test_mcc_mean"] is not None:
            row["gap_mcc"] = row["train_mcc_mean"] - row["test_mcc_mean"]
        else:
            row["gap_mcc"] = None
        summary_rows.append(row)

    out_path = TBL / "t07_sweep_summary.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {out_path} ({len(summary_rows)} rows)")

    # Best-of-grid per n_estimators (forest only)
    best_rows = []
    for ne in FOREST_NE:
        cells = [r for r in summary_rows if r["type"] == "forest" and r["n_estimators"] == ne]
        if not cells: continue
        cells_with_mcc = [c for c in cells if c["test_mcc_mean"] is not None]
        best = max(cells_with_mcc, key=lambda c: c["test_mcc_mean"])
        best_rows.append(best)
    bp = TBL / "t07_sweep_best_per_n_est.csv"
    with open(bp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0]))
        w.writeheader()
        for r in best_rows:
            w.writerow(r)
    print(f"Wrote {bp} ({len(best_rows)} rows)")

    # ----- Heatmaps -----
    def heatmap(rows, value_key, title, out_png, cmap="RdYlGn"):
        # rows must all share (type, n_estimators)
        # x = depth, y = K (or vice versa). Use depth on x, K on y.
        mat = np.full((len(KS), len(DEPTHS)), np.nan)
        for r in rows:
            try:
                i = KS.index(r["K"])
                j = DEPTHS.index(r["depth"])
            except ValueError:
                continue
            v = r.get(value_key)
            if v is not None:
                mat[i][j] = v
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(DEPTHS))); ax.set_xticklabels([str(d) for d in DEPTHS])
        ax.set_yticks(range(len(KS))); ax.set_yticklabels([str(k) for k in KS])
        ax.set_xlabel("depth"); ax.set_ylabel("K (features)")
        for i in range(len(KS)):
            for j in range(len(DEPTHS)):
                v = mat[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)

    # Forests: 3 panels (one per n_estimators) for test MCC and for gap
    for ne in FOREST_NE:
        rows = [r for r in summary_rows if r["type"] == "forest" and r["n_estimators"] == ne]
        heatmap(rows, "test_mcc_mean",
                f"forest n={ne}: test MCC (5-fold mean)",
                FIG / f"heatmap_forest_n{ne}_test_mcc.png", cmap="viridis")
        heatmap(rows, "gap_mcc",
                f"forest n={ne}: train-test MCC gap",
                FIG / f"heatmap_forest_n{ne}_gap_mcc.png", cmap="Reds")

    # Trees: 1 panel test MCC + 1 gap
    tree_rows = [r for r in summary_rows if r["type"] == "tree"]
    heatmap(tree_rows, "test_mcc_mean",
            "single tree: test MCC (5-fold mean)",
            FIG / "heatmap_tree_test_mcc.png", cmap="viridis")
    heatmap(tree_rows, "gap_mcc",
            "single tree: train-test MCC gap",
            FIG / "heatmap_tree_gap_mcc.png", cmap="Reds")

    print(f"Wrote {len(FOREST_NE)*2 + 2} heatmaps to {FIG}/")

    # ----- Console summary -----
    print()
    print("=== Best forest cell per n_estimators (5-fold) ===")
    print(f"{'n_est':>5} {'depth':>6} {'K':>4} {'test_MCC':>9}{'±':>3}{'std':>6} {'gap_MCC':>8}")
    for r in best_rows:
        print(f"{r['n_estimators']:>5} {r['depth']:>6} {r['K']:>4} "
              f"{r['test_mcc_mean']:>9.3f} ± {r['test_mcc_std']:>5.3f} "
              f"{r['gap_mcc']:>+8.3f}")


if __name__ == "__main__":
    main()
