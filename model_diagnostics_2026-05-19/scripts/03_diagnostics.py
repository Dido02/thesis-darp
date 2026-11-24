#!/usr/bin/env python3
"""Phase B — diagnostics, figures, and tables.

Reads from model_diagnostics_2026-05-19/enriched_reports/ and writes to:
  figures/   — PNGs (heatmaps, line plots, bar charts)
  tables/    — CSVs (long-format data feeding the report)

Produces:
  B.1 Per-config feature-importance heatmaps + per-batch contribution bars
  B.2 Top-5 features per pair CSV
  B.3 Train vs test gap heatmaps
  B.4 Tree-structure tables + tree-size-by-depth chart
  B.5 5f -> 13f -> 20f progression line plots (train + test on same axes)
  B.6 Depth sweep curves (test + train lift vs max_depth, with n_leaves on twin axis)
  Plus feature-ablation curves (test + train lift vs K)

Usage:
    python 03_diagnostics.py
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

REPO = Path(__file__).resolve().parents[2]
ENRICH = REPO / "model_diagnostics_2026-05-19" / "enriched_reports"
FIG = REPO / "model_diagnostics_2026-05-19" / "figures"
TBL = REPO / "model_diagnostics_2026-05-19" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "ml_train"))
from darpml.features import _REGISTRY  # noqa: E402

# feature_name -> batch name
BATCH_OF = {name: entry.batch for name, entry in _REGISTRY.items()}
BATCH_COLORS = {
    "core": "#1f77b4",
    "tightness": "#2ca02c",
    "worst_support": "#ff7f0e",
    "lns_state": "#9467bd",
    "rolling_horizon": "#d62728",
}
BATCH_ORDER = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]

DESTROY_OPS = ["random", "worst", "shaw", "ride_time_critical", "low_slack_route"]
REPAIR_OPS = ["greedy", "regret2", "regret3", "slack_insertion"]


def pair_to_d_r(pair: str) -> tuple[int, int]:
    """pair_3_2 -> (3, 2)"""
    _, d, r = pair.split("_")
    return int(d), int(r)


def pair_label(pair: str) -> str:
    d, r = pair_to_d_r(pair)
    return f"{DESTROY_OPS[d][:6]}+{REPAIR_OPS[r][:6]}"


def load_all_enriched() -> dict[str, dict]:
    """name -> enriched dict."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(ENRICH.glob("*.json"))}


def is_classifier(enr: dict) -> bool:
    return enr["model_kind"] == "classifier"


def primary_lift_key(enr: dict, split: str) -> str:
    return f"{split}_f1_lift" if is_classifier(enr) else f"{split}_mae_lift"


# ---------- B.1 + B.2: feature importance heatmaps + per-batch contribution ----------

def feature_importance_matrix(enr: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Pair x feature matrix of importance (forest: mean Gini; tree: also Gini).

    For both we read meta-style importances. We re-load joblib only if needed,
    but in the enriched report we don't have per-pair Gini importance — we have
    feature_use_count. So we load the joblib for Gini importance.
    """
    pair_names = [p["pair_name"] for p in enr["pairs"]]
    feat_names = enr["feature_names"]
    mat = np.zeros((len(pair_names), len(feat_names)))
    model_dir = REPO / enr["model_dir"]
    for i, pair in enumerate(pair_names):
        import joblib
        clf = joblib.load(model_dir / f"rf_{pair}.joblib")
        mat[i, :] = clf.feature_importances_
    return mat, pair_names, feat_names


def feature_use_matrix(enr: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Pair x feature matrix of split-node use counts. Only meaningful for tree models."""
    pair_names = [p["pair_name"] for p in enr["pairs"]]
    feat_names = enr["feature_names"]
    mat = np.zeros((len(pair_names), len(feat_names)), dtype=int)
    for i, p in enumerate(enr["pairs"]):
        struct = p.get("tree_structure")
        if struct is None:
            # Forest aggregate fallback: read forest_structure.feature_use_count_total
            f = p.get("forest_structure", {})
            mat[i, :] = f.get("feature_use_count_total", [0] * len(feat_names))
        else:
            mat[i, :] = struct["feature_use_count"]
    return mat, pair_names, feat_names


def _plot_heatmap(mat, pair_names, feat_names, title, out_path, cmap="viridis",
                  vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(0.45 * len(feat_names) + 2, 0.32 * len(pair_names) + 2))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(feat_names)))
    # color tick labels by batch
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    for tick, name in zip(ax.get_xticklabels(), feat_names):
        tick.set_color(BATCH_COLORS.get(BATCH_OF.get(name, "core"), "black"))
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels([pair_label(p) for p in pair_names], fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def emit_importance_artifacts(all_enr: dict[str, dict]) -> None:
    """For each variant, emit a Gini importance heatmap + split-count heatmap
    + per-batch contribution bar chart.

    Restrict to v3 forests and v6 trees (the canonical fleet). depth_sweep /
    feat_ablation get their own dedicated charts in dedicated functions.
    """
    canonical = {n: e for n, e in all_enr.items()
                 if n.startswith("models_v3_full_") or n.startswith("models_v6_full_")}
    print(f"  B.1: importance heatmaps for {len(canonical)} canonical variants")

    rows_top5 = []
    rows_batch = []
    for name, enr in canonical.items():
        try:
            gini_mat, pair_names, feat_names = feature_importance_matrix(enr)
            use_mat, _, _ = feature_use_matrix(enr)
        except Exception as e:
            print(f"    skip {name}: {e}")
            continue

        # Heatmaps
        _plot_heatmap(
            gini_mat, pair_names, feat_names,
            title=f"{name} — Gini importance (rows: pair, cols: feature)",
            out_path=FIG / f"importance_{name}.png",
            cmap="viridis",
        )
        # split-count: log scale because trees have 0 vs hundreds
        log_use = np.log1p(use_mat.astype(float))
        _plot_heatmap(
            log_use, pair_names, feat_names,
            title=f"{name} — log(1 + split-count) — splits testing this feature",
            out_path=FIG / f"use_count_{name}.png",
            cmap="magma",
        )

        # Per-batch contribution: % of total Gini summed across pairs
        per_batch = {b: 0.0 for b in BATCH_ORDER}
        for j, fn in enumerate(feat_names):
            b = BATCH_OF.get(fn, "core")
            per_batch[b] += gini_mat[:, j].sum()
        total = sum(per_batch.values())
        if total > 0:
            per_batch = {b: v / total for b, v in per_batch.items()}
        rows_batch.append({"variant": name, **per_batch})

        # Top-5 features per pair
        for i, pair in enumerate(pair_names):
            ordered = np.argsort(-gini_mat[i, :])[:5]
            for rank, j in enumerate(ordered, 1):
                rows_top5.append({
                    "variant": name,
                    "pair": pair,
                    "rank": rank,
                    "feature": feat_names[j],
                    "batch": BATCH_OF.get(feat_names[j], "?"),
                    "gini_importance": float(gini_mat[i, j]),
                    "split_count": int(use_mat[i, j]),
                })

    # Batch contribution bar chart (canonical variants stacked)
    if rows_batch:
        labels = [r["variant"].replace("models_", "") for r in rows_batch]
        x = np.arange(len(rows_batch))
        bottom = np.zeros(len(rows_batch))
        fig, ax = plt.subplots(figsize=(max(8, 0.30 * len(rows_batch)), 5))
        for b in BATCH_ORDER:
            vals = np.array([r[b] for r in rows_batch])
            ax.bar(x, vals, bottom=bottom, color=BATCH_COLORS[b], label=b, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("share of total Gini importance")
        ax.set_title("Per-batch feature-importance contribution, canonical variants")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "batch_contribution_canonical.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        with open(TBL / "batch_contribution.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant"] + BATCH_ORDER)
            w.writeheader()
            for r in rows_batch:
                w.writerow(r)

    if rows_top5:
        with open(TBL / "top_features_per_pair.csv", "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["variant", "pair", "rank", "feature",
                               "batch", "gini_importance", "split_count"],
            )
            w.writeheader()
            for r in rows_top5:
                w.writerow(r)


# ---------- B.3: train vs test gap heatmap ----------

def emit_gap_heatmaps(all_enr: dict[str, dict]) -> None:
    """One heatmap per variant of (train_lift - test_lift) across 20 pairs."""
    rows = []
    canonical = {n: e for n, e in all_enr.items()
                 if n.startswith("models_v3_full_") or n.startswith("models_v6_full_")}
    print(f"  B.3: gap heatmaps for {len(canonical)} canonical variants")

    for name, enr in canonical.items():
        pair_names = [p["pair_name"] for p in enr["pairs"]]
        train_lifts = np.array([p.get(primary_lift_key(enr, "train"), np.nan) for p in enr["pairs"]])
        test_lifts = np.array([p.get(primary_lift_key(enr, "test"), np.nan) for p in enr["pairs"]])
        gap = train_lifts - test_lifts
        # 5x4 grid (destroy x repair)
        grid = np.full((5, 4), np.nan)
        for i, p in enumerate(pair_names):
            d, r = pair_to_d_r(p)
            grid[d, r] = gap[i]
            rows.append({
                "variant": name,
                "pair": p,
                "destroy": DESTROY_OPS[d],
                "repair": REPAIR_OPS[r],
                "train_lift": float(train_lifts[i]) if not np.isnan(train_lifts[i]) else "",
                "test_lift": float(test_lifts[i]) if not np.isnan(test_lifts[i]) else "",
                "gap": float(gap[i]) if not np.isnan(gap[i]) else "",
                "model_kind": enr["model_kind"],
            })

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
                       vmin=-0.5, vmax=0.5 if is_classifier(enr) else 0.2)
        ax.set_xticks(range(4)); ax.set_xticklabels(REPAIR_OPS, rotation=30, ha="right")
        ax.set_yticks(range(5)); ax.set_yticklabels(DESTROY_OPS)
        for i in range(5):
            for j in range(4):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center",
                            fontsize=8, color="black")
        metric = "F1" if is_classifier(enr) else "MAE"
        ax.set_title(f"{name}\ntrain_{metric}_lift − test_{metric}_lift   (red = overfit)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG / f"gap_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(TBL / "overfit_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["variant", "pair", "destroy", "repair",
                           "train_lift", "test_lift", "gap", "model_kind"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- B.4: tree-structure summary + tree-size-by-depth chart ----------

def emit_tree_structure(all_enr: dict[str, dict]) -> None:
    print("  B.4: tree-structure summary")
    rows = []
    # All variants with tree_structure or forest_structure
    for name, enr in all_enr.items():
        depths, leaves, nodes, feats_used, frac_lt5 = [], [], [], [], []
        for p in enr["pairs"]:
            ts = p.get("tree_structure")
            if ts is not None:
                depths.append(ts["max_depth"])
                leaves.append(ts["n_leaves"])
                nodes.append(ts["node_count"])
                feats_used.append(ts["n_unique_features_used"])
                frac_lt5.append(ts["leaf_samples"]["frac_lt_5"])
            else:
                fs = p.get("forest_structure", {})
                if not fs:
                    continue
                # Use medians across the 100 estimators
                depths.append(fs["max_depth"]["median"])
                leaves.append(fs["n_leaves"]["median"])
                nodes.append(fs["node_count"]["median"])
                feats_used.append(fs["n_unique_features_used"]["median"])
                frac_lt5.append(np.nan)
        if not depths:
            continue
        rows.append({
            "variant": name,
            "model_kind": enr["model_kind"],
            "model_type": enr["model_type"],
            "n_features": enr["n_features"],
            "median_depth": float(np.median(depths)),
            "median_n_leaves": float(np.median(leaves)),
            "median_node_count": float(np.median(nodes)),
            "median_unique_feats_used": float(np.median(feats_used)),
            "mean_frac_leaves_lt_5_samples": float(np.nanmean(frac_lt5)) if any(~np.isnan(frac_lt5)) else "",
        })
    with open(TBL / "tree_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Tree-size-by-depth chart (uses the depth_sweep dirs)
    depth_rows = {  # config_target -> {depth -> (median_n_leaves, median_depth, median_node_count)}
        "20f_t07": {},
        "20f_regr": {},
        "20f_ei_log": {},
    }
    for name, enr in all_enr.items():
        if not name.startswith("depth"):
            continue
        # name like 'depth4_20f_t07_tree'
        prefix, *rest = name.split("_", 1)
        depth_lbl = prefix[5:]   # "4" or "none"
        config = name[len(prefix) + 1:].replace("_tree", "")  # 20f_t07 / 20f_regr / 20f_ei_log
        if config not in depth_rows:
            continue
        leaves = [p.get("tree_structure", {}).get("n_leaves") for p in enr["pairs"]]
        depths = [p.get("tree_structure", {}).get("max_depth") for p in enr["pairs"]]
        nodes = [p.get("tree_structure", {}).get("node_count") for p in enr["pairs"]]
        depth_rows[config][depth_lbl] = (
            float(np.median([l for l in leaves if l is not None])),
            float(np.median([d for d in depths if d is not None])),
            float(np.median([n for n in nodes if n is not None])),
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (config, by_depth) in zip(axes, depth_rows.items()):
        depth_order = ["2", "4", "8", "16", "none"]
        x = np.arange(len(depth_order))
        leaves = [by_depth.get(d, (np.nan,))[0] for d in depth_order]
        depths_med = [by_depth.get(d, (np.nan, np.nan))[1] for d in depth_order]
        nodes = [by_depth.get(d, (np.nan, np.nan, np.nan))[2] for d in depth_order]
        ax.bar(x - 0.2, leaves, 0.4, label="n_leaves (median)", color="#1f77b4")
        ax.bar(x + 0.2, nodes, 0.4, label="node_count (median)", color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(depth_order)
        ax.set_xlabel("max_depth")
        ax.set_title(f"{config}: tree growth by max_depth")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
        # Twin axis for the actual depth reached
        ax2 = ax.twinx()
        ax2.plot(x, depths_med, "o--", color="black", label="actual median depth")
        ax2.set_ylabel("actual median depth")
        ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "tree_size_by_depth.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- B.5: 5f -> 13f -> 20f progression ----------

def median_lift(enr: dict, split: str) -> float:
    """Median F1-lift (classifiers) or MAE-lift (regressors) across non-degenerate pairs."""
    key = primary_lift_key(enr, split)
    vals = []
    for p in enr["pairs"]:
        if is_classifier(enr):
            n = p.get("test_n", 0)
            pos = p.get("test_n_positive", 0)
            if n == 0 or pos / n < 0.05 or pos / n > 0.95:
                continue
        else:
            std = p.get("train_label_std")
            if std is None or std < 1e-3:
                continue
        v = p.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def emit_feature_count_progression(all_enr: dict[str, dict]) -> None:
    print("  B.5: 5f -> 13f -> 20f progression")
    rows = []
    targets = ["t02", "t03", "t05", "t07", "regr", "ei_log"]
    for target in targets:
        for variant_type in ("forest", "tree"):
            prefix = "models_v3_full_" if variant_type == "forest" else "models_v6_full_"
            suffix = "" if variant_type == "forest" else "_tree"
            data = {}
            for nf in ("5f", "13f", "20f"):
                if target in ("ei_log", "ei") and nf != "20f":
                    continue  # only 20f variants exist for ei
                name = f"{prefix}{nf}_{target}{suffix}"
                if name not in all_enr:
                    continue
                enr = all_enr[name]
                data[nf] = (median_lift(enr, "train"), median_lift(enr, "test"))
                rows.append({
                    "target": target,
                    "variant_type": variant_type,
                    "n_feat": nf,
                    "median_train_lift": data[nf][0],
                    "median_test_lift": data[nf][1],
                })
            if data:
                pass  # plot is emitted below per-target

    with open(TBL / "feature_count_progression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "variant_type", "n_feat",
                                          "median_train_lift", "median_test_lift"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Per-target plot
    for target in targets:
        if target in ("ei_log", "ei"):
            continue  # only 20f exists; nothing to plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for variant_type, color, marker in (("forest", "#1f77b4", "o"),
                                            ("tree", "#d62728", "s")):
            prefix = "models_v3_full_" if variant_type == "forest" else "models_v6_full_"
            suffix = "" if variant_type == "forest" else "_tree"
            train = []; test = []; xs = []
            for nf in ("5f", "13f", "20f"):
                name = f"{prefix}{nf}_{target}{suffix}"
                if name not in all_enr:
                    continue
                xs.append(int(nf.rstrip("f")))
                train.append(median_lift(all_enr[name], "train"))
                test.append(median_lift(all_enr[name], "test"))
            if xs:
                ax.plot(xs, test, color=color, marker=marker, linewidth=2,
                        label=f"{variant_type} (test)")
                ax.plot(xs, train, color=color, marker=marker, linewidth=1, linestyle="--",
                        alpha=0.5, label=f"{variant_type} (train)")
        ylab = "median F1-lift" if target in ("t02", "t03", "t05", "t07") else "median MAE-lift"
        ax.set_xticks([5, 13, 20])
        ax.set_xlabel("number of features")
        ax.set_ylabel(ylab)
        ax.set_title(f"{target}: feature-count progression (train dashed, test solid)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"feature_count_progression_{target}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)


# ---------- B.6: depth sweep curves ----------

def emit_depth_sweep_curves(all_enr: dict[str, dict]) -> None:
    print("  B.6: depth-sweep curves")
    rows = []
    depth_order_lbl = ["2", "4", "8", "16", "none"]
    for target in ("20f_t07", "20f_regr", "20f_ei_log"):
        ys_train = []; ys_test = []; leaves = []
        baseline_test = float("nan")
        baseline_train = float("nan")
        baseline_leaves = float("nan")
        canonical = f"models_v6_full_{target}_tree"
        if canonical in all_enr:
            baseline_test = median_lift(all_enr[canonical], "test")
            baseline_train = median_lift(all_enr[canonical], "train")
            # leaves on canonical
            lvs = [p.get("tree_structure", {}).get("n_leaves") for p in all_enr[canonical]["pairs"]]
            baseline_leaves = float(np.median([l for l in lvs if l is not None]))
        for d in depth_order_lbl:
            name = f"depth{d}_{target}_tree"
            if name not in all_enr:
                ys_train.append(float("nan")); ys_test.append(float("nan"))
                leaves.append(float("nan"))
                continue
            ys_train.append(median_lift(all_enr[name], "train"))
            ys_test.append(median_lift(all_enr[name], "test"))
            lvs = [p.get("tree_structure", {}).get("n_leaves") for p in all_enr[name]["pairs"]]
            leaves.append(float(np.median([l for l in lvs if l is not None])))
            rows.append({
                "target": target,
                "depth": d,
                "median_train_lift": ys_train[-1],
                "median_test_lift": ys_test[-1],
                "median_n_leaves": leaves[-1],
            })

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(depth_order_lbl))
        ax.plot(x, ys_test, "o-", color="#d62728", linewidth=2.2, label="test (depth-controlled)")
        ax.plot(x, ys_train, "o--", color="#d62728", linewidth=1.2, alpha=0.55,
                label="train (depth-controlled)")
        ax.axhline(baseline_test, color="grey", linestyle=":", linewidth=1.6,
                   label=f"existing v6 tree (test={baseline_test:.4f})")
        ax.set_xticks(x); ax.set_xticklabels(depth_order_lbl)
        ax.set_xlabel("max_depth")
        ylab = "median F1-lift" if target == "20f_t07" else "median MAE-lift"
        ax.set_ylabel(ylab)
        ax.set_title(f"{target} depth sweep — test lift vs max_depth")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(x, leaves, "s-", color="#1f77b4", alpha=0.6,
                 label="median n_leaves")
        ax2.set_ylabel("median n_leaves (log)", color="#1f77b4")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"depth_sweep_{target}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    with open(TBL / "depth_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "depth", "median_train_lift",
                                          "median_test_lift", "median_n_leaves"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- Feature ablation curves ----------

def emit_feat_ablation_curves(all_enr: dict[str, dict]) -> None:
    print("  B.5b: feat-ablation curves")
    rows = []
    K_order = [3, 5, 8, 12]
    K_extended = K_order + [20]   # 20 = the full 20f tree
    for target_label, canonical_name in (("t07", "models_v6_full_20f_t07_tree"),
                                         ("regr", "models_v6_full_20f_regr_tree")):
        ys_train = []; ys_test = []; xs = []
        for K in K_order:
            name = f"topK{K}_{target_label}_tree"
            if name not in all_enr:
                continue
            xs.append(K)
            ys_train.append(median_lift(all_enr[name], "train"))
            ys_test.append(median_lift(all_enr[name], "test"))
            rows.append({
                "target": target_label, "K": K,
                "median_train_lift": ys_train[-1],
                "median_test_lift": ys_test[-1],
                "features_used": ", ".join(
                    json.loads((REPO / "model_diagnostics_2026-05-19"
                               / "feat_ablation_models" / name / "meta.json").read_text())
                    .get("topK_features_ordered", [])
                ),
            })
        # Add K=20 (canonical full tree)
        if canonical_name in all_enr:
            xs.append(20)
            ys_train.append(median_lift(all_enr[canonical_name], "train"))
            ys_test.append(median_lift(all_enr[canonical_name], "test"))
            rows.append({
                "target": target_label, "K": 20,
                "median_train_lift": ys_train[-1],
                "median_test_lift": ys_test[-1],
                "features_used": "all 20 features",
            })

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys_test, "o-", color="#2ca02c", linewidth=2.2, label="test")
        ax.plot(xs, ys_train, "o--", color="#2ca02c", linewidth=1.2, alpha=0.55, label="train")
        ax.set_xticks(K_extended)
        ax.set_xlabel("number of features (top-K by 20f-forest Gini importance)")
        ylab = "median F1-lift" if target_label == "t07" else "median MAE-lift"
        ax.set_ylabel(ylab)
        ax.set_title(f"{target_label} feature-ablation — test lift vs K")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"feat_ablation_curve_{target_label}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(TBL / "feat_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "K", "median_train_lift",
                                          "median_test_lift", "features_used"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    print(f"Reading enriched reports from {ENRICH}")
    all_enr = load_all_enriched()
    print(f"  loaded {len(all_enr)} enriched reports")

    emit_importance_artifacts(all_enr)
    emit_gap_heatmaps(all_enr)
    emit_tree_structure(all_enr)
    emit_feature_count_progression(all_enr)
    emit_depth_sweep_curves(all_enr)
    emit_feat_ablation_curves(all_enr)

    print(f"\nFigures in {FIG}")
    print(f"Tables in {TBL}")


if __name__ == "__main__":
    main()
