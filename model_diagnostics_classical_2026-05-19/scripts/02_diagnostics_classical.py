#!/usr/bin/env python3
"""Classical-only diagnostics — filter every aggregate to the 9 USED pairs.

USED pairs (destroy in {random,worst,shaw} x repair in {greedy,regret2,regret3}):
    pair_0_0, pair_0_1, pair_0_2,
    pair_1_0, pair_1_1, pair_1_2,
    pair_2_0, pair_2_1, pair_2_2

Reads from BOTH:
  model_diagnostics_2026-05-19/enriched_reports/        (34 + 15 + 8 existing)
  model_diagnostics_classical_2026-05-19/enriched_reports_classical/ (16 new)

Emits:
  figures/  — gap heatmaps (3x3 grid for 9 used pairs), depth-sweep curves,
              feat-ablation curves (agg vs union), feature-count progression,
              classical metrics bar charts
  tables/   — classical_metrics.csv, depth_sweep_used.csv, feat_ablation_used.csv,
              top_features_per_used_pair.csv, plus the two top-K tables from
              01_train_classical_topK.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OLD_ENRICH = REPO / "model_diagnostics_2026-05-19" / "enriched_reports"
NEW_ENRICH = REPO / "model_diagnostics_classical_2026-05-19" / "enriched_reports_classical"
FIG = REPO / "model_diagnostics_classical_2026-05-19" / "figures"
TBL = REPO / "model_diagnostics_classical_2026-05-19" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "ml_train"))
from darpml.features import _REGISTRY  # noqa: E402

BATCH_OF = {name: entry.batch for name, entry in _REGISTRY.items()}
BATCH_COLORS = {
    "core": "#1f77b4", "tightness": "#2ca02c", "worst_support": "#ff7f0e",
    "lns_state": "#9467bd", "rolling_horizon": "#d62728",
}
BATCH_ORDER = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]

DESTROY_OPS = ["random", "worst", "shaw"]   # USED only
REPAIR_OPS = ["greedy", "regret2", "regret3"]   # USED only
USED_PAIRS = {f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)}


def load_all_enriched() -> dict[str, dict]:
    out = {}
    for src in (OLD_ENRICH, NEW_ENRICH):
        if not src.exists():
            continue
        for p in sorted(src.glob("*.json")):
            out[p.stem] = json.loads(p.read_text())
    return out


def is_classifier(enr: dict) -> bool:
    return enr["model_kind"] == "classifier"


def primary_lift_key(enr: dict, split: str) -> str:
    return f"{split}_f1_lift" if is_classifier(enr) else f"{split}_mae_lift"


def used_pairs_only(enr: dict) -> list[dict]:
    return [p for p in enr["pairs"] if p["pair_name"] in USED_PAIRS]


def median_metric(pair_list: list[dict], key: str) -> float:
    vs = [p.get(key) for p in pair_list if isinstance(p.get(key), (int, float))]
    return float(st.median(vs)) if vs else float("nan")


# ============== Classical metrics CSV (used-pairs only) ==============

def emit_classical_metrics(all_enr: dict[str, dict]) -> None:
    print("  emitting classical metrics CSV + summary plot")
    metric_keys = ["accuracy", "precision", "recall", "specificity",
                   "f1", "f1_lift", "mcc", "roc_auc", "pr_auc", "brier"]
    reg_keys = ["mae", "rmse", "r2", "mae_lift", "baseline_mae"]

    rows = []
    for name, enr in all_enr.items():
        used = used_pairs_only(enr)
        if not used:
            continue
        is_clf = is_classifier(enr)
        keys = metric_keys if is_clf else reg_keys
        for split in ("train", "test"):
            d = {"variant": name, "model_kind": enr["model_kind"],
                 "model_type": enr["model_type"], "split": split,
                 "n_used_pairs": len(used)}
            for k in keys:
                d[k] = median_metric(used, f"{split}_{k}")
            rows.append(d)
    # write
    if rows:
        all_cols = set()
        for r in rows:
            all_cols.update(r)
        cols = ["variant", "model_kind", "model_type", "split", "n_used_pairs",
                "accuracy", "precision", "recall", "specificity",
                "f1", "f1_lift", "mcc", "roc_auc", "pr_auc", "brier",
                "mae", "rmse", "r2", "mae_lift", "baseline_mae"]
        cols = [c for c in cols if c in all_cols]
        with open(TBL / "classical_metrics_used.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


# ============== Per-pair gap heatmaps (3x3 grids) ==============

def pair_to_dr(pair: str) -> tuple[int, int]:
    _, d, r = pair.split("_")
    return int(d), int(r)


def emit_gap_heatmaps(all_enr: dict[str, dict]) -> None:
    print("  emitting per-variant gap heatmaps (USED 9, 3x3 grid)")
    rows = []
    canonical = {n: e for n, e in all_enr.items()
                 if n.startswith("models_v3_full_") or n.startswith("models_v6_full_")
                 or n.startswith("topK") or n.startswith("depth")}
    for name, enr in canonical.items():
        used = used_pairs_only(enr)
        if not used:
            continue
        is_clf = is_classifier(enr)
        grid = np.full((3, 3), np.nan)
        for p in used:
            d, r = pair_to_dr(p["pair_name"])
            if d > 2 or r > 2:
                continue
            tk = primary_lift_key(enr, "train")
            tek = primary_lift_key(enr, "test")
            train_v = p.get(tk)
            test_v = p.get(tek)
            if isinstance(train_v, (int, float)) and isinstance(test_v, (int, float)):
                grid[d, r] = train_v - test_v
                rows.append({
                    "variant": name, "pair": p["pair_name"],
                    "destroy": DESTROY_OPS[d], "repair": REPAIR_OPS[r],
                    "train_lift": train_v, "test_lift": test_v,
                    "gap": train_v - test_v, "model_kind": enr["model_kind"],
                })

        fig, ax = plt.subplots(figsize=(4.3, 4.3))
        vmax = 0.5 if is_clf else 0.2
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels(REPAIR_OPS, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(DESTROY_OPS)
        for i in range(3):
            for j in range(3):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center",
                            fontsize=9, color="black")
        m = "F1" if is_clf else "MAE"
        ax.set_title(f"{name}\ntrain_{m}_lift − test_{m}_lift  (USED 9)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG / f"gap_used_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(TBL / "overfit_gap_used.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "pair", "destroy", "repair",
                                          "train_lift", "test_lift", "gap", "model_kind"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============== Feature importance & per-pair top-5 (USED only) ==============

def emit_importance(all_enr: dict[str, dict]) -> None:
    print("  emitting feature-importance heatmaps + per-batch contribution (USED 9)")
    canonical = {n: e for n, e in all_enr.items()
                 if n.startswith("models_v3_full_20f_") or n.startswith("models_v6_full_20f_")}

    top5_rows = []
    batch_rows = []

    for name, enr in canonical.items():
        used = used_pairs_only(enr)
        if not used:
            continue
        pair_names = [p["pair_name"] for p in used]
        feat_names = enr["feature_names"]
        model_dir = REPO / enr["model_dir"]
        gini = np.zeros((len(pair_names), len(feat_names)))
        try:
            for i, pn in enumerate(pair_names):
                clf = joblib.load(model_dir / f"rf_{pn}.joblib")
                gini[i, :] = clf.feature_importances_
        except Exception as e:
            print(f"    skip {name}: {e}")
            continue

        # heatmap
        fig, ax = plt.subplots(figsize=(0.45 * len(feat_names) + 2, 0.35 * len(pair_names) + 2))
        im = ax.imshow(gini, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
        for tick, fn in zip(ax.get_xticklabels(), feat_names):
            tick.set_color(BATCH_COLORS.get(BATCH_OF.get(fn, "core"), "black"))
        ax.set_yticks(range(len(pair_names)))
        ax.set_yticklabels(pair_names, fontsize=8)
        ax.set_title(f"{name} — Gini importance, USED 9 pairs only", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIG / f"importance_used_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # per-batch contribution averaged over used pairs
        per_batch = {b: 0.0 for b in BATCH_ORDER}
        for j, fn in enumerate(feat_names):
            per_batch[BATCH_OF.get(fn, "core")] += gini[:, j].sum()
        total = sum(per_batch.values())
        if total > 0:
            per_batch = {b: v / total for b, v in per_batch.items()}
        batch_rows.append({"variant": name, **per_batch})

        # top-5 per used pair
        for i, pn in enumerate(pair_names):
            order = np.argsort(-gini[i, :])[:5]
            for rank, j in enumerate(order, 1):
                top5_rows.append({
                    "variant": name, "pair": pn, "rank": rank,
                    "feature": feat_names[j],
                    "batch": BATCH_OF.get(feat_names[j], "?"),
                    "gini_importance": float(gini[i, j]),
                })

    # stacked bar of per-batch contribution
    if batch_rows:
        labels = [r["variant"].replace("models_", "") for r in batch_rows]
        x = np.arange(len(batch_rows))
        bottom = np.zeros(len(batch_rows))
        fig, ax = plt.subplots(figsize=(max(8, 0.40 * len(batch_rows)), 5.5))
        for b in BATCH_ORDER:
            vals = np.array([r[b] for r in batch_rows])
            ax.bar(x, vals, bottom=bottom, color=BATCH_COLORS[b], label=b,
                   edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("share of total Gini importance")
        ax.set_title("Per-batch importance, USED 9 pairs only, 20f variants")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "batch_contribution_used.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        with open(TBL / "batch_contribution_used.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant"] + BATCH_ORDER)
            w.writeheader()
            for r in batch_rows:
                w.writerow(r)

    if top5_rows:
        with open(TBL / "top_features_per_used_pair.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "pair", "rank", "feature",
                                              "batch", "gini_importance"])
            w.writeheader()
            for r in top5_rows:
                w.writerow(r)


# ============== Depth-sweep curve (USED filtering) ==============

def emit_depth_sweep(all_enr: dict[str, dict]) -> None:
    print("  emitting depth-sweep curves (USED 9)")
    rows = []
    depth_order = ["2", "4", "8", "16", "none"]
    targets = ("20f_t07", "20f_regr", "20f_ei_log")
    for target in targets:
        ys_train = []; ys_test = []; leaves = []
        baseline_test = baseline_train = baseline_leaves = float("nan")
        canonical = f"models_v6_full_{target}_tree"
        if canonical in all_enr:
            used = used_pairs_only(all_enr[canonical])
            baseline_train = median_metric(used, primary_lift_key(all_enr[canonical], "train"))
            baseline_test = median_metric(used, primary_lift_key(all_enr[canonical], "test"))
            lvs = [p.get("tree_structure", {}).get("n_leaves") for p in used]
            baseline_leaves = float(np.median([l for l in lvs if l is not None]))
        for d in depth_order:
            name = f"depth{d}_{target}_tree"
            if name not in all_enr:
                ys_train.append(float("nan")); ys_test.append(float("nan"))
                leaves.append(float("nan"))
                continue
            used = used_pairs_only(all_enr[name])
            tr = median_metric(used, primary_lift_key(all_enr[name], "train"))
            te = median_metric(used, primary_lift_key(all_enr[name], "test"))
            ys_train.append(tr); ys_test.append(te)
            lvs = [p.get("tree_structure", {}).get("n_leaves") for p in used]
            leaves.append(float(np.median([l for l in lvs if l is not None]) if lvs else float("nan")))
            rows.append({
                "target": target, "depth": d,
                "median_train_lift_used": tr,
                "median_test_lift_used": te,
                "median_n_leaves_used": leaves[-1],
            })

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(depth_order))
        ax.plot(x, ys_test, "o-", color="#d62728", linewidth=2.2, label="test (depth-controlled)")
        ax.plot(x, ys_train, "o--", color="#d62728", linewidth=1.2, alpha=0.55,
                label="train (depth-controlled)")
        ax.axhline(baseline_test, color="grey", linestyle=":", linewidth=1.6,
                   label=f"existing v6 tree test={baseline_test:.4f}")
        ax.set_xticks(x); ax.set_xticklabels(depth_order)
        ax.set_xlabel("max_depth")
        ylab = "median F1-lift (USED 9)" if target == "20f_t07" else "median MAE-lift (USED 9)"
        ax.set_ylabel(ylab)
        ax.set_title(f"{target} depth sweep — USED 9 pairs only")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(x, leaves, "s-", color="#1f77b4", alpha=0.6, label="median n_leaves")
        ax2.set_ylabel("median n_leaves (log)", color="#1f77b4")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"depth_sweep_used_{target}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(TBL / "depth_sweep_used.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "depth",
                                          "median_train_lift_used", "median_test_lift_used",
                                          "median_n_leaves_used"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============== Top-K agg + union curves ==============

def emit_topK_curves(all_enr: dict[str, dict]) -> None:
    print("  emitting top-K agg + union curves (USED 9)")
    rows = []
    K_order = [3, 5, 8, 12]
    for target in ("t02", "t03", "t05", "t07", "regr"):
        canonical = f"models_v6_full_20f_{target}_tree"
        canon_test = canon_train = float("nan")
        if canonical in all_enr:
            used = used_pairs_only(all_enr[canonical])
            canon_test = median_metric(used, primary_lift_key(all_enr[canonical], "test"))
            canon_train = median_metric(used, primary_lift_key(all_enr[canonical], "train"))

        # Forest reference
        forest = f"models_v3_full_20f_{target}"
        forest_test = forest_train = float("nan")
        if forest in all_enr:
            used = used_pairs_only(all_enr[forest])
            forest_test = median_metric(used, primary_lift_key(all_enr[forest], "test"))
            forest_train = median_metric(used, primary_lift_key(all_enr[forest], "train"))

        agg_x, agg_train, agg_test, agg_n = [], [], [], []
        un_x, un_train, un_test, un_n = [], [], [], []
        for K in K_order:
            agg_name = f"topK{K}_agg_{target}_classical_tree"
            un_name  = f"topK{K}_union_{target}_classical_tree"
            if agg_name in all_enr:
                used = used_pairs_only(all_enr[agg_name])
                agg_x.append(K)
                agg_train.append(median_metric(used, primary_lift_key(all_enr[agg_name], "train")))
                agg_test.append(median_metric(used, primary_lift_key(all_enr[agg_name], "test")))
                agg_n.append(all_enr[agg_name]["n_features"])
                rows.append({"target": target, "K": K, "mode": "aggregated",
                             "n_features_actual": all_enr[agg_name]["n_features"],
                             "median_train_lift_used": agg_train[-1],
                             "median_test_lift_used": agg_test[-1]})
            if un_name in all_enr:
                used = used_pairs_only(all_enr[un_name])
                un_x.append(K)
                un_train.append(median_metric(used, primary_lift_key(all_enr[un_name], "train")))
                un_test.append(median_metric(used, primary_lift_key(all_enr[un_name], "test")))
                un_n.append(all_enr[un_name]["n_features"])
                rows.append({"target": target, "K": K, "mode": "union",
                             "n_features_actual": all_enr[un_name]["n_features"],
                             "median_train_lift_used": un_train[-1],
                             "median_test_lift_used": un_test[-1]})

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(agg_x, agg_test, "o-", color="#2ca02c", linewidth=2.2, label="aggregated test")
        ax.plot(agg_x, agg_train, "o--", color="#2ca02c", linewidth=1.2, alpha=0.55,
                label="aggregated train")
        ax.plot(un_x, un_test, "s-", color="#9467bd", linewidth=2.2, label="union test (n_feat varies)")
        ax.plot(un_x, un_train, "s--", color="#9467bd", linewidth=1.2, alpha=0.55,
                label="union train")
        # Annotate the union points with the actual n_features used
        for x_val, y_val, n in zip(un_x, un_test, un_n):
            if isinstance(y_val, (int, float)) and not np.isnan(y_val):
                ax.annotate(f"n={n}", (x_val, y_val), textcoords="offset points",
                            xytext=(8, 5), fontsize=8, color="#9467bd")
        ax.axhline(canon_test, color="grey", linestyle=":", linewidth=1.4,
                   label=f"full 20f tree test={canon_test:.4f}")
        ax.axhline(forest_test, color="black", linestyle="-.", linewidth=1.4,
                   label=f"100-tree forest test={forest_test:.4f}")
        ax.set_xticks(K_order + [20])
        ax.set_xlabel("K (top-K cutoff)")
        ylab = "median F1-lift (USED 9)" if target == "t07" else "median MAE-lift (USED 9)"
        ax.set_ylabel(ylab)
        ax.set_title(f"{target} top-K classical — agg vs union vs full tree vs forest")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"topK_classical_{target}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(TBL / "topK_classical_used.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "K", "mode", "n_features_actual",
                                          "median_train_lift_used", "median_test_lift_used"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============== Feature-count progression filtered ==============

def emit_progression(all_enr: dict[str, dict]) -> None:
    print("  emitting 5f/13f/20f progression (USED 9)")
    rows = []
    for target in ("t02", "t03", "t05", "t07", "regr"):
        for variant_type in ("forest", "tree"):
            prefix = "models_v3_full_" if variant_type == "forest" else "models_v6_full_"
            suffix = "" if variant_type == "forest" else "_tree"
            for nf in ("5f", "13f", "20f"):
                name = f"{prefix}{nf}_{target}{suffix}"
                if name not in all_enr:
                    continue
                used = used_pairs_only(all_enr[name])
                tr = median_metric(used, primary_lift_key(all_enr[name], "train"))
                te = median_metric(used, primary_lift_key(all_enr[name], "test"))
                rows.append({"target": target, "variant_type": variant_type,
                             "n_feat": nf, "median_train_lift_used": tr,
                             "median_test_lift_used": te})
    with open(TBL / "feature_count_progression_used.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "variant_type", "n_feat",
                                          "median_train_lift_used", "median_test_lift_used"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for target in ("t02", "t03", "t05", "t07", "regr"):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        any_drawn = False
        for variant_type, color, marker in (("forest", "#1f77b4", "o"),
                                            ("tree", "#d62728", "s")):
            test_xs, test_ys, train_ys = [], [], []
            for r in rows:
                if r["target"] != target or r["variant_type"] != variant_type:
                    continue
                nf_int = int(r["n_feat"].rstrip("f"))
                test_xs.append(nf_int)
                test_ys.append(r["median_test_lift_used"])
                train_ys.append(r["median_train_lift_used"])
            if test_xs:
                any_drawn = True
                order = np.argsort(test_xs)
                test_xs = np.array(test_xs)[order]
                test_ys = np.array(test_ys)[order]
                train_ys = np.array(train_ys)[order]
                ax.plot(test_xs, test_ys, color=color, marker=marker, linewidth=2,
                        label=f"{variant_type} test")
                ax.plot(test_xs, train_ys, color=color, marker=marker, linewidth=1,
                        linestyle="--", alpha=0.5, label=f"{variant_type} train")
        if not any_drawn:
            plt.close(fig); continue
        ylab = "median F1-lift (USED 9)" if target in ("t02", "t03", "t05", "t07") else "median MAE-lift (USED 9)"
        ax.set_xticks([5, 13, 20])
        ax.set_xlabel("number of features")
        ax.set_ylabel(ylab)
        ax.set_title(f"{target}: feature-count progression, USED 9")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"progression_used_{target}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    print("Loading enriched reports from both folders...")
    all_enr = load_all_enriched()
    print(f"  loaded {len(all_enr)} enriched reports total")

    emit_classical_metrics(all_enr)
    emit_gap_heatmaps(all_enr)
    emit_importance(all_enr)
    emit_depth_sweep(all_enr)
    emit_topK_curves(all_enr)
    emit_progression(all_enr)

    print(f"\nFigures in {FIG}")
    print(f"Tables  in {TBL}")


if __name__ == "__main__":
    main()
