#!/usr/bin/env python3
"""Aggregate CV results: metric stability across folds + model-similarity across folds.

Reads:  tables/cv_raw_results.json
Writes:
  tables/cv_metrics_summary.csv     — per (target, model, metric): fold values + mean±std
  tables/cv_baselines.csv           — per (target, fold): always-0/1 acc & F1, prevalence
  tables/cv_model_similarity.csv    — per (target, model): cross-fold similarity metrics
  Prints markdown tables to stdout.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "model_diagnostics_cv_2026-05-19"
TBL = OUT / "tables"
RAW = TBL / "cv_raw_results.json"

USED_PAIRS = [f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)]

CLF_METRICS = ["f1", "accuracy", "precision", "recall", "specificity",
               "balanced_accuracy", "roc_auc", "pr_auc", "brier", "mcc",
               "f1_lift", "accuracy_lift", "precision_lift",
               "balanced_accuracy_lift", "roc_auc_lift"]

BASELINE_KEYS = ["prevalence", "always0_accuracy", "always1_accuracy",
                 "best_baseline_accuracy", "always0_f1", "always1_f1",
                 "best_baseline_f1"]


def load_raw():
    return json.loads(RAW.read_text())


def non_degenerate(pair_metrics: dict) -> bool:
    """A pair's test set is non-degenerate if 5% <= pos_rate <= 95%."""
    if "test_prevalence" not in pair_metrics:
        return False
    p = pair_metrics["test_prevalence"]
    return 0.05 <= p <= 0.95


def fold_mean(metrics_per_pair: dict, split: str, metric: str) -> float | None:
    """Mean of the given metric across USED non-degenerate pairs in one fold."""
    vals = []
    for pair in USED_PAIRS:
        if pair not in metrics_per_pair: continue
        pm = metrics_per_pair[pair]
        if not non_degenerate(pm): continue
        v = pm.get(f"{split}_{metric}")
        if isinstance(v, (int, float)):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def baseline_fold_mean(metrics_per_pair: dict, key: str) -> float | None:
    """Baseline value averaged over used non-degenerate pairs (test split)."""
    vals = []
    for pair in USED_PAIRS:
        if pair not in metrics_per_pair: continue
        pm = metrics_per_pair[pair]
        if not non_degenerate(pm): continue
        v = pm.get(f"test_{key}")
        if isinstance(v, (int, float)):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def stats(vals: list) -> dict:
    vs = [v for v in vals if isinstance(v, (int, float))]
    if not vs: return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    return {"n": len(vs), "mean": float(st.mean(vs)),
            "std": float(st.stdev(vs)) if len(vs) > 1 else 0.0,
            "min": float(min(vs)), "max": float(max(vs))}


# -------- Similarity helpers --------

def feature_importance_pearson(fold_results_for_model) -> float:
    """Average pairwise Pearson r of feature_importances across folds, averaged over USED pairs.

    For each USED pair, compute Pearson r between every pair of folds' importance vectors,
    then average across fold-pairs, then across USED pairs.
    """
    per_pair_pearsons = []
    n_folds = len(fold_results_for_model)
    if n_folds < 2: return float("nan")
    for pair in USED_PAIRS:
        vecs = []
        for fold_res in fold_results_for_model:
            ps = fold_res["pair_structural"].get(pair)
            if not ps: continue
            vecs.append(np.array(ps["feature_importances"], dtype=float))
        if len(vecs) < 2: continue
        rs = []
        for a, b in combinations(vecs, 2):
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            r, _ = pearsonr(a, b)
            rs.append(r)
        if rs:
            per_pair_pearsons.append(float(np.mean(rs)))
    return float(np.mean(per_pair_pearsons)) if per_pair_pearsons else float("nan")


def topK_feature_jaccard(fold_results_for_model, K: int = 5) -> float:
    """Average pairwise Jaccard@K of top-K features by importance, across folds, averaged over pairs."""
    per_pair = []
    for pair in USED_PAIRS:
        sets = []
        for fold_res in fold_results_for_model:
            ps = fold_res["pair_structural"].get(pair)
            if not ps: continue
            imp = np.array(ps["feature_importances"])
            feat_names = fold_res["feature_names"]
            order = np.argsort(-imp)
            top = set(feat_names[i] for i in order[:K])
            sets.append(top)
        if len(sets) < 2: continue
        jacs = []
        for a, b in combinations(sets, 2):
            u = a | b
            i = a & b
            jacs.append(len(i) / len(u) if u else 0)
        if jacs:
            per_pair.append(float(np.mean(jacs)))
    return float(np.mean(per_pair)) if per_pair else float("nan")


def root_split_agreement(fold_results_for_model) -> dict:
    """For each pair, what fraction of fold-pairs agree on the root-split feature?
    For forests we use the MAJORITY-vote root feature across the 100 estimators."""
    is_tree = False
    for fr in fold_results_for_model:
        for pair, ps in fr["pair_structural"].items():
            if ps.get("kind") == "tree":
                is_tree = True
                break
        if is_tree: break

    per_pair = []
    for pair in USED_PAIRS:
        roots = []
        for fold_res in fold_results_for_model:
            ps = fold_res["pair_structural"].get(pair)
            if not ps: continue
            if ps["kind"] == "tree":
                if ps["top_3_bfs_splits"]:
                    roots.append(ps["top_3_bfs_splits"][0]["feature_name"])
            else:  # forest
                if ps["top_3_root_features"]:
                    roots.append(ps["top_3_root_features"][0]["feature_name"])
        if len(roots) < 2: continue
        # Fraction of fold-pairs that agree
        agree = sum(1 for a, b in combinations(roots, 2) if a == b)
        total = sum(1 for _ in combinations(roots, 2))
        per_pair.append(agree / total if total else 0)
    return {
        "kind": "tree" if is_tree else "forest",
        "frac_fold_pairs_with_same_root_feature_mean":
            float(np.mean(per_pair)) if per_pair else float("nan"),
        "frac_fold_pairs_with_same_root_feature_min":
            float(np.min(per_pair)) if per_pair else float("nan"),
    }


def tree_structure_variance(fold_results_for_model) -> dict | None:
    """Per pair: across folds, what's the std of (max_depth, n_leaves) for trees;
    median(median_depth) and median(median_n_leaves) for forests."""
    depths = defaultdict(list); leaves = defaultdict(list)
    is_tree = False
    for fold_res in fold_results_for_model:
        for pair, ps in fold_res["pair_structural"].items():
            if ps.get("kind") == "tree":
                is_tree = True
                depths[pair].append(ps["max_depth"])
                leaves[pair].append(ps["n_leaves"])
            else:
                depths[pair].append(ps["median_depth"])
                leaves[pair].append(ps["median_n_leaves"])
    if not depths: return None
    return {
        "kind": "tree" if is_tree else "forest",
        "median_pair_depth_std": float(np.median([np.std(depths[p]) for p in depths])),
        "median_pair_n_leaves_std": float(np.median([np.std(leaves[p]) for p in leaves])),
        "median_pair_depth_mean": float(np.median([np.mean(depths[p]) for p in depths])),
        "median_pair_n_leaves_mean": float(np.median([np.mean(leaves[p]) for p in leaves])),
    }


# -------- Main --------

def main():
    raw = load_raw()
    results = raw["results"]
    n_folds = raw["config"]["K"]

    # Group by (target, model_label)
    groups = defaultdict(list)
    for r in results:
        groups[(r["target"], r["model_label"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["fold"])

    # --- Per-target baselines (model-agnostic, but they vary slightly per fold
    #     due to which test rows ended up in each fold). We aggregate from the
    #     forest_baseline rows since baselines are pair-test-set-dependent and
    #     the same across all models within a fold.
    print("\n==========================================")
    print("BASELINES — per target × fold")
    print("==========================================\n")
    baselines_rows = []
    for target in sorted(set(r["target"] for r in results)):
        # Pick the first model arbitrarily; baselines are model-independent
        any_label = next(ml for (t, ml) in groups if t == target)
        runs = groups[(target, any_label)]
        print(f"### Target {target}")
        print(f"| fold | n test logs | prevalence | best baseline accuracy | best baseline F1 |")
        print(f"|---|---|---|---|---|")
        prev_vals = []
        baccs = []; bf1s = []
        for r in runs:
            prev = baseline_fold_mean(r["pair_metrics"], "prevalence")
            bacc = baseline_fold_mean(r["pair_metrics"], "best_baseline_accuracy")
            bf1 = baseline_fold_mean(r["pair_metrics"], "best_baseline_f1")
            a0 = baseline_fold_mean(r["pair_metrics"], "always0_accuracy")
            a1 = baseline_fold_mean(r["pair_metrics"], "always1_accuracy")
            f1a0 = baseline_fold_mean(r["pair_metrics"], "always0_f1")
            f1a1 = baseline_fold_mean(r["pair_metrics"], "always1_f1")
            prev_vals.append(prev); baccs.append(bacc); bf1s.append(bf1)
            # count test logs from folds_meta
            test_logs = next(fm["n_test_logs"] for fm in raw["folds_meta"]
                             if fm["fold"] == r["fold"])
            print(f"| {r['fold']} | {test_logs} | {prev:.3f} | {bacc:.3f} | {bf1:.3f} |")
            baselines_rows.append({"target": target, "fold": r["fold"],
                                   "n_test_logs": test_logs,
                                   "prevalence_mean_over_used_pairs": prev,
                                   "always0_accuracy_mean": a0, "always1_accuracy_mean": a1,
                                   "best_baseline_accuracy_mean": bacc,
                                   "always0_f1_mean": f1a0, "always1_f1_mean": f1a1,
                                   "best_baseline_f1_mean": bf1})
        print(f"| **mean ± std** | — | "
              f"{st.mean(prev_vals):.3f} ± {st.stdev(prev_vals):.3f} | "
              f"{st.mean(baccs):.3f} ± {st.stdev(baccs):.3f} | "
              f"{st.mean(bf1s):.3f} ± {st.stdev(bf1s):.3f} |\n")
    with open(TBL / "cv_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baselines_rows[0].keys()))
        w.writeheader()
        for r in baselines_rows:
            w.writerow(r)

    # --- Metrics aggregation
    metric_rows = []
    print("\n==========================================")
    print("METRICS — fold values + across-fold mean ± std")
    print("==========================================\n")
    for target in sorted(set(t for (t, _) in groups)):
        print(f"\n### Target {target}\n")
        for metric in CLF_METRICS:
            print(f"\n#### Metric: {metric}  (test split, mean over 9 used non-degenerate pairs)\n")
            print(f"| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | **mean ± std** |")
            print(f"|---|---|---|---|---|---|---|")
            for (t, model_label), runs in sorted(groups.items()):
                if t != target: continue
                fold_means = []
                for r in runs:
                    v = fold_mean(r["pair_metrics"], "test", metric)
                    fold_means.append(v)
                ss = stats(fold_means)
                fold_strs = " | ".join(f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"
                                       for v in fold_means)
                summary = (f"**{ss['mean']:+.3f} ± {ss['std']:.3f}**"
                           if ss["n"] else "n/a")
                print(f"| {model_label} | {fold_strs} | {summary} |")
                metric_rows.append({
                    "target": target, "model_label": model_label, "metric": metric,
                    "split": "test",
                    **{f"fold_{i}": v for i, v in enumerate(fold_means)},
                    "across_fold_mean": ss["mean"], "across_fold_std": ss["std"],
                    "across_fold_min": ss["min"], "across_fold_max": ss["max"],
                    "n_folds": ss["n"],
                })
    with open(TBL / "cv_metrics_summary.csv", "w", newline="") as f:
        cols = ["target", "model_label", "metric", "split",
                "fold_0", "fold_1", "fold_2", "fold_3", "fold_4",
                "across_fold_mean", "across_fold_std", "across_fold_min",
                "across_fold_max", "n_folds"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in metric_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # --- Similarity
    print("\n\n==========================================")
    print("MODEL SIMILARITY ACROSS FOLDS")
    print("==========================================\n")
    sim_rows = []
    for target in sorted(set(t for (t, _) in groups)):
        print(f"\n### Target {target}\n")
        print(f"| model | feat-imp Pearson r (pair-avg) | top-5 feat Jaccard | same-root-feat across fold-pairs | depth std (per pair, median) | n_leaves std (per pair, median) |")
        print(f"|---|---|---|---|---|---|")
        for (t, model_label), runs in sorted(groups.items()):
            if t != target: continue
            pearson_r = feature_importance_pearson(runs)
            jacc = topK_feature_jaccard(runs, K=5)
            root_agree = root_split_agreement(runs)
            struct_var = tree_structure_variance(runs) or {}
            sim_rows.append({
                "target": target, "model_label": model_label,
                "feat_importance_pearson_pairavg": pearson_r,
                "top5_feat_jaccard_pairavg": jacc,
                "root_feat_agreement_pairavg": root_agree["frac_fold_pairs_with_same_root_feature_mean"],
                "root_feat_agreement_min": root_agree["frac_fold_pairs_with_same_root_feature_min"],
                "median_pair_depth_std": struct_var.get("median_pair_depth_std", ""),
                "median_pair_depth_mean": struct_var.get("median_pair_depth_mean", ""),
                "median_pair_n_leaves_std": struct_var.get("median_pair_n_leaves_std", ""),
                "median_pair_n_leaves_mean": struct_var.get("median_pair_n_leaves_mean", ""),
                "kind": root_agree["kind"],
            })
            print(f"| {model_label} | {pearson_r:.3f} | {jacc:.3f} | "
                  f"{root_agree['frac_fold_pairs_with_same_root_feature_mean']:.3f} | "
                  f"{struct_var.get('median_pair_depth_std', float('nan')):.2f} | "
                  f"{struct_var.get('median_pair_n_leaves_std', float('nan')):.2f} |")

    with open(TBL / "cv_model_similarity.csv", "w", newline="") as f:
        cols = ["target", "model_label", "kind",
                "feat_importance_pearson_pairavg", "top5_feat_jaccard_pairavg",
                "root_feat_agreement_pairavg", "root_feat_agreement_min",
                "median_pair_depth_mean", "median_pair_depth_std",
                "median_pair_n_leaves_mean", "median_pair_n_leaves_std"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sim_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"\nWrote: cv_metrics_summary.csv, cv_baselines.csv, cv_model_similarity.csv")


if __name__ == "__main__":
    main()
