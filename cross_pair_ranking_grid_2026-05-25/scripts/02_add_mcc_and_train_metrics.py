#!/usr/bin/env python3
"""Append per-pair MCC + train-side ranking metrics to ranking_grid_aggregate.csv.

For the 32 grid configs × {deployed_75_25, 5fold}, deterministically refit
the same 9 per-pair binary classifiers (same seed, hyperparams, data, K-
feature subset → byte-identical to the original grid run). For each fold:

  - test MCC per pair (at threshold 0.5), averaged across 9 pairs
  - train MCC per pair, averaged across 9 pairs
  - train-side top-1, top-3, abs_reg, rel_reg (state-weighted over valid
    train states with max(ei_pct) > 0)

Aggregation across folds (5-fold): simple mean across the 5 folds. For
75/25 there's only one fold so the mean is the fold value. Train metrics
use fold-mean rather than state-pooled because a state appears in train
of 4/5 folds, so pooling would over-weight states by fold-count.

Append 6 new columns to ranking_grid_aggregate.csv:
  test_mcc_mean, train_mcc_mean, train_top1, train_top3,
  train_abs_reg, train_rel_reg

Existing columns and rows are preserved verbatim. Baseline rows
(random, best_avg_pair) and LOIO rows get NaN in the new columns.

Then generate figure 4 (`forest_grid_diagnostic.{png,pdf}`).
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
GRID = REPO / "cross_pair_ranking_grid_2026-05-25"
TBL = GRID / "tables"
FIG = GRID / "figures"
PRIOR_SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
xpr = _load("xpr", PRIOR_SCRIPTS / "01_cross_pair_ranking.py")

DEPTHS = [4, 8, 16, None]
KS     = [4, 8, 16, 20]

def make_grid_configs():
    cfgs = []
    for kind in ("forest", "tree"):
        for d in DEPTHS:
            for K in KS:
                tag = f"{kind}_d{'None' if d is None else d}_K{K}"
                cfgs.append({"tag": tag, "type": kind, "max_depth": d, "K": K})
    return cfgs
GRID_CFGS = make_grid_configs()


def make_clf(cfg):
    if cfg["type"] == "forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=cfg["max_depth"],
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return DecisionTreeClassifier(
        max_depth=cfg["max_depth"], class_weight="balanced", random_state=42,
    )


def compute_ranking_metrics(P, ei, valid_mask):
    if not valid_mask.any(): return None
    P_v = P[valid_mask]; E_v = ei[valid_mask]
    true_best = E_v.argmax(axis=1)
    pred_rank = np.argsort(-P_v, axis=1)
    pred_top1 = pred_rank[:, 0]
    in_top3 = np.array([true_best[i] in pred_rank[i, :3] for i in range(len(true_best))])
    arange = np.arange(len(E_v))
    abs_reg = E_v[arange, true_best] - E_v[arange, pred_top1]
    rel_reg = abs_reg / E_v[arange, true_best] * 100.0
    return {
        "top1":    float((pred_top1 == true_best).mean()),
        "top3":    float(in_top3.mean()),
        "abs_reg": float(abs_reg.mean()),
        "rel_reg": float(rel_reg.mean()),
    }


def fit_eval_one_fold(df, tr_mask, te_mask, cfg):
    feat_idx = xpr.aggregated_top_k_indices(cfg["K"])
    X = df[xpr.FEATURE_NAMES_20].values
    EI = df[xpr.USED_EI].values
    X_tr = X[tr_mask][:, feat_idx]; X_te = X[te_mask][:, feat_idx]
    ei_tr = EI[tr_mask]; ei_te = EI[te_mask]
    valid_tr = ei_tr.max(axis=1) > 0
    valid_te = ei_te.max(axis=1) > 0

    P_te = np.zeros((te_mask.sum(), 9), dtype=float)
    P_tr = np.zeros((tr_mask.sum(), 9), dtype=float)
    test_mccs, train_mccs = [], []

    for j, p in enumerate(xpr.USED_PAIRS):
        y_tr = df[p].values[tr_mask].astype(int)
        y_te = df[p].values[te_mask].astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            prev = float(y_tr.mean())
            P_te[:, j] = prev; P_tr[:, j] = prev
            test_mccs.append(np.nan); train_mccs.append(np.nan)
            continue
        clf = make_clf(cfg)
        clf.fit(X_tr, y_tr)
        P_te[:, j] = clf.predict_proba(X_te)[:, 1]
        P_tr[:, j] = clf.predict_proba(X_tr)[:, 1]
        y_pred_te = (P_te[:, j] >= 0.5).astype(int)
        y_pred_tr = (P_tr[:, j] >= 0.5).astype(int)
        test_mccs.append(
            float(matthews_corrcoef(y_te, y_pred_te)) if len(set(y_te)) > 1 else np.nan
        )
        train_mccs.append(float(matthews_corrcoef(y_tr, y_pred_tr)))

    return {
        "test_mcc_mean":  float(np.nanmean(test_mccs))  if any(not np.isnan(x) for x in test_mccs)  else np.nan,
        "train_mcc_mean": float(np.nanmean(train_mccs)) if any(not np.isnan(x) for x in train_mccs) else np.nan,
        "test_metrics":  compute_ranking_metrics(P_te, ei_te, valid_te),
        "train_metrics": compute_ranking_metrics(P_tr, ei_tr, valid_tr),
    }


def eval_one_protocol(df, fold_iter_fn, cfg):
    fold_results = []
    for fold in fold_iter_fn(df):
        r = fit_eval_one_fold(df, fold["train_mask"], fold["test_mask"], cfg)
        if r["test_metrics"] is None or r["train_metrics"] is None:
            continue
        fold_results.append(r)
    if not fold_results: return None
    return {
        "test_mcc_mean":  float(np.nanmean([r["test_mcc_mean"]  for r in fold_results])),
        "train_mcc_mean": float(np.nanmean([r["train_mcc_mean"] for r in fold_results])),
        "train_top1":     float(np.mean([r["train_metrics"]["top1"]    for r in fold_results])),
        "train_top3":     float(np.mean([r["train_metrics"]["top3"]    for r in fold_results])),
        "train_abs_reg":  float(np.mean([r["train_metrics"]["abs_reg"] for r in fold_results])),
        "train_rel_reg":  float(np.mean([r["train_metrics"]["rel_reg"] for r in fold_results])),
    }


# ============================================================
# Figure 4 — forest_grid_diagnostic
# ============================================================
def generate_figure4(agg: pd.DataFrame):
    plt.rcParams["font.family"]   = "sans-serif"
    plt.rcParams["axes.facecolor"]   = "white"
    plt.rcParams["figure.facecolor"] = "white"

    # Compute the gap column for plotting
    agg = agg.copy()
    agg["mcc_gap"] = agg["train_mcc_mean"] - agg["test_mcc_mean"]

    def heatmap(ax, protocol, metric, title, cmap, fmt_fn):
        mat = np.full((len(KS), len(DEPTHS)), np.nan)
        for _, r in agg.iterrows():
            if r["protocol"] != protocol: continue
            if not r["model"].startswith("forest_d"): continue
            parts = r["model"].split("_")    # ['forest','dNone','K20']
            d_str = parts[1][1:]
            K = int(parts[2][1:])
            d = None if d_str == "None" else int(d_str)
            if d not in DEPTHS or K not in KS: continue
            i = KS.index(K); j = DEPTHS.index(d)
            v = r[metric]
            if pd.notna(v):
                mat[i][j] = v
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels(["None" if d is None else str(d) for d in DEPTHS])
        ax.set_yticks(range(len(KS)))
        ax.set_yticklabels([str(k) for k in KS])
        ax.set_xlabel("depth")
        ax.set_ylabel("K (top-K features by Gini)")
        for i in range(len(KS)):
            for j in range(len(DEPTHS)):
                v = mat[i][j]
                if not np.isnan(v):
                    ax.text(j, i, fmt_fn(v), ha="center", va="center",
                            fontsize=8, color="black")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)

    proto_label = {"deployed_75_25": "75/25 split", "5fold": "5-fold"}
    fmt_mcc = lambda v: f"{v:.3f}"     # 3 sig figs (MCC values in [-1, 1])
    fmt_abs = lambda v: f"{v:.4f}"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, proto in enumerate(["deployed_75_25", "5fold"]):
        heatmap(axes[row, 0], proto, "test_mcc_mean",
                f"Forest test MCC, {proto_label[proto]}",
                cmap="viridis", fmt_fn=fmt_mcc)
        heatmap(axes[row, 1], proto, "abs_reg",
                f"Forest abs regret, {proto_label[proto]}",
                cmap="Reds", fmt_fn=fmt_abs)
        heatmap(axes[row, 2], proto, "mcc_gap",
                f"Forest train−test MCC gap, {proto_label[proto]}",
                cmap="Reds", fmt_fn=fmt_mcc)
    fig.tight_layout()
    out_png = FIG / "forest_grid_diagnostic.png"
    out_pdf = FIG / "forest_grid_diagnostic.pdf"
    fig.savefig(out_png, dpi=300); fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"      {out_pdf}")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + reproducing folds...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")
    print(f"  Grid: {len(GRID_CFGS)} configs × 2 protocols (deployed_75_25 + 5fold) "
          f"× 9 pairs × (1 + 5) folds = {len(GRID_CFGS) * 9 * 6} model fits")

    new_metrics = {}     # (protocol, model_tag) -> dict
    t0 = time.time()
    for ci, cfg in enumerate(GRID_CFGS):
        for proto, fold_iter in [("deployed_75_25", xpr.deployed_split),
                                  ("5fold", lambda d: xpr.fivefold_splits(d))]:
            m = eval_one_protocol(df, fold_iter, cfg)
            if m is None: continue
            new_metrics[(proto, cfg["tag"])] = m
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] {ci+1:2d}/{len(GRID_CFGS)} "
                  f"{cfg['tag']:25s} {proto:15s} "
                  f"test_mcc={m['test_mcc_mean']:+.3f}  "
                  f"train_mcc={m['train_mcc_mean']:+.3f}  "
                  f"gap={m['train_mcc_mean']-m['test_mcc_mean']:+.3f}  "
                  f"train_top1={m['train_top1']:.3f}",
                  flush=True)

    print(f"\nTotal fit time: {time.time()-t0:.1f}s")

    # Append columns to existing aggregate CSV
    print("\nAppending columns to ranking_grid_aggregate.csv...")
    agg_csv = TBL / "ranking_grid_aggregate.csv"
    agg = pd.read_csv(agg_csv)
    print(f"  Existing: {len(agg)} rows × {len(agg.columns)} cols")
    new_cols = ["test_mcc_mean", "train_mcc_mean",
                "train_top1", "train_top3", "train_abs_reg", "train_rel_reg"]
    for c in new_cols:
        if c not in agg.columns:
            agg[c] = np.nan
    filled = 0
    for idx, row in agg.iterrows():
        key = (row["protocol"], row["model"])
        if key in new_metrics:
            for c in new_cols:
                agg.at[idx, c] = new_metrics[key][c]
            filled += 1
    agg.to_csv(agg_csv, index=False)
    print(f"  Updated {filled} rows ({len(agg) - filled} rows keep NaN for new cols)")
    print(f"  Wrote {agg_csv} (now {len(agg)} rows × {len(agg.columns)} cols)")

    # Generate figure 4
    print("\nGenerating figure 4...")
    generate_figure4(agg)


if __name__ == "__main__":
    main()
