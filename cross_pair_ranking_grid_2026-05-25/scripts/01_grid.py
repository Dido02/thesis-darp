#!/usr/bin/env python3
"""Cross-pair ranking — full grid (32 configs × 3 protocols).

Configs: 16 forests + 16 trees, depth ∈ {4, 8, 16, None} × K ∈ {4, 8, 16, 20}.
n_estimators=100 (forests). class_weight=balanced, random_state=42 (both).
Feature subsets for K<20: top-K by aggregate Gini importance from the
deployed forest, same convention as cross_pair_ranking_2026-05-24/.

Protocols: 75/25 (deployed split), 5-fold (log-stratified by (size, delay),
round-robin, seed=42), LOIO (14 folds, one per (size, delay)).

Metrics (all state-weighted, skipping states with max(ei_pct)=0 over 9 USED):
top-1, top-3, mean absolute regret, mean relative regret. Ground truth =
argmax over 9 USED ei_pct cols.

Baselines: random (top-1=1/9, top-3=3/9, per-state regrets) and
best_avg_pair (per-fold pair choice from TRAIN mean ei_pct, per-state
evaluation on TEST, then pooled).

Outputs:
  tables/ranking_grid_aggregate.csv
  tables/ranking_grid_per_instance_loio.csv
  tables/ranking_grid_per_instance_5fold.csv
  tables/side_by_side_5fold_vs_loio.csv
  figures/heatmap_{forest,tree}_{protocol}_top1.png       (6 figs)
  figures/heatmap_forest_loio_rel_regret.png              (1 fig)
  RANKING_GRID_REPORT.md
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

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
PRIOR_SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"
OUT = REPO / "cross_pair_ranking_grid_2026-05-25"
TBL = OUT / "tables"
FIG = OUT / "figures"


# ============================================================
# Reuse helpers from the patched prior analysis
# ============================================================
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

xpr = _load_module("xpr", PRIOR_SCRIPTS / "01_cross_pair_ranking.py")
fix = _load_module("fix", PRIOR_SCRIPTS / "02_fix_baselines.py")


# ============================================================
# Grid
# ============================================================
DEPTHS = [4, 8, 16, None]
KS     = [4, 8, 16, 20]

def make_grid_configs() -> list[dict]:
    cfgs = []
    for kind in ("forest", "tree"):
        for d in DEPTHS:
            for K in KS:
                tag = f"{kind}_d{'None' if d is None else d}_K{K}"
                cfgs.append({"tag": tag, "type": kind, "max_depth": d, "K": K})
    return cfgs

GRID = make_grid_configs()
print(f"Grid: {len(GRID)} configs")


# ============================================================
# Per-protocol evaluation — collects per-state rows then aggregates
# ============================================================
def evaluate_one_fold(df, fold, cfg, ei_test, valid, valid_inst, E_valid):
    """Fit + predict for one (fold, cfg), return per-state metric arrays."""
    P, _ = xpr.fit_and_predict(df, fold["train_mask"], fold["test_mask"], cfg)
    P_valid = P[valid]
    if len(P_valid) == 0: return None
    true_best = E_valid.argmax(axis=1)
    pred_rank = np.argsort(-P_valid, axis=1)
    pred_top1 = pred_rank[:, 0]
    in_top3 = np.array([true_best[i] in pred_rank[i, :3] for i in range(len(true_best))])
    abs_reg = E_valid[np.arange(len(E_valid)), true_best] - E_valid[np.arange(len(E_valid)), pred_top1]
    rel_reg = abs_reg / E_valid[np.arange(len(E_valid)), true_best] * 100.0
    return {
        "instance": valid_inst,
        "top1_hit": (pred_top1 == true_best).astype(float),
        "top3_hit": in_top3.astype(float),
        "abs_reg": abs_reg, "rel_reg": rel_reg,
    }


def evaluate_protocol(df, fold_iter_fn, protocol_label, grid):
    """For each (fold, model) compute per-state metrics. Returns long DF
    with per-state rows for models AND baselines (same convention as
    the patched cross_pair_ranking_2026-05-24/)."""
    rows = []
    folds = list(fold_iter_fn(df))
    print(f"\n[{protocol_label}] {len(folds)} folds × {len(grid)} configs")
    t0 = time.time()
    for fi, fold in enumerate(folds):
        tr, te = fold["train_mask"], fold["test_mask"]
        ei_train = df.loc[tr, xpr.USED_EI].values
        ei_test  = df.loc[te, xpr.USED_EI].values
        test_inst = fold["test_instances"]
        max_ei = ei_test.max(axis=1)
        valid = max_ei > 0
        if not valid.any(): continue
        valid_inst = test_inst[valid]; E_valid = ei_test[valid]
        n = int(valid.sum())

        # ----- Baselines (per-state) -----
        rnd = fix.baseline_random_per_state(E_valid)
        bap = fix.baseline_best_avg_pair_per_state(ei_train, E_valid)
        for k in range(n):
            rows.append({"protocol": protocol_label, "fold": fold["name"],
                         "model": "random", "instance": valid_inst[k],
                         "top1_hit": rnd["top1_hit"][k], "top3_hit": rnd["top3_hit"][k],
                         "abs_reg": rnd["abs_reg"][k], "rel_reg": rnd["rel_reg"][k]})
            rows.append({"protocol": protocol_label, "fold": fold["name"],
                         "model": "best_avg_pair", "instance": valid_inst[k],
                         "top1_hit": bap["top1_hit"][k], "top3_hit": bap["top3_hit"][k],
                         "abs_reg": bap["abs_reg"][k], "rel_reg": bap["rel_reg"][k]})

        # ----- Grid models -----
        for ci, cfg in enumerate(grid):
            t_fit = time.time()
            res = evaluate_one_fold(df, fold, cfg, ei_test, valid, valid_inst, E_valid)
            if res is None: continue
            for k in range(n):
                rows.append({"protocol": protocol_label, "fold": fold["name"],
                             "model": cfg["tag"], "instance": res["instance"][k],
                             "top1_hit": res["top1_hit"][k], "top3_hit": res["top3_hit"][k],
                             "abs_reg": res["abs_reg"][k], "rel_reg": res["rel_reg"][k]})
            if ci == 0 or (ci + 1) % 8 == 0:
                print(f"  fold {fi+1}/{len(folds)} ({fold['name']:30s}) "
                      f"cfg {ci+1}/{len(grid)} ({cfg['tag']:25s}) "
                      f"  +{time.time()-t_fit:5.2f}s elapsed={time.time()-t0:6.1f}s",
                      flush=True)
    print(f"[{protocol_label}] done in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


# ============================================================
# Aggregations
# ============================================================
def agg_per_protocol_model(state_df):
    out = []
    for (p, m), sub in state_df.groupby(["protocol", "model"]):
        t3 = float(sub["top3_hit"].mean()) if sub["top3_hit"].notna().any() else np.nan
        out.append({"protocol": p, "model": m, "n_states": int(len(sub)),
                    "top1": float(sub["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(sub["abs_reg"].mean()),
                    "rel_reg": float(sub["rel_reg"].mean())})
    return pd.DataFrame(out)

def agg_per_instance(state_df, protocol):
    sub = state_df[state_df.protocol == protocol]
    out = []
    for (m, inst), g in sub.groupby(["model", "instance"]):
        t3 = float(g["top3_hit"].mean()) if g["top3_hit"].notna().any() else np.nan
        out.append({"protocol": protocol, "model": m, "instance": inst,
                    "n_states": len(g),
                    "top1": float(g["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(g["abs_reg"].mean()),
                    "rel_reg": float(g["rel_reg"].mean())})
    return pd.DataFrame(out)

def side_by_side(p5, pl):
    a = p5[["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    a.columns = ["model", "instance", "fold5_top1", "fold5_top3", "fold5_abs", "fold5_rel"]
    b = pl[["model", "instance", "top1", "top3", "abs_reg", "rel_reg"]].copy()
    b.columns = ["model", "instance", "loio_top1", "loio_top3", "loio_abs", "loio_rel"]
    cmp = a.merge(b, on=["model", "instance"], how="outer")
    cmp["delta_top1"] = cmp["loio_top1"] - cmp["fold5_top1"]
    cmp["delta_top3"] = cmp["loio_top3"] - cmp["fold5_top3"]
    cmp["delta_abs"]  = cmp["loio_abs"]  - cmp["fold5_abs"]
    return cmp


# ============================================================
# Heatmaps (matches model_diagnostics_t07_sweep_2026-05-20 convention)
# ============================================================
def heatmap(agg_df, kind, protocol, metric, title, out_png, *,
            cmap="viridis", baseline_value=None, baseline_label="",
            invert_better=False):
    """kind='forest'|'tree'. x=depth, y=K. value = metric for the
    (kind, depth, K, protocol) cell. baseline_value annotates the title."""
    mat = np.full((len(KS), len(DEPTHS)), np.nan)
    for _, r in agg_df.iterrows():
        if r["protocol"] != protocol: continue
        if not r["model"].startswith(kind + "_d"): continue
        parts = r["model"].split("_")  # ['forest','dNone','K20'] or ['tree','d4','K4']
        d_str = parts[1][1:]
        K = int(parts[2][1:])
        d = None if d_str == "None" else int(d_str)
        if d not in DEPTHS or K not in KS: continue
        i = KS.index(K); j = DEPTHS.index(d)
        v = r[metric]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            mat[i][j] = v

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(DEPTHS))); ax.set_xticklabels(["None" if d is None else str(d) for d in DEPTHS])
    ax.set_yticks(range(len(KS))); ax.set_yticklabels([str(k) for k in KS])
    ax.set_xlabel("depth"); ax.set_ylabel("K (features)")
    for i in range(len(KS)):
        for j in range(len(DEPTHS)):
            v = mat[i][j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if invert_better else "black")
    plt.colorbar(im, ax=ax)
    full_title = title
    if baseline_value is not None:
        full_title += f"\n{baseline_label} = {baseline_value:.3f}"
    ax.set_title(full_title, fontsize=10)

    # Mark deployed configuration (forest, depth=None, K=20) on forest plots
    if kind == "forest":
        ji = DEPTHS.index(None); ki = KS.index(20)
        ax.plot(ji, ki, marker="*", color="red", markersize=14,
                markeredgecolor="black", markeredgewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110); plt.close(fig)


# ============================================================
# Markdown report
# ============================================================
def write_report(agg, p_lo, p_5f, side, out_path):
    L = []
    L.append("# Cross-pair Ranking — full grid (32 configs × 3 protocols)")
    L.append("")
    L.append("**Grid:** 16 forests + 16 trees, `depth ∈ {4, 8, 16, None} × K ∈ {4, 8, 16, 20}`.")
    L.append("Forests: `n_estimators=100`. Both: `class_weight='balanced'`, `random_state=42`.")
    L.append("K-feature subsets are the top-K features by aggregate Gini importance in the deployed forest.")
    L.append("")
    L.append("**Protocols:** 75/25 (deployed split), 5-fold (log-stratified by (size,delay), seed=42), LOIO (14 folds).")
    L.append("")
    L.append("**Aggregation:** state-weighted; states with `max(ei_pct over 9 USED) = 0` excluded.")
    L.append("")

    # Order: baselines first, then forests then trees, both by depth then K
    def model_sort_key(m):
        if m == "random": return (0, 0, 0)
        if m == "best_avg_pair": return (0, 1, 0)
        kind, d, K = m.split("_")
        d = 999 if d == "dNone" else int(d[1:])
        K = int(K[1:])
        kind_idx = 1 if kind == "forest" else 2
        return (kind_idx, d, K)
    proto_order = ["deployed_75_25", "5fold", "LOIO"]

    L.append("## 1. Aggregate per (model, protocol)")
    L.append("")
    L.append("| protocol | model | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for proto in proto_order:
        models = sorted(agg[agg.protocol == proto]["model"].unique(), key=model_sort_key)
        for m in models:
            r = agg[(agg.protocol == proto) & (agg.model == m)]
            if len(r) == 0: continue
            r = r.iloc[0]
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| {proto} | `{m}` | {r['n_states']} | "
                     f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 2. Per-instance breakdown — LOIO")
    L.append("")
    L.append("| model | instance | n | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m in sorted(p_lo["model"].unique(), key=model_sort_key):
        sub = p_lo[p_lo.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| `{m}` | {r['instance']} | {r['n_states']} | "
                     f"{r['top1']:.3f} | {t3} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 3. Side-by-side — 5-fold vs LOIO (per model, per instance)")
    L.append("")
    L.append("| model | instance | 5f top-1 | LOIO top-1 | Δ top-1 | 5f abs | LOIO abs | Δ abs |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in sorted(side["model"].unique(), key=model_sort_key):
        sub = side[side.model == m].sort_values("instance")
        for _, r in sub.iterrows():
            def f(v): return "n/a" if pd.isna(v) else f"{v:.3f}"
            def fa(v): return "n/a" if pd.isna(v) else f"{v:.4f}"
            L.append(f"| `{m}` | {r['instance']} | {f(r['fold5_top1'])} | "
                     f"{f(r['loio_top1'])} | {f(r['delta_top1'])} | "
                     f"{fa(r['fold5_abs'])} | {fa(r['loio_abs'])} | {fa(r['delta_abs'])} |")
    L.append("")

    # ---------------- Summary prose ----------------
    L.append("## 4. Summary")
    L.append("")
    loio_agg = agg[agg.protocol == "LOIO"].copy()
    bap_top1 = loio_agg[loio_agg.model == "best_avg_pair"]["top1"].iloc[0]
    rnd_top1 = loio_agg[loio_agg.model == "random"]["top1"].iloc[0]

    forests = loio_agg[loio_agg["model"].str.startswith("forest_")].copy()
    best_forest = forests.sort_values("top1", ascending=False).iloc[0]
    L.append(f"**(a) Best forest by LOIO top-1:** `{best_forest['model']}` with top-1 = {best_forest['top1']:.3f}, top-3 = {best_forest['top3']:.3f}, abs regret = {best_forest['abs_reg']:.4f}, rel regret = {best_forest['rel_reg']:.2f}%.")
    L.append("")
    L.append(f"**(b) Beating `best_avg_pair` on LOIO top-1 ({bap_top1:.3f}):**")
    beats = loio_agg[(loio_agg["model"].str.startswith(("forest_", "tree_"))) & (loio_agg["top1"] > bap_top1)]
    if len(beats) == 0:
        L.append(f"  - No cell beats `best_avg_pair` on LOIO top-1. The constant-predictor baseline (pair_1_1 = worst+regret2 in every fold) dominates every grid config.")
    else:
        L.append(f"  - {len(beats)} cell(s) beat the baseline:")
        for _, r in beats.sort_values("top1", ascending=False).iterrows():
            L.append(f"    - `{r['model']}`: top-1 = {r['top1']:.3f} (+{r['top1']-bap_top1:.3f} pp)")
    L.append("")

    # Forest vs tree gap at matched (depth, K)
    L.append("**(c) Forest vs tree at matched (depth, K) on LOIO top-1:**")
    L.append("")
    L.append("| depth | K | forest top-1 | tree top-1 | forest − tree |")
    L.append("|---|---|---:|---:|---:|")
    for d in DEPTHS:
        d_str = "None" if d is None else str(d)
        for K in KS:
            ftag = f"forest_d{d_str}_K{K}"; ttag = f"tree_d{d_str}_K{K}"
            f_row = loio_agg[loio_agg.model == ftag]
            t_row = loio_agg[loio_agg.model == ttag]
            if len(f_row) and len(t_row):
                ft = f_row["top1"].iloc[0]; tt = t_row["top1"].iloc[0]
                L.append(f"| {d_str} | {K} | {ft:.3f} | {tt:.3f} | {ft-tt:+.3f} |")
    L.append("")

    out_path.write_text("\n".join(L))


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)},  instances: {df.instance.nunique()}")

    print("\n=== Running 75/25 split ===")
    s_75 = evaluate_protocol(df, xpr.deployed_split, "deployed_75_25", GRID)
    print("\n=== Running 5-fold CV ===")
    s_5f = evaluate_protocol(df, lambda d: xpr.fivefold_splits(d), "5fold", GRID)
    print("\n=== Running LOIO ===")
    s_lo = evaluate_protocol(df, xpr.loio_splits, "LOIO", GRID)

    state_all = pd.concat([s_75, s_5f, s_lo], ignore_index=True)
    print(f"\nTotal per-state rows: {len(state_all)}")

    print("Aggregating...")
    agg = agg_per_protocol_model(state_all)
    agg.to_csv(TBL / "ranking_grid_aggregate.csv", index=False)

    p_lo = agg_per_instance(state_all, "LOIO")
    p_lo.to_csv(TBL / "ranking_grid_per_instance_loio.csv", index=False)
    p_5f = agg_per_instance(state_all, "5fold")
    p_5f.to_csv(TBL / "ranking_grid_per_instance_5fold.csv", index=False)
    side = side_by_side(p_5f, p_lo)
    side.to_csv(TBL / "side_by_side_5fold_vs_loio.csv", index=False)

    # ----- Heatmaps -----
    print("Heatmaps...")
    proto_titles = {"deployed_75_25": "75/25 split", "5fold": "5-fold CV", "LOIO": "LOIO"}
    for proto in ["deployed_75_25", "5fold", "LOIO"]:
        bap = agg[(agg.protocol == proto) & (agg.model == "best_avg_pair")]
        bap_top1 = float(bap["top1"].iloc[0]) if len(bap) else None
        for kind in ("forest", "tree"):
            heatmap(agg, kind, proto, "top1",
                    title=f"{kind.title()} top-1 ({proto_titles[proto]})",
                    out_png=FIG / f"heatmap_{kind}_{proto}_top1.png",
                    cmap="viridis",
                    baseline_value=bap_top1, baseline_label="best_avg_pair top-1",
                    invert_better=False)

    # Relative regret (forests, LOIO) — uses reversed cmap so dark = good
    heatmap(agg, "forest", "LOIO", "rel_reg",
            title="Forest LOIO mean relative regret (%)",
            out_png=FIG / "heatmap_forest_loio_rel_regret.png",
            cmap="viridis_r",
            baseline_value=float(agg[(agg.protocol=="LOIO") & (agg.model=="best_avg_pair")]["rel_reg"].iloc[0]),
            baseline_label="best_avg_pair rel regret (%)",
            invert_better=True)

    # ----- Report -----
    write_report(agg, p_lo, p_5f, side, OUT / "RANKING_GRID_REPORT.md")
    print(f"\nWrote {OUT / 'RANKING_GRID_REPORT.md'} + 4 CSVs + 7 heatmaps")


if __name__ == "__main__":
    main()
