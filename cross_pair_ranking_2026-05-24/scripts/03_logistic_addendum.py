#!/usr/bin/env python3
"""Multi-class logistic regression addendum.

Adds one more configuration to the cross-pair ranking analysis: a single
multi-class LogisticRegression that directly predicts which of the 9 USED
pairs has the highest ei_pct at each state.

Setup:
  - sklearn LogisticRegression(multi_class='multinomial', solver='lbfgs',
    class_weight='balanced', max_iter=1000, random_state=42)
  - 20 input features, 9 output classes
  - StandardScaler fit on TRAIN per fold, applied to TEST
  - Train and test BOTH filtered to states with max(ei_pct) > 0
  - Label per state = argmax(ei_pct) over the 9 USED pairs

Evaluation: same 3 protocols (75/25, 5-fold, LOIO), same 4 metrics
(top-1, top-3, mean abs regret, mean rel regret), state-weighted.

Outputs (updates to existing CSVs + new addendum MD):
  state_level_predictions.csv         — appended LR per-state rows
  aggregate_per_protocol.csv          — re-aggregated (LR row added per protocol)
  per_instance_loio.csv               — LR rows added per (size, delay)
  per_instance_5fold.csv              — LR rows added per (size, delay)
  side_by_side.csv                    — LR rows added per instance
  LOGISTIC_ADDENDUM.md                — new addendum describing model + results

No existing models retrained. No files under ml_train/models_v*/ touched.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/didodimitrov/Documents/thesis-darp")
SCRIPTS = REPO / "cross_pair_ranking_2026-05-24" / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Reuse helpers + constants from the patched prior analysis
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
xpr = _load("xpr", SCRIPTS / "01_cross_pair_ranking.py")
fix = _load("fix", SCRIPTS / "02_fix_baselines.py")

OUT = REPO / "cross_pair_ranking_2026-05-24"
TBL = OUT / "tables"

LR_MODEL_TAG = "logistic_9class"


# ============================================================
# Fit + predict for the multi-class LR
# ============================================================
def fit_and_predict_lr(df, train_mask, test_mask):
    """Returns per-state proba matrix (n_test_valid, 9) and the array of
    test-row indices (into the test slice) that were valid (max_ei > 0).

    The training set is filtered to valid states only (no signal at
    max_ei=0 states for the argmax-ei_pct label)."""
    EI = df[xpr.USED_EI].values
    X = df[xpr.FEATURE_NAMES_20].values

    # Train: valid states in train mask
    tr_idx = np.where(train_mask)[0]
    ei_tr = EI[tr_idx]
    tr_valid = ei_tr.max(axis=1) > 0
    tr_idx_v = tr_idx[tr_valid]
    y_tr = ei_tr[tr_valid].argmax(axis=1)            # 9-class label
    X_tr = X[tr_idx_v]

    # Test: valid states in test mask (caller will filter to these too)
    te_idx = np.where(test_mask)[0]
    ei_te = EI[te_idx]
    te_valid = ei_te.max(axis=1) > 0
    te_idx_v = te_idx[te_valid]
    X_te = X[te_idx_v]

    if len(np.unique(y_tr)) < 2 or len(X_tr) < 9 or len(X_te) == 0:
        return None

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # In sklearn >=1.7, the `multi_class` kwarg is removed; the `lbfgs`
    # solver fits multinomial logistic regression by default whenever
    # y has >2 classes. Behavior identical to multi_class='multinomial'.
    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=1000, random_state=42,
    )
    clf.fit(X_tr_s, y_tr)

    # predict_proba returns shape (n, n_classes_seen). If a class was absent
    # from train (e.g. one pair was never argmax there), it's missing.
    # Build a full (n, 9) matrix by mapping clf.classes_ → column index.
    proba_seen = clf.predict_proba(X_te_s)
    P = np.zeros((len(X_te_s), 9), dtype=float)
    for col, cls in enumerate(clf.classes_):
        P[:, int(cls)] = proba_seen[:, col]
    return P, te_idx_v


# ============================================================
# Per-state metric rows for one fold
# ============================================================
def evaluate_lr_one_fold(df, fold, protocol_label):
    """Returns list-of-dict rows in the same schema as the existing
    state_level_predictions.csv."""
    tr, te = fold["train_mask"], fold["test_mask"]
    out = fit_and_predict_lr(df, tr, te)
    if out is None:
        return []
    P, te_idx_v = out
    E_valid = df[xpr.USED_EI].values[te_idx_v]
    inst_valid = df["instance"].values[te_idx_v]
    true_best = E_valid.argmax(axis=1)
    pred_rank = np.argsort(-P, axis=1)
    pred_top1 = pred_rank[:, 0]
    in_top3 = np.array([true_best[i] in pred_rank[i, :3] for i in range(len(true_best))])
    arange = np.arange(len(E_valid))
    abs_reg = E_valid[arange, true_best] - E_valid[arange, pred_top1]
    rel_reg = abs_reg / E_valid[arange, true_best] * 100.0

    return [{
        "protocol": protocol_label, "fold": fold["name"],
        "model": LR_MODEL_TAG, "instance": inst_valid[k],
        "top1_hit": float(pred_top1[k] == true_best[k]),
        "top3_hit": float(in_top3[k]),
        "abs_reg": float(abs_reg[k]),
        "rel_reg": float(rel_reg[k]),
    } for k in range(len(E_valid))]


# ============================================================
# Re-aggregate (state-weighted, matches patched 02_fix_baselines.py)
# ============================================================
def agg_per_protocol_model_pooled(df_state):
    out = []
    for (p, m), sub in df_state.groupby(["protocol", "model"]):
        t3 = float(sub["top3_hit"].mean()) if sub["top3_hit"].notna().any() else np.nan
        out.append({"protocol": p, "model": m, "n_states": int(len(sub)),
                    "top1": float(sub["top1_hit"].mean()), "top3": t3,
                    "abs_reg": float(sub["abs_reg"].mean()),
                    "rel_reg": float(sub["rel_reg"].mean())})
    return pd.DataFrame(out)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + reproducing folds (no retraining of existing models)...")
    df = xpr.load_data()
    print(f"  joined rows: {len(df)}")

    # Existing state-level predictions
    state_csv = TBL / "state_level_predictions.csv"
    existing = pd.read_csv(state_csv)
    print(f"  loaded existing state_level_predictions.csv: {len(existing)} rows")
    if LR_MODEL_TAG in existing["model"].unique():
        print(f"  WARNING: {LR_MODEL_TAG} rows already present, dropping for re-fit")
        existing = existing[existing["model"] != LR_MODEL_TAG]

    # Generate LR rows for all 3 protocols
    new_rows = []
    print("\n[deployed_75_25] 1 fold")
    for fold in xpr.deployed_split(df):
        rows = evaluate_lr_one_fold(df, fold, "deployed_75_25")
        print(f"  {fold['name']}: {len(rows)} valid test states")
        new_rows.extend(rows)

    print("\n[5fold] 5 folds")
    for fold in xpr.fivefold_splits(df):
        rows = evaluate_lr_one_fold(df, fold, "5fold")
        print(f"  {fold['name']}: {len(rows)} valid test states")
        new_rows.extend(rows)

    print("\n[LOIO] 14 folds")
    for fold in xpr.loio_splits(df):
        rows = evaluate_lr_one_fold(df, fold, "LOIO")
        print(f"  {fold['name']}: {len(rows)} valid test states")
        new_rows.extend(rows)

    new_state = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_state], ignore_index=True)
    combined.to_csv(state_csv, index=False)
    print(f"\nAppended {len(new_state)} LR per-state rows -> "
          f"state_level_predictions.csv now has {len(combined)} rows")

    # Re-aggregate
    print("\nRe-aggregating...")
    agg = agg_per_protocol_model_pooled(combined)
    agg.to_csv(TBL / "aggregate_per_protocol.csv", index=False)

    p_lo = xpr.agg_per_instance(combined, "LOIO")
    p_lo.to_csv(TBL / "per_instance_loio.csv", index=False)
    p_5f = xpr.agg_per_instance(combined, "5fold")
    p_5f.to_csv(TBL / "per_instance_5fold.csv", index=False)
    side = xpr.side_by_side(p_5f, p_lo)
    side.to_csv(TBL / "side_by_side.csv", index=False)

    # ----- Print LR summary to stdout for quick read -----
    lr_agg = agg[agg.model == LR_MODEL_TAG].sort_values("protocol")
    print("\nLR aggregate (state-weighted):")
    print(lr_agg.to_string(index=False))

    # ----- Write addendum MD -----
    L = []
    L.append("# Logistic Regression Addendum — multi-class ranker")
    L.append("")
    L.append("**Date:** 2026-05-24")
    L.append("")
    L.append("## Model")
    L.append("")
    L.append("- `sklearn.linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)`")
    L.append("  (In sklearn ≥1.7 the `multi_class` kwarg was removed; the `lbfgs` solver fits multinomial logistic regression by default whenever `y` has >2 classes. Behavior is identical to the previous `multi_class='multinomial'`.)")
    L.append("- One model. 20 input features. 9 output classes (one per USED pair).")
    L.append("- Feature preprocessing: `StandardScaler` fit on the training fold and applied to the test fold (zero mean, unit variance).")
    L.append("- Training label per state: `argmax(ei_pct over 9 USED pairs)` — i.e. the pair with the highest oracle-expected improvement at that state.")
    L.append("- State filter: train and test BOTH restricted to states with `max(ei_pct) > 0`. States with no improving pair are dropped (the argmax label is undefined / arbitrary).")
    L.append("- Same valid mask used to compute the 4 metrics, so all aggregation is state-weighted and directly comparable to the existing forest/tree configs and the baselines.")
    L.append("")
    L.append("Why this configuration: the forests and trees in the existing analysis are 9 INDEPENDENT per-pair binary classifiers trained on the t07 threshold label. The deployed selector then stacks their 9 probabilities and samples. The LR here is a single JOINT model that directly predicts the ranking — a structurally different approach, but evaluated under the same metric and ground truth so it's apples-to-apples.")
    L.append("")

    L.append("## Aggregate results (state-weighted)")
    L.append("")
    L.append("| protocol | n states | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for proto in ["deployed_75_25", "5fold", "LOIO"]:
        r = lr_agg[lr_agg.protocol == proto]
        if len(r) == 0: continue
        r = r.iloc[0]
        L.append(f"| {proto} | {r['n_states']} | {r['top1']:.3f} | {r['top3']:.3f} | "
                 f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    # Side-by-side with the closest comparable model: deployed forest_d_none_K20 + best_avg_pair
    L.append("## Comparison vs deployed forest and best_avg_pair")
    L.append("")
    L.append("| protocol | model | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---|---:|---:|---:|---:|")
    for proto in ["deployed_75_25", "5fold", "LOIO"]:
        for m in [LR_MODEL_TAG, "forest_d_none_K20", "best_avg_pair", "random"]:
            r = agg[(agg.protocol == proto) & (agg.model == m)]
            if len(r) == 0: continue
            r = r.iloc[0]
            t3 = f"{r['top3']:.3f}" if pd.notna(r["top3"]) else "n/a"
            L.append(f"| {proto} | `{m}` | {r['top1']:.3f} | {t3} | "
                     f"{r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
        L.append("")

    L.append("## Per-instance LOIO breakdown")
    L.append("")
    L.append("| instance | n | top-1 | top-3 | abs regret | rel regret (%) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    sub_lo = p_lo[p_lo.model == LR_MODEL_TAG].sort_values("instance")
    for _, r in sub_lo.iterrows():
        L.append(f"| {r['instance']} | {r['n_states']} | {r['top1']:.3f} | "
                 f"{r['top3']:.3f} | {r['abs_reg']:.4f} | {r['rel_reg']:.2f}% |")
    L.append("")

    L.append("## 5-fold vs LOIO transfer (LR only)")
    L.append("")
    L.append("| instance | 5-fold top-1 | LOIO top-1 | Δ |")
    L.append("|---|---:|---:|---:|")
    sub_side = side[side.model == LR_MODEL_TAG].sort_values("instance")
    for _, r in sub_side.iterrows():
        def f(v): return "n/a" if pd.isna(v) else f"{v:.3f}"
        L.append(f"| {r['instance']} | {f(r['fold5_top1'])} | {f(r['loio_top1'])} | "
                 f"{f(r['delta_top1'])} |")
    L.append("")

    (OUT / "LOGISTIC_ADDENDUM.md").write_text("\n".join(L))
    print(f"\nWrote {OUT / 'LOGISTIC_ADDENDUM.md'}")
    print(f"     + updated {state_csv}")
    print(f"     + updated {TBL / 'aggregate_per_protocol.csv'}")
    print(f"     + updated {TBL / 'per_instance_loio.csv'}")
    print(f"     + updated {TBL / 'per_instance_5fold.csv'}")
    print(f"     + updated {TBL / 'side_by_side.csv'}")


if __name__ == "__main__":
    main()
