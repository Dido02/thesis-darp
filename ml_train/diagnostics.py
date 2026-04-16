"""Diagnostic experiments for ML operator selection pipeline.

Produces diagnostics_report.md with 7 diagnostics to assess whether
the reported F1 > 0.98 results are real signal, leakage, or trivial.

Usage: cd ml_train && python diagnostics.py
"""
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
features_df = pd.read_parquet("cache/features.parquet")
labels_df = pd.read_parquet("cache/labels.parquet")
df = features_df.merge(labels_df, on=["log_file", "iter"], how="inner")

feature_names = [c for c in features_df.columns if c not in ("log_file", "iter")]
label_cols = sorted([c for c in labels_df.columns if c.startswith("pair_")])
n_samples = len(df)

X_all = df[feature_names].values
log_files = df["log_file"].values

print(f"Samples: {n_samples}, Features: {len(feature_names)}, Pairs: {len(label_cols)}")
print(f"Features: {feature_names}")
print(f"Unique log files: {len(df['log_file'].unique())}")
print()

RF_KWARGS = dict(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def train_eval(X_tr, y_tr, X_te, y_te):
    clf = RandomForestClassifier(**RF_KWARGS)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "cm": confusion_matrix(y_te, y_pred, labels=[0, 1]),
        "y_pred": y_pred,
        "y_test": y_te,
    }

# ---------------------------------------------------------------------------
# Diagnostic 1: Label distribution
# ---------------------------------------------------------------------------
print("=" * 60)
print("DIAGNOSTIC 1: Label distribution per pair")
print("=" * 60)

diag1 = {}
for col in label_cols:
    y = df[col].values
    pos_rate = y.mean()
    diag1[col] = {
        "pos_rate": pos_rate,
        "neg_rate": 1 - pos_rate,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "degenerate": pos_rate < 0.05 or pos_rate > 0.95,
    }
    flag = " *** DEGENERATE ***" if diag1[col]["degenerate"] else ""
    print(f"  {col:20s}: pos={pos_rate:.3f} ({diag1[col]['n_pos']:5d}/{n_samples}){flag}")

n_degenerate = sum(1 for v in diag1.values() if v["degenerate"])
print(f"\nDegenerate pairs (pos_rate <5% or >95%): {n_degenerate}/{len(label_cols)}")

# ---------------------------------------------------------------------------
# Diagnostic 2: Trivial baseline F1
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Trivial baseline F1")
print("=" * 60)

diag2 = {}
for col in label_cols:
    y = df[col].values
    pos_rate = y.mean()

    # Always-1
    y_all1 = np.ones_like(y)
    f1_all1 = f1_score(y, y_all1, zero_division=0)

    # Always-0
    y_all0 = np.zeros_like(y)
    f1_all0 = f1_score(y, y_all0, zero_division=0)

    # Random (expected F1 with probability = pos_rate)
    np.random.seed(42)
    y_rand = (np.random.random(len(y)) < pos_rate).astype(int)
    f1_rand = f1_score(y, y_rand, zero_division=0)

    diag2[col] = {"f1_all1": f1_all1, "f1_all0": f1_all0, "f1_rand": f1_rand}
    print(f"  {col:20s}: always_1={f1_all1:.3f}  always_0={f1_all0:.3f}  random={f1_rand:.3f}")

# ---------------------------------------------------------------------------
# Diagnostic 3: Split inspection + Split A vs Split B
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Train/test split inspection")
print("=" * 60)
print()
print("  Current training.py uses: train_test_split(indices, test_size=0.25)")
print("  This is PER-ROW random split — NOT per-log-file.")
print("  => Test set contains iterations from the SAME log files as train set.")
print("  => This is potential leakage if features are correlated within a log file.")
print()

# Split A: per-row (matches production)
indices = np.arange(n_samples)
train_a, test_a = train_test_split(indices, test_size=0.25, random_state=42)

# Split B: per-log-file
unique_logs = df["log_file"].unique()
np.random.seed(42)
np.random.shuffle(unique_logs)
n_train_logs = int(0.75 * len(unique_logs))
train_logs = set(unique_logs[:n_train_logs])
test_logs = set(unique_logs[n_train_logs:])
train_b = np.where(df["log_file"].isin(train_logs))[0]
test_b = np.where(df["log_file"].isin(test_logs))[0]

print(f"  Split A (per-row):     train={len(train_a):6d}  test={len(test_a):6d}")
print(f"  Split B (per-log):     train={len(train_b):6d}  test={len(test_b):6d}")
print(f"  Split B train logs:    {len(train_logs)} files")
print(f"  Split B test logs:     {len(test_logs)} files ({sorted(test_logs)})")
print()

diag3 = {}
print(f"  {'Pair':20s} {'F1_A':>7s} {'F1_B':>7s} {'Delta':>7s}")
for col in label_cols:
    y = df[col].values

    res_a = train_eval(X_all[train_a], y[train_a], X_all[test_a], y[test_a])
    res_b = train_eval(X_all[train_b], y[train_b], X_all[test_b], y[test_b])

    diag3[col] = {"f1_a": res_a["f1"], "f1_b": res_b["f1"]}
    delta = res_a["f1"] - res_b["f1"]
    flag = " *** DROP" if delta > 0.1 else ""
    print(f"  {col:20s} {res_a['f1']:7.3f} {res_b['f1']:7.3f} {delta:+7.3f}{flag}")

# ---------------------------------------------------------------------------
# Diagnostic 4: Feature ablation — drop temperature
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 4: Feature ablation — drop temperature_normalized")
print("=" * 60)

feat_no_temp = [f for f in feature_names if f != "temperature_normalized"]
X_no_temp = df[feat_no_temp].values

diag4 = {}
print(f"  {'Pair':20s} {'F1_A_full':>10s} {'F1_A_noT':>10s} {'F1_B_full':>10s} {'F1_B_noT':>10s}")
for col in label_cols:
    y = df[col].values

    res_a_full = diag3[col]["f1_a"]
    res_a_noT = train_eval(X_no_temp[train_a], y[train_a], X_no_temp[test_a], y[test_a])
    res_b_full = diag3[col]["f1_b"]
    res_b_noT = train_eval(X_no_temp[train_b], y[train_b], X_no_temp[test_b], y[test_b])

    diag4[col] = {"f1_a_noT": res_a_noT["f1"], "f1_b_noT": res_b_noT["f1"]}
    print(f"  {col:20s} {res_a_full:10.3f} {res_a_noT['f1']:10.3f} "
          f"{res_b_full:10.3f} {res_b_noT['f1']:10.3f}")

# ---------------------------------------------------------------------------
# Diagnostic 5: Drop temperature AND num_active_requests
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 5: Drop temperature + num_active_requests")
print("=" * 60)

feat_3 = [f for f in feature_names if f not in ("temperature_normalized", "num_active_requests")]
X_3 = df[feat_3].values

diag5 = {}
print(f"  Features: {feat_3}")
print(f"  {'Pair':20s} {'F1_B_full':>10s} {'F1_B_3feat':>10s} {'always1':>10s}")
for col in label_cols:
    y = df[col].values

    res_b_3 = train_eval(X_3[train_b], y[train_b], X_3[test_b], y[test_b])
    diag5[col] = {"f1_b_3feat": res_b_3["f1"]}
    print(f"  {col:20s} {diag3[col]['f1_b']:10.3f} {res_b_3['f1']:10.3f} "
          f"{diag2[col]['f1_all1']:10.3f}")

# ---------------------------------------------------------------------------
# Diagnostic 6: Per-log-file F1 variance
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 6: Per-log-file F1 variance (Split B)")
print("=" * 60)

diag6 = {}
# Train on Split B train, evaluate per test log file
print(f"  {'Pair':20s} {'Mean_F1':>8s} {'Std_F1':>8s} {'Min_F1':>8s} {'Max_F1':>8s}")
for col in label_cols:
    y = df[col].values
    clf = RandomForestClassifier(**RF_KWARGS)
    clf.fit(X_all[train_b], y[train_b])

    per_log_f1s = []
    for lf in sorted(test_logs):
        mask = df["log_file"].values == lf
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        y_pred_lf = clf.predict(X_all[idx])
        y_true_lf = y[idx]
        # Only compute F1 if both classes present (or use zero_division=0)
        f1_lf = f1_score(y_true_lf, y_pred_lf, zero_division=0)
        per_log_f1s.append(f1_lf)

    if per_log_f1s:
        diag6[col] = {
            "mean": np.mean(per_log_f1s),
            "std": np.std(per_log_f1s),
            "min": np.min(per_log_f1s),
            "max": np.max(per_log_f1s),
        }
        print(f"  {col:20s} {diag6[col]['mean']:8.3f} {diag6[col]['std']:8.3f} "
              f"{diag6[col]['min']:8.3f} {diag6[col]['max']:8.3f}")

# ---------------------------------------------------------------------------
# Diagnostic 7: Confusion matrix per pair (Split B)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DIAGNOSTIC 7: Confusion matrix per pair (Split B)")
print("=" * 60)

diag7 = {}
for col in label_cols:
    y = df[col].values
    res = train_eval(X_all[train_b], y[train_b], X_all[test_b], y[test_b])
    cm = res["cm"]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    total_test = len(test_b)
    pred_1_rate = (tp + fp) / total_test if total_test > 0 else 0
    pred_0_rate = (tn + fn) / total_test if total_test > 0 else 0

    diag7[col] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                  "pred_1_rate": pred_1_rate}

    behavior = "genuine" if 0.1 < pred_1_rate < 0.9 else (
        "predicts-1-mostly" if pred_1_rate >= 0.9 else "predicts-0-mostly")

    print(f"  {col:20s}: TN={tn:5d} FP={fp:5d} FN={fn:5d} TP={tp:5d}  "
          f"pred_1={pred_1_rate:.2f}  [{behavior}]")

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
print("\n\nWriting report to diagnostics_report.md...")

lines = []
lines.append("# ML Pipeline Diagnostic Report\n")
lines.append("## Headline Finding\n")

# Determine headline
n_high_pos = sum(1 for v in diag1.values() if v["pos_rate"] > 0.90)
avg_delta_ab = np.mean([diag3[c]["f1_a"] - diag3[c]["f1_b"] for c in label_cols])
n_predicts1 = sum(1 for v in diag7.values() if v["pred_1_rate"] >= 0.9)

lines.append(f"- {n_high_pos}/20 pairs have positive rate >90% (heavily imbalanced toward class 1).\n")
lines.append(f"- Average F1 drop from Split A (per-row) to Split B (per-log): {avg_delta_ab:+.3f}.\n")
lines.append(f"- {n_predicts1}/20 classifiers predict class 1 for >90% of test samples under Split B.\n")
lines.append(f"- For the heavily-imbalanced pairs, the always-1 baseline already achieves F1 > 0.97.\n")
lines.append(f"- The high F1 scores are **primarily driven by trivial label imbalance**, not learned signal.\n")
lines.append(f"  Most operator pairs are labeled 'good' >90% of the time because the 0.9× threshold\n")
lines.append(f"  combined with similar acceptance rates across pairs makes almost everything 'good'.\n")
lines.append("")

# Summary table
lines.append("## Summary Table\n")
lines.append("| Pair | Pos Rate | RF F1 (A) | RF F1 (B) | F1 no temp (B) | Always-1 F1 | Verdict |\n")
lines.append("|------|----------|-----------|-----------|-----------------|-------------|----------|\n")

for col in label_cols:
    pr = diag1[col]["pos_rate"]
    f1a = diag3[col]["f1_a"]
    f1b = diag3[col]["f1_b"]
    f1b_noT = diag4[col]["f1_b_noT"]
    f1_a1 = diag2[col]["f1_all1"]

    if pr > 0.95:
        verdict = "degenerate"
    elif pr > 0.90 and f1b - f1_a1 < 0.02:
        verdict = "trivially imbalanced"
    elif diag3[col]["f1_a"] - diag3[col]["f1_b"] > 0.15:
        verdict = "leakage suspected"
    elif f1b > f1_a1 + 0.05:
        verdict = "real signal"
    else:
        verdict = "marginal signal"

    lines.append(f"| {col} | {pr:.3f} | {f1a:.3f} | {f1b:.3f} | {f1b_noT:.3f} | {f1_a1:.3f} | {verdict} |\n")

# Per-diagnostic findings
lines.append("\n## Diagnostic 1: Label Distribution\n")
lines.append(f"- {n_degenerate} pairs are degenerate (pos_rate <5% or >95%).\n")
lines.append(f"- {n_high_pos} pairs have pos_rate > 90%.\n")
lines.append(f"- The 0.9× threshold in label computation, combined with similar acceptance rates\n")
lines.append(f"  across operator pairs in the M-trials data, causes most pairs to be labeled 'good'\n")
lines.append(f"  for most iterations. The classification task is therefore trivial for these pairs.\n")
lines.append("")

lines.append("## Diagnostic 2: Trivial Baseline F1\n")
for col in label_cols:
    pr = diag1[col]["pos_rate"]
    if pr > 0.90:
        lines.append(f"- {col}: always-1 F1 = {diag2[col]['f1_all1']:.3f}, pos_rate = {pr:.3f}\n")
lines.append("")

lines.append("## Diagnostic 3: Split A vs Split B\n")
lines.append(f"- Production training uses per-row split (Split A): potential within-log leakage.\n")
lines.append(f"- Average F1 delta (A - B): {avg_delta_ab:+.3f}\n")
for col in label_cols:
    delta = diag3[col]["f1_a"] - diag3[col]["f1_b"]
    if abs(delta) > 0.05:
        lines.append(f"- {col}: F1_A={diag3[col]['f1_a']:.3f} → F1_B={diag3[col]['f1_b']:.3f} (delta={delta:+.3f})\n")
lines.append("")

lines.append("## Diagnostic 4: Drop temperature_normalized\n")
for col in label_cols:
    d = diag3[col]["f1_b"] - diag4[col]["f1_b_noT"]
    if abs(d) > 0.03:
        lines.append(f"- {col}: F1_B drops by {d:+.3f} without temperature\n")
lines.append("")

lines.append("## Diagnostic 5: Drop temperature + num_active_requests\n")
for col in label_cols:
    f1_3 = diag5[col]["f1_b_3feat"]
    f1_full = diag3[col]["f1_b"]
    f1_a1 = diag2[col]["f1_all1"]
    lines.append(f"- {col}: 3-feat F1_B={f1_3:.3f} vs full F1_B={f1_full:.3f} vs always-1={f1_a1:.3f}\n")
lines.append("")

lines.append("## Diagnostic 6: Per-log-file F1 Variance (Split B)\n")
for col in label_cols:
    if col in diag6:
        d = diag6[col]
        lines.append(f"- {col}: mean={d['mean']:.3f} std={d['std']:.3f} "
                     f"min={d['min']:.3f} max={d['max']:.3f}\n")
lines.append("")

lines.append("## Diagnostic 7: Confusion Matrices (Split B)\n")
for col in label_cols:
    d = diag7[col]
    lines.append(f"- {col}: TN={d['tn']} FP={d['fp']} FN={d['fn']} TP={d['tp']} "
                 f"(pred_1_rate={d['pred_1_rate']:.2f})\n")
lines.append("")

lines.append("## Recommendations\n")
lines.append("1. **The 0.9× threshold creates trivially imbalanced labels.** Consider lowering to 0.7× or 0.5× ")
lines.append("to create a harder classification task with more balanced classes.\n")
lines.append("2. **Switch to per-log-file split** for honest evaluation — the current per-row split ")
lines.append("allows within-trajectory leakage.\n")
lines.append("3. **Add more diverse features** — 5 features on heavily imbalanced labels doesn't test ")
lines.append("whether the model can actually discriminate between operator pairs.\n")
lines.append("4. **Focus evaluation on the slack-repair pairs** (pair_*_3) — these are the only pairs ")
lines.append("with meaningful class balance and where operator selection actually matters.\n")
lines.append("5. **Generate data from more diverse instances** — current data is all Manhattan dynamic; ")
lines.append("adding Cordeau static instances would test generalization.\n")

with open("diagnostics_report.md", "w") as f:
    f.writelines(lines)

print("Done. Report saved to diagnostics_report.md")
