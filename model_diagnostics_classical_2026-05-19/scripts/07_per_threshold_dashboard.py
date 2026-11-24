#!/usr/bin/env python3
"""Per-threshold dashboard tables.

For each classifier threshold (t02, t03, t05, t07), emit a curated subset of
~8 models — 4 forest variants + 4 tree variants spanning the depth × K grid —
and show every raw metric plus every lift (mean ± std), test split.

Outputs:
  tables/dashboard_t02.csv
  tables/dashboard_t03.csv
  tables/dashboard_t05.csv
  tables/dashboard_t07.csv

Also prints markdown-ready tables to stdout.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TBL = REPO / "model_diagnostics_classical_2026-05-19" / "tables"
LIFT_DIST = TBL / "lift_distribution_used.csv"
ALT_LIFT = TBL / "alt_lift_metrics_used.csv"

# 8 representative configs per threshold:
#   4 FOREST  : (none, K=20) baseline, (4, 20), (8, 20), (8, 8)
#   4 TREE    : (none, K=20) baseline, (2, 8), (4, 8), (8, 8)
# Naming convention:
#   baseline (depth=None, K=20):     models_v3_full_20f_<t>          (forest)
#                                    models_v6_full_20f_<t>_tree    (tree)
#   grid cells:                      grid_<t>_<model>_depth<d>_K<K>
SUBSET = [
    # (display_name, model_type, variant_name_template)
    ("FOREST baseline (depth=none, K=20)",   "forest", "models_v3_full_20f_{t}"),
    ("FOREST depth=4, K=20",                  "forest", "grid_{t}_forest_depth4_K20"),
    ("FOREST depth=8, K=20",                  "forest", "grid_{t}_forest_depth8_K20"),
    ("FOREST depth=8, K=8",                   "forest", "grid_{t}_forest_depth8_K8"),
    ("TREE   baseline (depth=none, K=20)",   "tree",   "models_v6_full_20f_{t}_tree"),
    ("TREE   depth=2, K=8",                   "tree",   "grid_{t}_tree_depth2_K8"),
    ("TREE   depth=4, K=8",                   "tree",   "grid_{t}_tree_depth4_K8"),
    ("TREE   depth=8, K=8",                   "tree",   "grid_{t}_tree_depth8_K8"),
]

RAW_METRICS = ["f1", "accuracy", "precision", "recall", "specificity",
               "roc_auc", "pr_auc", "brier"]
LIFT_METRICS = ["f1_lift", "accuracy_lift", "precision_lift",
                "balanced_accuracy_lift", "roc_auc_lift"]


def load_dist():
    return list(csv.DictReader(open(LIFT_DIST)))


def load_alt():
    return list(csv.DictReader(open(ALT_LIFT)))


def get(rows, variant, split, metric, stat):
    cand = [r for r in rows if r["variant"] == variant and r["split"] == split and r["metric"] == metric]
    if not cand: return None
    try: return float(cand[0][stat])
    except (ValueError, KeyError): return None


def fmt(mean, std):
    if mean is None or std is None: return "n/a"
    return f"{mean:+.3f} ± {std:.3f}"


def main():
    dist = load_dist()
    alt = load_alt()

    for t in ("t02", "t03", "t05", "t07"):
        out_path = TBL / f"dashboard_{t}.csv"
        rows_for_csv = []
        rows_for_md = []

        for display_name, model_type, tmpl in SUBSET:
            v = tmpl.format(t=t)
            row = {"display_name": display_name, "variant": v}

            for m in RAW_METRICS:
                mn = get(dist, v, "test", m, "mean")
                sd = get(dist, v, "test", m, "std")
                row[f"{m}__mean"] = mn
                row[f"{m}__std"] = sd
                row[f"{m}__display"] = fmt(mn, sd)
            for m in LIFT_METRICS:
                # accuracy_lift/precision_lift/balanced_accuracy_lift/roc_auc_lift come from alt;
                # f1_lift is in both files (alt re-computes it, but the dist one is canonical)
                source = dist if m == "f1_lift" else alt
                mn = get(source, v, "test", m, "mean")
                sd = get(source, v, "test", m, "std")
                row[f"{m}__mean"] = mn
                row[f"{m}__std"] = sd
                row[f"{m}__display"] = fmt(mn, sd)
            # Train F1-lift for the gap
            train_f1l = get(dist, v, "train", "f1_lift", "mean")
            row["train_f1_lift__mean"] = train_f1l
            row["gap_f1_lift"] = (train_f1l - row["f1_lift__mean"]) if (train_f1l is not None and row["f1_lift__mean"] is not None) else None

            rows_for_csv.append(row)
            rows_for_md.append(row)

        # Write CSV (flat: one row per model, all stats as columns)
        if rows_for_csv:
            cols = ["display_name", "variant"]
            for m in RAW_METRICS + LIFT_METRICS:
                cols += [f"{m}__mean", f"{m}__std"]
            cols += ["train_f1_lift__mean", "gap_f1_lift"]
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows_for_csv:
                    w.writerow({k: r.get(k, "") for k in cols})

        # Stdout markdown table
        print(f"\n## Dashboard — threshold {t}\n")
        # Group 1: raw metrics
        print("### Raw test metrics (mean ± std over 9 used non-degenerate pairs)\n")
        hdrs = ["model", "F1", "Acc", "Prec", "Recall", "Specif", "ROC-AUC", "PR-AUC", "Brier"]
        print("| " + " | ".join(hdrs) + " |")
        print("|" + "---|" * len(hdrs))
        for r in rows_for_md:
            cells = [r["display_name"][:40]]
            for m in ("f1", "accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc", "brier"):
                cells.append(r[f"{m}__display"])
            print("| " + " | ".join(cells) + " |")

        # Group 2: lifts
        print("\n### Lifts (mean ± std over 9 used non-degenerate pairs)\n")
        hdrs = ["model", "F1-lift", "Acc-lift", "Prec-lift", "BalAcc-lift", "ROC-AUC-lift", "gap (train-test F1-lift)"]
        print("| " + " | ".join(hdrs) + " |")
        print("|" + "---|" * len(hdrs))
        for r in rows_for_md:
            cells = [r["display_name"][:40]]
            for m in ("f1_lift", "accuracy_lift", "precision_lift", "balanced_accuracy_lift", "roc_auc_lift"):
                cells.append(r[f"{m}__display"])
            cells.append(f"{r['gap_f1_lift']:+.3f}" if r["gap_f1_lift"] is not None else "n/a")
            print("| " + " | ".join(cells) + " |")

    print(f"\nWrote tables/dashboard_t02.csv .. dashboard_t07.csv")


if __name__ == "__main__":
    main()
