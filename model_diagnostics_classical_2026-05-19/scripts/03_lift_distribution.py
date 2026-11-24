#!/usr/bin/env python3
"""Emit a richer per-variant metric distribution: mean, median, std, min, max, n.

For every variant we computed during the classical analysis, gather the
per-USED-pair F1-lift (classifiers) or MAE-lift (regressors), filtered to
non-degenerate pairs (positive rate in [0.05, 0.95] for classifiers; train
label std >= 1e-3 for regressors). Emit:

  tables/lift_distribution_used.csv

Columns:
  variant, model_kind, model_type, split (train|test),
  metric (f1_lift|mae_lift|f1|precision|recall|roc_auc),
  n, mean, median, std, min, max
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OLD_ENRICH = REPO / "model_diagnostics_2026-05-19" / "enriched_reports"
NEW_ENRICH = REPO / "model_diagnostics_classical_2026-05-19" / "enriched_reports_classical"
TBL = REPO / "model_diagnostics_classical_2026-05-19" / "tables"

USED_PAIRS = {f"pair_{d}_{r}" for d in (0, 1, 2) for r in (0, 1, 2)}

CLASSIFIER_METRICS = ["f1", "f1_lift", "precision", "recall", "specificity",
                      "accuracy", "roc_auc", "pr_auc", "brier", "mcc"]
REGRESSOR_METRICS = ["mae", "rmse", "r2", "mae_lift"]


def load_all() -> dict[str, dict]:
    out = {}
    for src in (OLD_ENRICH, NEW_ENRICH):
        if not src.exists():
            continue
        for p in sorted(src.glob("*.json")):
            out[p.stem] = json.loads(p.read_text())
    return out


def is_clf(enr: dict) -> bool:
    return enr["model_kind"] == "classifier"


def non_degenerate_used(enr: dict) -> list[dict]:
    """Keep USED pairs that are non-degenerate (filter applies to test prevalence
    for classifiers; train label std for regressors)."""
    keep = []
    for p in enr["pairs"]:
        if p["pair_name"] not in USED_PAIRS:
            continue
        if is_clf(enr):
            n = p.get("test_n", 0)
            pos = p.get("test_n_positive", 0)
            if n == 0:
                continue
            r = pos / n
            if r < 0.05 or r > 0.95:
                continue
        else:
            std = p.get("train_label_std")
            if std is None or std < 1e-3:
                continue
        keep.append(p)
    return keep


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": "", "median": "", "std": "", "min": "", "max": ""}
    return {
        "n": len(values),
        "mean": float(st.mean(values)),
        "median": float(st.median(values)),
        "std": float(st.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    print(f"Loading enriched reports...")
    all_enr = load_all()
    print(f"  {len(all_enr)} variants")
    TBL.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, enr in all_enr.items():
        pairs = non_degenerate_used(enr)
        metrics = CLASSIFIER_METRICS if is_clf(enr) else REGRESSOR_METRICS
        for split in ("train", "test"):
            for m in metrics:
                key = f"{split}_{m}"
                vals = [p.get(key) for p in pairs if isinstance(p.get(key), (int, float))]
                s = stats(vals)
                rows.append({
                    "variant": name, "model_kind": enr["model_kind"],
                    "model_type": enr["model_type"], "split": split, "metric": m,
                    **s,
                })

    out_path = TBL / "lift_distribution_used.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "model_kind", "model_type",
                                          "split", "metric",
                                          "n", "mean", "median", "std", "min", "max"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
