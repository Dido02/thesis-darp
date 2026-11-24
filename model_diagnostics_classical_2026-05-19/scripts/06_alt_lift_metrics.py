#!/usr/bin/env python3
"""Alternative lift metrics — F1-lift is statistically clean but not intuitive.
Compute lifts for more readable metrics, all on the 9 USED non-degenerate pairs.

For each (variant, split, pair) we compute:
  - **accuracy_lift**  = accuracy − max(p, 1−p)    (majority-class baseline)
  - **precision_lift** = precision − p             (base-rate baseline; "if I always predicted 1, my precision would be p")
  - **balanced_accuracy_lift** = (recall + specificity)/2 − 0.5   (0.5 = random)
  - **roc_auc_lift**   = roc_auc − 0.5             (already a lift over chance)

Where p = test_n_positive / test_n is per-pair positive prevalence.

These compose into a single "interpretable lift dashboard" per variant. Output:
  tables/alt_lift_metrics_used.csv
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


def load_all():
    out = {}
    for src in (OLD_ENRICH, NEW_ENRICH):
        if not src.exists(): continue
        for p in sorted(src.glob("*.json")):
            out[p.stem] = json.loads(p.read_text())
    return out


def non_degenerate_clf_used(enr):
    keep = []
    for p in enr["pairs"]:
        if p["pair_name"] not in USED_PAIRS: continue
        n = p.get("test_n", 0); pos = p.get("test_n_positive", 0)
        if n == 0: continue
        r = pos / n
        if r < 0.05 or r > 0.95: continue
        keep.append(p)
    return keep


def per_pair_alt_lifts(p):
    """Compute alt-lifts for one pair, for both splits."""
    out = {"pair_name": p["pair_name"]}
    for split in ("train", "test"):
        n = p.get(f"{split}_n", 0)
        pos = p.get(f"{split}_n_positive", 0)
        if n == 0: continue
        prev = pos / n
        acc = p.get(f"{split}_accuracy")
        prec = p.get(f"{split}_precision")
        rec = p.get(f"{split}_recall")
        spec = p.get(f"{split}_specificity")
        roc = p.get(f"{split}_roc_auc")
        f1 = p.get(f"{split}_f1")

        out[f"{split}_prevalence"] = prev
        out[f"{split}_accuracy_lift"] = (acc - max(prev, 1 - prev)) if isinstance(acc, (int, float)) else None
        out[f"{split}_precision_lift"] = (prec - prev) if isinstance(prec, (int, float)) else None
        if isinstance(rec, (int, float)) and isinstance(spec, (int, float)):
            bacc = (rec + spec) / 2
            out[f"{split}_balanced_accuracy"] = bacc
            out[f"{split}_balanced_accuracy_lift"] = bacc - 0.5
        else:
            out[f"{split}_balanced_accuracy"] = None
            out[f"{split}_balanced_accuracy_lift"] = None
        out[f"{split}_roc_auc_lift"] = (roc - 0.5) if isinstance(roc, (int, float)) else None
        out[f"{split}_f1_lift"] = p.get(f"{split}_f1_lift")
    return out


def stats(vals):
    vs = [v for v in vals if isinstance(v, (int, float))]
    if not vs: return {"n": 0, "mean": "", "median": "", "std": "", "min": "", "max": ""}
    return {"n": len(vs), "mean": float(st.mean(vs)),
            "median": float(st.median(vs)),
            "std": float(st.stdev(vs)) if len(vs) > 1 else 0.0,
            "min": float(min(vs)), "max": float(max(vs))}


def main():
    print("Loading enriched reports...")
    all_enr = load_all()
    print(f"  {len(all_enr)} variants")
    TBL.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy_lift", "precision_lift", "balanced_accuracy_lift",
               "roc_auc_lift", "f1_lift"]
    rows = []
    for name, enr in all_enr.items():
        if enr["model_kind"] != "classifier":
            continue
        used = non_degenerate_clf_used(enr)
        if not used: continue
        # Compute per-pair alt-lifts
        per_pair = [per_pair_alt_lifts(p) for p in used]
        for split in ("train", "test"):
            for m in metrics:
                vals = [pp.get(f"{split}_{m}") for pp in per_pair]
                s = stats(vals)
                rows.append({"variant": name, "model_kind": "classifier",
                             "model_type": enr["model_type"],
                             "split": split, "metric": m, **s})

    out_path = TBL / "alt_lift_metrics_used.csv"
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
