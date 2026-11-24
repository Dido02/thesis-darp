"""Step C — feature-batch ablation. Train on the same data with progressively
larger feature sets and report aggregate metrics + pair_3_3 specifically.

Bypasses the train CLI's sidecar sanity check by calling train_pipeline()
directly with a restricted feature_names list.

Models written to models_diag_*/ — separate from models_v2_dynamic so we don't
override the headline run.
"""
from __future__ import annotations

from pathlib import Path

from darpml.features import _REGISTRY
from darpml.training import train_pipeline

FEATURES_PATH = "cache_v2_dynamic/features.parquet"
LABELS_PATH = "cache_v2_dynamic/labels.parquet"


def names_for(batches: list[str], mode: str = "dynamic") -> list[str]:
    """Replicates features.get_features() ordering — registry insertion order."""
    bset = set(batches)
    out = []
    for name, entry in _REGISTRY.items():
        if entry.batch not in bset:
            continue
        if entry.mode == "both":
            out.append(name)
        elif entry.mode == mode:
            out.append(name)
    return out


CONFIGS = [
    ("core", ["core"], "models_diag_core"),
    ("core+tightness", ["core", "tightness"], "models_diag_core_t"),
    ("core+tightness+worst_support", ["core", "tightness", "worst_support"],
     "models_diag_core_tw"),
    ("all 5 batches",
     ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"],
     "models_diag_all"),
]


def main():
    summary_rows = []
    for label, batches, out_dir in CONFIGS:
        feat_names = names_for(batches)
        print(f"\n{'#' * 80}")
        print(f"# {label}: {len(feat_names)} features")
        print(f"# batches: {batches}")
        print(f"# features: {feat_names}")
        print(f"# out_dir: {out_dir}")
        print(f"{'#' * 80}\n")

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report = train_pipeline(
            features_path=FEATURES_PATH,
            labels_path=LABELS_PATH,
            output_dir=out_dir,
            n_estimators=100,
            feature_names=feat_names,
            mode="dynamic",
            batches=batches,
        )

        # Per-pair lift summary
        n_positive = sum(1 for m in report.pair_metrics if m.f1_lift > 0.01)
        n_negative = sum(1 for m in report.pair_metrics if m.f1_lift < -0.01)
        pair_3_3 = next(m for m in report.pair_metrics if m.pair_name == "pair_3_3")
        pair_1_3 = next(m for m in report.pair_metrics if m.pair_name == "pair_1_3")
        # Median lift on non-degen
        non_degen = [m for m in report.pair_metrics
                     if (m.n_positive_test / m.n_test if m.n_test else 0) >= 0.05
                     and (m.n_positive_test / m.n_test if m.n_test else 0) <= 0.95]
        median_lift = (sorted([m.f1_lift for m in non_degen])[len(non_degen) // 2]
                        if non_degen else 0.0)

        summary_rows.append({
            "label": label,
            "n_features": len(feat_names),
            "median_lift": median_lift,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_degenerate": report.n_degenerate_pairs,
            "pair_3_3_lift": pair_3_3.f1_lift,
            "pair_3_3_f1": pair_3_3.f1,
            "pair_3_3_recall": pair_3_3.recall,
            "pair_3_3_precision": pair_3_3.precision,
            "pair_1_3_lift": pair_1_3.f1_lift,
        })

    print("\n\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(f"{'label':32s} {'#feat':>6s} {'median_lift':>11s} "
          f"{'+lift':>5s} {'-lift':>5s} {'degen':>5s} | "
          f"{'p33_lift':>8s} {'p33_f1':>7s} {'p33_rec':>8s} {'p33_prec':>8s} | "
          f"{'p13_lift':>8s}")
    print("-" * 120)
    for r in summary_rows:
        print(f"{r['label']:32s} {r['n_features']:6d} {r['median_lift']:11.4f} "
              f"{r['n_positive']:5d} {r['n_negative']:5d} {r['n_degenerate']:5d} | "
              f"{r['pair_3_3_lift']:+8.4f} {r['pair_3_3_f1']:7.4f} "
              f"{r['pair_3_3_recall']:8.4f} {r['pair_3_3_precision']:8.4f} | "
              f"{r['pair_1_3_lift']:+8.4f}")
    print()
    print("Headline reference (5-feature baseline from CONTEXT.md): "
          "median_lift=+0.086, pair_3_3=+0.36")
    print("Headline current (21-feature, train CLI run):           "
          "median_lift=+0.032, pair_3_3=-0.192")


if __name__ == "__main__":
    main()
