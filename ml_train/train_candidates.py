"""Train the three benchmark candidate models on cache_v2_dynamic
(20-feature parquet, after dropping flexibility_horizon). Each candidate
goes in its own directory so models don't clash.

  models_v2_dyn_5f/   <-  core only
  models_v2_dyn_13f/  <-  core + tightness + worst_support
  models_v2_dyn_20f/  <-  all 5 batches

Bypasses the train CLI's sidecar check by calling train_pipeline directly
with a restricted feature_names list (the cached parquet's sidecar declares
all 5 batches; subset requests via the CLI would be rejected, which is the
correct behavior in production but in the way for ablation training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from darpml.features import _REGISTRY
from darpml.training import train_pipeline

FEATURES_PATH = "cache_v3_full/features.parquet"
DEFAULT_LABELS_PATH = "cache_v3_full/labels.parquet"


def names_for(batches: list[str], mode: str = "dynamic") -> list[str]:
    bset = set(batches)
    out = []
    for name, entry in _REGISTRY.items():
        if entry.batch not in bset:
            continue
        if entry.mode == "both" or entry.mode == mode:
            out.append(name)
    return out


ALL_BATCHES = ["core", "tightness", "worst_support", "lns_state", "rolling_horizon"]

# (label, batches, base_dir, model_kind, regressor_only_for_suffix, target_metric, log_transform_label)
# regressor_only_for_suffix: if set, the entry is trained ONLY when --out-suffix
# matches it. Lets us co-locate the regressor in train_candidates.py without
# re-training it for every threshold sweep run (its labels don't depend on threshold).
# target_metric: only meaningful for regressor entries. "improve_freq" = legacy
# frequency-only target; "ei_pct" = expected improvement % (magnitude-aware).
# log_transform_label: only meaningful when target_metric == "ei_pct". If True,
# labels are log-transformed and min-max scaled per pair (fit on TRAIN ONLY)
# before regressor training. The log_min/log_max are persisted to meta.json.
CONFIGS = [
    ("5f",         ["core"],                              "models_v3_full_5f",         "classifier", None,  "improve_freq", False),
    ("13f",        ["core", "tightness", "worst_support"], "models_v3_full_13f",        "classifier", None,  "improve_freq", False),
    ("20f",        ALL_BATCHES,                            "models_v3_full_20f",        "classifier", None,  "improve_freq", False),
    ("5f_regr",    ["core"],                              "models_v3_full_5f_regr",    "regressor",  "_t03", "improve_freq", False),
    ("13f_regr",   ["core", "tightness", "worst_support"], "models_v3_full_13f_regr",   "regressor",  "_t03", "improve_freq", False),
    ("20f_regr",   ALL_BATCHES,                            "models_v3_full_20f_regr",   "regressor",  "_t03", "improve_freq", False),
    ("20f_ei",     ALL_BATCHES,                            "models_v3_full_20f_ei",     "regressor",  "_t03", "ei_pct",       False),
    ("20f_ei_log", ALL_BATCHES,                            "models_v3_full_20f_ei_log", "regressor",  "_t03", "ei_pct",       True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS_PATH,
                    help="Path to labels parquet (default cache_v3_full/labels.parquet)")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to each output dir (e.g. _t03 for threshold 0.3)")
    ap.add_argument("--only", default=None,
                    help="Only train CONFIGS entries whose label matches this exact value "
                         "(e.g. --only 20f_ei). Default: train every entry not gated out.")
    ap.add_argument("--model-type", choices=("forest", "tree"), default="forest",
                    help="forest (default) = RandomForest with n_estimators=100. "
                         "tree = single DecisionTree (no ensembling). "
                         "Tree-mode dirs: 'v3' is rewritten to 'v6' and '_tree' is "
                         "appended to the dir name so artifacts don't collide.")
    ap.add_argument("--tree-max-depth", type=int, default=None,
                    help="Max depth for DecisionTree (only when --model-type tree). "
                         "Default: None = unrestricted, most direct vs-forest comparison.")
    args = ap.parse_args()

    summary = []
    for label, batches, base_dir, model_kind, gate_suffix, target_metric, log_transform_label in CONFIGS:
        if args.only is not None and label != args.only:
            continue
        if gate_suffix is not None and args.out_suffix != gate_suffix:
            print(f"\n[skip] {label}: gated to --out-suffix={gate_suffix!r} (current: {args.out_suffix!r})")
            continue

        # Regressor output dir is fixed (no suffix) since labels are
        # threshold-independent; classifier dirs get the suffix.
        out_dir = base_dir if model_kind == "regressor" else base_dir + args.out_suffix
        # Tree-mode artifacts: rename v3 -> v6 and append _tree so they
        # don't collide with the existing forest-mode dirs.
        if args.model_type == "tree":
            out_dir = out_dir.replace("v3", "v6") + "_tree"
        feat_names = names_for(batches)
        log_tag = ", log_transform=True" if log_transform_label else ""
        type_tag = f", model_type={args.model_type}"
        if args.model_type == "tree" and args.tree_max_depth is not None:
            type_tag += f", max_depth={args.tree_max_depth}"
        print(f"\n{'#' * 80}")
        print(f"# {label} [{model_kind}, target={target_metric}{log_tag}{type_tag}]: "
              f"{len(feat_names)} features  ->  {out_dir}")
        print(f"# batches: {batches}")
        print(f"{'#' * 80}\n")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report = train_pipeline(
            features_path=FEATURES_PATH,
            labels_path=args.labels,
            output_dir=out_dir,
            n_estimators=100,
            feature_names=feat_names,
            mode="dynamic",
            batches=batches,
            model_kind=model_kind,
            target_metric=target_metric,
            log_transform_label=log_transform_label,
            model_type=args.model_type,
            tree_max_depth=args.tree_max_depth,
        )
        if model_kind == "regressor":
            non_degen = [m for m in report.pair_metrics if m.label_std_train >= 1e-3]
            sorted_lifts = sorted(m.mae_lift for m in non_degen)
            median_lift = sorted_lifts[len(sorted_lifts) // 2] if sorted_lifts else 0.0
            n_pos = sum(1 for m in report.pair_metrics if m.mae_lift > 0.001)
            n_neg = sum(1 for m in report.pair_metrics if m.mae_lift < -0.001)
        else:
            non_degen = [m for m in report.pair_metrics
                         if 0.05 <= (m.n_positive_test / m.n_test if m.n_test else 0) <= 0.95]
            sorted_lifts = sorted(m.f1_lift for m in non_degen)
            median_lift = sorted_lifts[len(sorted_lifts) // 2] if sorted_lifts else 0.0
            n_pos = sum(1 for m in report.pair_metrics if m.f1_lift > 0.01)
            n_neg = sum(1 for m in report.pair_metrics if m.f1_lift < -0.01)
        summary.append((label, len(feat_names), median_lift,
                        report.n_pairs - report.n_degenerate_pairs, n_pos, n_neg))

    print("\n" + "=" * 80)
    print("CANDIDATE SUMMARY (median_lift = F1-lift for classifier, MAE-lift for regressor)")
    print("=" * 80)
    print(f"{'label':10s} {'#feat':>6s} {'median_lift':>11s} "
          f"{'non_degen':>10s} {'+lift':>6s} {'-lift':>6s}")
    for label, nf, ml, nd, npos, nneg in summary:
        print(f"{label:10s} {nf:6d} {ml:11.4f} {nd:10d} {npos:6d} {nneg:6d}")


if __name__ == "__main__":
    main()
