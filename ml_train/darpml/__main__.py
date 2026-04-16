"""CLI entry points for the darpml training pipeline.

Usage:
    python -m darpml validate --logs ../trail_logs/ --data ../data/
    python -m darpml extract-features --logs ../trail_logs/ --data ../data/ --out cache/features.parquet
    python -m darpml compute-labels --logs ../trail_logs/ --out cache/labels.parquet
    python -m darpml train --features cache/features.parquet --labels cache/labels.parquet --out models/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def cmd_validate(args):
    """Validate objective reconstruction for all logs."""
    from .instance import InstanceTooLargeError
    from .reconstruction import validate_log

    logs_dir = Path(args.logs)
    log_files = sorted(logs_dir.glob("*.jsonl"))
    if not log_files:
        print(f"No .jsonl files found in {logs_dir}")
        sys.exit(1)

    print(f"Validating {len(log_files)} log files against {args.data}")
    total_passed, total_failed, total_skipped = 0, 0, 0

    for lf in log_files:
        try:
            report = validate_log(lf, args.data, tolerance=args.tolerance)
            status = "PASS" if report.ok else "FAIL"
            print(f"  {status} {lf.name}: {report.passed}/{report.total_states} "
                  f"(max_err={report.max_abs_error:.4f})")
            total_passed += report.passed
            total_failed += report.failed
            if report.mismatches:
                for m in report.mismatches[:3]:
                    print(f"       iter={m['iter']}: logged={m['logged']:.4f} "
                          f"recomputed={m['recomputed']:.4f} err={m['error']:.4f}")
        except InstanceTooLargeError as e:
            print(f"  SKIP {lf.name}: {e}")
            total_skipped += 1
        except FileNotFoundError as e:
            print(f"  SKIP {lf.name}: {e}")
            total_skipped += 1

    print(f"\nTotal: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    if total_failed > 0:
        sys.exit(1)


def cmd_extract_features(args):
    """Extract features from all logs and save to parquet."""
    from .features import extract_all, get_feature_names
    from .instance import InstanceTooLargeError, load_instance
    from .loader import load_log, resolve_instance_path

    logs_dir = Path(args.logs)
    log_files = sorted(logs_dir.glob("*.jsonl"))
    if not log_files:
        print(f"No .jsonl files found in {logs_dir}")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features from {len(log_files)} logs")
    print(f"Features: {get_feature_names()}")

    all_rows = []
    for lf in log_files:
        try:
            header, state_trials = load_log(lf)
            instance_path = resolve_instance_path(header.instance_path, args.data)
            instance = load_instance(instance_path)
        except (InstanceTooLargeError, FileNotFoundError) as e:
            print(f"  SKIP {lf.name}: {e}")
            continue

        n_extracted = 0
        for st in state_trials:
            feats = extract_all(st.state, instance, header)
            feats["log_file"] = lf.name
            feats["iter"] = st.state.iter
            all_rows.append(feats)
            n_extracted += 1

        print(f"  {lf.name}: {n_extracted} states")

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


def cmd_compute_labels(args):
    """Compute labels from trial data and save to parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .labels import compute_labels_for_log
    from .loader import load_log

    logs_dir = Path(args.logs)
    log_files = sorted(logs_dir.glob("*.jsonl"))
    if not log_files:
        print(f"No .jsonl files found in {logs_dir}")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filter_epoch_1 = not args.keep_epoch_1
    print(f"Computing labels from {len(log_files)} logs")
    print(f"  Rule: {args.rule}, threshold: {args.threshold}, filter_epoch_1: {filter_epoch_1}")

    all_rows = []
    total_epoch1 = 0
    total_uninformative = 0

    for lf in log_files:
        try:
            header, state_trials = load_log(lf)
        except Exception as e:
            print(f"  SKIP {lf.name}: {e}")
            continue

        n_d = len(header.destroy_ops)
        n_r = len(header.repair_ops)

        rows, stats = compute_labels_for_log(
            state_trials, n_d, n_r,
            rule=args.rule,
            threshold=args.threshold,
            filter_epoch_1=filter_epoch_1,
        )
        for row in rows:
            row["log_file"] = lf.name
        all_rows.extend(rows)

        total_epoch1 += stats["n_epoch1_filtered"]
        total_uninformative += stats["n_uninformative_filtered"]

        print(f"  {lf.name}: {stats['n_labeled']} labeled "
              f"(epoch1={stats['n_epoch1_filtered']}, "
              f"uninformative={stats['n_uninformative_filtered']})")

    df = pd.DataFrame(all_rows)

    # Save with metadata
    table = pa.Table.from_pandas(df)
    meta = {
        b"rule": args.rule.encode(),
        b"threshold": str(args.threshold).encode(),
        b"filter_epoch_1": str(filter_epoch_1).encode(),
    }
    existing_meta = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_meta, **meta})
    pq.write_table(table, str(out_path))

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Filtered: {total_epoch1} epoch-1, {total_uninformative} uninformative")


def cmd_train(args):
    """Train 20 RF classifiers."""
    from .training import train_pipeline

    report = train_pipeline(
        features_path=args.features,
        labels_path=args.labels,
        output_dir=args.out,
        n_estimators=args.n_estimators,
    )

    print(f"\n{'='*60}")
    print(f"Training complete: {report.n_pairs} classifiers, "
          f"{report.n_samples} samples, {report.n_features} features")
    print(f"Per-log split: {report.n_train_logs} train logs ({report.n_train_rows} rows), "
          f"{report.n_test_logs} test logs ({report.n_test_rows} rows)")

    print(f"\nNon-degenerate pairs: {report.n_pairs - report.n_degenerate_pairs}/{report.n_pairs} "
          f"({report.n_degenerate_pairs} degenerate excluded)")
    print(f"Median F1 lift (non-degenerate): {report.median_f1_lift:+.3f}")
    print(f"  Positive lift (>0.01): {report.n_positive_lift},  "
          f"Zero (|lift|<=0.01): {report.n_zero_lift},  "
          f"Negative (<-0.01): {report.n_negative_lift}")

    print(f"\nFeature importances (averaged across {report.n_pairs} forests):")
    for name, imp in sorted(report.feature_importances.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}: {imp:.4f}")

    print(f"\nPer-pair metrics:")
    print(f"  {'Pair':20s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} "
          f"{'F1':>6s} {'A1_F1':>6s} {'Lift':>7s} {'Train+':>10s}")
    for m in sorted(report.pair_metrics, key=lambda x: -x.f1_lift):
        print(f"  {m.pair_name:20s} {m.accuracy:6.3f} {m.precision:6.3f} "
              f"{m.recall:6.3f} {m.specificity:6.3f} {m.f1:6.3f} "
              f"{m.always1_f1:6.3f} {m.f1_lift:+7.3f} "
              f"{m.n_positive_train:5d}/{m.n_train}")

    print(f"\nCompared to previous pipeline (SA-acceptance rule, per-row split):")
    print(f"  Old: median F1=0.993 (per-row), always-1 baseline F1=0.980 -> inflated")
    new_median_f1 = float(
        __import__("numpy").median([m.f1 for m in report.pair_metrics])
    )
    print(f"  New: median F1={new_median_f1:.3f} (per-log), "
          f"non-degenerate median lift={report.median_f1_lift:+.3f}")


def main():
    parser = argparse.ArgumentParser(prog="darpml", description="ML-guided ALNS operator selection")
    subparsers = parser.add_subparsers(dest="command")

    # validate
    p = subparsers.add_parser("validate", help="Validate objective reconstruction")
    p.add_argument("--logs", required=True, help="Directory of JSONL logs")
    p.add_argument("--data", required=True, help="Data root directory")
    p.add_argument("--tolerance", type=float, default=0.05)

    # extract-features
    p = subparsers.add_parser("extract-features", help="Extract features to parquet")
    p.add_argument("--logs", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)

    # compute-labels
    p = subparsers.add_parser("compute-labels", help="Compute labels to parquet")
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rule", choices=["improvement_freq", "sa_acceptance"],
                   default="improvement_freq",
                   help="Labeling rule (default: improvement_freq)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Threshold for labeling rule (default: 0.3)")
    p.add_argument("--keep-epoch-1", action="store_true",
                   help="Include epoch 1 states (filtered by default)")

    # train
    p = subparsers.add_parser("train", help="Train 20 RF classifiers")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "validate": cmd_validate,
        "extract-features": cmd_extract_features,
        "compute-labels": cmd_compute_labels,
        "train": cmd_train,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
