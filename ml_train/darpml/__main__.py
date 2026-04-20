"""CLI entry points for the darpml training pipeline.

Usage:
    python -m darpml validate --logs ../trail_logs/ --data ../data/
    python -m darpml extract-features --logs ../trail_logs/ --data ../data/ \
        --out cache/features.parquet --mode dynamic
    python -m darpml compute-labels --logs ../trail_logs/ \
        --out cache/labels.parquet --mode dynamic
    python -m darpml train --features cache/features.parquet \
        --labels cache/labels.parquet --out models/ --mode dynamic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _parse_batches(s: str | None) -> list[str] | None:
    """Comma-separated string -> list. None or empty stays None (= all batches)."""
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _validate_mode_batches(mode: str, batches: list[str] | None) -> None:
    """CLI-level guard: rolling_horizon batch is dynamic-only."""
    if mode == "static" and batches and "rolling_horizon" in batches:
        sys.exit("ERROR: --mode static is incompatible with --batches "
                 "rolling_horizon (rolling_horizon features are dynamic-only).")


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

    batches = _parse_batches(args.batches)
    _validate_mode_batches(args.mode, batches)
    feature_names = get_feature_names(batches=batches, mode=args.mode)
    if not feature_names:
        sys.exit(f"ERROR: no features matched batches={batches} mode={args.mode}")

    logs_dir = Path(args.logs)
    log_files = sorted(logs_dir.glob("*.jsonl"))
    if not log_files:
        print(f"No .jsonl files found in {logs_dir}")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features from {len(log_files)} logs")
    print(f"  Mode: {args.mode}, Batches: {batches if batches else 'all'}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

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
            feats = extract_all(st.state, instance, header,
                                batches=batches, mode=args.mode)
            feats["log_file"] = lf.name
            feats["iter"] = st.state.iter
            all_rows.append(feats)
            n_extracted += 1

        print(f"  {lf.name}: {n_extracted} states")

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)

    # Sidecar manifest so train/export_onnx can verify mode + batches without
    # re-reading the registry. Lives alongside the parquet.
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(sidecar, "w") as f:
        json.dump({
            "mode": args.mode,
            "batches": batches if batches else [],
            "feature_names": feature_names,
        }, f, indent=2)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Sidecar metadata: {sidecar}")


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
    print(f"  Mode: {args.mode}, Rule: {args.rule}, threshold: {args.threshold}, "
          f"filter_epoch_1: {filter_epoch_1}")

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
        b"mode": args.mode.encode(),
    }
    existing_meta = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_meta, **meta})
    pq.write_table(table, str(out_path))

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Filtered: {total_epoch1} epoch-1, {total_uninformative} uninformative")


def cmd_train(args):
    """Train 20 RF classifiers."""
    from .features import get_feature_names
    from .training import train_pipeline

    batches = _parse_batches(args.batches)
    _validate_mode_batches(args.mode, batches)

    # 1. Verify the requested batches/mode produce a non-empty feature set in the registry.
    expected_names = get_feature_names(batches=batches, mode=args.mode)
    if not expected_names:
        sys.exit(f"ERROR: no features matched batches={batches} mode={args.mode}")

    # 2. If the features parquet has a sidecar metadata file, verify it agrees
    #    with the requested mode/batches. Otherwise just check that all expected
    #    feature columns are present in the parquet.
    features_path = Path(args.features)
    sidecar = features_path.with_suffix(features_path.suffix + ".meta.json")
    if sidecar.exists():
        with open(sidecar) as f:
            sm = json.load(f)
        sm_mode = sm.get("mode")
        sm_batches = sm.get("batches") or None
        if sm_mode != args.mode or sm_batches != (batches if batches else []):
            sys.exit(
                f"ERROR: features parquet was extracted with mode={sm_mode} "
                f"batches={sm_batches}, but train was called with mode={args.mode} "
                f"batches={batches if batches else []}. Re-run extract-features."
            )
    else:
        df = pd.read_parquet(features_path, columns=None)
        actual_cols = set(df.columns) - {"log_file", "iter"}
        missing = [n for n in expected_names if n not in actual_cols]
        if missing:
            sys.exit(
                f"ERROR: features parquet missing columns required by "
                f"mode={args.mode} batches={batches if batches else 'all'}: {missing}. "
                f"Re-run extract-features with the same flags."
            )

    report = train_pipeline(
        features_path=args.features,
        labels_path=args.labels,
        output_dir=args.out,
        n_estimators=args.n_estimators,
        feature_names=expected_names,
        mode=args.mode,
        batches=batches if batches else [],
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
    p.add_argument("--mode", required=True, choices=["static", "dynamic"],
                   help="Solver mode (filters which features are extracted)")
    p.add_argument("--batches", default=None,
                   help="Comma-separated batch names (default: all registered batches)")

    # compute-labels
    p = subparsers.add_parser("compute-labels", help="Compute labels to parquet")
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True, choices=["static", "dynamic"],
                   help="Solver mode (recorded in parquet metadata)")
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
    p.add_argument("--mode", required=True, choices=["static", "dynamic"],
                   help="Solver mode (must match the features parquet)")
    p.add_argument("--batches", default=None,
                   help="Comma-separated batch names (default: all registered batches)")

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
