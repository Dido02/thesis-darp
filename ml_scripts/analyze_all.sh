#!/bin/bash
# Run all analysis scripts on every log in trail_logs/.
# Usage: ./analyze_all.sh [log_dir=../trail_logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${1:-$SCRIPT_DIR/../trail_logs}"

echo "=== Batch Analysis ==="
echo "Log dir: $LOG_DIR"
echo ""

LOGS=("$LOG_DIR"/*.jsonl)
if [ ${#LOGS[@]} -eq 0 ]; then
    echo "No .jsonl files found in $LOG_DIR"
    exit 1
fi

echo "Found ${#LOGS[@]} log files"
echo ""

for log in "${LOGS[@]}"; do
    name=$(basename "$log")
    echo "━━━ $name ━━━"

    echo "  → summary_stats..."
    python3 "$SCRIPT_DIR/summary_stats.py" "$log" > /dev/null 2>&1

    echo "  → plot_operator_performance..."
    python3 "$SCRIPT_DIR/plot_operator_performance.py" "$log" --no-show 2>&1 | tail -1 || echo "    (skipped)"

    echo "  → plot_trial_distributions..."
    python3 "$SCRIPT_DIR/plot_trial_distributions.py" "$log" --no-show 2>&1 | tail -1 || echo "    (skipped)"

    echo "  → extract_features..."
    python3 "$SCRIPT_DIR/extract_features.py" "$log" > /dev/null 2>&1

    echo "  ✓ done"
    echo ""
done

echo "=== Results ==="
echo "Plots:    $(ls "$LOG_DIR"/plots/*.png 2>/dev/null | wc -l | tr -d ' ') files in $LOG_DIR/plots/"
echo "Summaries: $(ls "$LOG_DIR"/summary/*.txt 2>/dev/null | wc -l | tr -d ' ') files in $LOG_DIR/summary/"
echo "Features: $(ls "$LOG_DIR"/summary/*.csv 2>/dev/null | wc -l | tr -d ' ') files in $LOG_DIR/summary/"
