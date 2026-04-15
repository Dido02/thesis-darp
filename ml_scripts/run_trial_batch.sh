#!/bin/bash
# Run trial logging across multiple instances and seeds.
# Usage: ./run_trial_batch.sh [max_parallel=4]
#
# Logic: bigger instances = slower iters = fewer iters per epoch = trial more often.

set -e

SOLVER="$(dirname "$0")/../cpp/build/darp_solver"
DATA_DIR="$(dirname "$0")/../data/fiedler_converted"
LOG_DIR="$(dirname "$0")/../trail_logs"
MAX_PARALLEL=${1:-4}

mkdir -p "$LOG_DIR"

# ── Instance configs ──
# Format: instance_file  epoch_budget_ms  m_trials_every  seeds
#
# Instance size vs iterations per 5s epoch (approximate):
#   30s   (124 veh,  280 req)  -> ~2500 iters/epoch  -> trial every 100
#   01min (200 veh,  572 req)  -> ~1000 iters/epoch  -> trial every 50
#   02min (352 veh, 1210 req)  -> ~300 iters/epoch   -> trial every 20
#   05min (781 veh, 3316 req)  -> ~50 iters/epoch    -> trial every 5
#   15min (1672 veh, 10226 req)-> ~300 iters/epoch   -> trial every 2
#   30min (1993 veh, 20724 req)-> ~300 iters/epoch   -> trial every 5
#
# NOTE: 30min instances need ~15GB RAM each. Don't run more than 1 in parallel.
CONFIGS=(
    # 30s instances (124 veh, 280 req) - fast iters, trial less often
    "manhattan_30_s_delay03_min.txt    5000   100  42,123,456"
    "manhattan_30_s_delay05_min.txt    5000   100  42,123,456"
    "manhattan_30_s_delay10_min.txt    5000   100  42,123,456"

    # 01_min instances (200 veh, 572 req)
    "manhattan_01_min_delay03_min.txt  5000   50   42,123,456"
    "manhattan_01_min_delay05_min.txt  5000   50   42,123,456"
    "manhattan_01_min_delay10_min.txt  5000   50   42,123,456"

    # 02_min instances (352 veh, 1210 req)
    "manhattan_02_min_delay03_min.txt  5000   20   42,123,456"
    "manhattan_02_min_delay05_min.txt  5000   20   42,123,456"
    "manhattan_02_min_delay10_min.txt  5000   20   42,123,456"

    # 05_min instances (781 veh, 3316 req) - slow iters, trial often
    "manhattan_05_min_delay03_min.txt  5000   5    42,123"
    "manhattan_05_min_delay05_min.txt  5000   5    42,123"
    "manhattan_05_min_delay10_min.txt  5000   5    42,123"

    # 15min instances (1672 veh, 10226 req) - very slow, trial every 2
    "manhattan_15min_03min.txt         5000   2    42"
    "manhattan_15min_05min.txt         5000   2    42"
    "manhattan_15min_10min.txt         5000   2    42"

)

# Heavy instances that need to run sequentially (one at a time) due to ~15GB RAM each
# Uncomment to enable:
HEAVY_CONFIGS=(
    # "manhattan_30_min_delay03_min.txt  5000   5    42"
    # "manhattan_30_min_delay05_min.txt  5000   5    42"
    # "manhattan_30_min_delay10_min.txt  5000   5    42"
)

# Track running jobs
RUNNING=0
PIDS=()

wait_for_slot() {
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" 2>/dev/null || true
                unset 'PIDS[$i]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        PIDS=("${PIDS[@]}")
        if [ $RUNNING -ge $MAX_PARALLEL ]; then
            sleep 1
        fi
    done
}

TOTAL=0
for config in "${CONFIGS[@]}"; do
    read -r instance budget every seeds <<< "$config"
    IFS=',' read -ra SEED_ARR <<< "$seeds"
    for seed in "${SEED_ARR[@]}"; do
        TOTAL=$((TOTAL + 1))
    done
done

echo "=== Trial Batch Runner ==="
echo "Instances: ${#CONFIGS[@]} configs, $TOTAL total runs"
echo "Max parallel: $MAX_PARALLEL"
echo "Log dir: $LOG_DIR"
echo ""

COUNT=0
for config in "${CONFIGS[@]}"; do
    read -r instance budget every seeds <<< "$config"
    instance_path="$DATA_DIR/$instance"

    if [ ! -f "$instance_path" ]; then
        echo "SKIP: $instance (file not found)"
        continue
    fi

    IFS=',' read -ra SEED_ARR <<< "$seeds"
    for seed in "${SEED_ARR[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] $instance seed=$seed budget=${budget}ms every=$every"

        wait_for_slot

        "$SOLVER" --dynamic \
            --epoch-length 30 \
            --epoch-budget-ms "$budget" \
            --instance "$instance_path" \
            --seed "$seed" \
            --penalty 10000 \
            --log-trials \
            --m-trials 50 --m-trials-every "$every" \
            --log-dir "$LOG_DIR" \
            > /dev/null 2>&1 &

        PIDS+=($!)
        RUNNING=$((RUNNING + 1))
    done
done

echo ""
echo "Waiting for $RUNNING remaining parallel jobs..."
wait
echo "Parallel batch done ($COUNT runs)."

# Run heavy instances sequentially (too much RAM for parallel)
if [ ${#HEAVY_CONFIGS[@]} -gt 0 ]; then
    echo ""
    echo "=== Heavy instances (sequential, ~15GB RAM each) ==="
    for config in "${HEAVY_CONFIGS[@]}"; do
        read -r instance budget every seeds <<< "$config"
        instance_path="$DATA_DIR/$instance"

        if [ ! -f "$instance_path" ]; then
            echo "SKIP: $instance (file not found)"
            continue
        fi

        IFS=',' read -ra SEED_ARR <<< "$seeds"
        for seed in "${SEED_ARR[@]}"; do
            COUNT=$((COUNT + 1))
            echo "[$COUNT] $instance seed=$seed budget=${budget}ms every=$every (sequential)"

            "$SOLVER" --dynamic \
                --epoch-length 30 \
                --epoch-budget-ms "$budget" \
                --instance "$instance_path" \
                --seed "$seed" \
                --penalty 10000 \
                --log-trials \
                --m-trials 50 --m-trials-every "$every" \
                --log-dir "$LOG_DIR" \
                > /dev/null 2>&1

            echo "  ✓ done"
        done
    done
fi

echo ""
echo "All runs complete."
echo ""

echo "=== Generated logs ==="
ls -lh "$LOG_DIR"/*.jsonl 2>/dev/null | awk '{print $5, $9}'
