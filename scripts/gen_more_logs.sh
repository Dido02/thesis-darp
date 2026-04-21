#!/bin/bash
# Generate additional dynamic ALNS training logs.
#
# Resumable: skips logs that already exist in trail_logs/. Safe to interrupt
# (Ctrl-C) and re-run; will pick up where it left off.
#
# Sequential by design — multiple 05_min runs in parallel would compete for
# RAM (~5 GB each). Total wall-clock estimate: ~4-5 hours for the full sweep.
#
# Run from project root:
#   ./scripts/gen_more_logs.sh                       # default out dir: trail_logs_v2/
#   ./scripts/gen_more_logs.sh --dry-run             # list what would run
#   ./scripts/gen_more_logs.sh --tier 30_s           # restrict to one tier
#   ./scripts/gen_more_logs.sh --log-dir trail_logs/ # write into existing dir
#
# Default writes to trail_logs_v2/ to keep the new logs separate from the
# original trail_logs/. Combine for training by either symlinking or copying:
#   ln -s "$(pwd)"/trail_logs_v2/*.jsonl trail_logs_combined/
# or just point extract-features at the v2 dir alone for ablation training.
#
# Match the naming + trial-cadence conventions of existing logs in trail_logs/
# so darpml/loader.py picks them up identically:
#   <instance>_s<SEED>_t<MTRIALS>e<EVERY>_dynamic.jsonl
#   m-trials-every: 50 for 30s/01min, 20 for 02min, 5 for 05min (existing convention).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER="$REPO_ROOT/cpp/build/darp_solver"
DATA="$REPO_ROOT/data/fiedler_converted"
LOGDIR="$REPO_ROOT/trail_logs_v2"   # default; override with --log-dir
TMPLOG="/tmp/gen_more_logs_$$.txt"

# Per-tier m-trials-every value (matches existing log convention).
# Kept as a function because bash associative arrays choke on numeric-leading keys.
every_for_tier() {
    case "$1" in
        30_s)   echo 100 ;;  # matches existing trail_logs convention; e=50 doubled overhead for no gain
        01_min) echo 50 ;;
        02_min) echo 20 ;;
        05_min) echo 10 ;;
        15min)  echo 10 ;;   # 15min runs few iters/epoch on its 5113 reqs; small e to capture variation
        *) echo "ERROR: unknown tier $1" >&2; return 1 ;;
    esac
}

# Instance request-release window in SECONDS (used to compute simulation end
# time). Calibrated against existing trail_logs/ epoch counts:
#   sim_window_sec = delay_sec + instance_window_sec
#   epochs ≈ sim_window_sec / 30  (epoch_length = 30s)
# Real epoch count varies ±1-2 with seed because MIN_REQUESTS_FOR_EPOCH gating
# can skip an early epoch and rejection timing can end the simulation early.
instance_window_sec_for() {
    case "$1" in
        30_s)   echo 30 ;;
        01_min) echo 60 ;;
        02_min) echo 120 ;;
        05_min) echo 300 ;;
        15min)  echo 900 ;;
        *)      echo 60 ;;
    esac
}

# Estimated wall-clock seconds per epoch using user's formula:
# slowdown_factor = 20 * M_TRIALS / E ; per-epoch wall time = epoch_budget * slowdown
# These are upper bounds — observed rate is ~60% of formula in early runs.
M_TRIALS=50
EPOCH_BUDGET_S=5
estimated_epoch_seconds() {
    local every=$1
    echo $(( EPOCH_BUDGET_S * 20 * M_TRIALS / every ))
}

# Extract delay in minutes from the delay string portion of a filename.
#   "delay03_min" -> 3   (smaller tiers)
#   "03min"       -> 3   (15min tier)
delay_int_from() {
    local d=$1
    # Try smaller-tier pattern first
    local n
    n=$(echo "$d" | sed -nE 's/^delay0*([0-9]+)_min$/\1/p')
    if [ -z "$n" ]; then
        n=$(echo "$d" | sed -nE 's/^0*([0-9]+)min$/\1/p')
    fi
    echo "${n:-3}"
}

# Existing seeds in trail_logs/ are 42, 123, 456. New seeds add training data.
NEW_SEEDS=(789 1024 2025)

# 05_min × delay10 has NO existing logs — fill all six seeds (originals + new).
GAP_05MIN_DELAY10_SEEDS=(42 123 456 789 1024 2025)

# 15min has NO existing logs — start with the canonical 3-seed set so it stays
# proportional to the additional load on the other tiers (3 new seeds each).
GAP_15MIN_SEEDS=(42 123 456)

# Naming conventions differ between Manhattan tiers in data/fiedler_converted/:
#   30_s/01_min/02_min/05_min  ->  manhattan_<tier>_<delay>.txt    where delay = "delay03_min" etc.
#   15min                      ->  manhattan_<tier>_<delay>.txt    where delay = "03min" etc. (no "delay" prefix)
delays_for_tier() {
    case "$1" in
        15min) echo "03min 05min 10min" ;;
        *)     echo "delay03_min delay05_min delay10_min" ;;
    esac
}

# Order matters — script processes tiers in this order, smallest first so quick
# feedback comes early. 15min is largest; default behavior runs it last.
TIERS=(30_s 01_min 02_min 05_min 15min)

# Parse args
DRY_RUN=0
RESTRICT_TIER=""
WORKERS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --tier) RESTRICT_TIER="$2"; shift 2 ;;
        --log-dir) LOGDIR="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
    echo "ERROR: --workers must be a positive integer"
    exit 1
fi

# Sanity checks
if [ ! -x "$SOLVER" ]; then
    echo "ERROR: solver not found at $SOLVER"
    echo "       run: cd cpp/build && make -j8"
    exit 1
fi
mkdir -p "$LOGDIR"

# Build the run plan (instance, seed, every) tuples.
plan=()
for tier in "${TIERS[@]}"; do
    if [ -n "$RESTRICT_TIER" ] && [ "$tier" != "$RESTRICT_TIER" ]; then continue; fi
    every=$(every_for_tier "$tier")
    # delays_for_tier returns space-separated; split into array
    read -ra delays_arr <<< "$(delays_for_tier "$tier")"
    for delay in "${delays_arr[@]}"; do
        inst="manhattan_${tier}_${delay}"
        inst_path="$DATA/${inst}.txt"
        if [ ! -f "$inst_path" ]; then continue; fi

        # Pick seed set:
        #   - 15min: canonical 3-seed set (no existing logs)
        #   - 05_min × delay10: gap-fill all 6 seeds (no existing logs)
        #   - everything else: 3 new seeds (additions to existing 42/123/456)
        if [ "$tier" = "15min" ]; then
            seeds=("${GAP_15MIN_SEEDS[@]}")
        elif [ "$tier" = "05_min" ] && [ "$delay" = "delay10_min" ]; then
            seeds=("${GAP_05MIN_DELAY10_SEEDS[@]}")
        else
            seeds=("${NEW_SEEDS[@]}")
        fi
        # Pre-compute estimates so the run loop doesn't have to.
        delay_min=$(delay_int_from "$delay")
        window_sec=$(instance_window_sec_for "$tier")
        total_eps=$(( (delay_min * 60 + window_sec) / 30 ))
        est_ep_s=$(estimated_epoch_seconds "$every")
        for seed in "${seeds[@]}"; do
            plan+=("$inst|$seed|$every|$total_eps|$est_ep_s")
        done
    done
done

# Show plan + dry-run exit
total=${#plan[@]}
existing=0
todo=0
total_est_seconds=0
for item in "${plan[@]}"; do
    IFS='|' read -r inst seed every total_eps est_ep_s <<< "$item"
    log_name="${inst}_s${seed}_t50e${every}_dynamic.jsonl"
    if [ -f "$LOGDIR/$log_name" ]; then
        existing=$((existing + 1))
        continue
    fi
    todo=$((todo + 1))
    # Sum formula-based wall-clock estimate over the to-do runs.
    total_est_seconds=$(( total_est_seconds + total_eps * est_ep_s ))
done

est_h=$(( total_est_seconds / 3600 ))
est_m=$(( (total_est_seconds % 3600) / 60 ))

echo "=== gen_more_logs.sh ==="
echo "Plan size:  $total tuples"
echo "Skip:       $existing (already present)"
echo "Will run:   $todo  (formula-est: ${est_h}h ${est_m}m wall clock; observed rate is ~60% of formula)"
echo "Workers:    $WORKERS  ($([ "$WORKERS" -eq 1 ] && echo "sequential, live in-place display" || echo "parallel pool, one-line-per-event display"))"
if [ "$WORKERS" -gt 1 ]; then
    echo "  Parallel-est: ~$((est_h * 60 + est_m / WORKERS))m wall clock (assumes perfect parallelism — actual will be longer due to seed/instance heterogeneity)"
fi
echo "Log dir:    $LOGDIR"
echo "Solver:     $SOLVER"
echo

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run — listing planned runs:"
    for item in "${plan[@]}"; do
        IFS='|' read -r inst seed every total_eps est_ep_s <<< "$item"
        log_name="${inst}_s${seed}_t50e${every}_dynamic.jsonl"
        marker="RUN "
        [ -f "$LOGDIR/$log_name" ] && marker="SKIP"
        printf "  %s %-60s  est: %d ep × %ds = %dm\n" \
            "$marker" "$log_name" "$total_eps" "$est_ep_s" "$((total_eps * est_ep_s / 60))"
    done
    exit 0
fi

# Live in-place status helpers. \033[K clears from cursor to end of line.
status_line() { printf "\r\033[K%s" "$1"; }
finalize_line() { printf "\r\033[K%s\n" "$1"; }

# Helper: count epoch_start records in a given JSONL log file.
count_epochs_in() {
    local path=$1
    local n=0
    if [ -f "$path" ]; then
        n=$(grep -c '"type":"epoch_start"' "$path" 2>/dev/null) || true
    fi
    echo "${n:-0}"
}

# Live in-place status helpers (sequential mode only).
status_line() { printf "\r\033[K%s" "$1"; }
finalize_line() { printf "\r\033[K%s\n" "$1"; }

# Execute
done_count=0
fail_count=0
launch_count=0
start_ts=$SECONDS

if [ "$WORKERS" -eq 1 ]; then
    # ------------------------------------------------------------------
    # Sequential mode — keep the live in-place display.
    # ------------------------------------------------------------------
    for item in "${plan[@]}"; do
        IFS='|' read -r inst seed every total_eps est_ep_s <<< "$item"
        log_name="${inst}_s${seed}_t50e${every}_dynamic.jsonl"
        log_path="$LOGDIR/$log_name"
        est_total_s=$(( total_eps * est_ep_s ))

        if [ -f "$log_path" ]; then
            continue
        fi

        done_count=$((done_count + 1))
        prefix=$(printf "[%2d/%d] %s" "$done_count" "$todo" "$log_name")

        run_start=$SECONDS
        status_line "$prefix  | launching ... (est ${total_eps} ep × ${est_ep_s}s = $((est_total_s/60))m)"

        : > "$TMPLOG"
        "$SOLVER" \
            --instance "$DATA/$inst.txt" \
            --seed "$seed" \
            --selection roulette \
            --penalty 10000 \
            --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
            --log-trials --m-trials 50 --m-trials-every "$every" \
            --log-dir "$LOGDIR" \
            > "$TMPLOG" 2>&1 &
        solver_pid=$!

        last_epoch_count=0
        last_epoch_change_t=$SECONDS

        while kill -0 "$solver_pid" 2>/dev/null; do
            epoch_n=$(count_epochs_in "$log_path")
            if [ "$epoch_n" -gt "$last_epoch_count" ]; then
                last_epoch_count=$epoch_n
                last_epoch_change_t=$SECONDS
            fi
            rt=$((SECONDS - run_start))
            in_ep_s=$((SECONDS - last_epoch_change_t))

            if [ "$est_ep_s" -gt 0 ]; then
                ep_pct=$(( 100 * in_ep_s / est_ep_s ))
                [ "$ep_pct" -gt 100 ] && ep_pct=100
            else
                ep_pct=0
            fi
            if [ "$total_eps" -gt 0 ]; then
                overall_pct=$(( (100 * epoch_n + ep_pct) / total_eps ))
                [ "$overall_pct" -gt 100 ] && overall_pct=100
            else
                overall_pct=0
            fi

            overall_elapsed=$((SECONDS - start_ts))
            spin_chars='|/-\'
            spin=${spin_chars:$((rt % 4)):1}

            if [ "$epoch_n" -gt 0 ]; then
                status_line "$(printf '%s  %s ep %d/%d [%ds/%ds %d%%]  overall %d%% (run %dm%ds, plan %dm%ds)' \
                    "$prefix" "$spin" "$epoch_n" "$total_eps" "$in_ep_s" "$est_ep_s" "$ep_pct" \
                    "$overall_pct" "$((rt/60))" "$((rt%60))" \
                    "$((overall_elapsed/60))" "$((overall_elapsed%60))")"
            else
                status_line "$(printf '%s  %s loading / pre-insert [%ds]  (est %d ep, run %dm%ds)' \
                    "$prefix" "$spin" "$rt" "$total_eps" "$((rt/60))" "$((rt%60))")"
            fi
            sleep 1
        done

        if wait "$solver_pid"; then exit_code=0; else exit_code=$?; fi
        rt=$((SECONDS - run_start))
        final_epochs=$(count_epochs_in "$log_path")

        if [ "$exit_code" -eq 0 ] && [ -f "$log_path" ]; then
            finalize_line "$prefix  done in $((rt/60))m $((rt%60))s ($final_epochs/$total_eps epochs)"
        else
            finalize_line "$prefix  FAILED (exit=$exit_code, $final_epochs/$total_eps epochs reached) - see $TMPLOG"
            fail_count=$((fail_count + 1))
        fi
    done
else
    # ------------------------------------------------------------------
    # Parallel mode — worker pool. No in-place display because multiple
    # solvers write concurrently. Print one launch line and one finish line
    # per run.
    # ------------------------------------------------------------------
    # Parallel arrays tracking running jobs.
    running_pids=()
    running_logs=()
    running_paths=()
    running_starts=()
    running_total_eps=()
    running_tmp=()

    # Wait for any running job to finish, prune it, print its finish line.
    wait_any_finish() {
        while true; do
            for i in "${!running_pids[@]}"; do
                local pid=${running_pids[$i]}
                if ! kill -0 "$pid" 2>/dev/null; then
                    local exit_code=0
                    wait "$pid" || exit_code=$?
                    local log_name=${running_logs[$i]}
                    local log_path=${running_paths[$i]}
                    local start_t=${running_starts[$i]}
                    local tot_eps=${running_total_eps[$i]}
                    local tmp_path=${running_tmp[$i]}
                    local rt=$((SECONDS - start_t))
                    local final_epochs
                    final_epochs=$(count_epochs_in "$log_path")
                    done_count=$((done_count + 1))

                    local plan_elapsed=$((SECONDS - start_ts))
                    local mins=$((plan_elapsed/60))
                    local secs=$((plan_elapsed%60))
                    local n_running=$(( ${#running_pids[@]} - 1 ))
                    local n_remaining=$(( todo - done_count - n_running ))

                    if [ "$exit_code" -eq 0 ] && [ -f "$log_path" ]; then
                        printf "[done %2d/%d in %dm%2ds (%d/%d ep)]  %s   [%d running, %d queued, plan %dm%ds]\n" \
                            "$done_count" "$todo" "$((rt/60))" "$((rt%60))" \
                            "$final_epochs" "$tot_eps" "$log_name" \
                            "$n_running" "$n_remaining" "$mins" "$secs"
                    else
                        printf "[FAIL %2d/%d in %dm%2ds]  %s   exit=%d, see %s\n" \
                            "$done_count" "$todo" "$((rt/60))" "$((rt%60))" \
                            "$log_name" "$exit_code" "$tmp_path"
                        fail_count=$((fail_count + 1))
                    fi

                    # Remove this index, repack arrays.
                    unset 'running_pids[i]'
                    unset 'running_logs[i]'
                    unset 'running_paths[i]'
                    unset 'running_starts[i]'
                    unset 'running_total_eps[i]'
                    unset 'running_tmp[i]'
                    running_pids=("${running_pids[@]}")
                    running_logs=("${running_logs[@]}")
                    running_paths=("${running_paths[@]}")
                    running_starts=("${running_starts[@]}")
                    running_total_eps=("${running_total_eps[@]}")
                    running_tmp=("${running_tmp[@]}")
                    return 0
                fi
            done
            sleep 1
        done
    }

    for item in "${plan[@]}"; do
        IFS='|' read -r inst seed every total_eps est_ep_s <<< "$item"
        log_name="${inst}_s${seed}_t50e${every}_dynamic.jsonl"
        log_path="$LOGDIR/$log_name"
        est_total_s=$(( total_eps * est_ep_s ))

        if [ -f "$log_path" ]; then
            continue
        fi

        # Wait for a free slot if at capacity.
        while [ "${#running_pids[@]}" -ge "$WORKERS" ]; do
            wait_any_finish
        done

        launch_count=$((launch_count + 1))
        # Per-run tmp log so concurrent solvers don't overwrite each other.
        tmp_path="/tmp/gen_more_logs_$$_${launch_count}.txt"
        : > "$tmp_path"

        "$SOLVER" \
            --instance "$DATA/$inst.txt" \
            --seed "$seed" \
            --selection roulette \
            --penalty 10000 \
            --dynamic --epoch-length 30 --epoch-budget-ms 5000 \
            --log-trials --m-trials 50 --m-trials-every "$every" \
            --log-dir "$LOGDIR" \
            > "$tmp_path" 2>&1 &
        solver_pid=$!

        running_pids+=("$solver_pid")
        running_logs+=("$log_name")
        running_paths+=("$log_path")
        running_starts+=("$SECONDS")
        running_total_eps+=("$total_eps")
        running_tmp+=("$tmp_path")

        plan_elapsed=$((SECONDS - start_ts))
        printf "[launch %2d/%d]  %s   (est %d ep × %ds = %dm, %d running, plan %dm%ds)\n" \
            "$launch_count" "$todo" "$log_name" "$total_eps" "$est_ep_s" \
            "$((est_total_s/60))" "${#running_pids[@]}" \
            "$((plan_elapsed/60))" "$((plan_elapsed%60))"
    done

    # Drain remaining workers.
    while [ "${#running_pids[@]}" -gt 0 ]; do
        wait_any_finish
    done
fi

echo
echo "=== Done ==="
echo "Completed:  $((done_count - fail_count))/$todo"
echo "Failed:     $fail_count"
echo "Total logs in $LOGDIR: $(ls "$LOGDIR"/*.jsonl 2>/dev/null | wc -l)"
echo "Wall clock: $((SECONDS - start_ts))s"
echo
echo "Next: combine v1 + v2 logs for retraining."
echo
echo "Option A — v2-only ablation (clean comparison vs current models):"
echo "  cd ml_train"
echo "  python -m darpml extract-features --logs $LOGDIR/ --data ../data/ \\"
echo "    --out cache_v2only/features.parquet --mode dynamic \\"
echo "    --batches core,tightness,worst_support,lns_state,rolling_horizon"
echo "  python -m darpml compute-labels --logs $LOGDIR/ \\"
echo "    --out cache_v2only/labels.parquet --mode dynamic --keep-epoch-1"
echo
echo "Option B — combine v1 + v2 (recommended for the actual retrain):"
echo "  mkdir -p $REPO_ROOT/trail_logs_combined"
echo "  ln -sf $REPO_ROOT/trail_logs/*.jsonl    $REPO_ROOT/trail_logs_combined/"
echo "  ln -sf $LOGDIR/*.jsonl                  $REPO_ROOT/trail_logs_combined/"
echo "  cd ml_train"
echo "  python -m darpml extract-features --logs ../trail_logs_combined/ --data ../data/ \\"
echo "    --out cache_v2_combined/features.parquet --mode dynamic \\"
echo "    --batches core,tightness,worst_support,lns_state,rolling_horizon"
echo "  python -m darpml compute-labels --logs ../trail_logs_combined/ \\"
echo "    --out cache_v2_combined/labels.parquet --mode dynamic --keep-epoch-1"
echo "  python train_candidates.py \\"
echo "    --labels cache_v2_combined/labels.parquet --out-suffix _v2combined"
