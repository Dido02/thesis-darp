#!/bin/bash
#SBATCH --job-name=darp_dyn_logs
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/dhdimitrov/trail_logs_v7_dynamic/slurm-%j.out
#SBATCH --error=/scratch/dhdimitrov/trail_logs_v7_dynamic/slurm-%j.err

# Generate ~800 dynamic-mode DARP trail logs in parallel on 64 CPUs.
# Submit from anywhere with: sbatch delftblue/run_dynamic_traillogs.sbatch

set -euo pipefail

module load 2024r1
module load gcc/11.3.0 cmake/3.27.7

REPO=/home/dhdimitrov/thesis-darp
OUT=/scratch/dhdimitrov/trail_logs_v7_dynamic
mkdir -p "$OUT"

CMDS=/tmp/dyn_cmds.$SLURM_JOB_ID
python3 "$REPO/delftblue/emit_dynamic_commands.py" > "$CMDS"
echo "Emitted $(wc -l < "$CMDS") commands, starting xargs -P 64..."
date

# xargs keeps 64 darp_solver processes alive at all times. As one finishes,
# the next CLI line from the work list starts on that freed CPU.
xargs -P 64 -I {} bash -c '{}' < "$CMDS"

echo ""
date
echo "Done. Files in $OUT:"
ls "$OUT" | wc -l
du -sh "$OUT"
