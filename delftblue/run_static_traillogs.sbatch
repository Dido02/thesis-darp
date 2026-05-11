#!/bin/bash
#SBATCH --job-name=darp_static_logs
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/dhdimitrov/trail_logs_v7_static/slurm-%j.out
#SBATCH --error=/scratch/dhdimitrov/trail_logs_v7_static/slurm-%j.err

# Generate ~490 static-mode DARP trail logs (Cordeau pr01..pr20 + small Manhattan).
# Submit with: sbatch delftblue/run_static_traillogs.sbatch

set -euo pipefail

module load 2024r1
module load gcc/11.3.0 cmake/3.27.7

REPO=/home/dhdimitrov/thesis-darp
OUT=/scratch/dhdimitrov/trail_logs_v7_static
mkdir -p "$OUT"

CMDS=/tmp/static_cmds.$SLURM_JOB_ID
python3 "$REPO/delftblue/emit_static_commands.py" > "$CMDS"
echo "Emitted $(wc -l < "$CMDS") commands, starting xargs -P 64..."
date

xargs -P 64 -I {} bash -c '{}' < "$CMDS"

echo ""
date
echo "Done. Files in $OUT:"
ls "$OUT" | wc -l
du -sh "$OUT"
