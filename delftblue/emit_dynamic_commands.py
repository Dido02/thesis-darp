#!/usr/bin/env python3
"""Emit one darp_solver CLI per line for the DYNAMIC trail-log corpus.

Output is consumed by `xargs -P 64` inside the sbatch script so the SLURM job
runs up to 64 solver processes in parallel.

Grid: 18 Manhattan instances x 22 seeds x 2 selection strategies = 792 logs.
"""

SOLVER   = "/home/dhdimitrov/thesis-darp/cpp/build/darp_solver"
DATA_DIR = "/scratch/dhdimitrov/data/fiedler_converted"
LOG_DIR  = "/scratch/dhdimitrov/trail_logs_v7_dynamic"

# (instance_base, m_trials_every). m_trials_every controls how often we pause
# the ALNS loop to run the 9-pair x M=50 trial sweep. Smaller = denser state
# logging. Tuned per instance size to keep wall-time per run roughly bounded.
INSTANCES = [
    ("manhattan_30_s_delay03_min",    100),
    ("manhattan_30_s_delay05_min",    100),
    ("manhattan_30_s_delay10_min",    100),
    ("manhattan_01_min_delay03_min",   50),
    ("manhattan_01_min_delay05_min",   50),
    ("manhattan_01_min_delay10_min",   50),
    ("manhattan_01min_03min",          50),
    ("manhattan_01min_05min",          50),
    ("manhattan_01min_10min",          50),
    ("manhattan_02_min_delay03_min",   20),
    ("manhattan_02_min_delay05_min",   20),
    ("manhattan_02_min_delay10_min",   20),
    ("manhattan_05_min_delay03_min",   10),
    ("manhattan_05_min_delay05_min",   10),
    ("manhattan_05_min_delay10_min",   10),
    ("manhattan_15min_03min",          10),
    ("manhattan_15min_05min",          10),
    ("manhattan_15min_10min",          10),
]

# 22 seeds, disjoint from bench seeds {1,2,3} (per CLAUDE.md rule).
SEEDS = list(range(100, 122))

# Two selection strategies broaden state-space coverage.
SELECTIONS = ["roulette", "random"]


def main() -> None:
    for inst, m_every in INSTANCES:
        for seed in SEEDS:
            for sel in SELECTIONS:
                cmd = (
                    f"{SOLVER}"
                    f" --instance {DATA_DIR}/{inst}.txt"
                    f" --seed {seed}"
                    f" --selection {sel}"
                    f" --penalty 10000"
                    f" --dynamic"
                    f" --epoch-length 30"
                    f" --epoch-budget-ms 5000"
                    f" --log-trials"
                    f" --m-trials 50"
                    f" --m-trials-every {m_every}"
                    f" --log-dir {LOG_DIR}"
                    f" > /dev/null 2>&1"
                )
                print(cmd)


if __name__ == "__main__":
    main()
