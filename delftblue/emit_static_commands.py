#!/usr/bin/env python3
"""Emit darp_solver CLI per line for the STATIC trail-log corpus.

Grid: Cordeau pr01..pr20 x 5 seeds x 2 sel x 2 iter counts (1k, 5k) = 400 logs
    + small Manhattan static x 5 seeds x 2 sel x 5000 iters             =  90 logs
                                                                  TOTAL = 490 logs

Static-mode runs are much faster than dynamic (Cordeau pr01 ~30s for 5k iters,
pr20 ~4min). Whole corpus generates in ~1.5h on 64 CPUs.
"""

SOLVER       = "/home/dhdimitrov/thesis-darp/cpp/build/darp_solver"
CORDEAU_DIR  = "/home/dhdimitrov/thesis-darp/data/cordeau"
FIEDLER_DIR  = "/scratch/dhdimitrov/data/fiedler_converted"
LOG_DIR      = "/scratch/dhdimitrov/trail_logs_v7_static"

CORDEAU_INSTANCES = [f"pr{i:02d}" for i in range(1, 21)]   # pr01..pr20

# Small Manhattan instances safe to run statically (short ALNS, classical DARP).
MANHATTAN_STATIC = [
    "manhattan_30_s_delay03_min",
    "manhattan_30_s_delay05_min",
    "manhattan_30_s_delay10_min",
    "manhattan_01_min_delay03_min",
    "manhattan_01_min_delay05_min",
    "manhattan_01_min_delay10_min",
    "manhattan_01min_03min",
    "manhattan_01min_05min",
    "manhattan_01min_10min",
]

SEEDS = list(range(100, 105))             # 5 seeds, disjoint from bench {1,2,3}
SELECTIONS = ["roulette", "random"]
M_TRIALS_EVERY = 20                       # denser logging in static (runs are short)


def emit(instance_path: str, n_iter: int, seed: int, sel: str) -> str:
    return (
        f"{SOLVER}"
        f" --instance {instance_path}"
        f" --seed {seed}"
        f" --selection {sel}"
        f" --penalty 10000"
        f" --iterations {n_iter}"
        f" --log-trials"
        f" --m-trials 50"
        f" --m-trials-every {M_TRIALS_EVERY}"
        f" --log-dir {LOG_DIR}"
        f" > /dev/null 2>&1"
    )


def main() -> None:
    # Cordeau: 20 x 5 x 2 x 2 = 400
    for inst in CORDEAU_INSTANCES:
        for seed in SEEDS:
            for sel in SELECTIONS:
                for n_iter in (1000, 5000):
                    print(emit(f"{CORDEAU_DIR}/{inst}.txt", n_iter, seed, sel))

    # Manhattan small: 9 x 5 x 2 x 1 = 90
    for inst in MANHATTAN_STATIC:
        for seed in SEEDS:
            for sel in SELECTIONS:
                print(emit(f"{FIEDLER_DIR}/{inst}.txt", 5000, seed, sel))


if __name__ == "__main__":
    main()
