"""Plot ALNS convergence: objective over iterations with SA temperature."""
import sys
import matplotlib.pyplot as plt
import numpy as np
from parse_logs import parse_log
from output_utils import get_output_path


def main(log_path: str):
    data = parse_log(log_path)
    states = data["states"]
    config = data["config"]

    if not states:
        print("No state records found - this log may be trials-only.")
        return

    iters = [s.iter for s in states]
    obj_before = [s.obj_before for s in states]
    obj_best = [s.obj_best for s in states]
    temperature = [s.temperature for s in states]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Objective plot
    ax1.plot(iters, obj_before, "b-", alpha=0.4, linewidth=0.5, label="Current obj")
    ax1.plot(iters, obj_best, "r-", linewidth=1.5, label="Best obj")

    # Mark accepted/rejected
    acc_iters = [s.iter for s in states if s.accept and s.delta != 0]
    acc_obj = [s.obj_candidate for s in states if s.accept and s.delta != 0]
    rej_iters = [s.iter for s in states if not s.accept]
    rej_obj = [s.obj_candidate for s in states if not s.accept]
    ax1.scatter(acc_iters, acc_obj, c="green", s=8, alpha=0.5, zorder=5, label="Accepted (non-zero)")
    ax1.scatter(rej_iters, rej_obj, c="red", s=4, alpha=0.2, zorder=4, label="Rejected")

    ax1.set_ylabel("Objective")
    ax1.legend(loc="upper right")
    ax1.set_title(f"ALNS Convergence - {config.instance.split('/')[-1]} (seed={config.seed})")
    ax1.grid(True, alpha=0.3)

    # Temperature plot
    ax2.plot(iters, temperature, "orange", linewidth=1)
    ax2.set_ylabel("SA Temperature")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = get_output_path(log_path, "_convergence.png", "plots")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_convergence.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
