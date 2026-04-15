"""Print summary statistics from an ALNS log. Works with trials-only or full logs."""
import sys
from collections import Counter
from parse_logs import parse_log
from output_utils import get_output_path
import numpy as np


def main(log_path: str):
    data = parse_log(log_path)
    states = data["states"]
    trials = data["trials"]
    config = data["config"]
    d_names = config.destroy_ops
    r_names = config.repair_ops
    nd, nr = len(d_names), len(r_names)

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p(f"=== ALNS Run Summary ===")
    p(f"Instance: {config.instance}")
    p(f"Seed: {config.seed}")
    p(f"Requests: {config.num_requests}, Vehicles: {config.num_vehicles}")
    p(f"Dynamic: {config.dynamic}, m_trials: {config.m_trials}")
    p(f"SA temp: {config.sa_start_temp:.1f} -> {config.sa_end_temp:.1f}")
    p(f"State records: {len(states)}, Trial records: {len(trials)}")
    p()

    # --- State-based stats (if available) ---
    if states:
        objs = [s.obj_before for s in states]
        bests = [s.obj_best for s in states]
        p(f"Initial obj: {objs[0]:.0f}")
        p(f"Final obj:   {objs[-1]:.0f}")
        p(f"Best obj:    {min(bests):.0f}")
        p()

        n_accept = sum(1 for s in states if s.accept)
        n_improve = sum(1 for s in states if s.delta < 0)
        n_worsen_accept = sum(1 for s in states if s.accept and s.delta > 0)
        n_neutral = sum(1 for s in states if s.delta == 0)
        p(f"Accepted: {n_accept}/{len(states)} ({100*n_accept/len(states):.1f}%)")
        p(f"  Improving (delta<0): {n_improve}")
        p(f"  Neutral (delta=0):   {n_neutral}")
        p(f"  Worsening accepted:  {n_worsen_accept}")
        p(f"Rejected: {len(states) - n_accept}")
        p()

        p("--- Destroy Operators (from states) ---")
        for di, name in enumerate(d_names):
            ss = [s for s in states if s.d_idx == di]
            if not ss:
                continue
            deltas = [s.delta for s in ss]
            acc = sum(1 for s in ss if s.accept)
            imp = sum(1 for s in ss if s.delta < 0)
            p(f"  {name:15s}: selected {len(ss):4d}, "
              f"accept {acc:3d} ({100*acc/len(ss):4.1f}%), "
              f"improve {imp:3d}, "
              f"avg_delta {np.mean(deltas):7.1f}, "
              f"min_delta {min(deltas):7.1f}")

        p()
        p("--- Repair Operators (from states) ---")
        for ri, name in enumerate(r_names):
            ss = [s for s in states if s.r_idx == ri]
            if not ss:
                continue
            deltas = [s.delta for s in ss]
            acc = sum(1 for s in ss if s.accept)
            imp = sum(1 for s in ss if s.delta < 0)
            p(f"  {name:15s}: selected {len(ss):4d}, "
              f"accept {acc:3d} ({100*acc/len(ss):4.1f}%), "
              f"improve {imp:3d}, "
              f"avg_delta {np.mean(deltas):7.1f}, "
              f"min_delta {min(deltas):7.1f}")
        p()

    # --- Trial-based stats (always useful) ---
    if trials:
        p("--- Operator Landscape (from trials) ---")
        p(f"{'Destroy':15s} {'Repair':10s} {'AvgMean':>9s} {'AvgMin':>9s} {'ZeroFrac':>9s} {'Samples':>8s}")
        for di in range(nd):
            for ri in range(nr):
                all_means = []
                all_mins = []
                all_zeros = []
                for tr in trials:
                    deltas = tr.deltas.get((di, ri))
                    if deltas:
                        all_means.append(np.mean(deltas))
                        all_mins.append(min(deltas))
                        all_zeros.append(sum(1 for d in deltas if d == 0) / len(deltas))
                if all_means:
                    p(f"  {d_names[di]:13s} {r_names[ri]:10s} "
                      f"{np.mean(all_means):9.1f} {np.mean(all_mins):9.1f} "
                      f"{np.mean(all_zeros):9.2f} {len(all_means):8d}")
        p()

        # Best combo by avg min-delta
        best_combo = None
        best_val = float("inf")
        for di in range(nd):
            for ri in range(nr):
                mins = []
                for tr in trials:
                    deltas = tr.deltas.get((di, ri))
                    if deltas:
                        mins.append(min(deltas))
                if mins and np.mean(mins) < best_val:
                    best_val = np.mean(mins)
                    best_combo = (d_names[di], r_names[ri])
        if best_combo:
            p(f"Best combo by avg best-delta: {best_combo[0]} + {best_combo[1]} ({best_val:.1f})")

    # Save to file
    out = get_output_path(log_path, "_summary.txt", "summary")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_stats.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
