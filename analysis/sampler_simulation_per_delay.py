#!/usr/bin/env python3
"""Simulate the deployed t07 sampler on cached training states.

For each of the 8,992 cached states in features.parquet, we:
  1. Run the 9 deployed (destroy, repair) forests → 9-vector of P(class=1)
  2. Apply softmax(p / 0.10) → 9-vector of sampling weights
  3. Pick under TWO strategies:
       a. ARGMAX  — deterministic, "what greedy sampler picks"
       b. SAMPLED — 100 samples/state from softmax weights, "what proportional sampler picks"
  4. Aggregate per delay tier

Outputs:
  - analysis/sampler_simulation_per_delay.csv (long-form table)
  - analysis/SAMPLER_SIMULATION_REPORT.md     (human-readable report)

IMPORTANT: This runs on TRAINING-LOG states, not bench states. The bench uses
disjoint seeds (11/22/33/44/55) and different (instance, budget, epoch length)
combinations from training. The state distribution overlaps but is not identical.
The report flags this explicitly.

Run:
  /Users/didodimitrov/Documents/thesis-darp/.venv/bin/python \\
    analysis/sampler_simulation_per_delay.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FEATURES = REPO / "ml_train" / "cache_v3_full" / "features.parquet"
MODEL_DIR = REPO / "ml_train" / "models_v3_full_20f_t07" / "onnx"

CLASSICAL = [
    ("random", "greedy"), ("random", "regret2"), ("random", "regret3"),
    ("worst",  "greedy"), ("worst",  "regret2"), ("worst",  "regret3"),
    ("shaw",   "greedy"), ("shaw",   "regret2"), ("shaw",   "regret3"),
]
# Abbreviated labels: dst+rep where rep ∈ {grd, r2, r3}
_REP_ABBR = {"greedy": "grd", "regret2": "r2", "regret3": "r3"}
_DST_ABBR = {"random": "rnd", "worst": "wst", "shaw": "shw"}
PAIR_LABELS = [f"{_DST_ABBR[d]}+{_REP_ABBR[r]}" for (d, r) in CLASSICAL]

TEMPERATURE = 0.10
N_SAMPLES_PER_STATE = 100  # for proportional sampling
ENTROPY_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.17]  # log2(9) = 3.17 max


def load_forests():
    sessions = {}
    for d_name, r_name in CLASSICAL:
        sess = ort.InferenceSession(
            str(MODEL_DIR / f"{d_name}_{r_name}.onnx"),
            providers=["CPUExecutionProvider"])
        sessions[(d_name, r_name)] = sess
    return sessions


def predict_p_class1(sess, X):
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    prob_idx = out_names.index("probabilities") if "probabilities" in out_names else 1
    out = sess.run([out_names[prob_idx]], {in_name: X})[0]
    if isinstance(out, list):
        return np.array([float(d[1]) for d in out], dtype=np.float64)
    arr = np.asarray(out)
    return (arr[:, 1] if arr.ndim == 2 else arr.flatten()).astype(np.float64)


def softmax_T(P, T):
    """P: shape (N, K). Returns row-normalised softmax(P / T)."""
    P_T = P / T
    P_T = P_T - P_T.max(axis=1, keepdims=True)
    e = np.exp(P_T)
    return e / e.sum(axis=1, keepdims=True)


def state_entropy(W):
    """Shannon entropy of each row (log2), shape (N,)."""
    return -np.sum(np.where(W > 0, W * np.log2(W), 0.0), axis=1)


def simulate_delay(df_delay, P_pairs, rng):
    """Run argmax + sampled picks on one delay tier."""
    n = len(df_delay)
    # Build (n, 9) prediction matrix in CLASSICAL order
    P = np.stack([P_pairs[k] for k in CLASSICAL], axis=1)
    W = softmax_T(P, TEMPERATURE)        # (n, 9)
    H = state_entropy(W)                 # (n,)
    eff = 2.0 ** H                       # effective # pairs

    # ARGMAX picks per state
    argmax_pick = W.argmax(axis=1)
    argmax_counts = np.bincount(argmax_pick, minlength=9)

    # SAMPLED picks: N_SAMPLES_PER_STATE per row, cumulative-distribution sampling
    cum = W.cumsum(axis=1)
    cum[:, -1] = 1.0  # guard against floating-point overshoot
    sampled_counts = np.zeros(9, dtype=np.int64)
    # Reuse a single big random matrix (n × N_SAMPLES) for speed
    u = rng.uniform(size=(n, N_SAMPLES_PER_STATE))
    # For each row, vectorised searchsorted across the 100 samples
    for i in range(n):
        picks = np.searchsorted(cum[i], u[i])
        picks = np.clip(picks, 0, 8)
        sampled_counts += np.bincount(picks, minlength=9)

    # State-by-state variance: std of w[k] across states for each pair k
    pair_w_std = W.std(axis=0, ddof=1)
    pair_w_mean = W.mean(axis=0)

    return dict(
        n_states=n,
        argmax_counts=argmax_counts,
        sampled_counts=sampled_counts,
        sampled_total=sampled_counts.sum(),
        eff_pairs=eff,
        entropy=H,
        pair_w_mean=pair_w_mean,
        pair_w_std=pair_w_std,
        W=W,
    )


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    df = pd.read_parquet(FEATURES)
    df["delay"] = df["log_file"].str.extract(r"delay(\d+)_min")[0]

    with open(MODEL_DIR / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_order = [f["name"] for f in manifest["features"]]
    X = df[feature_order].to_numpy(dtype=np.float32)

    print(f"Loaded {len(df):,} training states from {FEATURES.relative_to(REPO)}")
    print(f"By delay: {dict(df['delay'].value_counts().sort_index())}")
    print(f"Loading 9 forests from {MODEL_DIR.relative_to(REPO)} ...")

    sessions = load_forests()
    P_pairs = {}
    for k in CLASSICAL:
        P_pairs[k] = predict_p_class1(sessions[k], X)
    print(f"Predictions computed for all 9 pairs.")
    print()

    delays = sorted(df["delay"].dropna().unique())
    delay_indices = {d: (df["delay"] == d).to_numpy() for d in delays}

    results = {}
    for d in delays:
        mask = delay_indices[d]
        P_d = {k: P_pairs[k][mask] for k in CLASSICAL}
        results[d] = simulate_delay(df[mask], P_d, rng)
        print(f"Delay {d}: simulated {results[d]['n_states']} states.")
    print()

    # ====================================================================
    # WRITE CSV (long form)
    # ====================================================================
    rows = []
    for d in delays:
        r = results[d]
        for k_idx, k in enumerate(CLASSICAL):
            d_name, r_name = k
            rows.append({
                "delay": d,
                "destroy": d_name,
                "repair": r_name,
                "pair_label": PAIR_LABELS[k_idx],
                "n_states": r["n_states"],
                "argmax_count": int(r["argmax_counts"][k_idx]),
                "argmax_frac": float(r["argmax_counts"][k_idx] / r["n_states"]),
                "sampled_count": int(r["sampled_counts"][k_idx]),
                "sampled_frac": float(r["sampled_counts"][k_idx] / r["sampled_total"]),
                # state-level: mean and std of this pair's softmax weight ACROSS states
                "pair_w_mean": float(r["pair_w_mean"][k_idx]),
                "pair_w_std": float(r["pair_w_std"][k_idx]),
            })
    csv_path = REPO / "analysis" / "sampler_simulation_per_delay.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path.relative_to(REPO)}")

    # ====================================================================
    # WRITE MARKDOWN REPORT
    # ====================================================================
    lines = []
    P = lines.append

    P("# Sampler simulation per delay — t07 deployed forests")
    P("")
    P(f"**Date:** {pd.Timestamp.today().strftime('%Y-%m-%d')}  ")
    P(f"**Model:** `{MODEL_DIR.relative_to(REPO)}` — 9 classical pair forests, threshold-0.7 binary classifier")
    P(f"**Sampler temperature:** softmax(p / {TEMPERATURE})")
    P(f"**States:** {len(df):,} cached training states from `{FEATURES.relative_to(REPO)}`")
    P("")

    P("## ⚠ Important limitation")
    P("")
    P("This simulation runs the deployed forests on **training-log states**, not bench states.")
    P("")
    P("- The training trail logs use seeds `{789, 1024, 2025}` and 50 ms × 50 epochs runs")
    P("- The benchmarks (BENCH_FINAL, budget sweeps) use seeds `{11, 22, 33, 44, 55}` and different")
    P("  (instance × budget × epoch length) combinations")
    P("- The state distributions overlap (same instances, same operator config) but are **not")
    P("  identical** — bench states are drawn from a different SA-trajectory regime")
    P("")
    P("The pick distributions below say what the model **would** pick *given the kind of state")
    P("it saw during training data generation*. They are a strong proxy for, but not equivalent")
    P("to, the picks the model actually makes during a bench run. To get exact bench-distribution")
    P("picks, capture them with `--log-ml-probs` during a fresh bench run.")
    P("")

    P("## Methodology")
    P("")
    P("For each cached state:")
    P("")
    P("1. Extract the 20 features in manifest order from `features.parquet`")
    P("2. Run all 9 ONNX forests → 9-element vector of `P(class=1)`")
    P(f"3. Apply softmax(p / {TEMPERATURE}) → 9-element probability distribution")
    P("4. Compute two picks:")
    P("   - **argmax**: the most-likely pair (deterministic, like T → 0)")
    P(f"   - **sampled**: {N_SAMPLES_PER_STATE} independent draws from the softmax distribution (proportional sampler)")
    P("5. Aggregate per delay tier")
    P("")

    P("## 1. Pick distribution — ARGMAX (deterministic, 1 pick / state)")
    P("")
    P("If the sampler were temperature-0 (always pick argmax), this is the marginal:")
    P("")
    P("| delay |" + "".join(f" {lbl} |" for lbl in PAIR_LABELS) + " n_states |")
    P("|---|" + "".join(["---:|"] * 9) + "---:|")
    for d in delays:
        r = results[d]
        line = f"| d{d} |"
        for k_idx in range(9):
            frac = r["argmax_counts"][k_idx] / r["n_states"] * 100
            line += f" {frac:.1f}% |"
        line += f" {r['n_states']} |"
        P(line)
    P("")
    P("**Reading:** for each delay tier, what fraction of states picked each pair as their argmax?")
    P("(Uniform random would be 11.1% per pair.)")
    P("")

    P("## 2. Pick distribution — SAMPLED (proportional, 100 samples / state)")
    P("")
    P("Closer to the actual deployed C++ behavior — softmax weights drive a stochastic pick:")
    P("")
    P("| delay |" + "".join(f" {lbl} |" for lbl in PAIR_LABELS) + " n_picks |")
    P("|---|" + "".join(["---:|"] * 9) + "---:|")
    for d in delays:
        r = results[d]
        line = f"| d{d} |"
        for k_idx in range(9):
            frac = r["sampled_counts"][k_idx] / r["sampled_total"] * 100
            line += f" {frac:.1f}% |"
        line += f" {r['sampled_total']:,} |"
        P(line)
    P("")
    P(f"**Reading:** Over all {N_SAMPLES_PER_STATE} samples × n_states picks per delay, fraction going to each pair.")
    P("Uniform random would be 11.1%. Sampled approximates the true deployed behavior more closely than argmax.")
    P("")

    P("## 3. Concentration — softmax effective # pairs (state-by-state)")
    P("")
    P("Per state, `effective # pairs = 2^entropy`. Uniform random = 9; deterministic = 1.")
    P("")
    P("| delay | mean eff. pairs | median | std | frac states with eff ≥ 6 (near-uniform) | frac states with eff ≤ 2 (decisive) |")
    P("|---|---:|---:|---:|---:|---:|")
    for d in delays:
        r = results[d]
        eff = r["eff_pairs"]
        line = f"| d{d} | {eff.mean():.2f} | {np.median(eff):.2f} | {eff.std(ddof=1):.2f} | {(eff >= 6).mean()*100:.1f}% | {(eff <= 2).mean()*100:.1f}% |"
        P(line)
    P("")
    P("**Reading:** mean alone hides bimodality. The 'eff ≥ 6' column = fraction of states where the")
    P("sampler is essentially uniform (like random). The 'eff ≤ 2' column = fraction where the sampler")
    P("collapses to one or two pairs (decisive).")
    P("")

    P("## 4. Per-pair softmax-weight statistics across states")
    P("")
    P("For each pair, mean ± std of its softmax probability across all in-tier states.")
    P("Large std = the model's bias toward this pair varies a lot state-to-state.")
    P("Small std = state-invariant — model has a fixed opinion about this pair.")
    P("")
    P("| pair |" + "".join(f" d{d} mean |" for d in delays) + "".join(f" d{d} std |" for d in delays) + "")
    P("|---|" + "".join(["---:|"] * (2 * len(delays))))
    for k_idx, lbl in enumerate(PAIR_LABELS):
        line = f"| {lbl} |"
        for d in delays:
            line += f" {results[d]['pair_w_mean'][k_idx]:.3f} |"
        for d in delays:
            line += f" {results[d]['pair_w_std'][k_idx]:.3f} |"
        P(line)
    P("")

    P("## 5. Per-state entropy histogram")
    P("")
    P("Distribution of state-level Shannon entropy values (log₂, max = log₂(9) = 3.17).")
    P("Bin counts as fraction of in-tier states.")
    P("")
    # Slightly extend the top bin edge so values equal to log2(9) are included.
    hist_edges = ENTROPY_BINS.copy()
    hist_edges[-1] = hist_edges[-1] + 1e-9
    bin_labels = [f"[{ENTROPY_BINS[i]:.2f}, {ENTROPY_BINS[i+1]:.2f}]"
                  for i in range(len(ENTROPY_BINS) - 1)]
    P("| delay |" + "".join(f" {lbl} |" for lbl in bin_labels))
    P("|---|" + "".join(["---:|"] * len(bin_labels)))
    for d in delays:
        H = results[d]["entropy"]
        counts, _ = np.histogram(H, bins=hist_edges)
        n = len(H)
        line = f"| d{d} |"
        for c in counts:
            line += f" {c/n*100:.1f}% |"
        P(line)
    P("")
    P("**Reading:** entropy bins from 0 (deterministic) to ~3.17 (uniform). The histogram tells you")
    P("the SHAPE of the sampler's behavior — not just its mean.")
    P("")

    P("## Synthesis")
    P("")
    P("Look at columns side-by-side per delay:")
    P("")
    for d in delays:
        r = results[d]
        eff = r["eff_pairs"]
        argmax_top = int(np.argmax(r["argmax_counts"]))
        sampled_top = int(np.argmax(r["sampled_counts"]))
        P(f"- **delay {d}**:  "
          f"argmax favours `{PAIR_LABELS[argmax_top]}` "
          f"({r['argmax_counts'][argmax_top]/r['n_states']*100:.1f}%), "
          f"sampled favours `{PAIR_LABELS[sampled_top]}` "
          f"({r['sampled_counts'][sampled_top]/r['sampled_total']*100:.1f}%); "
          f"mean eff. pairs = {eff.mean():.2f}; "
          f"near-uniform on {(eff >= 6).mean()*100:.0f}% of states; "
          f"decisive on {(eff <= 2).mean()*100:.0f}% of states.")
    P("")
    P("**Interpretation cues:**")
    P("- If argmax and sampled distributions agree closely, softmax is sharp enough that proportional sampling barely differs from greedy.")
    P("- If they diverge significantly, the model's opinions are weak enough that stochastic sampling spreads picks across multiple plausible pairs.")
    P("- The 'frac near-uniform' column reveals when the sampler effectively gives up — those are states where the model has no opinion and the C++ selector behaves close to random.")
    P("- The 'frac decisive' column reveals when the sampler commits hard — those are the states where ML actually steers the search.")
    P("")
    P("## Files")
    P("")
    P("- Raw long-form numbers: [`analysis/sampler_simulation_per_delay.csv`](sampler_simulation_per_delay.csv)")
    P("- Analysis script: [`analysis/sampler_simulation_per_delay.py`](sampler_simulation_per_delay.py)")
    P("")

    report_path = REPO / "analysis" / "SAMPLER_SIMULATION_REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
