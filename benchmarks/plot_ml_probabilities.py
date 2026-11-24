"""Plot per-iteration ML selector probabilities from a --log-ml-probs JSONL file.

Reads a JSONL produced by darp_solver --log-ml-probs and emits two plots:
- Line plot: one curve per (destroy, repair) pair, smoothed with a rolling mean.
- Heatmap: x = cumulative call index, y = pair label, color = P(good).

The JSONL has one or more "header" records (re-emitted per epoch in dynamic
mode because MLSelector is reconstructed each epoch). All headers must agree
on pair_order; the script verifies and uses the first one. Subsequent
{"call_idx", "epoch_iter", "sim_clock", "probs"} records are concatenated in
file order — that order is the cumulative trajectory across epochs.

Usage:
    python benchmarks/plot_ml_probabilities.py \\
        --in /tmp/probs.jsonl \\
        --out-dir plots \\
        [--smooth 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    """Read JSONL file. Returns (pair_order, records).

    pair_order is a list of (destroy, repair) tuples taken from the FIRST
    header line. Headers re-emitted later are validated against the first.
    Records are returned in file order.
    """
    pair_order: Optional[list[tuple[str, str]]] = None
    records: list[dict] = []
    with open(path) as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln_no}: invalid JSON: {e}")
            if obj.get("type") == "header":
                p = [tuple(pair) for pair in obj.get("pair_order", [])]
                if pair_order is None:
                    pair_order = p
                elif pair_order != p:
                    raise SystemExit(
                        f"{path}:{ln_no}: header pair_order changed mid-file. "
                        f"First was {pair_order}, here {p}.")
            else:
                records.append(obj)
    if pair_order is None:
        raise SystemExit(f"{path}: no header record found.")
    return pair_order, records


def build_matrix(records: list[dict],
                 pair_order: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (cum_idx, probs_matrix). cum_idx is 0..N-1 in file order
    (the cumulative trajectory), probs_matrix is shape (N, n_pairs)."""
    n_pairs = len(pair_order)
    probs = np.zeros((len(records), n_pairs), dtype=np.float64)
    for i, rec in enumerate(records):
        p = rec.get("probs") or []
        if len(p) != n_pairs:
            raise SystemExit(
                f"Record {i}: probs length {len(p)} != n_pairs {n_pairs}")
        probs[i, :] = p
    return np.arange(len(records)), probs


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean over rows; pads ends with the local average."""
    if w <= 1 or x.shape[0] < w:
        return x
    kernel = np.ones(w) / w
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], kernel, mode="same")
    return out


def epoch_boundaries(records: list[dict]) -> list[int]:
    """Indices in file order where a new epoch starts (sim_clock changes
    or epoch_iter resets to 0)."""
    boundaries = []
    last_clock = None
    for i, rec in enumerate(records):
        sc = rec.get("sim_clock")
        if last_clock is None:
            last_clock = sc
            continue
        # New epoch if sim_clock changed OR epoch_iter went back to 0 with prior > 0
        if sc != last_clock:
            boundaries.append(i)
            last_clock = sc
    return boundaries


def plot_line(probs: np.ndarray, pair_order: list[tuple[str, str]],
              out_path: Path, smooth: int,
              boundaries: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    smoothed = rolling_mean(probs, smooth) if smooth > 1 else probs
    cum_idx = np.arange(probs.shape[0])
    cmap = matplotlib.colormaps.get_cmap("tab20")
    n = len(pair_order)
    for j, (d, r) in enumerate(pair_order):
        ax.plot(cum_idx, smoothed[:, j], color=cmap(j % 20),
                label=f"{d}+{r}", lw=1.4, alpha=0.9)
    # Light vertical lines at epoch boundaries
    for b in boundaries:
        ax.axvline(b, color="grey", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Cumulative call index (across all epochs)")
    ax.set_ylabel(f"P(good) — rolling mean (window={smooth})")
    ax.set_title(
        f"ML selector per-pair probabilities  ({probs.shape[0]} calls, "
        f"{n} pairs, {len(boundaries)+1} epochs)")
    ax.legend(loc="upper right", ncol=min(3, (n + 5) // 6), fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_heatmap(probs: np.ndarray, pair_order: list[tuple[str, str]],
                 out_path: Path,
                 boundaries: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4 + 0.25 * len(pair_order)))
    pair_labels = [f"{d}+{r}" for d, r in pair_order]
    # Heatmap: rows = pairs (transposed from probs), cols = call index
    im = ax.imshow(probs.T, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("Cumulative call index")
    ax.set_ylabel("Pair (destroy + repair)")
    ax.set_title(
        f"ML selector probability heatmap  ({probs.shape[0]} calls, "
        f"{len(pair_labels)} pairs, {len(boundaries)+1} epochs)")
    # Epoch boundary markers as light vertical lines
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", lw=0.6, ls="--", alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("P(good)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to JSONL produced by --log-ml-probs")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory for output PNGs (default: plots)")
    ap.add_argument("--smooth", type=int, default=50,
                    help="Rolling-mean window for line plot (default: 50)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"ERROR: input file not found: {in_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {in_path} ...")
    pair_order, records = load_jsonl(in_path)
    print(f"  pair_order: {len(pair_order)} pairs")
    print(f"  records: {len(records)}")

    if not records:
        raise SystemExit("ERROR: no probability records in file.")

    cum_idx, probs = build_matrix(records, pair_order)
    print(f"  matrix shape: {probs.shape}, prob range [{probs.min():.4g}, {probs.max():.4g}]")
    bounds = epoch_boundaries(records)
    print(f"  epoch boundaries detected: {len(bounds)}")

    basename = in_path.stem  # input filename without extension
    line_out = out_dir / f"ml_probs_line_{basename}.png"
    heat_out = out_dir / f"ml_probs_heat_{basename}.png"
    print("\nGenerating plots ...")
    plot_line(probs, pair_order, line_out, args.smooth, bounds)
    plot_heatmap(probs, pair_order, heat_out, bounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
