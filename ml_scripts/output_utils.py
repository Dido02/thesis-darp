"""Shared output path helpers for analysis scripts."""
from pathlib import Path


def get_output_path(log_path: str, suffix: str, subfolder: str) -> str:
    """Build output path in a subfolder next to the log file.

    Example: get_output_path("trial_logs/foo.jsonl", "_convergence.png", "plots")
             -> "trial_logs/plots/foo_convergence.png"
    """
    p = Path(log_path)
    out_dir = p.parent / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / (p.stem + suffix))
