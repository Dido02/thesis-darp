"""Configurable feasibility checker factory.

Three modes:
  - "slack":   Steps 1+2 + per-pickup slack correction (default, fast)
  - "cordeau": Pure AEAP forward pass, no optimization (most conservative)
  - "full":    3-step optimizer + constraint checks (most accurate, slowest)
"""

from __future__ import annotations

from typing import Callable

from ..instance import DARPInstance

FeasibilityChecker = Callable[[list[int], DARPInstance], tuple[bool, str]]

_VALID_MODES = ("slack", "cordeau", "full")


def get_feasibility_checker(mode: str) -> FeasibilityChecker:
    """Return the check() function for the given feasibility mode."""
    if mode == "slack":
        from .slack_corrected import check
    elif mode == "cordeau":
        from .cordeau import check
    elif mode == "full":
        from .full_optimizer import check
    else:
        raise ValueError(
            f"Unknown feasibility mode {mode!r}. "
            f"Valid modes: {_VALID_MODES}"
        )
    return check
