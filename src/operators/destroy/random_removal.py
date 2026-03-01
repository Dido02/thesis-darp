"""Random removal destroy operator for ALNS."""

from __future__ import annotations

import numpy as np

from ...solution import DARPSolution


def random_removal(
    sol: DARPSolution,
    rng: np.random.Generator,
    min_pct: float = 0.10,
    max_pct: float = 0.40,
) -> DARPSolution:
    """Remove a random subset of requests from the solution.

    Removes between min_pct and max_pct of currently assigned requests.
    """
    destroyed = sol.copy()
    n = destroyed.instance.num_requests

    # Collect assigned requests
    assigned = list(set(range(1, n + 1)) - destroyed.unassigned)
    if not assigned:
        return destroyed

    # Number to remove
    lo = max(1, int(len(assigned) * min_pct))
    hi = max(lo + 1, int(len(assigned) * max_pct) + 1)
    q = rng.integers(lo, hi)
    to_remove = rng.choice(assigned, size=min(q, len(assigned)), replace=False)

    for req in to_remove:
        _remove_request(destroyed, int(req))

    return destroyed


def _remove_request(sol: DARPSolution, req: int):
    """Remove a request (pickup + delivery) from its route."""
    pickup_id = req
    delivery_id = req + sol.instance.num_requests

    for route in sol.routes:
        if pickup_id in route:
            route.remove(pickup_id)
            route.remove(delivery_id)
            sol.unassigned.add(req)
            return