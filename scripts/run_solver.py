"""Run ALNS solver on a Cordeau instance."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.cordeau import parse_cordeau
from src.solver import solve

DATA_DIR = PROJECT_ROOT / "data" / "cordeau"

instance = parse_cordeau(DATA_DIR / "pr03.txt")
result = solve(instance, max_iterations=200, seed=42)

print("\n=== INITIAL SOLUTION ===")
result.initial.print_solution()

print("\n=== BEST SOLUTION ===")
best = result.best
best.print_solution()

feasible, reason = best.is_feasible()
print(f"\nFeasible: {feasible}" + (f" ({reason})" if not feasible else ""))
print(f"Unassigned: {len(best.unassigned)}")
print(f"Construction: {result.construction_time:.3f}s")
print(f"ALNS solve:   {result.solve_time:.1f}s")