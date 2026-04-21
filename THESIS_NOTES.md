# Thesis Notes — implementation findings

Findings from the Part 1–5 ML feature pipeline build-out that are worth pulling
into the thesis methodology and reproducibility chapters. Each entry: enough
context to write the writeup section + a one-sentence implication for what the
thesis should claim.

Structured into two groups:

- **Resolved methodology findings** are bugs / planning errors that surfaced
  during build-out and were fixed. They speak to the testing and design
  process: how the issue was caught, what the fix was, and what
  reproducibility-chapter lesson generalises from it.
- **Live system properties** are characteristics of the shipping system that
  remain in production. They speak to limitations the thesis must disclose,
  not bugs to fix.

---

# Resolved methodology findings

## 1. Default-ctor 2K-route corruption in the parity fixture

**Context.** Adding a direct scheduler-parity test (Python `compute_route_schedule`
vs C++ `DARPSolution::compute_route_schedule`) immediately surfaced a latent
test-harness bug: `test_feature_parity` had been *appending* the JSONL fixture's
routes to a `DARPSolution` whose default constructor pre-populates `routes` with
K depot+vstart+depot empty routes (and seeds `unassigned` with `1..n`). Result:
`sol.routes` ended up with 2K entries — K stale defaults at indices 0..K-1, then
the actual fixture data. The pre-existing feature parity test had been passing
because empty routes contribute nothing to active-pickup iteration or route-slack
averaging, so the features happened to be no-ops on the stale prefix. Direct
scheduler comparison surfaced the bug instantly: 40/40 states reported route
count mismatches at index `r < K`. Fixed by `clear()`ing both `routes` and
`unassigned` before populating from the fixture
([test_feature_parity.cpp:113-115](cpp/tests/test_feature_parity.cpp#L113-L115)).

**Implication for the thesis.** Indirect verification via aggregate features can
mask state-corruption bugs that direct per-component verification catches
immediately. The reproducibility chapter should make the case that ML pipeline
parity tests need to verify intermediate computations (scheduler outputs, helper
functions) directly, not only the final feature values that consume them.

## 2. FMA divergence in Cordeau distance parsing

**Context.** The Part 3 synthetic fixture (5 hand-crafted states on Cordeau pr01,
24 requests) failed parity at `cost_contribution_max_ratio` with
`max_abs_err=4.44e-16` — exactly 1 ULP at magnitude ~1.5. Trajectory fixtures on
Fiedler instances had been passing at `max_abs_err=0`. Tracing showed the
divergence wasn't in the new feature code: it was in the underlying travel-time
matrix. arm64 clang fuses `dx*dx + dy*dy` inside `std::sqrt(...)` into a single
FMA instruction, giving correctly-rounded distance with one rounding instead of
three. Numpy's `np.sqrt(dx**2 + dy**2)` doesn't FMA — three separate roundings,
last bit different. For the specific node pair `(3, 26)` in pr01, the two paths
produced doubles 1 ULP apart; that propagated through `removal_cost` and through
the sum/divide of the max-ratio formula. Fixed by switching the Python Cordeau
parser to per-pair `math.hypot(dx, dy)`
([instance.py:215-225](ml_train/darpml/instance.py#L215-L225)) — which matches
the FMA-fused C++ result bit-for-bit and is also more accurate than the previous
numpy path. Fiedler instances unaffected because they load tt from a precomputed
file (no parser arithmetic).

**Implication for the thesis.** Cross-language ML pipelines that share a
numerical reference must consider compiler-architecture interactions, not just
algorithmic equivalence. The methodology chapter should document the
`math.hypot` choice as a reproducibility decision and note that synthetic
fixtures with computed (rather than file-loaded) instances are needed to surface
this class of bug — file-loaded instances are bit-identical and would silently
hide the divergence.

## 3. `flexibility_horizon` was structurally degenerate on Manhattan dynamic logs

**Context.** The Part 5 `rolling_horizon` batch shipped with `flexibility_horizon`
= mean over active pickups of `max(0, pickup_tw_start - sim_clock) /
HORIZON_CONSTANT`, intended to capture how much remaining horizon there was
before each active pickup's window opens. After the first 21-feature training
run, a feature-distribution diagnostic on the resulting 13,762-row parquet
showed the feature was constant 0 across every single row. Tracing it: the
dynamic solver only fires an epoch when at least
`MIN_REQUESTS_FOR_EPOCH=10` requests have been released, and by that point all
"active" pickup nodes (in routes but not committed) have had their time
windows already opened — sim_clock has moved past every active pickup's
`earliest`. The clip-to-0-for-negative branch fires for every contribution,
so the mean is always 0. Confirmed empirically on the 02_min log:
`sim_clock=65070`, all 55 active pickup `earliests ∈ [64891, 64920]`. Removed
from the registry rather than reformulated; the original feature carried zero
information on this dataset and was diluting RF splits. Reformulation ideas
(time to window CLOSE: `(latest - sim_clock) / HORIZON`; or scheduler-derived
slack-until-actual-pickup) are deferred to a future feature batch.

**Implication for the thesis.** Features that look meaningful in static-DARP
formulations can become structurally degenerate in rolling-horizon settings
when the gating logic for epoch firing systematically delays evaluation past
the feature's informative regime. The methodology chapter should make this
generalizable: in dynamic settings, every proposed feature should be
distribution-checked on real extracted data before it's accepted into the
production set, not just verified for parity. The four constant-0 / mostly-0
features surfaced in this diagnostic (one structurally degenerate, two
real-world-rare, one near-constant) provide concrete examples for that
methodology recommendation.

## 4. `time_to_next_epoch_norm` was a degenerate spec

**Context.** The Part 5 plan called for a feature
`time_to_next_epoch_norm = (next_epoch_time - sim_clock) / epoch_length` to give
the ML model a sense of where in the rolling-horizon timeline the current epoch
sits. Pre-implementation reading of `dynamic_solver.cpp` revealed the feature
would be degenerate: `sim_clock_` is **frozen** during ALNS iterations (the only
context where features get evaluated), and `next_epoch_time` is updated in the
outer simulation loop, not within ALNS. Either interpretation
(in-loop value of `next_epoch_time` or post-update logical value) produces a
constant within an epoch and a near-constant across epochs — no signal for ML
training. Verified empirically by inspecting JSONL `epoch_start.sim_clock`
values across epochs of the 02_min log. Replaced with `horizon_elapsed_frac`
= `(sim_clock - sim_start) / (sim_end - sim_start)` per Bongiovanni 2022 #2
("planning horizon elapsed [%]"), clipped to [0, 1]. This *does* vary across
epochs and carries genuine signal about operational-day position.

**Implication for the thesis.** Feature specifications drawn from external
papers need to be checked against the local solver's actual timing behavior
before implementation, not just for semantic correctness. The methodology
chapter should cite Bongiovanni 2022 #2 attribution for `horizon_elapsed_frac`
and frame the original `time_to_next_epoch_norm` planning error as a
representative example of why pre-implementation verification (reading the
solver loop for the actual variable life-cycle) is part of the feature design
process, not an optional sanity check.

---

# Live system properties

## 5. `current_q` 1-iteration lag from solver loop ordering

**Context.** The `lns_state` feature `neighborhood_size_frac = q / q_max` needs
the destroy operator's neighborhood-size parameter `q` at feature evaluation
time. But `q` is sampled by the destroy operator itself, which runs *after*
`MLSelector::select_pair` in the solver loop. At runtime, the only knowable `q`
at feature-eval time on iteration N is `q[N-1]` (the previous iteration's
value); `q[N]` doesn't exist yet. JSONL log records `q[N]` (the q used during
iteration N), so parity-time training extraction sees `q[N]` while runtime
inference sees `q[N-1]`. Both populate `ALNSState.current_q` consistently for
parity tests (same fixture value to both Python and C++ feature code), but
introduce a 1-iteration semantic shift between training and inference.
Documented in [cpp/include/ml/alns_state.h](cpp/include/ml/alns_state.h)
header comment. Mitigations considered (track per-call shift in the loader;
defer feature evaluation post-destroy) were rejected as out-of-scope changes to
the solver loop or the JSONL schema.

**Implication for the thesis.** Real-time inference contexts where features
depend on operator outputs introduce inherent timing skew that training-data
extraction can't perfectly match without invasive solver loop changes. The
thesis should disclose this as a known training/inference distribution shift
and use the trained model's feature importance ranking for `neighborhood_size_frac`
as an empirical check: high importance means the lag is tolerable in practice;
low importance means the feature design is worth revisiting.
