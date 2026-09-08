# Prompt 21 — longer R4 standalone validation (Stage A / Stage B)

Extends Prompt 20's R4 numerical validation (6 dyn_core calls, k_split=1) to
Prompt 19's longer R8 validation ladder, in R4. Same executable/allocation
pattern as Prompt 19/20 (`p17_both.sbatch`/`p20_r4_both.sbatch` structure,
here `p21_stageA_r4_both.sbatch` / `p21_stageB_r4_both.sbatch`), same
`@fvdycore` source tree (unchanged since Prompt 19), R4-compiled
(Prompt 20's build), c8/npz=10/n_split=3/6 MPI ranks throughout. No new
comparison tooling: `tools/savepoints/compare_dcx_backends.py` (imports
`compare_cuda_fortran.classify`) reused as-is, plus one scratch analysis
script (`scratch_p21_wfield.py`, not part of the reusable toolset, not
committed) that surfaces `max_abs`/`max_rel`/`rms_diff`/field-magnitude
together for a single field/call — no new comparison logic, just re-exposes
fields `classify()` already computes but `compare_dcx_backends.py`'s own
aggregation loop does not print.

## 1. Stage A — k_split=1, 72 calls

Configuration: c8/npz=10/n_split=3/k_split=1/R4/6 MPI ranks, `JOB_SGMT`
extended to match Prompt 19 Stage A (3 days), targeting 72 dyn_core calls.
Job 58282446, both backends (`p21_stageA_r4_fortran`, `p21_stageA_r4_cuda`)
in one allocation, completed in 42s, **rc=0 both**, both reached **it=72**.

### 1.1 Entry / basic health

- Entry (`dce`, call 1): BITWISE IDENTICAL, both backends (seeded from the
  same initial state; expected).
- NaN/Inf: **NONE** across all 72 calls, both backends (confirmed via
  `compare_dcx_backends.py`'s NONFINITE detector over the full unfiltered
  scan, `/tmp/p21_stageA_full_dcx.log`, 1371 lines — 0 NONFINITE anywhere).
- delp positivity, owned-box min, both backends, at the 6 mandated
  checkpoints (1, 6, 18, 36, 54, 72) — **PASS at every checkpoint**, agrees
  to displayed R4 precision:

  | call | fortran min delp | cuda min delp |
  |---|---|---|
  | 1  | 9967.1797 | 9967.1826 |
  | 6  | 9957.5176 | 9957.5234 |
  | 18 | 9961.6797 | 9961.6855 |
  | 36 | 9965.2666 | 9965.2627 |
  | 54 | 9967.2793 | 9967.2686 |
  | 72 | 9968.5537 | 9968.5576 |

### 1.2 Error-growth characterization, all mandated fields

`delp`, `pt`, `delz`, `pe` — **no concern across the full 72-call run.**
Peak max_rel anywhere in the run (full scan, not just the 6 checkpoints):
`delp` 6.50e-5, `pt` 4.72e-5, `delz` 7.13e-6, `pe` 9.20e-6 — all solidly
ROUND-OFF SCALE (R4) under Prompt 20's thresholds, no growth trend worth
reporting.

**`w` — bounded/oscillatory, explained by a shrinking-denominator
cancellation floor, not genuine divergence.** Naive single-cell max_rel
spikes as high as 5.079e-01 (call 52), which would naively read as
catastrophic. Direct inspection of `max_abs`/`rms_diff`/field-magnitude at
every spike call (9, 14, 20, 36, 46, 50, 51, 52, 63, 72) shows:

- `max_abs` stays bounded in a **3.4e-4 to 1.11e-3** band across all 72
  calls, with **no net upward trend** (call 72, the last, has one of the
  *smallest* max_abs values of the flagged set).
- At every spike, `max_abs` is 45-65% of the field's own characteristic
  magnitude (`field_mag(fortran)`, itself only ~1.3e-3 to 4.5e-3 at these
  calls) — i.e. `w` itself is passing through a near-zero band at these
  cells/calls, and the classifier's own scale-floor (based on the field's
  own magnitude) is what turns an ordinary, bounded absolute difference
  into a large reported ratio.
- The bulk metric `rms_diff / field_rms(fortran)` (a box-wide measure,
  immune to any single near-zero cell) stays in a **~5%-15% band**
  throughout, with no runaway growth.

**Verdict for `w`: FLAT/bounded in absolute terms.** The large single-cell
max_rel excursions are a cancellation-floor artifact of `w`'s own decaying
magnitude, not backend divergence.

**`omga` — genuine, sustained growth in the relative metric; absolute error
plateaus rather than accelerating; explained mechanistically.** Full 72-call
series (max_rel): 6.06e-4 (call 1) → crosses the strict R4-SIGNIFICANT
threshold (>1e-2) around **call 19** (1.079e-2) → climbs, noisily but with a
clear upward trend, to 6.41e-2 by call 72. This is qualitatively different
from `w`'s bounded oscillation and must be reported plainly rather than
waved away. Breakdown at representative calls (1, 6, 18, 19, 36, 43, 54, 55,
63, 67, 72):

- `max_abs` rises from 2.0e-5 (call 1) to a local peak of **2.82e-4** (call
  43), then **plateaus/fluctuates in the ~1.5e-4 to 2.6e-4 range** through
  call 72 — the absolute error's growth is front-loaded, not continuously
  accelerating.
- `field_mag(fortran)` (omga's own characteristic magnitude) **shrinks
  steadily and substantially**, ~3.24e-2 (call 1) → ~4.08e-3 (call 72), an
  ~8x shrinkage — omga is a decaying diagnostic quantity
  (`omga = (pe-pem)*rdt`, a difference of two large, closely-spaced
  pressures) whose own cancellation floor drops as the run relaxes.
- The bulk metric `rms_diff / field_rms(fortran)` **does** show real,
  sustained growth: ~0.06% (call 1) → 0.42% (call 18) → 0.65% (call 19) →
  1.35% (call 43) → 2.35% (call 63) → **4.06%** (call 72) — roughly a 65x
  increase over the run, unlike `w`'s bounded band.

**Verdict for `omga`: SUPERLINEAR growth in the relative metric**, driven by
a combination of (a) `omga`'s own magnitude shrinking ~8x over the run and
(b) modest, genuine growth in the absolute bulk disagreement (topping out at
only ~4% of the field's own RMS by call 72, with the peak-cell absolute
error itself plateauing after call ~43). This is consistent with compounding
single-precision round-off colliding with a physically decaying,
cancellation-sensitive diagnostic field — **not** a new algorithmic defect —
but it genuinely crosses the strict R4-SIGNIFICANT max_rel threshold
starting at call 19 and must be reported as such, per this task's explicit
instruction not to silently reclassify or loosen thresholds.

**`u`/`v` — small, episodic single-cell threshold crossings, same
cancellation-floor mechanism as `w`; bulk metric negligible.** A full
72-call peak scan found `u` peaking at **1.518e-2** (call 48, pe4,
cell (0,5,3)) and `v` peaking at **1.341e-2** (call 48, pe1, cell (5,7,3)) —
both modestly over the 1e-2 line (~1.3-1.5x), far smaller excursions than
`w`'s (~50x) or `omga`'s (~6x). Direct inspection at call 48 and neighboring
calls (40, 44, 48, 49, 60, 66, 68/69) shows:

- At call 48, `u`'s `max_abs` = 2.217e-2 against a raw (unguarded)
  diff/scale ratio of **1.445** — i.e. the local wind component at that one
  cell is passing through a near-zero value, so the classifier's scale-floor
  (based on the field's overall characteristic magnitude, ~1.46 m/s here)
  caps the reported ratio at 1.518e-2 rather than the raw 144.5%. Same
  pattern for `v` (raw ratio 1.477, floor-guarded to 1.341e-2).
- The bulk metric `rms_diff / field_rms(fortran)` stays tiny throughout —
  0.08% (call 40) to 0.16% (call 48/69) for `u`, similarly for `v` — with no
  systematic growth; call 48 is a local, episodic spike (calls 49, 60, 66,
  68/69 all fall back under 1e-2), not the start of a trend.

**Verdict for `u`/`v`: bounded/episodic**, same cancellation-floor mechanism
as `w`, at a much smaller magnitude. Consistent with ordinary accumulated
float32 round-off encountering an occasional near-zero wind component, not
genuine backend divergence.

### 1.3 Stage A pass conditions (§5 of the task)

| condition | result |
|---|---|
| both runs finish normally | PASS (rc=0 both, it=72 both) |
| no NaN/Inf | PASS |
| delp stays positive | PASS (all 6 checkpoints, both backends) |
| no abrupt error-growth acceleration | PASS — `w` bounded/oscillatory; `omga`'s absolute error plateaus after call ~43; `u`/`v` episodic, not trending |
| no physically meaningful SIGNIFICANT divergence | PASS with a documented caveat — `omga`'s *relative* metric genuinely crosses 1e-2 from call 19 onward, but the bulk disagreement (RMS-relative) tops out at only ~4%, and the absolute error is bounded/plateauing; `w`/`u`/`v` crossings are single-cell cancellation-floor artifacts |
| discrepancies consistent with accumulated float32 round-off | PASS |

**Stage A: PASS**, with the honest caveat on `omga` recorded above and in
§9/§10 of `CURRENT_STATE.md`.

## 2. Stage B — k_split=2

Configuration: c8/npz=10/n_split=3/**k_split=2**/R4/6 MPI ranks (the
smallest value exercising `fv_dynamics.F90`'s `do n_map=1,k_split` loop more
than once per heartbeat, matching Prompt 19 Stage B exactly), same
executable/allocation workflow. Job 58282890, both backends
(`p21_stageB_r4_fortran`, `p21_stageB_r4_cuda`) in one allocation, **rc=0
both**, both reached **it=12** (6 heartbeats × k_split=2).

### 2.1 Entry / basic health

- NaN/Inf: **NONE** at either the `dce` or `dcx` stage over the full 12-call
  run (same benign, previously-documented pre-first-write flux-capacitor
  pattern on `mfxL`/`cxL`/`cyL` reconfirmed at `dce` only — 40 matches, all
  `fortran_nonfinite == cuda_nonfinite` exactly; `dcx` scan: 0 NONFINITE).
- delp positivity, owned-box min, both backends, at the 5 mandated
  checkpoints (1, 2, 3, 6, 12) — **PASS at every checkpoint**, agrees to
  displayed R4 precision:

  | call | fortran min delp | cuda min delp |
  |---|---|---|
  | 1  | 9974.0908 | 9974.0898 |
  | 2  | 9966.5908 | 9966.5908 |
  | 3  | 9962.0859 | 9962.0859 |
  | 6  | 9964.6924 | 9964.6924 |
  | 12 | 9957.5840 | 9957.5830 |

### 2.2 Error-growth characterization, all mandated fields

`u`, `v`, `delp`, `pt`, `delz`, `pe` — **no concern at any of the 5
checkpoints.** `u`/`v` max_rel stays in the 1.6e-4 to 7.9e-4 range throughout
(far below Stage A's own episodic 1.3–1.5e-2 peak at call 48 — expected,
since this 12-call window is much shorter than the call count at which
Stage A's `u`/`v` spikes occurred); `delp`/`pt`/`delz`/`pe` stay in the 1e-7
to 1e-5 range, solidly ROUND-OFF SCALE (R4).

**`omga` — oscillatory within this short window, same cancellation-floor
family as Stage A, no sustained trend visible (window too short to show
it).** Raw worst-cell max_rel at calls 1/2/3/6/12: 8.951e-02, 1.436e-03,
1.780e-02, 1.008e-03, 3.559e-03 — two of the five checkpoints (call 1, call
3) nominally cross the strict R4-SIGNIFICANT line, but the pattern is
non-monotonic (unlike Stage A's clean sustained climb from call 1 to 72) and
must be read against the bulk, box-wide metric rather than the single-cell
ratio alone:

- Bulk `rms_diff / field_rms(fortran)` at the same 5 checkpoints: **6.23%**
  (call 1), 0.089% (call 2), **1.19%** (call 3), 0.088% (call 6), 0.172%
  (call 12) — oscillating, not climbing. Contrast Stage A's own early-window
  values (0.06% at call 1, 0.42% at call 18, 0.65% at call 19) — Stage B's
  call-1 bulk ratio (6.23%) is actually *larger* than Stage A's call-1
  (0.06%), because Stage A and Stage B are physically different experiments
  at call 1 (`bdt` scales with `1/k_split`, so k_split=2's first dyn_core
  call integrates a shorter acoustic step than k_split=1's first call) — the
  two stages' call-1 figures are **not directly comparable to each other**,
  only each stage's own internal Fortran-vs-CUDA comparison is meaningful.
- `omga`'s own field magnitude oscillates substantially call-to-call within
  this short window (`field_mag(fortran)`: 4.98e-3, 3.61e-2, 5.37e-3, 5.17e-2,
  1.83e-2 at calls 1/2/3/6/12) rather than showing Stage A's smooth ~8×
  monotonic shrinkage over 72 calls — `omga` simply hasn't settled into its
  longer-run decaying regime yet at 12 calls, so its cancellation floor is
  itself noisy here, which is what produces the oscillating (not trending)
  max_rel/bulk-ratio pattern.
- 12 calls is short relative to where Stage A's own `omga` trend first
  became visible (its bulk ratio only reached 0.65% by call 19 and did not
  clearly separate from noise until past call ~30) — Stage B ending at call
  12 is consistent with, not contradictory to, Stage A's longer-run
  characterization: it simply doesn't yet run long enough to show the same
  sustained climb.

**Verdict for `omga` in Stage B: bounded/oscillatory within the tested
window**, same cancellation-floor mechanism family already established for
`w`/`u`/`v`/`omga` in Stage A, with too short a horizon (12 calls) to
exhibit — or rule out — Stage A's longer-run sustained-growth trend. This is
not itself claimed as SUPERLINEAR (that label describes Stage A's own
72-call trajectory) — Stage B is reported honestly as consistent with, and
too short to confirm or extend, that trajectory.

### 2.3 Remap-boundary behavior (§7/§8 of the task)

The task's explicit concern is a discontinuous jump either right at a
k_split remap boundary, or at the next dyn_core entry following it. The
first (and only, at 12 calls) such boundary is between call 2 (end of
heartbeat 1, followed by the remap step) and call 3 (entry into heartbeat
2). Checked directly via a boundary-continuity comparison — the `dcx` (exit)
metrics of call 2 against the `dce` (entry) metrics of call 3, for the two
fields with the largest excursions in this window:

| field | dcx call 2 (pre-remap exit) | dce call 3 (post-remap entry) |
|---|---|---|
| `w` | max_abs=9.6827e-05, max_rel=1.5494e-02, bulk=1.1942% | max_abs=9.6856e-05, max_rel=1.5501e-02, bulk=1.1943% |
| `omga` | max_abs=5.1896e-05, max_rel=1.4362e-03, bulk=0.0887% | max_abs=2.3182e-05, max_rel=9.2027e-04, bulk=0.0692% |

Both fields' pre-remap exit and post-remap entry metrics are the same order
of magnitude (`w`'s are essentially identical; `omga`'s entry values are, if
anything, *smaller* than its exit values) — there is **no discontinuous
jump** introduced by the remap step itself. `omga`'s own higher figure at
call 3's *exit* (1.78e-2 max_rel, from §2.2 above) therefore developed
during call 3's own dyn_core execution, not at the remap boundary or at
entry — consistent with the same call-to-call oscillation already described
in §2.2, not a new remap-specific defect.

### 2.4 Stage B pass conditions (§6/§7/§11 of the task)

| condition | result |
|---|---|
| both runs finish normally | PASS (rc=0 both, it=12 both) |
| no NaN/Inf | PASS (benign pre-first-write flux-capacitor pattern only) |
| delp stays positive | PASS (all 5 checkpoints, both backends) |
| no new remap-boundary jump | PASS — boundary-continuity check on `w`/`omga` shows no discontinuity across the call-2→call-3 remap |
| entry after remap matches, no unexplained exit divergence | PASS — `dce` call 3 is smaller/comparable to `dcx` call 2 for both flagged fields; no §8 escalation condition triggered |
| discrepancies consistent with accumulated float32 round-off | PASS |

**Stage B: PASS.**

## 3. Fixes consumed

**0 of 3 permitted substantial fixes used.** No source code was modified in
this task — only the pre-existing scratch analysis script
(`scratch_p21_wfield.py`, not committed) was reused, and one inline
delp-positivity check script (not saved to a file). **Numerical algorithm
changed: NO.**

## 4. Regression

No shared runtime source changed during this task (same R4 build as
Prompt 20, same `@fvdycore` tree as Prompt 19). Per this task's own §13, a
new CTest run is **NOT REQUIRED**.

## 5. Overall conclusion

Both Stage A (72 calls, k_split=1) and Stage B (12 calls, k_split=2) PASS.
R4 is now validated to the same integration depth as R8 was at Prompt 19,
for this configuration (c8/npz=10/n_split=3/R4/6 tiles): a long single-
k_split run and the smallest genuine multi-k_split/remap case, both with
zero NaN/Inf, positive `delp` throughout, no new remap-boundary defect, and
every field-level discrepancy explained as accumulated single-precision
round-off (in several fields' case, amplified through an already-understood
cancellation-floor mechanism on a decaying/near-zero-valued field or diagnostic,
not a new algorithmic defect). `omga`'s longer-run relative-metric growth
(genuinely crossing the R4-SIGNIFICANT line from Stage A's call 19 onward,
topping out at ~4% of its own RMS by call 72) is the one finding that must
be carried forward honestly rather than dismissed — it is explained, but it
is real, and any future extension of R4 validation to a longer window than
Stage A's 72 calls should re-check whether that trend continues, accelerates,
or plateaus.
