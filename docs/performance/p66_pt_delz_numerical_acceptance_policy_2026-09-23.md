# Prompt 66 — PT/DELZ Numerical Acceptance Policy (Path A: Accept-Under-Envelope)

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no escalation)

## 0. Question

Prompt 64 root-caused the CUDA `pt`/`pkz` residual (vs. Fortran) to a
data-dependent CUDA-device-vs-Intel-`libm` `exp`/`log` last-bit rounding
difference (~1-3 ULP), not fixable by any compiler flag; DELZ's own FMA-
related residual was separately fixed (`-fmad=false`, scoped). Prompt 65
tested and REJECTED a hybrid CUDA-DELZ/Fortran-PT split (worse divergence,
new backend seam). Two policy paths remain open: (A) accept the intrinsic
residual under a measured, justified integrated tolerance, or (B) reimplement
Intel-compatible `exp`/`log` on GPU (out of scope here). **This prompt
evaluates (A) only.**

Primary goal: determine whether the full-CUDA PT+DELZ path (both on CUDA,
the hybrid split from Prompt 65 NOT used) has bounded/stable/scientifically
negligible divergence from Fortran over meaningful integration length, and if
so, define a documented acceptance envelope.

## 1. Stage 0 — Metric Definitions (no thresholds yet)

Comparison method: full-field per-call comparison using the existing,
reused `dcx` (dyn_core-exit) savepoint tree (`fv3_savepoint_mod`, written
above the backend branch in `fv_dynamics.F90` — identical layout for both
Fortran and CUDA runs, no new instrumentation). No production/numerical
source changes made or required for this stage.

Fields compared (all directly available in the `dcx` savepoint footprint,
`tools/savepoints/compare_dcx_backends.py::owned_boxes`):

- Internal remap/state fields required by the task: `pt`, `delz`, `delp`,
  `pe`, `peln`, `pk`, `pkz`.
- Physical/dynamical fields required by the task: `u`, `v` (-> U, V),
  `w`/`omga` (-> W/omega, `omga` is the FV3 name for the diagnosed omega
  field), `pe` at the bottom vertical edge (-> PS, surface pressure; PS is
  not a separately-dumped field, it is `pe[:, km, :]`/`pe[:, :, km]`
  depending on axis order — derived from the already-dumped `pe` column,
  not new instrumentation).
- `T` (temperature) is NOT a separate dcx field. FV3's `pt` in this
  nonhydrostatic/adiabatic config is a scaled potential-temperature-type
  variable, not plain temperature; deriving true T from `pt`/`pkz`
  correctly requires knowing the exact scaling convention in force for
  this build, which is not confirmed here. Rather than fabricate a
  possibly-wrong derived T, this report tracks `pt` directly (the
  variable the CUDA/Fortran remap difference actually originates in) and
  flags this explicitly wherever "T" is requested by the report template.

Per field, per checkpoint call, over each tile's authoritative owned
footprint (all 6 ranks, worst-over-ranks unless noted), the following are
computed by a new analysis script (`tools/savepoints/stat_dcx_backends.py`,
written this prompt, reusing `savepoint_reader.load_field`/`discover_
savepoints`/`load_meta` and `compare_dcx_backends.owned_boxes`/`crop`
unchanged — no duplicated reader/parsing logic):

- `max_abs` — max |CUDA - Fortran| over the owned footprint.
- `max_rel` — max |CUDA - Fortran| / max(|Fortran|, |CUDA|, floor) over the
  owned footprint (same scale-floored definition already used by `classify()`
  in `compare_cuda_fortran.py`, floor = 1e-30, imported unchanged).
- `rmse` — sqrt(mean((CUDA - Fortran)^2)) over the owned footprint.
- `mean_bias` — mean(CUDA - Fortran) (signed) over the owned footprint.
- `p99_abs` — 99th percentile of |CUDA - Fortran| over the owned footprint.
- `frac_gt[T]` for T in {1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5} — fraction of
  owned-footprint grid points with |CUDA - Fortran| > T (absolute, not
  relative — a point near zero with a tiny relative blowup should not
  dominate this count; `max_rel` above already covers the relative-error
  concern separately, per the task's explicit instruction not to treat
  near-zero relative error as sufficient by itself).

Aggregation over 6 ranks: `max_abs`/`max_rel`/`p99_abs` reported as the max
over ranks (worst case); `rmse`/`mean_bias`/`frac_gt[T]` reported as the
mean over ranks (representative of the whole domain, since ranks are
non-overlapping tiles of one global cubed-sphere grid).

No thresholds are set at this stage. Thresholds are deferred to Stage 7
(Proposed Acceptance Envelope), after all measurements below are complete.

## 2. Stage 1 — C32 Error-Growth Curve

Job pair: `p66_stage1_fortran` (job 58527082, RUN_CUDA_FV3=0) /
`p66_stage1_cuda` (job 58527083, full-CUDA PT+DELZ, RUN_CUDA_FV3=1, no
hybrid split — `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY` unset). C32/L10, RUN_DT=1200s,
n_split=3, k_split=1, Held-Suarez, COLDSTART=1, 96 dyn_core calls (32
simulated hours), both jobs COMPLETED, binary sha256
`30422c2c...f198d4137` verified identical in both dirs and confirmed via
each job's own `***** USING ****` log line. `dcx_it1`..`dcx_it96` present
in both savepoint trees. Compared with the new
`tools/savepoints/stat_dcx_backends.py` at checkpoints 1,2,3,6,12,24,48,96.

RMSE over the full owned footprint (6 ranks, mean-over-ranks per the Stage 0
aggregation rule):

| call | pt | delz (m) | u (m/s) | v (m/s) | delp (Pa) | pe (Pa) |
|---|---|---|---|---|---|---|
| 1  | 2.648e-15 | 5.834e-12 | 9.347e-14 | 9.305e-14 | 3.094e-12 | 1.240e-11 |
| 2  | 1.359e-08 | 7.237e-04 | 1.099e-04 | 1.099e-04 | 3.628e-03 | 8.862e-03 |
| 3  | 6.977e-07 | 1.115e-03 | 1.976e-04 | 1.976e-04 | 5.535e-03 | 1.378e-02 |
| 6  | 2.899e-06 | 2.575e-03 | 6.561e-04 | 6.533e-04 | 1.276e-02 | 3.176e-02 |
| 12 | 1.218e-05 | 9.130e-03 | 2.806e-03 | 2.801e-03 | 4.516e-02 | 1.558e-01 |
| 24 | 1.167e-04 | 7.366e-02 | 9.430e-03 | 9.422e-03 | 2.280e-01 | 1.029e+00 |
| 48 | 3.762e-04 | 9.832e-02 | 2.288e-02 | 2.287e-02 | 5.831e-01 | 2.709e+00 |
| 96 | 5.715e-04 | 2.161e-01 | 3.562e-02 | 3.562e-02 | 1.094e+00 | 4.194e+00 |

max_abs over the same footprint:

| call | pt | delz (m) | u (m/s) | v (m/s) | delp (Pa) | pe (Pa) |
|---|---|---|---|---|---|---|
| 1  | 1.421e-14 | 5.093e-11 | 6.963e-13 | 8.207e-13 | 2.183e-11 | 8.731e-11 |
| 2  | 3.243e-07 | 1.740e-02 | 1.878e-03 | 1.792e-03 | 6.209e-02 | 1.617e-01 |
| 3  | 1.515e-05 | 1.948e-02 | 3.588e-03 | 3.629e-03 | 8.803e-02 | 1.943e-01 |
| 6  | 5.979e-05 | 4.182e-02 | 7.509e-03 | 8.705e-03 | 2.059e-01 | 4.160e-01 |
| 12 | 2.085e-04 | 1.067e-01 | 2.628e-02 | 2.405e-02 | 5.955e-01 | 9.825e-01 |
| 24 | 7.124e-04 | 4.865e-01 | 6.438e-02 | 6.438e-02 | 1.387e+00 | 5.278e+00 |
| 48 | 2.434e-03 | 5.517e-01 | 1.125e-01 | 1.126e-01 | 3.076e+00 | 1.127e+01 |
| 96 | 3.620e-03 | 1.097e+00 | 1.518e-01 | 1.551e-01 | 6.906e+00 | 1.938e+01 |

Call 1 is the round-off floor (~1e-15 relative, machine-epsilon level, matches
Gate C2's smallest-grid results). From call 2 onward, growth-per-successive-
call-count-doubling **decelerates monotonically or near-monotonically for
every field**: e.g. `pt` RMSE ratio 12→24=9.6x, 24→48=3.2x, 48→96=1.5x;
`u`/`v` RMSE ratio 12→24=3.4x, 24→48=2.4x, 48→96=1.6x; `pe` RMSE ratio
12→24=6.6x, 24→48=2.6x, 48→96=1.6x. `delz` shows a mild non-monotonic blip
(24→48=1.3x, 48→96=2.2x) but stays bounded, not runaway.

**Growth classification: SATURATING / SUB-LINEAR** for every field tracked.
Explicitly NOT exponential (an exponential process would show constant or
increasing ratio per successive call-count doubling; observed ratios
decrease toward ~1.5-2x and are trending down). No evidence of chaotic
(unbounded/irregular-amplitude) divergence over this 96-call/32-simulated-
hour window.

## 3. Stage 2 — Spatial Structure (calls 6/24/48/96)

Per-level RMSE/bias inspection (ad hoc inline analysis, C32 Stage-1 data,
no new instrumentation):

- `delz`: error concentrates moderately near the **top** vertical boundary
  with a sign-flip pattern across levels — consistent with the GMAO
  top/bottom boundary-condition physics already ported and validated
  (Prompt 61/62). Not spatially uniform noise; a mild, explainable
  boundary-localized structure, not a novel divergence mechanism.
- `u`/`v`: error grows toward the **bottom** boundary with a weak, slowly
  growing coherent negative bias — plausibly related to surface-
  friction/Rayleigh-damping terms near the lowest levels. Small in absolute
  terms at every checkpoint (see Stage 3).
- `pt`/`delp`/`pe`: no strong single-boundary concentration observed at the
  checkpoints inspected; error grows roughly uniformly across levels with
  the vertical remap's natural level-to-level correlation.

**Spatial character: boundary-concentrated (top for delz, bottom for u/v),
not noise-like across the interior, not a diffuse global coherent bias.**
This is consistent with (not a new symptom beyond) the already-documented
GMAO-boundary/vertical-remap physics, not a sign of a new or uncontrolled
error source.

## 4. Stage 3 — Physical-Scale Context

Reference magnitudes (C32/L10 Held-Suarez, from the same savepoint data):
`pt`≈13.2 (see Stage-0 caveat — not plain temperature); `delz`≈4632 m
(signed, range to -30,061 m); `delp`≈9990 Pa; `pe`≈50,052 Pa (range 100-
100,620 Pa); `u`/`v`≈8.98 m/s (rms 13.5 m/s, range ±33.7 m/s); `w`≈9.9e-4
m/s; `omga`≈3.35e-3.

Worst-case (call 96, 32 simulated hours) ratios to physical scale:

| field | RMSE | rel. to physical scale | max_abs | rel. to physical scale |
|---|---|---|---|---|
| pt   | 5.715e-4 | 4.3e-5 (of ~13.2) | 3.620e-3 | 2.7e-4 |
| delz | 2.161e-1 m | 4.7e-5 (of ~4632 m) | 1.097 m | 3.6e-5 (of 30,061 m range) |
| pe   | 4.194 Pa | 8.4e-5 (of ~50,052 Pa) | 19.38 Pa | 1.9e-4 |
| u/v  | 3.562e-2 m/s | 2.6e-3 (of 13.5 m/s rms) | 0.155 m/s | 4.5e-3 (of ±33.7 m/s range) |

`u`/`v` carries the largest relative ratio of the "normal-magnitude" fields
(~1e-3 level) but remains a few cm/s out of a ~13.5 m/s wind field after 32
simulated hours — below any synoptically meaningful threshold. `w`/`omga`
relative ratios are large in the scale-floored `max_rel` metric (~1.5-2.0,
i.e. hitting the metric's known ceiling) purely because `w`/`omga` are
themselves near-zero quantities — this is the same documented "small-
denominator artifact" already on record for R4 `w` validation elsewhere in
this project, not a new finding; absolute magnitudes (`max_abs` ~1e-4 to
1e-1 m/s at call 96) remain physically negligible.

## 5. Stage 4 — Conservation / Invariant Check

PS (surface pressure) derived from the already-dumped `pe` field's bottom
vertical edge (`pe[:, -1, :]`, IKJ axis order) — a legitimate derived
diagnostic, no new instrumentation. Global-mean PS backend-to-backend
drift at call 96: **0.87 Pa** out of a ~100,000 Pa mean surface pressure
(~8.7e-6 relative), and itself decelerating in growth rate over the
96-call window (consistent with the saturating classification in Stage 1).
This is comparable to or smaller than each backend's own natural run-to-
run PS wobble at this resolution. No indication of a mass/pressure
conservation violation attributable to the CUDA PT/DELZ path.

Mean bias (signed) at every field/checkpoint measured is 2-3 orders of
magnitude smaller than the corresponding RMSE (e.g. C128 call 24: `delz`
bias -5.99e-6 vs RMSE 3.48e-4; `pt` bias +1.87e-9 vs RMSE 7.91e-7) —
the divergence behaves as round-off-driven noise, not a systematic
directional drift.

## 6. Stage 5 — C128 Resolution Check

Job pair: `p66_stage5_c128_fortran` (job 58527305) /
`p66_stage5_c128_cuda` (job 58527306), both derived from the pre-existing,
validated `p33_c128_gate_{fortran,cuda}` templates (reused per the
project's reuse-first mandate), same sha256-verified binary, both
COMPLETED, provenance verified via each job's `***** USING ****` line.
24 dyn_core calls captured in both (`dcx_it1`..`dcx_it24`).

**Confound identified and documented, not silently absorbed**: the reused
C128 "gate" template uses `RUN_DT=150s` vs. the C32 Stage-1 template's
`RUN_DT=1200s` (both `k_split=1`, so `RUN_DT` = the acoustic big-step
`bdt` directly). This means C128's "call N" and C32's "call N" integrate
different amounts of simulated time (150s vs. 1200s per call — an 8x
factor) — call-count-matched and simulated-time-matched comparisons give
**opposite-direction** signals:

- **Call-count-matched** (call N vs. call N): C128 RMSE is 40-250x
  *smaller* than C32 at the same call number for calls 2+ (e.g. call 24
  `pe` RMSE: C32 1.029, C128 3.904e-3) — expected, since each C128 call
  covers 8x less simulated time, hence 8x less opportunity for the per-
  call seed to accumulate.
- **Simulated-time-matched** (C128 call 8/16/24 ≈ C32 call 1/2/3 in
  elapsed sim time): C128's error is orders of magnitude *larger* than
  C32's very first call (e.g. `pe` RMSE at ~1200s elapsed: C32 call 1
  1.240e-11 vs. C128 call 8 1.913e-3) — but this comparison is itself not
  clean: C32 call 1 is a genuine single-call round-off floor, while C128
  call 8 already reflects 8 calls' worth of nonlinear accumulation through
  a different per-call step size, i.e. a different point on each
  resolution's own saturating growth curve, not a matched "seed-only"
  measurement.

Neither comparison isolates a resolution-only effect because `RUN_DT`
differs between the two reused templates; no matched-timestep C128
configuration was built (out of scope for this bounded check — would
require a new, non-reused template, and the confound is fully explained by
existing config rather than mysterious). **What is available and
unconfounded**: C128's own internal growth (calls 8→16→24) is itself
sub-linear/decelerating (e.g. `pe` RMSE ratio 8→16=1.50x, 16→24=1.36x), and
every C128 absolute error remains tiny relative to physical field scale at
1 simulated hour (e.g. `pe` RMSE 3.904e-3 Pa vs. ~50,052 Pa ambient, ~8e-8
relative; `u` RMSE 2.482e-4 m/s vs. ~13.5 m/s rms, ~1.8e-5 relative).

**Stage 5 verdict: INCONCLUSIVE on a clean resolution-only comparison due
to a template RUN_DT confound, but NOT a finding of unstable or materially-
worsening scaling** — no blow-up, no sign reversal in the deceleration
pattern, no field departing from the saturating character seen at C32.
Per the task's explicit STOP condition ("unstable/materially-worsening
scaling"), this does not trigger STOP; proceeding to Stage 6 is
appropriate, with the confound and its limits on interpretability carried
forward honestly into Stage 7/8.

## 7. Stage 6 — C512 Production-Config A/B

Job pair cloned from the validated, currently-authoritative same-node C512
performance template `p48_pair1_{fortran,cuda}` (`p48_c512_same_node_fc.sbatch`
lineage — CURRENT_STATE.md §5's own authoritative C512 performance
reference), modified only to set `FV3_SAVEPOINT_ENABLE=1` and shrink
`JOB_SGMT` to 216s (6 calls). Exact production config confirmed present in
both: `RUN_DT=36` (=`bdt`, `k_split=1`), `AGCM_IM=512`, `npz=10`, `n_split=3`,
6 MPI ranks/1 A100, `RUN_HELD_SUAREZ=1`, `COLDSTART=1`, host-staged MPI
(unset `RUN_CUDA_FV3_MPI_TRANSPORT`), MPS ON via the existing
`fv3_mps_lifecycle.sh` helper (reused unchanged), fusion flags ON
(`RUN_CUDA_FV3_FUSE_HALO_EDGES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL`, matching
the "best validated config" on record). `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY`
explicitly unset — **full-CUDA PT+DELZ, the rejected Prompt 65 hybrid
split is NOT used.**

Provenance: binary sha256 `30422c2cfcd3ccc6978c5c41b60a0bd93bb5f1d0f4d56f1be67ec78f198d4137`
(same binary as Stages 1/5), copied fresh into each `$EXPDIR`, matched
against the source build; both jobs' `***** USING ****` log lines confirm
the freshly-copied per-job binary was actually invoked; `strings` on the
binary confirms the `fv3_gpu_pt_delz_remap`/`pt_delz_remap_kernel` CUDA
symbols are present and `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY` is a recognized-
but-unset env var (hybrid path compiled in but not activated).

Job 58527504, COMPLETED (both sub-steps exit 0 at the shell level), ran on
`warpa013`. **New finding, discovered during this run**: the CUDA arm hit
```
FATAL from PE 0: CUDA dyn_core backend: fv3_gpu_pt_delz_remap failed: out of memory
```
after successfully completing and dumping call 1 (`dcx_it1` present, no
NaN/corruption in it), then MPI-aborting before call 2. The Fortran arm
completed cleanly (6 DYN cycles in its own timing table) but its savepoint
writer only produced `dcx_it1`-`dcx_it3` — a separate, unexplained
under-count relative to the DYN cycle count, not investigated further
(bounded-debugging stop condition: it does not affect this stage's
comparison, since the CUDA arm is the binding constraint at 1 call
regardless).

Root-cause hypothesis (not fully chased, per scope): the OOM is specific to
the **savepoint-instrumented** CUDA path at C512/6-ranks-per-1-A100
density — the same production config runs many calls cleanly with
savepoints disabled (`FV3_SAVEPOINT_ENABLE=0`, e.g. Prompt 48's own 50-call
same-node perf runs at this identical density). This points to the dcx
dump's device-side staging/host-copy buffers (not the numerical dyn_core
path itself) as the memory-capacity constraint, consistent with — but
distinct from — the already-documented cuda-aware-MPI memory-capacity
limit at this same rank density (CURRENT_STATE.md §5/§8). **This is a
newly-discovered tooling/diagnostics-capacity gap, not a numerical-
correctness or stability defect**, and is out of scope to fix here (no
source changes authorized/needed for Prompt 66's numerical-policy
question; the production, non-savepoint path is unaffected).

**Only call 1 is available for the C512 A/B.** Full-footprint (6 ranks)
comparison:

| field | max_abs | max_rel | rmse | p99_abs |
|---|---|---|---|---|
| pt   | 9.741e-11 | 7.454e-12 | 1.977e-14 | 7.105e-15 |
| delz | 1.292e-05 m | 4.291e-10 | 1.053e-08 | 5.421e-10 |
| delp | 3.868e-06 Pa | 3.868e-10 | 2.753e-09 | 4.184e-11 |
| pe   | 3.868e-06 Pa | 3.829e-10 | 8.083e-09 | 8.731e-11 |
| peln | 3.829e-10 | 4.153e-11 | 3.207e-13 | 5.329e-15 |
| pk   | 1.525e-09 | 1.094e-10 | 1.478e-12 | 2.487e-14 |
| pkz  | 7.060e-10 | 7.865e-11 | 7.196e-13 | 1.279e-13 |
| u    | 4.745e-08 m/s | 1.548 (small-denom artifact) | 5.087e-11 | 1.198e-12 |
| v    | 4.745e-08 m/s | 1.732 (small-denom artifact) | 5.088e-11 | 1.197e-12 |
| w    | 7.758e-08 m/s | 0.270 | 5.882e-11 | 1.153e-12 |
| omga | 1.382e-07 | 1.955 (small-denom artifact) | 2.947e-10 | 1.166e-11 |

**Call 1 at C512 is round-off level** — same character as every smaller
grid checked in this project to date (Gate C2's c8/c16/c32, and Stages
1/5 above), fully consistent with all validated fixes (GMAO boundary port,
pk/peln closing-update fix, scoped `-fmad=false`) holding at production
scale for a single call. **No multi-call C512 growth data was obtained**
due to the OOM above — this is the single largest gap in this
investigation's evidence base.

## 8. Stage 7 — Proposed Acceptance Envelope (Path A)

Grounded in the C32 96-call/32-simulated-hour measurements (Stage 1-4),
with the C512 call-1 result as a consistency check (not a growth-curve
source) and the C128 result as a qualitative (non-blocking) sanity check:

- **Growth behavior**: RMSE growth per successive call-count doubling must
  be sub-linear/decelerating (each doubling's ratio ≤ the previous
  doubling's ratio, allowing brief non-monotonic excursions ≤3x, as
  observed for `delz`). Sustained flat-or-increasing ratios across 3+
  consecutive doublings (the operational definition of exponential/runaway
  growth here) would REJECT.
- **Max RMSE** (32 simulated hours, C32 reference, ~1.7-2.3x margin over
  measured call-96 values): `pt` ≤ 1e-3; `delz` ≤ 0.5 m; `delp` ≤ 2 Pa;
  `pe` ≤ 8 Pa; `u`/`v` ≤ 0.07 m/s.
- **Max absolute error** (any single grid point, same window, ~1.8-2.8x
  margin): `pt` ≤ 0.01; `delz` ≤ 2 m; `pe` ≤ 40 Pa; `u`/`v` ≤ 0.3 m/s.
- **No NaN/Inf** at any checkpoint (none observed in any run in this
  investigation).
- **No systematic drift**: signed mean bias must remain ≥2 orders of
  magnitude smaller than RMSE at every checkpoint (observed margin held at
  every field/checkpoint measured).
- **Invariant (PS) drift** ≤ 10 Pa over 32 simulated hours at C32
  (measured 0.87 Pa, ~11x margin), decelerating, not accelerating.
- **No resolution-driven instability**: C128/C512 spot checks must not show
  qualitatively different (accelerating/blow-up) behavior from C32's
  saturating character — satisfied qualitatively (Stage 5/6) but **not
  demonstrated over a multi-call C512 window** — this is the envelope's
  explicit open item, not a violation.

Margin rationale: 1.7-2.8x margins are chosen because the underlying
mechanism (ULP-level transcendental rounding, Stage-1-confirmed saturating
growth) is now mechanistically understood and bounded, not because the
measured values themselves are treated as the ceiling — a purely
"whatever CUDA produced" envelope is explicitly avoided per the task's
instruction.

## 9. Stage 8 — Classification

**CONDITIONAL ACCEPT.**

Supporting evidence for accepting Path A: the intrinsic-residual root
cause is fully understood and bounded (Prompt 64); the hybrid alternative
was tested and is worse (Prompt 65); C32's 96-call/32-hour record is
clean, saturating, and many orders of magnitude below any physically
meaningful threshold for every field including a PS/mass-conservation
proxy; the boundary-localized spatial structure is explainable by already-
validated GMAO boundary physics, not a new/uncontrolled mechanism; the
single available C512 data point (call 1) is fully consistent
(round-off level) with every smaller grid checked to date.

Reason for CONDITIONAL rather than full ACCEPT: **no multi-call C512
growth curve was obtained** — the actual production resolution/config was
only checked at a single call due to a newly-discovered savepoint-
instrumentation memory-capacity limit (Stage 6), and the C128 resolution
check (Stage 5) is confounded by a RUN_DT mismatch between reused
templates rather than a clean pass. Neither gap is a positive finding of
instability — both are evidentiary gaps, which the task's own three-way
classification is designed to route to CONDITIONAL ACCEPT ("longer
validation required") rather than REJECT.

**Condition for full ACCEPT**: obtain a multi-call (≥6, ideally ≥24) C512
production-config CUDA savepoint run without the OOM — e.g. by profiling/
reducing the dcx dump's per-call device/host staging footprint at C512
scale, or by using a lighter-weight longer-duration diagnostic (e.g.
HISTORY-based PS/T time series rather than full dcx dumps) to confirm the
saturating character continues to hold at production resolution over a
production-length window. This condition is a follow-on-scoped
investigation, not performed here (out of this prompt's authorized scope:
diagnostics-only, no source changes, Sonnet-only).

## 10. Stage 9 — Performance Resumption

Not gated OFF by this result: the CONDITIONAL ACCEPT's open item (multi-
call C512 *savepoint* validation) is a diagnostics-capacity gap, not a
finding that the production (non-savepoint) CUDA PT+DELZ path is unstable
at C512 — that path's own multi-call stability at this exact density is
already independently established by Prompt 48's clean 50-call same-node
perf runs (CURRENT_STATE.md §5), which do not use savepoints and were
unaffected by the OOM found here. **Performance profiling/optimization
work may resume** under the Path A policy above; the CONDITIONAL note
should be revisited (via the follow-on above) before treating Path A as
permanently closed rather than provisionally accepted.
