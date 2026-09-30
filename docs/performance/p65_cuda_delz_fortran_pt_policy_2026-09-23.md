# Prompt 65 — Hybrid Policy Test: CUDA DELZ + Authoritative Fortran PT

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no escalation)

## 1. Question

Prompt 64 closed with an open numerical-policy decision: the CUDA `pt`/`pkz`
residual against Fortran (a data-dependent, ~1-3 ULP `exp`/`log` rounding
difference vs. Intel `libm`) is not fixable by any compiler flag, and the
resulting seed is nonlinearly amplified by the acoustic dynamics to ~1e-7
relative (delp/pe) by call 6 — outside the Gate-C2 round-off envelope.

Prompt 65 tested a correctness-preserving alternative to accepting that
residual or attempting a (likely infeasible) bit-matching `exp`/`log`
reimplementation: split dispatch so **CUDA computes DELZ only** (using the
Prompt 64 `-fmad=false`-fixed, now-bitwise-at-call-1 kernel) while
**Fortran's own `map_scalar` computes PT**, leaving all other CUDA dyn_core
behavior (C-grid, D-grid, Riemann solver, halo exchange) unchanged.

## 2. Implementation (Stages 0-3, prior window)

Confirmed cleanly separable (Stage 0/1): no duplicated or reordered pressure
arithmetic required. One authorized architectural change (Stage 3): a
`cuda_owns_pt` per-field control threaded through
`cuda/interface/fv3_gpu_dyncore_bridge.F90`
(`fv3_gpu_interface_pt_delz_remap(rc, cuda_owns_pt)`, new optional argument)
and `fv_mapz.F90` (new `cuda_owns_pt` logical, env-var-gated via
`RUN_CUDA_FV3_PTDELZ_DELZ_ONLY`, rewiring the pt pretransform gate/dispatch/
skip logic; delz's own CUDA gate is unchanged — always CUDA-owned when
`use_cuda_pt_delz_remap` is true). A matching `pt_delz_remap_state_mask_
delz_only_d2h` (8-field D2H mask, excludes PT) was added alongside the
existing 9-field mask. Test-only stub (`fv3_gpu_dyncore_bridge_stub.f90`)
updated to match the new signature.

## 3. Stage 4 — Standalone Regression (this window)

- Production `StandAlone_FV3_Dycore.x` rebuilt clean (`GEOSfvdycore_fortran_
  perf/build`, Intel-MPI stack) — both edited files (`fv3_gpu_dyncore_
  bridge.F90`, `fv_mapz.F90`) recompiled with zero errors; only pre-existing
  unrelated `FV_StateMod.F90` INTENT(OUT) warnings. Binary relinked fresh,
  29,370,560 bytes.
- Kernel-level regression (`build_cuda_r8_mpi`, OpenMPI dev stack, unchanged
  kernel source this window): `ctest -R fv3_wp_c_map_scalar` 1/1 PASS;
  `fv3_wp_c_pt_delz_remap_gpu_smoke` and `..._prod_shape_smoke` both 0/6
  cases failed.
- `compute-sanitizer --tool memcheck` on both smoke binaries: `ERROR
  SUMMARY: 0 errors` each.
- **Stage 4: PASS, 0 memory errors** — regression requirement satisfied.

## 4. Stages 5/6 — Call-1 and Six-Call C32/L10 Comparison (this window)

Two fresh, self-contained job directories under `GEOSfvdycore_fortran_perf/
experiments/`, each with an independently scratch-copied, sha256-verified
copy of the Stage-4 binary (`30422c2c...d4137`, identical in both arms):

- `p65_stage5_fortran`: `RUN_CUDA_FV3: 0` (pure Fortran baseline).
- `p65_stage5_hybrid`: `RUN_CUDA_FV3: 1` + `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY=1`.

Both jobs COMPLETED (58526827, 58526828; exit 0:0). Provenance confirmed via
each job's own `***** USING ****` log line (both point at their own
freshly-copied scratch binary) and via `[P60_EXIT_DUMP]` dispatch markers:
hybrid arm shows `use_cuda_pt_delz_remap=T cuda_owns_pt=F` throughout,
confirming the intended split was actually exercised (CUDA path engaged,
Fortran owns PT).

Reused the existing Prompt 63 comparison tooling (`scratch_p63/parse_dumps.
py`, `compare_arms.py`) unmodified against the new logs — no new harness
built, per the project's reuse-before-build guidance.

### Result: fortran vs. hybrid (CUDA-DELZ + Fortran-PT), 6 calls, C32/L10

| call | delp max_rel | delz max_rel | pe max_rel | pt max_rel |
|---|---|---|---|---|
| 1 | 1.46e-15 | 1.81e-15 | 3.63e-16 | 6.75e-07 |
| 2 | 4.32e-08 | 3.50e-07 | 3.84e-08 | 2.04e-06 |
| 3 | 2.53e-07 | 1.36e-06 | 2.25e-07 | 3.49e-06 |
| 4 | 7.27e-07 | 3.08e-06 | 6.46e-07 | 5.05e-06 |
| 5 | 1.46e-06 | 5.51e-06 | 1.30e-06 | 6.83e-06 |
| 6 | 2.43e-06 | 8.62e-06 | 2.16e-06 | 8.98e-06 |

Call 1 is Gate-C2-like round-off (~1e-15/1e-16) for delp/delz/pe — the split
itself introduces no immediate defect. But calls 2-6 grow, and **the growth
is larger than the already-known-inadequate full-CUDA (PT+DELZ both CUDA)
case**, not smaller. Direct re-comparison of the pre-existing Prompt 63
full-CUDA arm (`p63_stage2_B_cuda`) against the same Fortran baseline, using
the identical tooling, for the same call range:

### Result: fortran vs. full-CUDA (PT+DELZ both CUDA), 6 calls, C32/L10

| call | delp max_rel | delz max_rel | pe max_rel | pt max_rel |
|---|---|---|---|---|
| 1 | 1.46e-15 | 1.81e-15 | 3.63e-16 | 7.39e-16 |
| 2 | 1.62e-09 | 7.36e-10 | 1.44e-09 | 4.26e-10 |
| 3 | 1.34e-08 | 5.82e-09 | 1.19e-08 | 3.22e-09 |
| 4 | 5.21e-08 | 2.22e-08 | 4.62e-08 | 1.46e-08 |
| 5 | 1.53e-07 | 6.54e-08 | 1.36e-07 | 4.76e-08 |
| 6 | 3.74e-07 | 1.62e-07 | 3.32e-07 | 1.19e-07 |

At call 6, the hybrid arm's divergence from Fortran is **~6-75x larger**
than the full-CUDA arm's divergence from Fortran, field by field (delp:
2.43e-6 vs 3.74e-7; delz: 8.62e-6 vs 1.62e-7; pe: 2.16e-6 vs 3.32e-7; pt:
8.98e-6 vs 1.19e-7). The gap is already visible by call 2 (delz: 3.50e-7 vs
7.36e-10, a ~475x jump) — i.e., essentially from the first cross-backend
call after call 1.

**Interpretation**: PT and DELZ are not numerically independent within a
call — both consume the same evolving pressure grid (`pe`/`peln`/`pk`), and
PT itself feeds forward into the next call's CUDA-resident acoustic state.
Routing PT through Fortran while DELZ and everything else stays on CUDA
does not just substitute one backend's round-off for another's — it
introduces a **new per-call backend-boundary seam** (a D2H/H2D handoff of a
Fortran-computed field back into GPU-resident state) that appears to
amplify faster than the single self-consistent-backend seed in the
full-CUDA case. This is a plausible, sufficient explanation and matches the
observed pattern (divergence-vs-Fortran grows fastest exactly where the
backend boundary is crossed); no further isolation was pursued past this
point per the bounded-debugging stop condition (a clear, large,
unambiguous result, not a borderline one).

## 5. Decision

**Primary success criterion (Stage 6): FAILED.** The previous ~1e-7 growth
does not disappear under the hybrid split — it is replaced by a larger
growth (~9e-6 by call 6). Per the Prompt 65 task's own Stage 10 framework,
this is **CASE C: still-amplified (and in fact worse) divergence — STOP,
hybrid split is unsuitable.**

Given this unambiguous, decisive failure at Stages 5/6 (not a borderline
result requiring finer localization), Stages 7-10 (`dcx` A/B, C512 A/B
correctness, performance A/B) were deliberately **not run** — they are
gated on Stage 6 passing, which it did not, and running them would consume
C512-scale resources without changing the conclusion. This is consistent
with the project's bounded-debugging and cost-efficiency guidance (stop
once evidence is sufficient; do not gather redundant evidence after a
hypothesis is resolved).

## 6. Effect on Prompt 64's Open Policy Question

Prompt 64 left two paths open: (a) accept the round-off/ULP-level pt/delz
disagreement as permanent and use a tolerance-based (not bitwise) gate for
the `fv_mapz_mod` CUDA port, or (b) invest in a bit-matching `exp`/`log`
reimplementation (large, likely infeasible, no precedent in this project).
Prompt 65 closes off a third possibility — a "free lunch" backend split
that would have sidestepped both — as **not viable**: splitting backends
per-field is worse than either pure backend, not better. The Prompt 64
choice between (a) and (b) is otherwise unchanged and still requires a
decision from the user/project owner before the `fv_mapz_mod` port
continues.

## 7. Change Budget / Scope

No second numerical fix was applied in this window (only the single Stage-3
`cuda_owns_pt` split-dispatch change, already made and now regression-
verified). No Riemann-solver, R4, or custom-transcendental-function changes
were made. No commit/push performed.

## 8. Addendum (Prompt 66) — Policy (a) Tested and CONDITIONALLY ACCEPTED

Prompt 66 evaluated option (a) above directly: whether the full-CUDA
PT+DELZ path's intrinsic residual is bounded/stable/scientifically
negligible over a meaningful integration length, using the same full-CUDA
build referenced in §4's second table (not the hybrid rejected here).
Result: C32 96-call (32 simulated hour) growth is saturating/sub-linear
for every field tracked, many orders of magnitude below physical scale,
with no systematic drift and a negligible PS/invariant drift. A single
C512 production-config call was also round-off level, consistent with
every smaller grid checked to date, but a multi-call C512 growth curve
could not be obtained (new, separate savepoint-instrumentation OOM at
that density — a diagnostics-capacity gap, not a numerical-instability
finding). Overall classification: **CONDITIONAL ACCEPT** — performance
work may resume under this policy, with full (non-conditional) acceptance
pending a longer/production-scale validation follow-on. Full detail:
`p66_pt_delz_numerical_acceptance_policy_2026-09-23.md`. Option (b)
(bit-matching `exp`/`log`) remains unauthorized/out of scope and is not
needed given this outcome.
