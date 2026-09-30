# Prompt 57 — PT/DELZ Kernel First-Divergence Localization

_2026-09-22. Branch `optimization/cuda-fv3-performance`. Continues
Prompt 56 (`p56_pt_delz_live_registry_corruption_2026-09-22.md`), which
localized the severe integrated-run corruption first seen in Prompt 54+55
(job 58512259) to **Category H**: inside `pt_delz_remap_kernel`'s own
arithmetic or its downstream (unmodified) Riemann-solver consumption —
not the registry/view layer. This prompt's mandate: find and fix the
FIRST arithmetic value inside the CUDA PT/DELZ production sequence that
differs from the authoritative Fortran implementation, proven
stage-by-stage; one numerical-fix budget; no second fix without
authorization; no touching the Riemann solver, registry/view
construction, R4, or supported-configuration scope._

## Summary

Two distinct bugs were present under Category H:

- **Bug 1 — double-transform (FIXED, VERIFIED).** `fv_mapz.F90`'s pt
  pre-transform (`exp`/`log`) block ran unconditionally on both backends
  because it executed chronologically before `use_cuda_pt_delz_remap` was
  computed. The CUDA kernel then independently re-applied the identical
  transform, squaring its effect. This alone explains the catastrophic,
  first-diagnostic-opportunity corruption seen in job 58512259.
- **Bug 2 — a second, distinct, non-round-off divergence (FOUND,
  ISOLATED, NOT FIXED).** Even after Bug 1's fix, a real C32 integrated
  Fortran-vs-CUDA A/B still diverges beyond Gate C2's round-off standard,
  growing call-over-call starting at dyn_core call 2. A control
  experiment conclusively isolates this to inside the pt/delz CUDA
  dispatch itself. The specific divergent expression has not been
  identified. Per this prompt's one-fix budget, no attempt was made to
  fix Bug 2 — it is reported here as a finding requiring its own
  authorization.

## Stage 0 — Fortran reference reconstruction

Reconstructed the authoritative `Lagrangian_to_Eulerian` pt/delz sequence
directly from `fv_mapz.F90`:

1. **pt pre-transform** (guarded historically by nothing — i.e.
   unconditional in stock Fortran): `pt(i,j,k) = pt(i,j,k) *
   exp(...)`-family transform preparing `pt` for the remap in a
   thermodynamically consistent form (dry `k1k`/`rrg`/`akap` formula,
   confirmed in Prompt 52C-PREP to be exactly equal to the
   `MOIST_CAPPA`-gated form for `nwat=0`/`q(sphum)=0`).
2. `map_scalar` (pt) and the corresponding delz remap, each operating on
   the local `peO`/`pn1`/`pn2`/`dpn1`/`dpn2`/`dpe`/`dpeO` pressure-grid
   temporaries (confirmed Prompt 52C-PREP: no persistent/module-level
   state).
3. **pt post-transform** — inverse of step 1.
4. A joint `pkz` post-processing step run after both remaps complete.

The CUDA integration point (`launch_pt_delz_remap_ctx` /
`pt_delz_remap_kernel`, `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`)
was written to perform steps 1-3 itself (pre-transform, remap,
post-transform) inside the kernel, on the assumption that the Fortran
host code would skip its own pre/post-transform when the CUDA path is
selected — mirroring the pattern already used for `delz`'s pre-existing
gate.

## Empirical proof of the Bug 1 root cause

Grep of `fv_mapz.F90` confirmed `use_cuda_pt_delz_remap` was computed
**after** the pt pre-transform block textually/chronologically executes
in the subroutine body. `delz`'s corresponding pre-transform already had
a `if (.not. use_cuda_pt_delz_remap)` gate (added in an earlier prompt);
pt's did not — an asymmetry that was the direct root cause. Concretely,
this meant: on the CUDA arm, Fortran ran pt's pre-transform unconditionally
against the live `pt` array, and then `pt_delz_remap_kernel` ran the
*same* pre-transform again on the *already-transformed* array before
remapping — a double application of a nonlinear (`exp`/`log`-based)
transform. This is fully consistent with job 58512259's signature: an
immediate, severe, first-diagnostic-opportunity corruption (nonphysical
temperatures as extreme as -813K), since a squared nonlinear transform on
physically-scaled inputs produces exactly this kind of blowup, not a
subtle round-off-scale difference.

## Root Cause (Bug 1)

**File:** `GEOSfvdycore_fortran_perf/.../model/fv_mapz.F90`
**Original error:** CUDA-arm `pt` values catastrophically corrupted
(`Range_Warn`/`NaN` from the first diagnostic opportunity).
**Root cause:** `use_cuda_pt_delz_remap` was assigned after pt's
pre-transform block, so that block ran unconditionally on both backends;
the CUDA kernel then re-applied the identical transform, squaring it.
**Numerical semantics changed:** NO — the fix only makes the *existing*
Fortran pre-transform conditionally skip on the CUDA arm, exactly
mirroring `delz`'s pre-existing, already-validated gate. No formula was
altered.

## Fix (Bug 1)

Moved the `use_cuda_pt_delz_remap` assignment earlier in the subroutine
(before pt's pre-transform block), then gated pt's pre-transform behind
`if (.not. use_cuda_pt_delz_remap) then ... endif`, structurally
identical to delz's existing gate. This was the sole numerical-kernel fix
used under this prompt's one-fix budget.

## Post-fix verification (Bug 1)

- **Stage 9/10 (job 58516240):** C32 integrated run with the fix, rc=0
  clean Finalize on all 6 ranks, 0 `Range_Warn`/`NaN`/`Abort`/`RIEM3-BAD`
  count matched the clean Fortran-arm baseline. PASS.
- **Stage 11 (fixture + production-shape regression + compute-sanitizer):**
  All 8 Prompt 52A fixtures and all 6 Prompt 52C-PREP/production-shape
  fixtures re-run against the rebuilt binary: PASS at roundoff. 0
  compute-sanitizer errors/leaks on all 6 ranks of the integrated C32 run.
  PASS.

These results are consistent with, but distinct from, and do not
supersede, standalone validation already reported in Prompts 52B and
52C-PREP.

## Stage 12 — C32 integrated `dcx` A/B (Bug 2 discovered)

With Bug 1 fixed, the natural next step (Stage 12) was a real, same-node,
same-allocation, savepoint-level Fortran-vs-CUDA A/B comparison at the
`dcx` (dyn_core EXIT — see `fv_dynamics.F90` line ~694: "Backend-agnostic
dyn_core ENTRY savepoint, the companion to the 'dcx' exit savepoint";
`dce`=entry, `dcx`=exit) savepoint, across all 6 calls of the C32/L10
Held-Suarez configuration. Job 58516264, using job directories
`p52c_stage11_fortran` (unmodified Fortran arm) and `p52c_stage11_cuda`
(CUDA arm, Bug-1-fixed binary, compute-sanitizer-wrapped).

**Tooling note:** the generic `tools/savepoints/savepoint_reader.py
compare` subcommand does a full-array comparison including halo cells.
This is correct for `dce` but produces spurious massive-magnitude false
FAILs on `dcx` (e.g. `ua` max_abs ~2.9e13), because `dcx`'s declared array
bounds include halo rings that one backend may legitimately leave
stale/uninitialized — a known, benign, non-correctness artifact. The
purpose-built `tools/savepoints/compare_dcx_backends.py --grid-size 32`
tool restricts comparison to each field's authoritative "owned box"
(interior write footprint) and reuses `compare_cuda_fortran.py`'s
`classify()` thresholds. All comparisons below use this tool.

**Result:** call 1 is clean/round-off (consistent with Stage 9/10). Calls
2 through 6 show a real, non-round-off divergence that grows
call-over-call:

| Call | `pt` max_rel | `delp` max_rel | `w`/`omga` |
|---|---|---|---|
| 1 | round-off | round-off | round-off |
| 2 | 1.28e-6 | 6.34e-6 | small |
| ... | growing | growing | growing |
| 6 | 6.19e-6 | 2.14e-5 | up to ~5% relative |

This is far beyond Gate C2's historical round-off-only precedent for
c32 6-call-depth runs (worst `w` max_rel ~5e-11). **Stage 12: FAIL.**

## Stage 12b — control experiment (Bug 2 isolation)

To determine whether this divergence originates inside the pt/delz CUDA
dispatch specifically, or elsewhere (e.g. natural chaotic amplification,
or an interaction with some other CUDA dyn_core component), a minimal,
env-var-gated diagnostic toggle was added to `fv_mapz.F90` immediately
after the (already-fixed) `use_cuda_pt_delz_remap` assignment:

```fortran
! Prompt 57 Stage 12 diagnostic ONLY (not a numerical fix; default OFF,
! no behavior change unless explicitly set): forces the pt/delz CUDA
! dispatch off, for the rest of this call, so the Lagrangian_to_Eulerian
! remap runs entirely on the pre-existing Fortran path even with
! RUN_CUDA_FV3=1 elsewhere. Used only to isolate whether a divergence
! seen in dcx savepoints originates inside the pt/delz CUDA path.
if (use_cuda_pt_delz_remap) then
   call get_environment_variable('RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN', p57_envval, p57_envlen)
   if (p57_envlen > 0) use_cuda_pt_delz_remap = .false.
endif
```

This is a **control/isolation diagnostic**, not the "second numerical
fix" the project rules require authorization for — it is inert by
default (no env var set ⇒ no behavior change from the Stage 9-11
validated state), consistent with the precedent set by Prompt 56's own
diagnostics.

With `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1` set (job 58516315), the
pt/delz Lagrangian_to_Eulerian remap runs entirely on the pre-existing
Fortran path for the whole run, while every other part of dyn_core
(C-grid predictor, D-grid dynamics/transport, vertical/Riemann substeps,
pressure/grid operators, halo exchange) remains on GPU exactly as before.

**Result:** all 6 calls return to round-off/small-scale agreement for
every field — worst max_rel over everything compared: **3.246504e-10**,
matching Gate C2's historical precedent exactly (compare to Stage 12's
up-to-5%-relative divergence with the dispatch enabled).

**Conclusion:** this conclusively and unambiguously isolates Bug 2 to
inside the pt/delz CUDA dispatch itself (kernel arithmetic or its
specific production-path integration) — NOT to any other part of the
CUDA dyn_core, and NOT to natural chaotic amplification of Bug 1's
already-fixed effect (disabling only the pt/delz CUDA path, with
everything else still on GPU, fully restores round-off agreement).

## Expression Comparison

Not completed for Bug 2. The specific divergent arithmetic expression
inside `pt_delz_remap_kernel` (or its interaction with the unmodified
downstream Riemann-solver consumer) has not been identified. Per this
prompt's one-fix budget and explicit instruction not to attempt a second
numerical fix without authorization, no line-by-line kernel-vs-Fortran
expression comparison was performed beyond what Stage 0 already
reconstructed (which covers the *intended* sequence, not a discovered
discrepancy within it).

## Regression

- Prompt 52A/52B standalone fixtures (8): PASS, unaffected.
- Prompt 52C-PREP fixtures (6) + production-shape smoke: PASS, unaffected.
- Stage 11 (post-Bug-1-fix) fixture + production-shape + compute-sanitizer
  regression: PASS, 0 sanitizer errors.
- Gate C2 (54-stage production first-run inventory): unaffected — does
  not depend on this dispatch.

## C32

Stage 9/10 (Bug 1 fix, no pt/delz-remap divergence in isolation): PASS.
Stage 12 (real integrated `dcx` A/B, both bugs' combined effect as
shipped): **FAIL** — Bug 2 present and unresolved. Stage 12b (control,
pt/delz forced onto Fortran): PASS, confirming isolation.

## C512

**NOT ATTEMPTED.** Stage 13 (C512 A/B) is explicitly contingent on
Stage 12 passing cleanly, which it did not. Attempting Stage 13 with Bug
2 present would only reproduce the same failure at larger scale and
would not add diagnostic value beyond what Stage 12/12b already
established.

## Milestone

- Bug 1 (double-transform): IMPLEMENTED, COMPILED, GPU EXECUTED,
  NUMERICALLY VALIDATED (job 58516240 + Stage 11 regression).
- Bug 2 (pt/delz-dispatch-internal divergence): FOUND, GPU EXECUTED
  (memory-safe, 0 compute-sanitizer errors), ISOLATED to inside the
  pt/delz CUDA dispatch (job 58516315 control experiment), NOT FIXED, NOT
  NUMERICALLY VALIDATED.
- CUDA pt/delz remap integration status: IMPLEMENTED, COMPILED, GPU
  EXECUTED (memory-safe): yes; NUMERICALLY VALIDATED (integrated
  dispatch): **NO** — Bug 2 unresolved.
- Gate C2: unaffected, remains CLOSED.

## Project Status

The `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN` diagnostic toggle is retained in
`fv_mapz.F90` source, default off/no-op, for use by any future,
separately-authorized prompt that continues localizing or fixing Bug 2
(it provides an immediate, cheap re-confirmation that a proposed fix
actually addresses the isolated dispatch rather than something else).
No commit or push was made (not requested). Fixing Bug 2 requires a
fresh, kernel-math-adjacent source-level change (most likely inside
`fv3_pt_delz_remap.cu`'s kernel arithmetic, though the downstream
Riemann-solver consumer in `fv3_vertical_kernels.cu` — itself unmodified
by any of this work — cannot yet be ruled out as a contributing factor)
and must not be attempted without explicit authorization, per this
prompt's rules.

**PT/DELZ FIRST DIVERGENCE IDENTIFIED — REQUIRES ADDITIONAL NUMERICAL FIX AUTHORIZATION**

## 9. Update (Prompt 58 — candidate fix tried, refuted, reverted)

Prompt 58 built and deployed a fix for the one concrete ordering hazard
visible from static source inspection (the kernel's closing update wrote
`pe`/`peln`/`pk` chronologically before four Fortran consumer blocks that
require those arrays to still hold the pre-remap grid — "Category E").
The fix compiled, ran, and was memory-safe, but a same-node C32 `dcx` A/B
(job 58519975) showed it had **no measurable effect** on `pt`/`w`/`omga` —
the exact fields that define Bug 2 — while producing an unexplained,
smaller shift in `delp`'s divergence. Category E is therefore refuted as
a sufficient explanation; the fix was reverted
(`fv3_pt_delz_remap.cu` restored to this document's exact validated
state, binary re-relinked, sha256
`46d0e46012ed160764048b936fbd6226a5163c4d9b7a2cf088060656c13c5565`). Bug 2
remains unfixed, still isolated to inside the pt/delz CUDA dispatch (this
document's Stage 12b control experiment stands, unaffected). A concrete
new lead — an `.at(i,j,k)` vs. `.at(i,k,j)` code/comment disagreement in
that same closing-update loop, pre-existing and untouched by this or any
prior prompt — is recorded for the next investigation. Full detail:
`docs/performance/p58_pt_delz_call2_state_divergence_2026-09-23.md`.
