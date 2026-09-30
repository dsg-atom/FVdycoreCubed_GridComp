# Prompt 75 — Map and Harness fv_tp_2d hord==8 for CUDA Extension

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet
Prior art: `p74_tracer2d_orchestration_cuda_2026-09-24.md` (measured
`fv_tp_2d` at production `hord_tr=8` as 69.7% of `TRACER_2D`, CPU-only,
NO-GO for the orchestration-only port; identified extending
`launch_scalar_transport_cuda` to `hord==8` as the actual prerequisite).

**Scope of this prompt: source mapping + authoritative host-only harness
only. No production CUDA implementation. No new C512 jobs. No nsys/ncu.**

> **Follow-up (Prompt 76, 2026-09-24 — FINAL):** the PROCEED decision
> below was acted on and completed. hord==8 CUDA face-flux kernels
> (`xppm_pass_kernel_hord8`/`yppm_pass_kernel_hord8`), the `pert_ppm_iv1`
> device helper, and dispatch-gate widening were implemented, compiled,
> and GPU-executed. Standalone unit correctness (Stages 0-8): ALL PASS —
> bitwise `pert_ppm_iv1` (5/5 cases), interior/x-edge/y-edge/all-corners
> face-flux fixtures at round-off (max_rel ~2-4e-16), hord==6 regression
> unaffected, `compute-sanitizer --tool memcheck` 0 errors, conservation
> follows analytically from the bitwise fx/fy match. The pre-existing
> Part L in `test_wp_b2_scalar_transport_cuda.cpp` (this file's own
> 11-fixture list, §7 above, run against the real
> `launch_fv_tp_2d_cuda`/`launch_scalar_transport_cuda` production entry
> points) independently confirms this: 10/11 fixtures PASS at round-off,
> and the 11th (`constant`) is a benign, pre-existing anti-degeneracy
> self-check artifact (the harness's `require_variation` guard flags any
> check with zero output variation, which fires on a constant field
> correctly preserved as a constant — not a numerical discrepancy).
> **Integration (Stages 9-15) is BLOCKED BY DESIGN, not by resource
> availability**: no existing Fortran-side selector routes the real
> production `tracer_2d_1L` -> `fv_tp_2d` call through any CUDA path (the
> only match, `tp_core_cuda`/`fv_tp_2d_cuda` in
> `tp-core-driver/test.F90`, is an isolated test driver not linked into
> `fv_tracer2d.F90`/`fv_mapz.F90`/`fv_dynamics.F90`), and creating one
> would constitute the tracer-orchestration wiring changes explicitly
> forbidden by Prompt 76's change budget. See
> `p76_fv_tp_2d_hord8_cuda_implementation_2026-09-24.md` for the full
> record.

## 0. Summary

- hord==6 and hord==8 are two branches of the SAME `xppm`/`yppm`
  subroutines, not separate routines (Stage 0/1).
- Existing CUDA orchestration (`launch_fv_tp_2d_cuda`) is entirely
  hord-independent and directly reusable; only the low-level face-flux
  device functions are hardcoded to hord==6 and need new hord==8
  counterparts, plus a new `pert_ppm` device helper (not ported anywhere
  yet) (Stage 2).
- Existing production halo (`ng=3`, `i_pack(10)`) already satisfies
  hord==8's stencil reach — **no halo widening required** (Stage 3).
- 11-fixture host-only harness: 0/11 NaN/Inf, 11/11 bitwise-repeatable
  across two independent runs (including an ASan-instrumented run), 11/11
  pass a flux-telescoping conservation identity to ~1e-15 (Stage 7-11).
- Estimated CUDA extension: existing orchestration + all non-face-flux
  kernels reusable as-is; new code is bounded to face-flux device
  functions + `pert_ppm` + a dispatch-gate widening — classified
  **MODERATE NEW NUMERICAL BRANCH**, effort **MEDIUM** (Stage 12/13).
- Theoretical performance ceiling (upper bound, current measured
  figures, no new profiling): up to 195.9 ms/call removable from
  `TRACER_2D` (69.7%), up to 8.34% of `FV_DYN_LOOP`, up to 3.07% of
  C512 `DYN_CORE` (Stage 14).
- **Decision: PROCEED TO CUDA HORD8 EXTENSION** (Stage 15) — all four
  gating criteria met (halo sufficient, architecture extends cleanly,
  harness complete, no major algorithm redesign).

## 1. Stage 0 — Authoritative hord==8 dispatch path

Source: `GEOSfvdycore_fortran_perf/.../@fvdycore/model/tp_core.F90`.

```
tracer_2d_1L (fv_tracer2d.F90)
  -> fv_tp_2d(q, ..., iord=hord_tr, jord=hord_tr, ...)   [tp_core.F90:109-243]
       ord_in = (hord==10) ? 8 : hord;  ord_ou = hord     [156-161]
       -> yppm(fy2, q, cry_adv, ord_in, ...)               [mirrors xppm]
       -> xppm(fx,  q_i, crx_adv, ord_ou, ...)             [324-708]
            iord < 7  -> mord==6 branch                    [365-557]
            iord >= 7 -> iord==8 sub-branch                 [567-581]
                 panel-edge correction -> pert_ppm(3,...,1) [640-678]
            shared final flux (iord != 7)                   [697-705]
```

For hord==8, `ord_in == ord_ou == 8` uniformly (the `ord_in=8`
special-case in `fv_tp_2d` only matters for hord==10's Huynh emulation) —
no dual-order handling needed.

`yppm` (712-1147) is the structural mirror of `xppm` (dya/npy in place of
dxa/npx); nothing in the hord==8 arithmetic differs between sweeps.

## 2. Stage 1 — hord==6 vs hord==8 semantic diff

| Aspect | hord==6 (mord==6, 365-557) | hord==8 (iord==8, 567-678) |
|---|---|---|
| Stencil width (interior) | `q1(i-2..i+1)` for `al(i)` | `q1(i-1..i+1)` for `dm(i)`; `dm(i-1),dm(i)` for `al(i)` |
| Reconstruction (`al`) | Fixed quadratic average: `p1*(q1(i-1)+q1(i))+p2*(q1(i-2)+q1(i+1))`, `p1=7/12,p2=-1/12` | Monotonized-slope form: `0.5*(q1(i-1)+q1(i)) + r3*(dm(i-1)-dm(i))`, `r3=1/3` |
| Slope limiter | None separate — `al` itself is unlimited; monotonicity enforced downstream via `smt5` | `dm(i)` is itself a limited (min-mag, sign-preserving) 3-point slope *before* `al` is built |
| `bl`/`br` derivation | `bl=al(i)-q1(i)`, `br=al(i+1)-q1(i)` (unlimited, corrected post hoc) | Sign-limited: `bl=-sign(min(|2dm|,|al(i)-q1(i)|),2dm)`, `br=sign(min(|2dm|,|al(i+1)-q1(i)|),2dm)` |
| Monotonicity test | `smt5 = 3|b0| < |bl-br|` (per-cell boolean gate on the *final flux*, not on `bl`/`br` themselves) | Limiting is built into `bl`/`br` directly (Lin's fast monotone constraint); no separate `smt5`-style gate |
| Boundary treatment | Built directly into `al()` at `is==1`/`(ie+1)==npx` via `c1,c2,c3` (-2/14,11/14,5/14) + a dxa-weighted blend, plus corresponding `smt5` override via `bl*br<0` | Separate block (640-678) using `s11=11/14,s14=4/7,s15=3/14` to build 3 boundary `al/ar` triples, cleaned up via `pert_ppm(3,...,iv=1)` |
| Halo width required | `q1(i-2)` at worst (interior `al`) | `q1(i-2)`/`q1(i+2)` at worst (boundary block's `dm(-1)`/`dm(npx+2)` reach) — same order as hord==6 |
| Temporary arrays | `al`, `bl`, `br`, `b0`, `smt5` (row-local) | `dm`, `al`, `bl`, `br` (row-local); no `b0`/`smt5` equivalent |
| Final flux formula | `donor + fx1` if `smt5(i-1)\|\|smt5(i)` else `donor`, `fx1=(1∓c)*(br/bl - c*b0)` | Unconditional (no `smt5`-style gate): `q1(i-1)+(1-c)*(br(i-1)-c*(bl(i-1)+br(i-1)))` (c>0 case), symmetric for c<0 |
| New helper required | none (fully covered by existing `al_x_at`/`xppm_face_flux`) | `pert_ppm` (iv=1 mode) — **not ported to CUDA anywhere** (`grep -rln pert_ppm cuda/dyn_core/` = empty) |

## 3. Stage 2 — Existing CUDA coverage classification

`cuda/dyn_core/dgrid/wp_b_cuda.{cu,cuh}`:

| Requirement | Existing CUDA state | Classification |
|---|---|---|
| `launch_fv_tp_2d_cuda` orchestration (copy_corners, `q_i`/`q_j` conservative update, flux-weight averaging) | Generic — operates on flux arrays only, no PPM-order dependence (`launch_fv_tp_2d_cuda`, wp_b_cuda.cu:713-785) | **REUSE DIRECTLY** |
| `deln_flux` tail (`launch_deln_flux_transport_cuda`) | hord-independent, already shared | **REUSE DIRECTLY** |
| Dispatch gate (`launch_scalar_transport_cuda`, line 1089) | Hardcoded `mord != 6` -> `FV3_GPU_UNSUPPORTED_CONFIGURATION` | **REUSE WITH PARAMETER CHANGE** (widen gate to accept `mord==8`, branch internally) |
| Interior face-flux (`al_x_at`/`al_y_at`, `xppm_face_flux`/`yppm_face_flux`, wp_b_cuda.cu:190-283) | Direct hardcoded transcription of the mord==6 formula only | **NEW HELPER** (parallel `al_x_at_h8`/`xppm_face_flux_h8`-style device functions) |
| Panel-edge correction | Built into the existing `al_x_at`'s `south_edge`/`north_edge` branches (mord==6 form only) | **NEW BRANCH** inside the new hord==8 helper |
| `pert_ppm(iv=1)` | Not present anywhere in `cuda/dyn_core/` | **NEW HELPER** (small, ~20-line device function, no loop/array — see Stage 6) |

**No architectural mismatch found.** The orchestration/face-flux split
already present in the existing code is exactly the seam needed to add
hord==8: the reusable layer does not know or care which face-flux
formula produced its input fluxes.

## 4. Stage 3 — Halo sufficiency

- Fortran domain setup (`fv_mp_mod.F90:644`): `shalo=nhalo=whalo=ehalo=ng`
  — same halo width for every field regardless of `hord`.
- `i_pack(10)` (tracer halo, `fv_dynamics.F90:773,894,900,904`) uses this
  same default domain halo.
- Existing CUDA `TransportBounds.ng = 3` (wp_b_cuda.cuh:57) already
  matches this convention.
- hord==8's worst-case reach (boundary block's `dm(-1)`, using
  `q1(-2..0)`, and symmetric `dm(npx+2)` using `q1(npx..npx+2)`) is the
  **same order** as hord==6's own worst-case reach (`al(0)` using
  `q1(-2..0)`) — both fit inside the existing `ng=3` halo.

**Conclusion: existing production halo is sufficient. No STOP
condition triggered; no halo widening required.**

## 5. Stage 4/5 — Branch enumeration and operation ordering

Enumerated production-relevant hord==8 branches, all confirmed present
in the harness transcription (`test_fv_tp_2d_hord8_hostharness.cpp`):

1. Interior `dm` (monotonized slope) — sign-preserving, magnitude-limited
   to `min(|0.25*(q1(i+1)-q1(i-1))|, local max-min spread)`.
2. Interior `al` — `r3`-corrected average, distinct from hord==6's fixed
   quadratic weights.
3. Interior `bl`/`br` — Lin's fast monotone constraint (sign-locked to
   `2*dm(i)`).
4. `is==1` boundary triple (`al(0)`/`al(1)`/`al(2)` region) — distinct
   `s11/s14/s15` formula, feeding `pert_ppm(iv=1)`.
5. `(ie+1)==npx` boundary triple — mirror of (4).
6. `pert_ppm(iv=1)` cleanup: `al*ar<0` rescale-toward-monotone, else zero
   both — new logic, no analog in the existing hord==6 CUDA path.
7. Final flux formula (`c>0`/`c<0` symmetric cases) — algebraically
   different closed form from hord==6's `donor + fx1` structure (no
   `smt5`-style conditional donor-cell fallback).

Operation ordering preserved exactly as authoritative (no algebraic
simplification): `dm` for the full row first, then `al` from `dm`, then
`bl`/`br` from `al` and `dm`, then the boundary overwrite (only at
`is==1`/`(ie+1)==npx`), then the final flux from `bl`/`br`/`c`. Matches
`xppm_row(..., iord==8)` in the harness file line-for-line against
tp_core.F90:567-705.

`b1-b5` constants (tp_core.F90 header, appearing twice under what is
likely a preprocessor conditional) were checked and are **not
referenced** anywhere in the iord==8 code path (567-705) — confirmed
irrelevant to this branch, not used in the harness transcription.

## 6. Stage 6 — Memory / scratch estimate

The harness's row-local arrays (`dm`, `al`, `bl`, `br`, sized to the full
halo row) are a **test-harness convenience only** — they exist to let
the harness assert against the authoritative row-oriented Fortran
formulas directly. The existing CUDA kernel does **not** use this
pattern: `al_x_at`/`xppm_face_flux` already compute `al`/`bl`/`br` values
**per-output-cell** from a handful of neighboring `q` loads, using a
small fixed number of registers, not a full-row array. The hord==8
`dm`/`al`/`bl`/`br` values needed for any one output flux depend only on
`q1(i-2..i+2)` (5 values) — directly computable the same way, per-cell,
with a fixed handful of scalar registers (`dm(i-1)`, `dm(i)`, `dm(i+1)`
as needed, `al(i)`, `al(i+1)`). `pert_ppm` operates on exactly 3 scalars
at the panel edge, not a loop or array.

**No fixed per-thread local array is required** — this avoids repeating
Prompt 68's ~3.06 GiB/rank implicit local-memory reservation (that came
from a *fixed-size local array* inside a kernel; hord==8's per-cell
formula needs no such array).

**Memory / local-memory risk: LOW.**

## 7. Stage 7-11 — Host-only harness

File: `cuda/dyn_core/tests/host/test_fv_tp_2d_hord8_hostharness.cpp`
(new, host-only, no CUDA/MPI dependency). Strategy: **C — exact
test-only transcription** of tp_core.F90 lines 324-708 (`xppm`, both the
`iord<7`/mord==6 branch and the `iord>=7`/iord==8 branch) and 1201-1260
(`pert_ppm`, iv=1 mode only), following the same "independent host
reference, treated as authoritative for comparison" methodology already
established in `tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp`. No
production Fortran or CUDA source read, called, or modified.

Compiled and run with the system compiler (no module load needed — no
CUDA/MPI dependency): `g++ -std=c++17 -O2`, confirmed available at
`/usr/bin/g++` (SUSE Linux 7.5.0).

### 11 fixtures (Stage 8)

All defined over a global index range with a 4-cell margin beyond the
existing `ng=3` production halo on both sides of a 20-cell synthetic
panel (`npx=20`), generated by evaluating each fixture's own smooth/
sharp/synthetic function over the *entire* halo+compute range (not
clamped-extension) so every stencil read is well-defined:

| # | Fixture | Courant | is,ie | Purpose |
|---|---|---|---|---|
| 1 | `constant` | 0.35 | 4,15 | trivial baseline |
| 2 | `linear_gradient` | 0.35 | 4,15 | exactness check (PPM must reproduce a line exactly) |
| 3 | `smooth_curved` | 0.35 | 4,15 | sinusoid, smooth curvature |
| 4 | `sharp_front` | 0.35 | 4,15 | step discontinuity |
| 5 | `local_max` | 0.35 | 4,15 | isolated spike |
| 6 | `local_min` | 0.35 | 4,15 | isolated negative-valued dip (legal for a passive tracer pre-positivity) |
| 7 | `near_zero_positive` | 0.2 | 4,15 | O(1e-12) magnitude, near_zero-constant-adjacent regime |
| 8 | `sign_changing` | -0.4 | 4,15 | oscillating sign, negative Courant |
| 9 | `high_courant_valid` | 0.9 | 4,15 | `\|c\|<1` upper edge, production-valid |
| 10 | `boundary_edge_is1_and_npx` | 0.3 | 1,19 | both panel edges (`is==1` AND `(ie+1)==npx`) exercised in one fixture |
| 11 | `c512_like_slice` | 0.42 | 4,15 | smooth large-scale + small high-freq noise, mimicking a real tracer field's shape |

### Results (Stage 9-11)

Run twice independently (once under `-O2`, once under
`-fsanitize=address -O0`) — **bitwise-identical output both times**:

```
fixture,is,ie,nan_h6,nan_h8,repeat_h8_bitwise,mass_in,mass_out_h8,residual_h8,diverges_from_h6
constant,4,15,0,0,1,6.000000e+01,0.000000e+00,0.000e+00,0
linear_gradient,4,15,0,0,1,6.900000e+01,-6.000000e+00,0.000e+00,0
smooth_curved,4,15,0,0,1,3.787342e+01,1.834793e+00,0.000e+00,1
sharp_front,4,15,0,0,1,6.600000e+01,-9.000000e+00,0.000e+00,0
local_max,4,15,0,0,1,4.200000e+01,0.000000e+00,0.000e+00,0
local_min,4,15,0,0,1,1.700000e+01,0.000000e+00,0.000e+00,0
near_zero_positive,4,15,0,0,1,1.314000e-11,-1.200000e-13,0.000e+00,0
sign_changing,4,15,0,0,1,-2.279999e+00,-9.844105e-02,4.441e-16,1
high_courant_valid,4,15,0,0,1,4.620000e+01,-3.600000e+00,0.000e+00,0
boundary_edge_is1_and_npx,1,19,0,0,1,9.500000e+01,-7.600000e+00,8.882e-16,0
c512_like_slice,4,15,0,0,1,3.044573e+03,2.627980e+00,0.000e+00,1
SUMMARY fixtures_ran=11 hord8_nan_free=11 repeatable_bitwise=11 conserve_pass=11 diverges_from_hord6=3
```

- **NaN/Inf: 0/11 for both hord==6 and hord==8** (Stage 11).
- **Repeatability: 11/11 bitwise-identical** across the `-O2` and
  ASan `-O0` runs (Stage 11).
- **Conservation (Stage 10)**: proxy identity — the interior sum of flux
  divergence must telescope exactly to `flux[ie+1]-flux[is]` for a
  closed (non-periodic) slice. All 11 fixtures pass to ~1e-15/1e-16
  (float round-off only; `residual_h8` column). This is the correct
  conservation statement *for `fv_tp_2d`'s flux computation alone* — the
  actual tracer-mass conservation check belongs to `tracer_2d_1L`'s q
  update, already out of scope (Prompt 74 NO-GO on that layer).
- **hord6-vs-hord8 divergence (Stage 9)**: diverges on exactly the 3
  fixtures with genuine curvature/sign-change (`smooth_curved`,
  `sign_changing`, `c512_like_slice`) — **by design**, not a defect.
  Notably, `linear_gradient` shows **zero** divergence — both PPM6 and
  PPM8 exactly reproduce a linear profile, a correct and expected PPM
  property and a strong sanity check that the transcription is faithful.
  `boundary_edge_is1_and_npx` also shows zero divergence, consistent with
  both boundary formulas collapsing to the same result on a pure linear
  ramp.

An AddressSanitizer pass (`-fsanitize=address -O0`) was run specifically
to catch any stencil-indexing bug in the transcription (a real risk
given the number of index-shifted arrays in the authoritative formulas)
— found and fixed two genuine out-of-bounds bugs during harness
development (row-local scratch arrays originally sized too tightly to
`[is1-1,ie3]`-style ranges, which the `is==1`/`(ie+1)==npx` boundary
blocks index outside of; fixed by sizing all row-local scratch to the
full halo row, matching how the authoritative Fortran automatics are
actually dimensioned). Final ASan run: **clean, 0 findings.**

## 8. Stage 12/13 — Port scope and classification

| Component | Existing reusable LOC | New CUDA LOC needed |
|---|---|---|
| `launch_fv_tp_2d_cuda` orchestration | ~90 (713-802) | 0 |
| `copy_corners`, `q_i`/`q_j` kernels, flux-weight kernels | not separately re-measured this prompt; used as-is | 0 |
| `deln_flux` tail | shared | 0 |
| Face-flux device functions (`al_x_at`/`xppm_face_flux` family) | ~94 (190-283, hord==6 reference only, not reusable for hord==8's different formula) | ~120-160 (parallel hord==8 device functions: `dm`, `al`, sign-limited `bl`/`br`, boundary triple, final flux) |
| `pert_ppm` device helper | 0 (unported) | ~20-30 (3-scalar operation, no loop) |
| Dispatch gate widening (`launch_scalar_transport_cuda`) | n/a | ~10-20 |
| New CUDA-vs-host-reference tests (mirroring `test_wp_b2_scalar_transport_cuda.cpp`'s existing pattern, extended with the 11 hord==8 fixtures above) | n/a | ~250-350 |

**Effort: MEDIUM.** Reusable orchestration keeps the change well
bounded, but the face-flux formula, boundary block, and `pert_ppm` are
all genuinely new numerics (not a coefficient/parameter swap).

**Classification: MODERATE NEW NUMERICAL BRANCH** — not CLEAN PARAMETRIC
EXTENSION (the reconstruction, limiter, boundary treatment, and final
flux formulas are all algebraically different from hord==6, not just
re-parameterized), and not MAJOR NEW ALGORITHM or POOR FIT (the existing
orchestration/face-flux architectural seam accommodates it cleanly with
no redesign; the new logic is bounded, well-localized, and has a direct
authoritative source to transcribe against).

## 9. Stage 14 — Performance ceiling (no new profiling; existing measured figures only)

Baseline figures (all previously measured, from `p74_...md` §2 and
`CURRENT_STATE.md` §5):

- `fv_tp_2d` (hord==8, CPU-only, current): **195.9 ms/call**, 69.7% of
  `TRACER_2D` (280.87 ms/call).
- `FV_DYN_LOOP` (context, same job): 2347.80 ms/call.
- C512 `DYN_CORE` (same-node authoritative, Fortran/CUDA comparable,
  §5 table): **6374.02 ms/call** (Fortran), 6373.86 ms/call (CUDA).

Theoretical ceiling (upper bound — assumes the *entire* CPU
`fv_tp_2d` cost is removed from the host critical path with zero
residual GPU/orchestration/transfer cost; a real implementation will
retain some nonzero cost, so this is a ceiling, not a projection):

| Denominator | Value | Max removable | Max reduction |
|---|---|---|---|
| `TRACER_2D` | 280.87 ms/call | 195.9 ms/call | **69.7%** |
| `FV_DYN_LOOP` | 2347.80 ms/call | 195.9 ms/call | **8.34%** |
| C512 `DYN_CORE` (TOTAL analog) | 6374.02 ms/call | 195.9 ms/call | **3.07%** |

## 10. Stage 15 — Implementation decision

| Gating criterion | Status |
|---|---|
| Halo requirements already satisfied | YES (§4) |
| Existing CUDA architecture represents hord==8 cleanly | YES — orchestration reusable, face-flux/pert_ppm additive (§3) |
| Authoritative harness complete | YES — 11/11 fixtures, 0 NaN/Inf, bitwise-repeatable, conservation-identity PASS (§7) |
| No major algorithm redesign required | YES — classified MODERATE NEW NUMERICAL BRANCH, not MAJOR NEW ALGORITHM (§8) |

**All four criteria met. Decision: PROCEED TO CUDA HORD8 EXTENSION.**

This prompt performed no CUDA implementation — the above is the
go-ahead determination only. The actual extension (new hord==8 device
functions, `pert_ppm` helper, dispatch-gate widening, and GPU-vs-host
correctness tests reusing this prompt's harness) is future work.

## 11. Files created/modified this prompt

- New: `cuda/dyn_core/tests/host/test_fv_tp_2d_hord8_hostharness.cpp`
  (host-only, no production source touched).
- New: this file.
- Updated: `p74_tracer2d_orchestration_cuda_2026-09-24.md` §5 (forward
  reference to this file).
- Updated: `cuda_fv3_optimization_roadmap_2026-09-21.md`.
- Updated: `docs/agent_handoff/CURRENT_STATE.md` §5/§9.

No production CUDA or Fortran source modified. No commit made.

---

**FV_TP_2D HORD8 HARNESS READY — PROCEED TO CUDA EXTENSION**

**Superseded by Prompt 76's final result:**
**FV_TP_2D HORD8 CUDA VALIDATION BLOCKED — no production dispatch
selector exists; creating one is out of this task's change budget.**
See `p76_fv_tp_2d_hord8_cuda_implementation_2026-09-24.md`.
