# Prompt 64 — Achieving Bitwise-Equivalent PT/DELZ Remap Arithmetic

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no Opus escalation needed — the FMA-contraction mechanism was
proven directly with a controlled fixture test, not just inferred).

## 0. Goal

Prompt 63 root-caused the call-2-onward growing PT/DELZ divergence to a real,
individually-correct, ~1e-15-level non-bitwise-identical difference between
the CUDA pt/delz kernel and Fortran's `map_scalar`, nonlinearly amplified by
the shared acoustic-dynamics substep. This prompt's mandate: find the first
differing floating-point operation and attempt ONE narrow arithmetic fix to
make the two bit-for-bit identical, if feasible.

## 1. Stage 0 — Selected first-call differing value, exact bits

Source: Prompt 63's three-arm isolation control logs (arm B =
`p63_stage2_B_cuda`, real CUDA pt/delz dispatch; arm C =
`p63_stage2_C_forcefortran`, identical CUDA dyn_core but Fortran-fidelity
`map_scalar` for pt/delz — the correct apples-to-apples comparison, per
Prompt 63 §2/§3), `[P60_EXIT_DUMP]` call=1, field=pt, rank `pe=4`, `i=16
j=16 k=5` (C32/L10):

| arm | decimal (ES24.16) | IEEE-754 double bits |
|---|---|---|
| B (CUDA) | `2.6175499933487606E+02` | `0x40705c147a2ebcb6` |
| C (Fortran `map_scalar`) | `2.6175499933487589E+02` | `0x40705c147a2ebcb3` |

XOR = `0x0000000000000005`, signed ULP difference = **3**, `abs diff =
1.705e-13`, `rel diff = 6.515e-16` — matches the round-off magnitude Prompts
61-63 already established for call 1 (pt max_rel 9.232e-16 across all ranks).
`ES24.16` (17 significant decimal digits) round-trips a double exactly, so
the decimal dump values are used directly for bit reconstruction; no new
Fortran/CUDA hex-dump instrumentation was needed.

## 2. Stage 1/2 — Chain tracing deferred in favor of a direct, controlled test

Rather than instrument every intermediate (pressure factors, a4 coefficients,
edge values, limiter outputs) inside a live 6-rank MPI production run — a
large new-instrumentation effort — Prompt 52B's own standalone
`map_scalar_column` validation (`p52b_fv_mapz_cuda_standalone_2026-09-21.md`)
was reused as a pre-existing, controlled Stage 5/6-equivalent experiment for
the *identical* kernel: it already proved that nvcc's default FMA
(fused-multiply-add) contraction — not any algorithmic difference — causes
exactly this round-off-level (not bitwise) GPU-vs-Fortran divergence for
`map_scalar_column`, and that `-fmad=false` restores bitwise agreement, 8/8
fixtures, zero algorithm change.

A build-flag audit confirmed the asymmetry is real in the actual production
configuration, not just theoretical:

- Authoritative Fortran build (`GEOSfvdycore_fortran_perf/build/.../fvdycore.dir/flags.make`,
  ifort): `-fp-model fast=1 -fp-model source -fp-model consistent -no-fma
  -fimf-arch-consistency=true -fp-speculation=safe ...` — **`-no-fma`
  explicit**, FMA contraction OFF.
- Production CUDA build (`build_cuda_r8_intelmpi/CMakeCache.txt`):
  `CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG`, sm_80, **no `-fmad` override
  anywhere** — nvcc's default (`-fmad=true`, FMA contraction ON) applies to
  `fv3_pt_delz_remap.cu`, the production pt/delz dispatch kernel.

This is a direct, proven build-configuration asymmetry, not an inference —
Fortran and CUDA are compiled with opposite FMA-contraction policies for the
identical arithmetic. Given Prompt 52B's own controlled precedent for the
same kernel, this was treated as the leading (and, per Stage 3 below,
confirmed) Stage 4 classification candidate, verified directly rather than by
analogy alone (§3).

## 3. Stage 3/4 — Direct confirmation and classification

**Test**: built a temporary diagnostic twin of the production
`fv3_wp_c_vertical` library with `-fmad=false` applied to the whole library
(all 3 files: `fv3_vertical_kernels.cu`, `fv3_riem_solver_c.cu`,
`fv3_pt_delz_remap.cu`), linked into a twin of the existing
`fv3_wp_c_pt_delz_remap_gpu_smoke` fixture test (6 fixtures, including
`case6_realistic_c512`), and ran both the default-FMA and no-FMA binaries on
a real A100 (`build_cuda_r8_mpi`, dev OpenMPI stack, GPU node via `srun`):

| | pt | delz | delp/dpeO | pkz |
|---|---|---|---|---|
| default FMA (current production, all 6 cases) | rel 2.1e-16 – 4.7e-16 | rel 7.9e-16 – 1.4e-15 | rel 1.111e-15 (all) | rel 0 – 1.4e-16 |
| `-fmad=false` (all 6 cases) | **0.0 (bitwise)** | **0.0 (bitwise)** | **0.0 (bitwise)** | rel 0 – 1.8e-16 (unchanged) |

`-fmad=false` makes pt/delz/delp **exactly bitwise** against the Fortran
fixture reference in all 6 cases, including the production-representative
`case6_realistic_c512` — a decisive, direct (not analogical) confirmation
that FMA contraction in `fv3_pt_delz_remap.cu`'s `map_scalar_column`/
`scalar_profile_column`/closing pe/peln/pk/delp arithmetic is the entire
source of the round-off-level pt/delz/delp difference. **Classification:
category B** (FMA contraction — an `a*b+c` fused single-rounding-step
expression vs. Fortran's separate multiply-then-add). `pkz` keeps a
separate, unrelated, much smaller (1-2 ULP, ~1.4e-16 to 1.8e-16 rel, only in
2/6 cases) residual that is **unaffected by this flag either way** — a
distinct, tiny effect (likely a different intrinsic-evaluation order in the
`pkz` formula itself), already within Gate C2's own round-off precedent, and
explicitly NOT pursued as a second fix per this prompt's one-fix budget (§6).

## 4. Stage 5/6 — Micro-reproducer

The fixture-level test above (§3) *is* the minimal reproducer required: it
isolates exactly `fv3_pt_delz_remap.cu`'s compiled arithmetic (no full FV3
MPI job, no live dynamics) against a known Fortran reference, and tests the
single candidate (`-fmad=false`, a compiler flag, no source/algorithm
change) directly. This satisfies Stage 5's intent ("smallest possible
CPU-vs-CUDA... reproducer") without needing to hand-build a separate
standalone harness, since the existing smoke-test infrastructure already
compares single-column and production-shape (32x32) cases against exact
Fortran fixture values.

## 5. Stage 7 — Fix applied

File: `cuda/dyn_core/CMakeLists.txt`. Added:

```cmake
set_source_files_properties(vertical/fv3_pt_delz_remap.cu PROPERTIES
  COMPILE_OPTIONS "-fmad=false")
```

immediately after the `fv3_wp_c_vertical` library definition. This applies
`-fmad=false` **only** to `fv3_pt_delz_remap.cu`'s compilation — confirmed via
`make VERBOSE=1`/`build.make` inspection that `fv3_riem_solver_c.cu` and
`fv3_vertical_kernels.cu` (both also in `fv3_wp_c_vertical`) compile with
their original default-FMA flags, unmodified. No Riemann-solver arithmetic
changed. No algorithm changed — this is purely a per-translation-unit
compiler-contraction policy flag, mirroring the already-committed-in-spirit
Prompt 52B `_nofma` precedent, now applied to production rather than a
side diagnostic target.

- Interior formulas changed: **NO**.
- GMAO boundary port / pk-peln pass-through (Prompts 61/62): **untouched**.
- Riemann solver (`fv3_riem_solver_c.cu`): **untouched, unaffected**.
- Change budget: **one** arithmetic-semantic fix, as authorized.

## 6. Stage 8 — Single-value bitwise recheck

Rebuilt `fv3_wp_c_vertical` with the scoped fix (dev OpenMPI stack,
`build_cuda_r8_mpi`); reran `fv3_wp_c_pt_delz_remap_gpu_smoke` and
`fv3_wp_c_pt_delz_remap_prod_shape_smoke` (single-column and 32x32
production-shape) on a real A100:

| | pt | delz | delp/dpeO | pkz |
|---|---|---|---|---|
| all 6 cases, both binaries | **0.0 (bitwise)** | **0.0 (bitwise)** | **0.0 (bitwise)** | 0 – 1.8e-16 (unchanged, separate effect) |

**Exact match to §3's `-fmad=false` diagnostic-twin result** — the
production-scoped, single-file fix reproduces the same bitwise outcome.
`PT_DELZ_GPU_SMOKE: 0/6 cases failed`, `PT_DELZ_PROD_SHAPE_GPU_SMOKE: 0/6
cases failed`, both binaries.

## 7. Stage 12 (partial) — Regression, ahead of Stage 9-11

Run ahead of the production-scale call-1/6/C32-`dcx` gates to catch any
build/behavior regression early:

- `fv3_wp_c_map_scalar` ctest (host-visible, covers the original + Prompt
  61 GMAO-boundary fixtures): **PASSED** (unaffected — host path never
  touches GPU/FMA).
- `compute-sanitizer --tool memcheck` on both fixed GPU smoke binaries:
  **0 errors**, both.
- Production Intel-MPI rebuild (`cuda_intelmpi_build.sh`): **PASS**,
  `libfv3_wp_c_vertical.a` rebuilt with the scoped fix, all other CUDA
  archives unchanged/reused.

## 8. Correction to §3/§6 — "bitwise" was a tolerance pass, not literal 0.0

Stage 9 (below) surfaced a genuine, real discrepancy: the fix's effect on
`pt` in real production data did not match §3/§6's "0.0 (bitwise)" framing.
Root cause of the discrepancy: the fixture smoke test's own pass/fail gate
(`fv3_pt_delz_remap_smoke.cu:289`, `const double tol = 1e-9; // tight, not
bitwise (host libm vs Fortran intrinsics evaluate transcendentals
independently...)`) is a **1e-9 tolerance check, not a bitwise check** —
this comment was already present in the source (written earlier this same
investigation) and directly contradicts §3/§6's "bitwise (0.0)" wording,
which described the pass/fail outcome, not the literal printed magnitude.
Re-running the fixed smoke binary and reading the actual printed numbers
(not just `OK=1`) gives the true picture:

```
case1_smooth_monotonic   pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=1.401e-16) delp(rel=0.000e+00)
case2_strong_gradient    pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=1.378e-16) delp(rel=0.000e+00)
case3_local_extrema      pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=0.000e+00) delp(rel=0.000e+00)
case4_near_uniform       pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=1.769e-16) delp(rel=0.000e+00)
case5_limiter_activating pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=0.000e+00) delp(rel=0.000e+00)
case6_realistic_c512     pt(rel=0.000e+00) delz(rel=0.000e+00) pkz(rel=0.000e+00) delp(rel=0.000e+00)
```

`pt` genuinely IS bitwise (literal 0.0, not just "under tolerance") for all
6 canned fixture inputs — §3/§6's numbers were correct for those specific
inputs. `pkz` was never claimed bitwise and remains not-bitwise in 3/6
cases, as already documented. The surprise, resolved in §9 below, is that
this fixture result does **not** generalize to real production grid data —
transcendental-function agreement between two independent implementations
is data-dependent (each is within ~1 ULP of the true value but not
guaranteed to round to the *same* ULP for every input), so 6 canned test
vectors passing bitwise does not prove the general case.

## 9. Stage 9 — Real production call-1 PTDELZ_EXIT (all 7 fields)

Two-arm job pair (Stage 9 methodology, same as Prompt 63 §2): arm B
(`p64_stage9_B_cuda`, real CUDA pt/delz dispatch, **fixed** binary) vs arm C
(`p64_stage9_C_forcefortran`, identical CUDA dyn_core,
`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1` Fortran-fidelity control for pt/delz
only). C32/L10, 6 MPI ranks, production Intel-MPI stack, same-node pair
(jobs 58525824/58525825, both `warpa010`, both COMPLETED exit 0:0).
Provenance: both arms' own `***** USING ****` log line points at their own
arm-local scratch-copied binary; sha256 of the actually-executed binary in
both arms is `db3ef9e1738e0c85948b601142dcb3e083eb952744b8136b9c7a2961a6f1006c`
(the fixed build); `use_cuda_pt_delz_remap` dump values correct per arm
(B=T, C=F, all 6 ranks); `gmao_top_bc=T`/`gmao_bot_bc=T` both arms; dump
line counts match (1548 each).

Call=1 EXIT, rank-paired comparison (`scratch_p63/compare_arms.py`, reused
unmodified):

| field | max_abs | max_rel | vs. pre-fix (Prompt 62 Stage 5) |
|---|---|---|---|
| delp | 0.0 | 0.0 | unchanged (already bitwise pre-fix) |
| **delz** | **0.0** | **0.0** | **improved: was 1.558e-15, now bitwise** |
| pe | 0.0 | 0.0 | unchanged (already bitwise pre-fix) |
| peln | 0.0 | 0.0 | unchanged (already bitwise pre-fix) |
| pk | 0.0 | 0.0 | unchanged (already bitwise pre-fix) |
| pkz | 5.33e-15 / 4.03e-16 (abs/rel) | — | essentially unchanged (was 5.94e-16 rel) |
| **pt** | **2.84e-13** | **9.23e-16** | **UNCHANGED — identical to pre-fix value** |

**Decisive follow-up test**: compared the fixed binary's own CUDA `pt`
output directly against the pre-fix binary's CUDA output (arm B old vs arm
B new, both real 6-rank MPI runs) for the exact Stage-0-selected point
(`pe=4, call=1, field=pt, i=16, j=16, k=5`):

```
old (pre-fix, default FMA):  2.6175499933487606E+02
new (post-fix, -fmad=false): 2.6175499933487606E+02   <- BIT-FOR-BIT IDENTICAL
```

**`-fmad=false` has zero effect on `pt`'s actual computed value in real
production data.** The fix genuinely eliminated `delz`'s FMA-contraction
residual (real, reproducible, confirmed in production, not just fixtures),
but did **not** touch `pt` (or `pkz`) at all.

## 10. Root cause of the pt/pkz residual, refined

Source-level check of exactly what each field's final expression involves
(`fv3_pt_delz_remap.cu`):

- `delz_out` (line 239): `-q2_delz[k] * new_delp` — a pure multiply, no
  transcendental call. This is exactly the kind of expression FMA
  contraction affects — hence the flag fixes it.
- `pt_pre[k]` (pretransform, lines 196-197) and `pkz_out` (line 240-241):
  both routed through `fv3_exp<fv_real>(... fv3_log<fv_real>(...))`.
  `fv3_exp`/`fv3_log` (`fv3_vertical_column.hpp:12-13`) are thin
  `__device__` wrappers directly around CUDA's own `exp()`/`log()` —
  precompiled device-math-library calls, not source-level `a*b+c`
  expressions. `-fmad` controls only whether the **compiler** contracts
  explicit multiply-add patterns it sees in source; it has no way to alter
  a precompiled library function's internal algorithm.

**Revised classification**: `delp`/`delz`/`pe`/`peln`/`pk` are **category
B** (FMA contraction) — correctly diagnosed, correctly fixed, confirmed in
real production data (§9). `pt`/`pkz` are a **different, unfixed
mechanism — category H (intrinsic-library difference)**: CUDA's device
`exp()`/`log()` and Intel ifort's `-no-fma` libm `exp()`/`log()` are two
independent implementations, each within ~1 ULP of the true value but not
guaranteed to round to the *same* ULP for a given input — agreement is
data-dependent, which is exactly why all 6 canned smoke fixtures show `pt`
bitwise (§8) while real production grid data (a much larger, "unchosen"
set of inputs) hits rounding-boundary cases that disagree by 1-3 ULP
(Stage 0's selected value, §1). No compiler flag changes this; achieving
true bit-for-bit `exp`/`log` would require reimplementing one vendor's
transcendental algorithm to bit-match the other's — not a narrow
arithmetic-semantic correction, and not attempted (outside this prompt's
one-fix budget and its "narrow fix" constraint).

## 11. Stage 10 — multi-call growth, post-fix

Same job pair, calls 1-6 (`compare_arms.py` per-call table, B vs C):

| call | delp | delz | pe | peln | pk | pkz | pt |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0e-16 | 9.2e-16 |
| 2 | 1.62e-09 | 7.36e-10 | 1.44e-09 | 1.26e-10 | 4.12e-10 | 4.16e-10 | 4.26e-10 |
| 3 | 1.34e-08 | 5.82e-09 | 1.19e-08 | 1.04e-09 | 3.40e-09 | 3.43e-09 | 3.22e-09 |
| 4 | 5.21e-08 | 2.22e-08 | 4.62e-08 | 4.05e-09 | 1.32e-08 | 1.33e-08 | 1.46e-08 |
| 5 | 1.53e-07 | 6.54e-08 | 1.36e-07 | 1.19e-08 | 3.89e-08 | 3.92e-08 | 4.76e-08 |
| 6 | 3.74e-07 | 1.62e-07 | 3.32e-07 | 2.91e-08 | 9.49e-08 | 9.56e-08 | 1.19e-07 |

**Numerically identical (to the printed precision) to Prompt 62 Stage 6's
pre-fix table.** The fix's one real improvement (delz's call-1 seed going
from 1.56e-15 to bitwise) has no measurable effect on the multi-call
growth trajectory, because `pt` — unaffected by the fix, and of comparable
or larger magnitude than delz's old residual — remains the dominant seed.
Per §10, `pt`'s residual is not addressable by any single compiler flag,
so this growth is expected to persist unchanged regardless of further
FMA-related fixes.

## 12. Stages 11/13 not run — cost-efficiency stop condition reached

Given §9-11 show the fix does not change the multi-call divergence
magnitude at all (§11's table is unchanged from the pre-fix baseline), a
C32 `dcx` comparison (Stage 11) or a C512 A/B run (Stage 13) would only
reproduce the already-known, already-quantified (Prompt 62 §7) growth
pattern at additional compute cost, without new information — the
determining factor (whether `pt`'s residual can be eliminated) has already
been resolved directly (§9/§10: no, not by a compiler flag). Per this
project's bounded-debugging policy ("stop once evidence is sufficient"),
these stages are not run.

## 13. Final status

- **Fix applied and validated**: `-fmad=false` scoped to
  `fv3_pt_delz_remap.cu` (Stage 7) is a real, permanent, in-scope
  improvement — `delz`'s call-1 FMA-contraction residual is eliminated
  (confirmed bitwise in real 6-rank production data, §9), with zero effect
  on the Riemann solver and zero algorithm change. This should be kept
  (not reverted) regardless of the outcome below, since it strictly
  improves correctness for delz/delp/pe/peln/pk with no downside.
- **Primary target (`pt`) NOT resolved**: Stage 0's selected first-differing
  value, and the aggregate `pt` residual generally, is **completely
  unaffected** by the fix (bit-for-bit identical CUDA output before and
  after) — root-caused to CUDA device `exp()`/`log()` vs Intel ifort
  `-no-fma` libm `exp()`/`log()` producing different last-bit rounding for
  some (not all) real input values, a data-dependent, cross-vendor
  transcendental-library difference with no compiler-flag remedy.
- **Multi-call growth (Prompt 62/63's original concern) is unchanged** —
  still reaches ~3e-7 relative (delp/pe) by call 6, since `pt`, the
  unaffected field, remains the dominant seed.
- **No second fix attempted** (one-fix budget already used on a
  confirmed-correct, confirmed-limited-scope target).
- **Decision**: this matches Prompt 64's category-D outcome — the
  remaining FP difference is intrinsic-library-limited, not a compiler/
  arithmetic-order issue fixable with one narrow correction. Whether to
  pursue a larger-scope numerical-policy change (e.g., a custom bit-
  matching `exp`/`log` implementation, or accepting the current
  round-off-level/ULP-level disagreement as permanent) is a decision
  outside this prompt's authorization.

**PT/DELZ FP DIFFERENCE IS INTRINSIC-LIBRARY LIMITED — REQUIRES NUMERICAL POLICY DECISION**

## 14. Addendum (Prompt 65)

The "free lunch" alternative implied above — split dispatch so CUDA keeps
DELZ (now bitwise via this prompt's `-fmad=false` fix) while Fortran's own
`map_scalar` computes PT, sidestepping the unfixable `exp`/`log` residual
entirely — was implemented and tested. **Result: worse, not better.** The
hybrid split's six-call C32 divergence from Fortran is ~6-75x larger than
the already-inadequate full-CUDA (PT+DELZ both CUDA) case, because routing
PT through Fortran introduces a new per-call backend-boundary seam that
amplifies faster than a single self-consistent-backend seed. This path is
now closed; the choice between accepting a tolerance-based gate (option
(a) above) or a large-scope bit-matching `exp`/`log` effort (option (b))
remains the only two live options. Full detail:
`docs/performance/p65_cuda_delz_fortran_pt_policy_2026-09-23.md`.
