# Prompt 52B: CUDA `map_scalar`/`scalar_profile` Standalone Implementation

Branch: `optimization/cuda-fv3-performance`, baseline commit `33e10c5`
(tag `cuda-perf-baseline-c512`). Continues Prompt 52A's Fortran baseline
harness (`tests/fortran_baseline/fv_mapz/`, `docs/performance/
p52a_fv_mapz_fortran_baseline_2026-09-21.md`) and the port plan
(`docs/performance/p52_fv_mapz_cuda_port_plan_2026-09-21.md`).

## 1. CUDA design

One CUDA/host thread per horizontal column, matching the existing WP-C
vertical convention (`fv3_vertical_column.hpp`'s `edge_profile_column` etc.):
the Fortran `i`-loop collapses to "this column", the `k`-loop stays serial
inside the thread.

Implementation is header-only, `__host__ __device__` (`FV3_VERTICAL_HD`)
templates, not a raw-pointer `Args`-struct + `launch_*` wrapper (the
`RiemSolverCArgs`/`launch_riem_solver_c` convention). This let host-only
correctness be validated before any GPU/Slurm involvement, and the same
code is what a `__global__` kernel calls per-thread with no rewrite.

Scope implemented, matching the Prompt 52A fixtures and the production
Held-Suarez C512 config (`kord_mt=kord_wz=kord_tm=9`):

- `scalar_profile`'s `kord==9` branch only; `iv` in `{0,1}` only; the
  `iv/=-2` tridiagonal solve only (no `qs`); ext5/ext6 not computed (dead
  code at `kord==9`).
- `map_scalar`'s "standard FV3 remapping" branch only; GMAO bottom/top
  branches excluded (confirmed unused by the fixture/production driver).
- `cs_limiters`, identified in Prompt 52A/52 planning as a third required
  routine (`scalar_profile` calls it at four points), fully ported.
- `mapn_tracer`, `ppm_profile`: explicitly NOT implemented, per directive.

Numerical semantics changed relative to the authoritative Fortran: **NO**.

## 2. Source

- `cuda/dyn_core/vertical/fv3_map_scalar_column.hpp` — `cs_limiters_column<T>`,
  `scalar_profile_column<T>`, `map_scalar_column<T>` device/host functions.
  Full Fortran-line-cited provenance in the file header and inline comments
  (`fv_mapz.F90:1519-1671`, `:1881-2288`, `:2699-2774`).
- `cuda/dyn_core/vertical/fv3_map_scalar_kernel.cu` — minimal `__global__`
  smoke kernel + host launcher wrapping the same header functions, one
  fixture-loading/comparison `main()`. GPU-execution sanity only; not a
  production kernel/launcher (no `ExecutionContext`, no `fv3_gpu_error`
  reporting layer — deferred, as those are launcher-layer concerns for a
  future production integration, not needed for standalone validation).
- `cuda/dyn_core/tests/vertical/test_map_scalar.cpp` — host-only ctest
  (`fv3_wp_c_map_scalar_test`/`fv3_wp_c_map_scalar`), parses all 8 Prompt
  52A fixtures, derives `dpe1`/`dpe2` by subtraction from `pe1`/`pe2` (per
  the driver's own `main.F90:76,79` formula), calls `map_scalar_column
  <double>`, compares bitwise, includes a 5-repeat determinism check.
- `cuda/dyn_core/CMakeLists.txt` — registers `fv3_wp_c_map_scalar_test`
  (ctest) and `fv3_wp_c_map_scalar_gpu_smoke`/`_nofma` (GPU smoke binaries,
  not registered as ctests — no GPU on the ctest-running login node).

Temporary storage (`gam`, `q`, `extm`, `c2/c3/c4`): host path uses
caller-owned arrays sized exactly to `km`/`km+1` (no oversized cap). The
GPU smoke kernel uses fixed local arrays capped at `kMaxLevels=32`
(matching the `riem_solver_c_max_local_levels` convention), rejecting
`km/kn > 32` in the launcher rather than truncating; all 8 fixtures use
`km=kn=10`, well inside the cap.

## 3. First case (Stage 9)

`case1_smooth_monotonic.txt` (`iv=1`, `kord=9`): host path bitwise PASS on
first attempt (`mismatch=0/10`, `max_abs=0`), confirming the index-mapping
derivation (Fortran 1-based → 0-based) was correct before expanding to the
other 7 fixtures.

## 4. All fixtures (Stage 10)

Host path (`fv3_wp_c_map_scalar_test`, R8, GCC 12.3.0, login node, no GPU):

| Case | iv | kord | max_abs | max_rel | mismatch | Result |
|---|---|---|---|---|---|---|
| case1_smooth_monotonic | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case2_strong_gradient | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case3_local_extrema | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case4_near_uniform | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case5_variable_layer_thickness | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case6_limiter_activating | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case7_realistic_c512_profile | 1 | 9 | 0 | 0 | 0/10 | BITWISE PASS |
| case8_positive_definite_iv0 | 0 | 9 | 0 | 0 | 0/10 | BITWISE PASS |

All 8/8 BITWISE PASS on the host path — no round-off fallback needed there.

GPU path (`fv3_wp_c_map_scalar_gpu_smoke`, one A100, `srun`, job 58497066,
node `warpa004`), default nvcc FMA contraction:

| Case | max_abs | max_rel | mismatch | Result |
|---|---|---|---|---|
| case1_smooth_monotonic | 5.68e-14 | 2.15e-16 | 3/10 | ROUND-OFF |
| case2_strong_gradient | 5.68e-14 | 1.96e-16 | 1/10 | ROUND-OFF |
| case3_local_extrema | 8.53e-14 | 3.71e-16 | 6/10 | ROUND-OFF |
| case4_near_uniform | 1.14e-13 | 3.79e-16 | 2/10 | ROUND-OFF |
| case5_variable_layer_thickness | 5.68e-14 | 2.18e-16 | 3/10 | ROUND-OFF |
| case6_limiter_activating | 2.84e-14 | 1.15e-16 | 2/10 | ROUND-OFF |
| case7_realistic_c512_profile | 5.68e-14 | 2.19e-16 | 3/10 | ROUND-OFF |
| case8_positive_definite_iv0 | 8.53e-14 | 3.71e-16 | 6/10 | ROUND-OFF |

Same job, same binary rebuilt with `-fmad=false` (FMA contraction disabled):
all 8/8 **BITWISE PASS** (`mismatch=0/10`, `max_abs=0` for every case).

## 5. Determinism (Stage 13)

Host path: `case6_limiter_activating` (limiter-activating case, `iv=1`)
repeated 5 times — bitwise-identical every repeat. PASS.

## 6. Root cause (Stage 11 — bitwise-first policy)

The GPU default-build round-off (max_abs ≤ 1.14e-13, max_rel ≤ 3.8e-16,
double precision) is fully explained and reproduced on demand: rebuilding
the identical kernel with `-fmad=false` eliminates it completely (8/8
bitwise PASS on the same GPU, same job). This is nvcc's default fused
multiply-add contraction reassociating `a*b+c` into a single rounding step,
which the host/Fortran path (separate multiply then add) does not do —
the same class of divergence already established as benign at Gate C2 and
explicitly anticipated by the fixtures' README ("fall back to Gate C2's
existing round-off tolerance only if a justified non-bitwise difference
appears (e.g. FMA/instruction-order differences on GPU)"). No further
localization needed; stop condition met per the bounded-debugging
doctrine. This is **not** treated as a numerical semantics change: the
`-fmad=false` build proves the underlying algorithm/translation is exactly
bitwise-equivalent, and default FMA contraction is a standard,
well-understood compiler optimization, not an algorithmic difference.

## 7. Temporary storage

See §2. No MaxLevels constant in the header itself (host path uses
caller-sized scratch); the GPU smoke kernel's own `kMaxLevels=32` cap is a
launcher-layer decision, not a header-level one, and does not affect the
header's own semantics or reusability at other `km`/`kn` (a future
production launcher can choose its own cap/workspace strategy
independently).

## 8. Microbenchmark (Stage 15)

Non-authoritative sanity check only (single job, no controlled A/B,
`case7_realistic_c512_profile` shape replicated to 100,000 columns):

- GPU kernel (1 launch, all 100,000 columns): **0.35 ms**
- Single-thread host loop (100,000 sequential calls): **~137 ms**

This confirms the kernel executes real, non-trivial per-column work on the
GPU and is not a no-op; it is **not** a production performance figure (no
multi-rank context, no halo/host-transfer overhead, no comparison against
a multi-threaded/vectorized host baseline) and must not be cited as a
`fv_mapz` speedup estimate elsewhere in the project.

## 9. R4 (Stage 14)

Not attempted in this prompt. All fixtures and both test binaries are R8
(`double`) only. Deferred to a future prompt if `fv_mapz` R4 validation is
requested (mirroring the R8/R4-separate-report convention established at
Gate C2).

## 10. 52C integration readiness (Stage 16)

Standalone `map_scalar`/`scalar_profile`/`cs_limiters` CUDA port is
correctness-validated (host bitwise, GPU bitwise-with-`-fmad=false`,
GPU round-off-only-and-explained with default FMA) against all 8 Prompt
52A fixtures, at the production `kord=9`/`iv∈{0,1}` configuration.

Before any 52C production-timestep integration, still needed (not
attempted here, by directive):

- A real production kernel launcher (`ExecutionContext`/`fv3_gpu_error`
  integration, batch-of-columns dispatch matching the existing WP-C
  runtime conventions) — the Stage-15 smoke kernel is deliberately minimal
  and not that launcher.
- Decision on whether to keep default FMA (round-off-only, already
  explained and precedented at Gate C2) or force `-fmad=false` for this
  translation unit specifically, consistent with whatever the rest of the
  CUDA dyn_core does today (not audited in this prompt — the existing
  Riemann-solver/D-grid kernels' FMA setting was out of scope here).
- R4 validation (§9).
- Wiring into the real production vertical grid sizes (`km`/`kn` beyond
  the fixtures' `km=kn=10`) and the `kMaxLevels` cap decision for a real
  launcher (the smoke kernel's `32` is illustrative only).
- `mapn_tracer` and any other `fv_mapz` entry points remain out of scope
  entirely (by directive, not evaluated here).

## Final status

**FV_MAPZ CUDA STANDALONE PASS — READY FOR 52C INTEGRATION**
