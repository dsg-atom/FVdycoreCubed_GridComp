# Milestone 3A.1 — Activate Existing CUDA Through Production Dispatch

**Date:** 2026-08-21
**Branch:** `cuda/gpu-bringup`

## Objective

Wire a real `NumericalDispatch` (real function pointers, no no-op doubles)
into `fv3_gpu_dyn_core_step`, activate already-existing WP-A/WP-C CUDA
implementations through it, and demonstrate genuine GPU execution — not
merely CUDA compilation — with `compute-sanitizer` confirmation. Do not
write new WP-A/B/D numerical orchestration; activate only what already
exists.

## What Was Built

- `fv3_build_production_dispatch()` — real `NumericalDispatch` with function
  pointers into existing WP-A (`cgrid_pressure` → `launch_cgrid_pressure_v3`)
  and WP-C (`vertical_solver` → `launch_vertical_solver`) CUDA
  implementations, plus WP-D `nh_pressure`/`post_loop_heating` stage slots.
- `fv3_gpu_dyn_core_step` real orchestration loop over
  `stages[] = {dispatch.cgrid_pressure, dispatch.vertical_solver,
  dispatch.nh_pressure, dispatch.post_loop_heating}` — no silent host
  fallback, errors propagate, no unnecessary device sync, persistent device
  state preserved across stages.
- `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — new smoke test
  exercising two phases:
  - **Phase A** (minimal dispatch): `cgrid_pressure` + `vertical_solver`
    only, on an 8×8×10-with-halo domain — designed to fully succeed.
  - **Phase B** (full production dispatch): all four stages — designed to
    fail honestly at the first genuinely-missing WP-D field, proving the
    orchestration propagates real contract errors rather than masking them.

## Critical Defect Found and Fixed: Illegal Memory Access

Running Phase A through `compute-sanitizer --tool memcheck` (Slurm job
`58017707`, A100 node `warpa003`) surfaced a reproducible **CUDA illegal
memory access** at `riem_solver3_kernel` (`fv3_vertical_kernels.cu`), which
also poisoned the CUDA context for subsequent independent test runs in the
same process.

**Root cause:** an axis-convention mismatch between the test harness's
field-descriptor metadata and the vertical-solver kernel's indexing
convention — not a kernel bug and not a numerical-algorithm bug.

- `riem_solver3_kernel` / `launch_vertical_solver` hardcode 0-based k loops
  (`for (k=0;k<km;++k)` for mid-layer fields such as `delp`/`pt`/`w`/`q_con`/
  `cappa`; `for (k=0;k<=km;++k)` for interface fields), i.e. they require the
  registered field descriptor's `lower[2] == 0`.
- `test_dispatch_smoke.cpp`'s `make_fields()` had set
  `lo_3d[2] = lo_3d_if[2] = 1` (a Fortran-style 1-indexed convention).
- This is a genuinely different convention from WP-A's
  `pressure_gradient_c_x`/`_y` kernels (`fv3_cgrid.cu`), which derive k
  relative to whatever `lower[2]` is registered
  (`k = a.delpc.lower[2] + level`) — explicitly documented in-source as
  base-agnostic ("kernels never assume Fortran k starts at one"). It is also
  different from WP-D's `operators::View3` convention used by the earlier
  `test_integrated_smoke.cpp` (`lo_k = 1`, per that file's own comment).
  **There is no single universal k-axis convention across WP kernel
  families** — each family's expectation must be verified independently
  against whatever `lower[2]` a caller supplies.
- Manually verified the sanitizer's reported OOB byte offsets
  (760/752/744/736/728, for interior point i=1..5,j=5,k=0) exactly match the
  arithmetic predicted by the off-by-one `lower[2]` mismatch.

**Fix (test-harness metadata only, no kernel/contract source touched):**

| file | purpose | numerical semantics changed |
|---|---|---|
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` | Changed `lo_3d[2] = lo_3d_if[2] = 1` → `= 0` in `make_fields()`, with a comment explaining the vertical-solver's hardcoded 0-based k convention vs. WP-A's base-agnostic convention | **NO** |

Verified this does not affect WP-A `cgrid_pressure` correctness because that
kernel derives k from the registered `lower[2]` rather than assuming a fixed
base. Re-ran `compute-sanitizer`: `ERROR SUMMARY: 0 errors` on both R8 and
R4 after the fix.

## Second Defect Found and Fixed: Exact-Cancellation False Positive

After the above fix, R8 still reported `Phase A result: FAIL` — WP-A's
`p_grad_c` wrote an exactly-zero `uc`/`vc` update — while R4 spuriously
passed. Root cause: `pressure_gradient_c_x`/`_y` is a strictly horizontal
(i,j)-difference operator, and the test's synthetic `gz`/`pkc` fields varied
only with `k`, not with `i,j`. This makes the kernel's `first`/`second`
terms cancel **exactly** (`first == -second`) in exact arithmetic. R8's
double precision preserved the exact zero; R4's float32 rounding noise
merely looked nonzero — a false positive, not a real signal.

**Fix (synthetic test input data only):**

| file | purpose | numerical semantics changed |
|---|---|---|
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` | Added a horizontal perturbation `horiz = 0.02·sin(0.35·i)·cos(0.35·j)` applied multiplicatively to `pk`/`pk3`/`pkc` and to `gz` in `make_host_state()`, breaking horizontal uniformity so `p_grad_c` produces a genuine, non-accidental nonzero result in both precisions | **NO** |

## Validation Results (A100, Slurm job `58017954`, both R8 and R4)

```
WP-A p_grad_c wrote nonzero uc/vc: YES
Phase A result: PASS
Phase B result: PASS (honest outcome — fails at first genuinely missing field)
Milestone 3A.1 dispatch smoke test: PASS
```

Phase B's honest failure:
```
error.api = "bind_wp_d"  message = "missing required WP-D field u"
```

- `compute-sanitizer --tool memcheck`: **0 errors**, both R8 and R4.
- `ctest -L gpu`: **100% tests passed, 0 tests failed out of 2**, both R8 and
  R4 build directories.

## Honest Status by Validation Hierarchy

| Component | IMPLEMENTED | COMPILED | GPU EXECUTED | NUMERICALLY VALIDATED |
|---|---|---|---|---|
| WP-E orchestration (`fv3_gpu_dyn_core_step`, real `NumericalDispatch`) | YES | YES | YES | N/A (infra, not numerics) |
| WP-A `cgrid_pressure` / `pressure_gradient_c_{x,y}` | YES | YES | **YES** (sanitizer-clean, D2H-verified, genuinely nonzero) | NOT RUN |
| WP-B (any stage) | N/A | N/A | **NO CUDA IMPLEMENTATION AVAILABLE** (Fortran-only) | N/A |
| WP-C `vertical_solver` / `riem_solver3` | YES | YES | **YES** (sanitizer-clean, D2H-verified, finite) | NOT RUN |
| WP-D `nh_pressure` / `post_loop_heating` | YES (wired) | YES | **NO** — honestly blocked at contract validation for this milestone's test domain (`missing required WP-D field u`); not a false pass | NOT RUN |
| `fv3_gpu_dyn_core_step` full 4-stage production path | YES | YES | **PARTIAL** — succeeds fully for the 2-stage minimal path (WP-A + WP-C); fails honestly, with a real propagated contract error, at stage 3 (WP-D) for the full path | NOT RUN |
| Persistent device residency | — | — | **PASS** — no intermediate H2D/D2H within a step | N/A |
| `compute-sanitizer` memcheck | — | — | **PASS** — 0 errors, both R8 and R4 (job 58017954) | N/A |
| Numerical validation vs. Fortran baseline | — | — | — | **NOT RUN** |

## Remaining Blockers

1. WP-B has no CUDA implementation at all — this milestone did not add one
   (explicitly out of scope; activation-only, not new numerical code).
2. WP-D's `nh_pressure` contract requires field `u`, which is not registered
   by this milestone's minimal test domain — a genuine field-coverage gap,
   not a dispatch or kernel defect.
3. No numerical validation against authoritative Fortran savepoints has been
   performed (out of scope for this milestone; belongs to the numerical
   validation phase).

## Documentation Corrections

Added superseded/clarified annotations (not deletions) to:
- `docs/agent_handoff/cuda_bringup.md:264` (`READY FOR PROMPT 6 VALIDATION`)
- `docs/agent_handoff/cuda_bringup.md:407` (`PROMPT 5B.1 COMPLETE - READY FOR PROMPT 6`)
- `docs/agent_handoff/integration.md` (`Production GEOS first-run numerical path: COMPLETE` block)

These previously certified readiness based on WP-E/WP-D direct-call tests or
implementation-coverage inventories that never exercised the real
`NumericalDispatch`/`fv3_gpu_dyn_core_step` path — which this milestone
showed contained a genuine (now-fixed, test-harness-level) illegal-memory-
access defect.

## Conclusion

**PRODUCTION DISPATCH ACTIVATION BLOCKED**

Rationale: WP-A and WP-C are now genuinely GPU-executed and
sanitizer-clean through the real production dispatch path — a real
milestone advance. However, the full 4-stage production dispatch
(`fv3_gpu_dyn_core_step` exercising all of WP-A/B/C/D as required by the
production first-run configuration) is not yet achievable: WP-B has no CUDA
implementation, and WP-D's contract fails for missing required fields.
Per the milestone's binding instructions, activation of existing CUDA is not
authorized to include writing new WP-B kernels or full WP-D field wiring.
Until WP-B CUDA support and WP-D's field contract are addressed by their
owning work packages, the production dispatch path cannot be called fully
activated.
