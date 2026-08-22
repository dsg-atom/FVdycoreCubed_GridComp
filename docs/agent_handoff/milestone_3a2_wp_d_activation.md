# Milestone 3A.2 — Complete WP-D CUDA Production Dispatch Activation

**Date:** 2026-08-21
**Branch:** `cuda/gpu-bringup`
**Starting commit:** `feaafc1` (add claude.md) — unchanged; all work in this
milestone is uncommitted on top of it, as is Milestone 3A.1's work (the
`fv3_build_production_dispatch`/`wp_d_contract_cuda`/`test_dispatch_smoke.cpp`
infrastructure it built on).
**Final commit:** `feaafc1` (no commit made this milestone, per instructions).

## Objective

Milestone 3A.1 activated WP-A `cgrid_pressure` and WP-C `vertical_solver`
through the real production `NumericalDispatch`, but WP-D's `nh_pressure`/
`post_loop_heating` stages honestly failed at `bind_wp_d` contract validation
(`missing required WP-D field u`) because the test harness's field set was
deliberately minimal. This milestone's objective was to complete the WP-D
field contract in the test harness only — no WP-D/WP-A/WP-C kernel or
contract source changes — so all four wired production-dispatch stages
(`cgrid_pressure`, `vertical_solver`, `nh_pressure`, `post_loop_heating`)
genuinely GPU-execute end to end through `fv3_gpu_dyn_core_step`.

## What Was Done

`cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` (the only file
touched) was extended:

1. Registered every remaining `bind_wp_d`-required field with a shape
   satisfying `wp_d_contract.cpp`'s `covers()`/`covers_2d()`/`covers_3d()`
   checks: `u`, `v`, `pkz`, `du`, `dv`, `heat_source` (mid-layer, reuse the
   `lo_3d`/`ex_3d` full-halo box), and `divg2`, `rdx`, `rdy`, `rarea`,
   `del6_u`, `del6_v`, `dx`, `dy`, `sina_u`, `sina_v`, `dxa`, `dya`,
   `grid_lonlat`, `agrid_lonlat` (2D grid metrics, reuse the `lo_2d`/`ex_2d`
   box, with `grid_lonlat`/`agrid_lonlat` getting a trailing lon/lat
   dimension of extent 2), plus presence-only `edge_w`/`edge_e`/`edge_s`/
   `edge_n` accumulator fields.
2. Rewrote Phase B's post-step assertions from "expected partial failure at
   contract validation" (Milestone 3A.1's honest blocker) to "expected full
   success," with a real D2H transfer and finite/changed-value verification
   of WP-D's actual output (`u`/`v` for `nh_pressure`; `pt`/`pkz`/
   `heat_source` for `post_loop_heating`), using pre-step snapshots and an
   `any_changed()` element-wise comparison (not a naive `any_nonzero()`)
   because those fields start out already nonzero — the same
   exact-cancellation-avoidance rigor as Milestone 3A.1's `uc`/`vc` check.
3. Found and fixed a genuine `bind_wp_d` contract-vs-kernel axis conflict for
   `pe`/`peln` (see below) — the one substantive bug this milestone
   uncovered.

## Critical Defect Found and Fixed: `pe`/`peln` Halo-Convention Conflict

Phase B initially failed at runtime (Slurm job `58020372`, R8, A100 node
`warpa003`) with:

```
error.api = "bind_wp_d"  message = "WP-D descriptor does not cover required bounds"
```

**Root cause:** `wp_d_contract.cpp`'s `bind_wp_d` requires, for `pe`/`peln`
specifically, that the field's registered axis-2 slot cover
`[js - p, je + p]` (`p = 1` for `pe`, `p = 2` for `peln`) — i.e. it expects
`lower[2] <= js - p`. But `launch_vertical_solver`'s `riem_solver3_kernel`
(WP-C) requires the *same* registered `lower[2]` to make every
`(k - lower[2])` view offset for its hardcoded `k = 0..npz` loop land inside
the allocated buffer — which, given `test_dispatch_smoke.cpp`'s domain
(`is=1, ie=8, js=1, je=8`) and the Milestone 3A.1 fix (`lower[2] = 0`), means
`lower[2]` must be `<= 0`, not `<= js - p = -1` (`peln`) or (coincidentally,
only at this exact domain size) `<= js - 1 = 0` (`pe`).

Both `pe` and `peln` are shared, single-descriptor fields consumed by both
WP-C (`vertical_solver`) and WP-D (`nh_pressure`'s `bind_wp_d` gate) within
the *same* `fv3_gpu_dyn_core_step` call, so one registered `lower[2]` must
satisfy both requirements simultaneously. `pe` happened to pass by boundary
coincidence (`js - 1 == 0`); `peln` did not (`js - 2 == -1 < 0`), and neither
requirement is robust to a different domain/halo choice.

Note: neither `launch_nh_pressure_cuda` nor `launch_post_loop_heating_cuda`
(`wp_d_contract_cuda.cpp`) actually dereferences `pe`/`peln` data in this
milestone's dispatched stages — `bind_wp_d` requires their *presence and
shape* only, as a gate. This confirms the conflict is a pure bookkeeping
mismatch, not a numerical-correctness issue in either kernel.

**Fix (test-harness metadata only):** gave `pe`/`peln` their own k-padded
buffer, `kPelnHaloPad = 2` dummy leading k-slices deeper than the shared
`pk`/`pk3`/`pkc`/`zh`/`gz` buffer, with `lower[2] = -kPelnHaloPad` registered
instead of `0`. Both WP-C's `view3()`/`cview3()` (`fv3_vertical_kernels.cu`)
and WP-D's `view3_const()` (`wp_d_operators.hpp`) build device views
generically from each field's own registered `lower[]`/`stride[]` — this was
verified by direct source read before making the change — so the vertical
solver's `k=0..npz` loop still correctly addresses the same physical data,
now at buffer offset `kPelnHaloPad + k` instead of `k`; `bind_wp_d`'s
`lower[2] <= js - p` check now passes for both fields (`-2 <= -1` for
`peln`, `-2 <= 0` for `pe`), robustly rather than by domain-size coincidence.
No field data used by any executed kernel moves in value or timing — only
its buffer offset changed.

## Fix Table (Section 9)

| file | reason | numerical semantics changed |
|---|---|---|
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — register `u`/`v`/`pkz`/`du`/`dv`/`heat_source` (mid-layer) | complete `bind_wp_d`'s required-field set so `nh_pressure`/`post_loop_heating` pass contract validation | NO |
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — register `divg2`/`rdx`/`rdy`/`rarea`/`del6_u`/`del6_v`/`dx`/`dy`/`sina_u`/`sina_v`/`dxa`/`dya`/`grid_lonlat`/`agrid_lonlat`/`edge_{w,e,s,n}` (2D grid metrics) | complete `bind_wp_d`'s required-field set | NO |
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — Phase B: replace "expect partial failure" with "expect full success" + D2H + `any_changed()` verification of `u`/`v`/`pt`/`pkz`/`heat_source` | Phase B must now genuinely validate WP-D output now that the field contract is complete, with pre/post snapshots to avoid a false-positive `any_nonzero()` on fields that start nonzero | NO |
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — give `pe`/`peln` a dedicated k-padded buffer (`kPelnHaloPad`), `lower[2] = -kPelnHaloPad` | resolve the `bind_wp_d`-vs-`riem_solver3_kernel` axis-2 halo conflict described above | NO |
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` — header/summary comment updates (3A.1 → 3A.2 narrative) | documentation only | NO |

No WP-A/WP-B/WP-C/WP-D kernel source, `wp_d_contract.cpp`/`wp_d_contract.hpp`
validation logic, or `fv3_gpu_runtime.cpp` orchestration was modified. All
changes are confined to the test harness (`test_dispatch_smoke.cpp`).

## Validation Results (A100, Slurm jobs `58020488`/`58020494`, both R8 and R4)

```
Phase A (minimal dispatch, full success + D2H verify):        PASS
Phase B (full production dispatch, full success + D2H verify): PASS
Persistent residency: PASS (no intermediate H2D/D2H between stages within a step)
Milestone 3A.2 dispatch smoke test: PASS
```

- WP-A `p_grad_c` wrote nonzero `uc`/`vc`: YES (both precisions — regression
  from Milestone 3A.1 confirmed unchanged).
- WP-D `nh_pressure` wrote nonzero `u`/`v`: YES (both precisions).
- WP-D `post_loop_heating` changed `pt`/`pkz`/`heat_source`: YES (both
  precisions, via element-wise `any_changed()` against pre-step snapshots).
- `compute-sanitizer --tool memcheck`: **0 errors**, both R8 and R4.
- `ctest -L gpu`: **100% tests passed, 0 tests failed out of 2**, both build
  directories.
- H2D/D2H inventory: exactly one `fv3_gpu_initialize_state` H2D and one
  `fv3_gpu_sync_to_host` D2H per phase/context; nothing between stages
  within a single `fv3_gpu_dyn_core_step` call — persistent device
  residency preserved, no architectural regression introduced.
- WP-B confirmed untouched: `fv3_gpu_production_dispatch.cu`'s
  `dgrid_dynamics`/`dgrid_height_transport`/`dgrid_height` remain explicit
  `nullptr`; the runtime log's `Phase A` printout independently confirms
  `nh_pressure = null (not exercised in this phase)` when only the WP-A/WP-C
  minimal table is installed, and `fv3_gpu_dyn_core_step`'s stage loop
  safely skips null entries.

## Honest Status by Validation Hierarchy

| Component | IMPLEMENTED | COMPILED | GPU EXECUTED | NUMERICALLY VALIDATED |
|---|---|---|---|---|
| WP-A `cgrid_pressure` | YES | YES | **YES** (sanitizer-clean, D2H-verified) | NOT RUN |
| WP-C `vertical_solver` | YES | YES | **YES** (sanitizer-clean, D2H-verified) | NOT RUN |
| WP-D `nh_pressure` | YES | YES | **YES** (sanitizer-clean, D2H-verified, nonzero `u`/`v`) | NOT RUN |
| WP-D `post_loop_heating` | YES | YES | **YES** (sanitizer-clean, D2H-verified, `pt`/`pkz`/`heat_source` genuinely changed) | NOT RUN |
| WP-B (any stage) | N/A | N/A | **NO CUDA IMPLEMENTATION AVAILABLE** (Fortran-only) — unchanged from 3A.1, not exercised, not attempted | N/A |
| `fv3_gpu_dyn_core_step` full 4-stage production path | YES | YES | **YES — full success**, both R8 and R4 | NOT RUN |
| Persistent device residency | — | — | **PASS** | N/A |
| `compute-sanitizer` memcheck | — | — | **PASS** — 0 errors, both R8 and R4 | N/A |
| Numerical validation vs. Fortran baseline | — | — | — | **NOT RUN** |

## Remaining Blockers

1. WP-B has no CUDA implementation at all — out of scope for this milestone
   (explicitly not authorized: "Do not implement WP-B").
2. `fv3_gpu_production_dispatch.cu`'s `corner_edge_service` remains a
   fail-adapter (`corner_service_not_implemented`); this milestone's test
   configuration (`config.nf_ke = 1`, i.e. `passes = 1`, `nt = 0`) never
   invokes it, so this is not exercised and is not a new blocker introduced
   here — carried over from Milestone 3A.1's dispatch wiring.
3. No numerical validation against authoritative Fortran savepoints has been
   performed — out of scope for this milestone (belongs to the Prompt 6
   numerical validation phase).
4. This milestone's test domain (8×8×10, halo=3, idealized synthetic grid
   metrics) is a smoke-test fixture, not a production GEOS/FV3 domain or
   configuration — production-first-run coverage (`hord_mt=6` etc., real
   grid metrics, real corner-service-requiring configurations) has not been
   attempted.

## Conclusion

**WP-D PRODUCTION DISPATCH ACTIVATION: COMPLETE (for this milestone's test
domain)**

All four wired production-dispatch stages — WP-A `cgrid_pressure`, WP-C
`vertical_solver`, WP-D `nh_pressure`, WP-D `post_loop_heating` — now
genuinely GPU-execute end to end through the real `fv3_gpu_dyn_core_step`
orchestration path, sanitizer-clean, with D2H-verified non-degenerate
output, in both R8 and R4. The one genuine defect found (a `pe`/`peln`
halo-convention conflict between `bind_wp_d`'s contract and
`riem_solver3_kernel`'s addressing requirement) was resolved with a
test-harness-only bookkeeping fix that changes no numerical semantics in any
executed kernel. WP-B remains explicitly unwired and out of scope.

**READY FOR WP-B CUDA IMPLEMENTATION**
