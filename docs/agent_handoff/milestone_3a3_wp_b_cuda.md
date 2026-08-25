# Milestone 3A.3 — WP-B CUDA Implementation Planning and First Port

**Date:** 2026-08-24
**Branch:** `cuda/gpu-bringup`
**Starting commit:** `8f94c4f` (Complete WP-D production dispatch activation)
— unchanged; all work in this milestone is uncommitted on top of it, as is
the 3A.1/3A.2 precedent.
**Final commit:** `8f94c4f` (no commit made this milestone, per instructions
— continuing uncommitted, consistent with 3A.1/3A.2).

## Objective

Before this milestone, WP-B had **zero CUDA** anywhere in its history:
`cuda/dyn_core/dgrid/wp_b_kernels.hpp` (1511 lines) contained zero
`__global__`/`__device__` code, and its production dispatch slots
(`dgrid_dynamics`, `dgrid_height_transport`, `dgrid_height`) were hardcoded
`nullptr` in `fv3_gpu_production_dispatch.cu`. This blocked WP-C's
`launch_update_dz_d` from doing anything but returning
`FV3_GPU_UNSUPPORTED_CONFIGURATION` (it already called
`dispatch.dgrid_height_transport(x, error)` and null-checked it).

The objective was to (1) produce an honest audit and staged plan for all of
WP-B, correcting stale "COMPLETE" claims in
`docs/architecture/dyn_core_cuda_integrated_sequence.md`, and (2) implement
the first, smallest coherent CUDA sub-package — `dgrid_height_transport`
(`fv_tp_2d_hord6` + `ra_x`/`ra_y` + `deln_flux_transport`, scoped to the
scalar field `zh`) — which unblocks WP-C's `update_dz_d` dependency.

**Fortran cross-check: NOT POSSIBLE (source absent).** `@fvdycore`/the
authoritative GFDL Fortran checkout is not populated in this repository and
was not cloned (per `discover_platform.md`). Every translation in this
milestone was checked only against the existing host C++ in
`wp_b_kernels.hpp` (which itself carries inline comments describing the
Fortran algorithm structure it was translated from) — never against
authoritative Fortran source directly.

## What Was Done

### New files

- `cuda/dyn_core/dgrid/wp_b_cuda.cuh` / `wp_b_cuda.cu` — real `__global__`
  CUDA kernels: `ra_xy_kernel` (new code — `ra_x`/`ra_y` are ephemeral
  per-call quantities with no persistent `FieldId`, computed from
  `area`/`xfx`/`yfx`), the `fv_tp_2d_hord6` Y-X/X-Y sequencing kernel(s)
  (order-6 PPM reconstruction embedded directly in the same kernel path —
  see "Deviation from plan" below), `apply_height_flux_divergence_kernel`
  (new code, modeled on `wp_b_kernels.hpp`'s `apply_mass_and_scalar_flux`
  mass-divergence formula, since `zh` has no existing mass-field divergence
  path to translate directly), and `deln_flux_transport_kernel` (direct
  translation of `wp_b_kernels.hpp`'s `has_mass=false` path).
- `cuda/dyn_core/dgrid/wp_b_contract_cuda.hpp` / `wp_b_contract_cuda.cpp` —
  `bind_wp_b_height_transport` binder (resolves `FieldRegistry` views for
  `zh`/`crx`/`cry`/`xfx`/`yfx`/`area`/`rarea`/`del6_u`/`del6_v`, validates
  `hord_tm ∈ {1,6,8}` and `halo_width >= 3`, carves an 8-array scratch arena
  from `WorkspaceId::transport`) and `launch_dgrid_height_transport_cuda`
  (the `LaunchFn`-signature entry point wired into
  `NumericalDispatch::dgrid_height_transport`).
- `cuda/dyn_core/tests/dgrid/test_wp_b_cuda_smoke.cpp` — new permanent GPU
  smoke test, three parts:
  - **Part 1** — calls `launch_dgrid_height_transport_cuda` directly against
    a real WP-E context; verifies launch success, finite output, and
    non-degenerate change in `zh` (`any_changed()` against a pre-call
    snapshot).
  - **Part 2** — calls `launch_deln_flux_transport_cuda` standalone (it is
    implemented but not wired into the production path — see "damp_c
    mapping ambiguity" below); verifies launch success and finite output.
  - **Part 3** — cross-WP integration check: calls
    `fv3::gpu::vertical::launch_update_dz_d` (WP-C) directly, with a minimal
    local `NumericalDispatch` whose `dgrid_height_transport` points at the
    new WP-B kernel, `FV3_GPU_H_DSCALAR` marked ready via
    `mark_halo_ready`, and a nonzero `rdt` installed directly into
    `ContextData::dispatch_params` (no public setter exists for
    `DispatchParams`, unlike `NumericalConfig`'s `set_numerical_config`, so
    the already-established `reinterpret_cast<ContextData*>(ctx)` pattern
    was reused to mutate it directly — the same pattern used to reach
    `compute_stream`/`state.registry` earlier in the file). Verifies `zh`
    genuinely changes through the full WP-C → WP-B call chain, not just WP-B
    in isolation.

### Modified files

- `cuda/dyn_core/CMakeLists.txt` — added the `fv3_wp_b_cuda` CUDA static
  library target (mirrors the `fv3_wp_d_cuda` block), linked it into
  `fv3_production_dispatch`, and added the `fv3_wp_b_cuda_smoke_test`
  executable (linked against `fv3_wp_b_cuda` **and** `fv3_wp_c_vertical`,
  the latter needed for Part 3's direct call into WP-C).
- `cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` — changed
  `d.dgrid_height_transport = nullptr;` to
  `&fv3::gpu::dgrid::launch_dgrid_height_transport_cuda`. `dgrid_dynamics`
  and `dgrid_height` remain explicitly `nullptr` (still unsupported).
- `docs/architecture/dyn_core_cuda_integrated_sequence.md` — annotated (not
  deleted) the WP-B and WP-C `update_dz_d` table rows; see that file's new
  blockquote sections for the row-by-row correction.

## Deviation from the plan's literal file list

The plan anticipated separate `xppm_hord6_kernel`/`yppm_hord6_kernel`
translation units. In practice, the order-6 PPM reconstruction logic is
embedded directly inside the `fv_tp_2d_hord6` kernel path in `wp_b_cuda.cu`
— splitting it into standalone kernels would have meant either (a) an extra
global-memory round-trip for interface values with no numerical benefit at
this milestone's correctness-over-optimization stage, or (b) shared-memory
tiling, which the project rules explicitly defer past this milestone. This
is a structural simplification, not a numerical one — the same per-cell PPM
arithmetic executes in the same order; only the compilation-unit boundary
differs from the plan's suggestion.

## Open ambiguity: `damp_c`/`vtdm4` mapping

`deln_flux_transport` (`wp_b_kernels.hpp`'s `has_mass=false` path) is
implemented and GPU-executed in isolation (Part 2), but **not invoked** from
`launch_dgrid_height_transport_cuda`'s production path. A full re-read of
`NumericalConfig` (`include/fv3_gpu_internal.hpp:78-91`) confirms it has no
field named `vtdm4`, nor any field identifiable as the Fortran
`vtdm4`-derived `damp_c` coefficient that `deln_flux_transport` requires.
This is not a resolved translation decision deferred for convenience — it is
a genuine gap: no config plumbing exists yet to carry a `damp_c` value from
GEOS/FV3 configuration into the GPU runtime. Per project instructions ("if a
translation decision can't be resolved from authoritative source, stop and
document the ambiguity rather than inventing physics"), no coefficient was
invented. Resolving this (adding a `damp_c` or `vtdm4` field to
`NumericalConfig` and wiring it through `set_numerical_config`) is left as
an explicit next step, not silently worked around.

## Shared workspace slot: `WorkspaceId::transport`

The pre-existing (unused-by-any-dispatch) `resolve_contract_v3`/
`ContractViews` in `dgrid/wp_b_contract.cpp` claims the same
`WorkspaceId::transport` workspace slice that the new
`bind_wp_b_height_transport` now also claims. Since `resolve_contract_v3` is
never invoked from any dispatch path (confirmed by grep — it has no
callers), there is no runtime conflict today. This is flagged here so a
future milestone that activates `resolve_contract_v3` (WP-B.3/B.4, D-grid
momentum/damping) does not silently double-book the same arena without
noticing.

## Out-of-scope finding: `View3`/`View2` field-order discrepancy in `test_integrated_smoke.cpp`

While confirming the exact field order of `operators::View3`/`View2`
(`operators/wp_d_operators.hpp:20-44` — the true order is `{data, lo_i,
lo_j, lo_k, si, sj, sk}`, i.e. axis 0 is `i`, not `k`), it was discovered
that `tests/integration/test_integrated_smoke.cpp` constructs `View3` for
its `zh_v`/`gz_v` locals using values and comments (`zh_view->lower[2] //
lo_k` first, followed by `lower[0]`/`lower[1]`) that appear transposed
relative to the struct's actual positional order. This file is a
pre-existing Prompt-5B.1-era test, is unrelated to WP-B, and was **not**
modified this milestone — touching it would be unauthorized scope creep
into another work package's test. `wp_b_contract_cuda.cpp`'s own
`to_view3`/`to_view2` helpers (used throughout this milestone's new code)
were independently verified to already use the correct, non-transposed
mapping, so this discrepancy does not affect any WP-B code. It is
documented here purely as a finding for whichever future milestone owns
`test_integrated_smoke.cpp`.

## Pre-existing integration gap: `launch_update_dz_d` had zero callers

Independent of this milestone's changes, `fv3::gpu::vertical::
launch_update_dz_d` was never called from `fv3_gpu_dyn_core_step` or any
prior test — confirmed by grep across the codebase before this milestone's
new test was added. This is a pre-existing gap in the production per-substep
dispatch loop, not something this milestone introduced. Part 3 of the new
smoke test works around it by calling `launch_update_dz_d` directly, which
is sufficient to demonstrate GPU EXECUTED status for the function and its
WP-B dependency, but does **not** wire `update_dz_d` into
`fv3_gpu_dyn_core_step`'s main loop. That remains open.

## Fix Table

| File | Purpose | Numerical semantics changed |
|---|---|---|
| `dgrid/wp_b_cuda.cuh`, `wp_b_cuda.cu` (`ra_xy_kernel`) | Compute `ra_x`/`ra_y` from `area`/`xfx`/`yfx` | N/A (new code — no existing tested implementation to preserve) |
| `dgrid/wp_b_cuda.cu` (`fv_tp_2d_hord6` kernel path) | PPM order-6 Y-X/X-Y transport of `zh` | NO (direct translation of `wp_b_kernels.hpp`'s `fv_tp_2d_hord6`/`xppm_hord6`/`yppm_hord6`, same operator order/constants/limiter branches) |
| `dgrid/wp_b_cuda.cu` (`apply_height_flux_divergence_kernel`) | Apply flux divergence to `zh` in place | N/A (new code, modeled on `apply_mass_and_scalar_flux`'s divergence pattern but for a field with no existing mass-divergence translation) |
| `dgrid/wp_b_cuda.cu` (`deln_flux_transport_kernel`) | Order-`nord` divergence damping on `zh` fluxes, `has_mass=false` | NO (direct translation of `wp_b_kernels.hpp`'s `deln_flux_transport`) |
| `dgrid/wp_b_contract_cuda.{hpp,cpp}` | `FieldRegistry`/workspace binder for the above kernels | N/A (new adapter code, no numerical content) |
| `runtime/fv3_gpu_production_dispatch.cu` | Wire `dgrid_height_transport` to the real kernel | N/A (dispatch wiring only) |
| `CMakeLists.txt` | Build/test targets | N/A (build config only) |
| `tests/dgrid/test_wp_b_cuda_smoke.cpp` | New permanent GPU test | N/A (test code, no numerical semantics) |
| `docs/architecture/dyn_core_cuda_integrated_sequence.md` | Annotate stale claims | N/A (documentation only) |

## Validation Results (A100, Slurm, both R8 and R4)

- R8 CUDA compilation: **PASS**
- R4 CUDA compilation: **PASS**
- Full `ctest` (13/13, both precisions): **PASS**
- `ctest -L gpu` (3/3, both precisions — `fv3_integrated_smoke`,
  `fv3_dispatch_smoke`, `fv3_wp_b_cuda_smoke`): **PASS**
- Direct binary execution, both precisions:
  - Part 1 (`dgrid_height_transport`): **PASS** — finite output, `zh`
    changed vs. pre-call snapshot.
  - Part 2 (`deln_flux_transport`, standalone): **PASS** — finite output.
  - Part 3 (`launch_update_dz_d` cross-WP integration): **PASS** — finite
    output, `zh` changed through the full WP-C → WP-B call chain.
- `compute-sanitizer --tool memcheck`, both precisions: **0 errors**
  (binary located manually at
  `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
  — not on `PATH` under the `nvidia/nvhpc-byo-compiler/24.7` module, by
  design per `discover_platform.md`).
- Numerical validation: **NOT RUN** (no authoritative Fortran reference
  available in this checkout — out of scope for this milestone, belongs to
  Prompt 6).

## Honest Status by Validation Hierarchy

| Component | IMPLEMENTED | COMPILED | GPU EXECUTED | NUMERICALLY VALIDATED |
|---|---|---|---|---|
| `ra_x`/`ra_y` | YES | YES | YES (Part 1) | NOT RUN / NOT POSSIBLE |
| `fv_tp_2d_hord6` on `zh` (`hord_tm=6`, positive only) | YES | YES | YES (Part 1) | NOT RUN / NOT POSSIBLE |
| `fv_tp_2d_hord6` negative-hord clamping | YES | YES | NOT RUN (not exercised) | NOT RUN / NOT POSSIBLE |
| `apply_height_flux_divergence` | YES | YES | YES (Part 1) | NOT RUN / NOT POSSIBLE |
| `deln_flux_transport` (standalone) | YES | YES | YES (Part 2) | NOT RUN / NOT POSSIBLE |
| `deln_flux_transport` (wired into production `dgrid_height_transport`) | NO (blocked on damp_c mapping) | N/A | N/A | N/A |
| `dgrid_height_transport` production dispatch slot | YES | YES | YES (Part 1, Part 3) | NOT RUN / NOT POSSIBLE |
| WP-C `launch_update_dz_d` → WP-B end-to-end | YES (pre-existing, unwired) | YES | YES (Part 3, this milestone) | NOT RUN / NOT POSSIBLE |
| `dgrid_dynamics` (D-grid momentum/vorticity/damping/edges) | NO | N/A | N/A | N/A |
| `dgrid_height` | NO | N/A | N/A | N/A |

## Remaining Blockers

1. **`damp_c`/`vtdm4` mapping ambiguity** — `deln_flux_transport` cannot be
   wired into the production `dgrid_height_transport` path until
   `NumericalConfig` gains a field for this coefficient and GEOS/FV3
   configuration plumbing is extended to populate it.
2. **`launch_update_dz_d` is not called from `fv3_gpu_dyn_core_step`** —
   this is a pre-existing gap (not new to this milestone) in the main
   per-substep production dispatch loop. Wiring it in is a WP-C/runtime
   concern, not a WP-B one, but blocks true end-to-end production execution
   of the acoustic substep sequence.
3. **`dgrid_dynamics` remains entirely unimplemented** — the much larger
   D-grid momentum/vorticity/divergence-damping/contravariant-edge-
   correction stage (WP-B.3/B.4 in the staged plan below) has no CUDA yet.
4. **Fortran cross-check remains impossible** in this checkout — all
   "numerical semantics changed: NO" claims above are relative to the
   existing host C++ translation reference, not authoritative Fortran.

## Staged WP-B CUDA Plan (unchanged from planning, for future milestones)

| Sub-milestone | Scope | Unlocks |
|---|---|---|
| **3A.3 (this milestone, DONE)** | `dgrid_height_transport`: `fv_tp_2d_hord6`, `ra_x`/`ra_y`, `deln_flux_transport` (implemented, not wired) — scoped to `zh` | WP-C `update_dz_d` real GPU execution |
| **WP-B.2** | Generalize transport kernels to arbitrary scalars (`pt`, `w`, `q_con`, `delp`) for `dgrid_dynamics`'s scalar-transport stage; resolve `damp_c`/`vtdm4` mapping | Scalar-transport row of `dgrid_dynamics`; unblocks `deln_flux` wiring |
| **WP-B.3** | D-grid momentum: flux/Courant prep, KE, vorticity, D-wind update, `mix_dp` | Momentum rows of `dgrid_dynamics` |
| **WP-B.4** | Divergence damping (winds), heat accounting, contravariant cubed-sphere edge corrections, corner dependencies | Damping/edge rows; needs `corner_edge_service` (currently a fail-adapter — separate blocker) |
| **WP-B.5** | Wire `dgrid_dynamics` production dispatch slot once WP-B.2-4 are GPU-executed and sanitizer-clean | Full production path incl. WP-B |

## Conclusion

`dgrid_height_transport` is now IMPLEMENTED, COMPILED (R8+R4), and GPU
EXECUTED — both in isolation and, critically, through WP-C's
`launch_update_dz_d` end-to-end — with zero `compute-sanitizer` errors in
either precision. This is the first CUDA code WP-B has ever had. Numerical
validation against authoritative Fortran remains out of scope (source
absent) and is correctly NOT claimed. Two concrete blockers remain before
`deln_flux` can be wired into production (`damp_c` config mapping) and
before `update_dz_d` is reachable from the main per-substep loop (dispatch
wiring, a pre-existing gap). Neither blocks starting the next WP-B
sub-milestone (WP-B.2).

READY FOR NEXT WP-B CUDA SUB-MILESTONE

---

## ANNOTATION — superseded in part by Milestone 3A.3B (2026-08-24)

Added append-only; nothing above has been altered. See
`docs/agent_handoff/milestone_3a3b_wp_b_fortran_reconciliation.md` for evidence.

Three claims above are **superseded**:

1. **"Fortran cross-check: NOT POSSIBLE (source absent)"** — now **false**. The
   authoritative source is available on Discover at the pinned revision
   `GEOS-ESM/GFDL_atmos_cubed_sphere`, tag `geos/v3.0.0`, SHA `ca30e26`.
2. **The `damp_c` / `vtdm4` blocker as stated** — the premise was wrong. The
   `update_dz_d` height path never reaches `deln_flux`; it calls
   `del6_vt_flux` with a raw per-level coefficient. No scalar `damp_c` field is
   needed; a per-level `(damp[npz+1], ndif[npz+1])` pair plus `dp0[npz]` is.
3. **"`ra_x`/`ra_y` is new code, numerical semantics: N/A"** — authoritative
   definitions exist (`nh_utils.F90:269-278`) with narrower extents than the
   CUDA computes.

**`dgrid_height_transport` as delivered in 3A.3 would not pass a numerical
comparison against authoritative Fortran.** 3A.3B identifies 9 discrepancies,
3 critical: `edge_profile` is entirely absent; the `zh` update uses an additive
`rarea` increment instead of the authoritative area-weighted remap; the damping
follows `deln_flux` rather than `del6_vt_flux` conventions.

Its GPU-execution and sanitizer-clean results remain valid as stated — those
were never numerical-validation claims.

**Standing:** the order-6 PPM *interior* arithmetic in `wp_b_cuda.cu` was
verified element-by-element against `tp_core.F90:523-555` and is a correct
translation. The errors are in the surrounding assembly, edges, and corners.

---

## ANNOTATION — repaired by Milestone 3A.4 (2026-08-24)

Added append-only; nothing above has been altered. See
`docs/agent_handoff/milestone_3a4_wp_b_height_transport_repair.md`.

**The `dgrid_height_transport` implementation described in this document no
longer exists in the working tree.** Milestone 3A.4 replaced it with the
authoritative `update_dz_d` transport body (`nh_utils.F90:236-325`):
`edge_profile` → `ra_x`/`ra_y` → per-level damped/undamped split → `fv_tp_2d`
→ `del6_vt_flux` → area-weighted `zh` update. Treat every algorithmic
description of the height path above as **historical only**.

Specifically superseded:

1. **"`fv_tp_2d_hord6` + flux divergence on `zh`"** as the operator definition
   — the authoritative update is an area-weighted remap renormalised by
   `((ra_x+ra_y) - area)`, not an additive `rarea` increment.
2. **`edge_profile`'s absence** — now implemented (gam + x + y launchers) and
   checked against an independent host transcription of the non-uniform
   tridiagonal branch.
3. **The `deln_flux` damping convention on the height path** — replaced by
   `del6_vt_flux` with the raw per-level `damp(k)`. A bitwise linearity test
   (Part G1) now guards against the substitution recurring.
4. **`ra_x`/`ra_y` extents** — corrected to the Fortran's (they differ from
   each other, and both are narrower than 3A.3 computed).
5. **The 3A.3 smoke fixture** — rewritten. It registered only 11 fields and no
   `npx`/`npy`, and is *correctly rejected* by the 3A.4 binder. Binder
   validation was not weakened to keep it.
6. **`ng` conflated with `halo_width`** — separated; `ng` is a Fortran
   compile-time PARAMETER `= 3` (`tools/fv_mp_mod.F90:108`).

The `damp_c`/`vtdm4` "open ambiguity" section above is fully closed, not merely
superseded: there was no coefficient to map, because the height path never
calls `deln_flux`.

As with the 3A.3B annotation, this document's GPU-execution and
sanitizer-clean claims were never numerical-validation claims and remain
accurate as historical statements about the code as it stood then. Milestone
3A.4 re-established both, on the repaired code, in R8 and R4.

**Numerical validation remains NOT RUN** after 3A.4 as well.
