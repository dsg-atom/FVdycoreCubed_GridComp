# Milestone 3A.4 — Authoritative WP-B Height Transport Repair

**Date:** 2026-08-24
**Branch:** `cuda/gpu-bringup`
**Starting HEAD:** `3b5ed92` (Reconcile WP-B CUDA transport with authoritative
FV3 Fortran)
**Working-tree state at start:** Steps 1–5 of this milestone were already
implemented and uncommitted on top of `3b5ed92`, and both R8 and R4 compiled.
That work was preserved, not reset.
**Final commit:** "Complete authoritative WP-B height transport repair" — the
single milestone commit on `cuda/gpu-bringup` whose parent is `3b5ed92`. This
document is part of that commit, so it cannot name its own SHA; resolve it with
`git log --oneline 3b5ed92..cuda/gpu-bringup`. **Not pushed.**

**Authoritative Fortran revision used:** `GEOS-ESM/GFDL_atmos_cubed_sphere`,
tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`, read-only
clone at `/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`.
This closes 3A.3's "Fortran cross-check: NOT POSSIBLE (source absent)".

**Numerical validation: NOT RUN.** Nothing in this document is authoritative
numerical validation. No Fortran-generated savepoint was produced or compared.
That is Prompt 6. What this milestone establishes is: source-level agreement
with the authoritative Fortran, GPU execution on a real A100 in both
precisions, sanitizer cleanliness, and targeted agreement between individual
CUDA stages and independent host transcriptions of the corresponding Fortran.

---

## Objective

Milestone 3A.3 shipped the first WP-B CUDA port of `dgrid_height_transport`
without access to the authoritative Fortran. 3A.3B (audit-only) then obtained
the Fortran and found that 3A.3's implementation was **not** `update_dz_d`:
it applied an additive flux-divergence increment using `deln_flux`'s damping
form, and omitted `edge_profile` entirely. Milestone 3A.4 replaces that path
with the authoritative `update_dz_d` transport body and validates it.

The authoritative sequence (`model/nh_utils.F90:236-325`) is:

```
edge_profile            -> interface Courant numbers / mass fluxes
ra_x / ra_y             -> swept-area denominators (different extents each)
per-level split on damp(k) > 1.E-5
  damped:   z2 = zh ; fv_tp_2d(z2) ; del6_vt_flux -> fx2/fy2
  undamped: fv_tp_2d(zh) in place
area-weighted zh update, renormalised by ((ra_x+ra_y) - area)
```

The `ws` / `dz_min` monotonicity closure (`nh_utils.F90:307-322`) is **not**
part of this operator; it belongs to WP-C's height boundary kernel, which
calls this service and then finishes the column.

---

## Steps 1–5 (implemented before this session's continuation, retained)

| Step | Change |
|---|---|
| 1 | Corrected `ra_x`/`ra_y` extents and parenthesisation to the Fortran's (`ra_x` on `is..ie × jsd..jed`, `ra_y` on `isd..ied × js..je`). |
| 2 | Rewrote `wp_b_cuda.cu` as the authoritative transport body: `edge_profile` (gam + x + y), `copy_corners`, `fv_tp_2d`, `del6_vt_flux`, `copy_z2`, area-weighted `zh` update. |
| 3 | Rewrote the launcher declarations in `wp_b_cuda.cuh`, adding `LevelFilter` (per-level branch predicate) and `HeightTransportBounds`. |
| 4 | Rewrote `bind_wp_b_height_transport` and the orchestration in `wp_b_contract_cuda.cpp`; grew the bound view set from 11 to 14 and the scratch arena to 17 slots. |
| 5 | Added the `write_delz` flag to WP-C's `height_boundary_v2_kernel` (see the `write_delz` finding below). |

Additionally, and central to correctness:

**`ng` was separated from `halo_width`.** `ng` is a compile-time PARAMETER
`= 3` in the authoritative Fortran (`tools/fv_mp_mod.F90:108`) that
`tp_core.F90` and `sw_core.F90` `use` directly, so every `isd/ied/jsd/jed`
loop bound and `copy_corners`' corner-block size is fixed at 3 regardless of
how much halo the caller registered. Before 3A.4 this code conflated the two,
so a caller registering `halo_width = 4` would have had four rings of corner
cells rotated where the Fortran rotates three. `HeightTransportBounds` now
carries both, with `halo_width` used only for arena geometry and bounds
validation and never as a Fortran loop bound.

---

## PRIORITY 1 — the `damping_ndif` type-contract hazard: CLOSED

### The hazard

`DeviceViewBase::stride` is expressed in **elements**, so nothing in the
struct recorded how wide an element actually was. For `LogicalType::integer`
fields this was exploitable: the public C ABI offers only
`FV3_GPU_ELEMENT_F32`/`F64` (`fv3_gpu_api.h`), so integral field values must
travel as reals, and `make_device_view`'s precision check degenerates to
`d.element_bytes != d.element_bytes` — a tautology — for that logical type.
A caller could therefore register `damping_ndif` with `element_bytes = 4`
inside an R8 build, and WP-B's `View1<const fv_real>` would reinterpret that
4-byte storage as 8-byte doubles. `damping_ndif` selects the del-2/4/6
damping order per level, so the consequence is silent wrong damping, not a
crash.

This was **not** left as a documented limitation.

### The resolution (minimal, no ABI redesign)

1. `DeviceViewBase` gained `uint32_t element_bytes = 0`
   (`include/fv3_gpu_internal.hpp`), documented as the registered storage
   width, with `0` meaning "not recorded" for backward compatibility with a
   default-constructed view that never went through `make_device_view`.
2. `make_device_view` populates it (`runtime/fv3_gpu_contract.cpp:76`), with a
   comment explaining precisely why it is the *only* record a consumer can
   validate against for the integer logical type.
3. A small predicate `view_element_width_is(view, expected_bytes)` was added
   next to the struct.
4. `bind_wp_b_height_transport` validates `damping_ndif` (and, defensively,
   `damping_v` and `dp_ref`) **before** constructing any
   `View1<const fv_real>`, and returns `FV3_GPU_INVALID_ARGUMENT` with a
   contract message on a mismatch
   (`dgrid/wp_b_contract_cuda.cpp:216-227`).
5. `internal_contract_version` was bumped `3 -> 4`.

No memory is reinterpreted. No storage is silently cast. The public C ABI was
not redesigned.

### FUTURE CONTRACT WORK (deliberately out of 3A.4 scope)

This is a **storage-contract workaround for the current ABI's lack of an
integer element type**. It is *not* an assertion that integer field semantics
should permanently travel in `fv_real` storage. The correct long-term fix is a
real integer element type in the public C ABI
(e.g. `FV3_GPU_ELEMENT_I32`) plus a `View1<const int32_t>` path for
`LogicalType::integer` fields, so the width check stops being the only line of
defence. `include/fv3_gpu_internal.hpp` forward-references this note. Doing it
here would have expanded the milestone into a public-ABI change affecting
every registered field and every caller.

---

## Smoke-fixture corrections

### `tests/dgrid/test_wp_b_cuda_smoke.cpp` — rewritten

The 3A.3 fixture was genuinely invalidated by the repair: it registered only
the 11 fields the old (incorrect) flux-divergence update read, and never
supplied `dxa`, `dya`, `dp_ref`, `damping_v` or `damping_ndif`, nor `npx`/`npy`.
Under the 3A.4 binder it is rejected — correctly.

Binder validation was **not** weakened to accommodate it. The fixture was
rewritten to conform to the production contract: it registers all 17 fields,
sets `NumericalConfig` to the production first-run values (`hord_tm/mt/vt = 6`,
`hord_dp = -6`, `grid_type = 0`, `hydrostatic = false`, `a_imp = 1.0`), and
establishes `npx`/`npy` through the real `set_topology` path rather than by
poking the struct. Values are physically reasonable and non-degenerate
(smoothly varying analytic `area`/`dxa`/`dya`/`zh`/`crx`/`xfx`, and a
deliberately **non-uniform** `dp_ref(k) = 700 + 900k` so `edge_profile`'s
non-uniform tridiagonal branch is the one exercised).

The fixture also installs a fail-adapter `corner_service` that returns
`FV3_GPU_UNSUPPORTED_CONFIGURATION`, so the tests prove WP-B never calls
`corner_edge_service`.

### `tests/integration/test_dispatch_smoke.cpp` — no change needed

The governing instruction assumed both smoke tests were invalidated. Only one
was. `fv3_gpu_dyn_core_step`'s stage array is
`{cgrid_pressure, vertical_solver, nh_pressure, post_loop_heating}` and never
reaches `dgrid_height_transport`, so this fixture never binds WP-B.
It passed **unmodified** in both precisions (confirmed empirically, not
assumed — 13/13 CTest in R8 and R4).

---

## Step 6 — behavior test inventory (Parts A–J)

`tests/dgrid/test_wp_b_cuda_smoke.cpp`. Every part checks finite output,
non-degenerate change, correct halo/edge write extents, absence of unexpected
mutation of read-only inputs, and expected error handling. Fixtures were
chosen to avoid passing by exact cancellation (e.g. Part H keeps
`(ra_x + ra_y) - area` away from zero; Part G counts only nonzero flux cells).

| Part | Scope | R8 | R4 |
|---|---|---|---|
| A | Interior order-6 transport, no panel edge in play (`is=9..20`, `npx=npy=33`) | PASS — 1296/1296 interior cells changed, **0** writes outside `[is..ie]×[js..je]`, `crx`/`area`/`dp_ref` bitwise unchanged | PASS — 1292/1296 (float rounding; threshold 4/5) |
| B | Panel-edge PPM: `cubed=1` vs `cubed=0`, identical inputs | PASS — 567 near-seam differences, deepest interior bit-identical | PASS — 350 near-seam differences, deepest interior bit-identical |
| C | `copy_corners` dir=1 and dir=2 vs host transcription of `tp_core.F90:257-318` | PASS — 324 cells rotated per direction (expected `4·ng·ng·(npz+1)` = 324), 0 mismatches | PASS — identical counts |
| D | `edge_profile` non-uniform tridiagonal vs independent host transcription | PASS — 135/135 sampled interfaces agree | PASS — 135/135 |
| E | `damp(k) <= 1.E-5` branch: no damped-only workspace may be touched | PASS — no damped-only slot written | PASS |
| F | `damp(k) > 1.E-5` branch, incl. the `kk = min(k, npz-1)` parity profile | PASS — 9/9 levels written for `z2` and `fx2`; alternating profile as expected | PASS |
| G | `del6_vt_flux`: G1 exact linearity in `damp`; G2 `nord=0` vs host reference; G3 `nord=2` differs from `nord=0` | PASS — G1 0 deviations over 1404 nonzero flux cells; G2 2808/2808 faces agree, `q` bitwise unchanged; G3 1404 differing faces | PASS — identical counts |
| H | Area-weighted `zh` update, both branches in one launch, vs host transcription of `nh_utils.F90:288-303` | PASS — 1296/1296 cells agree (4 damped, 5 undamped levels), halo untouched | PASS |
| I | Cross-WP: WP-C `launch_update_dz_d` → corrected WP-B service | PASS — `zh` changed at 1296 cells, `ws` written 144/144, monotonicity `zh(k) >= zh(k+1) + dz_min` holds with 0 violations, `delz` sentinel retained | PASS — 1284 cells, `ws` 144/144, 0 violations |
| J | Storage/topology contract rejection | PASS — native width accepted; other width, absent `npx`/`npy`, corner-flag-without-cubed-sphere all rejected `INVALID_ARGUMENT`; `nord_v = 3` rejected `UNSUPPORTED_CONFIGURATION`; `cudaGetLastError() == cudaSuccess` in every rejection case (nothing launched) | PASS |

**G1 is the discriminating check.** `del6_vt_flux` (`sw_core.F90:1590`) is
exactly linear in the raw `damp`; `deln_flux` (`tp_core.F90:1263`) is not, as
it uses `(damp_c*da_min)**(nord+1)`. Doubling `damping_v` and requiring
`fx2(2c) == 2·fx2(c)` bitwise is therefore a direct test that the zh path
calls the right Fortran routine — the exact error 3A.3 made.

**Part J2 note.** Registering `damping_ndif` with the other precision's width
makes `initialize_state` copy `npz × that width` bytes out of the host buffer,
so the fixture's `ndif` vector is deliberately oversized (`npz × 2` elements).
That keeps J2 a *device-side contract violation to be rejected* rather than a
host out-of-bounds read inside the test itself.

**Validation level of these parts:** GPU EXECUTED, plus targeted host-reference
agreement for the individually transcribed stages (`edge_profile`,
`copy_corners`, `del6_vt_flux` at nord=0, the `zh` update). The host references
are independent transcriptions of the authoritative Fortran written for this
test; they are **not** Fortran-generated savepoints, so this is not
authoritative numerical validation.

---

## Section 6 — source-level Fortran reconciliation: COMPLETE

Eight items, each classified against SHA `ca30e26`:

| # | Item | Classification |
|---|---|---|
| 1 | `ra_x`/`ra_y` extents + parenthesisation | MATCHES FORTRAN |
| 2 | PPM interior `al` stencil + mord=6 flux / `smt5` | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE |
| 3 | Panel-edge one-sided `al` + WMP `smt5` edge fix | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE |
| 4 | `edge_profile` non-uniform branch | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE |
| 5 | `copy_corners` (8 formulas, 4 ranges, guard) | MATCHES FORTRAN |
| 6 | `del6_vt_flux` (raw damp, extents, signs, nord loop, corners, no folding) | MATCHES FORTRAN |
| 7 | damped/undamped split + `damp(km+1)`/`ndif(km+1)` prologue | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE |
| 8 | Area-weighted `zh` update, both branches, denominator | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE |

**Known discrepancies remaining inside 3A.4 scope: NONE.**

### The explained structural differences (all semantics-preserving)

- **(2)** Fortran tabulates `bl`/`br`/`b0`/`smt5` once per row; CUDA recomputes
  `al` per face. Same expressions, same operand order, bitwise equal.
- **(3)** CUDA's `i <= 2` / `i >= npx-1` predicates cover exactly the index
  sets Fortran's clipped interior loop leaves to its edge block.
- **(4)** `gam` depends only on `dp_ref` and `k`, so it is hoisted into one
  i-independent column kernel; the tridiagonal sweeps then run in place on
  `q1e`/`q2e`.
- **(7)** Fortran's host-side `if` becomes a device-side per-level predicate
  (`LevelFilter`) with disjoint damped/undamped slabs, and `min(k, npz-1)`
  indexing reproduces the `damp(km+1) = damp(km)` prologue without requiring
  the caller to register `npz+1` damping entries or letting the GPU mutate an
  immutable grid field.
- **(8)** Structural only after defect D1 was fixed (below).

### In-scope defects found and closed during reconciliation

| ID | Defect | Resolution |
|---|---|---|
| D1 | The `zh` update numerator was reassociated (~1 ulp), which would have blocked any future bitwise validation | Restored Fortran's left-to-right `((q*a + fx_div) + fy_div)` |
| D2 | The `del6_vt_flux` nord-group loop was capped at `cfg.nord_v`, which defaults to `0` and which **no current caller populates** — silently zeroing all del-4/del-6 height damping while passing every contract check | Loop is now driven by `ng - 1 == 2` (the largest nord the port can represent); the authoritative per-level order is the registered `damping_ndif` array, and `cfg.nord_v` only *constructs* that array in Fortran (`dyn_core.F90:713`, `nord_v(k) = min(2, nord)`). Binder tightened to `0 <= nord_v <= 2`. |
| D3 | `launch_deln_flux_transport_cuda` omits the `2 + 2·nord` `copy_corners` rotations of `d2`, so it is correct only at `nord == 0` | Fenced with an early `UNSUPPORTED_CONFIGURATION`. Fixing it is WP-B.2 work; it is not on the height path and is never invoked from it. |
| D4 | A caller could set corner flags while clearing `global_cubed_sphere` (or requesting `nested`/`grid_type >= 3`), silently applying interior stencils at a seam | Binder rejection added; covered by Part J4 |
| D6 | Stale Fortran citations in comments | Corrected |

D5 remains informational and unreachable on the current path.

Also verified during reconciliation: no out-of-bounds access, no read of an
unwritten level, all 17 arena slots distinct, every launch on the single
borrowed compute stream, and the `fv_tp_2d` call order exact against
`tp_core.F90:163-234`.

---

## Section 9 — the `write_delz` finding

`launch_update_dz_d` passes `write_delz = false` to
`height_boundary_v2_kernel`. This is consistent with the authoritative call
paths: in `nh_utils.F90`'s `update_dz_d`, `delz` is `intent(out)` but is
**never assigned** — the Riemann solver derives `delz` from the final `zh`.

This was **not** changed for test convenience. Part I asserts the opposite
direction: `delz` must come back still holding its pre-call `-999` sentinel.
That assertion exists specifically to prevent a future change from quietly
making the CUDA path write `delz` where the Fortran does not, and the test
carries a comment saying not to relax it.

---

## Section 10 — scope discipline

No work was started on full `d_sw` CUDA, momentum/vorticity transport, the
complete `dgrid_dynamics` slot, full divergence damping, or a general
corner-edge architecture for all WP-B slices. `dgrid_dynamics` and
`dgrid_height` remain explicitly `nullptr` in
`runtime/fv3_gpu_production_dispatch.cu`.

---

## Build, execution and sanitizer results

Environment per `docs/agent_instructions/discover_platform.md`: GCC 12.3.0,
OpenMPI 4.1.6, CUDA 12.5 (`nvidia/nvhpc-byo-compiler/24.7`), CMake 3.30.3.
Configure/compile on the login node; all GPU execution through Slurm on
`gpu_a100` / `--constraint=rome` / 1 GPU / 60G / 12 CPUs. No CUDA executable
was run on a login node.

| Item | R8 | R4 |
|---|---|---|
| CUDA compilation | PASS | PASS |
| A100 GPU execution (`fv3_wp_b_cuda_smoke_test`, node `warpa003`) | PASS — all 10 parts, exit 0 | PASS — all 10 parts, exit 0 |
| Full CTest suite | 13/13 PASS | 13/13 PASS |
| `compute-sanitizer --tool memcheck` | PASS — 0 errors, exit 0 | PASS — 0 errors, exit 0 |
| `compute-sanitizer --tool initcheck` | PASS — 0 errors, exit 0 | PASS — 0 errors, exit 0 |
| `compute-sanitizer --tool racecheck` | PASS — 0 hazards (0 errors, 0 warnings) | PASS — 0 hazards |

`compute-sanitizer` is at
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
(not on `PATH`). All runs used `--error-exitcode 9`; the test's own
`RESULT: PASS (all 10 parts)` line was confirmed present under every tool, so
the sanitizers were exercising a fully passing run rather than an early exit.
Sanitizer attention was directed at exactly the risk areas named in the task:
shifted/negative lower bounds, panel-edge accesses, `ra_x`/`ra_y` extents,
`copy_corners`, `edge_profile`, `damping_v`, `damping_ndif`, `dp_ref` and the
per-level arrays.

### One initcheck finding, in the test harness, resolved

The first `initcheck` run reported 324 (R8) / 243 (R4) errors. Full
classification with `--print-limit 4000` showed **every one** was a host-side
`cudaMemcpy` in `Fixture::get_arena` called from `part_d`, and that there were
**zero** device-side uninitialized `__global__` reads. Part D pulled a whole
arena slot back to the host while `launch_edge_profile_x_cuda` legitimately
writes only `is..ie+1 × jsd..jed` of it, so the untouched remainder of the
workspace allocation was uninitialized device memory.

Fixed by zeroing the arena slots in Part D before the launch (a `zero_arena`
helper), which is harness hygiene, not masking: initcheck independently
confirmed no kernel reads uninitialized memory, the parts that probe "was this
level written?" overwrite those zeros with an explicit sentinel afterwards, and
no assertion was weakened. The fix was deliberately **not** applied to Parts
E–H, whose slots initcheck showed are fully written by the kernels — leaving
them un-zeroed preserves that as evidence. After the fix all six sanitizer
runs are clean.

---

## Section 12 — change accounting

| File | Reason | CUDA behavior changed | Authoritative Fortran semantics changed |
|---|---|---|---|
| `dgrid/wp_b_cuda.cu` | Replace the 3A.3 additive flux-divergence update with the authoritative `update_dz_d` body (`edge_profile`, `copy_corners`, `fv_tp_2d`, `del6_vt_flux`, `copy_z2`, area-weighted update); fix D1 operand order | YES | NO |
| `dgrid/wp_b_cuda.cuh` | New/changed launcher declarations, `LevelFilter`, `HeightTransportBounds` with `ng` distinct from `halo_width`, `nord_cap` contract, D3 fence | YES | NO |
| `dgrid/wp_b_contract_cuda.cpp` | Bind 14 views / 17 arena slots; `damping_ndif` width guard; `npx`/`npy`, corner-flag and `nord_v` rejections; drive the nord loop from `ng - 1` (D2) | YES | NO |
| `dgrid/wp_b_contract_cuda.hpp` | Grow the view/arena contract and document the added fields | YES (contract) | NO |
| `include/fv3_gpu_internal.hpp` | Add `DeviceViewBase::element_bytes`, `view_element_width_is`, `TopologyView::npx/npy`, `set_topology`; bump contract version 3→4 | Contract behavior changed: YES — incompatible storage is now rejected. Numerical algorithm changed: NO | NO |
| `runtime/fv3_gpu_contract.cpp` | Populate `element_bytes` in `make_device_view` | Contract behavior changed: YES — incompatible storage is now rejected. Numerical algorithm changed: NO | NO |
| `runtime/fv3_gpu_runtime.cpp` | Add `set_topology` so `npx`/`npy` can be established through a real setter | NO (new setter) | NO |
| `runtime/fv3_gpu_production_dispatch.cu` | Comment only: describe the authoritative path; remove the stale "damp_c mapping ambiguity" reference, which 3A.4 resolved (there was no coefficient to map — the zh path uses `del6_vt_flux`, not `deln_flux`) | NO | NO |
| `vertical/fv3_vertical_kernels.cu` | Add the `write_delz` flag; pass `false` per the authoritative call path | NO (preserves existing behavior explicitly) | NO |
| `tests/dgrid/test_wp_b_cuda_smoke.cpp` | Rewritten as the Parts A–J behavior test conforming to the production contract | Test only | NO |
| `CMakeLists.txt` | Comment only: the test is 3A.4 Parts A–J, not 3A.3 Part 3 | NO | NO |

**Every numerical file: authoritative Fortran semantics changed — NO.** No
scientific algorithm was altered to satisfy a compiler, and no CUDA numerical
algorithm was altered for compiler convenience. The numerical changes in
`wp_b_cuda.cu` move the CUDA path *toward* the authoritative Fortran, not away
from it.

---

## Known remaining WP-B work

- **WP-B.2** — generalize the transport kernels to arbitrary scalars (`pt`,
  `w`, `q_con`, `delp`) for the D-grid scalar-transport stage. Includes fixing
  D3 (`deln_flux`'s missing `2 + 2·nord` `copy_corners` rotations of `d2`),
  currently fenced.
- **WP-B.3** — D-grid momentum: flux/Courant prep, kinetic energy, relative
  vorticity, D-wind update, `mix_dp`.
- **WP-B.4** — divergence damping on winds, heat accounting, contravariant
  cubed-sphere edge corrections. Needs a real `corner_edge_service`, which is
  still a fail-adapter — a separate blocker.
- **WP-B.5** — wire the `dgrid_dynamics` production dispatch slot once
  WP-B.2–4 are GPU-executed and sanitizer-clean.
- **Contract** — a real integer element type in the public C ABI, so the
  `damping_ndif` width guard stops being the only line of defence (see FUTURE
  CONTRACT WORK above).
- **Prompt 6** — authoritative numerical validation against Fortran savepoints.
  Not started. The savepoint locations named in
  `docs/agent_instructions/project_architecture.md` (including "after
  vertical/`update_dz_d`") apply directly to this operator.

---

## Status summary

```
IMPLEMENTED:           YES (authoritative update_dz_d transport body)
COMPILED:              R8 PASS, R4 PASS
GPU EXECUTED:          R8 PASS, R4 PASS (A100, warpa003, via Slurm)
MEMORY-SANITIZED:      R8 PASS, R4 PASS (memcheck, initcheck, racecheck)
NUMERICALLY VALIDATED: NOT RUN
```

The last line is not a formality. Parts A–J and the Section 6 reconciliation
establish source-level agreement and correct GPU execution; they do not
establish numerical equivalence with the authoritative Fortran, which requires
Fortran-generated reference data.
