# Implementation Reality Audit — Reconciliation Report

**Date:** 2026-08-21
**Branch:** cuda/gpu-bringup
**Type:** AUDIT ONLY — no source modified, no commits, no pushes.

**Purpose:** Reconcile the prior claim ("Production first-run numerical path:
COMPLETE", 54 stages implemented across WP-E/A/B/C/D) against the actual state
of the source tree, using direct inspection of definitions (not filenames or
declarations) and git history archaeology across all WP branches.

---

## Part 1 — Stage Inventory by Work Package

### WP-A (cgrid/) — 6 stages actually itemized in docs (doc claims 7)

| stage | file | classification | called from integrated path | GPU executable today |
|---|---|---|---|---|
| d2a2c_vect | fv3_cgrid_d2a2c.cu | **REAL CUDA KERNEL** (7 `__global__` kernels, Fortran-faithful) | NO — no caller anywhere; the `_v3` adapter `launch_d2a2c_vect_v3` is declared but **never defined** | NO |
| c_sw | fv3_cgrid_csw.cu | **REAL CUDA KERNEL** (10 `__global__` kernels) | NO — no `_v3` adapter exists at all | NO |
| divergence_corner | fv3_cgrid_divergence.cu | **REAL CUDA KERNEL** (7 `__global__` kernels) | NO — same, no adapter | NO |
| update_dz_c | fv3_cgrid_dz.cu | **REAL CUDA KERNEL** (6 `__global__` kernels) | NO — same | NO |
| p_grad_c | fv3_cgrid.cu | **REAL CUDA KERNEL** + a genuinely complete `_v3` adapter (`launch_cgrid_pressure_v3`, fv3_cgrid.cu:78-127) — the one WP-A stage with a real adapter | Adapter exists and is correct, but `NumericalDispatch.cgrid_pressure` is **never assigned to it anywhere in production code** (only a no-op test double in `test_contract.cpp`) | NO |
| C-grid predictor orchestration | fv3_cgrid_predictor.cu:100-131 (`launch_cgrid_predictor_v3`) | **STUB** — resolves ~35 fields, validates config, then unconditionally `return fail(..., "C-grid predictor workspace resolution not yet integrated")`. Never calls any of the 5 kernels above. | NO | NO |

### WP-B (dgrid/) — 16 stages itemized in docs (doc claims 18)

| stage | classification | evidence |
|---|---|---|
| fv_tp_2d / fv_tp_2d_hord6 / PPM order-6 building blocks / ra_x/ra_y / deln_flux / D-grid damping / contravariant edge corrections | **HOST C++ IMPLEMENTATION** for all of them | `wp_b_kernels.hpp` (1511 lines): `grep -c "__global__"` = **0**, `grep -c "__device__"` = **0**. Directory's own README: *"currently host-callable reference implementations, not submitted CUDA kernels."* Zero `.cu` files exist in `cuda/dyn_core/dgrid/`. |
| corner_edge_service | **CONTRACT/DISPATCH SLOT ONLY** (audit's "does not exist" claim is slightly wrong — it exists as a declared function-pointer field, `fv3_gpu_internal.hpp:199`) | Never assigned in production; only a no-op test double in `test_contract.cpp` |
| dgrid_dynamics / dgrid_height_transport dispatch slots | **CONTRACT/DISPATCH SLOT ONLY** | Default `nullptr`, never assigned in production |

Confirmed: WP-B contains zero executable CUDA kernels — this holds not just
today but across its entire git history (see Part 4).

### WP-C (vertical/) — 7 stages itemized in docs (doc claims 9)

| stage | classification | called from integrated path | GPU executable today |
|---|---|---|---|
| sim1_column, sim3_column/sim3p0/rim_2d | **HEADER/INLINE MATH ONLY** (`__host__ __device__` templates, numerically complete, real formulas) — but they execute as real device code *inside* riem_solver3_kernel | NO | Only reachable via the orphaned launcher below |
| riem_solver3 (`riem_solver3_kernel` + `launch_vertical_solver`) | **REAL CUDA KERNEL** — the single strongest CUDA implementation found in the whole audit: genuine `__global__`, one thread/column, correct field resolution | NO — zero callers anywhere outside its own file and its own unit test | NO |
| update_dz_d | **CUDA LAUNCHER/ADAPTER ONLY**, non-functional in practice — real launcher exists, but hard-gates on `dispatch.dgrid_height_transport` which is null in every production path → immediately returns `FV3_GPU_UNSUPPORTED_CONFIGURATION` | NO (also has zero callers) | NO |
| C-grid vertical entry (`launch_cgrid_vertical_solver`) | **STUB** — literally `return launch_vertical_solver(x, error);` with a comment "for now, delegate to the same Riemann solver" | NO | NO |
| vertical workspace layout | **HOST C++ IMPLEMENTATION**, real and correct | used only by the orphaned launcher | N/A |

### WP-D (operators/) — 10 stages itemized in docs (matches doc claim of 10)

| stage | wired path | orphaned CUDA equivalent |
|---|---|---|
| pe/pk3/pln halo reconstruction | **HOST C++ IMPLEMENTATION**, wired into `launch_nh_pressure` | REAL CUDA KERNEL exists (`launch_reconstruct_pe_halo_cuda` etc.), zero callers except smoke test |
| a2b_ord4/ord2 (periodic + global) | **HOST C++**, wired | REAL CUDA KERNEL exists (~12 kernels), orphaned |
| nh_p_grad / pressure_gradient | **HOST C++**, wired | REAL CUDA KERNEL exists, orphaned |
| one_grad_p / grad1_p_update | **HOST C++ IMPLEMENTATION, dead code** — explicitly rejected by a `config.beta != 0` guard before it's ever reached | n/a |
| del2_cubed / hyperdiffusion | **HOST C++**, wired | REAL CUDA KERNEL exists, orphaned |
| diffusive heating | **HOST C++**, wired | REAL CUDA KERNEL exists, orphaned |
| geopotential from height | **HOST C++**, wired | REAL CUDA KERNEL (`launch_geopotential_from_height_cuda`) — exercised **only** by the standalone smoke test, which bypasses the contract layer entirely |
| edge/corner post-loop request builders | **HOST C++ IMPLEMENTATION**, real, wired | N/A |
| Ray_fast | **MISSING** — docs themselves say "optional adapter not implemented" | — |

WP-D's `wp_d_cuda.cu` alone contains ~30 real `__global__` kernels. None are
reachable via `launch_nh_pressure`/`launch_post_loop_heating` (the functions
the dispatch table would actually call) — those call the plain host C++
versions in `wp_d_operators.cpp` instead.

### WP-E (runtime/interface) — 10 stages itemized (matches doc claim of 10)

| stage | classification |
|---|---|
| GPU context, FieldRegistry, device state H2D/D2H, persistent workspace, streams, public C ABI | **HOST C++ IMPLEMENTATION — genuinely real**, driving actual `cudaMalloc`/`cudaMemcpyAsync`/`cudaStreamCreate`; this part of WP-E's claim is largely accurate |
| NumericalConfig | **HEADER/DATA STRUCT ONLY**, unpopulated by any caller |
| NumericalDispatch (dispatch metadata) | **CONTRACT/DISPATCH SLOT ONLY** — struct + `validate_dispatch()` are real, but a `NumericalDispatch` instance with real function pointers is constructed nowhere in the entire repo except one unit test, which fills every slot with a no-op dummy |
| halo/communication interfaces | **CONTRACT/DISPATCH SLOT ONLY** — headers/enums only, no MPI exchange implementation |
| `fv3_gpu_dyn_core_step` (the actual production entry point) | **STUB** — full body: `if(ev)*ev=0; return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__, "numerical work packages are not linked into this skeleton");` It touches neither `NumericalDispatch` nor any WP symbol. |

---

## Part 2 — Reconciliation with the Prior 54-Stage Inventory

**Where "54" came from, and why it doesn't even reconcile with itself:**
`docs/architecture/dyn_core_cuda_integrated_sequence.md` itemizes only **49**
distinct rows (WP-E:10, WP-A:6, WP-B:16, WP-C:7, WP-D:10) — its own summary
table's "54" doesn't match its own itemized rows. That is a bookkeeping error,
separate from the deeper problem below.

**What "COMPLETE" meant in that table:** every single one of the 49 rows
carries the identical status quadruple: `Status=COMPLETE, R4=PASS, R8=PASS,
CUDA=NOT RUN, GPU=NOT RUN, Savepoint=NOT RUN`. So "COMPLETE" was operationally
defined as *host C++ reference implementation exists and passes a CPU unit
test* — CUDA compilation and GPU execution were explicitly marked NOT RUN in
the same rows.

**The JSON contract inventory is narrower and honest about its scope:** it
only claims `"contract_available": "yes"` (field/geometry/workspace/halo
metadata exists), and its own prose says exactly that — it never asserts
kernel existence or GPU execution. It did not conflate contract vs. kernel.

**The conflation happened one level up, in the summary language, not the
tables:** `integration.md:184-187` explicitly states `CUDA compilation: NOT
RUN / GPU execution: NOT RUN / Numerical validation: NOT RUN` — then the same
document, 285 lines later:

> `integration.md:469-477`: **"Production GEOS first-run numerical path:
> COMPLETE. All 54 required numerical stages ... are now implemented ... No
> PARTIAL or UNIMPLEMENTED stages remain."**

That claim, per this project's own IMPLEMENTED / COMPILED / GPU EXECUTED /
NUMERICALLY VALIDATED hierarchy, reads as asserting production-readiness but
is supported only at the IMPLEMENTED tier at best — and the code audit above
shows even that was overstated for WP-B (never real CUDA, ever) and for every
orchestration/dispatch layer (stub or unpopulated everywhere).

### Totals (code-reality classification across all 5 WPs)

| classification | count (approx., by stage-cluster) |
|---|---|
| REAL CUDA KERNEL (all orphaned/unreachable from the entry point) | 12 clusters (WP-A: 5, WP-C: 1, WP-D: 6) — ~76 individual `__global__` kernels total across the tree |
| CUDA LAUNCHER/ADAPTER ONLY (real but unreachable) | 2 (WP-A p_grad_c adapter; WP-C update_dz_d, gated non-functional) |
| HOST C++ IMPLEMENTATION | ~28 (WP-B ~16, WP-D ~8 incl. dead code, WP-E ~6 infra pieces, WP-C workspace layout) |
| HEADER/INLINE MATH ONLY | ~5 (WP-C sim1/sim3/rim_2d column routines) |
| CONTRACT/DISPATCH SLOT ONLY | ~7 (corner_edge_service, dgrid_dynamics, dgrid_height_transport, NumericalDispatch itself, NumericalConfig, halo/comm interfaces) |
| STUB | 3 (WP-A predictor orchestration, WP-C C-grid-vertical alias, WP-E `fv3_gpu_dyn_core_step` — the decisive one) |
| MISSING | 1 (Ray_fast) |

**The single fact that dominates every other number in this table:**
regardless of any WP's internal state, `fv3_gpu_dyn_core_step` — the one
function GEOS/Fortran would actually call — unconditionally fails and touches
nothing. So "called from integrated path: YES" is false for every numerical
stage in every WP, full stop. There is no integrated path today.

---

## Part 3 — Answers to the Specific Questions

**1. Does WP-B truly contain zero executable CUDA numerical kernels?**
Yes, confirmed — zero `.cu` files, zero `__global__`/`__device__` in
`cuda/dyn_core/dgrid/`, and (per git archaeology) this was true on the WP-B
branch from its very first commit, not something integration stripped out.
The directory's own README says so.

**2. What exactly exists for fv_tp_2d_hord6?**
A complete, non-trivial host C++ template (`wp_b_kernels.hpp:967-1178`) —
4-pass Y-X/X-Y PPM order-6 transport with `ra_x`/`ra_y` corrections. Ordinary
CPU loops, no CUDA qualifiers anywhere. Called only from its own header's
`transport_service_for_update_dz`, which itself has no caller outside its own
unit test.

**3. WP-A: d2a2c_vect, c_sw, divergence_corner, update_dz_c, p_grad_c** — all
five are real, Fortran-faithful `__global__` CUDA kernels (35 kernels total).
The numerics are genuinely done. What's missing is entirely at the
integration layer above them: no adapter (for 4 of 5) or an adapter that's
real but never wired into the dispatch table (p_grad_c), and a predictor
orchestration function that explicitly stubs out and never calls any of them.

**4. WP-C: sim1_column, sim3_column, riem_solver3, update_dz_d** — sim1/
sim3_column are real `__host__ __device__` numerical templates; riem_solver3
is a genuine `__global__` kernel with a correct launcher — the
best-integrated numerical unit in the whole tree, still orphaned (zero
callers). update_dz_d's launcher is real code but hard-gated on a WP-B
dispatch pointer that's null everywhere in production, so it can never
proceed even if called (and nothing calls it anyway).

**5. WP-D beyond launch_geopotential_from_height_cuda** — a great deal: ~30
real `__global__` kernels covering pe/pk3/pln halo, A2B order-2/4, pressure
gradient, del2 hyperdiffusion, and diffusive heating. All real, all orphaned
— the contract layer calls host C++ equivalents instead, and only a
standalone smoke test (bypassing the contract layer) ever launches the CUDA
versions.

**6. Is NumericalDispatch the real orchestration mechanism, or was something
else built during Prompt 4.2?**
Neither, functionally. `NumericalDispatch` is a real struct with a real
validator, but it is never populated with actual function pointers anywhere
in production code — the only instantiation in the entire repo is a unit test
filled with no-op dummies. And no alternative orchestrator was built either:
`fv3_gpu_dyn_core_step`, the only candidate for "the real production
orchestration mechanism," is a one-line hardcoded failure that references
neither `NumericalDispatch` nor any WP symbol. There is currently no
orchestration mechanism, real or otherwise, in the production path.

**7. Git history — lost vs. never implemented?**
Neither happened in the way the audit worried about. Git archaeology
confirms: every WP branch's code (integrated via manual re-commit, not `git
merge`) survived intact into `cuda/gpu-bringup` — nothing was dropped. WP-A/
C/D's real kernels (35+5+39 `__global__` kernels) were written once and are
still present, fully intact. WP-B's gap is not a loss — it never contained
device code at any point in its history; its own commit messages say "CUDA
compilation: NOT RUN (nvcc unavailable)" from the start. The predictor stub,
the dead `NumericalDispatch`, and `fv3_gpu_dyn_core_step`'s stub body were
also present from their originating commits and simply carried forward
unchanged — they were never wired, not de-wired.

---

## Part 4 — Decision

**CASE B — Implementations are partial, with a specific, well-characterized
shape.**

Not Case A (nothing is stranded on a lost branch — git archaeology closes
that question). Not pure Case C either — WP-A, WP-C, and WP-D contain
substantial, numerically real, Fortran-faithful CUDA kernel code (~76
`__global__` kernels across the tree), which is a genuine asset, not
vaporware. But the prior "Production first-run numerical path: COMPLETE"
claim was overstated by construction: it measured "host C++ reference passes
a CPU test," not "CUDA kernel is reachable from the production entry point,"
and no doc flagged that gap in its own summary language even though its own
tables (CUDA/GPU = NOT RUN, every row) already contained the correction.

The concrete, uniform shape of the gap across every WP:
- WP-B has no kernels to wire — it needs original numerical CUDA work.
- WP-A, WP-C, WP-D have real kernels sitting idle behind either a missing
  adapter, an unpopulated dispatch slot, or a contract layer that calls the
  host version instead of the CUDA version.
- All of it is downstream of one single blocking stub:
  `fv3_gpu_dyn_core_step` never calls anything.

### Recommended smallest plan-consistent next step

Do not start with new numerical CUDA work. Start by making
`fv3_gpu_dyn_core_step` genuinely populate and invoke a real
`NumericalDispatch` table for the WPs that already have real kernels (WP-D
geopotential/pressure-gradient/del2/heating/a2b/halo, WP-A p_grad_c, WP-C
riem_solver3), even if the C-grid predictor and WP-B path remain explicitly
`UNSUPPORTED_CONFIGURATION` for now. That is a pure integration/wiring task —
zero new numerical semantics — and it would convert several already-real
kernels from "compiled" to genuinely "GPU executed" in the integrated path for
the first time, giving an honest, narrower foothold to build the remaining
WP-A predictor and WP-B kernel work on top of, rather than layering more
contract documents on top of an entry point that still calls nothing.

---

## Methodology note

This audit was produced by 7 independent read-only investigations: one
per-WP code audit each for WP-A, WP-B, WP-C, WP-D, and WP-E/dispatch (each
opened actual `.cu`/`.cpp`/`.hpp` definitions rather than inferring from
names or declarations), one git-history archaeology pass across
`cuda/wp-a-cgrid`, `cuda/wp-b-dgrid`, `cuda/wp-c-vertical`,
`cuda/wp-d-operators`, `cuda/wp-e-runtime`, `cuda/dyn-core-integration`, and
commits `077b729`/`c2a2bdc` (via `git show`/`git log`/`git ls-tree` only, no
branch/ref mutation), and one documentation-reconciliation pass over
`docs/architecture/dyn_core_numerical_contract_inventory.md`,
`artifacts/architecture/dyn_core_numerical_contract_inventory.json`,
`docs/architecture/dyn_core_cuda_integrated_sequence.md`, and
`docs/agent_handoff/integration.md`. No source was modified, committed, or
pushed as part of this audit.
