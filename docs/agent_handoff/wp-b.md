# WP-B handoff

## Work completed

- Verified the clean `cuda/wp-b-dgrid` worktree and WP-E skeleton commit
  `9834f77` (cherry-pick of source commit `965c8ca`).
- Read all frozen Gate A documents and the WP-E handoff before editing.
- Inspected authoritative GFDL FV3 commit
  `ca30e26305fda43831095334c02a3f5e0b8c0c45` in a read-only temporary
  checkout and PACE commit `0cdba14782dca7bf54e6e7da6a4bf50ddcb94032`.
- Established the full WP-B routine/dependency/reference/kernel map below.
- Translated and component-tested four contiguous numerical pieces: face
  flux/Courant preparation, real64 flux-capacitor accumulation, conservative
  mass/scalar application, and vertically ordered `mix_dp` repair.
- Did not create private prognostic storage, descriptors, streams, events,
  configuration objects, or workspace policy.

Translation of the complete `d_sw` family is **not complete**. WP-E v1 lacks
the internal interface needed to launch a numerical package, so this work
package is blocked before a correct integrated CUDA translation can be made.

## Exact scope and reference map

| Fortran routine/region | Direct dependencies | PACE reference | Proposed CUDA implementation |
|---|---|---|---|
| `d_sw`: contra-variant flux preparation | `uc/vc`, `cosa_u/v`, `rsin_u/v`, `sin_sg`, `rdxa/rdya`, `dx/dy`, corners | `FiniteVolumeFluxPrep`; `_uc_contra/_vc_contra` | interior face kernel plus panel-edge/corner boundary kernels; preparation arithmetic translated |
| `fv_tp_2d` mass transport | `xppm`, `yppm`, `copy_corners`, `deln_flux`, area metrics | `FiniteVolumeTransport` (`fvtp2d_dp`) | directional reconstruction kernels, transverse updates, flux average/damping; preserve Y-X and X-Y paths |
| flux capacitors | mass fluxes and local Courants | `_flux_capacitor_stencil` | one face accumulation kernel; translated with real64 outputs |
| `fv_tp_2d` for `w`, `q_con`, `pt`, inline `q` | mass fluxes, limiter/order branches, scalar halos | PACE `fvtp2d_vt_nodelnflux`, `fvtp2d_dp_t`, `fvtp2d_tm`; PACE rejects `inline_q` | per-scalar transport group followed by conservative apply; base apply translated |
| KE and momentum transport | `xtp_u`, `ytp_v`, `pert_ppm`, metrics, cubed-sphere corner fixes | `_compute_kinetic_energy` | traceable x/y momentum reconstruction groups, KE combine, then corner correction |
| relative/absolute vorticity | `u/v`, `dx/dy`, `rarea`, `f0`, optional `z_rat` | `_compute_vorticity_stencil`, `_rel_vorticity_to_abs` | curl kernel then absolute-vorticity kernel; ROT3/do_f3d specialization remains explicit |
| divergence damping | `divg_d`, `fill_corners`, `smag_corner`, `divg_u/v`, damping orders | `DivergenceDamping` | order-specialized repeated gradient/divergence passes separated by required corner fills |
| `del6_vt_flux` | `copy_corners`, `del6_u/v` (or `USE_SG` metrics), `rarea` | `DelnFluxNoSG` | order-specialized flux/Laplacian passes; boundary fills remain explicit |
| damping heat and SKEB accounting | vorticity/divergence damping increments, metrics, `delp` | heat/dissipation and accumulation stencils | compute local lost KE, then update `heat_source`/`diss_est` without changing operation order |
| final D momentum update | transported vorticity, KE, damping fluxes | `_u_and_v_from_ke_stencil`, `_update_u_and_v_stencil` | D-u and D-v update kernels |
| `mix_dp` adapter | `ak/bk`, vertically adjacent `delp/pt/w` | outside PACE D-grid object; no meaningful pinned D-grid equivalent found | one thread per column with serial top-to-bottom repair and bottom correction; translated |

`del2_cubed` and post-loop temperature adjustment remain WP-D-owned and were
not pulled into WP-B.

## CUDA implementation grouping

| Group | Inputs | Outputs | Temporary fields | Horizontal radius | Vertical dependence | Required halo state | Launch domain |
|---|---|---|---|---:|---|---|---|
| flux prep | `uc/vc`, grid metrics, `dt` | `ut/vt`, `crx/cry`, `xfx/yfx` | `ut/vt` | 1 interior; special panel edges/corners | independent levels | `H_CWIND`; vector/corner halos valid | x faces and y faces, split interior/boundary |
| mass transport | `delp`, faces, Courants, metrics | `fx/fy` | PPM interfaces, transverse states, `ra_x/ra_y` | order-dependent, up to active `hord` footprint | independent levels | scalar/corner input valid | A cells and face domains |
| scalar transport/apply | `w/pt/q_con/q`, mass flux, faces | updated scalars and `delp` | reusable scalar flux pair and old mass | order-dependent | independent levels/tracers | `H_Q_INLINE` for enabled tracers | A compute cells and faces |
| capacitor accumulation | local Courants, mass flux | `cx/cy/mfx/mfy` | none | 0 | independent levels | none beyond producer | face compute domains |
| KE/momentum transport | `u/v/uc/vc`, metrics | corner KE and transported momentum | `ub/vb/ke` | order-dependent plus corner corrections | independent levels | D/C vector halos | B corners then D faces |
| vorticity/divergence | D winds, `divgd`, metrics | vorticity and damped divergence | `wk/vort/ut/vt` | `nord+1`; corner fill each pass | independent levels | `H_DIVGD`; corner state per pass | A/B and face domains |
| damping accounting | damping fluxes, winds, `delp` | `u/v`, `heat_source`, `diss_est` | `fx/fy/gx/gy`, increments | 1 after damping fluxes | independent levels | producer-local | A cells and D faces |
| `mix_dp` | `delp/pt/w`, `ak/bk` | repaired `delp/pt/w` | scalar registers | 0 | full column, ordered | compute cells (or one halo ring for CG mode) | one worker per horizontal column |

Aggressive fusion is intentionally deferred. Flux reconstruction and
application remain separate comparison points matching Fortran and PACE.

## WP-E interfaces consumed

- `fv3_gpu_field_desc` lower bounds and explicit element strides.
- `fv3::gpu::fv_real`, selected by the shared precision contract.
- `fv3::gpu::accumulator_real` for `mfx/mfy/cx/cy/dpx`.

The implementation allocates nothing and accepts only non-owning descriptor
adapters. No host/device copies or synchronization are introduced.

## Inputs expected from other work packages

- WP-A: completed `uc/vc`, `divgd`, `ua/va` and ready events `H_CWIND` and
  `H_DIVGD`.
- WP-C: no same-substep input. WP-B produces transport state consumed by WP-C.
- WP-D: corrected D winds from the preceding substep through `H_DWIND_NEXT`;
  no WP-D post-loop operator is implemented here.
- WP-E: device field/grid/config views, transport workspace slice, compute
  stream submission, dependency/ready event handles, and halo/corner services.

## Outputs consumed by later work packages

WP-C consumes updated `delp/pt/q_con`, `crx/cry/xfx/yfx`, `w`, and the
`H_DSCALAR` readiness boundary. WP-D later consumes accumulated
`heat_source`; host tracer/remap consumers eventually consume the real64 flux
capacitors and updated prognostics through explicit subsystem synchronization.

## Halo requirements and communication boundaries

- Before flux prep: `H_CWIND` must make C-vector halos/corners valid.
- Before higher-order divergence damping: `H_DIVGD` must be valid.
- Before inline tracer transport: `H_Q_INLINE` must be valid for each enabled
  tracer group.
- `fv_tp_2d`, `xtp_u`, `ytp_v`, `del6_vt_flux`, and repeated divergence
  damping need order-dependent halos and local corner fills between passes.
- After prognostic updates, WP-B marks `delp/pt/q_con` (and enabled `q`) dirty;
  WP-E produces `H_DSCALAR` before WP-C.
- Interior face/scalar work can exclude the active stencil radius and run
  independently of boundary work later. No overlap was implemented.

No numerical code calls MPI, a host callback, `cudaDeviceSynchronize`, or a
host/device copy.

## Temporary/workspace usage

Fortran requires `ut/vt`, `fx2/fy2`, `dw`, `ub/vb`, `wk`, `ke`, `vort`,
`fx/fy`, `gx/gy`, `ra_x/ra_y`, and PPM reconstruction arrays. PACE persists
equivalent typed quantities on its D-grid object. These should occupy the
WP-E transport arena with event-proven lifetimes; stable `crx/cry/xfx/yfx`
and `heat_source` must remain named shared arrays. The current code requests
no allocation because WP-E exposes no arena slicing interface.

## Dissipation and flux accumulation behavior

- Courant and mass-flux capacitors accumulate every acoustic substep and use
  real64 even when FV fields are R4.
- `dpx` accumulates the mass-flux divergence after the mass update.
- `heat_source` is initialized per D step, receives vertical-velocity damping
  loss, then is multiplied by updated `delp` while divergence/vorticity losses
  are incorporated in the original order.
- `diss_est` follows the same local loss expression only when SKEB is enabled.
- No reassociation, reduction, or limiter substitution is authorized.

## Meaningful PACE differences

- PACE rejects `inline_q`, `do_f3d`, `d_ext > 0`, and grid types not covered by
  its KE implementation; Fortran contains these branches.
- PACE constructs persistent typed temporaries and separates verified stencils,
  while Fortran reuses arrays such as `vc/uc`, `ptc`, and `vort` across phases.
- PACE removes several copied-corner operations under DaCe orchestration. CUDA
  must retain explicit corner dependencies until equivalence is demonstrated.
- PACE assumes damping thresholds in constructor assertions; Fortran evaluates
  per-level thresholds. CUDA must preserve explicit capability/branch behavior.

## Status

- Translation complete: **NO** — four core pieces translated; full PPM,
  momentum, vorticity, divergence, damping, corner, tracer, and branch paths
  remain.
- Package-level host compilation: **PASS**, R8 and R4, C++17 with
  `-Wall -Wextra -Werror`.
- CUDA compilation: **NOT RUN** — `nvcc` unavailable and CUDA launch contract
  is missing.
- GPU execution tested: **NO**.
- Numerical validation complete: **NO** — no Gate A Fortran savepoints were
  available.
- GEOS integration complete: **NO**.

## Tests run

- R8 deterministic component test for descriptor indexing, signed Courant/flux
  preparation, and `mix_dp`: PASS.
- R4 version of the same test: PASS.
- Existing WP-E R8 CMake compilation/link: PASS. CTest from `/tmp` was NOT RUN
  because that filesystem is mounted `noexec`; rerun from executable NCCS
  scratch: PASS, 3/3 tests.

## Tests not run

CUDA compilation/execution, `DC40_DSW_OUT` savepoint comparison,
flux-before-application comparisons, conservation budgets, every `hord/nord`
combination, inline tracers, cubed-sphere edge/corner ranks, multi-rank halo
exchange, and full GEOS builds were not run.

## Unsupported branches/options

All full-step branches remain unsupported until complete translation and
capability wiring: nested grids, `SW_DYNAMICS`, `grid_type >= 3`, `inline_q`,
`USE_COND`, `ROT3`/`do_f3d`, hydrostatic mode, SKEB, `nord == 0`, higher-order
divergence/vorticity/temperature damping, all PPM order/limiter variants, and
CG halo-ring `mix_dp`. Unsupported configurations must fail before mutation.

## CONTRACT CHANGE REQUEST

Affected Gate A/WP-E contract: Shared internal state/runtime interface between
WP-E and numerical packages (Gate A persistent state, immutable device
configuration, workspace manifest, streams/events, and communication events).

Current definition: `State` is only `std::vector<DeviceField>` with no stable
field-id lookup/view API; immutable grid/config data are absent; `Workspace`
contains two untyped pointers without sizes or slicing; `ContextData` owns
opaque stream pointers but exposes no numerical enqueue interface; halo group
enums exist only in a C callback header and there is no internal dependency or
ready-event publication API. `fv3_gpu_dyn_core_step` always reports
`UNSUPPORTED_CONFIGURATION`.

Observed numerical requirement: The complete D-grid package must obtain typed,
strided device views for prognostic, cross-WP, metric, and configuration
fields; acquire a bounded typed transport workspace; enqueue ordered kernels on
the context compute stream after `H_CWIND/H_DIVGD/H_Q_INLINE`; request local
corner fills between order-dependent passes; and publish `H_DSCALAR`, without
allocating duplicate state or synchronizing/copying through the host.

Fortran evidence: pinned `sw_core.F90:d_sw` lines 520-1540 consumes persistent
state and many `fv_grid_type` metrics, uses multiple local work arrays, calls
corner-sensitive `fv_tp_2d`, `xtp_u`, `ytp_v`, `del6_vt_flux`, and
`smag_corner`, and reuses intermediates in strict phase order. `tp_core.F90:
fv_tp_2d` lines 109-243 requires order-dependent reconstruction and explicit
corner copies. `dyn_core.F90:mix_dp` lines 2033-2109 requires serial vertical
adjacency updates.

PACE reference evidence, if applicable: pinned
`stencils/d_sw.py:DGridShallowWaterLagrangianDynamics` constructs persistent
typed temporaries and component objects once, orchestrates all D-grid stencils
as one object, and receives persistent state/metric quantities plus configured
halo behavior. Its `__call__` ordering confirms the required workspace
lifetimes and coarse submission boundary.

Proposed minimum change: WP-E/integration owner adds a versioned internal C++
numerical-package interface (not public C ABI) providing (1) stable field and
immutable grid/config lookup by integration-owned IDs returning existing
descriptor-backed device views; (2) checked transport-arena byte slices with
documented alignment/lifetime; (3) access to enqueue on the owned compute
stream without exposing stream ownership; (4) wait/publish operations for the
existing named halo events; and (5) an integration-owned local corner-fill
service or declared callable adapter. No layout, precision, public ABI, device
policy, MPI ownership, or error ABI change is requested.

Affected work packages: WP-A, WP-B, WP-C, WP-D, WP-E, integration owner.

Blocking: **yes**.

## Known issues

- The translated routines are host-callable numerical references rather than
  CUDA launches because the contract request above is blocking.
- `FieldView` trusts descriptors already validated by WP-E; it performs no
  duplicate validation.
- The component test covers only compact contiguous descriptors; WP-E owns the
  exhaustive stride/layout tests.
- Gate A savepoints have not been supplied, so numerical correctness cannot be
  claimed.

## Files modified

- `cuda/dyn_core/dgrid/README.md`
- `cuda/dyn_core/dgrid/wp_b_kernels.hpp`
- `cuda/dyn_core/tests/dgrid/test_wp_b_kernels.cpp`
- `docs/agent_handoff/wp-b.md`

## Commit SHA

The local commit containing this handoff (report with `git rev-parse HEAD` after
commit); no push or merge was performed.

## Recommended integration order

1. WP-E/integration owner resolves the blocking internal-interface request.
2. WP-B wires descriptor-backed device views and transport-arena slices.
3. Translate and validate flux prep plus complete `fv_tp_2d` at the
   flux-before-application checkpoint.
4. Add scalar/tracer transport and real64 capacitors, then validate budgets.
5. Add KE/momentum/vorticity/divergence/damping groups with explicit corner
   events and per-order tests.
6. Add `mix_dp` at its actual Gate A sequence location and compare columns.
7. Emit `H_DSCALAR`, compare `DC40_DSW_OUT`, then integrate WP-C and WP-D.

WP-B BLOCKED
