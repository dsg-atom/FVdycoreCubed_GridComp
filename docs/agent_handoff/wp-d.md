# WP-D handoff

## Work completed

Established scope from the frozen Gate A manifest, inspected the available
GFDL Fortran, and checked out pinned PACE `0cdba14782dca7bf54e6e7da6a4bf50ddcb94032`.
Added precision-generic C++ operator translations and deterministic R4/R8 tests.
No WP-E, Fortran, PACE, other numerical package, or top-level build file changed.

- Translation complete: **partial**. Baseline local pressure/halo/diffusion and
  heating formulas are translated; global edge A-to-B work and optional
  branches remain.
- Compilation complete: **yes for the CPU package reference, R4 and R8**.
- GPU execution tested: **no**; `nvcc` and a CUDA execution environment were unavailable.
- Numerical validation complete: **no**; no Gate A savepoints were available.
- GEOS integration complete: **no**, and not authorized in this package.

## Exact scope and source map

The manifest overrides the prompt's tentative grouping: `update_dz_c` belongs
to WP-A and `update_dz_d` to WP-C. WP-D does not implement either.

| Fortran routine | Execution position | PACE reference | Proposed/implemented CUDA boundary |
|---|---|---|---|
| `pe_halo` | after `Riem_Solver3`, overlapped with `H_ZH/H_PKC2`, on remap steps | PACE edge-pressure orchestration | one-ring independent vertical recurrences; CPU package reference implemented |
| `pk3_halo` | same overlap, before pressure gradient for `use_logp=.false.` | `PK3Halo.edge_pe_update` | exactly two rings of independent columns; implemented |
| `pln_halo` | same point for `use_logp=.true.` | PACE explicitly rejects this branch | two-ring columns; local formula implemented, integration remains unsupported |
| `a2b_ord4` | immediately before gradient | `AGrid2BGridFourthOrder`, three persistent instances | separate launches preserve replacement order; periodic/interior formula implemented, global edge correction pending |
| `a2b_ord2` | selectable by `one_grad_p`/`grad1_p_update` | no baseline PACE path | periodic/interior formula implemented; global edge correction pending |
| `nh_p_grad` | after `H_ZH/H_PKC2` each substep | `NonHydrostaticPressureGradient` | A-to-B preparation then `u`/`v` edge kernels; post-A2B formula implemented |
| `split_p_grad` | same point for `beta>0` | PACE rejects nonzero beta | persistent `du/dv`; post-A2B formula implemented |
| `one_grad_p`, `grad1_p_update` | same point for `beta<-0.1` / external mode | none | unsupported pending `divg2`, `d_ext`, and edge interpolation integration |
| `del2_cubed` | after all substeps and `H_HEAT` | `HyperdiffusionDamping` | `min(3,nmax)` ordered passes with persistent face buffers; one pass implemented |
| temperature adjustment | after final diffusion pass | `temperature_adjust.apply_diffusive_heating` | baseline NH bounded update over resident state; implemented |
| `Ray_fast` | after pressure gradient when enabled | `RayleighDamping` | optional adapter not implemented |

## Cross-WP dependencies and ordering

| WP-D operation | Requires output from / state | Halo required | Produces input for |
|---|---|---|---|
| pressure-ring reconstruction | WP-C `Riem_Solver3`; `delp`, `pe/pk3` | local rings; overlaps `H_ZH/H_PKC2` | pressure interpolation |
| A-to-B conversion | WP-C `zh/pkc/pk3` | completed `H_ZH`, two-cell `H_PKC2` | D pressure kernel |
| pressure gradient | WP-C plus converted pressure fields | same two rings | `u/v`, optional `du/dv`; then WP-E `H_DWIND_NEXT` and WP-A |
| each diffusion pass | final WP-B accumulated `heat_source`; previous pass | `H_HEAT` before pass one; corner copy between passes | next pass or temperature update |
| temperature update | final diffusion; WP-C `delp/delz/cappa`, state `pt` | none after `H_HEAT` | `DC90_DYN_CORE_OUT` |

## WP-E interfaces, halos, and workspace

The code consumes `fv3_gpu_api.h`, `fv3_gpu_types.hpp`, `fv_real`, and strided
views intended to be derived from WP-E descriptors. It does not allocate shared
state, select a device, own a stream, synchronize, or define a public ABI.
Integration needs named resident `gz`, `pkc`, `pk3`, `delp`, `u/v`, `du/dv`,
`divg2`, `heat_source`, `pkz`, metrics, and hyperdiffusion workspace.

- Pressure needs `H_ZH` and exactly two valid `pkc` rings (`H_PKC2`).
- `pe_halo` reconstructs one ring; `pk3_halo`/`pln_halo` reconstruct two rings
  with full-column forward dependence.
- A-to-B order 2 has radius 1; order 4 uses two input cells.
- Diffusion has radius 1 per pass and at most three passes. Written halo width
  is `ntimes-n`; corner fill/copy precedes fluxes exactly as in Fortran.
- `fx/fy`, `qx/qy`, and B-point pressure work remain persistent. No host
  synchronization is needed between passes; only communication events intervene.

PACE keeps `_tmp_wk/_tmp_wk1`, three specialized A-to-B objects, separate
`u/v` stencils, and prebuilt varied-bound diffusion stencils. It performs one
heat halo update before all passes and applies temperature in the orchestrated
region. This implementation follows that lifetime/pass split. Unlike PACE, it
does not silently map `beta` or `use_logp` to the baseline path.

## Compile and test status

Both commands passed with `-Wall -Wextra -Werror`:

```text
c++ -std=c++17 -DFV3_GPU_FV_REAL_IS_DOUBLE=1 ... && build-wpd/test_wp_d
c++ -std=c++17 -DFV3_GPU_FV_REAL_IS_FLOAT=1 ... && build-wpd/test_wp_d_r4
```

Tests cover one/two-ring reconstruction, order-2/order-4 constant preservation,
constant-field diffusion, bounded top-level heating, and zero-gradient sanity.
They are deterministic invariants, not Fortran comparisons.

Not run: CUDA compilation/execution, cubed-sphere edges/corners, multi-pass
communication, `DC70/DC80/DC90` savepoints, acoustic recurrence, or GEOS.

The available Fortran checkout was `64a8a398da121f922b6fc4414f949282cd81c7b2`;
Gate A names `ca30e26305fda43831095334c02a3f5e0b8c0c45`, whose object was not
available locally. Exact-SHA validation is therefore still required.

## Unsupported branches and known issues

- Global cubed-sphere physical-edge/corner correction in `a2b_ord2/4`.
- `one_grad_p`, `grad1_p_update`, `d_ext`, `GAS_HYDRO_P`.
- Integrated `use_logp` selection (local `pln_halo` exists).
- `Ray_fast`, hydrostatic heating, and CUDA launch wrappers.
- `del2_cubed` corner ownership/copy launches; flux/update pass is present.

These configurations must fail before state mutation, never use the baseline silently.

## CONTRACT CHANGE REQUEST

Affected Gate A/WP-E contract: internal numerical dispatch and immutable grid
views between WP-E and numerical packages.

Current definition: WP-E exposes the public ABI, generic descriptors, `State`,
and only opaque transport/vertical workspace pointers. It has no frozen internal
operator interface for named immutable metrics, WP-D workspace, numerical
selectors, context compute stream, or named dependency/completion events.

Observed numerical requirement: WP-D needs typed strided views for `rdx/rdy`,
A-to-B edge geometry, `del6_u/del6_v/rarea`, pressure/heating state; stable A-to-B
and diffusion arenas; and ordered enqueue between `H_ZH/H_PKC2`,
`H_DWIND_NEXT`, and `H_HEAT`. Otherwise WP-D would invent private state lookup,
stream ownership, and selectors.

Fortran evidence: `dyn_core.F90` passes `gridstruct`, bounds, and selectors to
pressure, `del2_cubed`, and heating; `a2b_edge.F90` requires edge weights,
grid/agrid geometry, and distance metrics for global edges.

PACE reference evidence, if applicable: pinned `NonHydrostaticPressureGradient`
stores metrics and persistent A-to-B work; `HyperdiffusionDamping` stores metrics
and face work, prebuilds pass bounds, and is ordered by `AcousticDynamics` halos.

Proposed minimum change: WP-E/integration should add a versioned internal (not
public C ABI) read-only `GridViews`/`OperatorConfig` accessor and enqueue
registration accepting the context-owned compute stream and named events; add
named A-to-B and hyperdiffusion workspace to the manifest. Do not change
ownership, precision, device selection, or synchronization policy.

Affected work packages: WP-A and WP-D directly; WP-B/C and integration for the
common dispatch convention.

Blocking: yes for CUDA integration and global baseline completion; no for
isolated formula translation and CPU tests.

## Files modified

- `cuda/dyn_core/operators/wp_d_operators.hpp`
- `cuda/dyn_core/operators/wp_d_operators.cpp`
- `cuda/dyn_core/tests/operators/test_wp_d.cpp`
- `docs/agent_handoff/wp-d.md`

## Commit SHA

See the Git commit containing this handoff (the document cannot contain its own SHA).

## Recommended integration order

1. Resolve the internal WP-E request without changing the public ABI.
2. Integrate/validate global A-to-B edge/corner kernels.
3. Add CUDA launches and validate `DC70_PGRAD_D_OUT` before optional variants.
4. Integrate `H_DWIND_NEXT` and test recurrence with WP-A.
5. Add corner orchestration around resident diffusion passes, then integrate
   `H_HEAT` and validate every pass plus `DC90_DYN_CORE_OUT`.
6. Add optional pressure/Rayleigh branches only after baseline validation.

WP-D BLOCKED
