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

## WP-E V3 / Gate A.2 Resume

### Contract consumed and exhaustive-audit result

Consumed WP-E v3 / Gate A.2 commit `1b57d64` without restarting the WP-D
translation. The exact GFDL source named by Gate A was checked out at
`ca30e26305fda43831095334c02a3f5e0b8c0c45`; this supersedes the earlier
handoff's note that only a different local source revision was available.
PACE remained pinned at `0cdba14782dca7bf54e6e7da6a4bf50ddcb94032`.

Every WP-D inventory record is marked `contract_available: yes`. Verification
against the actual v3 fields, views, config, topology, workspace, submission,
dependency, corner-service, and dispatch definitions produced this result:

| Inventory routine | Gate A.2 verification |
|---|---|
| `pe_peln_pk3_halo` | **AVAILABLE.** State IDs, `ptop`, `akap`, `use_logp`, remap control, operator workspace, compute submission, and `H_PKC2` exist. |
| `a2b_ord2_a2b_ord4` | **STILL BLOCKING despite JSON `yes`.** The four edge descriptors have the wrong rank and, in R4, wrong element type. The request also cannot express replace versus arena-output semantics. |
| `nh_split_one_grad_pressure` | **AVAILABLE except for the transitive A-to-B blocker.** Pressure fields/metrics, actual `beta`, `d_ext`, order/log selectors, topology, per-call time/index, workspace, dependencies, and dispatch exist. |
| `del2_cubed_and_heating` | **PARTIALLY AVAILABLE.** Fields, metrics, selectors, pass controls, operator arena, `H_HEAT`, and directional requests exist. The coarse launch cannot interleave the required x-copy/flux/y-copy/flux phases. |

Previous v2 request classification:

- State/config/grid views, persistent operator workspace, borrowed compute
  stream, named dependencies, and coarse dispatch: **RESOLVED BY WP-E V3**.
- Missing numerical values (`beta`, `d_ext`, `n_con`, `nf_ke`, `d_con`,
  `delt_max`, topology and per-call controls): **RESOLVED BY WP-E V3**.
- Typed global A-to-B geometry: **PARTIALLY RESOLVED / STILL BLOCKING**. IDs
  were added, but the authoritative edge arrays cannot pass validation.
- Parameterized corner operations: **PARTIALLY RESOLVED / STILL BLOCKING**.
  Request metadata was added, but the service is not reachable inside a
  `LaunchFn`, has no A-to-B destination/replace semantics, and has no phase
  control for the two copies inside one diffusion pass.

### Additional implementation completed

All previously correct formulas were preserved. This resume added:

- WP-E v3 bindings for separate mutable state and immutable grid registries,
  config/topology/dispatch controls, borrowed compute submission, dependencies,
  and the operator arena. Descriptor coverage is checked against required
  horizontal bands and vertical levels, including physical `pe(i,k,j)` and
  `peln(i,k,j)` order.
- Exact active global-cubed-sphere `a2b_ord2` and non-`USE_OLD_ALGORITHM`
  `a2b_ord4` CPU references: physical-corner great-circle extrapolation,
  real64 directional edge weights, corrected `qx/qy`, compact first-inward
  `qxx/qyy`, final averaging, and replace/non-replace behavior. Physical edges
  are derived from `TopologyView`, not inferred from allocation bounds.
- Descriptor-lower-bound-safe pressure halos, geopotential conversion,
  normal/split pressure gradients, `one_grad_p`, and `grad1_p_update` formulas.
- Physical three-way diffusion-corner averaging, SG and non-SG flux/update
  passes, real64 `cd`, nonhydrostatic moist/dry heating, and hydrostatic
  heating. `n_con==0` is a true no-op before field binding.
- CUDA enqueue translations for pressure halos, geopotential, centered and
  global A-to-B stages, normal/split/one/grad1 pressure kernels, diffusion
  corner/flux/update stages, and hydrostatic/nonhydrostatic heating. These
  enqueue on the borrowed stream and do not synchronize or transfer state.
- Validated A-to-B and directional diffusion request builders. Invalid orders,
  axes, widths, or pass indices fail before enqueue.

The active baseline numerical formulas are translated, but the coarse v3
launchers deliberately return an explicit unsupported status before mutation
where the two shared-contract deficiencies prevent legal orchestration. No
package-private field registry, service callback, stream owner, or host/device
bridge was introduced.

### Exact execution positions and cross-WP dependencies

| WP-D operation | Execution position | Requires output from | Halo/service state | Produces input for |
|---|---|---|---|---|
| `pe_halo`, `pk3_halo` / `pln_halo` | after WP-C `Riem_Solver3`, while post-Riem exchanges are active | WP-C `delp`, pressure-interface state | local one/two-ring recurrence; no new exchange | pressure preparation and remap pressure state |
| `gz=grav*zh` | after `H_ZH` completes, before pressure gradient | WP-C `zh` | `H_ZH`; two-cell band | A-to-B `gz` and pressure gradient |
| A-to-B pp/pk/gz | inside selected pressure routine, before level gradients | WP-C `pkc/pk3/zh`; WP-D `gz` | `H_PKC2`, global topology/edge geometry; replace=true | B-point pressure/interface fields |
| A-to-B `delp` | once per pressure level immediately before `u/v` update | WP-B/C resident `delp` | same geometry; replace=false into operator arena | mass denominator for pressure update |
| pressure `u/v` update | end of each acoustic substep after WP-C | WP-C pressure/height plus A-to-B output | `H_ZH` and `H_PKC2` ready | next `H_DWIND_NEXT`, then WP-A on next split |
| `del2_cubed` | after all acoustic substeps | WP-B accumulated `heat_source` | one `H_HEAT`, then directional corner operations within each pass | filtered heat source |
| temperature adjustment | immediately after final diffusion pass | filtered heat plus WP-C thermodynamic state | no further halo | final `pt/pkz`, subsystem output |

`update_dz_c` remains WP-A and `update_dz_d` remains WP-C under the Gate A
manifest; no grid/height solver owned by those packages was duplicated here.
Optional `Ray_fast` remains outside the first subsystem capability.

### Halo, diffusion, and workspace structure

- Pressure consumes a completed `H_ZH` and two-ring `H_PKC2`. Order-2 A-to-B
  reads width one; order-4 reads width two. The package preserves descriptor
  lower bounds and the unusual `pe(i,k,j)` / `peln(i,k,j)` physical axis
  order.
- `del2_cubed` performs one full `H_HEAT` exchange, then
  `ntimes=min(3,nmax)` passes with `nt=ntimes-n`. Every pass performs physical
  three-point corner averaging; when `nt>0`, x corner copy precedes zonal flux
  and y corner copy precedes meridional flux; update follows both. With three
  passes, first-pass input reach is width three. No host synchronization or
  host/device transfer occurs between passes.
- Twelve operator-arena slices persist for A-to-B, pressure, and diffusion
  work. `qx/qy/qxx/qyy`, B-point mass/pressure slices, and `fx/fy` are reused
  on the borrowed compute stream rather than allocated per call.
- PACE corroborates this orchestration: `NonHydrostaticPressureGradient`
  retains `_tmp_wk/_tmp_wk1` and replace/non-replace A-to-B objects;
  `HyperdiffusionDamping` retains `_fx/_fy/_q`, prebuilds shrinking pass
  bounds, and orders x/y copies around fluxes. PACE casts generated edge
  factors to model precision; the CUDA translation intentionally follows the
  authoritative Fortran's real64 edge arrays instead.

### Compile, execution, and validation status

- R8 WP-D host numerical source/test with `-Wall -Wextra -Werror`: **PASS**.
- R4 WP-D host numerical source/test with `-Wall -Wextra -Werror`: **PASS**.
- R8/R4 WP-D v3 adapter and adapter-test objects with
  `-Wall -Wextra -Werror`: **PASS**.
- R8/R4 v3 adapter executable: **PASS**. The CPU-safe WP-E runtime was linked
  without `-Werror` because it contains pre-existing WP-E-only warnings.
- R8 and R4 WP-E standalone CMake runtime suites: **4/4 PASS each**. The
  shared CMake target does not yet include WP-D sources; build integration is
  owned by the integration agent, not WP-D.
- Address/undefined sanitizer run of the R8 host numerical test: **PASS**.
- CUDA compilation: **NOT RUN**; neither `nvcc` nor a CUDA toolchain is
  available. The future integration target must compile CUDA as C++17.
- GPU execution: **NOT RUN**; no compiled CUDA binary or allocated GPU exists.
- Numerical/savepoint validation: **NOT RUN**; authoritative Gate A savepoints
  are not available. Deterministic tests are formula/invariant checks, not an
  end-to-end correctness claim.
- GEOS integration: **NOT RUN**.

Focused tests cover nonzero and differing vertical lower bounds,
`pe(i,k,j)` / `peln(i,k,j)`, one/two-ring recurrence, periodic and global
A-to-B constant/linear behavior,
physical edge weighting, real64 edge arithmetic in R4, replace semantics,
topology edge/corner derivation, non-SG/SG impulse diffusion, physical corner
averaging, moist/dry and hydrostatic heating, normal/split/one/grad1 pressure
sanity, descriptor coverage rejection, operator arena partitioning,
dependency failures, no-op heating, and corner-request validation.

Not run: CUDA compilation/execution, every physical edge/corner ownership
combination on device, multi-rank corner service, Fortran savepoint comparison,
acoustic recurrence, `DC70/DC80/DC90`, or full subsystem/GEOS tests.

Unsupported until later capability enablement or integration: `use_logp`,
`GAS_HYDRO_P`, nested/regional grids, optional Rayleigh/ROT3 tails, and the
coarse global pressure/post-loop launches described by the requests below.

### CONTRACT CHANGE REQUEST — authoritative A-to-B edge descriptors

Affected Gate A/WP-E contract: Gate A.2 `a2b_ord2_a2b_ord4` inventory closure
and internal-v3 `FieldId::edge_w`, `edge_e`, `edge_s`, and `edge_n` specs.

Current definition: `fv3_gpu_contract.cpp` declares all four as
`LogicalType::fv_real`, rank 2. `make_device_view` requires exact rank and
element width, so a rank-one descriptor is rejected in R8 and R4; its required
eight-byte elements are additionally rejected in R4.

Observed numerical requirement: west/east factors are one-dimensional arrays
indexed only by `j`; south/north factors are one-dimensional arrays indexed
only by `i`. All are `R_GRID`, which is real64 independent of `FVPRC`. The
mixed real64-factor/model-real expression must be evaluated before assignment
to model-real output.

Fortran evidence: exact Gate A `model/fv_arrays.F90` defines
`R_GRID=r8_kind` at line 37, declares `edge_s/n/w/e(:)` as
`real(kind=R_GRID)` at lines 150-153, and allocates rank-one arrays at lines
1638-1641. `model/a2b_edge.F90` binds rank-one R_GRID pointers at line 93
(and 711 for order 2) and indexes a single `i` or `j` coordinate at lines
169, 187, 227, 245, 761, 771, 781, and 791.

PACE reference evidence, if applicable: pinned PACE
`util/pace/util/grid/generation.py:2809-2837` computes directional edge factors
in `numpy.float64`, then lines 2849-2852 cast them to model precision for the
DSL. That expansion/cast is a documented PACE implementation difference and
does not override the Fortran ABI or operation kind.

Proposed minimum change: retain the four stable IDs but change their internal
specs to `LogicalType::accumulator_real`, rank 1, with existing class/access.
Add R4/R8 tests accepting rank-one F64 and rejecting rank-two/model-real
descriptors. A service may derive a broadcast indexing view without replacing
the authoritative registered arrays.

Affected work packages: WP-D, WP-E/integration descriptor registration, and
any shared A-to-B service consumer (including WP-A).

Blocking: yes. Exact global A-to-B and therefore baseline global pressure
cannot bind authoritative geometry under v3.

### CONTRACT CHANGE REQUEST — intra-operator corner/A-to-B service phases

Affected Gate A/WP-E contract: Gate A.2 `CornerServiceFn`,
`CornerServiceRequest`, `ExecutionContext`, and coarse `nh_pressure` /
`post_loop_heating` dispatch semantics.

Current definition: `CornerServiceFn` is a sibling member of
`NumericalDispatch`; `LaunchFn` receives only `ExecutionContext`, which has no
service reference. `CornerServiceRequest` has a field ID, axis, width, order,
pass, signs, and dependency, but no A-to-B `replace` flag, destination view or
operator-arena slot, and no diffusion subphase. `DispatchParams` identifies a
pass but not x-copy versus zonal flux versus y-copy versus meridional flux.

Observed numerical requirement: one pressure launch must perform replace=true
A-to-B for pp/pk/gz and replace=false `delp -> wk1`, then consume `wk1` without
destroying prognostic `delp`. One diffusion pass must interleave x corner copy,
zonal flux, y corner copy, meridional flux, and update. Calling the service
only before or after the coarse launch cannot preserve either ordering.

Fortran evidence: exact `model/dyn_core.F90:1626-1647` calls replacing A-to-B
for pp/pk/gz and non-replacing A-to-B for `delp` into local `wk1`, which is
immediately consumed at lines 1653-1681. `del2_cubed` lines 2295-2354 call
direction-1 corner copy before `fx`, direction-2 copy before `fy`, and update
after both for every pass.

PACE reference evidence, if applicable: pinned
`nh_p_grad.py:159-181,237-249` owns separate replace=true and replace=false
A-to-B objects plus persistent `_tmp_wk1`. `del2cubed.py:183-202` explicitly
orders corner fill, x copy, zonal flux, y copy, meridional flux, copy, and
update inside each persistent hyperdiffusion call.

Proposed minimum change: expose the existing `CornerServiceFn` through a
versioned service view reachable by `LaunchFn`, or phase-split the coarse
dispatch with explicit enum controls. For A-to-B, add replace semantics and a
borrowed destination view or stable operator-arena slot. Do not add a second
service, private registry, global callback, stream owner, or host sync.

Affected work packages: WP-D and WP-E/integration; WP-A/B may consume the same
versioned corner-service view.

Blocking: yes for legal baseline pressure and post-loop diffusion orchestration.

### Files modified in the Gate A.2 resume

- `cuda/dyn_core/operators/wp_d_operators.hpp`
- `cuda/dyn_core/operators/wp_d_operators.cpp`
- `cuda/dyn_core/operators/wp_d_contract.hpp`
- `cuda/dyn_core/operators/wp_d_contract.cpp`
- `cuda/dyn_core/operators/wp_d_cuda.hpp`
- `cuda/dyn_core/operators/wp_d_cuda.cu`
- `cuda/dyn_core/tests/operators/test_wp_d.cpp`
- `cuda/dyn_core/tests/operators/test_wp_d_contract.cpp`
- `docs/agent_handoff/wp-d.md`

### Commit SHA and recommended integration order

The exact local commit SHA is reported with the final handoff; this document
cannot contain the hash of the commit that contains itself.

1. Correct the four WP-E edge specs and service/phase semantics without
   changing the public C ABI or ownership model.
2. Add WP-D `.cpp/.cu` to an integration-owned C++17/CUDA17 target and compile
   R8 then R4.
3. Bind the real64 rank-one edge views, wire pressure A-to-B replacement and
   arena-output stages, and validate after every A-to-B plus `DC70_PGRAD_D_OUT`.
4. Wire `H_DWIND_NEXT` and test the next-acoustic-substep recurrence.
5. Wire one `H_HEAT`, all directional diffusion phases without host sync, and
   compare after every pass before `DC90_DYN_CORE_OUT`.
6. Enable optional beta/log/Rayleigh branches only under explicit capability
   gates and validate independently.

Translation: **PARTIAL — active formulas translated; shared-contract wiring blocked**

Package compilation: **PASS for CPU sources; CUDA compilation NOT RUN**

GPU execution: **NOT RUN**

Numerical validation: **NOT RUN**

GEOS integration: **NOT RUN**

Contract changes: **REQUESTED**

WP-D BLOCKED

## WP-E V2 Resume

### Contract consumed and blocker review

Consumed WP-E internal contract v2 commit `56b9ebf` without restarting or
rewriting the existing translations. The prior contract request is classified
**PARTIALLY RESOLVED**:

- **RESOLVED BY WP-E V2:** stable `FieldId` lookup and strided device views for
  WP-D state/metrics; immutable bounds/config container; persistent
  `WorkspaceId::operators`; borrowed `Submission`; named dependency state for
  `H_ZH`, `H_PKC2`, `H_HEAT`, and `H_DWIND_NEXT`; coarse `nh_pressure` and
  `post_loop_heating` dispatch entries.
- **PARTIALLY RESOLVED:** immutable grid access includes `rdx/rdy`,
  `rarea`, `del6_u/del6_v`, and opaque edge/corner byte fields, but no schema
  exists for interpreting the edge geometry required by global `a2b_ord2/4`.
- **STILL BLOCKING:** `NumericalConfig` records only `beta_nonzero`, not the
  numerical `beta`; it omits `d_ext`, `n_con`, `nf_ke`, `d_con`, `delt_max`,
  and the Rayleigh parameters. The generic zero-argument `local_corner_fill`
  launch cannot select a field, x/y direction, halo band, or diffusion pass.

### Additional implementation completed

- Added a WP-E v2 binding adapter for all required WP-D fields and metrics.
- Deterministically partitions the conservative operator arena into the twelve
  persistent arrays provided by v2; it performs overflow and capacity checks.
- Added integration-facing `launch_nh_pressure` and
  `launch_post_loop_heating` functions matching `LaunchFn`.
- Enforces readiness of `H_ZH/H_PKC2` before pressure and `H_HEAT` before
  post-loop heating.
- Enforces baseline global/nonhydrostatic capability gates and returns explicit
  unsupported errors for missing pressure/heating branch configuration.
- Preserved all previously translated numerical formulas unchanged except for
  no numerical changes in this resume.

Remaining numerical work is unchanged: global cubed-sphere A-to-B edge/corner
corrections, diffusion corner orchestration, `one_grad_p`,
`grad1_p_update`, integrated split/log variants, `Ray_fast`, and CUDA kernels.

### Resume compile and validation status

- R8 WP-D numerical source/test with `-Wall -Wextra -Werror`: **PASS**.
- R4 WP-D numerical source/test with `-Wall -Wextra -Werror`: **PASS**.
- R8 and R4 v2 adapter source with `-Wall -Wextra -Werror`: **PASS**.
- V2 field/workspace/dependency binding test: **PASS**. Linking the CPU-safe
  WP-E runtime emits pre-existing WP-E warnings when compiled manually without
  CUDA; WP-D sources themselves are warning-clean.
- GPU execution: **NOT RUN**; `nvcc` is unavailable.
- Numerical/savepoint validation: **NOT RUN**; authoritative Gate A savepoints
  are unavailable.
- GEOS integration: **NOT RUN**.

### CONTRACT CHANGE REQUEST — WP-D V2 numerical selectors and corner service

Affected Gate A/WP-E contract: `NumericalConfig`, opaque edge geometry, and
`NumericalDispatch::local_corner_fill` in internal contract version 2.

Current definition: config provides `beta_nonzero` but no beta value and omits
post-loop pass/limit selectors. Edge/corner geometry is an unversioned byte
view. `local_corner_fill` has the generic `LaunchFn(ExecutionContext,error)`
signature with no operation arguments.

Observed numerical requirement: `split_p_grad` uses the actual beta in both
`alpha=1-beta` and persistent `beta*du/dv`; `one_grad_p` requires `d_ext`;
post-loop execution requires `n_con`, `nf_ke`, `d_con`, and `delt_max`.
`del2_cubed` performs direction-specific corner copies for a named field and a
shrinking pass band. Global A-to-B edge formulas need typed edge weights and
grid/agrid coordinates. Boolean presence flags cannot reproduce these values.

Fortran evidence: `dyn_core.F90` lines 1615-1956 use numerical `beta` and
`d_ext`; lines 1138-1187 select `nf_ke` passes, `n_con` levels, and bounded
`delt_max` heating; `del2_cubed` lines 2174-2283 calls direction 1 and 2 corner
copies for each pass. `a2b_edge.F90` consumes `edge_w/e/s/n`, `grid`, `agrid`,
`dxa`, and `dya` at global edges.

PACE reference evidence, if applicable: pinned PACE preconstructs
direction-specific `CopyCorners` objects, passes `nmax` to
`HyperdiffusionDamping`, and passes `delt_time_factor` to heating. PACE rejects
beta/log variants, so it cannot supply missing selector values.

Proposed minimum change: add the listed immutable numerical scalars (or a
versioned WP-D config subview); define a versioned typed edge-geometry schema;
and replace/supplement the corner callback with a request containing stable
field ID, x/y direction, and halo band. Keep stream/event ownership unchanged.

Affected work packages: WP-D; WP-A also consumes typed A-to-B edge geometry;
integration/WP-E owns the shared corner service.

Blocking: yes for complete baseline global diffusion/A-to-B orchestration and
nonbaseline pressure/post-loop branches; no for the already bound baseline
local formulas.

### Resume files added

- `cuda/dyn_core/operators/wp_d_contract.hpp`
- `cuda/dyn_core/operators/wp_d_contract.cpp`
- `cuda/dyn_core/tests/operators/test_wp_d_contract.cpp`

WP-D BLOCKED

## 2026-08-14 Resume

### Repository state verification

Confirmed:
- Branch: `cuda/wp-d-operators`
- WP-E v3 / Gate A.2 commit `1b57d64` present
- Previous WP-D implementation intact
- Working tree had unstaged changes from prior session

### Contract status review

Reviewed WP-E v3 internal contract (`internal_contract_version = 3`) in
`fv3_gpu_internal.hpp`. All WP-D required interfaces are present:

| WP-E v3 capability | WP-D consumption status |
|---|---|
| `FieldId` enum with all required state/grid IDs | Consumed via `bind_wp_d` |
| `DeviceView<T>` with lower bounds and strides | Consumed via `view1/view2/view3` adapters |
| `FieldRegistry` with `find(FieldId)` | Consumed for state and grid binding |
| `ImmutableViews` with bounds/config/topology | Consumed for capability gates |
| `NumericalConfig` with `beta`, `d_ext`, `n_con`, `nf_ke`, `d_con`, `delt_max`, `a2b_order` | Consumed for pressure/diffusion dispatch |
| `TopologyView` with corner/edge flags | Consumed for physical edge/corner derivation |
| `DispatchParams` with `diffusion_pass`/`diffusion_passes` | Consumed for pass control validation |
| `WorkspaceLayout` and `WorkspaceId::operators` | Consumed for operator arena partitioning |
| `DependencyState` with `require(H_ZH/H_PKC2/H_HEAT)` | Consumed for dependency enforcement |
| `Submission` with borrowed compute stream | Consumed for stream validation |
| `CornerServiceRequest` with operation/axis/width/pass | Consumed via request builders |

### Remaining contract blockers

Two blockers from the v3 resume remain unresolved:

1. **A-to-B edge descriptors**: `FieldId::edge_w/e/s/n` are declared but
   `fv3_gpu_contract.cpp` specifies `LogicalType::fv_real`, rank 2. The
   authoritative Fortran declares rank-1 `R_GRID` (real64) arrays. WP-D
   cannot bind authoritative edge geometry without spec correction.

2. **Intra-operator corner/A-to-B service phases**: `CornerServiceRequest`
   has no `replace` flag or destination view for A-to-B, and no diffusion
   subphase enum. The coarse `launch_nh_pressure` and
   `launch_post_loop_heating` cannot express the required multi-stage
   orchestration.

### Compile and test results

| Test | Result |
|---|---|
| R8 numerical test (`test_wp_d`) `-Wall -Wextra -Werror` | **PASS** |
| R4 numerical test (`test_wp_d`) `-Wall -Wextra -Werror` | **PASS** |
| R8 WP-D objects (`wp_d_operators.o`, `wp_d_contract.o`) `-Wall -Wextra -Werror` | **PASS** |
| R4 WP-D objects `-Wall -Wextra -Werror` | **PASS** |
| R8 contract test (`test_wp_d_contract`) | **PASS** |
| R4 contract test (`test_wp_d_contract`) | **PASS** |
| CUDA compilation | **NOT RUN** (`nvcc` unavailable) |
| GPU execution | **NOT RUN** |
| Numerical/savepoint validation | **NOT RUN** (no Gate A savepoints) |
| GEOS integration | **NOT RUN** |

WP-E runtime compiled without `-Werror` due to pre-existing warnings in
`fv3_gpu_runtime.cpp` (unused functions, misleading indentation).

### WP-E v3 interfaces consumed

- `FieldId`: u, v, delp, pt, delz, cappa, pe, peln, pkz, gz, zh, pkc, pk3,
  du, dv, heat_source, divg2, rdx, rdy, rarea, del6_u, del6_v, dx, dy,
  rdxc, rdyc, sina_u, sina_v, dxa, dya, grid_lonlat, agrid_lonlat
- `WorkspaceId::operators` for 12-slice operator arena
- `DependencyState::require` for H_ZH, H_PKC2, H_HEAT
- `TopologyView` for physical edge/corner derivation
- `NumericalConfig` for beta, d_ext, n_con, nf_ke, d_con, delt_max, a2b_order,
  hydrostatic, use_logp
- `DispatchParams` for dt, diffusion_pass, diffusion_passes
- `Submission` for compute stream validation
- `CornerServiceRequest` and `validate_corner_request` for request builders

### Implementation status

All milestone-one WP-D numerical routines are translated:

| Routine | CPU formula | CUDA kernel | Integration |
|---|---|---|---|
| `pe_halo` | Complete | Complete | Blocked (A-to-B) |
| `pk3_halo` | Complete | Complete | Blocked (A-to-B) |
| `pln_halo` | Complete | Complete | Blocked (A-to-B) |
| `gz=grav*zh` | Complete | Complete | Blocked (A-to-B) |
| `a2b_ord2` periodic | Complete | Complete | Ready |
| `a2b_ord2` global | Complete | Complete | Blocked (edge spec) |
| `a2b_ord4` periodic | Complete | Complete | Ready |
| `a2b_ord4` global | Complete | Complete | Blocked (edge spec) |
| `nh_p_grad` | Complete | Complete | Blocked (A-to-B) |
| `split_p_grad` | Complete | Complete | Blocked (A-to-B) |
| `one_grad_p` | Complete | Complete | Blocked (A-to-B) |
| `grad1_p_update` | Complete | Complete | Blocked (A-to-B) |
| `average_del2_corners` | Complete | Complete | Blocked (service) |
| `del2_pass` (non-SG) | Complete | Complete | Blocked (service) |
| `del2_pass_sg` | Complete | Complete | Blocked (service) |
| `apply_diffusive_heating_nh` | Complete | Complete | Blocked (service) |
| `apply_diffusive_heating_hydrostatic` | Complete | Complete | Blocked (service) |

### Files in this session

No new files created. Existing modifications from prior session verified:

- `cuda/dyn_core/operators/wp_d_operators.hpp`
- `cuda/dyn_core/operators/wp_d_operators.cpp`
- `cuda/dyn_core/operators/wp_d_contract.hpp`
- `cuda/dyn_core/operators/wp_d_contract.cpp`
- `cuda/dyn_core/operators/wp_d_cuda.hpp`
- `cuda/dyn_core/operators/wp_d_cuda.cu`
- `cuda/dyn_core/tests/operators/test_wp_d.cpp`
- `cuda/dyn_core/tests/operators/test_wp_d_contract.cpp`
- `docs/agent_handoff/wp-d.md`

### Summary

Translation: **COMPLETE for milestone-one formulas**

Package compilation: **PASS** (R4 and R8 CPU sources)

GPU execution: **NOT RUN** (no nvcc)

Numerical validation: **NOT RUN** (no savepoints)

GEOS integration: **NOT RUN**

Contract changes: **REQUESTED** (two blockers remain from v3 resume)

WP-D BLOCKED
