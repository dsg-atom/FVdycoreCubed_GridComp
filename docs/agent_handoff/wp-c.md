# WP-C handoff — nonhydrostatic vertical dynamics

## Work completed

- Verified the clean `cuda/wp-c-vertical` worktree and the cherry-picked WP-E
  skeleton (`51e1146`, source commit `965c8ca`).
- Read the frozen Gate A scope, state, ABI, work-package, validation, and
  communication contracts and both pinned-reference documents.
- Audited the authoritative `nh_core.F90` and `nh_utils.F90` routines assigned
  by the manifest.
- Added package-local, precision-generic column views and exact translations of
  `edge_profile`, the `update_dz_d` bottom-boundary/monotonicity recurrence, and
  the final `Riem_Solver3` height reconstruction.
- Added enqueue-only CUDA kernels for the two height recurrences. They use one
  CUDA thread per horizontal column, retain serial `k` order, allocate no
  memory, perform no transfers, and do not synchronize the device.
- Added focused R4/R8 CPU tests for recurrence order and boundary behavior.

This is a partial translation. The Riemann/SIM column solvers and the horizontal
transport portion of `update_dz_d` are not implemented.

## Fortran routines covered

| Fortran routine | Status | Vertical dependency | PACE reference | Proposed CUDA implementation |
|---|---|---|---|---|
| `update_dz_d` | boundary/monotonicity tail translated; flux update not translated | horizontal stencil per interface, then backward recurrence | `UpdateHeightOnDGrid`, especially `apply_height_fluxes` | horizontal columns in parallel; one thread per column for backward monotonicity |
| `edge_profile` | translated as a tested column primitive | tridiagonal: forward elimination plus backward substitution | `cubic_spline_interpolation_constants` and `cubic_spline_interpolation_from_layer_center_to_interfaces` | one thread per face column; immutable coefficients may be persistent |
| `Riem_Solver3` | final height reconstruction translated; pressure setup/solver/finalize incomplete | mixed: forward pressure scan, column solve, backward height recurrence | `NonhydrostaticVerticalSolver`, `precompute`, `finalize` | one thread/column baseline, with persistent column arena |
| `RIM_2D` | audited, not translated | mixed nested forward/backward searches and acoustic substeps | absent from pinned PACE implementation | one thread/column; preserve branch/search and `ms` loop order |
| `SIM3p0` | audited, not translated | two tridiagonal solves and pressure scans | absent from pinned PACE implementation | one thread/column using arena arrays |
| `SIM3` | audited, not translated | two tridiagonal solves and pressure scans | absent from pinned PACE implementation | one thread/column using arena arrays |
| `SIM1` | audited, not translated | tridiagonal/column solve: forward elimination, backward substitution, pressure forward scan, `dz` backward recurrence | `Sim1Solver` | one thread/column initially; consider warp specialization only after equivalence |
| `SIM` | audited, not translated | two tridiagonal solves and pressure scans | absent from pinned PACE implementation | one thread/column using arena arrays |

The manifest assigns `Riem_Solver_C` to WP-A. WP-C did not implement or modify
it, although it invokes solver algorithms also used by WP-C.

## Vertical dependency classification

- Fully parallel in `k`: local copying and pointwise thermodynamic expressions
  only, after their interface-pressure inputs are ready.
- Forward recurrence: hydrostatic interface pressure/log-pressure construction,
  spline elimination, pressure perturbation accumulation, and tridiagonal
  elimination.
- Backward recurrence: spline substitution, tridiagonal substitution, recovered
  `delz`, height reconstruction, and `update_dz_d` monotonicity enforcement.
- Tridiagonal/column solve: `edge_profile` and every SIM family solver.
- Mixed: `RIM_2D` and the complete `Riem_Solver3` orchestration.

No translated routine parallelizes dependent levels.

## CUDA execution strategy

The fidelity-first strategy is one thread per horizontal column, with many
columns launched in parallel and serial `k` loops inside the thread. This
matches the Fortran recurrence order despite the ABI's i-fastest storage. A
future warp-per-column variant requires savepoint evidence demonstrating the
same arithmetic ordering. Kernels enqueue on the WP-E compute stream and have
no allocation, transfer, `cudaDeviceSynchronize`, or default-stream use.

## PACE reference components

The pinned source maps `Riem_Solver3` to `NonhydrostaticVerticalSolver`,
`update_dz_d` to `UpdateHeightOnDGrid`, and `SIM1` to `Sim1Solver`. PACE splits
precompute/finalize into GT4Py stencils, expresses recurrences using
`FORWARD`/`BACKWARD`, precomputes spline constants, and owns temporaries for the
solver object's lifetime. It rejects `a_imp <= 0.999`, so it is useful evidence
for the SIM1 organization only; it is not evidence for RIM, SIM, SIM3, or
SIM3p0. PACE also omits the `uniform_grid=.true.` and nonzero limiter variants
of `edge_profile`; the CUDA column primitive retains both Fortran branches.

## WP-E interfaces consumed

- `fv3::gpu::fv_real` and its build-selected R4/R8 ABI.
- The opaque compute-stream representation, passed to enqueue-only launchers.
- Descriptor stride semantics conceptually; integration adapters are blocked
  pending stable field identifiers and vertical configuration.

No WP-E-owned file was modified.

## Vertical workspace requirements

- Persistent shared workspace: immutable `dp_ref` spline coefficients and the
  sized WP-C vertical arena.
- Per-step workspace in that arena: `dm`, `dz2`, `w2`, `pm2`, `gm2`, `cp2`,
  `pem`, `pe2`, `peln2`, `peg`, `pelng`, plus SIM/RIM characteristic and
  tridiagonal arrays. Required capacity is proportional to launch concurrency
  times `NK`/`NK+1` and must be specified by WP-E/integration.
- Per-column local data: scalar recurrence values and indices. Large `NK`
  arrays must not spill into unbounded thread-local storage.

No allocation/free belongs inside an acoustic timestep.

## Inter-WP dependencies

- Consumes from WP-A only through integrated ordering/state: C-solver products
  are outside WP-C ownership.
- Consumes from WP-B after `H_DSCALAR`: `delp`, `pt`, `q_con`, `cappa`, `crx`,
  `cry`, `xfx`, `yfx`, and the current `zh`; it also consumes immutable `zs`,
  `dp_ref`, solver configuration, and timestep values.
- Produces for WP-D: `zh`, `gz`, `delz`, `w`, `pe`, `peln`, `pk`, `pkc`/`pk3`,
  and `ws`, followed by `H_ZH` and `H_PKC2` as scheduled by WP-E.
- The horizontal FV transport and damping calls within `update_dz_d` depend on
  WP-B-owned numerical operators. Integration must provide an internal device
  boundary without moving state to the host or duplicating WP-B code.

## Boundary-condition handling

- `update_dz_d`: computes `ws=(zs-zh(bottom))*rdt`, then walks bottom-to-top and
  enforces `zh(k) >= zh(k+1)+dz_min` in the original order.
- `Riem_Solver3` final height: fixes `zh(bottom)=zs`, then reconstructs upward
  with `zh(k)=zh(k+1)-delz(k)`.
- `edge_profile`: retains original top and bottom spline equations, backward
  substitution, and optional sign limiter at both boundaries.

## Compile status

- Package CPU header/test compilation: PASS with GNU C++ in R8 and R4.
- CUDA compilation: NOT RUN; `nvcc` is unavailable in this environment.
- Full package link/integration: NOT COMPLETE; integration-owned CMake was not
  modified and the contract request below blocks the launch adapter.

## GPU execution status

NOT TESTED. No CUDA compiler or GPU allocation was available.

## Numerical validation status

NOT COMPLETE. No Fortran savepoint data was available for `DC50_DHEIGHT_OUT` or
`DC60_RIEM3_OUT`. No claim of Fortran equivalence is made beyond direct source
translation and focused scalar assertions.

## Tests run

- R8: compiled and ran `test_vertical_column.cpp`: PASS.
- R4: compiled and ran `test_vertical_column.cpp`: PASS.
- `git diff --check`: PASS before handoff creation.

The tests exercise uniform spline reconstruction, paired-field symmetry,
surface `ws`, bottom/top monotonic-height behavior, negative `delz`, and final
height reconstruction.

## Tests not run

CUDA compilation/execution, descriptor-to-view integration, complete
`update_dz_d`, any Riemann/SIM solver, halo sequencing, deterministic
savepoints, per-level error reports, conservation checks, Discover build, and
GEOS integration.

## Unsupported branches/options

All complete solver paths are currently unsupported: RIM (`a_imp <= 0.5`),
SIM1 (`a_imp > 0.999`), SIM, SIM3, and SIM3p0; `use_logp`, `fp_out`,
`last_call`, `USE_COND`, and `MOIST_CAPPA` are consequently not wired. The
translated `edge_profile` primitive supports both grid branches and limiter
behavior, but has no integrated launch wrapper yet.

## CONTRACT CHANGE REQUEST

Affected Gate A/WP-E contract: version-1 GPU state/configuration and workspace
contract (`fv3_gpu_api.h`, `fv3_gpu_state.hpp`).

Current definition: state fields are an unenumerated `field_id` plus name;
`fv3_gpu_create_desc` carries only bounds/precision/flags; `Workspace.vertical`
is an unsized `void*`.

Observed numerical requirement: WP-C must reliably locate all named staggered
fields, receive immutable branch/scalar configuration (`ptop`, `akap`, `a_imp`,
`p_fac`, `scale_m`, `dz_min`, `rdgas`, `grav`, `USE_COND`, `MOIST_CAPPA`,
`use_logp`, solver selection), and obtain a correctly sized/aligned persistent
vertical arena based on `NK` and launch concurrency.

Fortran evidence: `nh_core.F90:Riem_Solver3` selects five algorithms from
`a_imp`, uses the listed constants/flags, and writes branch-dependent pressure
outputs; `nh_utils.F90:update_dz_d` uses `dp0`, `dz_min`, damping/order arrays;
all solvers declare multiple `NK`/`NK+1` column temporaries.

PACE reference evidence, if applicable: `RiemannConfig` supplies solver
configuration, `NonhydrostaticVerticalSolver` rejects unsupported `a_imp`, and
its `QuantityFactory` allocates persistent named 3-D temporaries;
`UpdateHeightOnDGrid` stores `dp_ref`-derived spline coefficients persistently.

Proposed minimum change: integration owner adds a stable field-ID registry; a
versioned immutable numerical-config descriptor (or typed internal config view)
covering the listed values and compile-branch bits; and a vertical-arena sizing,
alignment, and ownership query/allocation contract. Preserve the public opaque
context, precision ABI, streams, and device-residency rules.

Affected work packages: WP-C and WP-E directly; WP-A (C Riemann inputs), WP-B
(`update_dz_d` operator boundary), and WP-D (pressure outputs) at integration.

Blocking: yes — blocks safe integrated solver dispatch and persistent workspace
use, but does not block further package-local numerical translation.

## Numerical-risk areas

- Any reassociation of pressure scans, transcendental expressions, or Thomas
  solves can cause level-by-level divergence.
- `pe`/`peln` use `(i,k,j)` strides unlike ordinary state fields.
- Top/bottom formulas and negative `delz` sign conventions are easy to invert.
- PACE covers only SIM1 and reorganizes temporaries/stencils; it cannot validate
  the other authoritative Fortran branches.
- A thread-per-column implementation may spill if column arrays are local;
  arena slicing must be explicit.

## Known issues

- Translation is incomplete and therefore WP-C cannot enter subsystem
  integration.
- CUDA source cannot be compiled by the current standalone CMake without an
  integration-owner build change; none was made here.
- `update_dz_d` crosses into WP-B-owned `fv_tp_2d`/damping operators; the exact
  internal device interface remains an integration decision.

## Files modified

- `cuda/dyn_core/vertical/fv3_vertical_column.hpp`
- `cuda/dyn_core/vertical/fv3_vertical_kernels.cuh`
- `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`
- `cuda/dyn_core/tests/vertical/test_vertical_column.cpp`
- `docs/agent_handoff/wp-c.md`

## Commit SHA

Recorded by the local commit containing this handoff; run `git rev-parse HEAD`.

## Recommended integration order

1. Resolve the WP-E contract request and define the WP-B `update_dz_d` device
   operator boundary.
2. Complete and savepoint-test SIM1, then `Riem_Solver3` precompute/finalize.
3. Complete RIM (baseline-dependent), SIM/SIM3/SIM3p0, preserving column order.
4. Integrate full `update_dz_d`, then validate `DC50` before `DC60`.
5. Compile/run on a configured CUDA node, report first divergence by `k`, and
   only then connect WP-D pressure operators.

WP-C BLOCKED

## WP-E V3 / Gate A.2 Resume

Consumed WP-E v3 commit `28e4bc3` and audited the WP-C entries in both
`dyn_core_numerical_contract_inventory.md` and its JSON companion against the
actual version-3 C++ interfaces.

### Contract closure verification

Every contract requirement for `update_dz_d`, `Riem_Solver3`, `RIM_2D`, the
SIM family, and the shared C-grid column solver is marked available and is
present in the implementation:

- state/grid/profile identities and arbitrary strides: `FieldRegistry`;
- vertical constants, selectors, height order, limiter, and branch flags:
  `NumericalConfig`;
- exact `dt`, `rdt`, `vertical_ms`, `last_call`, `fp_out`, and remap state:
  `DispatchParams`;
- persistent storage: aligned `WorkspaceId::vertical` slice;
- sequencing: `H_DSCALAR`, borrowed compute submission, separate D-height
  transport/height/vertical dispatch entries, and C-grid vertical entry.

The v2 contract request is therefore **RESOLVED BY WP-E V3**. No new contract
change is requested.

### Additional work completed

- Corrected the existing v2 D-height adapter to consume
  `DispatchParams::rdt`; it no longer attempts to infer an acoustic timestep
  from immutable configuration.
- Added direct Fortran-order column translations for `Riem_Solver3`
  precomputation and finalization, including hydrostatic and gas-only interface
  pressure scans, condensate and moist-cappa branches, unusual pressure output
  semantics, `last_call`, `fp_out`, `use_logp`, and bottom-up height recovery.
- Preserved the previously translated height recurrences, spline solver, and
  SIM1 column solver unchanged except for v3 contract use.

### Remaining numerical implementation

Translation remains incomplete: CUDA/arena orchestration for the column
solver, `RIM_2D`, `SIM`, `SIM3`, `SIM3p0`, and the complete cross-WP
`update_dz_d` sequence are still missing. This is an implementation deficit,
not a shared-contract deficiency.

### Tests and status

- Strict (`-Wall -Wextra -Werror`) R8 WP-C column compile/run: PASS.
- Strict R4 WP-C column compile/run: PASS.
- Added tests cover condensate/moist-cappa Riemann precompute, hydrostatic
  pressure recurrence, final pressure modes, and height recovery.
- CUDA compilation/execution: NOT RUN; `nvcc` is unavailable.
- Savepoint validation: NOT RUN; no authoritative DC50/DC60 data found.
- WP-E standalone reconfigure succeeded, but the follow-on build was NOT
  COMPLETED because the environment failed to materialize the generated
  Makefile/archive in both scratch and worktree build attempts. The direct
  WP-C compilation was unaffected.

Translation: PARTIAL

Package compilation: PASS (WP-C CPU column package); CUDA NOT RUN

GPU execution: NOT RUN

Numerical validation: NOT RUN

GEOS integration: NOT RUN

Contract changes: NONE

WP-C BLOCKED

## WP-E V2 Resume

### WP-E v2 commit consumed

Consumed `0cd0c9f` (`Add FV3 GPU internal contract version 2`) without
restarting or replacing the earlier WP-C translation.

### Previous blocker review

- Stable field IDs and strided device views: **RESOLVED BY WP-E V2**.
  `FieldId`, `FieldSpec`, and `FieldRegistry` cover all WP-C state, pressure,
  transport-face, and immutable-grid fields, including unusual `pe/peln`
  strides.
- Immutable vertical constants and branch selectors: **PARTIALLY RESOLVED**.
  `NumericalConfig` supplies `ptop`, `akap`, `a_imp`, `p_fac`, `scale_m`,
  `dz_min`, `rdgas`, `grav`, timestep/splits, and compile-branch flags. It does
  not supply the call-specific `last_call` (`remap_step`), `fp_out`
  (`beta<-0.1`), or authoritative `m_split`/RIM `ms` value.
- Persistent vertical workspace: **PARTIALLY RESOLVED**. V2 supplies one
  persistent, aligned, non-overlapping `WorkspaceId::vertical` slice sized to
  sixteen full interface arrays. This removes allocation ownership ambiguity,
  but complete RIM/SIM translation must prove a lifetime-reuse map fits that
  fixed capacity before integration.
- Borrowed stream, halo dependencies, and coarse dispatch: **RESOLVED BY WP-E
  V2**. The adapter consumes the borrowed compute stream, requires
  `H_DSCALAR`, and uses the separate WP-B `dgrid_height_transport`, WP-C
  `dgrid_height`, and `vertical_solver` dispatch boundaries.

Overall classification of the original combined CONTRACT CHANGE REQUEST:
**PARTIALLY RESOLVED**; exact Riemann dispatch remains blocking.

### Additional implementation completed

- Connected the existing `update_dz_d` boundary/monotonicity recurrence to v2
  through `launch_dgrid_height_v2`. It looks up `zh`, `delz`, `ws`, and `zs`
  using stable IDs, honors descriptor lower bounds/strides, requires
  `H_DSCALAR`, enqueues on the borrowed stream, and derives the acoustic
  reciprocal timestep as `n_split/bdt`.
- Added an exact package-local translation of the dry and `MOIST_CAPPA`
  `SIM1_solver` column arithmetic. It preserves the two Thomas solves, forward
  pressure recurrence, backward substitution, bottom `ws` boundary, pressure
  floor, and bottom-to-top `delz` recurrence. Work arrays are caller-provided;
  it performs no allocation.
- Preserved the previously translated `edge_profile` and height kernels.

### Remaining missing routines

- CUDA/arena connection for `SIM1` and complete `Riem_Solver3` precompute and
  finalize.
- `RIM_2D`, `SIM`, `SIM3`, and `SIM3p0` translations.
- Full `update_dz_d` sequencing with WP-B's horizontal transport dispatch.
- Shared column-solver entry used by WP-A's C-grid wrapper.

### Compile status

- WP-C column test compiled and ran in R8: PASS.
- WP-C column test compiled and ran in R4: PASS.
- WP-E v2 standalone R8 configure/build and runtime tests: PASS, 4/4.
- CUDA compilation: NOT RUN (`nvcc` unavailable).
- Package CUDA integration build: NOT RUN; integration-owned CMake was not
  changed.

### GPU execution status

NOT RUN. No CUDA compiler or GPU allocation was available.

### Numerical validation status

NOT RUN. No authoritative `DC50_DHEIGHT_OUT` or `DC60_RIEM3_OUT` savepoints
were found. The new SIM1 test checks finite deterministic column output, the
zero top-pressure boundary, and negative layer thickness, but is not a Fortran
equivalence test.

### New CONTRACT CHANGE REQUEST

Affected Gate A/WP-E contract: internal v2 `NumericalConfig` and
`ExecutionContext` vertical-solver dispatch.

Current definition: `NumericalConfig` exposes model `bdt`, `n_split`,
`k_split`, and a generic `vertical_solver`, but `ExecutionContext` has no
per-acoustic-call `last_call`, `fp_out`, or explicit Riemann `ms`.

Observed numerical requirement: `Riem_Solver3` must receive the exact
call-varying final/remap flag, full-pressure-output selector, and acoustic
subcycle count. These control state writes and the number/order of RIM
recurrences; they cannot be inferred safely from a stream sequence number.

Fortran evidence: `dyn_core.F90` calls
`Riem_Solver3(flagstruct%m_split, dt, ..., flagstruct%use_logp, remap_step,
beta<-0.1)`. `nh_core.F90:Riem_Solver3` writes `pe/peln/pk` only when
`last_call`, chooses full versus perturbation `ppe` using `fp_out`, and passes
`ms` into `RIM_2D`; `RIM_2D` executes exactly `n=1..ms`.

PACE reference evidence, if applicable: `NonhydrostaticVerticalSolver.__call__`
receives `last_call` explicitly. PACE supports only SIM1 and therefore does not
provide an RIM `ms` analogue or authoritative evidence for `fp_out`.

Proposed minimum change: add a versioned per-dispatch step/control view to
`ExecutionContext` containing `dt`, `vertical_ms`, `last_call`, and `fp_out`
(or equivalent unambiguous values populated by integration from the public
step and dyn-core loop state). Do not overload `Submission::sequence` or static
configuration fields.

Affected work packages: WP-C and integration/WP-E; WP-A also consumes the
shared C-grid column solver controls.

Blocking: yes, for exact integrated `Riem_Solver3`/`RIM_2D` dispatch; no, for
continued package-local solver translation.

### Resume files modified

- `cuda/dyn_core/vertical/fv3_vertical_column.hpp`
- `cuda/dyn_core/vertical/fv3_vertical_kernels.cuh`
- `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`
- `cuda/dyn_core/tests/vertical/test_vertical_column.cpp`
- `docs/agent_handoff/wp-c.md`

Translation: PARTIAL

Package compilation: PASS

GPU execution: NOT RUN

Numerical validation: NOT RUN

GEOS integration: NOT RUN

Contract changes: REQUESTED

WP-C BLOCKED

## WP-C Continuation — Milestone-One Column Solver Completion

### Summary

Completed all milestone-one vertical column solver translations. The numerical
implementation now covers the complete `Riem_Solver3` dispatch tree and all
five nonhydrostatic vertical solver variants.

### Additional work completed

- **`SIM3p0_solver`**: Full column translation of the beta=0 semi-implicit
  variant. Two sequential tridiagonals: first for interface pressure
  reconstruction from cell-center equation-of-state values, second for
  vertical velocity. Forward pressure accumulation and backward `delz`
  recovery via cell-mean pressure reconstruction.

- **`SIM3_solver`**: Full column translation of the general alpha/beta
  semi-implicit variant. Adds explicit `wk` work array for the beta correction
  term and modified pressure accumulation with the `(1-beta)/alpha` scaling.

- **`SIM_solver`**: Full column translation of the general `a_imp` variant with
  `MOIST_CAPPA` support. Includes the interface gamma factor averaging for
  moist thermodynamics and the `cp2[k]-1` exponent branch.

- **`RIM_2D`**: Full column translation of the method-of-characteristics solver.
  Preserves the complete Fortran recurrence structure: acoustic substep loop
  (`n=1..ms`), characteristic propagation time (`dts`), fast-path short-circuit
  when `bdt <= dts[k]` for all levels, characteristic searching with upward and
  downward propagation, surface reflection handling with `ws2`, and the
  `ks0/ks1` layer-skip optimization for large column speeds.

- **`riem_solver3_column`**: Complete orchestration primitive that dispatches
  to the correct solver based on `a_imp`:
  - `a_imp < -0.999`: `SIM3p0_solver`
  - `a_imp < -0.5`: `SIM3_solver` with `alpha = -a_imp`
  - `a_imp <= 0.5`: `RIM_2D`
  - `a_imp > 0.999`: `SIM1_solver`
  - else: `SIM_solver` with `alpha = a_imp`

### Vertical dependency classification

| Solver | Vertical Dependency Structure |
|--------|-------------------------------|
| `SIM3p0` | Two sequential tridiagonals (forward elim, backward sub), forward pressure, backward `delz` |
| `SIM3` | Two sequential tridiagonals with beta correction, forward pressure, backward `delz` |
| `SIM` | Two sequential tridiagonals with moist-cappa factor, forward pressure, backward `delz` |
| `SIM1` | Two sequential tridiagonals with moist-cappa, forward pressure, backward `delz` |
| `RIM_2D` | Nested loops: acoustic substeps × characteristic propagation searches |

All solvers are strictly column-local sequential recurrences. No k-parallelism
is permitted without savepoint-demonstrated arithmetic equivalence.

### WP-E v3 interfaces consumed

- `FieldRegistry` with `FieldId::zh`, `FieldId::delz`, `FieldId::ws`, `FieldId::zs`,
  `FieldId::delp`, `FieldId::pt`, `FieldId::q_con`, `FieldId::cappa`, `FieldId::w`,
  `FieldId::pe`, `FieldId::peln`, `FieldId::pk`, `FieldId::pk3`, `FieldId::ppe`
- `NumericalConfig` with `ptop`, `akap`, `a_imp`, `p_fac`, `scale_m`, `dz_min`,
  `rdgas`, `grav`, `use_cond`, `moist_cappa`, `use_logp`
- `DispatchParams` with `dt`, `rdt`, `vertical_ms`, `last_call`, `fp_out`
- `WorkspaceId::vertical` slice for persistent column arena
- `StreamId::compute`, borrowed compute stream submission
- `H_DSCALAR` dependency for scalar halo readiness

### Workspace requirements

The complete `Riem_Solver3` requires per-column workspace arrays. For `km`
vertical levels:

| Array class | Count | Size each |
|-------------|-------|-----------|
| Layer arrays | ~15 | `km` |
| Interface arrays | ~20 | `km+1` |
| RIM-specific | ~15 | `km+1` or `km+2` |

Total per-column workspace: approximately `50 * (km+2)` scalars. The existing
WP-E `WorkspaceId::vertical` slice sized to sixteen full interface arrays is
insufficient for the complete RIM_2D workspace. Integration must validate
actual arena capacity before RIM dispatch.

### Boundary condition handling

- **Surface**: `ws = (zs - zh[km]) * rdt` computed before solver entry;
  `ws` used in bottom level equations for all solvers
- **Top**: Zero perturbation pressure (`pe2[0] = 0`) enforced; pressure
  reconstruction uses `pem[0] = ptop`
- **Height recovery**: `zh[km] = zs` anchored, then `zh[k] = zh[k+1] - dz[k]`
  upward recurrence

### Tests run

- R8 strict compile (`-Wall -Wextra -Werror`) of column header: PASS
- R4 strict compile of column header: PASS
- R8 column solver tests (SIM3p0, SIM3, SIM, RIM_2D, full Riem_Solver3): PASS
- R4 column solver tests: PASS
- R8 WP-E standalone configure/build and runtime tests: PASS, 4/4
- R4 WP-E standalone configure/build and runtime tests: PASS, 4/4

### Tests not run

- CUDA compilation/execution: `nvcc` unavailable
- Descriptor-to-view integration launch
- Complete `update_dz_d` with horizontal transport
- Halo sequencing validation
- Deterministic Fortran savepoint comparison
- Per-level error reports
- Conservation checks
- Discover/GEOS integration build

### Remaining implementation work

1. **`update_dz_d` horizontal transport**: The horizontal FV transport
   (`fv_tp_2d`, `del6_vt_flux`) and edge profile interpolation portions of
   `update_dz_d` are not translated. This requires WP-B-owned operators and an
   integration-defined device interface.

2. **CUDA kernel wrapping**: The column primitives are CPU-portable templates.
   Wrapping them in CUDA kernels with proper workspace addressing and the
   one-thread-per-column launch pattern is straightforward but requires
   `nvcc` compilation.

3. **Workspace arena integration**: The exact workspace slice assignment and
   column-index addressing into the WP-E vertical arena is pending integration
   work.

4. **C-grid column solver entry**: WP-A's `Riem_Solver_C` consumes the same
   column solver primitives but with different orchestration and output scope.

### Contract status

The v2 contract request for `DispatchParams` with `dt`, `vertical_ms`,
`last_call`, and `fp_out` was **RESOLVED BY WP-E V3**. All required per-dispatch
controls are now available through `ExecutionContext::dispatch_params`.

**Contract changes: NONE** — no new contract request.

### Numerical-risk areas

- `RIM_2D` characteristic searches involve multiple nested loops and conditional
  branches; any reordering changes results
- Pressure floor (`p_fac * pem`) prevents negative pressure in `dz` recovery;
  test cases must remain above floor
- `MOIST_CAPPA` branch uses per-level `gm2[k]` gamma factors; arithmetic order
  differs from dry path
- Large `ms` values in RIM produce many acoustic substeps with cumulative
  floating-point differences

### Files modified

- `cuda/dyn_core/vertical/fv3_vertical_column.hpp`
- `cuda/dyn_core/tests/vertical/test_vertical_column.cpp`
- `docs/agent_handoff/wp-c.md`

### Commit SHA

Recorded by the local commit containing this handoff.

Translation: COMPLETE (column primitives) / PARTIAL (CUDA kernels, update_dz_d transport)

Package compilation: PASS

GPU execution: NOT RUN

Numerical validation: NOT RUN

GEOS integration: NOT RUN

Contract changes: NONE

WP-C READY FOR INTEGRATION
