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
