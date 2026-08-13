# WP-A C-grid predictor handoff

## Work completed

- Verified the clean `cuda/wp-a-cgrid` worktree and the cherry-picked WP-E
  skeleton (`b095bd5`, source commit `965c8ca`).
- Read all frozen Gate A architecture, communication, validation, execution,
  work-package, and reference-version records requested by the assignment.
- Inspected authoritative FV3 commit
  `ca30e26305fda43831095334c02a3f5e0b8c0c45` and pinned PACE commit
  `0cdba14782dca7bf54e6e7da6a4bf50ddcb94032` from read-only temporary clones.
- Added a non-owning typed view that uses WP-E descriptor lower bounds and
  strides without imposing another layout.
- Translated the normal nonhydrostatic `p_grad_c` equations into two coherent
  C-face CUDA kernels. Launches use the WP-E-owned stream, enqueue only, and do
  not allocate, transfer, synchronize, or call MPI.
- Added and ran a deterministic descriptor indexing test.

Translation is **partial**. The exact pressure-gradient operation is translated;
the complete C shallow-water predictor and C-height orchestration are not.

## Fortran routines covered

Internal routine map:

| Fortran routine | Called operation | PACE reference | Proposed/implemented CUDA unit |
|---|---|---|---|
| `c_sw` (`model/sw_core.F90:103`) | `d2a2c_vect`, corner divergence, first-order scalar fluxes, KE/vorticity, C-wind update | `CGridShallowWaterDynamics` in `stencils/c_sw.py` | proposed grouped predictor kernels; blocked before implementation |
| `d2a2c_vect` (`model/sw_core.F90:2800`) | D-to-A/C interpolation, cubed-sphere edge/corner transforms | `DGrid2AGrid2CGridVectors` | proposed predictor conversion/boundary units; blocked |
| `divergence_corner[_nest]` (`model/sw_core.F90:1659,1766`) | corner flux divergence | `divergence_corner` plus corner regions | proposed predictor corner unit; nested branch is milestone-one unsupported |
| local `fill_4corners`/`fill2_4corners` calls | directional cubed-sphere corner copies | PACE `corners` stencils | must use WP-E topology/corner metadata; blocked |
| `update_dz_c` (`model/nh_utils.F90:70`) | interface wind interpolation, upwind `gz`, monotonic height constraint, `ws3` | `UpdateGeopotentialHeightOnCGrid` | proposed height-advection unit; blocked on workspace/grid views |
| `Riem_Solver_C` (`model/nh_utils.F90:325`) | pre/post adapter around `RIM_2D`, `SIM3p0`, or `SIM1` | `NonhydrostaticVerticalSolverCGrid` | WP-A adapter only; WP-C solver entry is absent |
| `p_grad_c` (`model/dyn_core.F90:1529`) | nonhydrostatic C pressure-gradient update | `p_grad_c_stencil` in `stencils/dyn_core.py` | `pressure_gradient_c_x/y`, implemented |

Intentionally outside WP-A: `d_sw` and transport/damping (WP-B), the
`RIM_2D`/`SIM*` algorithms and vertical arena (WP-C), D-grid pressure-gradient
and post-loop operators (WP-D), and state/streams/events/halo MPI (WP-E).

## PACE reference components

- `CGridShallowWaterDynamics` splits conversion, corner fills, fluxes, and wind
  updates into grouped stencils and retains `delpc/ptc` as object-private state.
  This reorganizes the Fortran loops but preserves the same broad dependencies.
- `UpdateGeopotentialHeightOnCGrid` persists `gz_x/gz_y`, fuses vertical wind
  interpolation with flux work, and expresses monotonicity as a vertical pass.
- `NonhydrostaticVerticalSolverCGrid` separates precompute, column solve, and
  finalize and owns its column temporaries. Gate A assigns those primitives to
  WP-C, unlike PACE's physical C-grid object ownership.
- `p_grad_c_stencil` expresses the same face-local formula as Fortran. The CUDA
  version keeps each face's arithmetic ordering and separates x/y staggering.

PACE structural ideas were used only for coarse grouping; Fortran remains the
formula and branch authority.

## CUDA files created

- `cuda/dyn_core/cgrid/fv3_cgrid_view.hpp`
- `cuda/dyn_core/cgrid/fv3_cgrid.hpp`
- `cuda/dyn_core/cgrid/fv3_cgrid.cu`
- `cuda/dyn_core/tests/cgrid/test_cgrid_view.cpp`

## Shared WP-E interfaces consumed

`fv3_gpu_field_desc`, `fv3_gpu_status`, `fv_real`, and the externally supplied
WP-E compute-stream handle. No shared header/runtime file was changed.

## Field contract and stencil footprint

| Field | Logical dimensions/stagger | Role | Halo/neighbor radius | Vertical dependency |
|---|---|---|---|---|
| `delpc` | `(NI,NJ,NK)`, A | read | one adjacent cell normal to face | same `k` |
| `pkc` | `(NI,NJ,NK+1)`, interface | read | one adjacent cell normal to face | `k,k+1` |
| `gz` | `(NI,NJ,NK+1)`, interface | read | one adjacent cell normal to face | `k,k+1` |
| `uc` | `(NI+1,NJ,NK)`, C-u | read/write | output compute faces | same `k` |
| `vc` | `(NI,NJ+1,NK)`, C-v | read/write | output compute faces | same `k` |
| `rdxc` | C-u metric, 2-D | read | face-local | none |
| `rdyc` | C-v metric, 2-D | read | face-local | none |

The complete planned predictor additionally consumes D `u/v` halos, scalar
`w`, and first-split `gz`, and produces `delpc/ptc/ut/vt/divgd/uc/vc/ua/va/
omga/gz/pkc/ws3` with the layouts frozen by Gate A.

## Input halo requirements

Before the complete predictor: `H_DWIND_IN`, `H_W_IN`, and, on the first
acoustic split, `H_GZ_FIRST`. The implemented pressure-gradient operation is
after C vertical solve and requires valid radius-one `delpc/pkc/gz` values over
the face-adjacent band.

## Output/stale halo state

After `c_sw`, `divgd` is dirty and requires `H_DIVGD`. After `p_grad_c`, `uc`
and `vc` are dirty and require `H_CWIND` before WP-B. No exchange is performed
inside WP-A.

## Temporary/workspace usage

The implemented kernels allocate no persistent or dynamic temporary storage.
They use registers only. Full `c_sw` needs its Fortran `vort/ke/fx/fy` families;
`update_dz_c` needs `gz2/xfx/yfx/fx/fy`; and the C solver needs WP-C column
workspace. None was privately allocated because WP-E's manifest currently
exposes only untyped `transport` and `vertical` pointers.

## Compile status

- Descriptor/indexing C++ source: **compilation complete**, R8.
- CUDA source: **compilation not run**; `nvcc` is not installed/available.
- Full WP-A package: **compilation incomplete** because the contract adapters
  below do not yet exist.

## GPU execution status

**Not run.** No CUDA compiler/device test was available in this environment.

## Numerical validation status

**Not run.** No authoritative `DC10_CSW_OUT`, `DC20_CVERT_OUT`, or
`DC30_PGRAD_C_OUT` savepoints were available. No reference data was invented.

## Tests run

`c++ -std=c++17 -DFV3_GPU_FV_REAL_IS_DOUBLE=1` compiled and ran
`test_cgrid_view.cpp`; all i-fastest, nonzero-lower-bound, and explicit-stride
assertions passed.

## Tests not run

R4 compile, CUDA compile, kernel launch/bounds checks, GPU deterministic test,
Fortran savepoint comparison, PACE triangulation, multi-rank halos, and GEOS
integration.

## Unsupported branches/options

The only translated numerical branch is milestone-one normal
nonhydrostatic `p_grad_c`. Hydrostatic/SW dynamics, nested grids, grid types
requiring alternative corner behavior, C-solver selector variants, and the
remaining `c_sw` options are not implemented and must not be advertised as
capabilities.

## Contract change requests

### CONTRACT CHANGE REQUEST 1

Affected Gate A/WP-E contract: persistent state/workspace and immutable grid
configuration views.

Current definition: `State` is a vector of `DeviceField`; `Workspace` exposes
only `void *transport` and `void *vertical`. No stable numerical-package lookup,
named C workspace, or immutable grid/config device-view adapter is exposed.

Observed numerical requirement: `c_sw` and `update_dz_c` require named metric,
corner/topology, configuration, and C-private workspace views while remaining
device-resident.

Fortran evidence: `sw_core.F90:103-512,1659-1854,2800-3139` consumes `sin_sg`,
`cos_sg`, `cosa_*`, `sina_*`, `rsin_*`, `dx/dy/dxc/dyc`, area, corner flags,
and predictor work arrays; `nh_utils.F90:70-207` consumes `dp_ref`, `area`,
`zs`, corner flags, and five work families.

PACE reference evidence, if applicable: `CGridShallowWaterDynamics` retains
predictor temporaries and `UpdateGeopotentialHeightOnCGrid` retains `gz_x/gz_y`
and `dp_ref` in persistent objects.

Proposed minimum change: WP-E/integration owner supplies a stable internal
adapter that resolves the already-frozen descriptors for named state, immutable
grid fields, and named C workspace, plus the WP-E compute stream. Do not change
descriptor layout or public C ABI.

Affected work packages: WP-A primarily; likely WP-B through WP-D need the same
internal lookup pattern.

Blocking: yes.

### CONTRACT CHANGE REQUEST 2

Affected Gate A/WP-E contract: WP-A/WP-C internal vertical-solver boundary.

Current definition: Gate A assigns the C Riemann wrapper to WP-A and solver
primitives/vertical arena to WP-C, but no stable device-call/launch adapter is
defined in the cherry-picked WP-E skeleton.

Observed numerical requirement: `Riem_Solver_C` must select and enqueue the
WP-C full-column solver between its exact precompute and finalize operations,
without a host transfer or private duplicate solver.

Fortran evidence: `nh_utils.F90:325-432` selects `SIM3p0`, `RIM_2D`, or `SIM1`
from `a_imp` and passes ordered column arrays and configuration.

PACE reference evidence, if applicable: `NonhydrostaticVerticalSolverCGrid`
separates persistent precompute, `Sim1Solver`, and finalize operations, but the
pinned PACE configuration does not cover every Fortran selector.

Proposed minimum change: define one integration-owned internal C++ launch
contract accepting WP-E device views, solver selector/configuration, WP-C arena,
and WP-E stream/event dependency; return the stable `fv3_gpu_status`.

Affected work packages: WP-A, WP-C, WP-E/integration.

Blocking: yes.

## Known issues

- The pressure-gradient launch takes an opaque stream argument pending the
  WP-E internal adapter; it is not wired into the integrated sequence.
- CUDA syntax and execution remain unverified.
- Complete translation cannot proceed contract-faithfully without the two
  minimum interfaces above.

## Files modified

Only the four WP-A source/test files listed above and this handoff document.

## Commit SHA

Recorded in the final agent response and `git log`; a commit cannot embed its
own content-derived SHA.

## Recommended integration order

1. WP-E/integration resolves contract request 1.
2. WP-C and integration resolve contract request 2.
3. WP-A completes grouped `d2a2c_vect/c_sw/update_dz_c` translation and the
   C-solver adapter.
4. Compile R4/R8 with CUDA and run local bounds/determinism tests.
5. Validate `DC10`, `DC20`, and `DC30` against authoritative Fortran savepoints.
6. Integrate halo events, then hand off to WP-B.

WP-A BLOCKED
