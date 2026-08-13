# Dynamic-core CUDA contract change log

## Gate A.1 / WP-E internal contract version 2

This record refines Gate A from implementation evidence. It does not edit or
supersede the frozen Gate A documents. The public Fortran-facing ABI remains
version 1; the internal numerical execution contract is version 2.

| Requester | Requested capability and numerical reason | Fields/workspace | Shared request | Shared solution and impact | Status |
|---|---|---|---|---|---|
| WP-A | Stable state/grid lookup for `c_sw`, `update_dz_c`, and `p_grad_c` | `u/v/w`, `delpc/ptc/ut/vt/divgd`, `gz/pkc/ws3`, metrics, topology, C arena | All WPs request lookup/config/workspace | `FieldId`, `FieldSpec`, O(1) `FieldRegistry`, immutable views, and `WorkspaceId::cgrid`; internal C++ only | accepted |
| WP-A | Invoke WP-C column solver from `Riem_Solver_C` without copying or duplicating it | C solver inputs, vertical arena, compute submission | WP-C requests the same solver boundary | `NumericalDispatch::cgrid_vertical_solver` receives one restricted `ExecutionContext`; integration owns ordering | accepted |
| WP-B | Typed state/metric/config access and persistent transport arena | prognostics, `crx/cry/xfx/yfx`, real64 accumulators, metrics, transport temporaries | All WPs request equivalent facilities | Common registry/config views and checked `WorkspaceId::transport`; no package allocator | accepted |
| WP-B | Submit on the owned stream, wait/publish halo state, and perform local corner fills | `H_CWIND`, `H_DIVGD`, `H_Q_INLINE`, `H_DSCALAR`, corner data | WP-A/D also need corners; all need dependencies | Borrowed `Submission`, `DependencyState`, named halo groups, and `local_corner_fill` dispatch; no stream ownership transfer | accepted |
| WP-C | Stable fields, vertical constants/selectors, and sized persistent column arena | `zh/gz/delz/w`, pressure interfaces, `dp_ref`, solver constants, vertical temporaries | A shares solver/config; B/D share fields | Common field registry, `NumericalConfig`, immutable grid registry, and aligned `WorkspaceId::vertical` | accepted |
| WP-C | Call WP-B horizontal height transport from the integrated `update_dz_d` sequence | `crx/cry/xfx/yfx`, `zh`, transport workspace | Cross-WP requirement | `dgrid_height_transport` dispatch slot; orchestration invokes it, never WP-C private code directly | accepted |
| WP-D | Immutable grid/operator views, operator workspace, stream and named dependency dispatch | pressure/heating state, edge geometry, metrics, A-to-B/diffusion work, `H_ZH/H_PKC2/H_HEAT/H_DWIND_NEXT` | A/B request the same facilities | Common immutable/field views, `WorkspaceId::operators`, submission and dependency contracts, `nh_pressure`/`post_loop_heating` dispatch | accepted |

No request was rejected. CUDA-aware MPI and communication/computation overlap
were not requested for this checkpoint and remain deferred behind the same
dependency representation. Exact workspace optimization/aliasing is deferred;
version 2 allocates four non-overlapping conservative arenas deterministically.

## Device field/view contract

`FieldId` is the stable, compile-time identity. `field_id` in every descriptor
must be one of those values. `FieldSpec` records name, class, logical precision,
rank, staggering, and access. `FieldRegistry::find` is an array lookup and does
not compare names in a hot path. `DeviceViewBase` contains only a non-owning
device pointer and the frozen lower-bound/extent/stride metadata.

Field classes are separate: mutable model state, immutable grid/metric data,
and named cross-package temporaries. FV-real fields use the build precision;
`mfx/mfy/cx/cy/dpx` use `accumulator_real` (real64). The registry includes all
Gate A prognostics/interfaces, shared temporaries, vertical coefficients,
horizontal metrics, damping metrics, and opaque edge/corner geometry. Ordinary
arrays retain explicit Fortran i-fastest strides; `pe/peln` retain their actual
descriptor strides.

The dimension symbols below are descriptor-defined (`A3=NI,NJ,NK`,
`I3=NI,NJ,NK+1`, `X/Y` add the named face, and `Q4` adds tracer). Bounds,
halos, and strides always come from the descriptor, never these abbreviations.

| Stable IDs | Logical type / precision | Rank / dimensions | Stagger | Access |
|---|---|---|---|---|
| `u`, `v` | FV real | 3 / D-face `NK` | D-u, D-v | inout |
| `w,delz,pt,delp,q_con,cappa,omga,ua,va,diss_est` | FV real | 3 / A3 | A | inout |
| `q` | FV real | 4 / Q4 | A | inout |
| `pe,pk,peln` | FV real | 3 / I3; explicit actual strides | interface | inout |
| `pkz` | FV real | 3 / A3 | A | inout |
| `phis` | FV real | 2 / A horizontal | A | input |
| `uc,vc` | FV real | 3 / C faces | C-u, C-v | inout |
| `mfx,mfy,cx,cy` | real64 | 3 / faces | x/y face | inout |
| `dpx` | real64 | 3 / A3 | A | inout |
| `ws` | FV real | 2 / A horizontal | A | output |
| `delpc,ptc,ut,vt` | FV real | 3 / A3 | A | inout temporary |
| `divgd` | FV real | 3 / B corners | B | inout temporary |
| `crx,xfx,cry,yfx` | FV real | 3 / faces | x/y face | inout temporary |
| `gz,zh,pkc,pk3` | FV real | 3 / I3 | interface | inout temporary |
| `du,dv` | FV real | 3 / D faces | D-u, D-v | inout temporary |
| `heat_source,z_rat` | FV real | 3 / A3 | A | inout temporary |
| `ws3,divg2` | FV real | 2 / horizontal | A, B | inout temporary |
| `ak,bk,pfull,dp_ref` | FV real | 1 / `NK` or `NK+1` | A/interface | immutable input |
| `zs,area,rarea` | FV real | 2 / A horizontal | A | immutable input |
| `dx,dy,dxc,dyc,rdx,rdy,rdxc,rdyc` | FV real | 2 / named faces | x/y or C face | immutable input |
| `cosa_*,sina_*,rsin_*` | FV real | 2 / D faces | D-u/D-v | immutable input |
| `sin_sg,cos_sg` | FV real | 3 / corner component view | B | immutable input |
| `f0` | FV real | 2 / B horizontal | B | immutable input |
| `del6_u,del6_v` | FV real | 2 / D faces | D-u/D-v | immutable input |
| `edge_geometry,corner_geometry` | versioned bytes | 1 / descriptor extent | opaque grid data | immutable input |

## Immutable grid/config contract

`ImmutableViews` exposes only `DomainBounds`, `NumericalConfig`, and a read-only
grid registry. It does not expose the runtime context. The numerical config
carries timestep/split data, vertical constants/selectors, transport/damping
orders, and milestone-one branch flags. Shape/algorithm changes remain create-
or-reinitialize operations; orchestration may construct a per-step immutable
copy for permitted timestep values.

## Workspace contract

`WorkspaceId::{cgrid,transport,vertical,operators}` are the only shared arenas.
`build_workspace_layout` validates power-of-two alignment, duplicate IDs,
overflow, bounds, and deterministic non-overlap. Initialization allocates one
persistent device block and materializes aligned slices. The initial sizing is
conservative per allocated `(NI+2H)*(NJ+2H)*(NK+1)` values: 12, 24, 16, and 12
FV-real arrays respectively. Numerical kernels borrow slices and never allocate.
Changing these formulas requires an internal contract revision and memory test.

## Stream/submission contract

`Submission` contains a borrowed opaque stream token, its role, and sequence.
Numerical launchers may enqueue on it but may not create, destroy, replace, or
synchronize it. The runtime retains compute and transfer/pack stream ownership.
Launchers return `fv3_gpu_status`; the orchestration boundary records CUDA
launch errors in `fv3_gpu_error`. Public C ABI and CUDA types remain isolated.

## Halo/dependency contract

Each frozen halo group has `unknown`, `ready`, `stale`, or `exchange_pending`
state and an event token. A consumer calls `require`; a producer marks the
group stale; WP-E marks pending/ready around Stage-1 exchange. Interior launches
may run while a group is pending; boundary launchers consume the ready event.
This representation is unchanged when host staging is replaced by device MPI.

## Internal numerical dispatch contract

The integration-owned `NumericalDispatch` table has coarse entries in Gate A
order: C predictor, C vertical solver, C pressure, D dynamics, D height
transport, D height, vertical solver, NH pressure, post-loop heating, plus a
local corner-fill service. Every entry receives only `ExecutionContext`:
registered device views, immutable views, its persistent workspace layout, a
borrowed submission stream, and dependency state. It receives neither the
opaque public context nor another package's private implementation. The public
`fv3_gpu_dyn_core_step` remains the single Fortran entry for internal ordering.

## Compatibility

No public struct or function signature changed. Existing public source and ABI
remain compatible. Initialization now requires registered stable field IDs;
this is the intended v2 validation tightening before production integration.
