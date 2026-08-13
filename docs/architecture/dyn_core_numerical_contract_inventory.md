# Gate A.2 numerical contract inventory

This is the human-readable companion to
`artifacts/architecture/dyn_core_numerical_contract_inventory.json`. The JSON
is the machine-readable closure checklist. The audit used the authoritative
Fortran routine signatures and bodies in `model/sw_core.F90`, `nh_utils.F90`,
`nh_core.F90`, `dyn_core.F90`, and `a2b_edge.F90`, the Gate A execution map,
and pinned PACE `0cdba147...` as architectural evidence.

## Classification and closure

| Remaining routine/group | WP | Non-local contract classes | V3 facility | Available |
|---|---|---|---|---|
| `c_sw`, `d2a2c_vect`, divergence/corners | A | state, C/D/A/B metrics, topology, config, workspace, corner service | stable metric IDs; `TopologyView`; parameterized corner service; C arena | yes |
| `update_dz_c`, `Riem_Solver_C` | A/C | state, `dp_ref`, config, per-call time/ms, workspace, cross-WP solver | immutable config; `DispatchParams`; vertical dispatch | yes |
| `d_sw`, transport | B | state, distinct geometry, five transport orders, limiter, workspace, halos/corners | typed IDs/config; transport arena; dependency/corner services | yes |
| divergence/vorticity/scalar damping | B | metric coefficients, four damping orders, scalar and level profiles | distinct config members and descriptor-backed profile IDs | yes |
| `update_dz_d` remainder | C/B | area, profiles, height transport order, per-call reciprocal time, cross-WP transport | typed config/params; `dgrid_height_transport` | yes |
| `Riem_Solver3`, `RIM_2D`, SIM family | C | state, vertical constants, solver selectors, `ms/last_call/fp_out`, arena | config plus `DispatchParams`; vertical arena/dispatch | yes |
| pressure variants and pressure halos | D | state, beta/d_ext/log selectors, typed edge interpolation geometry | typed IDs/topology/config; NH dispatch | yes |
| `a2b_ord2/4` | D | grid/agrid coordinates, edge weights/orientation, per-call replace/order | typed geometry IDs; topology; A-to-B corner request | yes |
| `del2_cubed`, heating | D | area/diffusion metrics, post-loop selectors, pass controls, shrinking corner bands | config, dispatch pass parameters, corner service, operator arena | yes |

Every JSON routine entry has `contract_available: yes`; therefore no known
milestone-one numerical routine requires a package-private metadata, geometry,
stream, workspace, halo, corner, or cross-WP contract.

## Grid and geometry vocabulary

V3 adds distinct stable identities for `cosa_s`, `rsin2`, `dxa/dya`,
`area_c/rarea_c`, `f_c`, `rdxa/rdya`, `cosa/rsina`, `divg_u/divg_v`, grid and
A-grid longitude/latitude coordinates, and each west/east/south/north A-to-B
edge-weight array. These supplement, rather than alias, existing C/D face
metrics, areas, inverse areas, rotation factors, Coriolis, and del6 metrics.

## Topology vocabulary

`TopologyView` names the local tile; global, nested, regional, and periodic
modes; physical corner presence/ownership; and four neighbor records. Each
neighbor records rank, tile, adjoining edge, index orientation, and vector
component signs. Numerical packages consume this immutable view and do not
derive connectivity from bounds or scattered booleans.

Milestone one remains global cubed sphere. Nested/regional values are explicit
so unsupported configurations can fail unambiguously without changing the
interface; this does not advertise their numerical implementation.

## Configuration and per-dispatch vocabulary

Immutable `NumericalConfig` preserves five transport orders, height transport
order, A-to-B order, four damping orders, pressure/diffusion/Rayleigh selectors,
physical constants, limiter, damping scalars, and branch flags. Level-varying
damping values and integer `ndif` are immutable descriptor-backed arrays.

`DispatchParams` separately carries `dt/dt2/rdt`, step/acoustic identity,
split indices, Riemann `vertical_ms`, `last_call`, `fp_out`, remap state, and
diffusion pass/count. These must be populated from the actual integrated loop;
they are not inferred from stream sequence numbers.

## Corner/edge operations

`CornerServiceRequest` specifies operation (fill, copy, pair fill, vector
transform, or A-to-B edge), one/two stable fields, staggering, x/y/both axis,
scalar/component/vector semantics, halo width, interpolation order, pass,
sign transforms, and dependency event. Validation checks contract version,
registered fields, allocated width, paired operations, order, and signs. The
service borrows the execution context stream and topology; WP-E/integration
owns its implementation and ordering.

## Damping profiles

`d2_bg`, `d4_bg`, `damp_v`, `damp_w`, `damp_t`, `kgb`, and integer `ndif` are
persistent one-dimensional immutable views because authoritative use varies by
vertical level. `dddmp`, `d_con`, `da_min`, `da_min_c`, limiter and order
selectors are immutable scalars. Timestep/pass values stay in dispatch params.
Profiles are uploaded at initialization/config update and are not recomputed by
individual launches.

## Cross-WP sequencing

The integration dispatch remains coarse and acyclic: A C predictor calls the
registered C vertical service, then C pressure; B D dynamics follows A; C
height sequencing invokes B's registered height-transport service; C vertical
then produces pressure state for D; D pressure produces the next-wind halo and
post-loop heating follows B's accumulated heat. A and D call the WP-E corner/
edge service. No package includes another package's private header.
