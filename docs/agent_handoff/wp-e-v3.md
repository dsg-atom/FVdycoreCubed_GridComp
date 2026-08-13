# WP-E v3 Gate A.2 handoff

## Second-pass requests reviewed

Reviewed complete second-pass handoffs at WP-A `1677951`, WP-B `1f8c10b`,
WP-C `a89bef6`, and WP-D `a368b97`. Accepted their missing C-grid geometry,
D-grid metrics/orders/profiles/topology, vertical per-dispatch controls,
pressure/heating selectors, typed edge geometry, and parameterized corner
service requirements.

## Exhaustive remaining-routine audit

Audited authoritative remaining `c_sw`, `d2a2c_vect`, corner/divergence,
`update_dz_c`, `d_sw`, transport/damping, `update_dz_d`, `Riem_Solver3`,
`RIM_2D`, SIM1/SIM/SIM3/SIM3p0, pressure variants, `a2b_ord2/4`,
`del2_cubed`, and heating. The machine-readable inventory is
`artifacts/architecture/dyn_core_numerical_contract_inventory.json`; every
entry has `contract_available: yes`.

## New requirements discovered proactively

- Separate height-transport order and limiter controls.
- `do_f3d`, SKEB, use-SG, Rayleigh, A-to-B order, and diffusion pass controls.
- Grid/A-grid coordinates plus four separately typed edge-weight views.
- Level-varying `d2_bg/d4_bg/damp_v/damp_w/damp_t/kgb/ndif` profiles.
- Explicit tile neighbor orientation, vector signs, periodic modes, and corner
  ownership rather than package-derived booleans.
- Per-call `replace`, interpolation order, diffusion pass/count, remap state,
  and exact timestep variants.

## Grid/geometry contract

V3 adds stable, numerically distinct IDs for C/A/B/D geometry reported by A-D:
`cosa_s`, `rsin2`, `dxa/dya`, `area_c/rarea_c`, `f_c`, `rdxa/rdya`,
`cosa/rsina`, `divg_u/divg_v`, grid/A-grid coordinates, and west/east/south/
north edge weights. Existing metrics remain unchanged. Lookup remains O(1).

## Topology contract

`TopologyView` explicitly represents global/nested/regional/periodic mode,
tile and physical corners, corner ownership, and each edge neighbor's rank,
tile, adjoining edge, index orientation, and vector component signs.

## Numerical configuration contract

`NumericalConfig` now preserves five D-transport orders, height transport and
A-to-B orders, four damping orders, beta/d_ext, damping/limiter/heating and
Rayleigh scalars, physical constants, and all audited branch selectors. It does
not expose an arbitrary Fortran namelist object.

## Per-dispatch parameter contract

`DispatchParams` carries `dt`, `dt2`, `rdt`, step/substep indices, split values,
Riemann `vertical_ms`, `last_call`, `fp_out`, remap state, and diffusion pass/
count. It is a required `ExecutionContext` member and remains separate from
immutable configuration.

## Corner/edge service contract

The parameterless corner launch was replaced with `CornerServiceFn` and
`CornerServiceRequest`, which name operation, one/two fields, staggering,
axis, scalar/vector component semantics, width, interpolation order, pass,
sign transforms, and dependency. CPU validation rejects absent fields,
oversized bands, invalid paired operations, orders, signs, and versions.

## Damping/profile contract

Scalar lifetime values remain immutable config. Level-varying real profiles
and integer `ndif` have stable descriptor-backed field identities and are
uploaded/persisted with immutable grid/config data. Dispatch-varying timestep
and pass state is not stored in those profiles.

## Cross-WP dispatch contract

The existing acyclic coarse dispatch covers A-to-C C vertical solving, A
pressure, B dynamics, B height transport invoked by C orchestration, C height/
vertical output to D pressure, D next-wind transition, and D post-loop work.
A/B/D share the integration-owned corner/edge service. No private numerical
package header is required by another package.

## Tests

- Inventory JSON parse/closure check: PASS.
- R8 standalone configure/build and all runtime tests: recorded in final result.
- R4 standalone configure/build and all runtime tests: recorded in final result.
- New CPU contract coverage: distinct geometry IDs, topology mapping, immutable
  order/selector distinction, dispatch propagation, corner validation, damping
  profile access, workspaces, and stable IDs.
- CUDA compilation: NOT RUN — `nvcc` unavailable.
- GPU execution: NOT RUN — no allocated/configured GPU environment.

## Remaining uncertainties

GPU compilation may expose CUDA-language integration issues; the corner/edge
service implementation, Stage-1 MPI pack/unpack, exact workspace lifetime
optimization, and authoritative savepoint validation remain implementation
work. Nested/regional numerical paths remain unsupported by milestone one even
though topology represents them for explicit capability rejection. No known
milestone-one metadata or service contract remains open.

## Files modified

- `cuda/dyn_core/include/fv3_gpu_internal.hpp`
- `cuda/dyn_core/include/fv3_gpu_context.hpp`
- `cuda/dyn_core/runtime/fv3_gpu_contract.cpp`
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`
- `cuda/dyn_core/tests/runtime/test_contract.cpp`
- `artifacts/architecture/dyn_core_numerical_contract_inventory.json`
- `docs/architecture/dyn_core_numerical_contract_inventory.md`
- `docs/architecture/dyn_core_contract_change_log.md`
- `docs/agent_handoff/wp-e-v3.md`

## Closure table

| Package | Remaining requirements |
|---|---|
| WP-A | CLOSED |
| WP-B | CLOSED |
| WP-C | CLOSED |
| WP-D | CLOSED |

## Commit SHA

The local commit SHA is reported in the final response; a commit cannot contain
its own content-derived identifier.

WP-E V3 CONTRACT CLOSED — RESUME WP-A/B/C/D
