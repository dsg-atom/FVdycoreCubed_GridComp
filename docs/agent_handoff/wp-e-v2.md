# WP-E v2 handoff

## Contract requests reviewed

Read the complete WP-A, WP-B, WP-C, and WP-D handoffs and consolidated every
blocking contract request in
`docs/architecture/dyn_core_contract_change_log.md`.

## Accepted changes

- Stable `FieldId`/`FieldSpec` identities and constant-time device-view lookup.
- Separate mutable state, immutable grid/config, shared temporary, and runtime
  submission/dependency representations.
- Four deterministically sized/aligned non-overlapping persistent arenas.
- Borrowed runtime-owned stream submission with status/error propagation.
- Named halo readiness/stale/pending state and event tokens.
- Integration-owned coarse dispatch, including WP-A-to-WP-C C solver,
  WP-C sequencing of WP-B height transport, and local corner-fill services.
- CPU-safe tests for field lookup/views, workspace layout, immutable view
  construction, halo transitions, dispatch completeness, and contract version.

## Rejected/deferred changes

No WP request was rejected. CUDA-aware MPI, actual Stage-1 pack/unpack and MPI,
overlap scheduling, workspace alias optimization, and public exposure of
internal launch functions are deferred. These are not needed for packages to
resume translation against v2.

## API compatibility impact

The public C ABI remains `FV3_GPU_ABI_VERSION == 1`; no public struct layout or
function signature changed. The new C++ numerical contract is independently
versioned as 2. Initialization now rejects unknown field IDs and allocates the
four persistent arenas, a deliberate pre-integration validation tightening.

## Files modified

- `cuda/dyn_core/CMakeLists.txt`
- `cuda/dyn_core/include/fv3_gpu_context.hpp`
- `cuda/dyn_core/include/fv3_gpu_internal.hpp`
- `cuda/dyn_core/include/fv3_gpu_state.hpp`
- `cuda/dyn_core/runtime/fv3_gpu_contract.cpp`
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`
- `cuda/dyn_core/tests/runtime/test_contract.cpp`
- `docs/architecture/dyn_core_contract_change_log.md`
- `docs/agent_handoff/wp-e-v2.md`

## Tests

- R8 CMake configure/build: PASS.
- R8 tests (`descriptor`, `api`, `c_header`, `contract`): PASS, 4/4.
- R4 CMake configure/build: PASS.
- R4 tests (`descriptor`, `api`, `c_header`, `contract`): PASS, 4/4.
- CUDA compilation: NOT RUN — `nvcc` unavailable on this login environment.
- GPU runtime and MPI tests: NOT RUN — no GPU allocation/configured adapter.

## Resume decision

WP-A/B/C/D can resume without redefining shared state identity, grid/config,
workspace, stream, event, corner-service, or cross-package dispatch
infrastructure. They should implement adapters against
`fv3_gpu_internal.hpp`; integration remains responsible for registering the
coarse dispatch table and invoking it from `fv3_gpu_dyn_core_step`.

WP-E V2 CONTRACT READY — RESUME WP-A/B/C/D
