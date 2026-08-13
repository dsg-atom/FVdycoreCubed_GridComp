# WP-E handoff

## Work completed

Implemented the version-1 C ABI, descriptor/error ABI, build-selected FV
precision, opaque rank-local context, persistent state records, CUDA allocation
and transfer paths, owned compute and transfer streams, explicit event handles,
CPU-safe descriptor validation, and the named Stage-1 halo callback contract.
No numerical FV3 kernels were implemented.

## Shared interfaces created

- `cuda/dyn_core/include/fv3_gpu_api.h`: public C99/Fortran-compatible ABI.
- `cuda/dyn_core/include/fv3_gpu_types.hpp`: selected FV real and fixed real64
  accumulator definitions.
- `cuda/dyn_core/include/fv3_gpu_state.hpp`: persistent field/state/workspace
  records for numerical packages.
- `cuda/dyn_core/include/fv3_gpu_context.hpp`: context-owned runtime state and
  stream handles.
- `cuda/dyn_core/include/fv3_gpu_errors.hpp`: bounded stable error support.
- `cuda/dyn_core/include/fv3_gpu_communication.h`: ordered halo groups and
  integration-owned Stage-1 begin/complete callbacks.

The lifecycle API is `fv3_gpu_create`, `fv3_gpu_initialize_state`,
`fv3_gpu_sync_from_host`, `fv3_gpu_dyn_core_step`, `fv3_gpu_sync_to_host`,
`fv3_gpu_event_wait`, `fv3_gpu_get_capabilities`, and `fv3_gpu_finalize`.
`fv3_gpu_dyn_core_step` deliberately returns `UNSUPPORTED_CONFIGURATION` until
the numerical work packages are integrated.

## Gate A compliance

- State contract: follows explicit lower bounds/extents/strides, persistent
  device allocation, explicit coherence operations, and real64 accumulators.
- API contract: follows the versioned opaque C ABI and stable error contract.
- Ownership contract: changes are restricted to WP-E paths plus this handoff.
- Communication plan: named halo groups and host-adapter callbacks are stable;
  pack plans, pinned buffers, callback registration, and MPI execution remain
  to be implemented before Stage-1 integration.

## Platform status

- ADAPT login-node implementation status: implemented and CPU-compiled.
- ADAPT/Prism GPU test status: NOT RUN — no GPU allocation/toolkit available.
- Discover compilation status: NOT RUN — not run on Discover.
- Discover GPU test status: NOT RUN — not run on Discover.

## Build status

- Source generation: PASS — CMake generated R8 and R4 CPU-safe builds.
- CMake configuration: PASS — standalone runtime, CUDA disabled.
- CPU-side compilation: PASS — GNU C/C++ 8.5.0, R8 and R4.
- CUDA compilation: NOT RUN — `nvcc` unavailable on the login node.
- Link status: PASS — static runtime and three test executables linked.

The parent component `CMakeLists.txt` was not modified. Integration into the
full GEOS target is an integration-owner step; CUDA remains opt-in.

## Runtime status

- GPU initialization: implemented; NOT RUN — no GPU/toolkit.
- Allocation/free: implemented transactionally; NOT RUN on GPU.
- Streams: two non-default context-owned streams implemented; NOT RUN on GPU.
- Copies: asynchronous explicit field-mask copies implemented; NOT RUN on GPU.
- Synchronization: explicit event API implemented at skeleton level; CUDA
  stream synchronization NOT RUN on GPU.
- MPI/communication abstraction: named groups and callback ABI implemented;
  registration, packing, pinned buffers, and MPI adapter deferred.

## Tests run

- `cmake -S cuda/dyn_core ... FV3_DYN_CORE_PRECISION=R8`: PASS.
- R8 CPU compilation/link: PASS.
- `fv3_gpu_descriptor`: PASS (descriptor, stride, and ABI rejection cases).
- `fv3_gpu_api`: PASS (no-CUDA error, null-finalize, capabilities).
- `fv3_gpu_c_header`: PASS (public headers compile as C99-compatible C).
- R4 CPU configuration/compilation/link: PASS.

An initial `/tmp` CTest attempt was NOT RUN because that filesystem is mounted
`noexec`; the identical tests passed from NCCS scratch.

## Tests not run

CUDA compilation, GPU discovery, allocation/free, repeated GPU context
lifecycle, streams, transfers, invalid-device propagation, multi-context GPU
execution, MPI halo staging, and all Discover checks were not run because the
login environment exposed neither `nvcc` nor a GPU allocation.

## Shared API changes requested

None

## Known limitations

- Integrated numerical step sequencing is intentionally absent.
- Runtime CUDA source was not compiled in this environment.
- Event handles currently identify ordered context work; a CUDA event pool is
  still required before compute and transfer streams overlap independently.
- Coherence epochs are internal scaffolding; conflicting dirty-state detection
  needs the integration layer's explicit host-dirty notifications.
- Workspace sizing, named allocation manifest, halo pack plans, pinned staging,
  MPI capability probing, and callback registration are deferred.
- Milestone-1 `inline_q`/optional-tail choices and the parent CUDA toolchain are
  still Gate A conditions external to this skeleton.

## Files created or modified

- `cuda/dyn_core/CMakeLists.txt`
- `cuda/dyn_core/README.md`
- `cuda/dyn_core/cmake/common.cmake`
- `cuda/dyn_core/cmake/adapt_prism.cmake`
- `cuda/dyn_core/cmake/discover.cmake`
- `cuda/dyn_core/include/fv3_gpu_api.h`
- `cuda/dyn_core/include/fv3_gpu_types.hpp`
- `cuda/dyn_core/include/fv3_gpu_state.hpp`
- `cuda/dyn_core/include/fv3_gpu_context.hpp`
- `cuda/dyn_core/include/fv3_gpu_errors.hpp`
- `cuda/dyn_core/include/fv3_gpu_communication.h`
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`
- `cuda/dyn_core/tests/runtime/test_descriptor.cpp`
- `cuda/dyn_core/tests/runtime/test_api.cpp`
- `cuda/dyn_core/tests/runtime/test_c_header.c`
- `docs/agent_handoff/wp-e.md`

## Recommended next step

Share the version-1 descriptors, types, state records, context, error contract,
halo group names, and lifecycle declarations with WP-A through WP-D. Before
full integration, compile and exercise the CUDA paths on Prism, implement the
event pool and Stage-1 halo adapter, and connect this opt-in subdirectory from
the integration-owned parent build.

WP-E INTERFACE SKELETON READY FOR PARALLEL AGENTS
