# FV3 dyn_core GPU runtime

This directory contains the version-1 shared ABI and persistent runtime owned by
WP-E. It intentionally contains no FV3 numerical kernels.

The default build is CPU-safe and validates the ABI without initializing CUDA:

```sh
cmake -S cuda/dyn_core -B build -DFV3_DYN_CORE_PRECISION=R8
cmake --build build
ctest --test-dir build --output-on-failure
```

Set `FV3_DYN_CORE_ENABLE_CUDA=ON` only in a configured CUDA environment. The
caller must select `R4`, `R4R8`, or `R8` consistently with the parent
`FV_PRECISION`; R4R8 uses float FV fields while named accumulator fields remain
double. CUDA architectures, toolkit paths, MPI paths, devices, and scheduler
settings are deliberately left to site toolchains and runtime descriptors.

The C API is in `include/fv3_gpu_api.h`. Calls return stable status values and
never expose C++, CUDA, or MPI implementation types. A context owns all device
allocations and two non-default streams. Host visibility changes only through
the explicit synchronization APIs and their returned events.
