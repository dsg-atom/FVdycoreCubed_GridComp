# CUDA build and validation plan

## Overview

This document defines the procedure for building and validating the integrated
CUDA `dyn_core` subsystem on GPU-capable systems. This plan is prepared but
not yet executed.

## Target environments

### Development GPU environment

**Platform:** ADAPT/Prism
- NVIDIA GPU cluster
- CUDA toolkit available
- Module-based environment management

### Production validation target

**Platform:** Discover
- NCCS production system
- Final validation before GEOS integration
- Multi-GPU MPI testing

**Note:** Platform-specific paths or modules must NOT be hardcoded in numerical
source files. All platform detection should be in CMake configuration.

## GPU validation procedure

### Phase 1: Environment discovery

```bash
# Load required modules (platform-specific)
module list
module avail cuda

# Verify CUDA availability
which nvcc
nvcc --version

# Verify GPU availability
nvidia-smi

# Record environment
nvidia-smi -L
nvcc --version | head -3
```

**Expected output:** CUDA compiler path, version, and GPU device list.

### Phase 2: CUDA architecture detection

```bash
# Query GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Common architectures:
# V100: sm_70
# A100: sm_80
# H100: sm_90
```

**CMake configuration:**

```cmake
set(CMAKE_CUDA_ARCHITECTURES "70;80" CACHE STRING "CUDA architectures")
```

### Phase 3: Configure integrated CUDA target

```bash
cd /path/to/fv3cuda-integration/cuda/dyn_core

mkdir -p build-cuda-r8
cd build-cuda-r8

cmake .. \
  -DFV3_DYN_CORE_ENABLE_CUDA=ON \
  -DFV3_DYN_CORE_PRECISION=R8 \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_BUILD_TYPE=Debug
```

**R4 configuration:**

```bash
mkdir -p build-cuda-r4
cd build-cuda-r4

cmake .. \
  -DFV3_DYN_CORE_ENABLE_CUDA=ON \
  -DFV3_DYN_CORE_PRECISION=R4 \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_BUILD_TYPE=Debug
```

### Phase 4: Compile

```bash
make -j8 2>&1 | tee build.log

# Check for CUDA syntax errors
grep -i "error:" build.log
grep -i "warning:" build.log | head -50
```

**Expected outcome:** Successful compilation of all `.cu` files.

**Common issues to resolve:**

1. CUDA/C++17 compatibility
2. Host/device function annotations
3. Template instantiation for `fv_real`
4. Include path resolution
5. Linking with CUDA runtime

### Phase 5: Resolve CUDA syntax/link issues

If compilation fails:

1. Identify first error in build log
2. Check host/device annotation
3. Verify template parameter compatibility
4. Check CUDA version compatibility with C++17 features
5. Verify CMake CUDA language detection

### Phase 6: Single-rank GPU smoke test

```bash
# Run basic tests
ctest --output-on-failure

# Verify GPU was actually used
CUDA_VISIBLE_DEVICES=0 ./fv3_gpu_api_test
```

**Expected outcome:** All CPU-visible tests pass; no GPU-specific failures.

### Phase 7: Memory allocation test

```bash
# Create a simple allocation test
# Verify device memory can be allocated for expected field sizes

# Example: 180x180x72 grid requires approximately:
# - 50+ 3D fields × 4/8 bytes × 180 × 180 × 72 ≈ 400-800 MB
# Plus workspace arenas
```

### Phase 8: First integrated acoustic-substep execution

```bash
# This requires:
# 1. Authoritative Fortran savepoints for initial conditions
# 2. Host-side test driver that:
#    - Creates GPU context
#    - Uploads initial state
#    - Executes one dyn_core step
#    - Downloads results
#    - Compares against reference

# NOT YET AVAILABLE - requires savepoint capture
```

### Phase 9: Profiler sanity check

```bash
# Use nsight or nvprof for basic profiling
nsys profile --stats=true ./fv3_gpu_test

# Check:
# - Kernel launches occurring
# - No unexpected host-device copies
# - Memory utilization
# - Occupancy
```

## Pre-validation checklist

Before executing this plan, verify:

- [ ] All R4/R8 CPU-visible tests pass
- [ ] CMake CUDA path is complete
- [ ] Authoritative savepoints are captured
- [ ] Test driver exists for integrated execution
- [ ] Access to GPU compute node confirmed

## Status

| Phase | Status |
|-------|--------|
| Environment discovery | NOT RUN |
| Architecture detection | NOT RUN |
| Configure CUDA target | NOT RUN |
| Compile | NOT RUN |
| Resolve issues | NOT RUN |
| Smoke test | NOT RUN |
| Memory test | NOT RUN |
| Acoustic substep | NOT RUN |
| Profiler check | NOT RUN |

## Known issues

1. The integrated CMakeLists.txt may need updates for WP-A/B/C/D CUDA sources
2. Cross-translation-unit device function calls require verification
3. Template instantiation for precision may need explicit instantiation
4. The `update_dz_d` cross-WP boundary is unimplemented
5. Authoritative savepoints do not exist
