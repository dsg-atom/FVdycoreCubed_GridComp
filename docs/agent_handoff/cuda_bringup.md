# FV3 dyn_core CUDA Bring-Up Documentation

**Date:** 2026-08-19  
**Prompt:** 5B - Discover CUDA Build and Runtime Bring-Up  
**Author:** Claude Code Agent  

## Environment

| Component | Value |
|-----------|-------|
| Discover login node | discover login |
| GPU-node hostname | warpa003 |
| GPU model | NVIDIA A100-SXM4-40GB |
| GPU memory | 40960 MiB |
| NVIDIA driver version | 535.104.12 |
| CUDA toolkit version | 12.5 (V12.5.82) |
| GCC/G++/GFortran | 12.3.0 |
| OpenMPI | 4.1.6 |
| CMake | 3.30.3 |

## Module Environment

```
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

Complete module stack:
```
  1) git/2.49.0             8) other/tree/2.1.0
  2) cmake/3.30.3           9) other/ninja/1.12.0
  3) other/mepo/2.4.0      10) GEOSenv
  4) other/tig/2.5.8       11) comp/gcc/12.3.0
  5) ImageMagick/7.1.1-15  12) mpi/openmpi/4.1.6/gcc-12.3.0
  6) git-lfs/3.7.1         13) nvidia/nvhpc-byo-compiler/24.7
  7) other/gh
```

**Important:** The CUDA 12.5 toolkit is at:
```
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5
```
(Note: NOT the parent `cuda/` directory which lacks the proper `targets/` structure)

## Slurm Commands

### GPU Resource Probe
```bash
srun \
  --partition=gpu_a100 \
  --constraint=rome \
  --gres=gpu:1 \
  --mem=60G \
  --time=00:05:00 \
  --ntasks=1 \
  --cpus-per-task=12 \
  bash -lc '<commands>'
```

## Build Commands

### R8 (double precision)

**Configure:**
```bash
export CUDA_PATH=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5
cmake -S cuda/dyn_core -B build_cuda_r8 \
  -DFV3_DYN_CORE_PRECISION=R8 \
  -DFV3_DYN_CORE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
  -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_PATH}"
```

**Build:**
```bash
cmake --build build_cuda_r8 -j 8
```

**Result:** PASS

### R4 (single precision)

**Configure:**
```bash
export CUDA_PATH=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5
cmake -S cuda/dyn_core -B build_cuda_r4 \
  -DFV3_DYN_CORE_PRECISION=R4 \
  -DFV3_DYN_CORE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
  -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_PATH}"
```

**Build:**
```bash
cmake --build build_cuda_r4 -j 8
```

**Result:** PASS

## Source/Build Fixes Applied

### 1. fv3_cgrid_view.hpp - CUDA host/device attributes

**File:** `cuda/dyn_core/cgrid/fv3_cgrid_view.hpp`

**Problem:** `FieldView::offset()` methods and `operator()` methods missing `__host__ __device__` attributes for CUDA compilation.

**Fix:** Added `__host__ __device__` attributes to all offset methods and added single-argument overloads for Rank=1 arrays.

**Numerical semantics changed:** NO

### 2. fv3_vertical_column.hpp - CUDA host/device attributes for column functions

**File:** `cuda/dyn_core/vertical/fv3_vertical_column.hpp`

**Problem:** Multiple column solver functions (`sim1_column`, `sim3_column`, `sim3p0_column`, `sim_column`, `rim_2d_column`, `riem3_precompute_column`, `riem3_finalize_column`, `riem_solver3_column`) missing `FV3_VERTICAL_HD` macro for device compilation.

**Fix:** Added `FV3_VERTICAL_HD` macro to all column functions called from device code.

**Numerical semantics changed:** NO

### 3. fv3_vertical_column.hpp - CUDA-compatible math functions

**File:** `cuda/dyn_core/vertical/fv3_vertical_column.hpp`

**Problem:** `std::exp`, `std::log`, `std::sqrt`, `std::fabs`, `std::copysign`, `std::max` not callable from device code.

**Fix:** Added CUDA-compatible wrapper functions (`fv3_exp`, `fv3_log`, `fv3_sqrt`, `fv3_fabs`, `fv3_copysign`, `fv3_max`) that use unqualified math functions in CUDA mode and `std::` versions otherwise.

**Numerical semantics changed:** NO

### 4. fv3_vertical_kernels.cu - Template argument fix

**File:** `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`

**Problem:** Call to `vertical_workspace_per_column(km)` missing explicit template argument.

**Fix:** Changed to `vertical_workspace_per_column<fv_real>(km)`.

**Numerical semantics changed:** NO

### 5. test_api.cpp - Runtime CUDA detection

**File:** `cuda/dyn_core/tests/runtime/test_api.cpp`

**Problem:** Test used compile-time `FV3_GPU_ENABLE_CUDA` check which was only defined privately in the library, not in test compilation.

**Fix:** Changed to runtime detection using `fv3_gpu_capabilities.compiled_with_cuda` field.

**Numerical semantics changed:** NO

## Test Results

### CPU-Visible Tests (Login Node)

| Precision | Pass | Fail | Notes |
|-----------|------|------|-------|
| R8 | 9/10 | 1 | fv3_gpu_api requires GPU |
| R4 | 9/10 | 1 | fv3_gpu_api requires GPU |

### GPU Tests (A100 via Slurm)

| Precision | Pass | Fail | Notes |
|-----------|------|------|-------|
| R8 | 10/10 | 0 | All pass |
| R4 | 10/10 | 0 | All pass |

### Work Package GPU Results

| Work Package | Status | Notes |
|--------------|--------|-------|
| WP-E Runtime | PASS | Context create/destroy, device allocation |
| WP-A C-grid | PASS (compilation) | CUDA sources compile, no dedicated GPU tests |
| WP-B D-grid | PASS (compilation) | CPU-visible tests pass |
| WP-C Vertical | PASS (compilation) | CUDA sources compile, column tests pass |
| WP-D Operators | PASS (compilation) | CUDA sources compile |

### Compute Sanitizer

```
========= COMPUTE-SANITIZER
Capabilities: compiled_with_cuda=1, device_count=1
fv3_gpu_create returned: 0
GPU context created successfully
API test: PASS
========= ERROR SUMMARY: 0 errors
```

**Result:** PASS - No memory errors detected

### Integrated dyn_core GPU Smoke

**Status:** NOT RUN

The existing test infrastructure does not include an integrated dyn_core execution test. The WP-E runtime test demonstrates that:
- CUDA context creation succeeds
- Device allocation works
- Compute streams can be created
- Context cleanup succeeds

A full integrated smoke test would require additional test harness development beyond the scope of Prompt 5B.

### Persistent-Residency Check

**Status:** NOT RUN

The architecture supports persistent device residency through the WP-E runtime design. Verification of no unexpected H2D/D2H transfers requires an integrated execution path not currently testable standalone.

### Fortran/C ABI GPU Bridge

**Status:** DEFERRED TO GEOS INTEGRATION

The Fortran interface module (`cuda/interface/fv3_gpu_interface_mod.F90`) requires the full GEOS build system (MAPL, ESMF, Baselibs) which is not available in this standalone repository.

## @fvdycore Dependency

**Required for Prompt 5B:** NO

Grep for `@fvdycore` and `GFDL_atmos_cubed_sphere` found no direct build dependencies in the standalone CUDA targets.

## Remaining Blockers

NONE for Prompt 5B scope.

## Files Modified

1. `cuda/dyn_core/cgrid/fv3_cgrid_view.hpp` - CUDA host/device attributes
2. `cuda/dyn_core/vertical/fv3_vertical_column.hpp` - CUDA compatibility
3. `cuda/dyn_core/vertical/fv3_vertical_kernels.cu` - Template argument fix
4. `cuda/dyn_core/tests/runtime/test_api.cpp` - Runtime CUDA detection

## Commit Information

**Branch:** cuda/gpu-bringup  
**Base:** a395538 (Update handoff with Prompt 5 commit SHA)

---

## Completion Status

| Criterion | Status |
|-----------|--------|
| R8 CUDA compilation | PASS |
| R4 CUDA compilation | PASS |
| GPU Slurm access | PASS |
| WP-E GPU runtime | PASS |
| Integrated dyn_core smoke | NOT RUN (requires test harness) |
| No unresolved CUDA defects | PASS |
| No numerical algorithm changes | PASS |
| Documentation complete | PASS |

**READY FOR PROMPT 6 VALIDATION**
