# CUDA/MPI Environment Compatibility Audit for Discover HPC

**Date:** 2026-08-19  
**Branch:** `cuda/gpu-bringup`  
**Base commit:** a395538 (Prompt 5)  
**Purpose:** Determine safest compiler/CUDA/MPI module combination for Prompt 5B CUDA bring-up and eventual GEOS integration

---

## Executive Summary

This audit identifies `nvidia/nvhpc-byo-compiler/24.7` as the correct CUDA module for the FV3 CUDA port. Combined with GCC 12.3.0 and OpenMPI 4.1.6, this provides a complete, tested environment for Prompt 5B standalone CUDA compilation without disrupting future GEOS integration.

**Recommended module stack:**
```bash
module purge
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

**Status:** ✅ Safe to start Prompt 5B CUDA build

---

## 1. Current Environment Baseline (GEOSenv Only)

| Component | Value |
|-----------|-------|
| GCC/G++/GFortran | System GCC 7.5.0 (`/usr/bin/gcc`) |
| nvcc | NOT FOUND |
| mpicc/mpifort | NOT FOUND |
| cmake | 3.30.3 |
| CC/CXX/FC | Unset |
| CUDA_HOME/NVHPC | Unset |

**GEOSenv** is a utility meta-module that loads development tools (git, cmake, mepo, ninja) but does **not** load compilers or MPI.

---

## 2. Available NVIDIA Modules

| Module | nvcc | Host C/C++ | Fortran | MPI | Sets CC/CXX/FC |
|--------|------|------------|---------|-----|----------------|
| `nvidia/nvhpc-byo-compiler/24.7` | ✅ CUDA 12.5 | **None** (external) | **None** | None | **No** |
| `nvidia/nvhpc-nompi/24.7` | ✅ CUDA 12.5 | nvc/nvc++ | nvfortran | None | Yes (NVIDIA) |
| `nvidia/nvhpc-hpcx-cuda12/24.7` | ✅ CUDA 12.5 | nvc/nvc++ | nvfortran | HPC-X | Yes (NVIDIA) |
| `nvidia/nvhpc-hpcx/24.7` | ✅ CUDA 12.5 | nvc/nvc++ | nvfortran | HPC-X | Yes (NVIDIA) |
| `nvidia/nvhpc-openmpi3/24.7` | ✅ CUDA 12.5 | nvc/nvc++ | nvfortran | OpenMPI 3.1.5 | Yes (NVIDIA) |
| `nvidia/nvhpc/24.7` | ✅ CUDA 12.5 | nvc/nvc++ | nvfortran | NVHPC MPI | Yes (NVIDIA) |

### Key Finding

`nvidia/nvhpc-byo-compiler/24.7` is the **only** module that:
- Provides CUDA toolkit (nvcc, libraries) without forcing NVIDIA compilers
- Does **not** set CC/CXX/FC environment variables
- Allows use with externally loaded GCC or Intel compilers
- Adds only CUDA paths, not compiler paths

This is the correct choice for mixed compiler environments like GEOS.

---

## 3. GEOS Standard Build Configuration

### Production GEOS uses:
- **Compiler:** Intel ifort 2021.13.0 (classic) + Intel MPI 2021.13.0
- **Baselibs:** ESMA-Baselibs-9.4.0
- **Alternative:** GCC builds available (gfortran 15.2.0 + OpenMPI 4.1.6)

### Available GEOS-compatible MPI modules:

| Compiler | MPI Module |
|----------|------------|
| Intel 2021.6.0 | `mpi/openmpi/4.1.5/intel-2021.6.0` |
| GCC 12.3.0 | `mpi/openmpi/4.1.6/gcc-12.3.0` |
| GCC 13.2.0 | `mpi/openmpi/4.1.6/gcc-13.2.0` |
| GCC 14.2.0 | `mpi/openmpi/4.1.6/gcc-14.2.0` |

---

## 4. Tested Module Combinations

### ✅ RECOMMENDED: GCC 12.3.0 + OpenMPI + CUDA BYO

```bash
module purge
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

**Resulting environment:**

| Tool | Path | Version |
|------|------|---------|
| gcc | `/usr/local/other/gcc/12.3.0/bin/gcc` | GCC 12.3.0 |
| g++ | `/usr/local/other/gcc/12.3.0/bin/g++` | GCC 12.3.0 |
| gfortran | `/usr/local/other/gcc/12.3.0/bin/gfortran` | GCC 12.3.0 |
| nvcc | `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/bin/nvcc` | CUDA 12.5.82 |
| mpicc | `/discover/swdev/gmao_SIteam/MPI/openmpi/4.1.6-SLES15/gcc-12.3.0/bin/mpicc` | OpenMPI 4.1.6 |
| mpifort | `/discover/swdev/gmao_SIteam/MPI/openmpi/4.1.6-SLES15/gcc-12.3.0/bin/mpifort` | OpenMPI 4.1.6 |
| cmake | `/usr/local/other/cmake/3.30.3/bin/cmake` | 3.30.3 |

**Environment variables set:**
```
CC=/usr/local/other/gcc/12.3.0/bin/gcc
CXX=/usr/local/other/gcc/12.3.0/bin/g++
FC=/usr/local/other/gcc/12.3.0/bin/gfortran
NVHPC=/usr/local/other/nvidia/hpc_sdk
NVHPC_ROOT=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7
```

**Compilation test:** ✅ Minimal CUDA program compiles successfully with nvcc + GCC 12.3.0

### ✅ ALTERNATIVE: Intel + OpenMPI + CUDA BYO

```bash
module purge
module load comp/gcc/12.3.0           # Required prereq for Intel OpenMPI
module load comp/intel/2021.6.0
module load mpi/openmpi/4.1.5/intel-2021.6.0
module load nvidia/nvhpc-byo-compiler/24.7
```

**Result:**
- icc/icpc/ifort: Intel 2021.6.0
- nvcc: CUDA 12.5.82
- mpifort wraps ifort

This provides Intel Fortran for eventual GEOS production integration.

### ❌ AVOID: Full NVHPC modules

Modules like `nvhpc-nompi`, `nvhpc-hpcx`, `nvhpc-openmpi3`:
- Override CC/CXX/FC to NVIDIA compilers
- Would require rebuilding all GEOS dependencies with nvfortran
- Not ABI-compatible with existing Intel/GCC GEOS Baselibs

---

## 5. Candidate Evaluation Matrix

| Configuration | CUDA | Host C/C++ | Fortran | MPI | GEOS Compat | Prompt 5B | Risks |
|---------------|------|------------|---------|-----|-------------|-----------|-------|
| **GEOSenv + GCC 12.3.0 + OpenMPI 4.1.6 + BYO** | 12.5 | GCC 12.3.0 | gfortran 12.3.0 | OpenMPI 4.1.6 | ⚠️ Partial | ✅ Excellent | GCC vs Intel production |
| GEOSenv + Intel 2021.6.0 + OpenMPI 4.1.5 + BYO | 12.5 | GCC 12.3.0 | ifort 2021.6.0 | OpenMPI 4.1.5 | ✅ Good | ✅ Good | Mixed compiler stack |
| GEOSenv + BYO alone (system GCC 7.5.0) | 12.5 | GCC 7.5.0 | gfortran 7.5.0 | None | ❌ Poor | ⚠️ Minimal | Ancient compiler, no MPI |
| nvhpc-nompi/24.7 | 12.5 | nvc/nvc++ | nvfortran | None | ❌ Incompatible | ⚠️ Possible | No GEOS Baselibs |

---

## 6. BYO Compiler Module Details

**Module:** `nvidia/nvhpc-byo-compiler/24.7`

**What it provides:**
- CUDA toolkit 12.5 (nvcc, cudart, cuBLAS, cuFFT, etc.)
- CUDA include paths via CPATH
- CUDA library paths via LD_LIBRARY_PATH
- NCCL and NVSHMEM communication libraries
- NVHPC and NVHPC_ROOT environment variables

**What it does NOT provide:**
- Host compilers (nvc, nvc++, nvfortran)
- CC, CXX, FC environment variable settings
- MPI implementation

**Supported host compilers for CUDA 12.5:**
- GCC 7.x through 13.x (tested: 7.5.0 ✅, 12.3.0 ✅)
- Intel classic compilers (icc/icpc as nvcc host)

**Module conflicts:** All other nvidia/nvhpc-* modules

---

## 7. GPU Node Requirements

| Task | Login Node | GPU Node Required |
|------|------------|-------------------|
| Module loading/inspection | ✅ Yes | No |
| `nvcc --version` | ✅ Yes | No |
| CUDA compilation | ✅ Yes | No |
| `nvidia-smi` | ❌ No | Yes |
| GPU runtime execution | ❌ No | Yes |
| CUDA memory allocation | ❌ No | Yes |

### Slurm GPU Partition Info

**Partition:** `gpu_a100`
- Nodes: warpa003-warpa013 (10 nodes total)
- GPUs: 4× NVIDIA A100 per node (40 total)
- CPUs: 48 per node (AMD Rome)
- Memory: 510 GB per node
- Default allocation: 12 CPUs per GPU, 120 GB per GPU
- Max walltime: 24 hours

### Interactive GPU Allocation Command

```bash
salloc --partition=gpu_a100 --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:1 --time=01:00:00
```

For multi-GPU testing:
```bash
salloc --partition=gpu_a100 --nodes=1 --ntasks=4 --cpus-per-task=12 --gres=gpu:4 --time=02:00:00
```

---

## 8. Recommendations

### Best for Prompt 5B Standalone CUDA

```bash
module purge
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

**Rationale:**
- GCC 12.3.0 is well within CUDA 12.5 supported range
- Modern C++17 support (required by FV3 CUDA code)
- MPI available when needed
- Clean separation of concerns
- Tested and verified working

### Best for Eventual Full GEOS Integration

**Option A - GCC-based GEOS build:**

Same as Prompt 5B stack (matches available GCC Baselibs)

**Option B - Intel-based GEOS build:**

```bash
module purge
module load comp/gcc/12.3.0
module load comp/intel/2021.6.0
module load mpi/openmpi/4.1.5/intel-2021.6.0
module load nvidia/nvhpc-byo-compiler/24.7
```

Then configure CMake with:
```cmake
CMAKE_Fortran_COMPILER=ifort
CMAKE_CUDA_HOST_COMPILER=/usr/local/other/gcc/12.3.0/bin/g++
```

The same BYO CUDA module works for both paths.

---

## 9. Final Decision

### Recommended Prompt 5B Module Stack

```bash
module purge
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

### Component Summary

| Component | Value |
|-----------|-------|
| **CUDA toolkit** | 12.5.82 |
| **Host compiler** | GCC 12.3.0 |
| **Fortran compiler** | gfortran 12.3.0 |
| **MPI** | OpenMPI 4.1.6 |
| **CMake** | 3.30.3 |

### Compatibility Assessment

| Question | Answer |
|----------|--------|
| Compatible with current GEOS environment? | **PARTIAL** - Uses GCC instead of Intel production stack |
| GCC-based GEOS Baselibs available? | Yes (gfortran 15.2.0 builds exist) |
| Safe to start Prompt 5B CUDA build? | **YES** ✅ |
| Unresolved environment blockers? | **None** |

### Quick Start Commands

```bash
# On login node - load environment
module purge
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7

# Verify toolchain
which nvcc && nvcc --version
which g++ && g++ --version
which cmake && cmake --version

# For GPU runtime testing
salloc --partition=gpu_a100 --nodes=1 --gres=gpu:1 --time=01:00:00

# On GPU node - verify GPU access
nvidia-smi
nvidia-smi -L
```

---

## Appendix: Module Details

### nvidia/nvhpc-byo-compiler/24.7

```
setenv("NVHPC","/usr/local/other/nvidia/hpc_sdk")
setenv("NVHPC_ROOT","/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7")
prepend_path("PATH",".../24.7/cuda/bin")
prepend_path("CPATH",".../24.7/cuda/include")
prepend_path("CPATH",".../24.7/math_libs/include")
prepend_path("CPATH",".../24.7/comm_libs/nccl/include")
prepend_path("CPATH",".../24.7/comm_libs/nvshmem/include")
prepend_path("LD_LIBRARY_PATH",".../24.7/cuda/lib64")
prepend_path("LD_LIBRARY_PATH",".../24.7/cuda/extras/CUPTI/lib64")
prepend_path("LD_LIBRARY_PATH",".../24.7/math_libs/lib64")
prepend_path("LD_LIBRARY_PATH",".../24.7/comm_libs/nccl/lib")
prepend_path("LD_LIBRARY_PATH",".../24.7/comm_libs/nvshmem/lib")
```

### comp/gcc/12.3.0

```
setenv("CC","/usr/local/other/gcc/12.3.0/bin/gcc")
setenv("CXX","/usr/local/other/gcc/12.3.0/bin/g++")
setenv("FC","/usr/local/other/gcc/12.3.0/bin/gfortran")
prepend_path("PATH","/usr/local/other/gcc/12.3.0/bin")
prepend_path("LD_LIBRARY_PATH","/usr/local/other/gcc/12.3.0/lib64")
```

### mpi/openmpi/4.1.6/gcc-12.3.0

```
setenv("MPI_HOME","/discover/swdev/gmao_SIteam/MPI/openmpi/4.1.6-SLES15/gcc-12.3.0")
prepend_path("PATH",".../bin")
prepend_path("LD_LIBRARY_PATH",".../lib")
```
