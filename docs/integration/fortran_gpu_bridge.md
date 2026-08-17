# GEOS/FV3 CUDA Fortran-GPU Bridge

## Overview

This document describes the Fortran-to-GPU bridge that connects the GEOS/FV3
atmospheric model to the WP-E CUDA dyn_core subsystem.

## Architecture

```
GEOS/FVdycoreCubed GridComp
        │
        ▼
   FV_StateMod.F90
   (FV_Setup, FV_InitState, FV_Run, FV_Finalize)
        │
        ▼
   fv3_gpu_interface_mod.F90
   (ISO_C_BINDING wrapper)
        │
        ▼
   fv3_gpu_api.h (C ABI)
        │
        ▼
   WP-E GPU Runtime
   (fv3_gpu_runtime.cpp)
        │
        ▼
   Integrated CUDA dyn_core
   (WP-A/B/C/D kernels)
```

## Selected Fortran call site

### Initialization

**File:** `FV_StateMod.F90`
**Subroutine:** `FV_InitState`
**Location:** After `fv_init2` and before `RETURN_(ESMF_SUCCESS)`

The CUDA context is created here using:
- `FV_Atm(1)` state structure
- MPI communicator from ESMF VM
- Domain bounds from `FV_Atm(1)%bd`

### Runtime dispatch

**File:** `FV_StateMod.F90`
**Subroutine:** `FV_Run`
**Location:** Inside the `if (.not. FV_OFF)` block, replacing `fv_dynamics()` call

The dispatch logic:
1. Check `fv3_dyncore_backend == 1` and `fv3_cuda_initialized`
2. If CUDA: call `fv3_gpu_interface_step()`, then `fv3_gpu_interface_sync()`
3. If Fortran: call original `fv_dynamics()` path

### Finalization

**File:** `FV_StateMod.F90`
**Subroutine:** `FV_Finalize`
**Location:** After the `RUN_GTFV3` finalization block

## Backend selection mechanism

### Compile-time selection

The CUDA backend requires the `USE_FV3_CUDA` preprocessor define:

```cmake
option(BUILD_FV3_CUDA_INTERFACE "Build FV3 CUDA dyn_core interface" OFF)

if (BUILD_FV3_CUDA_INTERFACE)
  add_definitions(-DUSE_FV3_CUDA)
endif()
```

### Runtime selection

The runtime backend is controlled by the module variable `fv3_dyncore_backend`:

| Value | Backend |
|-------|---------|
| 0 | Fortran (default) |
| 1 | CUDA |

Currently set at module initialization. Future work may read from namelist.

## Init/step/finalize lifecycle

### 1. fv3_gpu_interface_init

Called once during `FV_InitState`:

```fortran
call fv3_gpu_interface_init(FV_Atm(1), comm, mpi_rank, mpi_size, rc)
```

Actions:
1. Create `fv3_gpu_create_desc` from FV_Atm bounds
2. Call `fv3_gpu_create()` to allocate GPU context
3. Build field descriptors for all state/grid arrays
4. Call `fv3_gpu_initialize_state()` for H2D transfer
5. Wait for completion

### 2. fv3_gpu_interface_step

Called every `FV_Run` invocation:

```fortran
call fv3_gpu_interface_step(FV_Atm(1), myDT, k_split, n_split, &
                            init_step, end_step, rc)
```

Actions:
1. Build `fv3_gpu_step_desc` from parameters
2. Call `fv3_gpu_dyn_core_step()` to execute GPU kernels
3. Wait for completion

### 3. fv3_gpu_interface_sync

Called after step to transfer results to host:

```fortran
call fv3_gpu_interface_sync(.true., rc)  ! sync all
```

Actions:
1. Build field mask for requested fields
2. Call `fv3_gpu_sync_to_host()` for D2H transfer
3. Wait for completion

### 4. fv3_gpu_interface_finalize

Called during `FV_Finalize`:

```fortran
call fv3_gpu_interface_finalize(rc)
```

Actions:
1. Call `fv3_gpu_finalize()` to release GPU resources
2. Clear module state

## C ABI used

The interface uses the stable WP-E C ABI defined in `fv3_gpu_api.h`:

| Function | Purpose |
|----------|---------|
| `fv3_gpu_create` | Create GPU context |
| `fv3_gpu_initialize_state` | Register fields, H2D transfer |
| `fv3_gpu_dyn_core_step` | Execute dyn_core step |
| `fv3_gpu_sync_to_host` | D2H transfer |
| `fv3_gpu_event_wait` | Synchronize |
| `fv3_gpu_finalize` | Release context |

ABI version: 1

## State mapping

See `docs/architecture/geos_cuda_field_mapping.md` for complete field mapping.

Summary:
- 18 mutable 3D state fields (u, v, w, delz, pt, delp, q, ...)
- 2 mutable 2D state fields (ps, ws)
- ~40 immutable grid fields
- 16 shared temporary fields (allocated by WP-E)

## Configuration mapping

| GEOS source | WP-E destination |
|-------------|------------------|
| `FV_Atm(1)%flagstruct%hord_mt` | `NumericalConfig::hord_mt` |
| `FV_Atm(1)%flagstruct%hord_vt` | `NumericalConfig::hord_vt` |
| `FV_Atm(1)%flagstruct%hord_tm` | `NumericalConfig::hord_tm` |
| `FV_Atm(1)%flagstruct%hord_dp` | `NumericalConfig::hord_dp` |
| `FV_Atm(1)%flagstruct%hydrostatic` | `NumericalConfig::hydrostatic` |
| `FV_Atm(1)%flagstruct%a_imp` | `NumericalConfig::a_imp` |
| `FV_Atm(1)%ptop` | `NumericalConfig::ptop` |

Actual GEOS first-run values:
- `hord_mt/vt/tm = 6`
- `hord_dp = -6`
- `hydrostatic = false`
- `a_imp = 1.0`

## MPI mapping

MPI communicator is obtained from ESMF:

```fortran
call ESMF_VMGet(VM, mpiCommunicator=comm, localPet=mpi_rank, petCount=mpi_size)
```

The communicator token is passed through `fv3_gpu_create_desc::communicator_token`.

Stage 1 uses host-staged halo exchange:
1. D2H transfer of boundary regions
2. MPI communication via `mpp_update_domains`
3. H2D transfer of received halos

## Host/device residency

See `docs/architecture/geos_cuda_residency_plan.md` for full details.

Summary:
- State fields are GPU-resident after initialization
- Grid fields are GPU-resident (immutable)
- D2H sync required only at physics/diagnostics boundaries
- Halos exchanged via host-staged MPI (Stage 1)

## Synchronization boundaries

| Boundary | Trigger | Action |
|----------|---------|--------|
| Halo exchange | MPI required | D2H → MPI → H2D |
| Physics coupling | End of dynamics | D2H all prognostic |
| Diagnostics/I/O | User request | D2H selected fields |
| Savepoint | Debug mode | D2H all fields |
| Finalization | Model end | D2H, release GPU |

## Build changes

### CMakeLists.txt modifications

1. Added `BUILD_FV3_CUDA_INTERFACE` option
2. Conditional `add_subdirectory(cuda/dyn_core)`
3. Add `cuda/interface/fv3_gpu_interface_mod.F90` to sources
4. Add `-DUSE_FV3_CUDA` compile definition
5. Link `fv3_dyn_core_gpu` library

### FV_StateMod.F90 modifications

1. Added `USE_FV3_CUDA` conditional module use
2. Added `fv3_dyncore_backend` and `fv3_cuda_initialized` state
3. Added CUDA init in `FV_InitState`
4. Added CUDA dispatch in `FV_Run`
5. Added CUDA finalize in `FV_Finalize`

## Files modified

| File | Change |
|------|--------|
| `CMakeLists.txt` | Added BUILD_FV3_CUDA_INTERFACE option |
| `FV_StateMod.F90` | Added CUDA interface hooks |
| `cuda/interface/fv3_gpu_interface_mod.F90` | New: ISO_C_BINDING wrapper |
| `cuda/interface/CMakeLists.txt` | New: Interface build |
| `cuda/interface/tests/test_interface_c.cpp` | New: C ABI test |

## Tests run

### Interface C ABI test

Tests the Fortran→C ABI layer:
- Struct sizes and layout
- Field descriptor validation
- Create/step/error descriptors
- Stagger types
- Field masks
- Context lifecycle (graceful failure without GPU)
- Configuration propagation

### Existing WP-E/A/B/C/D tests

All 20 integrated tests (10 R4 + 10 R8) continue to pass.

## CUDA compile status

**NOT RUN** — nvcc unavailable in current environment

## GPU execution status

**NOT RUN** — no GPU available

## Remaining blockers

1. **CUDA compilation** — Requires nvcc and CUDA toolkit
2. **GPU smoke test** — Requires GPU hardware
3. **Numerical validation** — Requires Fortran savepoints
4. **MPI halo exchange** — Host-staged implementation needed
5. **Precision matching** — R4/R8 runtime selection
