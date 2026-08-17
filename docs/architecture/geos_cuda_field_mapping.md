# GEOS/FV3 CUDA Field Mapping

## Overview

This document maps FV3 Fortran state arrays from `fv_atmos_type` to WP-E
`FieldId` entries for the CUDA dyn_core interface.

## Field access pattern

Fortran state is accessed via the global `FV_Atm(1)` structure (type
`fv_atmos_type`). Array bounds come from `FV_Atm(1)%bd`.

```fortran
isc = FV_Atm(1)%bd%isc   ! compute start i
iec = FV_Atm(1)%bd%iec   ! compute end i
jsc = FV_Atm(1)%bd%jsc   ! compute start j
jec = FV_Atm(1)%bd%jec   ! compute end j
isd = FV_Atm(1)%bd%isd   ! data start i (includes halo)
ied = FV_Atm(1)%bd%ied   ! data end i
jsd = FV_Atm(1)%bd%jsd   ! data start j
jed = FV_Atm(1)%bd%jed   ! data end j
npx = FV_Atm(1)%npx      ! global x points
npy = FV_Atm(1)%npy      ! global y points
npz = FV_Atm(1)%npz      ! vertical levels
ng  = FV_Atm(1)%ng       ! halo width
```

## Mutable state fields

### Primary prognostic fields

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | H2D init | Sync back | Notes |
|--------------|------------|-----------|---------|--------------|----------|-----------|-------|
| `FV_Atm(1)%u` | (isd:ied,jsd:jed+1,npz) | FVPRC | D_U | `u` | YES | YES | D-grid u-wind |
| `FV_Atm(1)%v` | (isd:ied+1,jsd:jed,npz) | FVPRC | D_V | `v` | YES | YES | D-grid v-wind |
| `FV_Atm(1)%w` | (isd:ied,jsd:jed,npz) | FVPRC | A | `w` | YES | YES | Vertical velocity (NH) |
| `FV_Atm(1)%delz` | (isd:ied,jsd:jed,npz) | FVPRC | A | `delz` | YES | YES | Layer thickness (NH) |
| `FV_Atm(1)%pt` | (isd:ied,jsd:jed,npz) | FVPRC | A | `pt` | YES | YES | Potential temperature |
| `FV_Atm(1)%delp` | (isd:ied,jsd:jed,npz) | FVPRC | A | `delp` | YES | YES | Pressure thickness |
| `FV_Atm(1)%q` | (isd:ied,jsd:jed,npz,ncnst) | FVPRC | A | `q` | YES | YES | Tracers |
| `FV_Atm(1)%q_con` | (isd:ied,jsd:jed,npz) | FVPRC | A | `q_con` | YES | YES | Condensate sum |

### Pressure/thermodynamic fields

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | H2D init | Sync back | Notes |
|--------------|------------|-----------|---------|--------------|----------|-----------|-------|
| `FV_Atm(1)%pe` | (is-1:ie+1,npz+1,js-1:je+1) | FVPRC | INTERFACE | `pe` | YES | YES | Edge pressure |
| `FV_Atm(1)%pk` | (is:ie,js:je,npz+1) | FVPRC | INTERFACE | `pk` | YES | YES | p^kappa at edges |
| `FV_Atm(1)%peln` | (is:ie,npz+1,js:je) | FVPRC | INTERFACE | `peln` | YES | YES | ln(p) at edges |
| `FV_Atm(1)%pkz` | (is:ie,js:je,npz) | FVPRC | A | `pkz` | YES | YES | p^kappa at layers |
| `FV_Atm(1)%ps` | (isd:ied,jsd:jed) | FVPRC | A | — | NO | — | Surface pressure (derived) |

### Diagnostic/derived fields

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | H2D init | Sync back | Notes |
|--------------|------------|-----------|---------|--------------|----------|-----------|-------|
| `FV_Atm(1)%omga` | (isd:ied,jsd:jed,npz) | FVPRC | A | `omga` | NO | YES | Omega (computed) |
| `FV_Atm(1)%ua` | (isd:ied,jsd:jed,npz) | FVPRC | A | `ua` | YES | YES | A-grid u-wind |
| `FV_Atm(1)%va` | (isd:ied,jsd:jed,npz) | FVPRC | A | `va` | YES | YES | A-grid v-wind |
| `FV_Atm(1)%uc` | (isd:ied+1,jsd:jed,npz) | FVPRC | C_U | `uc` | YES | YES | C-grid u-wind |
| `FV_Atm(1)%vc` | (isd:ied,jsd:jed+1,npz) | FVPRC | C_V | `vc` | YES | YES | C-grid v-wind |
| `FV_Atm(1)%diss_est` | (isd:ied,jsd:jed,npz) | FVPRC | A | `diss_est` | NO | YES | Dissipation estimate |

### Mass flux fields

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | H2D init | Sync back | Notes |
|--------------|------------|-----------|---------|--------------|----------|-----------|-------|
| `FV_Atm(1)%mfx` | (is:ie+1,js:je,npz) | FVPRC | X_FACE | `mfx` | YES | YES | X mass flux |
| `FV_Atm(1)%mfy` | (is:ie,js:je+1,npz) | FVPRC | Y_FACE | `mfy` | YES | YES | Y mass flux |
| `FV_Atm(1)%cx` | (is:ie+1,jsd:jed,npz) | FVPRC | X_FACE | `cx` | YES | YES | Courant X |
| `FV_Atm(1)%cy` | (isd:ied,js:je+1,npz) | FVPRC | Y_FACE | `cy` | YES | YES | Courant Y |

## Immutable grid fields

### 1D vertical coordinate

| Fortran field | Dimensions | Precision | WP-E FieldId | Notes |
|--------------|------------|-----------|--------------|-------|
| `FV_Atm(1)%ak` | (npz+1) | R_GRID | `ak` | Hybrid A coefficient |
| `FV_Atm(1)%bk` | (npz+1) | R_GRID | `bk` | Hybrid B coefficient |

### 2D surface/geometry

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | Notes |
|--------------|------------|-----------|---------|--------------|-------|
| `FV_Atm(1)%phis` | (isd:ied,jsd:jed) | FVPRC | A | `phis` | Surface geopotential |
| `FV_Atm(1)%gridstruct%area` | (isd:ied,jsd:jed) | R_GRID | A | `area` | Cell area |
| `FV_Atm(1)%gridstruct%rarea` | (isd:ied,jsd:jed) | R_GRID | A | `rarea` | 1/area |
| `FV_Atm(1)%gridstruct%dx` | (isd:ied,jsd:jed+1) | R_GRID | X_FACE | `dx` | X edge length |
| `FV_Atm(1)%gridstruct%dy` | (isd:ied+1,jsd:jed) | R_GRID | Y_FACE | `dy` | Y edge length |
| `FV_Atm(1)%gridstruct%dxc` | (isd:ied+1,jsd:jed) | R_GRID | C_U | `dxc` | C-grid X length |
| `FV_Atm(1)%gridstruct%dyc` | (isd:ied,jsd:jed+1) | R_GRID | C_V | `dyc` | C-grid Y length |
| `FV_Atm(1)%gridstruct%rdx` | (isd:ied,jsd:jed+1) | R_GRID | X_FACE | `rdx` | 1/dx |
| `FV_Atm(1)%gridstruct%rdy` | (isd:ied+1,jsd:jed) | R_GRID | Y_FACE | `rdy` | 1/dy |
| `FV_Atm(1)%gridstruct%rdxc` | (isd:ied+1,jsd:jed) | R_GRID | C_U | `rdxc` | 1/dxc |
| `FV_Atm(1)%gridstruct%rdyc` | (isd:ied,jsd:jed+1) | R_GRID | C_V | `rdyc` | 1/dyc |

### D-grid metric terms

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | Notes |
|--------------|------------|-----------|---------|--------------|-------|
| `FV_Atm(1)%gridstruct%cosa_u` | (isd:ied,jsd:jed+1) | R_GRID | D_U | `cosa_u` | cos(angle) at D_U |
| `FV_Atm(1)%gridstruct%cosa_v` | (isd:ied+1,jsd:jed) | R_GRID | D_V | `cosa_v` | cos(angle) at D_V |
| `FV_Atm(1)%gridstruct%sina_u` | (isd:ied,jsd:jed+1) | R_GRID | D_U | `sina_u` | sin(angle) at D_U |
| `FV_Atm(1)%gridstruct%sina_v` | (isd:ied+1,jsd:jed) | R_GRID | D_V | `sina_v` | sin(angle) at D_V |
| `FV_Atm(1)%gridstruct%rsin_u` | (isd:ied,jsd:jed+1) | R_GRID | D_U | `rsin_u` | 1/sin at D_U |
| `FV_Atm(1)%gridstruct%rsin_v` | (isd:ied+1,jsd:jed) | R_GRID | D_V | `rsin_v` | 1/sin at D_V |

### A-grid metric terms

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | Notes |
|--------------|------------|-----------|---------|--------------|-------|
| `FV_Atm(1)%gridstruct%cosa_s` | (isd:ied,jsd:jed) | R_GRID | A | `cosa_s` | cos(angle) at A |
| `FV_Atm(1)%gridstruct%rsin2` | (isd:ied,jsd:jed) | R_GRID | A | `rsin2` | 1/sin^2 at A |
| `FV_Atm(1)%gridstruct%dxa` | (isd:ied,jsd:jed) | R_GRID | A | `dxa` | X length at A |
| `FV_Atm(1)%gridstruct%dya` | (isd:ied,jsd:jed) | R_GRID | A | `dya` | Y length at A |

### Corner (B-grid) terms

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | Notes |
|--------------|------------|-----------|---------|--------------|-------|
| `FV_Atm(1)%gridstruct%sin_sg` | (isd:ied,jsd:jed,9) | R_GRID | B | `sin_sg` | Sub-grid sin |
| `FV_Atm(1)%gridstruct%cos_sg` | (isd:ied,jsd:jed,9) | R_GRID | B | `cos_sg` | Sub-grid cos |
| `FV_Atm(1)%gridstruct%f0` | (isd:ied+1,jsd:jed+1) | R_GRID | B | `f0` | Coriolis at corners |
| `FV_Atm(1)%gridstruct%rarea_c` | (isd:ied+1,jsd:jed+1) | R_GRID | B | `rarea_c` | 1/area at corners |

### Edge vectors (rank-1)

| Fortran field | Dimensions | Precision | Stagger | WP-E FieldId | Notes |
|--------------|------------|-----------|---------|--------------|-------|
| `FV_Atm(1)%gridstruct%edge_w` | (npz+1) | R_GRID | Y_FACE | `edge_w` | West edge |
| `FV_Atm(1)%gridstruct%edge_e` | (npz+1) | R_GRID | Y_FACE | `edge_e` | East edge |
| `FV_Atm(1)%gridstruct%edge_s` | (npx+1) | R_GRID | X_FACE | `edge_s` | South edge |
| `FV_Atm(1)%gridstruct%edge_n` | (npx+1) | R_GRID | X_FACE | `edge_n` | North edge |

## Configuration mapping

### Numerical configuration from flagstruct

| Fortran field | Type | WP-E NumericalConfig | Notes |
|--------------|------|---------------------|-------|
| `FV_Atm(1)%flagstruct%hord_mt` | int | `hord_mt` | Momentum transport order |
| `FV_Atm(1)%flagstruct%hord_vt` | int | `hord_vt` | Vorticity transport order |
| `FV_Atm(1)%flagstruct%hord_tm` | int | `hord_tm` | Scalar transport order |
| `FV_Atm(1)%flagstruct%hord_dp` | int | `hord_dp` | Mass transport order |
| `FV_Atm(1)%flagstruct%hord_tr` | int | `hord_tr` | Tracer transport order |
| `FV_Atm(1)%flagstruct%nord` | int | `nord` | Divergence damping order |
| `FV_Atm(1)%flagstruct%dddmp` | real | `dddmp` | Divergence damping coeff |
| `FV_Atm(1)%flagstruct%d_con` | real | `d_con` | Dissipative heating coeff |
| `FV_Atm(1)%flagstruct%vtdm4` | real | — | Vorticity damping (to `damping_v` array) |
| `FV_Atm(1)%flagstruct%do_vort_damp` | logical | — | Enable vorticity damping |
| `FV_Atm(1)%flagstruct%hydrostatic` | logical | `hydrostatic` | Hydrostatic mode |
| `FV_Atm(1)%flagstruct%grid_type` | int | `grid_type` | Grid type (0=cubed-sphere) |
| `FV_Atm(1)%ptop` | real | `ptop` | Model top pressure |
| `FV_Atm(1)%ks` | int | — | Pure pressure levels |

### Vertical solver configuration

| Fortran field | Type | WP-E NumericalConfig | Notes |
|--------------|------|---------------------|-------|
| `FV_Atm(1)%flagstruct%a_imp` | real | `a_imp` | Implicit weight (1.0=SIM1) |
| `FV_Atm(1)%flagstruct%beta` | real | `beta` | Pressure gradient weight |

### Constants

| Fortran source | Value | WP-E NumericalConfig | Notes |
|----------------|-------|---------------------|-------|
| `MAPL_KAPPA` | cp/cv | `akap` | Adiabatic exponent |
| `MAPL_GRAV` | 9.80665 | `grav` | Gravity |
| `MAPL_RGAS` | 287.0 | `rdgas` | Gas constant |

## Precision handling

GEOS FV3 supports two precision modes:

| Mode | FVPRC | R_GRID | WP-E fv_real |
|------|-------|--------|--------------|
| R4 | REAL4 | REAL8 | float |
| R8 | REAL8 | REAL8 | double |

The interface must:
1. Query `FV_PRECISION` at build time
2. Set `fv3_gpu_precision` accordingly
3. Use matching `fv3_gpu_element_type` in field descriptors

## Array layout notes

### Fortran array ordering

FV3 uses Fortran column-major ordering with index patterns:
- 2D: `(i,j)` where i is fast, j is slow
- 3D: `(i,j,k)` or `(i,k,j)` depending on field
- Halo regions: `isd:ied` includes `ng` halo points each side

### Stride calculation

For a Fortran array `A(isd:ied, jsd:jed, 1:npz)`:
```
stride(1) = 1                           ! i stride
stride(2) = ied - isd + 1               ! j stride  
stride(3) = (ied - isd + 1) * (jed - jsd + 1)  ! k stride
```

### WP-E view construction

The C ABI receives raw pointers. WP-E constructs `DeviceView` from:
- `lower[]: [isd, jsd, 1]`
- `extent[]: [ied-isd+1, jed-jsd+1, npz]`
- `stride_elements[]`: Fortran strides

## Host/device residency

### Initial H2D transfer (required)

All mutable state and immutable grid fields are transferred H2D during
`fv3_gpu_initialize_state()`. This is a one-time bulk transfer.

### During dyn_core execution

Fields remain GPU-resident. No H2D or D2H transfers within the acoustic loop.

### Synchronization boundaries (D2H required)

| Boundary | Triggered by | Fields affected |
|----------|--------------|-----------------|
| Halo exchange | MPI communication | Boundary regions only |
| Physics coupling | End of dynamics | All prognostic fields |
| Diagnostics/I/O | User request | Selected fields |
| Savepoint validation | Debug mode | All fields |

### MPI halo exchange

Stage 1 uses host-staged halo exchange:
1. D2H transfer of boundary regions
2. MPI_Sendrecv on host
3. H2D transfer of received halos

Future stages may use GPU-direct MPI or NCCL.

## Field count summary

| Category | Count |
|----------|-------|
| Mutable state (3D) | 18 |
| Mutable state (2D) | 2 |
| Immutable grid (1D) | 2 |
| Immutable grid (2D) | ~35 |
| Shared temporary | 16 |
| **Total registered** | ~73 |
