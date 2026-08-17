# First-run GEOS FV3 CUDA configuration

## Overview

This document traces the actual GEOS/FV3 configuration values into `dyn_core`
to determine the exact numerical settings required for the milestone-one CUDA
implementation.

## Configuration source hierarchy

1. **FV3 defaults**: `fv_arrays.F90` type definitions with initial values
2. **GEOS overrides**: `FV_StateMod.F90` explicit settings based on `FV3_CONFIG`
3. **Runtime namelist**: May override GEOS defaults (not examined here)

## GEOS configuration modes

GEOS supports multiple configuration modes via `FV3_CONFIG`:

| Mode | hydrostatic | hord_* | Description |
|------|-------------|--------|-------------|
| `MONOTONIC` | true | 10 | Monotonic advection |
| `MERRA-2` | true | 10 | MERRA-2 reanalysis |
| **Default (npz=72)** | true | 5/6/-6 | Standard L72 hydrostatic |
| **Default (npz≠72)** | **false** | 6/-6 | **Nonhydrostatic** |

The milestone-one CUDA target is the **nonhydrostatic default mode** (`npz≠72`).

## Complete first-run configuration

### Transport orders

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `hord_mt` | **6** | `FV_StateMod.F90:726` | `d_sw` momentum | **NO** | **YES** |
| `hord_vt` | **6** | `FV_StateMod.F90:727` | `d_sw` vorticity | **NO** | **YES** |
| `hord_tm` | **6** | `FV_StateMod.F90:728` | `d_sw` scalars, `update_dz_d` | **NO** | **YES** |
| `hord_dp` | **-6** | `FV_StateMod.F90:729` | `d_sw` mass transport | **NO** | **YES** |
| `hord_tr` | **8** | `FV_StateMod.F90:704` | tracer transport (outside `dyn_core`) | YES | No |

**Critical finding:** GEOS nonhydrostatic uses `hord=6` and `hord=-6`, NOT
`hord=1` or `hord=8`. The current CUDA translation supports only `hord=1,8`.

### Damping orders

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `nord` | **2** | `FV_StateMod.F90:671` | `d_sw` divergence damping | YES | YES |
| `nord_v` | derived | `dyn_core.F90:669` `min(2,nord)` | vertical damping | YES | YES |
| `nord_w` | derived | `dyn_core.F90:679-711` | w damping | YES | YES |
| `nord_t` | derived | `dyn_core.F90:680` | thermal damping | YES | YES |

### Vorticity damping

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `do_vort_damp` | **true** | `FV_StateMod.F90:733` | `d_sw` vorticity flux | YES | YES |
| `vtdm4` | **0.01-0.04** | `FV_StateMod.F90:734-754` | damping coefficient | YES | YES |

### Dissipation

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `d_con` | **1.0** | `FV_StateMod.F90:732` | dissipative heating | YES | YES |
| `dddmp` | **0.2** | `FV_StateMod.F90:672` | Smagorinsky coefficient | YES (approx) | YES |

### Tracer transport

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `inline_q` | **false** | `fv_arrays.F90:547` (non-GFS) | tracer inside `d_sw` | N/A | No |

### Hydrostatic mode

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `hydrostatic` | **false** | `FV_StateMod.F90:725` | entire `dyn_core` | YES | YES |

### Grid type

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `grid_type` | **0** | `fv_arrays.F90:270` | global cubed-sphere | YES | YES |
| `nested` | **false** | default | nest coupling | N/A (excluded) | No |

### Vertical solver

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `a_imp` | **1.0** | `FV_StateMod.F90:696` | SIM1 solver selection | YES | YES |
| `beta` | **0.0** | `FV_StateMod.F90:695` | pressure gradient | YES | YES |

### Height transport

| Option | Actual value | Source | Consumed in Fortran | CUDA support | Required |
|--------|--------------|--------|---------------------|--------------|----------|
| `update_dz_d` hord | **hord_tm=6** | `dyn_core.F90:847` | height transport | **NO** | **YES** |

## Branches controlling specific features

### `deln_flux` (transport flux damping)

Called when `damp_c > 1.e-4` in `fv_tp_2d`. The `damp_c` value comes from:
- `damp_v = vtdm4` for vorticity/delp transport
- `damp_t = vtdm4` for thermal transport

With `vtdm4 > 0` (GEOS nonhydrostatic), `deln_flux` **IS called**.

| Branch | Condition | GEOS value | Active | CUDA support | Required |
|--------|-----------|------------|--------|--------------|----------|
| `deln_flux` | `damp_c > 1.e-4` | `vtdm4=0.01+` | **YES** | **NO** | **YES** |

### Contravariant edge corrections

Called when `grid_type < 3` and at panel corners (`sw_corner`, etc.).

| Branch | Condition | GEOS value | Active | CUDA support | Required |
|--------|-----------|------------|--------|--------------|----------|
| Contravariant edges | `grid_type < 3` | `grid_type=0` | **YES** | **NO** | **YES** |
| Corner corrections | `sw_corner`, etc. | depends on rank | **YES** | PARTIAL | **YES** |

### `update_dz_d` transport

Uses `fv_tp_2d` with `hord=hord_tm` and `del6_vt_flux` when `damp(k) > 1.e-5`.

| Branch | Condition | GEOS value | Active | CUDA support | Required |
|--------|-----------|------------|--------|--------------|----------|
| Height transport | always | `hord_tm=6` | **YES** | **NO** | **YES** |
| Height damping | `damp(k) > 1.e-5` | `damp_vt > 0` | **YES** | **NO** | **YES** |

## Summary: Required but unsupported

The following features are **required** by GEOS nonhydrostatic configuration
but are **not supported** by the current CUDA translation:

| Feature | GEOS value | Current CUDA | Gap |
|---------|------------|--------------|-----|
| **hord=6** | `hord_mt/vt/tm=6` | Only 1,8 | **PPM order 6** |
| **hord=-6** | `hord_dp=-6` | Only 1,8 | **PPM order 6 negative** |
| **deln_flux** | `vtdm4 > 0` | Not implemented | **Transport flux damping** |
| **Contravariant edges** | `grid_type=0` | Not implemented | **Panel edge corrections** |
| **update_dz_d transport** | `hord_tm=6` | Not wired | **Cross-WP boundary** |

## Minimum implementation required

To run the first legitimate GEOS nonhydrostatic configuration:

### 1. PPM order 6 (`hord=6` and `hord=-6`)

**Location:** `wp_b_kernels.hpp`

Must implement:
- `xppm_hord6` / `yppm_hord6` — 6th-order reconstruction with monotonicity
- Negative `hord` handling — mass-weighted coordinate (same algorithm, different sign convention)

This is required for:
- Momentum transport (`hord_mt=6`)
- Vorticity transport (`hord_vt=6`)
- Scalar transport (`hord_tm=6`)
- Mass transport (`hord_dp=-6`)

### 2. `deln_flux` transport damping

**Location:** `wp_b_kernels.hpp` or new file

Must implement the transport flux damping used when `damp_c > 1.e-4`:
- `del6_vt_flux` for height transport
- Integration with `fv_tp_2d` when nord/damp_c are present

### 3. Contravariant edge corrections

**Location:** `wp_b_kernels.hpp`

Must implement the panel boundary `ut/vt` coupled 2×2 solve at corners
(`sw_core.F90` lines 791-863) for `grid_type < 3`.

### 4. `update_dz_d` cross-WP wiring

**Location:** Integration orchestration

Must wire WP-C `update_dz_d` sequence to use WP-B transport operators
for height advection using `hord_tm`.

## Alternative: Simplified test configuration

If the goal is to validate the CUDA infrastructure without full GEOS fidelity,
a simplified test configuration could use:

```
hord_mt = 8
hord_vt = 8
hord_tm = 8
hord_dp = 8
vtdm4 = 0.0  (disables deln_flux)
do_vort_damp = .false.
grid_type = 4  (doubly-periodic, no edge corrections)
```

This would bypass the missing PPM order 6, `deln_flux`, and contravariant edge
features. However, **this is NOT a valid GEOS production configuration** and
would not validate the intended numerical path.

## Conclusion

**FIRST-RUN CONFIGURATION UNRESOLVED**

The GEOS nonhydrostatic default configuration requires PPM order 6, which is
not implemented. The current CUDA translation (`hord=1,8` only) cannot execute
the intended GEOS configuration.

Options:
1. **Implement PPM order 6** — Required for GEOS milestone-one
2. **Use test-only configuration** — Does not validate intended GEOS path
3. **Modify GEOS to use hord=8** — Changes numerical behavior

Recommendation: **Implement PPM order 6** as the minimum additional work
required for milestone-one GEOS compatibility.
