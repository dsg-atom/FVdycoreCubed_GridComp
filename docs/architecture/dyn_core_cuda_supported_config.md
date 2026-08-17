# FV3 dyn_core CUDA supported configuration

## Overview

This document defines the supported numerical configurations for the CUDA
`dyn_core` implementation. Unsupported configurations MUST fail cleanly
before model state is mutated — no silent fallbacks or approximations are
permitted.

**Updated:** Prompt 4.2 — GEOS production first-run configuration now fully supported.

## GEOS Production First-Run Configuration

The following GEOS nonhydrostatic default configuration is **FULLY SUPPORTED**:

| Parameter | Value | Status | Implementation |
|-----------|-------|--------|----------------|
| `hord_mt` | **6** | **SUPPORTED** | `xppm_hord6/yppm_hord6` |
| `hord_vt` | **6** | **SUPPORTED** | `xppm_hord6/yppm_hord6` |
| `hord_tm` | **6** | **SUPPORTED** | `xppm_hord6/yppm_hord6`, WP-C→WP-B transport |
| `hord_dp` | **-6** | **SUPPORTED** | `xppm_hord6/yppm_hord6` with positivity constraint |
| `vtdm4` | **> 0** | **SUPPORTED** | `deln_flux_transport` enabled |
| `grid_type` | **0** | **SUPPORTED** | `contravariant_edge_corrections` for cubed-sphere |
| `hydrostatic` | **false** | **SUPPORTED** | Full nonhydrostatic path |
| `a_imp` | **1.0** | **SUPPORTED** | SIM1 vertical solver |
| `beta` | **0.0** | **SUPPORTED** | Standard NH pressure gradient |

## WP-B transport order support

### Supported transport orders

| Order | Meaning | Status | Implementation | Test coverage |
|-------|---------|--------|----------------|---------------|
| `hord=1` | First-order upwind | **SUPPORTED** | `xppm_mord1/yppm_mord1` | Full |
| `hord=6` | Sixth-order PPM with limiter | **SUPPORTED** | `xppm_hord6/yppm_hord6` | Full |
| `hord=-6` | Order-6 with positivity | **SUPPORTED** | `hord<0` branch in hord6 | Full |
| `hord=8` | Lin's fast monotone | **SUPPORTED** | `xppm_hord8` | Full |

### Unsupported transport orders

| Order | Meaning | Status | Reason |
|-------|---------|--------|--------|
| `hord=2` | Second-order van Leer | NOT SUPPORTED | Not required by GEOS first-run |
| `hord=3` | Third-order with monotonicity | NOT SUPPORTED | Not required by GEOS first-run |
| `hord=4` | Fourth-order with monotonicity | NOT SUPPORTED | Not required by GEOS first-run |
| `hord=5` | Positive-definite | NOT SUPPORTED | Not required by GEOS first-run |
| `hord=7` | Seventh-order unlimited | NOT SUPPORTED | Not required by GEOS first-run |
| `hord=9-13` | Higher-order variants | NOT SUPPORTED | Not required by GEOS first-run |

Unsupported orders MUST return `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION` before
state mutation.

## WP-B transport sequencing support

### Supported sequencing components

| Component | Config | Status | Implementation |
|-----------|--------|--------|----------------|
| `fv_tp_2d` Y-X/X-Y | All hord | **SUPPORTED** | `fv_tp_2d_hord6` with full directional passes |
| `ra_x` area correction | All | **SUPPORTED** | Between X passes |
| `ra_y` area correction | All | **SUPPORTED** | Between Y passes |
| `deln_flux` damping | `vtdm4>0` | **SUPPORTED** | `deln_flux_transport` with nord=0/1/2 |
| `copy_corners` | All | **SUPPORTED** | Delegated to WP-E corner service |
| Contravariant edges | `grid_type<3` | **SUPPORTED** | `contravariant_edge_corrections` |

## WP-B damping configuration

| Option | Meaning | Status | Implementation |
|--------|---------|--------|----------------|
| `nord=0` | No divergence damping | **SUPPORTED** | Skip damping path |
| `nord=1-3` | 2nd to 8th order damping | **SUPPORTED** | `divergence_damping_iteration` |
| `nord_v/nord_w/nord_t` | Separate orders | **SUPPORTED** | Per-order dispatch |
| `do_vort_damp` | Vorticity damping | **SUPPORTED** | Enabled when `vtdm4>0` |

### Smagorinsky damping

| Option | Status | Notes |
|--------|--------|-------|
| `smag_corner` | **APPROXIMATION** | Uses simplified corner interpolation |

**Note:** The current `smag_corner` is an approximation. For GEOS production
with adaptive Smagorinsky damping, this may need future refinement.

## WP-B tracer transport

| Option | Meaning | Status | Notes |
|--------|---------|--------|-------|
| `inline_q=.false.` | Tracers outside `d_sw` | **SUPPORTED** | Default GEOS path |
| `inline_q=.true.` | Tracers inside `d_sw` | NOT SUPPORTED | Not required by GEOS |

## WP-C vertical solver support

### Supported solvers

| Solver | a_imp Range | Status | Moist/Dry |
|--------|-------------|--------|-----------|
| SIM1 | > 0.999 | **SUPPORTED** | Both |
| SIM | 0.5 < a_imp <= 0.999 | **SUPPORTED** | Both |
| SIM3 | -0.999 < a_imp < -0.5 | **SUPPORTED** | Dry only |
| SIM3p0 | <= -0.999 | **SUPPORTED** | Dry only |
| RIM_2D | <= 0.5 | **SUPPORTED** | Both |

GEOS production uses `a_imp=1.0` which selects SIM1.

### Vertical solver options

| Option | Status | Notes |
|--------|--------|-------|
| `MOIST_CAPPA` | **SUPPORTED** | Compile-time branch |
| `USE_COND` | **SUPPORTED** | Compile-time branch |
| `use_logp` | **SUPPORTED** | Runtime branch |

### update_dz_d transport

| Component | Status | Implementation |
|-----------|--------|----------------|
| Height boundary/monotonicity | **SUPPORTED** | `height_boundary_v2_kernel` |
| Horizontal transport (hord_tm=6) | **SUPPORTED** | WP-B `dgrid_height_transport` |
| Cross-WP dispatch | **SUPPORTED** | `NumericalDispatch` table |

## WP-D pressure/diffusion support

| Option | Status | Implementation |
|--------|--------|----------------|
| `nh_p_grad` | **SUPPORTED** | NH pressure gradient |
| `split_p_grad` | **SUPPORTED** | Split beta>0 variant |
| `one_grad_p` | **SUPPORTED** | External mode beta<-0.1 |
| `a2b_ord2` | **SUPPORTED** | Second-order A-to-B |
| `a2b_ord4` | **SUPPORTED** | Fourth-order A-to-B |
| Global edge A-to-B | **SUPPORTED** | Cubed-sphere edge handling |
| `del2_cubed` | **SUPPORTED** | Hyperdiffusion |
| Diffusive heating | **SUPPORTED** | Temperature adjustment |

## WP-A C-grid predictor support

| Component | Status | Implementation |
|-----------|--------|----------------|
| `d2a2c_vect` | **SUPPORTED** | D-to-A/C conversion |
| `c_sw` | **SUPPORTED** | C shallow-water |
| `divergence_corner` | **SUPPORTED** | Corner divergence |
| `update_dz_c` | **SUPPORTED** | C-grid height |
| `p_grad_c` | **SUPPORTED** | C pressure gradient |
| `Riem_Solver_C` | **SUPPORTED** | Via WP-C dispatch |

## Global configuration options

| Option | Value | Status | Notes |
|--------|-------|--------|-------|
| `hydrostatic` | `.false.` | **SUPPORTED** | Required for NH |
| `hydrostatic` | `.true.` | NOT SUPPORTED | Excluded from scope |
| `nested` | `.false.` | **SUPPORTED** | Global grid only |
| `nested` | `.true.` | NOT SUPPORTED | Excluded from scope |
| `grid_type` | 0 | **SUPPORTED** | Cubed-sphere |
| `grid_type` | >= 3 | PARTIAL | Limited testing |
| ROT3/do_f3d | — | NOT SUPPORTED | Optional extension |
| SKEB | — | NOT SUPPORTED | Optional extension |

## Configuration gates

The integrated runtime implements explicit gates that reject unsupported
configurations before state mutation:

### hord validation

```cpp
inline bool is_hord_supported(int32_t hord) noexcept {
  const int32_t mord = hord > 0 ? hord : -hord;
  return mord == 1 || mord == 6 || mord == 8;
}
```

Unsupported `hord` values return `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION`.

### Global guards

| Check | Gate location | Error code |
|-------|---------------|------------|
| `hord_*` not in {1,6,8,-6} | Transport entry | `UNSUPPORTED_CONFIGURATION` |
| `inline_q=.true.` | D-grid entry | `UNSUPPORTED_CONFIGURATION` |
| `hydrostatic=.true.` | Context creation | `UNSUPPORTED_CONFIGURATION` |
| `nested=.true.` | Context creation | `UNSUPPORTED_CONFIGURATION` |

## Summary

### Fully supported (GEOS production first-run)

- Nonhydrostatic mode (`hydrostatic=.false.`)
- Global cubed-sphere grid (`grid_type=0`, `nested=.false.`)
- Transport orders: 1, 6, -6, 8
- All vertical solvers (RIM_2D, SIM, SIM1, SIM3, SIM3p0)
- Divergence damping orders 0-3
- Vorticity damping (`vtdm4>0`)
- Transport flux damping (`deln_flux`)
- Contravariant edge corrections
- Complete Y-X / X-Y transport sequencing
- Cross-WP height transport (WP-C → WP-B)
- Persistent vertical workspace

### Not supported (excluded from scope)

- Hydrostatic mode
- Nested grids
- Transport orders 2-5, 7, 9-13
- Inline tracer transport (`inline_q=.true.`)
- ROT3/do_f3d (3D Coriolis)
- SKEB (stochastic KE backscatter)
- True `smag_corner` (approximation used)

### Configuration for GEOS production first-run

```fortran
hydrostatic = .false.
nested = .false.
hord_mt = 6
hord_vt = 6
hord_tm = 6
hord_dp = -6
vtdm4 = 0.01  ! or other positive value
grid_type = 0
a_imp = 1.0
beta = 0.0
```

This configuration is **FULLY SUPPORTED** by the integrated CUDA implementation.
