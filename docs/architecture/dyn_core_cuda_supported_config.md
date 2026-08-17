# FV3 dyn_core CUDA supported configuration

## Overview

This document defines the supported numerical configurations for the initial
CUDA `dyn_core` implementation. Unsupported configurations MUST fail cleanly
before model state is mutated — no silent fallbacks or approximations are
permitted.

## WP-B transport order support

The D-grid dynamics package (WP-B) requires finite-volume transport operations
with configurable reconstruction orders. The current implementation status:

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `hord_dp=1` | First-order upwind mass transport | **YES** | `wp_b_kernels.hpp:xppm_mord1/yppm_mord1` | Basic | No |
| `hord_dp=2` | Second-order van Leer | NO | — | — | No |
| `hord_dp=3` | Third-order with monotonicity | NO | — | — | No |
| `hord_dp=4` | Fourth-order with monotonicity | NO | — | — | No |
| `hord_dp=5` | Positive-definite | NO | — | — | No |
| `hord_dp=6` | Sixth-order unlimited | NO | — | — | No |
| `hord_dp=7` | Seventh-order unlimited | NO | — | — | No |
| `hord_dp=8` | Lin's fast monotone | **YES** | `wp_b_kernels.hpp:xppm_hord8` | Basic | No |
| `hord_dp=9-13` | Higher-order variants | NO | — | — | No |

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `hord_tr` | Tracer transport order | See `hord_dp` | Same as `hord_dp` | Same | No |
| `hord_mt` | Momentum transport order | See `hord_dp` | Same as `hord_dp` | Same | No |
| `hord_vt` | Vorticity transport order | See `hord_dp` | Same as `hord_dp` | Same | No |
| `hord_tm` | Thermal transport order | See `hord_dp` | Same as `hord_dp` | Same | No |

### Unsupported transport orders

Orders 2-7 and 9-13 require PPM reconstruction variants not yet translated.
These orders MUST fail with `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION` before
state mutation.

## WP-B damping configuration

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `nord=0` | No divergence damping | **YES** | `wp_b_kernels.hpp` | Implicit | No |
| `nord=1-3` | 2nd to 8th order damping | **YES** | `wp_b_kernels.hpp:divergence_damping_iteration` | Basic | No |
| `nord_v/nord_w/nord_t` | Separate vorticity/vertical/thermal damping orders | **YES** | `wp_b_kernels.hpp` | Basic | No |

### Smagorinsky damping

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `smag_corner` | True A-to-B vorticity interpolation | **APPROXIMATION** | `wp_b_kernels.hpp:smag_corner_approx` | Basic | See note |

**Note:** The current `smag_corner` is an approximation that does not use the
full `a2b_ord4` interpolation from A-grid vorticity to B-grid corners. This
approximation affects adaptive divergence damping coefficient computation.
**This approximation is NOT suitable for production validation** unless the
GEOS configuration disables Smagorinsky-based adaptive damping.

## WP-B transport sequencing

| Component | Fortran meaning | Supported | Implementation file | Notes |
|-----------|-----------------|-----------|---------------------|-------|
| `fv_tp_2d` Y-X/X-Y | Complete directional transport | **PARTIAL** | `wp_b_kernels.hpp:fv_tp_2d_core` | Simplified; full `ra_x/ra_y` corrections missing |
| `deln_flux` | Damped flux divergence | NO | — | Required for full transport |
| `copy_corners` | Inter-phase corner copies | NO | — | Requires corner service orchestration |
| Contravariant edge | Panel edge `ut/vt` solve | NO | — | Required for cubed-sphere |

## WP-B tracer transport

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `inline_q=.false.` | Tracers outside `d_sw` | **YES** | — (no action in WP-B) | — | — |
| `inline_q=.true.` | Tracers inside `d_sw` | NO | — | — | No |

## WP-B PPM limiters

| Component | Fortran meaning | Supported | Implementation file | Notes |
|-----------|-----------------|-----------|---------------------|-------|
| `pert_ppm` positive-definite | Positive tracer constraint | NO | — | Required for tracers |
| `pert_ppm` standard | Standard PPM constraint | NO | — | Required for scalars |

## WP-C vertical solver support

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `a_imp <= 0.5` | RIM_2D explicit solver | **YES** | `fv3_vertical_column.hpp:RIM_2D` | Full | No |
| `a_imp > 0.999` | SIM1 semi-implicit | **YES** | `fv3_vertical_column.hpp:SIM1_solver` | Full | No |
| `-0.999 < a_imp < -0.5` | SIM3p0 variant | **YES** | `fv3_vertical_column.hpp:SIM3p0_solver` | Full | No |
| `-0.5 <= a_imp < 0.5` | SIM variant | **YES** | `fv3_vertical_column.hpp:SIM_solver` | Full | No |
| `0.5 < a_imp <= 0.999` | SIM3 variant | **YES** | `fv3_vertical_column.hpp:SIM3_solver` | Full | No |

### Vertical solver variants

| Option | Fortran meaning | Supported | Implementation file | Notes |
|--------|-----------------|-----------|---------------------|-------|
| `MOIST_CAPPA` | Moist thermodynamics | **YES** | Column primitives | Compile-time branch |
| `USE_COND` | Condensate loading | **YES** | Column primitives | Compile-time branch |
| `use_logp` | Logarithmic pressure | **YES** | Column primitives | — |

### update_dz_d transport

| Component | Fortran meaning | Supported | Implementation file | Notes |
|-----------|-----------------|-----------|---------------------|-------|
| Height boundary/monotonicity | D-grid height recurrence | **YES** | `fv3_vertical_kernels.cu` | — |
| Horizontal transport | Height FV transport | NO | Requires WP-B operator | Cross-WP boundary |

## WP-D pressure/diffusion support

| Option | Fortran meaning | Supported | Implementation file | Test coverage | Fallback |
|--------|-----------------|-----------|---------------------|---------------|----------|
| `nh_p_grad` | NH pressure gradient | **YES** | `wp_d_operators.hpp` | Full | No |
| `split_p_grad` | Split beta>0 variant | **YES** | `wp_d_operators.hpp` | Full | No |
| `one_grad_p` | External mode beta<-0.1 | **YES** | `wp_d_operators.hpp` | Full | No |
| `a2b_ord2` | Second-order A-to-B | **YES** | `wp_d_operators.hpp` | Full | No |
| `a2b_ord4` | Fourth-order A-to-B | **YES** | `wp_d_operators.hpp` | Full | No |
| Global edge A-to-B | Cubed-sphere edge | **YES** | `wp_d_operators.hpp` | Full | No |
| `del2_cubed` | Hyperdiffusion | **YES** | `wp_d_operators.hpp` | Full | No |
| Diffusive heating | Temperature adjustment | **YES** | `wp_d_operators.hpp` | Full | No |

### Pressure variants

| Option | Fortran meaning | Supported | Implementation file | Notes |
|--------|-----------------|-----------|---------------------|-------|
| `beta=0` | Standard NH pressure | **YES** | Default | — |
| `beta>0` | Split pressure gradient | **YES** | `split_p_grad` | — |
| `beta<-0.1` | External mode | **YES** | `one_grad_p` | — |
| `d_ext>0` | External mode depth | **YES** | Configuration | — |

## WP-A C-grid predictor support

| Component | Fortran meaning | Supported | Implementation file | Notes |
|-----------|-----------------|-----------|---------------------|-------|
| `d2a2c_vect` | D-to-A/C conversion | **YES** | `fv3_cgrid_d2a2c.cu` | — |
| `c_sw` | C shallow-water | **YES** | `fv3_cgrid_csw.cu` | — |
| `divergence_corner` | Corner divergence | **YES** | `fv3_cgrid_divergence.cu` | — |
| `update_dz_c` | C-grid height | **YES** | `fv3_cgrid_dz.cu` | — |
| `p_grad_c` | C pressure gradient | **YES** | `fv3_cgrid.cu` | — |
| Predictor orchestration | Full C-grid dispatch | **YES** | `fv3_cgrid_predictor.cu` | — |

### Unsupported C-grid options

| Option | Fortran meaning | Supported | Notes |
|--------|-----------------|-----------|-------|
| Nested grids | `divergence_corner_nest` | NO | Milestone-one excludes |
| Hydrostatic | SW_DYNAMICS branch | NO | NH only |
| `grid_type >= 3` | Regional/other | **PARTIAL** | Limited testing |

## Global configuration options

| Option | Fortran meaning | Supported | Notes |
|--------|-----------------|-----------|-------|
| `hydrostatic=.false.` | Nonhydrostatic | **YES** | Required |
| `hydrostatic=.true.` | Hydrostatic | NO | Excluded |
| `nested=.false.` | Global grid | **YES** | Required |
| `nested=.true.` | Nested grid | NO | Excluded |
| `beta=0` | Standard pressure | **YES** | Default |
| `use_logp=.false.` | Power pressure | **YES** | Default |
| ROT3/do_f3d | 3D Coriolis | NO | Optional extension |
| SKEB | Stochastic KE backscatter | NO | Optional extension |
| Rayleigh friction | `Ray_fast` | NO | Optional extension |

---

## FIRST RUNNABLE CUDA CONFIGURATION

Based on the integrated implementation, the narrowest legitimate FV3
configuration for which every required numerical stage is actually
implemented:

### Required settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| `hydrostatic` | `.false.` | Only NH implemented |
| `nested` | `.false.` | Only global grid |
| `hord_dp` | 1 or 8 | Only orders 1 and 8 translated |
| `hord_tr` | 1 or 8 | Only orders 1 and 8 translated |
| `hord_mt` | 1 or 8 | Only orders 1 and 8 translated |
| `hord_vt` | 1 or 8 | Only orders 1 and 8 translated |
| `hord_tm` | 1 or 8 | Only orders 1 and 8 translated |
| `nord` | 0-3 | All damping orders supported |
| `inline_q` | `.false.` | Inline tracers not implemented |
| `beta` | 0 | Standard pressure gradient (split/one_grad also available) |
| `use_logp` | `.false.` | Default |
| `a_imp` | any | All vertical solvers implemented |

### NOT supported in first runnable configuration

- `hord_*` values 2-7 or 9-13
- `inline_q=.true.`
- Nested grids
- Hydrostatic mode
- True `smag_corner` (approximation used)
- Full `fv_tp_2d` directional sequencing
- Contravariant edge corrections for panel boundaries
- `pert_ppm` limiters

### GEOS milestone-one compatibility

**WARNING:** The intended GEOS milestone-one configuration may use transport
orders other than 1 or 8. If GEOS requires `hord_dp=5` (positive-definite) or
other orders, the first runnable configuration cannot yet represent it.

The following must be verified against the target GEOS configuration:

1. Transport order compatibility
2. Smagorinsky damping requirements
3. Inline tracer usage

If the intended GEOS configuration cannot be represented by the current
implementation, it must be explicitly documented and the missing PPM orders
must be prioritized for completion.

---

## Unsupported configuration gates

The integrated runtime MUST implement explicit gates that reject unsupported
configurations before state mutation. The following checks are required:

1. `hord_*` values must be 1 or 8; others return `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION`
2. `inline_q=.true.` returns `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION`
3. `hydrostatic=.true.` returns `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION`
4. `nested=.true.` returns `FV3_GPU_ERR_UNSUPPORTED_CONFIGURATION`

These gates must be enforced at context creation or step entry, not during
kernel execution.
