# FV3 dyn_core CUDA integrated execution sequence

## Overview

This document defines the integrated execution sequence for the CUDA `dyn_core`
subsystem, the cross-WP boundaries, and the implementation status of each stage.

## Authoritative execution sequence

Per the Gate A architecture documents, the acoustic substep ordering is:

```
[Per n_split acoustic substep]
  H_DWIND_IN + H_W_IN + H_GZ_FIRST (input halos)
    → WP-A C-grid predictor
      → d2a2c_vect (D-to-A/C conversion)
      → c_sw (C shallow-water dynamics)
      → divergence_corner
      → H_DIVGD (corner divergence halo)
    → WP-A C-grid vertical
      → update_dz_c (C-grid height update)
      → Riem_Solver_C → WP-C column solver
    → WP-A C pressure gradient
      → p_grad_c
      → H_CWIND (C-wind halo)
    → WP-B D-grid dynamics
      → d_sw (full D-grid dynamics, transport, damping)
      → H_DSCALAR (scalar halo)
    → WP-C D-grid height/vertical
      → update_dz_d (horizontal transport via WP-B)
      → Riem_Solver3
      → H_ZH + H_PKC2 (post-solver halos)
    → WP-D pressure operators
      → pe_halo, pk3_halo (local reconstruction)
      → a2b_ord4/a2b_ord2 (A-to-B conversion)
      → nh_p_grad / split_p_grad / one_grad_p
      → H_DWIND_NEXT (D-wind halo for next substep)
[End substep loop]
  H_HEAT (heat source halo)
  → WP-D post-loop heating
    → del2_cubed (hyperdiffusion)
    → temperature adjustment
```

## Integrated execution coverage

### Input halo readiness

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `H_DWIND_IN` | WP-E | `DependencyState::require` | D `u/v` | — | D-vector `ng` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_W_IN` | WP-E | `DependencyState::require` | `w` | — | Scalar `ng` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_GZ_FIRST` | WP-E | `DependencyState::require` | `gz` (first split) | — | A/interface | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

### WP-A C-grid predictor

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `d2a2c_vect` | WP-A | `fv3_cgrid_d2a2c.cu` | D `u/v`, metrics | `ua/va`, `utmp/vtmp` | Input halos ready | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `c_sw` | WP-A | `fv3_cgrid_csw.cu` | `ua/va`, `delp/pt/w`, metrics | `delpc/ptc`, `ut/vt`, `uc/vc`, `omga` | Corner fills | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `divergence_corner` | WP-A | `fv3_cgrid_divergence.cu` | `uc/vc` | `divgd` | Corner ownership | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_DIVGD` | WP-E | `DependencyState::mark_stale` | `divgd` | — | B/corner | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

### WP-A/WP-C C-grid vertical

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `update_dz_c` | WP-A | `fv3_cgrid_dz.cu` | `ut/vt`, `gz`, `dp_ref` | `gz`, `ws3` | `gz` halo | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `Riem_Solver_C` | WP-A→WP-C | `NumericalDispatch::cgrid_vertical_solver` | `delpc/ptc`, `gz` | `pkc`, `gz`, `omga` | Column-local | PARTIAL | PASS | NOT RUN | NOT RUN | NOT RUN |

**Note:** `Riem_Solver_C` adapter in WP-A invokes WP-C column solver primitives.
The primitives are COMPLETE; the dispatch wiring is PARTIAL.

### WP-A C pressure gradient

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `p_grad_c` | WP-A | `fv3_cgrid.cu` | `delpc`, `pkc`, `gz` | `uc/vc` | Radius-1 | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_CWIND` | WP-E | `DependencyState::mark_stale` | `uc/vc` | — | C-vector | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

### WP-B D-grid dynamics

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| Flux prep | WP-B | `wp_b_kernels.hpp` | `uc/vc`, metrics | `ut/vt`, `crx/cry/xfx/yfx` | `H_CWIND` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| Mass transport | WP-B | `wp_b_kernels.hpp` | `delp`, faces, metrics | `delp`, `fx/fy` | Order-dependent | PARTIAL | PASS | NOT RUN | NOT RUN | NOT RUN |
| Scalar transport | WP-B | `wp_b_kernels.hpp` | `pt/w/q_con`, mass flux | `pt/w/q_con` | Order-dependent | PARTIAL | PASS | NOT RUN | NOT RUN | NOT RUN |
| Capacitor accumulation | WP-B | `wp_b_kernels.hpp` | Local Courants, flux | `mfx/mfy/cx/cy` | None | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| KE/momentum | WP-B | `wp_b_kernels.hpp` | `u/v/uc/vc`, metrics | `ke`, transported momentum | Corner corrections | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| Vorticity | WP-B | `wp_b_kernels.hpp` | D winds, metrics | Vorticity, `z_rat` | `H_DIVGD` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| Divergence damping | WP-B | `wp_b_kernels.hpp` | `divgd`, winds, damping config | `u/v`, damping fluxes | Corner fill per pass | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| Heat accounting | WP-B | `wp_b_kernels.hpp` | Damping increments, `delp` | `heat_source`, `diss_est` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| D-wind update | WP-B | `wp_b_kernels.hpp` | Transport, KE, damping | `u/v` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `mix_dp` | WP-B | `wp_b_kernels.hpp` | `delp/pt/w`, `ak/bk` | Repaired state | Column-local | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_DSCALAR` | WP-E | `DependencyState::mark_stale` | `delp/pt/q_con` | — | Scalar group | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

**PARTIAL notes:**
- Mass/scalar transport: only `hord=1,8` implemented
- Full `fv_tp_2d` Y-X/X-Y sequencing with `ra_x/ra_y` corrections: UNIMPLEMENTED
- `deln_flux` damping path: UNIMPLEMENTED
- Contravariant edge corrections: UNIMPLEMENTED
- `pert_ppm` limiters: UNIMPLEMENTED
- Inline tracer transport: UNIMPLEMENTED

### WP-C D-grid height/vertical

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `update_dz_d` (boundary/monotonicity) | WP-C | `fv3_vertical_kernels.cu` | `zh`, `delz`, `zs` | `zh`, `delz`, `ws` | `H_DSCALAR` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `update_dz_d` (horizontal transport) | WP-C→WP-B | `NumericalDispatch::dgrid_height_transport` | Transport faces | Height fluxes | Transport arena | UNIMPLEMENTED | — | — | — | — |
| `Riem_Solver3` precompute | WP-C | `fv3_vertical_column.hpp` | `delp/pt/q_con/cappa` | Column temporaries | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `Riem_Solver3` solver dispatch | WP-C | `fv3_vertical_column.hpp` | Column temporaries | `w`, `pe`, `peln`, `pk`, `delz` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `Riem_Solver3` finalize | WP-C | `fv3_vertical_column.hpp` | Solver outputs | `zh`, `gz`, `pkc`, `pk3` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_ZH` | WP-E | `DependencyState::mark_stale` | `zh` | — | A/interface | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_PKC2` | WP-E | `DependencyState::mark_stale` | `pkc` | — | Two-cell | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

**Cross-WP boundary:** `update_dz_d` horizontal transport requires WP-B transport
operators. The call path is:

```
WP-C update_dz_d sequence
  → WP-C boundary condition / initial setup
  → NumericalDispatch::dgrid_height_transport (WP-B operator)
  → WP-C monotonicity / finalization
```

This cross-WP interface is defined by `NumericalDispatch::dgrid_height_transport`
in the WP-E v3.1 contract but is UNIMPLEMENTED because the WP-B transport
operator for height is not yet wired.

### WP-D pressure operators

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `pe_halo` | WP-D | `wp_d_operators.hpp` | `delp`, `pe` | `pe` (one ring) | Local recurrence | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `pk3_halo` / `pln_halo` | WP-D | `wp_d_operators.hpp` | `pe/delp` | `pk3`/`peln` (two rings) | Local recurrence | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `gz = grav*zh` | WP-D | `wp_d_operators.hpp` | `zh` | `gz` | `H_ZH` | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `a2b_ord4` / `a2b_ord2` | WP-D | `wp_d_operators.hpp` | A-grid pressure/height | B-grid values | `H_PKC2`, global edges | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `nh_p_grad` | WP-D | `wp_d_operators.hpp` | B-grid pressure, `gz` | `u/v` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `split_p_grad` | WP-D | `wp_d_operators.hpp` | Same + `du/dv` | `u/v`, `du/dv` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `one_grad_p` | WP-D | `wp_d_operators.hpp` | External mode | `u/v` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `H_DWIND_NEXT` | WP-E | `DependencyState::mark_stale` | `u/v` | — | D-vector | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

### WP-D post-loop heating

| Fortran stage | Owning WP | CUDA entry point | Input fields | Output fields | Halo requirement | Status | CPU compile | CUDA compile | GPU test | Savepoint |
|---------------|-----------|------------------|--------------|---------------|------------------|--------|-------------|--------------|----------|-----------|
| `H_HEAT` | WP-E | `DependencyState::require` | `heat_source` | — | Damping width | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `del2_cubed` (corner avg) | WP-D | `wp_d_operators.hpp` | `heat_source` | Corner-averaged | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| `del2_cubed` (passes) | WP-D | `wp_d_operators.hpp` | `heat_source` | Filtered `heat_source` | x-copy, flux, y-copy | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |
| Temperature adjustment | WP-D | `wp_d_operators.hpp` | Filtered heat, `delp/delz/cappa` | `pt`, `pkz` | — | COMPLETE | PASS | NOT RUN | NOT RUN | NOT RUN |

## Implementation status summary

### By work package

| WP | Stages | COMPLETE | PARTIAL | UNIMPLEMENTED |
|----|--------|----------|---------|---------------|
| WP-E | Halos/dependencies | 10 | 0 | 0 |
| WP-A | C-grid predictor | 6 | 1 | 0 |
| WP-B | D-grid dynamics | 8 | 3 | 4 |
| WP-C | Vertical | 5 | 0 | 1 |
| WP-D | Pressure/heating | 10 | 0 | 0 |
| **Total** | | **39** | **4** | **5** |

### PARTIAL stages (formulas translated, orchestration incomplete)

1. `Riem_Solver_C` — WP-A adapter to WP-C solver exists but dispatch not wired
2. Mass transport — only `hord=1,8` implemented
3. Scalar transport — only `hord=1,8` implemented
4. `fv_tp_2d` sequencing — simplified core only

### UNIMPLEMENTED stages

1. `update_dz_d` horizontal transport — requires WP-B/WP-C cross-WP wiring
2. Full `fv_tp_2d` Y-X/X-Y with `ra_x/ra_y` — WP-B
3. `deln_flux` damping path — WP-B
4. Contravariant edge corrections — WP-B
5. Inline tracer transport — WP-B

## Cross-WP boundaries

### WP-A → WP-C: C-grid vertical solver

**Interface:** `NumericalDispatch::cgrid_vertical_solver`

**Call path:**
```
WP-A Riem_Solver_C adapter
  → validate ExecutionContext
  → prepare column inputs from C-grid state
  → invoke WP-C column solver primitive
  → extract outputs to C-grid state
```

**Status:** Interface defined, primitives complete, adapter wiring PARTIAL.

### WP-C → WP-B: update_dz_d height transport

**Interface:** `NumericalDispatch::dgrid_height_transport`

**Call path:**
```
WP-C update_dz_d sequence
  → prepare transport inputs
  → invoke WP-B height transport operator
  → continue with WP-C monotonicity enforcement
```

**Status:** Interface defined, WP-B operator UNIMPLEMENTED for height transport.

**Resolution required:** WP-B must expose a height-specific transport service
that accepts the cross-WP interface defined by WP-E v3.1. This service uses the
same `fv_tp_2d` infrastructure as scalar transport but operates on height arrays.
