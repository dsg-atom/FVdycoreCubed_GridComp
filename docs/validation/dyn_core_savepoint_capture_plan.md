# dyn_core savepoint capture plan

## Overview

Numerical validation of the CUDA `dyn_core` subsystem requires authoritative
Fortran savepoints captured at specific execution boundaries. This document
defines the capture locations, required fields, and metadata per the Gate A
validation contract.

## Capture protocol

Each savepoint record must contain:

1. Schema version
2. Source commit SHA (Fortran)
3. Build flags and precision
4. Namelist hash
5. Grid/rank/bounds
6. Model/acoustic/vertical-substep indices
7. Field descriptors with compute and halo extents
8. Array data for compute and requested halo regions

Reference and CUDA captures must start from identical serialized input bytes.

## Required savepoints

### DC00_STEP_IN — Subsystem entry

**Location:** Entry to `dyn_core`, before any computation

**Purpose:** Common initial condition; layout/upload verification

**Required fields:**

| Field | Stagger | Precision | Include halo |
|-------|---------|-----------|--------------|
| `u` | D-u | FV real | Yes |
| `v` | D-v | FV real | Yes |
| `w` | A | FV real | Yes |
| `delz` | A | FV real | Yes |
| `pt` | A | FV real | Yes |
| `delp` | A | FV real | Yes |
| `q_con` | A | FV real | Yes (if enabled) |
| `cappa` | A | FV real | Yes (if enabled) |
| `pe` | A/I (i,k,j) | FV real | Yes |
| `pk` | A/I | FV real | Yes |
| `peln` | A/I (i,k,j) | FV real | Yes |
| `pkz` | A | FV real | No |
| `phis` | A | FV real | Yes |
| `gz` | A/I | FV real | Yes |
| `ua` | A | FV real | Yes |
| `va` | A | FV real | Yes |

**Metadata:**
- Grid configuration hash
- `n_split`, `k_split`, `bdt`
- Enabled branches/options

### DC10_CSW_OUT — C shallow-water output

**Location:** After `c_sw`, before C vertical solve

**Purpose:** First divergence detection in C-grid predictor

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `delpc` | A | No |
| `ptc` | A | No |
| `ut` | A | No |
| `vt` | A | No |
| `divgd` | B | Yes (corners) |
| `uc` | C-u | Yes |
| `vc` | C-v | Yes |
| `ua` | A | No |
| `va` | A | No |
| `omga` | A | No |

### DC20_CVERT_OUT — C vertical solver output

**Location:** After `Riem_Solver_C`

**Purpose:** C-grid height and vertical solver validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `gz` | A/I | Yes |
| `pkc` | A/I | Yes |
| `ws3` | A (2D) | No |
| `omga` | A | No |
| `delpc` | A | No |
| `ptc` | A | No |

### DC30_PGRAD_C_OUT — C pressure gradient output

**Location:** After `p_grad_c`

**Purpose:** C pressure gradient and C-halo validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `uc` | C-u | Yes |
| `vc` | C-v | Yes |

### DC40_DSW_OUT — D-grid dynamics output

**Location:** After `d_sw` (complete D-grid step)

**Purpose:** D-grid dynamics, transport, damping validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `u` | D-u | Yes |
| `v` | D-v | Yes |
| `w` | A | Yes |
| `delp` | A | Yes |
| `pt` | A | Yes |
| `q_con` | A | Yes (if enabled) |
| `crx` | x-face | No |
| `cry` | y-face | No |
| `xfx` | x-face | No |
| `yfx` | y-face | No |
| `mfx` | x-face | No |
| `mfy` | y-face | No |
| `cx` | x-face | No |
| `cy` | y-face | No |
| `heat_source` | A | No |
| `diss_est` | A | No (if enabled) |

### DC50_DHEIGHT_OUT — D-grid height output

**Location:** After `update_dz_d`

**Purpose:** D-grid height update validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `zh` | A/I | Yes |
| `delz` | A | No |
| `ws` | A (2D) | No |

### DC60_RIEM3_OUT — Riemann solver output

**Location:** After `Riem_Solver3`

**Purpose:** Vertical solver and pressure reconstruction validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `w` | A | No |
| `delz` | A | No |
| `zh` | A/I | Yes |
| `pe` | A/I (i,k,j) | No |
| `peln` | A/I (i,k,j) | No |
| `pk` | A/I | No |
| `pkc` | A/I | Yes (two-ring) |
| `pk3` | A/I | Yes (two-ring) |

### DC70_PGRAD_D_OUT — D pressure gradient output

**Location:** After `nh_p_grad`/`split_p_grad`/`one_grad_p`

**Purpose:** Nonhydrostatic pressure gradient validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `u` | D-u | Yes |
| `v` | D-v | Yes |
| `du` | D-u | No (if beta path) |
| `dv` | D-v | No (if beta path) |

### DC80_ACOUSTIC_END — Acoustic substep end

**Location:** End of each `it` iteration in acoustic loop

**Purpose:** Per-substep recurrence and halo state validation

**Required fields:**

All prognostic state and accumulators at substep boundary.

### DC90_DYN_CORE_OUT — Subsystem output

**Location:** Exit from `dyn_core`, after post-loop heating

**Purpose:** Final subsystem contract validation

**Required fields:**

| Field | Stagger | Include halo |
|-------|---------|--------------|
| `u` | D-u | Yes |
| `v` | D-v | Yes |
| `w` | A | Yes |
| `delz` | A | Yes |
| `pt` | A | Yes |
| `delp` | A | Yes |
| `pe` | A/I (i,k,j) | Yes |
| `pk` | A/I | Yes |
| `peln` | A/I (i,k,j) | Yes |
| `pkz` | A | No |
| `mfx` | x-face | No |
| `mfy` | y-face | No |
| `cx` | x-face | No |
| `cy` | y-face | No |

## Capture implementation notes

### Fortran instrumentation

The authoritative Fortran should NOT be modified for production. Savepoint
capture should use one of:

1. **Test-only instrumentation build** — Conditional compilation with
   `ENABLE_SAVEPOINTS` that is never used in production
2. **External capture wrapper** — Interposition layer that captures state at
   boundaries without modifying numerical code
3. **Existing FV3 serialization** — If the codebase has existing diagnostic
   serialization, reuse it

### Serialization format

Each savepoint should produce:

1. Binary data file with field arrays
2. JSON/YAML metadata file with:
   - Field names and shapes
   - Lower bounds and strides
   - Precision
   - Capture location
   - Model state (step, substep indices)
   - Configuration hash

### First capture configuration

The first capture should use the "first runnable configuration":

- `hydrostatic=.false.`
- `nested=.false.`
- `hord_*=1` or `8`
- `inline_q=.false.`
- `beta=0`

Captures at other configurations should be deferred until those configurations
are implemented.

## Status

| Savepoint | Fields defined | Instrumentation | Captured |
|-----------|----------------|-----------------|----------|
| DC00_STEP_IN | YES | NO | NO |
| DC10_CSW_OUT | YES | NO | NO |
| DC20_CVERT_OUT | YES | NO | NO |
| DC30_PGRAD_C_OUT | YES | NO | NO |
| DC40_DSW_OUT | YES | NO | NO |
| DC50_DHEIGHT_OUT | YES | NO | NO |
| DC60_RIEM3_OUT | YES | NO | NO |
| DC70_PGRAD_D_OUT | YES | NO | NO |
| DC80_ACOUSTIC_END | YES | NO | NO |
| DC90_DYN_CORE_OUT | YES | NO | NO |

## Next steps

1. Identify existing serialization infrastructure in GFDL FV3
2. Determine whether test-only instrumentation is acceptable
3. Create capture driver for first runnable configuration
4. Execute capture on reference system
5. Verify round-trip reproducibility
