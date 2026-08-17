# GEOS CUDA GPU Residency Plan

## Overview

This document describes the GPU memory residency strategy for the GEOS/FV3
CUDA dyn_core integration. The goal is to minimize host-device data transfers
while maintaining correctness at physics and I/O boundaries.

## Residency model

### GPU-resident (persistent)

These fields remain on GPU from initialization to finalization:

| Category | Fields | Transfer pattern |
|----------|--------|------------------|
| Prognostic state | u, v, w, delz, pt, delp, q, q_con | H2D once, D2H at sync |
| Thermodynamic | pe, pk, peln, pkz | H2D once, D2H at sync |
| Diagnostic | omga, ua, va, uc, vc | H2D once, D2H at sync |
| Mass flux | mfx, mfy, cx, cy | H2D once, D2H at sync |
| Dissipation | diss_est | H2D once, D2H at sync |
| Immutable grid | All gridstruct fields | H2D once, never sync |
| Hybrid coordinates | ak, bk | H2D once, never sync |

### GPU-allocated (temporary)

These fields are allocated by WP-E workspace and live only during kernel execution:

| Field | Purpose | Lifetime |
|-------|---------|----------|
| delpc, ptc | C-grid intermediates | Acoustic substep |
| ut, vt | Transport intermediates | D-grid dynamics |
| divgd | Divergence on B-grid | Acoustic substep |
| crx, xfx, cry, yfx | Courant/flux | Transport |
| gz, zh, pkc, pk3 | Vertical intermediates | Vertical solver |
| du, dv | Wind tendencies | Pressure gradient |
| heat_source | Dissipative heating | Heat accounting |
| z_rat, divg2 | Diffusion | Divergence damping |

### Host-only (never transferred)

These fields are not needed by GPU kernels:

| Field | Reason |
|-------|--------|
| ps | Derived from pe(:,:,1) |
| varflt | Variance filter (not in dyn_core) |
| ze0 | Hybrid-z reference (not in production path) |
| tracer names | String data |

## Initial H2D transfer

At `fv3_gpu_interface_init()`:

```
Host                         GPU
-----                       -----
u[isd:ied,jsd:jed+1,npz] → u_device
v[isd:ied+1,jsd:jed,npz] → v_device
w[isd:ied,jsd:jed,npz]   → w_device
...
area[isd:ied,jsd:jed]    → area_device
...
ak[1:npz+1]              → ak_device
bk[1:npz+1]              → bk_device
```

This is a one-time bulk transfer of ~73 fields.

## During dyn_core execution

**No H2D or D2H transfers** within the acoustic loop.

All computation happens on GPU:
1. C-grid predictor (WP-A)
2. D-grid dynamics (WP-B)
3. Vertical solver (WP-C)
4. Pressure/diffusion (WP-D)

Intermediate results stay in GPU workspace.

## Synchronization boundaries

### Boundary 1: Halo exchange (Stage 1)

**Trigger:** MPI communication required between tiles/ranks

**Pattern:**
```
GPU                    Host                  MPI
---                    ----                  ---
u_boundary → D2H → u_host_boundary → sendrecv → u_host_halo → H2D → u_halo
```

**Affected fields:** u, v, delp, pt, w, delz (boundary regions only)

**Implementation (Stage 1):**
- Use `fv3_gpu_sync_to_host()` with boundary-only mask
- Call `mpp_update_domains()` on host
- Use `fv3_gpu_sync_from_host()` with halo-only mask

**Future (Stage 2+):**
- GPU-direct MPI or NCCL for direct GPU-to-GPU transfer
- No host involvement

### Boundary 2: Physics coupling

**Trigger:** End of dynamics, before physics parameterizations

**Pattern:**
```
GPU                    Host                Physics
---                    ----                -------
all_prognostic → D2H → Fortran arrays → physics calls → H2D → updated
```

**Affected fields:** All prognostic (u, v, w, pt, delp, q, etc.)

**Implementation:**
- `fv3_gpu_interface_sync(.true., rc)` for full sync
- Physics runs on host (unchanged Fortran code)
- `fv3_gpu_sync_from_host()` after physics completes

### Boundary 3: Diagnostics/I/O

**Trigger:** User-requested output (NetCDF, History, etc.)

**Pattern:**
```
GPU                    Host                I/O
---                    ----                ---
selected_fields → D2H → Fortran arrays → MAPL/ESMF I/O
```

**Affected fields:** User-selected subset

**Implementation:**
- `fv3_gpu_interface_sync()` with specific field mask
- I/O runs on host arrays as usual

### Boundary 4: Savepoint validation (debug)

**Trigger:** Debug mode numerical validation

**Pattern:**
```
GPU                    Host                Validation
---                    ----                ----------
all_fields → D2H → Fortran arrays → compare with reference
```

**Affected fields:** All registered fields

**Implementation:**
- Full `fv3_gpu_interface_sync(.true., rc)`
- Compare with Fortran reference values

### Boundary 5: Finalization

**Trigger:** Model shutdown

**Pattern:**
```
GPU                    Host
---                    ----
final_state → D2H → Fortran arrays (for restart file)
deallocate GPU memory
```

**Affected fields:** All prognostic (for potential restart)

## Memory budget

### Per-rank estimate (C180 resolution, L72)

| Category | Fields | Size per field | Total |
|----------|--------|----------------|-------|
| Prognostic 3D | ~15 | 180×180×72×8 = 149 MB | 2.2 GB |
| Thermodynamic 3D | ~5 | 180×180×72×8 = 149 MB | 0.7 GB |
| Grid 2D | ~35 | 180×180×8 = 0.3 MB | 10 MB |
| Workspace | ~16 | 180×180×72×8 = 149 MB | 2.4 GB |
| **Total** | | | **~5.3 GB** |

Fits comfortably on modern GPUs (16-80 GB memory).

### Scaling

| Resolution | Estimated GPU memory |
|------------|---------------------|
| C48 | ~0.4 GB |
| C90 | ~1.5 GB |
| C180 | ~5.3 GB |
| C360 | ~21 GB |
| C720 | ~85 GB (requires multi-GPU) |

## Transfer overhead

### Initial transfer (one-time)

For C180 L72: ~3 GB transfer at 12 GB/s PCIe = 0.25 seconds

### Per-step overhead (Stage 1)

| Operation | Data volume | Time at 12 GB/s |
|-----------|-------------|-----------------|
| Halo D2H | ~50 MB | 4 ms |
| MPI halo | ~50 MB | network-dependent |
| Halo H2D | ~50 MB | 4 ms |
| **Total** | | **~10 ms + MPI** |

This is acceptable overhead for ~7.5 minute model time steps.

### Physics sync overhead

Full prognostic sync: ~2.5 GB at 12 GB/s = 0.2 seconds

Acceptable for physics timesteps (typically 5-10 minutes model time).

## Implementation status

| Feature | Status |
|---------|--------|
| Initial H2D transfer | Implemented (fv3_gpu_initialize_state) |
| GPU-resident execution | Implemented (WP-E orchestration) |
| Halo exchange (Stage 1) | NOT IMPLEMENTED |
| Physics sync | Implemented (fv3_gpu_interface_sync) |
| Diagnostics sync | Implemented (fv3_gpu_interface_sync) |
| GPU-direct MPI | NOT IMPLEMENTED (Stage 2) |

## Blockers

1. **Halo exchange not wired** — Stage 1 host-staged exchange not integrated
2. **MPI topology mapping** — Need to map FV3 domain decomposition to GPU ranks
3. **Multi-GPU support** — Single GPU per rank assumed (Stage 1)
