# Milestone 3A.12 — Standalone Authoritative Fortran dyn_core Performance Baseline

Base commit at start of work: `466e4a8` (branch `cuda/gpu-bringup`)

Status: **FORTRAN PERFORMANCE BASELINE BLOCKED**

No source, build, or CI files were modified for this milestone. This document
records the investigation and the exact dependency that blocks it, per
Section 16/20 of the task prompt ("do not spend multiple sessions rebuilding
GEOS infrastructure merely to get a timing number").

---

## Section 1 — Authoritative Fortran source: RESOLVED

Required: GEOS-ESM/GFDL_atmos_cubed_sphere, tag `geos/v3.0.0`, SHA
`ca30e26305fda43831095334c02a3f5e0b8c0c45`.

Two independent, already-present locations on Discover confirm this exact
revision. Neither required cloning:

1. **Pristine standalone checkout:**
   `/gpfsm/dnb06/projects/p295/jli30/fv3_reference/GFDL_atmos_cubed_sphere`
   Detached HEAD, `git rev-parse HEAD` = `ca30e26305fda43831095334c02a3f5e0b8c0c45`,
   tag `geos/v3.0.0`, remote
   `https://github.com/GEOS-ESM/GFDL_atmos_cubed_sphere.git`, clean tree.

2. **`@fvdycore` submodule inside a fully-built production GEOSgcm model:**
   `/gpfsm/dnb06/projects/p295/jli30/GEOSgcm/src/Components/@GEOSgcm_GridComp/GEOSagcm_GridComp/GEOSsuperdyn_GridComp/@FVdycoreCubed_GridComp/@fvdycore`
   `git rev-parse HEAD` = `ca30e26305fda43831095334c02a3f5e0b8c0c45` — an
   independent, exact match to location (1). GEOSgcm's own `components.yaml`
   pins `fvdycore` to tag `geos/v3.0.0`. GEOSgcm HEAD is
   `3d2ea6aa083e196b49081d11f8c9be6f616e630c`, and its
   `@FVdycoreCubed_GridComp` layer (HEAD `ac43844bae40dcc9558deda85c7aeebb4d06cf41`,
   remotes `nasa-nccs-hpda/FVdycoreCubed_GridComp` and
   `GEOS-ESM/FVdycoreCubed_GridComp`) is a completely separate git working
   tree from the repository this project develops CUDA code in.

**Authoritative source path used for inspection:**
`/gpfsm/dnb06/projects/p295/jli30/fv3_reference/GFDL_atmos_cubed_sphere`
**Authoritative SHA confirmed:** `ca30e26305fda43831095334c02a3f5e0b8c0c45` (matches required)

`@fvdycore` inside this repository was NOT populated. No cloning was
performed. The "Do NOT clone an arbitrary latest branch" and BLOCKED escape
valves in Section 1 were not triggered — the exact revision was obtained
safely, twice, independently.

---

## Section 2 — Search for an existing Fortran driver: RESOLVED (none found)

Searched both `GFDL_atmos_cubed_sphere` (the authoritative source) and
`FVdycoreCubed_GridComp` (this repository) for anything that reaches
`dyn_core` directly with usable timing output, before considering building
new code, per the explicit instruction "Prefer reusing an existing executable
if it reaches dyn_core directly."

### 2.1 Standalone drivers in the source tree

`grep -rl '^\s*program\s'` across the whole authoritative checkout finds only:

- `model/tp-core-driver/main.f90` — drives `tp_core.F90`'s `fv_tp_2d` only.
- `model/mapz-driver/main.F90` — drives `fv_mapz.F90` only.
- `geos_utils/cub2latlon_driver.F90`, `geos_utils/CUB2CUB/cub2cub_driver.F90` —
  regridding utilities, unrelated to dyn_core.
- `model/tp-core-driver/test.F90` — an apparently-unused/leftover CUDA-Fortran
  (`cudafor`) GPU-vs-CPU leaf-kernel timing harness for `fv_tp_2d`. It is not
  referenced by `model/tp-core-driver/CMakeLists.txt`, which instead builds
  `driver_cpu.f90` + `main.f90`.

`esma_add_subdirectories` in the top-level `CMakeLists.txt` registers only
`model/mapz-driver` and `model/tp-core-driver`. **No dyn_core-level driver
exists anywhere in the authoritative source tree.**

### 2.2 Why the existing leaf-routine drivers cannot be extended to dyn_core

`tp-core-driver` reaches `fv_tp_2d` using hand-written stubs in place of real
infrastructure:

```
model/tp-core-driver/stubs/fv_mp_mod_stub.F90    (11 lines)
model/tp-core-driver/stubs/fv_grid_utils_stub.F90 (11 lines)
model/tp-core-driver/stubs/fv_arrays_stub.F90     (103 lines)
```

The `fv_mp_mod_stub.F90` stub in full:

```fortran
module fv_mp_mod
  implicit none
  private
  public ng
  integer, parameter :: ng = 3
end module fv_mp_mod
```

This is the entire replacement for `fv_mp_mod` used by the driver — it
supplies nothing but the halo-width constant. `tp_core`'s `fv_tp_2d` operates
on a single in-memory 2D tile array; it has no structural dependency on real
inter-tile communication, so a one-line stub is sufficient and the driver
never touches `mpp_domains_mod` at all.

`dyn_core` cannot be stubbed the same way. Its argument list requires real
`fv_grid_bounds_type`/`fv_grid_type`/`fv_flags_type`/`fv_nest_type` derived
types (normally built by `fv_control_mod`'s initialization), a live FMS
`domain2D` (`mpp_domains_mod`), and an array of `group_halo_update_type`
packs (`fv_mp_mod`'s `start_group_halo_update`/`complete_group_halo_update`).
Six-tile cubed-sphere cross-tile boundary exchange is not incidental
plumbing here — it is structurally part of dyn_core's numerics (C-grid/D-grid
edge and corner handling across tile seams), which is exactly what this
project's own CUDA benchmark had to implement and separately verify for
liveness (Milestone 3A.11V §4.3, `fv3_dispatch_smoke` Phase E). A
`fv_mp_mod` stub that returns a constant instead of performing real halo
exchange would silently make the Fortran side do materially less work than
the CUDA side — violating the task's own "meaningfully equivalent work"
requirement rather than satisfying it. There is no existing precedent in this
codebase for stubbing dyn_core's communication dependency this cheaply.

### 2.3 GEOSgcm — a full production model already built at the exact pinned SHA

`/gpfsm/dnb06/projects/p295/jli30/GEOSgcm` is a completed CMake/ecbuild
configure+build (not just a source checkout): `build/lib/libfvdycore.a`,
`build/lib/libFVdycoreCubed_GridComp.a`, `install/lib/libfvdycore.a`,
`install/lib/libFVdycoreCubed_GridComp.a` all exist, `dyn_core.F90.o` is
compiled with its `.mod` present, and a top-level model executable exists at
`install/bin/GEOSgcm.x`.

`fv_dynamics.F90` (the caller of `dyn_core`) already instruments the call
with the project's own timing module:

```fortran
call timing_on('DYN_CORE')
call dyn_core(npx, npy, npz, ng, sphum, nq, mdt, k_split, n_split, ...)
call timing_off('DYN_CORE')
```

This is exactly the kind of built-in instrumentation that could, in
principle, yield a per-call Fortran dyn_core time without writing new driver
code — but only from a run of `GEOSgcm.x`, which requires full model
bootstrap (MAPL/ESMF GridComp hierarchy, resource-file parsing, boundary
condition and restart datasets, ExtData). This is precisely what Section 4
says to avoid "if avoidable."

An already-staged GEOSgcm experiment was found at
`/gpfsm/dnb06/projects/p295/jli30/fv3_runs/fv3_baseline/scratch_c24_L181_T2_4x24_1threads_GEOSgcm-v11.10.1-R4R8/`
— a completed multi-hour, full-physics AMIP-style run (hourly netCDF
diagnostic output spanning `1891-03-01` through `1891-03-02`) at cube-24,
181 vertical levels. This run:

- is resolution `c24`/`L181`, not `8×8`, `16×16`, or `32×32` at 10 levels;
- includes full physics, not "no physics outside dyn_core";
- represents someone's completed baseline validation output and must not be
  disturbed or reused/mutated for an unrelated purpose ("Do not discard or
  reset existing work without explicit instruction").

It confirms that GEOSgcm.x is runnable on this system, but it does **not**
provide, and cannot be repurposed without a fresh reconfiguration effort to
provide, a dyn_core-only timing at the required problem sizes.

**Conclusion of Section 2: no existing executable, driver, or artifact reaches
dyn_core directly at the required problem sizes with usable timing output.**
A new mechanism would have to be built.

---

## Section 3 — Comparability classification: NOT COMPARABLE (no viable harness within scope)

Two candidate mechanisms for obtaining a Fortran timing number were assessed
against the "smallest practical" / "avoid full GEOS initialization if
avoidable" / "do not spend multiple sessions rebuilding GEOS infrastructure"
constraints (Sections 4 and 16):

**(a) Reconfigure/rerun GEOSgcm.x at low resolution (c8/c16/c32, L10, dyn_core
only).** This requires full MAPL/ESMF/ExtData bootstrap regardless of grid
size — the "full GEOS initialization" the prompt says to avoid — plus
locating or fabricating boundary-condition/restart data at three new,
non-standard resolutions, plus disabling every physics GridComp so only
dyn_core executes, plus locating or building an AGCM.rc/fvcore_layout.rc
configuration that supports `npz=10` and `n_split=3` at those grid sizes.
No such configuration is currently staged anywhere found on Discover. This is
open-ended GEOS-experiment engineering, not a quick timing run.

**(b) Write a new minimal Fortran driver that calls `dyn_core` directly,**
linking against GEOSgcm's already-built `libfvdycore.a` (compilation itself
is not the blocker — it is already done). This requires constructing, from
scratch, the pieces that `fv_control_mod`'s initialization sequence normally
builds: a genuine FMS `mpp_domains_mod` six-tile cubed-sphere mosaic/`domain2D`
(tile connectivity, halo width, corner handling), populated
`fv_grid_bounds_type`/`fv_grid_type`/`fv_flags_type`/`fv_nest_type`/
`fv_diag_type` derived types, and an array of `group_halo_update_type` packs
via `fv_mp_mod`'s `start_group_halo_update`. As shown in Section 2.2, no
existing stub pattern in this codebase provides a shortcut for this — the one
precedent driver (`tp-core-driver`) stubs communication down to a single
constant specifically because its target routine has no real cross-tile
dependency, and dyn_core's does. Building this correctly is materially the
same class of work as re-implementing a meaningful subset of
`fv_control_mod`'s grid/domain initialization — not a "smallest practical"
wrapper around an existing lifecycle.

Both paths cross into "extensive framework work" as described in Section 16.
Per that section's explicit instruction, this is reported as **BLOCKED**
rather than attempted piecemeal across additional sessions.

**Comparability classification: NOT COMPARABLE** — no Fortran-side timing
artifact exists, or can be produced within the stated scope, to compare
against the CUDA medians.

---

## Sections 4–15 — Not attempted

No driver was built (Section 4), no IC matching was performed (Section 5), no
compiler/build configuration was established for a Fortran benchmark
(Section 6), no CPU resource baseline was measured (Sections 7–8), no timing
loop was implemented or run (Sections 9–11), and consequently no speedup
ratios were computed (Sections 12–14), consistent with Section 16's
instruction to stop rather than continue once the "extensive framework work"
threshold is reached.

## Section 15 — Numerical validation boundary

Unaffected by this milestone. Numerical validation against authoritative
runtime Fortran savepoints remains **NOT RUN**, as recorded in prior
milestones. This document does not claim any Fortran/CUDA numerical
equivalence.

---

## Existing CUDA reference numbers (unchanged, from Milestone 3A.11)

R8, one A100, `warpa003`, PRE-VALIDATION, resident/transfer-inclusive median
of 100 timed calls after 20 warmup, both modes, no environment overrides:

| Grid  | CUDA resident (ms/call) | CUDA transfer-inclusive (ms/call) |
|-------|--------------------------|-------------------------------------|
| 8×8   | 28.332                   | 29.358                              |
| 16×16 | 29.690                   | 30.896                              |
| 32×32 | 28.827                   | 31.178                              |

These numbers are unchanged by this milestone; no Fortran counterpart exists
to compute a speedup ratio against them.

---

## Section 18 — Git discipline

No authoritative Fortran was modified. No driver/wrapper code was created,
because none could be justified within the scope of this milestone (see
Section 3). Nothing was built. This document is the only file added by this
milestone; it will be committed on its own. No build directories,
executables, Slurm logs, or timing logs are added.

---

## Caveats

- This investigation used only pre-existing artifacts on Discover (two
  independent checkouts of the pinned SHA, one fully-built GEOSgcm tree, one
  unrelated staged c24/L181 GEOSgcm experiment). Nothing was cloned, built,
  configured, or run for this milestone.
- The blocking dependency is specific and actionable if a future milestone is
  explicitly scoped for it: construct a genuine FMS six-tile cubed-sphere
  `mpp_domains_mod` domain plus the `fv_control_mod`-equivalent derived-type
  population (`gridstruct`/`flagstruct`/`neststruct`/`idiag`/`bd`) required by
  `dyn_core`'s signature (`model/dyn_core.F90:163-223`), OR obtain/construct a
  GEOSgcm experiment configuration at c8/c16/c32 with `npz=10`, `n_split=3`,
  and all non-dyn_core physics disabled.
- `fv_dynamics.F90`'s existing `timing_on('DYN_CORE')`/`timing_off('DYN_CORE')`
  instrumentation (around the `call dyn_core(...)` at
  `model/fv_dynamics.F90:645`) is a genuine, reusable asset for a future
  attempt via path (a) above — it does not need to be added.

---

## PRE-VALIDATION notice

All numbers in this document (the unchanged CUDA reference table) remain
PRE-VALIDATION. No Fortran baseline was produced this milestone.
**Numerical validation against authoritative runtime Fortran savepoints:
NOT RUN.**
