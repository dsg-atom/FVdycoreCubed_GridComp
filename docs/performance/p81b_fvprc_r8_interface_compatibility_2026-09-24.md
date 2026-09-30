# Prompt 81B — Generalize FVPRC/bare-real R8 Compatibility Fix

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Prerequisite: Prompt 80 (`p80_fv_statemod_build_tree_repair_2026-09-24.md`)
— classified the `FV_StateMod.F90`/`fv_diagnostics.F90` `#6633` build
failure as "Category G — genuine, pre-existing source/toolchain-level
mismatch," blocking without an authorized fix.

## 0. Correction to the inherited premise (read this first)

The task arrived with an inherited "verified root cause" naming five
routines (`bunkers_vector`, `prt_maxmin`, `adjust_fv3_splits`,
`updraft_helicity`, `helicity_relative_CAPS`) as the fix targets, and a
mid-task working estimate of "416 errors across ~140 distinct lines"
across the whole of `FV_StateMod.F90`. **Both are superseded.**

A fully clean rebuild of the `fvdycore` CMake target (all `.o`,
`.provides.build`, and `.mod` state deleted, `make -j4 fvdycore`, EXIT=0,
0 errors) collapsed a subsequent `-syntax-only -diag-error-limit=500`
rescan of `FV_StateMod.F90` from 416 errors / ~140 lines down to **8
errors / 5 lines, all in one call**. This is the same stale-build-tree
bug class Prompt 80 attributed to Category G from an *incremental*
rebuild — the actual root cause was staleness, not a toolchain-level
defect, and it required **zero source changes** to resolve for every
routine Prompt 81A/Prompt 79 had named. None of `bunkers_vector`,
`prt_maxmin`, `adjust_fv3_splits`, `updraft_helicity`,
`helicity_relative_CAPS`, `calculate_shear_06` error after a genuinely
clean rebuild.

The one call that survives a clean rebuild is `Held_Suarez_Forcing`
(`FV_StateMod.F90:2041-2043`, 8 arguments) — a routine **not named in any
prior prompt's list**. Three alternative explanations were checked and
ruled out before concluding a source fix was the right minimal action:

- Hand-written `interface` block out of sync with the routine body: ruled
  out — `FV_StateMod.F90` has no such block for this call (only
  `fv_computeMassFluxes` has one, unrelated).
- CMake `Fortran_DEFINES` asymmetry (`fvdycore` defines
  `-DMOIST_CAPPA -DUSE_COND -DSPMD -DTIMING -DINTERNAL_FILE_NML`,
  `FVdycoreCubed_GridComp` does not): ruled out — zero `#ifdef`
  references to these macros exist in `FVdycoreCubed_GridComp`'s own
  files, and `fv_arrays.F90`'s `#ifdef MOIST_CAPPA`/`USE_COND` blocks only
  add unrelated extra fields, never touching `u`/`v`/`pt`/`pe`/`ps`/
  `agrid`.
- Duplicate/shadowed `fv_arrays_mod.mod` via include-path ordering: ruled
  out — no duplicate found under `build/include/FVdycoreCubed_GridComp/`.

The exact ifort-internal mechanism for why a confirmed-fresh, confirmed-
uniformly-`-r8`'d, textually-bare-`real`-on-both-sides call still trips
`#6633` was not pinned down. Per this project's bounded-debugging
convention, that residual question was not pursued further once a
minimal, semantics-preserving fix was in hand — this is flagged
transparently as a cosmetic open question, not a blocker.

## 1. Precedent Pattern

Prompt 81A had already fixed one instance of this class (`shr06`'s
output side inside `updraft_helicity`) via a `real(FVPRC)` local
intermediate + copy-out at the call site. The `kappa` and `myDT` actuals
at the `Held_Suarez_Forcing` call site are already explicitly
`real(FVPRC)`-typed in `FV_StateMod.F90` (lines 240, 1318) — the pattern
of "give the FV3 state-facing quantity an explicit `real(FVPRC)` type
that matches `FV_Atm(1)%*`" is already established project-wide; this fix
generalizes it to the *callee's own dummy-argument declarations* instead
of a call-site-local copy, which is smaller and cleaner for a
single-call-site, self-contained routine.

## 2. Mismatches Table

| Call site | Args mismatched | Status after clean rebuild |
|---|---|---|
| `p_var` (`init_hydro.F90`) | — | 0 errors — stale-build artifact only |
| `bunkers_vector` | — | 0 errors — stale-build artifact only |
| `prt_maxmin` | — | 0 errors — stale-build artifact only |
| `adjust_fv3_splits` | — | 0 errors — stale-build artifact only |
| `updraft_helicity` | — | 0 errors — stale-build artifact only |
| `helicity_relative_CAPS` | — | 0 errors — stale-build artifact only |
| `calculate_shear_06` | — | 0 errors — stale-build artifact only |
| `Held_Suarez_Forcing` (`FV_StateMod.F90:2041-2043`) | `u,v,pt,pe,ps,agrid,dt(myDT),kappa` (8) | **Genuine — fixed this task** |

## 3. Source Changes

One file: `Held_Suarez_ForcingMod.F90`
(`GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/`).

- `use fv_arrays_mod, only: fv_grid_bounds_type` →
  `use fv_arrays_mod, only: fv_grid_bounds_type, FVPRC`
- 8 dummy-argument declarations (`u`, `v`, `pt`, `pe`, `ps`, `agrid`,
  `dt`, `kappa`) changed from bare `real` to `real(FVPRC)`.
- Local variables (`pfull, sigma, lat, sinlat2, coslat2, teq, kt, kv,
  sigma_ratio`) and the `HS_*` canonical-parameter constants deliberately
  left as bare `real` — `#6633` only checks actual/dummy correspondence
  at call sites, not intra-procedure arithmetic, and both resolve
  identically (8-byte) in this R8 build; changing them would be
  unrelated cleanup outside this task's change budget.
- Confirmed via `grep -n "call "` that `Held_Suarez_Forcing` makes no
  further internal subroutine calls — no secondary mismatch risk.
- Numerical semantics changed: **NO**. `FVPRC=REAL8` in this R8 build is
  byte-identical to what bare `real` already resolved to under `-r8`;
  `FVPRC=REAL4` in the R4 build matches what bare `real` resolves to
  there too (no `-r8` in R4 builds) — verified via `fv_arrays.F90`'s
  `#ifdef SINGLE_FV` branch and confirmed by an R4-defines syntax-only
  compile of this file (0 errors).
- Git status: `Held_Suarez_ForcingMod.F90` in this
  `GEOSfvdycore_fortran_perf` checkout is the same untracked,
  load-bearing file `CURRENT_STATE.md` already flagged (referenced by
  `CMakeLists.txt`/`FV_StateMod.F90` but never committed) — this edit
  does not change that pre-existing untracked status.

## 4. Build (Stage 3)

- Clean rebuild of `fvdycore` target: EXIT=0, 0 errors
  (`/tmp/p81b_fvdycore_clean_rebuild.log`).
- Rescan of `FV_StateMod.F90` after clean `fvdycore` rebuild: 8 errors /
  5 lines, all `Held_Suarez_Forcing` (`/tmp/p81b_rescan_after_clean.log`).
- Clean rebuild of `FVdycoreCubed_GridComp` target (all `.o`/
  `.provides.build`/`.mod` deleted first) after the source fix: **EXIT=0,
  0 `error #` matches total** (`/tmp/p81b_gridcomp_rebuild.log`),
  confirmed both `Held_Suarez_ForcingMod.F90.o` and `FV_StateMod.F90.o`
  were rebuilt (not skipped) in this run.
- Relink `StandAlone_FV3_Dycore.x`: `make -j4 StandAlone_FV3_Dycore.x`,
  EXIT=0 (`/tmp/p81b_link.log`), only pre-existing benign linker warnings
  (`libcrypto.so.1.1`/`libcrypto.so.3` conflict notices, unrelated).
- **Stage 3: 0 `#6633` errors end-to-end. PASS.**

## 5. R4 Spot Check (Stage 4)

`build_r4`'s own `flags.make` defines `SINGLE_FV` (no `-r8`); confirmed
`fv_arrays.F90`'s `#ifdef SINGLE_FV` branch sets `FVPRC = REAL4` in that
configuration. Syntax-only compile of `Held_Suarez_ForcingMod.F90` under
the R4 target's exact `Fortran_DEFINES`/`Fortran_INCLUDES`/`Fortran_FLAGS`:
0 errors. The fix is type-consistent by construction in both precision
regimes. (Noted, out of scope: `build_r4`'s `flags.make` `-module` path
points at `build/include` — the R8 tree — a pre-existing build-tree
oddity unrelated to this fix; not touched.) **Stage 4: PASS (compile-
compatibility only, no science validation, as scoped).**

## 6. Executable Provenance (Stage 5)

```
Path:   /discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
mtime:  2026-09-24 20:55:47
sha256: e82d526aa0b3b96541f27073ceb5e094112932f878989e931b349bace4474599
```

- `ldd`: linked against Intel MPI 2021.13 (`libmpi.so.12`,
  `libmpifort.so.12`, `libmpicxx.so.12` from
  `/usr/local/intel/oneapi/2024/mpi/2021.13/...`) — matches the
  authoritative production stack, not the OpenMPI dev stack.
- `strings` confirms all Prompt 77-79 dispatch symbols present:
  `use_cuda_tp2d=`, `use_cuda_remap_uv=`, `RUN_CUDA_FV3_TRACER_FORCE_FORTRAN`,
  `RUN_CUDA_FV3_PTDELZ_*`, `RUN_CUDA_FV3_REMAPUV_*`,
  `RUN_CUDA_FV3_FUSE_HALO_EDGES`/`_FUSE_CORNER_FILL`, and — critically for
  Prompt 79 — `[P79_TRANSFER_COUNT] h2d=%d d2h=%d total_bytes=%zu
  out_bytes=%zu` / `RUN_CUDA_FV3_TRACER_TRANSFER_COUNT_DUMP`, confirming
  this relink picked up the Prompt 79 batched-transfer CUDA bridge (via
  `CMakeCache.txt`'s `FV3_CUDA_LIBRARY_DIR` =
  `.../FVdycoreCubed_GridComp/build_cuda_r8_intelmpi`).
- `RUN_HELD_SUAREZ`/`fv_statemod_mp_held_suarez_`/
  `held_suarez_forcingmod_mp_held_suarez_forcing_` symbols present,
  confirming the fixed routine linked correctly.

## 7. Smoke Test (Stage 6)

Reused the `p77_stage7_c32`-style harness (new script
`p81b_stage6_c32_smoke.sbatch`, sha256-gated preflight, same experiment
directory `p77_stage7_cuda`). Job 58549955, `warpa009`, C32, 6 ranks/1
GPU: **rc=0, NaN=0, Inf=0, Abort=0, `SMOKE_PASS`.** Confirmed via the
job's own `USING ****` log line that the freshly-rebuilt binary (copied
via the experiment's wrapper script, which `exec`s
`build/bin/StandAlone_FV3_Dycore.x` directly) actually ran, and that
`use_cuda_tp2d=T` fired in the `[P77_EXIT_DUMP]` lines — confirming the
CUDA tracer-dispatch path is live on this binary.

## 8. Prompt 79 Resumption (Stage 7)

See the appended Prompt 79 continuation section in
`p79_tracer_batched_transfer_optimization_2026-09-24.md` for the full
Stage 9-13 report. Summary: Stages 9 (C32 dispatch) and 10 (C512
dispatch) both PASS; Stage 11 (same-node alternating A/B, 3 pairs, C512)
found TRACER_2D CUDA ~54% slower than Fortran (range 48-59% across the
3 pairs) — **the batched-transfer fix did not recover the Prompt 78
regression; Stage 13 verdict is REGRESSION-REMAINS.** No default flipped.

---

**FVPRC R8 COMPATIBILITY FIX APPLIED — STANDALONE REBUILT — PROMPT 79
UNBLOCKED**
