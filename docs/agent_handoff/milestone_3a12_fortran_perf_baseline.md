# Milestone 3A.12 — Standalone Authoritative Fortran dyn_core Performance Baseline

Base commit at start of work: `466e4a8` (branch `cuda/gpu-bringup`)

Status: **FORTRAN-CUDA PERFORMANCE BASELINE ESTABLISHED — READY FOR PROMPT 6A**

This document supersedes an earlier draft of this same milestone (previously
recorded as BLOCKED) after locating a working, already-buildable standalone
GEOSfvdycore fixture on Discover that reaches `dyn_core` at the required
problem sizes via its own `fv_dynamics.F90`
`timing_on('DYN_CORE')`/`timing_off('DYN_CORE')` instrumentation, without
writing any new driver code and without a full physics-enabled GEOSgcm run.

No CUDA source was modified. No authoritative Fortran source was modified.
All configuration changes are runtime resource-file content generated inside
an experiment job script (`fv3.j`), not committed source.

---

## Section 1 — Authoritative Fortran source: RESOLVED

Fixture: `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
(a standalone GEOSfvdycore build/run tree, independent of both the CUDA
repository and the `GEOSgcm` tree referenced in the earlier draft of this
document).

- Fixture repo HEAD: `8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`
- `@FVdycoreCubed_GridComp` layer HEAD:
  `ac43844bae40dcc9558deda85c7aeebb4d06cf41`
- `@fvdycore` (GFDL_atmos_cubed_sphere) SHA:
  `ca30e26305fda43831095334c02a3f5e0b8c0c45`, tag `geos/v3.0.0`

This is the exact same authoritative revision identified (but not run) in the
earlier draft of this milestone via two independent checkouts. It confirms
the fixture uses the correct pinned authoritative source.

**Executable used:**
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/install/bin/StandAlone_FV3_Dycore.x`
(pre-existing standalone dyn_core driver shipped with GEOSfvdycore — not
written for this milestone).

**Compiler:** Intel `ifort` (`/usr/local/intel/oneapi/2024/compiler/2024.2`),
CMake `Release` build, flags:
`-O3 -march=core-avx2 -align array32byte -traceback -assume realloc_lhs
-qopt-report0 -align dcommons -fno-alias -fp-model fast=1 -fp-model source
-fp-model consistent -no-fma -fimf-arch-consistency=true
-fp-speculation=safe -fpe1 -assume noold_maxminloc -diag-disable 10121
-diag-disable 10448`.

Note: this fixture is Intel-compiler-based, not the GCC/nvhpc-byo-compiler
stack used for CUDA bring-up. Per `docs/agent_instructions/project_architecture.md`'s
compiler-family/ABI caveat, this is acceptable here because the Fortran
baseline is a fully separate standalone executable with no linkage to the
CUDA runtime — no ABI compatibility claim is made or needed for this
milestone.

---

## Section 2 — Driver: RESOLVED (existing standalone executable used, no new driver written)

`StandAlone_FV3_Dycore.x` already builds a genuine six-tile cubed-sphere FMS
`mpp_domains_mod` domain, populated grid/flags/nest derived types, and real
`group_halo_update_type` MPI halo-exchange packs via its normal
`fv_control_mod` initialization/lifecycle — i.e. exactly the infrastructure
that the earlier draft of this document identified as the blocking
dependency for a hand-written driver (Section 2.2 of the prior draft). Using
this pre-existing executable instead of writing a new one avoids duplicating
that infrastructure and avoids the "extensive framework work" scope creep the
task explicitly says to avoid.

`fv_dynamics.F90` wraps the call with:

```fortran
call timing_on('DYN_CORE')
call dyn_core(npx, npy, npz, ng, sphum, nq, mdt, k_split, n_split, ...)
call timing_off('DYN_CORE')
```

`fv_timing_mod`'s accumulated `DYN_CORE` total (seconds, printed once per run
in the `timing_prt` table) divided by the number of `dyn_core` calls in that
run is the primary metric reported below.

---

## Section 3 — Comparability classification: EQUIVALENT (with one documented substrate difference)

Both sides execute the authoritative `dyn_core` numerics at 8×8×10,
16×16×10, and 32×32×10 (six cubed-sphere faces, `npz=10`), R8, `n_split=3`,
`k_split=1`, adiabatic (no physics). Both sides use exactly 6 ranks/execution
units — one per cubed-sphere face (Fortran: `FV_NX=1`/`FV_NY=6`, 6 real MPI
ranks; CUDA: 6 tiles on 1 GPU, Milestone 3A.11).

**Documented substrate difference (not a normalized hardware-efficiency
comparison, per Section 15 of the task prompt):** the Fortran side performs
real inter-process MPI halo exchange across 6 ranks on CPU; the CUDA side
performs its own device-side halo exchange across 6 tiles resident on a
single GPU (Milestone "Implement FV3 GPU cubed-sphere halo exchange",
commit `b0b1266`). This is a hardware/communication-substrate difference
intrinsic to comparing a multi-process CPU run against a single-GPU run, not
a scope or numerics difference in `dyn_core` itself.

---

## Section 4 — Resolution / configuration mapping

| Grid  | `AGCM_IM`/`FV3_NPX-1` | `RUN_DT` (s) | `JOB_SGMT` | Calls/rep |
|-------|-----------------------|--------------|------------|-----------|
| 8×8   | 8                     | 3600         | 7 days     | 168       |
| 16×16 | 16                    | 1800         | 7 days     | 336       |
| 32×32 | 32                    | 1200         | 7 days     | 504       |

`npz = 10` (`AGCM_LM=10`) for all three. Call counts were independently
confirmed via MAPL's own profiler (`MAPL_ENABLE_TIMERS: YES` in `CAP.rc`),
whose `--------DYN_CORE` row reports an exact `# cycles` column matching the
expected count in every one of the 15 reps run.

**Runtime layout:** `FV_NX=1`, `FV_NY=6` in each experiment's `fv3.j` (6
total MPI ranks, one per cubed-sphere face — the minimal layout, edited from
`fv3_setup`'s much larger per-resolution defaults, e.g. 4×24=96 ranks at
IM=16 and 6×36=216 ranks at IM=32).

**`n_split`/`k_split`:** added `k_split = 1` / `n_split = 3` to each
experiment's `&fv_core_nml`. Verified on every run via both of
`fv_control.F90`'s independent runtime echoes: `"Using n_split from the
namelist: 003"` and `"FV3 n_split:   3"`.

**Vertical eta levels (`AK:`/`BK:`):** the GMAO-specific `m_set_eta`
implementation (`src/Shared/@GMAO_Shared/GMAO_hermes/m_set_eta.F90`) — which
is what `DynCore_GridCompMod.F90` actually calls (`use m_set_eta, only:
set_eta`), not `@fvdycore`'s own `fv_eta_mod::set_eta` — has no lookup-table
entry for `km=10` and silently returns all-zero AK/BK via a "Bad KM"
fallback, tripping an assertion at `DynCore_GridCompMod.F90:8235`
(`_ASSERT(ANY(AK /= 0.0) .or. ANY(BK /= 0.0), ...)`). Fix: supply `AK:`/`BK:`
directly in each experiment's `AGCM.rc` (11 values each, one per line,
strictly below the label per `ESMF_ConfigFindLabel`/`ESMF_ConfigNextLine`
semantics), which makes `DynCore_GridCompMod.F90` skip the `set_eta` call
entirely. Values used are the generic hybrid sigma-pressure construction
found in `@fvdycore`'s own (unused-here, but algorithmically identical)
`fv_eta_mod::set_eta` `case default` branch: `pt=100 Pa`, `pint = pt +
1e5/km`, uniform sigma spacing below, producing a monotonic edge-pressure
profile from 100 Pa (top) to 100000 Pa (surface). This is a config-only
change (no source modified) and does not alter any scientific algorithm —
`@fvdycore`'s own set_eta would have produced the same values had it been
the one called.

**`COLLECTIONS:` (MAPL History):** kept at the default two collections
(`inst3_3d_diag`, `inst1_2d_diag`). An earlier attempt to set this fully
empty (`COLLECTIONS: ::`) to minimize non-dyn_core work crashed
`MAPL_HistoryGridComp.F90` (`"Premature end of fields list"`, status=41) —
MAPL's History component cannot parse a zero-collection list. Reverted to
default; History overhead is excluded from the reported metric regardless,
since the metric is `fv_timing_mod`'s own `DYN_CORE`-scoped accumulator, not
wall-clock.

**Sanity checks:** `range_warn = .T.` (NaN/Inf/out-of-bound monitoring)
enabled in `&fv_core_nml` for all runs; `fv_debug = .F.` kept off to avoid
timing contamination. All 15 reps (5 per grid size) were grepped for
`FAIL`/`error`/`abort`/`nan`/any `range_warn` trigger and all 15 were clean.
Slurm exit `Status = 0` confirmed for all 15.

---

## Section 5 — Methodology (statistical baseline via repeated full-length runs)

`fv_timing_mod` has no reset hook or exposed call counter, so a single
`dyn_core` call cannot be isolated by resetting the timer between calls.
Instead, each experiment's `JOB_SGMT` was set to 7 model-days, producing
168/336/504 `dyn_core` calls per run (dominating any one-time
initialization skew), and mean ms/call was computed as the accumulated
`DYN_CORE` total from the `timing_prt` table divided by the call count. Each
grid size was run 5 times (5 independent `sbatch` submissions); min / median
/ mean / max are reported, with **median as the primary number**.

Each run's `tee`'d scratch log was copied to a permanently-named `rep{N}.log`
immediately after completion and before the next submission, since the
`fv3.j` `tee` redirect overwrites the same log path on every resubmission of
the same experiment.

---

## Section 6 — Hardware and run identity

- Platform: NASA NCCS Discover, Slurm `--constraint=mil` (AMD Milan CPU
  nodes), `--nodes=1 --ntasks-per-node=120`, 6 MPI ranks actually used per
  run (`esma_mpirun -np 6`).
- Compute nodes used (via `sacct`): IM=8/IM=16 reps on assorted `mil`
  partition nodes; IM=32 reps confirmed on `borgl161`.
- Each run completed in 32–44 s wall-clock (Slurm `Elapsed`), well within
  the "keep turnaround short" instruction.

---

## Section 7 — Fortran timings (all 15 reps, 3 grid sizes)

### 8×8×10 (168 calls/rep)

| Rep | `DYN_CORE` total (s) | ms/call |
|-----|-----------------------|---------|
| 1   | 2.1166                | 12.599  |
| 2   | 2.1480                | 12.786  |
| 3   | 2.1492                | 12.793  |
| 4   | 2.1387                | 12.730  |
| 5   | 2.1187                | 12.611  |

min = 12.599, **median = 12.730**, mean = 12.704, max = 12.793 ms/call

### 16×16×10 (336 calls/rep)

| Rep | `DYN_CORE` total (s) | ms/call |
|-----|-----------------------|---------|
| 1   | 5.5631                | 16.557  |
| 2   | 5.7042                | 16.977  |
| 3   | 5.6127                | 16.704  |
| 4   | 5.6324                | 16.763  |
| 5   | 5.5756                | 16.594  |

min = 16.557, **median = 16.704**, mean = 16.719, max = 16.977 ms/call

### 32×32×10 (504 calls/rep)

| Rep | `DYN_CORE` total (s) | ms/call |
|-----|-----------------------|---------|
| 1   | 15.2044               | 30.168  |
| 2   | 15.0942               | 29.949  |
| 3   | 15.2516               | 30.261  |
| 4   | 15.2554               | 30.269  |
| 5   | 15.3686               | 30.494  |

min = 29.949, **median = 30.261**, mean = 30.228, max = 30.494 ms/call

All 15 reps: `n_split` confirmed = 3 (both echoes) on every rep; 0 of 15
reps showed any FAIL/error/abort/NaN/range_warn trigger; call count
independently confirmed via MAPL profiler `# cycles` on every rep.

---

## Section 8 — CUDA reference numbers (unchanged, from Milestone 3A.11)

R8, one A100, `warpa003`, PRE-VALIDATION, resident/transfer-inclusive median
of 100 timed calls after 20 warmup, 5 reps, median:

| Grid  | CUDA resident (ms/call) | CUDA transfer-inclusive (ms/call) |
|-------|--------------------------|-------------------------------------|
| 8×8   | 28.332                   | 29.358                              |
| 16×16 | 29.690                   | 30.896                              |
| 32×32 | 28.827                   | 31.178                              |

---

## Section 9 — Performance matrix and speedup ratios

Speedup ratio = Fortran ms/call ÷ CUDA ms/call (>1 means CUDA faster; <1
means the Fortran/CPU path is faster at this problem size).

| Grid  | Fortran median (ms/call) | CUDA resident (ms/call) | CUDA transfer-incl. (ms/call) | Resident speedup | Transfer-incl. speedup |
|-------|---------------------------|---------------------------|----------------------------------|-------------------|--------------------------|
| 8×8   | 12.730                    | 28.332                    | 29.358                           | 0.449×            | 0.434×                   |
| 16×16 | 16.704                    | 29.690                    | 30.896                           | 0.563×            | 0.541×                   |
| 32×32 | 30.261                    | 28.827                    | 31.178                           | 1.050×            | 0.971×                   |

**Finding:** at these small problem sizes (8×8×10 through 32×32×10, 6
ranks/tiles), the persistent-GPU-resident CUDA `dyn_core` path is *slower*
per call than the 6-rank CPU Fortran path at 8×8 and 16×16 (by roughly 1.8×–
2.3×), and only reaches approximate parity at 32×32 (a modest 5% CUDA
advantage on resident-only timing; still slightly slower once
transfer-inclusive). This is consistent with fixed per-call GPU kernel
launch/synchronization overhead not yet being amortized by problem size at
these small per-tile grids (max 32×32 points per face). This finding is
reported as measured; per Section 3/15 of the governing task prompt, it is
not a normalized hardware-efficiency comparison (6 real CPU MPI ranks vs. 1
GPU), and no conclusion about CUDA implementation correctness or quality is
implied — only relative wall-clock at these specific small problem sizes.

---

## Section 10 — Numerical validation boundary

Unaffected by this milestone. Numerical validation against authoritative
runtime Fortran savepoints remains **NOT RUN**. This document does not claim
any Fortran/CUDA numerical equivalence — only relative timing.

---

## Section 11 — Git / artifact discipline

No authoritative Fortran (`@fvdycore`, `GMAO_Shared`) was modified. No CUDA
source was modified. All configuration changes (AK/BK override, COLLECTIONS
revert, `FV_NX`/`FV_NY`, `JOB_SGMT`, `n_split`/`k_split`, `range_warn`) are
generated resource-file content inside each experiment's `fv3.j` job script,
which itself lives inside the GEOSfvdycore fixture
(`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/{c8,c16,c32}_L10_dyncore_perf/`),
not inside this CUDA repository. Only this handoff document is added to the
CUDA repository by this milestone. No build directories, executables, Slurm
logs, or scratch/timing logs from the fixture are added here.

---

## PRE-VALIDATION notice

All numbers in this document (both the newly-measured Fortran baseline and
the unchanged CUDA reference table) remain PRE-VALIDATION. **Numerical
validation against authoritative runtime Fortran savepoints: NOT RUN.**
This milestone measured relative wall-clock performance only.

---

**FORTRAN-CUDA PERFORMANCE BASELINE ESTABLISHED — READY FOR PROMPT 6A**
