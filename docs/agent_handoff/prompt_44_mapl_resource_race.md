# Prompt 44 — Root-Cause C512 MAPL Resource-Read Startup Blocker

## Objective (as given)

Root-cause and fix the pre-existing, nondeterministic C512 six-rank startup
crash (`MAPL_Resource_SetValue.h status=39` at `FV_StateMod:FV_Setup:829` /
`DYNSetServices:2606`) discovered blocking Prompt 43's Stage 7 correctness
gate. Occurs before any CUDA halo transport/dyn_core code executes. Model:
Sonnet only. Do not resume Prompt 43's transport A/B until fixed.

## Error meaning

`status=39`'s exact ESMF symbolic name could not be located (module-gated
ESMF headers/source not found on accessible paths — Stage 0 attempted, not
blocking). Not needed: Stage 3's empirical discriminator (below) provides
decisive causal evidence without the symbolic code.

## Resource lifecycle

`FV_StateMod.F90::FV_Setup:829` calls `MAPL_GetResource(MAPL, HELD_SUAREZ,
'RUN_HELD_SUAREZ:', default=HELD_SUAREZ, RC=STATUS)` ->
`MAPL_Generic.F90`'s `MAPL_GetResourceFromMAPL_scalar` ->
`MAPL_Resource.F90`'s `MAPL_GetResource_config_scalar` -> `get_actual_label`
(`ESMF_ConfigFindLabel` on rank-private `state%cf`) ->
`MAPL_Resource_SetValue.h:6` (`ESMF_ConfigGetAttribute`, only taken when the
label **is found** — ruling out a missing-key issue).

`HELD_SUAREZ` was declared `logical` and read **directly** as a `LOGICAL`
via `MAPL_GetResource`. This codebase's own established, already-safe
convention for numeric-literal `.rc` booleans (confirmed present in the same
file: `INT_ADIABATIC`->`ADIABATIC`, `INT_FV_OFF`->`FV_OFF`,
`INT_fix_mass`->`fix_mass`, `INT_check_mass`->`check_mass`) is instead to
read an `INTEGER` proxy through `MAPL_GetResource`, then manually convert to
`LOGICAL` with a plain Fortran `if`. `HELD_SUAREZ` did not follow this
convention. It is also the *only* directly-LOGICAL-read resource in this
file whose label is actually present in this job's `AGCM.rc` (the other two,
`DEBUG`/`ADJUST_SPLITS`, are absent and always take the safe
not-found-default branch, never touching `ESMF_ConfigGetAttribute`).

## Reproduction

- Prompt 43 baseline: 3/3 crashes on the unmodified `RUN_HELD_SUAREZ: 1`
  bare-integer `.rc` format, with a **different failing rank pair each run**
  ({3,0}, {4,5}, {0,3}) — reproduces identically even with the entire
  Prompt 43 transport diff `git stash`ed out (job 58451786), so unrelated to
  transport code.
- This prompt's discriminator (job 58452160, zero source/rebuild change):
  copied the failing run directory, changed only `AGCM.rc`'s
  `RUN_HELD_SUAREZ: 1` -> `RUN_HELD_SUAREZ: .TRUE.`, reran the same
  unmodified binary 3x. Result: **0/3 crashes**, ran the full 10-minute
  walltime making genuine simulated progress (TIMEOUT, not a failure). This
  single-variable experiment is the decisive evidence: the failure tracks
  the `.rc` text format/type-parse path, not any race in rank scheduling or
  filesystem contention.

## Root cause

**Classification: D — Unsafe resource mutation/API misuse** (not classified
as a race condition — no evidence of cross-rank shared mutable state;
`state%cf` is per-rank-process-private, and the mechanism identified is a
concrete type/format mismatch, not a guessed timing race).

`ESMF_ConfigGetAttribute`'s `LOGICAL`-typed parse of a bare-integer `.rc`
token (`1`) is not reliable and intermittently fails
(`MAPL_Resource_SetValue.h:6`, `status=39`) in a manner sensitive to
per-process memory layout — manifesting as rank-varying, but not actually
racing, nondeterminism. `HELD_SUAREZ` deviated from this file's own
established INTEGER-proxy convention for exactly this class of `.rc` value
and was the only such resource in this job's config that reached the
parsing branch at all.

## Fix

File: `GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90`
(tree #3 — the real upstream GEOS-ESM checkout backing the standalone
fixture; NOT the CUDA dev repo's Prompt 43 diff, which was not touched).

One substantial change (of a 2-change budget):

1. Declaration block (~line 291): added `integer, save ::
   INT_HELD_SUAREZ = 0` alongside the existing `logical, save ::
   HELD_SUAREZ = .false.`, with a comment noting the convention match to
   `INT_ADIABATIC`/`INT_FV_OFF`/etc.
2. `FV_Setup` (~line 828): changed the `MAPL_GetResource` call to read
   `INT_HELD_SUAREZ` (default `INT_HELD_SUAREZ`) instead of `HELD_SUAREZ`
   directly, then added:
   ```fortran
   if (INT_HELD_SUAREZ == 0) then
      HELD_SUAREZ = .false.
   else
      HELD_SUAREZ = .true.
   endif
   ```

No Held-Suarez forcing formulation, dyn_core numerics, MPI topology, or
CUDA transport code touched. Prompt 43's 8-file uncommitted diff in the
`FVdycoreCubed_GridComp` dev repo (tree #1) is untouched and unaffected.

Rebuild: verified `ifort`/`icpx`/`mpiifort` matched `build/CMakeCache.txt`'s
recorded toolchain before building (avoiding Prompt 43's documented
toolchain-mismatch cascade); `cmake --build . --target
StandAlone_FV3_Dycore.x -j 8` recompiled only `FV_StateMod.F90` and relinked
cleanly (only pre-existing unrelated `warning #6843`). `strings` on the
fresh binary confirmed Prompt 43's CUDA-aware transport strings/symbols
(`RUN_CUDA_FV3_MPI_TRANSPORT`, `cuda_aware`, `host_staged`,
`MPIX_Query_cuda_support`, `I_MPI_OFFLOAD`) remain fully intact —
no regression to that separate work from this fixture-tree-only rebuild.

## C512 acceptance (Stage 9)

`p44_c512_accept` (job 58452391), unmodified original `AGCM.rc`
(`RUN_HELD_SUAREZ: 1`, `RUN_CUDA_FV3: 1`, host-staged transport default), 10
independent `timeout 90 mpirun -np 6` attempts against the fixed/rebuilt
binary:

- **10/10 PASS.** 0 occurrences of `status=39`/`FAIL at line`. All 10
  attempts show `rc=124` (expected `timeout` SIGTERM kill, not a model
  error) with exactly 12 `DYN Date:` progress lines each (120 total,
  perfectly uniform) — every attempt reached `dyn_core` and made genuine,
  consistent timestep progress. 0 NaN/Inf/Range_Warn/Error.

## Regression check (Stage 10)

`p44_c32_regress` (job 58452598), C32, HS-on and HS-off in sequence against
the fixed/rebuilt binary, 60s timeout each:

- **PASS for both.** 0 `status=39`/`FAIL at line`, 0 NaN/Range_Warn/error.
  Both `REGRESS_hson` and `REGRESS_hsoff` completed with natural `rc=0`
  (full clean program exit, not a timeout kill) — the fix does not disturb
  normal short-run initialization or completion in either Held-Suarez mode.

## Prompt 43 resume gate (Stage 11)

`p44_c512_cuda_aware_resume` (job 58452629), unmodified original `AGCM.rc`,
`RUN_CUDA_FV3_MPI_TRANSPORT=cuda_aware`, `I_MPI_OFFLOAD=1`, one `timeout 90
mpirun -np 6` attempt against the fixed/rebuilt binary.

**MAPL blocker: CONFIRMED CLEARED under cuda_aware.** 0 occurrences of
`status=39`/`FAIL at line`. The run passed cleanly through `FV_Setup`, MAPL
Generic Initialize, and `fv3_gpu_bridge_init` (`NOTE from PE 0:
fv3_gpu_bridge_init: rank 0 halo transport = cuda_aware`), then into
`fv_dynamics`/dyn_core execution — far beyond the resource-read site this
prompt targeted.

**A separate, pre-existing, Prompt-43-scope failure surfaced further
downstream**, at the CUDA-aware transport layer itself: `FATAL from PE 4:
CUDA dyn_core backend: fv3_gpu_dyn_core_step failed:` (no further detail
string), traceback through `fv3_gpu_dyncore_bridge.F90:693` (called from
line 356) <- `fv_dynamics.F90:748` <- `FV_StateMod.F90:2050` <-
`DynCore_GridCompMod.F90:4464`, `MPI_Abort(..., 1)` on rank 4 only (the
other 5 ranks show no FATAL of their own before the abort). This is inside
the CUDA-aware `MpiHaloTransport`/dyn_core dispatch path
(`cuda/dyn_core/*`, `cuda/interface/fv3_gpu_dyncore_bridge.F90`) — the exact
subsystem Prompt 43 flagged as "NOT CLEANLY CLOSED" at Stage 6 (an
intermittent 2-rank device-buffer SIGSEGV, not previously reconciled at
full C512 scale). It is unrelated to `FV_StateMod.F90`'s resource-read fix
(the failure occurs well after that code path completes successfully) and
is out of Prompt 44's scope to fix (would require touching CUDA transport
implementation code, explicitly disallowed here).

Host-staged transport (§ C512 acceptance above) is unaffected by this
finding and remains 10/10 clean.

## Stage 12 (diagnostic removal)

N/A. No source-level diagnostic instrumentation was added at any point in
this investigation — the discriminator (job 58452160) was a disposable
`.rc`-file text-format copy in a scratch experiment directory, not a code
change. Nothing to remove.

## Terminal classification

**C512 MAPL STARTUP RACE FIXED — RESUME PROMPT 43 AT TRANSPORT VALIDATION.**

The Prompt 44 objective (the pre-existing, nondeterministic
`MAPL_Resource_SetValue.h status=39` startup crash) is fixed and verified:
10/10 host-staged C512 acceptance PASS, clean C32 HS-on/HS-off regression,
and confirmed cleared under the cuda_aware transport path as well (reached
`fv3_gpu_dyn_core_step` before hitting an unrelated, separate,
Prompt-43-scope transport-layer failure). Prompt 43 should resume at Stage
6/7 — reconcile the CUDA-aware transport's own runtime failure
(`fv3_gpu_dyn_core_step failed`, rank 4, `fv3_gpu_dyncore_bridge.F90:693`)
before attempting the C512 correctness/A-B gate under cuda_aware. The
host-staged transport path is unaffected and available for correctness
work immediately.
