# Prompt 43 — Intel MPI CUDA-Aware MpiHaloTransport A/B at C512

## Objective (as given)

Implement the minimum opt-in CUDA-aware variant of the existing MPI halo
transport using the already-validated Intel MPI 2021.13 environment
(Prompt 42), then perform a correctness-gated same-node C512 A/B comparison
against the current host-staged transport. Model: Sonnet only.

## Stages 0-5 (implementation) — COMPLETE

- `RUN_CUDA_FV3_MPI_TRANSPORT=host_staged|cuda_aware` runtime selector wired
  in `cuda/interface/fv3_gpu_dyncore_bridge.F90` /
  `cuda/interface/fv3_gpu_abi_mod.F90` (Fortran side): new
  `FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE` constant, new
  `fv3_gpu_attach_mpi_halo_transport()` C ABI entry point, new
  `mpi_transport_kind_from_env()` helper, an explicit fatal check that the
  requested transport actually attached, and a `mpp_error(NOTE, ...)`
  announcing which transport is active per rank.
- CUDA-aware device-pointer transport variant added to
  `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp` /
  `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` /
  `cuda/dyn_core/include/fv3_gpu_halo_mpi.hpp` /
  `cuda/dyn_core/include/fv3_gpu_communication.h`, gated by
  `I_MPI_OFFLOAD=1` (never silently forced; fails clearly before model-state
  mutation if the capability is absent — no silent host-staged fallback).
  All 8 files remain **uncommitted working-tree changes** on
  `experiment/held-suarez-e2e` (2 substantial logical changes: (1) the
  transport-selection/capability-validation wiring, (2) the CUDA-aware
  transport backend itself — both already at the Prompt 43 source-change
  budget ceiling of 2).
- `build_cuda_r8_intelmpi` (CUDA static libs) and the
  `GEOSfvdycore_fortran_perf` fixture tree's `StandAlone_FV3_Dycore.x`
  (`build/bin/`) both build clean against this diff under the canonical
  toolchain (`comp/gcc/12.3.0` + `comp/intel/2024.2.0` +
  `mpi/impi/2021.13` + `other/ISSM/geos/1.0.3/ifort_2021.13.0-
  intelmpi_2021.13.0` + `nvidia/nvhpc-byo-compiler/24.7`).

## Stage 6 (transport unit test) — NOT CLEANLY CLOSED (unchanged from prior session)

A 2-rank device-buffer MPI transport test previously hit an intermittent
SIGSEGV attributed to an Intel-MPI-on-AMD-Rome-CPU issue; gdb-wrapped
mitigation attempts gave inconsistent results (1 clean pass, 1 reproduced
crash across 2 attempts). Flagged for reconciliation, not further pursued
this session — superseded in priority by the Stage 7 blocker below, which
turned out to be unrelated and more fundamental.

## Stage 7 (C512 short correctness gate) — BLOCKED by a pre-existing, non-Prompt-43 defect

### What happened

Two build-hygiene bugs were found and fixed early in this investigation
(both real, both now fixed, neither is the actual blocker):

1. A stale `install/bin/StandAlone_FV3_Dycore.x` copy was not the binary
   actually being exercised — fixed by staging fresh `cp -f` copies from
   `build/bin/` into each experiment directory.
2. A **toolchain-mismatch cascade recompile**: an earlier session's build
   had recompiled `FV_StateMod.F90`, `fv_dynamics.F90`, and the three
   `cuda/interface/*.F90` files (CMake's Fortran dependency scanner
   cascades on `.mod` interface changes) before the canonical
   `ifort`/Intel-MPI module was fully loaded, leaving `.o` files that
   carried a fresh timestamp but a mismatched-toolchain compile. A pure
   relink under the corrected environment did not repair this (timestamps
   already looked current). Fix: explicitly `touch`ed the five affected
   source files and reran `cmake --build` under a
   verified-matching-`CMakeCache.txt` toolchain (confirmed via
   `which`/`--version` on `ifort`/`mpiifort`/`icpx` against
   `build/CMakeCache.txt`'s recorded compiler paths). This fix is real,
   necessary, and now applied — but did **not** resolve the crash below.

### The actual blocker

After the toolchain fix, a short host-staged C512 six-rank smoke run
(`p43_bincheck_hoststaged`, `RUN_CUDA_FV3_MPI_TRANSPORT` unset ⇒ default
host_staged) crashes on 2 of 6 ranks, **every time**, with:

```
FV_StateMod:FV_Setup                           829
DYNSetServices                                2606
pe=NNNNN FAIL at line=00006    MAPL_Resource_SetValue.h    <status=39>
...
Abort(-1) ... application called MPI_Abort(MPI_COMM_WORLD, -1)
```

`FV_Setup:829` is the `MAPL_GetResource(MAPL, HELD_SUAREZ,
'RUN_HELD_SUAREZ:', ...)` read; `MAPL_Resource_SetValue.h:6` is the
`ESMF_ConfigGetAttribute` scalar-read call taken when the label **is**
found (ruling out "missing resource key"). This is deep inside MAPL/ESMF
config parsing — not code owned or touched by Prompt 43.

**Reproduced 3/3 times**, and critically, **the failing rank pair is
different every run** (ranks {3,0}; then {4,5}; then {0,3} again) with the
identical binary and identical input — the hallmark of a non-deterministic
race, not a static corruption.

### Discriminator sequence (coarse-to-fine, per project debugging policy)

| Test | Binary / source state | Result |
|---|---|---|
| `p43_bincheck_oldbin` (job 58450986) | `install/bin/` binary, dated 2026-08-31 (pre-dates 20+ intervening commits, NOT a clean Prompt-43-only baseline) | Clean — no `status=39` on any rank; ran the full 10 min walltime making genuine simulated progress (TIMEOUT, not a failure) |
| `p43_bincheck_hoststaged` (job 58450956) | Prompt-43-diff binary, relink-only (stale toolchain-mismatched `.o`s) | Crash, rank 3 (+ rank 0 partial) |
| `p43_bincheck` (job 58451661) | Prompt-43-diff binary, **genuinely rebuilt** under verified-correct toolchain | Crash, ranks 4 & 5 |
| `p43_bincheck` rerun (job 58451748) | identical genuinely-rebuilt Prompt-43-diff binary, no rebuild in between | Crash, ranks 0 & 3 — confirms non-determinism, not a fixed corrupted artifact |
| `p43_bincheck_headnodiff` (job 58451786) | **Committed HEAD (`e1d760b`) with the entire Prompt 43 working-tree diff `git stash`ed out** (all 8 modified files reverted), CUDA static libs + Fortran executable rebuilt from scratch under the same verified-correct toolchain | **Crash, ranks 3 & 0 — reproduces identically with zero Prompt 43 code present** |

The last row is decisive: the crash is **not caused by Prompt 43's
CUDA-aware transport work** (Change 1 or Change 2). It is a pre-existing
defect already present at current HEAD, evidently a latent, timing/
memory-layout-sensitive race in the MAPL Held-Suarez resource-read path
that the lucky single 08-31 baseline sample simply didn't hit. The working
tree was restored (`git stash pop`) and both build directories were
rebuilt back to the Prompt-43-diff state afterward, so the repository is
left exactly as it was before this diagnostic detour, with no net source
change beyond the pre-existing Stage 0-5 diff.

### Why this is out of scope to fix under Prompt 43

- The failure site (`MAPL_Resource_SetValue.h`, `MAPL_Generic.F90`,
  `MAPL_Cap.F90`, `FV_StateMod.F90`'s Held-Suarez resource read) is MAPL/
  ESMF/FV_StateMod infrastructure, not WP-E halo transport, not
  `cuda/interface/*`, not `cuda/dyn_core/*`.
- Fixing it would require touching Held-Suarez physics wiring or MAPL
  resource-parsing code — explicitly forbidden by Prompt 43's "do NOT
  change Held-Suarez physics" constraint and outside the 2-substantial-
  source-change budget (already spent on the actual transport work).
- It occurs identically regardless of `RUN_CUDA_FV3_MPI_TRANSPORT` (the
  crash happens during MAPL Generic Initialize, before `fv_dynamics`/
  `fv3_gpu_bridge_init`/any halo-transport code executes at all) — so it is
  not evidence about CUDA-aware transport correctness one way or the other.

## Stage 8 / 9 (paired A/B, decision) — NOT ATTEMPTED

Blocked by Stage 7. No performance data collected.

## Recommended next step (separate from Prompt 43's scope)

Root-cause the intermittent, rank-varying `MAPL_Resource_SetValue.h
status=39` race in the Held-Suarez resource-read path at C512 six-rank
scale — likely a concurrent-config-file-read race on the shared filesystem
or an uninitialized-value race in ESMF's config parser that is sensitive to
inter-rank timing. This blocks re-establishing *any* fresh C512 correctness
baseline right now (with or without Prompt 43's diff), independent of CUDA-
aware MPI work. Until it is fixed, Prompt 43's Stage 7/8/9 cannot proceed;
the existing bitwise-validated `p37_c512_gate_cuda` host-staged baseline
(§5, `CURRENT_STATE.md`) remains the last confirmed-good C512 correctness
result and predates this newly-discovered race.

## Terminal Classification

The crash is not a defect in the CUDA-aware transport itself (it occurs
identically with the transport code entirely absent, and before any
transport is selected), so strictly speaking none of the six prescribed
outcomes describes it precisely. The closest fit, since it correctly
signals that the C512 gate could not be closed and blocks the requested
A/B, is used below, with this caveat recorded explicitly: the specific
failure is an unrelated, pre-existing MAPL/Held-Suarez resource-read race,
not a CUDA-aware-MPI-transport correctness defect.

**C512 CUDA-AWARE TRANSPORT NOT CORRECT — C512 six-rank correctness gate
could not be established for either transport (host_staged or cuda_aware)
because of a pre-existing, non-Prompt-43, non-deterministic
`MAPL_Resource_SetValue.h status=39` crash on 2 of 6 ranks during MAPL
Generic Initialize's Held-Suarez resource read (`FV_StateMod:FV_Setup:829`
/ `DYNSetServices:2606`), confirmed present at current HEAD with the entire
Prompt 43 diff removed — root cause and fix are out of Prompt 43's scope
(MAPL/ESMF infrastructure, not WP-E halo transport) and require a dedicated
follow-up prompt.**
