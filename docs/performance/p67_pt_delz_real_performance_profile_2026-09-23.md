# Prompt 67 — Real Performance Impact of the CUDA PT/DELZ Port: PROFILE BLOCKED

> **Update (Prompt 68, 2026-09-23): UNBLOCKED.** The OOM root-caused to a
> CUDA implicit local-memory reservation (~3.06 GiB/rank, grid-size-
> independent) triggered by `pt_delz_remap_kernel`'s unconditional
> 128-level-capped per-thread local arrays. Fixed by adopting the same
> device-workspace-reuse pattern `riem_solver_c_kernel` already uses
> (zero additional device bytes; bitwise-validated at unit level).
> Production-density re-validation (job 58530484, C512, 6-ranks/1-A100,
> full CUDA PT/DELZ, solo): 30/30 calls complete, no OOM, 865 MiB peak
> headroom. See `p68_pt_delz_c512_memory_footprint_2026-09-23.md` for
> full detail. The full paired performance A/B this document's Stages
> 3-13 could not run is deferred to Prompt 69 — not attempted in
> Prompt 68.

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no escalation)

## 1. Objective

Prompt 66 conditionally accepted the full-CUDA PT/DELZ port's intrinsic
`exp`/`log` numerical residual (round-off/ULP-level, bounded, sub-linear
growth at C32). Prompt 67's task was to measure the port's actual
performance impact at authoritative C512 production density via a
same-node A/B:

- **Arm A** — CUDA dyn_core, PT/DELZ forced through authoritative Fortran
  `map_scalar` (`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1`).
- **Arm B** — CUDA dyn_core, full CUDA PT/DELZ dispatch (no overrides;
  the path validated in Prompts 61-66).
- `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY` (Prompt 65's rejected hybrid) unset in
  both arms.

Measurement-only: no source changes, no numerical changes, no commit.

## 2. Stage 0 — Provenance Preflight

Confirmed via job 58527803's own captured preflight output:

- `FVdycoreCubed_GridComp`: branch `optimization/cuda-fv3-performance`,
  clean working tree relative to the known uncommitted diffs (§10 of
  `CURRENT_STATE.md`).
- `GEOSfvdycore_fortran_perf` build binary
  (`build/bin/StandAlone_FV3_Dycore.x`): sha256
  `30422c2cfcd3ccc6978c5c41b60a0bd93bb5f1d0f4d56f1be67ec78f198d4137`,
  29,370,560 bytes, mtime 2026-09-23 — the current fully-fixed binary
  containing all Prompt 57-66 PT/DELZ correctness fixes. CUDA and
  `pt_delz` symbols present (`nm -D` counts > 0). MPI linkage confirmed
  Intel MPI via `ldd`.
- **Stale-binary pitfall caught before submission**: `fv3.j`'s `GEOSBIN`
  falls back to `install/bin/StandAlone_FV3_Dycore.x` (stale, sha256
  `bea8f4739478ff873b191b5d908ab76a108b8a373394d4fca1ef02590871ece8`,
  Aug 31) unless a per-directory override binary exists. Fixed: the
  verified current binary was copied into all 6
  `experiments/p67_pair{1,2,3}_{A,B}/` directories, sha256-verified, and
  the sbatch script itself re-verifies the sha256 before every sub-run
  (`P67_RESULT: BINARY_SHA_MISMATCH` abort path, never triggered).

## 3. Stage 1 — Config Confirmation

Authoritative C512 production performance config (per Prompt 48's
template, cloned unmodified except job-name/`EXPDIR`): R8, `npz=10`,
`bdt=36`, `n_split=3`, `k_split=1`, 6 MPI ranks, 1 A100, MPS ON,
host-staged MPI transport, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, Graph OFF, Held-Suarez ON,
`COLDSTART=1`, `FV3_SAVEPOINT_ENABLE=0` (no savepoints — the production,
non-diagnostic path). Confirmed in each Arm-A log's own config header
(`npx: 513 npy: 513 npz: 10 ntiles: 6 hydrostatic: F gmao_top_bc: T
gmao_bot_bc: T`).

## 4. Stage 2 — Same-Node A/B Result

Job **58527803**, node `warpa011`, one Slurm allocation, one A100,
alternating A1 B1 A2 B2 A3 B3, ~10 minutes wall time.

### Arm A: 3/3 successful, 30-call `DYN_CORE` Block-table timings

| Pair | TOTAL (s) | COMM_TOTAL (s) | FV_DYN_LOOP (s) | DYN_CORE (s) | TRACER_2D (s) | REMAPPING (s) |
|---|---|---|---|---|---|---|
| 1 | 148.2197 | 2.0054 | 64.1294 | 25.5548 | 7.9162 | 31.5024 |
| 2 | 147.6154 | 2.0048 | 64.1509 | 25.5517 | 7.9875 | 31.5001 |
| 3 | 148.5902 | 2.0024 | 64.1620 | 25.6032 | 7.9631 | 31.4540 |

All 3 pairs completed 30/30 `DYN Date` progress lines (full completion),
tightly consistent run-to-run (as expected for a same-node paired
design). `DYN_CORE`/30calls ≈ 852ms/call; `FV_DYN_LOOP`/30calls ≈
2138ms/call.

### Arm B: 0/3 successful — reproducible OOM at call 1

All 3 Arm-B sub-runs (pairs 1, 2, 3) crashed identically:

```
FATAL from PE <n>: CUDA dyn_core backend: fv3_gpu_pt_delz_remap failed:
out of memory
```

with a full ESMF/MAPL `MPI_Abort` stack trace through
`fv3_gpu_dyncore_bridge.F90:767` -> `fv3_gpu_dyncore_bridge.F90:445` ->
`fv_mapz.F90:685` (inside `Lagrangian_to_Eulerian`) ->
`fv_dynamics.F90:952` -> `FV_StateMod.F90:2050` ->
`DynCore_GridCompMod.F90:4464`. PE that reported the abort varied (PE 5,
then PE 2 in later pairs) — consistent with all 6 ranks racing to
allocate and one losing first. Logs are 371 lines each, crash occurring
immediately after the config header — i.e. **at call 1**, before any
timing data is produced. No `DYN_CORE`/Block-table data exists for Arm B.

**Note on exit-code reliability**: `sacct` reported all 6 sub-steps
`COMPLETED 0:0`. This is misleading — traced to a csh `$status`-after-
`tee`-pipe artifact (`$status` after `... |& tee file.log` reflects
`tee`'s exit, not the model's) in the inherited `fv3.j` template. Actual
outcome was confirmed by grepping log content (`FATAL`, `Abort`, and
counting `DYN Date` lines), not by trusting the reported return code —
consistent with `CURRENT_STATE.md` §8's existing lesson about log
content being the only reliable evidence of what a job actually did.

## 5. Order/Accumulation-Artifact Check (Solo Diagnostic)

To rule out "Arm B only fails because it runs after an Arm-A run in the
same allocation" or "6 ranks' worth of prior CUDA state accumulates
across sub-runs," a dedicated diagnostic reran Arm B **alone**, as the
first and only job in a fresh allocation, on a different node:

Job **58528044**, node `warpa010`. `nvidia-smi` immediately before launch:

```
memory.total [MiB], memory.used [MiB], memory.free [MiB]
40960, 4, 40334
```

i.e. an effectively idle A100 (only 4 MiB used — driver overhead).
Result: **identical FATAL/OOM crash at call 1**, same stack trace.

**Conclusion: the OOM is an inherent capacity limit of the full-CUDA
PT/DELZ dispatch at 6-ranks/1-A100 density, not an artifact of run
order, savepoint diagnostics, or prior-job GPU-memory residue.** This
falsifies the existing `CURRENT_STATE.md` §7 claim (pre-Prompt-67) that
the Prompt-66-observed OOM was "isolated to the savepoint dump's
device/host staging buffers, not the production dyn_core path itself" —
that claim has now been corrected in place.

## 6. Root-Cause Hypothesis (Bounded, Not Instrumented)

Source search (bounded, per cost-control mandate — no new
instrumentation, no `cuda-memcheck`/`compute-sanitizer` deep-dive, no
source changes):

- `grep cudaMalloc\|cudaFree` across `fv3_pt_delz_remap.cu`/`.cuh`: only
  a trivial 3-element debug allocation (`p57_dbg_dev`, lines 466/524) —
  not a real workspace allocation site in this file.
- `grep -r "out of memory"\|"pt_delz_remap failed"` across all of
  `cuda/` and the Fortran bridge files: **zero matches**. The FATAL
  message is therefore a generic `cudaGetErrorString(cudaErrorMemory
  Allocation)` string relayed through a shared CUDA-error-to-Fortran
  translation path, not a custom-authored message — i.e. a genuine
  `cudaMalloc`-family failure somewhere in the persistent PT/DELZ
  workspace or in the broader CUDA dyncore context at the point the
  PT/DELZ stage's allocation is attempted.

**Working hypothesis**: with 6 MPI ranks sharing 1 A100 (40GB), each
rank's persistent CUDA dyncore state (WP-A/B/C/D device-resident
buffers, halo-transport buffers, existing dyn_core workspace) already
consumes a large fraction of device memory once all 6 ranks' contexts
are initialized at C512 resolution. The PT/DELZ CUDA workspace
allocation is what tips an already near-full GPU over the edge. This is
consistent with the project's own prior, independently-established
finding that CUDA-aware `MpiHaloTransport` is "Blocked at the original
6-ranks/1-A100 density by device memory capacity"
(`CURRENT_STATE.md` §5/§7) — i.e. this rank/GPU density is already
known to be memory-constrained for a different feature.

This hypothesis is **not proven** — it was not investigated further
because doing so (locating the exact allocation call site, measuring
per-rank footprint with `cuda-memcheck`/`nvidia-smi` polling mid-run,
sizing a fix) requires source-level investigation and/or new
instrumentation, both out of scope for Prompt 67's measurement-only
mandate. The bounded-debugging stop condition (a clear, reproducible,
decisive result) was met at Stage 2/the solo diagnostic.

## 7. Stages 3-13: Not Executable

Stages 3 (classification), 4 (Nsight Systems A/B), 5 (`fv_mapz` CPU
before/after), 6 (PT/DELZ kernel cost), 7 (transfer/sync deltas), 8
(critical-path reconciliation), 9 (remaining-hotspot ranking), 10
(keep/revert policy), 11 (2x-roadmap feasibility), 12/13 (next-target
selection informed by measured data) all require a completed Arm-B
profile. **None could be executed** — there is no Arm-B `DYN_CORE`
timing, no Arm-B Nsight capture, and no Arm-B kernel-cost data, because
Arm B never completes call 1. No Nsight Systems capture, kernel-cost
measurement, or transfer/sync analysis was attempted, since doing so
against a process that OOMs before producing any dynamics output would
not yield meaningful profiling data.

## 8. Numerical-Acceptance Status: Unaffected

Prompt 66's CONDITIONAL ACCEPT of the PT/DELZ numerical residual is
**unaffected** by this finding. The OOM is a device-memory-capacity
defect (a resource-exhaustion failure), not a numerical-divergence
finding — it says nothing about whether the port's arithmetic is
correct when it runs. No numerical evidence from Prompt 66 is
contradicted or superseded here; only the previously-documented
*cause* of the OOM (attributed there to savepoint-only diagnostic
buffers) has been corrected.

## 9. Conclusion

**PROFILE BLOCKED.** The full-CUDA PT/DELZ dispatch cannot complete a
single call at the authoritative C512 production rank/GPU density (6
ranks / 1 A100), independent of savepoints, run order, or prior GPU
state. This is a genuine device-memory-footprint defect in the CUDA
PT/DELZ port — more severe than previously documented (previously
believed savepoint-diagnostics-only, call-2-onward; now known to be
call-1, savepoint-independent). Fixing it (locating and reducing the
PT/DELZ workspace's device-memory footprint, or verifying total 6-rank
persistent-state footprint against the 40GB A100 budget) is a
prerequisite for any future performance measurement of this path at
production density, and is recommended as the next task (see
`CURRENT_STATE.md` §9 item 0).

No source changes, numerical changes, or commits were made in this
prompt.

> **Update (Prompt 68): OOM ROOT-CAUSED AND FIXED.** The device-memory
> footprint was a CUDA implicit local-memory reservation
> (~3.06 GiB/rank, grid-size-independent) in `pt_delz_remap_kernel`'s
> unconditional 128-level-capped per-thread local arrays — fixed via
> device-workspace reuse (`WorkspaceId::vertical`, zero extra bytes),
> mirroring `riem_solver_c_kernel`'s existing pattern. C512 production-
> density re-run: 30/30 calls, no OOM, 865 MiB peak headroom. See
> `p68_pt_delz_c512_memory_footprint_2026-09-23.md`.

> **Update (Prompt 69): PROFILE UNBLOCKED, PAIRED PERFORMANCE MEASURED
> — POSITIVE.** With Prompt 68's fix, the deferred same-node C512 A/B
> (3 pairs, 30 calls) completed cleanly. Result: PT/DELZ CUDA offload is
> **33.7% faster at REMAPPING, 16.6% faster at FV_DYN_LOOP, 7.3% faster
> at TOTAL** wall-clock (the `DYN_CORE` timer itself is unaffected, as
> it never contained PT/DELZ — a timer-hierarchy correction identified
> in Prompt 69). Decision: **KEEP**. Next target: REMAP_UV (wind/
> momentum vertical remap). Full data, GPU-cost/transfer-cost breakdown,
> and reconciliation: `p69_pt_delz_paired_performance_2026-09-23.md`.
