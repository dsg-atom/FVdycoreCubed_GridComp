# Prompt 68 — Root-Cause and Fix: C512 PT/DELZ Device-Memory OOM

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no escalation — mechanism and fix were unambiguous once
measured)

## 1. Objective

Prompt 67 found that the full-CUDA PT/DELZ dispatch OOMs at call 1 at the
authoritative C512 production density (6 MPI ranks sharing 1 A100),
reproducibly, independent of savepoints/run order/prior GPU state. This
prompt's task: find the exact allocation responsible, and fix it if a
narrow, non-numerical fix is possible.

## 2. Root Cause (Stages 1-2, 4-7) — CONFIRMED

`pt_delz_remap_kernel` (`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`)
declared 17 per-thread local arrays sized `[kMaxLevels]`/`[kMaxLevels+1]`
(`kMaxLevels = 128`) unconditionally, regardless of the actual runtime
`km` (10, at every grid this project runs). The CUDA driver responds to a
kernel with a large per-thread local/stack-array footprint by reserving a
**device-wide local-memory backing pool sized to the device's
theoretical maximum concurrently-resident-thread count** (108 SMs x 2048
threads/SM = 221,184 threads on an A100) — not the size of the actual
launched grid. This reservation happens lazily on the kernel's first
launch and persists for the CUDA context's lifetime (i.e., for the life
of the MPI rank's process).

**Direct empirical measurement** (Stage 2, job 58530194, node `warpa013`,
idle A100 immediately before launch: 40,334/40,960 MiB free):
`cudaMemGetInfo` bracketing a single, tiny (1024-thread) launch of the
unfixed kernel:

```
[P68_MEMPROBE] before_launch free=41847291904 total=42298834944 grid=(2,4) threads=(16,8)
[P68_MEMPROBE] after_launch  free=38636552192 total=42298834944 launch_status=0 (no error) delta_bytes=3210739712
```

A one-time, one-shot **3,210,739,712-byte (~3.06 GiB) drop in free device
memory**, caused by a 1024-thread launch — decisively confirming the
reservation is grid-size-independent and driven purely by the kernel's
compiled per-thread local-memory footprint. At the authoritative
6-ranks-share-1-A100 density, this reservation alone costs **~18.4 GiB**
(6x), on top of each rank's existing persistent dyncore state (WP-A/B/C/D
device-resident fields, halo-transport buffers, the existing
`WorkspaceId::vertical` scratch region) — sufficient by itself to exhaust
a 40 GiB A100, matching Prompt 67's observed OOM.

**Root-cause classification**: a structural coverage gap, not an
oversized/misconfigured allocation. `pt_delz_remap_kernel` never adopted
the device-workspace-reuse pattern its sibling `riem_solver_c_kernel`
(same file family, same `WP-C` ownership) already implements and prefers
by default. PT/DELZ's real per-column scratch need (17 slots x (km+1) =
187 scalars at km=10) is far smaller than the persistent
`WorkspaceId::vertical` slice already sized for Riem_Solver3's larger
need (422 scalars/column at km=10) — reuse requires **zero** additional
device bytes.

## 3. Fix (Stage 8) — IMPLEMENTED, narrow, mechanical

Mirrored `riem_solver_c_kernel`'s existing `template<bool
UseLocalScratch>` dual-scratch-path pattern exactly:

- `cuda/dyn_core/vertical/fv3_pt_delz_remap.cuh`: added
  `pt_delz_remap_scratch_slots=17`,
  `pt_delz_remap_scalars_per_column(km)`, `pt_delz_remap_columns(...)`,
  `pt_delz_remap_workspace_bytes(...)` helpers (same shape as
  `fv3_riem_solver_c.cuh`'s equivalents), and added
  `PtDelzRemapArgs::{workspace, workspace_scalars_per_column,
  workspace_bytes}` fields (default `nullptr`/`0`/`0` — no effect unless
  populated).
- `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`:
  - `pt_delz_remap_kernel` converted to
    `template <bool UseLocalScratch> __global__ void pt_delz_remap_kernel(...)`.
    All 17 local arrays replaced by pointers into a single scratch
    buffer (`sc`); the `<true>` instantiation still uses a per-thread
    `local_arena[kLocalArenaScalars]` (unchanged behavior, same 128-level
    cap); the `<false>` instantiation declares only a `[1]`-element dummy
    array and addresses `a.workspace` instead — eliminating the large
    local-memory footprint for that instantiation at compile time (each
    boolean instantiation is a physically distinct compiled kernel).
  - `launch_pt_delz_remap`: prefers the workspace path
    (`use_workspace = a.workspace != nullptr`) when a sufficiently large
    workspace is supplied; falls back to the local-scratch path
    (km<=128 enforced, unchanged) only when no workspace is supplied.
  - `launch_pt_delz_remap_ctx`: requests `workspace_slice(*x.workspace,
    WorkspaceId::vertical)` and populates `a.workspace`/
    `workspace_scalars_per_column`/`workspace_bytes` when the slice is
    large enough — exactly as `launch_riem_solver_c_ctx` already does.
    Safe by construction: every consumer of `WorkspaceId::vertical`
    (including this one) is required to enqueue on `StreamId::compute`,
    so GPU stream ordering, not any new synchronization, guarantees a
    prior consumer has finished before this kernel starts.

**Numerical semantics changed: NO.** Both code paths execute
byte-identical per-thread arithmetic; only the scratch backing-store
address differs (local stack vs. device workspace).

## 4. Correctness Validation (unit level, GPU-executed)

Extended `fv3_wp_c_pt_delz_remap_prod_shape_smoke` (6 C512-shaped
fixture cases) to re-run every case through the new workspace path and
assert **bitwise-identical** output against the unchanged local-scratch
path. Job 58530315 (node `warpa010`): all 6 cases pass both the original
fixture-tolerance check and the new bitwise self-consistency check
(`workspace-path bitwise-vs-local-scratch OK=1`, all 6). This is the
correctness oracle expected from a scratch-storage-only change.

## 5. Stage 9 — Production-Density Memory Validation: PASS

Rebuilt `libfv3_wp_c_vertical.a`
(`build_cuda_r8_intelmpi`, via `cuda_intelmpi_build.sh`) and the
production `StandAlone_FV3_Dycore.x`
(`GEOSfvdycore_fortran_perf/build`, sha256
`17a60c1941ac8bbb6f78cde79b19912577e9d8f31656ae024c30467aa1a7a829`,
verified deployed and actually invoked via the job's own `***** USING
****` log line, per the project's stale-binary lesson).

Job **58530484**, node `warpa003`, solo run (no preceding Arm-A run in
the allocation — same solo/fresh-node protocol as Prompt 67's order-
artifact check), authoritative C512 production config: R8, 6 MPI
ranks/1 A100, MPS ON, host-staged transport, fusion ON, Graph OFF,
Held-Suarez ON, `COLDSTART=1`, no savepoints, full CUDA PT/DELZ dispatch
(no `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN`/`_DELZ_ONLY` overrides) — i.e.
exactly Prompt 67's Arm B / solo-diagnostic config, rerun with the fix.

**Result: 30/30 `DYN Date` lines completed. RC=0. No `FATAL`, no `out of
memory`, no `Abort` anywhere in the log.** This is a decisive reversal of
Prompt 67's 0/3 (and 0/1 solo) result at the identical density and
config.

GPU memory checkpoints (`nvidia-smi`, background-polled every 5s during
the run):

| Checkpoint | Used (MiB) | Free (MiB) |
|---|---|---|
| idle, pre-MPS | 4 | 40,334 |
| post-MPS-start, pre-run | 4 | 40,334 |
| steady-state during run (poll_10 onward) | 39,473 | 865 |
| post-run (context torn down) | 44 | 40,295 |

Peak steady-state usage across all 6 ranks sharing the A100 is
**39,473/40,960 MiB (96.4%), leaving 865 MiB headroom** — tight, but
successful, and consistent with the ~18.4 GiB reservation having been
eliminated (Prompt 67's unfixed run had no headroom at all and OOM'd
immediately). No sanitizer run was performed against this full
production-scale job (impractical at this scale and outside Prompt 67's
own precedent, which likewise never ran full-scale `compute-sanitizer`);
kernel-level correctness is separately established by §4's bitwise proof.

## 6. Stage 10 — Numerical Smoke Check: PASS

Short runs (not a re-run of Prompt 66's 96-call growth-curve policy
experiment), same fixed binary:

- **C32, 6 calls** (job 58530703, node `warpa003`): RC=0, 6/6 `DYN Date`
  lines, no `NaN`/`FATAL`/`Abort`.
- **C512, 30 calls** (job 58530484, above): RC=0, 30/30 `DYN Date` lines,
  zero `NaN`/`Inf` occurrences, zero new `range_warn` lines beyond the
  existing documented benign ones.

Combined with §4's bitwise kernel-level proof, there is no evidence of
a new divergence mechanism introduced by this fix.

## 7. Bonus, Unvalidated Observation (NOT a Stage-11 performance A/B)

The now-successful Arm-B run's own `DYN_CORE` Block-table entry for
30 calls: **25.6188 s**. Prompt 67's Arm-A (PT/DELZ forced through
Fortran) same-node baseline: **25.5548–25.6032 s/30 calls** (3 reps).
These are strikingly close — but this is a **single, unpaired Arm-B
data point** from a different job/node/day than Prompt 67's Arm-A
numbers, not a controlled same-allocation A/B (`CURRENT_STATE.md` §8's
own node-variance lesson). **Per this prompt's explicit scope, the full
paired performance A/B is deferred to Prompt 69** — this number is
recorded only as a promising preliminary signal, not a result.

## 8. Conclusion

**Root cause: CONFIRMED** (CUDA implicit local-memory reservation,
grid-size-independent, ~3.06 GiB/rank, measured directly).
**Fix: IMPLEMENTED, GPU EXECUTED, NUMERICALLY VALIDATED** (bitwise
self-consistency at unit level; clean C32+C512 smoke runs).
**Production-density memory validation: PASS** (C512 Arm B, 6-ranks/
1-A100, 30/30 calls, no OOM, 865 MiB headroom at peak).

Prompt 67's profiling block is **RESOLVED**. The full-CUDA PT/DELZ
performance A/B (Prompt 67 Stages 3-13) can now proceed — recommended as
**Prompt 69**, not attempted here per this prompt's explicit scope.

No commit or push was made in this prompt (uncommitted diffs to
`cuda/dyn_core/vertical/fv3_pt_delz_remap.{cu,cuh}` and
`fv3_pt_delz_remap_prod_shape_smoke.cu`, per standing no-auto-commit
policy).

**SENTINEL: PROMPT_68_FIXED_AND_VALIDATED**
