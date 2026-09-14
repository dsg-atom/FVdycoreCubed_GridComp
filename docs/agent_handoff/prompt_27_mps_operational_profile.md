# Prompt 27 — MPS Operationalization and Post-MPS Bottleneck Profile

_Date: 2026-09-14. Branch `cuda/gpu-bringup`, HEAD `6817f8d`. Scope: make CUDA
MPS an opt-in operational mode for the standalone CUDA FV3 job workflow, take
one short `nsys` trace under it, and reclassify the bottleneck. No numerical,
topology, or rank/tile change. No CUDA-aware MPI, kernel tuning, CUDA Graph,
or multi-GPU work performed._

Entry point: `prompt_26_rank_gpu_architecture.md`, which found six MPI ranks
time-slicing six CUDA contexts on one A100 was the dominant cost, and that
CUDA MPS recovers 3.92× on `DYN_CORE` with zero source change.

---

## 1. MPS Operational Integration

New file: `<fortran_perf>/experiments/fv3_mps_lifecycle.sh` — two shell
functions, sourced by a job script, not touching any numerical source:

- `fv3_mps_start` — exports job-scoped `CUDA_MPS_PIPE_DIRECTORY` /
  `CUDA_MPS_LOG_DIRECTORY` under `/tmp/fv3mps_${USER}_${SLURM_JOB_ID}/`,
  starts `nvidia-cuda-mps-control -d`, and verifies the server actually came
  up via `get_server_list` before returning success. All 6 MPI ranks launched
  from the same job step inherit the exported variables automatically — no
  per-rank setup needed (confirmed working in Prompt 26 job 58380090 and again
  here).
- `fv3_mps_stop` — sends `quit` to the daemon and unsets the env vars.
- Default (script not sourced, or sourced but never called) is **MPS OFF**,
  identical to every prior job in this project.
- Job-scoped, not global: the pipe/log directories are keyed by `$USER` and
  `$SLURM_JOB_ID`, so this can never attach to another user's daemon (one
  already runs on these nodes, owned by a different account — verified in
  Prompt 26 and re-verified here via a pre-start `get_server_list` check) and
  never collides with a concurrent job of this project.
- No change to Slurm resource requests, GPU compute mode, or any
  administrator-level setting. Cleanup is explicit (`fv3_mps_stop` in-script)
  and additionally bounded by normal Slurm job teardown if the script were to
  fail before reaching it — `/tmp` is job/node-local, so an unquit daemon
  would die with the node's Slurm epilogue in any case.

Applied in new job script `<fortran_perf>/experiments/p27_mps_operational.sbatch`
(`gpu:1`, 6 ranks, ≤30 min), which sources the lifecycle helpers once and runs
both the operational smoke and the `nsys` trace inside the same MPS session
(§2, §3).

## 2. Scheduler Safety

- Runs entirely inside the allocated node/GPU (`--gres=gpu:1`); no daemon is
  started outside the job's own allocation.
- `PRE_START_CHECK` before `fv3_mps_start` confirmed no pre-existing daemon
  was visible to this account (`"Cannot find MPS control daemon process"`),
  so `fv3_mps_start` could not have attached to another user's server.
- `POST_STOP_CHECK` after `fv3_mps_stop` confirmed the same message returned,
  i.e. no daemon left running after the job's teardown call.
- No global GPU compute-mode change (Discover `gpu_a100` is already
  `compute_mode=Default`, which is what MPS requires — nothing was set here).
- No administrator action required; the daemon is a normal user-mode process
  started and stopped entirely inside this job's shell.

## 3. Operational Smoke — job 58380469

Reused the exact Prompt 24/26 job directory/config/methodology
(`p24_c32_r8_cuda`, c32/R8, `npz=10`, `n_split=3`/`k_split=1`, `bdt=1200s`,
100 `dyn_core` calls, `FV3_SAVEPOINT_ENABLE=0`), 1 rep, MPS on via the new
lifecycle helpers, same MPI/GPU topology as always (6 ranks, 1 tile/rank, 1
GPU):

| Check | Result |
|---|---|
| `fv3_mps_start` | succeeded — server PID `70647` confirmed live via `get_server_list` |
| Rank count / GPU | 6 ranks, `rank 0 owns tile 1, device 0` (all six ranks resolve the same device, as in every prior run) |
| `RUN_RC` | 0 |
| `Range_Warn` | 0 |
| `NaN` | 0 |
| `DYN_CORE` | 2.6965 s / 100 calls = **26.965 ms/call** |
| MPS server still up during run | yes (`SMOKE_MPS_SERVER_DURING: 70647`) |
| `fv3_mps_stop` | succeeded, server gone (`POST_STOP_CHECK: Cannot find MPS control daemon process`) |
| MPS lifecycle errors | none |

Delp-positivity / exit-state correctness under MPS is **not re-derived here**
— it was already established bitwise in Prompt 26 §4.1 (job 58380133 vs. the
58380203 MPS-off control): all 1230 non-dce-entry savepoints identical, and
the 66 dce-entry files that differ reproduce identically without MPS. Re-running
that campaign was out of scope for this task ("no new correctness campaign").

**26.965 ms/call reproduces Prompt 26's 27.075 ms/call to within 0.4%.**
Prompt 26's expected high-20s-ms range is reproduced. **YES.**

## 4. One Short `nsys` Trace Under MPS — job 58380469, same session

Reused the existing `p24_c32_r8_cuda_nsys` job directory/`fv3.j`/
`nsys_wrap.sh` unchanged — the same mechanism Prompt 24 used for its
diagnostic trace (job 58295418): `nsys profile --trace=cuda,osrt --sample=none`
wraps rank 0 only. 10 `dyn_core` calls (`JOB_SGMT`-bounded), started inside
the same MPS session as §3 (no restart between the two).

| Check | Result |
|---|---|
| `RUN_RC` | 0 |
| `Range_Warn` / `NaN` | 0 / 0 |
| `DYN_CORE` (this traced run) | 0.3466 s / 10 calls = 34.66 ms/call (nsys instrumentation overhead inflates this above the untraced §3 figure — expected, and it is used below only as the internal denominator for this trace's own bucket percentages, exactly as Prompt 24 did for its own trace) |
| `nsys stats` (`cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `osrt_sum`) | all four produced without error |

## 5. Post-MPS Profile Buckets

All percentages below are each category's total time over the 10-call trace,
divided by 10, expressed as a fraction of this trace's own `DYN_CORE` ms/call
(34.66 ms) — the same convention Prompt 24 used, so the two are comparable.

| Category | ms/call | % of DYN_CORE |
|---|---|---|
| **A. CUDA kernel execution** (171 distinct kernels, sum) | 12.81 | **37%** |
| **B. `cudaLaunchKernel` (launch/API overhead)** | 17.73 | **51%** |
| **C. CUDA synchronization** (`cudaEventSynchronize`, 250 calls) | 9.67 | **28%** |
| **D. Memory copy** — device-side H2D+D2H (`cuda_gpu_mem_time_sum`) | 0.62 | 1.8% |
| — host-side `cudaMemcpyAsync` API overhead (issuing the copies) | 3.79 | 10.9% |
| **F. MPI/halo host wait** | not separable from this single-rank `cuda,osrt` trace — see note below | — |
| **H. Kernel launches per `dyn_core`** | 29550 launches / 10 calls = **2955/call** | — |

These overlap (kernels execute asynchronously while the host issues more
launches), so they legitimately sum above 100% — this is the same convention
Prompt 24 used and is not a normalization error.

Average kernel busy time is only ~4.3 µs (128.1 ms / 29,550 launches), close
to the ~6.0 µs average time-in-`cudaLaunchKernel` itself — i.e. per-launch
host overhead is comparable to or larger than the GPU work each launch does.
No single kernel family dominates: the largest (`copy_corners_kernel`) is only
5.6% of total kernel time; the next six are all 2–4%. This is a many-tiny-
kernels regime, not a few-expensive-kernels regime.

**Note on F (MPI/halo host wait):** the `osrt_sum` report's largest entry is
`poll` (70.2%, 3.41 s total), but this trace wraps the *entire* rank-0 process
(model init, I/O, `dyn_core`, remap, finalize), not just the `dyn_core` window,
so this figure cannot be safely attributed to halo-exchange waiting inside
`dyn_core` specifically without an `--trace=mpi`-instrumented re-run, which
this task's "one short trace" / "no giant kernel table" scope does not call
for. It is flagged here as unattributed rather than assumed.

## 6. Comparison Against the Pre-MPS Profile (Prompt 24, job 58295418)

| Bucket | Pre-MPS (Prompt 24) | Post-MPS (this trace) |
|---|---|---|
| Host synchronization (`cudaEventSynchronize`) | ~59.27 ms/call, **~56%** | ~9.67 ms/call, **~28%** |
| Launch/orchestration (`cudaLaunchKernel`) | ~17.33 ms/call, ~16% | ~17.73 ms/call, **~51%** |
| Kernel compute (171 kernels) | ~11.19 ms/call, ~11% | ~12.81 ms/call, **~37%** |
| H2D+D2H device copy | ~0.66 ms/call, <1% | ~0.62 ms/call, ~1.8% |

The absolute per-call times for kernel compute and device-copy are essentially
unchanged (as expected — MPS does not change how much GPU work exists or how
big the halo messages are). Synchronization's *absolute* time fell ~6× (59.27
→ 9.67 ms/call); its *share* fell only 2× (56%→28%) because the denominator
(total `DYN_CORE` time) also shrank. Launch overhead's absolute time is
essentially flat, but its *share* roughly tripled (16%→51%) purely because the
denominator shrank around it.

**Pre-MPS bottleneck attribution still valid: NO.** Prompt 24's
"HOST-SYNCHRONIZATION-BOUND, with host-staged MPI across six single-GPU-sharing
CUDA contexts as a corroborated secondary factor" conclusion described the
contention regime, which Prompt 26 has removed operationally. With contention
removed, synchronization is no longer the largest single bucket.

## 7. New Bottleneck Classification

**MIXED**, per the task's own >~20%-threshold rule (three categories exceed
20%: launch 51%, kernel compute 37%, synchronization 28%).

- **Top bottleneck: LAUNCH/ORCHESTRATION-BOUND.** `cudaLaunchKernel` is the
  single largest API category (55.2% of API time, 51% of `DYN_CORE`
  wall-time-equivalent), driven by launch **count** (2955 launches/`dyn_core`
  call on rank 0) rather than any one expensive kernel — average per-launch
  host overhead (~6.0 µs) is comparable to average kernel GPU busy time
  (~4.3 µs).
- **Second bottleneck: KERNEL-COMPUTE-BOUND (diffuse).** 171 distinct
  kernels, 37% of `DYN_CORE` time, but no single family exceeds 5.6% of the
  kernel-time total — i.e. compute-bound in aggregate, not in any one hotspot,
  so a single-kernel `ncu` tuning pass would not be expected to move this
  bucket much even if `ncu` were available.

Synchronization (28%) has dropped from the dominant bucket to the third, and
is now largely a downstream consequence of A/B rather than an independent
target.

## 8. CUDA-Aware MPI Recheck (using this profile only)

Device-side H2D+D2H transfer is **0.62 ms/call = 1.8%** of `DYN_CORE` in this
trace (0.62/26.965 = 2.3% against the untraced §3 baseline) — essentially
unchanged from Prompt 24's <1% figure and consistent with Prompt 25a's
message census (312 messages/call, only 8712 B / 2904 B, 100% ≤64 KB, 24
D2H + 24 H2D). Nothing in this profile suggests transfer time grew; if
anything the *share* attributable to memory ops is now slightly more visible
only because the total shrank, not because copy time itself grew.

**Does CUDA-aware MPI now plausibly offer ≥5% whole-call benefit? NO.**
No new evidence contradicts Prompt 25a's ceiling estimate.
**CUDA-aware MPI priority remains LOW.**

## 9. Recommended Next Optimization Target

**Reduce per-`dyn_core` kernel launch count** (currently ~2955/call across
171 distinct kernels) is the best-supported next lever — it is the largest
single measured bucket (51%) and is architectural/orchestration-level, not a
single-kernel tuning problem. Candidate mechanisms already exist in the
codebase and were previously validated bitwise-identical but never enabled on
the standalone path: `--fuse-corner-fill`, `--fuse-halo-edges` (currently
hardcoded off at `fv3_gpu_dyncore_bridge.F90:604-605`), and CUDA Graph capture
(`--cuda-graph`, no ABI field exists yet to reach it from Fortran). This task
does **not** implement any of this — it is a target identification only, per
scope.

## 10. Numerical Algorithm Changed

**NO.** Only job scripts (`fv3_mps_lifecycle.sh`,
`p27_mps_operational.sbatch`) and documentation were added; no CUDA, Fortran,
or C ABI source file was touched.

## 11. Artifacts

New, not for commit unless requested: `<fortran_perf>/experiments/
fv3_mps_lifecycle.sh`, `p27_mps_operational.sbatch`,
`p27_mps_op_58380469.out`. Reused existing job directories `p24_c32_r8_cuda`
and `p24_c32_r8_cuda_nsys` unchanged (both already tracked as scratch
experiment directories from Prompt 24). `nsys` summary CSVs
(`p27_nsys_*_sum.csv`) and the underlying `.nsys-rep`/`.sqlite` live under
`p24_c32_r8_cuda_nsys/` — profiler artifacts, not committed.
