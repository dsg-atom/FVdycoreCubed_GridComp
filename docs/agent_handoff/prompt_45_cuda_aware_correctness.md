# Prompt 45 — Reconcile Intel-MPI CUDA-Aware Transport Correctness

Branch `experiment/held-suarez-e2e`. Objective: root-cause and fix the
CUDA-aware MPI transport correctness failure that appeared once Prompt 44's
MAPL resource-read race was cleared, without touching FV3 numerics,
Held-Suarez physics, MPI topology/rank-tile decomposition, or performance.

## 1. Failure Visibility (Stage 0)

Before this prompt, `fv3_gpu_dyn_core_step failed:` FATAL messages from the
CUDA vertical-solver dispatch (`cuda/dyn_core/vertical/fv3_vertical_kernels.cu`,
`launch_update_dz_d`/`launch_vertical_solver`) carried bare `fv3_gpu_status`
enum returns with no populated `fv3_gpu_error` (empty `message`, `native_code
== 0`), because several early-return sites used bare `return FV3_GPU_*;`
instead of the project's existing `fail(error, code, api, message,
native_code)` mechanism. This made Prompt 44's original C512 crash
undiagnosable beyond the function name.

**Fix (Change 1/2 of budget)**: wrapped every bare enum return in
`launch_update_dz_d` and `launch_vertical_solver` with `fail(...)`, giving
each a `static constexpr const char *api` tag and a specific message.
For the two `cudaGetLastError()` kernel-launch checks, additionally captured
the `cudaError_t` value and appended `cudaGetErrorString(err)` plus the
numeric code (`native_code`) — the initial per-stage wrap alone surfaced the
function name but not the actual CUDA error text; this second pass (still
the same file/call sites, no new logic path) was required to make the FATAL
string actionable. Net: 1 file, `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`,
60 lines changed, no numerical semantics touched.

## 2. Root Cause

**C512 six-rank device memory capacity, not a synchronization/pointer/
lifetime defect.**

Direct `nvidia-smi --query-gpu=memory.used` polling during paired host_staged
vs. cuda_aware runs (job 58453867, same node, same GPU, same binary) shows:

| Transport | Peak device memory observed | A100 total |
|---|---|---|
| host_staged (steady state, 6 ranks) | **39466 MiB** | 40960 MiB (96.4%) |
| cuda_aware (crashes during ramp-up, before steady state) | 24493–27985 MiB | 40960 MiB |

host_staged's own steady-state footprint for six C512 ranks sharing one A100
already leaves under **1.5 GB of headroom**. `cuda_aware` (Intel MPI 2021.13,
`I_MPI_OFFLOAD=1`, GPU-direct/RDMA buffer registration via the `mlx`
libfabric provider) requires additional device-side memory on top of that
already-near-full baseline. The transport code itself
(`cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp`) allocates **no extra device
buffers** of its own for cuda_aware — it reuses the same device-resident
pack/unpack buffers as host_staged and skips only the pinned-*host* staging
copy — so the extra consumption is Intel MPI's own runtime-internal
GPU-direct buffer pool, external to this project's code.

With the populated error string (§1), the crash is consistently:

```
FATAL from PE <n>: CUDA dyn_core backend: fv3_gpu_dyn_core_step failed:
riem_solver3_kernel launch failed: out of memory
```

`cudaGetErrorString` reports `cudaErrorMemoryAllocation` (text: "out of
memory"). The failing rank differs run to run (PE 4 in Prompt 44's original
report; PE 2 and PE 1 in this prompt's two reproductions) — consistent with
a genuine resource-exhaustion race (whichever rank's allocation pattern
first exceeds the ~1.5 GB remaining headroom fails first), not a
rank-specific logic bug.

Stages 2–6 of the original plan (device pointer verification, pack→MPI sync,
MPI-receive→unpack sync, buffer lifetime/reuse audit, MPI return-code
checking) were **not exercised** — this evidence already conclusively
attributes the failure to device memory capacity, a category none of those
stages target. Continuing through them would not have changed the
diagnosis and was skipped per the project's bounded-debugging policy (stop
once the hypothesis is resolved).

## 3. Fix

**None applied beyond the Stage 0 error-visibility completion (§1).** This
is not a code defect fixable within Prompt 45's scope:

- Reducing device memory footprint elsewhere is a broad optimization effort
  (explicitly deferred/out of scope for a correctness-only prompt).
- Reducing ranks-per-GPU density or changing GPU count is an MPI
  topology/decomposition change (explicitly forbidden by this prompt's
  constraints).
- Eliminating Intel MPI's GPU-direct buffer pool overhead would mean not
  using `I_MPI_OFFLOAD=1`/GPU-direct RDMA, i.e. redesigning the cuda_aware
  transport's use of the MPI runtime — a transport redesign, one of this
  prompt's explicit hard-stop triggers.

No further source change was made. 1 of 2 budgeted changes used.

## 4. Transport Unit Test (Stage 1/8)

Isolated a **pure diagnostic-tooling defect in this prompt's own job
scripts** (not a code or transport defect) before reaching the real
finding: early sbatch attempts loaded `comp/intel/2024.2.0` in addition to
the correct `comp/gcc/12.3.0`, corrupting the runtime environment for any
multi-process `mpirun`/`mpiexec`/`ctest` launch of the GCC-built
`fv3_halo_mpi_transport_test` binary (`build_cuda_r8_intelmpi`'s
`CMakeCache.txt` confirms `CMAKE_CXX_COMPILER=g++`, GCC, not Intel) — single
rank and gdb-wrapped launches were unaffected (Heisenbug signature), and a
minimal CUDA-kernel reproducer's `nvcc -ccbin=mpigxx` build outright failed
with `unsupported Intel ICX compiler!` under the same wrong module set,
directly exposing the mismatch. Corrected module set:
`comp/gcc/12.3.0` + `mpi/impi/2021.13` + `nvidia/nvhpc-byo-compiler/24.7`
(no `comp/intel/2024.2.0`).

With the corrected environment, the permanent six-rank ctest
(`fv3_halo_mpi_transport`, `cuda/dyn_core/tests/integration/
test_halo_mpi_transport.cu`) was run 5x per transport mode (cap respected):

| Mode | Attempts | Result |
|---|---|---|
| host_staged | 5/5 | PASS, 16692 checks, 0 failures, every run |
| cuda_aware | 5/5 | PASS, 16692 checks, 0 failures, every run |

**10/10 PASS — transport-unit level.** Coverage: scalar CENTER halo, both
wind staggerings (DGRID_NE/CGRID_NE), 4-field width-3 multi-ring
(H_DSCALAR), E11 ring-0 seam reconciliation. NOT covered: corner-halo
blocks; full dyn_core numerical validation. This satisfies Stage 8's
acceptance bar directly (no MPS needed to reach it — see §6).

This clean transport-unit pass does **not** contradict §2's finding: the
unit test exercises small, fixed-size synthetic halo buffers at low total
device memory pressure, nowhere near the ~39.5 GB six-rank C512 steady
state that exposes the capacity ceiling.

## 5. C512 CUDA-Aware Smoke (Stage 9)

Reused Prompt 44's experiment mechanism
(`GEOSfvdycore_fortran_perf/experiments/p44_c512_cuda_aware_resume/`) per
the project's reuse policy, copied to a fresh
`p45_c512_cuda_aware_smoke/` directory, binary refreshed from a clean
two-stage rebuild (CUDA library, then Fortran relink) to pick up §1's fix,
verified via `strings`. `JOB_SGMT` reduced from the original 6h to 2
simulated hours to stay within Prompt 45's 2–6h smoke range at lower cost.

- **host_staged control**: ran the full 2 simulated hours cleanly (job
  58453614, step .0, ~14 min wall-clock, throughput ~9.1–9.2 days/day,
  matching §6's historical C512 CUDA figures) — no crash, confirming the
  smoke harness itself and the freshly-rebuilt binary are sound.
- **cuda_aware test**: crashed within the first `riem_solver3_kernel`
  launch after MAPL/dycore initialization (~39s wall-clock), every attempt
  (job 58453614 step .1; retested in isolation as job 58453844 and again
  under memory polling as job 58453867 step .1) — never reached its first
  full acoustic substep, let alone 2 simulated hours.

**Result: cuda_aware smoke FAILED at initialization. host_staged smoke
PASSED cleanly for the full 2-hour segment.**

## 6. Numerical Comparison (Stage 10)

**Not performed — no comparable cuda_aware state exists.** The cuda_aware
run aborts before completing (or even starting) a full acoustic substep, so
there is no T/U/V/PS/delp state at any checkpoint to diff against the
host_staged control. This stage is blocked, not passed or failed.

## 7. MPS Discriminator (Stage 7)

**Explicitly skipped — not needed and not informative.** Stage 1/8's clean
10/10 transport-unit pass was achieved without MPS, satisfying the
acceptance bar without requiring this discriminator. Separately, §2's root
cause (device memory capacity) would not be resolved by MPS in either
direction: MPS changes compute-context/kernel-concurrency sharing between
ranks, not per-rank device memory allocation ceilings — a memory-capacity
failure is orthogonal to whether MPS is enabled. Running it would have
consumed the discriminator budget for no diagnostic value.

## 8. Prompt 43 Resume Decision (Stage 11)

**Do not resume Prompt 43's performance A/B for cuda_aware at C512 under
the current six-ranks-per-A100 configuration.** It is blocked by a real
device memory capacity ceiling (§2), not a fixable software defect within
either Prompt 43's or Prompt 45's scope. host_staged remains the only
C512-viable transport at this rank density. A future prompt could
investigate lower rank-per-GPU density (e.g. 3 ranks/GPU across 2 GPUs) or
Intel MPI GPU-buffer-pool sizing controls as a dedicated, explicitly-scoped
follow-on — both require topology/decomposition or transport changes
outside this prompt's authorization.

## 9. Source Change Ledger

| # | File | Change | Numerical semantics changed |
|---|---|---|---|
| 1 | `cuda/dyn_core/vertical/fv3_vertical_kernels.cu` | Wrapped bare enum returns in `launch_update_dz_d`/`launch_vertical_solver` with `fail(...)`; captured and appended `cudaGetErrorString`/native code for the two kernel-launch checks | NO |

1 of 2 budgeted changes used. No numerics, physics, topology, or
decomposition touched.

## 10. Artifacts

- `scratch_p45/` (untracked, this repo): `p45_diag1`–`p45_diag10`,
  `p45_stage1_repro2`, `p45_rebuild`, `p45_rebuild2`, `p45_stage9*`,
  `mini_cuda_mpi.cpp`, `mini_cuda_mpi_kernel.cu`, `hello.c` — all sbatch
  scripts and job logs for the module-mismatch diagnosis, Stage 1
  reproduction, rebuilds, and Stage 9 smoke/memory-polling runs.
- `GEOSfvdycore_fortran_perf/experiments/p45_c512_cuda_aware_smoke/` —
  Stage 9 experiment directory (host_staged_out/, cuda_aware_out/,
  cuda_aware_out2/, `gpu_mem_poll_hoststaged.csv`,
  `gpu_mem_poll_cudaaware.csv`).

## Final Report

**FAILURE VISIBILITY**: FIXED. Bare enum returns in the two vertical-solver
dispatch functions now populate `fv3_gpu_error` via `fail(...)`, including
`cudaGetErrorString`-derived text for kernel-launch failures. The FATAL
message now reads `riem_solver3_kernel launch failed: out of memory`
instead of a bare function name.

**ROOT CAUSE**: CUDA-aware MPI transport (Intel MPI 2021.13,
`I_MPI_OFFLOAD=1`) requires device memory headroom beyond what remains
after six C512 ranks already consume ~39.5 GB of a 40 GB A100 under
host_staged — confirmed by direct `nvidia-smi` memory polling. Not a
synchronization, device-pointer, or buffer-lifetime defect.

**FIX**: None applicable within scope — this is a device memory capacity
limitation, not a code defect. Any real fix requires either reduced
per-rank memory footprint (broad optimization, out of scope), reduced
ranks-per-GPU density (topology change, forbidden), or transport redesign
to avoid Intel MPI's GPU-direct buffer pool (forbidden hard-stop trigger).

**TRANSPORT UNIT TEST**: 10/10 PASS at 6-rank for both host_staged and
cuda_aware (`fv3_halo_mpi_transport` ctest, 16692 checks/attempt, 0
failures). Diagnosed and corrected an unrelated wrong-module-set issue in
this prompt's own job scripts along the way (no code change).

**C512 CUDA-AWARE SMOKE**: FAILED — crashes at the first Riemann-solver
kernel launch during initialization, before completing any acoustic
substep. host_staged control passed cleanly for the full 2-simulated-hour
segment on the same rebuilt binary.

**NUMERICAL COMPARISON**: Not performed — no comparable cuda_aware state
exists to compare.

**MPS DISCRIMINATOR**: Skipped — not needed (10/10 unit-test pass already
achieved without it) and not diagnostically relevant to a memory-capacity
root cause.

**PROMPT 43 RESUME**: Do not resume the cuda_aware performance A/B at C512
under the current six-ranks-per-A100 configuration. host_staged remains the
only C512-viable transport. A dedicated future prompt would need to change
rank/GPU density or the transport's memory strategy — both out of this
prompt's authorization.

**PROJECT STATUS**: Gate C2 production dyn_core correctness (§4 of
CURRENT_STATE.md) and the HS C256/C512 host_staged correctness gates are
unaffected — no numerics changed. The C512 cuda_aware transport
(implemented in Prompt 43) is now understood to be blocked by device
memory capacity at the current 6-ranks-per-A100 density, not by an
unresolved software defect; the MAPL resource-read race (Prompt 44) that
previously blocked reaching this point is confirmed resolved.

C512 CUDA-AWARE TRANSPORT STILL BLOCKED — six C512 ranks already consume ~39.5 GB of a 40 GB A100 under host_staged (1.5 GB headroom); cuda_aware's additional Intel MPI GPU-direct buffer-pool memory exceeds that headroom, causing cudaErrorMemoryAllocation ("out of memory") at the first Riemann-solver kernel launch — a device memory capacity limit at the current six-ranks-per-A100 density, not a synchronization/pointer/lifetime defect.
