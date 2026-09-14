# Prompt 24 — Real Standalone Fortran vs. CUDA Performance Validation

## 1. Task

Measure real Fortran-vs-CUDA `dyn_core` performance using the same real
standalone `StandAlone_FV3_Dycore.x` executable/workflow validated in
Prompts 19-23 (not the internal `run_small_perf_bench` C++ harness used for
all prior nsys/CUDA-Graph/fusion profiling), at the primary target
configuration: c32, R8, `npz=10`, `n_split=3`, `k_split=1`, `bdt=1200s`, 6
MPI ranks (`FV_NX=1`/`FV_NY=6`, one tile per rank), one A100, one node.

No CUDA/Fortran numerical algorithm source was touched. **Numerical
algorithm changed: NO.** This task made one runtime-behavior-preserving
diagnostic addition (a per-rank `nsys` profiling wrapper script used only
for a separate, throwaway 10-call diagnostic job; not part of any timed
result and not part of the production run path).

## 2. Critical Finding: Old Harness Numbers Are Not From the Standalone Path

Before running anything, the CUDA runtime flags cited in `CURRENT_STATE.md`
§8 (`--fuse-halo-edges`, `--cuda-graph`) were checked against the real
Fortran/CUDA integration boundary:

- `cuda/interface/fv3_gpu_dyncore_bridge.F90:604-605` hardcodes
  `cfg%fuse_corner_fill = 0` and `cfg%fuse_halo_edges = 0` at `NumericalConfig`
  install time. No Fortran-reachable knob sets either to 1.
- No CUDA Graph field exists anywhere in the ABI config struct
  (`fv3_gpu_abi_mod.F90`) or the internal runtime config
  (`fv3_gpu_internal.hpp`). CUDA Graph capture is exclusively a
  `run_small_perf_bench` command-line flag (`--cuda-graph`), not reachable
  through the Fortran/standalone integration path at all.
- Every prior `nsys`/`ncu` profiling artifact in `perf_rerun_logs/` (job
  58238910's CUDA-Graph on/off table, the fusion/halo-edge nsys traces, etc.)
  targeted `./build_cuda_r8_perf/fv3_gpu_perf_bench` directly with
  `--ntasks=1` (single process, no MPI, in-process six-tile execution).

**Conclusion: the `CURRENT_STATE.md` §8 CUDA performance numbers (and
`milestone_3a12`'s "CUDA reference numbers... Milestone 3A.11" row) were
measured on the internal single-process C++ harness, not on the real
standalone six-rank MPI production integration path.** They are not
comparable to, and are superseded for standalone-path purposes by, the
numbers in this document. This was directly confirmed, not assumed, by
inspecting the actual `mpirun` invocation line captured in a diagnostic job's
stdout (see §5) and by grepping the bridge/ABI source for the flag fields.

Also newly confirmed: the real standalone executable launches via **Intel
MPI** (`/usr/local/intel/oneapi/2024/mpi/2021.13/bin/mpirun`, via
`esma_mpirun`), not the OpenMPI 4.1.6 module used for CUDA source
compilation/bring-up. This is consistent with `milestone_3a12`'s note that
the standalone fixture is an Intel-`ifort` build; it is recorded here because
the platform docs' validated module stack names OpenMPI as the MPI
implementation, which applies to CUDA compilation, not to this Fortran
executable's own runtime launch.

## 3. Benchmark Design

Reused the exact Prompt 23 Stage B job template
(`p23_stageB_r8_c32_{fortran,cuda}`, already validated stable and
numerically agreeing same-day) unmodified except for `JOB_SGMT` (widened from
6 calls to 100 calls/rep: `'00000001 092000'` = 1 day 09:20:00 = 120000s /
1200s = 100 calls exactly) and savepoint dumping disabled
(`FV3_SAVEPOINT_ENABLE 0`) to avoid per-call binary-dump I/O contaminating
the `fv_timing_mod` `DYN_CORE` timer and to avoid ~15-17x disk growth for no
benefit (savepoints are not needed for a timing-only run; correctness is
covered separately, see §7).

New job directories: `p24_c32_r8_{fortran,cuda}` (in
`GEOSfvdycore_fortran_perf/experiments/`, not in this repo). One `sbatch`
driver (`p24_c32_r8_both.sbatch`) ran both backends back-to-back on the same
node/GPU allocation, 5 independent reps each (fresh `csh -f ./fv3.j`
invocation per rep, log copied to `rep{N}.log` before the next rep
overwrites it — same methodology as `milestone_3a12` §5). Job 58295284, one
`gpu_a100`/`rome` node (`warpa003`), single A100, 6 MPI ranks, walltime used
00:02:29 for all 10 reps.

Timer: `fv_timing_mod`'s own `DYN_CORE`-scoped accumulator, read from the
`timing_prt` "Block / User time / System Time / Total Time / GID" table
printed once per run — the same primary metric `milestone_3a12` established
and the task's "reuse existing timers" instruction calls for. `ms/call =
1000 * (DYN_CORE Total Time) / 100`. All 10 reps: `rc=0`, 0 `Range_Warn`, 0
`NaN`.

**CUDA options enabled: none reachable.** Per §2, the standalone integration
path always runs with `fuse_corner_fill=0`, `fuse_halo_edges=0`, no CUDA
Graph — this is the only CUDA execution path this integration currently
exposes, not a chosen configuration among alternatives.

**Warmup:** not separately excluded via a second timer region (`fv_timing_mod`
has no reset hook, confirmed already in `milestone_3a12`). Handled the same
way `milestone_3a12` did: 100 calls/rep dilutes one-time context/MPI-buffer
init to a small fraction of the per-rep total (a separate 10-call diagnostic
run, §5, measured ~111-118 ms/call vs. the 100-call steady-state ~105.6
ms/call median — a ~5-12% warmup skew at 10 calls, consistent with
"amortizes with more calls," not a large distorting effect at 100 calls).

## 4. Primary Timing Results

### Fortran (100 calls/rep, 5 reps)

| Rep | `DYN_CORE` total (s) | ms/call |
|---|---|---|
| 1 | 1.9836 | 19.836 |
| 2 | 1.9941 | 19.941 |
| 3 | 1.9856 | 19.856 |
| 4 | 1.9955 | 19.955 |
| 5 | 1.9930 | 19.930 |

min = 19.836, **median = 19.930**, mean = 19.904, max = 19.955 ms/call.
Spread (max−min) = 0.119 ms (0.6%).

### CUDA (100 calls/rep, 5 reps)

| Rep | `DYN_CORE` total (s) | ms/call |
|---|---|---|
| 1 | 10.5378 | 105.378 |
| 2 | 10.6050 | 106.050 |
| 3 | 10.5700 | 105.700 |
| 4 | 10.5563 | 105.563 |
| 5 | 10.5269 | 105.269 |

min = 105.269, **median = 105.563**, mean = 105.592, max = 106.050 ms/call.
Spread (max−min) = 0.781 ms (0.74%).

Both distributions are tight (sub-1% spread) — no scheduler/resource-noise
problem encountered.

### Dyn_core speedup

Fortran median ÷ CUDA median = 19.930 / 105.563 = **0.189×**. The CUDA path
is **~5.30× slower per call than the 6-rank CPU Fortran path** at this
configuration, in the real standalone integration. This is a reversal of
the informal near-parity/5%-CUDA-advantage picture in `CURRENT_STATE.md` §8
— that picture came from the internal single-process harness (§2) and does
not describe this integration's actual behavior.

Standalone whole-timed-region speedup: **NOT MEASURED**. Only the
`fv_timing_mod` `DYN_CORE`-scoped total was extracted, per the existing
methodology; isolating total-region wall-clock cleanly from MAPL/History/init
overhead was out of scope for this task and was not attempted to avoid
overstating whole-model acceleration.

## 5. CUDA Cost Breakdown (nsys, diagnostic only)

`fv_timing_mod` cannot see inside the CUDA `dyn_core` call (it wraps the
Fortran call site only), so no existing timer provides a compute/transfer/
sync/MPI split. Per the task's explicit allowance ("nsys only if required to
classify a large unexplained bottleneck" — the 5.30× reversal in §4
qualifies), one short (10-call) diagnostic-only `nsys` capture was taken on
MPI rank 0 of the real standalone CUDA job. This required a small rank-gated
wrapper script (`nsys_wrap.sh`, checking `PMI_RANK`/`MPI_LOCALRANKID` and
only invoking `nsys profile` for rank 0, `exec`-ing the plain binary for
other ranks) because `nsys`'s `%q{OMPI_COMM_WORLD_RANK}` per-rank
output-naming syntax is an OpenMPI convention and this launch uses Intel MPI
(§2) — the first attempt using that syntax failed cleanly (no trace file
produced, run itself still completed rc=0). This wrapper is a diagnostic-only
artifact, not part of the production path, and is not part of any timed
result. Job 58295418, rc=0, 0 NaN, 0 Range_Warn.

| Category | Source | Total (10 calls) | ms/call | % of CUDA `DYN_CORE` |
|---|---|---|---|---|
| Host synchronization (`cudaEventSynchronize`) | `cuda_api_sum` | 592.69 ms (250 calls) | 59.27 | ~56% |
| Launch/orchestration (`cudaLaunchKernel` API) | `cuda_api_sum` | 173.32 ms (29,550 calls) | 17.33 | ~16% |
| GPU compute (sum of all 171 distinct kernels) | `cuda_gpu_kern_sum` | 111.89 ms | 11.19 | ~11% |
| Host→Device copy | `cuda_gpu_mem_time_sum` | 3.84 ms (595 calls) | 0.38 | ~0.4% |
| Device→Host copy | `cuda_gpu_mem_time_sum` | 2.74 ms (500 calls) | 0.27 | ~0.3% |
| CUDA memset | `cuda_gpu_mem_time_sum` | 0.12 ms | 0.01 | ~0.01% |

(Percentages are relative to the 100-call steady-state CUDA median,
105.563 ms/call, since the buckets above are measured on a separate 10-call
run whose own average was ~111 ms/call.)

Supporting OS-runtime evidence (`osrt_sum`, same trace): `poll` accounts for
77.5% of all traced OS-runtime call time (5.17 s total over 67 calls),
consistent with either real cross-rank MPI wait or process init/shutdown
idle time — the coarse trace cannot cleanly separate the two without
per-call NVTX ranges, which were not added (would be new instrumentation,
out of scope per §15 of the task). This is reported as a supporting,
not fully call-attributed, signal.

D2H+H2D transfer time is small and directly measured (~0.66 ms/call,
<1%) — this **rules out H2D/D2H-TRANSFER-BOUND** as the explanation.
GPU kernel compute is also a minority contributor (~11%) — this **rules out
KERNEL-COMPUTE-BOUND** as the primary explanation, independent of the
existing `ncu` platform block (§9 of `CURRENT_STATE.md`; still confirmed
blocked, not re-tested this task since compute is not the bottleneck
regardless).

## 6. Interpretation

**Dominant bottleneck: HOST-SYNCHRONIZATION-BOUND**, with host-staged MPI as
a material, plausibly-overlapping secondary factor.

`cudaEventSynchronize` alone (~56% of per-call CUDA `DYN_CORE` time) is the
single largest measured category, well ahead of GPU compute (~11%) and an
order of magnitude ahead of transfer copy time (~0.7% combined). This is
consistent with the architecture: each host-staged halo exchange requires
the host to wait for device work to finish before it can safely stage a
buffer through MPI, and the six MPI ranks in this configuration are six
separate CUDA contexts sharing the one allocated A100 (only `--gres=gpu:1`
was requested, matching the validated production decomposition) — a
materially different execution model from the internal harness's single
process holding all six tiles resident together, and a plausible reason the
per-call sync/launch overhead is far larger here than in that harness's
prior measurements (`cudaStreamSynchronize` was already flagged as large in
Prompt 13's *single-process* regime; six independent processes each
serializing similarly against one physical device compounds it further).
This is reported as the most likely explanation given available evidence,
not as independently proven by this task's diagnostic-only trace.

The `poll`-dominated `osrt_sum` result is consistent with, but not cleanly
separable from, real inter-rank MPI wait time inside the timed region — it
is reported as corroborating, not decisive, evidence for the
`HOST-STAGED-MPI-BOUND` secondary classification.

## 7. Correctness Guard

Per the task's explicit instruction not to repeat full Prompt 23 depth: the
Prompt 23 Stage B job pair (`p23_stageB_r8_c32_{fortran,cuda}`) is the exact
same binary/configuration (c32, R8, `bdt=1200`) used for this benchmark, run
earlier the same day, and its results are directly reused rather than
re-run:

- Both backends: `rc=0`, 0 `Range_Warn`, 0 NaN (still present on disk,
  re-confirmed by direct `grep` this task, not re-derived from memory).
- `delp` positive/finite confirmed at all ranks/calls (Prompt 23 §4).
- CUDA-vs-Fortran comparison: **PASS**, worst max_rel ≈6.7e-11 (`w`, dcx) —
  well inside already-established bounds (Prompt 23 §5, Stage B).

No new savepoint-based comparison was run for this task.

## 8. Priority Assessment

- **Host-staging fraction**: direct D2H+H2D copy cost is small (<1%), but
  the synchronization required to safely sequence host-staged
  communication (§6) is ~56% of per-call CUDA cost — a material fraction by
  the intent of the task's threshold, even though the raw byte-copy time
  alone would not be. **CUDA-aware MPI priority: HIGH.** Eliminating the
  explicit D2H→MPI→H2D round trip (Stage 2 of the Communication Evolution
  in `project_architecture.md`) would remove not just the copy time but
  plausibly most of the surrounding event-synchronization currently required
  to serialize around it — the single largest lever identified by this
  measurement.
- **Kernel tuning priority: LOW.** GPU kernel compute is only ~11% of
  per-call CUDA cost in this real integration path; tuning
  `riem_solver3_kernel`/`riem_solver_c_kernel` (Prompt 13's target, valid in
  the *internal-harness* regime) would improve at most a small slice of an
  already-small bucket here. `ncu` remains platform-blocked
  (`ERR_NVGPUCTRPERM`, unchanged, not re-tested since irrelevant to this
  result) — this does not change the priority call, since compute was never
  the identified bottleneck in this path.

## 9. Scope / Constraints Honored

- Sonnet-only throughout; no escalation was warranted.
- c32 R8 only; no c16 lightweight check run (§4's result was unambiguous —
  a clean, tight, 5.30× reversal — no scaling check was needed to resolve
  ambiguity).
- No R4 benchmarking (no specific reason arose).
- One node, 6 ranks, 1 GPU throughout, including the diagnostic `nsys` job.
- No CUDA-aware MPI implementation, no kernel modifications, no CUDA Graph
  changes, no new optimization flags introduced or compared.
- `nsys` used once, for a short diagnostic-only 10-call run, specifically to
  classify the unexplained 5.30× reversal — not used for the primary timing
  result. `ncu` not used (still blocked; irrelevant to the finding).
- No commits/pushes made. No logs/profiler artifacts/build outputs
  committed to this repository; all job directories and `.nsys-rep`/`.csv`
  outputs live under `GEOSfvdycore_fortran_perf/experiments/`, outside this
  repo.

## 10. Result

Real standalone performance baseline established at c32/R8: Fortran median
19.930 ms/call, CUDA median 105.563 ms/call, CUDA **0.189× (5.30× slower)**
in this integration. Dominant bottleneck: host-synchronization (~56% of
per-call CUDA cost, measured), with host-staged MPI as a corroborated
secondary factor; GPU compute and H2D/D2H transfer are both ruled out as the
primary cause (measured at ~11% and <1% respectively). Next optimization
direction is clear and evidence-based: CUDA-aware MPI (host-staging
elimination) is the high-priority lever; kernel-level tuning is
low-priority regardless of `ncu` availability.

**STANDALONE PERFORMANCE BASELINE COMPLETE — READY FOR TARGETED
OPTIMIZATION**
