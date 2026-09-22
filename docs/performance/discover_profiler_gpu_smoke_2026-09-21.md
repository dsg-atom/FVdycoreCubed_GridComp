# Discover GPU-Node Profiler Smoke Test (Prompt 49A)

Date: 2026-09-21
Status: **PROFILER-FUNCTIONALITY SMOKE TEST ONLY** — no source changed, no
numerical/runtime configuration changed, no full profiling campaign, no
performance benchmarking. One GPU job submitted.

---

## 1. Job / node / GPU

| Field | Value |
|---|---|
| Job ID | 58493100 |
| Node | `warpa005` |
| GPU | `NVIDIA A100-SXM4-40GB`, driver `535.104.12` |
| Elapsed | 21s (per `sacct`), overall job exit code 0:0 |
| Git HEAD on compute node | `33e10c50b741fc53817fe035a1f2abe64c34bb65` (matches login node — no drift) |
| Script | `prompt49a_profiler_gpu_smoke.sbatch` (repo root, untracked) |

## 2. Environment (exact sequence used)

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load comp/intel/2024.2.0
module load mpi/impi/2021.13
module load nvidia/nvhpc-byo-compiler/24.7
export I_MPI_CXX=g++
export PATH=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/Nsight_Systems/bin:/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/Nsight_Compute:$PATH
```

Confirmed live on the compute node (not just login node):

- `mpirun --version` -> `Intel(R) MPI Library for Linux* OS, Version 2021.13 Build 20240515 (id: df72937)`
- `nsys --version` -> `NVIDIA Nsight Systems version 2024.4.1.61-244134315967v0`
- `ncu --version` -> `Version 2024.2.1.0 (build 34372528) (public-release)`
- `ldd $BENCH | grep -Ei 'mpi|cuda'` -> `fv3_gpu_perf_bench` (`build_cuda_r8_intelmpi/`) links against Intel MPI 2021.13 (`libmpi.so.12`, `libmpicxx.so.12` from `/usr/local/intel/oneapi/2024/mpi/2021.13/lib/release/`) and CUDA 12.5 runtime (`libcudart.so.12` from the NVHPC 24.7 tree) — same linkage previously confirmed on the login node.

**Intel MPI 2021.13 confirmed preserved on the GPU node with the profiler PATH active.**

## 3. Stage 1 — six-rank Intel-MPI host-staged dyn_core smoke: FAILED

```
mpirun -np 6 fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 --mode resident \
  --hardware A100_p49asmoke --warmup 0 --iters 2 --repeats 1 --mpi-halo
```

Result: **all six MPI ranks segfaulted** (`SIGNAL 11`, RC=255):

```
BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES
RANK 0 PID 26808 RUNNING AT warpa005
KILLED BY SIGNAL: 11 (Segmentation fault)
... (identical for RANK 1-5)
```

**Root-cause hypothesis (not yet confirmed by source inspection or a retry)**:
the only prior validated `--mpi-halo` invocation in this project
(`perf_rerun_logs/prompt15_mpi_dyncore.sbatch`, Milestone 3A.15 SS12/SS14)
always paired `--mpi-halo` with the full authoritative fixture set
(`--grid-metrics`, `--sp0-state`, `--sp1-boundary`, `--damp-n-sponge -1`,
`--akap-exner`). This smoke test's invocation used only synthetic/default
state (no fixture args) to keep the job minimal per the task's "shortest
practical" instruction — that deviation from the only previously-validated
pattern is the leading suspect for the crash. **This was not root-caused or
retried in this task** (out of scope: the task explicitly prohibits a full
profiling campaign or additional performance runs, and the Stage 2/3 goals
below were independently achieved via the same binary's default
single-process transport path, which does not depend on `--mpi-halo`).

This is recorded here as an **honest, unresolved caveat**, not papered over:
the six-rank MPI-halo host-staged path did **not** run cleanly in this smoke
test.

## 4. Stage 2 — Nsight Systems smoke capture: PASSED

Single-process invocation (default in-process/`PeerHaloTransport` path, no
`--mpi-halo`):

```
nsys profile --trace=cuda,osrt --force-overwrite=true \
  --output=perf_rerun_logs/p49a_nsys_smoke_58493100 \
  fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 --mode resident \
  --hardware A100_p49asmoke --warmup 0 --iters 2 --repeats 1
```

- RC = 0.
- Application's own self-check printed: `=== 3A.11 CUDA perf benchmark: PASS ===`
  (the benchmark completed and validated its own internal check cleanly under
  `nsys`).
- Report file: `perf_rerun_logs/p49a_nsys_smoke_58493100.nsys-rep`, **2,847,768
  bytes**, confirmed present via `ls -la`.

### CUDA kernels visible — YES, with recognizable FV3 kernel names

`nsys stats --report cuda_gpu_kern_sum` against the report produced a full,
non-empty **CUDA GPU Kernel Summary** table. Top entries (52 distinct kernel
names total), all real FV3 CUDA dyn_core kernels, e.g.:

```
fv3::gpu::dgrid::copy_corners_kernel               5.9%  9,015,918 ns  2,280 instances
fv3::gpu::cgrid::fill_4corners_kernel              4.1%  6,240,751 ns  1,512 instances
fv3::gpu::halo::pack_edge_kernel                   3.8%  5,843,718 ns  2,112 instances
fv3::gpu::halo::unpack_edge_kernel                 3.6%  5,573,106 ns  2,040 instances
fv3::gpu::dgrid::xppm_pass_kernel                  2.6%  3,935,206 ns    432 instances
fv3::gpu::vertical::riem_solver_c_kernel<false>    2.3%  3,519,158 ns     36 instances
fv3::gpu::vertical::riem_solver3_kernel            2.1%  3,271,280 ns     36 instances
fv3::gpu::operators::a2b4_global_corner_kernel     2.1%  3,272,661 ns    216 instances
fv3::gpu::cgrid::csw_ke_cubed / csw_scalars_nh / a2c_uc_east_edge / ...
fv3::gpu::cgrid::update_dz_c_winds / update_dz_c_update_gz / update_dz_c_x_flux / update_dz_c_y_flux
fv3::gpu::dgrid::yppm_pass_kernel / del6_flux_x_kernel / del6_flux_y_kernel / deln_flux_x_kernel / deln_flux_y_kernel
```

These match the WP-A/B/C/D kernel families documented in
`docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md` Part 1 (C-grid
predictor, D-grid transport, vertical Riemann solvers, halo pack/unpack,
`a2b4`/D2A2C operators) — confirming Nsight Systems is capturing the real
dyn_core execution, not a trivial/degenerate trace.

### CUDA API calls visible — YES

`nsys stats --report cuda_api_sum` initially failed with a benign,
non-fatal warning (`WARNING: Existing SQLite export found... File is older
than input file... Use --force-export=true`) because the prior
`cuda_gpu_kern_sum` call had already generated the `.sqlite` export from the
same `.nsys-rep`, and the subsequent call's own re-check flagged a timestamp
mismatch. This is a caching/ordering artifact, not a capture failure — it
was resolved via `nsys stats --force-export=true --report cuda_api_sum` run
directly against the already-captured `.nsys-rep` on the login node (a
post-processing/export step, not a new GPU capture; no new job needed):

```
cudaLaunchKernel           94.6%  1,053,888,608 ns  35,460 calls
cudaMemcpyAsync             2.5%     28,168,196 ns   2,754 calls
cudaEventRecord             1.1%     11,774,453 ns     918 calls
cudaStreamWaitEvent         1.0%     11,291,516 ns   3,618 calls
cudaMalloc                  0.3%      3,473,974 ns     594 calls
cudaFree                    0.3%      3,419,946 ns     594 calls
cudaMemsetAsync             0.1%      1,577,688 ns      84 calls
cudaStreamSynchronize       0.0%        344,249 ns      54 calls
cudaStreamCreateWithFlags   0.0%        184,998 ns      12 calls
...
```

`cudaStreamSynchronize` is present (54 calls) — the host-blocking
synchronization primitive is visible and exercised. `cudaEventSynchronize`
(the specific primitive used in the MPI-halo transport per
`fv3_gpu_halo_mpi.cpp:243`) does **not** appear, which is expected: this
single-process capture used the default in-process `PeerHaloTransport` path,
not `--mpi-halo` (which failed in Stage 1 before reaching a profiler). This
is noted as scope, not a defect.

## 5. Stage 3 — Nsight Compute permission check: BLOCKED (ERR_NVGPUCTRPERM)

```
ncu --launch-count 1 --log-file perf_rerun_logs/p49a_ncu_smoke_58493100.log \
  fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 --mode resident \
  --hardware A100_p49asmoke --warmup 0 --iters 1 --repeats 1
```

RC = 1. Exact captured output:

```
==PROF== Connected to process 26974 (fv3_gpu_perf_bench)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0. For instructions on
enabling permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 26974
```

This is the **identical error text** to the historical findings in Prompts
12/13/24/26, reproduced fresh on `warpa005` in this session. Per the task's
explicit instruction, no privilege workaround, driver configuration change,
or `sudo` was attempted — the result is recorded as-is.

## 6. Whether Prompt 49 may proceed

- **Nsight Systems: READY.** It attached to the real FV3 CUDA dyn_core
  executable on a live A100, produced a valid non-trivial report, and both
  `cuda_gpu_kern_sum` and `cuda_api_sum` post-processing succeeded with real
  FV3 kernel names and CUDA API activity visible. Prompt 49's Phase 0/Phase A
  (same-node critical-path timeline capture, per the optimization roadmap)
  may proceed using this exact environment/PATH sequence.
- **Nsight Compute: BLOCKED.** Live GPU performance-counter access is denied
  at the driver/permission level (`ERR_NVGPUCTRPERM`), unchanged from prior
  findings. Kernel-level occupancy/register/throughput profiling (roadmap
  Part 3 Phase B) cannot proceed until NCCS grants non-admin GPU
  performance-counter access, or an alternate profiler/approach
  (`compute-sanitizer`, static `nvcc --resource-usage`) is substituted.
- **Six-rank `--mpi-halo` host-staged path: UNRESOLVED, separate caveat.**
  This did not run cleanly in this smoke test (§3). It does not block the
  nsys/ncu conclusions above (both were independently exercised via the
  same binary's single-process transport), but any future profiling
  targeting the real six-rank/six-tile production decomposition must first
  resolve this — likely by supplying the same fixture arguments
  (`--grid-metrics`, `--sp0-state`, `--sp1-boundary`, `--akap-exner`) that
  the only prior validated `--mpi-halo` reference used.
