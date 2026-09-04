# Prompt 7 — Post-Gate-C2 CUDA Performance Baseline + Profiling Pass

_Written 2026-09-03. Scope: re-establish the CUDA dyn_core performance baseline
after the Gate C2 numerical-validation fixes, and run one bounded nsys
profiling pass to classify the current performance regime. No optimization was
performed. No numerical code was changed._

## 1. Git State

- Branch: `cuda/gpu-bringup`
- HEAD used for this benchmark: `191157f3a51fafbe0f3202e5f622cba96a04d0a7`
  ("Validate CUDA dyn_core against authoritative FV3 Fortran")
- HEAD matches `origin/cuda/gpu-bringup` exactly — already pushed (the
  "ahead by 4, not pushed" text previously in `CURRENT_STATE.md` §1 was stale;
  live git state is authoritative per `CLAUDE.md`).
- `build_cuda_r8_perf/` (Release, `CMAKE_CUDA_ARCHITECTURES=80`) rebuilt via
  `cmake --build build_cuda_r8_perf --target fv3_gpu_perf_bench -j 12`: no
  targets recompiled, confirming the existing benchmark binary already matches
  this HEAD (Gate C2 code).
- No source files were modified for this pass. Untracked scratch files at the
  repo root and the uncommitted Prompt 6B work described in
  `CURRENT_STATE.md` §1 are unrelated to this benchmark and were not touched.

## 2. Harness Reused (no new harness written)

`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`, CMake target
`fv3_gpu_perf_bench`, same harness used for the 3A.11/3A.12 baseline. No
validation-only flags used (`--grid-metrics`, `--sp0-*`, `--sp1-boundary`,
`--akap`, `--akap-exner`, `--damp-n-sponge` all omitted) — this exercises the
**default synthetic/performance path**, which is the same path the 3A.11V
baseline measured, except that four Gate C2 fixes now change that default path
unconditionally (see `CURRENT_STATE.md` §4 #9 `grid_lonlat` store order, #10
`cv_air`, #11 `n_con`, #13 `nf_ke`), plus E11, the heat accumulator, the new
corner service, and the `omga` finalization all add real work to it.

Methodology, unchanged from 3A.11V: `--warmup 20 --iters 100 --repeats 5
--mode both`, R8, `npz=10 n_split=3`, six MPI ranks/tiles, one A100 GPU.
Per-repeat statistic is the max elapsed time across the six tiles (harness
built-in, unchanged).

Command (Slurm job 58226816, node `warpa013`, A100-SXM4-40GB,
SM 1410 MHz / mem 1215 MHz / 400 W, 2 min elapsed, exit 0):

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv comp/gcc/12.3.0 mpi/openmpi/4.1.6/gcc-12.3.0 nvidia/nvhpc-byo-compiler/24.7
for GRID in 8 16 32; do
  mpirun -np 6 ./build_cuda_r8_perf/cuda/dyn_core/benchmarks/fv3_gpu_perf_bench \
    --grid $GRID --npz 10 --n_split 3 --warmup 20 --iters 100 --repeats 5 \
    --mode both --hardware A100-SXM4-40GB
done
```

Full log: `perf_rerun_logs/perf_bench_58226816.out` (not committed).

## 3. Correctness During Benchmarking

- 1080 `finite(...)` check lines, 180 `positive over owned box` check lines,
  zero occurrences of `NaN`, `Inf`, or `FAIL` anywhere in the log.
- Three `=== 3A.11 CUDA perf benchmark: PASS ===` banners (one per grid).
- `WIND_SCALE=1 HORIZ_SCALE=1 DAMPV_SCALE=1 D4_SCALE=1` banners confirm the
  default (non-scaled) synthetic path, matching the 3A.11V methodology.
- Slurm exit code 0 for all three grids.
- This is a runtime smoke/sanity check only — **not** a Gate C2 savepoint
  regeneration, and no numerical-validation claim is made or implied here.

**Correctness smoke: PASS.**

## 4. CUDA R8 Timing Results (median of 5 repeats, ms/call)

| Grid  | resident | transfer-inclusive |
|-------|----------|---------------------|
| 8×8   | 29.918   | 30.799              |
| 16×16 | 28.799   | 28.767              |
| 32×32 | 26.009   | 28.075              |

Reproducibility: resident spread (max−min)/median across the 5 repeats was
0.77% (8×8), 0.51% (16×16), 1.94% (32×32). Transfer-inclusive spread was
0.41% (8×8), 5.85% (16×16), 0.60% (32×32); the 16×16 transfer figure is
elevated because repeat 1 (29.862 ms) is a clear outlier against repeats 2–5
(28.18–29.17 ms) — consistent with a one-time clock/thermal-boost transient
noted previously in the 3A.11V baseline, not evidence of an unstable
measurement. Median (not mean) is used as the primary statistic specifically
because of this kind of outlier.

**Timing reproducibility: PASS.**

## 5. Fortran Comparison — REUSED, not remeasured

The `GEOSfvdycore_fortran_perf` fixture was **not** rerun this pass. Reused
values (12.730 / 16.704 / 30.261 ms/call for 8×8/16×16/32×32) are justified as
still valid because:

- Fixture git HEAD (`8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`) is unchanged
  from the recorded value in `milestone_3a12_fortran_perf_baseline.md`.
- The fixture's `@fvdycore` SHA, namelist config (`grid_type=0`, `hord_*=6/-6`,
  `vtdm4=0.01`, `n_split=3`, `k_split=1`, `a_imp=1.0`) and executable are
  unchanged — none of the Gate C2 fixes touch Fortran source or the fixture's
  runtime configuration; every Gate C2 fix (`CURRENT_STATE.md` §4) is CUDA-side
  only.
- Re-running the Fortran fixture costs a multi-day `JOB_SGMT`-driven `sbatch`
  cycle for no expected numerical change — not justified by task scope
  (section 5 of the task explicitly permits reuse when unchanged/reproducible).

**Fortran baseline: REUSED.**

| Grid  | Fortran ms/call | CUDA resident speedup | CUDA transfer speedup |
|-------|-----------------|------------------------|-------------------------|
| 8×8   | 12.730          | 0.426×                 | 0.413×                  |
| 16×16 | 16.704          | 0.580×                 | 0.581×                  |
| 32×32 | 30.261          | 1.163×                 | 1.078×                  |

(speedup = Fortran ÷ CUDA; >1× means CUDA is faster)

## 6. Change vs. Old (STALE, pre-Gate-C2) CUDA Baseline

Old baseline (3A.11V, node `warpa003`, pre-fidelity-fix code):

| Grid  | old resident | new resident | Δ%     | old transfer | new transfer | Δ%     |
|-------|--------------|---------------|--------|----------------|---------------|--------|
| 8×8   | 28.332       | 29.918        | +5.60% | 29.358         | 30.799        | +4.91% |
| 16×16 | 29.690       | 28.799        | −3.00% | 30.896         | 28.767        | −6.89% |
| 32×32 | 28.827       | 26.009        | −9.78% | 31.178         | 28.075        | −9.95% |

Despite four fidelity fixes that add real per-call GPU work on the default
path (`n_con`: 1→10 heated levels; `nf_ke`: 1→3 del2-smoothing passes plus a
previously-missing corner rotation; E11 seam resolve; the heat accumulator;
`omega_pem_snapshot`/`omega_finalize`, two new kernels per substep), resident
cost did not increase at 16×16/32×32 and only rose 5.6% at 8×8. This is
consistent with (not proof of) the launch-latency-bound regime identified in
3A.11V and confirmed by profiling below: added per-cell arithmetic is dwarfed
by fixed per-launch orchestration cost at these grid sizes. Note the old and
new runs used different A100 nodes (`warpa003` vs `warpa013`), so node-to-node
variance is a confound; no strong causal attribution beyond "no regression, and
plausibly a small net improvement" is made.

## 7. nsys Profiling Pass (one bounded case, no optimization)

Representative case: **32×32, resident mode** (largest grid; profiling used
reduced counts — `--warmup 5 --iters 20 --repeats 1 --mode resident` — to keep
nsys tracing overhead and output size bounded; this is a documented
methodology deviation from the full timing run, used for profiling only).

Job: Slurm 58226917, node `warpa011`, 24 s elapsed, exit 0.
`nsys` is not on `PATH` under `nvidia/nvhpc-byo-compiler/24.7`; located and used
directly at
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/Nsight_Systems/bin/nsys`
(version `2024.4.1.61-244134315967v0`) rather than switching module families,
per the platform doc's restriction against changing compiler/CUDA module
families to work around a tooling gap.

Reports extracted via `nsys stats --report {cuda_gpu_kern_sum,cuda_api_sum,
cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum} --format csv`.

### 7.1 Kernel launch volume

- Total `cudaLaunchKernel` calls in the trace (25 calls = 5 warmup + 20 timed):
  **443,250**.
- Per dyn_core call: **17,730** kernel launches.
- Per acoustic substep (n_split=3, 6 tiles): **5,910**.
- Per tile per substep: **985**.

### 7.2 CUDA API time breakdown (`cuda_api_sum`)

- `cudaLaunchKernel`: **90.6%** of CUDA API time (median ≈8.14 µs/launch).
- `cudaMemcpyAsync`: 5.4%.
- `cudaStreamWaitEvent`: 2.1%.
- `cudaEventRecord`: 1.5%.
- Remaining API calls: <1% combined.

### 7.3 Kernel time distribution (`cuda_gpu_kern_sum`)

- Total kernel busy time across the trace: ≈1.857 s.
- **No single kernel exceeds 6.3%** of total kernel time (largest:
  `a2b4_global_corner_kernel`).
- The two largest individual "real compute" kernels are the vertical Riemann
  solvers, `riem_solver3_kernel` and `riem_solver_c_kernel`
  (≈140–149 µs average each, but only 3.4–3.6% of total kernel time each —
  called relatively few times per call, unlike the thousands of small
  transport/halo kernels).
- Halo/corner-service kernels (`copy_corners_kernel`, `pack_edge_kernel`,
  `unpack_edge_kernel`, `fill_4corners_kernel`, `fill2_4corners_kernel`)
  together account for ≈18% of kernel time — real, but not dominant, and
  spread across many small launches rather than concentrated in one.
- No kernel is individually expensive enough to warrant a targeted `ncu` deep
  dive per the task's own criterion (section 8: only run `ncu` if nsys shows
  1–2 kernels consuming a meaningful fraction of dyn_core time).

### 7.4 Memory / transfer contribution (`cuda_gpu_mem_time_sum`,
`cuda_gpu_mem_size_sum`)

- Total memory-op time: ≈60.2 ms across the trace — small relative to
  ≈1.857 s of kernel busy time.
- Per dyn_core call: ≈2.41 ms.
- **91.3%** of memory-op time is Device-to-Device (D2D) copy, not
  Host↔Device — expected even in `resident` mode, because the six-tile
  GPU-side halo exchange (real communication, not a stub, per
  `project_architecture.md`) moves data device-to-device between tiles' own
  buffers, not to/from the host.

### 7.5 Note on aggregated time vs. wall clock

Summed CUDA-API/kernel time across the trace exceeds raw wall-clock time by
several-fold. This is expected, not an error: the six tiles execute
concurrently (one MPI rank/stream/host-thread per tile on the same GPU), so
per-tile API and kernel time is additive in the CSV summaries but overlapped in
wall clock. It does not change the per-category proportions used for
classification above.

### 7.6 Classification

**Primary performance classification: LAUNCH/ORCHESTRATION-BOUND.**

Evidence: (a) kernel-launch count (17,730/call) is far larger than would be
expected if compute or transfer dominated; (b) `cudaLaunchKernel` alone is
90.6% of API time; (c) no single kernel or kernel family concentrates enough
time to justify kernel-level tuning; (d) memory-op time is a small fraction
(~3%) of kernel time; (e) resident cost is nearly grid-size-independent
(26.0–29.9 ms across an 16× cell-count range), the same signature 3A.11V used
to establish the hypothesis originally. **The launch/orchestration-bound
hypothesis from the prior baseline still holds** after the Gate C2 fixes.

### 7.7 ncu

**ncu: NOT JUSTIFIED.** No kernel or small kernel group dominates dyn_core
time; per the task's own gating rule, a kernel-level `ncu` deep dive would not
change the classification or the next optimization target.

## 8. Top 3 Optimization Opportunities (for a future task — not implemented here)

1. **Kernel fusion/batching of the many small, repetitive per-tile kernels**
   (transport, damping, corner/edge kernels) — the highest-leverage target
   given ~17,730 launches/call and 90.6% API time in `cudaLaunchKernel` itself.
2. **CUDA Graphs** to capture and replay the largely-static per-substep launch
   sequence, amortizing per-launch issue latency (~8 µs median) across the
   whole graph instead of paying it per kernel.
3. **Halo/corner-service pipeline consolidation** — `copy_corners`/
   `pack_edge`/`unpack_edge`/`fill_4corners`/`fill2_4corners` together are
   ≈18% of kernel time spread over many small launches; consolidating this
   family is a narrower, lower-risk fusion candidate than the general-transport
   kernels in (1).

No numerical code was changed to reach any of these conclusions.

## 9. Numerical Code Changed

**NO.** This pass performed measurement and profiling only. No kernel fusion,
stream changes, CUDA Graphs, halo-strategy changes, memory-layout changes, or
numerical changes were made.

## 10. Next Recommended Milestone

Prototype CUDA Graphs capture for one acoustic substep on the 32×32 case as
the first concrete optimization experiment, since it requires no kernel
rewrites and directly targets the confirmed launch-latency bottleneck; use the
per-substep launch count (5,910) and kernel time (≈1.857 s / 25 calls ≈74 ms
per call) recorded here as the pre-optimization reference point.

## 11. Artifacts (not committed)

- `perf_rerun_logs/perf_bench.sbatch`, `perf_rerun_logs/perf_bench_58226816.out`
- `perf_rerun_logs/nsys_profile.sbatch`, `perf_rerun_logs/nsys_58226917.out`
- `perf_rerun_logs/nsys_32x32_resident.nsys-rep` (28,018,687 bytes, raw trace)
- `perf_rerun_logs/nsys_{api,kern,memtime,memsize}_sum.csv`

Per `docs/agent_instructions/discover_platform.md`, none of these are
committed (Slurm logs, sanitizer/profiler logs, and generated artifacts are
excluded from git).
