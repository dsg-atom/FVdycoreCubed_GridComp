# Prompt 12 — `a2b4_global_corner_kernel` Compute-Efficiency Tuning

Status: **COMPLETE — MODERATE VALUE, ncu PROFILING PLATFORM-BLOCKED.**
Job 58238626 (`perf_rerun_logs/prompt12_a2b4_tuning.sbatch`), node `warpa010`,
32×32/R8/npz=10/n_split=3/six tiles. Git HEAD at start of this pass: `7dda37b`
("Optimize FV3 CUDA halo edge launches" — Prompt 11's `--fuse-halo-edges`
work, committed between sessions, not by this pass). This pass's change is
**uncommitted** at time of writing (see repo `git status`); no commit or push
was requested.

## 1. Baseline re-establishment (Section 1)

Reference mode: `--fuse-halo-edges` (Prompt 11's validated best single
configuration; `--fuse-corner-fill` deliberately excluded per this task's own
instruction, since Prompt 11 measured it as very slightly slower combined).

32×32/R8/resident, 20 warmup/100 iters/5 repeats, pre-optimization kernel
(reverted to HEAD via `git checkout --` before any edit):

| repeat | ms/call |
|---|---|
| 1 | 22.146570 |
| 2 | 22.807953 |
| 3 | 22.779854 |
| 4 | 22.743409 |
| 5 | 22.259840 |

Median: **22.743409 ms/call**, spread (max−min) 0.661 ms. Within ~2.3% of
Prompt 11's own recorded reference (23.2813 ms/call, different node/session) —
consistent, not badly inconsistent. Section 1's reproducibility gate: **PASS**,
proceeded to optimization.

## 2. ncu profiling — PLATFORM-BLOCKED

Both scoped `ncu` invocations (`--kernel-name a2b4_global_corner_kernel
--launch-count 30`, sections `LaunchStats`/`Occupancy`/`SchedulerStats`/
`WarpStateStats`/`MemoryWorkloadAnalysis`/`ComputeWorkloadAnalysis`) returned
RC=1. The `--log-file` output (`perf_rerun_logs/ncu_p12_before.csv`,
`ncu_p12_after.csv`) shows the actual cause in both cases:

```
==PROF== Connected to process <pid> (.../fv3_gpu_perf_bench)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0. ...
==PROF== Disconnected from process <pid>
```

This is `NVreg_RestrictProfilingToAdminUsers`-style GPU performance-counter
access restriction, a **platform/driver permission constraint** on this
Discover `gpu_a100` allocation, not a flag error, not a kernel-name mismatch,
and not a defect in this task's script. The wrapped application itself ran to
completion normally in both cases (full benchmark stdout, `=== 3A.11 CUDA perf
benchmark: PASS ===`) — only the profiling counters were denied. No workaround
is available from an unprivileged job submission; escalation would require an
NCCS administrative change (loading the driver with counter access opened to
non-admin users, or a `setcap`/suid `ncu` install), which is out of scope for
this task to pursue unilaterally. **This is recorded as a confirmed platform
limitation**, not retried with adjusted flags (adjusting section/kernel-name
selection cannot bypass a permission wall).

## 3. Bottleneck classification — SOURCE-READING INFERENCE ONLY (not ncu-confirmed)

Because Section 2's profiler data could not be obtained, this classification
is based on direct source reading of the pre-optimization kernel, not measured
microarchitectural counters. Flagged as such, not presented as measured fact.

**Original launch config**: `a2b4_global_corner_kernel<<<(l+block1-1)/block1,
block1,0,stream>>>` with `block1=128` (shared file-scope constant) and
`l=levels=npz=10` ⇒ exactly **1 CUDA block of 128 threads**, of which only 10
threads do any work (`if(n>=levels)return;` guards the other 118) — **1 of the
A100's 108 SMs is used**, and even that SM's one active warp is only 10/32
lanes active. Within each of the 10 active threads, up to 4 physical-corner
branches (`if(c.sw)`/`se`/`ne`/`nw`) execute **serially**, each performing
`r3*(a2b_extrapolate + a2b_extrapolate + a2b_extrapolate)` — 3 calls, each
issuing 2 calls to `a2b_great_circle` (transcendental-heavy: `sin`, `cos`,
`asin`, `sqrt` in `accumulator_real` precision) — up to **24 serialized
great-circle evaluations per thread** when all 4 corners are active, which is
the typical case for this project's one-tile-per-cube-face six-tile
architecture (`consistent_corners` ties all 4 corner flags true whenever a
tile's domain touches all 4 physical edges).

**Classification: SERIAL/LOW-PARALLELISM (primary), OCCUPANCY-BOUND
(secondary).** The kernel's cost is not driven by memory bandwidth or
instruction throughput in any per-SM sense — it is driven by having almost the
entire device idle while one SM churns through a long serial chain of
transcendental math. This matches Prompt 11's own carried-forward
recommendation ("occupancy/shared-memory tuning or algorithmic restructuring
of the corner-extrapolation math itself").

## 4. Kernel/launcher inspection (Section 4)

Read in full: `a2b4_global_corner_kernel` (definition), its sole launch site
inside `launch_a2b_ord4_global_cuda`, `a2b_extrapolate`, `a2b_great_circle`,
`PhysicalCorners`/`HorizontalBounds`/`A2BGeometry`. No other a2b operators
were inspected (out of scope per Section 4). No formulas were touched.

Observed inefficiency: the four corner computations are fully independent of
each other (each writes a distinct output cell, reads distinct input cells)
and are only serialized because they are hard-coded as four sequential `if`
blocks inside one thread. No shared memory is used. No excessive register
pressure was evident relative to what the arithmetic requires. Grid geometry
(1 block, `block1=128` threads, 10 active) is the single dominant defect
found.

## 5. Optimization selected and applied (Section 5)

**Change**: reorganize the same four corner computations, unchanged formula,
unchanged floating-point summation order, across `blockIdx.x` (corner select,
`switch` on 0–3) × `blockIdx.y` (level, `n=blockIdx.y`, mirroring the
codebase's own pre-existing `grid1(n,levels)` idiom of using `blockIdx.y` for
levels), with 1 thread per block. New launch:
`a2b4_global_corner_kernel<<<dim3(4,l,1),1,0,stream>>>`. This is a **local,
low-risk change**: no shared-memory use, no numerical reordering within a
corner's own arithmetic, no topology redesign, no new host/device transfer,
and it is the single lowest-risk lever available given the "one local
optimization" and "no formula changes" bounds in Section 5 — a full
warp-level restructuring (e.g., parallelizing across threads within a corner's
6 `a2b_great_circle` calls) was considered and rejected as exceeding this
task's single-low-risk-change mandate; see §7 outlook.

File changed: `cuda/dyn_core/operators/wp_d_cuda.cu` (kernel body restructured
into a `switch(blockIdx.x)`, launch config changed from
`((l+block1-1)/block1, block1, 1)` to `(4, l, 1)` with 1 thread/block). Diff
saved at `perf_rerun_logs/prompt12_a2b4_kernel.patch`. **Numerical algorithm
changed: NO** — same per-corner formula, same summation order
(`r3*(A+B+C)`), same input/output addresses; only launch/dispatch granularity
changed.

## 6. Numerical validation — PASS, BITWISE IDENTICAL

`dyn_core_exit` dump, 13 fields (`u,delp,v,w,delz,pe,peln,pk,pkz,ua,va,pt,
omga`) × 6 tiles, compared before (pre-optimization, reverted kernel) vs after
(optimized kernel), for both the default path and `--fuse-halo-edges`:

- Default vs optimized: **78/78 IDENTICAL, 0 DIFFERS, 0 MISSING.**
- `--fuse-halo-edges` vs optimized+`--fuse-halo-edges`: **78/78 IDENTICAL, 0
  DIFFERS, 0 MISSING.**
- Combined total: **156/156 IDENTICAL.**

R8 CTest: **PASS, 22/22, both before and after** (`100% tests passed, 0 tests
failed out of 22`, twice in the log). Default synthetic smoke: PASS (both
before/after). `--fuse-halo-edges` synthetic smoke: PASS (both before/after).
All 12 benchmark invocations in the job printed `=== 3A.11 CUDA perf
benchmark: PASS ===`, all six tiles finite on `u,v,delp,pt,w,delz`, `delp`
positive over the owned box.

Note on an unrelated, pre-existing harness diagnostic: every invocation
(before and after) printed `u max|delta| = 6.352841e+00 CHANGED` /
`delp max|delta| = 4.436471e+01 CHANGED` with **identical values** in both the
before and after runs — this is the benchmark's own internal drift check
against a fixed stored reference (unrelated to this pass's change, since the
values do not move between before/after) and is not a Section 6 correctness
signal; the accepted correctness evidence is the binary dump diff above.

## 7. Performance validation (Section 7)

**Whole-dyn_core resident timing** (32×32, R8, `--fuse-halo-edges`, 20
warmup/100 iters/5 repeats):

| | median ms/call | spread (max−min) |
|---|---|---|
| before (pre-optimization) | 22.743409 | 0.661 |
| after (optimized a2b4) | **22.255549** | 0.104 |
| change | **−0.488 ms (−2.15%)** | — |

After-run values: 22.247640, 22.233925, 22.255549, 22.314776, 22.337748 —
tight spread (0.104 ms), well below the ~0.49 ms measured gap, so the
improvement is above the noise floor for this run.

**Kernel-level (a2b4-specific) timing**: not directly measurable without ncu
(§2, platform-blocked). No nsys profile was run in this pass (cost-control:
task explicitly bounds this pass to ncu-only profiling, no second profiling
tool). The whole-call figure above is the only performance evidence available
for this pass.

## 8. Decision threshold (Section 8)

Whole-dyn_core-call improvement: **2.15%**. Per the task's buckets (HIGH VALUE
≥3%, MODERATE 1–3%, LOW <1%, REGRESSION): **MODERATE.**

## 9. Iteration (Section 9)

**Zero fix cycles consumed.** The optimization validated bitwise-identical and
measured a real, reproducible-direction improvement on the first attempt; the
`ring_count`-style class of bug that hit Prompt 11 did not recur here (this
kernel has no analogous width/ring-count ambiguity). Iteration was stopped
after one implementation under the Section 9 stop condition "optimization
succeeds" — with two caveats justifying not spending further cycles chasing a
bigger local win on this same kernel: (a) ncu is platform-blocked (§2), so any
further local tuning would be unguided by real profiler evidence, contrary to
this project's evidence-first methodology; (b) the remaining headroom
identified by source reading (warp-level parallelization of the 6
`a2b_great_circle` calls per corner) is a larger, riskier restructuring that
exceeds Section 5's single-low-risk-change mandate — it would need to be
scoped as its own task with real occupancy data to justify it, not attempted
speculatively here.

## 10. Cost control (Section 10)

32×32/R8 only. No Fortran rerun, no R4, no c8/c16 sweep, no CUDA Graphs work
(not touched), no broad ncu sweep (kernel-name-scoped, 30-launch-limited, two
invocations only — both failed on permission, not on scope), one kernel only,
no new savepoints, no unrelated launch-fusion work.

## 11. Regression (Section 11)

R8 CTest: PASS (22/22, both before/after builds in the same job). Default
synthetic smoke: PASS (both). `--fuse-halo-edges` synthetic smoke: PASS
(both).

## 12. Outlook

`a2b4_global_corner_kernel`'s launch-count is already known (Prompt 11 §2) to
be too small (108/call) to matter as a launch-fusion target. This pass shows
its *compute*-side headroom via the most conservative possible fix
(block-level corner/level spread, unchanged per-thread work) is real but
modest (2.15% whole-call). A materially larger gain would require
warp-level restructuring (parallelizing the 6 great-circle evaluations per
corner across threads in a warp, with a small in-warp/shared-memory
reduction) — genuinely more invasive than this task's single-low-risk-change
bound, and unverifiable without ncu access to confirm occupancy/warp-stall
improvement. **Recommendation: do not continue local compute-tuning on this
one kernel further in an unguided way.** Either (a) request NCCS grant GPU
performance-counter access for this account/allocation so a real
occupancy/warp-stall-guided second pass on this kernel (or on the
higher-total-time `riem_solver3_kernel`/`riem_solver_c_kernel`, ~144–148 μs
avg each) becomes possible, or (b) shift to the other architectural class
already identified in Prompt 11's own outlook — kernel-time/occupancy tuning
of the individually expensive Riemann-solver kernels — once ncu access is
available. CUDA Graphs remain explicitly excluded from consideration per this
task's own instruction.
