# Prompt 8 — CUDA Graphs Prototype for the `dgrid_dynamics` Acoustic Substep

_Written 2026-09-03. Scope: prototype CUDA Graph capture/replay for one
acoustic substep (32×32, R8, resident, six tiles) as the first optimization
experiment following the Prompt 7 baseline. Result: **BLOCKED** — both on
numerical correctness and on runtime stability. No numerical code was changed._

## 1. Git State

- Branch: `cuda/gpu-bringup`, same uncommitted Prompt 6B tree as
  `CURRENT_STATE.md` §1, plus this pass's `--cuda-graph` opt-in addition
  (`fv3_gpu_set_cuda_graph_mode`, gated capture of the `dgrid_dynamics` stage
  only; halo/seam exchange stays outside the graph — it is cross-process,
  host-synchronized, and not graph-capturable).
- Default (non-graph) execution path: unchanged. `--cuda-graph` is opt-in.
- Nothing committed or pushed this pass.

## 2. What Was Tried

Graph-capture boundary: **`dgrid_dynamics` stage only**, one capture per
benchmark repeat, replayed for the repeat's timed iterations. Halo/seam
exchange (cross-process MPI-rank-per-tile, CUDA events, host barriers) is
excluded from the graph by design — already established in a prior pass as
not capturable.

Job 58229236 (node not recorded in excerpt, 4m34s elapsed, all Slurm exit
codes 0) ran, in order: R8 ctest, a 2-iter smoke, a 32×32 resident perf run
with `--cuda-graph` OFF, the same with `--cuda-graph` ON, a single-call
(`--warmup 0 --iters 1`) `dyn_core_exit` field dump with graph OFF and ON, a
binary diff between the two dumps, and one `nsys` profile in graph mode
(reduced counts).

## 3. Finding 1 — Graph capture mechanically succeeds

`cudaStreamBeginCapture`/`cudaStreamEndCapture`/`cudaGraphInstantiate` all
return success on every capture (one capture per repeat per tile,
`capture+instantiate_ms` ranging ~1–43 ms depending on repeat/tile). A single
graph-mode dyn_core call completes cleanly (RC 0) and produces a full
`dyn_core_exit` dump on all 6 tiles.

**Graph capture: PASS** (as a mechanical/API-level statement only — see
Finding 2 for why this does not mean the graph is usable).

## 4. Finding 2 — Numerical divergence on a single call (STOP condition)

Comparing the graph-OFF and graph-ON `dyn_core_exit` dumps from the **same
single dyn_core call** (`--warmup 0 --iters 1 --repeats 1`, no repeated
replay involved) shows every one of the 13 exit fields DIFFERS, on every
tile, far beyond the Gate C2 tolerance (SMALL ≤1e-8):

| field | max_abs (tile0) | max_rel (tile0) |
|-------|------------------|-------------------|
| `delp` | 4.437626e+01 | 1.201025e-03 |
| `delz` | 7.705607e-01 | 2.039089e-04 |
| `omga` | 9.999611e+00 | **1.965809e+00** |
| `pe`   | 1.047333e+02 | 1.046238e-03 |
| `peln` | 1.046785e-03 | 9.091434e-05 |
| `pk`   | 8.024646e-03 | 2.990368e-04 |
| `pkz`  | 1.149498e-02 | 4.562679e-04 |
| `pt`   | 3.570103e-04 | 3.115752e-05 |
| `ua`   | 2.575967e+01 | **1.026256e+00** |
| `u`    | 2.581822e+01 | **1.133594e+00** |
| `va`   | 2.575967e+01 | **1.911090e+00** |
| `v`    | 1.659884e+01 | **1.266784e+00** |
| `w`    | 1.297556e-01 | **1.996703e+00** |

Same pattern on tiles 1–5 (e.g. `omga` max_rel 1.965–1.999 across all six
tiles; `delp` max_rel consistently ≈1.20e-03). `max_rel` near 1–2 on the wind
fields (`u`, `v`, `ua`, `va`, `w`, `omga`) means the two results are
effectively uncorrelated, not a subtle round-off or ordering difference.

Per this task's own instruction — **"STOP and investigate ordering only if
numerically significant differences appear; do not loosen tolerance"** — this
condition is triggered. Tolerance was not loosened; the divergence is
reported as a genuine defect requiring investigation, not accepted.

**Numerical equivalence: FAIL.**

Leading hypothesis (not yet tested): the graph captures `dgrid_dynamics`
kernel launches bound to specific device pointers/parameters at capture time;
if any of those bindings (buffer generation in a ping-pong scheme, an
iteration-dependent scalar, or a stream-ordering assumption relative to the
halo exchange that runs *outside* the graph between substeps) differ between
the captured instant and a later replay — or even between capture and the
first replay — the replayed kernels can read stale or wrong data. The wind
fields (`u`/`v`/`ua`/`va`) being the most divergent is consistent with a halo
dependency being violated, since those are exactly the fields the seam/halo
exchange feeds.

## 5. Finding 3 — Sustained replay is unstable (separate from Finding 2)

The full timed run (`--warmup 20 --iters 100 --repeats 5`, graph mode)
completed repeat 1 (max-across-tiles 193.730 ms/call — see §6) but crashed at
repeat 2:

```
FAIL: dyn_core_step (mode A) returned 3 api=[launch_halo_transfer]
      msg=[an illegal memory access was encountered]
FAIL: dyn_core_step (mode A) returned 2 api=[operator()]
      msg=[the halo peer group lost a member mid-exchange]   (×5, other tiles)
FAIL: repeat 2 mode resident did not complete cleanly
FAIL: tile 1 fv3_gpu_create returned 3: an illegal memory access was encountered
FAIL: six-tile setup did not complete (repeat 3, mode resident)
... (repeats 4, 5 same failure — tile 1's GPU context stayed corrupted)
=== 3A.11 CUDA perf benchmark: FAIL ===
```

Tile 1 hit an illegal memory access inside the graph-replayed halo transfer
path, which cascaded into the other five tiles losing their halo peer (the
cross-process barrier protocol has no fault tolerance for a dead peer), and
then tile 1's GPU context stayed corrupted for the remainder of the job (every
subsequent `fv3_gpu_create` on tile 1 failed the same way). Notably, a smaller
nsys profiling run of the same graph-mode path (`--warmup 5 --iters 20`, 25
total calls) did **not** crash — consistent with this being a state-reuse
issue that manifests only after enough repeated capture/replay cycles, not an
immediate one-shot failure.

**Graph reuse across calls: FAIL.**

## 6. Timing (confounded — see caveat)

| mode | repeat(s) completed | max-across-tiles ms/call |
|------|----------------------|----------------------------|
| no graph | 1–5 (all) | 206.829, 210.335, 211.090, 208.673, 213.963 → **median 210.335** |
| graph | 1 only (crashed at repeat 2) | **193.730** (single data point, not a median) |

**Caveat — do not treat either number as a clean baseline.** Both are ~8×
the Prompt 7 reference (26.009 ms/call, same grid/config, job 58226917,
`mpirun -np 6` with no `--oversubscribe`, `--ntasks` unrecorded in this
excerpt). This job's sbatch script (`perf_rerun_logs/prompt8_graph.sbatch`)
requests `--ntasks=1 --cpus-per-task=12` and runs `mpirun -np 6
--oversubscribe`, which was a required fix for an earlier "not enough slots"
error but does not guarantee correct CPU affinity for 6 independent MPI
ranks. This resourcing mismatch is suspected — not confirmed — to be a
timing confound, and is also a plausible contributor to Finding 3 (host
scheduling delays could expose a synchronization race between graph replay
and the cross-process halo exchange that would not manifest under the tighter
timing of a properly-resourced job). **Do not compare this pass's absolute
timings against Prompt 7's 26.009 ms/call as if they were commensurate.**
The 193.730 vs. 210.335 relative comparison (single repeat each) is too
thin and too confounded to support any speedup claim either way.

## 7. nsys Profile (graph mode, reduced counts)

Job section completed successfully (unlike a prior broken attempt), producing
`perf_rerun_logs/nsys_32x32_graph.nsys-rep` (191.5 MB),
`nsys_32x32_graph.sqlite` (509.9 MB), and 4 CSV summaries. One CSV timing line
from this reduced run (25 calls, did not crash):
`ms_per_step=177.836`. Detailed kernel-count/API-overhead/memcpy extraction
from the CSVs was not completed this pass (deprioritized once Finding 2 forced
a stop) — the raw artifacts are preserved under `perf_rerun_logs/` if needed
later.

## 8. Regression Checks

- R8 ctest: **PASS** (job 58229236, `=== CTEST RC: 0 ===`).
- Default non-graph synthetic smoke: **PASS** (`=== SMOKE (no graph) RC: 0
  ===`).
- Graph mode remains strictly opt-in; default path is unaffected by this
  prototype's presence in the tree.

## 9. Numerical Algorithm Changed

**NO.** Only a capture/replay wrapper around the existing `dgrid_dynamics`
launch sequence was added, gated behind `--cuda-graph`. No kernel, no
numerical formula, and no default-path behavior was changed.

## 10. Fix Cycles Used

Two implementation/fix cycles were used in this pass, both mechanical
(sbatch job-script bugs unrelated to the graph logic itself), per the task's
"at most 2 implementation/fix cycles without asking" limit:

1. Added `--oversubscribe` to all `mpirun -np 6` invocations (fixed an
   OpenMPI "not enough slots" error from `--ntasks=1`).
2. Corrected `BENCH` executable path from a guessed source-mirroring path to
   the actual CMake build-root output path (`build_cuda_r8_perf/
   fv3_gpu_perf_bench`).

Per instructions, no third autonomous fix cycle was attempted. The graph
correctness/stability defects in §4–§5, and the suspected resourcing confound
in §6, are left for the user to direct.

## 11. Classification

**CUDA GRAPH PROTOTYPE BLOCKED — numerical divergence and replay instability
in the `dgrid_dynamics` capture boundary; requires runtime investigation
before further measurement.**

## 12. Recommended Next Steps (not implemented this pass)

1. Investigate whether `dgrid_dynamics`'s device-pointer/parameter bindings
   at graph-capture time correctly account for buffer rotation and the halo
   data the stage depends on — this is runtime-architecture work, not a
   quick fix, and matches the task's own "stop if graph support requires
   major runtime redesign" condition.
2. Separately, re-run the no-graph 32×32 baseline with a corrected resource
   request (e.g. `--ntasks=6 --cpus-per-task=2`, no `--oversubscribe`) to
   determine whether the ~8× timing anomaly in §6 is a job-configuration
   artifact or a real regression, before trusting any absolute timing number
   from this pass.
3. If CUDA Graphs are dropped as a near-term target, the Prompt 7 handoff's
   other two candidates remain available: kernel fusion/batching of the many
   small repetitive kernels, and halo/corner-service kernel-family
   consolidation (~18% of kernel time).

## 13. Artifacts (not committed)

- `perf_rerun_logs/prompt8_graph.sbatch`,
  `perf_rerun_logs/prompt8_graph_58229236.out` (and the earlier failed
  `..._58229121.out`, `..._58229017.out`)
- `perf_rerun_logs/nsys_32x32_graph.{nsys-rep,sqlite}`,
  `perf_rerun_logs/nsys_graph_cuda_{api_sum,gpu_kern_sum,gpu_mem_size_sum,gpu_mem_time_sum}.csv`
