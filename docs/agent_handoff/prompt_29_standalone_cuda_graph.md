# Prompt 29: Standalone CUDA Graph Production Path

## 1. Objective

Expose the already-validated `dgrid_dynamics` CUDA Graph capture/replay path
(Prompt 8/13) to `StandAlone_FV3_Dycore.x` on top of Prompt 28's production
config (MPS ON + `fuse_halo_edges` + `fuse_corner_fill`), and measure its
real whole-`dyn_core` benefit. No new kernel optimization, no expanded
capture region, no CUDA-aware MPI, no graph redesign.

## 2. Production Config Knob Added

The existing CUDA Graph ABI is a dedicated function
(`fv3_gpu_set_cuda_graph_mode(ctx, enabled, err)`), not a
`fv3_gpu_numerical_config` struct field like the two fusion flags — this
predates Prompt 28 and was already validated (Prompt 8/13). No new ABI
function or struct field was added; only the missing Fortran binding and a
call site were added, mirroring the Prompt 28 pattern exactly:

- `cuda/interface/fv3_gpu_abi_mod.F90`: added `public :: fv3_gpu_set_cuda_graph_mode`
  and its `bind(C)` interface (same style as the neighboring
  `fv3_gpu_get_capabilities`/`fv3_gpu_attach_mpi_halo_transport` interfaces).
- `cuda/interface/fv3_gpu_dyncore_bridge.F90`: new call, immediately after
  `fv3_gpu_create` (matching the runtime's own documented contract —
  "callers set this once, right after `fv3_gpu_create`"), reusing the
  existing `env_flag_enabled(name)` helper unchanged:
  ```fortran
  crc = fv3_gpu_set_cuda_graph_mode(ctx, &
       merge(1_c_int, 0_c_int, env_flag_enabled('RUN_CUDA_FV3_GRAPH')), err)
  call check(crc, err, 'fv3_gpu_set_cuda_graph_mode')
  ```
- Env var: `RUN_CUDA_FV3_GRAPH` (YES/1/TRUE, case-insensitive; unset/other =
  OFF, unchanged default behavior). Confirmed via `strings` on the rebuilt
  `StandAlone_FV3_Dycore.x` that the name is present in the binary.

No CUDA/C++ runtime source was touched — `fv3_gpu_set_cuda_graph_mode`
already existed in `fv3_gpu_runtime.cpp` (Prompt 8), including its existing
refusal behavior (`FV3_GPU_STATE_CONFLICT` if toggled after the first
capture) and its existing teardown at `fv3_gpu_finalize`
(`cudaGraphExecDestroy` on the persistent context's graph exec). **Numerical
algorithm changed: NO** — this is ABI-reachability/config plumbing only,
selecting between two already-validated execution paths.

## 3. Build

Rebuilt only the two edited `.F90` files and their dependents via
`cmake --build . --target StandAlone_FV3_Dycore.x` in
`GEOSfvdycore_fortran_perf/build` (Intel ifort/Intel MPI, unchanged
toolchain). No CUDA library rebuild was required (no C++/CUDA source
changed) — `build_cuda_r8_intelmpi` was rebuilt anyway as a sanity check and
produced zero new object changes. Refreshed the `StandAlone_FV3_Dycore.x`
copy staged in each of the three reused job directories
(`p24_c32_r8_cuda`, `p24_c32_r8_cuda_nsys`, `p28_c32_r8_cuda_guard`) — all
three previously held a stale pre-edit binary lacking the new symbol; `cp -f`
from `build/bin/` fixed this, verified via `strings` on all three.

## 4. Configuration Matrix and Numerical Guard

c32, R8, `npz=10`, `n_split=3`, `k_split=1`, `bdt=1200s`, 6 MPI ranks
(`FV_NX=1`/`FV_NY=6`), one A100/one node, CUDA MPS ON throughout (one MPS
session spanning both modes), `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` and
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` held fixed in both modes (Prompt 28's
best mode as the base). Mode A = Graph OFF (baseline), Mode B = Graph ON.
Job 58382771, both modes rc=0/0 Range_Warn/0 NaN, guard savepoint counts
2160/2160.

`tools/savepoints/compare_dcx_backends.py`, CUDA(Mode A) vs CUDA(Mode B),
10 dyn_core calls x 6 ranks x 17 fields, c32/R8:

| Stage | Result |
|---|---|
| `dcx` (dyn_core exit — the field that matters) | **BITWISE IDENTICAL**, worst max_rel = 0.000000e+00, all 17 fields, all 10 calls |
| `dce` (dyn_core entry) | `mfyL` shows the pre-existing, already-documented (§5/§9 of `CURRENT_STATE.md`) dce-entry-only 2-cell uninitialized-memory artifact — reproduced **identically** in both Graph OFF and Graph ON (same nonfinite pattern, same cell counts); not a new discrepancy, not caused by Graph |

**No new SIGNIFICANT numerical difference.** The one non-bitwise field is the
same known artifact already carried in `CURRENT_STATE.md`'s unvalidated
envelope, confirmed here to be independent of the Graph flag.

## 5. Replay Stability (§6 requirement)

Verified operationally, not just by source inspection: Mode B (Graph ON) ran
the 10-call guard pass plus 3 x 100-call timing reps (310 total dyn_core
calls, i.e. 310 x `n_split`=3 = 930 acoustic-substep graph replays after one
capture) with rc=0 in every rep, 0 Range_Warn, 0 NaN, no illegal-memory-access
or CUDA/MPI error string found in the job log (`grep` for
`illegal memory|cuda error|segmentation|MPI_ABORT|STATE_CONFLICT` returned
nothing). No stale-pointer or lost-halo-transport symptoms observed — halo
exchange is untouched by this change (still outside the captured region).
`dt`/`n_split` were held fixed for the whole job, so the runtime's own
refusal-on-change guard was not exercised operationally in this task (it is
unchanged Prompt-8 code, out of scope to re-test per §13).

## 6. Performance Measurement

Job 58382771, 3 reps/mode, 100 calls/rep, savepoints disabled during timing,
`fv_timing_mod`'s `DYN_CORE` timer (value reported as total seconds for the
100-call rep; ms/call = value/100*1000).

| Mode | rep1 | rep2 | rep3 | median ms/call | spread |
|---|---|---|---|---|---|
| A — Graph OFF (Prompt 28 best mode) | 25.448 | 25.589 | 25.526 | **25.526** | 0.55% |
| B — Graph ON | 24.255 | 24.300 | 24.206 | **24.300** | 0.39% |

Improvement: (25.526 − 24.300) / 25.526 = **4.80%**. All three Mode B reps
are faster than all three Mode A reps (non-overlapping ranges) — the effect
is small but reproducible, not noise. Mode A's 25.526 ms/call is consistent
with Prompt 28's own Mode C figure (25.464 ms/call, within 0.24%) —
confirms this job reproduces the known-good baseline.

## 7. nsys Trace (§9 gate: only if improvement >= 3%; 4.80% clears it)

One trace, Mode B (Graph ON), reusing `p24_c32_r8_cuda_nsys` unchanged
(10-call rank-0 trace, `--trace=cuda,osrt`, MPS ON). Job 58383079, rc=0/0
Range_Warn/0 NaN. Traced `DYN_CORE` = 31.72 ms/call (nsys overhead inflates
this above the untraced 24.300 figure, used only as this trace's own
percentage denominator, per the established convention).

| Bucket | Prompt 28 (fusion only, no Graph) | Prompt 29 (fusion + Graph ON) |
|---|---|---|
| `cudaLaunchKernel` calls/dyn_core | 2637 | **1802** (host-issued launches only) |
| `cudaGraphLaunch` calls/dyn_core | n/a | **3** (one per `n_split`=3 acoustic substep) |
| Total launch operations/dyn_core | 2637 | **1805 (−31.6%)** |
| `cudaLaunchKernel` API time | 15.81 ms/call (48.3%) | **12.30 ms/call (38.8%)** |
| `cudaGraphLaunch` API time | n/a | **0.26 ms/call (0.8%)** |
| Kernel execution (all distinct kernels) | 12.08 ms/call (36.9%) | **8.42 ms/call (26.5%)** |
| `cudaEventSynchronize` | 9.98 ms/call (30.5%) | **12.27 ms/call (38.7%)** |
| H2D+D2H device transfer | 0.618 ms/call (1.9%) | **0.619 ms/call (2.0%)** |

Launch operations fell steeply (−31.6%) and `cudaLaunchKernel` API time fell
in step (15.81 -> 12.30 ms/call), consistent with capturing the
`dgrid_dynamics` stage's individual kernel launches into 3 graph replays.
However `cudaEventSynchronize` **increased** in this trace (9.98 -> 12.27
ms/call) rather than falling or staying flat, largely offsetting the
launch-side savings in the whole-call wall-clock result. Per §11, this
bucket was not targeted or restructured. This is consistent with Prompt 28's
standing interpretation that the remaining synchronization cost is
GPU-producer-wait under MPS context-sharing (ranks blocked on a
time-sliced-shared A100), not host launch/orchestration slack — a cost
Graph capture (a host-side optimization) does not relieve, and in this
single trace appears to add trace-to-trace variance to. The kernel-execution
bucket's apparent drop (12.08 -> 8.42 ms/call) is also reported as measured
but not further investigated — kernel-level analysis is out of scope for
this task (§13).

## 8. Graph-Specific Checks (§10)

- Capture occurs once per context: consistent with `cudaGraphInstantiate`
  appearing exactly once (1 instance, 2.14 ms) in the Mode B trace, and
  `cudaStreamBeginCapture`/`cudaStreamEndCapture` each appearing exactly
  once — confirms capture happened on the first `dgrid_dynamics` occurrence
  only, not every call.
  the observed 3 `cudaGraphLaunch` calls in a 10-call trace, i.e.
- Replay on subsequent calls, no per-call recapture:
  1 graph launch per `n_split`=3-substep replayed 3x within each of the
  traced calls' dgrid_dynamics stage (930 total replays across the full
  310-call Mode B run in §5, versus one capture) — confirms replay, not
  recapture, dominates.
- Exec lifetime tied to context lifetime, clean teardown: unchanged from
  Prompt 8 source (`cudaGraphExecDestroy` inside `fv3_gpu_finalize`'s
  context teardown) — not re-verified operationally this task since the
  source path is untouched; the job completing all reps and exiting cleanly
  (rc=0 every time, no leak-related crash) is consistent with correct
  teardown.
- Refusal-on-incompatible-change: unchanged existing behavior
  (`FV3_GPU_STATE_CONFLICT` if `fv3_gpu_set_cuda_graph_mode` is called again
  after `dgrid_dynamics_graph_captured`); not exercised operationally here
  since `dt`/`n_split` were held fixed for the whole job (out of scope to
  perturb per §13's "small task" instruction).

## 9. Classification and Recommendation (§8/§12)

**4.80% < 5% -> classification LOW**, just under the MODERATE floor despite
being a real, reproducible (non-overlapping across all 3x3 rep pairs)
effect. Per the explicit instruction not to force a positive conclusion from
the historical Prompt 13 5.14% estimate: that estimate does **not** hold up
on the real standalone six-rank path once MPS and launch fusion are already
applied — the closest real measurement is 4.80%, which the acceptance
threshold classifies as LOW.

**Graph is NOT added to the recommended production mode.** Per §12's
explicit instruction ("If Graph benefit is LOW, do not enable it in the
recommended mode"), the Prompt 28 recommended optimized standalone CUDA mode
is unchanged: MPS ON + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` +
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`. `RUN_CUDA_FV3_GRAPH=YES` remains
available as an optional, non-default, independently-toggleable flag (default
OFF, unchanged default behavior) for any future re-measurement, but is not
part of the recommended default.

## 10. Next Bottleneck

`cudaEventSynchronize` (GPU-producer-wait under MPS context-sharing) is the
cost that neither Prompt 28's launch fusion nor this prompt's Graph capture
relieves — both are host-side/launch-side optimizations, and this bucket
tracks GPU-side contention on the shared, time-sliced A100. Per Prompt 26,
rank/GPU context-sharing redesign is NOT JUSTIFIED (MPS already reaches the
1-context ceiling for free) and CUDA-aware MPI remains LOW priority (device
transfer stays ~2% of `DYN_CORE`). No further standalone launch/orchestration
optimization is currently justified. The two open tracks are: (a) production
GEOS integration closures (`MOIST_CAPPA`/`USE_COND`, `a_imp <= 0.999` knobs,
Intel-ABI verification — CURRENT_STATE §9/§11), or (b) accept the current
MPS + fuse_halo_edges + fuse_corner_fill configuration as the final optimized
standalone CUDA mode and move to production integration.

## 11. Regression (§15)

Only ABI/config plumbing changed (a new Fortran binding to an
already-existing, already-validated C function, plus one call site); no
shared CUDA runtime behavior was modified. Per §15's rule, no R8 CTest rerun
was performed — the Graph OFF/Graph ON smoke runs plus the numerical guard
(§4) satisfy the required regression check for plumbing-only changes.

## 12. Files Touched This Prompt

- `cuda/interface/fv3_gpu_abi_mod.F90` (new `fv3_gpu_set_cuda_graph_mode`
  public declaration + `bind(C)` interface).
- `cuda/interface/fv3_gpu_dyncore_bridge.F90` (new call site right after
  `fv3_gpu_create`, `RUN_CUDA_FV3_GRAPH` env var via `env_flag_enabled`).
- `GEOSfvdycore_fortran_perf/experiments/p29_cuda_graph.sbatch`,
  `p29_nsys_graph_on.sbatch` (new job scripts, reusing existing job
  directories `p24_c32_r8_cuda`, `p24_c32_r8_cuda_nsys`,
  `p28_c32_r8_cuda_guard` unchanged).
- `docs/agent_handoff/prompt_29_standalone_cuda_graph.md` (this file).
- `docs/agent_handoff/CURRENT_STATE.md` (updated, compacted).

Not committed/pushed — awaiting explicit user request per project policy.
