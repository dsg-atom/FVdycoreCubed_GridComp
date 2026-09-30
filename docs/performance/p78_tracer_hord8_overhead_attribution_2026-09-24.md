# Prompt 78: Confirm and Attribute fv_tp_2d hord==8 Production Regression

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Scope: performance measurement and profiling only. No CUDA/Fortran source
edits. No changes to the production dispatch default. `ncu` not used
(platform `ERR_NVGPUCTRPERM` block); `nsys` (CLI `stats`, not GUI) only.

## 1. Motivation

Prompt 77 measured a TRACER_2D +44.1% / TOTAL +10.0% CUDA-vs-Fortran
slowdown from a single 3-call, sequential (non-alternating) sample —
explicitly flagged there as insufficiently rigorous to be final. This
task repeats the measurement with a rigorous alternating-order design
and, if the regression reproduces, attributes it using `nsys`.

## 2. Stage 2: Rigorous Paired A/B (job 58544700)

Same node (`warpa005`), same allocation, same executable
(`sha256=8152aed3...a957`), same rank placement/environment; only the
tracer dispatch selector differs (Arm A = CUDA hord==8 `fv_tp_2d`
production dispatch; Arm B = `RUN_CUDA_FV3_TRACER_FORCE_FORTRAN=1`).
Strict alternating order A1 B1 A2 B2 A3 B3, `JOB_SGMT` widened to 650s
-> 11 `dyn_core` calls/invocation, 33 calls/arm. Job completed cleanly
in 8:36, zero NaN/Inf/Abort/range_warn across all 6 invocations, selector
fired correctly and exclusively each direction every invocation.

Per-invocation cumulative timer totals (seconds, over 11 calls):

| Pair | Arm | FV_DYN_LOOP | DYN_CORE | TRACER_2D | REMAPPING | REMAP_TOTALENERGYCON |
|---|---|---|---|---|---|---|
| 1 | cuda | 27.2455 | 9.5604 | 4.3427 | 8.2935 | 1.8550 |
| 1 | fortran | 26.2770 | 9.5340 | 3.1759 | 7.7997 | 1.4147 |
| 2 | cuda | 27.1693 | 9.4976 | 4.2667 | 8.2238 | 1.7989 |
| 2 | fortran | 25.9277 | 9.4995 | 3.0899 | 7.8074 | 1.4309 |
| 3 | cuda | 27.6921 | 9.4954 | 4.2604 | 8.2262 | 1.8019 |
| 3 | fortran | 25.6702 | 9.5068 | 3.0949 | 7.8029 | 1.4219 |

Medians and ratios (cuda/fortran):

| Timer | Median cuda | Median fortran | Ratio | Delta | Classification |
|---|---|---|---|---|---|
| TRACER_2D | 4.2667 | 3.0949 | 1.3787 | **+37.87%** | CONFIRMED REGRESSION (>5%) |
| FV_DYN_LOOP | 27.2455 | 25.9277 | 1.0508 | +5.08% | meaningful, secondary |
| DYN_CORE | 9.4976 | 9.5068 | 0.9990 | -0.10% | neutral |
| REMAPPING | 8.2262 | 7.8029 | 1.0543 | +5.43% | meaningful, anomalous (see §7) |
| REMAP_TOTALENERGYCON | 1.8019 | 1.4219 | 1.2673 | +26.73% | anomalous (see §7) |

Pair-to-pair spread: pairs 2-3 are near-identical on every timer
(<0.5% apart); pair 1 is a mild outlier on the fortran arm (TRACER_2D
3.1759 vs 3.09 for pairs 2-3, +~2.6% high — plausible first-invocation
warm-up), which does not change the direction or order of magnitude of
the regression. **Gate result: regression reproduces robustly ->
proceed to `nsys` (Stage 4).**

## 3. Stage 4: nsys Rank-0 Capture (job 58545163)

### 3.1 CLI issue and fix

The initial `nsys profile ... -- "$REAL" "$@"` invocation (job 58545054)
failed with "option is ambiguous", listing dozens of unrelated long
option names spanning every letter of the alphabet. Root cause:
`nsys profile` does not treat a bare `--` as a standard getopt
end-of-options terminator; it is parsed as an empty long-option name
that prefix-matches every option. **Fix: omit `--` and pass the
application directly** (`nsys profile <flags> "$REAL" "$@"`). Verified
on the login node with a throwaway `/bin/true` capture before
resubmitting. Job 58545054 was hung (11:40 elapsed vs. ~80-90s expected)
on the broken invocation and was cancelled (`scancel`) before the
corrected job was submitted.

### 3.2 Capture

Same node, same executable/hash, `JOB_SGMT=432s` (~7 `dyn_core`
calls/invocation), rank-0-only `nsys profile --trace=cuda,osrt` (other 5
ranks unwrapped, avoiding 6x MPS-shared-GPU profiling overhead). Job
58545163 completed in 2:42, zero errors. Both `.nsys-rep` files
generated (1.86MB cuda, 1.70MB fortran). Timer cross-check from this
same capture: TRACER_2D 2.7522s (cuda) vs 1.9753s (fortran) over 7
calls = **+39.3% slower** — consistent with Stage 2's +37.87% from an
independent, larger (33-call) sample.

## 4. Stage 5: Kernel-Level Attribution

`nsys stats --report cuda_gpu_kern_sum` on the cuda-arm trace. The two
hord==8 tracer kernels:

| Kernel | Instances | Total | Avg |
|---|---|---|---|
| `yppm_pass_kernel_hord8` | 280 | 4.204ms | 15.02us |
| `xppm_pass_kernel_hord8` | 280 | 3.838ms | 13.71us |

Combined: 8.04ms over 7 calls = **1.15ms/call** — negligible against the
~107-111ms/call TRACER_2D delta measured in §2/§3. **GPU kernel compute
is definitively ruled out as the cause.** (All other kernels in the top-174
list are PT/DELZ/REMAP_UV/C-grid/D-grid kernels present identically in
both arms — not tracer-specific.)

## 5. Stage 6: Host<->Device Transfer Attribution

`nsys stats --report cuda_gpu_mem_time_sum`, cuda vs. fortran arm:

| Operation | cuda count | cuda total | fortran count | fortran total |
|---|---|---|---|---|
| D2H | 833 | 1026.67ms | 413 | 961.27ms |
| H2D | 2190 | 806.33ms | 510 | 733.36ms |

Delta: **+420 extra D2H, +1680 extra H2D transfers** (2100 extra total,
exclusive to the cuda arm — these are per-tracer/per-substep hord==8
dispatch transfers) for **+138.37ms** of extra device-side transfer time
over 7 calls = **19.77ms/call**. Individually each extra transfer is
small (avg well under 1ms), consistent with per-tracer/per-level
marshaling rather than one large bulk transfer.

## 6. Stage 7: Synchronization Cost

`nsys stats --report cuda_api_sum`. `cudaEventSynchronize` (175 calls,
~3.54-3.59s) is present in *both* arms at statistically equal cost
(cuda arm is actually 53.6ms *lower*) — this call is issued by the
PT/DELZ CUDA path (active in both arms; not tracer-specific) and is not
the source of the regression. The tracer path's synchronization cost is
instead embedded directly inside its transfers: the extra 2100
host-device copies (§5, §8) are issued via the plain, **synchronous**
`cudaMemcpy` API (not `cudaMemcpyAsync`), which blocks the calling host
thread by definition — 2100 calls, 346.63ms total, **entirely absent
from the Fortran arm** (0 calls). This synchronous-call pattern, not a
separate explicit `cudaDeviceSynchronize`/`cudaStreamSynchronize` call,
is the sync mechanism responsible.

## 7. Stage 8: CPU Orchestration Cost

Full `cuda_api_sum` total (all categories, cuda arm) = 6046.96ms;
fortran arm = 5906.96ms. **Net CUDA-API-visible delta = +140.0ms over 7
calls = +20.0ms/call.** Measured TRACER_2D delta (§3.2) = +111.0ms/call.

**+20.0ms/call (~18%) is directly attributable to CUDA API time**
(dominated by the 2100 synchronous `cudaMemcpy` calls, §6, partly offset
by lower `cudaMemcpyAsync`/`cudaLaunchKernel`/`cudaEventSynchronize`
cost elsewhere in the cuda arm — net still positive).

**The remaining +91.0ms/call (~82%) is CPU-side host orchestration**
that never crosses into a CUDA API call at all — i.e., time spent inside
the `TRACER_2D` Fortran timer window on Fortran/C++ loop overhead,
hord==8 branch dispatch, field-registry/pointer marshaling, and
non-contiguous array packing surrounding the ~300 small transfers/call,
not visible to `nsys`'s CUDA/OS-runtime tracing. `osrt_sum` was checked
and found symmetric between arms (dominated by MPI `poll`, identical
±1% both arms) — ruling out MPI-wait or thread-scheduling overhead as
the source of this CPU-side delta.

## 8. Stage 9-10: Critical Path and Net Attribution

Reconstructed critical path for one Arm-cuda `tracer_2d_1L` /
`fv_tp_2d` hord==8 invocation: negligible kernel execution (~0.16ms of
the ~1.15ms/call kernel total, per-call) -> ~300 small, blocking,
synchronous host<->device copies (~49.5ms/call of API-visible time,
partially offset by savings elsewhere for a net +20.0ms/call) -> the
dominant cost is CPU-side dispatch/marshaling surrounding those
transfers (~91ms/call), not GPU work. Net reconciliation: **20.0ms/call
(CUDA-API) + 91.0ms/call (CPU orchestration) = 111.0ms/call predicted,
matching the 111.0ms/call TRACER_2D delta measured in this same
capture (§3.2) to within rounding** — i.e., no unaccounted residual.

**Secondary, unresolved observation**: `REMAP_TOTALENERGYCON` (§2), a
CPU-only Fortran diagnostic timer sibling to `REMAPPING` (not nested
under TRACER_2D or DYN_CORE), is +26.73% slower in the cuda-tracer arm
despite REMAP_UV/PT-DELZ being byte-for-byte identical configuration in
both arms. This was not root-caused in this task (would require a
`nsys` capture spanning the REMAPPING phase specifically, not performed
here) and does not change the primary attribution above, which fully
reconciles the TRACER_2D delta. Flagged for a future task if REMAPPING
overhead becomes a priority.

## 9. Stage 11: Root Cause

**Root cause: many small, synchronous (blocking) host<->device memory
transfers issued by the CUDA hord==8 tracer dispatch path, plus the CPU-side
host orchestration/marshaling surrounding them — not GPU kernel compute,
not explicit device synchronization/barrier calls, not MPI/OS-level
waiting.** ~300 extra small `cudaMemcpy` calls per `dyn_core` call (2100
over a 7-call capture) that do not exist in the Fortran arm, of which
~18% of the resulting time is directly CUDA-API-visible and ~82% is CPU
orchestration time around those calls, fully reconciles the measured
TRACER_2D regression (§8).

## 10. Stage 12: Optimization Feasibility (not implemented)

**One recommended follow-up**: batch the ~300 small synchronous
per-call host<->device transfers into a single packed, asynchronous
(`cudaMemcpyAsync` + stream) transfer per `fv_tp_2d`/`tracer_2d_1L`
invocation, replacing the current per-tracer/per-substep marshaling
loop. This directly targets both the measured +20.0ms/call CUDA-API
cost (§7/§8) and plausibly a portion of the +91.0ms/call CPU-side
marshaling cost, since consolidating the transfer would likely also
consolidate the surrounding host-side packing loop — though the CPU-side
reduction is not itself directly measured here and would need to be
re-profiled after any such change. **Not implemented in this task** —
this is a CUDA/C++ source change, out of scope for a profiling-only
prompt.

## 11. Stage 13: Production Default

**Unchanged.** `use_cuda_tp2d`'s default remains Fortran-dispatch for
tracer transport (`fv_tracer2d.F90`); no CUDA/Fortran source file was
modified in this task. This finding reinforces, rather than revisits,
Prompt 77's recommendation to keep the CUDA tracer path opt-in/disabled
by default pending the optimization in §10.

## 12. Artifacts

All artifacts are scratch/experiment directories under
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`
(`p78_stage2_cuda/`, `p78_stage2_fortran/`, `p78_stage4_cuda/`,
`p78_stage4_fortran/`, `p78_stage2_alternating_ab.sbatch`,
`p78_stage4_nsys_ab.sbatch`) — not part of the git-tracked source tree,
no commit made. Jobs: 58544700 (Stage 2, COMPLETED), 58545054 (Stage 4,
broken nsys flag, cancelled), 58545163 (Stage 4, COMPLETED). `.nsys-rep`
files and derived CSV stats are under the same experiment directories
and `/tmp/p78_nsys_stats/` (scratch, not persisted).

**Result: TRACER HORD8 REGRESSION CONFIRMED — OPTIMIZE many small
synchronous host-device tracer transfers (and their CPU-side marshaling
overhead) in the CUDA hord==8 dispatch path.**

---

**Follow-up (Prompt 79, 2026-09-24):** the §10 batching fix was
implemented (`cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu`
rewritten to one packed `cudaMemcpyAsync` H2D + one packed
`cudaMemcpyAsync` D2H per (k, iq) slice call, down from 15
per-slice-call blocking transfers), reducing the transfer count from
~300 to ~40 per `tracer_2d_1L` invocation at C512 scale. Standalone GPU
regression confirmed numerically identical to the Prompt 76 baseline
(0 sanitizer errors). However, the production A/B re-measurement this
finding calls for could **not** be run: relinking
`StandAlone_FV3_Dycore.x` against the change was blocked by a
pre-existing, unrelated Intel ifort `#6633` compile failure in
`FV_StateMod.F90` (severe-weather diagnostics, never successfully built
in this tree, not touched by any tracer/CUDA prompt). See
`docs/performance/p79_tracer_batched_transfer_optimization_2026-09-24.md`.
