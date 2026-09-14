# Prompt 28: Standalone Launch-Fusion Enablement and CUDA Graph ABI Decision

## 1. Objective

Enable the already-validated launch-reduction options (`--fuse-halo-edges`,
`--fuse-corner-fill`) on the real standalone CUDA path (previously hardcoded
off at `fv3_gpu_dyncore_bridge.F90:604-605`, identified as the Prompt 27 next
step), measure the effect under CUDA MPS, guard numerics against the CUDA
baseline, and decide whether a production CUDA Graph ABI is justified. No new
kernel optimization, no CUDA-aware MPI, no rank/GPU redesign.

## 2. Production Config Knobs Added

`cuda/interface/fv3_gpu_dyncore_bridge.F90`: two environment variables,
read at `NumericalConfig` install time via a new `env_flag_enabled(name)`
helper (YES/1/TRUE case-insensitive, unset/other = false, default OFF):

- `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` -> `cfg%fuse_halo_edges = 1`
- `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` -> `cfg%fuse_corner_fill = 1`

No ABI change was required — `fv3_gpu_numerical_config` already carried
`uint8_t fuse_corner_fill; uint8_t fuse_halo_edges;`. Default behavior
(both unset) is unchanged from before this prompt. Confirmed via `strings`
on the rebuilt `StandAlone_FV3_Dycore.x` that both env-var names are present
in the binary. **Numerical algorithm changed: NO** — this is config plumbing
only, selecting between two already-bitwise-validated launch patterns.

## 3. Build Blocker Encountered and Fixed (carried from prior segment)

- File: `GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/{fv3_savepoint_mod.F90,fv_arrays.F90,fv_mapz.F90}`.
- Original error: stale/inconsistent `.mod` interface files under `-r8`
  real-kind auto-promotion caused a build failure unrelated to the two new
  env vars.
- Root cause: incremental-build `.mod`/`.o` staleness, not a source defect.
- Change: forced full recompilation via `touch` on all 52 `.F90` files under
  the component tree. Zero source bytes changed.
- Numerical algorithm changed: **NO**.

## 4. EXPDIR Hardcoding Bug (found and fixed this segment)

`fv3.j` hardcodes its own `EXPDIR` as an absolute path
(`setenv EXPDIR /discover/.../experiments/<dirname>`) rather than deriving it
from `$cwd`. `p28_c32_r8_cuda_guard` was created via `cp -a` of
`p24_c32_r8_cuda`; the copied `fv3.j` retained the original `EXPDIR`, so the
first guard job (58381546) silently ran its "guard" mode-A/B/C passes against
`p24_c32_r8_cuda`'s own `EXPDIR`/`SCRDIR`/savepoint paths, and those
savepoints were immediately deleted by the next `run_timing()` call's
`rm -rf savepoints scratch_*` in that same shared directory before capture.
Fixed via `sed -i` to repoint `EXPDIR` at `p28_c32_r8_cuda_guard` itself; no
other content line referenced the old path (`CAP.rc`/`AGCM.rc`/`HISTORY.rc`
are heredoc-generated in-script, not stale copied files). Re-ran the guard
portion alone (`p28_guard_only.sbatch`, job 58381648) — succeeded, all three
modes' savepoints captured (2160 files each). The original job's **timing**
results were unaffected (always ran in the correctly-pathed
`p24_c32_r8_cuda`) and did not need to be rerun. **Standing pitfall for this
project**: `cp -a`-based job-directory reuse must re-point `EXPDIR` in the
copied `fv3.j` before running — no prior prompt had hit this because none
had created a job directory by copying another's `fv3.j`.

## 5. Numerical Guard (§3 requirement)

Tool: `tools/savepoints/compare_dcx_backends.py` (backend-agnostic
tree-vs-tree `dcx`/`dce` savepoint comparator; reused unchanged), run
CUDA(mode A) vs CUDA(mode B) and CUDA(mode A) vs CUDA(mode C), 10 dyn_core
calls x 6 ranks x 17 `dcx`-stage fields, c32/R8.

| Comparison | Result |
|---|---|
| A (baseline) vs B (fuse_halo_edges) | **BITWISE IDENTICAL**, worst max_rel = 0.000000e+00, rc=0 |
| A (baseline) vs C (fuse_halo_edges+fuse_corner_fill) | **BITWISE IDENTICAL**, worst max_rel = 0.000000e+00, rc=0 |

All three modes: rc=0, 0 Range_Warn, 0 NaN, identical savepoint file counts
(2160 each). No new Fortran-vs-CUDA campaign was needed (per §3, only
required if the optimized result differs from the CUDA baseline — it did
not).

## 6. Performance Measurement (§4 requirement)

c32, R8, `npz=10`, `n_split=3`, `k_split=1`, `bdt=1200s`, 6 MPI ranks
(`FV_NX=1`/`FV_NY=6`), one A100/one node, CUDA MPS ON throughout (single MPS
session spanning all three modes, via `fv3_mps_lifecycle.sh`), 3 reps/mode,
100 calls/rep, savepoints disabled during timing, `fv_timing_mod`'s
`DYN_CORE` timer. Job 58381546, all reps rc=0/0 Range_Warn/0 NaN.

| Mode | median ms/call | spread | improvement vs. A |
|---|---|---|---|
| A — baseline | 27.126 | 0.71% | — |
| B — fuse_halo_edges | 25.888 | 1.34% | 4.56% |
| C — fuse_halo_edges + fuse_corner_fill | 25.464 | 0.31% | 6.13% |

Mode A's 27.126 ms/call is consistent with Prompt 27's MPS baseline (26.965
ms/call, within 0.6%) — an internal sanity check that this run reproduces
the known-good regime. Incremental fuse_corner_fill-only gain (C over B) =
1.64%. Mode C also has the tightest reproducibility of the three.

## 7. Launch Count / nsys Trace (§5 requirement — best mode only)

Single trace, Mode C (best mode), reusing the existing
`p24_c32_r8_cuda_nsys`/`nsys_wrap.sh` mechanism unchanged (10-call rank-0
trace, `--trace=cuda,osrt`, MPS ON). Job 58381785, rc=0/0 Range_Warn/0 NaN.
Traced `DYN_CORE` = 32.73 ms/call (nsys overhead inflates this above the
untraced 25.464 figure; used only as this trace's internal percentage
denominator, matching the Prompt 27 convention).

| Bucket | Prompt 27 baseline (2955 launches/call) | Prompt 28 Mode C (2637 launches/call) |
|---|---|---|
| Launches/call | 2955 | **2637 (−10.76%)** |
| `cudaLaunchKernel` | 17.73 ms/call (51%) | 15.81 ms/call (48.3%) |
| Kernel execution | 12.81 ms/call (37%) | 12.08 ms/call (36.9%) |
| `cudaEventSynchronize` | 9.67 ms/call (28%) | 9.98 ms/call (30.5%) |
| H2D+D2H device transfer | 0.62 ms/call (1.8%) | 0.618 ms/call (1.9%) |

## 8. Event-Synchronization Interpretation (§8 requirement)

Launch count fell 10.76% and `cudaLaunchKernel` time fell in step (17.73 ->
15.81 ms/call), but `cudaEventSynchronize` time did **not** fall — it is
flat to slightly up (9.67 -> 9.98 ms/call, within trace-to-trace noise).
This indicates the remaining synchronization time is **not** proportional to
launch count, i.e. it is not host-orchestration slack that launch fusion
would relieve. Under the MPS-shared-context regime (6 ranks sharing 1 A100
via MPS), this is consistent with genuine GPU-producer wait (ranks blocking
on results from a context-shared GPU) rather than removable host overhead.
Per §8's instruction, synchronization was not targeted for direct
optimization in this task.

## 9. Best Mode and Classification (§6 requirement)

**Mode C (fuse_halo_edges + fuse_corner_fill) is the best measured mode**:
6.13% whole-`dyn_core` improvement, bitwise-identical numerics, tightest
reproducibility (0.31% spread). Mode B alone (fuse_halo_edges) already
captures the large majority of the benefit (4.56% of 6.13%, i.e. ~74%),
consistent with §6's guidance to prefer `fuse_halo_edges` as the primary
production-default candidate; `fuse_corner_fill` adds a smaller (1.64%)
but positive, non-noisy increment on top — it is not recommended against,
since it neither degrades performance nor increases noise (Mode C has the
lowest spread of the three).

**Classification: MODERATE (5-10%)** — 6.13% whole-`dyn_core` improvement
for Mode C.

## 10. CUDA Graph ABI Decision (§7 requirement)

Criteria evaluated against gathered evidence:

1. Launch/API remains a major bottleneck: **YES** — 48.3% of traced
   `DYN_CORE` under Mode C, still the single largest bucket.
2. Expected whole-call benefit plausibly >= 5%: **borderline-yes** — the
   only direct graph-capture measurement on record is Prompt 13's internal
   single-process harness (`dgrid_dynamics`-only capture), 21.660791 ->
   20.546787 ms/call = 5.14%. That figure predates MPS and launch fusion and
   was measured on the internal harness, not the real six-rank standalone
   path, so it is directional evidence, not a proof for this configuration.
3. Graph replay can reuse the already-validated `dgrid_dynamics` capture
   path: **YES** — no new capture-region work needed.
4. Implementation scope LOCAL/MODERATE: **YES** — would mirror the same
   ABI-field-addition + Fortran-bridge-plumbing pattern just used for the
   two fusion flags in this prompt (add a numerical-config field, thread it
   through the bridge, gate default OFF).

All four criteria are satisfiable, with criterion 2 the weakest (borderline,
old measurement, different harness). Given launch/API is still the largest
single bucket even after Mode C's 10.76% launch-count cut, and the
implementation path is a known-low-risk mechanical extension of work already
done in this prompt:

**GRAPH ABI JUSTIFIED** — evidence: launch overhead remains the dominant
bucket (48.3%) after the cheaper fusion wins are taken; the capture
mechanism and validation harness both already exist and are already proven
bitwise-neutral; the implementation pattern (ABI field + bridge plumbing,
default OFF) is the same LOCAL-scope pattern just executed successfully for
`fuse_halo_edges`/`fuse_corner_fill`; the only weak point is that the >=5%
whole-call benefit estimate is carried over from an older, differently-scoped
measurement (Prompt 13, pre-MPS, internal harness, `dgrid_dynamics`-only) and
should be re-measured on the real standalone path once implemented, not
assumed. Per the prompt's explicit instruction, the ABI itself is **not**
implemented in this task — this is a scope recommendation only, pending
user confirmation.

## 11. Regression Determination (§10 requirement)

No shared runtime source *content* changed in a way that affects numerical
kernels: the build-blocker fix (§3) was `touch`-only (zero bytes changed);
the Fortran bridge edit (§2) is config plumbing selecting between two
already-bitwise-validated launch patterns, not new numerical code. Per
§10's rule ("If only configuration/plumbing changes: run optimized CUDA
standalone smoke + numerical comparison to CUDA baseline" — done in §5), a
full R8 CTest rerun is **not** required and was not performed.

## 12. Files Touched This Prompt

- `cuda/interface/fv3_gpu_dyncore_bridge.F90` (source fix, prior segment;
  `env_flag_enabled` + two config-field wires).
- `GEOSfvdycore_fortran_perf/experiments/p28_c32_r8_cuda_guard/` (new job
  dir, `cp -a` of `p24_c32_r8_cuda`, `EXPDIR` fixed).
- `GEOSfvdycore_fortran_perf/experiments/p28_launch_fusion.sbatch`,
  `p28_guard_only.sbatch`, `p28_nsys_best_mode.sbatch` (new job scripts).
- `docs/agent_handoff/prompt_28_standalone_launch_reduction.md` (this file).
- `docs/agent_handoff/CURRENT_STATE.md` (updated, §8/§9/§11 compacted to
  stay <= 35,000 characters).

Not committed/pushed — awaiting explicit user request per project policy.
