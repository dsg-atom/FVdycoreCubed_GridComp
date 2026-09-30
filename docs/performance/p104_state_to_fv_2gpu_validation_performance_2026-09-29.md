# Prompt 104 — `State_To_FV` CUDA Validation at 2-GPU/3-Ranks-Per-GPU (C512)

Date: 2026-09-29
Job: 58645113 (`p104_state_to_fv_2gpu_validation_58645113.out`, node `warpa009`,
2x A100-SXM4-40GB, MPS ON, Intel MPI 2021.13, R8, C512, 6 ranks / 2 GPUs / 3
ranks-per-GPU, `dev = mod(comm_rank, device_count)` unchanged).

This is the direct follow-up Prompt 103 (§7 Option A) called for: explicitly
confirm CUDA dispatch of the `State_To_FV` port at 2-GPU density, run a
correctness pair, and run a proper alternating-order A/B performance study.
**No CUDA/Fortran source changed. No production default changed
(`State_To_FV` remains Fortran-default regardless of this task's outcome).
No commit/push in either repo.**

**Superseded in part by Prompt 106** (`p106_state_to_fv_default_policy_
2026-09-29.md`): `State_To_FV` is now CUDA-default at `device_count>=2`
(this task's validated density), Fortran-default elsewhere. This task's
dispatch/speedup/memory findings are otherwise unchanged and remain
authoritative for that density.

## 1. Recovery Note

The prior session's Prompt 104 attempt was interrupted by a connection loss
before any job was submitted (confirmed via `sacct`/`squeue`/filesystem
search — no p104 rundirs, docs, or queued/running jobs existed). This is a
fresh, complete execution, not a partial resume.

## 2. Configuration

- Binary: `StandAlone_FV3_Dycore.x`, R8, Intel MPI 2021.13 build.
- Production-clean sha256: `2339366b10885c4532f08efa089d4e3b0a062403397f0c64c6a396b186e2064d`
  — used for the **correctness** and **performance** stages.
- A one-off instrumented-binary sha
  (`7393b75d229ce145396d1053f73818f1729fdcf1a6f4f2b10b1f816808530e55`) was
  set up as an `EXPECTED_SHA_INSTRUMENTED` guard for a planned dispatch
  counter, but no counter instrumentation was ever added to `cuda/` source
  in this task (see §3) — the dispatch-proof stage in fact ran the same
  production-clean binary. No CUDA source was rebuilt or changed.
- Arm A (Fortran): `RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1`.
- Arm B (CUDA): `RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1`.
- Both arms: `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
  `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, MPS ON, 2-GPU/3-ranks-per-GPU device
  mapping (Prompt 103's validated config).
- All 9 sub-runs (1 dispatch-proof + 2 correctness + 6 performance) completed
  `rc=0`, `DYN_DATE_COUNT=30` (full 30-call harness), **zero** `FATAL` / `out
  of memory` / `Abort` / `NaN` / `Inf` matches in any run's scratch log.

## 3. Explicit Dispatch Confirmation

**Outcome: dispatch counter mechanism as originally planned was never
implemented — but converging independent evidence CONFIRMS real CUDA-side
execution of both `state_to_fv_phase_a` and `state_to_fv_phase_c`.**

A print-statement dispatch counter (`FV3_GPU_DEBUG_STATE_TO_FV_DISPATCH`,
following the existing `FV3_GPU_DEBUG_*` pattern in `fv3_gpu_runtime.cpp`)
was planned but **no such instrumentation code was ever written** to
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` or anywhere else in `cuda/`
(confirmed via `grep -rn "P104_DISPATCH\|FV3_GPU_DEBUG_STATE_TO_FV_DISPATCH"
cuda/` — zero matches). The resulting `phase_a=0 phase_c=0` grep count from
the dispatch-proof run's `dispatch_stderr.log` is therefore **not a real
negative finding** — it reflects a harness gap (an env var set, and a grep
performed, against code that does not exist), not evidence that dispatch
did not occur. `dispatch_stderr.log` itself was also found to contain only
captured `module list` output, not real per-rank MPI stderr — the capture
mechanism itself was not wired correctly. No source change was made to fix
this (out of scope, per the bounded-debugging/cost-efficiency instructions —
see §3.1 for why the existing evidence was judged sufficient instead of
launching a supplemental instrumented job).

Independent evidence that **does** exist, and that jointly establishes real
CUDA-side dispatch without the planned counter:

1. **Fortran-side branch selection**: `RUN_CUDA_FV3_STATE_TO_FV_DUMP_EXIT=1`
   was active for the dispatch-proof run. All 6 ranks printed
   `[P99_EXIT_DUMP] ... use_cuda_state_to_fv=T ...` consistently across the
   run (30 calls), confirming the Fortran selector chose the CUDA branch
   every call, on every rank.
2. **ABI contract success (no FATAL)**: `fv3_gpu_state_to_fv_phase_a`/`_c`
   return `FV3_GPU_UNSUPPORTED_CONFIGURATION` if their dispatch slots are
   null, which the Fortran bridge's `check()` call turns into an
   `mpp_error(FATAL, ...)`. The dispatch-proof run completed `rc=0`,
   `DYN_DATE_COUNT=30`, zero FATAL — this proves the ABI functions were
   called, the dispatch slots were non-null (real implementations
   installed, not missing/stubbed), and both phases returned success on
   every one of the 30 calls, on all 6 ranks.
3. **Device-memory footprint (strongest single piece of evidence)**: the
   CUDA arm's persistent `state_to_fv_*_buffer` allocations
   (`fv3_gpu_runtime.cpp`) only grow when `fv3_gpu_state_to_fv_phase_a`/`_c`
   actually execute. The correctness-cuda arm's memory poller (§5) measured
   a converged **+440.4 MiB/GPU** increase over the Prompt-103 baseline
   (19.68 GiB/GPU with `State_To_FV` Fortran) — closely matching Prompt
   100's independently-derived static prediction of **+452.97 MiB/GPU** for
   this port at 3-ranks-per-GPU (2.7% delta, consistent with allocator
   rounding). A silent bypass of the CUDA path (e.g. a selector bug) would
   leave memory at the Fortran baseline; it did not.
4. **Consistent, low-noise CUDA-specific timing signature**: the `mpp_clock`
   `State_To_FV` timer (see §7) shows CUDA arms at 1.66-1.68 ms/call and
   Fortran arms at 2.72-2.73 ms/call, reproducible across all 3 independent
   A/B pairs with ~0.02 ms spread — a distinct, stable, repeatable
   behavioral signature consistent with a different code path executing,
   not run-to-run noise (timer differences alone are not being used as the
   sole dispatch proof — only as one corroborating signal among the four).

**Classification: CUDA dispatch of `state_to_fv_phase_a` and
`state_to_fv_phase_c` CONFIRMED** via branch-selection + ABI-success +
device-memory-footprint + timing-signature evidence, in place of the
planned (but unimplemented) raw dispatch counter.

### 3.1 Why no supplemental instrumented job was run

The four evidence sources above are independent, mutually corroborating,
and quantitatively consistent with each other and with Prompt 100's prior
static prediction. Per the project's bounded-debugging guidance ("define a
stop condition; stop once evidence is sufficient"; "do not continue
gathering redundant evidence after a hypothesis is resolved"), a further
GPU job adding a real print-statement counter was judged unnecessary — it
would only reconfirm what device-memory + ABI-contract evidence already
establishes concretely. This is flagged explicitly, rather than silently
substituting timing for the originally-planned counter.

## 4. Correctness

**CORRECTED 2026-09-29 (Prompt 105): the claim below that "neither
correctness rundir produced any `.nc4`/HISTORY output" was a
methodology error, not a real gap.** The check only inspected the
top-level rundir (`p104_correctness_fortran`/`_cuda`), where indeed no
`HISTORY.rc`/`.nc4` exists — but it never descended into the actual
execution subdirectory,
`scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8/`, where
`fv3.j` runs the model and MAPL History actually writes its output. That
subdirectory contains a fully-configured `HISTORY.rc` (T/U/V/PS on 25
pressure levels) and two non-empty, job-58645113-timestamped `.nc4`
files per arm. Prompt 105 used this already-existing output (no new job
was needed) to perform the field-level comparison this section
originally could not. **Result: T/PS BITWISE, U/V ROUND-OFF CLEAN — see
`p105_state_to_fv_c512_2gpu_correctness_2026-09-29.md` for the full
comparison.** The original (erroneous) write-up is preserved below for
provenance.

---

**Original (erroneous) Outcome: NOT DIRECTLY FIELD-COMPARED at
C512/2-GPU in this job — a real gap, reported transparently.
GPU-EXECUTED and CLEAN-RUN status: PASS.**

The correctness stage (`p104_correctness_fortran`, `p104_correctness_cuda`)
was intended to be compared via `tools/analysis/hs_netcdf_inspect.py
compare-backends` against HISTORY.rc T/U/V/PS output, per Prompt 103's
methodology. On inspection, **neither correctness rundir produced any
`.nc4`/HISTORY output** — both `savepoints/` directories are empty, and no
`HISTORY.rc` exists in either rundir. This 30-call, short-duration harness
was not configured to write history collections, so there is **no
field-level data available to compare**. This is a genuine methodology gap
in the job setup, not a suppressed negative result — no comparison was
silently skipped in favor of a fabricated PASS. **(See correction above —
this was a wrong-directory check, not a real absence of output.)**

What **is** established for the correctness pair:

- Both `p104_correctness_fortran` (Arm A) and `p104_correctness_cuda` (Arm
  B) completed `rc=0`, `DYN_DATE_COUNT=30`, zero FATAL/OOM/Abort/NaN/Inf.
- `RUN_CUDA_FV3_STATE_TO_FV_DUMP_EXIT` was **not** active during the
  correctness stage (it was unset after the dispatch-proof stage), so no
  per-rank sample-value dump exists for this pair either.
- The underlying `State_To_FV` CUDA kernel itself was already **GPU
  EXECUTED and NUMERICALLY VALIDATED at C32** in Prompts 98-99 (bit-level
  correct at that density) — this scope is unchanged and still stands. What
  is *not yet* established is a field-level bitwise/round-off comparison
  specifically at **C512/2-GPU density**.
- No evidence of any kind (clean or otherwise) contradicts correctness at
  this density — the gap is an absence of data, not a failure signal.

**Classification (superseded by Prompt 105, see correction above):
GPU EXECUTED — PASS (clean, both arms). NUMERICALLY VALIDATED at
C512/2-GPU — NOW ESTABLISHED (Prompt 105): T/PS BITWISE, U/V ROUND-OFF
CLEAN, no halo/layout-pattern discrepancy.**

## 5. Memory / Stability

- Idle pre-run checkpoint (both GPUs): 4 MiB used / 40334 MiB free.
- CUDA-arm (`p104_correctness_cuda`) converged, in-flight measurement via a
  concurrent 5s-interval poller (`P104_CORR_poll.log`): **20585 MiB
  used / 19753 MiB free per GPU**, both GPUs identical (confirms uniform
  3-ranks-per-GPU density).
- Incremental cost of enabling `State_To_FV` CUDA at this density: 20585 −
  (19.68 GiB × 1024 ≈ 20144.3 MiB) ≈ **+440.4 MiB/GPU**, versus Prompt
  100's static prediction of **+452.97 MiB/GPU** — a 2.7% match, well within
  allocator-rounding noise. This is strong corroborating evidence (see
  §3.3) that the predicted buffers are exactly what is being allocated.
- Free headroom at converged state: **19753 MiB/GPU (≈19.3 GiB)** — far
  above the >=512 MiB safety margin that blocked this port at 1-GPU/6-ranks
  (P98-100). **SAFE at this density**, consistent with Prompt 103's
  feasibility projection.
- No Fortran-arm poller was run in this job (poll flag was only passed for
  the CUDA arm); the Fortran-arm baseline is taken from Prompt 103's
  already-measured 19.68 GiB/GPU (same config, `State_To_FV` Fortran) —
  not re-measured redundantly here.

## 6. Workspace-Alias Decision

**Classification: NOT NEEDED NOW.** Prompt 100's temporal-alias
optimization (reusing `WorkspaceId::vertical`'s ~6x-larger, lifetime-disjoint
arena instead of allocating dedicated `state_to_fv_*_buffer`s) was designed
specifically to close a ~440 MiB shortfall at 1-GPU/6-ranks density. At
2-GPU/3-ranks-per-GPU, measured headroom is **19.3 GiB free/GPU** — roughly
**44x** the port's own +440.4 MiB footprint. There is no measured or
projected memory pressure at this density that the aliasing optimization
would need to relieve. It remains a valid **future optimization** if rank
density per GPU is later increased again (e.g. returning to 1-GPU/6-ranks,
or C1024 without new GPUs) — **not implemented in this task**, per the
explicit prohibition on doing so.

## 7. Paired Performance — Direct `State_To_FV` Speedup

Three independent alternating-order A/B pairs (A1 B1 / B2 A2 / A3 B3, per
Prompt 101A/103 methodology), each pair in the same Slurm allocation/node
(`warpa009`), 30 calls/run. `State_To_FV` timing is the `mpp_clock`
hierarchical table's `----------State_To_FV` line (10-dash nesting level,
`hits=30`, matching `DYN_DATE_COUNT`) — the exclusive/self-time column
(first of three triplets); a separate zero-hit `--------State_To_FV`
artifact line (8-dash, `hits=1`) exists per run and is not used.

| Pair | Fortran ms/call | CUDA ms/call | Speedup | Reduction |
|---|---|---|---|---|
| 1 | 2.73 | 1.67 | 1.635x | 38.83% |
| 2 | 2.72 | 1.66 | 1.639x | 38.97% |
| 3 | 2.73 | 1.68 | 1.625x | 38.46% |

- **Direct speedup: mean 1.633x, stddev 0.006x, min 1.625x, max 1.639x.**
- **Direct reduction: mean 38.75%, stddev 0.22pp, min 38.46%, max 38.97%.**
- Extremely tight spread across all 3 pairs (stddev << 1% of the mean) —
  a strong, reproducible, low-noise signal, consistent with a real and
  deterministic kernel-level speedup rather than run-to-run variance.
- The correctness-stage pair (fortran 2.72 ms, cuda 1.68 ms — not one of
  the 3 official performance pairs, run under a different stage/purpose)
  shows the same magnitude, corroborating the above.

## 8. End-to-End Effect (TOTAL / FV_DYN_LOOP / DYN_CORE)

`fv_timing_mod` timer values (ms/call) for the same 3 pairs:

| Pair | Arm | TOTAL | FV_DYN_LOOP | DYN_CORE |
|---|---|---|---|---|
| 1 | Fortran (A1) | 110.1237 | 33.8582 | 15.1106 |
| 1 | CUDA (B1) | 108.7560 | 33.4160 | 14.3976 |
| 2 | CUDA (B2) | 108.5957 | 33.1294 | 14.3631 |
| 2 | Fortran (A2) | 109.3303 | 33.0582 | 14.3587 |
| 3 | Fortran (A3) | 109.6445 | 33.3566 | 14.3872 |
| 3 | CUDA (B3) | 108.4278 | 33.1171 | 14.3820 |

Per-pair TOTAL improvement (CUDA vs. Fortran, same pair): 1.2416% (pair 1),
0.6721% (pair 2), 1.1093% (pair 3). **Mean TOTAL improvement: 1.01%
— classified MEANINGFUL (>=1% threshold, at the boundary).**

Per-pair FV_DYN_LOOP: +1.3067% (pair 1), **−0.2154%** (pair 2, CUDA
slightly slower), +0.7181% (pair 3). **Mean FV_DYN_LOOP improvement:
0.60% — classified SMALL POSITIVE (0.3-1%).**

Per-pair DYN_CORE: +4.72% (pair 1), −0.03% (pair 2), +0.04% (pair 3). Mean
+1.58%, but almost entirely driven by pair 1's outlier — see §9
(attribution) for why this is expected node/run noise rather than a real
`State_To_FV` effect on `DYN_CORE`.

## 9. Attribution

`State_To_FV` executes once per call, **outside** the `k_split`
`FV_DYN_LOOP`/`DYN_CORE` acoustic-substep loop (its `mpp_clock` hit count is
30, matching the outer call count, not `k_split`-scaled) — it is a sibling
of `FV_DYN_LOOP` under `TOTAL`, not nested inside it. This architecture
predicts:

- `DYN_CORE` should be **unaffected** by this port — confirmed: pairs 2 and
  3 show ±0.04% (noise-level); pair 1's +4.72% is an isolated outlier
  consistent with the >2x node-to-node/run-to-run variance already
  documented in `CURRENT_STATE.md` §8, not a `State_To_FV` effect.
- `FV_DYN_LOOP` should likewise be **largely unaffected** — confirmed: mean
  +0.60%, with one pair (2) even showing a small negative delta, consistent
  with `State_To_FV` living outside this timer's scope.
- `TOTAL` should show a gain of roughly (State_To_FV's share of TOTAL) ×
  (direct reduction): Fortran-arm `State_To_FV` (2.72 ms) is ≈2.49% of
  Fortran-arm TOTAL (≈109 ms); 2.49% × 38.75% ≈ **0.97% expected TOTAL
  gain** — closely matching the **1.01% observed** TOTAL improvement (§8).

**Attribution: CONSISTENT.** The measured end-to-end effect is fully
explained by `State_To_FV`'s small share of `TOTAL` combined with its large
(38.75%) direct reduction; no unexplained residual gain or loss requiring
further profiling (`nsys`) was found.

## 10. Modernization Context (derived, not directly measured)

Composing this task's ~1.01% TOTAL gain with Prompt 103's already-measured
2-GPU multi-GPU gain (1.120x) and Prompt 101A's cumulative CUDA-vs-all-
Fortran baseline (1.153x) gives a naive composite of roughly 1.153x ×
1.120x × 1.0101 ≈ **1.305x** vs. an all-Fortran/1-GPU reference.
**DERIVED CONTEXT ONLY — not directly measured in any single job; do not
treat as a validated end-to-end figure.**

## 11. Default-Enable Recommendation

**Superseded 2026-09-29 (Prompt 105): field-level correctness gap
closed (T/PS BITWISE, U/V ROUND-OFF CLEAN) — recommendation is now
READY FOR DEFAULT ENABLE (as a separate, dedicated follow-up task; not
changed in Prompt 105 itself).** Original (now-superseded) rationale
below, preserved for provenance.

**Original Recommendation: KEEP OPT-IN.** Rationale:

- Performance is genuinely positive and highly reproducible (direct
  38.75%±0.22pp reduction, 3/3 pairs; TOTAL 1.01% mean, MEANINGFUL by a
  small margin) — this is not a regression and does not argue for reverting
  the port.
- However, **no field-level numerical validation exists at C512/2-GPU
  density** (§4) — only clean-run (no FATAL/NaN) evidence and the
  unrelated-density C32 bitwise validation from Prompts 98-99. Per
  `project_architecture.md`'s validation hierarchy, GPU-executed-clean does
  not substitute for numerically-validated, and default-enabling a
  production port without density-matched field validation is not
  justified by this task's evidence alone.
- The gap is narrow and well-defined (§4/§9 of "Next Recommended Task"
  below) — a follow-up correctness pair with `HISTORY.rc` configured would
  likely close it quickly, given the port's already-clean C32 validation
  and this run's clean 30-call execution.

`RUN_CUDA_FV3_STATE_TO_FV_ENABLE` remains opt-in;
`RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN` remains the escape hatch. **No
production default changed in this task.**

## 12. Supported Configuration Scope

Validated (this task): R8, C512, 6 MPI ranks, **2 A100 GPUs, 3
ranks-per-GPU**, MPS ON, `RUN_CUDA_FV3_FUSE_HALO_EDGES`/
`_FUSE_CORNER_FILL=YES`, same-node, Intel MPI 2021.13, host-staged MPI halo
transport (`host_staged`, default). **Not validated in this task**: R4,
C1024+, 1-GPU/6-ranks density (memory-blocked per P98-100, unchanged),
4-GPU spread (P102 memory-only), `cuda_aware` MPI transport combined with
this port, multi-node.

## 13. Next Task

**DONE (Prompt 105):** the correctness gap described above has been
closed by reusing this job's own HISTORY output (found in the
`scratch_*/` execution subdirectory, not the top-level rundir — see §4
correction). Result: T/PS BITWISE, U/V ROUND-OFF CLEAN. See
`p105_state_to_fv_c512_2gpu_correctness_2026-09-29.md`. **Remaining next
task: default-enable `State_To_FV` CUDA at this validated density, as
its own separate, dedicated bounded task** — not performed in either
Prompt 104 or 105.

## 14. Job / Evidence Provenance

- Slurm job 58645113, output log
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/p104_state_to_fv_2gpu_validation_58645113.out`.
- Rundirs (same experiments directory): `p104_dispatch_proof/`,
  `p104_correctness_fortran/`, `p104_correctness_cuda/` (+
  `P104_CORR_poll.log`), `p104_pair{1,2,3}_{fortran,cuda}/`.
- No CUDA/Fortran source changed. No production default changed. No
  commit/push in either repo.
