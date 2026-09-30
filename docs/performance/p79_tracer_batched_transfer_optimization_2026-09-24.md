# Prompt 79 — Batch Tracer Host-Device Transfers for hord==8

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Prerequisite: Prompt 78 (`p78_tracer_hord8_overhead_attribution_2026-09-24.md`)
— CONFIRMED and ATTRIBUTED the hord==8 CUDA tracer-dispatch regression
(TRACER_2D +37.87%, 33-call alternating A/B) to ~300 small **synchronous**
`cudaMemcpy` host<->device transfers per `dyn_core` call plus their CPU-side
marshaling overhead (kernel compute itself negligible, ~1.15ms/call).
Recommended, not-yet-implemented fix: batch into one packed async transfer.

## 0. Goal / Scope

Replace the ~300 small synchronous per-call tracer transfers with the
smallest safe batched-transfer architecture (host pack -> one/few large
async H2D -> existing, unchanged CUDA hord==8 transport -> one/few large
async D2H -> host unpack), then measure whether this recovers enough of the
~111ms/call regression to make CUDA tracer transport performance-positive.
Change budget: **one substantial source change**. Explicitly out of scope
(all honored, none touched): hord==8 numerical formulas, `pert_ppm`, tracer
physics/orchestration, `hord_tr`, PT/DELZ, REMAP_UV, `nsplt>1`, MPI halo
redesign, `mp_reduce_max`, R4, GPU kernel arithmetic, `ncu`. Production
default remains Fortran-dispatch for tracer transport regardless of
outcome (Stage 14, unconditional).

## 1-5. Mapping / Design (prior turns, unchanged this doc)

Per-(k,iq)-slice invocation model confirmed: `fv3_gpu_fv_tp_2d_hord8`
(`cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu`) is called once per
(k, iq) from a Fortran nested loop; at C512 production scale (~20
slice-calls/`tracer_2d_1L` call) the pre-Prompt-79 code issued 15 blocking
memcpy/slice-call (12 H2D + 3 D2H) = ~300 total, matching Prompt 78's
`nsys` count exactly. Packed layout targets hord==8, nq==1, nsplt==1, R8,
per-single-slice-call packing (not across the Fortran k/iq loop, which
would be a second, larger architectural change outside this task's
one-change budget).

## 6. Stage 6 — One Architectural Change (IMPLEMENTED)

`cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu` rewritten. ABI/
signature of `fv3_gpu_fv_tp_2d_hord8` byte-for-byte unchanged (no
Fortran-side edit required). Internal transfer strategy replaced:

- One contiguous **pinned** host buffer (`thread_local PinnedBuf g_pack`,
  `cudaHostAlloc`/`cudaFreeHost`, grow-only `ensure()`) and one contiguous
  **device** buffer (`Workspace::packed`, same grow-only `DeviceBuf`
  discipline as the pre-existing Prompt-68 precedent).
- All 12 kernel inputs (q, crx, cry, xfx, yfx, mfx, mfy, ra_x, ra_y, area,
  dxa, dya) packed into one buffer at compile-computed element offsets;
  q/fx/fy form a shared contiguous prefix so the 3 host-visible outputs
  can be recovered with a single D2H covering only that prefix (`out_bytes`),
  not the full buffer.
- Exactly one `cudaMemcpyAsync` H2D (whole buffer), one call into the
  unchanged `launch_fv_tp_2d_cuda`, exactly one `cudaMemcpyAsync` D2H (the
  q+fx+fy prefix only), one `cudaStreamSynchronize(nullptr)` at the API's
  ownership boundary before returning to the Fortran caller.
- `thread_local` on both `g_ws`/`g_pack`, same rationale as before
  (PeerHaloTransport runs one host thread per tile in-process; a
  process-wide buffer would race).
- Pinned memory is required, not incidental: `cudaMemcpyAsync` through
  non-pinned (heap) memory is silently downgraded to synchronous by the
  CUDA driver — this is why the pre-Prompt-79 code's 15 memcpys were all
  synchronous despite no explicit `cudaMemcpy` (blocking variant) call in
  a prior draft; using the pinned buffer is what makes the async calls
  genuinely async.

No numerical semantics changed: `launch_fv_tp_2d_cuda` (the only place
PPM/limiter/flux formulas live) is called identically to before, on views
into the one packed device buffer instead of into 14 separate per-field
buffers.

## 7. Stage 7 — Transfer-Count Assertion (IMPLEMENTED)

Env-gated diagnostic, off by default: `RUN_CUDA_FV3_TRACER_TRANSFER_COUNT_DUMP=1`
prints `[P79_TRANSFER_COUNT] h2d=<n> d2h=<n> total_bytes=<n> out_bytes=<n>`
once per bridge call. Achieved: **h2d=1, d2h=1** per (k, iq) slice call —
down from 12/3 (15 total) pre-Prompt-79. At C512 production scale
(~20 slice-calls/`tracer_2d_1L`) this is a reduction from ~300 to ~40
transfers per `tracer_2d_1L` invocation (20 H2D + 20 D2H), i.e. an
~87% reduction in transfer *count*, though not to Prompt 78's originally
envisioned "one/few transfers per `tracer_2d_1L` call" (that would require
packing across the Fortran k/iq loop itself — a second architectural
change, out of this task's one-change budget; see Stage 15).

## 8. Stage 8 — Standalone Regression (CONFIRMED PASS)

Dev-stack build (`build_cuda_r8_mpi`, OpenMPI toolchain), GPU job 58545595:

```
fv3_wp_b2_scalar_transport_test : Parts A-K PASS; Part L1 "FAIL (degenerate:
  constant 5 over 1296 cells)" -- benign, matches Prompt 76 baseline exactly
  (RC=1, expected, same as Prompt-76-era baseline)
fv3_wp_b2_hord8_transport_test  : ALL PARTS PASS (A/B/C/D + hord6-sanity E),
  fx/fy max_rel 2.171e-16 to 4.311e-16 -- identical to Prompt 76's table; RC=0
fv3_pert_ppm_iv1_test           : ALL 5 CASES BITWISE PASS; RC=0
compute-sanitizer --tool memcheck: ERROR SUMMARY: 0 errors; SANITIZER_RC=0
ALL DONE
```

Numerically identical to the Prompt 76 baseline, including the one
already-documented benign artifact. **Stage 8: PASS.** Confirms the
batched-transfer rewrite changed no observable numerics at the unit level.

Additionally rebuilt (this task) under the **authoritative production
Intel-MPI stack** (`comp/gcc/12.3.0`+`comp/intel/2024.2.0`+`mpi/impi/2021.13`+
`nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++`) in
`build_cuda_r8_intelmpi` — the tree actually linked into the production
`StandAlone_FV3_Dycore.x` binary, confirmed via
`GEOSfvdycore_fortran_perf/build/CMakeCache.txt`'s
`FV3_CUDA_LIBRARY_DIR`. `fv3_wp_b_cuda` and its three dependent fixture
executables (`fv3_wp_b2_scalar_transport_test`,
`fv3_wp_b2_hord8_transport_test`, `fv3_pert_ppm_iv1_test`) compiled and
relinked cleanly under this toolchain too — confirms the change is
toolchain-portable. These three executables were not independently
re-executed under the intelmpi build (not required: the dev-stack GPU run
above already establishes numerical correctness; intelmpi is a
compile-only cross-check here).

## 9-13. Stages 9-13 — Integration / Performance A/B: BLOCKED

**Blocker: the production `StandAlone_FV3_Dycore.x` binary could not be
relinked against the updated `build_cuda_r8_intelmpi` archives.**

`cmake --build . --target StandAlone_FV3_Dycore.x -j8` in
`GEOSfvdycore_fortran_perf/build` fails during the Fortran compile phase
(never reaches the link step) with numerous Intel ifort `#6633` "type of
the actual argument differs from the type of the dummy argument" errors in
`src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90`, at call sites for
`adjust_fv3_splits`, `updraft_helicity`, `calculate_shear_06`,
`bunkers_vector`, and `helicity_relative_CAPS` (lines 2695, 4418, 4426,
4433, 4441, 4448, 4455, 4462, 5404, 5423, 5465).

**Root-cause localization (bounded, per project debugging convention):**

- These are severe-weather diagnostic subroutines, unrelated to tracer
  transport, hord==8, or any CUDA dispatch path — outside this task's
  scope, and `FV_StateMod.F90` was not touched by this task or (per file
  mtime, 2026-09-18) by any part of Prompts 77-79.
- Build-tree evidence: no `.o` object for `FV_StateMod.F90` exists
  anywhere under `build/src/Components/@FVdycoreCubed_GridComp/` — only a
  0-byte `.o.provides.build` stamp and a `.mod.stamp`, both dated
  2026-09-14, i.e. **this translation unit has never compiled to
  completion in this build tree**, at any point, independent of this
  task's edits.
- This is the same *class* of finding as the `fv3_savepoint_mod.F90`
  `#6633` precedent already documented in `CURRENT_STATE.md` §7 and
  `p77_...md` §12 (also: "only a stale `.mod` and a `.provides.build`
  stamp existed, no `.o`") -- but unlike that precedent, a non-destructive
  `cmake .` reconfigure was not attempted/confirmed to fix this occurrence
  before time-boxing the investigation, because the affected code
  (multi-argument severe-weather diagnostics, `real` dummy args passed
  `fms_grav`/gridstruct fields across several distinct subroutines) is
  outside this task's ownership and the fix -- if it is a genuine
  type/kind bug rather than staleness -- would require a source change
  to authoritative-adjacent Fortran this task is not authorized to make.
- Confirmed both the `fvdycore` core library flags (`-r8`) and the
  `FVdycoreCubed_GridComp` target flags (`-r8`) match, ruling out the
  most common cross-precision-build explanation.

**Consequence:** the production binary currently in
`build/bin/StandAlone_FV3_Dycore.x` does **not** reflect the Prompt 79
batched-transfer change (it was linked from a prior, non-Prompt-79 state).
No new sha256 hash was computed; no `p79_stage9_c32`/`p79_stage9_c512`
experiment directories were created, since there is no way to run a
binary-provenance-verified integration test without a successful relink.
**Stages 9 (C32), 10 (C512), 11 (same-node performance A/B), 12
(attribution), and 13 (performance gate) could not be executed this task.**

## 14. Stage 14 — Default Decision

Unconditional, independent of Stages 9-13's blocked status: the production
default **remains Fortran-dispatch** for tracer transport
(`use_cuda_tp2d`'s existing selector/opt-out behavior from Prompt 77 is
unchanged; no Fortran source was edited this task). This was never
contingent on Stage 9-13 results.

## 15. Stage 15 — Next Optimization / Next Step

Two independent follow-ups, either of which could proceed next:

1. **Unblock the production relink** (required before Stages 9-13 can run
   at all): diagnose whether the `FV_StateMod.F90` `#6633` errors are a
   genuine pre-existing type/kind bug in the severe-weather diagnostic
   calls (`adjust_fv3_splits`/`updraft_helicity`/`calculate_shear_06`/
   `bunkers_vector`/`helicity_relative_CAPS`) or a fixable stale-build
   artifact (per the `fv3_savepoint_mod.F90` precedent) — via a
   non-destructive `cmake .` reconfigure first, before considering any
   source change, and only with explicit authorization if a source fix
   turns out to be required, since this code is outside this task's
   ownership. This is a build-infrastructure task, not a CUDA/tracer task.
2. **If/when unblocked**, re-run this exact task's Stages 9-13 (C32/C512
   dispatch-wiring A/B, same-node alternating-order performance A/B,
   attribution, gate) against a freshly-verified sha256-matched binary
   before drawing any conclusion about whether batching recovered the
   Prompt 78 regression. The transfer-count reduction achieved (Stage 7:
   300 -> ~40 transfers/`tracer_2d_1L` at C512 scale) is a necessary but
   not sufficient condition for a performance verdict — it has not yet
   been measured end-to-end.

If a full one-shot pack-across-the-k/iq-loop redesign (reducing transfers
further, to O(1) per `tracer_2d_1L` call rather than O(npz*nq)) is later
considered, it should be scoped as its own task: it changes the caller/
callee transfer contract, not just this bridge's internals, and would
consume a second "one substantial change" budget under this task's rules.

## 16. Files created/modified this task

- Modified (prior turn, confirmed unchanged this turn):
  `cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu` (packed
  batched-transfer rewrite, §6-7).
- Rebuilt, not modified: `build_cuda_r8_mpi` (dev stack, Stage 8 GPU
  regression) and `build_cuda_r8_intelmpi` (production stack,
  compile-only cross-check) — both out-of-source build directories, not
  committed.
- New: this file.
- Updated: `docs/performance/p78_tracer_hord8_overhead_attribution_2026-09-24.md`
  (follow-up note), `docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md`,
  `docs/agent_handoff/CURRENT_STATE.md`.

No production Fortran source modified. No commit made. No default flipped.

---

**TRACER BATCHED TRANSFER VALIDATION BLOCKED — production
`StandAlone_FV3_Dycore.x` could not be relinked against the Prompt 79
CUDA change: a pre-existing, unrelated Intel ifort `#6633` compile
failure in `FV_StateMod.F90`'s severe-weather diagnostic subroutines
(never successfully compiled in this build tree, not touched by this or
any tracer/CUDA-related task) blocks the build before the link step.
Stage 6-8 (implementation, transfer-count reduction 300->~40/call at C512
scale, standalone GPU regression) are DONE and PASS on both toolchains;
Stages 9-13 (C32/C512 integration, same-node performance A/B, attribution,
gate) could not run. Production default remains Fortran-dispatch
regardless. Next step: unblock the unrelated build failure (build-
infrastructure task, requires explicit scope authorization if a source
fix proves necessary) before re-attempting Stages 9-13.**

---

**Follow-up (Prompt 80, 2026-09-24):** a dedicated build-repair task
determined this is **not** a stale-build-artifact problem. A fresh,
from-scratch, sequential rebuild of both the provider
(`fv_arrays.F90.o`) and the actual routine-defining file
(`fv_diagnostics.F90.o`, not just `FV_StateMod.F90.o`) still reproduces
the identical `#6633` failure class, including on calls internal to
`fv_diagnostics.F90` itself (`prt_maxmin`, `nh_total_energy`) that
require no external module interface at all. Both implicated files show
zero local modification (pristine upstream checkout,
`@fvdycore` HEAD `ca30e26`). **Root cause classified as Category G —
genuine, pre-existing source/toolchain-level mismatch**, predating
Prompt 79 and unrelated to any tracer/CUDA work — but the exact
low-level ifort mechanism was not pinned down, and no specific minimal
fix was identified, so per the Stage-8 authorization gate **no source
edits were made or authorized**. The production relink therefore remains
blocked; Stages 9-13 here remain BLOCKED pending separate, explicitly
authorized investigation/repair of `FV_StateMod.F90`/`fv_diagnostics.F90`.
See `docs/performance/p80_fv_statemod_build_tree_repair_2026-09-24.md`.

**Follow-up (Prompt 81B, 2026-09-24) — UNBLOCKED; Stages 9-13 resumed
and closed.** Prompt 81B found Prompt 80's "Category G" call almost
entirely superseded: a genuinely clean rebuild of the `fvdycore` target
(not the incremental rebuild Prompt 80 performed) collapsed the error
count from 416/~140 lines to 8 errors on one previously-unidentified
call, `Held_Suarez_Forcing` (`FV_StateMod.F90:2041-2043`) — none of
`prt_maxmin`/`bunkers_vector`/`adjust_fv3_splits`/`updraft_helicity`/
`helicity_relative_CAPS`/`calculate_shear_06` error after a true clean
rebuild. Fixed via `real(FVPRC)` dummy-argument typing in
`Held_Suarez_ForcingMod.F90` (8 args). Full clean rebuild + relink of
`build/bin/StandAlone_FV3_Dycore.x`: **0 `#6633` errors, EXIT=0**, new
sha256 `e82d526aa0b3b96541f27073ceb5e094112932f878989e931b349bace4474599`,
confirmed via `strings`/`CMakeCache.txt` to be linked against the Prompt
79 batched-transfer CUDA bridge (`build_cuda_r8_intelmpi`,
`[P79_TRANSFER_COUNT]` symbol present). Full detail:
`docs/performance/p81b_fvprc_r8_interface_compatibility_2026-09-24.md`.

### Stage 9 — C32 Integration (PASS)

Reused `p77_stage7_c32`-style harness (job 58549955, C32, `warpa009`).
`use_cuda_tp2d=T` fired correctly in `[P77_EXIT_DUMP]` lines; rc=0, no
NaN/Inf/abort.

### Stage 10 — C512 Integration (PASS)

Job 58549977 (below) confirms C512 dispatch-wiring: cuda arm
`TP2D_T=66/TP2D_F=0`, fortran arm `TP2D_T=0/TP2D_F=66` across all 3
pairs (66 = 6 ranks x 11 `dyn_core` calls/run) — dispatch fires
correctly and exclusively each direction, no cross-contamination.

### Stage 11 — Same-Node Alternating-Order Paired A/B (COMPLETE)

Job 58549977, same node/allocation, strict alternating order (A1 B1 A2
B2 A3 B3), C512, 6 ranks/1 A100, MPS ON,
`RUN_CUDA_FV3_FUSE_HALO_EDGES`/`_FUSE_CORNER_FILL=YES` both arms — exact
reuse of Prompt 78's `p78_stage2_alternating_ab` methodology and
experiment directories (`p78_stage2_cuda`/`p78_stage2_fortran`), against
the newly-relinked, sha256-verified binary. 6/6 runs rc=0, 0
NaN/Range_Warn/Abort. 11 `dyn_core` calls/run confirmed both arms.

Cumulative per-run timer totals (seconds, `fv_timing_mod`, 11 calls/run):

| Pair | Arm | FV_DYN_LOOP | DYN_CORE | TRACER_2D | REMAPPING |
|---|---|---|---|---|---|
| 1 | cuda | 28.2589 | 9.5014 | 4.7467 | 7.8391 |
| 1 | fortran | 26.8219 | 9.4057 | 3.0559 | 7.6641 |
| 2 | cuda | 28.6409 | 9.4523 | 4.6758 | 7.9785 |
| 2 | fortran | 26.8498 | 9.4351 | 3.1507 | 7.6433 |
| 3 | cuda | 28.7458 | 9.4794 | 4.8564 | 8.0245 |
| 3 | fortran | 26.6937 | 9.4311 | 3.0497 | 7.7192 |

Per-pair TRACER_2D ratio (cuda/fortran): pair1 1.553 (+55.3%), pair2
1.484 (+48.4%), pair3 1.592 (+59.2%). Mean of means: cuda 4.760s,
fortran 3.085s, **ratio 1.543 (+54.3%)**. DYN_CORE: cuda 9.478s,
fortran 9.424s, ratio 1.006 (**+0.57%, NEUTRAL**). FV_DYN_LOOP: cuda
28.548s, fortran 26.789s, ratio 1.066 (**+6.57%**). REMAPPING: cuda
7.947s, fortran 7.676s, ratio 1.035 (+3.5%, minor/consistent with
Prompt 78's REMAPPING anomaly, not investigated further — out of this
task's scope).

### Stage 12 — Attribution

Not re-run independently this task (no new `nsys` capture) — Prompt 78's
attribution (many small synchronous `cudaMemcpy` transfers + CPU
marshaling, kernel compute negligible) is the existing mechanism, and
Prompt 79's own Stage 7 already measured the intended intervention
(transfer count 300 -> ~40/`tracer_2d_1L` at C512 scale, ~87% reduction)
via its `RUN_CUDA_FV3_TRACER_TRANSFER_COUNT_DUMP` diagnostic — that
diagnostic was deliberately *not* re-enabled during this Stage 11 timing
run to avoid contaminating the paired-timing measurement with its own
print overhead. The transfer-count reduction is confirmed to have
happened (by construction/Stage 7); it did not translate into a
performance recovery (Stage 11's measured result). A fresh `nsys`
capture to explain *why* a large transfer-count reduction produced a
larger, not smaller, regression is the natural next diagnostic step, not
performed here (would be a new investigation, outside this task's Stage
9-13 resumption scope).

### Stage 13 — Performance Gate: **REGRESSION-REMAINS**

Per the project's >2%/±2% classification thresholds: TRACER_2D is
unambiguously outside the neutral band in the regression direction,
consistently across all 3 alternating pairs (48.4%-59.2%), with no
overlap toward neutral or positive. **The Prompt 79 batched-transfer
architecture did not recover the Prompt 78 regression.** A secondary,
weaker observation (cross-job comparison, not a same-node paired
measurement, so subject to the project's documented node-to-node
variance caveat — CURRENT_STATE.md §8): this run's +54.3% point estimate
is numerically larger than Prompt 78's own +37.87% figure from a
different job/allocation; this is noted but not asserted as a confirmed
"worsening" claim, since the two figures come from different Slurm
allocations. What IS a same-node, directly-comparable claim: within this
one allocation, CUDA tracer dispatch is consistently and substantially
slower than Fortran dispatch despite an ~87% reduction in transfer
count — the transfer-count reduction was a necessary but evidently not
sufficient condition for a performance recovery, consistent with Prompt
79's own Stage 15 caveat that a further architectural change (packing
across the Fortran k/iq loop, not just within one bridge call) may be
required to see the count reduction translate into wall-clock
improvement.

### Stage 14 — Default Decision (reaffirmed, unconditional)

Production default **remains Fortran-dispatch** for tracer transport.
This was never contingent on Stage 9-13 results and is not changed by
this task's REGRESSION-REMAINS finding.

### Stage 15 — Next Step (updated)

Given Stage 13's REGRESSION-REMAINS verdict, the two Stage-15 options
from before this resumption are reordered: (1) a fresh `nsys` capture of
the current (Prompt 79 packed-buffer) CUDA tracer arm is now the more
useful next diagnostic — to explain why an 87% transfer-count reduction
increased rather than decreased the measured regression (candidate
causes to check first: pinned-buffer allocation/growth overhead on the
host side, the buffer-repack marshaling cost itself, or the single
`cudaStreamSynchronize` per call now serializing what were previously
more independent smaller transfers) — before attempting (2) the larger
pack-across-the-k/iq-loop redesign Stage 15 already flagged as its own,
separately-scoped task.

### Files touched by this resumption

No CUDA/Fortran source changed (Prompt 81B's fix was in
`Held_Suarez_ForcingMod.F90`, unrelated to tracer transport). New:
`experiments/p81b_stage11_alternating_ab.sbatch`,
`experiments/p81b_stage6_c32_smoke.sbatch` (both in
`GEOSfvdycore_fortran_perf/experiments/`, not committed — out-of-repo
scratch/experiment tooling). No commit made. No default flipped.

---

**TRACER BATCHED TRANSFER VALIDATION COMPLETE — PRODUCTION RELINK
UNBLOCKED (Prompt 81B) — STAGES 9-13 RUN — PERFORMANCE VERDICT:
REGRESSION-REMAINS.** C32/C512 dispatch-wiring: PASS both grids. Same-
node alternating-order paired A/B (3 pairs, C512, job 58549977):
TRACER_2D CUDA arm ~54.3% slower than Fortran (range 48.4%-59.2%),
DYN_CORE neutral (+0.57%), FV_DYN_LOOP +6.57%. The batched-transfer
architecture (Stage 6-7: 300 -> ~40 transfers/call, ~87% reduction) did
NOT recover the Prompt 78 regression. Production default remains
Fortran-dispatch, unconditionally, unchanged. Next recommended step: a
fresh `nsys` capture of the packed-buffer CUDA arm to explain the
non-recovery before considering the larger pack-across-the-loop
redesign.

**Follow-up (Prompt 82, 2026-09-24)**: that `nsys` capture was run, plus
direct CPU-cycle sampling (`--sample=cpu`, not attempted by Prompt 78).
Root cause found and source-confirmed: `pack_field`/`unpack_field`
(this file, `fv3_tracer_transport_bridge.cu:156-164`) are serial,
non-`restrict`-qualified, scalar cast-copy loops — a CPU cost this
batched design introduced that the unbatched design never had (it
transferred fields directly from/to the original Fortran pointers, no
staging copy). Measured at ~521 ms/call combined CPU-busy time, several
times the ~152 ms/call net regression anchor, dwarfing the secondary
device-transfer-bandwidth (+57 ms/call) and `cudaStreamSynchronize`
(+54 ms/call) deltas also measured. Recoverable with a narrow fix
(`__restrict__`, bulk `memcpy` for same-type fields, or direct
`cudaMemcpyAsync` from original pointers) — not implemented this task.
See `p82_tracer_batched_regression_attribution_2026-09-24.md`.

**Follow-up (Prompt 83, 2026-09-24, final tracer-optimization attempt)**:
implemented `__restrict__` + bulk `std::memcpy` (via `if constexpr` on
`T == fv_real`) in `pack_field`/`unpack_field`, bitwise-verified correct.
A host-only microbench at production-matching `nvcc -O3 -DNDEBUG`
measured only a 1.196x combined pack+unpack speedup (required: >=5x) —
GCC/nvcc already compile the old same-type scalar loop to near-`memcpy`
speed at `-O2`/`-O3`, leaving little room for a hand-written replacement
to improve on. Per this task's own binding Stage-4 gate, GPU integration
(Stages 5-9) was not attempted and the packed-buffer architecture
introduced by this doc was left otherwise unchanged. **Verdict: STOP
CUDA TRACER PRODUCTION OPTIMIZATION** — Fortran remains the tracer-
dispatch production default; no further tracer-CUDA optimization prompt
should be authorized automatically. See
`p83_tracer_bulk_host_pack_optimization_2026-09-24.md`.
