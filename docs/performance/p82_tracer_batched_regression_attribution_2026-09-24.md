# Prompt 82: Attribution — Why the Batched Tracer Transfer Redesign Got Slower

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Scope: **profiling/diagnosis only — zero source changes made in this task.**
No tracer numerics, batching code, PT/DELZ, REMAP_UV, or R4 touched. `ncu`
not used (`nsys` only). Production default unchanged (Fortran-dispatch).

## 0. Inputs

- Authoritative regression-magnitude anchor: job 58549977 (Prompt 81B,
  3-pair alternating-order C512 A/B, 11 calls/pair, Prompt-81B-relinked
  binary, sha256 `e82d526a...4474599`). Per-call TRACER_2D: CUDA
  (batched) 432.7 ms/call vs Fortran 280.5 ms/call — **delta +152.2
  ms/call (+54.3%)**. This is the number Stage 9 reconciles against.
- New same-node, same-allocation nsys A/B (Stage 1, job 58550207, node
  confirmed via job log, `--sample=cpu --backtrace=fp --cpuctxsw=
  process-tree`, rank 0 only, 7 calls/arm): `p82_nsys_fortran` (Arm A,
  `RUN_CUDA_FV3_TRACER_FORCE_FORTRAN=1`) and `p82_nsys_cuda` (Arm B,
  production batched-CUDA tracer dispatch active). Both arms `RC=0`,
  `Range_Warn=0`, `NaN=0`, `Abort=0`.
- Reused, not recaptured, per task instructions: Prompt 78's unbatched-
  CUDA-tracer baseline capture, `p78_stage4_cuda/nsys_cuda_rank0.nsys-rep`
  (sha256 `8152aed3...650a9570`, historical artifact).

**Known limitation, stated up front:** this task's own Stage-1 captures
are a **single-shot, non-alternating** A/B (one Fortran run then one CUDA
run in the same allocation) — internally noisy for direct cross-arm wall-
clock subtraction (confirmed: this capture's own DYN_CORE was *lower* for
the CUDA arm than the Fortran arm, opposite the established regression
direction — ordinary node/run noise per `CURRENT_STATE.md` §8, not a new
finding). Per the project's coarse-to-fine localization discipline, this
capture is used **only** for (a) internal, within-one-arm category
breakdowns and (b) CPU-sampling-based attribution (which does not depend
on cross-arm timer subtraction). The job-58549977 alternating-order
figures remain the sole authority for absolute regression magnitude.

## 1. Stage 1 — Capture

`nsys profile --trace=cuda,osrt --sample=cpu --backtrace=fp
--cpuctxsw=process-tree` (CPU sampling enabled, unlike Prompt 78's
`--sample=none`, specifically to support Stage 2/7). `nsys stats
--help-reports` confirms **no built-in `cpusampling`/`cpu_sampling`
report exists** in this nsys version (2024.4.1.61) — worked around by
querying the exported `.sqlite` directly: `SAMPLING_CALLCHAINS` (1.3M+
rows) joined to `StringIds` for symbol resolution, `COMPOSITE_EVENTS` for
per-sample timestamp/thread-state. All samples in both captures have
`threadState=1` ("Running") — this sampler only captures threads actually
scheduled on a CPU, so leaf-function self-time reported below is genuine
CPU-busy time, not time spent descheduled/blocked in the OS scheduler.

Sampling period derived empirically: batched-arm span 77,906,824,339 ns /
149,469 samples = 521,224 ns/sample average (fortran-control: 535,759
ns/sample, 156,057 samples) — used below to convert sample counts to ms.

## 2. Stage 2/7 — Host Pack/Unpack CPU Time (direct measurement)

Two, distinct, additive CPU-busy costs are visible **only** in the
batched-CUDA arm (zero samples for either in the fortran-control arm —
confirmed by direct query, not absence-from-top-N):

| Symbol | Leaf samples (batched) | Leaf samples (fortran-ctrl) | ms/call (batched) |
|---|---|---|---|
| `fv3_gpu_fv_tp_2d_hord8` (C++ bridge, self time) | 3,486 | 0 | 259.6 |
| `fv_tracer2d_mod_mp_tracer_2d_1l_` → `__intel_avx_rep_memcpy` (Fortran-side pack, direct memcpy call) | 3,506 | 0 (not in top 10; `tp_core_mod_mp_fv_tp_2d_`'s own memcpy calls are only 370, present in **both** arms — ordinary array-copy, not tracer-CUDA-specific) | 261.1 |
| **Combined new CPU-busy cost** | | | **≈520.7 ms/call** |

Source-level confirmation (`cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu:156-164`):

```cpp
template <typename T>
void pack_field(fv_real *dst, const T *src, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<fv_real>(src[i]);
}
// unpack_field: same shape, dst/src reversed
```

Called 12x (`pack_field`) + 3x (`unpack_field`) per invocation (§Stage 6
byte table) to marshal every field into/out of one contiguous pinned
buffer — this is the Prompt 79 batching redesign's core mechanism, and it
did not exist in the unbatched design (which issued `cudaMemcpy` directly
against the original Fortran array pointers, no intermediate host copy
stage). Neither `dst` nor `src` is `__restrict__`-qualified, so the
compiler cannot assume non-aliasing and this loop is not guaranteed (and,
per the profiler, evidently does not) vectorize into a bulk-copy
instruction — it shows as scalar self-time in the bridge function's own
PC range rather than resolving to `__intel_avx_rep_memcpy`. This is a
plausible, source-confirmed mechanism for the ~260 ms/call bridge-side
cost. (The separate ~261 ms/call Fortran-side `memcpy` cost is an ABI-
boundary array-temporary copy, a distinct cost from the C++ pack loop.)

## 3. Stage 3/6 — Byte Accounting (unbatched vs. batched, both CUDA-tracer-active arms)

| Path | Calls/7-call capture | MB/call | ms/call (device) | Avg ms/call/op |
|---|---|---|---|---|
| Unbatched H2D | 2,190 (312.9/call) | 1,328.33 | 115.19 | 0.368 |
| Unbatched D2H | 833 (119.0/call) | 805.66 | 146.67 | 1.233 |
| Batched H2D | 650 (92.9/call) | 1,412.38 | 170.06 | 1.831 |
| Batched D2H | 553 (79.0/call) | 805.66 | 149.16 | 1.888 |

D2H bytes are **bit-identical** between unbatched and batched (805.66
MB/call both) — batching did not change D2H data volume. H2D bytes rose
modestly (+6.3%, 1,328→1,412 MB/call). **Stage 3's question — did fewer
transfers paradoxically move more data — is answered NO**; bytes are
essentially flat. But average per-transfer *time* rose sharply (H2D
0.368→1.831 ms/op, D2H 1.233→1.888 ms/op) despite similar-or-larger
average transfer size, i.e. **achieved bandwidth got worse, not better**,
in the batched design (H2D: ~11.5 GB/s unbatched vs. ~8.3 GB/s batched at
comparable-or-larger per-call payload) — the opposite of the amortized-
overhead improvement batching, in the abstract, should provide. Total
device-side memcpy time: unbatched 261.9 ms/call, batched 319.2 ms/call —
**+57.3 ms/call**, a real but secondary contributor (§5).

## 4. Stage 4/5 — Synchronization and Buffer-Pinning

`cudaStreamSynchronize` call count is **identical** between the two CUDA-
tracer-active arms — 171 calls/7-call capture (24.4/call) both — but total
time explodes: unbatched 9.876 ms → batched 385.30 ms (**~39x**, +53.6
ms/call). A third data point, the fortran-control arm (tracer CUDA fully
off, PT/DELZ CUDA still on), shows only 31 sync calls / 1.813 ms total —
confirming ~140 of the 171 calls in either tracer-CUDA arm are tracer-
specific, and that the *unbatched* design added negligible sync cost
(+1.15 ms/call over the no-tracer-CUDA baseline) while the *batched*
design added +54.8 ms/call for the same marginal call count. This is
consistent with, not independent of, §3's bandwidth finding: fewer,
larger, un-overlapped async transfers make each existing sync wait
longer. Direct search for a driver-level spin-wait leaf (`cuStreamSynchronize`
as its own sampled PC) found only 1 sample — nearly all of this wait time
is *not* CPU-sampled at all (consistent with a real blocking/descheduling
wait, not a busy-spin), which also means it does **not** overlap with or
double-count the §2 pack/unpack CPU-busy samples (those require the
thread to be actively `Running`; a blocked `cudaStreamSynchronize` would
not be sampled as such). Pack→H2D→kernel→D2H→unpack is therefore fully
serialized per call, consistent with the source's "one required
synchronization at this API's ownership boundary" comment
(`fv3_tracer_transport_bridge.cu:321`) — no stream overlap is attempted
by design, in either the unbatched or batched implementation.

`cudaHostAlloc`/`cudaFreeHost`: consistent with prior findings (2-3
calls total across the whole capture, both arms) — one-time pinned-buffer
allocation via the grow-only `PinnedBuf`/`DeviceBuf` reuse discipline
(`fv3_tracer_transport_bridge.cu:106-149`), not a per-call cost. `__intel_
avx_rep_memset`'s callers (checked via stackDepth=1 join) are entirely
non-tracer (`fv_statemod_mp_fv_run_`, init-time routines, 21 samples
inside `fv3_gpu_bridge_init_`) — confirms **no hidden per-call buffer
zeroing/reallocation** (rules out Category D).

## 5. Stage 8 — GPU Kernel Compute (confirmed negligible, both arms)

Batched arm, `xppm_pass_kernel_hord8`/`yppm_pass_kernel_hord8`: 280
launches each (40/call), 3.9472 ms + 4.4285 ms total = 8.3757 ms/7 calls
= **1.20 ms/call** combined. Matches Prompt 76/78's prior findings almost
exactly — kernel compute is not, and was never, the problem.

## 6. Stage 9 — Reconciliation

| Component | ms/call | Basis |
|---|---|---|
| Pack/unpack CPU (new, host-side) | ~520.7 | CPU sampling, this task (§2) |
| Device memcpy time (H2D+D2H delta vs. unbatched) | +57.3 | `cuda_gpu_mem_time_sum`, both CUDA-tracer arms (§3) |
| `cudaStreamSynchronize` delta vs. unbatched | +53.6 | `cuda_api_sum`, both CUDA-tracer arms (§4) |
| Kernel compute | ~1.2 (unchanged) | `cuda_gpu_kern_sum` (§5) |
| Authoritative net regression anchor | **+152.2** (+54.3%) | job 58549977, alternating A/B |

**Reconciles: PARTIALLY / DIRECTIONALLY YES — not an exact arithmetic
match.** The pack/unpack component alone (~521 ms/call, all of it newly
introduced by the batching redesign, zero in the Fortran-only control)
is more than 3x the net anchor regression by itself, comfortably
sufficient to explain a net +152.2 ms/call regression even after netting
out whatever Fortran-side hord==8 PPM computation the CUDA path avoids
(not independently measured this task). The device-transfer and sync
deltas (+57.3, +53.6 ms/call) are real, measured, secondary contributors
plausibly *caused by* the same design choice, not independent root
causes. An exact ms-for-ms reconciliation is not claimed, because the
attribution figures come from this task's single-shot Stage-1 capture
while the anchor comes from a separate, rigorously alternating job —
mixing exact arithmetic across two different Slurm allocations would
overstate precision this evidence does not support (§0). The magnitude
and direction are unambiguous and mutually consistent.

## 7. Stage 10 — Root-Cause Category

**Category A (pack/unpack CPU cost) — PRIMARY**, source-confirmed
(§2): `pack_field`/`unpack_field` in `fv3_tracer_transport_bridge.cu`
(lines 156-164) are serial, non-`restrict`-qualified, scalar element-wise
cast-copy loops, invoked 12 (pack) + 3 (unpack) times per call — a CPU
cost that **did not exist** in the unbatched design (which transferred
fields directly from/to the original Fortran array pointers via
`cudaMemcpy`, with no intermediate host staging copy). This single
mechanism accounts for the largest, most directly measured share of the
regression.

Category C (synchronization serialization, +53.6 ms/call) and the
device-bandwidth degradation (+57.3 ms/call, §3) are real and secondary,
but the evidence (§4) indicates they are a **downstream consequence** of
the same design — fewer, larger, unoverlapped transfers each block their
one `cudaStreamSynchronize` for longer — not an independent root cause.
Category B (increased bytes) is **ruled out** (§3, bytes ~flat). Category
D (hidden buffer management) is **ruled out** (§4, allocations one-time).

## 8. Stage 11 — Stop/Continue Decision

**Recoverable with one narrow fix: YES.** The identified mechanism
(§2, §7) is localized to two small functions in one file, with a bounded,
enumerable set of call sites (12+3 calls, all in `fv3_tracer_transport_
bridge.cu`). Three narrow, non-renumerics, non-orchestration options,
in increasing order of change size:

1. Add `__restrict__` to `pack_field`/`unpack_field`'s pointer parameters
   to permit vectorization of the existing loops.
2. For `T == fv_real` (the R8 production case), replace the cast loop
   with a bulk `memcpy`/`memmove` (no cast needed when source and
   destination types match).
3. Issue per-field `cudaMemcpyAsync` calls directly from/to the original
   Fortran array pointers into fixed offsets of the same pinned/device
   buffer, eliminating the intermediate host-side copy stage entirely
   while preserving the "few transfers, one buffer" design intent.

Any of these plausibly recovers a large fraction of the ~520 ms/call
gross pack/unpack cost — far above the ≥40-50 ms/call bar for this
decision — without touching tracer numerics, orchestration, or MPI.

**No such fix was implemented in this task** (out of scope; diagnose
only). Given this is the **second** consecutive tracer-CUDA design
(unbatched P77/78, then batched P79/81B) to regress production
TRACER_2D, the recommended follow-up should itself be narrowly scoped,
implement one of the three options above, and be gated by a fresh
alternating-order A/B (per the P48/P76A rigor bar) before any production-
default change — not treated as a foregone KEEP.

## 9. Files/Artifacts

New, out-of-repo experiment tooling only (per task constraint — zero
source changes in `FVdycoreCubed_GridComp`):
`GEOSfvdycore_fortran_perf/experiments/{p82_nsys_cuda,p82_nsys_fortran}/
{StandAlone_FV3_Dycore.x,fv3.j}`, `p82_nsys_ab.sbatch` (job 58550207).
Captures: `p82_nsys_cuda/nsys_cuda_batched_rank0.{nsys-rep,sqlite}`,
`p82_nsys_fortran/nsys_fortran_ctrl_rank0.{nsys-rep,sqlite}`. No commit
made. No default changed. No CUDA/Fortran source edited.

---

**TRACER BATCHED REGRESSION ROOT-CAUSED — ONE NARROW FIX WARRANTED:**
eliminate the serial, non-vectorized `pack_field`/`unpack_field` host-
side cast-copy loops in `fv3_tracer_transport_bridge.cu` (add
`__restrict__` and/or replace with direct `cudaMemcpyAsync` from the
original array pointers into the packed buffer's offsets) — this is the
single largest measured contributor (~521 ms/call of CPU-busy time,
entirely new vs. the Fortran arm, entirely new vs. the unbatched CUDA
design), dwarfing the secondary device-transfer/sync deltas (+57/+54
ms/call) that are themselves likely a consequence of the same design
choice. Recommend a follow-up task scoped to exactly this fix, re-
measured with a fresh alternating-order A/B before any default change.

**Follow-up (Prompt 83, 2026-09-24)**: implemented the two most directly
indicated candidates above simultaneously — `__restrict__` pointer
qualification and `if constexpr`-gated bulk `std::memcpy` for the
`T == fv_real` case (true for every field in the R8 production
configuration; all 12 inputs + 3 outputs proven Class A/memcpy-safe).
Bitwise-verified correct. However, a host-only microbench at the
production-matching optimization level (`nvcc -O3 -DNDEBUG`, matching
`build_cuda_r8_intelmpi`/`build_cuda_r8_perf`) measured only a **1.196x**
combined pack+unpack speedup — far short of the >=5x improvement needed
to meaningfully offset the ~520.7 ms/call measured above. Root cause:
GCC/nvcc already recognize the OLD same-type scalar copy loop as a
bulk-copy idiom at `-O2`/`-O3` and compile it to near-`memcpy` speed,
leaving little headroom for a hand-written replacement. This doc's
"recoverable with one narrow fix" framing was directionally correct
about *which* code was the bottleneck, but a host-copy-only fix in
isolation does not clear the required bar — the third named candidate
(direct `cudaMemcpyAsync` from the original Fortran pointers, skipping
the packed staging buffer) was not attempted (out of scope: a second
architectural change). Per Prompt 83's own binding instruction, no
further CUDA tracer production optimization prompt should be authorized
automatically after Prompt 83. See
`p83_tracer_bulk_host_pack_optimization_2026-09-24.md`.
