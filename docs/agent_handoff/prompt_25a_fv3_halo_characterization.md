# Prompt 25a — Real FV3 c32/R8 Halo Workload Characterization (Diagnostic Only)

Model: Sonnet. Diagnostic-only, per user directive: no CUDA-aware MPI
implemented, no numerical kernel changed, no runtime redesigned, no `ncu`,
no `nsys` run (existing Prompt 24 profile was sufficient), no source
changes, no GPU job submitted. This whole task was answered by static
derivation from the production CUDA source (`cuda/dyn_core/`) at HEAD
(`fa6623b`, unchanged relevant files since Prompt 24 — see §0) cross-checked
against the exact `input.nml` used by the real `job 58295284`/`58295418`
run that produced Prompt 24's 105.56 ms/call figure. Because every quantity
needed (group liveness, field membership, message geometry, resolve-call
count/gating) is a compile-time `constexpr` function of a fixed, already-run
production configuration — not a runtime-varying quantity — static
derivation is exact, not a sample, and needed no instrumentation or job.

## 0. Method and Its Validity

Inputs:
- `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp` (`MpiHaloTransport::transfer()` —
  the entire per-group D2H/sync/MPI/H2D sequence, one file, ~400 lines).
- `cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp` (`halo_group_spec()` field
  tables, `group_slot_elements()`/`group_buffer_elements()` geometry formulas,
  `fv3_domain_halo_width = 3`).
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (`perform_halo_exchange()`, the
  single `tr->transfer()` call site at :928; `resolve_halo()` at :952; the
  acoustic-substep script's `resolve(FV3_GPU_H_*)` call list at :2063-2346,
  which is an explicit ordered sequence, not a data table, so every gating
  condition is visible directly in source).
- `build_group_fields()` (:746-795), which reduces a group's static
  `field_count` to however many of its fields are actually registered for
  the run (this is where `cappa` drops out under non-`MOIST_CAPPA`).
- The real `input.nml` from
  `.../GEOSfvdycore_fortran_perf/experiments/p24_c32_r8_cuda/scratch_c32_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8/input.nml`:
  `npx=33, npy=33, npz=10, k_split=1, n_split=3, hydrostatic=.F., grid_type=0,
  hord_mt=hord_vt=hord_tm=6, hord_dp=-6, nord=2, vtdm4=0.01, a_imp=1.0`.
  `d_con` is **not** set in this file and its GFDL default
  (`fv_arrays.F90:766`) is `0.` — confirmed by direct grep of the checked-out
  `@fvdycore/model/fv_arrays.F90` in the fixture tree.
- Git history check: the halo/runtime files above were last touched by
  `7dda37b`/`eb6d689` (launch-fusion/orchestration optimizations, both
  strictly kernel-launch-count changes per their own commit messages, not
  halo-group/field/geometry changes) and `823158b` (validation only), all at
  or before `fa6623b` (current HEAD). No commit changes group liveness,
  field membership, or geometry formulas after Prompt 24's run. The
  derivation below is valid for the exact configuration Prompt 24 measured.

This is exact enumeration of a deterministic dispatch script, not a
statistical sample — there is no run-to-run variability in any of the counts
below at fixed `n_split`/`nord`/`d_con`/`grid_type`.

## 1. Real FV3 c32/R8 Halo Message-Size Distribution (TASK 1)

Geometry (from the real config): `nx = npx-1 = 32`, `ny = npy-1 = 32`,
`npz = 10`. `along = max(nx,ny)+1 = 33`. `fv3_domain_halo_width = 3`.
Ordinary groups: `ring_begin=1`, so ring count (`width`) `= 3`.
`H_DWIND_SEAM`: `ring_begin=0`, `halo_width=0`, ring count `= 1`.

**Per-message size is a function of ring count only (not of field), so
there are exactly two distinct message sizes in the entire real c32/R8
dyn_core call:**

| Message class | Ring count | elements/message (`width*along*(npz+1)`) | bytes/message (R8) |
|---|---:|---:|---:|
| Ordinary group edge/field message | 3 | 3×33×11 = 1089 | **8712 B (8.5 KB)** |
| `H_DWIND_SEAM` edge/field message | 1 | 1×33×11 = 363 | **2904 B (2.8 KB)** |

Per-group field count (after `build_group_fields()` drops unregistered
fields — `cappa` is absent because `MOIST_CAPPA` is off in this build/config)
and per-group message/byte totals for **one transfer() invocation**
(messages/invocation `= 8 × field_count`: 4 edges × field_count `Irecv` +
4 edges × field_count `Isend`; buffer bytes/invocation
`= field_count × 4 × slot_bytes`, i.e. one D2H and one H2D copy of that size):

| Group | field_count | msgs/invocation | bytes/invocation (D2H=H2D) | resolves/call |
|---|---:|---:|---:|---:|
| H_DWIND_IN | 2 (u,v) | 16 | 69,696 B | 3 |
| H_W_IN | 1 (w) | 8 | 34,848 B | 3 |
| H_GZ_FIRST | 1 (gz) | 8 | 34,848 B | 1 |
| H_DIVGD | 1 (divgd) | 8 | 34,848 B | 3 |
| H_CWIND | 2 (uc,vc) | 16 | 69,696 B | 3 |
| H_DSCALAR | 3 (q_con,delp,pt) | 24 | 104,544 B | 4 |
| H_ZH | 1 (zh) | 8 | 34,848 B | 3 |
| H_PKC2 | 1 (pkc) | 8 | 34,848 B | 3 |
| H_DWIND_SEAM | 2 (u,v) | 16 | 23,232 B | 1 |
| H_HEAT | — | — | — | **0 (d_con=0, gated off)** |
| H_Q_INLINE, H_DWIND_NEXT | — | — | — | 0 (never resolved, by design) |

**Histogram, aggregated over the whole dyn_core call** (312 individual
messages total — see §2):

| Size bucket | Message count | Fraction of messages | Total bytes | Fraction of bytes |
|---|---:|---:|---:|---:|
| ≤4 KB | 16 | 5.1% | 46,464 B | 1.8% |
| 4–16 KB | 296 | 94.9% | 2,579,952 B | 98.2% |
| 16–64 KB | 0 | 0% | 0 | 0% |
| 64–256 KB | 0 | 0% | 0 | 0% |
| 256 KB–1 MB | 0 | 0% | 0 | 0% |
| >1 MB | 0 | 0% | 0 | 0% |

**Every real FV3 c32/R8 halo message is ≤16 KB.** There are no
messages anywhere near the 256 KB–1 MB range where the synthetic 6-rank
probe (Prompt 25 preamble) measured its largest CUDA-aware benefit (1.30x).
At the sizes FV3 actually uses (~2.8–8.7 KB), that same probe measured only
1.018x–1.021x at 6 ranks sharing one A100 — i.e. next to nothing.

## 2. Exchange Count (TASK 2)

Per-substep `resolve(FV3_GPU_H_*)` calls, read directly off the acoustic-loop
script (`fv3_gpu_runtime.cpp:2063-2346`), gated exactly as the real config
resolves them (`nord=2>0` so `H_DIVGD` fires every substep; `d_con=0` so
`H_HEAT` never fires; `grid_type=0<4` and not nested so `H_DWIND_SEAM` fires
once at `it==n_split`):

| Group | Gating | Resolves at n_split=3 |
|---|---|---:|
| H_DWIND_IN | every substep | 3 |
| H_W_IN | every substep | 3 |
| H_DSCALAR (first site, :2089) | `it==1` only | 1 |
| H_DIVGD | `nord>0` (true) | 3 |
| H_GZ_FIRST | `it==1` only | 1 |
| H_CWIND | every substep | 3 |
| H_DSCALAR (second site, :2247) | every substep | 3 |
| H_ZH | every substep | 3 |
| H_PKC2 | every substep | 3 |
| H_DWIND_SEAM | `it==n_split` only | 1 |
| H_HEAT (post-loop, :2346) | `n_con!=0 && d_con>1e-5` — **false** (`d_con=0`) | 0 |

**Total halo transport invocations per dyn_core call: 24**
(= 3+3+1+3+1+3+3+3+3+1; each is exactly one `MpiHaloTransport::transfer()` call).

Because every invocation does exactly one D2H, one `cudaEventSynchronize`,
and one H2D (`fv3_gpu_halo_mpi.cpp:207-276`):

- **Total MPI messages per dyn_core call: 312** (sum of `8×field_count`
  over all 24 invocations — 156 `Isend` + 156 `Irecv`; see §1 table for the
  per-group breakdown).
- **D2H copies: 24**
- **H2D copies: 24**
- **Halo-related `cudaEventSynchronize` calls: 24**

Total bytes moved (D2H, and separately H2D) per dyn_core call: **1,312,608 B
(≈1.28 MB)** each direction — sum of the "bytes/invocation" column in §1
across all 24 invocations. This matches Prompt 24's independently-measured
"H2D+D2H copy ~0.66 ms/call (<1% of total)" nsys figure: ~1.3 MB each way is
a trivial DMA transfer, consistent with a sub-millisecond combined cost.

Per-acoustic-substep breakdown (n_split=3): substep 1 does 8 resolves
(DWIND_IN, W_IN, DSCALAR×2, DIVGD, GZ_FIRST, CWIND, ZH, PKC2 = 9, actually —
see exact list above; substeps 2 fires 8, substep 3 fires 9 including the
seam). Not tabulated further per-substep beyond the gating table above —
the totals in this section are what matters for the estimate in §4.

## 3. Synchronization Origin (TASK 3)

`git grep -n "cudaEventSynchronize" -- 'cuda/dyn_core'` (excluding
tests/benchmarks) returns **exactly one call site**:
`cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp:216`, inside
`MpiHaloTransport::transfer()`, on the transport's own `ev_d2h_` event. This
one call site accounts for all 24 halo-related `cudaEventSynchronize` calls
per dyn_core call (§2) — it is the **only** host-blocking wait anywhere in
the production halo path. (Other `cudaStreamSynchronize`/
`cudaDeviceSynchronize` calls exist in `fv3_gpu_runtime.cpp` at lines
243-244 — debug-checkpoint, gated by `FV3_GPU_DEBUG_TILE`/`_EVERY`, disabled
during Prompt 24's timed runs — and at 1366/2392-2393/2408 — one-time
teardown/detach boundaries, not on the per-call acoustic-loop path. None of
these contribute to the ~56% figure.)

**Classification (single category, all 24 occurrences):**

| Class | Meaning | Applies? |
|---|---|---|
| A | required before D2H staging | No — this wait happens *after* the D2H is issued |
| **B** | **required before MPI uses the staged host buffer** | **Yes — all 24/24** |
| C | required after MPI / before H2D or unpack | No — H2D is enqueued on the same stream immediately after `MPI_Waitall` returns, with no separate sync; the unpack that follows is stream-ordered, not host-synced |
| D | unrelated to halo staging | No |
| E | other | No |

Every one of the 24 per-call `cudaEventSynchronize` calls exists for exactly
one reason: `MPI_Isend`/`MPI_Irecv` are about to be called on a **host**
pointer (`h_send_`/`h_recv_`), and the host must know the D2H copy that
filled `h_send_` has actually landed before handing that pointer to MPI.

**How much of the ~56% would plausibly disappear under a device-direct
transport? Not all of it, and not straightforwardly estimable from this
alone — see §4.** The key fact bounding the answer: the *duration* of the
D2H copy this wait is nominally waiting for is at most a few hundred
microseconds per call (§2: ~1.28 MB total D2H per call, and Prompt 24
independently measured all D2H+H2D combined at <1% of the 105.56 ms
call, i.e. well under 1 ms). But Prompt 24 attributes ~56% (~59.27 ms/call)
to this same `cudaEventSynchronize` bucket — roughly **2.47 ms average per
occurrence** (59.27 ms / 24). A wait whose *nominal* target (a sub-µs-to-µs
scale D2H copy of a few KB–100 KB) is two-to-three orders of magnitude
smaller than its *measured* duration is not, in the main, waiting for that
copy — it is overwhelmingly waiting for the GPU to drain a backlog of
already-enqueued work (the pack kernel and whatever preceded it on the
transfer/compute streams) before the event it is watching can even fire.
That backlog-drain component is not specific to host staging: any transport
that still needs the host to know a device-resident buffer is ready before
touching it (see §5) pays the same wait for the same reason.

## 4. Estimated Device-Direct Ceiling (TASK 4)

**Estimate: LOW (<10% potential whole-call improvement).**

Do NOT read this as a precise performance prediction — it is a coarse
bound built from three independent pieces of evidence, all pointing the
same direction:

1. **Message-size evidence (§1).** Every real FV3 halo message is ≤16 KB
   (94.9% are exactly 8.5 KB). The synthetic 6-rank/1-A100 probe measured
   CUDA-aware speedup of only **1.018x–1.021x** at 4–16 KB under 6-way GPU
   sharing — i.e., 1.8–2.1%, the smallest benefit at any tested size, and
   specifically the size range FV3 actually needs. The probe's largest
   measured benefit (1.30x at 1 MB) is irrelevant here: FV3 never produces
   a message anywhere near that size.
2. **Copy-removal evidence (§2/§3).** The absolute ceiling on "benefit from
   no longer copying" is bounded above by the D2H+H2D time itself, already
   independently measured by Prompt 24 at <1% of the call (≈0.66 ms of
   105.56 ms). Even removing 100% of that yields well under 1% whole-call
   improvement by itself.
3. **Synchronization evidence (§3).** The ~56% `cudaEventSynchronize`
   bucket is 100% category B (waiting for MPI to be allowed to touch a host
   buffer) but its measured duration (~2.47 ms/occurrence) vastly exceeds
   the copy it is nominally gating (sub-ms), meaning most of that time is
   GPU-work-backlog drain, not "waiting on a copy" — and that backlog-drain
   wait does not go away with a device-direct transport; it is only
   retargeted at the pack-kernel-completion event instead of the D2H-copy
   event (§5). The *removable* slice is bounded by the *copy* portion of
   that wait, which item 2 already bounds at <1%.

Dominant assumption behind "LOW" rather than "near-zero": that some
fraction of the backlog-drain time is incidentally shorter when the
scheduler doesn't also have to fit the D2H/H2D copies into the same
work queue, and that eliminating 156 sends' worth of host-buffer
dependency might allow marginally better overlap of the *next* group's
pack with the *current* group's MPI wait. Neither effect is bounded by
direct measurement here; both are plausibly single-digit-percent at most,
which is why the estimate is LOW rather than "0%" — not because of a
positive measurement of a specific mechanism.

## 5. Six-Rank/One-GPU Architecture (TASK 5)

**Six-rank/one-GPU contention remains after device-direct MPI: YES.**

Answering the three sub-questions directly:

- **Would each process still need to wait for its pack stream before MPI
  accesses the device buffer? YES.** Standard (non-stream-aware) MPI,
  CUDA-aware or not, requires the buffer it is handed to be valid *at the
  time of the call*, not merely stream-ordered-valid-eventually. A
  device-direct transport still needs a host-side wait for the pack
  kernel's completion (functionally the same category-B wait as today,
  just watching a pack-completion event instead of a D2H-completion event)
  before it can safely call `MPI_Isend`/`MPI_Irecv` on the device pointer.
  This is exactly the finding foreshadowed in the "CURRENT FACTS": the
  wait does not disappear, only its *nominal* target changes.
- **Would each process still need to wait for MPI completion before
  unpack? YES**, unchanged from today — `MPI_Waitall` remains the
  synchronization point before the runtime can enqueue the unpack kernel
  that reads the received buffer, in both transports.
- **Would each of the 6 processes still synchronize independently from the
  other 5? YES.** Six MPI ranks are six separate host processes with six
  separate CUDA contexts sharing one A100 in both transports; nothing about
  handing MPI a device pointer merges those contexts or serializes them
  under one scheduler decision. The synthetic probe's own measurement is
  direct evidence of this: CUDA-aware's raw `MPI_Wait` cost, which should in
  principle be near-zero for tiny messages, instead *rose* from a 2-rank
  mean of 340–376 µs to a 6-rank mean of 1137–1263 µs — a **~3.3x increase
  driven purely by adding four more ranks sharing the same GPU**, not by
  message size (the 6-rank MPI-wait mean was nearly flat across all five
  tested sizes). That is GPU/CUDA-IPC context contention showing up
  *inside* the supposedly-fast device-direct path itself.

**Primary remaining bottleneck: six independent MPI processes / CUDA
contexts contending for scheduling on one shared A100 — not host-staging
copy overhead.** Host staging is real and category-B synchronization is
100% attributable to it by call site (§3), but its magnitude is dominated
by GPU work-queue backlog drain and (per the probe) by shared-GPU
scheduling contention among six independent contexts, neither of which a
transport-level change (host-staged → device-direct) removes.

## 6. Final Answers (verbatim template)

- Model: SONNET
- Actual FV3 c32 halo message distribution: two sizes only — 8712 B (94.9%
  of messages, 98.2% of bytes, ordinary groups, ring width 3) and 2904 B
  (5.1% of messages, 1.8% of bytes, `H_DWIND_SEAM` only, ring width 1).
  Nothing above 16 KB occurs.
- Total halo transport calls per dyn_core: 24
- Total MPI messages per dyn_core: 312 (156 `Isend` + 156 `Irecv`)
- D2H operations: 24
- H2D operations: 24
- Halo-related cudaEventSynchronize calls: 24 (100% of them, single call site)
- Fraction of messages ≤64 KB: 100%
- Fraction of bytes ≤64 KB: 100%
- Fraction of messages ≥256 KB: 0%
- Fraction of bytes ≥256 KB: 0%
- Estimated synchronization removable by CUDA-aware MPI: LOW — bounded
  above by the D2H+H2D copy time itself (<1% of call, ≈0.66 ms), since the
  dominant share of the measured ~56% is GPU-backlog-drain wait that a
  device-direct transport still incurs against a pack-completion event
  instead of a D2H-completion event.
- Estimated whole-call CUDA-aware MPI benefit: **LOW**
- Six-rank/one-GPU contention remains after device-direct MPI: **YES**
- Primary remaining bottleneck: six independent MPI processes/CUDA contexts
  sharing one A100's scheduler — evidenced directly by the synthetic
  probe's own CUDA-aware MPI-wait time rising ~3.3x from 2 to 6 ranks at
  matched message sizes, not by copy overhead.
- Recommendation: **DO NOT IMPLEMENT YET — INVESTIGATE RANK/GPU
  ARCHITECTURE.** The evidence points at the six-ranks-per-one-GPU mapping
  itself (context/scheduling contention), not at the host-staged-vs-
  device-direct transport choice, as the dominant lever. Implementing
  CUDA-aware MPI in FV3 now would touch the production MPI/runtime stack
  for a measured-LOW ceiling while leaving the actual dominant contention
  mechanism untouched.

## 7. Explicit Non-Claims

- No CUDA-aware MPI was implemented in FV3. No numerical kernel was
  changed. No runtime was redesigned. No `ncu` or `nsys` was run for this
  task. No source was modified. No GPU job was submitted.
- This task does not change any IMPLEMENTED/COMPILED/GPU EXECUTED/
  NUMERICALLY VALIDATED status for any component.
- The LOW estimate in §4 is explicitly a coarse bound built from existing
  measurements (Prompt 24's nsys profile, this session's synthetic 6-rank
  probe) plus exact source-derived counts — not a new performance
  measurement and not a claim of precision.
- `CURRENT_STATE.md` is not being updated by this task: the existing §11
  "Next Recommended Task" already correctly frames CUDA-aware MPI as
  needing scope confirmation before implementation and already flags
  host-staged MPI as the corroborated secondary factor behind six-rank GPU
  sharing; this task's conclusion (six-rank/one-GPU contention, not the
  transport choice, is the dominant lever) sharpens that framing but does
  not reverse or materially redirect it, so no edit was made.
