# Prompt 50 — Halo-Transport Host-Blocking Sync: Analysis Result

Date: 2026-09-21. Model: Sonnet (no Opus escalation — semantics were
resolved by source inspection, not left ambiguous).

**Outcome: NO CODE CHANGE MADE.** Stage 0/1 source analysis (before any
edit, per the task's own "do not modify code before this is understood"
rule) found that the one candidate minimal change permitted by the task's
Option A-D menu is **unsafe** — it would introduce a real host-buffer
reuse race, not just remove conservative synchronization. No Stages 3-8
were executed: implementing a change already shown unsafe, then spending a
GPU allocation to test it, was rejected as inconsistent with "do not trade
correctness for performance" and with cost-efficiency doctrine (measure/
reason first; do not implement from an already-falsified hypothesis).

## Stage 0 — Current ordering contract (`fv3_gpu_halo_mpi.cpp::transfer()`, host_staged mode)

Per halo group, on the caller-owned dedicated `transfer_stream` (`cs`,
passed in as `stream`; confirmed via `fv3_gpu_runtime.cpp:912-948` to be a
stream separate from `compute_stream`, joined only by explicit
`order_after`/`cudaStreamWaitEvent` edges — not a shared compute queue):

| # | Op | Where | Buffer | Blocks host? |
|---|---|---|---|---|
| 1 | `cudaMemcpyAsync(h_send_, send, D2H, cs)` | line 229 | device `send` → pinned `h_send_` | No (async, stream-ordered after the pack kernel the caller already enqueued on `cs`) |
| 2 | `cudaEventRecord(ev_d2h_, cs)` | line 240 | — | No |
| 3 | **`cudaEventSynchronize(ev_d2h_)`** | **line 243** | — | **Yes — the target call** |
| 4 | `MPI_Irecv` × 4 edges × field_count | lines 259-290 | writes `h_recv_` | No (posts only) |
| 5 | `MPI_Isend` × 4 edges × field_count | lines 291-302 | reads `h_send_` | No (posts only) |
| 6 | `MPI_Waitall` | line 303 | — | Yes (but not the profiled call — a different, required MPI-level wait, out of scope) |
| 7 | `cudaMemcpyAsync(recv, h_recv_, H2D, cs)` | line 314 | pinned `h_recv_` → device `recv` | No (async; caller's next-enqueued unpack kernel is stream-ordered after it) |

`h_send_`/`h_recv_` are transport-owned pinned buffers, allocated once in
`attach()`, sized for the largest group ever exchanged, and **reused
verbatim across every subsequent `transfer()` call** (every halo group,
every acoustic substep) — there is exactly one `h_send_`/`h_recv_` pair and
one `ev_d2h_` event object for the transport's lifetime, not one per call.

**Why `cudaEventSynchronize` is present (Stage 0's A-F menu): B, plus an
emergent D.**

- **B (D2H completion before MPI) — the documented, explicit reason.**
  `MPI_Isend` (step 5) reads `h_send_`, a plain pinned host buffer from
  MPI's point of view. MPI has no notion of CUDA stream order for a host
  pointer — it may read `h_send_` synchronously the instant `MPI_Isend` is
  called. The host must therefore have positive, host-observable
  confirmation that step 1's D2H has actually landed in `h_send_` before
  calling step 5. This is exactly the comment at lines 233-239: "the point
  at which `send` becomes safe for MPI to read."
- **D (MPI/buffer-reuse ordering) — a second, implicit function of the
  same wait, via stream-order transitivity.** Because every group's
  pack/D2H/H2D is enqueued on the *same* single `transfer_stream`, and
  streams execute their enqueued work strictly in issue order, waiting on
  *this* call's own D2H event (step 3) also transitively guarantees that
  **the previous call's H2D-out of `h_recv_` (its step 7) has already
  completed** — because that earlier H2D was enqueued on `cs` before this
  call's D2H, and CUDA stream ordering guarantees it finished first. This
  is never stated in the source comments, but it is a real, load-bearing
  consequence of the single-stream/single-buffer design: it is what makes
  it safe for *this* call's `MPI_Irecv` (step 4, immediately following the
  sync) to start overwriting `h_recv_` with newly arriving network data
  without racing the GPU DMA engine that is still allowed to be draining
  the *previous* group's `h_recv_`→`recv` H2D copy.

Not A (already satisfied — the D2H is already stream-ordered behind the
pack, no additional device-side dependency to add), not C as a *free*
change (see Stage 1), not E (this is not merely conservative slack — it
does required work), not F.

## Stage 1 — Option evaluation

- **Option A (stream-order the D2H so the host only waits when MPI needs
  it):** Already true. The D2H is issued on `cs`, stream-ordered after the
  pack. Nothing to change.
- **Option B (`cudaStreamWaitEvent` between streams instead of
  `cudaEventSynchronize`):** Not applicable. The consumer that needs the
  D2H result is `MPI_Isend`, a **host** call reading a **host** pointer —
  not another CUDA stream operation. `cudaStreamWaitEvent` only orders
  GPU-side stream work against an event; it cannot make a plain host MPI
  call wait on GPU completion. Making MPI-readiness depend on a CUDA event
  without a host wait would require a callback (`cudaLaunchHostFunc`) or a
  polling thread — a redesign, explicitly out of scope.
- **Option C (delay the host wait until immediately before MPI needs the
  payload):** This is the one option that looked promising and was
  evaluated in the most depth. `MPI_Irecv` (step 4) does not touch
  `h_send_` at all — only `MPI_Isend` (step 5) does. The apparent
  opportunity: post all four `MPI_Irecv` calls *before* `cudaEventSynchronize`,
  overlapping host-side receive-posting with the in-flight D2H, and
  sync only immediately before `MPI_Isend`.

  **This is unsafe as a bare reorder.** Moving `MPI_Irecv` ahead of the
  sync removes the transitive guarantee identified in Stage 0 (reason D):
  nothing would then stop a fast-arriving network message from writing
  into `h_recv_` while the *previous* call's `cudaMemcpyAsync(recv,
  h_recv_, H2D, cs)` (step 7 of the prior group) is still draining on the
  GPU DMA engine — a genuine host-buffer read/write race on `h_recv_`
  between the network progress engine and the CUDA DMA engine. It would
  very likely go unobserved in most runs (peer ranks proceed in lockstep,
  so the matching `Isend` usually isn't posted until the peer has done its
  own D2H+sync, by which time the prior H2D has almost certainly drained)
  — but "usually fine under current lockstep timing" is not a correctness
  argument for a production dycore, and the task's explicit rule is to
  never trade correctness for performance on exactly this kind of
  probabilistic argument.

  Making the reorder safe would require adding a **second** event,
  recorded after each group's H2D (step 7) and synchronized before the
  *next* group's `MPI_Irecv` — i.e., replacing one host wait with another
  host wait at a different point, not removing one. That is a net-neutral-
  or-negative change (added state: a second persistent event; no proven
  latency reduction, since the new wait would still gate on the same
  underlying GPU completion the current design already gates on for
  free) — it does not meet Stage 1's bar of "minimal change... not a
  redesign... overlap that naturally falls out of removing unnecessary
  sync."
- **Option D (reuse an existing later completion mechanism instead of
  adding a new one):** There is only one event (`ev_d2h_`) and one
  transfer stream in this transport; there is no other existing
  completion signal to repurpose.

## Stage 2 — Proposed change

**None.** No Option in A-D yields a change that is simultaneously (a)
minimal, (b) not a redesign, and (c) provably correctness-preserving. The
one option with real theoretical upside (C) requires adding synchronization
elsewhere to stay correct, which cancels its own benefit.

- Host-blocking calls removed per halo exchange: **0**
- Host-blocking calls remaining per halo exchange: **1** (unchanged, ~150
  calls/`DYN_CORE` call, matching Prompt 49)
- Correctness invariant preserved: the single `cudaEventSynchronize`
  continues to (i) guarantee `h_send_`/`send` is ready before
  `MPI_Isend`/`MPI_Irecv`(cuda_aware) read it, and (ii) transitively
  guarantee the previous group's `h_recv_`→`recv` H2D has drained before
  this group's `MPI_Irecv` reuses `h_recv_`.
- Theoretical ceiling if this bucket were fully eliminable (it is not, per
  above): Prompt 49's 14.4%-of-window attribution implies a maximum
  whole-`DYN_CORE`-call speedup of ≈1/(1-0.144)≈1.17x — never achievable
  here since nothing was removed.

## Stages 3-8: not executed

No source change (Stage 3), so no correctness tests (4), no FV3 smoke (5),
no C512 A/B (6), no metrics (7), no confirmation `nsys` run (8) were
needed or run. Zero GPU allocations used (0 of the 2-preferred/3-max
budget). Zero `ncu` use. No Opus escalation — the CUDA/MPI ordering
semantics were not ambiguous; they were fully resolved by reading the
existing source and reasoning through the single-stream/single-buffer
reuse contract, which is exactly the kind of source-grounded reasoning the
task asks for before writing any code.

## Decision

**Equivalent to Case C's non-beneficial branch, reached before
implementation rather than after:** "event sync visible in profile but not
reducible without a redesign that would just relocate the same wait." No
code was changed, so there is nothing to revert and nothing to commit.
`docs/agent_handoff/CURRENT_STATE.md` is not touched (no optimization was
retained).

**Recommended next step, unchanged from Prompt 49**: decompose the 77.7%
untraced host-side residual (Prompt 49 §9/§11, Rank 1) via an `nsys
--sample=cpu` (or MPI-level trace) capture — that bucket is 5.4x larger
than this one and its internal cause is not yet known, whereas this
bucket's cause is now fully known and shown not to be safely reducible at
the single-exchange level.

---

## Final Report

- **Model**: Sonnet. No Opus escalation (ordering semantics were resolved
  by source inspection, not left ambiguous).
- **Original Synchronization**: `cudaEventSynchronize(ev_d2h_)`,
  `fv3_gpu_halo_mpi.cpp:243`, host_staged `MpiHaloTransport::transfer()`.
  Reason: guarantees `h_send_` D2H completion before `MPI_Isend`/
  `MPI_Irecv` read it (documented, reason B), and — not documented in
  source, established here — transitively guarantees the *previous*
  group's `h_recv_`→device H2D has drained before this group's
  `MPI_Irecv` reuses `h_recv_` (reason D, emergent from the single-stream/
  single-buffer design). ~150 calls/`DYN_CORE` call; Prompt 49 measured
  2.85-3.20 s / 14.4% of the `DYN_CORE` window across 4 checked ranks.
- **Optimized Synchronization**: **Not changed.** The only candidate
  reorder (posting `MPI_Irecv` before the sync) was found unsafe — it
  would race the GPU DMA engine's still-in-flight drain of the previous
  group's `h_recv_`. No production source file was changed.
- **Correctness**: N/A — no change made. Existing behavior, already
  numerically validated (Gate C2, C512 bitwise/round-off), is untouched.
- **Performance**: N/A — no change made, no GPU allocation used (0
  jobs).
- **Nsys Confirmation**: Not run (no improvement to confirm; N/A per
  Stage 8's own "only if ≥3% improvement" gate).
- **Decision**: **REVERT-equivalent (nothing implemented to revert)** —
  reasoning: the sync at `fv3_gpu_halo_mpi.cpp:243` does two jobs at once
  (send-buffer readiness for the current exchange, and recv-buffer reuse
  safety for the previous exchange, the latter only via stream-order
  transitivity of the single shared buffer/stream/event design) and no
  minimal reorder among the task's own Option A-D menu removes it without
  adding an equivalent-cost wait elsewhere, which is a net-zero-or-negative
  trade, not a genuine optimization.
- **2x Roadmap Status**: baseline/current CUDA 6373.86 ms/call (unchanged
  — no run performed), 2x target ≤3187 ms/call, remaining gap unchanged.
  Next major target, unchanged from Prompt 49: CPU-side decomposition of
  the 77.7% untraced host residual (Prompt 49 §9/§11), which is both
  larger (5.4x) and — unlike this bucket — not yet root-caused.
- **Project Status**: numerical source changed: NO. Held-Suarez physics
  changed: NO. MPI topology changed: NO. CUDA-aware MPI changed: NO. Jobs
  submitted: 0. `ncu` used: NO. Opus escalation: NO.

HALO SYNC OPTIMIZATION REVERTED — READY TO PROFILE HOST-SIDE RESIDUAL
