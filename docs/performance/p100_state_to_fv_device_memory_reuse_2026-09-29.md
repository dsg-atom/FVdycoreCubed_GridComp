# Prompt 100 — `State_To_FV` C512 OOM: Device Memory Accounting and Buffer-Reuse Study

Date: 2026-09-29
Branch: `optimization/cuda-fv3-performance`
Scope: bounded device-memory architecture investigation only. No numerical
changes, no MPI changes, no production-default changes, no R4, no
resolution/rank-count changes, no commit/push. Static source analysis and
arithmetic accounting only — **no GPU jobs were run in this task** (see §14).

## 0. Provenance

Read: `CLAUDE.md`, `docs/agent_instructions/discover_platform.md`,
`docs/agent_instructions/project_architecture.md` (no separate
`cost_efficiency.md` file exists — its content is inlined in `CLAUDE.md`'s
"Cost-Efficient Agent Operation" section), `docs/agent_handoff/
CURRENT_STATE.md`, `docs/performance/p99_state_to_fv_cuda_implementation_
2026-09-28.md`, `docs/performance/p98_state_to_fv_cuda_port_plan_
2026-09-28.md`. `git status --short` matches CURRENT_STATE.md §10's
description (uncommitted P90-99 work; nothing reset/cleaned).

## 1. Stage 1 — Enumerate `State_To_FV` device allocations

Source: `cuda/dyn_core/include/fv3_gpu_context.hpp:170-185` (declarations),
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:2833-3100` (grow-on-demand
allocation + H2D/D2H), `cuda/dyn_core/vertical/fv3_state_to_fv.cuh`
(`StateInputView`/`PsPackView` accessors).

Exactly **8** new persistent, grow-only, context-owned (i.e. per-rank —
one `fv3_gpu_context` per MPI rank) device buffers were added, none of
which existed before Prompt 99:

| Buffer | Field | Elements (C512/L10, is:ie×js:je, no halo) | Bytes/rank (R8) | Aggregate, 6 ranks | Lifetime | Owner |
|---|---|---|---|---|---|---|
| `state_to_fv_u_buffer` | STATE%VARS%U | 512×512×10 = 2,621,440 | 20,971,520 (20.97 MB) | 125.83 MB | Phase A only (H2D→kernel read, dead immediately after) | Prompt 99 |
| `state_to_fv_v_buffer` | STATE%VARS%V | 512×512×10 | 20,971,520 | 125.83 MB | Phase A only | Prompt 99 |
| `state_to_fv_w_buffer` | STATE%VARS%W | 512×512×10 | 20,971,520 | 125.83 MB | Phase C only | Prompt 99 |
| `state_to_fv_pt_buffer` | STATE%VARS%PT | 512×512×10 | 20,971,520 | 125.83 MB | Phase C only | Prompt 99 |
| `state_to_fv_pkz_buffer` | STATE%VARS%PKZ | 512×512×10 | 20,971,520 | 125.83 MB | Phase C only | Prompt 99 |
| `state_to_fv_dz_buffer` | STATE%VARS%DZ | 512×512×10 | 20,971,520 | 125.83 MB | Phase C only | Prompt 99 |
| `state_to_fv_pe_buffer` | STATE%VARS%PE | 512×512×11 = 2,883,584 | 23,068,672 (23.07 MB) | 138.41 MB | Phase C only | Prompt 99 |
| `state_to_fv_ps_buffer` | packed `ps` scratch | 512×512 = 262,144 | 2,097,152 (2.10 MB) | 12.58 MB | Phase C only (D2H'd immediately, then dead) | Prompt 99 |
| **Total** | | | **150.99 MB/rank** | **905.94 MB** | | |

**Accounting reconciles: YES.** 150.99 MB/rank × 6 ranks = 905.94 MB ≈
0.9 GB, matching Prompt 99 §7's own "≈151MB × 6 ≈ 0.9GB" estimate exactly
(computed independently here from the actual `cudaMalloc` size expressions
at `fv3_gpu_runtime.cpp:2859`, `2990-2992`, not re-derived from the prior
document). All 8 buffers are accounted for; no missing allocation.

Sizing formula (verified against source, `fv3_gpu_runtime.cpp:2977-2992`):
`cols=(ie-is+1)*(je-js+1)`; u/v/w/pt/pkz/dz = `cols*km`; pe = `cols*(km+1)`;
ps = `cols`. All 8 are **grow-only** (freed only at `fv3_gpu_finalize`,
alongside `remap_q_buffer`/`remap_pressurevars_*_buffer` — confirmed at
`fv3_gpu_runtime.cpp`'s finalize block).

## 2. Stage 2 — Existing device buffers for the same physical fields

| Field | `State_To_FV` new buffer | Existing FieldRegistry buffer | Same layout? | Same lifetime? |
|---|---|---|---|---|
| U | `state_to_fv_u_buffer` (is:ie×js:je, no halo, plain contiguous) | `FieldId::u` (isd:ied×jsd:jed+1, haloed, `RiemViewC3`) | **NO** — different extent/stride, different lower bound | **NO** — `u`/`v` FieldRegistry buffers persist across the whole run (fed by every WP-A/B/C/D stage); the new buffer is dead within Phase A |
| V | `state_to_fv_v_buffer` | `FieldId::v` | NO | NO |
| W | `state_to_fv_w_buffer` | `FieldId::w` | NO (no halo vs haloed) | NO |
| PE | `state_to_fv_pe_buffer` | `FieldId::pe` (i,k,j) | NO — input is (i,j,k) pre-transpose, output is (i,k,j) post-transpose; genuinely different data, not an aliasable duplicate | NO |
| PELN | *(none — no new buffer)* | `FieldId::peln` | N/A | N/A |
| PK | *(none — no new buffer)* | `FieldId::pk` | N/A | N/A |
| PS | `state_to_fv_ps_buffer` | none exists — `ps` has no FieldRegistry identity by design (§0 of `fv3_state_to_fv.cuh`) | N/A | Phase C only, D2H'd then dead |
| DELP | *(none — no new buffer)* | `FieldId::delp` | N/A | N/A |
| PT | `state_to_fv_pt_buffer` | `FieldId::pt` | NO — input is raw `STATE.PT`, output is `STATE.PT*STATE.PKZ` (a computed field); not the same data | NO |
| DELZ | `state_to_fv_dz_buffer` | `FieldId::delz` | NO — input is `STATE.DZ`, copied as-is, but into a haloed buffer with tiny-number-filled halo cells; not identical bytes | NO |
| PKZ | `state_to_fv_pkz_buffer` | `FieldId::pkz` | NO — input value equals output value for interior cells, but shape/halo differ | NO |

**Finding**: none of the 8 new buffers duplicate an existing FieldRegistry
buffer's *storage* — they hold genuinely different data (plain contiguous,
no-halo, pre-transform host-side values) that the kernel reads to *produce*
the FieldRegistry buffer's contents. This matches Prompt 99 §11's own
claim ("only fields with no existing FieldRegistry identity get new
buffers") — confirmed here by direct layout/shape comparison rather than
taken on faith. **No Category B (duplicate storage) buffers exist.**

## 3. Stage 3 — Lifetime timeline (one `k_split` iteration)

```
Phase A: H2D state_u/state_v  -> kernel (fills u/v FieldRegistry) -> DEAD
         -> D2H u/v (existing fv3_gpu_sync_to_host mask)
         -> host mpp_get_boundary (Phase B, unchanged)
         -> H2D u/v halo back
Phase C: H2D state_w/state_pe/state_pt/state_pkz/state_dz
         -> kernel (writes w/pe/peln/pk/ps_out/delp/pt/delz/pkz FieldRegistry
            + ps_out buffers) -> state_w/pe/pt/pkz/dz DEAD; ps_out D2H'd then DEAD
fv_dynamics() [CUDA path]:
         -> PT/DELZ remap, REMAP_UV, TRACER_2D (Fortran), REMAP_Q,
            REMAP_PRESSUREVARS, vertical Riemann solver (WorkspaceId::vertical)
         -> all of these read/write FieldRegistry buffers and the
            cgrid/transport/vertical/operators WorkspaceId arenas; NONE of
            them reference any state_to_fv_*_buffer (grep-confirmed: no
            symbol `state_to_fv_` appears outside fv3_gpu_runtime.cpp's
            phase_a/phase_c functions and fv3_gpu_context.hpp/
            fv3_gpu_internal.hpp's declarations)
FV_To_State (host-only, per Prompt 97) -> DYN_EPILOGUE
[next k_split iteration] -> Phase A again
```

All 8 `state_to_fv_*` buffers are **DEAD** for the entire remainder of the
iteration (all of `fv_dynamics()`, `FV_To_State`, `DYN_EPILOGUE`) after
their one H2D/kernel-launch/(D2H for `ps`) use. Every numerical dispatch
call in this runtime — including `state_to_fv_phase_a/_c` themselves and
`launch_vertical_solver` (Riemann solver, `WorkspaceId::vertical`) — is
submitted through the same single per-context `Submission{...,
c->data.compute_stream, ...}` (confirmed at `fv3_gpu_runtime.cpp:2886-2887`
for `state_to_fv_phase_a`, and by the identical `ExecutionContext`/
`Submission` construction pattern used by every other production dispatch
call in this file, e.g. `fv3_gpu_remap_pressurevars`,
`fv3_gpu_dyn_core_step`). This single-stream, strictly-in-program-order
execution model is exactly the same one the project already relies on for
`WorkspaceId::transport`'s documented "shared, sequentially, by every WP-B
service" reuse (`fv3_gpu_runtime.cpp:1498-1510`) — i.e., this project
already treats same-stream sequential non-overlap as sufficient grounds
for safe buffer reuse; this is existing convention, not a new invention.

**Non-overlap holds cleanly: `state_to_fv_*` buffers and
`WorkspaceId::vertical`'s per-column Riemann-solver scratch are never live
at the same time within one iteration**, and the ordering (state_to_fv
Phase A/C strictly before `fv_dynamics`'s Riemann solver call, every
iteration, same stream) is enforced by simple program order, not a race
that depends on scheduling.

## 4. Stage 4 — Duplication classification

| Buffer | Classification | Evidence |
|---|---|---|
| `state_to_fv_u_buffer` | **C — can alias existing storage** | No existing buffer holds this data (Stage 2); lifetime is disjoint from `WorkspaceId::vertical`'s Riemann-solver scratch (Stage 3) |
| `state_to_fv_v_buffer` | C | same |
| `state_to_fv_w_buffer` | C | same |
| `state_to_fv_pe_buffer` | C | same |
| `state_to_fv_pt_buffer` | C | same |
| `state_to_fv_pkz_buffer` | C | same |
| `state_to_fv_dz_buffer` | C | same |
| `state_to_fv_ps_buffer` | C | same (dead even sooner — D2H'd within Phase C itself) |

No buffer classifies A (required unique, cannot alias — none, since Stage 3
found a clean non-overlapping window in `WorkspaceId::vertical`), B
(duplicate existing field storage — none, per Stage 2), D (workspace reuse
after lifetime end — folds into C here since the target is the persistent
`vertical` workspace, not a transient one), or E (eliminable outright — the
data itself is genuinely needed; only its *storage location* is
redundant).

## 5. Stage 5 — U/V halo special case

`state_to_fv_u_buffer`/`state_to_fv_v_buffer` are already the **minimum**
possible representation: plain `is:ie × js:je`, **no halo at all**
(confirmed at `fv3_state_to_fv.cuh:86-96` and by the sizing formula in
Stage 1 — `cols*km`, not `(ni_halo)*(nj_halo)*km`). They are *not* full
duplicates of the haloed `u`/`v` FieldRegistry buffers; they exist only
because `STATE%VARS%U/V` (MAPL State fields) are a physically distinct,
smaller-domain array from `FV_Atm%u/v` (FieldRegistry fields), and the H2D
copy needs *some* device destination before the fill/copy kernel can run.

**Answer: NO**, the implementation does not unnecessarily duplicate full
U/V fields — it already uses the smallest possible footprint for this
purpose. No savings available here beyond the general aliasing opportunity
in §3/§4.

## 6. Stage 6 — Pressure/thermo field direct-write check

| Field | Direct write to existing storage possible? | Notes |
|---|---|---|
| PE | **Already true for the output** — Phase C writes directly into `FieldId::pe`'s `RiemViewC3`, device-resident, feeding `fv_dynamics` with no extra transfer (Prompt 99 §0). The *input* (`state_to_fv_pe_buffer`) is unavoidably separate: it holds `STATE%VARS%PE` in `(i,j,k)` order before the transpose to `(i,k,j)` — an in-place transpose into the same buffer is possible only if the buffer is large enough and the transpose kernel is written to be safe for in-place aliasing (not attempted — see §10 Option 3, not selected). |
| PELN/PK | Already direct — derived in-kernel from `pe`, written straight into `FieldId::peln`/`FieldId::pk`. No new buffer exists for these; nothing to fix. |
| PS | No existing FieldRegistry buffer exists for `ps` by design (Stage 2) — it is a pure host/Fortran-consumed quantity with no on-device consumer, so there is no "existing storage" to write into. The one alternative already considered and rejected by Prompt 99's design (per `fv3_state_to_fv.cuh:98-106`) — giving `ps` a full FieldRegistry entry — would *increase* memory, not reduce it, and was correctly avoided. |
| DELP/PT/DELZ/PKZ | Same as PE: outputs already written directly into `FieldId::delp/pt/delz/pkz` (device-resident). Only the *inputs* (`state_to_fv_pt_buffer`, `state_to_fv_dz_buffer`, `state_to_fv_pkz_buffer`) are separate, for the same reason as PE — different source data (pre-copy `STATE%VARS%*`), not an aliasable duplicate of the destination. |

**Finding**: the *output* side of this port (Stage 6's actual question)
already has **zero** duplicate storage — this was correctly designed in
Prompt 99. The only recoverable memory is on the **input staging** side
(the 8 buffers themselves), via the temporal-aliasing opportunity in
§3/§4/§7, not via any further direct-write change to the outputs.

## 7. Stage 7 — Workspace reuse

`WorkspaceId::vertical` (`fv3_gpu_runtime.cpp:1512-1531`) is sized exactly
as `cells × vertical_workspace_per_column(km)`, where `cells =
(ie-is+1+2·halo_width)×(je-js+1+2·halo_width)` (halo-inclusive) and
`vertical_workspace_per_column` = 422 scalars/column at `km=10`
(`fv3_vertical_column.hpp:856-858`, `compute_workspace_layout`). At
C512/L10 (`halo_width=3`, `fv3_gpu_halo_topology.hpp:54`):

```
cells = (512+6) × (512+6) = 518 × 518 = 268,324
vertical_bytes/rank = 268,324 × 422 × 8B = 905,973,824 B ≈ 906 MB/rank
```

This single existing per-rank arena (≈906 MB/rank) is **already ~6×
larger** than the entire 8-buffer `state_to_fv` footprint it would need to
host (150.99 MB/rank, Stage 1) — with no growth required. Per Stage 3, its
lifetime (Riemann-solver scratch, consumed entirely within
`fv_dynamics()`'s own vertical-solver kernel calls) never overlaps
`state_to_fv`'s window. Aliasing `state_to_fv`'s 8 buffers into a
sub-region of this existing allocation (rather than issuing 8 new
`cudaMalloc` calls) would recover the **entire** 150.99 MB/rank / 905.94 MB
aggregate marginal addition, using memory that is already resident and
already paid for — this is the single highest-value, lowest-risk reuse
candidate identified in this study (see §10 Option 1). No workspace
capacity increase would be needed (confirming the task's "do not increase
workspace capacity unless necessary" constraint is satisfied: none is
needed).

`WorkspaceId::transport`/`cgrid`/`operators` were not pursued further once
`vertical` was confirmed sufficient by itself (bounded-debugging stop
condition: one candidate that fully covers the need is enough).

## 8. Stage 8 — Transient allocation option

Per-call `cudaMalloc`/`cudaFree` (naive transient allocation) is explicitly
disallowed by the task and would also reintroduce the exact
allocator-overhead problem Prompt 68's "no local-array-per-call" convention
already exists to avoid. **Classification: NOT VIABLE** as a standalone
approach.

A **lifetime-scoped alias into the existing persistent `WorkspaceId::
vertical` arena** (§7) is the correct instantiation of "transient
allocation" here — it does not allocate anything new; it borrows a
sub-range of an already-allocated, already-persistent arena for the
in-order-guaranteed-disjoint window described in §3. **Classification:
VIABLE**, using only existing project infrastructure (the `WorkspaceLayout`
slice mechanism / `Submission`-ordered single-stream execution), not a new
general-purpose memory manager.

## 9. Stage 9 — Required memory target

- Current aggregate free headroom on the shared A100 (6 ranks/1 GPU,
  MPS, **without** `state_to_fv` enabled, i.e. the Prompt 99 §7 baseline
  telemetry): **≈69–73 MiB** total (40265/40334 MiB used out of 40960 MiB
  total, `nvidia-smi`, job 58628363).
- Prompt 99's new allocation: **≈906 MB aggregate** (Stage 1).
- Required target (per task): **≥512 MiB free**, prefer **≥1 GiB free**,
  *after* `state_to_fv` is enabled.

**Decisive arithmetic**: the baseline (i.e., *without* `state_to_fv` at
all, with every other already-shipped production CUDA default active) is
already **≈440–443 MiB short of the 512 MiB target** on its own (69–73 MiB
free vs. 512 MiB required). A perfect `state_to_fv`-scoped fix — one that
reduces `state_to_fv`'s own marginal footprint to exactly **zero** new
bytes via the §7 aliasing scheme — returns the system to that same ≈69–73
MiB baseline headroom, **not** to ≥512 MiB. Recovering `state_to_fv`'s
entire 906 MB addition is *necessary* to avoid the OOM crash but is
**mathematically insufficient by itself** to reach the required safety
margin, because the deficit predates this port and is not attributable to
it.

## 10. Stage 10 — Narrow fix candidates

| Option | Memory saved | Code scope | Numerical risk | Architecture risk |
|---|---|---|---|---|
| **1. Alias all 8 `state_to_fv_*` input buffers into a sub-region of the existing `WorkspaceId::vertical` arena** (offsets carved out of the same `cudaMalloc`'d allocation, guarded by a static/runtime check that 150.99 MB/rank ≤ the vertical slice's actual allocated bytes) | Recovers the full ~906 MB aggregate marginal addition (→ net-zero *new* device memory for this port) | Localized to `fv3_gpu_context.hpp` (replace 8 owned pointers with computed offsets into `workspace.allocation`) + `fv3_gpu_runtime.cpp`'s phase_a/phase_c grow-logic (replace `cudaMalloc` calls with an offset/bounds check) | Low — same data, same kernels, same stream-ordering guarantee the project already relies on for `WorkspaceId::transport` (§3); requires one careful runtime assertion that the vertical slice is large enough (true today by a ~6× margin, §7) so it fails loudly rather than corrupting memory if that ratio ever changes (e.g., a future `km`/domain change that shrinks `vertical_workspace_per_column` more than `state_to_fv`'s own need) | Low — no new abstraction, uses the existing `WorkspaceLayout`/single-compute-stream convention verbatim |
| 2. Give `state_to_fv`'s PE/PT/PKZ/DZ/W inputs an in-place-transpose/overwrite path directly into the corresponding FieldRegistry buffer's memory (skip the separate input buffer entirely) | Would recover the same ~906 MB, more surgically | Higher — requires rewriting each kernel to safely alias its own output buffer as scratch during a per-thread-column sequential transform (PE's transpose in particular touches multiple `(i,k,j)` destinations from one `(i,j,k)` source per column — an in-place aliasing hazard needs a careful per-thread ordering proof, not just a stream-ordering one) | Higher — touches kernel internals, not just buffer provenance; not attempted in this task |
| 3. Reduce `WorkspaceId::vertical`'s own footprint (it is ~6× larger than `state_to_fv`'s need) so both fit with capacity to spare | Could in principle free up genuinely new headroom beyond `state_to_fv`'s own addition | Out of scope — `vertical` is owned by the pre-existing Riemann-solver ports (WP-C), not `state_to_fv`; the task's DO-NOT list forbids touching REMAP_PRESSUREVARS/REMAP_Q/etc. and this workspace is shared with those | Not evaluated; flagged only as a pointer for the broader study (§ "Project-phase decision") |

Option 1 is selected in principle (§11) as the correct **necessary** fix
for `state_to_fv`'s own marginal addition. Per Stage 9's arithmetic,
however, no `state_to_fv`-scoped option (1, 2, or any combination) can by
itself close the pre-existing ≈440 MiB baseline deficit, so none reaches
the task's required ≥512 MiB target.

## 11. Stage 11 — Selection

Per the task's explicit gate: *"Choose exactly ONE implementation
candidate only if it is ... expected to recover enough memory ... Target:
projected free headroom >=512 MiB/node. If no single narrow fix can reach
that: STOP."*

Option 1 (§10) is narrow, numerically transparent, lifetime-safe (§3), and
would recover the entire `state_to_fv`-attributable addition (§7) — but
per §9's arithmetic, even a **perfect** realization of Option 1 projects
free headroom of only **≈69–73 MiB** (the pre-existing baseline), not
≥512 MiB. **No single `state_to_fv`-scoped narrow fix — including the
best one identified — can reach the required target**, because the
shortfall is dominated by the baseline's own near-zero headroom, which
predates and is independent of this port.

**STOP. Classifying the blocker as: BROADER DEVICE-RESIDENCY/MEMORY
ARCHITECTURE REQUIRED** (per the task's own Stage 11 instruction for this
exact outcome).

## 12. Stage 12 — Implementation

**Not performed.** Per Stage 11's explicit gate ("Implement ONLY if Stage
11 finds one narrow, well-proven reuse fix" that reaches the ≥512 MiB
target) and per the task's own instruction ("If no narrow fix: make no
production source changes"), no production source was modified in this
task. `git status --short` is unchanged by this task (only the
documentation files listed at the end of this report were added/edited).

Option 1 (§10/§11) is recorded as a validated, ready-to-implement,
necessary-but-insufficient candidate for a future task that also recovers
headroom elsewhere (see "Project-phase decision" below) — implementing it
in isolation now would touch production CUDA source for a change that,
by this study's own arithmetic, cannot unblock C512 on its own, which
would violate the task's cost-efficiency convention (do not spend
implementation effort past the point a hypothesis is resolved).

## 13/14/15 — Revalidation / C512 feasibility / performance resume

**Not run.** Stages 13-15 are explicitly gated on Stage 12 producing a
production source change ("Only if a memory fix was implemented" / "Only
if Stage 13 passes"). Since no change was made (§12), no fixture/C32/C512
job was submitted in this task. No GPU jobs were run in Prompt 100.

## Final report

Baseline production peak: **40265 MiB / 40334 MiB used, 40960 MiB total**
(6 ranks/1 A100, MPS, every other production CUDA default active, telemetry
from job 58628363, Prompt 99 §7 — not re-measured in this task, cited as
the authoritative existing measurement).

Baseline free: **≈69–73 MiB**

Prompt-99 added allocation: **≈905.94 MB aggregate (150.99 MB/rank × 6)**

--------------------------------------------------
ALLOCATION ACCOUNTING
--------------------------------------------------

| Buffer | MiB/rank | Aggregate (6 ranks) | Classification |
|---|---|---|---|
| `state_to_fv_u_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_v_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_w_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_pt_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_pkz_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_dz_buffer` | 20.00 | 120.00 MiB | C |
| `state_to_fv_pe_buffer` | 22.00 | 132.01 MiB | C |
| `state_to_fv_ps_buffer` | 2.00 | 12.00 MiB | C |
| **Total** | **144.00 MiB** | **864.01 MiB** | |

(MiB figures; §1's MB — decimal — figures reconcile to the same 8 buffers,
905.94 MB ≈ 864.01 MiB — same bytes, different unit convention, both
shown since Prompt 99's own doc used decimal MB.)

Accounting reconciles: **YES**

--------------------------------------------------
REUSE OPPORTUNITIES
--------------------------------------------------

Duplicate storage: **NONE FOUND** — Stage 2/6 confirmed all 8 buffers hold
genuinely distinct data from every existing FieldRegistry buffer; the
output side of the port already writes directly into existing
device-resident storage with zero duplication.

Safe alias/reuse: **All 8 `state_to_fv_*` buffers, into the existing,
already-oversized, lifetime-disjoint `WorkspaceId::vertical` arena**
(§3/§4/§7 — Option 1, §10).

Potential savings: **≈864 MiB/node (≈906 MB), i.e. the entire Prompt-99
marginal addition** — but see Stage 9/11: this returns the system only to
its pre-existing ≈69–73 MiB baseline headroom, not to the required ≥512
MiB.

--------------------------------------------------
SELECTED FIX
--------------------------------------------------

**NONE IMPLEMENTED** (Stage 11 STOP condition reached — no
`state_to_fv`-scoped fix, including the best one identified, projects
≥512 MiB free headroom; see §9/§11 for the arithmetic).

Projected savings (if Option 1 were implemented in a future task, alongside
independent headroom recovery elsewhere): ≈864 MiB/node for `state_to_fv`
specifically.

Numerical semantics changed: NO (no change was made)

MPI changed: NO (no change was made)

--------------------------------------------------
VALIDATION
--------------------------------------------------

Fixtures: NOT RUN (no fix implemented; gated per §13)

C32: NOT RUN

C512: NOT RUN

Peak memory after fix: N/A (no fix implemented)

Peak free headroom: N/A

Safe production-density headroom: **NO**

--------------------------------------------------
PROMPT 99 STATUS
--------------------------------------------------

Performance validation: **STILL BLOCKED**

--------------------------------------------------
PROJECT PHASE
--------------------------------------------------

Standalone CUDA porting still viable: **NO** — not at the current
6-ranks-per-1-A100 production density. The decisive finding of this task
is that the density's own pre-existing headroom (≈69–73 MiB, established
by every already-shipped production CUDA default combined — PT/DELZ,
REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS — none audited individually in this
task) is already below the ≥512 MiB safety margin that any additional
per-rank persistent buffer allocation would need. This is not specific to
`state_to_fv`: **any** future CUDA port that adds persistent per-rank
device memory at this density will hit the same wall, regardless of how
efficiently that port itself manages its own buffers.

Next: **DEVICE-RESIDENCY MEMORY-ARCHITECTURE STUDY** — specifically, the
per-port memory audit already recommended by Prompt 99 §11/CURRENT_STATE.md
§9 (quantify exactly how much of the 40 GB budget each of PT/DELZ,
REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS, and the `cgrid`/`transport`/
`vertical`/`operators` `WorkspaceLayout` arenas already consumes per rank),
plus evaluating whether reducing ranks-per-GPU (2 GPUs/3-ranks-per-GPU,
already validated correct for CUDA-aware-MPI transport per CURRENT_STATE.md
§5) is a viable precondition for reviving `state_to_fv` — not another
isolated single-kernel CUDA port.

**STATE_TO_FV STILL MEMORY-BLOCKED — BROADER DEVICE-RESIDENCY ARCHITECTURE REQUIRED**

--------------------------------------------------
UPDATE (Prompt 101, 2026-09-29)
--------------------------------------------------

The per-port memory audit recommended above was executed as Prompt 101
(`docs/performance/p101_production_gpu_memory_inventory_2026-09-29.md`).
Result: static per-port/arena accounting (PT/DELZ and REMAP_UV confirmed
zero dedicated buffers; REMAP_Q/REMAP_PRESSUREVARS ≈82 MiB/rank combined;
`WorkspaceLayout` ≈2170.4 MiB/rank; FieldRegistry ≈870 MiB/rank) explains
only ≈46.6% of the live-measured converged C512 footprint — a ≈21.5 GiB
aggregate gap remains unattributed, traced to the first production
dispatch call but not resolved among 4 candidate causes. Per Prompt 101's
own reconciliation gate (<90%), **no memory fix was selected** — this
task's own recommendation (a broader architecture study) is therefore
still open, and is now itself gated on a follow-up barrier-synchronized
diagnostic (Prompt 101 §16) before any memory-recovery fix, including a
revival of this task's Option 1 alias, should be attempted.
