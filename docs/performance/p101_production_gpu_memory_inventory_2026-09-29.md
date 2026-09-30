# Prompt 101 — Production GPU Memory Inventory and Recoverable-Headroom Audit

Date: 2026-09-29
Branch: `optimization/cuda-fv3-performance`
Scope: **MEMORY INVENTORY ONLY.** No new CUDA kernels, no numerical changes,
no MPI-semantics changes, no production-default changes, no disabling of
validated CUDA ports, no `State_To_FV` modification, no multi-GPU redesign
(deferred to Prompt 102), no new memory manager, no R4, no commit/push.
One short (30-DYN_CORE-call, ~2m32s) C512 diagnostic job was run under an
explicit, reverted, TEST-ONLY instrumentation flag (§6); no other GPU jobs.

## 0. Provenance

Read: `CLAUDE.md`, `docs/agent_instructions/discover_platform.md`,
`docs/agent_instructions/project_architecture.md`,
`docs/agent_handoff/CURRENT_STATE.md`,
`docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md`,
`docs/performance/p99_state_to_fv_cuda_implementation_2026-09-28.md`,
`docs/performance/p100_state_to_fv_device_memory_reuse_2026-09-29.md`.
`git status` matched `CURRENT_STATE.md` §10 (uncommitted P90-99 work, plus
this task's own TEST-ONLY instrumentation, added and then fully reverted —
see §6/§17). Nothing reset/cleaned/reverted beyond that instrumentation.

## 1. Stage 1 — Enumerate all device allocations (source-level, exhaustive)

An exhaustive `cudaMalloc(` grep across all of `cuda/dyn_core` (excluding
`*_smoke.cu`, `*_prod_shape_smoke.cu`, and `test_*.cu`/`test_*.cpp`, which
are self-contained unit-test executables with their own local device
pointers, not part of the production runtime) finds production allocations
in exactly one file, `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`:

| Call site (pre-audit line #) | Allocation | Persistent? | Owner |
|---|---|---|---|
| ~1253/1254 | halo send/recv buffers | Yes, context lifetime | WP-E |
| ~1494 | FieldRegistry loop (~90 fields) | Yes, context lifetime | WP-E |
| ~1531 | `WorkspaceLayout` single arena (4 slices) | Yes, context lifetime | WP-E |
| ~2002 | `heat_accum` (Held-Suarez only, HS-track, not this branch's baseline) | Yes | WP-C |
| ~2598 | `remap_q_buffer` | Yes, grow-only | REMAP_Q (P84B-89) |
| ~2723/2732 | `remap_pressurevars` qsphum+cappa | Yes, grow-only | REMAP_PRESSUREVARS (P90-93) |
| ~2865/2874, ~2999 (`grow` lambda) | `state_to_fv` u/v/w/pe/pt/pkz/dz/ps (8 buffers) | Yes, grow-only | `State_To_FV` (P99, **not active** — default backend Fortran) |

**Confirmed finding (carried over from prior analysis this session, now
final): PT/DELZ and REMAP_UV add zero dedicated persistent device
buffers.** They read/write directly through FieldRegistry-resident state
and the shared `WorkspaceLayout` arena. This is a clean, favorable
finding — no hidden duplication from either port.

## 2. Stage 2 — `WorkspaceLayout` inventory

Single `cudaMalloc` arena, built once in `fv3_gpu_initialize_state` via
`build_workspace_layout()`, split into 4 concatenated slices. At C512/L10
(halo_width=3, npz=10, R8): `ni=nj=518`, `cells=268,324`,
`value_bytes≈22.52 MiB`.

| Slice | Sizing rule | Array count | MiB/rank |
|---|---|---|---|
| `cgrid` | 1× value_bytes per array | 12 | ≈270.2 |
| `transport` | max-of-all-users (see below) | 34 | ≈765.8 |
| `vertical` | max-of-all-users | 38 | ≈864.0 |
| `operators` | 1× value_bytes per array | 12 | ≈270.2 |
| **Total** | | | **≈2170.4 MiB/rank ≈ 2.17 GiB/rank** |

`transport`'s 34-array sizing is an explicit **overprovisioning-by-design**
pattern: `d_sw` needs all 34 arrays; `update_dz_d` (a different caller of
the same arena) needs only 17. The arena is sized for the largest caller
and reused by all callers within its lifetime window — see §10.

×6 ranks ≈ **12.71 GiB aggregate** — the single largest quantified,
named consumer in the static inventory.

## 3. Stage 3 — FieldRegistry inventory

`cuda/interface/fv3_gpu_field_registry.F90`, confirmed complete this
session: exactly 100 `call add*` registration lines, last at line 858; file
ends at line 938 with no further registrations. No `FV3_FID_PS` or
`FV3_FID_Q`/tracer field is directly registered (tracers stay Fortran-side
per the P77 cold-start finding; PS is handled elsewhere).

Field categories (category-based estimate, ±15% — exact per-field
`allocation_bytes` were not individually re-derived from the `.F90` source
this session; see §17 open item):

| Category | Count (approx.) | MiB each | MiB/rank |
|---|---|---|---|
| 3D 11(interface)/10-level core+scratch fields (U,V,W,DELZ,PT,DELP,Q_CON,OMGA,UA,VA,UC,VC,DISS_EST + 17 WP-A/C-grid scratch: CAPPA,DELPC,PTC,UT,VT,DIVGD,CRX,XFX,CRY,YFX,GZ,ZH,PKC,DU,DV,HEAT_SOURCE,Z_RAT + PK3) | ~31 | ≈22.52 | ≈698 |
| 2D grid-metric fields (AREA,RAREA,DXA,DYA,...,RSINA — ~30) + PHIS/WS/WS3/ZS/DIVG2 | ~36 | ≈2.05 | ≈74 |
| PE/PELN/PK/PKZ (interior-only, no halo, different level orderings) | 4 | ≈24 | ≈96 |
| SIN_SG/COS_SG (0:4,5 levels), GRID_LONLAT/AGRID_LONLAT, EDGE_W/E/S/N, damping 1D arrays (D2_BG,D4_BG,DAMP_V,DAMP_W,DAMP_T,KGB,NDIF,NORD_K,NORD_W,D_CON), AK/BK/DP_REF | ~20 | small | ≈2 |
| **Total** | ~91 | | **≈870 MiB/rank (±15%)** |

No REAL DUPLICATE fields were found against the `WorkspaceLayout` slices
by name/purpose during this pass — the ~17 WP-A/C-grid scratch fields
(CAPPA, DELPC, PTC, UT, VT, DIVGD, CRX, XFX, CRY, YFX, GZ, ZH, PKC, DU, DV,
HEAT_SOURCE, Z_RAT) are FieldRegistry-resident (not workspace slices);
whether any of these could instead be aliased into the `cgrid`/`transport`
workspace slices as a dedup opportunity is flagged for Stage 11 but not
resolved with certainty this session (see §11, §17).

×6 ranks ≈ **5.10 GiB aggregate**.

## 4. Stage 4 — Per-port accounting

| Port | Persistent buffer | Workspace-only | Total MiB/rank | Production default |
|---|---|---|---|---|
| PT/DELZ | 0 | uses shared arena | 0 (dedicated) | CUDA |
| REMAP_UV | 0 | uses shared arena | 0 (dedicated) | CUDA |
| REMAP_Q | `remap_q_buffer`, haloed×km×nq(=2) | — | ≈40.95 | CUDA |
| REMAP_PRESSUREVARS | qsphum+cappa, haloed×km each | — | ≈40.95 | CUDA |
| `State_To_FV` | 8 buffers (u,v,w,pe,pt,pkz,dz,ps) | — | ≈144.0 (P99/P100 figure) | **Fortran (not active)** |

Sum of *active* production ports' dedicated buffers: **≈81.9 MiB/rank ≈
491 MiB aggregate** — small relative to Workspace/FieldRegistry.

## 5. Stage 5 — `dyn_core` accounting

`vertical` workspace slice (≈864.0 MiB/rank) reconciles with Prompt 100's
prior ~906 MB/rank estimate (that estimate included the 8 `state_to_fv`
buffers hypothetically aliased in; without them the base slice is
≈864.0 MiB, consistent).

**Not separately sized this session:** halo send/recv buffer bytes
(6-tile boundary exchange, sized for all boundary-crossing fields ×
halo depth). This is a known, named gap in the static accounting — see
§8/§17.

## 6. Stage 6 — Runtime telemetry (live diagnostic, executed)

Per Stage 6's explicit allowance, one short C512 diagnostic job was run to
resolve what static accounting could not (§8's reconciliation gap, found
before this job — see below). Five `cudaMemGetInfo` checkpoints were added
as **TEST-ONLY** instrumentation to
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`, gated behind `FV3_GPU_MEMDIAG=1`
(default off, zero effect otherwise): `after_cudaSetDevice`,
`after_context_streams_events`, `after_field_registry_loop`,
`after_initialize_state_complete`, `after_first_remap_pressurevars_call`.

**Job**: `p101_memdiag_c512` (job 58638073), C512, 6 MPI ranks, 1 A100,
MPS ON, production defaults with **zero overrides** (PT/DELZ=CUDA,
REMAP_UV=CUDA, REMAP_Q=CUDA, REMAP_PRESSUREVARS=CUDA, TRACER=FORTRAN,
`STATE_TO_FV`=FORTRAN — i.e. exactly the configuration
`CURRENT_STATE.md` attributes ~40265-40334 MiB/40960 MiB to). Binary
provenance verified by sha256 before run (matched exactly). Ran cleanly:
rc=0, 30 DYN_CORE calls, no FATAL/OOM/NaN/Inf. Elapsed 00:02:32.

**Result — converged total confirmed, reached early:**

| Checkpoint (max observed across 6 ranks) | used MiB | free MiB | total MiB |
|---|---|---|---|
| `after_cudaSetDevice` | 2525.9 | 37813.4 | 40339.3 |
| `after_context_streams_events` | 2525.9 | 37813.4 | 40339.3 |
| `after_field_registry_loop` | 17577.9 | 22761.4 | 40339.3 |
| `after_initialize_state_complete` | 21917.9 | 18421.4 | 40339.3 |
| `after_first_remap_pressurevars_call` | **40265.9** | 73.4 | 40339.3 |

This directly reconfirms the previously-documented baseline
(~40265-40334 MiB used, ~69-73 MiB free) — same order, same magnitude,
independently reproduced. It also establishes that the full footprint is
**front-loaded**: fully resident by the time the *first* production
dispatch call completes (well within the first DYN_CORE step), with no
further growth observed through 30 steps — consistent with the project's
persistent-device-residency architecture goal.

**Important limitation of this data (stated explicitly, not glossed
over):** the 6 MPI ranks run concurrently under MPS, unsynchronized by any
barrier at each checkpoint. `cudaMemGetInfo` reads *global* device memory
state, so each rank's printed value reflects however far *all* ranks had
collectively progressed at that instant of wall-clock time — not that
rank's own isolated contribution. The `after_field_registry_loop` /
`after_initialize_state_complete` readings are therefore **not** a clean
per-category attribution; they bound a monotonically-advancing high-water
mark, confounded by cross-rank scheduling jitter under MPS multiplexing.
Only the endpoints — near-zero at job start (not captured) and the
converged ≈40.3 GiB ceiling by first dispatch — are unconfounded.

Instrumentation was fully reverted immediately after this job (§17); the
CUDA library and `StandAlone_FV3_Dycore.x` were rebuilt/relinked back to
byte-identical Prompt-99 state (sha256 `2339366b...186e2064d` — verified
matching Prompt 99's documented hash exactly).

## 7. Stage 7 — Rank-by-rank symmetry

Not independently verified with clean per-rank isolation this session
(blocked by the same MPS-concurrency confound as §6). The converged
ceiling (§6's last row) is consistent within 73-249 MiB free across all 6
ranks, i.e. **symmetric at the converged endpoint** — no rank is a
measurable outlier once the full pipeline has executed once.

## 8. Stage 8 — Reconciliation against the ≥90% gate: **NOT MET**

| Category | MiB/rank | Aggregate (×6) |
|---|---|---|
| FieldRegistry (§3) | ≈870 | ≈5.10 GiB |
| WorkspaceLayout, 4 arenas (§2) | ≈2170.4 | ≈12.71 GiB |
| REMAP_Q buffer (§4) | ≈40.95 | ≈0.24 GiB |
| REMAP_PRESSUREVARS buffers (§4) | ≈40.95 | ≈0.24 GiB |
| **Known-static sum** | **≈3122.3 (≈3.05 GiB)** | **≈18.74 GiB** |
| **Live-measured converged total (§6)** | ≈6710 (40265.9/6) | **≈40.26 GiB** |
| **Reconciled fraction** | | **≈46.6%** |
| **Unexplained gap** | | **≈21.5 GiB (≈53.4%)** |

**Per Stage 8's explicit instruction — "If >10% remains unexplained: do
not select memory fixes yet. Investigate the missing category first" —
this audit does NOT proceed to select or prioritize specific memory fixes
(Stages 10/14 are answered as CLASSIFICATION ONLY, not a fix package; see
§10/§14).**

The live diagnostic (§6) narrows *where in the lifecycle* the gap opens:
the jump from the highest observed `after_initialize_state_complete`
reading (21,917.9 MiB) to the converged `after_first_remap_pressurevars_
call` ceiling (40,265.9 MiB) — roughly **18-19 GiB** — occurs strictly
*after* `fv3_gpu_initialize_state` returns, i.e. *outside* everything
Stages 1-5's static accounting covers (FieldRegistry + Workspace +
grow-only buffers are all allocated *inside* `initialize_state`). This
narrows the missing category to whatever happens during the **first full
production dispatch call** (WP-A→B→C→D→PT/DELZ→REMAP_UV→REMAP_Q→
REMAP_PRESSUREVARS), before that call's own named buffers (§4, ≈82
MiB/rank) can plausibly account for it.

**Leading candidates for the ~18-21 GiB gap (hypotheses, not confirmed
this session):**
1. **CUDA lazy kernel/module loading.** Modern CUDA (12.x) defaults to
   lazy per-kernel cubin loading on first launch. WP-A/B/C/D collectively
   contain a large number of distinct templated CUDA kernel
   instantiations; loading all of them for the first time during the
   first full dispatch pass, potentially across multiple embedded SM
   architectures in the fat binary, is a plausible multi-GiB-per-context
   cost that would not show up until exactly this point in the lifecycle.
2. **Halo exchange buffers** (§5) — never numerically sized this session;
   sized for all boundary-crossing fields × halo depth × 6 neighbor
   directions, and possibly allocated/grown lazily on first use rather
   than in `initialize_state`.
3. **Underestimated FieldRegistry/WorkspaceLayout sizing** — the Stage
   1-5 static estimates were derived by reading source and computing
   byte counts by hand, not by adding a direct measurement inside
   `initialize_state`'s own allocation loop; a systematic undercount
   (wrong level count, wrong array count, or a missed
   `WorkspaceLayout` consumer) cannot be ruled out.
4. **CUDA runtime/library-internal reservations** not enumerated this
   session (e.g. any linked cuBLAS/CUB/Thrust internal pools, driver
   virtual-memory bookkeeping) — not investigated.

**None of these is confirmed.** Distinguishing them requires a
**follow-up, barrier-synchronized** diagnostic (an `MPI_Barrier` on all 6
ranks immediately before each checkpoint, plus a new checkpoint pair
bracketing the first production-dispatch call specifically) that this
session did not execute, per the cost-efficiency mandate to define a stop
condition and stop once evidence is sufficient to make the present
decision (§16) — evidence here is sufficient to certify the ≥90% gate is
**not** met and to say precisely where in the lifecycle to look next; it
is not sufficient to select a specific memory fix.

## 9-14. Stages 9-14 — Lifetime graph / overprovisioning / dedup / `State_To_FV` reconfirmation / scenarios / cleanup package: **CLASSIFICATION ONLY, DEFERRED PENDING §8**

Per Stage 8's own gate, these stages are answered at the classification
level only — no fix is selected or recommended for implementation:

- **Lifetime graph / overprovisioning (§9/§10):** the one *confirmed*
  overprovisioning-by-design instance remains the `transport` workspace
  slice (34-array sizing for `d_sw`'s peak need vs. 17 for
  `update_dz_d`'s, §2) — unchanged from Prompt 100's finding, ≈765.8
  MiB/rank, of which roughly half (≈17 arrays' worth, ≈383 MiB/rank ≈2.3
  GiB aggregate) is idle capacity during `update_dz_d`'s window. This is
  the only *quantified* overprovisioning candidate; it cannot by itself
  close a 21.5 GiB gap.
- **Dedup/alias opportunities (§11):** Prompt 100's `State_To_FV`
  zero-new-bytes alias into `WorkspaceId::vertical` is reconfirmed valid
  (§12) but, as Prompt 100 already established, restores only the
  pre-existing baseline — it does not touch the gap identified in §8.
  The ~17 WP-A/C-grid FieldRegistry scratch fields (§3) as *possible*
  workspace-slice aliases are flagged, not resolved (LOW confidence,
  unverified this session).
- **Scenario A/B/C (§13):** cannot be scored responsibly while the
  dominant ≈21.5 GiB component is unidentified — any Scenario A/B/C
  MiB-recovered estimate built only on the confirmed ≈2.3 GiB
  `transport`-slice overprovisioning would represent at most ~11% of the
  gap, not a resolution of the memory-capacity blocker.
- **Narrow cleanup package vs. architecture classification (§14):**
  **classification deferred** — this audit does not yet have enough
  information to certify whether the true driver is a *narrow, portable
  fix* (e.g., a workspace/FieldRegistry sizing bug) or a *architectural*
  one (e.g., inherent CUDA context/kernel-code overhead at this many
  concurrent contexts). That distinction is exactly what the
  barrier-synchronized, first-dispatch-bracketing follow-up diagnostic
  (§8) would resolve.

## 12. Stage 12 — `State_To_FV` zero-byte reconfirmation

Reconfirmed unchanged from Prompt 100: the 8 `state_to_fv_*` buffers
(≈144 MiB/rank aggregate, §4) hold no duplicate storage and alias safely,
at zero new bytes, into the lifetime-disjoint `vertical` workspace arena.
**This reconfirmation does not change Prompt 100's conclusion** — even a
perfect zero-byte fix only restores the *existing* baseline headroom,
which itself now sits inside a converged ≈40.26 GiB/40.34 GiB footprint
whose composition is only 46.6% reconciled (§8). `State_To_FV` remains
CLOSED — STILL MEMORY-BLOCKED (Prompt 100's decision stands, reinforced).

## 15. Stage 15 — Inputs for Prompt 102 multi-GPU study

| Input | Value | Source |
|---|---|---|
| Converged per-GPU footprint at 6 ranks/1 A100, current CUDA defaults (no `State_To_FV`) | ≈40.26-40.33 GiB used, ≈40.34 GiB total, ≈69-249 MiB free | §6, live-measured, job 58638073 |
| Footprint is front-loaded (fully resident by first dispatch call, stable through ≥30 steps) | Confirmed | §6 |
| Known-static, named per-rank floor (independent of the unexplained gap) | ≈3.05 GiB/rank (FieldRegistry ≈870 MiB + Workspace ≈2170.4 MiB + REMAP_Q/PRESSUREVARS ≈82 MiB) | §1-5 |
| Unexplained per-rank component (must scale with rank count until root-caused) | ≈3.58 GiB/rank (≈21.5 GiB / 6) — **treat as scaling with ranks-per-GPU, not confirmed fixed-per-GPU, until §8's follow-up resolves it** | §8 |
| Confirmed safe, zero-byte `State_To_FV` alias (does not change footprint) | Available, ≈144 MiB/rank if `State_To_FV` is ever re-enabled | §12, P100 |
| 2-GPU/3-ranks-per-GPU density | Previously validated correct (CUDA-aware MPI) at this density, memory-safe | `CURRENT_STATE.md` §5/§7 |
| **Caution for Prompt 102** | Do not assume the ≈21.5 GiB gap is fixed-overhead-per-GPU (would *not* scale with rank count, favoring fewer, larger ranks) vs. per-rank-context overhead (*would* scale, favoring more GPUs with fewer ranks each) — §8's open hypothesis #1 (CUDA lazy module loading) is inherently **per-context**, i.e. per-rank, which would argue for reducing ranks-per-GPU as a headroom-recovery lever independent of any workspace/FieldRegistry fix. This is a hypothesis to test in Prompt 102, not a established fact. | §8 |

### 15a. Addendum — Prompt 102 resolved the Caution above

Prompt 102 (`docs/performance/p102_multigpu_scaling_architecture_2026-09-29.md`
§Stage 1) directly measured, single-node, 1-rank-per-GPU and 2-ranks-per-GPU
C512 memory (job 58639804, `--gres=gpu:4`, zero source changes): 6746 MiB and
13450 MiB respectively, on two *different* physical GPUs at matched density
(dev2≡dev3, dev0≡dev1). **Answer: the ≈21.5 GiB gap is overwhelmingly
per-rank/per-context, not fixed-per-GPU** — `M_fixed_per_gpu ≈ 0.04 GiB`,
`M_per_rank_context ≈ 6.55–6.7 GiB` (HIGH confidence on the qualitative
split; MODERATE on exact magnitude — a small ~1.5% super-linear residual
appears only at 6-ranks-per-GPU density, not yet isolated). This confirms
§8's hypothesis #1 direction (per-context) over the fixed-per-GPU
alternative, and directly favors spreading ranks across more GPUs as a
headroom-recovery lever (Prompt 102 §Stage 8/12: 2+ GPUs alone clears
`State_To_FV`'s ~440 MiB shortfall with no code change to `State_To_FV`
itself). **RECONCILIATION status upgraded from INCOMPLETE to
CLASS-RESOLVED (per-rank), MAGNITUDE-APPROXIMATE** — no itemization of
which specific CUDA runtime/library/MPS sub-components make up the ~3.5-3.6
GiB/rank runtime-context residual was attempted; that itemization remains
open but is no longer load-bearing for the ranks-per-GPU vs. GPU-count
tradeoff, which is now resolved.

## 16. Stage 16 — Project decision

**MEMORY INVENTORY: PARTIALLY COMPLETE.** Static accounting (Stages 1-5)
is complete and self-consistent for every *named* production allocation.
Live telemetry (Stage 6) independently reconfirms the converged ≈40.3 GiB
total and proves it is front-loaded/stable. **Reconciliation (Stage 8)
fails its ≥90% gate at ≈46.6%.** Per that stage's own explicit rule, no
memory fix is selected, prioritized, or recommended for implementation
from this audit (Stages 9-14 are classification-only, not a fix package).

**Is `State_To_FV` safe to re-enable after any cleanup identified here?
NO** — no cleanup was identified with enough confirmed headroom to close
even the *known* ≈440 MiB-short gap from Prompt 100, let alone the newly
quantified ≈21.5 GiB unexplained component. `State_To_FV` remains CLOSED
— STILL MEMORY-BLOCKED (unchanged from Prompt 100).

**Recommended next step (not undertaken here, out of this audit's
scope):** a second, narrowly-scoped diagnostic job — barrier-synchronized
across all 6 ranks, with two new checkpoints bracketing the *first full
production dispatch call* specifically (immediately before/after, not
conflated with `initialize_state` or with `remap_pressurevars`) — to
determine which of §8's four hypotheses accounts for the ≈18-21 GiB
jump. Until that is done, Prompt 102's multi-GPU study should treat the
unexplained component's rank-count-scaling behavior as unknown (§15).

## 17. Reverted instrumentation record

5 `cudaMemGetInfo` checkpoints + 1 helper function, added to
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` for this audit's job 58638073
only, gated behind `FV3_GPU_MEMDIAG=1` (default off). **Fully reverted**
immediately after the job completed; CUDA library
(`build_cuda_r8_intelmpi`) and `StandAlone_FV3_Dycore.x`
(`GEOSfvdycore_fortran_perf/build`) rebuilt/relinked afterward. Verified
byte-identical to pre-audit state: `strings ... | grep -c P101_MEMDIAG`
→ 0; relinked binary sha256 `2339366b...186e2064d` matches Prompt 99's
own documented post-rebuild hash exactly. No production source, selector,
or default was changed by this audit. Nothing committed or pushed.
Experiment artifacts (job 58638073, `experiments/p101_memdiag_c512/`,
`experiments/p101_memdiag_c512.sbatch`) left on disk under
`GEOSfvdycore_fortran_perf/experiments/` as run evidence, matching the
project's existing convention for prior prompts' experiment directories.

## Open items for a future prompt

- Root-cause the ≈18-21 GiB first-dispatch-call memory jump (§8) via a
  barrier-synchronized, dispatch-bracketing diagnostic — this is the
  single highest-priority follow-up; nothing else in this audit's
  findings can be acted on with confidence until it is resolved.
- Numerically size the halo send/recv buffers (§5) — never computed this
  session, a named gap in the static accounting.
- Verify the FieldRegistry per-field byte counts directly from source
  (§3's estimate is category-based, ±15%, not derived field-by-field).
- Assess whether any of the ~17 WP-A/C-grid FieldRegistry scratch fields
  (§3/§11) could be aliased into `WorkspaceLayout` slices instead —
  flagged, not resolved.
