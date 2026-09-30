# Prompt 98 — Map and Harness `State_To_FV` for CUDA Porting

Date: 2026-09-28
Branch: `optimization/cuda-fv3-performance`
Scope: MAP + HARNESS only. No production CUDA source added or modified.
No commit/push. No MPI/residency redesign. No Nsight Compute.

Structurally parallel to Prompt 97 (`FV_To_State`, decision: KEEP ON CPU).
This prompt covers the opposite direction: `State_To_FV`
(`FV_StateMod.F90:2519-2702`), called before `fv_dynamics()` each
`k_split` sub-iteration, whereas `FV_To_State` is called after.

## Stage 0 — Location / call site

- File: `GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90:2519-2702` (184 lines).
- Called once per `k_split` sub-iteration from `FV_Run`, immediately
  before `fv_dynamics()`.
- No conditional call wrapper found — always executed when the CUDA or
  Fortran dyn_core path runs FV_Run's main loop.

## Stage 1 — Block / category inventory

Categories (project convention, continued from P95-97): A = pure
elementwise copy, B = index-transpose, C = elementwise
arithmetic/transcendental, D = communication.

| Block | Lines | Category | Active in production? |
|---|---|---|---|
| `MAPL_Get` LATS/LONS | 2552-2553 | metadata fetch, unused downstream in this routine | executed, but dead weight |
| U/V interior copy | 2589-2590 | A | YES |
| `mpp_get_boundary` | 2591-2594 | D | YES |
| U/V boundary scatter-back | 2595-2604 | A (from buffer) | YES |
| W copy | 2607 | A | YES (`hydrostatic=.false.`) |
| PE transpose | 2613-2629 (else-branch 2623-2629 active) | B | YES |
| `peln = log(pe)` | 2631-2637 | C | YES |
| `pk = exp(akap*peln)` | 2639-2645 | C | YES |
| `ps` slice | 2647 | A | YES |
| `delp = pe(k+1)-pe(k)` | 2651-2658 | C | YES |
| `pt = PT*PKZ` | 2665-2666 | C | YES |
| `delz` copy | 2676 | A | YES (`hydrostatic=.false.`) |
| `pkz` copy | 2681 | A | YES |
| Doubly-periodic branch | 2578-2587 | — | NO (`grid_type>=4` never true) |
| SW_DYNAMICS branch | 2614-2621 | — | NO (`SW_DYNAMICS=.false.`, npz≠1) |
| `adjust_fv3_splits` call | 2685-2696 | — | NO (`ADJUST_SPLITS` default `.false.`, no override found) |

## Stage 2 — Guard table

| Guard | Production value | Branch taken |
|---|---|---|
| `FV_Atm(1)%flagstruct%grid_type>=4` | `grid_type=0` | else (2588-2605): interior copy + `mpp_get_boundary` |
| `SW_DYNAMICS` | `.false.` (set `.true.` only if `npz==1`; production npz=10 or 79) | else (2622-2649): PE/peln/pk/ps all computed |
| `FV_Atm(1)%flagstruct%hydrostatic` | `.false.` | W (2607) and DELZ (2676) copies active |
| `ADJUST_SPLITS` | `.false.` default, no production override found | dead, not executed |

No `MOIST_CAPPA`/`USE_COND`/`a_imp`/tracer/`nq`/`nwat`/`do_omega`
reference exists anywhere in `State_To_FV` — confirmed by direct source
read of the full routine body.

## Stage 3 — Field mapping table

| Field | Direction | Shape (C512 L10, npz=10) | Category |
|---|---|---|---|
| U, V | STATE→FV_Atm (interior only) | (nx,ny,10) | A + D (halo edge fill) |
| W | STATE→FV_Atm | (nx,ny,10) | A |
| PE | STATE (i,j,k)→FV_Atm (i,k,j) | (nx,11,ny) | B |
| PELN | derived from PE | (nx,11,ny) storage (Fortran), value at (i,k,j) | C |
| PK | derived from PELN | (nx,ny,11) storage (Fortran), value at (i,j,k) | C |
| PS | derived (slice of PE at k=km+1) | (nx,ny) | A |
| DELP | derived (`pe(k+1)-pe(k)`) | (nx,ny,10) | C |
| PT (dry T) | STATE.PT × STATE.PKZ | (nx,ny,10) | C |
| DELZ | STATE.DZ → FV_Atm.delz | (nx,ny,10) | A |
| PKZ | STATE.PKZ → FV_Atm.pkz (separate copy, not the derived PK) | (nx,ny,10) | A |

PELN and PK are Fortran-allocated in different memory layouts from each
other (`fv_arrays.F90`) but this is a storage detail only — no value
transpose occurs beyond the one genuine PE transpose. All four derived
fields (PELN, PK, PS, DELP) are **final `FV_Atm` output fields**
consumed later by `fv_dynamics`, not discarded intermediates.

## Stage 4/5 — Entry / output residency

- **Entry**: `STATE%VARS%*` fields are host-resident MAPL Internal/
  Export-state arrays — the same structures `FV_To_State` (P97) reads
  from in the reverse direction, confirming both routines share the same
  host-residency boundary.
- **Output**: `FV_Atm(1)%{u,v,w,pe,peln,pk,ps,delp,pt,delz,pkz}` feed
  directly into `fv_dynamics()`, whose CUDA branch performs its own H2D
  at entry (existing architecture, unchanged by this task). This means
  `State_To_FV`'s output is already destined for an H2D that exists
  regardless of where `State_To_FV` itself executes — **the opposite
  residency situation from P97's `FV_To_State`**, where the destination
  (`DYN_EPILOGUE`) is host-only and a port would have added a transfer.
  Here, a port has the potential to *remove* a transfer rather than add
  one, since the CUDA-resident `fv_dynamics` path could consume
  `State_To_FV`'s output directly on-device instead of via a fresh H2D.

## Stage 6 — `mpp_get_boundary` semantics

- One call per invocation, `DGRID_NE` staggering.
- Computes 4 edge buffers (`wbuffer`, `sbuffer`, `ebuffer`, `nbuffer`)
  but `State_To_FV` only consumes 2 of them (`nbuffer` → `U(:,jec+1,:)`,
  `ebuffer` → `V(iec+1,:,:)`) in its scatter-back loops — `wbuffer`/
  `sbuffer` are computed but unused by this routine (API-signature
  requirement, not a bug).
- Message size per buffer ≈ `nx × 10 × 8B` ≈ 512×10×8 = 40KB
  (edge-only, not full-tile) at C512/L10.
- ~61 calls per run, matching the profiler's measured cycle count.

## Stage 7 — Cost-split measurement (TEST-ONLY instrumentation, reverted)

**Method**: `MAPL_Profiler` sub-timer nesting (`MAPL_TimerOn`/`Off` for
`---S2F_PREBND`/`---S2F_BND`/`---S2F_POST`) was attempted first but the
new sub-timer names never appeared in `report_generic_profile`'s output
despite a correct rebuild+run — root cause not fully diagnosed (ruled
out a naive depth-limit hypothesis via source read of
`MAPL_Generic.F90`'s `report_generic_profile`/`reduce()`). Per the
project's bounded-debugging convention, abandoned rather than
continuing to dig, and switched to direct `MPI_Wtime()` wall-clock
timestamps + `write(6,...)` prints tagged `P98_TIMING` (using the
already-available `mpif.h` include and `mpp_pe()` for per-rank
identification). This succeeded cleanly.

**Harness bug found+fixed along the way**: the first two attempts (jobs
58622054, 58622119) silently ran a **stale binary** from a copied
`fv3.j`'s hardcoded `setenv EXPDIR .../p90_c512_profile` line (an
unfixed instance of the already-documented CURRENT_STATE.md §7 issue).
Caught via the job's own `***** USING ****` log line (§8's documented
lesson). Fixed by editing `p98_state_to_fv_cost_split/fv3.j` line 161 to
point `EXPDIR` at the correct run directory; resubmission (job
58622163) then ran the correct binary and produced valid data.

**Result** (job 58622163, C512/L10, MPS ON + fusion ON, 60 calls, 366
`P98_TIMING` samples = 61/rank × 6 ranks):

| Phase | Mean ms/call | % of total | Notes |
|---|---|---|---|
| Pre-boundary local (U/V copy) | ~10.0 | 8.8% | |
| `mpp_get_boundary` | ~18.1 | 16.0% | **rank-imbalanced**: 0.2-12ms on 4 ranks, 40-51ms on ranks 0 and 3 — synchronization wait, not bandwidth (40KB/buffer is trivial) |
| Post-boundary local (W, PE transpose, peln/pk, ps, delp, pt, delz, pkz) | ~85.4 | 75.2% | |
| **Total** | **~113.5** | 100% | Cross-validated against the independently-measured MAPL_Profiler `State_To_FV` timer: **113.7 ms/call** (0.2% agreement) |

**Decisive finding: `State_To_FV` is LOCAL-COMPUTE DOMINATED (~84%
local, ~16% communication)** — a materially more CUDA-port-favorable
profile than P95's `REMAP_TOTALENERGYCON` (~89% communication-dominated,
NO-GO) and qualitatively different from P97's `FV_To_State` (zero
communication, but NO-GO for residency reasons).

Instrumentation fully reverted after measurement (4 edit points removed
from `FV_StateMod.F90`; confirmed via `grep -c "p98_\|P98_TIMING\|MPI_Wtime"`
= 0 matches); binary rebuilt clean (sha256 `9ae5d7cf...951c3`, verified
0 occurrences of the string `P98_TIMING` in the rebuilt executable via
`strings`).

## Stage 8 — Communication characterization

Latency/wait-bound, not bandwidth-bound. 40KB/buffer is trivial for an
A100-node interconnect; the 40-51ms cost concentrated on 2 of 6 ranks
reflects synchronization wait tied to cubed-sphere corner/edge topology
and message-arrival-order effects in the 1-tile-per-rank, 6-rank
decomposition — consistent with the project's prior finding (P95/96)
that "communication cost" in these profiler numbers is often
wait-dominated rather than transfer-dominated.

## Stage 9 — Data volume estimate

Per rank, C512/L10 (nx=ny=512, npz=10):

- U,V,W,DELP,PT,DELZ,PKZ (7 fields, npz=10): 7 × 512×512×10×8B ≈ 146.8MB
- PE,PELN,PK (3 fields, npz+1=11): 3 × 512×512×11×8B ≈ 69.2MB
- PS (1 field, 2D): 512×512×8B ≈ 2.1MB
- Read+write local traffic ≈ 2×(146.8+69.2+2.1) ≈ **~436MB/call/rank**

Only U and V (2 fields × 512×10×8B ≈ 82KB total, not the full 436MB)
need to cross the `mpp_get_boundary` host boundary — the remaining
~354MB of traffic (PE/PELN/PK/PS/DELP/PT/DELZ/PKZ/W) never needs to
touch `mpp_get_boundary` at all.

## Stage 10 — CUDA architecture options (A/B/C, not D)

- **Option A (lowest risk)**: port only the local Category A/B/C work
  (84% of cost: pre-boundary U/V copy feeding the halo call, plus all
  post-boundary work) as CUDA kernels; keep `mpp_get_boundary`
  host-side unchanged. Only U/V (82KB) round-trips through host memory
  around the halo call; PE/PELN/PK/PS/DELP/PT/DELZ/PKZ/W (the bulk of
  the ~436MB) stay fully on-device throughout.
- **Option B**: port pre+post-boundary local work identically to A, but
  additionally attempt to fold the halo exchange itself into an
  on-device path via the existing `HaloTransport` abstraction —
  eliminates even the small U/V round-trip, at the cost of extending
  `HaloTransport` usage to this call site (broader architectural change,
  same interface already exists per project architecture doc).
- **Option C**: full persistent-residency port where `State_To_FV`'s
  output feeds directly into the already-GPU-resident `fv_dynamics()`
  CUDA path, removing the redundant H2D that currently exists at
  `fv_dynamics`'s CUDA-branch entry (per Stage 4/5's residency finding)
  — most architecturally consistent with the project's persistent-
  residency goal, but requires coordination with `fv_dynamics.F90`'s
  existing CUDA-branch entry logic (out of scope for this MAP/HARNESS
  task).
- Option D (full communication/MPI redesign) explicitly excluded per
  task scope.

**Recommendation**: Option A is the correct first implementation step —
isolates the local-compute Category A/B/C work (which this task's
harness fully covers) with minimal risk, deferring the halo-bridging
question (Option B/C) to a follow-on task once Option A's real
CUDA-vs-Fortran speedup is measured.

## Stage 11 — Persistent residency question

Same architectural question as P97 but with the **opposite answer**:
unlike `FV_To_State` (destination is host/MAPL-only `DYN_EPILOGUE`, so a
port adds a redundant round trip — NO-GO), `State_To_FV`'s destination
(`fv_dynamics()`'s CUDA path) is itself GPU-resident. Porting
`State_To_FV`'s local compute has the potential to genuinely *remove* a
transfer (the `fv_dynamics`-entry H2D) rather than add one, assuming
Option C is eventually pursued; even under Option A alone (no residency
change to `fv_dynamics`'s own entry H2D), the local compute simply moves
from host to device with no residency regression.

## Stage 12 — CUDA reuse analysis

Strong, directly reusable precedent exists in the already-production
CUDA codebase:

- The PE transpose (Category B) is structurally similar to index-order
  transforms already implemented in `fv3_pt_delz_remap.cu`/
  `fv3_remap_q.cu`.
- The `peln=log(pe)` / `pk=exp(akap*peln)` pattern (Category C) is
  **already implemented in production CUDA** by the REMAP_PRESSUREVARS
  kernel (P90-93, `fv_mapz.F90:1239-1516`) — this is not a from-scratch
  kernel design, it is the same formula on a different call site.

## Stage 13 — Harness unit selection

Standalone C++ host-reference harness transcribing only the
production-active branches (Categories A/B/C), mirroring
`test_fv_to_state_host_reference.cpp`'s (P97) exact structural
convention. `mpp_get_boundary` (Category D) is explicitly out of scope
for a numerical-correctness harness — it is a communication primitive,
not a kernel to unit-test in isolation; only the interior
(`isc:iec,jsc:jec`) U/V copy is transcribed, matching `FV_To_State`'s own
interior-only convention.

## Stage 14 — Fixture design

10 categories considered (2 marked N/A and documented rather than
silently omitted, matching P97's convention):

1. `smooth_3d_field`
2. `varying_vertical_profile`
3. `horizontal_gradient`
4. `boundary_columns_no_special_path` (interior-only scope check)
5. `uvw_active_nonhydrostatic`
6. `pt_delz_pkz_strong_contrast`
7/8. q/tracers, nonzero sphum — **N/A**: no q/moisture dependence and no
   tracer array is referenced anywhere in `State_To_FV`'s transcribed
   scope (confirmed by direct source read); documented, not silently
   omitted.
9. `production_shaped_reduced_dims` (ni=6,nj=5,npz=10 — npz matches the
   confirmed C512 L10 perf-track vertical resolution exactly)
10. `sentinel_layout_transposition_check` — **MANDATORY**, bijective
    position-encoded values on PE and all copied fields to catch any
    axis-swap mistake in a future port.

## Stage 15 — Host harness build + run

File: `cuda/dyn_core/tests/host/test_state_to_fv_host_reference.cpp`
(new), registered in `cuda/dyn_core/CMakeLists.txt` as
`fv3_wp_c_state_to_fv_host_reference_test`, same pattern as P97's
`fv3_wp_c_fv_to_state_host_reference_test`.

Built standalone with `g++ -O2 -std=c++17 -Wall` (no CUDA/GPU
dependency — host-only harness): **0 warnings, clean compile.**

```
smooth_3d_field                            PASS
varying_vertical_profile                   PASS
horizontal_gradient                        PASS
boundary_columns_no_special_path           PASS
uvw_active_nonhydrostatic                  PASS
pt_delz_pkz_strong_contrast                PASS
production_shaped_reduced_dims             PASS
sentinel_layout_transposition_check        PASS
RESULT=PASS (8 cases)
```

**8/8 PASS**, including the mandatory sentinel-layout fixture. All
invariants checked: bitwise-copy correctness (U/V/W/DELZ/PKZ), exact
PE-transpose correctness, PE monotonicity, `ps`/`delp` consistency with
`pe`, `pt = PT*PKZ` exactness, 0 NaN/Inf across all output fields,
physically-plausible `pk` range, bitwise repeatability across two runs
on identical input.

## Stage 16 — Performance ceiling estimate

`State_To_FV` measured at 112.83-113.7 ms/call (P96/P98 agreement).
Local-compute portion (Stage 7) = ~95.4 ms (84%) is portable under
Option A; only ~18.1 ms (16%) `mpp_get_boundary` stays host-side,
unchanged.

Using the REMAP_PRESSUREVARS precedent (3.45-3.50x direct speedup on a
structurally similar `log`/`exp`-heavy kernel, P90-93) as a conservative
comparison point, even a 2-3x speedup on the ~95.4ms local-compute
portion would reduce it to ~32-48ms, i.e. a **~40-55% reduction in
`State_To_FV`'s own total cost**. Against Prompt 94's measured
`TOTAL=3836.74 ms/call`, a ~50ms/call savings translates to roughly
**+1.0-1.5% TOTAL** — comparable in magnitude to REMAP_UV's (+~1.0%
TOTAL) and REMAP_PRESSUREVARS's (+1.805% TOTAL) confirmed contributions,
and clearly outside the "blocked"/negligible tier of P95's
`REMAP_TOTALENERGYCON` (≈0.5% ceiling) or P97's `FV_To_State` (0% —
architecturally negative).

Data-transfer overhead under Option A is minimal: only U/V (82KB) needs
to round-trip through host memory around `mpp_get_boundary`; the bulk of
the field set (PE/PELN/PK/PS/DELP/PT/DELZ/PKZ/W, ~354MB of the ~436MB
Stage-9 estimate) never needs to touch the host boundary at all under
this design.

## Stage 17 — GO / NO-GO decision

**GO.** Rationale:
- Local-compute-dominated (~84%), unlike P95's NO-GO (~89% communication).
- Clear, direct CUDA-kernel reuse precedent already in production
  (REMAP_PRESSUREVARS's log/exp formula, Stage 12).
- Residency direction favors a port (Stage 4/5/11) — opposite of P97's
  NO-GO, where the port would have added a transfer.
- Host harness 8/8 PASS, 0 warnings, mandatory sentinel fixture passes.
- Estimated non-trivial TOTAL benefit (~1.0-1.5%), comparable to already
  shipped production wins (REMAP_UV, REMAP_PRESSUREVARS).
- Lowest-risk implementation path (Option A) isolates the halo call
  entirely, avoiding the HIGH-RISK label this routine carried in P96's
  initial (pre-measurement) assessment — that risk was specifically
  about the embedded `mpp_get_boundary` call, which Option A explicitly
  keeps host-side and unmodified.

## Stage 18 — Project-phase decision

Per the task's DO NOT list, no CUDA implementation is performed in this
prompt. This mapping/harness result should be handed off as the
recommended next authorized implementation target: **CUDA
implementation of `State_To_FV`'s local-compute portion (Option A)** —
pending explicit fresh authorization for a new implementation prompt,
consistent with how every prior CLOSED-CUDA-default target in this
project (PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS) was scoped as
its own separate authorized prompt after its own MAP/HARNESS phase.

This resolves one of the two remaining Prompt-96 candidates
(`DYN_EPILOGUE` remains open, unattempted, feasibility still uncertain
due to external-MAPL-library dominance).

## Summary

- IMPLEMENTED: NO (mapping/harness only, per task scope)
- COMPILED: harness only (host g++, 0 warnings)
- GPU EXECUTED: N/A (no CUDA source added)
- NUMERICALLY VALIDATED: host harness 8/8 PASS (not yet a CUDA-vs-host
  comparison, since no CUDA implementation exists yet)
- No production source changes: YES (instrumentation added then fully
  reverted; verified via grep + `strings` on the rebuilt binary)
- No commit/push: YES

**Decision: STATE_TO_FV HARNESS READY — PROCEED TO CUDA PORT**

**Update 2026-09-28 (Prompt 99):** the recommended Option-A CUDA port
was implemented and validated at C32 (bitwise/round-off correct), but
C512 integration hit a device out-of-memory blocker specific to the
6-ranks-per-1-A100 production density — the Fortran-only baseline
already consumes nearly the entire 40GB A100 before this port's new
buffers are added. See
[p99_state_to_fv_cuda_implementation_2026-09-28.md](p99_state_to_fv_cuda_implementation_2026-09-28.md)
for the full finding. Final verdict: **STATE_TO_FV CUDA VALIDATION
BLOCKED** (device OOM at C512 density, not a kernel defect).
Production default remains Fortran.

**Update 2026-09-29 (Prompt 100):** a follow-up device-memory
accounting/buffer-reuse study found this port's own buffers are fully
aliasable into existing, already-allocated device memory (no
duplicate storage, no numerical change) but concluded the pre-existing
baseline headroom at C512 density is already far below the safety
margin required for any additional persistent per-rank buffer, with or
without this port. **Decision: STATE_TO_FV STILL MEMORY-BLOCKED —
BROADER DEVICE-RESIDENCY ARCHITECTURE REQUIRED.** See
[p100_state_to_fv_device_memory_reuse_2026-09-29.md](p100_state_to_fv_device_memory_reuse_2026-09-29.md).
