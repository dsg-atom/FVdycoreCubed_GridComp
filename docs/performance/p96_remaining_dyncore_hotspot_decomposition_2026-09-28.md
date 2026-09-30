# Prompt 96 — Decompose Remaining FV_DYN_LOOP/DYN_CORE Hotspots After Remap Ports

Model: SONNET. Branch: `optimization/cuda-fv3-performance`. No CUDA kernels
implemented, no numerical/production code changed, no commit/push.

## 0. Provenance (Stage 0)

Executable: `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`.
- realpath: same physical file via both `/discover/nobackup/projects/imvi-nse/jli30/...`
  and `/gpfsm/dnb06/projects/p295/jli30/...` mounts.
- mtime: 2026-09-28 12:45:06 (rebuilt same day, 3 min after Prompt 95's
  `fv_mapz.F90` `TECons_*` test-only instrumentation edit at 12:42:55).
- sha256: `9ae5d7cf...2951c3` — **differs** from CURRENT_STATE.md's recorded
  P93/94 hash (`b482b647...9b3ec96`). Root cause confirmed: Prompt 95's own
  test-only timing probes in `fv_mapz.F90`, rebuilt. `DynCore_GridCompMod.F90`
  and `FV_StateMod.F90` (this prompt's own focus files) last modified
  2026-08-27 — untouched by that rebuild. Selectors/defaults unchanged
  (`strings` confirms `RUN_CUDA_FV3_{PTDELZ,REMAPQ,PRESSUREVARS}_FORCE_FORTRAN`
  present, none forced by default).
- Linkage: `ldd` confirms Intel MPI 2021.13 (`libmpi.so.12` from
  `/usr/local/intel/oneapi/2024/mpi/2021.13/lib/`) — authoritative production
  stack per `discover_platform.md`.
- **Decision**: reuse Prompt 94's job **58619479** (C512, 6 ranks/1 A100,
  MPS ON, fusion ON, current production defaults, 60 calls, rc=0, PASS) and
  its archived logs rather than submit a new Slurm job — no selector/default
  change has occurred since that run, and its logs (both the filtered `.out`
  and the full underlying `scratch_*.log`) contain sufficient granularity for
  every stage below. This is a `git`-adjacent artifact-reuse decision, not a
  numerical/production change.
- `cost_efficiency.md` referenced by `CLAUDE.md` does not exist in
  `docs/agent_instructions/` — confirmed absent, not read.

## 1. Timer accounting (Stage 1)

Two independent, orthogonal instrumentation frameworks coexist in the
production Fortran stack:

- **`fv_timing_mod`** (`@fvdycore/tools/fv_timing.F90`): flat, name-keyed
  accumulator (`nblks=100`), auto-registers and auto-prints every distinct
  name ever passed to `timing_on`. `timing_prt` prints ALL registered blocks
  — the "fixed 14-timer set" used by every prior prompt (P69-P95) was an
  artifact of the **sbatch script's own `grep -E` filter**
  (`p94_c512_profile.sbatch`), not a limitation of the instrumentation.
  Confirmed by re-grepping the full underlying log
  (`experiments/p94_c512_profile/scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8.log`)
  for every `dyn_core.F90` sub-timer name (`c_sw`, `d_sw`, `UPDATE_DZ`,
  `Riem_Solver`, `PG_D`, `FV3_BELJAARS`) — **zero matches**. These
  `timing_on`/`timing_off` call sites exist in `dyn_core.F90` source but do
  **not execute** under `RUN_CUDA_FV3=1`: the CUDA backend dispatches the
  entire WP-A/B/C/D subsystem through the WP-E C ABI as one opaque call: the
  Fortran-internal `dyn_core.F90` body (and its own timer brackets) is
  bypassed entirely, consistent with `project_architecture.md`'s "Fortran
  must not call individual CUDA kernels" / subsystem-level dispatch. This is
  a genuine, useful finding: **`DYN_CORE`'s 845.85 ms/call has no
  Fortran-visible internal breakdown in the current CUDA-default
  configuration** — any further decomposition would require GPU-side
  profiling (`nsys`, not attempted — see Stage 6) or WP-E runtime
  instrumentation (source change, out of scope).
- **MAPL_Profiler** (GridComp-level, `MAPL_TimerAdd`/`On`/`Off` in
  `DynCore_GridCompMod.F90`): a **separate, previously-unexamined**
  hierarchical timer tree wrapping the whole `Run()` method. Never grepped
  by P90/93/94 (they only matched `fv_timing_mod`'s fixed subset by name).
  Values are seconds accumulated over each row's own `#cycles` (here, 60 for
  every `Run`-level row) — verified, not assumed.

**Reconciled tree** (Mean column, ms/call = seconds/60*1000; all values from
job 58619479's existing logs, no new instrumentation):

```
RUN (MAPL)                                   3257.0   [NESTED root, 60 cycles]
+-- DYN_PROLOGUE (MAPL)                         37.5   [SIBLING, no MPI]
+-- DYN_CORE (MAPL)                           2396.67  [SIBLING]
|   +-- PULL_TRACERS                             5.17  [NESTED]
|   +-- State_To_FV                            112.83  [NESTED, MIXED-MPI]
|   +-- NH_ADIABATIC_IN                        ~0.00   [NESTED]
|   +-- FV_DYNAMICS                           2157.17  [NESTED]
|   |     +-- FV_DYN_LOOP (fv_timing)         1619.80  [NESTED, see below]
|   |     +-- non-loop residual                537.37  [UNKNOWN -- see note]
|   +-- PUSH_TRACERS                            18.83  [NESTED]
|   +-- FV_To_State                            100.50  [NESTED, clean CPU]
+-- DYN_EPILOGUE (MAPL)                        688.67  [SIBLING, no MPI]
(RUN-level residual, unattributed)            ~134.16  [UNKNOWN, ~4.1% of RUN]

FV_DYN_LOOP (fv_timing, own scope)            1619.80
+-- DYN_CORE (fv_timing == CUDA subsystem)      845.85  [NESTED, GPU-opaque]
+-- TRACER_2D                                   453.32  [NESTED, Fortran]
+-- REMAPPING                                   414.05  [NESTED, mixed]
    (children sum = 1713.22 vs parent 1619.80: +93.42/+5.77% OVERLAP/NOISE,
     same class as REMAPPING's own +3% internal sub-timer noise already
     documented in P94 -- NOT a new hidden hotspot)
```

**Answering the task's own Q1/Q2 directly**: `DYN_CORE`'s remaining 845.85
ms/call is the CUDA subsystem itself (already fully GPU-ported per Gate C2;
no further Fortran-side decomposition exists to mine). `FV_DYN_LOOP`'s own
1619.80 ms/call is **already fully (over-)explained** by
`DYN_CORE`+`TRACER_2D`+`REMAPPING` (net unexplained ≈ 0, within known
~5.8% timer-overlap noise) — there is **no hidden hotspot inside
`FV_DYN_LOOP`**. The genuinely new, previously-unexamined time is **outside
`FV_DYN_LOOP`**, at the MAPL GridComp level: `DYN_EPILOGUE` (688.67),
`State_To_FV`+`FV_To_State`+`PULL/PUSH_TRACERS` (237.33), and a partially
unattributed `FV_DYNAMICS` non-loop residual (537.37, see Stage 7). This
inverts the task's own framing (which expected the unexplained time to be
*inside* `DYN_CORE`/`FV_DYN_LOOP`) — the real remaining CPU-side material
cost lives in the surrounding GridComp orchestration layer, not inside the
dyn_core numerical subsystem.

## 2. DYN_CORE (fv_timing) decomposition (Stage 2)

| Timer | ms/call | % DYN_CORE | Backend | Structure |
|---|---|---|---|---|
| `DYN_CORE` (fv_timing, == entire CUDA dispatch) | 845.85 | 100% | CUDA (WP-A/B/C/D, opaque) | single C-ABI call, no Fortran sub-timers execute |

No further breakdown available without GPU-side profiling or WP-E source
instrumentation (both out of scope/unnecessary for this task — see Stage 6).
`dyn_core.F90`'s own `c_sw`/`d_sw`/`UPDATE_DZ_C`/`UPDATE_DZ`/`Riem_Solver`/
`PG_D`/`FV3_BELJAARS` timer brackets are dead code under the CUDA backend
(confirmed empirically, Stage 1) — they would only populate under a
Fortran-backend run, which is not this task's target configuration.

## 3. Bounded instrumentation (Stage 3)

**Judged unnecessary.** The existing MAPL_Profiler + `fv_timing_mod` data
already mined from job 58619479's archived logs provide decisive,
source-mapped attribution for the top candidates (Stage 7) without any new
instrumentation. No test-only timers added this prompt.

## 4. C512 profile (Stage 4)

Reused job **58619479** (node `warpa003`, rc=0, 60/60 calls, PASS, MPS ON,
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`,
current production defaults: PT/DELZ/REMAP_UV/REMAP_Q/REMAP_PRESSUREVARS =
CUDA, TRACER_2D/REMAP_TOTALENERGYCON = Fortran). Provenance caveat: the
executable's sha256 changed since that job ran (Stage 0) due to an unrelated
`fv_mapz.F90` instrumentation rebuild; selectors/defaults are unchanged
(`strings`-verified), so the job's numbers remain valid for this task's
purpose. See table below for ms/call, %DYN_CORE(MAPL), %RUN, %TOTAL.

| Region | ms/call | % DYN_CORE(MAPL) | % RUN | % TOTAL |
|---|---|---|---|---|
| DYN_CORE (MAPL, incl.) | 2396.67 | 100% | 73.6% | 62.5% |
| FV_DYNAMICS (incl.) | 2157.17 | 90.0% | 66.2% | 56.2% |
| FV_DYN_LOOP (fv_timing) | 1619.80 | 67.6% | 49.7% | 42.2% |
| DYN_EPILOGUE (MAPL) | 688.67 | n/a (sibling) | 21.2% | 17.9% |
| State_To_FV (nested in DYN_CORE) | 112.83 | 4.7% | 3.5% | 2.9% |
| FV_To_State (nested in DYN_CORE) | 100.50 | 4.2% | 3.1% | 2.6% |
| FV_DYNAMICS non-loop residual | 537.37 | 22.4% | 16.5% | 14.0% |

(TOTAL = 3836.74 ms/call, whole-run wall-clock convention per P69-P95; RUN =
3257.0 ms/call, MAPL's own per-call nested root.)

## 5. CPU/GPU classification (Stage 5)

| Region | Backend | MPI? | Notes |
|---|---|---|---|
| `DYN_CORE` (fv_timing/CUDA subsystem) | CUDA | YES (six-tile halo, internal) | Already Gate-C2 GPU-ported; opaque to Fortran timers |
| `TRACER_2D` | Fortran | Partial (`COMM_TRACER` nested) | CLOSED NO-GO (P83) — excluded from candidates |
| `REMAPPING` | MIXED | Partial (`REMAP_TOTALENERGYCON`'s `mpp_global_sum`) | PT/DELZ, UV, Q, PRESSUREVARS = CUDA default; NH/T/GETT/TOTALENERGYCON = Fortran |
| `DYN_PROLOGUE` | Fortran | NO (grep-confirmed zero `mpp_`) | Small (37.5 ms/call); diagnostic-tendency baseline snapshots only |
| `DYN_EPILOGUE` | Fortran | NO (grep-confirmed zero `mpp_`/`global_sum` across ~1000 lines) | Large, local-only, external-library-heavy (see Stage 7) |
| `State_To_FV` | Fortran | YES (`mpp_get_boundary` production branch; `mpp_update_domains` doubly-periodic branch) | MIXED |
| `FV_To_State` | Fortran | NO (grep-confirmed) | Clean, local array marshalling + `fv_getPKZ` |
| `PULL_TRACERS`/`PUSH_TRACERS` | Fortran | not chased (small: 5.17/18.83 ms/call) | Below material-cost threshold for this task |
| `FV_DYNAMICS` non-loop residual | Fortran | UNKNOWN (partial — includes 2 of 4 `COMM_TOTAL` brackets + `NEST_BCs`) | Not further decomposed (Stage 3 judged unnecessary; see Stage 11 note) |

## 6. Optional nsys (Stage 6)

**Skipped.** Source-map + timer attribution (Stages 1, 2, 5, 7) is already
decisive: a large (688.67 ms/call), zero-MPI, well-bounded new candidate
(`DYN_EPILOGUE`) was identified purely from existing logs and source grep,
with no ambiguity requiring a GPU timeline capture to resolve. No Nsight
Compute used under any circumstance (per task constraint); no Nsight Systems
capture taken this prompt.

## 7. Top-3 remaining CPU hotspots (Stage 7)

Excluding `TRACER_2D` (CLOSED NO-GO) and `REMAP_TOTALENERGYCON`'s exact
reduction (CLOSED BLOCKED BY COMMUNICATION, P95):

### Candidate 1 — `DYN_EPILOGUE`

- **File/range**: `DynCore_GridCompMod.F90:4470-5473` (~1000 lines).
- **Cost**: 688.67 ms/call (21.2% RUN, 17.9% TOTAL) — larger than
  `TRACER_2D` or `REMAPPING`.
- **Routines**: unconditional `getAllWinds`, `getVorticity`, `getDivergence`,
  `getEPV` (confirmed via direct read, no gating `if` precedes them);
  feature-flag-gated `tropovars`, `Energetics` (`DynCore_GridCompMod.F90:7751-7818`,
  ~67 lines, grep-confirmed zero `mpp_`), `getOmega`,
  `computeMassFluxes`/`fillMassFluxes` (gated by `AdvCore_Advection==2`);
  extensive individually-`if(associated(...))`-gated `FILLOUT3`/`FILLOUT3r8`
  export-diagnostic marshalling.
- **Loop structure**: per-column/per-level diagnostic computation, no
  cross-tile dependency evident in the reviewed calls.
- **Numerical role**: post-dyn_core diagnostic derivation (winds, vorticity,
  divergence, EPV, energetics, omega) for GEOS export/history streams — not
  part of the prognostic state update itself.
- **MPI/halo**: exhaustive grep of the full 4470-5473 range for
  `mpp_\|global_sum` returned **zero matches** — confirmed communication-free.
- **Data residency**: host-resident (GEOS `MAPL_State` arrays); no GPU
  residency today.
- **CUDA helper reuse**: LOW. `getAllWinds`/`getVorticity`/`getDivergence`/
  `getEPV`/`getOmega`/`tropovars` are **not owned source in this repo** —
  they resolve to the external, shared `MAPL` base-class library (imported
  via `use MAPL`; exhaustive `grep -rln`/`-rlin` across this checkout found
  no local definition). Porting would mean either duplicating external MAPL
  utility logic (against "do not duplicate numerical functionality") or
  reaching into a shared dependency outside this project's ownership
  boundary.
- **Branch complexity**: MODERATE — one large `SW_DYNAMICS` guard (inactive
  branch for this config), several independent feature-flag gates
  (`doTropvars`, `doEnergetics`, `AdvCore_Advection`), and dozens of
  individually-gated diagnostic-export calls.
- **Portability**: **MODERATE-HIGH RISK** — large and clean of MPI, but its
  dominant cost drivers are outside this project's source ownership.

### Candidate 2 — `State_To_FV`

- **File/range**: `FV_StateMod.F90:2519-2702` (~184 lines).
- **Cost**: 112.83 ms/call (3.5% RUN, 2.9% TOTAL).
- **Routine**: MAPL-state-array -> FV_Atm-internal-array marshalling for
  winds.
- **Numerical role**: boundary/halo preparation ahead of dyn_core entry.
- **MPI/halo**: contains genuine embedded communication —
  `mpp_get_boundary(FV_Atm(1)%u, FV_Atm(1)%v, ..., gridtype=DGRID_NE,
  complete=.true.)` (line 2591, the actual production `grid_type<4` branch)
  and `mpp_update_domains` (2582/2584, doubly-periodic `grid_type>=4` branch,
  inactive for `grid_type=0`).
- **Data residency**: host-resident.
- **CUDA helper reuse**: MODERATE — array marshalling shape is similar to
  existing WP-E field-registry patterns, but the embedded `mpp_get_boundary`
  call is exactly the class of halo/communication routine this project's
  constraints forbid replacing with a stub ("do not replace integrated halo
  communication with stubs").
- **Branch complexity**: LOW (one `grid_type` branch).
- **Portability**: **HIGH RISK** — real halo communication embedded in the
  timer; not a clean local-compute-only boundary like the prior remap ports.

### Candidate 3 — `FV_To_State`

- **File/range**: `FV_StateMod.F90:2704-2780` (~76 lines).
- **Cost**: 100.50 ms/call (3.1% RUN, 2.6% TOTAL).
- **Routine**: FV_Atm-internal -> MAPL-state-array marshalling
  (post-dyn_core), plus `call fv_getPKZ(STATE%VARS%PKZ,STATE%VARS%PE)`.
- **MPI/halo**: exhaustive grep — **zero `mpp_` calls**. Confirmed CLEAN.
- **Data residency**: host-resident.
- **CUDA helper reuse**: MODERATE — `fv_getPKZ` is a local per-column op,
  structurally similar to the already-ported REMAP_PRESSUREVARS pattern;
  the marshalling itself is mostly array copies.
- **Branch complexity**: LOW.
- **Portability**: **CLEAN** — smallest of the three, but the only one that
  is both fully MPI-free AND entirely within project-owned source (`FV_StateMod.F90`,
  not an external MAPL library call), unlike Candidate 1.

## 8. Remaining communication accounting (Stage 8)

| Source | ms/call | Notes |
|---|---|---|
| `BITWISE_EXACT_SUM` (`REMAP_TOTALENERGYCON`'s 2 `mpp_global_sum` calls) | ~135.0 | P95, reconfirmed, unchanged this prompt |
| `COMM_TOTAL` (fv_timing, all 4 on/off pairs in `fv_dynamics.F90`, 2 inside + 2 outside `FV_DYN_LOOP`) | 66.48 (total, not separable by fv_timing alone) | Existing metric, not re-split this prompt |
| `COMM_TRACER` / `COMM_TRACER_MAX` | 17.00 / 2.365 | Existing metric (nested tracer-halo comm) |
| `State_To_FV`'s embedded `mpp_get_boundary` | included in 112.83 (not separately timed) | New this prompt (Stage 7, Candidate 2) |
| `DYN_EPILOGUE` | 0 (confirmed MPI-free) | New this prompt |
| `FV_To_State` | 0 (confirmed MPI-free) | New this prompt |
| `DYN_PROLOGUE` | 0 (confirmed MPI-free) | New this prompt |

**Communication-limited assessment: PARTIAL.** The single largest known
communication cost remains `REMAP_TOTALENERGYCON`'s `BITWISE_EXACT_SUM`
reduction (~135 ms/call, CLOSED BLOCKED, P95) — unchanged, not reopened.
Outside that, communication is a secondary but non-zero factor (`COMM_TOTAL`
66.48, `State_To_FV`'s embedded halo call, both not further decomposed here)
next to a larger pool of confirmed MPI-free CPU compute (`DYN_EPILOGUE`,
`FV_To_State`, `DYN_PROLOGUE`) that remains a legitimate, if
lower-portability, target class.

## 9. BITWISE_EXACT_SUM policy classification (Stage 9 — classify only)

1. **Regression-only, production-reproducibility, or both?** **BOTH.**
   `exact_sum` defaults to `1` (`BITWISE_EXACT_SUM`) in `fv_arrays.F90:816`
   for every configuration, production or regression — there is no separate
   "regression mode" default; every HS/production run to date (P37-P95) has
   used the live default.
2. **Where is `flagstruct%exact_sum` set?** `fv_arrays.F90:815-817` declares
   the default (`0`=`NON_BITWISE_EXACT_SUM` and `2`=`BITWISE_EFP_SUM` exist
   as commented-out dead-code alternatives, confirming the field is a real,
   intentionally-configurable tri-state integer, not hardwired). Bound at
   runtime via `fv_control.F90:1263` (`exact_sum => Atm%flagstruct%exact_sum`).
3. **Configurable in any real runnable configuration?** **YES.** Confirmed
   this prompt: `exact_sum` is listed inside the `namelist /fv_core_nml/`
   declaration (`fv_control.F90:666-680`, specifically line 674) — it is a
   standard, user-facing FV3 namelist variable, settable via `input.nml`
   without any source change. No standing job script overrides it from the
   default `1`.
4. **Would disabling it alter numerical reproducibility?** **YES.**
   `BITWISE_EXACT_SUM`'s entire purpose (per its name and P95's finding) is
   order-independent, bit-reproducible global summation across an arbitrary
   rank/decomposition layout. Switching to `NON_BITWISE_EXACT_SUM` (`0`)
   would very likely change the least-significant bits of the global
   `tesum`/`zsum` reduction results (and therefore `dtmp`, and therefore
   every column's `pt` correction) as a function of rank count/decomposition
   — a genuine numerical-reproducibility change, not a pure performance
   knob.

**Classification: CORRECTNESS/POLICY STUDY.** Per the task's explicit
instruction, this is documented for awareness only — **no change to
`exact_sum` proposed or made.**

## 10. Candidate table (Stage 10, max 3)

| Candidate | Cost (ms/call) | Backend | MPI risk | Residency | Portability |
|---|---|---|---|---|---|
| `DYN_EPILOGUE` | 688.67 | Fortran | NONE (confirmed) | Host | MODERATE-HIGH-RISK (external-library-owned compute) |
| `State_To_FV` | 112.83 | Fortran | MODERATE (embedded `mpp_get_boundary`) | Host | HIGH-RISK |
| `FV_To_State` | 100.50 | Fortran | NONE (confirmed) | Host | CLEAN (smallest, but fully project-owned and MPI-free) |

`TRACER_2D` excluded per CLOSED NO-GO (P83), not merely for size.
`REMAP_TOTALENERGYCON` local loops excluded per CLOSED BLOCKED (P95); no new
evidence this prompt contradicts that finding (Stage 8 reconfirms the same
~135 ms/call reduction cost, unchanged).

## 11. Mainline gap reassessment (Stage 11)

| Track | Classification | Priority |
|---|---|---|
| `MOIST_CAPPA`/`USE_COND` CUDA | CORRECTNESS COVERAGE (unimplemented config path) | Dedicated future prompt; unaffected by this prompt's findings |
| `a_imp <= 0.999` CUDA | CORRECTNESS COVERAGE (unimplemented config path, broader scope) | Dedicated future prompt; unaffected |
| MPS + fusion flags into standing job scripts | OPERATIONAL WIRING | Low risk, low urgency; unaffected |
| `BITWISE_EXACT_SUM` policy | POLICY STUDY (Stage 9) | No action; correctness-sensitive, not to be changed without explicit fresh authorization |
| `FV_DYNAMICS` non-loop residual (537.37 ms/call, Stage 1/4) | MAP/HARNESS candidate (partially communication, not yet decomposed) | New, open, lower priority than Candidates 1-3 (smaller `NEST_BCs` component likely near-zero for HS; 2 of 4 `COMM_TOTAL` brackets live here — communication-flavored, not clean compute) |

No change to any of these tracks' priority ordering relative to
`CURRENT_STATE.md` §7/§9, except adding the new `FV_DYNAMICS` residual as a
distinct, lower-priority open item.

## 12. Next task selection (Stage 12)

**Selected: MAP/HARNESS task on `FV_To_State`** (`FV_StateMod.F90:2704-2780`).

Rationale: of the three candidates, `FV_To_State` is the only one that is
simultaneously (a) confirmed MPI-free, (b) fully within this project's own
source ownership (not an external MAPL library call, unlike `DYN_EPILOGUE`),
and (c) structurally simple (array marshalling + one local per-column
`fv_getPKZ` call, similar in shape to the already-successfully-ported
REMAP_PRESSUREVARS pattern). `DYN_EPILOGUE` is larger (688.67 vs 100.50
ms/call) but its dominant cost drivers (`getAllWinds`/`getVorticity`/
`getDivergence`/`getEPV`/`getOmega`) are owned by the external `MAPL`
library, not this repository — porting them would risk duplicating
numerical functionality outside this project's ownership boundary
(`project_architecture.md`: "do not duplicate numerical functionality
across work packages", implicitly extending to external libraries this
project does not own). `State_To_FV` carries real embedded halo
communication (`mpp_get_boundary`), which this project's constraints treat
as HIGH RISK to touch ("do not replace integrated halo communication with
stubs"). `FV_To_State`, while the smallest of the three, is the cleanest
next MAP/HARNESS candidate with a genuinely favorable risk/reward ratio,
consistent with the mapping-before-porting discipline already established
for PT/DELZ, REMAP_UV, REMAP_Q, and REMAP_PRESSUREVARS.

**NEXT TARGET**: `FV_To_State` (`FV_StateMod.F90:2704-2780`).
**MEASURED BASIS**: 100.50 ms/call (3.1% RUN, 2.6% TOTAL), confirmed
MPI-free, confirmed fully project-owned source.
**NEXT TASK TYPE**: MAP/HARNESS (source-map + host-reference harness, no
CUDA implementation yet — mirrors the Prompt 91/95 pattern).

`DYN_EPILOGUE` and `State_To_FV` remain documented, lower-priority
alternatives (Stage 10) if `FV_To_State`'s own mapping proves too small a
payoff once fully quantified (its ceiling is inherently capped by its own
100.50 ms/call size — smaller than any prior successfully-ported timer).

## 13. CUDA porting saturation (Stage 13)

**(E) Operational/runtime infrastructure boundary**, with a secondary
**(A) still-clean CPU compute** pool that is smaller and lower-payoff than
any previously-ported target, plus one large-but-externally-owned block.

- **(A) Still-clean CPU compute**: YES, but modest — `FV_To_State` (100.50),
  `DYN_PROLOGUE` (37.5), and the `NEST_BCs`-adjacent portion of the
  `FV_DYNAMICS` non-loop residual are genuinely MPI-free, project-owned, and
  unported, but individually smaller than every prior successful port
  (smallest prior port, REMAP_UV, was ~21-38 ms/call *pre-port*, but delivered
  outsized relative speedup; these candidates are more comparable in raw size
  to REMAP_UV/REMAP_Q but with less-established reuse infrastructure).
- **(B) Rejected tracer**: unchanged, CLOSED NO-GO, not reopened.
- **(C) Exact/global communication**: unchanged, ~135 ms/call
  `BITWISE_EXACT_SUM`, CLOSED BLOCKED, not reopened; `State_To_FV`'s embedded
  `mpp_get_boundary` adds a second, smaller, HIGH-RISK communication-touching
  candidate this prompt identified but does not recommend porting.
- **(D) Uncovered correctness configs**: unchanged from CURRENT_STATE.md §7
  (`MOIST_CAPPA`/`USE_COND`, `a_imp<=0.999`) — orthogonal to this prompt.
- **(E) Operational/runtime infrastructure**: the **largest single newly-found
  block** (`DYN_EPILOGUE`, 688.67 ms/call) is CPU-side, MPI-free, and
  material, but its cost is dominated by calls into the external `MAPL`
  library rather than this project's own numerical algorithms — closer to
  "infrastructure/diagnostic marshalling this project does not own" than to
  the clean, self-contained numerical kernels (PT/DELZ, REMAP_UV/Q/
  PRESSUREVARS) that made every prior port successful.

**Largest excluded hotspot**: `DYN_EPILOGUE` (688.67 ms/call) — excluded
from the Stage 10 top-3 recommendation not for size (it is the largest) but
because its dominant sub-computations are owned by the external `MAPL`
library, not this project's source tree, making a clean, ownership-respecting
CUDA port materially harder than any prior successful target.

## Final Report

- **Executable**: `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`
  (sha256 `9ae5d7cf...2951c3`, provenance caveat documented §0)
- **C512**: PASS (reused job 58619479, rc=0, 60/60 calls, 0 FATAL, 0 NaN/Inf,
  expected selectors — established by P94, unchanged)
- **Job id**: 58619479 (reused, not resubmitted)
- **FV_DYN_LOOP ACCOUNTING**: FV_DYN_LOOP 1619.80 ms/call; Known compute
  (DYN_CORE+TRACER_2D+REMAPPING) ≈1578.22 ms/call; Known communication
  (REMAP_TOTALENERGYCON reduction) ≈135.0 ms/call; **Unexplained ≈ 0**
  (children sum exceeds parent by 93.42 ms/+5.77%, same-class
  timer-overlap noise as REMAPPING's own documented +3% — not a hidden
  hotspot). The real newly-found time is OUTSIDE FV_DYN_LOOP (see DYN_CORE
  BREAKDOWN below).
- **DYN_CORE BREAKDOWN** (Rank | Region | ms/call | % DYN_CORE(MAPL) | Backend):
  1. FV_DYNAMICS(incl.) | 2157.17 | 90.0% | mixed (contains FV_DYN_LOOP)
  2. FV_DYN_LOOP | 1619.80 | 67.6% | mixed (DYN_CORE-CUDA/TRACER_2D-Fortran/REMAPPING-mixed)
  3. FV_DYNAMICS non-loop residual | 537.37 | 22.4% | Fortran, partially MPI (unattributed)
  4. State_To_FV | 112.83 | 4.7% | Fortran, MIXED-MPI
  5. FV_To_State | 100.50 | 4.2% | Fortran, clean
  6. PUSH_TRACERS | 18.83 | 0.8% | Fortran
  7. PULL_TRACERS | 5.17 | 0.2% | Fortran
  (DYN_EPILOGUE is a SIBLING of DYN_CORE at the RUN level, not its child —
  listed separately: 688.67 ms/call, 21.2% RUN)
- **TOP REMAINING CPU HOTSPOTS**:
  1. `DYN_EPILOGUE` (`DynCore_GridCompMod.F90:4470-5473`), 688.67 ms/call, MPI
     risk NONE, portability MODERATE-HIGH-RISK (external MAPL library)
  2. `State_To_FV` (`FV_StateMod.F90:2519-2702`), 112.83 ms/call, MPI risk
     MODERATE (`mpp_get_boundary`), portability HIGH-RISK
  3. `FV_To_State` (`FV_StateMod.F90:2704-2780`), 100.50 ms/call, MPI risk
     NONE, portability CLEAN
- **COMMUNICATION**: BITWISE_EXACT_SUM ≈135.0 ms/call (unchanged, P95);
  Other MPI (`COMM_TOTAL` 66.48 + `COMM_TRACER` 17.00 + `State_To_FV`'s
  embedded `mpp_get_boundary`, not separately timed) ≈83.48+ ms/call;
  Communication-limited: **PARTIAL**
- **MAINLINE TRACKS**: MOIST_CAPPA/USE_COND = CORRECTNESS COVERAGE (deferred);
  a_imp<=0.999 = CORRECTNESS COVERAGE (deferred); MPS/fusion wiring =
  OPERATIONAL WIRING (low priority); BITWISE_EXACT_SUM = POLICY STUDY (no
  change)
- **CUDA PORTING SATURATION**: Still major clean CPU compute: **NO** (only
  modest, smaller-than-any-prior-port amounts remain project-owned and
  MPI-free); Largest excluded hotspot: `DYN_EPILOGUE` (688.67 ms/call,
  excluded because its dominant compute is owned by the external `MAPL`
  library, not this project's source tree)
- **NEXT TARGET**: `FV_To_State` (`FV_StateMod.F90:2704-2780`), MAP/HARNESS
  task
- **No production numerical changes**: YES
- **No commit/push**: YES

`REMAINING DYN_CORE PROFILE DECOMPOSED — NEXT TARGET: FV_To_State (FV_StateMod.F90:2704-2780) — MAP/HARNESS`

## Addendum — Prompt 97 closure (2026-09-28)

`FV_To_State` was mapped and harnessed (host reference harness, 8/8 PASS
including a mandatory sentinel-layout-transposition fixture). **Decision:
KEEP ON CPU — MEMORY/STATE BOUND**, not a CUDA candidate: its source
fields (`FV_Atm(1)%u/v/w/pt/delz/pe`) are already host-resident at entry
(D2H'd by the preceding CUDA remap dispatch calls' own per-call sync
protocol), and its destination (`STATE%VARS%*`) is consumed immediately by
the host/MAPL-only `DYN_EPILOGUE` block. A GPU port would add a redundant
H2D+D2H round trip around work that is already host-to-host, with no host
work actually eliminated. See
`docs/performance/p97_fv_to_state_cuda_port_plan_2026-09-28.md` for the
full Stage 0-16 analysis. `DYN_EPILOGUE` (688.67 ms/call,
external-MAPL-library-dominated) and `State_To_FV` (112.83 ms/call,
embedded `mpp_get_boundary`, HIGH RISK) remain this decomposition's
lower-priority, undemonstrated alternatives.

## Addendum — Prompt 98 closure (2026-09-28)

`State_To_FV` (`FV_StateMod.F90:2519-2702`) was mapped and harnessed
(host reference harness, 8/8 PASS incl. mandatory sentinel-layout
fixture). Its "HIGH-RISK" label above was based on the presence of an
embedded `mpp_get_boundary` call, not on measurement; direct TEST-ONLY
`MPI_Wtime()` cost-split instrumentation (added, measured, then fully
reverted — no production source changes remain, verified via `strings`
on the rebuilt binary) found the routine is actually
**LOCAL-COMPUTE-DOMINATED (~84% local: pre-boundary U/V copy 8.8% +
post-boundary PE-transpose/peln/pk/ps/delp/pt/delz/pkz 75.2%; only
~16% is the `mpp_get_boundary` call itself)**. This revises the risk
downward for an Option-A port (local compute only, halo call stays
host-side unchanged) — direct reuse precedent exists in the already-
production REMAP_PRESSUREVARS `peln=log(pe)`/`pk=exp(akap*peln)`
kernel. **Decision: GO — STATE_TO_FV HARNESS READY, RECOMMENDED NEXT
CUDA IMPLEMENTATION TARGET** (pending a separate authorized
implementation prompt). See
`docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md` for the
full Stage 0-18 analysis. `DYN_EPILOGUE` (688.67 ms/call,
external-MAPL-library-dominated) remains the sole open, unattempted
Prompt-96 candidate.
