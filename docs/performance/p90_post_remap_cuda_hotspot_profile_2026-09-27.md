# Prompt 90 — Post-Remap-CUDA C512 Hotspot Profile

Date: 2026-09-27
Branch: `optimization/cuda-fv3-performance`
Scope: bounded profiling + next-target selection only. No numerical code
changed, no new CUDA ports added, no production defaults changed. PT/DELZ,
REMAP_UV, REMAP_Q, R4 not touched. Tracer-CUDA optimization not reopened
(P83 STOP remains binding). Nsight Compute not used.

## Stage 0/1 — Provenance and Selector Verification

- Branch `optimization/cuda-fv3-performance`, working tree unchanged from
  Prompt 89-Resume (uncommitted diffs described in `CURRENT_STATE.md` §10;
  nothing reset/reverted).
- `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x` sha256
  `82c1e9b61b31a2af0c599c261eeb490539c98fee4e93f1487567923c1ef9fae0` —
  re-verified identical to the Prompt 89-Resume rebuild (no rebuild
  occurred between then and now).
- Selector source verified by direct read of `fv_mapz.F90`
  (`GEOSfvdycore_fortran_perf` checkout, gitignored `@fvdycore` tree):
  - `use_cuda_pt_delz_remap`: CUDA-default, requires `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN`
    to opt OUT. No opt-in needed.
  - `use_cuda_remap_uv`: CUDA-default since P84A, requires
    `RUN_CUDA_FV3_REMAPUV_FORCE_FORTRAN` to opt OUT. No opt-in needed.
  - `use_cuda_remap_q`: CUDA-default since P89, requires
    `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN` to opt OUT. No opt-in needed.
  - `use_cuda_tp2d` (`fv_tracer2d.F90`): gate is `hord==8 .and. lim_fac==1.0
    .and. kind(q)==8 .and. fv3_gpu_backend_enabled()`. Production first-run
    config uses `hord_mt/vt/tm=6`, so this gate is **structurally false**
    in production regardless of any override — tracer stays Fortran not
    because of a selector default but because the shape condition never
    holds. This resolves an apparent inconsistency (the selector itself has
    no opt-in guard, unlike the other three) without contradicting the
    P83 STOP policy.
- Provenance matches expectations. No STOP condition triggered — proceeded
  to Stage 2.

## Stage 2 — Production C512 Run

Single-arm run, zero environment overrides beyond the established best
config (MPS ON, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, Graph OFF) — this is pure "whatever
`fv_mapz.F90`/`fv_tracer2d.F90` decide by themselves" production behavior.

- Job 58593086, node `warpa004`, 6 MPI ranks / 1 A100, R8, Intel MPI
  2021.13 production stack.
- `P90_PROVENANCE`: sha256 match confirmed.
- `rc=0`, `P90_DYN_DATE_COUNT count=60` (60/60 calls completed).
- No FATAL/OOM/Abort/NaN/Inf in the log.
- Sbatch script: `GEOSfvdycore_fortran_perf/experiments/p90_c512_profile.sbatch`.

## Stage 3 — Ranked Timer Table

Per-call figures (ms/call = summed value over 60 calls ÷ 60):

| Timer | ms/call | % of FV_DYN_LOOP | % of TOTAL | Backend |
|---|---|---|---|---|
| TOTAL | 3932.4 | — | 100% | mixed |
| FV_DYN_LOOP | 1706.8 | 100% | 43.4% | mixed |
| DYN_CORE | 850.5 | 49.8% | 21.6% | CUDA (Gate C2) |
| TRACER_2D | 430.7 | 25.2% | 11.0% | Fortran (P83 STOP, structurally hord≠8) |
| REMAPPING | 499.6 | 29.3% | 12.7% | mixed |
| — REMAP_PRESSUREVARS | 131.4 | 7.7% (26.3% of REMAPPING) | 3.3% | Fortran, never attempted |
| — REMAP_TOTALENERGYCON | 129.3 | 7.6% (25.9% of REMAPPING) | 3.3% | Fortran, never attempted, has `mpp_global_sum` |
| — REMAP_NH (delz) | 127.4 | 7.5% (25.5% of REMAPPING) | 3.2% | CUDA (P69) |
| — REMAP_T (pt) | 50.7 | 3.0% (10.1% of REMAPPING) | 1.3% | CUDA (P69) |
| — REMAP_Q | 35.3 | 2.1% (7.1% of REMAPPING) | 0.9% | CUDA (P89) |
| — REMAP_UV | 16.9 | 1.0% (3.4% of REMAPPING) | 0.4% | CUDA (P84A) |
| — REMAP_GETT | 4.8 | 0.3% (1.0% of REMAPPING) | 0.1% | Fortran, negligible |
| COMM_TOTAL | 69.7 | 4.1% | 1.8% | MPI (staged, host-based) |

No other timer in the requested set reached the ~2%-of-FV_DYN_LOOP
inclusion threshold.

## Stage 4 — Baseline Comparison

| Timer | Pre-port baseline | Current (P90) | Classification |
|---|---|---|---|
| REMAP_T (pt) | 197.16 ms (P69 Fortran) / 45.64 ms (P69 CUDA) | 50.7 ms | **REDUCED**, matches P69's CUDA figure (dispatching CUDA) |
| REMAP_NH (delz) | 258.43 ms (P69 Fortran) / 127.07 ms (P69 CUDA) | 127.4 ms | **REDUCED**, matches P69's CUDA figure |
| REMAP_UV | ~10.3x-slower misread (P71) corrected to ~10.23x faster (P76A), ~16.55 ms CUDA-side (P76A) | 16.9 ms | **REDUCED**, matches P76A's CUDA figure |
| REMAP_Q | 116.469 ms (P88B Fortran arm) / 36.033 ms (P88B CUDA arm) | 35.3 ms | **REDUCED**, matches P88B's CUDA figure |
| TRACER_2D | P78 measured regression window; P83 final gate 1.196x (still net slower/marginal, STOP) | 430.7 ms, Fortran-dispatch (structural, not override) | **UNCHANGED** — never left Fortran; today's absolute value is the plain Fortran cost, consistent with the closed-track decision |
| REMAP_PRESSUREVARS | not previously measured/isolated as a named target | 131.4 ms | **NOW DOMINANT** among unported hotspots — was previously masked by REMAP_T/REMAP_NH's larger pre-port Fortran cost |
| REMAP_TOTALENERGYCON | not previously measured/isolated as a named target | 129.3 ms | **NOW DOMINANT** (tied with REMAP_PRESSUREVARS), same masking effect |
| REMAPPING (total) | 17.74s/30 calls ≈ 591 ms/call pre-P88B-CUDA (P88B Fortran arm) | 499.6 ms/call | **REDUCED** overall, consistent with 3 successive positive CUDA ports |

**What is the new bottleneck?** With PT/DELZ, REMAP_UV, and REMAP_Q all
CUDA-dispatching, the previously second-tier Fortran stages
`REMAP_PRESSUREVARS` and `REMAP_TOTALENERGYCON` are now the two largest
*unported* contributors inside `REMAPPING` — each roughly 2.6-3.8x larger
than any remaining ported stage's Fortran-vs-CUDA gap could close further.
Outside `REMAPPING`, `TRACER_2D` (430.7 ms/call) remains the single
largest Fortran-side absolute cost in the whole `FV_DYN_LOOP`, but that
track is explicitly closed per P83's binding STOP and out of scope for
target selection.

## Stage 5 — Backend Coverage Table

| Timer | Backend | Note |
|---|---|---|
| DYN_CORE | CUDA | Gate C2, full WP-A/B/C/D acoustic loop |
| REMAP_T (pt) | CUDA | P69, CUDA-default |
| REMAP_NH (delz) | CUDA | P69, CUDA-default |
| REMAP_UV | CUDA | P84A, CUDA-default |
| REMAP_Q | CUDA | P89, CUDA-default |
| REMAP_PRESSUREVARS | Fortran | never attempted |
| REMAP_TOTALENERGYCON | Fortran | never attempted |
| REMAP_GETT | Fortran | negligible cost, not a target |
| TRACER_2D | Fortran | P83 STOP; structurally excluded from CUDA in production shape |
| COMM_TOTAL | MPI (host-staged) | Stage 1 of the staged communication plan; CUDA-aware exists but not perf-significant (§5 CURRENT_STATE) |
| FV_DYN_LOOP / TOTAL / REMAPPING | MIXED | sums of the above |

## Stage 6 — Static Map of Top Hotspots

### 1. `REMAP_PRESSUREVARS` (`fv_mapz.F90:1239-1516`)

- Timer wraps a block computing `pe`, `peln`, `pk`, `pkz` from the
  post-remap `delp` (and related pressure-derived diagnostics) after the
  vertical remap stages.
- Loop structure: `!$OMP parallel do` over `j=js,je`, with per-column
  (`i`,`k`) inner loops — same iteration shape as the already-ported
  PT/DELZ/REMAP_UV/REMAP_Q device kernels (one thread per column is a
  natural mapping).
- MPI/halo: **none** found in this range — only a diagnostic `mpp_pe()`
  rank-ID print (`[P60_EXIT_DUMP]`), not a reduction or exchange.
- Arrays: `pe`, `peln`, `pk`, `pkz`, `delp` — same tracer/state-array
  family already device-resident from PT/DELZ/REMAP_UV/REMAP_Q; no new
  host-side state expected.
- Residency: since PT/DELZ and REMAP_UV/REMAP_Q already operate on
  device-resident vertical-column data at this point in `Lagrangian_to_
  Eulerian`, a CUDA port of this block could plausibly reuse the same
  per-column dispatch pattern and scratch-slot conventions established by
  `fv3_remap_q.cu`/PT-DELZ's kernels — no new transfer expected by
  inspection, though this must be confirmed by a dedicated mapping task,
  not assumed here.
- Guards: none observed beyond the general `use_cuda_*` style selector
  pattern already used elsewhere; a new `use_cuda_remap_pressurevars`-
  style gate would follow the same pattern.
- **Verdict: MPI-free, same shape as already-ported stages, plausible
  reuse of existing device-column infrastructure.**

### 2. `REMAP_TOTALENERGYCON` (`fv_mapz.F90:1521-1773`)

- Timer wraps energy-conservation adjustment: `!$OMP parallel` computing
  `te_2d` per column, gated by `last_step .and. (.not.do_adiabatic_init)`
  and `consv > consv_min` (i.e., not unconditionally active every call —
  its measured cost here reflects a run where the gate was active).
- MPI/halo: **5 occurrences of `mpp_global_sum(domain, tmp_2D,
  flags=sflag)`** found in this range — genuine cross-tile global
  reductions, not diagnostics.
- This is a materially different risk class from every stage ported so
  far (PT/DELZ, REMAP_UV, REMAP_Q are all purely per-column/local — none
  cross tile boundaries). A CUDA port here would need either (a) a
  device-side global reduction plus a host sync point, or (b) staying
  host-resident for the reduction itself while porting only the
  per-column compute around it — a more invasive design decision than
  any prior port in this line of work.
- **Verdict: similar raw cost to REMAP_PRESSUREVARS, but a strictly
  higher engineering-risk candidate due to the `mpp_global_sum` calls.**

### 3. `TRACER_2D` (context only, not a candidate — explicitly closed)

- Largest single Fortran-side absolute cost (430.7 ms/call) but excluded
  by the binding P83 STOP decision; not static-mapped further here since
  it is out of scope for target selection.

## Stage 7 — Nsight Systems Decision

**Skipped.** The per-call timer table in Stage 3, cross-referenced
against each CUDA port's own previously-documented Fortran-vs-CUDA figures
(Stage 4), was sufficient to (a) confirm actual CUDA dispatch for all
three retained ports, and (b) unambiguously rank `REMAP_PRESSUREVARS` and
`REMAP_TOTALENERGYCON` as the two largest never-attempted hotspots, with a
clear, well-grounded discriminator (`mpp_global_sum` presence/absence)
between them. `nsys` would add confirmation but not new decision-relevant
information at this stage; per the task's own instruction to skip it when
the profile is already clear, it was not run. Nsight Compute was not used
(forbidden this task).

## Stage 8 — Next-Target Candidates

| Candidate | Measured cost (ms/call) | Portability (shape match to existing ports) | Comm/MPI risk | Reuse potential | Payoff (if ported at similar ratio to REMAP_Q's ~3.2x) |
|---|---|---|---|---|---|
| `REMAP_PRESSUREVARS` | 131.4 | High — per-column, `!$OMP parallel do` over `j`, same shape as PT/DELZ/REMAP_UV/REMAP_Q | **None** — no `mpp_global_sum`/halo found | High — likely reuses existing device-column dispatch/scratch conventions | ~90 ms/call recovered (≈2.3% of TOTAL) if ported at a comparable speedup ratio |
| `REMAP_TOTALENERGYCON` | 129.3 | Medium — per-column compute wrapped by cross-tile reduction | **High** — 5x `mpp_global_sum` calls, real cross-tile communication | Low — would need new device-reduction or host-boundary design, not a drop-in reuse | Similar raw ms/call to above, but design/engineering cost and correctness-risk are substantially higher per unit of measured benefit |
| `TRACER_2D` (reference only, NOT proposed) | 430.7 | N/A — explicitly closed track | N/A | N/A | Excluded: P83's binding STOP; not re-opened without fresh explicit authorization |

Only two real candidates emerged from this profile; a third was not
identified as materially close in cost (REMAP_GETT and COMM_TOTAL are
both far smaller and not comparable).

## Stage 9 — Mainline Track Reassessment

- **`MOIST_CAPPA`/`USE_COND`**: unaffected by this profile — this run used
  the standard non-moist Riemann-solver path (as it always does for this
  harness); no new timing evidence bears on its cost. Classification:
  **unchanged, still an open dedicated-prompt item** (§7 CURRENT_STATE);
  not started.
- **`a_imp <= 0.999`**: unaffected — this run used `a_imp=1.0` (SIM1) per
  the standing production first-run config. Classification: **unchanged,
  still an open dedicated-prompt item**; not started.
- **MPS/fusion operational wiring**: this run itself used MPS ON + both
  fusion flags manually set in the sbatch script (the established best
  config), consistent with prior findings — it did not reveal a new
  reason to fold this into every standing job script beyond what was
  already known. Classification: **unchanged, low-priority operational
  cleanup item** (§7 CURRENT_STATE); not started.

None of these three items were started, modified, or reprioritized above
the Stage 10 selection below.

## Stage 10 — Selected Next Target

**NEXT TARGET: `REMAP_PRESSUREVARS`** (`fv_mapz.F90:1239-1516`).

**WHY**: it is the single largest never-attempted Fortran hotspot
remaining in `REMAPPING` (131.4 ms/call, 26.3% of REMAPPING, 3.3% of
TOTAL) with the same per-column loop shape as every CUDA port already
successfully implemented in this line of work (PT/DELZ, REMAP_UV,
REMAP_Q), and — critically — **zero MPI/global-reduction involvement**,
unlike its similarly-sized neighbor `REMAP_TOTALENERGYCON` (129.3 ms/call,
5x `mpp_global_sum` calls). Given the measured near-tie in raw cost
between the two, the MPI-free candidate is the clearly lower-risk,
higher-reuse-potential choice — consistent with every prior successful
port in this project (PT/DELZ, REMAP_UV, REMAP_Q) having also been purely
per-column/local.

**NEXT PROMPT SHOULD BE**: a mapping-only task (numerics/shape/guard
survey of `REMAP_PRESSUREVARS`, no implementation), following the same
pattern established by P84B (REMAP_UV mapping) and P88A (REMAP_Q fill
mapping) before any CUDA code is written.

## Summary

| Item | Status |
|---|---|
| Provenance/selector verification (Stage 0/1) | PASS, no STOP triggered |
| Production C512 run (Stage 2) | PASS, rc=0, 60/60 DYN Date, job 58593086 |
| Ranked timer table (Stage 3) | complete |
| Baseline comparison (Stage 4) | complete — REMAP_T/REMAP_NH/REMAP_UV/REMAP_Q confirmed REDUCED (CUDA-dispatching); REMAP_PRESSUREVARS/REMAP_TOTALENERGYCON NOW DOMINANT among unported stages |
| Backend coverage (Stage 5) | complete |
| Static hotspot map (Stage 6) | complete, top 2 real candidates (TRACER_2D excluded as closed) |
| Nsight Systems (Stage 7) | SKIPPED — timer data sufficient |
| Candidate table (Stage 8) | complete, 2 candidates |
| Mainline reassessment (Stage 9) | complete, all 3 items unchanged/not started |
| Next target selection (Stage 10) | `REMAP_PRESSUREVARS` |

**No numerical code changed. No new CUDA ports added. No production
defaults changed. PT/DELZ, REMAP_UV, REMAP_Q, R4 not touched. Tracer-CUDA
optimization not reopened. Nsight Compute not used.**

## Addendum (Prompt 91) — Mapping/Harness Follow-Up

Prompt 91 performed the mapping-only follow-up this doc's Stage 10
recommended. Full result:
`docs/performance/p91_remap_pressurevars_cuda_port_plan_2026-09-27.md`.

Headline results: the 131.4 ms/call timer's active numerical content is
only ~50 of its 278 source lines (rest is dead branches/diagnostics); zero
MPI confirmed by full-body read (not just static mapping); data is HOST
ONLY at entry (same starting condition as every prior port); CUDA reuse
potential is VERY HIGH (nearly all arithmetic already exists in
`fv3_pt_delz_remap.cu`); a host-only golden-reference harness
(`test_remap_pressurevars_host_reference.cpp`, Option C transcription, 10
fixtures) was built and passed 10/10 under the project's real CMake/CTest
build. Decision: **GO — proceed to CUDA port** in a follow-up prompt. No
production source changed in Prompt 91.

## Addendum (Prompt 92/93) — Implementation and Default Enable

Prompt 92 implemented and GPU-validated the CUDA port (opt-in); Prompt 93
independently reconfirmed performance robustness (5/5 paired C512 wins,
mean TOTAL +1.805%; one 60-call longer run, TOTAL +1.676%) and flipped
`use_cuda_remap_pressurevars` to the CUDA production default. See
`p92_remap_pressurevars_cuda_implementation_2026-09-27.md` and
`p93_remap_pressurevars_default_policy_2026-09-28.md`.

## Addendum (Prompt 94) — Post-Default-Enable Profile Refresh

**This doc's own ranked timer table above (Stage 3) is now SUPERSEDED as a
current-state snapshot** — it reflects the pre-Prompt-93 configuration in
which `REMAP_PRESSUREVARS` was still Fortran by default. It remains valid
as the historical baseline Prompt 94 diffed against.

Prompt 94 reran this exact profiling methodology after Prompt 93's default
flip and found: `REMAP_PRESSUREVARS` dropped from 131.4 ms/call (Fortran)
to 38.11 ms/call (CUDA) — a 3.448x reduction, consistent with Prompt 93's
own paired/longer-run results. With that shift, `REMAP_TOTALENERGYCON`
(150.16 ms/call, unchanged/never attempted) is now unambiguously the
single largest hotspot within `REMAPPING` and the new selected next target
— this doc's own Stage 8 candidate-table entry for `REMAP_TOTALENERGYCON`
remains the operative static map, refined further by Prompt 94's Stage 8
re-verification (2 `mpp_global_sum` calls execute per production call, not
5 — 5 is the correct static source count across both mutually-exclusive
branches; likely bottleneck is COMPUTE, not COMMUNICATION). Full result:
`docs/performance/p94_post_pressurevars_cuda_hotspot_profile_2026-09-28.md`.
No numerical code changed in Prompt 94.
