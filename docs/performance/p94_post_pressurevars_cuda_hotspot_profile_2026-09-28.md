# Prompt 94 — Post-PRESSUREVARS-Default C512 Hotspot Profile Refresh

Date: 2026-09-28. Branch: `optimization/cuda-fv3-performance`. Scope:
bounded profiling + next-target selection only, run after Prompt 93 flipped
`use_cuda_remap_pressurevars` to the CUDA production default. **No numerical
code changed, no new CUDA ports added, no production defaults changed, tracer
CUDA not reopened (P83 STOP remains binding), R4 not touched, no Nsight
Compute run, no implementation of the selected next target started.**

## Stage 0/1 — Provenance and Selector Verification

- `git status --short` (this repo): unchanged from session start — no
  unexpected drift.
- Executable
  (`GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`): sha256
  `b482b64703f0db7652b41310131815bef48f610d4b8ed208e7efef5d89b3ec96` —
  **exact match** to Prompt 93's rebuild. Intel MPI 2021.13 linkage
  confirmed (`ldd` → `libmpi.so.12`/`libmpifort.so.12` from
  `/usr/local/intel/oneapi/2024/mpi/2021.13/`).
- Selector source (`fv_mapz.F90`/`fv_tracer2d.F90`, direct read): PT/DELZ,
  REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS all CUDA-default (each requiring an
  explicit `*_FORCE_FORTRAN` to opt OUT); `use_cuda_tp2d` structurally false
  in production (`hord==8` gate never holds at `hord_mt/vt/tm=6`).
  Exactly the expected post-Prompt-93 state — no STOP condition triggered.

## Stage 2 — Production C512 Run

Single-arm, zero-override run (MPS ON, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, Graph OFF), reusing the exact Prompt-90
sbatch template/methodology.

- Job **58619479**, node `warpa003`, 6 MPI ranks / 1 A100, R8, Intel MPI
  2021.13.
- `rc=0`, `DYN Date` count 60/60.
- Zero genuine `FATAL`/OOM/`Abort`/NaN/Inf matches (only benign "INFO"
  substring false positives).
- Script: `GEOSfvdycore_fortran_perf/experiments/p94_c512_profile.sbatch`.

## Stage 3 — CUDA Dispatch Verification

The zero-override C512 run above does not itself emit per-call dispatch
counters (all `*_DUMP_EXIT` env vars are intentionally unset in a pure
"production defaults" run — the same limitation applies equally to Prompt
90's own run). To satisfy this stage's "confirm by logs/counters, not
inference from defaults" requirement without re-running the expensive C512
job, a **separate, short C32/L10 dispatch-verification smoke** (job
**58619656**, node `warpa003`, same binary, sha256-verified, 14s wall time)
was run with all four `_DUMP_EXIT` flags set and **zero** `_FORCE_FORTRAN`/
`_ENABLE` overrides — i.e., pure current defaults, dispatch-counted:

| Selector | Dump tag | Dispatch count | Expected (6 calls × 6 ranks) |
|---|---|---|---|
| `use_cuda_pt_delz_remap` | `[P60_EXIT_DUMP]` | 36/36 `=T` | 36 |
| `use_cuda_remap_uv` | `[P71_EXIT_DUMP]` | 36/36 `=T` | 36 |
| `use_cuda_remap_q` | `[P85_EXIT_DUMP]` | 36/36 `=T` | 36 |
| `use_cuda_remap_pressurevars` | `[P92_EXIT_DUMP]` | 36/36 `=T` | 36 |

rc=0, `DYN Date` 6/6, zero FATAL/NaN/Abort. **All four confirmed
CUDA-dispatching under pure current defaults — Stage 3: PASS**, with direct
counter evidence (not inferred from selector source alone). Tracer remains
Fortran (structural, unchanged, not separately dispatch-counted here since
its gate is shape-based, not env-based — already verified in Stage 1 by
source read, consistent with Prompt 90's own finding).

## Stage 4 — Ranked Timer Table (C512, job 58619479, 60 calls)

ms/call = accumulated seconds ÷ 60.

| Rank | Timer | ms/call | % REMAPPING | % FV_DYN_LOOP | Backend |
|---|---|---|---|---|---|
| — | TOTAL | 3836.74 | N/A | N/A | mixed |
| — | FV_DYN_LOOP | 1619.80 | N/A | 100% | mixed |
| 1 | DYN_CORE | 845.85 | N/A | 52.2% | CUDA (Gate C2) |
| 2 | TRACER_2D | 453.32 | N/A | 27.98% | Fortran (P83 STOP, structural) |
| 3 | REMAPPING | 414.05 | 100% | 25.56% | mixed |
| 4 | — REMAP_TOTALENERGYCON | 150.16 | 36.26% | 9.27% | Fortran, never attempted |
| 5 | — REMAP_NH (delz) | 127.06 | 30.69% | 7.84% | CUDA (P69) |
| 6 | — REMAP_T (pt) | 49.01 | 11.84% | 3.03% | CUDA (P69) |
| 7 | — REMAP_PRESSUREVARS | 38.11 | 9.20% | 2.35% | **CUDA (P93, new default)** |
| 8 | — REMAP_Q | 36.24 | 8.76% | 2.24% | CUDA (P89) |
| 9 | — REMAP_UV | 21.29 | 5.14% | 1.31% | CUDA (P84A) |
| 10 | COMM_TOTAL | 66.48 | N/A | 4.10% | MPI (staged, host-based) |
| 11 | — REMAP_GETT | 4.81 | 1.16% | 0.30% | Fortran, negligible |
| 12 | COMM_TRACER | 17.00 | N/A | 1.05% | MPI |
| 13 | COMM_TRACER_MAX | 2.365 | N/A | 0.15% | MPI |

Sum of the 7 REMAPPING sub-timers (426.68 ms) vs. reported REMAPPING
(414.05 ms) — within the ~3% non-additive accounting noise already
documented for this fixed 14-timer instrumentation set (not
strictly hierarchical; consistent with Prompt 90's own observation).

## Stage 5 — Comparison to Prompt 90

Caveat: Prompt 90's run (job 58593086, node `warpa004`) and this run (job
58619479, node `warpa003`) are two independent single-arm runs on different
nodes — per `CURRENT_STATE.md` §8's node-variance caveat, this comparison is
valid for **direction/relative ranking**, not precise magnitude.

| Timer | Prompt 90 (ms/call) | Prompt 94 (ms/call) | Classification |
|---|---|---|---|
| TOTAL | 3932.4 | 3836.74 | REDUCED |
| FV_DYN_LOOP | 1706.8 | 1619.80 | REDUCED |
| DYN_CORE | 850.5 | 845.85 | UNCHANGED (no cross-contamination) |
| TRACER_2D | 430.7 | 453.32 | UNCHANGED (within node variance; still Fortran/structural) |
| REMAPPING | 499.6 | 414.05 | REDUCED |
| **REMAP_PRESSUREVARS** | **131.4 (Fortran)** | **38.11 (CUDA)** | **REDUCED — 3.448x, matches P93's 3.194–3.498x paired range** |
| REMAP_TOTALENERGYCON | 129.3 | 150.16 | **NOW DOMINANT** — largest single REMAPPING sub-timer, within node variance of its own P90 figure but now unambiguously #1 |
| REMAP_NH (delz) | 127.4 | 127.06 | UNCHANGED |
| REMAP_T (pt) | 50.7 | 49.01 | UNCHANGED |
| REMAP_Q | 35.3 | 36.24 | UNCHANGED |
| REMAP_UV | 16.9 | 21.29 | UNCHANGED (within node variance) |
| REMAP_GETT | 4.8 | 4.81 | NO LONGER MATERIAL (already negligible in P90) |
| COMM_TOTAL | 69.7 | 66.48 | UNCHANGED |

**The REMAP_PRESSUREVARS shift, quantified:** ~131 ms Fortran → ~38 ms CUDA
(this run), a direct 3.448x reduction — closely consistent with Prompt 93's
independently-measured 3.194x (60-call longer run) and 3.498x (5-pair mean),
strong cross-validation that the flipped default is behaving exactly as
predicted.

**Re-ranking inside REMAPPING** (previously PRESSUREVARS and
TOTALENERGYCON were tied for largest at ~130 ms each in Prompt 90; with
PRESSUREVARS now at 38 ms, REMAP_TOTALENERGYCON (150.16 ms) is
unambiguously the single largest REMAPPING sub-timer, ahead even of
REMAP_NH (127.06 ms, the largest already-CUDA remap stage).

**REMAPPING summary:** 499.6 → 414.05 ms/call (−17.1%), consistent with
four successive positive CUDA ports (PT/DELZ, REMAP_UV, REMAP_Q,
REMAP_PRESSUREVARS) now all dispatching by default.

## Stage 6 — Next-Candidate Identification (excluding TRACER_2D)

The fixed 14-timer instrumentation set contains exactly one Fortran/mixed
timer that clears the ~2%-of-FV_DYN_LOOP materiality threshold besides
TRACER_2D (excluded, P83 STOP) and COMM_TOTAL (MPI infrastructure, not a
remap-numerics port candidate): **`REMAP_TOTALENERGYCON`** (150.16 ms/call,
9.27% of FV_DYN_LOOP). `REMAP_GETT` (4.81 ms/call, 0.30%) remains far below
threshold, unchanged from Prompt 90. This is confirmed by this run's fresh
measurement, not assumed from Prompt 90's tie — `REMAP_TOTALENERGYCON` is
now measurably the single largest unported hotspot in the profile, with no
second or third material non-tracer host candidate existing among the
instrumented timers.

## Stage 7 — Bounded Static Source Map

Only one candidate qualifies at material scale (Stage 6); no second or
third candidate exists to map.

### `REMAP_TOTALENERGYCON`

- **Source location** (re-verified this session, not assumed from Prompt
  90): the `timing_on('Remap_TotalEnergyConsv')`/`timing_off(...)` markers
  in the current `fv_mapz.F90` are at **lines 1612–1864** — not
  1521–1773 as Prompt 90 cited. This is a **line-number shift, not a
  content discrepancy**: Prompt 92/93 inserted a 14-line `[P92_EXIT_DUMP]`
  diagnostic block (now at lines 1596–1607) between Prompt 90's read and
  now, shifting everything below it down by the same amount. The subroutine
  content itself is unchanged.
- **Dominant structure**: two large per-column/per-level (`j`,`k`,`i`)
  `!$OMP do`-parallel loops — (1) computing `te_2d`/`zsum1`/`zsum0` from
  `pt`/`delp`/`delz`/`u`/`v`/`w`/`pkz` (lines 1631–1746), and (2) reapplying
  a single scalar correction `dtmp` back into `pt` across all `k`,`j`,`i`
  (lines 1780–1861, branch selected by `adiabatic`/`hydrostatic` flags —
  production Held-Suarez config takes the `adiabatic=.true.` branch, line
  1841–1847: `pt(i,j,k) = pt(i,j,k)+dtmp*pkz(i,j,k)`). Both loops are
  identical in shape (same `i,j,k` domain, same field set) to the four
  already-ported remap kernels.
- **MPI/collective involvement**: exactly 2 `mpp_global_sum` calls execute
  per production call (see Stage 8) — a 2-scalar reduction (`tesum`,
  `zsum`), not a full-field collective.
- **Major fields**: `pt`, `delp`, `delz`, `u`, `v`, `w`, `pkz`, `pk`,
  `gridstruct%area_64`, `hs` — same field family already device-resident
  from the four existing remap ports in the same call.
- **Likely current data residency**: same as the other remap stages — host
  at entry (per Prompt 91's original finding), consistent self-contained
  per-call transfer pattern.
- **Existing reusable CUDA helpers**: the per-column device-function/
  launcher pattern from `fv3_pt_delz_remap.cu`/`fv3_remap_q.cu`/
  `fv3_remap_uv` is directly reusable for both loops; no precedent yet for
  bridging a host-side scalar reduction between two device kernels within
  one port — this would be a first for this line of work.
- **Runtime branch complexity**: `consv`-sign branch, `hydrostatic`/
  `adiabatic`/`do_sat_adj`/`USE_COND` branches — comparable branch density
  to REMAP_PRESSUREVARS's own dead-branch set (Prompt 91 found most of
  these inert in the production configuration; not re-verified in full
  here, flagged for the future mapping task).

## Stage 8 — REMAP_TOTALENERGYCON Precision Re-Verification

Re-read `fv_mapz.F90:1612-1864` directly (not assumed from Prompt 90):

- **Exact `mpp_global_sum` count**: 5 call sites in source (lines 1701,
  1706, 1710, 1738, 1742) — matches Prompt 90's static count exactly.
- **But only 2 execute per actual call**: the 5 sites are split across two
  mutually-exclusive branches (`consv > consv_min` at 1701/1706/1710,
  `consv < -consv_min` at 1738/1742), and within each branch a further
  `hydrostatic` switch selects exactly one of the two `zsum`-producing
  sites. Production runs non-hydrostatic (`hydrostatic = .F.` in every
  checked `input.nml`), so at most **`tesum` (1701) + `zsum` (1710)** —
  **2 scalar reductions per call** — actually execute (assuming the
  `consv > consv_min` branch is taken, the common case for a production
  run with a positive `consv` conservation setting). This is a **precision
  refinement of Prompt 90's finding, not a contradiction**: 5 remains the
  correct static source count; 2 is the correct per-call dynamic count in
  the production configuration.
- **What is reduced**: `tesum` = global sum of `te_2d × area` (energy
  imbalance flux), `zsum` = global sum of `zsum1 × area` (pressure-weighted
  reference column integral) — both are single scalars per call, not
  per-column or 3D fields.
- **Is the reduction strictly required inside the timer?** Yes for
  correctness (the physical formulation is a genuine global energy-
  conservation constraint — `dtmp` cannot be computed without the global
  sum), but the reduction operates on a payload of exactly 2 scalars, not a
  full field — this is a categorically different communication burden than
  a halo exchange or full-field `mpp_update_domains`.
- **Is the local per-column work separable?** **PARTIAL — yes for both
  large loops individually**, but the two loops are separated by the
  reduction: loop 1 (te_2d/zsum1) → reduction (tesum/zsum, host-side scalar
  arithmetic for `dtmp`) → loop 2 (pt update using `dtmp`). Each loop alone
  is directly portable using the existing per-column CUDA pattern; the
  novel engineering element is a "kernel → tiny scalar reduction → kernel"
  sandwich, not a large-data communication problem.
- **Do reductions dominate measured wall time?** No — a 2-scalar
  `mpp_global_sum` across 6 ranks is a microsecond-scale operation; at
  150.16 ms/call, the measured cost is overwhelmingly attributable to the
  two large 3D per-column loops, which are structurally and numerically
  comparable in cost to the already-ported REMAP_T/REMAP_NH/REMAP_UV/
  REMAP_Q kernels.
- **Distinction**: **(A) GPU-clean local compute + unavoidable (but
  cheap) reduction** — not (B) fundamentally communication-bound. The
  `mpp_global_sum` presence raises engineering/design risk (a first-of-
  kind kernel-reduction-kernel pattern for this project) but does not by
  itself explain the measured cost, and is not expected to be the
  performance bottleneck of a future CUDA port.

## Stage 9 — Nsight Systems

**Not used.** Timer/source evidence (Stage 4/7/8) is sufficient to
distinguish compute-bound vs. communication-bound: the reduction payload
(2 scalars) is quantifiably tiny relative to the 150.16 ms/call measured
cost, and the surrounding loops are structurally identical in shape to four
already-successful, already-profiled CUDA ports. An `nsys` capture would
not add discriminating information beyond what direct source/timer
inspection already established. No Nsight Compute run (platform-blocked,
also out of scope regardless).

## Stage 10 — Updated Candidate Table

| Candidate | ms/call | Structure | MPI risk | Residency | CUDA reuse | Likely payoff |
|---|---|---|---|---|---|---|
| `REMAP_TOTALENERGYCON` | 150.16 | MIXED (two large POINTWISE/per-column loops bracketing one 2-scalar reduction) | LOW (2-scalar reduction only, not full-field) | Host at entry, same family as existing ports | HIGH (per-column pattern directly reusable; reduction-bridging is new but bounded) | Largest remaining non-tracer hotspot; if ported at a comparable ~3x ratio to REMAP_Q/PRESSUREVARS, ~100 ms/call recoverable (~2.6% of TOTAL) |

Only one candidate qualifies at material scale (Stage 6) — no second or
third candidate table entry exists, consistent with Prompt 90's own finding
that only two non-tracer candidates ever existed before one (PRESSUREVARS)
was closed out.

## Stage 11 — Mainline Track Reassessment

| Item | Classification | Rationale |
|---|---|---|
| `MOIST_CAPPA`/`USE_COND` | CORRECTNESS COVERAGE | Not exposed as a timer-level hotspot by this profile (production runs `nwat==0`, the gate excluding this path); remains a correctness/coverage gap, not a performance target. Unchanged from Prompt 90. |
| `a_imp <= 0.999` | CORRECTNESS COVERAGE | Same reasoning — not measurable in this profile since production always runs `a_imp=1.0` (SIM1); a coverage gap for other configurations, not a performance finding. |
| MPS/fusion operational wiring | OPERATIONAL INFRASTRUCTURE | This run already used MPS+fusion (the established best config) manually; folding it into every standing job script as the default remains a low-risk operational cleanup item, not surfaced or changed by this profile. |

None of the three were started or implemented this prompt.

## Stage 12 — Next Target Selection

**NEXT TARGET: `REMAP_TOTALENERGYCON`** (`fv_mapz.F90:1612-1864`, current
line numbers).

**WHY** (measured, factual): 150.16 ms/call, confirmed by fresh measurement
as the single largest sub-timer within `REMAPPING` (ahead of every
already-CUDA remap stage, including REMAP_NH at 127.06 ms/call) and the
largest remaining never-attempted non-tracer hotspot in the whole profile.
Stage 8's precise re-verification found the `mpp_global_sum` communication
burden is a 2-scalar-per-call reduction, not a full-field collective — the
bulk of the measured cost is attributable to two large per-column loops
structurally identical to the four already-successful CUDA ports
(PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS). This satisfies the
low-risk-of-repeating-tracer-pathology criterion (the tracer regression was
caused by host-marshaling/transfer overhead on a self-contained kernel, not
by a communication-bound design — this target's design differs precisely
in having a small bridging reduction, which must be planned for explicitly
but does not resemble the tracer failure mode) and offers the largest
realistic remaining application-level payoff of any remaining candidate.

**NEXT TASK TYPE: MAP/HARNESS** — not IMPLEMENT. This target introduces a
genuinely new pattern (kernel → global reduction → kernel) not used by any
prior successful port in this line of work; per the project's own
established precedent (Prompt 91 mapping before Prompt 92 implementation),
a dedicated mapping+harness prompt should establish the exact numerically-
active line range, the precise reduction-bridging transfer design, and
fixture-level validation coverage before any CUDA implementation begins.

## Stage 13 — Another Profile Refresh Needed?

**NO.** The profile is stable and internally consistent: the measured
REMAP_PRESSUREVARS shift (3.448x) closely matches Prompt 93's independently
-measured range (3.194–3.498x); CUDA dispatch for all four production
default paths was directly confirmed by counter (Stage 3), not inferred;
no selector/provenance instability was found. The next mapping task may
proceed directly from this profile without a further refresh.

## Summary

```
Executable sha256: b482b64703f0db7652b41310131815bef48f610d4b8ed208e7efef5d89b3ec96
C512 run: job 58619479, node warpa003, rc=0, 60/60 calls, PASS
Dispatch-verification run: job 58619656 (C32), node warpa003, rc=0, 6/6 calls, PASS
CUDA dispatch confirmed by counter: PT/DELZ 36/36, REMAP_UV 36/36, REMAP_Q 36/36, PRESSUREVARS 36/36
REMAP_PRESSUREVARS shift: 131.4 ms (P90, Fortran) -> 38.11 ms (P94, CUDA), 3.448x
New dominant non-tracer host hotspot: REMAP_TOTALENERGYCON, 150.16 ms/call
REMAP_TOTALENERGYCON global reductions per call: 2 (tesum, zsum), not 5 (5 is the static source count)
REMAP_TOTALENERGYCON likely bottleneck: COMPUTE (local per-column loops), not COMMUNICATION
Next target: REMAP_TOTALENERGYCON
Next task type: MAP/HARNESS
No source changes: YES
No commit/push: YES
```

```
POST-PRESSUREVARS C512 PROFILE REFRESHED — NEXT TARGET: REMAP_TOTALENERGYCON
```

## Prompt 95 Addendum — Stage 8 Correction (SUPERSEDED)

Prompt 95's fine-grained `fv_timing_mod` instrumentation (a nested
`TECons_RedMPI` timer wrapping only the `mpp_global_sum` call lines, inside
the coarser `TECons_Red` block) directly measured the two reductions
themselves at ~135 ms/call — **~89% of the full `REMAP_TOTALENERGYCON`
timer**, not a small fraction. Local per-column compute
(pre-+post-reduction) is only ~13% (~19.75 ms/call).

**This session's Stage 8 conclusion above ("likely bottleneck: COMPUTE,
not COMMUNICATION... `mpp_global_sum` across 6 ranks is a microsecond-scale
operation") is INCORRECT and is superseded by Prompt 95's measured
result.** Root cause: `flags=sflag` resolves to `BITWISE_EXACT_SUM`
(`flagstruct%exact_sum=1`, the live default), an order-independent
bit-reproducible FMS summation algorithm — substantially more expensive
than assumed, not a plain fast `MPI_Allreduce`. This was not visible to
Stage 8's static source analysis and required direct timing measurement to
uncover.

Prompt 95's verdict: **BLOCKED BY COMMUNICATION** (see
`p95_remap_totalenergycon_cuda_port_plan_2026-09-28.md`). A CUDA port of
only the local-compute portion remains architecturally available but is
capped at an estimated ~0.5% TOTAL ceiling — not a recommended next
implementation target. See `CURRENT_STATE.md` §9 for the current
next-target selection.
