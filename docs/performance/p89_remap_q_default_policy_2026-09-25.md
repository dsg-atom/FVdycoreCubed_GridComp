# Prompt 89 — REMAP_Q Performance Robustness and Default Policy

Date: 2026-09-25 (Stages 0-7 source edit); resumed and closed 2026-09-27
(Stages 7 rebuild-verify through 10, "Prompt 89-Resume").
Branch: `optimization/cuda-fv3-performance`
Status: **COMPLETE. CUDA REMAP_Q is now the production default** (base gate:
`kord_tr(1)==9 .and. kord_tr(2)==9 .and. nq==2 .and. kind(q)==kind(1.0d0)
.and. fv3_gpu_backend_enabled()`), with `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`
retained as the explicit Fortran escape hatch. No commit/push performed.

## 1. Scope

Bounded performance-confirmation and policy task per explicit Prompt 89
instruction. No REMAP_Q numerics, `fillz`, PT/DELZ, REMAP_UV, tracer-CUDA, or
transfer-architecture changes. No R4. No CUDA REMAP_Q default-on change until
the evidence gate below is satisfied. No `nsys`/`ncu` unless an *unexplained*
timing anomaly appears (one anomaly did appear — see §4 — but it is explained
by established precedent, so profiling was not triggered).

Verified input from Prompt 88B (restated, not re-derived): correctness PASS
at every level (fixtures, full nq=2/fill=true pipeline, sanitizer, real C32
36/36 dispatch); transfer architecture 1H2D/1D2H per REMAP_Q call, unchanged;
3-pair same-node C512 A/B performance REMAP_Q 3.23x, REMAPPING −11.98%,
FV_DYN_LOOP −3.47%, TOTAL −1.51%; current policy CUDA REMAP_Q opt-in via
`RUN_CUDA_FV3_REMAPQ_ENABLE`, production default Fortran.

## 2. Stage 0 — Binary Provenance

Reused the Prompt 88B binary without rebuild, per instruction:

```
path:   /gpfsm/dnb06/projects/p295/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
sha256: eabd1c541fba755b99ea0038dfc940bca1c2fa092539058021990b20611e5258
```

Confirmed present via `strings`/symbol check: REMAP_Q CUDA symbols
(`fv3_gpu_remap_q`), `fillz` implementation symbols, REMAP_UV CUDA
default-on (P84A), PT/DELZ CUDA path (P69). Intel MPI linkage confirmed
(`libmpi.so.12`/`libmpifort.so.12`/`libmpicxx.so.12` → Intel oneAPI 2021.13).
This exact sha256 was copied into all 10 `p89_pair{1..5}_{A,B}` experiment
directories and independently re-verified via `sha256sum` immediately before
each of the 10 runs (see `P89_PROVENANCE` log lines, §3). No rebuild was
performed or required.

## 3. Stage 1 — Same-Node Paired Test

Single Slurm allocation, node `warpa003`, one A100 (`CUDA_VISIBLE_DEVICES=0`),
job 58570647. C512, 6 MPI ranks (Intel MPI 2021.13), 1 GPU, R8, MPS ON,
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`,
REMAP_UV CUDA on (P84A default), PT/DELZ CUDA on (P69), Graph OFF — identical
across every arm except the REMAP_Q selector. 5 pairs, alternating order per
spec: A1 B1 / B2 A2 / A3 B3 / B4 A4 / A5 B5. Arm A =
`RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1` (Fortran REMAP_Q). Arm B =
`RUN_CUDA_FV3_REMAPQ_ENABLE=1` (CUDA REMAP_Q, fill=true).

All 10 arms: rc=0, 30/30 `DYN Date` steps, correct binary sha256, correct
selector env-vars (verified against the intended per-arm assignment via
`P89_SELECTOR` log lines), zero `FATAL`/`out of memory`/`Abort` matches
across the full log.

## 4. Stage 2/3 — Timings and Robustness (TOTAL)

Raw per-run timers (seconds, 30 calls/run), extracted from
`fv_timing_mod` output:

| Pair | Arm | REMAP_Q | REMAPPING | FV_DYN_LOOP | TOTAL |
|---|---|---|---|---|---|
| 1 | A (Fortran) | 3.4931 | 17.1985 | 53.6567 | 138.0322 |
| 1 | B (CUDA)    | 1.0627 | 15.3862 | 52.1145 | 136.1797 |
| 2 | A (Fortran) | 3.4934 | 17.9651 | 54.8790 | 139.2891 |
| 2 | B (CUDA)    | 1.0938 | 15.6654 | 51.9684 | 135.8078 |
| 3 | A (Fortran) | 3.4888 | 17.2792 | 54.0720 | 138.1677 |
| 3 | B (CUDA)    | 1.0764 | 14.8328 | 51.4618 | 135.7886 |
| 4 | A (Fortran) | 3.4969 | 17.8148 | 59.4817 | 144.7705 |
| 4 | B (CUDA)    | 1.0662 | 15.3201 | 51.8580 | 135.9733 |
| 5 | A (Fortran) | 3.4890 | 16.8067 | 53.5219 | 137.3800 |
| 5 | B (CUDA)    | 1.0789 | 14.8132 | 51.4272 | 135.1399 |

Per-pair percent improvement, (A−B)/A×100:

| Pair | REMAP_Q % | REMAP_Q speedup (A/B) | REMAPPING % | FV_DYN_LOOP % | TOTAL % |
|---|---|---|---|---|---|
| 1 | 69.57 | 3.287x | 10.54 | 2.87 | 1.342 |
| 2 | 68.70 | 3.194x | 12.80 | 5.31 | 2.500 |
| 3 | 69.13 | 3.241x | 14.15 | 4.83 | 1.722 |
| 4 | 69.50 | 3.280x | 14.01 | 12.82 | 6.077 |
| 5 | 69.07 | 3.234x | 11.86 | 3.91 | 1.631 |

Aggregate statistics (n=5 pairs):

| Timer | Mean % | Median % | Stddev % (sample) | Min % | Max % |
|---|---|---|---|---|---|
| REMAP_Q | 69.19 | 69.13 | 0.37 | 68.70 | 69.57 |
| REMAPPING | 12.67 | 12.80 | 1.55 | 10.54 | 14.15 |
| FV_DYN_LOOP | 5.95 | 4.83 | 3.98 | 2.87 | 12.82 |
| **TOTAL** | **2.654** | **1.722** | **1.961** | **1.342** | **6.077** |

REMAP_Q direct speedup (A/B ratio): mean 3.247x, median 3.241x, min 3.194x,
max 3.287x — tight (CoV ≈1.1%), fully consistent with P88B's 3.23x.

**TOTAL robustness (Stage 3):**
- CUDA wins TOTAL in **5 of 5** pairs (B < A every time).
- Mean improvement: **2.654%**.
- Median improvement: **1.722%**.
- Sample standard deviation: **1.961%** (population stddev 1.754%).
- Coefficient of variation: stddev/mean ≈ **0.74** (population-stddev basis
  ≈0.66) — elevated, driven almost entirely by pair 4 (see below).
- Sensitivity check excluding pair 4 (the identified outlier, not used to
  override the gate, per "do not overfit to one fastest/slowest run" —
  shown only to confirm the median is representative): mean of the
  remaining 4 pairs = 1.799%, closely matching the 5-pair median of 1.722%.
  This confirms the median is the robust central estimate and that the
  qualitative conclusion (CUDA positive, comfortably above 1.0%) does not
  depend on the outlier pair.

## 5. Stage 4 — Local-Timer Consistency and the Pair-4 Anomaly

REMAP_Q itself is tight and consistent across all 5 pairs (Fortran arm
3.4888–3.4969 s, CUDA arm 1.0627–1.0938 s, direct speedup 3.19–3.29x) —
**no noise, no regression**. REMAPPING improves in every pair (10.5–14.2%).
FV_DYN_LOOP improves in every pair (2.9–12.8%). No STOP condition per Stage
4's instruction ("if REMAP_Q itself becomes noisy or regresses: STOP") is
triggered.

**Identified anomaly**: pair 4's Fortran arm (`p89_pair4_A`) shows
`DYN_CORE`=30.6254 s, roughly 5 s above every other arm's `DYN_CORE`
(clustered tightly at 25.6–25.7 s across all other 9 arms, including the
other 4 Fortran arms). `DYN_CORE` is the acoustic-substep timer, a *sibling*
of `REMAPPING`/`REMAP_Q` inside `FV_DYN_LOOP`, not a parent or child of
either — so this inflation is architecturally unrelated to the REMAP_Q
dispatch choice. It drags up that one arm's `FV_DYN_LOOP` (59.48 s, the
highest of all 10 runs) and `TOTAL` (144.77 s, the highest of all 10 runs),
which in turn inflates pair 4's *apparent* TOTAL improvement (6.08%, the
outlier high value in the table above) — the direction of the bias favors
CUDA, but it is not attributable to CUDA REMAP_Q; it is a same-node run-to-
run variance event on the Fortran arm.

Per-checkpoint throughput inside `p89_pair4_A`'s own log (`DYN Date`
lines) shows the run was already running at roughly one-third the
throughput of a normal run (`Avg=3.7 days/day`) from its very first
progress checkpoint (00:00:36 elapsed), versus `Avg=10.0` for a normal run
(e.g. `p89_pair1_A`) at the same elapsed time — i.e., the slowdown is a
whole-run characteristic present from the start, not a single late spike.
This is consistent with the project's established, previously-documented
lesson (`CURRENT_STATE.md` §8: "Node-to-node A100 variance is large (>2x
at C512)... confirming generic node/hardware variance, not a CUDA-specific
effect") — this is the same class of transient system-level contention
(other tenant activity, thermal, I/O), not a new or unexplained phenomenon.
Per Stage 4's explicit instruction, this does **not** trigger `nsys`/`ncu`
profiling, because it is explained by established, already-documented
precedent, not an unexplained anomaly. No investigation beyond this
provenance/runtime check was performed, consistent with "no optimization
work" and bounded-debugging cost-efficiency policy.

Conclusion: the anomaly is real, is isolated to one arm's whole-run
timing (not to REMAP_Q itself, which remained tight/consistent even in
that same arm), and does not change the qualitative or quantitative
robustness conclusion (median TOTAL improvement without the outlier is
statistically indistinguishable from the median with it — 1.80% vs.
1.72%).

## 6. Stage 5 — Default Policy Gate

| # | Criterion | Threshold | Observed | Pass? |
|---|---|---|---|---|
| 1 | Correctness unchanged | — | No numerics touched this task; P88B correctness carries forward unchanged | YES |
| 2 | CUDA wins TOTAL | ≥4/5 pairs | 5/5 | YES |
| 3 | Median TOTAL improvement | ≥1.0% | 1.722% | YES |
| 4 | Mean TOTAL improvement | ≥1.0% | 2.654% (1.799% excl. outlier) | YES |
| 5 | Direct REMAP_Q speedup | clearly positive | 3.19–3.29x (mean 3.25x) | YES |
| 6 | No new memory/runtime instability | — | rc=0 all 10 arms, 30/30 steps, 0 FATAL/OOM/Abort, all sha256/selector checks correct | YES |

All 6 criteria pass. Per Stage 5's explicit rule ("If TOTAL median is
≥1.0%: DEFAULT CANDIDATE"): **Stage 5 verdict = DEFAULT CANDIDATE.**

Proceeding to Stage 6 (mandatory given the DEFAULT CANDIDATE verdict) before
any policy/source change is made.

## 7. Stage 6 — Longer-Run Confirmation (IN PROGRESS)

Same node/allocation methodology, same binary (sha256 unchanged), same
config. `JOB_SGMT` doubled from 18 min (30 calls) to 36 min (60 calls) to
reduce the relative weight of one-time startup/MPS-init overhead. Single
A→B pair (`p89_long_A`/`p89_long_B`), job 58571255, submitted
2026-09-25. Result pending — this document will be updated in place once
the job completes, per Stage 6's requirement that a longer run must
independently confirm CUDA positive in TOTAL before any policy change; if
Stage 6 contradicts the paired result, the policy remains KEEP OPT-IN with
no source change.

Stage 6 result (job 58571255, single A→B pair, 60 calls, same node/
allocation methodology, same binary as Stage 1 — sha256 unchanged, same
config): rc=0 both arms, 60/60 `DYN Date` steps, 0 errors.

| Timer | Fortran | CUDA | Improvement |
|---|---|---|---|
| REMAP_Q | 6.9731 s | 2.1062 s | 69.8% (3.31x speedup) |
| REMAPPING | 33.3909 s | 30.7150 s | 8.02% |
| FV_DYN_LOOP | 106.7738 s | 104.1198 s | 2.49% |
| **TOTAL** | **241.7600 s** | **238.3431 s** | **1.41%** |

Stage 6 independently confirms Stage 1: CUDA positive on TOTAL, consistent
REMAP_Q speedup (3.31x here vs. 3.19-3.29x/mean 3.25x in Stage 1's 5 pairs),
no regression. Stage 6 gate (`if Stage 6 confirms Stage 1: proceed to
Stage 7 default-flip") — PASSED.

## 8. Stage 7 — Source Edit (made 2026-09-25, verified 2026-09-27)

Edited `fv_mapz.F90` (nested `@fvdycore` checkout inside
`GEOSfvdycore_fortran_perf`) at the `use_cuda_remap_q` selector:

- Removed the Prompt 87 `.not. fill` selector term (already removed in
  P88B — restated here for the record, not a new change).
- CUDA REMAP_Q now defaults **on** whenever the base gate holds:
  `kord_tr(1)==9 .and. nq==2 .and. kind(q)==kind(1.0d0) .and.
  fv3_gpu_backend_enabled() .and. kord_tr(2)==9`.
- `RUN_CUDA_FV3_REMAPQ_ENABLE` is retained but is now a no-op for the
  normal path (default is already true) — kept only for backward
  compatibility with existing harness scripts that still set it.
- `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1` remains the explicit escape hatch
  back to the authoritative Fortran path, evaluated last so it always
  wins.

Numerical semantics changed: **NO** — this is a selector/default-value
change only; no numerical code path, tolerance, or algorithm was touched.

## 9. Prompt 89-Resume (2026-09-27) — Rebuild, Provenance, C32 Selector Smoke

The Stage 7 edit was made 2026-09-25 but never rebuilt/validated before the
task was paused. This session (resumed under "Prompt 89-Resume") completed
the remaining closure work.

**Diff inspection**: the existing `fv_mapz.F90` diff was read in full and
confirmed exactly scoped to the default-policy change described in §8 —
no unrelated edits, `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN` correctly preserved
as the last-evaluated override. No modification needed.

**CUDA archive provenance (unchanged, confirmed)**: `build_cuda_r8_intelmpi/
libfv3_dyn_core_gpu.a` sha256 `9e6ab02110fce13770e9cb6110d0a1301b585c54141
fa17a89bf0928e5c0ac06` — identical to the P88B value. Expected: this task
touched Fortran selector logic only, no CUDA source.

**Rebuild**: production Intel-MPI stack (`comp/gcc/12.3.0`/
`comp/intel/2024.2.0`/`mpi/impi/2021.13`/`nvidia/nvhpc-byo-compiler/24.7`,
`I_MPI_CXX=g++`). `make -j8 StandAlone_FV3_Dycore.x` recompiled exactly
`fv_mapz.F90.o`, relinked `libfvdycore.a`, relinked the executable. Zero
build errors (only pre-existing benign `libcrypto.so.1.1`/`libcrypto.so.3`
ld warnings, unrelated).

**Executable provenance**:

```
path:    /gpfsm/dnb06/projects/p295/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
mtime:   2026-09-27 07:24:47 (postdates the 2026-09-25 22:11 source edit)
size:    30,289,928 bytes
sha256:  82c1e9b61b31a2af0c599c261eeb490539c98fee4e93f1487567923c1ef9fae0
         (supersedes P88B's eabd1c54...1258, the pre-flip binary)
MPI:     Intel MPI 2021.13 confirmed via ldd (libmpi.so.12/libmpifort.so.12
         from /usr/local/intel/oneapi/2024/mpi/2021.13/lib)
symbols: fv3_gpu_remap_q/fv3_gpu_remap_uv/fv3_gpu_pt_delz_remap/fillz/
         P85_EXIT_DUMP all present via `strings`
```

**C32 selector smoke** — two fresh experiment dirs cloned from the P88B
C32 template (`p88b_stage10_c32_cuda/fv3.j`), each staged with the new
binary (sha256 verified to match after copy):

- `p89resume_c32_default` (Arm A): **no** `RUN_CUDA_FV3_REMAPQ_ENABLE`/
  `_FORCE_FORTRAN` override — tests the bare default. Job 58592939,
  rc=0, 6/6 `DYN Date` steps. `use_cuda_remap_q=T` count: **36/36**
  (6 ranks x 6 calls). Fallback: **0**. 0 NaN/Inf/FATAL/abort/ERROR.
  `slurm-58592939.out` `***** USING ****` line confirms the executed
  binary was the scratch-dir copy of the new sha256 (verified by direct
  `sha256sum` of the scratch-dir copy post-run — matches).
- `p89resume_c32_forcefortran` (Arm B): `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`
  set. Job 58592940, rc=0, 6/6 `DYN Date` steps. `use_cuda_remap_q=T`
  count: **0**. `use_cuda_remap_q=F` count: **36/36**. 0 NaN/Inf/FATAL/
  abort/ERROR. Same binary-provenance check passed.

Both arms otherwise used the identical AGCM.rc/input.nml (production
first-run config: `hord_mt/vt/tm=6`, `hord_dp=-6`, `grid_type=0`,
`hydrostatic=.F.`, `a_imp=1.0`, `kord_tm=9`, `nq=2` via `N_TRACERS=2`,
`RUN_HELD_SUAREZ=1`, `COLDSTART=1`).

**Result: both arms PASS.** The default arm dispatches CUDA REMAP_Q with
zero override needed (confirming the flip took effect); the escape hatch
correctly forces 100% Fortran dispatch with zero CUDA fallback-in-the-
wrong-direction. No performance rerun was performed (not required — see
§10).

## 10. Stage 5 — No Performance Rerun (carried forward)

Per explicit instruction, C512/paired performance was **not** rerun this
session. The default decision rests on the evidence already gathered and
independently confirmed twice:

- Stage 1: 5/5 paired C512 A/B comparisons favored CUDA on TOTAL (median
  +1.72%, mean +2.65%), REMAP_Q ~3.25x.
- Stage 6: single longer-run (60 calls) pair independently confirmed —
  TOTAL +1.41%, REMAP_Q 3.31x.

## 11. Final Report

```
Existing default-flip diff:      VALID (no fix needed)
Rebuild:                          PASS
Executable:                       /gpfsm/dnb06/projects/p295/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
sha256:                            82c1e9b61b31a2af0c599c261eeb490539c98fee4e93f1487567923c1ef9fae0
Intel MPI:                        YES (2021.13)
Default arm:                      CUDA dispatch 36/36, fallback 0 — PASS
FORCE_FORTRAN arm:                CUDA dispatch 0, fallback 36/36 — PASS
Production REMAP_Q default:       CUDA
Performance evidence reused:      5/5 paired CUDA wins; 60-call longer run PASS;
                                   TOTAL +1.41%; REMAP_Q ~3.31x
Prompt 89:                        COMPLETE
```

No commit/push performed.

REMAP_Q PERFORMANCE ROBUST — CUDA RESTORED AS PRODUCTION DEFAULT
