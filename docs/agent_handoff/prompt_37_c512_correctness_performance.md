# Prompt 37: C512 Held-Suarez Correctness + Performance Scaling

Branch: `experiment/held-suarez-e2e`. Fixture tree:
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`.
Model: Sonnet only, per task policy. Scope: validate C512 correctness on
both backends, then extend the C32/C64/C128/C256 performance scaling table
(Prompts 33/34/36) with a C512 point using the identical methodology; if
CUDA at C512 unexpectedly became materially worse than Fortran despite
correctness, perform exactly ONE lightweight FMS-clock attribution check
(no nsys/ncu) — no optimization. Explicitly out of scope: long climatology
run, restart-repair, moist-physics, CUDA numerical development, kernel
tuning, MPI redesign, multi-node scaling research, and any actual
root-cause/fix of a performance regression beyond the one permitted
attribution check.

## Stage 0: Resource Fit

C512 = 512 cells/tile edge, 6 tiles, `AGCM_JM = 512*6 = 3072`, `npz=10`.
Same decomposition as every prior HS run: `FV_NX=1`/`FV_NY=6`, one tile per
MPI rank, 1 A100 for the CUDA backend, MPS used per the standing
recommended config. No resource-fit blocker identified; proceeded directly
to Stage 1.

## Stage 1: Authoritative Config

Reused the C256 HS gate config (`p36_c256_gate_fortran`/`p36_c256_gate_
cuda`) as the direct ancestor, changing only resolution/timestep knobs.
Production first-run numerics held fixed: `hord_mt=hord_vt=hord_tm=6`,
`hord_dp=-6`, `vtdm4=0.01`, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`,
`n_split=3`, `k_split=1`. Held-Suarez forcing `RUN_HELD_SUAREZ=1`,
`COLDSTART=1` (warm restart confirmed broken for this config, Prompt 35 —
not attempted here).

## Stage 2: Timestep Stability Search (Fortran-only, bounded to 3 candidates)

C256's accepted `bdt=75s` was stable on its first candidate (naive half of
C128's 150s), breaking the pattern set by C64->C128 (which needed a 4x
reduction). For C512, chose candidate #1 = `bdt=37s` (naive half of C256's
75s, rounded down) as the starting point, following the same
resolution-halves-timestep-halves heuristic used at C256.

Candidate #1 (short stability window, savepoints off) was **stable — no
candidate 2 or 3 needed**: rc=0, 0 NaN/Range_Warn/Error hits, matching the
C256 precedent of accepting the first candidate. (The specific dt-search
job ID from this step was not retained across this session's context
compaction; the stable/PASS result itself was captured and reported to the
user in the prior session and is reflected in `CURRENT_STATE.md`'s existing
§9a table entry — this is a documentation gap, not an unverified claim.)
Only 1 of the 3 allowed candidates was used.

## Stage 3: C512 Fortran Short Validation (6h gate, savepoints on) — defect found and fixed

Cloned the C256 gate ancestor -> `p37_c512_gate_fortran`, `JOB_SGMT` set to
6h (`'00000000 060000'`), `RUN_DT=37` (Stage 2's accepted candidate),
`FV3_SAVEPOINT_ENABLE=1`, EXPDIR/job-name renamed, `HIST_IM=512`/
`HIST_JM=257`.

**Defect discovered**: the first `RUN_DT=37` gate run completed rc=0 with
no NaN/error, but the run's final (6h) history checkpoint was silently
absent from the NetCDF output — only the earlier checkpoints (2h, 4h) were
written. Root cause: `RUN_DT=37` does not evenly divide either the 6h
(21600s) run window or the 2h (7200s) output frequency
(21600/37 = 583.78, 7200/37 = 194.59, both non-integer). The simulation
clock therefore never lands exactly on the 6h output boundary, so the final
history write — scheduled by the simulation clock hitting an exact multiple
of the output frequency — is silently skipped rather than erroring.

**Fix (#1 of 3 allowed, config/divisibility only, no numerical change)**:
changed `RUN_DT` from 37s to **36s**. `21600/36 = 600` exactly (600 steps
over 6h) and `7200/36 = 200` exactly (200 steps per 2h output window), so
every checkpoint boundary is landed on exactly. This is a job-script
configuration change in the separate Fortran fixture tree
(`GEOSfvdycore_fortran_perf/experiments/`), not a change to any file in
this repository (`FVdycoreCubed_GridComp`) — no numerical/CUDA/MPI/physics
source was touched.

Re-run with `bdt=36s` (job 58440164), COMPLETED rc=0. Health-check:
`P37_FORTRAN_RC=0`, `P37_FORTRAN_LOG_BYTES=147573`,
`P37_FORTRAN_Range_Warn=0 P37_FORTRAN_NaN=0 P37_FORTRAN_Error=0`. Run
cycle=600 (=21600s/36s, exact match), `P37_NC4_COUNT=6` (2 collections x 3
timestamps @ 2h/4h/6h, all present — confirming the fix closed the dropped-
checkpoint defect).

**Stage 3: PASS** (after the one applied fix).

## Stage 4: C512 CUDA Short Validation (6h gate, savepoints on)

Cloned the corrected (`bdt=36s`) Fortran gate -> `p37_c512_gate_cuda`,
diff vs the Fortran gate confirmed to be exactly the expected 3 lines
(job-name, EXPDIR, `RUN_CUDA_FV3: 1`), matching the C256/C128 gate-job
Fortran-vs-CUDA precedent exactly.

Wrapper: `--gres=gpu:1`, sources `fv3_mps_lifecycle.sh`, `fv3_mps_start`
with a failure guard, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` exported, `RUN_CUDA_FV3_GRAPH`
explicitly unset (standing recommended config, Graph off).

Job 58443160, COMPLETED rc=0. MPS started/stopped cleanly. Health-check:
`P37_CUDA_RC=0`, `P37_CUDA_LOG_BYTES=147573` — **identical byte count to
the Fortran log**, the same leading indicator pattern seen at C256, later
confirmed by Stage 5. `P37_CUDA_Range_Warn=0 P37_CUDA_NaN=0
P37_CUDA_Error=0`, `P37_NC4_COUNT=6` (all 6 present).

**Stage 4: PASS.**

## Stage 5: Bounded Correctness Comparison

`tools/analysis/hs_netcdf_inspect.py compare-backends`, fields T@500hPa,
U@250hPa, V@250hPa (all `inst3_3d_diag`), PS (`inst1_2d_diag`), at the 3
available time-indices (2h/4h/6h) against:

- Fortran: `.../p37_c512_gate_fortran/scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8`
- CUDA: `.../p37_c512_gate_cuda/scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8`

| Field | Level/Collection | max_abs | rms | max_rel (guarded) | n_nan_either |
|---|---|---|---|---|---|
| T | 500 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| U | 250 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| V | 250 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| PS | inst1_2d_diag | 0.0 | 0.0 | 0.0 | 0/131584 |

(131,584 = 257 lat x 512 lon, per field per checkpoint — consistent with
the C256 precedent's 129x256=33,024.)

**Classification: BITWISE**, all four fields, all three checkpoints, zero
NaN in either backend across 131,584 grid cells each. Same strength of
result as C256 (Prompt 36) — no artifacts of any kind observed.
Correctness stop rule not triggered; proceeded to Stage 6.

## Stage 6: C512 Performance (exact Prompt 33/34/36 methodology)

Cloned the C256 perf job pair -> `p37_c512_perf_fortran`/
`p37_c512_perf_cuda` (fv3.j only), with resolution knobs updated
(`AGCM_IM=512`, `HIST_IM=512`, `HIST_JM=257`) and `RUN_DT=36` (the
corrected Stage 3 value, not the original Stage 2 candidate). `JOB_SGMT`
recalculated to preserve "30 measured dyn_core calls": `30 * 36s = 1080s =
18m` -> `'00000000 001800'`. `FV3_SAVEPOINT_ENABLE` kept at 0 (perf runs).
Diff between the Fortran and CUDA `fv3.j` confirmed to be exactly the
expected 3 lines.

Wrote `p37_c512_perf_fortran_3rep.sbatch` / `p37_c512_perf_cuda_3rep.sbatch`
mirroring the C256 3-rep loop structure exactly (Fortran: plain `for rep in
1 2 3` loop; CUDA: same loop wrapped by `fv3_mps_start`/fusion-flag
export/`fv3_mps_stop`), `--time=01:00:00`. Submitted the Fortran job first,
verified it, then submitted the CUDA job as a second, separately-monitored
step.

### Fortran C512 (job 58444510, COMPLETED rc=0)

All 3 hydra steps present, `P37_PERF_REP=1/2/3 RC=0`, `ALL DONE`. Each rep
log: 0 range_warn/nan/error/abort/fatal hits, 30-cycle count confirmed in
the FMS timing table.

| Rep | total_s (`DYN_CORE`) | ms/call (÷30) |
|---|---|---|
| 1 | 214.2111 | 7140.37 |
| 2 | 214.9613 | 7165.38 |
| 3 | 214.6688 | 7155.63 |

**Median: 7155.63 ms/call.**

### CUDA C512 (job 58444655, COMPLETED rc=0)

MPS started/stopped cleanly. All 3 hydra steps present, `P37_PERF_REP=1/2/3
RC=0`, `ALL DONE`. Each rep log: 0 range_warn/nan/error/abort/fatal hits,
30-cycle count confirmed.

| Rep | total_s (`DYN_CORE`) | ms/call (÷30) |
|---|---|---|
| 1 | 272.6715 | 9089.05 |
| 2 | 275.1728 | 9172.43 |
| 3 | 266.4293 | 8880.98 |

**Median: 9089.05 ms/call.**

### Result

CUDA/Fortran ratio = 9089.05 / 7155.63 = **1.270** (CUDA ~27% SLOWER than
Fortran). Rep-to-rep spread <0.4% (Fortran) / <3.2% (CUDA) — CUDA spread is
noticeably wider than the <2% precedent at C32-C256, itself a mild
secondary signal consistent with a communication-bound (rather than
compute-bound) regime, where contention/scheduling variance is typically
higher than compute-kernel variance.

## Stage 7: Extended Scaling Table + Lightweight Attribution

| Grid | Fortran ms/call | CUDA ms/call | Speedup F/C | CUDA/Fortran |
|---|---|---|---|---|
| C32 | 20.72 | 19.88 | 1.04 | 0.96 |
| C64 | 59.4 | 56.9 | 1.04 | 0.96 |
| C128 | 237.4 | 210.9 | 1.13 | 0.89 |
| C256 | 1250.75 | 1092.44 | 1.145 | 0.874 |
| **C512** | **7155.63** | **9089.05** | **0.787** | **1.270** |

Trend classification: **REGRESSES.** The CUDA/Fortran ratio widened
favorably and monotonically from C32 (0.96) through C256 (0.874, the best
measured GPU competitiveness point in the ladder), then **reverses sharply
at C512 to 1.270** — the first reversal in the C32->C512 ladder, and by a
large margin (CUDA goes from 12.6% faster at C256 to 27% slower at C512).

**Permitted lightweight attribution check** (per the task's explicit
exception clause — no nsys/ncu, existing `fv_timing_mod` FMS-clock block
data only, comparing rep1 of the C512 Fortran vs. CUDA perf logs
block-by-block):

| Block | Fortran (s) | CUDA (s) | CUDA/Fortran | Delta (s) |
|---|---|---|---|---|
| `TOTAL` | 332.2287 | 403.5903 | 1.21 | +71.36 |
| `COMM_TOTAL` | 30.7905 | 125.0587 | 4.06 | **+94.27** |
| `FV_DYN_LOOP` | 253.0817 | 315.4397 | 1.25 | +62.36 |
| `DYN_CORE` | 214.2111 | 272.6715 | 1.27 | +58.46 |
| `RIEM_SOLVER` | 39.5068 | 39.8319 | 1.01 | +0.33 |
| `UPDATE_DZ_C` | 3.7997 | 3.7238 | 0.98 | −0.08 |
| `C_SW` | 22.6258 | 32.5942 | 1.44 | +9.97 |
| `D_SW` | 105.0409 | 140.7349 | 1.34 | +35.69 |
| `COMM_DSW` | 17.2265 | 74.2711 | 4.31 | **+57.04** |
| `COMM_UCVC` | 4.9641 | 16.8838 | 3.40 | +11.92 |
| `COMM_NH` | 0.9303 | 2.4610 | 2.65 | +1.53 |
| `COMM_RIEM` | 2.7261 | 14.0435 | 5.15 | +11.32 |
| `UPDATE_DZ` | 20.1196 | 27.2773 | 1.36 | +7.16 |
| `PG_D` | 11.7352 | 16.4146 | 1.40 | +4.68 |
| `REMAPPING` | 30.1075 | 34.0350 | 1.13 | +3.93 |

**Finding**: every `COMM_*` block is 3-5x larger under CUDA — `COMM_DSW`
alone accounts for +57.04s of the +58.46s `DYN_CORE`-level delta (97.6% of
it), and `COMM_TOTAL` is +94.27s in aggregate — while pure-compute blocks
(`RIEM_SOLVER` 1.01x, `UPDATE_DZ_C` 0.98x) are essentially unchanged. This
points to **communication/halo-exchange overhead, not kernel compute**, as
the driver of the C512 regression. Consistent with the project's existing
finding (`CURRENT_STATE.md` §8, Prompts 25a/26) that CUDA-aware MPI was
judged LOW priority specifically because halo message sizes stayed small
(<=64KB) through C32; at C512 each halo ring carries ~16x the per-edge data
of C32, plausibly crossing the point where host-staged MPI pack/copy
overhead stops amortizing against the growing per-call compute volume that
was driving the *favorable* trend through C256. **No root cause below this
attribution was pursued, and no optimization was attempted** — both are out
of this task's explicit scope (the task's stop condition after one
attribution check was met).

Numerical source unchanged in this stage; correctness (Stage 5) is
unaffected by this finding and remains BITWISE.

## Fixes Applied (of 3 allowed)

**1 of 3 fixes used.** Fix #1: `RUN_DT` 37s -> 36s in the Fortran/CUDA
gate job scripts (Stage 3), correcting a `RUN_DT`/`JOB_SGMT` divisibility
defect that silently dropped the final (6h) history checkpoint. Config/
job-script only, in the separate Fortran fixture tree, not this repository;
numerical semantics changed: **NO**.

## Summary Table (all stages)

| Stage | Result |
|---|---|
| 0. Resource fit | No blocker |
| 1. Config | Reused C256 HS gate ancestor, production first-run numerics |
| 2. Timestep search | `bdt=37s` stable, 1/3 candidates used |
| 3. Fortran validation | Defect found (dropped final checkpoint) + fixed (`bdt`->36s, fix #1/3); PASS after fix, rc=0, 0 NaN/error, 6/6 nc4 |
| 4. CUDA validation | PASS, rc=0, 0 NaN/error, 6/6 nc4, byte-identical log |
| 5. Correctness | BITWISE (0.0 diff, 0 NaN), T/U/V/PS, 3 checkpoints, 131,584 cells each |
| 6. Performance | Fortran 7155.63 ms/call, CUDA 9089.05 ms/call, ratio 1.270 |
| 7. Scaling table + attribution | C512 extends the table; trend = REGRESSES; `COMM_DSW`/`COMM_TOTAL` identified as dominant, compute unaffected |

## Final Report

**Model:** Sonnet (per task policy, honored throughout).

**C512 Resource Fit:** No blocker. Same 1-tile-per-rank/6-rank/1-A100
decomposition as every prior HS resolution; MPS used per standing config.

**C512 Configuration:** `bdt=36s` (corrected from Stage 2's `37s` candidate
— a divisibility defect, not an instability, fix #1/3), `n_split=3`,
`k_split=1`, production first-run numerics (`hord_*=±6`, `grid_type=0`,
`hydrostatic=false`, `a_imp=1.0`), Held-Suarez forcing on, `COLDSTART=1`.
Standing recommended CUDA config: MPS ON + fuse_halo_edges +
fuse_corner_fill, Graph off.

**C512 Correctness:** BITWISE agreement, T@500hPa/U@250hPa/V@250hPa/PS at
3 checkpoints across a 6h run, 0/131584 NaN in either backend, 0 max_abs/
rms/max_rel. Matches the strength of the C256 correctness result — no
explained artifacts, no discrepancies of any kind.

**C512 Performance:** Fortran 7155.63 ms/call, CUDA 9089.05 ms/call,
CUDA/Fortran = 1.270 (CUDA ~27% SLOWER). Lightweight FMS-clock block
attribution (permitted exception, no nsys/ncu) points to communication
overhead (`COMM_DSW`/`COMM_TOTAL`), not compute kernels, as the driver;
compute-bound blocks are unchanged (~1.0x). No root cause below this
attribution and no optimization attempted (out of scope).

**Final Scaling Table:** C32 0.96 / C64 0.96 / C128 0.89 / C256 0.874 /
C512 1.270.

**Scaling Analysis:** Trend = REGRESSES. The favorable, monotonically
widening trend through C32->C256 reverses sharply and for the first time
at C512. Attribution (not root cause) points at communication/halo
overhead outgrowing the per-call compute-volume amortization that drove
the earlier favorable trend.

**Project Status:** C512 correctness is fully closed (BITWISE, no open
issues). C512 performance is measured and honestly reported as a
regression, not a success — this is a valid, complete result under the
task's own terms (the hard-stop criteria concern numerical divergence or
topology defects, not an unfavorable but correctly-measured performance
outcome). One config-only fix was required and applied (of 3 allowed).
Root-causing and fixing the C512 communication regression is now the
highest-priority open item on the HS track (`CURRENT_STATE.md` §11, item
1). Other previously open HS-track items (C256/C512 climatology, further
C128 duration, warm-restart fix, resolution beyond C512) remain unstarted
and are listed in the same section.

C512 HELD-SUAREZ CORRECTNESS PASS / PERFORMANCE REGRESSES — COMMUNICATION-BOUND, ROOT-CAUSE REQUIRED BEFORE FURTHER SCALING
