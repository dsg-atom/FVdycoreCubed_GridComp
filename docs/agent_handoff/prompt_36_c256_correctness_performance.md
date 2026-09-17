# Prompt 36: C256 Held-Suarez Correctness + Performance Scaling

Branch: `experiment/held-suarez-e2e`. Fixture tree:
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`.
Model: Sonnet only, per task policy. Scope: validate C256 correctness on
both backends, then extend the C32/C64/C128 performance scaling table
(Prompts 33/34) with a C256 point using the identical methodology.
Explicitly out of scope: long climatology run, restart-repair,
moist-physics, CUDA numerical development, kernel tuning, MPI redesign,
multi-node scaling research. Zero of the allowed 3 substantial fixes were
needed — every stage passed cleanly on its first attempt.

## Stage 0: Resource Fit

C256 = 256 cells/tile edge, 6 tiles, `AGCM_JM = 256*6 = 1536`, `npz=10`.
Same decomposition as every prior HS run: `FV_NX=1`/`FV_NY=6`, one tile per
MPI rank, 1 A100 for the CUDA backend (MPS not required for a single-GPU
6-rank job but used per the standing recommended config anyway). No
resource-fit blocker identified; proceeded directly to Stage 1.

## Stage 1: Authoritative Config

Reused the C128 HS gate config (`p33_c128_gate_fortran`/`p33_c128_gate_
cuda`) as the direct ancestor, changing only resolution/timestep knobs.
Production first-run numerics held fixed: `hord_mt=hord_vt=hord_tm=6`,
`hord_dp=-6`, `vtdm4=0.01`, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`,
`n_split=3`, `k_split=1`. Held-Suarez forcing `RUN_HELD_SUAREZ=1`,
`COLDSTART=1` (warm restart confirmed broken for this config, Prompt 35 —
not attempted here).

## Stage 2: Timestep Stability Search (Fortran-only, bounded to 3 candidates)

C64->C128 (Prompt 33) needed a 4x reduction (150s->37.5s... actually the
accepted C128 value was 150s itself scaled from C64's larger value; the key
precedent is that the *naive half-step* candidate was CFL-unstable and a
smaller step was required). For C256, chose candidate #1 = 75s (naive half
of C128's accepted 150s) as the starting point, matching the general
resolution-halves-timestep-halves CFL heuristic, while flagging explicitly
that the C64->C128 transition had violated this heuristic once already.

Candidate #1 (`p36_c256_dt1_fortran`, `RUN_DT=75`, 3h/144-step stability
window, savepoints off): job 58436126, COMPLETED rc=0 in 00:04:47. Model
Throughput 39.154 days/day. Health-check: `P36_FORTRAN_RC=0`,
`P36_FORTRAN_Range_Warn=0 P36_FORTRAN_NaN=0 P36_FORTRAN_Error=0`. Max Real
Mem 5.84G. **Stable — no candidate 2 or 3 needed.** Only 1 of the 3 allowed
candidates was used, unlike the C64->C128 case.

## Stage 3: C256 Fortran Short Validation (24h gate, savepoints on)

Cloned `p36_c256_dt1_fortran` -> `p36_c256_gate_fortran` (rsync + sed):
`JOB_SGMT` 3h->24h (`'00000001 000000'`), `FV3_SAVEPOINT_ENABLE` 0->1,
EXPDIR/job-name renamed. `RUN_DT=75` unchanged from the accepted Stage 2
candidate.

Job 58436426, COMPLETED rc=0 in 01:07:44. Model Throughput 21.315 days/day.
Run cycle=1152 (=24h*3600s/75s, exact match, no early termination),
DYN=2304. Health-check: `P36_FORTRAN_RC=0`, `P36_FORTRAN_LOG_BYTES=259813`,
`P36_FORTRAN_Range_Warn=0 P36_FORTRAN_NaN=0 P36_FORTRAN_Error=0`,
`P36_NC4_COUNT=8` (2 collections x 4 timestamps @ 6/12/18/24h, all present).
Max Real Mem 5.99G.

`hs_netcdf_inspect.py describe`: dims `{time:4, lat:129, lon:256}`
(inst1_2d_diag), `{time:4, lat:129, lon:256, lev:25}` (inst3_3d_diag), 25
`lev` values 1000->100hPa — structurally correct, scaled from C128.
`global-mean-t --level-hpa 500 --hours-per-step 6`: min=260.21364
max=260.23308 first=260.22949 last=260.22015 (flat, expected for a fresh
24h cold-start). `jet-metrics` (default thresholds): NH lat=45.00
lev=250.0hPa U=35.150 m/s, SH lat=-45.00 lev=250.0hPa U=35.150 m/s —
symmetric and physically sane for a 24h HS spin-up.

**Stage 3: PASS.**

## Stage 4: C256 CUDA Short Validation (24h gate, savepoints on)

Cloned `p36_c256_gate_fortran` -> `p36_c256_gate_cuda` (rsync + sed +
targeted Edit for the `RUN_CUDA_FV3: 1` insertion). Diff vs the Fortran
gate confirmed to be exactly 3 lines (job-name, EXPDIR, `RUN_CUDA_FV3: 1`),
matching the p33 C128 gate-job Fortran-vs-CUDA precedent exactly — no other
source or config drift.

Wrapper: `--gres=gpu:1`, sources `fv3_mps_lifecycle.sh`, `fv3_mps_start`
with a failure guard (echoes `MPS_START_FAILED` + exit 1 on failure),
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`
exported, `RUN_CUDA_FV3_GRAPH` explicitly unset (standing recommended
config, Graph off).

Job 58437185, COMPLETED rc=0 in 01:01:57. MPS started
(`FV3_MPS_START: pipe=/tmp/fv3mps_jli30_58437185/pipe status=''`) and
stopped (`FV3_MPS_STOP: server_list_before_quit=`) cleanly. Health-check:
`P36_CUDA_RC=0`, `P36_CUDA_LOG_BYTES=259813` — **identical byte count to
the Fortran log**, a strong leading indicator later confirmed by Stage 5.
`P36_CUDA_Range_Warn=0 P36_CUDA_NaN=0 P36_CUDA_Error=0`,
`P36_NC4_COUNT=8` (all 8 present).

**Stage 4: PASS.**

## Stage 5: Bounded Correctness Comparison

`tools/analysis/hs_netcdf_inspect.py compare-backends`, fields T@500hPa,
U@250hPa, V@250hPa (all `inst3_3d_diag`), PS (`inst1_2d_diag`), at
`--time-indices 0,1,2,3` (6h/12h/18h/24h) against:

- Fortran: `.../p36_c256_gate_fortran/scratch_c256_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8`
- CUDA: `.../p36_c256_gate_cuda/scratch_c256_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8`

| Field | Level/Collection | max_abs | rms | max_rel (guarded) | n_nan_either |
|---|---|---|---|---|---|
| T | 500 hPa | 0.0 | 0.0 | 0.0 | 0/33024 |
| U | 250 hPa | 0.0 | 0.0 | 0.0 | 0/33024 |
| V | 250 hPa | 0.0 | 0.0 | 0.0 | 0/33024 |
| PS | inst1_2d_diag | 0.0 | 0.0 | 0.0 | 0/33024 |

**Classification: BITWISE**, all four fields, all four checkpoints, zero
NaN in either backend across 33,024 grid cells each. Stronger than several
historical c32/c64 rows in `CURRENT_STATE.md` §5 (which had small explained
artifacts) — no artifacts of any kind observed here. Correctness stop rule
not triggered; proceeded to Stage 6.

## Stage 6: C256 Performance (exact Prompt 33/34 methodology)

Cloned `p33_c128_perf_fortran`/`p33_c128_perf_cuda` (fv3.j only, via rsync
excluding `savepoints`/`scratch_*`/`perf_logs`) -> `p36_c256_perf_fortran`/
`p36_c256_perf_cuda`. Edits: `RUN_DT 150->75`, `AGCM_IM 128->256`,
`HIST_IM 128->256`, `HIST_JM 65->129`, EXPDIR/job-name renamed,
`FV3_SAVEPOINT_ENABLE` kept at 0 (perf runs, savepoints off, matching the
C128 perf-job convention).

**Critical recalculation, not a straight copy**: C128's perf `JOB_SGMT`
(`'00000000 011500'` = 1h15m = 4500s) divided by its `RUN_DT=150` gives
exactly 30 outer steps = 30 dyn_core calls/rep, confirmed by the cycle-count
column in the C128 `perf_logs/rep1.log`'s FMS timing table (trailing `30`).
To preserve "30 measured dyn_core calls" at C256's `RUN_DT=75`, `JOB_SGMT`
was recalculated as `30 * 75s = 2250s = 37m30s` ->
`'00000000 003730'` — this value does NOT appear anywhere in the C128
job and was derived fresh for this stage.

Also caught and fixed: the blanket `PC128x65-DC`->`PC256x129-DC` grid-label
sed did not touch the hardcoded `IM_WORLD: 128`/`JM_WORLD: 65` numeric
literals inside the `GRID_LABELS` block (they are plain literals at that
spot, not `${AGCM_IM}`/`${HIST_IM}` variable references) — fixed with a
second targeted sed in both `fv3.j` files. Low functional risk since perf
runs (37m30s duration vs. 6h output frequency) produce no NetCDF output at
all, but corrected for consistency with the gate jobs (which already had
this correct).

Diff between the Fortran and CUDA `fv3.j` confirmed to be exactly the
expected 3 lines (job-name, EXPDIR, `RUN_CUDA_FV3: 1`).

Wrote `p36_c256_perf_fortran_3rep.sbatch` / `p36_c256_perf_cuda_3rep.sbatch`
mirroring the p33 3-rep loop structure exactly (Fortran: plain `for rep in
1 2 3` loop; CUDA: same loop wrapped by `fv3_mps_start`/fusion-flag
export/`fv3_mps_stop`), `--time=00:45:00` (budgeted up from C128's 30 min
given C256's larger per-call cost).

Per the explicit instruction to sequence rather than fire all 6 reps
unattended, submitted only the Fortran job first, verified it, then
submitted the CUDA job as a second, separately-monitored step.

### Fortran C256 (job 58437887, COMPLETED rc=0, 00:03:27 elapsed)

All 3 hydra steps present, `P36_PERF_REP=1/2/3 RC=0`, `ALL DONE`. Each
rep log: 0 range_warn/nan/error/abort/fatal hits, 30-cycle count confirmed
in the FMS timing table.

| Rep | total_s (`DYN_CORE`) | ms/call (÷30) |
|---|---|---|
| 1 | 37.5260 | 1250.87 |
| 2 | 37.5224 | 1250.75 |
| 3 | 37.4780 | 1249.27 |

**Median: 1250.75 ms/call.**

### CUDA C256 (job 58438138, COMPLETED rc=0, 00:03:20 elapsed)

MPS started (`FV3_MPS_START: pipe=/tmp/fv3mps_jli30_58438138/pipe
status=''`) and stopped (`FV3_MPS_STOP: server_list_before_quit=`)
cleanly. All 3 hydra steps present, `P36_PERF_REP=1/2/3 RC=0`, `ALL DONE`.
Each rep log: 0 range_warn/nan/error/abort/fatal hits, 30-cycle count
confirmed.

| Rep | total_s (`DYN_CORE`) | ms/call (÷30) |
|---|---|---|
| 1 | 32.7963 | 1093.21 |
| 2 | 32.6019 | 1086.73 |
| 3 | 32.7732 | 1092.44 |

**Median: 1092.44 ms/call.**

### Result

CUDA/Fortran ratio = 1092.44 / 1250.75 = **0.874** (Fortran/CUDA speedup =
1.145x). Rep-to-rep spread <0.2% (Fortran) / <0.6% (CUDA) — well within the
<2% precedent from C32/C64/C128.

## Stage 7: Extended Scaling Table

| Grid | Fortran ms/call | CUDA ms/call | Speedup F/C | CUDA/Fortran |
|---|---|---|---|---|
| C32 | 20.72 | 19.88 | 1.04 | 0.96 |
| C64 | 59.4 | 56.9 | 1.04 | 0.96 |
| C128 | 237.4 | 210.9 | 1.13 | 0.89 |
| **C256** | **1250.75** | **1092.44** | **1.145** | **0.874** |

Trend classification: **WIDENS.** CUDA's advantage over Fortran grows
monotonically across all four measured points (0.96 -> 0.96 -> 0.89 ->
0.874), not just partway through the ladder as Prompt 33's original
(later-corrected) reading suggested. C256 is now the best measured GPU
competitiveness point, superseding C128. This supports (does not yet prove
asymptotically, only 4 points measured) the general hypothesis that fixed
per-call host/MPI/launch overhead amortizes better as grid-point count
(and therefore per-call GPU compute volume) grows — consistent with the
root-cause chain in `CURRENT_STATE.md` §8 (MPS removes the context-
contention bottleneck; remaining overhead is largely fixed per-call
launch/sync cost that shrinks in relative terms as compute volume grows).
No new profiling was performed this prompt (no unexpected regression
occurred, so none was warranted per task scope).

## Fixes Applied (of 3 allowed)

**Zero substantial fixes were needed.** One minor, non-numerical
correction was made (the `IM_WORLD`/`JM_WORLD` grid-label literal sed gap
in Stage 6) — cosmetic/metadata only, no effect on any measured result,
not counted against the 3-fix budget given its triviality and zero
functional impact (perf runs produce no NetCDF output at this duration).

## Summary Table (all stages)

| Stage | Result |
|---|---|
| 0. Resource fit | No blocker |
| 1. Config | Reused C128 HS gate ancestor, production first-run numerics |
| 2. Timestep search | `bdt=75s` stable, 1/3 candidates used |
| 3. Fortran validation | PASS, rc=0, 0 NaN/error, 8/8 nc4 |
| 4. CUDA validation | PASS, rc=0, 0 NaN/error, 8/8 nc4, byte-identical log |
| 5. Correctness | BITWISE (0.0 diff, 0 NaN), T/U/V/PS, 4 checkpoints |
| 6. Performance | Fortran 1250.75 ms/call, CUDA 1092.44 ms/call, ratio 0.874 |
| 7. Scaling table | C256 extends the table; trend = WIDENS |

## Final Report

**Model:** Sonnet (per task policy, honored throughout).

**C256 Resource Fit:** No blocker. Same 1-tile-per-rank/6-rank/1-A100
decomposition as every prior HS resolution; MPS used per standing config.

**C256 Configuration:** `bdt=75s`, `n_split=3`, `k_split=1`, production
first-run numerics (`hord_*=±6`, `grid_type=0`, `hydrostatic=false`,
`a_imp=1.0`), Held-Suarez forcing on, `COLDSTART=1`. Standing recommended
CUDA config: MPS ON + fuse_halo_edges + fuse_corner_fill, Graph off.

**C256 Correctness:** BITWISE agreement, T@500hPa/U@250hPa/V@250hPa/PS at
4 checkpoints across a 24h run, 0/33024 NaN in either backend, 0 max_abs/
rms/max_rel. Strongest correctness result in the HS ladder to date (no
explained artifacts, unlike some c32/c64 production-dyncore rows).

**C256 Performance:** Fortran 1250.75 ms/call, CUDA 1092.44 ms/call,
CUDA/Fortran = 0.874 (CUDA 1.145x faster).

**Final Scaling Table:** C32 0.96 / C64 0.96 / C128 0.89 / C256 0.874.

**Scaling Analysis:** Trend = WIDENS. Best speedup measured at C256.
Supports the resolution-amortization hypothesis from Prompts 33-35; not
yet asymptotically proven (4 points).

**Project Status:** C256 correctness and performance both closed with no
open issues, no fixes required, no scope deviation. HS track's higher-
resolution follow-on (flagged open at the end of Prompt 35) is now closed.
Remaining open HS-track items (C256 climatology/longer duration, C128
further-duration, warm-restart fix, resolution beyond C256) are unstarted
and listed in `CURRENT_STATE.md` §11.

C256 HELD-SUAREZ CORRECTNESS + PERFORMANCE PASS — READY FOR C512 OR HIGH-RES SCIENCE STUDY
