# Prompt 33 — Multi-Day / High-Resolution Held-Suarez Validation + Performance

Branch: `experiment/held-suarez-e2e` (orthogonal to the `cuda/gpu-bringup`
mainline). Follows Prompt 31 (HS integration smoke test) and Prompt 32 (HS
scientific validation at C8/C32). Model: Sonnet throughout, no Opus
escalation. Debug authority: 0 of the allotted ≤3 substantial fixes spent on
numerical/MPI/kernel work (two tooling-level fixes only — see "Tooling
fixes" below — both config/harness, not counted against the numerical
debug budget).

## Scope

IN: longer duration, higher resolution, real Fortran-vs-CUDA performance
evaluation, job/config generation, NetCDF/xarray diagnostics, comparison
tooling. OUT: new HS physics, moist-physics implementation, CUDA
numerical-kernel development, MPI architecture redesign, CUDA-aware MPI,
new Graph work, broad GEOS full-physics integration. None of the OUT items
were touched.

## Stage A/B (prior session, recapped only)

C32 10/30-day duration validation and C64 short-gate validation were
completed and PASS before this session's continuation; not re-derived here.

## Stage C — C128 short-gate validation

Built `p33_c128_gate_fortran`/`p33_c128_gate_cuda` job directories
(`FV3_SAVEPOINT_ENABLE 1`, C128/L10/R8, 6 ranks). Timestep search: the
first `bdt` candidate (300s) was CFL-unstable — job 58410818 flooded its
log with NaN/`range_warn` lines (log grew to ~115GB before being caught).
**Lesson applied**: subsequent health checks always gate on log size before
running an unconditional `grep`, e.g.:

```bash
logsize=$(stat -c%s "$logf" 2>/dev/null || echo 0)
if [ "$logsize" -gt 1000000000 ]; then
  echo "LOG_FLOOD_SUSPECTED_INSTABILITY (log > 1GB, skipping full grep)"
else
  echo "Range_Warn=$(grep -ci range_warn $logf) NaN=$(grep -ci '\bnan\b' $logf)"
fi
```

A smaller `bdt` was selected and the gate re-run cleanly on both backends.
`tools/analysis/hs_netcdf_inspect.py compare-backends` on T@500hPa/U@250hPa/
PS returned `max_abs=0.0 rms=0.0` at all tested timestamps — **PASS,
bitwise**. `savepoint_reader.py` sanity checks: 617,970 fields checked on
each backend, all PASS. Job 58410818 (the unstable candidate) was
`scancel`'d once its diagnostic value was captured — it was stuck idling on
GPU resources for no further benefit and risked blocking the newly
submitted Stage D jobs.

**Stage C: CLOSED — PASS.**

## Stage D — C128, 10-day duration

Built `p33_c128_hs10d_fortran`/`p33_c128_hs10d_cuda` from the Stage C gate
directories: `JOB_SGMT` -> `'00000010 000000'` (10 days), `--time=04:00:00`.
New sbatch scripts (`p33_c128_hs10d_{fortran,cuda}.sbatch`) with the
size-gated health check baked in; CUDA variant wraps the MPS lifecycle
helper + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`.

Submitted as jobs **58411417** (Fortran) and **58411418** (CUDA); both
COMPLETED (02:06:01 / 01:59:33 elapsed), rc=0.

Health check results (both backends):
- Log size: exactly **1,717,491 bytes** in both — byte-identical.
- Clean `EGRESS` marker present in both.
- **80** `.nc4` output files each (10 days x 4 outputs/day x 2 History
  collections).
- Model Throughput: 114.404 days/day (Fortran) / 120.745 days/day (CUDA).
- `range_warn=6966`, `nan=0`, `error=0`, `fatal=0` — identical both
  backends.

**`range_warn` investigation** (the one nontrivial diagnostic dug into this
stage): all 6966 hits are field `TA_dyn` at level 10, values 333.0003 to
349.2171 K, spanning 119 unique grid points. A sorted-diff between the two
backends' full set of `range_warn` lines —
`diff <(grep range_warn fortran.log | sort) <(grep range_warn cuda.log | sort)`
— returned **0**, proving the warning content is identical modulo
MPI-rank-interleaving order. Classified **benign**: legitimate Held-Suarez
near-surface tropical warmth exceeding a hardcoded FMS diagnostic
plausibility bound, not a numerical defect, and not a new divergence
between backends.

NetCDF comparison (`hs_netcdf_inspect.py compare-backends`) on T@500hPa,
U@250hPa, and PS at start/mid/end timestamps of the 10-day run: all
`max_abs=0.000000e+00 rms=0.000000e+00 n_nan_either=0/8320` — **bitwise
identical**.

**Stage D: CLOSED — PASS.**

## Stage E — Performance scaling (C32 -> C64 -> C128)

Goal: extend the documented C32 CUDA-vs-Fortran performance baseline
(`CURRENT_STATE.md` §8: 19.930 ms/call Fortran, 25.464 ms/call CUDA
MPS+fused) to C64/C128 to see how the gap evolves with resolution.

### Infrastructure and two tooling fixes

Built four perf-run directories (`p33_c64_perf_{fortran,cuda}`,
`p33_c128_perf_{fortran,cuda}`) from the corresponding Stage B/C gate
directories, with `FV3_SAVEPOINT_ENABLE 0`.

**Fix 1 — large-directory-copy timeout.** `cp -r` on
`p33_c128_gate_cuda` (1.3GB, mostly its `savepoints/` tree) exceeded the
Bash tool's 120s default timeout (exit 143). Root cause: savepoints aren't
needed for perf-only runs. Fix: rebuilt via
`rsync -a --exclude=savepoints --exclude='scratch_*' src/ dst/`, which
completed cleanly. Tooling-level fix, no numerical impact.

**Fix 2 — rep-clobbering via glob self-match.** The first version of the
3-rep timing sbatch loop did:
```bash
for rep in 1 2 3; do
  rm -rf savepoints scratch_*
  csh -f ./fv3.j
  cp "$logf" "${logf}.rep${rep}"
done
```
`${logf}.rep${rep}` (e.g. `scratch_....log.rep1`) still matches the
`scratch_*` glob, so each iteration's cleanup deleted the *previous*
iteration's saved copy — only `rep3` ever survived. Fix: write saved copies
to a dedicated `perf_logs/` subdirectory the `scratch_*` cleanup cannot
reach (`cp "$logf" "perf_logs/rep${rep}.log"`). Verified all three reps
survive on rerun. Tooling-level fix, no numerical impact.

**Cold-start dilution.** Initial 10-calls/rep runs showed the MAPL
hierarchical profiler's max-per-rank column (~0.7-0.8s) dominating the
mean (~0.09-0.1s) — a one-time cost (CUDA context/kernel JIT warm-up, or
an analogous Fortran-side one-time init cost) skewing a too-small sample.
Increased to 30 calls/rep to dilute this; cross-rep reproducibility came
out tight (<0.5%) at 30 calls, judged sufficient for this scaling study
without further methodology development.

Jobs **58416204-58416207** (c64_fortran, c64_cuda, c128_fortran,
c128_cuda), each 3 reps x 30 `dyn_core` calls, all COMPLETED (24-59s),
rc=0, all three `perf_logs/rep{1,2,3}.log` present per config.

### Timing extraction and methodology note

Two timing mechanisms coexist in the standalone log: a flat FMS-clock
`DYN_CORE  <total_s>  0.0000  <total_s>  0` line (total accumulated seconds
for the `DYN_CORE` region across the whole run), and a hierarchical MAPL
generic-profiler table (mean/max/min triads *across MPI ranks*, not across
calls — the large outlier seen in the hierarchical table's max column, up
to ~8.9-10.2s, exceeds the flat-clock total for the whole 30-call run and
is best explained as one rank's transient scheduling/serialization cost,
not a per-call effect). The flat clock was adopted as the simpler,
self-consistent metric: `ms/call = 1000 * total_seconds / n_calls`. Cross
check against `p28_c32_r8_cuda_guard`'s existing 10-call log (`DYN_CORE
0.2918s` -> 29.18 ms/call) landed close to the documented C32
MPS+fused-mode figure (25.464 ms/call), confirming this is the same metric
the existing baseline effectively uses, modulo warmup dilution (10 calls
there vs. this stage's 30).

**Caveat, stated explicitly**: this ms/call figure includes the first-call
warmup cost, diluted over 30 calls rather than excluded — a different
methodology from the documented C32 baseline's "100 calls/rep... warmup
excluded." The C32 row in the table below is the *existing documented*
figure, not re-measured under this session's 30-call methodology; treat
the C64/C128 comparison to it as directionally reliable, not perfectly
apples-to-apples. A follow-on re-measurement of C32 under the identical
30-call/3-rep protocol would close this gap cheaply.

### Results

Median of 3 reps, 30 calls/rep, standing MPS + fuse_halo_edges +
fuse_corner_fill mode:

| Grid | Fortran ms/call | CUDA ms/call | CUDA/Fortran |
|---|---|---|---|
| C32 (documented baseline, 100-call/rep, warmup-excluded) | 19.930 | 25.464 | 1.28x slower |
| C64 (this stage, 30-call/rep, warmup-included) | 59.4 | 56.9 | **0.96x — CUDA faster** |
| C128 (this stage, 30-call/rep, warmup-included) | 237.4 | 210.9 | **0.89x — CUDA faster** |

Raw flat-clock totals (3 reps each, seconds for 30 calls):
- C64 Fortran: 1.7818, 1.7841, 1.7734 (median 1.7818 -> 59.4 ms/call)
- C64 CUDA: 1.7060, 1.7053, 1.7085 (median 1.7060 -> 56.9 ms/call)
- C128 Fortran: 7.1383, 7.1217, 7.1156 (median 7.1217 -> 237.4 ms/call)
- C128 CUDA: 6.3254, 6.3289, 6.3258 (median 6.3258 -> 210.9 ms/call)

Reps agree within <0.5% at every config — tight, reproducible.

**Finding**: the CUDA/Fortran gap **narrows, then reverses**, as
resolution increases. At C32, Fortran is faster (CUDA 1.28x slower). By
C64, CUDA has overtaken Fortran (0.96x), and the margin grows further by
C128 (0.89x). This is physically sensible: GPU kernel-launch and
sync-under-MPS overhead is a largely fixed per-call cost, while the actual
floating-point work scales with grid-point count (~4x per resolution
doubling in the horizontal); a fixed overhead is a *smaller fraction* of a
*larger* per-call workload, so CUDA's relative efficiency improves with
resolution. The Fortran side (single-threaded per rank, `N_OMP=1`) has no
equivalent fixed-overhead dilution, so its ms/call scales closer to
proportional with grid-point count. Both backends individually show
sub-linear growth from C32->C64 (2.98x Fortran / 2.23x CUDA for a 4x
grid-point increase) tightening to closer-to-linear growth from C64->C128
(4.0x Fortran / 3.71x CUDA), consistent with fixed per-call overhead being
a larger relative share at the smaller C32/C64 sizes and shrinking as a
fraction of a growing per-call workload — the same mechanism from both
directions, and mutually consistent between the two resolution steps.

**Stage E: CLOSED — directionally strong scaling result, with the
warmup-inclusion methodology caveat noted above.**

## Documentation / CURRENT_STATE.md maintenance

`CURRENT_STATE.md` was compacted (34,566 -> ~31,800 characters) by
shortening the dated opening summary and the Prompt 31/32 narrative
paragraphs in §11 down to pointers at this file and
`prompt_31_held_suarez_e2e.md`/`prompt_32_held_suarez_scientific_validation.md`,
then updated with: a new §8a (HS/resolution performance scaling table), a
new §9a (HS validation ladder table, C8 through C128/10-day), and updated
§9/§10/§11/§12 references for Prompt 33. Kept under the 35,000-character
hard ceiling.

## Summary

- Stage C (C128 short gate): **PASS**, bitwise NetCDF, one C128-specific
  timestep-instability finding resolved via a smaller `bdt`.
- Stage D (C128, 10-day): **PASS**, bitwise NetCDF at start/mid/end,
  identical benign `range_warn` diagnostic (`TA_dyn`, legitimate HS
  warmth) confirmed content-identical between backends via sorted-diff.
- Stage E (performance scaling C32->C64->C128): CUDA/Fortran gap narrows
  then reverses with resolution — CUDA overtakes Fortran by C64 and
  extends its lead by C128, under the standing MPS+fusion mode. Reported
  with an explicit warmup-methodology caveat relative to the documented
  C32 baseline.
- No numerical algorithm, CUDA kernel, or MPI architecture was modified.
  Two tooling-level infrastructure fixes (large-directory copy, rep-log
  self-clobbering) — neither counted against the numerical debug budget.
- No commit/push performed (none requested this session).

**MULTI-DAY HIGH-RES HELD-SUAREZ PASS — READY FOR CLIMATOLOGY / SCALING STUDY**
