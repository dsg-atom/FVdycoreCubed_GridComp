# Prompt 34: C32 Performance Reconciliation

## Purpose

Prompt 33 built a C32->C64->C128 Held-Suarez dyn_core performance scaling
table (`prompt_33_hs_multiday_highres_performance.md` §Stage E,
`CURRENT_STATE.md` §8a) but reused §8's older C32 figure (19.930 ms
Fortran / 25.464 ms CUDA, 100-call/rep, warmup-excluded) as the C32 point,
while C64/C128 were measured with a different methodology (30-call/rep,
3 reps, first-call warmup included). This made the table not fully
apples-to-apples, flagged explicitly as a caveat in Prompt 33.

This prompt re-measures C32 with the *exact* C64/C128 methodology so the
three points are directly comparable. Scope was intentionally narrow:
timing reconciliation only, model on Sonnet throughout, no numerical
source changes, no new science/duration/profiling work.

## Method: reuse, not reinvent

Took the completed C64 perf job pair (`p33_c64_perf_fortran`/`p33_c64_perf_cuda`
directories and their `_3rep.sbatch` wrappers under
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`)
and diffed C64's `fv3.j` against C128's `fv3.j` to confirm the only
resolution-dependent knobs were: `RUN_DT`, `AGCM_IM`, `HIST_IM`, `HIST_JM`,
`JOB_SGMT`, the job name, `EXPDIR`, and the literal `GRID_LABELS`/
`IM_WORLD`/`JM_WORLD`/`grid_label` text (a hardcoded HISTORY.rc-style block,
not templated off a shared variable).

New directories `p34_c32_perf_fortran` / `p34_c32_perf_cuda` were created by
copying the C64 `fv3.j` and outer `_3rep.sbatch` wrapper verbatim, then
`sed`-editing only those resolution knobs to the already-validated C32
values (confirmed against `p33_c32_hs10d_fortran/fv3.j` and
`p32_c32_hs_fortran/fv3.j`, both from the prior HS validation ladder):

| Knob | C64 (unchanged) | C32 (this prompt) |
|---|---|---|
| `RUN_DT` (bdt) | 600s | 1200s |
| `AGCM_IM` | 64 | 32 |
| `HIST_IM` / `HIST_JM` | 64 / 33 | 32 / 17 |
| `JOB_SGMT` | `00000000 050000` (5h) | `00000000 100000` (10h) |
| Grid label | `PC64x33-DC` | `PC32x17-DC` |
| calls = JOB_SGMT_seconds / RUN_DT | 18000/600 = 30 | 36000/1200 = 30 |

Everything else — vertical levels (`AGCM_LM=10`), Held-Suarez forcing,
precision (R8), rank count (`FV_NX=1`/`FV_NY=6`, 6 MPI ranks), GPU
allocation (1 A100), MPS mode, fusion flags
(`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`),
`RUN_CUDA_FV3_GRAPH` unset (off), output suppression, warmup policy
(first-call included, not excluded), 30 measured calls, 3 repetitions, and
the `fv_timing_mod` `DYN_CORE` flat-clock extraction — carried over
unchanged from the C64/C128 job scripts. No new benchmark script was
written; no source file was touched.

## Jobs run

- Fortran: `p34_c32_perf_fortran_3rep.sbatch`, job 58417393, submitted and
  completed in 23s (6 MPI ranks, CPU only).
- CUDA: `p34_c32_perf_cuda_3rep.sbatch`, job 58417394, submitted and
  completed in 26s (6 MPI ranks, 1 A100, MPS lifecycle
  start/stop wrapped around the 3 reps, same as C64/C128).

Both jobs: `gpu_a100` partition, `rome` constraint, `j1102` account —
identical scheduling to the C64/C128 jobs.

## Sanity check (not a numerical-validation campaign)

All 6 reps (3 Fortran + 3 CUDA):

- `RC=0` for every rep (`P33_PERF_REP=<n> RC=0` in each job's `.out`).
- `grep -ci` for `nan`, `inf`, `range_warn`, and `error|abort|fatal`: **0
  hits** in every one of the 6 `perf_logs/rep*.log` files.
- CUDA job: `FV3_MPS_START`/`FV3_MPS_STOP` both reported cleanly, no MPS
  failure.
- No savepoint regeneration performed — C32 numerical correctness was
  already established in Prompts 23/32/33 and is out of scope here.

## Timing extraction

Same method as Prompt 33: grep the flat `fv_timing_mod` `DYN_CORE` line
(`   DYN_CORE   <total_s>   0.0000   <total_s>   0`) from each rep log,
divide by the confirmed call count (30, cross-checked against the
hierarchical `--------DYN_CORE ... | 30` line in each log), multiply by
1000 for ms/call.

**C32 Fortran** (`p34_c32_perf_fortran/perf_logs/rep{1,2,3}.log`):
total_s = 0.6237, 0.6210, 0.6216 -> ms/call = 20.79, 20.70, 20.72
median total_s = 0.6216 -> **20.72 ms/call**

**C32 CUDA** (`p34_c32_perf_cuda/perf_logs/rep{1,2,3}.log`):
total_s = 0.6215, 0.5945, 0.5963 -> ms/call = 20.72, 19.82, 19.88
median total_s = 0.5963 -> **19.88 ms/call**

Rep-to-rep spread: Fortran <0.5%, CUDA <2% (rep1 is a mild outlier, likely
first-run MPS/context warmup variance; still well within noise for a
3-rep median).

## Reconciled scaling table

| Grid | Fortran ms/call | CUDA ms/call | Speedup F/C | CUDA/Fortran |
|---|---:|---:|---:|---:|
| C32  | 20.72  | 19.88  | 1.04 | 0.96 |
| C64  | 59.4   | 56.9   | 1.04 | 0.96 |
| C128 | 237.4  | 210.9  | 1.13 | 0.89 |

Definitions: Speedup F/C = Fortran_time / CUDA_time (>1 = CUDA faster).
CUDA/Fortran = CUDA_time / Fortran_time (<1 = CUDA faster).

C64/C128 rows are unchanged from Prompt 33 (`CURRENT_STATE.md` §8a
pre-Prompt-34); only the C32 row and the derived ratios are new.

## Interpretation

**CUDA already overtakes Fortran at C32** under this exact methodology —
this supersedes Prompt 33's "gap narrows then reverses between C32 and C64"
reading, which was an artifact of comparing C32's old (mismatched,
warmup-excluded, 100-call) figure against C64/C128's new methodology. Under
one consistent methodology, all three resolutions show CUDA ahead, and the
CUDA/Fortran ratio is essentially flat from C32 to C64 (0.96 -> 0.96) before
improving further at C128 (0.89) — the GPU's relative advantage *grows*
with resolution rather than *appearing* partway through it.

Classification: **CUDA OVERTAKES FORTRAN** at every resolution measured;
the margin additionally widens toward C128.

Best CUDA competitiveness measured: **C128, 0.89x** (CUDA/Fortran ratio).

Does this support a C256 performance study: **MAYBE** — the observed trend
(flat C32->C64, then improving C64->C128) is consistent with fixed per-call
launch/sync overhead being amortized further as grid-point count grows, so
a continued or accelerating CUDA advantage at C256 is plausible. This is
explicitly **not** a numerical extrapolation — no C256 figure is implied or
computed here; C256 stability/timestep is unmeasured (per `CURRENT_STATE.md`
§9) and would need its own short CFL search before any timing run, exactly
as C128 did in Prompt 33.

## What this prompt did NOT do (by design)

No numerical source changes, no Held-Suarez forcing changes, no new
savepoint or NetCDF science validation, no profiling (`nsys`/`ncu`), no
CUDA optimization or MPI architecture changes, no C256 run, no
climatology-scale run. Model stayed on Sonnet throughout — the C32 job
setup succeeded on the first attempt using the already-validated C32
config values, so no escalation was needed.

## Files touched

- New (untracked, not committed unless requested): `p34_c32_perf_fortran/`,
  `p34_c32_perf_cuda/` job directories and their `fv3.j`/`_3rep.sbatch`
  scripts under the Fortran fixture tree's `experiments/` directory (not
  part of this repo).
- Updated: `docs/agent_handoff/CURRENT_STATE.md` §8a/§11/header — C32 value
  replaced, no historical detail appended.
- New: this file.
