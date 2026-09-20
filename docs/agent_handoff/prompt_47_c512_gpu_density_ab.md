# Prompt 47: Controlled Same-Node C512 GPU-Rank-Density A/B

## Context

Prompt 46 flagged an informal, uncontrolled ~2x gap between:
- Prompt 37's C512 1-GPU/6-rank `DYN_CORE` figure: 9089.05 ms/call
- Prompt 46's C512 2-GPU/3-ranks-per-GPU `DYN_CORE` figure: ~4701 ms/call

If real, this would mean the documented C512 CUDA-vs-Fortran performance
regression (`CURRENT_STATE.md` §6, ratio 1.270x CUDA-slower) is
substantially a per-GPU rank-density/resource-contention effect at the
production 6-ranks-per-A100 density, not (solely) the MPI transport/
protocol cost growth root-caused in Prompts 38-40. Prompt 46 explicitly
deferred this as out of its own scope (a topology-change question, not a
transport question) and flagged it as the top candidate for a dedicated
future prompt.

This prompt (48 from the session's own accounting inside the job, but the
next available slot after Prompt 46 in `CURRENT_STATE.md`'s handoff
sequence — filed as Prompt 47) runs that controlled comparison.

## Method

Single sbatch job (58459292, `--gres=gpu:3`, one node), so GPU hardware,
CPU, and node contention are held constant across all three layouts —
only `CUDA_VISIBLE_DEVICES` changes (existing
`dev = mod(comm_rank, device_count)` round-robin in
`cuda/interface/fv3_gpu_dyncore_bridge.F90`, **no source change**). Same 6
MPI ranks/6 tiles/1-tile-per-rank FV3 decomposition throughout.

| Layout | CVD | GPUs | Ranks/GPU |
|---|---|---|---|
| A | `0` | 1 | 6 |
| B | `0,1` | 2 | 3 |
| C | `0,1,2` | 3 | 2 |

(Layout D, 6 GPUs/1 rank each, is unavailable — every `gpu_a100` node has
a hard 4-GPU ceiling, confirmed via `sinfo`/`scontrol`.)

- MPS ON for every layout (`fv3_mps_lifecycle.sh`).
- `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`
  (recommended CUDA mode). `RUN_CUDA_FV3_GRAPH` unset (OFF).
- `RUN_CUDA_FV3_MPI_TRANSPORT` unset → `host_staged` (default) for all
  three layouts. No `cuda_aware`, no MPI tuning.
- 3 perf reps per layout, 30 `dyn_core` calls each (18 simulated minutes
  @ `bdt=36s`), same `fv_timing_mod` `DYN_CORE`/`COMM_DSW`/`COMM_TOTAL`
  extraction convention as Prompts 37/40/46.
- Layout C only: a Layout-C-only 2-simulated-hour correctness smoke
  first, because 3-GPU/2-ranks-per-GPU is a placement never tested at any
  MPS setting. Layout A is the long-established default; Layout B was
  already validated BITWISE-correct (Prompt 46); MPS itself was
  established bitwise-neutral (Prompt 26 §4.1) — so a fresh NetCDF
  bitwise check was not required for A/B (avoids a full numerical
  campaign per project cost policy, matching Prompt 46's own rationale).

sbatch script: `<fortran_perf>/experiments/p47_c512_mgpu_scaling.sbatch`.
Job output: `p47_c512_mgpu_scaling_58459292.out` (~5700 lines). Perf run
dir: `p47_c512_mgpu_perf/`. Correctness run dir:
`p47_c512_mgpu_correctness/`.

Separately, a same-session Fortran-backend rerun of the existing Prompt
37 3-rep C512 template was submitted (job 58459788,
`p37_c512_perf_fortran_3rep.sbatch`) as an attempted paired baseline.

## Harness defect discovered (not fixed here)

Both sbatch scripts use the pattern:
```
logf=$(ls -t scratch_*.log 2>/dev/null | head -1)
cp "$logf" "perf_logs/..."
grep -ci range_warn $logf ...
```
For every rep in both jobs, `logf` resolved to the empty string — no
`scratch_*.log` was produced by `fv3.j` in this harness configuration.
Consequences:
- `perf_logs/` is empty in both `p47_c512_mgpu_perf/` and
  `p37_c512_perf_fortran/perf_logs/` — no per-rep log was preserved.
- The correctness smoke's own `P47_C_CORRECTNESS_Range_Warn=0
  P47_C_CORRECTNESS_NaN=0 P47_C_CORRECTNESS_Error=0` line is **not a real
  check** — `grep -ci range_warn` with no file argument reads from
  (redirected/empty) stdin and trivially returns 0. It is not evidence of
  a clean run by itself.
- `P47_C_CORRECTNESS_NC4_COUNT=0` similarly reflects that no `.nc4` was
  found under the correctness run's cwd within the searched window (the
  correctness `fv3.j`'s `HISTORY.rc` requests `inst1_2d_diag`/
  `inst3_3d_diag` at `frequency=020000`/`duration=020000`, exactly at the
  end of the 2h segment — checked, not further pursued; out of this
  prompt's bounded-debugging scope).

**Actual correctness/perf data was recovered by parsing the raw
`.out` file directly** — the `fv_timing_mod` block
(`TOTAL`/`COMM_TOTAL`/`FV_DYN_LOOP`/`DYN_CORE`/.../`COMM_DSW`/...) is
still printed to stdout regardless of the broken log-copy step, and a
manual `grep -iE "nan|range_warn|error|abort|fatal"` over the full
`.out` file (excluding the script's own `set -x` trace lines) found
zero real hits, with `Status = 0` printed 10/10 times (9 perf reps + 1
correctness run).

**Recommendation**: fix the `scratch_*.log` glob (or switch to whatever
naming `fv3.j` actually produces in this harness setup) before reusing
this sbatch pattern for future prompts — otherwise its own reported
Range_Warn/NaN/Error counts are silently meaningless.

## Results

### Rank-density A/B (perf reps, `DYN_CORE`/`COMM_TOTAL`/`COMM_DSW`, ms/call = seconds/30*1000)

| Layout | Rep | DYN_CORE (s) | ms/call | COMM_TOTAL (s) | ms/call | COMM_DSW (s) | ms/call |
|---|---|---|---|---|---|---|---|
| A (1 GPU) | 1 | 116.7694 | 3892.31 | 9.0968 | 303.23 | 4.8482 | 161.61 |
| A (1 GPU) | 2 | 116.8798 | 3896.00 | 9.3330 | 311.10 | 5.0841 | 169.47 |
| A (1 GPU) | 3 | 116.6053 | 3886.84 | 9.1125 | 303.75 | 4.7347 | 157.82 |
| B (2 GPU) | 1 | 116.6198 | 3887.33 | 8.8288 | 294.29 | 4.5946 | 153.15 |
| B (2 GPU) | 2 | 117.1246 | 3904.15 | 9.7855 | 326.18 | 4.9840 | 166.13 |
| B (2 GPU) | 3 | 116.9883 | 3899.61 | 9.8522 | 328.41 | 5.3807 | 179.36 |
| C (3 GPU) | 1 | 116.5659 | 3885.53 | 8.9028 | 296.76 | 4.5470 | 151.57 |
| C (3 GPU) | 2 | 116.6875 | 3889.58 | 9.3116 | 310.39 | 4.9717 | 165.72 |
| C (3 GPU) | 3 | 116.5497 | 3885.00 | 9.3024 | 310.08 | 4.9960 | 166.53 |

**Median `DYN_CORE` ms/call**: A=3892.31, B=3899.61, C=3885.53. Full
9-value range: 3885.00-3904.15, a 0.49% spread. `COMM_TOTAL`/`COMM_DSW`
show mild rep-to-rep scatter (~10%) but no monotonic trend with GPU
count.

**Conclusion: rank/GPU density (1, 2, or 3 A100s at 6/3/2 ranks-per-GPU)
has no measurable effect on C512 `DYN_CORE`, `COMM_TOTAL`, or `COMM_DSW`
under `host_staged` transport, MPS ON, fusion ON, on this node
(`warpa007`).** Prompt 46's informal ~2x gap does not reproduce under a
controlled same-node/same-job comparison — it is refuted as a
rank-density/contention effect.

### Layout C correctness smoke

`rc=0`, `Status = 0`, zero real NaN/Range_Warn/Error/Abort/Fatal hits
across the full raw `.out` file (verified manually, see harness-defect
section above). This confirms Layout C (3 GPUs/2 ranks each, never
tested before at any MPS setting) does not crash or produce obviously
invalid output over a 2-simulated-hour run. This is a clean-exit smoke
confirmation only — not a bitwise NetCDF comparison (by design, per the
script's own stated rationale, matching Prompt 46's approach for its own
new placement).

### Fortran-vs-CUDA regression status: NOT resolved by this job

The paired Fortran rerun (job 58459788) was intended as a same-session
baseline, but `sacct` shows it landed on a **different node**
(`warpa005`) than the CUDA job (`warpa007`, job 58459292) — confirmed via
`sacct -j 58459788,58459292 --format=JobID,NodeList`. Per this project's
own documented policy (Prompt 40: "any future A/B performance comparison
must run all arms on the same allocated node/job"), this is **not** a
valid CUDA-vs-Fortran comparison.

Fortran rerun `DYN_CORE` (job 58459788, node `warpa005`, 3 reps):
211.8570s, 211.4386s, 211.7285s → 7061.90 / 7047.95 / 7057.62 ms/call,
median 7057.62 ms/call. (For reference, this is close to Prompt 37's
originally-documented Fortran figure of 7155.63 ms/call — consistent.)

The CUDA side's own absolute number is the surprising part: Layout A on
`warpa007` (this job) measured **3892.31 ms/call**, a **2.3x** speedup
over Prompt 37's originally-documented CUDA figure for the nominally
identical 1-GPU/6-rank/fused/MPS-on config (**9089.05 ms/call**). This is
far beyond the ~30% node-to-node variance Prompt 40 characterized
(`warpa005` vs `warpa009`). Possible explanations, none isolated here:
- Node hardware variance is larger than previously characterized at
  C512 scale (this job's node, `warpa007`, has not been characterized
  before).
- An environment/config difference between this job and Prompt 37's
  original run that isn't visible from `CURRENT_STATE.md`'s table alone
  (e.g. an MPS state difference, a different `I_MPI_*`/OpenMPI env var,
  or the original Prompt 37 CUDA measurement not actually having had the
  fusion flags on despite the table's heading).

**This is a real open item, not resolved by Prompt 47.** If the
regression documented in `CURRENT_STATE.md` §6 no longer holds at current
node conditions, that would be a materially good-news update — but it
must be established with a genuine same-node paired Fortran/CUDA rerun,
not inferred from this job alone.

## Recommended Next Step (Prompt 48 candidate)

Same-node paired Fortran-vs-CUDA C512 rerun, both arms in one
allocation/job (e.g. request one node, run 3 Fortran reps then 3 CUDA
Layout-A reps in the same sbatch, or interleave as Prompt 46 did for its
transport A/B). This single job would resolve both open items together:
whether the CUDA-vs-Fortran regression still holds, and whether the 2.3x
swing seen here is node-specific.

Cheaper first check before spending a new GPU allocation: diff the
environment/module state and exact `RUN_CUDA_FV3_*`/`I_MPI_*`/MPS
settings between this job's sbatch and Prompt 37's original sbatch/log,
to rule out a config difference before attributing the 2.3x swing to node
hardware.

## Bottom line

- Rank/GPU-density question (Prompt 46's motivating lead): **CLOSED, no
  effect** — the C512 regression is not substantially a per-GPU
  density/contention artifact.
- Layout C (3 GPU/2 ranks) placement: correctness smoke clean, no
  bitwise validation performed (by design).
- Fortran-vs-CUDA C512 regression status: **UNRESOLVED**, and a new,
  larger-than-expected node-variance data point was surfaced.
- Harness defect (broken `scratch_*.log` capture in the reused C512 perf
  sbatch template) discovered; not fixed.
- No production or CUDA source files changed in this prompt (Layout
  placement only, via `CUDA_VISIBLE_DEVICES`).
