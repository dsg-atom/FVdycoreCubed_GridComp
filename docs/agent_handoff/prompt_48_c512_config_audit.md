# Prompt 48 (audit phase): C512 Config/Reproducibility Audit — Prompt 37 CUDA vs Prompt 47 Layout A

## Purpose

Before spending a new GPU allocation on a same-node paired Fortran-vs-CUDA
rerun, audit whether Prompt 37's original C512 CUDA measurement
(9089.05 ms/call, job 58444655) and Prompt 47's Layout A measurement
(3892.31 ms/call median, job 58459292) were actually run under identical
configuration. No Slurm jobs submitted, no source modified — read-only
inspection of existing job artifacts.

## Method

Diffed/inspected, at minimum, the 31 items requested: sbatch wrapper
scripts, `fv3.j` (which generates `AGCM.rc`/`HISTORY.rc` inline), env-var
exports, module lists, `esma_mpirun`/`mpirun` invocation lines, binary
path/hash/mtime, and the raw `fv_timing_mod` `DYN_CORE` values in both
jobs' raw `.out` stdout captures.

## Findings by item

| # | Item | P37 CUDA (job 58444655) | P47 Layout A (job 58459292) | Class |
|---|---|---|---|---|
| 1 | Slurm sbatch script | `p37_c512_perf_cuda_3rep.sbatch`, `--gres=gpu:1` | `p47_c512_mgpu_scaling.sbatch`, `--gres=gpu:3` (job-level; only GPU 0 visible via CVD for Layout A) | UNKNOWN (low plausibility — reserving 2 extra idle GPUs would not speed up the 1 GPU actually used; flagged, not ruled fully impossible) |
| 2 | `AGCM.rc` | generated inline by `fv3.j` | generated inline by identical `fv3.j` (byte-diff below) | COSMETIC (no diff) |
| 3 | `fv3.j` | self-referencing `EXPDIR` | **stale `EXPDIR` pointing to `p46_c512_2gpu_perf`** (copy-paste leftover from Prompt 46, not updated) — full diff shows only this line + job-name comment differ | CORRECTNESS-ONLY / provenance bug — see below, does not change physics or timing |
| 4 | `HISTORY.rc` | generated inline, identical script | identical | COSMETIC |
| 5 | `CUDA_VISIBLE_DEVICES` | unset (Slurm gres binding supplies the 1 allocated GPU) | explicitly `0` | COSMETIC (functionally equivalent: 1 GPU visible either way) |
| 6 | MPS startup | `fv3_mps_lifecycle.sh` / `fv3_mps_start`, confirmed `nvidia-cuda-mps-control -d` launched | same script, same confirmation | COSMETIC (no diff) |
| 7 | `RUN_CUDA_FV3` | CUDA backend selected (both jobs are CUDA arms by construction) | same | — |
| 8 | `RUN_CUDA_FV3_MPI_TRANSPORT` | unset → `host_staged` | unset → `host_staged` | identical |
| 9 | `RUN_CUDA_FV3_FUSE_HALO_EDGES` | `YES` | `YES` | identical |
| 10 | `RUN_CUDA_FV3_FUSE_CORNER_FILL` | `YES` | `YES` | identical |
| 11 | `RUN_CUDA_FV3_GRAPH` | unset (off) | unset (off) | identical |
| 12 | `I_MPI_OFFLOAD` | not referenced/unset | explicitly unset | identical |
| 13 | MPI implementation/version | Intel MPI 2021.13 (`esma_mpirun: mpi_type = intel`) | Intel MPI 2021.13, same | identical |
| 14 | Compiler/module environment | `comp/gcc/12.3.0`, `comp/intel/2024.2.0`, `mpi/impi/2021.13`, full 17-module list | byte-identical module list | identical |
| 15 | Executable path/hash/mtime | `/gpfsm/.../install/bin/StandAlone_FV3_Dycore.x`, mtime `Aug 31 11:26`, md5 `77315a9c...` | same path; confirmed same file on disk (mtime/md5 unchanged between the two job dates, no rebuild occurred) | identical |
| 16 | CUDA library/archive path | not separately probed (no `ldd`); implied identical via identical module env | same | UNKNOWN (not directly verified, low risk) |
| 17 | C512 resolution | `c512` (`EXPID` string `c512_L10_T2_1x6_1threads`) | identical `EXPID` string | identical |
| 18 | `npz` | `L10` in EXPID, embedded in identical script | identical | identical |
| 19 | `RUN_DT`/`bdt` | embedded in identical `fv3.j` namelist block | identical | identical |
| 20 | `n_split` | embedded in identical script | identical | identical |
| 21 | `k_split` | embedded in identical script | identical | identical |
| 22 | Rank count | 6 (`mpirun -np 6 ...`, confirmed both) | 6, confirmed | identical |
| 23 | Node count | 1 (`--nodes=1`) | 1 | identical |
| 24 | GPU count (visible to app) | 1 | 1 (Layout A) | identical |
| 25 | MPS mode | ON | ON | identical |
| 26/27/29 | Warmup/call-count/inclusion methodology | 3 reps × 30 `dyn_core` calls/rep, cumulative `fv_timing_mod` total (no separate warmup phase in this ladder methodology) | same 3×30 structure | identical |
| 28 | Timing extraction method | **verified**: raw stdout `fv_timing_mod` block AND the (correctly captured, for this job) `scratch_*.log` copy contain the **same** `DYN_CORE` value at matching positions (272.6715 s, rep1) — real FMS clock data, not the vacuous-grep path | raw stdout only (scratch-log copy failed due to item 3's stale `EXPDIR`, landed in `p46_c512_2gpu_perf/` instead) — but the value used in the Prompt 47 writeup was already parsed from the same authoritative raw stdout `fv_timing_mod` block | both ultimately authoritative; P47's harness-level log copy failed but the number reported was never sourced from the broken path |
| 30 | NetCDF/history output settings | generated inline, identical script | identical | identical |
| 31 | Diagnostic/debug env vars | `fms_nml`: `print_memory_usage=.true.`, `domains_stack_size=12000000`, `clock_grain='MODULE'`, `clock_flags='DETAILED'` (embedded, identical script) | identical | identical |

## Historical timing provenance (item required by task: A vs B)

Extracted all three raw `DYN_CORE` values from job 58444655's `.out`:
- rep1: 272.6715 s → **9089.05 ms/call**
- rep2: 275.1728 s → 9172.43 ms/call
- rep3: 266.4293 s → 8880.98 ms/call
- **median = 9089.05 ms/call (rep1)** — exactly the value documented in
  `CURRENT_STATE.md` §6's ladder table, and consistent with this
  project's own median-of-3 reporting convention (matching how Prompt 47's
  own medians were computed).

Cross-checked rep1's raw-stdout value against `perf_logs/rep1.log` (the
scratch-log copy, which worked correctly for this job because its
`EXPDIR` self-referenced correctly) — **identical value**, confirming the
number is real FMS `fv_timing_mod` clock output, not an artifact of the
broken `scratch_*.log`-glob harness pattern.

**Conclusion: provenance is VALID (case A, not case B).** The historical
9089.05 ms/call figure is authentic, correctly computed, and not
contaminated by the log-capture defect that affected Prompt 47's own
harness run.

## The stale-`EXPDIR` bug (new finding, distinct from Prompt 47's original harness-defect report)

`p47_c512_mgpu_perf/fv3.j` and `p47_c512_mgpu_correctness/fv3.j` are
literal copies of Prompt 46's run-directory scripts with the job-name
comment left unchanged and, critically, `EXPDIR` still pointing at
`p46_c512_2gpu_perf` / `p46_c512_2gpu_host_staged` respectively — never
updated to self-reference their own `p47_*` directories. Confirmed by
finding the Prompt 47 job's scratch log
(`scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8.log`, mtime
`Sep 18 21:42`, matching the P47 job's end time) sitting inside
`p46_c512_2gpu_perf/` instead of `p47_c512_mgpu_perf/`. Because all 9
P47 perf reps (3 layouts × 3 reps) share the same `EXPID` (density/GPU
count isn't part of the `EXPID` string), each rep's `tee` **overwrote**
that one file — only the very last rep (Layout C, rep 3) survives.

**Does this affect the P47 results reported?** No. `EXPDIR` is used only
for `SCRDIR`/scratch-log/savepoint paths and an optional binary-override
check (`if -e $EXPDIR/StandAlone_FV3_Dycore.x`) — confirmed that file
does **not** exist in `p46_c512_2gpu_perf/`, so the `else` branch ran and
both jobs used the identical canonical `$GEOSBIN/StandAlone_FV3_Dycore.x`
binary (verified by the printed `***** USING ****` line in both jobs'
raw stdout, byte-identical path). `AGCM.rc`/`HISTORY.rc`/namelists are
written to the script's `cwd` (the actual `p47_c512_mgpu_perf`/
`p47_c512_mgpu_correctness` directories), not `EXPDIR`, so the executed
physics configuration was correct in both directories. This is a
**harness bookkeeping bug only** — it explains *why* Prompt 47's own
`scratch_*.log` capture and per-rep log preservation failed (a second,
more specific root cause than the generic "broken glob" theory floated
in Prompt 47's own handoff), but the `DYN_CORE` numbers Prompt 47 reported
were parsed from raw stdout directly and are unaffected.

**Recommendation**: before any future job reuses a copied `fv3.j` from a
prior prompt's run directory, grep for `setenv EXPDIR` and confirm it
self-references the new directory. This is a cheap, mechanical check to
add to the harness-defect fix already flagged in `CURRENT_STATE.md` §9.

## Answer

**A. CONFIGS IDENTICAL ENOUGH FOR SAME-NODE RERUN.**

Across all 31 requested axes, the only substantive differences found
between Prompt 37's CUDA arm and Prompt 47's Layout A arm are:
- a harness-only stale-`EXPDIR` bug (item 3) — confirmed to not affect
  physics, binary selection, or the reported timing;
- cosmetic wrapper-script differences (job-level `--gres` count, explicit
  vs. implicit `CUDA_VISIBLE_DEVICES`) that do not change the actual
  runtime configuration of the arm being measured (1 GPU, 6 ranks, same
  binary, same modules, same fusion/transport flags, same MPI, same
  resolution/timestep/split parameters).

The historical 9089.05 ms/call figure's provenance is **confirmed valid**
(authentic `fv_timing_mod` clock data, correct median-of-3 statistic, not
sourced from the broken log-capture path).

No config confound was found that could explain the observed 2.3x gap
(9089.05 vs. 3892.31 ms/call). The two jobs ran on different physical
nodes (`warpa009` or Prompt 37, `warpa007` for Prompt 47) — both the same
node class/feature set per `scontrol` (`rome,edr,cssro,notshort`, 48 CPU,
510GB RAM, 4× A100-SXM4-40GB) — so **node-to-node A100 hardware variance
remains the only unruled-out explanation**, and it is larger than the
~30% previously characterized in Prompt 40.

**Recommendation: proceed with Prompt 48's same-node paired
Fortran-vs-CUDA C512 rerun** (single allocation, both arms on the same
node) — this is now the only remaining way to resolve whether the
documented C512 CUDA-vs-Fortran regression (§6) still holds under
current node conditions, since configuration has been ruled out as the
cause of the 2.3x swing. Awaiting explicit go-ahead before submitting
(per current instruction: audit only, no job submission this round).
