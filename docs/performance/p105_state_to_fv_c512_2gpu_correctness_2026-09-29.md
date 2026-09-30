# Prompt 105 — `State_To_FV` C512/2-GPU Numerical Correctness Closure (2026-09-29)

## 0. Status

**STATE_TO_FV C512 2-GPU NUMERICAL VALIDATION COMPLETE — READY FOR DEFAULT
ENABLE.** Field-level comparison classifies BITWISE (T, PS) / ROUND-OFF
CLEAN (U, V) at C512/2-GPU/3-ranks-per-GPU, closing Prompt 104's remaining
gap. **No CUDA/Fortran source changed, no production default changed
(`State_To_FV` remains Fortran-default), no MPI change, no commit/push.**

**Superseded in part by Prompt 106** (`p106_state_to_fv_default_policy_
2026-09-29.md`): the "remains Fortran-default" statement above described
this task's own scope only. Prompt 106 made `State_To_FV` CUDA the
default specifically at `device_count>=2` (this task's exact validated
density), with Fortran retained as the default everywhere else
(including 1-GPU). This doc's correctness finding is otherwise unchanged
and remains the authoritative numerical-agreement evidence for that
density and code path.

## 1. Key Finding: Prompt 104's "no HISTORY output" claim was a look-in-the-
wrong-place error, not a real gap

Re-inspecting job 58645113's already-existing rundirs
(`p104_correctness_fortran`, `p104_correctness_cuda`) found that **HISTORY
output was in fact produced** by both arms — `HISTORY.rc` (defining
`inst1_2d_diag` and `inst3_3d_diag` collections with `PS`/`T`/`U`/`V`, among
others) and the corresponding `.nc4` files exist under each rundir's
`scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8/` execution
subdirectory. Prompt 104's correctness write-up checked only the top-level
rundir (where, correctly, no `HISTORY.rc`/`.nc4`/`savepoints` content
exists — the top level only ever holds `fv3.j`, the binary, and the
post-run copied log) and never descended into the `scratch_*` subdirectory
that `fv3.j` actually populates and where MAPL History writes its output.
**No new job was required for this task** — the existing job 58645113 data
already satisfies Stage 0-4 of this task's requirements; this task's real
work was Stage 5-11 (the comparison and classification that was never run).

This is recorded as a new "Do Not Repeat" lesson in `CURRENT_STATE.md` §8.

## 2. Configuration (recovered/reconfirmed, unchanged from Prompt 104)

- Executable: `StandAlone_FV3_Dycore.x`, R8, Intel MPI 2021.13 build.
- SHA256 (both arms, verified identical):
  `2339366b10885c4532f08efa089d4e3b0a062403397f0c64c6a396b186e2064d`
  (matches the production-clean binary used throughout P101A-P104).
- C512, 6 MPI ranks (`FV_NX=1`/`FV_NY=6`), 2 A100 GPUs, 3 ranks/GPU
  (`CUDA_VISIBLE_DEVICES` unset, `dev = mod(comm_rank, device_count)`
  unchanged), MPS ON (`fv3_mps_lifecycle.sh`, confirmed active for the full
  job including this pair), Held-Suarez forcing, `COLDSTART=1`.
- `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`
  (established best config), Graph OFF, all other CUDA ports at their
  current production defaults (PT/DELZ, REMAP_UV, REMAP_Q,
  REMAP_PRESSUREVARS = CUDA; TRACER_2D, REMAP_TOTALENERGYCON = Fortran).
- `HEARTBEAT_DT=36`, `JOB_SGMT=00:18:00` → 30 calls/run (`DYN_DATE_COUNT`),
  matching every other P101A-P104 run.
- `AGCM.rc`, `HISTORY.rc`, `input.nml`: byte-identical between the two arms
  (`diff` clean) — confirmed via direct diff of both rundirs' execution
  copies. **Only difference between arms**: the `State_To_FV` selector
  environment variable, set by the sbatch harness's `run_arm()` helper —
  Arm A `RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1`, Arm B
  `RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1` (confirmed via
  `p104_state_to_fv_2gpu_validation.sbatch` lines 105-110; `FORCE_FORTRAN`
  confirmed unset for Arm B).
- HISTORY output: `inst1_2d_diag` (`PS`, `PHIS`, `DXC`, `DYC`, `AREA`,
  frequency=duration=00:10:00) and `inst3_3d_diag` (`PS`, `PHIS`, `U`, `V`,
  `T`, `AREA`, on 25 standard pressure levels 1000-100 hPa, same
  frequency/duration). With `JOB_SGMT=00:18:00`, only the first 10-minute
  bucket completes before the run ends — **exactly one HISTORY checkpoint**
  (`*.18910301_0010z.nc4`) exists per collection per arm; no second
  checkpoint was ever going to be written by this harness's duration. This
  was sufficient for the task (no long run required) but means Stage 8
  (temporal-growth check) cannot use two independent checkpoints — see §7.

## 3. Runs

| Arm | Rundir | rc | DYN_DATE_COUNT | FATAL/OOM/Abort/NaN/Inf |
|---|---|---|---|---|
| Fortran (A) | `p104_correctness_fortran` | 0 | 30 | 0 real matches (grep hits are `MAPL: INFO` substring false-positives on "Inf") |
| CUDA (B) | `p104_correctness_cuda` | 0 | 30 | 0 real matches (same false-positive pattern) |

HISTORY output: **PRESENT** for both arms (`inst1_2d_diag`,
`inst3_3d_diag`, one timestamp each, `18910301_0010z`).

## 4. CUDA Arm Dispatch Verification (reused evidence, no new instrumentation)

- Selector: `RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1` active in Arm B's
  environment for this run (sbatch `run_arm()` logic, confirmed by direct
  script inspection — no `FORCE_FORTRAN` set).
- `State_To_FV` `mpp_clock` timer signature (10-dash `hits=30` line,
  ms/call, min/avg/max of the exclusive-time triplet): Fortran arm
  **2.61/2.72/3.11**, CUDA arm **1.62/1.68/2.27** — matches Prompt 104's
  independently-measured 2.72 ms (Fortran) / 1.68 ms (CUDA) signature
  exactly, corroborating real CUDA-path execution.
- Memory: Prompt 104's concurrent poller on this same CUDA-arm run
  (`P104_CORR_poll.log`) measured a converged **+440.4 MiB/GPU** device
  footprint increase over the Fortran-arm baseline — unchanged finding,
  not re-measured here (no new poller needed for a correctness-only task).
- No additional dispatch instrumentation was added to source, per this
  task's explicit prohibition — the above is Prompt 104's already-verified
  evidence, reused for the identical rundirs/binary.

## 5. Field-Level Comparison

Tool: `tools/analysis/hs_netcdf_inspect.py compare-backends`
(`module load python/GEOSpyD/24.11.3-0/3.12`), single available time index
(0 — the only HISTORY checkpoint written, §2).

### Required fields (25-level pressure grid, `inst3_3d_diag`; `PS` from `inst1_2d_diag`)

| Field | Level | max_abs | rms | max_rel (guarded) | NaN either side | Class |
|---|---|---|---|---|---|---|
| T | 500 hPa | 0.0 | 0.0 | 0.0 | 0/131584 | BITWISE |
| U | 250 hPa | 5.821e-11 | 2.269e-13 | 7.195e-08 | 0/131584 | ROUND-OFF CLEAN |
| V | 250 hPa | 2.095e-09 | 1.042e-11 | 1.715e-06 | 0/131584 | ROUND-OFF CLEAN |
| PS | surface | 0.0 | 0.0 | 0.0 | 0/131584 | BITWISE |

### Multi-level sweep (T/U/V, confirms the pattern is level-independent, not one lucky level)

| Level (hPa) | T max_abs | U max_abs | V max_abs | Notes |
|---|---|---|---|---|
| 1000 | NaN/NaN (131584/131584) | NaN/NaN | NaN/NaN | pre-existing benign MAPL near-surface masking (§6) |
| 975 | NaN/NaN (131584/131584) | — | — | same masking |
| 950 | 0.0 | — | — | BITWISE |
| 925 | 0.0 | — | — | BITWISE |
| 850 | 0.0 | 2.910e-11 | 2.328e-09 | ROUND-OFF CLEAN |
| 700 | 0.0 | 2.910e-11 | 3.027e-09 | ROUND-OFF CLEAN |
| 500 | 0.0 | 2.910e-11 | 2.794e-09 | ROUND-OFF CLEAN |
| 250 | 0.0 | 5.821e-11 | 2.095e-09 | ROUND-OFF CLEAN |
| 100 | 0.0 | 5.821e-11 | 1.118e-08 | ROUND-OFF CLEAN |

`T` and `PS` are bitwise identical at every valid (non-masked) level; `U`
and `V` carry consistent round-off-level differences (max_rel 1e-8 to
1e-6) at every level, with no growth trend across levels or with altitude.

**Optional fields** (`W`, `DELP`, `DZ`/`DELZ`, `PKZ`): **not available** —
neither collection includes them, and adding them would require a
`HISTORY.rc` field-list change plus a rerun; deferred per the explicit
"do not add new diagnostics" scope limit. `T`/`U`/`V`/`PS` (the required
set) are fully covered.

First checkpoint (only checkpoint, `0010z`): **PASS** (all four required
fields BITWISE or ROUND-OFF CLEAN).

Final checkpoint: **NOT AVAILABLE** — only one HISTORY bucket completed
within the 18-minute run (§2); not a defect, expected from this harness's
duration and not required by the task.

## 6. Known Pre-Existing Artifact (unchanged, not a new finding)

`T`/`U`/`V` are `NaN` at 1000 hPa and 975 hPa (`n_nan_either=131584/131584`
— i.e., every grid point) on **both** arms identically (`np.array_equal`
mask check confirmed the Fortran and CUDA NaN masks are pixel-for-pixel
identical, zero valid-overlap points to compare there). This is the
already-documented `CURRENT_STATE.md` §4 "near-surface NaN in 3/25 levels
(standard MAPL masking)" artifact — pre-existing in both backends, not a
new CUDA-specific divergence. It does not affect the classification above
since all four required fields have clean, non-masked levels (e.g. 500/250
hPa) actually compared.

## 7. Halo / Layout Sensitivity

MAPL History regrids all six cubed-sphere tiles to a single global
lat-lon `PC512x257-DC` grid before writing `HISTORY.rc` output, so no
fixed lat-lon feature corresponds to a tile boundary — a dedicated
tile-boundary-resolved comparison would require reading pre-regrid
cubed-sphere-native `.nc4`/savepoint data, which this harness's
`HISTORY.rc` does not produce (out of scope: "do not write new analysis
infrastructure unless needed"). As the cheapest available proxy, the
location of the maximum `U`/`V` discrepancy at 250 hPa was inspected
directly (ad hoc `xarray` read, no new tooling committed):

- `U`: max diff 5.821e-11 at lat=0.00, lon=-99.14 (single interior point).
- `V`: max diff 2.095e-09 at lat=42.89, lon=-37.27 (single interior point).
- In both cases only **2 of 131,584 points** exceed half the field's own
  maximum difference — i.e., the discrepancy is sparse and spatially
  unstructured, not clustered along any linear/banded feature that would
  indicate a halo-exchange or PE-transpose layout error (which would be
  expected to produce spatially coherent, much larger-magnitude errors
  concentrated along tile-edge lines, not isolated round-off-level points
  scattered across the equator and mid-latitudes).

**PE/layout evidence: PASS** (no boundary-clustered discrepancy pattern
found; differences are consistent with floating-point summation-order
effects in the `u`/`v` host-boundary-fill and PE-transpose arithmetic, not
a structural layout defect).

## 8. Temporal Stability

**NOT AVAILABLE this task** — only one HISTORY checkpoint exists (§2); no
second independent time to compare against. Per the task's own scope
("no long climate run is needed... bounded C512 correctness closure"),
this is not required — a single validated checkpoint at C512/2-GPU is the
task's stated goal, and it is met by §5-7 above.

## 9. Validation Hierarchy

- **GPU EXECUTED**: YES (already established, Prompt 104; reconfirmed
  here by memory footprint and timer signature, §4).
- **NUMERICALLY VALIDATED**: **YES**, at C512/6-ranks/2-GPU/3-ranks-per-GPU
  — closing Prompt 104's gap. T/PS bitwise, U/V round-off clean
  (max_rel <= 1.7e-6, no NaN divergence, no boundary-clustered error
  pattern) at every level checked.
- **PERFORMANCE VALIDATED**: YES (already established, Prompt 104: direct
  1.633x, TOTAL +1.01%; not re-measured in this correctness-only task).

## 10. Existing Performance (reused from Prompt 104, not re-measured)

- `State_To_FV` direct speedup: **1.633x mean** (38.75% reduction, 3/3
  pairs, stddev 0.006x).
- TOTAL improvement: **+1.01% mean** (MEANINGFUL, boundary).
- Memory: **SAFE** (+440.4 MiB/GPU, 19.3 GiB/GPU free at converged state).

No performance A/B was run in this task, per its explicit scope.

## 11. Decision

**READY FOR DEFAULT ENABLE.**

Reason: all three legs of the project's validation hierarchy are now
independently established at the exact target density (C512, 6 ranks, 2
GPUs, 3 ranks/GPU, MPS ON, R8): GPU EXECUTED (Prompt 104), PERFORMANCE
VALIDATED (Prompt 104, positive and reproducible), and now NUMERICALLY
VALIDATED (this task — BITWISE/ROUND-OFF CLEAN on all four required
fields, no halo/layout-pattern discrepancy, no unmasked NaN divergence).
This satisfies the condition Prompt 104 itself set for revisiting
default-enable. **Per this task's explicit scope, no default was changed
in Prompt 105** — default-enabling `State_To_FV` CUDA is deferred to a
separate, dedicated bounded task (§13).

## 12. Supported Scope (if a future default-enable task proceeds)

Validated: **R8, C512, 6 MPI ranks, 2 A100 GPUs, 3 ranks-per-GPU, MPS ON,
single-node**, Held-Suarez forcing, `SW_DYNAMICS=false`, `hydrostatic=false`
(nonhydrostatic SIM1/`a_imp=1.0`), grid_type/selector combination as used
by every P98-104 run. **Not validated by this task**: R4, C1024+,
multi-node, 1-GPU/6-ranks density (still memory-blocked per P98-100),
4-GPU spread (P102, memory-only), any density other than 3-ranks-per-GPU
at 2 GPUs, longer-than-18-minute/multi-checkpoint runs.

## 13. Next Task

**If a default-enable is desired**: open a separate, dedicated bounded
task to flip `State_To_FV`'s default from Fortran to CUDA (or make CUDA
the default only when `device_count>=2`/density matches the validated
scope above) — this requires its own authorization and is explicitly out
of scope for Prompt 105. Otherwise, no further action is required on this
port; return to the other open items in `CURRENT_STATE.md` §9 (e.g.
`DYN_EPILOGUE` feasibility, `MOIST_CAPPA`/`a_imp` CUDA implementation).

## 14. Job / Evidence Provenance

- Reused, not re-run: Slurm job 58645113 (`p104_state_to_fv_2gpu_validation_58645113.out`),
  rundirs `p104_correctness_fortran/` and `p104_correctness_cuda/`
  (`GEOSfvdycore_fortran_perf/experiments/`), specifically each rundir's
  `scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8/` subdirectory
  (`AGCM.rc`, `HISTORY.rc`, `input.nml`, `*.nc4`, `P104_CORR_result.log`,
  `P104_CORR_poll.log`).
- New in this task: field-level comparison only (`hs_netcdf_inspect.py
  compare-backends`, run interactively on the login node under
  `python/GEOSpyD/24.11.3-0/3.12`) plus one ad hoc `xarray` snippet for the
  §7 max-diff-location check (not committed as new tooling).
- No CUDA/Fortran source changed. No production default changed. No MPI
  change. No commit/push in either repo.
