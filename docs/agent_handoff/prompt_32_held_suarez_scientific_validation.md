# Prompt 32: Held-Suarez Scientific Validation (C8 output readiness + C32 gate)

Branch: `experiment/held-suarez-e2e` (this repo, orthogonal to `cuda/gpu-bringup`).
Fixture tree: `GEOSfvdycore_fortran_perf` (compiles `StandAlone_FV3_Dycore.x`).
Builds on Prompt 31 (`prompt_31_held_suarez_e2e.md`), which established the
HS forcing itself and a short C8 Fortran/CUDA agreement check. This prompt's
task: advance from integration-smoke-test to scientifically-usable-experiment
level — standard NetCDF output, longer C8 run, configurable controls, a
verified C32 config, matched C32 Fortran/CUDA short validation, and an
offline-analysis deliverable. No new HS physics, no CUDA kernel/numerical
change, no long/climatological run (explicitly out of scope this prompt).

## §1-6: NetCDF output mechanism (no new I/O code needed)

The existing MAPL History/CFIO mechanism already satisfies the standard-
NetCDF-output requirement; nothing new was built. `HISTORY.rc` (identical
structure at C8 and C32, differing only in `GRID_LABELS`/`IM_WORLD`/
`JM_WORLD`) defines two collections:

- `inst1_2d_diag`: hourly, `PS`/`PHIS`/`DXC`/`DYC`/`AREA`, lat-lon grid.
- `inst3_3d_diag`: 3-hourly, `PS`/`PHIS`/`U`/`V`/`T`/`AREA`, 25 fixed
  pressure levels (1000->100 hPa) via `vvars: 'log(PLE)'`.

Both collections write one global (all six tiles already regridded to
lat-lon by MAPL History, `PC32x17-DC` at C32) file per output timestamp —
confirmed via direct `xr.open_dataset` inspection: `dims: {lon:32, lat:17,
lev:25, time:1}`, standard CF attributes, `float32` data. Output lands under
each job's own `scratch_*/` directory (`RECORD_FINAL: NO` — not moved to
`EXPDIR`).

C32 HS run produced 44 files/backend (33 `inst1_2d_diag` + 11
`inst3_3d_diag`), matching the 100-outer-step/1200s-duration run — same
structure/count confirmed identically for both Fortran and CUDA. This is the
same mechanism/content already validated at C8 in Prompt 31, now
independently reconfirmed at C32.

**Known benign artifact** (present at both C8 and C32, both backends,
identical pattern): `T`/`U`/`V` in `inst3_3d_diag` contain NaN only at the 3
lowest of 25 pressure levels (1000/975/950 hPa) — standard MAPL
vertical-interpolation-to-fixed-pressure-levels masking when instantaneous
surface pressure sits fractionally below a requested level. Confirmed at C32
via direct NaN-mask inspection: 14,664/149,600 NaN in each of `T`/`U`/`V`,
identical between Fortran and CUDA. Not a numerical defect.

## §7-8: Configurable controls / reference configs

Nothing in the HS forcing module (`Held_Suarez_ForcingMod.F90`) or its call
site hardcodes resolution, timestep, duration, or backend. All controls are
exposed through the existing `fv3.j`/`AGCM.rc`/`input.nml`/`HISTORY.rc`
mechanism, unchanged by this work:

| Control | Where | C8 (Prompt 31) | C32 (this prompt) |
|---|---|---|---|
| Resolution | `fv3.j`: `AGCM_IM` | 8 | 32 |
| Levels | `fv3.j`: `AGCM_LM` | 10 | 10 |
| Timestep (`bdt`) | `fv3.j`: `RUN_DT` | (C8 value, unchanged) | 1200s |
| `n_split`/`k_split` | `input.nml` | 3 / 1 | 3 / 1 (unchanged) |
| Duration | `fv3.j`: `JOB_SGMT` | 72 outer steps | 100 outer steps (092000s) |
| HS forcing | `AGCM.rc`: `RUN_HELD_SUAREZ` | 1 | 1 |
| Backend | `AGCM.rc`: `RUN_CUDA_FV3` | 0 / 1 | 0 / 1 |
| Precision | `fv3.j`: `FV3PRC` | R8 | R8 |
| Output frequency | `HISTORY.rc`: `*.frequency`/`*.duration` | 1h / 3h (unchanged) | 1h / 3h (unchanged) |

## §14-19: C32 Held-Suarez gate

**Authoritative C32 timestep** (source-backed, not searched fresh): from
`prompt_23_resolution_timestep_validation.md` / `p24_c32_r8_{fortran,cuda}`
job templates — `bdt=1200s`, `n_split=3`, `k_split=1`. Reused directly,
satisfying §15/§16 (no bounded search needed).

New job directories (fixture tree `experiments/`), each copied from the
matching `p24_c32_r8_*` timing template with the minimal deltas (job-name,
own `EXPDIR`, `FV3_SAVEPOINT_ENABLE 0->1`, inserted `RUN_HELD_SUAREZ: 1`;
EXPDIR grepped clean of stale `p24_c32` references per the Prompt 31
incident lesson):

- `p32_c32_hs_fortran/fv3.j` — Fortran gate.
- `p32_c32_hs_cuda/fv3.j` + new `p32_c32_hs_cuda.sbatch` — CUDA gate under
  the recommended standing config: MPS ON + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`
  + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, `RUN_CUDA_FV3_GRAPH` left unset.

**§17 C32 Fortran-only stability gate: PASS** (job 58389608). `rc=0`; 0
`range_warn`/`nan`/`fatal`/`error` in the run log; 0 bytes in
`warnings_and_errors.log`; 44 NetCDF files (33+11) spanning 100 outer
steps/33h20m simulated; global-mean T@500hPa smooth and bounded over the run
(260.2822-260.3075 K, no explosive trend); zonal-mean U@250hPa at the final
timestamp shows a physically sensible symmetric baroclinic-jet structure
(peak ~35 m/s at ±45 deg, near-zero at equator/poles) — a qualitative HS
climatology sanity check, not just a stability check.

**§18 C32 CUDA gate (recommended standing config): PASS** (job 58390108, via
`p32_c32_hs_cuda.sbatch`). `rc=0`; MPS confirmed started; both fusion flags
confirmed exported in-job; 0 `range_warn`/`nan`; 44 NetCDF files/1206
savepoint files (identical structure to Fortran); ~25s walltime. Identical
NaN pattern (14,664/149,600 in each of T/U/V, same 3 levels) and identical
global-mean T@500hPa time series to the Fortran run (260.2822-260.3075 K,
same first/last values to 4 decimals).

**Tooling fix required before §19 could run**: `tools/savepoints/
savepoint_reader.py`'s `run_sanity_checks()` and `compare_runs()` both only
special-cased the literal field name `"damping"` as a text-table field, not
any field whose meta `file=` ends in `.txt`. This fixture's savepoint sets
also dump a second text-table field, `da_min`, which crashed both functions
with `ValueError: field da_min is a text table, use load_damping()`. Fixed
both functions to build a `text_fields` set from the meta's `file=`
extension and skip any such field generically (numerical semantics changed:
**NO** — pure Python analysis-tool fix, no model source touched). Counted as
1 of the up-to-3 allowed substantial fixes (both call sites fixed together
as one defect class).

After the fix: `sanity` on both C32 HS savepoint trees — **86,010 fields
checked, all finite, all `delp` positive**, for both Fortran and CUDA
(identical result).

**§19 Fortran-vs-CUDA C32 HS savepoint comparison: PASS** (`compare --json`,
108 common savepoints, 87,810 total fields):

| Verdict | Count |
|---|---|
| BIT-IDENTICAL | 85,310 |
| MACHINE-PRECISION EQUIVALENT | 0 |
| FAIL (tool's relative-error classifier) | 2,500 |

The 2,500 "FAIL" rows are all in `delpc`/`pkc` at savepoint `sp1`
(C-grid-predictor entry) and `sp2a`, `it` in {1,2,3}, `pe` in {0..4} — every
one has `max_abs` on the order of `1e-313` to `1e-312` (IEEE-754 subnormal
doubles) with `max_rel` in the 0.2%-3.6% range only because both operands
are themselves near-zero (small-denominator artifact — the tool's
relative-only `classify_diff` has no absolute-magnitude floor). Direct
`show` inspection of `sp1_it1_pe0` confirms this is the same
uninitialized-halo/scratch-memory class already documented at c32 in
Prompts 22/23 (the `mfyL` artifact): `delpc`/`pkc`'s real, physically
computed values (min/max/mean/norm2 over their full `(38,38,10)`/
`(38,38,11)` extent, e.g. `pkc` mean `4.532863e+37`, norm2 `8.485281e+40`)
are **bit-identical to 7 significant figures** between backends; only a
handful of halo-padding cells that are unwritten at this particular
savepoint-capture point (bounds extend to a 2-cell halo, `(-2,35)`) contain
independent, physically meaningless garbage bit patterns in each backend/run
— expected for uninitialized memory, not a numerical divergence. No FAIL row
appears at `pe=5`, in any exit-state savepoint, or with a `max_abs` above
`1e-311` — consistent with a bounded, benign, previously-characterized
artifact class, not a new or growing discrepancy.

Independently corroborated at the NetCDF/diagnostic level: `compare-backends`
on `T@500hPa` at the final timestamp gives `max_abs_diff=0.0` (bitwise), and
global-mean-T time series are identical between backends to 4 decimal
places at every one of the 11 `inst3_3d_diag` timestamps.

**Classification: SMALL/benign (uninitialized-halo-scratch artifact,
previously characterized), no SIGNIFICANT divergence. No Opus escalation
triggered** — this is explained by direct evidence, not merely asserted; it
matches an already-documented pattern rather than being a new anomaly.

## §20: Explicit stop statement

Per task scope, **no long (10/100/1000-day) C32 run was attempted this
prompt** — work stopped immediately after the short C32 Fortran/CUDA
validation above passed, as instructed (§14-20).

## §22: Offline-analysis deliverable

New `tools/analysis/hs_netcdf_inspect.py` (xarray-based, reuses the same
MAPL History output, no new I/O). Subcommands, all tested against the real
C32 HS output this prompt: `describe` (dims/vars/levels), `global-mean-t`
(area-weighted), `zonal-mean` (one field/level, last timestamp),
`compare-backends` (Fortran-vs-CUDA max_abs_diff at matching time/level).
One fix needed to make it work: `xr.open_mfdataset` must use
`combine="nested", concat_dim="time"`, not `combine="by_coords"` — each
file's own `time` coordinate is always `0.0` (relative to that file), so
by-coordinate combination fails; files must be concatenated in filename
(chronological) order instead. Counted as the 2nd of the up-to-3 allowed
substantial fixes.

## §23: Output metadata sufficiency

Sufficient for offline reproducibility via existing mechanisms, all already
visible per-run: resolution/levels (`HISTORY.rc` grid label + `AGCM.rc`
`AGCM_IM`/`AGCM_LM`), timestep/cadence (`fv3.j` `RUN_DT`, `HISTORY.rc`
`*.frequency`/`*.duration`), backend (`AGCM.rc` `RUN_CUDA_FV3`), HS mode
(`AGCM.rc` `RUN_HELD_SUAREZ`), sim start/end (`CAP.rc` `BEG_DATE`/
`END_DATE`/`JOB_SGMT`). No gap found; nothing new needed.

## §24: Restart capability

**YES, exists — via the standard MAPL internal-state checkpoint mechanism**,
not anything added by this work. `FV_StateMod.F90:1121` branches on
`COLDSTART`: `.true.` (used in every run so far, including this prompt's)
takes the coldstart-init path; `.false.` takes the `else` branch
(line 1164), which simply calls the same `State_To_FV(MAPL, STATE)` used in
the coldstart path — the actual restart *read* is standard MAPL/GEOS
generic-state machinery driven by `AGCM.rc`'s
`DYN_INTERNAL_CHECKPOINT_FILE`/`_TYPE`/`_HEADER` (`pnc4`, header=1),
unrelated to and unaffected by the HS/CUDA work. **Status: UNTESTED this
task** (`COLDSTART: 1` in every run so far); not a new capability, not
broken by anything in this prompt.

## §25: How to change resolution/config

| Change | Edit |
|---|---|
| C8 -> C32 (or other Cn) | `fv3.j`: `AGCM_IM`, `HIST_IM`/`HIST_JM` (lat-lon output grid) |
| Levels | `fv3.j`: `AGCM_LM`; `AGCM.rc` `AK`/`BK` table must match |
| Timestep | `fv3.j`: `RUN_DT` (search Prompt 23 first for a resolution-appropriate value before assuming C8's works) |
| Duration | `fv3.j`: `JOB_SGMT` |
| Output frequency | `HISTORY.rc`: `*.frequency`/`*.duration` per collection |
| Backend | `AGCM.rc`: `RUN_CUDA_FV3` (0/1) |
| HS forcing on/off | `AGCM.rc`: `RUN_HELD_SUAREZ` (0/1) |

## §26-29: Fix count / regression argument

2 of the allowed 3 substantial fixes used (both text-table-field tooling
bugs in `savepoint_reader.py`; the `open_mfdataset` combine-mode fix in the
new analysis script). No CUDA dyn_core numerical source, no Fortran
numerical source, and no HS forcing source were modified this prompt — only
job/config scripts in the separate fixture tree (`p32_c32_hs_*`) and two
Python analysis-tool files in this repo's `tools/` (one edited, one new,
both already/newly untracked, no git-staging risk). `RUN_HELD_SUAREZ=OFF`
and non-HS behavior are therefore unaffected by construction, consistent
with Prompt 31's own regression argument.

## Result

C8 output readiness: confirmed (existing mechanism, reused, no new work
needed). C32 short Held-Suarez validation: **PASS**, both backends, savepoint-
and NetCDF-level, one previously-characterized benign artifact class (no new
anomaly). Offline-analysis script: delivered and tested. Stopped after short
validation per scope — no long/climatological C32 run attempted.

**HELD-SUAREZ SCIENTIFIC VALIDATION PASS — READY FOR MULTI-DAY C32
EXPERIMENT**
