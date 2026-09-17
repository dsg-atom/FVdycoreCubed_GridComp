# Prompt 35 — Held-Suarez Climatology / Restart / Statistical Comparison

Branch: `experiment/held-suarez-e2e`. Follows Prompts 31-34 (HS integration,
scientific validation, multi-day/high-res ladder, C32 perf reconciliation).
Model: Sonnet throughout this document's scope, except one Opus escalation
(restart-failure root cause, Stage 1, prior session — 1 of 4 allotted
triggers used). Debug/fix budget: 1 of 3 allotted substantial fixes used
(Stage 1 restart config-correction attempt, prior session). No numerical,
CUDA-kernel, or MPI-architecture source code was touched in this document's
scope. Three additional issues were found and corrected across this
session (EXPDIR staleness, JOB_SGMT digit-packing/NUM_SGMT chaining, and a
DUR_DD hard-fail crash) — all are job-configuration/tooling mistakes I
introduced, not numerical-source fixes, and are logged separately below
(not counted against the ≤3 budget, consistent with the Prompt 33
precedent for tooling-level fixes).

## Stage 1 — Restart continuity (prior session, recapped)

`COLDSTART: 0` warm-restart was exercised for the Fortran backend at C128
and **FAILED**: root-caused via Opus escalation to a genuine, confirmed
source-level gap in the nonhydrostatic warm-restart path for this HS
standalone configuration (not a CUDA-side issue, not a config mistake). The
corresponding CUDA restart test was correctly skipped once Fortran's own
restart failed — there is nothing to restart-validate against.

**Finding, to be treated as a standing negative result**: `COLDSTART: 0`
is **NOT VALIDATED / confirmed broken** for this HS/nonhydrostatic
standalone configuration. Do not use warm restart for HS production runs
until a dedicated future fix closes this gap. `CURRENT_STATE.md` §9
("MAPL restart... never operationally exercised") should be updated to
reflect this negative finding explicitly rather than "unexercised."

**Decision** (also prior session, carried forward): abandon
restart-continuation as the mechanism for extending run duration; use fresh
from-scratch (`COLDSTART: 1`) runs of the desired total length instead.

## Stage 2 — 30-day matched Fortran/CUDA climatology runs

Built `p35_clim30_fortran`/`p35_clim30_cuda` (C128/L10/R8, 6 MPI ranks, HS
forcing, standing MPS+fusion CUDA config, `JOB_SGMT='00000030 000000'`
i.e. 30 literal days — this encoding was safe, see Stage 4 below for why).
Jobs 58421485 (Fortran)/58421486 (CUDA): both **COMPLETED**, `rc=0`,
1h49m/1h42m elapsed. Health checks: 0 NaN/error hits; only the
previously-documented benign `TA_dyn` range_warn artifact (§9a of
`CURRENT_STATE.md`), matching counts both backends; byte-identical log
sizes; 240 nc4 files each (30 days x 4/day x 2 collections), as expected.

`hs_netcdf_inspect.py compare-backends` at day 1/5/10/20/30 checkpoints on
T@500hPa, U@250hPa, PS: **bitwise identical** at every checkpoint
(`max_abs=0.0 rms=0.0`) — no backend decorrelation over 30 days, consistent
with this configuration's established deterministic bitwise-equivalence
property (Gate C2 and every subsequent HS validation stage).

**Stage 2: CLOSED — PASS.**

## Stage 3 — Scientific diagnostics / spin-up classification

Extended `tools/analysis/hs_netcdf_inspect.py` with a new `eke-proxy`
subcommand (transient EKE, `0.5*<(U-Ubar)'^2+(V-Vbar)'^2>` over a
time-mean window; no regridding needed since History output is already on
the lat-lon grid) — a legitimate extension of existing tooling, not new
infrastructure, per project reuse policy.

Ran the full required diagnostic suite against the Fortran 30-day run:
global-mean T, KE-proxy timeseries, jet-metrics (timeseries + scan),
zonal-mean T/U, EKE-proxy.

**Finding: day 30 is an active spin-up/transient state, not equilibrated
climatology.** Evidence: KE-proxy nearly triples from ~52 to ~163 m^2/s^2
through day ~24 before plateauing/slightly declining (still not flat by
day 30); jet strength and latitude are still visibly drifting through days
27-30; pronounced NH/SH jet-latitude asymmetry (53.44 deg vs -75.94 deg) far
from the expected near-symmetric equilibrium Held-Suarez climate. This is
consistent with the HS literature's typical ~200+ day equilibration time
from a near-zero-eddy initial condition, and is the expected behavior at
this duration — not a defect.

**Stage 3: CLOSED.** Statistical climate comparison across backends is, in
this exact deterministic configuration, trivially and fully satisfied by
the Stage 2 bitwise-agreement result for any duration (there is no
backend-divergence phenomenon to characterize statistically); the
remaining diagnostic value is in characterizing the single-backend
spin-up/climate state itself, which this stage did.

## Stage 4 — 100-day extension: two self-caused job-configuration bugs, found and fixed

### Bug 1 — stale `EXPDIR` when cloning job directories

Cloned `p35_clim100_{fortran,cuda}` from the 30-day directories via
`rsync --exclude=savepoints --exclude='scratch_*'` plus targeted `sed`
edits. The `sed` correctly updated the `JOB_SGMT` csh variable and the
`.sbatch` wrapper's job-name/output-path/`cd` target, but **did not**
update the `setenv EXPDIR ...` line inside `fv3.j` itself (line 161),
which still pointed at the old `p35_clim30_{fortran,cuda}` paths. Since
`EXPDIR`/`SCRDIR` (not the sbatch's own `cd` target) govern where the run
actually writes its scratch directory, logs, and RC files, both "100-day"
jobs (58422382/58422383) silently wrote all real output into the OLD
30-day directories instead of the new ones — the intended
`p35_clim100_{fortran,cuda}` directories ended up containing only `fv3.j`,
no output.

**Consequence for the Stage 2/3 dataset**: because the erroneous run used
the *same* `BEG_DATE`/config/`COLDSTART: 1` as the original 30-day run, the
first 30 days of its output are deterministically bit-identical to the
already-completed, already-analyzed 30-day dataset (same IC, same
numerics, same backend) — this was a redundant re-computation of the same
30 days, not a corruption, followed by additional days appended on top.
The Stage 2/3 results already captured in this document are therefore
unaffected in content. However, the `p35_clim30_{fortran,cuda}` scratch
directories no longer cleanly represent "exactly 30 days" (they were
extended to 31 days by the accidental second run, per Bug 2 below, and
their `cap_restart` file now reflects the later date) — do not reuse those
directories as a clean 30-day reference going forward; the printed
Stage 2/3 results in this document remain the authoritative record.

**Fix**: rebuilt `p35_clim100_{fortran,cuda}` fresh, this time correcting
`EXPDIR` via `sed` to point at the new directories (verified via
`grep -n "setenv EXPDIR"` before submission).

### Bug 2 — `JOB_SGMT` is a calendar YYYYMMDD-shaped duration, not a raw day count

Independently, the first "100-day" attempt (jobs 58422382/58422383) both
completed cleanly (`rc=0`, `ALL DONE`, clean `EGRESS`/`FINALIZE`) but only
executed **31 simulated days** (1891-03-01 through 1891-04-01, confirmed
via both the 248 nc4 filenames — 31 days x 4/day x 2 collections — and the
FMS-clock cycle count: 35,712 total DYN/EXTDATA cycles = 2 x 17,856
heartbeats x 150s/heartbeat = 31.0 days), despite `CAP.rc` correctly
showing `JOB_SGMT: 00000100 000000` / `NUM_SGMT: 1`.

Root cause, confirmed against the actual MAPL source
(`src/Shared/@MAPL/gridcomps/Cap/MAPL_CapGridComp.F90`,
`src/Shared/@MAPL/base/Base/Base_Base_implementation.F90`, in the
standalone Fortran fixture tree — not this repo, read-only inspection):
`JOB_SGMT:` is parsed into a 2-element integer array and unpacked via
`MAPL_UnpackDateTime`, which extracts
`dd = mod(date_time(1), 100)`, `mm = mod(date_time(1), 10000) / 100`,
`yy = date_time(1) / 10000` — pure decimal digit-group arithmetic on the
integer value of the RC string, then feeds `yy`/`mm`/`dd` into
`ESMF_TimeIntervalSet(duration, YY=DUR_YY, MM=DUR_MM, D=DUR_DD, ...)`. For
`'00000030 000000'` (integer 30): `dd=30, mm=0` — an unambiguous 30 days.
For `'00000100 000000'` (integer 100): `dd=mod(100,100)=0`,
`mm=100/100=1` — this is **structurally forced**, not a one-off
digit-placement mistake: `dd = mod(x,100)` can never exceed 99 for *any*
input, so writing 100 (or more) into this field always spills the excess
into `mm`, and `mm` months is a **calendar-dependent** span (28-31 days
each) applied via `ESMF_TimeIntervalSet`'s calendar arithmetic — hence
exactly 31 days for "1 month" starting March 1. This field cannot express
100 (or any value >= 100) literal days at all; there is no valid encoding
of `JOB_SGMT` that avoids the month rollover once the day count reaches
100.

**Second fix attempt (also wrong)**: initially worked around this by using
the already-validated 10-day encoding (`JOB_SGMT='00000010 000000'`) with
`NUM_SGMT: 10` in `CAP.rc`, assuming MAPL would internally chain 10
segments within one job invocation. Submitted as jobs 58424040/58424041;
both completed quickly (36min/34min, suspiciously fast) and on inspection
had run **only 10 days** (80 nc4 files, `cap_restart=18910311`) — MAPL's
`NUM_SGMT` did not loop internally as assumed. More importantly, this
approach was conceptually unsound regardless of whether it had "worked":
chaining multiple segments via `cap_restart` **is** a restart mechanism,
and Stage 1 already confirmed warm restart is broken for this HS/NH
configuration — so even a working multi-segment chain would have risked
hitting the same confirmed gap. Multi-segment chaining is not a safe path
for this configuration under any encoding.

**Third fix attempt (also wrong — crashed immediately)**: bypassed
`JOB_SGMT` entirely, replacing it in the `CAP.rc` heredoc with
`DUR_DD:       100` (reasoning at the time: the unpacked `DUR_DD` resource
item has no `mod`/digit-packing bound, so it should give exactly 100 days).
Submitted as jobs 58424490 (Fortran)/58424491 (CUDA). **Both crashed almost
immediately** (9s/18s elapsed, `ExitCode 0:0` in `sacct` was misleading —
see the standing lesson below): the `.out` logs showed
`pe=00000 FAIL at line=01605 MAPL_CapGridComp.F90 <status=-1>` cascading
through `MAPL_Cap.F90`/`StandAlone_FV3_Dycore.F90` to `MPI_Abort`. Line
1605 is exactly the `JOB_SGMT:` lookup:
```fortran
call MAPL_GetResource( MAPLOBJ, datetime, label='JOB_SGMT:',     _RC )
if(STATUS/=ESMF_SUCCESS) then
   call MAPL_GetResource( MAPLOBJ, datetime, label='JOB_DURATION:', _RC )
end if
```
and `_RC` (`MAPL_ErrLog.h`: `#define _RC_(rc,status) rc=status);_VERIFY(status`)
is a **hard-fail** assertion — it aborts immediately if the label is
missing, with no default. Removing `JOB_SGMT:`/`JOB_DURATION:` from
`CAP.rc` entirely does not fall through to reading `DUR_*` as a graceful
fallback; the apparent "fallback" code path is unreachable when the label
is absent altogether. **`JOB_SGMT:` (or `JOB_DURATION:`) must always be
present in `CAP.rc`.**

**Actual fix (fourth attempt)**: keep `JOB_SGMT:     ${JOB_SGMT}` in the
`CAP.rc` heredoc (satisfying the hard-fail lookup), but give it a
deliberately **oversized, unambiguous** value that safely exceeds the
target run length without tripping the `dd = mod(x,100)` cap — 1 year,
`JOB_SGMT='00010000 000000'` (integer 10000: `yy=1, mm=0, dd=0`; `dd=0`
never triggers the month-rollover ambiguity). Then constrain the *actual*
run length via `END_DATE`, changed from a far-future placeholder to
exactly 100 calendar days after `BEG_DATE='18910301 000000'`:
`END_DATE='18910609 000000'` (computed via
`datetime.date(1891,3,1)+timedelta(days=100)` = 1891-06-09). This relies on
`MAPL_CapGridComp.F90`'s own clamp (confirmed by reading the source, lines
~1719-1728):
```fortran
maxDuration = EndTime - currTime
if (duration > maxDuration) duration = maxDuration
stopTime = currTime + duration
```
`EndTime - currTime` is a genuine ESMF calendar-aware `Time` subtraction
(from `END_DATE`, a calendar date, not digit-packed like `JOB_SGMT`), so it
is immune to the `mod(x,100)` bug. Since the 1-year `JOB_SGMT` duration
safely exceeds the 100-day `maxDuration`, the clamp forces the actual run
to stop at exactly 1891-06-09 — precisely 100 days — regardless of
`JOB_SGMT`'s inability to directly encode "100" as a raw day count.
`NUM_SGMT: 1` stays unchanged (single cold-start segment, no chaining,
consistent with the Stage 1 restart-broken finding).

**Standing operational lesson (this stage)**: the `.sbatch` wrapper scripts
capture `rc=$?` after `csh -f ./fv3.j` but never `exit $rc` — so `sacct`'s
recorded `ExitCode` reflects the wrapper script's own exit status (from a
later `echo`), not whether the underlying run actually succeeded. The
third fix attempt's crash was caught only by noticing suspiciously short
elapsed times (9s/18s) and reading the `.out` log directly — `ExitCode
0:0` alone is not sufficient evidence of success for this job harness.

### Resubmission

Rebuilt `p35_clim100_fortran`/`p35_clim100_cuda`'s `fv3.j` in place (no
directory re-clone needed this time — `EXPDIR` was already correct from
the Bug 1 fix): restored `JOB_SGMT:     ${JOB_SGMT}` in the `CAP.rc`
heredoc, set `JOB_SGMT='00010000 000000'` (1 year) and
`END_DATE='18910609 000000'` (100 days after `BEG_DATE`). Verified via
`grep` before submission. Submitted as jobs **58424578** (Fortran) and
**58424579** (CUDA), same `--time=12:00:00` sbatch setting as before
(throughput estimate unchanged: ~6.1h/5.7h wall time, well inside 12h and
the platform's 24h hard ceiling).

**Stage 4: three configuration mistakes found and root-caused across four
fix attempts (two via direct MAPL/ESMF source inspection); the fourth
attempt is source-verified sound (calendar-based `END_DATE` clamp, immune
to the `JOB_SGMT` digit-packing bug) and has been submitted. Awaiting
completion — verification will be done by reading the `.out` log directly
and cross-checking elapsed wall time and nc4 file count/date span, not by
trusting `sacct`'s `ExitCode` alone (see lesson above) — before Stage 5
(100-day statistical comparison) can run.**

## Stage 5 — 100-day corrected runs: results

Jobs **58424578** (Fortran) / **58424579** (CUDA) confirmed COMPLETED via
`sacct` and independently verified by reading the `.out` logs directly (per
the Stage 4 lesson, not trusting `sacct` `ExitCode` alone): both show
`----FINALIZE` in the FMS timing table and a trailing `ALL DONE`, no
`FAIL`/`Abort`/`ERROR`/core-dump markers. Elapsed 6h00m24s (Fortran) /
5h38m18s (CUDA), both well inside the 12h request and matching the ~6.1h/
5.7h throughput estimate.

Run-length verification: `cap_restart` in both scratch dirs reads
`18910609 000000` (`BEG_DATE=18910301` + 100 days, matching the `END_DATE`
clamp), 800 nc4 files each (100 days x 4/day x 2 collections), 400 time
steps per collection — all as designed.

Health check: both backends' scratch logs are byte-identical in size
(14,275,392 B), 0 NaN hits, 0 `error` hits, and an identical `range_warn`
count (33,543 — the same benign `TA_dyn` near-surface-warmth artifact
characterized at C128/10-day in Prompt 33, now confirmed to persist
identically at 100 days).

Bitwise backend-comparison (`compare-backends`, Fortran vs CUDA) at day
1/5/10/20/30/50/75/100 on T@500hPa, U@250hPa, PS: **all eight checkpoints
bitwise identical** (`max_abs=0.0 rms=0.0`, 0 NaN either side, 8320 points
compared each). No backend decorrelation out to 100 days — consistent with
every prior HS validation stage's deterministic bitwise-equivalence
property.

Diagnostic suite (Fortran run, single-backend climate-state
characterization — statistically identical to CUDA per the bitwise result
above, so not duplicated):
- **Global-mean T@500hPa**: settled to a shallow, slowly-decaying plateau
  (258.4K at day 40 -> 257.9K at day 100, -0.5K over 60 days) vs. the
  still-actively-declining state at day 30 (258.3K, coming down fast from
  a day-10 minimum of 256.2K).
- **KE-proxy@850hPa**: peaked at day 30 (159.5 m^2/s^2) and has since been
  slowly relaxing (153.6 at day 40 -> 145.5 at day 100, a further -6% drift)
  — much flatter than the day-30 snapshot's description ("still not flat by
  day 30"), but not yet fully flat either.
- **Jet metrics, default all-levels scan (last timestamp, same command form
  as the day-30 Stage 3 result)**: day 100 gives NH lat=59.06 deg (150 hPa,
  U=41.331 m/s) vs. SH lat=-59.06 deg (150 hPa, U=41.718 m/s) — magnitude-
  symmetric across hemispheres. This is a marked improvement over the
  day-30 scan's pronounced 53.44N vs -75.94S asymmetry (§Stage 3): the
  large hemispheric skew present at day 30 has resolved by day 100.
- **Jet metrics (250 hPa, timeseries, a separate finer-grained view not
  taken at day 30)**: NH/SH track together and stabilize from day ~40
  onward (56.25N/-56.25S through day 70), drifting to a final 59.06N/
  -56.25S by day 90-100 — a one-grid-step (2.81 deg) residual asymmetry,
  likely within discretization noise of the 65-point lat grid. Jet
  strength (U) peaks near day 40-50 (~43.5 m/s) and decays slowly
  afterward (~39 m/s by day 100).
- **EKE-proxy@250hPa** (whole-run transient eddy KE, deviation from the
  400-step time-mean): global-mean 20.74 m^2/s^2 — a single summary number,
  consistent with an actively-eddying, not-yet-fully-steady jet.

**Finding**: day 100 is markedly closer to Held-Suarez equilibrium than day
30 (KE-proxy and global-mean-T trends both flattened by roughly 5-10x
relative to their day-30 rates of change; jet latitude/strength stabilized
by day ~40-70), but is **still not fully equilibrated** — KE-proxy and jet
strength both show a small continued decay trend through day 90-100. This
is consistent with the ~200+ day equilibration timescale cited in the HS
literature for a near-zero-eddy cold start; not a defect, and not evidence
of anything CUDA-specific (the backends are bitwise identical throughout,
so any residual spin-up transient is identical on both).

**Stage 5: CLOSED — PASS.** No numerical, CUDA-kernel, or MPI-architecture
issue found. Backend agreement remains bitwise through 100 days at C128.

## Overall Prompt 35 determination: **PASS**

Restart-continuation is confirmed broken (Stage 1, standing negative
result, not fixed in this document's scope) and from-scratch runs were
used instead (Stage 2/4 decision). Three job-configuration/tooling defects
were found and fixed (Stage 4), none numerical. The 100-day matched
Fortran/CUDA climatology run is bitwise-identical at every sampled
checkpoint and shows monotonically progressing (not diverging) climate
spin-up on both backends identically.

Next natural extension (not started, confirm scope before pursuing):
further duration toward the ~200+ day literature equilibration estimate,
or a dedicated fix for the Stage 1 warm-restart gap.

## Files touched

- `tools/analysis/hs_netcdf_inspect.py` — added `eke-proxy` subcommand
  (extension of existing tooling, not new).
- New (untracked, not committed unless requested), under
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`:
  `p35_clim30_{fortran,cuda}/` (30-day runs, now also containing an extra,
  redundant 31st day from Bug 1 — see caveat above), `p35_clim100_{fortran,cuda}/`
  (corrected 100-day runs, in progress) and their `.sbatch` wrappers.
- This file.
- `CURRENT_STATE.md` — to be updated once Stage 5 completes.
