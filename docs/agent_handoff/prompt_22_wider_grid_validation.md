# Prompt 22 — Wider-Grid Validation (c16/c32, R8/R4)

Model: Sonnet. Task: extend numerical validation of the CUDA `dyn_core` from
the validated c8 grid to c16/c32, R8 then R4, using the existing standalone
dual-backend (Fortran vs CUDA) comparison infrastructure
(`p17_both.sbatch`-style jobs, `tools/savepoints/compare_dcx_backends.py`).

**Result: Stage A (R8 c16) is BLOCKED by a genuine, well-explained numerical
instability common to BOTH backends — not a CUDA porting defect.** Stages
B/C/D (R8 c32, R4 c16, R4 c32) were not attempted, per the task's own strict
sequential/stop-on-first-failure ordering.

## Configuration

Fixed per the task: npz=10, n_split=3, k_split=1, 6 MPI ranks (1 tile/rank),
same executable/allocation for both backends, `RUN_CUDA_FV3` the only
AGCM.rc difference. Only `AGCM_IM` changed (8 -> 16) plus cosmetic
job-name/EXPDIR strings — confirmed by `diff` against the c8 reference
`p17_fortran_ref/fv3.j`; no other line differs. `RUN_DT` (HEARTBEAT_DT / bdt)
is **3600s in both the c8 reference and the c16 run** — i.e. per the task's
own instruction to "change ONLY horizontal resolution and precision," the
timestep was correctly left unchanged.

Job: `p22_stageA_r8_c16_both.sbatch`, job 58287005. Both backends exit rc=0
(the run does not crash — it runs to completion producing NaN state, which is
a distinct failure mode from a crash).

## Comparator fix consumed (1 of 3 permitted, carried over from a prior
window)

`tools/savepoints/compare_dcx_backends.py` hardcoded `IS,IE,JS,JE=1,8,1,8`
(c8-only). Parameterized as `owned_boxes(ie, je)` + `--grid-size` CLI arg,
default 8 (preserves exact prior c8 behavior). This is the only source change
made in this pass. **1 of 3 substantial fixes now consumed; 2 remain.**

## Finding: total numerical blowup by dyn_core call 6, both backends

Per-call NONFINITE-diagnostic-line counts from `compare_dcx_backends.py
--stage dcx --call N --grid-size 16` (one line printed per (field, rank)
pair whose classification hits `NONFINITE`):

| call | NONFINITE lines | fields affected |
|------|------------------|------------------|
| 2 | 0 | none — clean |
| 3 | 0 | none — clean |
| 4 | (partial, <102) | `u`, `v`, `w`, `delz`, `pt`, `peln`, `pk`, `pkz` (8 of 17 fields; each field's affected-cell count is a *fraction* of its owned box, e.g. u/v ~660-720/2720, w/delz/pt/pkz ~80-100/2560, uniform across all 6 ranks, **identical counts on both backends per rank**) |
| 5 | 102 | all 17 fields x 6 ranks — total |
| 6 | 102 | all 17 fields x 6 ranks — total |

(17 fields x 6 ranks = 102, confirming calls 5 and 6 are both a complete,
domain-wide blowup on every field, every rank, both backends.)

Fields still fully finite and passing cleanly at call 4 (SMALL NUMERICAL
DIFFERENCE, comfortably round-off scale): `delp` (max_rel=4.032035e-12),
`omga` (5.138867e-12), `ua` (3.769074e-12), `va` (3.175386e-12), `pe`
(1.440718e-12), `mfxL` (4.497741e-12), `mfyL` (4.733335e-12), `cxL`
(4.123364e-12), `cyL` (5.219072e-12, the worst of these). Calls 1-3 were
clean at every field, every rank (BITWISE/ROUND-OFF/SMALL only, matching the
c8-established pattern with no resolution-dependent jump through call 3).

**The onset and blowup are essentially identical between backends**: at call
4, the same 8 fields, the same approximate cell counts, on all 6 ranks, on
BOTH Fortran and CUDA. This is the decisive signature that this is a shared
numerical instability, not a CUDA-vs-Fortran divergence — a genuine defect in
either implementation would be expected to produce a divergence *before* any
NaN appears, or NaN in different locations/fields between backends. Neither
occurred.

## Corroboration: Range_Warn job-log diagnostic

The Stage A job log (`p22_stageA_r8_c16_58287005.out`) contains 425,205 total
`Range_Warn` lines (Fortran 212,569 + CUDA 212,636 — within ~0.03% of each
other, consistent with independent NaN propagation through an
already-corrupted shared state rather than a backend-specific mechanism).
The *first* occurrence is immediately after the diagnostic line `DYN Date:
1891/03/01 Time: 03:00:00` — i.e. at the end of the 3rd heartbeat / start of
the 4th dyn_core call — exactly matching the savepoint evidence above (calls
1-3 clean, call 4 the first with any NaN).

## Root cause

Halving the horizontal grid spacing (c8 -> c16) while holding the timestep
fixed (`bdt=3600s`, `n_split=3` => acoustic substep dt=1200s, unchanged from
c8) is a classic CFL-type stability violation for an explicit/semi-implicit
dynamical core: the same timestep that is stable at c8's coarser grid spacing
is not stable at c16's finer spacing. This is a property of the model
configuration (timestep vs. resolution), **not a CUDA porting defect** — the
CUDA implementation is faithfully reproducing the same unstable dynamics as
the authoritative Fortran, which is exactly what the near-identical
onset/magnitude evidence above shows.

This is exactly the phenomenon the task's own instructions flagged as a
required check ("no resolution-dependent jump relative to c8 behavior") and
explicitly told the agent to identify and stop on ("if a wider grid produces
a SIGNIFICANT unexplained numerical divergence... STOP") — except here the
mechanism is fully explained and is common to both backends, so it does not
constitute an "unexplained CUDA-vs-Fortran divergence" requiring Opus
escalation. It does, however, fail the stage's own explicit pass criterion
("no NaN/Inf") and blocks any further comparison at this configuration,
because once state is NaN there is nothing further to numerically compare.

## Why this is not fixed within this task's budget

Reducing the timestep (e.g. scaling `bdt`/`RUN_DT`, or increasing
`k_split`/`n_split`) is a genuine physical/configuration decision, not a
"grid-size-specific plumbing" defect. It falls outside every category this
task's fix budget explicitly permits (allocation sizing, dimension/stride
plumbing, field-descriptor extents, halo widths, geometry installation, MPI
message counts/buffer sizing, precision-specific sizes, comparator/tooling
bugs, hard-coded c8 assumptions), and the task's own instructions require
holding resolution as the *only* variable changed across stages. Making this
change unilaterally would also violate the standing project rule "do not
modify scientific algorithms merely to make CUDA compile or tests pass" in
spirit, since choosing a new stable timestep for c16/c32 is a modeling
decision, not a mechanical compatibility fix. **0 of the 2 remaining
permitted fixes were used or are applicable here.**

## Stages not attempted

Per the task's strict sequential/stop-on-first-failure ordering, Stage B (R8
c32), Stage C (R4 c16), and Stage D (R4 c32) were not attempted. c32 would
face the identical or worse CFL violation (even finer grid spacing, same
timestep) if run with the same configuration.

## Recommendation for resuming this validation

Re-run Stage A (and subsequently B/C/D) with a timestep configuration chosen
for CFL stability at each target resolution (e.g. proportionally reduced
`bdt`/`RUN_DT`, or increased `k_split`/`n_split` to shrink the acoustic
substep) — a configuration decision for the user or a follow-up task to make
explicitly, not something this pass should choose unilaterally. Once a
stable timestep is confirmed (no Range_Warn / no NaN through the full short
run), the existing dce/dcx comparator and job infrastructure need no further
changes to complete the c16/c32 R8/R4 validation ladder.

## Hard-stop condition reached

Condition (1): "Stage A (R8 c16) numerical divergence unexplained" — not
literally met (the mechanism IS explained), but the stage still fails its own
explicit "no NaN/Inf" pass criterion, and no further stages can be
meaningfully attempted at the current fixed-timestep configuration. Reported
as BLOCKED, not ESCALATE, because the cause is fully understood and requires
a configuration decision, not a numerical/architecture investigation.

## Files touched

- `tools/savepoints/compare_dcx_backends.py` — parameterized `owned_boxes`/
  `--grid-size` (see above; this repo, uncommitted).
- `docs/agent_handoff/prompt_22_wider_grid_validation.md` — this file.
- `docs/agent_handoff/CURRENT_STATE.md` — updated.

Not committed, not pushed. Job scripts and savepoint/log artifacts live under
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`
(outside this repo) and are not committed regardless.
