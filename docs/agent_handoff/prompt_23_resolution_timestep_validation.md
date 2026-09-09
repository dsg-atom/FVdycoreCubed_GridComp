# Prompt 23 — Resolution-Appropriate Timestep + Wider-Grid Validation

## 1. Task

Prompt 22 (R8 c16, `bdt=3600` unchanged from c8) hit a shared Fortran+CUDA
CFL-type NaN blowup by dyn_core call 5/6 — a configuration issue (timestep
too large for the halved grid spacing), not a CUDA defect. This task resolves
that blocker: determine a source-backed, resolution-appropriate timestep for
c16 and c32, gate each resolution in Fortran alone first, then re-run the
Fortran-vs-CUDA numerical comparison at each resolution/precision, in strict
order (c16 before c32, R8 before R4).

No CUDA/Fortran numerical algorithm source was touched this task. Only
job/config directories (copies of existing `fv3.j` templates with `RUN_DT`/
`AGCM_IM`/`JOB_SGMT` edited) and sbatch drivers were created.
**Numerical algorithm changed: NO.**

## 2. Authoritative Timestep Evidence (Step 1/7 — prefer existing tested config)

Found in pre-existing Prompt-7-era performance baseline fixtures under
`GEOSfvdycore_fortran_perf/experiments/`, each a previously-run, clean
(0 NaN, 0 Range_Warn) multi-hundred-call Fortran-only job:

| Resolution | `fv3.j` source | `RUN_DT` (bdt) | `n_split` | `k_split` | Effective acoustic dt (`bdt/n_split`) | Prior clean run length |
|---|---|---|---|---|---|---|
| c8  | `c8_L10_dyncore_perf/fv3.j`  | 3600 s | 3 | 1 | 1200 s | 168 dyn_core calls, 0 NaN/Range_Warn |
| c16 | `c16_L10_dyncore_perf/fv3.j` | 1800 s | 3 | 1 | 600 s  | 336 dyn_core calls, 0 NaN/Range_Warn |
| c32 | `c32_L10_dyncore_perf/fv3.j` | 1200 s | 3 | 1 | 400 s  | 504 dyn_core calls, 0 NaN/Range_Warn |

This is an existing, previously-tested rule (`RUN_DT` halves each time
`AGCM_IM` doubles, `n_split`/`k_split` held fixed) — no bounded trial-and-error
search (Step 3) was needed for either resolution. Per Step 2, `bdt` (outer
dyn_core-call timestep) and the acoustic substep dt (`bdt/n_split`) are
distinct; both scale together here since `n_split` is held fixed at 3
throughout — only `bdt` was varied, consistent with the existing configs and
with Step 4's preference not to vary both without evidence.

## 3. Job Construction

For each stage, a matched Fortran/CUDA directory pair was created by copying
the nearest existing template and editing only `RUN_DT`, `AGCM_IM` (already
correct in most templates), `JOB_SGMT` (shortened to hold dyn_core-call count
at 6, matching Prompt 22's depth), and cosmetic job-name/EXPDIR strings.
Diffed each fortran/cuda pair before submission to confirm the only
substantive difference is `RUN_CUDA_FV3: 0` vs `1`.

- Stage A: `p23_stageA_r8_c16_{fortran,cuda}` — R8, c16, `RUN_DT=1800`,
  `JOB_SGMT=00000000 030000` (3h → 6 calls). Job 58294889.
- Stage B: `p23_stageB_r8_c32_{fortran,cuda}` — R8, c32, `RUN_DT=1200`,
  `JOB_SGMT=00000000 020000` (2h → 6 calls). Job 58294931.
- Stage C: `p23_stageC_r4_c16_{fortran,cuda}` — R4, c16, `RUN_DT=1800` (reused
  from Stage A per Step 10 — no separate R4 search needed). Job 58294962.
- Stage D: `p23_stageD_r4_c32_{fortran,cuda}` — R4, c32, `RUN_DT=1200`
  (reused from Stage B). Job 58295046.

All four jobs: 6 MPI ranks (1x6, one tile per rank), `gpu_a100`/`rome`, single
A100, `npz=10`, production first-run numerics (`hord_mt=hord_vt=hord_tm=6`,
`hord_dp=-6`, `vtdm4=0.01`, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`).

## 4. Fortran Stability Gate (Step 5/8) — all four PASS

| Stage | Backend | rc | Range_Warn | NaN | dyn_core calls reached |
|---|---|---|---|---|---|
| A (R8 c16) | fortran | 0 | 0 | 0 | 6 |
| A (R8 c16) | cuda    | 0 | 0 | 0 | 6 |
| B (R8 c32) | fortran | 0 | 0 | 0 | 6 |
| B (R8 c32) | cuda    | 0 | 0 | 0 | 6 |
| C (R4 c16) | fortran | 0 | 0 | 0 | 6 |
| C (R4 c16) | cuda    | 0 | 0 | 0 | 6 |
| D (R4 c32) | fortran | 0 | 0 | 0 | 6 |
| D (R4 c32) | cuda    | 0 | 0 | 0 | 6 |

`delp` confirmed positive and finite at every rank, calls 1/3/6, dcx stage,
for all four stages (direct savepoint inspection, not just log absence of
Range_Warn).

## 5. Numerical Comparison Results (Step 6/9)

Ran `tools/savepoints/compare_dcx_backends.py` at `--stage {dce,dcx}`,
`--call {1,3,6}`, `--grid-size {16,32}` for each stage. No comparator source
was modified (its `--grid-size` parameterization was already added in the
Prompt 22 pass).

### Stage A — R8 c16: PASS

Clean ROUND-OFF/SMALL-scale agreement at every checkpoint; worst max_rel
≈6.7e-11 (`w`, dcx). No SIGNIFICANT-DIFFERENCE fields by R8 thresholds.

### Stage B — R8 c32: PASS (one explained, non-blocking anomaly)

Clean agreement at every real (dcx exit-state) checkpoint. One anomaly found
and root-caused: `mfyL` at the **`dce` (entry) stage only**, c32, rank pe0,
2 of 10560 cells — genuinely uninitialized memory in both backends
independently (one cell has an identical garbage value in both backends;
the other has unrelated garbage values in each), unchanged bit-for-bit
across calls 1/3/6, and absent from the analogous `mfxL` field and from the
real `dcx` exit savepoint (where `mfyL` legitimately reaches ~1e12-1e13 near
the model top in both backends, agreeing to ~12-13 significant figures —
confirmed this is real matching physics, not a bug: the comparator's
`max_abs`/`max_rel` are the cross-backend *difference*, not the raw field
magnitude). Classified as a benign entry-snapshot artifact on a
non-primary flux accumulator, not a CUDA-specific defect (present
independently in both backends) and not a real numerical divergence
(the actual computed/exit state matches cleanly). `delp` positive/finite
confirmed at all ranks/calls.

### Stage C — R4 c16: PASS (explained small-denominator artifact)

Field-by-field agreement is SMALL-scale to ROUND-OFF for every field except
`w` (and, at c32 below, `omga`), which repeatedly carries the largest
max_rel: 3.67e-2 to 6.80e-2 across dce/dcx calls 1/3/6. Directly inspected
`w`'s raw values at every reported worst-mismatch point: absolute
disagreement is consistently tiny (1.7e-6 to 4.1e-4) while `w` itself is a
small-magnitude field (bulk `min/max/absmean` ≈ -3.7e-3/6.1e-3/4.7e-4,
matching closely between backends). The large max_rel is a small-denominator
ratio artifact on a small-magnitude, sign-changing diagnostic field — the
same class of behavior already documented for `omga`'s R4 bulk-metric growth
in `CURRENT_STATE.md` §5 — not a new CUDA-specific divergence. No growth
trend across calls 1→3→6 (values oscillate, do not monotonically increase).
`delp` positive/finite confirmed at all ranks, calls 1/6.

### Stage D — R4 c32: PASS (same explained artifact, larger ratio)

Same pattern as Stage C, same field (`w`, and secondarily `omga`), larger
worst-case ratio: `w` max_rel reaches 2.39e-1 at dce call=3 (16.5e-2 at dcx
call=3, 7.65e-2 at dcx call=6); `omga` reaches 2.56e-2 at dcx call=6. Directly
inspected raw values at each worst-mismatch point: absolute disagreement
stays bounded (3.3e-6 to 5.9e-5 for `w`; 5.9e-5 for `omga`), bulk field
statistics (`min/max/absmean`) match closely between backends at every
checkpoint, and both fields' own magnitude at the worst-mismatch cell is
small relative to their typical range — the same denominator-driven artifact
as Stage C, scaled up because the finer c32 grid resolves smaller local
magnitudes (more near-zero-crossing cells for these two fields). `delp`
positive/finite confirmed at all ranks, calls 1/3/6. No NaN/Inf anywhere.
This is the least favorable numeric margin of the four stages and is called
out explicitly (not hidden) — recommend watching this field pair if c32 is
later used for extended-duration or moist-physics validation, but it does not
meet the interpretation policy's "CUDA diverges while Fortran is stable"
failure signature: both backends are equally stable, agree in absolute terms
and bulk statistics, and the pattern is identical in kind (not new) to
already-documented R4 diagnostic-field behavior.

## 6. Interpretation (per task's Step 11 policy)

All four stages: **Case C** — both backends stable, numerically close.
The Stage B `mfyL`-entry and Stage C/D `w`/`omga` findings are explained,
non-propagating, present-in-both-backends artifacts (uninitialized-memory
snapshot and small-denominator ratio, respectively), not unexplained
SIGNIFICANT differences and not CUDA-specific defects. **No STOP/escalation
triggered** — at no stage was Fortran stable while CUDA alone diverged or
was unstable.

## 7. Scope / Constraints Honored

- Sonnet-only throughout; no escalation was warranted.
- Existing tested timestep configs reused; no bounded search needed at
  either resolution.
- c16 before c32; R8 before R4; k_split=1 throughout; short (6-call) runs
  only; existing comparator/job infrastructure reused unmodified (source);
  no CUDA algorithm/halo/formula changes; no performance work; no new
  savepoint format.
- No commits/pushes made. No logs/build outputs committed.

## 8. Result

Prompt 22's wider-grid blocker is resolved for the tested envelope: R8 and
R4 are both validated at c16 and c32 (npz=10, 6-call depth, matching Prompt
22's own depth). c8-only scope from Prompt 19-21 is now extended to c16/c32
at both precisions.
