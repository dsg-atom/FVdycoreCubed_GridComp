# Prompt 19 — Longer Standalone Validation

Model: Sonnet. No source code was modified in this pass (0 of 3 permitted
integration fixes consumed — none were needed). All artifacts live under
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/`,
outside this repository; nothing was committed or pushed.

## Scope

Configuration fixed throughout: c8, npz=10, n_split=3, R8, 6 MPI ranks (1 tile
per rank), same executable/allocation for both backends, `RUN_CUDA_FV3` as the
only AGCM.rc difference. Reused `p17_both.sbatch`'s mechanics and
`tools/savepoints/compare_dcx_backends.py` unchanged — no new validation
framework, no new savepoints.

## Stage A — longer k_split=1 run

- Constructed by extending `JOB_SGMT` to `'00000003 000000'` (3 days) with
  `k_split=1` unchanged, HEARTBEAT_DT=3600s ⇒ **72 dyn_core calls**.
- Both backends exit rc=0, no CUDA/MPI errors, no NaN/Inf.
- Entry (dce) call 1: 13/13 primary state fields BITWISE identical, 6/6
  ranks. One benign pre-first-write flux-capacitor anomaly (`mfxL`, 1/6 ranks,
  max_rel=1.0) — uninitialized scratch before `dyn_core` ever writes it; does
  not recur at any exit checkpoint.
- Exit (dcx) checkpoints sampled at calls 1, 6, 18, 36, 54, 72: every field
  ROUND-OFF SCALE or SMALL NUMERICAL DIFFERENCE at every checkpoint, **zero
  SIGNIFICANT differences**, zero NaN/Inf.
- Worst discrepancy over the whole run: `w`, max_rel=5.116168e-11 at call 72
  (SMALL). `peln`/`pk`/`pkz` stay FLAT at the round-off floor over the full
  range; `u`/`v`/`delp`/`pt`/`delz`/`ua`/`va`/`pe`/`omga`/`w` grow
  **SUBLINEAR** (roughly 5-12x growth over a 72x increase in call count),
  consistent with accumulated floating-point noise, not divergence.
- **Stage A: PASS.**

## Stage B — smallest supported k_split > 1

- **k_split = 2 chosen.** Source-backed reason: `fv3.j:373-374` sets
  `k_split`/`n_split` as plain scalar assignments (`k_split=1`/`n_split=3` by
  default); `k_split` is the trip count of `fv_dynamics.F90`'s
  `do n_map=1,k_split` loop, which calls `dyn_core` once per iteration with
  `mdt = bdt / real(k_split)`. `k_split=2` is the smallest value that exercises
  the loop more than once per heartbeat while keeping `n_split=3` and the rest
  of the production configuration unchanged — no ambiguity in the source (hard
  stop condition 2 did not trigger).
- Built `p19_stageB_{fortran,cuda}` by the same lightweight
  copy-executable + sed-`fv3.j` pattern as Stage A: changed only `k_split`
  (1→2), `JOB_SGMT` (3 days → 6 hours ⇒ 6 heartbeats x 2 dyn_core calls/
  heartbeat = **12 dyn_core calls total**), and cosmetic EXPDIR/job-name
  fields. Verified via `diff` that no other lines changed and via
  `grep -c RUN_CUDA_FV3` that the backend split is correct (0 in the Fortran
  copy, 1 in the CUDA copy).
- Job 58246786 (`p19_stageB_both.sbatch`) completed in 22s, both backends
  rc=0, no error/nan/fail/abort/segv strings, savepoint trees on both sides
  reach exactly `it12` — confirming 12 matched dyn_core calls on both
  backends.
- **Numerical comparison** (`compare_dcx_backends.py`):
  - Entry (dce) call 1: 13/13 primary fields BITWISE identical, 6/6 ranks.
    Same benign `mfxL` pre-first-write anomaly as Stage A (1/6 ranks).
  - Entry (dce) call 3 — the first dyn_core entry immediately after the
    first heartbeat's remap step (the remap boundary): all 13 fields
    ROUND-OFF SCALE or SMALL NUMERICAL DIFFERENCE, **zero SIGNIFICANT**,
    worst `w` max_rel=2.69e-11. No discontinuity relative to the
    exit-call-2 trend — the remap boundary introduces no anomalous jump.
  - Exit (dcx) calls 1, 2, 6, 12: every field ROUND-OFF SCALE or SMALL
    NUMERICAL DIFFERENCE at every checkpoint, **zero SIGNIFICANT
    differences**, zero NaN/Inf. Call 2 is the end of the first k_split
    sequence (first heartbeat) — no jump there relative to call 1.
  - Worst Stage B discrepancy: `w`, max_rel=3.527908e-11 at call 6 (SMALL) —
    smaller than Stage A's 72-call worst (5.116e-11), consistent with
    call-count-driven accumulation rather than a k_split-specific effect.
  - `peln`/`pk`/`pkz` FLAT at round-off floor across all 12 calls;
    `u`/`v`/`delz`/`delp`/`pt`/`pe`/`ua`/`va`/`omga` SUBLINEAR growth,
    same character as Stage A, no acceleration at the k_split=2/remap
    boundary.
  - `delp` positivity explicitly checked at call 12 via `load_field`
    (`tools/savepoints/savepoint_reader.py`): min 9957.5868 / 9962.5076 Pa,
    **identical to displayed precision on both backends**, 6/6 ranks.
  - `mfxL`/`mfyL`/`cxL`/`cyL` flux capacitors: same benign
    pre-first-write-only SIGNIFICANT character as Stage A; ROUND-OFF SCALE at
    every exit checkpoint (calls 1, 2, 6, 12) — not a new defect.
- **Stage B: PASS.** No FMS `i_pack` ownership audit was needed — no evidence
  of any discrepancy pointed at that boundary (per the task's own
  "do not broadly audit unless evidence requires it").

## Hard-stop conditions (§11)

1. Stage A unexplained significant divergence — did not occur.
2. k_split semantics ambiguous — did not occur; resolved directly from
   `fv3.j`/`fv_dynamics.F90` source.
3. Stage B numerical divergence not explainable by integration/state
   ownership — did not occur; zero SIGNIFICANT differences found.
4. More than 3 substantial fixes required — did not occur; 0 of 3 used.
5. **Both Stage A and Stage B pass — TRUE.** Terminal condition reached.

## Regression

No source in this repository was modified, so per §13 no CTest rerun was
performed. Existing record stands unchanged and is reconfirmed by inference
(no transport-adjacent source touched): six-rank MPI transport 24/24
consecutive PASS (Prompt 18, job 58246343), Intel MPI 5/5 PASS (job
58246352), R8 CTest 23/23 (Prompt 18). Standalone short/default case
(6 dyn_core calls, k_split=1) remains PASS per Prompt 17/18. Longer
validation case (this pass): Stage A (72 calls) PASS, Stage B (12 calls,
k_split=2) PASS.

## Next validation envelope

R4 precision has never been savepoint-compared against Fortran (compiled and
ctest-clean only) and remains the next item; after that, wider grids, and
eventually `k_split` values matching the real production heartbeat/physics
cadence beyond the minimal k_split=2 probe used here.
