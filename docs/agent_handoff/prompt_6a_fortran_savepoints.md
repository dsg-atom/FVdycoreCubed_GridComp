# Prompt 6A — Authoritative Fortran dyn_core Savepoints

Status: **CLOSED — savepoints generated, reproducibility confirmed, ready for
Prompt 6B (CUDA comparison).**

## 1. Git State

- `FVdycoreCubed_GridComp` (this repo): branch `cuda/gpu-bringup`, HEAD
  `a060d98ca68a2996bf54d6bd214bbc3f3f977e8e`. No commits made by this task;
  only new files added under `tools/savepoints/` and this handoff doc (not
  committed — leave staging/commit decision to the user).
- Fortran fixture repo: `GEOSfvdycore_fortran_perf`,
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`,
  fixture HEAD `8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`.
- Instrumented file tree: fixture's nested
  `src/Components/@FVdycoreCubed_GridComp/@fvdycore` submodule, base SHA
  `ca30e26305fda43831095334c02a3f5e0b8c0c45` (matches the project's pinned
  `geos/v3.0.0` authoritative GFDL FV3 tag — confirmed in the prior session).
  Working tree there has 3 uncommitted changes (not committed by this task):
  `CMakeLists.txt` (+1 line, source-list registration),
  `model/dyn_core.F90` (+58 lines, savepoint call sites — 0 lines removed,
  0 numerical statements altered), and new file
  `model/fv3_savepoint_mod.F90` (277 lines). Full diff saved at
  `tools/savepoints/fortran_savepoint_instrumentation.patch`.

## 2. What Was Built

### 2.1 Instrumentation module (`fv3_savepoint_mod.F90`)

Env-var-gated (`FV3_SAVEPOINT_ENABLE=1`, `FV3_SAVEPOINT_DIR=<path>`),
zero-cost when disabled (every public entry point's first executable
statement is `if (.not. fv3_sp_enabled()) return`, and `fv3_sp_enabled()`
itself is a single cached logical read after the first call). No JSON
dependency in Fortran — raw binary field dumps (`access='stream'`,
`form='unformatted'`) plus plain-text `.meta` sidecar lines, and a formatted
per-level damping/order snapshot table flushed once per acoustic substep.

### 2.2 Four savepoints in `dyn_core.F90`

| Savepoint | Location | Tag |
|---|---|---|
| SP0 | dyn_core entry (before `#ifdef SW_DYNAMICS`) | `it=0` |
| SP1 | immediately after `p_grad_c`, before `COMM_UCVC` halo | `it` = live acoustic-substep index (1..3) |
| SP2 | immediately after `call timing_off('d_sw')` | `it` = live acoustic-substep index (1..3) |
| SP3 | dyn_core exit (after `heat_source` deallocate, before `end_step`) | tagged with literal `n_split` (not the loop variable, to avoid post-loop-`it` ambiguity) |

A per-level damping/order snapshot (`nord_k, nord_w, nord_t, nord_v(k),
damp_vt(k), damp_w, damp_t, d_ext(k), d_con_k, kgb, d2_divg, hord_m/v/t/p`)
is recorded once per `k` inside the existing `do k=1,npz` OMP loop
(immediately before `call d_sw(...)`) and flushed to `sp2_it<it>_pe<pe>_damping.txt`
right after the SP2 field dumps.

Full field inventory per savepoint instance (order/bounds explicitly recorded
per-field in the `.meta` sidecar, not assumed by any reader):

- **SP0** (10 3D fields): u, v, w, delp, pt, delz, q_con, pe, peln, pk.
  `pe`/`peln` are declared `(i,k,j)`; `pk` is declared `(i,j,k)` — this
  divergence is explicit in every meta line's `order=` token.
- **SP1** (6 3D + 1 2D field): uc, vc, pkc, gz, delpc, ptc, ws3.
- **SP2** (10 3D fields + damping table): u, v, w, delp, pt, divgd, mfx, mfy,
  cx, cy.
- **SP3** (13 3D fields): u, v, w, delp, pt, delz, pe, peln, pk, pkz, omga,
  ua, va.

`mpp_pe()` (already imported via `mpp_mod` in `dyn_core.F90`) identifies rank
in every filename.

### 2.3 Python tooling (`FVdycoreCubed_GridComp/tools/savepoints/`)

- `savepoint_reader.py` — parses `.meta` files, loads any field as a NumPy
  array respecting recorded Fortran bounds/order (`np.reshape(..., order='F')`),
  loads damping tables, runs finite/positivity sanity checks, classifies a
  pairwise field diff as `BIT-IDENTICAL` / `MACHINE-PRECISION EQUIVALENT`
  (max relative error ≤ 1e-12) / `FAIL`, and drives a `compare` subcommand
  across two full run directories. Also has `list`/`show` subcommands for
  ad hoc inspection.
- `manifest_builder.py` — enumerates every field across a run directory,
  computes a SHA256 of each underlying `.bin`/`.txt` file, and emits
  `manifest.json` recording per-field metadata (shape/bounds/order/precision)
  plus run-identifying context (fixture SHA, fvdycore SHA, grid, npz,
  n_split, precision, test_case, and a free-text `note`).

## 3. Runs Executed

Both jobs run under the validated Intel toolchain
(`comp/intel/2024.2.0` + `mpi/impi/2021.13`), R8, c8/npz=10/n_split=3,
`FV_NX=1`/`FV_NY=6` (6 MPI ranks, one per cubed-sphere face), single time
step (`JOB_SGMT='00000000 010000'`), `OMP_NUM_THREADS=1` (required — the
per-level damping accumulator in `fv3_savepoint_mod` is not thread-safe
under OMP>1; this is validation-only instrumentation, not a dyn_core
constraint).

- `experiments/c8_L10_savepoint_6a_run1/fv3.j` — job 58114327, COMPLETED,
  exit 0, elapsed 16s.
- `experiments/c8_L10_savepoint_6a_run2/fv3.j` — job 58114328, COMPLETED,
  exit 0, elapsed 16s. Byte-identical `fv3.j` to run1 except job-name/EXPDIR
  (confirmed via `diff`), submitted as an independent run for reproducibility
  classification.

Production first-run numerics were used in both:
`hord_mt=6, hord_vt=6, hord_tm=6, hord_dp=-6, nord=2, d4_bg_top=d4_bg_bot=0.12,
vtdm4=0.01, do_vort_damp=.T., grid_type=0, hydrostatic=.F., a_imp=1.0`.

### 3.1 Deviation from the 3A.12 baseline: `test_case`

The existing `c8_L10_dyncore_perf/fv3.j` baseline (3A.12 perf work) uses
`test_case = 5` with no `hydrostatic` override (defaults `.true.`). Tracing
`tools/test_cases.F90`, `test_case=5`'s non-SW_DYNAMICS branch (the branch
compiled here) only initializes vertical level 1 of `delp`/`phis`; levels
2..npz are left uninitialized for any `npz>1` run — independent of, and in
addition to, the `hydrostatic` setting. This is a genuine incompatibility
with a real (non-single-level) nonhydrostatic run, not merely a
hydrostatic-flag mismatch.

**Resolution**: switched `&test_case_nml` to `test_case = 14` (aqua-planet),
which calls `hydro_eq(...)` — a standard multi-level hydrostatic-balance
initializer that correctly populates `delp`/`pt`/`delz` at every level for
either `hydrostatic` setting, with calm (u=v=w=0) initial winds. This is a
driver-level initial-condition selection only; it does not touch any
`dyn_core` numerical algorithm and does not change any production namelist
default. Flagged here per the "genuine blocker — document, don't silently
deviate" rule; not silently substituted.

## 4. Results

### 4.1 Coverage

Both runs produced the full expected file inventory: 510 files each
(85 files × 6 MPI ranks), decomposing to SP0×11, SP1×24 (3 substeps × 8),
SP2×36 (3 substeps × 12, including damping table + meta), SP3×14 per rank —
exactly matching the field inventory in §2.2. 48 savepoint instances × 462
total fields (444 numeric-array fields + 18 per-substep damping tables) per
run, confirmed by `manifest_builder.py`.

### 4.2 A real bug found and fixed during this task

Initial inspection of `sp2_it1_pe0_damping.txt` showed corrupted/wrapped rows.
Root cause: the `fv3_sp_damping_flush` format string
`(I4,1X,4(I3,1X),6(ES23.15,1X),4(I3,1X))` declared only 6 `ES23.15`
descriptors but the write statement passes 7 real values (`damp_vt, damp_w,
damp_t, d_ext, d_con_k, kgb, d2_divg`) — a format/data-list-count mismatch
confined entirely to this task's own new instrumentation module. **Fix**:
changed to `7(ES23.15,1X)`. This is instrumentation-only; it does not touch
`dyn_core.F90` or any authoritative numerical algorithm. Rebuilt
(`StandAlone_FV3_Dycore.x`, target-scoped build to avoid an unrelated
pre-existing F2PY3 failure in the full `install` target), redeployed, and
reran both jobs. Post-fix, every damping row parses to the expected 16
whitespace-separated tokens with no wraparound.

### 4.3 Sanity checks

`savepoint_reader.py sanity` on both run directories: 444/444 numeric fields
all-finite, `delp` strictly positive at every point — **PASS** for both runs.

### 4.4 Reproducibility classification

`savepoint_reader.py compare run1 run2`: 48/48 common savepoint instances,
**462/462 fields BIT-IDENTICAL** (0 `MACHINE-PRECISION EQUIVALENT`, 0
`FAIL`), including all 18 damping tables (compared as exact list equality
after the format fix). No missing savepoints on either side.

**Reproducibility classification: BIT-IDENTICAL.**

### 4.5 Damping/order values (informational, consistent with input namelist)

Per-level damping table shows, at every level: `nord_k=nord_w=nord_t=nord_v=2`,
`damp_vt=damp_w=damp_t=0.01` (matches `vtdm4=0.01`), `d_con_k=1.0` (default),
`d_ext=kgb=d2_divg=0.0` (defaults, consistent with `nord=2` → `ndif∈{0,2}`
per the project's documented open item on `del6_vt_flux`), `hord_m=hord_v=
hord_t=6`, `hord_p=-6` — matches the namelist verbatim, confirming the
capture site executes downstream of all per-level override logic.

## 5. Artifacts

- `tools/savepoints/manifest_builder.py`, `tools/savepoints/savepoint_reader.py`
- `tools/savepoints/fortran_savepoint_instrumentation.patch` (git-apply-able
  diff: `CMakeLists.txt` +1, `dyn_core.F90` +58/-0, new
  `model/fv3_savepoint_mod.F90`, all relative to fixture `@fvdycore` HEAD
  `ca30e26...`)
- Raw savepoint data + manifests (NOT committed — large binary run output,
  lives on the shared filesystem only):
  - `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/c8_L10_savepoint_6a_run1/savepoints/` (+ `manifest.json`)
  - `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/experiments/c8_L10_savepoint_6a_run2/savepoints/` (+ `manifest.json`)
  - `experiments/c8_L10_savepoint_6a_run1/fv3.j`, `experiments/c8_L10_savepoint_6a_run2/fv3.j`

## 6. Known Limitations / Not In Scope Here

- Instrumentation is single-substep-loop, `OMP_NUM_THREADS=1` only (damping
  accumulator is not thread-safe under OMP>1). Fine for its intended
  validation purpose; do not reuse for a production OMP>1 run without adding
  synchronization.
- Only c8/npz=10/n_split=3/R8 was captured. R4 and larger grids (c16/c32) are
  not yet captured — extend by rerunning the same instrumented binary/build
  under the R4 CMake configuration and copying the same `fv3.j` pattern with
  `AGCM_IM=16/32`, if/when Prompt 6B needs them.
- This task is Fortran-only. No CUDA execution or comparison was performed.
  Do not infer anything about CUDA numerical agreement from this document.

## 7. Next Recommended Phase

Prompt 6B: run the CUDA candidate at the same c8/npz=10/n_split=3/R8
configuration, capture equivalent savepoints (or read CUDA device state at
equivalent boundaries), and use `savepoint_reader.py`'s comparison/
classification logic (or an extended version of it) against these
authoritative Fortran savepoints to identify the first numerical divergence,
per the coarse-to-fine localization procedure in
`docs/agent_instructions/project_architecture.md`.

---

## FINAL REPORT

**Task**: Prompt 6A — Authoritative Fortran dyn_core savepoint generation
**Config**: c8, npz=10, n_split=3, R8, 6 MPI ranks (FV_NX=1/FV_NY=6),
production first-run numerics (hord_mt=6/hord_vt=6/hord_tm=6/hord_dp=-6,
nord=2, d4_bg=0.12, vtdm4=0.01, do_vort_damp=.T., grid_type=0,
hydrostatic=.F., a_imp=1.0), test_case=14 (deviation from test_case=5, see §3.1)

**Savepoints captured**: SP0 (dyn_core entry), SP1 (post-C-grid-predictor,
×3 substeps), SP2 (post-D-grid/d_sw, ×3 substeps, + per-level damping table),
SP3 (dyn_core exit) — 48 savepoint instances, 462 fields total, across 2
independent runs.

**Coverage**: IMPLEMENTED, COMPILED, EXECUTED (Fortran, real 6-tile FMS MPI
communication, real `StandAlone_FV3_Dycore.x` driver — no new driver written).

**Reproducibility**: BIT-IDENTICAL (462/462 fields across run1 vs run2).

**Sanity checks**: PASS (all fields finite; delp > 0 everywhere) on both runs.

**Defects found/fixed this task**: 1 (damping-table write-format/data-list
mismatch, instrumentation-only, fixed and reverified — see §4.2). 0 defects
found in authoritative `dyn_core.F90` numerics; 0 numerical semantics
changed.

**Deviations documented**: 1 (`test_case` 5→14, driver-level IC selection
only — see §3.1). No production namelist default was changed.

**Blockers remaining**: none.

AUTHORITATIVE FORTRAN SAVEPOINTS READY — PROCEED TO PROMPT 6B
