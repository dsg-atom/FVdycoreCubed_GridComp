# Current Project State

_Last regenerated from Git + handoffs on 2026-08-27. Read this file first;
avoid re-reading older milestone handoffs unless a specific open item below
requires it._

## 1. Git State

- Repository: `FVdycoreCubed_GridComp`
  - Primary path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Branch: `cuda/gpu-bringup`
- HEAD: `6fc409a72eca2c43f06626cefcc4a8503880abaa` ("FORTRAN-CUDA PERFORMANCE BASELINE ESTABLISHED")
- Ahead of `origin/cuda/gpu-bringup` by 2 commits (`2c57d56`, `6fc409a`) — **not pushed**.
- Working tree: many untracked scratch files at repo root (`*_3a11*.sh`,
  `*.out`, `*.txt`, `dt3a11v/`) — scripts/logs from 3A.11 diagnostics, not
  committed, per repo policy. Do not commit them; do not delete without
  checking they're not needed for reproduction (some milestone docs cite them
  as reproduction scripts).

## 2. Authoritative Fortran Source

- `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag `geos/v3.0.0`,
  SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`.
- `@fvdycore` is **NOT populated** in this CUDA repo working tree (confirmed
  2026-08-27 — no `@fvdycore` entry under the repo root). Do not populate it
  automatically per `docs/agent_instructions/discover_platform.md`.
- Authoritative standalone Fortran fixture (separate tree, used for the 3A.12
  perf baseline only):
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Fixture repo HEAD `8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`
  - `@FVdycoreCubed_GridComp` layer HEAD `ac43844bae40dcc9558deda85c7aeebb4d06cf41`
  - `@fvdycore` SHA matches the pinned `ca30e26...` above.
  - Executable: `install/bin/StandAlone_FV3_Dycore.x`, Intel `ifort` build.

## 3. Target Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
```
CUDA backend optional; Fortran backend preserved. See
`docs/agent_instructions/project_architecture.md` for full detail (unchanged,
still current).

## 4. Implemented CUDA dyn_core Coverage

- Full 54-stage production first-run inventory (WP-E 10, WP-A 7, WP-B 18,
  WP-C 9, WP-D 10): **IMPLEMENTED**, **COMPILED**, **GPU EXECUTED**.
- n_split acoustic loop: IMPLEMENTED, GPU EXECUTED (n_split=3 authoritative;
  also run at 6/12 in diagnostics).
- Six-tile GPU-side halo exchange: IMPLEMENTED, GPU EXECUTED, liveness
  confirmed by `fv3_dispatch_smoke` Phase E causality test (R8 and R4).
- Persistent device residency: IMPLEMENTED, GPU EXECUTED (no per-step H2D for
  damping/grid arrays; transfer-inclusive vs resident-only both measured, see
  §6).
- Per-level damping translation layer (`fv3_damping_translation.{hpp,cpp}`):
  IMPLEMENTED, COMPILED, host-unit-tested. Closes prior gap where 8/10
  per-level damping arrays had no path from Fortran config into CUDA state.
- **NUMERICALLY VALIDATED: NO** for all of the above — no comparison against
  authoritative Fortran runtime savepoints has been run yet.

## 5. Current Correctness / Stability Status

As of commit `dd7b45e` (3A.11T/U/V consolidated) and `6fc409a`:

- Authoritative production first-run config (`d4_bg=0.12`, `nord=2`, `npz=10`,
  globally cube-consistent IC): 8×8, 16×16, 32×32 all **PASS**, `delp` stays
  positive, decelerating (not amplifying) growth, zero NaN/Inf across 192192
  diagnostic lines, 3/3 independent 8×8 runs bit-identical.
- R8 CTest: 22/22 PASS. R4 CTest: 22/22 PASS.
- compute-sanitizer memcheck: PASS, 0 errors.
- compute-sanitizer racecheck: PASS, 0 hazards.
- compute-sanitizer initcheck: PASS — only one previously-classified dead-read
  signature (`dgrid::copy_corners_kernel`, value-harmless in both Fortran and
  CUDA, not reproduced under the current authoritative `ndif` distribution).
- The `(d4_bg=0.16, nord=2)` fv_arrays default pairing is a **known foot-gun**
  (exceeds the del-2n stability limit) — not a production defect (GEOS
  overrides to 0.12), not a port defect (Fortran has the identical exposure).
  Runtime prints a warning; never clips/rejects.
- A real CUDA defect (two disagreeing gravitational constants across one
  acoustic substep — `9.8` hardcoded vs `cfg.grav`) was found and **repaired**
  this cycle (`dd7b45e`). It was NOT on the multi-step instability's causal
  path but set the instability's initial amplitude profile.

## 6. Performance Baseline

Fortran R8 median `DYN_CORE` ms/call (6 MPI ranks, one per cubed-sphere face,
Intel/AVX2, Discover `mil` nodes):

| Grid  | ms/call |
|-------|---------|
| 8×8   | 12.730  |
| 16×16 | 16.704  |
| 32×32 | 30.261  |

CUDA R8 resident median ms/call (1 A100, `warpa003`, persistent-residency
path, PRE-VALIDATION):

| Grid  | resident | transfer-inclusive |
|-------|----------|---------------------|
| 8×8   | 28.332   | 29.358              |
| 16×16 | 29.690   | 30.896              |
| 32×32 | 28.827   | 31.178              |

Speedup (Fortran ÷ CUDA, resident): 8×8 **0.449×**, 16×16 **0.563×**, 32×32
**1.050×**. CUDA is *slower* per call at small grids; only reaches rough
parity at 32×32, consistent with launch/orchestration-latency dominating at
these sizes (not compute-bound — a 16× cell-count increase changes resident
cost <2%).

**This is not a normalized hardware-efficiency comparison**: Fortran uses 6
real MPI ranks (CPU, one per face); CUDA uses one GPU. No CUDA
implementation-quality conclusion is implied. Numerical validation has **NOT
RUN** — do not infer numerical agreement from timing agreement.

## 7. Current Validation Boundary

- Gate C1 (integrated dyn_core GPU execution): **CLOSED** (established at
  milestone 3A.9, unaffected by later work).
- Fortran/CUDA runtime numerical validation: **NOT RUN**.
- Do not claim numerical equivalence from the performance baseline in §6.

## 8. Current Task

**Prompt 6A — Authoritative Fortran dyn_core savepoint generation.**

Goal: generate minimal authoritative Fortran savepoints (at the intermediate
locations listed in `project_architecture.md`'s Numerical Validation section)
for later CUDA comparison. The `GEOSfvdycore_fortran_perf` fixture (§2) is the
known-working standalone driver to reach `dyn_core` at the required
c8/c16/c32, `npz=10` sizes — reuse its configuration mapping (AK/BK override,
`FV_NX=1`/`FV_NY=6`, `n_split=3`/`k_split=1`) rather than rediscovering it.

## 9. Immediate Next Steps

1. Prompt 6A: instrument/extract authoritative Fortran savepoints from the
   fixture at dyn_core entry, post-C-grid-predictor, post-D-grid/transport,
   post-update_dz_d, post-D-pressure/NH, acoustic-substep end, dyn_core exit.
2. Prompt 6B: compare CUDA candidate against those savepoints; identify first
   divergence.
3. Gate C2 numerical agreement.
4. Only after Gate C2: resume deferred performance/profiling work (an
   `nsys`/`ncu` profile of the launch-bound inference in milestone 3A.11V §7.2
   is the first item queued there).

## 10. Known Open Issues / Deferred Items

- **E5 missing**: pre-`c_sw` scalar halo (`q_con,cappa,delp,pt`, authoritative
  `dyn_core.F90:476`, `it==1`) has no CUDA group; `H_DSCALAR` resolves too
  late (after `dgrid_dynamics`). Silent under current uniform-per-tile IC;
  would bite with a real (non-benchmark) IC.
- **E11 missing**: staggered seam-line reconciliation (`mpp_get_boundary`,
  ring h=0, `it==n_split` only) has no representable descriptor in the
  current halo API (ring-0 is structurally unreachable). Ruled out as cause
  of any measured instability (magnitude argument); still a genuine
  authoritative-fidelity gap to close before GEOS integration.
- **k=0 damping row not authoritative** in the benchmark harness (harness
  gives k=0 zero damping; Fortran floors `damp_vt` at every k). Harness-only
  fidelity defect; not implicated in any measured instability (amplifying
  mode lives at k=9, not k=0).
- **`clamp_nord_cap` (D1)** caps the WP-B group loop at `ng-1==2`, so
  `nord_k==3` silently gets zero divergence damping while still returning
  success. Not live at production `nord=2`. Docs incorrectly cite this dead
  code as evidence `nord=1-3` is supported.
- **`del6_vt_flux` n==1 uninitialized `d2` ring**: a dead-but-real
  uninitialized read, reachable only at `ndif==1` (not exercised by
  production `nord=2`, which yields `ndif ∈ {0,2}` only). Value-harmless in
  both Fortran and CUDA. Deferred: add an `ndif==1` case to
  `test_wp_b_cuda_smoke.cpp` to reproduce under initcheck, then decide on
  widening `del6_init_kernel`.
- **`docs/agent_instructions/cost_efficiency.md` does not exist yet** —
  referenced by §12 below but not yet created; create separately, do not
  inline a large policy here.
- Smaller recorded-not-fixed items (see `milestone_3A11T_...md` §7 and
  `milestone_3A11U_...md` "Genuine port defects" for full detail if ever
  needed): `divg2` read-without-writer guard; `heat_source`/`diss_est`
  per-substep accumulation not implemented (zeroed every substep); write-write
  race in the smag arena outer ring (appears inert); halo documentation
  defect at `fv3_gpu_halo_exchange.cu:44-52` / `fv3_gpu_halo_topology.hpp:289-297`
  (3 inaccuracies, contradicted with specifics in the 3A.11U handoff).

## 11. Hard Constraints

- Authoritative GFDL FV3 Fortran (`geos/v3.0.0`, `ca30e26...`) is the
  numerical source of truth.
- Do not modify `geos-gtfv3`.
- Do not claim numerical validation before Prompt 6B / Gate C2.
- Preserve persistent GPU residency; no full-state H2D/D2H between work
  packages.
- Use real six-tile halo semantics; do not stub communication for integrated
  validation.
- Do not populate `@fvdycore` in this repo without a demonstrated dependency,
  documented first.
- Generated build directories, Slurm logs, sanitizer logs, scratch scripts are
  not committed.
- Do not optimize numerical kernels before correctness localization, unless a
  task explicitly requests performance work (current phase does not).
- Do not push or merge unless explicitly requested.

## 12. Cost-Efficient Agent Rules

See `docs/agent_instructions/cost_efficiency.md` (not yet created as of this
writing — create as its own file if a dedicated policy is needed; do not
inline one here). Until then, apply directly:

- Read this file (`CURRENT_STATE.md`) first; avoid re-reading superseded
  milestone handoffs unless an open item above requires the detail.
- Use targeted greps/searches over broad repository scans.
- Use bounded, coarse-to-fine debugging when localizing numerical
  divergences (savepoint-by-savepoint, not whole-run diffs).
- Prefer the shortest Slurm job that answers the question (short `srun` probes
  over long `sbatch` runs where possible).
