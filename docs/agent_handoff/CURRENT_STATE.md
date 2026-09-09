# Current Project State

_Last updated 2026-09-09 (Prompt 23 pass: resolved the Prompt 22 c16/c32
CFL-timestep blocker — see §5/§9/§11). Full chronological detail for every
prompt below remains in `docs/agent_handoff/prompt_*.md` and
`docs/agent_handoff/milestone_*.md` — this file is deliberately no longer a
substitute for those._

## 1. Project Goal

Modernize the GEOS FV3 atmospheric dynamical core for heterogeneous CPU/GPU
execution via a staged path:

```
GEOS/FV3 Fortran -> ISO_C_BINDING -> stable C ABI -> C++/CUDA runtime
    -> persistent GPU dyn_core subsystem
```

Authoritative GFDL FV3 Fortran is the numerical source of truth. CUDA remains
optional; the original Fortran backend is always preserved. See
`docs/agent_instructions/project_architecture.md` for full architectural
constraints (work-package ownership, persistent-residency rules, validation
terminology) — unchanged and not repeated here.

## 2. Authoritative Sources and Repositories

- Authoritative Fortran: `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag
  `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`. Not
  populated as `@fvdycore` in this repo (expected; do not auto-populate).
- CUDA implementation repo (this repo): `FVdycoreCubed_GridComp`, branch
  `cuda/gpu-bringup`.
  - Discover path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Authoritative standalone Fortran fixture (separate tree, used for savepoint
  generation and dual-backend comparison):
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Executable: `install/bin/StandAlone_FV3_Dycore.x` (Intel `ifort` build).
  - Its nested `@fvdycore/model/{dyn_core.F90,fv_dynamics.F90}` carry
    uncommitted savepoint instrumentation only (no numerical change).
- CUDA source lives under `cuda/dyn_core/` (implementation) and
  `cuda/interface/` (Fortran ISO_C_BINDING bridge). Do not modify
  `geos-gtfv3` (historical GEOS PACE/GT4Py/DaCe implementation; reference
  only, not numerically authoritative).

## 3. Current Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
           -> HaloTransport: { PeerHaloTransport | MpiHaloTransport }
```

- Work packages: WP-A (C-grid predictor), WP-B (D-grid dynamics/transport),
  WP-C (vertical/nonhydrostatic), WP-D (pressure/grid/post-loop operators),
  WP-E (GPU runtime/state/API/communication). Ownership boundaries unchanged
  from `project_architecture.md`; not restated here.
- `HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp`) has two
  interchangeable backends sharing all topology/rotation/sign/pack/unpack
  logic:
  - `PeerHaloTransport` — in-process, six host threads ("Model B"), default.
  - `MpiHaloTransport` — host-staged real MPI, six separate processes, one
    tile per rank (the production decomposition). Opt-in via CMake
    (`FV3_DYN_CORE_ENABLE_MPI_HALO`, default OFF) and runtime
    (`--mpi-halo`, default off). This is Stage 1 of
    `project_architecture.md`'s Communication Evolution; Stage 2 (CUDA-aware
    MPI) and Stage 3 (comm/compute overlap) are unimplemented future backends
    behind the same interface, not a redesign.
- Public C ABI (`fv3_gpu_api.h`, `fv3_gpu_communication.h`) covers: lifecycle
  (create/topology/step/finalize), `NumericalConfig` installation
  (`fv3_gpu_set_numerical_config`), production dispatch installation
  (`fv3_gpu_install_production_dispatch`), and halo-transport selection
  (`fv3_gpu_attach_mpi_halo_transport` / `_detach_halo_transport` /
  `_halo_transport_kind_of`). No C++ or MPI type crosses the ABI — an MPI
  communicator crosses as a Fortran INTEGER handle via `MPI_Comm_f2c`.
- Fortran bridge layer: `cuda/interface/{fv3_gpu_dyncore_bridge.F90,
  fv3_gpu_abi_mod.F90, fv3_gpu_field_registry.F90}`. The real production call
  site is `fv_dynamics.F90`'s `do n_map=1,k_split` loop; the CUDA branch there
  does not issue the FMS `i_pack(1)`/`i_pack(8)` group-halo-update starts
  (both are duplicates of exchanges the CUDA dyn_core already performs before
  the same consumers), while `i_pack(10)` (tracer halo, distinct/required) is
  still issued. This ownership question is resolved — see §7.
- CPU/CUDA backend selection is via `RUN_CUDA_FV3` in `AGCM.rc`; default
  Fortran path is unchanged when CUDA is off.

## 4. Implemented Capabilities

- Full 54-stage production first-run dyn_core inventory (WP-E 10, WP-A 7,
  WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED, COMPILED, GPU EXECUTED.
- n_split acoustic loop (n_split=3 authoritative): IMPLEMENTED, GPU EXECUTED.
- Six-tile GPU-side halo exchange, both in-process and cross-process MPI:
  IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED (see §5/§6).
- Persistent device residency: IMPLEMENTED, GPU EXECUTED — no full-state
  host/device transfer between work packages.
- `omga` finalization path (pem snapshot, `omega_from_pe`, `adv_pe`,
  `ec1/ec2/en1/en2` derived from existing authoritative B-grid geometry, not
  invented), E11 seam reconciliation (`FV3_GPU_H_DWIND_SEAM`), the
  `heat_source`/`diss_est` per-substep accumulator, `corner_edge_service_cuda`
  (dir=1 and dir=2 `copy_corners` rotations), and the derived `n_con`/`nf_ke`
  config values (from `vtdm4`, `nord`) are all IMPLEMENTED and were the fixes
  that closed Gate C2 (§5). Detail of each fix (root cause, change, "numerical
  algorithm changed: NO" for all): `prompt_6b_cuda_fortran_validation.md`
  §19-§22.
- Thirteen input-fidelity/layout defects (grid metrics, `bdt`, `ptop`/`zs`,
  `dp_ref`, `rdgas`, `n_sponge`, `edge_*` weights, Exner-kappa split,
  `grid_lonlat` store order, `cv_air`, `n_con`, corner service, `nf_ke`) were
  found and fixed during Gate C2 closure — full list and evidence in
  `prompt_6b_cuda_fortran_validation.md` §4/§19-§22. All confirmed
  "numerical algorithm changed: NO".
- CUDA Graph capture mode (`--cuda-graph`, opt-in, default off): FUNCTIONAL.
  Captures `dgrid_dynamics` only (halo exchange NOT captured); freezes
  `DSwOptions::dt` by value (refuses replay if `bdt`/`n_split` change from
  capture time). Validated bitwise-identical to non-graph path at 1 and 5
  dyn_core calls, and a 600-call stress run with no illegal memory access.
  Detail: `prompt_13_architectural_optimization.md`.
- Launch-fusion optimizations, opt-in, default off, all bitwise-identical to
  default: `--fuse-corner-fill` (`c_sw` corner-fill pair fusion, Prompt 9),
  `--fuse-halo-edges` (4-edge pack/unpack fusion, Prompt 11).
  `a2b4_global_corner_kernel` was restructured for parallelism (not
  launch-count) in Prompt 12; also bitwise-identical, always-on (not a flag).

## 5. Numerical Validation Matrix

All entries below are against authoritative Fortran, same fixture, c8 grid
unless noted, `npz=10`, production first-run numerics (`hord_mt=hord_vt=
hord_tm=6`, `hord_dp=-6`, `vtdm4=0.01`, `grid_type=0`, `hydrostatic=false`,
`a_imp=1.0`), 6 MPI ranks = 6 tiles.

| Precision | Grid | k_split | Run depth | Status | Worst discrepancy |
|-----------|------|---------|-----------|--------|--------------------|
| R8 | c8 | 1 | 1 call (all 15 in-loop boundaries + 13 exit fields) | **PASS — Gate C2 CLOSED** | `w` exit max_rel 1.413e-11 (SMALL) |
| R8 | c8 | 1 | 72 dyn_core calls (Stage A) | **PASS** | `w` max_rel 5.12e-11 at call 72, flat/sublinear growth |
| R8 | c8 | 2 | 12 dyn_core calls (Stage B) | **PASS** | `w` max_rel 3.53e-11 at call 6; no jump at remap boundary |
| R4 | c8 | 1 | 6 dyn_core calls | **PASS** (R4-scaled thresholds) | `w` max_rel 1.02e-2 at call 6, bounded growth |
| R4 | c8 | 1 | 72 dyn_core calls (Stage A) | **PASS** | `omga` bulk-ratio grows superlinearly to ~4% by call 72 (magnitude-decay artifact, not divergence; see note below) |
| R4 | c8 | 2 | 12 dyn_core calls (Stage B) | **PASS** | no discontinuity at remap boundary |
| R8 | c16 | 1 (bdt=1800, resolution-scaled) | 6 calls | **PASS** | `w` max_rel ≈6.7e-11 |
| R8 | c32 | 1 (bdt=1200, resolution-scaled) | 6 calls | **PASS** (1 explained non-blocking anomaly) | `mfyL` dce-entry-only 2-cell uninitialized-memory artifact, both backends independently, see note below; real exit-state agreement clean |
| R4 | c16 | 1 (bdt=1800, reused from R8) | 6 calls | **PASS** | `w` max_rel 6.80e-2 (dcx call 3) — small-denominator artifact, see note |
| R4 | c32 | 1 (bdt=1200, reused from R8) | 6 calls | **PASS** (largest ratio of all rows, explained) | `w` max_rel 2.39e-1 (dce call 3) — same small-denominator artifact as c16, scaled by finer grid; absolute agreement stays ≤5.9e-5 and bulk stats match, see note |

Grading scale used throughout: BITWISE (0 ULP) / ROUND-OFF SCALE (R8 ≤1e-13,
R4 ≤1e-6) / SMALL NUMERICAL DIFFERENCE (R8 ≤1e-8, R4 ≤1e-2) / SIGNIFICANT
(above those). R4 thresholds were derived from the measured R4/R8 max_rel
ratio (≈4-6e8, matching the float32/float64 epsilon ratio), not assumed.

Notes carried forward (not re-derivable from the table):
- `omga`'s R4 bulk-metric growth (Stage A, 72 calls) is driven by the field's
  own magnitude shrinking (a decaying diagnostic), not by the absolute
  CUDA/Fortran discrepancy growing — that discrepancy plateaus after call
  ≈43. Re-check this trend if R4 is ever run past 72 calls.
- `pe`'s single Fortran halo ring was originally not written by CUDA; this
  was found to have a real downstream consumer (`Lagrangian_to_Eulerian`'s
  D-grid remap) and is now fixed via `fv3_gpu_interface_pe_halo` (host-side
  `pe_halo` reproduction, bit-for-bit from the already-resident `delp` ring).
- Zero NaN/Inf across all PASS rows; `delp` stays positive on both backends
  at every checkpoint in every PASS row.
- **R8/R4 c16/c32 rows (Prompt 23)**: resolution-scaled `bdt` (1800s/1200s)
  came from pre-existing, previously-run clean Fortran-only baseline configs
  (`c16_L10_dyncore_perf`, `c32_L10_dyncore_perf` — 336/504 dyn_core calls,
  0 NaN/Range_Warn each), not a new search; `n_split=3`/`k_split=1` held
  fixed at every resolution. The `w`/`omga` max_rel figures at R4 c16/c32 are
  a small-denominator ratio artifact (tiny, matching-magnitude fields with
  near-zero local values), not a growing or CUDA-specific divergence —
  verified by direct raw-value/bulk-statistic inspection, not assumed. Detail:
  `prompt_23_resolution_timestep_validation.md`.
- Numerical algorithm changed: **NO**, for every fix behind every row above.

Full field-by-field tables: `prompt_6b_cuda_fortran_validation.md` (Gate C2),
`prompt_19_longer_standalone_validation.md` (R8 depth), `prompt_20_r4_numerical_validation.md`
(R4 first pass), `prompt_21_longer_r4_validation.md` (R4 depth),
`prompt_22_wider_grid_validation.md` (c16 blocker, now resolved — see
`prompt_23_resolution_timestep_validation.md`).

## 6. MPI / Communication Status

- Cross-process six-rank halo transport (`MpiHaloTransport`, host-staged
  MPI): IMPLEMENTED, COMPILED, GPU EXECUTED across six separate MPI
  processes. Validated bitwise-identical to the in-process `PeerHaloTransport`
  path (78/78 fields = 13 exit fields × 6 tiles) at 1 and 5 dyn_core calls,
  plus a dedicated six-rank provenance test at 16,692 checks / 0 failures
  (covers scalar groups, both wind staggerings DGRID_NE/CGRID_NE, multi-ring
  groups, and the E11 seam group).
- Rank↔tile mapping is identity (rank r ↔ tile r+1), verified at runtime
  (not just trusted) via an allgathered, cross-checked tile map.
- **MPI transport determinism issue: RESOLVED.** An intermittent
  `fv3_halo_mpi_transport` CTest failure was root-caused to a CUDA
  stream-ordering defect in the *test harness* (plain `cudaMemset` /
  synchronous `cudaMemcpy` are not ordered against a
  `cudaStreamNonBlocking` stream) — not in the transport, not an MPI
  contract violation, not an OpenMPI bug. Fixed in the test and hardened in
  production (`fv3_gpu_runtime.cpp` attach-time clears now use
  `cudaMemsetAsync` on the transfer stream). Now deterministic at 24/24
  consecutive OpenMPI runs and 5/5 Intel MPI runs. **Standing coding
  constraint**: any buffer setup/clear/poison that a non-blocking stream will
  later read must be enqueued on that stream, not the legacy stream. Detail:
  `prompt_18_mpi_transport_determinism.md`.
- **FMS/CUDA halo ownership ambiguity: RESOLVED.** `mpp_start_group_update`
  packs/posts/increments its counter immediately; only
  `mpp_complete_group_update` unwinds it, so a started-but-abandoned handle
  leaks a slot and MPI requests per `k_split` iteration. Resolution (Design
  A): the CUDA branch does not issue `i_pack(1)`/`i_pack(8)` at all (both are
  duplicates of CUDA's own pre-consumer exchanges); `i_pack(10)` (tracer) is
  still issued. Default Fortran path is unchanged (skip is inside an
  `if (use_cuda_dyncore)` arm). Detail:
  `prompt_14_standalone_cuda_integration.md` §8,
  `prompt_15_mpi_halo_transport.md`.
- CUDA-aware MPI (Stage 2) and communication/computation overlap (Stage 3)
  remain unimplemented future work — deferred by project policy, not
  attempted.

## 7. Standalone Integration Status

- The CUDA dyn_core backend runs inside the real `StandAlone_FV3_Dycore.x`
  under the true production decomposition (`FV_NX=1`/`FV_NY=6`, six MPI
  ranks, one tile per rank), from the same executable as the Fortran backend
  (selected via `AGCM.rc`'s `RUN_CUDA_FV3`).
- Public C ABI for per-run setup (`NumericalConfig` install, production
  dispatch install, topology/transport attach) is complete; no internal-only
  C++ entry point is required for standalone integration anymore.
- Numerically validated at the depths in §5 (R8: 1/72/12-call ladders at c8,
  6-call at c16/c32; R4: 1/72/12-call ladders at c8, 6-call at c16/c32) — all
  PASS.
- ABI/compiler-family compatibility with a full **Intel-compiler production
  GEOS build** is not independently proven — this fixture successfully links
  GCC-built CUDA archives into an Intel-Fortran executable, which is evidence
  but not proof for the full GEOS production build. Revisit explicitly before
  final production linkage.

## 8. Performance Status

Latest measured baseline (32×32/R8/resident, single A100, same node/job,
`--fuse-halo-edges` on): job 58238910, `docs/agent_handoff/prompt_13_architectural_optimization.md`.

| Configuration | ms/call | vs. reference |
|---|---|---|
| Graph off (reference) | 21.660791 | — |
| Graph on (`--cuda-graph`) | 20.546787 | −5.14% |

Fortran R8 median `DYN_CORE` ms/call reference (6 MPI ranks, Intel/AVX2,
unchanged since Prompt 7): 8×8 12.730, 16×16 16.704, 32×32 30.261. Not a
normalized hardware-efficiency comparison (6 CPU ranks vs 1 GPU).

Key finding governing future performance work (Prompt 13, measured not
inferred): the workload crossed from **host-launch-bound to
GPU-execution-bound** — a −31.6% `cudaLaunchKernel` count reduction only
bought −5.14% wall clock, and `cudaStreamSynchronize` total time rose
0.44→135.5 ms/call. Further local launch-count fusion has diminishing
returns; the next real lever is kernel-time/occupancy tuning of
`riem_solver3_kernel`/`riem_solver_c_kernel`, which is **gated on Nsight
Compute (`ncu`) counter-permission access** (`ERR_NVGPUCTRPERM` on this
Discover allocation — confirmed platform restriction, not a flag error;
`nsys` remains usable and was used for all profiling above). No performance
work has been done since Prompt 13; performance is explicitly deferred while
correctness/validation work (§5) continues, per project policy.

## 9. Known Limitations / Unvalidated Envelope

- **Resolved (Prompt 23).** The Prompt 22 c16 CFL-timestep blocker (both
  backends NaN-blowup at c16 with the c8-tuned `bdt=3600s` unchanged) is
  closed: resolution-scaled `bdt` (1800s at c16, 1200s at c32, `n_split=3`/
  `k_split=1` unchanged) — sourced from pre-existing, previously-run clean
  Fortran-only baselines, not a new search — is stable in both backends at
  c16 and c32, R8 and R4, at the same 6-call depth as Prompt 22's own attempt.
  See §5 and `prompt_23_resolution_timestep_validation.md`.
- R4/R8 validation now covers c8 (deepest: 72/12-call ladders) and c16/c32
  (6-call depth, matching Prompt 22's original attempt depth). Do not infer
  behavior at run depths beyond what's in §5's table for any grid/precision
  combination — c16/c32 have not been run past 6 calls.
- `MOIST_CAPPA`/`USE_COND` are compiled in authoritative Fortran but not in
  CUDA. Proven round-off-neutral only because `q_con ≡ 0` in every tested
  fixture; must be closed before any moist (`q_con ≠ 0`) configuration.
- Four `a_imp <= 0.999`-only knobs (`scale_m`, `m_split`, `delt_max`,
  `scale_z`) are non-authoritative but proven inert under the tested
  `a_imp = 1.0` configuration. Must be corrected before any `a_imp <= 0.999`
  run.
- `ncu` (Nsight Compute) profiling is platform-blocked
  (`ERR_NVGPUCTRPERM`) on this Discover allocation; `nsys` is unaffected.
- `fv3_dispatch_smoke` Phase C fails if that test's own `cfg.nf_ke` pin is
  raised from 1 to the authoritative 3 — not localized (three `gpu_a100` job
  attempts to capture the diagnostic stayed queued); the test pin is left at
  1. Not evidence against the production corner service, which is exercised
  and validated at `nf_ke=3` in the real dyn_core path (§5).
- CUDA-aware MPI and communication/computation overlap (Stages 2/3) are
  unimplemented future work.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`, branch `cuda/gpu-bringup`.
- HEAD: `fa6623b` ("Document completed R4 FV3 CUDA validation"). Ahead of
  `origin/cuda/gpu-bringup` — not pushed. Push/merge only on explicit request.
- Tracked modification: this file only (`docs/agent_handoff/CURRENT_STATE.md`).
- Untracked at repo root: numerous scratch shell scripts/logs/probes from
  prior prompts (`task*_6c_cuda.sh`, `sp*rerun*.sh`, `p1[7-8]_*.sbatch`,
  `cln*.txt`, `cmp_nord*.txt`, `det_diff_*.txt`, `pre*.txt`, `scratch_*/`,
  build helpers) plus reusable tooling directories `tools/` (savepoint
  validation, e.g. `compare_dcx_backends.py`, `verify_omega_host.cpp`) and
  `perf_rerun_logs/` (sbatch scripts + job logs for Prompts 7-13). Also
  untracked: `docs/agent_handoff/prompt_22_wider_grid_validation.md`,
  `docs/agent_handoff/prompt_23_resolution_timestep_validation.md`.
- Do not commit: build directories (`build_cuda_r8*/`), profiler artifacts
  (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm log.
- Do commit (if explicitly requested): legitimate source fixes, permanent
  test/tooling additions under `tools/`, documentation.

## 11. Next Recommended Task

Wider-grid validation (c16/c32, R8/R4, 6-call depth) is now closed — see §5/§9.
Work should return to either (a)
`ncu`-gated kernel-time/occupancy tuning of `riem_solver3_kernel`/
`riem_solver_c_kernel` (requires requesting NCCS counter-permission access
first), or (b) production GEOS integration steps (moist configuration,
`a_imp <= 0.999`, Intel-compiler ABI verification) — confirm scope with the
user before starting either, per `project_architecture.md`'s "Performance
Work Comes Later" and this project's own deferred-item ordering.

## 12. Detailed Handoff References

Chronological/technical detail for every closed or attempted prompt below
remains in its own file under `docs/agent_handoff/`; read the specific file
only when the current task requires that level of detail (per
`docs/agent_instructions/cost_efficiency.md`).

- `prompt_6a_fortran_savepoints.md` — authoritative Fortran savepoint
  generation (bit-identical across two runs).
- `prompt_6b_cuda_fortran_validation.md` — Gate C2 closure: all thirteen
  input-fidelity fixes, all in-loop boundaries, all exit fields, `omga`
  finalization, E11, heat accumulator, `n_con`/`nf_ke`.
- `prompt_7_post_gate_c2_performance_baseline.md` — baseline re-establishment
  and nsys profiling after Gate C2.
- `prompt_8_cuda_graphs_prototype.md` — historical; conclusions retracted by
  Prompt 13 (see that file instead for the real CUDA Graphs status).
- `prompt_9_kernel_fusion_validation.md`, `prompt_10_copy_corners_batching.md`,
  `prompt_11_next_launch_target.md`, `prompt_12_a2b4_kernel_tuning.md` —
  launch-fusion/kernel-tuning characterization and implementation sequence.
- `prompt_13_architectural_optimization.md` — CUDA Graph capture repair;
  the GPU-execution-bound regime finding.
- `prompt_14_standalone_cuda_integration.md` — production call-boundary
  interface table; FMS/CUDA halo ownership resolution (§8).
- `prompt_15_mpi_halo_transport.md` — `HaloTransport` interface and
  `MpiHaloTransport` implementation/validation.
- `prompt_18_mpi_transport_determinism.md` — root cause and fix for the
  intermittent MPI transport test failure.
- `prompt_19_longer_standalone_validation.md` — R8 c8 72-call/12-call
  (k_split=2) validation ladder.
- `prompt_20_r4_numerical_validation.md`, `prompt_21_longer_r4_validation.md`
  — R4 c8 first-pass and depth-matched validation.
- `prompt_22_wider_grid_validation.md` — c16 CFL-instability blocker (resolved
  in Prompt 23).
- `prompt_23_resolution_timestep_validation.md` — resolution-scaled `bdt`
  evidence and c16/c32 R8/R4 validation closing the Prompt 22 blocker.
- `milestone_3a*.md`, `wp-*.md` — work-package-level implementation history
  (WP-A/B/C/D/E), pre-Gate-C2.

---

CURRENT_STATE maintenance policy:
- This file holds **current truth only** — replace superseded facts in place
  rather than appending a new dated paragraph.
- Detailed chronological history, per-field numerical tables, and resolved
  debugging narratives belong in the prompt-specific handoff docs (§12), not
  here.
- Keep this file at or below 35,000 characters. If an update would exceed
  that, compact an existing section before adding new content.
