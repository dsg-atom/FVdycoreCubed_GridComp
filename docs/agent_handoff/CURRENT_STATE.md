# Current Project State

_Last updated 2026-09-14 (Prompt 30 pass: production-readiness closure
attempt — confirmed via source, not new jobs, that `MOIST_CAPPA`/`USE_COND`/
`a_imp <= 0.999` are UNIMPLEMENTED (not merely unvalidated) in the CUDA
backend, by explicit, triply-gated, well-documented design predating this
prompt (milestone 3A.9); nonzero-`q_con`-alone cannot be isolated from that
gap because the authoritative Fortran fixture compiles `MOIST_CAPPA`/
`USE_COND` unconditionally. Intel-Fortran<->CUDA ABI verified sound by
construction (fixed-width `iso_c_binding` types, runtime
`abi_version`/`struct_bytes` guards, explicit `MPI_Comm_f2c`). See §5/§9/§11
and `prompt_30_production_config_closure.md`. Prompt 31 (separate branch
`experiment/held-suarez-e2e`, orthogonal to the `cuda/gpu-bringup` mainline
this file otherwise tracks): added a backend-neutral Held-Suarez idealized
dry-core forcing to the real standalone FV3 workflow and validated short
Fortran-vs-CUDA agreement — PASS, no numerical algorithm change, no Opus
escalation. See `prompt_31_held_suarez_e2e.md`. Prompt 32 (same branch)
advanced HS to scientific-validation level: confirmed the existing MAPL
History/CFIO NetCDF mechanism needs no new I/O code at C8 or C32; ran a
C32 (bdt=1200s/n_split=3/k_split=1, sourced from Prompt 23) Held-Suarez
stability+comparison gate, PASS both backends (savepoint-level 85,310/
87,810 fields BIT-IDENTICAL, remaining 2,500 FAILs confined to `delpc`/
`pkc` halo-padding cells — the same benign uninitialized-memory artifact
class as the Prompts 22/23 `mfyL` finding, not a new divergence; NetCDF-
level T@500hPa bitwise-identical); delivered `tools/analysis/
hs_netcdf_inspect.py` for offline xarray diagnostics. No Opus escalation.
See `prompt_32_held_suarez_scientific_validation.md`. Full chronological detail
for every prompt remains in `docs/agent_handoff/prompt_*.md` and
`docs/agent_handoff/milestone_*.md`; this file is deliberately no longer a
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
  - Executable: `build/bin/StandAlone_FV3_Dycore.x` (Intel `ifort` build) is
    the authoritative freshly-built copy; `install/bin/` is not reliably
    kept in sync with source rebuilds (found stale in Prompt 29). Each
    experiment job directory under `experiments/` keeps its own local copy
    of the binary — also not auto-synced; refresh with `cp -f` after any
    rebuild and verify via `strings <bin> | grep <new-env-var>` before
    relying on a new flag in a reused job directory.
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
  logic: `PeerHaloTransport` (in-process, six host threads, default) and
  `MpiHaloTransport` (host-staged real MPI, six processes, one tile/rank —
  opt-in via CMake `FV3_DYN_CORE_ENABLE_MPI_HALO`/runtime `--mpi-halo`,
  default off). This is Stage 1 of `project_architecture.md`'s Communication
  Evolution; Stages 2/3 are unimplemented future backends behind the same
  interface, not a redesign.
- Public C ABI (`fv3_gpu_api.h`, `fv3_gpu_communication.h`) covers: lifecycle
  (create/topology/step/finalize), `NumericalConfig` installation
  (`fv3_gpu_set_numerical_config`, includes `fuse_corner_fill`/
  `fuse_halo_edges`), production dispatch installation, halo-transport
  selection, and CUDA Graph mode as a dedicated function
  (`fv3_gpu_set_cuda_graph_mode`, distinct from the struct-field fusion
  flags — Prompt 8, Fortran-reachable since Prompt 29). No C++ or MPI type
  crosses the ABI — an MPI communicator crosses as a Fortran INTEGER handle
  via `MPI_Comm_f2c`.
- Fortran bridge layer: `cuda/interface/{fv3_gpu_dyncore_bridge.F90,
  fv3_gpu_abi_mod.F90, fv3_gpu_field_registry.F90}`. The real production call
  site is `fv_dynamics.F90`'s `do n_map=1,k_split` loop; the CUDA branch
  there skips the FMS `i_pack(1)`/`i_pack(8)` group-halo-update starts
  (duplicates of exchanges CUDA dyn_core already performs), keeps
  `i_pack(10)` (tracer halo, required). Resolved, see §6.
- CPU/CUDA backend selection is via `RUN_CUDA_FV3` in `AGCM.rc`; default
  Fortran path is unchanged when CUDA is off. Opt-in performance knobs, all
  default OFF, env-var controlled in `fv3_gpu_dyncore_bridge.F90` via a
  shared `env_flag_enabled(name)` helper (YES/1/TRUE, case-insensitive):
  `RUN_CUDA_FV3_FUSE_HALO_EDGES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL` (Prompt
  28), `RUN_CUDA_FV3_GRAPH` (Prompt 29).

## 4. Implemented Capabilities

- Full 54-stage production first-run dyn_core inventory (WP-E 10, WP-A 7,
  WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED, COMPILED, GPU EXECUTED.
- n_split acoustic loop (n_split=3 authoritative): IMPLEMENTED, GPU EXECUTED.
- Six-tile GPU-side halo exchange, both in-process and cross-process MPI:
  IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED (see §5/§6).
- Persistent device residency: IMPLEMENTED, GPU EXECUTED — no full-state
  host/device transfer between work packages.
- `omga` finalization path, E11 seam reconciliation
  (`FV3_GPU_H_DWIND_SEAM`), the `heat_source`/`diss_est` per-substep
  accumulator, `corner_edge_service_cuda`, and the derived `n_con`/`nf_ke`
  config values are all IMPLEMENTED and were the fixes that closed Gate C2
  (§5). All confirmed "numerical algorithm changed: NO"; detail in
  `prompt_6b_cuda_fortran_validation.md` §19-§22.
- Thirteen input-fidelity/layout defects (grid metrics, `bdt`, `ptop`/`zs`,
  `dp_ref`, `rdgas`, `n_sponge`, `edge_*` weights, Exner-kappa split,
  `grid_lonlat` store order, `cv_air`, `n_con`, corner service, `nf_ke`) were
  found/fixed during Gate C2 closure — same reference.
- CUDA Graph capture mode: FUNCTIONAL, IMPLEMENTED (Prompt 8, repaired
  Prompt 13), Fortran-reachable via `RUN_CUDA_FV3_GRAPH` on the standalone
  path since Prompt 29. Captures `dgrid_dynamics` only (halo exchange NOT
  captured); freezes `dt` by value (refuses replay if `bdt`/`n_split` change
  from capture time). Bitwise-identical to non-graph path at every depth
  tested (1/5/600 calls internal harness, Prompt 13; 10/100/310 calls real
  standalone path, Prompt 29). Real standalone benefit: 4.80%, LOW,
  see §8/§9.
- Launch-fusion optimizations, opt-in, default off, all bitwise-identical to
  default: `--fuse-corner-fill`/`RUN_CUDA_FV3_FUSE_CORNER_FILL` (Prompt 9,
  reachable from standalone path Prompt 28), `--fuse-halo-edges`/
  `RUN_CUDA_FV3_FUSE_HALO_EDGES` (Prompt 11, reachable Prompt 28).
  `a2b4_global_corner_kernel` was restructured for parallelism (Prompt 12);
  also bitwise-identical, always-on (not a flag).

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
| R4 | c8 | 1 | 72 dyn_core calls (Stage A) | **PASS** | `omga` bulk-ratio grows superlinearly to ~4% by call 72 (magnitude-decay artifact, not divergence; see note) |
| R4 | c8 | 2 | 12 dyn_core calls (Stage B) | **PASS** | no discontinuity at remap boundary |
| R8 | c16 | 1 (bdt=1800, resolution-scaled) | 6 calls | **PASS** | `w` max_rel ≈6.7e-11 |
| R8 | c32 | 1 (bdt=1200, resolution-scaled) | 6 calls | **PASS** (1 explained non-blocking anomaly) | `mfyL` dce-entry-only 2-cell uninitialized-memory artifact, both backends independently; real exit-state agreement clean |
| R4 | c16 | 1 (bdt=1800, reused from R8) | 6 calls | **PASS** | `w` max_rel 6.80e-2 (dcx call 3) — small-denominator artifact |
| R4 | c32 | 1 (bdt=1200, reused from R8) | 6 calls | **PASS** (largest ratio, explained) | `w` max_rel 2.39e-1 (dce call 3) — same small-denominator artifact, scaled by finer grid; absolute agreement stays ≤5.9e-5 |
| R8 | c32 | 1 | Graph ON vs Graph OFF, 10 calls (Prompt 29) | **PASS** | `dcx`/exit: BITWISE (0 ULP); `dce`/entry: same pre-existing `mfyL` artifact as the row above, reproduced identically in both modes |

Grading scale: BITWISE (0 ULP) / ROUND-OFF SCALE (R8 ≤1e-13, R4 ≤1e-6) /
SMALL NUMERICAL DIFFERENCE (R8 ≤1e-8, R4 ≤1e-2) / SIGNIFICANT (above those).
R4 thresholds derived from the measured R4/R8 max_rel ratio (≈4-6e8,
matching the float32/float64 epsilon ratio), not assumed.

Notes carried forward (not re-derivable from the table):
- `omga`'s R4 bulk-metric growth (Stage A, 72 calls) is driven by the
  field's own magnitude shrinking, not the absolute discrepancy growing —
  that plateaus after call ≈43.
- `pe`'s single Fortran halo ring is reproduced host-side via
  `fv3_gpu_interface_pe_halo` (bit-for-bit from the resident `delp` ring) —
  a real downstream consumer (`Lagrangian_to_Eulerian`) needed it.
- Zero NaN/Inf across all PASS rows; `delp` stays positive at every
  checkpoint in every PASS row.
- c16/c32 rows (Prompt 23): resolution-scaled `bdt` (1800s/1200s) came from
  pre-existing clean Fortran-only baselines, not a new search;
  `n_split=3`/`k_split=1` held fixed at every resolution.
- Numerical algorithm changed: **NO**, for every fix/flag behind every row.

Full field-by-field tables: `prompt_6b_cuda_fortran_validation.md` (Gate
C2), `prompt_19_longer_standalone_validation.md` (R8 depth),
`prompt_20_r4_numerical_validation.md`/`prompt_21_longer_r4_validation.md`
(R4), `prompt_22_wider_grid_validation.md`/
`prompt_23_resolution_timestep_validation.md` (c16/c32),
`prompt_29_standalone_cuda_graph.md` (Graph ON/OFF guard).

## 6. MPI / Communication Status

- Cross-process six-rank halo transport (`MpiHaloTransport`): IMPLEMENTED,
  COMPILED, GPU EXECUTED across six MPI processes. Validated
  bitwise-identical to in-process `PeerHaloTransport` (78/78 fields at 1/5
  calls) plus a dedicated six-rank provenance test (16,692 checks/0
  failures).
- Rank<->tile mapping is identity (rank r <-> tile r+1), verified at runtime.
- **MPI transport determinism: RESOLVED** (Prompt 18) — root cause was a
  CUDA stream-ordering defect in the *test harness*, not the transport;
  fixed and hardened (`cudaMemsetAsync` on the transfer stream at
  attach-time). Deterministic at 24/24 OpenMPI + 5/5 Intel MPI runs.
  **Standing constraint**: any buffer setup a non-blocking stream will read
  must be enqueued on that stream, not the legacy stream.
- **FMS/CUDA halo ownership ambiguity: RESOLVED** (Prompt 14/15) — CUDA
  branch skips `i_pack(1)`/`i_pack(8)` (duplicates of CUDA's own
  pre-consumer exchanges), keeps `i_pack(10)` (tracer). Default Fortran
  path unchanged (skip is inside `if (use_cuda_dyncore)`).
- CUDA-aware MPI (Stage 2) and comm/compute overlap (Stage 3) remain
  unimplemented future work — measured LOW priority, see §8.

## 7. Standalone Integration Status

- CUDA dyn_core runs inside the real `StandAlone_FV3_Dycore.x` under the
  true production decomposition (`FV_NX=1`/`FV_NY=6`, six MPI ranks, one
  tile/rank), same executable as Fortran (selected via `AGCM.rc`'s
  `RUN_CUDA_FV3`).
- Public C ABI for per-run setup (config/dispatch/topology/transport/Graph)
  is complete; no internal-only C++ entry point needed for standalone
  integration.
- Numerically validated at the depths in §5 — all PASS.
- ABI/compiler-family compatibility with a full **Intel-compiler production
  GEOS build** is not independently proven — this fixture links GCC-built
  CUDA archives into an Intel-Fortran executable (evidence, not proof for
  the full GEOS production build). Revisit before final production linkage.

## 8. Performance Status

Current best validated standalone-path configuration: **MPS ON +
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`**.
`RUN_CUDA_FV3_GRAPH` is available (Prompt 29) but NOT part of the
recommended default (LOW benefit — see below). All figures are the real
standalone six-rank MPI path (`StandAlone_FV3_Dycore.x`, c32, R8,
`n_split=3`/`k_split=1`, `bdt=1200s`, 6 MPI ranks/1 A100) unless marked
"internal harness"; measured via `fv_timing_mod`'s `DYN_CORE` timer,
100 calls/rep, median of 3-5 reps, savepoints off during timing.

| Stage | Config | ms/call | vs. prior |
|---|---|---|---|
| Fortran reference | 6 ranks, CPU | 19.930 | — |
| CUDA, no MPS, no fusion (Prompt 24, job 58295284) | 6 ranks, 1 A100, time-sliced contexts | 105.563 | 5.30x slower than Fortran |
| + CUDA MPS (Prompt 26/27, jobs 58380090/58380469) | same + MPS ON | 27.075 / 26.965 | 3.92x faster |
| + fuse_halo_edges + fuse_corner_fill (Prompt 28) | same + both fusion flags | 25.464 | −6.13% |
| + CUDA Graph (Prompt 29, job 58382771) | same + `RUN_CUDA_FV3_GRAPH=YES` | 24.300 | −4.80% (LOW, not default) |

Net: CUDA at 24.300 ms/call vs. Fortran's 19.930 ms/call = **1.22x slower**
(was 5.30x slower pre-MPS). All steps bitwise/numerically neutral (§5 + each
prompt's own guard).

Root-cause chain (full detail in the cited files, not repeated here):
- **MPS is the dominant lever** (Prompt 26): bottleneck was 6 CUDA contexts
  time-slicing one A100, not MPI/kernel-compute/H2D-D2H.
- **Post-MPS profile is MIXED/diffuse** (Prompt 27): largest bucket is
  `cudaLaunchKernel` (~51% of traced `DYN_CORE`), no single kernel >5.6%.
- **Launch fusion cut launches 2955->2637/call (−10.76%)**, but
  `cudaEventSynchronize` stayed flat (~28-30%) — sync is GPU-producer-wait
  under MPS context-sharing, not host slack; doesn't respond to launch cuts.
- **CUDA Graph (Prompt 29) cut launches further** (2637->1805/call,
  −31.6%), but whole-call improvement was only 4.80% (<5% floor -> **LOW**)
  because `cudaEventSynchronize` rose (9.98->12.27 ms/call) instead. BITWISE
  numerical guard; replay-stable over 310 calls/930 substeps, 0 errors; real
  and safe but **not the recommended default** per the <5% threshold.

CUDA-aware MPI, kernel-tuning, and rank/GPU redesign all remain LOW/NOT
JUSTIFIED (halo traffic 312 msgs/call all ≤64KB; device transfer ~2% of
`DYN_CORE`; MPS already at the 1-context ceiling) — Prompts 25a/26.

Internal single-process harness (`run_small_perf_bench`, no MPI, Prompt 13)
is NOT representative of the real path above — kept only as provenance for
the original (retracted) 5.14% CUDA Graph estimate: Graph off 21.66 / Graph
on 20.55 ms/call, internal harness only.

`ncu` (Nsight Compute) profiling remains platform-blocked
(`ERR_NVGPUCTRPERM`); `nsys` is unaffected and was used for all profiling
above.

## 9. Known Limitations / Unvalidated Envelope

- R4/R8 validation covers c8 (deepest: 72/12-call ladders) and c16/c32
  (6-call depth only, resolution-scaled `bdt`, Prompt 23). Do not infer
  behavior beyond the depths in §5 for any grid/precision combination.
- `MOIST_CAPPA`/`USE_COND`: **confirmed UNIMPLEMENTED in CUDA, not merely
  unvalidated** (Prompt 30) — triple-gated rejection (top-level runtime
  fence `fv3_gpu_runtime.cpp:1101-1102`, WP-B kernel-launch fence
  `wp_b_dsw_cuda.cu:889`, Fortran-bridge forced-zero
  `fv3_gpu_dyncore_bridge.F90:595-596`), by explicit design predating this
  prompt (flagged at milestone 3A.9). Prompt 30 also found nonzero `q_con`
  cannot be tested in isolation against the authoritative Fortran fixture,
  because that fixture compiles `MOIST_CAPPA`/`USE_COND` in unconditionally
  (`@fvdycore/CMakeLists.txt:130`) — closing this requires implementing the
  Riemann-solver `peg`/`gm2`/`cp2` terms (`nh_utils.F90:372-390`) in CUDA, a
  dedicated future WP-C/WP-B numerical-implementation prompt, not a
  config/validation task. See `prompt_30_production_config_closure.md`.
- `a_imp <= 0.999` (RIM_2D/SIM3p0 vertical branches): **confirmed
  UNIMPLEMENTED in CUDA** (Prompt 30), triple-gated (runtime fence
  `fv3_gpu_runtime.cpp:1082-1083`, Riemann-solver launcher fence
  `fv3_riem_solver_c.cu:247-251`, Fortran-bridge `mpp_error(FATAL)` guard
  `fv3_gpu_dyncore_bridge.F90:622-627`). `a_imp` is read from the real
  namelist (not hardcoded); only SIM1 (`a_imp=1.0`, the required first-run
  value) is implemented. Closing this is a second, independent, similarly
  dedicated future implementation prompt. The four downstream
  `a_imp <= 0.999`-only knobs (`scale_m`, `m_split`, `delt_max`, `scale_z`)
  remain proven inert at the tested `a_imp = 1.0` and unreachable while the
  gate above stands.
- `ncu` profiling is platform-blocked (`ERR_NVGPUCTRPERM`); `nsys`
  unaffected.
- `fv3_dispatch_smoke` Phase C fails if its own `cfg.nf_ke` pin is raised
  from 1 to the authoritative 3 — not localized, test pin left at 1; not
  evidence against the production corner service (validated at `nf_ke=3`
  in the real dyn_core path, §5).
- CUDA-aware MPI / comm-compute overlap (Stages 2/3): unimplemented,
  measured LOW priority (§8).
- CUDA MPS (`fv3_mps_lifecycle.sh`, Prompt 27) requires
  `compute_mode=Default` (true on Discover `gpu_a100`); only tested at
  c32/R8, up to ~310-call depth, one node. Not yet wired into every
  standing job script as the default (only one-off Prompt 27-29 scripts).
- Launch fusion (`RUN_CUDA_FV3_FUSE_HALO_EDGES`/`_FUSE_CORNER_FILL`, Prompt
  28) and CUDA Graph (`RUN_CUDA_FV3_GRAPH`, Prompt 29): both default OFF,
  both bitwise-safe, both only measured at c32/R8/6-rank/1-A100/MPS-ON; not
  yet tested at other grids, precisions, or without MPS. Graph's
  refusal-on-`dt`/`n_split`-change behavior (Prompt 8 source, unchanged)
  was not operationally re-exercised in Prompt 29 (held fixed all run).
- Multi-GPU (`dev = mod(comm_rank, device_count)`) implemented but
  validated only at 2 GPUs/3 contexts-per-GPU; 4-GPU and MPS+multi-GPU
  composition unmeasured (`--gres=gpu:4` would not schedule during
  Prompt 26).
- `install/bin/StandAlone_FV3_Dycore.x` and per-experiment job-directory
  binary copies are not auto-synced with `build/bin/` after a rebuild
  (discovered Prompt 29) — always refresh and verify via `strings` before
  trusting a newly-added flag in a reused job directory.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`, branch `cuda/gpu-bringup`.
- HEAD: `2e6327f` ("Document MPS rank-GPU performance architecture"). Ahead
  of `origin/cuda/gpu-bringup` — not pushed. Push/merge only on explicit
  request.
- Uncommitted source (Prompt 28+29, not committed — awaiting explicit
  request): `cuda/interface/fv3_gpu_dyncore_bridge.F90`
  (`env_flag_enabled` helper, `RUN_CUDA_FV3_FUSE_HALO_EDGES`/
  `RUN_CUDA_FV3_FUSE_CORNER_FILL`/`RUN_CUDA_FV3_GRAPH` wiring),
  `cuda/interface/fv3_gpu_abi_mod.F90` (new `fv3_gpu_set_cuda_graph_mode`
  public declaration + interface, Prompt 29).
- Tracked modification: `docs/agent_handoff/CURRENT_STATE.md`.
- Untracked at repo root: numerous scratch shell scripts/logs/probes from
  prior prompts (`task*_6c_cuda.sh`, `sp*rerun*.sh`, `p1[7-8]_*.sbatch`,
  `cln*.txt`, `cmp_nord*.txt`, `det_diff_*.txt`, `pre*.txt`, `scratch_*/`,
  build helpers), reusable tooling under `tools/` (savepoint validation,
  e.g. `compare_dcx_backends.py`, `tools/savepoints/savepoint_reader.py` —
  fixed this prompt, both `run_sanity_checks()` and `compare_runs()` had the
  same bug: hardcoded skip for the literal field name `"damping"` instead of
  any `.txt`-backed text-table field, breaking on the fixture's second such
  field `da_min`), new `tools/analysis/hs_netcdf_inspect.py` (Prompt 32,
  xarray offline-diagnostics CLI for the HS NetCDF output), and
  `perf_rerun_logs/` (Prompts 7-13 sbatch/logs). Untracked handoff docs for
  every prompt from 22 through 32 (see §12); Prompt 30
  (`prompt_30_production_config_closure.md`) is source-inspection only — no
  source file was modified. In the separate
  Fortran fixture tree
  `<fortran_perf>/experiments/`: per-prompt sbatch scripts/`.out` logs/nsys
  CSVs (Prompts 24-29), `fv3_mps_lifecycle.sh`, and job directories
  `p24_c32_r8_cuda(_nsys)`/`p28_c32_r8_cuda_guard` (reused across Prompts
  24/27/28/29) plus profiler artifacts (`.nsys-rep`/`.sqlite`) under
  `p24_c32_r8_cuda_nsys/`.
- Do not commit: build directories (`build_cuda_r8*/`), profiler artifacts
  (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm log.
- Do commit (if explicitly requested): legitimate source fixes, permanent
  test/tooling additions under `tools/`, documentation.

## 11. Next Recommended Task

Prompt 29 closed the CUDA Graph question raised at the end of Prompt 28:
Graph is now reachable from the standalone path (`RUN_CUDA_FV3_GRAPH`),
numerically safe (BITWISE), and stable under repeated replay, but its real
whole-`dyn_core` benefit (4.80%) falls just under the 5% MODERATE floor ->
**LOW**. It is NOT part of the recommended default production mode.
**Standing recommended optimized standalone CUDA mode remains unchanged
since Prompt 28: MPS ON + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` +
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`** (25.464-25.526 ms/call, 1.28-1.36x
slower than Fortran, down from 5.30x pre-MPS).

No further standalone dyn_core launch/orchestration optimization is
currently justified: `cudaEventSynchronize` (GPU-producer-wait under MPS
context-sharing) is now the dominant remaining cost and does not respond to
launch-count reduction (fusion or Graph); CUDA-aware MPI and kernel tuning
remain LOW priority (§8); rank/GPU redesign is NOT JUSTIFIED (MPS already
at the 1-context ceiling).

**Prompt 30 attempted Track 1 (production GEOS integration closures) and
found it BLOCKED-BY-DESIGN, not merely unvalidated**: `MOIST_CAPPA`,
`USE_COND`, and `a_imp <= 0.999` are confirmed unimplemented in CUDA
(triple-gated rejections, source-documented since milestone 3A.9); nonzero
`q_con` cannot be tested in isolation from that gap because the
authoritative Fortran fixture compiles `MOIST_CAPPA`/`USE_COND`
unconditionally. Intel-Fortran ABI verification (the one Track-1 item that
was reachable) PASSED. See §9 and `prompt_30_production_config_closure.md`.

Three open tracks — confirm scope with the user before starting any:
1. **CUDA MOIST_CAPPA/USE_COND implementation** — a dedicated WP-C/WP-B
   numerical-implementation prompt: add the Riemann-solver `peg`/`gm2`/
   `cp2` terms (`nh_utils.F90:372-390`) to `fv3_riem_solver_c.cu`, construct
   a source-backed nonzero-`q_con` fixture, then re-run Gate-C2-style
   validation. Not a config/plumbing task.
2. **CUDA `a_imp <= 0.999` (RIM_2D/SIM3p0) implementation** — a second,
   independent, similarly-scoped numerical-implementation prompt.
3. **Operational wiring** — fold the MPS lifecycle helper + the two fusion
   flags (the standing recommended mode) into every standing GEOS/
   standalone job script as the new default, not just the one-off Prompt
   27-29 job scripts. Lower-risk than 1/2; does not require new numerical
   work.

Until 1 or 2 closes, "broader GEOS integration" remains bounded to the
already-validated production first-run subset (`hord_*=±6`, `grid_type=0`,
`hydrostatic=false`, `a_imp=1.0`, `use_cond=moist_cappa=0`).

**Prompt 31 (separate track, branch `experiment/held-suarez-e2e`)** added
an idealized Held-Suarez dry-core forcing (`Held_Suarez_ForcingMod.F90`,
call site `FV_StateMod.F90::FV_Run` before the single per-outer-step
`fv_dynamics` call, gated by new `RUN_HELD_SUAREZ` `AGCM.rc` key,
default OFF) to the real standalone workflow in both the CUDA-tracking
repo and the separate `GEOSfvdycore_fortran_perf` build fixture (two-tree
architecture: the fixture, not this repo, actually compiles
`StandAlone_FV3_Dycore.x`). c8/L10/R8, 72 outer steps (144 `dyn_core`
calls), Fortran and CUDA both ran clean (jobs 58387366/58387442); `dcx`
comparison worst `max_rel` 7.999e-11 (ROUND-OFF/SMALL only, no
SIGNIFICANT). Resolved an apparent `pt`-units anomaly (savepoint `pt`
8.5-24.9, not Kelvin-plausible) by tracing `fv_dynamics.F90`: `pt` is
Kelvin at the HS call site (confirmed by the routine's own dummy-arg
contract) and is only converted to a virtual-potential-temperature-like
internal quantity (`/pkz`) later inside `fv_dynamics`, after the `dcx`/
`dce` dump points — not a bug, no escalation needed. One incident found
and fixed: copied job scripts retained stale `EXPDIR` pointing at the
original `p19_stageA_fortran`/`p19_stageA_cuda` baseline dirs, which were
accidentally overwritten (~16.9k/15.7k files) before the fix; documented
numeric results elsewhere are unaffected, raw artifacts not regenerated
(deferred, flagged for user). Full detail: `prompt_31_held_suarez_e2e.md`.

**Prompt 32 (same branch, follow-on to Prompt 31)** advanced HS from
integration-smoke-test to scientific-validation level and **PASSED**: (a)
confirmed the existing MAPL History/CFIO mechanism (`HISTORY.rc`
`inst1_2d_diag`/`inst3_3d_diag`) already satisfies standard-NetCDF-output
needs at both C8 and C32, no new I/O code required; (b) ran a C32 HS
stability+comparison gate using the authoritative Prompt-23 timestep
(`bdt=1200s`, `n_split=3`, `k_split=1`, no new search) — Fortran gate PASS
(job 58389608), CUDA gate PASS under the standing recommended mode (job
58390108); (c) fixed `savepoint_reader.py`'s text-table-field bug (see
§10); (d) sanity-checked both C32 HS savepoint trees clean (86,010 fields
each); (e) Fortran-vs-CUDA C32 HS savepoint comparison: 85,310/87,810
fields BIT-IDENTICAL, 0 MACHINE-PRECISION-EQUIVALENT, 2,500 FAIL — all
confined to `delpc`/`pkc` at `sp1`/`sp2a`, `pe` 0-4, with `max_abs` in the
IEEE-754-subnormal range (1e-313 to 1e-312); direct inspection confirmed
the real/physical field statistics are identical between backends and only
unwritten halo-padding cells (2-cell halo, capture-time-specific) differ —
same benign uninitialized-memory artifact class as the Prompts 22/23
`mfyL` finding, not a new or growing divergence, no Opus escalation; (f)
independently corroborated bitwise-identical at the NetCDF level
(`compare-backends` on T@500hPa, `max_abs_diff=0.0`); (g) confirmed the
known near-surface-level NaN artifact (T/U/V, 3 lowest of 25 pressure
levels, standard MAPL vertical-interpolation masking) is present
identically in both backends at C32, matching the pre-existing C8 pattern;
(h) delivered `tools/analysis/hs_netcdf_inspect.py` (describe/
global-mean-t/zonal-mean/compare-backends), tested against real C32 HS
output including a qualitative HS baroclinic-jet sanity check
(zonal-mean U@250hPa peaking ~35 m/s at ±45°); (i) confirmed (source-only,
untested) that MAPL's standard internal-checkpoint restart mechanism
(`COLDSTART: 0`) is available and unaffected by the HS/CUDA work, though
every HS run so far used `COLDSTART: 1`. **Per task scope, no long
(10/100/1000-day) C32 run was attempted** — stopped after the short
validation passed. No CUDA/Fortran/HS-forcing source was modified; only two
Python analysis-tool fixes/additions in this repo's `tools/` and job/config
scripts in the separate fixture tree. Full detail:
`prompt_32_held_suarez_scientific_validation.md`.

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
- `prompt_7_post_gate_c2_performance_baseline.md` — baseline
  re-establishment and nsys profiling after Gate C2.
- `prompt_8_cuda_graphs_prototype.md` — historical; conclusions retracted
  by Prompt 13.
- `prompt_9_kernel_fusion_validation.md`, `prompt_10_copy_corners_batching.md`,
  `prompt_11_next_launch_target.md`, `prompt_12_a2b4_kernel_tuning.md` —
  launch-fusion/kernel-tuning characterization and implementation sequence.
- `prompt_13_architectural_optimization.md` — CUDA Graph capture repair;
  the GPU-execution-bound regime finding.
- `prompt_14_standalone_cuda_integration.md` — production call-boundary
  interface table; FMS/CUDA halo ownership resolution.
- `prompt_15_mpi_halo_transport.md` — `HaloTransport` interface and
  `MpiHaloTransport` implementation/validation.
- `prompt_18_mpi_transport_determinism.md` — root cause and fix for the
  intermittent MPI transport test failure.
- `prompt_19_longer_standalone_validation.md` — R8 c8 72-call/12-call
  (k_split=2) validation ladder.
- `prompt_20_r4_numerical_validation.md`, `prompt_21_longer_r4_validation.md`
  — R4 c8 first-pass and depth-matched validation.
- `prompt_22_wider_grid_validation.md` — c16 CFL-instability blocker
  (resolved in Prompt 23).
- `prompt_23_resolution_timestep_validation.md` — resolution-scaled `bdt`
  evidence and c16/c32 R8/R4 validation closing the Prompt 22 blocker.
- `prompt_24_standalone_performance_validation.md` — real standalone-path
  Fortran-vs-CUDA timing, internal-harness-vs-real-path attribution fix,
  nsys-based bottleneck classification.
- `prompt_25a_fv3_halo_characterization.md` — halo message-size/count
  census and the LOW CUDA-aware-MPI benefit ceiling.
- `prompt_26_rank_gpu_architecture.md` — rank/GPU/context architecture map,
  context-scaling measurements, the CUDA MPS 3.92x result and its bitwise
  guard, and the A-F candidate ranking.
- `prompt_27_mps_operational_profile.md` — MPS operationalized as a
  job-scoped opt-in lifecycle helper; post-MPS `nsys` profile and the MIXED
  launch-orchestration/kernel-compute reclassification.
- `prompt_28_standalone_launch_reduction.md` — `fuse_halo_edges`/
  `fuse_corner_fill` exposed on the standalone path; bitwise numerical
  guard; MODERATE (6.13%) timing result; CUDA Graph ABI JUSTIFIED decision
  (not implemented at that point).
- `prompt_29_standalone_cuda_graph.md` — CUDA Graph exposed on the
  standalone path (`RUN_CUDA_FV3_GRAPH`); bitwise numerical guard; replay
  stability; 4.80% timing result classified LOW; nsys bucket comparison.
- `prompt_30_production_config_closure.md` — production-readiness closure
  attempt: confirmed `MOIST_CAPPA`/`USE_COND`/`a_imp<=0.999` unimplemented
  by design (triple-gated); Stage-A/`q_con` coupling to that gap; Intel
  Fortran ABI verification PASS; no remaining undocumented hardcoded
  production assumptions found; no source changed.
- `milestone_3a*.md`, `wp-*.md` — work-package-level implementation history
  (WP-A/B/C/D/E), pre-Gate-C2.
- `prompt_31_held_suarez_e2e.md` — Held-Suarez idealized dry-core forcing:
  two-tree build architecture, forcing/call-site design, `pt`-units
  resolution, Fortran/CUDA comparison, EXPDIR incident and fix. Separate
  branch (`experiment/held-suarez-e2e`), orthogonal to this file's
  `cuda/gpu-bringup` mainline tracking.
- `prompt_32_held_suarez_scientific_validation.md` — HS scientific
  validation: NetCDF mechanism confirmation (C8+C32), C32 timestep-config
  sourcing, C32 Fortran/CUDA gate PASS, `savepoint_reader.py` text-field
  fix, savepoint comparison (benign `delpc`/`pkc` artifact explained),
  `hs_netcdf_inspect.py` deliverable, NetCDF NaN cross-check, restart
  capability note. Same branch as Prompt 31.

---

CURRENT_STATE maintenance policy:
- This file holds **current truth only** — replace superseded facts in
  place rather than appending a new dated paragraph.
- Detailed chronological history, per-field numerical tables, and resolved
  debugging narratives belong in the prompt-specific handoff docs (§12),
  not here.
- Keep this file at or below 30,000 characters (hard ceiling 35,000). If an
  update would exceed that, compact an existing section before adding new
  content.
