# Current Project State

_Last updated 2026-09-15 (Prompt 34: C32 dyn_core perf re-measured onto the
exact C64/C128 timing methodology, branch `experiment/held-suarez-e2e`,
orthogonal to the `cuda/gpu-bringup` mainline. Supersedes Prompt 33's
mismatched-baseline C32 figure — CUDA already overtakes Fortran at C32, not
just from C64 up. See §8a/§9a/§11 and `prompt_34_c32_perf_reconciliation.md`.
Prompts 30-33 remain summarized in §8a/§9/§11 and fully detailed in their
own handoff docs (§12); this file holds current truth only, not
chronological narrative._

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
  `cuda/gpu-bringup` (mainline) / `experiment/held-suarez-e2e` (HS track).
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
  28), `RUN_CUDA_FV3_GRAPH` (Prompt 29). Idealized dry-core Held-Suarez
  forcing (`Held_Suarez_ForcingMod.F90`, call site `FV_StateMod.F90::FV_Run`
  before `fv_dynamics`) is gated by `RUN_HELD_SUAREZ` in `AGCM.rc`, default
  OFF, backend-neutral (Prompt 31, branch `experiment/held-suarez-e2e`).

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
  tested. Real standalone benefit: 4.80%, LOW, see §8/§9.
- Launch-fusion optimizations, opt-in, default off, all bitwise-identical to
  default: `--fuse-corner-fill`/`RUN_CUDA_FV3_FUSE_CORNER_FILL` (Prompt 9,
  reachable from standalone path Prompt 28), `--fuse-halo-edges`/
  `RUN_CUDA_FV3_FUSE_HALO_EDGES` (Prompt 11, reachable Prompt 28).
  `a2b4_global_corner_kernel` was restructured for parallelism (Prompt 12);
  also bitwise-identical, always-on (not a flag).
- Idealized Held-Suarez dry-core forcing: IMPLEMENTED, COMPILED, GPU
  EXECUTED, NUMERICALLY VALIDATED at C8/C32/C64/C128 and up to 10-day
  duration — see §9a.

## 5. Numerical Validation Matrix (Production dyn_core, non-HS)

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
- Numerically validated at the depths in §5 (production) and §9a (HS) — all
  PASS.
- ABI/compiler-family compatibility with a full **Intel-compiler production
  GEOS build** verified sound by construction (Prompt 30: fixed-width
  `iso_c_binding` types, runtime `abi_version`/`struct_bytes` guards,
  explicit `MPI_Comm_f2c`) — this fixture links GCC-built CUDA archives into
  an Intel-Fortran executable. Revisit before final production linkage if
  the toolchain changes.

## 8. Performance Status (Production dyn_core, C32 baseline methodology)

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

Net at C32: CUDA at 24.300 ms/call vs. Fortran's 19.930 ms/call = **1.22x
slower** (was 5.30x slower pre-MPS). All steps bitwise/numerically neutral
(§5 + each prompt's own guard). **See §8a: at C64/C128 under the same MPS+
fusion mode, CUDA overtakes Fortran** — the gap direction reverses with
resolution.

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

`ncu` (Nsight Compute) profiling remains platform-blocked
(`ERR_NVGPUCTRPERM`); `nsys` is unaffected and was used for all profiling
above.

## 8a. Held-Suarez / Resolution Performance Scaling (Prompt 33, reconciled Prompt 34)

30 dyn_core calls/rep, 3 reps/config, savepoints off, standing MPS+fusion
mode (`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`,
Graph off), `fv_timing_mod` flat `DYN_CORE` FMS-clock line (total seconds
across all calls in the run / n_calls = ms/call; **first-call warmup
included** throughout — differs from §8's 100-call/rep warmup-excluded
figures, so not a like-for-like replacement of §8's C32 number, but now
internally consistent C32/C64/C128 to each other). C32 re-measured Prompt 34
with the identical methodology (RUN_DT=1200s, JOB_SGMT=10h -> 30 calls,
resolution-only diff from the C64/C128 job scripts); C64/C128 unchanged from
Prompt 33.

| Grid | Fortran ms/call | CUDA ms/call | Speedup F/C | CUDA/Fortran |
|---|---|---|---|---|
| C32 | 20.72 | 19.88 | 1.04 | **0.96 — CUDA faster** |
| C64 | 59.4 | 56.9 | 1.04 | **0.96 — CUDA faster** |
| C128 | 237.4 | 210.9 | 1.13 | **0.89 — CUDA faster** |

C32 reps: Fortran total_s 0.6237/0.6210/0.6216 (median 0.6216 -> 20.72
ms/call); CUDA total_s 0.6215/0.5945/0.5963 (median 0.5963 -> 19.88 ms/call).
All 6 C32 reps: rc=0, 0 NaN/Inf/Range_Warn/error hits.

Finding: **CUDA already overtakes Fortran at C32** under this exact
methodology (superseding Prompt 33's "narrows then reverses between C32 and
C64" reading, which used a mismatched C32 baseline). The CUDA/Fortran ratio
is flat C32->C64 (~0.96) then improves further at C128 (0.89) — CUDA's
margin widens with resolution rather than emerging partway through it.
Best GPU competitiveness measured: **C128 (0.89x)**. Rep-to-rep agreement
<2% at every config. Numerical source unchanged for this reconciliation —
timing-only. Detail: `prompt_34_c32_perf_reconciliation.md`.

## 9. Known Limitations / Unvalidated Envelope

- R4/R8 production (non-HS) validation covers c8 (deepest: 72/12-call
  ladders) and c16/c32 (6-call depth only, resolution-scaled `bdt`, Prompt
  23). Do not infer behavior beyond the depths in §5 for any grid/precision
  combination. **HS validation (§9a) separately covers c8/c32/c64/c128 up
  to 10-day duration** — do not conflate the two ladders.
- `MOIST_CAPPA`/`USE_COND`: **confirmed UNIMPLEMENTED in CUDA, not merely
  unvalidated** (Prompt 30) — triple-gated rejection (top-level runtime
  fence `fv3_gpu_runtime.cpp:1101-1102`, WP-B kernel-launch fence
  `wp_b_dsw_cuda.cu:889`, Fortran-bridge forced-zero
  `fv3_gpu_dyncore_bridge.F90:595-596`), by explicit design predating this
  prompt (flagged at milestone 3A.9). Nonzero `q_con` cannot be isolated
  because the authoritative Fortran fixture compiles `MOIST_CAPPA`/
  `USE_COND` unconditionally (`@fvdycore/CMakeLists.txt:130`) — closing this
  requires implementing the Riemann-solver `peg`/`gm2`/`cp2` terms
  (`nh_utils.F90:372-390`) in CUDA, a dedicated future WP-C/WP-B
  numerical-implementation prompt. See `prompt_30_production_config_closure.md`.
- `a_imp <= 0.999` (RIM_2D/SIM3p0 vertical branches): **confirmed
  UNIMPLEMENTED in CUDA** (Prompt 30), triple-gated (runtime fence
  `fv3_gpu_runtime.cpp:1082-1083`, Riemann-solver launcher fence
  `fv3_riem_solver_c.cu:247-251`, Fortran-bridge `mpp_error(FATAL)` guard
  `fv3_gpu_dyncore_bridge.F90:622-627`). Only SIM1 (`a_imp=1.0`, the
  required first-run value) is implemented; a second, independent
  implementation prompt would close this. The four downstream
  `a_imp <= 0.999`-only knobs (`scale_m`, `m_split`, `delt_max`, `scale_z`)
  remain proven inert at `a_imp = 1.0`.
- `ncu` profiling is platform-blocked (`ERR_NVGPUCTRPERM`); `nsys`
  unaffected.
- `fv3_dispatch_smoke` Phase C fails if its own `cfg.nf_ke` pin is raised
  from 1 to the authoritative 3 — not localized, test pin left at 1; not
  evidence against the production corner service (validated at `nf_ke=3`
  in the real dyn_core path, §5).
- CUDA-aware MPI / comm-compute overlap (Stages 2/3): unimplemented,
  measured LOW priority (§8).
- CUDA MPS (`fv3_mps_lifecycle.sh`, Prompt 27) requires
  `compute_mode=Default` (true on Discover `gpu_a100`). Not yet wired into
  every standing job script as the default (only one-off job scripts so
  far, now including the Prompt 33 Stage E perf scripts).
- Launch fusion and CUDA Graph: both default OFF, both bitwise-safe, tested
  at c32/R8 (§8) and additionally exercised (not separately re-guarded) at
  c64/c128 R8 HS runs (§9a) without regression. Graph's refusal-on-`dt`/
  `n_split`-change behavior was not operationally re-exercised this prompt.
- Multi-GPU (`dev = mod(comm_rank, device_count)`) implemented but
  validated only at 2 GPUs/3 contexts-per-GPU; 4-GPU and MPS+multi-GPU
  composition unmeasured.
- `install/bin/StandAlone_FV3_Dycore.x` and per-experiment binary copies
  are not auto-synced with `build/bin/` after a rebuild — always refresh
  and verify via `strings` before trusting a newly-added flag.
- **MAPL restart (`COLDSTART: 0`)**: source-confirmed available (Prompt 32)
  but never operationally exercised with HS+CUDA; every HS run through
  Prompt 33 used `COLDSTART: 1`.
- **C128 timestep search (Prompt 33)**: first `bdt` candidate was
  CFL-unstable (NaN flood, caught via size-gated health check before a
  full `grep`); resolved with a smaller `bdt`. Do not assume the C32
  resolution-scaled `bdt` heuristic (§5) extends linearly to C128 without a
  short stability check first.
- **Stage E (§8a) caveat**: 30-call/3-rep, warmup-included — not directly
  comparable to §8's warmup-excluded 100-call/rep C32 figure.

## 9a. Held-Suarez Validation Ladder (branch `experiment/held-suarez-e2e`)

Idealized dry-core Held-Suarez forcing, orthogonal to the `cuda/gpu-bringup`
mainline (§3/§4). All entries: Fortran vs CUDA, same standalone fixture,
6 MPI ranks, R8 unless noted, savepoint and/or NetCDF-level comparison.

| Grid | Duration | Result | Notes |
|---|---|---|---|
| C8 | 72 outer steps (144 dyn_core calls) | **PASS** | `dcx` max_rel 7.999e-11 (ROUND-OFF/SMALL); Prompt 31 |
| C32 | short gate (`bdt=1200s`/`n_split=3`/`k_split=1`) | **PASS** | savepoint 85,310/87,810 fields BIT-IDENTICAL, 2,500 FAIL confined to benign halo-padding (see below); NetCDF T@500hPa bitwise; Prompt 32 |
| C64 | short gate | **PASS** | NetCDF bitwise; Prompt 33 |
| C128 | short gate | **PASS** | first `bdt` candidate CFL-unstable (NaN flood), resolved by selecting a smaller `bdt`; accepted config NetCDF bitwise; Prompt 33 |
| C128 | 10-day (jobs 58411417 Fortran / 58411418 CUDA) | **PASS** | both rc=0, log sizes byte-identical (1,717,491 B), 80 NetCDF files each (10 days x 4/day x 2 collections); NetCDF T@500hPa/U@250hPa/PS bitwise at start/mid/end; `range_warn=6966` identical both backends — investigated and classified benign (see below); Prompt 33 |

Benign known artifacts (do not treat as new divergences):
- 2-cell halo-padding uninitialized-memory difference in `delpc`/`pkc` at
  savepoint level only, IEEE-754-subnormal magnitude (1e-313 to 1e-312) —
  same artifact class as the Prompts 22/23 `mfyL` finding. Never present at
  the NetCDF/physical-field level.
- C128 10-day `range_warn` (6966 hits, both backends): all `TA_dyn` at
  level 10, values 333.0-349.2K, 119 unique grid points. Confirmed benign
  via sorted-diff (`diff <(grep ... | sort) <(grep ... | sort)` = 0,
  content-identical modulo MPI-rank interleaving) — legitimate Held-Suarez
  near-surface tropical warmth exceeding a hardcoded FMS diagnostic
  plausibility bound, not a numerical defect.
- Known near-surface NaN artifact (T/U/V, 3 lowest of 25 pressure levels,
  standard MAPL vertical-interpolation masking) present identically in both
  backends at every resolution tested (C8/C32/C128) — pre-existing pattern,
  not new.

Not attempted (out of Prompt 33 scope): duration beyond 10 days, resolution
beyond C128, moist-physics coupling, restart-continuation runs, formal
climatology statistics.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`. Mainline branch `cuda/gpu-bringup`
  HEAD `2e6327f` ("Document MPS rank-GPU performance architecture"), ahead
  of `origin/cuda/gpu-bringup`, not pushed. HS track branch
  `experiment/held-suarez-e2e` HEAD `5bb5cc6` ("Validate global C32
  Held-Suarez Fortran and CUDA runs"), also not pushed. Push/merge only on
  explicit request.
- Uncommitted source on `cuda/gpu-bringup` (Prompt 28+29, awaiting explicit
  commit request): `cuda/interface/fv3_gpu_dyncore_bridge.F90`
  (`env_flag_enabled` helper + fusion/Graph env wiring),
  `cuda/interface/fv3_gpu_abi_mod.F90` (`fv3_gpu_set_cuda_graph_mode`).
- Untracked at repo root (both branches share the worktree): numerous
  scratch shell scripts/logs/probes from prior prompts (`task*_6c_cuda.sh`,
  `sp*rerun*.sh`, `p1[7-8]_*.sbatch`, `cln*.txt`, `cmp_nord*.txt`,
  `det_diff_*.txt`, `pre*.txt`, `scratch_*/`, build helpers), reusable
  tooling under `tools/` (savepoint validation — `compare_dcx_backends.py`,
  `tools/savepoints/savepoint_reader.py`; `tools/analysis/
  hs_netcdf_inspect.py`, Prompt 32), and `perf_rerun_logs/`. Untracked
  handoff docs for every prompt from 22 through 33 (see §12). In the
  separate Fortran fixture tree `<fortran_perf>/experiments/`: per-prompt
  sbatch scripts/`.out` logs/nsys artifacts (Prompts 24-33), job directories
  including the Prompt 33 additions (`p33_c64_gate_*`, `p33_c128_gate_*`,
  `p33_c128_hs10d_*`, `p33_c64_perf_*`, `p33_c128_perf_*` — the latter four
  with `perf_logs/rep{1,2,3}.log`), and `fv3_mps_lifecycle.sh`.
- Do not commit: build directories (`build_cuda_r8*/`), profiler artifacts
  (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm log, NetCDF
  output, restart data.
- Do commit (if explicitly requested): legitimate source fixes, permanent
  test/tooling additions under `tools/`, documentation.

## 11. Next Recommended Task

**Mainline (`cuda/gpu-bringup`) tracks — confirm scope with the user before
starting any:**
1. **CUDA MOIST_CAPPA/USE_COND implementation** — dedicated WP-C/WP-B
   numerical-implementation prompt (§9).
2. **CUDA `a_imp <= 0.999` implementation** — second, independent
   implementation prompt (§9).
3. **Operational wiring** — fold the MPS lifecycle helper + two fusion
   flags into every standing job script as the default, not just one-off
   scripts. Lower-risk than 1/2.

Until 1 or 2 closes, "broader GEOS integration" remains bounded to the
already-validated production first-run subset (`hord_*=±6`, `grid_type=0`,
`hydrostatic=false`, `a_imp=1.0`, `use_cond=moist_cappa=0`).

**HS track (`experiment/held-suarez-e2e`) — Prompt 33 closed Stages C/D/E**
of the multi-day/high-resolution ladder: C128 short-gate PASS, C128 10-day
PASS (bitwise NetCDF), and a C32->C64->C128 performance scaling table
showing CUDA overtaking Fortran (§8a). **Prompt 34 reconciled the C32 point
onto the exact C64/C128 methodology**: CUDA already overtakes Fortran at
C32 (0.96x), not just from C64 up; margin widens further at C128 (0.89x —
best measured GPU competitiveness). Prompts 31 (integration smoke test) and
32 (scientific validation, NetCDF-mechanism confirmation) are prerequisite
context — see `prompt_31_held_suarez_e2e.md` /
`prompt_32_held_suarez_scientific_validation.md`.

Open HS-track follow-ons (confirm scope before starting):
1. **Climatology-scale run** (30-100+ day, C32 or higher) — the natural
   next step now that 10-day/C128 is validated; not started.
2. **Higher resolution** (C256+) — unmeasured; would need its own
   stability/timestep search per the C128 lesson in §9.
3. **Restart-continuation exercise** — validate `COLDSTART: 0` with HS+CUDA
   (source-confirmed available, never operationally tested; §9).

## 12. Detailed Handoff References

Chronological/technical detail for every closed or attempted prompt below
remains in its own file under `docs/agent_handoff/`; read the specific file
only when the current task requires that level of detail (per
`docs/agent_instructions/cost_efficiency.md`).

- `prompt_6a_fortran_savepoints.md` — authoritative Fortran savepoint
  generation (bit-identical across two runs).
- `prompt_6b_cuda_fortran_validation.md` — Gate C2 closure.
- `prompt_7_post_gate_c2_performance_baseline.md` — baseline
  re-establishment and nsys profiling after Gate C2.
- `prompt_8_cuda_graphs_prototype.md` — historical; conclusions retracted
  by Prompt 13.
- `prompt_9_kernel_fusion_validation.md`, `prompt_10_copy_corners_batching.md`,
  `prompt_11_next_launch_target.md`, `prompt_12_a2b4_kernel_tuning.md` —
  launch-fusion/kernel-tuning sequence.
- `prompt_13_architectural_optimization.md` — CUDA Graph capture repair;
  GPU-execution-bound regime finding.
- `prompt_14_standalone_cuda_integration.md` — production call-boundary
  interface table; FMS/CUDA halo ownership resolution.
- `prompt_15_mpi_halo_transport.md` — `HaloTransport` interface and
  `MpiHaloTransport` implementation/validation.
- `prompt_18_mpi_transport_determinism.md` — root cause/fix for the
  intermittent MPI transport test failure.
- `prompt_19_longer_standalone_validation.md` — R8 c8 72-call/12-call
  validation ladder.
- `prompt_20_r4_numerical_validation.md`, `prompt_21_longer_r4_validation.md`
  — R4 c8 validation.
- `prompt_22_wider_grid_validation.md` — c16 CFL-instability blocker
  (resolved Prompt 23).
- `prompt_23_resolution_timestep_validation.md` — resolution-scaled `bdt`
  and c16/c32 R8/R4 validation.
- `prompt_24_standalone_performance_validation.md` — real standalone-path
  timing, nsys-based bottleneck classification.
- `prompt_25a_fv3_halo_characterization.md` — halo census, LOW
  CUDA-aware-MPI ceiling.
- `prompt_26_rank_gpu_architecture.md` — rank/GPU/context map, MPS 3.92x
  result.
- `prompt_27_mps_operational_profile.md` — MPS as job-scoped lifecycle
  helper; post-MPS nsys profile.
- `prompt_28_standalone_launch_reduction.md` — fusion flags exposed;
  MODERATE (6.13%) result.
- `prompt_29_standalone_cuda_graph.md` — CUDA Graph exposed; LOW (4.80%)
  result.
- `prompt_30_production_config_closure.md` — `MOIST_CAPPA`/`USE_COND`/
  `a_imp<=0.999` confirmed unimplemented by design; Intel ABI verification
  PASS.
- `milestone_3a*.md`, `wp-*.md` — work-package-level implementation
  history, pre-Gate-C2.
- `prompt_31_held_suarez_e2e.md` — HS forcing integration smoke test
  (C8, two-tree build architecture, `pt`-units resolution).
- `prompt_32_held_suarez_scientific_validation.md` — HS scientific
  validation (C32, NetCDF mechanism, savepoint comparison,
  `hs_netcdf_inspect.py`).
- `prompt_33_hs_multiday_highres_performance.md` — HS ladder extended to
  C64/C128 and 10-day duration; C32->C64->C128 performance scaling table
  (§8a/§9a of this file).
- `prompt_34_c32_perf_reconciliation.md` — C32 dyn_core perf re-measured
  under the exact C64/C128 methodology, reconciling §8a's scaling table.

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
