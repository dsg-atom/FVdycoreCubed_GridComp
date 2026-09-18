# Current Project State

_Last updated 2026-09-17 (Prompt 39: C512 PeerHalo-vs-MPI discriminator,
branch `experiment/held-suarez-e2e`, diagnostic-only — 1 new GPU job,
existing `fv3_gpu_perf_bench` harness, no source changes. Result: the
in-process `PeerHaloTransport` path scales C256->C512 at 3.69x (below
even the Fortran path's 5.72x), while the six-rank `MpiHaloTransport`
path collapses at 8.32x — isolating Prompt 37/38's C512 CUDA/Fortran=
1.270 regression to the MPI transport/protocol layer specifically, not
the shared device-side halo pack/unpack/staging code. CUDA-aware-MPI
priority MODERATE(contingent)->HIGH CANDIDATE. One next experiment
recommended (OpenMPI eager/rendezvous MCA threshold sweep at C512), not
yet run. See §6/§8a/§9/§11 and `prompt_39_c512_peerhalo_discriminator.md`.
This file holds current truth only._

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
    the authoritative freshly-built copy; `install/bin/` and per-experiment
    job-directory copies are not auto-synced — refresh with `cp -f` after
    any rebuild and verify via `strings <bin> | grep <new-env-var>`.
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
  WP-E (GPU runtime/state/API/communication). Ownership unchanged from
  `project_architecture.md`; not restated here.
- `HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp`): two
  interchangeable backends sharing all topology/rotation/sign/pack/unpack
  logic — `PeerHaloTransport` (in-process, six host threads, default) and
  `MpiHaloTransport` (host-staged real MPI, six processes, one tile/rank —
  opt-in via CMake `FV3_DYN_CORE_ENABLE_MPI_HALO`/runtime `--mpi-halo`,
  default off). Stage 1 of `project_architecture.md`'s Communication
  Evolution; Stages 2/3 unimplemented future backends behind the same
  interface.
- Public C ABI (`fv3_gpu_api.h`, `fv3_gpu_communication.h`) covers:
  lifecycle (create/topology/step/finalize), `NumericalConfig` installation
  (includes `fuse_corner_fill`/`fuse_halo_edges`), production dispatch,
  halo-transport selection, and CUDA Graph mode as a dedicated function
  (`fv3_gpu_set_cuda_graph_mode`, Fortran-reachable since Prompt 29). No
  C++/MPI type crosses the ABI — an MPI communicator crosses as a Fortran
  INTEGER via `MPI_Comm_f2c`.
- Fortran bridge layer: `cuda/interface/{fv3_gpu_dyncore_bridge.F90,
  fv3_gpu_abi_mod.F90, fv3_gpu_field_registry.F90}`. Real production call
  site is `fv_dynamics.F90`'s `do n_map=1,k_split` loop; CUDA branch skips
  FMS `i_pack(1)`/`i_pack(8)` (duplicates of CUDA's own exchanges), keeps
  `i_pack(10)` (tracer halo, required). Resolved, see §6.
- CPU/CUDA backend selection via `RUN_CUDA_FV3` in `AGCM.rc`; default
  Fortran path unchanged when CUDA off. Opt-in perf knobs, default OFF,
  env-var controlled via shared `env_flag_enabled(name)` helper:
  `RUN_CUDA_FV3_FUSE_HALO_EDGES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL` (Prompt
  28), `RUN_CUDA_FV3_GRAPH` (Prompt 29). Idealized Held-Suarez forcing
  (`Held_Suarez_ForcingMod.F90`, call site `FV_StateMod.F90::FV_Run` before
  `fv_dynamics`) gated by `RUN_HELD_SUAREZ`, default OFF, backend-neutral
  (Prompt 31, branch `experiment/held-suarez-e2e`).

## 4. Implemented Capabilities

- Full 54-stage production first-run dyn_core inventory (WP-E 10, WP-A 7,
  WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED, COMPILED, GPU EXECUTED.
- n_split acoustic loop (n_split=3 authoritative): IMPLEMENTED, GPU EXECUTED.
- Six-tile GPU-side halo exchange, in-process and cross-process MPI:
  IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED (§5/§6).
- Persistent device residency: IMPLEMENTED, GPU EXECUTED — no full-state
  host/device transfer between work packages.
- `omga` finalization, E11 seam reconciliation (`FV3_GPU_H_DWIND_SEAM`),
  the `heat_source`/`diss_est` per-substep accumulator,
  `corner_edge_service_cuda`, and derived `n_con`/`nf_ke` config values —
  all IMPLEMENTED, the fixes that closed Gate C2 (§5). All confirmed
  "numerical algorithm changed: NO"; detail in
  `prompt_6b_cuda_fortran_validation.md` §19-§22.
- Thirteen input-fidelity/layout defects (grid metrics, `bdt`, `ptop`/`zs`,
  `dp_ref`, `rdgas`, `n_sponge`, `edge_*` weights, Exner-kappa split,
  `grid_lonlat` store order, `cv_air`, `n_con`, corner service, `nf_ke`)
  found/fixed during Gate C2 closure — same reference.
- CUDA Graph capture mode: FUNCTIONAL, IMPLEMENTED (Prompt 8, repaired
  Prompt 13), Fortran-reachable via `RUN_CUDA_FV3_GRAPH` since Prompt 29.
  Captures `dgrid_dynamics` only (halo NOT captured); freezes `dt` by value
  (refuses replay if `bdt`/`n_split` change). Bitwise-identical to non-graph
  path at every depth tested. Real benefit: 4.80%, LOW, see §8/§9.
- Launch-fusion optimizations, opt-in/default-off, bitwise-identical to
  default: `--fuse-corner-fill`/`RUN_CUDA_FV3_FUSE_CORNER_FILL` (Prompt 9,
  reachable Prompt 28), `--fuse-halo-edges`/`RUN_CUDA_FV3_FUSE_HALO_EDGES`
  (Prompt 11, reachable Prompt 28). `a2b4_global_corner_kernel` restructured
  for parallelism (Prompt 12); also bitwise-identical, always-on.
- Idealized Held-Suarez dry-core forcing: IMPLEMENTED, COMPILED, GPU
  EXECUTED, NUMERICALLY VALIDATED at C8-C512 — see §9a.

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
| R4 | c8 | 1 | 72 dyn_core calls (Stage A) | **PASS** | `omga` bulk-ratio grows superlinearly to ~4% (magnitude-decay artifact, not divergence) |
| R4 | c8 | 2 | 12 dyn_core calls (Stage B) | **PASS** | no discontinuity at remap boundary |
| R8 | c16 | 1 (bdt=1800, resolution-scaled) | 6 calls | **PASS** | `w` max_rel ≈6.7e-11 |
| R8 | c32 | 1 (bdt=1200, resolution-scaled) | 6 calls | **PASS** (1 explained anomaly) | `mfyL` dce-entry-only 2-cell uninitialized-memory artifact, both backends; exit-state agreement clean |
| R4 | c16 | 1 (bdt=1800, reused from R8) | 6 calls | **PASS** | `w` max_rel 6.80e-2 (dcx call 3) — small-denominator artifact |
| R4 | c32 | 1 (bdt=1200, reused from R8) | 6 calls | **PASS** (largest ratio, explained) | `w` max_rel 2.39e-1 (dce call 3) — same artifact, scaled; absolute agreement ≤5.9e-5 |
| R8 | c32 | 1 | Graph ON vs OFF, 10 calls (Prompt 29) | **PASS** | `dcx`/exit: BITWISE; `dce`/entry: same `mfyL` artifact, reproduced identically both modes |

Grading scale: BITWISE (0 ULP) / ROUND-OFF SCALE (R8 ≤1e-13, R4 ≤1e-6) /
SMALL NUMERICAL DIFFERENCE (R8 ≤1e-8, R4 ≤1e-2) / SIGNIFICANT (above those).
R4 thresholds derived from the measured R4/R8 max_rel ratio (≈4-6e8,
matching float32/float64 epsilon ratio), not assumed.

Notes: `omga`'s R4 growth is driven by the field's own magnitude shrinking,
not absolute discrepancy growing (plateaus ~call 43); `pe`'s single
Fortran halo ring reproduced host-side via `fv3_gpu_interface_pe_halo`
(bit-for-bit) for `Lagrangian_to_Eulerian`; zero NaN/Inf, `delp` stays
positive, all PASS rows; numerical algorithm changed **NO** every row.
Full field-by-field tables: `prompt_6b_cuda_fortran_validation.md`,
`prompt_19`-`prompt_23` handoffs, `prompt_29_standalone_cuda_graph.md`.

## 6. MPI / Communication Status

- Cross-process six-rank halo transport (`MpiHaloTransport`): IMPLEMENTED,
  COMPILED, GPU EXECUTED. Validated bitwise-identical to in-process
  `PeerHaloTransport` (78/78 fields, 1/5 calls) plus a six-rank provenance
  test (16,692 checks/0 failures). Rank<->tile mapping is identity.
- **MPI transport determinism: RESOLVED** (Prompt 18) — root cause was a
  CUDA stream-ordering defect in the *test harness*, not the transport;
  fixed (`cudaMemsetAsync` on the transfer stream at attach-time).
  **Standing constraint**: any buffer setup a non-blocking stream will read
  must be enqueued on that stream, not the legacy stream.
- **FMS/CUDA halo ownership ambiguity: RESOLVED** (Prompt 14/15) — CUDA
  branch skips `i_pack(1)`/`i_pack(8)`, keeps `i_pack(10)` (tracer).
  Default Fortran path unchanged.
- CUDA-aware MPI (Stage 2) and comm/compute overlap (Stage 3): unimplemented
  future work. Priority was LOW at C32-C128 (§8), MODERATE (contingent) at
  C256/C512 after Prompt 38, and **raised to HIGH CANDIDATE after Prompt
  39's discriminator isolated the C512 regression to the MPI transport
  layer specifically** (in-process `PeerHaloTransport` scales normally at
  C512; six-rank `MpiHaloTransport` does not) — see §8a/§9. Still not
  implemented; one MCA-threshold discriminator recommended before any
  implementation begins.

## 7. Standalone Integration Status

- CUDA dyn_core runs inside the real `StandAlone_FV3_Dycore.x` under the
  true production decomposition (`FV_NX=1`/`FV_NY=6`, six MPI ranks, one
  tile/rank), same executable as Fortran (selected via `AGCM.rc`'s
  `RUN_CUDA_FV3`).
- Public C ABI for per-run setup is complete; no internal-only C++ entry
  point needed.
- Numerically validated at the depths in §5 (production) and §9a (HS) —
  all PASS.
- ABI/compiler-family compatibility with a full **Intel-compiler production
  GEOS build** verified sound by construction (Prompt 30: fixed-width
  `iso_c_binding` types, runtime `abi_version`/`struct_bytes` guards,
  explicit `MPI_Comm_f2c`). Revisit if the toolchain changes.

## 8. Performance Status (Production dyn_core, C32 baseline methodology)

Current best validated standalone-path configuration: **MPS ON +
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`**.
`RUN_CUDA_FV3_GRAPH` available (Prompt 29) but NOT the recommended default
(LOW benefit). Figures below: real standalone six-rank MPI path
(`StandAlone_FV3_Dycore.x`, c32, R8, `n_split=3`/`k_split=1`, `bdt=1200s`,
6 MPI ranks/1 A100); `fv_timing_mod`'s `DYN_CORE` timer, 100 calls/rep,
median of 3-5 reps, savepoints off.

| Stage | Config | ms/call | vs. prior |
|---|---|---|---|
| Fortran reference | 6 ranks, CPU | 19.930 | — |
| CUDA, no MPS, no fusion (Prompt 24) | 6 ranks, 1 A100, time-sliced contexts | 105.563 | 5.30x slower |
| + CUDA MPS (Prompt 26/27) | same + MPS ON | 27.075 / 26.965 | 3.92x faster |
| + fuse_halo_edges + fuse_corner_fill (Prompt 28) | same + both flags | 25.464 | −6.13% |
| + CUDA Graph (Prompt 29) | same + `RUN_CUDA_FV3_GRAPH=YES` | 24.300 | −4.80% (LOW, not default) |

Net at C32: CUDA 24.300 ms/call vs. Fortran 19.930 = **1.22x slower** (was
5.30x pre-MPS). All steps bitwise/numerically neutral. **See §8a: at
C64/C128/C256 under the same mode, CUDA overtakes Fortran** — gap reverses
with resolution (then reverses back at C512).

Root-cause chain (detail in cited files): MPS is the dominant lever
(Prompt 26: bottleneck was 6 CUDA contexts time-slicing one A100); post-MPS
profile MIXED/diffuse (Prompt 27: `cudaLaunchKernel` ~51% of `DYN_CORE`, no
kernel >5.6%); launch fusion cut launches 2955->2637/call (−10.76%);
CUDA Graph cut launches further (2637->1805/call, −31.6%) but whole-call
gain only 4.80% (`cudaEventSynchronize` rose instead) — BITWISE,
replay-stable, not the recommended default.

CUDA-aware MPI, kernel-tuning, rank/GPU redesign were LOW/NOT JUSTIFIED at
C32 (halo traffic 312 msgs/call all ≤64KB; device transfer ~2% of
`DYN_CORE`; MPS at the 1-context ceiling) — Prompts 25a/26. **Superseded
at C256/C512, see §8a.**

`ncu` profiling remains platform-blocked (`ERR_NVGPUCTRPERM`); `nsys`
unaffected, used for all profiling above.

## 8a. Held-Suarez / Resolution Performance Scaling

30 dyn_core calls/rep, 3 reps/config, savepoints off, standing MPS+fusion
mode, `fv_timing_mod` flat `DYN_CORE` FMS-clock line (total_s/n_calls,
warmup-included — not directly comparable to §8's warmup-excluded C32
figure, but internally consistent C32-C512).

| Grid | Fortran ms/call | CUDA ms/call | CUDA/Fortran |
|---|---|---|---|
| C32 | 20.72 | 19.88 | **0.96 — CUDA faster** |
| C64 | 59.4 | 56.9 | **0.96 — CUDA faster** |
| C128 | 237.4 | 210.9 | **0.89 — CUDA faster** |
| C256 | 1250.75 | 1092.44 | **0.874 — CUDA faster (best)** |
| C512 | 7155.63 | 9089.05 | **1.270 — CUDA SLOWER (REGRESSES)** |

Rep-level totals: `prompt_34_c32_perf_reconciliation.md` /
`prompt_36_c256_correctness_performance.md` / `prompt_37_c512_
correctness_performance.md`. All reps rc=0, 0 NaN/Inf/Range_Warn/error,
30-cycle count confirmed.

**Trend**: widens favorably C32->C256, reverses sharply at C512 (first
reversal in the ladder). **Root cause (Prompt 38, diagnostic-only — 0 new
instrumentation/nsys/jobs, static geometry + existing logs)**: halo
message size (fixed message/invocation count every resolution; only
per-message size scales) crosses the small-message regime the LOW
CUDA-aware-MPI rating relied on — C256 ordinary msg ≈67.8KB already
exceeds the ~64KB anchor (Prompt 25a); C512 ≈135.4KB, ~2x again.
`COMM_DSW`'s CUDA-Fortran delta grows **19.5x** (97.7ms/call at C256 ->
1901.5ms/call at C512) for only ~2x message-size increase —
disproportionate, not bandwidth-proportional. Every halo-adjacent compute
block (`C_SW`/`D_SW`/`UPDATE_DZ`/`PG_D`) flips from a CUDA win at C256
(0.79-0.87x) to a loss at C512 (1.34-1.44x); halo-independent local-solve
blocks (`RIEM_SOLVER`/`UPDATE_DZ_C`) stay flat (~1.0x) at both — corroborates
a halo-driven mechanism beyond the `COMM_*` labels alone. Pure D2H/H2D
bandwidth bounded (≈10ms/call estimated, scaled from Prompt 25a's C32
measurement) and ruled out — two orders of magnitude below the observed
growth. MPS/6-context contention (Prompt 26) is resolution-independent —
ruled out as the resolution-specific driver.

**Mechanism isolated (Prompt 39)**: the existing `fv3_gpu_perf_bench`
harness's in-process `PeerHaloTransport` path (which shares all topology/
pack/unpack code with `MpiHaloTransport`) was run at C256 and C512
(`--mode resident`, fusion flags on, Graph off, job 58445168, rc=0, 0
FAIL/NaN/Inf). Its C256->C512 growth factor is **3.69x** — below the
Fortran path's own 5.72x growth, and far below the six-rank
`MpiHaloTransport` path's **8.32x** growth. PeerHalo stays healthy at
C512 while MpiHaloTransport collapses, isolating the regression to the
**MPI transport/protocol layer specifically**, not the shared device-side
pack/unpack/staging mechanism. **CUDA-aware-MPI priority raised
MODERATE(contingent)->HIGH CANDIDATE.** One next experiment recommended
(OpenMPI eager/rendezvous MCA-threshold sweep at C512 vs. the 9089.05
ms/call baseline), not yet run — see `prompt_39_c512_peerhalo_
discriminator.md`. No optimization/CUDA-aware-MPI implementation
attempted; numerical source unchanged; correctness (§9a) BITWISE,
unaffected.

## 9. Known Limitations / Unvalidated Envelope

- R4/R8 production (non-HS) validation covers c8 (deepest: 72/12-call
  ladders) and c16/c32 (6-call depth only, Prompt 23). **HS validation
  (§9a) separately covers c8/c32/c64/c128 up to 10-day duration, plus C256
  (24h gate) and C512 (6h gate)** — do not conflate the two ladders.
  C256/C512 each have only a short bitwise gate, not a multi-day run.
- **C512 CUDA performance regression**: CUDA ~27% SLOWER than Fortran at
  C512 (§8a), first reversal in the C32->C512 ladder. **Isolated to the
  MPI transport/protocol layer (Prompt 39)**: `COMM_DSW` = 97.6% of the
  `DYN_CORE` delta (Prompt 38), and the in-process `PeerHaloTransport`
  path — which shares all device-side pack/unpack code with
  `MpiHaloTransport` — scales normally at C512 (3.69x) while the six-rank
  MPI path collapses (8.32x). One MCA-threshold discriminator recommended
  before any CUDA-aware-MPI implementation (§8a/§11). No fix attempted.
  Do not assume CUDA competitive above C256 without remeasuring.
- `MOIST_CAPPA`/`USE_COND`: **confirmed UNIMPLEMENTED in CUDA** (Prompt 30)
  — triple-gated rejection, by design predating this prompt (milestone
  3A.9). Closing needs the Riemann-solver `peg`/`gm2`/`cp2` terms
  (`nh_utils.F90`) in CUDA, a dedicated future prompt.
  See `prompt_30_production_config_closure.md`.
- `a_imp <= 0.999` (RIM_2D/SIM3p0 vertical branches): **confirmed
  UNIMPLEMENTED in CUDA** (Prompt 30), triple-gated. Only SIM1
  (`a_imp=1.0`) implemented. `scale_m`/`m_split`/`delt_max`/`scale_z`
  remain proven inert at `a_imp=1.0`.
- `ncu` profiling platform-blocked (`ERR_NVGPUCTRPERM`); `nsys` unaffected.
- `fv3_dispatch_smoke` Phase C fails if its own `cfg.nf_ke` pin is raised
  from 1 to authoritative 3 — not localized, test pin left at 1; not
  evidence against production corner service (validated at `nf_ke=3` in
  the real dyn_core path, §5).
- CUDA MPS (`fv3_mps_lifecycle.sh`, Prompt 27) requires
  `compute_mode=Default` (true on Discover `gpu_a100`). Not yet wired into
  every standing job script as the default (one-off scripts only so far).
- Launch fusion and CUDA Graph: default OFF, bitwise-safe, tested at
  c32/R8 (§8) and exercised at c64/c128 R8 HS runs (§9a) without
  regression. Graph's refusal-on-`dt`/`n_split`-change behavior not
  operationally re-exercised recently.
- Multi-GPU (`dev = mod(comm_rank, device_count)`) implemented but
  validated only at 2 GPUs/3 contexts-per-GPU; 4-GPU and MPS+multi-GPU
  composition unmeasured.
- `install/bin/StandAlone_FV3_Dycore.x` and per-experiment binary copies
  not auto-synced with `build/bin/` after rebuild — always refresh and
  verify via `strings` before trusting a new flag.
- **MAPL restart (`COLDSTART: 0`)**: **confirmed BROKEN** for this
  HS/nonhydrostatic standalone config (Prompt 35, Fortran backend,
  Opus-escalated: genuine source-level gap in the nonhydrostatic
  warm-restart path, not a CUDA/config issue). CUDA restart correctly
  skipped once Fortran's own failed. Use fresh `COLDSTART: 1` runs
  instead. Every HS run through Prompt 38 uses `COLDSTART: 1`.
- **C128 timestep search (Prompt 33)**: first `bdt` candidate CFL-unstable
  (NaN flood); resolved with a smaller `bdt`. Do not assume the C32
  resolution-scaled `bdt` heuristic (§5) extends linearly to higher
  resolutions without a short stability check first (C128/C256/C512 each
  needed their own search, though C256/C512 stabilized on their first
  candidate — do not assume that holds at C1024+).
- **§8a caveat**: 30-call/3-rep, warmup-included — not directly comparable
  to §8's warmup-excluded 100-call/rep C32 figure.

## 9a. Held-Suarez Validation Ladder (branch `experiment/held-suarez-e2e`)

Idealized dry-core Held-Suarez forcing, orthogonal to the `cuda/gpu-bringup`
mainline (§3/§4). All entries: Fortran vs CUDA, same standalone fixture,
6 MPI ranks, R8 unless noted, savepoint and/or NetCDF-level comparison.

| Grid | Duration | Result | Notes |
|---|---|---|---|
| C8 | 72 outer steps (144 calls) | **PASS** | `dcx` max_rel 7.999e-11 (ROUND-OFF/SMALL); Prompt 31 |
| C32 | short gate | **PASS** | savepoint 85,310/87,810 fields BIT-IDENTICAL, 2,500 FAIL confined to benign halo-padding; NetCDF T@500hPa bitwise; Prompt 32 |
| C64 | short gate | **PASS** | NetCDF bitwise; Prompt 33 |
| C128 | short gate | **PASS** | first `bdt` candidate CFL-unstable, resolved with smaller `bdt`; NetCDF bitwise; Prompt 33 |
| C128 | 10-day | **PASS** | rc=0, byte-identical logs, 80 nc4 each; bitwise T/U/PS start/mid/end; `range_warn=6966` both backends, classified benign; Prompt 33 |
| C128 | 30-day | **PASS** | rc=0, bitwise T/U/PS day 1/5/10/20/30; day 30 = active spin-up; Prompt 35 |
| C128 | 100-day | **PASS** | rc=0, byte-identical logs, bitwise T/U/PS through day 100; jet NH/SH symmetrized vs day-30; not yet fully equilibrated (~200+ days lit. estimate); Prompt 35 |
| C256 | dt-search (1 cand.) + 24h gate | **PASS** | `bdt=75s` stable 1st try; byte-identical logs, 0 NaN/Range_Warn/Error; T/U/V/PS at 4 checkpoints **BITWISE**; Prompt 36 |
| C512 | dt-search (1 cand.) + 6h gate | **PASS** | `bdt=37s` stable but dropped final checkpoint (divisibility, §12) — fixed to `bdt=36s` (fix #1/3); byte-identical logs, 0 NaN/Range_Warn/Error; T/U/V/PS at 3 checkpoints **BITWISE**; Prompt 37 |

Benign known artifacts (do not treat as new divergences):
- 2-cell halo-padding uninitialized-memory difference in `delpc`/`pkc` at
  savepoint level only, IEEE-754-subnormal magnitude — same class as the
  Prompts 22/23 `mfyL` finding. Never present at the NetCDF level.
- C128 10-day `range_warn` (6966 hits, both backends): `TA_dyn` level 10,
  119 unique points, confirmed benign (sorted-diff = 0, content-identical
  modulo MPI-rank interleaving) — legitimate HS near-surface warmth
  exceeding a hardcoded FMS plausibility bound, not a defect.
- Known near-surface NaN artifact (T/U/V, 3 lowest of 25 levels, standard
  MAPL vertical-interpolation masking) identical both backends at every
  resolution tested — pre-existing, not new.

Not attempted: duration beyond 100 days at C128, C256/C512 duration beyond
their short gates, resolution beyond C512, moist-physics coupling,
warm-restart-continuation (confirmed broken, §9), formal climatology
statistics, and a fix for the C512 comm regression (§8a — root-caused,
not fixed).

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`. Mainline branch `cuda/gpu-bringup`
  HEAD `2e6327f` ("Document MPS rank-GPU performance architecture"), ahead
  of `origin/cuda/gpu-bringup`, not pushed. HS track branch
  `experiment/held-suarez-e2e` HEAD `a76b429` ("Document Held-Suarez
  performance scaling"), also not pushed. Push/merge only on explicit
  request. Prompts 35-38's `docs/agent_handoff/*.md` and this
  `CURRENT_STATE.md` update are uncommitted, awaiting explicit commit
  request. No source files changed in Prompt 37, 38, or 39 (38 was
  diagnostic-only: static derivation + reads of existing logs, no job
  submitted; 39 ran one new GPU job against the existing, unmodified
  `fv3_gpu_perf_bench` binary — no file under `cuda/`/`include/` or the
  Fortran fixture tree touched in either).
- Uncommitted source on `cuda/gpu-bringup` (Prompt 28+29, awaiting explicit
  commit request): `cuda/interface/fv3_gpu_dyncore_bridge.F90`
  (`env_flag_enabled` helper + fusion/Graph env wiring),
  `cuda/interface/fv3_gpu_abi_mod.F90` (`fv3_gpu_set_cuda_graph_mode`).
- Untracked at repo root (both branches share the worktree): numerous
  scratch shell scripts/logs/probes from prior prompts, reusable tooling
  under `tools/` (savepoint validation — `compare_dcx_backends.py`,
  `tools/savepoints/savepoint_reader.py`; `tools/analysis/
  hs_netcdf_inspect.py`, Prompt 32), and `perf_rerun_logs/`. Untracked
  handoff docs for every prompt from 22 through 38 (see §12). In the
  separate Fortran fixture tree `<fortran_perf>/experiments/`: per-prompt
  sbatch scripts/`.out` logs/job directories for every prompt 24-37 (each
  perf job pair keeps `perf_logs/rep{1,2,3}.log`), and
  `fv3_mps_lifecycle.sh`.
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
   flags into every standing job script as the default. Lower-risk than
   1/2.

Until 1 or 2 closes, "broader GEOS integration" remains bounded to the
already-validated production first-run subset (`hord_*=±6`, `grid_type=0`,
`hydrostatic=false`, `a_imp=1.0`, `use_cond=moist_cappa=0`).

**HS track — Prompt 39 isolated the C512 performance regression (§8a/§9)
to the MPI transport/protocol layer specifically**, via the PeerHalo-vs-
MPI discriminator. C512 correctness remains BITWISE (§9a, unaffected).
Highest-priority open HS-track item, now narrowed to one specific MCA-
level next experiment (below), ahead of any CUDA-aware-MPI implementation.

Open HS-track follow-ons (confirm scope before starting):
1. **Run the recommended MCA-threshold discriminator**: rerun the
   existing six-rank `MpiHaloTransport` C512 job (Prompt 37) under an
   alternative OpenMPI eager/rendezvous-limit MCA setting (raised above
   C512's ~135.4KB ordinary message size) and compare `COMM_DSW`/
   `DYN_CORE` timing against the 9089.05 ms/call baseline — pure runtime
   config, no source change. Confirms or refutes the eager/rendezvous
   crossover before CUDA-aware MPI (now HIGH CANDIDATE) is implemented.
   See `prompt_39_c512_peerhalo_discriminator.md`.
2. **C256/C512 climatology / longer duration** — only short gates have run
   at either (§9a); no multi-day run attempted.
3. **Further duration at C128** toward the ~200+ day HS-literature
   equilibration estimate — day 100 closer but not yet flat; not started.
4. **Warm-restart fix** — dedicated future prompt to close the confirmed
   nonhydrostatic warm-restart gap (§9).
5. **Higher resolution than C512** (C1024+) — unmeasured; needs its own
   stability search; competitiveness cannot be assumed either way, and
   depends on item 1's outcome if pursued.

## 12. Detailed Handoff References

Chronological/technical detail for every closed or attempted prompt below
remains in its own file under `docs/agent_handoff/`; read the specific file
only when the current task requires that level of detail (per
`docs/agent_instructions/cost_efficiency.md`).

- `prompt_6a_fortran_savepoints.md`/`prompt_6b_cuda_fortran_validation.md`
  — authoritative Fortran savepoints, Gate C2 closure.
- `prompt_7_post_gate_c2_performance_baseline.md` — post-Gate-C2 nsys
  baseline. `prompt_8_cuda_graphs_prototype.md` — historical, retracted by
  Prompt 13. `prompt_9`-`prompt_12` — launch-fusion/kernel-tuning sequence.
  `prompt_13_architectural_optimization.md` — CUDA Graph capture repair.
- `prompt_14_standalone_cuda_integration.md` — FMS/CUDA halo resolution.
  `prompt_15_mpi_halo_transport.md` — `HaloTransport`/`MpiHaloTransport`.
  `prompt_18_mpi_transport_determinism.md` — MPI test-flake root cause/fix.
- `prompt_19`-`prompt_21` — R8/R4 c8 depth validation. `prompt_22`/
  `prompt_23` — c16/c32 CFL blocker + resolution-scaled `bdt` validation.
- `prompt_24_standalone_performance_validation.md` — real-path nsys
  bottleneck classification. `prompt_25a_fv3_halo_characterization.md` —
  halo census, original LOW CUDA-aware-MPI ceiling (superseded at
  C256/C512 by Prompt 38).
- `prompt_26_rank_gpu_architecture.md` — MPS 3.92x. `prompt_27_mps_
  operational_profile.md` — MPS lifecycle helper, post-MPS profile.
- `prompt_28` — fusion flags, MODERATE (6.13%). `prompt_29` — CUDA Graph,
  LOW (4.80%). `prompt_30_production_config_closure.md` —
  `MOIST_CAPPA`/`USE_COND`/`a_imp<=0.999` unimplemented by design; Intel
  ABI verification PASS. `milestone_3a*.md`/`wp-*.md` — pre-Gate-C2
  work-package history.
- `prompt_31_held_suarez_e2e.md` — HS forcing smoke test (C8). `prompt_32_
  held_suarez_scientific_validation.md` — C32 NetCDF/savepoint validation.
- `prompt_33_hs_multiday_highres_performance.md` — HS ladder to C64/C128,
  10-day duration, scaling table. `prompt_34_c32_perf_reconciliation.md`
  — C32 perf re-measured under Prompt 33's methodology.
- `prompt_35_hs_climatology.md` — warm-restart confirmed broken
  (Opus-escalated); 30/100-day matched C128 climatology runs, bitwise to
  100 days; 3 job-config bugs fixed.
- `prompt_36_c256_correctness_performance.md` — C256 dt-search, 24h
  BITWISE gate, perf point (0.874, widening trend).
- `prompt_37_c512_correctness_performance.md` — C512 dt-search;
  `RUN_DT`/`JOB_SGMT` divisibility defect fixed (`bdt`->36s); 6h BITWISE
  gate; perf point reversing the trend (1.270, REGRESSES), lightweight
  COMM_*-block attribution.
- `prompt_38_c512_comm_root_cause.md` — diagnostic-only (0 new
  instrumentation/nsys/jobs) static-geometry + log-based root cause:
  message-size-regime-driven disproportionate halo-transport cost growth;
  CUDA-aware-MPI priority LOW->MODERATE (contingent); 4 ranked
  unimplemented remediation options; 1 recommended discriminator
  experiment (PeerHaloTransport at C512).
- `prompt_39_c512_peerhalo_discriminator.md` — 1 GPU job (58445168),
  existing `fv3_gpu_perf_bench` harness, C256+C512 `PeerHaloTransport`
  runs (`--mode resident`, fusion on, Graph off); PeerHalo growth 3.69x
  vs. Fortran 5.72x vs. six-rank MpiHaloTransport 8.32x; isolates the
  C512 regression to the MPI transport/protocol layer; CUDA-aware-MPI
  priority MODERATE->HIGH CANDIDATE; halo sub-timers NOT AVAILABLE in
  harness; 1 recommended MCA-threshold-sweep next experiment, not
  implemented.

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
