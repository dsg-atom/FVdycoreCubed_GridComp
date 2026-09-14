# Current Project State

_Last updated 2026-09-14 (Prompt 26 pass: rank/GPU execution-architecture
characterization — six CUDA contexts time-slicing one A100 was the dominant
performance cost, and CUDA MPS recovers 3.92× with no source change; see
§8/§9/§11). Full chronological detail for every
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

**Corrected attribution (Prompt 24).** The table below (job 58238910,
Prompt 13) and the `milestone_3a12` §8 "CUDA reference" numbers were both
measured on the internal single-process `run_small_perf_bench` C++ harness
(`--ntasks=1`, no MPI, all six tiles resident in one process) — **not** on
the real standalone `StandAlone_FV3_Dycore.x` six-rank MPI integration path.
Confirmed by direct inspection of `perf_rerun_logs/nsys_profile.sbatch` and
the CLI-flag source locations (`cuda/dyn_core/CMakeLists.txt`,
`benchmarks/run_small_perf_bench.cpp`). Neither `--fuse-halo-edges` nor
`--cuda-graph` is reachable from the Fortran/standalone integration path at
all — `fv3_gpu_dyncore_bridge.F90:604-605` hardcodes
`cfg%fuse_corner_fill=0`/`cfg%fuse_halo_edges=0` at config install, and no
CUDA Graph field exists in the ABI. Kept below for internal-harness history
only; do not cite as standalone-integration performance.

| Configuration (internal harness only) | ms/call | vs. reference |
|---|---|---|
| Graph off (reference) | 21.660791 | — |
| Graph on (`--cuda-graph`) | 20.546787 | −5.14% |

**Real standalone-integration baseline (Prompt 24, authoritative for the
production path).** c32, R8, `npz=10`, `n_split=3`, `k_split=1`,
`bdt=1200s`, 6 MPI ranks (`FV_NX=1`/`FV_NY=6`, one tile/rank), one A100/one
node, real host-staged MPI (Intel MPI, not OpenMPI — the runtime launcher
for this Intel-compiled fixture, distinct from the OpenMPI module used for
CUDA compilation). Both backends measured via `fv_timing_mod`'s existing
`DYN_CORE` timer, 100 calls/rep, 5 reps, savepoints disabled to avoid timer
contamination. Job 58295284, all reps rc=0/0 Range_Warn/0 NaN.

| Backend | median ms/call | spread |
|---|---|---|
| Fortran (6 ranks, CPU) | 19.930 | 0.6% |
| CUDA (6 ranks, 1 A100) | 105.563 | 0.74% |

CUDA is **0.189× (~5.30× slower)** than Fortran per `dyn_core` call in this
real integration — the opposite of the internal-harness picture above. No
CUDA runtime perf flags are enabled or selectable here; this is the only
CUDA execution path this integration currently exposes.

nsys diagnostic breakdown (10-call rank-0 trace, job 58295418, diagnostic
only, not part of the timed result): host synchronization
(`cudaEventSynchronize`) ~59.27 ms/call (~56%), launch/orchestration
(`cudaLaunchKernel`) ~17.33 ms/call (~16%), full GPU kernel-compute sum (all
171 distinct kernels) ~11.19 ms/call (~11%), H2D+D2H copy ~0.66 ms/call
(<1%). **Dominant bottleneck: HOST-SYNCHRONIZATION-BOUND**, with
host-staged MPI (six separate CUDA contexts sharing one A100, one per MPI
rank) as a corroborated secondary factor; both H2D/D2H-transfer-bound and
kernel-compute-bound are ruled out by direct measurement. Detail, full
per-rep tables, and reasoning: `prompt_24_standalone_performance_validation.md`.
**Prompt 26 re-attributes the ~56% host-synchronization term**: most of it was
ranks blocked on a *time-sliced shared GPU*, not on MPI or on copies — see
"Execution architecture" below. Re-profile under MPS before using this
breakdown to choose any further optimization.

**Execution architecture and CUDA MPS (Prompt 26 — the dominant lever).**
Confirmed from source and runtime, not assumed: the standalone production
mapping is `6 MPI ranks -> 6 tiles (1/rank) -> 6 primary CUDA contexts ->
1 A100`, time-sliced, 2 `cudaStreamNonBlocking` streams per context, MPS not
in use (a node MPS daemon exists but belongs to another user; this account
never attached to it). One tile per PE is fixed by the authoritative FMS
decomposition (`fv_mp_mod.F90:662`), so multi-tile-per-rank is a Fortran
decomposition redesign, not a CUDA change.

Context count on the shared GPU — not transport, not kernels — is the
dominant cost. Measured three independent ways at c32/R8:

| Experiment | 1 ctx | 2 ctx | 3 ctx | 6 ctx |
|---|---|---|---|---|
| Synthetic probe, total work fixed, no MPI (job 58380039) | 28.085 | 30.711 | 35.724 | 72.484 ms/call (**2.58×**) |
| Real dyn_core, internal harness (job 58380039) | 28.725 | — | — | 102.980 ms/call (3.59×, transport confound recorded) |
| Real standalone path, ctx/GPU via `CUDA_VISIBLE_DEVICES` (job 58380134) | — | — | 54.410 (2 GPUs) | 106.007 ms/call (1.95×) |

Scaling is superlinear; the knee is between 3 and 6 contexts.

**CUDA MPS result (job 58380090, same node/GPU/config/binary, zero source
change, 3 reps each):** `DYN_CORE` median 106.261 ms/call MPS-off →
**27.075 ms/call MPS-on = 3.924×**; whole-run `TOTAL` 14.7257 s → 5.8631 s
(2.51×); `COMM_TOTAL` 0.1151 s → 0.0296 s (3.89×); `REMAPPING` unchanged.
The MPS-off leg reproduces Prompt 24's 105.563 within 0.7%. Against the
same-node Fortran 19.930 ms/call, CUDA goes from **0.188× (5.33× slower)** to
**0.736× (1.36× slower)**. MPS on one GPU beats two GPUs without MPS
(27.1 vs 54.4) and reaches the 1-context ceiling (28.1-28.7).

MPS is **bitwise-neutral**: 1230/1296 savepoints byte-identical vs the
Prompt 23 c32/R8 reference; the 66 that differ are all `dce_*_mfyL.bin` /
`dce_*_cxL.bin` dce-*entry* dumps, and an MPS-off control re-run (job
58380203) differs in exactly the same 66 files — i.e. the already-documented
§5 uninitialized-read artifact, not an MPS effect. No exit/state field differs.

Current priority call (Prompts 25a + 26, supersedes Prompt 24's "CUDA-aware
MPI: HIGH"): **rank/GPU context sharing — RESOLVED OPERATIONALLY via MPS**;
**CUDA-aware MPI priority: LOW** (halo traffic is 312 messages/call of only
8712 B and 2904 B, 100% ≤64 KB, device-direct ceiling ~1.02-1.04×, and
`COMM_TOTAL` fell 3.89× under MPS alone, showing the "MPI wait" was GPU
serialization); **kernel-tuning priority: LOW** (compute ~11% of per-call
cost); **rank/tile ownership redesign (multi-tile rank): NOT JUSTIFIED** (it
cannot beat the 1-context ceiling MPS already reaches for free).
Detail: `prompt_25a_fv3_halo_characterization.md`,
`prompt_26_rank_gpu_architecture.md`.

Fortran R8 median `DYN_CORE` ms/call reference (6 MPI ranks, Intel/AVX2,
unchanged since Prompt 7): 8×8 12.730, 16×16 16.704, 32×32 30.261 — the
32×32 figure differs slightly from Prompt 24's 19.930 because Prompt 7 used
a different `bdt`/run depth; not a discrepancy, just a different sample.

`ncu` (Nsight Compute) profiling remains platform-blocked
(`ERR_NVGPUCTRPERM`) on this Discover allocation — carried forward from
Prompt 13, not re-tested in Prompts 24/26 since compute was never the
bottleneck; `nsys` remains usable and was used for all profiling above.

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
  unimplemented future work — and now measured LOW priority (§8).
- **CUDA MPS is an operational configuration, not yet integrated.** The 3.92×
  `DYN_CORE` gain in §8 requires a per-node user MPS daemon
  (`CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` + `nvidia-cuda-mps-control
  -d`, torn down with `quit`) started by the job script; no GEOS/standalone job
  script does this yet. Requires `compute_mode=Default` (true on Discover
  `gpu_a100`). Not tested at grids other than c32, at depths beyond 100
  `dyn_core` calls, at R4, or across multiple nodes.
- Multi-GPU (candidate F) is already implemented (`dev = mod(comm_rank,
  device_count)`) but was validated only at 2 GPUs / 3 contexts per GPU;
  `--gres=gpu:4` would not schedule during Prompt 26, so 4-GPU and
  MPS+multi-GPU composition are unmeasured.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`, branch `cuda/gpu-bringup`.
- HEAD: `6817f8d` ("Document wider-grid FV3 CUDA validation"). Ahead of
  `origin/cuda/gpu-bringup` — not pushed. Push/merge only on explicit request.
- Tracked modification: this file only (`docs/agent_handoff/CURRENT_STATE.md`).
- Untracked at repo root: numerous scratch shell scripts/logs/probes from
  prior prompts (`task*_6c_cuda.sh`, `sp*rerun*.sh`, `p1[7-8]_*.sbatch`,
  `cln*.txt`, `cmp_nord*.txt`, `det_diff_*.txt`, `pre*.txt`, `scratch_*/`,
  build helpers) plus reusable tooling directories `tools/` (savepoint
  validation, e.g. `compare_dcx_backends.py`, `verify_omega_host.cpp`) and
  `perf_rerun_logs/` (sbatch scripts + job logs for Prompts 7-13). Untracked
  handoff docs: `prompt_22_wider_grid_validation.md`,
  `prompt_24_standalone_performance_validation.md`,
  `prompt_25a_fv3_halo_characterization.md`, and (created this pass)
  `prompt_26_rank_gpu_architecture.md`. Also untracked from this pass:
  `scratch_p26/` (context-scaling probe `ctx_scaling_probe.cu`, its sbatch
  driver, job logs) and, in the separate Fortran fixture tree
  `<fortran_perf>/experiments/`, `p26_*.sbatch` + `p26_*.out`.
- Do not commit: build directories (`build_cuda_r8*/`), profiler artifacts
  (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm log.
- Do commit (if explicitly requested): legitimate source fixes, permanent
  test/tooling additions under `tools/`, documentation.

## 11. Next Recommended Task

The standalone CUDA performance gap is now explained and mostly closed
operationally (§8): six MPI ranks time-slicing one A100 through six separate
CUDA contexts was the dominant cost, and enabling CUDA MPS — no source,
topology, or numerical change — gives 3.92× on `DYN_CORE` (106.3 → 27.1
ms/call) and 2.51× whole-run, moving CUDA from 5.33× slower than Fortran to
1.36× slower, bitwise-neutral. This **supersedes** Prompt 24's "CUDA-aware
MPI: HIGH" call; CUDA-aware MPI, kernel tuning, and rank/tile-ownership
redesign are all now LOW priority or not justified.

Recommended next step: **integrate MPS operationally and re-profile under
it.** Specifically (a) add MPS daemon start/teardown to the standalone (and
later GEOS) job scripts as a documented, opt-in operational setting, and
(b) take a short `nsys` trace *under MPS* before choosing any further
optimization, since MPS invalidates Prompt 24's ~56%-host-synchronization
bucket attribution. Confirm scope with the user first, per
`project_architecture.md`'s "Performance Work Comes Later".

Alternative track, unchanged: production GEOS integration (moist
`q_con ≠ 0` / `MOIST_CAPPA`/`USE_COND`, `a_imp <= 0.999` knobs,
Intel-compiler ABI verification). Confirm scope with the user before starting
either track.

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
- `prompt_24_standalone_performance_validation.md` — real standalone-path
  Fortran-vs-CUDA timing (§8), the internal-harness-vs-real-path attribution
  fix, and the nsys-based bottleneck classification.
- `prompt_25a_fv3_halo_characterization.md` — halo message-size/count census
  (24 transports, 312 messages, 8712 B / 2904 B only) and the LOW
  CUDA-aware-MPI benefit ceiling.
- `prompt_26_rank_gpu_architecture.md` — rank/GPU/context architecture map,
  context-scaling measurements, the CUDA MPS 3.92× result and its bitwise
  guard, and the A-F candidate ranking (§8).
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
