# Current Project State

_Last updated 2026-09-18 (Prompt 46: tested Prompt 45's device-memory-
capacity finding by lowering C512 rank density from 6-ranks/1-A100 to
3-ranks/GPU across 2 A100s (`CUDA_VISIBLE_DEVICES=0,1`, existing
`dev = mod(comm_rank, device_count)` round-robin, no source change — same
6 ranks/6 tiles/1 tile-per-rank FV3 decomposition throughout).
**Result: `cuda_aware` transport is now VIABLE (no OOM)** — job 58456709
completed the full 6h C512 gate cleanly, peak GPU memory only ~5053
MiB/GPU (12.3% of an A100), vs. Prompt 45's 96.4%-utilized crisis at
1-GPU density. Numerically BITWISE identical to `host_staged` at the same
density. **But performance is NOT significant**: paired 3-pair A/B (job
58458525, same allocation, alternating transport) shows `DYN_CORE`
median-of-3 ms/call 4701.72 (`host_staged`) vs 4700.39 (`cuda_aware`) —
0.03% difference; `COMM_DSW`/`COMM_TOTAL` favor `cuda_aware` by only
1.5-2.3%. **Decision: Case C — VIABLE BUT NOT PERFORMANCE-SIGNIFICANT;
keep `host_staged` as the C512 default** (simpler, no Intel-MPI GPU-direct
dependency, equally fast at this density). **Notable side finding (informal,
not a controlled same-node A/B)**: 2-GPU `host_staged` itself measures
~4702 ms/call `DYN_CORE` vs. Prompt 37's 1-GPU figure of 9089.05 ms/call —
suggesting the C512 CUDA-vs-Fortran regression (§6/§7) may be substantially
a per-GPU rank-density/contention effect, not solely MPI-transport-protocol
cost growth. Flagged as the top candidate for a dedicated future prompt,
not investigated further here (topology-change territory, out of Prompt
46's scope). See §7/§9/§11 and
`prompt_46_c512_two_gpu_cuda_aware_ab.md`. This file holds current truth
only._

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

## 2. Repositories / Build Architecture

- Authoritative Fortran: `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag
  `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`. Not
  populated as `@fvdycore` in this repo (expected; do not auto-populate).
- CUDA implementation repo (this repo): `FVdycoreCubed_GridComp`, branch
  `cuda/gpu-bringup` (mainline) / `experiment/held-suarez-e2e` (HS track,
  current branch).
  - Discover path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Authoritative standalone Fortran fixture tree (separate tree, used for
  savepoint generation, dual-backend comparison, and all HS performance
  jobs): `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Executable: `build/bin/StandAlone_FV3_Dycore.x` (Intel `ifort` build) is
    authoritative; `install/bin/` and per-experiment job-directory copies
    are NOT auto-synced — refresh with `cp -f` and verify via
    `strings <bin> | grep <new-env-var>` after any rebuild.
  - Its nested `@fvdycore/model/{dyn_core.F90,fv_dynamics.F90}` carry
    uncommitted savepoint instrumentation only (no numerical change).
  - **MPI implementation used by every C256/C512 job to date (Prompts
    36-40) is Intel MPI 2021.13** (`comp/intel/2024.2.0` +
    `mpi/impi/2021.13`, confirmed from job logs' `esma_mpirun: mpi_type =
    intel`) — NOT the "validated" GCC+OpenMPI 4.1.6 dev stack in
    `discover_platform.md` (that stack is used for standalone CUDA-only
    dev/PeerHalo work, e.g. `build_cuda_r8_mpi`). The CUDA build matched
    to this MPI is `build_cuda_r8_intelmpi`/`build_cuda_r4_intelmpi`
    (`MPI_CXX_COMPILER=.../impi/2021.13/bin/mpigxx`, `CMAKE_CXX_COMPILER=
    g++` — GCC host compiler, Intel MPI runtime). Do not assume an
    OpenMPI-only finding (MCA/BTL tuning, `ompi_info` build flags) applies
    to the production C512 path without checking which stack that specific
    job actually loaded.
- CUDA source: `cuda/dyn_core/` (implementation), `cuda/interface/`
  (Fortran ISO_C_BINDING bridge). Do not modify `geos-gtfv3` (historical
  GEOS PACE/GT4Py/DaCe implementation; reference only).
- **No-reset/clean/discard policy**: do not reset, clean, revert, push, or
  merge existing work unless explicitly instructed. Push/merge only on
  explicit request.

## 3. Current Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
           -> HaloTransport: { PeerHaloTransport | MpiHaloTransport }
```

- Work packages: WP-A (C-grid predictor), WP-B (D-grid dynamics/transport),
  WP-C (vertical/nonhydrostatic), WP-D (pressure/grid/post-loop operators),
  WP-E (GPU runtime/state/API/communication). Ownership per
  `project_architecture.md`.
- `HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp`): two
  interchangeable backends sharing all topology/rotation/sign/pack/unpack
  logic — `PeerHaloTransport` (in-process, six host threads, default) and
  `MpiHaloTransport` (host-staged real MPI, six processes, one tile/rank —
  opt-in via CMake `FV3_DYN_CORE_ENABLE_MPI_HALO`/runtime `--mpi-halo`,
  default off). Stage 1 of the Communication Evolution model; Stages 2/3
  (CUDA-aware MPI, comm/compute overlap) unimplemented.
- Public C ABI (`fv3_gpu_api.h`, `fv3_gpu_communication.h`): lifecycle,
  `NumericalConfig` installation, production dispatch, halo-transport
  selection, CUDA Graph mode (`fv3_gpu_set_cuda_graph_mode`). No C++/MPI
  type crosses the ABI — MPI communicator crosses as Fortran INTEGER via
  `MPI_Comm_f2c`.
- Fortran bridge: `cuda/interface/{fv3_gpu_dyncore_bridge.F90,
  fv3_gpu_abi_mod.F90, fv3_gpu_field_registry.F90}`. Real production call
  site: `fv_dynamics.F90`'s `do n_map=1,k_split` loop; CUDA branch skips
  FMS `i_pack(1)`/`i_pack(8)` (duplicates of CUDA's own exchanges), keeps
  `i_pack(10)` (tracer halo, required).
- CPU/CUDA backend selection via `RUN_CUDA_FV3` in `AGCM.rc`; default
  Fortran path unchanged when CUDA off. Opt-in perf knobs (env-var gated
  via shared `env_flag_enabled(name)` helper, default OFF):
  `RUN_CUDA_FV3_FUSE_HALO_EDGES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL`,
  `RUN_CUDA_FV3_GRAPH`. Idealized Held-Suarez forcing
  (`Held_Suarez_ForcingMod.F90`, call site `FV_StateMod.F90::FV_Run` before
  `fv_dynamics`) gated by `RUN_HELD_SUAREZ`, default OFF, backend-neutral.

## 4. CUDA/FV3 Correctness Status (Production dyn_core, non-HS)

- **Gate C2: CLOSED.** Full 54-stage production first-run dyn_core
  inventory (WP-E 10, WP-A 7, WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED,
  COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED against authoritative
  Fortran savepoints.
- R8: PASS at c8 (72-call depth), c16/c32 (6-call depth), all round-off/
  small-scale discrepancies only (worst `w` max_rel ~5e-11). R4: PASS at
  same grids with R4-scaled thresholds (worst `w` max_rel ~2.4e-1,
  explained small-denominator artifact, absolute agreement ≤5.9e-5).
  n_split=3 acoustic loop, persistent device residency, CUDA Graph mode,
  launch-fusion optimizations: all IMPLEMENTED, GPU EXECUTED, bitwise-
  identical to default where applicable. Full field tables:
  `prompt_6b_cuda_fortran_validation.md`, `prompt_19`-`prompt_23`,
  `prompt_29_standalone_cuda_graph.md`.
- Six-tile GPU halo exchange (in-process and cross-process MPI):
  IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED bitwise vs.
  in-process path. MPI transport determinism issue (Prompt 18, a test-
  harness stream-ordering defect) and FMS/CUDA halo-ownership ambiguity
  (Prompt 14/15) both RESOLVED.
- **Recommended CUDA mode**: MPS ON + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`
  + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, `RUN_CUDA_FV3_GRAPH` OFF (LOW
  benefit, not default). All fusion/Graph steps bitwise/numerically
  neutral vs. default.
- Unimplemented by design (Prompt 30, triple-gated rejection, not a
  defect): `MOIST_CAPPA`/`USE_COND`, `a_imp <= 0.999` (only SIM1/`a_imp=
  1.0` implemented). Closing `MOIST_CAPPA` needs Riemann-solver `peg`/
  `gm2`/`cp2` terms (`nh_utils.F90`) in CUDA — dedicated future prompt.
- `ncu` profiling platform-blocked (`ERR_NVGPUCTRPERM`); `nsys` unaffected.

## 5. Held-Suarez Status (branch `experiment/held-suarez-e2e`)

Idealized dry-core Held-Suarez forcing, orthogonal to the `cuda/gpu-bringup`
mainline. Fortran vs CUDA, same standalone fixture, 6 MPI ranks, R8 unless
noted, savepoint and/or NetCDF-level comparison. `COLDSTART: 1` (fresh
start) used for every run below.

| Grid | Duration | Result |
|---|---|---|
| C8-C64 | short gates | PASS (bitwise/round-off) |
| C128 | 10-day, 30-day, **100-day** | PASS, byte-identical logs, bitwise T/U/PS at all checkpoints through day 100; not yet fully equilibrated (~200+ days lit. estimate) |
| **C256** | **24h gate** | **PASS** — byte-identical logs, 0 NaN/Range_Warn/Error, T/U/V/PS at 4 checkpoints **BITWISE**; CUDA faster than Fortran (§6) |
| **C512** | **6h gate** | **PASS** — byte-identical logs, 0 NaN/Range_Warn/Error, T/U/V/PS at 3 checkpoints **BITWISE**; CUDA performance REGRESSES vs Fortran (§6/§7) |

Benign known artifacts (not new divergences): 2-cell halo-padding
uninitialized-memory difference in `delpc`/`pkc` at savepoint level only
(subnormal magnitude, never at NetCDF level); C128 10-day `range_warn`
(6966 hits, both backends, confirmed benign FMS plausibility-bound
artifact); near-surface NaN artifact in 3 lowest of 25 levels (standard
MAPL vertical-interpolation masking, identical both backends, pre-existing).

**Not attempted**: duration beyond 100 days at C128; C256/C512 duration
beyond their short gates; resolution beyond C512; moist-physics coupling;
warm-restart-continuation (confirmed broken, below); formal climatology
statistics; a fix for the C512 comm regression (root-caused, not fixed).

**Warm restart (`COLDSTART: 0`)**: **confirmed BROKEN** for this HS/
nonhydrostatic standalone config (Prompt 35, Fortran backend — genuine
source-level gap in the nonhydrostatic warm-restart path, not a CUDA/config
issue). CUDA restart correctly skipped once Fortran's own failed. Remains
broken unless a future prompt fixes it; use fresh `COLDSTART: 1` runs.

## 6. Performance Status

Current best validated standalone-path CUDA config: **MPS ON +
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`**,
Graph OFF. Figures below: real standalone six-rank MPI path
(`StandAlone_FV3_Dycore.x`, R8, 6 MPI ranks/1 A100), `fv_timing_mod`
`DYN_CORE` timer.

HS/resolution-scaling table (30 dyn_core calls/rep, 3 reps, warmup-included):

| Grid | Fortran ms/call | CUDA ms/call | CUDA/Fortran |
|---|---|---|---|
| C32 | 20.72 | 19.88 | 0.96 — CUDA faster |
| C64 | 59.4 | 56.9 | 0.96 — CUDA faster |
| C128 | 237.4 | 210.9 | 0.89 — CUDA faster |
| **C256** | 1250.75 | 1092.44 | **0.874 — CUDA faster (best)** |
| **C512** | 7155.63 | 9089.05 | **1.270 — CUDA SLOWER (regression)** |

(Separate warmup-excluded C32 production-config baseline, §8 methodology,
100 calls/rep: Fortran 19.930 ms/call vs CUDA 24.300 ms/call = 1.22x
slower — not directly comparable to the table above; both internally
consistent within their own methodology.)

**Trend**: gap widens in CUDA's favor C32->C256, reverses sharply at C512
(first reversal in the ladder). Root cause and discriminators: §7.

**Node-to-node A100 timing variability is material**: Prompt 40's C512
baseline ran ~30% faster in absolute terms than Prompt 37's historical
figure, traced to compute-node hardware variance (`warpa005` vs.
`warpa009`), not a config defect. **Any future A/B performance comparison
must run all arms on the same allocated node/job where possible** — do
not compare absolute timings across separately-scheduled jobs.

Full tables/reconciliation: `prompt_34_c32_perf_reconciliation.md`,
`prompt_36_c256_correctness_performance.md`,
`prompt_37_c512_correctness_performance.md`.

**Informal 2-GPU data point (Prompt 46, not a controlled A/B against the
row above — different job/possibly different node)**: at 3-ranks/GPU
across 2 A100s, C512 `DYN_CORE` drops to ~4701 ms/call for both
`host_staged` and `cuda_aware` — roughly half the 1-GPU-density figure
above and, if it held under a controlled comparison, would flip the C512
regression into a large CUDA win. Not yet formally re-measured; see §7
Prompt 46 and `prompt_46_c512_two_gpu_cuda_aware_ab.md` §7.

## 7. C512 Root-Cause Track (Prompts 38-40)

- **Prompts 38-42 (diagnostic chain, condensed — full detail in their own
  files)**: 38 root-caused message-size-driven halo-transport cost growth
  (C256->C512 msg ~2x, but CUDA-Fortran `COMM_DSW` delta grows 19.5x). 39
  (job 58445168) isolated the regression to the **MPI transport/protocol
  layer specifically** — in-process `PeerHaloTransport` (shares all
  topology/pack/unpack code with `MpiHaloTransport`) scales healthily
  (3.69x C256->C512, below Fortran's own 5.72x) while six-rank
  `MpiHaloTransport` grows 8.32x and collapses. 40 (job 58445328) ruled out
  OpenMPI eager/rendezvous threshold tuning (mildly slower, not faster).
  41 found OpenMPI 4.1.6 has no CUDA-aware build — **superseded by 42**,
  which discovered the real C256/C512 job actually runs under **Intel MPI
  2021.13**, not OpenMPI (so 40's OpenMPI MCA sweep was inert and 41
  answered the wrong question); a standalone 2-rank device-buffer test
  (job 58450084) confirmed Intel MPI CUDA-aware PASS at the exact C512
  halo message size with `I_MPI_OFFLOAD=1` (SIGSEGV without it) — same
  Intel MPI already linked into production, no toolchain/ABI risk.
  **CUDA-aware-MPI priority: HIGH, environment PASS.** See
  `prompt_38_c512_comm_root_cause.md` through
  `prompt_42_hpcx_cuda_aware_environment.md`.

- **Prompt 43 (implementation, 0 net source changes beyond the existing
  Stage 0-5 diff)**: implemented the
  `RUN_CUDA_FV3_MPI_TRANSPORT=host_staged|cuda_aware` CUDA-aware
  `MpiHaloTransport` variant (Intel MPI 2021.13, `I_MPI_OFFLOAD=1`-gated,
  per Prompt 42's environment PASS). Attempting the Stage 7 C512 short
  correctness gate surfaced an unrelated, pre-existing MAPL resource-read
  startup crash (Prompt 43 diagnosed + isolated it via `git stash`+
  rebuild; **fixed under Prompt 44**, below). Stage 6 (a 2-rank
  device-buffer transport unit test's intermittent SIGSEGV) flagged but not
  reconciled — **now confirmed as the same class of issue blocking the
  cuda_aware resume attempt below**. Stages 7/8/9 (correctness gate, paired
  A/B, decision) still NOT completed. See `prompt_43_intelmpi_cuda_aware_ab.md`.
- **Prompt 44 (fix, 1 source file changed in the fixture tree — not the
  CUDA dev repo)**: root-caused and fixed the MAPL/Held-Suarez
  resource-read crash blocking Prompt 43's Stage 7 (see header for detail).
  **C512 host-staged: 10/10 PASS.** Resuming Prompt 43 with
  `cuda_aware` transport now reaches `fv_dynamics`/dyn_core (MAPL blocker
  cleared) but hits a **separate, transport-layer runtime failure**
  (`fv3_gpu_dyn_core_step failed`, rank 4 only,
  `fv3_gpu_dyncore_bridge.F90:693`, job 58452629) — this is Prompt 43's own
  Stage 6 item, now confirmed reproducible at full C512 scale, not a MAPL
  issue. See `prompt_44_mapl_resource_race.md`.
- **Prompt 45 (fix + root cause, 1 CUDA dev-repo source file changed —
  `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`)**: made the transport
  failure diagnosable (embedded `cudaGetErrorString()` into the FATAL
  message) and used it to root-cause Prompt 44's remaining transport-layer
  failure as **device memory capacity, not a code defect** — see header
  for full detail. Transport unit test 10/10 PASS both modes; C512 smoke:
  `host_staged` PASS (2 simulated hours), `cuda_aware` FAILS at
  initialization (`cudaErrorMemoryAllocation` at the Riemann-solver kernel
  launch, six C512 ranks already at 96.4% of the A100's 40GB under the
  working transport). **Decision: do not resume Prompt 43's cuda_aware
  performance A/B at the current 6-ranks-per-A100 density.** See
  `prompt_45_cuda_aware_correctness.md`.
- **Prompt 46 (3 GPU jobs — 58454451, 58456709, 58458525 — 0 source
  changes, GPU-placement-only via `CUDA_VISIBLE_DEVICES=0,1`)**: tested
  Prompt 45's deferred follow-on #1 (lower rank density). At 3-ranks/GPU
  across 2 A100s: `host_staged` 2-GPU correctness confirmed (BITWISE vs.
  1-GPU baseline); `cuda_aware` **no longer OOMs** (peak ~5053 MiB/GPU,
  12.3% of capacity, vs. Prompt 45's 96.4%) and is BITWISE-identical to
  `host_staged`; paired 3-pair performance A/B on the same allocation
  shows `DYN_CORE` median-of-3 ms/call 4701.72 (`host_staged`) vs 4700.39
  (`cuda_aware`) — 0.03% difference, `COMM_DSW`/`COMM_TOTAL` favor
  `cuda_aware` by only 1.5-2.3%. **Decision Case C: VIABLE BUT NOT
  PERFORMANCE-SIGNIFICANT — keep `host_staged` as the C512 default.**
  Notable side finding (informal, not a controlled A/B): 2-GPU
  `host_staged` itself runs at ~4701 ms/call vs. Prompt 37's 1-GPU figure
  of 9089.05 ms/call — a ~2x gap suggesting the C512 regression (§6) may
  be substantially a per-GPU rank-density effect, not solely MPI-transport
  protocol cost; flagged, not investigated further (out of scope). See
  `prompt_46_c512_two_gpu_cuda_aware_ab.md`.

Numerical correctness (§5) unaffected by any of Prompts 38-46 — no
production source changed, C512 remains BITWISE PASS as of that
validation. **Preferred C512 transport remains HOST-STAGED** at any tested
density — correctness-gate-ready (10/10 startup PASS, Prompt 44) and
simpler than `cuda_aware`. The `cuda_aware` variant is now confirmed
environment-viable (Prompt 42), correctness-viable at 2-GPU density
(Prompt 46), and numerically bitwise-equivalent to `host_staged` — but not
performance-significant at 2-GPU density (Prompt 46), so it does not
displace `host_staged` as the default. At the original 6-ranks/1-GPU
density it remains blocked by device memory capacity (Prompt 45).

## 8. Standalone Integration Status

- CUDA dyn_core runs inside the real `StandAlone_FV3_Dycore.x` under the
  true production decomposition (`FV_NX=1`/`FV_NY=6`, six MPI ranks, one
  tile/rank), same executable as Fortran (selected via `AGCM.rc`'s
  `RUN_CUDA_FV3`).
- Public C ABI complete for per-run setup.
- ABI/compiler-family compatibility with a full **Intel-compiler
  production GEOS build** verified sound by construction (Prompt 30:
  fixed-width `iso_c_binding` types, runtime `abi_version`/`struct_bytes`
  guards, explicit `MPI_Comm_f2c`). Revisit if the toolchain changes.

## 9. Known Limitations / Unvalidated Envelope

- R4/R8 production (non-HS) validation covers c8 (deepest: 72/12-call
  ladders) and c16/c32 (6-call depth only). HS validation separately
  covers c8-c128 up to 100-day duration, plus C256 (24h gate) and C512
  (6h gate) — short gates only at C256/C512, not multi-day.
- **C512 CUDA performance regression**: unresolved, root-caused to the
  MPI transport/protocol layer (§7), not fixed. Do not assume CUDA
  competitive above C256 without remeasuring.
- `MOIST_CAPPA`/`USE_COND` and `a_imp <= 0.999`: confirmed UNIMPLEMENTED
  in CUDA (§4), by design — future dedicated prompts.
- `ncu` profiling platform-blocked; `nsys` unaffected.
- `fv3_dispatch_smoke` Phase C fails if its own `cfg.nf_ke` test pin is
  raised from 1 to authoritative 3 — not localized, not evidence against
  production corner service (validated at `nf_ke=3` in the real dyn_core
  path, §4).
- CUDA MPS (`fv3_mps_lifecycle.sh`) requires `compute_mode=Default` (true
  on Discover `gpu_a100`); not yet wired into every standing job script as
  the default.
- Multi-GPU (`dev = mod(comm_rank, device_count)`) validated at C512 scale
  at 2 GPUs/3-ranks-per-GPU (Prompt 46, both transports, correctness +
  performance); 4-GPU and MPS+multi-GPU composition still unmeasured.
- `install/bin/StandAlone_FV3_Dycore.x` and per-experiment binary copies
  not auto-synced with `build/bin/` after rebuild — always refresh and
  verify via `strings` before trusting a new flag.
- **Warm restart confirmed BROKEN** (§5) — use fresh `COLDSTART: 1`.
- C128/C256/C512 each needed their own timestep-stability search (C128's
  first `bdt` candidate was CFL-unstable); do not assume linear scaling of
  a resolution-scaled `bdt` heuristic without a short stability check —
  do not assume C256/C512's first-candidate stability holds at C1024+.
- **C512 MAPL resource-read startup crash: FIXED (Prompt 44)** — see
  header/§7. The harness's own `fv3.j` unconditionally `exit 0`s regardless
  of the model's real return code, so a short (~30-40s) "COMPLETED" run at
  C512 was, and remains, a useful red flag worth checking logs for even
  post-fix (a healthy run takes the full requested walltime to make
  partial progress, or exits `rc=0` cleanly at a natural end-of-run).
- **`cuda_aware` MpiHaloTransport at C512**: blocked by device memory
  capacity at the original 6-ranks/1-A100 density (`cudaErrorMemoryAllocation`,
  Prompt 45); **confirmed viable and BITWISE-correct at 3-ranks/GPU across
  2 A100s (Prompt 46)**, but not performance-significant there (<2.5%
  delta vs. `host_staged` on any metric) — `host_staged` remains the
  recommended default at any density. See
  `prompt_45_cuda_aware_correctness.md`, `prompt_46_c512_two_gpu_cuda_aware_ab.md`.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`. Mainline branch `cuda/gpu-bringup`
  HEAD `2e6327f` ("Document MPS rank-GPU performance architecture"), ahead
  of `origin/cuda/gpu-bringup`, not pushed. HS track branch
  `experiment/held-suarez-e2e` HEAD `9bc277f` ("Rule out MPI eager
  threshold as C512 regression cause"), also not pushed. Push/merge only
  on explicit request.
- Uncommitted source on `cuda/gpu-bringup` (Prompt 28+29, awaiting
  explicit commit request): `cuda/interface/fv3_gpu_dyncore_bridge.F90`
  (`env_flag_enabled` helper + fusion/Graph env wiring),
  `cuda/interface/fv3_gpu_abi_mod.F90` (`fv3_gpu_set_cuda_graph_mode`).
- Uncommitted source on `experiment/held-suarez-e2e` (Prompt 43+45,
  awaiting explicit commit request — 3 substantial logical changes, over
  the informal per-prompt budget when counted cumulatively, but each
  prompt's own 2-change ceiling was individually respected): 9 files —
  2 Fortran interface files (transport-selection/capability-validation
  wiring: `cuda/interface/fv3_gpu_dyncore_bridge.F90`,
  `fv3_gpu_abi_mod.F90`) + 6 C++/CUDA files (CUDA-aware transport backend:
  `cuda/dyn_core/{runtime/fv3_gpu_halo_mpi.cpp,runtime/fv3_gpu_runtime.cpp,
  include/fv3_gpu_halo_mpi.hpp,include/fv3_gpu_communication.h,
  benchmarks/run_small_perf_bench.cpp,
  tests/integration/test_halo_mpi_transport.cu}`, all Prompt 43) + 1 file
  from Prompt 45 (`cuda/dyn_core/vertical/fv3_vertical_kernels.cu` —
  `fail()`-wrapped early returns + `cudaGetErrorString()` embedding in
  `launch_update_dz_d`/`launch_vertical_solver`, diagnostic-visibility
  only, no numerical semantics changed). Prompt 43's diff verified via a
  `git stash push -u` + full rebuild + `git stash pop` + rebuild cycle
  (job 58451786) that it is NOT the cause of the Stage 7 blocker (§7/§9) —
  net 0 tracked-tree change from that diagnostic cycle. Prompt 45's diff
  confirmed via `git status --short`/`git diff --stat` to be the only
  change made in that prompt; Prompt 43's 8 files confirmed untouched.
- No production source files changed by Prompts 37-42 (38/41 diagnostic-
  only, no GPU job; 39/40 each ran one GPU job against existing unmodified
  binaries/harnesses; 40 changed only a runtime `OMPI_MCA_btl_vader_eager_
  limit` env var; 41 was a login-node-only `ompi_info`/`module avail`
  build-support check; 42 ran one GPU job against a new, tiny, standalone
  test file, no `cuda/`/`include/` file touched). New untracked from
  Prompt 42: `docs/agent_handoff/prompt_42_hpcx_cuda_aware_environment.md`,
  `scratch_p42/` (`intelmpi_cuda_aware_test.cu`, its sbatch script, job
  output, and a login-node `cuda_query.c` probe). New untracked from
  Prompt 43: `docs/agent_handoff/prompt_43_intelmpi_cuda_aware_ab.md`,
  `scratch_p43/` (sbatch scripts/job logs for the discriminator sequence,
  §7/§9), plus new experiment directories under
  `<fortran_perf>/experiments/` (`p43_bincheck_hoststaged/`,
  `p43_c512_gate_cuda_aware/`, `p43_bincheck_headnodiff/`). New untracked
  from Prompt 44: `docs/agent_handoff/prompt_44_mapl_resource_race.md`.
  New untracked from Prompt 45:
  `docs/agent_handoff/prompt_45_cuda_aware_correctness.md`, `scratch_p45/`
  (module-mismatch diagnostics, rebuild jobs, Stage 9 smoke/memory-polling
  scripts and logs), and
  `<fortran_perf>/experiments/p45_c512_cuda_aware_smoke/` (host_staged_out/,
  cuda_aware_out/, cuda_aware_out2/, GPU memory-poll CSVs). New untracked
  from Prompt 46 (0 CUDA-dev-repo source files touched, GPU-placement-only
  via `CUDA_VISIBLE_DEVICES`):
  `docs/agent_handoff/prompt_46_c512_two_gpu_cuda_aware_ab.md`, and under
  `<fortran_perf>/experiments/`: `p46_c512_2gpu_host_staged{,.sbatch}/`,
  `p46_c512_2gpu_cuda_aware{,.sbatch}/`, `p46_c512_2gpu_perf/` (incl.
  `fv3.j`, `perf_logs/pair{1,2,3}_{host_staged,cuda_aware}.log`),
  `p46_c512_2gpu_perf_ab.sbatch`, job `.out` files and GPU-memory-poll CSVs
  for jobs 58454451/58456709/58458525.
- Untracked at repo root (both branches share the worktree): numerous
  scratch shell scripts/logs/probes from prior prompts, reusable tooling
  under `tools/` (savepoint validation — `compare_dcx_backends.py`,
  `tools/savepoints/savepoint_reader.py`; `tools/analysis/
  hs_netcdf_inspect.py`), and `perf_rerun_logs/`. Untracked handoff docs
  for every prompt from 22 through 40 (§12). In the separate Fortran
  fixture tree `<fortran_perf>/experiments/`: per-prompt sbatch scripts/
  logs/job directories for prompts 24-40, and `fv3_mps_lifecycle.sh`.
- Do not commit: build directories (`build_cuda_r8*/`), profiler artifacts
  (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm log, NetCDF
  output, restart data.
- Do commit (if explicitly requested): legitimate source fixes, permanent
  test/tooling additions under `tools/`, documentation.

## 11. Next Recommended Task

**Active next task (Prompt 47 candidate): controlled 1-GPU-vs-2-GPU C512
performance A/B (same-node/same-job where feasible), transport held
constant at `host_staged`.** Prompt 46's strongest new lead (§7/§9) is an
informal ~2x `DYN_CORE` gap between 1-GPU/6-rank density (9089.05 ms/call,
Prompt 37) and 2-GPU/3-ranks-per-GPU density (~4701 ms/call, Prompt 46) —
if confirmed under a rigorous same-node comparison, this would mean the
documented C512 CUDA-vs-Fortran regression (§6) is substantially a
per-GPU rank/resource-contention effect at 6-ranks-per-A100, not solely
the MPI-transport-protocol cost growth root-caused in Prompts 38-40. This
is topology-adjacent (GPU placement, not FV3 decomposition) — confirm
scope explicitly before starting, per this project's no-unscoped-topology-
change norm.

**Lower priority now that Prompt 46 closed the cuda_aware viability
question**: Intel MPI GPU-buffer-pool sizing controls (`I_MPI_*` env
vars) — cuda_aware is confirmed viable at 2-GPU density and not
performance-significant there, so this investigation is no longer
blocking anything.

See `prompt_46_c512_two_gpu_cuda_aware_ab.md` (2-GPU A/B, Case C decision,
the 1-GPU-vs-2-GPU lead), `prompt_45_cuda_aware_correctness.md`
(device-memory root cause at 1-GPU density), `prompt_44_mapl_resource_race.md`
(MAPL fix), `prompt_43_intelmpi_cuda_aware_ab.md` (transport implementation).

**Mainline (`cuda/gpu-bringup`) tracks — confirm scope with the user before
starting any:**
1. CUDA `MOIST_CAPPA`/`USE_COND` implementation (WP-C/WP-B, §4/§9).
2. CUDA `a_imp <= 0.999` implementation (§4/§9).
3. Operational wiring — fold MPS lifecycle + fusion flags into every
   standing job script as the default (lower risk than 1/2).

**Other open HS-track follow-ons (confirm scope before starting):**
- C256/C512 climatology / longer duration (only short gates run so far).
- Further duration at C128 toward ~200+ day HS equilibration estimate.
- Warm-restart fix (dedicated future prompt).
- Higher resolution than C512 (C1024+) — needs its own stability search;
  competitiveness depends on Prompt 41's outcome if pursued.

## 12. Detailed Handoff References

Chronological/technical detail for every closed or attempted prompt remains
in its own file under `docs/agent_handoff/`; read the specific file only
when the current task requires that level of detail.

- `prompt_6a_fortran_savepoints.md` / `prompt_6b_cuda_fortran_validation.md`
  — authoritative Fortran savepoints, Gate C2 closure.
- `prompt_7_post_gate_c2_performance_baseline.md`,
  `prompt_9`-`prompt_13` — launch-fusion/CUDA Graph sequence.
- `prompt_14_standalone_cuda_integration.md`,
  `prompt_15_mpi_halo_transport.md`,
  `prompt_18_mpi_transport_determinism.md` — MPI/halo transport buildout.
- `prompt_19`-`prompt_23` — R8/R4 c8/c16/c32 depth validation, CFL/`bdt`
  scaling.
- `prompt_24_standalone_performance_validation.md`,
  `prompt_25a_fv3_halo_characterization.md` — nsys bottleneck
  classification, original halo census (superseded at C256/C512, §7).
- `prompt_26_rank_gpu_architecture.md`,
  `prompt_27_mps_operational_profile.md` — MPS lifecycle/profile.
- `prompt_28`, `prompt_29` — fusion flags, CUDA Graph.
- `prompt_30_production_config_closure.md` — unimplemented-config closure,
  Intel ABI verification.
- `prompt_31_held_suarez_e2e.md`,
  `prompt_32_held_suarez_scientific_validation.md` — HS forcing, C32
  validation.
- `prompt_33_hs_multiday_highres_performance.md`,
  `prompt_34_c32_perf_reconciliation.md` — HS ladder to C128, perf
  reconciliation.
- `prompt_35_hs_climatology.md` — warm-restart broken; 30/100-day
  climatology.
- `prompt_36_c256_correctness_performance.md`,
  `prompt_37_c512_correctness_performance.md` — C256/C512 dt-search,
  correctness gates, perf points.
- `prompt_38_c512_comm_root_cause.md`,
  `prompt_39_c512_peerhalo_discriminator.md`,
  `prompt_40_c512_mpi_protocol_ab.md`,
  `prompt_41_c512_cuda_aware_mpi_ab.md` — C512 root-cause chain; Prompt
  41's OpenMPI CUDA-aware-build Hard Stop, later corrected by Prompt 42
  (wrong MPI implementation characterized; §7).
- `prompt_42_hpcx_cuda_aware_environment.md` — redirect from HPC-X to
  Intel MPI 2021.13 (the actual production MPI); Intel MPI CUDA-aware
  device-buffer transport confirmed PASS at the C512 halo message size
  (job 58450084); environment gate closed, implementation (Prompt 43)
  next (§7/§11).
- `prompt_43_intelmpi_cuda_aware_ab.md` — CUDA-aware `MpiHaloTransport`
  implementation (Stages 0-5 COMPLETE); Stage 7 C512 correctness gate
  BLOCKED by a pre-existing, non-Prompt-43 `MAPL_Resource_SetValue.h`
  crash, isolated via `git stash`+rebuild discriminator (job 58451786);
  root-cause/fix delivered by Prompt 44.
- `prompt_44_mapl_resource_race.md` — root cause (unsafe LOGICAL resource
  read of a bare-integer `.rc` token, not a race) + fix
  (`FV_StateMod.F90` INTEGER-proxy convention) for the MAPL startup crash;
  10/10 C512 host-staged acceptance; C32 regression clean; `cuda_aware`
  resume gate confirms MAPL blocker cleared but surfaces a separate,
  Prompt-43-scope transport-layer failure at C512 scale (§7/§9/§11).
- `prompt_45_cuda_aware_correctness.md` — made the transport failure
  diagnosable (`cudaGetErrorString()` embedded in the FATAL message,
  `fv3_vertical_kernels.cu`); root-caused it to device memory capacity
  (six C512 ranks already at 96.4% of one A100's 40GB under the working
  `host_staged` transport, `cuda_aware`'s extra Intel-MPI GPU-direct buffer
  registration exceeds the ~1.5GB headroom); transport unit test 10/10 PASS
  both modes; decision: do not resume Prompt 43's cuda_aware performance
  A/B at the current rank density (§7/§9/§11).
- `prompt_46_c512_two_gpu_cuda_aware_ab.md` — lowered C512 rank density to
  3-ranks/GPU across 2 A100s (GPU-placement-only, no source change);
  `cuda_aware` no longer OOMs (~5053 MiB/GPU peak) and is BITWISE-correct
  vs. `host_staged`, but paired A/B shows <2.5% performance delta on any
  metric (Case C: viable but not performance-significant; `host_staged`
  remains the default); flagged an informal ~2x 1-GPU-vs-2-GPU `DYN_CORE`
  gap as the top candidate for a future controlled A/B (§7/§9/§11).

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
