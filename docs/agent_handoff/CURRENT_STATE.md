# Current Project State

_Last updated 2026-09-18 (Prompt 44: root-caused and FIXED the C512
six-rank MAPL/Held-Suarez resource-read startup crash that blocked Prompt
43's Stage 7. **Root cause**: `FV_StateMod.F90`'s `HELD_SUAREZ` was
declared `LOGICAL` and read directly via `MAPL_GetResource` against
`AGCM.rc`'s bare-integer `RUN_HELD_SUAREZ: 1` — deviating from this same
file's own established INTEGER-proxy convention for numeric-literal `.rc`
booleans (`INT_ADIABATIC`, `INT_FV_OFF`, etc.). `ESMF_ConfigGetAttribute`'s
LOGICAL-typed parse of that bare-integer token is unreliable, producing the
rank-varying (but not actually racing — `state%cf` is per-rank-private)
`MAPL_Resource_SetValue.h status=39` crash. **Fix** (tree #3,
`GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/
FV_StateMod.F90` — NOT the CUDA dev repo, Prompt 43's diff untouched):
added an `INT_HELD_SUAREZ` INTEGER proxy read via `MAPL_GetResource`,
manually converted to the existing `LOGICAL HELD_SUAREZ`. Confirmed via a
zero-rebuild `.rc`-format discriminator (`1`->`.TRUE.`, 0/3 crashes vs.
3/3 before) before committing to the source fix. **C512 host-staged
acceptance: 10/10 PASS** (job 58452391, 0 `status=39`, all 10 reached
`dyn_core` with uniform progress). **C32 HS-on/HS-off regression: PASS**
(job 58452598, clean `rc=0` both modes). **CUDA-aware resume gate**: MAPL
blocker CONFIRMED CLEARED under `cuda_aware` too (job 58452629 — passed
`FV_Setup`/MAPL init/`fv3_gpu_bridge_init` cleanly into `fv_dynamics`), but
surfaced a **separate, pre-existing, Prompt-43-scope** transport-layer
failure (`fv3_gpu_dyn_core_step failed`, rank 4,
`fv3_gpu_dyncore_bridge.F90:693`) — this is Prompt 43's previously-flagged,
unreconciled Stage 6 item, not a MAPL/resource issue, and out of Prompt
44's scope to fix. Host-staged transport unaffected, available now. See
§7/§9/§11 and `prompt_44_mapl_resource_race.md`. This file holds current
truth only._

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

## 7. C512 Root-Cause Track (Prompts 38-40)

- **Prompt 38 (diagnostic-only, 0 new jobs)**: root cause is message-size-
  regime-driven disproportionate halo-transport cost growth — C256
  ordinary msg ≈67.8KB, C512 ≈135.4KB (~2x), but `COMM_DSW`'s CUDA-Fortran
  delta grows 19.5x. CUDA-aware-MPI priority LOW -> MODERATE (contingent).
  See `prompt_38_c512_comm_root_cause.md`.
- **Prompt 39 (1 GPU job, job 58445168)**: in-process `PeerHaloTransport`
  (shares all topology/pack/unpack code with `MpiHaloTransport`) run at
  C256/C512 via `fv3_gpu_perf_bench`. PeerHalo C256->C512 growth = **3.69x**
  — healthy, below Fortran's own 5.72x growth — while the six-rank
  `MpiHaloTransport` path grows **8.32x** and collapses. **Isolates the
  regression to the MPI transport/protocol layer specifically**, not the
  shared device-side pack/unpack/staging mechanism. Priority ->
  HIGH CANDIDATE. See `prompt_39_c512_peerhalo_discriminator.md`.
- **Prompt 40 (1 GPU job, job 58445328, runtime-config-only, 0 source
  changes)**: OpenMPI 4.1.6; active same-node transport confirmed =
  **`vader`** (no `smcuda`/`openib` in this build). Factory-default
  eager (4096B)/rendezvous (32768B) thresholds already sit far below both
  C256 (~67.8KB) and C512 (~135.4KB) messages — both already on the
  rendezvous/pipelined path under defaults, no unique-to-C512 crossover.
  Raised `btl_vader_eager_limit` to 145000B and 524288B (2 reps each): both
  mildly **SLOWER**, not faster (+1.4-1.6% on `DYN_CORE`/`COMM_DSW`).
  **Decision Case B: EAGER/RENDEZVOUS THRESHOLD NOT SUFFICIENT.**
  **CUDA-aware-MPI priority confirmed HIGH.** See
  `prompt_40_c512_mpi_protocol_ab.md`.

- **Prompt 41 (0 GPU jobs, 0 source changes)**: verified via direct
  `ompi_info` query (login node, static build-info, no GPU needed) that
  the validated `mpi/openmpi/4.1.6/gcc-12.3.0` module's OpenMPI build has
  **`opal_built_with_cuda_support=false`** and no `--with-cuda` in its
  configure line. Confirmed (but did not adopt) that a CUDA-aware build
  already exists on Discover: `nvidia/nvhpc-hpcx-cuda12/24.7` (HPC-X
  OpenMPI 4.1.7a1, `cuda_support=true`). **Superseded by Prompt 42 below**
  — this finding is correct for OpenMPI 4.1.6 but that library is not the
  one used by the actual C512 job.
- **Prompt 42 correction + redirect (1 GPU job, job 58450084, 0 production
  source changes)**: discovered (via `CMakeCache.txt`/`ldd`/`g5_modules`/
  the actual p37 job log) that the real C256/C512 six-rank job runs under
  **Intel MPI 2021.13**, not OpenMPI — so Prompt 40's MCA eager-limit sweep
  was inert (`OMPI_MCA_*` vars are not read by Intel MPI; its ±1.4-1.6%
  deltas were noise, not a real threshold test) and Prompt 41's "CUDA-aware
  MPI blocked" finding, while true for OpenMPI 4.1.6, was answering the
  wrong question — Intel MPI was never evaluated. User approved redirecting
  Prompt 42 to Intel MPI in place of the originally-planned HPC-X
  evaluation. New standalone 2-rank device-buffer `MPI_Send`/`MPI_Recv`
  test (`scratch_p42/intelmpi_cuda_aware_test.cu`) run on a real A100
  (job 58450084, `warpa003`, rc=0): **PASS** at 4KB, 135KB (exact C512
  ordinary-group halo message size), and 256KB when `I_MPI_OFFLOAD=1` is
  set (`MPIX_Query_cuda_support()==1`, `libfabric provider: mlx`); **SIGSEGV**
  without it. Because this is already the same Intel MPI 2021.13 linked
  into `build_cuda_r8_intelmpi`/the production `StandAlone_FV3_Dycore.x`
  (§2), no toolchain migration, no new build, no ABI risk.
  **Classification: CUDA-AWARE MPI ENVIRONMENT PASS (Intel MPI 2021.13).**
  Prompt 39's PeerHalo-vs-MPI discriminator (root cause = MPI transport/
  protocol layer, not generic GPU halo work) is unaffected by this
  correction. See `prompt_42_hpcx_cuda_aware_environment.md`.

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

Numerical correctness (§5) unaffected by any of Prompts 38-44 — no
production source changed, C512 remains BITWISE PASS as of that
validation. **Preferred C512 transport remains HOST-STAGED** — now
unblocked and correctness-gate-ready (10/10 startup PASS, Prompt 44). The
`cuda_aware` variant (Prompt 43) is environment-viable (Prompt 42) and no
longer blocked by MAPL, but has its own unresolved transport-layer
correctness issue (Stage 6/above) that must be fixed before its C512 gate
can run.

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
- Multi-GPU (`dev = mod(comm_rank, device_count)`) implemented but
  validated only at 2 GPUs/3 contexts-per-GPU; 4-GPU and MPS+multi-GPU
  composition unmeasured.
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
- **`cuda_aware` MpiHaloTransport correctness gate currently BLOCKED
  (Prompt 43 Stage 6, confirmed at C512 scale by Prompt 44)**:
  `fv3_gpu_dyn_core_step failed` on 1 of 6 ranks inside the CUDA-aware
  transport/dyn_core dispatch path (`fv3_gpu_dyncore_bridge.F90:693`), no
  further detail string logged. Host-staged transport unaffected. See
  `prompt_44_mapl_resource_race.md`.

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
- Uncommitted source on `experiment/held-suarez-e2e` (Prompt 43, awaiting
  explicit commit request — 2 substantial logical changes, at budget
  ceiling): 8 files — 2 Fortran interface files (transport-selection/
  capability-validation wiring: `cuda/interface/fv3_gpu_dyncore_bridge.F90`,
  `fv3_gpu_abi_mod.F90`) + 6 C++/CUDA files (CUDA-aware transport backend:
  `cuda/dyn_core/{runtime/fv3_gpu_halo_mpi.cpp,runtime/fv3_gpu_runtime.cpp,
  include/fv3_gpu_halo_mpi.hpp,include/fv3_gpu_communication.h,
  benchmarks/run_small_perf_bench.cpp,
  tests/integration/test_halo_mpi_transport.cu}`). Verified via a
  `git stash push -u` + full rebuild + `git stash pop` + rebuild cycle
  (job 58451786) that this diff is NOT the cause of the Stage 7 blocker
  (§7/§9) — net 0 tracked-tree change from that diagnostic cycle.
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
  `p43_c512_gate_cuda_aware/`, `p43_bincheck_headnodiff/`).
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

**Active next task (Prompt 45 candidate): resume Prompt 43's remaining
scope now that the MAPL startup blocker is fixed (Prompt 44).** Two
independent paths, either can proceed now:

1. **Host-staged C512 correctness gate (Stages 7-9, low risk)** — the MAPL
   blocker is fixed and 10/10-startup-verified; run the actual short C512
   correctness gate (savepoint/NetCDF comparison, not just startup) against
   `host_staged` transport, following §5's established methodology.
2. **CUDA-aware transport fix (Prompt 43 Stage 6, now confirmed at C512
   scale)**: `fv3_gpu_dyn_core_step failed` on rank 4,
   `fv3_gpu_dyncore_bridge.F90:693`, no detail string. Needs source-level
   investigation inside the CUDA-aware `MpiHaloTransport` dispatch path
   (`cuda/dyn_core/*`, `cuda/interface/fv3_gpu_dyncore_bridge.F90`) —
   Prompt 43's own uncommitted diff, still untouched/unmodified. Suggested
   approach: add targeted logging around line 693 (what exactly fails —
   device malloc, MPI call, kernel launch) rather than assuming Stage 6's
   earlier 2-rank SIGSEGV and this C512 failure are the same bug without
   evidence.

See `prompt_44_mapl_resource_race.md` (MAPL fix + resume-gate evidence),
`prompt_43_intelmpi_cuda_aware_ab.md` (transport implementation, Stage 6
flag), and `prompt_42_hpcx_cuda_aware_environment.md` (environment-gate
evidence; filename retained from original HPC-X scope).

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
