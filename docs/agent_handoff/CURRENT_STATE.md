# Current Project State

_Last updated 2026-10-01 (checkpoint: Prompt 116 — DESIGN ONLY, no source
changes. Designed the production `SubTileId{face,sub_i,sub_j}` derivation
from `(mpi_rank, tile, layout_x, layout_y)` using Prompt 115's verified
formula: **C++ runtime** location, **no new ABI field**, `pelist(1)` **not
required**, exact 1×1 reduction, 24-rank oracle, **shadow-mode-first**
builder migration. Recommended bounded next slice (SMALL, §9.1): pure
derivation helper + oracle unit test + env-gated shadow-topology check — no
ABI change, no gate relaxation, no real >6-rank run. No commit/push. Doc:
`docs/performance/p116_subtile_id_derivation_design_2026-10-01.md`.
Full history in `docs/agent_handoff/milestone_summary_2026-09-19.md`.
This file holds only what the next coding agent needs to continue._

## 1. Project Goal

Modernize the GEOS FV3 atmospheric dynamical core for heterogeneous CPU/GPU
execution via a staged path:

```
GEOS/FV3 Fortran -> ISO_C_BINDING -> stable C ABI -> C++/CUDA runtime
    -> persistent GPU dyn_core subsystem
```

Authoritative GFDL FV3 Fortran is the numerical source of truth. CUDA
remains optional; the original Fortran backend is always preserved. See
`docs/agent_instructions/project_architecture.md` for full architectural
constraints (work-package ownership, persistent-residency rules,
validation terminology) — unchanged, not repeated here.

## 2. Repositories / Build Architecture

- Authoritative Fortran: `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag
  `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`. Not
  populated as `@fvdycore` in this repo (expected; do not auto-populate).
- CUDA implementation repo (this repo): `FVdycoreCubed_GridComp`, branch
  `cuda/gpu-bringup` (mainline) / `experiment/held-suarez-e2e` (HS track,
  current branch).
  - Discover path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Authoritative standalone Fortran fixture tree (savepoint generation,
  dual-backend comparison, all HS performance jobs):
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Executable: `build/bin/StandAlone_FV3_Dycore.x` (Intel `ifort` build)
    is authoritative; `install/bin/`/job-directory copies are NOT
    auto-synced — refresh with `cp -f`, verify via `strings <bin> |
    grep <new-env-var>` after any rebuild.
  - **Every C256/C512 job to date uses Intel MPI 2021.13**
    (`comp/intel/2024.2.0`+`mpi/impi/2021.13`), NOT the OpenMPI dev
    stack in `discover_platform.md` (that's for CUDA-only dev/PeerHalo
    work). Matched CUDA build: `build_cuda_r8_intelmpi`. Check which
    stack a job loaded before generalizing an MPI-tuning finding.
- CUDA source: `cuda/dyn_core/` (implementation), `cuda/interface/`
  (Fortran ISO_C_BINDING bridge). Do not modify `geos-gtfv3` (historical
  GEOS PACE/GT4Py/DaCe reference implementation).
- **No-reset/clean/discard policy**: do not reset, clean, revert, push, or
  merge existing work unless explicitly instructed. Push/merge only on
  explicit request.

## 3. Current Validated CUDA Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
           -> HaloTransport: { PeerHaloTransport | MpiHaloTransport }
```

- Work packages: WP-A (C-grid predictor), WP-B (D-grid dynamics/
  transport), WP-C (vertical/nonhydrostatic), WP-D (pressure/grid/post-
  loop operators), WP-E (GPU runtime/state/API/communication). Ownership
  per `project_architecture.md`.
- `HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp`): two
  interchangeable backends sharing all topology/rotation/sign/pack/unpack
  logic — `PeerHaloTransport` (in-process, six host threads, default) and
  `MpiHaloTransport` (host-staged or CUDA-aware real MPI, opt-in via
  CMake `FV3_DYN_CORE_ENABLE_MPI_HALO`/runtime `--mpi-halo`, default
  off). Variant selected via `RUN_CUDA_FV3_MPI_TRANSPORT=
  host_staged|cuda_aware` (default `host_staged`).
- Public C ABI (`fv3_gpu_api.h`, `fv3_gpu_communication.h`): lifecycle,
  `NumericalConfig` installation, production dispatch, halo-transport
  selection, CUDA Graph mode. No C++/MPI type crosses the ABI — MPI
  communicator crosses as Fortran INTEGER via `MPI_Comm_f2c`.
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
  (`Held_Suarez_ForcingMod.F90`, call site `FV_StateMod.F90::FV_Run`
  before `fv_dynamics`) gated by `RUN_HELD_SUAREZ`, default OFF, backend-
  neutral.

## 4. Current Correctness Status

- **Gate C2: CLOSED.** Full 54-stage production first-run dyn_core
  inventory (WP-E 10, WP-A 7, WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED,
  COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED against authoritative
  Fortran savepoints. R8 PASS at c8/c16/c32 — round-off only (worst `w`
  max_rel ~5e-11). R4 PASS at the same grids, R4-scaled thresholds
  (worst `w` max_rel ~2.4e-1, explained small-denominator artifact,
  absolute agreement ≤5.9e-5).
- Six-tile GPU halo exchange (in-process and cross-process MPI):
  IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED bitwise vs.
  in-process path.
- **Held-Suarez (branch `experiment/held-suarez-e2e`)**: dry-core forcing,
  orthogonal to the `cuda/gpu-bringup` mainline, `COLDSTART: 1` used for
  every run below.

  | Grid | Duration | Result |
  |---|---|---|
  | C8-C64 | short gates | PASS (bitwise/round-off) |
  | C128 | 10-day, 30-day, 100-day | PASS, bitwise T/U/PS through day 100; not yet fully equilibrated (~200+ days lit. estimate) |
  | C256 | 24h gate | PASS, bitwise T/U/V/PS at 4 checkpoints; CUDA faster |
  | C512 | 6h gate | PASS, bitwise T/U/V/PS at 3 checkpoints; CUDA/Fortran performance COMPARABLE (§6) |

  Benign known artifacts (pre-existing both backends, not new
  divergences): subnormal halo-padding diff in `delpc`/`pkc`
  (savepoint-level only); C128 10-day benign FMS `range_warn`;
  near-surface NaN in 3/25 levels (standard MAPL masking).
- **Warm restart (`COLDSTART: 0`) confirmed BROKEN** for this HS/
  nonhydrostatic config — genuine Fortran-backend gap, not CUDA-
  specific; CUDA restart correctly skips once Fortran's own restart
  fails. Use fresh `COLDSTART: 1` only.
- Unimplemented by design (triple-gated rejection, not a defect):
  `MOIST_CAPPA`/`USE_COND`, `a_imp <= 0.999` (only SIM1/`a_imp=1.0`
  implemented). See §7 for the next recommended target.
- `ncu` profiling platform-blocked (`ERR_NVGPUCTRPERM`); `nsys` unaffected.

## 5. Current Performance Status

Standalone six-rank MPI path (`StandAlone_FV3_Dycore.x`, R8, 6 MPI
ranks/1 A100), `fv_timing_mod` `DYN_CORE` timer. Best validated config:
**MPS ON + `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` +
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`**, Graph OFF (30 calls/rep, 3 reps,
warmup-included):

| Grid | Fortran ms/call | CUDA ms/call | CUDA/Fortran |
|---|---|---|---|
| C32 | 20.72 | 19.88 | 0.96 — CUDA faster |
| C64 | 59.4 | 56.9 | 0.96 — CUDA faster |
| C128 | 237.4 | 210.9 | 0.89 — CUDA faster |
| C256 | 1250.75 | 1092.44 | 0.874 — CUDA faster (best) |
| **C512 (same-node, authoritative)** | **6374.02** | **6373.86** | **1.00 — COMPARABLE** |

**C512 performance question: CLOSED** (job 58468244, same-node rerun,
ratio of medians 0.99998 — backends comparable; the old 1.270x "CUDA
slower" figure was a cross-node artifact, **SUPERSEDED**, §8). GPU
rank-density ruled out as a **timing** cause (memory-density findings
superseded by P102, §8). CUDA-aware MPI transport: correct/bitwise-
equivalent at a memory-safe density but negligible perf benefit
(~0.03%) — `host_staged` remains default. Full narrative:
`milestone_summary_2026-09-19.md` §B/§D, `prompt_38`-`prompt_48_*.md`.

**Closed CUDA production ports (detail in each doc under §11 — do not
reopen without fresh authorization):**
- PT/DELZ (P49-69): CUDA default. REMAPPING −33.7%, TOTAL −7.3%.
- REMAP_UV (P71-84A): CUDA default. Bitwise-exact, ~10.23x, TOTAL +~1%.
- TRACER_2D (P73-83): **Fortran default — STOP CUDA TRACER PRODUCTION
  OPTIMIZATION.** hord==8 kernel GPU-validated but regression-confirmed
  (+37.87%), short of the required ≥5x gate. Moot in production:
  `use_cuda_tp2d` requires `hord==8`, production uses `hord_mt/vt/tm=6`.
- REMAP_Q (P84B-89): CUDA default, CUDA `fillz`. 3.23x/TOTAL −1.51%.
- REMAP_PRESSUREVARS (P90-93): CUDA default (escape hatch
  `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`). Bitwise, sanitizer
  clean; 3.50x direct, mean TOTAL +1.805%.
- REMAP_TOTALENERGYCON (P95): no port taken — **BLOCKED BY
  COMMUNICATION**, `mpp_global_sum` ~89% of the timer.
- `FV_To_State` (P97): **KEEP ON CPU** — already host-resident both
  ends.
- `State_To_FV` (P98-100): CUDA port GPU-executed/C32-correct but
  memory-blocked at 6-ranks/1-GPU (~440 MiB short). Selector
  `RUN_CUDA_FV3_STATE_TO_FV_ENABLE`/`_FORCE_FORTRAN`. Memory feasibility
  later **CONFIRMED at 2-GPU density by Prompt 103** (below).

Prompt 94 was the then-current-production reference point (all 4 remap
ports CUDA, `State_To_FV` Fortran): TOTAL 3836.74 ms/call; also found
MAPL `DYN_EPILOGUE` (688.67 ms/call). Prompt 101's C512 memory-inventory
gap was **RESOLVED by Prompt 102** (below).

**Prompts 101A-107 (all CLOSED, full detail in §11)**: cumulative CUDA
baseline (TOTAL 1.153x) -> C512 memory gap is per-rank not per-GPU ->
2-GPU C512 validated (bitwise-exact, 19.68 GiB/GPU, TOTAL 1.120x,
**recommended C512 mode: 2-GPU/3-ranks-per-GPU**) -> `State_To_FV`
CUDA-ported and made **CONDITIONAL DEFAULT** at `device_count>=2` ->
node-local `MPI_Comm_split_type` GPU binding validated single-/
multi-node; `alla100` QOS caps job-wide (not per-node) GPU/mem;
multi-node MPS needs an explicit `srun --overlap` broadcast (§7).

**Prompts 108-112 (sub-tile C1024+ halo chain, all CLOSED)**: 108 mapped
the 24-stage C1024 intra-tile architecture (design only); 109
implemented/validated the topology/donor-identification layer for a
synthetic 24-rank layout (bridge gate untouched); 110 designed the
Phase-5 wiring slice; 111 implemented it and found `launch_halo_transfer`
indexed `peer_send` directly by `.tile` (fixed-6), blocking any
layout=2x2 cross-face donor; 112 fixed this by separating physical face
(`.tile`) from transport peer identity (`.rank`), GPU-validated at real
24-rank MPI scale (**0/22224 failures**, 0 sanitizer errors/rank). No
bridge/gate relaxed, no corner transport, no numerical change, no
commit/push throughout. Docs listed in §11.

**Prompts 113-115 (CLOSED)**: 113 designed, 114 implemented, the FMS
layout-metadata -> CUDA bridge integration. `Atm%layout(1:2)` confirmed
authoritative; `layout_x/layout_y` now threaded additively through the
ABI/runtime, `<=0`-defaults-to-1, GPU-executed clean at C32/C512-2GPU/MPS
(jobs 58665460/58665576), `layout=1x1` correct every rank,
behavior-neutral within node-variance noise. All four >6-rank gates
re-confirmed untouched; a fifth, independent layout-consistency
`MPI_Allgather` FATAL was added, not a relaxation. **115 VERIFIED** the
previously-UNVERIFIED FMS within-tile rank ordering via a Fortran-only
24-rank `layout=(2,2)` diagnostic (zero CUDA involvement):
`local_rank_within_tile = sub_j*layout_x+sub_i` exactly, 4
ranks/tile-contiguous, `mpp_pe()==MPI_Comm_rank()`. **116 DESIGNED**
(no source change) the production `SubTileId` derivation: C++-runtime
location, NO new ABI field, `pelist(1)` not required, exact 1×1
reduction, 24-rank oracle, shadow-mode-first builder migration. See §9
for the next step. Docs in §11.

## 6. Standalone Integration Status

- CUDA dyn_core runs inside the real `StandAlone_FV3_Dycore.x` under the
  true production decomposition (`FV_NX=1`/`FV_NY=6`, six MPI ranks, one
  tile/rank), same executable as Fortran (selected via `AGCM.rc`'s
  `RUN_CUDA_FV3`).
- Public C ABI complete for per-run setup.
- ABI/compiler-family compatibility with a full **Intel-compiler
  production GEOS build** verified sound by construction (fixed-width
  `iso_c_binding` types, runtime `abi_version`/`struct_bytes` guards,
  explicit `MPI_Comm_f2c`). Revisit if the toolchain changes.
- Multi-GPU (`dev = mod(comm_rank, device_count)`, single-node only —
  see §7) validated at C512 at 2 GPUs/3-ranks-per-GPU: correctness
  (bitwise-exact), memory (19.68 GiB/GPU), performance (TOTAL
  1.120x/10.70% — Prompt 103). Up to 4-GPU spread memory/correctness-
  clean with zero code changes (P102) — 4-GPU performance unmeasured.
- CUDA MPS (`fv3_mps_lifecycle.sh`) requires `compute_mode=Default` (true
  on Discover `gpu_a100`); not yet wired into every standing job script
  as the default.

## 7. Active Limitations / Open Gaps

- `MOIST_CAPPA`/`USE_COND` unimplemented in CUDA — needs Riemann-solver
  `peg`/`gm2`/`cp2` terms (`nh_utils.F90`) — dedicated future prompt.
- `a_imp <= 0.999` unimplemented in CUDA (only `a_imp=1.0`/SIM1) —
  dedicated future prompt.
- Warm restart (`COLDSTART: 0`) confirmed broken (Fortran-backend source
  gap, not CUDA) — fix not yet attempted.
- `cuda_aware` `MpiHaloTransport`: correct/bitwise-validated at safe
  density but not performance-significant — retained as optional,
  non-default transport; no standing job script defaults to it.
- MPS + fusion flags not yet folded into every standing job script as
  the default (operational cleanup, low risk).
- Stale-`EXPDIR`/`scratch_*.log`-capture bug in reused C512 perf sbatch
  templates: **upgraded severity (Prompt 107)** — under concurrency (two
  jobs both pointing at the same stale directory) this caused a real
  `MPI_Abort`/job crash, not just harmless log-capture loss. Always grep
  any copied `fv3.j` for `setenv EXPDIR` before submission.
- **Prompt 107 (RESOLVED/VALIDATED)**: node-local `MPI_Comm_split_type`
  GPU binding validated single-node (bit-identical) and genuine 2-node
  multi-node, no regression — does **not** by itself resolve the
  separate `comm_size==6` constraint below. `alla100` QOS caps a job's
  summed (not per-node) GPU/mem (`MaxTRES=gres/gpu=4,mem=510000M`) —
  budget job-wide for any multi-node job. Multi-node CUDA MPS needs an
  explicit broadcast (`fv3_mps_lifecycle.sh` is head-node-only; use
  `srun --overlap --nodes=N --ntasks-per-node=1`). See
  `p107_node_local_gpu_binding_2026-09-30.md` §5.
- R4/R8 production (non-HS) validation covers only c8/c16/c32. HS
  validation covers c8-c128 up to 100-day, C256 (24h), C512 (6h) — no
  multi-day runs yet at C256/C512.
- `install/bin/` and per-experiment binary copies are not auto-synced
  with `build/bin/` after rebuild — always refresh and verify via
  `strings` before trusting a new flag.
- C128/C256/C512 each needed their own `bdt` stability search; don't
  assume linear `bdt` scaling holds at C1024+ without checking first.
- **Closed CUDA-port decisions — full detail in §5, do not reopen
  without fresh authorization.** `State_To_FV` is now CUDA-default at
  `device_count>=2` (Prompt 106); Fortran-default at 1-GPU and any
  other unsupported/unvalidated density. `DYN_EPILOGUE` (688.67 ms/
  call, MAPL-dominated) remains the sole open Prompt-96 candidate,
  feasibility uncertain.
- **Build blocker RESOLVED (Prompt 81B)**: `FV_StateMod.F90` mismatch
  was incremental-rebuild staleness — use a clean whole-target rebuild.
- **Structural cold-start tracer-zero limitation (Prompt 77, applies to
  the whole standalone harness family)**: default `COLDSTART=1`+
  `external_ic=.false.`+empty `ExtData.rc` means `fv_restart.F90` never
  seeds `Atm%q`, so it stays zero all run. Future tracer-transport
  validation needs a dedicated seeding mechanism — do not modify
  `fv_restart.F90` without authorization (declined for P77).
- **CUDA dyn_core bridge hardcodes `comm_size==6`** (one tile/rank), no
  intra-tile (same-face) halo path — the binding constraint for C1024+
  (P102 §9: memory gap is per-rank, HIGH confidence). Full 108-112
  sub-tile chain (architecture mapping -> topology/donor layer ->
  Phase-5 wiring -> transport-peer-identity fix) is **CLOSED and
  GPU-VALIDATED at real 24-rank MPI scale** — see §5/§11. Bridge
  metadata plumbing (`layout_x/layout_y` from `Atm%layout(1:2)`,
  Prompts 113-114) flows additively through ABI/bridge/runtime, WIRES
  ONLY, no sub-tile derivation used in production. **Still open**: the
  Fortran bridge's own `comm_size==6` gate
  (`fv3_gpu_dyncore_bridge.F90:1516`) and `create_mpi_transport`'s
  production `comm_size==6` gate remain intentionally untouched — no
  real >6-rank production path exists yet. FMS within-tile rank
  ordering is now **VERIFIED by Prompt 115** (`p115_*.md`); the
  `SubTileId` derivation from it is now **DESIGNED by Prompt 116**
  (`p116_*.md`, no source change) — C++-runtime location, no new ABI
  field, no `pelist(1)` needed. See §9 for the next (implementation)
  step before any sub-tile topology-builder work. Corner transport
  remains metadata-only, no message mechanism, separately gated.

## 8. Critical "Do Not Repeat" Lessons

- **Node-to-node A100 variance is large (>2x at C512)** — never treat a
  single-node absolute timing as representative; always pair Fortran/
  CUDA A/B in the same Slurm allocation.
- Rank/GPU density (1/2/3 GPUs at fixed rank count) has **no effect** on
  C512 **timing**. Density **does** dominate C512 **memory** — scales
  overwhelmingly per-rank/per-context, not fixed-per-GPU (P102) —
  spreading ranks across more GPUs is the primary headroom-recovery
  lever.
- **A post-hoc `nvidia-smi` checkpoint after a job exits reads near-idle,
  not the in-run footprint** (CUDA context already destroyed) — use a
  concurrent background poller, not a checkpoint after.
- **Before trusting an env-var-gated debug/counter mechanism, verify the
  instrumentation exists in source first** — Prompt 104 grepped for
  output from a dispatch counter that was never implemented, yielding a
  meaningless `0/0`. `grep -rn` the marker string in `cuda/` first.
- Production C256/C512 jobs use **Intel MPI 2021.13**, not the OpenMPI
  dev stack in `discover_platform.md` — check which stack a job loaded
  before generalizing an MPI-tuning finding.
- Before reusing a copied `fv3.j`/sbatch template, grep for
  `setenv EXPDIR` — a stale copy-paste value silently breaks
  `scratch_*.log` capture.
- **A job's own log line (`***** USING **** <path>`) is the only
  reliable evidence of which binary ran** — Prompt 53 found jobs that
  silently fell back to a stale `install/bin` binary. Always grep the
  log for the executable path before trusting a comparison result.
- **At C512/6-ranks-per-1-A100 density the shared GPU is already
  near-saturated** (§7 `State_To_FV`) — check `nvidia-smi` headroom
  before assuming a new persistent-buffer CUDA port will fit.
- **A single unsynchronized `cudaMemGetInfo`/`nvidia-smi` diagnostic
  across MPS-shared MPI ranks cannot cleanly attribute memory to a
  lifecycle stage** (P101) — only the converged end-state ceiling is
  trustworthy; per-stage needs an `MPI_Barrier` per checkpoint.
- Never infer GPU execution from compilation, or numerical validation
  from GPU execution — always state the specific level achieved
  (IMPLEMENTED / COMPILED / GPU EXECUTED / NUMERICALLY VALIDATED /
  PERFORMANCE VALIDATED).
- **Before concluding "no HISTORY output was produced," check the run's
  actual `scratch_*/` execution subdirectory, not just the top-level
  rundir** — MAPL History writes there, never at the top level. Prompt
  104 checked only the top level and wrongly concluded none existed;
  Prompt 105 found it in the same completed job, no rerun needed.

## 9. Next Recommended Task

**Performance track (`optimization/cuda-fv3-performance`, current
branch) — confirm scope before starting:**
0. DONE through Prompt 116 — full detail in §5/§7 (do not reopen closed
   CUDA-port decisions, re-verify FMS layout ordering, or redo the
   derivation design without fresh authorization). Sub-tile halo chain
   (108-112): C1024 architecture mapped, topology/donor-identification
   layer and Phase-5 field-data slice IMPLEMENTED + GPU EXECUTED +
   VALIDATED at real 24-rank MPI scale — see §7's sub-tile bullet and
   `p112_generalized_halo_transport_peer_ids_2026-09-30.md`. FMS layout
   metadata bridge-ABI plumbing (113-114): `layout_x/layout_y` flow
   through the ABI/bridge/runtime, GPU-executed clean at C32/C512, all
   >6-rank gates intact (`p114_*.md`). FMS within-tile rank-ordering
   verification (115): **VERIFIED** — see §5/§7/`p115_*.md`. Production
   `SubTileId` derivation design (116): **DESIGNED, no source change** —
   C++-runtime location, no new ABI field, no `pelist(1)` needed, exact
   1×1 reduction, 24-rank oracle, shadow-mode-first migration strategy
   — see §5/§7/`p116_*.md`.
1. **Recommended next step (implementation, bounded, SMALL)**: execute
   Prompt 116's slice — pure `subtile_id_from_rank(mpi_rank, tile,
   layout_x, layout_y)` helper in `fv3_gpu_halo_subtile_topology.hpp` +
   24-rank test-only oracle unit test (no MPI, no gate) + env-gated
   (`RUN_CUDA_FV3_SHADOW_SUBTILE_TOPOLOGY`, default OFF) 1×1
   shadow-topology equivalence check in `fv3_gpu_set_cubed_sphere_topology`
   (FATAL-on-mismatch, debug/test only). No new ABI field, no gate
   relaxation, no builder switch for real traffic, no real >6-rank run —
   see `p116_*.md` §15-17 for the file list/stop-condition check. A
   future C1024 run still needs its own `bdt` stability search (§7).
   Lower priority: (a) Phase B corner-transport design (deferred three
   times); (b) fold Prompt 109's duplicated vector-sign formula into one
   shared helper.
2. Lower priority: (a) tracer-numerics validation for the §7 cold-start
   tracer-zero limitation; (b) narrow `use_cuda_tp2d` to always-Fortran
   (hygiene); (c) `FV_DYNAMICS` non-loop residual (537.37 ms/call,
   partially MPI) remains unattributed.

**Mainline (`cuda/gpu-bringup`) tracks — confirm scope first:** (1) CUDA
`MOIST_CAPPA`/`USE_COND` (WP-C/WP-B, §7); (2) CUDA `a_imp <= 0.999`
(§7); (3) fold MPS lifecycle + fusion flags into standing job scripts.

**Other open HS-track follow-ons (confirm scope first):** C256/C512
climatology/longer duration; C128 toward ~200+ day equilibration;
warm-restart fix; higher resolution than C512 (needs its own stability
search); fix the `scratch_*.log`/stale-`EXPDIR` harness bug (§7).

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`. Mainline branch `cuda/gpu-bringup`
  HEAD `2e6327f`. HS track branch `experiment/held-suarez-e2e` HEAD
  `21da304` ("Validate C512 CUDA-aware MPI at reduced rank density").
  Current working branch `optimization/cuda-fv3-performance`. Neither
  pushed to `origin`. Push/merge only on explicit request.
- Working tree on `optimization/cuda-fv3-performance` has UNCOMMITTED
  diffs spanning PT/DELZ, REMAP_UV, Prompt 76 hord==8 extension (Stages
  0-8 GPU-VALIDATED, 9-15 BLOCKED BY DESIGN), REMAP_Q,
  REMAP_PRESSUREVARS, host-reference test files, `State_To_FV`'s
  Option-A port, and Prompt 106's `fv3_gpu_device_count()` getter in
  `cuda/interface/fv3_gpu_dyncore_bridge.F90` — all GPU-validated, no
  core remap numerics touched. Fortran-side selector edits (incl.
  Prompt 106's device_count-gated default) live in the separate
  `GEOSfvdycore_fortran_perf` checkout; current binary sha256
  `f6e30af3...016470` (supersedes the pre-P106 `2339366b...186e2064d`).
  Nothing committed/reverted in either repo. See each prompt's own doc
  (§11) for the per-prompt file list.
- Untracked, load-bearing gap (STILL OPEN, lives in the
  `GEOSfvdycore_fortran_perf` checkout, a different repo):
  `Held_Suarez_ForcingMod.F90` (this repo's root) is referenced by the
  tracked `CMakeLists.txt`/`FV_StateMod.F90` but never committed here —
  build is broken from a fresh checkout of *this* repo until added.
- Prompts 100-116 each added one untracked doc under `docs/performance/`
  plus forward-pointing edits to prior docs — no bridge/gate/numerical
  production source touched by any doc-only prompt (113, 115, 116).
  Prompt 111 added the additive builder header + 2 new/extended test
  files; Prompt 112 modified `fv3_gpu_halo_mpi.{hpp,cpp}`,
  `fv3_gpu_halo_exchange.cu`, `fv3_gpu_runtime.cpp` (peer-identity
  generalization) + one new test file + one `CMakeLists.txt`
  registration (§7's sub-tile bullet). Prompt 114 (detailed in §5) added
  4 additive-only ABI/bridge/runtime source edits, one new doc, and
  reconstructed the previously-missing `build_cuda_r8_intelmpi/` build
  dir plus two rundirs/one sbatch script. Prompt 115 added one 2-edit,
  env-gated diagnostic to `fv_mp_mod.F90` (`GEOSfvdycore_fortran_perf`
  checkout only) + one doc + two rundirs. Prompt 116 added one new doc
  only (`p116_*.md`), zero source files touched in either repo. Job
  scripts/rundirs live under the separate `GEOSfvdycore_fortran_perf`
  checkout — not source, not committed.
- Untracked scratch scripts/logs/probes exist at repo root and under
  `perf_rerun_logs/`, `scratch_*/` — not source; review file-by-file
  before any future commit. Reusable tooling lives under `tools/`.
- Do not commit: build dirs, profiler artifacts, `scratch_*/`, `*.out`
  Slurm logs, NetCDF output, restart data, binaries. Do commit (if
  explicitly requested): source fixes, permanent test/tooling, docs.

## 11. Documentation References

Chronological/technical detail for every closed or attempted prompt
remains in its own file under `docs/agent_handoff/`; read the specific
file only when the current task requires that level of detail. Full
project history, LOC accounting, and the complete experiment table are
now in `docs/agent_handoff/milestone_summary_2026-09-19.md` — start
there for anything historical. Below is condensed in §5/§7/§9; read a
file only for step-by-step detail.

- `prompt_43`-`prompt_107_*` — full CUDA-aware `MpiHaloTransport`
  bring-up through node-local multi-node GPU binding (PT/DELZ, REMAP_UV,
  TRACER_2D, REMAP_Q/PRESSUREVARS, REMAP_TOTALENERGYCON, `FV_To_State`,
  `State_To_FV`, 2-GPU C512, GPU binding) — all **CLOSED**, condensed in
  §5/§7/§9; read a specific `p*.md` only for step-by-step detail.
- `p108_*`-`p112_*` — C1024 sub-tile halo-exchange chain: 108 mapped the
  24-stage architecture (design only); 109 implemented/validated the
  topology/donor-identification layer at synthetic 24-rank scale; 110
  designed the Phase-5 wiring slice; 111 implemented it and found the
  `launch_halo_transfer` fixed-6-tile blocker; 112 fixed and
  GPU-validated it at real 24-rank MPI scale (0/22224 failures). All
  **CLOSED**, condensed in §5/§7/§9.
- `p113_*`-`p116_*` — FMS layout metadata -> CUDA bridge integration: 113
  designed, 114 implemented `layout_x/layout_y` ABI/bridge/runtime
  plumbing (GPU-executed clean at C32/C512, all >6-rank gates intact);
  115 VERIFIED the previously-UNVERIFIED within-tile rank ordering via a
  Fortran-only 24-rank `layout=(2,2)` diagnostic
  (`local_rank_within_tile = sub_j*layout_x+sub_i`, zero exceptions); 116
  DESIGNED (no source change) the production `SubTileId` derivation from
  that formula — C++-runtime location, no new ABI field, no `pelist(1)`
  needed, exact 1×1 reduction, 24-rank oracle, shadow-mode-first builder
  migration. All **CLOSED**, condensed in §5/§7/§9.

---

CURRENT_STATE maintenance policy:
- This file holds **current truth only** — replace superseded facts in
  place rather than appending a new dated paragraph.
- Detailed chronological history, per-field numerical tables, and
  resolved debugging narratives belong in
  `docs/agent_handoff/milestone_summary_2026-09-19.md` and the prompt-
  specific handoff docs, not here.
- Keep this file at or below 28,000 characters (hard ceiling 30,000). If
  an update would exceed that, compact an existing section before adding
  new content.
