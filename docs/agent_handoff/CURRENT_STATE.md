# Current Project State

_Last updated 2026-09-19 (checkpoint: milestone summary + CURRENT_STATE
compaction). Full project history moved to
`docs/agent_handoff/milestone_summary_2026-09-19.md` — read that file for
chronological milestones, the full experiment table, and detailed
performance reconciliation narrative. This file holds only what the next
coding agent needs to continue work: current architecture, current
validated status, active limitations, and the next recommended target._

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
    is authoritative; `install/bin/` and per-experiment job-directory
    copies are NOT auto-synced — refresh with `cp -f` and verify via
    `strings <bin> | grep <new-env-var>` after any rebuild.
  - **MPI implementation used by every C256/C512 job to date is Intel
    MPI 2021.13** (`comp/intel/2024.2.0` + `mpi/impi/2021.13`) — NOT the
    "validated" GCC+OpenMPI 4.1.6 dev stack in `discover_platform.md`
    (that stack is for standalone CUDA-only dev/PeerHalo work, build dir
    `build_cuda_r8_mpi`). The CUDA build matched to Intel MPI is
    `build_cuda_r8_intelmpi`/`build_cuda_r4_intelmpi` (GCC host compiler,
    Intel MPI runtime). Do not assume an OpenMPI-only finding applies to
    the production C512 path without checking which stack a given job
    actually loaded.
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
  `MpiHaloTransport` (host-staged or CUDA-aware real MPI, six processes,
  one tile/rank — opt-in via CMake
  `FV3_DYN_CORE_ENABLE_MPI_HALO`/runtime `--mpi-halo`, default off).
  Transport variant selected via `RUN_CUDA_FV3_MPI_TRANSPORT=
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
  Fortran savepoints. R8 PASS at c8 (72-call depth), c16/c32 (6-call
  depth) — round-off only (worst `w` max_rel ~5e-11). R4 PASS at the same
  grids with R4-scaled thresholds (worst `w` max_rel ~2.4e-1, explained
  small-denominator artifact, absolute agreement ≤5.9e-5).
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

  Benign known artifacts (not new divergences): 2-cell halo-padding
  uninitialized-memory difference in `delpc`/`pkc` at savepoint level
  only (subnormal magnitude); C128 10-day benign FMS `range_warn`
  (identical both backends); near-surface NaN in 3 lowest of 25 levels
  (standard MAPL vertical-interpolation masking, identical both
  backends, pre-existing).
- **Warm restart (`COLDSTART: 0`) is confirmed BROKEN** for this HS/
  nonhydrostatic standalone config — a genuine Fortran-backend source-
  level gap, not CUDA/config-specific. CUDA restart correctly skipped
  once Fortran's own restart fails. Use fresh `COLDSTART: 1` runs only.
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

**C512 performance question: CLOSED.** A same-node, same-allocation,
alternating-order Fortran/CUDA rerun (job 58468244, node `warpa004`)
found the backends performance-comparable (ratio of medians 0.99998).
The historical 1.270x "CUDA slower" figure is **SUPERSEDED** — it
compared runs on different physical nodes; three nodes sampled so far
(`warpa009`/`warpa007`/`warpa004`) show a >2x spread in absolute CUDA
`DYN_CORE` figures (9089/3892/6374 ms/call) for nominally identical
config, and Fortran tracks CUDA's number almost exactly on a given
node — confirming generic node/hardware variance, not a CUDA-specific
effect. GPU rank-density (1/2/3 GPUs at C512) was separately ruled out
as a cause (all layouts within 0.5%, job 58459292).

**CUDA-aware MPI transport**: correct, environment-viable (Intel MPI
2021.13 + `I_MPI_OFFLOAD=1`), numerically bitwise-equivalent to
host-staged at a memory-safe rank density (2 GPUs, 3 ranks/GPU), but
performance benefit is negligible (~0.03% `DYN_CORE` delta) — does not
displace `host_staged` as the default. At the original 6-ranks/1-A100
density it is blocked by device memory capacity
(`cudaErrorMemoryAllocation`, six ranks already at 96.4% of one A100's
40GB under `host_staged`).

**Any future A/B performance comparison must run all arms on the same
allocated node/job** — node-to-node A100 variance is large enough (>2x
at C512) to dominate any smaller effect being measured.

Full reconciliation narrative, per-node tables, and the full C512
root-cause chain: `docs/agent_handoff/milestone_summary_2026-09-19.md`
§B/§D, and `prompt_38` through `prompt_48_c512_same_node_fc.md`.

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
- Multi-GPU (`dev = mod(comm_rank, device_count)`) validated at C512
  scale at 2 GPUs/3-ranks-per-GPU (both transports, correctness +
  performance); 4-GPU and MPS+multi-GPU composition still unmeasured.
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
- A harness-only stale-`EXPDIR`/`scratch_*.log`-capture bug in reused
  C512 perf sbatch templates (confirmed harmless to physics/timing where
  it occurred) not yet fixed at the template level — grep any copied
  `fv3.j` for `setenv EXPDIR` and confirm self-reference before reuse.
- R4/R8 production (non-HS) validation covers only c8 (deepest) and
  c16/c32 (6-call depth). HS validation covers c8-c128 up to 100-day
  duration, C256 (24h gate), C512 (6h gate) — no multi-day runs yet at
  C256/C512.
- `install/bin/StandAlone_FV3_Dycore.x` and per-experiment binary copies
  are not auto-synced with `build/bin/` after rebuild — always refresh
  and verify via `strings` before trusting a new flag.
- C128/C256/C512 each needed their own `bdt` stability search; do not
  assume linear scaling of a resolution-scaled `bdt` heuristic holds at
  C1024+ without a short stability check first.

## 8. Critical "Do Not Repeat" Lessons

- **Node-to-node A100 variance is large (>2x observed at C512)** — never
  treat a single-node absolute timing as representative, and never
  compare backends measured on different nodes/jobs as if it were a
  controlled A/B. Always pair Fortran/CUDA (or any A/B) in the same
  Slurm allocation.
- Rank/GPU density (1/2/3 GPUs at fixed total rank count) has been
  directly tested and shown to have **no effect** on C512 timing — don't
  re-investigate density as a performance lever without new evidence.
- The MPI implementation actually used in production C256/C512 jobs is
  **Intel MPI 2021.13**, not the OpenMPI dev stack in
  `discover_platform.md` — check which stack a job loaded before
  generalizing an MPI-tuning finding.
- Before reusing any copied `fv3.j`/sbatch template from a prior prompt's
  run directory, grep for `setenv EXPDIR` and confirm it self-references
  the new directory — a stale copy-paste `EXPDIR` silently breaks
  `scratch_*.log` capture (though not physics/binary selection).
- Do not infer GPU execution from compilation, and do not infer numerical
  validation from GPU execution — always state the specific validation
  level achieved (IMPLEMENTED / COMPILED / GPU EXECUTED / NUMERICALLY
  VALIDATED / PERFORMANCE VALIDATED).

## 9. Next Recommended Task

**Mainline (`cuda/gpu-bringup`) tracks — confirm scope with the user
before starting any:**
1. CUDA `MOIST_CAPPA`/`USE_COND` implementation (WP-C/WP-B, §7). Smaller,
   better-bounded gap of the two remaining correctness gaps if the
   Riemann-solver term addition can be scoped tightly.
2. CUDA `a_imp <= 0.999` implementation (§7) — broader implicit-solver
   change, likely larger scope than #1.
3. Operational wiring — fold MPS lifecycle + fusion flags into every
   standing job script as the default (lower risk than 1/2).

**Other open HS-track follow-ons (confirm scope before starting):**
- C256/C512 climatology / longer duration (only short gates run so far).
- Further duration at C128 toward ~200+ day HS equilibration estimate.
- Warm-restart fix (dedicated future prompt).
- Higher resolution than C512 (C1024+) — needs its own stability search.
- Fix the reused C512 perf-harness `scratch_*.log`/stale-`EXPDIR` bug
  (§7) before it is reused again — low priority.

## 10. Current Git / Working-Tree State

- Repository: `FVdycoreCubed_GridComp`. Mainline branch `cuda/gpu-bringup`
  HEAD `2e6327f`. HS track branch `experiment/held-suarez-e2e` HEAD
  `21da304` ("Validate C512 CUDA-aware MPI at reduced rank density").
  Neither pushed to `origin`. Push/merge only on explicit request.
- Working tree is source-clean as of this checkpoint: no pending diffs
  against any tracked CUDA/C++/Fortran source file. The only tracked
  modification is this file (`CURRENT_STATE.md`).
- Untracked, load-bearing production gap: `Held_Suarez_ForcingMod.F90`
  (repo root) is referenced by the tracked `CMakeLists.txt` build-source
  list and imported by the tracked `FV_StateMod.F90`, but has never been
  committed — the build is currently broken from a fresh checkout of
  this branch until this file is added.
- Numerous untracked scratch shell scripts/logs/probes/build artifacts
  exist at the repo root and under `perf_rerun_logs/`, `scratch_*/` —
  none of these are source and none should be committed. Reusable
  project tooling lives under `tools/` (savepoint validation, grid-
  metrics conversion, NetCDF inspection) — legitimate but should be
  reviewed file-by-file (excluding `tools/savepoints/__pycache__/`)
  before adding to any future commit.
- Do not commit: build directories (`build_cuda_r8*/`), profiler
  artifacts (`*.nsys-rep`, `*.sqlite`), `scratch_*/`, any `*.out` Slurm
  log, NetCDF output, restart data, compiled binaries.
- Do commit (if explicitly requested): legitimate source fixes,
  permanent test/tooling additions under `tools/`, documentation.

## 11. Documentation References

Chronological/technical detail for every closed or attempted prompt
remains in its own file under `docs/agent_handoff/`; read the specific
file only when the current task requires that level of detail. Full
project history, LOC accounting, and the complete experiment table are
now in `docs/agent_handoff/milestone_summary_2026-09-19.md` — start
there for anything historical. Otherwise, key recent files:

- `prompt_43_intelmpi_cuda_aware_ab.md` — CUDA-aware `MpiHaloTransport`
  impl (Stages 0-5 complete); Stage 7 blocked by a MAPL crash fixed in
  Prompt 44.
- `prompt_44_mapl_resource_race.md` — MAPL crash root cause + fix; 10/10
  C512 host-staged acceptance.
- `prompt_45_cuda_aware_correctness.md` — root-caused Prompt 44's
  transport failure to device memory capacity.
- `prompt_46_c512_two_gpu_cuda_aware_ab.md` — 2-GPU `cuda_aware`
  correctness + negligible perf delta.
- `prompt_47_c512_gpu_density_ab.md` — rank-density refuted as a C512
  performance cause.
- `prompt_48_c512_config_audit.md` / `prompt_48_c512_same_node_fc.md` —
  config audit + authoritative same-node C512 Fortran-vs-CUDA rerun
  (current authoritative C512 performance conclusion, §5).

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
