# FV3 CUDA Modernization — Milestone Summary (2026-09-19)

## Purpose

Standalone, dated snapshot of the full project history to date. This
file is the durable historical record; `CURRENT_STATE.md` is being
compacted separately to hold only what the next coding agent needs to
continue work, and will reference this file for history instead of
repeating it.

Branch: `experiment/held-suarez-e2e`, HEAD `21da304`
("Validate C512 CUDA-aware MPI at reduced rank density"), branched from
`cuda/gpu-bringup` at merge-base `ce7f943`. Mainline branch
`cuda/gpu-bringup` HEAD `2e6327f` ("Document MPS rank-GPU performance
architecture"). Neither branch pushed to `origin` as of this writing.

---

## A. Code Modernization / Lines of Code

**Measurement basis**: no `cloc` available on this platform. All figures
below are `git ls-files | xargs wc -l` counts against the current
`experiment/held-suarez-e2e` HEAD (`21da304`), filtered by directory role
and file extension. These are **current tracked-tree totals**, not a
"Fortran lines converted to CUDA" figure — no single original pre-CUDA
commit exists as a clean baseline to diff against (the CUDA modernization
work was built up incrementally across ~43 commits touching `cuda/`,
starting from commit `a936aef`/`9834f77` "Add FV3 GPU runtime and state
contract skeleton", 2026-08-13/17). Where a conversion ratio would
require guessing, it is explicitly omitted rather than invented.

| Category | Lines | Files |
|---|---:|---:|
| CUDA/C++ production (`cuda/dyn_core/{cgrid,dgrid,vertical,operators,runtime,include,cmake}`) | 25,581 | 70 |
| CUDA test/benchmark (`cuda/dyn_core/{tests,benchmarks}`) | 21,673 | 27 |
| Fortran ISO_C_BINDING bridge production (`cuda/interface/*.F90`) | 3,218 | 5 |
| Fortran bridge tests (`cuda/interface/tests/`) | 379 | 2 |
| Tracked documentation (`docs/agent_handoff/`) | 29,719 | (per-prompt handoff docs) |
| Tracked documentation (`docs/agent_instructions/`) | 1,222 | 4 |
| Pre-existing root Fortran GridComp baseline (37 `.F90` files, present since the 2019 GEOSgcm_GridComp split, NOT part of the CUDA modernization) | 33,113 | 37 |

**Statement of measurement basis**: "The current CUDA modernization
branch contains 25,581 lines of CUDA/C++ production runtime code across
70 files, 21,673 lines of CUDA test/benchmark code across 27 files, and
3,218 lines of Fortran ISO_C_BINDING bridge code across 5 files, measured
against HEAD `21da304`." This is not a claim that a specific number of
authoritative Fortran lines were "converted" — the CUDA implementation is
an additive, coherent-subsystem reimplementation of the dyn_core numerical
path (WP-A/B/C/D), not a line-by-line port, and the pre-existing
33,113-line root Fortran GridComp tree remains present, unmodified in
its core numerics, and is the backend selected when `RUN_CUDA_FV3` is
off.

**Main code areas converted/added**:
- FV3 dyn_core numerical path: C-grid predictor (WP-A), D-grid dynamics/
  transport/damping (WP-B), vertical/nonhydrostatic dynamics (WP-C),
  pressure/grid/post-loop operators (WP-D) — all implemented as CUDA
  kernels under `cuda/dyn_core/{cgrid,dgrid,vertical,operators}`.
- Persistent GPU runtime/state (WP-E): GPU context, FieldRegistry, device
  state, persistent workspace, streams, dispatch metadata
  (`cuda/dyn_core/runtime/`, `cuda/dyn_core/include/`).
- Fortran/C ABI bridge: `iso_c_binding` wrapper, stable C ABI, public
  headers (`fv3_gpu_api.h`, `fv3_gpu_communication.h`), Fortran-side
  bridge/field-registry modules (`cuda/interface/*.F90`).
- Six-tile GPU-side cubed-sphere halo transport: `PeerHaloTransport`
  (in-process, six host threads) and `MpiHaloTransport` (host-staged and
  CUDA-aware variants, real six-rank MPI) behind a shared
  `HaloTransport` interface.
- Held-Suarez idealized forcing integration (`Held_Suarez_ForcingMod.F90`,
  backend-neutral, called from `FV_StateMod.F90::FV_Run` before dyn_core
  dispatch) — enables an end-to-end dry-dynamical-core Fortran-vs-CUDA
  comparison independent of moist physics.
- R4/R8 dual-precision support sharing one algorithmic structure.
- Multi-rank (six MPI ranks, one tile/rank) and multi-GPU
  (`dev = mod(comm_rank, device_count)` round-robin) runtime support.

---

## B. Major Technical Milestones (chronological)

1. **WP-E runtime skeleton** (2026-08-13/17, `9834f77`/`a936aef`): GPU
   context, FieldRegistry, device-state contract. IMPLEMENTED, COMPILED.
2. **WP-A/B/C/D CUDA kernel development**: C-grid predictor, D-grid
   transport/damping, vertical/nonhydrostatic Riemann solver, pressure/
   grid operators. IMPLEMENTED, COMPILED, unit-level GPU EXECUTED.
3. **Prompt 4/4.2 — integrated numerical path closure**: full 54-stage
   production first-run dyn_core inventory (WP-E 10, WP-A 7, WP-B 18,
   WP-C 9, WP-D 10) confirmed present in the integrated source path.
   IMPLEMENTATION COVERAGE: COMPLETE (not yet GPU-executed/validated as
   an integrated whole at this point).
4. **Prompt 5 — GEOS/FV3 interface**: `iso_c_binding` bridge, stable C
   ABI, init/step/finalize lifecycle, MPI communicator crossing via
   `MPI_Comm_f2c`. IMPLEMENTED, COMPILED.
5. **Prompt 5B — Discover CUDA bring-up**: validated GCC 12.3.0/OpenMPI
   4.1.6/CUDA 12.5 (nvcc V12.5.82) toolchain on Discover `gpu_a100`. R8/R4
   CUDA compilation PASS, 10/10 R8 and 10/10 R4 CPU-visible regression
   tests PASS, WP-E real GPU runtime PASS, compute-sanitizer PASS with 0
   errors. GPU EXECUTED (unit/package level).
6. **Prompt 6a/6b — Gate C2 closure**: authoritative Fortran savepoints
   generated; CUDA-vs-Fortran field comparison at c8 (72-call depth),
   c16/c32 (6-call depth). R8 PASS (worst `w` max_rel ~5e-11, round-off
   only). R4 PASS with R4-scaled thresholds. **Gate C2: CLOSED** — full
   54-stage inventory IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY
   VALIDATED against authoritative Fortran.
7. **Prompts 7-13 — performance baseline + launch fusion + CUDA Graph**:
   nsys bottleneck classification; halo-edge fusion, corner-fill fusion,
   CUDA Graph mode. All bitwise/numerically neutral vs. default. GPU
   EXECUTED, NUMERICALLY VALIDATED (neutrality).
8. **Prompts 14/15/18 — halo transport buildout**: `PeerHaloTransport`
   (in-process six-thread) and `MpiHaloTransport` (host-staged six-rank
   real MPI) built behind a shared `HaloTransport` interface; MPI
   transport determinism issue (test-harness stream-ordering defect) and
   FMS/CUDA halo-ownership ambiguity both root-caused and RESOLVED. Six-
   tile GPU halo exchange NUMERICALLY VALIDATED bitwise vs. in-process
   path.
9. **Prompts 19-23 — R4/R8 depth validation**: extended validation ladder
   at c8/c16/c32, CFL/`bdt` timestep-scaling methodology established for
   later resolution scaling.
10. **Prompts 24-27 — standalone integration + MPS**: CUDA dyn_core
    running inside the real `StandAlone_FV3_Dycore.x` under the true
    production six-rank/one-tile decomposition; nsys-based halo
    characterization; MPS rank-GPU architecture discovery and lifecycle
    scripting (`fv3_mps_lifecycle.sh`).
11. **Prompts 28-29 — fusion flags + CUDA Graph productionization**:
    env-var-gated (`RUN_CUDA_FV3_FUSE_HALO_EDGES`,
    `RUN_CUDA_FV3_FUSE_CORNER_FILL`, `RUN_CUDA_FV3_GRAPH`) opt-in perf
    knobs, default OFF, all NUMERICALLY VALIDATED neutral.
12. **Prompt 30 — production config closure**: confirmed
    `MOIST_CAPPA`/`USE_COND` and `a_imp <= 0.999` as UNIMPLEMENTED by
    design (triple-gated rejection, not a defect); verified Intel-
    compiler production-GEOS ABI compatibility by construction (fixed-
    width `iso_c_binding` types, runtime `abi_version`/`struct_bytes`
    guards).
13. **Prompts 31/32 — Held-Suarez end-to-end integration**: idealized
    dry-core forcing wired in, backend-neutral, gated by
    `RUN_HELD_SUAREZ`; C32 scientific validation PASS.
14. **Prompts 33/34 — HS multi-day/high-res ladder + perf reconciliation**:
    C64/C128 validation; C32 perf-methodology reconciliation (warmup-
    included vs. warmup-excluded conventions).
15. **Prompt 35 — HS climatology**: C128 30-day and 100-day runs, PASS,
    byte-identical logs, bitwise T/U/PS agreement through day 100 (not
    yet fully equilibrated — ~200+ day literature estimate). Warm restart
    (`COLDSTART: 0`) discovered **confirmed BROKEN** for this
    nonhydrostatic HS config — a genuine Fortran-backend source-level gap,
    not CUDA/config-specific.
16. **Prompts 36/37 — C256/C512 correctness + first performance points**:
    per-resolution `bdt` stability search; C256 24h gate PASS (bitwise,
    CUDA faster than Fortran, 0.874x); C512 6h gate PASS (bitwise), but
    first CUDA-vs-Fortran performance point showed CUDA 1.270x SLOWER
    (later superseded, see Prompt 48).
17. **Prompts 38-40 — C512 regression root-cause chain (part 1)**: message-
    size-driven halo-transport cost growth root-caused; regression
    isolated to the MPI transport/protocol layer (in-process
    `PeerHaloTransport` scales healthily; six-rank `MpiHaloTransport`
    collapses); OpenMPI eager/rendezvous tuning ruled out.
18. **Prompts 41/42 — CUDA-aware MPI environment**: Prompt 41's OpenMPI
    CUDA-aware-build "Hard Stop" was corrected by Prompt 42 — the real
    C256/C512 jobs run under **Intel MPI 2021.13**, not OpenMPI; a
    standalone device-buffer test then confirmed Intel MPI CUDA-aware
    transport PASS at C512 message size with `I_MPI_OFFLOAD=1`.
19. **Prompt 43 — CUDA-aware `MpiHaloTransport` implementation**:
    `RUN_CUDA_FV3_MPI_TRANSPORT=host_staged|cuda_aware` variant
    implemented (8 files: 2 Fortran interface + 6 C++/CUDA). Surfaced an
    unrelated pre-existing MAPL resource-read startup crash at C512
    (isolated via `git stash`+rebuild, fixed in Prompt 44) and a 2-rank
    device-buffer transport unit test intermittent SIGSEGV (later
    explained by Prompt 45). Now committed under `ed4871c`.
20. **Prompt 44 — MAPL resource-read crash fix**: root cause + fix
    (`FV_StateMod.F90` INTEGER-proxy for the HS resource read) for the
    startup crash blocking Prompt 43's C512 correctness gate. **C512
    host-staged: 10/10 PASS.** Surfaced a separate transport-layer
    runtime failure when resuming with `cuda_aware`.
21. **Prompt 45 — CUDA-aware failure root-cause**: embedded
    `cudaGetErrorString()` diagnostics into `fv3_vertical_kernels.cu`;
    root-caused Prompt 44's `cuda_aware` failure to **device memory
    capacity** (`cudaErrorMemoryAllocation`), not a code defect — six
    ranks already at 96.4% of one A100's 40GB under `host_staged`.
    Decision: do not resume the `cuda_aware` perf A/B at 6-ranks/A100.
    Now committed under `ed4871c`.
22. **Prompt 46 — 2-GPU CUDA-aware validation**: at 3-ranks/GPU across 2
    A100s, `cuda_aware` no longer OOMs (peak ~5053 MiB/GPU, 12.3% of
    capacity), BITWISE-identical to `host_staged`; paired perf A/B shows
    only a 0.03% `DYN_CORE` difference. Decision: VIABLE BUT NOT
    PERFORMANCE-SIGNIFICANT — `host_staged` remains the default. Flagged
    an informal ~2x 1-vs-2-GPU gap as an open item.
23. **Prompt 47 — GPU rank-density A/B**: controlled same-node/same-job
    1/2/3-GPU comparison (job 58459292); refuted Prompt 46's ~2x gap as a
    density/contention effect (all layouts within 0.5%). Surfaced a new,
    larger-than-expected 2.3x node-to-node CUDA absolute-timing swing vs.
    Prompt 37's original figure, left open pending a same-node paired
    rerun.
24. **Prompt 48 — authoritative same-node C512 Fortran-vs-CUDA
    reconciliation**: config audit (31 items, no confound found, historical
    9089.05 ms/call figure's provenance confirmed valid) followed by a
    same-node, same-allocation, alternating-order paired rerun (job
    58468244, node `warpa004`, 1 A100, 6 ranks). **Result: C512
    CUDA-vs-Fortran performance "regression" RESOLVED** — Fortran median
    6374.02 ms/call vs. CUDA median 6373.86 ms/call (ratio of medians
    0.99998). The historical 1.270x ratio is SUPERSEDED — attributable to
    node-to-node A100 hardware variance (a 3rd node, `warpa004`, adds a
    3rd distinct absolute CUDA figure alongside `warpa009`'s 9089 and
    `warpa007`'s 3892 ms/call, a >2x spread), not a CUDA-specific defect.
    This is the current authoritative C512 performance conclusion.

---

## C. Experiments Conducted

| Experiment | Resolution | Ranks/GPUs | Duration/calls | Purpose | Result |
|---|---|---|---|---|---|
| Gate C2 correctness | c8 | 1 rank, 1 GPU | 72-call depth | CUDA vs. Fortran savepoint agreement, production first-run config | PASS, round-off only (worst `w` max_rel ~5e-11) |
| R8 longer-depth run | c8 | 1 rank, 1 GPU | 72-call ladder | Extended R8 stability/agreement | PASS |
| R4 validation | c8 | 1 rank, 1 GPU | 72-call ladder | R4-scaled tolerance agreement | PASS (worst `w` max_rel ~2.4e-1, explained small-denominator artifact; absolute agreement ≤5.9e-5) |
| c16/c32 tests | c16, c32 | 1 rank, 1 GPU | 6-call depth | Broader-grid correctness at shallower depth | PASS |
| HS C32 | C32 | 6 ranks, 1 GPU | short gate | Held-Suarez scientific validation, smallest scale | PASS, bitwise |
| HS C64 | C64 | 6 ranks, 1 GPU | short gate | HS ladder step | PASS, bitwise |
| HS C128 | C128 | 6 ranks, 1 GPU | 10-day, 30-day, 100-day | HS multi-day stability + early climatology | PASS through day 100; not fully equilibrated |
| HS C128 100-day climatology | C128 | 6 ranks, 1 GPU | 100 simulated days | Longest-duration HS run to date | PASS, byte-identical logs, bitwise T/U/PS |
| HS C256 | C256 | 6 ranks, 1 GPU | 24h gate | Resolution-scaling correctness + first C256 perf point | PASS, bitwise; CUDA 0.874x (faster) |
| HS C512 | C512 | 6 ranks, 1 GPU | 6h gate | Resolution-scaling correctness + first C512 perf point | PASS, bitwise; CUDA 1.270x (slower) — later superseded |
| PeerHalo scaling discriminator | C256/C512 | 6 ranks (in-process) | comm-cost sweep | Isolate MPI-transport-layer vs. in-process halo scaling | PeerHalo scales healthily (3.69x); MpiHaloTransport collapses (8.32x) |
| CUDA-aware device-buffer unit test | N/A | 2 ranks | unit test | Confirm Intel MPI CUDA-aware transport viability at C512 message size | PASS with `I_MPI_OFFLOAD=1` |
| C512 1-GPU CUDA-aware OOM | C512 | 6 ranks, 1 GPU | startup | Resume Prompt 43's cuda_aware A/B at production density | FAILS — `cudaErrorMemoryAllocation`, 96.4% of 40GB already used by host_staged |
| C512 2-GPU CUDA-aware validation | C512 | 6 ranks, 2 GPUs (3/GPU) | 3-rep perf + correctness | Validate cuda_aware at a memory-safe density | PASS, BITWISE vs. host_staged; only 0.03% perf delta |
| Prompt 47 GPU-density A/B | C512 | 6 ranks, 1/2/3 GPUs | 3 reps × 3 layouts | Test rank-density as C512 regression cause | REFUTED — all layouts within 0.5% |
| Prompt 48 same-node Fortran/CUDA A/B | C512 | 6 ranks, 1 GPU | 3 alternating pairs (F1 C1 F2 C2 F3 C3) | Authoritative same-node CUDA-vs-Fortran reconciliation | RESOLVED — ratio of medians 0.99998, COMPARABLE |

---

## D. CUDA Performance vs. Fortran (current authoritative figures)

Standalone six-rank MPI path (`StandAlone_FV3_Dycore.x`, R8, 6 MPI
ranks/1 A100), `fv_timing_mod` `DYN_CORE` timer, MPS ON +
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES` + `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`,
Graph OFF (30 `dyn_core` calls/rep, 3 reps, warmup-included convention):

| Grid | Fortran ms/call | CUDA ms/call | CUDA/Fortran | Speedup (Fortran/CUDA) |
|---|---:|---:|---:|---:|
| C32 | 20.72 | 19.88 | 0.96 | 1.042x |
| C64 | 59.4 | 56.9 | 0.96 | 1.044x |
| C128 | 237.4 | 210.9 | 0.89 | 1.126x |
| C256 | 1250.75 | 1092.44 | 0.874 | 1.145x |
| **C512 (same-node, authoritative)** | **6374.02** | **6373.86** | **~1.00** | **~1.00x** |

**C512 conclusion: FORTRAN/CUDA COMPARABLE.** The historical C512 ratio
of 1.270 (CUDA slower) is **SUPERSEDED** — it compared Fortran and CUDA
runs measured on different physical nodes (`warpa009` vs. a separate
Fortran run) with materially different absolute per-node performance,
not a genuine CUDA-specific slowdown. Prompt 48's same-node, same-
allocation, alternating-order paired rerun (job 58468244, node
`warpa004`) is the current authoritative C512 comparison: Fortran median
6374.02 ms/call, CUDA median 6373.86 ms/call, ratio of medians 0.99998,
all 6 reps within ~0.3-0.5%.

Prompt 47's controlled same-node 1/2/3-GPU rank-density A/B separately
**refuted GPU rank-density as a cause** of the previously observed C512
CUDA-timing spread (all three layouts' medians within 0.49% of each
other). The dominant explanation for the wide swings in absolute C512
CUDA timings across prompts (9089 ms/call on `warpa009`, 3892 ms/call on
`warpa007`, 6374 ms/call on `warpa004` — a >2x spread for a nominally
identical configuration) is **node-to-node A100 hardware/thermal
variance**, confirmed generic (not CUDA-specific) because Fortran tracks
CUDA's absolute number almost exactly when measured on the same node
(Prompt 48).

CUDA-aware MPI transport was found **correct and environment-viable**
(Intel MPI 2021.13 + `I_MPI_OFFLOAD=1`) and **numerically bitwise-
identical** to host-staged transport at a memory-safe rank density
(2 GPUs, 3 ranks/GPU), but gave a **negligible performance benefit**
(~0.03% `DYN_CORE` delta, 1.5-2.3% on `COMM_DSW`/`COMM_TOTAL`) — not
performance-significant enough to change the default. **Host-staged
transport remains the preferred/default C512 transport** for both
simplicity and correctness-gate readiness (10/10 startup PASS).

(A separately-tracked, methodologically distinct C32 production-config
baseline — warmup-excluded, 100 calls/rep — measured Fortran 19.930
ms/call vs. CUDA 24.300 ms/call, 1.22x slower. This figure is internally
consistent within its own methodology but not directly comparable to the
warmup-included ladder above; both are retained for provenance.)

---

## E. Scientific / Numerical Validation Summary

- Full 54-stage production first-run dyn_core inventory (WP-E 10, WP-A 7,
  WP-B 18, WP-C 9, WP-D 10): IMPLEMENTED, COMPILED, GPU EXECUTED,
  NUMERICALLY VALIDATED against authoritative Fortran savepoints at
  meaningful in-loop boundaries and at dyn_core exit — not final-output-
  only comparison.
- R8: PASS at c8 (72-call depth) and c16/c32 (6-call depth); worst
  discrepancies are round-off-level only (`w` max_rel ~5e-11).
- R4: PASS at the same grids with R4-scaled thresholds; worst `w`
  max_rel ~2.4e-1 is an explained small-denominator artifact with
  absolute agreement ≤5.9e-5 — not a genuine divergence.
- Six-tile GPU halo exchange (in-process `PeerHaloTransport` and
  cross-process `MpiHaloTransport`): IMPLEMENTED, COMPILED, GPU EXECUTED,
  NUMERICALLY VALIDATED bitwise against the in-process reference path,
  at six real MPI ranks/six tiles.
- Held-Suarez idealized dry-core forcing: NUMERICALLY VALIDATED bitwise/
  round-off at C8-C64 (short gates), C128 (10/30/100-day, bitwise T/U/PS
  at all checkpoints through day 100), C256 (24h gate, bitwise T/U/V/PS
  at 4 checkpoints), and C512 (6h gate, bitwise T/U/V/PS at 3
  checkpoints) — all with byte-identical Fortran/CUDA logs and 0
  NaN/Range_Warn/Error hits.
- Known benign artifacts, not genuine divergences: a subnormal-magnitude
  2-cell halo-padding uninitialized-memory difference in `delpc`/`pkc` at
  savepoint level only (never surfaces at NetCDF level); a benign FMS
  plausibility-bound `range_warn` at C128 10-day (6966 hits, identical
  both backends, confirmed pre-existing/benign); a near-surface NaN
  artifact in the 3 lowest of 25 levels from standard MAPL vertical-
  interpolation masking (identical both backends, pre-existing).
- **Warm restart (`COLDSTART: 0`) is confirmed BROKEN** for this HS/
  nonhydrostatic standalone configuration — a genuine Fortran-backend
  source-level gap, not a CUDA-specific or configuration issue. CUDA
  restart is correctly skipped once the Fortran side's own restart fails.
  All validated runs to date use fresh `COLDSTART: 1` starts.
- **Do not overclaim full-physics GEOS support**: all HS validation is a
  dry dynamical core with idealized forcing, not the full moist-physics
  GEOS production configuration. `MOIST_CAPPA`/`USE_COND` and
  `a_imp <= 0.999` remain unimplemented in CUDA by design (§F) — the
  validated envelope is the SIM1 (`a_imp=1.0`), non-`MOIST_CAPPA`
  production first-run configuration only.

---

## F. Current Limitations / Open Items

- `MOIST_CAPPA`/`USE_COND` unimplemented in CUDA (needs Riemann-solver
  `peg`/`gm2`/`cp2` terms from `nh_utils.F90`) — a dedicated future
  prompt.
- `a_imp <= 0.999` unimplemented in CUDA (only SIM1/`a_imp=1.0` is
  implemented) — a dedicated future prompt.
- Warm restart (`COLDSTART: 0`) confirmed broken for the HS/
  nonhydrostatic standalone config — a Fortran-backend source gap, fix
  not yet attempted.
- `cuda_aware` `MpiHaloTransport` is implemented, correct, and bitwise-
  validated at a memory-safe rank density, but not performance-
  significant — retained as an optional, non-default transport; not
  production-cleanup-complete (e.g. not yet the target of any standing
  job script default).
- Operational/production-script cleanup pending: MPS lifecycle + fusion
  flags are not yet folded into every standing job script as the default.
- A harness-only stale-`EXPDIR`/`scratch_*.log`-capture bug in reused
  C512 perf sbatch templates (root-caused in Prompt 48's config audit,
  confirmed harmless to physics/timing where it occurred) has not been
  fixed at the template level — low priority, not urgent.

(The C512 CUDA-vs-Fortran performance question, the GPU rank-density
question, and the CUDA-aware-MPI environment/correctness question are
all now CLOSED — see §B/§D — and are intentionally not re-listed here as
open items.)

---

## G. Project Status in Plain English

- The FV3 dynamical core's numerical path (C-grid predictor, D-grid
  transport, vertical/nonhydrostatic solver, pressure/grid operators) has
  been reimplemented as a persistent-GPU-resident CUDA subsystem,
  reachable from the authoritative Fortran GEOS/FV3 code through a
  stable `iso_c_binding`/C ABI bridge — the original Fortran backend is
  always preserved and selectable.
- That CUDA implementation has been proven numerically correct against
  the authoritative Fortran reference at the individual-field level, not
  just at final output, across R8 and R4 precision, and across single-
  rank and real six-rank/six-tile MPI cubed-sphere execution.
- An idealized Held-Suarez dry-core configuration has been validated
  end-to-end, bitwise, from small test grids (C8) up through production-
  relevant global resolutions (C256, C512), including a 100-simulated-day
  climatology run at C128.
- At every resolution tested, CUDA is numerically indistinguishable from
  Fortran (bitwise or round-off only); performance-wise CUDA is faster
  than Fortran from C32 through C256 (up to ~1.15x), and performance-
  comparable (not faster, not slower) to Fortran at C512 once node-to-
  node hardware variance is controlled for.
- A significant amount of investigative work went into a C512
  performance discrepancy that initially looked like a real CUDA
  regression; it was root-caused this session to be an artifact of
  comparing runs on different physical GPU nodes, not a genuine defect —
  this question is now closed.
- CUDA-aware MPI transport was explored as a potential C512 performance
  improvement; it works correctly and is numerically bitwise-equivalent
  to the simpler host-staged transport, but offers no measurable speed
  benefit at a memory-safe rank density — the simpler host-staged
  transport remains the default.
- What remains before broader GEOS production readiness: two known
  numerical-configuration gaps in the CUDA implementation
  (`MOIST_CAPPA`/moist-physics coupling, and `a_imp <= 0.999` for looser
  implicit-solver configurations), a Fortran-backend warm-restart defect
  unrelated to CUDA, and routine operational cleanup (folding
  performance-flag defaults into standing job scripts).
- No moist-physics, full-GEOS-production validation has been performed
  yet — everything validated so far is either the idealized dry
  Held-Suarez configuration or short synthetic production-config test
  grids; that distinction should not be blurred in any future summary.
