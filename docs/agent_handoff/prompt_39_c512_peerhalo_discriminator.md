# Prompt 39: C512 PeerHalo vs MPI Discriminator

Branch: `experiment/held-suarez-e2e`. Model: Sonnet only, per task policy.
Scope: single discriminator experiment to determine whether Prompt 38's
C512 CUDA-vs-Fortran communication regression (CUDA/Fortran=1.270,
`COMM_DSW` = 97.6% of the delta) is primarily (A) MPI transport/protocol
cost or (B) generic GPU halo pack/unpack/staging cost. No CUDA-aware MPI,
MCA tuning, halo batching, kernel optimization, or numerical change
attempted — diagnostic only, using the existing `fv3_gpu_perf_bench`
harness at C512-equivalent dimensions.

## Stage 1: Harness Support Verification

Read `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` in full (3672
lines) and `perf_rerun_logs/perf_bench.sbatch` (historical C8/16/32 usage).
Confirmed, no source changes needed:

- **Tile dimensions**: `--grid N` sets `NI=NJ=N` at runtime (no
  compile-time bound found in `main()`/`setup()`; arrays are
  `std::vector`-backed, sized dynamically). `--grid 512` requires no
  source modification.
- **Vertical levels**: `--npz 10`, matching Prompt 37.
- **Transport selected**: default path (no `--mpi-halo`) = single-process,
  six-host-thread `PeerHaloTransport` (in-process `HaloPeerGroup`),
  confirmed at `run_one_repeat()` (line ~3403) vs. the `#if
  FV3_DYN_CORE_ENABLE_MPI_HALO` / `g_mpi_halo` branch used only when
  `--mpi-halo` is passed. This is exactly the harness Prompts 25a/26/38
  refer to.
- **Fusion settings**: `--fuse-corner-fill`/`--fuse-halo-edges` are real
  CLI flags (confirmed at argv-parsing, lines 3542-3543), matching the
  standing recommended CUDA config
  (`RUN_CUDA_FV3_FUSE_HALO_EDGES`/`FUSE_CORNER_FILL=YES` in the production
  path). Both passed. `--cuda-graph` exists but was omitted (default off,
  matches Graph-off requirement); confirmed `cuda_graph=0` in the run's own
  banner line.
- **Iteration/call count**: `--warmup`/`--iters`/`--repeats` are real CLI
  flags. Historical convention (Prompt 26-era `perf_bench.sbatch`) is
  `warmup=20 iters=100 repeats=5`. Reduced to `warmup=5 iters=30 repeats=3`
  to bound wall-clock cost at C256/C512 while preserving the project's
  established "30 measured calls/rep, 3 reps" convention used everywhere
  else in the HS ladder (Prompts 33/34/36/37) — not an invented protocol.
- **Mode**: `--mode resident` selected (not `both`/`transfer`). Mode A
  ("resident") is a pure chained-dependency loop of repeated
  `fv3_gpu_dyn_core_step` calls with no explicit per-iteration
  `sync_from_host`/`sync_to_host` — the correct analog to the production
  path's `DYN_CORE` FMS-clock timer (which times repeated dyn_core calls
  only, not explicit end-of-loop transfers, per the persistent-residency
  architecture). Mode B ("transfer_inclusive") would add a per-iteration
  H2D+D2H round trip not present in the production `DYN_CORE` measurement
  and was not used.
- **GPU count**: 1 (`--gres=gpu:1`). **Process count**: 1 (single process,
  six threads, one per tile) — this is the in-process PeerHaloTransport
  decomposition, distinct from the six-MPI-rank production path; the
  comparison is scaling-behavior-only, not an absolute-speed comparison
  (per the task's explicit limitation).
- Binary: existing `./build_cuda_r8_perf/fv3_gpu_perf_bench` (built
  2026-09-04), used as-is — no rebuild. `strings` confirmed the fusion/
  mpi-halo flag literals are present in the binary, consistent with the
  source read. (Minor, immaterial note: the binary's mtime is ~12 minutes
  older than the source file's current mtime; the flags this experiment
  needs are present in both, so this was not treated as a blocker.)

**Stage 1 result: harness fully supports C512 without any source change.**
No STOP condition triggered.

## Stage 2: C256 Control

Searched `perf_rerun_logs/` and the whole repo for any prior grid=256 or
grid=512 measurement from this harness: **none found** — the only existing
log (`perf_bench_58226816.out`) covers grids 8/16/32 only (Prompt 26-era).
Per the task's instruction, ran exactly one C256 control (combined into
the same job as the C512 run, sequential, same session/GPU/config).

## Stage 3: C512 PeerHalo Run

Job 58445168, `gpu_a100`/`rome`, 1 GPU, COMPLETED, exit code 0, elapsed
5:31. Command (both grids, same job):

```
$BIN --grid {256,512} --npz 10 --n_split 3 --warmup 5 --iters 30 --repeats 3 \
     --mode resident --fuse-corner-fill --fuse-halo-edges \
     --hardware A100_p39_discriminator
```

## Stage 4: Health Check

- rc = 0 for both grids (`3A.11 CUDA perf benchmark: PASS` printed for
  both the 256 and 512 blocks).
- No FAIL, no NaN/Inf reported anywhere in the 1751-line log (grep for
  `fail|skip|nan|inf|error` returned zero matches outside error-message
  format strings that never fired).
- `delp` positivity verified over the owned box, all 6 tiles, all 3
  repeats, both grids (`tile{1..6} delp: positive over owned box
  (256x256x10)` / `(512x512x10)`).
- No allocation failure, no CUDA error.

**Stage 4: PASS. No STOP condition triggered.**

## Stage 5: C256 vs C512 PeerHalo Scaling

Raw CSV (`--mode resident`, `ms_per_step` column = ms/call):

| Grid | Rep 1 | Rep 2 | Rep 3 | Median |
|---|---|---|---|---|
| C256 | 149.1047 | 148.8074 | 148.9582 | **148.9582** |
| C512 | 550.5161 | 550.2646 | 550.2192 | **550.2646** |

Rep-to-rep spread: C256 <0.2%, C512 <0.03% — both far tighter than the
production MPI CUDA C512 spread (<3.2%, Prompt 37), a first secondary
signal that the in-process path is not under the same contention/
scheduling variance as the six-rank MPI path.

**PeerHalo growth factor (C512/C256) = 550.2646 / 148.9582 = 3.694x**

Compared against the two reference growth factors (production path,
Prompt 36/37 figures, not re-derived):

| Path | C256 ms/call | C512 ms/call | Growth (C512/C256) |
|---|---|---|---|
| Fortran (production) | 1250.75 | 7155.63 | **5.72x** |
| CUDA six-rank MpiHaloTransport (production) | 1092.44 | 9089.05 | **8.32x** |
| CUDA PeerHaloTransport (this benchmark) | 148.9582 | 550.2646 | **3.69x** |

PeerHalo's growth factor (3.69x) is **lower than both** the Fortran
growth factor (5.72x) and, dramatically, the six-rank MPI CUDA growth
factor (8.32x). It is also close to (slightly under) the naive
compute-bound expectation for a resolution doubling (~4x, from O(N^2)
cell-count growth with the halo/edge-length term growing only ~2x) — i.e.
PeerHaloTransport shows **no disproportionate cost growth** at C512.
(Absolute magnitudes are not directly comparable across the two harnesses
per the task's stated limitation — different workload composition,
single-process vs. six-rank, no MPS in this benchmark — but the growth
*factor* comparison is valid and is the only thing being used here.)

## Root-Cause Classification

**CASE A**: PeerHalo scales normally (growth 3.69x, in line with or better
than Fortran's own 5.72x scaling) while the six-rank `MpiHaloTransport`
path collapses (8.32x growth, and an absolute regression from 0.874x to
1.270x CUDA/Fortran, Prompt 37). The generic GPU-side halo pack/unpack/
staging mechanism — which PeerHaloTransport also exercises, since it
shares all topology/rotation/sign/pack/unpack logic with MpiHaloTransport
per the `HaloTransport` interface design (`CURRENT_STATE.md` §3) — remains
healthy at C512. The disproportionate cost growth is confined to whichever
mechanism differs between the two backends: MPI process-to-process
transport (rendezvous/eager protocol crossover, inter-process
synchronization) rather than the shared device-side pack/copy/unpack code.

**PRIMARY ROOT CAUSE = MPI TRANSPORT/PROTOCOL PATH.**

## Stage 6: Halo Timing Breakdown

**NOT AVAILABLE.** Confirmed by reading the full harness source: the only
timer exposed is one aggregate elapsed_ms/ms_per_step per repeat (max
across the six tiles), printed via the `CSV,...` line. No pack/transport/
unpack/synchronization sub-timers exist in this harness. No instrumentation
was added, per the task's explicit prohibition.

## Stage 7: MPI-Specific Next-Experiment Ranking

Per the task's fixed ranking for this branch of the decision tree:

1. **Eager/rendezvous MCA threshold tuning** (cheapest, no code/rebuild,
   pure environment/MCA parameter — e.g. `btl_vader_eager_limit`/
   `btl_smcuda_eager_limit`/PML eager-limit variants — swept across the
   observed message-size crossover, C256 ordinary message ≈67.8KB vs C512
   ≈135.4KB, Prompt 38 §1).
2. CUDA-aware MPI (highest implementation cost; not the cheapest
   discriminator, and this experiment already narrows the target to the
   MPI layer specifically, so a full CUDA-aware-MPI implementation is not
   the next diagnostic step).
3. Message batching/fusion (changes the halo protocol shape; more
   invasive than an MCA parameter sweep for a first discriminator).

**Recommended next experiment (ONE, not implemented in this prompt):**
Rerun the existing six-rank `MpiHaloTransport` C512 configuration
(Prompt 37's exact job) under one or two alternative OpenMPI eager-limit
MCA settings (raising the eager threshold above C512's ~135.4KB ordinary
message size) and compare `COMM_DSW`/`DYN_CORE` FMS-clock timing against
the existing 9089.05 ms/call baseline — pure runtime configuration, no
source change, no CUDA-aware MPI library integration.

## CUDA-Aware MPI Priority Update

Per the task's rule: "PeerHalo healthy and MPI slow → MODERATE→HIGH
CANDIDATE." PeerHalo is healthy (3.69x growth, below both reference paths)
while the six-rank MPI path is slow (8.32x growth, regression). **Priority
raised: MODERATE (contingent) -> HIGH CANDIDATE.** Still **not
implemented** — this remains a diagnostic-only prompt.

## Summary Table (all stages)

| Stage | Result |
|---|---|
| 1. Harness support | Confirmed, no source change needed |
| 2. C256 control | No prior measurement existed; ran one (combined with Stage 3 job) |
| 3. C512 PeerHalo run | Job 58445168, rc=0, 5:31 elapsed |
| 4. Health check | PASS — 0 FAIL/NaN/Inf, delp positive, all 6 tiles x 3 reps x 2 grids |
| 5. Scaling comparison | PeerHalo 3.69x vs. Fortran 5.72x vs. MPI CUDA 8.32x |
| Root cause | MPI TRANSPORT/PROTOCOL PATH |
| 6. Halo breakdown | NOT AVAILABLE (no sub-timers in harness) |
| 7. Next experiment | Eager/rendezvous MCA threshold sweep at C512 (not implemented) |
| CUDA-aware MPI priority | MODERATE (contingent) -> HIGH CANDIDATE |

## Final Report

**HARNESS:** Existing `fv3_gpu_perf_bench` (`cuda/dyn_core/benchmarks/
run_small_perf_bench.cpp`), no source changes. Default path = single-
process, six-host-thread `PeerHaloTransport`. `--grid 512` supported with
no hardcoded limit. Config used: `--npz 10 --n_split 3 --warmup 5 --iters
30 --repeats 3 --mode resident --fuse-corner-fill --fuse-halo-edges`
(Graph off), R8, 1 GPU, 1 process/6 threads.

**REFERENCE (not re-derived, Prompt 36/37):** C256 Fortran 1250.75 ms/call,
CUDA (6-rank MPI) 1092.44 ms/call (0.874, CUDA faster). C512 Fortran
7155.63 ms/call, CUDA (6-rank MPI) 9089.05 ms/call (1.270, CUDA SLOWER).
Both resolutions BITWISE-correct on both backends.

**PEERHALO:** C256 median 148.9582 ms/call, C512 median 550.2646 ms/call
(3 reps each, <0.2% spread), rc=0, 0 FAIL/NaN/Inf, delp positive on all
6 tiles at both resolutions.

**COMPARISON:** PeerHalo growth C256->C512 = 3.69x. Fortran growth = 5.72x.
Six-rank MPI CUDA growth = 8.32x. PeerHalo scales *better* than Fortran
and far better than the six-rank MPI CUDA path — no disproportionate
cost growth in the shared device-side halo pack/unpack/staging code.

**ROOT CAUSE:** MPI TRANSPORT/PROTOCOL PATH (Case A — PeerHalo healthy,
six-rank MPI collapses).

**HALO BREAKDOWN:** NOT AVAILABLE (harness exposes only aggregate
elapsed_ms/ms_per_step; no pack/transport/unpack/sync sub-timers; no
instrumentation added).

**CUDA-AWARE MPI PRIORITY:** MODERATE (contingent) -> **HIGH CANDIDATE**.
Not implemented.

**NEXT EXPERIMENT (ONE, not implemented):** Rerun the existing six-rank
`MpiHaloTransport` C512 job under an alternative OpenMPI eager-limit MCA
setting (threshold raised above C512's ~135.4KB ordinary message size)
and compare `COMM_DSW`/`DYN_CORE` timing against the 9089.05 ms/call
baseline.

**PROJECT STATUS:** C512 Held-Suarez correctness remains BITWISE and
unaffected (Prompt 37, §9a). The C512 performance regression is now
isolated to the MPI transport/protocol layer specifically, not the
generic GPU halo mechanism. CUDA-aware MPI is now a HIGH CANDIDATE
remediation, contingent on the recommended MCA-threshold discriminator
confirming an eager/rendezvous crossover before any implementation work
begins. No source code changed in this prompt; no optimization attempted.

C512 MPI REGRESSION ISOLATED — READY FOR TARGETED MPI REMEDIATION EXPERIMENT
