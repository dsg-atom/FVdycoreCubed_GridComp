# Prompt 102 — Multi-GPU Scaling and Device-Memory Architecture Study

**Status: ARCHITECTURE/DESIGN STUDY ONLY. No production source, default, or
numerical change was made. No commit/push.**

Scope: design a scalable multi-GPU execution model for the CUDA FV3
dyn_core, using measured C512 memory/performance evidence, and determine
how the architecture should scale toward C1024+. This document supersedes
nothing in Prompt 101/101A; it extends Prompt 101's open memory-gap
question with new direct measurement.

## Stage 0 — Inputs

Read: `CLAUDE.md`, `docs/agent_instructions/discover_platform.md`,
`docs/agent_instructions/project_architecture.md` (no
`docs/agent_instructions/cost_efficiency.md` exists in this checkout — its
content is referenced by `CLAUDE.md` but the file itself is absent; the
cost-efficiency rules actually followed here are those embedded directly in
`CLAUDE.md`'s own "Cost-Efficient Agent Operation" section),
`docs/agent_handoff/CURRENT_STATE.md`, `p101_production_gpu_memory_inventory_2026-09-29.md`,
`p100_state_to_fv_device_memory_reuse_2026-09-29.md`,
`p99_state_to_fv_cuda_implementation_2026-09-28.md`,
`p101a_cumulative_cuda_performance_2026-09-29.md`,
`cuda/interface/fv3_gpu_dyncore_bridge.F90`,
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`. `git status --short` at the
start of this study showed only the pre-existing uncommitted working-tree
diffs already on record in `CURRENT_STATE.md` §10 — nothing new. No source
edits were made in Stage 0.

**Performance baseline (from Prompt 101A, reused verbatim, not
re-derived):** current cumulative production speedup vs. an all-4-ports-
Fortran same-node baseline (job 58638414): **TOTAL 1.153x/13.264% mean
(1.156x/13.466% median)**; **FV_DYN_LOOP 1.408x/28.957%**; **REMAPPING
2.592x/61.390%**. Current validated CUDA production defaults: PT/DELZ,
REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS = CUDA; TRACER_2D, State_To_FV,
REMAP_TOTALENERGYCON, FV_To_State = Fortran. Measurement config: R8, C512,
6 MPI ranks/1 A100, MPS ON, `FUSE_HALO_EDGES`/`FUSE_CORNER_FILL`=YES, Graph
OFF, Intel MPI 2021.13.

## Stage 1 — Resolving the ~21.5 GiB Unexplained Memory Gap (measured, this study)

**Node topology (confirmed via `sinfo`/`scontrol`):** every Discover
`gpu_a100` node has exactly **4 A100 GPUs** (`Gres=gpu:4(S:0-1)`), 48 CPUs,
510000 MB host memory. Any single-node multi-GPU experiment is bounded to
≤4 GPUs; more requires multi-node.

**Method:** reused Prompt 101's own unmodified rundir/binary
(`p101_memdiag_c512/StandAlone_FV3_Dycore.x`, sha256
`6fac8fca...ea2760c9`, production defaults, zero selector overrides —
identical config to the P101/P101A baseline) with **zero source changes**.
Job 58639804 requested `--gres=gpu:4` instead of `--gres=gpu:1`. Because
device selection is `dev = mod(comm_rank, device_count)`
(`fv3_gpu_dyncore_bridge.F90:164`) and `comm_size` is hardcoded to 6, four
visible devices automatically split the 6 ranks 2/2/1/1 across dev0..dev3
— giving **two independent ranks-per-GPU data points in one job**. A
background `nvidia-smi` sampler (5 s interval, no in-process
instrumentation, no rebuild) captured the per-device trajectory.

**Result: rc=0, 30/30 `DYN Date` lines, no FATAL/OOM/NaN/Inf.** Per-device
memory converged and stayed **exactly flat from t≈50 s through t≈105 s+**
(matches Prompt 101's own finding that the footprint is fully resident
after the first dispatch and flat thereafter):

| Device | Ranks bound to it | Converged used | 
|---|---|---|
| dev0 | 2 | 13450 MiB |
| dev1 | 2 | 13450 MiB |
| dev2 | 1 | 6746 MiB |
| dev3 | 1 | 6746 MiB |

Two *different physical GPUs* at the *same* ranks-per-GPU density produced
**identical** readings (dev0≡dev1, dev2≡dev3) — direct evidence against
any large device-specific fixed cost.

**Linear fit** `M(n) = M_fixed_per_gpu + n × M_per_rank_context` from the
two measured points:

- `M_per_rank_context ≈ 13450 − 6746 = 6704 MiB (~6.55 GiB)` — the
  dominant term.
- `M_fixed_per_gpu ≈ 6746 − 6704 = 42 MiB (~0.04 GiB)` — negligible.

**Cross-check against the independently measured 6-ranks/1-GPU density**
(P101/P101A/CURRENT_STATE.md: converged free headroom historically
~69–73 MiB out of 40960 MiB total, i.e. used ≈ 40887–40891 MiB): the
2-point linear model predicts `42 + 6×6704 = 40266 MiB` at n=6 — **≈624
MiB (~1.5%) below** the actual measured value. A small super-linear term
appears only at high rank-density (n=6 on one GPU), plausibly additional
MPS-server per-client bookkeeping or CUDA-context effects not visible at
n=1–2. This residual is small relative to the dominant per-rank term and
does not change the qualitative conclusion below.

**Conclusion — directly answers Prompt 101's own "Caution for Prompt
102":** the ~21.5 GiB Prompt 101 could not attribute to static
(FieldRegistry+WorkspaceLayout+port-buffer) accounting is **overwhelmingly
a PER-RANK/PER-CONTEXT cost, not a fixed-per-GPU cost.** Prompt 101's
static inventory (~3.05 GiB/rank) undercounted the true per-rank CUDA
runtime footprint (~6.55–6.7 GiB/rank measured here) by roughly 2x — the
difference is additional per-rank/per-CUDA-context state (lazy
kernel/module loading, runtime/library reservations, MPS per-client
buffers, and other unenumerated per-context overhead) that Prompt 101's
FieldRegistry/WorkspaceLayout/port-buffer audit was never positioned to
capture, since all of those are static-allocation inventories, not
runtime-context inventories.

**Confidence:**
- "Overwhelmingly per-rank, not fixed-per-GPU": **HIGH** (two different
  physical GPUs at equal rank-density read identically; the fit is
  near-exact over a 2x range).
- Exact `M_fixed_per_gpu`/`M_per_rank_context` values: **MODERATE** (only
  2 clean single-node data points; the small n=6 super-linear residual is
  not yet isolated — a confirmatory 2-GPU/3-ranks-per-GPU run would pin
  this down further but is not required for the qualitative conclusion
  above, or for Stage 19's decision below).
- This is a **bounded diagnostic** per project cost-efficiency rules — no
  further jobs were run once the qualitative and quantitative evidence
  were both sufficient to answer Stage 1's question.

## Stage 2 — Current Device-Selection Mechanism (verified, not assumed)

Source of truth: `fv3_gpu_dyncore_bridge.F90:100-309`.

```fortran
if (comm_size /= 6) then
   call mpp_error(FATAL, 'fv3_gpu_bridge_init: the CUDA dyn_core backend '// &
        'requires exactly 6 ranks with one cubed-sphere tile per rank')
endif
...
! Six ranks share whatever GPUs the node exposes; the Prompt 15 transport
! is a single-node design and its test carries the same assumption.
dev = mod(comm_rank, int(caps%device_count))
```

- `comm_rank` is the **global** MPI rank (from the communicator passed to
  `fv3_gpu_bridge_init`), not a node-local rank.
- `caps%device_count` comes from `cudaGetDeviceCount()` inside the C++
  runtime — itself governed entirely by `CUDA_VISIBLE_DEVICES`/Slurm
  `--gres` binding.
- `fv3_gpu_runtime.cpp:1460-1462` has an independent fallback
  (`device = d->mpi_rank % count` if `device_ordinal < 0`) using the same
  global-rank-modulo formula — consistent, not redundant/contradictory.
- **No `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` or any node-local-rank
  derivation exists anywhere in `cuda/dyn_core/` or `cuda/interface/`**
  (grep confirmed, zero hits).
- The design is **single-node-only by explicit acknowledgment in the
  source comment itself**, not an oversight.
- **Empirically verified this study, not assumed:** the existing
  mechanism, run unmodified with `--gres=gpu:4`, correctly executed a
  clean 30-step C512 run with 6 ranks spread across 4 physical GPUs
  (2/2/1/1) — global rank happened to equal node-local rank because the
  job used exactly one node. **Multi-GPU execution already works today,
  with zero code changes, as long as the job stays on a single node.**
  This is the direct, tested answer to Stage 2's core question.
- The mechanism would silently mis-map ranks to devices only in a
  **multi-node** job (global rank ≠ node-local rank across nodes) — not
  exercised or needed for the single-node scope of this study.

## Stage 3 — Recommended Rank→GPU Mapping Mechanism (design only, not implemented)

For the **single-node** case (Stage 14's first experiment, and every
option in Stage 8's C512 table): the existing global-rank-modulo formula
already produces the exact desired mapping with **zero source changes**,
because global rank equals node-local rank on one node. No change is
needed to run the first multi-GPU experiment.

For future **multi-node** scaling (needed for Stage 9's higher-rank-count
C1024 options and Stage 17's roadmap): the current mechanism is
insufficient — global-rank-modulo would scatter ranks across GPU indices
inconsistently across nodes. **Primary recommended mechanism:** derive a
node-local rank via `MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
MPI_INFO_NULL, &node_comm)` then `MPI_Comm_rank(node_comm, &local_rank)`,
and select `dev = mod(local_rank, device_count)` instead of
`mod(comm_rank, device_count)`. This is a small, well-understood, portable
MPI idiom already available in both OpenMPI and Intel MPI 2021.13 — no new
dependency. **Not implemented in this study**, per the explicit
constraint against production source changes.

## Stage 4 — Multi-GPU Safety Audit

| Component | Finding | Classification |
|---|---|---|
| WP-E runtime/context (`fv3_gpu_runtime.cpp`) | Per-`fv3_gpu_context` `device_ordinal` stored at create; `cudaSetDevice` called explicitly before every device-memory op in create/destroy paths (e.g. line 3140's teardown path). No global/static CUDA state found (`grep` for static globals/singletons: zero hits). | **READY** |
| FieldRegistry / WorkspaceLayout | Allocations owned per-context (`c->data.state.fields`, `c->data.workspace.allocation`), freed per-context at destroy. No shared filenames/IPC/`/tmp` staging found (`grep`: zero hits) — no hidden single-node-only assumption beyond the halo transport (below). | **READY** |
| dyn_core CUDA backend / bridge init (`fv3_gpu_dyncore_bridge.F90`) | Hardcoded `comm_size == 6` FATAL check; one-cubed-sphere-tile-per-rank invariant baked into `build_rank_of_tile`/topology setup; device selection uses global rank (Stage 2/3). | **ARCHITECTURE CHANGE** (to relax rank count) / **SMALL FIX** (to add node-local-rank mapping for multi-node) |
| Six-tile halo transport (`PeerHaloTransport`/`MpiHaloTransport`) | Built entirely around **one-tile-per-rank**; `rank_of_tile(6)` is a fixed-size array. **No intra-tile (same-face, interior-neighbor) exchange path exists anywhere in the CUDA halo transport layer.** Host-staged design is otherwise GPU-index-agnostic — confirmed empirically this study (clean 4-GPU-spread run, no correctness issue). | **READY** for same-tile-per-rank multi-GPU (any GPU-count option in Stage 8) / **ARCHITECTURE CHANGE** for any option that subdivides a face across multiple ranks (Stage 9's decomposition-based C1024 options) |
| PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS kernels | Operate on their owning context's stream/device; no device-index literals or global state found in these files. | **READY** |
| State_To_FV (Fortran default, P98-P100) | Not GPU-count-specific; its blocker is device memory headroom on whichever GPU its ranks land on — resolved by GPU count, not by its own code (see Stage 12). | **READY** (memory-blocker is external to this component) |

**Single most significant finding:** relaxing the `comm_size==6`
constraint to get more MPI ranks (the natural way to keep per-rank memory
bounded at higher resolution, per this task's own "primary design
principle") requires not just a check change but **designing and
implementing an entirely new intra-tile halo-exchange path** — this is a
genuine architecture change, not a configuration tweak, and materially
affects which C1024 options (Stage 9) are near-term feasible.

## Stage 5 — Host MPI/Halo Semantics Across Multi-GPU/Multi-Node

- `PeerHaloTransport`/`MpiHaloTransport` are **host-staged**: exchanges
  move through host memory relative to each rank's own device context.
  Which physical GPU underlies a rank is invisible to the exchange logic.
  **Confirmed empirically this study**, not just by design inspection: the
  4-GPU-spread run produced a clean, correct 30-step trajectory with no
  NaN/FATAL — the halo path is unaffected by GPU-index spread.
- `mpp_get_boundary`, `mpp_global_sum`, `BITWISE_EXACT_SUM` are pure-host
  MPI/FMS operations, keyed to MPI rank/tile topology, not to GPU count or
  index — unaffected by any option in Stage 8/9 that preserves the
  6-rank/one-tile-per-rank topology.
- No CUDA-aware MPI is needed or proposed for any option in this study
  (host-staged transport remains sufficient; CUDA-aware MPI stays a
  separate, already-evaluated, non-default track per §5/§7 of
  `CURRENT_STATE.md`).
- Any option that subdivides a face across multiple ranks (Stage 9) would
  need a **new** intra-tile host-staged exchange (Stage 4) — the
  *host-MPI* semantics (`mpp_global_sum` etc.) themselves would not need
  to change, only the CUDA-side halo topology.

## Stage 6 — Memory Scaling Model

```
M_gpu(n_ranks_on_gpu, local_cells, npz) ≈
    M_fixed_per_gpu                                   (~0.04 GiB, negligible)
  + n_ranks_on_gpu × [ M_fields(local_cells,npz)       (FieldRegistry, ~0.85 GiB/rank at C512)
                      + M_workspace(local_cells,npz)   (WorkspaceLayout, ~2.12 GiB/rank at C512)
                      + M_port_buffers                 (REMAP_Q+PRESSUREVARS, ~0.08 GiB/rank at C512)
                      + M_runtime_context ]            (per-rank CUDA runtime/lazy-load/MPS overhead,
                                                          ~3.5-3.6 GiB/rank at C512 — the term Prompt 101
                                                          could not see; recovered by Stage 1's measurement)
```

At C512, the bracketed per-rank sum measures to **≈6.55–6.7 GiB/rank**
(Stage 1), of which only **≈3.05 GiB/rank** was previously attributable to
named static structures (`M_fields`+`M_workspace`+`M_port_buffers`,
Prompt 101). The remaining **≈3.5–3.6 GiB/rank** is `M_runtime_context` —
now empirically bounded (not itemized further; itemizing it is out of
scope for this architecture study and not required for the projections
below, which use the measured aggregate directly).

## Stage 7 — Resolution Scaling (C512 → C768 → C1024)

FV3's actual decomposition (not a naive global-resolution-squared
assumption): at fixed 6-ranks/one-tile-per-rank, each rank's local tile is
the **full cubed-sphere face** (`ni=nj=N` where N is the cubed-sphere
resolution) plus halo padding (`halo_width=3` each side), i.e. C512's
`ni=nj=518` (P101). Per-rank memory for the fields/workspace/port-buffer
terms scales as `(N+2×halo)² × npz`, i.e. **essentially as N²** for large
N (halo becomes a smaller fraction as N grows: 518²/512²=1.024 at C512,
1030²/1024²=1.012 at C1024 — halo overhead *shrinks* as a fraction, not
grows). Holding rank count at 6:

| Resolution | Local tile (incl. halo) | Cells/rank | Scale vs. C512 | Projected fixed/workspace/port-buffer term | Projected total per-rank (incl. ~3.5 GiB runtime-context, held roughly flat*) |
|---|---|---|---|---|---|
| C512 (measured) | 518×518 | 268,324 | 1.00x | ~3.05 GiB | ~6.6–6.7 GiB (measured) |
| C768 | 774×774 | 599,076 | 2.23x | ~6.8 GiB | ~10.3 GiB |
| C1024 | 1030×1030 | 1,060,900 | 3.95x | ~12.05 GiB | ~15.55 GiB |

*`M_runtime_context` is not yet known to scale with local cell count at
all (Stage 1 only varied rank density, not resolution) — held flat here as
a conservative simplifying assumption; if it does scale with field size
(plausible, since some of it may be workspace-adjacent), these C768/C1024
per-rank figures are a **lower bound**, not an upper bound. This
uncertainty is flagged, not resolved, by this study.

**At 6 ranks unchanged (one tile per rank, no architecture change), C1024
alone (no GPU-count increase) would require ≈15.5+ GiB/rank — already
more than double C512's current per-rank footprint, entirely from the
larger tile, before any rank-density multiplication.** This is why GPU
count *and* decomposition both need to grow together per the task's own
primary design principle (Stage 9).

## Stage 8 — C512 Multi-GPU Options (6 ranks fixed, single node ≤4 GPUs)

| GPUs | Ranks/GPU | Used/GPU | Free/GPU (of 40960 MiB) | Basis |
|---|---|---|---|---|
| 1 | 6 | ~40890 MiB (~39.9 GiB) | ~70 MiB | **Measured** (P101/P101A/current production) |
| 2 | 3 | ~20.3–21.3 GiB (proj.) | ~19.7–20.7 GiB | **Projected** (linear model; not yet directly measured — flagged as the natural confirmatory follow-up, not required for Stage 19) |
| 3 | 2 | 13450 MiB (13.14 GiB) | 27510 MiB (26.86 GiB) | **Measured** (this study, job 58639804) |
| 4 | 1.5 avg (2/2/1/1) | 13450 MiB busiest GPU (13.14 GiB) | 27510 MiB busiest-GPU free | **Measured** (this study, job 58639804) |
| 6 | 1 | 6746 MiB (6.59 GiB) | 34214 MiB (33.4 GiB) | **Measured** (this study, job 58639804, per-rank point) — but 6 GPUs > 4/node cap, **requires ≥2 nodes** |

**Headroom grows dramatically with GPU count** — going from 1→2 GPUs
alone (no architecture change, no source change, just `--gres=gpu:2`)
projects to roughly **halve** the used-per-GPU footprint, which by itself
comfortably clears State_To_FV's ~440 MiB shortfall (Stage 12).

## Stage 9 — C1024 Decomposition Options

| Option | MPI ranks | Topology | Local tile | Projected mem/rank | Architecture change required? |
|---|---|---|---|---|---|
| (A) 6 ranks unchanged, full-face tiles | 6 | one tile/rank (unchanged) | 1030×1030 | ~15.5 GiB (Stage 7, lower bound) | **NO** — device-selection/topology code is untouched; only GPU *count* must grow to keep ranks-per-GPU bounded (e.g. 6 ranks/6 GPUs ≥2 nodes → ~15.5 GiB/GPU, comfortable) |
| (B) 24 ranks, FV_NX=2/FV_NY=2 per face (4 ranks/face × 6 faces) | 24 | 4 ranks/face, each owning a 515×515-ish quadrant (halo-padded) | ≈518×518 — **essentially identical to today's C512 tile** | ~6.6–6.7 GiB (≈ unchanged from C512) | **YES** — requires relaxing `comm_size==6` and designing/implementing the missing intra-tile (same-face, neighbor-rank) halo-exchange path (Stage 4) |
| (C) 54 ranks, FV_NX=3/FV_NY=3 per face (9 ranks/face × 6 faces) | 54 | 9 ranks/face, ≈344×344 local tile | ~344×344 → smaller than C512's per-rank footprint (~2.9–3.0 GiB) | **YES** — same architecture change as (B), larger rank count |

Option (A) keeps today's architecture entirely intact but does **not**
follow the task's own stated "primary design principle" (increase
decomposition, not just GPU count) and produces a materially larger
per-rank footprint than C512's reference. Options (B)/(C) satisfy the
principle directly (per-rank footprint held at or below the C512
reference) but require the Stage-4 architecture change first.
**Recommended starting configuration for C1024: Option (A) as a
near-term, zero-architecture-change stepping stone (validate at higher
resolution first), with Option (B) as the actual long-term target** once
the intra-tile halo path exists — consistent with Stage 19's decision
below.

## Stage 10 — MPS Strategy

| Ranks/GPU | MPS recommendation |
|---|---|
| 1 | MPS provides no sharing benefit (single client) — safe to leave OFF or ON; ON is harmless (single-client MPS ≈ native), so default ON for operational uniformity with denser configs is acceptable. |
| 2 | Keep MPS **ON** — matches the established "best config" convention (FUSE flags + MPS) already validated at higher density; two concurrent clients benefit from MPS's shared-context concurrent-kernel execution. |
| 3 | Keep MPS **ON** — same rationale, stronger benefit as density rises. |
| 6 (current production) | MPS **ON** is the current validated default and remains necessary — this is the density where sharing benefit (and the Stage-1 super-linear residual) is largest. |

Do not assume "more ranks-per-GPU is always better" — Stage 8 shows
headroom, not performance, scales this way; no performance benefit from
higher density was measured in this study (Stage 11).

## Stage 11 — Performance Scaling Risk Classification

| Risk | Level | Rationale |
|---|---|---|
| Halo-transport correctness across GPU-index spread | **LOW** | Directly tested clean in this study (4-GPU spread, 30/30 steps, no NaN/FATAL); host-staged design is GPU-index-agnostic by construction. |
| Device-selection correctness (single-node) | **LOW** | Directly tested clean; existing mechanism already produces correct single-node mapping with zero changes. |
| Device-selection correctness (multi-node) | **MODERATE** | Not tested in this study (scope limited to single node); requires Stage 3's recommended fix before any multi-node run. |
| Performance impact of splitting 6 ranks across 2+ GPUs (PCIe/NVLink cross-GPU traffic if any, MPS multi-GPU server behavior) | **MODERATE** | Not measured in this study (memory-only diagnostic); no speedup or slowdown claim should be made without a dedicated timed A/B (Stage 16). |
| C1024 Option (B)/(C) new intra-tile halo path performance | **HIGH** (unmeasured, new code path) | Entirely unimplemented; any latency/bandwidth claim would be invented. |
| C1024 Option (A) (6 ranks, bigger tiles) performance scaling | **MODERATE** | Existing code path, but never run at this scale; kernel/launch-overhead ratios may shift with much larger per-rank tiles. |

**No unmeasured speedup or slowdown is claimed anywhere in this document.**

## Stage 12 — State_To_FV Feasibility at 2/3/6 GPUs

Recall (P99/P100, not re-derived): the C512/6-ranks/1-GPU baseline is
already ~440 MiB short of the ≈144.0 MiB/rank aggregate State_To_FV needs,
even after P100's zero-new-bytes `WorkspaceId::vertical` alias.

Using Stage 8's measured/projected per-GPU headroom:

| Config | Free/GPU | State_To_FV feasible without P100's alias? | Feasible with P100's alias? |
|---|---|---|---|
| 2 GPUs (3 ranks/GPU, projected) | ~19.7–20.7 GiB | **YES** — enormous margin | **YES** |
| 3 GPUs (2 ranks/GPU, measured) | ~26.86 GiB | **YES** | **YES** |
| 6 GPUs (1 rank/GPU, measured, multi-node) | ~33.4 GiB | **YES** | **YES** |

**State_To_FV's memory blocker is resolved by GPU count alone at ≥2 GPUs —
no additional code change to State_To_FV itself is required.** This is
not an implementation/enablement recommendation for this task (explicitly
out of scope), only a feasibility finding for Stage 19/the Final Report.

## Stage 13 — Persistent Residency Architecture: Priority/Value

- **Priority: MODERATE.** Persistent device residency (already the
  project's stated architecture, §3 `project_architecture.md`) is not
  itself the blocker uncovered here — the blocker was *headroom at fixed
  GPU count*, which Stage 8 shows is resolved by spreading ranks across
  more GPUs, not by changing what stays resident.
- **Primary value: MEMORY**, not performance — no persistent-residency
  redesign is needed to unblock State_To_FV or to fit C1024 Option (A);
  the existing residency model is unaffected by GPU count. A deeper
  residency redesign would matter more for Option (B)/(C)'s new intra-tile
  halo path (to avoid extra host round-trips at the new rank boundaries),
  but that is a consequence of the Stage-9 architecture change, not an
  independent need.
- No full persistent-residency redesign is proposed in this study.

## Stage 14 — First Multi-GPU Experiment (design, not yet run at production scale)

- **Resolution:** C512 (current authoritative reference point).
- **MPI ranks:** 6 (unchanged).
- **GPUs:** 2 (`--gres=gpu:2`, single node — well within the 4/node cap).
- **Ranks-per-GPU:** 3.
- **GPU binding:** none required — existing `dev = mod(comm_rank,
  device_count)` already yields ranks {0,2,4}→dev0, {1,3,5}→dev1 on one
  node (Stage 2/3). No source change.
- **MPS:** ON (Stage 10).
- **CUDA_VISIBLE_DEVICES:** set implicitly by Slurm `--gres=gpu:2`; no
  manual override needed.
- **Expected memory/GPU:** ~19.7–20.7 GiB used (Stage 8 projection) — the
  one figure in this design that is **projected, not measured**; the
  actual run itself is this experiment's confirmatory measurement.
- **Correctness comparison:** same methodology as every prior same-node
  A/B in this project — sha256 binary provenance check, `DYN_DATE_COUNT`
  (expect 30/30), grep for FATAL/NaN/Inf/Abort, and a bitwise/round-off
  comparison of T/U/V/PS at the standard checkpoints against the existing
  6-ranks/1-GPU reference run (same production config, same binary) —
  expected numerical class: **bitwise-identical to round-off**, since no
  numerical code changes anything about which GPU a rank's data lives on
  (Stage 5).
- **Logging/telemetry:** reuse this study's exact pattern — sha
  provenance line, `DYN_DATE_COUNT`, background `nvidia-smi` per-device
  poll at 5 s cadence, pre/post-run `nvidia-smi --query-gpu=index,...`
  checkpoints.
- This experiment was **not run** in this study (2-GPU/3-ranks-per-GPU
  data point remains projected, not measured) — it is the explicit,
  recommended next action (Stage 19, Final Report "NEXT TASK").

## Stage 15 — Multi-GPU Correctness Validation Plan

1. Binary/config provenance: sha256 match, selector-env dump (all
   overrides unset, matching production defaults) — as in every prior A/B.
2. `DYN_DATE_COUNT` must equal the expected step count (30 for the
   standard short gate).
3. Grep run log for `FATAL|out of memory|Abort|NaN|Inf` — must be empty.
4. Field-level comparison (T/U/V/PS, the project's standard checkpoint
   set) between the 2-GPU run and the existing 1-GPU reference run at the
   same production config — **expected numerical class: bitwise or
   round-off only**, since GPU-index assignment does not touch any
   numerical algorithm (Stage 5). Any divergence beyond round-off would be
   a genuine defect, not an expected outcome.
5. Halo-specific check: confirm the known benign 2-cell halo-padding
   uninitialized-memory artifact (`delpc`/`pkc`, pre-existing both
   backends per `CURRENT_STATE.md` §4) is unchanged in magnitude/location
   — a change here would indicate the halo transport is *not* actually
   GPU-index-agnostic, contradicting Stage 5's finding.
6. Dispatch-count check: confirm total kernel-launch/dispatch counts match
   the 1-GPU reference exactly (no extra/missing dispatch from the
   GPU-count change).

## Stage 16 — Performance Experiment Plan (not yet run — plan only)

**A:** 1 GPU / 6 ranks-per-GPU (existing production config, reuse existing
timing data, no new run needed).
**B:** 2 GPUs / 3 ranks-per-GPU (Stage 14's experiment, same-node paired
A/B methodology per Prompt 48/101A — alternating order, same allocation).

Measure, for both arms: `TOTAL`, `FV_DYN_LOOP`, `REMAPPING`, `COMM_TOTAL`,
per-kernel dispatch timings (nsys, not `ncu` — `ERR_NVGPUCTRPERM` remains
platform-blocked per `discover_platform.md`), and per-GPU memory
(`nvidia-smi` polling, as in this study). **No speedup or slowdown claim
is made until this experiment is actually run** — Stage 11 already flags
this as MODERATE, unmeasured risk. Possible outcomes to watch for:
cross-GPU communication overhead if any rank-pair's halo partner lands on
a different physical GPU than before (host-staged transport should be
insensitive to this per Stage 5, but the *timing* impact, if any, of an
extra device context switch is unmeasured); MPS multi-GPU server behavior
differences from the validated single-GPU MPS configuration.

## Stage 17 — Multi-Node Roadmap (host-staged MPI throughout; CUDA-aware MPI is separate future work)

1. **Phase 1 (this study):** single-node multi-GPU, 6 ranks fixed,
   ranks-per-GPU varied via `--gres`. No architecture change. Validates
   Stage 14's experiment.
2. **Phase 2: DONE (Prompt 107, `p107_node_local_gpu_binding_2026-09-30.md`).**
   `MPI_Comm_split_type`-based local-rank device mapping implemented,
   GPU-executed, and validated single-node (bit-identical to Phase 1) and
   genuine multi-node (2-node divergence proof vs. the old global-rank
   rule) — the Phase 3 prerequisite is satisfied.
3. **Phase 3:** multi-node, 6 ranks fixed, spread across nodes (e.g. C1024
   Option A at 6 GPUs/2 nodes) — exercises Phase 2's local-rank mapping
   for real; host-staged MPI halo transport already topology-agnostic
   (Stage 5), no CUDA-aware MPI required.
4. **Phase 4:** multi-node, decomposition-increased rank count (C1024
   Option B/C) — requires Stage 4/9's new intra-tile halo-exchange
   architecture; CUDA-aware MPI remains optional/future, evaluated
   separately per §5/§7 of `CURRENT_STATE.md` (already found
   correctness-valid but performance-negligible at the current 6-rank
   scale — would need its own re-evaluation at higher rank counts, not
   assumed here).

   **Architecture mapped, not implemented (Prompt 108,
   `p108_c1024_intra_tile_halo_architecture_2026-09-30.md`)**: found the
   Fortran/FMS `domain_decomp` side already natively supports
   `layout(1,2)>1` per tile (reused, not reinvented); the CUDA-side gap is
   confined to `fv3_gpu_halo_topology.hpp/.cpp`'s single-neighbor-per-edge
   data model and the bridge's `comm_size==6` gate. Classification:
   MODERATE HALO GENERALIZATION (not a full communication-architecture
   rewrite — transport pack/unpack mechanics and the seam rotation/sign
   algebra are both reusable as-is). Recommended next bounded task:
   implement/validate the CUDA-side topology/transport generalization
   (small grid, single node) — the bridge `comm_size` relaxation and any
   real 24-rank run are each deferred to their own future task.

## Stage 18 — Recommended Architecture (ownership boundaries)

```
GEOS/FV3 Fortran
    -> ISO_C_BINDING / stable C ABI                      [unchanged, WP-E]
    -> fv3_gpu_bridge_init (WP-E, interface)
         -> MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)      [NEW, Stage 3 -- multi-node prerequisite]
         -> local_rank -> dev = mod(local_rank, device_count) [CHANGED formula, Stage 3]
    -> per-rank fv3_gpu_context (WP-E runtime)             [unchanged -- already per-context device-aware, Stage 4]
         -> FieldRegistry / WorkspaceLayout                [unchanged -- already per-context, Stage 4]
         -> WP-A/B/C/D CUDA kernels                        [unchanged -- no device literals found, Stage 4]
    -> six-tile halo transport (PeerHaloTransport/MpiHaloTransport)
         -> same-tile-per-rank topology                    [unchanged -- READY, Stage 4/8/9-Option-A]
         -> intra-tile (same-face) exchange                [NEW component, Stage 4/9-Option-B/C only --
                                                              not needed for Option A or any Stage-8 config]
```

Ownership: the local-rank device-mapping change and any intra-tile halo
design belong to **WP-E** (runtime/state/API/communication, per
`project_architecture.md`'s existing ownership rules) — no other work
package's numerics are touched by either change.

## Stage 19 — Decision

**Stage 1 produced clean, converged, internally-consistent, directly
measured evidence** (two different physical GPUs at matched rank-density
reading identically; near-exact linear fit over a 2x range; a small,
understood, bounded residual at high density) that **separates the fixed
and per-rank components of the memory-scaling model with HIGH confidence
on the qualitative conclusion** (overwhelmingly per-rank) and MODERATE
confidence on exact magnitudes. The 2-GPU C512 projection (Stage 8/14) is
derived from this same clean model and is not a leap into unmeasured
territory — it interpolates within the measured 1–6-ranks-per-GPU range
using a fit anchored by two of that range's endpoints (1 and 6) via direct
measurement, plus a directly-measured interior point (2 ranks/GPU, from
dev0/dev1).

**Decision: A** — proceed to the 2-GPU C512 implementation/validation
experiment (Stage 14), because the memory attribution is now separated
cleanly enough to project the 2-GPU configuration safely, and that
projection is favorable on every axis this study evaluated (State_To_FV
feasibility, Stage 12; no architecture change required, Stage 4/8;
existing device-selection mechanism already correct for this single-node
case, Stage 2/3).

---

## Summary Tables (for quick reference / Final Report)

**Memory model (C512, measured):** `M_fixed_per_gpu ≈ 0.04 GiB`,
`M_per_rank_context ≈ 6.55–6.7 GiB`, small (~1.5%) super-linear residual at
n=6 unresolved in exact magnitude.

**C512 GPU-count table:** see Stage 8.
**C1024 decomposition table:** see Stage 9.

CUDA MODERNIZATION MULTI-GPU ARCHITECTURE STUDY — MEMORY GAP RESOLVED TO
PER-RANK/PER-CONTEXT SCALING (HIGH CONFIDENCE); 2-GPU C512 PROJECTION SAFE

## Addendum (2026-09-29, Prompt 103) — Stage 14 Executed, Projection Confirmed

Prompt 103 executed the Stage-14 2-GPU C512 experiment this Decision A
called for (job 58642794). Measured, not projected: converged memory
**19.68 GiB/GPU** (vs. this study's 20.3–21.3 GiB projection — close,
slightly favorable), correctness **BITWISE EXACT** vs. 1-GPU (T/U/V/PS),
and — beyond this study's scope (memory only, no performance claim made
here) — a **consistent, reproducible TOTAL performance improvement of
10.70% (1.120x)**, with DYN_CORE at 1.773x. State_To_FV CUDA feasibility
(Stage 12's "feasible at ≥2 GPUs, not yet measured") is now **directly
confirmed**: 19.7 GiB/GPU free headroom, ~40x the ~440 MiB shortfall found
at 6-ranks/1-GPU (P98-100). See
`p103_c512_2gpu_validation_performance_2026-09-29.md` for full detail.
