# CUDA FV3 Performance Optimization Roadmap

Date: 2026-09-21
Branch: `optimization/cuda-fv3-performance`
Baseline tag: `cuda-perf-baseline-c512` (commit `33e10c5`)
Status: **PLANNING ONLY** — no source changes, no GPU jobs, no profiling runs
were executed to produce this document.

## 0. Framing

C512 CUDA dyn_core is currently **at performance parity** with Fortran
(ratio of medians 0.99998, same-node paired rerun, job 58468244 — see
Prompt 48). The historical "CUDA 1.27x slower" figure is superseded and
attributable entirely to node-to-node A100 variance (>2x spread observed
across three nodes for a nominally identical config), not a CUDA-specific
regression. Every optimization proposed below is a candidate for pushing
CUDA **ahead** of Fortran from parity, not for fixing a regression — which
sets a harder bar than the original problem statement implied, and the
confidence ratings in Part 6 reflect that.

Correctness of the C512 production first-run configuration (Gate C2, HS
tracks) remains the constraint on every change below: no optimization may
alter numerics, MPI topology, or the production `hord_mt=6 / hord_vt=6 /
hord_tm=6 / hord_dp=-6 / grid_type=0 / hydrostatic=false / a_imp=1.0`
configuration.

**Update (Prompt 53, measured finding — see
`docs/performance/p53_pt_delz_performance_neutrality_2026-09-22.md`):** the
`fv_mapz_mod` CUDA pt/delz remap dispatch cited below and in Prompt 52 as
progress toward closing the coverage gap is **not actually integrated** —
it crashes with an immediate `MPI_Abort` on every rank on its first real
invocation (`ak`/`bk` field-registration gap between
`fv3_gpu_field_registry.F90`, the registry path `fv3_gpu_bridge_init`
actually calls, and `fv3_gpu_interface_mod.F90`, a disconnected/dead
registration module that does register `ak`/`bk`). The previously-recorded
"Stage 11/13 GPU EXECUTED / NUMERICALLY VALIDATED" results for this
dispatch are invalid: both arms of both jobs ran a stale, pre-CUDA-
integration `install/bin` binary, so the dispatch was never actually
exercised. `fv_mapz_mod`'s 18.67%-of-window coverage gap (below) therefore
remains fully open — no fraction of it has actually been closed by CUDA
execution yet, contrary to what the Prompt 52C-series updates implied.

**Update (Prompt 67, measured finding — see
`docs/performance/p67_pt_delz_real_performance_profile_2026-09-23.md`):**
after Prompt 66 conditionally accepted the CUDA PT/DELZ port's numerical
residual, an attempted same-node C512 A/B to measure its actual
performance impact found the **full-CUDA PT/DELZ dispatch cannot
complete a single call at the authoritative 6-ranks/1-A100 production
density** — it OOMs (`cudaMalloc`-family failure) at call 1,
reproducibly across 4 runs on 2 nodes, including a solo run on a
completely idle 40GB A100. This is a genuine device-memory-footprint
defect in the port (not a savepoint-diagnostics artifact, as previously
believed — see `CURRENT_STATE.md` §7 correction), and it **blocks any
performance measurement of the ported `fv_mapz_mod`/PT-DELZ path at
production density** until the footprint issue is fixed. This
supersedes Prompt 53's status below only insofar as it adds a second,
more severe blocker (device-memory capacity, on top of Prompt 53's
now-resolved dispatch-crash issue) — see Part 10 item 6.

**Update (Prompt 51, superseding the priority ordering below — see
`docs/performance/p51_c512_host_residual_2026-09-21.md`):** a same-node
Nsight Systems CPU-sampling capture decomposed the `DYN_CORE` window
directly. Existing GPU kernel compute is only **3.9%** of the window; the
`update_dz_c` km+1-launch issue (Priority 1 below) and the two Riemann
kernels' occupancy (Priority 2 below) are therefore both bounded to a
small fraction of that 3.9%, not the dominant lever. The dominant cost is
**untranslated CPU FV3 Fortran work running alongside the CUDA-replaced
acoustic loop**: 64.66% of the window, of which `fv_mapz_mod` (vertical
Lagrangian-to-Eulerian remapping, not yet ported to CUDA) alone is
**18.67%** — the single largest individually-attributed component found
so far. This is a **CUDA coverage gap**, not a defect in the existing
GPU kernels or a correctness gap (Gate C2 remains closed). Halo-transport
blocking sync (Priority 4 below) remains real at 14.4% but was evaluated
and found not safely reducible without a redesign (Prompt 50). Kernel-level
Phase 2-4 work below is not deprecated — it is still valid, bounded upside
— but reaching the 2x target now requires closing CPU coverage gaps
first; `fv_mapz_mod` is the first candidate (Prompt 52 port plan:
`docs/performance/p52_fv_mapz_cuda_port_plan_2026-09-21.md`).

---

## Part 1 — Current Architecture Summary

### 1. Fortran -> ISO_C_BINDING -> CUDA runtime flow

`fv3_gpu_bridge_init` (`cuda/interface/fv3_gpu_dyncore_bridge.F90:111-294`)
calls `fv3_gpu_create` (`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1448`)
**exactly once per MPI rank**, guarded by a `bridge_ready` flag. This creates
2 non-blocking CUDA streams (`compute`, `transfer`) and 2 events, via the
CUDA primary context (`cudaSetDevice`, no explicit `cuCtxCreate`).
`fv3_gpu_initialize_state` (runtime.cpp:1487-1539) performs the **one and
only** per-field `cudaMalloc` + initial H2D copy, plus a single workspace
`cudaMalloc`. `fv3_gpu_finalize` (runtime.cpp:2417) syncs both streams and
frees everything at shutdown. The per-step hot path
(`fv3_gpu_dyn_core_step`, runtime.cpp:1702-2388) contains **no
`cudaMalloc`/`cudaFree`** — confirmed by explicit design comments
(`fv3_gpu_context.hpp:75,115`: "A `cudaMalloc` inside the acoustic loop is
an implicit sync"). One narrow exception: `heat_accum` grows lazily on
first use/size change, not per call.

### 2. dyn_core call structure

`fv3_gpu_dyn_core_step` (runtime.cpp:2035-2327) runs a `for(it=1..n_split)`
acoustic-substep loop, wired via `fv3_build_production_dispatch`
(`fv3_gpu_production_dispatch.cu:115-207`). Every stage is a direct,
line-by-line replay of `dyn_core.F90` — comments cite the exact Fortran
line numbers per stage.

### 3. Major CUDA kernel groups (per acoustic substep)

1. `cgrid_predictor` → `launch_cgrid_predictor_v3` (WP-A: d2a2c_vect +
   divergence_corner + c_sw)
2. `cgrid_vertical_pre_halo` → `launch_cgrid_vertical_pre_halo` (gz rebuild)
3. `cgrid_vertical_solver` → `launch_cgrid_vertical_solver` (gz<->zh sync,
   `update_dz_c`, `Riem_Solver_C`)
4. `cgrid_pressure` → `launch_cgrid_pressure_v3` (`p_grad_c`)
5. `dgrid_dynamics` → `launch_dgrid_dynamics_cuda` (WP-B full `d_sw`,
   S1-S38) — **only stage captured in a CUDA Graph** when
   `RUN_CUDA_FV3_GRAPH=YES`
6. heat_source accumulation kernel
7. `dgrid_height` → `launch_dgrid_height_stage` (`update_dz_d`)
8. `vertical_solver` → `launch_vertical_solver` (`Riem_Solver3`)
9. `nh_pressure` → `launch_nh_pressure_cuda` (`pk3_halo`/`pe_halo`,
   `gz=zh*grav`, `nh_p_grad`)
10. (final substep only) seam reconcile + `omega_finalize`

After the loop: `post_loop_heating` runs `nf_ke` times, host-looped, one
kernel launch per `del2_cubed` pass (mirrors Fortran's pass loop rather
than batching).

### 4. Vertical solver path

`update_dz_d` (`launch_update_dz_d`, `fv3_vertical_kernels.cu:128-210`)
delegates the horizontal-advection body to WP-B's
`dispatch.dgrid_height_transport`, then launches
`height_boundary_v2_kernel` for the monotonicity clamp — reading `zh` back
from global memory (real kernel-boundary round trip). `Riem_Solver3`
(`riem_solver3_kernel`, lines 283-454) is one thread per (i,j) column,
copies `delp/pt/q_con/cappa/zh/w` columns into **large per-thread local
arrays** (~10 arrays of 128/129 elements, lines 344-361) — no shared
workspace with `update_dz_d`. C-grid `update_dz_c`
(`cgrid::launch_update_dz_c_interface`/`_finalize`) is driven by a **host
loop over `k=1..km+1`** because it reuses a single `gz2` level buffer
(comment lines 907-912) — this is a genuine forced serialization, not a
data dependency, since all columns are independent. `Riem_Solver_C`
(`riem_solver_c_kernel<UseLocalScratch>`, `fv3_riem_solver_c.cu:40-192`)
uses a `local_arena[19*(128+1)]` per thread when no device workspace is
supplied.

All vertical kernels are strictly **per-column independent** (no
cross-thread reads) — the whole `update_dz_d -> Riem_Solver3` /
`update_dz_c -> Riem_Solver_C` chain is embarrassingly parallel across
columns and could in principle keep a column resident across stages.
Currently every stage round-trips through global memory instead.

### 5. Horizontal/dynamics path

WP-A (`c_sw`/predictor) and WP-B (`d_sw`, transport, `fv_tp_2d`, `ra_x`/
`ra_y`, `deln_flux`) are dispatched as coherent per-stage kernel groups
(not decomposed further here — Prompt 9/10/12 already validated kernel
fusion/tuning within WP-B). `dgrid_dynamics` is the single stage wrapped in
a CUDA Graph.

### 6. Halo communication path

`HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp:108-152`)
with two backends sharing all pack/unpack/topology/rotation/sign logic:
`PeerHaloTransport` (in-process, six host threads, used only in the
standalone perf-bench harness) and `MpiHaloTransport` (`host_staged` or
`cuda_aware`, six real MPI ranks, the production path). Pack/unpack kernels
(`pack_edge_kernel`/`unpack_edge_kernel`, plus fused 4-edge variants,
`runtime/fv3_gpu_halo_exchange.cu:82-196`) are separate launches from
interior compute, enqueued on the same compute stream. The documented
sequence is strictly serialized: `compute -> pack -> record pack_done ->
transfer -> unpack -> record xfer_done -> consumer` — **no
compute/communication overlap exists today**; the header explicitly defers
this ("Performance Work Comes Later").

### 7. Synchronization points

No `cudaDeviceSynchronize` exists anywhere in `cuda/dyn_core` or
`cuda/interface`. Only 2 non-blocking streams are used throughout
(`compute`, `transfer`). The one host-blocking synchronization on the
production hot path is `cudaEventSynchronize(ev_d2h_)`
(`fv3_gpu_halo_mpi.cpp:243`), issued **once per resolved halo group** in the
`host_staged` MPI transport — approximately 7-11 groups per substep
(DWIND_IN, W_IN, DIVGD, CWIND, DSCALAR, ZH, PKC2, plus the C-grid
`H_GZ_FIRST` exchange), matching the "~24 syncs/call" figure cited in
`prompt_26_rank_gpu_architecture.md`. `cuda_aware` transport skips the
D2H/H2D memcpys but not this synchronization point.

### 8. Persistent GPU state

Confirmed genuinely persistent: single `Workspace` allocation
(`fv3_gpu_state.hpp:15`), built once via `build_workspace_layout`, with
individual scratch consumers getting pointer-offset `WorkspaceSlice`s — no
per-call scratch allocation anywhere in the hot path.

### 9. Host/device transfers

One H2D burst at init (`fv3_gpu_initialize_state`). Per halo-group resolve
in `host_staged` mode: one D2H + one H2D per group per substep, sized to
`used_elements * sizeof(fv_real)`. `cuda_aware` mode skips both memcpys,
letting Intel MPI read/write device pointers directly. Explicit
`fv3_gpu_sync_to_host`/`sync_from_host` ABI calls exist for boundaries
(diagnostics, physics) but are not on the acoustic-substep hot path.

### 10. MPI/MPS interaction

Rank→GPU mapping: `dev = mod(comm_rank, device_count)`
(`fv3_gpu_dyncore_bridge.F90:156`), production requires exactly 6 ranks (1
tile/rank). MPS is **operational-only** — a shell lifecycle script
(`fv3_mps_lifecycle.sh`, outside this repo) that sets
`CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` and starts
`nvidia-cuda-mps-control`; nothing in `cuda/` source is MPS-aware. Without
MPS, 6 unshared CUDA contexts on one GPU cost ~2.6-3.9x vs. 1 context
(Prompt 26) — MPS is already load-bearing for current performance but is
not yet the default in every standing job script (operational cleanup
item, `CURRENT_STATE.md` §7).

### Where CUDA still mirrors Fortran decomposition too literally

- The acoustic substep is a hand-transcribed, line-by-line replay of
  `dyn_core.F90` — one dispatch-table call per Fortran subroutine rather
  than a cross-stage fused design.
- `update_dz_c` replays Fortran's per-level loop as `km+1` **separate
  kernel launches** instead of one kernel that loops over levels
  internally.
- `post_loop_heating` loops `nf_ke` times host-side, one launch per pass.
- Only 1 of ~10 per-substep stages is CUDA-Graph-eligible because the rest
  are interleaved with halo resolves; the graph boundary is drawn at the
  Fortran subroutine boundary, not at the true data-dependency boundary.

---

## Part 2 — Performance Hypotheses (ranked, highest to lowest probability)

| # | Hypothesis | Why it may matter here | Evidence available | Evidence missing | How to test | Expected upside |
|---|---|---|---|---|---|---|
| 1 | Host-blocking halo synchronization (`cudaEventSynchronize` per group, ~7-11×/substep) | Code-confirmed real; each is a host-side stall that cannot overlap with any GPU work, multiplied by n_split substeps | Exact call site + count (Agent B); prompt_26's "~24 syncs/call" figure | Actual wall-time cost per sync in a live nsys trace; whether these dominate vs. are cheap/short | Nsight Systems timeline, count+duration of `cudaEventSynchronize` regions during one dyn_core call | MEDIUM-HIGH if syncs are individually long; unknown until measured |
| 2 | `update_dz_c` forced into km+1 sequential kernel launches | Code-confirmed serialization bug, not a data dependency (columns are independent) | Source + comment explaining the `gz2` buffer reuse constraint (Agent C) | Actual launch-overhead cost at production `km` (~91 levels) | Nsight Systems: count+total duration of the per-level launch sequence; compare to a single-kernel prototype (measurement only, no source change yet) | MEDIUM — bounded, well-understood, cheap to prototype |
| 3 | Register pressure / local-memory spill in `riem_solver3_kernel`/`riem_solver_c_kernel` | Both kernels carry ~10 arrays of 128/129 elements per thread, no `__shared__` anywhere in `cuda/` | Source-confirmed large per-thread local arrays (Agent C) | Actual occupancy/registers-per-thread/local-memory-traffic numbers | Nsight Compute on these two kernels: achieved occupancy, registers/thread, local memory throughput | MEDIUM-HIGH if occupancy is low; these are two of the most complex per-column kernels |
| 4 | Global-memory round trips between vertical stages (`update_dz_d`→`Riem_Solver3`, `update_dz_c`→`Riem_Solver_C`) | Columns are independent and could stay resident; currently every stage re-reads/rewrites `zh`/`delz`/`gz` from global memory | Source-confirmed data flow (Agent C) | Actual bandwidth cost vs. compute cost per kernel (arithmetic intensity) | Nsight Compute DRAM throughput + L2 hit rate on the 4 kernels; only pursue fusion if bandwidth-bound | MEDIUM — real but must be sequenced after occupancy work (fusing high-register kernels can make register pressure worse) |
| 5 | Narrow CUDA Graph coverage (only `dgrid_dynamics`) / per-kernel launch overhead across ~10 stages + 7-11 halo resolves | Each un-graphed launch pays full CPU dispatch latency; with n_split substeps this multiplies | Source-confirmed graph boundary (Agent B) | Measured CPU-side launch overhead vs. GPU busy time | Nsight Systems: CPU-side API timeline gaps between kernel launches | LOW-MEDIUM — likely smaller than #1/#2 on a modern host, but cheap to quantify |
| 6 | Halo pack/unpack serialized with interior compute (no comm/compute overlap) | Explicitly deferred by design comment; real architectural gap | Source-confirmed serialized sequence (Agent C) | Whether halo transport is actually a large fraction of wall time *once measured correctly* (see caveat below) | Nsight Systems: time in pack/transfer/unpack vs. interior kernels, same-node paired run | UNKNOWN — do not assume large; see caveat |
| 7 | MPI/rank-count/GPU-density effects | Already directly tested | Prompt 46 (negligible cuda_aware benefit), Prompt 47 (density refuted, <0.5% spread) | Nothing further needed at current scale | N/A — closed | LOW — do not re-investigate without new evidence (per `CURRENT_STATE.md` §8) |
| 8 | Branch divergence | No evidence of significant per-column branching found in exploration | None specific found | Actual warp branch-efficiency numbers | Nsight Compute branch efficiency on top kernels (collect as part of Part 3 Phase B, not a dedicated campaign) | LOW — no code-level indicator yet |
| 9 | Memory coalescing / layout | Layout preserves Fortran column-major, i is stride-1 and threads map to i — coalesced by construction for the kernels inspected | Source-confirmed stride/thread mapping (Agent C) | Whether `pe`/`peln`'s `(i,k,j)` layout (k in the middle stride) causes any issue for kernels that index by k | Nsight Compute memory-efficiency metric on any kernel touching `pe`/`peln` | LOW — likely already fine, worth one targeted check, not a redesign |
| 10 | Expensive reductions | No specific reduction-heavy kernel identified in exploration | None found | N/A until profiling surfaces a candidate | Revisit only if Nsight Systems/Compute flags a reduction kernel | LOW — no current evidence |
| 11 | Duplicate/redundant intermediate buffers beyond the vertical-stage round trips already listed (#4) | Workspace is a single persistent allocation reused via slices — unlikely to have gross duplication | Agent B confirms single workspace, no per-call allocation | Whether any WorkspaceSlice is written and never effectively reused | Low priority; would need a workspace-usage audit, not profiling | LOW |

**Superseding evidence-quality update (Prompt 51)**: hypotheses 1-5 above
are all bounded by the measured 3.9%-of-window existing-GPU-kernel-compute
figure and the (separately evaluated, not-safely-reducible per Prompt 50)
14.4% halo-sync figure — together they cap the *combined* upside of every
hypothesis in this table at roughly 18-20% of the window, not the 64.66%
occupied by untranslated CPU Fortran work. The dominant, newly-measured
hypothesis is **#0 — CUDA coverage gap**: large portions of the production
timestep (`fv_mapz_mod` vertical remap, 18.67%; `fv_tracer2d_mod`/state
conversion/HS-physics/transcendental math, ~46.0%) still run on the host
in Fortran, alongside rather than inside the CUDA-replaced acoustic-substep
loop. This does not invalidate hypotheses 1-5 (they remain individually
correct, bounded findings) but reorders priority: closing coverage gaps
(starting with `fv_mapz_mod`, Prompt 52) now dominates the path to 2x over
any remaining kernel-level tuning in this table. See
`docs/performance/p51_c512_host_residual_2026-09-21.md` and
`docs/performance/p52_fv_mapz_cuda_port_plan_2026-09-21.md`.

**Critical evidence-quality caveat**: Prompt 37's FMS clock-block
attribution (`COMM_DSW` cited as "97.6% of the CUDA-vs-Fortran delta") was
collected on the same `warpa009` run later shown (Prompt 48) to be a
node-variance outlier. Because that attribution is an internal breakdown of
an already-confounded absolute delta, it **must not be treated as
established evidence** of where CUDA time actually goes. It is downgraded
from "evidence available" to "hypothesis only" everywhere it would
otherwise support hypothesis #1/#6, and re-collecting it correctly (Part 3,
Part 10 item 5) is one of the first actions.

---

## Part 3 — Profiling Plan

### Phase A — Nsight Systems

**Goal**: identify the critical path and account for >=90% of dyn_core wall
time, under a methodology immune to the node-variance confound that
invalidated the original Prompt 37 attribution.

- **Run size / configuration**: C512, production first-run config
  (`hord_mt=6/hord_vt=6/hord_tm=6/hord_dp=-6/grid_type=0/
  hydrostatic=false/a_imp=1.0`), `host_staged` MPI transport (current
  default), MPS ON, fusion flags ON (matching the current best-validated
  config in `CURRENT_STATE.md` §5), `RUN_HELD_SUAREZ=1`, `COLDSTART=1`.
- **Same-node requirement**: single Slurm allocation, single node — do not
  compare across jobs/nodes (Prompt 48 lesson). If a Fortran-vs-CUDA
  comparison is wanted alongside the CUDA-only profile, run both in the
  same allocation, alternating order, as in Prompt 48.
- **Number of dyn_core calls to capture**: 5-10 calls after a short
  untraced warmup (1-2 calls) to exclude first-call lazy allocation
  (`heat_accum`) and cold caches/JIT effects.
- **Ranks to profile**: full `nsys` timeline on rank 0 (representative
  interior tile) for kernel/API-level detail; lightweight
  (`--trace=nvtx,cuda` only, no full CPU sampling) timeline on all 6 ranks
  simultaneously to catch any single outlier rank/tile (e.g. a
  polar-tile-specific halo cost) that a single-rank profile would miss.
- **Bounding profiler overhead**: limit capture to the 5-10-call window via
  `nsys start`/`stop` triggers or `--capture-range=cudaProfilerApi` around
  the target dyn_core calls, rather than profiling full model startup/
  physics/I-O.
- **Metrics/events to extract**: per-kernel wall time and count, CUDA API
  timeline (especially `cudaEventSynchronize`/`cudaStreamSynchronize`
  regions and their durations), MPI activity if NVTX-annotated, gaps
  between consecutive kernel launches (CPU-side dispatch overhead).
- **Required output format**:

  | Rank | Component | Time | % total |
  |------|-----------|------|---------|
  | 0 | (top-level stages: cgrid_predictor, cgrid_vertical, dgrid_dynamics, dgrid_height, vertical_solver, nh_pressure, halo pack/xfer/unpack, sync waits) | ms | % |

  plus a **top 10 kernels/API waits by wall time** table (kernel or API
  name, total ms across the capture window, call count, avg ms/call).

### Phase B — Nsight Compute

Only after Phase A identifies the dominant kernels/regions. Based on the
architecture review, the most likely top-3-5 candidates to prioritize are
`riem_solver3_kernel`, `riem_solver_c_kernel`, the per-level `update_dz_c`
kernel, the `dgrid_dynamics`/`d_sw` graph node, and the halo pack/unpack
kernels — but Phase A's actual ranking governs, not this prior list.

For each of the top 3-5 kernels collect:

- kernel duration
- achieved occupancy
- registers/thread
- shared memory/block (expected near-zero given no `__shared__` usage
  found — confirms whether that absence is actually costing occupancy)
- SM utilization
- DRAM throughput
- L1/L2 hit rate and behavior
- memory efficiency / coalescing (with specific attention to any kernel
  touching `pe`/`peln`'s `(i,k,j)` layout)
- warp stall reasons
- branch efficiency
- instruction mix
- eligible warps/cycle
- achieved compute throughput (vs. roofline, to establish compute-bound vs.
  memory-bound vs. latency-bound for each candidate)

Do **not** profile the whole application with Nsight Compute — targeted
kernel replay only, on the same 5-10-call same-node window used in Phase A.

---

## Part 4 — Optimization Opportunity Map

| Priority | Area | Proposed change | Expected gain | Risk | Effort | Validation |
|---|---|---|---|---|---|---|
| 1 | C-grid `update_dz_c` launch loop | Replace `km+1` sequential kernel launches with one kernel that loops over levels internally (per-column parallel, level loop in-kernel using double-buffered or in-register level state instead of the shared `gz2` aliasing pattern) | MEDIUM | LOW — purely a launch-structure change, same per-level math | MEDIUM | ctest + C512 sampled-field bitwise/round-off compare vs. `cuda-perf-baseline-c512`, same-node A/B |
| 2 | `riem_solver3_kernel` / `riem_solver_c_kernel` occupancy | Reduce per-thread local-array footprint (recompute vs. store, or move to `__shared__` per block if column count per block allows) — **only after Nsight Compute confirms occupancy is actually depressed** | MEDIUM-HIGH (if confirmed) | MEDIUM — touches core Riemann-solver code, requires careful numerical equivalence proof | HIGH | Targeted kernel unit test + C512 dyn_core compare + Nsight Compute re-check of occupancy delta |
| 3 | Vertical-stage fusion (`update_dz_d`→`Riem_Solver3`, `update_dz_c`→`Riem_Solver_C`) | Fuse adjacent per-column kernels to eliminate the `zh`/`delz`/`gz` global-memory round trip — **only after Part 3 confirms these are bandwidth-bound, not latency/occupancy-bound** (fusing two high-register kernels can worsen #2) | MEDIUM | MEDIUM-HIGH — largest semantic-preservation burden of the kernel-level changes | HIGH | Same as #2, plus explicit register-pressure re-check post-fusion |
| 4 | Halo-group host-blocking sync reduction | Investigate whether the ~7-11 `cudaEventSynchronize` calls/substep can be reduced (batch groups, or replace host-blocking wait with a stream-ordered dependency where the consumer kernel is launched on the same stream after `cudaStreamWaitEvent` instead of blocking the host) | MEDIUM (if sync cost is confirmed material) | MEDIUM — touches the halo transport critical path shared by both MPI modes | MEDIUM | Nsight Systems before/after sync-count comparison + full correctness suite (bitwise) |
| 5 | Widen CUDA Graph coverage beyond `dgrid_dynamics` | Extend graph capture to adjacent halo-free stages (e.g. `cgrid_predictor`+`cgrid_vertical_pre_halo` if provably halo-free within the capture) | LOW-MEDIUM | MEDIUM — CUDA Graph capture-safety constraints are easy to violate silently | MEDIUM | Same-node A/B + graph capture-safety review + full correctness suite |
| 6 | Communication/computation overlap (pack boundary + compute interior concurrently) | See Part 4-F — architectural change, gated on profiling evidence | UNKNOWN | HIGH | HIGH | Full correctness suite + same-node A/B; do not attempt before Phase 1 evidence |
| 7 | Process/GPU architecture change (fewer MPI ranks/contexts, more tiles/process) | See Part 4-G — later-phase option only | UNKNOWN | HIGH | VERY HIGH | Full re-validation of Gate C2 at the new topology |

### A. Vertical kernel fusion

Candidates: `update_dz_d` → `Riem_Solver3` (both read/write `zh`); C-grid
`update_dz_c` → `Riem_Solver_C` (both read/write `gz`/`delz`). All four
kernels are per-column independent (Agent C), so fusion is *data-flow*
legal. **What must first be proven before fusing**:

1. Nsight Compute must show these kernels are DRAM-bandwidth-bound (not
   occupancy- or latency-bound) — otherwise fusion just concentrates more
   registers/local-memory pressure into one kernel and can make things
   worse (these kernels already have large per-thread local arrays).
2. The `gz2`-buffer-reuse constraint behind `update_dz_c`'s km+1 launches
   (Priority 1) must be resolved *before* attempting to fuse it with
   `Riem_Solver_C`, since fusing across an already-broken launch structure
   compounds risk.
3. A per-kernel register budget check (Nsight Compute registers/thread)
   for the *fused* prototype, not just the two original kernels
   separately.

Do not fuse blindly — this is Priority 3, after Priority 1-2.

### B. Memory layout / coalescing

CUDA threads map `threadIdx.x` → `i` (fastest/logical dimension 1); field
strides are supplied by the Fortran caller and preserved as Fortran
column-major (`stride_elements(1)=1`), so `i`-indexed thread access is
coalesced by construction for every kernel inspected. The one flagged
exception is `pe`/`peln`, stored `(i,k,j)` rather than `(i,j,k)` — `i` is
still stride-1 there, so `threadIdx.x=i` access remains coalesced; this is
**not** currently believed to be a real coalescing problem, but should get
one targeted Nsight Compute memory-efficiency check (Part 3 Phase B) to
close the question rather than assume it. No layout/indexing redesign is
recommended at this time — the numerical risk of touching field storage
order (used across the whole ABI/field-registry boundary) far outweighs
the currently-unconfirmed upside.

### C. Reduce global memory traffic

Concrete write-then-immediately-read-once candidates identified (Agent C):
`zh` (written by WP-B's `dgrid_height_transport`, read once by
`height_boundary_v2_kernel` in the same `launch_update_dz_d` call); the
`gz2`/`xfx`/`yfx`/`fx`/`fy` scratch arena reused per level in `update_dz_c`
(fusion target once the launch-loop issue is fixed); `gz` written by
`gz_from_delz_kernel`, consumed only by the halo exchange +
`gz_zh_sync_kernel`. These map directly onto Priority 1 and Priority 3
above — no additional untracked duplication was found in the persistent
workspace (Agent B confirms single-allocation, slice-based reuse).

### D. Occupancy / register pressure

`riem_solver3_kernel` and `riem_solver_c_kernel` are the strongest
candidates: ~10 arrays of 128/129 elements per thread and a
`local_arena[19*129]` per thread respectively, with **zero `__shared__` or
`__launch_bounds__` usage anywhere in `cuda/`**. Recommended approach:
measure first (Nsight Compute registers/thread, achieved occupancy, local
memory traffic), then experiment with block-geometry changes (columns per
block) and/or selective `__launch_bounds__` only if occupancy is
confirmed low **and** a specific target occupancy is justified by the
roofline data — not occupancy-tuning for its own sake, per the task's
explicit caution.

### E. Synchronization

| Sync | Classification |
|---|---|
| `cudaEventSynchronize(ev_d2h_)` in `MpiHaloTransport` (per halo group, ~7-11×/substep) | LIKELY REMOVABLE / POTENTIALLY OVERLAPPABLE — investigate replacing host-blocking wait with stream-ordered `cudaStreamWaitEvent` where the consumer can be enqueued without a host stall; requires care since MPI progress still needs a host-visible completion signal in `host_staged` mode |
| `cudaStreamSynchronize` at `fv3_gpu_runtime.cpp:243-244,1366,1543,2403-2404,2419` | REQUIRED — debug/registry, sync-to-host, halo teardown, and finalize paths; not on the acoustic hot path or only at explicit boundaries |
| `cudaEventRecord`/`cudaStreamWaitEvent` (`fv3_gpu_runtime.cpp:63-68,850,872,919-940`) | REQUIRED but not host-blocking — stream-to-stream ordering only, already non-blocking |
| Halo pack → transfer → unpack serialization (no dedicated sync call, but structurally serial) | POTENTIALLY OVERLAPPABLE — see Part 4-F |
| No `cudaDeviceSynchronize` anywhere | N/A — confirmed absent, nothing to remove |

### F. Communication overlap

Feasibility: architecturally possible in principle (columns/tiles are
independent, interior compute for the *next* stage does not need
boundary-halo data for interior points), but the current `HaloTransport`
design explicitly defers this and shares no overlap infrastructure between
`PeerHaloTransport`/`MpiHaloTransport`. Minimum architectural change
required: split each stage's kernel into "interior" and "boundary-adjacent"
launches, pack boundary data as soon as it's ready (not after the full
stage completes), and begin MPI transport while interior-only work for the
*next* stage proceeds on the compute stream. **Do not make this Phase 1** —
per the task's explicit gating and because Part 2's hypothesis #6 currently
has no reliable evidence (the only prior attribution pointing at
communication cost, Prompt 37's FMS breakdown, is confounded). Revisit only
if Part 3's Nsight Systems data shows halo pack/transfer/unpack accounting
for a material fraction of wall time under the corrected same-node
methodology.

### G. Process / GPU architecture

Current: 6 MPI ranks, 6 CUDA contexts sharing 1 A100 via MPS. Already
directly tested: rank/GPU density (1/2/3 GPUs at fixed 6 ranks) has **no
effect** on C512 timing (Prompt 47, <0.5% spread) — this rules out "more
GPUs, same rank count" as a lever. The *unexplored* variant is fewer
**contexts** (e.g. 1 process per GPU running all its tiles via host
threads, à la `PeerHaloTransport`'s in-process model, which showed
healthier scaling in the isolated Prompt 39 discriminator — though that
comparison used a different harness/methodology and is not directly
comparable in absolute terms). This is explicitly a **later-phase
architecture option only** (Phase 6) — do not pursue before kernel-level
Priority 1-4 work is exhausted and profiling shows single-GPU efficiency
is not the limiter.

---

## Part 5 — Roadmap Phases

### PHASE 0 — Reproducible profiling baseline
- **Goal**: establish a same-node, repeatable C512 nsys capture methodology immune to the node-variance confound.
- **Actions**: write/adapt a Slurm sbatch template (reuse Prompt 48's same-node alternating-run pattern) that wraps `nsys profile` around a 5-10-call, warmup-excluded dyn_core window.
- **Files likely involved**: new sbatch script under repo root or `tools/`; no source files.
- **Profiler/tool required**: Nsight Systems (`nsys`).
- **Expected performance range**: no performance change (measurement only).
- **Correctness gate**: N/A (no source change).
- **Stop criteria**: a single `.nsys-rep` accounting for >=90% of dyn_core wall time is captured and reviewable.
- **Rollback criteria**: N/A.
- **Git checkpoint**: none required (tooling/script only; commit only if the harness is judged reusable).

### PHASE 1 — Critical-path characterization
- **Goal**: rank the Part 2 hypotheses by measured evidence; identify top 3-5 kernels/regions.
- **Actions**: run Phase 0's capture; produce the Part 3 Phase A tables; re-run Prompt 37's FMS block attribution under the corrected same-node methodology (Part 10 item 5) to get a trustworthy comm-vs-compute split.
- **Files likely involved**: none (analysis only).
- **Profiler/tool required**: Nsight Systems.
- **Expected performance range**: no change.
- **Correctness gate**: N/A.
- **Stop criteria**: top kernels/regions identified, hypotheses 1-6 from Part 2 confirmed/refuted with real data.
- **Rollback criteria**: N/A.
- **Git checkpoint**: commit the profiling report/tables under `docs/performance/`.

### PHASE 2 — Top kernel optimization
- **Goal**: address occupancy/register-pressure findings on the top-ranked kernels (likely `riem_solver3_kernel`/`riem_solver_c_kernel`, pending Phase 1).
- **Actions**: Nsight Compute deep-dive (Part 3 Phase B); targeted register/occupancy fixes.
- **Files likely involved**: `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`, `fv3_riem_solver_c.cu`.
- **Profiler/tool required**: Nsight Compute (`ncu`).
- **Expected performance range**: MEDIUM confidence, magnitude TBD by Phase 1/2 data.
- **Correctness gate**: targeted kernel unit test + C512 sampled-field compare vs. `cuda-perf-baseline-c512`.
- **Stop criteria**: measured occupancy improvement with no numerical regression, or evidence the kernel is not occupancy-bound (stop pursuing this path).
- **Rollback criteria**: any bitwise/round-off tolerance violation, or a performance regression on same-node A/B.
- **Git checkpoint**: one commit per kernel change.

### PHASE 3 — Kernel fusion / memory-traffic reduction
- **Goal**: fix `update_dz_c`'s km+1 launch loop; evaluate vertical-stage fusion if Phase 1/2 shows bandwidth-bound behavior.
- **Actions**: single-kernel level-loop replacement for `update_dz_c`; fusion prototypes for `update_dz_d`→`Riem_Solver3` gated on Part 4-A's proof requirements.
- **Files likely involved**: `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`, `fv3_cgrid_dz.hpp`.
- **Profiler/tool required**: Nsight Systems (launch-count/overhead re-check), Nsight Compute (register re-check post-fusion).
- **Expected performance range**: MEDIUM confidence for the launch-loop fix; LOW-MEDIUM for fusion.
- **Correctness gate**: full ctest suite + C512 dyn_core sampled-field bitwise/round-off compare.
- **Stop criteria**: launch-loop fix validated and merged; fusion abandoned if register pressure worsens post-fusion (per Part 4-A gate).
- **Rollback criteria**: any correctness regression; any same-node A/B performance regression.
- **Git checkpoint**: separate commits for launch-loop fix and any fusion attempt; sub-branch for fusion experiments (Part 8).

### PHASE 4 — Synchronization reduction
- **Goal**: reduce host-blocking `cudaEventSynchronize` cost in the halo transport if Phase 1 confirms material cost; widen CUDA Graph coverage where capture-safe.
- **Actions**: prototype stream-ordered halo completion signaling; extend graph capture boundary if a halo-free stage sequence is provably capture-safe.
- **Files likely involved**: `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp`, `fv3_gpu_runtime.cpp`.
- **Profiler/tool required**: Nsight Systems.
- **Expected performance range**: LOW-MEDIUM confidence.
- **Correctness gate**: full correctness suite (bitwise) — synchronization changes are high-risk for subtle race conditions.
- **Stop criteria**: measured sync-time reduction with 10/10 correctness PASS.
- **Rollback criteria**: any nondeterminism or correctness failure (even intermittent).
- **Git checkpoint**: one commit per discrete sync change; extensive same-node repeat-run validation before merge (nondeterminism risk).

### PHASE 5 — Communication/computation overlap
- **Goal**: only if Phase 1 shows halo pack/transfer/unpack on the critical path — implement pack-interior-overlap per Part 4-F.
- **Actions**: split interior/boundary kernel launches; begin transport before full stage completion.
- **Files likely involved**: `cuda/dyn_core/include/fv3_gpu_halo_transport.hpp` and both backend implementations.
- **Profiler/tool required**: Nsight Systems.
- **Expected performance range**: UNKNOWN — do not estimate before Phase 1 evidence.
- **Correctness gate**: full correctness suite; this is the highest-risk phase for subtle race conditions given the shared-topology halo code.
- **Stop criteria**: explicitly gated — do not start this phase unless Phase 1 shows communication on the critical path.
- **Rollback criteria**: any correctness or determinism regression.
- **Git checkpoint**: dedicated sub-branch (Part 8), not direct commits to `optimization/cuda-fv3-performance`.

### PHASE 6 — Multi-GPU / process architecture
- **Goal**: only if Phase 2-5 kernel/sync-level work is insufficient to reach the 2x target — evaluate fewer-CUDA-contexts / more-tiles-per-process architecture.
- **Actions**: see Part 9 prerequisites before starting.
- **Files likely involved**: `cuda/interface/fv3_gpu_dyncore_bridge.F90` (rank/GPU mapping), process launch/job scripts.
- **Profiler/tool required**: Nsight Systems, same-node controlled multi-GPU A/B design (to avoid the Prompt 37/47 node-variance confound).
- **Expected performance range**: UNKNOWN.
- **Correctness gate**: full Gate-C2-equivalent re-validation at the new topology.
- **Stop criteria**: explicitly gated on Phase 2-5 insufficiency.
- **Rollback criteria**: any correctness regression, or same-node A/B showing no benefit (as Prompt 47 already found for the density-only variant).
- **Git checkpoint**: dedicated sub-branch; do not merge without full re-validation.

### PHASE 7 — Productionization
- **Goal**: fold validated optimizations + already-recommended operational items (MPS lifecycle, fusion flags) into every standing job script as defaults.
- **Actions**: update job-script templates; final same-node A/B confirming the cumulative effect of all merged Phase 2-6 changes.
- **Files likely involved**: standing `.sbatch`/`.j` templates (outside `cuda/` source).
- **Profiler/tool required**: none (validation run only).
- **Expected performance range**: cumulative effect of whichever earlier phases succeeded.
- **Correctness gate**: full C512 acceptance suite (10/10, matching Prompt 44's precedent).
- **Stop criteria**: all validated optimizations wired as defaults, one final documented same-node A/B result.
- **Rollback criteria**: any regression vs. the last known-good tag.
- **Git checkpoint**: tag the final state (e.g. `cuda-perf-optimized-c512-<date>`).

---

## Part 6 — Performance Targets

Baseline: **C512 CUDA ~6374 ms/call** (Fortran also ~6374 ms/call — parity).

| Milestone | Target | Optimization class expected to achieve it | Confidence |
|---|---|---|---|
| A | <5500 ms/call | Sync reduction (Phase 4) + `update_dz_c` launch-loop fix (Phase 3) | MEDIUM |
| B | <5000 ms/call | Milestone A + top-kernel occupancy fixes (Phase 2), if confirmed occupancy-bound | MEDIUM |
| C | <4000 ms/call | Milestone B + vertical-stage fusion (Phase 3) + graph-coverage widening (Phase 4), assuming all individually validated gains are stackable without interaction penalties | LOW-MEDIUM |
| Final | <=3187 ms/call (>=2x vs. Fortran) | Requires stacking most/all of Phase 2-5 wins, and plausibly Phase 6 architecture change | LOW |

No gain is promised. Given that the baseline is already at parity (not a
regression to recover), reaching Milestone C or the Final target likely
requires more than kernel-level tuning alone — the roadmap should be
treated as a sequence of independently-validated, stackable wins, with the
Final target as an aspirational ceiling contingent on Phase 1 evidence that
has not yet been collected.

**Update (Prompt 51/52)**: Phase 1 evidence has now been collected and
confirms the above caution — existing GPU kernel compute is only 3.9% of
the window, so Phase 2-4 kernel/sync tuning alone (Milestones A-C) cannot
reach the Final <=3187 ms/call target regardless of how completely it is
executed; their combined ceiling is bounded by ~18-20% of the window (3.9%
kernels + 14.4% sync, per Prompt 49/51), i.e. at most ~1.2x even if every
kernel/sync inefficiency in Phase 2-4 were fully eliminated. Reaching the
Final target requires a new phase class — **CUDA coverage expansion**
(porting currently-Fortran-only large-timestep work, starting with
`fv_mapz_mod`, 18.67% of the window, Amdahl ceiling ≈1.23x alone — see
`docs/performance/p52_fv_mapz_cuda_port_plan_2026-09-21.md`) — stacked on
top of, not instead of, Phase 2-4. Milestones A-C's confidence ratings are
otherwise unchanged; this update only clarifies that they are necessary
but no longer believed sufficient.

---

## Part 7 — Correctness Policy

Every substantial optimization must be validated against `cuda-perf-baseline-c512`:

1. Targeted kernel/unit correctness test where available (existing ctest infrastructure).
2. Short C512 dyn_core comparison (matching the Gate C2 savepoint methodology).
3. Sampled physical fields (T/U/V/PS at minimum, matching the HS validation checkpoints already established).
4. Established bitwise/round-off tolerance (matching Gate C2's demonstrated worst-case `w` max_rel ~5e-11 R8 / ~2.4e-1 R4-with-explained-small-denominator precedent — any new divergence outside these envelopes is a regression, not noise).
5. Performance A/B on the same node/allocation (mandatory — Prompt 48 lesson; no cross-node comparison is acceptable evidence).

Classification:

- **KEEP**: faster + numerically valid.
- **REVERT**: slower (even if numerically valid).
- **REVERT**: numerical regression (even if faster).
- **INVESTIGATE**: performance gain but unexplained numerical differences — do not merge until explained.

Do not stack unvalidated optimizations — each candidate is validated and
classified independently before the next is attempted, per Part 8's
one-optimization-per-commit rule.

---

## Part 8 — Git Strategy

- Branch: `optimization/cuda-fv3-performance` (current).
- Baseline tag: `cuda-perf-baseline-c512` — immutable reference, never rebased/moved.
- Commit pattern (illustrative):
  ```
  profile: capture same-node C512 nsys baseline (Phase 0/1)
  fix: replace update_dz_c km+1 launch loop with single leveled kernel (Phase 3)
  validate: update_dz_c fix vs cuda-perf-baseline-c512 (bitwise PASS)
  perf: reduce riem_solver3_kernel register pressure (Phase 2)
  validate: riem_solver3_kernel occupancy fix (bitwise PASS, occupancy +X%)
  perf: fuse update_dz_d height-boundary clamp into riem_solver3 (Phase 3)
  validate: vertical fusion vs baseline (bitwise PASS / REVERT — record outcome)
  perf: reduce halo-group host-blocking sync count (Phase 4)
  validate: sync reduction — 10/10 C512 acceptance
  ```
- One meaningful optimization per commit; do not combine unrelated performance changes in a single commit.
- Risky/exploratory work (vertical fusion, comm/compute overlap, multi-GPU architecture) goes on temporary sub-branches, e.g. `optimization/cuda-fv3-performance/vertical-fusion-spike`, `optimization/cuda-fv3-performance/comm-overlap-spike`, merged back only after full validation; abandoned if REVERTed.
- Tag any milestone that passes full correctness + same-node performance validation (e.g. `cuda-perf-milestone-a-c512`).
- No push/merge without explicit request, per project policy.

---

## Part 9 — Multi-GPU Strategy

**When to start using >1 GPU on the same node**: **NOT NOW.** Prompt 47's
controlled same-node A/B directly refuted rank/GPU density as a C512
performance lever (all layouts within 0.49%). The only "multi-GPU looks
faster" signal on record (Prompt 46's informal ~2x 1-vs-2-GPU comparison)
was later shown by Prompt 48 to be the same node-to-node variance artifact
that produced the original false "CUDA slower" result — it was never a
real multi-GPU effect. Current single-GPU arithmetic intensity and
communication/computation ratio have not been profiled (Phase 1 has not
run yet), so there is no basis to believe additional GPUs would help
before single-GPU kernel efficiency is even characterized.

**When to start using multiple GPU nodes**: not before single-node,
single-GPU efficiency work (Phase 2-5) is judged insufficient to reach the
2x target, and even then only with multi-node communication overlap
(Phase 5) already in place — otherwise inter-node communication cost would
likely dominate more than the already-serialized single-node halo cost.

**Prerequisites before multi-GPU scaling is worth revisiting**:

1. Phase 1 profiling shows single-GPU dyn_core is not communication-bound (or communication overlap from Phase 5 has already been implemented and validated).
2. Phase 2-4 kernel/sync-level optimizations have been implemented and validated, and the measured result is still short of the 2x target.
3. A same-node-controlled multi-GPU A/B design is ready (same allocation, alternating layouts, multiple repeats) specifically to avoid repeating the Prompt 37/47 node-variance confound.
4. A specific architectural hypothesis for *why* more GPUs would help (e.g. arithmetic-intensity-limited single-GPU throughput, not just "more hardware") — Prompt 47 already shows density alone is not that hypothesis.

---

## Part 10 — Top 5 Next Actions

1. Capture a same-node, 5-10-call C512 Nsight Systems profile (Phase 0/1 methodology above) and account for >=90% of dyn_core wall time, reporting the Rank/Component/Time/%total table and top-10-kernels-by-wall-time table specified in Part 3.
2. Run Nsight Compute on `riem_solver3_kernel` and `riem_solver_c_kernel` from that same capture window; report achieved occupancy, registers/thread, and local-memory throughput for each.
3. Prototype a single-kernel replacement for the `update_dz_c` km+1 sequential-launch loop (`cuda/dyn_core/vertical/fv3_vertical_kernels.cu`) and validate it bitwise against `cuda-perf-baseline-c512` via a same-node C512 A/B.
4. Run Nsight Compute on the halo pack/unpack kernels (`runtime/fv3_gpu_halo_exchange.cu`) and separately count/time the actual `cudaEventSynchronize` occurrences per dyn_core call in a live C512 run (`fv3_gpu_halo_mpi.cpp:243`).
5. Re-run Prompt 37's FMS clock-block attribution (`COMM_DSW`/`COMM_TOTAL` vs. compute blocks) under the Prompt 48 same-node paired methodology to produce a trustworthy communication-vs-compute split, replacing the confounded original figure.
6. **(Added, Prompt 52; status per Prompt 69: PORTED, MEMORY-FIXED, PERFORMANCE-VALIDATED POSITIVE)** Close the `fv_mapz_mod` coverage gap. The `ak`/`bk` registration crash (Prompt 53) and all four integration bugs (Prompts 57/59/61/62) are FIXED and NUMERICALLY VALIDATED (Prompt 66: CONDITIONAL ACCEPT of the residual `exp`/`log` round-off). The Prompt 67 production-density OOM was root-caused and fixed in Prompt 68 (CUDA implicit local-memory reservation, ~3.06 GiB/rank; fixed via device-workspace reuse, zero extra bytes). **Prompt 69 measured the fixed port's real paired same-node C512 performance: 33.7% faster at REMAPPING, 16.6% faster at FV_DYN_LOOP, 7.3% faster at TOTAL wall-clock — KEEP.** `DYN_CORE` itself is unaffected (timer-hierarchy correction: `DYN_CORE` is a sibling of `REMAPPING`, not its parent — PT/DELZ never showed up there). See `docs/performance/p69_pt_delz_paired_performance_2026-09-23.md` for full data, GPU-cost/transfer breakdown, and reconciliation.
7. **(Added, Prompt 69; status per Prompt 71: IMPLEMENTED, BITWISE-EXACT VALIDATED; PERFORMANCE-NEGATIVE CLASSIFICATION BELOW SUPERSEDED BY PROMPT 76A (item 12) — CORRECTED TO PERFORMANCE POSITIVE, CUDA ~10.23x FASTER ON REMAP_UV)** `REMAP_UV` (wind/momentum vertical remap in `fv_mapz_mod`) was implemented per the Prompt 70 port plan (2-line `iv=-1` sign-clamp addition to `scalar_profile_column` + new `fv3_remap_uv.cu`/`.cuh` dispatch reusing PT/DELZ's device-workspace pattern). Correctness closed at every level, including matched C512: **0/216 (C32, calls 1-6) and 0/108 (C512, calls 1-3) bitwise differences** — exact IEEE-754 identity, stronger than PT/DELZ's own round-off-only ceiling (no transcendental calls in the `iv=-1` path). Performance A/B (same-node, 3-pair paired C512 run, job 58533089) found CUDA REMAP_UV's own timer at ~169 ms/call vs. ~16.5 ms/call for the Fortran-forced control; this was reported at the time as CUDA being ~10.3x slower, regressing `REMAPPING` by +27.0% and `FV_DYN_LOOP` by +6.0% (reproducible within ~1% across all 3 pairs); `DYN_CORE`/`TOTAL` unaffected (same sibling-timer relationship as Prompt 69). **That directionality is now flagged as an open, unresolved question** — a smaller CUDA timer value reads as faster, not slower — per `p71_remap_uv_cuda_implementation_2026-09-23.md` §3 caveat; it is explicitly NOT re-analyzed by Prompt 73. Prompt 72A disabled CUDA REMAP_UV from the default production path pending that re-analysis, opt-in preserved via `RUN_CUDA_FV3_REMAPUV_ENABLE`. `REMAP_Q` (tracer remap, 116.8 ms/call, 6.5% of `FV_DYN_LOOP`) remains a candidate but should first account for whether the same per-row-dispatch risk applies before porting, and the REMAP_UV re-analysis should happen before further `fv_mapz` family porting to avoid repeating whatever labeling issue is in play.
8. **(Added, Prompt 73 — MAPPING/HARNESS ONLY, no CUDA implementation)** `TRACER_2D` (~280 ms/call, largest unassessed dynamics-side CPU cost after PT/DELZ/REMAP_UV) was mapped end-to-end: production dispatches to `tracer_2d_1L` (`Z_TRACER=T`, `Q_SPLIT=0`, confirmed via runtime namelist echo); exactly **1 tracer** is allocated/transported in production Held-Suarez runs despite `NWAT=0`, via GEOS's `GRID%NQ=MAX(1,ncnst)` floor (`FV_StateMod.F90:960,1490-1494`) — proven from source, not assumed. **Key architectural finding (later qualified by Prompt 74, item 9 below)**: `tracer_2d_1L`'s numerical core, `fv_tp_2d`, is a WP-B-owned transport service already GPU-executed in this repo (`dgrid::launch_scalar_transport_cuda`) — re-porting it would duplicate existing validated work. The genuinely new, TRACER_2D-unique CUDA work is the per-k-level orchestration layer (`xfx`/`yfx`, `ra_x`/`ra_y`, `dp2`, the conservative mass-weighted `q` update) — selected as the port unit, explicitly scoped to the `nsplt==1` case first (the `nsplt>1` mid-loop-halo case and a global-max `mp_reduce_max` allreduce are real, recurring production communication points inside this routine and are flagged as required follow-on work, not silently dropped). A host-only harness transcribing this unit's exact arithmetic (`cuda/dyn_core/tests/host/test_tracer2d_orchestration_harness.cpp`) built/ran on the login node: 9/9 fixtures PASS, mass-conservation residuals at floating-point roundoff, 0 NaN/Inf, deterministic. Portability: GPU FEASIBLE WITH RESTRUCTURING (GPU FRIENDLY for the `nsplt==1` slice alone). No performance ceiling number is claimed — `fv_tp_2d`'s share of the 280 ms/call figure is not separately timed today; a sub-routine timer split is the recommended first step of the actual port prompt. See `docs/performance/p73_tracer2d_cuda_port_plan_2026-09-24.md`. **Result at the time: TRACER_2D MAPPED, HARNESS READY — recommend PORT TRACER_2D CUDA UNIT.**

9. **(Added, Prompt 74 — MEASURED, NO-GO, no CUDA implementation)** Instrumented `tracer_2d_1L` with five diagnostic-only sub-timers (env-gated `RUN_CUDA_FV3_TRACER2D_DIAG`, zero default-path overhead) and measured a real C512 production-density breakdown (job 58539236): **A — initial halo 32.7 ms/call (11.7%); B — Courant/flux setup+allreduce 47.4 ms/call (16.9%); C — `fv_tp_2d` 195.9 ms/call (69.7%, dominant); D — host orchestration 39.9 ms/call (14.2%); E — nsplt>1 mid-split 0 ms/call (0%, never entered — 180/180 dumped `(call,rank,k)` triples show `nsplt=1`)**. Component D (the orchestration slice Prompt 73 selected to port) landed just under both GO thresholds (40 ms/15%) in the "engineering judgment" zone — but the decisive finding was independent of that: **`hord_tr` resolves to 8 in production (confirmed live from the job log), not the 6 required by the existing `launch_scalar_transport_cuda` GPU kernel** — invalidating Prompt 73's "fv_tp_2d is already GPU-executed" premise for the real tracer path. Porting orchestration alone would therefore require either re-porting `fv_tp_2d` for `hord==8` (a WP-B-sized task, out of scope this prompt) or wrapping GPU orchestration around a Fortran-only `fv_tp_2d` — the same CPU/GPU hybrid-split pattern already rejected in Prompt 65. **Result: TRACER_2D ORCHESTRATION PORT NOT WARRANTED. NO-GO. No CUDA implementation.** New dominant, unaddressed component: `fv_tp_2d` itself at production `hord_tr=8` (69.7% of TRACER_2D), which would need its own dedicated WP-B-scoped port (extending `launch_scalar_transport_cuda` to `hord==8`) before any TRACER_2D GPU work is worthwhile. See `docs/performance/p74_tracer2d_orchestration_cuda_2026-09-24.md`.

10. **(Added, Prompt 75 — MAPPING/HARNESS ONLY, no CUDA implementation)**
    Mapped and harnessed `fv_tp_2d` at production `hord_tr=8` (the new
    dominant, unaddressed component identified by item 9) against the
    existing hord==6 CUDA implementation. Found hord==6/hord==8 are two
    branches of the SAME `xppm`/`yppm` subroutines (not separate
    routines), and that the existing CUDA orchestration
    (`launch_fv_tp_2d_cuda`) is entirely hord-independent and directly
    reusable — only the low-level face-flux device functions
    (hardcoded to hord==6) and a not-yet-ported `pert_ppm` helper need
    new, bounded logic. No architectural mismatch. Confirmed the
    existing production halo (`ng=3`, `i_pack(10)`) already covers
    hord==8's stencil reach — no halo widening required. Built an
    11-fixture host-only harness (exact transcription of tp_core.F90's
    hord==8 branch, no production source touched): 0/11 NaN/Inf, 11/11
    bitwise-repeatable (including an ASan pass), conservation-identity
    PASS on all 11. Estimated port scope MEDIUM, classified MODERATE
    NEW NUMERICAL BRANCH. Theoretical performance ceiling (existing
    measured figures only): up to 69.7% of `TRACER_2D`, 8.34% of
    `FV_DYN_LOOP`, 3.07% of C512 `DYN_CORE`. **Result: PROCEED TO CUDA
    HORD8 EXTENSION** — the actual CUDA implementation is future work,
    not performed this prompt. See
    `docs/performance/p75_fv_tp_2d_hord8_cuda_plan_2026-09-24.md`.

11. **(Added, Prompt 76 — FINAL: STANDALONE UNIT CORRECTNESS COMPLETE
    AND GPU-VALIDATED; INTEGRATION BLOCKED BY DESIGN)** Acted on item
    10's PROCEED decision: hord==8 x/y face-flux kernels
    (`xppm_pass_kernel_hord8`/`yppm_pass_kernel_hord8`, Lin's fast
    monotone constraint + panel-edge one-sided treatment) and the
    `pert_ppm(iv=1)` device helper in
    `cuda/dyn_core/dgrid/wp_b_cuda.cu`/`wp_b_ppm_shared.cuh`, plus
    dispatch-gate widening in `launch_fv_tp_2d_cuda`/
    `launch_scalar_transport_cuda` (`mord==6` unchanged, `mord==8`
    added, other `mord` still rejected) — GPU-executed and validated
    this session: `pert_ppm_iv1` 5/5 cases BITWISE PASS; hord==8
    face-flux fixtures (interior/x-edge/y-edge/all-corners,
    C512-like geometry) ALL PASS at round-off (max_rel ~2-4e-16); the
    pre-existing 11-fixture Part L (calling the real production
    `launch_fv_tp_2d_cuda`/`launch_scalar_transport_cuda` entry points)
    independently confirms 10/11 PASS, with the 11th
    (`constant`) a benign pre-existing anti-degeneracy test-harness
    artifact, not a numerical defect; hord==6 regression unaffected
    (all pre-existing parts still PASS, bitwise); `compute-sanitizer
    --tool memcheck` 0 errors; conservation follows analytically from
    the bitwise fx/fy match (no independent residual measurable).
    Register footprint: 56 registers/kernel, 0 stack frame, 0 spills
    (vs. 48-49 registers for the hord==6 kernels) — no local-memory
    growth, no Prompt-68-style OOM risk. **Integration (C32/C512
    correctness, performance) is BLOCKED BY DESIGN, not by resources**:
    direct inspection of the authoritative Fortran tree confirms no
    dispatch selector routes the real production `tracer_2d_1L` ->
    `fv_tp_2d` call through any CUDA path (the only CUDA-tagged
    `fv_tp_2d` symbol, `tp_core_cuda`/`fv_tp_2d_cuda`, lives only in an
    isolated `tp-core-driver/test.F90` driver, not linked into
    `fv_tracer2d.F90`/`fv_mapz.F90`/`fv_dynamics.F90`); adding one is
    itself a tracer-orchestration change, explicitly forbidden by this
    task's change budget. Recommended follow-up: a dedicated task to
    add that selector (mirroring `use_cuda_remap_uv`'s opt-in pattern),
    which would unblock C32/C512 integrated correctness and paired
    performance. See
    `docs/performance/p76_fv_tp_2d_hord8_cuda_implementation_2026-09-24.md`.

12. **(Added, Prompt 76A — CPU/bookkeeping only, no GPU jobs, no source
    change)** Re-derived item 7's flagged-unresolved REMAP_UV performance
    directionality directly from job 58533089's raw `fv_timing_mod`
    output (not prose). Arm labels re-proved from the sbatch mechanism
    (`RUN_CUDA_FV3_REMAPUV_FORCE_FORTRAN=1` = Arm A/Fortran, unset =
    Arm B/CUDA) — unchanged from item 7, confirming the arm labels were
    never the issue. **Root cause: a simple inversion of ratio-
    interpretation direction** — the original analysis intuitively
    assigned the larger (~169 ms/call) timer value to "CUDA" and the
    smaller (~16.5 ms/call) value to "Fortran" (a launch-overhead
    narrative), when the sbatch selector proves the opposite assignment.
    Recomputed `speedup = Fortran/CUDA`: **REMAP_UV 10.23x faster on
    CUDA (90.2% reduction); REMAPPING improves 21.3%; FV_DYN_LOOP
    improves 5.7%; TOTAL improves ~1.0%** — all in CUDA's favor, <1%
    spread across all 3 pairs. **Result: REMAP_UV CUDA PERFORMANCE
    POSITIVE — item 7's "PERFORMANCE NEGATIVE — REVERT/DISABLE"
    classification above is SUPERSEDED.** Prompt 72A's default-disable
    of CUDA REMAP_UV was made on this incorrect premise; this item does
    **not** flip `use_cuda_remap_uv`'s default back — that remains
    Fortran-default/opt-in-CUDA (`RUN_CUDA_FV3_REMAPUV_ENABLE`) pending a
    future, explicitly-authorized re-enable decision. See
    `docs/performance/p71_remap_uv_cuda_implementation_2026-09-23.md` §7
    for the full re-derivation and Stage-by-stage evidence trail.

13. **(Prompt 77)** Added the production dispatch selector item 11 flagged
    as the blocker: `use_cuda_tp2d` in `fv_tracer2d.F90`
    (`hord==8 .and. lim_fac==1.0 .and. kind(q)==8`, opt-out
    `RUN_CUDA_FV3_TRACER_FORCE_FORTRAN`), routing the real
    `tracer_2d_1L` -> `fv_tp_2d` call through the validated hord==8 CUDA
    kernel. Mid-task source audit of `fv_restart.F90` found the entire
    standalone cold-start harness family (`COLDSTART=1`+
    `external_ic=.false.`+empty `ExtData.rc`, not HS-specific) leaves
    `Atm%q` structurally zero. Per explicit user direction,
    `fv_restart.F90` was not modified and no synthetic tracer seed was
    introduced; Stage 7/9 C32/C512 same-node A/B (jobs 58543176/58543590)
    instead validate **dispatch/integration wiring only** (selector
    fires correctly/exclusively each direction, matched call/PE
    structure, bitwise-identical degenerate `q`, zero NaN/Inf/Abort) —
    kernel numerical correctness for non-degenerate content remains
    established solely by Prompt 76's fixtures. A pre-performance source
    audit confirmed no data-dependent zero-`q` early exit in the CUDA
    path. **Stage 11 same-node C512 performance (job 58543590):
    MEASURED REGRESSION** — TRACER_2D +44.1% slower, DYN_CORE +13.9%
    slower, FV_DYN_LOOP/TOTAL +10.0% slower (CUDA vs Fortran, 3-call
    sample). Per the KEEP/DISABLE thresholds this is a REGRESSION;
    recommend the default remain Fortran-dispatch for tracer transport
    pending a dedicated overhead-attribution follow-up and a replicated
    alternating-order measurement. See
    `docs/performance/p77_fv_tp_2d_hord8_production_integration_2026-09-24.md`.

14. **(Prompt 78)** Confirmed and attributed item 13's TRACER_2D
    regression with a rigorous same-node, same-allocation, strictly
    alternating A1 B1 A2 B2 A3 B3 C512 rerun (job 58544700, 11 dyn_core
    calls/invocation, 33 calls/arm) plus a paired `nsys` (CLI stats
    only, no `ncu`) rank-0 capture (job 58545163, 7 calls/arm).
    **CONFIRMED REGRESSION**: TRACER_2D +37.87% slower (medians,
    33-call sample) — cross-validated by the independent 7-call `nsys`
    sample at +39.3%. `nsys` kernel-level evidence rules out GPU compute
    as the cause (the two hord8 tracer kernels total only ~1.15ms/call,
    negligible against the ~107-111ms/call TRACER_2D delta). Root cause:
    the CUDA tracer dispatch path issues ~300 small, **synchronous**
    (blocking) `cudaMemcpy` host<->device transfers per `dyn_core` call
    (2100 calls total in the 7-call capture, entirely absent from the
    Fortran arm), accounting for ~20ms/call (~18%) of the regression via
    CUDA-API-visible time; the remaining ~91ms/call (~82%) is CPU-side
    host orchestration/marshaling around those transfers, invisible to
    CUDA API tracing. Also found: `REMAP_TOTALENERGYCON` (a CPU-only
    Fortran diagnostic timer, sibling to `REMAPPING`) is +26.7% slower
    in the CUDA-tracer arm despite REMAP_UV/PT-DELZ being config-
    identical both arms — flagged for future investigation, not yet
    root-caused, did not change the primary attribution. Recommended
    (not implemented): batch the ~300 small synchronous per-call
    transfers into a single packed asynchronous H2D/D2H transfer per
    `fv_tp_2d` invocation. Default dispatch remains Fortran for tracer
    transport (`use_cuda_tp2d` unchanged). See
    `docs/performance/p78_tracer_hord8_overhead_attribution_2026-09-24.md`.

15. **(Prompt 79)** Implemented item 14's recommended fix: rewrote
    `cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu` to pack all 12
    inputs + 3 outputs of one (k, iq) hord==8 `fv_tp_2d` slice call into
    one contiguous pinned host buffer / one contiguous device buffer,
    replacing 15 per-slice-call blocking `cudaMemcpy` calls with exactly
    one `cudaMemcpyAsync` H2D + one `cudaMemcpyAsync` D2H (verified via
    a new env-gated diagnostic, `RUN_CUDA_FV3_TRACER_TRANSFER_COUNT_DUMP`).
    At C512 production scale this cuts the transfer count from ~300 to
    ~40 per `tracer_2d_1L` invocation. ABI/signature unchanged; no
    Fortran edit needed. Standalone GPU regression (dev-stack job
    58545595): numerically identical to the Prompt 76 baseline, 0
    sanitizer errors — confirms no numerical semantics changed. **Could
    not complete the planned C32/C512 integration A/B**: relinking the
    production `StandAlone_FV3_Dycore.x` binary against the rebuilt
    production (`build_cuda_r8_intelmpi`) archives failed on a
    pre-existing, unrelated Intel ifort `#6633` error in `FV_StateMod.F90`
    (severe-weather diagnostics, never successfully compiled in this
    build tree, untouched by any tracer/CUDA work). Default dispatch
    remains Fortran for tracer transport. See
    `docs/performance/p79_tracer_batched_transfer_optimization_2026-09-24.md`.

16. **(Prompt 80)** Diagnosed item 15's build blocker: forced a
    from-scratch rebuild of both `fv_arrays.F90.o` (the provider) and
    `fv_diagnostics.F90.o` (the actual definer of the failing severe-
    weather-diagnostic routines, not just the `FV_StateMod.F90.o`
    consumer) — both still reproduced the identical `#6633` class,
    including on calls internal to `fv_diagnostics.F90` needing no
    external interface at all. Concluded (Category G) this was a
    genuine, pre-existing source/toolchain-level mismatch, not a stale
    build artifact, and — per the task's own Stage-8 authorization
    gate — made no source edits. **This conclusion was itself incorrect
    (superseded by item 17)**: forcing two individual objects to rebuild
    is not equivalent to a clean rebuild of the whole target when other,
    non-obviously-implicated `.mod` files in the same target are stale.
    See `docs/performance/p80_fv_statemod_build_tree_repair_2026-09-24.md`.

17. **(Prompt 81B)** Corrected and resolved item 16's build blocker, then
    resumed and closed item 15's blocked integration A/B. A genuinely
    clean rebuild of the entire `fvdycore` CMake target (all `.o`/
    `.provides.build`/`.mod` state deleted first, not just the two
    objects Prompt 80 forced) collapsed a 416-error/~140-line
    `FV_StateMod.F90` rescan down to 8 errors on one call,
    `Held_Suarez_Forcing` (`FV_StateMod.F90:2041-2043`, 8 args) — never
    named by any prior prompt and not one of Prompt 80's implicated
    severe-weather-diagnostic routines, all of which turned out to be
    stale-build noise. Fixed by typing `Held_Suarez_Forcing`'s 8 dummy
    arguments `real(FVPRC)` in `Held_Suarez_ForcingMod.F90` (bare `real`
    -> `real(FVPRC)`; intra-procedure local variables and `HS_*`
    canonical constants deliberately left as bare `real` — semantically
    identical under both R8/`-r8` and R4/`SINGLE_FV`, confirmed by an R4
    spot check). Clean rebuild + relink of `StandAlone_FV3_Dycore.x`:
    0 `#6633`, EXIT=0
    (sha256 `e82d526aa0b3b96541f27073ceb5e094112932f878989e931b349bace4474599`,
    confirmed linked against Intel MPI 2021.13 and carrying Prompt 79's
    batched-transfer bridge symbols). C32 smoke: PASS. Resumed item 15's
    blocked measurement: dispatch-wiring PASS at C32/C512, but a rigorous
    same-node, alternating-order paired A/B (3 pairs, C512, job
    58549977) found the CUDA tracer arm still **~54.3% slower** on
    TRACER_2D (range 48.4%-59.2%), DYN_CORE neutral (+0.57%), FV_DYN_LOOP
    +6.57% — item 15's 87%-fewer-transfers batching fix **did not**
    recover item 14's regression. **Final verdict for the hord==8 CUDA
    tracer-dispatch line of work: REGRESSION-REMAINS.** Default dispatch
    remains Fortran for tracer transport (`use_cuda_tp2d` unchanged).
    Recommended next step (not done): a fresh `nsys` capture of the
    packed-buffer CUDA arm, since transfer count dropped sharply but wall
    time barely moved — the overhead has relocated, not disappeared. See
    `docs/performance/p81b_fvprc_r8_interface_compatibility_2026-09-24.md`
    and the Prompt-79-continuation section appended to
    `docs/performance/p79_tracer_batched_transfer_optimization_2026-09-24.md`.

18. **(Prompt 82 — profiling/attribution only, `nsys` only, no `ncu`, zero
    source changes)** Root-caused item 17's still-unresolved batched-
    design regression via a same-node `nsys` A/B with CPU sampling enabled
    (`--sample=cpu --backtrace=fp`, a deliberate deviation from Prompt
    78's `--sample=none`) plus direct `SAMPLING_CALLCHAINS`/`StringIds`/
    `COMPOSITE_EVENTS` SQL queries against the exported `.sqlite` (no
    built-in `nsys stats` report covers CPU sampling in this nsys version,
    2024.4.1.61). **Root cause: `pack_field`/`unpack_field`
    (`cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu:156-164`) are
    serial, non-`__restrict__`-qualified, scalar cast-copy loops that
    marshal all 12 inputs + 3 outputs into/out of the one pinned staging
    buffer** — a CPU-side cost the unbatched design never had (it
    transferred fields directly from/to the original Fortran pointers,
    no staging copy). Measured at ~520.7 ms/call combined CPU-busy time
    (259.6 ms/call inside the C++ bridge + 261.1 ms/call in a related
    Fortran-side ABI-boundary copy), more than 3x the +152.2 ms/call
    (+54.3%) net TRACER_2D regression anchor (job 58549977) by itself.
    Two secondary, smaller contributors, both plausibly downstream
    consequences of the same design rather than independent causes:
    device-transfer bandwidth degraded (H2D ~11.5 GB/s unbatched -> ~8.3
    GB/s batched, +57.3 ms/call) and `cudaStreamSynchronize` cost
    exploded at identical call count (9.876ms -> 385.30ms total over a
    7-call capture, +53.6 ms/call) — both consistent with fewer, larger,
    fully-serialized/unoverlapped transfers each blocking longer. Bytes
    moved are ~flat (D2H bit-identical, H2D +6.3%) — rules out "more data
    moved" as a cause. Kernel compute remains negligible (~1.2 ms/call).
    **Result: ROOT-CAUSED, recoverable with one narrow fix** (add
    `__restrict__`; bulk `memcpy` for same-type fields; or direct
    `cudaMemcpyAsync` from the original Fortran pointers, bypassing the
    host staging copy entirely) — not implemented this task (diagnose
    only). Given this is the **second** consecutive tracer-CUDA design to
    regress production (unbatched P77/78, batched P79/81B), any follow-up
    fix should be narrowly scoped to exactly this mechanism and re-
    verified with a fresh alternating-order A/B (P48/P76A rigor bar)
    before any production-default change. See
    `docs/performance/p82_tracer_batched_regression_attribution_2026-09-24.md`.

19. **(Prompt 83 — final bounded tracer-optimization attempt, per its own
    task framing)** Implemented item 18's two most directly indicated
    fix candidates simultaneously in `pack_field`/`unpack_field`:
    `__restrict__` pointer qualification plus `if constexpr`-gated bulk
    `std::memcpy` for the `T == fv_real` case (proven true for all 12
    inputs + 3 outputs in the R8 production configuration via a Stage
    0/1 memory-layout/classification pass — all fields Class A,
    contiguous, identical representation, memcpy-safe). Bitwise-verified
    correct in all configurations. **Stage 4 host microbench GATE
    FAILED**: at production-matching optimization (`nvcc -O3 -DNDEBUG`,
    matching `build_cuda_r8_intelmpi`/`build_cuda_r8_perf`'s
    `CMAKE_BUILD_TYPE=Release`), combined pack+unpack speedup measured
    only **1.196x** (g++ -O2: 1.416x) — far short of the required >=5x.
    Root cause: GCC/nvcc already recognize the old same-type scalar
    copy loop as a bulk-copy idiom at `-O2`/`-O3` and compile it to
    near-`memcpy` speed, leaving little headroom for a hand-written
    replacement to exploit. Per this task's own binding Stage-4
    contingency ("if bulk copy is unexpectedly not much faster: STOP
    and investigate only the copy implementation; do not proceed to GPU
    performance"), Stages 5-9 (GPU regression, C32/C512 integration,
    paired performance, in-situ attribution) were **not run** — no
    Slurm job submitted. **Final verdict: STOP CUDA TRACER PRODUCTION
    OPTIMIZATION** per the task's explicit binding instruction. The
    Stage 2 code fix is retained (harmless, bitwise-correct, marginally
    faster) even though it doesn't clear the required improvement bar.
    Fortran remains the tracer-dispatch production default
    (`use_cuda_tp2d` selector unchanged). The one untried named
    alternative — direct `cudaMemcpyAsync` from the original Fortran
    pointers, bypassing the packed staging buffer entirely — was out of
    this task's change budget (a second architectural redesign) and,
    per the binding STOP instruction, should not be pursued as an
    automatic follow-on. See
    `docs/performance/p83_tracer_bulk_host_pack_optimization_2026-09-24.md`.

20. **(Prompt 84A)** Acted on item 12/Prompt 76A's corrected REMAP_UV
    performance evidence (10.23x faster on CUDA, 90.2% reduction,
    bitwise-exact correctness) and flipped `fv_mapz.F90`'s
    `use_cuda_remap_uv` selector default from Fortran/opt-in-CUDA to
    CUDA-on-by-default whenever the base gate (`kord==9`, no dead
    `mfx`/`mfy`/`cx`/`cy` args, CUDA backend enabled) holds.
    `RUN_CUDA_FV3_REMAPUV_FORCE_FORTRAN` preserved as the escape hatch;
    `RUN_CUDA_FV3_REMAPUV_ENABLE` preserved as a now-redundant
    compatibility no-op. No numerics/dispatch-body change. Rebuilt
    `StandAlone_FV3_Dycore.x` (new SHA256 `d2d14df8...4ff1`). C32
    selector smoke (job 58550862): default arm selects CUDA 36/36,
    force-Fortran arm selects Fortran 36/36, both rc=0, zero NaN/Inf/
    Abort. No C512 rerun performed — Prompt 71/76A's existing
    correctness/performance evidence was deemed sufficient per explicit
    instruction. **Result: REMAP_UV CUDA RESTORED AS PRODUCTION
    DEFAULT.** See
    `docs/performance/p71_remap_uv_cuda_implementation_2026-09-23.md` §8.

21. **(Prompt 84B — REMAP_Q source map + authoritative CPU harness;
    bounded task, explicitly NOT a CUDA implementation task)** Mapped
    `fv_mapz.F90`'s `Remap_Q` timer down to its sole numerical routine,
    `mapn_tracer`. Confirmed MOIST_CAPPA/USE_COND not relevant (no
    references anywhere in `mapn_tracer`'s body). Confirmed production
    `nq=1` (same GEOS `MAX(1,ncnst)` floor-tracer finding as Prompt 73),
    unconditionally processed by REMAP_Q's `mapn_tracer` call (no
    `nq>0`/`present()` guard exists at the call site, unlike REMAP_UV's
    dead flux-remap calls). Confirmed `kord_tr=8` hits `scalar_profile`'s
    `abs(kord)<9` branch — **not yet CUDA-ported** (the existing
    `scalar_profile_column` only implements `abs(kord)==9`, used by
    PT/DELZ/REMAP_UV) — but a strict *simplification* of the already-
    ported branch (same `pmp_1`/`lac_1`/`pmp_2`/`lac_2` formulas, no
    extm-gating/qmin-recheck), so classified REUSE WITH NEW KORD MODE,
    not a new algorithm. Confirmed `mapn_tracer`'s Mapping do-loop
    (linear-scan search+integrate) is algebraically identical to the
    already-ported `map_scalar_column`'s standard-remap else-branch —
    REUSE DIRECTLY. Confirmed via `fv3_gpu_field_registry.F90` grep:
    `pe`/`ak`/`bk`/`delp` already device-resident; the tracer array `q`
    itself is host-only (same architectural gap TRACER_2D hit) — any
    port needs new H2D/D2H wiring for `q`, batched as one packed
    transfer/call (P78/P82's small-transfer regression is the explicit
    cautionary precedent, not to be repeated). Confirmed no MPI/halo/
    FMS/collective calls inside the `Remap_Q` timer. Extended the
    existing `map_scalar-driver/main.F90` (separate
    `GEOSfvdycore_fortran_perf` checkout, per the project's reuse
    policy) with 10 new fixture cases (constant, smooth gradient, sharp
    front, local max/min, near-q_min, q_min-clipping, boundary-
    sensitive, C512-like, `nq=3` multi-tracer) calling the unmodified
    authoritative `mapn_tracer` directly, plus a mass-conservation
    invariant check per case. Found and fixed one pre-existing,
    independently-discovered build-staleness gap (a missing
    `fv3_gpu_interface_remap_uv` stub, dating to Prompt 71/72A's REMAP_UV
    wiring) and two harness-only bugs (a mass-check that reused an
    already-overwritten array; one fixture using an unphysical
    negative-mean input against a routine whose positivity guarantee
    only holds for non-negative cell means) — all diagnostic/test-only
    fixes, zero numerical-source changes. **Final harness run: 23/23
    cases PASS (`MAP_SCALAR_DRIVER: RESULT=PASS`, rc=0), 0 NaN/Inf,
    bitwise-repeatable, positive-definite floor holds, mass conservation
    holds.** LOC estimate for the actual CUDA port: ~120-180
    (SMALL) — one new `scalar_profile_column` kord branch, nq-batched
    reuse wiring, a new `q` H2D/D2H wrapper, bridge/dispatch selector.
    Performance ceiling estimate: REMAP_Q's 116.8 ms/call C512 baseline,
    single-field port (unlike REMAP_UV's 2-field port) — a PT/DELZ-
    magnitude improvement (REMAPPING -33.7%/TOTAL -7.3% precedent) is
    the more defensible ceiling than REMAP_UV's larger multiple,
    contingent on the new `q` transfer overhead staying small. **Result:
    REMAP_Q HARNESS READY — PROCEED TO CUDA PORT.** No production CUDA
    source was written or changed in this prompt. See
    `docs/performance/p84b_remap_q_cuda_port_plan_2026-09-25.md`.

---

CUDA PERFORMANCE ROADMAP COMPLETE — READY FOR PROFILING PHASE.
TRACER CUDA DISPATCH LINE OF WORK CLOSED (Prompt 83): three consecutive
tracer-CUDA optimization attempts (unbatched P77/78, batched P79/81B,
host-copy-fix P82/83) all failed to recover or reverse the TRACER_2D
regression; per Prompt 83's binding instruction, no further CUDA tracer
production optimization prompt should be authorized automatically.
Fortran remains the production default for tracer dispatch.
REMAP_UV CUDA IS NOW THE PRODUCTION DEFAULT (Prompt 84A) — 10.23x
faster, bitwise-exact, no longer requires an opt-in env var.
REMAP_Q CUDA PORT IMPLEMENTED, UNIT-VALIDATED, BUT VALIDATION BLOCKED
AT INTEGRATION (Prompt 85): the kord=8 numerics + batched single-H2D/
single-D2H `q` transfer + narrow `use_cuda_remap_q` selector were
implemented per Prompt 84B's plan and passed all 23 unit fixtures plus
a sanitizer-clean production-shape smoke test, but a real C32
integration A/B found the CUDA path never dispatches — the kernel's
deliberate `nq==1`-only scope (from Prompt 84B/85's own Stage-0/1
contract) never matches production's actual tracer count (`nq>=2`,
confirmed via this run's `FV3 ncnst: 2`). One genuine selector bug
(`kord_tr(1)==9` should be `==8`) was found and fixed en route and
confirmed not to be the blocker. C512/performance stages were not run
— the `nq` mismatch is structural and would reproduce identically at
any resolution. Default dispatch
unaffected (Fortran remains default; `use_cuda_remap_q` is unreachable
under any real tracer count regardless of its opt-in env var). See
`docs/performance/p85_remap_q_cuda_implementation_2026-09-25.md`.
**Prompt 86A mapped the real `nq=2` case (sphum kord=8, cld_amt
kord=9-forced) and found it a clean, scoped extension, NOT a dead
end** — both kord=8 and kord=9 branches already exist in the shared
`scalar_profile_column` with independent validation precedent (kord=8
via this kernel, kord=9 via REMAP_UV/P84A); `q`'s contiguous layout
keeps one batched H2D/D2H clean at 2x bytes; recommended design is
exactly 2 fixed kernel launches (one per tracer/kord) sharing that
transfer — not a generic multi-tracer loop. A host-only `nq=2`
mass-conservation/positivity test PASSED. **REMAP_Q NQ2 SCOPE CLEAN —
PROCEED TO CUDA EXTENSION** (mapping-only; no CUDA/production code
changed by Prompt 86A). See
`docs/performance/p86a_remap_q_nq2_scope_2026-09-25.md`.
**Prompt 86B implemented the `nq=2` extension (two fixed kernel
launches, sphum kord=8 then cld_amt kord=9, one shared H2D/D2H) and
GPU-validated it in isolation** — a new production-shaped smoke test
found both tracers bitwise-exact vs. host oracle, mass-conserved,
sanitizer-clean, `H2D_CALLS=1 D2H_CALLS=1` confirmed. **C32 integration
is again BLOCKED, for a new, more fundamental reason than Prompt 85's**:
`FV_StateMod.F90:563` unconditionally forces `kord_tr=9` for every GEOS
FV_State (not the `8` Prompts 85/86A assumed from the GFDL bare-code
default) — so real production's tracer schedule is `kord_tracer=(9,9)`,
not `(8,9)`. The selector's `kord_tr(1)==8` precondition is therefore
never true; CUDA dispatch count was 0/0 in a real C32 A/B (job pair
58560147/58560148), independent of the `RUN_CUDA_FV3_REMAPQ_ENABLE`
opt-in. This also retroactively corrects Prompt 85 §6's claim that
`kord_tr`'s default of 8 was "not overridden" — it already was, visible
in Prompt 85's own C32 run log, just not checked. C512/performance
stages not attempted (blocked upstream). Default dispatch unaffected
(Fortran remains default). Recommendation deferred to a human/
orchestrator decision: redesign the kord schedule for the real `(9,9)`
case (new scope, not authorized here), or leave the implemented-but-
unreachable code as zero-risk infrastructure. See
`docs/performance/p86b_remap_q_nq2_cuda_implementation_2026-09-25.md`.
**Prompt 87 adapted REMAP_Q CUDA to the real `(9,9)` kord schedule
found by Prompt 86B and re-attempted C32 integration** — corrected every
kord1==8 gate/literal to kord1==9 across `fv_mapz.F90`'s selector and
the CUDA/C++ side, including a **critical hardcoded `8_c_int32_t`
literal at the actual Fortran ABI call site** in
`cuda/interface/fv3_gpu_dyncore_bridge.F90` (a distinct bug from
Prompt 86B's doc-level staleness, not previously caught — would have
silently defeated the kord=9 correction on its own). Re-validated the
kord=(9,9) fixture and GPU smoke test bitwise-exact/mass-conserved/
sanitizer-clean, and REMAP_UV/map_scalar/kord=8-regression coverage
all still PASS. Rebuilt and re-linked the authoritative standalone
binary across all three build trees involved (this repo's OpenMPI dev
build, this repo's Intel-MPI build actually linked by the Fortran
executable via `FV3_CUDA_LIBRARY_DIR`, and the Fortran build/link
itself — the first Fortran rebuild was caught, via `strings`, silently
linking a stale, not-yet-rebuilt CUDA archive), with full provenance
(sha256 `c43845c76ea9c6845bd04816a28de9f664696deb9f4ed4f1f14b7a6e77d1b84b`).
**A real C32 A/B (jobs 58562204/58562205) found CUDA REMAP_Q dispatch
still 0/36 in the opt-in arm.** `kord_tr(1)==9` is now empirically
confirmed true (`FV3 kord_tr: 9` in the log), so this prompt's kord fix
is necessary but not sufficient: a **second, independent** unconditional
GEOS forcing was found at `FV_StateMod.F90:568` —
`flagstruct%fill = .true.` for every FV_State (confirmed `FV3 fill: T`
in both arms) — which the CUDA REMAP_Q selector's `.not. fill` clause
rejects regardless of kord. Per the task's explicit Stage-8 instruction
("if CUDA still does not dispatch: STOP and identify exact remaining
gate"), this is a STOP, not a further fix attempt; C512/performance
stages (9-12) were correctly not attempted. Default dispatch unaffected
(Fortran remains default, zero risk). Recommendation deferred to a
human/orchestrator decision, same shape as Prompt 86B's but now
covering both `kord_tr` and `fill`: (a) redesign CUDA REMAP_Q to
support `fill=.true.` (new scope, not authorized here), (b) leave the
implemented/GPU-validated-but-unreachable code as zero-risk
infrastructure, or (c) first verify whether *any* runnable config in
this project ever produces `fill=.false.` (unverified). See
`docs/performance/p87_remap_q_actual_kord9_production_2026-09-25.md`.
**Prompt 88A resolved option (c) above with a bounded, no-implementation
source-mapping task.** Exhaustively searched every writer of
`flagstruct%fill` (both repos) and every runnable config's `input.nml`/
job template/`.rc` files (57 experiment directories, all C8-C512,
Fortran/CUDA, R4/R8, HS/non-HS): **`fill=.false.` was found in zero
runnable configurations** — `FV_StateMod.F90:568`'s unconditional
override survives untouched into every checked production/HS/gate run
(the one other writer, `geos-gtfv3/driver/FV_State_Utilities.F90:74`, is
confirmed unreachable — a separate `fv3_driver.x` build target never
linked into `StandAlone_FV3_Dycore.x`). Traced `fill`'s numerical role to
a single call site (`mapn_tracer`, `fv_mapz.F90:2291`,
`if (fill) call fillz(...)`): `fillz` is a bounded, per-column,
per-tracer, mass-conserving negative-tracer-value post-fix (top/
interior/bottom neighbor-borrowing, then a conditional final non-local
rescale-and-clip that only fires when a column's aggregate mass is
still net-positive) — cleanly separable from REMAP_Q's core PPM remap
math, with no cross-tracer coupling and no moisture/thermodynamic
interaction. Read the CUDA REMAP_Q kernel and found its two required
inputs (`q2_out`, `dpeO`) are already fully device-resident, per-thread
local arrays at the exact point a `fillz`-equivalent post-loop would
run, before the kernel's existing single write-back — meaning the
existing 1H2D+1D2H transfer architecture is fully preservable, no new
residency needed. Built and validated (9/9 PASS, repeatable) a new
host-only golden-reference test for `fillz`
(`test_fillz_host_reference.cpp`, wired into CMake as ctest target
`fv3_wp_c_fillz_host_reference`) as the required fill=true numerical
target — no existing host harness had covered it before. Estimated port
size **SMALL** (one new small per-thread helper function, no new PPM/
pressure-interval work, no new transfers, a one-line Fortran selector
change). **Verdict: REMAP_Q FILL=TRUE SCOPE CLEAN — PROCEED TO CUDA
EXTENSION.** No CUDA/production source was changed; this remains a
decision point pending explicit authorization to implement. See
`docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md`.

22. **(Prompt 88B — REMAP_Q `fill=true` CUDA IMPLEMENTED, GPU EXECUTED,
    NUMERICALLY VALIDATED, PERFORMANCE VALIDATED POSITIVE)** Implemented
    the CUDA extension P88A scoped: a device-portable `fillz_column<T>`
    helper (mechanical port of `fv_fill.F90`'s `fillz`, 9/9 bitwise vs.
    the P88A golden oracle, 9/9 GPU-EXECUTED bitwise via a dedicated
    `__global__` smoke kernel), integrated into REMAP_Q's kernel as a
    post-`map_scalar_column` call reusing the dead `c2` scratch slot —
    zero new device memory, H2D/D2H calls confirmed still 1/1. Removed
    the `.not. fill` selector term in `fv_mapz.F90` in favor of an
    opt-in `RUN_CUDA_FV3_REMAPQ_ENABLE` gate (default off;
    `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN` escape hatch retained). Full
    nq=2/kord=(9,9)/fill=true GPU fixture: round-off-clean, mass-
    conserving, 0 NaN/Inf, 0 illegal negatives. Regression 28/30 PASS
    (2 pre-existing login-node GPU-absence failures, out of scope),
    `compute-sanitizer` 0 errors. Rebuilt+relinked the authoritative
    binary (sha256 `eabd1c54...5258`). **Real C32 A/B: CUDA REMAP_Q
    dispatch 36/36, 0/36 fallback** — the first successful dispatch in
    this entire line of work, resolving both the Prompt 87 `kord_tr`
    gate and the `fill` gate. **Real C512 same-node paired A/B (job
    58566492, 3 pairs, 6/6 arms rc=0): REMAP_Q ~3.23x direct speedup
    (116.5 ms/call Fortran vs. 36.0 ms/call CUDA), REMAPPING -11.98%,
    FV_DYN_LOOP -3.47%, TOTAL -1.51%**, consistent across all 3 pairs
    (no regression in any pair). **Verdict (P88B): KEEP AS OPT-IN
    INFRASTRUCTURE** — production default remained Fortran REMAP_Q per
    that task's explicit instruction not to flip it. See
    `docs/performance/p88b_remap_q_fill_true_cuda_2026-09-25.md`.

23. **(Prompt 89 / Prompt 89-Resume — REMAP_Q default-flip evidence gate
    and closure, CLOSED, DEFAULT NOW CUDA).** Independently confirmed
    P88B's C512 result twice more: Stage 1 paired A/B (5 pairs) — CUDA
    wins TOTAL 5/5 (median +1.72%, mean +2.65%), REMAP_Q ~3.25x; Stage 6
    60-call longer run (job 58571255) — TOTAL +1.41%, REMAP_Q 3.31x. All
    6 default-policy gate criteria passed (correctness, ≥4/5 wins,
    ≥1.0% median/mean TOTAL, clear REMAP_Q speedup, no instability).
    Edited `fv_mapz.F90`'s `use_cuda_remap_q` selector so CUDA is now
    default-on whenever the base gate holds (`kord_tr==(9,9)`, `nq==2`,
    R8, CUDA backend enabled); `RUN_CUDA_FV3_REMAPQ_ENABLE` is now a
    no-op kept for compatibility; `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`
    is the explicit escape hatch. Rebuilt under the production Intel-MPI
    stack (sha256 `82c1e9b6...9fae0`, supersedes P88B's `eabd1c54...
    1258`); CUDA archives unchanged (`libfv3_dyn_core_gpu.a` sha256
    identical to P88B's). Real C32 selector smoke: default arm dispatch
    **36/36** CUDA with **no override set**; `RUN_CUDA_FV3_REMAPQ_FORCE_
    FORTRAN=1` arm dispatch **0/36** CUDA (36/36 Fortran) — both rc=0,
    0 errors. **Verdict: CUDA REMAP_Q IS NOW THE PRODUCTION DEFAULT**
    (mirrors Prompt 84A's REMAP_UV precedent). See
    `docs/performance/p89_remap_q_default_policy_2026-09-25.md`.

24. **(Prompt 90 — post-port C512 hotspot profile refresh, bounded
    profiling-only, next target selected).** Verified (not assumed) that
    PT/DELZ, REMAP_UV, and REMAP_Q selectors are all CUDA-default with
    zero required overrides in the real `fv_mapz.F90`, and that
    `use_cuda_tp2d` (`fv_tracer2d.F90`) is structurally inert in
    production because `hord`!=8 there -- resolving an apparent
    inconsistency without contradicting the P83 tracer STOP. Ran one
    real 60-call C512 job (58593086, node `warpa004`, zero env
    overrides = pure production defaults): rc=0, 60/60 `DYN Date`, 0
    FATAL/NaN/Inf. Per-call timer values for every already-ported stage
    matched that stage's known *CUDA*-side figure (REMAP_T 50.7,
    REMAP_NH 127.4, REMAP_UV 16.9, REMAP_Q 35.3 ms/call) -- confirming
    all three are actively dispatching CUDA, not merely default-true in
    source. **New finding**: with those three ports CUDA-dispatching,
    the two largest never-attempted Fortran hotspots inside `REMAPPING`
    are now `REMAP_PRESSUREVARS` (131.4 ms/call, 26.3% of REMAPPING,
    zero MPI/global-reduction -- purely per-column, `fv_mapz.F90:
    1239-1516`) and `REMAP_TOTALENERGYCON` (129.3 ms/call, 25.9%,
    contains 5x `mpp_global_sum` cross-tile reduction calls,
    `fv_mapz.F90:1521-1773`) -- nearly tied in raw cost but materially
    different engineering risk. TRACER_2D remains the largest single
    Fortran-side stage overall (430.7 ms/call) but stays out of scope
    (P83 STOP, unaffected). `nsys` capture skipped (timer data alone
    sufficient to distinguish candidates); Nsight Compute not used
    (forbidden this task). MOIST_CAPPA/USE_COND, `a_imp<=0.999`, and
    MPS/fusion operational wiring reassessed against the fresh profile
    -- all three unchanged/still open, none started. **Selected next
    target: `REMAP_PRESSUREVARS`** -- same per-column shape as every
    prior successful port (PT/DELZ, REMAP_UV, REMAP_Q), zero MPI risk,
    preferred over the similarly-sized but `mpp_global_sum`-bearing
    `REMAP_TOTALENERGYCON`. No numerical code changed; no production
    defaults changed; PT/DELZ/REMAP_UV/REMAP_Q/R4 not touched;
    tracer-CUDA optimization not reopened. Recommended next prompt:
    mapping-only survey of `REMAP_PRESSUREVARS` (same pattern as
    P84B/P88A), before any implementation. See
    `docs/performance/p90_post_remap_cuda_hotspot_profile_2026-09-27.md`.

25. **(Prompt 91 -- REMAP_PRESSUREVARS mapping + authoritative host
    harness, bounded, NO CUDA implementation, GO decision).** Full
    17-stage source mapping of `fv_mapz.F90:1239-1516` by direct read (not
    inferred): of the 278-line timer body only ~50 lines are numerically
    active in production (unconditional pe2/pn2/pk2/delp recompute plus
    the remap_t+MOIST_CAPPA cappa/pkz block); hydrostatic, remap_te,
    remap_pt, and do_omega branches are all confirmed DEAD (production
    values false); `fill`/`kord`/`a_imp`/`k_split`/`n_split`/`gmao_remap`
    confirmed IRRELEVANT (never read in an active statement). Zero MPI/
    halo/collective calls confirmed by full-body read. Data residency:
    HOST ONLY at entry, same starting condition as every existing port
    (PT/DELZ/REMAP_UV/REMAP_Q each complete their own self-contained H2D/
    kernel/D2H round trip before returning to Fortran); a new port would
    need an equivalent self-contained round trip, not a novel design.
    CUDA reuse potential VERY HIGH: the pe2/pn2/pk2/delp pattern and the
    dry-degenerate `cappa==akap` pkz formula are numerically identical to
    code already implemented in `fv3_pt_delz_remap.cu`. **Correction to a
    pre-existing project convention**: `cappa==akap` actually requires
    both `nwat=0` AND `q(sphum)=0`, not `nwat=0` alone (`fv_mapz.F90:
    1430`) -- the latter holds today only via the separately-documented
    cold-start tracer-zero limitation (CURRENT_STATE Sec.7); not a new
    risk, but a materially more precise statement of an assumption every
    existing CUDA vertical-remap port already implicitly relies on. Built
    a new host-only golden-reference harness
    (`cuda/dyn_core/tests/vertical/test_remap_pressurevars_host_reference.cpp`,
    Option C exact transcription -- no standalone subroutine exists to
    call directly, same precedent as `test_fillz_host_reference.cpp`),
    10 fixtures covering production values plus boundary/stress/repeat-
    ability cases, transcribing the general q(sphum)-dependent formula
    (not hard-coded to the degenerate case) so the degenerate identity is
    itself a checked invariant. Verified 10/10 PASS standalone (g++) and
    via the project's real CMake/CTest build under the authoritative
    Intel-MPI stack (1/1 ctest PASS); confirmed zero regression via a
    full `cmake --build . -j8` of the whole target. Estimated performance
    ceiling: ~90.7 ms/call recoverable (~18% of REMAPPING, ~5% of
    FV_DYN_LOOP, ~2% of TOTAL) at a REMAP_Q-comparable 3.23x speedup
    floor. **Decision: GO -- proceed to CUDA port** in a follow-up prompt;
    proposed unit is one new per-column device function plus a thin ABI
    wrapper following the `fv3_gpu_pt_delz_remap` self-contained H2D/D2H
    pattern. No production source changed; no PT/DELZ/REMAP_UV/REMAP_Q/
    tracer-CUDA/REMAP_TOTALENERGYCON/MOIST_CAPPA-numerics/a_imp/R4
    changes; no profiling job run; no nsys/ncu used; no commit/push. See
    `docs/performance/p91_remap_pressurevars_cuda_port_plan_2026-09-27.md`.

26. **(Prompt 92 -- REMAP_PRESSUREVARS CUDA implementation, GPU-validated,
    performance POSITIVE, opt-in only).** Implemented the CUDA port
    proposed by Prompt 91: per-column device kernel + thin ABI wrapper,
    reusing `fv3_pt_delz_remap.cu`-style arithmetic; supports the general
    `q(sphum)`-dependent MOIST_CAPPA formula (not just the degenerate
    `cappa==akap` case). 10/10 host-authoritative fixtures PASS on GPU
    (essentially bitwise, `pk` ~1 ULP round-off only); compute-sanitizer
    0 errors on a production-shape smoke test. New selector
    `use_cuda_remap_pressurevars` (`RUN_CUDA_FV3_PRESSUREVARS_ENABLE`/
    `_FORCE_FORTRAN`/`_DUMP_EXIT`, same convention as REMAP_Q) --
    **Fortran remains the production default**, CUDA is opt-in only, per
    this prompt's binding scope. Real C32 Held-Suarez integration: 36/36
    vs 0/36 dispatch, rc=0 both arms, no divergence. Real 5-pair
    same-node C512 paired A/B: direct speedup ~3.50x (131.64 ->
    37.66 ms/call), REMAPPING -15.75%, FV_DYN_LOOP -4.84%, TOTAL -1.81%,
    consistent across all 5 pairs, closely matching Prompt 91's own
    a-priori ceiling estimate. **Decision: PERFORMANCE POSITIVE -- KEEP
    AND CONSIDER DEFAULT ENABLE** (default-flip deferred to a future
    prompt, mirroring REMAP_Q's own two-step Prompt 88B/89 precedent). No
    PT/DELZ/REMAP_UV/REMAP_Q/tracer-CUDA/REMAP_TOTALENERGYCON/
    MOIST_CAPPA-policy/`a_imp`/R4 changes; no commit/push. See
    `docs/performance/p92_remap_pressurevars_cuda_implementation_2026-09-27.md`.

27. **(Prompt 93 -- REMAP_PRESSUREVARS default-policy flip, CLOSED).**
    Statistical robustness gate over Prompt 92's own 5 same-node pairs:
    5/5 CUDA-positive, TOTAL improvement mean 1.805%/median 1.633%/
    stddev 0.337%, REMAP_PRESSUREVARS direct speedup mean 3.498x --
    **Stage 2: DEFAULT CANDIDATE**. Independent longer 60-call same-node
    A/B confirmation (job 58618735): REMAP_PRESSUREVARS 131.95 ->
    41.31 ms/call (3.194x), REMAPPING +10.887%, FV_DYN_LOOP +4.086%,
    TOTAL +1.676%, DYN_CORE unchanged -- **Stage 4: DEFAULT CONFIRMED**
    (>=1.0% threshold, fixed before the run, not tuned after). Flipped
    `use_cuda_remap_pressurevars` to the CUDA production default in
    `fv_mapz.F90` (mirrors REMAP_Q's own Prompt 88B->89 selector-flip
    precedent exactly); `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`
    remains the escape hatch; base configuration gate (non-hydrostatic,
    `remap_t`, not `remap_pt`/`remap_te`/`do_omega`, `nwat==0`, R8)
    unchanged, so unsupported configurations still auto-fall-back to
    Fortran. Rebuilt under the canonical Intel-MPI production stack
    (sha256 `b482b647...9b3ec96`, Intel MPI 2021.13 linkage confirmed via
    `ldd`); C32 selector smoke PASS both arms (DEFAULT: 36/36 CUDA
    dispatch; `FORCE_FORTRAN`: 0/36 CUDA, 36/36 Fortran). No numerical
    code, kernel, or transfer architecture changed; no PT/DELZ/REMAP_UV/
    REMAP_Q/tracer-CUDA/REMAP_TOTALENERGYCON/R4/`ncu` touched; no
    commit/push. **REMAP_PRESSUREVARS is now a CUDA production default.**
    See `docs/performance/p93_remap_pressurevars_default_policy_2026-09-28.md`.

28. **(Prompt 94 -- post-PRESSUREVARS-default C512 hotspot profile
    refresh, next target selected, CLOSED -- profiling/mapping only, no
    implementation).** Reran the Prompt 90 zero-override C512 profiling
    methodology (job 58619479, node `warpa003`, rc=0, 60/60 calls) now
    that PT/DELZ, REMAP_UV, REMAP_Q, and REMAP_PRESSUREVARS are all CUDA
    production defaults. CUDA dispatch confirmed by direct counter (not
    inferred from selector defaults) via a supplementary short C32
    dispatch-verification smoke (job 58619656, all four `*_DUMP_EXIT`
    flags set, zero force-fortran overrides): 36/36 dispatch for all
    four. `REMAP_PRESSUREVARS`: 131.4 ms/call (Fortran, Prompt 90) ->
    38.11 ms/call (CUDA, this run) -- 3.448x, matching Prompt 93's
    3.194-3.498x paired range. With that shift, `REMAP_TOTALENERGYCON`
    (150.16 ms/call, unchanged, never attempted) is now unambiguously
    the single largest `REMAPPING` sub-timer -- confirmed by fresh
    measurement, not assumed from Prompt 90's near-tie. Stage 8
    re-verified `REMAP_TOTALENERGYCON`'s `mpp_global_sum` structure by
    direct source read (`fv_mapz.F90:1612-1864`, shifted from Prompt 90's
    cited 1521-1773 by P92/93's inserted dump-diagnostic lines, same
    content): 5 call sites in source (matches Prompt 90's static count
    exactly), but only 2 execute per production call (`tesum`+`zsum`,
    non-hydrostatic branch) -- a 2-scalar reduction, not a full-field
    collective. The bulk of the 150.16 ms/call cost is attributable to
    two large per-column loops structurally identical to the four
    already-ported remap kernels; **likely bottleneck: COMPUTE, not
    COMMUNICATION** (GPU-clean local compute + a cheap, bounded
    reduction, not a fundamentally communication-bound target). No
    `nsys`/`ncu` run (timer/source evidence sufficient). MOIST_CAPPA/
    USE_COND and `a_imp<=0.999` reclassified CORRECTNESS COVERAGE
    (unchanged, not exposed by this profile); MPS/fusion wiring
    OPERATIONAL INFRASTRUCTURE (unchanged). **NEXT TARGET:
    `REMAP_TOTALENERGYCON`, NEXT TASK TYPE: MAP/HARNESS** (a Prompt-91-
    style mapping+harness prompt, mirroring the precedent of mapping
    before implementation -- this target introduces a first-of-kind
    kernel-reduction-kernel pattern not used by any prior port). No
    numerical/source changes; no new CUDA ports; no production defaults
    changed; tracer-CUDA not reopened; R4 not touched; no commit/push.
    See `docs/performance/p94_post_pressurevars_cuda_hotspot_profile_2026-09-28.md`.

29. **(Prompt 95 -- REMAP_TOTALENERGYCON mapping + harness, CLOSED --
    MAP/HARNESS only, no implementation, decision: BLOCKED BY
    COMMUNICATION, corrects Prompt 94's Stage 8).** Fully mapped
    `Remap_TotalEnergyConsv` (`fv_mapz.F90:1612-1875`): confirmed
    production-active branches (non-hydrostatic pre-reduction `te_2d`/
    `zsum1` compute with MOIST_CAPPA numerically degenerate to `cv_air`
    since `nwat=0`; `adiabatic=.true.` post-reduction `pt=pt+dtmp*pkz`
    correction, confirmed via the C512 HS log's "Running In Adiabatic
    Mode"; only 2 of 5 static `mpp_global_sum` sites execute, matching
    Prompt 94). Added fine-grained test-only `fv_timing_mod`
    instrumentation (`TECons_Pre`/`TECons_RedMPI`/`TECons_Post`, the
    latter nested to isolate the `mpp_global_sum` calls themselves from
    surrounding local array-weighting compute) and measured a real C512
    run (`GEOSfvdycore_fortran_perf` checkout, 6 ranks/1 A100, 60
    calls): the two `mpp_global_sum` reductions alone cost ~135 ms/call
    -- **~89% of the full 151.63 ms/call timer**, not a small fraction.
    Root cause: `flags=sflag` resolves to `BITWISE_EXACT_SUM`
    (`flagstruct%exact_sum=1`, confirmed live via the log), FMS's
    order-independent bit-reproducible summation algorithm, substantially
    more expensive than a plain fast reduction. **This directly
    contradicts and supersedes Prompt 94 Stage 8's "likely bottleneck:
    COMPUTE, not COMMUNICATION" conclusion**, which was based on static
    source structure only, not direct measurement. Local compute
    (pre-+post-reduction) is only ~13% (~19.75 ms/call) -- capping any
    CUDA local-compute-only port at an estimated ceiling of ~0.5% TOTAL /
    ~4.5% REMAPPING, far below every prior successful port. Built and ran
    a host-only golden-reference harness
    (`test_remap_totalenergycon_host_reference.cpp`, 21 fixtures across
    Unit A/Bridge/Unit B/end-to-end composition): **21/21 PASS**,
    bitwise-repeatable, all invariants hold. Per the task's own explicit
    stop criterion ("STOP if reductions dominate the timer") and the
    "do not redesign MPI/global reductions" constraint, **decision:
    BLOCKED BY COMMUNICATION** -- no CUDA implementation recommended for
    this target. No production source changes; no PT/DELZ/REMAP_UV/
    REMAP_Q/REMAP_PRESSUREVARS/tracer-CUDA touched; no MPI/reduction
    redesign attempted; R4 not touched; no commit/push. See
    `docs/performance/p95_remap_totalenergycon_cuda_port_plan_2026-09-28.md`.

30. **(Prompt 96 -- decompose remaining FV_DYN_LOOP/DYN_CORE hotspots,
    CLOSED -- ANALYSIS ONLY, no implementation).** Reused Prompt 94's C512
    job 58619479's existing logs (no new Slurm run). Found `fv_timing_mod`'s
    "fixed 14-timer set" (used by every prior prompt, P69-95) was an
    artifact of the sbatch script's own `grep` filter, not the
    instrumentation itself; confirmed `dyn_core.F90`'s own `c_sw`/`d_sw`/
    `Riem_Solver`/`PG_D`/etc. sub-timers never execute under
    `RUN_CUDA_FV3=1` (the CUDA backend dispatches the whole WP-A/B/C/D
    subsystem as one opaque call) -- `DYN_CORE`'s 845.85 ms/call has no
    further Fortran-visible breakdown. Discovered a **separate,
    previously-unexamined MAPL_Profiler GridComp-level timer tree**
    (`DynCore_GridCompMod.F90`, distinct from `fv_timing_mod`) revealing
    `DYN_EPILOGUE` at **688.67 ms/call** -- larger than `TRACER_2D` or
    `REMAPPING` -- living entirely outside `FV_DYN_LOOP`'s scope;
    source-mapped it as MPI-free (grep-confirmed zero `mpp_`/`global_sum`
    across ~1000 lines) but dominated by calls into the external `MAPL`
    library (`getAllWinds`/`getVorticity`/`getDivergence`/`getEPV`/
    `getOmega`), not this project's own source. Also source-mapped
    `State_To_FV` (112.83 ms/call, contains real `mpp_get_boundary` halo
    communication -- HIGH RISK) and `FV_To_State` (100.50 ms/call,
    grep-confirmed MPI-free, fully project-owned, `FV_StateMod.F90:2704-2780`).
    Confirmed `FV_DYN_LOOP` itself is already fully (over-)explained by
    `DYN_CORE`+`TRACER_2D`+`REMAPPING` (net unexplained ~=0, within ~5.8%
    known timer-overlap noise) -- the real newly-found time sits outside
    `FV_DYN_LOOP`, at the MAPL GridComp level. Classified `BITWISE_EXACT_SUM`
    as CORRECTNESS/POLICY (confirmed real, namelist-configurable via
    `fv_core_nml`'s `exact_sum`, default `1`; changing it would alter
    numerical reproducibility) -- no change made. **Selected next task:
    MAP/HARNESS on `FV_To_State`** (cleanest of the three: MPI-free AND
    fully project-owned, unlike `DYN_EPILOGUE`'s external-library
    dependency or `State_To_FV`'s embedded halo call). No CUDA
    implementation; no numerical/production changes; no commit/push. See
    `docs/performance/p96_remaining_dyncore_hotspot_decomposition_2026-09-28.md`.

31. **(Prompt 97 -- FV_To_State mapping + harness, CLOSED -- MAP/HARNESS
    only, no implementation, decision: KEEP ON CPU).** Fully mapped
    `FV_To_State` (`FV_StateMod.F90:2704-2780`, sole live call site
    line 2459, once per timestep, unconditional): confirmed
    production-active branches (`hydrostatic=.false.` -> W/DZ copies
    active; `SW_DYNAMICS=.false.`, structurally dead for any `npz>1`
    config -> PE-transpose/PT/DZ/`fv_getPKZ`/PT-divide block active);
    confirmed by direct body grep that MOIST_CAPPA/USE_COND/do_omega/
    a_imp/nwat/nq/grid_type/halo/stagger tokens are entirely absent
    from this routine's source (not merely inactive); confirmed zero
    `mpp_`/halo/collective calls. Mapped all 7 field
    copies/transforms (U/V/W/PT/DZ pure copies, PE the one genuine
    `(i,k,j)`->`(i,j,k)` transpose, PKZ the one derived-compute field
    via the inlined dry-`akap` `fv_getPKZ`). **Decisive residency
    finding**: the FV source fields are already HOST AUTHORITATIVE at
    entry -- not because this routine syncs them, but because every
    existing CUDA remap/dyn_core dispatch call
    (`cuda/interface/fv3_gpu_dyncore_bridge.F90`) already performs its
    own D2H as part of its per-call boundary protocol -- and the
    destination `STATE%VARS%*` is consumed immediately afterward by
    the host/MAPL-only `DYN_EPILOGUE` block (direct source trace of
    `DynCore_GridCompMod.F90`'s `DynRun`->`Write_Profile`/`getAllWinds`
    sequence). A GPU port would therefore add a redundant H2D+D2H
    round trip around work that is already host-to-host on both ends,
    not eliminate any existing transfer. Built and ran a host-only
    golden-reference harness
    (`cuda/dyn_core/tests/host/test_fv_to_state_host_reference.cpp`,
    Option C exact transcription, registered in CMakeLists.txt as
    `fv3_wp_c_fv_to_state_host_reference_test`): 8 fixtures including
    the mandatory sentinel-layout-transposition check (bijective
    position-encoded values proving no accidental axis swap in the PE
    transpose); 2 of the 10 requested categories (tracers/q, nonzero
    sphum) explicitly documented N/A since this routine references
    neither. **8/8 PASS, 0 NaN/Inf, bitwise-repeatable.** Analytical
    (not measured -- residency argument judged dispositive, no new
    Slurm run) byte-volume estimate: ~518.9 MB/call, ~5.16 GB/s
    effective bandwidth at the measured 100.50 ms/call baseline (P94/
    96), MIXED copy/layout/compute split. **Decision: KEEP ON CPU --
    MEMORY/STATE BOUND** -- FV_To_State is a host-to-host state-
    materialization routine sitting immediately adjacent to an
    existing H2D/D2H sync boundary one layer up in the CUDA bridge;
    porting it would not remove that materialization, only duplicate
    it. Estimated `TOTAL` impact of a hypothetical port: negligible to
    slightly negative (FV_To_State is only ~2.6% of `TOTAL`, and any
    direct-compute speedup is plausibly offset by the new transfer
    overhead). No CUDA/production source changed; no selector/default
    changes; no commit/push. See
    `docs/performance/p97_fv_to_state_cuda_port_plan_2026-09-28.md`.

32. **(Prompt 98 -- State_To_FV mapping + harness, CLOSED -- MAP/HARNESS
    only, no implementation, decision: GO -- HARNESS READY).** Fully
    mapped `State_To_FV` (`FV_StateMod.F90:2519-2702`, called once per
    `k_split` sub-iteration before `fv_dynamics()`, the structurally
    opposite direction from Prompt 97's `FV_To_State`). Confirmed
    production-active branches (`grid_type=0` -> interior D-grid U/V
    copy + `mpp_get_boundary` edge fill active, doubly-periodic branch
    dead; `SW_DYNAMICS=.false.` -> PE-transpose/`peln`/`pk`/`ps`/`delp`
    block active; `hydrostatic=.false.` -> W/DELZ copies active;
    `ADJUST_SPLITS=.false.` -> dead). Added TEST-ONLY `MPI_Wtime()`
    cost-split instrumentation (pre-boundary/boundary/post-boundary),
    measured on a real C512/L10 job (58622163, 366 samples,
    cross-validated 0.2% against the independently-measured
    MAPL_Profiler `State_To_FV` timer of 113.7 ms/call), then fully
    reverted (grep-confirmed 0 residue; binary rebuilt clean, sha256
    `9ae5d7cf...951c3`, 0 `P98_TIMING` strings via `strings`).
    **Decisive finding**: unlike its "HIGH RISK" label in Prompt 96
    (based on the mere presence of an embedded `mpp_get_boundary`
    call, not measurement), `State_To_FV` is actually
    **LOCAL-COMPUTE-DOMINATED (~84% local: 8.8% pre-boundary U/V copy +
    75.2% post-boundary PE-transpose/`peln`/`pk`/`ps`/`delp`/`pt`/
    `delz`/`pkz`; only ~16% is the `mpp_get_boundary` call itself,
    rank-imbalanced/latency-bound, not bandwidth-bound at ~40KB/buffer)
    ** -- the opposite residency direction from `FV_To_State` too: this
    routine's output feeds the already-GPU-resident `fv_dynamics()`
    CUDA path, so a port here has the potential to remove an existing
    transfer rather than add one. Direct CUDA-kernel reuse precedent
    identified: the `peln=log(pe)`/`pk=exp(akap*peln)` pattern already
    exists in the production REMAP_PRESSUREVARS kernel (P90-93). Built
    and ran a host-only golden-reference harness
    (`cuda/dyn_core/tests/host/test_state_to_fv_host_reference.cpp`,
    Option C transcription of production-active branches only,
    `mpp_get_boundary` itself explicitly out of scope as Category-D
    communication, registered in CMakeLists.txt as
    `fv3_wp_c_state_to_fv_host_reference_test`): 8 fixtures including
    the mandatory sentinel-layout-transposition check; 2 of the 10
    requested categories (tracers/q) explicitly documented N/A since
    this routine references neither. **8/8 PASS, 0 NaN/Inf,
    bitwise-repeatable, 0 compile warnings.** Estimated `TOTAL` impact
    of a future Option-A port (local compute only, halo call stays
    host-side unchanged): ~+1.0-1.5% TOTAL, comparable to REMAP_UV/
    REMAP_PRESSUREVARS's shipped gains. **Decision: GO -- HARNESS
    READY, recommended next CUDA implementation target** (pending a
    separate authorized implementation prompt; no CUDA source added in
    this prompt). No CUDA/production source changed (instrumentation
    added then fully reverted); no selector/default changes; no
    commit/push. See
    `docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md`.

33. **(Prompt 99 -- State_To_FV CUDA implementation, CLOSED -- decision:
    VALIDATION BLOCKED).** Implemented the Option-A CUDA port
    recommended by Prompt 98: Phase A (u/v tiny_number fill + interior
    copy, before `mpp_get_boundary`) and Phase C (w copy, PE transpose,
    `peln`/`pk`, `ps` pack, `delp`, `pt`, `delz`, `pkz`, after
    `mpp_get_boundary`) CUDA kernels
    (`cuda/dyn_core/vertical/fv3_state_to_fv.{cuh,cu}`), runtime
    dispatch (`fv3_gpu_runtime.cpp`), and a selector
    (`use_cuda_state_to_fv`, opt-in via
    `RUN_CUDA_FV3_STATE_TO_FV_ENABLE`, escape hatch
    `RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN`, production default
    Fortran, unchanged). `mpp_get_boundary` (Phase B) untouched. Stage 9
    GPU smoke test (8/8 PASS, bitwise, real device) and Stage 13 C32
    integration (6 ranks, both arms `COMPLETED`/`ExitCode 0:0`, no
    FATAL/NaN/Inf, field agreement at round-off) both GPU EXECUTED and
    NUMERICALLY VALIDATED. Also root-caused (not a defect, documented
    to preempt future confusion) two diagnostic artifacts: a benign
    one-time cold-start dispatch fallback (`State_To_FV`'s
    `FV_InitState`-internal COLDSTART call runs before
    `fv3_gpu_bridge_init`, so CUDA dispatch is architecturally
    impossible there) and a stale-host-read diagnostic artifact in
    `pkz_sample` (Phase C intentionally leaves `pe`/`peln`/`pk`/`delp`/
    `pt`/`delz`/`pkz` device-resident by design -- only `ps` is D2H'd --
    so a host-side debug print of `pkz` reads stale memory, not a
    kernel defect; confirmed by direct source read plus the Stage 9
    harness's own explicit D2H+bitwise `pkz` check).
    **C512 integration (Stage 14) then hit a device out-of-memory
    blocker specific to the 6-ranks-per-1-A100 production density**: 5
    same-node paired A/B runs (job 58628363,
    `p99_state_to_fv_ab5.sbatch`, same methodology as P92/P89) showed
    every Fortran-arm run succeeding cleanly (30/30 `DYN Date` cycles,
    5/5) while every CUDA-arm run aborted via
    `FATAL ... out of memory` + `MPI_Abort` within ~37-38s, before a
    single cycle completed (0/5). Root cause confirmed via `nvidia-smi`
    telemetry, not assumed: the Fortran-only baseline (with every other
    already-shipped production CUDA default active) already drives the
    shared A100 to 40265/40334 MiB used -- only ~69-73 MiB of headroom
    -- before this port's new persistent per-rank buffers (~0.9GB
    aggregate across 6 independent per-rank CUDA contexts) are even
    added. **No C512 numerical validation or performance comparison
    (Stage 15/16) was possible** -- the CUDA arm never produced a
    single timer sample to pair against the Fortran baseline. This is a
    device-memory capacity blocker at C512 density, not a kernel
    defect (kernel correctness at C32/smoke-test scale is unaffected).
    Production default remains Fortran. No commit/push. See
    `docs/performance/p99_state_to_fv_cuda_implementation_2026-09-28.md`.

34. **(Prompt 100 -- State_To_FV device memory accounting and
    buffer-reuse study, CLOSED -- decision: STILL MEMORY-BLOCKED,
    BROADER DEVICE-RESIDENCY ARCHITECTURE REQUIRED).** Audited all 8
    `state_to_fv_*` persistent device buffers added by Prompt 99
    (150.99 MB/rank, 905.94 MB aggregate/6 ranks -- reconciles exactly
    with P99's own estimate) via direct source read of
    `fv3_gpu_context.hpp`/`fv3_gpu_runtime.cpp`/`fv3_state_to_fv.cuh`.
    Found zero duplicate storage (every buffer holds data with no
    existing FieldRegistry equivalent; the port's own output side
    already writes directly into existing device-resident FieldRegistry
    storage with no duplication). Found one safe, low-risk reuse
    candidate: all 8 buffers are lifetime-disjoint (per the project's
    existing single-compute-stream, in-order-execution convention
    already relied on for `WorkspaceId::transport`'s documented shared
    reuse) from the existing, ~6x-larger (~906 MB/rank)
    `WorkspaceId::vertical` Riemann-solver arena, and could alias into
    it at zero new `cudaMalloc` cost. **Decisive finding: even a
    perfect realization of that alias only returns the system to its
    pre-existing baseline headroom (~69-73 MiB, measured by Prompt 99,
    without this port at all), which is already ~440 MiB short of the
    required >=512 MiB safety target on its own.** No
    `State_To_FV`-scoped fix -- including the best one identified --
    can reach the required target, so none was implemented; no
    production source changed. Recommends a per-port device-memory
    audit of every already-shipped production CUDA default (PT/DELZ,
    REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS) plus the four
    `WorkspaceLayout` arenas as the next project phase, rather than
    another isolated CUDA kernel port. No commit/push. See
    `docs/performance/p100_state_to_fv_device_memory_reuse_2026-09-29.md`.

35. **(Prompt 101 -- production GPU memory inventory and
    recoverable-headroom audit, CLOSED -- decision: RECONCILIATION
    INCOMPLETE, <90% GATE NOT MET).** Executed Prompt 100's recommended
    per-port/arena audit. Static accounting of every named production
    device allocation -- FieldRegistry (~870 MiB/rank, category-based
    estimate), `WorkspaceLayout`'s 4 arenas (~2170.4 MiB/rank),
    REMAP_Q + REMAP_PRESSUREVARS grow-only buffers (~82 MiB/rank
    combined); PT/DELZ and REMAP_UV independently confirmed to add zero
    dedicated persistent buffers -- sums to ~3.05 GiB/rank (~18.74 GiB
    aggregate across 6 ranks), only **~46.6%** of the live-measured
    converged C512 footprint (~40.26-40.33 GiB used/40.34 GiB total, 6
    ranks/1 A100, MPS ON). One short TEST-ONLY-instrumented diagnostic
    job (58638073, `cudaMemGetInfo` at 5 lifecycle checkpoints, fully
    reverted after -- rebuild sha256 verified byte-identical to Prompt
    99's own hash) independently reconfirmed the converged total and
    established that it is front-loaded: fully resident by the first
    production dispatch call, stable through 30 DYN_CORE steps. The
    unexplained ~21.5 GiB gap (~53.4%) was narrowed to occurring
    strictly during that first dispatch call, but not attributed
    further among 4 candidate causes (CUDA lazy kernel/module loading,
    unsized halo exchange buffers, static-estimate undercount of
    FieldRegistry/WorkspaceLayout, or unenumerated CUDA-runtime/library
    reservations) -- recognized that unsynchronized, MPS-shared,
    cross-rank `cudaMemGetInfo` readings cannot cleanly attribute
    memory to individual lifecycle stages without an `MPI_Barrier`
    before each checkpoint. **Per the audit's own Stage-8 rule (<90%
    reconciliation means do not select a fix), no memory fix is
    selected or recommended.** `State_To_FV` remains CLOSED -- STILL
    MEMORY-BLOCKED (Prompt 100's decision reinforced, not superseded).
    Recommended next diagnostic: a barrier-synchronized job bracketing
    the first dispatch call specifically, to distinguish the 4
    candidates before any memory-recovery fix or Prompt 102 multi-GPU
    scaling decision is made. No production source changed; no
    commit/push. See
    `docs/performance/p101_production_gpu_memory_inventory_2026-09-29.md`.

36. **(Prompt 101A -- consolidate end-to-end cumulative CUDA performance
    improvement so far, CLOSED).** Answered "how much faster is the
    current production CUDA config than the pre-port baseline" directly,
    since no historical run was comparable: Prompt 48 (the only
    same-node zero-remap-port C512 comparison) measured only the
    `DYN_CORE` fv_timing timer, which per Prompt 69 is a **sibling** of
    `REMAPPING` and therefore structurally blind to all four retained
    remap ports (PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS); it also
    ran on a different node than the current-production reference (job
    58619479, Prompt 94). Reconstructed the baseline directly on the
    current codebase instead: one clean same-node 5-pair alternating A/B
    (job 58638414, node `warpa010`, sha256 `2339366b...186e2064d`, Arm A
    = all four `_FORCE_FORTRAN=1`, Arm B = all four unset/CUDA-default,
    tracer/State_To_FV/REMAP_TOTALENERGYCON Fortran in both arms). Result
    (CUDA wins 5/5 pairs, low variance): **TOTAL improvement mean
    13.264%/median 13.466% (speedup 1.153x/1.156x)**, **FV_DYN_LOOP mean
    28.957%/median 28.746% (speedup 1.408x)**, **REMAPPING mean
    61.390%/median 61.220% (speedup 2.592x)**. Individual-port direct
    speedups tabulated separately (PT/DELZ 3.06x, REMAP_UV ~10.23x,
    REMAP_Q ~3.25x, REMAP_PRESSUREVARS ~3.50x) with an explicit
    do-not-sum-percentages caveat, since each was measured against a
    different evolving historical baseline. Bottleneck-shift narrative
    (from P94/P96): the ports correctly eliminated the pre-port
    `REMAPPING` bottleneck; the new leading costs are
    `REMAP_TOTALENERGYCON` (comm-bound, P95 BLOCKED), `TRACER_2D`
    (Fortran, P83 STOP), already-GPU `DYN_CORE`, and MAPL-level
    `DYN_EPILOGUE` (688.67 ms/call, not yet profiled before P96). GPU
    memory cited from P99/P101 (~40.3 GiB/40.96 GiB, ~70 MiB free,
    State_To_FV's allocation excluded since it stays disabled). No
    numerics/production defaults changed; no commit/push. See
    `docs/performance/p101a_cumulative_cuda_performance_2026-09-29.md`.

37. **(Prompt 102 -- multi-GPU scaling and device-memory architecture
    study, ARCHITECTURE/DESIGN ONLY, CLOSED).** Resolved Prompt 101's own
    open "Caution for Prompt 102" (fixed-per-GPU vs. per-rank-context
    memory scaling, unresolved at P101 close): a bounded, zero-source-
    change diagnostic (job 58639804, reused the existing
    `p101_memdiag_c512` rundir/binary unmodified with `--gres=gpu:4`
    instead of `--gres=gpu:1`) measured converged C512 memory at 1- and
    2-ranks-per-GPU density directly (6746 MiB, 13450 MiB; two different
    physical GPUs read identically at matched density) and found the
    ~21.5 GiB gap is **overwhelmingly per-rank/per-context, not
    fixed-per-GPU** (`M_fixed_per_gpu` ~0.04 GiB, `M_per_rank_context`
    ~6.55-6.7 GiB; HIGH confidence on the split, MODERATE on exact
    magnitude given a small ~1.5% super-linear residual only visible at
    6-ranks/1-GPU density). The same job also empirically confirmed (30/30
    `DYN Date`, no FATAL/NaN) that **multi-GPU execution already works
    today with zero code changes** on a single node, since
    `dev = mod(comm_rank, device_count)` coincides with a node-local-rank
    mapping when the job stays on one node -- true multi-node support
    needed a `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`-based fix,
    **implemented and validated by Prompt 107**
    (`p107_node_local_gpu_binding_2026-09-30.md`): single-node
    bit-identical, genuine 2-node divergence proof, no numerical/
    performance regression. Found the CUDA bridge hardcodes
    `comm_size == 6` and the six-tile halo transport has no intra-tile
    exchange path -- increasing MPI rank count for C1024 (to keep
    per-rank memory bounded, per the task's own design principle) is
    therefore an **architecture change**, not a config tweak; the
    zero-architecture-change path (6 ranks unchanged, bigger per-rank
    tiles, spread across more GPUs) was recommended as the near-term C1024
    stepping stone. Projected 2-GPU/3-ranks-per-GPU C512 headroom
    (~19.7-20.7 GiB free/GPU) comfortably clears `State_To_FV`'s ~440 MiB
    P99/P100 shortfall with no change to `State_To_FV` itself. Decision:
    proceed to a 2-GPU C512 implementation/validation experiment next. No
    numerical/production-default/source changes; no commit/push. See
    `docs/performance/p102_multigpu_scaling_architecture_2026-09-29.md`.

18. **Prompt 103 (2026-09-29)**: executed the 2-GPU C512 experiment Prompt
    102 called for (job 58642794, resumed after a connection-loss
    interruption; found 9 rundirs already staged with a stale-`EXPDIR`
    defect, fixed before submission). Measured, not projected: converged
    memory **19.68 GiB/GPU** (vs. 19.7-20.7 GiB projected — matches
    closely), correctness **BITWISE EXACT** (T/U/V/PS) vs. 1-GPU, and a
    **consistent, reproducible performance improvement**: TOTAL
    1.120x/10.70% (3/3 pairs, ±0.5pp), FV_DYN_LOOP 1.397x, DYN_CORE 1.773x.
    `State_To_FV` CUDA memory feasibility (P98-100's ~440 MiB shortfall)
    **confirmed cleared** — 19.7 GiB/GPU free, ~40x the shortfall; opt-in
    smoke run clean (rc=0, 30/30, 0 FATAL/NaN) with timing suggestive of
    CUDA dispatch. **Recommended C512 mode: 2-GPU/3-ranks-per-GPU**
    (strictly better on memory and performance, zero correctness cost,
    zero source change). Next: enable/validate State_To_FV CUDA
    performance at 2-GPU density (Option A). No numerical/production-
    default/source changes; no commit/push. See
    `docs/performance/p103_c512_2gpu_validation_performance_2026-09-29.md`.

19. **Prompt 104 (2026-09-29)**: executed Prompt 103's Option A —
    enabled/validated `State_To_FV` CUDA at 2-GPU/3-ranks-per-GPU (job
    58645113). CUDA dispatch of both phases **CONFIRMED** via converging
    indirect evidence (branch selection, ABI-success/no-FATAL, +440.4
    MiB/GPU device-memory footprint matching P100's +452.97 MiB
    prediction, reproducible timing signature) — a planned raw dispatch
    counter was never actually implemented in source, so that specific
    mechanism's `0/0` grep result is a harness gap, not a negative
    finding. Direct `State_To_FV` speedup **1.633x mean/38.75% reduction**
    (3/3 alternating A/B pairs, stddev 0.006x — highly reproducible).
    End-to-end TOTAL **+1.01% mean** (MEANINGFUL, boundary), FV_DYN_LOOP
    +0.60% (SMALL POSITIVE), DYN_CORE flat (architecturally expected,
    `State_To_FV` sits outside the `k_split` loop) — attribution
    consistent (expected ≈0.97% vs. observed 1.01%). Memory: +440.4
    MiB/GPU, 19.3 GiB/GPU free at converged state — SAFE; workspace-alias
    optimization (P100) classified NOT NEEDED NOW at this density.
    **Correctness gap**: the short harness produced no HISTORY/`.nc4`
    output, so no field-level C512/2-GPU comparison was possible — only
    clean-run (rc=0, no FATAL/NaN, 30/30 calls) evidence exists at this
    density; C32 bitwise validation (P98-99) stands unchanged but does not
    cover this density. **Decision: KEEP OPT-IN** — performance is
    positive and reproducible, but default-enable requires closing the
    C512/2-GPU field-correctness gap first. No numerical/production-
    default/source/MPI changes; no commit/push. See
    `docs/performance/p104_state_to_fv_2gpu_validation_performance_2026-09-29.md`.

20. **Prompt 105 (2026-09-29)**: closed Prompt 104's correctness gap —
    **and found that gap was a methodology error, not a real absence of
    output.** Re-inspecting job 58645113's existing rundirs found
    `HISTORY.rc`/`.nc4` output present all along under each rundir's
    `scratch_c512_L10_T2_1x6_1threads_GEOSfvdycore-v3.2.0-R8/` execution
    subdirectory — Prompt 104 checked only the top-level rundir (correctly
    empty; that level only ever holds `fv3.j`/binary/copied logs) and never
    descended into the subdirectory MAPL History actually writes to. No new
    job was required. Field comparison via `hs_netcdf_inspect.py
    compare-backends` at C512/2-GPU/3-ranks-per-GPU: **T and PS BITWISE
    identical, U and V ROUND-OFF CLEAN** (max_rel 1e-8 to 1.7e-6, 0/131584
    NaN divergence) across every pressure level checked (850-100 hPa; 1000/
    975 hPa masked identically both arms — pre-existing benign MAPL
    artifact, unchanged). Max-diff-location check found no halo/PE-layout-
    clustered error pattern (2/131,584 points near the max, scattered, not
    tile-edge-aligned) — **PE/layout evidence: PASS**. All three validation-
    hierarchy legs now satisfied at this exact density: GPU EXECUTED,
    NUMERICALLY VALIDATED, and PERFORMANCE VALIDATED (Prompt 104's 1.633x
    direct/+1.01% TOTAL, reused unchanged — no performance A/B rerun, per
    this task's scope). **Decision: READY FOR DEFAULT ENABLE** — default
    NOT changed in Prompt 105 itself; default-enable is deferred to its own
    separate, dedicated bounded task. No CUDA/Fortran source changed, no
    production default changed, no MPI change, no commit/push. See
    `docs/performance/p105_state_to_fv_c512_2gpu_correctness_2026-09-29.md`.

21. **Prompt 106 (2026-09-29)**: executed Prompt 105's deferred
    default-enable task. Selector change only (`cuda/interface/
    fv3_gpu_dyncore_bridge.F90`: 4-line cached `fv3_gpu_device_count()`
    getter reusing the existing `fv3_gpu_get_capabilities` call;
    `FV_StateMod.F90` in the `GEOSfvdycore_fortran_perf` checkout: base
    default gated on `device_count>=2`). Precedence: `FORCE_FORTRAN`
    (always wins) > unsupported config > `ENABLE` (dev opt-in, any
    density) > `device_count>=2` (new default) > Fortran (safe
    fallback, covers the known-unsafe 1-GPU/6-ranks density). Selector
    matrix (C32, job 58650663): **6/6 PASS**, including 1-GPU-safety and
    FORCE-wins-over-ENABLE-at-1-GPU cases. C512/2-GPU smoke (job
    58651017, reused P104's rundirs): bare new default correctly
    selects CUDA (rc=0, `DYN_DATE_COUNT=30`, 0 Abort/FATAL/OOM, peak
    20585 MiB/GPU matching the ~20.6 GiB/GPU prediction); escape hatch
    (`FORCE_FORTRAN=1`) still correctly selects Fortran. No numerical/
    CUDA-kernel/ABI/other-selector change; no commit/push. **Decision:
    STATE_TO_FV CUDA CONDITIONAL DEFAULT ENABLED** at `device_count>=2`
    only — 1-GPU stays Fortran-default. Documented (not implemented) a
    multi-node device-mapping limitation (`dev = mod(comm_rank,
    device_count)` is single-node-safe only) as the recommended next
    task. See
    `docs/performance/p106_state_to_fv_default_policy_2026-09-29.md`.

22. **Prompt 107 (2026-09-30)**: implemented and validated Prompt 106's
    recommended next task — replaced the single-node-only
    `dev = mod(global_comm_rank, device_count)` GPU-selection rule in
    `fv3_gpu_dyncore_bridge.F90` with a node-local rule via
    `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED,...)` +
    `dev = mod(local_rank, device_count)`. Fixed a real logging bug found
    along the way (`mpp_error(NOTE,...)` prints from PE 0 only; switched
    to per-rank `write(0,...)`). Validated single-node 1-/2-GPU
    (bit-identical to Prompt 103/106's mapping) and genuine 2-node
    multi-node (divergence proof vs. the old global-rank rule — node 1's
    actual mapping differs from what the old rule would have given).
    Exit-dump numerical samples bitwise identical across all 4 arms
    (routing-only change); no performance regression. Discovered two new
    Discover platform constraints: `alla100` QOS caps a job's summed GPU
    count/memory at `gres/gpu=4`/`mem=510000M` job-wide, not per-node;
    multi-node CUDA MPS needs an explicit `srun --overlap` broadcast. Also
    upgraded the severity of the "stale EXPDIR" bug (previously thought to
    cause only harmless log-capture loss) to a real job-collision crash
    risk under concurrency. No numerical/CUDA-kernel/ABI/other-selector
    change; no commit/push. **This closes the multi-node device-mapping
    limitation** — the separate `comm_size==6`/no-intra-tile-halo
    constraint (§Part 10 below) is untouched and remains the next
    architectural blocker. See
    `docs/performance/p107_node_local_gpu_binding_2026-09-30.md`.

23. **Prompt 108 (2026-09-30, DESIGN/MAPPING ONLY)**: designed (did not
    implement) the generalized intra-tile halo-exchange architecture
    needed to remove the `comm_size==6` assumption, targeting a future
    C1024 decomposition of 6 faces x 2x2 ranks/face = 24 ranks. Key
    finding: the authoritative Fortran/FMS `domain_decomp`
    (`fv_mp_mod.F90`) already natively supports `layout(1,2)>1` per
    cubed-sphere tile (`npes_per_tile = npes_x*npes_y`,
    `mpp_define_layout`/`mpp_define_mosaic`) — this capability is reused
    for the numbering/neighbor-metadata design, not reinvented. The
    CUDA-side gap is confined to `fv3_gpu_halo_topology.hpp/.cpp`'s
    single-neighbor-per-edge `TopologyView`/`EdgeNeighbor` data model
    (needs to grow to multi-segment-per-edge plus diagonal/mixed
    corners) and the bridge's `comm_size==6` gate — the transport
    pack/unpack pipeline and the seam rotation/vector-sign algebra are
    both reusable as-is. Corner-fill kernels need generalization (each
    2x2 sub-rank has exactly 1 true cube-panel corner, not 4).
    Classification: **MODERATE HALO GENERALIZATION** (not a full
    communication-architecture rewrite, not a config-only fix). Derived
    an 8-phase minimal implementation plan with STOP/PASS gates.
    Recommended next bounded task: implement/validate Phases 1-4
    (CUDA-side topology/transport generalization, small grid, single
    node) — explicitly not the bridge `comm_size` relaxation (Phase 6) or
    a real 24-rank/C1024 run (Phase 8), each deferred to its own future
    task. No production source changes; no commit/push. See
    `docs/performance/p108_c1024_intra_tile_halo_architecture_2026-09-30.md`.

24. **Prompt 109 (2026-09-30, IMPLEMENTED/COMPILED/GPU EXECUTED)**:
    implemented Prompt 108's §15 Phases 1-4 — a new, additive, header-only
    `fv3_gpu_halo_subtile_topology.hpp` giving `layout(2,2)`/24-rank
    sub-tile rank numbering, same-face cardinal edges, cross-face segment
    donors, and diagonal/mixed/true-cube corner taxonomy, all
    `FV3_HALO_HD` (host+device). Found and fixed one genuine
    device-callability gap in the existing `cubed_sphere_neighbor()`
    (host-only, called from the new device-callable code): relocated its
    mosaic table/lookup into `fv3_gpu_halo_topology.hpp` with a
    macro-generated dual host/`static __device__` table (numerical
    semantics unchanged, confirmed via legacy-test regression + host/
    device bit-identical comparison). GPU-EXECUTED on a live A100
    (24/24 ranks bit-identical device-vs-host), compute-sanitizer clean
    (0 errors), 24-rank real-MPI routing test PASS (after fixing a
    self-contained test-harness deadlock, non-blocking post-all/wait-all
    fix), layout=1x1 exact regression preserved. Bridge `comm_size==6`
    gate (`fv3_gpu_dyncore_bridge.F90:163`) and transport-layer gate
    (`fv3_gpu_halo_mpi.cpp:416`) both confirmed untouched. Topology-only —
    no field data movement, no numerical validation applicable this
    phase, no production default changed, no commit/push. Recommended
    next bounded task: Phase 5 (transport-backend generalization, same
    small-grid/single-node scope). See
    `docs/performance/p109_cuda_subtile_halo_phases1_4_2026-09-30.md`.

25. **Prompt 110 (2026-09-30, DESIGN/SCOPING ONLY)**: mapped how Prompt
    109's sub-tile topology wires into the production halo pipeline.
    Findings: `launch_halo_pack` is topology-agnostic (no change needed);
    `launch_halo_unpack` depends only on donor-edge metadata (no kernel
    change, only correctly-populated `TopologyView`); `MpiHaloTransport`'s
    real MPI addressing already keys on rank alone (small, deferred
    generalization); corner exchange for diagonal/mixed corners is a
    wholly new mechanism, not a generalization, and was deferred.
    Corrected one input from Prompt 109's handoff: the transport-side
    6-rank gate lives in `fv3_gpu_halo_mpi.cpp:416`
    (`create_mpi_transport()`), not `fv3_gpu_halo_mpi.hpp`. Selected
    Phase-5 implementation slice: an additive `SubTileId`+layout to
    `TopologyView` builder plus a synthetic-rank field-data test harness
    (edges only, layout=2x2, no bridge/gate change). No production source
    changed, no commit/push. See
    `docs/performance/p110_subtile_halo_transport_phase5_design_2026-09-30.md`.

26. **Prompt 111 (2026-09-30, IMPLEMENTED + GPU EXECUTED, BLOCKED at
    layout=2x2)**: implemented Prompt 110's selected Phase-5 slice — new
    additive builder (`fv3_gpu_halo_subtile_view.hpp`) plus field-data
    test harness (`test_halo_subtile_exchange.cu`) calling the real,
    unmodified `launch_halo_pack`/`launch_halo_transfer`/
    `launch_halo_unpack` chain. layout=1x1 (N=6) is a bitwise-clean
    regression (0/20736 failures) — the new builder is a proven, exact,
    lossless substitute for the legacy one at that layout. layout=2x2
    (N=24) is **BLOCKED**: `launch_halo_transfer`
    (`fv3_gpu_halo_exchange.cu:273`) hardcodes `nb.tile` to
    `[1,cubed_sphere_tiles=6]` for its `peer_send` table lookup — a
    whole-subsystem constraint (recurs independently in
    `fv3_gpu_halo_peer.hpp` and `fv3_gpu_halo_mpi.cpp`), not a one-off
    check — which the sub-tile builder's donor-rank-based `.tile` (up to
    24) legitimately exceeds for any cross-face donor. All 6 halo groups
    fail identically (confirmed via instrumented rerun: not a per-group
    or vector-vs-scalar defect). Topology-only correctness at layout=2x2
    remains fully validated (T12/T13, host-only). compute-sanitizer: 0
    errors on all 3 CUDA binaries touched. No production source changed,
    no gate relaxed, no commit/push. Next: Phase 6 bridge metadata
    plumbing must now also generalize this transfer-layer gate/indexing
    — a larger scope than Prompt 110 assumed; requires fresh
    authorization, not attempted here. See
    `docs/performance/p111_subtile_topology_fielddata_validation_2026-09-30.md`.

27. **Prompt 112 (2026-09-30, IMPLEMENTED + GPU EXECUTED + NUMERICALLY
    VALIDATED at 24-rank real MPI scale)**: fixed Prompt 111's blocker by
    separating physical face identity (`EdgeNeighbor.tile`, 1..6,
    permanent seam-math fact) from transport peer identity
    (`EdgeNeighbor.rank`, 0..peer_count-1). Generalized
    `launch_halo_transfer` to index `peer_send` by `.rank` with an
    explicit `peer_count` parameter (was hardcoded `.tile`/6);
    generalized `MpiHaloTransport`'s internal peer storage to
    `std::vector` (was fixed `std::array<int32_t,6>`). New test-only
    factory `create_mpi_transport_for_testing()` and new test
    `test_halo_subtile_mpi_transport.cu` (CMake target
    `fv3_halo_subtile_mpi_transport_test`) GPU-executed a real 24-rank
    MPI run (`mpirun --oversubscribe -np 24` on one A100) exercising the
    generalized transfer/peer-storage path end-to-end: **0/22,224
    failures**, **0 compute-sanitizer errors on all 24 ranks**. Combined
    sanitizer sweep also reconfirmed the two pre-existing binaries
    unregressed (`fv3_halo_subtile_exchange_test` 0/102,402,
    `fv3_halo_exchange_multitile_test` 0/20,484). Fixed-6 re-audit found
    no remaining ungeneralized transport storage — every other
    `cubed_sphere_tiles`/literal-6 occurrence is either the still-active
    Fortran-bridge or MPI-factory production gate (both untouched,
    re-verified by source read) or a legitimate physical-6-face bound
    check. Corrected two prior docs' inaccurate "`launch_halo_unpack`
    never reads `.tile`" claim (it does, as a sanity bound, never as an
    index — `p110_*.md`/`p111_*.md` updated in place). No bridge/MPI-
    factory gate relaxed, no real >6-rank production path opened, no
    corner transport, no numerical/seam-math change, no R4 change, no
    commit/push. Bridge metadata plumbing (extracting real FMS
    `layout_x/layout_y` into the bridge) remains a separate, larger,
    fresh-authorization-required task — recommended next step. See
    `docs/performance/p112_generalized_halo_transport_peer_ids_2026-09-30.md`.

28. **Prompt 113 (2026-09-30, DESIGN/MAPPING ONLY, CLOSED)**: designed the
    FMS decomposition metadata to CUDA bridge integration. Found
    `Atm%layout(1:2)` (`fv_arrays.F90:1287`, pointer-aliased from the
    `fv_core_nml` namelist at `fv_control.F90:1333`) is the authoritative,
    already-resident runtime source for `layout_x/layout_y` - no FMS
    query call needed. Found the bridge call site
    (`fv3_gpu_bridge_init(Atm, comm, ...)`, `FV_StateMod.F90:1264`)
    already receives the full `Atm` and already reads `Atm%tile`
    (`fv3_gpu_dyncore_bridge.F90:300`), so reaching `Atm%layout` needs
    zero new plumbing - the gap is an ABI/gate gap, not an access gap.
    Found a second, previously undocumented one-rank-per-tile gate inside
    `build_rank_of_tile` (`fv3_gpu_dyncore_bridge.F90:1451`) beyond the
    known top-level `comm_size==6` check - both must generalize together.
    Specified a minimal additive ABI extension (`layout_x`/`layout_y`
    appended to `fv3_gpu_create_desc` and to
    `fv3_gpu_set_cubed_sphere_topology`, `(sub_i,sub_j)` derived
    CUDA-side, not duplicated over the ABI) and a bounded, SMALL-
    complexity next task (wire-only: add the fields, populate them,
    take no new branch on their value - gates stay exactly as they are).
    Confirmed FMS's within-tile row-major/x-fastest rank convention
    remains an assumed, not source-verified, fact (FMS is a precompiled
    external dependency, not vendored in either repo) and recommended a
    bounded runtime cross-check as mitigation rather than leaving it a
    silent assumption. No ABI change, no gate relaxation, no 24-rank run,
    no corner transport, no numerical/seam-math change, no CUDA-aware
    MPI, no production-default change, no R4 change, no commit/push. See
    `docs/performance/p113_fms_bridge_subtile_metadata_design_2026-09-30.md`.

29. **Prompt 114 (2026-09-30, IMPLEMENTED + GPU EXECUTED, CLOSED)**:
    implemented Prompt 113's bounded next task — wired `layout_x`/
    `layout_y` (derived from `Atm%layout(1:2)`) through
    `fv3_gpu_create_desc`, `fv3_gpu_set_cubed_sphere_topology` (both the
    active-bridge ABI and the legacy `fv3_gpu_interface_mod.F90`
    mirror), the bridge, and the CUDA runtime's `TopologyView` storage —
    additive-only, same `<=0`-defaults-to-1 normalization as existing
    fields, zero new production branch. Added a second `MPI_Allgather`
    in `build_rank_of_tile` cross-checking every rank reports the same
    layout (FATAL on mismatch), and an `FV3_GPU_LAYOUT` debug print
    under the existing `RUN_CUDA_FV3_LOG_GPU_BINDING` gate. Compile-
    verified the C++/CUDA side (`ctest` 37/39, 2 pre-existing no-GPU-on-
    login-node failures) and, critically, the active-bridge Fortran/ABI
    side against the REAL production fixture (`fvdycore` target, Intel
    MPI toolchain) — zero new errors. Reconstructed the missing
    `build_cuda_r8_intelmpi` archive directory (9 archives, unrelated
    build-infra gap, not a code defect) to reach a fully linked
    `StandAlone_FV3_Dycore.x`. GPU-executed clean smoke tests at both
    C32 (6 ranks/1 GPU) and C512 (6 ranks/2 GPUs, MPS ON) — `rc=0`, full
    untruncated step counts, zero FATAL/Abort/NaN/Inf, correct
    `layout=1x1` on every rank at both scales, and C512 timers within
    documented node-variance noise of the pre-P114 baseline (behavioral-
    neutrality smoke check, not a bitwise/savepoint comparison). Re-
    confirmed by source read that all four >6-rank production gates
    (bridge `comm_size==6`; `build_rank_of_tile`'s tile-map FATAL;
    runtime's `mpi_size!=cubed_sphere_tiles` check;
    `create_mpi_transport()`'s `comm_size==6` gate) remain untouched. No
    sub-tile derivation added to production. `fv3_gpu_interface_mod.F90`
    mirror edits are structural-only, not independently compiled (not
    part of the production fixture's build — flagged). Recommended next
    task: verify FMS's within-tile rank ordering via a Fortran-only
    `layout=(2,2)` diagnostic (Option A) before any topology-builder
    switch. No gate relaxed, no corner transport, no numerical/seam-math
    change, no CUDA-aware MPI, no production-default change, no R4
    change, no commit/push. See
    `docs/performance/p114_fms_layout_bridge_abi_plumbing_2026-09-30.md`.
