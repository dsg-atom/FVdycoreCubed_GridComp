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
6. **(Added, Prompt 52 — now the highest-value action, per the Part 0/2/6 updates above)** Build the standalone Fortran baseline harness for `fv_mapz_mod`'s `map_scalar`/`scalar_profile` (18.67%-of-window, largest single measured CPU-coverage gap) per Prompt 52's Phase 52A, as the first step toward closing the dominant coverage gap identified by Prompt 51 — see `docs/performance/p52_fv_mapz_cuda_port_plan_2026-09-21.md`.

---

CUDA PERFORMANCE ROADMAP COMPLETE — READY FOR PROFILING PHASE
