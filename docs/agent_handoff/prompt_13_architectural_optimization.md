# Prompt 13 — Architecture-Level Optimization: CUDA Graph Replay Repair vs. Non-Graph Batching

Status: **COMPLETE — CUDA GRAPH REPAIR IMPLEMENTED, VALIDATED BITWISE, MEASURED
−5.14%.** Job 58238910 (`perf_rerun_logs/prompt13_graph_repair.sbatch`), node
`warpa013`. Zero of the three permitted fix cycles consumed. Prompt 8's
"BLOCKED" status is superseded: its divergence root cause is found and fixed
(one missing `cudaGraphLaunch`), and its replay crash did not reproduce across
1,770 replays under correct job resourcing.

Git HEAD at start: `7dda37b` ("Optimize FV3 CUDA halo edge launches"). Working
tree carried Prompt 12's uncommitted `cuda/dyn_core/operators/wp_d_cuda.cu`
change, which is retained (it is part of the reference baseline).

Reference configuration for everything below: 32×32 / npz=10 / n_split=3 / R8 /
six tiles / resident mode / `--fuse-halo-edges`. Reference timing:
**22.255549 ms/call** (Prompt 12, job 58238626, node `warpa010`).

---

## 1. Current host orchestration (one acoustic substep)

Established by direct source reading of `runtime/fv3_gpu_runtime.cpp`,
`benchmarks/run_small_perf_bench.cpp`, `include/fv3_gpu_halo_peer.hpp`,
`runtime/fv3_gpu_halo_exchange.cu`, and `include/fv3_gpu_internal.hpp`.
Individual kernels are deliberately not enumerated.

**Threading and context model.** The benchmark is a **single process with six
host threads**, one per cubed-sphere face (`run_small_perf_bench.cpp:3297`
`std::thread th[kTiles+1]`, launch loop `:3298-3309`, join `:3310`, `kTiles=6`
at `:2339`). Each thread owns one `fv3_gpu_context` (created at `:2822`). All
six contexts **share the device's CUDA primary context** — the runtime only ever
calls `cudaSetDevice` (`fv3_gpu_runtime.cpp:1353`), never
`cuCtxCreate`. Confirmed by trace: `cudaStreamCreateWithFlags` Num Calls = 12 =
2 streams × 6 tiles.

**Streams and events.** Two non-blocking streams per context (compute +
transfer, `:1359`). Six events per context, all `cudaEventDisableTiming`,
recorded and waited on, never queried.

**Acoustic loop.** The n_split loop is on the **host**
(`fv3_gpu_runtime.cpp:1826`, closing `:2071`). Stage dispatch is a fixed 13-slot
`NumericalDispatch` function-pointer table (`include/fv3_gpu_internal.hpp:371-407`).

**Host synchronization.** **Zero device synchronizations inside the acoustic
loop on the default path.** The only blocking constructs are two *host* barriers
inside each halo exchange (`pg->barrier()`, `:830` and `:874`), which order
enqueue only. D2H/H2D happen at defined boundaries outside the loop.

**Pointer/arena bindings.** Stable across substeps *and* across calls: one
`cudaMalloc` for fields (`:1385`), one for workspace (`:1422`), halo buffers
once (`:1188-1189`). Arena slice bases are re-derived each call as
`arena.arrays[n] = base + n*bytes`, i.e. byte-identical addresses. The only
allocation reachable during a call is a lazy at-most-once `heat_accum`
`cudaMalloc` (`:1793`), before the loop.

```
                      ONE ACOUSTIC SUBSTEP  (per tile thread; x6 concurrently)
  host thread t (tile t)                          device (single primary context)
  ─────────────────────                           ────────────────────────────────
  cgrid_predictor        ──enqueue──►  compute_stream ─┐
  resolve(H_CWIND) ─┐                                  │ kernels
      pack          ├─enqueue──►  compute_stream       │
      rec event_halo_pack                              │
      barrier()  ◄── all 6 threads ──►                 │  (host-side rendezvous:
      transfer      ├─enqueue──►  transfer_stream ─────┤   orders ENQUEUE only,
      rec event_halo_xfer                              │   no device sync)
      barrier()  ◄── all 6 threads ──►                 │
      compute_stream waits event_halo_xfer  ───────────┤
      unpack        ┘─enqueue──►  compute_stream ──────┤
  cgrid_vertical_pre_halo / solver / pressure ────────►│
  resolve(H_CWIND)                          ──────────►│
  ┌──────────────────────────────────────────┐         │
  │ dgrid_dynamics  (d_sw; the CAPTURE       │────────►│  ~201 launches/tile
  │ REGION — bounded by a halo resolve on    │         │
  │ both sides, all on compute_stream)       │         │
  └──────────────────────────────────────────┘         │
  resolve(H_DSCALAR) ────────────────────────────────►│
  dgrid_height_transport / dgrid_height / vertical_solver / nh_pressure ─►│
  heat accumulate (device) ─────────────────────────────────────────────►│
  [it == n_split] E11 seam reconcile ───────────────────────────────────►│
  ─ loop back (host) ───────────────────────────────────────────────────►
```

Ten halo resolve sites sit inside the loop (seven fire every substep); `H_HEAT`
resolves after the loop (`:2101`).

**Consequence for optimization.** Wall clock tracks host launch issuance almost
exactly. Fitting `wall ≈ launches_per_tile × median cudaLaunchKernel latency`:
Prompt 7 gives 2,955 × 8.14 µs = 24.05 ms vs 26.01 ms measured (92%); Prompt 11
gives 2,697 × 7.54 µs = 20.34 ms vs 23.28 ms (87%). **The task's ~5% whole-call
bar therefore equals ≈148 launches/tile/call ≈ 886 launches/dyn_core call, i.e.
5.5% of the current 16,074 total.**

---

## 2. CUDA Graph failure root cause

### Finding: the capture occurrence never executed. Repair scope: **LOCAL**.

At `fv3_gpu_runtime.cpp:1946-1974` (pre-repair) the capture arm was:

```
cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal)
bool ok = run(dispatch.dgrid_dynamics)
cudaStreamEndCapture(cs, &graph)
cudaGraphInstantiate(&exec, graph, 0)
   ... no launch ...
} else {                       // replay arm, later occurrences only
cudaGraphLaunch(...)
}
} else                         // :1975 — binds the plain non-graph call at :1977
if(!run(dispatch.dgrid_dynamics)) return rc;
```

Stream-captured work is **recorded and not executed**. The `} else` at `:1975`
binds the fallback `run(dispatch.dgrid_dynamics)` at `:1977` to
`!cuda_graph_enabled` only, so nothing else covered for the capture occurrence.
Therefore **one entire `d_sw` substep — horizontal transport, divergence and
vorticity damping, KE/heat_source generation — was silently dropped on the first
acoustic substep of the first `fv3_gpu_dyn_core_step` per context per benchmark
repeat.**

This is a precise match for Prompt 8's reported symptom: O(1) *uncorrelated*
error on the wind fields (`u`/`v`/`ua`/`va`/`w`/`omga` at max_rel ≈1.0–2.0),
which is what dropping a whole dynamics stage looks like. A stale-pointer or
frozen-parameter bug would not produce that; it would produce either a crash or
a bounded, structured difference.

### Alternative mechanisms the task listed, each affirmatively excluded

| Candidate mechanism | Verdict | Evidence |
|---|---|---|
| Captured stale device pointers | **NOT the cause** | All device memory allocated once per context (`:1385` fields, `:1422` workspace, `:1188-1189` halo buffers); no free/realloc during a call. |
| Arena buffer rebinding between calls | **NOT the cause** | Arena bases are deterministically re-derived from one persistent allocation to byte-identical addresses each call. |
| Per-call parameter changes needing node updates | **ZERO required** | Grep over `cuda/dyn_core/dgrid/` for `acoustic_index`/`remap_step`/`last_call`/`init_step`/`end_step` returns no hits — `dgrid_dynamics` reads none of the per-substep scalars. The one frozen-by-value scalar is `DSwOptions::dt` (`wp_b_dsw_contract_cuda.cpp:513`), and `dt = bdt/n_split` is computed **once per call** (`:1547-1548`), so it is substep-invariant. Guarded in the repair rather than assumed. |
| Stream/event lifetime | **NOT the cause** | The capture region contains no CUDA API call other than `cudaPeekAtLastError()`, and no event record/wait. |
| Halo buffer ownership / frozen halo data | **NOT the cause** | Only the halo buffer *base pointer* is frozen; the data is live, written on `transfer_stream`, and ordered into the graph's stream by the `cudaStreamWaitEvent(compute_stream, event_halo_xfer)` enqueued at `:866` before the graph launch. |
| Cross-thread / cross-context graph interaction | **NOT the cause** | Capture is bounded by `resolve(H_CWIND)` (`:1932`) and `resolve(H_DSCALAR)` (`:2002`), so no peer thread's stream or event is inside the region; capture mode is `ThreadLocal`. |
| Capture across an invalid synchronization boundary | **NOT the cause** | No host sync inside the region. Prompt 8's capture also succeeded mechanically, which is consistent. |
| Launch topology varying by substep | **NOT the cause** | Launch geometry is config/topology-derived; per-level gating is done device-side (`level_active(lf,k)`), so one captured topology validly represents every substep. |

### Prompt 8's replay crash (illegal memory access near halo transfer)

**Root cause NOT ESTABLISHED**, deliberately. Bounded as follows:
`launch_halo_transfer` (`fv3_gpu_halo_exchange.cu:267-308`) has no
data-dependent addressing and no per-call allocation, and its `cudaMemcpyAsync`
is a **sticky-error reporting site** — an asynchronous fault raised earlier
surfaces there. Two independent plausible causes are removed by this pass: the
NaN/Inf-corrupted state produced by the dropped substep, and the 36-context /
12-CPU oversubscription of Prompt 8's own job (see §8 note). Re-tested by the
§6 stress ladder rather than debugged speculatively.

### Prompt 8's timing confound — fully explained

Its sbatch ran `mpirun -np 6 --oversubscribe $BENCH` over a benchmark that
**already** spawns six internal tile threads ⇒ **36 tile-contexts on one A100
with 36 threads on 12 CPUs**. Confirmed exactly by its own trace:
`cudaStreamBeginCapture` = 36 = 6 procs × 6 tiles, and `cudaGraphLaunch` =
2,664 = (25 calls × 3 substeps − 1 capture) × 36 = 74 × 36. `cudaMalloc`
3,564/6 = 594, `cudaFree` 3,564/6 = 594, `cudaStreamSynchronize` 396/6 = 66 —
all normalizing to the single-process baseline by exactly 6.

### Normalized graph benefit from Prompt 8's own trace (measured, not modeled)

Graph mode = 2,102.5 launches/tile/call vs. baseline 2,697.0 ⇒ **−594.5/tile/call
(−22.0%)**. Implied `dgrid_dynamics` cost = 594.5 × 25 / 74 ≈ **201 launches per
invocation per tile**; steady-state saving = 3 × 201 = **603/tile/call**.
Predicted wall clock 603 × 7.54 µs = −4.54 ms ⇒ **22.26 → ~17.7 ms/call
(−20.4%)**, which stays above the GPU kernel-busy floor of ~11.24 ms/tile/call.

---

## 3. Non-graph orchestration reduction candidates

283 `<<<` launch sites exist. The launch-*multiplying* host loops, ranked by
launches per dyn_core call (all six tiles, 3 substeps):

| # | Host loop | Launches/call | Share of 16,074 |
|---|---|---|---|
| L1 | `c_sw` per-vertical-level (`cgrid/fv3_cgrid_predictor.cu:719`), 35 launches × 10 levels × 18 | 6,300 | 39.2% |
| L2 | `update_dz_c` per-interface-level (`vertical/fv3_vertical_kernels.cu:883`), 7 × 11 × 18 | 1,386 | 8.6% |
| L3 | `a2b_ord4` per-field (`operators/wp_d_contract_cuda.cpp:101-110`), ~19 × 4 × 18 | 1,368 | 8.5% |
| L4 | per-`nord` damping | ~1,900 | ~11.8% |
| L5 | halo per-field | 528 | 3.3% |
| L6 | `copy_corners` | 1,140 | 7.1% |

Candidates, scored:

| Candidate | Expected launch reduction | Implementation scope | Numerical risk | Runtime risk |
|---|---|---|---|---|
| **4a** Collapse `c_sw`'s per-level loop into full-depth launches | **−5,670 (−35.3%)** | LOCAL per phase, **MODERATE aggregate** (~31 kernel signature edits + 5–6 launcher restructurings) | **NONE — bitwise by construction.** Level independence proven three ways: the driver's own comment (`:710-716`), the full-depth per-level-disjoint arena (`:44-91`, `:481-528`), and zero `k±1` accesses in the c_sw kernel files. | Low |
| **4b** Collapse `update_dz_c`'s per-interface loop | −1,260 (−7.8%) | MODERATE (requires promoting `gz2` from 2-D to 3-D) | Low | Low |
| **5a** Batch `a2b_ord4` across its 4 fields | −1,026 (−6.4%) | MODERATE (arena 12 → 24 arrays) | Low | Low |
| C1 Persistent / cooperative per-tile driver kernel | large | **RUNTIME-REDESIGN** | high | high |
| C3 Batch tiles within a GPU context | large | **RUNTIME-REDESIGN** | high | high |

4a + 4b + 5a together would be −7,956 (−49.5%).

Note on C3: its usual premise is **refuted** — the six tiles already share one
primary context in this benchmark, so a shared launch surface exists today.
Separate contexts arrive only with production MPI, and the real blocker is that
portability, not context separation. C1 and C3 are **NOT RECOMMENDED** under
this task's no-redesign constraint.

---

## 4. Ranking

| Approach | Expected benefit | Implementation scope | Correctness risk | Communication risk | Validation complexity | Recommended? |
|---|---|---|---|---|---|---|
| **A. CUDA Graph replay repair** (`dgrid_dynamics` only) | **−22.0% launches (measured from Prompt 8's own trace); predicted −20.4% wall clock** | **LOCAL** — one missing API call + one invariance guard (~15 lines, 2 files) | Low — replay reissues the identical kernel sequence; expect bitwise identity | **NONE** — halo exchange stays outside the capture region | Low — existing dump/`cmp` comparison + existing ctest/smokes | **YES** |
| B. Non-graph batching (best: 4a `c_sw` level collapse) | −35.3% launches; modeled ~−30% wall clock | MODERATE aggregate — ~31 kernel signature edits + 5–6 launcher restructurings | Low, but many independent edit sites, each a chance to consume a fix cycle | NONE | Moderate | Not this pass |

Against the five selection criteria: A satisfies all five (Gate C2 semantics
preserved — same kernels, same order, same stream; ≥5% of orchestration cost by
a factor of four; no runtime redesign; validated with the existing savepoint/dump
tooling; prototypable in one bounded task). B satisfies (1)–(4) but strains (5):
its edit surface is an order of magnitude larger than A's, against a 3-fix-cycle
budget.

## 5. GO/NO-GO

**GO — Direction A (CUDA GRAPH REPAIR).** Classified **LOCAL**; expected
whole-call improvement ≈20%, well above the ~5% bar. Selected over B because it
is the smaller change with a *measured* rather than modeled benefit, and because
it settles whether an architectural avenue previously written off as
fundamentally broken is in fact usable. B remains available and quantified as
the next candidate if A fails or under-delivers.

---

## 6. Prototype implemented

Job **58238910**, node `warpa013`, elapsed 1:25. Two files, ~15 functional lines.
Graph mode remains **opt-in** (`--cuda-graph`, default off), so the default path
is byte-for-byte the same code as before.

1. **`runtime/fv3_gpu_runtime.cpp` — the missing launch.** After a successful
   `cudaGraphInstantiate`, the capture arm now issues
   `cudaGraphLaunch(exec, cs)`. This is stream-ordered on `compute_stream`, so
   the `cudaStreamWaitEvent(compute_stream, event_halo_xfer)` that
   `resolve(H_CWIND)` enqueued *before* capture still orders the work after the
   halo transfer, and every following stage still sees it as an ordinary
   same-stream predecessor. No event, sync, or communication semantics changed.
2. **`runtime/fv3_gpu_runtime.cpp` + `include/fv3_gpu_context.hpp` — timestep
   invariance guard.** New `ContextData::graph_capture_bdt` /
   `graph_capture_n_split` record the capture-time pair; the replay arm refuses
   with `FV3_GPU_UNSUPPORTED_CONFIGURATION` if a later call presents a different
   `bdt`/`n_split`. This is the explicit handling of the one host scalar frozen
   into captured nodes (`DSwOptions::dt`) that §2 identified — a guard rather
   than `cudaGraphExecKernelNodeSetParams` over hundreds of nodes, because the
   port never varies the timestep within a context lifetime.
3. Corrected the now-inaccurate invariance comments in both files.

Capture region unchanged: **`dgrid_dynamics` only**. Halo exchange is not
captured, as §6 of the task requires.

**Numerical algorithm changed: NO.** Replay reissues the identical kernel
sequence with identical arguments on the identical stream.

## 7. Validation ladder — PASS at every rung

| Rung | Result |
|---|---|
| R8 CTest | **PASS, 22/22** (`100% tests passed, 0 tests failed out of 22`) |
| Default smoke (graph OFF) | PASS |
| Graph-mode smoke | PASS |
| Graph-mode + `--fuse-halo-edges` smoke | PASS |
| **Single call** (`--iters 1`; this is the *capture* occurrence, i.e. exactly what Prompt 8 got wrong) | `dyn_core_exit` **78/78 BITWISE IDENTICAL** to the non-graph reference (13 fields × 6 tiles), 0 DIFFERS, 0 MISSING |
| **Repeated calls** (`--iters 5` ⇒ 1 capture + 14 replays per tile) | `dyn_core_exit` **78/78 BITWISE IDENTICAL**, 0 DIFFERS, 0 MISSING |
| **Stress** (5 repeats × 120 calls = **600 dyn_core calls, 1,800 substeps, 1,770 replays**) | RC 0, `=== 3A.11 CUDA perf benchmark: PASS ===`, all six tiles finite, `delp` positive |
| Illegal memory access | **NONE** anywhere in the job |
| NaN/Inf / NONFINITE / FAIL lines | none |

So both Prompt 8 failure modes are gone: the numerical divergence is explained
and fixed (it was the dropped substep), and the replay crash did not reproduce
across 1,770 replays under correct resourcing. Prompt 8's crash is therefore
**most plausibly a consequence of the same defect** (subsequent substeps ran on
corrupted state) and/or of its 36-context oversubscription — but that remains an
inference, not a proven root cause, since the crash cannot now be reproduced to
test against.

Capture cost, measured: `capture+instantiate_ms` = **11.9–16.2 ms, once per
context per benchmark repeat** (60 messages = 6 tiles × 10 graph-mode
invocations — confirming nothing re-captures mid-run). In the timing runs this
lands entirely inside warmup.

## 8. Performance

**Resident wall clock**, 32×32/R8/`--fuse-halo-edges`, 20 warmup / 100 iters /
5 repeats, same binary, same node, same job:

| | repeats (ms/call) | median | spread |
|---|---|---|---|
| BEFORE (graph OFF) | 21.761185, 21.660791, 21.631239, 21.733000, 21.612384 | **21.660791** | 0.149 |
| AFTER (graph ON) | 20.546787, 20.172703, 20.549954, 20.053290, 20.548308 | **20.546787** | 0.497 |

**Whole-call improvement: −1.114 ms = −5.14%.** Above the task's ~5% GO bar, and
above both spreads. (Against the cross-session Prompt 12 reference of 22.2555
ms/call it would read −7.7%, but the same-node BEFORE is the honest comparand;
that 2.7% node/session difference is itself why BEFORE was retaken here.)

**Launch/API reduction** (nsys, both modes, 5 warmup + 20 iters + 1 repeat, so
25 dyn_core calls; normalized per dyn_core call over all six tiles):

| | BEFORE | AFTER | change |
|---|---|---|---|
| `cudaLaunchKernel` (Num Calls) | 404,550 → **16,182.0/call** | 276,678 → **11,067.1/call** | **−5,114.9/call (−31.6%)** |
| `cudaGraphLaunch` | 0 | 450 → 18.0/call | +18 (= 3 substeps × 6 tiles) |
| net host launch-issuing calls | 16,182.0 | 11,085.1 | **−31.5%** |
| per tile | 2,697.0 | 1,844.5 | −852.5 (= ~284 launches per `dgrid_dynamics` invocation) |
| `cudaLaunchKernel` share of CUDA API time | 89.0% | 78.2% | — |

The BEFORE figure of 2,697.0 launches/tile/call reproduces Prompt 11's recorded
value exactly, which validates the normalization. `cudaGraphLaunch` = 450 =
25 calls × 3 substeps × 6 tiles **including the 6 capture occurrences** — direct
confirmation that the repaired capture arm now launches.

### The material finding: launch reduction no longer buys proportional wall clock

−31.6% of kernel launches bought only −5.14% of wall clock. That breaks the
launch-latency model that §1 fitted to Prompts 7 and 11 (which tracked measured
wall clock to 87–92%), and it is the most important result of this pass. Two
corroborating observations from the same traces:

- Host issue time per tile per call fell 22.07 → 16.45 ms (−25%, nsys-inflated),
  far more than the 5% wall clock recovered.
- `cudaStreamSynchronize` total time went from **0.44 ms to 135.5 ms** over the
  same 66 calls (Max 0.036 ms → 12.999 ms). The host now *waits* for the device:
  it drains a backlog it previously never accumulated.

Interpretation: with `--fuse-halo-edges` plus graph replay, the 32×32/R8 case has
crossed from **host-launch-bound to GPU-execution-bound**. The remaining
headroom is kernel execution time (Prompt 11 measured the kernel-busy floor at
67.44 ms/call = 11.24 ms/tile/call against the 20.55 ms/call now achieved), not
host orchestration.

**This retires Direction B's expected value.** §3 candidate 4a would remove
−35.3% of launches — comparable to what graph replay just removed — and we have
now measured that such a reduction is worth ~5%, not the ~30% a proportional
model predicts. Direction B should not be attempted as a wall-clock optimization
on this evidence.

Classification: **HIGH VALUE by the ≥3% bucket, but at the low end** (5.14%), and
**terminal for this optimization class**.

## 9. Cost control

One architecture choice, one prototype, 32×32 only, R8 only, no Fortran rerun,
no R4, no grid sweep, no new savepoints, no individual-kernel tuning, no ncu.
Two short nsys runs (before/after, same job, `--iters 20 --repeats 1`) rather
than one — deliberately, because a mis-normalized single-sided profiler
comparison is exactly what cost Prompt 8's timing claim its credibility (§2).

## 10. Scope, limits, and what comes next

**Scope of the graph-mode claim.** Opt-in `--cuda-graph`, 32×32 / npz=10 /
n_split=3 / R8 / six tiles / resident / single process / one A100. Bitwise
equality to the default path is demonstrated at the `dyn_core_exit` boundary for
1 call and 5 calls; stability for 600 calls. **Not** claimed for R4, other grids,
`k_split > 1`, a varying timestep (explicitly refused at runtime), production MPI
with one process per tile (each rank would then have its own CUDA primary
context — capture is per-context and thread-local, so this should carry over, but
it is untested), or any configuration outside Gate C2's closed scope.

**Known limitation carried forward.** Graph mode cannot be toggled after capture
(`fv3_gpu_set_cuda_graph_mode` already refuses), and replay now also refuses a
changed `bdt`/`n_split`. A caller that legitimately needs a varying timestep with
graph mode would need node-parameter updates for `DSwOptions::dt`; nothing in the
current port does.

**Recommended next milestone: standalone CUDA integration, not further
launch-side optimization.** The evidence for stopping here is positive, not
exhausted-effort: three consecutive orchestration passes (Prompt 11 halo-edge
fusion −9.3% launches, Prompt 12 a2b4 occupancy, Prompt 13 graph replay −31.6%
launches) have moved 32×32/R8 resident from 29.20 to 20.55 ms/call, and the last
of them demonstrates that the case is now GPU-execution-bound, so the remaining
launch-side candidates (§3) can no longer be expected to pay. What is left splits
into two things this project has explicitly deferred: (a) kernel *execution* time
in the Riemann solvers (`riem_solver3_kernel` 153.1 µs avg / `riem_solver_c_kernel`
147.3 µs avg, together 7.3% of kernel time and the two largest individual
kernels), which needs occupancy/warp-stall data and is therefore blocked on the
`ncu` `ERR_NVGPUCTRPERM` platform restriction recorded in Prompt 12 §2; and
(b) extending validation beyond Gate C2's configuration (R4, other grids, moist).
Neither is a bounded local optimization, and (a) cannot be done evidence-first
today. Integration work does not depend on either.
