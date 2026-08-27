# Milestone 3A.11R — Cross-Tile Halo Synchronization Race Isolation

> **RACE HYPOTHESIS REJECTED.**
>
> There is no cross-tile synchronization race. The reported nondeterminism was
> an artifact of the diagnostic instrument (see §2 below). Three independent runs
> are bitwise identical, `racecheck` reports 0 hazards, and maximal per-stage
> synchronization does not remove the corruption.
>
> The defect identified here as #2 (H_GZ_FIRST resolving before its producer) was
> **fixed and verified against the authoritative tagged Fortran source** in
> Milestone 3A.11S. See
> [`milestone_3a11s_h_gz_first_ordering.md`](milestone_3a11s_h_gz_first_ordering.md),
> which also records a third, distinct residual defect isolated to
> `Riem_Solver_C` that still blocks repeated-step stability.

Branch: `cuda/gpu-bringup`
Base commit: `78c07e5` (`78c07e5b5b999bc2809e23e32e8a179d73e5114e`, "Record milestone 3A.10 commit SHA in handoff") (working tree uncommitted; diagnostic instrumentation retained per §18)

## Headline

**The premise of this milestone — a cross-tile synchronization race — is disproven.**

The reported nondeterminism was an artifact of the diagnostic instrument, not a
property of the computation. Under a corrected diagnostic the failure is fully
deterministic and bitwise reproducible. Two distinct **deterministic** defects
were found in its place; one is fixed, one is identified but not yet fixed.

The performance benchmark remains BLOCKED.

## 1. Reproducer

```
build_cuda_r8_perf/fv3_gpu_perf_bench \
  --grid 8 --npz 10 --n_split 3 --warmup 2 --iters 20 \
  --repeats 1 --mode resident --hardware A100
```
R8, 6 tiles in one process, one host thread + one context per tile, one A100,
`n_split=3`. `FV3_GPU_DEBUG_FAIL=1` enables `debug_check_finite_stage`.

## 2. The reported nondeterminism was a measurement artifact

`debug_check_finite_stage` synchronized only `compute_stream`, then performed a
blocking `cudaMemcpy` on the **legacy null stream**. Every runtime stream is
created `cudaStreamNonBlocking`, so that copy is unordered with respect to
`transfer_stream`. The diagnostic therefore sampled fields while halo unpacks
were still in flight, and *which* fields it caught mid-unpack varied per run.

Fix applied to the diagnostic (`fv3_gpu_runtime.cpp`): drain **both**
`compute_stream` and `transfer_stream` at every checkpoint, and emit a bitwise
FNV-1a field hash keyed by `tile`/`step` so two runs' logs are diffable.

Result — §13 determinism test, 3 independent runs of the identical binary:

| Comparison | Differing lines |
|---|---|
| run1 vs run2 | **0** of 5400 |
| run1 vs run3 | **0** of 5400 |

Bitwise identical. The computation is deterministic. The earlier
"non-deterministic first failing stage / differs across the six tiles" report
was produced entirely by the instrument.

This doubles as §11's serialization probe: draining both streams at every stage
boundary is maximal per-stage serialization, and **the NaN persisted under it**.
A missing cross-stream ordering edge *within* a stage is therefore excluded.

## 3. §2 Event lifetime and reuse audit — NO DEFECT

Protocol (documented in `fv3_gpu_halo_peer.hpp`), all ordering via
`cudaEventRecord`/`cudaStreamWaitEvent`:

```
compute_stream --(event_compute_done)--> transfer_stream    producer visible
pack on transfer_stream ; record pack_done
------- BARRIER 1: every member has ENQUEUED its pack -------
transfer_stream waits each neighbour's pack_done
D2D transfer, then unpack, on transfer_stream ; record xfer_done
compute_stream waits xfer_done                              consumer ordered
------- BARRIER 2: every member has ENQUEUED its transfer -------
transfer_stream waits every member's xfer_done
```

| Event | Creator | Recorded on | Waited by | Reuse | Generation-safe? |
|---|---|---|---|---|---|
| `event_compute_done` | owning tile | compute_stream | own transfer_stream | per group/substep/call | YES — same-tile, same-thread, record precedes wait |
| `pack_done` | owning tile | transfer_stream | neighbours' transfer_stream | per group/substep/call | YES — protected by BARRIER 1 |
| `xfer_done` | owning tile | transfer_stream | own compute_stream + all members | per group/substep/call | YES — protected by BARRIER 2 |

`pack_done`/`xfer_done` are one object per member reused for every group,
substep and top-level call. That is nonetheless safe: `cudaStreamWaitEvent`
captures the event's *recorded state at the time of the host call*, and no
member can re-record an event until after a barrier that all of its waiters
have already passed. No ABA window exists.

Independent corroboration: `compute-sanitizer --tool racecheck` reports
**0 hazards, 0 errors, 0 warnings** on the binary that still produced NaN.

## 4. §3 Barrier generation audit — NO DEFECT

`HaloPeerGroup::barrier` is sense-reversing with a `uint64_t generation_`
counter under `mutex` + `condition_variable`; it returns `false` rather than
deadlocking when not fully attached; `abort_group()` is a one-way brake.

Barrier **symmetry** confirmed: every conditional resolve depends only on
shared configuration (`config.nord>0` for H_DIVGD, `it==1` for H_GZ_FIRST), so
all six tiles execute an identical barrier count. No misparing is possible.

Buffers: `send`/`recv` are sized to the max over live groups and `cudaMemset`
to zero once at attach.

## 5. §8 Host-thread-safety audit — NO DEFECT

Shared state across the six tile threads is confined to `HaloPeerGroup`, whose
mutable members are guarded by its mutex. `send_table()` / `member()` are read
without the mutex, but member records are immutable after attach and before
detach, so those reads are benign. No shared non-atomic counters, no
concurrently-mutated containers, no shared callback or mutable static state.

## 6. Actual defect #1 — FIXED: harness `delz`/`zh` 10x inconsistency

**File:** `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp:333`

With the diagnostic corrected, the first failure was deterministic:
`step=1` (call #2), substep `it=2`, stage `cgrid_vertical_solver`, fields `gz`
and `pkc`, at **exactly** `(i=8,j=1)` and `(i=8,j=2)` for all k — 20 cells —
identically on **every** tile (`ni=nj=15, isd=-2, is=1, ie=8`, halo_width 3,
npx 9). That is `i=ie`, `j=js` and `js+1`: the **southeast tile corner**.
A fixed spatial signature on all six tiles is not a race signature.

`zh` and `ws3` were clean at that checkpoint, so the `it>1` `gz_zh_sync_kernel`
copy is not the source; the producer is Phase 2 `update_dz_c`, which writes the
persistent `gz` directly, ending in
`gz = (gz2*area + fx_diff + fy_diff) / (area + xfx_diff + yfx_diff)` — the only
NaN-capable operation in the path (denominator can cross zero).

Root cause: the synthetic initial state was internally inconsistent. `zh`/`gz`
were built from `z = 10000 - k*1000` (1000 m per layer), but `delz` was set to
`-1000.0/NPZ` = **-100 m**, understating layer thickness by exactly 10x. Since
`gz_from_delz_kernel` reconstructs gz's **interior** from `delz` while gz's
**halo** retains the analytic `grav*z`, this left a ~9x geopotential cliff one
cell outside each tile edge. `update_dz_c` then differences `gz2` across that
cliff and the denominator degenerates — first at the southeast corner, where
the east and south cliffs compound.

Fix: `h.delz[idx] = -10000.0 / NPZ;` (= -1000 m), consistent with the `zh`
column. **Numerical semantics changed: NO** — this is test-input data in an
untracked benchmark harness, not a kernel or an algorithm. The harness's own
comment at line 1023 already documented this "consistency constraint with
delz/zh"; the initialization simply violated it.

Effect: first failure moved from **call #2** to **call #4**
(`step=3`, `it=1`, `vertical_solver`, `w`/`delz`/`zh`, 30/2250 cells).
Necessary, correct, and not sufficient.

## 7. Actual defect #2 — IDENTIFIED, NOT FIXED: H_GZ_FIRST resolves before its producer

**File:** `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1368`

```cpp
if(it==1 && !resolve(FV3_GPU_H_GZ_FIRST)) return rc;   // :459 -> :552
if(!run(dispatch.cgrid_vertical_solver)) return rc;    // :543-583
```

The exchange is enqueued **before** `launch_cgrid_vertical_solver`, but the
interior values it packs are written by `gz_from_delz_kernel` **inside** that
stage. The halo is therefore filled from the previous call's (or, on call #1,
the analytic initial) gz interior and is never made consistent with the
interior the stage then computes. Authoritative Fortran splits this across the
C-grid vertical group (`:459` start, `:552` complete) precisely so the exchange
lands *between* the gz reconstruction and the gz/zh sync.

The in-code justification at lines 1309-1312 states that *"on the only topology
resolve_halo will accept, the exchange copies nothing, so resolving before
rather than inside the stage skips no operation."* **That assumption is false
for a real six-tile cubed sphere**, where all four edge neighbours are
populated and the exchange does copy. It held only for the degenerate topology
earlier tests used. This is a genuine ordering defect in the production path,
independent of the harness.

It is an ordering/staleness defect, not a NaN generator on its own; with
defect #1 fixed the residual interior/halo inconsistency is the un-reconstructed
`horiz` perturbation (~0.02*grav*10000 ≈ 2.0e3 m^2/s^2 at the edge), which is
consistent with the observed slower blowup (call #4 rather than call #2) and
with the failure region now *growing* outward from the east/southeast edge
(`i=6..8, j=1..4`) rather than sitting on two fixed columns.

**Fix not applied.** The narrow correct fix requires splitting
`dispatch.cgrid_vertical_solver` so the exchange can be resolved between the gz
reconstruction and the gz/zh sync. That changes the dispatch stage contract and
should be done deliberately, not as a tail-end edit.

## 8. §7 Halo group audit

| Group | Producer visible to pack | Pack->xfer dep | Consumer dep | Generation safe | Status |
|---|---|---|---|---|---|
| H_DWIND_IN | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK |
| H_W_IN | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK |
| H_DIVGD | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK (nord>0 only, symmetric) |
| **H_GZ_FIRST** | **producer runs AFTER resolve** | BARRIER 1 + pack_done | xfer_done -> compute | YES | **DEFECT — see §7** |
| H_CWIND | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK — resolved after `cgrid_pressure`, whose stage writes uc/vc; ordering correct |
| H_DSCALAR | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK |
| H_ZH | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK |
| H_PKC2 | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK |
| H_HEAT | event_compute_done | BARRIER 1 + pack_done | xfer_done -> compute | YES | OK (post-loop) |
| H_Q_INLINE | n/a | n/a | n/a | n/a | Never exchanged (inline_q outside supported subset) |
| H_DWIND_NEXT | n/a | n/a | n/a | n/a | Same physical exchange as next iteration's H_DWIND_IN |

Note: the exchange is **four-edge only** (`for q in 0..3`); diagonal/corner
halo blocks are never exchanged. Corner halo cells are produced instead by
`fill_4corners`/`fill2_4corners`. All four corner flags are set on every tile
in this configuration, so the SE-only localization of defect #1 was not a
corner-flag asymmetry.

## 9. Sanitizers (§14)

Run on the **pre-fix** binary (which still produced NaN), absolute path
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
(not on PATH under the validated module stack).

| Tool | Result |
|---|---|
| memcheck | **PASS** — `ERROR SUMMARY: 0 errors` |
| racecheck | **PASS** — `0 hazards displayed (0 errors, 0 warnings)` |
| initcheck | **NOT RUN here** — see below; re-run and closed in 3A.11S |

No new signatures from memcheck/racecheck.

The initcheck failure here was misdiagnosed as "binary mid-rebuild". The real
cause, found in 3A.11S: `--track-unused-memory` is a **bare boolean flag**, so
`--track-unused-memory no` made `no` the target application, producing
`Target application doesn't exist or is not a valid executable`. Omit the flag.

initcheck was the one remaining direct test of the un-zeroed `WorkspaceId::cgrid`
arena (`gz2/xfx/yfx/fx/fy` in `update_dz_c` are arena scratch, `cudaMalloc`'d and
never zeroed, and invisible to the registered-field diagnostic). It has since been
run exhaustively in 3A.11S and that hypothesis is **REFUTED** — see §11 item 1.

## 10. Stability matrix (§12) — NOT ACHIEVED

| Case | Target | Result |
|---|---|---|
| 8x8x10 R8, 6 tiles, n_split=3 | >=50 calls | **FAIL** at call #4 |
| 16x16x10 R8 | >=50 calls | NOT RUN (§9: do not grow the grid until the small reproducer is stable) |
| 32x32x10 R8 | >=20 calls | NOT RUN |
| R4 8x8 / 16x16 | >=50 / >=20 | NOT RUN |

## 11. Remaining work

1. ~~Re-run initcheck to close out the un-zeroed workspace-arena hypothesis.~~
   **DONE in 3A.11S — hypothesis REFUTED.** The exhaustive census
   (`--print-limit 5000`, 3456 findings) reports zero uninitialized reads in
   `update_dz_c` or any other C-grid-arena consumer; the un-zeroed
   `WorkspaceId::cgrid` arena is fully written before it is read. All 3456
   findings are the two previously classified `del6_vt_flux` → `copy_corners`
   authoritative dead reads.
2. Fix defect #2: split `cgrid_vertical_solver` so H_GZ_FIRST resolves between
   the gz reconstruction and the gz/zh sync, per Fortran `:459 -> :552`.
   Correct the now-false topology justification comment at lines 1309-1312.
3. Re-run the 20-call reproducer; only then grow to the §12 matrix.
4. Re-run §13 determinism (>=5 runs) and the full R8/R4 CTest suites, and
   re-confirm the single-call Gate C1 path (§15).
5. Only then resume the 3A.11 performance benchmark (§16).

## 12. Git state

Uncommitted, intentionally retained (§18):
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` — temporary diagnostic
  (`FV3_GPU_DBG_STAGE`, `debug_check_finite_stage`, gated behind
  `FV3_GPU_DEBUG_FAIL`) plus a legitimate `WorkspaceId::vertical` sizing fix.
- ~~`cuda/dyn_core/operators/wp_d_contract.cpp` — legitimate axis-order fix.~~ **REVERTED in 3A.11S: proven wrong against authoritative Fortran, which dimensions `pe`/`peln` as (i, k, j). It broke `fv3_wp_d_contract` in both precisions.**
- `cuda/dyn_core/CMakeLists.txt` — benchmark wiring.
- `cuda/dyn_core/benchmarks/` — untracked harness, including the defect #1 fix.

Not committed and not to be committed: `build_cuda_r8_perf/`, `*.log`, `*.out`,
`*.sorted`, `sanitize_3a11r.sh`, `determinism_3a11r.sh`.

Nothing pushed.

## 13. Known instrument limitation

Six concurrent host threads interleave `fprintf` output, so `[FV3DBG-IDX]`
lines can be spliced mid-line by another tile's `[FV3DBG]` line. Worked around
with `grep -o`; would need a single buffered write or a per-line mutex.
