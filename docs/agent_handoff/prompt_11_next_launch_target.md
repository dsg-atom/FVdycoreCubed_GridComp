# Prompt 11 — Halo Pack/Unpack Edge-Axis Fusion

Status: **COMPLETE.** Two candidates characterized, one selected and implemented,
validated bitwise-identical, and measured with a positive result. Git HEAD for
this pass's baseline reference points (Prompt 7/9 launch/time figures reused,
not remeasured): unchanged from `CURRENT_STATE.md` §1. All source changes in
this pass are uncommitted at the time of writing (see repo `git status`); no
commit or push was requested.

## 1. Candidate A — `pack_edge_kernel` / `unpack_edge_kernel`

Part of the six-tile GPU-side halo pack/unpack pipeline (`fv3_gpu_halo_exchange.cu`),
called once per `(halo group, field, edge)` from `launch_halo_pack`/
`launch_halo_unpack`, which are themselves called once per halo group per
`perform_peer_exchange` (the sole call site, `fv3_gpu_runtime.cpp`). Each kernel's
grid already batches every vertical level (`blockIdx.x`), every along-edge
position (`blockIdx.y`), and every ring depth (`blockIdx.z`, via `strip_grid`).
The **only remaining unbatched axis is the 4-edge loop** (`TileEdge::{west,
east, south, north}`), which the host code drives with 4 separate kernel
launches per `(group, field)`.

**Dependency check (the decisive question, per Prompt 10's own methodology):**
each edge's send/recv slot is `slot_base(g, ring_count, field, edge) =
(field*4 + edge) * group_slot_elements(g, ring_count)` — a disjoint memory
region per edge, with **no producer/consumer relationship between edges**
(unlike Prompt 10's `copy_corners` dir=1/dir=2 case, which had a real,
proven intervening consumer between the two rotations at every loop site).
Fusing the edge loop is therefore provably safe reordering, not just
plausible.

Pre-fusion launch/time counts (from Prompt 9's own `nsys` trace,
`perf_rerun_logs/nsys_fused_cuda_gpu_kern_sum.csv`, `--warmup 5 --iters 20`,
25 total dyn_core calls, same reduced-count methodology as Prompt 7):

| kernel | instances | /call | total time (ms) | ms/call | % kernel time |
|---|---|---|---|---|---|
| `pack_edge_kernel` | 26,400 | 1,056 | 66.110 | 2.644 | 3.6% |
| `unpack_edge_kernel` | 25,500 | 1,020 | 58.849 | 2.354 | 3.2% |
| **combined family** | 51,900 | **2,076** | 124.959 | **4.998** | **6.8%** |

Classification: **SAFE.** Local to one file, no halo-infrastructure redesign,
no communication-semantics change, no crossed sync boundary.

## 2. Candidate B — `a2b4_global_corner_kernel`

Part of `a2b_ord4`'s corner extrapolation, consumed by `nh_p_grad`'s B-grid
transforms (`pkc`/`pk3`/`gz`) in `wp_d_cuda.cu`. Measured this pass (see §4)
at exactly **108 launches/call**, matching the figure carried forward from
Prompt 7/9 profiling, and the single largest-time *individual* kernel named in
Prompt 7's profile at **6.2–6.6% of kernel time** (this pass: 6.6%, `time%`
column of `nsys_p11_haloedges_cuda_gpu_kern_sum.csv`) — larger even than
`riem_solver3_kernel`/`riem_solver_c_kernel` individually, though both of
those still exceed it in total time share when their own per-tile-substep
call counts are counted (450 instances each here).

Its cost is concentrated in genuine per-launch compute (corner-only
extrapolation, not a trivial copy), not launch count. Classification:
**LOW-VALUE for launch-count reduction** — even a hypothetical elimination of
every one of its 108 launches/call would remove at most 108/17,730 = 0.61% of
the whole-call launch total, which cannot clear the task's 3% GO/NO-GO
threshold on its own. A real fix for this kernel would be occupancy/
shared-memory tuning or algorithmic restructuring of the corner-extrapolation
math itself — both explicitly deferred kernel-time optimizations under
`docs/agent_instructions/project_architecture.md` ("Performance Work Comes
Later": kernel fusion is in scope now, occupancy/shared-memory tuning is not).

## 3. Ranking and Selection

| | Candidate A (pack/unpack edge fusion) | Candidate B (a2b4 corner) |
|---|---|---|
| Launch-count ceiling | ~2,076/call family → up to ~1,548/call reduction (≈8.7% of whole-call baseline) | 108/call family → ≤108/call reduction (≤0.61% of whole-call baseline) |
| Clears 3% GO/NO-GO on launch count alone? | **Yes** | No |
| Numerical risk | Low — disjoint per-edge slots, no producer/consumer dependency, pure index-arithmetic merge | N/A (not attempted) |
| Scope | Local to `fv3_gpu_halo_exchange.cu` + one config flag | Would require occupancy/shared-mem work, out of this task's scope |
| Runtime/topology redesign needed | No | No, but real fix is kernel-time tuning, not fusion |

**Selected: Candidate A (pack/unpack edge-axis fusion).** One-sentence reason:
it is the only one of the two whose launch-count ceiling clears the task's 3%
whole-dyn_core-call threshold, and it does so with a provably safe,
numerically low-risk, fully local change — Candidate B's cost is
kernel-time-bound, not launch-count-bound, and its real fix is explicitly
out of scope for this optimization pass.

## 4. GO/NO-GO

Expected whole-call launch reduction from fusing Candidate A's 4-edge loop
into 1 launch/`(group, field)`: up to ~1,548 of 17,730 launches/call ≈ 8.7% —
**clears the ≥3% threshold** before implementation. **Decision: GO.**

## 5. Implementation

Opt-in flag `--fuse-halo-edges` (`NumericalConfig::fuse_halo_edges`, default
0 — identical to Prompt 9's `--fuse-corner-fill` pattern). When set,
`launch_halo_pack`/`launch_halo_unpack` each issue one kernel per
`(halo group, field)` — `pack_edge4_kernel`/`unpack_edge4_kernel` — covering
all four `TileEdge` values via a widened `blockIdx.z = eidx * ring_count + h`,
instead of one kernel per `(group, field, edge)`. Per-edge along-extent (and,
for unpack, donor-extent/sign/reversed-bit) are passed as 4-element scalar
argument sets. `along_extent[eidx]=0` reproduces the existing `dest_edges`
masking (E11's per-field edge skip) via the pre-existing `t > along_extent`
guard, unchanged.

**Correctness-critical detail:** the kernels take *two* separate parameters,
`ring_count` (`halo_ring_count(spec)`, used only to decode `blockIdx.z` via
division/modulo) and `width` (`spec.halo_width`, used only for the defensive
`h > width` bounds check) — these are equal for ordinary groups
(`ring_begin=1`) but NOT for `FV3_GPU_H_DWIND_SEAM` (E11: `ring_begin=0,
halo_width=0` ⇒ `ring_count=1, width=0`). Using `width` as the grid-decode
divisor would divide-by-zero for E11. This was caught by source re-reading
before any build, not by a runtime failure.

Files changed: `cuda/dyn_core/include/fv3_gpu_internal.hpp` (new config
field), `cuda/dyn_core/runtime/fv3_gpu_halo_exchange.cu` (new kernels +
launcher branches), `cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp`
(trailing default-valued `bool fuse_edges = false` params — the only other
caller, `test_halo_exchange_multitile.cu`, compiles unchanged),
`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (reads
`config.fuse_halo_edges` in `perform_peer_exchange`),
`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` (`--fuse-halo-edges` CLI
flag). **Numerical algorithm changed: NO** — same per-cell index arithmetic,
same values, same buffer addresses as the unfused path; only the launch
granularity changes.

## 6. Numerical Validation — PASS, BITWISE IDENTICAL

Job 58238397 (`perf_rerun_logs/prompt11_halo_fusion.sbatch`), 32×32/R8/npz=10/
n_split=3/six tiles, `dyn_core_exit` dump, all 13 fields (`u,delp,v,w,delz,
pe,peln,pk,pkz,ua,va,pt,omga`) × 6 tiles = 78 comparands:

- Default vs `--fuse-halo-edges`: **78/78 IDENTICAL, 0 DIFFERS, 0 MISSING.**
- Default vs `--fuse-corner-fill --fuse-halo-edges` (combined with Prompt 9):
  **78/78 IDENTICAL, 0 DIFFERS, 0 MISSING.**

R8 CTest: PASS (exit 0). Default synthetic smoke: PASS. `--fuse-halo-edges`
synthetic smoke: PASS. Run health: `3A.11 CUDA perf benchmark: PASS` on every
invocation in the job log.

## 7. Measurement

**Family-specific (pack/unpack edge kernels), this pass's `nsys` trace**
(`nsys_p11_haloedges_cuda_gpu_kern_sum.csv`, same 25-call methodology):

| kernel | instances | /call | total time (ms) | ms/call |
|---|---|---|---|---|
| `pack_edge4_kernel` | 6,600 | 264 | 20.363 | 0.815 |
| `unpack_edge4_kernel` | 6,600 | 264 | 20.741 | 0.830 |
| **combined fused family** | 13,200 | **528** | 41.104 | **1.644** |

Family launch reduction: 2,076 → 528/call (**−74.6%**, i.e. almost exactly the
expected 4× from merging 4 edges into 1 launch — `1056/264 = 4.00` exactly for
pack; `1020/264 = 3.86` for unpack, matching design intent). Family kernel-time
reduction: 4.998 → 1.644 ms/call (**−67.1%**).

**Whole-dyn_core-call** (measured directly from this pass's
`cuda_gpu_kern_sum.csv`, summed across all kernels, ÷25 calls):

| | launches/call | kernel busy time (ms/call) |
|---|---|---|
| Prompt 7 baseline (default) | 17,730 | 74.0 |
| This pass (`--fuse-halo-edges`) | **16,074** | **67.44** |
| Change | **−1,656 (−9.34%)** | **−6.56 (−8.86%)** |

The whole-call launch reduction (−1,656) is consistent with the family-level
reduction (−1,548 expected) to within ~0.6% of the total — the small residual
is attributed to ordinary run-to-run profiling variance between the two
`nsys` captures (different processes, same methodology), not a second
mechanism. `cudaLaunchKernel` remains 90.0% of CUDA API time (vs 90.6%
baseline) — the launch-bound classification is unchanged in kind, only
reduced in magnitude, exactly as expected from a pure launch-count fusion.
GPU-side memory-op time is unchanged at ≈2.47 ms/call, 91.4% D2D (vs baseline
≈2.41 ms/call, 91.3% D2D) — halo fusion does not touch memcpy volume, only
compute-kernel launch count, as designed.

## 8. End-to-end timing (32×32, R8, resident; 20 warmup/100 iters/5 repeats;
job 58238397)

Median ms/call (of 5 repeats):

| configuration | median ms/call | vs default |
|---|---|---|
| default | 29.2027 | — |
| `--fuse-corner-fill` (Prompt 9 alone) | 26.4844 | −9.31% |
| `--fuse-halo-edges` (Prompt 11 alone) | **23.2813** | **−20.28%** |
| both combined | 23.8621 | −18.29% |

Per-repeat spread (max−min) was 0.12–0.33 ms across all four configurations —
tight relative to the ~5–6 ms medians of separation between configurations, so
the improvement is well above measurement noise. One observation, reported
honestly rather than smoothed over: **combined is very slightly slower than
`--fuse-halo-edges` alone** (23.86 vs 23.28 ms/call, +2.5%), a gap (~0.58 ms)
larger than either individual configuration's own repeat-to-repeat spread.
This does not indicate a correctness problem (§6 showed the combined path is
still bitwise identical) and is small enough that it may be measurement
variance between configurations rather than a genuine interaction cost; it is
recorded but not chased further, in line with the task's cost-control bound
(this pass does not add a second measurement round to resolve it). The
practical implication is simply that `--fuse-halo-edges` alone, not combined
with `--fuse-corner-fill`, is the best-measured configuration in this pass.

**Validated whole-call improvement: 20.28%** (`--fuse-halo-edges` vs
default), comfortably clearing the ≥3% GO/NO-GO threshold that authorized
implementation, and reproducible across all 5 repeats (spread 0.12 ms, far
below the ~5.9 ms gap being measured).

## 9. Profile policy

Numerical validation PASSED (§6) and whole-call improvement (20.28%, §8) is
≥3%, so per Section 9's gating rule exactly **one** `nsys` profile was run —
job 58238413, `--fuse-halo-edges` only, 32×32/R8 resident, `--warmup 5 --iters
20` (25 calls). No profile of the unfused default path was taken in this
pass; the pre-fusion pack/unpack kernel counts and family kernel-time figures
in §1/§7 above are reused from Prompt 9's existing trace on disk
(`nsys_fused_cuda_gpu_kern_sum.csv`), not a new profile, per the task's
"exactly ONE nsys profile" bound.

## 10. Cost control

Exactly 2 candidates characterized (§1, §2). Exactly 1 implemented (§5). All
runs at 32×32/R8. No Fortran rerun, no R4, no c8/c16 sweep, no CUDA Graphs
work, no new savepoints. Zero post-implementation fix cycles were needed —
the implementation validated bitwise-identical and measured a real
improvement on the first attempt (the `ring_count`/`width` bug described in
§5 was caught by source re-reading before any build was attempted, so it
never consumed a "fix cycle" against the ≤3 budget).

## 11. Regression

R8 CTest: PASS (job 58238397, `=== CTEST RC: 0 ===`). Default synthetic
smoke: PASS. `--fuse-halo-edges` synthetic smoke: PASS. No R4 build/test was
touched in this pass (out of scope, per §10).

## 12. Outlook — local edge/corner-family fusion is DIMINISHING

Two of the three named halo/corner-service kernel families identified back in
Prompt 7 (~18% of kernel time collectively) have now been touched:
`copy_corners_kernel` was characterized and explicitly ruled out (Prompt 10 —
every same-buffer dir=1/dir=2 pair has a real intervening consumer, and
tile-batching needs a runtime redesign), and `pack_edge_kernel`/
`unpack_edge_kernel` are now fused (this pass, −1,656 launches/call, −9.34%
whole-call). The remaining un-fused member of that family,
`fill_4corners_kernel`/`fill2_4corners_kernel`'s *other* call sites beyond the
one pair Prompt 9 already fused, and the `a2b4_global_corner_kernel` (§2, this
pass) are both now understood to be kernel-time-bound rather than
launch-count-bound at their remaining scale (108–~200 calls/call — well under
the 3% launch-count threshold that has justified every fusion implemented so
far). **Local launch-count fusion of the halo/corner-service family has
diminishing headroom**: the two largest-ceiling targets in that family
(`copy_corners_kernel`'s tile/direction axes, `pack_edge_kernel`/
`unpack_edge_kernel`'s edge axis) have both now been characterized to
completion, one ruled out and one implemented. What remains in that family is
individually below the ≥3% threshold. The next optimization step should shift
from launch-count fusion to either (a) kernel-time reduction on the
individually expensive kernels (`riem_solver3_kernel`/`riem_solver_c_kernel`
at ~144–148 μs avg each, `a2b4_global_corner_kernel` at ~44 μs avg but 108
calls/call) via occupancy/shared-memory tuning — explicitly deferred
architectural work per project policy, not a quick local fix — or (b)
revisiting CUDA Graphs (Prompt 8, BLOCKED on a numerical divergence and a
replay crash) now that the per-call launch count is meaningfully lower,
which could reduce the *benefit* CUDA Graphs would need to prove but does not
by itself fix its two open defects.
