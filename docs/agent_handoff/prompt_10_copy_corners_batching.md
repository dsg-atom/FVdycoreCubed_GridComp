# Prompt 10 — `copy_corners_kernel` Batching: Characterization + Decision (NO IMPLEMENTATION)

Git HEAD at time of this pass: `b456897` (`Close Gate C2 and add initial CUDA
launch optimizations` — the Prompt 6B `omga` finalization + Prompt 9 kernel
fusion, now committed by the user between sessions; not authored by this
pass). Working tree otherwise clean (only pre-existing untracked scratch files
at repo root). **No source file was modified in this pass.** No Slurm job was
submitted. Explicitly excluded from this pass per the task: CUDA Graphs work,
R4, c8/c16 sweep, `ncu`, new savepoints.

## 1. Characterization (from source, no new profiling)

Reused Prompt 9's existing nsys trace (`--warmup 5 --iters 20`, 25 iterations,
32x32/R8/resident) rather than re-profiling: **1,140 `copy_corners_kernel`
instances / dyn_core call** (whole six-tile call, i.e. summed across the six
tiles' concurrently-running `fv3_gpu_dyn_core_step` calls — see §2 for why
that unit matters). `copy_corners_kernel` itself already batches **all 10
vertical levels into one launch** via `blockIdx.z = k` + `LevelFilter`
gating — per-level fusion is not an available lever; it was already done.

There is exactly **one** production launcher,
`launch_copy_corners_cuda` (`wp_b_cuda.cu:681`), called from **11 call
sites**: 10 inside `wp_b_cuda.cu` (three operator families) and 1 inside
`fv3_gpu_production_dispatch.cu` (`corner_edge_service_cuda`, the `del2_cubed`
adapter added in a prior Prompt 6B pass). Reconstructed per-tile,
per-dyn_core-call inventory (production defaults: `nord_cap = 2`, `nf_ke = 3`,
`n_split = 3`), read directly from the call sites and their callers
(`wp_b_dsw_cuda.cu`, `wp_b_contract_cuda.cpp`, `wp_b_divergence_cuda.cu`):

| Operator family | Call sites | Calls/invocation | Invocations/substep | Per-substep | ×3 substeps |
|---|---|---|---|---|---|
| `fv_tp_2d` direct (`update_dz_d`: `z2` damped, `zh` undamped) | 2 | 2 (dir 2, dir 1) | 2 | 4 | 12 |
| `fv_tp_2d` via `scalar_transport` (`d_sw`: `delp`, `w`, `pt`, `vort`) | 4 | 2 | 4 | 8 | 24 |
| `del6_vt_flux` (`z2`, `wk`, `w`) | 3 | 10 (`nord_cap=2`: outer 4 + pass-loop 6) | 3 | 30 | 90 |
| `deln_flux` per-level tail (`delp`, `pt`) | 2 | 10 (same shape as `del6_vt_flux`) | 2 | 20 | 60 |
| `corner_edge_service` (`del2_cubed`/`heat_source`, `nf_ke=3`) | 1 | 6 (3 passes × 2 dirs) | once, `it==n_split` only | — | 6 |
| **Total, one tile, one dyn_core call** | | | | | **≈192** |

×6 tiles ≈ 1,152 — within ~1% of the empirical 1,140/call, which
cross-checks the reconstruction (small gap consistent with vort/`deln`
branches whose `deln_flux_enabled` is 0). By direction, calls split
essentially 50/50 dir=1 vs dir=2 (every rotation is paired). By tile: uniform
(all six tiles are structurally identical topology; only `sw/se/ne/nw_corner`
booleans and buffer contents differ). By field: `delp`/`pt`/`w`/`z2` are the
heaviest single-field contributors (10 calls each, via `del6_vt_flux`/`deln`),
`vort`/undamped-`zh` the lightest (2 each).

## 2. Batching-dimension analysis (all four ruled out at meaningful scale)

**(A) Batch multiple tiles into one launch — RULED OUT, redesign-level.** The
six tiles are not batched today by any shared kernel-launch surface: the
benchmark harness (`run_small_perf_bench.cpp:3277-3303`, `run_one_repeat`)
spins up **six independent host threads**, each owning its own
`Member`/`fv3_gpu_context` and calling `fv3_gpu_dyn_core_step` on its own
device arena/stream concurrently. There is no per-tile-batched view struct
anywhere in the WP-B contract. Merging tiles into one launch would mean
restructuring the persistent per-tile arena/context model into a
multi-tile-aware one — squarely the Section 9 HARD STOP ("general
halo/topology redesign is required"), not attempted.

**(B) Batch multiple corner ops for the same field/direction — RULED OUT by a
real producer/consumer dependency, confirmed by reading every loop site.** In
both `del6_vt_flux_cuda`'s pass loop (`wp_b_cuda.cu:845-873`) and
`deln_flux_pass`'s iteration loop (`:958-986`), a `dir=1` copy_corners call is
followed by a flux kernel that *consumes* that exact rotation, then a `dir=2`
call *overwrites the same corner cells with a different rotation formula*,
consumed by a second flux kernel. `dir=1` and `dir=2` do not commute on the
same buffer and cannot be reordered or merged.

**(C) Batch multiple fields sharing identical geometry — RULED OUT for the one
candidate group that looked promising.** The four `scalar_transport` calls in
`d_sw` (`wp_b_dsw_cuda.cu`, stages S7/S11/S14/S34: `delp`, `w`, `pt`, `vort`)
looked like independent same-shape fields at first, but reading the
intervening stages proves a **strict sequential chain**: S7 (`delp`
transport) produces `fx`/`fy` → S8 (`flux_capacitor`) turns those into
`cx,cy,mfx,mfy` → S10/S11 (`w` damping + transport) consumes those mass
fluxes → S12 updates `w` using `delp` → S14 (`pt` transport) consumes the
*updated* `delp` and S8's fluxes. These are not parallel/independent fields at
a shared pipeline position; they are consecutive stages of one timestep. The
three direct `del6_vt_flux_cuda` calls (`z2`, `wk`, `w`) live in three
different subsystems (`update_dz_d`/contract, divergence, `d_sw`) invoked at
different orchestration points; no evidence was found that their relative
timing is provably reorderable, and the task instructs not to cross an
unproven ordering requirement.

One genuine, narrower opportunity **was** found and quantified, then rejected
on cost/benefit: within `del6_vt_flux_cuda`'s and `deln_flux_pass`'s outer
`for nord in 0..nord_cap` loop, each pass's `del6_init_kernel`/`deln_init_kernel`
writes a **disjoint** vertical-level subset of the same buffer (partitioned by
`ndif(k)==nord`), and the copy_corners rotation formula itself does not depend
on `nord` — only the `LevelFilter` gating a pass's kernels does. In principle
the `nord=1` and `nord=2` outer corner calls (4 of the 5 relevant call sites'
10 copy_corners calls) could be hoisted and merged into one call per direction
using a new "any active nord > 0" `LevelFilter` mode, since they write/read
disjoint levels. Quantified ceiling: **≈2 launches saved per site per
direction-pair, 5 sites (3 `del6_vt_flux` + 2 `deln` per-level tails), 3
substeps ⇒ ≈30 launches/tile ⇒ ≈180/six-tile-call**, i.e. **≈16% of the
copy_corners family** but **≈1.0% of the whole-dyn_core 17,730-launch
total** — below what Prompt 9 already showed is distinguishable from
run-to-run wall-clock noise (a 2.0%-of-total launch reduction there produced
no defensible wall-clock signal). Implementing it would also require a new
`LevelFilter` mode and a loop-nest restructuring of two Gate-C2-validated
damping operators used at 5 call sites — real risk to re-validate for a
sub-noise-floor payoff.

**(D) Fuse adjacent copy_corners where dependencies allow — RULED OUT**, same
evidence as (B): every adjacent same-buffer `dir=1`/`dir=2` pair found in the
codebase has a real intervening consumer.

## 3. Decision

No batching dimension survives Section 2's own constraints ("do not cross
producer/consumer dependencies," "do not cross required stream/event
boundaries," "do not require general halo/topology redesign") at a scale that
would plausibly clear wall-clock noise, per Prompt 9's own measured noise
floor. Per Section 9's HARD STOP list ("batching gives negligible launch
reduction") and the task's own permission ("do not force [the 50%] target if
dependencies prove otherwise"), **no code was written**: an unsafe or
sub-noise-floor change is not preferable to stopping with a documented
negative result. This is a characterization/decision deliverable, not an
implementation.

## 4. What would change this conclusion

- If a future pass revisits **Option A** by making the six-tile execution
  model itself batched (a genuine architecture change, out of scope for a
  single-kernel-family task), tile-batching would dominate over anything
  found here — `copy_corners_kernel`'s total instance count is
  tile-count-multiplied by construction.
- The del6/deln outer-nord merge in §2 remains available as a small,
  well-understood, low-但-nonzero-value follow-on if a *different* motivation
  (not raw launch count) makes it worthwhile — e.g. if it's bundled with other
  changes to the same operators.
- Higher-leverage unfused kernel families identified by Prompt 7/9 and still
  open: `a2b4_global_corner_kernel` (108 calls/call, 6.2% of kernel *time*,
  vs. copy_corners' 5.1%), `pack_edge_kernel` (1,056 calls/call, 3.6%), and the
  halo/corner-service family generally (~18% of kernel time per Prompt 7).
  `pack_edge_kernel` in particular has a higher call count than `copy_corners`
  and has not been characterized at all yet.

## 5. Regression

No source file was modified, so no rebuild/ctest/smoke rerun was performed
(Section 11 conditions it on "if shared code changed"). Gate C2 and the
existing Prompt 9 fusion validation are unaffected.
