# Prompt 52 — CUDA Port Plan: `fv_mapz_mod` Vertical Remapping

Date: 2026-09-21. Model: Sonnet (no Opus escalation — see Decision).
Status: **PLANNING/MAPPING ONLY.** No source changed. No GPU jobs. No
Nsight rerun. Static source analysis only, per task scope.

Baseline: branch `optimization/cuda-fv3-performance`, HEAD
`33e10c50b741fc53817fe035a1f2abe64c34bb65`, tag `cuda-perf-baseline-c512`
→ same commit. `git status --short` clean of tracked changes (only
pre-existing untracked scratch files/dirs, unrelated to this task).

Source inspected (targeted, not broad):
`GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`
(3949 lines) and its production caller,
`.../model/fv_dynamics.F90` (call site + surrounding `n_map` loop,
lines 626-1019). `fv_fill.F90::fillz` inspected for the tracer mass-fill
dependency. No physics/GEOS-interface files outside this call chain were
read.

---

## STAGE 0 — Performance Model Restatement

Measured decomposition of the C512 `DYN_CORE`-window (Prompt 49/51,
19.87 s window, 6-call steady-state):

| Bucket | % of window |
|---|---|
| GPU kernel compute (171 kernels) | 3.9% |
| Halo-transport blocking sync (`cudaEventSynchronize`) | 14.4% |
| Host-staged `cudaMemcpyAsync` (blocking-like) | 7.9% |
| `cudaLaunchKernel` overhead | 1.7% |
| Other CUDA API | 0.4% |
| **FV3 Fortran CPU compute (Prompt 51 decomposition of the residual)** | **64.66%** of window (83.9% of the 77.7% untraced residual) |
| — of which `fv_mapz_mod` alone | **18.67% of total window** |
| — of which `fv_tracer2d_mod`/state-conversion/HS physics/transcendental math (remainder) | ~46.0% of window |
| Intel MPI progress (remaining measured fraction of the residual) | ~13.17% of the residual ≈ 10.2% of window |

Amdahl ceiling for eliminating `fv_mapz_mod` alone (f = 0.1867):

```
speedup_max = 1 / (1 - f) = 1 / (1 - 0.1867) = 1 / 0.8133 = 1.2296
```

**Upper bound only, not an expected result**: even an infinitely fast
`fv_mapz_mod` caps the whole-`DYN_CORE` speedup at **≈1.23x** — well short
of the ≥2x (≤3187 ms/call) target. This is expected and consistent with
Prompt 51's framing: fv_mapz is "the first coverage candidate," not a
standalone path to 2x. Reaching 2x requires eventually addressing
multiple CPU-coverage buckets (fv_mapz + tracer transport + halo-sync),
not this one alone. See Stage 11 for finite-speedup scenarios.

---

## STAGE 1 — Hot Routines Table

Resolved from Prompt 51's CPU-sampling call-stack symbols against this
segment's source read of `fv_mapz.F90`'s subroutine boundaries (line
ranges below are this file's actual definitions, not estimates):

| Routine | Prompt 51 samples | LOC | Called from | Purpose |
|---|---|---|---|---|
| `map_scalar` | 2724 | 153 (1519-1671) | `Lagrangian_to_Eulerian`, 9 call sites: `te`/`pt` (:393/400), `delz` (:475), `w` (:487), `u`/`mfy`/`cy` (:575/581/588), `v`/`mfx`/`cx` (:622/628/635) | Per-column vertical remap of one non-tracer scalar/vector field from the Lagrangian (deformed) grid to the new Eulerian pressure grid: builds a sub-grid profile then integrates it onto target layers. |
| `scalar_profile` | 663 | 408 (1881-2288) | `map_scalar` (when `kord>7`, all 9 call sites above use `kord_mt`/`kord_wz`/`kord_tm`, all validated production values are ≤`8` per project defaults — confirm `kord>7` path is live), `mapn_tracer` (once per tracer, unconditional) | Builds a monotonicity-constrained PPM-style 4-coefficient (cell mean + 2 edges + curvature) vertical sub-grid reconstruction per column via a tridiagonal forward/backward sweep in `k`. |
| `Lagrangian_to_Eulerian` (own frame, not children) | 523 | 1110 (130-1239) | `fv_dynamics.F90:952` | Orchestrator: builds per-`j`-row local pe/dpe arrays, calls `map_scalar`/`mapn_tracer`, and — in its own frame, not a sub-call — runs the pointwise `pt`→temperature conversion (`exp`/`log` transform, :387-389) and total-energy conservation correction. This is the routine whose own-frame time carries the `__bwr_log/exp/pow` transcendental-math samples Prompt 51 attributed to FV3 core dynamics. |
| `mapn_tracer` | 413 | 112 (1673-1784) | `Lagrangian_to_Eulerian`, once per `j`-row, all `nq` tracers together (:441) | Same PPM remap algorithm as `map_scalar`, generalized to remap all `nq` tracers in one call (shared profile-building loop over `iq`, shared search/integrate loop with `qsum(nq)` accumulator), followed by `fillz` mass-conservation fill and a `fill`-gated negative-tracer correction. |

**Not individually distinguishable in Prompt 51's samples** (module-level
or below threshold): `ppm_profile` (used when `kord<=7`; algorithmically
parallel to `scalar_profile` but simpler/non-monotonic), `pkez`,
`compute_total_energy`, `moist_cv`/`moist_cp` (dry-branch `nwat==0` is
the currently-validated HS config's live path — see Stage 12), `fillz`.
These were not treated as a single black box regardless: their call
sites and roles are identified above/in Stage 2-3 from source, even
where Prompt 51's sampling resolution did not separate their cost.

**map_scalar + scalar_profile alone ≈ 78% of the ~4323 total resolved
fv_mapz-attributed samples** — the two-routine remap engine described in
Stage 7 is where the measured cost concentrates.

---

## STAGE 2 — Authoritative Call Graph

All 10 required questions, answered from source (`fv_dynamics.F90:626-1019`,
`fv3_gpu_dyncore_bridge.F90` call comments at `fv_dynamics.F90:735-776`):

1. **What calls `fv_mapz`?** `fv_dynamics.F90:952`, `call
   Lagrangian_to_Eulerian(...)`, inside the `n_map` loop, guarded by
   `if (npz > 4)`.
2. **How often per outer timestep?** Once per `n_map` iteration, i.e.
   `k_split` times per call to `fv_dynamics`. The validated C512 config
   uses `k_split=1` (Prompt 49 config table), so in practice: **once per
   outer timestep**.
3. **How often per `dyn_core` call?** **Exactly once, 1:1.** With
   `k_split=1` the `n_map` loop body executes once, and inside that one
   iteration the sequence is: one CUDA `dyn_core` call
   (`fv3_gpu_interface_step_dyn`) → tracer transport
   (`tracer_2d`/`tracer_2d_1L`/`tracer_2d_nested`) → one
   `Lagrangian_to_Eulerian` call. At `k_split>1` (not the currently
   validated config) the ratio remains 1:1 per iteration, just repeated
   `k_split` times.
4. **Before/after the CUDA-resident `dyn_core`?** **After.** Sequence
   confirmed by source: CUDA `dyn_core` (:748) → host pe-halo fix (:763)
   → tracer q-halo start (:773) → [optional dry-mass control, :856] →
   tracer transport (:890-908) → `Lagrangian_to_Eulerian` (:952).
   `fv_mapz` is a **companion stage after** the CUDA-replaced acoustic
   substep loop, not inside it — consistent with the CURRENT_STATE.md
   "CUDA coverage gap, not correctness gap" framing.
5. **Which arrays enter?** From the call site (:952-960): `ps, pe, delp,
   pkz, pk, q_con, u, v, w, delz, pt, q, phis, cappa, peln, te_2d, ua,
   va, omga, dp1 (as `te` scratch), ws`. All are `intent(inout)` or
   `intent(in)` host arrays already populated by the immediately
   preceding tracer-transport call (for `q`) or directly by the CUDA
   `dyn_core` exit state (for everything else).
6. **Which arrays leave?** Same set, in-place: `pk, q, delp, ps` updated
   in place to the new Eulerian grid's values; `pkz` computed
   `intent(out)`; `te` (`dp1`) computed `intent(out)` (total energy per
   layer, consumed later in `fv_dynamics` for diagnostics/energy
   conservation, not shown as feeding back into the next `dyn_core`
   call directly). `pt` becomes actual temperature (not virtual
   potential temperature) after the pointwise conversion at :387-389
   when `remap_t` is the active mode.
7. **Are those arrays already resident on GPU?** **No — they are already
   on the host** by the time `fv_mapz` runs. The comment block at
   `fv_dynamics.F90:735-747` confirms the CUDA `dyn_core`
   (`fv3_gpu_interface_step_dyn`) performs its own full-state D2H before
   returning, specifically so that `fv_tracer2d` and `Lagrangian_to_Eulerian`
   — both host-only Fortran — can consume the results. This is not a
   transfer added for `fv_mapz`'s sake; it is the existing architecture's
   already-paid transfer.
8. **Are host/device copies occurring around this phase?** Yes, but they
   predate `fv_mapz` specifically: (a) the CUDA `dyn_core`'s own D2H at
   its exit (already required for tracer transport, not fv_mapz-specific);
   (b) the flux-capacitor copy-back (`mfxL/mfyL/cxL/cyL`, comment at
   :824-826, "does NOT ride the persistent-residency path — copied back
   per call") consumed by `tracer_2d`, not by `fv_mapz` (fv_mapz's `te`
   argument reuses the `dp1` buffer as scratch, not the flux capacitors).
   No copy happens *between* tracer transport and `fv_mapz` — they run
   back-to-back on the same already-host-resident arrays. The next H2D
   (back onto GPU) happens implicitly at the *next* `n_map` iteration's
   `fv3_gpu_interface_step_dyn` call, not measured/attributed separately
   in Prompt 49/51 (out of scope here — would need a new capture).
9. **Per column/tile/tracer/vertical level?** **Per horizontal column
   (i,j), independently.** `Lagrangian_to_Eulerian` iterates `do
   j=js,je` and, within each `j`-row, calls `map_scalar`/`mapn_tracer`
   which loop `do i=i1,i2` — every column is independent of every other
   column (no i/j-neighbor references anywhere in the search/integrate
   loops read in Stage 1). Within a column, the vertical (`k`) direction
   is **not** independent (Stage 3). Tracers are batched together inside
   `mapn_tracer` (shared search index `k0`/`l`/`m` across all `nq`
   tracers per column, since they share the same source/target pressure
   grids) but are otherwise independent of each other numerically.
10. **What neighboring-column dependencies exist?** **None inside the
    profiled hot routines** (`map_scalar`, `scalar_profile`,
    `mapn_tracer`) — purely column-local. Two exceptions exist elsewhere
    in the same module, **outside** the Stage-1 hot list: (a) the
    optional total-energy conservation correction
    (`mpp_global_sum`, :1074-1115) is a genuine **cross-tile, cross-rank
    MPI global reduction**, gated on `consv > consv_min`; (b) `pkez`
    uses a `!$OMP parallel do` over `j` with no `i`/`j` cross-references
    either — still column-local. Neither reduction currently in the hot
    list requires new communication design beyond what WP-E's existing
    halo/MPI machinery already models, but a and the mpp_global_sum
    reduction is a real complication if consv-based energy correction is
    active in the target validated config (Stage 12).

**Sequence diagram** (one `n_map` iteration, `k_split=1` validated
config):

```
[GPU-resident state from previous iteration]
        |
   fv3_gpu_interface_step_dyn()   <- CUDA dyn_core (WP-A/B/C/D), device-resident
        |
   [full-state D2H — ALREADY HAPPENS, not fv_mapz-specific]
        |
   fv3_gpu_interface_pe_halo()    <- host, pe-halo ring fix
        |
   start_group_halo_update(q)     <- host/MPI, tracer halo (non-blocking post)
        |
   tracer_2d / tracer_2d_1L / tracer_2d_nested()   <- host Fortran, consumes
        |                                             mfxL/mfyL/cxL/cyL + q
   Lagrangian_to_Eulerian()       <- host Fortran, THIS TASK'S TARGET
        |    map_scalar (u,v,w,delz,te/pt,mfx,mfy,cx,cy)  x9 calls/j-row
        |    mapn_tracer (all nq tracers)                  x1 call/j-row
        |    [optional: fv_sat_adj, mpp_global_sum energy correction]
        |
   [next n_map iteration's H2D — implicit, not separately measured]
        |
   fv3_gpu_interface_step_dyn()   <- next iteration
```

---

## STAGE 3 — Numerical Algorithm Map

From source (`map_scalar`/`scalar_profile`/`mapn_tracer`, Stage 1 code
read in full or in structural detail):

- **Independent horizontal columns**: confirmed (Stage 2 Q9/Q10) — every
  `(i,j)` column remaps independently; the outer `do j=js,je` loop and
  inner `do i=i1,i2` loops carry no cross-column state.
- **Vertical loops — two distinct phases per column, each with a real
  `k`-dependency**:
  1. **Profile reconstruction** (`scalar_profile`): a genuine
     **tridiagonal solve in `k`** — forward sweep computes edge value
     `q(i,k)` from `q(i,k-1)` and a running coefficient `gam(i,k)`
     (:1932-1948, `iv /= -2` branch), then a **backward substitution**
     `q(i,k) = q(i,k) - gam(i,k)*q(i,k+1)` for `k=km,1,-1` (:1950-1954).
     This is structurally identical to the tridiagonal-solve pattern
     already implemented and GPU-validated in `riem_solver3_kernel`/
     `riem_solver_c_kernel` (per-thread sequential `k` loop within an
     otherwise-parallel `(i,j)` grid) — a direct, existing precedent for
     how to map this onto CUDA (Stage 5/6).
  2. **Search-and-integrate onto the new grid** (`map_scalar`'s
     `do 555 k=1,kn` / `do while (LP0.le.km)` loop, :1584-1667;
     `mapn_tracer`'s analogous `do 555`/`do 100 l=k0,km` loop,
     :1713-1771): for each new-grid layer `k`, walks forward through the
     old-grid layers `l` starting from `k0` (a **rolling pointer carried
     forward across `k` iterations**, reset only per-column) until the
     new layer's pressure interval is bracketed, then integrates the
     PPM sub-grid polynomial (from phase 1) analytically over the
     overlap. The rolling `k0` pointer is a **sequential-search
     optimization exploiting monotonically increasing pressure**, not a
     true numerical dependency — it can be replaced by a binary/direct
     search per `k` without changing the result, which is relevant for
     a GPU port where per-thread sequential `k0` state is fine (one
     thread per column already loops `k` sequentially) but a
     warp/block-parallelized-over-`k` design would need this
     re-expressed.
- **Reconstruction/interpolation**: PPM (piecewise parabolic method),
  4 coefficients per cell (`a4(1..4,i,k)` = cell mean, left edge, right
  edge, curvature term) — standard Lin (2004) vertical remap, matching
  the module's own header citation.
- **Monotonicity constraints**: enforced inside `scalar_profile`/
  `ppm_profile`/`cs_profile` (not separately read line-by-line in this
  pass, but present per `kord` value: `kord>7` selects the constrained
  `scalar_profile` path with `ext5`/`ext6`/`extm` monotonicity flags
  declared at :1893-1895; `kord<=7` selects `ppm_profile`, whose own
  `ppm_limiters` subroutine (3035-3111) applies a different, older
  limiter formulation). **The exact `kord>7` vs `kord<=7` branch taken
  in the validated production config was not directly confirmed in this
  pass** — `flagstruct%kord_mt/kord_wz/kord_tr/kord_tm` values were not
  re-read from the AGCM.rc/config source in this task (out of scope per
  "do not broadly inspect"); this is a **named gap**, not a guess (see
  Stage 9).
- **Mass conservation**: `mapn_tracer`'s call to `fillz` (:1773, `fv_fill.F90:51-156`)
  redistributes any negative post-remap tracer mass to/from adjacent
  levels within the same column, conserving column-integrated tracer
  mass exactly (arithmetic redistribution, no iteration/reduction). This
  is itself `k`-sequential (top-down / bottom-up passes) but still
  column-local.
- **Pressure/thickness quantities**: `pe`/`peln`/`pk`/`pkz`/`delp` are
  both inputs (defining the source Lagrangian grid) and, for `pk`/`delp`/`ps`,
  overwritten in place to reflect the target Eulerian grid.
- **Tracer/state handling**: non-tracer fields (`u,v,w,delz,pt`+flux
  diagnostics) go through `map_scalar` one field at a time; all `nq`
  tracers go through one shared `mapn_tracer` call per `j`-row.
- **Boundary treatment**: `map_scalar` supports optional GMAO-style
  top/bottom extrapolation (`optional_bot`/`optional_top`, :1600-1621) —
  branches not exercised unless the GMAO remap option is selected
  (`remap_option`/`gmao_remap` arguments at the `Lagrangian_to_Eulerian`
  signature, not traced further in this pass).
- **Reductions**: the only true reduction is the optional cross-tile/
  cross-rank `mpp_global_sum` total-energy conservation correction
  (Stage 2 Q10), gated on `consv > consv_min`.
- **k-level dependencies**: real, sequential, tridiagonal (phase 1) plus
  a monotonic rolling-pointer search (phase 2, not a true dependency).
- **Horizontal-cell dependencies**: none in the hot-list routines.
- **Exact ordering CUDA must preserve**: (a) the forward-then-backward
  tridiagonal sweep in `scalar_profile` must execute in `k` order
  (forward pass strictly needs `k-1`'s result; backward pass strictly
  needs `k+1`'s result) — this is a true numerical dependency, not
  reorderable; (b) the search-and-integrate phase's rolling `k0` can be
  reordered/parallelized (it is a search optimization) but the
  **floating-point summation order** inside the "fractional area" branch
  (:1639-1661, summing partial contributions from old layer `l` through
  `m`) must be preserved identically to match Fortran bit-for-bit or
  round-off, per the project's established R8 validation methodology
  (Stage 9).

---

## STAGE 4 — Data Residency Analysis

All arrays below are **already host-resident** by the time `fv_mapz`
runs, per Stage 2 Q7/Q8 (the CUDA `dyn_core`'s own exit D2H already
covers them, for tracer-transport's sake, not newly required by
`fv_mapz`):

| Array | Shape (per rank, npz=10) | Host currently | GPU currently | Direction needed for a `fv_mapz`-only port |
|---|---|---|---|---|
| `pe` | `(is-1:ie+1, km+1, js-1:je+1)` | Yes (post-`dyn_core` D2H + host pe-halo fix) | Written by CUDA `dyn_core`, then further corrected on host | H2D before kernel, D2H after (or kept resident — see below) |
| `delp` | `(isd:ied, jsd:jed, km)` | Yes | Yes (dyn_core writes it) | H2D/D2H around remap |
| `pk`, `pkz`, `peln` | `(is:ie, js:je, km[+1])` | Yes | Yes | H2D/D2H |
| `u`, `v`, `w`, `delz`, `pt` | `(i-range, j-range, km)` | Yes | Yes | H2D/D2H |
| `q` (tracers) | `(isd:ied, jsd:jed, km, nq)` | Yes, but written most-recently by `tracer_2d` (host Fortran), **not** by CUDA `dyn_core` (comment: dyn_core "neither reads nor writes" `q`) | **No** — `q` is a Fortran-only array on this path today | H2D of a value CUDA has never held before; this is a **genuinely new transfer**, not a reclaimed existing one |
| `ua`, `va`, `omga` | `(isd:ied, jsd:jed, km)` | Yes | Yes | H2D/D2H |
| `ws`, `phis`(hs) | 2-D | Yes | `ws` written by CUDA `dyn_core`, copied back per comment at :743-746 | H2D |
| `dp1` (`te` scratch) | `(isd:ied, jsd:jed, km)` | Yes (host-only buffer, comment :824-826: "does NOT ride persistent-residency path") | No | Output-only from fv_mapz's perspective; stays host or D2H |

**No transfer numbers were measured for this specific phase** — Prompt
49/51 measured the halo-transport transfers and the `DYN_CORE`-internal
D2H/H2D totals (734.99/724.63 ms, §8 of Prompt 49), but did **not**
separately instrument the `dyn_core`-exit-to-`fv_mapz` boundary or the
implicit H2D at the *next* iteration's `dyn_core` entry. **Not
quantified here — would require a new, separate capture, out of scope
for this planning task.**

**Secondary benefit, honestly bounded**: if `fv_mapz` (and eventually
`tracer_2d`) were ported, the non-tracer fields (`pe/delp/pk/pkz/u/v/w/
delz/pt/ua/va/omga/ws`) could in principle stay device-resident across
the `dyn_core` → tracer-transport → `fv_mapz` → next-`dyn_core` chain,
**eliminating** the current full-state D2H/H2D round trip entirely for
those arrays. This is real architectural upside, but **cannot be
claimed for a `fv_mapz`-only port**: `tracer_2d` (unported) still needs
`q` and the flux capacitors on the host, and the host-only `pe_halo`
fix (:763) still needs `pe`/`delp` on the host too, in between. A
`fv_mapz`-only port would therefore likely need its **own** H2D/D2H
pair bracketing just the remap call, not a free ride on the existing
transfer, until `tracer_2d` is also addressed. This should not be
assumed as a free performance win without measurement.

---

## STAGE 5 — GPU Parallelization Analysis

Candidates evaluated against the Stage 3 algorithm (column-independent
horizontally, sequential-tridiagonal + monotonic-search vertically):

| Candidate | Parallelism | Vertical deps | Coalescing | Register pressure | Shared mem | Ordering | npz portability |
|---|---|---|---|---|---|---|---|
| **A. One thread per horizontal column** | `(ni x nj)` threads, one per `(i,j)` | Sequential `k` loop inside the thread — direct analogue of `riem_solver3_kernel`'s existing pattern | Good if `i` is the stride-1 dim (matches existing project memory layout, per roadmap doc's memory-layout finding) and `threadIdx.x` maps to `i` | Moderate — `km`-sized local arrays (`q4(4,km)`, `gam(km)`, `q(km+1)`) per thread, similar order to existing Riemann-solver kernels' local arrays | Not required | Trivially preserves the required `k`-order forward/backward sweep and floating-point summation order | Scales with `npz` at runtime (loop bound, not compile-time) — same approach the existing vertical kernels already use |
| B. One warp per vertical column | 32 threads cooperate on one column's `k`-loop | Same sequential dependency now has to be expressed as a warp-synchronous scan — extra complexity for no clear benefit at `km~10-91` (short columns) | Same | Lower per-thread register pressure, but warp-scan overhead likely dominates for short `km` | Possibly, for scan buffers | More complex to guarantee identical summation order | Same |
| C. One block per horizontal cell/column | One block (e.g. 32-128 threads) per column, threads split work across `k`/tracers | Over-provisioned for a single-column job at `km~10`; more block-launch overhead than benefit | N/A | Lower per-thread, higher scheduling overhead | Possible use for intra-block reduction (not needed — no cross-k reduction) | More complex | Same |
| D. Horizontal threads + sequential k loop | Same as A, phrased generically | Same as A | Same as A | Same as A | Same as A | Same as A | Same as A |
| E. Batch tracers across the warp (`mapn_tracer` specifically) | One thread per column, but iterate `nq` tracers together sharing the same search index `k0`/`l`/`m` (mirrors the Fortran structure exactly) | Same sequential `k`-loop per column; tracers share search state, so this is a natural extension of A, not a new candidate | Same | Higher — `q4(4,km,nq)` scales with `nq` (up to a few dozen tracers) | Possibly worthwhile for very large `nq` to avoid register spill | Same | Same |

**Selected initial design: Candidate A (one thread per horizontal
column, sequential loop over `k` inside the thread), extended per E for
the tracer batch case.** Rationale: (1) directly reuses the exact
per-column-thread pattern already implemented, compiled, GPU-executed,
and numerically validated for the vertical Riemann solver
(`riem_solver3_kernel`/`riem_solver_c_kernel`, Gate C2 closed) — no new
GPU-architecture invention required; (2) `km` (`npz`) is a runtime loop
bound already, not hard-coded, so this generalizes beyond the
currently-validated `npz=10` without extra work, consistent with the
existing vertical-kernel convention (CLAUDE.md's "do not hard-code an
architecture that only works for npz=10" is already satisfied by the
existing pattern this reuses); (3) columns are short (`km` order 10s,
not 100s), so warp/block-cooperative designs (B/C) add synchronization
overhead disproportionate to the small per-column work.

---

## STAGE 6 — Reusable Existing CUDA Infrastructure

From the roadmap doc's Part 1 architecture summary and this task's
targeted read, directly reusable without modification:

- **Persistent `Workspace` allocation** (`fv3_gpu_state.hpp:15`) —
  `fv_mapz`'s arrays (`pe/delp/pk/pkz/u/v/w/delz/pt/ua/va/omga/ws`) are
  already-existing fields in this workspace; only `q` and `dp1`/`te`
  scratch would need new persistent device buffers.
- **2 non-blocking streams** (`compute`/`transfer`) — new remap kernels
  slot into the existing `compute` stream; any new H2D/D2H for `q` uses
  the existing `transfer` stream and event-based ordering pattern
  already used elsewhere (no new synchronization primitive needed).
- **Tile geometry / indexing helpers** — the existing `(i,j)` column
  indexing convention (stride-1 `i`, matching Fortran column-major
  layout) used by `riem_solver3_kernel` applies unchanged to a
  `map_scalar`/`scalar_profile` kernel.
- **`fv3_gpu_dyncore_bridge.F90` ISO_C_BINDING bridge pattern** — the
  same lifecycle (init/step/finalize) and per-array registration
  convention used for `dyn_core`'s arrays extends naturally to
  `q`/`dp1` if/when a `fv3_gpu_interface_step_remap`-style entry point
  is added.
- **Dispatch-table / launch-and-error-wrapper mechanism** — reused
  as-is for any new kernel launches.
- **R4/R8 templates** — the existing precision-templated kernel pattern
  (used throughout WP-A/B/C/D) applies directly; no new precision
  abstraction needed.
- **Savepoint/validation infrastructure** (`fv3_savepoint_mod`,
  `fv3_sp_dump3d`, already used at the `dyn_core` exit boundary,
  `fv_dynamics.F90:809-830`) — directly reusable for Stage 8's Level-1/2
  validation without inventing a new dump mechanism, per the project's
  reuse-before-create rule.
- **Not reusable as-is**: no existing tridiagonal-solve *device
  function* is factored out for reuse — `riem_solver3_kernel`'s
  tridiagonal logic is specific to the Riemann-solver's own equations,
  not a generic shared helper. A new `scalar_profile`-equivalent device
  function must be written, but can follow the same per-thread-sequential-k
  structural pattern.

---

## STAGE 7 — Minimum Porting Unit

**Selected unit: `map_scalar` together with its `scalar_profile`
dependency** (the `kord>7` profile-reconstruction path), ported as one
coherent per-column CUDA kernel (Candidate A above).

Why this and not a smaller or larger unit:
- **Clean I/O boundary**: `map_scalar`'s inputs (`pe1, dpe1, pe2, dpe2,
  q1(one field, one j-row), kord, q_min, q_bot`) and output (`q2`) are
  self-contained per call — no dependency on `mapn_tracer`'s `nq`-batched
  loop or on `fillz`'s mass-conservation logic.
- **Meaningful measured cost**: `map_scalar` + `scalar_profile` are
  **63% of the ~4323 total fv_mapz-attributed Prompt-51 samples** (2724
  + 663 of 4323) — the single largest concentration of the measured
  18.67%-of-window cost, without pulling in the tracer-transport
  entanglement.
- **Does not require tracer-transport conversion at the same time**:
  `map_scalar` is called on `u,v,w,delz,te/pt,mfx,mfy,cx,cy` — none of
  which are tracers, so this unit is fully decoupled from `q`'s
  producer/consumer relationship with `tracer_2d` (Stage 12). The `q`
  H2D-transfer complication identified in Stage 4 is **avoided
  entirely** by choosing this unit over `mapn_tracer`.
- **Directly comparable against Fortran**: called once per non-tracer
  field per `j`-row with a small, fixed argument list — straightforward
  to build a standalone harness feeding it identical `pe1/dpe1/pe2/dpe2/q1`
  arrays captured from a real run (via the existing savepoint
  mechanism, Stage 6) and comparing `q2` bit-for-bit or at round-off.

`mapn_tracer` (same algorithm, `nq`-tracer-batched, entangled with
`fillz` and the `q`-transfer complication) is **explicitly deferred** to
a follow-on port that reuses the same validated kernel pattern, not
duplicated effort — consistent with the "reuse existing project
mechanism" rule, applied here prospectively (reuse the *pattern*
established by porting `map_scalar` first).

---

## STAGE 8 — Validation Design (before any implementation)

- **Level 1 — standalone routine comparison.** Build a harness (not
  implemented in this task, per scope) that: (a) captures real
  `pe1, dpe1, pe2, dpe2, q1, kord, q_min` values for one or more
  `map_scalar` call sites (e.g. via the existing savepoint mechanism —
  Stage 6 — or a targeted debug dump added at the 9 call sites in
  `Lagrangian_to_Eulerian`); (b) runs the authoritative Fortran
  `map_scalar`/`scalar_profile` on those inputs; (c) runs the CUDA
  kernel on the same inputs; (d) compares `q2` element-wise.
  Classification target: **BITWISE or ROUND-OFF** — the algorithm's only
  non-reorderable step (Stage 3) is the tridiagonal sweep and the
  fractional-area summation order, both of which a faithful per-thread
  port preserves exactly; this is the same standard already met by the
  Riemann-solver kernels (Gate C2, `w` max_rel ~5e-11 at R8).
- **Level 2 — first integrated timestep.** Swap only the CUDA `map_scalar`
  kernel into the *existing* CUDA-backend `n_map` iteration (still
  calling Fortran `Lagrangian_to_Eulerian` for everything else — i.e.
  intercept only the 9 `map_scalar` call sites, either via a
  compile-time/runtime backend switch inside `fv_mapz.F90` itself or an
  ISO_C_BINDING call-out), compare the full `Lagrangian_to_Eulerian`
  exit state (`pk, pkz, q, delp, ps, pt`, etc.) against the existing
  CUDA+Fortran-companion baseline for one C512 `DYN_CORE` call.
  Classification target: **ROUND-OFF** (same standard as Gate C2).
- **Level 3 — multi-step C512 validation.** Short Held-Suarez run (same
  methodology as the existing HS validation table in CURRENT_STATE.md
  §4 — bitwise/round-off T/U/V/PS comparison at fixed checkpoints),
  comparing T/U/V/PS/`delp` and the directly-modified remap arrays
  (`pk`, `pkz`, `q`) against the existing all-Fortran-companion CUDA
  baseline. Classification target: **ROUND-OFF**, same tolerance regime
  already established for Gate C2 (worst `w` max_rel ~5e-11 at R8,
  ~2.4e-1 at R4-scaled-threshold-but-≤5.9e-5-absolute).

**Performance is not used as evidence of correctness at any level** —
each level above is a pure numerical comparison; timing is measured
separately (Stage 10 implementation phases, 52F).

---

## STAGE 9 — Numerical Tolerance Policy

- **Bitwise equivalence is plausible but not guaranteed** for the
  tridiagonal-sweep phase (fixed operation order, same as Fortran, if
  the CUDA kernel performs the identical forward/backward sweep in the
  identical order with the identical `real`/`real(kind=8)` type usage —
  the Fortran source uses default `real` (R4/R8-templated per project
  convention) throughout `scalar_profile`/`map_scalar`, no `kind=8`
  promotion visible in the routines read).
- **The search-and-integrate phase's floating-point summation order is
  the main bitwise risk**: a GPU port that reorders the "fractional
  area" summation (Stage 3, :1639-1661) — e.g. to vectorize across `l`
  instead of accumulating sequentially — would only be round-off
  equivalent, not bitwise. **Recommendation: preserve the exact
  sequential-accumulation order in the initial CUDA kernel** (matches
  Candidate A's per-thread-sequential design naturally; no reordering
  is actually required by the parallelization design chosen in Stage
  5), which maximizes the chance of bitwise or near-bitwise agreement
  and avoids introducing an avoidable round-off source.
- **Tolerance if not bitwise**: fall back to the project's established
  R8 methodology exactly as used for Gate C2 — round-off-level relative
  tolerance (~1e-10 to 1e-11 range demonstrated for the vertical solver
  kernels), documented per-field, not loosened merely to make CUDA
  pass, per CLAUDE.md's explicit rule. **R8 first; R4 validation follows
  only after R8 passes**, matching the project's stated precision
  policy.

---

## STAGE 10 — Implementation Phase Plan

Proposed staged breakdown (**not implemented in this task** — this
Stage only determines whether the breakdown is appropriate):

- **52A — Standalone baseline harness.** Capture real
  `pe1/dpe1/pe2/dpe2/q1/kord/q_min` inputs (one or more of the 9
  `map_scalar` call sites) from a live C512 HS run via the existing
  savepoint mechanism; build a minimal standalone Fortran driver that
  calls `map_scalar`/`scalar_profile` on the captured inputs and dumps
  `q2`. No CUDA yet.
- **52B — CUDA kernel + standalone validation.** Implement the
  per-column CUDA kernel (Candidate A) mirroring `map_scalar` +
  `scalar_profile` exactly (preserving the Stage 9 summation order);
  validate against 52A's Fortran output at Level 1 (Stage 8).
- **52C — Persistent-runtime integration.** Register `q`/`dp1` (if not
  already present) in the `Workspace`; add the ISO_C_BINDING entry
  point per the existing bridge pattern (Stage 6); wire the 9 call
  sites in `Lagrangian_to_Eulerian` to call the CUDA path when the CUDA
  backend is active (keeping the Fortran path unconditionally available,
  per CLAUDE.md's backend-selection rule).
- **52D — First-step integrated comparison.** Level 2 validation (Stage
  8) — one C512 `DYN_CORE` call, CUDA `map_scalar` + Fortran everything
  else, vs. existing all-Fortran-companion CUDA baseline.
- **52E — Multi-step C512 validation.** Level 3 validation (Stage 8) —
  short Held-Suarez run, same checkpoint methodology as the existing HS
  table.
- **52F — Same-node performance A/B.** Measure `DYN_CORE` ms/call with
  CUDA `map_scalar` active vs. the current all-Fortran-companion
  baseline, same-node/same-allocation paired methodology (mandatory per
  the project's "node variance" lesson, CURRENT_STATE.md §8) — this is
  where the Stage 11 speedup estimate gets its first real data point.

**This breakdown is appropriate and recommended as-is**: each phase has
a single, testable deliverable, an explicit correctness gate before the
next phase, and matches the project's existing validation-hierarchy
discipline (IMPLEMENTED → COMPILED → GPU EXECUTED → NUMERICALLY
VALIDATED, applied per-phase rather than promised up front).

---

## STAGE 11 — Performance Potential (finite-speedup scenarios)

Using `S = 1/((1-f)+f/s)`, `f = 0.1867` (measured `fv_mapz_mod` fraction
of the whole `DYN_CORE` window):

| Component speedup `s` | Whole-`DYN_CORE` speedup `S` | Note |
|---|---|---|
| 2x | 1.1030 | `fv_mapz` twice as fast |
| 5x | 1.1756 | |
| 10x | 1.2020 | |
| ∞ (Stage 0 ceiling) | 1.2296 | Absolute theoretical maximum |

**None of these scenarios reach the ≥2x (≤3187 ms/call) project target
on their own.** `fv_mapz_mod` alone — even eliminated entirely — closes
only a small fraction of the gap between the current comparable-to-parity
baseline (6373.86 ms/call) and the 2x target. This is reported honestly
per the task's explicit instruction not to promise a speedup: **porting
`fv_mapz` is a necessary, well-evidenced *step* in closing the CUDA
coverage gap, not a sufficient one.** The remaining ~46% of the window
attributed to other unported FV3 Fortran CPU work (`fv_tracer2d_mod`,
state conversion, Held-Suarez/transcendental math per Prompt 51) and the
14.4% halo-sync bucket (Prompt 49/50, shown not safely reducible without
a redesign) remain the larger remaining opportunities; this prompt
addresses the first of several coverage gaps, chosen because it is the
single largest *individually* attributable and *source-mapped* CPU
component identified so far.

---

## STAGE 12 — Hidden Coupling Check

- **`fv_mapz_mod` vs. `fv_tracer2d_mod`**: **MOSTLY INDEPENDENT.** The
  only shared array is `q` (tracers): `tracer_2d`/`tracer_2d_1L`/
  `tracer_2d_nested` write `q` (host Fortran transport using accumulated
  mass fluxes `mfxL/mfyL/cxL/cyL` and Courant numbers `cxL/cyL`, none of
  which `fv_mapz` reads directly — `fv_mapz`'s `te` argument reuses the
  `dp1` buffer as unrelated output scratch, not the flux capacitors).
  `Lagrangian_to_Eulerian`'s `mapn_tracer` call then vertically remaps
  the already-transported `q`. This is a genuine, real **sequential
  producer/consumer ordering requirement** (transport must happen before
  vertical remap in a given `n_map` iteration) but **not** a tight
  algorithmic coupling — no interleaved loop bodies, no shared local
  state beyond the one array. **The chosen minimum porting unit
  (`map_scalar`, Stage 7) sidesteps this coupling entirely** by covering
  only non-tracer fields; the coupling becomes relevant only for a
  future `mapn_tracer` port, at which point the smallest safe boundary
  is: transport `q` on the host (unchanged), then H2D just `q` before
  the (future) CUDA `mapn_tracer` kernel, then D2H (or keep resident, if
  `tracer_2d` has by then also been ported) — no expansion of scope is
  needed for the current unit.
- **`fv_mapz_mod` internal coupling to physics/microphysics**
  (`fv_sat_adj`, `moist_cv`/`moist_cp`, `mpp_global_sum` energy
  correction): these sit in `Lagrangian_to_Eulerian`'s body but are
  **gated off the chosen minimum unit's code path** — `map_scalar` does
  not call any of them. For the currently-validated Held-Suarez
  (dry-core) configuration, `nwat==0` is expected to select the dry
  branch at `fv_dynamics.F90:434` (not independently re-confirmed from
  the live AGCM.rc in this pass — a named gap, not a claim), which
  further simplifies `moist_cv`/`moist_cp` to their trivial dry forms
  and likely disables `fv_sat_adj` (`do_sat_adj .and. nwat>=6` gate at
  `fv_mapz.F90:346` — false when `nwat==0`). **No expansion of scope
  required**; these remain outside the chosen unit's boundary regardless.

---

## STAGE 13 — LOC / Effort Estimate (measured, not guessed)

| Metric | Value | Source |
|---|---|---|
| `fv_mapz.F90` total LOC | 3949 | `wc -l` |
| Routines in module | 19 (`Lagrangian_to_Eulerian`, `compute_total_energy`, `pkez`, `remap_z`, `map_scalar`, `mapn_tracer`, `remap_2d`, `scalar_profile`, `cs_profile`, `cs_limiters`, `ppm_profile`, `ppm_limiters`, `steepz`, `rst_remap`, `mappm`, `moist_cv`, `moist_cp`, `map1_gmao`) | `grep -n "^\s*subroutine"` |
| **Chosen unit LOC** (`map_scalar` + `scalar_profile`) | 153 + 408 = **561 LOC** | direct line-range measurement |
| Major arrays in chosen unit | 9 (`pe1, pe2, dpe1, dpe2, q1, q2, q_bot` + `scalar_profile`'s internal `gam, q, d4, extm/ext5/ext6`) | signature + local-declaration read |
| Direct dependencies of chosen unit | 1 (`scalar_profile`, or `ppm_profile` on the `kord<=7` branch — **both** may need porting depending on which `kord` values the validated production config actually uses; not confirmed in this pass, Stage 3's named gap) | call-site read |
| Helper routines also required | `ppm_profile` (255 LOC) if the `kord<=7` branch is live in production — **unconfirmed**, adds up to 255 LOC to the effort if so | named gap |

**Effort classification: SMALL to MEDIUM.** 561 LOC (possibly +255 if
`ppm_profile` is also required) is an order of magnitude smaller than
the full module (3949 LOC) or even just `Lagrangian_to_Eulerian`'s own
body (1110 LOC). The algorithm itself (tridiagonal sweep + monotonic
search-integrate) is well-understood from source (Stage 3) and has a
direct structural precedent already validated on GPU (the Riemann
solver kernels), which lowers implementation risk relative to raw LOC
count. The classification would move to MEDIUM-LARGE if the `kord<=7`
(`ppm_profile`) branch must also be ported for full production coverage
— resolving which `kord` branch is live is a fast, cheap first step of
Phase 52A (reading the production `AGCM.rc`/flag defaults), not a
blocker to starting.

---

## STAGE 14 — Decision

**B. READY FOR HARNESS FIRST.**

The algorithm is bounded, source-mapped, and structurally well
understood (Stage 3), the minimum porting unit is clearly scoped and
decoupled from tracer transport (Stage 7/12), and a direct GPU-design
precedent already exists and is validated in this codebase (Stage 5/6).
However, per this task's own "do not implement broadly" constraint and
Stage 8's "design validation before implementation" requirement, the
next concrete step is **52A**: build the standalone Fortran baseline
harness (capture real `map_scalar`/`scalar_profile` inputs via the
existing savepoint mechanism, run them through a minimal standalone
driver) — this must exist before any CUDA kernel is written, so that
52B has ground truth to validate against from the first line of CUDA
code. This also resolves the two named gaps (Stage 3/13's `kord` branch
question) as a cheap first sub-step, before committing to porting
`ppm_profile` as well as `scalar_profile`.

No Opus escalation: the vertical-remap dependency/ordering semantics
were fully resolved by direct source inspection (Stage 3) — the
tridiagonal-sweep and monotonic-search structure is unambiguous, and its
GPU mapping has a direct, already-validated precedent in this same
codebase (the Riemann-solver kernels).

---

## Final Report

- **MEASURED MOTIVATION**: `fv_mapz_mod` = 18.67% of the C512 `DYN_CORE`
  window (Prompt 51); Amdahl ceiling if eliminated entirely = **1.23x**
  whole-`DYN_CORE` speedup (Stage 0) — an upper bound, not a promise.
- **HOT ROUTINES**: `map_scalar` (2724 samples, 153 LOC) +
  `scalar_profile` (663 samples, 408 LOC) = 63% of resolved fv_mapz
  samples; `mapn_tracer` (413 samples, 112 LOC) and
  `Lagrangian_to_Eulerian`'s own frame (523 samples, pointwise `pt`→T
  conversion + orchestration) round out the profile (Stage 1).
- **CALL GRAPH**: `fv_mapz` runs exactly once per CUDA `dyn_core` call
  (1:1 at the validated `k_split=1` config), strictly *after* the CUDA
  acoustic-substep loop and *after* Fortran tracer transport, on arrays
  already host-resident from `dyn_core`'s own existing exit D2H (Stage 2).
- **DATA RESIDENCY**: no new transfer is needed for the chosen
  non-tracer unit beyond what a `map_scalar`-only port itself requires;
  `q` (tracers) would need a genuinely new H2D if `mapn_tracer` were
  ported later, since CUDA `dyn_core` never touches `q` today (Stage 4).
- **ALGORITHM**: PPM vertical remap — a real, order-preserving
  tridiagonal sweep (profile reconstruction) followed by a monotonic
  search-and-integrate onto the new grid; column-independent
  horizontally, sequential in `k`, directly analogous to the
  already-validated Riemann-solver kernel structure (Stage 3).
- **GPU DESIGN**: Candidate A — one thread per horizontal column,
  sequential `k` loop inside, reusing the existing vertical-kernel
  pattern; `npz`-portable by construction (Stage 5).
- **PORTING UNIT**: `map_scalar` + `scalar_profile` (561 LOC), decoupled
  from tracer transport, 63% of measured fv_mapz cost (Stage 7).
- **VALIDATION**: 3-level plan (standalone routine, first-integrated-step,
  multi-step C512), target BITWISE-or-ROUND-OFF at Level 1, ROUND-OFF at
  Levels 2-3, matching Gate C2's existing tolerance regime (Stage 8/9).
- **PERFORMANCE POTENTIAL**: even a 10x-faster `fv_mapz` yields only
  ~1.20x whole-`DYN_CORE` speedup; the ≥2x target requires this plus
  additional coverage-gap work not in this task's scope (Stage 11).
- **DECISION**: B — READY FOR HARNESS FIRST (Stage 14).
- **NEXT PROMPT**: 52A — build the standalone Fortran baseline harness
  for `map_scalar`/`scalar_profile` (savepoint-captured real inputs),
  and confirm the production `kord_mt`/`kord_wz`/`kord_tm` values to
  resolve whether `ppm_profile` must also be ported.
- **PROJECT STATUS**: numerical source changed: NO. Held-Suarez physics
  changed: NO. MPI topology changed: NO. Jobs submitted: 0 (no GPU
  allocations used). `ncu`/`nsys` used: NO (static analysis only, reused
  Prompt 49/51 measurements). Opus escalation: NO.

FV_MAPZ CUDA PORT MAPPED — READY FOR 52A (STANDALONE map_scalar/scalar_profile BASELINE HARNESS)
