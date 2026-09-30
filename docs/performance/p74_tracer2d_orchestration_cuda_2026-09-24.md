# Prompt 74 — TRACER_2D Measurement, GO/NO-GO, and hord_tr=8 Blocker

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet
Prior art: `p73_tracer2d_cuda_port_plan_2026-09-24.md` (mapping/harness only).

## 0. Summary

Measurement-first task: instrument `tracer_2d_1L` with diagnostic-only
sub-timers, measure a real C512 production-density breakdown, apply the
Stage-3 GO/NO-GO gate, and only implement CUDA if warranted. **Result: NO-GO.
No CUDA implementation performed.**

Two independent findings drove the decision:

1. **hord_tr production value is 8, not 6 or 12** — confirmed directly from
   a production job log (`FV3 hord_tr:   8`), not from any `input.nml`
   override (none exist across 159 checked experiment configs; both the
   Fortran default (`fv_arrays.F90:326`, `hord_tr = 12`) and this value are
   superseded by a runtime override whose exact source line has not yet been
   located). The existing WP-B GPU scalar-transport service
   (`launch_scalar_transport_cuda`, `wp_b_cuda.cuh`) **only supports
   `abs(hord) == 6`** ("only the order-6 PPM path is ported" — explicit
   in-source documentation) and returns
   `FV3_GPU_UNSUPPORTED_CONFIGURATION` otherwise. **This invalidates Prompt
   73's premise that "fv_tp_2d is already GPU-executed" for the actual
   production tracer path** — it is GPU-executed only for `hord==6` callers
   (e.g. D-grid transport), not for tracer transport's `hord_tr==8`.
2. **Host orchestration cost is in the engineering-judgment zone, not a
   clear GO**: 39.9 ms/call, 14.2% of `TRACER_2D` — just under both GO
   thresholds (40 ms / 15%).

Given finding 1, finding 2 is moot: this prompt's mandated architecture
(GPU orchestration wrapping the *existing* GPU `fv_tp_2d`) is not buildable
without either (a) re-porting `fv_tp_2d` for `hord==8` — explicitly
prohibited this prompt — or (b) wrapping GPU orchestration around a
Fortran-only `fv_tp_2d`, which requires the same CPU/GPU hybrid-split
pattern **already rejected in Prompt 65** for PT/DELZ, and which Stage 5's
own stop condition ("if transfer cost likely exceeds removable work: STOP")
argues against for a 14%-of-timer slice.

**Result: TRACER_2D ORCHESTRATION PORT NOT WARRANTED — PROFILE SHOWS
fv_tp_2d (hord=8, CPU-only, 69.7% of TRACER_2D) AS THE DOMINANT COMPONENT,
OUT OF SCOPE THIS PROMPT.**

## 1. Stage 1 — Instrumentation (diagnostic-only, numerical semantics: NO change)

File: `fv_tracer2d.F90`, subroutine `tracer_2d_1L` only (the subroutine
production actually dispatches to, per Prompt 73's proof). New env-gated
timers (idiom matches the existing `RUN_CUDA_FV3_PTDELZ_*`/
`RUN_CUDA_FV3_REMAPUV_*` precedent in `fv_mapz.F90`), all gated by new env
var `RUN_CUDA_FV3_TRACER2D_DIAG` (unset by default, zero behavior change):

| Timer | Component | Wraps |
|---|---|---|
| `P74_TR_HALO_INIT` | A — initial q halo | `complete_group_halo_update(q_pack, domain)` |
| `P74_TR_SETUP` | B — Courant/flux setup | xfx/yfx OMP loop + qmax/cmax loop + mfx2/cx2 nsplt-scaling loop |
| `COMM_TRACER_MAX` (pre-existing) | B cont'd — allreduce | `mp_reduce_max(qmax,...)` |
| `P74_TR_FVTP2D` | C — fv_tp_2d calls | both `nsplt/=1` and `nsplt==1` call sites |
| `P74_TR_ORCH` | D — host orchestration | ra_x/ra_y, dp2 computation, both conservative q-update blocks |
| `P74_TR_MIDSPLIT` | E — nsplt>1 mid-split | the `if (it<nsplt)` block (dp1 copy + `mpp_update_domains`) |

Plus a per-`(call,rank,k)` diagnostic dump `[P74_NSPLT]` in a new serial loop
(recomputes `int(1.+cmax(k))`, read-only, no state mutated).

One compile fix required: `p74_diag_active` had to be added to the
`shared(...)` clause of the pre-existing `!$OMP parallel do default(none)`
pragma wrapping the `do iq=1,nq` loop (ifort `#6752` — `default(none)`
requires every referenced variable explicitly classified). No other changes.

Fortran fixture rebuilt clean via `p17_rebuild_exe.sh`. New SHA256:
`12cfa63571567e35746b1c7200b1cf40e4a95bbe24fb6acb5896cf622e37c93b`. All six
new timer/env-var strings confirmed present via `strings` on the rebuilt
binary.

## 2. Stage 2 — C512 measurement (job 58539236, node `warpa007`)

R8, npz=10, `RUN_DT=36` (`k_split=1`, `n_split=3`), 6 MPI ranks/1 A100, MPS
ON, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`,
Held-Suarez ON, `COLDSTART=1`, `RUN_CUDA_FV3_TRACER2D_DIAG=1`,
`N_TRACERS=2` in the job template (GEOS's own `MAX(1,ncnst)` floor still
yields `nq=1` reaching `fv_dynamics`, per Prompt 73's proof — reconfirmed
below). Reused the `p71_stage11_cuda` experiment as template
(`p74_tracer2d_diag` dir/job). 3 dyn_core calls executed (confirmed via the
`[P74_NSPLT]` dump: exactly `call=1,2,3`). Elapsed 68s, rc=0.

Provenance: SHA256 preflight PASS (matches §1); job log confirms
`***** USING **** .../p74_tracer2d_diag/scratch_.../StandAlone_FV3_Dycore.x`
(the intended binary, not a stale copy — Prompt 53 lesson applied); 0
`range_warn`/NaN/abort.

**hord_tr reconfirmed**: `FV3 hord_tr:   8` (log line 251/332).

### Timer table (raw, 3-call accumulated seconds, `fv_timing_mod` printout)

| Block | Seconds (3 calls) | ms/call |
|---|---|---|
| TRACER_2D | 0.8426 | 280.87 |
| P74_TR_HALO_INIT | 0.0982 | 32.73 |
| COMM_TRACER (≈ HALO_INIT, cross-check) | 0.1001 | 33.37 |
| P74_TR_SETUP | 0.1272 | 42.40 |
| COMM_TRACER_MAX | 0.0151 | 5.03 |
| P74_TR_FVTP2D | 0.5876 | 195.87 |
| P74_TR_ORCH | 0.1198 | 39.93 |
| P74_TR_MIDSPLIT | (not printed — never entered) | 0.00 |
| FV_DYN_LOOP (context) | 7.0434 | 2347.80 |
| REMAP_UV (context, confirms Fortran-default per Prompt 72A) | 0.5101 | 170.03 |

`COMM_TRACER` is confirmed via `git grep`-equivalent source search to be
used **only** inside `tracer_2d_1L`'s halo call in the production path (the
other `COMM_TRACER`/`COMM_TRACER_MAX` call sites in this file belong to the
unused `tracer_2d`/`tracer_2d_nested` subroutines) — a clean cross-check
against the new `P74_TR_HALO_INIT` timer (32.73 vs. 33.37 ms/call, small
diff from timer-call overhead only).

### Component breakdown (% of TRACER_2D = 280.87 ms/call)

| Component | ms/call | % of TRACER_2D |
|---|---|---|
| A — Initial halo | 32.73 | 11.66% |
| B — Courant/flux setup + allreduce | 47.43 | 16.89% |
| C — fv_tp_2d | 195.87 | **69.73%** |
| D — Host orchestration | 39.93 | 14.22% |
| E — nsplt>1 mid-split | 0.00 | 0.00% |
| **Sum of A–E** | 315.97 | 112.50% |
| **"Unaccounted"** | −35.10 | **−12.50%** |

**Note on the negative "unaccounted" figure**: `fv_timing_mod`'s
`timing_prt` independently `mp_reduce_max`s each named block across the 6
MPI ranks (`fv_timing.F90:255-262`). Different blocks' worst-case rank need
not coincide (MPI/tile load imbalance), so per-block cross-rank-maxed sums
can legitimately exceed the maxed outer total — a known property of this
project's existing timer methodology (identical convention used for every
prior `DYN_CORE`/`REMAPPING`/`REMAP_UV` figure in Prompts 69/71), not a new
instrumentation defect. The ~12.5% overshoot is consistent with ordinary
per-rank skew at this scale and does not change any conclusion below.

## 3. NSPLT distribution

180/180 dumped `(call, rank, k)` triples (3 calls x 6 ranks x 10 k-levels)
report **`nsplt=1`, 100%**. Zero occurrences of `nsplt=2` or `nsplt>2`.
Directly confirms, from live production execution (not just static source
reading), that the `it<nsplt` mid-split branch is never taken at this
resolution/timestep — consistent with the Prompt-73 source-level proof that
`1<1` is always false when `nsplt==1`.

| nsplt value | Count | % |
|---|---|---|
| 1 | 180 | 100.0% |
| 2 | 0 | 0.0% |
| >2 | 0 | 0.0% |

## 4. GO/NO-GO gate (Stage 3)

Host orchestration (D) = 39.93 ms/call, 14.22% of TRACER_2D — **both figures
fall just under the GO thresholds (40 ms / 15%) and just over the NO-GO
thresholds (20 ms / 8%)**: the explicit "engineering judgment" zone.

Judgment, resolved **NO-GO**, for a reason independent of D's exact value:
**finding 1 (§0)**. The mandated Stage 7 architecture is "GPU
orchestration-pre → existing GPU fv_tp_2d → GPU conservative update, with no
unnecessary host round-trip between them." For the actual production
`hord_tr=8`, there is no "existing GPU fv_tp_2d" to wrap — the validated
kernel supports only `hord==6`. Porting orchestration alone would force one
of:

- **Re-port `fv_tp_2d` for `hord==8`** — explicitly prohibited this prompt
  ("do NOT re-port fv_tp_2d"), and a materially larger, differently-shaped
  task (different PPM/limiter formula per `hord` value in `tp_core.F90`).
- **Wrap new GPU orchestration around a Fortran-only `fv_tp_2d`** — the
  same CPU/GPU hybrid-split pattern **Prompt 65 already rejected** for
  PT/DELZ. This would add a D2H/H2D round-trip to reach the still-CPU
  `fv_tp_2d` (69.7% of the timer) in exchange for keeping only the 14.2%
  orchestration slice on GPU — Stage 5's own stop condition ("if transfer
  cost likely exceeds removable work: STOP") applies squarely here.

**Decision: NO-GO. No CUDA implementation performed (Stages 4-15 not
executed).**

## 5. Open item (not resolved this prompt)

The exact GEOS-side source line overriding `hord_tr` from the Fortran
default (12) to the production runtime value (8) has not been located (not
in any of the 159 checked `input.nml` files). This does not block this
prompt's GO/NO-GO conclusion (the conclusion depends only on the *value*
being 8, confirmed live from the job log, not on *why*), but should be
resolved before any future prompt considers extending
`launch_scalar_transport_cuda` to support `hord==8` — that extension (a
WP-B-owned scope item, larger than this prompt's orchestration-only slice)
is the actual prerequisite for any tracer_2d GPU work being worthwhile.

**Update (Prompt 75)**: the hord==8 mapping/harness prerequisite named
above is now done. Prompt 75 mapped hord==8 against the existing
hord==6 CUDA implementation (orchestration fully reusable; face-flux
kernels + a new `pert_ppm` helper need new, bounded logic — no
architectural mismatch), confirmed the existing `ng=3` production halo
already covers hord==8's stencil reach, and built an 11-fixture
host-only harness (0 NaN/Inf, bitwise-repeatable, conservation-identity
PASS). Decision: **PROCEED TO CUDA HORD8 EXTENSION** (classified
MODERATE NEW NUMERICAL BRANCH, effort MEDIUM). The exact `hord_tr=8`
override source line is still not located, but is confirmed
non-blocking for the extension itself. See
`p75_fv_tp_2d_hord8_cuda_plan_2026-09-24.md`.

## 6. Documentation state

- This file: full Prompt 74 record.
- `p73_tracer2d_cuda_port_plan_2026-09-24.md`: needs a correction note that
  its "fv_tp_2d is already GPU-executed" premise does not hold for the
  actual production `hord_tr=8` (only true for `hord==6` callers elsewhere).
- `cuda_fv3_optimization_roadmap_2026-09-21.md`: updated with the measured
  breakdown and NO-GO/blocked status.
- `CURRENT_STATE.md`: updated with the breakdown table, nsplt distribution,
  GO/NO-GO, and the hord_tr=8 finding as the new next-target driver.

---

**TRACER_2D ORCHESTRATION PORT NOT WARRANTED — PROFILE SHOWS fv_tp_2d
(hord_tr=8, CPU-only, 69.7% of TRACER_2D) AS THE DOMINANT, OUT-OF-SCOPE
COMPONENT**
