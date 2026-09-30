# Prompt 77: `fv_tp_2d` hord==8 CUDA — Production `tracer_2d_1L` Integration

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Prerequisite: Prompt 76 (`p76_fv_tp_2d_hord8_cuda_implementation_2026-09-24.md`)
— hord==8 CUDA kernel implemented and GPU-validated at the unit level, but
integration into production `tracer_2d_1L` was **BLOCKED BY DESIGN**: no
Fortran-side dispatch selector routed the real call through any CUDA path.

## 1. Goal

Add the smallest production dispatch so `tracer_2d_1L` calls the existing,
validated CUDA `fv_tp_2d` hord==8 implementation for the validated
production configuration (`hord_tr==8`, `nsplt==1`, R8), preserving the
authoritative Fortran fallback, existing halo/Courant/allreduce semantics,
existing tracer orchestration, and current numerical ordering. Then measure
C32/C512 correctness and paired performance.

## 2. Change Budget (respected, not exceeded)

Two substantial changes were authorized and consumed:
1. Production `fv_tp_2d` dispatch bridge (CUDA-side entry point, from a
   prior session).
2. Narrow Fortran dispatch selector: `use_cuda_tp2d` in `fv_tracer2d.F90`.

Diagnostic/test infrastructure added this task (not counted against the
budget): the `[P77_EXIT_DUMP]` exit-dump print (gated by
`RUN_CUDA_FV3_TRACER_DUMP_EXIT`) and the `RUN_CUDA_FV3_TRACER_FORCE_FORTRAN`
control flag.

Explicit Do-NOTs honored throughout: hord==8 CUDA numerics unchanged;
`pert_ppm` unchanged; hord==6 path unchanged; PT/DELZ and REMAP_UV
unchanged; `nsplt>1` not implemented; halo exchange/`mp_reduce_max` not
redesigned; R4 excluded at the dispatch level only; no host-orchestration
porting beyond invoking the existing validated CUDA `fv_tp_2d` path.
**New this task: `fv_restart.F90` was not modified, and no synthetic
tracer seed was introduced**, per explicit user direction (§4).

## 3. Dispatch Selector

`fv_tracer2d.F90`:

```fortran
use_cuda_tp2d = (hord == 8) .and. (lim_fac == 1.0) .and. (kind(q) == 8) &
                .and. <existing RUN_CUDA_FV3-style gating>
if (use_cuda_tp2d .and. env_flag_enabled('RUN_CUDA_FV3_TRACER_FORCE_FORTRAN')) then
  use_cuda_tp2d = .false.
end if
...
if (use_cuda_tp2d) then
  <call CUDA fv_tp_2d hord==8 path>
else
  <call authoritative Fortran fv_tp_2d>
end if
```

Source-grep confirmed (this task) that the selector's condition depends
only on `hord`/`lim_fac`/`kind(q)`/environment — never on `q`'s runtime
values. This matters for §6 below.

## 4. Mid-Task Finding: Structural Cold-Start Tracer-Zero Degeneracy

While preparing the Stage 7 C32 boundary comparison, direct inspection of
`fv_restart.F90` (lines 185-374) established that the entire standalone
cold-start harness family — not just the Held-Suarez track — leaves
`Atm(n)%q` at its Fortran allocation-time default (effectively zero) for
the whole run:

- `cold_start_grids(:) = cold_start` (line 226).
- `fv_io_read_restart` (line 320) is gated on
  `.not.cold_start_grids(n) .and. .not.external_ic`.
- `get_external_ic` (line 353) is gated on `Atm(n)%flagstruct%external_ic`.
- With `cold_start=.true.` and `external_ic=.false.` — confirmed the
  default in every checked `AGCM.rc`, including non-HS Gate-C2-era
  harnesses (`p23_stageD_r4_c32_cuda`, `p24_c32_r8_cuda`,
  `p28_c32_r8_cuda_guard`, `p44_c512_regress/run_hsoff`) — neither branch
  executes.
- `ExtData.rc` is empty (no moisture/tracer import) in every checked case.

This is a genuine, previously-undocumented gap in the standalone harness
family's cold-start IC path, not a Held-Suarez-specific curiosity, and it
also implicates the original Gate C2 validation harnesses.

**User decision (binding, verbatim intent preserved):** do not modify
`fv_restart.F90`; do not introduce a synthetic tracer seed in this task.
Instead, narrow the Stage 7/8/9 validation claim to production
dispatch/integration wiring only, and rely on Prompt 76's non-degenerate
fixture-based GPU validation as the numerical-correctness evidence for the
kernel itself. Record the limitation for future tracer-validation work.

**Resulting evidentiary split, stated explicitly:**

> Kernel numerical correctness is established by Prompt 76's non-degenerate
> GPU fixtures. Prompt 77's cold-start production harness validates
> integration wiring only, because its tracer state is structurally zero.

## 5. Stage 7/8 — C32 Dispatch-Wiring Validation (PASS, narrowed claim)

Same-node A/B, job 58543176 (`p77_stage7_c32.sbatch`), sha256 preflight
matched (`8152aed3...`). Arm A: production CUDA dispatch. Arm B:
`RUN_CUDA_FV3_TRACER_FORCE_FORTRAN=1` control.

Evidence against the user's 10-point checklist:

1. Fortran-control arm selects authoritative `fv_tp_2d`: confirmed
   (`use_cuda_tp2d=F` on every dump line, arm B).
2. CUDA arm selects the hord==8 CUDA path: confirmed
   (`use_cuda_tp2d=T` on every dump line, arm A).
3. Correct CUDA dispatch count: 36 `use_cuda_tp2d=T` headers (6 calls x 6
   PEs), arm A.
4. Expected Fortran fallback count: `P77_fortran_DUMP_LINES=108`
   (36 headers x 3 lines/header), arm B.
5. Same call count/order between arms: both arms 6 calls x 6 PEs;
   `DUMP_LINES=108` both arms.
6. `q` bitwise identical for the degenerate all-zero input: confirmed —
   every `q_iq1`/`q_iqN` value `0.0000000000000000E+00`, both arms.
7. No NaN/Inf: confirmed (grep clean).
8. No crash/MPI abort/CUDA error: `Status=0`, `Range_Warn=0`, `NaN=0`,
   `Abort=0`, both arms.
9. No unintended change in dynamically active fields: not applicable at
   this narrowed scope (tracer-only dispatch change; no other field
   touched by this selector).
10. State ownership/transfer behavior correct: CUDA arm's tracer state
    stays device-resident through the call per the existing bridge; no
    full-state host/device transfer introduced by this selector.

`ALL DONE` marker reached. **Stage 7/8: PASS** under the narrowed claim.
Stage 8 (implicitly covered by the same 6-call C32 run structure, per the
Prompt 71 precedent) is closed by the same evidence.

## 6. Pre-Stage-11 Zero-Tracer Early-Exit Check (required by user directive)

Grepped `fv_tracer2d.F90`'s selector logic (§3) and the CUDA kernel/
dispatch sources (`wp_b_cuda.cu`, `wp_b_cuda.cuh`,
`fv3_gpu_production_dispatch.cu`) for any all-zero/early-exit/skip/
is-constant pattern. **None found.** The selector depends only on
`hord`/`lim_fac`/`kind(q)`/environment; the CUDA kernel path contains no
data-dependent fast path for zero input. Per the user's explicit gate:
no such early exit exists, therefore performance timing proceeds and is
treated as representative of dispatch/kernel-launch/control-flow cost
(though not of non-degenerate transport-content cost, which Prompt 76's
fixtures already cover).

## 7. Stage 9 — C512 Dispatch-Wiring Validation (PASS, narrowed claim)

Same-node A/B, job 58543590 (`p77_stage9_c512_ab.sbatch`), node `warpa005`,
sha256 preflight matched. MPS lifecycle managed cleanly
(`fv3_mps_start`/`fv3_mps_stop`). Both arms' `fv3.j` retained
`RUN_CUDA_FV3: 1` (PT/DELZ CUDA stays on in both arms); only the tracer
dispatch differs, via the wrapper-script env var
(`RUN_CUDA_FV3_TRACER_FORCE_FORTRAN` for the Fortran-control arm).
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` set
(matching CURRENT_STATE.md's best-validated config); `RUN_CUDA_FV3_GRAPH`,
`RUN_CUDA_FV3_MPI_TRANSPORT`, `I_MPI_OFFLOAD`,
`RUN_CUDA_FV3_PTDELZ_DELZ_ONLY`, `RUN_CUDA_FV3_REMAPUV_ENABLE` all unset
(defaults preserved).

Results:

- Both arms: `RC=0`, `Range_Warn=0`, `NaN=0`, `Abort=0`.
- CUDA arm: `DUMP_LINES=54`, `TP2D_T=18`, `TP2D_F=0`.
- Fortran arm: `DUMP_LINES=54`, `TP2D_T=0`, `TP2D_F=18`.
- Binary provenance (`USING ****` lines) confirmed each arm ran its own
  scratch-dir-copied binary at the correct arm-specific `EXPDIR`.
- `ALL DONE` reached.

**Call-count reconciliation** (open item at a prior checkpoint, resolved
this pass): `DUMP_LINES=54` at C512 (18 header groups x 3 lines) is lower
than C32's 108 (36 header groups). Confirmed via
`RUN_DT:36`/`n_split:003` in the C512 log — C512 uses a resolution-specific
shorter timestep (36s vs C32's, giving only 3 dyn_core calls x 6 PEs = 18
groups within the same `JOB_SGMT`), consistent with CURRENT_STATE.md's
documented fact that each grid resolution needs its own `bdt`/timestep
stability search. **This is a benign, expected resolution-dependent
difference, not a harness defect** — confirmed by `call=1,2,3` being the
only values present in both arms.

Additional confirmation: after normalizing the `use_cuda_tp2d=[TF]` tag to
a common placeholder, the two arms' full exit-dump files are **byte-for-
byte identical** (`diff` clean) — call indices, PE indices, field names,
and all `q` values (`0.0000000000000000E+00` throughout, confirmed via
`grep -vc` returning 0 nonzero matches) match exactly between arms.

**Stage 9: PASS** under the narrowed claim — real production hord==8 CUDA
dispatch confirmed, correct/matching call counts, correct exclusive
selector behavior each direction, no fallback where not expected, no OOM,
no NaN/Inf, no regression in active model fields (PT/DELZ CUDA path
unaffected in both arms), degenerate `q` bitwise consistent. Does **not**
claim non-trivial tracer-transport numerical content was exercised.

**Stage 10** (Fortran fallback count proof) is covered by the same
evidence: `TP2D_F=18` in the Fortran-control arm exactly matches the
expected call x PE depth.

## 8. Stage 11 — Same-Node Paired C512 Performance: MEASURED REGRESSION

Reusing job 58543590's sequential same-allocation arms (cuda arm ran
first, fortran arm second, per the sbatch's `for ARM in cuda fortran`
loop — order confirmed, no swapped-label risk per the P76A lesson).
`fv_timing_mod` totals over the 3-call window:

| Timer | CUDA (s) | Fortran (s) | Delta | % (cuda vs fortran) |
|---|---|---|---|---|
| TRACER_2D | 1.2002 | 0.8331 | +0.3671 | **+44.06% slower** |
| DYN_CORE | 3.0601 | 2.6867 | +0.3734 | **+13.90% slower** |
| FV_DYN_LOOP (TOTAL) | 7.8319 | 7.1188 | +0.7131 | **+10.02% slower** |
| REMAPPING | 2.2701 | 2.1525 | +0.1176 | +5.46% (noise — REMAP_UV disabled both arms) |

TRACER_2D's delta (0.3671s) accounts for nearly all of DYN_CORE's delta
(0.3734s) — clean attribution: the regression is localized to the tracer
dispatch itself, not spread across unrelated components. REMAPPING's
small delta is not expected to differ (REMAP_UV is at its default-disabled
Fortran path in both arms) and is attributed to ordinary same-node
run-to-run variance, not a hidden interaction.

Per the project's KEEP/DISABLE thresholds (established for this decision
class): ≥5% STRONG POSITIVE, 2-5% MEANINGFUL POSITIVE, 0.5-2% SMALL
POSITIVE, ±0.5% NEUTRAL, >0.5% slower REGRESSION. **TOTAL +10.02% slower
is a REGRESSION.**

**Caveats on this measurement** (stated per the project's evidentiary
rigor precedent, e.g. P48/P76A): this is a single job, 3-call sample,
sequential (not alternating) order, on one node. It is not yet a
replicated, alternating-order, multi-rep measurement of the kind used to
close the original C512 "comparable" baseline finding (§5, Prompt 48).
The direction (CUDA slower on every relevant sub-timer, with clean
attribution to TRACER_2D) is consistent and unlikely to be pure noise
given the same-node same-allocation design, but the exact magnitude
should be treated as a first data point pending corroboration.

## 9. Stage 12 — Attribution

The regression is isolated to `TRACER_2D`, and specifically to the portion
of `tracer_2d_1L` now dispatched through the CUDA `fv_tp_2d` hord==8 path.
Given:

- Prompt 76's kernel-level unit fixtures show round-off-clean numerics
  with no correctness concern and no unusual register/occupancy footprint
  (56 regs/kernel, 0 spills — better than PT/DELZ's historical OOM-risk
  profile).
- The degenerate all-zero `q` field means no non-trivial arithmetic work
  is skipped or added by content (§6 confirms no data-dependent branch).
- The measured slowdown is therefore most plausibly attributable to
  **per-call CUDA dispatch/launch overhead** (kernel launch, any
  host-side setup/staging specific to entering the CUDA branch,
  H2D/D2H synchronization at the tracer-transport call boundary) rather
  than to the transport arithmetic itself.

This has NOT been isolated via profiling (`nsys`) in this task — it is
the recommended next step (§10) before any default-flip decision.

## 10. Stage 13 — Decision

**Verdict: REGRESSION.** Recommendation: the production default should
remain (or explicitly become) Fortran-dispatch for tracer transport
(`use_cuda_tp2d` effectively off, e.g. via
`RUN_CUDA_FV3_TRACER_FORCE_FORTRAN` default-on, or narrowing the selector
condition) pending:

1. A dedicated overhead-attribution follow-up (e.g. `nsys` timeline
   capture across the tracer-dispatch call boundary) to confirm the
   +44.1% TRACER_2D slowdown is launch/dispatch overhead and not a
   latent inefficiency in the integration path itself.
2. A replicated, alternating-order, multi-rep C512 measurement (per the
   P48 rigor bar) before this magnitude is treated as final.

No code change implementing this recommendation was made in this task —
flipping the default is itself a decision outside this task's change
budget and is deferred, consistent with how REMAP_UV's default-disable
was deferred after P71/P72A (later corrected by P76A). **This
recommendation should not be assumed final without the corroboration in
(1)/(2) above** — the same caution P76A demonstrated is warranted here.

## 11. Stage 14 — Next Tracer Target

Unchanged from the pre-Prompt-77 recommendation: `REMAP_Q` (tracer remap,
~116.8 ms/call, same `scalar_profile` family as the already-ported
PT/DELZ) remains the next unmapped candidate. Re-verify arm-labeling/
timer-direction conventions before implementing, per the P76A precedent.
Given this task's regression finding, any new tracer-path CUDA work
should budget for an explicit dispatch-overhead check before declaring
performance results, not just a correctness check.

## 12. Documentation-Requirement Notes

- **Build-infrastructure fix (from a prior part of this task, recorded
  here as required)**: `fv3_savepoint_mod.F90` intermittently failed
  Intel ifort with `#6633` "type of the actual argument differs from the
  type of the dummy argument" on calls like
  `fv3_sp_dump2d(sp, 0, 'area', 'i,j', gridstruct%area, isd, ied, jsd, jed)`.
  Root cause: a stale `.mod`/`.o` build-tree state — this translation
  unit had never previously compiled to completion in either the R8
  (`build/`) or R4 (`build_r4/`) trees (only a stale `.mod` and a
  `.provides.build` stamp existed, no `.o`). Fixed by a clean-and-
  reconfigure of both affected build trees, not by any source edit.
  **Numerical semantics changed: NO.**
- **Cold-start tracer-zero limitation**: recorded in
  `CURRENT_STATE.md` §7 as a structural, harness-family-wide limitation
  (not HS-specific), with an explicit instruction that future
  tracer-numerics validation work must not assume non-zero `q` from a
  cold-start run without a dedicated, explicitly-scoped fix.
- **Prompt76-vs-Prompt77 evidentiary split**: recorded verbatim in
  `CURRENT_STATE.md`'s lead paragraph and §5 — Prompt 76 = non-degenerate
  kernel numerical validation; Prompt 77 = production-dispatch/
  integration validation only.

## 13. Summary Table

| Stage | Item | Result |
|---|---|---|
| 0-6 | Build/static regression | PASS (prior session) |
| 7/8 | C32 dispatch-wiring A/B | PASS (narrowed claim) |
| 9/10 | C512 dispatch-wiring A/B | PASS (narrowed claim) |
| — | Pre-Stage-11 zero-tracer early-exit check | No early exit found — performance timing valid |
| 11 | Same-node paired C512 performance | MEASURED REGRESSION (TOTAL +10.02% slower) |
| 12 | Attribution | Isolated to TRACER_2D/CUDA dispatch overhead (not yet profiled) |
| 13 | Decision | REGRESSION — recommend default remains Fortran-dispatch, pending corroboration |
| 14 | Next target | `REMAP_Q` (unchanged); dispatch-overhead check now a required step for future tracer CUDA work |

**Result: PRODUCTION INTEGRATION WIRING COMPLETE AND VALIDATED (narrowed
claim); PERFORMANCE REGRESSION MEASURED — DO NOT ENABLE BY DEFAULT
PENDING FURTHER INVESTIGATION.**

**Follow-up (Prompt 78, 2026-09-24)**: the regression flagged here was
confirmed and attributed with a rigorous 33-call alternating-order A/B
(+37.87% TRACER_2D, replacing this doc's single 3-call sequential
sample) plus an `nsys` rank-0 capture. Root cause: ~300 small
synchronous host<->device `cudaMemcpy` calls per `dyn_core` call in the
CUDA tracer path (absent from the Fortran arm), ~18% of the regression
directly CUDA-API-visible, ~82% CPU-side orchestration overhead around
those transfers. GPU kernel compute itself is negligible (~1.15ms/call).
See `docs/performance/p78_tracer_hord8_overhead_attribution_2026-09-24.md`.
