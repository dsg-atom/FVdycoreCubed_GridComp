# Prompt 86A — REMAP_Q Real Production `nq=2` CUDA Scope Mapping

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`

> **Superseded by Prompt 87**
> (`p87_remap_q_actual_kord9_production_2026-09-25.md`): the kord=(8,9)
> mapping here assumed the wrong kord schedule (real production forces
> `kord_tr=(9,9)`, found by Prompt 86B). Prompt 87 corrected the kord
> and found a further, independent blocker (`fill=.true.`). Read
> Prompt 87's doc for the current state of this line of work.

## 0. Task Type

Source-mapping + architecture-design task, no CUDA implementation. Builds
directly on Prompt 85's finding that the CUDA REMAP_Q kernel (kord=8,
scoped to `nq==1` by design) never activates in production because every
real `Lagrangian_to_Eulerian` call has `nq>=2`. Do-NOT list observed in
full: no `nq>1` CUDA implementation; no modification to the existing
`nq==1` kernel's numerics; no touching PT/DELZ, REMAP_UV, the tracer-CUDA
production path, `MOIST_CAPPA`/`USE_COND`, or R4; no broadening beyond the
actual production `nq==2` configuration. No C512 run, no `nsys`/`ncu`, no
commit/push.

## 1. PRODUCTION NQ (Stage 0)

Confirmed by direct Fortran source read (not inference from P85's doc):
`fv_dynamics.F90:941-944`,

```fortran
do iq=1,nq
   kord_tracer(iq) = flagstruct%kord_tr
   if ( iq==cld_amt )  kord_tracer(iq) = 9      ! monotonic
enddo
```

runs unconditionally on every `Lagrangian_to_Eulerian` call
(lines 952-962). Combined with the P85 doc's own run-log evidence
(`FV3 ncnst: 2`, `nq = nq_tot - dnats`) and `fv_arrays.F90:333`'s
production `kord_tr` default of 8: real production has **exactly two
tracers, `nq=2`**, with `kord_tracer(1)=8` (sphum) and `kord_tracer(2)=9`
(cld_amt, forced, regardless of `flagstruct%kord_tr`). This is not
conditional on Held-Suarez/no-moisture configuration — `sphum`+`cld_amt`
are registered tracers even in the minimal HS harness that produced P85's
`nq=2` finding.

## 2. TRACERS (Stage 1 — array layout)

`q1(isd:ied,jsd:jed,km,nq)` — `iq` is the outermost (last) Fortran
dimension. The whole `nq`-tracer array for one tile is therefore one
contiguous memory region: tracer 1's full 3D block, then tracer 2's, with
no interleaving. A flat single H2D/D2H covering both tracers remains
legal — same base pointer, size scaled by `nq` (2x the current `nq=1`
buffer), not a per-tracer transfer.

## 3. INDEPENDENT (Stage 2 — mapn_tracer tracer coupling)

`mapn_tracer`'s pressure-interval search geometry (`pl`, `pr`, `fac1`,
`fac2`, `dp`, `esl`, `k0`, `l`, `m` — functions of `(i,k)` only) is
computed once per column and reused across the `do iq=1,nq` value-copy
loop. Classification: **B — tracers share only pressure-grid geometry;
final per-tracer values are numerically independent.** `scalar_profile`
itself is called separately per `iq` with that tracer's own `kord_tr(iq)`
and its own `q1(...,iq)` column — there is no cross-tracer numerical
coupling inside the profile/remap math itself.

## 4. SPECIAL NUMERICS (Stage 3 / Stage 7 — kord and moisture semantics)

**The critical, previously-open question — does `kord_tracer(2)` also
equal 8? — is answered NO.** `cld_amt`'s kord is unconditionally forced to
9 (monotonic), per `fv_dynamics.F90:941-944` above. In `fv_mapz.F90`'s
`scalar_profile`, `abs(kord)<9` (kord=8's branch) and
`elseif (abs(kord)==9)` (its own distinct extrema/monotonicity `extm`
logic) are separate, mutually exclusive branches — kord=8 and kord=9 are
genuinely different code paths, not two settings of one shared path.

**De-risking finding**: the CUDA-side shared `__host__ __device__`
function `scalar_profile_column` in
`cuda/dyn_core/vertical/fv3_map_scalar_column.hpp` **already implements
both branches** in one runtime-parameterized function — `kord==9`
(its own original/default behavior, already production-validated via
REMAP_UV's `kord_mt=kord_wz=kord_tm=9` configuration, restored to the
production default by P84A) and `abs(kord)<9` (Prompt 85's kord=8
addition for REMAP_Q). Both branches are read-only reused here; neither
REMAP_UV nor the existing kord=8 branch was modified in this task.

`mapn_tracer`/REMAP_Q itself contains no moisture-identity-specific
branching — the only tracer-identity-specific behavior (kord=8 vs. 9)
happens upstream in the Fortran caller (`fv_dynamics.F90`), and stays
within REMAP_Q's own mapping-order-selection semantics (which `kord` to
apply), not thermodynamic/moisture coupling. This satisfies Stage 7's
"REMAP_Q-local only" requirement — no `MOIST_CAPPA`/`USE_COND`/Riemann
terms are implicated.

**`q_min`/positivity**: shared literal `0.` at the production call site
(`fv_mapz.F90:903`) for all tracers — not per-tracer, no special-casing
needed. `fill` is always `.false.` in production (confirmed by P85);
`fillz` remains correctly unimplemented for both tracers.

## 5. ONE BATCHED H2D / ONE BATCHED D2H (Stage 4 — transfer architecture)

The existing Prompt-85 architecture already does exactly one contiguous
H2D before the kernel(s) and one contiguous D2H after (`fv3_gpu_remap_q`,
1+1, the strict end of the Stage-3 gate). Because `q`'s `nq`-tracer block
is contiguous (§2), extending to `nq=2` requires only:
- one H2D of `2x` the current `nq=1` byte count (same base pointer,
  `q(isd:ied,jsd:jed,km,1:2)`), before either kernel launch;
- one D2H of the same size, after both kernels complete.

No new FieldRegistry integration is implied — `q` remains host-only,
owned by `fv3_gpu_interface_remap_q`'s own persistent grow-only device
buffer (now sized for 2 tracers instead of 1), exactly as designed in
P85 §2.

## 6. Stage 5 — kernel parallelism design (A/B/C)

Three designs considered:

- **A — single kernel, per-thread `kord` branch based on tracer index**:
  rejected. Would require every thread to carry/branch on a tracer index
  inside the kernel body, mixing kord=8/kord=9 control flow in one
  launch; more divergence-prone and a bigger diff to the existing kernel
  than necessary.
- **B — exactly 2 kernel launches, one per tracer, each parameterized by
  its own `kord` (8 or 9), sharing the one combined H2D/D2H.** Selected.
  Bounded, fixed launch count (never data-dependent), each launch is a
  near-verbatim reuse of the existing `nq==1` kernel with `kord` promoted
  from a compile-time literal (`fv3_remap_q.cu:116`,
  `/*kord=*/8`) to a runtime/launch parameter — mechanical, not new
  algorithm work, since `scalar_profile_column` already supports both
  values (§4).
- **C — a single generic per-tracer loop/batch kernel for arbitrary
  `nq`**: rejected as out of scope — the Do-NOT list explicitly forbids
  broadening beyond the actual production `nq==2` configuration, and a
  generic unbounded design is exactly the shape of the P77-82 tracer-CUDA
  regression (that regression's root cause was ~300 small, unbounded,
  synchronous transfers/call — not kernel launches, but the same
  "generalize now" instinct). Design B keeps both the launch count and
  the transfer count small, fixed, and explicit — exactly 2 launches, 1
  H2D, 1 D2H — the opposite of that failure mode.

**Decision: Design B.**

## 7. WORKSPACE (Stage 6)

Per-column scratch is unchanged in kind (11 slots x (km+1) `fv_real`, same
convention as PT/DELZ/REMAP_UV/P85's REMAP_Q), just needed by two
launches instead of one. Options: (a) two independent scratch allocations
(one per tracer's launch), or (b) reuse the same scratch buffer
sequentially across the two launches (tracer 2's launch only starts after
tracer 1's scratch is no longer needed, since the launches are not
required to overlap). Either is linear in `nq` — no super-linear scaling,
no new large per-thread local arrays, no change to the existing
`WorkspaceId::vertical` shared-device-scratch reuse pattern (Prompt 68)
or the `remap_q_max_local_levels=128` fallback cap. Recommend (b) for
this `nq=2` scope: minimal added device memory over the existing `nq=1`
footprint.

## 8. HARNESS NQ2 (Stage 8 — host-only production-shaped test)

Existing 23-fixture host test (`test_remap_q_tracer.cpp`) validates only
`nq=1`. A pre-existing `nq=3` fixture from Prompt 84B
(`case23_tracer_multi_nq3.txt`) was found incomplete — input `q1` profiles
only, no `pe1`/`pe2` pressure grid and no golden Fortran output — not
usable as a bitwise-comparison fixture without running new Fortran jobs
(out of this bounded task's scope).

Instead, wrote a new host-only (no CUDA) self-consistency test,
`cuda/dyn_core/vertical/tests/../test_remap_q_nq2_scope.cpp`
(`cuda/dyn_core/tests/vertical/test_remap_q_nq2_scope.cpp`), using
**only the existing, already-validated** `map_scalar_column<double>`
host reference (no new numerical code):

- a shared, perturbed `pe1`/`pe2` pressure grid (10 levels, a modest
  Lagrangian-to-Eulerian shift, same shape family as the existing
  `case22_tracer_c512_like_profile.txt` fixture), boundary-preserving
  (`pe2[0]=pe1[0]`, `pe2[km]=pe1[km]`), mirroring `mapn_tracer`'s
  once-per-column shared geometry;
- **tracer 1**: a smooth, strictly positive, decaying-with-height
  `sphum`-shaped profile, run with `kord=8`;
- **tracer 2**: a bounded-fraction `cld_amt`-shaped profile with a sharp
  front, run with `kord=9`;
- checks, per tracer: column mass conservation
  (`sum(q1*dpe1)` vs. `sum(q2*dpe2)`, relative tolerance `1e-12`), zero
  NaN/Inf, and positivity (`q2 >= 0` everywhere).

Compiled (`g++ -std=c++17 -O2 -DFV3_GPU_FV_REAL_IS_DOUBLE=1`) and run
directly on the login node (host-only C++, no CUDA/GPU/Slurm required).
**Result:**

```
sphum_like(kord=8)     mass_in=2.955594828832147e+01 mass_out=2.955594828832147e+01 mass_rel=1.202e-16 nan=0 any_negative=no PASS
cld_amt_like(kord=9)   mass_in=3.060000000000000e+03 mass_out=3.060000000000000e+03 mass_rel=0.000e+00 nan=0 any_negative=no PASS
RESULT=PASS
```

Both tracers PASS mass conservation (round-off, ~1e-16 relative) and
positivity, on a synthetic but production-shaped 2-tracer case, using the
already-validated host reference for both kord values. This is **not** a
claim of GPU execution or CUDA-kernel equivalence — no CUDA code was run
in this task. It is scope evidence: the numerics needed for both
production tracers are already known-good on the host reference that the
existing CUDA kernel is itself validated against.

Wired into the build as a permanent CTest target,
`fv3_wp_c_remap_q_nq2_scope_test` (`cuda/dyn_core/CMakeLists.txt`,
inserted after the existing `fv3_wp_c_remap_q_tracer` target).

## 9. PORT SIZE (Stage 9)

Estimated **SMALL-to-MEDIUM** — smaller than initially expected before
this mapping. Both numerical branches needed (kord=8, kord=9) already
exist in `scalar_profile_column` with independent production-validation
precedent (P85 for kord=8, REMAP_UV/P84A for kord=9). Remaining work is
ABI/dispatch/gate plumbing, not new algorithm risk:
1. Promote `fv3_remap_q.cu`'s hardcoded `/*kord=*/8` (line 116) to a
   per-launch runtime parameter.
2. Size the persistent device `q` buffer for `nq=2` (2x current
   allocation) and issue exactly 2 kernel launches (Design B, §6) sharing
   one H2D/one D2H (§5).
3. Narrow the Fortran selector from `nq==1` to an explicit
   `nq==2 .and. kord_tr(1)==8 .and. kord_tracer(2)==9`
   (or equivalently gate on `cld_amt`'s forced kord) — **not** a general
   `nq>1` relaxation, preserving the Do-NOT list's scope constraint.
4. A real (not synthetic) `nq=2` fixture set derived the same way the
   existing 23 `nq=1` fixtures were derived from authoritative
   `mapn_tracer` (out of this task's scope; needed before any CUDA
   implementation is authorized).

## 10. PERFORMANCE CEILING (Stage 10 — estimate only, no new measurement)

**No new CUDA measurement was performed in this task.** Estimate grounded
in same-family precedent:
- REMAP_UV (kord=9 port, same `scalar_profile_column` family): ~10.23x
  kernel speedup (P76A), TOTAL +~1.0% at C512, now the restored
  production default (P84A).
- PT/DELZ (same family): REMAPPING −33.7%, TOTAL −7.3% at C512 (P69).

Given REMAP_Q's own `Remap_Q` timer contribution (116.8 ms/call, per
prior mapping) and that Design B doubles work (2 launches, 2x transfer
bytes) without doubling transfer *count*, a favorable outcome in the same
qualitative range as REMAP_UV/PT-DELZ is plausible, but this is
explicitly an **estimate, not a measurement** — no C512 run, no `nsys`,
no `ncu` was performed or is claimed.

## 11. DECISION (Stage 11)

Criteria checked:
- Transfer/workspace architecture: clean (§5, §7) — rules out "TRANSFER/
  WORKSPACE ARCHITECTURE NOT CLEAN."
- Numerical scope: stays strictly REMAP_Q-local; both needed kord
  branches already exist and are independently validated; no
  moisture/thermodynamic coupling implicated (§4) — rules out "REQUIRES
  BROADER MOISTURE/TRACER WORK."
- No blocker or STOP-triggering finding was identified anywhere in
  Stages 0-10.

**Verdict: REMAP_Q NQ2 SCOPE CLEAN — PROCEED TO CUDA EXTENSION**, with
the explicit caveat that any future implementation must stay narrowly
scoped to the exact `nq==2`/`kord=8`+`kord=9` production case (Design B,
§6) — not a generic multi-tracer/arbitrary-`kord` relaxation — mirroring
the disciplined pattern that worked for PT/DELZ and REMAP_UV, and
explicitly avoiding the "generalize now" pattern that caused the P77-82
tracer-CUDA regression.

## 12. PROJECT STATUS

No CUDA implementation, no C512 run, no `nsys`/`ncu`, no commit/push
performed in this task, per the Do-NOT list. Changes made: one new
host-only test file (`cuda/dyn_core/tests/vertical/
test_remap_q_nq2_scope.cpp`) and one new CMake test target in
`cuda/dyn_core/CMakeLists.txt` — both additive, no existing CUDA/
production source modified. `use_cuda_remap_q` and the existing `nq==1`
kernel are unaffected; Fortran REMAP_Q remains the sole active production
path.

---

**REMAP_Q NQ2 SCOPE CLEAN — PROCEED TO CUDA EXTENSION**

## 13. CORRECTION (Prompt 86B, 2026-09-25)

**§1's `kord_tracer(1)=8 (sphum)` claim is WRONG for this codebase's
actual GEOS-integrated runtime.** §1 read `flagstruct%kord_tr`'s bare
GFDL code default (`fv_arrays.F90:333`, `= 8`) but did not check that
`FV_StateMod.F90:563` unconditionally overrides it to 9
(`kord_tm=kord_mt=kord_wz=kord_tr=9`, no gating condition, fired on every
GEOS FV_State init). The run log this doc's own §1/§8 cited already
contained the contradicting evidence one line below the cited `FV3
ncnst: 2` — `FV3 kord_tr: 9` — and was not checked. Real production
schedule is **kord_tracer(1)=9 (sphum), kord_tracer(2)=9 (cld_amt)** —
both tracers use the same kord, not the assumed (8, 9) split. This
invalidates §4's "genuinely different code paths" framing as a
description of the *actual* production case (both branches still exist
and are individually valid/validated — only the claim that production
*uses* both is wrong), and is the direct cause of Prompt 86B's Stage 10
BLOCKED finding (CUDA dispatch count 0/0 in a real C32 A/B, because the
Prompt-86B selector correctly checks `kord_tr(1)==8`, which is never
true). See `p86b_remap_q_nq2_cuda_implementation_2026-09-25.md` §4 for
full evidence and recommendation. No other section of this doc (nq=2,
array layout, tracer independence, transfer architecture, workspace) is
affected — only the kord *values* assumed, not the design built around
supporting two independently-parameterized kord launches.
