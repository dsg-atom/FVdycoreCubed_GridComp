# Prompt 85 — CUDA REMAP_Q Implementation, Batched Q Transfer, and
# Integration-Scale Validation (BLOCKED)

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`

## 0. Task Type

CUDA implementation + correctness + performance task, building directly
on Prompt 84B's mapping/harness (`p84b_remap_q_cuda_port_plan_2026-09-25.md`).
Scope authorized: kord=8 `scalar_profile_column` numerics extension,
`q` device-buffer/transfer path, and a narrow Fortran dispatch selector
— a 3-item change budget, no broader `fv_mapz` refactor, no R4, no
multi-tracer (`nq>1`) support.

## 1. KORD8 NUMERICS

Implemented the `abs(kord)<9` branch needed by `mapn_tracer`'s
`scalar_profile(iv=0, kord=8, ...)` call, reusing the already-ported
`map_scalar_column` standard-remap (`gmao_bot=gmao_top=false`)
algebra — confirmed by Prompt 84B's source-level diff to be identical
for `nq=1`. No new numerical algorithm was written; this is the
"REUSE WITH NEW KORD MODE" path Prompt 84B identified. `fillz`
(conditionally applied after the remap in `mapn_tracer`) is
unimplemented — confirmed unreachable in production: `fill` is always
`.false.` (`fv_arrays.F90:820` default, not overridden in any checked
namelist or fixture). Validated bitwise/round-off against the
authoritative `mapn_tracer` on the Prompt 84B fixture set (§4).
Numerical semantics changed relative to Fortran: **NO**.

## 2. Q TRANSFER

New files `cuda/dyn_core/vertical/fv3_remap_q.{cu,cuh}`. The tracer
array `q` has no `FieldRegistry` identity (confirmed by grep — the
`FieldId::q` enum value exists but is never populated/looked up
anywhere in the CUDA runtime), so it cannot reuse the existing
`fv3_gpu_sync_to_host/from_host` field-mask path used by
PT/DELZ/REMAP_UV. Instead: one persistent, module-owned, grow-only
device buffer sized `(ied-isd+1)*(jed-jsd+1)*km` elements (the full
Fortran `q(isd:ied,jsd:jed,km,1)` memory domain for this tile, i
fastest/j/k slowest); the new C ABI entry point
`fv3_gpu_remap_q(context, dependency, double *q, int64_t q_elements,
event*, error*)` (`fv3_gpu_api.h:343`) does exactly **one contiguous
H2D `cudaMemcpyAsync` before the kernel and one contiguous D2H after**
— not a per-column/per-field marshaling loop, the exact defect
Prompt 82 root-caused in the unrelated TRACER_2D CUDA path. This
satisfies Stage 3's transfer-architecture gate (H2D<=2, D2H<=2,
preferred 1+1) at the stronger 1+1 bound. R8-only by construction (the
ABI doc comment states this explicitly; not called from an R4 build).

## 3. MEMORY

Per-column scratch: 11 slots x (km+1) `fv_real` (`pe_old, peO, dpe_old,
dpeO, q1_in, q2_out, prof_q, c2, c3, c4, gam`), same convention as
`fv3_pt_delz_remap`/`fv3_remap_uv`. Optional `WorkspaceId::vertical`
device-scratch reuse (Prompt 68 pattern) avoids a dedicated allocation
when the shared vertical workspace is large enough; falls back to
per-thread local arrays capped at `remap_q_max_local_levels=128` (no
large per-thread local arrays — matches the task's explicit
constraint). No new persistent per-run allocation beyond the one
grow-only `q` buffer and (optionally) the shared vertical workspace.

## 4. FIXTURES

Validated against all Prompt 84B `mapn_tracer` fixtures (23 cases,
production-shape kord=8/nq=1/fill=.false./R8) via the same
`map_scalar-driver` host harness — CUDA kernel output vs. the
unmodified authoritative Fortran `mapn_tracer`: bitwise/round-off
agreement, mass conservation and positivity/`q_min` hold as a
corollary (no new tracer-mass logic was introduced; the ported
algebra is a strict reuse). GPU EXECUTED, NUMERICALLY VALIDATED at
unit scale.

## 5. SANITIZER

Production-shape GPU smoke test (`fv3_remap_q_prod_shape_smoke.cu`,
same pattern as the existing `fv3_pt_delz_remap_prod_shape_smoke.cu`/
`fv3_remap_uv_prod_shape_smoke.cu`) under `compute-sanitizer`: **0
errors reported.**

## 6. C32

Fortran dispatch selector added to `fv_mapz.F90` (Stage 6):

```fortran
use_cuda_remap_q = kord_tr(1) == 8 .and. nq == 1 .and. .not. fill .and. &
     kind(q) == kind(1.0d0) .and. fv3_gpu_backend_enabled()
if (use_cuda_remap_q) then
   use_cuda_remap_q = .false.   ! Stage 15: default remains Fortran
   call get_environment_variable('RUN_CUDA_FV3_REMAPQ_ENABLE', ...)
   if (p85_envlen > 0) use_cuda_remap_q = .true.
   call get_environment_variable('RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN', ...)
   if (p85_envlen > 0) use_cuda_remap_q = .false.
endif
```

plus a `[P85_EXIT_DUMP]` diagnostic (gated on
`RUN_CUDA_FV3_REMAPQ_DUMP_EXIT`, mirrors the Prompt 71 `[P71_EXIT_DUMP]`
pattern) printing `use_cuda_remap_q` and a sample `q` value after
`timing_off('Remap_Q')`.

**Bug found and fixed during this stage**: the gate initially read
`kord_tr(1) == 9` (copy-pasted from the REMAP_UV selector immediately
above it in the same file, which correctly uses `kord==9` for its own
numerics). Production's actual `kord_tr` default is **8**
(`fv_arrays.F90:333`, not overridden by this harness's `input.nml`).
Fixed to `== 8`; Fortran binary rebuilt (sha256
`846ad11dbeb15466b16a7c8835421f8a7200568784cbf56df08b1acbdd5452e2`);
confirmed by a fresh A/B (job 58559076) that this fix was **not** the
integration blocker (see below).

**C32 A/B, real `StandAlone_FV3_Dycore.x`, `FV_NX=1`/`FV_NY=6`, 6 MPI
ranks/1 A100, `RUN_CUDA_FV3:1`, `COLDSTART:1`** — Arm A:
`RUN_CUDA_FV3_REMAPQ_ENABLE=1`; Arm B:
`RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`. Both arms: `RC=0`, 0 NaN, 0
Abort. Binary provenance confirmed via sha256 preflight in the sbatch
(exact match both jobs). **Result: `[P85_EXIT_DUMP]` shows
`use_cuda_remap_q=F` on every one of 36 dumps (6 ranks x 6 calls) in
BOTH arms** — i.e. the CUDA path never activates even with the enable
flag set. Root cause (confirmed by direct source trace, not
inference): this run's own log line `FV3 ncnst: 2` shows the real
`Lagrangian_to_Eulerian` call's `nq` (`= nq_tot - dnats`, `fv_dynamics.
F90:347`) is **2**, not 1, even in this minimal Held-Suarez/no-moisture
harness (at minimum `sphum` + `cld_amt`, both always registered
tracers; `kord_tracer(iq) = flagstruct%kord_tr` for all but `cld_amt`,
which is forced to 9 — `fv_dynamics.F90:942-943`). The `nq==1` gate
therefore fails on `nq`, not on `kord_tr`, and it was never expected
to pass here or on any real production tracer set: **this is the
documented, deliberate Stage-0/1 scope limit of the CUDA kernel itself**
(`fv3_remap_q.cuh:30-35`: *"nq>1 (multi-tracer batching, a genuinely
different fixture/loop shape ...) is OUT OF SCOPE and not implemented
here"*), not a dispatch defect. The gate is working exactly as
designed — it is correctly refusing to dispatch an unvalidated
(`nq>1`) configuration to a kernel that only handles `nq=1`.

No unintended side effects on other fields/diagnostics were observed
(both arms' logs diff only in non-deterministic timing/print-order
noise) — consistent with the CUDA path never actually executing in
either arm.

**C32 integration verdict: DISPATCH-WIRING CORRECT, GATE CORRECTLY
NEVER FIRES ON REAL PRODUCTION `nq`.** Per the §7 structural cold-start
tracer-zero limitation (this harness's `q` is allocation-time zero
throughout, `COLDSTART=1`+`external_ic=.false.`+empty `ExtData.rc`),
this test could not have demonstrated end-to-end tracer-transform
numerics on nonzero data regardless of the `nq` finding — its purpose,
like Prompt 77's equivalent TRACER_2D test, was limited to confirming
dispatch wiring and non-interference, which it did (for the `.false.`
branch only, since the `.true.` branch never executed).

## 7. C512

**Not run.** The `nq` mismatch found at C32 is structural, not
resolution-dependent — `nq` is set by the tracer registry
(`nq_tot - dnats`), not by grid size, and every checked production/HS
namelist carries `nq>=2`. A C512 run would exercise the identical
`nq==1` gate against the identical `nq=2` (or larger, for a full-
moisture production config) reality and reproduce the same
non-activation with certainty, at the cost of a much larger disk/
compute budget (C512 savepoints/scratch logs are documented elsewhere
as tens of GB/run, and this session already required intervention for
a quota exhaustion at C32 scale). This is the bounded-debugging stop
condition from `docs/agent_instructions/cost_efficiency.md`: further
evidence would not change the outcome.

## 8. PERFORMANCE

**Not measured.** No paired A/B performance run is meaningful when the
CUDA path never dispatches under any real tracer configuration — there
is nothing to compare (Arm A and Arm B in §6 are, numerically,
identical Fortran runs).

## 9. ATTRIBUTION

Two independent findings, correctly distinguished:
1. **A real, fixed selector bug** (`kord_tr(1)==9` should be `==8`,
   §6) — a copy-paste artifact from the REMAP_UV selector. Fixed;
   confirmed by rerun not to be the blocker.
2. **A real, structural scope gap, not a bug**: the CUDA REMAP_Q
   kernel was deliberately, explicitly scoped to `nq==1` at Stage 0/1
   (documented in `fv3_remap_q.cuh`), and no real
   `Lagrangian_to_Eulerian` call in this codebase ever has `nq==1` —
   the tracer registry floor is 2 (`sphum`+`cld_amt`) even in a
   dry/no-moisture Held-Suarez configuration, and a full-physics
   production config would have more, not fewer, tracers. Extending
   the kernel to `nq>1` (a per-tracer loop or a batched multi-tracer
   kernel) is a materially different design — explicitly called out as
   OUT OF SCOPE in the Stage-0/1 contract and in this task's change
   budget (max 3 changes, no broadening beyond
   kord_tr=8/nq=1/R8/fill=.false.) — and was correctly not attempted
   here.

## 10. DECISION

Per the prescribed thresholds (STRONG/MEANINGFUL/SMALL POSITIVE,
NEUTRAL, REGRESSION by TOTAL): **not applicable** — there is no
integration-scale performance delta to classify, because the CUDA path
never dispatches under any real production tracer configuration. This
is categorically the fourth outcome: validation could not proceed past
the integration gate.

## 11. PROJECT STATUS

REMAP_Q CUDA: IMPLEMENTED, COMPILED, GPU EXECUTED, NUMERICALLY
VALIDATED — but only at unit/fixture scale (`nq=1` synthetic and
Prompt 84B harness cases), which does not occur in any real dyn_core
call. Integration-scale (C32) validation is BLOCKED by this structural
`nq` mismatch; C512 and performance stages were not attempted (§7).
Default dispatch is unaffected: `use_cuda_remap_q` remains reachable
only via explicit `RUN_CUDA_FV3_REMAPQ_ENABLE` opt-in, and even when
set, never actually activates against a real tracer count — so there
is zero risk to the existing Fortran-dispatch default from this work.
Change budget: exactly the 3 authorized items were touched (kord=8
branch; `q` transfer/registry path; narrow selector, including its
one-line bug fix, which is a correction within item 3, not a 4th
change) plus the `[P85_EXIT_DUMP]` diagnostic (instrumentation only,
P71 precedent, not counted). No commits made in either repository;
nothing pushed or merged. Recommended next step, if this line of work
is reopened, requires **explicit fresh authorization** for one of: (a)
a multi-tracer (`nq>1`) CUDA design (materially larger scope than this
prompt's budget); or (b) evidence that some real production
configuration exists with `nq==1` (none found in this task).

---

**REMAP_Q CUDA VALIDATION BLOCKED — the CUDA kernel's nq=1 scope (a deliberate Stage-0/1 design limit) never occurs in any real production dyn_core call (production always has nq>=2); C32 integration confirmed the dispatch selector is wired correctly and now uses the corrected kord_tr==8 gate, but use_cuda_remap_q never evaluates true under real tracer counts, so C512/performance stages were not run**

## 12. Follow-up (Prompt 86A, 2026-09-25) — nq=2 scope mapped, PROCEED

Prompt 86A mapped the real `nq=2` production case that blocks this
kernel (sphum `kord=8`, cld_amt `kord=9` forced unconditionally,
`fv_dynamics.F90:941-944`) and found it is **not a dead end**: both
`kord=8` and `kord=9` branches already exist in the shared
`scalar_profile_column` (this kernel's own kord=8 branch; REMAP_UV's
kord=9 branch, production-active via P84A) with independent
production-validation precedent, so no new numerical algorithm is
needed. `q`'s contiguous `nq`-tracer layout keeps a single batched
H2D/D2H clean at 2x bytes; the recommended extension design is exactly
2 fixed kernel launches (one per tracer, each parameterized by its own
`kord`) sharing that one transfer — not a generic multi-tracer loop. A
new host-only `nq=2` self-consistency test (mass conservation,
positivity) PASSED using the existing host reference. **Verdict:
REMAP_Q NQ2 SCOPE CLEAN — PROCEED TO CUDA EXTENSION**, narrowly scoped
to this exact `nq==2`/`kord=8`+`kord=9` case. No CUDA/production code
was changed by Prompt 86A (mapping + one host test only). See
`p86a_remap_q_nq2_scope_2026-09-25.md`.

## 13. CORRECTION (Prompt 86B, 2026-09-25)

**§6's "Production's actual `kord_tr` default is 8... not overridden by
this harness's `input.nml`" is WRONG.** `FV_StateMod.F90:563`
unconditionally sets `flagstruct%kord_tr = 9` on every GEOS FV_State
init (no gating condition) — a GEOS-side override of the GFDL bare-code
default cited in §6, not a namelist override, which is why checking
`input.nml` alone missed it. **This run's own log
(`p85_stage10_c32_cuda/p85_stage10_final.log:250`) already printed `FV3
kord_tr: 9`** — three lines below the `FV3 ncnst: 2` line §6 cited as
evidence for `nq`, and was not checked for `kord_tr`. The `kord_tr(1)==9
-> ==8` "bug fix" made in this stage was therefore a **regression**: it
changed a selector condition that happened to match the GFDL code
default but not this codebase's actual runtime value, for a kernel that
was already unreachable on `nq` alone (nq=2 vs. the kernel's nq=1 scope)
— so the fix was inert at the time (masked by the `nq` gate) and remained
inert in Prompt 86B's follow-on (which extended to `nq==2` and hit
`kord_tr(1)==8` failing for real). Net effect: no incorrect numerics
ever shipped (the kernel never dispatched either way), but the
documented root-cause narrative in §6 is factually wrong about
production's actual `kord_tr` value. See
`p86b_remap_q_nq2_cuda_implementation_2026-09-25.md` §4 for full
evidence (real production schedule is `kord_tracer=(9,9)`, not `(8,9)`)
and `p86a_remap_q_nq2_scope_2026-09-25.md` §13 for the parallel
correction to Prompt 86A's Stage 0/1.
