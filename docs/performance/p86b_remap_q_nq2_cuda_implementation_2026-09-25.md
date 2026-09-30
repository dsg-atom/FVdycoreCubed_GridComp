# Prompt 86B — REMAP_Q `nq=2` CUDA Implementation: BLOCKED at Stage 10

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`

> **Superseded by Prompt 87**
> (`p87_remap_q_actual_kord9_production_2026-09-25.md`): the kord=(8,9)
> schedule assumed here was wrong (real production forces
> `kord_tr=(9,9)`, correctly diagnosed below); Prompt 87 corrected the
> kord to `(9,9)` and re-ran C32 integration, finding a **second**,
> independent blocker (`FV_StateMod.F90:568`'s `flagstruct%fill=.true.`).
> Read Prompt 87's doc for the current state of this line of work.

## 0. Task Type

Bounded CUDA implementation task extending Prompt 85's `nq=1`/kord=8 CUDA
REMAP_Q path to Prompt 86A's mapped `nq=2` production schedule
(sphum kord=8, cld_amt kord=9), per a fixed 3-change budget: (1) nq=2/
kord-schedule launcher, (2) `q` buffer/ABI extension, (3) production
selector update. Stages 0-9 (implementation + GPU-level validation):
**PASS**. Stage 10 (C32 Fortran-vs-CUDA production integration):
**BLOCKED** — CUDA dispatch count is 0 in both arms of a real C32 A/B, for
a hard architectural reason unrelated to this task's own 3 changes (see
§4). Per the governing operational instruction ("stop at the first
failing stage, report BLOCKED"), Stages 11-14 (C512 integration, paired
performance, attribution) were not attempted.

## 1. Implementation (Stages 1-6) — DONE

Change 1 — nq=2/kord-schedule launcher: `RemapQArgs` gained a `kord`
field; `launch_remap_q_ctx` issues exactly two sequential
`launch_remap_q` calls on the same stream/workspace, first with
`kord=8`/`q` at the buffer base, second with `kord=9`/`q` offset by
`remap_q_buffer_elements(isd,ied,jsd,jed,km)` (the single-tracer-plane
stride) — Design B from Prompt 86A §6, unmodified.

Change 2 — `q` buffer/ABI extension: the persistent device buffer in
`fv3_gpu_interface_remap_q` is now sized for 2 contiguous tracer planes
(2x the Prompt-85 `nq=1` allocation); `fv3_gpu_remap_q`'s bind(C)
signature gained `nq`/`kord1`/`kord2` arguments. One H2D of the full
2-plane buffer, one D2H of the same — unchanged from Prompt 85's 1+1
transfer-count architecture (verified directly, §2).

Change 3 — production selector update (`fv_mapz.F90:486-500`): narrowed
from `nq==1` to
`kord_tr(1)==8 .and. nq==2 .and. .not.fill .and. kind(q)==kind(1.0d0) .and. fv3_gpu_backend_enabled()`,
then `kord_tr(2)==9`, gated further by opt-in
`RUN_CUDA_FV3_REMAPQ_ENABLE`/override `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN`
(both env-var, default off — default remains Fortran). No numerical
helper redesigned; `scalar_profile_column`'s existing kord=8/kord=9
branches reused as-is, per the Do-NOT list.

No R4 touched. No PT/DELZ/REMAP_UV/tracer-CUDA/MOIST_CAPPA/USE_COND
files modified.

## 2. Stage 7 — nq=2 GPU golden fixtures — PASS

New GPU-executed test, `cuda/dyn_core/vertical/
fv3_remap_q_nq2_prod_shape_smoke.cu` (registered in `CMakeLists.txt`,
GPU-only smoke, not a ctest — matches sibling convention). Builds a real
device `q` buffer at C32 production shape (`isd:ied`/`jsd:jed` = 38x38,
km=10), issues the exact Change-1 two-launch sequence, and compares each
tracer against a host oracle (`map_scalar_column<double>`, the same
already-validated reference Prompt 86A used for its host-only scope
test) on a shared, perturbed `pe1`/`pe2` Lagrangian-to-Eulerian grid.

Run on a live A100 (`srun --partition=gpu_a100 --constraint=rome
--gres=gpu:1`):

```
[sphum(kord=8)] BITWISE abs=0.000e+00 rel=0.000e+00 mass_rel=1.202e-16 nan=0 neg=no PASS
[cld_amt(kord=9)] BITWISE abs=0.000e+00 rel=0.000e+00 mass_rel=0.000e+00 nan=0 neg=no PASS
H2D_CALLS=1 D2H_CALLS=1
REMAP_Q_NQ2_PROD_SHAPE_GPU_SMOKE: 0 checks failed
```

Both tracers bitwise-exact vs. the host oracle, 0 NaN/Inf, 0 negative,
mass conserved to round-off, and the Stage-3 hard-fail transfer-count
gate directly confirmed (`H2D_CALLS=1 D2H_CALLS=1`).

## 3. Stages 8-9 — kord regression / sanitizer — PASS

Stage 8: re-ran the Prompt-85 `nq=1`/kord=8 sibling smoke
(`fv3_wp_c_remap_q_prod_shape_smoke`) on the same GPU allocation — all 9
fixtures + all 9 workspace-path bitwise cross-checks PASS, 0/9 failures
(unaffected by this task's changes; PT/DELZ and REMAP_UV smoke tests were
not separately re-run since no file either owns was touched).

Stage 9: `compute-sanitizer --tool memcheck` on the new nq=2 smoke test
(full path used: `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/
cuda/12.5/compute-sanitizer/compute-sanitizer` — not on the
`discover_platform.md`-documented `nsys`/`ncu` PATH addition, found by
`find`) — **0 errors**. Device `q` buffer for this test's C32/10-level
geometry: `stride = 38*38*10 = 14440` elements/tracer-plane;
`14440*2*sizeof(fv_real)=231040 bytes ≈ 0.22 MiB/rank` (R8, this test's
own synthetic 10-level geometry — not a separately-computed real
production C512 estimate).

Both CUDA and Intel-MPI-linked Fortran builds recompiled clean end to
end this task (`build_cuda_r8_mpi`'s GPU-test target,
`build_cuda_r8_intelmpi`'s `fv3_dyn_core_gpu` static library, then the
`GEOSfvdycore_fortran_perf` checkout's `fvdycore` → `FVdycoreCubed_GridComp`
→ `StandAlone_FV3_Dycore.x`, all EXIT=0) — confirming the extended
`fv3_gpu_remap_q` bind(C) signature is interface-compatible with the
Fortran caller (no compiler-caught mismatch).

## 4. Stage 10 — C32 integration — BLOCKED

**Root cause: `kord_tr(1)` is 9, not 8, in this codebase's actual
GEOS-integrated runtime configuration — the (sphum=8, cld_amt=9)
schedule Prompt 86A's Stage 0/1 verified never occurs in practice.**

`fv_arrays.F90:333` sets the GFDL bare-code default `kord_tr = 8`. But
`FV_StateMod.F90:563` (exercised by every GEOS-integrated standalone run,
including this project's own C32/C512 harnesses) unconditionally
overrides this:

```fortran
FV_Atm(1)%flagstruct%kord_tm =  9
FV_Atm(1)%flagstruct%kord_mt =  9
FV_Atm(1)%flagstruct%kord_wz =  9
FV_Atm(1)%flagstruct%kord_tr =  9
```

with no gating condition — this fires on every GEOS FV_State
initialization. `fv_dynamics.F90:941-944`'s per-tracer loop then sets
`kord_tracer(1) = flagstruct%kord_tr = 9` (sphum) and
`kord_tracer(2) = 9` (cld_amt, forced) — **both** production tracers use
kord=9, not the assumed (8, 9) split. This was independently confirmed
via three lines of direct evidence, not inference:

1. `grep FV_StateMod.F90` — the unconditional override above.
2. This task's own C32 run log (`FV3 kord_tr: 9`, printed once per rank
   at init, both the `RUN_CUDA_FV3_REMAPQ_ENABLE=1` and the baseline
   arm).
3. **Prompt 85's own prior C32 run log
   (`p85_stage10_c32_cuda/p85_stage10_final.log:250`) already shows the
   identical `FV3 kord_tr: 9`** — this was true at Prompt 85 time too;
   Prompt 86A's Stage 0/1 read `fv_arrays.F90`'s bare default (8) instead
   of checking whether GEOS's own `FV_StateMod.F90` override chain
   preserves it, and did not cross-check its own cited run log's nearby
   `kord_tr` line even though `ncnst`/`hord_tr` from the same log block
   were used as evidence.

Consequence: the Stage-6 selector's `kord_tr(1)==8` precondition
(`fv_mapz.F90:486`) is never true in this codebase's actual runtime
configuration, so `use_cuda_remap_q` is `F` unconditionally, regardless
of `RUN_CUDA_FV3_REMAPQ_ENABLE` (the env-var override is nested inside
the base-condition check, by design — an override cannot fire a
selector whose own preconditions are false).

**Real C32 A/B evidence** (`p86b_stage10_c32_cuda`/`_fortran`, jobs
58560147/58560148, C32/L10/6-rank, `COLDSTART=1`,
`RUN_CUDA_FV3_REMAPQ_DUMP_EXIT=1`, cuda arm additionally
`RUN_CUDA_FV3_REMAPQ_ENABLE=1`; both linked against the freshly rebuilt
`StandAlone_FV3_Dycore.x`, sha256 `070dcce9...892a42`, identical in both
experiment dirs): both jobs completed (exit 0). `[P85_EXIT_DUMP]`
diagnostic fired 36/36 times (6 ranks x 6 dyn_core calls) in **both**
arms, every line `use_cuda_remap_q=F` — CUDA dispatch count 0, Fortran
fallback count 36, in both arms. (An early red herring in this
investigation — a distinct, earlier-printed FMS tracer-manager counter
also called `ncnst` reads `0` at line 98 of the same log, before FV3's
own `nq`/`ncnst=2` is set later at line 248 — this is unrelated to the
real root cause and is noted only to avoid the same false lead being
re-chased.)

This is **not** a defect in this task's 3 changes — Stages 7-9 directly
prove the two-launch/kord=8+kord=9 mechanism itself is numerically
correct and GPU-executes cleanly. It is a **scope/verification gap in
Prompt 86A's Stage 0/1**: the production kord schedule it verified
(sphum=8, cld_amt=9) does not match this codebase's actual
GEOS-integrated runtime configuration (sphum=9, cld_amt=9 — both
tracers use the same kord). Per the governing operational instruction,
Stage 10 stops here; Stages 11-14 (C512 integration, paired performance,
attribution) were not attempted — they are unreachable while Stage 10
fails.

Binary provenance: the project's usual `"USING ****"` grep pattern was
not found in this build/logging configuration's stdout; the `mpirun`
invocation line in each run's own log (showing each experiment
directory's own local `StandAlone_FV3_Dycore.x` path) was used instead
as equivalent-strength evidence of which binary ran, confirmed identical
sha256 in both experiment directories.

## 5. Stage 15 — Default decision

Not reached as a performance decision (no performance data — blocked
before Stage 11). Structurally, default is unaffected either way:
`use_cuda_remap_q` requires an explicit opt-in env var
(`RUN_CUDA_FV3_REMAPQ_ENABLE`) that no standing job script sets, and its
own base condition is false in this codebase's real configuration
regardless — **Default: FORTRAN**, unchanged.

## 6. Recommendation

This finding is a decision point for a human/orchestrator, not something
resolvable inside this task's 3-change budget:

- **(a) Redesign the kord schedule for the real (9, 9) case.** If both
  production tracers actually use kord=9, the two-launch mechanism could
  in principle collapse to two kord=9 launches (or, further out, a
  design revisit of whether a single-launch/loop shape is now
  justified) — but changing the selector's kord literals and/or the
  launch design is new scope beyond this task's authorized 3 changes and
  needs explicit re-authorization; it also reopens Prompt 86A's Stage 4
  "kord=8 vs kord=9 are separate branches" framing, since the schedule
  actually needed is (9, 9), not (8, 9).
- **(b) Re-verify whether *any* runnable configuration in this project
  produces `kord_tr(1)==8`** (e.g., a bare-GFDL/non-GEOS-wrapped harness,
  if one exists) before doing (a) — if none exists, the `kord_tr(1)==8`
  branch of the selector, and by extension half of Prompt 85's own
  original `nq=1`/kord=8 port, may never dispatch in this codebase's
  GEOS-integrated production path either, independent of this task.
  **Not verified in this task** — flagged as a deferred, scope-relevant
  finding for Prompt 85 as well, not just 86B.
- **(c) Leave as-is.** The code is implemented, GPU-validated, and
  correctly fails closed to Fortran — zero risk, zero regression,
  simply unreachable pending (a)/(b).

No source change is recommended without the human/orchestrator choosing
one of the above.

## 7. Project status

No commit/push performed (not authorized/requested). Uncommitted
working-tree changes this task: `cuda/dyn_core/CMakeLists.txt` (new
target), new file `cuda/dyn_core/vertical/
fv3_remap_q_nq2_prod_shape_smoke.cu`, plus the Stages-1-6 source edits
described in §1 (RemapQArgs/launch_remap_q_ctx, ABI, fv_mapz.F90
selector — files as tracked in `git status` at task start). New
experiment directories `p86b_stage10_c32_cuda`/`_fortran` under
`GEOSfvdycore_fortran_perf/experiments/` (job logs, not source).

## 8. Addendum — interrupted/resumed session (2026-09-25)

This task's connection was lost after the findings above were reached
and documented, but before a final resume-verification pass. A resume
session the same day re-checked, rather than re-derived, both categories
of evidence:

- Stage 7/9 (cheap `srun`, no on-disk log survived the interruption):
  reran the already-built `fv3_wp_c_remap_q_nq2_prod_shape_smoke`
  binary fresh on a live A100 (`warpa003`) — identical output to §2
  (`BITWISE abs=0 rel=0`, both tracers, `H2D_CALLS=1 D2H_CALLS=1`, 0
  checks failed) and a fresh `compute-sanitizer` pass, 0 errors.
- Stage 10 (expensive `sbatch`, jobs 58560147/58560148, already
  completed with valid on-disk logs): not rerun — instead
  re-inspected in place. Both logs still show 36/36 `use_cuda_remap_q=F`
  in both arms, `FV3 kord_tr: 9`, and matching
  `StandAlone_FV3_Dycore.x` sha256 `070dcce9...892a42` in both
  experiment directories — unchanged from the original findings in §4.

No source file was touched during the resume; only this addendum and
the cross-referenced handoff docs (§7/CURRENT_STATE.md) were updated.
Conclusion is unchanged: **BLOCKED at Stage 10**, same root cause.

---

**STAGE 10 BLOCKED — kord_tr(1)==9 in actual GEOS-integrated runtime
configuration, not the assumed 8; CUDA REMAP_Q dispatch count 0/0 in a
real C32 A/B; Stages 11-14 not attempted.**
