# Prompt 87 — Adapt REMAP_Q CUDA to Actual Production kord=(9,9)

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`
Status: **STOPPED at Stage 8 per explicit task instruction — new blocking gate
identified empirically, not inferred.** No commit/push performed.

> **Update (Prompt 88A, 2026-09-25)**: the `fill=.true.` open question left in
> §9 has been resolved by a dedicated source-mapping task. Finding: `fill` is
> **universal** across every runnable config in this project (no
> `fill=.false.` found anywhere); its numerical effect (`fillz`, a bounded
> per-column/per-tracer mass-conserving negative-value post-fix) is cleanly
> separable from REMAP_Q's core remap math, requires no new data residency,
> and preserves the existing 1H2D+1D2H transfer architecture. Verdict:
> `REMAP_Q FILL=TRUE SCOPE CLEAN — PROCEED TO CUDA EXTENSION`, port size
> **SMALL**. See `p88a_remap_q_fill_semantics_2026-09-25.md` for the full
> 8-stage analysis and the new host-only `fillz` golden-reference test. No
> CUDA implementation was made by Prompt 88A itself — this remains a decision
> point pending explicit authorization to implement.

## 1. Background

Prompt 86B closed with CUDA REMAP_Q dispatch 0/0 in a real C32 A/B, root-caused
to `FV_StateMod.F90:563` unconditionally forcing `flagstruct%kord_tr=9` for
every GEOS `FV_State`, so real production's tracer kord schedule is
`kord_tracer=(9,9)`, not the `(8,9)` Prompts 85/86A assumed from the GFDL
bare-code default of `8`. This left an open decision: redesign the CUDA
REMAP_Q kord schedule for the real `(9,9)` case (new scope, needs
authorization), or leave the P85/86A/86B kord=(8,9) implementation as
zero-risk unreachable infrastructure.

Prompt 87 authorized the redesign: adapt the existing CUDA REMAP_Q
implementation (already proven bitwise-exact for the sphum-kord=9 numerics
via the shared `scalar_profile_column`/`map_scalar_column` path used by
REMAP_UV) to dispatch on `kord_tr(1)==9` instead of `==8`, re-validate, and
attempt real C32 integration.

## 2. Scope Discipline (unchanged from Prompts 84B-86B)

Not touched: REMAP_Q/scalar_profile numerics, FV_StateMod forcing/config,
`kord_tr` itself, REMAP_UV, PT/DELZ, tracer CUDA (`fv_tp_2d`/`tracer_2d_1L`,
closed per P83 STOP), R4, MOIST_CAPPA/USE_COND.

## 3. Stages 1-4 — CUDA/C++ and Fortran-selector kord correction

All kord1==8 gates/dispatch literals were audited and corrected to kord1==9,
across both repositories:

| File | Change |
|---|---|
| `GEOSfvdycore_fortran_perf/.../fv_mapz.F90` | Selector: `use_cuda_remap_q = kord_tr(1) == 8 ...` → `== 9`; rewrote adjoining comment block to state the real Fortran value is `kord_tr(1)==9` (not `8`), citing `FV_StateMod.F90:563`; documented the newly-found `FV_StateMod.F90:568` `fill=.true.` forcing as a second, independent blocker pending its own authorization. |
| `cuda/interface/fv3_gpu_dyncore_bridge.F90` | **Critical fix**: the actual ABI call site inside `fv3_gpu_interface_remap_q` hardcoded the literal `8_c_int32_t` for kord1 — a distinct bug from the doc-level staleness, not caught by Prompt 86B's edits. Changed to `9_c_int32_t`. Without this fix, every CUDA-side kord=9 gate correction would have been silently defeated (C++ side would reject kord1=8 as unsupported — a different, misleading failure mode). Also updated two doc comments (header, "literal 2/8/9" → "2/9/9"). |
| `cuda/interface/fv3_gpu_abi_mod.F90` | Doc comment: "sphum kord=8" → "kord=9"; "2/8/9" → "2/9/9". |
| `cuda/dyn_core/vertical/fv3_remap_q.cu` | Functional gate `if (nq != 2 \|\| kord1 != 9 \|\| kord2 != 9)` (already correct from the prior session; re-verified). |
| `cuda/dyn_core/vertical/fv3_remap_q.cuh` | Header doc already correct, including explicit documentation of the `fill=.true.` second blocker. |
| `cuda/dyn_core/include/fv3_gpu_api.h` | `fv3_gpu_remap_q` doc comment already correct (kord=[9,9], cites `FV_StateMod.F90:563`). |
| `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` | Functional gate `if(nq!=2\|\|kord1!=9\|\|kord2!=9)` already correct; transfer architecture (1 H2D/1 D2H) unaffected. |
| `cuda/dyn_core/CMakeLists.txt` | Fixed a stale comment above the nq=2 smoke-test target: "kord=8, cld_amt kord=9" → "kord=9, cld_amt kord=9", credited to Prompt 87. |

Numerical semantics changed: **NO** — only the dispatch selector threshold and
literal changed; the kord=9 numerics themselves (`scalar_profile_column`) were
already implemented and independently validated via the REMAP_UV/map_scalar
kord=9 path before this prompt.

## 4. Stage 5 — Fixture / GPU smoke re-validation (kord=9/9)

- `test_remap_q_nq2_scope.cpp` (host oracle, kord=9 sphum / kord=9 cld_amt):
  **PASS**.
- `fv3_remap_q_nq2_prod_shape_smoke.cu` (GPU smoke, production-shaped
  nq=2/kord=(9,9)): **GPU EXECUTED, NUMERICALLY VALIDATED** — bitwise-exact
  vs. host oracle for both tracers, mass conserved to round-off, 0 NaN/Inf,
  `H2D_CALLS=1 D2H_CALLS=1` confirmed (transfer architecture preserved).

## 5. Stage 6 — Shared-helper regression

- REMAP_UV kord=9 path: **PASS** (unaffected, not touched).
- `map_scalar` kord=9 path: **PASS** (unaffected, not touched).
- REMAP_Q kord=8/nq=1 regression (`test_remap_q_tracer.cpp`, P85's original
  fixture set, retained as non-production coverage): **PASS**, correctly
  still requires `kord != 8` to fail — left unchanged.

## 6. Stage 7 — compute-sanitizer

Real GPU run (Slurm `gpu_a100`) of `fv3_remap_q_nq2_prod_shape_smoke` under
`compute-sanitizer`: **0 invalid reads/writes reported.**

## 7. Stage 8 — Real C32 integration (STOP point)

### 7.1 Authoritative binary rebuild + provenance

Rebuilding required touching three separate build trees (this repo's OpenMPI
dev build `build_cuda_r8`, this repo's Intel-MPI production build
`build_cuda_r8_intelmpi` — the one the Fortran executable actually links via
its `FV3_CUDA_LIBRARY_DIR` CMakeCache entry — and the `GEOSfvdycore_fortran_perf`
Fortran build/link step itself). A first Fortran rebuild silently linked
against a **stale** `build_cuda_r8_intelmpi` (caught via `strings` showing the
old "kord1=8" error text before the CUDA archive was rebuilt); fixed by
rebuilding `fv3_dyn_core_gpu`/`fv3_wp_c_vertical`/`fv3_production_dispatch` in
`build_cuda_r8_intelmpi` under the Intel-MPI module stack, then re-linking.

Final authoritative binary, confirmed fresh via provenance check:

```
/gpfsm/dnb06/projects/p295/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
mtime=2026-09-25 13:19:48 -0400
sha256=c43845c76ea9c6845bd04816a28de9f664696deb9f4ed4f1f14b7a6e77d1b84b
symbols present: fv3_gpu_remap_q, fv3_gpu_dyncore_bridge_mp_fv3_gpu_interface_remap_q_
embedded string: "REMAP_Q requires nq=2, kord1=9 (sphum), kord2=9 (cld_amt) ..."
MPI linkage: libmpi.so.12 / libmpifort.so.12 / libmpicxx.so.12 -> Intel oneAPI 2021.13
```

### 7.2 C32 A/B run

Two fresh experiment directories were created (not reusing P86B's, whose
per-experiment binary copies were confirmed stale, sha256
`070dcce9...892a42`), each seeded with the fresh sha256 `c43845c7...b84b`
binary:

- `experiments/p87_stage8_c32_cuda/` — `RUN_CUDA_FV3=1`,
  `RUN_CUDA_FV3_REMAPQ_ENABLE=1` (opt-in arm, "B")
- `experiments/p87_stage8_c32_fortran/` — `RUN_CUDA_FV3=1`, REMAPQ_ENABLE unset
  (arm "A")

Both jobs (Slurm 58562204/58562205, `gpu_a100`) completed cleanly, no
errors/aborts, in ~21s each.

### 7.3 Result

```
FV3 kord_tr:   9        <- confirms the P86B-diagnosed real production value
FV3 fill: T              <- NEW: confirms the second, independently-flagged blocker
FV3 fill_dp: F
FV3 fill_wz: F

[P85_EXIT_DUMP] ... use_cuda_remap_q=F ...   (all 36/36 lines, BOTH arms)
```

**CUDA REMAP_Q dispatch: 0/36 in arm A (expected, opt-in flag unset). CUDA
REMAP_Q dispatch: 0/36 in arm B too (opt-in flag set) — dispatch did NOT
occur.** Unexpected fallback: 36/36 in arm B.

### 7.4 STOP — exact remaining gate identified

Per Stage 8's explicit instruction ("If CUDA still does not dispatch: STOP
and identify exact remaining gate"), this is a STOP, not a re-attempt.

The selector in `fv_mapz.F90` is:

```fortran
use_cuda_remap_q = kord_tr(1) == 9 .and. nq == 2 .and. .not. fill .and. &
     kind(q) == kind(1.0d0) .and. fv3_gpu_backend_enabled()
```

The kord1==9 term (this prompt's fix) is now satisfied — confirmed
empirically by `FV3 kord_tr: 9` in the log. However `FV_StateMod.F90:568`
**unconditionally** sets `flagstruct%fill = .true.` for every GEOS `FV_State`,
the same unconditional-override pattern already found for `kord_tr` at line
563. Confirmed empirically by `FV3 fill: T` in both A and B logs. This makes
`.not. fill` evaluate to `.FALSE.` unconditionally, independent of kord,
`RUN_CUDA_FV3_REMAPQ_ENABLE`, or any other Prompt-87 change — dispatch is
blocked at this second, independent gate.

**Exact remaining gate: `FV_StateMod.F90:568`'s unconditional
`flagstruct%fill = .true.`, enforced by the CUDA REMAP_Q selector's
`.not. fill` term.** This is architecturally identical in kind to the P86B
`kord_tr` finding — a GEOS-level forced configuration value, not a CUDA
defect, not something Prompt 87 was authorized to change (`FV_StateMod`
forcing/configuration is explicitly out of scope), and not fixable by any
change confined to `cuda/` or `fv_mapz.F90`'s selector logic alone.

Per Stage 8, Stages 9-12 (C512, paired performance, attribution, final
decision) are **not attempted** — correctly gated on Stage 8, which did not
pass.

## 8. Numerical Semantics

No scientific/numerical algorithm was changed in this prompt. All changes
were dispatch-selector thresholds/literals (kord1 8→9) plus doc-comment
corrections. Numerical semantics changed: **NO**.

## 9. Decision Point for Next Task

This is now, again, a decision point rather than an implementation task, with
the same two options CURRENT_STATE.md flagged for the `kord_tr` finding,
extended to now cover `fill` as well:

(a) Redesign CUDA REMAP_Q to support `fill=.true.` (filling/positivity-limiter
    fixer logic) — new scope, needs explicit authorization, likely
    nontrivial (the CUDA kernel path currently assumes the no-fill branch).
(b) Leave the kord=(9,9) CUDA REMAP_Q implementation as further validated,
    GPU-executed, but still-unreachable zero-risk infrastructure — default
    Fortran dispatch is completely unaffected either way.
(c) As a prerequisite to (a), verify whether *any* runnable GEOS
    configuration in this project ever produces `fill=.false.` — unverified;
    if `fill=.true.` is forced everywhere the way `kord_tr=9` is, (a) may be
    the only way CUDA REMAP_Q ever dispatches in production, independent of
    kord.

## 10. Files Changed This Prompt (uncommitted)

- `GEOSfvdycore_fortran_perf/.../fv_mapz.F90` (separate repo)
- `FVdycoreCubed_GridComp/cuda/interface/fv3_gpu_dyncore_bridge.F90`
- `FVdycoreCubed_GridComp/cuda/interface/fv3_gpu_abi_mod.F90`
- `FVdycoreCubed_GridComp/cuda/dyn_core/CMakeLists.txt` (comment only)

No commit/push performed in either repository, per standing instructions.

---

**STAGE 8 STOP — `kord_tr(1)==9` fix confirmed correct and necessary
(empirically true in a real C32 run), but insufficient: a second,
independent unconditional GEOS forcing, `flagstruct%fill=.true.`
(`FV_StateMod.F90:568`), rejects CUDA REMAP_Q dispatch via the
selector's `.not. fill` term regardless of kord. CUDA REMAP_Q dispatch
count 0/36 in a real C32 A/B; Stages 9-12 not attempted.**

**Superseded**: Prompt 88A mapped `fill`'s semantics (universal
`fill=.true.`, port size SMALL); Prompt 88B implemented CUDA `fillz`
support and removed the `.not. fill` gate in favor of opt-in
`RUN_CUDA_FV3_REMAPQ_ENABLE`. Real C32 A/B now dispatches 36/36 with 0
fallback; C512 A/B is PERFORMANCE VALIDATED POSITIVE (REMAPPING
-11.98%, TOTAL -1.51%), verdict KEEP AS OPT-IN INFRASTRUCTURE, default
remains Fortran. See `p88b_remap_q_fill_true_cuda_2026-09-25.md`.
