# Prompt 61 — GMAO Top/Bottom Boundary Conditions Ported into CUDA map_scalar

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`

## 0. Goal

Fix Prompt 60's "Bug 2": the CUDA `pt` remap dispatch omitted the GMAO
top/bottom boundary-condition branches that Fortran's `map_scalar` applies
whenever `optional_top`/`optional_bot` are `.true.` — the unconditional GEOS
production default (`FV_StateMod.F90:558-559`). Port those branches into the
CUDA path, then validate through the staged plan (Stages 0-12) before
resuming performance work.

## 1. Fortran contract

`map_scalar` (`fv_mapz.F90`, public) accepts optional logical arguments
`optional_bot`, `optional_top`. When both true (the pt/TE call site's
unconditional production default — delz's own call site does not pass
them), 5 GMAO-specific branches are engaged (`fv_mapz.F90:~1746-1767`):

- B1: entire new-grid level below the old surface pressure
- B2: bottom-boundary extrapolate
- B3: bottom-boundary interpolate
- B4: top-boundary extrapolate
- B5: top-boundary interpolate

Only the true k=1 boundary case in the validated production grid (where
`pe2(i,1)==pe1(i,1)==ptop` exactly) reduces to a trivial `pfac0=0` identity;
the other 4 branches (and B4 in the general case) do not.

## 2. CUDA contract

`map_scalar_column<T>` (`cuda/dyn_core/vertical/fv3_map_scalar_column.hpp`)
extended with two template/runtime booleans, default-preserving:

```
gmao_bot = false, gmao_top = false
```

PT's call site (`fv3_pt_delz_remap.cu`) passes `gmao_bot=true, gmao_top=true`.
delz's call site is unchanged (`false, false`), matching its Fortran call
site (no `optional_bot`/`optional_top` passed). Branches B1-B5 ported as an
exact line-for-line translation of the Fortran conditionals — no
simplification beyond the algebraically-exact k=1 identity case.

## 3. Targeted fixtures (Stages 4-6)

- New Fortran fixture case (`map_scalar-driver/main.F90::run_case_gmao_bc`,
  `case9_gmao_boundary_conditions`): hand-built pe1/pe2 pair, km=kn=6,
  deliberately routes one level through each of B1-B5 plus one ordinary
  interior level (region-by-region derivation in the driver's own comments).
- `pt_delz-driver/main.F90`: pt's call site fixed to pass
  `optional_bot=.true., optional_top=.true.` (previously omitted — this
  driver's own "Fortran reference" did not match real production pt
  behavior until this fix). delz's call site deliberately left unchanged.
- CUDA host/device unit tests (`cuda/dyn_core/tests/vertical/test_map_scalar.cpp`)
  extended to cover all 5 branches against the new Fortran fixture.
- Result: **PASS** — standalone fixtures + compute-sanitizer, 0 errors.

## 4. PTDELZ_EXIT call 1 (Stage 7) — hard STOP/GO gate

### 4.1 Methodology fix (this session)

Reused Prompt 60 arm directories (`p60_stage2_cuda`/`p60_stage2_fortran`)
were discovered to never actually set `RUN_CUDA_FV3_PTDELZ_DUMP_EXIT` or
`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN` via `setenv` — the directory names were
aspirational, not wired. New arms `p61_stage7_cuda`/`p61_stage7_fortran`
were built with explicit `setenv` lines for both variables; the CUDA arm
runs the real CUDA pt/delz dispatch, the Fortran arm sets
`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1` to force the unmodified Fortran
`map_scalar` path for pt/delz even with `RUN_CUDA_FV3=1` elsewhere (Prompt
57 Stage 12 mechanism). Confirmed via dump: `use_cuda_pt_delz_remap=T`
(cuda arm, all 6 ranks) vs `=F` (fortran arm, all 6 ranks).

A first rank-untagged run pair (jobs 58523097/58523098) attempted a naive
sort+diff of `[P60_EXIT_DUMP]` lines across arms and found apparent
mismatches — recognized as methodologically unsound: all 6 MPI ranks (one
cubed-sphere tile each, `FV_NX=1`/`FV_NY=6`) print identical local index
labels (`is=1,ie=32,js=1,je=32`), so a naive text sort cannot guarantee two
lines being diffed came from the same physical tile. Fixed by adding an
`mpp_pe()` rank tag to all 21 `[P60_EXIT_DUMP]` write statements in
`fv_mapz.F90` (scripted transform, verified structurally consistent),
rebuilding (`StandAlone_FV3_Dycore.x` sha256
`c8ce0b785af90fc41543e247663f3fd6f87042ac000cb6002dec0c14e3f81c08`), and
rerunning both arms (jobs 58523158 cuda / 58523159 fortran-control, both
COMPLETED, exit 0).

### 4.2 Result: pt/delz/delp/pe/peln — CLEAN

Rank-paired comparison (`pe=<mpp_pe()>` as join key) of call=1 dumps at
k=1/5/10, C32/L10, 6 ranks:

| field | n | max rel diff |
|---|---|---|
| pt   | 18 | 9.232e-16 |
| delz | 18 | 1.558e-15 |
| delp | 12 | 0.000e+00 (bitwise) |
| pe   | 18 | 0.000e+00 (bitwise) |
| peln | 18 | 0.000e+00 (bitwise) |

pt's k=1/k=10 boundary values (the exact site of Prompt 60's Bug 2 evidence,
rel ≈3.6e-9 to 4.3e-9) are now at round-off (~1e-15/1e-16) — **the GMAO
boundary port fixes Bug 2** for its targeted fields (pt, and delz which was
already unaffected/unchanged).

### 4.3 New finding: `pk` non-round-off divergence at the k=1 boundary edge

| field | n | max rel diff |
|---|---|---|
| pk   | 18 | 7.843e-08 (all 6 ranks, k=1 only; k=5/k=10 clean at ~1e-16) |

Root cause (code inspection, unambiguous — not a semantic-mapping question):
Fortran's closing update (`fv_mapz.F90:951-953`) treats the two vertical
boundary edges of `peln`/`pk` as **architecturally invariant pass-throughs**:

```fortran
pn2(i,   1) = peln(i,   1,j)   ! read back unchanged, not log(pe(i,1))
pn2(i,km+1) = peln(i,km+1,j)
pk2(i,   1) = pk(i,j,   1)     ! read back unchanged, not exp(akap*pn2(i,1))
pk2(i,km+1) = pk(i,j,km+1)
```

Only the *interior* edges (k=2..km) are recomputed via
`pk2(i,k)=exp(akap*pn2(i,k))` (`fv_mapz.F90:955-960`). CUDA's closing-update
loop (`fv3_pt_delz_remap.cu:255-259`) has no such carve-out:

```cpp
for (int k = 0; k <= km; ++k) {
  a.peln.at(i, k, j) = pn2[k];
  a.pk.at(i, j, k) = fv3_exp<fv_real>(a.akap * pn2[k]);
}
```

This unconditionally recomputes `peln`/`pk` at **every** edge, including
both boundaries, via a fresh log/exp round-trip from the kernel's own
Lagrangian `pe_old` — rather than preserving the model's existing invariant
boundary values. The result differs from Fortran's preserved value by
~7.8e-8 relative (~2.9e-7 absolute on a value of magnitude ~3.7) — far above
round-off, confirming a genuine structural mismatch, not FMA-contraction
noise.

**This defect is pre-existing in the pt/delz dispatch's closing-update
loop (introduced when that loop was implemented, Prompt 52C+), not part of
and not introduced by this port's GMAO branches** (Prompt 61 touched only
`map_scalar_column` and PT's call site). It was never previously observed
because no prior diagnostic dumped `pk` directly — Prompt 60's dumps
covered only pt (k=1/5/10) and delz/delp/pkz (k=5). It has not contaminated
any already-closed milestone: the fv_mapz/pt_delz CUDA dispatch as a whole
remains IMPLEMENTED/COMPILED/GPU EXECUTED but not yet NUMERICALLY VALIDATED
(per CURRENT_STATE.md §9's pre-existing "BLOCKED" status for this work),
so this is a newly-surfaced gap within already-unvalidated work, not a
regression of validated work.

### 4.4 Stage 7 gate decision

The Stage 7 gate requires pt/delz/delp/pe/peln/pk/pkz all BITWISE/Gate-C2-
like roundoff at k=1/5/10 for call 1. pt/delz/delp/pe/peln pass cleanly.
`pk` fails at k=1 (interior edges clean). Per the gate's explicit stop
condition ("if k=1 ... remains significantly different: STOP, do not
proceed to multi-call validation"), **STOP is triggered** — Stages 8-12
(six-call PTDELZ_EXIT, C32 `dcx`, regression, C512 A/B, milestone
restoration) are not attempted this prompt.

### 4.5 `pkz` gate leg not actually exercised (diagnostic gap, not a CUDA defect)

Stage 7's gate also names `pkz` as a required-clean field, but zero
`field=pkz` dump lines appeared in either arm's log. Root cause (code
inspection, `fv_mapz.F90:1098-1128`): the Prompt 60 `pkz`/`delp` dump
write statements sit only inside the `#else` (non-`MOIST_CAPPA`) branch of
the `remap_t` pkz-computation block; the sibling `#ifdef MOIST_CAPPA`
branch (`:1101-1107`, the one actually compiled and executed) has no dump
at all. `MOIST_CAPPA`/`USE_COND` are defined **unconditionally** for this
build (`@fvdycore/CMakeLists.txt:130`,
`target_compile_definitions(${this} PRIVATE MOIST_CAPPA USE_COND)` — not
conditioned on any option), so the dump-bearing branch is dead code here
and always was, since Prompt 60 first added it. This is a pre-existing
instrumentation gap in Prompt 60's own diagnostic (not a Prompt 61 or
CUDA-side defect, and not related to the already-refuted
`cappa`/`MOIST_CAPPA` hypothesis for `pt`/`delz` themselves — see §2 of
`p60_pt_delz_dispatch_boundary_divergence_2026-09-23.md`). Net effect:
Stage 7's `pkz` leg was never actually executed/checked in this or the
Prompt 60 diagnostic; only pt/delz/delp/pe/peln/pk have real call-1
evidence. This does not change the Stage 7 STOP decision, which is
already correctly triggered by the measured `pk` divergence (§4.3-4.4).

## 5. Scope decision

Fixing the `pk`/`peln` boundary closing-update defect requires editing
`fv3_pt_delz_remap.cu`'s closing-update loop (lines 255-259) to special-case
the two boundary edges (k=0 and k=km) as pass-throughs, matching Fortran's
`fv_mapz.F90:951-953`. This is a distinct, well-understood, unambiguous fix
— but it falls outside Prompt 61's declared single-feature change budget
(the GMAO `map_scalar` boundary-condition port only). Per project practice
(mirroring how Prompt 60 deferred Bug 2's own fix for exceeding a narrow
budget), this is deferred to a dedicated future prompt rather than folded in
here.

## 6. Summary

- GMAO boundary-condition port (Stages 0-6): **COMPLETE, VALIDATED**
  (standalone fixtures PASS, compute-sanitizer 0 errors).
- Bug 2 (pt boundary divergence, Prompt 60's root cause): **FIXED, VALIDATED**
  at Stage 7 call 1 — pt/delz/delp/pe/peln all round-off/bitwise.
- New, separate defect discovered: CUDA pt/delz closing-update loop
  incorrectly recomputes `pk`/`peln` at the two vertical boundary edges
  instead of preserving them as Fortran does — root-caused, not fixed,
  deferred to a dedicated future prompt.
- `pkz` gate leg not actually exercised — Prompt 60's own diagnostic dump
  for `pkz` is dead code under this build's unconditional `MOIST_CAPPA`
  define; a pre-existing instrumentation gap, not a new defect (§4.5).
- Stages 8-12 not attempted (Stage 7 STOP condition triggered by `pk`).

## 7. Update (Prompt 62)

The `pk`/`peln` boundary closing-update defect deferred in §5 has been
**fixed and validated at PTDELZ_EXIT call 1** (bitwise, was 7.843e-08 at
k=1 — now exactly 0.0) and the `pkz` instrumentation gap noted in §4.5/§6
has been **fixed** (now observable in the production `MOIST_CAPPA`
build). However, extending the same comparison to all six PTDELZ_EXIT
calls (Prompt 61's own Stage 8, never previously exercised because this
Stage 7 STOP blocked it) found a **separate, previously-unmeasured
growing divergence** across pt/delz/delp/pe/peln/pk/pkz jointly, from
call 2 onward — smaller than but not eliminating the pre-Prompt-61 "Bug
2" magnitude. This triggers a new STOP at Prompt 62's own Stage 6. Full
detail: `p62_pk_peln_boundary_semantics_2026-09-23.md`.
