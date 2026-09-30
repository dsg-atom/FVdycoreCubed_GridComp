# Prompt 60 — PT/DELZ Bug 2: First-Expression Localization at the Dispatch Boundary

_2026-09-23. Branch `optimization/cuda-fv3-performance`. Model: Sonnet — no
Opus escalation used. The root cause was resolved by direct source
comparison plus algebraic proof (`pfac0=0` at k=1) and numeric confirmation,
with no genuine Fortran/CUDA semantic ambiguity at any point. Continues
Prompt 57 (`p57_pt_delz_kernel_first_divergence_2026-09-22.md`), Prompt 58
(`p58_pt_delz_call2_state_divergence_2026-09-23.md`), and Prompt 59
(`p59_pt_delz_pe_peln_axis_order_2026-09-23.md`)._

## 1. Mandate

Find the first numerical difference produced directly by the CUDA pt/delz
dispatch itself, at `PTDELZ_EXIT` (immediately after Fortran's pt/delz remap
completes, before tracer remap / NH w-remap / UV-momentum remap / Riemann
solver). Do not use `dcx` as the discriminator. Apply at most one numerical
fix; do not broaden supported remap modes; do not rewrite the kernel; escalate
to Opus only if a genuine Fortran/CUDA semantic ambiguity is found.

## 2. Stages 0-2 (recap, established before this doc)

- Stage 0: `pt`/`delz` (not `dcx`) are the correct discriminating fields —
  `dcx` is a diffused, several-stages-downstream consequence.
- Stage 1: added a `PTDELZ_EXIT` diagnostic (`[P60_EXIT_DUMP]`) printing
  `pt`/`delz`/`delp`/`pkz`/sphum at a fixed single column, immediately after
  the pt/delz remap call site, both backends.
- Stage 2: at mid-column (k=5), `pt`/`delz` are round-off-identical at call 1
  and diverge non-round-off starting call 2 — this was the original basis for
  believing "call 1 is clean." (Refuted at the boundary — see Stage 3 below.)
- A `cappa`/`MOIST_CAPPA` hypothesis was checked and refuted (this run has
  `nwat` water species active but the CUDA pt/delz path's `rrg`/`akap`
  constants are confirmed non-`cappa`-dependent for this configuration).
- A pk-boundary-recompute discrepancy in the CUDA kernel's closing-update
  loop (CUDA recomputes `pk` at k=1/k=km via `exp(akap*pn2[boundary])`;
  Fortran's own closing block, `fv_mapz.F90:931-948`, preserves the OLD
  `pk`/`peln` at those same boundaries) was identified but reasoned away
  without a fresh test: Prompt 58 already proved that removing the ENTIRE
  `pe`/`peln`/`pk` closing-update block (a strict superset of this
  discrepancy) had zero measurable effect on `pt`/`w`/`omga`. Since the
  superset change was inert for those fields, this narrower sub-issue cannot
  be their cause either.

## 3. Stage 3 — first bad expression, proven

### 3.1 The asymmetry

`fv_mapz.F90`'s pt/TE remap call site (`:622-633`) passes
`optional_top=gmao_top_bc, optional_bot=gmao_bot_bc` into `map_scalar`.
delz's own remap call site (`:712-716`) does **not** pass either argument.
So this mechanism is pt/TE-specific; delz's remap algorithm itself should be
Fortran/CUDA-equivalent regardless of these flags.

Confirmed at runtime (this run's own scratch log): `FV3 gmao_top_bc: T`,
`FV3 gmao_bot_bc: T`.

Confirmed these are **unconditional GEOS production defaults**, not an HS-
specific override: `FV_StateMod.F90:558-559`,

```fortran
FV_Atm(1)%flagstruct%gmao_top_bc = .true.
FV_Atm(1)%flagstruct%gmao_bot_bc = .true.
```

sit directly among the "resolution dependent defaults for FV3 in GEOS"
(`:521` comment header) alongside `kord_tm=9`/`kord_mt=9`/`remap_option=0`
(`:556-563`) — set once, unconditionally, for every GEOS FV3 run using this
`FV_StateMod.F90`, before any namelist override is applied (the header notes
these CAN be overridden via `fv_core_nml`, but no override to `.false.` was
found or is active for this run, consistent with the observed `T`/`T`).
`fv_arrays.F90` declares both flags' hardcoded default as `.false.` — i.e.
GEOS's `FV_StateMod.F90` deliberately flips FV3-core's own default on for
every standard GEOS run. This is a general production code path, not a
narrow/experimental one.

### 3.2 What `map_scalar` does with `gmao_top_bc=.true.`

`map_scalar` (`fv_mapz.F90`, subroutine at `:1653`) branches at `:1734-1755`
into 5 GMAO-specific cases instead of the standard PPM-based remap:

- `gmao_bot .and. pe2(i,k)>=pe1(i,km+1)` — entire grid below old ps
- `gmao_bot .and. LM1==km .and. LP0==km` — extrapolate linearly below last level
- `gmao_bot .and. LP0==km` — interpolate linearly between levels (bottom)
- `gmao_top .and. LM1==1 .and. LP0==1` — extrapolate linearly above first level
- `gmao_top .and. LM1==1` — interpolate linearly between levels (top)

At k=1 (top target level) for this grid, `pe2(i,1)=ptop` (fixed by
construction, `:913`) and `pe1(i,1)=ptop` (fixed by construction on the old
grid too) — so `LM1=LP0=1` and `pfac0=(pe2(i,1)-pe1(i,1))/(...)=0/(...)=0`
**exactly, algebraically, unconditionally**. The branch taken at k=1 reduces
to `q2(i,1)=q1(i,j,1)` — an exact identity pass-through of the OLD pt value,
not a remap at all.

`map_scalar_column` (`fv3_map_scalar_column.hpp`, the CUDA template invoked
for pt) has **no parameter and no branch** corresponding to
`optional_top`/`optional_bot`. It always applies the standard PPM-based
remap formula at every level, including k=1/k=km — which does not collapse
to an identity in general, so it produces a different value than Fortran's
GMAO-BC identity pass-through.

### 3.3 Numeric proof — divergence present from call 1, at the boundary

Added a `PTDELZ_EXIT` diagnostic printing `pt` at k=1 and k=10 (in addition
to the existing k=5 mid-column print), rebuilt
(`build_cuda_r8_intelmpi`/relink, sha256
`c6c793957d85525c37ec28a5aa65a1db87535c708de96748c2d5922a667180fd`), reran
Stage 2's driver (job 58521894, `COMPLETED`, exit 0, elapsed 22s).

Call 1, k=1 (top boundary), one tile/rank sample:

| Backend | value |
|---|---|
| CUDA | 2.2094192155389615E+02 |
| Fortran | 2.2094192076404045E+02 |

abs diff ≈7.9e-7, rel ≈3.57e-9 — non-round-off (Gate C2 precedent is ~1e-10 to
1e-11 relative). A second rank sample (~223.33 cluster) shows the same
pattern (CUDA 2.2332889490692727E+02 vs Fortran 2.2332891721080586E+02).

Call 1, k=10 (bottom boundary, `km=10` for this L10 test):

| Backend | value |
|---|---|
| CUDA | 3.0786495870283500E+02 |
| Fortran | 3.0786495738787272E+02 |

abs diff ≈1.31e-6, rel ≈4.27e-9 — non-round-off, same call.

**This overturns the Stage 2 "call 1 is clean" conclusion.** Call 1 is only
clean at mid-column (k=5, where neither GMAO branch applies — the standard
remap is exact there for both backends at call 1, since Held-Suarez
coldstart initializes exactly on the `ak`/`bk` grid). The true first
divergence is at the k=1/k=10 boundaries, present from call 1, and it
diffuses into the interior (k=5) only from call 2 onward as the acoustic/
vertical substeps couple boundary error into the rest of the column. The
previously-reported call-2-onward mid-column growth (Prompt 57/58/59,
`pt` max_rel 1.28e-6→6.19e-6) is a secondary, propagated effect of this
boundary-level error, not an independent interior bug.

`delz`'s own remap call site does not use `gmao_top_bc`/`gmao_bot_bc` at all
(§3.1) — its observed divergence (Stage 2) is therefore attributed to the
same root cause only indirectly, via the shared thermodynamic/dynamical
coupling between `pt` and `delz`/`delp`/`pkz` in the joint post-transform
(`fv3_pt_delz_remap.cu:220-231`) and across timesteps, not a second,
independent delz-side defect.

## 4. Root-cause classification

**Category I — missing boundary-condition branch.** The CUDA `pt` remap
dispatch (`map_scalar_column` call site in `fv3_pt_delz_remap.cu`) omits the
GMAO top/bottom boundary-condition special-casing that Fortran's `map_scalar`
applies whenever `gmao_top_bc`/`gmao_bot_bc` are true — which is the
unconditional GEOS production default (§3.1), not a config-specific or
experimental setting. Proven, not inferred: exact algebraic identity at k=1
(§3.2) plus direct numeric confirmation of a non-round-off divergence at
k=1/k=10 from call 1 (§3.3). This is a distinct defect from Bug 1 (fixed,
Prompt 57) and from the pe/peln OOB write (fixed, Prompt 59) — neither of
those touches `map_scalar`'s BC branching at all.

## 5. Fix decision — NOT applied this prompt

A full, correct fix requires porting **5 conditional branches** from
`map_scalar` (§3.2) into the CUDA pt dispatch: one is a provably-exact,
trivial identity (k=1 top case, `pfac0=0` always) but the other four
(3 bottom-boundary variants plus the "entire grid below old ps" case) are
*not* uniformly reducible to a simple closed form for an arbitrary column —
they depend on runtime comparisons between `pe2`/`pe1` at the bottom edge
that are not identically zero/one in general, unlike the top case. Porting
all 5 branches faithfully, with correct behavior across arbitrary grid
geometries (not just this L10 test column), is a materially-sized numerical
port — not a single narrow fix, and not achievable without touching the
kernel's per-column boundary logic in a way that risks a partial or
incorrect port if done in the same pass as this localization work.

Per Prompt 60's explicit budget ("apply at most one fix... do not rewrite the
kernel... STOP if more than one correction seems necessary"), and given a
partial fix (e.g. top-only) would leave the bottom-boundary divergence
unresolved and Bug 2 only partially closed, **no fix was applied this
prompt**. Zero numerical/kernel changes were made; only diagnostic prints
(k=1/k=10 `pt` dump lines) were added, gated behind the existing
`p60_dump_active` diagnostic flag, zero effect by default.

## 6. Change budget accounting

Zero numerical fixes applied (within the prescribed "at most one" budget —
none was spent, since a correct fix requires a multi-branch port exceeding
"one narrow fix"). One diagnostic addition: two `write` statements in
`fv_mapz.F90` printing `pt` at k=1/k=10, gated behind the pre-existing
`p60_dump_active` flag. No CUDA source was modified. No Riemann-solver
change, no registry change, no R4 work, no revisiting of pe/peln axis order
or registry metadata (Prompt 56/59 findings stand unchanged), no refactor.

## 7. Milestone

- Bug 1 (double-transform): unaffected, remains FIXED/VERIFIED (Prompt 57).
- pe/peln axis-order OOB write: unaffected, remains FIXED/VERIFIED
  (Prompt 59), refuted as Bug 2's cause (unchanged from Prompt 59).
- Bug 2: **ROOT-CAUSED** (not merely localized) to a missing GMAO top/bottom
  boundary-condition branch in the CUDA pt remap dispatch, proven via exact
  algebraic identity (k=1) plus direct numeric evidence at both boundaries
  from call 1 onward. **NOT FIXED** — the fix is a genuine 5-branch numerical
  port, judged too large to apply as this prompt's single authorized change
  without risking an incomplete or incorrect port. Requires a dedicated
  future prompt authorized specifically to port `map_scalar`'s GMAO
  boundary-condition logic into the CUDA pt dispatch, with its own
  Stage 10-14 validation (PTDELZ_EXIT boundary revalidation at k=1 AND k=10,
  `dcx` A/B, regression, C512 A/B).

## 8. Update (Prompt 61) — Bug 2 FIXED; §2's pk-boundary discrepancy CONFIRMED, still open

Prompt 61 ported the 5 GMAO branches into `map_scalar_column` and PT's call
site. Rank-paired PTDELZ_EXIT call-1 comparison (C32/L10, all 6 tiles)
confirms **Bug 2 is FIXED**: `pt`/`delz`/`delp`/`pe`/`peln` are all now
round-off or bitwise (max rel ~1.6e-15), down from ~3.6e-9 to ~4.3e-9 at the
k=1/k=10 boundaries. See `p61_map_scalar_gmao_boundary_conditions_2026-09-23.md`
§4.2 for the full table.

§2's `pk`-boundary-recompute discrepancy (identified but reasoned-inert-by-
superset-test here, never directly measured) is now **empirically
confirmed real** by a direct `pk` dump: ~7.8e-8 relative divergence at the
k=1 edge specifically (interior edges k=5/10 remain round-off-clean). It
remains a distinct, pre-existing defect in the CUDA closing-update loop
(`fv3_pt_delz_remap.cu:255-259`, unconditionally recomputes `peln`/`pk` at
both boundary edges where Fortran preserves them as invariants,
`fv_mapz.F90:951-953`) — not part of, and not introduced by, the Prompt 61
GMAO port. Still not fixed; still judged inert for pt/w/omga (Prompt 58's
superset test); triggers Prompt 61's own Stage 7 STOP condition on the
letter of its pk-cleanliness requirement. See
`p61_map_scalar_gmao_boundary_conditions_2026-09-23.md` §4.3-4.4/§5 for the
full analysis and scope decision.
