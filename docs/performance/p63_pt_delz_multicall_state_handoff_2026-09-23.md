# Prompt 63 — Root-Causing the Post-Call-1 PT/DELZ State-Handoff Divergence

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no Opus escalation — see §7).

## 0. Goal

Prompt 62 found that PTDELZ_EXIT calls 1-6 show a real, monotonically
growing, non-round-off divergence across pt/delz/delp/pe/peln/pk/pkz from
call 2 onward, even though call 1 itself is bitwise/round-off clean. This
prompt's mandate was diagnosis only: determine whether the growth is a
state-handoff/synchronization defect (stale host or device copy, incomplete
sync mask, wrong sync ordering) or something else, apply at most one narrow
state-handoff fix if found, and otherwise report the root cause precisely.
No profiling, no second numerical fix, no touching the already-validated
GMAO boundary port (Prompt 61) or pk/peln pass-through fix (Prompt 62).

## 1. Stage 1 — New PTDELZ_ENTRY diagnostic

Added a `PTDELZ_ENTRY` dump (`fv_mapz.F90`, gated by new env var
`RUN_CUDA_FV3_PTDELZ_DUMP_ENTRY`, tag `[P63_ENTRY_DUMP]`) immediately after
the existing header/sphum dump, i.e. at true subroutine entry, before this
call's pt pretransform, `pe`/`peln` update, or the CUDA/Fortran dispatch
branch have touched anything. Dumps pt/delz/delp/pe/peln/pk at
i=(is+ie)/2, j=(js+je)/2, k=1/5/10, tagged with `mpp_pe()` and the existing
`p60_ncalls` counter for rank/call joining with `[P60_EXIT_DUMP]`.

Source tracing of `Lagrangian_to_Eulerian` (`fv_mapz.F90` ~lines 220-1019)
confirmed this placement is valid for both pt and delz: nothing between
subroutine entry and the single CUDA/Fortran dispatch branch
(`use_cuda_pt_delz_remap`, line ~599-603 — one call handles pt AND delz
together, there is no separate delz dispatch) mutates any of the 7 tracked
fields on the CUDA-active arm. The dispatch call's own closing-update block
(lines 925-1019) is the last write to pe/peln/pk/delp within the
subroutine — nothing in the subsequent tracer remap (`Remap_Q`), NH
w-remap (`Remap_NH`), or UV/momentum remap (`Remap_UV`) touches these 7
fields again.

## 2. Stage 2 — Three-arm isolation control (C32/L10, 6 tiles)

- **A** (`p63_stage2_A_fortran`): `RUN_CUDA_FV3=0` — pure Fortran, no GPU.
- **B** (`p63_stage2_B_cuda`): `RUN_CUDA_FV3=1`, CUDA pt/delz dispatch
  active (`use_cuda_pt_delz_remap=T`).
- **C** (`p63_stage2_C_forcefortran`): `RUN_CUDA_FV3=1` +
  `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1` — identical CUDA dyn_core/halo/sync
  machinery to B for everything else, but Fortran `map_scalar` for pt/delz
  (`use_cuda_pt_delz_remap=F`). The ONLY code-path difference between B
  and C is which implementation computes pt/delz for one remap step.

Jobs 58524852/58524853/58524854 (warpa007/008/009), all COMPLETED exit
0:0. Provenance confirmed per arm: binary sha256 `a131e3e2...` identical
across all three; each job's own `***** USING ****` log line points at its
own arm-local scratch-copied binary (not the stale `install/bin`
fallback); `use_cuda_pt_delz_remap` dump values correct per arm (A=F, B=T,
C=F, all 6 ranks).

**Result: A vs C** — pure round-off (~1e-14 to ~1e-15 max_rel, both
EXIT and ENTRY dumps), flat across all 6 calls, no growth. This empirically
re-derives (and is consistent with, tighter than) Gate C2's own round-off
ceiling for this grid/call-depth.

**Result: B vs C** — real, monotonically growing divergence from call 2
onward, both EXIT and ENTRY dumps (call 6 EXIT max_rel: delp 3.74e-07, delz
1.62e-07, pe 3.32e-07, peln 2.91e-08, pk 9.49e-08, pkz 9.56e-08, pt
1.19e-07 — matching Prompt 62's PTDELZ_EXIT finding almost exactly,
cross-validating the old and new diagnostics).

**Conclusion: isolation CONFIRMED, not refuted.** C (same CUDA dyn_core,
same sync/mask machinery as B, Fortran-fidelity pt/delz only) tracks A at
round-off with zero growth; B alone diverges. The defect's origin remains
specifically the pt/delz CUDA dispatch choice, not the broader CUDA
dyn_core or its sync machinery — this directly rules out root-cause
category **I**.

## 3. Stage 3 — Call-transition table (decisive question)

Using the B-vs-C comparison (the meaningful comparison: two independent,
deterministic runs differing only in the pt/delz implementation, joined at
identical call number/rank/i/j/k):

| dump | call | pt | delz | delp | pe | peln | pk |
|---|---|---|---|---|---|---|---|
| EXIT | 1 | 9.2e-16 | 1.6e-15 | 0.0 | 0.0 | 0.0 | 0.0 |
| ENTRY | 2 | 2.5e-12 | 1.1e-09 | 1.8e-09 | 1.5e-09 | 1.3e-10 | 4.3e-10 |
| EXIT | 2 | 4.5e-08 | 1.9e-07 | 3.5e-07 | 3.4e-07 | 3.0e-08 | 9.8e-08 |
| EXIT | 6 | 1.2e-07 | 1.6e-07 | 3.7e-07 | 3.3e-07 | 2.9e-08 | 9.5e-08 |

**Decisive answer: the first significant (non-round-off) difference
occurs at CALL 2 ENTRY — before call 2's own PT/DELZ dispatch has run.**
Call 1 EXIT is bitwise (delp/pe/peln/pk) or round-off (pt/delz), exactly
matching Prompt 62's Stage 5 result. This means call-2's PT/DELZ inputs
are already numerically different between arms B and C before any of
call 2's own remap arithmetic executes — triggering Stage 4 rather than
Stage 9 (internal-arithmetic-only investigation is not the right frame in
isolation; see §6 for why it is still the correct root-cause classification
once combined with the sync-architecture audit below).

## 4. Stage 4 — Ordered stage-map trace

Because call-1 EXIT sits at the last write to pe/peln/pk/delp within
`Lagrangian_to_Eulerian` (§1), and nothing later in that subroutine call
touches these fields, **100% of the growth from call-1 EXIT to call-2
ENTRY happens outside `Lagrangian_to_Eulerian`** — i.e. during the return
to `fv_dynamics.F90` and the one full intervening `dyn_core`/acoustic-
substep cycle (`fv3_gpu_interface_step_dyn`) before `Lagrangian_to_Eulerian`
is called again. Crucially, **that intervening stage runs identical GPU
code, with identical sync masks, for both arm B and arm C** — the only
difference feeding it is pt/delz's own tiny (round-off-level) numerical
difference at call-1 EXIT. This localizes the growth mechanism to the
acoustic dynamics stage without yet saying whether that stage's *sync
machinery* is defective (Stage 5) or whether it is *legitimately*
amplifying a real, tiny arithmetic difference (Stage 6+).

## 5. Stage 5/7 — Host/device ownership and sync-mask audit

Direct source citation (`cuda/interface/fv3_gpu_dyncore_bridge.F90`, `
fv3_gpu_abi_mod.F90`), not mask inspection alone — cross-checked against
the empirical A-vs-C result in §2:

- `fv3_gpu_interface_pt_delz_remap` (pt/delz dispatch) uses
  `pt_delz_remap_state_mask`: exactly 9 field IDs — PE(10)/PELN(12)/PT(5)/
  DELZ(4)/DELP(6)/PK(11)/PKZ(13)/AK(46)/BK(47). H2D → kernel → D2H → one
  host wait, every call, unconditionally.
- `fv3_gpu_interface_step_dyn` (the acoustic-substep dispatch, called once
  per full dyn_core timestep between `Lagrangian_to_Eulerian` calls) uses
  `mutable_state_mask`: the full contiguous ID range `FV3_FID_U(1)` through
  `FV3_FID_WS(26)`, which is a strict superset containing all 7 tracked
  fields (DELZ=4, PT=5, DELP=6, PE=10, PK=11, PELN=12, PKZ=13). This
  function does an **unconditional full H2D at entry and full D2H at
  exit, every call** — no persistent-residency skip-if-unchanged logic
  exists in this subroutine. Its own header comment states the intent
  explicitly: "H2D: pick up whatever tracer advection and the
  Lagrangian-to-Eulerian remap wrote on the host since the previous
  call." Ordering is enforced via explicit CUDA event dependencies
  (`ev_h2d` gates the step launch; D2H is ordered after the compute stream
  via `event_compute_done`), not convenience/implicit ordering.
- AK/BK are correctly excluded from `mutable_state_mask` (time-invariant
  grid coefficients, not mutable state) — irrelevant to this divergence
  since neither changes during a run.

**No missing field, no incomplete mask, no wrong chronological sync point,
no stale-copy opportunity exists in this call chain for any of the 7
tracked fields** — every call to either dispatch function does a full,
correctly-ordered, unconditional round trip through exactly the masks that
include all 7 fields. This is confirmed independently by the empirical
result in §2: if this sync architecture were defective, arm C (which uses
the *exact same* `fv3_gpu_interface_step_dyn`/`mutable_state_mask` path as
arm B for every field, differing only in which implementation computes
pt/delz) would ALSO diverge from arm A over 6 calls — it does not (flat
round-off, §2). This rules out root-cause categories **A, B, C, D, E**
with both source-level and empirical evidence, not mask inspection alone.

## 6. Stage 6 — Derived-state audit

`fv_mapz.F90`'s unconditional "Update pe / logP&pk / delp" block
(lines 925-1019, the source of the existing PTDELZ_EXIT dump) recomputes
`delp`/`pe`(interior)/`peln`(interior)/`pk`(interior) from `ak`/`bk`/the
surface `pe` and pt/delz's own remap output — it runs identically on the
CUDA and Fortran-fidelity arms and is exactly where Prompt 61/62 already
validated bitwise agreement at call 1. No stale-derived-state pattern:
this block's inputs (surface pe, ak, bk) are unchanged between arms; its
only true dependency on the arm-specific pt/delz computation is through
pt/delz's own round-off-level output difference, which is precisely the
seed identified in §3.

## 7. Root cause and classification

Given §5's airtight sync-architecture audit (categories A-E refuted with
source AND empirical evidence) and §4's localization (100% of the growth
from call-1 EXIT to call-2 ENTRY happens during the shared, sync-clean
acoustic-dynamics stage, seeded only by pt/delz's own tiny round-off
difference), the mechanism is: **the CUDA pt/delz kernel and Fortran's
`map_scalar` are not bit-for-bit identical (a real, ~1e-15-level
difference, most plausibly from differing FMA-contraction/instruction
ordering in the shared log/exp evaluations — both are individually
correct to their own precision, consistent with the same round-off levels
already accepted at Gate C2 and Prompt 61/62's own call-1 gates). That
tiny seed is then propagated through one full acoustic-substep dyn_core
cycle — itself independently Gate-C2-validated as bitwise-correct in
isolation — which is a nonlinear, non-diagonal system (C-grid/D-grid
transport, Riemann solver, pressure-gradient terms) and is not obligated
to keep a small perturbation small. By call 6 the amplification reaches
3-4 orders of magnitude above Gate C2's own established round-off-growth
ceiling for this exact grid/call-depth (§ Prompt 62 Stage 6).**

This is **not** a state-handoff/synchronization defect — no category A-E
applies, and F does not apply either (no asymmetric "later Fortran remap"
exists; the acoustic dynamics code and masks are identical between arms).
It is closest in spirit to category **G** ("remaining bug is internal
arithmetic," not state handoff) — the qualifier is that the internal
non-bitwise-identical arithmetic in question is pt/delz's own kernel vs.
`map_scalar`, and its effect only becomes visible one call later because
of the intervening (correct, previously-validated) dynamics, not because
call-2's own map_scalar/pt_delz arithmetic itself is buggy in isolation.
No category fires that would justify a further **state-handoff** fix —
Prompt 63's change budget (state-handoff/sync fixes only) has no defect to
apply it to. A fix, if pursued, would require making the CUDA pt/delz
kernel bit-for-bit identical to Fortran's `map_scalar` (matching
FMA-contraction and transcendental-function evaluation order) — an
internal **numerical-arithmetic** change outside this prompt's authorized
scope, and of debatable value given Gate C2's own precedent already
accepts round-off (not bitwise) agreement as the correctness bar.

Whether the resulting call-6 magnitude (~3e-7 relative in delp/pe) is
*acceptable* for the eventual C512 A/B correctness gate is a separate,
still-open question this prompt does not resolve — it depends on how that
gate's tolerance is defined at C512/production call-depth, which has not
yet been established for this specific coverage-gap work (unlike Gate C2's
existing C32/C8 precedent, which was derived from the fully-validated
mainline dyn_core, not this in-progress `fv_mapz` port).

## 8. No fix applied

No state-handoff defect was found to fix. Per Prompt 63's decision tree,
this is exactly the STOP condition attached to root-cause classification
G — "new numerical-arithmetic fix requires separate authorization." No
change was made to `fv3_pt_delz_remap.cu`, `fv3_map_scalar_column.hpp`,
the GMAO boundary port, or the pk/peln pass-through fix. Only additive,
env-var-gated diagnostics were added (`PTDELZ_ENTRY` dump, §1) — zero
effect unless `RUN_CUDA_FV3_PTDELZ_DUMP_ENTRY` is set.

## 9. Stages not attempted

Stage 8 (three-way host/device/Fortran runtime value comparison) was not
needed — §5's source-level sync-mask audit combined with §2's full,
independent A-vs-C round-trip empirical result is stronger evidence than a
single-snapshot device-memory dump would have been, and both converge on
the same conclusion. Stages 9 (internal-arithmetic S0-S9 breakdown), 11
(fix), 12 (two-call re-validation), 13 (six-call re-validation), 14 (C32
`dcx`), 15 (regression), 16 (C512 A/B) are not attempted — no fix was
authorized or applied to validate.

## 10. Summary

- New `PTDELZ_ENTRY` diagnostic: **added, additive, zero default effect.**
- Three-arm isolation control: **CONFIRMED** — defect isolated to the
  pt/delz CUDA dispatch specifically, not the broader CUDA dyn_core/sync
  machinery (rules out category I).
- Call-transition table: first significant difference at **CALL 2 ENTRY**
  (before call 2's own dispatch runs).
- Host/device ownership + sync-mask audit: **no defect found** — both
  masks (narrow 9-field pt/delz mask, full `mutable_state_mask`)
  correctly include all 7 tracked fields; both dispatch functions perform
  unconditional, correctly-ordered full round trips every call (rules out
  categories A, B, C, D, E).
- Derived-state audit: no stale-derived-state pattern (rules out
  category F).
- **Root cause: category G** — the growing divergence is legitimate
  nonlinear-dynamics amplification of a real, round-off-level (~1e-15),
  individually-correct arithmetic difference between the CUDA pt/delz
  kernel and Fortran's `map_scalar`, not a state-handoff/synchronization
  defect. Fixing it (if pursued) requires a numerical-arithmetic change
  outside this prompt's authorized scope.
- No fix applied. Stages 9/11-16 not attempted.

**PT/DELZ CALL-TRANSITION ROOT CAUSE IDENTIFIED — REQUIRES ADDITIONAL FIX AUTHORIZATION**

## 11. Update (Prompt 64)

The numerical-arithmetic fix authorized and applied in Prompt 64
(`-fmad=false` scoped to `fv3_pt_delz_remap.cu`) confirmed §7's category-G
diagnosis in more precise form: the round-off-level seed is not one
uniform mechanism. It splits into two distinct sub-causes — (1) FMA
contraction, which affected only `delz`'s pure-multiply arithmetic and IS
now fixed (bitwise at call 1, confirmed in real production data); and (2)
a CUDA-vs-Fortran transcendental-library (`exp`/`log`) rounding difference
affecting `pt`/`pkz`, which is NOT fixable by any compiler flag and
remains exactly as before. Because `pt` (unaffected) remains the dominant
seed, the calls-1-6 growth trajectory in §3/Prompt 62 §7 is numerically
unchanged by the fix. Full detail:
`docs/performance/p64_pt_delz_bitwise_arithmetic_2026-09-23.md`.
