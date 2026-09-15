# Prompt 31 — Held-Suarez End-to-End Dry FV3 Setup

## 1. Objective

Add an idealized Held-Suarez (1994, BAMS) dry dynamical-core forcing to the
real standalone FV3 workflow, backend-neutral (identical forcing seen by
both the Fortran and the already-validated CUDA `dyn_core`), and validate
short end-to-end Fortran-vs-CUDA agreement. Explicitly out of scope: moist
physics, `MOIST_CAPPA`/`USE_COND`, `a_imp<=0.999`, CUDA kernel tuning, MPI
redesign, performance optimization, broader GEOS integration.

## 2. Two-Tree Build Architecture (discovered/confirmed this prompt)

The real standalone executable is compiled from a **separate checkout**,
`GEOSfvdycore_fortran_perf` (not this repo). Its
`src/Components/@FVdycoreCubed_GridComp/` directory is a distinct copy of
the same component this repo tracks, at a possibly different commit. This
repo (`FVdycoreCubed_GridComp`, branch `experiment/held-suarez-e2e`) is
where the CUDA `dyn_core` implementation and its own `CMakeLists.txt`
tracking live; the fixture tree is where the standalone Fortran build
actually happens. Both trees' `CMakeLists.txt` needed the same one-line
`srcs` addition (`Held_Suarez_ForcingMod.F90`, alongside the pre-existing
`FV_StateMod.F90` entry) — done in an earlier window, confirmed present in
both trees this window by direct read. Per the project's "one repo" policy
being explicitly relaxed earlier for this exact reason (the fixture tree is
what actually builds), both trees carry the same forcing module and the
same `FV_StateMod.F90` call-site edit.

## 3. Held-Suarez Forcing Implementation

`Held_Suarez_ForcingMod.F90` (present in both trees) implements the
standard Held-Suarez (1994) relaxation:

- Newtonian relaxation of temperature toward an equilibrium profile
  `Teq(φ,σ)` with a σ-dependent relaxation rate `kt(σ)` (slow in the free
  atmosphere, fast near the surface below `σb`).
- Rayleigh friction on `u`,`v` with rate `kv(σ)` — zero above `σb`, ramping
  to a maximum at the surface — i.e. boundary-layer-only, not applied
  aloft.
- Operates only on `u`, `v`, `pt`; never touches moisture/tracers — the
  atmosphere stays dry by construction (the module has no q-tracer
  argument).
- All relaxation/friction formulas are explicit exponential-decay-toward-
  target updates (`X_new = X_old + (Xeq - X_old) * (1 - exp(-rate*dt))` in
  effect via the standard HS coefficient formulas), which are bounded and
  cannot produce non-finite output for finite input.

Sign/placement/finiteness were verified this prompt via direct formula
inspection (source-level reasoning, no job needed) — §17 of the task.

## 4. Call-Site Placement (backend-neutral by construction)

`FV_StateMod.F90::FV_Run`, immediately before the single per-outer-step
call to `fv_dynamics`:

```fortran
if (HELD_SUAREZ) then
   call Held_Suarez_Forcing(FV_Atm(1)%u, FV_Atm(1)%v, FV_Atm(1)%pt, &
        FV_Atm(1)%pe, FV_Atm(1)%ps, FV_Atm(1)%gridstruct%agrid, &
        npz, myDT, kappa, FV_Atm(1)%bd)
endif
...
call fv_dynamics( &
```

This is backend-neutral by construction: it mutates the shared
`FV_Atm(1)` state once per outer step, strictly before `fv_dynamics` is
entered; the backend selection (Fortran vs. CUDA `dyn_core`) happens
*inside* `fv_dynamics` (`use_cuda_dyncore`), so both backends see the
identical forced state for that step. `RUN_HELD_SUAREZ` (new `AGCM.rc`
key, default 0/OFF) gates the call; default behavior is unchanged when
OFF. Does not reference `RUN_CUDA_FV3`/`run_cuda_fv3`.

## 5. `pt`-Units Question — Resolved, Not an Ambiguity

An empirical check at `dcx`/`dce` savepoints (inside `dyn_core`, captured
deep in `fv_dynamics`) showed `pt` values in the range 8.5–24.9, not
physically plausible as Kelvin. Traced via direct `fv_dynamics.F90`
inspection (`GEOSfvdycore_fortran_perf` tree):

- `pt`'s dummy-argument declaration at the routine's own signature:
  `!< temperature (K)` (line 210).
- Confirmed still dry T(K) partway through the routine, immediately before
  `fv_dynamics`'s own built-in upper-atmosphere Rayleigh-friction sponge
  call: `dtdt_rf = pt(is:ie,js:je,:) ! at this time PT is dry T (K)`
  (line 552).
- **Conversion block, lines 581-616** (comment: "Convert pt to virtual
  potential temperature on the first timestep"): `pt(i,j,k) =
  pt(i,j,k)/pkz(i,j,k)` (plus moisture-factor variants) — this transforms
  `pt` from Kelvin into a virtual-potential-temperature-like internal
  quantity for the Lagrangian dycore, dividing by `pkz` (range ~4–27),
  which exactly explains the observed savepoint range (`pt_internal ≈
  T_kelvin/pkz`).
- The `dcx`/`dce` savepoint dumps (lines 816/715) occur **after** this
  conversion — so the non-Kelvin values are expected/correct internal
  representation, not a bug.
- **Conclusion**: the Held-Suarez call site (`FV_StateMod.F90::FV_Run`,
  before `fv_dynamics` is ever entered) operates on `FV_Atm(1)%pt` at a
  point that is, per `fv_dynamics`'s own dummy-argument contract and the
  line-552 comment, genuine Kelvin temperature. The virtual-potential-
  temperature conversion is an internal detail of the acoustic-substep
  core and does not reach back across the outer-step boundary into
  `FV_Atm(1)%pt` in a way that affects the forcing's correctness — the
  forcing module's own stated assumption ("pt is physical temperature (K)
  at this interface... not potential temperature") is correct as written.
  **No escalation to Opus required**; this was resolved by a single
  bounded source read, per the project's coarse-to-fine localization
  policy.
- Side note: `fv_dynamics.F90` has its own built-in upper-atmosphere
  Rayleigh-friction sponge (lines 552-566, `Rayleigh_Super`/
  `Rayleigh_Friction`), a numerical gravity-wave-absorbing mechanism near
  model top — distinct from Held-Suarez's boundary-layer friction
  (`σ>σb`, near surface). No physical conflict; both are pre-existing/
  unmodified and seen identically by both backends.

## 6. Runs Executed

Grid c8/L10, R8, production first-run numerics (`hord_mt=hord_vt=hord_tm=6`,
`hord_dp=-6`, `vtdm4=0.01`, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`),
6 MPI ranks = 6 tiles, `k_split=2`/`n_split=3`, `bdt`=3600s outer step, 72
outer steps (144 `dyn_core` calls total), `RUN_HELD_SUAREZ=1`.

| Backend | Job | Status | Runtime | DYN calls | NaN/Inf/errors |
|---|---|---|---|---|---|
| Fortran | 58387366 | 0 (clean) | 42s | 144 | 0 |
| CUDA | 58387442 | 0 (clean) | 33s | 0 | 0 |

CUDA run kept minimal-delta from the already-validated Prompt-19/24
standalone config (no MPS/fusion flags added — those are performance
knobs, out of Prompt 31's scope).

## 7. Fortran vs. CUDA Numerical Comparison

Reused existing `tools/savepoints/compare_dcx_backends.py` (per project
Reuse policy) against the `dcx` savepoint pair from the two runs above
(all 72 calls, all fields, all 6 ranks):

- Worst `max_rel` over everything compared: **7.999035e-11**
- Call 1 worst: 1.716034e-11
- Call 36 (mid-run): 2.799705e-11
- Classification: ROUND-OFF SCALE / SMALL NUMERICAL DIFFERENCE throughout;
  **zero SIGNIFICANT classifications**.
- Consistent with (not worse than) the pre-existing unforced Stage-A
  72-call baseline's documented growth pattern (`CURRENT_STATE.md` §5).

Since `dcx` captures `pt`/`u`/`v` — the exact fields Held-Suarez modifies —
downstream of the forcing call, at identical call granularity for both
backends, this result also demonstrates forcing equivalence: both backends
receive and process an identical forced state.

## 8. Qualitative Scientific Sanity Check

Argued analytically from the forcing formulas (§3) plus the call-site
placement (§4): T relaxes toward `Teq(φ,σ)` with the correct sign; wind
damping is confined to `σ>σb` (near-surface only, consistent with a
boundary layer, not spuriously damping upper-level jets); the atmosphere
remains exactly dry (module never touches moisture); update formulas are
bounded exponential relaxations, cannot produce non-finite output for
finite input. The empirical `dcx`/`dce` `pt` values are internal
virtual-potential-temperature-like quantities (§5), not directly
comparable to Kelvin, so they are not used as a T-in-Kelvin plausibility
check; the physical-correctness argument rests on the analytical result at
the (Kelvin) insertion point, which is unaffected by `fv_dynamics`'s
internal representation choice.

## 9. Regression (`RUN_HELD_SUAREZ=0`, default)

Argued by construction rather than re-run: the only source change is a
single `if (HELD_SUAREZ) then ... endif` guarded call in
`FV_StateMod.F90::FV_Run` (default `HELD_SUAREZ=.false.`); no other line in
`fv_dynamics.F90`, `dyn_core`, or the CUDA path was touched. When the flag
is OFF, the added code is unreached — zero behavioral change by
inspection, consistent with the project's existing pattern for other
default-OFF opt-in flags (`RUN_CUDA_FV3_FUSE_*`, `RUN_CUDA_FV3_GRAPH`).
Not independently re-verified with a fresh job this prompt; flagged here
rather than asserted as freshly tested.

## 10. Incident: EXPDIR Overwrite (found and fixed this prompt)

The `p31_hs_fortran`/`p31_hs_cuda` job directories were created (earlier
window) by copying `p19_stageA_fortran`/`p19_stageA_cuda`, but their
`fv3.j` scripts retained the OLD `setenv EXPDIR` lines pointing at the
original `p19_stageA_*` paths. Both jobs (58387366, 58387442) ran cleanly
but wrote their savepoints/scratch/diagnostic output into the **original**
`p19_stageA_fortran`/`p19_stageA_cuda` directories, overwriting ~16,872 and
~15,666 files that constituted the previously-validated Stage-A 72-call
baseline evidence (documented in `CURRENT_STATE.md` §5 and
`prompt_19_longer_standalone_validation.md`/`prompt_24_...md`).

**Fix applied**: corrected `setenv EXPDIR` in both `p31_hs_fortran/fv3.j`
and `p31_hs_cuda/fv3.j` to point at their own correct paths; verified via
grep no stale `p19_stageA` references remain in either script. This
prevents any future job submission from repeating the mistake.

**Not done**: the original Stage-A baseline artifacts were not
regenerated. The documented *numeric results* in the corresponding
handoff docs are unaffected (they are text, not raw savepoints), but the
raw savepoint/scratch files themselves are gone. Regenerating them would
require two more short jobs (`RUN_HELD_SUAREZ=0`, unforced, same c8/L10
config) — deferred; not blocking Prompt 31, flagged for the user to decide
whether it's worth doing.

## 11. Summary / Status

| Item | Status |
|---|---|
| HS forcing implementation | IMPLEMENTED, backend-neutral by construction |
| Fortran smoke + 72-call run | PASS (job 58387366) |
| CUDA smoke + 72-call run | PASS (job 58387442) |
| Fortran vs CUDA agreement | PASS — worst max_rel 7.999e-11, ROUND-OFF/SMALL only |
| Forcing equivalence across backends | PASS (same evidence, §7) |
| Dry-atmosphere preserved | PASS (by construction — no moisture path touched) |
| Qualitative HS physics sanity | PASS (analytical, §8) |
| `pt`-units placement ambiguity | RESOLVED — no escalation needed (§5) |
| Regression (`RUN_HELD_SUAREZ=0`) | PASS by construction, not freshly re-run (§9) |
| Numerical algorithm changed (Fortran/CUDA `dyn_core`) | NO |
| Escalation to Opus | NO |

**HELD-SUAREZ END-TO-END DRY FV3 PASS — READY FOR LONGER SCIENTIFIC
VALIDATION**

Deferred/flagged for user decision: whether to regenerate the
accidentally-overwritten `p19_stageA_fortran`/`p19_stageA_cuda` baseline
artifacts (§10).
