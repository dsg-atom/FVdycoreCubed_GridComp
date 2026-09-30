# Prompt 62 — pk/peln Boundary Pass-Through Semantics Fix

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no Opus escalation — the Fortran/CUDA mapping was unambiguous).

## 0. Goal

Fix Prompt 61's residual defect: CUDA's pt/delz closing-update loop
unconditionally recomputed `pk` at the two vertical boundary edges via a
fresh `exp(akap*log(...))` round-trip, where Fortran preserves the old
`pk`/`peln` values there as exact pass-throughs. Then advance the
correctness ladder (PTDELZ_EXIT call 1 -> six-call -> C32 `dcx` -> C512).

## 1. Stage 0 — Fortran pk/peln contract (re-derived from source)

`fv_mapz.F90:926-966`, unconditional (no hydrostatic/MOIST_CAPPA/remap-mode
gating — runs before any such branch, for every configuration):

| Field | k region | Fortran behavior |
|---|---|---|
| peln | top (k=1) | pass-through: `pn2(i,1) = peln(i,1,j)` (old value, read back unchanged) |
| peln | interior (k=2..km) | recompute: `pn2(i,k) = log(pe2(i,k))` |
| peln | bottom (k=km+1) | pass-through: `pn2(i,km+1) = peln(i,km+1,j)` |
| pk | top (k=1) | pass-through: `pk2(i,1) = pk(i,j,1)` (old value, read back unchanged) |
| pk | interior (k=2..km) | recompute: `pk2(i,k) = exp(akap*pn2(i,k))` |
| pk | bottom (k=km+1) | pass-through: `pk2(i,km+1) = pk(i,j,km+1)` |

## 2. Stage 1 — CUDA closing-update audit

`fv3_pt_delz_remap.cu`'s pre-fix closing loop (0-indexed, k=0..km):

- `peln`: `pn2[0]`/`pn2[km]` were already set to `pn1[0]`/`pn1[km]` (the OLD
  peln, read earlier in the kernel) rather than a fresh `log()` —
  **already matched** Fortran's pass-through. This is why Prompt 61 found
  peln bitwise-clean.
- `pk`: `a.pk.at(i,j,k) = fv3_exp<fv_real>(a.akap * pn2[k])` for **every**
  `k=0..km`, including the boundaries — recomputed unconditionally, not
  preserved. **Mismatch identified**: the boundary `pk` should instead be
  the OLD `pk` value, not a fresh `exp()` of the (already-correct)
  boundary `pn2`.

## 3. Stage 2 — Fix applied

File: `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`.

1. Added `pk_old_top`/`pk_old_bot` reads (`a.pk.at(i,j,0)`, `a.pk.at(i,j,km)`)
   immediately after the existing `pe_old`/`pn1` read loop, before the
   closing update overwrites `a.pk`.
2. In the closing-update loop, special-cased `k==0`/`k==km` to write back
   `pk_old_top`/`pk_old_bot`; interior `0<k<km` unchanged
   (`fv3_exp<fv_real>(a.akap * pn2[k])`).

- Interior formulas changed: **NO**.
- `peln` changed: **NO** (already correct).
- GMAO branches (`map_scalar_column`, PT's call site `gmao_bot`/`gmao_top`):
  **untouched**.
- `pk` write layout (`.at(i,j,k)`): **unchanged**.

## 4. Stage 3 — pkz diagnostic coverage fix

File (separate checkout): `GEOSfvdycore_fortran_perf/.../fv_mapz.F90`,
inside the `#ifdef MOIST_CAPPA` branch (`~1101-1107`), the only branch this
build actually compiles (`MOIST_CAPPA`/`USE_COND` are unconditional defines,
`@fvdycore/CMakeLists.txt:130`). Added a `delp`/`pkz` dump mirroring the
sibling `#else` branch's existing (dead-code-here) dump, same gating
(`p60_dump_active .and. j==(js+je)/2 .and. i==(is+ie)/2 .and.
(k==1 .or. k==5 .or. k==10)`). Inert unless the env var is set; no
numerical-path change; no new algorithm.

## 5. Stage 4 — Standalone regression

Rebuilt `build_cuda_r8_mpi` (dev OpenMPI stack, reconfigured + targets
`fv3_wp_c_map_scalar_test`, `fv3_wp_c_pt_delz_remap_gpu_smoke`,
`fv3_wp_c_pt_delz_remap_prod_shape_smoke`).

- `fv3_wp_c_map_scalar` ctest (host-visible, covers Prompt 61's GMAO
  fixture `case9_gmao_boundary_conditions`): **PASSED**.
- `fv3_wp_c_pt_delz_remap_gpu_smoke` (GPU, srun `gpu_a100`/`rome`, node
  `warpa003`): all 6 fixture cases OK (pt/delz/pkz/delp round-off,
  unchanged from pre-fix — the fix does not touch these fields' formulas).
- `fv3_wp_c_pt_delz_remap_prod_shape_smoke` (GPU, same job): all 6 cases OK.
- `compute-sanitizer --tool memcheck` on both GPU smoke binaries: **0
  errors** both.

## 6. Stage 5 — PTDELZ_EXIT call 1 (production Intel-MPI stack)

Rebuilt `build_cuda_r8_intelmpi` (`cuda_intelmpi_build.sh`) and relinked
`GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x` against it.
Binary sha256
`1a88bd067e27121e1445d663423574cb51ae86f54d98ea012da000bfab25702b`, copied
into both arm directories (`p62_stage5_cuda`, `p62_stage5_fortran`),
identical sha256 confirmed in both. Same-node pair (jobs 58524186
cuda-arm / 58524187 fortran-control-arm, both `warpa003`, both COMPLETED
exit 0). Provenance confirmed via each job's own `***** USING ****` log
line (points at the copied, matching-sha256 binary) and dump-verified
`gmao_top_bc=T`/`gmao_bot_bc=T` (both arms) and `use_cuda_pt_delz_remap=T`
(cuda arm, all 6 ranks) / `=F` (fortran arm, all 6 ranks).

Rank-paired (`pe=<mpp_pe()>` join key) comparison, call=1, k=1/5/10, C32/L10:

| field | n | max_abs | max_rel | classification |
|---|---|---|---|---|
| pt | 18 | 2.842e-13 | 9.232e-16 | round-off (unchanged from Prompt 61) |
| delz | 18 | 1.478e-12 | 1.558e-15 | round-off (unchanged from Prompt 61) |
| delp | 18 | 0.0 | 0.0 | bitwise |
| pe | 18 | 0.0 | 0.0 | bitwise |
| peln | 18 | 0.0 | 0.0 | bitwise |
| **pk** | 18 | **0.0** | **0.0** | **bitwise — was 7.843e-08 at k=1, now GONE** |
| pkz | 18 | 5.329e-15 | 5.937e-16 | round-off (newly observable — Stage 3 fix; previously never dumped) |

**Stage 5: PASS.** The pk boundary fix resolves Prompt 61's Stage 7 STOP
condition at call 1.

## 7. Stage 6 — PTDELZ_EXIT calls 1-6

Same job pair, all six dyn_core calls, same rank-paired methodology:

| call | pt max_rel | delz max_rel | delp max_rel | pe max_rel | peln max_rel | pk max_rel | pkz max_rel |
|---|---|---|---|---|---|---|---|
| 1 | 9.23e-16 | 1.56e-15 | 0.0 | 0.0 | 0.0 | 0.0 | 5.94e-16 |
| 2 | 4.26e-10 | 7.36e-10 | 1.62e-09 | 1.44e-09 | 1.26e-10 | 4.12e-10 | 4.16e-10 |
| 3 | 3.22e-09 | 5.82e-09 | 1.34e-08 | 1.19e-08 | 1.04e-09 | 3.40e-09 | 3.43e-09 |
| 4 | 1.46e-08 | 2.22e-08 | 5.21e-08 | 4.62e-08 | 4.05e-09 | 1.32e-08 | 1.33e-08 |
| 5 | 4.76e-08 | 6.54e-08 | 1.53e-07 | 1.36e-07 | 1.19e-08 | 3.89e-08 | 3.92e-08 |
| 6 | 1.19e-07 | 1.62e-07 | 3.74e-07 | 3.32e-07 | 2.91e-08 | 9.49e-08 | 9.56e-08 |

**Growing divergence, confirmed real, NOT round-off.** Every field grows
monotonically from call 2 through call 6, roughly 3-5x per call. By call 6,
`delp`/`pe` reach ~3e-7 relative — over three orders of magnitude above
Gate C2's own established round-off ceiling for this exact grid and call
depth (C32, 6-call depth: worst `w` max_rel ~5e-11, per
`CURRENT_STATE.md` §4). This is smaller than the pre-Prompt-61 magnitude
of the same call-2-onward symptom (pt max_rel 1.28e-6->6.19e-6, Prompt
57/58/59) but **not eliminated** — the GMAO boundary port (Prompt 61) plus
this prompt's pk/peln boundary fix reduced it roughly 10x but did not
remove it.

**Stage 6: FAIL.** Per Prompt 62's explicit gate ("no growing divergence;
... bitwise or Gate-C2-like roundoff"), this triggers a STOP. Stages 7
(C32 `dcx`), 8 (C512 A/B), and 9 (milestone restoration) are **NOT
attempted** this prompt.

## 8. Scope note — no second fix applied

This growing divergence is a distinct, unidentified residual mechanism —
not part of, and not fixed by, either the GMAO boundary port (Prompt 61)
or this prompt's pk/peln boundary pass-through fix (both of which are
independently validated correct via their own call-1 bitwise/round-off
results). Per Prompt 62's change budget ("exactly one pk/peln boundary-
semantics fix... no second numerical fix"), root-causing this is
deliberately deferred to a dedicated future prompt rather than attempted
here.

## 9. Summary

- pk/peln boundary pass-through fix (Stages 0-2): **FIXED, VALIDATED** at
  call 1 (bitwise, was 7.843e-08 at k=1).
- pkz diagnostic coverage (Stage 3): **FIXED** — pkz now observable in the
  production MOIST_CAPPA build; zero numerical-path change.
- Standalone regression (Stage 4): **ALL PASS**, compute-sanitizer 0
  errors both smoke binaries.
- PTDELZ_EXIT call 1 (Stage 5): **PASS** — all 7 fields bitwise/round-off.
- PTDELZ_EXIT calls 1-6 (Stage 6): **FAIL** — growing, non-round-off
  divergence across all 7 fields from call 2 onward, smaller than but not
  eliminating the pre-existing "Bug 2" secondary symptom. Root cause
  unidentified; not investigated further this prompt (budget/scope).
- Stages 7-9 (C32 `dcx`, C512, milestone): **NOT ATTEMPTED** (Stage 6 STOP).

## 10. Update (Prompt 63)

The call-2-onward growing divergence left open in §7 has been
root-caused: a three-arm isolation control (pure Fortran / CUDA pt-delz /
CUDA-dyncore-with-Fortran-fidelity-pt-delz) confirmed the defect is
specific to the pt/delz CUDA dispatch, not the broader CUDA dyn_core sync
machinery, and a direct source audit of the H2D/D2H sync masks
(`fv3_gpu_dyncore_bridge.F90`) found no state-handoff/synchronization
defect — both the narrow pt/delz mask and the broad dyncore mask
correctly and unconditionally cover all 7 fields every call. The growth is
instead legitimate nonlinear-dynamics amplification of a real,
round-off-level (~1e-15), individually-correct arithmetic difference
between the CUDA pt/delz kernel and Fortran's `map_scalar` (call-2 ENTRY
is already non-round-off, before call 2's own dispatch runs). No further
fix applied — this requires numerical-arithmetic-level authorization
outside Prompt 63's state-handoff-only scope. Full detail:
`p63_pt_delz_multicall_state_handoff_2026-09-23.md`.
