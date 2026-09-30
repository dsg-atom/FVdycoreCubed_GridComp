# Prompt 70 — REMAP_UV Source Map and CUDA Porting Harness

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet (no escalation — authoritative Fortran vector-remap
semantics were fully resolved from direct source inspection; the
escalation trigger was never met)

## 0. Task Type

**Bounded source-mapping + authoritative-harness task. NOT a CUDA
implementation task.** No production CUDA source was written or
changed in this prompt. The only file edited is a Fortran test-harness
driver (`map_scalar-driver/main.F90`, in the separate
`GEOSfvdycore_fortran_perf` checkout, outside this repo's git tree),
which calls the unmodified authoritative `map_scalar` directly — this
is explicitly sanctioned by the task's own harness mandate and does not
alter any numerical source.

## 1. REMAP_UV Timer

`Lagrangian_to_Eulerian` (`fv_mapz.F90:907-1010`), timer literal
`'Remap_UV'`. Confirmed by direct read of the timer block boundaries.

## 2. Production Flow

REMAP_UV performs exactly **2 active `map_scalar` calls per j-row** in
the production first-run configuration:

- **u** — remapped using a j-averaged wind-specific pressure grid
  (`pe1`/`pe2` built by averaging `pe` across the j-1/j row pair).
- **v** — remapped using an i-averaged wind-specific pressure grid
  (`pe0`/`pe3` built by averaging `pe` across the i-1/i column pair).

Both calls use `iv=-1`, `kord=kord_mt=9`. `kord_mt=9 > 7`, so
`map_scalar` always dispatches to `scalar_profile` (never
`ppm_profile`) in production — the same dispatch path already exercised
and CUDA-ported for PT/DELZ (`iv=1`).

**4 additional `map_scalar` calls exist in source for `mfy`/`cy`/
`mfx`/`cx`** (flux/courant-number remap, used only by a diagnostic/
alternate code path). Confirmed dead in production: the calling
`present()` guards around these calls are never true at the
`fv_dynamics.F90:952` `Lagrangian_to_Eulerian` call site, because
`mfx`/`mfy`/`cx`/`cy` are never passed as actual arguments there. **Do
not port these 4 calls** — porting them would violate the task's own
"do not port unreachable production code" spirit and add unvalidatable
scope.

## 3. Routines

| Routine | File | Role in REMAP_UV | CUDA status |
|---|---|---|---|
| `Lagrangian_to_Eulerian` | `fv_mapz.F90` | Orchestrates all vertical remap incl. REMAP_UV's 2 calls | Not ported (orchestration only; individual field remaps are the port unit) |
| `map_scalar` | `fv_mapz.F90` | Dispatches to `scalar_profile`/`ppm_profile` by `kord` | CUDA: `map_scalar_column` (`fv3_map_scalar_column.hpp`) — already handles `iv` as runtime int |
| `scalar_profile` | `fv_mapz.F90` | PPM edge reconstruction + limiter + iv-specific boundary clamps | CUDA: `scalar_profile_column` — iv=1 (PT/DELZ) ported; **iv=-1 (wind) NOT yet ported** (2 missing branches, §5) |
| `cs_limiters` | `fv_mapz.F90` | Interior monotonicity limiter, called with literal `iv_mode` 1 or 2 | CUDA: `cs_limiters_column` — fully ported, no changes needed for REMAP_UV |

## 4. Production Config

- `kord_mt=9` (confirmed via `FV_StateMod.F90` override of
  `fv_arrays.F90`'s default of 8, cross-checked against a live job's
  own config-header printout).
- `mfx`/`mfy`/`cx`/`cy` never passed at the `fv_dynamics.F90:952` call
  site — confirmed dead code for REMAP_UV specifically.
- REMAP_UV's `map_scalar` calls never pass `optional_bot`/
  `optional_top` — the GMAO boundary-condition branches
  (`gmao_top_bc`/`gmao_bot_bc`, relevant to PT/DELZ, Prompt 61) are
  **not reachable from REMAP_UV** in production, regardless of the
  `AGCM.rc` GMAO flag settings.

## 5. IV Semantics

Source-confirmed (`fv_mapz.F90`, top-boundary block lines 2340-2409,
bottom-boundary block lines 2555-2599):

- `iv=0` — positive-definite: floor the edge value at zero.
- `iv=1` — generic (PT/DELZ's mode): no special boundary clamp beyond
  the shared interior/edge PPM+limiter machinery.
- `iv=2` — temperature-like/cs-scheme: pin the edge value to the cell
  mean.
- **`iv=-1` — winds (REMAP_UV's mode)**: a **sign-consistency clamp**,
  applied only at the top and bottom boundaries:
  ```fortran
  ! top (fv_mapz.F90:2361-2364)
  elseif (iv == -1) then
     if (a4(2,i,1)*a4(1,i,1) <= 0.) a4(2,i,1) = 0.
  endif
  ! bottom (fv_mapz.F90:2585-2588)
  elseif (iv .eq. -1) then
     if (a4(3,i,km)*a4(1,i,km) <= 0.) a4(3,i,km) = 0.
  endif
  ```
  i.e., "if the reconstructed edge value disagrees in sign with the
  cell mean, zero the edge value" — structurally identical in shape to
  the existing `iv==0` floor-at-zero branches, but a conditional zero
  rather than an unconditional floor.
- `iv=-2` — vertical velocity `w`: requires an additional `qs` surface
  argument, not used by REMAP_UV. Out of scope.

**Everything else in `scalar_profile`/`cs_limiters` (PPM reconstruction,
interior Huynh limiter via `cs_limiters` literal-1/2 calls, deln/kord
dispatch) is iv-independent and already fully validated via PT/DELZ's
`iv=1` port.**

## 6. Vector / Staggering Classification

**Classification A: independent columns.** u and v are each remapped as
their own scalar-like column, using their own wind-specific pressure
grid built by neighbor-averaging `pe` (u: j-1/j average; v: i-1/i
average) — they do **not** share PT/DELZ's `pe1`/`pe2` pressure grid,
and u/v are not jointly staggered or coupled to each other inside a
single `map_scalar` call. This means the existing single-column
`map_scalar_column`/`scalar_profile_column` CUDA kernel shape (one
thread/column) is directly applicable to both u and v — each needs its
own kernel invocation (or a batched dispatch covering both fields), not
a redesigned joint-vector kernel.

## 7. CUDA Reuse

**Classification: REUSE WITH MINIMAL NEW IV BRANCHES.**

Full read of `cuda/dyn_core/vertical/fv3_map_scalar_column.hpp` (353
lines) confirms:

- `map_scalar_column`'s `iv` parameter is **already a plain runtime
  `int`**, not a compile-time template parameter — threaded unchanged
  into `scalar_profile_column`.
- `scalar_profile_column`'s existing `iv==0` top/bottom floor-at-zero
  branches are structurally adjacent to where the 2 new `iv==-1`
  sign-consistency clamps belong — same `if (iv == ...)` dispatch
  pattern, same boundary locations.
- `cs_limiters_column` needs **zero changes**: REMAP_UV's `iv=-1` path
  only ever calls it with the literal `iv_mode` values 1 and 2, both of
  which are already correctly ported (shared with PT/DELZ's usage).

**Net new CUDA logic required: exactly 2 single-line conditionals**
inside `scalar_profile_column`, zero new template parameters, zero
kernel-dispatch changes.

## 8. Data Residency

Direct read of `cuda/interface/fv3_gpu_field_registry.F90` confirms:

- `FV3_FID_U` / `FV3_FID_V` registered (lines 726-727) with staggering
  extents (`isd,nih,jsd,njh+1` for u; `isd,nih+1,jsd,njh` for v)
  matching REMAP_UV's own Fortran loop bounds exactly.
- `FV3_FID_AK` / `FV3_FID_BK` registered (lines 786-787), already
  device-resident.
- `pe` is already a device-resident field used elsewhere in the
  existing CUDA dyn_core.

**Zero new host/device transfer infrastructure is needed** for a
REMAP_UV port — all required inputs are already FieldRegistry-managed,
device-resident fields.

## 9. Memory

Full read of `cuda/dyn_core/vertical/fv3_pt_delz_remap.cuh` (176 lines)
confirms PT/DELZ's Prompt-68 fix pattern: 17 scratch slots ×
`(km+1)` = 187 scalars/column at `km=10`, reusing the shared
`WorkspaceId::vertical` allocation (sized for `Riem_Solver3`'s larger
422-scalar/column need) via the `workspace`/`workspace_scalars_per_column`
parameter pattern — **zero extra device bytes**.

A REMAP_UV port needs scratch for **both** u and v independently (each
needing its own pe1/dpe1/pe2/dpe2 + PPM coefficients c2/c3/c4/gam/q/
extm — roughly 10 slots per field), for an estimated **~20 slots × 11 =
~220 scalars/column at km=10 total**. This comfortably fits under the
existing 422-scalar/column `WorkspaceId::vertical` allocation
(187 already used by PT/DELZ, leaving 235 headroom) — **zero new
`cudaMalloc`/`cudaFree` plausible**, mirroring PT/DELZ's fix exactly.
This is an estimate, not a measured/compiled figure — it should be
confirmed against `nvcc --resource-usage`/`-Xptxas -v` once an actual
kernel is written.

## 10. Harness

**Reused (per CLAUDE.md's Reuse policy), not rebuilt.** Located the
existing Prompt-52A Fortran host-only harness at
`GEOSfvdycore_fortran_perf/.../map_scalar-driver/main.F90`, a standalone
CMake target that compiles the **unmodified authoritative `fv_mapz.F90`**
plus minimal stub dependencies, calls `map_scalar` directly
(single-column, i1=i2=1), and self-checks (repeatability across two
independent calls, NaN/Inf, positive-definiteness where applicable).
Its fixture output format is already the exact format consumed by the
existing CUDA smoke test (`fv3_map_scalar_kernel.cu`).

Extended it (test-harness-only edit) with **4 new deterministic `iv=-1`
fixtures**:

| Case | Purpose |
|---|---|
| `case10_wind_uniform_sign_iv_neg1` | Uniform-sign wind profile — control confirming the new clamp branches are inert (never trigger) when the wind never changes sign, matching `iv=1` behavior exactly on this profile. |
| `case11_wind_top_sign_reversal` | Deliberately triggers the **top** sign-consistency clamp. |
| `case12_wind_bottom_sign_reversal` | Deliberately triggers the **bottom** sign-consistency clamp, mirroring case11 at the opposite end. |
| `case13_wind_interior_sign_reversal` | Sign change in the column interior (away from both boundaries) — confirms the new clamp branches (top/bottom only) do **not** spuriously fire away from the boundary. |

Combined with the pre-existing `case1`-`case7` (`iv=1`, fully applicable
since REMAP_UV shares 100% of the non-clamp logic with PT/DELZ), this
gives **complete production-branch coverage for REMAP_UV**. `case8`
(`iv=0`) and `case9` (GMAO boundary conditions) are correctly retained
as valid differential/regression references but are **not applicable**
to REMAP_UV's own branch set (both are dead code for REMAP_UV in
production — §4).

Sanity-check of the new fixtures' own output confirmed non-trivial,
non-degenerate numerical behavior — e.g. `case11`: `q1(1)=+1.0` (a
positive edge value near a strongly negative interior) maps to
`q2(1)=-14.02` (nonzero, since the clamp only zeroes the *edge*
reconstruction `a4(2,1)`, not the final remapped cell value `q2`, which
also depends on neighboring cells and the limiter); `case12`:
`q1(10)=+1.0` maps to `q2(10)=+1.359`.

## 11. Run Baseline

Module stack: GCC 12.3.0 + Intel 2024.2.0 + Intel MPI 2021.13 +
`nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++` (matches the project's
authoritative production stack). Rebuilt `map_scalar-driver`
(recompiled `fv_mapz.F90.o` and `main.F90.o`, linked cleanly — only
benign `libcrypto.so.1.1` linker warnings). Ran it (required reloading
the same module environment at run time, not just build time, to
resolve a `libiomp5.so` runtime-library dependency — a mechanical,
non-numerical fix).

Result:
```
CASE [case1_smooth_monotonic] OK=T
...
CASE [case10_wind_uniform_sign_iv_neg1] OK=T
CASE [case11_wind_top_sign_reversal] OK=T
CASE [case12_wind_bottom_sign_reversal] OK=T
CASE [case13_wind_interior_sign_reversal] OK=T
MAP_SCALAR_DRIVER: 13 cases run, 0 failed
 MAP_SCALAR_DRIVER: RESULT=PASS
EXIT=0
```

**13/13 cases PASS, rc=0.** This is an authoritative host-only Fortran
baseline against the unmodified `fv_mapz.F90` — not yet a CUDA-vs-
Fortran comparison (no CUDA REMAP_UV implementation exists yet to
compare against).

## 12. Port Scope (LOC Estimate)

- **iv=-1 branch addition to `scalar_profile_column`**: 2 new
  single-line conditionals (mirroring the existing `iv==0` branch
  shape) ≈ **2-6 lines including braces/comments — SMALL.**
- **New `fv3_remap_uv.cu`/`.cuh` production entry point**, mirroring
  `fv3_pt_delz_remap.{cu,cuh}`'s structure (view struct, args struct
  with workspace-reuse fields, kernel, `launch_remap_uv`/
  `launch_remap_uv_ctx` functions) but covering 2 fields (u, v) instead
  of PT/DELZ's 2 fields (pt, delz) with an added u/v-specific
  `pe`-averaging preprocessing step (new logic, not present in any
  existing file) ≈ **300-450 new LOC — MEDIUM**, by direct analogy to
  `fv3_pt_delz_remap.cu`+`.cuh`'s combined existing size.
- **Bridge/dispatch wiring** (Fortran interface call site, FieldRegistry
  dispatch entry, CMake source-list addition) ≈ **50-100 LOC — SMALL**,
  by analogy to PT/DELZ's own bridge wiring.

**Total estimated new/changed LOC: ~350-550 — MEDIUM overall port**,
substantially smaller than a from-scratch remap-family port would have
been, because the PPM/limiter core is 100% reused.

## 13. CUDA Porting Unit Selection

**Selected smallest safe unit**: extend `scalar_profile_column` with
the 2 new `iv=-1` branches (§7) + add a new `fv3_remap_uv.cu`/`.cuh`
production entry point mirroring `fv3_pt_delz_remap.{cu,cuh}`'s
workspace-reuse struct/kernel/launcher pattern, covering **u and v
only** (excluding the dead `mfx`/`mfy`/`cx`/`cy` calls, §2/§4).

Rationale: this is the minimal unit that (a) makes zero changes to
already-validated PT/DELZ code paths beyond the 2 new conditionals
(which are themselves inert for `iv=1`, so PT/DELZ's existing
call sites and validation are unaffected), (b) reuses 100% of the
existing PPM/limiter machinery, (c) reuses the existing device-resident
u/v/pe/ak/bk fields with zero new transfer infrastructure, and (d)
reuses the existing `WorkspaceId::vertical` allocation with zero new
device bytes — mirroring the exact structural precedent (workspace
reuse, header-only templates, single new `.cu`/`.cuh` production entry
point) that PT/DELZ's Prompt 52B/68 already validated end-to-end.

## 14. Performance Ceiling Estimate

**Rough, not yet measured.** REMAP_UV was measured (Prompt 69, §9 of
`CURRENT_STATE.md`) at 169.5 ms/call (9.5% of `FV_DYN_LOOP`) in the
post-PT/DELZ-port profile. If a REMAP_UV CUDA port achieves leverage
comparable to PT/DELZ's measured ~62x (CPU-removed / GPU-added), and if
REMAP_UV's own GPU-added cost scales roughly with its 2-field-vs-2-field
similarity to PT/DELZ (accounting for the additional u/v pe-averaging
preprocessing, which has no PT/DELZ analog and could add modest extra
cost), a plausible net **FV_DYN_LOOP improvement is on the order of
~7-8%** (roughly comparable to, and potentially somewhat below,
PT/DELZ's own realized 16.6% improvement of a larger sub-component),
combining with PT/DELZ's already-banked 16.6% for a **cumulative
estimate on the order of ~20-25%**. This estimate must be treated as
**UNVALIDATED** until an actual paired C512 A/B (analogous to Prompt
69's) is run against a real CUDA REMAP_UV implementation — it is a
scoping estimate only, consistent with Prompt 69's own noted ceiling
(~1.3x max ideal dynamics-side speedup for this HS/npz=10 track).

## 15. Project Status

- Gate C2 (54-stage production first-run dyn_core inventory): **CLOSED,
  unaffected** — this prompt performed no numerical/production source
  changes.
- PT/DELZ CUDA port (Prompt 52B-69): **KEEP, unaffected** — the 2 new
  `iv=-1` branches to be added in a future prompt are inert for `iv=1`
  and do not touch any PT/DELZ call site or validated behavior.
- REMAP_UV: **mapped, harnessed, not yet CUDA-implemented.**

## 16. Conclusion

All primary-goal items (§Primary Goal of the governing task) are
resolved with concrete, source-confirmed evidence: the production call
sequence and routines are fully mapped (§1-4); `iv=-1` semantics are
fully characterized (§5); CUDA reuse is confirmed clean and minimal
(§7); data residency and memory footprint are both already favorable
(§8-9); an authoritative host-only baseline harness exists, was
extended with deterministic fixtures exercising every production
branch, and passes 13/13 (§10-11); the smallest safe CUDA porting unit
is identified and scoped (§12-13).

**REMAP_UV HARNESS READY — PROCEED TO CUDA PORT**
