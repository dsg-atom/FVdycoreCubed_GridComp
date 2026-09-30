# Prompt 91 — Map and Harness REMAP_PRESSUREVARS for CUDA Porting

Date: 2026-09-27
Branch: `optimization/cuda-fv3-performance`
Scope: bounded source-mapping + authoritative-harness task only. No
production CUDA implementation, no PT/DELZ/REMAP_UV/REMAP_Q/tracer-CUDA
changes, no REMAP_TOTALENERGYCON/MOIST_CAPPA-numerics/`a_imp`/R4 changes, no
performance profiling job, no `nsys`/`ncu`. No production selector, no
default flip, no commit/push.

## Stage 0 — Exact Timer Contents

`Remap_PressureVars` (`fv_mapz.F90:1239-1516`, `GEOSfvdycore_fortran_perf`
checkout, `src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`).
`call timing_on('Remap_PressureVars')` at line 1239; `call
timing_off('Remap_PressureVars')` at line 1516. Single `!$OMP parallel do`
over `j=js,je` (line 1251), `i`/`k` looped inside; closes at `2000 continue`
(line 1514).

Full body, categorized by direct read (not inferred from the timer name):

| Lines | Content | Category |
|---|---|---|
| 1256-1269 | `pe` recompute from `ak`/`bk`/surface `pe` | numerical, **unconditional/active** |
| 1274-1291 | `peln`/`pk` recompute (`log`/`exp(akap*...)`) | numerical, **unconditional/active** |
| 1296-1300 | `delp` recompute from the new `pe` | numerical, **unconditional/active** |
| 1302-1344 | `[P60_EXIT_DUMP]` writes | diagnostic, inert unless `p60_dump_active` |
| 1349-1380 | `hydrostatic` branch (pkz from pk deltas, T_v/pt updates) | **DEAD** — production is nonhydrostatic |
| 1381 | `else` (nonhydrostatic) — production takes this branch | control |
| 1382-1422 | `remap_te` block (TE inversion → pt/pkz, `mpp`-free) | **DEAD** — production `remap_te=.false.` |
| 1423-1446 | `remap_t` block, `#ifdef MOIST_CAPPA` (`moist_cv`+`cappa`+`pkz`) | numerical, **active** |
| 1432-1445 | `[P60_EXIT_DUMP]` writes inside the active branch | diagnostic, inert unless `p60_dump_active` |
| 1447-1467 | `remap_t` block, `#else` (dry `akap`-only `pkz`) | **DEAD SOURCE** — `MOIST_CAPPA` always compiled in (see Stage 4) |
| 1470-1479 | `remap_pt` block (Theta_v → T_v) | **DEAD** — production `remap_pt=.false.` |
| 1483-1512 | `do_omega` interpolation (`pe3`/`dpe2`/search loop) | **DEAD** — production `do_omega=.false.` (Stage 1) |

No preprocessing/postprocessing outside the numerical blocks above; no
collective/global operation of any kind (Stage 7).

**Net finding**: of the 278-line timer body, the numerically active,
production-reachable content is only ~50 executable lines (1256-1300,
1423-1446 minus its dump lines) — the remainder is dead branches (three
different `remap_*`/`hydrostatic` combinations that can never all be true at
once) or inert diagnostics. This is the single most consequential Stage 0
finding: **the 131.4 ms/call cost is attributable to a very small numerical
kernel**, not to the large amount of surrounding dead source.

## Stage 1 — Production Call Flow

`Remap_PressureVars` executes inside `Lagrangian_to_Eulerian` (the same
subroutine housing PT/DELZ's Fortran fallback, REMAP_UV's Fortran fallback,
and REMAP_Q's Fortran fallback), immediately after `Remap_Q` and the
`Remap_NH`/`Remap_T`/`Remap_UV` blocks earlier in the file, and immediately
before `Remap_TotalEnergyConsv` (out of scope, line 1521).

Loop nesting: single `j` loop (OMP-parallel), `i`/`k` loops nested inside —
identical shape to every already-ported vertical-remap stage (PT/DELZ,
REMAP_UV, REMAP_Q).

Runtime branch values actually taken in the production first-run
nonhydrostatic config (traced to source, not assumed):

- `hydrostatic = .false.` — production nonhydrostatic config
  (`project_architecture.md`'s production first-run: `hydrostatic = false`).
- `remap_option = 0` ⇒ `remap_t = .true.`, `remap_pt = .false.`,
  `remap_te = .false.` (`fv_mapz.F90:372-385`; already the pinned value the
  existing, validated `use_cuda_pt_delz_remap` selector itself requires via
  its own `remap_option == 0` condition — not re-derived from scratch here).
- `MOIST_CAPPA`/`USE_COND` are **unconditionally compiled in**
  (`@fvdycore/CMakeLists.txt:129-130`:
  `target_compile_definitions(${this} PRIVATE MOIST_CAPPA USE_COND)`) — the
  Fortran always executes the `#ifdef MOIST_CAPPA` branch; the `#else`
  branch is dead SOURCE, not merely a runtime-false branch.
- `do_omega = hydrostatic .and. last_step` (`fv_dynamics.F90:946`) — always
  `.false.` in the nonhydrostatic production config, independent of
  `last_step`.
- `fill`, `kord`, `a_imp`, `k_split`, `n_split`, `gmao_remap` — confirmed by
  direct grep of the timer body (`awk`-bounded to lines 1239-1516) to be
  **absent from every executable statement** in this timer (only `fill`
  appears, and only in the OMP `shared` clause — never read in an
  executable line). See Stage 4.

Inputs → outputs for the active path only:

```
inputs:  ak, bk, ptop, pe(surface, i.e. pe(:,km+1,j))   [pressure-grid basis]
         peln(:,1,j), peln(:,km+1,j), pk(:,j,1), pk(:,j,km+1)  [boundary pass-through]
         pt(:,j,:), delz(:,j,:), q(:,j,:,sphum)          [pkz formula inputs]
calc:    pe2(k)   = ak(k) + bk(k)*pe(surface)             (interior k)
         peln(k)  = log(pe2(k)); pk(k) = exp(akap*peln(k)) (interior k)
         delp(k)  = pe2(k+1) - pe2(k)
         cvm      = moist_cv(...) = cv_air (nwat=0 case-default; q untouched)
         cappa(k) = rdgas / (rdgas + cvm/(1+r_vir*q(sphum)))
         pkz(k)   = exp(cappa(k)*log(rrg*delp(k)/delz(k)*pt(k)))
helper:  moist_cv (Stage 2)
outputs: pe(:,:,j), peln(:,:,j), pk(:,j,:), delp(:,j,:), pkz(:,j,:), cappa(:,j,:)
```

`pt` itself is **not** rewritten in the active `remap_t` branch (comment at
line 1424: "pt at this stage is T_v or T_m" — already finalized upstream by
PT/DELZ's own pretransform, which is itself skipped when `cuda_owns_pt` is
true). This simplifies the I/O table (Stage 3): `pt` is read-only here.

## Stage 2 — Numerical Subroutines Table

| Routine | Role | Active in production? | Called per column? |
|---|---|---|---|
| `moist_cv` (`fv_mapz.F90:4116-4196`) | returns per-`i` `cvm` (heat capacity) for a given `k` | YES — called once per `k` inside the active `remap_t` block, covering all `i` in one call (not literally once-per-column; once per `(j,k)` pair, vectorized over `i`) | effectively yes — its result varies at most per `(i,k)`, and here degenerates to a true constant, `cv_air`, via the `nwat` not-in-1-7 `case default` (lines 4188-4193), which never touches `q` |
| inline arithmetic (`pe2`/`pn2`/`pk2`/`delp`/`cappa`/`pkz`) | the entire remaining active numerical content | YES | yes, pointwise per `(i,k)` |

**No dominant subroutine** — this is inline-arithmetic-dominated, with
`moist_cv` degenerating to a constant fetch under the active `nwat=0`
configuration. There is no PPM/remap machinery invoked here (unlike PT/DELZ/
REMAP_UV/REMAP_Q, which call `map_scalar`/`mapn_tracer`) — `Remap_
PressureVars` is purely pointwise pressure-variable bookkeeping following
those remaps, not itself a vertical-remap routine. This resolves Stage 2's
warning against premature CUDA-unit selection: the natural CUDA unit is a
single pointwise per-column kernel, not a template instantiation of the
existing PPM-based `map_scalar_column`.

## Stage 3 — Exact I/O Table

| Field | Shape | R/W | Meaning | Current authority |
|---|---|---|---|---|
| `pe` | `(i, km+1, j)` | R (surface only, `pe(:,km+1,j)`) / W (all `k`) | layer-interface pressure | Fortran host array (see Stage 6) |
| `peln` | `(i, km+1, j)` | R (`k=1,km+1` boundary) / W (`k=1..km+1`) | `log(pe)` | Fortran host array |
| `pk` | `(i, j, km+1)` | R (`k=1,km+1` boundary) / W (`k=1..km+1`) | `pe^akap` | Fortran host array |
| `pkz` | `(i, j, km)` | W only | layer-mean `pe^cappa` | Fortran host array |
| `delp` | `(i, j, km)` | W only (recomputed from new `pe`) | layer pressure thickness | Fortran host array |
| `pt` | `(i, j, km)` | R only | virtual/modified temperature (already finalized) | Fortran host array |
| `delz` | `(i, j, km)` | R only | layer depth (already remapped by `Remap_NH`) | Fortran host array |
| `ak`, `bk` | `(km+1)` | R only | hybrid vertical coordinate coefficients | Fortran host array (rank-1, registry field for PT/DELZ/REMAP_UV) |
| `q(:,:,:,sphum)` | `(i, j, km)` | R only | specific humidity (`sphum` tracer index) | Fortran host array; structurally `0` in every validated harness run (cold-start limitation, CURRENT_STATE §7) |
| `cappa` | `(i, j, km)` | W only | derived heat-capacity ratio | Fortran host array |
| `akap`, `rdgas`, `cv_air`, `r_vir`, `rrg`, `ptop` | scalars | R only | constants | compile-time/runtime scalar config |

Excluded (confirmed untouched in the active branches by direct source read):
`w`, `phis`, `te`, `u`, `v`, `gridstruct`, `hs`, `omga`, `pe3`, `dpe2`,
`k1k`, `mfx`/`mfy`/`cx`/`cy` — all referenced only inside the dead
`hydrostatic`/`remap_te`/`remap_pt`/`do_omega` branches or the OMP `shared`
clause without being read in an active statement.

## Stage 4 — Runtime Flag Dependencies

| Flag | Classification | Evidence |
|---|---|---|
| `hydrostatic` | **ACTIVE** (selects the branch; production value `.false.`) | line 1349/1381 |
| `remap_t`/`remap_pt`/`remap_te` | **ACTIVE** (selects the sub-branch; production `remap_t=.true.` only) | lines 1382/1423/1470, `remap_option` derivation `fv_mapz.F90:372-385` |
| `MOIST_CAPPA`/`USE_COND` | **ACTIVE at compile time, not a runtime branch** — unconditionally defined (`CMakeLists.txt:129-130`), so the Fortran source only ever contains the `#ifdef` arm | direct `CMakeLists.txt` read |
| `nwat` | **ACTIVE indirectly** via `moist_cv`'s `case default` (production `nwat=0`) | `moist_cv`, `fv_mapz.F90:4188-4193` |
| `do_omega` | **INACTIVE** (structurally `.false.` in production) | `fv_dynamics.F90:946` |
| `fill`, `kord`, `a_imp`, `k_split`, `n_split`, `gmao_remap` | **IRRELEVANT** (not referenced in any active statement of this timer) | direct grep of lines 1239-1516's executable body |

**Important nuance (Stage 4/Stage 5 correction to the pre-existing project
convention)**: `cappa(i,j,k) = rdgas / ( rdgas + cvm(i)/(1.+r_vir*
q(i,j,k,sphum)) )` (line 1430) depends on `q(sphum)`, not on `nwat` alone.
The `cappa == akap` identity (used by the existing, already-validated
`fv3_pt_delz_remap.cu:19-20` "MOIST_CAPPA/nwat contract" comment) requires
**both** `nwat=0` (so `cvm=cv_air`, a constant) **and** `q(sphum)=0` — the
latter holds in every validated run to date only because of the separate,
already-documented structural cold-start tracer-zero limitation
(`CURRENT_STATE.md` §7, found Prompt 77), not because of `nwat=0` by
itself. This is not a new risk — it is the exact same implicit dependency
PT/DELZ's CUDA kernel already relies on — but it had not previously been
stated with this precision, and is now recorded explicitly (see the harness
comment header and Stage 16).

## Stage 5 — Numerical Semantics

Authoritative formula contract (active branches only), Fortran operation
order preserved exactly:

```
pe2(1)     = ptop
pe2(km+1)  = pe(km+1)                      ! old surface pressure
pe2(k)     = ak(k) + bk(k)*pe2(km+1)       ! k = 2..km
pe(k)      = pe2(k)                        ! k = 1..km+1, write-back

pn2(1)     = peln(1)      [pass-through]
pn2(km+1)  = peln(km+1)   [pass-through]
pk2(1)     = pk(1)        [pass-through]
pk2(km+1)  = pk(km+1)     [pass-through]
pn2(k)     = log(pe2(k))                   ! k = 2..km
pk2(k)     = exp(akap*pn2(k))              ! k = 2..km
peln(k)    = pn2(k); pk(k) = pk2(k)        ! k = 1..km+1, write-back

delp(k)    = pe2(k+1) - pe2(k)             ! k = 1..km

cvm        = cv_air                        ! moist_cv case-default (nwat=0)
cappa(k)   = rdgas / (rdgas + cvm/(1+r_vir*q(sphum,k)))
pkz(k)     = exp(cappa(k)*log(rrg*delp(k)/delz(k)*pt(k)))   ! rrg = -rdgas/grav
```

Boundary handling: `pk`/`peln` at `k=1` and `k=km+1` are **pass-through
identities** (untouched by the recompute) — an exact invariant, not an
approximation. Top/bottom layers otherwise participate in the same formula
as interior layers (no separate top/bottom special-casing inside `pe2`'s
interior loop; only the two boundary values are special-cased).

Transcendental ops: `log`/`exp` appear twice per column-level (once for
`pk`, once for `pkz`) — same ULP-sensitivity class already documented for
PT/DELZ's `fv3_exp<fv_real>`/`fv3_log<fv_real>` helpers (Intel-vs-CUDA
transcendental agreement already characterized and accepted at round-off
level for those kernels); a CUDA port here should reuse the same helper
templates rather than raw `expf`/`logf`/`exp`/`log` (Stage 9).

Accumulation order: none — every quantity here is a closed-form pointwise
expression per `(i,k)`, with no reduction, scan, or order-dependent sum.
This is a materially simpler numerical contract than PPM remap (no
monotonicity limiter, no multi-level stencil) — closer in shape to a single
elementwise kernel than to `map_scalar_column`'s per-column state machine.

## Stage 6 — Data Residency

Traced by direct read of `fv3_gpu_dyncore_bridge.F90` and
`fv3_gpu_runtime.cpp` (not assumed):

- `fv3_gpu_interface_pt_delz_remap` (`fv3_gpu_dyncore_bridge.F90:419-463`)
  does its own `fv3_gpu_sync_from_host` (H2D) **before** its kernel and its
  own `fv3_gpu_sync_to_host` (D2H) **after** it, gated by
  `pt_delz_remap_state_mask` (`FV3_FID_PE, FV3_FID_PELN, FV3_FID_PT,
  FV3_FID_DELZ, FV3_FID_DELP, FV3_FID_PK, FV3_FID_PKZ, FV3_FID_AK,
  FV3_FID_BK` — `fv3_gpu_dyncore_bridge.F90:1022-1034`) — a full,
  self-contained round trip **within a single call**.
- `fv3_gpu_interface_remap_uv` (lines 482-510) does the identical
  H2D-then-D2H pattern, masked to `FV3_FID_U, FV3_FID_V, FV3_FID_PE,
  FV3_FID_AK, FV3_FID_BK`.
- `fv3_gpu_interface_remap_q` has **no** `FieldRegistry`/mask entry for
  `q` at all — it owns one explicit contiguous `cudaMemcpyAsync`
  H2D-then-D2H pair directly against the Fortran host pointer
  (`fv3_gpu_runtime.cpp:2620-2656`).

| Field | Resident at `Remap_PressureVars` entry? | Existing registry/workspace | New transfer needed if ported? |
|---|---|---|---|
| `pe`, `peln`, `pk`, `pkz`, `delp`, `pt`, `delz`, `ak`, `bk` | **HOST ONLY** — every prior CUDA stage (PT/DELZ, REMAP_UV, REMAP_Q) syncs fully back to host before returning control to Fortran | `FieldRegistry` entries exist (`FV3_FID_*`) for all except `q` | YES — one H2D at entry |
| `q(sphum)` | HOST ONLY (no registry entry at all; always host-resident between calls) | none (REMAP_Q owns a private device buffer, not reused elsewhere) | YES, but only a `(i,j,km)` single-tracer slice — much smaller than REMAP_Q's `nq`-tracer buffer |
| `cappa` | n/a (new output, not previously produced by any CUDA stage) | none | new device-resident scratch/output, same pattern as `pkz` |

**Key question answered**: *can this operate mostly on data already
resident from PT/DELZ/REMAP_UV/REMAP_Q without new host/device traffic?*
**No** — under the CURRENT integration architecture, every existing CUDA
vertical-remap stage deliberately completes its own H2D→kernel→D2H round
trip and hands fully-host-resident state back to Fortran, specifically so
that still-Fortran stages (today including `Remap_PressureVars` itself)
can operate correctly. A CUDA port of `Remap_PressureVars` would therefore
need its **own** H2D-at-entry/D2H-at-exit round trip, following the exact
same self-contained pattern as the three existing ports — not a novel
residency design, and not a violation of the persistent-device-residency
architecture goal (each *port* is self-contained; eliminating the
inter-stage round trips entirely is a separate, larger future
optimization, out of scope here per Prompt 91's bounds).

## Stage 7 — Communication

Confirmed by direct read of the full 1239-1516 body: **no MPI call, no
halo exchange, no collective of any kind**. The only cross-rank-adjacent
artifact is the inert `mpp_pe()` call inside the `[P60_EXIT_DUMP]`
diagnostic (rank-ID string tagging only, not a reduction), gated off by
default. This explicitly reconfirms Prompt 90's zero-MPI static-mapping
conclusion with direct source evidence (not merely re-citing it).

## Stage 8 — Workspace / Memory

Per-column scratch actually needed by the active branches: `pe2`, `pn2`,
`pk2` (each `km+1`-length) — declared in the OMP `private` clause
(`fv_mapz.F90:1249-1250`) alongside `pe3`/`dpe2`/`bkh`/`gz`/`cvm`/`phis`/
`tpe`/`dlnp`/`tmp`/`kp`/`k_next`, all of which belong to the **dead**
`hydrostatic`/`remap_te`/`do_omega` branches and are excludable from any
CUDA port's scratch budget.

Estimated scratch per column: 3 arrays × `(km+1)` `fv_real` elements
(`pe2`/`pn2`/`pk2`) ≈ `3*(km+1)*8` bytes — for `km=91` (GEOS standard),
≈2.2 KB/column. This is the same order of magnitude as PT/DELZ's own
per-column scratch (`q2`/`pn1`/`pn2`/etc. in `fv3_pt_delz_remap.cu`), and
fits the existing per-thread-register/shared-memory pattern already used
there — **no new large workspace class is implied**. `WorkspaceId::vertical`
(the existing named device workspace) is a plausible, likely-sufficient
reuse target for these three small scratch arrays, following the same
allocation convention as the existing vertical kernels; confirming the
exact `WorkspaceId::vertical` call site is deferred to the implementation
task itself (out of scope for this mapping prompt, per Prompt 91 Stage 8's
own instruction to avoid Prompt-68-style per-thread local-memory
allocation without over-specifying the eventual kernel's exact code).

**Memory risk: LOW.** The scratch shape is a strict subset (in count and
size) of what PT/DELZ's existing kernel already manages successfully at
C512 scale (after the Prompt 67/68 OOM fix); no whole-field temporaries,
no per-thread dynamic allocation, no new large buffer class.

## Stage 9 — CUDA Reuse Analysis

| Required operation | Classification | Evidence |
|---|---|---|
| `pe2`/`delp` recompute from `ak`/`bk`/surface pressure | **REUSE DIRECTLY** (identical arithmetic pattern) | `fv3_pt_delz_remap.cu`'s own "delp/pe/peln/pk closing update" (its header comment cites `fv_mapz.F90:789-848`, the analogous block in PT/DELZ's own Fortran call site) computes the same `pe2(k)=ak(k)+bk(k)*pe(surface)` / `delp` shape |
| `pn2`/`pk2` via `log`/`exp(akap*...)` | **REUSE DIRECTLY** — PT/DELZ's kernel already implements `pk = fv3_exp<fv_real>(a.akap * pn2[k])` for interior levels with boundary pass-through at top/bottom, the exact same formula and boundary-handling shape | `fv3_pt_delz_remap.cu` (per prior session's read, lines ~290-315) |
| `cappa`/`pkz` via the dry-degenerate `akap` formula | **REUSE DIRECTLY** under the production `q(sphum)=0` condition — numerically identical to PT/DELZ's own `pkz = fv3_exp<fv_real>(a.akap * fv3_log<fv_real>(rrg * new_delp / delz_out * pt))` | `fv3_pt_delz_remap.cu:19-20` contract comment + this task's Stage 4/5 finding |
| `cappa`/`pkz` via the **general** `q(sphum)`-dependent formula | **NEW SMALL HELPER** — only needed if a future task decides to support nonzero `q(sphum)` (not reached in any validated run to date); the degenerate case above is sufficient for every currently-validated production configuration | this task's Stage 4 finding |
| `fv3_exp<fv_real>`/`fv3_log<fv_real>` transcendental helpers | **REUSE DIRECTLY** (already `__host__ __device__` templates) | `fv3_pt_delz_remap.cuh` |
| Per-column dispatch pattern (`Column<T>`/one-thread-per-column launch) | **REUSE WITH SMALL EXTENSION** — same per-column launch shape as `map_scalar_column`/PT-DELZ/REMAP_Q, but `Remap_PressureVars` needs no PPM/monotonicity machinery, only the pointwise arithmetic, so it would extend the *launch/dispatch* convention without needing `map_scalar_column` itself | Stage 2/5 finding |
| H2D/D2H round trip, `FieldRegistry` mask | **REUSE DIRECTLY** — same `pt_delz_remap_state_mask`-style pattern (extended with `FV3_FID_CAPPA` if a device-resident `cappa` output field is desired, or omitted if `cappa` is a pure kernel-internal scratch never read by Fortran downstream — needs confirming whether any later stage reads `cappa` on the host; not yet checked, deferred to the implementation task) | `fv3_gpu_dyncore_bridge.F90` mask subroutines |

**Overall reuse potential: VERY HIGH.** Nearly the entire active numerical
contract of `Remap_PressureVars` already has a directly-analogous,
previously-validated CUDA implementation pattern in
`fv3_pt_delz_remap.cu`/`.cuh` — this is a stronger reuse position than any
of PT/DELZ, REMAP_UV, or REMAP_Q had at their own respective mapping
stages, each of which needed either new PPM logic or a new `fillz` helper.

## Stage 10 — Authoritative Harness Unit

`Remap_PressureVars` is **inline code inside `Lagrangian_to_Eulerian`**, not
a standalone callable subroutine — confirmed by the absence of any
`subroutine`/`function` boundary between line 1239 and line 1516 (it sits
between the `Remap_Q` and `Remap_TotalEnergyConsv` timer blocks inside the
same routine). Per Prompt 91's stated preference order:

- **Option A (direct call)**: not available — there is no isolated
  subroutine to call.
- **Option B (minimal wrapper)**: not available without modifying
  production source to expose a new entry point, which Prompt 91 (and the
  project's general "do not modify production source to expose a test
  entry" rule) forbids.
- **Option C (exact test-only transcription)**: **selected**, following the
  exact precedent already established for `fillz`
  (`test_fillz_host_reference.cpp`, Prompt 88A/88B) — a mechanical,
  line-for-line C++ transcription of the production-active branches only
  (Stage 0/1/4/5), with every excluded branch documented and justified by
  direct source citation rather than by assumption.

**Why this is authoritative**: every formula, boundary case, and constant
in the harness (`cuda/dyn_core/tests/vertical/
test_remap_pressurevars_host_reference.cpp`) is copied in Fortran operation
order from the exact cited line ranges, with the same variable role
(`pe2`/`pn2`/`pk2`/`cappa`/`pkz`), the same boundary pass-through structure,
and — critically — the harness transcribes the **general** `q(sphum)`-
dependent `cappa` formula (not a hard-coded `cappa=akap` shortcut), so it
independently *verifies* the dry-degenerate identity as a checked invariant
rather than assuming it.

## Stage 11 — Fixtures

`test_remap_pressurevars_host_reference.cpp`, 10 fixtures (`main()`):

1. `c512_like_smooth_q0` — realistic C512-like column (`km=91`, smooth
   pressure/temperature profile).
2. `strongly_stratified_q0` — large lapse-rate contrast top/bottom.
3. `top_boundary_thin_ptop_q0` — very small `ptop`, thin top layer.
4. `bottom_boundary_low_ps_q0` — reduced surface pressure (thin near-surface
   layer relative to `ps`).
5. `weak_delp_fine_levels_q0` — finer vertical spacing (`km=120`), narrower
   individual layer thicknesses.
6. `nonhydrostatic_large_delz_contrast_q0` — production's only reachable
   branch is nonhydrostatic (already exercised by every fixture); this one
   additionally stresses a large `|delz|` contrast.
7. `q_sphum_zero_production_value` — the actual production/cold-start value
   (`q(sphum)=0` everywhere).
8. `q_sphum_nonzero_hypothetical` — nonzero `q(sphum)`, exercising the
   *general* formula (not reached in production to date, but must still be
   transcribed correctly) — does **not** assert `cappa==akap`.
9. `transcendental_stress_small_delz_q0` — large `ps` and sharply reduced
   `|delz|` together, stressing `log`/`exp` argument magnitude.
10. `independent_reconstruction_repeatable` — two independently constructed,
    identical columns must produce bitwise-identical output.

Only physically/legally valid states are used (positive pressures,
monotonic `ak+bk*ps`, positive `pt`, nonzero `delz`).

## Stage 12 — Invariants

Checked programmatically in the harness for every fixture (guaranteed by
source, not invented):

- `delp(k) > 0` for all `k` (positive layer thickness — direct consequence
  of monotonic `pe2`).
- `pe(k+1) > pe(k)` (monotonic pressure interfaces).
- `pk(k) == pe(k)^akap` for interior `k` (round-off) — the defining
  `pn2`/`pk2` identity.
- Boundary pass-through: `pk(1)`, `pk(km+1)`, `peln(1)`, `peln(km+1)` are
  bitwise-unchanged from input (exact identity, not approximate).
- No NaN/Inf in any output field.
- `0 < cappa(k) < 1` (physical range given `rdgas>0`, `cvm>0`, `q(sphum)>=0`).
- Dry-degenerate identity (`q_sphum==0` fixtures only): `cappa(k)==akap`
  to `1e-12`, and `pkz` matches the dead-`#else`-branch formula bitwise.
- Deterministic repeatability: identical input ⇒ bitwise-identical output,
  both via immediate re-run and via independent reconstruction.

## Stage 13 — Host Harness Build/Run

- File: `cuda/dyn_core/tests/vertical/test_remap_pressurevars_host_reference.cpp`
  (self-contained, zero project-header/CUDA dependency — same class as
  `test_fillz_host_reference.cpp`).
- CMake wiring: `cuda/dyn_core/CMakeLists.txt`, new target
  `fv3_wp_c_remap_pressurevars_host_reference_test`, new test
  `fv3_wp_c_remap_pressurevars_host_reference`.
- Built and run two ways, both on the Discover login node (CPU-only, no
  Slurm required for a host-only harness):
  1. Direct `g++ -std=c++17 -O2 -Wall` compile — **rc=0, 10/10 fixtures
     PASS**.
  2. Full project build under the authoritative Intel-MPI production stack
     (`build_cuda_r8_intelmpi`, `comp/gcc/12.3.0`/`comp/intel/2024.2.0`/
     `mpi/impi/2021.13`/`nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++`):
     clean `cmake .` reconfigure + `cmake --build .` (whole target, 100%,
     no errors) + `ctest -R fv3_wp_c_remap_pressurevars_host_reference` —
     **1/1 test PASSED**.
- 0 NaN/Inf (checked explicitly per fixture). Deterministic/bitwise
  repeatability confirmed (fixture 10 + the internal repeat check in every
  other fixture). No harness bugs found requiring a fix. **No production
  numerical source was modified.**

## Stage 14 — Source Size / Port Unit Estimate

- Production Fortran LOC (full timer body): 278 lines (1239-1516).
- Active numerical LOC: ~50 lines (pe/peln/pk/delp recompute + the
  `remap_t`/MOIST_CAPPA `cappa`/`pkz` block), excluding dump/dead branches.
- Helper LOC: `moist_cv`'s `case default` reduces to a 1-line constant
  fetch in the active path (no new helper needed given the reuse in
  Stage 9; a new small helper would only be needed for the never-reached
  general-`q(sphum)` branch).
- Expected Fortran bridge LOC (mirroring `fv3_gpu_interface_pt_delz_remap`):
  ~40-50 lines (mask setup, H2D, kernel call, D2H, event wait) — same
  shape as the existing three bridge subroutines.
- Expected CUDA LOC: ~60-90 lines for a new `fv3_pressurevars_column`-style
  device function plus a thin kernel/dispatch wrapper, given the very high
  Stage 9 reuse (most of the arithmetic can be lifted nearly verbatim from
  `fv3_pt_delz_remap.cu`'s own closing-update/pkz code, not written from
  scratch).

**Classification: SMALL.** This is smaller in active-numerics scope than
any of PT/DELZ, REMAP_UV, or REMAP_Q at their respective mapping stages —
no PPM/monotonicity machinery, no `fillz`-style iterative fix, purely
closed-form pointwise arithmetic with very high direct-reuse potential.

**Proposed CUDA unit**: one new per-column device function (working name
`fv3_pressurevars_column`) implementing exactly the Stage 5 formula
contract (dry-degenerate `q(sphum)=0` path only, matching every other
currently-validated CUDA vertical-remap stage's scope), dispatched via a
new `fv3_gpu_remap_pressurevars` C ABI entry point following the
`fv3_gpu_pt_delz_remap`/`fv3_gpu_remap_uv` self-contained H2D/D2H pattern —
not a larger unit (e.g., not folding in `Remap_TotalEnergyConsv`, not
attempting the general `q(sphum)`-dependent formula).

## Stage 15 — Performance Ceiling

Using the Prompt 90 current profile baseline of **131.4 ms/call** (not any
pre-port historical figure, since this stage was never previously ported):

- If a CUDA port achieves a speedup ratio comparable to REMAP_Q's directly
  observed 3.23x (`P88B`, the most conservative of the three prior ports —
  REMAP_UV was ~10.2x, PT/DELZ's own delz/pt speedup was larger still),
  `Remap_PressureVars` would drop to roughly 131.4/3.23 ≈ **40.7 ms/call**,
  recovering **≈90.7 ms/call**.
- As a fraction of the Prompt 90 profile: ≈90.7 ms/call is **≈18.2% of
  REMAPPING** (499.6 ms/call), **≈5.3% of FV_DYN_LOOP** (1706.8 ms/call),
  and **≈2.3% of TOTAL** (3932.4 ms/call).
- This does **not** assume zero CUDA cost: the estimate already nets out
  a new H2D/D2H round trip (Stage 6) at a size comparable to PT/DELZ's own
  transfer (same field set, same shape) — the 3.23x figure it is
  benchmarked against is itself REMAP_Q's own post-transfer net speedup,
  not a transfer-free idealization.
- A more conservative bound, using REMAP_Q's ratio as the floor (since this
  stage's numerics are strictly simpler and more directly reusable than
  REMAP_Q's, the realized ratio is more likely to exceed 3.23x than fall
  short of it): **≥18% of REMAPPING, ≥5% of FV_DYN_LOOP, ≥2% of TOTAL** is
  a reasonable expected-improvement floor, not a ceiling.

## Stage 16 — GO / NO-GO

| Criterion | Assessment |
|---|---|
| Numerical unit cleanly separable | YES — closed-form pointwise arithmetic, no PPM/limiter coupling (Stage 2/5) |
| Production branch set bounded | YES — exactly one `hydrostatic`/`remap_t`/`remap_pt`/`remap_te`/`do_omega` combination is reachable, all confirmed by direct source trace (Stage 1/4) |
| No MPI/halo redesign required | YES — zero MPI/collective calls confirmed by direct read (Stage 7) |
| Data mostly resident | **Partially** — data is currently HOST ONLY at entry (Stage 6), same as every prior port's own starting condition; a new self-contained H2D/D2H round trip is required and budgeted into the Stage 15 estimate, consistent with existing precedent, not a blocking gap |
| Workspace fits existing vertical architecture | YES — 3 small per-column scratch arrays, LOW memory risk, no new class of allocation (Stage 8) |
| No unresolved correctness semantics | YES, with one explicitly-flagged, already-inherited (not new) caveat: the `cappa==akap` simplification depends on `q(sphum)=0`, which holds today only via the separate cold-start limitation — documented precisely in Stage 4/5/9 and in the harness itself, matching (and clarifying) the existing PT/DELZ CUDA convention rather than introducing a new risk |

**No stop condition triggered.** `Remap_PressureVars`'s active numerical
content does not depend on broad non-resident state beyond the same
self-contained-transfer pattern already used successfully three times; it
does not require any unimplemented high-risk numerical branch (the
`MOIST_CAPPA`/`nwat` general path is not reached in production); host/device
traffic is bounded to the same order of magnitude as the existing ports,
not dominant; and the selected unit captures the *entire* active numerical
content of the 131.4 ms/call timer (not a small fraction of it).

**DECISION: GO.**

## Summary Table

| Stage | Result |
|---|---|
| 0 — Timer contents | ~50 active LOC out of 278; rest dead/diagnostic |
| 1 — Call flow | traced; `hydrostatic=.false.`, `remap_t=.true.`, `do_omega=.false.` |
| 2 — Numerical routines | inline-arithmetic-dominated; `moist_cv` degenerates to a constant |
| 3 — I/O table | 8 R/W fields + 3 read-only + constants; `w`/`phis`/`te`/`u`/`v` excluded |
| 4 — Flag dependencies | `fill`/`kord`/`a_imp`/`k_split`/`n_split`/`gmao_remap` IRRELEVANT; `cappa==akap` caveat identified |
| 5 — Numerical semantics | closed-form pointwise; boundary pass-through exact |
| 6 — Data residency | HOST ONLY at entry (every prior port's own starting condition) |
| 7 — Communication | zero MPI/halo/collective, confirmed by direct read |
| 8 — Workspace | 3 small per-column scratch arrays; LOW memory risk |
| 9 — CUDA reuse | VERY HIGH — nearly all arithmetic already implemented in `fv3_pt_delz_remap.cu` |
| 10 — Harness unit | Option C, test-only transcription (inline production code, no subroutine to call) |
| 11 — Fixtures | 10 fixtures, all production-relevant behaviors covered |
| 12 — Invariants | 7 invariant classes, all derived from source |
| 13 — Harness result | rc=0, 10/10 PASS, 0 NaN/Inf, deterministic, builds clean under project CMake |
| 14 — Port size | SMALL; proposed unit: one new per-column device function + thin ABI wrapper |
| 15 — Performance ceiling | ≈90.7 ms/call recoverable (≈18% REMAPPING / ≈5% FV_DYN_LOOP / ≈2% TOTAL), floor not ceiling |
| 16 — GO/NO-GO | **GO** |

**No production CUDA implementation added. No PT/DELZ/REMAP_UV/REMAP_Q/
tracer-CUDA changes. No REMAP_TOTALENERGYCON/MOIST_CAPPA-numerics/`a_imp`/R4
changes. No performance profiling job run. No `nsys`/`ncu` used. No
production selector or default changed. No commit/push performed.**

## Addendum (2026-09-27, Prompt 92)

The statements immediately above are now superseded by follow-on work in
the same task line. Prompt 92 implemented, GPU-validated, integration-
tested, and performance-tested the CUDA `REMAP_PRESSUREVARS` port whose
plan this document only proposed: device kernel + thin ABI wrapper,
`use_cuda_remap_pressurevars` production selector (Fortran remains
default; CUDA is opt-in via `RUN_CUDA_FV3_PRESSUREVARS_ENABLE`), 10/10
fixture PASS on GPU, compute-sanitizer 0 errors, C32 integration PASS,
and a real 5-pair same-node C512 A/B performance result (direct speedup
≈3.50x; REMAPPING −15.75%; FV_DYN_LOOP −4.84%; TOTAL −1.81%). See
`docs/performance/p92_remap_pressurevars_cuda_implementation_2026-09-27.md`
for the full result. This plan document's own content (Stages 0-16
mapping/harness work) remains accurate as a historical record and is not
rewritten.

## Addendum (2026-09-28, Prompt 93)

Prompt 93 closed the "CONSIDER DEFAULT ENABLE" question Prompt 92 left
open: a statistical robustness gate over Prompt 92's own 5 same-node
pairs (5/5 CUDA-positive, mean TOTAL improvement 1.805%) plus one
independent longer 60-call confirmation run (TOTAL improvement 1.676%)
both passed the project's default-policy gate, and
`use_cuda_remap_pressurevars` was flipped to the CUDA production default
(escape hatch `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`). See
`docs/performance/p93_remap_pressurevars_default_policy_2026-09-28.md`.
