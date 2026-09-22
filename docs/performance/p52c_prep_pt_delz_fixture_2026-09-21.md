# Prompt 52C-PREP: pt/delz remap surrounding-arithmetic fixture

Resolves the blocker reported in
`p52c_fv_mapz_cuda_integration_2026-09-21.md` §4/§6: a fixture-validated,
host-only baseline for the exact production arithmetic that surrounds the
already-validated `map_scalar`/`scalar_profile` port (52A/52B) at the real
`pt` and `delz` call sites in `fv_mapz_mod::Lagrangian_to_Eulerian`
(`fv_mapz.F90`). No CUDA implementation, GPU job, or production
integration was attempted here — see §7.

## 1. Production sequence (Stage 0)

Traced from
`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`,
for the validated C512 Held-Suarez configuration (`remap_option=0` →
`remap_t=.true.`; `gmao_remap=0`; `kord_mt=kord_wz=kord_tm=9`;
`hydrostatic=.false.`). Order matters — the pe/peln/pk/delp
new-Eulerian-grid update (lines ~789-838) runs **after** both remap calls
but **before** pt's post-processing `pkz`, so `pkz` uses the *new* delp,
the *new* (remapped) `delz`, and the *new* (remapped) `pt`, in that order.

**PT path:**
1. Pre-transform (line 388): `pt = pt*exp(k1k*log(rrg*delp/delz*pt))`
   using the *original* (pre-remap) `delp`/`delz`.
2. `peO` derivation (482-488): `peO(1)=ptop`; `peO(km+1)=pe(km+1)`;
   `peO(2:km)=ak(2:km)+bk(2:km)*pe(km+1)`.
3. `gmao_remap` gate (502): `>0` → `map1_gmao` (unreachable in production,
   `gmao_remap=0`); else: `pn1=peln`, `pn2(1)=peln(1)`,
   `pn2(km+1)=peln(km+1)`, `pn2(2:km)=log(peO(2:km))`,
   `dpn1=pn1(2:km+1)-pn1(1:km)`, `dpn2=pn2(2:km+1)-pn2(1:km)`.
4. `map_scalar` call (529-533): `iv=1, kord=9, q_min=t_min=184.0`,
   `optional_top=optional_bot=.false.` (matches 52B's "no GMAO bot/top"
   restriction).
5. Post-processing (928-930, **after** the pe/peln/pk/delp update block):
   `pkz = exp(akap*log(rrg*delp_new/delz_new*pt_new))`.

**DELZ path (576-613, no MOIST_CAPPA dependency anywhere):**
1. `dpe`/`dpeO` derivation (576-585): `dpe=pe(k+1)-pe(k)`,
   `dpeO=peO(k+1)-peO(k)` — shares `peO` with the pt path.
2. Pre-transform (601): `delz = -delz/dpe` ("specific volume"/grav).
3. `map_scalar` call (604-608): `iv=1, kord=9`, no `q_min`, no bounds.
4. Post-transform (611): `delz = -q2*dpeO`.

## 2. Config (Stage 2)

Confirmed from `p37_c512_perf_fortran_3rep_58444510.out` (runtime log) and
`fv_arrays.F90` (module defaults):

| Parameter | Value | Effect |
|---|---|---|
| `remap_option` | 0 | `remap_t=.true.` (T remapped in logP, not `remap_pt`/`remap_te`) |
| `gmao_remap` | 0 | GFDL/`map_scalar` path used; `map1_gmao` unreachable |
| `kord_mt`/`kord_wz`/`kord_tm` | 9 | `map_scalar`→`scalar_profile` (kord>7) for all three |
| `gmao_top_bc`/`gmao_bot_bc` | `.false.` (default, unconfirmed override) | `optional_top`/`optional_bot` both absent from `map_scalar` calls |
| `hydrostatic` | `.false.` | non-hydrostatic `delz`/`w` remap paths active |
| `nwat` | 0 | see §3 MOIST_CAPPA finding |

## 3. MOIST_CAPPA/`nwat=0` finding (new, resolved — not an Opus escalation)

The real production `fvdycore` library is compiled with `-DMOIST_CAPPA
-DUSE_COND` (confirmed via `flags.make` and via `nm`/`strings` of the
production binary `StandAlone_FV3_Dycore.x`, which contains the
`moist_cv` symbol and a CUDA-side error string documenting this as an
already-known, already-documented CUDA gap — consistent with
`CURRENT_STATE.md` §7). This means the pt pre-transform/post-processing
are, in general, gated on `#ifdef MOIST_CAPPA` and use `moist_cv`/`cappa`,
not the plain `k1k`/`akap` formulas traced in §1.

However, for the validated Held-Suarez C512 configuration, `nwat=0`
(confirmed via grep of the same production log: `"FV3 nwat: 0"`).
`moist_cv`'s `case default` branch (triggered when `nwat` matches none of
cases 1-7) unconditionally returns `cvm=cv_air`. Combined with
`q(sphum)≈0` (true for this dry idealized run, no moisture
init/forcing), `cappa = rdgas/(rdgas+cv_air/(1+r_vir*0)) = rdgas/cp_air =
akap`, and `cappa/(1-cappa) = rdgas/cv_air = k1k` — algebraically
**identical** to the non-MOIST_CAPPA `#else` formulas.

**Conclusion:** for this specific validated configuration, the dry
`k1k`/`rrg`/`akap` formula used in this fixture is the exact production
value, not an approximation — contingent on `q(sphum)=0` holding, which
is well-justified for dry Held-Suarez but not independently field-
verified against a live run's tracer state. This is documented as a
caveat, not a blocker: it does not meet the "genuinely ambiguous" bar for
Opus escalation, since it was fully resolvable by source/binary
inspection plus the `moist_cv` case-default algebra.

## 4. Harness (Stage 3, Stage 9)

New target: `model/pt_delz-driver/` (mirrors `model/map_scalar-driver/`
from 52A exactly — same dependency set, same CMake structure, same
`FREAL8` R8-promotion fix).

- **Kernel call (`map_scalar`):** DIRECT AUTHORITATIVE SOURCE — calls the
  unmodified `fv_mapz_mod::map_scalar` in `../fv_mapz.F90`, not a copy.
- **Surrounding glue (`peO`/`pn1`/`pn2`/`dpn1`/`dpn2`/`dpe`/`dpeO`,
  pt/delz pre/post-transform, joint `pkz`):** TEST WRAPPER — reproduces
  the exact production formulas (§1) as new code in `main.F90`, since
  this arithmetic lives inline in `Lagrangian_to_Eulerian` and is not
  separately callable. No production source was copied or edited.
- **Module-state check:** `peO`, `dpe`, `dpeO` are all local temporaries
  declared inside `Lagrangian_to_Eulerian` itself (e.g. `real,
  dimension(is:ie,km,js:je) :: dpe`, confirmed via source read) — **not**
  module-level/persistent state. No hidden state to carry across calls.
- **Simplification documented as algebraic equivalence, not a numerical
  change:** the harness reuses `dpeO` directly as "new delp" for the
  joint `pkz` step, rather than literally re-deriving `pe2`/`delp` a
  second time via the later pe/peln/pk/delp update block — both derive
  from the identical `ak(k)+bk(k)*pe(km+1)` formula, so this is exact,
  not approximate.

## 5. Fixtures (Stage 4)

6 combined pt+delz single-column (`i1=i2=1`) cases, `km=kn=10`, shared
Lagrangian `pe` (exponent 1.5) / hybrid `ak,bk` (exponent 1.4) grid across
all cases (mirroring 52A's dual-exponent convention for a genuinely
non-trivial remap):

1. `case1_smooth_monotonic`
2. `case2_strong_gradient`
3. `case3_local_extrema`
4. `case4_near_uniform`
5. `case5_limiter_activating`
6. `case6_realistic_c512` (npz=10 standard-atmosphere-like pt/delz profile)

**Intentionally not covered, with justification:** a `gmao_remap`-active
branch (unreachable — `gmao_remap=0` confirmed in production) and any
other GMAO-only edge branch (none identified as reachable). No
unsupported artificial case was added.

## 6. Intermediates saved (Stage 5)

Each fixture (`pt_delz_baseline_output/<case>.txt`, ES25.17) records,
sufficient to localize a future CUDA divergence to the first mismatching
stage: `pe`, `peO`, `dpe`, `dpeO` per level; `pt_in`, `pt_pretransform`,
`pt_out` (remapped); `delz_in`, `delz_pretransform`, `delz_out`
(remapped); and joint `pkz_out`.

## 7. Build / run / self-checks (Stages 6-8)

Built on the login node, R8, no GPU/Slurm, using the validated stack:
`GEOSenv`, `comp/gcc/12.3.0`, `comp/intel/2024.2.0`, `mpi/impi/2021.13`,
`nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++`, from the
`GEOSfvdycore_fortran_perf` build tree (`cmake .` + `make pt_delz-driver
-j4`). Build: **PASS**, no errors/warnings from the new source.

Run (login node, module env loaded for runtime libs): all 6 cases
`OK=T`; `PT_DELZ_DRIVER: 6 cases run, 0 failed`, `RESULT=PASS`.

Self-checks, all passing for all 6 cases:
- `rc`/exit code 0.
- No NaN/Inf in any pre-transform, remapped, or `pkz` value.
- Valid pressure ordering: `pe`/`peO` strictly increasing (checked).
- Positive thickness: `dpe`/`dpeO` > 0 (checked).
- `delz_out` sign check: strictly negative (checked, matches production
  height-sign convention).
- Deterministic repeatability: two independent in-process `map_scalar`
  calls per path (pt, delz) compared bitwise — identical. Additionally
  verified bitwise-identical fixture output across two fully independent
  process invocations (`diff -rq`, no differences).

## 8. CUDA next unit (Stage 10)

The pt and delz surrounding-arithmetic can share **one** GPU kernel unit
with the existing `map_scalar` port: `peO`/`dpe`/`dpeO`/`pn1`/`pn2`/
`dpn1`/`dpn2` are shared or near-identical closed-form per-column
arithmetic for both fields, and both call the same `map_scalar`
kernel with only `iv`/`kord`/`q_min` differing. A single launcher computing
the shared pressure-grid derivation once, then dispatching both pt and
delz remaps (plus the joint `pkz` post-processing, which needs both
outputs), is the natural unit — consistent with 52C's own launcher/
dispatch recommendation (§5 of that doc).

## 9. Data residency preview (Stage 11, no implementation)

- **Host-resident inputs (already on device via existing `ExecutionContext`
  field registry):** `pe`, `peln`(`pn1`), `ak`, `bk`, `pt`, `delz`, `delp`
  — all already device-resident fields in the CUDA dyn_core state; no new
  H2D needed for these.
- **New device-side temporaries (per-column, bounded by `km`):** `peO`,
  `pn2`, `dpn1`, `dpn2`, `dpe`, `dpeO` — small, `O(km)` per column,
  computed on-device, never touch the host (consistent with the
  persistent-residency architecture rule).
- **D2H:** none required beyond existing synchronization boundaries;
  `pt`/`delz`/`pkz` remain device-resident outputs of this stage, consumed
  by later same-substep device-resident stages.
- No implementation was written; this is a preview only.

## 10. Scope / LOC classification (Stage 12)

**SMALL.** The surrounding arithmetic is closed-form, per-column,
`O(km)` work with no new algorithm — the harness (`main.F90`) is ~220
lines including 6 fixture-shape definitions, self-checks, and file I/O;
the actual production-matching arithmetic is ~40 lines. A future CUDA
launcher implementing this is expected to be similarly small, on the
order of the existing `map_scalar` launcher plus a short per-column
pressure-grid-derivation helper.

## Change budget

2 substantial changes, as scoped: (1) new host-only `pt_delz-driver`
harness (`main.F90` + `CMakeLists.txt`) under
`GEOSfvdycore_fortran_perf/.../model/pt_delz-driver/`, (2) one-line
`esma_add_subdirectories` registration in the parent
`@fvdycore/CMakeLists.txt`. No production numerical source (`fv_mapz.F90`
or any other authoritative Fortran) was modified. No CUDA source was
touched. No GPU job was run.

## Final status

**PT/DELZ REMAP FIXTURE READY — RESUME 52C CUDA IMPLEMENTATION**
