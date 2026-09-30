# Prompt 95 — REMAP_TOTALENERGYCON Mapping + Harness (MAP/HARNESS only, no CUDA implementation)

Model: SONNET. Branch: `optimization/cuda-fv3-performance`. No production
source changed. No commit/push.

## 0. Timer scope (Stage 0)

`Remap_TotalEnergyConsv` (`fv_mapz.F90:1612` `call timing_on(...)` through
`fv_mapz.F90:1875` `call timing_off(...)`), inline code inside
`Lagrangian_to_Eulerian`, not a standalone callable routine. Body groups
into 5 blocks by statement range:

| Block | Lines | Content |
|---|---|---|
| A | 1612-1627 | Guard/setup: `if (consv>consv_min .or. consv<-consv_min)`, `te0_2d`/`zsum0`/`zsum1` prep |
| B | 1628-1722 | `consv>consv_min` branch: per-column `te_2d`/`zsum1` accumulation (hydrostatic + non-hydrostatic sub-branches) then `tesum`/`zsum` `mpp_global_sum` + `dtmp` |
| C | 1725-1755 | `consv<-consv_min` branch — mirrors B, mutually exclusive, **DEAD in production** (`consv_te=1.0>0`) |
| D | 1760-1799 | `do_sat_adj`-gated block — unrelated to conservation, **DEAD in production** (`nwat=0<6`) |
| E | 1802-1869 | Post-reduction `pt`/`w` correction: `USE_COND`/`adiabatic`/plain three-way branch |

Only Blocks A, B (its executed sub-branches), and E (its executed
sub-branch) are numerically active in production; C and D are statically
present but never execute.

## 1. Guard / runtime-value / active-branch table (Stage 1)

| Variable | Production value | Source of confirmation |
|---|---|---|
| `consv` (te) | `1.0` (`consv_te`) | prior-session `AGCM.rc`/log |
| `consv_min` | `0.001` (compile-time) | `fv_arrays.F90` default |
| `hydrostatic` | `.false.` | prior-session log `hydrostatic: F` |
| `nwat` | `0` | prior-session log `nwat: 0` |
| `do_sat_adj` | `.false.` | prior-session log `do_sat_adj: F` |
| `do_adiabatic_init` | `.false.` | standalone harness never sets this (IAU/digital-filter only) |
| `last_step` | `.true.` (checked call) | HS end-of-substep timing convention, consistent with prior ports |
| `adiabatic` | `.true.` | this session's C512 log: "Running In Adiabatic Mode" |
| `sflag` (`flagstruct%exact_sum`) | `1` -> `BITWISE_EXACT_SUM` | this session's log `FV3 exact_sum: 1`, `fv_arrays.F90:815-817` |

Active branch selection: Block B's `consv>consv_min` (not C), its
non-hydrostatic sub-branch (1651-1691, not the hydrostatic 1642-1650), and
Block E's `last_step .and. adiabatic` sub-branch (1848-1857, not the
`USE_COND`-gated 1802-1836 or the plain-theta_v 1859-1869 branches).

## 2. Execution flow (Stage 2)

```
Block A: allocate/derive te0_2d, zsum0/zsum1 baseline
Block B (consv>consv_min, ACTIVE):
  for each (i,j) column:
    non-hydrostatic te_2d/zsum1 local compute      <- UNIT A (this harness)
  tesum   = mpp_global_sum(te_2d * area, flags=BITWISE_EXACT_SUM)   <- REDUCTION 1
  zsum    = mpp_global_sum(zsum1 * area, flags=BITWISE_EXACT_SUM)   <- REDUCTION 2
  dtmp    = tesum / (cv_air * zsum)                                  <- BRIDGE (this harness)
Block E (last_step .and. adiabatic, ACTIVE):
  for each (i,j,k):
    pt(i,j,k) = pt(i,j,k) + dtmp*pkz(i,j,k)                          <- UNIT B (this harness)
```

No cross-column dependency inside Block B's local loop or Block E's loop;
`dtmp` is the single point of global coupling.

## 3. Per-block formulas (Stage 3)

**Unit A** (fv_mapz.F90:1651-1691, non-hydrostatic, MOIST_CAPPA
case-default `cvm=cv_air` since `nwat=0`):

```
phis(km+1) = hs
phis(k) = phis(k+1) - grav*delz(k)                      for k=km..1
te_2d = sum_k delp(k) * ( cv_air*pt(k)
        + 0.5*(phis(k)+phis(k+1) + w(k)^2
               + 0.5*rsin2*(u(k)^2+u_n(k)^2+v(k)^2+v_e(k)^2
                            - (u(k)+u_n(k))*(v(k)+v_e(k))*cosa_s)) )
te_2d = te0_2d - te_2d
zsum1 = sum_k pkz(k)*delp(k)
```

Inputs: `hs`, `pt`, `delz`, `delp`, `pkz`, `u`,`u_north`,`v`,`v_east`
(D-grid edge winds), `w`, `rsin2`, `cosa_s`, `te0_2d`. Outputs: `te_2d`,
`zsum1` (both per-column, pre-area-weight, pre-reduction).

**Bridge** (fv_mapz.F90:1720, non-hydrostatic): `dtmp = tesum /
(cv_air*zsum)`. Inputs: global scalars `tesum`, `zsum` (post-reduction).
Output: global scalar `dtmp`.

**Unit B** (fv_mapz.F90:1850-1857): `pt(k) = pt(k) + dtmp*pkz(k)` for all
`k`. Inputs: `dtmp` (global scalar), `pkz` (per-column). Output: updated
`pt`.

## 4. Per-reduction-site table (Stage 4)

5 static `mpp_global_sum` call sites (1705, 1709, 1712, 1748, 1752); only 2
execute per production call (reconfirms Prompt 94):

| Site | Line | Executes? | Scalar | Flags | Consumer |
|---|---|---|---|---|---|
| `tesum` | 1705 | YES (always, `consv>consv_min`) | 1 | `BITWISE_EXACT_SUM` | `dtmp` numerator, `E_Flux` diagnostic |
| `zsum` (hydrostatic) | 1709 | NO (`hydrostatic=.false.`) | — | — | — |
| `zsum` (non-hydrostatic) | 1712/1718 | YES | 1 | `BITWISE_EXACT_SUM` | `dtmp` denominator |
| `tesum`/`zsum` (`consv<0` branch) | 1748/1752 | NO | — | — | — |

Global-result usage: `dtmp` is a single global scalar applied uniformly to
every grid column in Block E — all columns are affected, and
`BITWISE_EXACT_SUM`'s order-independent-summation design is intrinsically
reproducibility-sensitive (that is its entire purpose).

## 5. Measured pre/reduction/post split — DECISIVE FINDING (Stage 5)

Real C512 Held-Suarez runs (6 MPI ranks/1 A100, `OMP_NUM_THREADS=1`,
`GEOSfvdycore_fortran_perf` checkout, test-only `fv_timing_mod`
instrumentation, 60 calls/run):

| Sub-phase | Job (coarse) 58620131 | Job (fine) 58620343 |
|---|---|---|
| `REMAP_TOTALENERGYCON` (full timer) | 125.28 ms/call | 151.63 ms/call |
| `TECons_Pre` (Unit A local compute) | 13.81 ms/call | 13.67 ms/call |
| `TECons_Red` (reduction block, coarse) | 109.01 ms/call | 135.43 ms/call |
| `TECons_RedMPI` (mpp_global_sum calls only, fine split) | not measured | **134.99 ms/call** |
| `TECons_Post` (Unit B local compute) | 5.96 ms/call | 6.08 ms/call |

The fine split (`TECons_RedMPI` nested inside `TECons_Red`) isolates the
actual `mpp_global_sum` MPI collective from the surrounding local
`tmp_2D` area-weighting array assignment. Result: `TECons_RedMPI` accounts
for **~99.5% of `TECons_Red`** and **~89% of the entire timer**. Local
compute (`TECons_Pre`+`TECons_Post`) is only **~13%** (~19.75 ms/call of
151.63).

**Dominant: COMMUNICATION** (the `mpp_global_sum` reduction call itself,
not local per-column compute). This reverses Prompt 94 Stage 8's
assumption ("likely bottleneck: COMPUTE, not COMMUNICATION").

**Root cause**: `sflag` (passed as `flags=` to every `mpp_global_sum` call
in this timer) resolves to `BITWISE_EXACT_SUM` (`flagstruct%exact_sum=1`,
the FMS-and-project default per `fv_arrays.F90:815-817`, confirmed live via
this session's log `FV3 exact_sum: 1`). `BITWISE_EXACT_SUM` performs an
order-independent, bit-reproducible global summation algorithm inside FMS
— substantially more expensive per call than a plain `MPI_Allreduce`-based
sum — which is the well-grounded explanation for why a nominally
"2-scalar" reduction costs ~135 ms/call across 6 ranks.

## 6. Data residency (Stage 6)

All inputs (`pt`, `delp`, `delz`, `u`, `v`, `w`, `pkz`, `pk`, `hs`,
`q(sphum)`, `gridstruct%area_64`/`cosa_s`/`rsin2`) are host-resident at
entry to this timer in the current CPU-only Fortran path, same pattern
already established for REMAP_PRESSUREVARS/REMAP_Q/REMAP_UV/PT-DELZ (P91).
No new residency concern beyond the two global scalars crossing the
reduction boundary.

## 7. CUDA/MPI integration boundary (Stage 7)

Architecturally clean boundary remains: GPU precompute Unit A (`te_2d`,
`zsum1` per column) -> D2H 2 scalars per rank (post local-column
accumulation) -> unchanged host `mpp_global_sum(..., flags=
BITWISE_EXACT_SUM)` -> H2D 1 scalar (`dtmp`) -> GPU postcompute Unit B.
Host-side-MPI-only integration is feasible without touching
`mpp_global_sum` itself (satisfies "do not redesign MPI/global
reductions"). **However**, given Stage 5's measurement, this boundary
would only ever accelerate the ~13% local-compute fraction — the ~89%
reduction cost is untouched by construction.

## 8. CUDA-aware MPI optionality (Stage 8)

Not applicable/no benefit. The dominant cost is the `BITWISE_EXACT_SUM`
algorithm's own compute (order-independent summation across ranks), not
host/device transfer bandwidth of the 2 scalars involved. CUDA-aware MPI
addresses transfer latency/bandwidth, not algorithmic cost inside FMS's
reduction implementation — it would not measurably help here. No
experiments run (per task constraint: "do not redesign MPI/global
reductions").

## 9. Memory/workspace risk (Stage 9)

**LOW.** Unit A/B workspace shape (per-column scalar accumulators, no new
3D arrays) is smaller than REMAP_PRESSUREVARS's already-validated
workspace; no new OOM risk class introduced.

## 10. CUDA reuse classification (Stage 10)

Unit A's per-column formula (phis integration, KE cross term, `cv_air`
weighting) is structurally similar to REMAP_PRESSUREVARS's pressure-var
loop (P91/92) — HIGH reuse potential for the vertical-integration pattern.
Unit B is a trivial affine per-point update, similar in shape to REMAP_UV's
simplest paths. However, **the payoff ceiling is capped at ~13% of the
timer regardless of reuse quality** (Stage 5), unlike prior ports where
the entire timer was portable.

## 11. Unit A/Unit B harness split (Stage 11)

- **UNIT A**: pre-reduction per-column compute (fv_mapz.F90:1651-1691,
  non-hydrostatic + MOIST_CAPPA-degenerate `cv_air`) — produces `te_2d`,
  `zsum1`.
- **BRIDGE**: fv_mapz.F90:1720, `dtmp = tesum/(cv_air*zsum)` — pure
  global-scalar arithmetic, included as the explicit, minimal
  reduction-consumer boundary.
- **UNIT B**: post-reduction per-point compute (fv_mapz.F90:1850-1857,
  `last_step .and. adiabatic` branch, confirmed the branch actually
  executed by the measured C512 HS harness) — applies `dtmp` to `pt`.

Justification: this split isolates exactly the two GPU-clean local units
bracketing the two `mpp_global_sum` calls, matching Stage 7's proposed
integration boundary, and targets only the branches confirmed
production-active (Stage 1), not the full general-case complexity
(`USE_COND`, hydrostatic, `consv<0`).

## 12-14. Fixtures and invariants (Stage 12/13/14)

Implemented in
`cuda/dyn_core/tests/vertical/test_remap_totalenergycon_host_reference.cpp`:
11 Unit A fixtures (smooth/C512-like columns, strong lapse rate, large
wind KE, large nonhydrostatic `w`, fine vertical levels, top/bottom-terrain
sensitivity, alternate seed, positive/negative energy imbalance, multi-
column accumulation sanity), 5 Bridge fixtures (small/large correction,
small-positive denominator, negative-tesum sign-following, repeatability),
5 Unit B fixtures (`dtmp=0` identity, linearity in `dtmp`, negative-`dtmp`
cooling, repeatability), and 1 end-to-end Unit-A -> Bridge -> Unit-B
composition check. Invariants asserted: all outputs finite; `zsum1`
strictly positive (source-guaranteed positive layer mass/`pkz`); `dtmp=0`
is an exact identity on `pt`; the adiabatic correction is linear in
`dtmp`; every unit is bitwise-repeatable across two identical calls.

## 15. Harness build/run result (Stage 15)

Built host-only (no CUDA, no project headers) in `build_cuda_r8_mpi`
(OpenMPI dev stack, GCC 12.3.0) via
`cmake --build . --target fv3_wp_c_remap_totalenergycon_host_reference_test`,
registered as ctest `fv3_wp_c_remap_totalenergycon_host_reference`
(`CMakeLists.txt`, following the `fillz`/`remap_pressurevars` convention).

```
RESULT=PASS (21 cases)
```

All 21 cases PASS, including bitwise-repeatability checks and the
`zsum1>0`/finite-output invariants. (One fixture, `bridge_large_correction`,
initially used a `tesum` value too small to actually exceed `dtmp>1` given
`cv_air~=717.55`; corrected to `tesum=1e13` and reran — genuine test-fixture
arithmetic error, not a defect in the transcribed formula.)

## 16. Smallest CUDA/MPI port architecture (design only, Stage 16)

If ever revisited: one device kernel for Unit A (per-column, same
tile/launch pattern as REMAP_PRESSUREVARS) producing `te_2d`/`zsum1` device
arrays; D2H copy of the *already area-weighted, already per-rank-reduced*
2 scalars (a rank-local sum-reduction on GPU before the D2H, to avoid
transferring a full 2D field); unchanged host `mpp_global_sum` call; H2D of
the single resulting `dtmp` scalar; one device kernel for Unit B (per-point
affine update, trivial). Synchronization points: one device-to-host sync
before the MPI call, one host-to-device sync after. This design is sound
but has an intrinsically small payoff ceiling (Stage 17) — noted as a
low-priority option only, not a recommendation to implement now.

## 17. Performance ceiling estimate (Stage 17)

Local-compute fraction: ~13% (~19.75 ms/call of 151.63). Reduction
fraction: ~89% (~134.99 ms/call). Remaining overhead: ~2% (block-A
setup/scalar bookkeeping). **If the entire local-compute portion were
perfectly ported to GPU at zero marginal cost** (an optimistic upper
bound — ignoring residency management and D2H/H2D scalar-transfer
overhead), the recoverable ceiling is ~19.75 ms/call:

- vs. REMAPPING (439.70 ms/call, P94): ~4.5%
- vs. FV_DYN_LOOP (1817.05 ms/call, P94): ~1.1%
- vs. TOTAL (4044.68 ms/call, P94): ~0.5%

This ceiling is small relative to every prior successful port (PT/DELZ
−7.3% TOTAL, REMAP_PRESSUREVARS −1.81% TOTAL, REMAP_Q −1.51% TOTAL) despite
those ports' own timers being comparable in raw size to this one — because
in this timer, unlike those, the great majority of wall time is
`mpp_global_sum` itself, which a CUDA port of the *local* compute cannot
touch.

## 18. GO/NO-GO decision (Stage 18)

**BLOCKED BY COMMUNICATION.** The task's own explicit stop criterion
("STOP if reductions dominate the timer") is met: `mpp_global_sum`
(`BITWISE_EXACT_SUM`) accounts for ~89% of `REMAP_TOTALENERGYCON`'s
measured cost, and redesigning that reduction is out of scope for this
project track ("do not redesign MPI/global reductions"). The
local-compute-only port described in Stage 16 remains architecturally
available but is capped at an estimated ~0.5% TOTAL ceiling — a small,
low-priority residual option, not a recommended next CUDA-implementation
target.

## Final Report

- **Model**: SONNET
- **TIMER**: `Remap_TotalEnergyConsv`, `fv_mapz.F90:1612-1875`
- **PRODUCTION BRANCH**: `optimization/cuda-fv3-performance`
- **EXECUTED REDUCTIONS**: 2 of 5 static `mpp_global_sum` sites (`tesum`,
  `zsum`, non-hydrostatic branch), `flags=BITWISE_EXACT_SUM`
  (`flagstruct%exact_sum=1`)
- **MEASURED SPLIT**: Pre-compute ~13.7 ms/call (9%), Reduction ~135.0
  ms/call (89%), Post-compute ~6.1 ms/call (4%). **Dominant: COMMUNICATION**
- **DATA RESIDENCY**: host-resident at entry, same pattern as prior ports
  (§6)
- **CUDA/MPI BOUNDARY**: GPU precompute -> D2H 2 scalars -> unchanged host
  `mpp_global_sum` -> H2D 1 scalar (`dtmp`) -> GPU postcompute; feasible but
  bounded (§7/§16)
- **MEMORY**: LOW risk (§9)
- **CUDA REUSE**: HIGH structural similarity to REMAP_PRESSUREVARS/REMAP_UV
  patterns, but payoff-capped (§10)
- **HARNESS**: `test_remap_totalenergycon_host_reference.cpp`, 21/21 PASS,
  bitwise-repeatable, all invariants hold (§15)
- **PORT SCOPE**: Unit A (pre-reduction per-column) + Bridge (global-scalar
  formula) + Unit B (post-reduction per-point), excluding the
  `mpp_global_sum` calls themselves (§11)
- **PERFORMANCE CEILING**: ~19.75 ms/call recoverable at best (~0.5% of
  TOTAL, ~4.5% of REMAPPING) — small relative to every prior port (§17)
- **DECISION**: BLOCKED BY COMMUNICATION
- No production source changes: YES
- No commit/push: YES

`REMAP_TOTALENERGYCON MAPPED — BLOCKED BY COMMUNICATION`

## Addendum (Prompt 96, 2026-09-28)

Prompt 96 reconfirmed this decision unchanged: the `BITWISE_EXACT_SUM`
reduction cost (~135 ms/call) is still the dominant term, and no new
evidence contradicts the BLOCKED BY COMMUNICATION verdict. Prompt 96 also
found that Prompt 94's Stage 8 framing (expecting the next hotspot to be
found *inside* `DYN_CORE`/`FV_DYN_LOOP`) was incomplete: a separate,
previously-unexamined MAPL_Profiler GridComp-level timer tree
(`DynCore_GridCompMod.F90`) revealed `DYN_EPILOGUE` (688.67 ms/call,
MPI-free) as a larger, orthogonal hotspot living *outside*
`fv_timing_mod`'s scope entirely. See
`docs/performance/p96_remaining_dyncore_hotspot_decomposition_2026-09-28.md`
for the full decomposition and the selected next target (`FV_To_State`,
`FV_StateMod.F90:2704-2780`, MAP/HARNESS).
