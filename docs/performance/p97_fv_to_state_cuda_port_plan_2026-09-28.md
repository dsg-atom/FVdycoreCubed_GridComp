# Prompt 97 — Map and Harness `FV_To_State` for CUDA Porting

Date: 2026-09-28
Task type: MAP + HARNESS only. No production CUDA implementation. No
selector/default changes. No commit/push.

## Stage 0 — Location and call sites

`FV_To_State` is defined at
`FV_StateMod.F90:2704-2780` (GEOSfvdycore_fortran_perf checkout,
`src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90`).

Two call sites, found by direct grep + read (not inferred from the timer
name):

- **Line 1148**, inside `Initialize`, gated by
  `COLDSTART .and. grid_type==4` (doubly-periodic idealized IC
  generation). **DEAD in production** — production always uses
  `grid_type=0` (cubed sphere).
- **Line 2459**, inside `FV_Run`, **unconditional**, the *last statement*
  of `FV_Run` before `RETURN_(ESMF_SUCCESS)`, immediately after the
  `PUSH_TRACERS` timer block closes. This is the sole live production
  call site.

**Invocation count: exactly once per `FV_Run` call (once per dynamics
timestep), unconditionally.** Not multiple, not conditional in
production.

State before the call: every FV3-internal prognostic array
(`FV_Atm(1)%u/v/w/pt/delz/pe`) has just been finalized for this timestep
by the preceding tracer-transport/remap/PUSH_TRACERS work inside
`FV_Run`. State after: `STATE%VARS%U/V/W/PE/PT/DZ/PKZ` are populated for
MAPL-side consumption; `FV_Run` returns immediately after.

Consumer: `DynCore_GridCompMod.F90`'s `DYN_EPILOGUE` block (see Stage 5).

## Stage 1 — Full body, operation categorization

Body (production-active branches only; see Stage 2 for guards):

| Block | Source lines | Category | Estimated work |
|---|---|---|---|
| U copy | 2727 | A (copy) | 1 field, N3 elements |
| V copy | 2728 | A (copy) | 1 field, N3 elements |
| W copy (hydrostatic=.false. branch) | 2730-2734 | A (copy) | 1 field, N3 elements |
| PE transpose (SW_DYNAMICS=.false. branch) | 2741-2745 | B (layout transpose) | 1 field, N_pe elements, poor stride both sides |
| PT copy | 2753 | A (copy) | 1 field, N3 elements |
| DZ copy (hydrostatic=.false. branch) | 2759-2763 | A (copy) | 1 field, N3 elements |
| `fv_getPKZ` call | 2769 | D (derived compute: log/exp) | N_pe elements ×2 transcendentals + N3 divide |
| PT = PT/PKZ | 2773 | D (derived compute: divide) | N3 elements |
| `MAPL_TimerOn`/`Off` | 2725, 2775 | G (bookkeeping) | negligible |

No Category C (unit/scaling — none present), E (masking/branching beyond
the two guards below), or F (allocation) blocks exist in this routine.
`delp` and all tracers are **not touched at all** — confirmed by direct
source read; this routine copies/derives only U, V, W, PE, PT, DZ, PKZ.

## Stage 2 — Guards

| Guard | Current runtime value | Active code |
|---|---|---|
| `hydrostatic` (`FV_Atm(1)%flagstruct%hydrostatic`) | `.false.` (production nonhydrostatic config) | W/DZ copy branches (2730-2734, 2759-2763) active; the `else` zero-fill branches are DEAD |
| `SW_DYNAMICS` | `.false.` — only set `.true.` when `FV_Atm(1)%flagstruct%npz==1` (FV_StateMod.F90:411); never true for any production or performance-track config (npz=10 this L10 perf track, npz=79 full production) | PE-transpose/PT/DZ/`fv_getPKZ`/PT-divide block (2741-2774) active; the `if (SW_DYNAMICS)` phis-seeded PE block (2737-2739) is DEAD |

Confirmed structurally inapplicable to this routine (grepped directly
inside lines 2704-2780, zero matches): `MOIST_CAPPA`, `USE_COND`,
`do_omega`, `a_imp`, `nwat`, tracer count/`nq`, `grid_type` (only relevant
to the *other*, dead call site), halo/bounds flags, GMAO-specific
options. `fv_getPKZ` as called here (FV_StateMod.F90:3078, 2-arg
`fv_getPKZ(pkz,pe)`) is the plain dry-`akap` formula — the moist
`fv_getPKZ_NH(temp,qv,pe,delz)` variant (line 3035) is a different
subroutine, never called from `FV_To_State`.

## Stage 3 — Field mapping table

| FV source field | Source shape/order | GEOS/MAPL destination | Dest shape/order | Operation |
|---|---|---|---|---|
| `FV_Atm(1)%u(isc:iec,jsc:jec,:)` | (i,j,k), k=1..npz | `STATE%VARS%U` | (i,j,k) | Copy (A) |
| `FV_Atm(1)%v(isc:iec,jsc:jec,:)` | (i,j,k), k=1..npz | `STATE%VARS%V` | (i,j,k) | Copy (A) |
| `FV_Atm(1)%w(isc:iec,jsc:jec,:)` | (i,j,k), k=1..npz | `STATE%VARS%W` | (i,j,k) | Copy (A) |
| `FV_Atm(1)%pe(i,:,j)` (per-column) | **(i,k,j)**, k=1..npz+1 (fv_arrays.F90:1472) | `STATE%VARS%PE(i,j,:)` | (i,j,k) | **Transpose (B)** |
| `FV_Atm(1)%pt(isc:iec,jsc:jec,:)` | (i,j,k), k=1..npz | `STATE%VARS%PT` | (i,j,k) | Copy (A), then overwritten by divide |
| `FV_Atm(1)%delz(isc:iec,jsc:jec,:)` | (i,j,k), k=1..npz | `STATE%VARS%DZ` | (i,j,k) | Copy (A) |
| (derived from `STATE%VARS%PE`) | — | `STATE%VARS%PKZ` | (i,j,k), k=1..npz | Derived compute (D) |

No units/scaling conversion occurs anywhere in this routine (all
quantities pass through in native FV3 units: m/s, Pa, K, m). No explicit
halo is read (`isc:iec,jsc:jec` interior-only slices on every source
field; `pe`'s own halo, `is-1:ie+1`, is never indexed since the loop
bound is `i=isc,iec`). Tracer dimension: not present, not touched by
this routine.

## Stage 4 — Residency classification

| Field | Entry state | Evidence |
|---|---|---|
| `FV_Atm(1)%u/v/w/pt/delz` | **HOST AUTHORITATIVE** | `cuda/interface/fv3_gpu_dyncore_bridge.F90` (lines 1-100, this repo): each of `fv3_gpu_interface_step_dyn`, `fv3_gpu_interface_pt_delz_remap`, `fv3_gpu_interface_remap_uv`, `fv3_gpu_interface_remap_q`, `fv3_gpu_interface_remap_pressurevars` ends with its own `fv3_gpu_sync_to_host` (D2H) call, specifically because intervening Fortran-side tracer-advection/remap work "rewrites delp/pt/u/v/w/delz/pe/pk/pkz on the host" between successive CUDA dispatches. By the time `FV_Run` reaches `FV_To_State`, these fields are already resident on the host from the last such sync. |
| `FV_Atm(1)%pe` | **HOST AUTHORITATIVE** | Same evidence; `pe` is explicitly named among the fields re-synced to host by the remap dispatch calls. |
| `STATE%VARS%*` (destination) | Host memory required immediately | `T_FVDYCORE_VARS` (FV_StateMod.F90:145-156) is a plain Fortran-pointer struct with no device shadow/registry entry; its only consumer (`DYN_EPILOGUE`, Stage 5) is host/MAPL code. |

**Key question answered: YES** — FV fields are already host-resident at
`FV_To_State` entry, *because* the existing CUDA dyn_core/remap dispatch
calls each perform their own D2H as part of their per-call
synchronization protocol (this is one layer up from `FV_To_State`
itself, inside WP-E's dispatch wrappers).

## Stage 5 — Consumer chain

Direct source read of `DynCore_GridCompMod.F90` confirms the exact
sequence:

```
MAPL_TimerOn("-DYN_CORE")   [line 4462]
  call DynRun(...)          [line 4464; DynRun => FV_Run, includes FV_To_State as its last statement]
MAPL_TimerOff("-DYN_CORE")  [line 4468]
MAPL_TimerOn("-DYN_EPILOGUE") [line 4470]
  Write_Profile(grid, vars%u, 'U-after-DynRun')  [line 4482]
  Write_Profile(grid, vars%v, 'V-after-DynRun')  [line 4483]
  ... getAllWinds/getVorticity/getDivergence/getEPV/getOmega (Prompt 96) ...
MAPL_TimerOff("-DYN_EPILOGUE") [line 5473]
```

`STATE%VARS%U/V` (and by construction the rest of `STATE%VARS`) are read
**immediately** upon return from `DynRun`, entirely by host/CPU/MAPL
diagnostic code. There is no GPU consumer anywhere in this chain.

**Critical finding**: since the sole immediate consumer of
`FV_To_State`'s output requires host-resident data right away, a GPU
port of `FV_To_State` itself would need to (a) upload already
host-resident source fields (H2D), (b) transform on-device, (c)
download the result back (D2H) to satisfy `DYN_EPILOGUE` — adding a full
transfer round trip around work that is already bounded by host-to-host
data movement, not eliminating any of it.

## Stage 6 — Cost decomposition (ANALYTICAL ESTIMATE, not measured)

**Decision: no new Slurm run was performed for this stage.** Rationale
(bounded-debugging stop condition): the residency finding in Stage 4/5
is dispositive for the Stage 16 GO/NO-GO decision regardless of the
exact ms-level split between copy/layout/compute — even in the
best case (100% of the "compute" portion is data-parallel and GPU-fast),
the mandatory H2D+D2H round trip described above negates the benefit,
because source is already host-resident and destination must return to
host immediately. Running a fresh C512 job to refine this split would
not change the Stage 16 verdict; it is deferred as unnecessary evidence
gathering, consistent with the project's bounded-debugging discipline
(cf. Prompt 96's decision to skip a fresh `nsys` run once the timer
hierarchy alone was dispositive).

The byte-volume-weighted estimate below (Stage 7 data feeds this table)
is presented as an ESTIMATE, explicitly labeled, not a measurement:

| Block | Est. MB moved/call | Est. ms/call (byte-proportional) | % FV_To_State |
|---|---|---|---|
| Copies (U,V,W,PT-copy,DZ) | ~209.7 | ~40.6 | ~40% |
| Layout/reindex (PE transpose) | ~46.1 | ~8.9 | ~9% |
| Numerical compute (`fv_getPKZ` log/exp + PT divide) | ~263.1 | ~51.0 | ~51% |
| **Total** | **~518.9** | **~100.5 (matches measured baseline)** | 100% |

Caveat: the "compute" block's true wall-time share is likely *larger*
than its byte share, because `log`/`exp` are latency-bound transcendental
ops, not pure loads/stores — the byte-proportional split under-weights
it. This does not change the classification below.

**Classification: MIXED** (copy/layout ~49%, compute ~51%, with the
important caveat that even the "compute" fraction operates on data that
must be host-resident before and after — see Stage 4/5/14).

## Stage 7 — Data volume

Grid (confirmed from job 58619479's `AGCM.rc`, this project's standing
C512 L10 performance-track config): `IM_WORLD=512`, `NX=1`/`NY=6` (one
full cubed-sphere face per rank, 512×512 interior columns per rank),
`npz=10` (11 `AK:` eta-edge values). REAL8 (8 bytes/element).

| Field | Elements/rank | Bytes/rank |
|---|---|---|
| U, V, W, PT, DZ, PKZ (npz=10) | 512×512×10 = 2,621,440 | 20,971,520 (~20.97 MB) each |
| PE (npz+1=11) | 512×512×11 = 2,883,584 | 23,068,672 (~23.07 MB) |

Total estimated host memory traffic (read+write, all blocks): **~518.9
MB/call/rank** (Stage 6 table). Effective CPU bandwidth = 518.9 MB /
100.50 ms ≈ **5.16 GB/s** (single OMP thread per rank, per job
58619479's `1threads` config) — a plausible, moderate figure for
strided/non-vectorized Fortran array-slice code (well below typical
single-core DDR peak of ~10-20 GB/s, consistent with the PE transpose's
poor stride pattern on both sides and `fv_getPKZ`'s multiple full-array
passes). This supports a **memory-traffic-bound**, not
compute-throughput-bound, classification for the routine as a whole.

## Stage 8 — CUDA architecture options (design only, no implementation)

| Option | Transfers required | Host work removed | Sync | Complexity |
|---|---|---|---|---|
| A. GPU kernel transform + D2H | H2D of ~107 MB (u,v,w,pt,delz,pe) + D2H of ~130 MB (U,V,W,PE,PT,DZ,PKZ) every call | All of it, *in principle* | Full sync each call (destination needed immediately by `DYN_EPILOGUE`) | New transpose kernel + reuse of existing pk/peln kernel pattern (medium) |
| B. Device-resident source + one coarse D2H before transform | N/A — **source is not device-resident at this point** (Stage 4); this option requires a residency fact that does not hold here | None (moot) | N/A | Not applicable — precondition false |
| **C. No port** | None beyond current | None removed, none added | Unchanged | None |

Option B is ruled out directly by the Stage 4 finding, not by
assumption. Option A is technically implementable but adds a transfer
round trip around work whose endpoints are both host-bound already.

## Stage 9 — Reuse analysis

If ever revisited: H2D/D2H would **REUSE DIRECTLY** the existing
`FieldRegistry`/`fv3_gpu_sync_to_host`/`sync_from_host` machinery already
used by every other CUDA remap stage (WP-E). A PE-transpose kernel and a
`fv_getPKZ`-equivalent log/exp kernel would **REUSE WITH SMALL
EXTENSION** of the pattern already established in
`fv3_pt_delz_remap.cu`/`fv3_map_scalar_kernel.cu` (both already contain
pk/peln-style transcendental kernels from the PT/DELZ and
REMAP_PRESSUREVARS ports). No new architecture or new persistent-state
system would be required *if* this were ported — but Stage 16 does not
recommend porting.

## Stage 10 — Communication re-verification

Direct full-body read of `FV_StateMod.F90:2704-2780` (reproduced in
Stage 1): **zero** occurrences of `mpp_`, halo-exchange calls, or
collective/reduction calls of any kind.

- MPI: **NO**
- Halo: **NO**
- Collectives: **NO**

This directly re-confirms (not merely cites) Prompt 96's finding.

## Stage 11-13 — Harness

**Strategy: C (exact transcription).** `FV_To_State` is a standalone
Fortran subroutine (unlike `Remap_PressureVars`/`Remap_TotalEnergyConsv`,
which are inline code), but its arguments are
`type(MAPL_MetaComp)`/`type(T_FVDYCORE_STATE)` pointers — a direct call
(Option A) would require building a live MAPL grid-component environment,
which every prior harness in this project has deliberately avoided. A
minimal wrapper (Option B) does not remove that dependency. Option C
follows the established convention (`test_remap_pressurevars_host_reference.cpp`,
`test_remap_totalenergycon_host_reference.cpp`).

New file: `cuda/dyn_core/tests/host/test_fv_to_state_host_reference.cpp`
(host-only, no CUDA/project headers, registered in `cuda/dyn_core/CMakeLists.txt`
as `fv3_wp_c_fv_to_state_host_reference_test`).

Fixtures (8 implemented; 2 of the 10 requested categories are
structurally N/A for this routine and documented as such rather than
silently omitted):

1. `smooth_3d_field` — smooth sin/cos U/V/W, smooth PT/DZ, hybrid-sigma PE.
2. `varying_vertical_profile` — PT/DZ vary strongly with k (flat in i,j); exercises `fv_getPKZ`'s k/k+1 indexing.
3. `horizontal_gradient` — PT/DZ vary linearly in i,j (flat in k).
4. `boundary_columns_no_special_path` — confirms no separate code path exists at i=0/ni-1, j=0/nj-1 (the routine has no halo-adjacent branching to miss).
5. `uvw_active_nonhydrostatic` — the only production branch (hydrostatic=.false.).
6. `pt_delz_strong_contrast` — stresses the two fields FV_To_State actually owns thermodynamically.
7/8. q/tracers, nonzero sphum — **N/A, documented in the harness source**: `fv_getPKZ` as called here has no moisture dependence and no tracer array is referenced anywhere in the routine.
9. `production_shaped_reduced_dims` — ni=6,nj=5 (reduced from 512×512), npz=10 (exact match to the confirmed C512 L10 vertical resolution, not an arbitrary reduction).
10. `sentinel_layout_transposition_check` (**mandatory**) — `pe_iKj(i,k,j)` encoded as a bijective, position-dependent value; U/V/W/DZ likewise position-encoded. Any accidental axis swap in the transpose would fail this fixture immediately.

Invariants checked per fixture: bitwise-copy exactness for U/V/W/DZ;
exact transpose match for PE (`out.PE(i,j,k) == src.pe(i,k,j)`);
PE monotonicity in k; 0 NaN/Inf anywhere; physically-plausible PKZ range
(0, 200) — PKZ here is `pe^kappa` in Pa directly (no reference-pressure
divide), so realistic values range roughly 1-30, not O(1); bitwise
repeatability across two independent invocations.

**Result: compiled with `g++ -O2 -std=c++17` (host-only, no GPU/Slurm
needed), run directly on the login node:**

```
smooth_3d_field                            PASS
varying_vertical_profile                   PASS
horizontal_gradient                        PASS
boundary_columns_no_special_path           PASS
uvw_active_nonhydrostatic                  PASS
pt_delz_strong_contrast                    PASS
production_shaped_reduced_dims             PASS
sentinel_layout_transposition_check        PASS
RESULT=PASS (8 cases)
```

**8/8 PASS, 0 NaN/Inf, bitwise repeatable, sentinel-layout check PASS.**

## Stage 14 — CUDA feasibility

Coverable fraction (in isolation, ignoring transfer overhead): ~100% of
the arithmetic/layout work is data-parallel and could run on a GPU.
Unavoidable host fraction: the *transfer* boundary, not the compute —
100% of the input must already exist on host (Stage 4) and 100% of the
output must return to host before `DYN_EPILOGUE` runs (Stage 5).
Required D2H: none avoided (source already host-resident, so a GPU port
would need a **new H2D** of ~107 MB, not save an existing D2H). Required
D2H after compute: ~130 MB, mandatory, every call. Likely kernel type: a
handful of small elementwise/transpose/log-exp kernels, low arithmetic
intensity. Expected launch count: ~4-6 kernels/call (U/V/W copy could be
fused or done via `cudaMemcpy`, PE transpose as one kernel, `fv_getPKZ`
as 2-3 kernels).

**Hard question, answered explicitly: would CUDA remove work, or merely
move the same memory copy through PCIe/NVLink plus synchronization?**
**It would merely move the work, and add a new transfer stage on top.**
Source is already host-resident (not device-resident) at this point in
the timestep, and the destination must be host-resident again
immediately afterward. A GPU port adds one H2D (~107 MB) and one D2H
(~130 MB) — neither of which is currently needed, since today's routine
is entirely host-to-host — around a compute portion that is a minority
(Stage 6) of the routine's own already-small cost.

## Stage 15 — Performance ceiling

Baseline: ~100.50 ms/call (P94 measurement, job 58619479).

- Estimated direct speedup: **not favorable** — even optimistic
  best-case on-device compute (5-10x faster than host for the log/exp
  portion) is likely offset or exceeded by the new mandatory H2D+D2H
  round trip (~237 MB combined at typical PCIe Gen4 effective bandwidth
  ~15-20 GB/s ≈ 12-16 ms added, plus kernel-launch/sync overhead) layered
  on top of unavoidable host-side data assembly. Net direct-routine
  speedup plausibly in the 0.8x-1.3x range — i.e., possibly slower.
- Estimated `FV_DYNAMICS`/`TOTAL` improvement: **negligible to
  slightly negative.** `FV_To_State` is already only ~2.6% of P94's
  measured `TOTAL` (100.50 / 3836.74 ms). Even a generous 2x *direct*
  speedup nets under 1.3% `TOTAL` improvement before subtracting the new
  transfer overhead described above; a realistic outcome is ~0% or a
  small regression.

## Stage 16 — GO/NO-GO

**STOP criteria met**: source fields are already host-resident (no
device-resident source to exploit — Option B is structurally
unavailable); the destination must be host-resident immediately for a
purely host/MAPL consumer (`DYN_EPILOGUE`); a GPU port would add
transfer/synchronization overhead without eliminating any host work;
expected end-to-end (`TOTAL`) payoff is negligible-to-negative even in
an optimistic case. The **PROCEED** criteria (>50% meaningfully
GPU-coverable *net of transfer costs*, material end-to-end payoff) are
**not met** — the routine is coverable in isolation but not net of the
mandatory round trip its own residency constraints impose.

**Decision: KEEP ON CPU — MEMORY/STATE BOUND.** `FV_To_State` is
fundamentally a host-to-host state-materialization routine: it exists
*because* GPU-resident state must periodically be made host-visible for
MAPL, and it runs at exactly the point in the timestep where that
materialization has already happened (one layer up, inside the existing
CUDA dispatch wrappers' own D2H calls). Porting it to CUDA would not
remove that materialization step — it would insert a redundant
upload/download pair around it.

## Cost/benefit summary vs. prior targets

Unlike `REMAP_TOTALENERGYCON` (blocked by a genuine MPI collective,
P95) or `TRACER_2D` (blocked by a measured >=5x gate failure, P83),
`FV_To_State` is blocked by a **residency/architecture** fact discovered
one layer up in this project's own existing CUDA bridge code, not by an
external library or a failed optimization attempt. This is a distinct
and useful category of NO-GO for future target selection: **routines
that sit immediately adjacent to an existing H2D/D2H sync boundary are
poor CUDA-port candidates**, because the sync boundary itself already
did the residency work any new port would need to redo.

## Cross-reference — Prompt 98 (`State_To_FV`, 2026-09-28)

Prompt 98 mapped the structurally-opposite-direction routine,
`State_To_FV` (`FV_StateMod.F90:2519-2702`, called before
`fv_dynamics()` rather than after), and reached the **opposite**
decision: GO. The key distinguishing fact is residency direction —
`State_To_FV`'s output feeds an already-GPU-resident `fv_dynamics()`
CUDA path (a port there could remove a transfer), whereas
`FV_To_State`'s output feeds the host-only `DYN_EPILOGUE` (a port here
would add one). Measured cost-split also differs: `State_To_FV` is
~84% local-compute (favorable), not blocked by the residency argument
that governs this routine. See
`docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md`.

**Update 2026-09-28 (Prompt 99):** `State_To_FV`'s Option-A CUDA port
was implemented and validated correct at C32, but C512 integration hit
a device out-of-memory blocker at the 6-ranks-per-1-A100 production
density (unrelated to either routine's residency argument — a device
memory *capacity* limit, not a correctness or architecture finding).
See `docs/performance/p99_state_to_fv_cuda_implementation_2026-09-28.md`.
This does not change `FV_To_State`'s own KEEP-ON-CPU decision above.
