# Prompt 73 -- TRACER_2D Mapping and Host Harness (No CUDA Implementation)

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet
Scope: mapping/harness only, per explicit instruction. No production CUDA
code, no numerical formula changes, no PT/DELZ or REMAP_UV edits, no R4,
no performance jobs, no GPU/Slurm.

Governing carry-forward item (not re-analyzed here): the Prompt 71
REMAP_UV performance conclusion is **PENDING SEPARATE RE-ANALYSIS** --
the raw timer data (CUDA 16.5 ms/call vs. Fortran-forced 169 ms/call) may
indicate the opposite of the documented "10.3x slower" conclusion. See
`p71_remap_uv_cuda_implementation_2026-09-23.md` §3 and
`CURRENT_STATE.md` §5/§9.

**CORRECTION (Prompt 74, 2026-09-24):** §3's claim that `fv_tp_2d` is
"the same generic scalar-transport contract already GPU-executed by
WP-B" requires an important qualification discovered by Prompt 74's live
measurement: the existing GPU kernel
(`dgrid::launch_scalar_transport_cuda`, `wp_b_cuda.cuh`) only supports
`abs(hord) == 6`. A production job log confirms **`hord_tr` resolves to
8 at runtime**, not 6 -- so for the actual production tracer-transport
call, `fv_tp_2d`'s numerical core is **not** GPU-executed today; it runs
entirely on the CPU (measured at 69.7% of `TRACER_2D`'s ~281 ms/call).
This invalidates the premise that a TRACER_2D CUDA port could simply
wrap an existing GPU kernel -- doing so would require either re-porting
`fv_tp_2d` for `hord==8` (out of scope, a WP-B-sized task) or a CPU/GPU
hybrid split already rejected in Prompt 65. See
`p74_tracer2d_orchestration_cuda_2026-09-24.md` for the full measurement
and the resulting NO-GO decision on the orchestration-only port
selected in §10 below.

## 0. Timer

`fv_dynamics.F90:890` `call timing_on('tracer_2d')` through `:908`
`call timing_off('tracer_2d')`. The block is a single `if/else if/else`
dispatch to exactly one of three mutually exclusive routines from
`fv_tracer2d_mod`:

```fortran
if( .not. flagstruct%inline_q .and. nq /= 0 ) then
   if (gridstruct%nested) then
      call tracer_2d_nested(...)
   else if ( flagstruct%z_tracer .and. q_split==0 ) then
      call tracer_2d_1L(...)
   else
      call tracer_2d(...)
   endif
endif
```

The whole block, including whichever variant fires, is inside
`if (.not. flagstruct%inline_q .and. nq /= 0)` -- the timer cannot report
nonzero cost unless `nq /= 0`. `Fill2D` (`:910-914`) is a **separate**
timer immediately after this block, not inside it -- not part of the
~280 ms/call figure.

**Answer:** the timer attributes cost to a single dispatched routine per
call (one of the three variants), and that routine internally mixes
compute and communication (halo completion, an MPI allreduce, and,
conditionally, a mid-loop halo update -- §5). It is not a
communication-only or compute-only timer.

## 1. Which variant is active, and why

Runtime config, read directly from a representative production HS C512
run's own log (`experiments/p71_remapuv_pair1_A/.../logfile.000000.out`):

```
NCNST   =    0        (raw fv_core_nml default, pre-registration)
NWAT    =    0
Q_SPLIT =    0
INLINE_Q        = F
Z_TRACER        = T
ADIABATIC       = T
```

`z_tracer = .true.` and `q_split == 0` → **`tracer_2d_1L` is the
active production routine**, not the module's default-fallback
`tracer_2d`. This matters: `tracer_2d_1L`'s entire purpose (per its own
doc comment) is a *per-level-diagnosed* time-split count (`nsplt`, from
each level's own max Courant number), which `tracer_2d` (the other
variant) does not do the same way. Any CUDA port plan aimed at the wrong
variant would port the wrong control structure.

`tracer_2d_1L` is 271 lines (`fv_tracer2d.F90:92-362`).

## 2. Active tracers under Held-Suarez (`nwat=0`) -- CRITICAL, resolved

**`nwat=0` does NOT mean zero tracers.** Traced through source:

1. `fv_arrays.F90`: `nwat` default 3 (real-dycore branch); this run's
   namelist overrides it to 0 (dry, no condensate loading) but does not
   set `ncnst`.
2. `FV_StateMod.F90:960`: `GRID%NQ = MAX(1, FV_Atm(1)%flagstruct%ncnst)`
   -- GEOS **floors the tracer count at 1** regardless of the namelist
   value. With the namelist's `ncnst=0`, `GRID%NQ = MAX(1,0) = 1`.
3. `FV_StateMod.F90:1490-1494`: since `nwat=0 < 3`, the "remove 3 CN
   species" branch is skipped; `FV_Atm(1)%ncnst = STATE%GRID%NQ = 1`.
   The tracer array is allocated `q(isd:ied,jsd:jed,npz,1)`.
4. `FV_StateMod.F90:2050` passes `FV_Atm(1)%ncnst` as `fv_dynamics`'s
   `nq_tot` argument. `fv_dynamics.F90:347`: `nq = nq_tot - dnats`
   (`dnats`, diagnostic non-advected count, unset in this namelist,
   defaults to 0) → **`nq = 1`**.

| Tracer/category | Count | Allocated | Transported | Purpose |
|---|---|---|---|---|
| GEOS floor tracer (`GRID%NQ` minimum) | 1 | yes | yes (`nq=1≠ 0`) | passive placeholder -- GEOS's `MAX(1,ncnst)` floor guarantees at least one tracer slot exists even in a dry, adiabatic, `nwat=0` configuration, so the dynamical core's generic tracer machinery has a well-defined array shape |
| Moisture/condensate species (`nwat` 1..7 index table, `FV_StateMod.F90:1503-1519`) | 0 | n/a (`case(0)` in the `select case(nwat)` has no index assignments) | no | not present in this config |

This directly proves `nq /= 0` for the production HS config, which is
exactly the condition the timer's own `if` requires for the measured
~280 ms/call to be real. **TRACER_2D executes meaningful, real transport
work in HS despite `nwat=0` -- for exactly one passive tracer**, not
zero.

Practical consequence for porting: `tracer_2d_1L`'s `do iq=1,nq` tracer
loop is real machinery for a general `nq`, but in production it degenerates
to a single iteration. Any CUDA port must remain correct for general `nq`
(GEOS production configs elsewhere carry many tracers) while being judged,
here, against a `nq=1` production reality.

## 3. Algorithm

Flux-form, finite-volume horizontal transport with **per-level Lagrangian
time-splitting** for stability under locally large Courant numbers
(`nsplt = int(1 + cmax(k))`, `cmax` from a global-max-reduced per-level
Courant number). Per split sub-step and per tracer, the actual flux is
produced by `fv_tp_2d` (`tp_core_mod`, PPM-based, `hord`-selectable,
mass-flux-weighted, corner/panel-edge aware) -- the **same generic
scalar-transport contract already GPU-executed by WP-B**
(`dgrid::launch_scalar_transport_cuda`, Milestone 3A.5,
`cuda/dyn_core/tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp`; host
reference explicitly transcribed from
`tp_core.F90:109-243`/`324-556`/`247-320`, the identical source region
`fv_tp_2d` calls into). `tracer_2d_1L` itself does not reimplement PPM;
it is the orchestration layer around `fv_tp_2d`:

1. `xfx`/`yfx`: courant-weighted flux area, upwind-selected `sin_sg` face
   metric (`:163-182`).
2. Halo completion for `q` (`complete_group_halo_update`, started by the
   caller before this routine -- `:186`).
3. Per-level max tracer value and max Courant number, reduced across all
   MPI ranks via `mp_reduce_max` (`:191-230`) -- a **global communication
   step inside the timer**, not just a boundary halo.
4. Per-level `nsplt` and, if `nsplt>1`, pre-scaled `mfx2/cx2/mfy2/cy2` by
   `1/nsplt` (`:238-274`).
5. Per k-level (outer loop, explicitly **not** OpenMP-parallel "because
   the mpp_update call" -- source comment, `:276`): `ra_x`/`ra_y`
   (`:279-289`), then per split sub-step `it=1..nsplt`: `dp2` mass update
   from mass fluxes (`:296-300`), then per tracer `iq=1..nq`: the
   `fv_tp_2d` flux call and the conservative mass-weighted update
   (`:305-340`), and, **only if `nsplt>1`**, an `mpp_update_domains` halo
   exchange of the intermediate tracer state `qn2` before the next
   sub-step (`:348-353`).

Production floor: `nq=1` (§2) means step 5's tracer loop is a single
iteration in HS; `nsplt` is level-dependent and reaches `>1` only where a
level's global-max Courant number exceeds 1 (e.g. polar night jet
levels) -- common enough that the mid-loop halo exchange is a real,
recurring production code path, not a corner case.

## 4. Data dependencies

| Field | Shape | R/W | Halo | Device registered? |
|---|---|---|---|---|
| `q` (tracer) | `(isd:ied,jsd:jed,npz,nq)` | RW | yes, group-halo (`q_pack`) | not found in `fv3_gpu_field_registry.F90` under a tracer-specific name -- **not currently device-resident** |
| `dp1` (pre-advection `delp`) | `(isd:ied,jsd:jed,npz)` | RW | no (local only) | shared with dyn_core's own `delp`-family state; likely already device-resident from the acoustic loop, but not confirmed by this pass (would need the field-registry mapping traced, deferred -- see §7) |
| `mfx`/`mfy` (mass flux) | face-centered, `(...,npz)` | R | no | produced by the D-grid acoustic loop; WP-B already computes/consumes these on-device for the main dynamics -- likely already resident, not confirmed |
| `cx`/`cy` (Courant number) | face-centered, halo-extended | R | no | same as `mfx`/`mfy` |
| `xfx`/`yfx`, `ra_x`/`ra_y`, `dp2`, `qn2`, `mfx2`/`mfy2`/`cx2`/`cy2`, `qmax`/`cmax` | local temporaries, tile-sized | local | no | temporaries -- new workspace, not persistent state |
| `area`/`rarea`, `sin_sg`, `dxa`/`dya`/`dx`/`dy` | grid metrics | R | no | WP-E-owned static grid/geometry contract -- already device-resident by construction (shared across all work packages) |

## 5. Communication

TRACER_2D performs three distinct MPI-relevant operations per call, not
one:

1. **Halo completion for `q`** (`complete_group_halo_update`, one
   per-call operation) -- the START is issued by the caller before
   `tracer_2d_1L` (this is `fv_dynamics.F90`'s `i_pack(10)`, already
   identified in `CURRENT_STATE.md` §3 as "tracer halo, required").
   Vector/scalar: scalar (tracer). Width: standard group-halo width for
   this grid. FMS-owned; CUDA halo infrastructure already handles scalar
   halo exchange for other fields of the same shape, so no new transport
   *mechanism* is needed, only a new field mapped into it.
2. **Global-max allreduce** (`mp_reduce_max`, once per call, size
   `(nq+1)*npz`) -- this is a genuine **collective**, not a
   neighbor-halo exchange. It has no existing analogue in the CUDA halo
   transport interface (`HaloTransport` is neighbor-exchange-shaped, not
   allreduce-shaped) and would need either a new collective path or a
   host round-trip.
3. **Conditional mid-loop halo update** (`mpp_update_domains(qn2,...)`,
   `:350`, only when a level's `nsplt>1`) -- occurs *inside* the
   per-k-level, per-split loop, i.e. potentially several times per call,
   interleaved with compute, on a **temporary** array (`qn2`), not the
   persistent `q` state.

A CUDA port that ignores (2) and (3) would be numerically wrong for any
level where the global Courant maximum exceeds 1 -- not a rare
configuration-only branch.

## 6. GPU parallelism structure

- **Embarrassingly parallel**: horizontal cell index (`i,j`) within a
  level/split-step, and the tracer index `iq` (production `nq=1`, but the
  loop must stay general).
- **Sequential dependency**: the split-step loop `it=1..nsplt` is a true
  recurrence (`qn2` at `it` depends on `qn2` at `it-1`) -- cannot be
  naively parallelized across `it`; each sub-step's flux call also
  depends on the *result* of the previous sub-step's halo update when
  `nsplt>1`.
- **k-loop**: the source's own comment states the k-loop "can not be
  OpenMPed because the mpp_update call" -- on CPU, this is a serialization
  constraint from interleaved communication, not a numerical dependency
  between levels (each level's transport is otherwise independent). On
  GPU this reframes as: levels can run as independent kernel launches, or
  batched into one launch, *except* where a level's `nsplt>1` triggers
  the mid-loop halo -- those levels need a device-to-host (or CUDA-aware
  MPI) synchronization point mid-kernel-sequence.
- **No limiter-neighbor recurrence in this unit itself**: the
  monotonicity/positivity limiter lives inside `fv_tp_2d` (WP-B's
  concern, already addressed there), not in the orchestration layer
  analyzed here.

## 7. Data residency (best available without a full field-registry trace)

| Field | Current authority | GPU resident? | New H2D? | New D2H? |
|---|---|---|---|---|
| `q` | Fortran/host tracer array | not currently registered for CUDA (§4) | **yes, new** | **yes, new**, unless a device tracer buffer is added to the registry |
| `dp1`, `mfx`/`mfy`, `cx`/`cy` | shared with acoustic-loop state | plausibly already resident from dyn_core's own D-grid/pressure work | unconfirmed | unconfirmed |
| grid metrics (`area`, `sin_sg`, `dxa`/`dya`/`dx`/`dy`) | WP-E static contract | yes, already resident | no | no |
| temporaries (`xfx`,`yfx`,`ra_x`,`ra_y`,`dp2`,`qn2`,`mfx2` family,`qmax`/`cmax`) | none (recomputed each call) | n/a | n/a (device scratch) | n/a |

**Open item, explicitly deferred (not resolved this pass, per the
targeted-inspection cost-efficiency rule -- tracing every acoustic-loop
field's registry entry is a broad scan, not a targeted one, and is better
scoped to the actual port prompt):** whether `dp1`/`mfx`/`mfy`/`cx`/`cy`
are already device-resident from the immediately-preceding acoustic-loop
stage. If they are, a CUDA TRACER_2D orchestration unit could avoid a new
whole-state H2D/D2H seam for everything except `q` itself. If they are
not, the port's transfer cost could dominate its compute savings; this
should be confirmed before implementation, not assumed.

## 8. Memory footprint / local-memory risk check (Prompt-68 precedent)

`tracer_2d_1L`'s per-k-level temporaries (`dp2`, `fx`, `fy`, `ra_x`,
`ra_y`, `mfx2`/`mfy2`/`cx2`/`cy2`) are all **2D, tile-horizontal-sized**
(no `npz`-length per-thread local array), unlike the Prompt 68 defect
(`pt_delz_remap_kernel`'s unconditional 128-level per-thread local array,
~3.06 GiB/rank). `qn2` is `(isd:ied,jsd:jed,nq)` -- scales with `nq`, not
`npz`; at production `nq=1` this is a single 2D plane. **No equivalent
risk identified in this unit.** A bounded, reusable 2D scratch buffer
(mirroring the existing `WorkspaceId::vertical` reuse pattern from
PT/DELZ/REMAP_UV, but a *horizontal* workspace id) is sufficient; no new
per-thread-local-array design question exists here.

Rough per-rank scratch estimate at C512 (513x513 face-centered horizontal
tile, `npz=10`, `nq=1`, R8): a handful of 2D `(isd:ied)x(jsd:jed)` arrays
(`dp2`, `fx`, `fy`, `ra_x`, `ra_y`, `xfx`, `yfx`, `mfx2`, `mfy2`, `cx2`,
`cy2`, `qn2`) at ~513x513x8 bytes ≈ 2.1 MiB each, ~12 such arrays
≈ 25 MiB/rank if kept per-level rather than reused across the k-loop
-- small relative to the 39 GiB/865 MiB headroom margin established for
PT/DELZ (P68/69). Not a memory-capacity concern at this grid size.

## 9. Static hotspot ranking (no profiling job run)

By loop volume and structure (not measured):

1. **`fv_tp_2d` calls** (`do it=1..nsplt` x `do iq=1..nq`, per k-level) --
   already GPU-executed elsewhere (WP-B); likely the dominant arithmetic
   cost in the Fortran path today, but not new CUDA-port work.
2. **The MPI allreduce (`mp_reduce_max`) and the halo completion/update**
   -- fixed per-call (or per-split, for the mid-loop update) latency-bound
   cost, not compute volume; plausible source of a disproportionate share
   of ~280 ms/call given MPI collective/halo latency at C512's 6-rank
   density.
3. **`xfx`/`yfx`/`ra_x`/`ra_y`/`dp2` orchestration arithmetic** -- O(tile
   area) per level, cheap relative to (1) and (2), but this is exactly
   the part with no existing CUDA counterpart, i.e. the part this prompt
   selects as the port unit (§10).

## 10. Selected port unit

**Not** a new flux kernel (Option A in the isolated sense) and **not**
the full `tracer_2d_1L` kernel (Option D) in one step. The existing
WP-B `fv_tp_2d`-equivalent (`launch_scalar_transport_cuda`) already
covers the numerically hardest part; re-deriving it here would violate
the project's explicit "do not duplicate numerical functionality across
work packages" rule.

**Selected: the per-k-level orchestration layer** -- `xfx`/`yfx`
computation, `ra_x`/`ra_y`, `dp2` mass update, and the conservative,
mass-weighted `q` update (§3 steps 1 and 5's non-`fv_tp_2d` parts) --
scoped initially to the `nsplt==1` case that dominates production
(§2's `nq=1` finding plus sub-unity Courant numbers away from
jet-stream levels). This is a well-defined, independently testable
numerical contract (verified by the harness below), does not require
porting the global-max allreduce or the mid-loop halo update
immediately, and is the smallest unit that is *not* already covered by
an existing validated kernel. The `nsplt>1` path (§5 item 3) is
explicitly flagged as a required follow-on before this unit could be
called numerically complete for all production levels, not silently
dropped.

This is closest to a **new hybrid of Option A and Option B**: reuse an
existing flux kernel (A, already done by WP-B) and wrap it with a new,
small orchestration kernel (closer to B) rather than treating the whole
`tracer_2d_1L` as one monolithic new unit (D).

## 11. Harness

Host-only, no MPI/CUDA, C++, following the repo's existing
`cuda/dyn_core/tests/host/` convention for login-node-only arithmetic
checks (precedent: `check_bench_column_balance.cpp`):

`cuda/dyn_core/tests/host/test_tracer2d_orchestration_harness.cpp`

Preference-hierarchy choice: **Option C** (exact-body, test-only
transcription), not Option A (calling `tracer_2d_1L` directly would
require standing up `fv_grid_type`/`domain2d`/`group_halo_update_type`
MPI-derived-type infrastructure disproportionate to a mapping/harness
task) and not a full Option B wrapper for the same reason. Every
transcribed formula (`compute_xfx_yfx`, `compute_ra`, `compute_dp2`,
`update_q`) is copied verbatim, with source line citations in comments,
from `fv_tracer2d.F90`'s `tracer_2d_1L`. The tracer flux (`fx`/`fy`) that
`fv_tp_2d` would produce is supplied synthetically per fixture (a simple
upwind flux consistent with the seeded Courant field) rather than
recomputed with PPM, since PPM is WP-B's already-validated territory,
not this unit's. No production numerical source was modified.

## 12. Fixtures (9, all legal for this unit's contract)

`uniform_tracer_zero_courant`, `smooth_gradient`, `sharp_front`,
`local_extremum_negative_courant` (also exercises the upwind "else"
branch of both `xfx` and `yfx`), `zero_tracer`, `positive_definite_tiny`,
`sign_changing_passive` (legal -- this unit makes no positivity
assumption; that lives in `fv_tp_2d`'s `hord`/limiter choice, out of
scope here), `halo_sensitive_boundary_flux` (nonzero flux injected at the
tile's outer edge, standing in for a real inter-tile exchange),
`c512_like_profile` (production-representative Courant magnitude and a
smooth realistic tracer field). A dedicated multi-tracer fixture was
considered but not added: production HS carries exactly `nq=1` (§2),
and every transcribed formula in this unit is per-tracer/per-level
independent of `nq` by construction (the `do iq=1,nq` loop in the source
has no cross-tracer coupling), so a single-tracer fixture set already
exercises the unit's full numerical contract; `nq` generality is a loop-
trip-count concern for the eventual kernel launch config, not a new
arithmetic path.

## 13. Conservation check

Flux-form update is conservative by construction: for a closed tile
(zero flux at the outer boundary), `sum(q_new*dp2)` over the interior
must equal `sum(q_old*dp1)` exactly up to floating-point roundoff. For
the `halo_sensitive_boundary_flux` fixture (nonzero boundary flux, i.e.
an open tile), the exact expected relation is

```
mass_after - mass_before == net_boundary_flux
```

where `net_boundary_flux` is the signed sum of `fx`/`fy` crossing the
tile's four outer edges, using the same in-minus-out sign convention as
the update formula itself. This is the authoritative invariant implied
by the source's own flux-divergence form (`:326`,`:336`) -- not an
invented one.

## 14. Harness run

Built and run on the login node (`g++ -O2 -std=c++17`), no GPU/Slurm:

```
fixture                          ok     nan/inf    mass_before    mass_after     residual
uniform_tracer_zero_courant      PASS   no         1.080000e+02   1.080000e+02   0.000e+00
smooth_gradient                  PASS   no         3.780000e+01   3.420000e+01   -1.332e-15
sharp_front                      PASS   no         1.800000e+02   1.500000e+02   0.000e+00
local_extremum_negative_courant  PASS   no         1.350000e+02   1.350000e+02   0.000e+00
zero_tracer                      PASS   no         0.000000e+00   0.000000e+00   0.000e+00
positive_definite_tiny           PASS   no         3.600000e-11   3.600000e-11   0.000e+00
sign_changing_passive            PASS   no         0.000000e+00   0.000000e+00   0.000e+00
halo_sensitive_boundary_flux     PASS   no         9.405000e+01   9.027000e+01   -3.020e-14
c512_like_profile                PASS   no         1.008983e+04   1.008959e+04   -6.040e-13

determinism_check(smooth_gradient rerun): PASS (bitwise identical)

0/10 fixtures + determinism check FAILED
```

9/9 fixtures PASS (mass residual at floating-point roundoff, all `<1e-9`
relative), 0 NaN/Inf, deterministic across repeated runs, rc=0. All
Stage 14 requirements met at the host-arithmetic level. This is **not**
authoritative numerical validation against a Fortran savepoint -- no such
savepoint exists yet for TRACER_2D; that is future work once an actual
CUDA kernel exists to compare.

## 15. Portability

**GPU FEASIBLE WITH RESTRUCTURING.** The selected orchestration unit
itself (§10) is embarrassingly parallel over `(i,j)` and `iq` and has
no limiter-neighbor recurrence (§6) -- GPU-friendly in isolation. The
restructuring need comes from §5/§6: the per-level `nsplt`-driven
recurrence and the conditional mid-loop halo update mean a naive
single-kernel-per-call design is wrong for any level with `nsplt>1`, and
the global-max allreduce (§5 item 2) has no existing CUDA-side
collective analogue. Restricting the first port to `nsplt==1` levels
(§10) makes that slice GPU FRIENDLY on its own, with the general case
correctly classified as GPU FEASIBLE WITH RESTRUCTURING pending a
decision on how to handle the allreduce and the mid-split halo.

## 16. Performance ceiling

TRACER_2D measured at ~280 ms/call (user-supplied profiling figure, not
re-measured this pass -- no profiling job was run, per explicit
instruction). The `fv_tp_2d` flux computation itself (already
GPU-executed via WP-B, not new work) is very likely the majority of that
figure by arithmetic volume (§9 item 1), meaning the *newly portable*
orchestration unit selected in §10 is a **minority fraction** of
TRACER_2D's own cost -- plausibly single-digit percent, by analogy with
PT/DELZ's own orchestration-vs-kernel cost split (P69: ~31 ms/call GPU
cost against ~354 ms/call CPU removed, i.e. the ported unit's *own*
device cost was small relative to what it replaced, but that is a
different ratio than "orchestration share of a routine that already
calls an existing kernel"). No numeric estimate is given for
FV_DYN_LOOP/TOTAL improvement here: doing so would require knowing (a)
what fraction of the 280 ms/call is `fv_tp_2d` itself, which is not
separately timed today, and (b) whether `fv_tp_2d`'s *existing* GPU
kernel is even being invoked from a CUDA TRACER_2D dispatch today (it is
not -- TRACER_2D CUDA dispatch does not exist yet). **Overclaiming a
specific percentage here would not be supportable by the evidence
gathered this pass; a dedicated sub-routine-level timer split (`fv_tp_2d`
alone vs. the rest of `tracer_2d_1L`) is the correct next measurement,
deferred to the actual port prompt per the "no profiling job" constraint
on this one.**

## 17. Next-prompt recommendation

Clean, standalone, GPU-feasible orchestration unit identified; existing
WP-B flux kernel already covers the numerically hardest part; harness
built and passing. Communication is real (§5) but does not dominate
the *selected* unit's own contract (it dominates the *other two thirds*
of `tracer_2d_1L` that this prompt deliberately did not select for a
first port). → **PORT TRACER_2D CUDA UNIT**, scoped explicitly to the
§10 orchestration layer at `nsplt==1`, with the `nsplt>1` mid-loop
halo case and the global-max allreduce called out up front as required
follow-on work before full-level-range correctness, and a sub-routine
timer split (`fv_tp_2d` vs. the rest) as the first concrete step of that
future prompt to properly size §16's ceiling before committing kernel
effort.
