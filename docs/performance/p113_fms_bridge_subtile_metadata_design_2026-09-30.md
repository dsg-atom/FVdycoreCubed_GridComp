# Prompt 113 — FMS Decomposition Metadata → CUDA Bridge Integration Design

Status: **DESIGN/MAPPING ONLY — CLOSED.** No bridge ABI change, no gate
relaxation, no production 24-rank run, no corner transport, no numerical/
seam-math change, no CUDA-aware MPI, no production-default change, no R4
change, no commit/push. All work is source reading (grep/read only) plus
this document and small pointer updates in `p112`/`p108`/the roadmap/
`CURRENT_STATE.md`.

## 0. Scope note vs. Prompt 108/112

Prompt 108 already found and stated the core FMS `pe_start`/`layout(1)*
layout(2)`/tile-major-numbering facts (§4 of that doc) as part of its C1024
architecture mapping. This prompt re-verifies those facts by direct source
read (confirmed, unchanged), extends them with facts Prompt 108 did not
capture (`Atm%layout` is a *retained* pointer-aliased field, not just a
transient `domain_decomp` argument; the bridge already reads `Atm%tile`
and already receives the full `Atm` object at its call site; a second,
previously-undocumented one-rank-per-tile gate lives inside
`build_rank_of_tile` itself, not only in the top-level `comm_size==6`
check), and answers the bridge-integration-specific questions (ABI shape,
call-site readiness, gate design) that Prompt 108 explicitly deferred as
"Phase 6"/out of scope.

## 1. Production decomposition call chain (Stage 1)

| Step | File / function | Metadata created |
|---|---|---|
| Namelist `fv_core_nml` reads `layout(2)`, `io_layout(2)` | `fv_control.F90:665` | initial `layout(1:2)` guess (0,0 = "auto") |
| `layout => Atm%layout` (pointer alias) | `fv_control.F90:1333` | the namelist variable **is** `Atm%layout` from this point on — no copy |
| `call domain_decomp(npx,npy,ntiles,grid_type,nested,Atm(n),layout,io_layout)` | `fv_control.F90:967` | entry into FMS decomposition |
| `select case(nregions)`, `case(6)` (cubed-sphere, non-nested, the production path) | `fv_mp_mod.F90:464-494` | `npes_per_tile = npes_x*npes_y` (or `npes/6` if unset); `call mpp_define_layout(...,npes_per_tile,layout)`; `layout = (/npes_x,npes_y/)` written back through the `intent(INOUT)` argument, i.e. into `Atm%layout` directly |
| `pe_start(n) = pelist(1)+(n-1)*layout(1)*layout(2))`, `pe_end(n) = pe_start(n)+layout(1)*layout(2)-1`, for `n=1..6` | `fv_mp_mod.F90:505-506` | tile-major contiguous PE ranges — **confirmed by source, not inference** |
| `tile_id(n) = n` (non-nested) | `fv_mp_mod.F90:645-648` | face id 1..6 fed to FMS mosaic |
| `mpp_define_mosaic(global_indices, layout2D, domain, 6, 12, tile1, tile2, ..., pe_start=pe_start, pe_end=pe_end, ...)` | `fv_mp_mod.F90:650-656` | builds the real FMS `domain2D`, 12 hardcoded cubed-sphere contact lines (`fv_mp_mod.F90:577-628`) |
| `Atm%tile = (gid-pelist(1))/npes_per_tile+1` | `fv_mp_mod.F90:662` | **the live, authoritative face-id derivation actually used in production** |

Production first-run C512 values today: `layout=(1,1)`, `npes_per_tile=1`,
`comm_size=6` (one rank per face — the only configuration the CUDA bridge
accepts). Target C1024+ values per Prompt 108: `layout=(2,2)`,
`npes_per_tile=4`, `comm_size=24`. Nothing above is layout-value-specific —
the same `case(6)` code path already handles `layout=(2,2)` for the
Fortran backend today; only the CUDA bridge is narrower than FMS itself.

`case(6)` in the *second* `select case(nregions)` (contact-line table,
`fv_mp_mod.F90:577`) is the true cubed-sphere branch; the standalone
`case(6)` noted in the prior session inside the *first-select*'s
`grid_type` dispatch ("latlon strip") is a different, unrelated `case`
under a different `select case(grid_type)` for `nregions==1`
(non-cubed-sphere lat-lon test configs) and does not apply to production —
this resolves the prior session's open ambiguity.

## 2. Authoritative runtime source for layout_x/layout_y (Stage 2)

**`Atm%layout(1)` / `Atm%layout(2)`.** Confirmed by source
(`fv_arrays.F90:1287`, `integer :: layout(2), io_layout(2)`, doc comment
"Processor layout on each tile") and by the pointer-alias chain above
(`fv_control.F90:1333`, `layout => Atm%layout`) — the namelist value and
every `domain_decomp` mutation land directly on the persistent `Atm`
derived-type field, not a local copy. Any code holding a reference to
`Atm` after `fv_control.F90`'s init completes can read `Atm%layout(1:2)`
directly.

A second, independent path also exists and is already used elsewhere in
this exact codebase: **`mpp_get_layout(Atm%domain, layout)`**, an FMS
query API, called at `fv_io.F90:642` and `:787`. Either path is
authoritative; `Atm%layout` requires no additional FMS call (already
resident), while `mpp_get_layout` is the query-API alternative Question 4
of this prompt asked to be evaluated (see §7 below) — both agree by
construction since `mpp_get_layout` reads the same `domain2D` that
`domain_decomp` built from `Atm%layout` in the first place.

## 3. Face/tile-id source (Stage 3)

**`Atm%tile = (gid-pelist(1))/npes_per_tile+1`** (`fv_mp_mod.F90:662`),
1-based, `gid = mpp_pe()` (confirmed at `fv_mp_mod.F90:253`, inside
`setup_master`). This is the *only* topology-relevant field the CUDA
bridge already reads (`bridge_tile = Atm%tile`,
`fv3_gpu_dyncore_bridge.F90:300`) — already 1-based, already directly
consistent with `EdgeNeighbor.tile`'s 1..6 physical-face convention
(Prompt 112 semantics). No translation needed for face id; the gap is
entirely in *layout*/*rank-within-tile*, not face id.

## 4. Rank-within-tile → (sub_i, sub_j) convention (Stage 4)

**Not verifiable against literal FMS source in either project repository
— FMS itself is a precompiled external dependency here** (confirmed:
`find` for `mpp_domains*.F90` under `GEOSfvdycore_fortran_perf` returns
nothing; no FMS submodule in `components.yaml`/`.gitmodules`; only
compiled `.mod` interface files are available). This is a genuine,
disclosed gap, not a resolved fact.

What *is* available: `pe_start(n)+local_rank` is tile-major-contiguous by
construction (§1, confirmed by source), and Prompt 108 already recorded
the standard, widely-documented GFDL FMS `mpp_define_layout` convention —
row-major, x/i-direction fastest within a tile's `layout(1) x layout(2)`
sub-grid (`p108_*.md` §4, "row-major, x fastest", explicitly flagged there
as "derived," not literally quoted from FMS source either). This prompt
does not newly resolve that convention to hard source-level certainty; it
also finds no evidence contradicting it. Table for `layout=(2,2)`, under
that (assumed, not independently source-verified) convention, with
`local_rank = sub_j*layout_x + sub_i`:

| local_rank | sub_i | sub_j |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |

**This must be flagged as an assumed, not source-confirmed, convention in
any future implementation task** — the recommended mitigation (§9 below)
is a runtime cross-check (Stage 16), not blind trust.

## 5. Global-rank numbering vs. `EdgeNeighbor.rank`/CUDA Prompt-109 numbering (Stage 5)

FMS: `global_rank = pe_start(tile) + local_rank = (tile-1)*layout(1)*
layout(2) + local_rank` — **confirmed by source** (§1).

CUDA (Prompt 109/112): `subtile_global_rank`'s formula is `global_rank =
(face-1)*layout_x*layout_y + sub_j*layout_x + sub_i`.

These are **structurally identical** — both are tile-major/face-major,
both use a contiguous `layout_x*layout_y`-sized block per face, and both
reduce to the same value *if and only if* `local_rank = sub_j*layout_x +
sub_i` on the FMS side, which is exactly §4's assumed-but-not-source-
verified convention. Given §4's convention (best available evidence, not
proven), the two numbering schemes are consistent.

**Matches CUDA Prompt-109 numbering: YES, conditional on §4's unverified
row-major assumption holding** — this is the honest answer; it cannot be
upgraded to an unconditional YES without either finding literal FMS
source or a runtime cross-check (§9's Stage-16 recommendation).

## 6. Communicator identity (Stage 6)

- **FMS communicator**: implicit — `Atm%pelist`/`mpp_pe()`/`mpp_npes()`
  operate on whatever communicator `mpp_init` was given at startup (not
  re-traced to its own call site this prompt; out of this prompt's
  bounded scope since the standalone driver initializes MPI once, before
  either FMS or ESMF).
- **Bridge communicator**: `comm_cuda`, obtained via `ESMF_VMGet(VM,
  mpiCommunicator=comm_cuda)` (`FV_StateMod.F90:1263`), passed into
  `fv3_gpu_bridge_init(FV_Atm(1), comm_cuda, ...)`
  (`FV_StateMod.F90:1264`).
- **Rank ordering identical: YES, with moderate (not microscopically
  sourced) confidence.** Standalone FV3/GEOS initializes MPI exactly once
  at process start; the ESMF VM and FMS's `pelist` are both standard
  wrappers around that single process-global communicator in this
  architecture (MAPL/GEOS convention) — no evidence was found of the CUDA
  path creating or being handed a re-ordered sub-communicator. This is a
  structural inference from the single-MPI-init architecture, not a line
  of source that directly compares `comm_cuda`'s rank 0 against
  `pelist(1)` bit-for-bit.
- **Translation required: NO** under the above, for the current 6-rank
  configuration (already relied upon implicitly today — `comm_rank` from
  `MPI_Comm_rank(comm_cuda,...)` is compared directly against
  `rank_of_tile`, itself built from `Atm%tile`, with no translation layer,
  and this is validated in production). A future >6-rank design should
  still add the assertion recommended in §9 rather than continuing to
  assume this silently.

## 7. FMS query APIs vs. direct field access (Stage 7)

| Candidate | Confirmed to exist | Already used in this codebase | Notes |
|---|---|---|---|
| `Atm%layout(1:2)` direct field read | YES (`fv_arrays.F90:1287`) | YES, pervasively | No new FMS call; needs no new `use` dependency beyond `Atm` itself, already passed to the bridge |
| `mpp_get_layout(domain, layout)` | YES (`mpp_domains_mod`, imported at `fv_mp_mod.F90:82-83`'s sibling `fv_io.F90`) | YES (`fv_io.F90:642,787`) | Independent cross-check path; requires the `domain2D` object, not just `Atm` |
| `mpp_get_compute_domain`/`mpp_get_data_domain` | YES, imported and used (`fv_mp_mod.F90:83,666-667`) | YES | Not needed for layout/rank — already used for grid extents, unrelated to this prompt's gap |
| `mpp_get_ntile_count` | YES, imported (`fv_mp_mod.F90:82`) but call site is a **dead/commented line** (`fv_mp_mod.F90:994-995`) | NO (commented out) | Not a live path; do not resurrect without separate justification |

**Chosen authoritative path: `Atm%layout(1:2)` direct field read**, not an
FMS query call — it is already-resident on the exact object
(`fv3_gpu_bridge_init(Atm, comm, ...)`) the bridge call site already
receives, requires no new module dependency, and is definitionally
identical to what `mpp_get_layout` would return (both trace back to the
same `domain_decomp` write). `mpp_get_layout` remains available as an
optional independent cross-check if a future implementation wants
belt-and-suspenders validation, not as the primary source.

## 8. Bridge call-site audit (Stage 8)

`fv3_gpu_bridge_init(Atm, comm, enable, rc)` — `Atm` is `intent(inout),
target`, the **full** `fv_atmos_type`, already in scope at the exact call
site that today reads `Atm%tile` (`fv3_gpu_dyncore_bridge.F90:300`) and
calls `fv3_gpu_set_cubed_sphere_topology` (`:304`). Called once, from
`FV_StateMod.F90:1264`, after `FV_Atm(1)%gridstruct` is fully built
(confirmed by the existing code comment at `FV_StateMod.F90` immediately
above the call: "FV_Atm(1)%gridstruct is fully built by this point").

**Metadata available there: YES — trivially, no new plumbing required.**
`Atm%layout(1:2)` is already reachable through the exact same `Atm`
argument the bridge already dereferences for `Atm%tile`. No new
subroutine argument, no new `use` statement beyond what's already present,
no call-site relocation needed. This is the single strongest finding of
this prompt: **the metadata gap is not an access gap, it is an ABI/gate
gap** — the bridge *can* read `Atm%layout` today with a one-line addition
at an already-open call site; it simply doesn't yet, and even if it did,
`fv3_gpu_set_cubed_sphere_topology`'s signature has nowhere to put the
value, and `build_rank_of_tile`/the `comm_size==6` check would still
reject anything but the current 6-rank case (§10).

Required upstream plumbing: **none** beyond reading `Atm%layout(1)`/
`Atm%layout(2)` into two local integers at the existing call site — no
change to `FV_StateMod.F90`'s own call, no change to when the call
happens, no new field needed on `Atm` (it already exists).

## 9. Minimal ABI extension (Stage 9)

Two independent, additive concerns, both following the existing
`abi_version`/`struct_bytes` additive-tail convention (every ABI struct
already carries this header; `struct_bytes` lets a receiver detect
whether a caller's struct includes newly-appended fields):

1. **`fv3_gpu_create_desc`** (`fv3_gpu_api.h:99-109`): append
   `int32_t layout_x; int32_t layout_y;` after the existing `mpi_size`
   field. Redundant with nothing already present (`mpi_size` is the
   *total* communicator size, not the per-face layout decomposition —
   distinct information).
2. **`fv3_gpu_set_cubed_sphere_topology(ctx, tile, npx, npy,
   rank_of_tile[], err)`** (`fv3_gpu_api.h:292-295`): today's
   `rank_of_tile` is a fixed-size-6 array, one entry per *face*, built by
   `build_rank_of_tile`'s `MPI_Allgather` (`fv3_gpu_dyncore_bridge.F90:
   1432-1463`). Generalizing this signature by widening the array to
   `layout_x*layout_y*6` entries would duplicate exactly the same
   `(sub_i,sub_j)`→rank information already implied by `layout_x`/
   `layout_y` plus the tile-major numbering (§5) — **recommend NOT**
   widening `rank_of_tile`; instead add `int32_t layout_x, int32_t
   layout_y` as two new trailing parameters and let the CUDA side derive
   the full sub-tile topology itself via the already-validated
   `subtile_edge_neighbor`/`subtile_global_rank` (Prompt 109) math, given
   only `(tile, sub_i, sub_j, layout_x, layout_y, npx, npy)` — the same
   division of responsibility Prompt 111's additive builder already
   uses, just fed real metadata instead of a synthetic test harness's.

**Redundant metadata avoided: YES.** `rank_of_tile`'s only remaining job
under this design becomes validating that the topology CUDA derives
matches what MPI-allgathered `Atm%tile`/rank pairs actually observed
(a runtime consistency assertion, not a second source of truth) — see
§10.

`(sub_i, sub_j)` itself: **derive on the CUDA side** from `comm_rank` and
`(layout_x, layout_y, tile)` using the same tile-major-then-row-major
formula as §5, rather than adding two more ABI fields for values every
rank could reproduce as-called-for by "do not force CUDA to infer it, use
source truth" (Stage 5's own instruction) — **this is the one place this
recommendation deliberately allows CUDA-side derivation, but only because
the alternative (also passing `sub_i`/`sub_j` explicitly) is answered by
§16's proposed *cross-check*, not blind trust**: the design in §16 has the
Fortran side allgather `Atm%tile` (as it already does) plus one new
`MPI_Allgather` of raw `comm_rank` ordering, so any mismatch between
"derived (sub_i,sub_j)" and "observed rank ordering" is caught at init
time rather than silently producing wrong halo donors.

## 10. Gate design (future, not implemented) (Stage 10)

**Future bridge condition** (`fv3_gpu_dyncore_bridge.F90:163`, currently
`if (comm_size /= 6)`): would become `if (mod(comm_size, 6) /= 0 .or.
Atm%layout(1)*Atm%layout(2) /= comm_size/6) then` — i.e., require
`comm_size` to be an exact multiple of 6 (6 faces) equal to
`layout_x*layout_y*6`, replacing the single-value equality check with a
factorization check. `build_rank_of_tile`'s own internal one-rank-per-tile
assumption (`fv3_gpu_dyncore_bridge.F90:1451`, "tile ... is claimed by
more than one rank") would ALSO need to change — this is a second,
previously undocumented gate discovered this prompt, not just the
top-level `comm_size==6` check; a >6-rank design must generalize
`build_rank_of_tile` into something that allgathers `(tile, comm_rank)`
pairs and validates *layout_x·layout_y* ranks per tile, not exactly one.

**Future MPI-factory condition** (`fv3_gpu_halo_mpi.cpp:426-427`,
currently `if (size != cubed_sphere_tiles)`): would become `if (size % 6
!= 0) return refuse(...)`, mirroring the bridge's relaxed check —
Prompt 112's `create_mpi_transport_for_testing` already proves the
underlying `MpiHaloTransport` machinery itself needs no further change to
accept this; only the production factory's refusal condition would need
loosening.

**`fv3_gpu_runtime.cpp:1579`**'s `mpi_size != halo::cubed_sphere_tiles`
check (the third gate identified in Prompt 111/112) would need the same
factorization-based relaxation.

**Implemented: NO.** All three of the above are descriptions of a future
change, not diffs applied by this prompt.

## 11. layout=(1,1) compatibility (Stage 11 / Stage 18 continuation)

The proposed `mod(comm_size,6)==0 && layout_x*layout_y==comm_size/6` check
reduces, at `layout_x=layout_y=1`, to `comm_size%6==0 && 1==comm_size/6`,
i.e. `comm_size==6` exactly — the identical condition as today's literal
equality check. **Exact regression feasible: YES** — the relaxed gate is
a strict superset of the current gate's accepted input, not a behavior
change at `comm_size==6`. Likewise, CUDA-side `(sub_i,sub_j)` derivation
at `layout_x=layout_y=1` always yields `(0,0)`, and
`subtile_edge_neighbor(SubTileId{tile,0,0}, edge, 1, 1)` was already
proven (Prompt 109 T1-T2) to reduce exactly to the legacy per-face
calculation.

## 12. 24-rank metadata oracle (Stage 12)

| global_rank | tile (face) | local_rank (assumed row-major) | sub_i | sub_j |
|---|---|---|---|---|
| 0-3 | 1 | 0-3 | 0,1,0,1 | 0,0,1,1 |
| 4-7 | 2 | 0-3 | 0,1,0,1 | 0,0,1,1 |
| 8-11 | 3 | 0-3 | 0,1,0,1 | 0,0,1,1 |
| 12-15 | 4 | 0-3 | 0,1,0,1 | 0,0,1,1 |
| 16-19 | 5 | 0-3 | 0,1,0,1 | 0,0,1,1 |
| 20-23 | 6 | 0-3 | 0,1,0,1 | 0,0,1,1 |

Formula: `tile = global_rank/4 + 1`, `local_rank = global_rank mod 4`,
`sub_i = local_rank mod 2`, `sub_j = local_rank / 2`. **Complete: YES**
for the metadata *arithmetic*; **conditional on §4's unverified row-major
assumption** for whether it matches FMS's actual internal PE placement —
the same caveat as §4/§5, not a new one.

## 13. Corner dependency (Stage 13)

**Can a real 24-rank production run happen before corner transport: YES,
with reduced but non-zero physical accuracy at cube-panel corners.** All
of Prompts 109-112's validation is edges-only; corner transport is
metadata-populated but never message-delivered (Prompt 110 §5, Prompt 112
§11). A real production run at `layout=(2,2)` would exchange all 4 edges
correctly (once bridge metadata plumbing lands) but would leave 4
corner-adjacent halo cells per sub-rank stale/uninitialized relative to
their true cross-rank corner donor — this is a numerical-accuracy gap at
the corner stencil width, not a crash or a gate rejection. Whether that
residual error is acceptable for an initial 24-rank bring-up (vs.
requiring corner transport first) is a numerical-tolerance decision for
Gate-C2-style validation, not a software-architecture one — flagged here,
not resolved.

## 14. Next-phase order (Stage 14 / Stage 18)

**BRIDGE METADATA PLUMBING FIRST.**

Reason: §8 found the metadata-access gap is nearly zero-cost (one field
already resident on an already-in-scope struct); §9-11 show the ABI/gate
generalization is additive and provably regression-safe at
`layout=(1,1)`; §13 shows corner transport is not a hard blocker for a
first 24-rank bring-up, only an accuracy caveat that can be measured and
reported once bridge plumbing exists. Attempting corner transport first
would still leave the bridge's `comm_size==6` gate in place, so no real
>6-rank run could happen regardless of corner-transport progress — bridge
plumbing is the binding constraint, corner transport is not.

## 15. Bounded next task (Stage 15 / mandatory NEXT TASK)

**STATUS: IMPLEMENTED by Prompt 114** — see
`p114_fms_layout_bridge_abi_plumbing_2026-09-30.md`. `layout_x`/`layout_y`
are now threaded through the ABI, GPU-executed and behavior-neutral at C32
and C512 production scale, all four >6-rank gates re-confirmed untouched.
Prompt 114's own §16 recommended Option A (verify FMS within-tile rank
ordering via a Fortran-only `layout=(2,2)` diagnostic, superseding this
doc's §16 allgather-based approach below) as the next task, in preference to
implementing any topology-builder switch.

**Task** (as originally specified, now completed): Implement (not just design) the additive ABI extension and
bridge-side metadata read described in §9 — add `layout_x`/`layout_y` to
`fv3_gpu_create_desc` and as new trailing parameters to
`fv3_gpu_set_cubed_sphere_topology`; have the bridge populate them from
`Atm%layout(1:2)` at the existing call site; have the CUDA runtime store
them but **take no new branch on their value** (i.e., continue to require
`layout_x==1 && layout_y==1` via the *existing*, still-unrelaxed
`comm_size==6` gate — this task adds the wires, not the switch).
**Complexity: SMALL** (two struct/signature additions following an
already-established additive-tail pattern; one bridge-side read of an
already-resident field; zero behavior change at any existing call site,
verifiable by the same kind of bitwise/no-op regression check used in
every prior sub-tile prompt). **No production changes: YES** (gates
untouched, only new always-1-valued fields added and threaded through).
**No commit/push: YES.**

## 16. Runtime-assertion design (Stage 16, folded from §9)

At bridge-init time, in addition to the existing `Atm%tile`-based
allgather (`build_rank_of_tile`), add one new `MPI_Allgather` of each
rank's own `comm_rank` (trivial — already known locally) keyed the same
way, and cross-check that the tile-major-then-row-major formula (§5)
applied to the allgathered `(tile, comm_rank)` pairs reproduces the
locally-derived `(sub_i, sub_j)` before trusting CUDA-side derivation for
any real multi-rank-per-face run. This directly mitigates §4's unverified
FMS convention risk with a real runtime check instead of a silent
assumption — recommended as part of the same bounded next task (§15), not
a separate one, since it is a small addition to the same call site.

## 17. Stop conditions (Stage 21)

None of the listed stop conditions are triggered:

- Layout metadata is **not** inaccessible without invasive FMS changes —
  §2/§7/§8 show it is already resident on `Atm` and already reachable at
  the existing bridge call site with zero FMS-side changes.
- No numerical/seam-math redesign is implied — §9-11 are purely additive
  ABI/gate changes; the underlying edge topology math (Prompt 109) is
  reused unchanged.
- The one open uncertainty (§4's row-major convention) is not a hard stop
  — it is mitigated by a bounded, small runtime assertion (§16), not by
  blocking further work.
- Corner transport's absence does not block bridge metadata plumbing
  (§13/§14) — it is a separate, already-deferred track.

## 18. Summary of new facts vs. Prompt 108/110

| Fact | Prompt 108/110 status | This prompt |
|---|---|---|
| `pe_start`/tile-major numbering | Found, stated as §4 of p108 | Re-confirmed unchanged |
| `Atm%layout` retained field | Not distinguished from transient `layout` arg | **New**: confirmed pointer-aliased, persistent, already-resident on `Atm` |
| Bridge already reads `Atm%tile` | Known (p108 §3) | **New**: bridge call site already receives full `Atm`, so `Atm%layout` needs zero new plumbing to reach the same call site |
| `build_rank_of_tile`'s own one-rank-per-tile gate | Not documented as a separate gate | **New**: a second, independent gate beyond the top-level `comm_size==6` check, must also be relaxed |
| `mpp_get_layout` query API | Not evaluated | **New**: confirmed to exist and already be used elsewhere in this codebase (`fv_io.F90`), evaluated and not chosen over direct field access |
| Row-major within-tile convention | Asserted, not source-verified | Re-confirmed still not source-verified (FMS not vendored); explicitly flagged with a mitigation (§16) rather than left silent |

---

```
FMS→CUDA SUB-TILE METADATA PATH MAPPED — READY FOR BRIDGE METADATA
PLUMBING IMPLEMENTATION (ADDITIVE ABI FIELDS layout_x/layout_y, ALWAYS-1
VALUES, NO GATE RELAXATION, FRESH AUTHORIZATION REQUIRED BEFORE
IMPLEMENTATION)
```
