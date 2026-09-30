# Prompt 108 — C1024 Intra-Tile Halo Exchange Architecture (Design/Mapping Only)

**Status: DESIGN/MAPPING ONLY. No production halo implementation. No source
changes. No commit/push.**

**Update 2026-09-30 (Prompt 109):** §15 Phases 1-4 (rank-numbering identity,
intra-face cardinal edges, cross-face segment donors, diagonal/mixed corner
donors) are now IMPLEMENTED/COMPILED/GPU-EXECUTED and validated at small
grid/single node. See `p109_cuda_subtile_halo_phases1_4_2026-09-30.md`. Phase
5 (transport-backend generalization) is the recommended next step; Phase 6
(bridge `comm_size==6` gate) and Phase 8 (real 24-rank/C1024 run) remain
untouched and out of scope until explicitly authorized.

Branch `optimization/cuda-fv3-performance`. This document answers Prompt 108's
7 primary questions and 24 stages entirely from direct source inspection (both
this repo and the separate `GEOSfvdycore_fortran_perf` checkout, which
contains the populated authoritative `@fvdycore/` tree) plus already-recorded
project findings (P101-P103). No numerical algorithm, discretization, physics,
CUDA kernel (beyond read-only mapping), or production-default change was made.

## 0. Source/Repo Context

- CUDA repo (this repo): `cuda/interface/fv3_gpu_dyncore_bridge.F90` (Fortran
  bridge), `cuda/dyn_core/include/fv3_gpu_halo_topology.hpp` +
  `runtime/fv3_gpu_halo_topology.cpp` (seam algebra + mosaic table),
  `include/fv3_gpu_halo_transport.hpp`/`fv3_gpu_halo_peer.hpp`/
  `fv3_gpu_halo_mpi.cpp` (transport backends), `include/fv3_gpu_communication.h`
  (halo group enum), `cgrid/fv3_cgrid_corners.hpp` (panel-corner kernels).
- Authoritative Fortran/FMS: `GEOSfvdycore_fortran_perf/src/Components/
  @FVdycoreCubed_GridComp/@fvdycore/tools/fv_mp_mod.F90` (`domain_decomp`),
  `@fvdycore/model/fv_grid_utils.F90` (`BITWISE_EXACT_SUM`/`mpp_global_sum`).
- `@fvdycore` IS populated in the `GEOSfvdycore_fortran_perf` checkout
  (unlike this repo, where it is documented absent) — this is where Stage 4's
  answer was found.

## 1. Six-Rank Assumptions (evidence table)

| # | Location | Assumption | Evidence |
|---|---|---|---|
| 1 | `fv3_gpu_dyncore_bridge.F90:163` | `if (comm_size /= 6) call mpp_error(FATAL, ...)` — hard FATAL, not a soft default | Direct read |
| 2 | `fv3_gpu_dyncore_bridge.F90:146` | `rank_of_tile(6)` fixed-size array; one rank ↔ one tile, no sub-tile index | Direct read |
| 3 | `fv3_gpu_dyncore_bridge.F90:1432-1463` (`build_rank_of_tile`) | Builds a 6-entry rank map keyed by `Atm%tile` (1..6) alone — no `layout`/local-(i,j) concept | Direct read |
| 4 | `fv3_gpu_runtime.cpp:1557-1566` (`fv3_gpu_set_cubed_sphere_topology`) | `if (c->data.create.mpi_size != halo::cubed_sphere_tiles) return FV3_GPU_UNSUPPORTED_CONFIGURATION` (`cubed_sphere_tiles = 6`); own comment: "sub-tile decomposition is not yet implemented" | Direct read |
| 5 | `fv3_gpu_halo_topology.hpp`/`.cpp` (entire seam algebra) | `TileEdge` has exactly 4 values (W/E/S/N) and `EdgeNeighbor` carries exactly ONE `rank`/`tile` per edge — a whole-tile-to-whole-tile contact model, structurally unable to express a partial-edge, multi-donor boundary | Direct read (`mosaic[6][4]` table, `EdgeNeighbor` single-rank field) |
| 6 | `fv3_gpu_halo_topology.cpp:157-176` (`make_cubed_sphere_topology`) | `t.sw_corner = t.se_corner = t.ne_corner = t.nw_corner = 1` unconditionally — every rank's corner is treated as a genuine cube-panel corner, correct only because today one rank = one whole face | Direct read, with own comment acknowledging corner-owner resolution is deferred |
| 7 | `fv3_cgrid_corners.hpp` (`fill_4corners`/`fill2_4corners` callers) | Panel-corner pole algebra (`sw_core.F90` `fill_4corners`/`fill2_4corners`) driven only by the 4 corner flags above, with no concept of "this corner is intra-face, skip the pole algebra" | Direct read |
| 8 | `fv3_gpu_halo_mpi.hpp:52-56` (`MpiHaloTransport` constructor doc/check) | Independent `comm_size==6`-equivalent gate: construction fails unless communicator size is 6 and this rank's tile matches the rank↔tile identity the production decomposition requires — a *separate* enforcement point from bridge item #1, at the transport-construction layer | Direct read (sub-investigation C) |

**Total: 8 distinct six-rank assumption sites, spanning 2 files in the Fortran
bridge and 4 files in the CUDA runtime (the transport layer enforces its own
independent `comm_size==6` check, not merely inheriting the bridge's).** All
are confined to the CUDA integration layer (see Stage 4) — the authoritative
Fortran/FMS core is not among them.

## 2. Current C512 Domain Ownership Per Rank

One rank owns exactly one whole cubed-sphere face: `Atm%bd%is/ie/js/je` span
the full `1..npx-1` × `1..npy-1` face interior (`cd%is/ie/js/je` in
`fv3_gpu_bridge_init`), halo width `Atm%bd%ng` (compile-time FV3 parameter,
`fv3_domain_halo_width = 3` per `fv3_gpu_halo_topology.hpp`). Each rank has
exactly 4 edge neighbors (one per whole adjacent face, from the 12-contact
mosaic) and 4 corners, all genuine cube-panel corners. No same-face
(intra-tile) neighbor exists today because no two ranks ever share a face.

## 3. Current Halo Path — Classification

**Classification: (C) mixture, with a deliberate split at the topology layer:**

- **Topology/seam algebra (`fv3_gpu_halo_topology.hpp/.cpp`): (A) custom
  code**, but *cross-validated against, not divorced from*, the two
  authoritative FMS-derived tables (`CubeHalo.F90:213-261` and
  `fv_mp_mod.F90:577-625`) and against FMS's own rotation/sign tables
  (`mpp_domains_define.inc`, `mpp_do_updateV.fh`) and its test oracle
  (`test_mpp_domains.F90:4092-4126`). I independently re-read
  `fv_mp_mod.F90:571-625`'s `case(6)` contact-line block and confirmed it
  reproduces the same 12 tile-pairings as the CUDA header's `mosaic[6][4]`
  table verbatim (e.g. contact 1: T1 EAST↔T2 WEST; contact 3: T1 WEST↔T5
  NORTH, reversed). This is a from-scratch re-derivation for GPU execution,
  not a wrapper around FMS at runtime.
- **Transport (`PeerHaloTransport`/`MpiHaloTransport`, `fv3_gpu_halo_peer.hpp`,
  `fv3_gpu_halo_mpi.cpp`): (A) custom**, entirely independent of FMS's own
  MPI/domain machinery — a five-stage pack→transfer→unpack pipeline
  per named halo *group* (12 groups, `FV3_GPU_H_DWIND_IN` .. `H_DWIND_SEAM`,
  `fv3_gpu_communication.h`), driven purely by `TopologyView`'s 4 edges.
- **Decomposition/rank↔tile mapping**: **(A) custom**, entirely inside the
  CUDA bridge — the authoritative Fortran core's own decomposition machinery
  (`fv_mp_mod.F90::domain_decomp`) is bypassed for the CUDA path (see Stage 4).

## 4. Existing FV3/FMS Multi-Rank-Per-Face Support

**EXISTING SUPPORT: YES — in the authoritative Fortran/FMS core, already
exercised by the Fortran backend; NOT reachable from the CUDA bridge.**

Direct evidence, `fv_mp_mod.F90::domain_decomp`, `case(6)` (cubed-sphere):

```fortran
npes_per_tile = npes_x*npes_y          ! layout(1)*layout(2), NOT hardcoded to 1
if (npes_per_tile==0) npes_per_tile = npes/nregions
call mpp_define_layout( (/1,npx-1,1,npy-1/), npes_per_tile, layout )
...
layout2D(:,n) = layout                  ! per tile n=1..6
pe_start(n) = pelist(1) + (n-1)*layout(1)*layout(2)
pe_end(n)   = pe_start(n) + layout(1)*layout(2) - 1
```

then `mpp_define_mosaic(global_indices, layout2D, pe_start, pe_end, ...,
num_contact, tile1, tile2, istart1, iend1, ..., is_symmetry=.true.)` using
the *identical* 12-contact table re-derived independently in the CUDA header
(cross-checked line-for-line against `fv_mp_mod.F90:571-625` in this session).

This means **FMS `mpp_define_mosaic`/`mpp_define_layout`/`mpp_domains` already
natively supports `layout(1)>1`/`layout(2)>1` (multiple ranks per cubed-sphere
face)** for any Fortran-backend FV3 run — it is a first-class, already-used
code path (`npes_per_tile` is a generic variable, not special-cased to 1). The
CUDA bridge simply never calls into or mirrors this generality: it derives its
own 6-entry `rank_of_tile` from `Atm%tile` alone and never reads
`Atm%layout`/`Atm%npes_per_tile`/local-(i,j)-within-tile.

**Canonical PE numbering (derived from `pe_start`/`pe_end` above): tile-major,
then row-major within a tile** — `global_rank = (tile-1)*npes_per_tile +
local_rank`, `local_rank` following `mpp_define_layout`'s row-major
convention (x fastest). This is the numbering Stage 10 adopts (not a new
invention). **Update (Prompt 113)**: the tile-major part (`pe_start`/
`pe_end`) is confirmed directly by source, unchanged. The row-major/
x-fastest *within-tile* part remains an assumed, not source-verified,
FMS convention — FMS itself is a precompiled external dependency in both
project repos (no vendored `mpp_domains*.F90`), so this cannot be upgraded
to a sourced fact here. Prompt 113 recommends a bounded runtime
cross-check (allgather `comm_rank` alongside the existing `Atm%tile`
allgather) as mitigation rather than continuing to assume it silently —
see `p113_fms_bridge_subtile_metadata_design_2026-09-30.md` §4/§16.

**Conclusion: architecture choice should reuse this existing FMS layout/mosaic
convention's numbering and neighbor semantics, rather than re-deriving them —
strong support for "prefer reuse" at the decomposition-metadata level.** The
CUDA-side topology/transport code, however, still needs its own generalization
(Stage 14), because it does not call FMS at runtime at all (by design — see
`project_architecture.md`'s ABI boundary: Fortran does not call individual
kernels, and the CUDA runtime is C++/CUDA, not linked against `mpp_domains`).

## 5. Target C1024 Topology (2×2 ranks/face, 24 ranks)

Layout `(layout_x, layout_y) = (2,2)` per face, `npes_per_tile = 4`.
Numbering (per Stage 4's derivation): `global_rank = (face-1)*4 + local_index`,
`local_index = bj*2 + bi`, `bi,bj ∈ {0,1}` (bi = x/east block, bj = y/north
block): `local_index 0=SW, 1=SE, 2=NW, 3=NE`.

Each sub-rank has:
- **2 intra-face cardinal neighbors** (same face, adjacent block) — e.g. SW's
  EAST neighbor is SE (same face), SW's NORTH neighbor is NW (same face).
  These are plain Cartesian, always-aligned, always-`SeamRotation::zero`
  exchanges: no rotation, no sign flip, no index reversal — the *simplest*
  new case, structurally identical to any ordinary (non-cube) rectangular
  MPI-domain halo.
- **2 cross-face cardinal neighbors** (the 2 face-boundary edges) — reuse the
  existing 12-contact mosaic/seam-rotation/vector-sign algebra unchanged,
  but the donor is no longer the *whole* opposite face: only the ONE donor
  sub-rank (of the 2 along that boundary) whose along-seam interval overlaps
  mine. This is a **partial-edge, single-relevant-donor-per-segment**
  exchange — the existing `t'=t` / `t'=n+1-t` index algebra still applies
  per segment, but `EdgeNeighbor` must become "1 or more neighbor segments
  per edge," not "exactly 1 rank per edge."
- **4 corners**: for a 2×2 layout, each sub-rank sits at exactly **one**
  true cube-panel corner of its face (SW sub-rank ↔ face's SW corner, etc.)
  and its other **3** corners are ordinary intra-face/mixed corners:
  - 1 pure intra-face diagonal corner (e.g. SW's NE corner ↔ NE sub-rank,
    same face, identity transform, no rotation) — bog-standard rectangular
    corner exchange.
  - 2 mixed corners, each combining one cross-face edge-neighbor with one
    intra-face edge-neighbor's corner cell — this is the same "ordinary
    interior corner" case any rectangular multi-rank decomposition handles
    (owned by whichever single neighbor block's domain rectangle contains
    the point); it is NOT a new cubed-sphere-specific case, it is exactly
    what `mpp_domains` already does for any `layout(1,2)>1` decomposition on
    any tile, cube or not.
  - Exactly **6 global cube vertices** (one per face-triple junction) need
    `fill_4corners`/`fill2_4corners`'s special 3-tile pole algebra; every
    other corner in the 24-rank topology is an ordinary 4-block junction.

## 6. Halo Exchange Type Classification

| Type | Present today | Present at 24 ranks | Rotation/sign |
|---|---|---|---|
| A. Intra-face cardinal (new) | No | Yes (2 of 4 edges/sub-rank for interior-of-face blocks; all 4 for a hypothetical larger layout) | None — identity |
| B. Cross-face cardinal (existing algebra, new partial-donor logic) | Yes (whole-edge) | Yes (segmented) | Existing seam rotation/sign table, unchanged per-segment |
| C. Intra-face diagonal corner (new) | No | Yes (1 of 4 corners/sub-rank in a 2×2 layout) | None — identity |
| D. Mixed edge+intra-face corner (new, but generic to any multi-rank layout) | No | Yes (2 of 4 corners/sub-rank) | Whichever single edge's rotation applies at that point — same table |
| E. True cube-panel corner (existing pole algebra) | Yes (all 4 corners/rank) | Yes, but only 1 of 4 corners/sub-rank (exactly 6 global occurrences) | `fill_4corners`/`fill2_4corners`, unchanged |

## 7. Field Inventory (halo-participating groups)

Directly enumerated from `fv3_gpu_communication.h`'s group enum and each
group's authoritative dyn_core.F90 call site (`fv3_gpu_runtime.cpp:644-655`):

| Group | Fields | Stagger | Vector? |
|---|---|---|---|
| `H_DWIND_IN` | u, v | D-grid (NORTH/EAST) | Yes (DGRID_NE) |
| `H_W_IN` | w | CENTER | No |
| `H_GZ_FIRST` | gz | CENTER | No |
| `H_Q_INLINE` | inline_q tracers | CENTER | No |
| `H_DIVGD` | divgd | CORNER | No (scalar_pair metric-style) |
| `H_CWIND` | uc, vc | C-grid (EAST/NORTH) | Yes (CGRID_NE) |
| `H_DSCALAR` | q_con, delp, pt (up to 4 fields) | CENTER | No |
| `H_ZH` | zh | CENTER | No |
| `H_PKC2` | pkc | CENTER | No |
| `H_HEAT` | heat_source (del2_cubed) | CENTER | No |
| `H_DWIND_NEXT` | u, v (next substep) | D-grid | Yes |
| `H_DWIND_SEAM` | u, v (E11 shared seam line, `mpp_get_boundary`) | D-grid | Yes, h=0 only |

12 groups total, 2 genuinely vector (u/v D-grid and C-grid), 1 seam-only
reconciliation group, 9 scalar/metric. PE, PK/PKZ, and DELZ are handled by
separate WP-C/vertical remap ports (not part of this dyn_core halo group
list) and are out of scope for this design (per the "Do NOT modify" list).

## 8. Vector Orientation vs. Plain Copy (documentation only, not implemented)

`fv3_gpu_halo_topology.hpp`'s `seam_rotation`/`seam_vector_transform` derive
the +90°/-90°/0° rotation **purely from the pair of tile edges involved**
(mine vs. donor's), never from rank identity. Consequently:

- **At an internal (intra-face) rank boundary, the rotation is always
  `SeamRotation::zero`** — both edges are on the same face, so
  `edge_is_x_normal(mine) == edge_is_x_normal(donor)` trivially (same face,
  same axis convention), which is exactly `seam_is_reversed`'s `false`
  branch. No implementation change is needed to make this true — it falls
  out of the existing, unmodified rotation formula once the donor is
  correctly identified as "same face, adjacent block" rather than "different
  face." The generalization work is entirely in *identifying the right
  donor*, not in the rotation math.
- **At a cross-face boundary, rotation is unchanged** — it is still a
  function of the two face edges only, independent of how many ranks tile
  each face, so a 24-rank sub-block inherits its parent face's edge identity
  and therefore the identical rotation the whole-face rank would have used.

This distinction — "plain copy at internal boundaries, existing rotation
table at face boundaries" — is the central correctness invariant Stage 15's
implementation phases must preserve.

## 9. Communicator/Domain Infrastructure

`Atm%tile` (1..6) is the only topology-relevant field the bridge reads today;
no `Atm%layout`/`Atm%npes_per_tile`/local-(i,j) metadata is consumed by
`fv3_gpu_dyncore_bridge.F90`, even though (Stage 4) the Fortran type almost
certainly carries `layout`/`npes_per_tile` already (passed into
`domain_decomp` by reference and stored per `fv_arrays.F90`'s `fv_atmos_type`
grid/bounds derived types — not re-verified line-by-line here per the
change-budget, but strongly implied by `domain_decomp`'s own signature using
`Atm%npes_per_tile`, `tools/fv_mp_mod.F90:379`). **Tile-id exists as
first-class state (`Atm%tile`); rank-within-tile/layout does not currently
cross the CUDA ABI, but the underlying Fortran-side data it would be derived
from already exists.**

## 10. Proposed 24-Rank Numbering (not implemented)

Adopt Stage 4's derived FMS-canonical scheme directly: `global_rank =
(face-1)*4 + local_index`, `local_index = bj*2+bi`. This requires no new
convention — it is the same `pe_start(n) = pelist(1) + (n-1)*layout(1)*
layout(2)` arithmetic FMS's own `mpp_define_mosaic` call already uses for the
Fortran backend, so a CUDA-side rank ↔ (face, bi, bj) decode is a **pure
arithmetic derivation** (`face = rank/4 + 1`, `local = rank%4`, `bi =
local%2`, `bj = local/2`), not a new numbering invented for this task.

## 11. GPU / Host-Staged-MPI Interaction (no CUDA-aware MPI)

`fv3_gpu_halo_transport.hpp`'s five-stage contract (pack → pack_done event →
transfer (D2H/MPI/H2D or device-to-device) → xfer_done event → unpack) is
expressed per named **group**, not per neighbor count — a group's pack/unpack
kernels are agnostic to how many donor segments feed the group's receive
buffer. **The transport mechanics do not need to change** to support more
neighbors per rank; only the **topology layer feeding it** (`TopologyView`/
`EdgeNeighbor`, Stage 1 item 5) needs to grow from "exactly 1 rank per edge"
to "1 or more neighbor segments per edge, plus up to 4 corner neighbors."
Host-staged D2H/H2D buffer sizing scales with the *number of segments*, not
a fundamentally new staging mechanism.

## 12. Buffer/Packing Impact (estimate, no optimization)

At C1024/24-rank, each sub-rank's local domain is 512×512 (same horizontal
extent as a full C512 face) — i.e. each rank packs/unpacks the *same*
per-edge halo byte count as today's C512 whole-face rank, just across up to
6 neighbor directions (4 cardinal + up to 2 corner-bearing) instead of 4,
and with a face-boundary edge's donor buffer split across (up to) 2 segments
instead of 1. Total per-rank pack/unpack byte volume is essentially
unchanged from the current per-rank C512 pattern; message *count* increases
(more, smaller messages at face-boundary segments; new small messages for
the 2 previously-nonexistent intra-face corner directions).

Rough order-of-magnitude bytes (R8, `ng=3`, cross-checked independently by
sub-investigation C against the halo-group table in Stage 7): ~13 live
field-instances × 3-cell-deep × 511-cell edge × 8 bytes ≈ 160 KB/edge ⇒
**≈640 KB/rank/exchange-cycle at C512 today, and ≈640 KB/rank/exchange-cycle
at C1024/24-rank** (same per-rank edge length, Stage 12 above) — i.e. per-rank
byte volume is flat across the 6→24-rank transition; only message count and
system-wide aggregate byte volume grow.

## 13. Corner-Fill Readiness Classification

**`RUN_CUDA_FV3_FUSE_CORNER_FILL` / `launch_fill_4corners`/
`launch_fill2_4corners` (`fv3_cgrid_corners.hpp`): NEEDS GENERALIZATION.**

The kernels themselves are pure, rank-agnostic local algebra driven only by
`CornerFillArgs{npx, npy, sw/se/ne/nw_corner}` — no change needed to the
kernel bodies. What needs to change is **who calls them and with which flags
true**: today `make_cubed_sphere_topology` sets all 4 flags true
unconditionally (Stage 1 item 6), which is only correct because one rank
currently owns all 4 of a face's true corners. At 24 ranks, exactly one of a
sub-rank's 4 corner flags may be true (the one true cube-vertex corner, if
any — most sub-ranks in a >2×2 layout would have zero), and the other 3
corners must instead go through an ordinary (new, but generic — Stage 6 type
C/D) rectangular corner exchange, not the pole kernels.

**Important scope correction (from sub-investigation C, elevates this beyond
a pure flag change):** today's corner handling is entirely local, zero-MPI —
`fill_4corners`/`fill2_4corners` operate on a single rank's own halo-filled
data with no diagonal-neighbor message of any kind, because a true cube
corner never needs one (it is a local 3-tile-adjacent construction from
already-exchanged edge halos). Type C/D corners at 24 ranks, by contrast,
need an actual diagonal-neighbor exchange that does not exist in any form
today — `TopologyView`/`EdgeNeighbor` (Stage 1 item 5) has no corner-message
slot at all, only 4 cardinal edges. So this is not just "flip which corner
flags are true": it requires adding a genuinely new communication path (a
diagonal MPI/peer message, or equivalently deriving the corner value from
already-exchanged edge-halo overlap without a dedicated message — an
implementation choice for Stage 15, not decided here) alongside the
flag-assignment fix. Net classification: **NEEDS GENERALIZATION, with a real
new-capability component (diagonal corner messaging) — moderate scope, but
not "well-isolated" in the sense of a pure parameter change.**

## 14. Architecture Choice

**Chosen: Hybrid — reuse FMS/mpp's existing decomposition metadata and
numbering (Stage 4/10) at the Fortran/bridge boundary; extend the existing
custom CUDA topology/transport layer (Stage 3) to express partial-edge and
diagonal-corner neighbors, rather than replacing it with a runtime FMS/mpp
dependency.**

Justification: full reuse of `mpp_domains` at the CUDA runtime layer (Option
"pure A: reuse FMS/mpp end-to-end") is not viable without violating the
project's own stable-ABI/architecture rule that Fortran does not call
individual CUDA kernels and the CUDA runtime is deliberately independent
C++/CUDA with no FMS/MPI-library link dependency of its own
(`project_architecture.md`); the CUDA halo topology header is *already* an
intentional, cross-validated re-derivation of FMS's tables for exactly this
reason (Stage 3's own header comments), not an oversight to be undone. Full
custom reimplementation of decomposition/layout logic (Option "pure custom,"
ignoring Stage 4's finding) would duplicate work FMS already does correctly
and risk a numbering/neighbor-graph mismatch with the Fortran side. The
hybrid — read `layout`/local-(i,j) from the Fortran side (which already has
it via `domain_decomp`), derive rank↔(face,bi,bj) with Stage 10's pure
arithmetic, and generalize the existing custom `TopologyView`/transport layer
to carry more neighbor slots — reuses everything that already agrees between
Fortran and CUDA (the mosaic table, the rotation/sign algebra, the pack/
transfer/unpack pipeline) and localizes all new code to exactly the two
structurally six-rank-shaped data structures identified in Stage 1 (items 2-6).

## 15. Minimal Incremental Implementation Phases (design only)

| Phase | Scope | STOP/PASS gate |
|---|---|---|
| 1 | Extend `TopologyView`/`EdgeNeighbor` to carry N≥1 neighbor segments per edge + up to 4 corner-neighbor slots (data structure only, no new logic wired) | PASS: existing 6-rank tests unchanged (N=1 degenerates to today) |
| 2 | Add intra-face cardinal neighbor derivation (Stage 5 type A) for a 2×2 layout, identity transform | PASS: new host-only unit test (Stage 16) matches by construction (no rotation) |
| 3 | Generalize cross-face edge donor lookup to select the correct sub-segment (Stage 5 type B) using existing seam algebra unchanged | PASS: same-value regression against today's whole-face algebra at layout=(1,1) |
| 4 | Add intra-face diagonal + mixed corner exchange (Stage 5 type C/D) | PASS: sentinel provenance test (Stage 16) round-trips correctly |
| 5 | Generalize corner-flag assignment (Stage 13) to per-sub-rank true-corner geometry, gate `fill_4corners`/`fill2_4corners` accordingly | PASS: at layout=(1,1) all 4 flags remain true (regression); at 2×2 exactly 1 flag true per sub-rank |
| 6 | Generalize `fv3_gpu_dyncore_bridge.F90`'s `comm_size==6` gate and `build_rank_of_tile` to `comm_size == 6*npes_per_tile`, reading `layout`/local-(i,j) from the Fortran side | STOP if `Atm` does not already expose `layout`/`npes_per_tile` cross-checked against Stage 9 |
| 7 | Wire host-staged `MpiHaloTransport` to the new per-segment neighbor list (pack/unpack mechanics unchanged, Stage 11) | PASS: bitwise-identical to today's transport at layout=(1,1) |
| 8 | End-to-end 2×2/face (24-rank) smoke at a SMALL grid (not C1024) with the full sentinel harness (Stage 16), decomposition-invariant correctness check (Stage 17) | PASS gate for any future C1024 attempt; NOT a C1024 run itself |

Phase 6 is flagged STOP-first because Stage 9 found the Fortran-side
`layout`/`npes_per_tile` fields are *implied*, not directly re-verified in
this session — the first concrete implementation task should confirm this
before writing any bridge code.

## 16. Test Harness Design (sentinel fields, not implemented)

**Reuse, do not reinvent**: extend the existing three-part harness
(`tests/runtime/test_halo_topology.cpp` host-only algebra unit tests,
`tests/integration/test_halo_exchange_multitile.cu` device sentinel-provenance
smoke, `tests/integration/test_halo_mpi_transport.cu` real-MPI variant) with a
`layout=(2,2)` mode. Extend the existing provenance encoding
(`value(tile,i,j,k)` from `test_halo_exchange_multitile.cu`) to
`value(face, bi, bj, i, j, k)`, still exact in R4/R8 integer range. Decode
after exchange and assert: intra-face neighbors decode to the correct
adjacent (bi,bj) on the SAME face with no sign flip; cross-face neighbors
decode to the correct opposite face/segment with the existing sign table;
corner cells decode to the correct one of {diagonal same-face, mixed,
true-panel} per Stage 6's classification. Include the same negative-control
discipline already used (`test_halo_exchange_multitile.cu`'s documented
fault injections) for at least: wrong intra-face neighbor, missing corner
donor, and a face-boundary segment picking the wrong half.

## 17. Decomposition-Invariant Validation Hierarchy

Do NOT assume global bitwise equality between the 6-rank and 24-rank results
(different reduction/communication order for anything not going through
`BITWISE_EXACT_SUM`, Stage 18). The validation ladder:

1. **IMPLEMENTED/COMPILED** — Phases 1-7 (Stage 15).
2. **GPU EXECUTED** — Phase 8 smoke at a small grid.
3. **Topology-level exact** — sentinel provenance round-trip (Stage 16):
   this level DOES require bitwise-exact decode, because it tests index/sign
   algebra, not floating-point summation order.
4. **Numerically validated at fixed decomposition** — 24-rank vs. its own
   Fortran 24-rank (`layout=(2,2)`) run at the SAME decomposition, same
   round-off tolerance methodology as Gate C2 (P103/P105 precedent).
5. **Cross-decomposition validated** — 24-rank CUDA vs. 6-rank CUDA/Fortran:
   expect round-off-level agreement only (like P103's C512 2-GPU vs. 1-GPU
   bitwise-exact result, which held because that change was routing-only,
   not decomposition-order-changing) EXCEPT wherever `BITWISE_EXACT_SUM` is
   in the path, where exact reproducibility across decomposition is the
   explicit FMS design intent (Stage 18).

## 18. BITWISE_EXACT_SUM Interaction (determination, not a policy change)

`BITWISE_EXACT_SUM` is an FMS `mpp_global_sum` flag
(`mpp_domains_mod::BITWISE_EXACT_SUM`, used e.g. in
`fv_grid_utils.F90:2939-2943`'s `sflag` selection) that selects a
decomposition-independent (ordered-tree) summation algorithm, as distinct
from the faster `NON_BITWISE_EXACT_SUM` default. It is **entirely a Fortran/
FMS-native mechanism, unrelated to and untouched by the CUDA dyn_core halo
work in this design** — `REMAP_TOTALENERGYCON` (the one dyn_core stage that
uses `mpp_global_sum` heavily, CURRENT_STATE.md §5) remains
**Fortran-only/BLOCKED BY COMMUNICATION** and was never ported to CUDA. A
24-rank decomposition simply hands FMS a different `domain2d` (built by the
same `mpp_define_mosaic`/`layout` call already generalized for
multi-rank-per-tile, Stage 4) — `mpp_global_sum`'s bitwise-exact mode is
*designed* to be reproducible across exactly this kind of decomposition
change. **No policy change needed or made; no interaction gap found.**

## 19. Communication Cost Model

Per-rank halo width `ng=3` (Stage 2), local domain edge length `n` (owned
cells per side).

| Config | Ranks | Local domain (n×n) | Halo cells/edge (3-ring) | Cardinal neighbors/rank | Corner neighbors/rank | Messages/rank (approx.) |
|---|---|---|---|---|---|---|
| C512, 1 rank/face (current) | 6 | 511×511 | 3×511 | 4 (all cross-face) | 4 (all true panel, local pole algebra, 0 messages) | 4 edge + seam/corner service calls |
| C1024, 2×2/face (target) | 24 | 511×511 (same) | 3×511 | 4 (2 intra-face + 2 cross-face, per boundary block) | 4 (1 true panel [0 msg, local], 1 intra-face diagonal [1 msg], 2 mixed [1 msg each]) | 4 edge + up to 3 corner messages |
| C1024, 1 rank/face (naive, memory-infeasible per P102) | 6 | 1023×1023 | 3×1023 | 4 (all cross-face) | 4 (all true panel) | 4 edge |

Per-rank halo **byte volume** at 24-rank C1024 is essentially the same as
today's per-rank C512 volume (identical local domain size, Stage 12) — the
cost-model delta from 6→24 ranks is almost entirely in **message count**
(new small corner messages; face-boundary edges split into ≤2 segments) and
**total system message count** (24 ranks × up to 7 neighbor directions vs. 6
ranks × 4), not per-rank byte volume.

## 20. Memory Scaling Estimate (4 / 6 / 12 GPUs, ESTIMATED — not measured)

Anchored to two already-measured points (A100-SXM4-40GB, `docs/performance/
p101_production_gpu_memory_inventory_2026-09-29.md`,
`p103_c512_2gpu_validation.md`): C512, 6 ranks total —
1 GPU/6-ranks-per-GPU ≈ 40.3 GiB/GPU (near-saturated, ~70-250 MiB free);
2 GPU/3-ranks-per-GPU ≈ 19.68 GiB/GPU ⇒ **≈6.56 GiB/rank** (converged,
production defaults). Because C1024/24-rank's local domain per rank
(511×511, Stage 19) is the SAME size as C512's whole-face local domain, the
**per-rank footprint is projected to be comparable, ≈6.5-7 GiB/rank** (same
field set, same vertical levels assumption; not re-measured for C1024).

| GPUs | Ranks/GPU (24 total) | Projected mem/GPU | Headroom vs. 40 GiB | Risk |
|---|---|---|---|---|
| 4 | 6 | ≈39.4 GiB | ~0.6 GiB | **UNSAFE** — same density pattern as the documented unsafe C512 6-ranks/1-GPU case |
| 6 | 4 | ≈26.2 GiB | ~13.8 GiB | Moderate, comparable ratio to C512's validated 2-GPU/3-ranks-per-GPU config |
| 12 | 2 | ≈13.1 GiB | ~26.9 GiB | Generous, comparable to a hypothetical C512 4-6 GPU spread (P102, memory-only, unmeasured for performance) |

**Estimate only** — does not account for any C1024-specific vertical-level
count difference, additional per-segment buffer overhead (Stage 12), or the
new corner-message buffers, all of which are believed small relative to the
6.5 GiB/rank baseline but are not separately quantified here.

## 21. Implementation Scope Estimate

**Complexity: MEDIUM.** Not LARGE, because Stage 4 found the hard
prerequisite (decomposition metadata, canonical numbering) already exists in
FMS/Fortran and needs only to be read, not invented; not SMALL, because the
CUDA-side `TopologyView`/`EdgeNeighbor` data model and its ~12 halo-group call
sites in `fv3_gpu_runtime.cpp` are genuinely single-neighbor-per-edge today
and need real structural change (Stage 15 phases 1-5), plus a bridge-side
`comm_size` relaxation with Fortran-side metadata plumbing (phase 6).

Files requiring change (scope estimate, not LOC count):

- `cuda/dyn_core/include/fv3_gpu_halo_topology.hpp` + `.cpp` — data model,
  neighbor-segment/corner generalization (Stage 1 items 5, 6).
- `cuda/dyn_core/include/fv3_gpu_internal.hpp` — `TopologyView` struct.
- `cuda/dyn_core/cgrid/fv3_cgrid_corners.hpp` (+ its `.cu`) — corner-flag
  consumer, no kernel-body change (Stage 13).
- `cuda/dyn_core/include/fv3_gpu_halo_peer.hpp`, `fv3_gpu_halo_mpi.hpp` (its
  own independent `comm_size==6` construction-time gate, Stage 1 item 8),
  `runtime/fv3_gpu_halo_mpi.cpp` — neighbor-segment iteration plus a new
  corner/diagonal-message path in the transport backends (Stage 11, 13).
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` — `fv3_gpu_set_cubed_sphere_
  topology`'s `mpi_size != cubed_sphere_tiles` gate (Stage 1 item 4).
- `cuda/interface/fv3_gpu_dyncore_bridge.F90` — `comm_size==6` gate,
  `build_rank_of_tile`, new layout/local-(i,j) plumbing (Stage 1 items 1-3).
- New/extended tests: `tests/runtime/test_halo_topology.cpp`,
  `tests/integration/test_halo_exchange_multitile.cu`,
  `tests/integration/test_halo_mpi_transport.cu` (Stage 16).

No change anywhere in `State_To_FV`, PT/DELZ, REMAP_UV, REMAP_Q,
REMAP_PRESSUREVARS, tracer CUDA, `FV_To_State`, or any production selector
default (all explicitly out of scope and untouched).

## 22. Top Risks (ranked)

1. **Stage 9's Fortran-side `layout`/`npes_per_tile` availability is implied,
   not directly re-verified this session** (change-budget boundary) — Phase
   6 (Stage 15) is gated STOP-first on confirming this before any bridge code
   is written; if it turns out `Atm`'s CUDA-visible bounds type does not
   already carry it, this becomes a larger Fortran-side plumbing task.
2. **Corner ownership precedence at mixed (type D) corners is resolved "from
   source separately" per the existing code's own comment** (Stage 5/13) —
   this is exactly the piece `make_cubed_sphere_topology` today defers
   (`corner owners... leave unset (-1)`); generalizing it correctly for 24
   ranks requires establishing that precedence rule from FMS source, not
   guessing it, or corner data could silently come from the wrong neighbor.
   **Compounding this (sub-investigation C): the diagonal/corner message path
   itself does not exist in any form today** (today's corner-fill is entirely
   local/zero-MPI, Stage 13) — so this risk is not just "which neighbor owns
   the corner" but "build a new message path and then get ownership right on
   it," a larger unit of new work than a pure precedence-rule lookup.
3. **Message-count growth (Stage 19) at `MpiHaloTransport`'s host-staged path
   could shift the balance between many-small-messages latency and the
   already-established (P43-48) finding that CUDA-aware MPI gave negligible
   benefit at 6 ranks** — a 24-rank topology with more, smaller messages per
   rank may have a different latency-vs-bandwidth balance than the 6-rank
   case; performance revalidation, not correctness, is at risk here.
2b. Memory (Stage 20) is a real but *quantifiable* risk, not open-ended: the
    4-GPU density is flagged UNSAFE by direct analogy to an already-known
    unsafe pattern, not a new unknown.
4. **Test-harness generalization (Stage 16) is nontrivial in its own right**
   — the existing sentinel-provenance test is a carefully-built, independently
   negative-control-verified artifact; extending its encoding to
   (face,bi,bj) without weakening its fault-detection guarantees needs the
   same rigor as the original (documented fault injections), which is real
   effort, not a mechanical extension.
5. **No C1024 `bdt` stability search exists yet** (CURRENT_STATE.md §7,
   §9 — unrelated to halo architecture) — even after this design closes, an
   actual C1024 run needs its own independent numerical-stability work,
   compounding schedule risk if the two efforts are not sequenced
   deliberately.

## 23. GO/NO-GO Classification

**Classification: MODERATE HALO GENERALIZATION.**

Not "EXISTING INFRASTRUCTURE WIRING" — the CUDA-side topology/transport data
model genuinely needs new code (Stages 1 items 5-6, 15 phases 1-5), not just
a config flag flip. Not "MAJOR COMMUNICATION ARCHITECTURE CHANGE" — the
transport mechanics (pack/transfer/unpack, Stage 11), the seam rotation/sign
algebra (Stage 3/8), and the decomposition metadata/numbering (Stage 4/10)
are ALL already correct and reusable; only the neighbor-graph *data
structure* and *donor-selection logic* feeding them need to grow from
single-neighbor-per-edge to multi-segment-per-edge-plus-corners.

**Recommended next bounded task (exactly one, NOT full C1024):**
**Implement and validate Phases 1-4 of Stage 15 (TopologyView/EdgeNeighbor
generalization + intra-face cardinal + cross-face segment donor selection +
intra-face/mixed corner exchange) at a SMALL host-only/single-node grid using
the extended sentinel-provenance test harness (Stage 16), entirely on the
CUDA side, with `layout=(1,1)` kept as an exact-regression check throughout.
Do NOT touch the Fortran bridge's `comm_size` gate or attempt any real
24-rank MPI run in that task** — Phase 6 (bridge/Fortran-metadata plumbing)
and Phase 8 (end-to-end 24-rank smoke) should each be their own
separately-scoped follow-on task, confirmed with the user first, per Stage
15's own phase gating.

**Status update**: Phases 1-4 above are CLOSED — see
`docs/performance/p109_cuda_subtile_halo_phases1_4_2026-09-30.md`. Phase 5
design/scoping is CLOSED — see
`docs/performance/p110_subtile_halo_transport_phase5_design_2026-09-30.md`
(selected slice: additive `TopologyView` builder + synthetic-rank field-data
test harness, edges only, no bridge/gate change). Phase 6 (bridge/Fortran
plumbing) and Phase 8 (real 24-rank run) remain open, unauthorized.

## Final Status

```
No production source changes: YES
No commit/push: YES
```

**C1024 DECOMPOSITION ARCHITECTURE MAPPED — READY FOR TOPOLOGY/TRANSPORT
GENERALIZATION PHASES 1-4 (SMALL-GRID, SINGLE-NODE, CUDA-SIDE ONLY)**
