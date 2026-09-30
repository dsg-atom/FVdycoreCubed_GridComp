# Prompt 110 — Design Phase 5: Wire Sub-Tile Topology into CUDA Halo Transport

Date: 2026-09-30
Branch: `optimization/cuda-fv3-performance`
Status: **DESIGN/SCOPING ONLY — CLOSED. No production source changed. No
commit/push.**

## 0. Scope and ground-truth reconciliation

This is a bounded source-mapping + design task, not an implementation task.
Deliverable is this document plus pointer updates in `p108`/`p109`/the
roadmap/`CURRENT_STATE.md`. No CUDA/C++/Fortran production source was
edited; no build behavior changed; nothing committed or pushed.

**Discrepancy vs. the Prompt 110 task text**: the task's "VERIFIED INPUT
FROM PROMPT 109" section states the transport-side `comm_size==6`-equivalent
gate lives in `fv3_gpu_halo_mpi.hpp`. Direct read this session (and the
prior Prompt 109 doc, §4 table) confirms it actually lives in
**`cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp:416`**, inside
`create_mpi_transport()`:

```cpp
if (size != cubed_sphere_tiles)
  return refuse("the halo communicator must have exactly 6 ranks, one per "
                "cubed-sphere tile; a different decomposition is not "
                "supported by this transport");
```

`fv3_gpu_halo_mpi.hpp` only declares the factory function's signature; the
gate itself is in the `.cpp` implementation. This does not change any
conclusion below — the gate's existence and behavior are exactly as
described, just at a different file. Treat the `.cpp` location as
authoritative going forward.

The other two gates are confirmed exactly as stated:
- Bridge: `cuda/interface/fv3_gpu_dyncore_bridge.F90:163`,
  `if (comm_size /= 6)` (plus the fixed `rank_of_tile(6)` array at line 146
  and `build_rank_of_tile` at 1432-1463, itself hardcoded to 6 tiles).
- Runtime: `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1562`,
  `fv3_gpu_set_cubed_sphere_topology`'s
  `if (c->data.create.mpi_size != halo::cubed_sphere_tiles)`.

All three remain untouched by this task.

## 1. Current production halo pipeline (Stage 1)

`perform_halo_exchange()` (`fv3_gpu_runtime.cpp:888-949`) is the single
orchestration function for every one of the 12 halo groups
(`fv3_gpu_halo_exchange.hpp`'s `halo_group_spec()`):

| Step | Function | Topology input | Notes |
|---|---|---|---|
| Pack | `launch_halo_pack` (`fv3_gpu_halo_exchange.cu:206-249`) | **none** | reads only this rank's own `nx`/`ny` and field view; writes into `(field,edge)`-addressed slots |
| Order | event record | — | pack ordered after compute |
| Transfer | `HaloTransport::transfer()` → `PeerHaloTransport::transfer` (`fv3_gpu_runtime.cpp`) or `MpiHaloTransport::transfer` (`fv3_gpu_halo_mpi.cpp:259-302`) | `TopologyView.neighbors[4]`: `.tile`/`.rank`/`.edge` | Peer: D2D copy indexed by `.tile`. Mpi: Irecv/Isend addressed by `.rank` only, `.edge` used only by `check_topology_once()`'s validation |
| Unpack | `launch_halo_unpack` (`fv3_gpu_halo_exchange.cu:312-380+`) | `TopologyView.neighbors[q].edge` drives all rotation/sign/donor-slot math; `.tile` is also read, but only as a `1..cubed_sphere_tiles` sanity bound (`fv3_gpu_halo_exchange.cu:351,377`), never as a peer-buffer index — **correction, Prompt 112 Stage 15**: the original "never `.tile`" claim here was inaccurate | derives `reversed`/rotation/`vector_sign` from `donor_edge` alone; existing assert `along[q] != donor[q] -> FAIL` already guards a segment-length mismatch |
| Release | `HaloTransport::release_send()` | — | frees send buffer for reuse |

`TopologyView` is built once, at attach time, by
`make_cubed_sphere_topology()` (`fv3_gpu_halo_topology.cpp`), which requires
`npx == npy` (square tiles) and populates exactly 4 `EdgeNeighbor` entries
(`.rank`, `.tile`, `.edge`, `.orientation`, `.vector_sign_x/y`) per rank via
the unchanged mosaic/seam-rotation algebra. It leaves `corner_owner_rank[4]`
at the -1 sentinel — an existing, never-populated struct field.

Corner-fill (`fv3_cgrid_corners.hpp`, WP-A-owned) is a **separate, fully
local, zero-MPI mechanism** — driven only by `CornerFillArgs{npx,npy,
sw/se/ne/nw_corner}` — not part of the halo-group dispatch enum, not
touched by `HaloTransport` at all, called directly from WP-A/C-grid Fortran
call sites.

## 2. Where current kernels/consumers touch the old one-rank-per-face model (Stage 2)

| Consumer | Reads | Sub-tile impact |
|---|---|---|
| `make_cubed_sphere_topology` (builder) | `rank_of_tile[6]`, exactly 1 rank/tile | **NEW, additive builder needed** — legacy builder untouched |
| `launch_halo_pack` | nothing topology-shaped | **NONE — already compatible** |
| `launch_halo_unpack` | `neighbors[q].edge` only | **NONE at the signature level — metadata-only**, correctness depends on the builder populating `.edge`/`.orientation`/`.vector_sign` correctly for a sub-tile's real donor |
| `launch_halo_transfer` (Model B, in-process D2D) | `neighbors[q].tile`, fixed `cubed_sphere_tiles+1`-sized peer arrays | Structurally 6-tile/6-thread; **not generalized — out of scope** (see §4) |
| `PeerHaloTransport::transfer` (`fv3_gpu_runtime.cpp`) | same as above | same — out of scope |
| `MpiHaloTransport::transfer`/`check_topology_once` | `neighbors[q].{tile,rank,edge}`, `tile_of_rank_[6]`, `size_==6` gate | Real MPI addressing already keys on `.rank` alone; the fixed-6 table and gate are the actual generalization surface — **deferred to Phase 6/7**, not this document's implementation recommendation |
| Bridge (`comm_size` gate, `rank_of_tile(6)`, `build_rank_of_tile`) | fixed 6 | **Phase 6, explicitly out of scope** |
| `fv3_gpu_set_cubed_sphere_topology` gate | `mpi_size != cubed_sphere_tiles` | **Phase 6, explicitly out of scope** |
| `fv3_cgrid_corners.{hpp,cu}` | local corner flags only | edges untouched; diagonal/mixed corners need a **wholly new** mechanism, not a generalization (§5) |

Net finding: **exactly one new production-shaped artifact is required for
edges** — a new topology builder that produces the existing `TopologyView`
shape from `SubTileId`+layout using the already-validated
`subtile_edge_neighbor()` math. Neither `launch_halo_pack` nor
`launch_halo_unpack` needs a signature or body change.

## 3. Old vs. new data model — migration strategy (Stage 3)

Decision: **adapt new into existing descriptors (no struct changes).**

- Keep `EdgeNeighbor`/`TopologyView` as the *sole* runtime-consumed shape.
  Do not add a `segment_index`/`segment_count` field to `EdgeNeighbor` —
  cross-face segment identity is needed only to *identify* the one donor at
  attach time; once identified, pack/unpack see an ordinary single-donor
  edge exchange, identical in shape to the legacy 1x1 case.
- `fv3_gpu_halo_subtile_topology.hpp` remains the sole *math* authority
  (`subtile_edge_neighbor`, `subtile_corner_donor`); it is not itself
  consumed at runtime — a new builder translates its output into
  `TopologyView` once, at attach time, exactly as `make_cubed_sphere_topology`
  does today for the legacy case.
- Corners: reuse the existing, currently-unpopulated
  `TopologyView.corner_owner_rank[4]` field rather than inventing a new one,
  when corner transport is eventually implemented (deferred, §5).

This keeps exactly one struct shape flowing through pack/transfer/unpack for
both layouts, and confines all sub-tile-specific logic to the attach-time
builder.

## 4. Pack/unpack/HaloTransport contract for same-face and cross-face edges (Stages 4-9)

- **Pack: UNCHANGED.** Purely local; no topology dependency of any kind.
- **Unpack: METADATA-ONLY.** Depends only on `donor_edge` (already supplies
  rotation/reversal/vector-sign); the existing along-extent consistency
  assert already guards mismatched segment lengths.
- **Same-face edges**: `subtile_edge_neighbor`'s same-face branch already
  yields the identity/aligned orientation via the unchanged
  `seam_orientation`/`seam_rotation(mine, opposite(mine))` — no bypass
  logic needed; zero rotation falls out of the existing algebra for free.
- **Cross-face segmented edges**: also fall out for free below the topology
  layer — pack/unpack only ever see "my one local edge, one donor edge";
  the segment concept is fully resolved into a single donor identity before
  pack/unpack run, exactly like the legacy case.
- **`HaloTransport` audit**:
  - `attach(tile)`: `tile` semantically becomes "my slot id" (a global rank
    under the new model) — reinterpretation, not a structural rewrite.
  - `transfer(..., const TopologyView&, ...)`: **no signature change**
    under the Stage-3 decision (topology stays `TopologyView`-shaped).
  - `neighbors[4]`: correct at *any* layout — always exactly 4 cardinal
    edges per sub-rank, not a limitation.
  - `MpiHaloTransport`'s fixed `tile_of_rank_` / `size_==6` gate is the one
    real generalization surface, but relaxing it is Phase 6/7 (touches a
    production gate) — **explicitly not part of this document's
    recommended Phase-5 slice**.
  - `PeerHaloTransport`/`HaloPeerGroup` (Model B): hardcoded
    `cubed_sphere_tiles+1`-sized arrays. **Recommend NOT generalizing** —
    a real 24-rank decomposition is 24 OS processes, not 24 in-process
    threads; Model B's role at scale is moot. It may still be reused,
    unmodified, *inside a test-only harness* with a locally-scoped larger
    peer table (see §9) without touching the production class.

Classification: pack — no change; unpack — metadata-only; abstract
`HaloTransport` interface — no change; `MpiHaloTransport` internals — small
generalization, deferred; `PeerHaloTransport`/bridge/runtime gates — major
change, explicitly deferred to Phase 6/7.

## 5. Corners (Stages 10-11, 14)

Corner-fill kernels are local/WP-A-owned/zero-MPI and entirely disconnected
from `HaloTransport`. `subtile_corner_donor` already returns the rank
*itself* for `true_cube_corner` — i.e., that case is already a pure local
fill and needs no new mechanism at all. `intra_face_diagonal` and `mixed`
corners have **no existing mechanism whatsoever**: this is a new message
type, not a generalization of an existing one.

- **Recommended integration shape (not implemented)**: extend the existing
  per-group message loop from 4 slots (edges) to up to 8 (4 edges + 4
  corners), reusing the already-built `subtile_edge_tag`/`subtile_corner_tag`
  scheme, rather than fully independent corner messages (more duplication)
  or folding corners into edge payloads (different, smaller geometry, poor
  fit) or sequential edge-relay propagation (adds serialization rounds for
  no benefit — direct donor identity is already known).
- **Cost estimate**: one corner message ≈ halo_width² × npz × field_count ×
  8 bytes (e.g. 3×3×91×4×8 ≈ 78 KB), ×4 corners × (send+recv) × (host+device)
  ≈ low single-digit MB/rank — negligible next to edge buffers (which scale
  with along-extent, hundreds of cells). No full-field allocation required.
- **Kernel impact**: `fill_4corners`/`fill2_4corners` need **no change**
  for `true_cube_corner`. `intra_face_diagonal`/`mixed` need a **new
  kernel** (or a new remote-input branch) that reads a *remote* corner
  buffer instead of only local halo cells — classified NEW, not a
  generalization, and explicitly deferred.

## 6. MPI tags, memory, performance (Stages 12, 16-17)

- **Tag scheme**: today's production formula is
  `message_tag(group,edge,field) = (group*4+edge)*max_halo_group_fields+field`
  (0-3 edge slots). Proposed (not implemented) extension for a future
  edge+corner unification: widen the slot domain to 0-7
  (`(group*8+q)*max_halo_group_fields+field`), doubling the tag range to
  ~415 — still far under `MPI_TAG_UB` (32767), and collision-free by the
  same disjoint-triple argument already used for edges today.
- **Memory**: the new builder adds no allocations (pure metadata, computed
  once at attach time, same `TopologyView` size as today). Deferred corner
  buffers would add low single-digit MB/rank (§5) — negligible against the
  current ~6-7 GiB/rank footprint.
- **Performance**: per-rank edge message count/bytes is **identical**
  between layout=1x1 and layout=NxM at fixed local grid size — each local
  edge always has exactly one donor, in full, regardless of layout. Going
  to more, smaller sub-ranks is ordinary weak/strong scaling, not a
  per-rank message-count increase. Deferred corners would add up to 4 more
  small messages per rank per group — bounded, not a launch/message
  explosion. No wall-clock performance claim is made by this design task.

## 7. layout=(1,1) compatibility strategy (Stage 18)

**One generalized builder handles both layouts.**
`subtile_edge_neighbor(SubTileId{tile,0,0}, edge, 1, 1)` already reduces
exactly to the legacy calculation (proven by Prompt 109's T1-T2 exact
regression). The new builder should be the one general math path for both
layout=1x1 and layout=NxM; the *existing* `make_cubed_sphere_topology`
should be left wired into production, unmodified, until a future Phase 6
call-site swap — avoiding any change to current production behavior. No
adapter/legacy-branch complexity is needed at the math layer; only the call
site is a future decision.

## 8. Selected Phase-5 implementation slice (Stage 19)

**Refined Option A — additive builder + field-data test harness, zero
production dispatch touched:**

1. A new, additive `SubTileId`+layout → `TopologyView` builder (new
   header/source, mirrors `make_cubed_sphere_topology`'s shape, ~100-150
   LOC). No existing file modified.
2. A new test-only harness that feeds that `TopologyView` into the **real,
   unmodified** `launch_halo_pack`/`launch_halo_unpack` (and a
   test-local, non-production peer array, not `HaloPeerGroup`) against a
   synthetic multi-sub-rank single-process setup, to prove **actual field
   data** — not just routing — moves correctly for same-face and
   cross-face edge exchange at layout=2x2, alongside a mandatory layout=1x1
   exact-regression check.

This closes the one gap Prompt 109 left open: Prompt 109 validated
*topology/routing* only, never real halo field data through the production
pack/unpack kernels. Corners and cross-process multi-rank MPI transport
remain deferred. The harness never calls `fv3_gpu_set_cubed_sphere_topology`
or the bridge, so no production gate is touched or relaxed.

Why not other options: a full production wiring (calling this from
`fv3_gpu_set_cubed_sphere_topology`) would require relaxing the runtime gate
— explicitly forbidden this task. A corners-first slice has no existing
mechanism to extend (§5) and is strictly harder/riskier than edges. A
transport-rewrite-first slice touches `MpiHaloTransport`'s production gate
for no additional proof value beyond what a test harness already gives.

## 9. Test plan (Stage 20)

- layout=1x1 exact regression: new builder vs. legacy, bit-identical
  `TopologyView` for all 6 tiles.
- layout=2x2 same-face sentinel: known field pattern, verify unpacked halo
  matches the neighbor's value with zero rotation.
- layout=2x2 cross-face segment sentinel: verify the correct segment donor's
  data lands, including reversal where applicable.
- Vector (U/V) orientation sentinel: a vector field pair through a rotated
  cross-face seam, verify component swap + sign.
- Real A100 execution (small grid, layout=2x2, synthetic multi-rank
  harness), `compute-sanitizer` clean.
- Reuse (not reimplement) Prompt 109's existing 24-rank real-MPI routing
  test for donor-identity coverage.

## 10. Source change budget (Stage 21)

| Artifact | Purpose | Size |
|---|---|---|
| New topology-builder header/source | `SubTileId`+layout → `TopologyView` | SMALL (~100-150 LOC) |
| New field-exchange test harness | proves real pack/unpack correctness at 2x2 | MEDIUM (~250-400 LOC) |
| Docs (this file + pointer updates) | — | N/A |

No existing production file is modified. Overall: **SMALL**.

## 11. Phase-6 bridge metadata contract (Stage 22)

Required, not yet plumbed: `layout_x`/`layout_y` (uniform across faces, per
current scope) and either the sub-rank's own `(sub_i,sub_j)` or a
face-major global-rank numbering agreement with FMS's `pe_start`
(`subtile_from_global_rank` already assumes this). FMS/`mpp_define_layout`
already carries `layout(1:2)` in this same numbering (confirmed by Prompt
108) — so the *data* exists upstream; the bridge today only reads/passes
`comm_size`+`comm_rank`+the fixed 6-entry `rank_of_tile`, not `layout(1:2)`.
Phase 6 must add extraction of `layout_x/layout_y` from the FMS domain and
pass them across the ABI as an additive field (consistent with the existing
`internal_contract_version` additive-tail convention) — not attempted here.

## 12. Stop conditions (Stage 23)

None triggered. Pack needs zero change; unpack needs only correct metadata;
`HaloTransport`'s abstract interface needs no signature change; the one
real generalization surface (`MpiHaloTransport`'s fixed-6 table/gate) is
narrowly scoped and explicitly deferred, not blocking. Corner transport is
additive-new, does not require redesigning unrelated dynamical kernels, and
does not block an edges-only Phase-5 slice. No full-field copies are needed
anywhere. layout=1x1 remains exact by construction. This design does not
depend on relaxing the Fortran `comm_size` gate — the recommended slice is
entirely test-harness-side.

```
SUB-TILE HALO PHASE-5 DESIGN COMPLETE — READY FOR ADDITIVE SUB-TILE
TOPOLOGY BUILDER + SYNTHETIC-RANK FIELD-DATA TEST HARNESS (EDGES ONLY,
LAYOUT=2x2, NO BRIDGE/GATE CHANGE)
```

## 13. Update (Prompt 111, CLOSED) — implementation found a real blocker

Prompt 111 implemented the additive builder and field-data harness
selected above. §7's "one real generalization surface" undersold the
`MpiHaloTransport`-only gate: the SAME `nb.tile < 1 || nb.tile >
cubed_sphere_tiles` bound also lives in `launch_halo_transfer` itself
(`fv3_gpu_halo_exchange.cu:273`), which this design assumed needed zero
changes (§4/§9 "pack needs zero change... unpack needs only correct
metadata"). That assumption was correct for pack and unpack but not for
transfer — `launch_halo_transfer`'s `peer_send` table is indexed
directly by `EdgeNeighbor.tile`, hardcoded to the 6-tile model
(`fv3_gpu_halo_peer.hpp`'s `PeerMember member_[cubed_sphere_tiles+1]`).
At layout=2x2 the builder's `.tile = donor_rank+1` slot id legitimately
exceeds 6 for any cross-face donor, and the real, unmodified
`launch_halo_transfer` rejects it (`FV3_GPU_INVALID_ARGUMENT`) before any
field data moves. Confirmed layout=1x1 (N=6) still passes bitwise
(0/20736 checks failed); confirmed all 6 halo groups fail identically at
layout=2x2 (100% attributable to this one gate, not a per-group issue).
See `p111_subtile_topology_fielddata_validation_2026-09-30.md` for full
detail. A real Phase-6 fix must generalize this indexing/gate together
with the bridge metadata plumbing already planned above — not a new,
separate surface, but a larger one than §7 scoped.
