#pragma once
//===========================================================================
// Prompt 111 (Phase 5, edges-only slice): ADDITIVE SubTileId+layout ->
// TopologyView builder.
//
// SCOPE. This is the one new production-shaped artifact Prompt 110 §2/§8
// identified as required: a translator from Prompt 109's math-only
// SubTileId/subtile_edge_neighbor donor-identification layer
// (fv3_gpu_halo_subtile_topology.hpp) into the EXISTING TopologyView shape
// (fv3_gpu_internal.hpp) that launch_halo_pack/launch_halo_unpack/
// launch_halo_transfer already consume unmodified. It duplicates no seam
// math -- every field below is copied from subtile_edge_neighbor's/
// subtile_corner_donor's own output, never re-derived.
//
// It does NOT touch:
//   - fv3_gpu_halo_topology.{hpp,cpp}      (legacy one-rank-per-face builder,
//     left wired into production, unmodified);
//   - fv3_gpu_halo_subtile_topology.hpp    (the math authority, unmodified);
//   - launch_halo_pack/launch_halo_transfer/launch_halo_unpack (unmodified,
//     called by the Prompt 111 field-data test harness with this builder's
//     output in place of make_cubed_sphere_topology's);
//   - the bridge comm_size==6 gate, fv3_gpu_set_cubed_sphere_topology's
//     mpi_size!=cubed_sphere_tiles gate, or MpiHaloTransport::
//     create_mpi_transport's size!=cubed_sphere_tiles gate. This builder is
//     reachable only from tests; no production call site references it.
//
// FIELD MAPPING (Prompt 112 correction): the legacy builder keeps two
// DISTINCT identities per neighbor -- EdgeNeighbor.tile is the donor's
// PHYSICAL CUBE FACE (1..6, a topology/seam-math fact, independent of how
// many ranks sit on that face) and EdgeNeighbor.rank is the donor's actual
// TRANSPORT/MPI PEER IDENTITY (0..comm_size-1, via whatever rank-of-tile
// permutation the deployment uses -- see fv3_gpu_halo_topology.cpp's
// `nb.rank = rank_of_tile[d.tile - 1]; nb.tile = d.tile;`). Prompt 111
// originally set `.tile = donor_rank + 1` (a transport-slot id, conflating
// the two identities); that was harmless only because layout=1x1 has
// exactly one rank per face, so slot id and face id coincide. At sub-tile
// granularity they do not: this builder now sets
//     .rank = donor's global rank                    (transport peer identity)
//     .tile = donor's physical cube face (sn.donor.face, 1..6)
// matching the legacy builder's own convention exactly. launch_halo_transfer
// (Prompt 112) indexes its peer_send table by `.rank`, not `.tile`; unpack
// still reads only `.edge` for rotation/reversal and `.tile` solely as a
// physical-face connectivity check (1..cubed_sphere_tiles, always true for a
// global cube) -- neither depends on `.tile` being a transport slot.
//
// LAYOUT=1x1 EXACT REDUCTION: subtile_global_rank(SubTileId{face,0,0},1,1)
// == face-1 (0-based, face-major numbering -- Prompt 109 T1). With the same
// rank_of_tile[t-1]=t-1 convention the existing regression tests already use,
// this makes .rank here equal legacy's rank_of_tile[donor.face-1] exactly,
// and .tile here (sn.donor.face, set directly, not derived from rank) equal
// legacy's donor face id exactly by construction -- the basis for Stage 2's
// exact-equivalence and Stage 12's bitwise pack/unpack regression tests.
//
// CORNER FIELDS: populated for 1x1-metadata-equivalence completeness only
// (Stage 2 compares "overlapping corner/edge flags"). Neither
// launch_halo_pack nor launch_halo_unpack reads any corner field, so this is
// NOT a corner MESSAGE mechanism and makes no corner-transport claim (Prompt
// 111 Stage 9). Corner index convention follows subtile_corner_edges'
// existing numbering (0=SW, 1=SE, 2=NW, 3=NE); corner_owner_rank[] uses the
// SAME index order. sw/se/ne/nw_corner is 1 only for a TRUE_CUBE_CORNER
// (local corner-fill algebra applies, matching legacy's unconditional 1 at
// layout=1x1); corner_owner_rank stays at the existing -1 sentinel for those,
// exactly as the legacy builder leaves it, and is set to the donor's global
// rank otherwise (mixed/intra_face_diagonal -- corner MESSAGE delivery for
// these remains unimplemented, per Prompt 110 §5).
//===========================================================================

#include "fv3_gpu_halo_subtile_topology.hpp"
#include "fv3_gpu_internal.hpp"

namespace fv3::gpu::halo {

/* Builds a TopologyView for sub-rank `me` in a layout_x x layout_y-per-face
 * synthetic decomposition, using ONLY subtile_edge_neighbor/
 * subtile_corner_donor/subtile_global_rank -- no new seam or rank algebra.
 * `npx_local`/`npy_local` are this sub-rank's OWN local corner-point counts
 * (must be square, mirroring the legacy npx==npy requirement); they are
 * opaque to the builder itself, only propagated into the output.
 *
 * Returns false (out left untouched) on any malformed input, matching
 * make_cubed_sphere_topology's contract.
 *
 * Host-only, deliberately NOT FV3_HALO_HD: mirrors make_cubed_sphere_
 * topology's own convention exactly -- TopologyView (std::array-backed) is
 * one-time-per-run host-side metadata setup, never built per-thread on the
 * device, so there is no device-callable use case, and marking it
 * __host__ __device__ would only produce nvcc constexpr-std::array
 * warnings for a code path that is never instantiated. */
inline bool make_subtile_cubed_sphere_topology(SubTileId me, int layout_x, int layout_y,
                                                int64_t npx_local, int64_t npy_local,
                                                TopologyView &out) noexcept {
  if (me.face < 1 || me.face > cubed_sphere_tiles) return false;
  if (layout_x < 1 || layout_y < 1) return false;
  if (me.sub_i < 0 || me.sub_i >= layout_x) return false;
  if (me.sub_j < 0 || me.sub_j >= layout_y) return false;
  if (npx_local < 2 || npy_local < 2) return false;
  if (npx_local != npy_local) return false;

  TopologyView t{};
  t.contract_version = internal_contract_version;
  t.tile = me.face;
  t.npx = npx_local;
  t.npy = npy_local;
  t.global_cubed_sphere = 1;
  t.nested = 0;
  t.regional = 0;
  t.periodic_x = 0;
  t.periodic_y = 0;

  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (TileEdge e : edges) {
    const SubTileEdgeNeighbor sn = subtile_edge_neighbor(me, e, layout_x, layout_y);
    const int32_t donor_rank = subtile_global_rank(sn.donor, layout_x, layout_y);
    EdgeNeighbor &nb = t.neighbors[static_cast<size_t>(e)];
    nb.rank = donor_rank;
    nb.tile = static_cast<int16_t>(sn.donor.face);
    nb.edge = sn.donor_edge;
    nb.orientation = sn.orientation;
    nb.vector_sign_x = sn.vector_sign_x;
    nb.vector_sign_y = sn.vector_sign_y;
  }

  bool corner_is_true_cube[4];
  int8_t corner_owner[4];
  for (int c = 0; c < 4; ++c) {
    SeamVectorTransform xf{};
    const SubTileId donor = subtile_corner_donor(me, c, layout_x, layout_y, xf);
    const bool is_true_cube = (classify_subtile_corner(me, c, layout_x, layout_y) ==
                               SubTileCornerType::true_cube_corner);
    corner_is_true_cube[c] = is_true_cube;
    corner_owner[c] = is_true_cube
                          ? int8_t(-1)
                          : static_cast<int8_t>(subtile_global_rank(donor, layout_x, layout_y));
  }
  t.sw_corner = corner_is_true_cube[0] ? 1 : 0;
  t.se_corner = corner_is_true_cube[1] ? 1 : 0;
  t.nw_corner = corner_is_true_cube[2] ? 1 : 0;
  t.ne_corner = corner_is_true_cube[3] ? 1 : 0;
  t.corner_owner_rank = {corner_owner[0], corner_owner[1], corner_owner[2], corner_owner[3]};

  out = t;
  return true;
}

}  // namespace fv3::gpu::halo
