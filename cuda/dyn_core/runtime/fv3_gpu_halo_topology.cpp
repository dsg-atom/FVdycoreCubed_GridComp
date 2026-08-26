//===========================================================================
// WP-E: cubed-sphere halo connectivity builder.
//
// The table below IS the mosaic. It is a per-tile decode of the 12 contact
// lines specified identically in two independent authoritative places:
//
//   FVdycoreCubed_GridComp/CubeHalo.F90:213-261
//   GFDL_atmos_cubed_sphere (geos/v3.0.0, ca30e26)
//     tools/fv_mp_mod.F90:577-625      domain_decomp, case ( 6 )
//
// Contact lines as written there (tile1 edge, tile2 edge, and the index
// direction on each side; `rev` marks a contact whose along-seam index order
// runs opposite on the two sides):
//
//    1  T1 EAST  (j:1->ny)  <-> T2 WEST  (j:1->ny)
//    2  T1 NORTH (i:1->nx)  <-> T3 WEST  (j:ny->1)   rev
//    3  T1 WEST  (j:1->ny)  <-> T5 NORTH (i:nx->1)   rev
//    4  T1 SOUTH (i:1->nx)  <-> T6 NORTH (i:1->nx)
//    5  T2 NORTH (i:1->nx)  <-> T3 SOUTH (i:1->nx)
//    6  T2 EAST  (j:1->ny)  <-> T4 SOUTH (i:nx->1)   rev
//    7  T2 SOUTH (i:1->nx)  <-> T6 EAST  (j:ny->1)   rev
//    8  T3 EAST  (j:1->ny)  <-> T4 WEST  (j:1->ny)
//    9  T3 NORTH (i:1->nx)  <-> T5 WEST  (j:ny->1)   rev
//   10  T4 NORTH (i:1->nx)  <-> T5 SOUTH (i:1->nx)
//   11  T4 EAST  (j:1->ny)  <-> T6 SOUTH (i:nx->1)   rev
//   12  T5 EAST  (j:1->ny)  <-> T6 WEST  (j:1->ny)
//
// Two invariants hold over all 12 and are asserted at build time below, so a
// future edit that breaks the table cannot pass silently:
//
//   (a) reciprocity -- if my edge names (T,E) then T's edge E names me back;
//   (b) reversal occurs exactly when my edge and the donor edge lie on
//       different axes, equivalently when tile1 and tile2 share parity.
//       Every tile therefore has exactly two aligned and two rotated edges:
//       odd tiles (1,3,5) rotate on WEST and NORTH, even tiles (2,4,6) rotate
//       on EAST and SOUTH.
//===========================================================================

#include "fv3_gpu_halo_topology.hpp"

namespace fv3::gpu::halo {
namespace {

struct Donor {
  int8_t tile;      // 1-based
  TileEdge edge;    // the donor's OWN edge that abuts mine
};

constexpr TileEdge W = TileEdge::west;
constexpr TileEdge E = TileEdge::east;
constexpr TileEdge S = TileEdge::south;
constexpr TileEdge N = TileEdge::north;

/* Indexed [tile-1][static_cast<int>(TileEdge)]. TileEdge's enumerator order is
 * west, east, south, north (fv3_gpu_internal.hpp), so the columns below are in
 * that order and NOT in the W/E/S/N reading order of the comment block above by
 * accident -- they happen to coincide. */
constexpr Donor mosaic[cubed_sphere_tiles][4] = {
    /* tile 1 */ {{5, N}, {2, W}, {6, N}, {3, W}},
    /* tile 2 */ {{1, E}, {4, S}, {6, E}, {3, S}},
    /* tile 3 */ {{1, N}, {4, W}, {2, N}, {5, W}},
    /* tile 4 */ {{3, E}, {6, S}, {2, E}, {5, S}},
    /* tile 5 */ {{3, N}, {6, W}, {4, N}, {1, W}},
    /* tile 6 */ {{5, E}, {2, S}, {4, E}, {1, S}},
};

constexpr TileEdge all_edges[4] = {W, E, S, N};

/* (a) reciprocity, checked by the compiler. */
constexpr bool reciprocal_at(int tile, int e) {
  const Donor d = mosaic[tile - 1][e];
  const Donor back = mosaic[d.tile - 1][static_cast<int>(d.edge)];
  return back.tile == tile && back.edge == all_edges[e];
}
/* (b) reversal <=> different axis <=> same tile parity. */
constexpr bool parity_at(int tile, int e) {
  const Donor d = mosaic[tile - 1][e];
  const bool different_axis =
      edge_is_x_normal(all_edges[e]) != edge_is_x_normal(d.edge);
  const bool same_parity = ((tile & 1) == (int(d.tile) & 1));
  return different_axis == same_parity;
}
constexpr bool table_is_consistent() {
  for (int t = 1; t <= cubed_sphere_tiles; ++t)
    for (int e = 0; e < 4; ++e)
      if (!reciprocal_at(t, e) || !parity_at(t, e)) return false;
  return true;
}
static_assert(table_is_consistent(),
              "cubed-sphere mosaic table violates reciprocity or the "
              "reversal<=>parity invariant; re-derive it from "
              "CubeHalo.F90:213-261 before proceeding");

/* Every tile has exactly two aligned and two rotated edges. */
constexpr bool two_and_two() {
  for (int t = 1; t <= cubed_sphere_tiles; ++t) {
    int rotated = 0;
    for (int e = 0; e < 4; ++e)
      if (seam_is_reversed(all_edges[e], mosaic[t - 1][e].edge)) ++rotated;
    if (rotated != 2) return false;
  }
  return true;
}
static_assert(two_and_two(), "each cube tile must have 2 aligned and 2 rotated edges");

} // namespace

bool cubed_sphere_neighbor(int tile, TileEdge mine, int &donor_tile,
                           TileEdge &donor_edge) noexcept {
  if (tile < 1 || tile > cubed_sphere_tiles) return false;
  const int e = static_cast<int>(mine);
  if (e < 0 || e > 3) return false;
  const Donor d = mosaic[tile - 1][e];
  donor_tile = static_cast<int>(d.tile);
  donor_edge = d.edge;
  return true;
}

bool make_cubed_sphere_topology(int tile, int64_t npx, int64_t npy,
                                const int32_t *rank_of_tile,
                                TopologyView &out) noexcept {
  if (tile < 1 || tile > cubed_sphere_tiles) return false;
  if (npx < 2 || npy < 2) return false;
  /* CubeHalo.F90:103 and fv_mp_mod.F90:583 both require square tiles on a cube
   * (`npx-1 == npy-1`). The reversed seams depend on it: a transposed seam maps
   * my along-seam extent onto the donor's perpendicular extent, so nx != ny
   * would make the mapping non-total. Refuse rather than truncate. */
  if (npx != npy) return false;
  if (!rank_of_tile) return false;
  for (int t = 0; t < cubed_sphere_tiles; ++t)
    if (rank_of_tile[t] < 0) return false;

  TopologyView t{};
  t.contract_version = internal_contract_version;
  t.tile = tile;
  t.npx = npx;
  t.npy = npy;
  t.global_cubed_sphere = 1;
  t.nested = 0;
  t.regional = 0;
  t.periodic_x = 0;
  t.periodic_y = 0;

  for (int e = 0; e < 4; ++e) {
    const TileEdge mine = all_edges[e];
    const Donor d = mosaic[tile - 1][e];
    EdgeNeighbor &nb = t.neighbors[static_cast<size_t>(mine)];
    nb.rank = rank_of_tile[d.tile - 1];
    nb.tile = static_cast<int16_t>(d.tile);
    nb.edge = d.edge;
    nb.orientation = seam_orientation(mine, d.edge);
    /* The vector signs record the DIAGONAL-FREE part of the rotation: the
     * component swap is carried by `orientation != aligned`, and these two are
     * the signs attached to the swapped components. For +90, (u,v) <- (+v,-u),
     * so the x slot takes +1 and the y slot -1; for -90 it is the mirror. On an
     * aligned seam there is no rotation and both are +1.
     *
     * Before this milestone these four fields (edge, orientation,
     * vector_sign_x, vector_sign_y) had zero readers anywhere in the tree --
     * every seam fact was declared and then ignored. seam_vector_transform()
     * is now the single consumer, and cross-checks them. */
    const SeamRotation r = seam_rotation(mine, d.edge);
    const SeamVectorTransform vt = seam_vector_transform(r);
    nb.vector_sign_x = vt.swaps_components() ? (vt.cxy >= 0 ? 1 : -1) : 1;
    nb.vector_sign_y = vt.swaps_components() ? (vt.cyx >= 0 ? 1 : -1) : 1;
  }

  /* A global cubed sphere has a genuine tile at all four corners of every tile,
   * and every corner is a three-tile junction rather than a four-tile one --
   * which is exactly why FV3 carries per-corner flags and why copy_corners /
   * fill_corners exist. The corner OWNER ranks are the ranks of the two tiles
   * meeting my corner; with one rank per tile the owner is whichever of the two
   * adjacent edge neighbours the authoritative precedence selects. That
   * precedence is resolved from source separately; until it is, leave the
   * corner owners unset (-1) rather than guess, and set the corner FLAGS, which
   * are unambiguous: on a cube every tile corner is a panel corner. */
  t.sw_corner = 1;
  t.se_corner = 1;
  t.ne_corner = 1;
  t.nw_corner = 1;

  out = t;
  return true;
}

} // namespace fv3::gpu::halo
