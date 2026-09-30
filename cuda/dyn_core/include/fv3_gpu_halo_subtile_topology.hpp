#pragma once
//===========================================================================
// Prompt 109 Phases 1-4: generalized sub-tile (multi-rank-per-face) cubed-
// sphere halo topology, ADDITIVE to fv3_gpu_halo_topology.hpp.
//
// SCOPE. This header extends the topology layer only, to a synthetic
// layout_x x layout_y ranks-per-face decomposition (target: 2x2, 24 ranks
// total on 6 faces). It does not touch:
//   - fv3_gpu_halo_topology.hpp/.cpp        (legacy one-rank-per-face
//     builder and seam algebra -- reused UNCHANGED below, never modified);
//   - fv3_gpu_halo_mpi.hpp/.cpp             (production HaloTransport /
//     MpiHaloTransport, Phase 7 per the Prompt 108 plan);
//   - cuda/interface/fv3_gpu_dyncore_bridge.F90 (the comm_size==6 production
//     gate stays in force -- see docs/performance/p109_*.md Stage 18).
//
// EVERY FUNCTION HERE IS BUILT ON THE EXISTING, UNCHANGED SEAM ALGEBRA in
// fv3_gpu_halo_topology.hpp (seam_orientation, seam_is_reversed,
// seam_rotation, seam_vector_transform, cubed_sphere_neighbor). No new
// rotation/sign math is introduced. What is new is DONOR IDENTIFICATION:
// which sub-rank, on which face, owns the data a given local edge/corner
// needs, generalizing the legacy "the one other rank owning that whole
// face edge" answer to "the specific sub-rank owning the matching segment
// or diagonal neighbor."
//
// LAYOUT=1x1 REDUCTION (the Stage 16 regression proof, by construction):
// every function below takes layout_x/layout_y as ordinary parameters, and
// at layout_x = layout_y = 1 there is exactly one sub-rank per face
// (sub_i = sub_j = 0 always), every edge is a face boundary
// (subtile_edge_is_face_boundary always true), every corner is a
// true_cube_corner (SubTileCornerType::true_cube_corner for all 4 corners,
// matching the legacy unconditional sw/se/ne/nw_corner = 1), and
// subtile_edge_neighbor's segment_count is always 1 with
// segment_index = donor_segment_index = 0 -- i.e. the legacy single-full-
// edge-donor case, reached by construction rather than by a special case.
//===========================================================================

#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_internal.hpp"
#include <cstdint>

namespace fv3::gpu::halo {

/* ---------------------------------------------------------------------------
 * Stage 2: generalized topology identity.
 *
 * `face` is 1-based, matching the legacy `tile` numbering (1..6) and the
 * mosaic table in fv3_gpu_halo_topology.cpp. `sub_i`/`sub_j` are 0-based
 * coordinates within that face's layout_x x layout_y sub-rank grid
 * (sub_i in [0, layout_x), sub_j in [0, layout_y)). At layout_x = layout_y
 * = 1 the only valid value of both is 0, recovering the legacy one-rank-
 * per-face identity (the `face` field alone).
 * ------------------------------------------------------------------------- */
struct SubTileId {
  int face = 1;
  int sub_i = 0;
  int sub_j = 0;

  FV3_HALO_HD inline constexpr bool operator==(const SubTileId &o) const noexcept {
    return face == o.face && sub_i == o.sub_i && sub_j == o.sub_j;
  }
  FV3_HALO_HD inline constexpr bool operator!=(const SubTileId &o) const noexcept {
    return !(*this == o);
  }
};

/* ---------------------------------------------------------------------------
 * Stage 4: synthetic rank numbering, matching FMS's own pe_start convention
 * (fv_mp_mod.F90: pe_start(n) = pelist(1) + (n-1)*layout(1)*layout(2), i.e.
 * ranks are grouped face-major, and within a face ordered row-major in
 * (sub_i fastest, sub_j slowest) -- the same convention mpp_define_layout
 * uses for a 2-D domain decomposition of one face).
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr int subtile_rank_within_face(int sub_i, int sub_j,
                                                           int layout_x) noexcept {
  return sub_j * layout_x + sub_i;
}

FV3_HALO_HD inline constexpr int32_t subtile_global_rank(const SubTileId &id, int layout_x,
                                                          int layout_y) noexcept {
  return static_cast<int32_t>((id.face - 1) * layout_x * layout_y +
                               subtile_rank_within_face(id.sub_i, id.sub_j, layout_x));
}

FV3_HALO_HD inline constexpr SubTileId subtile_from_global_rank(int32_t global_rank,
                                                                 int layout_x,
                                                                 int layout_y) noexcept {
  const int per_face = layout_x * layout_y;
  const int face0 = static_cast<int>(global_rank) / per_face; /* 0-based */
  const int within = static_cast<int>(global_rank) - face0 * per_face;
  const int sub_j = within / layout_x;
  const int sub_i = within - sub_j * layout_x;
  return SubTileId{face0 + 1, sub_i, sub_j};
}

/* ---------------------------------------------------------------------------
 * Whether a sub-rank's edge `mine` is a physical cube-face boundary (touches
 * the neighbouring cube face, so the legacy mosaic/seam lookup applies) as
 * opposed to an interior seam shared with another sub-rank on the SAME
 * face. At layout_x = layout_y = 1 this is unconditionally true for all
 * four edges, matching the legacy assumption that every edge is a face
 * boundary.
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr bool subtile_edge_is_face_boundary(TileEdge mine, int sub_i,
                                                                 int sub_j, int layout_x,
                                                                 int layout_y) noexcept {
  switch (mine) {
    case TileEdge::west:  return sub_i == 0;
    case TileEdge::east:  return sub_i == layout_x - 1;
    case TileEdge::south: return sub_j == 0;
    default:              return sub_j == layout_y - 1; /* north */
  }
}

/* ---------------------------------------------------------------------------
 * Stage 3: generalized edge-neighbor descriptor for one sub-rank's edge.
 *
 * Generalizes EdgeNeighbor (fv3_gpu_internal.hpp) from "the one other rank
 * that owns this whole face edge" to "the specific sub-rank that owns the
 * matching segment (cross-face) or the adjacent sub-rank (same-face)."
 * `same_face` records which of the two donor-identification rules applied.
 * `segment_index`/`segment_count`/`donor_segment_index` are meaningful only
 * when !same_face; at layout 1x1 they are always 0/1/0.
 * ------------------------------------------------------------------------- */
struct SubTileEdgeNeighbor {
  SubTileId donor{};
  TileEdge donor_edge = TileEdge::west;
  EdgeOrientation orientation = EdgeOrientation::aligned;
  int8_t vector_sign_x = 1, vector_sign_y = 1;
  bool same_face = true;
  int32_t segment_index = 0;
  int32_t segment_count = 1;
  int32_t donor_segment_index = 0;
};

/* Shared by both branches below: the vector-sign derivation is copied
 * verbatim from make_cubed_sphere_topology (fv3_gpu_halo_topology.cpp) --
 * the DIAGONAL-FREE part of the rotation, keyed on whether the transform
 * swaps components at all. No new sign algebra. */
FV3_HALO_HD inline constexpr void subtile_vector_signs(SeamRotation r, int8_t &sign_x,
                                                        int8_t &sign_y) noexcept {
  const SeamVectorTransform vt = seam_vector_transform(r);
  sign_x = vt.swaps_components() ? (vt.cxy >= 0 ? int8_t(1) : int8_t(-1)) : int8_t(1);
  sign_y = vt.swaps_components() ? (vt.cyx >= 0 ? int8_t(1) : int8_t(-1)) : int8_t(1);
}

/* Stage 5 (same-face cardinal) + Stage 6 (cross-face partial-segment donor).
 *
 * SAME-FACE: step exactly one sub-rank in the cardinal direction of `mine`,
 * on the SAME face. Calling the existing seam_orientation/seam_rotation
 * with `mine` and its OPPOSITE edge yields aligned/zero automatically
 * (edge_is_x_normal(mine) == edge_is_x_normal(opposite(mine)) always
 * holds), so identity orientation falls out of the unchanged algebra with
 * no new rotation math -- only the new donor SubTileId is new.
 *
 * CROSS-FACE: face-level donor comes from the unchanged
 * cubed_sphere_neighbor() mosaic lookup. My position along the
 * perpendicular layout axis (segment index k, 0-based) maps to the donor's
 * segment index k (aligned seam) or N-1-k (reversed seam), N being the
 * segment count along that axis -- exactly the seam_along_index rule
 * already used for cell indices, now applied one level up to sub-rank
 * selection. PHASES 1-4 ASSUME A UNIFORM layout_x/layout_y ACROSS ALL 6
 * FACES (stated in the Prompt 109 target scope), so the donor's own
 * sub-rank grid uses the same layout_x/layout_y as mine.
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline SubTileEdgeNeighbor subtile_edge_neighbor(SubTileId me, TileEdge mine,
                                                              int layout_x,
                                                              int layout_y) noexcept {
  SubTileEdgeNeighbor out{};
  const bool face_boundary =
      subtile_edge_is_face_boundary(mine, me.sub_i, me.sub_j, layout_x, layout_y);

  if (!face_boundary) {
    out.same_face = true;
    out.donor = me;
    switch (mine) {
      case TileEdge::west:  out.donor.sub_i -= 1; out.donor_edge = TileEdge::east;  break;
      case TileEdge::east:  out.donor.sub_i += 1; out.donor_edge = TileEdge::west;  break;
      case TileEdge::south: out.donor.sub_j -= 1; out.donor_edge = TileEdge::north; break;
      default:              out.donor.sub_j += 1; out.donor_edge = TileEdge::south; break;
    }
    out.orientation = seam_orientation(mine, out.donor_edge);
    subtile_vector_signs(seam_rotation(mine, out.donor_edge), out.vector_sign_x,
                         out.vector_sign_y);
    out.segment_index = 0;
    out.segment_count = 1;
    out.donor_segment_index = 0;
    return out;
  }

  out.same_face = false;
  int donor_face = me.face;
  TileEdge donor_edge = mine;
  cubed_sphere_neighbor(me.face, mine, donor_face, donor_edge);
  out.donor_edge = donor_edge;
  out.orientation = seam_orientation(mine, donor_edge);
  const bool reversed = seam_is_reversed(mine, donor_edge);
  subtile_vector_signs(seam_rotation(mine, donor_edge), out.vector_sign_x, out.vector_sign_y);

  int32_t k, n;
  if (edge_is_x_normal(mine)) { k = me.sub_j; n = layout_y; }
  else                        { k = me.sub_i; n = layout_x; }
  out.segment_index = k;
  out.segment_count = n;
  out.donor_segment_index = reversed ? (n - 1 - k) : k;

  const int32_t dseg = out.donor_segment_index;
  int donor_sub_i, donor_sub_j;
  if (edge_is_x_normal(donor_edge)) {
    donor_sub_j = dseg;
    donor_sub_i = (donor_edge == TileEdge::west) ? 0 : layout_x - 1;
  } else {
    donor_sub_i = dseg;
    donor_sub_j = (donor_edge == TileEdge::south) ? 0 : layout_y - 1;
  }
  out.donor = SubTileId{donor_face, donor_sub_i, donor_sub_j};
  return out;
}

/* ---------------------------------------------------------------------------
 * Stage 8: corner taxonomy.
 *
 * A corner is identified by the pair of cardinal edges that meet there
 * (0=SW, 1=SE, 2=NW, 3=NE). Classification counts how many of that pair are
 * physical face boundaries for THIS sub-rank:
 *   0 face-boundary edges -> INTRA_FACE_DIAGONAL (all 4 corner-adjacent
 *       sub-ranks are on the same face)
 *   1 face-boundary edge  -> MIXED (one cardinal neighbor is same-face, the
 *       diagonal corner partner is reached by crossing the cube-face seam)
 *   2 face-boundary edges -> TRUE_CUBE_CORNER (this sub-rank sits at a
 *       physical cube-panel corner; preserve existing local corner-fill
 *       algebra unchanged, no new donor)
 *
 * At layout_x = layout_y = 1 every edge is a face boundary, so every corner
 * is TRUE_CUBE_CORNER for all 4 corners -- exactly the legacy unconditional
 * sw_corner = se_corner = ne_corner = nw_corner = 1.
 * ------------------------------------------------------------------------- */
enum class SubTileCornerType : uint8_t {
  intra_face_diagonal = 0,
  mixed = 1,
  true_cube_corner = 2,
};

FV3_HALO_HD inline constexpr void subtile_corner_edges(int corner, TileEdge &e1,
                                                        TileEdge &e2) noexcept {
  switch (corner) {
    case 0: e1 = TileEdge::west; e2 = TileEdge::south; break; /* SW */
    case 1: e1 = TileEdge::east; e2 = TileEdge::south; break; /* SE */
    case 2: e1 = TileEdge::west; e2 = TileEdge::north; break; /* NW */
    default: e1 = TileEdge::east; e2 = TileEdge::north; break; /* NE */
  }
}

FV3_HALO_HD inline SubTileCornerType classify_subtile_corner(SubTileId me, int corner,
                                                              int layout_x,
                                                              int layout_y) noexcept {
  TileEdge e1, e2;
  subtile_corner_edges(corner, e1, e2);
  const int face_boundary_count =
      (subtile_edge_is_face_boundary(e1, me.sub_i, me.sub_j, layout_x, layout_y) ? 1 : 0) +
      (subtile_edge_is_face_boundary(e2, me.sub_i, me.sub_j, layout_x, layout_y) ? 1 : 0);
  switch (face_boundary_count) {
    case 0:  return SubTileCornerType::intra_face_diagonal;
    case 1:  return SubTileCornerType::mixed;
    default: return SubTileCornerType::true_cube_corner;
  }
}

/* ---------------------------------------------------------------------------
 * Stage 9 (diagonal donor) + Stage 10 (mixed-corner precedence) + Stage 11
 * (true cube corner).
 *
 * CANONICAL MIXED-CORNER PRECEDENCE RULE (Prompt 108 risk #2, resolved
 * here): resolve the INTRA-FACE cardinal step FIRST, then apply the
 * CROSS-FACE segment-donor lookup from the resulting sub-rank. This is a
 * single well-defined composition of subtile_edge_neighbor's two existing
 * branches -- no new algebra, and no dependence on any sequential halo-fill
 * order.
 *
 * INTRA_FACE_DIAGONAL: composing the two same-face cardinal steps (order is
 * commutative on this orthogonal sub-rank grid) reaches the diagonal
 * neighbor directly; both steps are same-face, so the composed transform is
 * identity (`xf_out` returned as identity).
 *
 * TRUE_CUBE_CORNER: both corner edges are face boundaries. Preserve the
 * existing LOCAL, zero-MPI corner-fill algebra (fv3_cgrid_corners.hpp)
 * unchanged: returns `me` itself (no cross-rank donor at all), because at a
 * true cube-panel corner the fill kernel already produces the corner value
 * from this rank's own owned data plus its two edge halos, in the same way
 * it does today for every corner in the legacy layout=1x1 case.
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline SubTileId subtile_corner_donor(SubTileId me, int corner, int layout_x,
                                                   int layout_y,
                                                   SeamVectorTransform &xf_out) noexcept {
  TileEdge e1, e2;
  subtile_corner_edges(corner, e1, e2);
  const SubTileCornerType type = classify_subtile_corner(me, corner, layout_x, layout_y);
  xf_out = SeamVectorTransform{};

  if (type == SubTileCornerType::intra_face_diagonal) {
    const SubTileEdgeNeighbor n1 = subtile_edge_neighbor(me, e1, layout_x, layout_y);
    const SubTileEdgeNeighbor n2 = subtile_edge_neighbor(n1.donor, e2, layout_x, layout_y);
    return n2.donor;
  }

  if (type == SubTileCornerType::mixed) {
    const bool e1_is_boundary =
        subtile_edge_is_face_boundary(e1, me.sub_i, me.sub_j, layout_x, layout_y);
    const TileEdge intra_edge = e1_is_boundary ? e2 : e1;
    const TileEdge cross_edge = e1_is_boundary ? e1 : e2;
    const SubTileEdgeNeighbor n1 = subtile_edge_neighbor(me, intra_edge, layout_x, layout_y);
    const SubTileEdgeNeighbor n2 = subtile_edge_neighbor(n1.donor, cross_edge, layout_x, layout_y);
    xf_out = seam_vector_transform(seam_rotation(cross_edge, n2.donor_edge));
    return n2.donor;
  }

  return me; /* TRUE_CUBE_CORNER: local fill, no cross-rank donor. */
}

/* ---------------------------------------------------------------------------
 * Stage 13 support: reciprocal corner lookup.
 *
 * Given that `me`'s corner `corner` resolves to donor rank `donor` (via
 * subtile_corner_donor), this answers "which of `donor`'s OWN 4 corner
 * indices is the relationship that points back to `me`?" -- needed so a
 * receiver can compute, in advance, the exact tag its sender will use
 * (subtile_corner_tag(that index)), without guessing or requiring a
 * wildcard receive. Pure exhaustive search over the donor's 4 corners
 * (identical pattern to the reciprocity proof in
 * test_halo_subtile_topology.cpp T8/T9); returns -1 if none matches (should
 * not happen for any corner with donor != me, by the same reciprocity the
 * unit tests establish).
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline int subtile_corner_reciprocal_index(SubTileId me, SubTileId donor,
                                                        int layout_x, int layout_y) noexcept {
  for (int c = 0; c < 4; ++c) {
    SeamVectorTransform xf{};
    const SubTileId back = subtile_corner_donor(donor, c, layout_x, layout_y, xf);
    if (back == me) return c;
  }
  return -1;
}

/* ---------------------------------------------------------------------------
 * Stage 13: deterministic, collision-free MPI tag scheme.
 *
 * MPI tags are scoped per ordered (source, dest) rank pair, so reuse of the
 * SAME tag values across DIFFERENT rank pairs is safe by construction; what
 * must never collide is two distinct messages between the SAME ordered pair
 * (which cannot happen here: a sub-rank has at most one cardinal neighbor
 * per direction and at most one corner donor per corner, each a distinct
 * tag below).
 * ------------------------------------------------------------------------- */
inline constexpr int32_t subtile_tag_cardinal_west = 100;
inline constexpr int32_t subtile_tag_cardinal_east = 101;
inline constexpr int32_t subtile_tag_cardinal_south = 102;
inline constexpr int32_t subtile_tag_cardinal_north = 103;
inline constexpr int32_t subtile_tag_corner_sw = 110;
inline constexpr int32_t subtile_tag_corner_se = 111;
inline constexpr int32_t subtile_tag_corner_nw = 112;
inline constexpr int32_t subtile_tag_corner_ne = 113;

FV3_HALO_HD inline constexpr int32_t subtile_edge_tag(TileEdge e) noexcept {
  switch (e) {
    case TileEdge::west:  return subtile_tag_cardinal_west;
    case TileEdge::east:  return subtile_tag_cardinal_east;
    case TileEdge::south: return subtile_tag_cardinal_south;
    default:              return subtile_tag_cardinal_north;
  }
}

FV3_HALO_HD inline constexpr int32_t subtile_corner_tag(int corner) noexcept {
  switch (corner) {
    case 0: return subtile_tag_corner_sw;
    case 1: return subtile_tag_corner_se;
    case 2: return subtile_tag_corner_nw;
    default: return subtile_tag_corner_ne;
  }
}

} // namespace fv3::gpu::halo
