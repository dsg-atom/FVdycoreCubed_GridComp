//===========================================================================
// Prompt 109 Phases 1-4: host-only unit tests for the generalized sub-tile
// halo topology (fv3_gpu_halo_subtile_topology.hpp).
//
// Style follows test_halo_topology.cpp: check()/check_eq(), a global
// `failures` counter, one Tn per requirement, printf-based reporting, a
// final pass/fail summary with explicit "does NOT cover" caveats.
//
// WHAT THIS VALIDATES: pure topology/donor-identification logic only (no
// CUDA, no MPI, no numerics). Layout=1x1 tests are the Stage 16 regression
// oracle -- every one of them compares the new generalized functions
// against the UNCHANGED legacy fv3_gpu_halo_topology.{hpp,cpp} outputs.
// Layout=2x2 tests exercise Stages 4-8, 10-11 for the target synthetic
// 24-rank topology. NOT covered here: MPI message routing (separate
// integration test), CUDA small-grid halo application (separate test),
// sanitizer.
//
// Prompt 111 (Phase 5, edges-only slice) ADDS: T12/T13, validating the new
// additive fv3_gpu_halo_subtile_view.hpp builder
// (make_subtile_cubed_sphere_topology) -- Stage 2 (exhaustive layout=1x1
// TopologyView equivalence vs the UNCHANGED legacy builder, edges+corners)
// and Stage 3 (layout=2x2 TopologyView correctness vs the Phase 1-4 oracle
// functions directly, not by calling the new builder a second time).
//===========================================================================
#include "fv3_gpu_halo_subtile_topology.hpp"
#include "fv3_gpu_halo_subtile_view.hpp"

#include <cstdio>
#include <cstdlib>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

static int failures = 0;

static void check(bool cond, const char *what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

template <typename A, typename B>
static void check_eq(const A &got, const B &want, const char *what) {
  if (!(got == want)) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

static int32_t rank_of_face[cubed_sphere_tiles] = {0, 1, 2, 3, 4, 5};

// ---------------------------------------------------------------------------
// T1: rank <-> SubTileId bijection, both layout=1x1 (6 ranks) and layout=2x2
// (24 ranks). Round-trip identity for every rank in range.
// ---------------------------------------------------------------------------
static void T1_rank_bijection() {
  for (int r = 0; r < 6; ++r) {
    const SubTileId id = subtile_from_global_rank(r, 1, 1);
    check(id.sub_i == 0 && id.sub_j == 0, "T1: layout1x1 sub_i/sub_j must be 0");
    check_eq(subtile_global_rank(id, 1, 1), r, "T1: layout1x1 round trip");
  }
  for (int r = 0; r < 24; ++r) {
    const SubTileId id = subtile_from_global_rank(r, 2, 2);
    check(id.face >= 1 && id.face <= 6, "T1: layout2x2 face range");
    check(id.sub_i >= 0 && id.sub_i < 2, "T1: layout2x2 sub_i range");
    check(id.sub_j >= 0 && id.sub_j < 2, "T1: layout2x2 sub_j range");
    check_eq(subtile_global_rank(id, 2, 2), r, "T1: layout2x2 round trip");
  }
  // face-major ordering: ranks 0-3 all on face 1, 4-7 all on face 2, etc.
  for (int r = 0; r < 24; ++r) {
    const SubTileId id = subtile_from_global_rank(r, 2, 2);
    check_eq(id.face, r / 4 + 1, "T1: face-major synthetic numbering");
  }
}

// ---------------------------------------------------------------------------
// T2: layout=1x1 face-boundary regression -- every edge on every face is a
// face boundary (legacy assumption).
// ---------------------------------------------------------------------------
static void T2_layout1x1_all_edges_are_face_boundary() {
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (int face = 1; face <= 6; ++face)
    for (TileEdge e : edges)
      check(subtile_edge_is_face_boundary(e, 0, 0, 1, 1),
            "T2: layout1x1 every edge must be a face boundary");
}

// ---------------------------------------------------------------------------
// T3: layout=1x1 edge-neighbor regression against the UNCHANGED legacy
// builder (make_cubed_sphere_topology). Donor face/edge, orientation and
// vector signs must match exactly; segment fields must be the degenerate
// single-donor values.
// ---------------------------------------------------------------------------
static void T3_layout1x1_edge_neighbor_matches_legacy() {
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (int face = 1; face <= 6; ++face) {
    TopologyView legacy{};
    check(make_cubed_sphere_topology(face, 8, 8, rank_of_face, legacy),
          "T3: legacy builder must succeed");
    for (TileEdge e : edges) {
      const SubTileEdgeNeighbor sn =
          subtile_edge_neighbor(SubTileId{face, 0, 0}, e, 1, 1);
      const EdgeNeighbor &ln = legacy.neighbors[static_cast<size_t>(e)];

      check(!sn.same_face, "T3: layout1x1 every edge is cross-face");
      check_eq(sn.donor.face, static_cast<int>(ln.tile), "T3: donor face vs legacy tile");
      check_eq(static_cast<int>(sn.donor_edge), static_cast<int>(ln.edge),
               "T3: donor edge vs legacy edge");
      check_eq(static_cast<int>(sn.orientation), static_cast<int>(ln.orientation),
               "T3: orientation vs legacy");
      check_eq(sn.vector_sign_x, ln.vector_sign_x, "T3: vector_sign_x vs legacy");
      check_eq(sn.vector_sign_y, ln.vector_sign_y, "T3: vector_sign_y vs legacy");
      check_eq(sn.segment_count, 1, "T3: layout1x1 segment_count degenerate");
      check_eq(sn.segment_index, 0, "T3: layout1x1 segment_index degenerate");
      check_eq(sn.donor_segment_index, 0, "T3: layout1x1 donor_segment_index degenerate");
      check_eq(sn.donor.sub_i, 0, "T3: layout1x1 donor sub_i degenerate");
      check_eq(sn.donor.sub_j, 0, "T3: layout1x1 donor sub_j degenerate");
    }
  }
}

// ---------------------------------------------------------------------------
// T4: layout=1x1 corner regression -- every corner of every face classifies
// as TRUE_CUBE_CORNER, matching legacy's unconditional
// sw_corner=se_corner=ne_corner=nw_corner=1.
// ---------------------------------------------------------------------------
static void T4_layout1x1_all_corners_true_cube() {
  for (int face = 1; face <= 6; ++face)
    for (int corner = 0; corner < 4; ++corner)
      check(classify_subtile_corner(SubTileId{face, 0, 0}, corner, 1, 1) ==
                SubTileCornerType::true_cube_corner,
            "T4: layout1x1 every corner must be TRUE_CUBE_CORNER");
}

// ---------------------------------------------------------------------------
// T5: layout=2x2 same-face cardinal neighbors. Every one of the 24
// synthetic ranks' INTERIOR (non-face-boundary) edges must point to the
// adjacent sub-rank on the SAME face, with identity orientation (no
// rotation) and reciprocity (the neighbor's opposite edge points back).
// ---------------------------------------------------------------------------
static void T5_layout2x2_same_face_cardinal() {
  const int LX = 2, LY = 2;
  int same_face_edges_seen = 0;
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south,
                                    TileEdge::north};
        for (TileEdge e : edges) {
          if (subtile_edge_is_face_boundary(e, si, sj, LX, LY)) continue;
          ++same_face_edges_seen;
          const SubTileEdgeNeighbor n = subtile_edge_neighbor(me, e, LX, LY);
          check(n.same_face, "T5: interior edge must resolve same-face");
          check_eq(n.donor.face, face, "T5: same-face donor must stay on my face");
          check(n.orientation == EdgeOrientation::aligned,
                "T5: same-face cardinal must be identity orientation");
          check_eq(n.vector_sign_x, int8_t(1), "T5: same-face vector_sign_x must be +1");
          check_eq(n.vector_sign_y, int8_t(1), "T5: same-face vector_sign_y must be +1");

          // Reciprocity: donor's opposite edge must point back to me.
          const SubTileEdgeNeighbor back = subtile_edge_neighbor(n.donor, n.donor_edge, LX, LY);
          check_eq(back.donor, me, "T5: same-face reciprocity");
        }
      }
  // For a 2x2 face layout every sub-rank has exactly 2 interior cardinal
  // edges (one of {W,E}, one of {S,N}); 4 sub-ranks/face x 6 faces x 2.
  check_eq(same_face_edges_seen, 4 * 6 * 2, "T5: interior edge count for layout2x2");
}

// ---------------------------------------------------------------------------
// T6: layout=2x2 cross-face partial-segment donors. Every FACE-BOUNDARY
// edge must resolve cross-face (same_face=false), with segment_count=2
// (layout_y or layout_x, whichever is perpendicular), and the donor's
// reciprocal lookup must map back to me (involution proof, mirroring
// test_halo_topology.cpp's T7).
// ---------------------------------------------------------------------------
static void T6_layout2x2_cross_face_segments() {
  const int LX = 2, LY = 2;
  int cross_face_edges_seen = 0;
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south,
                                    TileEdge::north};
        for (TileEdge e : edges) {
          if (!subtile_edge_is_face_boundary(e, si, sj, LX, LY)) continue;
          ++cross_face_edges_seen;
          const SubTileEdgeNeighbor n = subtile_edge_neighbor(me, e, LX, LY);
          check(!n.same_face, "T6: face-boundary edge must resolve cross-face");
          check(n.donor.face != face, "T6: cross-face donor must be on a different face");
          check_eq(n.segment_count, 2, "T6: layout2x2 segment_count must be 2");
          check(n.segment_index == 0 || n.segment_index == 1,
                "T6: segment_index must be 0 or 1");
          check(n.donor_segment_index == 0 || n.donor_segment_index == 1,
                "T6: donor_segment_index must be 0 or 1");

          // Involution: looking up my edge from the donor's side, at the
          // donor's segment index, must map back to me.
          const SubTileEdgeNeighbor back =
              subtile_edge_neighbor(n.donor, n.donor_edge, LX, LY);
          check_eq(back.donor, me, "T6: cross-face segment involution");
          check_eq(back.donor_segment_index, (edge_is_x_normal(e) ? sj : si),
                   "T6: cross-face involution recovers my own segment index");
        }
      }
  // Every sub-rank has exactly 2 face-boundary edges in a 2x2 layout.
  check_eq(cross_face_edges_seen, 4 * 6 * 2, "T6: face-boundary edge count for layout2x2");
}

// ---------------------------------------------------------------------------
// T7: layout=2x2 corner taxonomy. Per-subtile corner-type counts must be
// exactly {1 true_cube_corner, 2 mixed, 1 intra_face_diagonal} (derived by
// hand for a 2x2 face layout: each subtile sits at exactly one physical
// cube corner, is diagonally adjacent to exactly one same-face subtile, and
// has two mixed corners reaching a neighboring face). Face-wide and
// cube-wide totals must match the physical cube (8 vertices x 3 faces each
// = 24 true_cube corner-instances cube-wide).
// ---------------------------------------------------------------------------
static void T7_layout2x2_corner_taxonomy() {
  const int LX = 2, LY = 2;
  int true_cube_total = 0, mixed_total = 0, diagonal_total = 0;
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        int true_cube = 0, mixed = 0, diagonal = 0;
        for (int corner = 0; corner < 4; ++corner) {
          switch (classify_subtile_corner(me, corner, LX, LY)) {
            case SubTileCornerType::true_cube_corner: ++true_cube; break;
            case SubTileCornerType::mixed:             ++mixed;     break;
            case SubTileCornerType::intra_face_diagonal: ++diagonal; break;
          }
        }
        check_eq(true_cube, 1, "T7: exactly one TRUE_CUBE_CORNER per sub-rank at 2x2");
        check_eq(mixed, 2, "T7: exactly two MIXED corners per sub-rank at 2x2");
        check_eq(diagonal, 1, "T7: exactly one INTRA_FACE_DIAGONAL corner per sub-rank at 2x2");
        true_cube_total += true_cube;
        mixed_total += mixed;
        diagonal_total += diagonal;
      }
  check_eq(true_cube_total, 24, "T7: cube-wide TRUE_CUBE_CORNER instances (8 vertices x 3 faces)");
  check_eq(mixed_total, 48, "T7: cube-wide MIXED corner instances");
  check_eq(diagonal_total, 24, "T7: cube-wide INTRA_FACE_DIAGONAL corner instances");
}

// ---------------------------------------------------------------------------
// T8: layout=2x2 diagonal (intra-face) corner donor reciprocity. The 4
// diagonal-typed corners on a face must pair up into two mutual 2-cycles:
// (0,0)<->(1,1) and (1,0)<->(0,1) -- the single interior grid point where
// all 4 sub-ranks meet, seen from each sub-rank's own diagonal corner.
// Vector transform for a diagonal donor must be identity (two same-face
// steps compose to no rotation).
// ---------------------------------------------------------------------------
static void T8_layout2x2_diagonal_corner_reciprocity() {
  const int LX = 2, LY = 2;
  const int pairs[4][2] = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
  for (int face = 1; face <= 6; ++face) {
    for (int idx = 0; idx < 4; ++idx) {
      const SubTileId me{face, pairs[idx][0], pairs[idx][1]};
      for (int corner = 0; corner < 4; ++corner) {
        if (classify_subtile_corner(me, corner, LX, LY) !=
            SubTileCornerType::intra_face_diagonal)
          continue;
        SeamVectorTransform xf{};
        const SubTileId donor = subtile_corner_donor(me, corner, LX, LY, xf);
        check_eq(donor.face, face, "T8: diagonal donor stays on my face");
        check(donor != me, "T8: diagonal donor must differ from me");
        check(!xf.swaps_components(), "T8: diagonal corner transform must be identity");
        check_eq(xf.cxx, int8_t(1), "T8: diagonal transform cxx");
        check_eq(xf.cyy, int8_t(1), "T8: diagonal transform cyy");

        // Reciprocity: the donor's own diagonal corner must point back to me.
        bool found_back = false;
        for (int c2 = 0; c2 < 4; ++c2) {
          if (classify_subtile_corner(donor, c2, LX, LY) !=
              SubTileCornerType::intra_face_diagonal)
            continue;
          SeamVectorTransform xf2{};
          const SubTileId back = subtile_corner_donor(donor, c2, LX, LY, xf2);
          if (back == me) { found_back = true; break; }
        }
        check(found_back, "T8: diagonal corner reciprocity");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// T9: layout=2x2 mixed-corner canonical precedence. A mixed corner's donor
// must equal composing the intra-face cardinal step (Stage 5) and then the
// cross-face segment lookup (Stage 6) from the resulting sub-rank -- i.e.
// exactly the composition subtile_corner_donor is documented to perform,
// checked here independently by re-deriving it from the two primitives
// rather than by calling subtile_corner_donor a second time.
// ---------------------------------------------------------------------------
static void T9_layout2x2_mixed_corner_precedence() {
  const int LX = 2, LY = 2;
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        for (int corner = 0; corner < 4; ++corner) {
          if (classify_subtile_corner(me, corner, LX, LY) != SubTileCornerType::mixed)
            continue;
          TileEdge e1, e2;
          subtile_corner_edges(corner, e1, e2);
          const bool e1_boundary = subtile_edge_is_face_boundary(e1, si, sj, LX, LY);
          const TileEdge intra_edge = e1_boundary ? e2 : e1;
          const TileEdge cross_edge = e1_boundary ? e1 : e2;

          const SubTileEdgeNeighbor n1 = subtile_edge_neighbor(me, intra_edge, LX, LY);
          check(n1.same_face, "T9: mixed corner's first step must be same-face");
          const SubTileEdgeNeighbor n2 = subtile_edge_neighbor(n1.donor, cross_edge, LX, LY);
          check(!n2.same_face, "T9: mixed corner's second step must be cross-face");

          SeamVectorTransform xf{};
          const SubTileId donor = subtile_corner_donor(me, corner, LX, LY, xf);
          check_eq(donor, n2.donor, "T9: mixed corner precedence matches manual composition");
        }
      }
}

// ---------------------------------------------------------------------------
// T10: layout=2x2 true-cube-corner gating. Exactly the physically-exterior
// sub-rank at each face corner is classified TRUE_CUBE_CORNER; its donor is
// itself (local fill, no cross-rank donor), matching Stage 11's requirement
// that only the appropriate outer sub-rank executes cube-corner behavior.
// ---------------------------------------------------------------------------
static void T10_layout2x2_true_cube_corner_gating() {
  const int LX = 2, LY = 2;
  // The 4 physically-exterior sub-ranks of a face, and which corner is
  // theirs: (0,0)->SW, (1,0)->SE, (0,1)->NW, (1,1)->NE.
  struct Case { int si, sj, corner; };
  const Case cases[4] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 2}, {1, 1, 3}};
  for (int face = 1; face <= 6; ++face) {
    for (const Case &c : cases) {
      const SubTileId me{face, c.si, c.sj};
      check(classify_subtile_corner(me, c.corner, LX, LY) ==
                SubTileCornerType::true_cube_corner,
            "T10: expected physically-exterior corner must be TRUE_CUBE_CORNER");
      SeamVectorTransform xf{};
      const SubTileId donor = subtile_corner_donor(me, c.corner, LX, LY, xf);
      check_eq(donor, me, "T10: true cube corner donor must be self (local fill)");
    }
  }
}

// ---------------------------------------------------------------------------
// T11: layout=2x2 reciprocal corner index (Stage 13 support used by the
// 24-rank MPI routing test to compute matching tags in advance). For every
// corner whose donor differs from me, the reciprocal index must exist and
// must map back to me.
// ---------------------------------------------------------------------------
static void T11_layout2x2_corner_reciprocal_index() {
  const int LX = 2, LY = 2;
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        for (int corner = 0; corner < 4; ++corner) {
          SeamVectorTransform xf{};
          const SubTileId donor = subtile_corner_donor(me, corner, LX, LY, xf);
          if (donor == me) continue; // TRUE_CUBE_CORNER: no reciprocal needed.
          const int idx = subtile_corner_reciprocal_index(me, donor, LX, LY);
          check(idx >= 0, "T11: reciprocal corner index must be found");
          if (idx < 0) continue;
          SeamVectorTransform xf2{};
          const SubTileId back = subtile_corner_donor(donor, idx, LX, LY, xf2);
          check_eq(back, me, "T11: reciprocal corner index must map back to me");
        }
      }
}

// ---------------------------------------------------------------------------
// T12 (Prompt 111 Stage 2): layout=1x1 EXHAUSTIVE TopologyView equivalence
// between the new additive builder (make_subtile_cubed_sphere_topology) and
// the UNCHANGED legacy builder (make_cubed_sphere_topology), for all 6 faces.
// Every scalar field, all 4 edge neighbors (rank/tile/edge/orientation/
// vector signs), all 4 corner flags, and corner_owner_rank must match
// exactly -- not just the edge-neighbor subset T3 already checks.
// ---------------------------------------------------------------------------
static void T12_layout1x1_full_topologyview_equivalence_vs_legacy() {
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (int face = 1; face <= 6; ++face) {
    TopologyView legacy{};
    check(make_cubed_sphere_topology(face, 8, 8, rank_of_face, legacy),
          "T12: legacy builder must succeed");

    TopologyView built{};
    check(make_subtile_cubed_sphere_topology(SubTileId{face, 0, 0}, 1, 1, 8, 8, built),
          "T12: new builder must succeed");

    check_eq(built.contract_version, legacy.contract_version, "T12: contract_version");
    check_eq(built.tile, legacy.tile, "T12: tile");
    check_eq(built.npx, legacy.npx, "T12: npx");
    check_eq(built.npy, legacy.npy, "T12: npy");
    check_eq(built.global_cubed_sphere, legacy.global_cubed_sphere, "T12: global_cubed_sphere");
    check_eq(built.nested, legacy.nested, "T12: nested");
    check_eq(built.regional, legacy.regional, "T12: regional");
    check_eq(built.periodic_x, legacy.periodic_x, "T12: periodic_x");
    check_eq(built.periodic_y, legacy.periodic_y, "T12: periodic_y");

    for (TileEdge e : edges) {
      const EdgeNeighbor &bn = built.neighbors[static_cast<size_t>(e)];
      const EdgeNeighbor &ln = legacy.neighbors[static_cast<size_t>(e)];
      check_eq(bn.rank, ln.rank, "T12: neighbor.rank");
      check_eq(bn.tile, ln.tile, "T12: neighbor.tile");
      check_eq(static_cast<int>(bn.edge), static_cast<int>(ln.edge), "T12: neighbor.edge");
      check_eq(static_cast<int>(bn.orientation), static_cast<int>(ln.orientation),
               "T12: neighbor.orientation");
      check_eq(bn.vector_sign_x, ln.vector_sign_x, "T12: neighbor.vector_sign_x");
      check_eq(bn.vector_sign_y, ln.vector_sign_y, "T12: neighbor.vector_sign_y");
    }

    check_eq(built.sw_corner, legacy.sw_corner, "T12: sw_corner");
    check_eq(built.se_corner, legacy.se_corner, "T12: se_corner");
    check_eq(built.ne_corner, legacy.ne_corner, "T12: ne_corner");
    check_eq(built.nw_corner, legacy.nw_corner, "T12: nw_corner");
    for (int c = 0; c < 4; ++c)
      check_eq(built.corner_owner_rank[c], legacy.corner_owner_rank[c],
               "T12: corner_owner_rank (both -1 sentinel at layout1x1)");
  }
}

// ---------------------------------------------------------------------------
// T13 (Prompt 111 Stage 3): layout=2x2 TopologyView correctness for all 24
// synthetic ranks, checked against the Phase 1-4 oracle functions
// (subtile_edge_neighbor/subtile_global_rank/classify_subtile_corner/
// subtile_corner_donor) DIRECTLY -- not by calling the new builder a second
// time -- following the T9 anti-circularity precedent.
// ---------------------------------------------------------------------------
static void T13_layout2x2_topologyview_matches_oracle() {
  const int LX = 2, LY = 2;
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (int face = 1; face <= 6; ++face)
    for (int si = 0; si < LX; ++si)
      for (int sj = 0; sj < LY; ++sj) {
        const SubTileId me{face, si, sj};
        TopologyView built{};
        check(make_subtile_cubed_sphere_topology(me, LX, LY, 8, 8, built),
              "T13: new builder must succeed at layout2x2");
        check_eq(built.tile, face, "T13: TopologyView.tile is my own face id");

        for (TileEdge e : edges) {
          const SubTileEdgeNeighbor sn = subtile_edge_neighbor(me, e, LX, LY);
          const int32_t want_rank = subtile_global_rank(sn.donor, LX, LY);
          const EdgeNeighbor &bn = built.neighbors[static_cast<size_t>(e)];
          check_eq(bn.rank, want_rank, "T13: neighbor.rank vs oracle global rank");
          check_eq(static_cast<int32_t>(bn.tile), sn.donor.face,
                   "T13: neighbor.tile is donor's physical cube face (Prompt 112 correction)");
          check_eq(static_cast<int>(bn.edge), static_cast<int>(sn.donor_edge),
                   "T13: neighbor.edge vs oracle donor_edge");
          check_eq(static_cast<int>(bn.orientation), static_cast<int>(sn.orientation),
                   "T13: neighbor.orientation vs oracle");
          check_eq(bn.vector_sign_x, sn.vector_sign_x, "T13: neighbor.vector_sign_x vs oracle");
          check_eq(bn.vector_sign_y, sn.vector_sign_y, "T13: neighbor.vector_sign_y vs oracle");
        }

        // Corner index convention (subtile_corner_edges): 0=SW,1=SE,2=NW,3=NE.
        // TopologyView field declaration order is sw,se,ne,nw -- deliberately
        // different -- so corner index 2 (NW) must land in nw_corner (index 3
        // in field order) and index 3 (NE) must land in ne_corner (index 2 in
        // field order). This checks that mapping was not swapped.
        const uint8_t want_flag[4] = {
            built.sw_corner, built.se_corner,
            built.nw_corner /* index 2 = NW */, built.ne_corner /* index 3 = NE */};
        for (int c = 0; c < 4; ++c) {
          SeamVectorTransform xf{};
          const SubTileId donor = subtile_corner_donor(me, c, LX, LY, xf);
          const bool is_true_cube =
              classify_subtile_corner(me, c, LX, LY) == SubTileCornerType::true_cube_corner;
          check_eq(static_cast<int>(want_flag[c]), is_true_cube ? 1 : 0,
                   "T13: corner flag vs oracle classification (index/field-order mapping)");
          const int8_t want_owner =
              is_true_cube ? int8_t(-1)
                           : static_cast<int8_t>(subtile_global_rank(donor, LX, LY));
          // corner_owner_rank[] uses the SAME 0=SW,1=SE,2=NW,3=NE index order
          // as subtile_corner_edges (unlike the sw/se/ne/nw named fields).
          check_eq(built.corner_owner_rank[c], want_owner,
                   "T13: corner_owner_rank vs oracle donor global rank");
        }
      }
}

int main() {
  T1_rank_bijection();
  T2_layout1x1_all_edges_are_face_boundary();
  T3_layout1x1_edge_neighbor_matches_legacy();
  T4_layout1x1_all_corners_true_cube();
  T5_layout2x2_same_face_cardinal();
  T6_layout2x2_cross_face_segments();
  T7_layout2x2_corner_taxonomy();
  T8_layout2x2_diagonal_corner_reciprocity();
  T9_layout2x2_mixed_corner_precedence();
  T10_layout2x2_true_cube_corner_gating();
  T11_layout2x2_corner_reciprocal_index();
  T12_layout1x1_full_topologyview_equivalence_vs_legacy();
  T13_layout2x2_topologyview_matches_oracle();

  if (failures == 0) {
    std::printf("PASS: all sub-tile halo topology tests passed (T1-T13)\n");
    std::printf("  Covers: rank<->subtile bijection (1x1, 2x2), layout=1x1 "
                "regression vs UNCHANGED legacy builder (edges + corners), "
                "layout=2x2 same-face cardinal, cross-face segment donors, "
                "corner taxonomy + counts, diagonal/mixed/true-cube corner "
                "donor resolution, and (Prompt 111) the new additive "
                "SubTileId+layout->TopologyView builder's exact 1x1 "
                "equivalence vs legacy and 2x2 correctness vs the oracle.\n");
    std::printf("  Does NOT cover: MPI message routing (see the 24-rank "
                "integration test), CUDA halo kernel application, real "
                "pack/unpack field-data movement (see the field-data "
                "harness), numerical validation.\n");
    return 0;
  }
  std::printf("FAIL: %d sub-tile halo topology check(s) failed\n", failures);
  return 1;
}
