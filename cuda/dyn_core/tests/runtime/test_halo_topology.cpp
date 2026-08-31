//===========================================================================
// Milestone 3A.10 -- host-side oracle for the cubed-sphere seam algebra.
//
// This test deliberately does NOT restate the mosaic table from
// fv3_gpu_halo_topology.cpp. Restating it would only prove the table equals
// itself. Instead it checks properties that a table encoding a real cube must
// satisfy and a mistyped one almost certainly will not:
//
//   T4  the 24 tile-corners must partition into exactly 8 three-tile junctions,
//       each appearing exactly 3 times -- a cube has 8 vertices, each shared by
//       3 faces. A single wrong entry breaks this.
//   T5  the two edge-neighbours meeting at any tile corner must themselves be
//       adjacent, since all three faces meet at one vertex.
//   T7  the index map must be an involution across a seam: if my halo point
//       receives the donor's owned point, the donor's halo point at the mirrored
//       coordinate must receive MY owned point at the original coordinate.
//   T8  the vector transforms across a seam must compose to the identity.
//
// Only T6 quotes concrete index formulae, and those four are quoted directly
// from the mosaic contact lines in CubeHalo.F90:213-261 rather than from the
// implementation.
//
// Runs on the host with no GPU and no CUDA.
//===========================================================================

#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_halo_exchange.hpp"

#include <cstdio>
#include <set>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  }
}

void check_eq(int64_t got, int64_t want, const char *what) {
  if (got != want) {
    std::printf("  FAIL: %s (got %lld, want %lld)\n", what,
                (long long)got, (long long)want);
    ++failures;
  }
}

const char *edge_name(TileEdge e) {
  switch (e) {
    case TileEdge::west: return "W";
    case TileEdge::east: return "E";
    case TileEdge::south: return "S";
    case TileEdge::north: return "N";
  }
  return "?";
}

constexpr TileEdge kEdges[4] = {TileEdge::west, TileEdge::east, TileEdge::south,
                                TileEdge::north};

bool adjacent(int a, int b) {
  for (TileEdge e : kEdges) {
    int nt = 0;
    TileEdge ne{};
    if (cubed_sphere_neighbor(a, e, nt, ne) && nt == b) return true;
  }
  return false;
}

// -------------------------------------------------------------------------
void t1_builder_rejects_bad_arguments() {
  std::printf("T1 builder argument validation\n");
  const int32_t ranks[6] = {0, 1, 2, 3, 4, 5};
  TopologyView t{};

  check(!make_cubed_sphere_topology(0, 49, 49, ranks, t), "tile 0 rejected");
  check(!make_cubed_sphere_topology(7, 49, 49, ranks, t), "tile 7 rejected");
  check(!make_cubed_sphere_topology(1, 1, 1, ranks, t), "npx<2 rejected");
  // A cube requires square tiles (CubeHalo.F90:103); a reversed seam maps my
  // along-seam extent onto the donor's perpendicular one, so nx != ny would make
  // the mapping non-total. It must be refused, not truncated.
  check(!make_cubed_sphere_topology(1, 49, 25, ranks, t), "npx != npy rejected");
  check(!make_cubed_sphere_topology(1, 49, 49, nullptr, t), "null ranks rejected");
  const int32_t bad_ranks[6] = {0, 1, -1, 3, 4, 5};
  check(!make_cubed_sphere_topology(1, 49, 49, bad_ranks, t),
        "negative rank rejected");

  // And the success case must populate, not partially fill.
  check(make_cubed_sphere_topology(3, 49, 49, ranks, t), "valid tile 3 accepted");
  check_eq(t.tile, 3, "tile recorded");
  check_eq(t.npx, 49, "npx recorded");
  check_eq(t.global_cubed_sphere, 1, "global_cubed_sphere set");
  for (TileEdge e : kEdges) {
    const EdgeNeighbor &nb = t.neighbors[(size_t)e];
    check(nb.rank >= 0, "every edge has a rank");
    check(nb.tile >= 1 && nb.tile <= 6, "every edge has a tile in 1..6");
    check(nb.tile != t.tile, "no edge names its own tile");
  }
}

// -------------------------------------------------------------------------
void t2_reciprocity() {
  std::printf("T2 reciprocity of the 12 contacts\n");
  int contacts = 0;
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    for (TileEdge mine : kEdges) {
      int donor_tile = 0;
      TileEdge donor_edge{};
      check(cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge),
            "neighbour lookup succeeds");
      int back_tile = 0;
      TileEdge back_edge{};
      check(cubed_sphere_neighbor(donor_tile, donor_edge, back_tile, back_edge),
            "reverse lookup succeeds");
      if (back_tile != tile || back_edge != mine) {
        std::printf("  FAIL: T%d %s -> T%d %s does not come back (got T%d %s)\n",
                    tile, edge_name(mine), donor_tile, edge_name(donor_edge),
                    back_tile, edge_name(back_edge));
        ++failures;
      }
      ++contacts;
    }
  }
  // 6 tiles x 4 edges = 24 directed halves of 12 undirected contacts.
  check_eq(contacts, 24, "24 directed edge relations");
}

// -------------------------------------------------------------------------
void t3_two_aligned_two_rotated() {
  std::printf("T3 each tile has 2 aligned and 2 rotated edges\n");
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    int rotated = 0;
    for (TileEdge mine : kEdges) {
      int donor_tile = 0;
      TileEdge donor_edge{};
      cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge);
      const bool rev = seam_is_reversed(mine, donor_edge);
      if (rev) ++rotated;
      // Reversal <=> the two tiles share parity. Independent formulation of the
      // same invariant, so a table edit that fixes one and not the other fails.
      const bool same_parity = ((tile & 1) == (donor_tile & 1));
      check(rev == same_parity, "reversal <=> equal tile parity");
      // Odd tiles rotate on W and N; even tiles rotate on E and S.
      const bool expect_rev =
          (tile & 1) ? (mine == TileEdge::west || mine == TileEdge::north)
                     : (mine == TileEdge::east || mine == TileEdge::south);
      check(rev == expect_rev, "odd tiles rotate on W/N, even on E/S");
    }
    check_eq(rotated, 2, "exactly two rotated edges per tile");
  }
}

// -------------------------------------------------------------------------
void t4_cube_has_eight_vertices() {
  std::printf("T4 the 24 tile-corners form exactly 8 three-tile junctions\n");
  // For each tile corner, the junction is {me, my edge neighbour on one side,
  // my edge neighbour on the other}. On a cube every such junction is a vertex
  // shared by 3 faces, so there must be exactly 8 distinct junctions and each
  // must be seen exactly 3 times.
  struct Corner { TileEdge a, b; };
  const Corner corners[4] = {{TileEdge::west, TileEdge::south},   // SW
                             {TileEdge::east, TileEdge::south},   // SE
                             {TileEdge::west, TileEdge::north},   // NW
                             {TileEdge::east, TileEdge::north}};  // NE
  std::vector<std::set<int>> seen;
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    for (const Corner &c : corners) {
      int ta = 0, tb = 0;
      TileEdge ea{}, eb{};
      cubed_sphere_neighbor(tile, c.a, ta, ea);
      cubed_sphere_neighbor(tile, c.b, tb, eb);
      // T5: the two neighbours meeting at this corner must be adjacent to each
      // other, since all three faces meet at the same cube vertex.
      if (!adjacent(ta, tb)) {
        std::printf("  FAIL: T%d corner %s%s: T%d and T%d are not adjacent\n",
                    tile, edge_name(c.a), edge_name(c.b), ta, tb);
        ++failures;
      }
      seen.push_back(std::set<int>{tile, ta, tb});
    }
  }
  check_eq((int64_t)seen.size(), 24, "24 tile-corners examined");
  std::set<std::set<int>> distinct(seen.begin(), seen.end());
  check_eq((int64_t)distinct.size(), 8, "exactly 8 distinct cube vertices");
  for (const auto &v : distinct) {
    check_eq((int64_t)v.size(), 3, "each vertex joins exactly 3 tiles");
    int count = 0;
    for (const auto &s : seen)
      if (s == v) ++count;
    check_eq(count, 3, "each vertex is reached from all 3 of its tiles");
  }
}

// -------------------------------------------------------------------------
void t5_rotation_matches_fms_and_composes_to_identity() {
  std::printf("T5 seam rotations follow FMS and compose to the identity\n");
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    for (TileEdge mine : kEdges) {
      int donor_tile = 0;
      TileEdge donor_edge{};
      cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge);

      const SeamRotation r = seam_rotation(mine, donor_edge);
      // mpp_domains_define.inc:5451/5478/5505/5532 -- an aligned seam never
      // rotates. :5459/:5486/:5513/:5540 -- a rotated seam takes +90 on the
      // x-normal edges and -90 on the y-normal ones.
      if (!seam_is_reversed(mine, donor_edge)) {
        check(r == SeamRotation::zero, "aligned seam has rotation ZERO");
      } else if (edge_is_x_normal(mine)) {
        check(r == SeamRotation::plus_ninety, "rotated W/E seam is +90");
      } else {
        check(r == SeamRotation::minus_ninety, "rotated S/N seam is -90");
      }

      // Composition: crossing the seam and coming back must be the identity.
      const SeamRotation rback = seam_rotation(donor_edge, mine);
      const SeamVectorTransform A = seam_vector_transform(r);
      const SeamVectorTransform B = seam_vector_transform(rback);
      const int cxx = A.cxx * B.cxx + A.cxy * B.cyx;
      const int cxy = A.cxx * B.cxy + A.cxy * B.cyy;
      const int cyx = A.cyx * B.cxx + A.cyy * B.cyx;
      const int cyy = A.cyx * B.cxy + A.cyy * B.cyy;
      if (cxx != 1 || cxy != 0 || cyx != 0 || cyy != 1) {
        std::printf("  FAIL: T%d %s round trip is [%d %d; %d %d], not identity\n",
                    tile, edge_name(mine), cxx, cxy, cyx, cyy);
        ++failures;
      }

      // A rotated seam must swap components; an aligned one must not.
      check(A.swaps_components() == seam_is_reversed(mine, donor_edge),
            "component swap iff seam is rotated");

      // Rotation determinant is +1 (a proper rotation, never a reflection).
      check_eq(A.cxx * A.cyy - A.cxy * A.cyx, 1, "transform determinant is +1");
    }
  }

  // SCALAR_PAIR omits the sign flip (mpp_do_updateV.fh:491, :606), leaving a
  // pure swap with determinant -1. Nothing in dyn_core.F90 passes SCALAR_PAIR,
  // but the grid metrics do (fv_grid_utils.F90:821-824), so the branch must
  // behave.
  for (SeamRotation r : {SeamRotation::plus_ninety, SeamRotation::minus_ninety}) {
    const SeamVectorTransform sp = seam_vector_transform(r, /*scalar_pair=*/true);
    check(sp.cxx == 0 && sp.cyy == 0 && sp.cxy == 1 && sp.cyx == 1,
          "SCALAR_PAIR rotation is a pure swap with no sign change");
  }
  check(!seam_vector_transform(SeamRotation::zero, true).swaps_components(),
        "SCALAR_PAIR at an aligned seam is the identity");
}

// -------------------------------------------------------------------------
void t6_index_formulae_from_the_mosaic() {
  std::printf("T6 index map reproduces four contact lines quoted from the mosaic\n");
  // nx = ny = 48 (npx = npy = 49), ng = 3.
  const int64_t nx = 48, ny = 48;
  const int ng = fv3_domain_halo_width;
  check_eq(ng, 3, "FV3 domain halo width is 3 (fv_mp_mod.F90:108)");

  struct Case {
    const char *name;
    int tile;
    TileEdge mine;
    // Expected donor point as a function of (t, h), written out from the
    // contact line rather than from the implementation.
    int expect_tile;
    int64_t (*expect_i)(int64_t t, int64_t h, int64_t nx, int64_t ny);
    int64_t (*expect_j)(int64_t t, int64_t h, int64_t nx, int64_t ny);
  };
  // contact 1: T1 EAST (j:1->ny) <-> T2 WEST (j:1->ny)
  //   T1(nx+h, j) = T2(h, j)
  // contact 4: T1 SOUTH (i:1->nx) <-> T6 NORTH (i:1->nx)
  //   T1(i, 1-h) = T6(i, ny+1-h)
  // contact 2: T1 NORTH (i:1->nx) <-> T3 WEST (j:ny->1)
  //   T1(i, ny+h) = T3(h, ny+1-i)
  // contact 3: T1 WEST (j:1->ny) <-> T5 NORTH (i:nx->1)
  //   T1(1-h, j) = T5(nx+1-j, ny+1-h)
  const Case cases[4] = {
      {"contact 1 T1 EAST <-> T2 WEST", 1, TileEdge::east, 2,
       [](int64_t, int64_t h, int64_t, int64_t) { return h; },
       [](int64_t t, int64_t, int64_t, int64_t) { return t; }},
      {"contact 4 T1 SOUTH <-> T6 NORTH", 1, TileEdge::south, 6,
       [](int64_t t, int64_t, int64_t, int64_t) { return t; },
       [](int64_t, int64_t h, int64_t, int64_t ny_) { return ny_ + 1 - h; }},
      {"contact 2 T1 NORTH <-> T3 WEST", 1, TileEdge::north, 3,
       [](int64_t, int64_t h, int64_t, int64_t) { return h; },
       [](int64_t t, int64_t, int64_t, int64_t ny_) { return ny_ + 1 - t; }},
      {"contact 3 T1 WEST <-> T5 NORTH", 1, TileEdge::west, 5,
       [](int64_t t, int64_t, int64_t nx_, int64_t) { return nx_ + 1 - t; },
       [](int64_t, int64_t h, int64_t, int64_t ny_) { return ny_ + 1 - h; }},
  };

  for (const Case &c : cases) {
    int donor_tile = 0;
    TileEdge donor_edge{};
    cubed_sphere_neighbor(c.tile, c.mine, donor_tile, donor_edge);
    check_eq(donor_tile, c.expect_tile, c.name);

    const bool rev = seam_is_reversed(c.mine, donor_edge);
    const int64_t n = edge_seam_extent(c.mine, nx, ny);
    for (int64_t t = 1; t <= n; ++t) {
      for (int h = 1; h <= ng; ++h) {
        int64_t di = 0, dj = 0;
        halo_dest_index_center(c.mine, t, h, nx, ny, di, dj);
        // The destination must actually lie in the halo, never in the interior.
        const bool in_halo = di < 1 || di > nx || dj < 1 || dj > ny;
        check(in_halo, "destination point lies outside the owned interior");

        const int64_t tp = seam_along_index(t, n, rev);
        int64_t si = 0, sj = 0;
        halo_src_index_center(donor_edge, tp, h, nx, ny, si, sj);
        // The source must lie in the donor's OWNED interior; a source in the
        // donor's own halo would be reading data the donor has not itself
        // received yet.
        const bool owned = si >= 1 && si <= nx && sj >= 1 && sj <= ny;
        check(owned, "source point lies in the donor's owned interior");

        if (si != c.expect_i(t, h, nx, ny) || sj != c.expect_j(t, h, nx, ny)) {
          std::printf("  FAIL: %s at t=%lld h=%d: got (%lld,%lld) want (%lld,%lld)\n",
                      c.name, (long long)t, h, (long long)si, (long long)sj,
                      (long long)c.expect_i(t, h, nx, ny),
                      (long long)c.expect_j(t, h, nx, ny));
          ++failures;
          return; // one report per case is enough
        }
      }
    }
  }
}

// -------------------------------------------------------------------------
void t7_index_map_is_an_involution() {
  std::printf("T7 the seam index map is mutual across every contact\n");
  const int64_t nx = 24, ny = 24;
  const int ng = fv3_domain_halo_width;
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    for (TileEdge mine : kEdges) {
      int donor_tile = 0;
      TileEdge donor_edge{};
      cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge);
      const bool rev = seam_is_reversed(mine, donor_edge);
      const int64_t n = edge_seam_extent(mine, nx, ny);
      const int64_t nd = edge_seam_extent(donor_edge, nx, ny);

      for (int64_t t = 1; t <= n; ++t) {
        const int64_t tp = seam_along_index(t, n, rev);
        // Mirroring back from the donor's side must land on t again.
        check_eq(seam_along_index(tp, nd, rev), t, "along-seam map is involutive");

        for (int h = 1; h <= ng; ++h) {
          // The donor's owned point that feeds MY halo, and MY owned point that
          // feeds the DONOR's halo, must be the same pair of cells seen from the
          // two sides -- i.e. src(donor_edge, t', h) is what dest(mine, t, h)
          // receives, and src(mine, t, h) is what dest(donor_edge, t', h)
          // receives. Both must be owned on their own tile.
          int64_t si = 0, sj = 0, mi = 0, mj = 0;
          halo_src_index_center(donor_edge, tp, h, nx, ny, si, sj);
          halo_src_index_center(mine, t, h, nx, ny, mi, mj);
          check(si >= 1 && si <= nx && sj >= 1 && sj <= ny,
                "donor source is owned");
          check(mi >= 1 && mi <= nx && mj >= 1 && mj <= ny,
                "my source is owned");

          int64_t ddi = 0, ddj = 0;
          halo_dest_index_center(donor_edge, tp, h, nx, ny, ddi, ddj);
          check(ddi < 1 || ddi > nx || ddj < 1 || ddj > ny,
                "donor destination is in the donor's halo");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------
void t8_topology_view_carries_the_derived_seam_facts() {
  std::printf("T8 TopologyView's previously-dead seam fields are populated\n");
  // Before this milestone EdgeNeighbor::edge / orientation / vector_sign_x /
  // vector_sign_y had no readers anywhere in the tree. They must now agree with
  // the derivation, or the transform kernels and the descriptor disagree.
  const int32_t ranks[6] = {10, 11, 12, 13, 14, 15};
  for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
    TopologyView t{};
    check(make_cubed_sphere_topology(tile, 49, 49, ranks, t), "build succeeds");
    for (TileEdge mine : kEdges) {
      const EdgeNeighbor &nb = t.neighbors[(size_t)mine];
      int donor_tile = 0;
      TileEdge donor_edge{};
      cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge);
      check_eq(nb.tile, donor_tile, "neighbour tile matches the mosaic");
      check_eq(nb.rank, ranks[donor_tile - 1], "neighbour rank is the tile's rank");
      check(nb.edge == donor_edge, "neighbour edge is the DONOR's own edge");
      check(nb.orientation == seam_orientation(mine, donor_edge),
            "orientation matches the derivation");

      const SeamVectorTransform vt =
          seam_vector_transform(seam_rotation(mine, donor_edge));
      if (vt.swaps_components()) {
        check_eq(nb.vector_sign_x, vt.cxy >= 0 ? 1 : -1, "vector_sign_x");
        check_eq(nb.vector_sign_y, vt.cyx >= 0 ? 1 : -1, "vector_sign_y");
        // A proper rotation flips exactly one of the two components.
        check_eq(nb.vector_sign_x * nb.vector_sign_y, -1,
                 "a rotated seam flips exactly one component");
      } else {
        check_eq(nb.vector_sign_x, 1, "aligned seam has no x sign change");
        check_eq(nb.vector_sign_y, 1, "aligned seam has no y sign change");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// T9. The staggered index algebra, checked against FMS's OWN halo oracle.
//
// GFDL FMS 2024.03, test_fms/mpp/test_mpp_domains.F90:4092-4126,
// subroutine fill_cubic_grid_halo. That routine is what FMS itself checksums
// mpp_update_domains against on a cubic grid, so it is an independent statement
// of the answer rather than a restatement of the implementation. Its eight
// branches are transcribed literally below, in Fortran index terms, and compared
// against halo_dest_index / halo_src_index / donor_seam_extent for every
// position, every tile, every edge, every halo ring and every along-seam point.
//
// It also transcribes the sign arguments from the two cubic-grid call sites --
//   :1552  x-component  (ioff,joff)=(1,0)  (sign1,sign2)=(+1,-1)
//   :1553  y-component  (ioff,joff)=(0,1)  (sign1,sign2)=(-1,+1)
// -- and checks them against seam_vector_transform, closing §6 from a second
// direction. sign1 governs the odd tiles' WEST and even tiles' EAST halo,
// sign2 the odd tiles' NORTH and even tiles' SOUTH; those are exactly the four
// rotated seams.
// ---------------------------------------------------------------------------
void t9_staggered_index_map_matches_the_fms_oracle() {
  std::printf("T9 staggered index map vs FMS fill_cubic_grid_halo\n");

  const int64_t nx = 7, ny = 7;          // square, as a cube requires
  const int32_t ng = fv3_domain_halo_width;

  struct Position { const char *name; StaggerShift s; };
  const Position positions[4] = {{"CENTER", stagger_center},
                                 {"EAST", stagger_east},
                                 {"NORTH", stagger_north},
                                 {"CORNER", stagger_corner}};

  for (const auto &pos : positions) {
    const int64_t ioff = pos.s.i, joff = pos.s.j;

    for (int tile = 1; tile <= cubed_sphere_tiles; ++tile) {
      const bool even = (tile % 2) == 0;

      // The oracle's own neighbour arithmetic, :4100-4103 and :4116-4119.
      int lw, le, ls, ln;
      if (even) { lw = tile - 1; le = tile + 2; ls = tile - 2; ln = tile + 1; }
      else      { lw = tile - 2; le = tile + 1; ls = tile - 1; ln = tile + 2; }
      if (le > 6) le -= 6;
      if (ln > 6) ln -= 6;
      if (lw < 1) lw += 6;
      if (ls < 1) ls += 6;

      for (const TileEdge mine : {TileEdge::west, TileEdge::east,
                                  TileEdge::south, TileEdge::north}) {
        int donor_tile = 0; TileEdge donor_edge{};
        check(cubed_sphere_neighbor(tile, mine, donor_tile, donor_edge),
              "neighbour lookup");

        // (a) donor tile must be the oracle's.
        int want_tile = 0;
        switch (mine) {
          case TileEdge::west:  want_tile = lw; break;
          case TileEdge::east:  want_tile = le; break;
          case TileEdge::south: want_tile = ls; break;
          case TileEdge::north: want_tile = ln; break;
        }
        check_eq(donor_tile, want_tile, "T9 donor tile matches FMS oracle");

        // (b) the oracle applies a sign and reads the OTHER component array on
        //     exactly the seams my derivation calls rotated.
        const bool oracle_rotates =
            even ? (mine == TileEdge::east || mine == TileEdge::south)
                 : (mine == TileEdge::west || mine == TileEdge::north);
        const bool reversed = seam_is_reversed(mine, donor_edge);
        check(oracle_rotates == reversed,
              "T9 rotated seams are exactly where FMS applies a sign");

        const StaggerShift src = source_shift(pos.s, reversed);
        const int64_t n_dest = edge_seam_extent(mine, nx, ny, pos.s);
        const int64_t n_donor = donor_seam_extent(donor_edge, nx, ny, src);
        check_eq(n_dest, n_donor, "T9 seam extents agree across the seam");

        // (c) the index map itself, point by point.
        for (int32_t h = 1; h <= ng; ++h) {
          for (int64_t t = 1; t <= n_dest; ++t) {
            int64_t di = 0, dj = 0, si = 0, sj = 0;
            halo_dest_index(mine, t, h, nx, ny, pos.s, di, dj);
            halo_src_index(donor_edge, seam_along_index(t, n_donor, reversed), h,
                           nx, ny, src, si, sj);

            // The oracle, transcribed. Fortran `i` there is my `h`.
            int64_t wi = 0, wj = 0, wsi = 0, wsj = 0;
            if (!even) {
              switch (mine) {
                case TileEdge::west:   // :4116-4118
                  wi = 1 - h;            wj = t;
                  wsi = (nx + joff) + 1 - t;  wsj = ny - h + 1;      break;
                case TileEdge::east:   // :4119
                  wi = (nx + ioff) + h;  wj = t;
                  wsi = ioff + h;        wsj = t;                    break;
                case TileEdge::south:  // :4120
                  wi = t;                wj = 1 - h;
                  wsi = t;               wsj = ny - h + 1;           break;
                case TileEdge::north:  // :4121-4123
                  wi = t;                wj = (ny + joff) + h;
                  wsi = h + joff;        wsj = (ny + ioff) + 1 - t;  break;
              }
            } else {
              switch (mine) {
                case TileEdge::west:   // :4105
                  wi = 1 - h;            wj = t;
                  wsi = nx - h + 1;      wsj = t;                    break;
                case TileEdge::east:   // :4106-4108
                  wi = (nx + ioff) + h;  wj = t;
                  wsi = (nx + joff) + 1 - t;  wsj = h + ioff;        break;
                case TileEdge::south:  // :4109-4111
                  wi = t;                wj = 1 - h;
                  wsi = nx - h + 1;      wsj = (ny + ioff) + 1 - t;  break;
                case TileEdge::north:  // :4112
                  wi = t;                wj = (ny + joff) + h;
                  wsi = t;               wsj = joff + h;             break;
              }
            }
            check_eq(di, wi, "T9 dest i");
            check_eq(dj, wj, "T9 dest j");
            check_eq(si, wsi, "T9 src i");
            check_eq(sj, wsj, "T9 src j");

            // The destination must be outside my own owned block, and the
            // source inside the donor's -- for the SOURCE ARRAY's shape, which
            // on a rotated seam is the transposed one.
            check(di < 1 || di > nx + ioff || dj < 1 || dj > ny + joff,
                  "T9 destination is a halo point");
            check(si >= 1 && si <= nx + src.i && sj >= 1 && sj <= ny + src.j,
                  "T9 source is owned by the donor, in the source array's shape");
          }
        }

        // (d) the component signs, for the rotated seams only.
        if (!reversed) continue;
        const SeamVectorTransform vt = seam_vector_transform(seam_rotation(mine, donor_edge));
        // FMS passes sign1 for the odd-W / even-E halo and sign2 for the
        // odd-N / even-S halo; x-component is (+1,-1), y-component (-1,+1).
        const bool is_sign1 =
            even ? (mine == TileEdge::east) : (mine == TileEdge::west);
        const int want_x = is_sign1 ? +1 : -1;   // :1552 (sign1, sign2)
        const int want_y = is_sign1 ? -1 : +1;   // :1553 (sign1, sign2)
        check_eq(vt.cxy, want_x, "T9 x-component sign matches FMS oracle");
        check_eq(vt.cyx, want_y, "T9 y-component sign matches FMS oracle");
        check_eq(vt.cxx, 0, "T9 rotated seam takes no same-component x term");
        check_eq(vt.cyy, 0, "T9 rotated seam takes no same-component y term");
      }
    }
  }
}


//---------------------------------------------------------------------------
// T10 -- the halo GROUP tables and the panel-seam sign rule.
//
// The point of this test is the C-grid / D-grid distinction that §6 warns about:
// "Do not assume all vector seams use one transform," and "Where authoritative
// behavior uses different transforms for C-grid and D-grid winds, preserve the
// distinction."
//
// The distinction is real, and it is invisible if you look at the sign table
// alone, because in ARGUMENT-SLOT terms C-grid and D-grid are identical -- FMS
// has one table with no gridtype test (group_update_pack.inc). It only becomes
// visible in STAGGERING terms, because the two calls bind the slots opposite ways
// round: dyn_core.F90:618 passes (uc, vc) with uc EAST-shaped, while :402/:1109
// pass (u, v) with u NORTH-shaped, and mpp_group_update.fh:356-371 assigns
// position_x = EAST for CGRID_NE but position_x = NORTH for DGRID_NE.
//
// So the load-bearing check here is the LAST one: at one and the same seam, the
// EAST-staggered member of the C-grid pair and the EAST-staggered member of the
// D-grid pair take OPPOSITE signs. An implementation that keyed the sign on
// staggering with a single table -- the plausible, wrong design -- would give
// them the same sign and would fail this check. Nothing else in the suite would
// notice.
//---------------------------------------------------------------------------
void t10_group_tables_and_the_slot_keyed_sign_rule() {
  std::printf("T10 halo group tables and the slot-keyed seam sign rule\n");
  using namespace fv3::gpu::halo;

  // (a) Scalars never take a sign, at any rotation. FIELD_S is `+field` in every
  // branch of group_update_pack.inc (:39/:51/:63/:75). This is what makes divgd
  // legal on a rotated seam at position=CORNER.
  for (SeamRotation r : {SeamRotation::zero, SeamRotation::plus_ninety,
                         SeamRotation::minus_ninety}) {
    check_eq(seam_component_sign(VectorRole::scalar, r), 1,
             "T10 a scalar crosses a seam unsigned");
  }

  // (b) The four explicit assignments, from group_update_pack.inc. A receiver
  // whose own halo edge is WEST or EAST sees rotation NINETY; SOUTH or NORTH sees
  // MINUS_NINETY (mpp_domains_define.inc:5459/:5486/:5513/:5540, rotateRecv).
  check_eq(seam_component_sign(VectorRole::slot_x, SeamRotation::plus_ninety), +1,
           "T10 FIELD_X at NINETY takes +fieldy");        // :128
  check_eq(seam_component_sign(VectorRole::slot_y, SeamRotation::plus_ninety), -1,
           "T10 FIELD_Y at NINETY takes -fieldx");        // :206
  check_eq(seam_component_sign(VectorRole::slot_x, SeamRotation::minus_ninety), -1,
           "T10 FIELD_X at MINUS_NINETY takes -fieldy");  // :115
  check_eq(seam_component_sign(VectorRole::slot_y, SeamRotation::minus_ninety), +1,
           "T10 FIELD_Y at MINUS_NINETY takes +fieldx");  // :181

  // (c) An unrotated seam is a plain copy for every role.
  for (VectorRole role : {VectorRole::scalar, VectorRole::slot_x, VectorRole::slot_y}) {
    check_eq(seam_component_sign(role, SeamRotation::zero), 1,
             "T10 an aligned seam applies no sign");
  }

  // (d) Every live group must be internally consistent, and every field's seam
  // partner must be resolvable inside its OWN group -- the transfer step relies
  // on that, and a partner living elsewhere would silently read slot -1.
  int live_groups = 0;
  for (uint32_t g = 1; g <= FV3_GPU_H_GROUP_LAST; ++g) {
    const HaloGroupSpec s = halo_group_spec(g);
    check_eq(static_cast<int>(s.group), static_cast<int>(g), "T10 group id round-trips");
    if (!s.live) {
      check_eq(s.field_count, 0, "T10 a group that is not live carries no fields");
      continue;
    }
    ++live_groups;
    check(s.field_count >= 1 && s.field_count <= max_halo_group_fields,
          "T10 live group has a sane field count");
    // Under-exchanging is a silent wrong answer; the authoritative domain halo
    // is fv_mp_mod.F90:108's ng = 3. In particular H_PKC2's name says 2 and its
    // width must still be 3, or the misnomer becomes a bug.
    //
    // H_DWIND_SEAM is the one group that is NOT a halo exchange: it is E11
    // (dyn_core.F90:1178), the reconciliation of the shared staggered seam LINE,
    // which lives at ring 0 -- inside the owned block, not outside it. It
    // therefore has ring_begin == 0 and halo_width == 0, i.e. exactly one ring,
    // and asserting the domain halo width for it would be wrong. Every other
    // group must still exchange ng = 3 rings starting at ring 1.
    if (g == FV3_GPU_H_DWIND_SEAM) {
      check_eq(s.ring_begin, 0, "T10 E11 reconciles ring 0, the shared owned line");
      check_eq(s.halo_width, 0, "T10 E11 spans the single ring 0");
      check_eq(halo_ring_count(s), 1, "T10 E11 is one ring wide");
      // Its destinations are OWNED cells, so each field must decline three of
      // the four edges: u writes only NORTH (u(is:ie,je+1)), v only EAST
      // (v(ie+1,js:je)). A 0xF mask here would overwrite owned values on the
      // south/west lines that the authoritative code never touches.
      for (int n = 0; n < s.field_count; ++n) {
        const FieldId id = s.fields[n].id;
        const uint8_t want = (id == FieldId::u)
            ? static_cast<uint8_t>(1u << static_cast<int>(TileEdge::north))
            : static_cast<uint8_t>(1u << static_cast<int>(TileEdge::east));
        check(id == FieldId::u || id == FieldId::v, "T10 E11 carries u and v only");
        check_eq((int)s.fields[n].dest_edges, (int)want,
                 "T10 E11 unpacks one edge per field, matching :1181-1188");
        // And the ring-0 destination really is the authoritative line, extents
        // included -- this is the whole reason E11 needs no new index algebra.
        const int64_t nx = 12, ny = 12;
        const TileEdge de = (id == FieldId::u) ? TileEdge::north : TileEdge::east;
        const int64_t along = edge_seam_extent(de, nx, ny, s.fields[n].stagger);
        check_eq((long long)along, (long long)((id == FieldId::u) ? nx : ny),
                 "T10 E11's along-extent is exactly is:ie (u) or js:je (v)");
        for (int64_t t = 1; t <= along; ++t) {
          int64_t i = 0, j = 0;
          halo_dest_index(de, t, 0, nx, ny, s.fields[n].stagger, i, j);
          if (id == FieldId::u)
            check(i == t && j == ny + 1, "T10 E11 writes exactly u(is:ie,je+1,:)");
          else
            check(i == nx + 1 && j == t, "T10 E11 writes exactly v(ie+1,js:je,:)");
        }
      }
    } else {
      check_eq(s.ring_begin, 1, "T10 a halo group starts at the first ring outside the owned block");
      check_eq(s.halo_width, fv3_domain_halo_width,
               "T10 group exchanges the authoritative domain halo width");
      for (int n = 0; n < s.field_count; ++n)
        check_eq((int)s.fields[n].dest_edges, 0xF,
                 "T10 an ordinary halo group unpacks all four edges");
    }
    for (int n = 0; n < s.field_count; ++n) {
      const HaloFieldSpec &f = s.fields[n];
      check(f.id != FieldId::invalid, "T10 field id is set");
      if (f.role == VectorRole::scalar) {
        check(f.partner == FieldId::invalid, "T10 a scalar has no seam partner");
        check(f.stagger == stagger_center || f.stagger == stagger_corner,
              "T10 a scalar sits at CENTER or CORNER");
        continue;
      }
      // The partner must exist in this group, must name us back, and must hold
      // the other slot.
      int pn = -1;
      for (int m = 0; m < s.field_count; ++m)
        if (s.fields[m].id == f.partner) pn = m;
      check(pn >= 0, "T10 vector field's partner is a member of its own group");
      if (pn < 0) continue;
      const HaloFieldSpec &p = s.fields[pn];
      check(p.partner == f.id, "T10 partnership is symmetric");
      check(p.role != f.role, "T10 the pair occupies both slots, not one twice");
      // The index algebra and the slot algebra must independently agree on WHICH
      // array is read across a rotated seam. This is claimed in a comment in
      // fv3_gpu_halo_exchange.hpp; here it is checked.
      check(source_shift(f.stagger, true) == p.stagger,
            "T10 the transposed source shift is exactly the partner's own stagger");
    }
  }
  // Twelve enum members (H_DWIND_IN .. H_DWIND_SEAM); eleven are live. The one
  // that is not is H_Q_INLINE. H_DWIND_SEAM is a real group AND doubles as the
  // array-sizing sentinel (FV3_GPU_H_GROUP_LAST), which is why the count is
  // eleven and not ten.
  check_eq(live_groups, 11, "T10 eleven of the twelve enum groups are live (q is not)");
  check(!halo_group_spec(FV3_GPU_H_Q_INLINE).live,
        "T10 H_Q_INLINE is not live: no stage in the supported path reads a q halo");

  // (e) THE §6 CHECK. Same seam, same staggering, opposite sign, because the two
  // grids bind the argument slots opposite ways round.
  const HaloGroupSpec cg = halo_group_spec(FV3_GPU_H_CWIND);
  const HaloGroupSpec dg = halo_group_spec(FV3_GPU_H_DWIND_IN);
  auto sign_of_stagger = [](const HaloGroupSpec &s, StaggerShift want, SeamRotation r) {
    for (int n = 0; n < s.field_count; ++n)
      if (s.fields[n].stagger == want) return (int)seam_component_sign(s.fields[n].role, r);
    return 0;
  };
  for (SeamRotation r : {SeamRotation::plus_ninety, SeamRotation::minus_ninety}) {
    const int c_east = sign_of_stagger(cg, stagger_east, r);
    const int d_east = sign_of_stagger(dg, stagger_east, r);
    const int c_north = sign_of_stagger(cg, stagger_north, r);
    const int d_north = sign_of_stagger(dg, stagger_north, r);
    check(c_east != 0 && d_east != 0 && c_north != 0 && d_north != 0,
          "T10 both wind pairs cover both staggerings");
    check_eq(c_east, -d_east,
             "T10 C-grid and D-grid sign the EAST-staggered member OPPOSITELY");
    check_eq(c_north, -d_north,
             "T10 C-grid and D-grid sign the NORTH-staggered member OPPOSITELY");
    // ...while in slot terms they are the same one table.
    check_eq((int)seam_component_sign(cg.fields[0].role, r),
             (int)seam_component_sign(dg.fields[0].role, r),
             "T10 in ARGUMENT-SLOT terms the two grids use one identical transform");
  }

  // (f) Buffer slots must not overlap: 4 edges x field_count, each of one slot.
  for (uint32_t g = 1; g <= FV3_GPU_H_GROUP_LAST; ++g) {
    const HaloGroupSpec s = halo_group_spec(g);
    if (!s.live) continue;
    const HaloGeometry geom{12, 12, 5};
    const int64_t slot = group_slot_elements(geom, halo_ring_count(s));
    check(slot > 0, "T10 slot is non-empty");
    int64_t expect = 0;
    for (int n = 0; n < s.field_count; ++n) {
      for (int q = 0; q < 4; ++q) {
        check_eq((long long)slot_base(geom, halo_ring_count(s), n, (TileEdge)q),
                 (long long)expect, "T10 slots tile the buffer without overlap");
        expect += slot;
      }
    }
    check_eq((long long)group_buffer_elements(s, geom), (long long)expect,
             "T10 buffer is exactly the sum of its slots");
    // Every (edge, staggering) strip must fit inside one slot.
    for (int n = 0; n < s.field_count; ++n) {
      for (int q = 0; q < 4; ++q) {
        const int64_t along =
            edge_seam_extent((TileEdge)q, geom.nx, geom.ny, s.fields[n].stagger);
        check(slot_offset(along, geom.npz + 1, halo_ring_count(s), along, geom.npz) < slot,
              "T10 the last element of a strip lies inside its slot");
      }
    }
  }
}

} // namespace

int main() {
  std::printf("=== Milestone 3A.10 cubed-sphere seam algebra ===\n");
  t1_builder_rejects_bad_arguments();
  t2_reciprocity();
  t3_two_aligned_two_rotated();
  t4_cube_has_eight_vertices();
  t5_rotation_matches_fms_and_composes_to_identity();
  t6_index_formulae_from_the_mosaic();
  t7_index_map_is_an_involution();
  t8_topology_view_carries_the_derived_seam_facts();
  t9_staggered_index_map_matches_the_fms_oracle();
  t10_group_tables_and_the_slot_keyed_sign_rule();

  if (failures) {
    std::printf("\nFAILED: %d check(s)\n", failures);
    return 1;
  }
  std::printf("\nAll cubed-sphere seam algebra checks passed.\n");
  std::printf("  * This validates CONNECTIVITY, the index map for all four FMS\n");
  std::printf("    positions (CENTER / EAST / NORTH / CORNER), and the panel-seam\n");
  std::printf("    vector signs, against FMS's own cubic-grid halo oracle.\n");
  std::printf("  * It also validates the halo GROUP tables and the slot-keyed\n");
  std::printf("    vector sign rule, including that C-grid and D-grid winds take\n");
  std::printf("    OPPOSITE signs at one seam in staggering terms.\n");
  std::printf("  * It does NOT cover the shared staggered seam line, which FV3\n");
  std::printf("    reconciles separately via mpp_get_boundary (exchange E11), nor\n");
  std::printf("    the three-tile junction points at the cube vertices.\n");
  std::printf("  * Numerical validation against authoritative Fortran savepoints:\n");
  std::printf("    NOT RUN.\n");
  return 0;
}
