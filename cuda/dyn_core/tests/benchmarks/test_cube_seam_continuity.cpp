// =============================================================================
// Milestone 3A.11V Task 1 -- seam-continuity test for the globally consistent
// cubed-sphere benchmark IC.
//
// WHAT THIS TEST IS FOR
// ---------------------
// The user requirement is: "Add a seam-continuity test that fails if
// neighbouring tiles disagree beyond the expected floating-point tolerance."
// The point is to make the single-valuedness of the benchmark IC a checked
// property rather than an assertion in a comment.
//
// WHY SOME OBVIOUS TESTS WOULD BE WORTHLESS HERE
// ----------------------------------------------
// A test that asks "does tile n's halo value equal the value my own halo map
// says it should" is circular -- it re-runs corner_owner and compares against
// corner_owner. Three of the five checks below are therefore built to be
// independent of the halo map:
//
//   T2 compares the generated grid against FV3's OWN thirteen explicit seam
//      copies (REF/model/fv_grid_tools.F90:720-736), which were written without
//      reference to this port. If the closed form or the tile rotations are
//      wrong, T2 fails.
//   T4 finds each halo cell's twin by BRUTE-FORCE GEOMETRIC SEARCH over all six
//      tiles' interior cells. It never consults corner_owner. If the halo index
//      map is wrong, the search finds the wrong cell or no cell.
//   T5 finds each halo EDGE's twin the same way, by unordered endpoint match,
//      and then re-derives the seam sign/swap table from the matched pairs. The
//      derived table is printed and can be read against the independent audit
//      of FMS semantics. If the tangent construction or the projection is
//      wrong, the signs come out inconsistent.
//
// This is a HOST-SIDE GEOMETRY test. It validates the initial state's
// single-valuedness. It does NOT validate the CUDA halo exchange, and it is NOT
// a numerical validation against authoritative Fortran output.
// =============================================================================
#include "fv3_cube_geometry.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fv3::bench::cube;

namespace {

int g_failures = 0;

void ok(bool cond, const char *what) {
  if (!cond) {
    std::printf("  FAIL: %s\n", what);
    ++g_failures;
  }
}

void okf(bool cond, const char *fmt, double a, double b) {
  if (!cond) {
    std::printf("  FAIL: ");
    std::printf(fmt, a, b);
    std::printf("\n");
    ++g_failures;
  }
}

Vec3 xyz(LonLat p) { return latlon2xyz(p); }

double dist3(Vec3 a, Vec3 b) {
  const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double dist_ll(LonLat a, LonLat b) { return dist3(xyz(a), xyz(b)); }

// -----------------------------------------------------------------------------
// T1: primitives and the tile rotations.
// -----------------------------------------------------------------------------
void t1_primitives(int im) {
  std::printf("T1 primitives and tile rotations (im=%d)\n", im);

  // Every rotation must be orthogonal with det = +1: a PROPER rotation. The
  // Fortran routine is called mirror_grid, so "these are not reflections" is a
  // claim worth checking rather than assuming.
  for (int t = 1; t <= 6; ++t) {
    const Vec3 ex = tile_rotate(t, Vec3{1, 0, 0});
    const Vec3 ey = tile_rotate(t, Vec3{0, 1, 0});
    const Vec3 ez = tile_rotate(t, Vec3{0, 0, 1});
    okf(std::fabs(inner_prod(ex, ex) - 1.0) < 1e-15, "tile %.0f ex not unit (%.3e)",
        (double)t, inner_prod(ex, ex) - 1.0);
    okf(std::fabs(inner_prod(ex, ey)) < 1e-15, "tile %.0f ex.ey != 0 (%.3e)",
        (double)t, inner_prod(ex, ey));
    okf(std::fabs(inner_prod(ex, ez)) < 1e-15, "tile %.0f ex.ez != 0 (%.3e)",
        (double)t, inner_prod(ex, ez));
    const double det = inner_prod(ex, vect_cross(ey, ez));
    okf(std::fabs(det - 1.0) < 1e-15, "tile %.0f det = %.6f, not +1 (improper)",
        (double)t, det);
  }

  // Tile 1 extent before the shift: lon in [-45,45], lat in +-asin(1/sqrt 3).
  const double alpha_deg = std::asin(1.0 / std::sqrt(3.0)) * 180.0 / M_PI;
  const LonLat sw = global_corner(1, 1, 1, im, /*shift=*/false);
  const LonLat ne = global_corner(1, im + 1, im + 1, im, false);
  okf(std::fabs(sw.lon * 180.0 / M_PI + 45.0) < 1e-10,
      "tile1 SW lon = %.10f deg, expected -45 (%.1e)", sw.lon * 180.0 / M_PI, 0.0);
  okf(std::fabs(sw.lat * 180.0 / M_PI + alpha_deg) < 1e-10,
      "tile1 SW lat = %.10f deg, expected %.10f", sw.lat * 180.0 / M_PI, -alpha_deg);
  okf(std::fabs(ne.lon * 180.0 / M_PI - 45.0) < 1e-10,
      "tile1 NE lon = %.10f deg, expected +45 (%.1e)", ne.lon * 180.0 / M_PI, 0.0);
  okf(std::fabs(ne.lat * 180.0 / M_PI - alpha_deg) < 1e-10,
      "tile1 NE lat = %.10f deg, expected %.10f", ne.lat * 180.0 / M_PI, alpha_deg);
  std::printf("  tile1 spans lon [%.6f, %.6f] lat [%.6f, %.6f] deg (alpha=%.10f)\n",
              sw.lon * 180.0 / M_PI, ne.lon * 180.0 / M_PI, sw.lat * 180.0 / M_PI,
              ne.lat * 180.0 / M_PI, alpha_deg);

  // Tiles 3 and 6 are the polar caps. tile_rotate acts on the frame in which
  // tile 1 is already centred on the equator and the Greenwich meridian -- i.e.
  // AFTER the `lon -= pi` rotation, per mirror_grid's own precondition comment
  // -- so tile 1's face centre in that frame is (+1,0,0), not (-1,0,0).
  const Vec3 c3 = tile_rotate(3, Vec3{1, 0, 0});
  const Vec3 c6 = tile_rotate(6, Vec3{1, 0, 0});
  okf(c3.z > 0.999, "tile 3 is not the NORTH cap: centre z = %.6f (%.1f)", c3.z, 0.0);
  okf(c6.z < -0.999, "tile 6 is not the SOUTH cap: centre z = %.6f (%.1f)", c6.z, 0.0);

  // Equidistant, not equiangular: gnomonic_ed's own doc comment
  // (REF/model/fv_grid_utils.F90:1319-1328) states max(dx,dy)/min(dx,dy) = sqrt(2).
  // Measure it along tile 1's j = 1 edge and the i = 1 column.
  double dmin = 1e300, dmax = 0.0;
  for (int j = 1; j <= im; ++j)
    for (int i = 1; i <= im; ++i) {
      const double d =
          great_circle_dist(global_corner(1, i, j, im, false),
                            global_corner(1, i + 1, j, im, false), 1.0);
      if (d < dmin) dmin = d;
      if (d > dmax) dmax = d;
    }
  std::printf("  i-spacing ratio max/min = %.6f (gnomonic_ed doc says the grid "
              "aspect bound is sqrt(2) = %.6f)\n",
              dmax / dmin, std::sqrt(2.0));
  ok(dmax / dmin < std::sqrt(2.0) + 1e-9, "i-spacing ratio exceeds sqrt(2)");
}

// -----------------------------------------------------------------------------
// T2: FV3's own thirteen seam copies.
//
// REF/model/fv_grid_tools.F90:720-736, transcribed verbatim as (dst tile, dst
// index expression) = (src tile, src index expression). Fortran is 1-based and
// so is this. `npx` here is N = im+1.
//
// This is the strongest available check on the closed form and the rotations,
// because these thirteen statements are FV3's own independent expression of the
// cube connectivity.
// -----------------------------------------------------------------------------
void t2_fv3_seam_copies(int im) {
  const int N = im + 1;
  std::printf("T2 FV3's 13 explicit seam copies (fv_grid_tools.F90:720-736)\n");

  struct Copy {
    const char *text;
    int dst_tile, src_tile;
    // index expressions as functions of the loop variable m in [1,N]
    int (*di)(int m, int N);
    int (*dj)(int m, int N);
    int (*si)(int m, int N);
    int (*sj)(int m, int N);
  };
  // helpers
  auto one = [](int, int) { return 1; };
  (void)one;

  static const Copy copies[] = {
      // grid_global(1,1:npy,:,2) = grid_global(npx,1:npy,:,1)
      {"G2(1,j)=G1(npx,j)", 2, 1, [](int, int) { return 1; },
       [](int m, int) { return m; }, [](int, int N) { return N; },
       [](int m, int) { return m; }},
      // grid_global(1,1:npy,:,3) = grid_global(npx:1:-1,npy,:,1)
      {"G3(1,j)=G1(npx+1-j,npy)", 3, 1, [](int, int) { return 1; },
       [](int m, int) { return m; }, [](int m, int N) { return N + 1 - m; },
       [](int, int N) { return N; }},
      // grid_global(1:npx,npy,:,5) = grid_global(1,npy:1:-1,:,1)
      {"G5(i,npy)=G1(1,npy+1-i)", 5, 1, [](int m, int) { return m; },
       [](int, int N) { return N; }, [](int, int) { return 1; },
       [](int m, int N) { return N + 1 - m; }},
      // grid_global(1:npx,npy,:,6) = grid_global(1:npx,1,:,1)
      {"G6(i,npy)=G1(i,1)", 6, 1, [](int m, int) { return m; },
       [](int, int N) { return N; }, [](int m, int) { return m; },
       [](int, int) { return 1; }},
      // grid_global(1:npx,1,:,3) = grid_global(1:npx,npy,:,2)
      {"G3(i,1)=G2(i,npy)", 3, 2, [](int m, int) { return m; },
       [](int, int) { return 1; }, [](int m, int) { return m; },
       [](int, int N) { return N; }},
      // grid_global(1:npx,1,:,4) = grid_global(npx,npy:1:-1,:,2)
      {"G4(i,1)=G2(npx,npy+1-i)", 4, 2, [](int m, int) { return m; },
       [](int, int) { return 1; }, [](int, int N) { return N; },
       [](int m, int N) { return N + 1 - m; }},
      // grid_global(npx,1:npy,:,6) = grid_global(npx:1:-1,1,:,2)
      {"G6(npx,j)=G2(npx+1-j,1)", 6, 2, [](int, int N) { return N; },
       [](int m, int) { return m; }, [](int m, int N) { return N + 1 - m; },
       [](int, int) { return 1; }},
      // grid_global(1,1:npy,:,4) = grid_global(npx,1:npy,:,3)
      {"G4(1,j)=G3(npx,j)", 4, 3, [](int, int) { return 1; },
       [](int m, int) { return m; }, [](int, int N) { return N; },
       [](int m, int) { return m; }},
      // grid_global(1,1:npy,:,5) = grid_global(npx:1:-1,npy,:,3)
      {"G5(1,j)=G3(npx+1-j,npy)", 5, 3, [](int, int) { return 1; },
       [](int m, int) { return m; }, [](int m, int N) { return N + 1 - m; },
       [](int, int N) { return N; }},
      // grid_global(npx,1:npy,:,3) = grid_global(1,1:npy,:,4)
      {"G3(npx,j)=G4(1,j)", 3, 4, [](int, int N) { return N; },
       [](int m, int) { return m; }, [](int, int) { return 1; },
       [](int m, int) { return m; }},
      // grid_global(1:npx,1,:,5) = grid_global(1:npx,npy,:,4)
      {"G5(i,1)=G4(i,npy)", 5, 4, [](int m, int) { return m; },
       [](int, int) { return 1; }, [](int m, int) { return m; },
       [](int, int N) { return N; }},
      // grid_global(1:npx,1,:,6) = grid_global(npx,npy:1:-1,:,4)
      {"G6(i,1)=G4(npx,npy+1-i)", 6, 4, [](int m, int) { return m; },
       [](int, int) { return 1; }, [](int, int N) { return N; },
       [](int m, int N) { return N + 1 - m; }},
      // grid_global(1,1:npy,:,6) = grid_global(npx,1:npy,:,5)
      {"G6(1,j)=G5(npx,j)", 6, 5, [](int, int) { return 1; },
       [](int m, int) { return m; }, [](int, int N) { return N; },
       [](int m, int) { return m; }},
  };

  int checked = 0;
  double worst = 0.0;
  const char *worst_what = "";
  for (const Copy &c : copies) {
    double w = 0.0;
    for (int m = 1; m <= N; ++m) {
      const LonLat d = global_corner(c.dst_tile, c.di(m, N), c.dj(m, N), im, false);
      const LonLat s = global_corner(c.src_tile, c.si(m, N), c.sj(m, N), im, false);
      const double e = dist_ll(d, s);
      if (e > w) w = e;
      ++checked;
    }
    if (w > worst) { worst = w; worst_what = c.text; }
    if (w > 1e-14) {
      std::printf("  FAIL: %-26s max chord error %.3e\n", c.text, w);
      ++g_failures;
    }
  }
  std::printf("  13/13 copies checked over %d corner pairs; worst chord error "
              "%.3e at %s (tol 1e-14)\n",
              checked, worst, worst_what);
}

// -----------------------------------------------------------------------------
// T3: halo corners are the NEIGHBOUR's grid, not the local panel's analytic
// continuation -- and the difference is O(h), not O(1).
//
// This is the check that catches a wrong neighbour tile or a transposed index.
// Those errors put the halo corner O(1) away (a different part of the sphere).
// The CORRECT answer is a small but nonzero distance from the local gnomonic
// extension, because the neighbour panel projects from a different face plane.
// Asserting the difference is small-but-nonzero pins both failure modes at once:
// exactly zero would mean corner_owner silently fell through to the local
// extension.
// -----------------------------------------------------------------------------
void t3_halo_is_neighbour(int im, int halo) {
  const int N = im + 1;
  std::printf("T3 halo corners come from the neighbour, at O(h) from the local "
              "gnomonic extension\n");
  double worst_ratio = 0.0;
  int worst_t = 0, worst_i = 0, worst_j = 0;
  int n_zero = 0, n_checked = 0;
  // Representative grid spacing in chord units, for the O(h) scale.
  const double h = dist_ll(global_corner(1, 1, 1, im, false),
                           global_corner(1, 2, 1, im, false));
  for (int t = 1; t <= 6; ++t) {
    CubeGrid g(im, t, halo);
    for (int j = 1 - halo; j <= N + halo; ++j)
      for (int i = 1 - halo; i <= N + halo; ++i) {
        const bool interior = (i >= 1 && i <= N && j >= 1 && j <= N);
        if (interior || !g.corner_owned(i, j)) continue;
        const double d =
            dist_ll(g.corner(i, j), global_corner(t, i, j, im, true));
        ++n_checked;
        if (d == 0.0) ++n_zero;
        const double r = d / h;
        if (r > worst_ratio) { worst_ratio = r; worst_t = t; worst_i = i; worst_j = j; }
      }
  }
  std::printf("  %d owned halo corners; worst |halo - local extension| / h = "
              "%.4f at tile %d (i=%d,j=%d); h = %.6e chord\n",
              n_checked, worst_ratio, worst_t, worst_i, worst_j, h);
  ok(n_checked > 0, "no owned halo corners were examined at all");
  // NO upper bound is asserted on that ratio, and the reason is worth recording
  // rather than quietly omitting. The gnomonic continuation of a panel is
  // tan()-based and diverges as it leaves the panel, so at halo depth 3 -- and
  // especially near a cube corner, where BOTH indices are far outside the panel
  // -- the local extension is legitimately far from the neighbour's grid. The
  // measured worst case sits exactly there (tile 3, i=12, j=9 at im=8: three
  // cells east, on the north edge). An "is it within O(h)" bound is therefore
  // not a property of a correct implementation, and asserting one would fail on
  // correct code. T4 and T5 below establish halo correctness properly, by
  // matching against real cells and edges instead of against an extrapolation.
  //
  // What IS asserted is the direction that cannot have an innocent explanation:
  // exactly zero everywhere would mean the neighbour lookup never fired and
  // corner_owner silently fell through to the local extension.
  okf(n_zero == 0,
      "%.0f halo corners coincide EXACTLY with the local gnomonic extension, "
      "which cannot happen for a real neighbour panel (%.0f checked)",
      (double)n_zero, (double)n_checked);
}

// -----------------------------------------------------------------------------
// T4: THE SCALAR SEAM-CONTINUITY TEST.
//
// Every owned halo CELL of every tile must BE an interior cell of some tile, at
// the same place on the sphere. Found by brute-force geometric search, with no
// reference to the halo index map. If that holds, any function of position is
// automatically single-valued across every seam, which is the property the
// benchmark IC needs.
//
// The test then evaluates the actual analytic scalar the harness uses and
// reports the largest cross-seam disagreement, which is the number the user's
// requirement is phrased in terms of.
// -----------------------------------------------------------------------------
void t4_scalar_seam_continuity(int im, int halo) {
  const int N = im + 1;
  std::printf("T4 SCALAR SEAM CONTINUITY: every owned halo cell is a real "
              "interior cell of some tile\n");
  std::vector<CubeGrid> g;
  g.reserve(6);
  for (int t = 1; t <= 6; ++t) g.emplace_back(im, t, halo);

  // Precompute all interior cell centres, as xyz for a cheap distance test.
  struct Cell { int t, i, j; Vec3 p; };
  std::vector<Cell> interior;
  interior.reserve((size_t)6 * (size_t)im * (size_t)im);
  for (int t = 1; t <= 6; ++t)
    for (int j = 1; j <= im; ++j)
      for (int i = 1; i <= im; ++i)
        interior.push_back(Cell{t, i, j, xyz(g[(size_t)t - 1].center(i, j))});

  const FlowParams f{};
  int n_halo = 0, n_unmatched = 0, n_ambiguous = 0;
  double worst_pos = 0.0, worst_scalar = 0.0;
  int wt = 0, wi = 0, wj = 0;

  for (int t = 1; t <= 6; ++t) {
    for (int j = 1 - halo; j <= im + halo; ++j)
      for (int i = 1 - halo; i <= im + halo; ++i) {
        const bool interior_cell = (i >= 1 && i <= im && j >= 1 && j <= im);
        if (interior_cell) continue;
        if (!g[(size_t)t - 1].center_owned(i, j)) continue; // diagonal block
        ++n_halo;
        const LonLat c = g[(size_t)t - 1].center(i, j);
        const Vec3 cp = xyz(c);
        int matches = 0;
        double best = 1e300;
        const Cell *bc = nullptr;
        for (const Cell &ic : interior) {
          const double d = dist3(cp, ic.p);
          if (d < best) { best = d; bc = &ic; }
          if (d < 1e-12) ++matches;
        }
        if (matches == 0) {
          ++n_unmatched;
          if (n_unmatched <= 5)
            std::printf("  FAIL: tile %d halo cell (%d,%d) matches NO interior "
                        "cell; nearest is tile %d (%d,%d) at chord %.3e\n",
                        t, i, j, bc ? bc->t : -1, bc ? bc->i : -1,
                        bc ? bc->j : -1, best);
          ++g_failures;
          continue;
        }
        if (matches > 1) ++n_ambiguous;
        if (best > worst_pos) { worst_pos = best; wt = t; wi = i; wj = j; }
        // The property that actually matters: the analytic field agrees.
        const LonLat oc = g[(size_t)bc->t - 1].center(bc->i, bc->j);
        const double s1 = flow_shape(c, f), s2 = flow_shape(oc, f);
        const double e = std::fabs(s1 - s2);
        if (e > worst_scalar) worst_scalar = e;
      }
  }
  std::printf("  %d owned halo cells examined; unmatched %d; ambiguous %d\n",
              n_halo, n_unmatched, n_ambiguous);
  std::printf("  worst halo-vs-interior centre chord error %.3e at tile %d "
              "(i=%d,j=%d)\n", worst_pos, wt, wi, wj);
  std::printf("  worst cross-seam disagreement in the analytic scalar "
              "flow_shape = %.3e (tol 1e-13)\n", worst_scalar);
  ok(n_halo > 0, "no owned halo cells were examined at all");
  ok(n_unmatched == 0, "some halo cell is not a real cube cell");
  ok(n_ambiguous == 0, "a halo cell matched more than one interior cell");
  ok(worst_scalar < 1e-13,
     "the analytic scalar disagrees across a seam beyond floating-point "
     "tolerance -- the IC is NOT single-valued");
}

// -----------------------------------------------------------------------------
// T5: THE VECTOR SEAM-CONTINUITY TEST, and an independent re-derivation of the
// seam sign/swap table.
//
// Each owned halo D-grid edge is matched to an interior edge of some tile by
// UNORDERED ENDPOINT PAIR -- so the match is geometric and says nothing about
// which of u or v either side calls it. Then:
//
//   * the two stored components must have equal magnitude (both are the
//     projection of the same physical wind onto the same great-circle arc);
//   * the sign relating them must equal the sign relating the two tangents,
//     which is +1 if the two tiles traverse the arc in the same direction and
//     -1 if opposed;
//   * whether the twin is a u-edge or a v-edge is the COMPONENT SWAP.
//
// Aggregating over each seam re-derives the sign/swap table from geometry alone.
// The table is printed so it can be compared against the independent audit of
// FMS mpp_do_updateV semantics.
// -----------------------------------------------------------------------------
void t5_vector_seam_continuity(int im, int halo) {
  std::printf("T5 VECTOR SEAM CONTINUITY: D-grid components across seams, and "
              "the derived sign/swap table\n");
  std::vector<CubeGrid> g;
  g.reserve(6);
  for (int t = 1; t <= 6; ++t) g.emplace_back(im, t, halo);
  const FlowParams f{};

  // Interior edges: u on (i,j)->(i+1,j) for i in [1,im], j in [1,im+1];
  //                 v on (i,j)->(i,j+1) for i in [1,im+1], j in [1,im].
  struct Edge {
    int t, i, j;
    bool is_u;
    Vec3 a, b;   // endpoint corners
    double comp; // stored component
  };
  std::vector<Edge> interior;
  for (int t = 1; t <= 6; ++t) {
    const CubeGrid &gg = g[(size_t)t - 1];
    for (int j = 1; j <= im + 1; ++j)
      for (int i = 1; i <= im; ++i)
        interior.push_back(Edge{t, i, j, true, xyz(gg.corner(i, j)),
                                xyz(gg.corner(i + 1, j)), dgrid_u(gg, i, j, f)});
    for (int j = 1; j <= im; ++j)
      for (int i = 1; i <= im + 1; ++i)
        interior.push_back(Edge{t, i, j, false, xyz(gg.corner(i, j)),
                                xyz(gg.corner(i, j + 1)), dgrid_v(gg, i, j, f)});
  }

  // side: 0=S 1=E 2=N 3=W. Aggregated derived relation per (tile, side, is_u).
  //
  // The aggregation deliberately EXCLUDES edges near the panel ends. An edge
  // lying on the last couple of indices of a side also lies on a DIFFERENT seam
  // (three tiles meet at a cube corner), so "which side is this edge on" has no
  // single answer there and the geometric search may legitimately return the
  // other seam's twin. Including those edges makes every side report MIXED --
  // which is what the first run of this test did, and it was the aggregation at
  // fault, not the grid: the per-edge checks above passed with 0 unmatched and 0
  // sign-inconsistent at the same time. Away from the ends each edge belongs to
  // exactly one seam and the rule is single-valued.
  struct Rel { int n_pos = 0, n_neg = 0, n_swap = 0, n_same = 0, n = 0; };
  Rel rel[7][4][2];

  int n_edges = 0, n_unmatched = 0, n_signbad = 0;
  double worst_mag = 0.0;

  auto examine = [&](int t, int i, int j, bool is_u) {
    const CubeGrid &gg = g[(size_t)t - 1];
    const int i2 = is_u ? i + 1 : i;
    const int j2 = is_u ? j : j + 1;
    if (!gg.corner_owned(i, j) || !gg.corner_owned(i2, j2)) return;
    // An edge's own interior index range depends on which way it runs: u spans
    // i in [1,im] x j in [1,im+1], v spans i in [1,im+1] x j in [1,im].
    const int i_hi = is_u ? im : im + 1;
    const int j_hi = is_u ? im + 1 : im;
    const bool i_out = (i < 1 || i > i_hi);
    const bool j_out = (j < 1 || j > j_hi);
    if (!i_out && !j_out) return; // interior edge
    // Both out of range puts the edge in a diagonal region, where "which seam"
    // has no single answer. Skip rather than attribute it to an arbitrary side;
    // the corner_owned guard above already rejects the truly nonexistent ones.
    if (i_out && j_out) return;
    const int side = j_out ? (j < 1 ? 0 : 2) : (i < 1 ? 3 : 1); // S,E,N,W = 0..3
    ++n_edges;

    const Vec3 a = xyz(gg.corner(i, j)), b = xyz(gg.corner(i2, j2));
    const double comp = is_u ? dgrid_u(gg, i, j, f) : dgrid_v(gg, i, j, f);

    const Edge *m = nullptr;
    bool reversed = false;
    for (const Edge &e : interior) {
      if (e.t == t) continue;
      if (dist3(a, e.a) < 1e-12 && dist3(b, e.b) < 1e-12) { m = &e; reversed = false; break; }
      if (dist3(a, e.b) < 1e-12 && dist3(b, e.a) < 1e-12) { m = &e; reversed = true; break; }
    }
    if (!m) {
      ++n_unmatched;
      if (n_unmatched <= 5)
        std::printf("  FAIL: tile %d halo %c-edge (%d,%d) matches no interior "
                    "edge on any other tile\n", t, is_u ? 'u' : 'v', i, j);
      ++g_failures;
      return;
    }
    // Magnitudes must agree: same physical wind, same arc.
    const double dm = std::fabs(std::fabs(comp) - std::fabs(m->comp));
    if (dm > worst_mag) worst_mag = dm;
    // The sign relating the stored components must be the sign relating the
    // traversal directions. `reversed` means the twin walks the arc the other
    // way, so its tangent is the negative of ours.
    const int expect = reversed ? -1 : +1;
    const bool consistent =
        std::fabs(comp - (double)expect * m->comp) < 1e-9 * (1.0 + std::fabs(comp));
    if (!consistent) {
      ++n_signbad;
      if (n_signbad <= 5)
        std::printf("  FAIL: tile %d halo %c(%d,%d) = %.9f but twin tile %d "
                    "%c(%d,%d) = %.9f with expected sign %+d\n",
                    t, is_u ? 'u' : 'v', i, j, comp, m->t, m->is_u ? 'u' : 'v',
                    m->i, m->j, m->comp, expect);
      ++g_failures;
    }
    // Along-seam coordinate: i for the S/N sides, j for the E/W sides. Stay well
    // clear of both ends so this edge sits on exactly one seam.
    const int along = (side == 0 || side == 2) ? i : j;
    if (along < 3 || along > im - 1) return;
    Rel &r = rel[t][side][is_u ? 1 : 0];
    ++r.n;
    if (expect > 0) ++r.n_pos; else ++r.n_neg;
    if (m->is_u != is_u) ++r.n_swap; else ++r.n_same;
  };

  for (int t = 1; t <= 6; ++t) {
    for (int j = 1 - halo; j <= im + 1 + halo; ++j)
      for (int i = 1 - halo; i <= im + 1 + halo; ++i) {
        examine(t, i, j, true);
        examine(t, i, j, false);
      }
  }

  std::printf("  %d owned halo edges examined; unmatched %d; sign-inconsistent %d\n",
              n_edges, n_unmatched, n_signbad);
  std::printf("  worst |component| disagreement across a seam = %.3e (tol 1e-9)\n",
              worst_mag);
  ok(n_edges > 0, "no owned halo edges were examined at all");
  ok(n_unmatched == 0, "a halo edge is not a real cube edge");
  ok(n_signbad == 0, "a halo component does not match its twin with the sign "
                     "implied by the two traversal directions");
  ok(worst_mag < 1e-9, "cross-seam component magnitudes disagree");

  // ---------------------------------------------------------------------------
  // The derived table, checked against the INDEPENDENT audit of FMS
  // mpp_do_updateV semantics. FMS itself is absent from the reference checkout
  // (the D-grid seam vector rule lives behind gridtype=DGRID_NE), so that audit
  // derived the twelve-seam aligned/reversed pairing separately. This block
  // re-derives the same classification from pure geometry -- unordered endpoint
  // matching and traversal direction -- and compares. Two independent routes to
  // the same table is the evidence; either alone would just be an assertion.
  //
  // Expected, from the audit:
  //   ALIGNED  (component preserved): 1E-2W, 1S-6N, 2N-3S, 3E-4W, 4N-5S, 5E-6W
  //   REVERSED (u<->v swapped):       1N-3W, 1W-5N, 2E-4S, 2S-6E, 3N-5W, 4E-6S
  // and on a REVERSED seam the audit found the sign is ASYMMETRIC between the
  // two components, e.g. u_n(i,npy+k) = -v_{n+2}(1+k, npx-i) while
  // v_n(i,npy+k) = +u_{n+2}(1+k, npx+1-i). That asymmetry is a sharp, falsifiable
  // prediction: aligned seams must show +1 for BOTH components, reversed seams
  // must show OPPOSITE signs for u and v.
  // ---------------------------------------------------------------------------
  static const char *sname[4] = {"S", "E", "N", "W"};
  // side indices: S=0 E=1 N=2 W=3. true => aligned, false => reversed.
  static const bool expect_aligned[7][4] = {
      /*unused*/ {false, false, false, false},
      /* t1 */ {true, true, false, false},   // S aligned(6N), E aligned(2W), N rev(3W), W rev(5N)
      /* t2 */ {false, false, true, true},   // S rev(6E),     E rev(4S),     N aligned(3S), W aligned(1E)
      /* t3 */ {true, true, false, false},   // S aligned(2N), E aligned(4W), N rev(5W), W rev(1N)
      /* t4 */ {false, false, true, true},   // S rev(2E),     E rev(6S),     N aligned(5S), W aligned(3E)
      /* t5 */ {true, true, false, false},   // S aligned(4N), E aligned(6W), N rev(1W), W rev(3N)
      /* t6 */ {false, false, true, true},   // S rev(4E),     E rev(2S),     N aligned(1S), W aligned(5E)
  };

  std::printf("\n  DERIVED seam table (geometry only; the halo index map is "
              "never consulted).\n"
              "  'swap' = the twin tile stores the OTHER component. 'sign' = the "
              "factor\n  relating the two stored values, set by whether the two "
              "tiles traverse\n  the shared arc in the same direction.\n\n");
  std::printf("  %-5s %-5s %-4s %-12s %-6s   %-10s %s\n", "tile", "side", "cmp",
              "component", "sign", "class", "vs audit");
  int n_class = 0;
  for (int t = 1; t <= 6; ++t)
    for (int s = 0; s < 4; ++s) {
      int sign[2] = {0, 0};
      bool swapped[2] = {false, false};
      bool have[2] = {false, false};
      for (int c = 0; c < 2; ++c) {
        const Rel &r = rel[t][s][c];
        if (r.n == 0) continue;
        const bool pure_swap = (r.n_swap == r.n), pure_same = (r.n_same == r.n);
        const bool pure_pos = (r.n_pos == r.n), pure_neg = (r.n_neg == r.n);
        ok(pure_swap || pure_same,
           "one side+component mixes swapped and preserved twins, so the seam "
           "rule is not single-valued even away from the panel ends");
        ok(pure_pos || pure_neg,
           "one side+component mixes signs, so the vector transform is not "
           "single-valued even away from the panel ends");
        have[c] = pure_swap != pure_same && (pure_pos != pure_neg);
        swapped[c] = pure_swap;
        sign[c] = pure_pos ? +1 : -1;
        std::printf("  %-5d %-5s %-4s %-12s %-6s", t, sname[s], c ? "u" : "v",
                    pure_swap ? "SWAP u<->v" : (pure_same ? "preserved" : "MIXED"),
                    pure_pos ? "+1" : (pure_neg ? "-1" : "MIXED"));
        if (c == 0) std::printf("\n");
      }
      if (!have[0] || !have[1]) { std::printf("\n"); continue; }
      ++n_class;
      // Both components must agree on aligned-vs-reversed: an aligned seam
      // preserves both, a reversed seam swaps both.
      ok(swapped[0] == swapped[1],
         "u and v disagree on whether this seam swaps components, which no "
         "single seam transform can produce");
      const bool aligned = !swapped[0];
      const bool audit_ok = (aligned == expect_aligned[t][s]);
      std::printf("   %-10s %s\n", aligned ? "ALIGNED" : "REVERSED",
                  audit_ok ? "agrees" : "*** DISAGREES ***");
      okf(audit_ok,
          "tile %.0f side %.0f classification contradicts the independent FMS "
          "audit", (double)t, (double)s);
      if (aligned) {
        ok(sign[0] == +1 && sign[1] == +1,
           "an ALIGNED seam must preserve both components with sign +1");
      } else {
        ok(sign[0] == -sign[1],
           "a REVERSED seam must carry OPPOSITE signs on u and v -- the "
           "asymmetry the FMS audit predicted");
      }
    }
  std::printf("  %d of 24 (tile,side) seams classified; all compared against "
              "the audit.\n", n_class);
  ok(n_class == 24, "not every one of the 24 (tile,side) seams was classified");
}

// -----------------------------------------------------------------------------
// T6: an orientation-sensitive invariant that spans seams.
//
// Solid-body rotation is NON-DIVERGENT. Form the discrete C-grid divergence with
// REAL spherical metrics (great-circle face lengths, spherical-excess areas) and
// compare seam-adjacent cells against interior cells. A tangent-orientation,
// sign, swap or stagger error does not merely degrade the divergence, it puts an
// O(1) spike on the seam. The discriminator is therefore the RATIO of the worst
// seam-cell residual to the worst interior-cell residual, not the absolute value.
//
// This uses the real metrics on purpose. The benchmark harness deliberately keeps
// fabricated uniform 100 km metrics (so the 3A.11V damping stability calibration
// stays valid), which means the harness state is NOT in discrete balance. That is
// a separate, documented property; this test is about orientation, not balance.
// -----------------------------------------------------------------------------
void t6_nondivergence(int im, int halo) {
  std::printf("T6 discrete divergence of a non-divergent field: seam cells vs "
              "interior cells\n");
  const FlowParams f{};
  double worst_seam = 0.0, worst_int = 0.0;
  int ws_t = 0, ws_i = 0, ws_j = 0;
  const double scale = f.ubar / f.radius; // the natural |div| scale

  for (int t = 1; t <= 6; ++t) {
    CubeGrid g(im, t, halo);
    for (int j = 1; j <= im; ++j)
      for (int i = 1; i <= im; ++i) {
        // Face lengths: dy on the j-running (west/east) faces, dx on the
        // i-running (south/north) faces.
        const double dy_w = great_circle_dist(g.corner(i, j), g.corner(i, j + 1), f.radius);
        const double dy_e = great_circle_dist(g.corner(i + 1, j), g.corner(i + 1, j + 1), f.radius);
        const double dx_s = great_circle_dist(g.corner(i, j), g.corner(i + 1, j), f.radius);
        const double dx_n = great_circle_dist(g.corner(i, j + 1), g.corner(i + 1, j + 1), f.radius);
        const double area = spherical_quad_area(g.corner(i, j), g.corner(i + 1, j),
                                                g.corner(i + 1, j + 1),
                                                g.corner(i, j + 1), f.radius);
        const double div = (cgrid_uc(g, i + 1, j, f) * dy_e -
                            cgrid_uc(g, i, j, f) * dy_w +
                            cgrid_vc(g, i, j + 1, f) * dx_n -
                            cgrid_vc(g, i, j, f) * dx_s) / area;
        const double r = std::fabs(div) / scale;
        // Exclude the panel-corner region from BOTH populations. cgrid_uc/vc are
        // documented ANALOGUES (centre-to-centre tangents, not FV3's ew/es), and
        // that approximation is worst exactly at a cube corner, where the grid is
        // most skewed. Including corners made the ratio grow with resolution
        // (8.1 at im=8, 15.7 at im=16), i.e. it was measuring the analogue's
        // truncation error rather than seam orientation. Away from corners the
        // comparison isolates what it is meant to.
        const int edge = 2;
        const bool near_corner = (i <= edge || i > im - edge) && (j <= edge || j > im - edge);
        if (near_corner) continue;
        const bool seam = (i <= 1 || i >= im || j <= 1 || j >= im);
        if (seam) {
          if (r > worst_seam) { worst_seam = r; ws_t = t; ws_i = i; ws_j = j; }
        } else {
          if (r > worst_int) worst_int = r;
        }
      }
  }
  const double ratio = (worst_int > 0.0) ? worst_seam / worst_int : 0.0;
  std::printf("  worst |div|/(Ubar/R): seam cells %.6e (tile %d i=%d j=%d), "
              "interior cells %.6e\n", worst_seam, ws_t, ws_i, ws_j, worst_int);
  std::printf("  seam/interior ratio = %.3f\n", ratio);
  ok(worst_int > 0.0, "interior divergence residual is exactly zero, so the "
                      "ratio has no denominator and would mean nothing");
  // DIAGNOSTIC ONLY -- no pass/fail is asserted on that ratio, deliberately.
  //
  // I originally asserted ratio < 20. It passed (8.1 at im=8, 15.7 at im=16 with
  // corners included; 3.0 and 8.3 with corners excluded) but the ratio GROWS with
  // resolution in both variants, roughly 2.8x per doubling. That growth is a
  // property of cgrid_uc/vc being an ANALOGUE rather than a port of FV3's ew/es
  // tangents: grid skewness peaks at panel edges, so the analogue's truncation
  // error is largest there and does not converge at the interior rate. A fixed
  // ratio bound is therefore not a property of a correct implementation -- it
  // would have become a false failure at im=32 -- and choosing a threshold that
  // makes it pass would be tuning, not testing.
  //
  // The load-bearing vector check is T5, which is exact: 0 unmatched edges, 0
  // sign-inconsistent edges, and cross-seam component disagreement at the
  // 1e-14 level over thousands of edges. T6 is retained because a gross
  // orientation error would still show up here as an obvious blow-up, and
  // because the printed numbers document the analogue's actual quality rather
  // than leaving it as a claim in a comment.
}

} // namespace

int main() {
  std::printf("=== Milestone 3A.11V Task 1: cubed-sphere seam continuity ===\n");
  std::printf("HOST-SIDE GEOMETRY TEST. Validates that the benchmark's initial\n"
              "state is single-valued on the cubed sphere. Does NOT validate the\n"
              "CUDA halo exchange, and is NOT a numerical validation against\n"
              "authoritative Fortran output.\n\n");

  const int halo = 3;
  for (int im : {8, 16}) {
    std::printf("---------------- im = %d (npx = %d) ----------------\n", im, im + 1);
    t1_primitives(im);
    t2_fv3_seam_copies(im);
    t3_halo_is_neighbour(im, halo);
    t4_scalar_seam_continuity(im, halo);
    t5_vector_seam_continuity(im, halo);
    t6_nondivergence(im, halo);
    std::printf("\n");
  }

  std::printf("=== cube seam continuity: %s (%d failure%s) ===\n",
              g_failures == 0 ? "PASS" : "FAIL", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
