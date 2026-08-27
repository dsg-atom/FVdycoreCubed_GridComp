// =============================================================================
// Milestone 3A.11V Task 1 -- implementation. See fv3_cube_geometry.hpp for the
// purpose, the authoritative citations, and the precision caveat.
//
// numerical semantics changed: NO for the production CUDA path. Nothing here is
// compiled into fv3_dyn_core_gpu; this is benchmark/test infrastructure that
// replaces one synthetic initial state with another. It DOES change the
// benchmark's initial state, deliberately and by design.
// =============================================================================
#include "fv3_cube_geometry.hpp"

#include <cmath>
#include <cstdio>

namespace fv3 {
namespace bench {
namespace cube {

namespace {

constexpr double kPi = 3.14159265358979323846;
// REF/model/fv_grid_utils.F90:1751 -- `esl = 1.d-10`, the pole cutoff below
// which cart_to_latlon declares lon == 0 rather than trusting atan2.
constexpr double kEsl = 1.0e-10;
// REF/model/fv_arrays.F90:561 -- shift_fac = 18., i.e. pi/18 = 10 degrees
// westward, "Shift the corner away from Japan" (fv_grid_tools.F90:699-718).
constexpr double kShiftFac = 18.0;

double wrap_pi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

} // namespace

// REF/model/fv_grid_utils.F90:1642-1668. Right-handed and unconditional -- there
// is no RIGHT_HAND ifdef in this routine, unlike mirror_grid's embedding.
Vec3 latlon2xyz(LonLat p) {
  return Vec3{std::cos(p.lat) * std::cos(p.lon),
              std::cos(p.lat) * std::sin(p.lon),
              std::sin(p.lat)};
}

// REF/model/fv_grid_utils.F90:1738-1776. The Fortran normalizes `q` IN PLACE and
// returns it normalized; that side effect is irrelevant to a by-value port but
// is worth knowing when reading the Fortran call sites.
LonLat cart_to_latlon(Vec3 p) {
  const double dist = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  p.x /= dist;
  p.y /= dist;
  p.z /= dist;
  LonLat o{};
  if (std::fabs(p.x) + std::fabs(p.y) < kEsl) {
    o.lon = 0.0;
  } else {
    o.lon = std::atan2(p.y, p.x); // [-pi, pi]
    if (o.lon < 0.0) o.lon += 2.0 * kPi;
  }
  o.lat = std::asin(p.z);
  return o;
}

Vec3 vect_cross(Vec3 a, Vec3 b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}

Vec3 normalize_vect(Vec3 a) {
  const double d = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  return Vec3{a.x / d, a.y / d, a.z / d};
}

double inner_prod(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// REF/model/fv_grid_utils.F90:1993-2019 -- e = (p1+p2)/|p1+p2|.
Vec3 mid_pt3_cart(Vec3 a, Vec3 b) {
  return normalize_vect(Vec3{a.x + b.x, a.y + b.y, a.z + b.z});
}

// REF/model/fv_grid_utils.F90:1978-1989 -- latlon -> xyz -> mid_pt3_cart -> latlon.
LonLat mid_pt_sphere(LonLat a, LonLat b) {
  return cart_to_latlon(mid_pt3_cart(latlon2xyz(a), latlon2xyz(b)));
}

// REF/model/fv_grid_utils.F90:1845-1873, get_unit_vect2, verbatim structure:
//     call mid_pt3_cart(p1, p2, pc)
//     call vect_cross(p3, p2, p1)
//     call vect_cross(uc, pc, p3)
//     call normalize_vect(uc)
// Note the argument order of the first cross: p2 x p1, NOT p1 x p2. Reversing it
// flips the returned tangent, which is exactly the kind of sign error that shows
// up only at a seam.
Vec3 get_unit_vect2(LonLat p1, LonLat p2) {
  const Vec3 a = latlon2xyz(p1);
  const Vec3 b = latlon2xyz(p2);
  const Vec3 pc = mid_pt3_cart(a, b);
  const Vec3 p3 = vect_cross(b, a);
  return normalize_vect(vect_cross(pc, p3));
}

// REF/model/fv_grid_utils.F90:3356-3370. The commented-out left-hand alternative
// there has elat(3) = -COS(pp(2)); the active RIGHT_HAND branch is +COS.
void get_latlon_vector(LonLat pp, Vec3 *elon, Vec3 *elat) {
  elon->x = -std::sin(pp.lon);
  elon->y = std::cos(pp.lon);
  elon->z = 0.0;
  elat->x = -std::sin(pp.lat) * std::cos(pp.lon);
  elat->y = -std::sin(pp.lat) * std::sin(pp.lon);
  elat->z = std::cos(pp.lat);
}

// REF/model/fv_grid_utils.F90:2684-2709. A 4-point sum of UNIT corner vectors,
// renormalized. Not a great-circle-midpoint construction, and not an average of
// lon/lat (which would be wrong at the dateline).
//
// FV3 orders the four corners through sorted_inta (REF/tools/sorted_index.F90:
// 21-27) purely so the sum is bitwise decomposition-independent. That ordering
// is NOT reproduced here; see the header's precision caveat.
LonLat cell_center2(LonLat q1, LonLat q2, LonLat q3, LonLat q4) {
  const Vec3 p1 = latlon2xyz(q1), p2 = latlon2xyz(q2);
  const Vec3 p3 = latlon2xyz(q3), p4 = latlon2xyz(q4);
  return cart_to_latlon(Vec3{p1.x + p2.x + p3.x + p4.x,
                             p1.y + p2.y + p3.y + p4.y,
                             p1.z + p2.z + p3.z + p4.z});
}

// REF/model/fv_grid_utils.F90:2037-2059:
//   beta = 2*asin( sqrt( sin((lat1-lat2)/2)^2
//                      + cos(lat1)*cos(lat2)*sin((lon1-lon2)/2)^2 ) )
// The haversine form, chosen over acos(dot) because it stays accurate for small
// separations -- which is the regime a grid spacing lives in.
double great_circle_dist(LonLat q1, LonLat q2, double radius) {
  const double dlat = 0.5 * (q1.lat - q2.lat);
  const double dlon = 0.5 * (q1.lon - q2.lon);
  const double h = std::sin(dlat) * std::sin(dlat) +
                   std::cos(q1.lat) * std::cos(q2.lat) * std::sin(dlon) *
                       std::sin(dlon);
  return 2.0 * std::asin(std::sqrt(h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h))) * radius;
}

// Girard's theorem on two triangles: for a spherical polygon the area is
// (sum of interior angles) - (n-2)*pi, times radius^2. Written as two triangles
// rather than a direct quad sum so a non-planar quad still gives the area of the
// surface actually bounded by the four great-circle arcs.
double spherical_quad_area(LonLat q1, LonLat q2, LonLat q3, LonLat q4,
                           double radius) {
  auto tri = [](Vec3 a, Vec3 b, Vec3 c) {
    // Interior angle at `a` between arcs a->b and a->c: the angle between the
    // two arc tangents at a, obtained by projecting b and c off a.
    auto ang = [](Vec3 v, Vec3 p, Vec3 q) {
      const Vec3 tp = normalize_vect(
          Vec3{p.x - inner_prod(p, v) * v.x, p.y - inner_prod(p, v) * v.y,
               p.z - inner_prod(p, v) * v.z});
      const Vec3 tq = normalize_vect(
          Vec3{q.x - inner_prod(q, v) * v.x, q.y - inner_prod(q, v) * v.y,
               q.z - inner_prod(q, v) * v.z});
      double d = inner_prod(tp, tq);
      if (d > 1.0) d = 1.0;
      if (d < -1.0) d = -1.0;
      return std::acos(d);
    };
    return ang(a, b, c) + ang(b, c, a) + ang(c, a, b) - kPi;
  };
  const Vec3 p1 = latlon2xyz(q1), p2 = latlon2xyz(q2);
  const Vec3 p3 = latlon2xyz(q3), p4 = latlon2xyz(q4);
  return (tri(p1, p2, p3) + tri(p1, p3, p4)) * radius * radius;
}

// -----------------------------------------------------------------------------
// Tile-1 corners, closed form.
//
// REF/model/fv_grid_utils.F90:3491-3502 (get_gnomonic_ed_coords):
//     rsq3 = 1./sqrt(3.) ;  alpha = asin( rsq3 ) ;  dely = alpha / im
//     beta = -dely * (im + 2 - 2*i) ;  b(i) = tan(beta)*cos(alpha)
//     pp(:,i,j) = [-rsq3, -b(i), b(j)]
//
// This is the LOCAL closed form. The DEFAULT production path is the global
// gnomonic_ed + symm_ed + (lon -= pi) sequence (fv_grid_utils.F90:1329-1416,
// 1592-1631, 1296-1317), because compute_coords_locally defaults to .false.
// (fv_arrays.F90:1053). The audit that produced this file ported BOTH to Python
// and compared them at im = 4, 6, 12, 48: max difference 2.8e-16 .. 8.9e-16 rad.
// So the closed form is used here as an equivalent, not as a different grid.
//
// Note the closed form is a well-defined analytic function of i and j for i,j
// OUTSIDE [1, im+1]. That extension is the gnomonic continuation of tile 1's own
// face plane and is NOT the neighbouring panel's grid -- the two are provably
// not parallel. So it must never be used to fill a cross-halo corner; see
// corner_owner. It is used only as the convention fill in the four diagonal
// ghost blocks, which have no owner at all.
//
// FORK BUG, flagged and worked around by construction: the Fortran at :3491-3502
// reuses ONE array `b` across both the i and j loops, which is harmless only
// because npx == npy. This port takes i and j as independent arguments and so
// does not inherit the hazard.
// -----------------------------------------------------------------------------
Vec3 gnomonic_ed_corner_xyz(int i, int j, int im) {
  const double rsq3 = 1.0 / std::sqrt(3.0);
  const double alpha = std::asin(rsq3);
  const double dely = alpha / (double)im;
  const double bi = std::tan(-dely * (double)(im + 2 - 2 * i)) * std::cos(alpha);
  const double bj = std::tan(-dely * (double)(im + 2 - 2 * j)) * std::cos(alpha);
  return normalize_vect(Vec3{-rsq3, -bi, bj});
}

// -----------------------------------------------------------------------------
// Tiles 2..6.
//
// The Fortran (REF/model/fv_grid_tools.F90:2309-2439, mirror_grid) applies
// rot_3d(axis, angle) with `convert` set, which routes through
// spherical_to_cartesian / cartesian_to_spherical. Because RIGHT_HAND is NOT
// defined in this build, that embedding is LEFT-handed
// (fv_grid_tools.F90:1942-1946: z = -r*sin(lat)) while latlon2xyz above is
// unconditionally right-handed. Reimplementing rot_3d directly would inherit
// that mismatch.
//
// The audit instead conjugated the coded rot_3d matrices with F = diag(1,1,-1)
// to undo the left-handed embedding, giving the per-tile maps in RIGHT-handed
// xyz below, and verified them against a verbatim mirror_grid port to a max
// absolute error of 1.915e-15. Every one has det = +1: despite the routine's
// name, tiles 2..6 are PURE ROTATIONS of tile 1, with no reflection, so all six
// tiles have the same (i,j) chirality relative to the outward normal.
//
// MARKED DERIVED. This is the audit's derivation plus numerical verification,
// not a transcription of the Fortran matrices.
//
// Tile centres before the 10-degree shift: 1 (0,0); 2 (90,0); 3 NORTH pole;
// 4 (180,0); 5 (-90,0); 6 SOUTH pole. Tile 1 spans lon [-45,45], lat +-asin(1/sqrt 3)
// = +-35.26438968275466 degrees.
// -----------------------------------------------------------------------------
Vec3 tile_rotate(int tile, Vec3 p) {
  switch (tile) {
    case 1: return Vec3{ p.x,  p.y,  p.z};
    case 2: return Vec3{-p.y,  p.x,  p.z};
    case 3: return Vec3{-p.y, -p.z,  p.x};
    case 4: return Vec3{-p.x, -p.z, -p.y};
    case 5: return Vec3{ p.z, -p.x, -p.y};
    case 6: return Vec3{ p.z,  p.y, -p.x};
    default: return p;
  }
}

LonLat global_corner(int tile, int i, int j, int im, bool shift) {
  Vec3 p = gnomonic_ed_corner_xyz(i, j, im);
  // grid_type < 3 => lon -= pi (REF/model/fv_grid_utils.F90:1314-1316).
  //
  // THE ORDER HERE IS LOAD-BEARING. The Fortran applies `lon -= pi` while
  // generating TILE 1 only, and mirror_grid rotates tiles 2..6 from the
  // ALREADY-SHIFTED tile 1 (REF/tools/fv_grid_tools.F90:686-699: gnomonic_grids
  // fills grid_global(:,:,:,1), then `call mirror_grid`, whose own comment reads
  // "mirror_grid assumes that the tile=1 is centered on equator and greenwich
  // meridian Lon[-pi,pi]").
  //
  // `lon -= pi` is itself a rotation -- (x,y,z) -> (-x,-y,z) about the z-axis --
  // and it does NOT commute with the four tile maps that mix z into x or y
  // (tiles 3,4,5,6). Doing it after tile_rotate leaves tiles 1 and 2 correct and
  // puts the other four in the wrong place. That is not a hypothetical: it is
  // what the first run of the seam test reported, and the pass/fail split was
  // exactly the pure-z-rotation seams (1-2, 3-4, 5-6 closed; all nine copies
  // involving a z-mixing tile failed, by up to a full antipode). Keeping the
  // rotation in Cartesian form here, before tile_rotate, is what fixes it.
  p = Vec3{-p.x, -p.y, p.z};
  LonLat o = cart_to_latlon(tile_rotate(tile, p));
  // The production shift, guarded in the Fortran by
  // `.not. do_schmidt .and. abs(shift_fac) > 1.E-4`; do_schmidt defaults false
  // and shift_fac = 18. Every field in this module is a function of cos(lon)
  // and sin(lon), so the shift is a rigid rotation of the whole setup and
  // changes no seam property -- it is applied for fidelity, not necessity.
  if (shift) o.lon -= kPi / kShiftFac;
  o.lon = wrap_pi(o.lon);
  return o;
}

int tile_wrap(int n, int m) {
  int r = (n - 1 + m) % 6;
  if (r < 0) r += 6;
  return r + 1;
}

// -----------------------------------------------------------------------------
// Halo corner ownership.
//
// DERIVED from FV3's own 13 explicit seam copies, REF/model/fv_grid_tools.F90:
// 720-736, which the audit verified against a verbatim mirror_grid port with
// 0/9 mismatches on every one of the 13. Those statements fix the k == 0 seam
// line; extending them to halo depth k >= 1 is the step done here.
//
// Worked example, tile 1 NORTH (a reversed seam). The Fortran statement is
//     grid_global(1,1:npy,:,3) = grid_global(npx:1:-1,npy,:,1)
// i.e. G_3(1,j) = G_1(npx+1-j, npy). Inverting: tile 1's row npy at index i
// corresponds to tile 3's column 1 at j = npx+1-i. Tile 1's halo row npy+k is
// therefore tile 3's column 1+k, giving
//     G_1(i, npy+k) = G_3(1+k, npx+1-i).
// The other eleven follow the same way. The result is the general form:
//
//   ODD tile n (1,3,5):
//     E  aligned  -> (n+1)  G_n(N+k, j)  = G_{n+1}(1+k,   j)
//     S  aligned  -> (n+5)  G_n(i, 1-k)  = G_{n+5}(i,     N-k)
//     N  reversed -> (n+2)  G_n(i, N+k)  = G_{n+2}(1+k,   N+1-i)
//     W  reversed -> (n+4)  G_n(1-k, j)  = G_{n+4}(N+1-j, N-k)
//
//   EVEN tile n (2,4,6):
//     W  aligned  -> (n-1)  G_n(1-k, j)  = G_{n-1}(N-k,   j)
//     N  aligned  -> (n+1)  G_n(i, N+k)  = G_{n+1}(i,     1+k)
//     E  reversed -> (n+2)  G_n(N+k, j)  = G_{n+2}(N+1-j, 1+k)
//     S  reversed -> (n+4)  G_n(i, 1-k)  = G_{n+4}(N-k,   N+1-i)
//
// with N == npx == im+1 and tile arithmetic cyclic on [1,6]. Enumerating these
// over n = 1..6 reproduces exactly the audit's twelve-seam pairing: the aligned
// set {1E-2W, 1S-6N, 2N-3S, 3E-4W, 4N-5S, 5E-6W} and the reversed set
// {1N-3W, 1W-5N, 2E-4S, 2S-6E, 3N-5W, 4E-6S}. That cross-check is what makes
// the extension trustworthy rather than merely plausible, and the seam test
// verifies it again numerically without reference to this table.
//
// The four DIAGONAL blocks return false. Three panels meet at a cube corner, so
// those cells exist on no tile. FV3 poisons them with big_number and states the
// intent outright at REF/model/fv_grid_utils.F90:598-600: "Force the model to
// fail if incorrect corner values are to be used."
// -----------------------------------------------------------------------------
bool corner_owner(int tile, int i, int j, int im, CornerOwner *out) {
  const int N = im + 1;
  const bool odd = (tile % 2) == 1;
  const bool i_in = (i >= 1 && i <= N);
  const bool j_in = (j >= 1 && j <= N);

  if (i_in && j_in) {
    *out = CornerOwner{tile, i, j};
    return true;
  }
  if (i > N && j_in) { // EAST
    const int k = i - N;
    *out = odd ? CornerOwner{tile_wrap(tile, 1), 1 + k, j}
               : CornerOwner{tile_wrap(tile, 2), N + 1 - j, 1 + k};
  } else if (i < 1 && j_in) { // WEST
    const int k = 1 - i;
    *out = odd ? CornerOwner{tile_wrap(tile, 4), N + 1 - j, N - k}
               : CornerOwner{tile_wrap(tile, 5), N - k, j};
  } else if (j > N && i_in) { // NORTH
    const int k = j - N;
    *out = odd ? CornerOwner{tile_wrap(tile, 2), 1 + k, N + 1 - i}
               : CornerOwner{tile_wrap(tile, 1), i, 1 + k};
  } else if (j < 1 && i_in) { // SOUTH
    const int k = 1 - j;
    *out = odd ? CornerOwner{tile_wrap(tile, 5), i, N - k}
               : CornerOwner{tile_wrap(tile, 4), N - k, N + 1 - i};
  } else {
    return false; // diagonal ghost block: no owner anywhere on the cube
  }
  // A halo deeper than the neighbour tile is wide would index outside the
  // owner's compute domain. Report it rather than silently reading a
  // meaningless index -- at the grid sizes this harness uses it cannot happen,
  // and if it ever does the caller needs to know.
  if (out->i < 1 || out->i > N || out->j < 1 || out->j > N) return false;
  return true;
}

// -----------------------------------------------------------------------------
// CubeGrid
// -----------------------------------------------------------------------------
CubeGrid::CubeGrid(int im, int tile, int halo, bool shift)
    : im_(im), tile_(tile), halo_(halo), shift_(shift) {
  // Stored corner range [1-halo, im+2+halo]: one beyond the halo on the high
  // side so the outermost halo CELL still has all four of its corners, and the
  // extra +1 that npx itself carries.
  lo_ = 1 - halo_ - 1;
  const int hi = im_ + 2 + halo_;
  n_ = hi - lo_ + 1;
  corner_.assign((size_t)n_ * (size_t)n_, LonLat{});
  owned_.assign((size_t)n_ * (size_t)n_, 0);
  for (int j = lo_; j <= hi; ++j) {
    for (int i = lo_; i <= hi; ++i) {
      CornerOwner o{};
      const size_t s = idx_(i, j);
      if (corner_owner(tile_, i, j, im_, &o)) {
        corner_[s] = global_corner(o.tile, o.i, o.j, im_, shift_);
        owned_[s] = 1;
      } else {
        // Diagonal ghost block. Convention fill with tile-local gnomonic
        // continuation: finite, smooth, and deterministic. FV3 poisons these
        // instead; this harness must put SOMETHING finite here because it seeds
        // the whole allocated box, and no consumer in this configuration reads a
        // physically meaningful value from them. owned_ stays 0 so the seam test
        // can exclude them.
        corner_[s] = global_corner(tile_, i, j, im_, shift_);
        owned_[s] = 0;
      }
    }
  }
}

size_t CubeGrid::idx_(int i, int j) const {
  int ii = i - lo_, jj = j - lo_;
  if (ii < 0) ii = 0;
  if (jj < 0) jj = 0;
  if (ii >= n_) ii = n_ - 1;
  if (jj >= n_) jj = n_ - 1;
  return (size_t)jj * (size_t)n_ + (size_t)ii;
}

LonLat CubeGrid::corner(int i, int j) const { return corner_[idx_(i, j)]; }
bool CubeGrid::corner_owned(int i, int j) const { return owned_[idx_(i, j)] != 0; }

LonLat CubeGrid::center(int i, int j) const {
  return cell_center2(corner(i, j), corner(i + 1, j), corner(i + 1, j + 1),
                      corner(i, j + 1));
}

bool CubeGrid::center_owned(int i, int j) const {
  return corner_owned(i, j) && corner_owned(i + 1, j) &&
         corner_owned(i + 1, j + 1) && corner_owned(i, j + 1);
}

// -----------------------------------------------------------------------------
// Analytic fields
// -----------------------------------------------------------------------------
void solid_body_wind(LonLat p, const FlowParams &f, double *u_lon, double *u_lat) {
  // REF/tools/test_cases.F90:496-499 and 512-515, verbatim:
  //   utmp =  Ubar * ( COS(pt(2))*COS(alpha) + SIN(pt(2))*COS(pt(1))*SIN(alpha) )
  //   vtmp = -Ubar *   SIN(pt(1))*SIN(alpha)
  *u_lon = f.ubar * (std::cos(p.lat) * std::cos(f.alpha) +
                     std::sin(p.lat) * std::cos(p.lon) * std::sin(f.alpha));
  *u_lat = -f.ubar * std::sin(p.lon) * std::sin(f.alpha);
}

double flow_shape(LonLat p, const FlowParams &f) {
  // REF/tools/test_cases.F90:780-787 / 1013-1015 / 346-347 all share this shape.
  return -std::cos(p.lon) * std::cos(p.lat) * std::sin(f.alpha) +
         std::sin(p.lat) * std::cos(f.alpha);
}

double analytic_vorticity(LonLat p, const FlowParams &f) {
  return (2.0 * f.ubar / f.radius) * flow_shape(p, f);
}

namespace {

// The one projection all four staggered accessors share: evaluate the analytic
// lat-lon wind at `pt`, take the local east/north basis at THE SAME point, and
// return the component along `e`. This co-location is the load-bearing part --
// evaluating the wind at the cell centre while taking the tangent at the face
// midpoint introduces a first-order error that looks exactly like a seam defect.
double project(LonLat pt, Vec3 e, const FlowParams &f) {
  double utmp = 0.0, vtmp = 0.0;
  solid_body_wind(pt, f, &utmp, &vtmp);
  Vec3 ex{}, ey{};
  get_latlon_vector(pt, &ex, &ey);
  return utmp * inner_prod(e, ex) + vtmp * inner_prod(e, ey);
}

} // namespace

// REF/tools/test_cases.F90:509-521. u(i,j) sits on the i-running edge from
// corner (i,j) to (i+1,j) and stores the component along that edge's tangent.
// Array shape confirms the stagger: REF/model/fv_arrays.F90:1462-1463 allocates
// u over (isd:ied, jsd:jed+1) -- the +1 is on j -- and v over (isd:ied+1, jsd:jed).
double dgrid_u(const CubeGrid &g, int i, int j, const FlowParams &f) {
  const LonLat p1 = g.corner(i, j), p2 = g.corner(i + 1, j);
  return project(mid_pt_sphere(p1, p2), get_unit_vect2(p1, p2), f);
}

// REF/tools/test_cases.F90:487-500. v(i,j) sits on the j-running edge from
// corner (i,j) to (i,j+1).
double dgrid_v(const CubeGrid &g, int i, int j, const FlowParams &f) {
  const LonLat p1 = g.corner(i, j), p2 = g.corner(i, j + 1);
  return project(mid_pt_sphere(p1, p2), get_unit_vect2(p1, p2), f);
}

// C-grid stagger, and the reason it is NOT a transcription.
//
// FV3 allocates uc over (isd:ied+1, jsd:jed) and vc over (isd:ied, jsd:jed+1) --
// the transpose of the D-grid -- so uc(i,j) sits at the midpoint of the
// j-running (west) face, the same LOCATION as v(i,j), but carries the
// i-DIRECTION component. The authoritative i-direction tangent at that face is
// gridstruct%ew(:,i,j,1), built at REF/model/fv_grid_utils.F90:332-386 with
// special cases at i==1/npx and j==1/npy, and FV3 does not construct C-grid
// winds analytically at all: init_winds runs dtoa then atoc
// (REF/tools/test_cases.F90:519-521).
//
// This function is therefore an ANALOGUE, not a port: the i-direction tangent is
// taken as the arc joining the two adjacent cell CENTRES, which is single-valued
// on the sphere and smooth across seams. That is adequate and honest here
// because the harness's uc/vc are d_sw INPUTS whose only requirements are
// finite, smooth, globally consistent and non-degenerate -- S1's contravariant
// transform overwrites ut/vt from them, and no correctness claim rests on uc/vc
// matching FV3's atoc output. Do not promote this to a validation reference.
double cgrid_uc(const CubeGrid &g, int i, int j, const FlowParams &f) {
  const LonLat c0 = g.center(i - 1, j), c1 = g.center(i, j);
  const LonLat pt = mid_pt_sphere(g.corner(i, j), g.corner(i, j + 1));
  return project(pt, get_unit_vect2(c0, c1), f);
}

double cgrid_vc(const CubeGrid &g, int i, int j, const FlowParams &f) {
  const LonLat c0 = g.center(i, j - 1), c1 = g.center(i, j);
  const LonLat pt = mid_pt_sphere(g.corner(i, j), g.corner(i + 1, j));
  return project(pt, get_unit_vect2(c0, c1), f);
}

// A-grid winds on an (ec1, ec2) analogue at the cell centre: e1 along the arc
// joining the two i-faces, e2 along the arc joining the two j-faces. FV3's
// authoritative version is get_center_vect (REF/model/fv_grid_utils.F90:
// 1792-1842), which builds the same two directions through cross products with
// the centre vector; the same "analogue not port" caveat as cgrid_uc applies,
// and for the same reason (ua/va are d_sw inputs, not validated outputs).
void agrid_wind(const CubeGrid &g, int i, int j, const FlowParams &f,
                double *ua, double *va) {
  const LonLat w = mid_pt_sphere(g.corner(i, j), g.corner(i, j + 1));
  const LonLat e = mid_pt_sphere(g.corner(i + 1, j), g.corner(i + 1, j + 1));
  const LonLat s = mid_pt_sphere(g.corner(i, j), g.corner(i + 1, j));
  const LonLat n = mid_pt_sphere(g.corner(i, j + 1), g.corner(i + 1, j + 1));
  const LonLat c = g.center(i, j);
  *ua = project(c, get_unit_vect2(w, e), f);
  *va = project(c, get_unit_vect2(s, n), f);
}

} // namespace cube
} // namespace bench
} // namespace fv3
