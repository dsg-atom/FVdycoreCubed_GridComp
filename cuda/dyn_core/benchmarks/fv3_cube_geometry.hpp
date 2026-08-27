// -*- c++ -*-
// =============================================================================
// Milestone 3A.11V Task 1: globally single-valued cubed-sphere geometry and
// analytic initial fields for the CUDA benchmark harness.
//
// PURPOSE
// -------
// The benchmark harness seeded its horizontal structure from TILE-LOCAL array
// indices -- `0.02*sin(0.35*i)*cos(0.35*j)` for the scalars, `0.29*i/0.23*j`
// for the winds, `0.01*i/0.01*j` for grid_lonlat -- with the SAME pattern on
// all six tiles. That is not a single-valued field on a cubed sphere: the
// value in tile T's ring-1 halo is not the value the neighbour owns at the
// cell that halo shadows, so the state carries a genuine O(1) discontinuity
// across all twelve panel seams. The production halo exchange then correctly
// transports the neighbour's differing value inward and the resulting cliff is
// real forcing. It was measured: at WIND_SCALE=0 the ring-1 surface pkc
// overshoot was +4182.3 Pa on tiles 1/3/5 and +1211.4 Pa on tiles 2/4/6, and
// the ratio matched r^gama - 1 to 0.069%.
//
// This module removes that forcing at the source by building the ACTUAL
// gnomonic cubed-sphere geometry and evaluating every horizontal field as a
// smooth function of position on the sphere. A field defined that way is
// single-valued by construction: tile T's halo corner IS the neighbour's
// corner (copied, per FV3), so the field evaluated there IS the neighbour's
// value, to round-off.
//
// AUTHORITATIVE SOURCE OF TRUTH
// -----------------------------
// /discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere
// at ca30e26305fda43831095334c02a3f5e0b8c0c45, tag geos/v3.0.0. This is the
// GEOS-ESM fork, NOT upstream GFDL, and it is treated as THE authority here.
// Referenced below as REF/. Specific provenance:
//
//   REF/model/fv_grid_utils.F90:1296-1317   grid_type==0 -> gnomonic_ed (the
//                                           default; equiDISTANT, not equiangular)
//   REF/model/fv_grid_utils.F90:1329-1416   gnomonic_ed, the global algorithm
//   REF/model/fv_grid_utils.F90:3462-3528   gnomonic_grids_local /
//                                           get_gnomonic_ed_coords -- the CLOSED
//                                           FORM this file implements
//   REF/model/fv_grid_tools.F90:685-698     init_grid: gnomonic_grids + mirror_grid
//   REF/model/fv_grid_tools.F90:2309-2439   mirror_grid: the tile 2..6 maps
//   REF/model/fv_grid_tools.F90:720-736     the 13 explicit seam copies
//   REF/model/fv_grid_utils.F90:1642-1668   latlon2xyz (right-handed)
//   REF/model/fv_grid_utils.F90:1738-1776   cart_to_latlon
//   REF/model/fv_grid_utils.F90:1845-1873   get_unit_vect2
//   REF/model/fv_grid_utils.F90:3356-3370   get_latlon_vector
//   REF/model/fv_grid_utils.F90:2684-2709   cell_center2
//   REF/tools/test_cases.F90:485-521        init_winds defOnGrid==5, THE
//                                           authoritative D-grid wind projection
//   REF/tools/test_cases.F90:1013-1015      Williamson-2 balanced height
//
// PRECISION CAVEAT -- READ BEFORE CLAIMING AGREEMENT WITH FORTRAN
// ---------------------------------------------------------------
// FV3 builds grid geometry in `f_p = selected_real_kind(20)`, i.e. QUAD
// precision, because NO_QUAD_PRECISION is not defined in this build
// configuration (REF/model/fv_grid_utils.F90:102; the macro appears only in
// REF/FVdycore_arch.mk:13 under `ifeq ($(ESMA_FC), pgfortran)`). This module
// uses `double` throughout. Consequences:
//
//   - Agreement with FV3's grid to ~1e-16 relative is expected.
//   - BITWISE agreement is NOT claimed and is NOT achievable without
//     __float128 in latlon2xyz / cart_to_latlon / mid_pt3_cart /
//     normalize_vect / inner_prod.
//   - FV3 additionally orders the cell_center2 corner sum through
//     `sorted_inta` (REF/tools/sorted_index.F90:21-27) specifically so the
//     4-point sum is decomposition-independent and bitwise reproducible. This
//     module uses the plain (i,j),(i+1,j),(i+1,j+1),(i,j+1) order. That is
//     adequate for a benchmark IC and inadequate for a bitwise comparison.
//
// This is benchmark/test infrastructure, not a numerical-validation reference.
// It is deliberately NOT part of the production fv3_dyn_core_gpu library.
//
// WHAT THIS MODULE DOES *NOT* DO
// ------------------------------
// It does not supply the grid METRICS (dx, dy, area, sina/cosa, dxc/dyc, the
// 9-point cos_sg/sin_sg supergrid, edge_vect_*). The harness keeps its
// fabricated uniform 100 km metrics. So the analytic state below is NOT in
// discrete balance with respect to those metrics, and a smooth O(1) forcing
// remains. That is deliberate and bounded:
//
//   - The 3A.11V del-2n damping stability calibration is expressed in terms of
//     da_min/area, and replacing the metrics would invalidate it.
//   - The forcing that had to be removed for a usable stability benchmark was
//     the SEAM DISCONTINUITY, which is a grid-scale O(1) jump the damping
//     operators amplify. A smooth large-scale imbalance is not that.
//
// Do not read "globally consistent IC" as "balanced IC". Those are different
// claims and only the first is made here.
// =============================================================================
#ifndef FV3_CUBE_GEOMETRY_HPP
#define FV3_CUBE_GEOMETRY_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace fv3 {
namespace bench {
namespace cube {

// -----------------------------------------------------------------------------
// Primitives. Each mirrors one authoritative Fortran routine; the citation is
// on the definition in the .cpp.
// -----------------------------------------------------------------------------
struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;
};

struct LonLat {
  double lon = 0.0, lat = 0.0; // radians
};

Vec3 latlon2xyz(LonLat p);
LonLat cart_to_latlon(Vec3 p); // normalizes internally, like the Fortran
Vec3 vect_cross(Vec3 a, Vec3 b);
Vec3 normalize_vect(Vec3 a);
Vec3 mid_pt3_cart(Vec3 a, Vec3 b);
LonLat mid_pt_sphere(LonLat a, LonLat b);
double inner_prod(Vec3 a, Vec3 b);

// Unit tangent at the great-circle midpoint of p1->p2. REF get_unit_vect2.
Vec3 get_unit_vect2(LonLat p1, LonLat p2);
// Local east/north unit vectors at pp. REF get_latlon_vector.
void get_latlon_vector(LonLat pp, Vec3 *elon, Vec3 *elat);

// Cell centre of the four corners, as a centroid of unit vectors renormalized
// to the sphere. REF cell_center2. NOT a great-circle midpoint construction.
LonLat cell_center2(LonLat q1, LonLat q2, LonLat q3, LonLat q4);

// REF/model/fv_grid_utils.F90:2037-2059.
double great_circle_dist(LonLat q1, LonLat q2, double radius);

// Spherical excess area of the quadrilateral q1-q2-q3-q4 (traversed in order),
// times radius^2. Used ONLY by the seam test, to form a real discrete
// divergence; the benchmark harness keeps its fabricated uniform metrics. FV3's
// production area comes from get_area / grid_area, which this does not port.
double spherical_quad_area(LonLat q1, LonLat q2, LonLat q3, LonLat q4,
                           double radius);

// -----------------------------------------------------------------------------
// Tile-1 gnomonic-equidistant corner, closed form, and the tile 2..6 maps.
//
// `im` is the number of CELLS per tile edge, so corner indices run 1..im+1 and
// npx == im+1 (REF/model/fv_arrays.F90:674: "Number of grid corners in the
// x-direction on one tile").
//
// The rotations are PROPER (det = +1) despite the Fortran routine being named
// `mirror_grid`: tiles 2..6 are obtained from tile 1 by pure rotation, so every
// tile's local (i,j) frame has the same chirality relative to the outward
// normal. See the .cpp for the matrices and their derivation.
// -----------------------------------------------------------------------------
Vec3 gnomonic_ed_corner_xyz(int i, int j, int im); // tile 1, no shift
Vec3 tile_rotate(int tile, Vec3 p);                // tile in [1,6]

// Tile `tile` corner (i,j) as lon/lat, for i,j in [1, im+1]. Applies the
// production 10-degree westward shift when `shift` is true (REF
// fv_grid_tools.F90:699-718, shift_fac = 18).
LonLat global_corner(int tile, int i, int j, int im, bool shift);

// Cyclic tile arithmetic on [1,6]: tile_wrap(n, m) == ((n-1+m) mod 6) + 1.
int tile_wrap(int n, int m);

// Where a halo corner of `tile` at (i,j) actually lives. Returns false for the
// four DIAGONAL ghost blocks, which have no owner anywhere on the cube because
// a cube corner joins only three panels (FV3 poisons these with big_number and
// says so: REF/model/fv_grid_utils.F90:598-600, "Force the model to fail if
// incorrect corner values are to be used").
struct CornerOwner {
  int tile = 0, i = 0, j = 0;
};
bool corner_owner(int tile, int i, int j, int im, CornerOwner *out);

// -----------------------------------------------------------------------------
// One tile's corner and centre coordinates over the harness's full halo box.
//
// Index convention matches the harness exactly: arrays are addressed by the
// FV3 index, i.e. valid i runs [1-halo, im+1+halo+1]. Storage is dense over
// that range; use corner()/center() and let them do the offset.
// -----------------------------------------------------------------------------
class CubeGrid {
 public:
  // `pad` extends the stored corner range one beyond the halo so the outermost
  // halo CELL still has all four of its corners.
  CubeGrid(int im, int tile, int halo, bool shift = true);

  int im() const { return im_; }
  int npx() const { return im_ + 1; }
  int tile() const { return tile_; }
  int halo() const { return halo_; }

  // Corner (i,j), FV3 indexing, i,j in [1-halo, im+2+halo].
  LonLat corner(int i, int j) const;
  // True when this corner is a real cube point rather than a diagonal-block
  // convention fill.
  bool corner_owned(int i, int j) const;

  // A-grid cell centre of cell (i,j), i,j in [1-halo, im+1+halo].
  LonLat center(int i, int j) const;
  bool center_owned(int i, int j) const;

 private:
  size_t idx_(int i, int j) const;
  int im_, tile_, halo_, lo_, n_;
  bool shift_;
  std::vector<LonLat> corner_;
  std::vector<char> owned_;
};

// -----------------------------------------------------------------------------
// Analytic global fields. Every one is a smooth single-valued function on the
// sphere, so it is seam-consistent by construction.
// -----------------------------------------------------------------------------
struct FlowParams {
  double ubar = 20.0;                  // m/s solid-body speed
  double alpha = 0.7853981633974483;   // rotation tilt, pi/4
  double radius = 6371.0e3;            // m
  double omega = 7.292e-5;             // s^-1
};

// Solid-body rotation, REF/tools/test_cases.F90:496-499 / 512-515.
//   u_lon =  Ubar*( cos(lat)*cos(alpha) + sin(lat)*cos(lon)*sin(alpha) )
//   u_lat = -Ubar*  sin(lon)*sin(alpha)
void solid_body_wind(LonLat p, const FlowParams &f, double *u_lon, double *u_lat);

// The dimensionless shape shared by the Williamson-2 height, the Coriolis
// parameter and the streamfunction (REF/tools/test_cases.F90:780-787,
// 1013-1015, 346-347):
//   S(lon,lat) = -cos(lon)*cos(lat)*sin(alpha) + sin(lat)*cos(alpha),  S in [-1,1]
// Used here as the horizontal shape for every scalar perturbation, replacing
// the tile-local sin(0.35*i)*cos(0.35*j).
double flow_shape(LonLat p, const FlowParams &f);

// Project the solid-body wind onto the D-grid components, exactly as
// REF/tools/test_cases.F90:485-521 does it: evaluate the analytic lat-lon wind
// AT THE GREAT-CIRCLE MIDPOINT of the same two corners whose arc defines the
// tangent, take (ex,ey) at that same point, and store
// utmp*(e.ex) + vtmp*(e.ey). No dx/dy scaling -- the stored value is a physical
// m/s component along a UNIT tangent.
//
//   u(i,j): tangent e1 from corner (i,j) -> (i+1,j)   [the i-running edge]
//   v(i,j): tangent e2 from corner (i,j) -> (i,j+1)   [the j-running edge]
double dgrid_u(const CubeGrid &g, int i, int j, const FlowParams &f);
double dgrid_v(const CubeGrid &g, int i, int j, const FlowParams &f);

// The same projection onto the C-grid stagger. On a C-grid uc sits on the
// j-running (west/east) face and vc on the i-running (south/north) face, i.e.
// the transposed stagger from the D-grid, so uc reuses the v edge geometry's
// LOCATION with the i-direction tangent and vice versa. See the .cpp.
double cgrid_uc(const CubeGrid &g, int i, int j, const FlowParams &f);
double cgrid_vc(const CubeGrid &g, int i, int j, const FlowParams &f);

// A-grid winds at the cell centre, projected on the local (ec1,ec2)-analogue.
// For the benchmark these only need to be finite, smooth and global; they are
// d_sw inputs that S1 overwrites. Uses the cell-centre east/north basis.
void agrid_wind(const CubeGrid &g, int i, int j, const FlowParams &f,
                double *ua, double *va);

// Analytic relative vorticity of solid_body_wind. INFERENCE (standard result,
// not quoted from the reference tree):
//   zeta = (2*Ubar/radius) * S(lon,lat)
// Used by the seam test as an orientation-sensitive invariant.
double analytic_vorticity(LonLat p, const FlowParams &f);

} // namespace cube
} // namespace bench
} // namespace fv3

#endif // FV3_CUBE_GEOMETRY_HPP
