#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"

#include <cmath>
#include <cstdint>

namespace fv3::gpu::operators {

#if defined(__CUDACC__)
#define FV3_WP_D_HD __host__ __device__
#define FV3_WP_D_INLINE __forceinline__
template <class T> __host__ __device__ inline T fv3_wpd_sin(T x){return sin(x);}
template <class T> __host__ __device__ inline T fv3_wpd_cos(T x){return cos(x);}
template <class T> __host__ __device__ inline T fv3_wpd_sqrt(T x){return sqrt(x);}
#else
#define FV3_WP_D_HD
#define FV3_WP_D_INLINE inline
template <class T> inline T fv3_wpd_sin(T x){return std::sin(x);}
template <class T> inline T fv3_wpd_cos(T x){return std::cos(x);}
template <class T> inline T fv3_wpd_sqrt(T x){return std::sqrt(x);}
#endif

/* Lightweight kernel views are produced from WP-E descriptors.  They do not
 * define storage, precision, ownership, or an ABI. */
template <class T> struct View1 {
  T *data;
  int64_t lo_i;
  int64_t si;
  FV3_WP_D_HD FV3_WP_D_INLINE T &operator()(int64_t i) const {
    return data[(i - lo_i) * si];
  }
};

template <class T> struct View2 {
  T *data;
  int64_t lo_i, lo_j;
  int64_t si, sj;
  FV3_WP_D_HD FV3_WP_D_INLINE T &operator()(int64_t i, int64_t j) const {
    return data[(i - lo_i) * si + (j - lo_j) * sj];
  }
};

template <class T> struct View3 {
  T *data;
  int64_t lo_i, lo_j, lo_k;
  int64_t si, sj, sk;
  FV3_WP_D_HD FV3_WP_D_INLINE T &operator()(int64_t i, int64_t j,
                                             int64_t k) const {
    return data[(i - lo_i) * si + (j - lo_j) * sj + (k - lo_k) * sk];
  }
};

struct HorizontalBounds {
  int64_t is, ie, js, je;
  int64_t npz;
};

enum class PressureVariant { normal_nh, split };
enum class A2BOrder : int { second = 2, fourth = 4 };

struct PressureInputs {
  View3<fv_real> pp;       // perturbation pressure, already on B points
  View3<fv_real> gz;       // geopotential, already on B points
  View3<fv_real> delp_b;   // delp interpolated to B points
  View3<fv_real> pk_b;     // pressure coordinate on B points
  View2<const fv_real> rdx;
  View2<const fv_real> rdy;
};

struct PressureOutputs {
  View3<fv_real> u;
  View3<fv_real> v;
  View3<fv_real> du;
  View3<fv_real> dv;
};

struct PressureConfig {
  PressureVariant variant;
  fv_real dt;
  fv_real beta;
};

/* Exact Fortran loop formulas after a2b_ord{2,4}.  Keeping interpolation as a
 * separate launch preserves the Fortran replacement/ordering boundary. */
void pressure_gradient(const HorizontalBounds &, const PressureInputs &,
                       const PressureOutputs &, const PressureConfig &);

/* nh_p_grad's k==1 branch, dyn_core.F90:1628-1635.  The model top is NOT
 * interpolated: pp is zero there by definition and pk is the constant
 * top_value = ptop**akap, so a2b_ord4 is skipped for that level entirely.
 * Callers that batch the interpolation over all npz+1 levels must apply this
 * afterwards to restore the authoritative top. */
void set_pressure_top(const HorizontalBounds &, fv_real top_value,
                      View3<fv_real> pp, View3<fv_real> pk);

struct OneGradientInputs {
  View3<fv_real> pk_b;
  View3<fv_real> gz_b;
  View3<fv_real> mass_b;
  View2<const fv_real> divg2;
  View2<const fv_real> rdx;
  View2<const fv_real> rdy;
};

void one_gradient_pressure(const HorizontalBounds &, const OneGradientInputs &,
                           View3<fv_real> u, View3<fv_real> v, fv_real dt,
                           fv_real d_ext);
void grad1_pressure_update(const HorizontalBounds &,
                           const OneGradientInputs &, View3<fv_real> u,
                           View3<fv_real> v, View3<fv_real> du,
                           View3<fv_real> dv, fv_real dt, fv_real beta);

void reconstruct_pe_halo(const HorizontalBounds &, fv_real ptop,
                         View3<const fv_real> delp, View3<fv_real> pe);
void reconstruct_pk3_halo(const HorizontalBounds &, fv_real ptop,
                          fv_real akap, View3<const fv_real> delp,
                          View3<fv_real> pk3);
void reconstruct_pln_halo(const HorizontalBounds &, fv_real ptop,
                          View3<const fv_real> delp, View3<fv_real> pln);
void geopotential_from_height(const HorizontalBounds &, int halo_band,
                              fv_real grav, View3<const fv_real> zh,
                              View3<fv_real> gz);

/* Interior/doubly-periodic formulas from a2b_edge.F90.  Global cubed-sphere
 * physical-edge correction is deliberately a distinct integration launch. */
void a2b_ord2_periodic(const HorizontalBounds &, View2<const fv_real> in,
                       View2<fv_real> out);
void a2b_ord4_periodic(const HorizontalBounds &, View2<const fv_real> in,
                       View2<fv_real> out, View2<fv_real> qx,
                       View2<fv_real> qy);

struct Del2Metrics {
  View2<const fv_real> del6_u;
  View2<const fv_real> del6_v;
  View2<const fv_real> rarea;
};

struct Del2SgMetrics {
  View2<const fv_real> dx;
  View2<const fv_real> dy;
  View2<const fv_real> sina_u;
  View2<const fv_real> sina_v;
  View2<const fv_real> rdxc;
  View2<const fv_real> rdyc;
  View2<const fv_real> rarea;
};

struct PhysicalCorners {
  bool sw=false, se=false, ne=false, nw=false;
};

struct PhysicalEdges {
  bool west=false, east=false, south=false, north=false;
};

/* Geometry used only by the authoritative global-cubed-sphere A-to-B path.
 * Edge weights remain real64 even when fv_real is real32. */
struct A2BGeometry {
  View2<const fv_real> dxa;
  View2<const fv_real> dya;
  View3<const fv_real> grid_lonlat;
  View3<const fv_real> agrid_lonlat;
  View1<const accumulator_real> edge_w;
  View1<const accumulator_real> edge_e;
  View1<const accumulator_real> edge_s;
  View1<const accumulator_real> edge_n;
};

void a2b_ord2_global(const HorizontalBounds &, PhysicalEdges,
                     PhysicalCorners, View2<fv_real> input,
                     View2<fv_real> output, const A2BGeometry &,
                     bool replace);
void a2b_ord4_global(const HorizontalBounds &, PhysicalEdges,
                     PhysicalCorners, View2<fv_real> input,
                     View2<fv_real> output, const A2BGeometry &,
                     View2<fv_real> qx, View2<fv_real> qy,
                     View2<fv_real> qxx, View2<fv_real> qyy,
                     bool replace);

void average_del2_corners(const HorizontalBounds &, PhysicalCorners,
                          View3<fv_real> q);

/* The three phases of one del2_cubed pass, split so a caller can interleave the
 * two copy_corners rotations the authoritative routine performs BETWEEN them --
 * dir=1 before the fx loop (dyn_core.F90:2419) and dir=2 between the fx and fy
 * loops (:2437). use_sg selects the #ifdef USE_SG flux form, matching the CUDA
 * launchers' single-entry-point-per-phase shape. */
void del2_zonal_flux(const HorizontalBounds &, int halo_band, bool use_sg,
                     View3<fv_real> q, const Del2Metrics &,
                     const Del2SgMetrics &, View3<fv_real> fx);
void del2_meridional_flux(const HorizontalBounds &, int halo_band, bool use_sg,
                          View3<fv_real> q, const Del2Metrics &,
                          const Del2SgMetrics &, View3<fv_real> fy);
void del2_update(const HorizontalBounds &, int halo_band, accumulator_real cd,
                 View3<fv_real> q, View2<const fv_real> rarea,
                 View3<fv_real> fx, View3<fv_real> fy);

/* One complete del2_cubed pass with NO corner rotation -- retained for the unit
 * tests and any single-tile caller.  Orchestration repeats min(3,nmax) passes,
 * shrinking the written band by one each pass and placing H_HEAT only before
 * pass one. */
void del2_pass(const HorizontalBounds &, int halo_band, accumulator_real cd,
               View3<fv_real> q, const Del2Metrics &, View3<fv_real> fx,
               View3<fv_real> fy);
void del2_pass_sg(const HorizontalBounds &, int halo_band, accumulator_real cd,
                  View3<fv_real> q, const Del2SgMetrics &, View3<fv_real> fx,
                  View3<fv_real> fy);

/* The post-loop conversion in dyn_core: bounded heat source is divided by
 * delp and pkz before being added to potential temperature. */
void apply_diffusive_heating_nh(const HorizontalBounds &, fv_real bdt,
                             fv_real delt_max, fv_real cv_air, fv_real rdg,
                             fv_real akap, bool moist_cappa,
                             int64_t n_con, View3<fv_real> heat_source,
                             View3<const fv_real> delp,
                             View3<const fv_real> delz,
                             View3<const fv_real> cappa,
                             View3<fv_real> pkz, View3<fv_real> pt);

void apply_diffusive_heating_hydrostatic(
    const HorizontalBounds &, fv_real bdt, fv_real delt_max, fv_real cp_air,
    int64_t n_con, View3<const fv_real> heat_source,
    View3<const fv_real> delp, View3<const fv_real> pkz, View3<fv_real> pt);

/* ===========================================================================
 * omega finalization -- the `last_step` diagnostic tail of dyn_core.
 *
 * Authoritative live path for use_old_omega = .true. (fv_arrays.F90:465, not
 * overridden by the fixture namelist), nonhydrostatic, grid_type = 0, not
 * nested:
 *
 *   dyn_core.F90:503-517   pem(i,1,j) = ptop
 *                          pem(i,k+1,j) = pem(i,k,j) + delp(i,j,k)
 *                          -- snapshot taken INSIDE the acoustic loop, under
 *                             `it==n_split .and. end_step`, immediately after
 *                             the it==1 i_pack(1) completion at :496 and BEFORE
 *                             anything in the substep updates delp.
 *   dyn_core.F90:1205-1211 omga(i,j,k) = (pe(i,k+1,j) - pem(i,k+1,j)) * rdt
 *   dyn_core.F90:1218      call adv_pe(ua, va, pem, omga, ...)  -- :1535-1636
 *
 * Only pem levels 2..npz+1 are ever read (both consumers index pem(i,k+1,j)
 * for k = 1..npz), so the workspace below stores exactly those npz levels:
 * pem_view(i, j, k0) holds Fortran pem(i, k0+2, j), k0 = 0..npz-1.  That also
 * makes it directly usable as adv_pe's `pin` for level k0+1 with no shift.
 * ======================================================================== */

/* Grid quantities adv_pe needs.  en1/en2/ec1/ec2 are NOT registered fields:
 * every one of the six unit vectors a cell requires is derived here from that
 * cell's own four B-grid corners of grid_lonlat, which is the authoritative
 * geometry already validated by the a2b_ord4 pressure path.  See
 * fv_grid_utils.F90:308-318 (grid3), :759-772 (en1/en2) and :1792-1841
 * (get_center_vect -> ec1/ec2). */
struct AdvPeMetrics {
  View3<const fv_real> grid_lonlat;  // B-grid corner (lon,lat), component-slowest
  View2<const fv_real> dx, dy, rarea;
};

// fv_grid_utils.F90:1644-1668.
FV3_WP_D_HD FV3_WP_D_INLINE void
adv_pe_latlon2xyz(accumulator_real lon, accumulator_real lat,
                  accumulator_real e[3]) {
  e[0] = fv3_wpd_cos(lat) * fv3_wpd_cos(lon);
  e[1] = fv3_wpd_cos(lat) * fv3_wpd_sin(lon);
  e[2] = fv3_wpd_sin(lat);
}

// fv_grid_utils.F90:1780-1789 -- e = p1 X p2.
FV3_WP_D_HD FV3_WP_D_INLINE void
adv_pe_vect_cross(const accumulator_real p1[3], const accumulator_real p2[3],
                  accumulator_real e[3]) {
  e[0] = p1[1] * p2[2] - p1[2] * p2[1];
  e[1] = p1[2] * p2[0] - p1[0] * p2[2];
  e[2] = p1[0] * p2[1] - p1[1] * p2[0];
}

// fv_grid_utils.F90:1876-1889.
FV3_WP_D_HD FV3_WP_D_INLINE void adv_pe_normalize(accumulator_real e[3]) {
  const accumulator_real pdot =
      fv3_wpd_sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] /= pdot;
  e[1] /= pdot;
  e[2] /= pdot;
}

// fv_grid_utils.F90:1993-2019.
FV3_WP_D_HD FV3_WP_D_INLINE void
adv_pe_mid_pt3_cart(const accumulator_real p1[3], const accumulator_real p2[3],
                    accumulator_real e[3]) {
  accumulator_real e1 = p1[0] + p2[0];
  accumulator_real e2 = p1[1] + p2[1];
  accumulator_real e3 = p1[2] + p2[2];
  const accumulator_real dd = fv3_wpd_sqrt(e1 * e1 + e2 * e2 + e3 * e3);
  e[0] = e1 / dd;
  e[1] = e2 / dd;
  e[2] = e3 / dd;
}

// fv_grid_utils.F90:2712-2728.
FV3_WP_D_HD FV3_WP_D_INLINE void
adv_pe_cell_center3(const accumulator_real p1[3], const accumulator_real p2[3],
                    const accumulator_real p3[3], const accumulator_real p4[3],
                    accumulator_real ec[3]) {
  ec[0] = p1[0] + p2[0] + p3[0] + p4[0];
  ec[1] = p1[1] + p2[1] + p3[1] + p4[1];
  ec[2] = p1[2] + p2[2] + p3[2] + p4[2];
  const accumulator_real dd =
      fv3_wpd_sqrt(ec[0] * ec[0] + ec[1] * ec[1] + ec[2] * ec[2]);
  ec[0] /= dd;
  ec[1] /= dd;
  ec[2] /= dd;
}

/* fv_grid_utils.F90:1792-1841, the #else (non-OLD_VECT) branch, which is the
 * live one: OLD_VECT is absent from the authoritative build's flags.make.  The
 * corner-region zeroing at :1811-1813 covers only the four cells outside the
 * cubed-sphere panel, which adv_pe never visits (it runs is..ie / js..je). */
FV3_WP_D_HD FV3_WP_D_INLINE void
adv_pe_center_vect(const accumulator_real p00[3], const accumulator_real p10[3],
                   const accumulator_real p01[3], const accumulator_real p11[3],
                   accumulator_real e1[3], accumulator_real e2[3]) {
  accumulator_real pc[3], q1[3], q2[3], p3[3];
  adv_pe_cell_center3(p00, p10, p01, p11, pc);
  // e1:
  adv_pe_mid_pt3_cart(p00, p01, q1);
  adv_pe_mid_pt3_cart(p10, p11, q2);
  adv_pe_vect_cross(q2, q1, p3);
  adv_pe_vect_cross(pc, p3, e1);
  adv_pe_normalize(e1);
  // e2:
  adv_pe_mid_pt3_cart(p00, p10, q1);
  adv_pe_mid_pt3_cart(p01, p11, q2);
  adv_pe_vect_cross(q2, q1, p3);
  adv_pe_vect_cross(pc, p3, e2);
  adv_pe_normalize(e2);
}

/* One (i,j,k) cell of adv_pe's body (dyn_core.F90:1583-1636), returning the
 * increment added to om.  `up`/`vp` are the vertically averaged winds of
 * :1583-1596; `pb` is the a2b_ord2 output for this level.
 *
 * The precision ladder mirrors the Fortran exactly, which matters only for R4
 * (in R8 every type below is double): adv_pe's own locals up/vp/v3/pin/pb/grad/
 * pdx/pdy are all plain `real` (= fv_real) while ec1/ec2/en1/en2/dx/dy/rarea are
 * R_GRID, so each mixed product evaluates wide and truncates on store. */
FV3_WP_D_HD FV3_WP_D_INLINE accumulator_real
adv_pe_cell_increment(const AdvPeMetrics &m, View3<fv_real> pb, int64_t kb,
                      int64_t i, int64_t j, fv_real up, fv_real vp) {
  const int64_t c = m.grid_lonlat.lo_k;
  accumulator_real p00[3], p10[3], p01[3], p11[3];
  adv_pe_latlon2xyz(m.grid_lonlat(i, j, c), m.grid_lonlat(i, j, c + 1), p00);
  adv_pe_latlon2xyz(m.grid_lonlat(i + 1, j, c), m.grid_lonlat(i + 1, j, c + 1), p10);
  adv_pe_latlon2xyz(m.grid_lonlat(i, j + 1, c), m.grid_lonlat(i, j + 1, c + 1), p01);
  adv_pe_latlon2xyz(m.grid_lonlat(i + 1, j + 1, c), m.grid_lonlat(i + 1, j + 1, c + 1), p11);

  accumulator_real ec1[3], ec2[3];
  adv_pe_center_vect(p00, p10, p01, p11, ec1, ec2);

  /* fv_grid_utils.F90:761-771:
   *   en1(:,i,j) = normalize(grid3(i,j)   X grid3(i+1,j))
   *   en2(:,i,j) = normalize(grid3(i,j+1) X grid3(i,j))
   * adv_pe differences pdx over j and pdy over i, so this cell needs en1 at
   * (i,j) and (i,j+1) and en2 at (i,j) and (i+1,j). */
  accumulator_real en1_j[3], en1_jp[3], en2_i[3], en2_ip[3];
  adv_pe_vect_cross(p00, p10, en1_j);   adv_pe_normalize(en1_j);
  adv_pe_vect_cross(p01, p11, en1_jp);  adv_pe_normalize(en1_jp);
  adv_pe_vect_cross(p01, p00, en2_i);   adv_pe_normalize(en2_i);
  adv_pe_vect_cross(p11, p10, en2_ip);  adv_pe_normalize(en2_ip);

  // dyn_core.F90:1613/:1620 -- (pb+pb) is formed in fv_real, then widened.
  const fv_real sx_j  = pb(i, j, kb) + pb(i + 1, j, kb);
  const fv_real sx_jp = pb(i, j + 1, kb) + pb(i + 1, j + 1, kb);
  const fv_real sy_i  = pb(i, j, kb) + pb(i, j + 1, kb);
  const fv_real sy_ip = pb(i + 1, j, kb) + pb(i + 1, j + 1, kb);
  const accumulator_real dx_j  = static_cast<accumulator_real>(m.dx(i, j));
  const accumulator_real dx_jp = static_cast<accumulator_real>(m.dx(i, j + 1));
  const accumulator_real dy_i  = static_cast<accumulator_real>(m.dy(i, j));
  const accumulator_real dy_ip = static_cast<accumulator_real>(m.dy(i + 1, j));
  const accumulator_real upa = static_cast<accumulator_real>(up);
  const accumulator_real vpa = static_cast<accumulator_real>(vp);

  fv_real dot = fv_real(0);
  for (int n = 0; n < 3; ++n) {
    // :1600 v3(n,i,j) = up*ec1(n,i,j) + vp*ec2(n,i,j)
    const fv_real v3n = static_cast<fv_real>(upa * ec1[n] + vpa * ec2[n]);
    const fv_real pdx_j =
        static_cast<fv_real>((static_cast<accumulator_real>(sx_j) * dx_j) * en1_j[n]);
    const fv_real pdx_jp =
        static_cast<fv_real>((static_cast<accumulator_real>(sx_jp) * dx_jp) * en1_jp[n]);
    const fv_real pdy_i =
        static_cast<fv_real>((static_cast<accumulator_real>(sy_i) * dy_i) * en2_i[n]);
    const fv_real pdy_ip =
        static_cast<fv_real>((static_cast<accumulator_real>(sy_ip) * dy_ip) * en2_ip[n]);
    // :1628 grad(n,i,j) = pdx(n,i,j+1)-pdx(n,i,j)-pdy(n,i,j)+pdy(n,i+1,j)
    const fv_real gradn = pdx_jp - pdx_j - pdy_i + pdy_ip;
    dot = (n == 0) ? v3n * gradn : static_cast<fv_real>(dot + v3n * gradn);
  }
  // :1633-1634 om = om + 0.5*rarea*(v3.grad)
  return (accumulator_real(0.5) *
          static_cast<accumulator_real>(m.rarea(i, j))) *
         static_cast<accumulator_real>(dot);
}

/* dyn_core.F90:505-516.  Fills pem over is-1..ie+1 / js-1..je+1, the exact box
 * the Fortran allocates, which is also everything omga and a2b_ord2 read. */
void omega_pem_snapshot(const HorizontalBounds &, fv_real ptop,
                        View3<const fv_real> delp, View3<fv_real> pem);

/* dyn_core.F90:1208-1212.  `pe` keeps its registered (i,k,j) axis order, so the
 * level index is Fortran k+1 minus one, i.e. exactly k0+1 for pem level k0. */
void omega_from_pe(const HorizontalBounds &, fv_real rdt,
                   View3<const fv_real> pe_ikj, View3<fv_real> pem,
                   View3<fv_real> omga);

/* dyn_core.F90:1535-1636, with a2b_ord2 already applied into `pb`. */
void adv_pe(const HorizontalBounds &, View3<const fv_real> ua,
            View3<const fv_real> va, View3<fv_real> pb, const AdvPeMetrics &,
            View3<fv_real> om);

} // namespace fv3::gpu::operators

#undef FV3_WP_D_HD
#undef FV3_WP_D_INLINE
