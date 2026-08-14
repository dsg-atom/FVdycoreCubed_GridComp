#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"

#include <cstdint>

namespace fv3::gpu::operators {

#if defined(__CUDACC__)
#define FV3_WP_D_HD __host__ __device__
#define FV3_WP_D_INLINE __forceinline__
#else
#define FV3_WP_D_HD
#define FV3_WP_D_INLINE inline
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

/* One del2_cubed pass.  Orchestration repeats min(3,nmax) passes, shrinking
 * the written band by one each pass and placing H_HEAT only before pass one. */
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

} // namespace fv3::gpu::operators

#undef FV3_WP_D_HD
#undef FV3_WP_D_INLINE
