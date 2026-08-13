#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"

#include <cstdint>

namespace fv3::gpu::operators {

/* Lightweight kernel views are produced from WP-E descriptors.  They do not
 * define storage, precision, ownership, or an ABI. */
template <class T> struct View2 {
  T *data;
  int64_t lo_i, lo_j;
  int64_t si, sj;
  T &operator()(int64_t i, int64_t j) const {
    return data[(i - lo_i) * si + (j - lo_j) * sj];
  }
};

template <class T> struct View3 {
  T *data;
  int64_t lo_i, lo_j, lo_k;
  int64_t si, sj, sk;
  T &operator()(int64_t i, int64_t j, int64_t k) const {
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

void reconstruct_pe_halo(const HorizontalBounds &, fv_real ptop,
                         View3<const fv_real> delp, View3<fv_real> pe);
void reconstruct_pk3_halo(const HorizontalBounds &, fv_real ptop,
                          fv_real akap, View3<const fv_real> delp,
                          View3<fv_real> pk3);
void reconstruct_pln_halo(const HorizontalBounds &, fv_real ptop,
                          View3<const fv_real> delp, View3<fv_real> pln);

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

/* One del2_cubed pass.  Orchestration repeats min(3,nmax) passes, shrinking
 * the written band by one each pass and placing H_HEAT only before pass one. */
void del2_pass(const HorizontalBounds &, int halo_band, fv_real cd,
               View3<fv_real> q, const Del2Metrics &, View3<fv_real> fx,
               View3<fv_real> fy);

/* The post-loop conversion in dyn_core: bounded heat source is divided by
 * delp and pkz before being added to potential temperature. */
void apply_diffusive_heating_nh(const HorizontalBounds &, fv_real bdt,
                             fv_real delt_max, fv_real cv_air, fv_real rdg,
                             int64_t n_con, View3<fv_real> heat_source,
                             View3<const fv_real> delp,
                             View3<const fv_real> delz,
                             View3<const fv_real> cappa,
                             View3<fv_real> pkz, View3<fv_real> pt);

} // namespace fv3::gpu::operators
