#pragma once

#include "fv3_gpu_types.hpp"
#include <cstddef>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_VERTICAL_HD __host__ __device__
#else
#define FV3_VERTICAL_HD
#endif

// Package-local, zero-based views.  Integration constructs these from the
// versioned WP-E descriptors; no layout or ownership is inferred here.
template <class T> struct Column {
  T *data;
  std::ptrdiff_t stride_k;
  FV3_VERTICAL_HD inline T &operator[](int k) const { return data[k * stride_k]; }
};

// Exact translation of nh_utils.F90:edge_profile for one (i,j) column and one
// transported pair.  The forward elimination and backward substitution are
// deliberately serial in k.
template <class T>
FV3_VERTICAL_HD inline void edge_profile_column(const Column<const T> q1,
                                const Column<const T> q2,
                                Column<T> q1e, Column<T> q2e, T *gamma,
                                const T *dp0, int km, bool uniform_grid,
                                int limiter) {
  if (uniform_grid) {
    const T r2o3 = T(2) / T(3);
    const T r4o3 = T(4) / T(3);
    q1e[0] = r4o3 * q1[0] + r2o3 * q1[1];
    q2e[0] = r4o3 * q2[0] + r2o3 * q2[1];
    gamma[0] = T(7) / T(3);
    for (int k = 1; k < km; ++k) {
      gamma[k] = T(1) / (T(4) - gamma[k - 1]);
      q1e[k] = (T(3) * (q1[k - 1] + q1[k]) - q1e[k - 1]) * gamma[k];
      q2e[k] = (T(3) * (q2[k - 1] + q2[k]) - q2e[k - 1]) * gamma[k];
    }
    const T bet = T(1) / (T(1.5) - T(3.5) * gamma[km - 1]);
    q1e[km] = (T(4) * q1[km - 1] + q1[km - 2] -
                 T(3.5) * q1e[km - 1]) * bet;
    q2e[km] = (T(4) * q2[km - 1] + q2[km - 2] -
                 T(3.5) * q2e[km - 1]) * bet;
  } else {
    const T g0 = dp0[1] / dp0[0];
    const T xt1 = T(2) * g0 * (g0 + T(1));
    const T bet0 = g0 * (g0 + T(0.5));
    q1e[0] = (xt1 * q1[0] + q1[1]) / bet0;
    q2e[0] = (xt1 * q2[0] + q2[1]) / bet0;
    gamma[0] = (T(1) + g0 * (g0 + T(1.5))) / bet0;
    T gk = g0;
    for (int k = 1; k < km; ++k) {
      gk = dp0[k - 1] / dp0[k];
      const T bet = T(2) + T(2) * gk - gamma[k - 1];
      q1e[k] = (T(3) * (q1[k - 1] + gk * q1[k]) - q1e[k - 1]) / bet;
      q2e[k] = (T(3) * (q2[k - 1] + gk * q2[k]) - q2e[k - 1]) / bet;
      gamma[k] = gk / bet;
    }
    const T a_bot = T(1) + gk * (gk + T(1.5));
    const T xtb = T(2) * gk * (gk + T(1));
    const T den = gk * (gk + T(0.5)) - a_bot * gamma[km - 1];
    q1e[km] = (xtb * q1[km - 1] + q1[km - 2] -
                 a_bot * q1e[km - 1]) / den;
    q2e[km] = (xtb * q2[km - 1] + q2[km - 2] -
                 a_bot * q2e[km - 1]) / den;
  }
  for (int k = km - 1; k >= 0; --k) {
    q1e[k] -= gamma[k] * q1e[k + 1];
    q2e[k] -= gamma[k] * q2e[k + 1];
  }
  if (limiter != 0) {
    if (q1[0] * q1e[0] < T(0)) q1e[0] = T(0);
    if (q2[0] * q2e[0] < T(0)) q2e[0] = T(0);
    if (q1[km - 1] * q1e[km] < T(0)) q1e[km] = T(0);
    if (q2[km - 1] * q2e[km] < T(0)) q2e[km] = T(0);
  }
}

template <class T>
FV3_VERTICAL_HD inline T update_height_boundary_column(Column<T> zh, Column<T> delz, T zs,
                                       T rdt, T dz_min, int km) {
  const T ws = (zs - zh[km]) * rdt;
  for (int k = km - 1; k >= 0; --k) {
    const T floor = zh[k + 1] + dz_min;
    if (zh[k] < floor) zh[k] = floor;
    delz[k] = zh[k + 1] - zh[k];
  }
  return ws;
}

template <class T>
FV3_VERTICAL_HD inline void reconstruct_height_column(Column<T> zh, const Column<const T> dz,
                                      T surface_height, int km) {
  zh[km] = surface_height;
  for (int k = km - 1; k >= 0; --k) zh[k] = zh[k + 1] - dz[k];
}

} // namespace fv3::gpu::vertical

#undef FV3_VERTICAL_HD
