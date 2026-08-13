#pragma once

#include "fv3_gpu_types.hpp"
#include <cstddef>
#include <cmath>

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

// Exact dry/MOIST_CAPPA translation of nh_utils.F90:SIM1_solver for one
// horizontal column. Callers provide all work arrays from persistent storage.
template <class T>
inline void sim1_column(T dt, T rgas, T gama, T kappa, T p_fac,
                        bool moist_cappa, const T *gm2, const T *cp2,
                        const T *dm, const T *pm, const T *pem, T *w, T *dz,
                        const T *pt, T ws, T *pe, T *aa, T *bb, T *dd,
                        T *w1, T *g_rat, T *gam, T *pp, T *bet, T *p1,
                        int km) {
  const T t1g = (moist_cappa ? T(1) : gama) * T(2) * dt * dt;
  const T rdt = T(1) / dt, capa1 = kappa - T(1);
  for (int k=0;k<km;++k) {
    pe[k]=std::exp((moist_cappa?gm2[k]:gama)*std::log(-dm[k]/dz[k]*rgas*pt[k]))-pm[k];
    w1[k]=w[k];
  }
  for(int k=0;k<km-1;++k){g_rat[k]=dm[k]/dm[k+1];bb[k]=T(2)*(T(1)+g_rat[k]);dd[k]=T(3)*(pe[k]+g_rat[k]*pe[k+1]);}
  bet[0]=bb[0];pp[0]=T(0);pp[1]=dd[0]/bet[0];bb[km-1]=T(2);dd[km-1]=T(3)*pe[km-1];
  for(int k=1;k<km;++k){gam[k]=g_rat[k-1]/bet[k-1];bet[k]=bb[k]-gam[k];pp[k+1]=(dd[k]-pp[k])/bet[k];}
  for(int k=km-1;k>=1;--k)pp[k]-=gam[k]*pp[k+1];
  for(int k=1;k<km;++k){const T f=moist_cappa?T(.5)*(gm2[k-1]+gm2[k]):T(1);aa[k]=t1g*f/(dz[k-1]+dz[k])*(pem[k]+pp[k]);}
  bet[0]=dm[0]-aa[1];w[0]=(dm[0]*w1[0]+dt*pp[1])/bet[0];
  for(int k=1;k<km-1;++k){gam[k]=aa[k]/bet[k-1];bet[k]=dm[k]-(aa[k]+aa[k+1]+aa[k]*gam[k]);w[k]=(dm[k]*w1[k]+dt*(pp[k+1]-pp[k])-aa[k]*w[k-1])/bet[k];}
  const int kb=km-1;p1[kb]=t1g*(moist_cappa?gm2[kb]:T(1))/dz[kb]*(pem[km]+pp[km]);
  gam[kb]=aa[kb]/bet[kb-1];bet[kb]=dm[kb]-(aa[kb]+p1[kb]+aa[kb]*gam[kb]);
  w[kb]=(dm[kb]*w1[kb]+dt*(pp[km]-pp[kb])-p1[kb]*ws-aa[kb]*w[kb-1])/bet[kb];
  for(int k=km-2;k>=0;--k)w[k]-=gam[k+1]*w[k+1];
  pe[0]=T(0);for(int k=0;k<km;++k)pe[k+1]=pe[k]+dm[k]*(w[k]-w1[k])*rdt;
  p1[kb]=(pe[kb]+T(2)*pe[km])/T(3);
  for(int k=kb;k>=0;--k){
    if(k<kb)p1[k]=(pe[k]+bb[k]*pe[k+1]+g_rat[k]*pe[k+2])/T(3)-g_rat[k]*p1[k+1];
    const T pg=p_fac*pm[k]>p1[k]+pm[k]?p_fac*pm[k]:p1[k]+pm[k];
    dz[k]=-dm[k]*rgas*pt[k]*std::exp((moist_cappa?cp2[k]-T(1):capa1)*std::log(pg));
  }
}

} // namespace fv3::gpu::vertical

#undef FV3_VERTICAL_HD
