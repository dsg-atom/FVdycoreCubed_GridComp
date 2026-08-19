#pragma once

#include "fv3_gpu_types.hpp"
#include <cstddef>
#include <cmath>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_VERTICAL_HD __host__ __device__
// CUDA-compatible math functions that work in both host and device code
template <typename T> __host__ __device__ inline T fv3_exp(T x) { return exp(x); }
template <typename T> __host__ __device__ inline T fv3_log(T x) { return log(x); }
template <typename T> __host__ __device__ inline T fv3_sqrt(T x) { return sqrt(x); }
template <typename T> __host__ __device__ inline T fv3_fabs(T x) { return fabs(x); }
template <typename T> __host__ __device__ inline T fv3_copysign(T x, T y) { return copysign(x, y); }
template <typename T> __host__ __device__ inline T fv3_max(T x, T y) { return x > y ? x : y; }
#else
#define FV3_VERTICAL_HD
template <typename T> inline T fv3_exp(T x) { return std::exp(x); }
template <typename T> inline T fv3_log(T x) { return std::log(x); }
template <typename T> inline T fv3_sqrt(T x) { return std::sqrt(x); }
template <typename T> inline T fv3_fabs(T x) { return std::fabs(x); }
template <typename T> inline T fv3_copysign(T x, T y) { return std::copysign(x, y); }
template <typename T> inline T fv3_max(T x, T y) { return std::max(x, y); }
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
FV3_VERTICAL_HD inline void sim1_column(T dt, T rgas, T gama, T kappa, T p_fac,
                        bool moist_cappa, const T *gm2, const T *cp2,
                        const T *dm, const T *pm, const T *pem, T *w, T *dz,
                        const T *pt, T ws, T *pe, T *aa, T *bb, T *dd,
                        T *w1, T *g_rat, T *gam, T *pp, T *bet, T *p1,
                        int km) {
  const T t1g = (moist_cappa ? T(1) : gama) * T(2) * dt * dt;
  const T rdt = T(1) / dt, capa1 = kappa - T(1);
  for (int k=0;k<km;++k) {
    pe[k]=fv3_exp((moist_cappa?gm2[k]:gama)*fv3_log(-dm[k]/dz[k]*rgas*pt[k]))-pm[k];
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
    dz[k]=-dm[k]*rgas*pt[k]*fv3_exp((moist_cappa?cp2[k]-T(1):capa1)*fv3_log(pg));
  }
}

// Riem_Solver3 column setup/finalization. Pressure output columns may have
// arbitrary strides at the wrapper; these arrays are the contiguous arena
// representation used by the sequential solver.
template <class T>
FV3_VERTICAL_HD inline void riem3_precompute_column(const T *delp, const T *pt,
                                    const T *q_con, const T *cappa,
                                    const T *zh, const T *w, T ptop, T akap,
                                    T grav, bool use_cond, bool moist_cappa,
                                    T *dm, T *pm, T *pem, T *peln2, T *pk3,
                                    T *gm, T *cp, T *dz, T *w2, T *peg,
                                    T *pelng, int km) {
  const T rgrav=T(1)/grav, peln1=fv3_log(ptop), ptk=fv3_exp(akap*peln1);
  pem[0]=ptop;peln2[0]=peln1;pk3[0]=ptk;
  peg[0]=ptop;pelng[0]=peln1;
  for(int k=0;k<km;++k){
    dm[k]=delp[k];
    pem[k+1]=pem[k]+dm[k];peln2[k+1]=fv3_log(pem[k+1]);
    pk3[k+1]=fv3_exp(akap*peln2[k+1]);
    if(use_cond){peg[k+1]=peg[k]+dm[k]*(T(1)-q_con[k]);pelng[k+1]=fv3_log(peg[k+1]);}
  }
  for(int k=0;k<km;++k){
    const T layer_pm=use_cond?(peg[k+1]-peg[k])/(pelng[k+1]-pelng[k])
                             :dm[k]/(peln2[k+1]-peln2[k]);
    pm[k]=layer_pm;
    if(moist_cappa){cp[k]=cappa[k];gm[k]=T(1)/(T(1)-cp[k]);}
    dm[k]*=rgrav;dz[k]=zh[k+1]-zh[k];w2[k]=w[k];
  }
  (void)pt;
}

template <class T>
FV3_VERTICAL_HD inline void riem3_finalize_column(const T *pem, const T *peln2,
                                  const T *pe2, const T *dz, T *w,
                                  const T *w2, T *delz, T *zh, T zs,
                                  T *pe, T *peln, T *pk, T *pk3,
                                  bool last_call, bool fp_out, bool use_logp,
                                  T *ppe, int km) {
  for(int k=0;k<km;++k){w[k]=w2[k];delz[k]=dz[k];}
  if(last_call)for(int k=0;k<=km;++k){peln[k]=peln2[k];pk[k]=pk3[k];pe[k]=pem[k];}
  for(int k=0;k<=km;++k)ppe[k]=fp_out?pe2[k]+pem[k]:pe2[k];
  if(use_logp)for(int k=1;k<=km;++k)pk3[k]=peln2[k];
  zh[km]=zs;for(int k=km-1;k>=0;--k)zh[k]=zh[k+1]-dz[k];
}

// Exact translation of nh_utils.F90:SIM3p0_solver for one horizontal column.
// Beta=0 variant of SIM3. All work arrays are caller-provided from persistent arena.
template <class T>
FV3_VERTICAL_HD inline void sim3p0_column(T dt, T rgas, T gama, T kappa, T p_fac, T scale_m,
                          const T *dm, const T *pt, const T *pem, T *w,
                          T *dz, T ws, T *pe2, T *aa, T *bb, T *dd, T *w1,
                          T *g_rat, T *gam, T *pp, T *bet, T *p1, int km) {
  const T t1g = T(2) * gama * dt * dt;
  const T rdt = T(1) / dt;
  const T capa1 = kappa - T(1);
  const T r2g = T(9.8) / T(2);
  const T r6g = T(9.8) / T(6);
  const T r3 = T(1) / T(3);

  for (int k = 0; k < km; ++k) {
    w1[k] = w[k];
    aa[k] = fv3_exp(gama * fv3_log(-dm[k] / dz[k] * rgas * pt[k]));
  }

  for (int k = 0; k < km - 1; ++k) {
    g_rat[k] = dm[k] / dm[k + 1];
    bb[k] = T(2) * (T(1) + g_rat[k]);
    dd[k] = T(3) * (aa[k] + g_rat[k] * aa[k + 1]);
  }

  bet[0] = bb[0];
  pe2[0] = pem[0];
  pe2[1] = (dd[0] - pem[0]) / bet[0];
  bb[km - 1] = T(2);
  dd[km - 1] = T(3) * aa[km - 1] + r2g * dm[km - 1];

  for (int k = 1; k < km; ++k) {
    gam[k] = g_rat[k - 1] / bet[k - 1];
    bet[k] = bb[k] - gam[k];
    pe2[k + 1] = (dd[k] - pe2[k]) / bet[k];
  }

  for (int k = km - 1; k >= 1; --k) {
    pe2[k] = pe2[k] - gam[k] * pe2[k + 1];
  }

  for (int k = 0; k <= km; ++k) {
    pp[k] = pe2[k] - pem[k];
  }

  for (int k = 1; k < km; ++k) {
    aa[k] = t1g / (dz[k - 1] + dz[k]) * pe2[k] - scale_m * dm[0];
  }

  bet[0] = dm[0] - aa[1];
  w[0] = (dm[0] * w1[0] + dt * pp[1]) / bet[0];

  for (int k = 1; k < km - 1; ++k) {
    gam[k] = aa[k] / bet[k - 1];
    bet[k] = dm[k] - (aa[k] + aa[k + 1] + aa[k] * gam[k]);
    w[k] = (dm[k] * w1[k] + dt * (pp[k + 1] - pp[k]) - aa[k] * w[k - 1]) / bet[k];
  }

  const T wk1 = t1g / dz[km - 1] * pe2[km];
  gam[km - 1] = aa[km - 1] / bet[km - 2];
  bet[km - 1] = dm[km - 1] - (aa[km - 1] + wk1 + aa[km - 1] * gam[km - 1]);
  w[km - 1] = (dm[km - 1] * w1[km - 1] + dt * (pp[km] - pp[km - 1]) -
               wk1 * ws - aa[km - 1] * w[km - 2]) / bet[km - 1];

  for (int k = km - 2; k >= 0; --k) {
    w[k] = w[k] - gam[k + 1] * w[k + 1];
  }

  pe2[0] = T(0);
  for (int k = 0; k < km; ++k) {
    pe2[k + 1] = pe2[k] + dm[k] * (w[k] - w1[k]) * rdt;
  }

  pe2[0] = pem[0];
  for (int k = 1; k <= km; ++k) {
    pe2[k] = fv3_max(p_fac * pem[k], pe2[k] + pem[k]);
  }

  p1[km - 1] = (pe2[km - 1] + T(2) * pe2[km]) * r3 - r6g * dm[km - 1];
  dz[km - 1] = -dm[km - 1] * rgas * pt[km - 1] *
               fv3_exp(capa1 * fv3_log(p1[km - 1]));

  for (int k = km - 2; k >= 0; --k) {
    p1[k] = (pe2[k] + bb[k] * pe2[k + 1] + g_rat[k] * pe2[k + 2]) * r3 -
            g_rat[k] * p1[k + 1];
    dz[k] = -dm[k] * rgas * pt[k] * fv3_exp(capa1 * fv3_log(p1[k]));
  }

  for (int k = 0; k <= km; ++k) {
    pe2[k] = pe2[k] - pem[k];
  }
}

// Exact translation of nh_utils.F90:SIM3_solver for one horizontal column.
// General alpha/beta variant. All work arrays are caller-provided from persistent arena.
template <class T>
FV3_VERTICAL_HD inline void sim3_column(T dt, T rgas, T gama, T kappa, T alpha, T p_fac,
                        T scale_m, const T *dm, const T *pt, const T *pem,
                        T *w, T *dz, T ws, T *pe2, T *aa, T *bb, T *dd,
                        T *w1, T *wk, T *g_rat, T *gam, T *pp, T *bet, T *p1,
                        int km) {
  const T beta = T(1) - alpha;
  const T ra = T(1) / alpha;
  const T t2 = beta / alpha;
  const T t1g = gama * T(2) * (alpha * dt) * (alpha * dt);
  const T rdt = T(1) / dt;
  const T capa1 = kappa - T(1);
  const T r2g = T(9.8) / T(2);
  const T r6g = T(9.8) / T(6);
  const T r3 = T(1) / T(3);

  for (int k = 0; k < km; ++k) {
    w1[k] = w[k];
    aa[k] = fv3_exp(gama * fv3_log(-dm[k] / dz[k] * rgas * pt[k]));
  }

  for (int k = 0; k < km - 1; ++k) {
    g_rat[k] = dm[k] / dm[k + 1];
    bb[k] = T(2) * (T(1) + g_rat[k]);
    dd[k] = T(3) * (aa[k] + g_rat[k] * aa[k + 1]);
  }

  bet[0] = bb[0];
  pe2[0] = pem[0];
  pe2[1] = (dd[0] - pem[0]) / bet[0];
  bb[km - 1] = T(2);
  dd[km - 1] = T(3) * aa[km - 1] + r2g * dm[km - 1];

  for (int k = 1; k < km; ++k) {
    gam[k] = g_rat[k - 1] / bet[k - 1];
    bet[k] = bb[k] - gam[k];
    pe2[k + 1] = (dd[k] - pe2[k]) / bet[k];
  }

  for (int k = km - 1; k >= 1; --k) {
    pe2[k] = pe2[k] - gam[k] * pe2[k + 1];
  }

  for (int k = 0; k <= km; ++k) {
    pp[k] = pe2[k] - pem[k];
  }

  for (int k = 1; k < km; ++k) {
    aa[k] = t1g / (dz[k - 1] + dz[k]) * pe2[k];
    wk[k] = t2 * aa[k] * (w1[k - 1] - w1[k]);
    aa[k] = aa[k] - scale_m * dm[0];
  }

  bet[0] = dm[0] - aa[1];
  w[0] = (dm[0] * w1[0] + dt * pp[1] + wk[1]) / bet[0];

  for (int k = 1; k < km - 1; ++k) {
    gam[k] = aa[k] / bet[k - 1];
    bet[k] = dm[k] - (aa[k] + aa[k + 1] + aa[k] * gam[k]);
    w[k] = (dm[k] * w1[k] + dt * (pp[k + 1] - pp[k]) + wk[k + 1] - wk[k] -
            aa[k] * w[k - 1]) / bet[k];
  }

  const T wk1 = t1g / dz[km - 1] * pe2[km];
  gam[km - 1] = aa[km - 1] / bet[km - 2];
  bet[km - 1] = dm[km - 1] - (aa[km - 1] + wk1 + aa[km - 1] * gam[km - 1]);
  w[km - 1] = (dm[km - 1] * w1[km - 1] + dt * (pp[km] - pp[km - 1]) -
               wk[km - 1] + wk1 * (t2 * w1[km - 1] - ra * ws) -
               aa[km - 1] * w[km - 2]) / bet[km - 1];

  for (int k = km - 2; k >= 0; --k) {
    w[k] = w[k] - gam[k + 1] * w[k + 1];
  }

  pe2[0] = T(0);
  for (int k = 0; k < km; ++k) {
    pe2[k + 1] = pe2[k] + (dm[k] * (w[k] - w1[k]) * rdt - beta * (pp[k + 1] - pp[k])) * ra;
  }

  pe2[0] = pem[0];
  for (int k = 1; k <= km; ++k) {
    pe2[k] = fv3_max(p_fac * pem[k], pe2[k] + pem[k]);
  }

  p1[km - 1] = (pe2[km - 1] + T(2) * pe2[km]) * r3 - r6g * dm[km - 1];
  dz[km - 1] = -dm[km - 1] * rgas * pt[km - 1] *
               fv3_exp(capa1 * fv3_log(fv3_max(p1[km - 1], T(1))));

  for (int k = km - 2; k >= 0; --k) {
    p1[k] = (pe2[k] + bb[k] * pe2[k + 1] + g_rat[k] * pe2[k + 2]) * r3 -
            g_rat[k] * p1[k + 1];
    dz[k] = -dm[k] * rgas * pt[k] * fv3_exp(capa1 * fv3_log(fv3_max(p1[k], T(1))));
  }

  for (int k = 0; k <= km; ++k) {
    pe2[k] = pe2[k] - pem[k];
    pe2[k] = pe2[k] + beta * (pp[k] - pe2[k]);
  }
}

// Exact translation of nh_utils.F90:SIM_solver for one horizontal column.
// General a_imp variant with moist_cappa support. All work arrays are caller-provided.
template <class T>
FV3_VERTICAL_HD inline void sim_column(T dt, T rgas, T gama, T kappa, T alpha, T p_fac,
                       T scale_m, bool moist_cappa, const T *gm2, const T *cp2,
                       const T *dm, const T *pm, const T *pt, const T *pem,
                       T *w, T *dz, T ws, T *pe2, T *aa, T *bb, T *dd,
                       T *w1, T *wk, T *g_rat, T *gam, T *pp, T *bet, T *p1,
                       int km) {
  const T beta = T(1) - alpha;
  const T ra = T(1) / alpha;
  const T t2 = beta / alpha;
  const T t1g = moist_cappa ? T(2) * (alpha * dt) * (alpha * dt)
                            : T(2) * gama * (alpha * dt) * (alpha * dt);
  const T rdt = T(1) / dt;
  const T capa1 = kappa - T(1);
  const T r3 = T(1) / T(3);

  for (int k = 0; k < km; ++k) {
    w1[k] = w[k];
    pe2[k] = fv3_exp((moist_cappa ? gm2[k] : gama) *
                      fv3_log(-dm[k] / dz[k] * rgas * pt[k])) - pm[k];
  }

  for (int k = 0; k < km - 1; ++k) {
    g_rat[k] = dm[k] / dm[k + 1];
    bb[k] = T(2) * (T(1) + g_rat[k]);
    dd[k] = T(3) * (pe2[k] + g_rat[k] * pe2[k + 1]);
  }

  bet[0] = bb[0];
  pp[0] = T(0);
  pp[1] = dd[0] / bet[0];
  bb[km - 1] = T(2);
  dd[km - 1] = T(3) * pe2[km - 1];

  for (int k = 1; k < km; ++k) {
    gam[k] = g_rat[k - 1] / bet[k - 1];
    bet[k] = bb[k] - gam[k];
    pp[k + 1] = (dd[k] - pp[k]) / bet[k];
  }

  for (int k = km - 1; k >= 1; --k) {
    pp[k] = pp[k] - gam[k] * pp[k + 1];
  }

  for (int k = 0; k <= km; ++k) {
    pe2[k] = pem[k] + pp[k];
  }

  for (int k = 1; k < km; ++k) {
    const T gf = moist_cappa ? T(0.5) * (gm2[k - 1] + gm2[k]) : T(1);
    aa[k] = t1g * gf / (dz[k - 1] + dz[k]) * pe2[k];
    wk[k] = t2 * aa[k] * (w1[k - 1] - w1[k]);
    aa[k] = aa[k] - scale_m * dm[0];
  }

  bet[0] = dm[0] - aa[1];
  w[0] = (dm[0] * w1[0] + dt * pp[1] + wk[1]) / bet[0];

  for (int k = 1; k < km - 1; ++k) {
    gam[k] = aa[k] / bet[k - 1];
    bet[k] = dm[k] - (aa[k] + aa[k + 1] + aa[k] * gam[k]);
    w[k] = (dm[k] * w1[k] + dt * (pp[k + 1] - pp[k]) + wk[k + 1] - wk[k] -
            aa[k] * w[k - 1]) / bet[k];
  }

  const T wk1 = (moist_cappa ? t1g * gm2[km - 1] : t1g) / dz[km - 1] * pe2[km];
  gam[km - 1] = aa[km - 1] / bet[km - 2];
  bet[km - 1] = dm[km - 1] - (aa[km - 1] + wk1 + aa[km - 1] * gam[km - 1]);
  w[km - 1] = (dm[km - 1] * w1[km - 1] + dt * (pp[km] - pp[km - 1]) -
               wk[km - 1] + wk1 * (t2 * w1[km - 1] - ra * ws) -
               aa[km - 1] * w[km - 2]) / bet[km - 1];

  for (int k = km - 2; k >= 0; --k) {
    w[k] = w[k] - gam[k + 1] * w[k + 1];
  }

  pe2[0] = T(0);
  for (int k = 0; k < km; ++k) {
    pe2[k + 1] = pe2[k] + (dm[k] * (w[k] - w1[k]) * rdt - beta * (pp[k + 1] - pp[k])) * ra;
  }

  p1[km - 1] = (pe2[km - 1] + T(2) * pe2[km]) * r3;
  const T pg_km = fv3_max(p_fac * pm[km - 1], p1[km - 1] + pm[km - 1]);
  dz[km - 1] = -dm[km - 1] * rgas * pt[km - 1] *
               fv3_exp((moist_cappa ? cp2[km - 1] - T(1) : capa1) * fv3_log(pg_km));

  for (int k = km - 2; k >= 0; --k) {
    p1[k] = (pe2[k] + bb[k] * pe2[k + 1] + g_rat[k] * pe2[k + 2]) * r3 -
            g_rat[k] * p1[k + 1];
    const T pg = fv3_max(p_fac * pm[k], p1[k] + pm[k]);
    dz[k] = -dm[k] * rgas * pt[k] *
            fv3_exp((moist_cappa ? cp2[k] - T(1) : capa1) * fv3_log(pg));
  }

  for (int k = 0; k <= km; ++k) {
    pe2[k] = pe2[k] + beta * (pp[k] - pe2[k]);
  }
}

// Exact translation of nh_utils.F90:RIM_2D for one horizontal column.
// Method of characteristics with acoustic substeps. All work arrays are caller-provided.
template <class T>
FV3_VERTICAL_HD inline void rim_2d_column(int ms, T bdt, T rgas, T gama, bool moist_cappa,
                          const T *gm2, const T *dm2, const T *pm2, const T *pt2,
                          T ws, T *dz2, T *w2, T *pe2, T *dts, T *pf1,
                          T *r_lo, T *r_hi, T *m_bot, T *m_top, T *r_bot,
                          T *r_top, T *wbar, T *pbar, T *cm, T *wc, T *pp,
                          T *dm_tmp, T *wm, T *dz_tmp, int km) {
  const T grg = gama * rgas;
  const T rdt = T(1) / bdt;
  const T dt = bdt / T(ms);
  const T ws2 = T(2) * ws;

  for (int k = 0; k <= km; ++k) {
    pbar[k] = T(0);
    wbar[k] = T(0);
  }
  wbar[km] = ws;

  for (int k = 0; k < km; ++k) {
    dz_tmp[k] = dz2[k];
    dm_tmp[k] = dm2[k];
    wm[k] = w2[k] * dm2[k];
  }

  for (int k = 0; k <= km; ++k) pe2[k] = T(0);

  int ks0 = 0;
  if (ms > 1 && ms < 8) {
    for (int k = 0; k < km; ++k) {
      const T rden = -rgas * dm_tmp[k] / dz_tmp[k];
      pf1[k] = fv3_exp((moist_cappa ? gm2[k] : gama) * fv3_log(rden * pt2[k]));
      dts[k] = -dz_tmp[k] / fv3_sqrt(grg * pf1[k] / rden);
      if (bdt > dts[k]) {
        ks0 = k;
        goto fast_path_check;
      }
    }
    ks0 = km;
  fast_path_check:
    if (ks0 == 0) goto slow_path;

    for (int k = 0; k < ks0; ++k) {
      cm[k] = dm_tmp[k] / dts[k];
      wc[k] = wm[k] / dts[k];
      pp[k] = pf1[k] - pm2[k];
    }

    wbar[0] = (wc[0] + pp[0]) / cm[0];
    for (int k = 1; k < ks0; ++k) {
      wbar[k] = (wc[k - 1] + wc[k] + pp[k] - pp[k - 1]) / (cm[k - 1] + cm[k]);
      pbar[k] = bdt * (cm[k - 1] * wbar[k] - wc[k - 1] + pp[k - 1]);
      pe2[k] = pbar[k];
    }

    if (ks0 == km) {
      pbar[km] = bdt * (cm[km - 1] * wbar[km] - wc[km - 1] + pp[km - 1]);
      for (int k = 0; k < km; ++k) {
        dz2[k] = dz_tmp[k] + bdt * (wbar[k + 1] - wbar[k]);
        w2[k] = (wm[k] + pbar[k + 1] - pbar[k]) / dm_tmp[k];
      }
      pe2[0] = T(0);
      for (int k = 1; k <= km; ++k) pe2[k] = pbar[k] * rdt;
      return;
    } else {
      for (int k = 0; k < ks0 - 1; ++k) {
        dz2[k] = dz_tmp[k] + bdt * (wbar[k + 1] - wbar[k]);
        w2[k] = (wm[k] + pbar[k + 1] - pbar[k]) / dm_tmp[k];
      }
      pbar[ks0 - 1] = pbar[ks0 - 1] / T(ms);
    }
  }
slow_path:
  int ks1 = ks0;

  for (int n = 0; n < ms; ++n) {
    for (int k = ks1; k < km; ++k) {
      const T rden = -rgas * dm_tmp[k] / dz_tmp[k];
      const T pf = fv3_exp((moist_cappa ? gm2[k] : gama) * fv3_log(rden * pt2[k]));
      dts[k] = -dz_tmp[k] / fv3_sqrt(grg * pf / rden);
      const T ptmp1 = dts[k] * (pf - pm2[k]);
      r_lo[k] = wm[k] + ptmp1;
      r_hi[k] = wm[k] - ptmp1;
    }

    int ktop = ks1;
    for (int k = ks1; k < km; ++k) {
      if (dt > dts[k]) {
        ktop = k;
        goto found_ktop;
      }
    }
    ktop = km;
  found_ktop:

    int kt1 = ks1 > 0 ? ks1 : 0;

    if (ktop >= ks1) {
      for (int k = ks1; k < ktop; ++k) {
        const T z_frac = dt / dts[k];
        r_bot[k] = z_frac * r_lo[k];
        r_top[k + 1] = z_frac * r_hi[k];
        m_bot[k] = z_frac * dm_tmp[k];
        m_top[k + 1] = m_bot[k];
      }
      if (ktop == km) goto finalize_wbar;
    }

    for (int k = ktop + 1; k <= km; ++k) {
      m_top[k] = T(0);
      r_top[k] = T(0);
    }
    for (int ke = km; ke >= ktop + 1; --ke) {
      T time_left = dt;
      for (int k = ke - 1; k >= kt1; --k) {
        if (time_left > dts[k]) {
          time_left = time_left - dts[k];
          m_top[ke] = m_top[ke] + dm_tmp[k];
          r_top[ke] = r_top[ke] + r_hi[k];
        } else {
          const T z_frac = time_left / dts[k];
          m_top[ke] = m_top[ke] + z_frac * dm_tmp[k];
          r_top[ke] = r_top[ke] + z_frac * r_hi[k];
          break;
        }
      }
    }

    for (int k = ktop; k < km; ++k) {
      m_bot[k] = T(0);
      r_bot[k] = T(0);
    }

    for (int ke = ktop; ke < km; ++ke) {
      T time_left = dt;
      T m_surf = T(0);
      for (int k = ke; k < km; ++k) {
        if (time_left > dts[k]) {
          time_left = time_left - dts[k];
          m_bot[ke] = m_bot[ke] + dm_tmp[k];
          r_bot[ke] = r_bot[ke] + r_lo[k];
        } else {
          const T z_frac = time_left / dts[k];
          m_bot[ke] = m_bot[ke] + z_frac * dm_tmp[k];
          r_bot[ke] = r_bot[ke] + z_frac * r_lo[k];
          goto next_ke;
        }
      }
      m_surf = m_bot[ke];
      for (int k = km - 1; k >= kt1; --k) {
        if (time_left > dts[k]) {
          time_left = time_left - dts[k];
          m_bot[ke] = m_bot[ke] + dm_tmp[k];
          r_bot[ke] = r_bot[ke] - r_hi[k];
        } else {
          const T z_frac = time_left / dts[k];
          m_bot[ke] = m_bot[ke] + z_frac * dm_tmp[k];
          r_bot[ke] = r_bot[ke] - z_frac * r_hi[k] + (m_bot[ke] - m_surf) * ws2;
          break;
        }
      }
    next_ke:;
    }

  finalize_wbar:
    if (ks1 == 0) wbar[0] = r_bot[0] / m_bot[0];
    for (int k = ks1; k < km; ++k) {
      wbar[k + 1] = (r_bot[k + 1] + r_top[k + 1]) / (m_top[k + 1] + m_bot[k + 1]);
    }
    for (int k = ks1; k <= km; ++k) {
      pbar[k] = m_top[k] * wbar[k] - r_top[k];
      pe2[k] = pe2[k] + pbar[k];
    }

    if (n == ms - 1) {
      for (int k = ks1; k < km; ++k) {
        dz2[k] = dz_tmp[k] + dt * (wbar[k + 1] - wbar[k]);
        w2[k] = (wm[k] + pbar[k + 1] - pbar[k]) / dm_tmp[k];
      }
    } else {
      for (int k = ks1; k < km; ++k) {
        dz_tmp[k] = dz_tmp[k] + dt * (wbar[k + 1] - wbar[k]);
        wm[k] = wm[k] + pbar[k + 1] - pbar[k];
      }
    }
  }

  pe2[0] = T(0);
  for (int k = 1; k <= km; ++k) pe2[k] = pe2[k] * rdt;
}

// Complete Riem_Solver3 column orchestration with a_imp dispatch.
// Callers provide all workspace arrays from persistent arena.
template <class T>
FV3_VERTICAL_HD inline void riem_solver3_column(int ms, T dt, T rgas, T gama, T kappa,
                                T ptop, T p_fac, T a_imp, T scale_m,
                                bool use_cond, bool moist_cappa,
                                bool last_call, bool fp_out, bool use_logp,
                                const T *delp, const T *pt, const T *q_con,
                                const T *cappa_in, const T *zh_in, const T *w_in,
                                T zs, T ws, T *w_out, T *dz_out, T *zh_out,
                                T *pe_out, T *peln_out, T *pk_out, T *pk3_out,
                                T *ppe_out,
                                T *dm, T *pm, T *pem, T *peln2, T *pk3_tmp,
                                T *gm, T *cp, T *dz, T *w2, T *peg, T *pelng,
                                T *pe2, T *aa, T *bb, T *dd, T *w1, T *wk,
                                T *g_rat, T *gam, T *pp, T *bet, T *p1,
                                T *dts, T *pf1, T *r_lo, T *r_hi, T *m_bot,
                                T *m_top, T *r_bot, T *r_top, T *wbar, T *pbar,
                                T *cm, T *wc, T *dm_tmp, T *wm, T *dz_tmp,
                                int km) {
  riem3_precompute_column(delp, pt, q_con, cappa_in, zh_in, w_in, ptop, kappa,
                          T(9.8), use_cond, moist_cappa, dm, pm, pem, peln2,
                          pk3_tmp, gm, cp, dz, w2, peg, pelng, km);

  if (a_imp < T(-0.999)) {
    sim3p0_column(dt, rgas, gama, kappa, p_fac, scale_m, dm, pt, pem, w2, dz,
                  ws, pe2, aa, bb, dd, w1, g_rat, gam, pp, bet, p1, km);
  } else if (a_imp < T(-0.5)) {
    sim3_column(dt, rgas, gama, kappa, -a_imp, p_fac, scale_m, dm, pt, pem, w2,
                dz, ws, pe2, aa, bb, dd, w1, wk, g_rat, gam, pp, bet, p1, km);
  } else if (a_imp <= T(0.5)) {
    rim_2d_column(ms, dt, rgas, gama, moist_cappa, gm, dm, pm, pt, ws, dz, w2,
                  pe2, dts, pf1, r_lo, r_hi, m_bot, m_top, r_bot, r_top, wbar,
                  pbar, cm, wc, pp, dm_tmp, wm, dz_tmp, km);
  } else if (a_imp > T(0.999)) {
    sim1_column(dt, rgas, gama, kappa, p_fac, moist_cappa, gm, cp, dm, pm, pem,
                w2, dz, pt, ws, pe2, aa, bb, dd, w1, g_rat, gam, pp, bet, p1, km);
  } else {
    sim_column(dt, rgas, gama, kappa, a_imp, p_fac, scale_m, moist_cappa, gm,
               cp, dm, pm, pt, pem, w2, dz, ws, pe2, aa, bb, dd, w1, wk, g_rat,
               gam, pp, bet, p1, km);
  }

  riem3_finalize_column(pem, peln2, pe2, dz, w_out, w2, dz_out, zh_out, zs,
                        pe_out, peln_out, pk_out, pk3_tmp, last_call, fp_out,
                        use_logp, ppe_out, km);

  for (int k = 0; k <= km; ++k) pk3_out[k] = pk3_tmp[k];
}

// Workspace slot offsets within per-column arena. All offsets are in
// sizeof(fv_real) units from the column's workspace base.
struct VerticalWorkspaceLayout {
  // Riem_Solver3 precompute/finalize arrays (km+1 each)
  uint32_t dm, pm, pem, peln2, pk3_tmp, gm, cp, dz, w2, peg, pelng;
  // Solver work arrays (km+1 each)
  uint32_t pe2, aa, bb, dd, w1, wk, g_rat, gam, pp, bet, p1;
  // RIM_2D characteristic arrays (km+2 each for surface reflection)
  uint32_t dts, pf1, r_lo, r_hi, m_bot, m_top, r_bot, r_top;
  uint32_t wbar, pbar, cm, wc, dm_tmp, wm, dz_tmp;
  // Total scalars required per column
  uint32_t total_scalars;
};

// Workspace layout computation for vertical solvers.
// Deterministic, non-overlapping assignment for RIM/SIM/Riemann arrays.
inline VerticalWorkspaceLayout compute_workspace_layout(int km) noexcept {
  VerticalWorkspaceLayout layout{};
  const uint32_t km1 = static_cast<uint32_t>(km + 1);
  const uint32_t km2 = static_cast<uint32_t>(km + 2);
  uint32_t offset = 0;

  // Riem_Solver3 precompute/finalize arrays (km+1 each)
  layout.dm = offset; offset += km1;
  layout.pm = offset; offset += km1;
  layout.pem = offset; offset += km1;
  layout.peln2 = offset; offset += km1;
  layout.pk3_tmp = offset; offset += km1;
  layout.gm = offset; offset += km1;
  layout.cp = offset; offset += km1;
  layout.dz = offset; offset += km1;
  layout.w2 = offset; offset += km1;
  layout.peg = offset; offset += km1;
  layout.pelng = offset; offset += km1;

  // Solver work arrays (km+1 each)
  layout.pe2 = offset; offset += km1;
  layout.aa = offset; offset += km1;
  layout.bb = offset; offset += km1;
  layout.dd = offset; offset += km1;
  layout.w1 = offset; offset += km1;
  layout.wk = offset; offset += km1;
  layout.g_rat = offset; offset += km1;
  layout.gam = offset; offset += km1;
  layout.pp = offset; offset += km1;
  layout.bet = offset; offset += km1;
  layout.p1 = offset; offset += km1;

  // RIM_2D characteristic arrays (km+2 each for surface reflection)
  layout.dts = offset; offset += km2;
  layout.pf1 = offset; offset += km2;
  layout.r_lo = offset; offset += km2;
  layout.r_hi = offset; offset += km2;
  layout.m_bot = offset; offset += km2;
  layout.m_top = offset; offset += km2;
  layout.r_bot = offset; offset += km2;
  layout.r_top = offset; offset += km2;
  layout.wbar = offset; offset += km2;
  layout.pbar = offset; offset += km2;
  layout.cm = offset; offset += km2;
  layout.wc = offset; offset += km2;
  layout.dm_tmp = offset; offset += km2;
  layout.wm = offset; offset += km2;
  layout.dz_tmp = offset; offset += km2;

  layout.total_scalars = offset;
  return layout;
}

// Workspace sizing query for vertical solvers.
// Returns required bytes per horizontal column for the complete Riem_Solver3.
// Total workspace = columns * vertical_workspace_per_column(km).
// Note: Uses sizeof(double) for R8; for R4 use sizeof(float).
template <class T>
inline uint64_t vertical_workspace_per_column(int km) noexcept {
  const auto layout = compute_workspace_layout(km);
  return static_cast<uint64_t>(layout.total_scalars) * sizeof(T);
}

} // namespace fv3::gpu::vertical

#undef FV3_VERTICAL_HD
