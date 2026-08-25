#pragma once

// Device-side primitives shared by every WP-B CUDA translation unit.
//
// This header exists so that Milestone 3A.6's momentum transport
// (dgrid/wp_b_momentum_cuda.cu) can consume the SAME definitions Milestone
// 3A.4/3A.5's transport (dgrid/wp_b_cuda.cu) uses, instead of re-typing them.
// Everything here was factored out of wp_b_cuda.cu unchanged.
//
// It declares __device__ functions, so it must only be included from a .cu
// translation unit. wp_b_cuda.cuh stays host-parseable because
// dgrid/wp_b_contract_cuda.cpp includes it.

#include "dgrid/wp_b_cuda.cuh"

#include <cmath>
#include <cstdint>

namespace fv3::gpu::dgrid {

// tp_core.F90:89-96 PPM reconstruction constants. sw_core.F90's xtp_u
// (:1950) and ytp_v (:2320) use the identical p1/p2 interior weights and the
// identical c1/c2/c3 one-sided edge weights, so these are shared verbatim
// rather than restated per work-package slice.
__device__ __forceinline__ fv_real ppm_p1() { return fv_real(7.0 / 12.0); }
__device__ __forceinline__ fv_real ppm_p2() { return fv_real(-1.0 / 12.0); }
__device__ __forceinline__ fv_real ppm_c1() { return fv_real(-2.0 / 14.0); }
__device__ __forceinline__ fv_real ppm_c2() { return fv_real(11.0 / 14.0); }
__device__ __forceinline__ fv_real ppm_c3() { return fv_real(5.0 / 14.0); }

// nh_utils.F90:280 damping threshold.
__device__ __forceinline__ fv_real damp_threshold() { return fv_real(1.0e-5); }

struct Launch2D {
  dim3 block, grid;
};

inline Launch2D plan2d(int64_t ni, int64_t nj, int64_t nk) {
  Launch2D p;
  p.block = dim3(16, 16, 1);
  p.grid = dim3(static_cast<unsigned>((ni + 15) / 16),
                static_cast<unsigned>((nj + 15) / 16), static_cast<unsigned>(nk));
  return p;
}

inline fv3_gpu_status launch_status() {
  return cudaPeekAtLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_INTERNAL_ERROR;
}

// --- per-level activation ------------------------------------------------
__device__ __forceinline__ bool level_active(const LevelFilter &f, int64_t k) {
  if (f.mode == 0) return true;
  const int64_t kk = k < f.npz ? k : f.npz - 1;   // damp(km+1) = damp(km)
  const bool damped = f.damp(kk) > damp_threshold();
  if (f.mode == 1) return damped;
  if (f.mode == 2) return !damped;
  const int64_t nord = static_cast<int64_t>(llrint(static_cast<double>(f.ndif(kk))));
  return damped && nord == static_cast<int64_t>(f.nord_sel);
}

// tp_core.F90:532-541 / 936-950 and sw_core.F90:2144-2156 / 2551-2566: at the
// two cells adjacent to a panel seam the mord=6 smoothness test degrades to
// the mord=5 sign test. xppm/yppm and xtp_u/ytp_v apply the identical WMP edge
// fix over the identical index sets, so one predicate serves all four.
__device__ __forceinline__ bool smt5_from(fv_real bl, fv_real br, fv_real b0,
                                          bool seam_cell) {
  if (seam_cell) return bl * br < fv_real(0);
  return fv_real(3) * fabs(b0) < fabs(bl - br);
}

__device__ __forceinline__ bool seam_cell_x(int64_t i, const TransportBounds &b) {
  if (!(b.cubed_sphere && !b.nested)) return false;
  if (b.west_edge && (i == 0 || i == 1)) return true;
  if (b.east_edge && (i == b.npx - 1 || i == b.npx)) return true;
  return false;
}

__device__ __forceinline__ bool seam_cell_y(int64_t j, const TransportBounds &b) {
  if (!(b.cubed_sphere && !b.nested)) return false;
  if (b.south_edge && (j == 0 || j == 1)) return true;
  if (b.north_edge && (j == b.npy - 1 || j == b.npy)) return true;
  return false;
}

inline View3<const fv_real> as_const(const View3<fv_real> &v) {
  return View3<const fv_real>{v.data, v.lo_i, v.lo_j, v.lo_k, v.si, v.sj, v.sk};
}

} // namespace fv3::gpu::dgrid
