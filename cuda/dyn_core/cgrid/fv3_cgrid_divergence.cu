// Authoritative source: model/sw_core.F90:divergence_corner (lines 1694-1799)
// Corner flux divergence for nord > 0 damping on global cubed-sphere.
//
// The nested branch (divergence_corner_nest) is milestone-one unsupported.

#include "fv3_cgrid_divergence.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// Simple case: grid_type > 3 (Cartesian-like)
__global__ void divergence_corner_simple(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  // Compute uf (is-2:ie+2, js-1:je+2)
  if (i >= a.bounds.is - 2 && i <= a.bounds.ie + 2 &&
      j >= a.bounds.js - 1 && j <= a.bounds.je + 2) {
    a.uf(i, j, k) = a.u(i, j, k) * a.dyc(i, j);
  }

  // Compute vf (is-1:ie+2, js-2:je+2)
  if (i >= a.bounds.is - 1 && i <= a.bounds.ie + 2 &&
      j >= a.bounds.js - 2 && j <= a.bounds.je + 2) {
    a.vf(i, j, k) = a.v(i, j, k) * a.dxc(i, j);
  }
}

__global__ void divergence_corner_simple_divg(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 2 || j > a.bounds.je + 2) return;

  a.divg_d(i, j, k) = a.rarea_c(i, j) *
      (a.vf(i, j - 1, k) - a.vf(i, j, k) + a.uf(i - 1, j, k) - a.uf(i, j, k));
}

// Cubed-sphere case: grid_type <= 3, non-nested
// Phase 1: Compute uf with sin_sg/cos_sg correction
__global__ void divergence_corner_uf(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  // sin_sg indices: Fortran 1-9 maps to 0-8 in C
  // sin_sg(:,:,4) is index 3, sin_sg(:,:,2) is index 1
  const fv_real sin_sg_jm1_4 = a.sin_sg(i, j - 1, 3);
  const fv_real sin_sg_j_2 = a.sin_sg(i, j, 1);
  const fv_real sin_avg = fv_real(0.5) * (sin_sg_jm1_4 + sin_sg_j_2);

  if (j == 1 || j == a.npy) {
    // Edge rows: simple formula
    a.uf(i, j, k) = a.u(i, j, k) * a.dyc(i, j) * sin_avg;
  } else {
    // Interior: cos_sg correction
    const fv_real cos_sg_jm1_4 = a.cos_sg(i, j - 1, 3);
    const fv_real cos_sg_j_2 = a.cos_sg(i, j, 1);
    const fv_real va_avg = fv_real(0.25) * (a.va(i, j - 1, k) + a.va(i, j, k));
    const fv_real cos_avg = cos_sg_jm1_4 + cos_sg_j_2;

    a.uf(i, j, k) = (a.u(i, j, k) - va_avg * cos_avg) * a.dyc(i, j) * sin_avg;
  }
}

// Phase 2: Compute vf with sin_sg/cos_sg correction
__global__ void divergence_corner_vf(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  // is2 = max(2, is); ie1 = min(npx-1, ie+1) for non-nested
  const int64_t is2 = max(int64_t(2), a.bounds.is);
  const int64_t ie1 = min(int64_t(a.npx - 1), a.bounds.ie + 1);

  if (j > a.bounds.je + 1) return;

  // Interior points
  if (i >= is2 && i <= ie1) {
    // sin_sg indices: sin_sg(:,:,3) is index 2, sin_sg(:,:,1) is index 0
    const fv_real sin_sg_im1_3 = a.sin_sg(i - 1, j, 2);
    const fv_real sin_sg_i_1 = a.sin_sg(i, j, 0);
    const fv_real sin_avg = fv_real(0.5) * (sin_sg_im1_3 + sin_sg_i_1);

    const fv_real cos_sg_im1_3 = a.cos_sg(i - 1, j, 2);
    const fv_real cos_sg_i_1 = a.cos_sg(i, j, 0);
    const fv_real ua_avg = fv_real(0.25) * (a.ua(i - 1, j, k) + a.ua(i, j, k));
    const fv_real cos_avg = cos_sg_im1_3 + cos_sg_i_1;

    a.vf(i, j, k) = (a.v(i, j, k) - ua_avg * cos_avg) * a.dxc(i, j) * sin_avg;
  }
}

// Phase 2b: vf at west/east boundaries (i==1 and i==npx)
__global__ void divergence_corner_vf_edges(DivergenceCornerArgs a, int64_t k) {
  const int64_t j = a.bounds.js - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (j > a.bounds.je + 1) return;

  // West boundary: is == 1
  if (a.bounds.is == 1) {
    const fv_real sin_sg_0_3 = a.sin_sg(0, j, 2);
    const fv_real sin_sg_1_1 = a.sin_sg(1, j, 0);
    a.vf(1, j, k) = a.v(1, j, k) * a.dxc(1, j) * fv_real(0.5) * (sin_sg_0_3 + sin_sg_1_1);
  }

  // East boundary: ie+1 == npx
  if (a.bounds.ie + 1 == a.npx) {
    const int64_t npx = a.npx;
    const fv_real sin_sg_npm1_3 = a.sin_sg(npx - 1, j, 2);
    const fv_real sin_sg_np_1 = a.sin_sg(npx, j, 0);
    a.vf(npx, j, k) = a.v(npx, j, k) * a.dxc(npx, j) * fv_real(0.5) * (sin_sg_npm1_3 + sin_sg_np_1);
  }
}

// Phase 3: Compute divg_d from uf and vf
__global__ void divergence_corner_divg(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  a.divg_d(i, j, k) = a.vf(i, j - 1, k) - a.vf(i, j, k) +
                      a.uf(i - 1, j, k) - a.uf(i, j, k);
}

// Phase 4: Corner corrections and final scaling
__global__ void divergence_corner_finalize(DivergenceCornerArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  fv_real divg = a.divg_d(i, j, k);

  // Remove the extra term at the corners
  if (a.sw_corner && i == 1 && j == 1) {
    divg -= a.vf(1, 0, k);
  }
  if (a.se_corner && i == a.npx && j == 1) {
    divg -= a.vf(a.npx, 0, k);
  }
  if (a.ne_corner && i == a.npx && j == a.npy) {
    divg += a.vf(a.npx, a.npy, k);
  }
  if (a.nw_corner && i == 1 && j == a.npy) {
    divg += a.vf(1, a.npy, k);
  }

  a.divg_d(i, j, k) = a.rarea_c(i, j) * divg;
}

}  // namespace

fv3_gpu_status launch_divergence_corner_level(const DivergenceCornerArgs &a,
                                              int64_t k, void *stream) {
  if (!stream || !a.u.data || !a.v.data || !a.ua.data || !a.va.data ||
      !a.divg_d.data || !a.uf.data || !a.vf.data)
    return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  if (a.grid_type > 3) {
    // Simple Cartesian case
    const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 5);
    const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 5);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    divergence_corner_simple<<<grid, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    divergence_corner_simple_divg<<<grid, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  } else {
    // Cubed-sphere case
    const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3);
    const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 2);

    // Phase 1: uf
    dim3 grid_uf((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    divergence_corner_uf<<<grid_uf, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    // Phase 2: vf interior
    dim3 grid_vf((nx + block.x - 1) / block.x, ((ny + 1) + block.y - 1) / block.y);
    divergence_corner_vf<<<grid_vf, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    // Phase 2b: vf edges
    const auto nj_edge = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);
    divergence_corner_vf_edges<<<(nj_edge + 31) / 32, 32, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    // Phase 3: divg_d
    dim3 grid_divg(((nx - 1) + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    divergence_corner_divg<<<grid_divg, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    // Phase 4: corner corrections and scaling
    divergence_corner_finalize<<<grid_divg, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  }

  return FV3_GPU_SUCCESS;
}

}  // namespace fv3::gpu::cgrid
