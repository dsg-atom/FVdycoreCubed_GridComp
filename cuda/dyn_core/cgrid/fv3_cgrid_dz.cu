// Authoritative source: model/nh_utils.F90:update_dz_c (lines 70-207)
// Interface geopotential height transport with upwind flux and monotonicity.
//
// Corner fills for gz2 must be invoked via corner_service by the caller
// between x and y flux phases.

#include "fv3_cgrid_dz.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// Compute interface wind xfx, yfx for level k (lines 108-154)
__global__ void update_dz_c_winds(UpdateDzCArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  const int32_t km = a.km;

  // Precompute ratios
  fv_real top_ratio, bot_ratio;
  if (k == 1) {
    top_ratio = a.dp0(0) / (a.dp0(0) + a.dp0(1));  // dp0(1)/(dp0(1)+dp0(2))
  }
  if (k == km + 1) {
    bot_ratio = a.dp0(km - 1) / (a.dp0(km - 2) + a.dp0(km - 1));  // dp0(km)/(dp0(km-1)+dp0(km))
  }

  // xfx: (is-1:ie+2, js-1:je+1)
  if (i >= a.bounds.is - 1 && i <= a.bounds.ie + 2 &&
      j >= a.bounds.js - 1 && j <= a.bounds.je + 1) {
    fv_real xfx_val;
    if (k == 1) {
      // Top extrapolation
      xfx_val = a.ut(i, j, 0) + (a.ut(i, j, 0) - a.ut(i, j, 1)) * top_ratio;
    } else if (k == km + 1) {
      // Bottom extrapolation
      xfx_val = a.ut(i, j, km - 1) + (a.ut(i, j, km - 1) - a.ut(i, j, km - 2)) * bot_ratio;
    } else {
      // Interior interpolation
      const fv_real int_ratio = fv_real(1) / (a.dp0(k - 2) + a.dp0(k - 1));
      xfx_val = (a.dp0(k - 1) * a.ut(i, j, k - 2) + a.dp0(k - 2) * a.ut(i, j, k - 1)) * int_ratio;
    }
    a.xfx(i, j, k - 1) = xfx_val;
  }

  // yfx: (is-1:ie+1, js-1:je+2)
  if (i >= a.bounds.is - 1 && i <= a.bounds.ie + 1 &&
      j >= a.bounds.js - 1 && j <= a.bounds.je + 2) {
    fv_real yfx_val;
    if (k == 1) {
      yfx_val = a.vt(i, j, 0) + (a.vt(i, j, 0) - a.vt(i, j, 1)) * top_ratio;
    } else if (k == km + 1) {
      yfx_val = a.vt(i, j, km - 1) + (a.vt(i, j, km - 1) - a.vt(i, j, km - 2)) * bot_ratio;
    } else {
      const fv_real int_ratio = fv_real(1) / (a.dp0(k - 2) + a.dp0(k - 1));
      yfx_val = (a.dp0(k - 1) * a.vt(i, j, k - 2) + a.dp0(k - 2) * a.vt(i, j, k - 1)) * int_ratio;
    }
    a.yfx(i, j, k - 1) = yfx_val;
  }
}

// Copy gz to gz2 (lines 156-160)
__global__ void update_dz_c_copy_gz(UpdateDzCArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x - 1;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y - 1;

  // ng is assumed from bounds; copy full halo region
  if (i >= a.bounds.is - 3 && i <= a.bounds.ie + 3 &&
      j >= a.bounds.js - 3 && j <= a.bounds.je + 3) {
    a.gz2(i, j) = a.gz(i, j, k - 1);  // k is 1-based interface, k-1 for 0-based
  }
}

// X-direction upwind flux (lines 163-172)
// Requires gz2 corner fill for direction 1 before this
__global__ void update_dz_c_x_flux(UpdateDzCArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 2 || j > a.bounds.je + 1) return;

  const fv_real xfx_ij = a.xfx(i, j, k - 1);
  fv_real fx_val;
  if (xfx_ij > fv_real(0)) {
    fx_val = a.gz2(i - 1, j);
  } else {
    fx_val = a.gz2(i, j);
  }
  a.fx(i, j, k - 1) = xfx_ij * fx_val;
}

// Y-direction upwind flux (lines 175-184)
// Requires gz2 corner fill for direction 2 before this
__global__ void update_dz_c_y_flux(UpdateDzCArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 2) return;

  const fv_real yfx_ij = a.yfx(i, j, k - 1);
  fv_real fy_val;
  if (yfx_ij > fv_real(0)) {
    fy_val = a.gz2(i, j - 1);
  } else {
    fy_val = a.gz2(i, j);
  }
  a.fy(i, j, k - 1) = yfx_ij * fy_val;
}

// Update gz (lines 186-191)
__global__ void update_dz_c_update_gz(UpdateDzCArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  const fv_real area_ij = a.area(i, j);
  const fv_real gz2_ij = a.gz2(i, j);
  const fv_real fx_diff = a.fx(i, j, k - 1) - a.fx(i + 1, j, k - 1);
  const fv_real fy_diff = a.fy(i, j, k - 1) - a.fy(i, j + 1, k - 1);
  const fv_real xfx_diff = a.xfx(i, j, k - 1) - a.xfx(i + 1, j, k - 1);
  const fv_real yfx_diff = a.yfx(i, j, k - 1) - a.yfx(i, j + 1, k - 1);

  a.gz(i, j, k - 1) = (gz2_ij * area_ij + fx_diff + fy_diff) /
                      (area_ij + xfx_diff + yfx_diff);
}

// Finalize: ws output and monotonicity (lines 196-205)
__global__ void update_dz_c_finalize(UpdateDzCArgs a) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  const fv_real rdt = fv_real(1) / a.dt;
  const int32_t km = a.km;

  // ws = (zs - gz(:,:,km+1)) * rdt
  a.ws(i, j) = (a.zs(i, j) - a.gz(i, j, km)) * rdt;

  // Monotonicity: gz(k) = max(gz(k), gz(k+1) + dz_min)
  for (int64_t kk = km - 1; kk >= 0; --kk) {
    a.gz(i, j, kk) = max(a.gz(i, j, kk), a.gz(i, j, kk + 1) + a.dz_min);
  }
}

}  // namespace

fv3_gpu_status launch_update_dz_c_interface(const UpdateDzCArgs &a, int64_t k,
                                            void *stream) {
  if (!stream || k < 1 || k > a.km + 1) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 5);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 5);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // Compute interface winds
  update_dz_c_winds<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Copy gz to gz2
  update_dz_c_copy_gz<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Note: caller must invoke corner_service for gz2 direction 1 here

  // X flux
  update_dz_c_x_flux<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Note: caller must invoke corner_service for gz2 direction 2 here

  // Y flux
  update_dz_c_y_flux<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Update gz
  const auto nx2 = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3);
  const auto ny2 = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);
  dim3 grid2((nx2 + block.x - 1) / block.x, (ny2 + block.y - 1) / block.y);
  update_dz_c_update_gz<<<grid2, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_update_dz_c_finalize(const UpdateDzCArgs &a, void *stream) {
  if (!stream) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  update_dz_c_finalize<<<grid, block, 0, cuda_stream>>>(a);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

}  // namespace fv3::gpu::cgrid
