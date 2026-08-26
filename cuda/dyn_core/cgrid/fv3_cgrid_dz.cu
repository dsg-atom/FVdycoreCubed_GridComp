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

// Finalize: monotonicity then ws output (nh_utils.F90:195-205)
//
// Milestone 3A.9 semantics repair. The previous body had THREE independent
// inversions against the authoritative update_dz_c closure -- it had been
// written with update_dz_d's (nh_utils.F90:310-321) closure instead:
//
//   1. ws was computed BEFORE the monotonicity clamp. Fortran clamps first
//      (:196-200) and only then forms ws from the CLAMPED gz(km+1) (:202-204),
//      so ws must see the post-clamp surface height.
//   2. The sweep ran bottom-up (k=km-1 down to 0). Fortran sweeps TOP-DOWN,
//      `do k=2, km+1`, so each level is limited against the level ABOVE it
//      after that level has itself been limited.
//   3. The sense was `max(gz(k), gz(k+1) + dz_min)` -- a floor pushing layers
//      up from below. Fortran applies `min( gz(k), gz(k-1) - dz_min )`, a
//      ceiling that enforces a minimum layer thickness measured DOWNWARD from
//      the level above. gz here is geopotential, which DECREASES with k.
//
// update_dz_d's tail legitimately does the opposite (ws from the unclamped
// zh(km+1), then a bottom-up `max` floor) -- see
// vertical/fv3_vertical_kernels.cu:height_boundary_v2_kernel, which is correct
// for that routine. The two closures are genuinely different and must stay so.
//
// Numerical semantics changed: NO -- restores authoritative nh_utils.F90:195-205.
// (The prior code was numerically wrong, not a validated alternative; nothing
// had ever executed it, because launch_cgrid_predictor_v3 was a fail-stub.)
//
// k index convention in this file: CUDA index = Fortran level - 1, so Fortran
// gz(:,:,km+1) is a.gz(i,j,km) and the Fortran k=2..km+1 loop is kk=1..km.
__global__ void update_dz_c_finalize(UpdateDzCArgs a) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  const int32_t km = a.km;

  // :196-200  do k=2, km+1 : gz(k) = min( gz(k), gz(k-1) - dz_min )
  for (int64_t kk = 1; kk <= km; ++kk) {
    a.gz(i, j, kk) = min(a.gz(i, j, kk), a.gz(i, j, kk - 1) - a.dz_min);
  }

  // :202-204  ws = ( zs - gz(km+1) ) * rdt, using the CLAMPED gz(km+1)
  const fv_real rdt = fv_real(1) / a.dt;
  a.ws(i, j) = (a.zs(i, j) - a.gz(i, j, km)) * rdt;
}

}  // namespace

fv3_gpu_status launch_update_dz_c_interface(const UpdateDzCArgs &a,
                                            const CornerFillArgs &corners,
                                            int64_t k, void *stream) {
  if (!stream || k < 1 || k > a.km + 1) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 5);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 5);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // Compute interface winds (:115-154)
  update_dz_c_winds<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Copy gz to gz2 (:156-160)
  update_dz_c_copy_gz<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  /* :162  if (grid_type < 3) call fill_4corners(gz2, 1, ...)
   *
   * Milestone 3A.9: previously a comment telling the caller to do this, which
   * made a numerically incomplete level body callable. The fill has to happen
   * here, between the gz2 copy and the x flux, and it has to happen once per
   * level, so it belongs inside this launcher. */
  if (a.grid_type < 3) {
    const fv3_gpu_status s =
        launch_fill_4corners(corners, a.gz2, kCornerFillDirX, stream);
    if (s != FV3_GPU_SUCCESS) return s;
  }

  // X flux (:163-172)
  update_dz_c_x_flux<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // :174  if (grid_type < 3) call fill_4corners(gz2, 2, ...)
  if (a.grid_type < 3) {
    const fv3_gpu_status s =
        launch_fill_4corners(corners, a.gz2, kCornerFillDirY, stream);
    if (s != FV3_GPU_SUCCESS) return s;
  }

  // Y flux (:175-184)
  update_dz_c_y_flux<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Update gz (:186-191)
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
