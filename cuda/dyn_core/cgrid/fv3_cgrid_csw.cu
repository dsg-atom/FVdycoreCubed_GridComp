// Authoritative source: model/sw_core.F90:c_sw (lines 103-512)
// C-grid shallow water predictor: transport, KE, vorticity, and C-wind update.
//
// This translation assumes d2a2c_vect has already populated ua,va,uc,vc,ut,vt.
// Corner fills (fill_4corners, fill2_4corners) are invoked via corner_service
// by the caller at the appropriate phases.

#include "fv3_cgrid_csw.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// Phase 1: Scale ut/vt by sin_sg and compute first-order scalar fluxes.
// Lines 183-310 in Fortran.
// Note: Corner fills for delp/pt/w happen before this via corner_service.

// ut scaling (lines 183-191)
__global__ void csw_scale_ut(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 2 || j > a.bounds.je + 1) return;

  fv_real ut_ij = a.ut(i, j, k);
  // sin_sg indices: Fortran 3->2, 1->0 in C
  if (ut_ij > fv_real(0)) {
    ut_ij = a.dt2 * ut_ij * a.dy(i, j) * a.sin_sg(i - 1, j, 2);  // sin_sg(i-1,j,3)
  } else {
    ut_ij = a.dt2 * ut_ij * a.dy(i, j) * a.sin_sg(i, j, 0);      // sin_sg(i,j,1)
  }
  a.ut(i, j, k) = ut_ij;
}

// vt scaling (lines 192-200)
__global__ void csw_scale_vt(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 2) return;

  fv_real vt_ij = a.vt(i, j, k);
  // sin_sg indices: Fortran 4->3, 2->1 in C
  if (vt_ij > fv_real(0)) {
    vt_ij = a.dt2 * vt_ij * a.dx(i, j) * a.sin_sg(i, j - 1, 3);  // sin_sg(i,j-1,4)
  } else {
    vt_ij = a.dt2 * vt_ij * a.dx(i, j) * a.sin_sg(i, j, 1);      // sin_sg(i,j,2)
  }
  a.vt(i, j, k) = vt_ij;
}

// Nonhydrostatic x-fluxes (lines 238-254)
__global__ void csw_x_flux_nh(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 2 || j > a.bounds.je + 1) return;

  const fv_real ut_ij = a.ut(i, j, k);
  fv_real fx1_val, fx_val, fx2_val;

  if (ut_ij > fv_real(0)) {
    fx1_val = a.delp(i - 1, j, k);
    fx_val = a.pt(i - 1, j, k);
    fx2_val = a.w(i - 1, j, k);
  } else {
    fx1_val = a.delp(i, j, k);
    fx_val = a.pt(i, j, k);
    fx2_val = a.w(i, j, k);
  }

  fx1_val = ut_ij * fx1_val;
  fx_val = fx1_val * fx_val;
  fx2_val = fx1_val * fx2_val;

  a.fx1(i, j, k) = fx1_val;
  a.fx(i, j, k) = fx_val;
  a.fx2(i, j, k) = fx2_val;
}

// Nonhydrostatic y-fluxes (lines 285-300)
__global__ void csw_y_flux_nh(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 2) return;

  const fv_real vt_ij = a.vt(i, j, k);
  fv_real fy1_val, fy_val, fy2_val;

  if (vt_ij > fv_real(0)) {
    fy1_val = a.delp(i, j - 1, k);
    fy_val = a.pt(i, j - 1, k);
    fy2_val = a.w(i, j - 1, k);
  } else {
    fy1_val = a.delp(i, j, k);
    fy_val = a.pt(i, j, k);
    fy2_val = a.w(i, j, k);
  }

  fy1_val = vt_ij * fy1_val;
  fy_val = fy1_val * fy_val;
  fy2_val = fy1_val * fy2_val;

  a.fy1(i, j, k) = fy1_val;
  a.fy(i, j, k) = fy_val;
  a.fy2(i, j, k) = fy2_val;
}

// Compute delpc, ptc, wc (lines 301-310)
__global__ void csw_scalars_nh(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  const fv_real rarea_ij = a.rarea(i, j);
  const fv_real delp_ij = a.delp(i, j, k);
  const fv_real pt_ij = a.pt(i, j, k);
  const fv_real w_ij = a.w(i, j, k);

  const fv_real flux_div_x = a.fx1(i, j, k) - a.fx1(i + 1, j, k);
  const fv_real flux_div_y = a.fy1(i, j, k) - a.fy1(i, j + 1, k);
  const fv_real delpc_ij = delp_ij + (flux_div_x + flux_div_y) * rarea_ij;

  const fv_real pt_flux_x = a.fx(i, j, k) - a.fx(i + 1, j, k);
  const fv_real pt_flux_y = a.fy(i, j, k) - a.fy(i, j + 1, k);
  const fv_real ptc_ij = (pt_ij * delp_ij + (pt_flux_x + pt_flux_y) * rarea_ij) / delpc_ij;

  const fv_real w_flux_x = a.fx2(i, j, k) - a.fx2(i + 1, j, k);
  const fv_real w_flux_y = a.fy2(i, j, k) - a.fy2(i, j + 1, k);
  const fv_real wc_ij = (w_ij * delp_ij + (w_flux_x + w_flux_y) * rarea_ij) / delpc_ij;

  a.delpc(i, j, k) = delpc_ij;
  a.ptc(i, j, k) = ptc_ij;
  a.wc(i, j, k) = wc_ij;
}

// KE computation (lines 321-391)
// Cubed-sphere version with edge handling
__global__ void csw_ke_cubed(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  fv_real ke_val, vort_val;

  if (a.nested || a.grid_type >= 3) {
    // Simple case
    if (a.ua(i, j, k) > fv_real(0)) {
      ke_val = a.uc(i, j, k);
    } else {
      ke_val = a.uc(i + 1, j, k);
    }
    if (a.va(i, j, k) > fv_real(0)) {
      vort_val = a.vc(i, j, k);
    } else {
      vort_val = a.vc(i, j + 1, k);
    }
  } else {
    // Cubed-sphere: edge handling for KE
    const int32_t npx = a.npx;
    const int32_t npy = a.npy;

    // u-component (ke)
    if (a.ua(i, j, k) > fv_real(0)) {
      if (i == 1) {
        ke_val = a.uc(1, j, k) * a.sin_sg(1, j, 0) + a.v(1, j, k) * a.cos_sg(1, j, 0);
      } else if (i == npx) {
        ke_val = a.uc(npx, j, k) * a.sin_sg(npx, j, 0) + a.v(npx, j, k) * a.cos_sg(npx, j, 0);
      } else {
        ke_val = a.uc(i, j, k);
      }
    } else {
      if (i == 0) {
        ke_val = a.uc(1, j, k) * a.sin_sg(0, j, 2) + a.v(1, j, k) * a.cos_sg(0, j, 2);
      } else if (i == npx - 1) {
        ke_val = a.uc(npx, j, k) * a.sin_sg(npx - 1, j, 2) + a.v(npx, j, k) * a.cos_sg(npx - 1, j, 2);
      } else {
        ke_val = a.uc(i + 1, j, k);
      }
    }

    // v-component (vort)
    if (a.va(i, j, k) > fv_real(0)) {
      if (j == 1) {
        vort_val = a.vc(i, 1, k) * a.sin_sg(i, 1, 1) + a.u(i, 1, k) * a.cos_sg(i, 1, 1);
      } else if (j == npy) {
        vort_val = a.vc(i, npy, k) * a.sin_sg(i, npy, 1) + a.u(i, npy, k) * a.cos_sg(i, npy, 1);
      } else {
        vort_val = a.vc(i, j, k);
      }
    } else {
      if (j == 0) {
        vort_val = a.vc(i, 1, k) * a.sin_sg(i, 0, 3) + a.u(i, 1, k) * a.cos_sg(i, 0, 3);
      } else if (j == npy - 1) {
        vort_val = a.vc(i, npy, k) * a.sin_sg(i, npy - 1, 3) + a.u(i, npy, k) * a.cos_sg(i, npy - 1, 3);
      } else {
        vort_val = a.vc(i, j + 1, k);
      }
    }
  }

  // dt4 = 0.5*dt2
  const fv_real dt4 = fv_real(0.5) * a.dt2;
  a.ke(i, j, k) = dt4 * (a.ua(i, j, k) * ke_val + a.va(i, j, k) * vort_val);
}

// C-grid circulation: fx = uc*dxc, fy = vc*dyc (lines 396-406)
__global__ void csw_circulation(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  // fx: (is:ie+1, js-1:je+1)
  if (i >= a.bounds.is && i <= a.bounds.ie + 1 &&
      j >= a.bounds.js - 1 && j <= a.bounds.je + 1) {
    a.fx(i, j, k) = a.uc(i, j, k) * a.dxc(i, j);
  }

  // fy: (is-1:ie+1, js:je+1)
  if (i >= a.bounds.is - 1 && i <= a.bounds.ie + 1 &&
      j >= a.bounds.js && j <= a.bounds.je + 1) {
    a.fy(i, j, k) = a.vc(i, j, k) * a.dyc(i, j);
  }
}

// Vorticity: fx(i,j-1) - fx(i,j) - fy(i-1,j) + fy(i,j) (lines 408-412)
__global__ void csw_vorticity(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  a.vort(i, j, k) = a.fx(i, j - 1, k) - a.fx(i, j, k) -
                    a.fy(i - 1, j, k) + a.fy(i, j, k);
}

// Corner vorticity corrections (lines 415-418)
__global__ void csw_vort_corners(CSwArgs a, int64_t k) {
  // Single-threaded kernel for corner corrections
  if (threadIdx.x != 0) return;

  const int32_t npx = a.npx;
  const int32_t npy = a.npy;

  if (a.sw_corner) a.vort(1, 1, k) += a.fy(0, 1, k);
  if (a.se_corner) a.vort(npx, 1, k) -= a.fy(npx, 1, k);
  if (a.ne_corner) a.vort(npx, npy, k) -= a.fy(npx, npy, k);
  if (a.nw_corner) a.vort(1, npy, k) += a.fy(0, npy, k);
}

// Absolute vorticity: fC + rarea_c * vort (lines 423-427)
__global__ void csw_abs_vorticity(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 1) return;

  a.vort(i, j, k) = a.f_c(i, j) + a.rarea_c(i, j) * a.vort(i, j, k);
}

// Vorticity flux transport: fy1 and fx1 for vort transport (lines 438-498)
__global__ void csw_vort_transport_cubed(CSwArgs a, int64_t k) {
  // fy1 for j in [js, je], i in [is, ie+1]
  {
    const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= a.bounds.ie + 1 && j <= a.bounds.je) {
      fv_real fy1_val;
      const int32_t npx = a.npx;

      if (a.nested || a.grid_type >= 3) {
        fy1_val = a.dt2 * (a.v(i, j, k) - a.uc(i, j, k) * a.cosa_u(i, j)) / a.sina_u(i, j);
      } else {
        if (i == 1 || i == npx) {
          fy1_val = a.dt2 * a.v(i, j, k);
        } else {
          fy1_val = a.dt2 * (a.v(i, j, k) - a.uc(i, j, k) * a.cosa_u(i, j)) / a.sina_u(i, j);
        }
      }

      fv_real fy_val;
      if (fy1_val > fv_real(0)) {
        fy_val = a.vort(i, j, k);
      } else {
        fy_val = a.vort(i, j + 1, k);
      }

      a.fy1(i, j, k) = fy1_val;
      a.fy(i, j, k) = fy_val;
    }
  }
}

__global__ void csw_vort_transport_fx_cubed(CSwArgs a, int64_t k) {
  // fx1 for i in [is, ie], j in [js, je+1]
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie || j > a.bounds.je + 1) return;

  fv_real fx1_val;
  const int32_t npy = a.npy;

  if (a.nested || a.grid_type >= 3) {
    fx1_val = a.dt2 * (a.u(i, j, k) - a.vc(i, j, k) * a.cosa_v(i, j)) / a.sina_v(i, j);
  } else {
    if (j == 1 || j == npy) {
      fx1_val = a.dt2 * a.u(i, j, k);
    } else {
      fx1_val = a.dt2 * (a.u(i, j, k) - a.vc(i, j, k) * a.cosa_v(i, j)) / a.sina_v(i, j);
    }
  }

  fv_real fx_val;
  if (fx1_val > fv_real(0)) {
    fx_val = a.vort(i, j, k);
  } else {
    fx_val = a.vort(i + 1, j, k);
  }

  a.fx1(i, j, k) = fx1_val;
  a.fx(i, j, k) = fx_val;
}

// C-wind update (lines 500-510)
__global__ void csw_cwind_update(CSwArgs a, int64_t k) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;

  // uc update: i in [is, ie+1], j in [js, je]
  if (i <= a.bounds.ie + 1 && j <= a.bounds.je) {
    a.uc(i, j, k) = a.uc(i, j, k) + a.fy1(i, j, k) * a.fy(i, j, k) +
                    a.rdxc(i, j) * (a.ke(i - 1, j, k) - a.ke(i, j, k));
  }

  // vc update: i in [is, ie], j in [js, je+1]
  if (i <= a.bounds.ie && j <= a.bounds.je + 1) {
    a.vc(i, j, k) = a.vc(i, j, k) - a.fx1(i, j, k) * a.fx(i, j, k) +
                    a.rdyc(i, j) * (a.ke(i, j - 1, k) - a.ke(i, j, k));
  }
}

}  // namespace

fv3_gpu_status launch_csw_fluxes_level(const CSwArgs &a, int64_t k, void *stream) {
  if (!stream) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 4);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 4);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // Scale ut/vt
  csw_scale_ut<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  csw_scale_vt<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Nonhydrostatic fluxes and scalars
  if (!a.hydrostatic) {
    csw_x_flux_nh<<<grid, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    csw_y_flux_nh<<<grid, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    csw_scalars_nh<<<grid, block, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  }

  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_csw_ke_vort_level(const CSwArgs &a, int64_t k, void *stream) {
  if (!stream) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // KE computation
  csw_ke_cubed<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Circulation
  csw_circulation<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Vorticity
  const auto nx2 = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 2);
  const auto ny2 = static_cast<unsigned>(a.bounds.je - a.bounds.js + 2);
  dim3 grid2((nx2 + block.x - 1) / block.x, (ny2 + block.y - 1) / block.y);
  csw_vorticity<<<grid2, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Corner corrections
  csw_vort_corners<<<1, 1, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Absolute vorticity
  csw_abs_vorticity<<<grid2, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Vorticity transport
  csw_vort_transport_cubed<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  csw_vort_transport_fx_cubed<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_csw_cwind_update_level(const CSwArgs &a, int64_t k, void *stream) {
  if (!stream) return FV3_GPU_INVALID_ARGUMENT;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 2);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 2);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  csw_cwind_update<<<grid, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

}  // namespace fv3::gpu::cgrid
