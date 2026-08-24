#include "dgrid/wp_b_cuda.cuh"

#include <cmath>
#include <cstdint>

namespace fv3::gpu::dgrid {
namespace {

// PPM order-6 interior stencil weights, matching wp_b_kernels.hpp's
// fv3::gpu::dgrid::ppm namespace (p1 = 7/12, p2 = -1/12).
__device__ __forceinline__ fv_real ppm_p1() { return fv_real(7.0 / 12.0); }
__device__ __forceinline__ fv_real ppm_p2() { return fv_real(-1.0 / 12.0); }

// Direct translation of the PPM-6 face-flux arithmetic that
// fv_tp_2d_hord6 repeats inline for each of its four directional passes
// (wp_b_kernels.hpp:988-1022 [fy2], :1039-1073 [fx], :1079-1113 [fx2],
// :1129-1163 [fy]). Every pass evaluates this same 6-point stencil,
// (qm3..qp2) = q at (face-3 .. face+2) along the transport direction,
// centered on the face between "qm1" and "q0".
__device__ __forceinline__ fv_real ppm6_face_flux(int32_t hord, fv_real c, fv_real qm3,
                                                   fv_real qm2, fv_real qm1, fv_real q0,
                                                   fv_real qp1, fv_real qp2) {
  fv_real al_m1 = ppm_p1() * (qm2 + qm1) + ppm_p2() * (qm3 + q0);
  fv_real al_0 = ppm_p1() * (qm1 + q0) + ppm_p2() * (qm2 + qp1);
  fv_real al_p1 = ppm_p1() * (q0 + qp1) + ppm_p2() * (qm1 + qp2);

  if (hord < 0) {
    al_m1 = al_m1 > fv_real(0) ? al_m1 : fv_real(0);
    al_0 = al_0 > fv_real(0) ? al_0 : fv_real(0);
    al_p1 = al_p1 > fv_real(0) ? al_p1 : fv_real(0);
  }

  const fv_real bl_m1 = al_m1 - qm1;
  const fv_real br_m1 = al_0 - qm1;
  const fv_real b0_m1 = bl_m1 + br_m1;
  const fv_real bl_0 = al_0 - q0;
  const fv_real br_0 = al_p1 - q0;
  const fv_real b0_0 = bl_0 + br_0;
  const bool smt5_m1 = fv_real(3) * fabs(b0_m1) < fabs(bl_m1 - br_m1);
  const bool smt5_0 = fv_real(3) * fabs(b0_0) < fabs(bl_0 - br_0);

  fv_real fx1, donor;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (br_m1 - c * b0_m1);
    donor = qm1;
  } else {
    fx1 = (fv_real(1) + c) * (bl_0 + c * b0_0);
    donor = q0;
  }
  return (smt5_m1 || smt5_0) ? donor + fx1 : donor;
}

struct Launch2D {
  dim3 block, grid;
};

Launch2D plan2d(int64_t ni, int64_t nj, int64_t nk) {
  Launch2D p;
  p.block = dim3(16, 16, 1);
  p.grid = dim3(static_cast<unsigned>((ni + 15) / 16),
               static_cast<unsigned>((nj + 15) / 16), static_cast<unsigned>(nk));
  return p;
}

fv3_gpu_status launch_status() {
  return cudaPeekAtLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_INTERNAL_ERROR;
}

// --- ra_x / ra_y ---------------------------------------------------------
// New code (not a translation of an existing host function) -- see
// docs/agent_handoff/milestone_3a3_wp_b_cuda.md, "ra_x/ra_y derivation".
// Computed over the full extended [is-h,ie+h] x [js-h,je+h] box (a harmless
// superset of the [isd,ied]x[js,je] / [is,ie]x[jsd,jed] boxes fv_tp_2d_hord6
// actually reads ra_y/ra_x from).
__global__ void ra_xy_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j, int64_t hi_j,
                             int64_t npz, View3<const fv_real> xfx,
                             View3<const fv_real> yfx, View2<const fv_real> area,
                             View3<fv_real> ra_x, View3<fv_real> ra_y) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  const fv_real a = area(i, j);
  ra_x(i, j, k) = a + xfx(i, j, k) - xfx(i + 1, j, k);
  ra_y(i, j, k) = a + yfx(i, j, k) - yfx(i, j + 1, k);
}

// --- fv_tp_2d_hord6, pass 1: Y-direction on q_in -> fy2 (:983-1024) ------
__global__ void fy2_from_qin_kernel(int64_t lo_i, int64_t hi_i, int64_t js, int64_t je,
                                    int64_t npz, int32_t hord, View3<const fv_real> q_in,
                                    View3<const fv_real> cry, View3<fv_real> fy2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > je + 1 || k > npz) return;
  const fv_real c = cry(i, j, k);
  fy2(i, j, k) = ppm6_face_flux(hord, c, q_in(i, j - 3, k), q_in(i, j - 2, k),
                                q_in(i, j - 1, k), q_in(i, j, k), q_in(i, j + 1, k),
                                q_in(i, j + 2, k));
}

// --- fv_tp_2d_hord6, pass 2: q_work = (q_in*area + ra_y flux div) (:1026-1034)
__global__ void qwork_from_ray_kernel(int64_t lo_i, int64_t hi_i, int64_t js, int64_t je,
                                      int64_t npz, View3<const fv_real> q_in,
                                      View3<const fv_real> yfx, View3<fv_real> fy2,
                                      View2<const fv_real> area, View3<const fv_real> ra_y,
                                      View3<fv_real> q_work) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > je || k > npz) return;
  const fv_real fyy_j = yfx(i, j, k) * fy2(i, j, k);
  const fv_real fyy_jp1 = yfx(i, j + 1, k) * fy2(i, j + 1, k);
  q_work(i, j, k) = (q_in(i, j, k) * area(i, j) + fyy_j - fyy_jp1) / ra_y(i, j, k);
}

// --- fv_tp_2d_hord6, pass 3: X-direction on q_work -> fx (:1036-1074) ----
__global__ void fx_from_qwork_kernel(int64_t is, int64_t ie, int64_t js, int64_t je,
                                     int64_t npz, int32_t hord, View3<fv_real> q_work,
                                     View3<const fv_real> crx, View3<fv_real> fx) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie + 1 || j > je || k > npz) return;
  const fv_real c = crx(i, j, k);
  fx(i, j, k) = ppm6_face_flux(hord, c, q_work(i - 3, j, k), q_work(i - 2, j, k),
                               q_work(i - 1, j, k), q_work(i, j, k), q_work(i + 1, j, k),
                               q_work(i + 2, j, k));
}

// --- fv_tp_2d_hord6, pass 4: X-direction on original q_in -> fx2 (:1076-1114)
__global__ void fx2_from_qin_kernel(int64_t is, int64_t ie, int64_t lo_j, int64_t hi_j,
                                    int64_t npz, int32_t hord, View3<const fv_real> q_in,
                                    View3<const fv_real> crx, View3<fv_real> fx2) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie + 1 || j > hi_j || k > npz) return;
  const fv_real c = crx(i, j, k);
  fx2(i, j, k) = ppm6_face_flux(hord, c, q_in(i - 3, j, k), q_in(i - 2, j, k),
                                q_in(i - 1, j, k), q_in(i, j, k), q_in(i + 1, j, k),
                                q_in(i + 2, j, k));
}

// --- fv_tp_2d_hord6, pass 5: q_work = (q_in*area + ra_x flux div) (:1116-1124)
// Overwrites q_work from pass 2 -- safe because pass 3 (fx_from_qwork) has
// already consumed pass 2's q_work by the time this launches on the same
// stream (kernels on one stream execute in issued order).
__global__ void qwork_from_rax_kernel(int64_t is, int64_t ie, int64_t lo_j, int64_t hi_j,
                                      int64_t npz, View3<const fv_real> q_in,
                                      View3<const fv_real> xfx, View3<fv_real> fx2,
                                      View2<const fv_real> area, View3<const fv_real> ra_x,
                                      View3<fv_real> q_work) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie || j > hi_j || k > npz) return;
  const fv_real fxx_i = xfx(i, j, k) * fx2(i, j, k);
  const fv_real fxx_ip1 = xfx(i + 1, j, k) * fx2(i + 1, j, k);
  q_work(i, j, k) = (q_in(i, j, k) * area(i, j) + fxx_i - fxx_ip1) / ra_x(i, j, k);
}

// --- fv_tp_2d_hord6, pass 6: Y-direction on new q_work -> fy (:1126-1164)
__global__ void fy_from_qwork_kernel(int64_t is, int64_t ie, int64_t js, int64_t je,
                                     int64_t npz, int32_t hord, View3<fv_real> q_work,
                                     View3<const fv_real> cry, View3<fv_real> fy) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie || j > je + 1 || k > npz) return;
  const fv_real c = cry(i, j, k);
  fy(i, j, k) = ppm6_face_flux(hord, c, q_work(i, j - 3, k), q_work(i, j - 2, k),
                               q_work(i, j - 1, k), q_work(i, j, k), q_work(i, j + 1, k),
                               q_work(i, j + 2, k));
}

// --- fv_tp_2d_hord6, pass 7: flux averaging/weighting (:1166-1177) ------
__global__ void flux_weight_x_kernel(int64_t is, int64_t ie, int64_t js, int64_t je,
                                     int64_t npz, View3<fv_real> fx2,
                                     View3<const fv_real> xfx, View3<fv_real> fx) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie + 1 || j > je || k > npz) return;
  fx(i, j, k) = fv_real(0.5) * (fx(i, j, k) + fx2(i, j, k)) * xfx(i, j, k);
}

__global__ void flux_weight_y_kernel(int64_t is, int64_t ie, int64_t js, int64_t je,
                                     int64_t npz, View3<fv_real> fy2,
                                     View3<const fv_real> yfx, View3<fv_real> fy) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie || j > je + 1 || k > npz) return;
  fy(i, j, k) = fv_real(0.5) * (fy(i, j, k) + fy2(i, j, k)) * yfx(i, j, k);
}

// --- deln_flux_transport (has_mass = false path only; :1183-1273) -------
__global__ void deln_init_kernel(int64_t i1, int64_t i2, int64_t j1, int64_t j2,
                                 int64_t npz, fv_real damp_coef, View3<const fv_real> q,
                                 View3<fv_real> d2) {
  const int64_t i = i1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = j1 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > i2 || j > j2 || k > npz) return;
  d2(i, j, k) = damp_coef * q(i, j, k);
}

__global__ void deln_first_flux_x_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j,
                                         int64_t hi_j, int64_t npz, View3<fv_real> d2,
                                         View2<const fv_real> del6_v,
                                         View3<fv_real> fx2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  fx2(i, j, k) = del6_v(i, j) * (d2(i - 1, j, k) - d2(i, j, k));
}

__global__ void deln_first_flux_y_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j,
                                         int64_t hi_j, int64_t npz, View3<fv_real> d2,
                                         View2<const fv_real> del6_u,
                                         View3<fv_real> fy2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  fy2(i, j, k) = del6_u(i, j) * (d2(i, j - 1, k) - d2(i, j, k));
}

__global__ void deln_iter_div_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j,
                                     int64_t hi_j, int64_t npz, View3<fv_real> fx2,
                                     View3<fv_real> fy2, View2<const fv_real> rarea,
                                     View3<fv_real> d2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  d2(i, j, k) = (fx2(i, j, k) - fx2(i + 1, j, k) + fy2(i, j, k) - fy2(i, j + 1, k)) *
               rarea(i, j);
}

// Note the sign flip vs. deln_first_flux_{x,y}_kernel: d2(i,j)-d2(i-1,j)
// rather than d2(i-1,j)-d2(i,j) -- exact translation of wp_b_kernels.hpp:1238/1243.
__global__ void deln_iter_flux_x_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j,
                                        int64_t hi_j, int64_t npz, View3<fv_real> d2,
                                        View2<const fv_real> del6_v,
                                        View3<fv_real> fx2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  fx2(i, j, k) = del6_v(i, j) * (d2(i, j, k) - d2(i - 1, j, k));
}

__global__ void deln_iter_flux_y_kernel(int64_t lo_i, int64_t hi_i, int64_t lo_j,
                                        int64_t hi_j, int64_t npz, View3<fv_real> d2,
                                        View2<const fv_real> del6_u,
                                        View3<fv_real> fy2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k > npz) return;
  fy2(i, j, k) = del6_u(i, j) * (d2(i, j, k) - d2(i, j - 1, k));
}

__global__ void deln_add_no_mass_kernel(int64_t is, int64_t ie, int64_t js, int64_t je,
                                        int64_t npz, bool along_x, View3<fv_real> flux2,
                                        View3<fv_real> flux) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  const int64_t hi_i = along_x ? ie + 1 : ie;
  const int64_t hi_j = along_x ? je : je + 1;
  if (i > hi_i || j > hi_j || k > npz) return;
  flux(i, j, k) = flux(i, j, k) + flux2(i, j, k);
}

// --- zh flux-divergence update -------------------------------------------
// New code: zh has no existing host-side flux-application function to
// translate (transport_service_for_update_dz only produces fx/fy). Modeled
// directly on wp_b_kernels.hpp's apply_mass_and_scalar_flux mass_div formula
// (same file, same divergence pattern) -- see handoff.
__global__ void apply_height_flux_divergence_kernel(int64_t is, int64_t ie, int64_t js,
                                                     int64_t je, int64_t npz,
                                                     View3<const fv_real> fx,
                                                     View3<const fv_real> fy,
                                                     View2<const fv_real> rarea,
                                                     View3<fv_real> zh) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ie || j > je || k > npz) return;
  const fv_real div = (fx(i, j, k) - fx(i + 1, j, k) + fy(i, j, k) - fy(i, j + 1, k)) *
                      rarea(i, j);
  zh(i, j, k) = zh(i, j, k) + div;
}

} // namespace

fv3_gpu_status launch_ra_xy_cuda(const HeightTransportBounds &b,
                                 View3<const fv_real> xfx, View3<const fv_real> yfx,
                                 View2<const fv_real> area, View3<fv_real> ra_x,
                                 View3<fv_real> ra_y, void *stream) noexcept {
  const int64_t h = b.halo_width;
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;
  Launch2D p = plan2d(hi_i - lo_i + 1, hi_j - lo_j + 1, b.npz + 1);
  ra_xy_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      lo_i, hi_i, lo_j, hi_j, b.npz, xfx, yfx, area, ra_x, ra_y);
  return launch_status();
}

fv3_gpu_status launch_fv_tp_2d_hord6_cuda(
    const HeightTransportBounds &b, int32_t hord, View3<const fv_real> q_in,
    View3<const fv_real> crx, View3<const fv_real> cry, View3<const fv_real> xfx,
    View3<const fv_real> yfx, View2<const fv_real> area, View3<const fv_real> ra_x,
    View3<const fv_real> ra_y, View3<fv_real> fx, View3<fv_real> fy,
    View3<fv_real> q_work, View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npz = b.npz;
  const int64_t isd = is - 3, ied = ie + 3, jsd = js - 3, jed = je + 3;
  auto s = static_cast<cudaStream_t>(stream);

  {
    Launch2D p = plan2d(ied - isd + 1, je - js + 2, npz + 1);
    fy2_from_qin_kernel<<<p.grid, p.block, 0, s>>>(isd, ied, js, je, npz, hord, q_in, cry,
                                                   fy2);
  }
  {
    Launch2D p = plan2d(ied - isd + 1, je - js + 1, npz + 1);
    qwork_from_ray_kernel<<<p.grid, p.block, 0, s>>>(isd, ied, js, je, npz, q_in, yfx, fy2,
                                                     area, ra_y, q_work);
  }
  {
    Launch2D p = plan2d(ie - is + 2, je - js + 1, npz + 1);
    fx_from_qwork_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, hord, q_work, crx,
                                                    fx);
  }
  {
    Launch2D p = plan2d(ie - is + 2, jed - jsd + 1, npz + 1);
    fx2_from_qin_kernel<<<p.grid, p.block, 0, s>>>(is, ie, jsd, jed, npz, hord, q_in, crx,
                                                   fx2);
  }
  {
    Launch2D p = plan2d(ie - is + 1, jed - jsd + 1, npz + 1);
    qwork_from_rax_kernel<<<p.grid, p.block, 0, s>>>(is, ie, jsd, jed, npz, q_in, xfx, fx2,
                                                     area, ra_x, q_work);
  }
  {
    Launch2D p = plan2d(ie - is + 1, je - js + 2, npz + 1);
    fy_from_qwork_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, hord, q_work, cry,
                                                    fy);
  }
  {
    Launch2D p = plan2d(ie - is + 2, je - js + 1, npz + 1);
    flux_weight_x_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, fx2, xfx, fx);
  }
  {
    Launch2D p = plan2d(ie - is + 1, je - js + 2, npz + 1);
    flux_weight_y_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, fy2, yfx, fy);
  }
  return launch_status();
}

fv3_gpu_status launch_apply_height_flux_divergence_cuda(
    const HeightTransportBounds &b, View3<const fv_real> fx, View3<const fv_real> fy,
    View2<const fv_real> rarea, View3<fv_real> zh, void *stream) noexcept {
  Launch2D p = plan2d(b.ie - b.is + 1, b.je - b.js + 1, b.npz + 1);
  apply_height_flux_divergence_kernel<<<p.grid, p.block, 0,
                                        static_cast<cudaStream_t>(stream)>>>(
      b.is, b.ie, b.js, b.je, b.npz, fx, fy, rarea, zh);
  return launch_status();
}

fv3_gpu_status launch_deln_flux_transport_cuda(
    const HeightTransportBounds &b, int32_t nord, fv_real damp, fv_real da_min,
    View3<const fv_real> q, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> fx, View3<fv_real> fy, View3<fv_real> d2,
    View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npz = b.npz;
  auto s = static_cast<cudaStream_t>(stream);
  const fv_real damp_coef = std::pow(damp * da_min, static_cast<fv_real>(nord + 1));

  const int64_t i1 = is - 1 - nord, i2 = ie + 1 + nord;
  const int64_t j1 = js - 1 - nord, j2 = je + 1 + nord;
  {
    Launch2D p = plan2d(i2 - i1 + 1, j2 - j1 + 1, npz + 1);
    deln_init_kernel<<<p.grid, p.block, 0, s>>>(i1, i2, j1, j2, npz, damp_coef, q, d2);
  }
  {
    Launch2D p = plan2d(ie + nord + 1 - (is - nord) + 1, je + nord - (js - nord) + 1,
                        npz + 1);
    deln_first_flux_x_kernel<<<p.grid, p.block, 0, s>>>(is - nord, ie + nord + 1, js - nord,
                                                        je + nord, npz, d2, del6_v, fx2);
  }
  {
    Launch2D p = plan2d(ie + nord - (is - nord) + 1, je + nord + 1 - (js - nord) + 1,
                        npz + 1);
    deln_first_flux_y_kernel<<<p.grid, p.block, 0, s>>>(is - nord, ie + nord, js - nord,
                                                        je + nord + 1, npz, d2, del6_u, fy2);
  }

  for (int32_t n = 1; n <= nord; ++n) {
    const int32_t nt = nord - n;
    {
      Launch2D p = plan2d(ie + nt + 1 - (is - nt - 1) + 1, je + nt + 1 - (js - nt - 1) + 1,
                          npz + 1);
      deln_iter_div_kernel<<<p.grid, p.block, 0, s>>>(is - nt - 1, ie + nt + 1, js - nt - 1,
                                                       je + nt + 1, npz, fx2, fy2, rarea, d2);
    }
    {
      Launch2D p = plan2d(ie + nt + 1 - (is - nt) + 1, je + nt - (js - nt) + 1, npz + 1);
      deln_iter_flux_x_kernel<<<p.grid, p.block, 0, s>>>(is - nt, ie + nt + 1, js - nt,
                                                         je + nt, npz, d2, del6_v, fx2);
    }
    {
      Launch2D p = plan2d(ie + nt - (is - nt) + 1, je + nt + 1 - (js - nt) + 1, npz + 1);
      deln_iter_flux_y_kernel<<<p.grid, p.block, 0, s>>>(is - nt, ie + nt, js - nt,
                                                         je + nt + 1, npz, d2, del6_u, fy2);
    }
  }

  {
    Launch2D p = plan2d(ie - is + 2, je - js + 1, npz + 1);
    deln_add_no_mass_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, true, fx2, fx);
  }
  {
    Launch2D p = plan2d(ie - is + 1, je - js + 2, npz + 1);
    deln_add_no_mass_kernel<<<p.grid, p.block, 0, s>>>(is, ie, js, je, npz, false, fy2, fy);
  }
  return launch_status();
}

} // namespace fv3::gpu::dgrid
