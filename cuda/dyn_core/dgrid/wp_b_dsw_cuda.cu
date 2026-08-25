// Milestone 3A.8 / WP-B.5 -- the authoritative FV3 d_sw driver.
//
// See dgrid/wp_b_dsw_cuda.cuh for the ordered operation table (§1), the
// driver-owned kernel inventory (§2), the anti-dependence audit (§3), the
// per-level control map (§4), the flux-capacitor semantics (§5), the
// unsupported-configuration fences (§6) and the corner_edge_service disposition
// (§7). Every kernel below carries the sw_core.F90 line range it transcribes.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45.

#include "dgrid/wp_b_dsw_cuda.cuh"
#include "dgrid/wp_b_ppm_shared.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <initializer_list>

namespace fv3::gpu::dgrid {
namespace {

View1<const fv_real> as_const1(const View1<fv_real> &v) {
  return View1<const fv_real>{v.data, v.lo_i, v.si};
}

// Fortran's `x**(n+1)` with an INTEGER exponent is repeated multiplication, not
// exp(log). For the exponents that occur here (1, 2, 3) that distinction is
// observable in the last bits, so the coefficient kernel multiplies explicitly
// rather than calling pow().
__device__ __forceinline__ fv_real ipow(fv_real base, int32_t n) {
  fv_real r = fv_real(1);
  for (int32_t i = 0; i < n; ++i) r *= base;
  return r;
}

__device__ __forceinline__ int32_t round_nord(fv_real v) {
  return static_cast<int32_t>(llrint(static_cast<double>(v)));
}

// Maps a 2D thread to an inclusive Fortran box [i0,i1] x [j0,j1] with
// blockIdx.z carrying the level. Returns false when the thread is outside.
struct Box {
  int64_t i, j, k;
};
__device__ __forceinline__ bool in_box(int64_t i0, int64_t i1, int64_t j0, int64_t j1,
                                       Box &t) {
  t.i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
  t.j = j0 + blockIdx.y * blockDim.y + threadIdx.y;
  t.k = blockIdx.z;
  return t.i <= i1 && t.j <= j1;
}

inline Launch2D plan_box(int64_t i0, int64_t i1, int64_t j0, int64_t j1, int64_t nk) {
  return plan2d(i1 - i0 + 1, j1 - j0 + 1, nk);
}

// ===========================================================================
// K1 -- contravariant winds, interior (sw_core.F90:686-705)
// ===========================================================================
// The excluded j rows are NOT written. K2 supplies them on a rank that owns a
// tile edge; elsewhere they are caller-supplied halo values.
__global__ void dsw_ut_interior_kernel(TransportBounds b, View3<const fv_real> uc,
                                       View3<const fv_real> vc,
                                       View2<const fv_real> cosa_u,
                                       View2<const fv_real> rsin_u, View3<fv_real> ut) {
  Box t;
  if (!in_box(b.is - 1, b.ie + 2, b.js - b.ng, b.je + b.ng, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  if (j == 0 || j == 1 || j == b.npy - 1 || j == b.npy) return;
  ut(i, j, k) = (uc(i, j, k) - fv_real(0.25) * cosa_u(i, j) *
                                   (vc(i - 1, j, k) + vc(i, j, k) + vc(i - 1, j + 1, k) +
                                    vc(i, j + 1, k))) *
                rsin_u(i, j);
}

__global__ void dsw_vt_interior_kernel(TransportBounds b, View3<const fv_real> uc,
                                       View3<const fv_real> vc,
                                       View2<const fv_real> cosa_v,
                                       View2<const fv_real> rsin_v, View3<fv_real> vt) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, b.js - 1, b.je + 2, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  if (j == 1 || j == b.npy) return;
  vt(i, j, k) = (vc(i, j, k) - fv_real(0.25) * cosa_v(i, j) *
                                   (uc(i, j - 1, k) + uc(i + 1, j - 1, k) + uc(i, j, k) +
                                    uc(i + 1, j, k))) *
                rsin_v(i, j);
}

// ===========================================================================
// K2 -- tile-edge overrides (sw_core.F90:709-779)
// ===========================================================================
// Eight kernels, launched West/East/South/North and within each face
// sign-switch-then-4-point, because the 4-point pass reads the row the
// sign-switch pass just wrote.

// :710-716 -- ut(1,j) from the upwind sin_sg component.
__global__ void dsw_edge_ut_west_kernel(TransportBounds b, fv_real dt,
                                        View3<const fv_real> uc,
                                        View3<const fv_real> sin_sg, View3<fv_real> ut) {
  Box t;
  if (!in_box(1, 1, b.js - b.ng, b.je + b.ng, t)) return;
  const int64_t j = t.j, k = t.k;
  ut(1, j, k) = uc(1, j, k) * dt > fv_real(0) ? uc(1, j, k) / sin_sg(0, j, 2)
                                              : uc(1, j, k) / sin_sg(1, j, 0);
}

// :717-722 -- vt(0,j) and vt(1,j) from the freshly written ut column.
__global__ void dsw_edge_vt_west_kernel(TransportBounds b, View3<const fv_real> vc,
                                        View2<const fv_real> cosa_v,
                                        View3<const fv_real> ut, View3<fv_real> vt) {
  Box t;
  const int64_t j0 = b.js > 3 ? b.js : 3;
  const int64_t j1 = (b.npy - 2) < (b.je + 1) ? (b.npy - 2) : (b.je + 1);
  if (!in_box(0, 1, j0, j1, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  vt(i, j, k) = vc(i, j, k) - fv_real(0.25) * cosa_v(i, j) *
                                  (ut(i, j - 1, k) + ut(i + 1, j - 1, k) + ut(i, j, k) +
                                   ut(i + 1, j, k));
}

// :726-732
__global__ void dsw_edge_ut_east_kernel(TransportBounds b, fv_real dt,
                                        View3<const fv_real> uc,
                                        View3<const fv_real> sin_sg, View3<fv_real> ut) {
  Box t;
  if (!in_box(b.npx, b.npx, b.js - b.ng, b.je + b.ng, t)) return;
  const int64_t npx = b.npx, j = t.j, k = t.k;
  ut(npx, j, k) = uc(npx, j, k) * dt > fv_real(0)
                      ? uc(npx, j, k) / sin_sg(npx - 1, j, 2)
                      : uc(npx, j, k) / sin_sg(npx, j, 0);
}

// :734-739
__global__ void dsw_edge_vt_east_kernel(TransportBounds b, View3<const fv_real> vc,
                                        View2<const fv_real> cosa_v,
                                        View3<const fv_real> ut, View3<fv_real> vt) {
  Box t;
  const int64_t j0 = b.js > 3 ? b.js : 3;
  const int64_t j1 = (b.npy - 2) < (b.je + 1) ? (b.npy - 2) : (b.je + 1);
  if (!in_box(b.npx - 1, b.npx, j0, j1, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  vt(i, j, k) = vc(i, j, k) - fv_real(0.25) * cosa_v(i, j) *
                                  (ut(i, j - 1, k) + ut(i + 1, j - 1, k) + ut(i, j, k) +
                                   ut(i + 1, j, k));
}

// :745-751
__global__ void dsw_edge_vt_south_kernel(TransportBounds b, fv_real dt,
                                         View3<const fv_real> vc,
                                         View3<const fv_real> sin_sg, View3<fv_real> vt) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, 1, 1, t)) return;
  const int64_t i = t.i, k = t.k;
  vt(i, 1, k) = vc(i, 1, k) * dt > fv_real(0) ? vc(i, 1, k) / sin_sg(i, 0, 3)
                                              : vc(i, 1, k) / sin_sg(i, 1, 1);
}

// :753-758
__global__ void dsw_edge_ut_south_kernel(TransportBounds b, View3<const fv_real> uc,
                                         View2<const fv_real> cosa_u,
                                         View3<const fv_real> vt, View3<fv_real> ut) {
  Box t;
  const int64_t i0 = b.is > 3 ? b.is : 3;
  const int64_t i1 = (b.npx - 2) < (b.ie + 1) ? (b.npx - 2) : (b.ie + 1);
  if (!in_box(i0, i1, 0, 1, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  ut(i, j, k) = uc(i, j, k) - fv_real(0.25) * cosa_u(i, j) *
                                  (vt(i - 1, j, k) + vt(i, j, k) + vt(i - 1, j + 1, k) +
                                   vt(i, j + 1, k));
}

// :763-769
__global__ void dsw_edge_vt_north_kernel(TransportBounds b, fv_real dt,
                                         View3<const fv_real> vc,
                                         View3<const fv_real> sin_sg, View3<fv_real> vt) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, b.npy, b.npy, t)) return;
  const int64_t i = t.i, npy = b.npy, k = t.k;
  vt(i, npy, k) = vc(i, npy, k) * dt > fv_real(0)
                      ? vc(i, npy, k) / sin_sg(i, npy - 1, 3)
                      : vc(i, npy, k) / sin_sg(i, npy, 1);
}

// :770-775
__global__ void dsw_edge_ut_north_kernel(TransportBounds b, View3<const fv_real> uc,
                                         View2<const fv_real> cosa_u,
                                         View3<const fv_real> vt, View3<fv_real> ut) {
  Box t;
  const int64_t i0 = b.is > 3 ? b.is : 3;
  const int64_t i1 = (b.npx - 2) < (b.ie + 1) ? (b.npx - 2) : (b.ie + 1);
  if (!in_box(i0, i1, b.npy - 1, b.npy, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  ut(i, j, k) = uc(i, j, k) - fv_real(0.25) * cosa_u(i, j) *
                                  (vt(i - 1, j, k) + vt(i, j, k) + vt(i - 1, j + 1, k) +
                                   vt(i, j + 1, k));
}

// ===========================================================================
// K3 -- 2x2 corner solves (sw_core.F90:790-861)
// ===========================================================================
// One thread per level per corner, executing the block's statements in exactly
// the Fortran order. Serializing them this way means the port does not depend
// on any argument about which statements happen to be independent.
__device__ __forceinline__ fv_real corner_damp(fv_real cu, fv_real cv) {
  return fv_real(1) / (fv_real(1) - fv_real(0.0625) * cu * cv);
}

// :790-804
__global__ void dsw_corner_sw_kernel(View3<const fv_real> uc, View3<const fv_real> vc,
                                     View2<const fv_real> cu, View2<const fv_real> cv,
                                     View3<fv_real> ut, View3<fv_real> vt) {
  const int64_t k = blockIdx.z;
  fv_real d = corner_damp(cu(2, 0), cv(1, 0));
  ut(2, 0, k) =
      (uc(2, 0, k) -
       fv_real(0.25) * cu(2, 0) *
           (vt(1, 1, k) + vt(2, 1, k) + vt(2, 0, k) + vc(1, 0, k) -
            fv_real(0.25) * cv(1, 0) *
                (ut(1, 0, k) + ut(1, -1, k) + ut(2, -1, k)))) * d;
  d = corner_damp(cu(0, 1), cv(0, 2));
  vt(0, 2, k) =
      (vc(0, 2, k) -
       fv_real(0.25) * cv(0, 2) *
           (ut(1, 1, k) + ut(1, 2, k) + ut(0, 2, k) + uc(0, 1, k) -
            fv_real(0.25) * cu(0, 1) *
                (vt(0, 1, k) + vt(-1, 1, k) + vt(-1, 2, k)))) * d;
  d = corner_damp(cu(2, 1), cv(1, 2));
  ut(2, 1, k) =
      (uc(2, 1, k) -
       fv_real(0.25) * cu(2, 1) *
           (vt(1, 1, k) + vt(2, 1, k) + vt(2, 2, k) + vc(1, 2, k) -
            fv_real(0.25) * cv(1, 2) *
                (ut(1, 1, k) + ut(1, 2, k) + ut(2, 2, k)))) * d;
  vt(1, 2, k) =
      (vc(1, 2, k) -
       fv_real(0.25) * cv(1, 2) *
           (ut(1, 1, k) + ut(1, 2, k) + ut(2, 2, k) + uc(2, 1, k) -
            fv_real(0.25) * cu(2, 1) *
                (vt(1, 1, k) + vt(2, 1, k) + vt(2, 2, k)))) * d;
}

// :806-823
__global__ void dsw_corner_se_kernel(int64_t npx, View3<const fv_real> uc,
                                     View3<const fv_real> vc, View2<const fv_real> cu,
                                     View2<const fv_real> cv, View3<fv_real> ut,
                                     View3<fv_real> vt) {
  const int64_t k = blockIdx.z;
  fv_real d = corner_damp(cu(npx - 1, 0), cv(npx - 1, 0));
  ut(npx - 1, 0, k) =
      (uc(npx - 1, 0, k) -
       fv_real(0.25) * cu(npx - 1, 0) *
           (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 0, k) + vc(npx - 1, 0, k) -
            fv_real(0.25) * cv(npx - 1, 0) *
                (ut(npx, 0, k) + ut(npx, -1, k) + ut(npx - 1, -1, k)))) * d;
  d = corner_damp(cu(npx + 1, 1), cv(npx, 2));
  vt(npx, 2, k) =
      (vc(npx, 2, k) -
       fv_real(0.25) * cv(npx, 2) *
           (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx + 1, 2, k) + uc(npx + 1, 1, k) -
            fv_real(0.25) * cu(npx + 1, 1) *
                (vt(npx, 1, k) + vt(npx + 1, 1, k) + vt(npx + 1, 2, k)))) * d;
  d = corner_damp(cu(npx - 1, 1), cv(npx - 1, 2));
  ut(npx - 1, 1, k) =
      (uc(npx - 1, 1, k) -
       fv_real(0.25) * cu(npx - 1, 1) *
           (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 2, k) + vc(npx - 1, 2, k) -
            fv_real(0.25) * cv(npx - 1, 2) *
                (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx - 1, 2, k)))) * d;
  vt(npx - 1, 2, k) =
      (vc(npx - 1, 2, k) -
       fv_real(0.25) * cv(npx - 1, 2) *
           (ut(npx, 1, k) + ut(npx, 2, k) + ut(npx - 1, 2, k) + uc(npx - 1, 1, k) -
            fv_real(0.25) * cu(npx - 1, 1) *
                (vt(npx - 1, 1, k) + vt(npx - 2, 1, k) + vt(npx - 2, 2, k)))) * d;
}

// :825-843
__global__ void dsw_corner_ne_kernel(int64_t npx, int64_t npy, View3<const fv_real> uc,
                                     View3<const fv_real> vc, View2<const fv_real> cu,
                                     View2<const fv_real> cv, View3<fv_real> ut,
                                     View3<fv_real> vt) {
  const int64_t k = blockIdx.z;
  fv_real d = corner_damp(cu(npx - 1, npy), cv(npx - 1, npy + 1));
  ut(npx - 1, npy, k) =
      (uc(npx - 1, npy, k) -
       fv_real(0.25) * cu(npx - 1, npy) *
           (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy + 1, k) +
            vc(npx - 1, npy + 1, k) -
            fv_real(0.25) * cv(npx - 1, npy + 1) *
                (ut(npx, npy, k) + ut(npx, npy + 1, k) + ut(npx - 1, npy + 1, k)))) * d;
  d = corner_damp(cu(npx + 1, npy - 1), cv(npx, npy - 1));
  vt(npx, npy - 1, k) =
      (vc(npx, npy - 1, k) -
       fv_real(0.25) * cv(npx, npy - 1) *
           (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx + 1, npy - 2, k) +
            uc(npx + 1, npy - 1, k) -
            fv_real(0.25) * cu(npx + 1, npy - 1) *
                (vt(npx, npy, k) + vt(npx + 1, npy, k) + vt(npx + 1, npy - 1, k)))) * d;
  d = corner_damp(cu(npx - 1, npy - 1), cv(npx - 1, npy - 1));
  ut(npx - 1, npy - 1, k) =
      (uc(npx - 1, npy - 1, k) -
       fv_real(0.25) * cu(npx - 1, npy - 1) *
           (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy - 1, k) +
            vc(npx - 1, npy - 1, k) -
            fv_real(0.25) * cv(npx - 1, npy - 1) *
                (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx - 1, npy - 2, k)))) *
      d;
  vt(npx - 1, npy - 1, k) =
      (vc(npx - 1, npy - 1, k) -
       fv_real(0.25) * cv(npx - 1, npy - 1) *
           (ut(npx, npy - 1, k) + ut(npx, npy - 2, k) + ut(npx - 1, npy - 2, k) +
            uc(npx - 1, npy - 1, k) -
            fv_real(0.25) * cu(npx - 1, npy - 1) *
                (vt(npx - 1, npy, k) + vt(npx - 2, npy, k) + vt(npx - 2, npy - 1, k)))) *
      d;
}

// :845-861
__global__ void dsw_corner_nw_kernel(int64_t npy, View3<const fv_real> uc,
                                     View3<const fv_real> vc, View2<const fv_real> cu,
                                     View2<const fv_real> cv, View3<fv_real> ut,
                                     View3<fv_real> vt) {
  const int64_t k = blockIdx.z;
  fv_real d = corner_damp(cu(2, npy), cv(1, npy + 1));
  ut(2, npy, k) =
      (uc(2, npy, k) -
       fv_real(0.25) * cu(2, npy) *
           (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy + 1, k) + vc(1, npy + 1, k) -
            fv_real(0.25) * cv(1, npy + 1) *
                (ut(1, npy, k) + ut(1, npy + 1, k) + ut(2, npy + 1, k)))) * d;
  d = corner_damp(cu(0, npy - 1), cv(0, npy - 1));
  vt(0, npy - 1, k) =
      (vc(0, npy - 1, k) -
       fv_real(0.25) * cv(0, npy - 1) *
           (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(0, npy - 2, k) + uc(0, npy - 1, k) -
            fv_real(0.25) * cu(0, npy - 1) *
                (vt(0, npy, k) + vt(-1, npy, k) + vt(-1, npy - 1, k)))) * d;
  d = corner_damp(cu(2, npy - 1), cv(1, npy - 1));
  ut(2, npy - 1, k) =
      (uc(2, npy - 1, k) -
       fv_real(0.25) * cu(2, npy - 1) *
           (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy - 1, k) + vc(1, npy - 1, k) -
            fv_real(0.25) * cv(1, npy - 1) *
                (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(2, npy - 2, k)))) * d;
  vt(1, npy - 1, k) =
      (vc(1, npy - 1, k) -
       fv_real(0.25) * cv(1, npy - 1) *
           (ut(1, npy - 1, k) + ut(1, npy - 2, k) + ut(2, npy - 2, k) + uc(2, npy - 1, k) -
            fv_real(0.25) * cu(2, npy - 1) *
                (vt(1, npy, k) + vt(2, npy, k) + vt(2, npy - 1, k)))) * d;
}

// ===========================================================================
// K4 -- advective fluxes and upwind Courant numbers (sw_core.F90:882-921)
// ===========================================================================
__global__ void dsw_xfx_kernel(TransportBounds b, fv_real dt, View3<const fv_real> ut,
                               View3<fv_real> xfx) {
  Box t;
  if (!in_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, t)) return;
  xfx(t.i, t.j, t.k) = dt * ut(t.i, t.j, t.k);
}

__global__ void dsw_yfx_kernel(TransportBounds b, fv_real dt, View3<const fv_real> vt,
                               View3<fv_real> yfx) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, t)) return;
  yfx(t.i, t.j, t.k) = dt * vt(t.i, t.j, t.k);
}

// :898-909. The rescale consumes the pre-rescale xfx, so it stays in the same
// thread as the crx computation.
__global__ void dsw_crx_kernel(TransportBounds b, View2<const fv_real> dy,
                               View2<const fv_real> rdxa, View3<const fv_real> sin_sg,
                               View3<fv_real> crx, View3<fv_real> xfx) {
  Box t;
  if (!in_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  const fv_real f = xfx(i, j, k);
  if (f > fv_real(0)) {
    crx(i, j, k) = f * rdxa(i - 1, j);
    xfx(i, j, k) = dy(i, j) * f * sin_sg(i - 1, j, 2);
  } else {
    crx(i, j, k) = f * rdxa(i, j);
    xfx(i, j, k) = dy(i, j) * f * sin_sg(i, j, 0);
  }
}

// :910-921
__global__ void dsw_cry_kernel(TransportBounds b, View2<const fv_real> dx,
                               View2<const fv_real> rdya, View3<const fv_real> sin_sg,
                               View3<fv_real> cry, View3<fv_real> yfx) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  const fv_real f = yfx(i, j, k);
  if (f > fv_real(0)) {
    cry(i, j, k) = f * rdya(i, j - 1);
    yfx(i, j, k) = dx(i, j) * f * sin_sg(i, j - 1, 3);
  } else {
    cry(i, j, k) = f * rdya(i, j);
    yfx(i, j, k) = dx(i, j) * f * sin_sg(i, j, 1);
  }
}

// ===========================================================================
// K5 -- flux capacitors (sw_core.F90:943-960)
// ===========================================================================
// Plain `+=` in accumulator_real. d_sw never initializes these: dyn_core zeroes
// them once per k_split, so accumulation across acoustic substeps is the
// authoritative behaviour and this driver must not reset them.
__global__ void dsw_cx_kernel(TransportBounds b, View3<const fv_real> crx,
                              View3<accumulator_real> cx) {
  Box t;
  if (!in_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, t)) return;
  cx(t.i, t.j, t.k) += static_cast<accumulator_real>(crx(t.i, t.j, t.k));
}

__global__ void dsw_xflux_kernel(TransportBounds b, View3<const fv_real> fx,
                                 View3<accumulator_real> xflux) {
  Box t;
  if (!in_box(b.is, b.ie + 1, b.js, b.je, t)) return;
  xflux(t.i, t.j, t.k) += static_cast<accumulator_real>(fx(t.i, t.j, t.k));
}

__global__ void dsw_cy_kernel(TransportBounds b, View3<const fv_real> cry,
                              View3<accumulator_real> cy) {
  Box t;
  if (!in_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, t)) return;
  cy(t.i, t.j, t.k) += static_cast<accumulator_real>(cry(t.i, t.j, t.k));
}

__global__ void dsw_yflux_kernel(TransportBounds b, View3<const fv_real> fy,
                                 View3<accumulator_real> yflux) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je + 1, t)) return;
  yflux(t.i, t.j, t.k) += static_cast<accumulator_real>(fy(t.i, t.j, t.k));
}

// ===========================================================================
// K6 -- heat_source / diss_est reset (sw_core.F90:963-968)
// ===========================================================================
__global__ void dsw_heat_reset_kernel(TransportBounds b, View3<fv_real> heat_source,
                                      View3<fv_real> diss_est) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  heat_source(t.i, t.j, t.k) = fv_real(0);
  diss_est(t.i, t.j, t.k) = fv_real(0);
}

// ===========================================================================
// K7 -- dw and the damp_w heat-source seed (sw_core.F90:975-984)
// ===========================================================================
// `w` is the PRE-transport w: S11's fv_tp_2d has not run yet. d_con is the
// per-level d_con_k, and dd8 = kgb(k)*abs(dt).
__global__ void dsw_w_damping_heat_kernel(TransportBounds b, LevelFilter lf, fv_real adt,
                                          View1<const fv_real> kgb,
                                          View1<const fv_real> d_con,
                                          View3<const fv_real> w,
                                          View3<const fv_real> fx2,
                                          View3<const fv_real> fy2,
                                          View2<const fv_real> rarea, View3<fv_real> dw,
                                          View3<fv_real> heat_source) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  if (!level_active(lf, k)) return;
  const int64_t kk = k < lf.npz ? k : lf.npz - 1;
  const fv_real d = (fx2(i, j, k) - fx2(i + 1, j, k) + fy2(i, j, k) - fy2(i, j + 1, k)) *
                    rarea(i, j);
  dw(i, j, k) = d;
  heat_source(i, j, k) =
      kgb(kk) * adt - d_con(kk) * d * (w(i, j, k) + fv_real(0.5) * d);
}

// ===========================================================================
// K8 -- mass-weighted w update, OLD delp (sw_core.F90:988-992)
// ===========================================================================
__global__ void dsw_w_mass_kernel(TransportBounds b, View3<const fv_real> delp,
                                  View3<const fv_real> gx, View3<const fv_real> gy,
                                  View2<const fv_real> rarea, View3<fv_real> w) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  w(i, j, k) = delp(i, j, k) * w(i, j, k) +
               (gx(i, j, k) - gx(i + 1, j, k) + gy(i, j, k) - gy(i, j + 1, k)) *
                   rarea(i, j);
}

// ===========================================================================
// K9 -- pt / delp update (sw_core.F90:1034-1047)
// ===========================================================================
// All three statements in one thread and in Fortran order, so the OLD delp is a
// register value: no double buffer is required (see §3 of the header).
__global__ void dsw_pt_delp_kernel(TransportBounds b, View3<const fv_real> fx,
                                   View3<const fv_real> fy, View3<const fv_real> gx,
                                   View3<const fv_real> gy, View2<const fv_real> rarea,
                                   View3<fv_real> pt, View3<fv_real> delp) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  fv_real p = pt(i, j, k) * delp(i, j, k) +
              (gx(i, j, k) - gx(i + 1, j, k) + gy(i, j, k) - gy(i, j + 1, k)) *
                  rarea(i, j);
  const fv_real d = delp(i, j, k) +
                    (fx(i, j, k) - fx(i + 1, j, k) + fy(i, j, k) - fy(i, j + 1, k)) *
                        rarea(i, j);
  delp(i, j, k) = d;
  pt(i, j, k) = p / d;
}

// ===========================================================================
// K10 -- dpx accumulation (sw_core.F90:1050-1054)
// ===========================================================================
// The Fortran parenthesisation ((fx(i)-fx(i+1)) + (fy(j)-fy(j+1))) is preserved:
// dpx is kind=8 while fx/fy are working precision, so the grouping is
// observable in an R4 build.
__global__ void dsw_dpx_kernel(TransportBounds b, View3<const fv_real> fx,
                               View3<const fv_real> fy, View2<const fv_real> rarea,
                               View3<accumulator_real> dpx) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  const int64_t i = t.i, j = t.j, k = t.k;
  const fv_real d = ((fx(i, j, k) - fx(i + 1, j, k)) + (fy(i, j, k) - fy(i, j + 1, k))) *
                    rarea(i, j);
  dpx(i, j, k) += static_cast<accumulator_real>(d);
}

// ===========================================================================
// K11 / K12 -- w normalize and damping add (sw_core.F90:1250-1262)
// ===========================================================================
__global__ void dsw_w_normalize_kernel(TransportBounds b, View3<const fv_real> delp,
                                       View3<fv_real> w) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  w(t.i, t.j, t.k) = w(t.i, t.j, t.k) / delp(t.i, t.j, t.k);
}

__global__ void dsw_w_damping_add_kernel(TransportBounds b, LevelFilter lf,
                                         View3<const fv_real> dw, View3<fv_real> w) {
  Box t;
  if (!in_box(b.is, b.ie, b.js, b.je, t)) return;
  if (!level_active(lf, t.k)) return;
  w(t.i, t.j, t.k) += dw(t.i, t.j, t.k);
}

// ===========================================================================
// Derived per-level coefficients (§4)
// ===========================================================================
__global__ void dsw_level_coef_kernel(int64_t npz, int32_t nord_t, fv_real da_min,
                                      fv_real da_min_c, View1<const fv_real> damp_v,
                                      View1<const fv_real> damp_w,
                                      View1<const fv_real> damp_t,
                                      View1<const fv_real> nord_v,
                                      View1<const fv_real> nord_w,
                                      View1<fv_real> deln_damp_delp,
                                      View1<fv_real> deln_gate_delp,
                                      View1<fv_real> deln_damp_pt,
                                      View1<fv_real> deln_gate_pt,
                                      View1<fv_real> nord_t_level,
                                      View1<fv_real> del6_damp_w,
                                      View1<fv_real> del6_damp_v) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= npz) return;
  const int32_t nv = round_nord(nord_v(k));
  const int32_t nw = round_nord(nord_w(k));
  const fv_real dv = damp_v(k), dwk = damp_w(k), dtk = damp_t(k);
  // fv_tp_2d's deln_flux coefficient uses gridstruct%da_min (tp_core.F90:217/237)
  // and its activation threshold is 1.e-4, which is NOT del6_vt_flux's 1.E-5.
  deln_damp_delp(k) = ipow(dv * da_min, nv + 1);
  deln_gate_delp(k) = dv > fv_real(1.0e-4) ? fv_real(1) : fv_real(0);
  deln_damp_pt(k) = ipow(dtk * da_min, nord_t + 1);
  deln_gate_pt(k) = dtk > fv_real(1.0e-4) ? fv_real(1) : fv_real(0);
  nord_t_level(k) = static_cast<fv_real>(nord_t);
  // del6_vt_flux's coefficient uses gridstruct%da_min_c (sw_core.F90:973, :1477)
  // and its activation threshold is 1.E-5, applied to the RAW damp array.
  del6_damp_w(k) = ipow(dwk * da_min_c, nw + 1);
  del6_damp_v(k) = ipow(dv * da_min_c, nv + 1);
}

// --- validation helpers ---------------------------------------------------

bool any_null_3(std::initializer_list<const void *> p) {
  for (const void *q : p)
    if (q == nullptr) return true;
  return false;
}

int32_t clamp_nord_cap(int32_t nord, int64_t ng) {
  const int32_t hard = static_cast<int32_t>(ng) - 1;
  return nord < hard ? nord : hard;
}

} // namespace

// ===========================================================================
// Public driver-owned launchers
// ===========================================================================

fv3_gpu_status launch_dsw_contravariant_cuda(const TransportBounds &b,
                                            View3<const fv_real> uc,
                                            View3<const fv_real> vc,
                                            View2<const fv_real> cosa_u,
                                            View2<const fv_real> cosa_v,
                                            View2<const fv_real> rsin_u,
                                            View2<const fv_real> rsin_v,
                                            View3<fv_real> ut, View3<fv_real> vt,
                                            void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  {
    Launch2D p = plan_box(b.is - 1, b.ie + 2, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_ut_interior_kernel<<<p.grid, p.block, 0, s>>>(b, uc, vc, cosa_u, rsin_u, ut);
  }
  {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, b.js - 1, b.je + 2, b.nk);
    dsw_vt_interior_kernel<<<p.grid, p.block, 0, s>>>(b, uc, vc, cosa_v, rsin_v, vt);
  }
  return launch_status();
}

fv3_gpu_status launch_dsw_edge_winds_cuda(const TransportBounds &b, fv_real dt,
                                          View3<const fv_real> uc,
                                          View3<const fv_real> vc,
                                          View2<const fv_real> cosa_u,
                                          View2<const fv_real> cosa_v,
                                          View3<const fv_real> sin_sg, View3<fv_real> ut,
                                          View3<fv_real> vt, void *stream) noexcept {
  if (b.nested) return FV3_GPU_SUCCESS;
  auto s = static_cast<cudaStream_t>(stream);
  const int64_t j0 = b.js > 3 ? b.js : 3;
  const int64_t j1 = (b.npy - 2) < (b.je + 1) ? (b.npy - 2) : (b.je + 1);
  const int64_t i0 = b.is > 3 ? b.is : 3;
  const int64_t i1 = (b.npx - 2) < (b.ie + 1) ? (b.npx - 2) : (b.ie + 1);

  if (b.west_edge) {
    Launch2D p = plan_box(1, 1, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_edge_ut_west_kernel<<<p.grid, p.block, 0, s>>>(b, dt, uc, sin_sg, ut);
    if (j1 >= j0) {
      Launch2D q = plan_box(0, 1, j0, j1, b.nk);
      dsw_edge_vt_west_kernel<<<q.grid, q.block, 0, s>>>(b, vc, cosa_v, as_const(ut), vt);
    }
  }
  if (b.east_edge) {
    Launch2D p = plan_box(b.npx, b.npx, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_edge_ut_east_kernel<<<p.grid, p.block, 0, s>>>(b, dt, uc, sin_sg, ut);
    if (j1 >= j0) {
      Launch2D q = plan_box(b.npx - 1, b.npx, j0, j1, b.nk);
      dsw_edge_vt_east_kernel<<<q.grid, q.block, 0, s>>>(b, vc, cosa_v, as_const(ut), vt);
    }
  }
  if (b.south_edge) {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, 1, 1, b.nk);
    dsw_edge_vt_south_kernel<<<p.grid, p.block, 0, s>>>(b, dt, vc, sin_sg, vt);
    if (i1 >= i0) {
      Launch2D q = plan_box(i0, i1, 0, 1, b.nk);
      dsw_edge_ut_south_kernel<<<q.grid, q.block, 0, s>>>(b, uc, cosa_u, as_const(vt), ut);
    }
  }
  if (b.north_edge) {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, b.npy, b.npy, b.nk);
    dsw_edge_vt_north_kernel<<<p.grid, p.block, 0, s>>>(b, dt, vc, sin_sg, vt);
    if (i1 >= i0) {
      Launch2D q = plan_box(i0, i1, b.npy - 1, b.npy, b.nk);
      dsw_edge_ut_north_kernel<<<q.grid, q.block, 0, s>>>(b, uc, cosa_u, as_const(vt), ut);
    }
  }
  return launch_status();
}

fv3_gpu_status launch_dsw_corner_winds_cuda(const TransportBounds &b,
                                            View3<const fv_real> uc,
                                            View3<const fv_real> vc,
                                            View2<const fv_real> cosa_u,
                                            View2<const fv_real> cosa_v,
                                            View3<fv_real> ut, View3<fv_real> vt,
                                            void *stream) noexcept {
  if (b.nested) return FV3_GPU_SUCCESS;
  auto s = static_cast<cudaStream_t>(stream);
  const dim3 block(1, 1, 1), grid(1, 1, static_cast<unsigned>(b.nk));
  if (b.sw_corner)
    dsw_corner_sw_kernel<<<grid, block, 0, s>>>(uc, vc, cosa_u, cosa_v, ut, vt);
  if (b.se_corner)
    dsw_corner_se_kernel<<<grid, block, 0, s>>>(b.npx, uc, vc, cosa_u, cosa_v, ut, vt);
  if (b.ne_corner)
    dsw_corner_ne_kernel<<<grid, block, 0, s>>>(b.npx, b.npy, uc, vc, cosa_u, cosa_v, ut,
                                                vt);
  if (b.nw_corner)
    dsw_corner_nw_kernel<<<grid, block, 0, s>>>(b.npy, uc, vc, cosa_u, cosa_v, ut, vt);
  return launch_status();
}

fv3_gpu_status launch_dsw_advective_flux_cuda(const TransportBounds &b, fv_real dt,
                                              View3<const fv_real> ut,
                                              View3<const fv_real> vt,
                                              View3<fv_real> xfx_adv,
                                              View3<fv_real> yfx_adv,
                                              void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  {
    Launch2D p = plan_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_xfx_kernel<<<p.grid, p.block, 0, s>>>(b, dt, ut, xfx_adv);
  }
  {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, b.nk);
    dsw_yfx_kernel<<<p.grid, p.block, 0, s>>>(b, dt, vt, yfx_adv);
  }
  return launch_status();
}

fv3_gpu_status launch_dsw_courant_finalize_cuda(const TransportBounds &b,
                                                View2<const fv_real> dx,
                                                View2<const fv_real> dy,
                                                View2<const fv_real> rdxa,
                                                View2<const fv_real> rdya,
                                                View3<const fv_real> sin_sg,
                                                View3<fv_real> crx_adv,
                                                View3<fv_real> cry_adv,
                                                View3<fv_real> xfx_adv,
                                                View3<fv_real> yfx_adv,
                                                void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  {
    Launch2D p = plan_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_crx_kernel<<<p.grid, p.block, 0, s>>>(b, dy, rdxa, sin_sg, crx_adv, xfx_adv);
  }
  {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, b.nk);
    dsw_cry_kernel<<<p.grid, p.block, 0, s>>>(b, dx, rdya, sin_sg, cry_adv, yfx_adv);
  }
  return launch_status();
}

fv3_gpu_status launch_dsw_flux_capacitor_cuda(const TransportBounds &b,
                                              View3<const fv_real> crx_adv,
                                              View3<const fv_real> cry_adv,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View3<accumulator_real> cx,
                                              View3<accumulator_real> cy,
                                              View3<accumulator_real> xflux,
                                              View3<accumulator_real> yflux,
                                              void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  {
    Launch2D p = plan_box(b.is, b.ie + 1, b.js - b.ng, b.je + b.ng, b.nk);
    dsw_cx_kernel<<<p.grid, p.block, 0, s>>>(b, crx_adv, cx);
  }
  {
    Launch2D p = plan_box(b.is, b.ie + 1, b.js, b.je, b.nk);
    dsw_xflux_kernel<<<p.grid, p.block, 0, s>>>(b, fx, xflux);
  }
  {
    Launch2D p = plan_box(b.is - b.ng, b.ie + b.ng, b.js, b.je + 1, b.nk);
    dsw_cy_kernel<<<p.grid, p.block, 0, s>>>(b, cry_adv, cy);
  }
  {
    Launch2D p = plan_box(b.is, b.ie, b.js, b.je + 1, b.nk);
    dsw_yflux_kernel<<<p.grid, p.block, 0, s>>>(b, fy, yflux);
  }
  return launch_status();
}

fv3_gpu_status launch_dsw_heat_reset_cuda(const TransportBounds &b,
                                          View3<fv_real> heat_source,
                                          View3<fv_real> diss_est,
                                          void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_heat_reset_kernel<<<p.grid, p.block, 0, s>>>(b, heat_source, diss_est);
  return launch_status();
}

fv3_gpu_status launch_dsw_w_damping_heat_cuda(const TransportBounds &b,
                                              const LevelFilter &lf, fv_real dt,
                                              View1<const fv_real> kgb,
                                              View1<const fv_real> d_con,
                                              View3<const fv_real> w,
                                              View3<const fv_real> fx2,
                                              View3<const fv_real> fy2,
                                              View2<const fv_real> rarea,
                                              View3<fv_real> dw,
                                              View3<fv_real> heat_source,
                                              void *stream) noexcept {
  if (kgb.data == nullptr || d_con.data == nullptr) return FV3_GPU_INVALID_ARGUMENT;
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  // sw_core.F90:974 -- dd8 = kgb*abs(dt).
  const fv_real adt = dt < fv_real(0) ? -dt : dt;
  dsw_w_damping_heat_kernel<<<p.grid, p.block, 0, s>>>(b, lf, adt, kgb, d_con, w, fx2, fy2,
                                                       rarea, dw, heat_source);
  return launch_status();
}

fv3_gpu_status launch_dsw_w_mass_update_cuda(const TransportBounds &b,
                                             View3<const fv_real> delp,
                                             View3<const fv_real> gx,
                                             View3<const fv_real> gy,
                                             View2<const fv_real> rarea,
                                             View3<fv_real> w, void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_w_mass_kernel<<<p.grid, p.block, 0, s>>>(b, delp, gx, gy, rarea, w);
  return launch_status();
}

fv3_gpu_status launch_dsw_pt_delp_update_cuda(const TransportBounds &b,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View3<const fv_real> gx,
                                              View3<const fv_real> gy,
                                              View2<const fv_real> rarea,
                                              View3<fv_real> pt, View3<fv_real> delp,
                                              void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_pt_delp_kernel<<<p.grid, p.block, 0, s>>>(b, fx, fy, gx, gy, rarea, pt, delp);
  return launch_status();
}

fv3_gpu_status launch_dsw_dpx_accumulate_cuda(const TransportBounds &b,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View2<const fv_real> rarea,
                                              View3<accumulator_real> dpx,
                                              void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_dpx_kernel<<<p.grid, p.block, 0, s>>>(b, fx, fy, rarea, dpx);
  return launch_status();
}

fv3_gpu_status launch_dsw_w_normalize_cuda(const TransportBounds &b,
                                           View3<const fv_real> delp, View3<fv_real> w,
                                           void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_w_normalize_kernel<<<p.grid, p.block, 0, s>>>(b, delp, w);
  return launch_status();
}

fv3_gpu_status launch_dsw_w_damping_add_cuda(const TransportBounds &b,
                                             const LevelFilter &lf,
                                             View3<const fv_real> dw, View3<fv_real> w,
                                             void *stream) noexcept {
  auto s = static_cast<cudaStream_t>(stream);
  Launch2D p = plan_box(b.is, b.ie, b.js, b.je, b.nk);
  dsw_w_damping_add_kernel<<<p.grid, p.block, 0, s>>>(b, lf, dw, w);
  return launch_status();
}

fv3_gpu_status launch_dsw_level_coefficients_cuda(const TransportBounds &b,
                                                  const DSwOptions &opt,
                                                  const DSwLevels &lv,
                                                  const WpBDSwArena &ws,
                                                  void *stream) noexcept {
  if (lv.damp_v.data == nullptr || lv.damp_w.data == nullptr ||
      lv.damp_t.data == nullptr || lv.nord_v.data == nullptr ||
      lv.nord_w.data == nullptr)
    return FV3_GPU_INVALID_ARGUMENT;
  if (ws.deln_damp_delp.data == nullptr || ws.deln_gate_delp.data == nullptr ||
      ws.deln_damp_pt.data == nullptr || ws.deln_gate_pt.data == nullptr ||
      ws.nord_t_level.data == nullptr || ws.del6_damp_w.data == nullptr ||
      ws.del6_damp_v.data == nullptr)
    return FV3_GPU_INVALID_ARGUMENT;
  auto s = static_cast<cudaStream_t>(stream);
  const unsigned threads = 64;
  const unsigned blocks = static_cast<unsigned>((b.npz + threads - 1) / threads);
  dsw_level_coef_kernel<<<blocks, threads, 0, s>>>(
      b.npz, opt.nord_t, static_cast<fv_real>(opt.da_min),
      static_cast<fv_real>(opt.da_min_c), lv.damp_v, lv.damp_w, lv.damp_t, lv.nord_v,
      lv.nord_w, ws.deln_damp_delp, ws.deln_gate_delp, ws.deln_damp_pt, ws.deln_gate_pt,
      ws.nord_t_level, ws.del6_damp_w, ws.del6_damp_v);
  return launch_status();
}

// ===========================================================================
// The complete d_sw driver
// ===========================================================================

fv3_gpu_status launch_d_sw_cuda(const TransportBounds &b, const DSwOptions &opt,
                                const DSwLevels &lv, const DSwGeometry &g,
                                const DSwState &st, const WpBDSwArena &ws,
                                void *stream) noexcept {
  // --- §6 unsupported-configuration fences, all before the first launch ----
  if (opt.grid_type != 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.nested) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.hydrostatic) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.inline_q || opt.use_cond) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.do_f3d || opt.do_skeb || opt.rot3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.a_imp != 1.0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  {
    const int32_t h[4] = {opt.hord_mt, opt.hord_vt, opt.hord_tm, opt.hord_dp};
    for (int32_t v : h) {
      const int32_t a = v < 0 ? -v : v;
      if (a != 6) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
    }
  }
  if (opt.nord < 0 || opt.nord > 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.nord_t < 0 || opt.nord_t > static_cast<int32_t>(b.ng) - 1)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.npx == 0 || b.npy == 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  // An explicit stream is required. S31's smag_corner reuses WP-D's
  // launch_a2b_ord4_global_cuda, which rejects a null stream, so a null stream
  // would be discovered thirty stages in -- after pt, delp, w and dpx had
  // already been mutated. Fencing it here keeps the "reject before mutating
  // state" property that every other fence in this block has. WP-E always
  // supplies a stream (validate_execution_context rejects a null submission
  // stream), so this costs production nothing.
  if (stream == nullptr) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.nk != b.npz) return FV3_GPU_INVALID_ARGUMENT;

  if (any_null_3({st.u.data, st.v.data, st.w.data, st.delp.data, st.pt.data, st.uc.data,
                  st.vc.data, st.ua.data, st.va.data, st.divgd.data, st.delpc.data,
                  st.ptc.data, st.ut.data, st.vt.data, st.crx.data, st.cry.data,
                  st.xfx.data, st.yfx.data, st.heat_source.data, st.diss_est.data,
                  st.mfx.data, st.mfy.data, st.cx.data, st.cy.data, st.dpx.data}))
    return FV3_GPU_INVALID_ARGUMENT;
  if (any_null_3({ws.ub.data, ws.vb.data, ws.wk.data, ws.ke.data, ws.vort.data, ws.fx.data,
                  ws.fy.data, ws.gx.data, ws.gy.data, ws.ra_x.data, ws.ra_y.data,
                  ws.fx2.data, ws.fy2.data, ws.dw.data, ws.q_i.data, ws.q_j.data,
                  ws.fx2_tp.data, ws.fy2_tp.data, ws.d2.data, ws.fx2_dn.data,
                  ws.fy2_dn.data, ws.bl.data, ws.br.data, ws.b0.data, ws.smt.data}))
    return FV3_GPU_INVALID_ARGUMENT;
  if (any_null_3({g.area.data, g.rdxa.data, g.rdya.data, g.del6_u.data, g.del6_v.data,
                  g.cosa.data, g.rsina.data, g.rsin_u.data, g.rsin_v.data, g.f0.data,
                  g.div.rarea.data, g.div.dx.data, g.div.dy.data, g.div.rdx.data,
                  g.div.rdy.data, g.dxa.data, g.dya.data, g.div.sin_sg.data,
                  g.div.cosa_u.data, g.div.cosa_v.data, g.div.rarea_c.data,
                  g.div.dxc.data, g.div.dyc.data, g.div.sina_u.data, g.div.sina_v.data,
                  g.div.cosa_s.data, g.div.rsin2.data, g.div.divg_u.data,
                  g.div.divg_v.data}))
    return FV3_GPU_INVALID_ARGUMENT;

  const fv_real dt = opt.dt;
  const int32_t nord_cap_k = clamp_nord_cap(opt.nord, b.ng);
  const int32_t nord_cap_v = clamp_nord_cap(opt.nord < 2 ? opt.nord : 2, b.ng);

  fv3_gpu_status rc;

  // --- derived per-level coefficients (device-side; no D2H anywhere) -------
  rc = launch_dsw_level_coefficients_cuda(b, opt, lv, ws, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // ---------------------------------------------------------------- S1..S3
  rc = launch_dsw_contravariant_cuda(b, as_const(st.uc), as_const(st.vc), g.div.cosa_u,
                                     g.div.cosa_v, g.rsin_u, g.rsin_v, st.ut, st.vt,
                                     stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_dsw_edge_winds_cuda(b, dt, as_const(st.uc), as_const(st.vc), g.div.cosa_u,
                                  g.div.cosa_v, g.div.sin_sg, st.ut, st.vt, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_dsw_corner_winds_cuda(b, as_const(st.uc), as_const(st.vc), g.div.cosa_u,
                                    g.div.cosa_v, st.ut, st.vt, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // ------------------------------------------------------------------ S4/S5
  rc = launch_dsw_advective_flux_cuda(b, dt, as_const(st.ut), as_const(st.vt), st.xfx,
                                      st.yfx, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_dsw_courant_finalize_cuda(b, g.div.dx, g.div.dy, g.rdxa, g.rdya,
                                        g.div.sin_sg, st.crx, st.cry, st.xfx, st.yfx,
                                        stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // --------------------------------------------------------------------- S6
  rc = launch_ra_x_cuda(b, as_const(st.xfx), g.area, ws.ra_x, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_ra_y_cuda(b, as_const(st.yfx), g.area, ws.ra_y, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // A LevelFilter that selects every level. fv_tp_2d itself is not level
  // filtered in d_sw; only the deln_flux/del6 tails are.
  LevelFilter all{};
  all.npz = b.npz;
  all.mode = 0;

  // --------------------------------------------------------------------- S7
  // fv_tp_2d(delp, hord_dp, fx, fy, nord=nord_v(k), damp_c=damp_v(k)).
  {
    ScalarTransportOptions o{};
    o.hord = opt.hord_dp;
    o.mass_flux_weighting = 0;
    o.deln_flux_enabled = 1;
    o.per_level_deln = 1;
    o.nord_cap = nord_cap_v;
    o.da_min = static_cast<fv_real>(opt.da_min);
    o.deln_damp_level = as_const1(ws.deln_damp_delp);
    o.deln_gate_level = as_const1(ws.deln_gate_delp);
    o.deln_nord_level = lv.nord_v;
    rc = launch_scalar_transport_cuda(
        b, o, all, st.delp, View3<const fv_real>{}, as_const(st.crx), as_const(st.cry),
        as_const(st.xfx), as_const(st.yfx), View3<const fv_real>{},
        View3<const fv_real>{}, g.area, g.div.rarea, g.dxa, g.dya, g.del6_u, g.del6_v,
        as_const(ws.ra_x), as_const(ws.ra_y), ws.q_i, ws.q_j, ws.fx2_tp, ws.fy2_tp, ws.d2,
        ws.fx2_dn, ws.fy2_dn, ws.fx, ws.fy, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // --------------------------------------------------------------------- S8
  rc = launch_dsw_flux_capacitor_cuda(b, as_const(st.crx), as_const(st.cry),
                                      as_const(ws.fx), as_const(ws.fy), st.cx, st.cy,
                                      st.mfx, st.mfy, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // --------------------------------------------------------------------- S9
  rc = launch_dsw_heat_reset_cuda(b, st.heat_source, st.diss_est, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // -------------------------------------------------------------------- S10
  // del6_vt_flux(nord_w(k), damp4=(damp_w(k)*da_min_c)**(nord_w+1), w, wk, fx2,
  // fy2) on the damp_w > 1.E-5 levels, then dw and the heat-source seed.
  LevelFilter lf_w{};
  lf_w.npz = b.npz;
  lf_w.mode = 3;
  lf_w.damp = as_const1(ws.del6_damp_w);
  lf_w.gate = lv.damp_w;
  lf_w.ndif = lv.nord_w;
  {
    rc = launch_del6_vt_flux_cuda(b, nord_cap_v, lf_w, st.w, g.del6_u, g.del6_v,
                                  g.div.rarea, ws.wk, ws.fx2, ws.fy2, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
    LevelFilter gate_w{};
    gate_w.npz = b.npz;
    gate_w.mode = 1;
    gate_w.gate = lv.damp_w;
    gate_w.damp = lv.damp_w;
    rc = launch_dsw_w_damping_heat_cuda(b, gate_w, dt, lv.kgb, lv.d_con, as_const(st.w),
                                        as_const(ws.fx2), as_const(ws.fy2), g.div.rarea,
                                        ws.dw, st.heat_source, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S11
  // fv_tp_2d(w, hord_vt, gx, gy, mfx=fx, mfy=fy) -- no mass, no deln_flux.
  {
    ScalarTransportOptions o{};
    o.hord = opt.hord_vt;
    o.mass_flux_weighting = 1;
    o.deln_flux_enabled = 0;
    rc = launch_scalar_transport_cuda(
        b, o, all, st.w, View3<const fv_real>{}, as_const(st.crx), as_const(st.cry),
        as_const(st.xfx), as_const(st.yfx), as_const(ws.fx), as_const(ws.fy), g.area,
        g.div.rarea, g.dxa, g.dya, g.del6_u, g.del6_v, as_const(ws.ra_x),
        as_const(ws.ra_y), ws.q_i, ws.q_j, ws.fx2_tp, ws.fy2_tp, ws.d2, ws.fx2_dn,
        ws.fy2_dn, ws.gx, ws.gy, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S12
  rc = launch_dsw_w_mass_update_cuda(b, as_const(st.delp), as_const(ws.gx),
                                     as_const(ws.gy), g.div.rarea, st.w, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // -------------------------------------------------------------------- S14
  // fv_tp_2d(pt, hord_tm, gx, gy, mfx=fx, mfy=fy, mass=delp (OLD), nord=nord_t,
  // damp_c=damp_t(k)).
  {
    ScalarTransportOptions o{};
    o.hord = opt.hord_tm;
    o.mass_flux_weighting = 1;
    o.deln_flux_enabled = 1;
    o.per_level_deln = 1;
    o.nord_cap = opt.nord_t;
    o.da_min = static_cast<fv_real>(opt.da_min);
    o.deln_damp_level = as_const1(ws.deln_damp_pt);
    o.deln_gate_level = as_const1(ws.deln_gate_pt);
    o.deln_nord_level = as_const1(ws.nord_t_level);
    rc = launch_scalar_transport_cuda(
        b, o, all, st.pt, as_const(st.delp), as_const(st.crx), as_const(st.cry),
        as_const(st.xfx), as_const(st.yfx), as_const(ws.fx), as_const(ws.fy), g.area,
        g.div.rarea, g.dxa, g.dya, g.del6_u, g.del6_v, as_const(ws.ra_x),
        as_const(ws.ra_y), ws.q_i, ws.q_j, ws.fx2_tp, ws.fy2_tp, ws.d2, ws.fx2_dn,
        ws.fy2_dn, ws.gx, ws.gy, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // ------------------------------------------------------------------- S15/16
  rc = launch_dsw_pt_delp_update_cuda(b, as_const(ws.fx), as_const(ws.fy),
                                      as_const(ws.gx), as_const(ws.gy), g.div.rarea,
                                      st.pt, st.delp, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_dsw_dpx_accumulate_cuda(b, as_const(ws.fx), as_const(ws.fy), g.div.rarea,
                                      st.dpx, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // ---------------------------------------------------------------- S18..S24
  MomentumOptions mo{};
  mo.hord_mt = opt.hord_mt;
  mo.lim_fac = opt.lim_fac;
  mo.grid_type = opt.grid_type;
  mo.do_f3d = opt.do_f3d;
  PpmMomentumWorkspace pw{ws.bl, ws.br, ws.b0, ws.smt};

  rc = launch_bgrid_vb_cuda(b, mo, dt, as_const(st.uc), as_const(st.vc), as_const(st.ut),
                            as_const(st.vt), g.cosa, g.rsina, ws.vb, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_ytp_v_cuda(b, mo, as_const(ws.vb), as_const(st.v), g.div.dy, g.div.rdy, pw,
                         ws.ub, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_ke_product_cuda(b, as_const(ws.vb), as_const(ws.ub), ws.ke, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_bgrid_ub_cuda(b, mo, dt, as_const(st.uc), as_const(st.vc), as_const(st.ut),
                            as_const(st.vt), g.cosa, g.rsina, ws.ub, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_xtp_u_cuda(b, mo, as_const(ws.ub), as_const(st.u), g.div.dx, g.div.rdx, pw,
                         ws.vb, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_ke_average_cuda(b, as_const(ws.ub), as_const(ws.vb), ws.ke, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = launch_ke_corner_fix_cuda(b, dt, as_const(st.ut), as_const(st.vt), as_const(st.u),
                                 as_const(st.v), ws.ke, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // ---------------------------------------------------------------- S25..S27
  // DESTROYS the contravariant ut/vt; nothing after this point reads them with
  // that meaning (§3).
  rc = launch_vorticity_precursor_cuda(b, as_const(st.u), as_const(st.v), g.div.dx,
                                       g.div.dy, g.div.rarea, st.ut, st.vt, ws.wk, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // ------------------------------------------------------------------- S28/29
  rc = launch_dsw_w_normalize_cuda(b, as_const(st.delp), st.w, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;
  {
    LevelFilter gate_w{};
    gate_w.npz = b.npz;
    gate_w.mode = 1;
    gate_w.gate = lv.damp_w;
    gate_w.damp = lv.damp_w;
    rc = launch_dsw_w_damping_add_cuda(b, gate_w, as_const(ws.dw), st.w, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S31
  // Divergence damping. Both branch families are launched on every call, each
  // filtered to the levels whose nord_k selects it (§4).
  DivergenceDampingOptions dopt{};
  dopt.grid_type = opt.grid_type;
  dopt.do_skeb = opt.do_skeb;
  dopt.dddmp = static_cast<fv_real>(opt.dddmp);
  dopt.da_min = static_cast<fv_real>(opt.da_min);
  dopt.da_min_c = static_cast<fv_real>(opt.da_min_c);

  DivergenceLevels dlv{};
  dlv.nord_k = lv.nord_k;
  dlv.d2_bg = lv.d2_bg;
  dlv.d4_bg = lv.d4_bg;
  dlv.d_con = lv.d_con;
  dlv.npz = b.npz;

  {
    // Branch L: nord_k == 0.
    DivergenceLevels l = dlv;
    l.mode = 1;
    rc = launch_divdamp_nord0_cuda(b, dopt, l, dt, as_const(st.u), as_const(st.v), st.ua,
                                   st.va, as_const(st.uc), as_const(st.vc), g.div, st.ptc,
                                   ws.vort, st.delpc, ws.ke, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }
  if (nord_cap_k >= 1) {
    // Branch H rows H1/H2: the save plus the n-loop. DESTROYS uc/vc.
    DivergenceLevels h = dlv;
    h.mode = 2;
    rc = launch_divdamp_high_order_cuda(b, dopt, h, nord_cap_k, g.div, st.divgd, st.uc,
                                        st.vc, st.delpc, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;

    // Row H3, then row H4, each as a complete pass over the nord groups so the
    // two Fortran statements stay in source order for every level.
    for (int32_t n = 1; n <= nord_cap_k; ++n) {
      DivergenceLevels h3 = dlv;
      h3.mode = 2;
      h3.nord_sel = n;
      if (!(opt.dddmp >= fv_real(1.0e-5))) {
        rc = launch_divdamp_vort_zero_cuda(b, h3, ws.vort, stream);
      } else {
        const fv_real adt = dt < fv_real(0) ? -dt : dt;
        rc = launch_smag_corner_cuda(b, dopt, h3, adt, as_const(st.u), as_const(st.v),
                                     g.div, g.a2b, ws.smag, ws.vort, stream);
      }
      if (rc != FV3_GPU_SUCCESS) return rc;
    }
    for (int32_t n = 1; n <= nord_cap_k; ++n) {
      DivergenceLevels h4 = dlv;
      h4.mode = 2;
      h4.nord_sel = n;
      rc = launch_divdamp_high_order_finalize_cuda(b, dopt, h4, as_const(st.delpc),
                                                   as_const(st.divgd), ws.vort, ws.ke,
                                                   stream);
      if (rc != FV3_GPU_SUCCESS) return rc;
    }
  }

  // -------------------------------------------------------------------- S32
  // MUST precede S33: it reads the damping vort that S33 overwrites.
  {
    DivergenceLevels dc = dlv;
    dc.mode = 3;
    rc = launch_dcon_vort_diff_cuda(b, dc, as_const(ws.vort), ws.ub, ws.vb, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S33
  rc = launch_vorticity_absolute_cuda(b, mo, as_const(ws.wk), g.f0, st.z_rat, ws.vort,
                                      stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // -------------------------------------------------------------------- S34
  // fv_tp_2d(vort, hord_vt, fx, fy) -- no mfx/mass/nord.
  {
    ScalarTransportOptions o{};
    o.hord = opt.hord_vt;
    o.mass_flux_weighting = 0;
    o.deln_flux_enabled = 0;
    rc = launch_scalar_transport_cuda(
        b, o, all, ws.vort, View3<const fv_real>{}, as_const(st.crx), as_const(st.cry),
        as_const(st.xfx), as_const(st.yfx), View3<const fv_real>{},
        View3<const fv_real>{}, g.area, g.div.rarea, g.dxa, g.dya, g.del6_u, g.del6_v,
        as_const(ws.ra_x), as_const(ws.ra_y), ws.q_i, ws.q_j, ws.fx2_tp, ws.fy2_tp, ws.d2,
        ws.fx2_dn, ws.fy2_dn, ws.fx, ws.fy, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S35
  rc = launch_dwind_update_cuda(b, as_const(st.ut), as_const(st.vt), as_const(ws.ke),
                                as_const(ws.fx), as_const(ws.fy), st.u, st.v, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // -------------------------------------------------------------------- S36
  // del6_vt_flux(nord_v(k), damp4, wk, vort, ut, vt): THIRD meaning of ut/vt.
  LevelFilter lf_v{};
  lf_v.npz = b.npz;
  lf_v.mode = 3;
  lf_v.damp = as_const1(ws.del6_damp_v);
  lf_v.gate = lv.damp_v;
  lf_v.ndif = lv.nord_v;
  rc = launch_vorticity_damping_cuda(b, nord_cap_v, lf_v, ws.wk, g.del6_u, g.del6_v,
                                     g.div.rarea, ws.vort, st.ut, st.vt, stream);
  if (rc != FV3_GPU_SUCCESS) return rc;

  // -------------------------------------------------------------------- S37
  {
    DivergenceLevels dc = dlv;
    dc.mode = 3;
    rc = launch_dcon_scale_cuda(b, dc, g.div, as_const(st.u), as_const(st.v),
                                as_const(st.ut), as_const(st.vt), ws.ub, ws.vb, ws.fx,
                                ws.fy, ws.gx, ws.gy, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
    rc = launch_dcon_heat_source_cuda(b, dopt, dc, g.div, as_const(st.delp),
                                      as_const(ws.ub), as_const(ws.vb), as_const(ws.fx),
                                      as_const(ws.fy), as_const(ws.gx), as_const(ws.gy),
                                      st.heat_source, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  // -------------------------------------------------------------------- S38
  {
    LevelFilter gate_v{};
    gate_v.npz = b.npz;
    gate_v.mode = 1;
    gate_v.gate = lv.damp_v;
    gate_v.damp = lv.damp_v;
    rc = launch_dwind_damping_add_cuda(b, gate_v, as_const(st.ut), as_const(st.vt), st.u,
                                       st.v, stream);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  return launch_status();
}

} // namespace fv3::gpu::dgrid
