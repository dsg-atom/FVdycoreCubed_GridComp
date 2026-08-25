#include "dgrid/wp_b_cuda.cuh"

#include <cmath>
#include <cstdint>

namespace fv3::gpu::dgrid {
namespace {

// tp_core.F90:89-96 PPM reconstruction constants.
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

// --- edge_profile (nh_utils.F90:1595-1629, uniform_grid = .false.) -------
// gam(i,k) in the Fortran is independent of i (bet and gk depend only on dp0
// and k), so one thread fills the whole column once.
__global__ void edge_profile_gam_kernel(int64_t npz, View1<const fv_real> dp0,
                                        View1<fv_real> gam) {
  if (blockIdx.x || blockIdx.y || blockIdx.z || threadIdx.x || threadIdx.y || threadIdx.z)
    return;
  const fv_real g0 = dp0(1) / dp0(0);
  fv_real bet = g0 * (g0 + fv_real(0.5));
  gam(0) = (fv_real(1) + g0 * (g0 + fv_real(1.5))) / bet;
  for (int64_t k = 1; k < npz; ++k) {
    const fv_real gk = dp0(k - 1) / dp0(k);
    bet = fv_real(2) + fv_real(2) * gk - gam(k - 1);
    gam(k) = gk / bet;
  }
}

// Tridiagonal interface reconstruction of the (q1, q2) pair for one column.
// q1e/q2e carry npz+1 interface levels; the Fortran's local qe1/qe2 scratch is
// exactly these arrays, so the forward sweep, bottom closure, and backward
// substitution are written in place just as Fortran does.
__device__ __forceinline__ void edge_profile_column(
    int64_t i, int64_t j, int64_t npz, View1<const fv_real> dp0,
    View1<const fv_real> gam, View3<const fv_real> q1, View3<const fv_real> q2,
    View3<fv_real> q1e, View3<fv_real> q2e) {
  const fv_real g0 = dp0(1) / dp0(0);
  const fv_real xt1_top = fv_real(2) * g0 * (g0 + fv_real(1));
  const fv_real bet0 = g0 * (g0 + fv_real(0.5));
  q1e(i, j, 0) = (xt1_top * q1(i, j, 0) + q1(i, j, 1)) / bet0;
  q2e(i, j, 0) = (xt1_top * q2(i, j, 0) + q2(i, j, 1)) / bet0;

  fv_real gk = fv_real(0);
  for (int64_t k = 1; k < npz; ++k) {
    gk = dp0(k - 1) / dp0(k);
    const fv_real bet = fv_real(2) + fv_real(2) * gk - gam(k - 1);
    q1e(i, j, k) =
        (fv_real(3) * (q1(i, j, k - 1) + gk * q1(i, j, k)) - q1e(i, j, k - 1)) / bet;
    q2e(i, j, k) =
        (fv_real(3) * (q2(i, j, k - 1) + gk * q2(i, j, k)) - q2e(i, j, k - 1)) / bet;
  }

  const fv_real a_bot = fv_real(1) + gk * (gk + fv_real(1.5));
  const fv_real xt1 = fv_real(2) * gk * (gk + fv_real(1));
  const fv_real xt2 = gk * (gk + fv_real(0.5)) - a_bot * gam(npz - 1);
  q1e(i, j, npz) =
      (xt1 * q1(i, j, npz - 1) + q1(i, j, npz - 2) - a_bot * q1e(i, j, npz - 1)) / xt2;
  q2e(i, j, npz) =
      (xt1 * q2(i, j, npz - 1) + q2(i, j, npz - 2) - a_bot * q2e(i, j, npz - 1)) / xt2;

  for (int64_t k = npz - 1; k >= 0; --k) {
    q1e(i, j, k) = q1e(i, j, k) - gam(k) * q1e(i, j, k + 1);
    q2e(i, j, k) = q2e(i, j, k) - gam(k) * q2e(i, j, k + 1);
  }
}

__global__ void edge_profile_x_kernel(TransportBounds b, View1<const fv_real> dp0,
                                      View1<const fv_real> gam, View3<const fv_real> crx,
                                      View3<const fv_real> xfx, View3<fv_real> crx_adv,
                                      View3<fv_real> xfx_adv) {
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > b.ie + 1 || j > jed) return;
  edge_profile_column(i, j, b.npz, dp0, gam, crx, xfx, crx_adv, xfx_adv);
}

__global__ void edge_profile_y_kernel(TransportBounds b, View1<const fv_real> dp0,
                                      View1<const fv_real> gam, View3<const fv_real> cry,
                                      View3<const fv_real> yfx, View3<fv_real> cry_adv,
                                      View3<fv_real> yfx_adv) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ied || j > b.je + 1) return;
  edge_profile_column(i, j, b.npz, dp0, gam, cry, yfx, cry_adv, yfx_adv);
}

// --- copy_corners (tp_core.F90:257-318) ----------------------------------
// (t, s) enumerate the ng x ng corner block; the four corners are applied
// sequentially by every thread, which is safe because each corner reads from a
// band that no corner writes (see wp_b_cuda.cuh).
__global__ void copy_corners_kernel(TransportBounds b, int32_t dir, LevelFilter lf,
                                    View3<fv_real> q) {
  const int64_t ng = b.ng;
  const int64_t npx = b.npx, npy = b.npy;
  const int64_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t s = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (t >= ng || s >= ng || k >= b.nk) return;
  if (!level_active(lf, k)) return;

  if (dir == 1) {
    if (b.sw_corner) {                       // i,j in [1-ng, 0]
      const int64_t i = 1 - ng + t, j = 1 - ng + s;
      q(i, j, k) = q(j, 1 - i, k);
    }
    if (b.se_corner) {                       // i in [npx, npx+ng-1], j in [1-ng, 0]
      const int64_t i = npx + t, j = 1 - ng + s;
      q(i, j, k) = q(npy - j, i - npx + 1, k);
    }
    if (b.ne_corner) {
      const int64_t i = npx + t, j = npy + s;
      q(i, j, k) = q(j, 2 * npx - 1 - i, k);
    }
    if (b.nw_corner) {
      const int64_t i = 1 - ng + t, j = npy + s;
      q(i, j, k) = q(npy - j, i - 1 + npx, k);
    }
  } else {
    if (b.sw_corner) {
      const int64_t i = 1 - ng + t, j = 1 - ng + s;
      q(i, j, k) = q(1 - j, i, k);
    }
    if (b.se_corner) {
      const int64_t i = npx + t, j = 1 - ng + s;
      q(i, j, k) = q(npy + j - 1, npx - i, k);
    }
    if (b.ne_corner) {
      const int64_t i = npx + t, j = npy + s;
      q(i, j, k) = q(2 * npy - 1 - j, i, k);
    }
    if (b.nw_corner) {
      const int64_t i = 1 - ng + t, j = npy + s;
      q(i, j, k) = q(j + 1 - npx, npy - i, k);
    }
  }
}

// --- xppm / yppm interface reconstruction (tp_core.F90:369-386, 749-772) --
// al at interface i (between cells i-1 and i). The one-sided panel-edge
// formulas replace the interior stencil exactly where the Fortran's clipped
// loop bounds (is1 = max(3,is-1), ie3 = min(npx-2,ie+2)) leave it unwritten.
__device__ __forceinline__ fv_real al_x_at(View3<const fv_real> q,
                                           View2<const fv_real> dxa, int64_t i, int64_t j,
                                           int64_t k, const TransportBounds &b,
                                           bool negative_hord) {
  fv_real al;
  const int64_t npx = b.npx;
  const bool seam = b.cubed_sphere && !b.nested;
  if (seam && b.west_edge && i <= 2) {
    if (i == 0) {
      al = ppm_c1() * q(-2, j, k) + ppm_c2() * q(-1, j, k) + ppm_c3() * q(0, j, k);
    } else if (i == 1) {
      al = fv_real(0.5) *
           (((fv_real(2) * dxa(0, j) + dxa(-1, j)) * q(0, j, k) - dxa(0, j) * q(-1, j, k)) /
                (dxa(-1, j) + dxa(0, j)) +
            ((fv_real(2) * dxa(1, j) + dxa(2, j)) * q(1, j, k) - dxa(1, j) * q(2, j, k)) /
                (dxa(1, j) + dxa(2, j)));
    } else {
      al = ppm_c3() * q(1, j, k) + ppm_c2() * q(2, j, k) + ppm_c1() * q(3, j, k);
    }
  } else if (seam && b.east_edge && i >= npx - 1) {
    if (i == npx - 1) {
      al = ppm_c1() * q(npx - 3, j, k) + ppm_c2() * q(npx - 2, j, k) +
           ppm_c3() * q(npx - 1, j, k);
    } else if (i == npx) {
      al = fv_real(0.5) *
           (((fv_real(2) * dxa(npx - 1, j) + dxa(npx - 2, j)) * q(npx - 1, j, k) -
             dxa(npx - 1, j) * q(npx - 2, j, k)) /
                (dxa(npx - 2, j) + dxa(npx - 1, j)) +
            ((fv_real(2) * dxa(npx, j) + dxa(npx + 1, j)) * q(npx, j, k) -
             dxa(npx, j) * q(npx + 1, j, k)) /
                (dxa(npx, j) + dxa(npx + 1, j)));
    } else {
      al = ppm_c3() * q(npx, j, k) + ppm_c2() * q(npx + 1, j, k) +
           ppm_c1() * q(npx + 2, j, k);
    }
  } else {
    al = ppm_p1() * (q(i - 1, j, k) + q(i, j, k)) +
         ppm_p2() * (q(i - 2, j, k) + q(i + 1, j, k));
  }
  if (negative_hord && al < fv_real(0)) al = fv_real(0);
  return al;
}

__device__ __forceinline__ fv_real al_y_at(View3<const fv_real> q,
                                           View2<const fv_real> dya, int64_t i, int64_t j,
                                           int64_t k, const TransportBounds &b,
                                           bool negative_hord) {
  fv_real al;
  const int64_t npy = b.npy;
  const bool seam = b.cubed_sphere && !b.nested;
  if (seam && b.south_edge && j <= 2) {
    if (j == 0) {
      al = ppm_c1() * q(i, -2, k) + ppm_c2() * q(i, -1, k) + ppm_c3() * q(i, 0, k);
    } else if (j == 1) {
      al = fv_real(0.5) *
           (((fv_real(2) * dya(i, 0) + dya(i, -1)) * q(i, 0, k) - dya(i, 0) * q(i, -1, k)) /
                (dya(i, -1) + dya(i, 0)) +
            ((fv_real(2) * dya(i, 1) + dya(i, 2)) * q(i, 1, k) - dya(i, 1) * q(i, 2, k)) /
                (dya(i, 1) + dya(i, 2)));
    } else {
      al = ppm_c3() * q(i, 1, k) + ppm_c2() * q(i, 2, k) + ppm_c1() * q(i, 3, k);
    }
  } else if (seam && b.north_edge && j >= npy - 1) {
    if (j == npy - 1) {
      al = ppm_c1() * q(i, npy - 3, k) + ppm_c2() * q(i, npy - 2, k) +
           ppm_c3() * q(i, npy - 1, k);
    } else if (j == npy) {
      al = fv_real(0.5) *
           (((fv_real(2) * dya(i, npy - 1) + dya(i, npy - 2)) * q(i, npy - 1, k) -
             dya(i, npy - 1) * q(i, npy - 2, k)) /
                (dya(i, npy - 2) + dya(i, npy - 1)) +
            ((fv_real(2) * dya(i, npy) + dya(i, npy + 1)) * q(i, npy, k) -
             dya(i, npy) * q(i, npy + 1, k)) /
                (dya(i, npy) + dya(i, npy + 1)));
    } else {
      al = ppm_c3() * q(i, npy, k) + ppm_c2() * q(i, npy + 1, k) +
           ppm_c1() * q(i, npy + 2, k);
    }
  } else {
    al = ppm_p1() * (q(i, j - 1, k) + q(i, j, k)) +
         ppm_p2() * (q(i, j - 2, k) + q(i, j + 1, k));
  }
  if (negative_hord && al < fv_real(0)) al = fv_real(0);
  return al;
}

// tp_core.F90:532-541 / 936-950: at the two cells adjacent to a panel seam the
// mord=6 smoothness test degrades to the mord=5 sign test.
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

// mord = 6 face flux (tp_core.F90:524-555). Interior arithmetic is unchanged
// from milestone 3A.3; only al and smt5 became index-aware.
__device__ __forceinline__ fv_real xppm_face_flux(View3<const fv_real> q,
                                                  View2<const fv_real> dxa, int64_t i,
                                                  int64_t j, int64_t k, fv_real c,
                                                  const TransportBounds &b,
                                                  bool negative_hord) {
  const fv_real alm = al_x_at(q, dxa, i - 1, j, k, b, negative_hord);
  const fv_real al0 = al_x_at(q, dxa, i, j, k, b, negative_hord);
  const fv_real alp = al_x_at(q, dxa, i + 1, j, k, b, negative_hord);
  const fv_real qm = q(i - 1, j, k), q0 = q(i, j, k);

  const fv_real bl_m1 = alm - qm, br_m1 = al0 - qm, b0_m1 = bl_m1 + br_m1;
  const fv_real bl_0 = al0 - q0, br_0 = alp - q0, b0_0 = bl_0 + br_0;
  const bool smt5_m1 = smt5_from(bl_m1, br_m1, b0_m1, seam_cell_x(i - 1, b));
  const bool smt5_0 = smt5_from(bl_0, br_0, b0_0, seam_cell_x(i, b));

  fv_real fx1, donor;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (br_m1 - c * b0_m1);
    donor = qm;
  } else {
    fx1 = (fv_real(1) + c) * (bl_0 + c * b0_0);
    donor = q0;
  }
  return (smt5_m1 || smt5_0) ? donor + fx1 : donor;
}

__device__ __forceinline__ fv_real yppm_face_flux(View3<const fv_real> q,
                                                  View2<const fv_real> dya, int64_t i,
                                                  int64_t j, int64_t k, fv_real c,
                                                  const TransportBounds &b,
                                                  bool negative_hord) {
  const fv_real alm = al_y_at(q, dya, i, j - 1, k, b, negative_hord);
  const fv_real al0 = al_y_at(q, dya, i, j, k, b, negative_hord);
  const fv_real alp = al_y_at(q, dya, i, j + 1, k, b, negative_hord);
  const fv_real qm = q(i, j - 1, k), q0 = q(i, j, k);

  const fv_real bl_m1 = alm - qm, br_m1 = al0 - qm, b0_m1 = bl_m1 + br_m1;
  const fv_real bl_0 = al0 - q0, br_0 = alp - q0, b0_0 = bl_0 + br_0;
  const bool smt5_m1 = smt5_from(bl_m1, br_m1, b0_m1, seam_cell_y(j - 1, b));
  const bool smt5_0 = smt5_from(bl_0, br_0, b0_0, seam_cell_y(j, b));

  fv_real fx1, donor;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (br_m1 - c * b0_m1);
    donor = qm;
  } else {
    fx1 = (fv_real(1) + c) * (bl_0 + c * b0_0);
    donor = q0;
  }
  return (smt5_m1 || smt5_0) ? donor + fx1 : donor;
}

// --- fv_tp_2d passes -----------------------------------------------------
__global__ void xppm_pass_kernel(TransportBounds b, LevelFilter lf, bool negative,
                                 int64_t lo_i, int64_t hi_i, int64_t lo_j, int64_t hi_j,
                                 View3<const fv_real> q, View2<const fv_real> dxa,
                                 View3<const fv_real> crx, View3<fv_real> flux) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  flux(i, j, k) = xppm_face_flux(q, dxa, i, j, k, crx(i, j, k), b, negative);
}

__global__ void yppm_pass_kernel(TransportBounds b, LevelFilter lf, bool negative,
                                 int64_t lo_i, int64_t hi_i, int64_t lo_j, int64_t hi_j,
                                 View3<const fv_real> q, View2<const fv_real> dya,
                                 View3<const fv_real> cry, View3<fv_real> flux) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  flux(i, j, k) = yppm_face_flux(q, dya, i, j, k, cry(i, j, k), b, negative);
}

// tp_core.F90:168-177: q_i = (q*area + fyy(j) - fyy(j+1)) / ra_y with
// fyy = yfx*fy2.
__global__ void q_i_kernel(TransportBounds b, LevelFilter lf, View3<const fv_real> q,
                           View3<const fv_real> yfx, View3<const fv_real> fy2,
                           View2<const fv_real> area, View3<const fv_real> ra_y,
                           View3<fv_real> q_i) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ied || j > b.je || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real fyy_j = yfx(i, j, k) * fy2(i, j, k);
  const fv_real fyy_jp1 = yfx(i, j + 1, k) * fy2(i, j + 1, k);
  q_i(i, j, k) = (q(i, j, k) * area(i, j) + fyy_j - fyy_jp1) / ra_y(i, j, k);
}

// tp_core.F90:186-193: q_j = (q*area + fx1(i) - fx1(i+1)) / ra_x with
// fx1 = xfx*fx2.
__global__ void q_j_kernel(TransportBounds b, LevelFilter lf, View3<const fv_real> q,
                           View3<const fv_real> xfx, View3<const fv_real> fx2,
                           View2<const fv_real> area, View3<const fv_real> ra_x,
                           View3<fv_real> q_j) {
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie || j > jed || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real fx1_i = xfx(i, j, k) * fx2(i, j, k);
  const fv_real fx1_ip1 = xfx(i + 1, j, k) * fx2(i + 1, j, k);
  q_j(i, j, k) = (q(i, j, k) * area(i, j) + fx1_i - fx1_ip1) / ra_x(i, j, k);
}

// tp_core.F90:216-234. Both optional-argument branches share this expression;
// only the weight differs (wx = xfx_adv / yfx_adv in the `else` branch,
// wx = mfx / mfy when mfx/mfy are present), so one kernel serves both and the
// caller selects the weight view. Milestone 3A.5 generalised the parameter name
// from xfx/yfx; the arithmetic is unchanged.
__global__ void flux_weight_x_kernel(TransportBounds b, LevelFilter lf,
                                     View3<const fv_real> fx2, View3<const fv_real> wx,
                                     View3<fv_real> fx) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  fx(i, j, k) = fv_real(0.5) * (fx(i, j, k) + fx2(i, j, k)) * wx(i, j, k);
}

__global__ void flux_weight_y_kernel(TransportBounds b, LevelFilter lf,
                                     View3<const fv_real> fy2, View3<const fv_real> wy,
                                     View3<fv_real> fy) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie || j > b.je + 1 || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  fy(i, j, k) = fv_real(0.5) * (fy(i, j, k) + fy2(i, j, k)) * wy(i, j, k);
}

// --- ra_x / ra_y ---------------------------------------------------------
__global__ void ra_x_kernel(TransportBounds b, View3<const fv_real> xfx,
                            View2<const fv_real> area, View3<fv_real> ra_x) {
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie || j > jed || k >= b.nk) return;
  ra_x(i, j, k) = area(i, j) + (xfx(i, j, k) - xfx(i + 1, j, k));
}

__global__ void ra_y_kernel(TransportBounds b, View3<const fv_real> yfx,
                            View2<const fv_real> area, View3<fv_real> ra_y) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ied || j > b.je || k >= b.nk) return;
  ra_y(i, j, k) = area(i, j) + (yfx(i, j, k) - yfx(i, j + 1, k));
}

// --- del6_vt_flux (sw_core.F90:1588-1652) --------------------------------
__global__ void del6_zero_kernel(TransportBounds b, LevelFilter lf,
                                 View3<fv_real> fx2, View3<fv_real> fy2) {
  const int64_t h = b.ng;
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  fx2(i, j, k) = fv_real(0);
  fy2(i, j, k) = fv_real(0);
}

// d2 = damp(k)*q -- the raw per-level coefficient, not deln_flux's
// (damp_c*da_min)**(nord+1).
__global__ void del6_init_kernel(TransportBounds b, LevelFilter lf, int64_t i1,
                                 int64_t i2, int64_t j1, int64_t j2,
                                 View3<const fv_real> q, View3<fv_real> d2) {
  const int64_t i = i1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = j1 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > i2 || j > j2 || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const int64_t kk = k < b.npz ? k : b.npz - 1;
  d2(i, j, k) = lf.damp(kk) * q(i, j, k);
}

// forward = true reproduces the first-pass sign (d2(i-1)-d2(i)); the nord
// iterations use the opposite sign (sw_core.F90:1601 vs :1635).
__global__ void del6_flux_x_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                   int64_t hi_i, int64_t lo_j, int64_t hi_j, bool forward,
                                   View3<const fv_real> d2, View2<const fv_real> del6_v,
                                   View3<fv_real> fx2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real diff =
      forward ? (d2(i - 1, j, k) - d2(i, j, k)) : (d2(i, j, k) - d2(i - 1, j, k));
  fx2(i, j, k) = del6_v(i, j) * diff;
}

__global__ void del6_flux_y_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                   int64_t hi_i, int64_t lo_j, int64_t hi_j, bool forward,
                                   View3<const fv_real> d2, View2<const fv_real> del6_u,
                                   View3<fv_real> fy2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real diff =
      forward ? (d2(i, j - 1, k) - d2(i, j, k)) : (d2(i, j, k) - d2(i, j - 1, k));
  fy2(i, j, k) = del6_u(i, j) * diff;
}

__global__ void del6_div_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                int64_t hi_i, int64_t lo_j, int64_t hi_j,
                                View3<const fv_real> fx2, View3<const fv_real> fy2,
                                View2<const fv_real> rarea, View3<fv_real> d2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  d2(i, j, k) = (fx2(i, j, k) - fx2(i + 1, j, k) + fy2(i, j, k) - fy2(i, j + 1, k)) *
               rarea(i, j);
}

// --- z2 copy + zh update (nh_utils.F90:281-305) --------------------------
__global__ void copy_z2_kernel(TransportBounds b, LevelFilter lf,
                               View3<const fv_real> zh, View3<fv_real> z2) {
  const int64_t h = b.ng;
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  z2(i, j, k) = zh(i, j, k);
}

__global__ void zh_update_kernel(TransportBounds b, LevelFilter lf,
                                 View3<const fv_real> z2, View3<const fv_real> fx,
                                 View3<const fv_real> fy, View3<const fv_real> fx2,
                                 View3<const fv_real> fy2, View3<const fv_real> ra_x,
                                 View3<const fv_real> ra_y, View2<const fv_real> area,
                                 View2<const fv_real> rarea, View3<fv_real> zh) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie || j > b.je || k >= b.nk) return;

  const int64_t kk = k < b.npz ? k : b.npz - 1;
  const bool damped = lf.damp(kk) > damp_threshold();
  const fv_real a = area(i, j);
  const fv_real denom = (ra_x(i, j, k) + ra_y(i, j, k)) - a;
  // The numerator MUST keep Fortran's left-to-right association
  //   ((q*area + (fx(i)-fx(i+1))) + (fy(j)-fy(j+1)))
  // (nh_utils.F90:291 / :300). Hoisting (fxdiff + fydiff) into a single
  // `transported` temporary is algebraically equal but not IEEE-equal, and the
  // ~1 ulp difference is amplified through the acoustic substeps, which would
  // block bitwise comparison against the Fortran baseline.
  const fv_real fx_div = fx(i, j, k) - fx(i + 1, j, k);
  const fv_real fy_div = fy(i, j, k) - fy(i, j + 1, k);

  if (damped) {
    const fv_real diffused = ((fx2(i, j, k) - fx2(i + 1, j, k)) +
                              (fy2(i, j, k) - fy2(i, j + 1, k))) *
                             rarea(i, j);
    zh(i, j, k) = ((z2(i, j, k) * a + fx_div) + fy_div) / denom + diffused;
  } else {
    zh(i, j, k) = ((zh(i, j, k) * a + fx_div) + fy_div) / denom;
  }
}

// --- deln_flux (tp_core.F90:1263) ----------------------------------------
// tp_core's deln_flux, distinct from sw_core's del6_vt_flux above. See the
// comparison table in wp_b_cuda.cuh.

// tp_core.F90:1274-1287. The `mass` branch initialises d2 = q and defers the
// coefficient to the final add; the no-mass branch folds it in here.
//
// STRUCTURAL DIFFERENCE (documented, no used value changes): Fortran's init
// loop runs over [is-1-nord, ie+1+nord] x [js-1-nord, je+1+nord], leaving the
// rest of d2(isd:ied, jsd:jed) undefined. For nord < 2 the subsequent
// copy_corners rotations then READ some of those undefined cells -- their
// results land only in d2 cells that no later loop reads, so Fortran tolerates
// it. This kernel instead initialises the whole [is-ng, ie+ng] box, which is
// identical to Fortran at nord == 2, a superset at nord < 2, and provably
// changes no value that is read downstream (the extra ring feeds only
// copy_corners destinations outside every fx2/fy2 read box). Doing so keeps
// compute-sanitizer initcheck clean instead of importing an uninitialised read.
__global__ void deln_init_kernel(TransportBounds b, LevelFilter lf, bool mass_weighted,
                                 fv_real damp_coef, View3<const fv_real> q,
                                 View3<fv_real> d2) {
  const int64_t i = (b.is - b.ng) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - b.ng) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + b.ng || j > b.je + b.ng || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  d2(i, j, k) = mass_weighted ? q(i, j, k) : damp_coef * q(i, j, k);
}

// tp_core.F90:1316-1330 (first pass) and 1345-1368 (iteration passes). The two
// differ only by the sign of the difference, so `flipped` selects it:
//   first pass:  fx2 = del6_v*(d2(i-1,j) - d2(i,j))
//   iterations:  fx2 = del6_v*(d2(i,j)   - d2(i-1,j))
__global__ void deln_flux_x_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                   int64_t hi_i, int64_t lo_j, int64_t hi_j, bool flipped,
                                   View3<const fv_real> d2, View2<const fv_real> del6_v,
                                   View3<fv_real> fx2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real diff = flipped ? (d2(i, j, k) - d2(i - 1, j, k))
                               : (d2(i - 1, j, k) - d2(i, j, k));
  fx2(i, j, k) = del6_v(i, j) * diff;
}

__global__ void deln_flux_y_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                   int64_t hi_i, int64_t lo_j, int64_t hi_j, bool flipped,
                                   View3<const fv_real> d2, View2<const fv_real> del6_u,
                                   View3<fv_real> fy2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real diff = flipped ? (d2(i, j, k) - d2(i, j - 1, k))
                               : (d2(i, j - 1, k) - d2(i, j, k));
  fy2(i, j, k) = del6_u(i, j) * diff;
}

// tp_core.F90:1348-1352.
__global__ void deln_iter_div_kernel(TransportBounds b, LevelFilter lf, int64_t lo_i,
                                     int64_t hi_i, int64_t lo_j, int64_t hi_j,
                                     View3<const fv_real> fx2, View3<const fv_real> fy2,
                                     View2<const fv_real> rarea, View3<fv_real> d2) {
  const int64_t i = lo_i + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = lo_j + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  d2(i, j, k) = (fx2(i, j, k) - fx2(i + 1, j, k) + fy2(i, j, k) - fy2(i, j + 1, k)) *
               rarea(i, j);
}

// tp_core.F90:1386-1394 (`else`, mass absent): fx = fx + fx2 on is..ie+1 x js..je,
// fy = fy + fy2 on is..ie x js..je+1.
__global__ void deln_add_no_mass_kernel(TransportBounds b, LevelFilter lf, bool along_x,
                                        View3<const fv_real> flux2, View3<fv_real> flux) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  const int64_t hi_i = along_x ? b.ie + 1 : b.ie;
  const int64_t hi_j = along_x ? b.je : b.je + 1;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  flux(i, j, k) = flux(i, j, k) + flux2(i, j, k);
}

// tp_core.F90:1373-1384 (`present(mass)`): damp2 = 0.5*damp and the diffusive
// flux is weighted by the two masses straddling the face. The mass neighbour is
// taken along the flux direction: (i-1,j)/(i,j) for x, (i,j-1)/(i,j) for y.
__global__ void deln_add_mass_kernel(TransportBounds b, LevelFilter lf, bool along_x,
                                     fv_real damp2, View3<const fv_real> mass,
                                     View3<const fv_real> flux2, View3<fv_real> flux) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  const int64_t hi_i = along_x ? b.ie + 1 : b.ie;
  const int64_t hi_j = along_x ? b.je : b.je + 1;
  if (i > hi_i || j > hi_j || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  const fv_real m_lo = along_x ? mass(i - 1, j, k) : mass(i, j - 1, k);
  flux(i, j, k) =
      flux(i, j, k) + damp2 * (m_lo + mass(i, j, k)) * flux2(i, j, k);
}

View3<const fv_real> as_const(const View3<fv_real> &v) {
  return View3<const fv_real>{v.data, v.lo_i, v.lo_j, v.lo_k, v.si, v.sj, v.sk};
}

} // namespace

// =========================== launchers ==================================

fv3_gpu_status launch_edge_profile_gam_cuda(const TransportBounds &b,
                                            View1<const fv_real> dp_ref,
                                            View1<fv_real> gam, void *stream) noexcept {
  edge_profile_gam_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(b.npz, dp_ref,
                                                                         gam);
  return launch_status();
}

fv3_gpu_status launch_edge_profile_x_cuda(const TransportBounds &b,
                                          View3<const fv_real> crx,
                                          View3<const fv_real> xfx,
                                          View1<const fv_real> dp_ref,
                                          View1<const fv_real> gam,
                                          View3<fv_real> crx_adv, View3<fv_real> xfx_adv,
                                          void *stream) noexcept {
  const int64_t h = b.ng;
  Launch2D p = plan2d(b.ie + 1 - b.is + 1, (b.je + h) - (b.js - h) + 1, 1);
  edge_profile_x_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, dp_ref, gam, crx, xfx, crx_adv, xfx_adv);
  return launch_status();
}

fv3_gpu_status launch_edge_profile_y_cuda(const TransportBounds &b,
                                          View3<const fv_real> cry,
                                          View3<const fv_real> yfx,
                                          View1<const fv_real> dp_ref,
                                          View1<const fv_real> gam,
                                          View3<fv_real> cry_adv, View3<fv_real> yfx_adv,
                                          void *stream) noexcept {
  const int64_t h = b.ng;
  Launch2D p = plan2d((b.ie + h) - (b.is - h) + 1, b.je + 1 - b.js + 1, 1);
  edge_profile_y_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, dp_ref, gam, cry, yfx, cry_adv, yfx_adv);
  return launch_status();
}

fv3_gpu_status launch_copy_corners_cuda(const TransportBounds &b, int32_t dir,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        void *stream) noexcept {
  if (b.nested || !b.cubed_sphere) return FV3_GPU_SUCCESS;
  if (!(b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner))
    return FV3_GPU_SUCCESS;
  Launch2D p = plan2d(b.ng, b.ng, b.nk);
  copy_corners_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(b, dir,
                                                                                lf, q);
  return launch_status();
}

fv3_gpu_status launch_ra_x_cuda(const TransportBounds &b,
                                View3<const fv_real> xfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_x, void *stream) noexcept {
  const int64_t h = b.ng;
  Launch2D p = plan2d(b.ie - b.is + 1, (b.je + h) - (b.js - h) + 1, b.nk);
  ra_x_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(b, xfx_adv, area,
                                                                        ra_x);
  return launch_status();
}

fv3_gpu_status launch_ra_y_cuda(const TransportBounds &b,
                                View3<const fv_real> yfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_y, void *stream) noexcept {
  const int64_t h = b.ng;
  Launch2D p = plan2d((b.ie + h) - (b.is - h) + 1, b.je - b.js + 1, b.nk);
  ra_y_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(b, yfx_adv, area,
                                                                        ra_y);
  return launch_status();
}

fv3_gpu_status launch_fv_tp_2d_cuda(const TransportBounds &b, int32_t hord,
                                    const LevelFilter &lf, View3<fv_real> q,
                                    View3<const fv_real> crx_adv,
                                    View3<const fv_real> cry_adv,
                                    View3<const fv_real> xfx_adv,
                                    View3<const fv_real> yfx_adv,
                                    View3<const fv_real> wx, View3<const fv_real> wy,
                                    View2<const fv_real> area, View2<const fv_real> dxa,
                                    View2<const fv_real> dya, View3<const fv_real> ra_x,
                                    View3<const fv_real> ra_y, View3<fv_real> q_i,
                                    View3<fv_real> q_j, View3<fv_real> fx,
                                    View3<fv_real> fy, View3<fv_real> fx2,
                                    View3<fv_real> fy2, void *stream) noexcept {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, h = b.ng;
  const int64_t isd = is - h, ied = ie + h, jsd = js - h, jed = je + h;
  auto s = static_cast<cudaStream_t>(stream);
  const bool neg = hord < 0;
  View3<const fv_real> qc = as_const(q);

  // copy_corners(q, dir = 2)
  fv3_gpu_status st = launch_copy_corners_cuda(b, 2, lf, q, stream);
  if (st != FV3_GPU_SUCCESS) return st;

  // yppm(fy2, q, cry_adv) on isd..ied x js..je+1
  {
    Launch2D p = plan2d(ied - isd + 1, je + 1 - js + 1, b.nk);
    yppm_pass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, neg, isd, ied, js, je + 1, qc, dya,
                                                cry_adv, fy2);
  }
  {
    Launch2D p = plan2d(ied - isd + 1, je - js + 1, b.nk);
    q_i_kernel<<<p.grid, p.block, 0, s>>>(b, lf, qc, yfx_adv, as_const(fy2), area, ra_y,
                                          q_i);
  }
  // xppm(fx, q_i, crx_adv) on is..ie+1 x js..je
  {
    Launch2D p = plan2d(ie + 1 - is + 1, je - js + 1, b.nk);
    xppm_pass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, neg, is, ie + 1, js, je,
                                                as_const(q_i), dxa, crx_adv, fx);
  }

  // copy_corners(q, dir = 1)
  st = launch_copy_corners_cuda(b, 1, lf, q, stream);
  if (st != FV3_GPU_SUCCESS) return st;

  // xppm(fx2, q, crx_adv) on is..ie+1 x jsd..jed
  {
    Launch2D p = plan2d(ie + 1 - is + 1, jed - jsd + 1, b.nk);
    xppm_pass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, neg, is, ie + 1, jsd, jed, qc, dxa,
                                                crx_adv, fx2);
  }
  {
    Launch2D p = plan2d(ie - is + 1, jed - jsd + 1, b.nk);
    q_j_kernel<<<p.grid, p.block, 0, s>>>(b, lf, qc, xfx_adv, as_const(fx2), area, ra_x,
                                          q_j);
  }
  // yppm(fy, q_j, cry_adv) on is..ie x js..je+1
  {
    Launch2D p = plan2d(ie - is + 1, je + 1 - js + 1, b.nk);
    yppm_pass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, neg, is, ie, js, je + 1,
                                                as_const(q_j), dya, cry_adv, fy);
  }

  {
    Launch2D p = plan2d(ie + 1 - is + 1, je - js + 1, b.nk);
    flux_weight_x_kernel<<<p.grid, p.block, 0, s>>>(b, lf, as_const(fx2), wx, fx);
  }
  {
    Launch2D p = plan2d(ie - is + 1, je + 1 - js + 1, b.nk);
    flux_weight_y_kernel<<<p.grid, p.block, 0, s>>>(b, lf, as_const(fy2), wy, fy);
  }
  return launch_status();
}

fv3_gpu_status launch_del6_vt_flux_cuda(const TransportBounds &b, int32_t nord_cap,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        View2<const fv_real> del6_u,
                                        View2<const fv_real> del6_v,
                                        View2<const fv_real> rarea, View3<fv_real> d2,
                                        View3<fv_real> fx2, View3<fv_real> fy2,
                                        void *stream) noexcept {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  auto s = static_cast<cudaStream_t>(stream);

  // Damped levels with an out-of-range nord contribute zero damping rather
  // than uninitialized workspace. With nord_cap == ng-1 == 2 below, and the
  // binder rejecting nord_v > 2, no level carrying a value FV3 can actually
  // produce (ndif(k) = nord_v(k) = min(2, nord), dyn_core.F90:713) lands
  // outside a group -- so this pre-zero is a belt-and-braces guard against a
  // malformed damping_ndif, not a routine code path.
  {
    LevelFilter all_damped = lf;
    all_damped.mode = 1;
    Launch2D p = plan2d((ie + b.ng) - (is - b.ng) + 1,
                        (je + b.ng) - (js - b.ng) + 1, b.nk);
    del6_zero_kernel<<<p.grid, p.block, 0, s>>>(b, all_damped, fx2, fy2);
  }

  for (int32_t nord = 0; nord <= nord_cap; ++nord) {
    LevelFilter g = lf;
    g.mode = 3;
    g.nord_sel = nord;

    const int64_t n = nord;
    const int64_t i1 = is - 1 - n, i2 = ie + 1 + n;
    const int64_t j1 = js - 1 - n, j2 = je + 1 + n;
    {
      Launch2D p = plan2d(i2 - i1 + 1, j2 - j1 + 1, b.nk);
      del6_init_kernel<<<p.grid, p.block, 0, s>>>(b, g, i1, i2, j1, j2, as_const(q), d2);
    }

    if (n > 0) {
      fv3_gpu_status st = launch_copy_corners_cuda(b, 1, g, d2, stream);
      if (st != FV3_GPU_SUCCESS) return st;
    }
    {
      Launch2D p = plan2d((ie + n + 1) - (is - n) + 1, (je + n) - (js - n) + 1, b.nk);
      del6_flux_x_kernel<<<p.grid, p.block, 0, s>>>(b, g, is - n, ie + n + 1, js - n,
                                                    je + n, true, as_const(d2), del6_v,
                                                    fx2);
    }
    if (n > 0) {
      fv3_gpu_status st = launch_copy_corners_cuda(b, 2, g, d2, stream);
      if (st != FV3_GPU_SUCCESS) return st;
    }
    {
      Launch2D p = plan2d((ie + n) - (is - n) + 1, (je + n + 1) - (js - n) + 1, b.nk);
      del6_flux_y_kernel<<<p.grid, p.block, 0, s>>>(b, g, is - n, ie + n, js - n,
                                                    je + n + 1, true, as_const(d2),
                                                    del6_u, fy2);
    }

    for (int64_t pass = 1; pass <= n; ++pass) {
      const int64_t nt = n - pass;
      {
        Launch2D p = plan2d((ie + nt + 1) - (is - nt - 1) + 1,
                            (je + nt + 1) - (js - nt - 1) + 1, b.nk);
        del6_div_kernel<<<p.grid, p.block, 0, s>>>(b, g, is - nt - 1, ie + nt + 1,
                                                   js - nt - 1, je + nt + 1,
                                                   as_const(fx2), as_const(fy2), rarea,
                                                   d2);
      }
      fv3_gpu_status st = launch_copy_corners_cuda(b, 1, g, d2, stream);
      if (st != FV3_GPU_SUCCESS) return st;
      {
        Launch2D p =
            plan2d((ie + nt + 1) - (is - nt) + 1, (je + nt) - (js - nt) + 1, b.nk);
        del6_flux_x_kernel<<<p.grid, p.block, 0, s>>>(b, g, is - nt, ie + nt + 1, js - nt,
                                                      je + nt, false, as_const(d2),
                                                      del6_v, fx2);
      }
      st = launch_copy_corners_cuda(b, 2, g, d2, stream);
      if (st != FV3_GPU_SUCCESS) return st;
      {
        Launch2D p =
            plan2d((ie + nt) - (is - nt) + 1, (je + nt + 1) - (js - nt) + 1, b.nk);
        del6_flux_y_kernel<<<p.grid, p.block, 0, s>>>(b, g, is - nt, ie + nt, js - nt,
                                                      je + nt + 1, false, as_const(d2),
                                                      del6_u, fy2);
      }
    }
  }
  return launch_status();
}

fv3_gpu_status launch_copy_z2_cuda(const TransportBounds &b, const LevelFilter &lf,
                                   View3<const fv_real> zh, View3<fv_real> z2,
                                   void *stream) noexcept {
  const int64_t h = b.ng;
  Launch2D p = plan2d((b.ie + h) - (b.is - h) + 1, (b.je + h) - (b.js - h) + 1,
                      b.nk);
  copy_z2_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(b, lf, zh, z2);
  return launch_status();
}

fv3_gpu_status launch_zh_update_cuda(const TransportBounds &b, const LevelFilter &lf,
                                     View3<const fv_real> z2, View3<const fv_real> fx,
                                     View3<const fv_real> fy, View3<const fv_real> fx2,
                                     View3<const fv_real> fy2, View3<const fv_real> ra_x,
                                     View3<const fv_real> ra_y, View2<const fv_real> area,
                                     View2<const fv_real> rarea, View3<fv_real> zh,
                                     void *stream) noexcept {
  Launch2D p = plan2d(b.ie - b.is + 1, b.je - b.js + 1, b.nk);
  zh_update_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lf, z2, fx, fy, fx2, fy2, ra_x, ra_y, area, rarea, zh);
  return launch_status();
}

fv3_gpu_status launch_deln_flux_transport_cuda(
    const TransportBounds &b, int32_t nord, fv_real damp_c, fv_real da_min,
    bool mass_weighted, const LevelFilter &lf, View3<const fv_real> q,
    View3<const fv_real> mass, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> fx, View3<fv_real> fy, View3<fv_real> d2,
    View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  auto s = static_cast<cudaStream_t>(stream);

  // The stencil reaches 1 + nord cells into the halo and copy_corners rotates
  // only ng rings, so nord > ng - 1 would read un-rotated corner data no matter
  // how much halo the caller registered. Fortran cannot produce it either:
  // dyn_core.F90:713/716 set nord_v(k)/nord_t = min(2, nord).
  if (nord < 0 || nord > static_cast<int32_t>(b.ng) - 1)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  // tp_core.F90:213/231 -- fv_tp_2d forms this immediately before the call.
  // This is the coefficient rule that distinguishes deln_flux from
  // del6_vt_flux, which uses the raw per-level damp(k) instead.
  const fv_real damp_coef = std::pow(damp_c * da_min, static_cast<fv_real>(nord + 1));

  // tp_core.F90:1274-1287
  {
    Launch2D p = plan2d((ie + b.ng) - (is - b.ng) + 1, (je + b.ng) - (js - b.ng) + 1, b.nk);
    deln_init_kernel<<<p.grid, p.block, 0, s>>>(b, lf, mass_weighted, damp_coef, q, d2);
  }

  // Rotation 1 of 2 + 2*nord (tp_core.F90:1313). Guarded on nord > 0 exactly as
  // in Fortran, so nord == 0 performs no rotation at all.
  if (nord > 0) {
    fv3_gpu_status st = launch_copy_corners_cuda(b, 1, lf, d2, stream);
    if (st != FV3_GPU_SUCCESS) return st;
  }
  // tp_core.F90:1316-1320
  {
    Launch2D p = plan2d((ie + nord + 1) - (is - nord) + 1, (je + nord) - (js - nord) + 1,
                        b.nk);
    deln_flux_x_kernel<<<p.grid, p.block, 0, s>>>(b, lf, is - nord, ie + nord + 1,
                                                  js - nord, je + nord, false,
                                                  as_const(d2), del6_v, fx2);
  }
  // Rotation 2 of 2 + 2*nord (tp_core.F90:1326).
  if (nord > 0) {
    fv3_gpu_status st = launch_copy_corners_cuda(b, 2, lf, d2, stream);
    if (st != FV3_GPU_SUCCESS) return st;
  }
  // tp_core.F90:1329-1333
  {
    Launch2D p = plan2d((ie + nord) - (is - nord) + 1, (je + nord + 1) - (js - nord) + 1,
                        b.nk);
    deln_flux_y_kernel<<<p.grid, p.block, 0, s>>>(b, lf, is - nord, ie + nord,
                                                  js - nord, je + nord + 1, false,
                                                  as_const(d2), del6_u, fy2);
  }

  // tp_core.F90:1343-1370. Two more rotations per iteration, unconditional
  // inside the loop (the loop itself only runs when nord > 0), giving the
  // authoritative total of 2 + 2*nord.
  for (int32_t n = 1; n <= nord; ++n) {
    const int64_t nt = nord - n;
    {
      Launch2D p = plan2d((ie + nt + 1) - (is - nt - 1) + 1,
                          (je + nt + 1) - (js - nt - 1) + 1, b.nk);
      deln_iter_div_kernel<<<p.grid, p.block, 0, s>>>(b, lf, is - nt - 1, ie + nt + 1,
                                                      js - nt - 1, je + nt + 1,
                                                      as_const(fx2), as_const(fy2), rarea,
                                                      d2);
    }
    fv3_gpu_status st = launch_copy_corners_cuda(b, 1, lf, d2, stream);  // :1354
    if (st != FV3_GPU_SUCCESS) return st;
    // tp_core.F90:1357-1361 -- note the flipped difference vs. the first pass.
    {
      Launch2D p = plan2d((ie + nt + 1) - (is - nt) + 1, (je + nt) - (js - nt) + 1, b.nk);
      deln_flux_x_kernel<<<p.grid, p.block, 0, s>>>(b, lf, is - nt, ie + nt + 1, js - nt,
                                                    je + nt, true, as_const(d2), del6_v,
                                                    fx2);
    }
    st = launch_copy_corners_cuda(b, 2, lf, d2, stream);                 // :1366
    if (st != FV3_GPU_SUCCESS) return st;
    // tp_core.F90:1369-1373
    {
      Launch2D p = plan2d((ie + nt) - (is - nt) + 1, (je + nt + 1) - (js - nt) + 1, b.nk);
      deln_flux_y_kernel<<<p.grid, p.block, 0, s>>>(b, lf, is - nt, ie + nt, js - nt,
                                                    je + nt + 1, true, as_const(d2),
                                                    del6_u, fy2);
    }
  }

  // tp_core.F90:1377-1394
  if (mass_weighted) {
    const fv_real damp2 = fv_real(0.5) * damp_coef;
    {
      Launch2D p = plan2d(ie - is + 2, je - js + 1, b.nk);
      deln_add_mass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, true, damp2, mass,
                                                      as_const(fx2), fx);
    }
    {
      Launch2D p = plan2d(ie - is + 1, je - js + 2, b.nk);
      deln_add_mass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, false, damp2, mass,
                                                      as_const(fy2), fy);
    }
  } else {
    {
      Launch2D p = plan2d(ie - is + 2, je - js + 1, b.nk);
      deln_add_no_mass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, true, as_const(fx2), fx);
    }
    {
      Launch2D p = plan2d(ie - is + 1, je - js + 2, b.nk);
      deln_add_no_mass_kernel<<<p.grid, p.block, 0, s>>>(b, lf, false, as_const(fy2), fy);
    }
  }
  return launch_status();
}

// --- WP-B.2 arbitrary scalar transport service ---------------------------
//
// Authoritative composition (tp_core.F90:109-243), i.e. one fv_tp_2d call plus
// its optional deln_flux tail. Every numerical stage below is the SHARED
// Milestone 3A.4 implementation -- xppm/yppm interior order-6 PPM, the
// one-sided panel-edge reconstruction, the smt5 seam override, the hord < 0
// positivity clamp, copy_corners and the q_i/q_j intermediates are reused
// unchanged, not re-derived. The only scalar-specific additions are the
// flux-averaging weight selection and the deln_flux tail.
fv3_gpu_status launch_scalar_transport_cuda(
    const TransportBounds &b, const ScalarTransportOptions &opt, const LevelFilter &lf,
    View3<fv_real> q, View3<const fv_real> mass, View3<const fv_real> crx_adv,
    View3<const fv_real> cry_adv, View3<const fv_real> xfx_adv,
    View3<const fv_real> yfx_adv, View3<const fv_real> mfx, View3<const fv_real> mfy,
    View2<const fv_real> area, View2<const fv_real> rarea, View2<const fv_real> dxa,
    View2<const fv_real> dya, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View3<const fv_real> ra_x, View3<const fv_real> ra_y, View3<fv_real> q_i,
    View3<fv_real> q_j, View3<fv_real> fx2_tp, View3<fv_real> fy2_tp, View3<fv_real> d2,
    View3<fv_real> fx2_dn, View3<fv_real> fy2_dn, View3<fv_real> fx, View3<fv_real> fy,
    void *stream) noexcept {
  // Reject BEFORE launching anything: an unsupported configuration must not
  // leave q half-rotated or fx/fy half-written.
  const int32_t mord = opt.hord >= 0 ? opt.hord : -opt.hord;
  if (mord != 6) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  // tp_core.F90:145-146: ord_in = 8 when hord == 10, else hord. abs(hord) == 6
  // excludes hord == 10, so ord_in == ord_ou == hord holds and the single-hord
  // launch_fv_tp_2d_cuda signature is faithful here.
  const bool deln = opt.deln_flux_enabled != 0;
  if (deln && (opt.nord < 0 || opt.nord > static_cast<int32_t>(b.ng) - 1))
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  const bool mass_weighted = opt.mass_flux_weighting != 0;
  // tp_core.F90:216-234: only the final averaging weight differs. xfx_adv /
  // yfx_adv still drive the q_i/q_j intermediates in BOTH branches.
  View3<const fv_real> wx = mass_weighted ? mfx : xfx_adv;
  View3<const fv_real> wy = mass_weighted ? mfy : yfx_adv;

  fv3_gpu_status st = launch_fv_tp_2d_cuda(b, opt.hord, lf, q, crx_adv, cry_adv, xfx_adv,
                                           yfx_adv, wx, wy, area, dxa, dya, ra_x, ra_y,
                                           q_i, q_j, fx, fy, fx2_tp, fy2_tp, stream);
  if (st != FV3_GPU_SUCCESS) return st;

  // tp_core.F90:212-214 / 230-232. The damp_c threshold branch belongs to
  // fv_tp_2d, not to deln_flux, so it is applied here. Note it is 1.e-4 --
  // del6_vt_flux's activation threshold is 1.E-5 and the two must not be
  // conflated.
  if (!deln || !(opt.damp_c > fv_real(1.0e-4))) return FV3_GPU_SUCCESS;

  // Fortran passes `mass` only in the mfx/mfy branch (tp_core.F90:222-224) and
  // never in the xfx/yfx branch (:236-238), so the weighting mode selects the
  // deln_flux mass mode as well.
  return launch_deln_flux_transport_cuda(b, opt.nord, opt.damp_c, opt.da_min,
                                         mass_weighted, lf, as_const(q), mass, del6_u,
                                         del6_v, rarea, fx, fy, d2, fx2_dn, fy2_dn,
                                         stream);
}

} // namespace fv3::gpu::dgrid
