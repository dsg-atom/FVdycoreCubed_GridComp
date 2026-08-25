// Milestone 3A.7 / WP-B.4 -- authoritative FV3 divergence damping, Smagorinsky
// corner strain, the d_con heating chain and the local cubed-sphere corner
// rotations d_sw needs.
//
// Every formula below was transcribed from
//   GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
//   SHA ca30e26305fda43831095334c02a3f5e0b8c0c45
// model/sw_core.F90 (d_sw :1277-1534, smag_corner :1857-1947) and
// tools/fv_mp_mod.F90 (fill_corners_2d_r8 :1113-1186,
// fill_corners_dgrid_r8 :1338-1370).
//
// The contract, the ordered operation table and the anti-dependence analysis
// live in dgrid/wp_b_divergence_cuda.cuh. This file adds no numerical decision
// that is not in the Fortran.

#include "dgrid/wp_b_divergence_cuda.cuh"
#include "dgrid/wp_b_ppm_shared.cuh"
#include "operators/wp_d_cuda.hpp"

#include <cmath>

namespace fv3::gpu::dgrid {
namespace {

// ---------------------------------------------------------------------------
// per-level activation
// ---------------------------------------------------------------------------

// DivergenceLevels' counterpart of level_active. It is a separate predicate
// because dyn_core's per-k quantities for this block (nord_k, d2_divg, d4dmp,
// d_con_k) are not LevelFilter's (damp_vt, nord_v) and the two must not be
// conflated -- see the header's §3 and §8.
__device__ __forceinline__ int64_t lv_clamp(const DivergenceLevels &lv, int64_t k) {
  return k < lv.npz ? k : lv.npz - 1;
}

__device__ __forceinline__ int64_t lv_nord(const DivergenceLevels &lv, int64_t k) {
  // Integral values travel through an fv_real array (the public C ABI admits
  // only F32/F64), so they are recovered with a nearest-integer round rather
  // than a truncating cast.
  return static_cast<int64_t>(llrint(static_cast<double>(lv.nord_k(lv_clamp(lv, k)))));
}

__device__ __forceinline__ bool divergence_level_active(const DivergenceLevels &lv,
                                                        int64_t k) {
  if (lv.mode == 0) return true;
  if (lv.mode == 3) return lv.d_con(lv_clamp(lv, k)) > damp_threshold();
  const int64_t nord = lv_nord(lv, k);
  if (lv.mode == 1) return nord == 0;
  return nord == static_cast<int64_t>(lv.nord_sel);
}

// ---------------------------------------------------------------------------
// configuration rejection -- always before any launch
// ---------------------------------------------------------------------------

// The edge/corner flags and the tile indices are two spellings of the same
// fact in the authoritative Fortran, which tests `is==1` / `(ie+1)==npx`
// literally at every site. A caller that disagrees with itself would get the
// interior stencil where FV3 uses the one-sided form (or vice versa) with no
// diagnostic at all, so the disagreement is rejected instead.
//
// Note the BGRID fill_corners predicates `ie==npx-1` / `je==npy-1`
// (fv_mp_mod.F90:1119-1122) and the DGRID ones `(ie+1)==npx` / `(je+1)==npy`
// (fv_mp_mod.F90:1347-1350) are algebraically identical, so one flag serves
// both. The header previously claimed these were different index conventions;
// they are not.
bool divergence_flags_consistent(const TransportBounds &b) {
  if ((b.west_edge != 0) != (b.is == 1)) return false;
  if ((b.east_edge != 0) != (b.ie + 1 == b.npx)) return false;
  if ((b.south_edge != 0) != (b.js == 1)) return false;
  if ((b.north_edge != 0) != (b.je + 1 == b.npy)) return false;
  if ((b.sw_corner != 0) != (b.west_edge && b.south_edge)) return false;
  if ((b.se_corner != 0) != (b.east_edge && b.south_edge)) return false;
  if ((b.ne_corner != 0) != (b.east_edge && b.north_edge)) return false;
  if ((b.nw_corner != 0) != (b.west_edge && b.north_edge)) return false;
  return true;
}

// Shape/decomposition checks that every launcher in this file needs, including
// the two fill_corners rotations (which are no-ops rather than errors on a
// nested or doubly-periodic domain and so must not reject those).
fv3_gpu_status validate_divergence_bounds(const TransportBounds &b) {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.is > b.ie || b.js > b.je) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.npx <= 0 || b.npy <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.ng < 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!divergence_flags_consistent(b)) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  return FV3_GPU_SUCCESS;
}

fv3_gpu_status validate_levels(const DivergenceLevels &lv, bool needs_nord,
                               bool needs_d2, bool needs_d4, bool needs_dcon) {
  if (lv.npz <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (lv.mode < 0 || lv.mode > 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  const bool reads_nord = needs_nord || lv.mode == 1 || lv.mode == 2;
  if (reads_nord && lv.nord_k.data == nullptr)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if ((needs_dcon || lv.mode == 3) && lv.d_con.data == nullptr)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (needs_d2 && lv.d2_bg.data == nullptr) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (needs_d4 && lv.d4_bg.data == nullptr) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  return FV3_GPU_SUCCESS;
}

// The full check for the numerical launchers. Branch L's `else` arm
// (sw_core.F90:1297-1336) is the non-nested cubed-sphere one; the nested arm at
// :1281-1295 drops every sin_sg switch and is a different algorithm, so a
// nested or non-cubed-sphere domain is rejected rather than silently given the
// wrong stencil.
fv3_gpu_status validate_divergence(const TransportBounds &b,
                                   const DivergenceDampingOptions &opt) {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  if (opt.grid_type != 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (opt.do_skeb != 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.nested != 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!b.cubed_sphere) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  return FV3_GPU_SUCCESS;
}

// ---------------------------------------------------------------------------
// branch L -- sw_core.F90:1277-1358, the nord == 0 arm
// ---------------------------------------------------------------------------

// L1, sw_core.F90:1298-1313. j == 1 / j == npy replace the cosa_v/sina_v
// contravariant form with the one-sided sin_sg form, keyed on sign(vc).
// sin_sg components are 0-based here: Fortran sin_sg(:,:,4) is component 3 and
// sin_sg(:,:,2) is component 1.
__global__ void divdamp_ptc_kernel(TransportBounds b, DivergenceLevels lv,
                                   View3<const fv_real> u, View3<const fv_real> va,
                                   View3<const fv_real> vc, DivergenceGeometry g,
                                   View3<fv_real> ptc) {
  const int64_t i = (b.is - 1) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;

  if (j == 1 || j == b.npy) {
    ptc(i, j, k) = vc(i, j, k) > fv_real(0)
                       ? u(i, j, k) * g.dyc(i, j) * g.sin_sg(i, j - 1, 3)
                       : u(i, j, k) * g.dyc(i, j) * g.sin_sg(i, j, 1);
  } else {
    ptc(i, j, k) =
        (u(i, j, k) - fv_real(0.5) * (va(i, j - 1, k) + va(i, j, k)) * g.cosa_v(i, j)) *
        g.dyc(i, j) * g.sina_v(i, j);
  }
}

// L2, sw_core.F90:1315-1335. The interior loop runs i = is2..ie1 with
// is2 = max(2,is) and ie1 = min(npx-1,ie+1) (sw_core.F90:1072); the two columns
// that clamp excludes are then overwritten by the sin_sg forms. Launching over
// the union is..ie+1 is exact because the three index sets are disjoint:
// i == 1 exists only when is == 1 (so is2 == 2) and i == npx only when
// ie+1 == npx (so ie1 == npx-1).
__global__ void divdamp_vort_kernel(TransportBounds b, DivergenceLevels lv,
                                    View3<const fv_real> v, View3<const fv_real> ua,
                                    View3<const fv_real> uc, DivergenceGeometry g,
                                    View3<fv_real> vort) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - 1) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;

  if (b.west_edge && i == 1) {
    vort(1, j, k) = uc(1, j, k) > fv_real(0)
                        ? v(1, j, k) * g.dxc(1, j) * g.sin_sg(0, j, 2)
                        : v(1, j, k) * g.dxc(1, j) * g.sin_sg(1, j, 0);
    return;
  }
  if (b.east_edge && i == b.npx) {
    vort(b.npx, j, k) =
        uc(b.npx, j, k) > fv_real(0)
            ? v(b.npx, j, k) * g.dxc(b.npx, j) * g.sin_sg(b.npx - 1, j, 2)
            : v(b.npx, j, k) * g.dxc(b.npx, j) * g.sin_sg(b.npx, j, 0);
    return;
  }
  const int64_t is2 = b.is < 2 ? 2 : b.is;
  const int64_t ie1 = (b.ie + 1) < (b.npx - 1) ? (b.ie + 1) : (b.npx - 1);
  if (i < is2 || i > ie1) return;
  vort(i, j, k) =
      (v(i, j, k) - fv_real(0.5) * (ua(i - 1, j, k) + ua(i, j, k)) * g.cosa_u(i, j)) *
      g.dxc(i, j) * g.sina_u(i, j);
}

// L3, sw_core.F90:1338-1342.
__global__ void divdamp_delpc_kernel(TransportBounds b, DivergenceLevels lv,
                                     View3<const fv_real> vort,
                                     View3<const fv_real> ptc, View3<fv_real> delpc) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  delpc(i, j, k) =
      vort(i, j - 1, k) - vort(i, j, k) + ptc(i - 1, j, k) - ptc(i, j, k);
}

// L4, sw_core.F90:1345-1349 and H2g, sw_core.F90:1400-1403. The two are the
// same four corrections with a different source array (vort vs uc), and both
// run BEFORE any rarea_c scaling, so one kernel serves both.
//
// Note the sign pattern: the south corners subtract, the north corners add.
__global__ void corner_term_removal_kernel(TransportBounds b, DivergenceLevels lv,
                                           View3<const fv_real> src,
                                           View3<fv_real> dst) {
  // Launched as plan2d(1, 1, nk), i.e. one 16x16 block per level. Only lane
  // (0,0) may write: letting all 256 lanes store the same value to the same
  // address is a genuine data race and racecheck reports it even though the
  // stored value would be identical.
  if (threadIdx.x != 0 || threadIdx.y != 0) return;
  const int64_t k = blockIdx.z;
  if (k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  const int64_t npx = b.npx, npy = b.npy;
  if (b.sw_corner) dst(1, 1, k) = dst(1, 1, k) - src(1, 0, k);
  if (b.se_corner) dst(npx, 1, k) = dst(npx, 1, k) - src(npx, 0, k);
  if (b.ne_corner) dst(npx, npy, k) = dst(npx, npy, k) + src(npx, npy, k);
  if (b.nw_corner) dst(1, npy, k) = dst(1, npy, k) + src(1, npy, k);
}

// L5, sw_core.F90:1351-1358. delpc is scaled in place, then the limiter uses
// the SCALED value -- and here, unlike H4, inside an abs().
__global__ void divdamp_nord0_finalize_kernel(TransportBounds b,
                                             DivergenceDampingOptions opt,
                                             DivergenceLevels lv, fv_real dt,
                                             DivergenceGeometry g, View3<fv_real> delpc,
                                             View3<fv_real> vort, View3<fv_real> ke) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;

  const fv_real d2 = lv.d2_bg(lv_clamp(lv, k));
  const fv_real dc = g.rarea_c(i, j) * delpc(i, j, k);
  delpc(i, j, k) = dc;
  const fv_real lim = fmin(fv_real(0.20), opt.dddmp * fabs(dc * dt));
  const fv_real damp = opt.da_min_c * fmax(d2, lim);
  const fv_real inc = damp * dc;
  vort(i, j, k) = inc;
  ke(i, j, k) = ke(i, j, k) + inc;
}

// ---------------------------------------------------------------------------
// branch H -- sw_core.F90:1360-1421, the higher-order arm
// ---------------------------------------------------------------------------

// H1, sw_core.F90:1364-1368 -- the external-mode filter save.
__global__ void divdamp_save_delpc_kernel(TransportBounds b, DivergenceLevels lv,
                                          View3<const fv_real> divg_d,
                                          View3<fv_real> delpc) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  delpc(i, j, k) = divg_d(i, j, k);
}

// H2b, sw_core.F90:1378-1382 -- vc is CLOBBERED, see the header's §4(a).
__global__ void divdamp_divg_dx_kernel(TransportBounds b, DivergenceLevels lv, int64_t nt,
                                       View3<const fv_real> divg_d,
                                       View2<const fv_real> divg_u, View3<fv_real> vc) {
  const int64_t i = (b.is - 1 - nt) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - nt) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 + nt || j > b.je + 1 + nt || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  vc(i, j, k) = (divg_d(i + 1, j, k) - divg_d(i, j, k)) * divg_u(i, j);
}

// H2d, sw_core.F90:1385-1389 -- uc is CLOBBERED.
__global__ void divdamp_divg_dy_kernel(TransportBounds b, DivergenceLevels lv, int64_t nt,
                                       View3<const fv_real> divg_d,
                                       View2<const fv_real> divg_v, View3<fv_real> uc) {
  const int64_t i = (b.is - nt) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - 1 - nt) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 + nt || j > b.je + 1 + nt || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  uc(i, j, k) = (divg_d(i, j + 1, k) - divg_d(i, j, k)) * divg_v(i, j);
}

// H2f, sw_core.F90:1392-1396.
__global__ void divdamp_recombine_kernel(TransportBounds b, DivergenceLevels lv,
                                         int64_t nt, View3<const fv_real> uc,
                                         View3<const fv_real> vc,
                                         View3<fv_real> divg_d) {
  const int64_t i = (b.is - nt) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - nt) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 + nt || j > b.je + 1 + nt || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  divg_d(i, j, k) = uc(i, j - 1, k) - uc(i, j, k) + vc(i - 1, j, k) - vc(i, j, k);
}

// H3, sw_core.F90:1408. `vort(:,:) = 0.` covers the WHOLE declared array
// isd..ied x jsd..jed, not the compute box.
__global__ void divdamp_vort_zero_kernel(TransportBounds b, DivergenceLevels lv,
                                         View3<fv_real> vort) {
  const int64_t i = (b.is - b.ng) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - b.ng) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + b.ng || j > b.je + b.ng || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  vort(i, j, k) = fv_real(0);
}

// H4, sw_core.F90:1413-1421.
//   dd8   = da_min * d4_bg**n2   with n2 = nord_k + 1
//   damp2 = da_min_c * max(d2_bg, min(0.20, dddmp*vort))   -- NO abs() here
//   vort  = damp2*delpc + dd8*divg_d ;  ke += vort
__global__ void divdamp_high_order_finalize_kernel(TransportBounds b,
                                                   DivergenceDampingOptions opt,
                                                   DivergenceLevels lv,
                                                   View3<const fv_real> delpc,
                                                   View3<const fv_real> divg_d,
                                                   View3<fv_real> vort,
                                                   View3<fv_real> ke) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;

  const int64_t kk = lv_clamp(lv, k);
  const int64_t n2 = lv_nord(lv, k) + 1;
  // Left-to-right repeated multiplication, matching the naive expansion of the
  // Fortran's integer power rather than a libm pow() call.
  fv_real p = lv.d4_bg(kk);
  for (int64_t e = 1; e < n2; ++e) p *= lv.d4_bg(kk);
  const fv_real dd8 = opt.da_min * p;

  const fv_real lim = fmin(fv_real(0.20), opt.dddmp * vort(i, j, k));
  const fv_real damp2 = opt.da_min_c * fmax(lv.d2_bg(kk), lim);
  const fv_real inc = damp2 * delpc(i, j, k) + dd8 * divg_d(i, j, k);
  vort(i, j, k) = inc;
  ke(i, j, k) = ke(i, j, k) + inc;
}

// ---------------------------------------------------------------------------
// fill_corners -- tools/fv_mp_mod.F90, LOCAL panel index algebra
// ---------------------------------------------------------------------------

// fill_corners_2d_r8, BGRID branch (fv_mp_mod.F90:1117-1148). dir == 2 is
// FILL=YDir; every other value takes the XDir path, which is exactly what the
// Fortran's `case default` does.
//
// One launch covers all four corners and the whole ng x ng block. Within a
// direction the write set (halo corner blocks) and the read set are disjoint:
// XDir writes rows/cols offset by <= 0 or >= npx+1 and reads at second index
// 2..4 or npy-3..npy-1, and YDir writes reading at first index 2..4 or
// npx-3..npx-1. Verified corner by corner against the Fortran.
__global__ void fill_corners_bgrid_kernel(TransportBounds b, int32_t dir,
                                          DivergenceLevels lv, View3<fv_real> q) {
  const int64_t i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ng || j > b.ng || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  const int64_t npx = b.npx, npy = b.npy;
  const bool w = b.west_edge != 0, e = b.east_edge != 0;
  const bool s = b.south_edge != 0, n = b.north_edge != 0;

  if (dir == 2) {  // YDir, fv_mp_mod.F90:1134-1141
    if (w && s) q(1 - j, 1 - i, k) = q(i + 1, 1 - j, k);
    if (w && n) q(1 - j, npy + i, k) = q(i + 1, npy + j, k);
    if (e && s) q(npx + j, 1 - i, k) = q(npx - i, 1 - j, k);
    if (e && n) q(npx + j, npy + i, k) = q(npx - i, npy + j, k);
  } else {  // XDir, fv_mp_mod.F90:1119-1126
    if (w && s) q(1 - i, 1 - j, k) = q(1 - j, i + 1, k);
    if (w && n) q(1 - i, npy + j, k) = q(1 - j, npy - i, k);
    if (e && s) q(npx + i, 1 - j, k) = q(npx + j, i + 1, k);
    if (e && n) q(npx + i, npy + j, k) = q(npx + j, npy - i, k);
  }
}

// fill_corners_dgrid_r8 (fv_mp_mod.F90:1345-1368), reached from
// fill_corners_xy_2d with VECTOR=.true. so mySign = -1.
//
// Two things a symmetric-looking port gets wrong:
//   * mySign multiplies the SW and NE corners ONLY. NW and SE are unsigned.
//     (The commented-out block just above the live code puts the signs on
//     NW/SE and reads different indices -- it is dead and must be ignored.)
//   * the x writes use npx-1+i / npy+j while the y writes use npx+i / npy-1+j.
//     That +-1 asymmetry IS the D-grid staggering, not a typo.
//
// The two Fortran loops are kept as two launches. Fusing them is race-free only
// for npx, npy >= 5 (loop 2 reads x at first index npx-3..npx-1 while loop 1
// writes x at npx..npx+2), and buying one saved launch with an npx-dependent
// hazard is not worth it. The header's earlier claim that the fusion is
// unconditionally safe was too strong.
__global__ void fill_corners_dgrid_x_kernel(TransportBounds b, DivergenceLevels lv,
                                            fv_real my_sign, View3<fv_real> x,
                                            View3<const fv_real> y) {
  const int64_t i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ng || j > b.ng || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  const int64_t npx = b.npx, npy = b.npy;
  const bool w = b.west_edge != 0, e = b.east_edge != 0;
  const bool s = b.south_edge != 0, n = b.north_edge != 0;

  if (w && s) x(1 - i, 1 - j, k) = my_sign * y(1 - j, i, k);
  if (w && n) x(1 - i, npy + j, k) = y(1 - j, npy - i, k);
  if (e && s) x(npx - 1 + i, 1 - j, k) = y(npx + j, i, k);
  if (e && n) x(npx - 1 + i, npy + j, k) = my_sign * y(npx + j, npy - i, k);
}

__global__ void fill_corners_dgrid_y_kernel(TransportBounds b, DivergenceLevels lv,
                                            fv_real my_sign, View3<const fv_real> x,
                                            View3<fv_real> y) {
  const int64_t i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ng || j > b.ng || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  const int64_t npx = b.npx, npy = b.npy;
  const bool w = b.west_edge != 0, e = b.east_edge != 0;
  const bool s = b.south_edge != 0, n = b.north_edge != 0;

  if (w && s) y(1 - i, 1 - j, k) = my_sign * x(j, 1 - i, k);
  if (w && n) y(1 - i, npy - 1 + j, k) = x(j, npy + i, k);
  if (e && s) y(npx + i, 1 - j, k) = x(npx - j, 1 - i, k);
  if (e && n) y(npx + i, npy - 1 + j, k) = my_sign * x(npx - j, npy + i, k);
}

// ---------------------------------------------------------------------------
// smag_corner -- sw_core.F90:1903-1945
// ---------------------------------------------------------------------------

// S1/S2: ut = u*dyc over is-1..ie+1 x js..je+1, vt = v*dxc over is..ie+1 x
// js-1..je+1. Both are the routine's OWN locals, NOT d_sw's ut/vt.
__global__ void smag_tension_winds_kernel(TransportBounds b, View3<const fv_real> u,
                                          View3<const fv_real> v,
                                          DivergenceGeometry g, View3<fv_real> ut,
                                          View3<fv_real> vt) {
  const int64_t i = (b.is - 1) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - 1) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (i >= b.is - 1 && j >= b.js) ut(i, j, k) = u(i, j, k) * g.dyc(i, j);
  if (i >= b.is) vt(i, j, k) = v(i, j, k) * g.dxc(i, j);
}

// S3, sw_core.F90:1915. rarea_c is applied INSIDE the sum, i.e. BEFORE the
// corner removals at :1919-1922 -- the opposite order from divergence_corner
// and from branch L, so the corner correction here is added UNSCALED. This is
// reproduced deliberately; see the header's §6.
//
// The tension sign pattern is (vt(i,j-1) - vt(i,j) - ut(i-1,j) + ut(i,j)):
// the ut pair is sign-flipped relative to divergence_corner's
// (vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j)).
__global__ void smag_tension_kernel(TransportBounds b, DivergenceLevels lv,
                                    DivergenceGeometry g, View3<const fv_real> ut,
                                    View3<const fv_real> vt, View3<fv_real> smag_c) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  smag_c(i, j, k) = g.rarea_c(i, j) * (vt(i, j - 1, k) - vt(i, j, k) -
                                       ut(i - 1, j, k) + ut(i, j, k));
}

// S5/S6: vt = u*dx over isd..ied x jsd..jed+1, ut = v*dy over isd..ied+1 x
// jsd..jed. These OVERWRITE the tension winds, so S3 and S4 must already be
// complete -- guaranteed by stream ordering.
__global__ void smag_shear_winds_kernel(TransportBounds b, View3<const fv_real> u,
                                        View3<const fv_real> v, DivergenceGeometry g,
                                        View3<fv_real> ut, View3<fv_real> vt) {
  const int64_t i = (b.is - b.ng) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - b.ng) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  const int64_t ied = b.ie + b.ng, jed = b.je + b.ng;
  if (i > ied + 1 || j > jed + 1 || k >= b.nk) return;
  if (i <= ied && j <= jed + 1) vt(i, j, k) = u(i, j, k) * g.dx(i, j);
  if (i <= ied + 1 && j <= jed) ut(i, j, k) = v(i, j, k) * g.dy(i, j);
}

// S7, sw_core.F90:1937. Note the sign pattern
// (vt(i,j) - vt(i,j+1) + ut(i,j) - ut(i+1,j)): d_sw's own relative vorticity
// (sw_core.F90:1233) uses the same two arrays with the OPPOSITE sign on the
// ut pair. Do not share code between them.
__global__ void smag_shear_kernel(TransportBounds b, DivergenceGeometry g,
                                  View3<const fv_real> ut, View3<const fv_real> vt,
                                  View3<fv_real> wk) {
  const int64_t i = (b.is - b.ng) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - b.ng) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + b.ng || j > b.je + b.ng || k >= b.nk) return;
  wk(i, j, k) = g.rarea(i, j) * (vt(i, j, k) - vt(i, j + 1, k) + ut(i, j, k) -
                                 ut(i + 1, j, k));
}

// S9, sw_core.F90:1943.
__global__ void smag_combine_kernel(TransportBounds b, DivergenceLevels lv, fv_real dt,
                                    View3<const fv_real> sh, View3<fv_real> smag_c) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  const fv_real s = sh(i, j, k), t = smag_c(i, j, k);
  smag_c(i, j, k) = dt * sqrt(s * s + t * t);
}

// ---------------------------------------------------------------------------
// the d_con chain -- sw_core.F90:1425-1534
// ---------------------------------------------------------------------------

// D1, sw_core.F90:1425-1436. Two Fortran loops over different rectangles; they
// read only vort so one launch is safe.
__global__ void dcon_vort_diff_kernel(TransportBounds b, DivergenceLevels lv,
                                      View3<const fv_real> vort, View3<fv_real> ub,
                                      View3<fv_real> vb) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  if (i <= b.ie) ub(i, j, k) = vort(i, j, k) - vort(i + 1, j, k);
  if (j <= b.je) vb(i, j, k) = vort(i, j, k) - vort(i, j + 1, k);
}

// D2, sw_core.F90:1482-1495. ub/vb are updated in place; fx/fy are overwritten
// (they held fv_tp_2d's vorticity fluxes, already consumed at :1461-1471).
__global__ void dcon_scale_kernel(TransportBounds b, DivergenceLevels lv,
                                  DivergenceGeometry g, View3<const fv_real> u,
                                  View3<const fv_real> v, View3<const fv_real> ut,
                                  View3<const fv_real> vt, View3<fv_real> ub,
                                  View3<fv_real> vb, View3<fv_real> fx,
                                  View3<fv_real> fy, View3<fv_real> gx,
                                  View3<fv_real> gy) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;
  if (i <= b.ie) {
    const fv_real rdx = g.rdx(i, j);
    const fv_real ubv = (ub(i, j, k) + vt(i, j, k)) * rdx;
    const fv_real fyv = u(i, j, k) * rdx;
    ub(i, j, k) = ubv;
    fy(i, j, k) = fyv;
    gy(i, j, k) = fyv * ubv;
  }
  if (j <= b.je) {
    const fv_real rdy = g.rdy(i, j);
    const fv_real vbv = (vb(i, j, k) - ut(i, j, k)) * rdy;
    const fv_real fxv = v(i, j, k) * rdy;
    vb(i, j, k) = vbv;
    fx(i, j, k) = fxv;
    gx(i, j, k) = fxv * vbv;
  }
}

// D3, sw_core.F90:1497-1520. heat_source is genuinely an accumulator: the
// pre-existing value (the w-damping contribution) is INSIDE the parenthesis
// that delp multiplies. diss_est is not produced -- it exists only under
// do_skeb, which this milestone rejects.
__global__ void dcon_heat_source_kernel(TransportBounds b, DivergenceLevels lv,
                                        DivergenceGeometry g,
                                        View3<const fv_real> delp,
                                        View3<const fv_real> ub,
                                        View3<const fv_real> vb,
                                        View3<const fv_real> fx,
                                        View3<const fv_real> fy,
                                        View3<const fv_real> gx,
                                        View3<const fv_real> gy,
                                        View3<fv_real> heat_source) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie || j > b.je || k >= b.nk) return;
  if (!divergence_level_active(lv, k)) return;

  const fv_real damp = fv_real(0.25) * lv.d_con(lv_clamp(lv, k));
  const fv_real u2 = fy(i, j, k) + fy(i, j + 1, k);
  const fv_real du2 = ub(i, j, k) + ub(i, j + 1, k);
  const fv_real v2 = fx(i, j, k) + fx(i + 1, j, k);
  const fv_real dv2 = vb(i, j, k) + vb(i + 1, j, k);
  const fv_real ubc = ub(i, j, k), ubn = ub(i, j + 1, k);
  const fv_real vbc = vb(i, j, k), vbe = vb(i + 1, j, k);
  heat_source(i, j, k) =
      delp(i, j, k) *
      (heat_source(i, j, k) -
       damp * g.rsin2(i, j) *
           ((ubc * ubc + ubn * ubn + vbc * vbc + vbe * vbe) +
            fv_real(2) * (gy(i, j, k) + gy(i, j + 1, k) + gx(i, j, k) +
                          gx(i + 1, j, k)) -
            g.cosa_s(i, j) * (u2 * dv2 + v2 * du2 + du2 * dv2)));
}

// V2, sw_core.F90:1523-1534.
__global__ void dwind_damping_add_kernel(TransportBounds b, LevelFilter lf,
                                         View3<const fv_real> ut,
                                         View3<const fv_real> vt, View3<fv_real> u,
                                         View3<fv_real> v) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  if (!level_active(lf, k)) return;
  if (i <= b.ie) u(i, j, k) = u(i, j, k) + vt(i, j, k);
  if (j <= b.je) v(i, j, k) = v(i, j, k) - ut(i, j, k);
}

}  // namespace

// =========================== launchers ==================================

fv3_gpu_status launch_divdamp_nord0_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, fv_real dt, View3<const fv_real> u,
    View3<const fv_real> v, View3<const fv_real> ua, View3<const fv_real> va,
    View3<const fv_real> uc, View3<const fv_real> vc, const DivergenceGeometry &g,
    View3<fv_real> ptc, View3<fv_real> vort, View3<fv_real> delpc, View3<fv_real> ke,
    void *stream) noexcept {
  fv3_gpu_status st = validate_divergence(b, opt);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, /*nord*/ false, /*d2*/ true, /*d4*/ false, /*dcon*/ false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!u.data || !v.data || !ua.data || !va.data || !uc.data || !vc.data ||
      !ptc.data || !vort.data || !delpc.data || !ke.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!g.rarea_c.data || !g.dxc.data || !g.dyc.data || !g.cosa_u.data ||
      !g.cosa_v.data || !g.sina_u.data || !g.sina_v.data || !g.sin_sg.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  auto s = static_cast<cudaStream_t>(stream);
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;

  {  // L1
    Launch2D p = plan2d((ie + 1) - (is - 1) + 1, (je + 1) - js + 1, b.nk);
    divdamp_ptc_kernel<<<p.grid, p.block, 0, s>>>(b, lv, u, va, vc, g, ptc);
  }
  {  // L2
    Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - (js - 1) + 1, b.nk);
    divdamp_vort_kernel<<<p.grid, p.block, 0, s>>>(b, lv, v, ua, uc, g, vort);
  }
  {  // L3
    Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - js + 1, b.nk);
    divdamp_delpc_kernel<<<p.grid, p.block, 0, s>>>(b, lv, as_const(vort),
                                                    as_const(ptc), delpc);
  }
  if (b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner) {  // L4
    Launch2D p = plan2d(1, 1, b.nk);
    corner_term_removal_kernel<<<p.grid, p.block, 0, s>>>(b, lv, as_const(vort), delpc);
  }
  {  // L5
    Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - js + 1, b.nk);
    divdamp_nord0_finalize_kernel<<<p.grid, p.block, 0, s>>>(b, opt, lv, dt, g, delpc,
                                                             vort, ke);
  }
  return launch_status();
}

fv3_gpu_status launch_divdamp_high_order_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, int32_t nord_cap, const DivergenceGeometry &g,
    View3<fv_real> divg_d, View3<fv_real> uc, View3<fv_real> vc, View3<fv_real> delpc,
    void *stream) noexcept {
  fv3_gpu_status st = validate_divergence(b, opt);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, /*nord*/ true, /*d2*/ false, /*d4*/ false, /*dcon*/ false);
  if (st != FV3_GPU_SUCCESS) return st;
  // nt = nord - 1 at the first sweep, and H2b reads divg_d(ie+2+nt), so
  // nord > 3 would index past ied+1 = ie+ng+1. That is the whole reason the
  // authoritative halo is exactly wide enough for nord <= 3.
  if (nord_cap < 1 || nord_cap > 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!divg_d.data || !uc.data || !vc.data || !delpc.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!g.divg_u.data || !g.divg_v.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  auto s = static_cast<cudaStream_t>(stream);
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  const bool any_corner = b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner;

  // lv.mode / lv.nord_sel are ignored: this launcher builds its own mode-2
  // group filter per nord value, because a single column can carry several
  // distinct nord_k (dyn_core's sponge overrides, header §3).
  for (int32_t nord = 1; nord <= nord_cap; ++nord) {
    DivergenceLevels gl = lv;
    gl.mode = 2;
    gl.nord_sel = nord;

    {  // H1
      Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - js + 1, b.nk);
      divdamp_save_delpc_kernel<<<p.grid, p.block, 0, s>>>(b, gl, as_const(divg_d),
                                                           delpc);
    }

    for (int32_t n = 1; n <= nord; ++n) {
      const int64_t nt = nord - n;
      // sw_core.F90:1374-1376. grid_type < 3 is guaranteed by validate_*.
      const bool fill_c = (nt != 0) && any_corner && !b.nested && b.cubed_sphere;

      if (fill_c) {  // H2a
        st = launch_fill_corners_bgrid_cuda(b, 1, gl, divg_d, stream);
        if (st != FV3_GPU_SUCCESS) return st;
      }
      {  // H2b
        Launch2D p = plan2d((ie + 1 + nt) - (is - 1 - nt) + 1,
                            (je + 1 + nt) - (js - nt) + 1, b.nk);
        divdamp_divg_dx_kernel<<<p.grid, p.block, 0, s>>>(b, gl, nt, as_const(divg_d),
                                                          g.divg_u, vc);
      }
      if (fill_c) {  // H2c
        st = launch_fill_corners_bgrid_cuda(b, 2, gl, divg_d, stream);
        if (st != FV3_GPU_SUCCESS) return st;
      }
      {  // H2d
        Launch2D p = plan2d((ie + 1 + nt) - (is - nt) + 1,
                            (je + 1 + nt) - (js - 1 - nt) + 1, b.nk);
        divdamp_divg_dy_kernel<<<p.grid, p.block, 0, s>>>(b, gl, nt, as_const(divg_d),
                                                          g.divg_v, uc);
      }
      if (fill_c) {  // H2e -- x = vc, y = uc, mySign = -1
        st = launch_fill_corners_dgrid_vector_cuda(b, gl, vc, uc, stream);
        if (st != FV3_GPU_SUCCESS) return st;
      }
      {  // H2f
        Launch2D p =
            plan2d((ie + 1 + nt) - (is - nt) + 1, (je + 1 + nt) - (js - nt) + 1, b.nk);
        divdamp_recombine_kernel<<<p.grid, p.block, 0, s>>>(b, gl, nt, as_const(uc),
                                                            as_const(vc), divg_d);
      }
      if (any_corner) {  // H2g -- runs every sweep, even when fill_c is false
        Launch2D p = plan2d(1, 1, b.nk);
        corner_term_removal_kernel<<<p.grid, p.block, 0, s>>>(b, gl, as_const(uc),
                                                              divg_d);
      }
    }
  }
  return launch_status();
}

fv3_gpu_status launch_divdamp_vort_zero_cuda(const TransportBounds &b,
                                             const DivergenceLevels &lv,
                                             View3<fv_real> vort,
                                             void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!vort.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  const int64_t h = b.ng;
  Launch2D p =
      plan2d((b.ie + h) - (b.is - h) + 1, (b.je + h) - (b.js - h) + 1, b.nk);
  divdamp_vort_zero_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lv, vort);
  return launch_status();
}

fv3_gpu_status launch_smag_corner_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, fv_real dt, View3<const fv_real> u,
    View3<const fv_real> v, const DivergenceGeometry &g, const A2BGeometry &a2b,
    const SmagCornerWorkspace &ws, View3<fv_real> smag_c, void *stream) noexcept {
  fv3_gpu_status st = validate_divergence(b, opt);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!u.data || !v.data || !smag_c.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!ws.ut.data || !ws.vt.data || !ws.wk.data || !ws.sh.data || !ws.qx.data ||
      !ws.qy.data || !ws.qxx.data || !ws.qyy.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!g.rarea_c.data || !g.rarea.data || !g.dxc.data || !g.dyc.data || !g.dx.data ||
      !g.dy.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  // WP-D's a2b launcher requires an explicit stream (it rejects a null `s`),
  // so smag_corner does too rather than silently taking the default stream for
  // its own kernels and failing only at the a2b call.
  if (stream == nullptr) return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  auto s = static_cast<cudaStream_t>(stream);
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, h = b.ng;

  {  // S1 + S2 -- unfiltered: the a2b call below has no per-level filter and
     // must not read a partially initialised wk.
    Launch2D p = plan2d((ie + 1) - (is - 1) + 1, (je + 1) - (js - 1) + 1, b.nk);
    smag_tension_winds_kernel<<<p.grid, p.block, 0, s>>>(b, u, v, g, ws.ut, ws.vt);
  }
  {  // S3
    Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - js + 1, b.nk);
    smag_tension_kernel<<<p.grid, p.block, 0, s>>>(b, lv, g, as_const(ws.ut),
                                                   as_const(ws.vt), smag_c);
  }
  if (b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner) {
    // S4 -- the same four corrections as L4/H2g, with vt as the source. Note
    // smag_c has ALREADY been scaled by rarea_c, so this correction is added
    // unscaled. That asymmetry is authoritative.
    Launch2D p = plan2d(1, 1, b.nk);
    corner_term_removal_kernel<<<p.grid, p.block, 0, s>>>(b, lv, as_const(ws.vt),
                                                          smag_c);
  }
  {  // S5 + S6 -- these OVERWRITE ut/vt, so S3/S4 must be done (stream order)
    Launch2D p = plan2d((ie + h + 1) - (is - h) + 1, (je + h + 1) - (js - h) + 1, b.nk);
    smag_shear_winds_kernel<<<p.grid, p.block, 0, s>>>(b, u, v, g, ws.ut, ws.vt);
  }
  {  // S7
    Launch2D p = plan2d((ie + h) - (is - h) + 1, (je + h) - (js - h) + 1, b.nk);
    smag_shear_kernel<<<p.grid, p.block, 0, s>>>(b, g, as_const(ws.ut),
                                                 as_const(ws.vt), ws.wk);
  }
  {  // S8 -- a2b_ord4(wk, sh, ..., replace = .false.), REUSED from WP-D.
    operators::HorizontalBounds hb{is, ie, js, je, b.nk};
    operators::PhysicalEdges e{b.west_edge != 0, b.east_edge != 0, b.south_edge != 0,
                               b.north_edge != 0};
    operators::PhysicalCorners c{e.west && e.south, e.east && e.south,
                                 e.east && e.north, e.west && e.north};
    st = operators::launch_a2b_ord4_global_cuda(hb, e, c, b.nk, ws.wk, ws.sh, a2b,
                                                ws.qx, ws.qy, ws.qxx, ws.qyy,
                                                /*replace*/ false, stream);
    if (st != FV3_GPU_SUCCESS) return st;
  }
  {  // S9
    Launch2D p = plan2d((ie + 1) - is + 1, (je + 1) - js + 1, b.nk);
    smag_combine_kernel<<<p.grid, p.block, 0, s>>>(b, lv, dt, as_const(ws.sh), smag_c);
  }
  return launch_status();
}

fv3_gpu_status launch_divdamp_high_order_finalize_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, View3<const fv_real> delpc,
    View3<const fv_real> divg_d, View3<fv_real> vort, View3<fv_real> ke,
    void *stream) noexcept {
  fv3_gpu_status st = validate_divergence(b, opt);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, /*nord*/ true, /*d2*/ true, /*d4*/ true, /*dcon*/ false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!delpc.data || !divg_d.data || !vort.data || !ke.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  Launch2D p = plan2d((b.ie + 1) - b.is + 1, (b.je + 1) - b.js + 1, b.nk);
  divdamp_high_order_finalize_kernel<<<p.grid, p.block, 0,
                                       static_cast<cudaStream_t>(stream)>>>(
      b, opt, lv, delpc, divg_d, vort, ke);
  return launch_status();
}

fv3_gpu_status launch_fill_corners_bgrid_cuda(const TransportBounds &b, int32_t dir,
                                              const DivergenceLevels &lv,
                                              View3<fv_real> q,
                                              void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!q.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  // Mirrors `fill_c` at sw_core.F90:1373-1376: not an error, just nothing to do.
  if (b.nested || !b.cubed_sphere) return FV3_GPU_SUCCESS;
  if (!(b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner))
    return FV3_GPU_SUCCESS;
  Launch2D p = plan2d(b.ng, b.ng, b.nk);
  fill_corners_bgrid_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, dir, lv, q);
  return launch_status();
}

fv3_gpu_status launch_fill_corners_dgrid_vector_cuda(const TransportBounds &b,
                                                     const DivergenceLevels &lv,
                                                     View3<fv_real> x,
                                                     View3<fv_real> y,
                                                     void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!x.data || !y.data || x.data == y.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.nested || !b.cubed_sphere) return FV3_GPU_SUCCESS;
  if (!(b.sw_corner || b.se_corner || b.ne_corner || b.nw_corner))
    return FV3_GPU_SUCCESS;

  auto s = static_cast<cudaStream_t>(stream);
  const fv_real my_sign = fv_real(-1);  // VECTOR = .true., fv_mp_mod.F90:1206
  Launch2D p = plan2d(b.ng, b.ng, b.nk);
  fill_corners_dgrid_x_kernel<<<p.grid, p.block, 0, s>>>(b, lv, my_sign, x,
                                                         as_const(y));
  fill_corners_dgrid_y_kernel<<<p.grid, p.block, 0, s>>>(b, lv, my_sign, as_const(x),
                                                         y);
  return launch_status();
}

fv3_gpu_status launch_dcon_vort_diff_cuda(const TransportBounds &b,
                                          const DivergenceLevels &lv,
                                          View3<const fv_real> vort, View3<fv_real> ub,
                                          View3<fv_real> vb, void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!vort.data || !ub.data || !vb.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  Launch2D p = plan2d((b.ie + 1) - b.is + 1, (b.je + 1) - b.js + 1, b.nk);
  dcon_vort_diff_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lv, vort, ub, vb);
  return launch_status();
}

fv3_gpu_status launch_dcon_scale_cuda(const TransportBounds &b,
                                      const DivergenceLevels &lv,
                                      const DivergenceGeometry &g,
                                      View3<const fv_real> u, View3<const fv_real> v,
                                      View3<const fv_real> ut, View3<const fv_real> vt,
                                      View3<fv_real> ub, View3<fv_real> vb,
                                      View3<fv_real> fx, View3<fv_real> fy,
                                      View3<fv_real> gx, View3<fv_real> gy,
                                      void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, false);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!u.data || !v.data || !ut.data || !vt.data || !ub.data || !vb.data || !fx.data ||
      !fy.data || !gx.data || !gy.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!g.rdx.data || !g.rdy.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  Launch2D p = plan2d((b.ie + 1) - b.is + 1, (b.je + 1) - b.js + 1, b.nk);
  dcon_scale_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lv, g, u, v, ut, vt, ub, vb, fx, fy, gx, gy);
  return launch_status();
}

fv3_gpu_status launch_dcon_heat_source_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, const DivergenceGeometry &g, View3<const fv_real> delp,
    View3<const fv_real> ub, View3<const fv_real> vb, View3<const fv_real> fx,
    View3<const fv_real> fy, View3<const fv_real> gx, View3<const fv_real> gy,
    View3<fv_real> heat_source, void *stream) noexcept {
  fv3_gpu_status st = validate_divergence(b, opt);
  if (st != FV3_GPU_SUCCESS) return st;
  st = validate_levels(lv, false, false, false, /*dcon*/ true);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!delp.data || !ub.data || !vb.data || !fx.data || !fy.data || !gx.data ||
      !gy.data || !heat_source.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!g.rsin2.data || !g.cosa_s.data) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  Launch2D p = plan2d(b.ie - b.is + 1, b.je - b.js + 1, b.nk);
  dcon_heat_source_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lv, g, delp, ub, vb, fx, fy, gx, gy, heat_source);
  return launch_status();
}

// Deliberately non-numerical: the whole point is to pin down the argument
// aliasing that sw_core.F90:1478 hides. No new numerics -- Milestone 3A.4's
// del6_vt_flux is the authoritative implementation and is reused verbatim.
//
// One behavioural note that is NOT in the Fortran: launch_del6_vt_flux_cuda
// pre-zeroes fx2/fy2 (here ut/vt) over the full ng-wide halo on every gated
// level, whereas Fortran only assigns the cells it computes. Every consumer in
// this block -- D2 at sw_core.F90:1482-1495 and V2 at :1523-1534 -- reads
// exactly the is..ie+1 x js..je and is..ie x js..je+1 rectangles that the
// nt == 0 final pass writes, so no consumer observes a zeroed cell. The
// difference is confined to halo cells and is recorded rather than hidden.
fv3_gpu_status launch_vorticity_damping_cuda(
    const TransportBounds &b, int32_t nord_cap, const LevelFilter &lf,
    View3<fv_real> wk, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> vort, View3<fv_real> ut,
    View3<fv_real> vt, void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  // nord_v(k) = min(2, nord) (dyn_core.F90:713), so 2 is the largest value FV3
  // can produce and ng - 1 == 2 is the largest the halo admits.
  if (nord_cap < 0 || nord_cap > static_cast<int32_t>(b.ng) - 1)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (lf.mode != 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (lf.gate.data == nullptr || lf.damp.data == nullptr || lf.ndif.data == nullptr)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!wk.data || !vort.data || !ut.data || !vt.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  return launch_del6_vt_flux_cuda(b, nord_cap, lf, wk, del6_u, del6_v, rarea, vort, ut,
                                  vt, stream);
}

fv3_gpu_status launch_dwind_damping_add_cuda(const TransportBounds &b,
                                             const LevelFilter &lf,
                                             View3<const fv_real> ut,
                                             View3<const fv_real> vt, View3<fv_real> u,
                                             View3<fv_real> v, void *stream) noexcept {
  fv3_gpu_status st = validate_divergence_bounds(b);
  if (st != FV3_GPU_SUCCESS) return st;
  if (!ut.data || !vt.data || !u.data || !v.data)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  Launch2D p = plan2d((b.ie + 1) - b.is + 1, (b.je + 1) - b.js + 1, b.nk);
  dwind_damping_add_kernel<<<p.grid, p.block, 0, static_cast<cudaStream_t>(stream)>>>(
      b, lf, ut, vt, u, v);
  return launch_status();
}

}  // namespace fv3::gpu::dgrid
