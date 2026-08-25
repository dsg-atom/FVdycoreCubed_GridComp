// ===========================================================================
// Milestone 3A.6 / WP-B.3 -- authoritative FV3 D-grid momentum transport.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45, model/sw_core.F90.
//
// The ordered operation table and the field/staggering contract live in
// dgrid/wp_b_momentum_cuda.cuh. Every kernel below carries the Fortran line
// range it transcribes.
//
// Structural differences from the Fortran, and why each is semantics-preserving:
//
//  (a) The Fortran keeps bl/br/b0/smt5 as automatic arrays scoped inside
//      xtp_u's `do j` loop (1-D) and as full 2-D arrays in ytp_v. Both are
//      materialized here as caller-supplied device workspaces, because the
//      flux loop reads index i-1 (resp. j-1) and a GPU thread cannot see
//      another thread's registers. Values are identical.
//
//  (b) xtp_u/ytp_v write bl/br in several passes (interior loop, then edge
//      overrides that overwrite some entries). Here each (i,j) evaluates the
//      one authoritative formula that survives the Fortran's overwrite order
//      directly. This was verified to cover the full b0 range exactly once --
//      see momentum_decomposition_supported() for the one decomposition family
//      where the Fortran itself leaves entries undefined.
//
//  (c) d_sw's vb/ub constructions are likewise a pure function of (i,j) once
//      the Fortran's write order is resolved, so each is a single kernel.
//
//  (d) Row 8 (vorticity precursor) is split into two launches so that the wk
//      pass observes the completed vt/ut, which the Fortran gets from loop
//      sequencing.
//
// No operation is fused across the ut/vt anti-dependence at row 8, and the
// ub/vb workspace reuse chain is enforced by launch order on one stream.
// ===========================================================================

#include "dgrid/wp_b_momentum_cuda.cuh"
#include "dgrid/wp_b_ppm_shared.cuh"

#include <cmath>
#include <cstdint>

namespace fv3::gpu::dgrid {
namespace {

// ---------------------------------------------------------------------------
// configuration gates
// ---------------------------------------------------------------------------

// The production first-run configuration (CLAUDE.md) is grid_type = 0,
// hydrostatic = false, on the cubed sphere with no nest. d_sw's momentum block,
// xtp_u and ytp_v all branch on `grid_type < 3`, `grid_type > 3` and `nested`,
// and those branches are genuinely different algorithms -- not ported here, so
// not silently accepted either.
bool momentum_topology_supported(const TransportBounds &b, const MomentumOptions &opt) {
  if (opt.grid_type != 0) return false;
  if (b.nested) return false;
  if (!b.cubed_sphere) return false;
  return true;
}

// xtp_u's interior reconstruction is clipped to is3 = max(3, is-1),
// ie3 = min(npx-3, ie+1), while b0/smt span is-1 .. ie+1 and the flux loop
// reads them. For a subdomain with is in {2,3} the clip leaves bl/br at
// i = is-1 .. 2 unwritten, yet still read -- the authoritative Fortran has no
// defined value there, so there is nothing to reproduce. Rather than invent
// one, those decompositions are rejected. Realistic GEOS cubed-sphere
// decompositions satisfy the condition (a tile edge either starts at 1 or is
// at least a full PPM stencil away from it).
//
// npx/npy >= 8 additionally keeps the west index set {0,1,2} disjoint from the
// east set {npx-2, npx-1, npx}, which the Fortran's sequential
// west-then-east overwrite would otherwise resolve by position.
bool momentum_decomposition_supported(const TransportBounds &b) {
  if (b.npx < 8 || b.npy < 8) return false;
  const bool i_lo = (b.is == 1) || (b.is >= 4);
  const bool i_hi = ((b.ie + 1) == b.npx) || ((b.ie + 1) <= b.npx - 3);
  const bool j_lo = (b.js == 1) || (b.js >= 4);
  const bool j_hi = ((b.je + 1) == b.npy) || ((b.je + 1) <= b.npy - 3);
  return i_lo && i_hi && j_lo && j_hi;
}

// xtp_u/ytp_v contain no iord<0 branch (unlike xppm/yppm, which add an al>=0
// clamp), so hord_mt = -6 and +6 are the same code path. abs() here is
// therefore faithful, not a convenience.
bool momentum_hord_supported(const MomentumOptions &opt) {
  const int32_t mord = opt.hord_mt >= 0 ? opt.hord_mt : -opt.hord_mt;
  return mord == 6;
}

fv3_gpu_status validate_momentum(const TransportBounds &b, const MomentumOptions &opt,
                                 bool needs_ppm) {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!momentum_topology_supported(b, opt)) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (!momentum_decomposition_supported(b)) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (needs_ppm && !momentum_hord_supported(opt)) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  return FV3_GPU_SUCCESS;
}

// ---------------------------------------------------------------------------
// row 5: xtp_u (sw_core.F90:1950-2317, iord = 6)
// ---------------------------------------------------------------------------

// The unclipped interior interface value, sw_core.F90:1989.
__device__ __forceinline__ fv_real xtp_al(View3<const fv_real> u, int64_t i, int64_t j,
                                          int64_t k) {
  return ppm_p1() * (u(i - 1, j, k) + u(i, j, k)) +
         ppm_p2() * (u(i - 2, j, k) + u(i + 1, j, k));
}

// sw_core.F90:2009-2010 -- the dx-weighted two-sided extrapolation straddling
// the west seam. Note this uses gridstruct%dx, NOT the dxa that xppm's
// analogous formula (tp_core.F90:378-381) uses; they are different metrics and
// must not be interchanged.
__device__ __forceinline__ fv_real xtp_seam_xt(View3<const fv_real> u,
                                               View2<const fv_real> dx, int64_t i0,
                                               int64_t i1, int64_t j, int64_t k) {
  // i0 = the cell just inside one panel, i1 = just inside the other.
  const fv_real a =
      ((fv_real(2) * dx(i0, j) + dx(i0 - 1, j)) * u(i0, j, k) - dx(i0, j) * u(i0 - 1, j, k)) /
      (dx(i0, j) + dx(i0 - 1, j));
  const fv_real c =
      ((fv_real(2) * dx(i1, j) + dx(i1 + 1, j)) * u(i1, j, k) - dx(i1, j) * u(i1 + 1, j, k)) /
      (dx(i1, j) + dx(i1 + 1, j));
  return fv_real(0.5) * (a + c);
}

// Resolves the Fortran's interior-then-west-then-east write order into the one
// surviving formula for (i,j). East is tested first because the Fortran's east
// block runs last and would overwrite the west block on overlap.
__device__ void xtp_u_bl_br(View3<const fv_real> u, View2<const fv_real> dx, int64_t i,
                            int64_t j, int64_t k, const TransportBounds &b, fv_real &bl,
                            fv_real &br) {
  const int64_t npx = b.npx, npy = b.npy;
  const bool seam = b.cubed_sphere && !b.nested;
  // sw_core.F90:2002 / 2021. j can only equal 1 when js == 1 and npy when
  // je+1 == npy, so the raw value test in the Fortran and the panel-ownership
  // test are equivalent; the value test is kept to match the source.
  const bool edge_row = (j == 1) || (j == npy);

  if (seam && b.east_edge && i >= npx - 2) {
    // sw_core.F90:2016-2034
    const fv_real xt1 =
        ppm_c1() * u(npx - 3, j, k) + ppm_c2() * u(npx - 2, j, k) + ppm_c3() * u(npx - 1, j, k);
    if (i == npx - 2) {
      bl = xtp_al(u, npx - 2, j, k) - u(npx - 2, j, k);   // :2017
      br = xt1 - u(npx - 2, j, k);                        // :2019
      return;
    }
    if (edge_row) { bl = fv_real(0); br = fv_real(0); return; }  // :2022-2025
    const fv_real xt2 = xtp_seam_xt(u, dx, npx - 1, npx, j, k);  // :2027-2028
    if (i == npx - 1) {
      bl = xt1 - u(npx - 1, j, k);                        // :2020
      br = xt2 - u(npx - 1, j, k);                        // :2029
    } else {                                              // i == npx
      bl = xt2 - u(npx, j, k);                            // :2030
      br = ppm_c3() * u(npx, j, k) + ppm_c2() * u(npx + 1, j, k) +
           ppm_c1() * u(npx + 2, j, k) - u(npx, j, k);    // :2031
    }
    return;
  }

  if (seam && b.west_edge && i <= 2) {
    // sw_core.F90:1997-2015
    const fv_real xt1 =
        ppm_c3() * u(1, j, k) + ppm_c2() * u(2, j, k) + ppm_c1() * u(3, j, k);  // :1998
    if (i == 2) {
      bl = xt1 - u(2, j, k);                              // :2000
      br = xtp_al(u, 3, j, k) - u(2, j, k);               // :2001 (al(3))
      return;
    }
    if (edge_row) { bl = fv_real(0); br = fv_real(0); return; }  // :2003-2006
    const fv_real xt2 = xtp_seam_xt(u, dx, 0, 1, j, k);   // :2009-2010
    if (i == 0) {
      bl = ppm_c1() * u(-2, j, k) + ppm_c2() * u(-1, j, k) + ppm_c3() * u(0, j, k) -
           u(0, j, k);                                    // :2008
      br = xt2 - u(0, j, k);                              // :2011
    } else {                                              // i == 1
      bl = xt2 - u(1, j, k);                              // :2012
      br = xt1 - u(1, j, k);                              // :1999
    }
    return;
  }

  bl = xtp_al(u, i, j, k) - u(i, j, k);                   // :1992
  br = xtp_al(u, i + 1, j, k) - u(i, j, k);               // :1993
}

// sw_core.F90:2037-2038 + 2140-2156. The order-6 smoothness flag and the WMP
// seam degradation are bit-for-bit the same test xppm/yppm use, so
// smt5_from/seam_cell_x are reused unchanged from wp_b_ppm_shared.cuh.
__global__ void xtp_u_profile_kernel(TransportBounds b, View3<const fv_real> u,
                                     View2<const fv_real> dx, View3<fv_real> bl,
                                     View3<fv_real> br, View3<fv_real> b0,
                                     View3<int32_t> smt) {
  const int64_t i = (b.is - 1) + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;

  fv_real l, r;
  xtp_u_bl_br(u, dx, i, j, k, b, l, r);
  const fv_real z = l + r;
  bl(i, j, k) = l;
  br(i, j, k) = r;
  b0(i, j, k) = z;
  smt(i, j, k) = smt5_from(l, r, z, seam_cell_x(i, b)) ? 1 : 0;
}

// sw_core.F90:2159-2171. cfl uses rdx at the UPWIND cell (i-1 for c>0, i for
// c<=0) -- not a single cell-centered Courant number, which is the main reason
// xppm_face_flux cannot be reused here.
__global__ void xtp_u_flux_kernel(TransportBounds b, View3<const fv_real> c,
                                  View3<const fv_real> u, View2<const fv_real> rdx,
                                  View3<const fv_real> bl, View3<const fv_real> br,
                                  View3<const fv_real> b0, View3<const int32_t> smt,
                                  View3<fv_real> flux) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;

  const fv_real cc = c(i, j, k);
  fv_real fx0, f;
  if (cc > fv_real(0)) {
    const fv_real cfl = cc * rdx(i - 1, j);
    fx0 = (fv_real(1) - cfl) * (br(i - 1, j, k) - cfl * b0(i - 1, j, k));
    f = u(i - 1, j, k);
  } else {
    const fv_real cfl = cc * rdx(i, j);
    fx0 = (fv_real(1) + cfl) * (bl(i, j, k) + cfl * b0(i, j, k));
    f = u(i, j, k);
  }
  if (smt(i - 1, j, k) || smt(i, j, k)) f = f + fx0;
  flux(i, j, k) = f;
}

// ---------------------------------------------------------------------------
// row 2: ytp_v (sw_core.F90:2320-2792, jord = 6)
// ---------------------------------------------------------------------------
// Verified by explicit inspection to be the exact (i <-> j, u -> v, dx -> dy)
// transpose of xtp_u for the bl/br/b0/smt construction and the flux loop, with
// three differences that are structural only:
//   * xtp_u's locals are 1-D inside a j loop, ytp_v's are 2-D;
//   * the order-6 flag is named smt5 in xtp_u and smt6 in ytp_v;
//   * ytp_v evaluates the dy-weighted xt for the columns it then zeroes
//     (i == 1 / i == npx), whereas xtp_u skips the dx-weighted xt on the rows
//     it zeroes. The evaluated value is discarded either way.
// It is nonetheless written out rather than obtained by transposing xtp_u,
// because the two read genuinely different staggered fields and metrics.

__device__ __forceinline__ fv_real ytp_al(View3<const fv_real> v, int64_t i, int64_t j,
                                          int64_t k) {
  return ppm_p1() * (v(i, j - 1, k) + v(i, j, k)) +
         ppm_p2() * (v(i, j - 2, k) + v(i, j + 1, k));
}

// sw_core.F90:2370-2371 -- dy-weighted, using gridstruct%dy (not dya).
__device__ __forceinline__ fv_real ytp_seam_xt(View3<const fv_real> v,
                                               View2<const fv_real> dy, int64_t i,
                                               int64_t j0, int64_t j1, int64_t k) {
  const fv_real a =
      ((fv_real(2) * dy(i, j0) + dy(i, j0 - 1)) * v(i, j0, k) - dy(i, j0) * v(i, j0 - 1, k)) /
      (dy(i, j0) + dy(i, j0 - 1));
  const fv_real c =
      ((fv_real(2) * dy(i, j1) + dy(i, j1 + 1)) * v(i, j1, k) - dy(i, j1) * v(i, j1 + 1, k)) /
      (dy(i, j1) + dy(i, j1 + 1));
  return fv_real(0.5) * (a + c);
}

__device__ void ytp_v_bl_br(View3<const fv_real> v, View2<const fv_real> dy, int64_t i,
                            int64_t j, int64_t k, const TransportBounds &b, fv_real &bl,
                            fv_real &br) {
  const int64_t npx = b.npx, npy = b.npy;
  const bool seam = b.cubed_sphere && !b.nested;
  // sw_core.F90:2379/2385 and 2406/2412: the zeroing is keyed on the CROSS
  // index here, mirroring xtp_u's `j==1 .or. j==npy`.
  const bool edge_col = (i == 1) || (i == npx);

  if (seam && b.north_edge && j >= npy - 2) {
    // sw_core.F90:2394-2417
    const fv_real xt1 = ppm_c1() * v(i, npy - 3, k) + ppm_c2() * v(i, npy - 2, k) +
                        ppm_c3() * v(i, npy - 1, k);          // :2397
    if (j == npy - 2) {
      bl = ytp_al(v, i, npy - 2, k) - v(i, npy - 2, k);       // :2396
      br = xt1 - v(i, npy - 2, k);                            // :2398
      return;
    }
    if (edge_col) { bl = fv_real(0); br = fv_real(0); return; }  // :2407-2416
    const fv_real xt2 = ytp_seam_xt(v, dy, i, npy - 1, npy, k);  // :2400-2401
    if (j == npy - 1) {
      bl = xt1 - v(i, npy - 1, k);                            // :2399
      br = xt2 - v(i, npy - 1, k);                            // :2402
    } else {                                                  // j == npy
      bl = xt2 - v(i, npy, k);                                // :2403
      br = ppm_c3() * v(i, npy, k) + ppm_c2() * v(i, npy + 1, k) +
           ppm_c1() * v(i, npy + 2, k) - v(i, npy, k);        // :2404
    }
    return;
  }

  if (seam && b.south_edge && j <= 2) {
    // sw_core.F90:2367-2390
    const fv_real xt1 =
        ppm_c3() * v(i, 1, k) + ppm_c2() * v(i, 2, k) + ppm_c1() * v(i, 3, k);  // :2374
    if (j == 2) {
      bl = xt1 - v(i, 2, k);                                  // :2376
      br = ytp_al(v, i, 3, k) - v(i, 2, k);                   // :2377 (al(i,3))
      return;
    }
    if (edge_col) { bl = fv_real(0); br = fv_real(0); return; }  // :2380-2389
    const fv_real xt2 = ytp_seam_xt(v, dy, i, 0, 1, k);       // :2370-2371
    if (j == 0) {
      bl = ppm_c1() * v(i, -2, k) + ppm_c2() * v(i, -1, k) + ppm_c3() * v(i, 0, k) -
           v(i, 0, k);                                        // :2369
      br = xt2 - v(i, 0, k);                                  // :2372
    } else {                                                  // j == 1
      bl = xt2 - v(i, 1, k);                                  // :2373
      br = xt1 - v(i, 1, k);                                  // :2375
    }
    return;
  }

  bl = ytp_al(v, i, j, k) - v(i, j, k);                       // :2361
  br = ytp_al(v, i, j + 1, k) - v(i, j, k);                   // :2362
}

__global__ void ytp_v_profile_kernel(TransportBounds b, View3<const fv_real> v,
                                     View2<const fv_real> dy, View3<fv_real> bl,
                                     View3<fv_real> br, View3<fv_real> b0,
                                     View3<int32_t> smt) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = (b.js - 1) + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;

  fv_real l, r;
  ytp_v_bl_br(v, dy, i, j, k, b, l, r);
  const fv_real z = l + r;
  bl(i, j, k) = l;
  br(i, j, k) = r;
  b0(i, j, k) = z;
  smt(i, j, k) = smt5_from(l, r, z, seam_cell_y(j, b)) ? 1 : 0;  // :2562 + :2570-2577
}

// sw_core.F90:2582-2596.
__global__ void ytp_v_flux_kernel(TransportBounds b, View3<const fv_real> c,
                                  View3<const fv_real> v, View2<const fv_real> rdy,
                                  View3<const fv_real> bl, View3<const fv_real> br,
                                  View3<const fv_real> b0, View3<const int32_t> smt,
                                  View3<fv_real> flux) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;

  const fv_real cc = c(i, j, k);
  fv_real fx0, f;
  if (cc > fv_real(0)) {
    const fv_real cfl = cc * rdy(i, j - 1);
    fx0 = (fv_real(1) - cfl) * (br(i, j - 1, k) - cfl * b0(i, j - 1, k));
    f = v(i, j - 1, k);
  } else {
    const fv_real cfl = cc * rdy(i, j);
    fx0 = (fv_real(1) + cfl) * (bl(i, j, k) + cfl * b0(i, j, k));
    f = v(i, j, k);
  }
  if (smt(i, j - 1, k) || smt(i, j, k)) f = f + fx0;
  flux(i, j, k) = f;
}

// ---------------------------------------------------------------------------
// rows 1 and 4: B-grid contravariant wind constructions
// ---------------------------------------------------------------------------

// sw_core.F90:1086-1111. Resolution of the Fortran's write order: the j==1 and
// j==npy rows are written by their own loops over the FULL i range and the
// interior j loop is clipped to js2..je1 = max(2,js)..min(npy-1,je+1), so a
// tile-edge ROW wins over a tile-edge COLUMN. That is the source of the
// "corner values are incorrect" comment and is preserved exactly.
__global__ void bgrid_vb_kernel(TransportBounds b, fv_real dt5, fv_real dt4,
                                View3<const fv_real> uc, View3<const fv_real> vc,
                                View3<const fv_real> vt, View2<const fv_real> cosa,
                                View2<const fv_real> rsina, View3<fv_real> vb) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  const int64_t npx = b.npx, npy = b.npy;

  if (j == 1) {                                        // :1086-1090
    vb(i, j, k) = dt5 * (vt(i - 1, 1, k) + vt(i, 1, k));
    return;
  }
  if (j == npy) {                                      // :1107-1111
    vb(i, j, k) = dt5 * (vt(i - 1, npy, k) + vt(i, npy, k));
    return;
  }
  if (i == 1) {                                        // :1097-1100
    vb(i, j, k) =
        dt4 * (-vt(-1, j, k) + fv_real(3) * (vt(0, j, k) + vt(1, j, k)) - vt(2, j, k));
    return;
  }
  if (i == npx) {                                      // :1101-1104
    vb(i, j, k) = dt4 * (-vt(npx - 2, j, k) +
                         fv_real(3) * (vt(npx - 1, j, k) + vt(npx, j, k)) -
                         vt(npx + 1, j, k));
    return;
  }
  vb(i, j, k) = dt5 *                                  // :1094
                (vc(i - 1, j, k) + vc(i, j, k) -
                 (uc(i, j - 1, k) + uc(i, j, k)) * cosa(i, j)) *
                rsina(i, j);
}

// sw_core.F90:1145-1168. NOT the transpose of bgrid_vb_kernel: here the i==1 /
// i==npx COLUMNS are written over the full j range and the interior i loop is
// clipped to is2..ie1, so a tile-edge COLUMN wins over a tile-edge ROW -- the
// opposite precedence. The 4-point one-sided formula also sits on the j==1 /
// j==npy rows (reading ut at j-2..j+1) rather than on the columns.
__global__ void bgrid_ub_kernel(TransportBounds b, fv_real dt5, fv_real dt4,
                                View3<const fv_real> uc, View3<const fv_real> vc,
                                View3<const fv_real> ut, View2<const fv_real> cosa,
                                View2<const fv_real> rsina, View3<fv_real> ub) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  const int64_t npx = b.npx, npy = b.npy;

  if (i == 1) {                                        // :1145-1149
    ub(i, j, k) = dt5 * (ut(1, j - 1, k) + ut(1, j, k));
    return;
  }
  if (i == npx) {                                      // :1164-1168
    ub(i, j, k) = dt5 * (ut(npx, j - 1, k) + ut(npx, j, k));
    return;
  }
  if (j == 1 || j == npy) {                            // :1152-1156
    ub(i, j, k) = dt4 * (-ut(i, j - 2, k) +
                         fv_real(3) * (ut(i, j - 1, k) + ut(i, j, k)) - ut(i, j + 1, k));
    return;
  }
  ub(i, j, k) = dt5 *                                  // :1159
                (uc(i, j - 1, k) + uc(i, j, k) -
                 (vc(i - 1, j, k) + vc(i, j, k)) * cosa(i, j)) *
                rsina(i, j);
}

// ---------------------------------------------------------------------------
// rows 3, 6, 7: kinetic energy
// ---------------------------------------------------------------------------

__global__ void ke_product_kernel(TransportBounds b, View3<const fv_real> vb,
                                  View3<const fv_real> ub, View3<fv_real> ke) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  ke(i, j, k) = vb(i, j, k) * ub(i, j, k);             // :1127
}

__global__ void ke_average_kernel(TransportBounds b, View3<const fv_real> ub,
                                  View3<const fv_real> vb, View3<fv_real> ke) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > b.ie + 1 || j > b.je + 1 || k >= b.nk) return;
  ke(i, j, k) = fv_real(0.5) * (ke(i, j, k) + ub(i, j, k) * vb(i, j, k));  // :1184
}

// sw_core.F90:1191-1216. Four independent single-point overrides. One thread
// per corner per level; the corner index sets are disjoint (npx, npy >= 8), so
// there is no write conflict between the four branches.
__global__ void ke_corner_fix_kernel(TransportBounds b, fv_real dt6,
                                     View3<const fv_real> ut, View3<const fv_real> vt,
                                     View3<const fv_real> u, View3<const fv_real> v,
                                     View3<fv_real> ke) {
  const int64_t corner = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = blockIdx.z;
  if (corner > 3 || k >= b.nk) return;
  const int64_t npx = b.npx, npy = b.npy;

  if (corner == 0 && b.sw_corner) {                    // :1193-1197
    ke(1, 1, k) = dt6 * ((ut(1, 1, k) + ut(1, 0, k)) * u(1, 1, k) +
                         (vt(1, 1, k) + vt(0, 1, k)) * v(1, 1, k) +
                         (ut(1, 1, k) + vt(1, 1, k)) * u(0, 1, k));
  } else if (corner == 1 && b.se_corner) {             // :1198-1203
    const int64_t i = npx;
    ke(i, 1, k) = dt6 * ((ut(i, 1, k) + ut(i, 0, k)) * u(i - 1, 1, k) +
                         (vt(i, 1, k) + vt(i - 1, 1, k)) * v(i, 1, k) +
                         (ut(i, 1, k) - vt(i - 1, 1, k)) * u(i, 1, k));
  } else if (corner == 2 && b.ne_corner) {             // :1204-1209
    const int64_t i = npx, j = npy;
    ke(i, j, k) = dt6 * ((ut(i, j, k) + ut(i, j - 1, k)) * u(i - 1, j, k) +
                         (vt(i, j, k) + vt(i - 1, j, k)) * v(i, j - 1, k) +
                         (ut(i, j - 1, k) + vt(i - 1, j, k)) * u(i, j, k));
  } else if (corner == 3 && b.nw_corner) {             // :1210-1215
    const int64_t j = npy;
    ke(1, j, k) = dt6 * ((ut(1, j, k) + ut(1, j - 1, k)) * u(1, j, k) +
                         (vt(1, j, k) + vt(0, j, k)) * v(1, j - 1, k) +
                         (ut(1, j - 1, k) - vt(1, j, k)) * u(0, j, k));
  }
}

// ---------------------------------------------------------------------------
// row 8: vorticity precursor (sw_core.F90:1219-1235)
// ---------------------------------------------------------------------------

__global__ void vorticity_uv_dxdy_kernel(TransportBounds b, View3<const fv_real> u,
                                         View3<const fv_real> v, View2<const fv_real> dx,
                                         View2<const fv_real> dy, View3<fv_real> ut,
                                         View3<fv_real> vt) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (k >= b.nk) return;
  // Two different staggered index spaces in one launch; each guarded
  // separately so neither writes outside its Fortran loop bounds.
  if (i <= ied && j <= jed + 1) vt(i, j, k) = u(i, j, k) * dx(i, j);       // :1219-1223
  if (i <= ied + 1 && j <= jed) ut(i, j, k) = v(i, j, k) * dy(i, j);       // :1224-1228
}

__global__ void vorticity_wk_kernel(TransportBounds b, View3<const fv_real> ut,
                                    View3<const fv_real> vt, View2<const fv_real> rarea,
                                    View3<fv_real> wk) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ied || j > jed || k >= b.nk) return;
  wk(i, j, k) = rarea(i, j) * (vt(i, j, k) - vt(i, j + 1, k) - ut(i, j, k) +
                               ut(i + 1, j, k));                           // :1233
}

// ---------------------------------------------------------------------------
// row 9: absolute vorticity (sw_core.F90:1237-1451)
// ---------------------------------------------------------------------------

__global__ void vorticity_absolute_kernel(TransportBounds b, bool use_z_rat,
                                          View3<const fv_real> wk,
                                          View2<const fv_real> f0,
                                          View3<const fv_real> z_rat, View3<fv_real> vort) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (i > ied || j > jed || k >= b.nk) return;
  vort(i, j, k) = use_z_rat ? wk(i, j, k) + f0(i, j) * z_rat(i, j, k)
                            : wk(i, j, k) + f0(i, j);
}

// ---------------------------------------------------------------------------
// row 10: D-wind update (sw_core.F90:1466-1478)
// ---------------------------------------------------------------------------
// u and v are updated from DISJOINT index spaces and each reads only ut/vt/ke/
// fx/fy, never the other's output, so one kernel carries both without
// obscuring the Fortran order. ke is read at (i,j) and (i+1,j) / (i,j+1) but
// never written here, so there is no anti-dependence to serialize.
__global__ void dwind_update_kernel(TransportBounds b, View3<const fv_real> ut,
                                    View3<const fv_real> vt, View3<const fv_real> ke,
                                    View3<const fv_real> fx, View3<const fv_real> fy,
                                    View3<fv_real> u, View3<fv_real> v) {
  const int64_t i = b.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = b.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = blockIdx.z;
  if (k >= b.nk) return;
  if (i <= b.ie && j <= b.je + 1)                                          // :1466-1470
    u(i, j, k) = vt(i, j, k) + ke(i, j, k) - ke(i + 1, j, k) + fy(i, j, k);
  if (i <= b.ie + 1 && j <= b.je)                                          // :1473-1477
    v(i, j, k) = ut(i, j, k) + ke(i, j, k) - ke(i, j + 1, k) - fx(i, j, k);
}

} // namespace

// =========================== launchers ==================================

fv3_gpu_status launch_bgrid_vb_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                    fv_real dt, View3<const fv_real> uc,
                                    View3<const fv_real> vc, View3<const fv_real> ut,
                                    View3<const fv_real> vt, View2<const fv_real> cosa,
                                    View2<const fv_real> rsina, View3<fv_real> vb,
                                    void *stream) noexcept {
  (void)ut;  // d_sw's vb path reads vt only; ut is taken so rows 1 and 4 share
             // one call signature and a caller cannot pass them swapped.
  const fv3_gpu_status v = validate_momentum(b, opt, false);
  if (v != FV3_GPU_SUCCESS) return v;
  auto *s = static_cast<cudaStream_t>(stream);
  const Launch2D p = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  bgrid_vb_kernel<<<p.grid, p.block, 0, s>>>(b, fv_real(0.5) * dt, fv_real(0.25) * dt, uc,
                                             vc, vt, cosa, rsina, vb);
  return launch_status();
}

fv3_gpu_status launch_bgrid_ub_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                    fv_real dt, View3<const fv_real> uc,
                                    View3<const fv_real> vc, View3<const fv_real> ut,
                                    View3<const fv_real> vt, View2<const fv_real> cosa,
                                    View2<const fv_real> rsina, View3<fv_real> ub,
                                    void *stream) noexcept {
  (void)vt;  // symmetric: d_sw's ub path reads ut only.
  const fv3_gpu_status v = validate_momentum(b, opt, false);
  if (v != FV3_GPU_SUCCESS) return v;
  auto *s = static_cast<cudaStream_t>(stream);
  const Launch2D p = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  bgrid_ub_kernel<<<p.grid, p.block, 0, s>>>(b, fv_real(0.5) * dt, fv_real(0.25) * dt, uc,
                                             vc, ut, cosa, rsina, ub);
  return launch_status();
}

fv3_gpu_status launch_xtp_u_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                 View3<const fv_real> c, View3<const fv_real> u,
                                 View2<const fv_real> dx, View2<const fv_real> rdx,
                                 const PpmMomentumWorkspace &ws, View3<fv_real> flux,
                                 void *stream) noexcept {
  const fv3_gpu_status v = validate_momentum(b, opt, true);
  if (v != FV3_GPU_SUCCESS) return v;
  auto *s = static_cast<cudaStream_t>(stream);

  // Profile spans is-1 .. ie+1 so the flux pass can read i-1.
  const Launch2D pp = plan2d(b.ie + 3 - b.is, b.je + 2 - b.js, b.nk);
  xtp_u_profile_kernel<<<pp.grid, pp.block, 0, s>>>(b, u, dx, ws.bl, ws.br, ws.b0, ws.smt);
  fv3_gpu_status st = launch_status();
  if (st != FV3_GPU_SUCCESS) return st;

  const Launch2D pf = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  xtp_u_flux_kernel<<<pf.grid, pf.block, 0, s>>>(
      b, c, u, rdx, as_const(ws.bl), as_const(ws.br), as_const(ws.b0),
      View3<const int32_t>{ws.smt.data, ws.smt.lo_i, ws.smt.lo_j, ws.smt.lo_k, ws.smt.si,
                           ws.smt.sj, ws.smt.sk},
      flux);
  return launch_status();
}

fv3_gpu_status launch_ytp_v_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                 View3<const fv_real> c, View3<const fv_real> v,
                                 View2<const fv_real> dy, View2<const fv_real> rdy,
                                 const PpmMomentumWorkspace &ws, View3<fv_real> flux,
                                 void *stream) noexcept {
  const fv3_gpu_status vs = validate_momentum(b, opt, true);
  if (vs != FV3_GPU_SUCCESS) return vs;
  auto *s = static_cast<cudaStream_t>(stream);

  const Launch2D pp = plan2d(b.ie + 2 - b.is, b.je + 3 - b.js, b.nk);
  ytp_v_profile_kernel<<<pp.grid, pp.block, 0, s>>>(b, v, dy, ws.bl, ws.br, ws.b0, ws.smt);
  fv3_gpu_status st = launch_status();
  if (st != FV3_GPU_SUCCESS) return st;

  const Launch2D pf = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  ytp_v_flux_kernel<<<pf.grid, pf.block, 0, s>>>(
      b, c, v, rdy, as_const(ws.bl), as_const(ws.br), as_const(ws.b0),
      View3<const int32_t>{ws.smt.data, ws.smt.lo_i, ws.smt.lo_j, ws.smt.lo_k, ws.smt.si,
                           ws.smt.sj, ws.smt.sk},
      flux);
  return launch_status();
}

fv3_gpu_status launch_ke_product_cuda(const TransportBounds &b, View3<const fv_real> vb,
                                      View3<const fv_real> ub, View3<fv_real> ke,
                                      void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const Launch2D p = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  ke_product_kernel<<<p.grid, p.block, 0, s>>>(b, vb, ub, ke);
  return launch_status();
}

fv3_gpu_status launch_ke_average_cuda(const TransportBounds &b, View3<const fv_real> ub,
                                      View3<const fv_real> vb, View3<fv_real> ke,
                                      void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const Launch2D p = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  ke_average_kernel<<<p.grid, p.block, 0, s>>>(b, ub, vb, ke);
  return launch_status();
}

fv3_gpu_status launch_ke_corner_fix_cuda(const TransportBounds &b, fv_real dt,
                                         View3<const fv_real> ut, View3<const fv_real> vt,
                                         View3<const fv_real> u, View3<const fv_real> v,
                                         View3<fv_real> ke, void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  // sw_core.F90:1191 -- the whole block is `.not. nested`.
  if (b.nested) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  if (b.npx < 8 || b.npy < 8) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const dim3 block(4, 1, 1);
  const dim3 grid(1, 1, static_cast<unsigned>(b.nk));
  ke_corner_fix_kernel<<<grid, block, 0, s>>>(b, dt / fv_real(6), ut, vt, u, v, ke);
  return launch_status();
}

fv3_gpu_status launch_vorticity_precursor_cuda(const TransportBounds &b,
                                               View3<const fv_real> u,
                                               View3<const fv_real> v,
                                               View2<const fv_real> dx,
                                               View2<const fv_real> dy,
                                               View2<const fv_real> rarea,
                                               View3<fv_real> ut, View3<fv_real> vt,
                                               View3<fv_real> wk, void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const int64_t ni = (b.ie + b.ng) - (b.is - b.ng) + 2;   // covers isd..ied+1
  const int64_t nj = (b.je + b.ng) - (b.js - b.ng) + 2;   // covers jsd..jed+1
  const Launch2D p = plan2d(ni, nj, b.nk);
  vorticity_uv_dxdy_kernel<<<p.grid, p.block, 0, s>>>(b, u, v, dx, dy, ut, vt);
  fv3_gpu_status st = launch_status();
  if (st != FV3_GPU_SUCCESS) return st;
  // Separate launch: wk must see the completed vt/ut, which the Fortran gets
  // from loop sequencing.
  vorticity_wk_kernel<<<p.grid, p.block, 0, s>>>(b, as_const(ut), as_const(vt), rarea, wk);
  return launch_status();
}

fv3_gpu_status launch_vorticity_absolute_cuda(const TransportBounds &b,
                                              const MomentumOptions &opt,
                                              View3<const fv_real> wk,
                                              View2<const fv_real> f0,
                                              View3<const fv_real> z_rat,
                                              View3<fv_real> vort, void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  const bool use_z_rat = opt.do_f3d != 0;
  if (use_z_rat && z_rat.data == nullptr) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const int64_t ni = (b.ie + b.ng) - (b.is - b.ng) + 1;
  const int64_t nj = (b.je + b.ng) - (b.js - b.ng) + 1;
  const Launch2D p = plan2d(ni, nj, b.nk);
  vorticity_absolute_kernel<<<p.grid, p.block, 0, s>>>(b, use_z_rat, wk, f0, z_rat, vort);
  return launch_status();
}

fv3_gpu_status launch_dwind_update_cuda(const TransportBounds &b, View3<const fv_real> ut,
                                        View3<const fv_real> vt, View3<const fv_real> ke,
                                        View3<const fv_real> fx, View3<const fv_real> fy,
                                        View3<fv_real> u, View3<fv_real> v,
                                        void *stream) noexcept {
  if (b.nk <= 0) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  auto *s = static_cast<cudaStream_t>(stream);
  const Launch2D p = plan2d(b.ie + 2 - b.is, b.je + 2 - b.js, b.nk);
  dwind_update_kernel<<<p.grid, p.block, 0, s>>>(b, ut, vt, ke, fx, fy, u, v);
  return launch_status();
}

} // namespace fv3::gpu::dgrid
