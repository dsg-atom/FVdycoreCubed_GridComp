// Authoritative source: model/sw_core.F90:d2a2c_vect (lines 2800-3139)
// (GFDL_atmos_cubed_sphere tag geos/v3.0.0, ca30e26).
// D-grid to A-grid to C-grid wind interpolation for the cubed sphere.
//
// Target configuration: grid_type = 0, non-nested, dord4 = .true., ng = 3.
// The grid_type >= 3 (Cartesian) paths are transcribed where they are free
// (npt = -2 at :2848-2852, the unclamped ifirst/ilast at :2984-2986, and the
// vt = vc branch at :3129-3136). The nested branch (:2858-2887) is NOT
// implemented and is REJECTED by the launcher.
//
// Every kernel below states the Fortran loop range it implements and is
// launched with a grid that actually covers that range.
//
// ---------------------------------------------------------------------------
// Repairs in this file (all verified against the authoritative Fortran):
//
// D1  d2a_interpolate_interior hardcoded `isd = is-1; ied = ie+1; ...`. The
//     authoritative bounds are bd%isd/ied/jsd/jed = is-ng/ie+ng/js-ng/je+ng
//     (:2827-2830), so the utmp/vtmp workspace was left uninitialized over
//     most of the halo and then read by the A->C stencils. D2A2CArgs now
//     carries `ng` and the real bounds are used everywhere.
//     Numerical semantics changed: NO.
//
// D2  The interior/edge partition was a single per-point choice shared by utmp
//     and vtmp. The Fortran uses two ASYMMETRIC 4th-order boxes (:2893-2897
//     and :2898-2902) followed by four 2nd-order bands that OVERWRITE both
//     arrays (:2907-2943). Restructured into those phases.
//     Numerical semantics changed: NO.
//
// D3  `utmp(:,:) = big_number ; vtmp(:,:) = big_number` (:2854-2856) was not
//     implemented. Added as a prefill kernel over (isd:ied, jsd:jed). This is
//     load bearing, not cosmetic: for an interior tile the Fortran ua/va loop
//     (:2946-2951) reaches j = js-2 / je+2 and i = is-2 / ie+2, which the
//     4th-order boxes never fill, so the authoritative result there IS
//     big_number-derived and must be reproduced exactly.
//     Numerical semantics changed: NO.
//
// D4  All four corner-reflection blocks were missing (:2960-2979, :3000-3015,
//     :3053-3072, :3073-3088). All four are implemented, in Fortran order.
//     Numerical semantics changed: NO.
//
// D5  compute_a_grid_winds was based at (is-1, js-1) with a grid of
//     (ie-is+3, je-js+3), so its own `i >= is-1-id` guard was unreachable and
//     the outer ring of ua/va was never written -- while a2c_uc_west_edge
//     reads ua(-1:2,j) and a2c_vc_edge_rows reads va(i,-1:2). Base and extent
//     now match `do j=js-1-id,je+1+id ; do i=is-1-id,ie+1+id` (:2946-2947).
//     Numerical semantics changed: NO.
//
// D6  The never-defined declaration `launch_d2a2c_vect_v3(const
//     ExecutionContext&, fv3_gpu_error*)` was deleted from the header.
//     Numerical semantics changed: NO.
//
// D7  (found while repairing) a2c_vc_edge_rows implemented the special-row
//     bodies of :3092-3128 but not the OUTER loop bound `do j=js-1,je+2`
//     (:3092). For any tile that does not touch a face boundary the six rows
//     j = 0,1,2,npy-1,npy,npy+1 lie outside [jsd,jed], so the kernel both
//     wrote vc/vt out of bounds and applied edge formulas to rows the Fortran
//     never touches. One guard added.
//     Numerical semantics changed: NO.
//
// D8  (found while repairing) a2c_interpolate_vc_interior always computed
//     vt = (vc - u*cosa_v)*rsin_v, but the grid_type >= 3 branch at
//     :3129-3136 sets vt = vc. Branch added.
//     Numerical semantics changed: NO.
//
// D9  (found while repairing) a2c_uc_west_edge / a2c_uc_east_edge computed the
//     edge_interpolate4 result into a local and never stored it, but the
//     Fortran assigns it: `ut(1,j) = edge_interpolate4(...)` (:3020) and
//     `ut(npx,j) = edge_interpolate4(...)` (:3036). Nothing else writes
//     ut(1,j) or ut(npx,j) -- the interior loop starts at ifirst = max(3,is-1)
//     and ends at ilast = min(npx-2,ie+2) -- so those two columns of ut were
//     left stale/uninitialized for c_sw to consume. Stores added.
//     Numerical semantics changed: NO.
// ---------------------------------------------------------------------------

#include "fv3_cgrid_d2a2c.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// big_number, sw_core.F90:63-67. The Fortran selects 1.E8 under OVERLOAD_R4
// and 1.E30 otherwise; the R4 CUDA build corresponds to the OVERLOAD_R4
// Fortran build. The value must be selected per precision rather than written
// once as fv_real(1e30), which is not representable in binary32.
#if defined(FV3_GPU_FV_REAL_IS_FLOAT)
inline constexpr fv_real big_number = fv_real(1.0e8);
#else
inline constexpr fv_real big_number = fv_real(1.0e30);
#endif

// npt, sw_core.F90:2848-2852.
__device__ __forceinline__ int64_t d2a2c_npt(const D2A2CArgs &a) {
  return (a.grid_type < 3 && !a.nested) ? int64_t(4) : int64_t(-2);
}

// ---------------------------------------------------------------------------
// Phase 0 (:2854-2856)  utmp(:,:) = big_number ; vtmp(:,:) = big_number
// over the declared workspace extent (bd%isd:bd%ied, bd%jsd:bd%jed) (:2813).
//
// REPAIR D3. Also removes the compute-sanitizer --tool=initcheck exposure of
// the cells that no later phase writes but the corner blocks read.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_prefill_workspace(D2A2CArgs a, int64_t k) {
  const int64_t i = a.isd() + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.jsd() + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.ied() || j > a.jed()) return;

  a.utmp(i, j, k) = big_number;
  a.vtmp(i, j, k) = big_number;
}

// ---------------------------------------------------------------------------
// Phase 1 (:2890-2902)  Interior 4th-order fill, non-nested branch.
//
//   utmp:  do j = max(npt,js-1), min(npy-npt,je+1)     (:2893)
//          do i = max(npt,isd),  min(npx-npt,ied)      (:2894)
//   vtmp:  do j = max(npt,jsd),  min(npy-npt,jed)      (:2898)
//          do i = max(npt,is-1), min(npx-npt,ie+1)     (:2899)
//
// REPAIR D2: the two boxes are deliberately ASYMMETRIC -- utmp is wide in i
// and narrow in j, vtmp is wide in j and narrow in i. They are transcribed
// literally and are NOT unified. Fusing the two Fortran loops into one kernel
// is safe: the utmp loop reads only u, the vtmp loop reads only v, so there is
// no cross-loop dependency.
//
// Launched over the whole halo box; both boxes are subsets of it because
// ng >= 1 gives js-1 >= jsd, je+1 <= jed, is-1 >= isd, ie+1 <= ied.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_interior_fourth_order(D2A2CArgs a, int64_t k) {
  const int64_t i = a.isd() + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.jsd() + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.ied() || j > a.jed()) return;

  const int64_t npt = d2a2c_npt(a);
  const int64_t npx = a.npx;
  const int64_t npy = a.npy;

  // :2893-2896
  if (j >= max(npt, a.bounds.js - 1) && j <= min(npy - npt, a.bounds.je + 1) &&
      i >= max(npt, a.isd()) && i <= min(npx - npt, a.ied())) {
    a.utmp(i, j, k) = a2 * (a.u(i, j - 1, k) + a.u(i, j + 2, k)) +
                      a1 * (a.u(i, j, k) + a.u(i, j + 1, k));  // :2895
  }

  // :2898-2901
  if (j >= max(npt, a.jsd()) && j <= min(npy - npt, a.jed()) &&
      i >= max(npt, a.bounds.is - 1) && i <= min(npx - npt, a.bounds.ie + 1)) {
    a.vtmp(i, j, k) = a2 * (a.v(i - 1, j, k) + a.v(i + 2, j, k)) +
                      a1 * (a.v(i, j, k) + a.v(i + 1, j, k));  // :2900
  }
}

// ---------------------------------------------------------------------------
// Phase 2 (:2904-2943)  "edges": four 2nd-order bands, grid_type < 3 only.
// Each band OVERWRITES both utmp and vtmp:
//
//   (a) :2909-2916  j = jsd, npt-1                      i = isd, ied
//   (b) :2917-2924  j = npy-npt+1, jed                  i = isd, ied
//   (c) :2926-2933  j = max(npt,jsd), min(npy-npt,jed)  i = isd, npt-1
//   (d) :2934-2941  j = max(npt,jsd), min(npy-npt,jed)  i = npx-npt+1, ied
//
// REPAIR D2. Notes on the transcription:
//  * The Fortran `if` guards at :2909, :2917, :2926, :2934 are supersets of
//    the non-emptiness condition of their own loop range (e.g. `js==1` implies
//    jsd = 1-ng < npt = 4, and `jsd<npt` IS the non-emptiness test for
//    j=jsd,npt-1). Implementing the ranges therefore reproduces the guards
//    exactly; no separate flag is needed.
//  * All four bands write the identical expressions (:2912-2913), so evaluating
//    them in a single pass is equivalent to the sequential Fortran even where
//    two bands touch the same cell.
//  * This is a SEPARATE kernel launch from phase 1 so that the overwrite
//    strictly follows the 4th-order fill, as in the Fortran. (The bands and the
//    4th-order boxes are in fact disjoint for npt = 4, but the ordering is not
//    relied upon.)
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_second_order_bands(D2A2CArgs a, int64_t k) {
  if (a.grid_type >= 3) return;  // :2907

  const int64_t i = a.isd() + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.jsd() + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.ied() || j > a.jed()) return;

  const int64_t npt = d2a2c_npt(a);
  const int64_t npx = a.npx;
  const int64_t npy = a.npy;

  // j range shared by bands (c) and (d): :2927 / :2935
  const int64_t jlo = max(npt, a.jsd());
  const int64_t jhi = min(npy - npt, a.jed());

  const bool band_a = (j <= npt - 1);                                    // :2910
  const bool band_b = (j >= npy - npt + 1);                              // :2918
  const bool band_c = (j >= jlo && j <= jhi) && (i <= npt - 1);           // :2928
  const bool band_d = (j >= jlo && j <= jhi) && (i >= npx - npt + 1);     // :2936

  if (!(band_a || band_b || band_c || band_d)) return;

  a.utmp(i, j, k) = fv_real(0.5) * (a.u(i, j, k) + a.u(i, j + 1, k));  // :2912
  a.vtmp(i, j, k) = fv_real(0.5) * (a.v(i, j, k) + a.v(i + 1, j, k));  // :2913
}

// ---------------------------------------------------------------------------
// Phase 3 (:2945-2951)  Contra-variant components at cell center.
//   do j = js-1-id, je+1+id ; do i = is-1-id, ie+1+id     (id = 1 if dord4,
//                                                          :2842-2846)
// REPAIR D5: base was (is-1, js-1); it is now (is-1-id, js-1-id) and the
// launcher supplies (ie-is+3+2*id, je-js+3+2*id).
// The formula body is unchanged.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void compute_a_grid_winds(D2A2CArgs a, int64_t k) {
  const int64_t id = a.dord4 ? int64_t(1) : int64_t(0);
  const int64_t is_m = a.bounds.is - 1 - id;
  const int64_t ie_p = a.bounds.ie + 1 + id;
  const int64_t js_m = a.bounds.js - 1 - id;
  const int64_t je_p = a.bounds.je + 1 + id;

  const int64_t i = is_m + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js_m + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > ie_p || j > je_p) return;

  const fv_real utmp_ij = a.utmp(i, j, k);
  const fv_real vtmp_ij = a.vtmp(i, j, k);
  const fv_real cosa_s_ij = a.cosa_s(i, j);
  const fv_real rsin2_ij = a.rsin2(i, j);

  a.ua(i, j, k) = (utmp_ij - vtmp_ij * cosa_s_ij) * rsin2_ij;  // :2948
  a.va(i, j, k) = (vtmp_ij - utmp_ij * cosa_s_ij) * rsin2_ij;  // :2949
}

// ---------------------------------------------------------------------------
// Phase 4 (:2955-2979)  "A -> C / Fix the edges / Xdir": utmp corner
// reflections across the cubed-sphere panel corners.
//
// REPAIR D4. Unconditional in the Fortran (only the corner flags gate it, NOT
// grid_type), and it sits after the ua/va loop at :2946-2951 and before the uc
// interpolation at :2991 that reads utmp.
//
// Reads vtmp only, writes utmp only, and the four written spans are disjoint,
// so a single kernel has no intra-kernel dependency. Each Fortran loop is 3
// cells, so one 32-thread block with a t < 3 guard covers it.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_utmp_corner_reflect_x(D2A2CArgs a, int64_t k) {
  const int64_t t = static_cast<int64_t>(threadIdx.x);
  if (t > 2) return;

  const int64_t npx = a.npx;
  const int64_t npy = a.npy;
  const int64_t je = a.bounds.je;

  if (a.sw_corner) {                                     // :2960
    const int64_t i = int64_t(-2) + t;                   // do i=-2,0    :2961
    a.utmp(i, 0, k) = -a.vtmp(0, 1 - i, k);              // :2962
  }
  if (a.se_corner) {                                     // :2965
    const int64_t i = int64_t(0) + t;                    // do i=0,2     :2966
    a.utmp(npx + i, 0, k) = a.vtmp(npx, i + 1, k);       // :2967
  }
  if (a.ne_corner) {                                     // :2970
    const int64_t i = int64_t(0) + t;                    // do i=0,2     :2971
    a.utmp(npx + i, npy, k) = -a.vtmp(npx, je - i, k);   // :2972
  }
  if (a.nw_corner) {                                     // :2975
    const int64_t i = int64_t(-2) + t;                   // do i=-2,0    :2976
    a.utmp(i, npy, k) = a.vtmp(0, je + i, k);            // :2977
  }
}

// ---------------------------------------------------------------------------
// Phase 5 (:2981-2996)  4th-order A->C interpolation for uc/ut, interior.
//   ifirst = max(3,is-1), ilast = min(npx-2,ie+2) for grid_type<3 .and.
//   .not.nested (:2982-2983), else is-1 / ie+2 (:2985-2986)
//   do j = js-1, je+1 ; do i = ifirst, ilast             (:2991-2992)
// Previously verified correct; unchanged apart from citations.
// ---------------------------------------------------------------------------
__global__ void a2c_interpolate_uc_interior(D2A2CArgs a, int64_t k) {
  const int64_t ifirst = (a.grid_type < 3 && !a.nested)
                             ? max(int64_t(3), a.bounds.is - 1)
                             : a.bounds.is - 1;
  const int64_t ilast = (a.grid_type < 3 && !a.nested)
                            ? min(int64_t(a.npx - 2), a.bounds.ie + 2)
                            : a.bounds.ie + 2;

  const int64_t i = ifirst + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > ilast || j > a.bounds.je + 1) return;

  // Reads utmp(i-2,j) down to isd = is-3 and utmp(i+1,j) up to ied = ie+3:
  // this is the stencil that makes ng >= 3 mandatory.
  const fv_real uc_val = a2 * (a.utmp(i - 2, j, k) + a.utmp(i + 1, j, k)) +
                         a1 * (a.utmp(i - 1, j, k) + a.utmp(i, j, k));  // :2993
  a.uc(i, j, k) = uc_val;
  a.ut(i, j, k) = (uc_val - a.v(i, j, k) * a.cosa_u(i, j)) *
                  a.rsin_u(i, j);  // :2994
}

// ---------------------------------------------------------------------------
// Phase 6 (:2998-3015)  ua corner reflections, inside `if (grid_type < 3)`.
//
// REPAIR D4. Reads va, writes ua, and must precede the west/east uc edge
// kernels which read ua(-1:2,j) / ua(npx-2:npx+1,j).
//
// The Fortran has no loop here: two explicit scalar assignments per corner.
// They are transcribed one statement per statement and the kernel is therefore
// launched with a single thread.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_ua_corner_reflect(D2A2CArgs a, int64_t k) {
  if (a.grid_type >= 3) return;  // :2998

  const int64_t npx = a.npx;
  const int64_t npy = a.npy;

  if (a.sw_corner) {                             // :3000
    a.ua(-1, 0, k) = -a.va(0, 2, k);             // :3001
    a.ua(0, 0, k) = -a.va(0, 1, k);              // :3002
  }
  if (a.se_corner) {                             // :3004
    a.ua(npx, 0, k) = a.va(npx, 1, k);           // :3005
    a.ua(npx + 1, 0, k) = a.va(npx, 2, k);       // :3006
  }
  if (a.ne_corner) {                             // :3008
    a.ua(npx, npy, k) = -a.va(npx, npy - 1, k);      // :3009
    a.ua(npx + 1, npy, k) = -a.va(npx, npy - 2, k);  // :3010
  }
  if (a.nw_corner) {                             // :3012
    a.ua(-1, npy, k) = a.va(0, npy - 2, k);      // :3013
    a.ua(0, npy, k) = a.va(0, npy - 1, k);       // :3014
  }
}

// ---------------------------------------------------------------------------
// Phase 7a (:3017-3031)  uc/ut west edge, `is==1 .and. .not. nested`.
//   do j = js-1, je+1
// Body previously verified correct; the only change is REPAIR D9, storing
// ut(1,j) (:3020). Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void a2c_uc_west_edge(D2A2CArgs a, int64_t k) {
  const int64_t j = a.bounds.js - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (j > a.bounds.je + 1) return;
  if (a.nested || a.bounds.is != 1) return;  // :3017

  // uc(0,j) = c1*utmp(-2,j) + c2*utmp(-1,j) + c3*utmp(0,j)          :3019
  a.uc(0, j, k) =
      c1 * a.utmp(-2, j, k) + c2 * a.utmp(-1, j, k) + c3 * a.utmp(0, j, k);

  // ut(1,j) = edge_interpolate4(ua(-1:2,j), dxa(-1:2,j))            :3020
  const fv_real ut1 = edge_interpolate4(
      a.ua(-1, j, k), a.ua(0, j, k), a.ua(1, j, k), a.ua(2, j, k),
      a.dxa(-1, j), a.dxa(0, j), a.dxa(1, j), a.dxa(2, j));
  a.ut(1, j, k) = ut1;  // REPAIR D9: the Fortran assigns ut(1,j) here.

  // Want to use the UPSTREAM value                                  :3022-3026
  if (ut1 > fv_real(0)) {
    a.uc(1, j, k) = ut1 * a.sin_sg(0, j, 2);  // sin_sg(0,j,3)
  } else {
    a.uc(1, j, k) = ut1 * a.sin_sg(1, j, 0);  // sin_sg(1,j,1)
  }

  // uc(2,j) = c1*utmp(3,j) + c2*utmp(2,j) + c3*utmp(1,j)            :3027
  a.uc(2, j, k) =
      c1 * a.utmp(3, j, k) + c2 * a.utmp(2, j, k) + c3 * a.utmp(1, j, k);

  a.ut(0, j, k) = (a.uc(0, j, k) - a.v(0, j, k) * a.cosa_u(0, j)) *
                  a.rsin_u(0, j);  // :3028
  a.ut(2, j, k) = (a.uc(2, j, k) - a.v(2, j, k) * a.cosa_u(2, j)) *
                  a.rsin_u(2, j);  // :3029
}

// ---------------------------------------------------------------------------
// Phase 7b (:3033-3046)  uc/ut east edge, `(ie+1)==npx .and. .not. nested`.
//   do j = js-1, je+1
// Body previously verified correct; the only change is REPAIR D9, storing
// ut(npx,j) (:3036). Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void a2c_uc_east_edge(D2A2CArgs a, int64_t k) {
  const int64_t j = a.bounds.js - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  if (j > a.bounds.je + 1) return;
  if (a.nested || a.bounds.ie + 1 != a.npx) return;  // :3033

  const int64_t npx = a.npx;

  // :3035
  a.uc(npx - 1, j, k) = c1 * a.utmp(npx - 3, j, k) +
                        c2 * a.utmp(npx - 2, j, k) +
                        c3 * a.utmp(npx - 1, j, k);

  // :3036
  const fv_real ut_npx = edge_interpolate4(
      a.ua(npx - 2, j, k), a.ua(npx - 1, j, k), a.ua(npx, j, k),
      a.ua(npx + 1, j, k), a.dxa(npx - 2, j), a.dxa(npx - 1, j),
      a.dxa(npx, j), a.dxa(npx + 1, j));
  a.ut(npx, j, k) = ut_npx;  // REPAIR D9: the Fortran assigns ut(npx,j) here.

  // :3037-3041
  if (ut_npx > fv_real(0)) {
    a.uc(npx, j, k) = ut_npx * a.sin_sg(npx - 1, j, 2);  // sin_sg(npx-1,j,3)
  } else {
    a.uc(npx, j, k) = ut_npx * a.sin_sg(npx, j, 0);      // sin_sg(npx,j,1)
  }

  // :3042
  a.uc(npx + 1, j, k) = c3 * a.utmp(npx, j, k) + c2 * a.utmp(npx + 1, j, k) +
                        c1 * a.utmp(npx + 2, j, k);

  a.ut(npx - 1, j, k) =
      (a.uc(npx - 1, j, k) - a.v(npx - 1, j, k) * a.cosa_u(npx - 1, j)) *
      a.rsin_u(npx - 1, j);  // :3043
  a.ut(npx + 1, j, k) =
      (a.uc(npx + 1, j, k) - a.v(npx + 1, j, k) * a.cosa_u(npx + 1, j)) *
      a.rsin_u(npx + 1, j);  // :3044
}

// ---------------------------------------------------------------------------
// Phase 8 (:3050-3072)  "Ydir": vtmp corner reflections.
//
// REPAIR D4. Unconditional in the Fortran (NOT inside the `if (grid_type < 3)`
// block, which closes at :3048), and it runs after the ua corner block and
// before the vc/vt phase at :3090 that reads vtmp. Fortran order within the
// block is sw, nw, se, ne.
//
// Reads utmp only, writes vtmp only; the four written spans are disjoint.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_vtmp_corner_reflect_y(D2A2CArgs a, int64_t k) {
  const int64_t t = static_cast<int64_t>(threadIdx.x);
  if (t > 2) return;

  const int64_t npx = a.npx;
  const int64_t npy = a.npy;
  const int64_t ie = a.bounds.ie;

  if (a.sw_corner) {                                     // :3053
    const int64_t j = int64_t(-2) + t;                   // do j=-2,0    :3054
    a.vtmp(0, j, k) = -a.utmp(1 - j, 0, k);              // :3055
  }
  if (a.nw_corner) {                                     // :3058
    const int64_t j = int64_t(0) + t;                    // do j=0,2     :3059
    a.vtmp(0, npy + j, k) = a.utmp(j + 1, npy, k);       // :3060
  }
  if (a.se_corner) {                                     // :3063
    const int64_t j = int64_t(-2) + t;                   // do j=-2,0    :3064
    a.vtmp(npx, j, k) = a.utmp(ie + j, 0, k);            // :3065
  }
  if (a.ne_corner) {                                     // :3068
    const int64_t j = int64_t(0) + t;                    // do j=0,2     :3069
    a.vtmp(npx, npy + j, k) = -a.utmp(ie - j, npy, k);   // :3070
  }
}

// ---------------------------------------------------------------------------
// Phase 9 (:3073-3088)  va corner reflections.
//
// REPAIR D4. Also unconditional. Reads ua (including the cells written by
// phase 6, though the spans happen not to overlap) and writes va, and must
// precede the vc/vt phase whose j==1 / j==npy rows read va(i,-1:2).
//
// No Fortran loop: two explicit scalar assignments per corner, so a single
// thread is launched.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void d2a_va_corner_reflect(D2A2CArgs a, int64_t k) {
  const int64_t npx = a.npx;
  const int64_t npy = a.npy;

  if (a.sw_corner) {                               // :3073
    a.va(0, -1, k) = -a.ua(2, 0, k);               // :3074
    a.va(0, 0, k) = -a.ua(1, 0, k);                // :3075
  }
  if (a.se_corner) {                               // :3077
    a.va(npx, 0, k) = a.ua(npx - 1, 0, k);         // :3078
    a.va(npx, -1, k) = a.ua(npx - 2, 0, k);        // :3079
  }
  if (a.ne_corner) {                               // :3081
    a.va(npx, npy, k) = -a.ua(npx - 1, npy, k);        // :3082
    a.va(npx, npy + 1, k) = -a.ua(npx - 2, npy, k);    // :3083
  }
  if (a.nw_corner) {                               // :3085
    a.va(0, npy, k) = a.ua(1, npy, k);             // :3086
    a.va(0, npy + 1, k) = a.ua(2, npy, k);         // :3087
  }
}

// ---------------------------------------------------------------------------
// Phase 10a (:3090-3137)  vc/vt, generic rows.
//   do j = js-1, je+2 ; do i = is-1, ie+1
// For grid_type < 3 the rows j = 0,1,2,npy-1,npy,npy+1 take the special forms
// at :3092-3120 and are handled by a2c_vc_edge_rows; everything else takes the
// 4th-order form at :3124-3125. For grid_type >= 3 the whole range takes the
// 4th-order form with vt = vc (:3131-3135).
//
// The j-row exclusion was previously verified correct and is unchanged.
// REPAIR D8: the grid_type >= 3 branch now sets vt = vc instead of
// (vc - u*cosa_v)*rsin_v. Numerical semantics changed: NO (the previous form
// was simply not the Fortran's for that branch; the production grid_type = 0
// path is unaffected).
// ---------------------------------------------------------------------------
__global__ void a2c_interpolate_vc_interior(D2A2CArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js - 1 + blockIdx.y * blockDim.y + threadIdx.y;

  if (i > a.bounds.ie + 1 || j > a.bounds.je + 2) return;

  // Skip special edge rows that need special treatment (handled separately)
  if (a.grid_type < 3 && !a.nested) {
    if (j == 0 || j == 1 || j == 2 || j == a.npy - 1 || j == a.npy ||
        j == a.npy + 1) {
      return;  // Handled by a2c_vc_edge_rows
    }
  }

  // Reads vtmp(i,j-2) down to jsd = js-3 and vtmp(i,j+1) up to jed = je+3.
  const fv_real vc_val = a2 * (a.vtmp(i, j - 2, k) + a.vtmp(i, j + 1, k)) +
                         a1 * (a.vtmp(i, j - 1, k) + a.vtmp(i, j, k));
  a.vc(i, j, k) = vc_val;
  if (a.grid_type >= 3) {
    a.vt(i, j, k) = vc_val;  // :3134
  } else {
    a.vt(i, j, k) = (vc_val - a.u(i, j, k) * a.cosa_v(i, j)) *
                    a.rsin_v(i, j);  // :3125
  }
}

// ---------------------------------------------------------------------------
// Phase 10b (:3092-3128)  vc/vt special rows, grid_type < 3, non-nested.
//   the outer `do j = js-1, je+2` (:3092) with the six special values of j;
//   inner `do i = is-1, ie+1`.
// blockIdx.y selects one of the six rows.
//
// The per-row bodies were previously verified correct and are unchanged.
// REPAIR D7: the outer loop bound `do j=js-1,je+2` was missing, so for a tile
// that does not touch the south/north face boundary the six rows lie outside
// [jsd,jed] and vc/vt were written out of bounds. Guard added.
// Numerical semantics changed: NO.
// ---------------------------------------------------------------------------
__global__ void a2c_vc_edge_rows(D2A2CArgs a, int64_t k) {
  const int64_t i = a.bounds.is - 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j_idx = blockIdx.y;  // 0..5 for the 6 special rows

  if (i > a.bounds.ie + 1) return;
  if (a.grid_type >= 3 || a.nested) return;

  const int64_t npy = a.npy;
  int64_t j;
  switch (j_idx) {
    case 0: j = 0; break;
    case 1: j = 1; break;
    case 2: j = 2; break;
    case 3: j = npy - 1; break;
    case 4: j = npy; break;
    case 5: j = npy + 1; break;
    default: return;
  }

  // Outer Fortran loop bound (:3092). REPAIR D7.
  if (j < a.bounds.js - 1 || j > a.bounds.je + 2) return;

  if (j == 1) {  // :3093
    // vt(i,1) = edge_interpolate4(va(i,-1:2), dya(i,-1:2))          :3095
    const fv_real vt_val = edge_interpolate4(
        a.va(i, -1, k), a.va(i, 0, k), a.va(i, 1, k), a.va(i, 2, k),
        a.dya(i, -1), a.dya(i, 0), a.dya(i, 1), a.dya(i, 2));
    if (vt_val > fv_real(0)) {
      a.vc(i, j, k) = vt_val * a.sin_sg(i, j - 1, 3);  // sin_sg(i,j-1,4) :3097
    } else {
      a.vc(i, j, k) = vt_val * a.sin_sg(i, j, 1);      // sin_sg(i,j,2)   :3099
    }
    a.vt(i, j, k) = vt_val;
  } else if (j == 0 || j == npy - 1) {  // :3102
    // vc(i,j) = c1*vtmp(i,j-2) + c2*vtmp(i,j-1) + c3*vtmp(i,j)      :3104
    a.vc(i, j, k) = c1 * a.vtmp(i, j - 2, k) + c2 * a.vtmp(i, j - 1, k) +
                    c3 * a.vtmp(i, j, k);
    a.vt(i, j, k) = (a.vc(i, j, k) - a.u(i, j, k) * a.cosa_v(i, j)) *
                    a.rsin_v(i, j);  // :3105
  } else if (j == 2 || j == npy + 1) {  // :3107
    // vc(i,j) = c1*vtmp(i,j+1) + c2*vtmp(i,j) + c3*vtmp(i,j-1)      :3109
    a.vc(i, j, k) = c1 * a.vtmp(i, j + 1, k) + c2 * a.vtmp(i, j, k) +
                    c3 * a.vtmp(i, j - 1, k);
    a.vt(i, j, k) = (a.vc(i, j, k) - a.u(i, j, k) * a.cosa_v(i, j)) *
                    a.rsin_v(i, j);  // :3110
  } else if (j == npy) {  // :3112
    // vt(i,npy) = edge_interpolate4(va(i,npy-2:npy+1), dya(...))    :3114
    const fv_real vt_val = edge_interpolate4(
        a.va(i, npy - 2, k), a.va(i, npy - 1, k), a.va(i, npy, k),
        a.va(i, npy + 1, k), a.dya(i, npy - 2), a.dya(i, npy - 1),
        a.dya(i, npy), a.dya(i, npy + 1));
    if (vt_val > fv_real(0)) {
      a.vc(i, j, k) = vt_val * a.sin_sg(i, j - 1, 3);  // sin_sg(i,j-1,4) :3116
    } else {
      a.vc(i, j, k) = vt_val * a.sin_sg(i, j, 1);      // sin_sg(i,j,2)   :3118
    }
    a.vt(i, j, k) = vt_val;
  }
}

}  // namespace

fv3_gpu_status launch_d2a2c_vect_level(const D2A2CArgs &a, int64_t k,
                                       void *stream) {
  if (!stream || !a.u.data || !a.v.data || !a.ua.data || !a.va.data ||
      !a.uc.data || !a.vc.data || !a.ut.data || !a.vt.data ||
      !a.utmp.data || !a.vtmp.data)
    return FV3_GPU_INVALID_ARGUMENT;
  if (!a.cosa_u.data || !a.cosa_v.data || !a.rsin_u.data || !a.rsin_v.data ||
      !a.cosa_s.data || !a.rsin2.data || !a.dxa.data || !a.dya.data ||
      !a.sin_sg.data)
    return FV3_GPU_INVALID_ARGUMENT;

  // REPAIR D1: the halo width is now explicit rather than assumed to be 1.
  // ng >= 3 is mandatory: a2c_interpolate_uc_interior reads utmp(i-2,j) at
  // i = is-1 (=> is-3) and utmp(i+1,j) at i = ie+2 (=> ie+3), and the vc phase
  // reads vtmp(i,j-2) at j = js-1 (=> js-3). A smaller halo cannot be served,
  // and no substitute value is silently used.
  if (a.ng < 3) return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  // The nested branch (sw_core.F90:2858-2887) is a distinct algorithm:
  // different utmp/vtmp fill ranges, no npt banding, and ua/va over the full
  // (isd:ied, jsd:jed) box. It is not implemented here, so reject rather than
  // run the global-cubed-sphere formulas on a nested grid.
  if (a.nested) return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  constexpr dim3 block(16, 8);

  // Full halo box (isd:ied, jsd:jed) -- phases 0, 1 and 2.
  const auto nxd = static_cast<unsigned>(a.ied() - a.isd() + 1);
  const auto nyd = static_cast<unsigned>(a.jed() - a.jsd() + 1);
  const dim3 grid_halo((nxd + block.x - 1) / block.x,
                       (nyd + block.y - 1) / block.y);

  // Phase 0 (:2854-2856): utmp/vtmp = big_number.
  d2a_prefill_workspace<<<grid_halo, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Phase 1 (:2893-2902): 4th-order interior fill (two asymmetric boxes).
  d2a_interior_fourth_order<<<grid_halo, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Phase 2 (:2907-2943): four 2nd-order bands overwrite utmp AND vtmp.
  d2a_second_order_bands<<<grid_halo, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  // Phase 3 (:2946-2951): ua/va over is-1-id:ie+1+id, js-1-id:je+1+id.
  // REPAIR D5: extent now includes the id ring on BOTH sides.
  const int64_t id = a.dord4 ? int64_t(1) : int64_t(0);
  const auto nxa =
      static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3 + 2 * id);
  const auto nya =
      static_cast<unsigned>(a.bounds.je - a.bounds.js + 3 + 2 * id);
  const dim3 grid_a((nxa + block.x - 1) / block.x,
                    (nya + block.y - 1) / block.y);
  compute_a_grid_winds<<<grid_a, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  const bool any_corner =
      a.sw_corner || a.se_corner || a.ne_corner || a.nw_corner;

  // Phase 4 (:2960-2979): utmp x-direction corner reflections. Unconditional
  // in the Fortran except for the corner flags themselves; must precede the uc
  // interpolation below, which reads utmp.
  if (any_corner) {
    d2a_utmp_corner_reflect_x<<<1, 32, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  }

  // Phase 5 (:2991-2996): uc/ut 4th-order interior, i = ifirst..ilast,
  // j = js-1..je+1. Grid spans the widest possible i range (is-1..ie+2); the
  // kernel clips to ifirst/ilast.
  const auto nx_uc = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 4);
  const auto ny_uc = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);
  const dim3 grid_uc((nx_uc + block.x - 1) / block.x,
                     (ny_uc + block.y - 1) / block.y);
  a2c_interpolate_uc_interior<<<grid_uc, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  const auto nj_edge = static_cast<unsigned>(a.bounds.je - a.bounds.js + 3);

  // Phases 6 and 7 (:2998-3048) are inside `if (grid_type < 3)`.
  if (a.grid_type < 3) {
    // Phase 6 (:3000-3015): ua corner reflections, before the uc edge kernels
    // which read ua(-1:2,j) / ua(npx-2:npx+1,j).
    if (any_corner) {
      d2a_ua_corner_reflect<<<1, 1, 0, cuda_stream>>>(a, k);
      if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
    }

    // Phase 7a (:3017-3031) and 7b (:3033-3046): j = js-1..je+1.
    if (a.bounds.is == 1) {
      a2c_uc_west_edge<<<(nj_edge + 31) / 32, 32, 0, cuda_stream>>>(a, k);
      if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
    }
    if (a.bounds.ie + 1 == a.npx) {
      a2c_uc_east_edge<<<(nj_edge + 31) / 32, 32, 0, cuda_stream>>>(a, k);
      if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
    }
  }

  // Phases 8 (:3053-3072) and 9 (:3073-3088) are OUTSIDE the grid_type < 3
  // block -- the `endif` for that block is at :3048 -- so they are launched
  // unconditionally (subject only to the corner flags).
  if (any_corner) {
    d2a_vtmp_corner_reflect_y<<<1, 32, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

    d2a_va_corner_reflect<<<1, 1, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  }

  // Phase 10 (:3090-3137): vc/vt over i = is-1..ie+1, j = js-1..je+2.
  const auto nx_vc = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 3);
  const auto ny_vc = static_cast<unsigned>(a.bounds.je - a.bounds.js + 4);
  const dim3 grid_vc((nx_vc + block.x - 1) / block.x,
                     (ny_vc + block.y - 1) / block.y);
  a2c_interpolate_vc_interior<<<grid_vc, block, 0, cuda_stream>>>(a, k);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  if (a.grid_type < 3) {
    // Six special rows; each block row handles one. The kernel clips to
    // j = js-1..je+2 (REPAIR D7).
    const dim3 grid_vc_edge((nx_vc + 31) / 32, 6);
    a2c_vc_edge_rows<<<grid_vc_edge, 32, 0, cuda_stream>>>(a, k);
    if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;
  }

  return FV3_GPU_SUCCESS;
}

}  // namespace fv3::gpu::cgrid
