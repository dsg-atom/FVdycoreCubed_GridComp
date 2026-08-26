// Authoritative source: model/sw_core.F90
//   fill2_4corners  :3227-3286
//   fill_4corners   :3289-3347
// (GFDL_atmos_cubed_sphere tag geos/v3.0.0, ca30e26).
//
// A-grid scalar corner halo fills as needed by c_core. WP-A-local: see the
// header for why these are plain kernels and not a WP-E corner_edge_service
// dispatch.
//
// EQUIVALENCE OF THE TWO ROUTINES
// ------------------------------------------------------------------------
// fill2_4corners(q1,q2,dir,...) IS exactly fill_4corners(q1,dir,...) followed
// by fill_4corners(q2,dir,...). Statement-by-statement correspondence:
//
//   dir=1 sw : :3250 q1(-1,0)=q1(0,2)           == :3311 q(-1,0)=q(0,2)
//              :3250 q1(0,0)=q1(0,1)            == :3312 q(0,0)=q(0,1)
//   dir=1 se : :3254 q1(npx+1,0)=q1(npx,2)      == :3315
//              :3254 q1(npx,0)=q1(npx,1)        == :3316
//   dir=1 nw : :3258 q1(0,npy)=q1(0,npy-1)      == :3319
//              :3258 q1(-1,npy)=q1(0,npy-2)     == :3320
//   dir=1 ne : :3262 q1(npx,npy)=q1(npx,npy-1)  == :3323
//              :3262 q1(npx+1,npy)=q1(npx,npy-2)== :3324
//   dir=2 sw : :3268 q1(0,0)=q1(1,0)            == :3329
//              :3268 q1(0,-1)=q1(2,0)           == :3330
//   dir=2 se : :3272 q1(npx,0)=q1(npx-1,0)      == :3333
//              :3272 q1(npx,-1)=q1(npx-2,0)     == :3334
//   dir=2 nw : :3276 q1(0,npy)=q1(1,npy)        == :3337
//              :3276 q1(0,npy+1)=q1(2,npy)      == :3338
//   dir=2 ne : :3280 q1(npx,npy)=q1(npx-1,npy)  == :3341
//              :3280 q1(npx,npy+1)=q1(npx-2,npy)== :3342
//
// and the q2 lines (:3251, :3255, :3259, :3263, :3269, :3273, :3277, :3281)
// are the identical expressions on q2. There is no cross-field coupling: no
// statement in fill2_4corners reads q1 to write q2 or vice versa. The
// interleaved q1/q2 ordering inside each Fortran corner block is therefore
// immaterial for distinct arrays, and one shared device routine can serve
// both entry points. Numerical semantics changed: NO.
//
// Corner-block ordering: fill_4corners and fill2_4corners both order the
// blocks sw, se, nw, ne (note: fill3_4corners at :3178-3223 uses sw, se, ne,
// nw instead). We reproduce sw, se, nw, ne. To make the ordering exactly
// sequential -- and thus bit-identical to Fortran even on a degenerate
// single-rank tile that owns more than one corner -- the whole fill runs on
// one device thread. There are at most 8 assignments, so this is not a
// performance concern at this stage.
//
// In-place mutation is load-bearing, not cosmetic: for c_sw the targets are
// the live prognostic delp, pt and w, and the values written at the corners
// are read afterwards (e.g. the nonhydrostatic scalar stage reads w(0,0)).

#include "fv3_cgrid_corners.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// fill_4corners, select case(dir) case(1): sw_core.F90:3309-3325.
// Also serves fill2_4corners case(1) (:3248-3264) per the equivalence table
// above.
__device__ __forceinline__ void fill_4corners_x(const CornerFillArgs &a,
                                                const FieldView<fv_real, 2> &q) {
  const int64_t npx = int64_t(a.npx);
  const int64_t npy = int64_t(a.npy);

  if (a.sw_corner) {                            // :3310
    q(-1, 0) = q(0, 2);                         // :3311
    q(0, 0) = q(0, 1);                          // :3312
  }                                             // :3313
  if (a.se_corner) {                            // :3314
    q(npx + 1, 0) = q(npx, 2);                  // :3315
    q(npx, 0) = q(npx, 1);                      // :3316
  }                                             // :3317
  if (a.nw_corner) {                            // :3318
    q(0, npy) = q(0, npy - 1);                  // :3319
    q(-1, npy) = q(0, npy - 2);                 // :3320
  }                                             // :3321
  if (a.ne_corner) {                            // :3322
    q(npx, npy) = q(npx, npy - 1);              // :3323
    q(npx + 1, npy) = q(npx, npy - 2);          // :3324
  }                                             // :3325
}

// fill_4corners, select case(dir) case(2): sw_core.F90:3327-3343.
// Also serves fill2_4corners case(2) (:3266-3282).
__device__ __forceinline__ void fill_4corners_y(const CornerFillArgs &a,
                                                const FieldView<fv_real, 2> &q) {
  const int64_t npx = int64_t(a.npx);
  const int64_t npy = int64_t(a.npy);

  if (a.sw_corner) {                            // :3328
    q(0, 0) = q(1, 0);                          // :3329
    q(0, -1) = q(2, 0);                         // :3330
  }                                             // :3331
  if (a.se_corner) {                            // :3332
    q(npx, 0) = q(npx - 1, 0);                  // :3333
    q(npx, -1) = q(npx - 2, 0);                 // :3334
  }                                             // :3335
  if (a.nw_corner) {                            // :3336
    q(0, npy) = q(1, npy);                      // :3337
    q(0, npy + 1) = q(2, npy);                  // :3338
  }                                             // :3339
  if (a.ne_corner) {                            // :3340
    q(npx, npy) = q(npx - 1, npy);              // :3341
    q(npx, npy + 1) = q(npx - 2, npy);          // :3342
  }                                             // :3343
}

// fill_4corners (:3289-3347), one field.
__global__ void fill_4corners_kernel(CornerFillArgs a, FieldView<fv_real, 2> q,
                                     int32_t dir) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  if (dir == kCornerFillDirX) {          // select case(dir) / case(1)  :3308-3309
    fill_4corners_x(a, q);
  } else if (dir == kCornerFillDirY) {   // case(2)                     :3327
    fill_4corners_y(a, q);
  }
  // No `case default` in the Fortran (:3345 end select): any other dir is a
  // no-op there. The launcher rejects it before reaching this point.
}

// fill2_4corners (:3227-3286), two fields. q1 first then q2, matching the
// Fortran statement order within each corner block; see the equivalence table
// at the top of this file.
__global__ void fill2_4corners_kernel(CornerFillArgs a,
                                      FieldView<fv_real, 2> q1,
                                      FieldView<fv_real, 2> q2, int32_t dir) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  if (dir == kCornerFillDirX) {          // :3247-3248
    fill_4corners_x(a, q1);
    fill_4corners_x(a, q2);
  } else if (dir == kCornerFillDirY) {   // :3266
    fill_4corners_y(a, q1);
    fill_4corners_y(a, q2);
  }
}

}  // namespace

fv3_gpu_status launch_fill_4corners(const CornerFillArgs &a,
                                    const FieldView<fv_real, 2> &q,
                                    int32_t dir, void *stream) {
  if (!stream || !q.data) return FV3_GPU_INVALID_ARGUMENT;
  if (dir != kCornerFillDirX && dir != kCornerFillDirY)
    return FV3_GPU_INVALID_ARGUMENT;

  // Nothing to do if this rank owns no cubed-sphere corner. The Fortran would
  // fall through every `if` block; skipping the launch is equivalent.
  if (!a.sw_corner && !a.se_corner && !a.ne_corner && !a.nw_corner)
    return FV3_GPU_SUCCESS;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  fill_4corners_kernel<<<1, 1, 0, cuda_stream>>>(a, q, dir);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_fill2_4corners(const CornerFillArgs &a,
                                     const FieldView<fv_real, 2> &q1,
                                     const FieldView<fv_real, 2> &q2,
                                     int32_t dir, void *stream) {
  if (!stream || !q1.data || !q2.data) return FV3_GPU_INVALID_ARGUMENT;
  if (dir != kCornerFillDirX && dir != kCornerFillDirY)
    return FV3_GPU_INVALID_ARGUMENT;

  if (!a.sw_corner && !a.se_corner && !a.ne_corner && !a.nw_corner)
    return FV3_GPU_SUCCESS;

  auto cuda_stream = static_cast<cudaStream_t>(stream);
  fill2_4corners_kernel<<<1, 1, 0, cuda_stream>>>(a, q1, q2, dir);
  if (cudaPeekAtLastError() != cudaSuccess) return FV3_GPU_CUDA_ERROR;

  return FV3_GPU_SUCCESS;
}

}  // namespace fv3::gpu::cgrid
