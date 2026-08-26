#pragma once

// Authoritative source: model/sw_core.F90
//   fill2_4corners  :3227-3286
//   fill_4corners   :3289-3347
// (GFDL_atmos_cubed_sphere tag geos/v3.0.0, ca30e26).
//
// A-grid scalar corner halo fills "as needed by c_core". These are LOCAL
// corner algebra on a single rank's own halo, so per the milestone 3A.8
// precedent they are plain WP-A device kernels with plain launchers and are
// deliberately NOT routed through the generic WP-E corner_edge_service
// dispatch slot. Nothing here touches NumericalDispatch, CornerServiceRequest,
// or the WP-E contract.
//
// Callers in the production nonhydrostatic path:
//   sw_core.F90:206  fill2_4corners(delp, pt, 1, ...)
//   sw_core.F90:237  fill_4corners (w,       1, ...)
//   sw_core.F90:257  fill2_4corners(delp, pt, 2, ...)
//   sw_core.F90:284  fill_4corners (w,       2, ...)
//   nh_utils.F90:162 fill_4corners (gz2,     1, ...)
//   nh_utils.F90:174 fill_4corners (gz2,     2, ...)

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// Direction argument of the Fortran `dir` dummy (:3229, :3291).
// 1: x-dir; 2: y-dir.
constexpr int32_t kCornerFillDirX = 1;
constexpr int32_t kCornerFillDirY = 2;

// Everything the Fortran bodies actually read besides the fields themselves.
//
// NOTE: the Fortran subroutines take `bd` and unpack is/ie/js/je/isd/ied/
// jsd/jed (:3238-3245, :3299-3306), but the bodies never reference any of
// them -- they are dead locals. The index expressions depend only on npx,
// npy and the four corner flags. So no Bounds member is carried here.
struct CornerFillArgs {
  int32_t npx{}, npy{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};
};

// Host-side helper: reinterpret level k of a 3-D (i,j,k) field as the 2-D
// (i,j) view these kernels operate on. Pure pointer arithmetic on the
// descriptor data already in the view -- no new layout, no copy.
//
// `k` is in the same index space as the 3-D view's own third dimension
// (i.e. it must already have had any Fortran 1-based -> 0-based adjustment
// applied by the caller, exactly as when calling the per-level kernels in
// this directory).
template <typename T>
inline FieldView<T, 2> level_slice(const FieldView<T, 3> &f, int64_t k) {
  FieldView<T, 2> view{};
  view.data = f.data + (k - f.lower[2]) * f.stride[2];
  view.lower[0] = f.lower[0];
  view.lower[1] = f.lower[1];
  view.stride[0] = f.stride[0];
  view.stride[1] = f.stride[1];
  return view;
}

// fill_4corners (sw_core.F90:3289-3347). Mutates `q` in place in the corner
// halo. `dir` must be kCornerFillDirX or kCornerFillDirY.
//
// c_sw passes a level slice of the live prognostic w; update_dz_c passes the
// genuinely 2-D gz2. Both arrive here as a 2-D (i,j) view.
fv3_gpu_status launch_fill_4corners(const CornerFillArgs &args,
                                    const FieldView<fv_real, 2> &q,
                                    int32_t dir, void *stream);

// fill2_4corners (sw_core.F90:3227-3286). Mutates `q1` and `q2` in place in
// the corner halo. `dir` must be kCornerFillDirX or kCornerFillDirY.
//
// q1 and q2 must not alias (c_sw passes delp and pt, which are distinct).
fv3_gpu_status launch_fill2_4corners(const CornerFillArgs &args,
                                     const FieldView<fv_real, 2> &q1,
                                     const FieldView<fv_real, 2> &q2,
                                     int32_t dir, void *stream);

}  // namespace fv3::gpu::cgrid
