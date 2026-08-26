#pragma once

#include "fv3_cgrid_view.hpp"
#include "fv3_cgrid_corners.hpp"

namespace fv3::gpu::cgrid {

// Arguments for update_dz_c (interface geopotential height transport)
// Authoritative source: model/nh_utils.F90:70-207
struct UpdateDzCArgs {
  Bounds bounds{};
  int32_t npx{}, npy{}, km{};
  fv_real dt{}, dz_min{};
  uint8_t grid_type{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};

  // Input transport velocities (from c_sw)
  FieldView<const fv_real, 3> ut;  // (is-ng:ie+ng, js-ng:je+ng, km)
  FieldView<const fv_real, 3> vt;  // (is-ng:ie+ng, js-ng:je+ng, km)

  // Vertical coordinate
  FieldView<const fv_real, 1> dp0;  // (km) reference layer thickness

  // Input/output interface height
  FieldView<fv_real, 3> gz;  // (is-ng:ie+ng, js-ng:je+ng, km+1)

  // Input surface height
  FieldView<const fv_real, 2> zs;  // A stagger

  // Input area
  FieldView<const fv_real, 2> area;  // A stagger

  // Output surface vertical velocity
  FieldView<fv_real, 2> ws;  // A stagger

  // Workspace
  FieldView<fv_real, 2> gz2;  // copy of gz(:,:,k) for corner fill
  FieldView<fv_real, 3> xfx;  // x interface wind
  FieldView<fv_real, 3> yfx;  // y interface wind
  FieldView<fv_real, 3> fx;   // x flux
  FieldView<fv_real, 3> fy;   // y flux
};

// Launch the complete authoritative body of nh_utils.F90's `do 6000 k=1,km+1`
// loop for one interface level k (Fortran 1-based: 1 <= k <= km+1).
//
// Milestone 3A.9: this now includes the two fill_4corners(gz2, dir) calls at
// nh_utils.F90:162 and :174. They CANNOT be hoisted out of the level loop (gz2
// holds a different level each pass) and they cannot be left to the caller
// without making a partial, numerically wrong sequence callable. Both are
// guarded by `if (grid_type < 3)` in Fortran, which is honoured here from
// args.grid_type rather than assumed.
//
// These are the WP-A-local corner kernels in fv3_cgrid_corners.hpp, not the
// WP-E corner_edge_service dispatch slot -- see that header for why.
fv3_gpu_status launch_update_dz_c_interface(const UpdateDzCArgs &args,
                                            const CornerFillArgs &corners,
                                            int64_t k, void *stream);

// Monotonicity enforcement and ws output (after all interfaces)
fv3_gpu_status launch_update_dz_c_finalize(const UpdateDzCArgs &args, void *stream);

}  // namespace fv3::gpu::cgrid
