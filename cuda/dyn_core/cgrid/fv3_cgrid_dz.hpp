#pragma once

#include "fv3_cgrid_view.hpp"

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

// Launch update_dz_c for one interface level k.
// Corner fills must be done via corner_service by caller.
fv3_gpu_status launch_update_dz_c_interface(const UpdateDzCArgs &args, int64_t k,
                                            void *stream);

// Monotonicity enforcement and ws output (after all interfaces)
fv3_gpu_status launch_update_dz_c_finalize(const UpdateDzCArgs &args, void *stream);

}  // namespace fv3::gpu::cgrid
