#pragma once

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// Arguments for divergence_corner (global cubed-sphere, non-nested)
// Authoritative source: model/sw_core.F90:1659-1764
struct DivergenceCornerArgs {
  Bounds bounds{};
  int32_t npx{}, npy{};
  uint8_t grid_type{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};

  // Input D-grid winds
  FieldView<const fv_real, 3> u;   // (isd:ied, jsd:jed+1, k)
  FieldView<const fv_real, 3> v;   // (isd:ied+1, jsd:jed, k)

  // Input A-grid winds (for cos_sg correction)
  FieldView<const fv_real, 3> ua;  // (isd:ied, jsd:jed, k)
  FieldView<const fv_real, 3> va;  // (isd:ied, jsd:jed, k)

  // Output: corner divergence on B grid
  FieldView<fv_real, 3> divg_d;    // (isd:ied+1, jsd:jed+1, k)

  // Geometry
  FieldView<const fv_real, 2> dxc;      // C-u stagger
  FieldView<const fv_real, 2> dyc;      // C-v stagger
  FieldView<const fv_real, 2> rarea_c;  // B stagger
  FieldView<const fv_real, 3> sin_sg;   // B stagger (i,j,component)
  FieldView<const fv_real, 3> cos_sg;   // B stagger (i,j,component)

  // Workspace for uf/vf (from C-grid workspace)
  FieldView<fv_real, 3> uf;
  FieldView<fv_real, 3> vf;
};

// Launch divergence_corner for one vertical level. Returns status.
// Nested grids are milestone-one unsupported and, because this struct carries
// no `nested` flag, are not currently *rejected* either -- see the note at the
// top of fv3_cgrid_divergence.cu.
fv3_gpu_status launch_divergence_corner_level(const DivergenceCornerArgs &args,
                                              int64_t k, void *stream);

}  // namespace fv3::gpu::cgrid
