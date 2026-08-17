#pragma once

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// Arguments for c_sw (C-grid shallow water predictor)
// Authoritative source: model/sw_core.F90:103-512
struct CSwArgs {
  Bounds bounds{};
  int32_t npx{}, npy{};
  int32_t nord{};
  fv_real dt2{};
  uint8_t grid_type{};
  uint8_t nested{};
  uint8_t hydrostatic{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};

  // Input/output D-grid winds (used for vorticity transport)
  FieldView<fv_real, 3> u;   // (isd:ied, jsd:jed+1, k)
  FieldView<fv_real, 3> v;   // (isd:ied+1, jsd:jed, k)

  // Input/output A-grid winds (from d2a2c_vect)
  FieldView<fv_real, 3> ua;  // (isd:ied, jsd:jed, k)
  FieldView<fv_real, 3> va;  // (isd:ied, jsd:jed, k)

  // Input/output C-grid winds
  FieldView<fv_real, 3> uc;  // (isd:ied+1, jsd:jed, k)
  FieldView<fv_real, 3> vc;  // (isd:ied, jsd:jed+1, k)

  // Input/output transport velocities (from d2a2c_vect)
  FieldView<fv_real, 3> ut;  // (isd:ied, jsd:jed, k)
  FieldView<fv_real, 3> vt;  // (isd:ied, jsd:jed, k)

  // Input scalars
  FieldView<const fv_real, 3> delp;  // A stagger
  FieldView<const fv_real, 3> pt;    // A stagger
  FieldView<fv_real, 3> w;           // A stagger (nonhydrostatic)

  // Output: C-grid predictor scalars
  FieldView<fv_real, 3> delpc;  // A stagger
  FieldView<fv_real, 3> ptc;    // A stagger
  FieldView<fv_real, 3> wc;     // A stagger (nonhydrostatic only)

  // Output: B-grid corner divergence (if nord > 0)
  FieldView<fv_real, 3> divg_d;  // B stagger

  // Geometry
  FieldView<const fv_real, 2> dx;       // X face
  FieldView<const fv_real, 2> dy;       // Y face
  FieldView<const fv_real, 2> dxc;      // C-u
  FieldView<const fv_real, 2> dyc;      // C-v
  FieldView<const fv_real, 2> rdxc;     // C-u
  FieldView<const fv_real, 2> rdyc;     // C-v
  FieldView<const fv_real, 2> rarea;    // A stagger
  FieldView<const fv_real, 2> cosa_u;   // D-u
  FieldView<const fv_real, 2> cosa_v;   // D-v
  FieldView<const fv_real, 2> sina_u;   // D-u
  FieldView<const fv_real, 2> sina_v;   // D-v
  FieldView<const fv_real, 2> f_c;      // B stagger (corner Coriolis)
  FieldView<const fv_real, 2> rarea_c;  // B stagger
  FieldView<const fv_real, 3> sin_sg;   // B stagger
  FieldView<const fv_real, 3> cos_sg;   // B stagger

  // Workspace arrays
  FieldView<fv_real, 3> vort;   // ke/vort workspace
  FieldView<fv_real, 3> ke;     // kinetic energy
  FieldView<fv_real, 3> fx;     // x flux
  FieldView<fv_real, 3> fy;     // y flux
  FieldView<fv_real, 3> fx1;    // delp x flux
  FieldView<fv_real, 3> fy1;    // delp y flux
  FieldView<fv_real, 3> fx2;    // w x flux (nonhydrostatic)
  FieldView<fv_real, 3> fy2;    // w y flux (nonhydrostatic)
};

// Launch c_sw for one vertical level.
// Assumes d2a2c_vect and divergence_corner (if nord>0) already called.
// Corner fills must be invoked via corner_service by caller.
fv3_gpu_status launch_csw_fluxes_level(const CSwArgs &args, int64_t k, void *stream);
fv3_gpu_status launch_csw_ke_vort_level(const CSwArgs &args, int64_t k, void *stream);
fv3_gpu_status launch_csw_cwind_update_level(const CSwArgs &args, int64_t k, void *stream);

}  // namespace fv3::gpu::cgrid
