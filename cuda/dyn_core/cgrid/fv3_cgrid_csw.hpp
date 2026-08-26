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

/* Launch c_sw for one vertical level. Assumes d2a2c_vect and, when nord > 0,
 * divergence_corner have already been called for this level.
 *
 * Milestone 3A.9: the former single launch_csw_fluxes_level is split into the
 * three phases below, because sw_core.F90 interleaves the delp/pt/w corner
 * fills BETWEEN the two flux directions and the two directions write the same
 * corner cells with different values. See the block comment at
 * launch_csw_scale_level in fv3_cgrid_csw.cu for the full defect analysis.
 *
 * The authoritative per-level order the caller must emit is:
 *
 *   launch_csw_scale_level                (:183-200)
 *   fill2_4corners(delp, pt, X)           (:206)   } grid_type < 3 and
 *   fill_4corners(w, X)                   (:237)   } corner-flag guarded
 *   launch_csw_x_flux_level               (:238-253)
 *   fill2_4corners(delp, pt, Y)           (:257)
 *   fill_4corners(w, Y)                   (:284)
 *   launch_csw_y_flux_scalars_level       (:285-309)
 *   launch_csw_ke_vort_level              (:321-498)
 *   launch_csw_cwind_update_level         (:500-510)
 *
 * The corner fills are the WP-A-local kernels in fv3_cgrid_corners.hpp, NOT the
 * WP-E corner_service dispatch slot (an earlier comment here said otherwise;
 * fv3_cgrid_corners.hpp:8-13 records why the local kernels are correct).
 *
 * Both flux launchers reject a hydrostatic config: the hydrostatic flux arms
 * (:208-234, :258-282) are not translated, and silently skipping them would
 * leave delpc/ptc unwritten. */
fv3_gpu_status launch_csw_scale_level(const CSwArgs &args, int64_t k, void *stream);
fv3_gpu_status launch_csw_x_flux_level(const CSwArgs &args, int64_t k, void *stream);
fv3_gpu_status launch_csw_y_flux_scalars_level(const CSwArgs &args, int64_t k,
                                              void *stream);
fv3_gpu_status launch_csw_ke_vort_level(const CSwArgs &args, int64_t k, void *stream);
fv3_gpu_status launch_csw_cwind_update_level(const CSwArgs &args, int64_t k, void *stream);

}  // namespace fv3::gpu::cgrid
