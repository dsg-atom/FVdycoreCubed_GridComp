// WP-A C-grid predictor dispatch adapter.
// Orchestrates d2a2c_vect, c_sw, divergence_corner, and update_dz_c using
// WP-E v3.1 ExecutionContext and corner_service.
//
// Authoritative ordering from model/sw_core.F90:c_sw and dyn_core.F90.

#include "fv3_cgrid_predictor.hpp"
#include "fv3_cgrid_d2a2c.hpp"
#include "fv3_cgrid_csw.hpp"
#include "fv3_cgrid_divergence.hpp"
#include "fv3_cgrid_dz.hpp"
#include "fv3_gpu_errors.hpp"

namespace fv3::gpu::cgrid {

int launch_cgrid_predictor_v3(const ExecutionContext &context, fv3_gpu_error *error) {
  int rc = validate_execution_context(context, error);
  if (rc != FV3_GPU_SUCCESS) return rc;

  if (context.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "C-grid predictor requires the WP-E compute stream");

  // Reject unsupported configurations
  if (context.immutable->config.hydrostatic)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "hydrostatic C-grid predictor is not translated");
  if (context.immutable->topology->nested)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "nested C-grid predictor is not translated");

  // Field resolution helpers
  const auto field = [&](FieldId id) -> const DeviceViewBase * {
    return context.fields->find(id);
  };
  const auto grid = [&](FieldId id) -> const DeviceViewBase * {
    return context.immutable->grid->find(id);
  };

  // Resolve all required fields
  const auto *u_view = field(FieldId::u);
  const auto *v_view = field(FieldId::v);
  const auto *w_view = field(FieldId::w);
  const auto *delp_view = field(FieldId::delp);
  const auto *pt_view = field(FieldId::pt);
  const auto *ua_view = field(FieldId::ua);
  const auto *va_view = field(FieldId::va);
  const auto *uc_view = field(FieldId::uc);
  const auto *vc_view = field(FieldId::vc);
  const auto *ut_view = field(FieldId::ut);
  const auto *vt_view = field(FieldId::vt);
  const auto *delpc_view = field(FieldId::delpc);
  const auto *ptc_view = field(FieldId::ptc);
  const auto *divgd_view = field(FieldId::divgd);
  const auto *gz_view = field(FieldId::gz);

  const auto *dx_view = grid(FieldId::dx);
  const auto *dy_view = grid(FieldId::dy);
  const auto *dxc_view = grid(FieldId::dxc);
  const auto *dyc_view = grid(FieldId::dyc);
  const auto *rdxc_view = grid(FieldId::rdxc);
  const auto *rdyc_view = grid(FieldId::rdyc);
  const auto *rarea_view = grid(FieldId::rarea);
  const auto *rarea_c_view = grid(FieldId::rarea_c);
  const auto *cosa_u_view = grid(FieldId::cosa_u);
  const auto *cosa_v_view = grid(FieldId::cosa_v);
  const auto *cosa_s_view = grid(FieldId::cosa_s);
  const auto *sina_u_view = grid(FieldId::sina_u);
  const auto *sina_v_view = grid(FieldId::sina_v);
  const auto *rsin_u_view = grid(FieldId::rsin_u);
  const auto *rsin_v_view = grid(FieldId::rsin_v);
  const auto *rsin2_view = grid(FieldId::rsin2);
  const auto *dxa_view = grid(FieldId::dxa);
  const auto *dya_view = grid(FieldId::dya);
  const auto *f_c_view = grid(FieldId::f_c);
  const auto *sin_sg_view = grid(FieldId::sin_sg);
  const auto *cos_sg_view = grid(FieldId::cos_sg);

  // Validate required fields exist
  if (!u_view || !v_view || !w_view || !delp_view || !pt_view ||
      !ua_view || !va_view || !uc_view || !vc_view || !ut_view || !vt_view ||
      !delpc_view || !ptc_view || !divgd_view || !gz_view ||
      !dx_view || !dy_view || !dxc_view || !dyc_view || !rdxc_view || !rdyc_view ||
      !rarea_view || !rarea_c_view || !cosa_u_view || !cosa_v_view || !cosa_s_view ||
      !sina_u_view || !sina_v_view || !rsin_u_view || !rsin_v_view || !rsin2_view ||
      !dxa_view || !dya_view || !f_c_view || !sin_sg_view || !cos_sg_view)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "C-grid predictor missing required registered fields");

  // Extract bounds and config
  const auto &bounds = context.immutable->bounds;
  const auto &config = context.immutable->config;
  const auto &topo = *context.immutable->topology;
  const auto &params = *context.dispatch_params;
  void *stream = context.submission.stream;

  // Build common bounds struct
  Bounds b{bounds.is, bounds.ie, bounds.js, bounds.je,
           static_cast<int64_t>(bounds.npz)};

  // TODO: Resolve workspace fields from WorkspaceId::cgrid
  // For now, this adapter demonstrates the structure but cannot execute
  // without workspace allocation. Integration must provide workspace views.

  // The predictor sequence is:
  // 1. d2a2c_vect (per level)
  // 2. divergence_corner (if nord > 0, per level)
  // 3. corner fills for delp/pt (direction 1)
  // 4. corner fills for w (direction 1)
  // 5. x fluxes
  // 6. corner fills for delp/pt (direction 2)
  // 7. corner fills for w (direction 2)
  // 8. y fluxes
  // 9. delpc/ptc/wc computation
  // 10. KE/vorticity
  // 11. C-wind update

  // Mark C-wind halo stale at the end
  rc = context.dependencies->mark_stale(FV3_GPU_H_CWIND);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "failed to mark C-wind halo stale");

  // Note: Full implementation requires workspace resolution and per-level
  // kernel orchestration. This adapter establishes the contract interface;
  // kernel launches are demonstrated in the individual component files.

  return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
              "C-grid predictor workspace resolution not yet integrated; "
              "individual kernel translations are complete");
}

}  // namespace fv3::gpu::cgrid
