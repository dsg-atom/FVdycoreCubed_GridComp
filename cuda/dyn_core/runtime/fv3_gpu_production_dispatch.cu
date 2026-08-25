#include "fv3_gpu_production_dispatch.hpp"
#include "fv3_gpu_errors.hpp"
#include "../cgrid/fv3_cgrid.hpp"
#include "../vertical/fv3_vertical_kernels.cuh"
#include "../operators/wp_d_contract_cuda.hpp"
#include "../dgrid/wp_b_contract_cuda.hpp"

namespace fv3::gpu {
namespace {
int corner_service_not_implemented(const ExecutionContext &, const CornerServiceRequest &,
                                    fv3_gpu_error *e) noexcept {
  return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
              "corner_edge_service is not integrated in this milestone's production dispatch; "
              "only configurations that never require it (config.nf_ke<=1) are supported");
}
} // namespace

int fv3_build_production_dispatch(NumericalDispatch &out, fv3_gpu_error *e) noexcept {
  clear_error(e);
  NumericalDispatch d{};
  d.contract_version = internal_contract_version;
  d.cgrid_predictor = nullptr;
  d.cgrid_vertical_solver = nullptr;
  d.cgrid_pressure = &fv3::gpu::cgrid::launch_cgrid_pressure_v3;
  // Milestone 3A.8 (WP-B.5) populates this slot. It was deliberately nullptr
  // through 3A.7, when only d_sw's individual slices existed:
  //   3A.4 -- update_dz_d height transport (launch_dgrid_height_transport_cuda)
  //   3A.5 -- arbitrary scalar transport   (launch_scalar_transport_cuda)
  //   3A.6 -- momentum transport           (launch_xtp_u_cuda / launch_ytp_v_cuda,
  //           the vb/ub constructions, the kinetic-energy terms, the vorticity
  //           precursor and the D-wind update)
  //   3A.7 -- divergence damping           (both nord_k branches, the two
  //           fill_corners rotations, smag_corner, the d_con heat-source chain
  //           and the damp_v del6_vt_flux add -- wp_b_divergence_cuda.cuh)
  // Wiring a PARTIAL dgrid_dynamics would have let the production path silently
  // skip stages, so the rule was: not until the complete ordered driver exists.
  //
  // It now does. launch_d_sw_cuda (dgrid/wp_b_dsw_cuda.cuh) executes the full
  // sw_core.F90 d_sw sequence S1..S38 -- the four services above plus the twelve
  // driver-owned kernels (contravariant/edge/corner winds, advective-flux
  // finalize, the joint pt/delp update, the w chain, the flux capacitors and the
  // heat_source reset) -- with the per-level nord_k / nord_v / nord_w / damp_vt /
  // damp_w / damp_t / d_con / kgb selectors that dyn_core.F90:699-765 recomputes
  // inside its k loop, so both divergence-damping branch families can be live in
  // one call. Anything it has not been validated against
  // (grid_type != 0, nested, hydrostatic, hord_* other than the production
  // first-run set, a_imp != 1, inline_q, use_cond, do_f3d, do_skeb, rot3) is
  // rejected with FV3_GPU_UNSUPPORTED_CONFIGURATION before it mutates any state.
  d.dgrid_dynamics = &fv3::gpu::dgrid::launch_dgrid_dynamics_cuda;
  // Milestone 3A.4: the authoritative update_dz_d transport body
  // (nh_utils.F90:255-305) -- edge_profile -> ra_x/ra_y -> per-level
  // damped/undamped split -> fv_tp_2d -> del6_vt_flux -> area-weighted height
  // update -- wired for WP-C's update_dz_d dependency.
  //
  // Milestone 3A.3's "damp_c mapping ambiguity" is resolved and that note is
  // gone: the zh path calls sw_core.F90's del6_vt_flux, which uses the raw
  // per-level damp(k) and emits separate fx2/fy2, NOT tp_core.F90's deln_flux
  // with its (damp_c*da_min)**(nord+1) coefficient folded into fx/fy. There
  // was no coefficient to map.
  //
  // Milestone 3A.5 repaired launch_deln_flux_transport_cuda to the authoritative
  // tp_core.F90:1263 deln_flux semantics (coefficient (damp_c*da_min)**(nord+1),
  // 2+2*nord copy_corners rotations, flux folded into fx/fy, 1.e-4 damp_c
  // threshold) and consumes it from launch_scalar_transport_cuda. It is still
  // never invoked from the height path -- the two damping operators stay
  // distinct, exactly as the Fortran keeps them distinct.
  d.dgrid_height_transport = &fv3::gpu::dgrid::launch_dgrid_height_transport_cuda;
  d.dgrid_height = nullptr;
  d.vertical_solver = &fv3::gpu::vertical::launch_vertical_solver;
  d.nh_pressure = &fv3::gpu::operators::launch_nh_pressure_cuda;
  d.post_loop_heating = &fv3::gpu::operators::launch_post_loop_heating_cuda;
  d.corner_edge_service = &corner_service_not_implemented;
  out = d;
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu
