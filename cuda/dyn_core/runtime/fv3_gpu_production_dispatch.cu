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
  // Milestone 3A.9 §3: the complete per-level c_sw group of dyn_core.F90:514-524
  // -- d2a2c_vect, the nord>0 divergence_corner, the interleaved
  // fill2_4corners/fill_4corners direction pairs, the ut/vt scaling, both flux
  // directions, delpc/ptc/wc, the kinetic-energy and vorticity chain, and the
  // time-centred uc/vc update. It was a fail-stub through 3A.8.
  d.cgrid_predictor = &fv3::gpu::cgrid::launch_cgrid_predictor_v3;
  // Milestone 3A.9 §4: the C-grid vertical group, a COMPOSITE, not an alias.
  // Through 3A.8 this slot's implementation delegated to launch_vertical_solver
  // (Riem_Solver3); dyn_core.F90:583 calls Riem_Solver_C, a different routine
  // with a different argument list and dt2 rather than dt, so wiring the old body
  // would have been numerically wrong rather than merely redundant.
  // It now runs: gz from delz/zs on the first substep (:445-455), the gz<->zh
  // sync over the full halo box (:557-573), update_dz_c per interface level
  // (:577, dt2) with both fill_4corners(gz2) inside the level body, and
  // Riem_Solver_C (:583, dt2).
  d.cgrid_vertical_solver = &fv3::gpu::vertical::launch_cgrid_vertical_solver;
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
  // Milestone 3A.9 §5/§6: dgrid_height IS update_dz_d (dyn_core.F90:905-906),
  // the WP-C sequencer that CONSUMES the WP-B dgrid_height_transport service
  // above. Wiring it is what finally makes that service reachable through
  // fv3_gpu_dyn_core_step -- it was implemented and wired but had no caller on
  // the production path, because its only caller (launch_update_dz_d) needed the
  // dispatch table and so could not itself be a LaunchFn.
  //
  // NOT launch_dgrid_height_v2. That launcher recomputes delz, which
  // authoritatively belongs to Riem_Solver3 (nh_core.F90:181); wiring it here
  // would write delz a substep early. See the §5 block comment in
  // vertical/fv3_vertical_kernels.cu.
  d.dgrid_height = &fv3::gpu::vertical::launch_dgrid_height_stage;
  d.vertical_solver = &fv3::gpu::vertical::launch_vertical_solver;
  d.nh_pressure = &fv3::gpu::operators::launch_nh_pressure_cuda;
  d.post_loop_heating = &fv3::gpu::operators::launch_post_loop_heating_cuda;
  d.corner_edge_service = &corner_service_not_implemented;
  out = d;
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu
