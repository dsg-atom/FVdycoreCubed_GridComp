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
  // Still nullptr after Milestone 3A.5. WP-B.2 landed the reusable arbitrary
  // scalar transport service (launch_scalar_transport_cuda), but d_sw also needs
  // momentum/vorticity transport, divergence damping, the kinetic-energy /
  // heat-source terms and the order-critical pt/delp/w state updates before this
  // slot can be populated. Wiring a partial dgrid_dynamics would let the
  // production path silently skip those stages, so it stays unwired until
  // WP-B.3/WP-B.5 complete it.
  d.dgrid_dynamics = nullptr;
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
