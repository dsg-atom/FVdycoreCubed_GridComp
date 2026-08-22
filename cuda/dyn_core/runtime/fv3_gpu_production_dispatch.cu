#include "fv3_gpu_production_dispatch.hpp"
#include "fv3_gpu_errors.hpp"
#include "../cgrid/fv3_cgrid.hpp"
#include "../vertical/fv3_vertical_kernels.cuh"
#include "../operators/wp_d_contract_cuda.hpp"

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
  d.dgrid_dynamics = nullptr;
  d.dgrid_height_transport = nullptr;
  d.dgrid_height = nullptr;
  d.vertical_solver = &fv3::gpu::vertical::launch_vertical_solver;
  d.nh_pressure = &fv3::gpu::operators::launch_nh_pressure_cuda;
  d.post_loop_heating = &fv3::gpu::operators::launch_post_loop_heating_cuda;
  d.corner_edge_service = &corner_service_not_implemented;
  out = d;
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu
