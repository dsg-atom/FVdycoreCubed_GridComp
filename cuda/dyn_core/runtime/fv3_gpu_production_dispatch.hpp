#pragma once

#include "fv3_gpu_internal.hpp"

namespace fv3::gpu {

/* Milestone 3A.1: builds the production NumericalDispatch table out of the
 * CUDA implementations that are actually wired today. Populates only slots
 * with a real, integrated CUDA adapter:
 *   cgrid_pressure     -> fv3::gpu::cgrid::launch_cgrid_pressure_v3   (WP-A p_grad_c)
 *   vertical_solver    -> fv3::gpu::vertical::launch_vertical_solver (WP-C riem_solver3)
 *   nh_pressure        -> fv3::gpu::operators::launch_nh_pressure_cuda       (WP-D)
 *   post_loop_heating  -> fv3::gpu::operators::launch_post_loop_heating_cuda (WP-D)
 * Every other stage (cgrid_predictor, cgrid_vertical_solver, dgrid_dynamics,
 * dgrid_height_transport, dgrid_height) is left null: no adapter exists yet
 * (cgrid_predictor), the stage is a non-functional delegating stub
 * (cgrid_vertical_solver), or WP-B contains no CUDA kernels at all
 * (dgrid_dynamics / dgrid_height_transport / dgrid_height). This is not a
 * complete acoustic timestep -- it demonstrates that the wired stages
 * genuinely execute on the GPU through fv3_gpu_dyn_core_step.
 *
 * corner_edge_service is populated with an adapter that unconditionally
 * fails: none of the wired stages call it for the single-pass
 * (config.nf_ke<=1) production configuration this milestone targets. */
int fv3_build_production_dispatch(NumericalDispatch &out, fv3_gpu_error *e) noexcept;

} // namespace fv3::gpu
