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
 * STALE AS OF MILESTONE 3A.9: every stage listed above as null is now wired,
 * and fv3_build_production_dispatch is the complete production table. Read the
 * per-slot comments in fv3_gpu_production_dispatch.cu, which are current.
 *
 * corner_edge_service was likewise an adapter that unconditionally failed,
 * which forced config.nf_ke <= 1. Prompt 6B replaced it with a real scalar
 * A-grid copy_corners service (tp_core.F90:247) so del2_cubed can run the
 * authoritative nf_ke = min(3, nord+1) passes. */
int fv3_build_production_dispatch(NumericalDispatch &out, fv3_gpu_error *e) noexcept;

} // namespace fv3::gpu
