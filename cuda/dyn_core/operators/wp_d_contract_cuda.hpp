#pragma once

#include "wp_d_contract.hpp"

namespace fv3::gpu::operators {

/* Milestone 3A.1: CUDA-backed counterparts of launch_nh_pressure /
 * launch_post_loop_heating (wp_d_contract.hpp). Same contract validation,
 * same field binding, same arena layout -- only the numerical substeps
 * dispatch to the existing wp_d_cuda.hpp launchers instead of the host C++
 * implementations in wp_d_operators.cpp. LaunchFn-compatible so they can be
 * assigned directly into NumericalDispatch::nh_pressure /
 * NumericalDispatch::post_loop_heating. */
int launch_nh_pressure_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;
int launch_post_loop_heating_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;
/* Prompt 6B: CUDA counterparts of the omega finalization entries. Same contract
 * validation (omega_preconditions / bind_omega are shared with the host twin),
 * same arena slots 10/11. */
int launch_omega_pem_snapshot_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;
int launch_omega_finalize_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::operators
