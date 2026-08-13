#include "fv3_cgrid.hpp"
#include "fv3_gpu_errors.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::cgrid {
namespace {

// Authoritative source: model/dyn_core.F90:p_grad_c, nonhydrostatic branch.
// One thread owns one C face and executes the two Fortran products and final
// sum in source order. The block z index is converted to the descriptor's
// logical vertical lower bound; kernels never assume Fortran k starts at one.
__global__ void pressure_gradient_c_x(PressureGradientCArgs a) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t level = blockIdx.z;
  const int64_t k = a.delpc.lower[2] + level;
  const int64_t ki = a.pkc.lower[2] + level;
  if (i > a.bounds.ie + 1 || j > a.bounds.je || level >= a.bounds.npz)
    return;

  const fv_real denom = a.delpc(i - 1, j, k) + a.delpc(i, j, k);
  const fv_real first =
      (a.gz(i - 1, j, ki + 1) - a.gz(i, j, ki)) *
      (a.pkc(i, j, ki + 1) - a.pkc(i - 1, j, ki));
  const fv_real second =
      (a.gz(i - 1, j, ki) - a.gz(i, j, ki + 1)) *
      (a.pkc(i - 1, j, ki + 1) - a.pkc(i, j, ki));
  a.uc(i, j, k) = a.uc(i, j, k) +
                    a.dt2 * a.rdxc(i, j) / denom * (first + second);
}

__global__ void pressure_gradient_c_y(PressureGradientCArgs a) {
  const int64_t i = a.bounds.is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = a.bounds.js + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t level = blockIdx.z;
  const int64_t k = a.delpc.lower[2] + level;
  const int64_t ki = a.pkc.lower[2] + level;
  if (i > a.bounds.ie || j > a.bounds.je + 1 || level >= a.bounds.npz)
    return;

  const fv_real denom = a.delpc(i, j - 1, k) + a.delpc(i, j, k);
  const fv_real first =
      (a.gz(i, j - 1, ki + 1) - a.gz(i, j, ki)) *
      (a.pkc(i, j, ki + 1) - a.pkc(i, j - 1, ki));
  const fv_real second =
      (a.gz(i, j - 1, ki) - a.gz(i, j, ki + 1)) *
      (a.pkc(i, j - 1, ki + 1) - a.pkc(i, j, ki));
  a.vc(i, j, k) = a.vc(i, j, k) +
                    a.dt2 * a.rdyc(i, j) / denom * (first + second);
}

} // namespace

fv3_gpu_status launch_pressure_gradient_c(const PressureGradientCArgs &a,
                                          void *stream) {
  if (stream == nullptr || a.bounds.npz <= 0 || a.bounds.ie < a.bounds.is ||
      a.bounds.je < a.bounds.js || a.delpc.data == nullptr ||
      a.pkc.data == nullptr || a.gz.data == nullptr || a.uc.data == nullptr ||
      a.vc.data == nullptr || a.rdxc.data == nullptr || a.rdyc.data == nullptr)
    return FV3_GPU_INVALID_ARGUMENT;

  constexpr dim3 block(16, 8, 1);
  const auto nx = static_cast<unsigned>(a.bounds.ie - a.bounds.is + 2);
  const auto ny = static_cast<unsigned>(a.bounds.je - a.bounds.js + 2);
  const dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y,
                  static_cast<unsigned>(a.bounds.npz));
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  pressure_gradient_c_x<<<grid, block, 0, cuda_stream>>>(a);
  if (cudaPeekAtLastError() != cudaSuccess)
    return FV3_GPU_CUDA_ERROR;
  pressure_gradient_c_y<<<grid, block, 0, cuda_stream>>>(a);
  return cudaPeekAtLastError() == cudaSuccess ? FV3_GPU_SUCCESS
                                              : FV3_GPU_CUDA_ERROR;
}

int launch_cgrid_pressure_v2(const ExecutionContext &context,
                             fv3_gpu_error *error) {
  int rc = validate_execution_context(context, error);
  if (rc != FV3_GPU_SUCCESS)
    return rc;
  if (context.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "C pressure gradient requires the WP-E compute stream");

  const auto required = [&](FieldId id, bool immutable = false)
      -> const DeviceViewBase * {
    const FieldRegistry *registry = immutable ? context.immutable->grid
                                               : context.fields;
    return registry == nullptr ? nullptr : registry->find(id);
  };
  const auto *delpc = required(FieldId::delpc);
  const auto *pkc = required(FieldId::pkc);
  const auto *gz = required(FieldId::gz);
  const auto *uc = required(FieldId::uc);
  const auto *vc = required(FieldId::vc);
  const auto *rdxc = required(FieldId::rdxc, true);
  const auto *rdyc = required(FieldId::rdyc, true);
  if (!delpc || !pkc || !gz || !uc || !vc || !rdxc || !rdyc)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "C pressure gradient is missing a required registered field");
  if (context.immutable->config.hydrostatic)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "hydrostatic p_grad_c is not translated");

  const auto &b = context.immutable->bounds;
  PressureGradientCArgs args{};
  args.bounds = {b.is, b.ie, b.js, b.je, b.npz};
  args.dt2 = static_cast<fv_real>(0.5 * context.immutable->config.bdt /
                                  context.immutable->config.n_split);
  args.delpc = make_view<const fv_real, 3>(*delpc);
  args.pkc = make_view<const fv_real, 3>(*pkc);
  args.gz = make_view<const fv_real, 3>(*gz);
  args.uc = make_view<fv_real, 3>(*uc);
  args.vc = make_view<fv_real, 3>(*vc);
  args.rdxc = make_view<const fv_real, 2>(*rdxc);
  args.rdyc = make_view<const fv_real, 2>(*rdyc);
  rc = launch_pressure_gradient_c(args, context.submission.stream);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "failed to enqueue C pressure-gradient kernels");
  rc = context.dependencies->mark_stale(FV3_GPU_H_CWIND);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "failed to mark C-wind halo stale");
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::cgrid
