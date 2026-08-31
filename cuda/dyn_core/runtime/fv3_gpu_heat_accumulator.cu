/* Prompt 6B -- dyn_core.F90:853-861. See fv3_gpu_heat_accumulator.hpp for the
 * authoritative contract and for why a second array is required rather than
 * accumulating in place. */
#include "fv3_gpu_heat_accumulator.hpp"

#include <cuda_runtime_api.h>

namespace fv3::gpu {

namespace {

__global__ void heat_accumulate_kernel(fv_real *dst, const fv_real *src,
                                       int64_t lo_i, int64_t lo_j, int64_t lo_k,
                                       int64_t st_i, int64_t st_j, int64_t st_k,
                                       int64_t i0, int64_t j0, int64_t k0,
                                       int64_t nx, int64_t ny, int64_t nk) {
  const int64_t i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = j0 + blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = k0 + blockIdx.z;
  if (i >= i0 + nx || j >= j0 + ny || k >= k0 + nk) return;
  const int64_t off = (i - lo_i) * st_i + (j - lo_j) * st_j + (k - lo_k) * st_k;
  dst[off] += src[off];
}

} // namespace

int fv3_heat_source_accumulate(fv_real *dst, const fv_real *src,
                               std::int64_t lo_i, std::int64_t lo_j, std::int64_t lo_k,
                               std::int64_t st_i, std::int64_t st_j, std::int64_t st_k,
                               std::int64_t i0, std::int64_t i1,
                               std::int64_t j0, std::int64_t j1,
                               std::int64_t k0, std::int64_t k1,
                               void *stream) noexcept {
  if (!dst || !src) return static_cast<int>(cudaErrorInvalidValue);
  const std::int64_t nx = i1 - i0 + 1, ny = j1 - j0 + 1, nk = k1 - k0 + 1;
  if (nx <= 0 || ny <= 0 || nk <= 0) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 threads(16, 16, 1);
  const dim3 grid(static_cast<unsigned>((nx + threads.x - 1) / threads.x),
                  static_cast<unsigned>((ny + threads.y - 1) / threads.y),
                  static_cast<unsigned>(nk));
  heat_accumulate_kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      dst, src, lo_i, lo_j, lo_k, st_i, st_j, st_k, i0, j0, k0, nx, ny, nk);
  return static_cast<int>(cudaGetLastError());
}

} // namespace fv3::gpu
