#include "fv3_gpu_state_poison.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu {
namespace {

/* Flat over the whole allocation rather than (i,j,k)-shaped on purpose. The
 * authoritative call is
 *     init_ijk_mem(isd,ied, jsd,jed, npz+1, gz, huge_r)
 * (dyn_core.F90:330), which covers every element of gz's declared extent, so
 * there is no cell a descriptor-aware launch would skip and none it would add.
 * Staying flat also means this cannot go wrong for a caller whose padding
 * differs from the bench's. */
__global__ void fill_constant_kernel(fv_real *p, size_t n, fv_real v) {
  const size_t i =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

/* One thread per (i,j,k) of the DECLARED extent; writes only where i and j are
 * both outside the compute box. Launched over the full box rather than over the
 * four corner rectangles separately because the halo widths are per-side and a
 * four-launch version would need six more descriptor fields to get right; the
 * wasted threads are a few thousand predicated-off lanes once per step. */
__global__ void fill_corner_ghosts_kernel(fv_real *p, int64_t lo_i, int64_t lo_j,
                                          int64_t lo_k, int64_t ni, int64_t nj,
                                          int64_t nk, int64_t si, int64_t sj,
                                          int64_t sk, int64_t ci_lo,
                                          int64_t ci_hi, int64_t cj_lo,
                                          int64_t cj_hi, fv_real v) {
  const int64_t a = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (a >= ni * nj * nk) return;
  const int64_t i = lo_i + a % ni;
  const int64_t j = lo_j + (a / ni) % nj;
  const int64_t k = lo_k + a / (ni * nj);
  const bool i_out = (i < ci_lo) || (i > ci_hi);
  const bool j_out = (j < cj_lo) || (j > cj_hi);
  if (!(i_out && j_out)) return;
  p[(i - lo_i) * si + (j - lo_j) * sj + (k - lo_k) * sk] = v;
}

} // namespace

int fv3_fill_device_corner_ghosts(void *device, const std::int64_t lower[3],
                                  const std::uint64_t extent[3],
                                  const std::int64_t stride[3],
                                  std::int64_t ci_lo, std::int64_t ci_hi,
                                  std::int64_t cj_lo, std::int64_t cj_hi,
                                  fv_real value, void *stream) noexcept {
  if (device == nullptr) return static_cast<int>(cudaSuccess);
  const std::int64_t ni = static_cast<std::int64_t>(extent[0]);
  const std::int64_t nj = static_cast<std::int64_t>(extent[1]);
  const std::int64_t nk = static_cast<std::int64_t>(extent[2]);
  if (ni <= 0 || nj <= 0 || nk <= 0) return static_cast<int>(cudaSuccess);
  /* An empty or inverted compute box would make every cell "outside" and turn
   * this into the flat fill it exists to replace. Report it instead. */
  if (ci_hi < ci_lo || cj_hi < cj_lo)
    return static_cast<int>(cudaErrorInvalidValue);
  const std::int64_t n = ni * nj * nk;
  constexpr unsigned block = 256;
  const std::int64_t grid = (n + block - 1) / block;
  if (grid > 2147483647LL) return static_cast<int>(cudaErrorInvalidValue);
  fill_corner_ghosts_kernel<<<static_cast<unsigned>(grid), block, 0,
                              static_cast<cudaStream_t>(stream)>>>(
      static_cast<fv_real *>(device), lower[0], lower[1], lower[2], ni, nj, nk,
      stride[0], stride[1], stride[2], ci_lo, ci_hi, cj_lo, cj_hi, value);
  return static_cast<int>(cudaGetLastError());
}

int fv3_fill_device_constant(void *device, std::size_t bytes, fv_real value,
                             void *stream) noexcept {
  const std::size_t n = bytes / sizeof(fv_real);
  if (device == nullptr || n == 0) return static_cast<int>(cudaSuccess);
  constexpr unsigned block = 256;
  const std::size_t grid = (n + block - 1) / block;
  /* A 1-D grid caps at 2^31-1 blocks, which at 256 elements per block is 5.5e11
   * elements -- far beyond any field this runtime allocates. Guard anyway so an
   * absurd descriptor is a reported error rather than a silent partial fill. */
  if (grid > 2147483647u) return static_cast<int>(cudaErrorInvalidValue);
  fill_constant_kernel<<<static_cast<unsigned>(grid), block, 0,
                         static_cast<cudaStream_t>(stream)>>>(
      static_cast<fv_real *>(device), n, value);
  return static_cast<int>(cudaGetLastError());
}

} // namespace fv3::gpu
