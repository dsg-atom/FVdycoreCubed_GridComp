#include "fv3_vertical_kernels.cuh"
#include "fv3_vertical_column.hpp"
#include <cuda_runtime.h>

namespace fv3::gpu::vertical {
namespace {
__global__ void reconstruct_height_kernel(fv_real *zh, const fv_real *delz,
                                          const fv_real *zs,
                                          ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<const fv_real> dz{delz + c * l.column_stride, l.level_stride};
  reconstruct_height_column(z, dz, zs[c], l.levels);
}

__global__ void height_boundary_kernel(fv_real *zh, fv_real *delz, fv_real *ws,
                                       const fv_real *zs, fv_real rdt,
                                       fv_real dz_min, ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<fv_real> dz{delz + c * l.column_stride, l.level_stride};
  ws[c] = update_height_boundary_column(z, dz, zs[c], rdt, dz_min, l.levels);
}
} // namespace

int launch_reconstruct_height(fv_real *zh, const fv_real *delz,
                              const fv_real *zs, ColumnLaunch l, void *stream) {
  if (!zh || !delz || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  reconstruct_height_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                                static_cast<cudaStream_t>(stream)>>>(zh, delz, zs, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int launch_height_boundary(fv_real *zh, fv_real *delz, fv_real *ws,
                           const fv_real *zs, fv_real rdt, fv_real dz_min,
                           ColumnLaunch l, void *stream) {
  if (!zh || !delz || !ws || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  height_boundary_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      zh, delz, ws, zs, rdt, dz_min, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
} // namespace fv3::gpu::vertical
