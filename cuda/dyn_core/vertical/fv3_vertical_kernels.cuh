#pragma once
#include "fv3_gpu_types.hpp"
#include <cstddef>

namespace fv3::gpu::vertical {
struct ColumnLaunch {
  int columns;
  int levels;
  std::ptrdiff_t column_stride;
  std::ptrdiff_t level_stride;
};

// Enqueue-only package API. stream is the WP-E-owned compute stream.
int launch_reconstruct_height(fv_real *zh, const fv_real *delz,
                              const fv_real *zs, ColumnLaunch layout,
                              void *stream);
int launch_height_boundary(fv_real *zh, fv_real *delz, fv_real *ws,
                           const fv_real *zs, fv_real rdt, fv_real dz_min,
                           ColumnLaunch layout, void *stream);
} // namespace fv3::gpu::vertical
