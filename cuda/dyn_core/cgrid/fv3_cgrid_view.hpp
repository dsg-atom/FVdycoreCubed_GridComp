#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"

#include <cstddef>
#include <cstdint>

namespace fv3::gpu::cgrid {

// A non-owning typed interpretation of the WP-E descriptor. This is not a
// second descriptor or layout: all bounds and strides remain descriptor data.
template <typename T, int Rank> struct FieldView {
  T *data = nullptr;
  int64_t lower[Rank]{};
  int64_t stride[Rank]{};

  constexpr int64_t offset(int64_t i, int64_t j) const {
    static_assert(Rank == 2);
    return (i - lower[0]) * stride[0] + (j - lower[1]) * stride[1];
  }

  constexpr int64_t offset(int64_t i, int64_t j, int64_t k) const {
    static_assert(Rank == 3);
    return (i - lower[0]) * stride[0] + (j - lower[1]) * stride[1] +
           (k - lower[2]) * stride[2];
  }

#if defined(__CUDACC__)
  __device__ __forceinline__ T &operator()(int64_t i, int64_t j) const {
    return data[offset(i, j)];
  }
  __device__ __forceinline__ T &operator()(int64_t i, int64_t j,
                                            int64_t k) const {
    return data[offset(i, j, k)];
  }
#endif
};

template <typename T, int Rank>
inline FieldView<T, Rank> make_view(const fv3_gpu_field_desc &desc,
                                    void *device_base) {
  FieldView<T, Rank> view{};
  view.data = static_cast<T *>(device_base);
  for (int d = 0; d < Rank; ++d) {
    view.lower[d] = desc.lower[d];
    view.stride[d] = desc.stride_elements[d];
  }
  return view;
}

struct Bounds {
  int64_t is, ie, js, je;
  int64_t npz;
};

} // namespace fv3::gpu::cgrid
