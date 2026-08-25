#pragma once

// Shared, non-numerical plumbing for the WP-B contract binders
// (wp_b_contract_cuda.cpp, wp_b_dsw_contract_cuda.cpp).
//
// These are DeviceViewBase -> View{1,2,3} adapters, bounds-coverage predicates
// and overflow-safe size arithmetic. Nothing here computes anything physical,
// so factoring it out does not centralize a numerical algorithm -- it just
// stops two binders from carrying two copies of the same reinterpretation
// rules, which is exactly the kind of divergence that produces a silent
// stride/origin mismatch in one binder and not the other.

#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_types.hpp"
// View1/View2/View3 live in fv3::gpu::operators (operators/wp_d_operators.hpp);
// wp_b_cuda.cuh is what imports them into fv3::gpu::dgrid.
#include "dgrid/wp_b_cuda.cuh"

#include <cstdint>

namespace fv3::gpu::dgrid::contract_views {

using fv3::gpu::dgrid::View1;
using fv3::gpu::dgrid::View2;
using fv3::gpu::dgrid::View3;

template <class T>
inline View3<T> to_view3(const DeviceViewBase *v) {
  return View3<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->lower[2],
                  v->stride[0], v->stride[1], v->stride[2]};
}

template <class T>
inline View2<T> to_view2(const DeviceViewBase *v) {
  return View2<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->stride[0],
                  v->stride[1]};
}

template <class T>
inline View1<T> to_view1(const DeviceViewBase *v) {
  return View1<T>{static_cast<T *>(v->data), v->lower[0], v->stride[0]};
}

inline View3<const fv_real> to_const3(const View3<fv_real> &v) {
  return View3<const fv_real>{v.data, v.lo_i, v.lo_j, v.lo_k, v.si, v.sj, v.sk};
}

inline View1<const fv_real> to_const1(const View1<fv_real> &v) {
  return View1<const fv_real>{v.data, v.lo_i, v.si};
}

// Arena arrays are all sized identically ([is-h,ie+h] x [js-h,je+h] x
// [0,npz]); wrap each raw pointer as a batched View3 over that box. Both WP-B
// arenas document this layout and both binders carve it the same way.
inline View3<fv_real> to_arena_view3(void *p, const DomainBounds &b) {
  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t ni = (b.ie + h) - (b.is - h) + 1;
  const int64_t nj = (b.je + h) - (b.js - h) + 1;
  return View3<fv_real>{static_cast<fv_real *>(p), b.is - h, b.js - h, 0, 1, ni, ni * nj};
}

// Same box, int32 payload. d_sw's `smt` is a Fortran LOGICAL; an fv_real-sized
// slot always holds an int32 array of the same shape, in either precision.
inline View3<int32_t> to_arena_view3_i(void *p, const DomainBounds &b) {
  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t ni = (b.ie + h) - (b.is - h) + 1;
  const int64_t nj = (b.je + h) - (b.js - h) + 1;
  return View3<int32_t>{static_cast<int32_t *>(p), b.is - h, b.js - h, 0, 1, ni, ni * nj};
}

// A length-npz column living in the first values of an arena slot.
inline View1<fv_real> to_arena_view1(void *p) {
  return View1<fv_real>{static_cast<fv_real *>(p), 0, 1};
}

inline bool covers1(const DeviceViewBase *v, int64_t lo, int64_t hi) {
  if (!v || v->rank < 1 || !v->data) return false;
  if (v->lower[0] > lo) return false;
  if (v->lower[0] + static_cast<int64_t>(v->extent[0]) - 1 < hi) return false;
  return true;
}

inline bool covers2(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
                    int64_t hi_j) {
  if (!v || v->rank < 2) return false;
  if (v->lower[0] > lo_i || v->lower[0] + static_cast<int64_t>(v->extent[0]) - 1 < hi_i)
    return false;
  if (v->lower[1] > lo_j || v->lower[1] + static_cast<int64_t>(v->extent[1]) - 1 < hi_j)
    return false;
  return true;
}

inline bool covers3(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
                    int64_t hi_j, int64_t lo_k, int64_t hi_k) {
  if (!covers2(v, lo_i, hi_i, lo_j, hi_j)) return false;
  if (v->rank < 3) return false;
  if (v->lower[2] > lo_k || v->lower[2] + static_cast<int64_t>(v->extent[2]) - 1 < hi_k)
    return false;
  return true;
}

// Overflow-safe uint64_t multiply-accumulate, mirroring the arena-sizing rigor
// used by wp_d_contract.cpp's bind_wp_d. Returns true on overflow.
inline bool mul_into(uint64_t &acc, uint64_t factor) {
  if (factor != 0 && acc > (UINT64_MAX / factor)) return true;
  acc *= factor;
  return false;
}

} // namespace fv3::gpu::dgrid::contract_views
