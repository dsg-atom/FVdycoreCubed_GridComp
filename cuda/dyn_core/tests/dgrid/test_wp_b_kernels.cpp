#include "dgrid/wp_b_kernels.hpp"

#include <cassert>
#include <cstring>
#include <vector>

using fv3::gpu::accumulator_real;
using fv3::gpu::dgrid::FieldView;
using fv3::gpu::dgrid::HorizontalBounds;

template <class T>
fv3_gpu_field_desc desc(std::vector<T> &a, int ni, int nj, int nk = 1,
                        int lo_i = 0, int lo_j = 0) {
  fv3_gpu_field_desc d{};
  d.abi_version = FV3_GPU_ABI_VERSION;
  d.struct_bytes = sizeof(d);
  d.base = a.data();
  d.rank = nk == 1 ? 2 : 3;
  d.lower[0] = lo_i; d.lower[1] = lo_j; d.lower[2] = 0;
  d.extent[0] = ni; d.extent[1] = nj; d.extent[2] = nk;
  d.stride_elements[0] = 1; d.stride_elements[1] = ni;
  d.stride_elements[2] = ni * nj;
  return d;
}

int main() {
  using T = fv3::gpu::fv_real;
  std::vector<T> ut(12, T(0)), vt(12, T(0)), metric(12, T(1));
  std::vector<T> crx(12), cry(12), xfx(12), yfx(12);
  auto du = desc(ut, 4, 3); auto dv = desc(vt, 4, 3);
  auto dm = desc(metric, 4, 3); auto dcx = desc(crx, 4, 3);
  auto dcy = desc(cry, 4, 3); auto dxf = desc(xfx, 4, 3);
  auto dyf = desc(yfx, 4, 3);
  ut[1] = T(2); ut[2] = T(-3); vt[4] = T(4); vt[8] = T(-5);
  fv3::gpu::dgrid::prepare_flux_courant<T>(
      {1, 1, 0, 1}, 0, T(0.5), FieldView<const T>(du),
      FieldView<const T>(dv), FieldView<const T>(dm), FieldView<const T>(dm),
      FieldView<const T>(dm), FieldView<const T>(dm), FieldView<const T>(dm),
      FieldView<const T>(dm), FieldView<T>(dcx), FieldView<T>(dcy),
      FieldView<T>(dxf), FieldView<T>(dyf));
  assert(crx[1] == T(1));
  assert(crx[2] == T(-1.5));

  std::vector<T> delp{T(5), T(15)}, pt{T(3), T(7)}, w{T(5), T(9)};
  std::vector<T> ak{T(0), T(1000), T(2000)}, bk(3, T(0));
  auto dd = desc(delp, 1, 1, 2); auto dp = desc(pt, 1, 1, 2);
  auto dw = desc(w, 1, 1, 2); auto da = desc(ak, 3, 1);
  auto db = desc(bk, 3, 1);
  fv3::gpu::dgrid::mix_dp_columns<T>(
      {0, 0, 0, 0}, 2, false, FieldView<const T>(da),
      FieldView<const T>(db), FieldView<T>(dd), FieldView<T>(dp),
      FieldView<T>(dw));
  assert(delp[0] == T(10));
  assert(delp[1] == T(10));
  assert(pt[0] == T(5));
  assert(w[0] == T(7));

  std::vector<T> uu(9, T(0)), vv(9, T(0)), vort(9, T(0));
  uu[1] = T(4); uu[4] = T(1); vv[3] = T(2); vv[4] = T(5);
  auto duu = desc(uu, 3, 3); auto dvv = desc(vv, 3, 3);
  auto dvo = desc(vort, 3, 3); auto dm3 = desc(metric, 4, 3);
  fv3::gpu::dgrid::compute_relative_vorticity<T>(
      {1, 1, 1, 1}, 0, FieldView<const T>(duu), FieldView<const T>(dvv),
      FieldView<const T>(dm3), FieldView<const T>(dm3),
      FieldView<const T>(dm3), FieldView<T>(dvo));
  assert(vort[4] == T(-4));
  return 0;
}
