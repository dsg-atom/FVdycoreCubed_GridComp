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

  std::vector<T> ones(25, T(1)), impulse(25, T(0));
  std::vector<T> d2(25), fdx(25), fdy(25), tend(25);
  impulse[12] = T(2);
  auto done = desc(ones, 5, 5);
  auto dimp = desc(impulse, 5, 5); auto dd2 = desc(d2, 5, 5);
  auto dfdx = desc(fdx, 5, 5); auto dfdy = desc(fdy, 5, 5);
  auto dtend = desc(tend, 5, 5);
  fv3::gpu::dgrid::deln_flux_stage<T>(
      {2, 2, 2, 2}, 0, FieldView<const T>(dimp), FieldView<const T>(done),
      FieldView<const T>(done), T(3), FieldView<T>(dd2), FieldView<T>(dfdx),
      FieldView<T>(dfdy));
  fv3::gpu::dgrid::flux_divergence<T>(
      {2, 2, 2, 2}, 0, FieldView<const T>(dfdx), FieldView<const T>(dfdy),
      FieldView<const T>(done), FieldView<T>(dtend));
  assert(tend[12] == T(-24));

  // Test PPM constants are valid: q(i+0.5) = p1*(q(i-1)+q(i)) + p2*(q(i-2)+q(i+1))
  // For uniform q, p1*2*q + p2*2*q = q requires 2*p1 + 2*p2 = 1, i.e. p1+p2 = 0.5
  // p1 = 7/12, p2 = -1/12, so p1+p2 = 6/12 = 0.5 ✓
  static_assert(fv3::gpu::dgrid::ppm::p1 + fv3::gpu::dgrid::ppm::p2 == 0.5,
                "PPM constants should satisfy p1+p2=0.5 for uniform field reconstruction");

  // Test kinetic energy computation
  std::vector<T> ub_ke(9, T(2)), vb_ke(9, T(3)), ke_out(9, T(0));
  auto dub = desc(ub_ke, 3, 3); auto dvb = desc(vb_ke, 3, 3);
  auto dke = desc(ke_out, 3, 3);
  fv3::gpu::dgrid::compute_kinetic_energy<T>(
      {1, 1, 1, 1}, 0, T(1.0),
      fv3::gpu::dgrid::FieldView<const T>(dub),
      fv3::gpu::dgrid::FieldView<const T>(dvb),
      fv3::gpu::dgrid::FieldView<T>(dke));
  assert(ke_out[4] == T(3)); // 0.5 * 1.0 * 2.0 * 3.0 = 3.0

  // Test update_d_winds
  std::vector<T> vt_wind(16, T(1)), ut_wind(16, T(2)), ke_wind(16, T(0.5));
  std::vector<T> fy_wind(16, T(0.1)), fx_wind(16, T(0.2));
  std::vector<T> u_out(16, T(0)), v_out(16, T(0));
  auto dvt_w = desc(vt_wind, 4, 4); auto dut_w = desc(ut_wind, 4, 4);
  auto dke_w = desc(ke_wind, 4, 4); auto dfy_w = desc(fy_wind, 4, 4);
  auto dfx_w = desc(fx_wind, 4, 4); auto du_out = desc(u_out, 4, 4);
  auto dv_out = desc(v_out, 4, 4);
  fv3::gpu::dgrid::update_d_winds<T>(
      {1, 2, 1, 2}, 0,
      fv3::gpu::dgrid::FieldView<const T>(dvt_w),
      fv3::gpu::dgrid::FieldView<const T>(dut_w),
      fv3::gpu::dgrid::FieldView<const T>(dke_w),
      fv3::gpu::dgrid::FieldView<const T>(dfy_w),
      fv3::gpu::dgrid::FieldView<const T>(dfx_w),
      fv3::gpu::dgrid::FieldView<T>(du_out),
      fv3::gpu::dgrid::FieldView<T>(dv_out));
  // u = vt + ke - ke(i+1) + fy = 1 + 0.5 - 0.5 + 0.1 = 1.1
  // v = ut + ke - ke(j+1) - fx = 2 + 0.5 - 0.5 - 0.2 = 1.8
  assert(std::abs(u_out[5] - T(1.1)) < T(1e-6));
  assert(std::abs(v_out[5] - T(1.8)) < T(1e-6));

  // Test damping increment computation
  std::vector<T> vort_damp(16, T(0));
  vort_damp[5] = T(10); vort_damp[6] = T(7); vort_damp[9] = T(4);
  std::vector<T> ub_damp(16, T(0)), vb_damp(16, T(0));
  auto dvort_d = desc(vort_damp, 4, 4);
  auto dub_d = desc(ub_damp, 4, 4); auto dvb_d = desc(vb_damp, 4, 4);
  fv3::gpu::dgrid::compute_damping_increments<T>(
      {1, 1, 1, 1}, 0,
      fv3::gpu::dgrid::FieldView<const T>(dvort_d),
      fv3::gpu::dgrid::FieldView<T>(dub_d),
      fv3::gpu::dgrid::FieldView<T>(dvb_d));
  // ub[1,1] = vort[1,1] - vort[2,1] = 10 - 7 = 3
  // vb[1,1] = vort[1,1] - vort[1,2] = 10 - 4 = 6
  assert(ub_damp[5] == T(3));
  assert(vb_damp[5] == T(6));

  return 0;
}
