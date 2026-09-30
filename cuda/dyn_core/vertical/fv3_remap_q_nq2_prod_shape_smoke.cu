//===========================================================================
// WP-C fv_mapz REMAP_Q nq=2 production-SHAPE GPU smoke test (Prompt 86B,
// Stage 7; kord schedule corrected to the actual GEOS-integrated runtime
// value by Prompt 87 -- see fv3_remap_q.cuh's header comment). Extends
// fv3_remap_q_prod_shape_smoke.cu (Prompt 85, nq=1/kord=8-only) to the real
// production nq=2 schedule: sphum (kord=9) + cld_amt (kord=9) -- both
// tracers, NOT the (8,9) split Prompts 85/86A assumed -- on ONE shared
// pe1/pe2 grid, exercised through the SAME device-side mechanism
// launch_remap_q_ctx now uses (Prompt 86B Stage 4, Prompt 87 Stage 2): two
// sequential launch_remap_q calls on the same stream, second call's
// RemapQArgs::q offset by exactly remap_q_buffer_elements(isd,ied,jsd,jed,
// km) -- the two-tracer-plane stride -- and RemapQArgs::kord set to 9 for
// both launches. This test builds its own device q buffer directly
// (bypassing fv3_gpu_remap_q/DispatchParams, which need a live
// fv3_gpu_context) but performs the transfers/launches in the exact
// production shape: ONE contiguous H2D covering BOTH tracer planes, two
// kernel launches, ONE contiguous D2H covering both planes (Stage 3
// hard-fail requirement).
//
// Correctness oracle: reuses test_remap_q_nq2_scope.cpp's synthetic
// production-shaped sphum-like and cld_amt-like profiles (both kord=9,
// Prompt 87) on the SAME shared pe1/pe2 grid, run through the identical
// host map_scalar_column path (already GPU-validated for kord=9 via
// fv3_remap_uv_prod_shape_smoke.cu and, for the retained kord=8 regression
// path, via fv3_remap_q_prod_shape_smoke.cu) as the authoritative oracle.
// Each fixture profile is broadcast over the full isd:ied x jsd:jed C32
// buffer; every interior column of the production-shaped launch must
// reproduce the host oracle's q2 for its own tracer. Reports each tracer
// separately (bitwise-first, else round-off with tolerance), plus mass
// conservation, positivity, and NaN/Inf counts per tracer. Run manually via
// srun (+ optionally compute-sanitizer), like its siblings; not registered
// as a ctest.
//===========================================================================

#include "fv3_remap_q.cuh"
#include "fv3_map_scalar_column.hpp"
#include "fv3_gpu_types.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::vertical;

namespace {

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(_e));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

// Column-major linear offset matching add()'s stride formula
// (fv3_gpu_field_registry.F90:460-472), identical to
// fv3_remap_q_prod_shape_smoke.cu's off3.
inline int64_t off3(int64_t a0, int64_t e0, int64_t a1, int64_t e1,
                    int64_t a2) {
  return a0 + e0 * (a1 + e1 * a2);
}

// Host oracle -- identical call shape to test_remap_q_nq2_scope.cpp's
// run_tracer, duplicated rather than shared (standalone GPU smoke binary
// convention, matching fv3_remap_q_prod_shape_smoke.cu).
std::vector<double> host_oracle(const std::vector<double> &pe1,
                                const std::vector<double> &pe2,
                                const std::vector<double> &q1, int kord,
                                double q_min) {
  const int km = static_cast<int>(q1.size());
  std::vector<double> dpe1(km), dpe2(km);
  for (int k = 0; k < km; ++k) dpe1[k] = pe1[k + 1] - pe1[k];
  for (int k = 0; k < km; ++k) dpe2[k] = pe2[k + 1] - pe2[k];

  std::vector<double> q2(km, 0.0);
  std::vector<double> c2(km), c3(km), c4(km), gam(km), qs(km + 1);
  std::vector<char> extm(km);

  Column<const double> pe1_c{pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
  Column<const double> pe2_c{pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
  Column<const double> q1_c{q1.data(), 1};
  Column<double> q2_c{q2.data(), 1};

  map_scalar_column<double>(pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c,
                            c2.data(), c3.data(), c4.data(), gam.data(),
                            qs.data(), reinterpret_cast<bool *>(extm.data()),
                            km, km, /*iv=*/0, q_min, /*gmao_bot=*/false,
                            /*gmao_top=*/false, kord);
  return q2;
}

} // namespace

int main() {
  // Production-shaped C32, 1 tile/rank -- identical bounds/halo convention
  // to fv3_remap_q_prod_shape_smoke.cu.
  const int64_t is = 1, ie = 32, js = 1, je = 32;
  const int64_t isd = is - 3, ied = ie + 3, jsd = js - 3, jed = je + 3; // ng=3
  const int64_t ni2 = (ie - is + 1) + 2, nj2 = (je - js + 1) + 2; // pe 1-cell halo
  const int64_t nx_d = ied - isd + 1, ny_d = jed - jsd + 1; // 38, 38

  // Shared pressure grid, identical to test_remap_q_nq2_scope.cpp.
  const int km = 10;
  std::vector<double> pe1(km + 1), pe2(km + 1);
  for (int k = 0; k <= km; ++k) {
    pe1[k] = 1000.0 + 900.0 * k;
    pe2[k] = 1000.0 + 900.0 * k + 15.0 * std::sin(0.6 * k);
  }
  pe2[0] = pe1[0];
  pe2[km] = pe1[km];

  std::vector<double> sphum_like(km);
  for (int k = 0; k < km; ++k) sphum_like[k] = 1.0e-2 * std::exp(-0.35 * k);

  std::vector<double> cld_amt_like(km);
  for (int k = 0; k < km; ++k)
    cld_amt_like[k] = (k >= 3 && k <= 6) ? (0.7 + 0.05 * (k - 3)) : 0.05;

  const double q_min = 0.0;
  // Prompt 87: actual production schedule is kord=9 for BOTH tracers
  // (FV_StateMod.F90:563 unconditionally forces kord_tr=9; the (8,9) split
  // Prompts 85/86A assumed never occurs -- see p87 doc Stage 0).
  const std::vector<double> sphum_oracle = host_oracle(pe1, pe2, sphum_like, 9, q_min);
  const std::vector<double> cld_oracle = host_oracle(pe1, pe2, cld_amt_like, 9, q_min);

  const int64_t nk2 = km + 1;
  const double ptop = pe1[0];
  const uint64_t stride = remap_q_buffer_elements(isd, ied, jsd, jed, km);

  std::vector<fv_real> pe_h(ni2 * nk2 * nj2, fv_real(0));   // (i,k,j)
  std::vector<fv_real> ak_h(km + 1), bk_h(km + 1, fv_real(0));
  std::vector<fv_real> q_h(stride * 2, fv_real(0));         // nq=2 planes, contiguous

  for (int k = 0; k <= km; ++k) ak_h[k] = fv_real(pe2[k]); // bk==0 trick
  for (int64_t jj = 0; jj < nj2; ++jj)
    for (int64_t ii = 0; ii < ni2; ++ii)
      for (int k = 0; k <= km; ++k)
        pe_h[off3(ii, ni2, k, nk2, jj)] = fv_real(pe1[k]);
  for (int64_t jj = 0; jj < ny_d; ++jj)
    for (int64_t ii = 0; ii < nx_d; ++ii)
      for (int k = 0; k < km; ++k) {
        q_h[off3(ii, nx_d, jj, ny_d, k)] = fv_real(sphum_like[k]);
        q_h[stride + off3(ii, nx_d, jj, ny_d, k)] = fv_real(cld_amt_like[k]);
      }

  fv_real *d_pe, *d_ak, *d_bk, *d_q;
  CUDA_CHECK(cudaMalloc(&d_pe, pe_h.size() * sizeof(fv_real)));
  CUDA_CHECK(cudaMalloc(&d_ak, (km + 1) * sizeof(fv_real)));
  CUDA_CHECK(cudaMalloc(&d_bk, (km + 1) * sizeof(fv_real)));
  CUDA_CHECK(cudaMalloc(&d_q, q_h.size() * sizeof(fv_real)));

  CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), pe_h.size() * sizeof(fv_real),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ak, ak_h.data(), (km + 1) * sizeof(fv_real),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bk, bk_h.data(), (km + 1) * sizeof(fv_real),
                        cudaMemcpyHostToDevice));
  // ONE contiguous H2D covering BOTH tracer planes (Stage 3 hard-fail gate).
  CUDA_CHECK(cudaMemcpy(d_q, q_h.data(), q_h.size() * sizeof(fv_real),
                        cudaMemcpyHostToDevice));
  int h2d_calls = 1;

  RemapQArgs a{};
  a.pe = RiemViewC3{d_pe, is - 1, 0, js - 1, 1, ni2, ni2 * nk2};
  a.ak = PtDelzView1{d_ak, 0, 1};
  a.bk = PtDelzView1{d_bk, 0, 1};
  a.is = is; a.ie = ie; a.js = js; a.je = je;
  a.isd = isd; a.ied = ied; a.jsd = jsd; a.jed = jed;
  a.km = km;
  a.ptop = fv_real(ptop);

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // Two sequential single-tracer launches on the SAME stream/workspace --
  // exactly launch_remap_q_ctx's Stage 4 mechanism. Prompt 87: both
  // launches use kord=9 (actual production schedule).
  a.q = d_q;
  a.kord = 9;
  fv3_gpu_error err1{};
  fv3_gpu_status st1 = launch_remap_q(a, s, &err1);

  a.q = d_q + static_cast<int64_t>(stride);
  a.kord = 9;
  fv3_gpu_error err2{};
  fv3_gpu_status st2 = launch_remap_q(a, s, &err2);

  CUDA_CHECK(cudaStreamSynchronize(s));

  int nfail = 0;
  if (st1 != FV3_GPU_SUCCESS) {
    std::fprintf(stderr, "sphum launch failed: %s\n", err1.message);
    ++nfail;
  }
  if (st2 != FV3_GPU_SUCCESS) {
    std::fprintf(stderr, "cld_amt launch failed: %s\n", err2.message);
    ++nfail;
  }

  std::vector<fv_real> q_out(q_h.size());
  // ONE contiguous D2H covering BOTH tracer planes (Stage 3 hard-fail gate).
  CUDA_CHECK(cudaMemcpy(q_out.data(), d_q, q_h.size() * sizeof(fv_real),
                        cudaMemcpyDeviceToHost));
  int d2h_calls = 1;

  struct TracerCheck {
    const char *name;
    int kord;
    uint64_t plane_offset;
    const std::vector<double> *oracle;
    const std::vector<double> *input;
  };
  TracerCheck checks[2] = {
      {"sphum(kord=9)", 9, 0, &sphum_oracle, &sphum_like},
      {"cld_amt(kord=9)", 9, stride, &cld_oracle, &cld_amt_like},
  };

  for (const auto &c : checks) {
    double max_abs = 0, max_rel = 0;
    bool bitwise = true;
    int nan_count = 0;
    bool any_negative = false;
    double mass_in = 0.0, mass_out = 0.0;
    for (int k = 0; k < km; ++k)
      mass_in += (*c.input)[k] * (pe1[k + 1] - pe1[k]);

    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i)
        for (int k = 0; k < km; ++k) {
          const int64_t off = c.plane_offset +
                              off3(i - isd, nx_d, j - jsd, ny_d, k);
          const double val = double(q_out[off]);
          const double exp = (*c.oracle)[k];
          if (val != exp) bitwise = false;
          const double d = std::fabs(val - exp);
          max_abs = std::max(max_abs, d);
          max_rel = std::max(max_rel, d / std::max(1.0, std::fabs(exp)));
          if (std::isnan(val) || std::isinf(val)) ++nan_count;
          if (val < 0.0) any_negative = true;
          if (j == js && i == is) mass_out += val * (pe2[k + 1] - pe2[k]);
        }

    const double tol = 1e-9; // round-off tolerance (host libm vs GPU device
                             // math may not be bit-identical even with
                             // identical formulas -- same rationale as
                             // fv3_remap_q_prod_shape_smoke.cu).
    const bool roundoff_ok = max_rel < tol;
    const double mass_rel = std::fabs(mass_out - mass_in) /
                            std::max(std::fabs(mass_in), 1.0e-30);
    const bool mass_ok = mass_rel < 1.0e-9;
    const bool ok = roundoff_ok && nan_count == 0 && !any_negative && mass_ok;

    std::printf("[%s] %s abs=%.3e rel=%.3e mass_rel=%.3e nan=%d neg=%s %s\n",
                c.name, bitwise ? "BITWISE" : "ROUND-OFF", max_abs, max_rel,
                mass_rel, nan_count, any_negative ? "yes" : "no",
                ok ? "PASS" : "FAIL");
    if (!ok) ++nfail;
  }

  std::printf("H2D_CALLS=%d D2H_CALLS=%d\n", h2d_calls, d2h_calls);
  std::printf("REMAP_Q_NQ2_PROD_SHAPE_GPU_SMOKE: %d checks failed\n", nfail);

  cudaFree(d_pe); cudaFree(d_ak); cudaFree(d_bk); cudaFree(d_q);
  cudaStreamDestroy(s);
  return nfail == 0 ? 0 : 1;
}
