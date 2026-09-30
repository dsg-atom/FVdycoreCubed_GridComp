//===========================================================================
// WP-C fv_mapz REMAP_UV production-SHAPE GPU smoke test (Prompt 71, Stage
// 7). Same methodology as fv3_pt_delz_remap_prod_shape_smoke.cu (Prompt 55):
// builds device views with the SAME lower/stride layout the real
// fv3_gpu_field_registry.F90 registration produces for a production-shaped
// C32 tile (pe: 1-cell halo, (i,k,j) physical axis order, matching
// Atm%pe's real allocation; u: 3-cell halo, extra row (jsd..je+1), matching
// Atm%u; v: 3-cell halo, extra column (isd..ie+1), matching Atm%v), then
// launches the real production launcher (launch_remap_uv) over the full
// union-bounding-box 2D grid -- exactly the launch shape REMAP_UV uses in
// production.
//
// Correctness oracle: reuses the Prompt 71 Stage 2 iv=-1 fixtures
// (tests/fortran_baseline/fv_mapz/fixtures/case10..13*.txt), already
// validated bitwise against the authoritative Fortran map_scalar via the
// host-only test (fv3_wp_c_map_scalar_test). Each fixture's pe1/pe2 are
// reproduced EXACTLY at every column via a broadcast pe field plus an
// ak=pe2, bk=0 trick: the kernel computes
//   pe_src[k>=1] = 0.5*(pe(.,k,j-1)+pe(.,k,j))   -> pe1[k] when pe is
//                                                   broadcast in i/j
//   pe_tgt[k]    = ak(k) + 0.5*bk(k)*(...)       -> pe2[k] when bk==0
// reproducing the u-block's pe1/pe2 construction bitwise; the v-block's
// i-averaged construction reduces identically since pe is broadcast in i
// as well. u/v are broadcast with each fixture's q1, so every interior
// column of the production-shaped launch must reproduce that fixture's
// authoritative q2 (case10-13 were generated with GMAO boundary flags both
// .false. and q_min absent -- i.e. the exact no_min/iv=-1 path this kernel
// hardcodes). Run manually via srun + compute-sanitizer, like its sibling;
// not registered as a ctest.
//===========================================================================

#include "fv3_remap_uv.cuh"
#include "fv3_gpu_types.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
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

// Same fixture format as test_map_scalar.cpp's Fixture/load_fixture
// (tests/fortran_baseline/fv_mapz/fixtures/*.txt); duplicated rather than
// shared since neither file exposes a header-level parser and this is a
// standalone, GPU-only smoke binary like its PT/DELZ sibling.
struct Fixture {
  int km = 0, kn = 0, iv = 0, kord = 0;
  double q_min = 0.0;
  std::vector<double> pe1, pe2;       // km+1 / kn+1 entries
  std::vector<double> q1, q2_fortran; // km / kn entries
};

bool load_fixture(const std::string &path, Fixture &f) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open fixture: %s\n", path.c_str());
    return false;
  }
  std::string line;
  int section = 0; // 0=none, 1=pe block, 2=q block
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.rfind("# km=", 0) == 0) {
      std::istringstream iss(line.substr(1));
      std::string tok;
      while (iss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
        if (key == "km") f.km = std::atoi(val.c_str());
        else if (key == "kn") f.kn = std::atoi(val.c_str());
        else if (key == "iv") f.iv = std::atoi(val.c_str());
        else if (key == "kord") f.kord = std::atoi(val.c_str());
      }
      continue;
    }
    if (line.rfind("# q_min=", 0) == 0) {
      f.q_min = std::atof(line.substr(line.find('=') + 1).c_str());
      continue;
    }
    if (line.rfind("# k  pe1", 0) == 0) { section = 1; continue; }
    if (line.rfind("# k  q1_input", 0) == 0) { section = 2; continue; }
    if (line[0] == '#') { section = 0; continue; }
    std::istringstream iss(line);
    int k; double a, b;
    if (!(iss >> k >> a >> b)) continue;
    if (section == 1) { f.pe1.push_back(a); f.pe2.push_back(b); }
    else if (section == 2) { f.q1.push_back(a); f.q2_fortran.push_back(b); }
  }
  return f.km > 0 && f.kn > 0 &&
         static_cast<int>(f.pe1.size()) == f.km + 1 &&
         static_cast<int>(f.pe2.size()) == f.kn + 1 &&
         static_cast<int>(f.q1.size()) == f.km &&
         static_cast<int>(f.q2_fortran.size()) == f.kn;
}

// Column-major linear offset matching add()'s stride formula
// (fv3_gpu_field_registry.F90:460-472): stride(1)=1, stride(2)=ex(1),
// stride(3)=ex(1)*ex(2), axes taken in the order the registration lists
// them.
inline int64_t off3(int64_t a0, int64_t e0, int64_t a1, int64_t e1,
                    int64_t a2) {
  return a0 + e0 * (a1 + e1 * a2);
}

} // namespace

int main() {
  // Production-shaped C32, 1 tile/rank -- identical bounds/halo convention
  // to fv3_pt_delz_remap_prod_shape_smoke.cu.
  const int64_t is = 1, ie = 32, js = 1, je = 32;
  const int64_t ni = ie - is + 1, nj = je - js + 1;      // 32, 32
  const int64_t isd = is - 3, jsd = js - 3;              // 3-cell halo
  const int64_t nih = ie - isd + 1, njh = je - jsd + 1;  // 38, 38 (x/y halo)
  const int64_t ni2 = ni + 2, nj2 = nj + 2;              // pe 1-cell halo

  const char *names[4] = {
      "case10_wind_uniform_sign_iv_neg1", "case11_wind_top_sign_reversal",
      "case12_wind_bottom_sign_reversal", "case13_wind_interior_sign_reversal"};

  int nfail = 0;
  for (int c = 0; c < 4; ++c) {
    const std::string path =
        std::string(FV3_MAP_SCALAR_FIXTURE_DIR) + "/" + names[c] + ".txt";
    Fixture f;
    if (!load_fixture(path, f)) {
      nfail++;
      continue;
    }
    if (f.iv != -1 || f.km != f.kn) {
      std::fprintf(stderr, "[%s] unexpected fixture shape iv=%d km=%d kn=%d\n",
                   names[c], f.iv, f.km, f.kn);
      nfail++;
      continue;
    }
    const int km = f.km;
    const int64_t nk2 = km + 1;
    const double ptop = f.pe1[0];

    // u: (isd..ie) x (jsd..je+1) x (0..km-1); v: (isd..ie+1) x (jsd..je) x
    // (0..km-1) -- exactly Atm%u/Atm%v's real registered extents
    // (fv3_gpu_field_registry.F90:726-727).
    const int64_t nih_u = nih, njh_u = njh + 1;
    const int64_t nih_v = nih + 1, njh_v = njh;

    std::vector<fv_real> pe_h(ni2 * nk2 * nj2, fv_real(0));     // (i,k,j)
    std::vector<fv_real> ak_h(km + 1), bk_h(km + 1, fv_real(0));
    std::vector<fv_real> u_h(nih_u * njh_u * km, fv_real(0));   // (i,j,k)
    std::vector<fv_real> v_h(nih_v * njh_v * km, fv_real(0));   // (i,j,k)

    for (int k = 0; k <= km; ++k) ak_h[k] = fv_real(f.pe2[k]); // bk==0 trick
    for (int64_t jj = 0; jj < nj2; ++jj)
      for (int64_t ii = 0; ii < ni2; ++ii)
        for (int k = 0; k <= km; ++k)
          pe_h[off3(ii, ni2, k, nk2, jj)] = fv_real(f.pe1[k]);
    for (int64_t jj = 0; jj < njh_u; ++jj)
      for (int64_t ii = 0; ii < nih_u; ++ii)
        for (int k = 0; k < km; ++k)
          u_h[off3(ii, nih_u, jj, njh_u, k)] = fv_real(f.q1[k]);
    for (int64_t jj = 0; jj < njh_v; ++jj)
      for (int64_t ii = 0; ii < nih_v; ++ii)
        for (int k = 0; k < km; ++k)
          v_h[off3(ii, nih_v, jj, njh_v, k)] = fv_real(f.q1[k]);

    fv_real *d_pe, *d_ak, *d_bk, *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_pe, pe_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_ak, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_bk, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_u, u_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_v, v_h.size() * sizeof(fv_real)));

    CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), pe_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ak, ak_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, bk_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, u_h.data(), u_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v_h.data(), v_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));

    RemapUvArgs a{};
    a.pe = RiemViewC3{d_pe, is - 1, 0, js - 1, 1, ni2, ni2 * nk2};
    a.u = RiemViewC3{d_u, isd, jsd, 0, 1, nih_u, nih_u * njh_u};
    a.v = RiemViewC3{d_v, isd, jsd, 0, 1, nih_v, nih_v * njh_v};
    a.ak = PtDelzView1{d_ak, 0, 1};
    a.bk = PtDelzView1{d_bk, 0, 1};
    a.is = is; a.ie = ie; a.js = js; a.je = je;
    a.km = km;
    a.ptop = fv_real(ptop);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    fv3_gpu_error err{};
    fv3_gpu_status st = launch_remap_uv(a, s, &err);
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (st != FV3_GPU_SUCCESS) {
      std::fprintf(stderr, "[%s] launch failed: %s\n", names[c], err.message);
      nfail++;
      cudaStreamDestroy(s);
      cudaFree(d_pe); cudaFree(d_ak); cudaFree(d_bk); cudaFree(d_u);
      cudaFree(d_v);
      continue;
    }

    std::vector<fv_real> u_out(u_h.size()), v_out(v_h.size());
    CUDA_CHECK(cudaMemcpy(u_out.data(), d_u, u_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_out.data(), d_v, v_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));

    // Every interior column of BOTH u's domain (is..ie x js..je+1) and v's
    // domain (is..ie+1 x js..je) must reproduce this fixture's authoritative
    // q2 -- broadcast input, so broadcast output.
    double max_rel_u = 0, max_rel_v = 0, max_abs_u = 0, max_abs_v = 0;
    auto upd = [](double val, double exp, double &max_abs, double &max_rel) {
      const double d = std::fabs(val - exp);
      max_abs = std::max(max_abs, d);
      max_rel = std::max(max_rel, d / std::max(1.0, std::fabs(exp)));
    };
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie; ++i)
        for (int k = 0; k < km; ++k) {
          const int64_t off = off3(i - isd, nih_u, j - jsd, njh_u, k);
          upd(double(u_out[off]), f.q2_fortran[k], max_abs_u, max_rel_u);
        }
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie + 1; ++i)
        for (int k = 0; k < km; ++k) {
          const int64_t off = off3(i - isd, nih_v, j - jsd, njh_v, k);
          upd(double(v_out[off]), f.q2_fortran[k], max_abs_v, max_rel_v);
        }
    const double tol = 1e-9; // tight, not bitwise -- same rationale as the
                             // PT/DELZ prod-shape smoke test (host libm vs
                             // Fortran intrinsics evaluate transcendentals
                             // independently inside scalar_profile_column).
    const bool ok = max_rel_u < tol && max_rel_v < tol;
    std::printf("[%s] u_cols=%ldx%ld v_cols=%ldx%ld u(abs=%.3e,rel=%.3e) "
                "v(abs=%.3e,rel=%.3e) OK=%d\n",
                names[c], (long)ni, (long)(nj + 1), (long)(ni + 1), (long)nj,
                max_abs_u, max_rel_u, max_abs_v, max_rel_v, ok ? 1 : 0);
    if (!ok) nfail++;

    // Stage 4/Prompt 68-pattern unit check: re-run the SAME case through the
    // device-workspace path (RemapUvArgs::workspace set) and require
    // bitwise-identical output vs. the local-scratch path above -- both
    // branches execute identical per-thread arithmetic, only the scratch
    // backing store differs, so any difference would indicate an indexing
    // bug in the new workspace path, not a numerical-formula change. This
    // is also the executable proof (not just an inline estimate) that the
    // workspace-path column-count/stride bookkeeping matches
    // remap_uv_scalars_per_column/remap_uv_columns at production shape.
    {
      // u/v are mutated in place by the first run; restore the original
      // broadcast inputs before re-running. pe/ak/bk are read-only.
      CUDA_CHECK(cudaMemcpy(d_u, u_h.data(), u_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_v, v_h.data(), v_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));

      const uint64_t ws_scalars_per_col = remap_uv_scalars_per_column(km);
      const uint64_t ws_columns = remap_uv_columns(is, ie, js, je);
      const uint64_t ws_bytes = ws_columns * ws_scalars_per_col * sizeof(fv_real);
      fv_real *d_workspace = nullptr;
      CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));

      RemapUvArgs a2 = a;
      a2.workspace = d_workspace;
      a2.workspace_scalars_per_column = ws_scalars_per_col;
      a2.workspace_bytes = ws_bytes;

      cudaStream_t s2;
      CUDA_CHECK(cudaStreamCreate(&s2));
      fv3_gpu_error err2{};
      fv3_gpu_status st2 = launch_remap_uv(a2, s2, &err2);
      CUDA_CHECK(cudaStreamSynchronize(s2));
      bool ws_ok = (st2 == FV3_GPU_SUCCESS);
      if (!ws_ok) {
        std::fprintf(stderr, "[%s] workspace-path launch failed: %s\n",
                     names[c], err2.message);
        nfail++;
      } else {
        std::vector<fv_real> u_out2(u_h.size()), v_out2(v_h.size());
        CUDA_CHECK(cudaMemcpy(u_out2.data(), d_u, u_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_out2.data(), d_v, v_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        bool bitwise = (u_out2 == u_out) && (v_out2 == v_out);
        std::printf("[%s] workspace-path bitwise-vs-local-scratch OK=%d "
                    "(ws_bytes=%llu)\n",
                    names[c], bitwise ? 1 : 0,
                    (unsigned long long)ws_bytes);
        if (!bitwise) nfail++;
      }
      cudaStreamDestroy(s2);
      cudaFree(d_workspace);
    }

    cudaFree(d_pe); cudaFree(d_ak); cudaFree(d_bk); cudaFree(d_u);
    cudaFree(d_v);
    cudaStreamDestroy(s);
  }

  std::printf("REMAP_UV_PROD_SHAPE_GPU_SMOKE: %d/4 cases failed\n", nfail);
  return nfail == 0 ? 0 : 1;
}
