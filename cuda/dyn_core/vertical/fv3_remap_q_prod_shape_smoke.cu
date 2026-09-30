//===========================================================================
// WP-C fv_mapz REMAP_Q production-SHAPE GPU smoke test (Prompt 85, Stage
// 9). Same methodology as fv3_remap_uv_prod_shape_smoke.cu (Prompt 71) and
// fv3_pt_delz_remap_prod_shape_smoke.cu (Prompt 55): builds device views
// with the SAME lower/stride layout the real fv3_gpu_field_registry.F90
// registration produces for a production-shaped C32 tile (pe: 1-cell halo,
// (i,k,j) physical axis order, matching Atm%pe's real allocation), then
// launches the real production launcher (launch_remap_q) over the plain
// interior is:ie x js:je grid -- exactly the launch shape REMAP_Q uses in
// production. Unlike pe/ak/bk, q has no FieldRegistry view at all (see
// fv3_remap_q.cuh's header comment) -- this test allocates its OWN device
// buffer sized to the full isd:ied x jsd:jed x km memory domain, exactly
// mirroring what fv3_gpu_remap_q's persistent context buffer would hold,
// and populates/reads it with a single contiguous cudaMemcpy each way (the
// same one-H2D/one-D2H transfer shape Stage 3/4 mandate for production).
//
// Correctness oracle: reuses the Prompt 84B mapn_tracer (REMAP_Q) fixtures
// (tests/fortran_baseline/fv_mapz/fixtures/case14..22_tracer_*.txt),
// already validated bitwise against the authoritative Fortran mapn_tracer
// via the host-only test (fv3_wp_c_remap_q_tracer_test). Each fixture's
// pe1/pe2 are reproduced EXACTLY at every column via a broadcast pe field
// plus an ak=pe2, bk=0 trick, identical to the REMAP_UV smoke test's
// construction. q is broadcast with each fixture's q1 over the FULL
// isd:ied x jsd:jed buffer (halo cells are moved by the transfer but never
// read/written by the kernel, matching production), so every interior
// column of the production-shaped launch must reproduce that fixture's
// authoritative q2. Run manually via srun + compute-sanitizer, like its
// siblings; not registered as a ctest.
//===========================================================================

#include "fv3_remap_q.cuh"
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

// Same fixture format as test_remap_q_tracer.cpp's Fixture/load_fixture
// (tests/fortran_baseline/fv_mapz/fixtures/case14..22_tracer_*.txt),
// duplicated rather than shared since neither file exposes a header-level
// parser and this is a standalone, GPU-only smoke binary like its PT/DELZ
// and REMAP_UV siblings.
struct Fixture {
  int km = 0, kord = 0;
  double q_min = 0.0;
  std::vector<double> pe1, pe2;       // km+1 entries
  std::vector<double> q1, q2_fortran; // km entries
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
  return f.km > 0 &&
         static_cast<int>(f.pe1.size()) == f.km + 1 &&
         static_cast<int>(f.pe2.size()) == f.km + 1 &&
         static_cast<int>(f.q1.size()) == f.km &&
         static_cast<int>(f.q2_fortran.size()) == f.km;
}

// Column-major linear offset matching add()'s stride formula
// (fv3_gpu_field_registry.F90:460-472): stride(1)=1, stride(2)=ex(1),
// stride(3)=ex(1)*ex(2), axes taken in the order the registration lists
// them. Used for pe's (i,k,j) layout.
inline int64_t off3(int64_t a0, int64_t e0, int64_t a1, int64_t e1,
                    int64_t a2) {
  return a0 + e0 * (a1 + e1 * a2);
}

} // namespace

int main() {
  // Production-shaped C32, 1 tile/rank -- identical bounds/halo convention
  // to fv3_pt_delz_remap_prod_shape_smoke.cu / fv3_remap_uv_prod_shape_smoke.cu.
  const int64_t is = 1, ie = 32, js = 1, je = 32;
  const int64_t isd = is - 3, ied = ie + 3, jsd = js - 3, jed = je + 3; // ng=3
  const int64_t ni2 = (ie - is + 1) + 2, nj2 = (je - js + 1) + 2; // pe 1-cell halo
  const int64_t nx_d = ied - isd + 1, ny_d = jed - jsd + 1; // 38, 38

  const char *names[9] = {
      "case14_tracer_constant", "case15_tracer_smooth_gradient",
      "case16_tracer_sharp_front", "case17_tracer_local_max",
      "case18_tracer_local_min", "case19_tracer_near_qmin",
      "case20_tracer_qmin_clipping", "case21_tracer_boundary_sensitive",
      "case22_tracer_c512_like_profile"};

  int nfail = 0;
  for (int c = 0; c < 9; ++c) {
    const std::string path =
        std::string(FV3_MAP_SCALAR_FIXTURE_DIR) + "/" + names[c] + ".txt";
    Fixture f;
    if (!load_fixture(path, f)) {
      nfail++;
      continue;
    }
    if (f.kord != 8) {
      std::fprintf(stderr, "[%s] unexpected fixture kord=%d (expected 8)\n",
                   names[c], f.kord);
      nfail++;
      continue;
    }
    const int km = f.km;
    const int64_t nk2 = km + 1;
    const double ptop = f.pe1[0];

    std::vector<fv_real> pe_h(ni2 * nk2 * nj2, fv_real(0));   // (i,k,j)
    std::vector<fv_real> ak_h(km + 1), bk_h(km + 1, fv_real(0));
    std::vector<fv_real> q_h(nx_d * ny_d * km, fv_real(0));   // (i,j,k)

    for (int k = 0; k <= km; ++k) ak_h[k] = fv_real(f.pe2[k]); // bk==0 trick
    for (int64_t jj = 0; jj < nj2; ++jj)
      for (int64_t ii = 0; ii < ni2; ++ii)
        for (int k = 0; k <= km; ++k)
          pe_h[off3(ii, ni2, k, nk2, jj)] = fv_real(f.pe1[k]);
    for (int64_t jj = 0; jj < ny_d; ++jj)
      for (int64_t ii = 0; ii < nx_d; ++ii)
        for (int k = 0; k < km; ++k)
          q_h[off3(ii, nx_d, jj, ny_d, k)] = fv_real(f.q1[k]);

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
    // ONE contiguous H2D, matching production's transfer shape (Stage 3/4).
    CUDA_CHECK(cudaMemcpy(d_q, q_h.data(), q_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));

    RemapQArgs a{};
    a.pe = RiemViewC3{d_pe, is - 1, 0, js - 1, 1, ni2, ni2 * nk2};
    a.ak = PtDelzView1{d_ak, 0, 1};
    a.bk = PtDelzView1{d_bk, 0, 1};
    a.q = d_q;
    a.is = is; a.ie = ie; a.js = js; a.je = je;
    a.isd = isd; a.ied = ied; a.jsd = jsd; a.jed = jed;
    a.km = km;
    a.ptop = fv_real(ptop);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    fv3_gpu_error err{};
    fv3_gpu_status st = launch_remap_q(a, s, &err);
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (st != FV3_GPU_SUCCESS) {
      std::fprintf(stderr, "[%s] launch failed: %s\n", names[c], err.message);
      nfail++;
      cudaStreamDestroy(s);
      cudaFree(d_pe); cudaFree(d_ak); cudaFree(d_bk); cudaFree(d_q);
      continue;
    }

    std::vector<fv_real> q_out(q_h.size());
    // ONE contiguous D2H, matching production's transfer shape (Stage 3/4).
    CUDA_CHECK(cudaMemcpy(q_out.data(), d_q, q_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));

    // Every interior column (is..ie x js..je) must reproduce this fixture's
    // authoritative q2 -- broadcast input, so broadcast output.
    double max_rel = 0, max_abs = 0;
    auto upd = [](double val, double exp, double &max_abs, double &max_rel) {
      const double d = std::fabs(val - exp);
      max_abs = std::max(max_abs, d);
      max_rel = std::max(max_rel, d / std::max(1.0, std::fabs(exp)));
    };
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i)
        for (int k = 0; k < km; ++k) {
          const int64_t off = off3(i - isd, nx_d, j - jsd, ny_d, k);
          upd(double(q_out[off]), f.q2_fortran[k], max_abs, max_rel);
        }
    const double tol = 1e-9; // tight, not bitwise -- same rationale as the
                             // PT/DELZ/REMAP_UV prod-shape smoke tests (host
                             // libm vs Fortran intrinsics evaluate
                             // transcendentals independently inside
                             // map_scalar_column).
    const bool ok = max_rel < tol;
    std::printf("[%s] cols=%ldx%ld q(abs=%.3e,rel=%.3e) OK=%d\n", names[c],
                (long)(ie - is + 1), (long)(je - js + 1), max_abs, max_rel,
                ok ? 1 : 0);
    if (!ok) nfail++;

    // Stage 4/Prompt 68-pattern unit check: re-run the SAME case through the
    // device-workspace path (RemapQArgs::workspace set) and require
    // bitwise-identical output vs. the local-scratch path above -- both
    // branches execute identical per-thread arithmetic, only the scratch
    // backing store differs, so any difference would indicate an indexing
    // bug in the new workspace path, not a numerical-formula change. This
    // is also the executable proof (not just an inline estimate) that the
    // workspace-path column-count/stride bookkeeping matches
    // remap_q_scalars_per_column/remap_q_columns at production shape.
    {
      // q is mutated in place by the first run; restore the original
      // broadcast input before re-running. pe/ak/bk are read-only.
      CUDA_CHECK(cudaMemcpy(d_q, q_h.data(), q_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));

      const uint64_t ws_scalars_per_col = remap_q_scalars_per_column(km);
      const uint64_t ws_columns = remap_q_columns(is, ie, js, je);
      const uint64_t ws_bytes = ws_columns * ws_scalars_per_col * sizeof(fv_real);
      fv_real *d_workspace = nullptr;
      CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));

      RemapQArgs a2 = a;
      a2.workspace = d_workspace;
      a2.workspace_scalars_per_column = ws_scalars_per_col;
      a2.workspace_bytes = ws_bytes;

      cudaStream_t s2;
      CUDA_CHECK(cudaStreamCreate(&s2));
      fv3_gpu_error err2{};
      fv3_gpu_status st2 = launch_remap_q(a2, s2, &err2);
      CUDA_CHECK(cudaStreamSynchronize(s2));
      bool ws_ok = (st2 == FV3_GPU_SUCCESS);
      if (!ws_ok) {
        std::fprintf(stderr, "[%s] workspace-path launch failed: %s\n",
                     names[c], err2.message);
        nfail++;
      } else {
        std::vector<fv_real> q_out2(q_h.size());
        CUDA_CHECK(cudaMemcpy(q_out2.data(), d_q, q_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        bool bitwise = (q_out2 == q_out);
        std::printf("[%s] workspace-path bitwise-vs-local-scratch OK=%d "
                    "(ws_bytes=%llu)\n",
                    names[c], bitwise ? 1 : 0,
                    (unsigned long long)ws_bytes);
        if (!bitwise) nfail++;
      }
      cudaStreamDestroy(s2);
      cudaFree(d_workspace);
    }

    cudaFree(d_pe); cudaFree(d_ak); cudaFree(d_bk); cudaFree(d_q);
    cudaStreamDestroy(s);
  }

  std::printf("REMAP_Q_PROD_SHAPE_GPU_SMOKE: %d/9 cases failed\n", nfail);
  return nfail == 0 ? 0 : 1;
}
