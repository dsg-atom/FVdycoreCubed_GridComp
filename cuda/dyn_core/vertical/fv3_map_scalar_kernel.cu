//===========================================================================
// WP-C: minimal map_scalar_column/scalar_profile_column CUDA smoke kernel
// (Prompt 52B, Stage 15). One thread per horizontal column, matching the
// execution model already established for the host path in
// fv3_map_scalar_column.hpp (same file, same FV3_VERTICAL_HD-tagged device
// functions -- this file adds no new numerical logic, only a __global__
// wrapper + host launcher around it).
//
// Scope: GPU-execution + bitwise-vs-Fortran sanity check and a rough,
// non-authoritative CPU-vs-GPU timing sanity check only. This is NOT a
// controlled performance benchmark (single job, small batch) and is not
// used for any performance claim elsewhere in the project.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_map_scalar_column.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using fv3::gpu::vertical::Column;

namespace {

// Fixed per-thread local-array cap, matching the riem_solver_c convention
// (fv3_riem_solver_c.cuh: riem_solver_c_max_local_levels). All 8 Prompt 52A
// fixtures use km=kn=10; a launcher that receives km/kn>kMaxLevels rejects
// rather than silently truncating.
constexpr int kMaxLevels = 32;

// One thread per column. Flat batch layout: column c occupies
// [c*(km+1) .. c*(km+1)+km] for pe1/pe2 (km+1 entries) and
// [c*km .. c*km+km-1] for q1/q2 (km entries). All columns in a batch share
// km/kn/iv/qmin (true for every case in this smoke test).
__global__ void map_scalar_smoke_kernel(const double *pe1, const double *dpe1,
                                         const double *pe2, const double *dpe2,
                                         const double *q1, double *q2, int km,
                                         int kn, int iv, double qmin,
                                         int ncols, bool gmao_bot,
                                         bool gmao_top) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols) return;
  if (km > kMaxLevels || kn > kMaxLevels) return; // launcher-level reject below

  double c2[kMaxLevels], c3[kMaxLevels], c4[kMaxLevels], gam[kMaxLevels];
  double q[kMaxLevels + 1];
  bool extm[kMaxLevels];

  Column<const double> pe1_c{pe1 + c * (km + 1), 1};
  Column<const double> dpe1_c{dpe1 + c * km, 1};
  Column<const double> pe2_c{pe2 + c * (kn + 1), 1};
  Column<const double> dpe2_c{dpe2 + c * kn, 1};
  Column<const double> q1_c{q1 + c * km, 1};
  Column<double> q2_c{q2 + c * kn, 1};

  fv3::gpu::vertical::map_scalar_column<double>(
      pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2, c3, c4, gam, q, extm, km,
      kn, iv, qmin, gmao_bot, gmao_top);
}

struct Fixture {
  std::string name;
  int km = 0, kn = 0, iv = 0, kord = 0;
  double q_min = 0.0;
  bool gmao_bot = false, gmao_top = false; // Prompt 61: GMAO boundary flags
  std::vector<double> pe1, pe2, q1, q2_fortran;
};

bool load_fixture(const std::string &path, Fixture &f) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  int section = 0;
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
    if (line.rfind("# gmao_bot=", 0) == 0) {
      std::istringstream iss(line.substr(1));
      std::string tok;
      while (iss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
        if (key == "gmao_bot") f.gmao_bot = std::atoi(val.c_str()) != 0;
        else if (key == "gmao_top") f.gmao_top = std::atoi(val.c_str()) != 0;
      }
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
         static_cast<int>(f.q1.size()) == f.km;
}

bool check_cuda(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
  }
  return true;
}

} // namespace

int main() {
#ifndef FV3_MAP_SCALAR_FIXTURE_DIR
#error "FV3_MAP_SCALAR_FIXTURE_DIR must be defined by the build"
#endif
  const std::string dir = FV3_MAP_SCALAR_FIXTURE_DIR;
  static const char *kCases[] = {
      "case1_smooth_monotonic.txt",       "case2_strong_gradient.txt",
      "case3_local_extrema.txt",          "case4_near_uniform.txt",
      "case5_variable_layer_thickness.txt", "case6_limiter_activating.txt",
      "case7_realistic_c512_profile.txt", "case8_positive_definite_iv0.txt",
      "case9_gmao_boundary_conditions.txt",
      // Prompt 71 Stage 2: iv=-1 (winds, REMAP_UV) boundary-clamp coverage.
      "case10_wind_uniform_sign_iv_neg1.txt", "case11_wind_top_sign_reversal.txt",
      "case12_wind_bottom_sign_reversal.txt",
      "case13_wind_interior_sign_reversal.txt",
  };

  bool all_ok = true;
  for (const char *name : kCases) {
    Fixture f;
    if (!load_fixture(dir + "/" + name, f)) {
      std::fprintf(stderr, "FAIL %s: could not parse fixture\n", name);
      return 1;
    }
    if (f.km > kMaxLevels || f.kn > kMaxLevels) {
      std::fprintf(stderr, "REJECT %s: km/kn=%d/%d exceeds kMaxLevels=%d\n",
                   name, f.km, f.kn, kMaxLevels);
      return 1;
    }
    std::vector<double> dpe1(f.km), dpe2(f.kn);
    for (int k = 0; k < f.km; ++k) dpe1[k] = f.pe1[k + 1] - f.pe1[k];
    for (int k = 0; k < f.kn; ++k) dpe2[k] = f.pe2[k + 1] - f.pe2[k];

    double *d_pe1, *d_dpe1, *d_pe2, *d_dpe2, *d_q1, *d_q2;
    if (!check_cuda(cudaMalloc(&d_pe1, f.pe1.size() * sizeof(double)), "malloc pe1") ||
        !check_cuda(cudaMalloc(&d_dpe1, dpe1.size() * sizeof(double)), "malloc dpe1") ||
        !check_cuda(cudaMalloc(&d_pe2, f.pe2.size() * sizeof(double)), "malloc pe2") ||
        !check_cuda(cudaMalloc(&d_dpe2, dpe2.size() * sizeof(double)), "malloc dpe2") ||
        !check_cuda(cudaMalloc(&d_q1, f.q1.size() * sizeof(double)), "malloc q1") ||
        !check_cuda(cudaMalloc(&d_q2, f.kn * sizeof(double)), "malloc q2"))
      return 1;

    cudaMemcpy(d_pe1, f.pe1.data(), f.pe1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dpe1, dpe1.data(), dpe1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pe2, f.pe2.data(), f.pe2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dpe2, dpe2.data(), dpe2.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1, f.q1.data(), f.q1.size() * sizeof(double), cudaMemcpyHostToDevice);

    map_scalar_smoke_kernel<<<1, 1>>>(d_pe1, d_dpe1, d_pe2, d_dpe2, d_q1, d_q2,
                                       f.km, f.kn, f.iv, f.q_min, 1,
                                       f.gmao_bot, f.gmao_top);
    if (!check_cuda(cudaGetLastError(), "kernel launch") ||
        !check_cuda(cudaDeviceSynchronize(), "kernel sync"))
      return 1;

    std::vector<double> q2_gpu(f.kn);
    cudaMemcpy(q2_gpu.data(), d_q2, f.kn * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_pe1); cudaFree(d_dpe1); cudaFree(d_pe2);
    cudaFree(d_dpe2); cudaFree(d_q1); cudaFree(d_q2);

    int mismatch = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (int k = 0; k < f.kn; ++k) {
      if (q2_gpu[k] != f.q2_fortran[k]) ++mismatch;
      const double d = std::fabs(q2_gpu[k] - f.q2_fortran[k]);
      if (d > max_abs) max_abs = d;
      const double rel = d / std::max(std::fabs(f.q2_fortran[k]), 1e-12);
      if (rel > max_rel) max_rel = rel;
    }
    const bool ok = mismatch == 0;
    all_ok &= ok;
    std::printf("%-32s GPU EXECUTED, mismatch=%d/%d max_abs=%.3e max_rel=%.3e %s\n",
                name, mismatch, f.kn, max_abs, max_rel, ok ? "PASS" : "FAIL");
    if (!ok) {
      for (int k = 0; k < f.kn; ++k) {
        if (q2_gpu[k] != f.q2_fortran[k])
          std::printf("  k=%2d fortran=%.17e gpu=%.17e diff=%.3e\n", k + 1,
                      f.q2_fortran[k], q2_gpu[k],
                      q2_gpu[k] - f.q2_fortran[k]);
      }
    }
  }

  // Rough, non-authoritative CPU-vs-GPU sanity timing: replicate case7's
  // shape into a batch and compare one kernel launch against an equivalent
  // serial host loop. Not a controlled A/B; sanity-check only.
  {
    Fixture f;
    load_fixture(dir + "/case7_realistic_c512_profile.txt", f);
    std::vector<double> dpe1(f.km), dpe2(f.kn);
    for (int k = 0; k < f.km; ++k) dpe1[k] = f.pe1[k + 1] - f.pe1[k];
    for (int k = 0; k < f.kn; ++k) dpe2[k] = f.pe2[k + 1] - f.pe2[k];

    const int ncols = 100000;
    std::vector<double> pe1_b(ncols * (f.km + 1)), dpe1_b(ncols * f.km);
    std::vector<double> pe2_b(ncols * (f.kn + 1)), dpe2_b(ncols * f.kn);
    std::vector<double> q1_b(ncols * f.km), q2_b(ncols * f.kn);
    for (int c = 0; c < ncols; ++c) {
      for (int k = 0; k <= f.km; ++k) pe1_b[c * (f.km + 1) + k] = f.pe1[k];
      for (int k = 0; k < f.km; ++k) dpe1_b[c * f.km + k] = dpe1[k];
      for (int k = 0; k <= f.kn; ++k) pe2_b[c * (f.kn + 1) + k] = f.pe2[k];
      for (int k = 0; k < f.kn; ++k) dpe2_b[c * f.kn + k] = dpe2[k];
      for (int k = 0; k < f.km; ++k) q1_b[c * f.km + k] = f.q1[k];
    }

    double *d_pe1, *d_dpe1, *d_pe2, *d_dpe2, *d_q1, *d_q2;
    cudaMalloc(&d_pe1, pe1_b.size() * sizeof(double));
    cudaMalloc(&d_dpe1, dpe1_b.size() * sizeof(double));
    cudaMalloc(&d_pe2, pe2_b.size() * sizeof(double));
    cudaMalloc(&d_dpe2, dpe2_b.size() * sizeof(double));
    cudaMalloc(&d_q1, q1_b.size() * sizeof(double));
    cudaMalloc(&d_q2, q2_b.size() * sizeof(double));
    cudaMemcpy(d_pe1, pe1_b.data(), pe1_b.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dpe1, dpe1_b.data(), dpe1_b.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pe2, pe2_b.data(), pe2_b.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dpe2, dpe2_b.data(), dpe2_b.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q1, q1_b.data(), q1_b.size() * sizeof(double), cudaMemcpyHostToDevice);

    const int threads = 128;
    const int blocks = (ncols + threads - 1) / threads;
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    map_scalar_smoke_kernel<<<blocks, threads>>>(d_pe1, d_dpe1, d_pe2, d_dpe2,
                                                  d_q1, d_q2, f.km, f.kn, f.iv,
                                                  f.q_min, ncols, f.gmao_bot,
                                                  f.gmao_top);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::steady_clock::now();
    std::vector<double> c2(f.km), c3(f.km), c4(f.km), gam(f.km), qs(f.km + 1);
    std::vector<char> extm(f.km);
    std::vector<double> q2_host(f.kn);
    for (int c = 0; c < ncols; ++c) {
      Column<const double> pe1_c{f.pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
      Column<const double> pe2_c{f.pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
      Column<const double> q1_c{f.q1.data(), 1};
      Column<double> q2_c{q2_host.data(), 1};
      fv3::gpu::vertical::map_scalar_column<double>(
          pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2.data(), c3.data(),
          c4.data(), gam.data(), qs.data(),
          reinterpret_cast<bool *>(extm.data()), f.km, f.kn, f.iv, f.q_min);
    }
    auto t3 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    cudaFree(d_pe1); cudaFree(d_dpe1); cudaFree(d_pe2);
    cudaFree(d_dpe2); cudaFree(d_q1); cudaFree(d_q2);

    std::printf("SANITY TIMING (non-authoritative, ncols=%d): "
                "GPU %.3f ms, single-thread host %.3f ms\n",
                ncols, gpu_ms, cpu_ms);
  }

  std::printf("RESULT=%s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
