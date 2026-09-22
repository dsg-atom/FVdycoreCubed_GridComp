// Prompt 52B: host-path correctness test for map_scalar_column /
// scalar_profile_column / cs_limiters_column (fv3_map_scalar_column.hpp)
// against the Prompt 52A authoritative Fortran fixtures
// (tests/fortran_baseline/fv_mapz/fixtures/*.txt, repo root).
//
// This exercises the __host__ side of the __host__ __device__ header --
// the same inline code a future CUDA kernel would call per-column. It does
// NOT itself prove GPU execution; see docs/performance/
// p52b_fv_mapz_cuda_standalone_2026-09-21.md for the separate GPU-side
// status.
//
// R8 bitwise-first policy (tests/fortran_baseline/fv_mapz/README.md):
// mismatch_count must be 0 for every case; this test fails otherwise.
#include "fv3_map_scalar_column.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using fv3::gpu::vertical::Column;

namespace {

struct Fixture {
  std::string name;
  int km = 0, kn = 0, iv = 0, kord = 0;
  double q_min = 0.0;
  std::vector<double> pe1, pe2;       // km+1 / kn+1 entries
  std::vector<double> q1, q2_fortran; // km / kn entries
};

bool load_fixture(const std::string &path, Fixture &f) {
  std::ifstream in(path);
  if (!in) return false;
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

struct Metrics {
  double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
  int nan_count = 0, mismatch_count = 0;
};

Metrics compare(const std::vector<double> &fortran, const std::vector<double> &cuda) {
  Metrics m;
  const double rel_floor = 1.0e-12;
  double sq = 0.0;
  for (size_t k = 0; k < fortran.size(); ++k) {
    if (std::isnan(cuda[k]) || std::isinf(cuda[k])) ++m.nan_count;
    if (fortran[k] != cuda[k]) ++m.mismatch_count;
    const double d = cuda[k] - fortran[k];
    const double ad = std::fabs(d);
    if (ad > m.max_abs) m.max_abs = ad;
    const double rel = ad / std::max(std::fabs(fortran[k]), rel_floor);
    if (rel > m.max_rel) m.max_rel = rel;
    sq += d * d;
  }
  m.rms = std::sqrt(sq / fortran.size());
  return m;
}

bool run_case(const std::string &fixture_dir, const std::string &case_name, bool verbose) {
  Fixture f;
  if (!load_fixture(fixture_dir + "/" + case_name, f)) {
    std::fprintf(stderr, "FAIL %s: could not parse fixture\n", case_name.c_str());
    return false;
  }

  std::vector<double> dpe1(f.km), dpe2(f.kn);
  for (int k = 0; k < f.km; ++k) dpe1[k] = f.pe1[k + 1] - f.pe1[k];
  for (int k = 0; k < f.kn; ++k) dpe2[k] = f.pe2[k + 1] - f.pe2[k];

  std::vector<double> q2_cuda(f.kn, 0.0);
  std::vector<double> c2(f.km), c3(f.km), c4(f.km), gam(f.km), q(f.km + 1);
  std::vector<char> extm(f.km);

  Column<const double> pe1_c{f.pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
  Column<const double> pe2_c{f.pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
  Column<const double> q1_c{f.q1.data(), 1};
  Column<double> q2_c{q2_cuda.data(), 1};

  fv3::gpu::vertical::map_scalar_column<double>(
      pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2.data(), c3.data(), c4.data(),
      gam.data(), q.data(), reinterpret_cast<bool *>(extm.data()), f.km, f.kn,
      f.iv, f.q_min);

  const Metrics m = compare(f.q2_fortran, q2_cuda);
  const bool ok = m.mismatch_count == 0 && m.nan_count == 0;
  std::printf("%-32s iv=%d kord=%d max_abs=%.3e max_rel=%.3e rms=%.3e "
              "nan=%d mismatch=%d/%d %s\n",
              case_name.c_str(), f.iv, f.kord, m.max_abs, m.max_rel, m.rms,
              m.nan_count, m.mismatch_count, f.kn, ok ? "PASS" : "FAIL");
  if (verbose && !ok) {
    for (int k = 0; k < f.kn; ++k) {
      std::printf("  k=%2d fortran=%.17e cuda=%.17e %s\n", k + 1,
                  f.q2_fortran[k], q2_cuda[k],
                  f.q2_fortran[k] == q2_cuda[k] ? "" : "<-- MISMATCH");
    }
  }
  return ok;
}

} // namespace

int main() {
#ifndef FV3_MAP_SCALAR_FIXTURE_DIR
#error "FV3_MAP_SCALAR_FIXTURE_DIR must be defined by the build"
#endif
  const std::string dir = FV3_MAP_SCALAR_FIXTURE_DIR;

  // Stage 9: debug case1 alone first.
  const bool case1_ok = run_case(dir, "case1_smooth_monotonic.txt", true);
  if (!case1_ok) {
    std::fprintf(stderr, "case1 FAILED -- stopping before running the other 7 cases\n");
    return 1;
  }

  static const char *kCases[] = {
      "case2_strong_gradient.txt",
      "case3_local_extrema.txt",
      "case4_near_uniform.txt",
      "case5_variable_layer_thickness.txt",
      "case6_limiter_activating.txt",
      "case7_realistic_c512_profile.txt",
      "case8_positive_definite_iv0.txt",
  };
  bool all_ok = true;
  for (const char *c : kCases) all_ok &= run_case(dir, c, true);

  // Stage 13: determinism -- repeat case1 (smooth) and case6 (limiter-
  // activating) several times, require bitwise-identical repeats.
  std::vector<double> first_run;
  for (int rep = 0; rep < 5; ++rep) {
    Fixture f;
    load_fixture(dir + "/case6_limiter_activating.txt", f);
    std::vector<double> dpe1(f.km), dpe2(f.kn);
    for (int k = 0; k < f.km; ++k) dpe1[k] = f.pe1[k + 1] - f.pe1[k];
    for (int k = 0; k < f.kn; ++k) dpe2[k] = f.pe2[k + 1] - f.pe2[k];
    std::vector<double> q2_cuda(f.kn, 0.0);
    std::vector<double> c2(f.km), c3(f.km), c4(f.km), gam(f.km), q(f.km + 1);
    std::vector<char> extm(f.km);
    Column<const double> pe1_c{f.pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
    Column<const double> pe2_c{f.pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
    Column<const double> q1_c{f.q1.data(), 1};
    Column<double> q2_c{q2_cuda.data(), 1};
    fv3::gpu::vertical::map_scalar_column<double>(
        pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2.data(), c3.data(),
        c4.data(), gam.data(), q.data(), reinterpret_cast<bool *>(extm.data()),
        f.km, f.kn, f.iv, f.q_min);
    if (rep == 0) first_run = q2_cuda;
    else if (q2_cuda != first_run) {
      std::fprintf(stderr, "DETERMINISM FAIL: case6 repeat %d differs\n", rep);
      all_ok = false;
    }
  }
  std::printf("determinism check (case6, 5 repeats): %s\n", all_ok ? "PASS" : "FAIL");

  std::printf("RESULT=%s\n", (case1_ok && all_ok) ? "PASS" : "FAIL");
  return (case1_ok && all_ok) ? 0 : 1;
}
