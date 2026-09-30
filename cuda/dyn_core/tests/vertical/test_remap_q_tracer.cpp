// Prompt 85 Stage 1/7: host-path correctness test for the CUDA REMAP_Q
// reuse design -- calling the existing map_scalar_column<double>(..., iv=0,
// gmao_bot=false, gmao_top=false, kord=8) directly against the Prompt 84B
// authoritative mapn_tracer (REMAP_Q) fixtures
// (tests/fortran_baseline/fv_mapz/fixtures/case14..22_tracer_*.txt).
//
// Design note (see fv3_map_scalar_column.hpp's Prompt 85 header comment):
// mapn_tracer's own nq-batched "Mapping" search/integrate section
// (fv_mapz.F90:2120-2179) is, for nq=1, byte-identical to map_scalar_column's
// existing gmao_bot=false/gmao_top=false "standard FV3 remapping" path -- so
// REMAP_Q's nq=1 production case needs NO new mapping-loop code, only the
// Stage 1 kord=8 scalar_profile_column branch (already added, default-9
// behavior unchanged). case23 (nq=3, multi-tracer batching) is a different
// fixture format and is explicitly OUT OF SCOPE for this reuse design; not
// loaded here.
//
// This exercises the __host__ side of the __host__ __device__ header only --
// it does NOT itself prove GPU execution; see the Stage 9 GPU smoke test for
// that separate claim.
//
// R8 bitwise-first policy (same convention as test_map_scalar.cpp):
// mismatch_count must be 0 for every case; this test fails otherwise.
#include "fv3_map_scalar_column.hpp"

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
  int km = 0, nq = 0, kord = 0;
  double q_min = 0.0;
  double mass_in = 0.0, mass_out = 0.0;
  std::vector<double> pe1, pe2;       // km+1 entries
  std::vector<double> q1, q2_fortran; // km entries
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
        else if (key == "nq") f.nq = std::atoi(val.c_str());
        else if (key == "kord") f.kord = std::atoi(val.c_str());
      }
      continue;
    }
    if (line.rfind("# q_min=", 0) == 0) {
      f.q_min = std::atof(line.substr(line.find('=') + 1).c_str());
      continue;
    }
    if (line.rfind("# mass_in=", 0) == 0) {
      std::istringstream iss(line.substr(1));
      std::string tok;
      while (iss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
        if (key == "mass_in") f.mass_in = std::atof(val.c_str());
        else if (key == "mass_out") f.mass_out = std::atof(val.c_str());
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
  return f.km > 0 && f.nq == 1 &&
         static_cast<int>(f.pe1.size()) == f.km + 1 &&
         static_cast<int>(f.pe2.size()) == f.km + 1 &&
         static_cast<int>(f.q1.size()) == f.km &&
         static_cast<int>(f.q2_fortran.size()) == f.km;
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
    std::fprintf(stderr, "FAIL %s: could not parse fixture (or nq != 1)\n",
                 case_name.c_str());
    return false;
  }
  if (f.kord != 8) {
    std::fprintf(stderr, "FAIL %s: expected kord=8, fixture has kord=%d\n",
                 case_name.c_str(), f.kord);
    return false;
  }

  std::vector<double> dpe1(f.km), dpe2(f.km);
  for (int k = 0; k < f.km; ++k) dpe1[k] = f.pe1[k + 1] - f.pe1[k];
  for (int k = 0; k < f.km; ++k) dpe2[k] = f.pe2[k + 1] - f.pe2[k];

  std::vector<double> q2_cuda(f.km, 0.0);
  std::vector<double> c2(f.km), c3(f.km), c4(f.km), gam(f.km), q(f.km + 1);
  std::vector<char> extm(f.km);

  Column<const double> pe1_c{f.pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
  Column<const double> pe2_c{f.pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
  Column<const double> q1_c{f.q1.data(), 1};
  Column<double> q2_c{q2_cuda.data(), 1};

  // iv=0 (scalar, matching mapn_tracer's hardcoded iv=0), kn=km (REMAP_Q
  // remaps within the same layer count), gmao_bot/gmao_top=false (mapn_tracer
  // has no GMAO-boundary concept), kord=8 (this fixture family's validated
  // production kord_tr value).
  fv3::gpu::vertical::map_scalar_column<double>(
      pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2.data(), c3.data(), c4.data(),
      gam.data(), q.data(), reinterpret_cast<bool *>(extm.data()), f.km, f.km,
      /*iv=*/0, f.q_min, /*gmao_bot=*/false, /*gmao_top=*/false, /*kord=*/8);

  const Metrics m = compare(f.q2_fortran, q2_cuda);
  const bool ok = m.mismatch_count == 0 && m.nan_count == 0;
  std::printf("%-32s kord=%d max_abs=%.3e max_rel=%.3e rms=%.3e "
              "nan=%d mismatch=%d/%d %s\n",
              case_name.c_str(), f.kord, m.max_abs, m.max_rel, m.rms,
              m.nan_count, m.mismatch_count, f.km, ok ? "PASS" : "FAIL");
  if (verbose && !ok) {
    for (int k = 0; k < f.km; ++k) {
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

  static const char *kCases[] = {
      "case14_tracer_constant.txt",
      "case15_tracer_smooth_gradient.txt",
      "case16_tracer_sharp_front.txt",
      "case17_tracer_local_max.txt",
      "case18_tracer_local_min.txt",
      "case19_tracer_near_qmin.txt",
      "case20_tracer_qmin_clipping.txt",
      "case21_tracer_boundary_sensitive.txt",
      "case22_tracer_c512_like_profile.txt",
      // case23_tracer_multi_nq3.txt intentionally excluded: nq=3, a
      // different fixture format, and out of scope for this reuse design.
  };
  bool all_ok = true;
  for (const char *c : kCases) all_ok &= run_case(dir, c, true);

  std::printf("RESULT=%s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
