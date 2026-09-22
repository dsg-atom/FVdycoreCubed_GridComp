//===========================================================================
// WP-C fv_mapz pt/delz production-remap GPU smoke test (Prompt 52C, Stage
// 8-9). Same pattern as fv3_map_scalar_kernel.cu (Prompt 52B): a real
// __global__ kernel launched on a real GPU via the raw (non-ExecutionContext)
// launcher, single column (is=ie=js=je=0), compared against the validated
// standalone Fortran fixtures at
// tests/fortran_baseline/fv_mapz/pt_delz_fixtures/case*.txt (Prompt
// 52C-prep, pt_delz-driver/main.F90). GPU-execution + tolerance-comparison
// smoke check only; not registered as a ctest (requires an actual GPU) --
// run manually via srun, matching fv3_wp_c_map_scalar_gpu_smoke.
//
// Grid/pt/delz generation below reproduces pt_delz-driver/main.F90's
// run_case/fill_pt_delz exactly (same formulas, evaluated in double
// precision on the host) rather than re-parsing them out of the fixture
// text, since the fixture only records the RESULTING pe/peO/dpe/dpeO
// columns, not the ak/bk coefficients used to derive peO.
//===========================================================================

#include "fv3_pt_delz_remap.cuh"
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

// pt_delz-driver/main.F90:93-98 -- Lagrangian pe (exponent 1.5), hybrid
// ak/bk (exponent 1.4).
void gen_grid(int km, double ptop, double psfc, fv_real *pe, fv_real *ak,
              fv_real *bk) {
  for (int k = 0; k <= km; ++k) {
    const double frac = double(k) / double(km);
    pe[k] = fv_real(ptop + (psfc - ptop) * std::pow(frac, 1.5));
    bk[k] = fv_real(std::pow(frac, 1.4));
    ak[k] = fv_real(ptop * (1.0 - std::pow(frac, 1.4)));
  }
}

// pt_delz-driver/main.F90:235-288 fill_pt_delz, transcribed verbatim
// (shape_id is 1-based, matching the Fortran select case).
void fill_pt_delz(int shape_id, int km, fv_real *pt, fv_real *delz) {
  for (int k = 0; k < km; ++k) {
    switch (shape_id) {
    case 1:
      pt[k] = fv_real(300.0 - 3.0 * double(k));
      delz[k] = fv_real(-(200.0 + 150.0 * double(k)));
      break;
    case 2:
      if (k < km / 2) {
        pt[k] = fv_real(220.0);
        delz[k] = fv_real(-100.0);
      } else {
        pt[k] = fv_real(290.0);
        delz[k] = fv_real(-600.0);
      }
      break;
    case 3:
      pt[k] = fv_real(250.0 + 30.0 * std::sin(2.0 * 3.14159265 * double(k) /
                                               double(km)));
      delz[k] = fv_real(-(500.0 + 200.0 * std::sin(2.0 * 3.14159265 *
                                                        double(k) / double(km) +
                                                    1.0)));
      break;
    case 4:
      pt[k] = fv_real(300.0 + 1.0e-6 * double(k + 1));
      delz[k] = fv_real(-500.0 - 1.0e-6 * double(k + 1));
      break;
    case 5:
      if ((k + 1) % 2 == 0) {
        pt[k] = fv_real(280.0);
        delz[k] = fv_real(-700.0);
      } else {
        pt[k] = fv_real(240.0);
        delz[k] = fv_real(-300.0);
      }
      break;
    case 6: {
      const double frac = double(k) / double(km - 1);
      pt[k] = fv_real(195.0 + 95.0 * std::pow(frac, 0.7));
      delz[k] = fv_real(-(150.0 + 1300.0 * std::pow(1.0 - frac, 1.5)));
      break;
    }
    default:
      pt[k] = fv_real(250.0);
      delz[k] = fv_real(-500.0);
    }
  }
}

struct Expected {
  std::vector<double> dpeO, pt_out, delz_out, pkz_out;
  double akap = 0.0, rrg = 0.0;  // parsed from the "# k1k= ... rrg= ... akap=
                                  // ... t_min= ..." header line -- the exact
                                  // constants_mod values the Fortran driver
                                  // actually used, not re-derived from a
                                  // guessed RDGAS/CP_AIR/GRAV combination
                                  // (constants_mod is an external MAPL/FMS
                                  // dependency not vendored in this checkout,
                                  // so its literals cannot be read directly).
};

// Parses the two fixed-format tables written by pt_delz-driver/main.F90's
// run_case (lines 195-209): table 1 is "k pe peO dpe dpeO" (km data rows +
// one 2-column edge row), table 2 is "k pt_in pt_pre pt_out delz_in
// delz_pre delz_out pkz_out" (km rows).
bool parse_fixture(const std::string &path, int km, Expected &e) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open fixture: %s\n", path.c_str());
    return false;
  }
  e.dpeO.assign(km, 0.0);
  e.pt_out.assign(km, 0.0);
  e.delz_out.assign(km, 0.0);
  e.pkz_out.assign(km, 0.0);
  bool in_table2 = false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.find("pt_in") != std::string::npos) in_table2 = true;
      if (line.find("rrg=") != std::string::npos) {
        std::string tag;
        std::istringstream hs(line);
        double val;
        while (hs >> tag) {
          if (tag == "rrg=" && (hs >> val)) e.rrg = val;
          else if (tag == "akap=" && (hs >> val)) e.akap = val;
        }
      }
      continue;
    }
    std::istringstream ss(line);
    int k;
    if (!(ss >> k) || k < 1 || k > km) continue;
    if (!in_table2) {
      double pe_v, peO_v, dpe_v, dpeO_v;
      if (ss >> pe_v >> peO_v >> dpe_v >> dpeO_v) e.dpeO[k - 1] = dpeO_v;
    } else {
      double pt_in, pt_pre, pt_o, dz_in, dz_pre, dz_o, pkz_o;
      if (ss >> pt_in >> pt_pre >> pt_o >> dz_in >> dz_pre >> dz_o >> pkz_o) {
        e.pt_out[k - 1] = pt_o;
        e.delz_out[k - 1] = dz_o;
        e.pkz_out[k - 1] = pkz_o;
      }
    }
  }
  return true;
}

} // namespace

int main() {
  const int km = 10;
  const double ptop = 100.0, psfc = 100000.0;

  const char *names[6] = {"case1_smooth_monotonic", "case2_strong_gradient",
                          "case3_local_extrema",     "case4_near_uniform",
                          "case5_limiter_activating", "case6_realistic_c512"};

  int nfail = 0;
  for (int c = 0; c < 6; ++c) {
    const std::string path =
        std::string(FV3_PT_DELZ_FIXTURE_DIR) + "/" + names[c] + ".txt";
    Expected e;
    if (!parse_fixture(path, km, e)) {
      nfail++;
      continue;
    }
    // akap/rrg come straight from the fixture's own header line ("# k1k=
    // ... rrg= ... akap= ... t_min= ..."), i.e. the exact constants_mod
    // values the Fortran driver actually used -- NOT re-derived from a
    // guessed RDGAS/CP_AIR/GRAV triple (constants_mod is an external
    // MAPL/FMS dependency not vendored in this checkout, so its literals
    // are not independently readable). rdgas/grav below are just ONE pair
    // whose ratio reproduces the fixture's rrg exactly; the kernel only
    // ever uses them through the ratio rrg = -rdgas/grav.
    const fv_real akap = fv_real(e.akap);
    const fv_real rdgas = fv_real(-e.rrg);
    const fv_real grav = fv_real(1.0);

    std::vector<fv_real> pe_h(km + 1), ak_h(km + 1), bk_h(km + 1);
    std::vector<fv_real> pt_h(km), delz_h(km), peln_h(km + 1);
    gen_grid(km, ptop, psfc, pe_h.data(), ak_h.data(), bk_h.data());
    fill_pt_delz(c + 1, km, pt_h.data(), delz_h.data());
    for (int k = 0; k <= km; ++k)
      peln_h[k] = fv_real(std::log(double(pe_h[k])));

    fv_real *d_pe, *d_peln, *d_ak, *d_bk, *d_pt, *d_delz, *d_delp, *d_pk, *d_pkz;
    CUDA_CHECK(cudaMalloc(&d_pe, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_peln, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_ak, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_bk, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pt, km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delz, km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delp, km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pk, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pkz, km * sizeof(fv_real)));

    CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_peln, peln_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ak, ak_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, bk_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt, pt_h.data(), km * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delz, delz_h.data(), km * sizeof(fv_real),
                          cudaMemcpyHostToDevice));

    PtDelzRemapArgs a{};
    a.pe = RiemViewC3{d_pe, 0, 0, 0, 0, 0, 1};
    a.peln = RiemViewC3{d_peln, 0, 0, 0, 0, 0, 1};
    a.pt = RiemViewC3{d_pt, 0, 0, 0, 0, 0, 1};
    a.delz = RiemViewC3{d_delz, 0, 0, 0, 0, 0, 1};
    a.delp = RiemViewC3{d_delp, 0, 0, 0, 0, 0, 1};
    a.pk = RiemViewC3{d_pk, 0, 0, 0, 0, 0, 1};
    a.pkz = RiemViewC3{d_pkz, 0, 0, 0, 0, 0, 1};
    a.ak = PtDelzView1{d_ak, 0, 1};
    a.bk = PtDelzView1{d_bk, 0, 1};
    a.is = a.ie = a.js = a.je = 0;
    a.km = km;
    a.akap = akap;
    a.rdgas = rdgas;
    a.grav = grav;
    a.ptop = fv_real(ptop);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    fv3_gpu_error err{};
    fv3_gpu_status st = launch_pt_delz_remap(a, s, &err);
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (st != FV3_GPU_SUCCESS) {
      std::fprintf(stderr, "[%s] launch failed: %s\n", names[c], err.message);
      nfail++;
      cudaStreamDestroy(s);
      continue;
    }

    std::vector<fv_real> pt_out(km), delz_out(km), pkz_out(km), delp_out(km);
    CUDA_CHECK(cudaMemcpy(pt_out.data(), d_pt, km * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delz_out.data(), d_delz, km * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pkz_out.data(), d_pkz, km * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delp_out.data(), d_delp, km * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));

    double max_rel_pt = 0, max_rel_dz = 0, max_rel_pkz = 0, max_rel_dp = 0;
    double max_abs_pt = 0, max_abs_dz = 0, max_abs_pkz = 0, max_abs_dp = 0;
    for (int k = 0; k < km; ++k) {
      auto upd = [](double val, double exp, double &max_abs, double &max_rel) {
        const double d = std::fabs(val - exp);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1.0, std::fabs(exp)));
      };
      upd(double(pt_out[k]), e.pt_out[k], max_abs_pt, max_rel_pt);
      upd(double(delz_out[k]), e.delz_out[k], max_abs_dz, max_rel_dz);
      upd(double(pkz_out[k]), e.pkz_out[k], max_abs_pkz, max_rel_pkz);
      upd(double(delp_out[k]), e.dpeO[k], max_abs_dp, max_rel_dp);
    }
    const double tol = 1e-9; // tight, not bitwise (host libm vs Fortran
                             // intrinsics evaluate transcendentals
                             // independently; see report for the actual
                             // achieved magnitude).
    const bool ok = max_rel_pt < tol && max_rel_dz < tol &&
                    max_rel_pkz < tol && max_rel_dp < tol;
    std::printf(
        "[%s] pt(abs=%.3e,rel=%.3e) delz(abs=%.3e,rel=%.3e) "
        "pkz(abs=%.3e,rel=%.3e) delp/dpeO(abs=%.3e,rel=%.3e) OK=%d\n",
        names[c], max_abs_pt, max_rel_pt, max_abs_dz, max_rel_dz, max_abs_pkz,
        max_rel_pkz, max_abs_dp, max_rel_dp, ok ? 1 : 0);
    if (!ok) nfail++;

    cudaFree(d_pe); cudaFree(d_peln); cudaFree(d_ak); cudaFree(d_bk);
    cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_delp); cudaFree(d_pk);
    cudaFree(d_pkz);
    cudaStreamDestroy(s);
  }

  std::printf("PT_DELZ_GPU_SMOKE: %d/6 cases failed\n", nfail);
  return nfail == 0 ? 0 : 1;
}
