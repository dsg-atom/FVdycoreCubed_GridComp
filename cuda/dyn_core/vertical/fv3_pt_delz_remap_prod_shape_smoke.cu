//===========================================================================
// WP-C fv_mapz pt/delz production-SHAPE GPU smoke test (Prompt 55, Stage 8).
//
// fv3_pt_delz_remap_smoke.cu (Prompt 52C) launches a single column
// (is=ie=js=je=0) with hand-built degenerate RiemViewC3 views (s0=s1=0,
// s2=1) -- it never exercises the multi-dimensional strides the real
// fv3_gpu_field_registry.F90 registration produces, so it cannot reproduce
// the Prompt 54 out-of-bounds finding at all. This test closes that gap: it
// builds device views with the SAME lower/stride layout the registry
// actually produces for a production-shaped C32 tile (is=js=1, ie=je=32,
// npz=10, 1-cell halo on pe/peln matching Atm%pe/peln's real allocation,
// 3-cell halo on pt/delz/delp matching Atm%pt/delz/delp, no halo on pk/pkz
// matching Atm%pk/pkz), then launches the real production launcher
// (launch_pt_delz_remap) over the full 2D interior grid -- exactly the
// launch shape that triggered the Prompt 54 sanitizer finding.
//
// Every column is filled with the SAME per-level pt/delz/pe/ak/bk values as
// one of the 6 existing pt_delz-driver fixtures (case1..case6), so the
// existing fixture expectations remain the correctness oracle: every column
// must reproduce the same case's single-column result bitwise/near-bitwise
// (host path, no FMA-contraction concern here since this binary is not the
// -fmad=false diagnostic twin). Run manually via srun + compute-sanitizer,
// like its sibling; not registered as a ctest.
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

// Same generation formulas as fv3_pt_delz_remap_smoke.cu (pt_delz-driver
// main.F90:93-98 / 235-288), duplicated rather than shared since neither
// file exposes a header-level API for them and this is a standalone,
// GPU-only smoke binary like its sibling.
void gen_grid(int km, double ptop, double psfc, fv_real *pe, fv_real *ak,
              fv_real *bk) {
  for (int k = 0; k <= km; ++k) {
    const double frac = double(k) / double(km);
    pe[k] = fv_real(ptop + (psfc - ptop) * std::pow(frac, 1.5));
    bk[k] = fv_real(std::pow(frac, 1.4));
    ak[k] = fv_real(ptop * (1.0 - std::pow(frac, 1.4)));
  }
}

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
  double akap = 0.0, rrg = 0.0;
};

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

// Column-major linear offset matching add()'s stride formula
// (fv3_gpu_field_registry.F90:460-472): stride(1)=1, stride(2)=ex(1),
// stride(3)=ex(1)*ex(2), axes taken in the order the registration lists
// them -- i.e. the REAL physical nesting order of the Fortran array, not
// necessarily (i,j,k).
inline int64_t off3(int64_t a0, int64_t e0, int64_t a1, int64_t e1,
                    int64_t a2) {
  return a0 + e0 * (a1 + e1 * a2);
}

} // namespace

int main() {
  // Production-shaped C32, 1 tile/rank: matches the Fortran local bounds
  // used by the real registration (fv3_gpu_field_registry.F90:729-746).
  const int64_t is = 1, ie = 32, js = 1, je = 32;
  const int64_t ni = ie - is + 1, nj = je - js + 1;      // 32, 32
  const int64_t isd = is - 3, jsd = js - 3;              // 3-cell halo
  const int64_t nih = ie - isd + 1, njh = je - jsd + 1;  // 38, 38
  const int km = 10;
  const int64_t ni2 = ni + 2, nj2 = nj + 2;              // pe/peln 1-cell halo
  const int64_t nk2 = km + 1;
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
    const fv_real akap = fv_real(e.akap);
    const fv_real rdgas = fv_real(-e.rrg);
    const fv_real grav = fv_real(1.0);

    std::vector<fv_real> pe_col(km + 1), ak_h(km + 1), bk_h(km + 1);
    std::vector<fv_real> pt_col(km), delz_col(km);
    gen_grid(km, ptop, psfc, pe_col.data(), ak_h.data(), bk_h.data());
    fill_pt_delz(c + 1, km, pt_col.data(), delz_col.data());

    // ---- broadcast the single-column fixture data across every (i,j) of
    // the full production-shaped domain, in each field's REAL physical
    // layout -----------------------------------------------------------
    // pe has a 1-cell halo (is-1..ie+1, js-1..je+1 -> ni2 x nj2); peln does
    // NOT (is..ie, js..je -> ni x nj) -- matching add3's real registered
    // extents (fv3_gpu_field_registry.F90:743-744). Using pe's padded
    // extent/stride for peln's buffer here previously produced a
    // fill-vs-view stride mismatch (peln_h filled with stride ni2, but
    // a.peln's view declares stride ni) -- silently wrong reads without any
    // out-of-bounds access, since both buffers were still large enough.
    std::vector<fv_real> pe_h(ni2 * nk2 * nj2, fv_real(0));      // (i,k,j)
    std::vector<fv_real> peln_h(ni * nk2 * nj, fv_real(0));      // (i,k,j)
    std::vector<fv_real> pt_h(nih * njh * km, fv_real(0));       // (i,j,k)
    std::vector<fv_real> delz_h(nih * njh * km, fv_real(0));     // (i,j,k)
    std::vector<fv_real> delp_h(nih * njh * km, fv_real(0));     // (i,j,k)
    std::vector<fv_real> pk_h(ni * nj * (km + 1), fv_real(0));   // (i,j,k)
    std::vector<fv_real> pkz_h(ni * nj * km, fv_real(0));        // (i,j,k)

    for (int64_t jj = 0; jj < nj2; ++jj)
      for (int64_t ii = 0; ii < ni2; ++ii)
        for (int k = 0; k <= km; ++k) {
          const int64_t off = off3(ii, ni2, k, nk2, jj);
          pe_h[off] = pe_col[k];
        }
    for (int64_t jj = 0; jj < nj; ++jj)
      for (int64_t ii = 0; ii < ni; ++ii)
        for (int k = 0; k <= km; ++k) {
          const int64_t off = off3(ii, ni, k, nk2, jj);
          peln_h[off] = fv_real(std::log(double(pe_col[k])));
        }
    for (int64_t jj = 0; jj < njh; ++jj)
      for (int64_t ii = 0; ii < nih; ++ii)
        for (int k = 0; k < km; ++k) {
          const int64_t off = off3(ii, nih, jj, njh, k);
          pt_h[off] = pt_col[k];
          delz_h[off] = delz_col[k];
        }

    fv_real *d_pe, *d_peln, *d_ak, *d_bk, *d_pt, *d_delz, *d_delp, *d_pk,
        *d_pkz;
    CUDA_CHECK(cudaMalloc(&d_pe, pe_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_peln, peln_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_ak, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_bk, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pt, pt_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delz, delz_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delp, delp_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pk, pk_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pkz, pkz_h.size() * sizeof(fv_real)));

    CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), pe_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_peln, peln_h.data(),
                          peln_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ak, ak_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, bk_h.data(), (km + 1) * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt, pt_h.data(), pt_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delz, delz_h.data(),
                          delz_h.size() * sizeof(fv_real),
                          cudaMemcpyHostToDevice));

    // Views match exactly what riem_c_view3()/ptdz_cview1() would build from
    // the real registry descriptors for this shape (see off3's comment).
    PtDelzRemapArgs a{};
    a.pe = RiemViewC3{d_pe, is - 1, 0, js - 1, 1, ni2, ni2 * nk2};
    a.peln = RiemViewC3{d_peln, is, 0, js, 1, ni, ni * nk2};
    a.pt = RiemViewC3{d_pt, isd, jsd, 0, 1, nih, nih * njh};
    a.delz = RiemViewC3{d_delz, isd, jsd, 0, 1, nih, nih * njh};
    a.delp = RiemViewC3{d_delp, isd, jsd, 0, 1, nih, nih * njh};
    a.pk = RiemViewC3{d_pk, is, js, 0, 1, ni, ni * nj};
    a.pkz = RiemViewC3{d_pkz, is, js, 0, 1, ni, ni * nj};
    a.ak = PtDelzView1{d_ak, 0, 1};
    a.bk = PtDelzView1{d_bk, 0, 1};
    a.is = is; a.ie = ie; a.js = js; a.je = je;
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
      cudaFree(d_pe); cudaFree(d_peln); cudaFree(d_ak); cudaFree(d_bk);
      cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_delp); cudaFree(d_pk);
      cudaFree(d_pkz);
      continue;
    }

    std::vector<fv_real> pt_out(pt_h.size()), delz_out(delz_h.size()),
        pkz_out(pkz_h.size()), delp_out(delp_h.size());
    CUDA_CHECK(cudaMemcpy(pt_out.data(), d_pt, pt_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delz_out.data(), d_delz,
                          delz_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pkz_out.data(), d_pkz,
                          pkz_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delp_out.data(), d_delp,
                          delp_h.size() * sizeof(fv_real),
                          cudaMemcpyDeviceToHost));

    // Every interior column (i=is..ie, j=js..je) must match the same
    // single-column fixture exactly -- broadcast input, so broadcast output.
    double max_rel_pt = 0, max_rel_dz = 0, max_rel_pkz = 0, max_rel_dp = 0;
    double max_abs_pt = 0, max_abs_dz = 0, max_abs_pkz = 0, max_abs_dp = 0;
    auto upd = [](double val, double exp, double &max_abs, double &max_rel) {
      const double d = std::fabs(val - exp);
      max_abs = std::max(max_abs, d);
      max_rel = std::max(max_rel, d / std::max(1.0, std::fabs(exp)));
    };
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i)
        for (int k = 0; k < km; ++k) {
          const int64_t pt_off = off3(i - isd, nih, j - jsd, njh, k);
          const int64_t pkz_off = off3(i - is, ni, j - js, nj, k);
          upd(double(pt_out[pt_off]), e.pt_out[k], max_abs_pt, max_rel_pt);
          upd(double(delz_out[pt_off]), e.delz_out[k], max_abs_dz, max_rel_dz);
          upd(double(pkz_out[pkz_off]), e.pkz_out[k], max_abs_pkz,
              max_rel_pkz);
          upd(double(delp_out[pt_off]), e.dpeO[k], max_abs_dp, max_rel_dp);
        }
    const double tol = 1e-9;
    const bool ok = max_rel_pt < tol && max_rel_dz < tol &&
                    max_rel_pkz < tol && max_rel_dp < tol;
    std::printf(
        "[%s] cols=%ldx%ld pt(abs=%.3e,rel=%.3e) delz(abs=%.3e,rel=%.3e) "
        "pkz(abs=%.3e,rel=%.3e) delp/dpeO(abs=%.3e,rel=%.3e) OK=%d\n",
        names[c], (long)ni, (long)nj, max_abs_pt, max_rel_pt, max_abs_dz,
        max_rel_dz, max_abs_pkz, max_rel_pkz, max_abs_dp, max_rel_dp,
        ok ? 1 : 0);
    if (!ok) nfail++;

    // Prompt 68 Stage 9 (unit-level): re-run the SAME case through the new
    // device-workspace path (PtDelzRemapArgs::workspace set) and require
    // bitwise-identical output vs. the local-scratch path above -- both
    // branches execute identical per-thread arithmetic, only the scratch
    // backing store differs, so any difference would indicate an indexing
    // bug in the new workspace path, not a numerical-formula change.
    {
      // pe/peln/pt/delz are mutated in place by the first run; restore the
      // original inputs before re-running. delp/pk/pkz are fully
      // overwritten at every (i,j,k) regardless, so no reset needed.
      CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), pe_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_peln, peln_h.data(),
                            peln_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_pt, pt_h.data(), pt_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_delz, delz_h.data(),
                            delz_h.size() * sizeof(fv_real),
                            cudaMemcpyHostToDevice));

      const uint64_t ws_scalars_per_col =
          pt_delz_remap_scalars_per_column(km);
      const uint64_t ws_columns = pt_delz_remap_columns(is, ie, js, je);
      const uint64_t ws_bytes = ws_columns * ws_scalars_per_col * sizeof(fv_real);
      fv_real *d_workspace = nullptr;
      CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));

      PtDelzRemapArgs a2 = a;
      a2.workspace = d_workspace;
      a2.workspace_scalars_per_column = ws_scalars_per_col;
      a2.workspace_bytes = ws_bytes;

      cudaStream_t s2;
      CUDA_CHECK(cudaStreamCreate(&s2));
      fv3_gpu_error err2{};
      fv3_gpu_status st2 = launch_pt_delz_remap(a2, s2, &err2);
      CUDA_CHECK(cudaStreamSynchronize(s2));
      bool ws_ok = (st2 == FV3_GPU_SUCCESS);
      if (!ws_ok) {
        std::fprintf(stderr, "[%s] workspace-path launch failed: %s\n",
                     names[c], err2.message);
        nfail++;
      } else {
        std::vector<fv_real> pt_out2(pt_h.size()), delz_out2(delz_h.size()),
            pkz_out2(pkz_h.size()), delp_out2(delp_h.size());
        CUDA_CHECK(cudaMemcpy(pt_out2.data(), d_pt,
                              pt_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(delz_out2.data(), d_delz,
                              delz_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pkz_out2.data(), d_pkz,
                              pkz_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(delp_out2.data(), d_delp,
                              delp_h.size() * sizeof(fv_real),
                              cudaMemcpyDeviceToHost));
        bool bitwise = (pt_out2 == pt_out) && (delz_out2 == delz_out) &&
                       (pkz_out2 == pkz_out) && (delp_out2 == delp_out);
        std::printf("[%s] workspace-path bitwise-vs-local-scratch OK=%d\n",
                    names[c], bitwise ? 1 : 0);
        if (!bitwise) nfail++;
      }
      cudaStreamDestroy(s2);
      cudaFree(d_workspace);
    }

    cudaFree(d_pe); cudaFree(d_peln); cudaFree(d_ak); cudaFree(d_bk);
    cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_delp); cudaFree(d_pk);
    cudaFree(d_pkz);
    cudaStreamDestroy(s);
  }

  std::printf("PT_DELZ_PROD_SHAPE_GPU_SMOKE: %d/6 cases failed\n", nfail);
  return nfail == 0 ? 0 : 1;
}
