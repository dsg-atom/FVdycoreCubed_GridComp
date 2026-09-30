//===========================================================================
// WP-C fv_mapz REMAP_PRESSUREVARS production-SHAPE GPU smoke test (Prompt
// 92, Stage 3/8/9). Same broadcast methodology as
// fv3_remap_uv_prod_shape_smoke.cu / fv3_remap_q_prod_shape_smoke.cu:
// builds device views with a realistic 3-cell-halo C32-ish tile layout
// (is=1,ie=20,js=1,je=12, isd=is-3,jsd=js-3 -- larger than any offset the
// kernel itself ever dereferences, chosen deliberately so l0/l1/l2 differ
// from is/js/0 and a coordinate-shift bug in view construction would be
// caught, same rationale as the PT/DELZ prod-shape smoke test's own halo
// choice), then launches the real production launcher
// (launch_remap_pressurevars) over the full interior 2D grid, spanning
// multiple (16,8) thread blocks in both directions so the kernel's own
// i>ie/j>je boundary-return guard is exercised.
//
// Correctness oracle: reuses the exact 10 fixtures and the exact
// remap_pressurevars_host() transcription from
// tests/vertical/test_remap_pressurevars_host_reference.cpp (Prompt 91),
// duplicated into this file rather than shared, matching the project's
// existing convention for standalone GPU-only smoke binaries with no
// header-level export (fv3_remap_uv_prod_shape_smoke.cu / etc.). Every
// column is filled with the SAME fixture profile (broadcast), so every
// column of the GPU output must reproduce that fixture's single-column
// host-oracle result.
//
// Classification (Stage 3): pe (Pass 1, add/multiply only) and delp/cappa
// (Pass 2, add/subtract/divide only) involve no transcendentals and are
// compared for BITWISE equality; peln/pk (interior k, fv3_log/fv3_exp) and
// pkz (fv3_log/fv3_exp) are compared with a tight ROUND-OFF tolerance,
// since CUDA device log/exp and host libm log/exp are not guaranteed
// bit-identical even though both are within ~1 ULP of the true value.
// Boundary-passthrough peln/pk (k=0,km) involve no arithmetic at all and
// are additionally checked bitwise on their own.
//
// GPU-execution + tolerance-comparison check only; not registered as a
// ctest (requires an actual GPU) -- run manually via srun, ideally under
// compute-sanitizer memcheck.
//===========================================================================

#include "fv3_remap_pressurevars.cuh"
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

// ---------------------------------------------------------------------------
// Duplicated host oracle (exact transcription, matching
// test_remap_pressurevars_host_reference.cpp bit-for-bit) -- standalone GPU
// smoke binary, no header-level export exists to share this from.
// ---------------------------------------------------------------------------
constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kAkap = kRdgas / kCpAir;
constexpr double kCvAir = kCpAir - kRdgas;
constexpr double kGrav = 9.80665;
constexpr double kRrg = -kRdgas / kGrav;
constexpr double kRvir = 0.6077338443;

struct ColumnIn {
  int km;
  double ptop;
  std::vector<double> ak, bk, pe_old, peln_old, pk_old, pt, delz, q_sphum;
};

struct ColumnOut {
  std::vector<double> pe, peln, pk, delp, pkz, cappa;
};

ColumnOut remap_pressurevars_host(const ColumnIn &in) {
  const int km = in.km;
  ColumnOut out;
  out.pe.resize(km + 1); out.peln.resize(km + 1); out.pk.resize(km + 1);
  out.delp.resize(km); out.pkz.resize(km); out.cappa.resize(km);

  std::vector<double> pe2(km + 1);
  pe2[0] = in.ptop;
  pe2[km] = in.pe_old[km];
  for (int k = 1; k <= km - 1; ++k) pe2[k] = in.ak[k] + in.bk[k] * in.pe_old[km];
  for (int k = 0; k <= km; ++k) out.pe[k] = pe2[k];

  std::vector<double> pn2(km + 1), pk2(km + 1);
  pn2[0] = in.peln_old[0]; pn2[km] = in.peln_old[km];
  pk2[0] = in.pk_old[0]; pk2[km] = in.pk_old[km];
  for (int k = 1; k <= km - 1; ++k) {
    pn2[k] = std::log(pe2[k]);
    pk2[k] = std::exp(kAkap * pn2[k]);
  }
  for (int k = 0; k <= km; ++k) { out.peln[k] = pn2[k]; out.pk[k] = pk2[k]; }

  for (int k = 0; k <= km - 1; ++k) out.delp[k] = pe2[k + 1] - pe2[k];

  for (int k = 0; k <= km - 1; ++k) {
    const double cvm = kCvAir;
    const double cappa = kRdgas / (kRdgas + cvm / (1.0 + kRvir * in.q_sphum[k]));
    out.cappa[k] = cappa;
    out.pkz[k] = std::exp(cappa * std::log(kRrg * out.delp[k] / in.delz[k] * in.pt[k]));
  }
  return out;
}

ColumnIn make_column(int km, double ptop, double ps, double pt_surf,
                     double lapse_K_per_km, double q_sphum_const) {
  ColumnIn c;
  c.km = km;
  c.ptop = ptop;
  c.ak.assign(km + 1, 0.0);
  c.bk.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) {
    const double frac = static_cast<double>(k) / km;
    c.bk[k] = frac * frac;
    c.ak[k] = ptop * (1.0 - frac);
  }
  c.pe_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.pe_old[k] = c.ak[k] + c.bk[k] * ps;
  c.peln_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.peln_old[k] = std::log(c.pe_old[k]);
  c.pk_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.pk_old[k] = std::exp(kAkap * c.peln_old[k]);

  c.pt.assign(km, 0.0);
  c.delz.assign(km, 0.0);
  c.q_sphum.assign(km, q_sphum_const);
  for (int k = 0; k < km; ++k) {
    const double frac = static_cast<double>(k) / (km - 1);
    c.pt[k] = pt_surf + lapse_K_per_km * (1.0 - frac) * 10.0;
    const double dp = c.pe_old[k + 1] - c.pe_old[k];
    c.delz[k] = -(kRdgas * c.pt[k] / ps) * dp / 1.0e2;
    if (c.delz[k] == 0.0) c.delz[k] = -1.0;
  }
  return c;
}

struct Fixture { const char *name; ColumnIn col; };

std::vector<Fixture> build_fixtures() {
  std::vector<Fixture> f;
  f.push_back({"c512_like_smooth_q0", make_column(91, 1.0, 101325.0, 288.0, 4.0, 0.0)});
  f.push_back({"strongly_stratified_q0", make_column(60, 1.0, 101325.0, 300.0, 60.0, 0.0)});
  f.push_back({"top_boundary_thin_ptop_q0", make_column(60, 0.01, 101325.0, 280.0, 4.0, 0.0)});
  f.push_back({"bottom_boundary_low_ps_q0", make_column(60, 1.0, 60000.0, 270.0, 4.0, 0.0)});
  f.push_back({"weak_delp_fine_levels_q0", make_column(120, 1.0, 101325.0, 288.0, 4.0, 0.0)});
  f.push_back({"nonhydrostatic_large_delz_contrast_q0",
               make_column(60, 1.0, 101325.0, 320.0, 120.0, 0.0)});
  f.push_back({"q_sphum_zero_production_value", make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.0)});
  f.push_back({"q_sphum_nonzero_hypothetical", make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.01)});
  {
    ColumnIn c = make_column(60, 1.0, 108000.0, 320.0, 4.0, 0.0);
    for (int k = 0; k < c.km; ++k) c.delz[k] *= 0.01;
    f.push_back({"transcendental_stress_small_delz_q0", c});
  }
  f.push_back({"repeatability_case", make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.0)});
  return f;
}

} // namespace

int main() {
  // Production-shaped tile, 3-cell halo (deliberately wider than anything
  // the kernel dereferences -- see file header). Multiple (16,8) thread
  // blocks in both directions.
  const int64_t is = 1, ie = 20, js = 1, je = 12;
  const int64_t isd = is - 3, jsd = js - 3;
  const int64_t ni_h = ie - isd + 1, nj_h = je - jsd + 1; // 23, 15

  const auto fixtures = build_fixtures();
  int nfail = 0;

  for (const auto &fx : fixtures) {
    const int km = fx.col.km;
    const int64_t nk_pe = km + 1;
    const ColumnOut oracle = remap_pressurevars_host(fx.col);

    // pe/peln layout: (i,k,j) call convention -> l0/s0=i, l1/s1=k, l2/s2=j.
    const int64_t s0_pe = 1, s1_pe = ni_h, s2_pe = ni_h * nk_pe;
    const size_t n_pe = static_cast<size_t>(ni_h) * nk_pe * nj_h;
    // pk/pt/delz/delp/pkz layout: natural (i,j,k) -> l0/s0=i, l1/s1=j, l2/s2=k.
    const int64_t s0_ijk = 1, s1_ijk_pk = ni_h, s2_ijk_pk = ni_h * nj_h;
    const size_t n_pk = static_cast<size_t>(ni_h) * nj_h * nk_pe;
    const size_t n_km = static_cast<size_t>(ni_h) * nj_h * km;

    std::vector<fv_real> pe_h(n_pe), peln_h(n_pe), pk_h(n_pk);
    std::vector<fv_real> pt_h(n_km), delz_h(n_km), qs_h(n_km);
    std::vector<fv_real> ak_h(km + 1), bk_h(km + 1);
    for (int k = 0; k <= km; ++k) { ak_h[k] = fv_real(fx.col.ak[k]); bk_h[k] = fv_real(fx.col.bk[k]); }

    for (int64_t j = jsd; j <= je; ++j)
      for (int64_t i = isd; i <= ie; ++i) {
        for (int k = 0; k <= km; ++k) {
          const size_t off_pe = static_cast<size_t>(i - isd) * s0_pe +
                                static_cast<size_t>(k) * s1_pe +
                                static_cast<size_t>(j - jsd) * s2_pe;
          pe_h[off_pe] = fv_real(fx.col.pe_old[k]);
          peln_h[off_pe] = fv_real(fx.col.peln_old[k]);
          const size_t off_pk = static_cast<size_t>(i - isd) * s0_ijk +
                                static_cast<size_t>(j - jsd) * s1_ijk_pk +
                                static_cast<size_t>(k) * s2_ijk_pk;
          pk_h[off_pk] = fv_real(fx.col.pk_old[k]);
        }
        for (int k = 0; k < km; ++k) {
          const size_t off_km = static_cast<size_t>(i - isd) * s0_ijk +
                                static_cast<size_t>(j - jsd) * s1_ijk_pk +
                                static_cast<size_t>(k) * s2_ijk_pk;
          pt_h[off_km] = fv_real(fx.col.pt[k]);
          delz_h[off_km] = fv_real(fx.col.delz[k]);
          qs_h[off_km] = fv_real(fx.col.q_sphum[k]);
        }
      }

    fv_real *d_pe, *d_peln, *d_pk, *d_delp, *d_pkz, *d_pt, *d_delz, *d_ak, *d_bk, *d_qs, *d_cappa;
    CUDA_CHECK(cudaMalloc(&d_pe, n_pe * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_peln, n_pe * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pk, n_pk * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delp, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pkz, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pt, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delz, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_ak, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_bk, (km + 1) * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_qs, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_cappa, n_km * sizeof(fv_real)));

    CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), n_pe * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_peln, peln_h.data(), n_pe * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pk, pk_h.data(), n_pk * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt, pt_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delz, delz_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qs, qs_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ak, ak_h.data(), (km + 1) * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, bk_h.data(), (km + 1) * sizeof(fv_real), cudaMemcpyHostToDevice));

    PressureVarsArgs a{};
    a.pe = RiemViewC3{d_pe, isd, 0, jsd, s0_pe, s1_pe, s2_pe};
    a.peln = RiemViewC3{d_peln, isd, 0, jsd, s0_pe, s1_pe, s2_pe};
    a.pk = RiemViewC3{d_pk, isd, jsd, 0, s0_ijk, s1_ijk_pk, s2_ijk_pk};
    a.delp = RiemViewC3{d_delp, isd, jsd, 0, s0_ijk, s1_ijk_pk, s2_ijk_pk};
    a.pkz = RiemViewC3{d_pkz, isd, jsd, 0, s0_ijk, s1_ijk_pk, s2_ijk_pk};
    a.pt = RiemViewC3{d_pt, isd, jsd, 0, s0_ijk, s1_ijk_pk, s2_ijk_pk};
    a.delz = RiemViewC3{d_delz, isd, jsd, 0, s0_ijk, s1_ijk_pk, s2_ijk_pk};
    a.ak = PtDelzView1{d_ak, 0, 1};
    a.bk = PtDelzView1{d_bk, 0, 1};
    a.q_sphum = PressureVarsPlaneView{d_qs, isd, jsd, ni_h, nj_h};
    a.cappa = PressureVarsPlaneView{d_cappa, isd, jsd, ni_h, nj_h};
    a.is = is; a.ie = ie; a.js = js; a.je = je;
    a.km = km;
    a.akap = fv_real(kAkap);
    a.rdgas = fv_real(kRdgas);
    a.grav = fv_real(kGrav);
    a.ptop = fv_real(fx.col.ptop);
    a.r_vir = fv_real(kRvir);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    fv3_gpu_error err{};
    fv3_gpu_status st = launch_remap_pressurevars(a, s, &err);
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (st != FV3_GPU_SUCCESS) {
      std::fprintf(stderr, "[%s] launch failed: %s\n", fx.name, err.message);
      nfail++;
      cudaStreamDestroy(s);
      cudaFree(d_pe); cudaFree(d_peln); cudaFree(d_pk); cudaFree(d_delp);
      cudaFree(d_pkz); cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_ak);
      cudaFree(d_bk); cudaFree(d_qs); cudaFree(d_cappa);
      continue;
    }

    std::vector<fv_real> pe_out(n_pe), peln_out(n_pe), pk_out(n_pk);
    std::vector<fv_real> delp_out(n_km), pkz_out(n_km), cappa_out(n_km);
    CUDA_CHECK(cudaMemcpy(pe_out.data(), d_pe, n_pe * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(peln_out.data(), d_peln, n_pe * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pk_out.data(), d_pk, n_pk * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delp_out.data(), d_delp, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pkz_out.data(), d_pkz, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cappa_out.data(), d_cappa, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));

    double max_abs_pe = 0, max_abs_delp = 0, max_abs_cappa = 0;      // expect bitwise (0)
    double max_rel_peln = 0, max_rel_pk = 0, max_rel_pkz = 0;        // expect round-off
    bool boundary_bitwise = true;
    bool any_nonfinite = false;

    auto rel = [](double v, double e) { return std::fabs(v - e) / std::max(1.0, std::fabs(e)); };

    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i) {
        for (int k = 0; k <= km; ++k) {
          const size_t off_pe = static_cast<size_t>(i - isd) * s0_pe +
                                static_cast<size_t>(k) * s1_pe +
                                static_cast<size_t>(j - jsd) * s2_pe;
          const size_t off_pk = static_cast<size_t>(i - isd) * s0_ijk +
                                static_cast<size_t>(j - jsd) * s1_ijk_pk +
                                static_cast<size_t>(k) * s2_ijk_pk;
          const double pe_v = double(pe_out[off_pe]);
          const double peln_v = double(peln_out[off_pe]);
          const double pk_v = double(pk_out[off_pk]);
          if (!std::isfinite(pe_v) || !std::isfinite(peln_v) || !std::isfinite(pk_v))
            any_nonfinite = true;
          max_abs_pe = std::max(max_abs_pe, std::fabs(pe_v - oracle.pe[k]));
          max_rel_peln = std::max(max_rel_peln, rel(peln_v, oracle.peln[k]));
          max_rel_pk = std::max(max_rel_pk, rel(pk_v, oracle.pk[k]));
          if (k == 0 || k == km) {
            if (peln_v != oracle.peln[k] || pk_v != oracle.pk[k]) boundary_bitwise = false;
          }
        }
        for (int k = 0; k < km; ++k) {
          const size_t off_km = static_cast<size_t>(i - isd) * s0_ijk +
                                static_cast<size_t>(j - jsd) * s1_ijk_pk +
                                static_cast<size_t>(k) * s2_ijk_pk;
          const double delp_v = double(delp_out[off_km]);
          const double pkz_v = double(pkz_out[off_km]);
          const double cappa_v = double(cappa_out[off_km]);
          if (!std::isfinite(delp_v) || !std::isfinite(pkz_v) || !std::isfinite(cappa_v))
            any_nonfinite = true;
          max_abs_delp = std::max(max_abs_delp, std::fabs(delp_v - oracle.delp[k]));
          max_abs_cappa = std::max(max_abs_cappa, std::fabs(cappa_v - oracle.cappa[k]));
          max_rel_pkz = std::max(max_rel_pkz, rel(pkz_v, oracle.pkz[k]));
        }
      }

    const double round_off_tol = 1e-12;
    const bool ok = (max_abs_pe == 0.0) && (max_abs_delp == 0.0) &&
                    (max_abs_cappa == 0.0) && boundary_bitwise &&
                    !any_nonfinite && max_rel_peln < round_off_tol &&
                    max_rel_pk < round_off_tol && max_rel_pkz < round_off_tol;
    std::printf("[%-40s] pe_abs=%.3e delp_abs=%.3e cappa_abs=%.3e "
                "peln_rel=%.3e pk_rel=%.3e pkz_rel=%.3e boundary_bitwise=%d OK=%d\n",
                fx.name, max_abs_pe, max_abs_delp, max_abs_cappa, max_rel_peln,
                max_rel_pk, max_rel_pkz, boundary_bitwise ? 1 : 0, ok ? 1 : 0);
    if (!ok) nfail++;

    cudaStreamDestroy(s);
    cudaFree(d_pe); cudaFree(d_peln); cudaFree(d_pk); cudaFree(d_delp);
    cudaFree(d_pkz); cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_ak);
    cudaFree(d_bk); cudaFree(d_qs); cudaFree(d_cappa);
  }

  std::printf("REMAP_PRESSUREVARS_PROD_SHAPE_GPU_SMOKE: %d/%zu cases failed\n",
              nfail, fixtures.size());
  return nfail == 0 ? 0 : 1;
}
