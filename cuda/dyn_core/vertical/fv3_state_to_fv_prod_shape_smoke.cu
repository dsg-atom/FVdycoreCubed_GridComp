//===========================================================================
// WP-C FV_StateMod State_To_FV production-SHAPE GPU smoke test (Prompt 99,
// Stage 9). Same broadcast/tile methodology as
// fv3_remap_pressurevars_prod_shape_smoke.cu, adapted to State_To_FV's own
// halo-free interior-only STATE%VARS%* inputs (see fv3_state_to_fv.cuh's
// header comment): builds device views for u/v/w/pe/peln/pk/delp/pt/delz/
// pkz over a realistic 3-cell-halo tile (isd=is-3, jsd=js-3, deliberately
// wider than anything the kernel dereferences), calls the real production
// launchers launch_state_to_fv_phase_a / launch_state_to_fv_phase_c
// directly (plain-Args overload, bypassing ExecutionContext/FieldRegistry
// exactly as the precedent prod-shape smoke files do), and checks the
// interior region against the exact host oracle duplicated from
// tests/host/test_state_to_fv_host_reference.cpp (Prompt 98).
//
// Same 8 named fixtures as the Prompt 98 host harness, including the
// MANDATORY sentinel_layout_transposition_check (fixture 8) that catches an
// axis-swap mistake in the pe (i,j,k)->(i,k,j) transpose.
//
// Classification (matching remap_pressurevars's precedent exactly, since
// Phase C's arithmetic shape is the same pe->peln->pk chain): u/v/w/delz/
// pkz/pe/delp/ps/pt are compared BITWISE (pure copy, transpose, subtract,
// or single multiply -- no transcendentals); peln (fv3_log) and pk
// (fv3_exp) are compared with a tight ROUND-OFF tolerance, since CUDA
// device log/exp and host libm log/exp are not guaranteed bit-identical.
//
// GPU-execution + tolerance-comparison check only; not registered as a
// ctest (requires an actual GPU) -- run manually via srun, ideally under
// compute-sanitizer memcheck.
//===========================================================================

#include "fv3_state_to_fv.cuh"
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
// test_state_to_fv_host_reference.cpp bit-for-bit) -- standalone GPU smoke
// binary, no header-level export exists to share this from.
// ---------------------------------------------------------------------------
constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kAkap = kRdgas / kCpAir;

struct StateIn {
  int ni, nj, npz;
  std::vector<double> u, v, w, pt, delz, pkz; // (i,j,k) order, ni*nj*npz
  std::vector<double> pe_ijk;                 // (i,j,k) order, ni*nj*(npz+1)
};

struct FvOut {
  int ni, nj, npz;
  std::vector<double> u, v, w, pt, delz, pkz;
  std::vector<double> pe_ikj;
  std::vector<double> peln_can, pk_can;
  std::vector<double> ps;
  std::vector<double> delp;
};

inline int idx_ijk(int i, int j, int k, int nj, int npz) { return (i * nj + j) * npz + k; }
inline int idx_ikj(int i, int k, int j, int npzp1, int nj) { return (i * npzp1 + k) * nj + j; }
inline int idx_ij(int i, int j, int nj) { return i * nj + j; }

FvOut state_to_fv_host(const StateIn &s) {
  const int ni = s.ni, nj = s.nj, npz = s.npz, npzp1 = npz + 1;
  FvOut out;
  out.ni = ni; out.nj = nj; out.npz = npz;
  out.u.resize(ni * nj * npz);
  out.v.resize(ni * nj * npz);
  out.w.resize(ni * nj * npz);
  out.pt.resize(ni * nj * npz);
  out.delz.resize(ni * nj * npz);
  out.pkz.resize(ni * nj * npz);
  out.pe_ikj.resize(ni * npzp1 * nj);
  out.peln_can.resize(ni * nj * npzp1);
  out.pk_can.resize(ni * nj * npzp1);
  out.ps.resize(ni * nj);
  out.delp.resize(ni * nj * npz);

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.u[d] = s.u[d];
        out.v[d] = s.v[d];
        out.w[d] = s.w[d];
      }

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k)
        out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)] = s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)];

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        const int d = idx_ijk(i, j, k, nj, npzp1);
        const double pe_val = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        out.peln_can[d] = std::log(pe_val);
        out.pk_can[d] = std::exp(kAkap * out.peln_can[d]);
      }

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      out.ps[idx_ij(i, j, nj)] = out.pe_ikj[idx_ikj(i, npz, j, npzp1, nj)];

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const double pe_k = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        const double pe_kp1 = out.pe_ikj[idx_ikj(i, k + 1, j, npzp1, nj)];
        out.delp[idx_ijk(i, j, k, nj, npz)] = pe_kp1 - pe_k;
      }

  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.pt[d] = s.pt[d] * s.pkz[d];
        out.delz[d] = s.delz[d];
        out.pkz[d] = s.pkz[d];
      }

  return out;
}

StateIn make_state(int ni, int nj, int npz,
                    double (*u_fn)(int, int, int),
                    double (*v_fn)(int, int, int),
                    double (*w_fn)(int, int, int),
                    double (*pt_fn)(int, int, int),
                    double (*delz_fn)(int, int, int),
                    double (*pkz_fn)(int, int, int),
                    double (*pe_fn)(int, int, int)) {
  StateIn s;
  s.ni = ni; s.nj = nj; s.npz = npz;
  const int npzp1 = npz + 1;
  s.u.resize(ni * nj * npz); s.v.resize(ni * nj * npz); s.w.resize(ni * nj * npz);
  s.pt.resize(ni * nj * npz); s.delz.resize(ni * nj * npz); s.pkz.resize(ni * nj * npz);
  s.pe_ijk.resize(ni * nj * npzp1);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        s.u[d] = u_fn(i, j, k);
        s.v[d] = v_fn(i, j, k);
        s.w[d] = w_fn(i, j, k);
        s.pt[d] = pt_fn(i, j, k);
        s.delz[d] = delz_fn(i, j, k);
        s.pkz[d] = pkz_fn(i, j, k);
      }
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k)
        s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)] = pe_fn(i, j, k);
  return s;
}

double smooth_uv(int i, int j, int k) { return std::sin(0.3 * i) * std::cos(0.2 * j) + 0.01 * k; }
double smooth_w(int i, int j, int k) { return 0.05 * std::sin(0.4 * (i + j + k)); }
double vertical_profile_pt(int i, int j, int k) { return 300.0 - 4.0 * k; }
double vertical_profile_delz(int i, int j, int k) { return -500.0 * std::exp(-0.1 * k); }
double horiz_gradient_pt(int i, int j, int k) { return 260.0 + 2.0 * i + 3.0 * j; }
double horiz_gradient_delz(int i, int j, int) { return -300.0 - 10.0 * i; }
double pkz_reasonable(int i, int j, int k) { return 0.85 + 0.001 * (i + j) - 0.01 * k; }
double zero_field(int, int, int) { return 0.0; }
double pe_hybrid(int i, int j, int k) {
  const double ptop = 1.0, ps = 101325.0 + 50.0 * i - 30.0 * j;
  const double frac = static_cast<double>(k) / 10.0;
  return ptop + frac * frac * (ps - ptop);
}

struct Fixture { const char *name; StateIn s; };

std::vector<Fixture> build_fixtures(int ni, int nj, int npz) {
  std::vector<Fixture> f;
  f.push_back({"smooth_3d_field",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"varying_vertical_profile",
      make_state(ni, nj, npz, zero_field, zero_field, zero_field,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"horizontal_gradient",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, horiz_gradient_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"boundary_columns_no_special_path",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, horiz_gradient_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"uvw_active_nonhydrostatic",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"pt_delz_pkz_strong_contrast",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid)});
  f.push_back({"production_shaped_reduced_dims",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid)});
  return f;
}

// MANDATORY sentinel fixture (Prompt 98 fixture 10 / Stage 9 requirement):
// pe_src(i,j,k) is a bijective encoding of (i,j,k) so an axis-swap mistake
// in the pe transpose is detectable.
StateIn make_sentinel(int ni, int nj, int npz) {
  StateIn s;
  s.ni = ni; s.nj = nj; s.npz = npz;
  const int npzp1 = npz + 1;
  s.u.resize(ni * nj * npz); s.v.resize(ni * nj * npz); s.w.resize(ni * nj * npz);
  s.pt.resize(ni * nj * npz); s.delz.resize(ni * nj * npz); s.pkz.resize(ni * nj * npz);
  s.pe_ijk.resize(ni * nj * npzp1);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        s.u[d] = 1000.0 * i + 10.0 * j + 0.1 * k;
        s.v[d] = 2000.0 * i + 20.0 * j + 0.2 * k;
        s.w[d] = 3000.0 * i + 30.0 * j + 0.3 * k;
        s.delz[d] = -400.0 - 0.001 * (100.0 * i + 10.0 * j + k);
        s.pkz[d] = 0.9 + 0.0001 * (100.0 * i + 10.0 * j + k);
        s.pt[d] = 280.0 + 0.001 * (100.0 * i + 10.0 * j + k);
      }
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        const double frac = static_cast<double>(k) / npz;
        s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)] =
            1.0 + frac * frac * (100000.0 + 1000.0 * i + 10.0 * j);
      }
  return s;
}

} // namespace

int main() {
  // Production-shaped tile matching the Prompt 98 host harness dims
  // exactly (ni=6, nj=5, npz=10 -- reduced from the confirmed 512x512 C512
  // L10 per-rank interior, npz=10 matches C512 L10 exactly), 3-cell halo
  // (deliberately wider than anything the kernel dereferences).
  const int64_t is = 1, ie = 6, js = 1, je = 5;
  const int64_t isd = is - 3, jsd = js - 3;
  const int64_t ni_h = ie - isd + 1, nj_h = je - jsd + 1; // 9, 8
  const int ni = static_cast<int>(ie - is + 1), nj = static_cast<int>(je - js + 1);
  const int npz = 10;

  auto fixtures = build_fixtures(ni, nj, npz);
  fixtures.push_back({"sentinel_layout_transposition_check", make_sentinel(ni, nj, npz)});

  int nfail = 0;

  for (const auto &fx : fixtures) {
    const StateIn &s = fx.s;
    const FvOut oracle = state_to_fv_host(s);
    const int64_t nk_pe = npz + 1;

    // FV_Atm-side (halo-carrying) layouts: pe/peln (i,k,j); everything else
    // (u,v,w,pk,delp,pt,delz,pkz) natural (i,j,k) -- same convention
    // remap_pressurevars_prod_shape_smoke.cu uses; internally consistent
    // for this standalone smoke test regardless of the real FieldRegistry
    // strides, since every .at() call here goes through the same view.
    const int64_t s0_pe = 1, s1_pe = ni_h, s2_pe = ni_h * nk_pe;
    const size_t n_pe = static_cast<size_t>(ni_h) * nk_pe * nj_h;
    const int64_t s0 = 1, s1 = ni_h, s2 = ni_h * nj_h;
    const size_t n_km = static_cast<size_t>(ni_h) * nj_h * npz;

    // pk is registered natural (i,j,k) (k LAST), unlike pe/peln's (i,k,j) --
    // give it its own natural-layout strides sized to km+1 levels so the
    // kernel's a.pk.at(i,j,k) and this test's readback use the same layout.
    const int64_t s0_pk = 1, s1_pk = ni_h, s2_pk = ni_h * nj_h;

    std::vector<fv_real> u_h(n_km, fv_real(0)), v_h(n_km, fv_real(0)), w_h(n_km, fv_real(0));
    std::vector<fv_real> pe_h(n_pe, fv_real(0)), peln_h(n_pe, fv_real(0)), pk_h(n_pe, fv_real(0));
    std::vector<fv_real> delp_h(n_km, fv_real(0)), pt_h(n_km, fv_real(0));
    std::vector<fv_real> delz_h(n_km, fv_real(0)), pkz_h(n_km, fv_real(0));

    std::vector<fv_real> state_u_h(size_t(ni) * nj * npz), state_v_h(size_t(ni) * nj * npz);
    std::vector<fv_real> state_w_h(size_t(ni) * nj * npz), state_pt_h(size_t(ni) * nj * npz);
    std::vector<fv_real> state_pkz_h(size_t(ni) * nj * npz), state_dz_h(size_t(ni) * nj * npz);
    std::vector<fv_real> state_pe_h(size_t(ni) * nj * (npz + 1));
    // StateIn's own arrays (s.u/v/w/pt/pkz/delz/pe_ijk) are laid out
    // k-fastest via idx_ijk ((i*nj+j)*npz+k) -- the oracle's convenient
    // internal convention. StateInputView (fv3_state_to_fv.cuh) instead
    // expects i-fastest: p[(i-is)+(j-js)*ni+k*ni*nj]. Repack rather than
    // assume the two orderings coincide.
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < npz; ++k) {
          const int src = idx_ijk(i, j, k, nj, npz);
          const size_t dst = static_cast<size_t>(i) + static_cast<size_t>(j) * ni +
                              static_cast<size_t>(k) * ni * nj;
          state_u_h[dst] = fv_real(s.u[src]);
          state_v_h[dst] = fv_real(s.v[src]);
          state_w_h[dst] = fv_real(s.w[src]);
          state_pt_h[dst] = fv_real(s.pt[src]);
          state_pkz_h[dst] = fv_real(s.pkz[src]);
          state_dz_h[dst] = fv_real(s.delz[src]);
        }
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k <= npz; ++k) {
          const int src = idx_ijk(i, j, k, nj, npz + 1);
          const size_t dst = static_cast<size_t>(i) + static_cast<size_t>(j) * ni +
                              static_cast<size_t>(k) * ni * nj;
          state_pe_h[dst] = fv_real(s.pe_ijk[src]);
        }

    fv_real *d_u, *d_v, *d_w, *d_pe, *d_peln, *d_pk, *d_delp, *d_pt, *d_delz, *d_pkz, *d_ps;
    fv_real *d_su, *d_sv, *d_sw, *d_spt, *d_spkz, *d_sdz, *d_spe;
    CUDA_CHECK(cudaMalloc(&d_u, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_v, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_w, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pe, n_pe * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_peln, n_pe * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pk, n_pe * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delp, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pt, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_delz, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_pkz, n_km * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_ps, static_cast<size_t>(ni) * nj * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_su, state_u_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_sv, state_v_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_sw, state_w_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_spt, state_pt_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_spkz, state_pkz_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_sdz, state_dz_h.size() * sizeof(fv_real)));
    CUDA_CHECK(cudaMalloc(&d_spe, state_pe_h.size() * sizeof(fv_real)));

    CUDA_CHECK(cudaMemcpy(d_u, u_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pe, pe_h.data(), n_pe * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_peln, peln_h.data(), n_pe * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pk, pk_h.data(), n_pe * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delp, delp_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt, pt_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delz, delz_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pkz, pkz_h.data(), n_km * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_su, state_u_h.data(), state_u_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sv, state_v_h.data(), state_v_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sw, state_w_h.data(), state_w_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spt, state_pt_h.data(), state_pt_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spkz, state_pkz_h.data(), state_pkz_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sdz, state_dz_h.data(), state_dz_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spe, state_pe_h.data(), state_pe_h.size() * sizeof(fv_real), cudaMemcpyHostToDevice));

    StateToFvPhaseAArgs pa{};
    pa.u = RiemViewC3{d_u, isd, jsd, 0, s0, s1, s2};
    pa.v = RiemViewC3{d_v, isd, jsd, 0, s0, s1, s2};
    pa.state_u = StateInputView{d_su, is, js, ni, nj};
    pa.state_v = StateInputView{d_sv, is, js, ni, nj};
    pa.is = is; pa.ie = ie; pa.js = js; pa.je = je;
    pa.km = npz;

    StateToFvPhaseCArgs pc{};
    pc.w = RiemViewC3{d_w, isd, jsd, 0, s0, s1, s2};
    pc.pe = RiemViewC3{d_pe, isd, 0, jsd, s0_pe, s1_pe, s2_pe};
    pc.peln = RiemViewC3{d_peln, isd, 0, jsd, s0_pe, s1_pe, s2_pe};
    pc.pk = RiemViewC3{d_pk, isd, jsd, 0, s0_pk, s1_pk, s2_pk};
    pc.delp = RiemViewC3{d_delp, isd, jsd, 0, s0, s1, s2};
    pc.pt = RiemViewC3{d_pt, isd, jsd, 0, s0, s1, s2};
    pc.delz = RiemViewC3{d_delz, isd, jsd, 0, s0, s1, s2};
    pc.pkz = RiemViewC3{d_pkz, isd, jsd, 0, s0, s1, s2};
    pc.state_w = StateInputView{d_sw, is, js, ni, nj};
    pc.state_pe = StateInputView{d_spe, is, js, ni, nj};
    pc.state_pt = StateInputView{d_spt, is, js, ni, nj};
    pc.state_pkz = StateInputView{d_spkz, is, js, ni, nj};
    pc.state_dz = StateInputView{d_sdz, is, js, ni, nj};
    pc.ps_out = PsPackView{d_ps, is, js, ni};
    pc.is = is; pc.ie = ie; pc.js = js; pc.je = je;
    pc.km = npz;
    pc.akap = fv_real(kAkap);

    cudaStream_t s_stream;
    CUDA_CHECK(cudaStreamCreate(&s_stream));
    fv3_gpu_error err{};
    fv3_gpu_status sta = launch_state_to_fv_phase_a(pa, s_stream, &err);
    fv3_gpu_status stc = FV3_GPU_SUCCESS;
    if (sta == FV3_GPU_SUCCESS) stc = launch_state_to_fv_phase_c(pc, s_stream, &err);
    CUDA_CHECK(cudaStreamSynchronize(s_stream));

    if (sta != FV3_GPU_SUCCESS || stc != FV3_GPU_SUCCESS) {
      std::fprintf(stderr, "[%s] launch failed: %s\n", fx.name, err.message);
      nfail++;
      cudaStreamDestroy(s_stream);
      cudaFree(d_u); cudaFree(d_v); cudaFree(d_w); cudaFree(d_pe); cudaFree(d_peln);
      cudaFree(d_pk); cudaFree(d_delp); cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_pkz);
      cudaFree(d_ps); cudaFree(d_su); cudaFree(d_sv); cudaFree(d_sw); cudaFree(d_spt);
      cudaFree(d_spkz); cudaFree(d_sdz); cudaFree(d_spe);
      continue;
    }

    std::vector<fv_real> u_out(n_km), v_out(n_km), w_out(n_km);
    std::vector<fv_real> pe_out(n_pe), peln_out(n_pe), pk_out(n_pe);
    std::vector<fv_real> delp_out(n_km), pt_out(n_km), delz_out(n_km), pkz_out(n_km);
    std::vector<fv_real> ps_out(static_cast<size_t>(ni) * nj);
    CUDA_CHECK(cudaMemcpy(u_out.data(), d_u, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_out.data(), d_v, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w_out.data(), d_w, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pe_out.data(), d_pe, n_pe * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(peln_out.data(), d_peln, n_pe * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pk_out.data(), d_pk, n_pe * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delp_out.data(), d_delp, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pt_out.data(), d_pt, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(delz_out.data(), d_delz, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pkz_out.data(), d_pkz, n_km * sizeof(fv_real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ps_out.data(), d_ps, ps_out.size() * sizeof(fv_real), cudaMemcpyDeviceToHost));

    double max_abs_u = 0, max_abs_v = 0, max_abs_w = 0, max_abs_delz = 0;
    double max_abs_pe = 0, max_abs_delp = 0, max_abs_pt = 0, max_abs_pkz_copy = 0, max_abs_ps = 0;
    double max_rel_peln = 0, max_rel_pk = 0;
    bool any_nonfinite = false;
    auto rel = [](double v, double e) { return std::fabs(v - e) / std::max(1.0, std::fabs(e)); };

    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j) {
        const int64_t gi = is + i, gj = js + j;
        for (int k = 0; k < npz; ++k) {
          const size_t off = static_cast<size_t>(gi - isd) * s0 + static_cast<size_t>(gj - jsd) * s1 +
                              static_cast<size_t>(k) * s2;
          const int d = idx_ijk(i, j, k, nj, npz);
          const double u_v = double(u_out[off]), v_v = double(v_out[off]), w_v = double(w_out[off]);
          const double delz_v = double(delz_out[off]), pt_v = double(pt_out[off]), pkz_v = double(pkz_out[off]);
          if (!std::isfinite(u_v) || !std::isfinite(v_v) || !std::isfinite(w_v) ||
              !std::isfinite(delz_v) || !std::isfinite(pt_v) || !std::isfinite(pkz_v))
            any_nonfinite = true;
          max_abs_u = std::max(max_abs_u, std::fabs(u_v - oracle.u[d]));
          max_abs_v = std::max(max_abs_v, std::fabs(v_v - oracle.v[d]));
          max_abs_w = std::max(max_abs_w, std::fabs(w_v - oracle.w[d]));
          max_abs_delz = std::max(max_abs_delz, std::fabs(delz_v - oracle.delz[d]));
          max_abs_pt = std::max(max_abs_pt, std::fabs(pt_v - oracle.pt[d]));
          max_abs_pkz_copy = std::max(max_abs_pkz_copy, std::fabs(pkz_v - oracle.pkz[d]));
          const double delp_v = double(delp_out[off]);
          if (!std::isfinite(delp_v)) any_nonfinite = true;
          max_abs_delp = std::max(max_abs_delp, std::fabs(delp_v - oracle.delp[d]));
        }
        for (int k = 0; k <= npz; ++k) {
          const size_t off_pe = static_cast<size_t>(gi - isd) * s0_pe + static_cast<size_t>(k) * s1_pe +
                                static_cast<size_t>(gj - jsd) * s2_pe;
          // pk is registered natural (i,j,k), a different layout than pe/peln's
          // (i,k,j) -- see the s0_pk/s1_pk/s2_pk view construction above.
          const size_t off_pk = static_cast<size_t>(gi - isd) * s0_pk + static_cast<size_t>(gj - jsd) * s1_pk +
                                static_cast<size_t>(k) * s2_pk;
          const int d = idx_ijk(i, j, k, nj, npz + 1);
          const double pe_v = double(pe_out[off_pe]), peln_v = double(peln_out[off_pe]), pk_v = double(pk_out[off_pk]);
          if (!std::isfinite(pe_v) || !std::isfinite(peln_v) || !std::isfinite(pk_v)) any_nonfinite = true;
          const double pe_oracle = oracle.pe_ikj[idx_ikj(i, k, j, npz + 1, nj)];
          max_abs_pe = std::max(max_abs_pe, std::fabs(pe_v - pe_oracle));
          max_rel_peln = std::max(max_rel_peln, rel(peln_v, oracle.peln_can[d]));
          max_rel_pk = std::max(max_rel_pk, rel(pk_v, oracle.pk_can[d]));
        }
        // ps_out follows PsPackView's i-fastest packing (i + j*ni), not the
        // oracle's own j-fastest idx_ij(i,j,nj) convention -- extract with
        // the matching formula for each side.
        const double ps_v = double(ps_out[static_cast<size_t>(i) + static_cast<size_t>(j) * ni]);
        if (!std::isfinite(ps_v)) any_nonfinite = true;
        max_abs_ps = std::max(max_abs_ps, std::fabs(ps_v - oracle.ps[idx_ij(i, j, nj)]));
      }

    const double round_off_tol = 1e-12;
    const bool ok = (max_abs_u == 0.0) && (max_abs_v == 0.0) && (max_abs_w == 0.0) &&
                    (max_abs_delz == 0.0) && (max_abs_pkz_copy == 0.0) && (max_abs_pt == 0.0) &&
                    (max_abs_pe == 0.0) && (max_abs_delp == 0.0) && (max_abs_ps == 0.0) &&
                    !any_nonfinite && max_rel_peln < round_off_tol && max_rel_pk < round_off_tol;
    std::printf("[%-42s] u_abs=%.3e v_abs=%.3e w_abs=%.3e pe_abs=%.3e delp_abs=%.3e "
                "pt_abs=%.3e ps_abs=%.3e peln_rel=%.3e pk_rel=%.3e OK=%d\n",
                fx.name, max_abs_u, max_abs_v, max_abs_w, max_abs_pe, max_abs_delp,
                max_abs_pt, max_abs_ps, max_rel_peln, max_rel_pk, ok ? 1 : 0);
    if (!ok) nfail++;

    cudaStreamDestroy(s_stream);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w); cudaFree(d_pe); cudaFree(d_peln);
    cudaFree(d_pk); cudaFree(d_delp); cudaFree(d_pt); cudaFree(d_delz); cudaFree(d_pkz);
    cudaFree(d_ps); cudaFree(d_su); cudaFree(d_sv); cudaFree(d_sw); cudaFree(d_spt);
    cudaFree(d_spkz); cudaFree(d_sdz); cudaFree(d_spe);
  }

  std::printf("STATE_TO_FV_PROD_SHAPE_GPU_SMOKE: %d/%zu cases failed\n", nfail, fixtures.size());
  return nfail == 0 ? 0 : 1;
}
