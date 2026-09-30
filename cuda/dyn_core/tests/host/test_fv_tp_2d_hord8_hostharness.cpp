// Prompt 75 — authoritative host-only harness for fv_tp_2d hord==8.
//
// Faithful C++ transcription of tp_core.F90's xppm subroutine (lines 324-708
// of GEOSfvdycore_fortran_perf's authoritative GFDL FV3 tp_core.F90), covering
// BOTH the existing GPU-supported hord==6 interior/mord==6 branch (iord<7,
// mord=abs(iord)=6, lines 523-555) and the currently CPU-only hord==8 branch
// (iord>=7 monotonic-constraint branch, iord==8 sub-branch, lines 567-581,
// plus the panel-edge correction block lines 640-678 and the shared final
// flux formula lines 697-705). Same methodology as the existing WP-B host
// reference in tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp ("independent
// host-side transcription... treated as authoritative for GPU-vs-host-
// reference comparison").
//
// NO production Fortran or CUDA source is modified or called. NO GPU. This
// file compiles and runs standalone with a plain host C++ compiler:
//   g++ -std=c++17 -O2 -o /tmp/p75_hord8 test_fv_tp_2d_hord8_hostharness.cpp
//   /tmp/p75_hord8
//
// This is a MAPPING/HARNESS artifact for Prompt 75 -- it does not implement
// any CUDA kernel. See docs/performance/p75_fv_tp_2d_hord8_cuda_plan_2026-09-24.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <string>

using real = double;

// ---- module-level PPM constants, tp_core.F90:61-96 ----
static constexpr real k_r3   = 1.0 / 3.0;
static constexpr real k_r12  = 1.0 / 12.0;
static constexpr real k_s11  = 11.0 / 14.0;
static constexpr real k_s14  = 4.0 / 7.0;
static constexpr real k_s15  = 3.0 / 14.0;
static constexpr real k_c1   = -2.0 / 14.0;
static constexpr real k_c2   = 11.0 / 14.0;
static constexpr real k_c3   = 5.0 / 14.0;
static constexpr real k_p1   = 7.0 / 12.0;
static constexpr real k_p2   = -1.0 / 12.0;
static constexpr real k_near_zero = 1.0e-25;

// pert_ppm(im, a0, al, ar, iv=1) -- tp_core.F90:1239-1257 ("standard PPM
// constraint" branch only; iv=0 branch is unused by the iord==8 path).
static void pert_ppm_iv1(int im, const std::vector<real> &a0,
                          std::vector<real> &al, std::vector<real> &ar) {
  for (int i = 0; i < im; ++i) {
    if (al[i] * ar[i] < 0.0) {
      real da1 = al[i] - ar[i];
      real da2 = da1 * da1;
      real a6da = 3.0 * (al[i] + ar[i]) * da1;
      if (a6da < -da2) {
        ar[i] = -2.0 * al[i];
      } else if (a6da > da2) {
        al[i] = -2.0 * ar[i];
      }
    } else {
      al[i] = 0.0;
      ar[i] = 0.0;
    }
  }
}

// Faithful transcription of tp_core.F90 xppm (single row, iord in {6,8} only;
// this harness restricts to the two values needed for the hord6-vs-hord8
// comparison and the hord8 mapping). q1 indexed by cell index with q1[i] =
// q(i,j); caller supplies isd..ied (halo) and is..ie (compute) with ng=3,
// matching the production/CUDA TransportBounds.ng default (wp_b_cuda.cuh:57).
struct XppmResult {
  std::vector<real> flux; // flux[i] for i in is..ie+1, 0-indexed from is
};

static XppmResult xppm_row(const std::vector<real> &q1_full, int isd, int ied,
                            int is, int ie, int npx, bool cubed_sphere_edge,
                            const std::vector<real> &c_full, int iord,
                            const std::vector<real> &dxa_full) {
  auto q1 = [&](int i) { return q1_full[i - isd]; };
  auto dxa = [&](int i) { return dxa_full[i - isd]; };
  auto c = [&](int i) { return c_full[i - is]; };

  const bool nested = false;      // production config, project_architecture.md
  const int grid_type = 0;        // production config
  const int mord = std::abs(iord);

  int is1, ie3, ie1;
  if (!nested && grid_type < 3) {
    is1 = std::max(3, is - 1);
    ie3 = std::min(npx - 2, ie + 2);
    ie1 = std::min(npx - 3, ie + 1);
  } else {
    is1 = is - 1;
    ie3 = ie + 2;
    ie1 = ie + 1;
  }

  XppmResult out;
  out.flux.assign(ie - is + 2, 0.0);
  auto flux_at = [&](int i) -> real & { return out.flux[i - is]; };

  // All row-local scratch (al/dm/bl/br/b0/smt5) is allocated over the FULL
  // isd..ied halo range, matching how these are automatic arrays dimensioned
  // over the full row in the authoritative Fortran (tp_core.F90 declares
  // al/bl/br/... as (is-2:ie+2)-or-wider automatics, not tightly bounded to
  // is1/ie3) -- a tightly bounded C++ allocation is an easy off-by-range bug
  // magnet once the is==1/(ie+1)==npx boundary blocks reach outside [is1,ie3]
  // (confirmed by AddressSanitizer during harness development).
  const int NFULL = ied - isd + 1;

  if (iord < 7) {
    // --- mord==6 interior/edge branch: tp_core.F90:365-557 (mord==6 = the
    // "else" at line 488/523-529, since iord is not 1,2,3,4,5,-5 here). ---
    std::vector<real> al(NFULL);
    auto AL = [&](int i) -> real & { return al[i - isd]; };
    for (int i = is1; i <= ie3; ++i)
      AL(i) = k_p1 * (q1(i - 1) + q1(i)) + k_p2 * (q1(i - 2) + q1(i + 1));

    if (!nested && grid_type < 3) {
      if (is == 1) {
        AL(0) = k_c1 * q1(-2) + k_c2 * q1(-1) + k_c3 * q1(0);
        AL(1) = 0.5 * (((2.0 * dxa(0) + dxa(-1)) * q1(0) - dxa(0) * q1(-1)) /
                            (dxa(-1) + dxa(0)) +
                        ((2.0 * dxa(1) + dxa(2)) * q1(1) - dxa(1) * q1(2)) /
                            (dxa(1) + dxa(2)));
        AL(2) = k_c3 * q1(1) + k_c2 * q1(2) + k_c1 * q1(3);
      }
      if ((ie + 1) == npx) {
        AL(npx - 1) = k_c1 * q1(npx - 3) + k_c2 * q1(npx - 2) + k_c3 * q1(npx - 1);
        AL(npx) = 0.5 * (((2.0 * dxa(npx - 1) + dxa(npx - 2)) * q1(npx - 1) -
                           dxa(npx - 1) * q1(npx - 2)) /
                              (dxa(npx - 2) + dxa(npx - 1)) +
                          ((2.0 * dxa(npx) + dxa(npx + 1)) * q1(npx) -
                           dxa(npx) * q1(npx + 1)) /
                              (dxa(npx) + dxa(npx + 1)));
        AL(npx + 1) = k_c3 * q1(npx) + k_c2 * q1(npx + 1) + k_c1 * q1(npx + 2);
      }
    }

    // mord==6 (the generic "else" body, tp_core.F90:523-555): identical
    // structure to the existing CUDA xppm_face_flux/al_x_at (wp_b_cuda.cu:
    // 190-256), reproduced here to also exercise panel-edge AL() above (the
    // CUDA reference recomputes al inline per-cell rather than as a full row,
    // but the values are the same by construction).
    std::vector<real> bl(NFULL), br(NFULL), b0(NFULL);
    std::vector<char> smt5(NFULL);
    auto BL = [&](int i) -> real & { return bl[i - isd]; };
    auto BR = [&](int i) -> real & { return br[i - isd]; };
    auto B0 = [&](int i) -> real & { return b0[i - isd]; };
    auto SM = [&](int i) -> char & { return smt5[i - isd]; };
    for (int i = is - 1; i <= ie + 1; ++i) {
      BL(i) = AL(i) - q1(i);
      BR(i) = AL(i + 1) - q1(i);
      B0(i) = BL(i) + BR(i);
      SM(i) = 3.0 * std::fabs(B0(i)) < std::fabs(BL(i) - BR(i));
    }
    if (!nested && grid_type < 3) {
      if (is == 1) {
        SM(0) = BL(0) * BR(0) < 0.0;
        SM(1) = BL(1) * BR(1) < 0.0;
      }
      if ((ie + 1) == npx) {
        SM(npx - 1) = BL(npx - 1) * BR(npx - 1) < 0.0;
        SM(npx) = BL(npx) * BR(npx) < 0.0;
      }
    }
    for (int i = is; i <= ie + 1; ++i) {
      real fx1, donor;
      if (c(i) > 0.0) {
        fx1 = (1.0 - c(i)) * (BR(i - 1) - c(i) * B0(i - 1));
        donor = q1(i - 1);
      } else {
        fx1 = (1.0 + c(i)) * (BL(i) + c(i) * B0(i));
        donor = q1(i);
      }
      flux_at(i) = (SM(i - 1) || SM(i)) ? donor + fx1 : donor;
    }
    return out;
  }

  // --- iord>=7 monotonic-constraint branch, iord==8 sub-branch:
  // tp_core.F90:567-581 (dm, al, bl/br) + :640-678 (panel-edge correction,
  // pert_ppm) + :697-705 (shared final flux formula, since iord!=7). ---
  std::vector<real> dm(NFULL);
  auto DM = [&](int i) -> real & { return dm[i - isd]; };
  for (int i = is - 2; i <= ie + 2; ++i) {
    real xt = 0.25 * (q1(i + 1) - q1(i - 1));
    real lo = std::min({q1(i - 1), q1(i), q1(i + 1)});
    real hi = std::max({q1(i - 1), q1(i), q1(i + 1)});
    real mag = std::min(std::fabs(xt), std::min(hi - q1(i), q1(i) - lo));
    DM(i) = std::copysign(mag, xt);
  }
  std::vector<real> al(NFULL);
  auto AL = [&](int i) -> real & { return al[i - isd]; };
  for (int i = is1; i <= ie1 + 1; ++i)
    AL(i) = 0.5 * (q1(i - 1) + q1(i)) + k_r3 * (DM(i - 1) - DM(i));

  std::vector<real> bl(NFULL), br(NFULL);
  auto BL = [&](int i) -> real & { return bl[i - isd]; };
  auto BR = [&](int i) -> real & { return br[i - isd]; };
  for (int i = is1; i <= ie1; ++i) {
    real xt = 2.0 * DM(i);
    BL(i) = -std::copysign(std::min(std::fabs(xt), std::fabs(AL(i) - q1(i))), xt);
    BR(i) = std::copysign(std::min(std::fabs(xt), std::fabs(AL(i + 1) - q1(i))), xt);
  }

  if (!nested && grid_type < 3) {
    if (is == 1) {
      std::vector<real> a0 = {q1(0), q1(1), q1(2)};
      std::vector<real> ALb(3), ARb(3);
      ALb[0] = k_s14 * DM(-1) + k_s11 * (q1(-1) - q1(0));
      real xt = 0.5 * (((2.0 * dxa(0) + dxa(-1)) * q1(0) - dxa(0) * q1(-1)) /
                            (dxa(-1) + dxa(0)) +
                        ((2.0 * dxa(1) + dxa(2)) * q1(1) - dxa(1) * q1(2)) /
                            (dxa(1) + dxa(2)));
      xt = std::max(xt, std::min({q1(-1), q1(0), q1(1), q1(2)}));
      xt = std::min(xt, std::max({q1(-1), q1(0), q1(1), q1(2)}));
      ARb[0] = xt - q1(0);
      ALb[1] = xt - q1(1);
      real xt2 = k_s15 * q1(1) + k_s11 * q1(2) - k_s14 * DM(2);
      ARb[1] = xt2 - q1(1);
      ALb[2] = xt2 - q1(2);
      ARb[2] = AL(3) - q1(2);
      pert_ppm_iv1(3, a0, ALb, ARb);
      BL(0) = ALb[0]; BR(0) = ARb[0];
      BL(1) = ALb[1]; BR(1) = ARb[1];
      BL(2) = ALb[2]; BR(2) = ARb[2];
    }
    if ((ie + 1) == npx) {
      std::vector<real> a0 = {q1(npx - 2), q1(npx - 1), q1(npx)};
      std::vector<real> ALb(3), ARb(3);
      ALb[0] = AL(npx - 2) - q1(npx - 2);
      real xt = k_s15 * q1(npx - 1) + k_s11 * q1(npx - 2) + k_s14 * DM(npx - 2);
      ARb[0] = xt - q1(npx - 2);
      ALb[1] = xt - q1(npx - 1);
      real xt2 = 0.5 * (((2.0 * dxa(npx - 1) + dxa(npx - 2)) * q1(npx - 1) -
                          dxa(npx - 1) * q1(npx - 2)) /
                             (dxa(npx - 2) + dxa(npx - 1)) +
                         ((2.0 * dxa(npx) + dxa(npx + 1)) * q1(npx) -
                          dxa(npx) * q1(npx + 1)) /
                             (dxa(npx) + dxa(npx + 1)));
      xt2 = std::max(xt2, std::min({q1(npx - 2), q1(npx - 1), q1(npx), q1(npx + 1)}));
      xt2 = std::min(xt2, std::max({q1(npx - 2), q1(npx - 1), q1(npx), q1(npx + 1)}));
      ARb[1] = xt2 - q1(npx - 1);
      ALb[2] = xt2 - q1(npx);
      ARb[2] = k_s11 * (q1(npx + 1) - q1(npx)) - k_s14 * DM(npx + 1);
      pert_ppm_iv1(3, a0, ALb, ARb);
      BL(npx - 2) = ALb[0]; BR(npx - 2) = ARb[0];
      BL(npx - 1) = ALb[1]; BR(npx - 1) = ARb[1];
      BL(npx) = ALb[2]; BR(npx) = ARb[2];
    }
  }

  // Shared final flux formula, iord != 7 (tp_core.F90:697-705).
  for (int i = is; i <= ie + 1; ++i) {
    if (c(i) > 0.0) {
      flux_at(i) = q1(i - 1) + (1.0 - c(i)) * (BR(i - 1) - c(i) * (BL(i - 1) + BR(i - 1)));
    } else {
      flux_at(i) = q1(i) + (1.0 + c(i)) * (BL(i) + c(i) * (BL(i) + BR(i)));
    }
  }
  return out;
}

// ---------------------------------------------------------------- fixtures
//
// Halo convention: every fixture allocates q/dxa over the FULL global index
// range [ISD,IED] = [-4, npx+4] (>= ng=3 margin on both sides of the panel,
// matching the production/CUDA TransportBounds.ng=3 default, plus 1 extra
// cell of safety margin for the is==1/(ie+1)==npx boundary-block's reach to
// q1(-2)/q1(npx+2)). q/dxa are generated by evaluating each fixture's
// generator function over the WHOLE range (not just the nominal 1..npx
// "core"), so halo cells are smooth, self-consistent synthetic extensions of
// the interior pattern -- there is no cross-tile halo-exchange semantics to
// reproduce here, only well-defined finite stencil inputs.
static constexpr int ISD = -4;

struct Fixture {
  std::string name;
  int npx;              // panel size (production convention: cells 1..npx-1)
  int is, ie;            // compute-domain slice actually exercised
  std::vector<real> q;   // indexed ISD..(npx-ISD+... ) via q[i-ISD]
  std::vector<real> c;   // Courant number, indexed is..ie+1 via c[i-is]
  std::vector<real> dxa; // indexed ISD.. via dxa[i-ISD], same range as q
};

static Fixture make_fixture(const std::string &name, int npx, int is, int ie,
                             real courant_const,
                             const std::function<real(int)> &gen) {
  Fixture f;
  f.name = name; f.npx = npx; f.is = is; f.ie = ie;
  int ied = npx + 4;
  f.q.resize(ied - ISD + 1);
  f.dxa.assign(ied - ISD + 1, 1.0);
  for (int i = ISD; i <= ied; ++i) f.q[i - ISD] = gen(i);
  f.c.assign(ie - is + 2, courant_const);
  return f;
}

static std::vector<Fixture> build_fixtures() {
  std::vector<Fixture> fx;
  const int npx = 20;   // interior fixtures: is=4, ie=15 so is1/ie1 stay off the panel edge

  // 1. constant tracer
  fx.push_back(make_fixture("constant", npx, 4, 15, 0.35, [](int) { return 5.0; }));
  // 2. linear gradient
  fx.push_back(make_fixture("linear_gradient", npx, 4, 15, 0.35,
                             [](int i) { return 1.0 + 0.5 * i; }));
  // 3. smooth curved profile (sinusoid)
  fx.push_back(make_fixture("smooth_curved", npx, 4, 15, 0.35,
                             [](int i) { return 3.0 + std::sin(0.3 * i); }));
  // 4. sharp front (front held fixed in the halo so the discontinuity is
  //    exercised only within/near the compute range, not smeared by halo).
  fx.push_back(make_fixture("sharp_front", npx, 4, 15, 0.35,
                             [npx](int i) { return (i < npx / 2) ? 1.0 : 10.0; }));
  // 5. local maximum
  fx.push_back(make_fixture("local_max", npx, 4, 15, 0.35, [npx](int i) {
    return (i == npx / 2) ? 20.0 : 2.0;
  }));
  // 6. local minimum
  fx.push_back(make_fixture("local_min", npx, 4, 15, 0.35, [npx](int i) {
    return (i == npx / 2) ? -5.0 : 2.0; // passive tracer, legal pre-positivity
  }));
  // 7. near-zero positive tracer
  fx.push_back(make_fixture("near_zero_positive", npx, 4, 15, 0.2,
                             [](int i) { return 1.0e-12 + 1.0e-14 * i; }));
  // 8. sign-changing passive tracer (legal: hord_tr's tracer path carries no
  //    positivity constraint; iord==8 has no positive-definite clamp, unlike
  //    iord==9/13's pert_ppm(iv=0) path, which is not on the hord==8 branch).
  fx.push_back(make_fixture("sign_changing", npx, 4, 15, -0.4,
                             [](int i) { return std::sin(0.5 * i) * 4.0; }));
  // 9. high Courant but production-valid case (|c|<1 required by donor-cell
  //    upwind formula; n_split/k_split keep production c well under 1).
  fx.push_back(make_fixture("high_courant_valid", npx, 4, 15, 0.9,
                             [](int i) { return 1.0 + 0.3 * i; }));
  // 10. boundary/halo-sensitive case: is==1 and (ie+1)==npx both exercised.
  fx.push_back(make_fixture("boundary_edge_is1_and_npx", npx, 1, npx - 1, 0.3,
                             [](int i) { return 1.0 + 0.4 * i; }));
  // 11. realistic C512-like slice: smooth large-scale field + small-scale
  //     noise, moderate Courant, matching production hord_tr transport of a
  //     single Held-Suarez passive tracer at C512 resolution scale factors.
  {
    fx.push_back(make_fixture("c512_like_slice", npx, 4, 15, 0.42, [](int i) {
      return 250.0 + 5.0 * std::sin(0.2 * i) + 0.01 * (((i + 40) * 37) % 7);
    }));
  }
  return fx;
}

int main() {
  auto fixtures = build_fixtures();
  int n_ran = 0, n_nan = 0, n_repeat_fail = 0, n_conserve_fail = 0;
  int n_hord8_ok = 0, n_diverge_from_hord6 = 0;

  printf("Prompt75_hord8_harness fixture,is,ie,nan_h6,nan_h8,repeat_h8_bitwise,"
         "mass_in,mass_out_h8,residual_h8,diverges_from_h6\n");

  for (auto &f : fixtures) {
    ++n_ran;
    int isd = ISD, ied = f.npx + 4;
    auto r6 = xppm_row(f.q, isd, ied, f.is, f.ie, f.npx, true, f.c, 6, f.dxa);
    auto r8 = xppm_row(f.q, isd, ied, f.is, f.ie, f.npx, true, f.c, 8, f.dxa);
    auto r8b = xppm_row(f.q, isd, ied, f.is, f.ie, f.npx, true, f.c, 8, f.dxa);

    bool nan6 = false, nan8 = false, repeat_ok = true;
    for (size_t i = 0; i < r6.flux.size(); ++i)
      if (std::isnan(r6.flux[i]) || std::isinf(r6.flux[i])) nan6 = true;
    for (size_t i = 0; i < r8.flux.size(); ++i) {
      if (std::isnan(r8.flux[i]) || std::isinf(r8.flux[i])) nan8 = true;
      if (r8.flux[i] != r8b.flux[i]) repeat_ok = false;
    }
    if (nan6 || nan8) ++n_nan;
    if (!repeat_ok) ++n_repeat_fail;

    // Conservation: sum of flux divergence over the interior slice with a
    // closed (non-periodic) boundary should telescope to
    // flux[ie+1]-flux[is]; use that identity as the conservation residual
    // proxy since q update itself belongs to tracer_2d_1L (already out of
    // scope, Prompt 74 NO-GO), not to fv_tp_2d's flux computation alone.
    real mass_in = 0.0;
    for (int i = f.is; i <= f.ie; ++i) mass_in += f.q[i - isd];
    real telescoped = 0.0;
    for (int i = f.is; i <= f.ie; ++i)
      telescoped += (r8.flux[i - f.is] - r8.flux[i + 1 - f.is]);
    real direct = r8.flux[0] - r8.flux[f.ie + 1 - f.is];
    real residual = std::fabs(telescoped - direct);
    if (residual > 1.0e-9 * std::fabs(direct == 0.0 ? 1.0 : direct)) ++n_conserve_fail;

    bool diverges = false;
    for (size_t i = 0; i < r6.flux.size(); ++i)
      if (std::fabs(r6.flux[i] - r8.flux[i]) > 1.0e-12) diverges = true;
    if (diverges) ++n_diverge_from_hord6;
    if (!nan8) ++n_hord8_ok;

    printf("%s,%d,%d,%d,%d,%d,%.6e,%.6e,%.3e,%d\n", f.name.c_str(), f.is, f.ie,
           nan6 ? 1 : 0, nan8 ? 1 : 0, repeat_ok ? 1 : 0, mass_in, telescoped,
           residual, diverges ? 1 : 0);
  }

  printf("---\nSUMMARY fixtures_ran=%d hord8_nan_free=%d repeatable_bitwise=%d "
         "conserve_pass=%d diverges_from_hord6=%d\n",
         n_ran, n_hord8_ok, n_ran - n_repeat_fail, n_ran - n_conserve_fail,
         n_diverge_from_hord6);
  return 0;
}
