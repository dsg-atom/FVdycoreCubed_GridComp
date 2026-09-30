// Milestone 3A.5 WP-B.2 arbitrary scalar transport + authoritative deln_flux test.
//
// Exercises dgrid::launch_scalar_transport_cuda and
// dgrid::launch_deln_flux_transport_cuda on a real GPU. Deliberately separate
// from tests/dgrid/test_wp_b_cuda_smoke.cpp: that file owns the Milestone 3A.4
// update_dz_d HEIGHT contract (npz+1 interface levels, per-level damped /
// undamped LevelFilter split, del6_vt_flux, the renormalised zh update, the
// WpBHeightTransportArena slot map). The scalar contract shares none of those,
// only the geometry struct and the PPM/copy_corners machinery, so conflating the
// two fixtures would couple two independent contracts.
//
// Parts:
//   A  interior order-6 fv_tp_2d, no panel edge, no corner  -> host reference
//   B  panel-edge PPM (all four seams, no corner)           -> host reference
//   C  corner behaviour: copy_corners rotation of q, all four corners
//   D  deln_flux ENABLED, nord = 0,1,2                      -> host reference
//   E  deln_flux DISABLED, and the damp_c <= 1.e-4 no-op branch
//   F  nord-dependent behaviour (0/1/2 are mutually distinct)
//   G  damp_c / da_min coefficient law -- the del6_vt_flux discriminator
//   H  mass-flux weighting, mass-weighted deln_flux, hord < 0 positivity
//   I  R8-only precision assertion (skipped in the R4 build)
//   J  R4-only precision assertion (skipped in the R8 build)
//   K  unsupported-configuration rejection before any kernel launch
//
// The host reference is an independent host-side transcription of the
// authoritative Fortran (GFDL_atmos_cubed_sphere geos/v3.0.0,
// ca30e26305fda43831095334c02a3f5e0b8c0c45): tp_core.F90 fv_tp_2d (:109-243),
// xppm (:324-556), yppm, copy_corners (:247-320) and deln_flux (:1263-1411). It
// is a different loop structure on a different processor, not a second numerical
// implementation to be maintained: nothing outside this file includes it.
//
// VALIDATION LEVEL: GPU EXECUTED plus host-reference agreement for the stages
// transcribed here. This is NOT authoritative numerical validation -- no
// Fortran-generated savepoint is compared. See
// docs/agent_handoff/milestone_3a5_wp_b_scalar_transport.md.

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "dgrid/wp_b_cuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace fv3::gpu;
using dgrid::LevelFilter;
using dgrid::ScalarTransportOptions;
using dgrid::TransportBounds;

static const double TOL = sizeof(fv_real) == 4 ? 2e-5 : 1e-11;

static bool close_rel(double a, double b, double tol = TOL) {
  const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
  return std::fabs(a - b) <= tol * scale;
}

// ------------------------------------------------------------------ host arrays
//
// Every 3-D array shares one box: i in [is-ng, ie+ng], j in [js-ng, je+ng],
// k in [0, nk). That is a superset of each individual Fortran array's declared
// bounds, which lets one allocation serve q, the Courant/mass-flux faces, the
// workspaces and the fluxes without a per-array descriptor. It is legitimate
// only because every launcher takes explicit loop bounds and never infers them
// from the allocation.

struct Box {
  int64_t is = 1, ie = 12, js = 1, je = 12, ng = 3, nk = 4;
  int64_t ni() const { return (ie + ng) - (is - ng) + 1; }
  int64_t nj() const { return (je + ng) - (js - ng) + 1; }
  int64_t loi() const { return is - ng; }
  int64_t loj() const { return js - ng; }
  size_t n3() const { return (size_t)ni() * nj() * nk; }
  size_t n2() const { return (size_t)ni() * nj(); }
};

struct A3 {
  std::vector<fv_real> v;
  Box b;
  void init(const Box &box, fv_real fill = fv_real(0)) { b = box; v.assign(box.n3(), fill); }
  size_t idx(int64_t i, int64_t j, int64_t k) const {
    return (size_t)((i - b.loi()) + (j - b.loj()) * b.ni() + k * b.ni() * b.nj());
  }
  fv_real &operator()(int64_t i, int64_t j, int64_t k) { return v[idx(i, j, k)]; }
  fv_real operator()(int64_t i, int64_t j, int64_t k) const { return v[idx(i, j, k)]; }
};

struct A2 {
  std::vector<fv_real> v;
  Box b;
  void init(const Box &box, fv_real fill = fv_real(0)) { b = box; v.assign(box.n2(), fill); }
  size_t idx(int64_t i, int64_t j) const {
    return (size_t)((i - b.loi()) + (j - b.loj()) * b.ni());
  }
  fv_real &operator()(int64_t i, int64_t j) { return v[idx(i, j)]; }
  fv_real operator()(int64_t i, int64_t j) const { return v[idx(i, j)]; }
};

// ------------------------------------------------- authoritative host reference
//
// tp_core.F90:89-96.
static const fv_real P1 = fv_real(7.0 / 12.0);
static const fv_real P2 = fv_real(-1.0 / 12.0);
static const fv_real C1 = fv_real(-2.0 / 14.0);
static const fv_real C2 = fv_real(11.0 / 14.0);
static const fv_real C3 = fv_real(5.0 / 14.0);

static bool seam_active(const TransportBounds &b) { return b.cubed_sphere && !b.nested; }

// tp_core.F90:369-394. The clipped interior bounds is1 = max(3,is-1) /
// ie3 = min(npx-2,ie+2) leave al(0..2) and al(npx-1..npx+1) unwritten on a
// seam-owning block; the one-sided formulas fill exactly those.
static fv_real h_al_x(const A3 &q, const A2 &dxa, int64_t i, int64_t j, int64_t k,
                      const TransportBounds &b, bool neg) {
  const int64_t npx = b.npx;
  fv_real al;
  if (seam_active(b) && b.west_edge && i <= 2) {
    if (i == 0)
      al = C1 * q(-2, j, k) + C2 * q(-1, j, k) + C3 * q(0, j, k);
    else if (i == 1)
      al = fv_real(0.5) *
           (((fv_real(2) * dxa(0, j) + dxa(-1, j)) * q(0, j, k) - dxa(0, j) * q(-1, j, k)) /
                (dxa(-1, j) + dxa(0, j)) +
            ((fv_real(2) * dxa(1, j) + dxa(2, j)) * q(1, j, k) - dxa(1, j) * q(2, j, k)) /
                (dxa(1, j) + dxa(2, j)));
    else
      al = C3 * q(1, j, k) + C2 * q(2, j, k) + C1 * q(3, j, k);
  } else if (seam_active(b) && b.east_edge && i >= npx - 1) {
    if (i == npx - 1)
      al = C1 * q(npx - 3, j, k) + C2 * q(npx - 2, j, k) + C3 * q(npx - 1, j, k);
    else if (i == npx)
      al = fv_real(0.5) *
           (((fv_real(2) * dxa(npx - 1, j) + dxa(npx - 2, j)) * q(npx - 1, j, k) -
             dxa(npx - 1, j) * q(npx - 2, j, k)) /
                (dxa(npx - 2, j) + dxa(npx - 1, j)) +
            ((fv_real(2) * dxa(npx, j) + dxa(npx + 1, j)) * q(npx, j, k) -
             dxa(npx, j) * q(npx + 1, j, k)) /
                (dxa(npx, j) + dxa(npx + 1, j)));
    else
      al = C3 * q(npx, j, k) + C2 * q(npx + 1, j, k) + C1 * q(npx + 2, j, k);
  } else {
    al = P1 * (q(i - 1, j, k) + q(i, j, k)) + P2 * (q(i - 2, j, k) + q(i + 1, j, k));
  }
  // tp_core.F90:390-394 -- applied AFTER the edge overrides, over is-1..ie+2.
  if (neg && al < fv_real(0)) al = fv_real(0);
  return al;
}

static fv_real h_al_y(const A3 &q, const A2 &dya, int64_t i, int64_t j, int64_t k,
                      const TransportBounds &b, bool neg) {
  const int64_t npy = b.npy;
  fv_real al;
  if (seam_active(b) && b.south_edge && j <= 2) {
    if (j == 0)
      al = C1 * q(i, -2, k) + C2 * q(i, -1, k) + C3 * q(i, 0, k);
    else if (j == 1)
      al = fv_real(0.5) *
           (((fv_real(2) * dya(i, 0) + dya(i, -1)) * q(i, 0, k) - dya(i, 0) * q(i, -1, k)) /
                (dya(i, -1) + dya(i, 0)) +
            ((fv_real(2) * dya(i, 1) + dya(i, 2)) * q(i, 1, k) - dya(i, 1) * q(i, 2, k)) /
                (dya(i, 1) + dya(i, 2)));
    else
      al = C3 * q(i, 1, k) + C2 * q(i, 2, k) + C1 * q(i, 3, k);
  } else if (seam_active(b) && b.north_edge && j >= npy - 1) {
    if (j == npy - 1)
      al = C1 * q(i, npy - 3, k) + C2 * q(i, npy - 2, k) + C3 * q(i, npy - 1, k);
    else if (j == npy)
      al = fv_real(0.5) *
           (((fv_real(2) * dya(i, npy - 1) + dya(i, npy - 2)) * q(i, npy - 1, k) -
             dya(i, npy - 1) * q(i, npy - 2, k)) /
                (dya(i, npy - 2) + dya(i, npy - 1)) +
            ((fv_real(2) * dya(i, npy) + dya(i, npy + 1)) * q(i, npy, k) -
             dya(i, npy) * q(i, npy + 1, k)) /
                (dya(i, npy) + dya(i, npy + 1)));
    else
      al = C3 * q(i, npy, k) + C2 * q(i, npy + 1, k) + C1 * q(i, npy + 2, k);
  } else {
    al = P1 * (q(i, j - 1, k) + q(i, j, k)) + P2 * (q(i, j - 2, k) + q(i, j + 1, k));
  }
  if (neg && al < fv_real(0)) al = fv_real(0);
  return al;
}

// tp_core.F90:532-541 WMP override: at the two cells adjacent to a seam the
// mord=6 smoothness test degrades to the mord=5 sign test.
static bool h_seam_cell_x(int64_t i, const TransportBounds &b) {
  if (!seam_active(b)) return false;
  if (b.west_edge && (i == 0 || i == 1)) return true;
  if (b.east_edge && (i == b.npx - 1 || i == b.npx)) return true;
  return false;
}
static bool h_seam_cell_y(int64_t j, const TransportBounds &b) {
  if (!seam_active(b)) return false;
  if (b.south_edge && (j == 0 || j == 1)) return true;
  if (b.north_edge && (j == b.npy - 1 || j == b.npy)) return true;
  return false;
}
static bool h_smt5(fv_real bl, fv_real br, fv_real b0, bool seam) {
  if (seam) return bl * br < fv_real(0);
  return fv_real(3) * std::fabs(b0) < std::fabs(bl - br);
}

// tp_core.F90:524-555 (mord == 6).
// Both admissible values of an x face flux, plus whether the limiter decision
// that selects between them is round-off-determined.
//
// The mord=6 limiter switches on the SIGN or relative MAGNITUDE of
// bl(i) = al(i) - q(i) and br(i) = al(i+1) - q(i). Where the transported field
// is locally constant along the sweep direction, al is mathematically exactly
// q, so bl/br are pure round-off and their computed sign depends on how the
// compiler contracts the multiply-adds that build al. Two builds of the
// AUTHORITATIVE Fortran with different fma contraction disagree at such a cell
// by the full size of the PPM correction fx1 -- this is a property of the FV3
// algorithm, not of any port -- so bitwise agreement with a host transcription
// cannot be demanded there. What can be demanded, and is, is that the device
// value equal one of the two admissible branches exactly.
struct XppmAlt {
  fv_real on;         // donor + fx1  (limiter engaged)
  fv_real off;        // donor        (limiter suppressed)
  bool decided;       // true when both limiter inputs are well clear of round-off
};

static XppmAlt h_xppm_alt(const A3 &q, const A2 &dxa, int64_t i, int64_t j, int64_t k,
                          fv_real c, const TransportBounds &b, bool neg) {
  const fv_real alm = h_al_x(q, dxa, i - 1, j, k, b, neg);
  const fv_real al0 = h_al_x(q, dxa, i, j, k, b, neg);
  const fv_real alp = h_al_x(q, dxa, i + 1, j, k, b, neg);
  const fv_real qm = q(i - 1, j, k), q0 = q(i, j, k);
  const fv_real blm = alm - qm, brm = al0 - qm, b0m = blm + brm;
  const fv_real bl0 = al0 - q0, br0 = alp - q0, b00 = bl0 + br0;
  XppmAlt r;
  fv_real fx1;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (brm - c * b0m);
    r.off = qm;
  } else {
    fx1 = (fv_real(1) + c) * (bl0 + c * b00);
    r.off = q0;
  }
  r.on = r.off + fx1;
  // A difference al - q is meaningful only above round-off of the operands that
  // formed it. 32 eps is a generous margin: al is a 3-4 term weighted sum, so a
  // few ulps of the operand scale is the floor of what it can resolve.
  const double eps = sizeof(fv_real) == 4 ? 1.2e-7 : 2.3e-16;
  auto resolved = [&](fv_real d, fv_real ref) {
    return std::fabs((double)d) > 32.0 * eps * std::fmax(1e-300, std::fabs((double)ref));
  };
  r.decided = resolved(blm, qm) && resolved(brm, qm) && resolved(bl0, q0) &&
              resolved(br0, q0);
  return r;
}

static fv_real h_xppm(const A3 &q, const A2 &dxa, int64_t i, int64_t j, int64_t k, fv_real c,
                      const TransportBounds &b, bool neg) {
  const fv_real alm = h_al_x(q, dxa, i - 1, j, k, b, neg);
  const fv_real al0 = h_al_x(q, dxa, i, j, k, b, neg);
  const fv_real alp = h_al_x(q, dxa, i + 1, j, k, b, neg);
  const fv_real qm = q(i - 1, j, k), q0 = q(i, j, k);
  const fv_real blm = alm - qm, brm = al0 - qm, b0m = blm + brm;
  const fv_real bl0 = al0 - q0, br0 = alp - q0, b00 = bl0 + br0;
  const bool sm = h_smt5(blm, brm, b0m, h_seam_cell_x(i - 1, b));
  const bool s0 = h_smt5(bl0, br0, b00, h_seam_cell_x(i, b));
  fv_real fx1, donor;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (brm - c * b0m);
    donor = qm;
  } else {
    fx1 = (fv_real(1) + c) * (bl0 + c * b00);
    donor = q0;
  }
  return (sm || s0) ? donor + fx1 : donor;
}

static fv_real h_yppm(const A3 &q, const A2 &dya, int64_t i, int64_t j, int64_t k, fv_real c,
                      const TransportBounds &b, bool neg) {
  const fv_real alm = h_al_y(q, dya, i, j - 1, k, b, neg);
  const fv_real al0 = h_al_y(q, dya, i, j, k, b, neg);
  const fv_real alp = h_al_y(q, dya, i, j + 1, k, b, neg);
  const fv_real qm = q(i, j - 1, k), q0 = q(i, j, k);
  const fv_real blm = alm - qm, brm = al0 - qm, b0m = blm + brm;
  const fv_real bl0 = al0 - q0, br0 = alp - q0, b00 = bl0 + br0;
  const bool sm = h_smt5(blm, brm, b0m, h_seam_cell_y(j - 1, b));
  const bool s0 = h_smt5(bl0, br0, b00, h_seam_cell_y(j, b));
  fv_real fx1, donor;
  if (c > fv_real(0)) {
    fx1 = (fv_real(1) - c) * (brm - c * b0m);
    donor = qm;
  } else {
    fx1 = (fv_real(1) + c) * (bl0 + c * b00);
    donor = q0;
  }
  return (sm || s0) ? donor + fx1 : donor;
}

// --- hord==8 host reference (Prompt 76 Stage 5) ---------------------------
//
// Mirrors dgrid::{dm8_x_at, al8_x_at, bl_br8_x_at, xppm_face_flux_hord8} in
// wp_b_cuda.cu line-for-line (same operand order, same fmin/fmax/copysign
// call shape) rather than reusing tests/host/test_fv_tp_2d_hord8_hostharness.cpp's
// row-based transcription: a bitwise CUDA-vs-host check demands the same
// operation order the device code actually executes, not just an
// independently-authoritative one. hord==8's tp_core.F90 branch has no al
// positivity clamp (:388 gates `neg` inside iord<7 only), so unlike h_al_x
// there is no `neg` parameter here.
static const fv_real HORD8_R3 = fv_real(1.0 / 3.0);
static const fv_real HORD8_S11 = fv_real(11.0 / 14.0);
static const fv_real HORD8_S14 = fv_real(4.0 / 7.0);
static const fv_real HORD8_S15 = fv_real(3.0 / 14.0);

// Host-only mirror of dgrid::BlBr8 (wp_b_cuda.cu) -- that struct is defined
// inside the .cu translation unit, not exposed via wp_b_cuda.cuh, so it is
// not visible to this .cpp test file.
struct BlBr8 {
  fv_real bl, br;
};

static void h_pert_ppm_iv1(fv_real &al, fv_real &ar) {
  if (al * ar < fv_real(0)) {
    const fv_real da1 = al - ar;
    const fv_real da2 = da1 * da1;
    const fv_real a6da = fv_real(3) * (al + ar) * da1;
    if (a6da < -da2) {
      ar = fv_real(-2) * al;
    } else if (a6da > da2) {
      al = fv_real(-2) * ar;
    }
  } else {
    al = fv_real(0);
    ar = fv_real(0);
  }
}

static fv_real h_dm8_x(const A3 &q, int64_t i, int64_t j, int64_t k) {
  const fv_real qm = q(i - 1, j, k), q0 = q(i, j, k), qp = q(i + 1, j, k);
  const fv_real xt = fv_real(0.25) * (qp - qm);
  const fv_real lo = std::fmin(std::fmin(qm, q0), qp);
  const fv_real hi = std::fmax(std::fmax(qm, q0), qp);
  const fv_real mag = std::fmin(std::fabs(xt), std::fmin(hi - q0, q0 - lo));
  return std::copysign(mag, xt);
}
static fv_real h_dm8_y(const A3 &q, int64_t i, int64_t j, int64_t k) {
  const fv_real qm = q(i, j - 1, k), q0 = q(i, j, k), qp = q(i, j + 1, k);
  const fv_real xt = fv_real(0.25) * (qp - qm);
  const fv_real lo = std::fmin(std::fmin(qm, q0), qp);
  const fv_real hi = std::fmax(std::fmax(qm, q0), qp);
  const fv_real mag = std::fmin(std::fabs(xt), std::fmin(hi - q0, q0 - lo));
  return std::copysign(mag, xt);
}
static fv_real h_al8_x(const A3 &q, int64_t i, int64_t j, int64_t k) {
  return fv_real(0.5) * (q(i - 1, j, k) + q(i, j, k)) +
         HORD8_R3 * (h_dm8_x(q, i - 1, j, k) - h_dm8_x(q, i, j, k));
}
static fv_real h_al8_y(const A3 &q, int64_t i, int64_t j, int64_t k) {
  return fv_real(0.5) * (q(i, j - 1, k) + q(i, j, k)) +
         HORD8_R3 * (h_dm8_y(q, i, j - 1, k) - h_dm8_y(q, i, j, k));
}

static BlBr8 h_bl_br8_x(const A3 &q, const A2 &dxa, int64_t i, int64_t j, int64_t k,
                        const TransportBounds &b) {
  const int64_t npx = b.npx;
  const bool seam = b.cubed_sphere && !b.nested;
  BlBr8 r{};
  if (seam && b.west_edge && i >= 0 && i <= 2) {
    const fv_real xt_a_raw =
        fv_real(0.5) *
        (((fv_real(2) * dxa(0, j) + dxa(-1, j)) * q(0, j, k) - dxa(0, j) * q(-1, j, k)) /
             (dxa(-1, j) + dxa(0, j)) +
         ((fv_real(2) * dxa(1, j) + dxa(2, j)) * q(1, j, k) - dxa(1, j) * q(2, j, k)) /
             (dxa(1, j) + dxa(2, j)));
    const fv_real lo4 = std::fmin(std::fmin(q(-1, j, k), q(0, j, k)),
                                  std::fmin(q(1, j, k), q(2, j, k)));
    const fv_real hi4 = std::fmax(std::fmax(q(-1, j, k), q(0, j, k)),
                                  std::fmax(q(1, j, k), q(2, j, k)));
    fv_real xt_a = std::fmax(xt_a_raw, lo4);
    xt_a = std::fmin(xt_a, hi4);
    const fv_real xt_b = HORD8_S15 * q(1, j, k) + HORD8_S11 * q(2, j, k) -
                        HORD8_S14 * h_dm8_x(q, 2, j, k);
    if (i == 0) {
      fv_real bl0 = HORD8_S14 * h_dm8_x(q, -1, j, k) + HORD8_S11 * (q(-1, j, k) - q(0, j, k));
      fv_real br0 = xt_a - q(0, j, k);
      h_pert_ppm_iv1(bl0, br0);
      r.bl = bl0; r.br = br0;
    } else if (i == 1) {
      fv_real bl1 = xt_a - q(1, j, k);
      fv_real br1 = xt_b - q(1, j, k);
      h_pert_ppm_iv1(bl1, br1);
      r.bl = bl1; r.br = br1;
    } else {
      fv_real bl2 = xt_b - q(2, j, k);
      fv_real br2 = h_al8_x(q, 3, j, k) - q(2, j, k);
      h_pert_ppm_iv1(bl2, br2);
      r.bl = bl2; r.br = br2;
    }
    return r;
  }
  if (seam && b.east_edge && i >= npx - 2 && i <= npx) {
    const fv_real xt_a = HORD8_S15 * q(npx - 1, j, k) + HORD8_S11 * q(npx - 2, j, k) +
                        HORD8_S14 * h_dm8_x(q, npx - 2, j, k);
    const fv_real lo4 = std::fmin(std::fmin(q(npx - 2, j, k), q(npx - 1, j, k)),
                                  std::fmin(q(npx, j, k), q(npx + 1, j, k)));
    const fv_real hi4 = std::fmax(std::fmax(q(npx - 2, j, k), q(npx - 1, j, k)),
                                  std::fmax(q(npx, j, k), q(npx + 1, j, k)));
    const fv_real xt_b_raw =
        fv_real(0.5) *
        (((fv_real(2) * dxa(npx - 1, j) + dxa(npx - 2, j)) * q(npx - 1, j, k) -
          dxa(npx - 1, j) * q(npx - 2, j, k)) /
             (dxa(npx - 2, j) + dxa(npx - 1, j)) +
         ((fv_real(2) * dxa(npx, j) + dxa(npx + 1, j)) * q(npx, j, k) -
          dxa(npx, j) * q(npx + 1, j, k)) /
             (dxa(npx, j) + dxa(npx + 1, j)));
    fv_real xt_b = std::fmax(xt_b_raw, lo4);
    xt_b = std::fmin(xt_b, hi4);
    if (i == npx - 2) {
      fv_real bl = h_al8_x(q, npx - 2, j, k) - q(npx - 2, j, k);
      fv_real br = xt_a - q(npx - 2, j, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    } else if (i == npx - 1) {
      fv_real bl = xt_a - q(npx - 1, j, k);
      fv_real br = xt_b - q(npx - 1, j, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    } else {
      fv_real bl = xt_b - q(npx, j, k);
      fv_real br = HORD8_S11 * (q(npx + 1, j, k) - q(npx, j, k)) -
                  HORD8_S14 * h_dm8_x(q, npx + 1, j, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    }
    return r;
  }
  const fv_real dmi = h_dm8_x(q, i, j, k);
  const fv_real ali = h_al8_x(q, i, j, k);
  const fv_real alip1 = h_al8_x(q, i + 1, j, k);
  const fv_real xt = fv_real(2) * dmi;
  r.bl = -std::copysign(std::fmin(std::fabs(xt), std::fabs(ali - q(i, j, k))), xt);
  r.br = std::copysign(std::fmin(std::fabs(xt), std::fabs(alip1 - q(i, j, k))), xt);
  return r;
}

static BlBr8 h_bl_br8_y(const A3 &q, const A2 &dya, int64_t i, int64_t j, int64_t k,
                        const TransportBounds &b) {
  const int64_t npy = b.npy;
  const bool seam = b.cubed_sphere && !b.nested;
  BlBr8 r{};
  if (seam && b.south_edge && j >= 0 && j <= 2) {
    const fv_real xt_a_raw =
        fv_real(0.5) *
        (((fv_real(2) * dya(i, 0) + dya(i, -1)) * q(i, 0, k) - dya(i, 0) * q(i, -1, k)) /
             (dya(i, -1) + dya(i, 0)) +
         ((fv_real(2) * dya(i, 1) + dya(i, 2)) * q(i, 1, k) - dya(i, 1) * q(i, 2, k)) /
             (dya(i, 1) + dya(i, 2)));
    const fv_real lo4 = std::fmin(std::fmin(q(i, -1, k), q(i, 0, k)),
                                  std::fmin(q(i, 1, k), q(i, 2, k)));
    const fv_real hi4 = std::fmax(std::fmax(q(i, -1, k), q(i, 0, k)),
                                  std::fmax(q(i, 1, k), q(i, 2, k)));
    fv_real xt_a = std::fmax(xt_a_raw, lo4);
    xt_a = std::fmin(xt_a, hi4);
    const fv_real xt_b = HORD8_S15 * q(i, 1, k) + HORD8_S11 * q(i, 2, k) -
                        HORD8_S14 * h_dm8_y(q, i, 2, k);
    if (j == 0) {
      fv_real bl0 = HORD8_S14 * h_dm8_y(q, i, -1, k) + HORD8_S11 * (q(i, -1, k) - q(i, 0, k));
      fv_real br0 = xt_a - q(i, 0, k);
      h_pert_ppm_iv1(bl0, br0);
      r.bl = bl0; r.br = br0;
    } else if (j == 1) {
      fv_real bl1 = xt_a - q(i, 1, k);
      fv_real br1 = xt_b - q(i, 1, k);
      h_pert_ppm_iv1(bl1, br1);
      r.bl = bl1; r.br = br1;
    } else {
      fv_real bl2 = xt_b - q(i, 2, k);
      fv_real br2 = h_al8_y(q, i, 3, k) - q(i, 2, k);
      h_pert_ppm_iv1(bl2, br2);
      r.bl = bl2; r.br = br2;
    }
    return r;
  }
  if (seam && b.north_edge && j >= npy - 2 && j <= npy) {
    const fv_real xt_a = HORD8_S15 * q(i, npy - 1, k) + HORD8_S11 * q(i, npy - 2, k) +
                        HORD8_S14 * h_dm8_y(q, i, npy - 2, k);
    const fv_real lo4 = std::fmin(std::fmin(q(i, npy - 2, k), q(i, npy - 1, k)),
                                  std::fmin(q(i, npy, k), q(i, npy + 1, k)));
    const fv_real hi4 = std::fmax(std::fmax(q(i, npy - 2, k), q(i, npy - 1, k)),
                                  std::fmax(q(i, npy, k), q(i, npy + 1, k)));
    const fv_real xt_b_raw =
        fv_real(0.5) *
        (((fv_real(2) * dya(i, npy - 1) + dya(i, npy - 2)) * q(i, npy - 1, k) -
          dya(i, npy - 1) * q(i, npy - 2, k)) /
             (dya(i, npy - 2) + dya(i, npy - 1)) +
         ((fv_real(2) * dya(i, npy) + dya(i, npy + 1)) * q(i, npy, k) -
          dya(i, npy) * q(i, npy + 1, k)) /
             (dya(i, npy) + dya(i, npy + 1)));
    fv_real xt_b = std::fmax(xt_b_raw, lo4);
    xt_b = std::fmin(xt_b, hi4);
    if (j == npy - 2) {
      fv_real bl = h_al8_y(q, i, npy - 2, k) - q(i, npy - 2, k);
      fv_real br = xt_a - q(i, npy - 2, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    } else if (j == npy - 1) {
      fv_real bl = xt_a - q(i, npy - 1, k);
      fv_real br = xt_b - q(i, npy - 1, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    } else {
      fv_real bl = xt_b - q(i, npy, k);
      fv_real br = HORD8_S11 * (q(i, npy + 1, k) - q(i, npy, k)) -
                  HORD8_S14 * h_dm8_y(q, i, npy + 1, k);
      h_pert_ppm_iv1(bl, br);
      r.bl = bl; r.br = br;
    }
    return r;
  }
  const fv_real dmj = h_dm8_y(q, i, j, k);
  const fv_real alj = h_al8_y(q, i, j, k);
  const fv_real aljp1 = h_al8_y(q, i, j + 1, k);
  const fv_real xt = fv_real(2) * dmj;
  r.bl = -std::copysign(std::fmin(std::fabs(xt), std::fabs(alj - q(i, j, k))), xt);
  r.br = std::copysign(std::fmin(std::fabs(xt), std::fabs(aljp1 - q(i, j, k))), xt);
  return r;
}

static fv_real h_xppm8(const A3 &q, const A2 &dxa, int64_t i, int64_t j, int64_t k, fv_real c,
                       const TransportBounds &b) {
  if (c > fv_real(0)) {
    const BlBr8 m1 = h_bl_br8_x(q, dxa, i - 1, j, k, b);
    return q(i - 1, j, k) + (fv_real(1) - c) * (m1.br - c * (m1.bl + m1.br));
  } else {
    const BlBr8 c0 = h_bl_br8_x(q, dxa, i, j, k, b);
    return q(i, j, k) + (fv_real(1) + c) * (c0.bl + c * (c0.bl + c0.br));
  }
}
static fv_real h_yppm8(const A3 &q, const A2 &dya, int64_t i, int64_t j, int64_t k, fv_real c,
                       const TransportBounds &b) {
  if (c > fv_real(0)) {
    const BlBr8 m1 = h_bl_br8_y(q, dya, i, j - 1, k, b);
    return q(i, j - 1, k) + (fv_real(1) - c) * (m1.br - c * (m1.bl + m1.br));
  } else {
    const BlBr8 c0 = h_bl_br8_y(q, dya, i, j, k, b);
    return q(i, j, k) + (fv_real(1) + c) * (c0.bl + c * (c0.bl + c0.br));
  }
}

// tp_core.F90:255-318. All four corners, in the Fortran's order. The eight
// regions are pairwise disjoint (every source block has an INTERIOR second
// index, every destination block a HALO second index), so the order is in fact
// immaterial -- which is what licenses the single-launch CUDA version.
static void h_copy_corners(A3 &q, int32_t dir, const TransportBounds &b) {
  if (b.nested) return;
  const int64_t ng = b.ng, npx = b.npx, npy = b.npy;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (dir == 1) {
      if (b.sw_corner)
        for (int64_t j = 1 - ng; j <= 0; ++j)
          for (int64_t i = 1 - ng; i <= 0; ++i) q(i, j, k) = q(j, 1 - i, k);
      if (b.se_corner)
        for (int64_t j = 1 - ng; j <= 0; ++j)
          for (int64_t i = npx; i <= npx + ng - 1; ++i) q(i, j, k) = q(npy - j, i - npx + 1, k);
      if (b.ne_corner)
        for (int64_t j = npy; j <= npy + ng - 1; ++j)
          for (int64_t i = npx; i <= npx + ng - 1; ++i) q(i, j, k) = q(j, 2 * npx - 1 - i, k);
      if (b.nw_corner)
        for (int64_t j = npy; j <= npy + ng - 1; ++j)
          for (int64_t i = 1 - ng; i <= 0; ++i) q(i, j, k) = q(npy - j, i - 1 + npx, k);
    } else {
      if (b.sw_corner)
        for (int64_t j = 1 - ng; j <= 0; ++j)
          for (int64_t i = 1 - ng; i <= 0; ++i) q(i, j, k) = q(1 - j, i, k);
      if (b.se_corner)
        for (int64_t j = 1 - ng; j <= 0; ++j)
          for (int64_t i = npx; i <= npx + ng - 1; ++i) q(i, j, k) = q(npy + j - 1, npx - i, k);
      if (b.ne_corner)
        for (int64_t j = npy; j <= npy + ng - 1; ++j)
          for (int64_t i = npx; i <= npx + ng - 1; ++i) q(i, j, k) = q(2 * npy - 1 - j, i, k);
      if (b.nw_corner)
        for (int64_t j = npy; j <= npy + ng - 1; ++j)
          for (int64_t i = 1 - ng; i <= 0; ++i) q(i, j, k) = q(j + 1 - npx, npy - i, k);
    }
  }
}

// The whole synthetic input set for one scalar-transport case.
struct Inputs {
  Box box;
  TransportBounds b{};
  A3 q, mass, crx, cry, xfx, yfx, mfx, mfy, ra_x, ra_y;
  A2 area, rarea, dxa, dya, del6_u, del6_v;
};

// tp_core.F90:109-243 for hord == ord_in == ord_ou. Mutates q in place through
// the two copy_corners calls, exactly as the Fortran intent(inout) dummy does,
// and returns the fluxes plus fv_tp_2d's own fx2/fy2 scratch (which the CUDA
// path leaves in the caller's workspace and this test inspects directly).
struct TpResult {
  A3 fx, fy, fx2, fy2, q_i, q_j, q_out;
};

static TpResult h_fv_tp_2d(const Inputs &in, int32_t hord, bool mass_weighting) {
  const TransportBounds &b = in.b;
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, ng = b.ng;
  const int64_t isd = is - ng, ied = ie + ng, jsd = js - ng, jed = je + ng;
  const int32_t mord = hord >= 0 ? hord : -hord;
  const bool hord8 = mord == 8;
  const bool neg = hord < 0;
  const A2 &dxa = in.dxa, &dya = in.dya;

  TpResult r;
  r.q_out = in.q;
  r.fx.init(in.box);
  r.fy.init(in.box);
  r.fx2.init(in.box);
  r.fy2.init(in.box);
  r.q_i.init(in.box);
  r.q_j.init(in.box);
  A3 &q = r.q_out;

  h_copy_corners(q, 2, b);                                            // :158
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = js; j <= je + 1; ++j)                            // yppm, :161
      for (int64_t i = isd; i <= ied; ++i)
        r.fy2(i, j, k) = hord8 ? h_yppm8(q, dya, i, j, k, in.cry(i, j, k), b)
                               : h_yppm(q, dya, i, j, k, in.cry(i, j, k), b, neg);
    for (int64_t j = js; j <= je; ++j)                                // q_i, :168-177
      for (int64_t i = isd; i <= ied; ++i) {
        const fv_real fyy_j = in.yfx(i, j, k) * r.fy2(i, j, k);
        const fv_real fyy_jp = in.yfx(i, j + 1, k) * r.fy2(i, j + 1, k);
        r.q_i(i, j, k) = (q(i, j, k) * in.area(i, j) + fyy_j - fyy_jp) / in.ra_y(i, j, k);
      }
    for (int64_t j = js; j <= je; ++j)                                // xppm(q_i), :180
      for (int64_t i = is; i <= ie + 1; ++i)
        r.fx(i, j, k) = hord8 ? h_xppm8(r.q_i, dxa, i, j, k, in.crx(i, j, k), b)
                              : h_xppm(r.q_i, dxa, i, j, k, in.crx(i, j, k), b, neg);
  }

  h_copy_corners(q, 1, b);                                            // :183
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = jsd; j <= jed; ++j)                              // xppm(q), :186
      for (int64_t i = is; i <= ie + 1; ++i)
        r.fx2(i, j, k) = hord8 ? h_xppm8(q, dxa, i, j, k, in.crx(i, j, k), b)
                               : h_xppm(q, dxa, i, j, k, in.crx(i, j, k), b, neg);
    for (int64_t j = jsd; j <= jed; ++j)                              // q_j, :186-193
      for (int64_t i = is; i <= ie; ++i) {
        const fv_real fx1_i = in.xfx(i, j, k) * r.fx2(i, j, k);
        const fv_real fx1_ip = in.xfx(i + 1, j, k) * r.fx2(i + 1, j, k);
        r.q_j(i, j, k) = (q(i, j, k) * in.area(i, j) + fx1_i - fx1_ip) / in.ra_x(i, j, k);
      }
    for (int64_t j = js; j <= je + 1; ++j)                            // yppm(q_j), :196
      for (int64_t i = is; i <= ie; ++i)
        r.fy(i, j, k) = hord8 ? h_yppm8(r.q_j, dya, i, j, k, in.cry(i, j, k), b)
                              : h_yppm(r.q_j, dya, i, j, k, in.cry(i, j, k), b, neg);
  }

  // :205-234. Only the weight differs between the two branches.
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        const fv_real w = mass_weighting ? in.mfx(i, j, k) : in.xfx(i, j, k);
        r.fx(i, j, k) = fv_real(0.5) * (r.fx(i, j, k) + r.fx2(i, j, k)) * w;
      }
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie; ++i) {
        const fv_real w = mass_weighting ? in.mfy(i, j, k) : in.yfx(i, j, k);
        r.fy(i, j, k) = fv_real(0.5) * (r.fy(i, j, k) + r.fy2(i, j, k)) * w;
      }
  }
  return r;
}

// tp_core.F90:1263-1411, folded into fx/fy in place.
//
// d2 is POISONED with NaN outside Fortran's [is-1-nord, ie+1+nord] x
// [js-1-nord, je+1+nord] init box, which is exactly the region the Fortran
// leaves undefined. The CUDA kernel instead initialises the full [is-ng, ie+ng]
// box (documented structural difference). If any poisoned cell reached an
// output, this reference would produce NaN and the comparison against the CUDA
// result would fail -- so agreement is a positive test that the superset
// initialisation changes no value that is read downstream.
static void h_deln_flux(const A3 &q, const Inputs &in, int32_t nord, fv_real damp_c,
                        fv_real da_min, bool mass_weighted, A3 &fx, A3 &fy) {
  const TransportBounds &b = in.b;
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  const fv_real damp = (fv_real)std::pow((double)(damp_c * da_min), (double)(nord + 1));
  const fv_real poison = std::numeric_limits<fv_real>::quiet_NaN();

  A3 d2, fx2, fy2;
  d2.init(in.box, poison);
  fx2.init(in.box, poison);
  fy2.init(in.box, poison);

  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = js - 1 - nord; j <= je + 1 + nord; ++j)          // :1274-1287
      for (int64_t i = is - 1 - nord; i <= ie + 1 + nord; ++i)
        d2(i, j, k) = mass_weighted ? q(i, j, k) : damp * q(i, j, k);

  if (nord > 0) h_copy_corners(d2, 1, b);                             // :1313
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = js - nord; j <= je + nord; ++j)                  // :1316-1320
      for (int64_t i = is - nord; i <= ie + nord + 1; ++i)
        fx2(i, j, k) = in.del6_v(i, j) * (d2(i - 1, j, k) - d2(i, j, k));

  if (nord > 0) h_copy_corners(d2, 2, b);                             // :1326
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = js - nord; j <= je + nord + 1; ++j)              // :1329-1333
      for (int64_t i = is - nord; i <= ie + nord; ++i)
        fy2(i, j, k) = in.del6_u(i, j) * (d2(i, j - 1, k) - d2(i, j, k));

  for (int32_t n = 1; n <= nord; ++n) {                               // :1343-1374
    const int64_t nt = nord - n;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = js - nt - 1; j <= je + nt + 1; ++j)
        for (int64_t i = is - nt - 1; i <= ie + nt + 1; ++i)
          d2(i, j, k) = (fx2(i, j, k) - fx2(i + 1, j, k) + fy2(i, j, k) - fy2(i, j + 1, k)) *
                        in.rarea(i, j);
    h_copy_corners(d2, 1, b);                                         // :1354
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = js - nt; j <= je + nt; ++j)                    // note flipped sign
        for (int64_t i = is - nt; i <= ie + nt + 1; ++i)
          fx2(i, j, k) = in.del6_v(i, j) * (d2(i, j, k) - d2(i - 1, j, k));
    h_copy_corners(d2, 2, b);                                         // :1366
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = js - nt; j <= je + nt + 1; ++j)
        for (int64_t i = is - nt; i <= ie + nt; ++i)
          fy2(i, j, k) = in.del6_u(i, j) * (d2(i, j, k) - d2(i, j - 1, k));
  }

  if (mass_weighted) {                                                // :1377-1384
    const fv_real damp2 = fv_real(0.5) * damp;
    for (int64_t k = 0; k < b.nk; ++k) {
      for (int64_t j = js; j <= je; ++j)
        for (int64_t i = is; i <= ie + 1; ++i)
          fx(i, j, k) = fx(i, j, k) +
                        damp2 * (in.mass(i - 1, j, k) + in.mass(i, j, k)) * fx2(i, j, k);
      for (int64_t j = js; j <= je + 1; ++j)
        for (int64_t i = is; i <= ie; ++i)
          fy(i, j, k) = fy(i, j, k) +
                        damp2 * (in.mass(i, j - 1, k) + in.mass(i, j, k)) * fy2(i, j, k);
    }
  } else {                                                            // :1386-1394
    for (int64_t k = 0; k < b.nk; ++k) {
      for (int64_t j = js; j <= je; ++j)
        for (int64_t i = is; i <= ie + 1; ++i) fx(i, j, k) = fx(i, j, k) + fx2(i, j, k);
      for (int64_t j = js; j <= je + 1; ++j)
        for (int64_t i = is; i <= ie; ++i) fy(i, j, k) = fy(i, j, k) + fy2(i, j, k);
    }
  }
}

// ------------------------------------------------------------- synthetic inputs
//
// Non-degenerate and non-symmetric throughout: a constant area/dxa/del6 or a
// symmetric q would let a wrong stencil, a wrong extent or a swapped x/y pass
// agree by cancellation. Courant numbers straddle zero so both the c > 0 and
// c <= 0 upwind branches of xppm/yppm execute.
//
// q_mode 0 -> smooth, strictly positive (al never clamps: hord<0 is inert)
// q_mode 1 -> STRICTLY POSITIVE with a sharp ~25x step in i, which drives the
//             order-6 reconstruction negative near the jump so the hord < 0
//             clamp is genuinely live. Strict positivity is deliberate: this is
//             the delp-like situation the clamp exists for (hord_dp = -6), and a
//             field that crossed zero would additionally make the flux magnitude
//             pass through zero, so a comparison against the host reference
//             would be dominated by cancellation rather than by the clamp.
static Inputs make_inputs(const Box &box, const TransportBounds &b, int q_mode = 0) {
  Inputs in;
  in.box = box;
  in.b = b;
  in.q.init(box);
  in.mass.init(box);
  in.crx.init(box);
  in.cry.init(box);
  in.xfx.init(box);
  in.yfx.init(box);
  in.mfx.init(box);
  in.mfy.init(box);
  in.ra_x.init(box);
  in.ra_y.init(box);
  in.area.init(box);
  in.rarea.init(box);
  in.dxa.init(box);
  in.dya.init(box);
  in.del6_u.init(box);
  in.del6_v.init(box);

  const int64_t i0 = box.loi(), i1 = box.ie + box.ng;
  const int64_t j0 = box.loj(), j1 = box.je + box.ng;
  const int64_t imid = (box.is + box.ie) / 2;

  for (int64_t j = j0; j <= j1; ++j)
    for (int64_t i = i0; i <= i1; ++i) {
      const double a = 1.0e8 * (1.0 + 0.02 * std::sin(0.4 * i) + 0.03 * std::cos(0.3 * j));
      in.area(i, j) = (fv_real)a;
      in.rarea(i, j) = (fv_real)(1.0 / a);
      in.dxa(i, j) = (fv_real)(1.0e4 * (1.0 + 0.05 * std::sin(0.25 * i) + 0.01 * j));
      in.dya(i, j) = (fv_real)(1.0e4 * (1.0 + 0.04 * std::cos(0.22 * j) + 0.01 * i));
      in.del6_u(i, j) = (fv_real)(0.30 * (1.0 + 0.10 * std::sin(0.3 * i + 0.2 * j)));
      in.del6_v(i, j) = (fv_real)(0.28 * (1.0 + 0.12 * std::cos(0.27 * i - 0.19 * j)));
    }

  for (int64_t k = 0; k < box.nk; ++k)
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = i0; i <= i1; ++i) {
        double qv;
        if (q_mode == 2)
          // Deliberately CONSTANT along i inside each plateau, which makes the
          // mord=6 limiter decision round-off-determined (see h_xppm_alt).
          qv = 0.5 + (i >= imid ? 12.0 : 0.0) + 0.05 * std::sin(0.31 * j) + 0.02 * k;
        else if (q_mode == 1)
          // Same ~25x step, but with a small monotone tilt in i so that
          // al(i) - q(i) stays far above round-off everywhere and every limiter
          // decision is well posed. Without the tilt the plateaus are exactly
          // constant in i and the branch is decided by fma contraction order.
          qv = 0.5 + 0.004 * i + (i >= imid ? 12.0 : 0.0) + 0.05 * std::sin(0.31 * j) +
               0.02 * k;
        else
          qv = 300.0 + 20.0 * std::sin(0.37 * i) * std::cos(0.29 * j) +
               5.0 * std::cos(0.21 * (i + j)) + 2.0 * k;
        in.q(i, j, k) = (fv_real)qv;
        // delp-like mass: strictly positive, non-uniform in all three indices.
        in.mass(i, j, k) = (fv_real)(800.0 + 40.0 * k + 3.0 * std::sin(0.23 * i + 0.17 * j));
        // Courant numbers change sign, so both upwind branches run.
        in.crx(i, j, k) = (fv_real)(0.18 * std::sin(0.33 * i + 0.11 * j + 0.5 * k));
        in.cry(i, j, k) = (fv_real)(0.16 * std::cos(0.29 * i - 0.13 * j + 0.4 * k));
        in.xfx(i, j, k) = (fv_real)(1.0e6 * (1.0 + 0.10 * std::sin(0.27 * i + 0.19 * j)) *
                                    (1.0 + 0.03 * k));
        in.yfx(i, j, k) = (fv_real)(1.1e6 * (1.0 + 0.09 * std::cos(0.24 * i - 0.21 * j)) *
                                    (1.0 + 0.02 * k));
        in.mfx(i, j, k) = (fv_real)(2.0e6 * (1.0 + 0.12 * std::cos(0.31 * i + 0.07 * j)) *
                                    (1.0 + 0.04 * k));
        in.mfy(i, j, k) = (fv_real)(1.8e6 * (1.0 + 0.11 * std::sin(0.19 * i - 0.29 * j)) *
                                    (1.0 + 0.05 * k));
      }

  // ra_x / ra_y on the authoritative -- and deliberately asymmetric -- extents
  // (sw_core.F90:928-936, nh_utils.F90:269-278): ra_x on is..ie x jsd..jed,
  // ra_y on isd..ied x js..je. The device values come from launch_ra_x_cuda /
  // launch_ra_y_cuda and are checked against these.
  for (int64_t k = 0; k < box.nk; ++k) {
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = box.is; i <= box.ie; ++i)
        in.ra_x(i, j, k) = in.area(i, j) + (in.xfx(i, j, k) - in.xfx(i + 1, j, k));
    for (int64_t j = box.js; j <= box.je; ++j)
      for (int64_t i = i0; i <= i1; ++i)
        in.ra_y(i, j, k) = in.area(i, j) + (in.yfx(i, j, k) - in.yfx(i, j + 1, k));
  }
  return in;
}

// --------------------------------------------------------------- device harness

struct Dev {
  std::vector<void *> owned;
  ~Dev() {
    for (void *p : owned) cudaFree(p);
  }
  void *raw(size_t bytes) {
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    owned.push_back(p);
    cudaMemset(p, 0, bytes);  // scratch hygiene: keeps a whole-slot D2H initcheck-clean
    return p;
  }
  fv_real *up3(const A3 &a) {
    auto *p = (fv_real *)raw(a.v.size() * sizeof(fv_real));
    cudaMemcpy(p, a.v.data(), a.v.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return p;
  }
  fv_real *up2(const A2 &a) {
    auto *p = (fv_real *)raw(a.v.size() * sizeof(fv_real));
    cudaMemcpy(p, a.v.data(), a.v.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return p;
  }
  fv_real *scratch(const Box &b) { return (fv_real *)raw(b.n3() * sizeof(fv_real)); }
};

template <class T> static operators::View3<T> V3(void *p, const Box &b) {
  return operators::View3<T>{(T *)p, b.loi(), b.loj(), 0, 1, b.ni(), b.ni() * b.nj()};
}
template <class T> static operators::View2<T> V2(void *p, const Box &b) {
  return operators::View2<T>{(T *)p, b.loi(), b.loj(), 1, b.ni()};
}

static A3 down3(const void *p, const Box &b) {
  A3 a;
  a.init(b);
  cudaMemcpy(a.v.data(), p, b.n3() * sizeof(fv_real), cudaMemcpyDeviceToHost);
  return a;
}

// One GPU scalar-transport run. Everything the test inspects afterwards is
// pulled back explicitly; nothing is inferred from a device-side assertion.
struct GpuRun {
  A3 fx, fy, q_out, fx2_tp, fy2_tp, q_i_dev, q_j_dev, ra_x_dev, ra_y_dev;
  fv3_gpu_status status = FV3_GPU_SUCCESS;
};

static GpuRun run_gpu(const Inputs &in, const ScalarTransportOptions &opt) {
  const Box &box = in.box;
  Dev d;
  GpuRun out;

  void *dq = d.up3(in.q), *dmass = d.up3(in.mass);
  void *dcrx = d.up3(in.crx), *dcry = d.up3(in.cry);
  void *dxfx = d.up3(in.xfx), *dyfx = d.up3(in.yfx);
  void *dmfx = d.up3(in.mfx), *dmfy = d.up3(in.mfy);
  void *darea = d.up2(in.area), *drarea = d.up2(in.rarea);
  void *ddxa = d.up2(in.dxa), *ddya = d.up2(in.dya);
  void *ddel6u = d.up2(in.del6_u), *ddel6v = d.up2(in.del6_v);
  void *dra_x = d.scratch(box), *dra_y = d.scratch(box);
  void *dqi = d.scratch(box), *dqj = d.scratch(box);
  void *dfx2t = d.scratch(box), *dfy2t = d.scratch(box);
  void *dd2 = d.scratch(box), *dfx2d = d.scratch(box), *dfy2d = d.scratch(box);
  void *dfx = d.scratch(box), *dfy = d.scratch(box);

  // Reuse of the corrected Milestone 3A.4 ra_x/ra_y launchers, including their
  // asymmetric extents. Not recomputed here.
  dgrid::launch_ra_x_cuda(in.b, V3<const fv_real>(dxfx, box), V2<const fv_real>(darea, box),
                          V3<fv_real>(dra_x, box), nullptr);
  dgrid::launch_ra_y_cuda(in.b, V3<const fv_real>(dyfx, box), V2<const fv_real>(darea, box),
                          V3<fv_real>(dra_y, box), nullptr);

  // d_sw transports every level identically -- there is no damped/undamped
  // split as in update_dz_d -- so mode 0 (all levels) is the authoritative
  // filter here and damp/ndif are never dereferenced.
  LevelFilter lf{};
  lf.npz = in.b.npz;
  lf.mode = 0;

  out.status = dgrid::launch_scalar_transport_cuda(
      in.b, opt, lf, V3<fv_real>(dq, box), V3<const fv_real>(dmass, box),
      V3<const fv_real>(dcrx, box), V3<const fv_real>(dcry, box),
      V3<const fv_real>(dxfx, box), V3<const fv_real>(dyfx, box),
      V3<const fv_real>(dmfx, box), V3<const fv_real>(dmfy, box),
      V2<const fv_real>(darea, box), V2<const fv_real>(drarea, box),
      V2<const fv_real>(ddxa, box), V2<const fv_real>(ddya, box),
      V2<const fv_real>(ddel6u, box), V2<const fv_real>(ddel6v, box),
      V3<const fv_real>(dra_x, box), V3<const fv_real>(dra_y, box), V3<fv_real>(dqi, box),
      V3<fv_real>(dqj, box), V3<fv_real>(dfx2t, box), V3<fv_real>(dfy2t, box),
      V3<fv_real>(dd2, box), V3<fv_real>(dfx2d, box), V3<fv_real>(dfy2d, box),
      V3<fv_real>(dfx, box), V3<fv_real>(dfy, box), nullptr);
  cudaDeviceSynchronize();

  out.fx = down3(dfx, box);
  out.fy = down3(dfy, box);
  out.q_out = down3(dq, box);
  out.fx2_tp = down3(dfx2t, box);
  out.fy2_tp = down3(dfy2t, box);
  out.q_i_dev = down3(dqi, box);
  out.q_j_dev = down3(dqj, box);
  out.ra_x_dev = down3(dra_x, box);
  out.ra_y_dev = down3(dra_y, box);
  return out;
}

// Isolated deln_flux run against pre-set fx/fy, used by Part G to measure the
// coefficient law without the fv_tp_2d contribution in the way.
static bool run_deln_only(const Inputs &in, int32_t nord, fv_real damp_c, fv_real da_min,
                          bool mass_weighted, const A3 &fx_in, const A3 &fy_in, A3 &fx_out,
                          A3 &fy_out, fv3_gpu_status &st) {
  const Box &box = in.box;
  Dev d;
  void *dq = d.up3(in.q), *dmass = d.up3(in.mass);
  void *drarea = d.up2(in.rarea);
  void *ddel6u = d.up2(in.del6_u), *ddel6v = d.up2(in.del6_v);
  void *dd2 = d.scratch(box), *dfx2 = d.scratch(box), *dfy2 = d.scratch(box);
  void *dfx = d.up3(fx_in), *dfy = d.up3(fy_in);
  if (!dq || !dfx) return false;

  LevelFilter lf{};
  lf.npz = in.b.npz;
  lf.mode = 0;
  st = dgrid::launch_deln_flux_transport_cuda(
      in.b, nord, damp_c, da_min, mass_weighted, lf, V3<const fv_real>(dq, box),
      V3<const fv_real>(dmass, box), V2<const fv_real>(ddel6u, box),
      V2<const fv_real>(ddel6v, box), V2<const fv_real>(drarea, box), V3<fv_real>(dfx, box),
      V3<fv_real>(dfy, box), V3<fv_real>(dd2, box), V3<fv_real>(dfx2, box),
      V3<fv_real>(dfy2, box), nullptr);
  cudaDeviceSynchronize();
  fx_out = down3(dfx, box);
  fy_out = down3(dfy, box);
  return true;
}

// ------------------------------------------------------------------ comparators

struct Stats {
  size_t n = 0, bad = 0, nonfinite = 0;
  double worst = 0.0;
  double min_v = 1e300, max_v = -1e300;
  int64_t wi = 0, wj = 0, wk = 0;
  double w_got = 0.0, w_ref = 0.0;
};

// Compare over an explicit Fortran write box only. Cells outside it are the
// caller's uninitialised workspace and carry no contract.
static Stats compare_box(const A3 &got, const A3 &ref, int64_t i0, int64_t i1, int64_t j0,
                         int64_t j1, int64_t nk, double tol = TOL) {
  Stats s;
  for (int64_t k = 0; k < nk; ++k)
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = i0; i <= i1; ++i) {
        const double a = (double)got(i, j, k), b = (double)ref(i, j, k);
        ++s.n;
        if (!std::isfinite(a) || !std::isfinite(b)) {
          ++s.nonfinite;
          continue;
        }
        s.min_v = std::fmin(s.min_v, a);
        s.max_v = std::fmax(s.max_v, a);
        const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
        const double rel = std::fabs(a - b) / scale;
        if (rel > s.worst) {
          s.worst = rel;
          s.wi = i;
          s.wj = j;
          s.wk = k;
          s.w_got = a;
          s.w_ref = b;
        }
        if (!close_rel(a, b, tol)) ++s.bad;
      }
  return s;
}

static bool report(const char *name, const Stats &s, bool require_variation = true) {
  bool ok = s.bad == 0 && s.nonfinite == 0;
  if (require_variation && s.max_v - s.min_v <= 0.0) {
    printf("    %-28s FAIL (degenerate: constant %.6g over %zu cells)\n", name, s.min_v, s.n);
    return false;
  }
  if (!ok)
    printf("    %-28s FAIL (%zu/%zu mismatched, %zu non-finite, worst rel %.3e at "
           "(%lld,%lld,%lld) got %.17g ref %.17g)\n",
           name, s.bad, s.n, s.nonfinite, s.worst, (long long)s.wi, (long long)s.wj,
           (long long)s.wk, s.w_got, s.w_ref);
  else
    printf("    %-28s ok   (%zu cells, worst rel %.3e, range [%.6g, %.6g])\n", name, s.n,
           s.worst, s.min_v, s.max_v);
  return ok;
}

// Intermediate-pass localisation. Reported ONLY on failure, so ordinary output
// stays readable while a divergence is attributed to the exact fv_tp_2d pass
// that produced it (fy2 -> q_i -> fx -> fx2 -> q_j -> fy) instead of only
// showing up in the final flux.
static bool localize(const char *tag, const char *what, const Stats &s) {
  if (s.bad == 0 && s.nonfinite == 0) return true;
  printf("    %-28s FAIL (%s: %zu/%zu mismatched, %zu non-finite, worst rel %.3e at "
         "(%lld,%lld,%lld) got %.17g ref %.17g)\n",
         tag, what, s.bad, s.n, s.nonfinite, s.worst, (long long)s.wi, (long long)s.wj,
         (long long)s.wk, s.w_got, s.w_ref);
  return false;
}

// How many cells differ at all, over a box. Used to prove a code path is LIVE
// rather than dead-but-agreeing.
static size_t count_diff(const A3 &a, const A3 &b, int64_t i0, int64_t i1, int64_t j0,
                         int64_t j1, int64_t nk) {
  size_t n = 0;
  for (int64_t k = 0; k < nk; ++k)
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = i0; i <= i1; ++i)
        if (!close_rel((double)a(i, j, k), (double)b(i, j, k))) ++n;
  return n;
}

static TransportBounds interior_bounds(const Box &box, int64_t npx, int64_t npy) {
  TransportBounds b{};
  b.is = box.is;
  b.ie = box.ie;
  b.js = box.js;
  b.je = box.je;
  b.npz = box.nk;
  b.nk = box.nk;  // scalar transport: nk == npz cell-centred levels
  b.halo_width = box.ng;
  b.ng = box.ng;
  b.npx = npx;
  b.npy = npy;
  b.cubed_sphere = 1;
  b.nested = 0;
  b.west_edge = (box.is == 1);
  b.east_edge = (box.ie + 1 == npx);
  b.south_edge = (box.js == 1);
  b.north_edge = (box.je + 1 == npy);
  return b;
}

// ------------------------------------------------- damping coefficient scaling
//
// Each del-2 iteration inside deln_flux applies rarea*del6_{u,v}, i.e. a
// Laplacian, which on this fixture's physical grid (area ~ 1e8 m^2) shrinks the
// result by ~1/area ~ 1e-8 per iteration. The authoritative coefficient
// (damp_c*da_min)**(nord+1) is exactly what compensates that, because da_min is
// a cell AREA (fv_control: the global minimum grid-cell area). Passing an O(1)
// da_min would leave the nord=1 diffusive flux ~1e-8 and the nord=2 flux ~1e-16
// of the transport flux -- underneath round-off -- and a liveness assertion
// would then fail for a purely dimensional reason rather than a code defect.
//
// GRID_DA_MIN matches the fixture's area scale, so the damping is physically
// proportioned exactly as in FV3.
static const double GRID_DA_MIN = 1.0e8;

// For tests that must isolate the del-2/4/6 STENCIL from the coefficient, pick
// damp_c so the diffusive flux stays comparable in magnitude across nord:
// undo the nord Laplacian shrink factors, then take the (nord+1)-th root
// because that is the power the coefficient is raised to.
static fv_real nord_damp_c(int32_t nord, double da_min) {
  const double target = std::pow(1.0e8, (double)nord);
  return (fv_real)(std::pow(target, 1.0 / (double)(nord + 1)) / da_min);
}

// Compare a full scalar-transport run against the host reference.
static bool check_against_reference(const char *tag, const Inputs &in,
                                    const ScalarTransportOptions &opt) {
  const Box &box = in.box;
  GpuRun g = run_gpu(in, opt);
  if (g.status != FV3_GPU_SUCCESS) {
    printf("    %-28s FAIL (launch_scalar_transport_cuda -> %d)\n", tag, (int)g.status);
    return false;
  }
  cudaError_t ce = cudaGetLastError();
  if (ce != cudaSuccess) {
    printf("    %-28s FAIL (CUDA error: %s)\n", tag, cudaGetErrorString(ce));
    return false;
  }

  TpResult r = h_fv_tp_2d(in, opt.hord, opt.mass_flux_weighting != 0);
  if (opt.deln_flux_enabled && opt.damp_c > fv_real(1.0e-4))
    h_deln_flux(r.q_out, in, opt.nord, opt.damp_c, opt.da_min,
                opt.mass_flux_weighting != 0, r.fx, r.fy);

  bool ok = true;
  // Localise first: the single-pass fluxes and the two intermediate states, on
  // their authoritative Fortran write boxes (tp_core.F90:161/168/186/196).
  ok &= localize(tag, "fy2 pass",
                 compare_box(g.fy2_tp, r.fy2, box.loi(), box.ie + box.ng, box.js,
                             box.je + 1, box.nk));
  ok &= localize(tag, "q_i", compare_box(g.q_i_dev, r.q_i, box.loi(), box.ie + box.ng,
                                         box.js, box.je, box.nk));
  ok &= localize(tag, "fx2 pass",
                 compare_box(g.fx2_tp, r.fx2, box.is, box.ie + 1, box.loj(),
                             box.je + box.ng, box.nk));
  ok &= localize(tag, "q_j", compare_box(g.q_j_dev, r.q_j, box.is, box.ie, box.loj(),
                                         box.je + box.ng, box.nk));
  ok &= report((std::string(tag) + " fx").c_str(),
               compare_box(g.fx, r.fx, box.is, box.ie + 1, box.js, box.je, box.nk));
  ok &= report((std::string(tag) + " fy").c_str(),
               compare_box(g.fy, r.fy, box.is, box.ie, box.js, box.je + 1, box.nk));
  ok &= report((std::string(tag) + " q(inout)").c_str(),
               compare_box(g.q_out, r.q_out, box.loi(), box.ie + box.ng, box.loj(),
                           box.je + box.ng, box.nk));
  return ok;
}

// ---------------------------------------------------------------------- Part A

static bool part_a() {
  printf("Part A: interior order-6 fv_tp_2d (no panel edge, no corner)\n");
  Box box;
  box.is = 9;
  box.ie = 20;
  box.js = 9;
  box.je = 20;
  box.nk = 4;
  TransportBounds b = interior_bounds(box, 33, 33);
  if (b.west_edge || b.east_edge || b.south_edge || b.north_edge) {
    printf("    fixture FAIL: interior block must not flag a panel edge\n");
    return false;
  }
  Inputs in = make_inputs(box, b);

  // Verify the reused ra_x/ra_y launchers first: everything downstream divides
  // by them, so a wrong extent there would surface as a diffuse flux error.
  GpuRun probe = run_gpu(in, ScalarTransportOptions{6, 0, 0, 0, fv_real(0), fv_real(0)});
  bool ok = probe.status == FV3_GPU_SUCCESS;
  if (!ok) printf("    ra probe FAIL (status %d)\n", (int)probe.status);
  ok &= report("ra_x (is..ie x jsd..jed)",
               compare_box(probe.ra_x_dev, in.ra_x, box.is, box.ie, box.loj(),
                           box.je + box.ng, box.nk));
  ok &= report("ra_y (isd..ied x js..je)",
               compare_box(probe.ra_y_dev, in.ra_y, box.loi(), box.ie + box.ng, box.js,
                           box.je, box.nk));

  ok &= check_against_reference("A hord=6", in,
                               ScalarTransportOptions{6, 0, 0, 0, fv_real(0), fv_real(0)});
  return ok;
}

// ---------------------------------------------------------------------- Part B

static bool part_b() {
  printf("Part B: panel-edge PPM (all four seams, no corner)\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  if (!(b.west_edge && b.east_edge && b.south_edge && b.north_edge)) {
    printf("    fixture FAIL: expected all four panel edges\n");
    return false;
  }
  Inputs in = make_inputs(box, b);
  const ScalarTransportOptions opt{6, 0, 0, 0, fv_real(0), fv_real(0)};
  bool ok = check_against_reference("B seam", in, opt);

  // fv_tp_2d's own fx2 = xppm(q, crx_adv) over is..ie+1 x jsd..jed survives in
  // the caller's workspace, and is a PURE x-direction PPM of a known field --
  // the cleanest available probe of the one-sided edge reconstruction.
  GpuRun g = run_gpu(in, opt);
  TpResult r = h_fv_tp_2d(in, 6, false);
  ok &= report("B fx2 = xppm(q)",
               compare_box(g.fx2_tp, r.fx2, box.is, box.ie + 1, box.loj(), box.je + box.ng,
                           box.nk));
  ok &= report("B fy2 = yppm(q)",
               compare_box(g.fy2_tp, r.fy2, box.loi(), box.ie + box.ng, box.js, box.je + 1,
                           box.nk));

  // Prove the seam path is LIVE: the same inputs with the cubed-sphere seam
  // switched off must differ, and differ only in the seam-influenced columns.
  TransportBounds bi = b;
  bi.cubed_sphere = 0;
  Inputs in_i = in;
  in_i.b = bi;
  GpuRun gi = run_gpu(in_i, opt);
  if (gi.status != FV3_GPU_SUCCESS) {
    printf("    B interior-stencil run FAIL (status %d)\n", (int)gi.status);
    return false;
  }
  const size_t nd =
      count_diff(g.fx, gi.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
  if (nd == 0) {
    printf("    %-28s FAIL (seam formulas inert: cubed_sphere=0 gave identical fx)\n",
           "B seam is live");
    ok = false;
  } else {
    printf("    %-28s ok   (%zu of %zu fx cells changed)\n", "B seam is live", nd,
           (size_t)(box.ie + 2 - box.is) * (box.je + 1 - box.js) * box.nk);
  }
  return ok;
}

// ---------------------------------------------------------------------- Part C

static bool part_c() {
  printf("Part C: corner behaviour (copy_corners rotation of q, all four corners)\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);
  const ScalarTransportOptions opt{6, 0, 0, 0, fv_real(0), fv_real(0)};

  bool ok = check_against_reference("C corners", in, opt);

  // q must come back rotated exactly as copy_corners(dir=2) then (dir=1) leaves
  // it, and must NOT equal the input -- a no-op copy_corners would otherwise
  // pass the reference check trivially if the reference were also a no-op.
  GpuRun g = run_gpu(in, opt);
  A3 qref = in.q;
  h_copy_corners(qref, 2, b);
  h_copy_corners(qref, 1, b);
  ok &= report("C q after rotations",
               compare_box(g.q_out, qref, box.loi(), box.ie + box.ng, box.loj(),
                           box.je + box.ng, box.nk));
  const size_t moved = count_diff(g.q_out, in.q, box.loi(), box.ie + box.ng, box.loj(),
                                  box.je + box.ng, box.nk);
  if (moved == 0) {
    printf("    %-28s FAIL (copy_corners wrote nothing)\n", "C rotation is live");
    ok = false;
  } else {
    printf("    %-28s ok   (%zu halo cells rotated)\n", "C rotation is live", moved);
  }

  // The corner path must not be routed through the generic corner_edge_service
  // dispatch slot: WP-B does its own device-level rotation. Nothing here
  // supplies a corner service, so a dependency on one would fail to launch.
  TransportBounds bn = b;
  bn.nested = 1;
  Inputs in_n = in;
  in_n.b = bn;
  GpuRun gn = run_gpu(in_n, opt);
  const size_t nested_moved = count_diff(gn.q_out, in.q, box.loi(), box.ie + box.ng,
                                         box.loj(), box.je + box.ng, box.nk);
  if (gn.status != FV3_GPU_SUCCESS || nested_moved != 0) {
    printf("    %-28s FAIL (nested must suppress copy_corners; %zu cells moved)\n",
           "C nested suppression", nested_moved);
    ok = false;
  } else {
    printf("    %-28s ok   (0 cells rotated)\n", "C nested suppression");
  }
  return ok;
}

// ---------------------------------------------------------------------- Part D

static bool part_d() {
  printf("Part D: deln_flux ENABLED (nord = 0, 1, 2)\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);

  bool ok = true;
  const ScalarTransportOptions off{6, 0, 0, 0, fv_real(0), fv_real(0)};
  GpuRun base = run_gpu(in, off);
  for (int32_t nord = 0; nord <= 2; ++nord) {
    ScalarTransportOptions opt{6, 0, 1, nord, fv_real(0.05), (fv_real)GRID_DA_MIN};
    char tag[64];
    std::snprintf(tag, sizeof(tag), "D nord=%d", (int)nord);
    ok &= check_against_reference(tag, in, opt);
    // The damping must actually change the flux.
    GpuRun g = run_gpu(in, opt);
    const size_t nd = count_diff(g.fx, base.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
    if (nd == 0) {
      printf("    %-28s FAIL (deln_flux contributed nothing)\n", tag);
      ok = false;
    } else {
      printf("    %-28s ok   (%zu fx cells changed vs. no-deln)\n",
             (std::string(tag) + " is live").c_str(), nd);
    }
  }
  return ok;
}

// ---------------------------------------------------------------------- Part E

static bool part_e() {
  printf("Part E: deln_flux DISABLED, and the damp_c <= 1.e-4 no-op branch\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);

  const ScalarTransportOptions off_opt{6, 0, 0, 2, fv_real(0.05), (fv_real)GRID_DA_MIN};
  GpuRun off = run_gpu(in, off_opt);
  bool ok = check_against_reference("E deln off", in, off_opt);

  // fv_tp_2d's threshold is damp_c > 1.e-4 (NOT del6_vt_flux's 1.E-5). At
  // damp_c = 1.e-4 exactly the branch must not fire.
  //
  // These cases run at nord = 0. The threshold is nord-independent, and nord = 0
  // is the only order whose diffusive contribution at a damp_c of 2e-4 is still
  // well above round-off (see GRID_DA_MIN): a higher nord would make the
  // "damping applied" case indistinguishable from the no-op case for a reason
  // that has nothing to do with the threshold being tested.
  const ScalarTransportOptions off0{6, 0, 0, 0, fv_real(0.05), (fv_real)GRID_DA_MIN};
  GpuRun off0_run = run_gpu(in, off0);

  // With deln_flux disabled, nord must be completely inert.
  {
    const size_t nd =
        count_diff(off.fx, off0_run.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
    printf("    %-28s %s (%zu fx cells differ)\n", "E nord inert when off",
           nd == 0 ? "ok  " : "FAIL", nd);
    if (nd != 0) ok = false;
  }

  struct Case {
    const char *name;
    fv_real damp_c;
    bool expect_identical;
  } cases[] = {
      {"E damp_c = 0", fv_real(0), true},
      {"E damp_c = 1e-5", fv_real(1.0e-5), true},
      {"E damp_c = 1e-4 (==thr)", fv_real(1.0e-4), true},
      {"E damp_c = 2e-4 (>thr)", fv_real(2.0e-4), false},
  };
  for (const Case &c : cases) {
    GpuRun g = run_gpu(in, ScalarTransportOptions{6, 0, 1, 0, c.damp_c, (fv_real)GRID_DA_MIN});
    if (g.status != FV3_GPU_SUCCESS) {
      printf("    %-28s FAIL (status %d)\n", c.name, (int)g.status);
      ok = false;
      continue;
    }
    const size_t nd =
        count_diff(g.fx, off0_run.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
    const bool identical = nd == 0;
    if (identical != c.expect_identical) {
      printf("    %-28s FAIL (%zu fx cells differ; expected %s)\n", c.name, nd,
             c.expect_identical ? "no change" : "a change");
      ok = false;
    } else {
      printf("    %-28s ok   (%s)\n", c.name, identical ? "no-op" : "damping applied");
    }
  }
  return ok;
}

// ---------------------------------------------------------------------- Part F

static bool part_f() {
  printf("Part F: nord-dependent behaviour\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 2;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);

  const double da_min = 1.0;
  bool ok = true;
  A3 zero_pre;
  zero_pre.init(box);
  A3 prev;
  bool have_prev = false;
  for (int32_t nord = 0; nord <= 2; ++nord) {
    // Full path at the physically-proportioned coefficient: each order must
    // reproduce the host reference exactly.
    char tag[64];
    std::snprintf(tag, sizeof(tag), "F nord=%d stencil", (int)nord);
    ok &= check_against_reference(
        tag, in, ScalarTransportOptions{6, 0, 1, nord, fv_real(0.05), (fv_real)GRID_DA_MIN});

    // Distinctness is measured on the ISOLATED diffusive flux (deln_flux run
    // against a zeroed fx/fy) with the coefficient normalised per nord. Both
    // choices matter: differencing against the ~1e8-magnitude transport flux
    // would hide the del-4/del-6 contribution under the comparison scale, and a
    // fixed coefficient would let the nord >= 1 stencils shrink into round-off.
    // What is left varying is only the stencil order.
    A3 fx_n, fy_n;
    fv3_gpu_status s = FV3_GPU_SUCCESS;
    run_deln_only(in, nord, nord_damp_c(nord, da_min), (fv_real)da_min, false, zero_pre,
                  zero_pre, fx_n, fy_n, s);
    if (s != FV3_GPU_SUCCESS) {
      printf("    %-28s FAIL (isolated deln_flux -> %d)\n", tag, (int)s);
      ok = false;
    }
    if (have_prev) {
      const size_t nd = count_diff(fx_n, prev, box.is, box.ie + 1, box.js, box.je, box.nk);
      if (nd == 0) {
        printf("    %-28s FAIL (nord=%d gave the same fx as nord=%d)\n", "F orders distinct",
               (int)nord, (int)nord - 1);
        ok = false;
      } else {
        printf("    %-28s ok   (nord=%d vs %d: %zu fx cells differ)\n", "F orders distinct",
               (int)nord, (int)nord - 1, nd);
      }
    }
    prev = fx_n;
    have_prev = true;
  }

  // nord == 0 must perform ZERO copy_corners rotations of d2 (tp_core.F90:1313
  // and :1326 are both guarded on nord > 0, and the iteration loop does not
  // run). Setting all four corner flags must therefore not change the nord=0
  // result relative to a corner-free block, once the transport contribution is
  // removed by isolating deln_flux.
  {
    A3 zero;
    zero.init(box);
    A3 fx_c, fy_c, fx_n, fy_n;
    fv3_gpu_status s1 = FV3_GPU_SUCCESS, s2 = FV3_GPU_SUCCESS;
    Inputs in_nc = in;
    in_nc.b.sw_corner = in_nc.b.se_corner = in_nc.b.ne_corner = in_nc.b.nw_corner = 0;
    run_deln_only(in, 0, nord_damp_c(0, 1.0), fv_real(1), false, zero, zero, fx_c, fy_c, s1);
    run_deln_only(in_nc, 0, nord_damp_c(0, 1.0), fv_real(1), false, zero, zero, fx_n, fy_n,
                  s2);
    const size_t nd = count_diff(fx_c, fx_n, box.is, box.ie + 1, box.js, box.je, box.nk);
    if (s1 != FV3_GPU_SUCCESS || s2 != FV3_GPU_SUCCESS || nd != 0) {
      printf("    %-28s FAIL (%zu cells differ; nord=0 must not rotate corners)\n",
             "F nord=0 no rotation", nd);
      ok = false;
    } else {
      printf("    %-28s ok   (corner flags inert at nord=0)\n", "F nord=0 no rotation");
    }
    // ...whereas nord >= 1 MUST be corner-sensitive (that is defect D3).
    run_deln_only(in, 1, nord_damp_c(1, 1.0), fv_real(1), false, zero, zero, fx_c, fy_c, s1);
    run_deln_only(in_nc, 1, nord_damp_c(1, 1.0), fv_real(1), false, zero, zero, fx_n, fy_n,
                  s2);
    const size_t nd1 = count_diff(fx_c, fx_n, box.is, box.ie + 1, box.js, box.je, box.nk);
    if (nd1 == 0) {
      printf("    %-28s FAIL (nord=1 ignored the corner rotations -- defect D3)\n",
             "F nord=1 rotates corners");
      ok = false;
    } else {
      printf("    %-28s ok   (%zu fx cells corner-sensitive)\n", "F nord=1 rotates corners",
             nd1);
    }
  }
  return ok;
}

// ---------------------------------------------------------------------- Part G
//
// THE del6_vt_flux DISCRIMINATOR.
//
// deln_flux's coefficient is (damp_c * da_min)**(nord+1); del6_vt_flux's is the
// raw per-level damp(k). With mass absent, d2 = damp*q and every subsequent
// operation is linear, so the diffusive flux is exactly proportional to damp.
// Doubling damp_c must therefore scale it by 2**(nord+1) -- 2, 4, 8 for
// nord = 0, 1, 2 -- and NOT by 2. A port that reused del6_vt_flux's raw-damp
// semantics would give 2 for every nord and fail at nord >= 1.

static bool part_g() {
  printf("Part G: damp_c / da_min coefficient law (del6_vt_flux discriminator)\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 2;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);

  A3 zero;
  zero.init(box);
  bool ok = true;

  for (int32_t nord = 0; nord <= 2; ++nord) {
    const double expect = std::pow(2.0, (double)(nord + 1));
    // The ratio is exact for ANY base damp_c because everything downstream of
    // d2 = damp*q is linear; the base is normalised per nord only so the
    // measured flux stays clear of round-off (see GRID_DA_MIN / nord_damp_c).
    const fv_real base = nord_damp_c(nord, 1.0);
    A3 f1, g1, f2, g2, f3, g3;
    fv3_gpu_status s = FV3_GPU_SUCCESS;
    run_deln_only(in, nord, base, fv_real(1.0), false, zero, zero, f1, g1, s);
    run_deln_only(in, nord, fv_real(2) * base, fv_real(1.0), false, zero, zero, f2, g2, s);
    // da_min scaling enters the coefficient identically (it is the same product)
    run_deln_only(in, nord, base, fv_real(2.0), false, zero, zero, f3, g3, s);

    double worst_c = 0.0, worst_a = 0.0;
    size_t used = 0;
    for (int64_t k = 0; k < box.nk; ++k)
      for (int64_t j = box.js; j <= box.je; ++j)
        for (int64_t i = box.is; i <= box.ie + 1; ++i) {
          const double a = (double)f1(i, j, k);
          if (std::fabs(a) < 1e-20) continue;  // ratio undefined at an exact zero
          ++used;
          worst_c = std::fmax(worst_c, std::fabs((double)f2(i, j, k) / a - expect) / expect);
          worst_a = std::fmax(worst_a, std::fabs((double)f3(i, j, k) / a - expect) / expect);
        }
    const double rtol = sizeof(fv_real) == 4 ? 1e-4 : 1e-10;
    const bool pass = used > 100 && worst_c <= rtol && worst_a <= rtol;
    printf("    %-28s %s (nord=%d: expect x%.0f, worst dev damp_c %.2e / da_min %.2e over "
           "%zu cells)\n",
           "G coefficient**(nord+1)", pass ? "ok  " : "FAIL", (int)nord, expect, worst_c,
           worst_a, used);
    if (!pass) ok = false;

    // Explicitly reject the del6_vt_flux (raw-damp, linear) law for nord >= 1.
    if (nord >= 1) {
      const bool looks_linear = std::fabs(expect - 2.0) > 1e-12 && worst_c <= rtol &&
                                std::fabs(2.0 - expect) > rtol;
      if (!looks_linear) {
        printf("    %-28s FAIL (could not distinguish from raw-damp linearity)\n",
               "G not raw-damp linear");
        ok = false;
      } else {
        printf("    %-28s ok   (x%.0f, not x2)\n", "G not raw-damp linear", expect);
      }
    }
  }
  return ok;
}

// ---------------------------------------------------------------------- Part H

static bool part_h() {
  printf("Part H: mass-flux weighting, mass-weighted deln_flux, hord<0 positivity\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);
  bool ok = true;

  // The four authoritative d_sw modes (sw_core.F90:939, 986, 993, 1004).
  // delp: xfx/yfx weighting + deln_flux(nord_v, damp_v), mass ABSENT.
  ok &= check_against_reference("H delp (hord_dp=-6)", in,
                                ScalarTransportOptions{-6, 0, 1, 2, fv_real(0.05),
                                                       (fv_real)GRID_DA_MIN});
  // w: mfx/mfy weighting, no deln_flux.
  ok &= check_against_reference("H w (hord_vt=6)", in,
                                ScalarTransportOptions{6, 1, 0, 0, fv_real(0), fv_real(0)});
  // pt / q_con: mfx/mfy weighting + deln_flux(nord_t, damp_t) WITH mass = delp.
  ok &= check_against_reference("H pt (hord_tm=6, mass)", in,
                                ScalarTransportOptions{6, 1, 1, 2, fv_real(0.05),
                                                       (fv_real)GRID_DA_MIN});
  ok &= check_against_reference("H q_con (hord_dp=-6,mass)", in,
                                ScalarTransportOptions{-6, 1, 1, 2, fv_real(0.05),
                                                       (fv_real)GRID_DA_MIN});

  // Mass weighting must actually change the deln_flux contribution, and must do
  // so via the 0.5*damp*(m(i-1)+m(i)) face average rather than the plain add.
  {
    A3 zero, fm, gm, fn, gn;
    zero.init(box);
    fv3_gpu_status s = FV3_GPU_SUCCESS;
    run_deln_only(in, 2, fv_real(0.05), (fv_real)GRID_DA_MIN, true, zero, zero, fm, gm, s);
    run_deln_only(in, 2, fv_real(0.05), (fv_real)GRID_DA_MIN, false, zero, zero, fn, gn, s);
    const size_t nd = count_diff(fm, fn, box.is, box.ie + 1, box.js, box.je, box.nk);
    printf("    %-28s %s (%zu fx cells differ from the unweighted add)\n",
           "H mass weighting is live", nd > 0 ? "ok  " : "FAIL", nd);
    if (nd == 0) ok = false;
  }

  // Positivity: q_mode 1 is a positive field with a sharp step, which drives the
  // order-6 reconstruction below zero near the jump. hord = -6 must clamp
  // al to >= 0 and therefore differ from hord = +6.
  {
    Inputs step = make_inputs(box, b, /*q_mode=*/1);
    ok &= check_against_reference("H step hord=+6", step,
                                  ScalarTransportOptions{6, 0, 0, 0, fv_real(0),
                                                         fv_real(0)});
    ok &= check_against_reference("H step hord=-6", step,
                                  ScalarTransportOptions{-6, 0, 0, 0, fv_real(0),
                                                         fv_real(0)});
    GpuRun gp = run_gpu(step, ScalarTransportOptions{6, 0, 0, 0, fv_real(0), fv_real(0)});
    GpuRun gn = run_gpu(step, ScalarTransportOptions{-6, 0, 0, 0, fv_real(0), fv_real(0)});
    const size_t nd = count_diff(gp.fx, gn.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
    printf("    %-28s %s (%zu fx cells changed by the al>=0 clamp)\n",
           "H hord<0 clamp is live", nd > 0 ? "ok  " : "FAIL", nd);
    if (nd == 0) ok = false;

    // Sanity: the host reference confirms the clamp fires, i.e. some
    // unclamped al really is negative. If not, the test above proves nothing.
    size_t neg = 0;
    A3 qc = step.q;
    h_copy_corners(qc, 2, step.b);
    h_copy_corners(qc, 1, step.b);
    for (int64_t k = 0; k < box.nk; ++k)
      for (int64_t j = box.js; j <= box.je; ++j)
        for (int64_t i = box.is - 1; i <= box.ie + 2; ++i)
          if (h_al_x(qc, step.dxa, i, j, k, step.b, false) < fv_real(0)) ++neg;
    printf("    %-28s %s (%zu unclamped al values are negative)\n", "H clamp precondition",
           neg > 0 ? "ok  " : "FAIL", neg);
    if (neg == 0) ok = false;
  }

  // The round-off-determined limiter branch, tested head-on rather than avoided.
  //
  // q_mode 2 is exactly constant along i inside each plateau, so al(i) - q(i) is
  // mathematically zero and the mord=6 limiter is decided by the round-off sign
  // of a quantity the compiler is free to compute either way. The device is NOT
  // required to reproduce the host branch. It IS required to land exactly on one
  // of the two admissible branch values -- anything else would mean the flux
  // arithmetic itself, not just the branch, differs from the authoritative
  // formula. This is checked on the single-pass fx2 = xppm(q), which is the
  // first place the decision is taken, before any downstream mixing.
  {
    Inputs flat = make_inputs(box, b, /*q_mode=*/2);
    const ScalarTransportOptions opt{6, 0, 0, 0, fv_real(0), fv_real(0)};
    GpuRun g = run_gpu(flat, opt);
    A3 qc = flat.q;
    h_copy_corners(qc, 2, flat.b);
    h_copy_corners(qc, 1, flat.b);  // fx2 sees q after BOTH rotations

    size_t n = 0, off_branch = 0, undecided = 0, bad = 0;
    double worst = 0.0;
    int64_t bi = 0, bj = 0, bk = 0;
    for (int64_t k = 0; k < box.nk; ++k)
      for (int64_t j = box.loj(); j <= box.je + box.ng; ++j)
        for (int64_t i = box.is; i <= box.ie + 1; ++i) {
          const XppmAlt a =
              h_xppm_alt(qc, flat.dxa, i, j, k, flat.crx(i, j, k), flat.b, false);
          const double got = (double)g.fx2_tp(i, j, k);
          ++n;
          if (!a.decided) ++undecided;
          const bool on = close_rel(got, (double)a.on, TOL);
          const bool of = close_rel(got, (double)a.off, TOL);
          if (!on && !of) {
            ++bad;
            const double d = std::fmin(std::fabs(got - (double)a.on),
                                       std::fabs(got - (double)a.off));
            if (d > worst) { worst = d; bi = i; bj = j; bk = k; }
          } else if (!on) {
            // matched the suppressed branch; only interesting when the two differ
            if (!close_rel((double)a.on, (double)a.off, TOL)) ++off_branch;
          }
        }
    if (bad != 0) {
      printf("    %-28s FAIL (%zu/%zu of %zu cells matched NEITHER branch, worst abs "
             "%.3e at (%lld,%lld,%lld))\n",
             "H limiter branch admissible", bad, n, n, worst, (long long)bi, (long long)bj,
             (long long)bk);
      ok = false;
    } else {
      printf("    %-28s ok   (%zu cells, all on an admissible branch; %zu round-off-"
             "undecided, %zu took the suppressed branch)\n",
             "H limiter branch admissible", n, undecided, off_branch);
    }
    // The fixture must actually exercise the indeterminacy, else this proves
    // nothing about it.
    if (undecided == 0) {
      printf("    %-28s FAIL (fixture produced no round-off-determined decision)\n",
             "H tie precondition");
      ok = false;
    } else {
      printf("    %-28s ok   (%zu round-off-determined limiter decisions)\n",
             "H tie precondition", undecided);
    }
  }
  return ok;
}

// ------------------------------------------------------------------ Parts I / J
//
// One build per precision, so exactly one of these runs and the other reports
// itself not applicable. Both drive the identical configuration; only the
// asserted tolerance differs.

static bool precision_case(const char *tag, double tol) {
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 3;
  TransportBounds b = interior_bounds(box, 13, 13);
  b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
  Inputs in = make_inputs(box, b);
  const ScalarTransportOptions opt{-6, 1, 1, 2, fv_real(0.05), (fv_real)GRID_DA_MIN};

  GpuRun g = run_gpu(in, opt);
  if (g.status != FV3_GPU_SUCCESS) {
    printf("    %-28s FAIL (status %d)\n", tag, (int)g.status);
    return false;
  }
  TpResult r = h_fv_tp_2d(in, opt.hord, true);
  h_deln_flux(r.q_out, in, opt.nord, opt.damp_c, opt.da_min, true, r.fx, r.fy);
  bool ok = report(tag, compare_box(g.fx, r.fx, box.is, box.ie + 1, box.js, box.je, box.nk,
                                    tol));
  ok &= report((std::string(tag) + " fy").c_str(),
               compare_box(g.fy, r.fy, box.is, box.ie, box.js, box.je + 1, box.nk, tol));
  return ok;
}

static bool part_i() {
  printf("Part I: R8 precision\n");
  if (sizeof(fv_real) != 8) {
    printf("    not applicable in this build (fv_real is %zu bytes)\n", sizeof(fv_real));
    return true;
  }
  return precision_case("I R8 full path", 1e-11);
}

static bool part_j() {
  printf("Part J: R4 precision\n");
  if (sizeof(fv_real) != 4) {
    printf("    not applicable in this build (fv_real is %zu bytes)\n", sizeof(fv_real));
    return true;
  }
  return precision_case("J R4 full path", 2e-5);
}

// ---------------------------------------------------------------------- Part K

static bool part_k() {
  printf("Part K: unsupported configurations rejected before any kernel launch\n");
  Box box;
  box.is = 1;
  box.ie = 12;
  box.js = 1;
  box.je = 12;
  box.nk = 2;
  TransportBounds b = interior_bounds(box, 13, 13);
  Inputs in = make_inputs(box, b);

  struct Case {
    const char *name;
    ScalarTransportOptions opt;
  } cases[] = {
      // hord=8 is supported as of Prompt 76 (see Part L); only hord magnitudes
      // that remain genuinely unimplemented are rejected here now.
      {"K hord=10 rejected", {10, 0, 0, 0, fv_real(0), fv_real(0)}},
      {"K hord=0 rejected", {0, 0, 0, 0, fv_real(0), fv_real(0)}},
      {"K nord=3 rejected", {6, 0, 1, 3, fv_real(0.05), (fv_real)GRID_DA_MIN}},
      {"K nord=-1 rejected", {6, 0, 1, -1, fv_real(0.05), (fv_real)GRID_DA_MIN}},
  };
  bool ok = true;
  for (const Case &c : cases) {
    GpuRun g = run_gpu(in, c.opt);
    if (g.status != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      printf("    %-28s FAIL (status %d, expected UNSUPPORTED_CONFIGURATION)\n", c.name,
             (int)g.status);
      ok = false;
      continue;
    }
    // A rejected configuration must leave the caller's state untouched.
    const size_t moved = count_diff(g.q_out, in.q, box.loi(), box.ie + box.ng, box.loj(),
                                    box.je + box.ng, box.nk);
    if (moved != 0) {
      printf("    %-28s FAIL (rejected but mutated q in %zu cells)\n", c.name, moved);
      ok = false;
    } else {
      printf("    %-28s ok   (rejected, q untouched)\n", c.name);
    }
  }
  return ok;
}

// ---------------------------------------------------------------------- Part L
//
// hord==8 (Prompt 76): interior, all-four-seam, and corner check_against_reference
// coverage, seeded with the 11 named Prompt-75 fixture *shapes*
// (tests/host/test_fv_tp_2d_hord8_hostharness.cpp) broadcast across j and k --
// that harness is row-based/1D and does not share this file's 3D Box/A3
// convention, so it cannot be reused directly, but its fixture patterns
// (constant, linear_gradient, smooth_curved, sharp_front, local_max, local_min,
// near_zero_positive, sign_changing, high_courant_valid,
// boundary_edge_is1_and_npx, c512_like_slice) are reproduced here as the
// x-varying q(i)/crx patterns that each fixture actually exercises.
// check_against_reference itself compares fully against h_fv_tp_2d(hord=8),
// which now dispatches to the bitwise device-mirroring h_xppm8/h_yppm8/
// h_bl_br8_x/h_bl_br8_y added above -- so a PASS here is a bitwise (not just
// round-off) claim unless compare_box's TOL is actually exercised.

static void set_fixture_x(Inputs &in, int64_t imid, fv_real c,
                          const std::function<double(int64_t)> &qshape) {
  const Box &box = in.box;
  const int64_t i0 = box.loi(), i1 = box.ie + box.ng;
  const int64_t j0 = box.loj(), j1 = box.je + box.ng;
  for (int64_t k = 0; k < box.nk; ++k)
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = i0; i <= i1; ++i) {
        in.q(i, j, k) = (fv_real)qshape(i);
        in.crx(i, j, k) = c;
      }
  (void)imid;
}

static bool part_l() {
  printf("Part L: hord=8 interior/seam/corner fv_tp_2d (Prompt 76 Stage 5)\n");
  bool ok = true;

  // L1 -- interior (no panel edge), 9 of the 11 Prompt-75 fixture shapes that
  // do not specifically target a panel edge / production-shape slice.
  {
    Box box;
    box.is = 9;
    box.ie = 20;
    box.js = 9;
    box.je = 20;
    box.nk = 4;
    TransportBounds b = interior_bounds(box, 33, 33);
    const int64_t imid = (box.is + box.ie) / 2;
    struct Fixture {
      const char *name;
      fv_real c;
      std::function<double(int64_t)> q;
    } fixtures[] = {
        {"L1 constant", fv_real(0.35), [](int64_t) { return 5.0; }},
        {"L1 linear_gradient", fv_real(0.35), [](int64_t i) { return 1.0 + 0.5 * i; }},
        {"L1 smooth_curved", fv_real(0.35),
         [](int64_t i) { return 3.0 + std::sin(0.3 * i); }},
        {"L1 sharp_front", fv_real(0.35),
         [imid](int64_t i) { return i < imid ? 1.0 : 10.0; }},
        {"L1 local_max", fv_real(0.35),
         [imid](int64_t i) { return i == imid ? 20.0 : 2.0; }},
        {"L1 local_min", fv_real(0.35),
         [imid](int64_t i) { return i == imid ? -5.0 : 2.0; }},
        {"L1 near_zero_positive", fv_real(0.2),
         [](int64_t i) { return 1.0e-12 + 1.0e-14 * i; }},
        {"L1 sign_changing", fv_real(-0.4),
         [](int64_t i) { return std::sin(0.5 * i) * 4.0; }},
        {"L1 high_courant_valid", fv_real(0.9), [](int64_t i) { return 1.0 + 0.3 * i; }},
    };
    for (const Fixture &f : fixtures) {
      Inputs in = make_inputs(box, b);
      set_fixture_x(in, imid, f.c, f.q);
      ok &= check_against_reference(f.name, in, ScalarTransportOptions{8, 0, 0, 0,
                                                                       fv_real(0),
                                                                       fv_real(0)});
    }
  }

  // L2 -- all four panel-edge seams (Prompt-75 "boundary_edge_is1_and_npx").
  {
    Box box;
    box.is = 1;
    box.ie = 12;
    box.js = 1;
    box.je = 12;
    box.nk = 3;
    TransportBounds b = interior_bounds(box, 13, 13);
    if (!(b.west_edge && b.east_edge && b.south_edge && b.north_edge)) {
      printf("    fixture FAIL: expected all four panel edges\n");
      ok = false;
    } else {
      Inputs in = make_inputs(box, b);
      set_fixture_x(in, 0, fv_real(0.3), [](int64_t i) { return 1.0 + 0.4 * i; });
      ok &= check_against_reference("L2 boundary_edge_is1_and_npx", in,
                                    ScalarTransportOptions{8, 0, 0, 0, fv_real(0),
                                                            fv_real(0)});
    }
  }

  // L3 -- all four corners, "c512_like_slice" (smooth field + small
  // deterministic perturbation, standing in for the fixture's noise term --
  // a real RNG would make the test non-reproducible).
  {
    Box box;
    box.is = 1;
    box.ie = 12;
    box.js = 1;
    box.je = 12;
    box.nk = 3;
    TransportBounds b = interior_bounds(box, 13, 13);
    b.sw_corner = b.se_corner = b.ne_corner = b.nw_corner = 1;
    Inputs in = make_inputs(box, b);
    set_fixture_x(in, 0, fv_real(0.42), [](int64_t i) {
      return 250.0 + 5.0 * std::sin(0.2 * i) + 0.01 * std::sin(12.9898 * i);
    });
    ok &= check_against_reference("L3 c512_like_slice", in,
                                  ScalarTransportOptions{8, 0, 0, 0, fv_real(0),
                                                          fv_real(0)});
  }

  // L4 -- hord=8 is genuinely dispatched (not silently falling back to the
  // mord==6 kernel or a CPU path): fx must actually change vs. hord=6 on the
  // same interior fixture, mirroring Part B's "seam is live" check.
  {
    Box box;
    box.is = 9;
    box.ie = 20;
    box.js = 9;
    box.je = 20;
    box.nk = 4;
    TransportBounds b = interior_bounds(box, 33, 33);
    Inputs in = make_inputs(box, b);
    GpuRun g6 = run_gpu(in, ScalarTransportOptions{6, 0, 0, 0, fv_real(0), fv_real(0)});
    GpuRun g8 = run_gpu(in, ScalarTransportOptions{8, 0, 0, 0, fv_real(0), fv_real(0)});
    ok &= (g6.status == FV3_GPU_SUCCESS) && (g8.status == FV3_GPU_SUCCESS);
    const size_t nd = count_diff(g6.fx, g8.fx, box.is, box.ie + 1, box.js, box.je, box.nk);
    if (nd == 0) {
      printf("    %-28s FAIL (hord=8 fx identical to hord=6 -- not dispatched?)\n",
             "L4 hord8 is live");
      ok = false;
    } else {
      printf("    %-28s ok   (%zu of %zu fx cells differ from hord=6)\n", "L4 hord8 is live",
             nd, (size_t)(box.ie + 2 - box.is) * (box.je + 1 - box.js) * box.nk);
    }
  }

  return ok;
}

// ---------------------------------------------------------------------------

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    printf("SKIP: no CUDA device visible (login node). GPU execution must run under "
           "Slurm gpu_a100.\n");
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  printf("WP-B.2 scalar transport CUDA test: device '%s' sm_%d%d, fv_real = %zu bytes "
         "(%s)\n\n",
         prop.name, prop.major, prop.minor, sizeof(fv_real),
         sizeof(fv_real) == 8 ? "R8" : "R4");

  struct Part {
    const char *name;
    bool (*fn)();
  } parts[] = {{"A", part_a}, {"B", part_b}, {"C", part_c}, {"D", part_d}, {"E", part_e},
               {"F", part_f}, {"G", part_g}, {"H", part_h}, {"I", part_i}, {"J", part_j},
               {"K", part_k}, {"L", part_l}};

  int failed = 0;
  std::string failed_names;
  for (const Part &p : parts) {
    const bool ok = p.fn();
    printf("  -> Part %s: %s\n\n", p.name, ok ? "PASS" : "FAIL");
    if (!ok) {
      ++failed;
      failed_names += std::string(failed_names.empty() ? "" : ",") + p.name;
    }
  }

  const cudaError_t last = cudaGetLastError();
  if (last != cudaSuccess) {
    printf("Residual CUDA error: %s\n", cudaGetErrorString(last));
    ++failed;
  }

  if (failed) {
    printf("WP-B.2 scalar transport: %d part(s) FAILED (%s)\n", failed, failed_names.c_str());
    return 1;
  }
  printf("WP-B.2 scalar transport: ALL PARTS PASS (%zu parts)\n",
         sizeof(parts) / sizeof(parts[0]));
  return 0;
}
