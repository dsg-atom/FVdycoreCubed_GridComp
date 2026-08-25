// ===========================================================================
// Milestone 3A.6 / WP-B.3 -- authoritative D-grid momentum transport test.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45, model/sw_core.F90.
//
// Every numerical part compares the CUDA result against the host transcriptions
// ref_xtp_u / ref_ytp_v / ref_vb / ref_ub / ref_ke_* / ref_vorticity /
// ref_dwind below. Those transcriptions are written in the FORTRAN's OWN
// multi-pass structure -- poisoned automatic locals, the clipped interior loop,
// then the west/east (south/north) override blocks, then b0, then the order-6
// smoothness flag, then the WMP seam fix, then the flux loop -- whereas the CUDA
// kernels resolve the same overwrite order per point. The two therefore differ
// structurally and agreement is evidence, not tautology. The GPU is never
// compared against itself.
//
// The NaN poison is load-bearing: if the CUDA per-point resolution covered a
// cell the Fortran's pass structure leaves undefined, the reference would carry
// a NaN into flux and the comparison would fail rather than silently agree.
//
// Fixtures:
//   seam     npx = npy = 13, is/js = 1, ie/je = 12 -- owns all four tile edges
//            and all four corners. Exactly fits ng = 3: bl(0) reads u(-2) and
//            isd = -2; br(npx) reads u(npx+2) = u(15) and ied + 1 = 16.
//   interior npx = npy = 25, is/js = 7, ie/je = 18 -- owns no edge, so every
//            reconstruction takes the unclipped interior branch. Satisfies the
//            decomposition fence (is >= 4, ie+1 = 19 <= npx-3 = 22).
//
// Exit codes: 0 pass, 1 fail, 77 no CUDA device (CTest SKIP_RETURN_CODE).
// ===========================================================================

#include "dgrid/wp_b_momentum_cuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using fv3::gpu::fv_real;
using fv3::gpu::operators::View2;
using fv3::gpu::operators::View3;
namespace dgrid = fv3::gpu::dgrid;

// R4 carries ~7 decimal digits; the reference and the kernel evaluate the same
// expressions in the same order, so the residual is pure last-bit reassociation.
static const double TOL = sizeof(fv_real) == 4 ? 3e-5 : 1e-11;

static bool close_rel(double a, double b, double tol) {
  const double d = std::fabs(a - b);
  const double s = std::fmax(std::fabs(a), std::fabs(b));
  return d <= tol * (s > 1.0 ? s : 1.0);
}

// ---------------------------------------------------------------------------
// index box: one uniform storage rectangle for every field
// ---------------------------------------------------------------------------
// i in [is-ng, ie+ng+1], j in [js-ng, je+ng+1]. The +1 covers the staggered
// upper bounds (ut over isd..ied+1, vt over jsd..jed+1, dx/dy's ghost columns).
struct Box {
  int64_t is = 1, ie = 12, js = 1, je = 12, ng = 3, nk = 3;
  int64_t npx = 13, npy = 13;
  int32_t west = 1, east = 1, south = 1, north = 1;

  int64_t loi() const { return is - ng; }
  int64_t loj() const { return js - ng; }
  int64_t ni() const { return (ie + ng + 1) - loi() + 1; }
  int64_t nj() const { return (je + ng + 1) - loj() + 1; }
  int64_t n2() const { return ni() * nj(); }
  int64_t n3() const { return n2() * nk; }
};

static Box seam_box() { return Box{}; }

static Box interior_box() {
  Box b;
  b.is = 7; b.ie = 18; b.js = 7; b.je = 18;
  b.npx = 25; b.npy = 25;
  b.west = b.east = b.south = b.north = 0;
  return b;
}

static dgrid::TransportBounds tb_of(const Box &b) {
  dgrid::TransportBounds t{};
  t.is = b.is; t.ie = b.ie; t.js = b.js; t.je = b.je;
  t.npz = b.nk; t.nk = b.nk;
  t.halo_width = b.ng; t.ng = b.ng;
  t.npx = b.npx; t.npy = b.npy;
  t.cubed_sphere = 1; t.nested = 0;
  t.west_edge = b.west; t.east_edge = b.east;
  t.south_edge = b.south; t.north_edge = b.north;
  t.sw_corner = b.west && b.south;
  t.se_corner = b.east && b.south;
  t.ne_corner = b.east && b.north;
  t.nw_corner = b.west && b.north;
  return t;
}

// ---------------------------------------------------------------------------
// host storage
// ---------------------------------------------------------------------------
struct A3 {
  const Box *b = nullptr;
  std::vector<fv_real> d;
  void init(const Box &bx, fv_real fill = fv_real(0)) {
    b = &bx;
    d.assign(static_cast<size_t>(bx.n3()), fill);
  }
  size_t off(int64_t i, int64_t j, int64_t k) const {
    return static_cast<size_t>((i - b->loi()) + (j - b->loj()) * b->ni() +
                               k * b->ni() * b->nj());
  }
  fv_real &operator()(int64_t i, int64_t j, int64_t k) { return d[off(i, j, k)]; }
  fv_real operator()(int64_t i, int64_t j, int64_t k) const { return d[off(i, j, k)]; }
};

struct A2 {
  const Box *b = nullptr;
  std::vector<fv_real> d;
  void init(const Box &bx, fv_real fill = fv_real(0)) {
    b = &bx;
    d.assign(static_cast<size_t>(bx.n2()), fill);
  }
  size_t off(int64_t i, int64_t j) const {
    return static_cast<size_t>((i - b->loi()) + (j - b->loj()) * b->ni());
  }
  fv_real &operator()(int64_t i, int64_t j) { return d[off(i, j)]; }
  fv_real operator()(int64_t i, int64_t j) const { return d[off(i, j)]; }
};

// ---------------------------------------------------------------------------
// device harness
// ---------------------------------------------------------------------------
struct Dev {
  std::vector<void *> owned;
  ~Dev() {
    for (void *p : owned) cudaFree(p);
  }
  void *raw(size_t bytes) {
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    // Zero rather than leave indeterminate: keeps compute-sanitizer initcheck
    // silent about halo cells no kernel writes.
    cudaMemset(p, 0, bytes);
    owned.push_back(p);
    return p;
  }
  fv_real *up3(const A3 &a) {
    void *p = raw(a.d.size() * sizeof(fv_real));
    if (p) cudaMemcpy(p, a.d.data(), a.d.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return static_cast<fv_real *>(p);
  }
  fv_real *up2(const A2 &a) {
    void *p = raw(a.d.size() * sizeof(fv_real));
    if (p) cudaMemcpy(p, a.d.data(), a.d.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return static_cast<fv_real *>(p);
  }
  fv_real *scratch(const Box &b) {
    return static_cast<fv_real *>(raw(static_cast<size_t>(b.n3()) * sizeof(fv_real)));
  }
  int32_t *scratch_i(const Box &b) {
    return static_cast<int32_t *>(raw(static_cast<size_t>(b.n3()) * sizeof(int32_t)));
  }
};

template <class T> static View3<T> V3(void *p, const Box &b) {
  return View3<T>{static_cast<T *>(p), b.loi(), b.loj(), 0, 1, b.ni(), b.ni() * b.nj()};
}
template <class T> static View2<T> V2(void *p, const Box &b) {
  return View2<T>{static_cast<T *>(p), b.loi(), b.loj(), 1, b.ni()};
}

static A3 down3(const void *p, const Box &b) {
  A3 a;
  a.init(b);
  cudaMemcpy(a.d.data(), p, a.d.size() * sizeof(fv_real), cudaMemcpyDeviceToHost);
  return a;
}

// ---------------------------------------------------------------------------
// synthetic fields
// ---------------------------------------------------------------------------
// DX0 != DY0 deliberately: a swapped x/y metric changes the answer by O(20%)
// rather than by round-off, which is what makes Part G's swap detector bite.
static const double DX0 = 1.0e5;
static const double DY0 = 1.2e5;
static const double DT = 300.0;

// q_mode 0: smooth, tilted so no plateau makes the mord=6 limiter branch
//           round-off-decided. 1: plateau (Part H admissibility).
static fv_real fill_u(int64_t i, int64_t j, int64_t k, int q_mode) {
  const double x = static_cast<double>(i), y = static_cast<double>(j);
  if (q_mode == 1) return fv_real(12.0 + 0.5 * k);
  return fv_real(20.0 * std::sin(0.4 * x + 0.2 * y) + 3.0 * std::cos(0.13 * y) + 0.05 * x +
                 0.7 * k);
}

static fv_real fill_v(int64_t i, int64_t j, int64_t k, int q_mode) {
  const double x = static_cast<double>(i), y = static_cast<double>(j);
  if (q_mode == 1) return fv_real(-8.0 + 0.25 * k);
  return fv_real(17.0 * std::cos(0.31 * y + 0.17 * x) - 4.0 * std::sin(0.23 * x) + 0.04 * y +
                 0.6 * k);
}

static void fill_grid(const Box &b, A2 &dx, A2 &rdx, A2 &dy, A2 &rdy, A2 &cosa, A2 &rsina,
                      A2 &rarea, A2 &f0) {
  dx.init(b); rdx.init(b); dy.init(b); rdy.init(b);
  cosa.init(b); rsina.init(b); rarea.init(b); f0.init(b);
  for (int64_t j = b.loj(); j <= b.je + b.ng + 1; ++j) {
    for (int64_t i = b.loi(); i <= b.ie + b.ng + 1; ++i) {
      const double x = static_cast<double>(i), y = static_cast<double>(j);
      const double vdx = DX0 * (1.0 + 0.03 * std::sin(0.3 * x) + 0.02 * std::cos(0.2 * y));
      const double vdy = DY0 * (1.0 + 0.02 * std::cos(0.25 * x) + 0.03 * std::sin(0.17 * y));
      dx(i, j) = fv_real(vdx);
      rdx(i, j) = fv_real(1.0 / vdx);
      dy(i, j) = fv_real(vdy);
      rdy(i, j) = fv_real(1.0 / vdy);
      const double ca = 0.08 * std::sin(0.21 * x + 0.11 * y);
      cosa(i, j) = fv_real(ca);
      rsina(i, j) = fv_real(1.0 / std::sqrt(1.0 - ca * ca));
      rarea(i, j) = fv_real(1.0 / (vdx * vdy));
      f0(i, j) = fv_real(1.0e-4 * (1.0 + 0.1 * std::sin(0.19 * x - 0.13 * y)));
    }
  }
}

// Courant-like argument of xtp_u / ytp_v. The Fortran `c` is a length (dt times
// a contravariant wind), rescaled inside by rdx/rdy, so it must be scaled by the
// corresponding metric. Both signs occur so both flux branches are exercised.
static fv_real fill_c(int64_t i, int64_t j, int64_t k, double scale) {
  const double x = static_cast<double>(i), y = static_cast<double>(j);
  return fv_real(0.35 * scale * std::sin(0.27 * x + 0.41 * y + 0.3 * k));
}

// ---------------------------------------------------------------------------
// host transcription: xtp_u (sw_core.F90:1950-2317, iord = 6, grid_type = 0,
// not nested)
// ---------------------------------------------------------------------------
static void ref_xtp_u(const Box &b, const A3 &u, const A3 &c, const A2 &dx, const A2 &rdx,
                      A3 &flux) {
  const fv_real p1 = fv_real(7.0 / 12.0), p2 = fv_real(-1.0 / 12.0);
  const fv_real c1 = fv_real(-2.0 / 14.0), c2 = fv_real(11.0 / 14.0), c3 = fv_real(5.0 / 14.0);
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  const int64_t lo = is - 3, n = (ie + 3) - lo + 1;
  const fv_real POISON = std::numeric_limits<fv_real>::quiet_NaN();

  std::vector<fv_real> al(n), bl(n), br(n), b0(n);
  std::vector<int> smt(n);
  auto AL = [&](int64_t i) -> fv_real & { return al[i - lo]; };
  auto BL = [&](int64_t i) -> fv_real & { return bl[i - lo]; };
  auto BR = [&](int64_t i) -> fv_real & { return br[i - lo]; };
  auto B0 = [&](int64_t i) -> fv_real & { return b0[i - lo]; };
  auto SMT = [&](int64_t i) -> int & { return smt[i - lo]; };

  flux.init(b, POISON);

  const int64_t is3 = is - 1 > 3 ? is - 1 : 3;
  const int64_t ie3 = npx - 3 < ie + 1 ? npx - 3 : ie + 1;

  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = js; j <= je + 1; ++j) {
      for (int64_t t = 0; t < n; ++t) {
        al[t] = POISON; bl[t] = POISON; br[t] = POISON; b0[t] = POISON; smt[t] = -1;
      }
      // :1987-1989
      for (int64_t i = is3; i <= ie3 + 1; ++i)
        AL(i) = p1 * (u(i - 1, j, k) + u(i, j, k)) + p2 * (u(i - 2, j, k) + u(i + 1, j, k));
      // :1990-1993
      for (int64_t i = is3; i <= ie3; ++i) {
        BL(i) = AL(i) - u(i, j, k);
        BR(i) = AL(i + 1) - u(i, j, k);
      }
      if (is == 1) {  // :1996-2014
        fv_real xt = c3 * u(1, j, k) + c2 * u(2, j, k) + c1 * u(3, j, k);
        BR(1) = xt - u(1, j, k);
        BL(2) = xt - u(2, j, k);
        BR(2) = AL(3) - u(2, j, k);
        if (j == 1 || j == npy) {
          BL(0) = fv_real(0); BR(0) = fv_real(0);
          BL(1) = fv_real(0); BR(1) = fv_real(0);
        } else {
          BL(0) = c1 * u(-2, j, k) + c2 * u(-1, j, k) + c3 * u(0, j, k) - u(0, j, k);
          xt = fv_real(0.5) *
               (((fv_real(2) * dx(0, j) + dx(-1, j)) * u(0, j, k) - dx(0, j) * u(-1, j, k)) /
                    (dx(0, j) + dx(-1, j)) +
                ((fv_real(2) * dx(1, j) + dx(2, j)) * u(1, j, k) - dx(1, j) * u(2, j, k)) /
                    (dx(1, j) + dx(2, j)));
          BR(0) = xt - u(0, j, k);
          BL(1) = xt - u(1, j, k);
        }
      }
      if ((ie + 1) == npx) {  // :2015-2033
        BL(npx - 2) = AL(npx - 2) - u(npx - 2, j, k);
        fv_real xt = c1 * u(npx - 3, j, k) + c2 * u(npx - 2, j, k) + c3 * u(npx - 1, j, k);
        BR(npx - 2) = xt - u(npx - 2, j, k);
        BL(npx - 1) = xt - u(npx - 1, j, k);
        if (j == 1 || j == npy) {
          BL(npx - 1) = fv_real(0); BR(npx - 1) = fv_real(0);
          BL(npx) = fv_real(0); BR(npx) = fv_real(0);
        } else {
          xt = fv_real(0.5) * (((fv_real(2) * dx(npx - 1, j) + dx(npx - 2, j)) * u(npx - 1, j, k) -
                                dx(npx - 1, j) * u(npx - 2, j, k)) /
                                   (dx(npx - 1, j) + dx(npx - 2, j)) +
                               ((fv_real(2) * dx(npx, j) + dx(npx + 1, j)) * u(npx, j, k) -
                                dx(npx, j) * u(npx + 1, j, k)) /
                                   (dx(npx, j) + dx(npx + 1, j)));
          BR(npx - 1) = xt - u(npx - 1, j, k);
          BL(npx) = xt - u(npx, j, k);
          BR(npx) = c3 * u(npx, j, k) + c2 * u(npx + 1, j, k) + c1 * u(npx + 2, j, k) -
                    u(npx, j, k);
        }
      }
      for (int64_t i = is - 1; i <= ie + 1; ++i) B0(i) = BL(i) + BR(i);  // :2036-2038
      // :2139-2142 (iord == 6)
      for (int64_t i = is - 1; i <= ie + 1; ++i)
        SMT(i) = (fv_real(3) * std::fabs(B0(i)) < std::fabs(BL(i) - BR(i))) ? 1 : 0;
      // :2144-2156 WMP seam fix
      if (is == 1) {
        SMT(0) = (BL(0) * BR(0) < fv_real(0)) ? 1 : 0;
        SMT(1) = (BL(1) * BR(1) < fv_real(0)) ? 1 : 0;
      }
      if ((ie + 1) == npx) {
        SMT(npx - 1) = (BL(npx - 1) * BR(npx - 1) < fv_real(0)) ? 1 : 0;
        SMT(npx) = (BL(npx) * BR(npx) < fv_real(0)) ? 1 : 0;
      }
      // :2159-2171
      for (int64_t i = is; i <= ie + 1; ++i) {
        fv_real cfl, fx0, f;
        if (c(i, j, k) > fv_real(0)) {
          cfl = c(i, j, k) * rdx(i - 1, j);
          fx0 = (fv_real(1) - cfl) * (BR(i - 1) - cfl * B0(i - 1));
          f = u(i - 1, j, k);
        } else {
          cfl = c(i, j, k) * rdx(i, j);
          fx0 = (fv_real(1) + cfl) * (BL(i) + cfl * B0(i));
          f = u(i, j, k);
        }
        if (SMT(i - 1) || SMT(i)) f = f + fx0;
        flux(i, j, k) = f;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// host transcription: ytp_v (sw_core.F90:2320-2792, jord = 6)
// ---------------------------------------------------------------------------
// Transcribed from the Fortran independently of ref_xtp_u -- 2-D locals filled
// by i-loops inside the south/north blocks, the zeroing keyed on the CROSS
// index (is==1 / ie+1==npx), and br(i,npy) evaluated inside the main i loop so
// that the zeroing overwrites it. Not obtained by transposing ref_xtp_u.
static void ref_ytp_v(const Box &b, const A3 &v, const A3 &c, const A2 &dy, const A2 &rdy,
                      A3 &flux) {
  const fv_real p1 = fv_real(7.0 / 12.0), p2 = fv_real(-1.0 / 12.0);
  const fv_real c1 = fv_real(-2.0 / 14.0), c2 = fv_real(11.0 / 14.0), c3 = fv_real(5.0 / 14.0);
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  const fv_real POISON = std::numeric_limits<fv_real>::quiet_NaN();

  // al over js-1..je+2, bl/br/b0/smt over js-1..je+1, all over is..ie+1.
  const int64_t iof = is, jof = js - 3;
  const int64_t nix = (ie + 1) - iof + 1, njx = (je + 3) - jof + 1;
  std::vector<fv_real> al(nix * njx), bl(nix * njx), br(nix * njx), b0(nix * njx);
  std::vector<int> smt(nix * njx);
  auto IX = [&](int64_t i, int64_t j) { return (i - iof) + (j - jof) * nix; };
  auto AL = [&](int64_t i, int64_t j) -> fv_real & { return al[IX(i, j)]; };
  auto BL = [&](int64_t i, int64_t j) -> fv_real & { return bl[IX(i, j)]; };
  auto BR = [&](int64_t i, int64_t j) -> fv_real & { return br[IX(i, j)]; };
  auto B0 = [&](int64_t i, int64_t j) -> fv_real & { return b0[IX(i, j)]; };
  auto SMT = [&](int64_t i, int64_t j) -> int & { return smt[IX(i, j)]; };

  flux.init(b, POISON);

  const int64_t js3 = js - 1 > 3 ? js - 1 : 3;
  const int64_t je3 = npy - 3 < je + 1 ? npy - 3 : je + 1;

  for (int64_t k = 0; k < b.nk; ++k) {
    for (size_t t = 0; t < al.size(); ++t) {
      al[t] = POISON; bl[t] = POISON; br[t] = POISON; b0[t] = POISON; smt[t] = -1;
    }
    // :2354-2358
    for (int64_t j = js3; j <= je3 + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i)
        AL(i, j) = p1 * (v(i, j - 1, k) + v(i, j, k)) + p2 * (v(i, j - 2, k) + v(i, j + 1, k));
    // :2359-2364
    for (int64_t j = js3; j <= je3; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        BL(i, j) = AL(i, j) - v(i, j, k);
        BR(i, j) = AL(i, j + 1) - v(i, j, k);
      }
    if (js == 1) {  // :2367-2390
      for (int64_t i = is; i <= ie + 1; ++i) {
        BL(i, 0) = c1 * v(i, -2, k) + c2 * v(i, -1, k) + c3 * v(i, 0, k) - v(i, 0, k);
        fv_real xt =
            fv_real(0.5) *
            (((fv_real(2) * dy(i, 0) + dy(i, -1)) * v(i, 0, k) - dy(i, 0) * v(i, -1, k)) /
                 (dy(i, 0) + dy(i, -1)) +
             ((fv_real(2) * dy(i, 1) + dy(i, 2)) * v(i, 1, k) - dy(i, 1) * v(i, 2, k)) /
                 (dy(i, 1) + dy(i, 2)));
        BR(i, 0) = xt - v(i, 0, k);
        BL(i, 1) = xt - v(i, 1, k);
        xt = c3 * v(i, 1, k) + c2 * v(i, 2, k) + c1 * v(i, 3, k);
        BR(i, 1) = xt - v(i, 1, k);
        BL(i, 2) = xt - v(i, 2, k);
        BR(i, 2) = AL(i, 3) - v(i, 2, k);
      }
      if (is == 1) {
        BL(1, 0) = fv_real(0); BR(1, 0) = fv_real(0);
        BL(1, 1) = fv_real(0); BR(1, 1) = fv_real(0);
      }
      if ((ie + 1) == npx) {
        BL(npx, 0) = fv_real(0); BR(npx, 0) = fv_real(0);
        BL(npx, 1) = fv_real(0); BR(npx, 1) = fv_real(0);
      }
    }
    if ((je + 1) == npy) {  // :2393-2417
      for (int64_t i = is; i <= ie + 1; ++i) {
        BL(i, npy - 2) = AL(i, npy - 2) - v(i, npy - 2, k);
        fv_real xt = c1 * v(i, npy - 3, k) + c2 * v(i, npy - 2, k) + c3 * v(i, npy - 1, k);
        BR(i, npy - 2) = xt - v(i, npy - 2, k);
        BL(i, npy - 1) = xt - v(i, npy - 1, k);
        xt = fv_real(0.5) *
             (((fv_real(2) * dy(i, npy - 1) + dy(i, npy - 2)) * v(i, npy - 1, k) -
               dy(i, npy - 1) * v(i, npy - 2, k)) /
                  (dy(i, npy - 1) + dy(i, npy - 2)) +
              ((fv_real(2) * dy(i, npy) + dy(i, npy + 1)) * v(i, npy, k) -
               dy(i, npy) * v(i, npy + 1, k)) /
                  (dy(i, npy) + dy(i, npy + 1)));
        BR(i, npy - 1) = xt - v(i, npy - 1, k);
        BL(i, npy) = xt - v(i, npy, k);
        BR(i, npy) = c3 * v(i, npy, k) + c2 * v(i, npy + 1, k) + c1 * v(i, npy + 2, k) -
                     v(i, npy, k);
      }
      if (is == 1) {
        BL(1, npy - 1) = fv_real(0); BR(1, npy - 1) = fv_real(0);
        BL(1, npy) = fv_real(0); BR(1, npy) = fv_real(0);
      }
      if ((ie + 1) == npx) {
        BL(npx, npy - 1) = fv_real(0); BR(npx, npy - 1) = fv_real(0);
        BL(npx, npy) = fv_real(0); BR(npx, npy) = fv_real(0);
      }
    }
    for (int64_t j = js - 1; j <= je + 1; ++j)  // :2423-2427
      for (int64_t i = is; i <= ie + 1; ++i) B0(i, j) = BL(i, j) + BR(i, j);
    // :2560-2564 (jord == 6)
    for (int64_t j = js - 1; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i)
        SMT(i, j) = (fv_real(3) * std::fabs(B0(i, j)) < std::fabs(BL(i, j) - BR(i, j))) ? 1 : 0;
    // :2567-2580 WMP seam fix
    if (js == 1)
      for (int64_t i = is; i <= ie + 1; ++i) {
        SMT(i, 0) = (BL(i, 0) * BR(i, 0) < fv_real(0)) ? 1 : 0;
        SMT(i, 1) = (BL(i, 1) * BR(i, 1) < fv_real(0)) ? 1 : 0;
      }
    if ((je + 1) == npy)
      for (int64_t i = is; i <= ie + 1; ++i) {
        SMT(i, npy - 1) = (BL(i, npy - 1) * BR(i, npy - 1) < fv_real(0)) ? 1 : 0;
        SMT(i, npy) = (BL(i, npy) * BR(i, npy) < fv_real(0)) ? 1 : 0;
      }
    // :2582-2596
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        fv_real cfl, fx0, f;
        if (c(i, j, k) > fv_real(0)) {
          cfl = c(i, j, k) * rdy(i, j - 1);
          fx0 = (fv_real(1) - cfl) * (BR(i, j - 1) - cfl * B0(i, j - 1));
          f = v(i, j - 1, k);
        } else {
          cfl = c(i, j, k) * rdy(i, j);
          fx0 = (fv_real(1) + cfl) * (BL(i, j) + cfl * B0(i, j));
          f = v(i, j, k);
        }
        if (SMT(i, j - 1) || SMT(i, j)) f = f + fx0;
        flux(i, j, k) = f;
      }
  }
}

// ---------------------------------------------------------------------------
// host transcription: d_sw's vb construction (sw_core.F90:1077-1113)
// ---------------------------------------------------------------------------
// Written as the Fortran writes it -- the js==1 row first over the FULL i range,
// then the interior j loop clipped to js2..je1 with the is==1 / (ie+1)==npx
// column overrides INSIDE it, then the (je+1)==npy row. A tile-edge ROW
// therefore beats a tile-edge COLUMN here. Poison exposes any cell left unset.
static void ref_vb(const Box &b, fv_real dt, const A3 &uc, const A3 &vc, const A3 &vt,
                   const A2 &cosa, const A2 &rsina, A3 &vb) {
  const fv_real dt5 = fv_real(0.5) * dt, dt4 = fv_real(0.25) * dt;
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  const int64_t is2 = is > 2 ? is : 2, ie1 = (ie + 1) < (npx - 1) ? (ie + 1) : (npx - 1);
  const int64_t js2 = js > 2 ? js : 2, je1 = (je + 1) < (npy - 1) ? (je + 1) : (npy - 1);
  vb.init(b, std::numeric_limits<fv_real>::quiet_NaN());

  for (int64_t k = 0; k < b.nk; ++k) {
    if (js == 1)
      for (int64_t i = is; i <= ie + 1; ++i)
        vb(i, 1, k) = dt5 * (vt(i - 1, 1, k) + vt(i, 1, k));
    for (int64_t j = js2; j <= je1; ++j) {
      for (int64_t i = is2; i <= ie1; ++i)
        vb(i, j, k) = dt5 *
                      (vc(i - 1, j, k) + vc(i, j, k) -
                       (uc(i, j - 1, k) + uc(i, j, k)) * cosa(i, j)) *
                      rsina(i, j);
      if (is == 1)
        vb(1, j, k) =
            dt4 * (-vt(-1, j, k) + fv_real(3) * (vt(0, j, k) + vt(1, j, k)) - vt(2, j, k));
      if ((ie + 1) == npx)
        vb(npx, j, k) = dt4 * (-vt(npx - 2, j, k) +
                               fv_real(3) * (vt(npx - 1, j, k) + vt(npx, j, k)) -
                               vt(npx + 1, j, k));
    }
    if ((je + 1) == npy)
      for (int64_t i = is; i <= ie + 1; ++i)
        vb(i, npy, k) = dt5 * (vt(i - 1, npy, k) + vt(i, npy, k));
  }
}

// ---------------------------------------------------------------------------
// host transcription: d_sw's ub construction (sw_core.F90:1143-1169)
// ---------------------------------------------------------------------------
// The is==1 COLUMN is written first over the FULL j range, then the j loop runs
// over js..je+1 (unclipped) with the i loop clipped to is2..ie1, then the
// (ie+1)==npx COLUMN. So a tile-edge COLUMN beats a tile-edge ROW -- the
// opposite precedence to ref_vb -- and the 4-point one-sided formula sits on the
// j==1 / j==npy rows rather than the columns.
static void ref_ub(const Box &b, fv_real dt, const A3 &uc, const A3 &vc, const A3 &ut,
                   const A2 &cosa, const A2 &rsina, A3 &ub) {
  const fv_real dt5 = fv_real(0.5) * dt, dt4 = fv_real(0.25) * dt;
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  const int64_t is2 = is > 2 ? is : 2, ie1 = (ie + 1) < (npx - 1) ? (ie + 1) : (npx - 1);
  ub.init(b, std::numeric_limits<fv_real>::quiet_NaN());

  for (int64_t k = 0; k < b.nk; ++k) {
    if (is == 1)
      for (int64_t j = js; j <= je + 1; ++j)
        ub(1, j, k) = dt5 * (ut(1, j - 1, k) + ut(1, j, k));
    for (int64_t j = js; j <= je + 1; ++j) {
      if (j == 1 || j == npy) {
        for (int64_t i = is2; i <= ie1; ++i)
          ub(i, j, k) = dt4 * (-ut(i, j - 2, k) +
                               fv_real(3) * (ut(i, j - 1, k) + ut(i, j, k)) - ut(i, j + 1, k));
      } else {
        for (int64_t i = is2; i <= ie1; ++i)
          ub(i, j, k) = dt5 *
                        (uc(i, j - 1, k) + uc(i, j, k) -
                         (vc(i - 1, j, k) + vc(i, j, k)) * cosa(i, j)) *
                        rsina(i, j);
      }
    }
    if ((ie + 1) == npx)
      for (int64_t j = js; j <= je + 1; ++j)
        ub(npx, j, k) = dt5 * (ut(npx, j - 1, k) + ut(npx, j, k));
  }
}

// --- rows 3 / 6 / 7: kinetic energy (sw_core.F90:1123-1216) ---------------
static void ref_ke_product(const Box &b, const A3 &vb, const A3 &ub, A3 &ke) {
  ke.init(b, std::numeric_limits<fv_real>::quiet_NaN());
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) ke(i, j, k) = vb(i, j, k) * ub(i, j, k);
}

static void ref_ke_average(const Box &b, const A3 &ub, const A3 &vb, A3 &ke) {
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        ke(i, j, k) = fv_real(0.5) * (ke(i, j, k) + ub(i, j, k) * vb(i, j, k));
}

static void ref_ke_corner_fix(const Box &b, fv_real dt, const A3 &ut, const A3 &vt,
                              const A3 &u, const A3 &v, A3 &ke) {
  const fv_real dt6 = dt / fv_real(6);
  const int64_t npx = b.npx, npy = b.npy;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (b.west && b.south)
      ke(1, 1, k) = dt6 * ((ut(1, 1, k) + ut(1, 0, k)) * u(1, 1, k) +
                           (vt(1, 1, k) + vt(0, 1, k)) * v(1, 1, k) +
                           (ut(1, 1, k) + vt(1, 1, k)) * u(0, 1, k));
    if (b.east && b.south) {
      const int64_t i = npx;
      ke(i, 1, k) = dt6 * ((ut(i, 1, k) + ut(i, 0, k)) * u(i - 1, 1, k) +
                           (vt(i, 1, k) + vt(i - 1, 1, k)) * v(i, 1, k) +
                           (ut(i, 1, k) - vt(i - 1, 1, k)) * u(i, 1, k));
    }
    if (b.east && b.north) {
      const int64_t i = npx, j = npy;
      ke(i, j, k) = dt6 * ((ut(i, j, k) + ut(i, j - 1, k)) * u(i - 1, j, k) +
                           (vt(i, j, k) + vt(i - 1, j, k)) * v(i, j - 1, k) +
                           (ut(i, j - 1, k) + vt(i - 1, j, k)) * u(i, j, k));
    }
    if (b.west && b.north) {
      const int64_t j = npy;
      ke(1, j, k) = dt6 * ((ut(1, j, k) + ut(1, j - 1, k)) * u(1, j, k) +
                           (vt(1, j, k) + vt(0, j, k)) * v(1, j - 1, k) +
                           (ut(1, j - 1, k) - vt(1, j, k)) * u(0, j, k));
    }
  }
}

// --- row 8: vorticity precursor (sw_core.F90:1219-1235) -------------------
static void ref_vorticity(const Box &b, const A3 &u, const A3 &v, const A2 &dx, const A2 &dy,
                          const A2 &rarea, A3 &ut, A3 &vt, A3 &wk) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  const fv_real POISON = std::numeric_limits<fv_real>::quiet_NaN();
  ut.init(b, POISON); vt.init(b, POISON); wk.init(b, POISON);
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = jsd; j <= jed + 1; ++j)
      for (int64_t i = isd; i <= ied; ++i) vt(i, j, k) = u(i, j, k) * dx(i, j);
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied + 1; ++i) ut(i, j, k) = v(i, j, k) * dy(i, j);
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        wk(i, j, k) =
            rarea(i, j) * (vt(i, j, k) - vt(i, j + 1, k) - ut(i, j, k) + ut(i + 1, j, k));
  }
}

// --- row 9: absolute vorticity (sw_core.F90:1439-1459) --------------------
static void ref_vort_absolute(const Box &b, bool do_f3d, const A3 &wk, const A2 &f0,
                              const A3 &z_rat, A3 &vort) {
  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  vort.init(b, std::numeric_limits<fv_real>::quiet_NaN());
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        vort(i, j, k) = do_f3d ? wk(i, j, k) + f0(i, j) * z_rat(i, j, k) : wk(i, j, k) + f0(i, j);
}

// --- row 10: D-wind update (sw_core.F90:1463-1473) ------------------------
static void ref_dwind(const Box &b, const A3 &ut, const A3 &vt, const A3 &ke, const A3 &fx,
                      const A3 &fy, A3 &u, A3 &v) {
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i)
        u(i, j, k) = vt(i, j, k) + ke(i, j, k) - ke(i + 1, j, k) + fy(i, j, k);
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        v(i, j, k) = ut(i, j, k) + ke(i, j, k) - ke(i, j + 1, k) - fx(i, j, k);
  }
}

// ---------------------------------------------------------------------------
// comparison helper
// ---------------------------------------------------------------------------
static bool cmp_box(const char *tag, const Box &b, const A3 &got, const A3 &ref, int64_t i0,
                    int64_t i1, int64_t j0, int64_t j1, double tol, double *worst_out) {
  double worst = 0.0;
  int64_t bad = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = j0; j <= j1; ++j)
      for (int64_t i = i0; i <= i1; ++i) {
        const double g = static_cast<double>(got(i, j, k));
        const double r = static_cast<double>(ref(i, j, k));
        if (!std::isfinite(g) || !std::isfinite(r)) {
          if (bad < 6)
            std::printf("  %s NONFINITE at (%lld,%lld,%lld) got=%g ref=%g\n", tag,
                        static_cast<long long>(i), static_cast<long long>(j),
                        static_cast<long long>(k), g, r);
          ++bad;
          continue;
        }
        const double s = std::fmax(std::fabs(g), std::fabs(r));
        const double rel = std::fabs(g - r) / (s > 1.0 ? s : 1.0);
        if (rel > worst) worst = rel;
        if (!close_rel(g, r, tol)) {
          if (bad < 6)
            std::printf("  %s MISMATCH at (%lld,%lld,%lld) got=%.17g ref=%.17g rel=%.3e\n", tag,
                        static_cast<long long>(i), static_cast<long long>(j),
                        static_cast<long long>(k), g, r, rel);
          ++bad;
        }
      }
  if (worst_out) *worst_out = worst;
  if (bad) {
    std::printf("  %s: %lld cells outside tol %.1e (worst rel %.3e)\n", tag,
                static_cast<long long>(bad), tol, worst);
    return false;
  }
  std::printf("  %s worst rel %.3e\n", tag, worst);
  return true;
}

// ---------------------------------------------------------------------------
// single-launcher GPU drivers
// ---------------------------------------------------------------------------
static fv3_gpu_status gpu_xtp_u(const Box &box, const A3 &u, const A3 &c, const A2 &dx,
                                const A2 &rdx, int32_t hord, A3 *out) {
  Dev dev;
  dgrid::TransportBounds b = tb_of(box);
  dgrid::MomentumOptions opt;
  opt.hord_mt = hord;

  fv_real *d_u = dev.up3(u), *d_c = dev.up3(c);
  fv_real *d_dx = dev.up2(dx), *d_rdx = dev.up2(rdx);
  fv_real *d_flux = dev.scratch(box);
  dgrid::PpmMomentumWorkspace ws;
  ws.bl = V3<fv_real>(dev.scratch(box), box);
  ws.br = V3<fv_real>(dev.scratch(box), box);
  ws.b0 = V3<fv_real>(dev.scratch(box), box);
  ws.smt = V3<int32_t>(dev.scratch_i(box), box);
  if (!d_u || !d_c || !d_dx || !d_rdx || !d_flux || !ws.bl.data) return FV3_GPU_INTERNAL_ERROR;

  const fv3_gpu_status st = dgrid::launch_xtp_u_cuda(
      b, opt, V3<const fv_real>(d_c, box), V3<const fv_real>(d_u, box),
      V2<const fv_real>(d_dx, box), V2<const fv_real>(d_rdx, box), ws,
      V3<fv_real>(d_flux, box), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  if (out) *out = down3(d_flux, box);
  return FV3_GPU_SUCCESS;
}

static fv3_gpu_status gpu_ytp_v(const Box &box, const A3 &v, const A3 &c, const A2 &dy,
                                const A2 &rdy, int32_t hord, A3 *out) {
  Dev dev;
  dgrid::TransportBounds b = tb_of(box);
  dgrid::MomentumOptions opt;
  opt.hord_mt = hord;

  fv_real *d_v = dev.up3(v), *d_c = dev.up3(c);
  fv_real *d_dy = dev.up2(dy), *d_rdy = dev.up2(rdy);
  fv_real *d_flux = dev.scratch(box);
  dgrid::PpmMomentumWorkspace ws;
  ws.bl = V3<fv_real>(dev.scratch(box), box);
  ws.br = V3<fv_real>(dev.scratch(box), box);
  ws.b0 = V3<fv_real>(dev.scratch(box), box);
  ws.smt = V3<int32_t>(dev.scratch_i(box), box);
  if (!d_v || !d_c || !d_dy || !d_rdy || !d_flux || !ws.bl.data) return FV3_GPU_INTERNAL_ERROR;

  const fv3_gpu_status st = dgrid::launch_ytp_v_cuda(
      b, opt, V3<const fv_real>(d_c, box), V3<const fv_real>(d_v, box),
      V2<const fv_real>(d_dy, box), V2<const fv_real>(d_rdy, box), ws,
      V3<fv_real>(d_flux, box), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  if (out) *out = down3(d_flux, box);
  return FV3_GPU_SUCCESS;
}

// Unclipped interior-formula flux, valid at every (i,j) because the synthetic
// halo is fully populated. Used as the "edge treatment absent" counterfactual:
// where the authoritative seam formulas are live, the GPU must NOT agree with
// this.
static fv_real interior_flux_x(const A3 &u, const A3 &c, const A2 &rdx, int64_t i, int64_t j,
                               int64_t k) {
  const fv_real p1 = fv_real(7.0 / 12.0), p2 = fv_real(-1.0 / 12.0);
  auto al = [&](int64_t ii) {
    return p1 * (u(ii - 1, j, k) + u(ii, j, k)) + p2 * (u(ii - 2, j, k) + u(ii + 1, j, k));
  };
  fv_real bl[2], br[2], b0[2];
  int smt[2];
  for (int t = 0; t < 2; ++t) {
    const int64_t ii = i - 1 + t;
    bl[t] = al(ii) - u(ii, j, k);
    br[t] = al(ii + 1) - u(ii, j, k);
    b0[t] = bl[t] + br[t];
    smt[t] = (fv_real(3) * std::fabs(b0[t]) < std::fabs(bl[t] - br[t])) ? 1 : 0;
  }
  fv_real fx0, f;
  if (c(i, j, k) > fv_real(0)) {
    const fv_real cfl = c(i, j, k) * rdx(i - 1, j);
    fx0 = (fv_real(1) - cfl) * (br[0] - cfl * b0[0]);
    f = u(i - 1, j, k);
  } else {
    const fv_real cfl = c(i, j, k) * rdx(i, j);
    fx0 = (fv_real(1) + cfl) * (bl[1] + cfl * b0[1]);
    f = u(i, j, k);
  }
  if (smt[0] || smt[1]) f = f + fx0;
  return f;
}

static fv_real interior_flux_y(const A3 &v, const A3 &c, const A2 &rdy, int64_t i, int64_t j,
                               int64_t k) {
  const fv_real p1 = fv_real(7.0 / 12.0), p2 = fv_real(-1.0 / 12.0);
  auto al = [&](int64_t jj) {
    return p1 * (v(i, jj - 1, k) + v(i, jj, k)) + p2 * (v(i, jj - 2, k) + v(i, jj + 1, k));
  };
  fv_real bl[2], br[2], b0[2];
  int smt[2];
  for (int t = 0; t < 2; ++t) {
    const int64_t jj = j - 1 + t;
    bl[t] = al(jj) - v(i, jj, k);
    br[t] = al(jj + 1) - v(i, jj, k);
    b0[t] = bl[t] + br[t];
    smt[t] = (fv_real(3) * std::fabs(b0[t]) < std::fabs(bl[t] - br[t])) ? 1 : 0;
  }
  fv_real fx0, f;
  if (c(i, j, k) > fv_real(0)) {
    const fv_real cfl = c(i, j, k) * rdy(i, j - 1);
    fx0 = (fv_real(1) - cfl) * (br[0] - cfl * b0[0]);
    f = v(i, j - 1, k);
  } else {
    const fv_real cfl = c(i, j, k) * rdy(i, j);
    fx0 = (fv_real(1) + cfl) * (bl[1] + cfl * b0[1]);
    f = v(i, j, k);
  }
  if (smt[0] || smt[1]) f = f + fx0;
  return f;
}

// ---------------------------------------------------------------------------
// field setup shared by the transport parts
// ---------------------------------------------------------------------------
struct Fields {
  Box box;
  A3 u, v, cx, cy, uc, vc, ut, vt, z_rat;
  A2 dx, rdx, dy, rdy, cosa, rsina, rarea, f0;

  void build(const Box &bx, int q_mode) {
    box = bx;
    fill_grid(box, dx, rdx, dy, rdy, cosa, rsina, rarea, f0);
    u.init(box); v.init(box); cx.init(box); cy.init(box);
    uc.init(box); vc.init(box); ut.init(box); vt.init(box); z_rat.init(box);
    for (int64_t k = 0; k < box.nk; ++k)
      for (int64_t j = box.loj(); j <= box.je + box.ng + 1; ++j)
        for (int64_t i = box.loi(); i <= box.ie + box.ng + 1; ++i) {
          const double x = static_cast<double>(i), y = static_cast<double>(j);
          u(i, j, k) = fill_u(i, j, k, q_mode);
          v(i, j, k) = fill_v(i, j, k, q_mode);
          cx(i, j, k) = fill_c(i, j, k, DX0);
          cy(i, j, k) = fill_c(i, j, k, DY0);
          uc(i, j, k) = fv_real(11.0 * std::sin(0.29 * x - 0.19 * y + 0.4 * k) + 2.0);
          vc(i, j, k) = fv_real(-9.0 * std::cos(0.22 * x + 0.33 * y - 0.2 * k) + 1.5);
          ut(i, j, k) = fv_real(7.0 * std::cos(0.18 * x + 0.26 * y + 0.15 * k) - 1.0);
          vt(i, j, k) = fv_real(6.0 * std::sin(0.24 * x - 0.21 * y + 0.11 * k) + 0.8);
          z_rat(i, j, k) = fv_real(1.0 + 0.05 * std::sin(0.3 * x + 0.2 * y));
        }
  }
};

// ---------------------------------------------------------------------------
// Part A -- xtp_u, interior fixture
// ---------------------------------------------------------------------------
static bool part_a() {
  Fields f;
  f.build(interior_box(), 0);
  A3 got, ref;
  const fv3_gpu_status st = gpu_xtp_u(f.box, f.u, f.cx, f.dx, f.rdx, 6, &got);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  A launch_xtp_u_cuda status %d\n", static_cast<int>(st));
    return false;
  }
  ref_xtp_u(f.box, f.u, f.cx, f.dx, f.rdx, ref);
  double worst = 0;
  if (!cmp_box("A xtp_u interior", f.box, got, ref, f.box.is, f.box.ie + 1, f.box.js,
               f.box.je + 1, TOL, &worst))
    return false;
  // The limiter must actually fire somewhere, else this part would pass on a
  // first-order upwind implementation too.
  int64_t lim = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t j = f.box.js; j <= f.box.je + 1; ++j)
      for (int64_t i = f.box.is; i <= f.box.ie + 1; ++i) {
        const fv_real up = f.cx(i, j, k) > fv_real(0) ? f.u(i - 1, j, k) : f.u(i, j, k);
        if (std::fabs(static_cast<double>(got(i, j, k) - up)) > 1e-9) ++lim;
      }
  std::printf("  A PPM correction is live (%lld of %lld cells differ from pure upwind)\n",
              static_cast<long long>(lim),
              static_cast<long long>(f.box.nk * 13 * 13));
  if (lim < 100) {
    std::printf("  A PPM correction never fires -- fixture is degenerate\n");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Part B -- ytp_v, interior fixture
// ---------------------------------------------------------------------------
static bool part_b() {
  Fields f;
  f.build(interior_box(), 0);
  A3 got, ref;
  const fv3_gpu_status st = gpu_ytp_v(f.box, f.v, f.cy, f.dy, f.rdy, 6, &got);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  B launch_ytp_v_cuda status %d\n", static_cast<int>(st));
    return false;
  }
  ref_ytp_v(f.box, f.v, f.cy, f.dy, f.rdy, ref);
  double worst = 0;
  if (!cmp_box("B ytp_v interior", f.box, got, ref, f.box.is, f.box.ie + 1, f.box.js,
               f.box.je + 1, TOL, &worst))
    return false;
  // xtp_u and ytp_v must not be the same field: transporting u in x and v in y
  // over the same box has to give different answers, otherwise a copy-paste of
  // one into the other would go unnoticed.
  A3 xg;
  if (gpu_xtp_u(f.box, f.u, f.cx, f.dx, f.rdx, 6, &xg) != FV3_GPU_SUCCESS) return false;
  int64_t diff = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t j = f.box.js; j <= f.box.je + 1; ++j)
      for (int64_t i = f.box.is; i <= f.box.ie + 1; ++i)
        if (!close_rel(static_cast<double>(got(i, j, k)), static_cast<double>(xg(i, j, k)), 1e-6))
          ++diff;
  std::printf("  B xtp_u and ytp_v are distinct (%lld cells differ)\n",
              static_cast<long long>(diff));
  return diff > 300;
}

// ---------------------------------------------------------------------------
// Part C -- xtp_u, panel-edge fixture
// ---------------------------------------------------------------------------
static bool part_c() {
  Fields f;
  f.build(seam_box(), 0);
  A3 got, ref;
  const fv3_gpu_status st = gpu_xtp_u(f.box, f.u, f.cx, f.dx, f.rdx, 6, &got);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  C launch_xtp_u_cuda status %d\n", static_cast<int>(st));
    return false;
  }
  ref_xtp_u(f.box, f.u, f.cx, f.dx, f.rdx, ref);
  double worst = 0;
  if (!cmp_box("C xtp_u seam", f.box, got, ref, f.box.is, f.box.ie + 1, f.box.js, f.box.je + 1,
               TOL, &worst))
    return false;

  // Edge liveness: at the west/east seam columns the authoritative one-sided
  // reconstruction (and the WMP smt5 degradation) must make the answer differ
  // from the unclipped interior formula.
  const int64_t npx = f.box.npx;
  const int64_t cols[4] = {1, 2, npx - 1, npx};
  int64_t live = 0, checked = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t j = f.box.js + 2; j <= f.box.je - 1; ++j)
      for (int t = 0; t < 4; ++t) {
        const fv_real iv = interior_flux_x(f.u, f.cx, f.rdx, cols[t], j, k);
        ++checked;
        if (!close_rel(static_cast<double>(got(cols[t], j, k)), static_cast<double>(iv), 1e-9))
          ++live;
      }
  std::printf("  C west/east seam treatment is live (%lld of %lld seam cells differ from the "
              "interior formula)\n",
              static_cast<long long>(live), static_cast<long long>(checked));
  if (live < checked / 2) {
    std::printf("  C seam treatment not live enough\n");
    return false;
  }

  // The j==1 / j==npy rows zero bl/br at the four seam columns, so the flux
  // there must be exactly the unlimited upwind value (fx0 is multiplied by a
  // zero smt5 pair).
  int64_t zeroed = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t jj = 0; jj < 2; ++jj) {
      const int64_t j = jj == 0 ? 1 : f.box.npy;
      for (int t = 0; t < 4; ++t) {
        const int64_t i = cols[t];
        if (i != 1 && i != npx) continue;  // i==2 / npx-1 keep a nonzero side
        const fv_real up = f.cx(i, j, k) > fv_real(0) ? f.u(i - 1, j, k) : f.u(i, j, k);
        if (close_rel(static_cast<double>(got(i, j, k)), static_cast<double>(up), TOL)) ++zeroed;
      }
    }
  std::printf("  C zeroed-row bl/br honoured (%lld exact-upwind cells)\n",
              static_cast<long long>(zeroed));
  return zeroed >= 2 * f.box.nk;
}

// ---------------------------------------------------------------------------
// Part D -- ytp_v, panel-edge fixture
// ---------------------------------------------------------------------------
static bool part_d() {
  Fields f;
  f.build(seam_box(), 0);
  A3 got, ref;
  const fv3_gpu_status st = gpu_ytp_v(f.box, f.v, f.cy, f.dy, f.rdy, 6, &got);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  D launch_ytp_v_cuda status %d\n", static_cast<int>(st));
    return false;
  }
  ref_ytp_v(f.box, f.v, f.cy, f.dy, f.rdy, ref);
  double worst = 0;
  if (!cmp_box("D ytp_v seam", f.box, got, ref, f.box.is, f.box.ie + 1, f.box.js, f.box.je + 1,
               TOL, &worst))
    return false;

  const int64_t npy = f.box.npy;
  const int64_t rows[4] = {1, 2, npy - 1, npy};
  int64_t live = 0, checked = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t i = f.box.is + 2; i <= f.box.ie - 1; ++i)
      for (int t = 0; t < 4; ++t) {
        const fv_real iv = interior_flux_y(f.v, f.cy, f.rdy, i, rows[t], k);
        ++checked;
        if (!close_rel(static_cast<double>(got(i, rows[t], k)), static_cast<double>(iv), 1e-9))
          ++live;
      }
  std::printf("  D south/north seam treatment is live (%lld of %lld seam cells differ from the "
              "interior formula)\n",
              static_cast<long long>(live), static_cast<long long>(checked));
  if (live < checked / 2) {
    std::printf("  D seam treatment not live enough\n");
    return false;
  }

  // ytp_v zeroes bl/br on the i==1 / i==npx COLUMNS (the cross index), which is
  // the transposed counterpart of xtp_u's rows. Confirm the column, not the row.
  int64_t zeroed = 0;
  for (int64_t k = 0; k < f.box.nk; ++k)
    for (int64_t ii = 0; ii < 2; ++ii) {
      const int64_t i = ii == 0 ? 1 : f.box.npx;
      for (int t = 0; t < 4; ++t) {
        const int64_t j = rows[t];
        if (j != 1 && j != npy) continue;
        const fv_real up = f.cy(i, j, k) > fv_real(0) ? f.v(i, j - 1, k) : f.v(i, j, k);
        if (close_rel(static_cast<double>(got(i, j, k)), static_cast<double>(up), TOL)) ++zeroed;
      }
    }
  std::printf("  D zeroed-column bl/br honoured (%lld exact-upwind cells)\n",
              static_cast<long long>(zeroed));
  return zeroed >= 2 * f.box.nk;
}

// ---------------------------------------------------------------------------
// the ordered momentum sequence (rows 1-10)
// ---------------------------------------------------------------------------
// Snapshots are taken between launches so that the vb/ub buffer REUSE chain is
// exercised exactly as d_sw does it (row 1 writes vb, row 2 reads it as `c` and
// writes ub, row 3 consumes both, row 4 overwrites ub, row 5 overwrites vb),
// while still letting each row be compared individually.
struct SeqOut {
  A3 vb1, ub1, ke1, ub2, vb2, ke2, ke3, ut8, vt8, wk8, vort, vort_f3d, fx, fy, un, vn;
};

static fv3_gpu_status gpu_sequence(const Fields &f, SeqOut *o) {
  const Box &box = f.box;
  Dev dev;
  dgrid::TransportBounds b = tb_of(box);
  dgrid::MomentumOptions opt;
  opt.hord_mt = -6;  // production hord_mt magnitude; sign is immaterial (Part K)

  fv_real *d_u = dev.up3(f.u), *d_v = dev.up3(f.v);
  fv_real *d_uc = dev.up3(f.uc), *d_vc = dev.up3(f.vc);
  fv_real *d_ut = dev.up3(f.ut), *d_vt = dev.up3(f.vt);
  fv_real *d_zrat = dev.up3(f.z_rat);
  fv_real *d_dx = dev.up2(f.dx), *d_rdx = dev.up2(f.rdx);
  fv_real *d_dy = dev.up2(f.dy), *d_rdy = dev.up2(f.rdy);
  fv_real *d_cosa = dev.up2(f.cosa), *d_rsina = dev.up2(f.rsina);
  fv_real *d_rarea = dev.up2(f.rarea), *d_f0 = dev.up2(f.f0);
  fv_real *d_vb = dev.scratch(box), *d_ub = dev.scratch(box), *d_ke = dev.scratch(box);
  fv_real *d_wk = dev.scratch(box), *d_vort = dev.scratch(box);
  fv_real *d_fx = dev.scratch(box), *d_fy = dev.scratch(box);
  dgrid::PpmMomentumWorkspace ws;
  ws.bl = V3<fv_real>(dev.scratch(box), box);
  ws.br = V3<fv_real>(dev.scratch(box), box);
  ws.b0 = V3<fv_real>(dev.scratch(box), box);
  ws.smt = V3<int32_t>(dev.scratch_i(box), box);
  if (!d_f0 || !d_fy || !ws.smt.data) return FV3_GPU_INTERNAL_ERROR;

  const auto C3 = [&](fv_real *p) { return V3<const fv_real>(p, box); };
  const auto C2 = [&](fv_real *p) { return V2<const fv_real>(p, box); };
  const auto W3 = [&](fv_real *p) { return V3<fv_real>(p, box); };
  const fv_real dt = fv_real(DT);
  fv3_gpu_status st;

  // row 1
  st = dgrid::launch_bgrid_vb_cuda(b, opt, dt, C3(d_uc), C3(d_vc), C3(d_ut), C3(d_vt),
                                   C2(d_cosa), C2(d_rsina), W3(d_vb), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->vb1 = down3(d_vb, box);

  // row 2: ytp_v(c = vb) -> ub
  st = dgrid::launch_ytp_v_cuda(b, opt, C3(d_vb), C3(d_v), C2(d_dy), C2(d_rdy), ws, W3(d_ub),
                                nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ub1 = down3(d_ub, box);

  // row 3
  st = dgrid::launch_ke_product_cuda(b, C3(d_vb), C3(d_ub), W3(d_ke), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ke1 = down3(d_ke, box);

  // row 4 (overwrites ub)
  st = dgrid::launch_bgrid_ub_cuda(b, opt, dt, C3(d_uc), C3(d_vc), C3(d_ut), C3(d_vt),
                                   C2(d_cosa), C2(d_rsina), W3(d_ub), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ub2 = down3(d_ub, box);

  // row 5: xtp_u(c = ub) -> vb (overwrites vb)
  st = dgrid::launch_xtp_u_cuda(b, opt, C3(d_ub), C3(d_u), C2(d_dx), C2(d_rdx), ws, W3(d_vb),
                                nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->vb2 = down3(d_vb, box);

  // row 6
  st = dgrid::launch_ke_average_cuda(b, C3(d_ub), C3(d_vb), W3(d_ke), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ke2 = down3(d_ke, box);

  // row 7
  st = dgrid::launch_ke_corner_fix_cuda(b, dt, C3(d_ut), C3(d_vt), C3(d_u), C3(d_v), W3(d_ke),
                                        nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ke3 = down3(d_ke, box);

  // row 8 (destroys the contravariant ut/vt)
  st = dgrid::launch_vorticity_precursor_cuda(b, C3(d_u), C3(d_v), C2(d_dx), C2(d_dy),
                                              C2(d_rarea), W3(d_ut), W3(d_vt), W3(d_wk),
                                              nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->ut8 = down3(d_ut, box);
  o->vt8 = down3(d_vt, box);
  o->wk8 = down3(d_wk, box);

  // row 9, both branches
  st = dgrid::launch_vorticity_absolute_cuda(b, opt, C3(d_wk), C2(d_f0), C3(d_zrat), W3(d_vort),
                                             nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->vort = down3(d_vort, box);

  dgrid::MomentumOptions opt3 = opt;
  opt3.do_f3d = 1;
  st = dgrid::launch_vorticity_absolute_cuda(b, opt3, C3(d_wk), C2(d_f0), C3(d_zrat),
                                             W3(d_vort), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->vort_f3d = down3(d_vort, box);

  // row 9b: REUSED Milestone 3A.5 service. fv_tp_2d(vort, crx_adv, cry_adv,
  // hord_vt, fx, fy, xfx_adv, yfx_adv, ..., ra_x, ra_y) -- no mfx/mfy, no
  // nord/damp_c, so mass_flux_weighting = 0 and deln_flux_enabled = 0.
  A3 crx, cry, xfx, yfx;
  A2 area, dxa, dya, del6_u, del6_v;
  crx.init(box); cry.init(box); xfx.init(box); yfx.init(box);
  area.init(box); dxa.init(box); dya.init(box); del6_u.init(box); del6_v.init(box);
  for (int64_t j = box.loj(); j <= box.je + box.ng + 1; ++j)
    for (int64_t i = box.loi(); i <= box.ie + box.ng + 1; ++i) {
      const double x = static_cast<double>(i), y = static_cast<double>(j);
      area(i, j) = f.dx(i, j) * f.dy(i, j);
      dxa(i, j) = f.dx(i, j);
      dya(i, j) = f.dy(i, j);
      del6_u(i, j) = fv_real(1.0);
      del6_v(i, j) = fv_real(1.0);
      for (int64_t k = 0; k < box.nk; ++k) {
        crx(i, j, k) = fv_real(0.3 * std::sin(0.21 * x + 0.33 * y + 0.2 * k));
        cry(i, j, k) = fv_real(0.3 * std::cos(0.19 * x - 0.27 * y + 0.1 * k));
        xfx(i, j, k) = fv_real(static_cast<double>(crx(i, j, k)) * static_cast<double>(area(i, j)) /
                               static_cast<double>(f.dx(i, j)));
        yfx(i, j, k) = fv_real(static_cast<double>(cry(i, j, k)) * static_cast<double>(area(i, j)) /
                               static_cast<double>(f.dy(i, j)));
      }
    }
  fv_real *d_crx = dev.up3(crx), *d_cry = dev.up3(cry);
  fv_real *d_xfx = dev.up3(xfx), *d_yfx = dev.up3(yfx);
  fv_real *d_area = dev.up2(area), *d_dxa = dev.up2(dxa), *d_dya = dev.up2(dya);
  fv_real *d_d6u = dev.up2(del6_u), *d_d6v = dev.up2(del6_v);
  fv_real *d_rax = dev.scratch(box), *d_ray = dev.scratch(box);
  fv_real *d_qi = dev.scratch(box), *d_qj = dev.scratch(box);
  fv_real *d_fx2t = dev.scratch(box), *d_fy2t = dev.scratch(box);
  fv_real *d_d2 = dev.scratch(box), *d_fx2d = dev.scratch(box), *d_fy2d = dev.scratch(box);
  if (!d_fy2d) return FV3_GPU_INTERNAL_ERROR;

  st = dgrid::launch_ra_x_cuda(b, C3(d_xfx), C2(d_area), W3(d_rax), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  st = dgrid::launch_ra_y_cuda(b, C3(d_yfx), C2(d_area), W3(d_ray), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;

  dgrid::ScalarTransportOptions sopt;
  sopt.hord = 6;
  sopt.mass_flux_weighting = 0;
  sopt.deln_flux_enabled = 0;
  dgrid::LevelFilter lf;
  lf.npz = box.nk;
  lf.mode = 0;
  const View3<const fv_real> empty3{};
  st = dgrid::launch_scalar_transport_cuda(
      b, sopt, lf, W3(d_vort), empty3, C3(d_crx), C3(d_cry), C3(d_xfx), C3(d_yfx), empty3,
      empty3, C2(d_area), C2(d_rarea), C2(d_dxa), C2(d_dya), C2(d_d6u), C2(d_d6v), C3(d_rax),
      C3(d_ray), W3(d_qi), W3(d_qj), W3(d_fx2t), W3(d_fy2t), W3(d_d2), W3(d_fx2d), W3(d_fy2d),
      W3(d_fx), W3(d_fy), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->fx = down3(d_fx, box);
  o->fy = down3(d_fy, box);

  // row 10
  st = dgrid::launch_dwind_update_cuda(b, C3(d_ut), C3(d_vt), C3(d_ke), C3(d_fx), C3(d_fy),
                                       W3(d_u), W3(d_v), nullptr);
  if (st != FV3_GPU_SUCCESS) return st;
  if (cudaDeviceSynchronize() != cudaSuccess) return FV3_GPU_INTERNAL_ERROR;
  o->un = down3(d_u, box);
  o->vn = down3(d_v, box);
  return FV3_GPU_SUCCESS;
}

// Host counterpart. fx/fy come from the GPU (they are row 10's INPUT, produced
// by the Milestone 3A.5 service this milestone reuses rather than reimplements),
// so row 10 is still compared against an independent transcription.
static void host_sequence(const Fields &f, const A3 &fx, const A3 &fy, SeqOut *o) {
  const Box &box = f.box;
  const fv_real dt = fv_real(DT);
  ref_vb(box, dt, f.uc, f.vc, f.vt, f.cosa, f.rsina, o->vb1);
  ref_ytp_v(box, f.v, o->vb1, f.dy, f.rdy, o->ub1);
  ref_ke_product(box, o->vb1, o->ub1, o->ke1);
  ref_ub(box, dt, f.uc, f.vc, f.ut, f.cosa, f.rsina, o->ub2);
  ref_xtp_u(box, f.u, o->ub2, f.dx, f.rdx, o->vb2);
  o->ke2 = o->ke1;
  ref_ke_average(box, o->ub2, o->vb2, o->ke2);
  o->ke3 = o->ke2;
  ref_ke_corner_fix(box, dt, f.ut, f.vt, f.u, f.v, o->ke3);
  ref_vorticity(box, f.u, f.v, f.dx, f.dy, f.rarea, o->ut8, o->vt8, o->wk8);
  ref_vort_absolute(box, false, o->wk8, f.f0, f.z_rat, o->vort);
  ref_vort_absolute(box, true, o->wk8, f.f0, f.z_rat, o->vort_f3d);
  o->un = f.u;
  o->vn = f.v;
  ref_dwind(box, o->ut8, o->vt8, o->ke3, fx, fy, o->un, o->vn);
}

// A tile block that owns the SOUTH-WEST corner only: is/js are tile edges but
// ie+1/je+1 are interior. Exercises the four edge flags independently, so a
// kernel that keys off "any edge" instead of the specific flag fails here.
static Box corner_box() {
  Box b;
  b.is = 1; b.ie = 12; b.js = 1; b.je = 12;
  b.npx = 25; b.npy = 25;
  b.west = 1; b.south = 1; b.east = 0; b.north = 0;
  return b;
}

// ---------------------------------------------------------------------------
// Part E -- vb / ub construction, vorticity precursor, D-wind update
// ---------------------------------------------------------------------------
static bool part_e() {
  Fields f;
  f.build(seam_box(), 0);
  const Box &b = f.box;
  SeqOut g, h;
  const fv3_gpu_status st = gpu_sequence(f, &g);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  E launch failed: status=%d\n", static_cast<int>(st));
    return false;
  }
  host_sequence(f, g.fx, g.fy, &h);

  const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
  const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
  double w = 0.0, worst = 0.0;
  bool ok = true;
  ok &= cmp_box("E vb (row 1)", b, g.vb1, h.vb1, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E ub (row 4)", b, g.ub2, h.ub2, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E vt (row 8)", b, g.vt8, h.vt8, isd, ied, jsd, jed + 1, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E ut (row 8)", b, g.ut8, h.ut8, isd, ied + 1, jsd, jed, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E wk (row 8)", b, g.wk8, h.wk8, isd, ied, jsd, jed, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E vort (row 9)", b, g.vort, h.vort, isd, ied, jsd, jed, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E vort do_f3d (row 9)", b, g.vort_f3d, h.vort_f3d, isd, ied, jsd, jed, TOL,
                &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E u (row 10)", b, g.un, h.un, b.is, b.ie, b.js, b.je + 1, TOL, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("E v (row 10)", b, g.vn, h.vn, b.is, b.ie + 1, b.js, b.je, TOL, &w);
  worst = w > worst ? w : worst;

  // The row-9b reuse path (Milestone 3A.5 fv_tp_2d) must have produced usable
  // fluxes -- row 10 is meaningless otherwise.
  int64_t finite_fx = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        if (!std::isfinite(static_cast<double>(g.fx(i, j, k)))) {
          std::printf("  E row 9b fx nonfinite at (%lld,%lld,%lld)\n",
                      static_cast<long long>(i), static_cast<long long>(j),
                      static_cast<long long>(k));
          return false;
        }
        if (g.fx(i, j, k) != fv_real(0)) ++finite_fx;
      }
  if (finite_fx < 100) {
    std::printf("  E row 9b vorticity flux is degenerate (%lld nonzero)\n",
                static_cast<long long>(finite_fx));
    ok = false;
  }

  // Momentum must actually change, and by more than round-off.
  int64_t moved = 0;
  double biggest = 0.0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const double d = std::fabs(static_cast<double>(g.un(i, j, k)) -
                                   static_cast<double>(f.u(i, j, k)));
        if (d > 1.0e-6) ++moved;
        biggest = d > biggest ? d : biggest;
      }
  if (moved < 300) {
    std::printf("  E D-wind update is degenerate (only %lld u cells moved)\n",
                static_cast<long long>(moved));
    ok = false;
  }

  // vb and ub are built by two DIFFERENT branch precedences; if one kernel were
  // a transpose of the other they would coincide far more often than this.
  int64_t differ = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        if (!close_rel(static_cast<double>(g.vb1(i, j, k)),
                       static_cast<double>(g.ub2(i, j, k)), 1.0e-6))
          ++differ;
  if (differ < 300) {
    std::printf("  E vb/ub are suspiciously similar (%lld of %lld differ)\n",
                static_cast<long long>(differ),
                static_cast<long long>(b.nk * (b.ie + 2 - b.is) * (b.je + 2 - b.js)));
    ok = false;
  }

  // do_f3d must select a different vorticity than the hydrostatic branch.
  int64_t f3d_differ = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        if (g.vort(i, j, k) != g.vort_f3d(i, j, k)) ++f3d_differ;
  if (f3d_differ < 100) {
    std::printf("  E do_f3d branch is not live (%lld cells differ)\n",
                static_cast<long long>(f3d_differ));
    ok = false;
  }

  std::printf("  E worst rel = %.4g, u cells moved = %lld (max |du| = %.4g), "
              "vb!=ub cells = %lld, do_f3d cells = %lld\n",
              worst, static_cast<long long>(moved), biggest,
              static_cast<long long>(differ), static_cast<long long>(f3d_differ));
  return ok;
}

// ---------------------------------------------------------------------------
// Part F -- kinetic energy (rows 3, 6, 7)
// ---------------------------------------------------------------------------
static bool part_f() {
  Fields f;
  f.build(seam_box(), 0);
  const Box &b = f.box;
  SeqOut g, h;
  const fv3_gpu_status st = gpu_sequence(f, &g);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  F launch failed: status=%d\n", static_cast<int>(st));
    return false;
  }
  host_sequence(f, g.fx, g.fy, &h);

  double w = 0.0, worst = 0.0;
  bool ok = true;
  ok &= cmp_box("F ke = vb*ub (row 3)", b, g.ke1, h.ke1, b.is, b.ie + 1, b.js, b.je + 1, TOL,
                &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("F ke average (row 6)", b, g.ke2, h.ke2, b.is, b.ie + 1, b.js, b.je + 1, TOL,
                &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("F ke corner fix (row 7)", b, g.ke3, h.ke3, b.is, b.ie + 1, b.js, b.je + 1, TOL,
                &w);
  worst = w > worst ? w : worst;

  // Row 6 must not be a no-op, and row 7 must change EXACTLY the four corners.
  int64_t avg_moved = 0, corner_moved = 0, noncorner_moved = 0;
  const int64_t cx[4] = {1, b.npx, b.npx, 1};
  const int64_t cy4[4] = {1, 1, b.npy, b.npy};
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        if (g.ke2(i, j, k) != g.ke1(i, j, k)) ++avg_moved;
        if (g.ke3(i, j, k) == g.ke2(i, j, k)) continue;
        bool is_corner = false;
        for (int c = 0; c < 4; ++c)
          if (i == cx[c] && j == cy4[c]) is_corner = true;
        if (is_corner) ++corner_moved; else ++noncorner_moved;
      }
  if (avg_moved < 300) {
    std::printf("  F row 6 average is degenerate (%lld cells moved)\n",
                static_cast<long long>(avg_moved));
    ok = false;
  }
  if (corner_moved != 4 * b.nk) {
    std::printf("  F corner fix touched %lld of %lld corner cells\n",
                static_cast<long long>(corner_moved), static_cast<long long>(4 * b.nk));
    ok = false;
  }
  if (noncorner_moved != 0) {
    std::printf("  F corner fix leaked into %lld non-corner cells\n",
                static_cast<long long>(noncorner_moved));
    ok = false;
  }

  // The four corner expressions are genuinely different (they differ in which
  // ut/vt neighbours are summed and in the sign of the third term), so a
  // copy-pasted single formula would make some of them coincide.
  bool distinct = true;
  for (int c = 0; c < 4; ++c)
    for (int d = c + 1; d < 4; ++d)
      if (close_rel(static_cast<double>(g.ke3(cx[c], cy4[c], 0)),
                    static_cast<double>(g.ke3(cx[d], cy4[d], 0)), 1.0e-9))
        distinct = false;
  if (!distinct) {
    std::printf("  F two corner KE values coincide -- corner formulas may be shared\n");
    ok = false;
  }
  std::printf("  F worst rel = %.4g, row-6 moved = %lld, corner cells fixed = %lld\n", worst,
              static_cast<long long>(avg_moved), static_cast<long long>(corner_moved));
  return ok;
}

// ---------------------------------------------------------------------------
// Part G -- cubed-sphere seam / corner behavior
// ---------------------------------------------------------------------------
static bool part_g() {
  bool ok = true;

  // G1: a block owning only the south-west corner. Every stage must honour the
  // individual edge flags, not "is this block on any tile boundary".
  {
    Fields f;
    f.build(corner_box(), 0);
    const Box &b = f.box;
    SeqOut g, h;
    const fv3_gpu_status st = gpu_sequence(f, &g);
    if (st != FV3_GPU_SUCCESS) {
      std::printf("  G1 launch failed: status=%d\n", static_cast<int>(st));
      return false;
    }
    host_sequence(f, g.fx, g.fy, &h);
    double w = 0.0;
    ok &= cmp_box("G1 vb", b, g.vb1, h.vb1, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
    ok &= cmp_box("G1 ub", b, g.ub2, h.ub2, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
    ok &= cmp_box("G1 xtp_u flux", b, g.vb2, h.vb2, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
    ok &= cmp_box("G1 ytp_v flux", b, g.ub1, h.ub1, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);
    ok &= cmp_box("G1 ke", b, g.ke3, h.ke3, b.is, b.ie + 1, b.js, b.je + 1, TOL, &w);

    // Only the SW corner fix may fire on this block.
    int64_t fixed = 0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i)
          if (g.ke3(i, j, k) != g.ke2(i, j, k)) {
            ++fixed;
            if (i != 1 || j != 1) {
              std::printf("  G1 unexpected corner fix at (%lld,%lld)\n",
                          static_cast<long long>(i), static_cast<long long>(j));
              ok = false;
            }
          }
    if (fixed != b.nk) {
      std::printf("  G1 SW corner fix fired %lld times, expected %lld\n",
                  static_cast<long long>(fixed), static_cast<long long>(b.nk));
      ok = false;
    }
    std::printf("  G1 south-west-only block ok (worst rel = %.4g, SW fixes = %lld)\n", w,
                static_cast<long long>(fixed));
  }

  // G2: the vb/ub branch-precedence asymmetry, asserted directly at (1,1).
  // sw_core.F90:1063-1097 writes the js==1 ROW after nothing and the is==1
  // COLUMN only inside the clipped j loop, so vb(1,1) is the ROW formula.
  // sw_core.F90:1143-1169 writes the is==1 COLUMN first over the full j range
  // and clips the j-loop i range to is2..ie1, so ub(1,1) is the COLUMN formula.
  // Swapping the two precedences changes both values, so this pins the order.
  {
    Fields f;
    f.build(seam_box(), 0);
    const Box &b = f.box;
    SeqOut g;
    if (gpu_sequence(f, &g) != FV3_GPU_SUCCESS) return false;
    const double dt5 = 0.5 * DT, dt4 = 0.25 * DT;
    const int64_t k = 0;
    const double vb_row = dt5 * (static_cast<double>(f.vt(0, 1, k)) +
                                 static_cast<double>(f.vt(1, 1, k)));
    const double vb_col = dt4 * (-static_cast<double>(f.vt(-1, 1, k)) +
                                 3.0 * (static_cast<double>(f.vt(0, 1, k)) +
                                        static_cast<double>(f.vt(1, 1, k))) -
                                 static_cast<double>(f.vt(2, 1, k)));
    const double ub_col = dt5 * (static_cast<double>(f.ut(1, 0, k)) +
                                 static_cast<double>(f.ut(1, 1, k)));
    const double ub_row = dt4 * (-static_cast<double>(f.ut(1, -1, k)) +
                                 3.0 * (static_cast<double>(f.ut(1, 0, k)) +
                                        static_cast<double>(f.ut(1, 1, k))) -
                                 static_cast<double>(f.ut(1, 2, k)));
    const double vb_got = static_cast<double>(g.vb1(1, 1, k));
    const double ub_got = static_cast<double>(g.ub2(1, 1, k));
    if (!close_rel(vb_got, vb_row, TOL) || close_rel(vb_got, vb_col, 1.0e-6)) {
      std::printf("  G2 vb(1,1) = %.12g, ROW = %.12g, COLUMN = %.12g\n", vb_got, vb_row,
                  vb_col);
      ok = false;
    }
    if (!close_rel(ub_got, ub_col, TOL) || close_rel(ub_got, ub_row, 1.0e-6)) {
      std::printf("  G2 ub(1,1) = %.12g, COLUMN = %.12g, ROW = %.12g\n", ub_got, ub_col,
                  ub_row);
      ok = false;
    }
    if (ok) std::printf("  G2 vb row-first / ub column-first precedence confirmed at (1,1)\n");
    (void)b;
  }

  // G3: swapped x/y metrics must change the answer. xtp_u must use dx/rdx and
  // ytp_v must use dy/rdy; DX0 != DY0 so a swap is observable.
  {
    Fields f;
    f.build(seam_box(), 0);
    const Box &b = f.box;
    A3 good, swapped;
    if (gpu_xtp_u(b, f.u, f.cx, f.dx, f.rdx, -6, &good) != FV3_GPU_SUCCESS) return false;
    if (gpu_xtp_u(b, f.u, f.cx, f.dy, f.rdy, -6, &swapped) != FV3_GPU_SUCCESS) return false;
    int64_t differ = 0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i)
          if (!close_rel(static_cast<double>(good(i, j, k)),
                         static_cast<double>(swapped(i, j, k)), 1.0e-9))
            ++differ;
    if (differ < 200) {
      std::printf("  G3 xtp_u is insensitive to a dx->dy metric swap (%lld cells differ)\n",
                  static_cast<long long>(differ));
      ok = false;
    }
    A3 goody, swappedy;
    if (gpu_ytp_v(b, f.v, f.cy, f.dy, f.rdy, -6, &goody) != FV3_GPU_SUCCESS) return false;
    if (gpu_ytp_v(b, f.v, f.cy, f.dx, f.rdx, -6, &swappedy) != FV3_GPU_SUCCESS) return false;
    int64_t differy = 0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i)
          if (!close_rel(static_cast<double>(goody(i, j, k)),
                         static_cast<double>(swappedy(i, j, k)), 1.0e-9))
            ++differy;
    if (differy < 200) {
      std::printf("  G3 ytp_v is insensitive to a dy->dx metric swap (%lld cells differ)\n",
                  static_cast<long long>(differy));
      ok = false;
    }
    std::printf("  G3 metric-swap detectors live (xtp_u %lld cells, ytp_v %lld cells)\n",
                static_cast<long long>(differ), static_cast<long long>(differy));
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Part H -- precision-sensitive cases
// ---------------------------------------------------------------------------
// H1: a plateau wind. al(i) = p1*(u(i-1)+u(i)) + p2*(u(i-2)+u(i+1)) is
// algebraically u(i) for a constant field but not in floating point, so bl/br
// are pure round-off and `3|b0| < |bl-br|` is a round-off-decided branch.
// Unlike Milestone 3A.5's scalar flux (where the same round-off is multiplied by
// a mass flux), xtp_u adds fx0 directly to an O(u) upwind value, so the branch
// indeterminacy CANNOT escape round-off. Asserting that at a tight tolerance is
// the meaningful statement: it fails if the CUDA flux assembly reorders
// operations enough to amplify the cancellation.
//
// H2: a large mean wind. wk = rarea*(vt - vt - ut + ut) then differences terms
// of size |u|*dx, so its relative accuracy is amplified by |vt|/|wk|. Only the
// Fortran's own grouping survives this at a tight tolerance.
static bool part_h() {
  const double tight = sizeof(fv_real) == 4 ? 2.0e-6 : 1.0e-14;
  bool ok = true;

  {
    Fields f;
    f.build(interior_box(), 1);
    const Box &b = f.box;
    A3 gx, hx, gy, hy;
    if (gpu_xtp_u(b, f.u, f.cx, f.dx, f.rdx, -6, &gx) != FV3_GPU_SUCCESS) return false;
    if (gpu_ytp_v(b, f.v, f.cy, f.dy, f.rdy, -6, &gy) != FV3_GPU_SUCCESS) return false;
    ref_xtp_u(b, f.u, f.cx, f.dx, f.rdx, hx);
    ref_ytp_v(b, f.v, f.cy, f.dy, f.rdy, hy);
    double w = 0.0;
    ok &= cmp_box("H1 xtp_u plateau", b, gx, hx, b.is, b.ie + 1, b.js, b.je + 1, tight, &w);
    double wy = 0.0;
    ok &= cmp_box("H1 ytp_v plateau", b, gy, hy, b.is, b.ie + 1, b.js, b.je + 1, tight, &wy);

    // Evidence that the limiter branch really is round-off-decided here: every
    // flux must equal one of the two pure upwind values to within round-off.
    double worst_dev = 0.0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i) {
          const double got = static_cast<double>(gx(i, j, k));
          const double up = static_cast<double>(f.cx(i, j, k) > fv_real(0) ? f.u(i - 1, j, k)
                                                                          : f.u(i, j, k));
          const double dev = std::fabs(got - up) / (std::fabs(up) + 1.0e-30);
          worst_dev = dev > worst_dev ? dev : worst_dev;
        }
    if (worst_dev > tight) {
      std::printf("  H1 plateau flux deviates from pure upwind by %.4g (tol %.4g)\n", worst_dev,
                  tight);
      ok = false;
    }
    std::printf("  H1 plateau worst rel = %.4g (x) %.4g (y), max deviation from upwind = %.4g\n",
                w, wy, worst_dev);
  }

  {
    Fields f;
    f.build(seam_box(), 0);
    const Box &b = f.box;
    const double mean = 1.0e6;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.loj(); j <= b.je + b.ng + 1; ++j)
        for (int64_t i = b.loi(); i <= b.ie + b.ng + 1; ++i) {
          f.u(i, j, k) = fv_real(static_cast<double>(f.u(i, j, k)) + mean);
          f.v(i, j, k) = fv_real(static_cast<double>(f.v(i, j, k)) + mean);
        }
    SeqOut g, h;
    if (gpu_sequence(f, &g) != FV3_GPU_SUCCESS) return false;
    host_sequence(f, g.fx, g.fy, &h);
    const int64_t isd = b.is - b.ng, ied = b.ie + b.ng;
    const int64_t jsd = b.js - b.ng, jed = b.je + b.ng;
    double w = 0.0;
    ok &= cmp_box("H2 wk cancellation", b, g.wk8, h.wk8, isd, ied, jsd, jed, tight, &w);

    // Confirm the cancellation is real, otherwise H2 proves nothing.
    double amp = 0.0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = jsd; j <= jed; ++j)
        for (int64_t i = isd; i <= ied; ++i) {
          const double big = std::fabs(static_cast<double>(f.rarea(i, j)) *
                                       static_cast<double>(g.vt8(i, j, k)));
          const double small = std::fabs(static_cast<double>(g.wk8(i, j, k)));
          if (small > 0.0) amp = (big / small) > amp ? (big / small) : amp;
        }
    if (amp < 20.0) {
      std::printf("  H2 cancellation amplification is only %.4g -- fixture too benign\n", amp);
      ok = false;
    }
    std::printf("  H2 wk worst rel = %.4g at tol %.4g, cancellation amplification = %.4g\n", w,
                tight, amp);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Parts I / J -- per-precision reporting
// ---------------------------------------------------------------------------
// The two parts are deliberately separate so a run reports R8 and R4 status
// independently rather than as one "precision" verdict.
static bool precision_part(const char *tag, size_t want_bytes, double tol) {
  if (sizeof(fv_real) != want_bytes) {
    std::printf("  %s not this build (fv_real is %zu bytes) -- reported separately\n", tag,
                sizeof(fv_real));
    return true;
  }
  Fields f;
  f.build(seam_box(), 0);
  const Box &b = f.box;
  SeqOut g, h;
  if (gpu_sequence(f, &g) != FV3_GPU_SUCCESS) return false;
  host_sequence(f, g.fx, g.fy, &h);
  double worst = 0.0, w = 0.0;
  bool ok = true;
  ok &= cmp_box("xtp_u flux", b, g.vb2, h.vb2, b.is, b.ie + 1, b.js, b.je + 1, tol, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("ytp_v flux", b, g.ub1, h.ub1, b.is, b.ie + 1, b.js, b.je + 1, tol, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("ke", b, g.ke3, h.ke3, b.is, b.ie + 1, b.js, b.je + 1, tol, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("u", b, g.un, h.un, b.is, b.ie, b.js, b.je + 1, tol, &w);
  worst = w > worst ? w : worst;
  ok &= cmp_box("v", b, g.vn, h.vn, b.is, b.ie + 1, b.js, b.je, tol, &w);
  worst = w > worst ? w : worst;
  std::printf("  %s worst rel = %.4g at tol %.4g\n", tag, worst, tol);
  return ok;
}

static bool part_i() { return precision_part("I R8", 8, 1.0e-11); }
static bool part_j() { return precision_part("J R4", 4, 3.0e-5); }

// ---------------------------------------------------------------------------
// Part K -- configuration gating and hord_mt sign equivalence
// ---------------------------------------------------------------------------
struct Bad {
  int32_t hord = -6;
  int32_t grid_type = 0;
  int32_t nested = 0;
  int32_t cubed = 1;
  int64_t nk = -1;   // -1 -> keep box.nk
  int64_t is_ = -1;  // -1 -> keep box.is
  int64_t npx = -1;  // -1 -> keep box.npx
};

// Returns the launch status and, on rejection, whether the output was left
// untouched. A rejected configuration must not have mutated model state.
static fv3_gpu_status try_xtp_u(const Box &box, const Bad &bad, bool *intact) {
  Fields f;
  f.build(box, 0);
  Dev dev;
  dgrid::TransportBounds b = tb_of(box);
  if (bad.nk >= 0) b.nk = bad.nk;
  if (bad.is_ >= 0) b.is = bad.is_;
  if (bad.npx >= 0) b.npx = bad.npx;
  b.nested = bad.nested;
  b.cubed_sphere = bad.cubed;
  dgrid::MomentumOptions opt;
  opt.hord_mt = bad.hord;
  opt.grid_type = bad.grid_type;

  A3 sentinel;
  sentinel.init(box, fv_real(7.5));
  fv_real *d_u = dev.up3(f.u), *d_c = dev.up3(f.cx);
  fv_real *d_dx = dev.up2(f.dx), *d_rdx = dev.up2(f.rdx);
  fv_real *d_flux = dev.up3(sentinel);
  dgrid::PpmMomentumWorkspace ws;
  ws.bl = V3<fv_real>(dev.scratch(box), box);
  ws.br = V3<fv_real>(dev.scratch(box), box);
  ws.b0 = V3<fv_real>(dev.scratch(box), box);
  ws.smt = V3<int32_t>(dev.scratch_i(box), box);
  if (!ws.smt.data || !d_flux) return FV3_GPU_INTERNAL_ERROR;

  const fv3_gpu_status st = dgrid::launch_xtp_u_cuda(
      b, opt, V3<const fv_real>(d_c, box), V3<const fv_real>(d_u, box),
      V2<const fv_real>(d_dx, box), V2<const fv_real>(d_rdx, box), ws,
      V3<fv_real>(d_flux, box), nullptr);
  cudaDeviceSynchronize();
  if (intact) {
    *intact = true;
    const A3 back = down3(d_flux, box);
    for (size_t n = 0; n < back.d.size(); ++n)
      if (back.d[n] != fv_real(7.5)) { *intact = false; break; }
  }
  return st;
}

static bool part_k() {
  bool ok = true;
  const Box box = seam_box();
  struct Case { const char *name; Bad bad; };
  Bad b_hord;   b_hord.hord = 8;
  Bad b_grid;   b_grid.grid_type = 4;
  Bad b_nest;   b_nest.nested = 1;
  Bad b_cube;   b_cube.cubed = 0;
  Bad b_nk;     b_nk.nk = 0;
  Bad b_is;     b_is.is_ = 2;
  Bad b_npx;    b_npx.npx = 6;
  const Case cases[] = {
      {"hord_mt = 8", b_hord},   {"grid_type = 4", b_grid}, {"nested = 1", b_nest},
      {"cubed_sphere = 0", b_cube}, {"nk = 0", b_nk},       {"is = 2 (PPM clip fence)", b_is},
      {"npx = 6", b_npx},
  };
  for (const Case &c : cases) {
    bool intact = false;
    const fv3_gpu_status st = try_xtp_u(box, c.bad, &intact);
    if (st != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  K %s was not rejected (status=%d)\n", c.name, static_cast<int>(st));
      ok = false;
      continue;
    }
    if (!intact) {
      std::printf("  K %s was rejected but mutated the output\n", c.name);
      ok = false;
    }
  }

  // The hord gate is scoped to the PPM stages: vb/ub carry no reconstruction,
  // so sw_core's vb construction is valid for any hord_mt. Asserting this keeps
  // a future d_sw driver from over-rejecting.
  {
    Fields f;
    f.build(box, 0);
    Dev dev;
    dgrid::TransportBounds b = tb_of(box);
    dgrid::MomentumOptions opt;
    opt.hord_mt = 8;
    fv_real *d_uc = dev.up3(f.uc), *d_vc = dev.up3(f.vc);
    fv_real *d_ut = dev.up3(f.ut), *d_vt = dev.up3(f.vt);
    fv_real *d_cosa = dev.up2(f.cosa), *d_rsina = dev.up2(f.rsina);
    fv_real *d_vb = dev.scratch(box);
    if (!d_vb) return false;
    const fv3_gpu_status st = dgrid::launch_bgrid_vb_cuda(
        b, opt, fv_real(DT), V3<const fv_real>(d_uc, box), V3<const fv_real>(d_vc, box),
        V3<const fv_real>(d_ut, box), V3<const fv_real>(d_vt, box),
        V2<const fv_real>(d_cosa, box), V2<const fv_real>(d_rsina, box),
        V3<fv_real>(d_vb, box), nullptr);
    cudaDeviceSynchronize();
    if (st != FV3_GPU_SUCCESS) {
      std::printf("  K vb construction wrongly rejected hord_mt = 8 (status=%d)\n",
                  static_cast<int>(st));
      ok = false;
    }
  }

  // hord_mt = -6 and +6 must be bit-identical: xtp_u/ytp_v have no iord<0
  // branch (unlike xppm/yppm, which add an al >= 0 clamp).
  {
    Fields f;
    f.build(box, 0);
    const Box &b = f.box;
    A3 xm, xp, ym, yp;
    if (gpu_xtp_u(b, f.u, f.cx, f.dx, f.rdx, -6, &xm) != FV3_GPU_SUCCESS) return false;
    if (gpu_xtp_u(b, f.u, f.cx, f.dx, f.rdx, 6, &xp) != FV3_GPU_SUCCESS) return false;
    if (gpu_ytp_v(b, f.v, f.cy, f.dy, f.rdy, -6, &ym) != FV3_GPU_SUCCESS) return false;
    if (gpu_ytp_v(b, f.v, f.cy, f.dy, f.rdy, 6, &yp) != FV3_GPU_SUCCESS) return false;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i) {
          if (xm(i, j, k) != xp(i, j, k)) {
            std::printf("  K xtp_u hord_mt sign changed the answer at (%lld,%lld,%lld)\n",
                        static_cast<long long>(i), static_cast<long long>(j),
                        static_cast<long long>(k));
            return false;
          }
          if (ym(i, j, k) != yp(i, j, k)) {
            std::printf("  K ytp_v hord_mt sign changed the answer at (%lld,%lld,%lld)\n",
                        static_cast<long long>(i), static_cast<long long>(j),
                        static_cast<long long>(k));
            return false;
          }
        }
  }
  if (ok)
    std::printf("  K 7 unsupported configurations rejected with state intact; hord_mt +/-6 "
                "bit-identical\n");
  return ok;
}

// ---------------------------------------------------------------------------
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::printf("SKIP: no CUDA device visible (WP-B.3 momentum transport requires a GPU)\n");
    return 77;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("FAIL: cudaGetDeviceProperties\n");
    return 1;
  }
  std::printf("WP-B.3 momentum transport on %s (sm_%d%d), fv_real = %zu bytes\n", prop.name,
              prop.major, prop.minor, sizeof(fv_real));

  struct Part { const char *name; bool (*fn)(); };
  const Part parts[] = {
      {"A xtp_u interior", part_a},
      {"B ytp_v interior", part_b},
      {"C xtp_u panel edge", part_c},
      {"D ytp_v panel edge", part_d},
      {"E vb/ub, vorticity precursor, D-wind update", part_e},
      {"F kinetic energy", part_f},
      {"G cubed-sphere seam / corner", part_g},
      {"H precision-sensitive", part_h},
      {"I R8", part_i},
      {"J R4", part_j},
      {"K configuration gating", part_k},
  };
  const size_t n = sizeof(parts) / sizeof(parts[0]);
  size_t failed = 0;
  for (size_t p = 0; p < n; ++p) {
    std::printf("Part %s\n", parts[p].name);
    if (!parts[p].fn()) {
      std::printf("FAIL: part %s\n", parts[p].name);
      ++failed;
    }
  }
  const cudaError_t residual = cudaGetLastError();
  if (residual != cudaSuccess) {
    std::printf("FAIL: residual CUDA error: %s\n", cudaGetErrorString(residual));
    ++failed;
  }
  if (failed != 0) {
    std::printf("FAILED %zu of %zu parts\n", failed, n);
    return 1;
  }
  std::printf("ALL PARTS PASS (%zu parts)\n", n);
  return 0;
}
