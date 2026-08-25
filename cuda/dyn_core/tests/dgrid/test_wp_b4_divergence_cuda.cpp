// ===========================================================================
// Milestone 3A.7 / WP-B.4 -- authoritative divergence-damping behavior test.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45,
// model/sw_core.F90 (d_sw :1277-1534, smag_corner :1857-1947) and
// tools/fv_mp_mod.F90 (fill_corners_2d_r8, fill_corners_dgrid_r8).
//
// Every numerical part compares the CUDA result against a host transcription
// written directly from that Fortran, in the FORTRAN'S OWN multi-pass
// structure: poisoned locals, the clipped interior loop, then the edge
// override blocks, then the corner removals, then the scale/limiter pass. The
// CUDA kernels resolve the same overwrite order per point instead. The two
// therefore differ structurally and agreement is evidence, not tautology.
//
// TWO DELIBERATE EXCEPTIONS, both stated rather than hidden:
//   * Part G (smag_corner) transcribes S1..S7 and S9 on the host but obtains
//     the reference `sh` by calling WP-D's already-validated
//     operators::launch_a2b_ord4_global_cuda on the HOST-computed wk. What is
//     under test is therefore that smag_corner feeds a2b_ord4 the right input
//     and combines its output correctly -- not a2b_ord4's interior, which is
//     WP-D's test's job. Re-transcribing a2b_ord4 here would duplicate
//     numerical functionality, which the project rules forbid.
//   * Part J (launch_vorticity_damping_cuda) is a deliberately non-numerical
//     adapter over Milestone 3A.4's launch_del6_vt_flux_cuda. It is checked for
//     BITWISE identity against a direct call to that launcher, which is exactly
//     what an adapter's contract is, plus rejection of the argument mistakes
//     the adapter exists to prevent.
//
// The NaN poison is load-bearing: if a CUDA kernel covered a cell the Fortran's
// pass structure leaves undefined, the reference would carry a NaN into the
// comparison and the part would fail rather than silently agree.
//
// Fixtures:
//   seam     npx = npy = 13, is/js = 1, ie/je = 12 -- owns all four tile edges
//            and all four corners. Storage i,j in [-2, 16], which is exactly
//            what nord = 3 needs: H2b at nt = 2 reads divg_d(ie+2+nt) = 16.
//   interior npx = npy = 25, is/js = 7, ie/je = 18 -- owns no edge, so every
//            formula takes the unclipped contravariant branch.
//
// Exit codes: 0 pass, 1 fail, 77 no CUDA device (CTest SKIP_RETURN_CODE).
// ===========================================================================

#include "dgrid/wp_b_divergence_cuda.cuh"
#include "operators/wp_d_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using fv3::gpu::accumulator_real;
using fv3::gpu::fv_real;
using fv3::gpu::operators::View1;
using fv3::gpu::operators::View2;
using fv3::gpu::operators::View3;
namespace dgrid = fv3::gpu::dgrid;
namespace ops = fv3::gpu::operators;

static const double TOL = sizeof(fv_real) == 4 ? 3e-5 : 1e-11;

static bool close_rel(double a, double b, double tol) {
  if (std::isnan(a) || std::isnan(b)) return false;
  const double d = std::fabs(a - b);
  const double s = std::fmax(std::fabs(a), std::fabs(b));
  return d <= tol * (s > 1.0 ? s : 1.0);
}

static fv_real nan_poison() { return std::numeric_limits<fv_real>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// index box: one uniform storage rectangle for every field
// ---------------------------------------------------------------------------
// i in [is-ng, ie+ng+1], j in [js-ng, je+ng+1]. The +1 covers the staggered
// upper bounds: smag_corner's ut over isd..ied+1 and vt over jsd..jed+1, and
// H2b/H2d's uc/vc at nt = nord-1.
struct Box {
  int64_t is = 1, ie = 12, js = 1, je = 12, ng = 3, nk = 3;
  int64_t npx = 13, npy = 13;
  int32_t west = 1, east = 1, south = 1, north = 1;

  int64_t loi() const { return is - ng; }
  int64_t loj() const { return js - ng; }
  int64_t hii() const { return ie + ng + 1; }
  int64_t hij() const { return je + ng + 1; }
  int64_t ni() const { return hii() - loi() + 1; }
  int64_t nj() const { return hij() - loj() + 1; }
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

// gridstruct%sin_sg: four components, carried as the k axis of a View3 with the
// component index 0-BASED (Fortran sin_sg(:,:,4) is component 3).
struct SG {
  const Box *b = nullptr;
  std::vector<fv_real> d;
  void init(const Box &bx) {
    b = &bx;
    d.assign(static_cast<size_t>(bx.n2() * 4), fv_real(0));
  }
  size_t off(int64_t i, int64_t j, int64_t c) const {
    return static_cast<size_t>((i - b->loi()) + (j - b->loj()) * b->ni() + c * b->n2());
  }
  fv_real &operator()(int64_t i, int64_t j, int64_t c) { return d[off(i, j, c)]; }
  fv_real operator()(int64_t i, int64_t j, int64_t c) const { return d[off(i, j, c)]; }
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
  template <class A> fv_real *up(const A &a) {
    void *p = raw(a.d.size() * sizeof(fv_real));
    if (p)
      cudaMemcpy(p, a.d.data(), a.d.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return static_cast<fv_real *>(p);
  }
  fv_real *scratch(const Box &b) {
    return static_cast<fv_real *>(raw(static_cast<size_t>(b.n3()) * sizeof(fv_real)));
  }
  fv_real *lev(const std::vector<fv_real> &v) {
    void *p = raw(v.size() * sizeof(fv_real));
    if (p) cudaMemcpy(p, v.data(), v.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return static_cast<fv_real *>(p);
  }
  accumulator_real *acc(const std::vector<accumulator_real> &v) {
    void *p = raw(v.size() * sizeof(accumulator_real));
    if (p)
      cudaMemcpy(p, v.data(), v.size() * sizeof(accumulator_real),
                 cudaMemcpyHostToDevice);
    return static_cast<accumulator_real *>(p);
  }
};

template <class T> static View3<T> V3(void *p, const Box &b) {
  return View3<T>{static_cast<T *>(p), b.loi(), b.loj(), 0, 1, b.ni(), b.ni() * b.nj()};
}
template <class T> static View2<T> V2(void *p, const Box &b) {
  return View2<T>{static_cast<T *>(p), b.loi(), b.loj(), 1, b.ni()};
}
static View3<const fv_real> VSG(void *p, const Box &b) {
  return View3<const fv_real>{static_cast<const fv_real *>(p), b.loi(), b.loj(), 0, 1,
                              b.ni(), b.n2()};
}
static View1<const fv_real> VL(void *p) {
  return View1<const fv_real>{static_cast<const fv_real *>(p), 0, 1};
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
// rather than by round-off.
static const double DX0 = 1.0e5;
static const double DY0 = 1.2e5;
static const double DT = 300.0;
static const double DA_MIN = 1.15e10;
static const double DA_MIN_C = 1.0e10;

static fv_real f_u(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(20.0 * std::sin(0.4 * x + 0.2 * y) + 3.0 * std::cos(0.13 * y) +
                 0.05 * x + 0.7 * double(k));
}
static fv_real f_v(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(17.0 * std::cos(0.31 * y + 0.17 * x) - 4.0 * std::sin(0.23 * x) +
                 0.04 * y + 0.6 * double(k));
}
static fv_real f_ua(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(11.0 * std::sin(0.27 * x - 0.19 * y) + 0.5 * double(k));
}
static fv_real f_va(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(-9.0 * std::cos(0.22 * x + 0.31 * y) + 0.4 * double(k));
}
// uc/vc are read ONLY for sign(uc)/sign(vc) in the one-sided sin_sg formulas.
// These cross zero many times inside the seam box, so both arms are live.
static fv_real f_uc(int64_t i, int64_t j, int64_t k) {
  return fv_real(std::sin(0.9 * double(i) + 0.35 * double(j) + 0.2 * double(k)));
}
static fv_real f_vc(int64_t i, int64_t j, int64_t k) {
  return fv_real(std::cos(0.8 * double(j) + 0.45 * double(i) + 0.3 * double(k)));
}

// Grid metrics. sin_sg components are separated by 0.05 so a component mix-up
// shows up as a several-percent error, far above TOL, not as round-off.
struct Grid {
  A2 rarea_c, rarea, dxc, dyc, dx, dy, rdx, rdy, divg_u, divg_v;
  A2 cosa_u, cosa_v, sina_u, sina_v, cosa_s, rsin2, del6_u, del6_v;
  SG sin_sg;

  void init(const Box &b) {
    rarea_c.init(b); rarea.init(b); dxc.init(b); dyc.init(b); dx.init(b); dy.init(b);
    rdx.init(b); rdy.init(b); divg_u.init(b); divg_v.init(b);
    cosa_u.init(b); cosa_v.init(b); sina_u.init(b); sina_v.init(b);
    cosa_s.init(b); rsin2.init(b); del6_u.init(b); del6_v.init(b);
    sin_sg.init(b);
    for (int64_t j = b.loj(); j <= b.hij(); ++j) {
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const double x = double(i), y = double(j);
        const double vdx = DX0 * (1.0 + 0.03 * std::sin(0.3 * x) + 0.02 * std::cos(0.2 * y));
        const double vdy = DY0 * (1.0 + 0.02 * std::cos(0.25 * x) + 0.03 * std::sin(0.17 * y));
        dx(i, j) = fv_real(vdx);
        dy(i, j) = fv_real(vdy);
        rdx(i, j) = fv_real(1.0 / vdx);
        rdy(i, j) = fv_real(1.0 / vdy);
        dxc(i, j) = fv_real(vdx * 1.04);
        dyc(i, j) = fv_real(vdy * 0.97);
        rarea(i, j) = fv_real(1.0 / (vdx * vdy));
        rarea_c(i, j) = fv_real(1.0 / (vdx * vdy * 1.01));
        divg_u(i, j) = fv_real(0.37 * (1.0 + 0.05 * std::sin(0.28 * x + 0.11 * y)));
        divg_v(i, j) = fv_real(0.41 * (1.0 + 0.04 * std::cos(0.19 * x - 0.23 * y)));
        const double cu = 0.08 * std::sin(0.21 * x + 0.11 * y);
        const double cv = 0.07 * std::cos(0.17 * x - 0.13 * y);
        cosa_u(i, j) = fv_real(cu);
        cosa_v(i, j) = fv_real(cv);
        sina_u(i, j) = fv_real(std::sqrt(1.0 - cu * cu));
        sina_v(i, j) = fv_real(std::sqrt(1.0 - cv * cv));
        cosa_s(i, j) = fv_real(0.06 * std::sin(0.15 * x - 0.24 * y));
        rsin2(i, j) = fv_real(1.0 / (1.0 - 0.0036 * std::sin(0.15 * x) * std::sin(0.15 * x)));
        del6_u(i, j) = fv_real(vdy / vdx * 0.9);
        del6_v(i, j) = fv_real(vdx / vdy * 1.1);
        for (int64_t c = 0; c < 4; ++c)
          sin_sg(i, j, c) = fv_real(0.80 + 0.05 * double(c) +
                                    0.01 * std::sin(0.3 * x + 0.4 * y + double(c)));
      }
    }
  }
};

static dgrid::DivergenceGeometry dg_of(Dev &d, const Box &b, const Grid &g) {
  dgrid::DivergenceGeometry o{};
  o.rarea_c = V2<const fv_real>(d.up(g.rarea_c), b);
  o.rarea = V2<const fv_real>(d.up(g.rarea), b);
  o.dxc = V2<const fv_real>(d.up(g.dxc), b);
  o.dyc = V2<const fv_real>(d.up(g.dyc), b);
  o.dx = V2<const fv_real>(d.up(g.dx), b);
  o.dy = V2<const fv_real>(d.up(g.dy), b);
  o.rdx = V2<const fv_real>(d.up(g.rdx), b);
  o.rdy = V2<const fv_real>(d.up(g.rdy), b);
  o.divg_u = V2<const fv_real>(d.up(g.divg_u), b);
  o.divg_v = V2<const fv_real>(d.up(g.divg_v), b);
  o.cosa_u = V2<const fv_real>(d.up(g.cosa_u), b);
  o.cosa_v = V2<const fv_real>(d.up(g.cosa_v), b);
  o.sina_u = V2<const fv_real>(d.up(g.sina_u), b);
  o.sina_v = V2<const fv_real>(d.up(g.sina_v), b);
  o.cosa_s = V2<const fv_real>(d.up(g.cosa_s), b);
  o.rsin2 = V2<const fv_real>(d.up(g.rsin2), b);
  o.sin_sg = VSG(d.up(g.sin_sg), b);
  return o;
}

// ---------------------------------------------------------------------------
// host reference: branch L, sw_core.F90:1277-1358
// ---------------------------------------------------------------------------
// Limiter-arm counters, so the test can assert that the max()/min() branches
// are actually exercised instead of one arm winning everywhere.
struct LimStats {
  long clamp20 = 0, dddmp_arm = 0, d2_arm = 0;
};

static void ref_branch_l(const Box &b, const Grid &g, double dddmp, double dt,
                         const std::vector<fv_real> &d2_bg, A3 &ptc, A3 &vort,
                         A3 &delpc, A3 &ke, LimStats *ls) {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  for (int64_t k = 0; k < b.nk; ++k) {
    // Fortran automatic locals are undefined on entry; poison so that a read of
    // a cell the pass structure never writes propagates a NaN.
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        ptc(i, j, k) = nan_poison();
        vort(i, j, k) = nan_poison();
        delpc(i, j, k) = nan_poison();
      }

    // :1298-1313
    for (int64_t j = js; j <= je + 1; ++j) {
      if (j == 1 || j == npy) {
        for (int64_t i = is - 1; i <= ie + 1; ++i)
          ptc(i, j, k) = f_vc(i, j, k) > fv_real(0)
                             ? f_u(i, j, k) * g.dyc(i, j) * g.sin_sg(i, j - 1, 3)
                             : f_u(i, j, k) * g.dyc(i, j) * g.sin_sg(i, j, 1);
      } else {
        for (int64_t i = is - 1; i <= ie + 1; ++i)
          ptc(i, j, k) = (f_u(i, j, k) - fv_real(0.5) * (f_va(i, j - 1, k) +
                                                         f_va(i, j, k)) *
                                             g.cosa_v(i, j)) *
                         g.dyc(i, j) * g.sina_v(i, j);
      }
    }

    // :1316-1335, with is2 = max(2,is) and ie1 = min(npx-1, ie+1)
    const int64_t is2 = is < 2 ? 2 : is;
    const int64_t ie1 = (ie + 1) < (npx - 1) ? (ie + 1) : (npx - 1);
    for (int64_t j = js - 1; j <= je + 1; ++j) {
      for (int64_t i = is2; i <= ie1; ++i)
        vort(i, j, k) = (f_v(i, j, k) - fv_real(0.5) * (f_ua(i - 1, j, k) +
                                                        f_ua(i, j, k)) *
                                            g.cosa_u(i, j)) *
                        g.dxc(i, j) * g.sina_u(i, j);
      if (is == 1)
        vort(1, j, k) = f_uc(1, j, k) > fv_real(0)
                            ? f_v(1, j, k) * g.dxc(1, j) * g.sin_sg(0, j, 2)
                            : f_v(1, j, k) * g.dxc(1, j) * g.sin_sg(1, j, 0);
      if (ie + 1 == npx)
        vort(npx, j, k) =
            f_uc(npx, j, k) > fv_real(0)
                ? f_v(npx, j, k) * g.dxc(npx, j) * g.sin_sg(npx - 1, j, 2)
                : f_v(npx, j, k) * g.dxc(npx, j) * g.sin_sg(npx, j, 0);
    }

    // :1338-1342
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i)
        delpc(i, j, k) =
            vort(i, j - 1, k) - vort(i, j, k) + ptc(i - 1, j, k) - ptc(i, j, k);

    // :1345-1349 -- unscaled, before rarea_c
    if (b.west && b.south) delpc(1, 1, k) = delpc(1, 1, k) - vort(1, 0, k);
    if (b.east && b.south) delpc(npx, 1, k) = delpc(npx, 1, k) - vort(npx, 0, k);
    if (b.east && b.north) delpc(npx, npy, k) = delpc(npx, npy, k) + vort(npx, npy, k);
    if (b.west && b.north) delpc(1, npy, k) = delpc(1, npy, k) + vort(1, npy, k);

    // :1351-1358
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        const fv_real dc = g.rarea_c(i, j) * delpc(i, j, k);
        delpc(i, j, k) = dc;
        const fv_real raw = fv_real(dddmp) * fv_real(std::fabs(double(dc) * dt));
        const fv_real lim = raw < fv_real(0.20) ? raw : fv_real(0.20);
        const fv_real d2 = d2_bg[size_t(k)];
        const fv_real m = d2 > lim ? d2 : lim;
        if (ls) {
          if (raw >= fv_real(0.20)) ++ls->clamp20;
          if (d2 > lim) ++ls->d2_arm; else ++ls->dddmp_arm;
        }
        const fv_real damp = fv_real(DA_MIN_C) * m;
        const fv_real inc = damp * dc;
        vort(i, j, k) = inc;
        ke(i, j, k) = ke(i, j, k) + inc;
      }
  }
}

// ---------------------------------------------------------------------------
// host reference: fill_corners, tools/fv_mp_mod.F90
// ---------------------------------------------------------------------------
static void ref_fill_bgrid(const Box &b, int dir, A3 &q, int64_t k) {
  const int64_t npx = b.npx, npy = b.npy, ng = b.ng;
  const bool w = b.west, e = b.east, s = b.south, n = b.north;
  for (int64_t j = 1; j <= ng; ++j)
    for (int64_t i = 1; i <= ng; ++i) {
      if (dir == 2) {  // :1134-1141
        if (w && s) q(1 - j, 1 - i, k) = q(i + 1, 1 - j, k);
        if (w && n) q(1 - j, npy + i, k) = q(i + 1, npy + j, k);
        if (e && s) q(npx + j, 1 - i, k) = q(npx - i, 1 - j, k);
        if (e && n) q(npx + j, npy + i, k) = q(npx - i, npy + j, k);
      } else {  // :1119-1126
        if (w && s) q(1 - i, 1 - j, k) = q(1 - j, i + 1, k);
        if (w && n) q(1 - i, npy + j, k) = q(1 - j, npy - i, k);
        if (e && s) q(npx + i, 1 - j, k) = q(npx + j, i + 1, k);
        if (e && n) q(npx + i, npy + j, k) = q(npx + j, npy - i, k);
      }
    }
}

// mySign = -1 (VECTOR = .true.) and it multiplies SW and NE ONLY. The two
// Fortran loops are sequential, and loop 2 reads x -- which loop 1 has already
// modified. Reproduced in that order.
static void ref_fill_dgrid(const Box &b, A3 &x, A3 &y, int64_t k) {
  const int64_t npx = b.npx, npy = b.npy, ng = b.ng;
  const bool w = b.west, e = b.east, s = b.south, n = b.north;
  const fv_real ms = fv_real(-1);
  for (int64_t j = 1; j <= ng; ++j)
    for (int64_t i = 1; i <= ng; ++i) {
      if (w && s) x(1 - i, 1 - j, k) = ms * y(1 - j, i, k);
      if (w && n) x(1 - i, npy + j, k) = y(1 - j, npy - i, k);
      if (e && s) x(npx - 1 + i, 1 - j, k) = y(npx + j, i, k);
      if (e && n) x(npx - 1 + i, npy + j, k) = ms * y(npx + j, npy - i, k);
    }
  for (int64_t j = 1; j <= ng; ++j)
    for (int64_t i = 1; i <= ng; ++i) {
      if (w && s) y(1 - i, 1 - j, k) = ms * x(j, 1 - i, k);
      if (w && n) y(1 - i, npy - 1 + j, k) = x(j, npy + i, k);
      if (e && s) y(npx + i, 1 - j, k) = x(npx - j, 1 - i, k);
      if (e && n) y(npx + i, npy - 1 + j, k) = ms * x(npx - j, npy + i, k);
    }
}

// ---------------------------------------------------------------------------
// shared level-array plumbing
// ---------------------------------------------------------------------------
struct Levels {
  std::vector<fv_real> nord_k, d2_bg, d4_bg, d_con;
  void init(int64_t nk) {
    nord_k.assign(size_t(nk), fv_real(0));
    d2_bg.assign(size_t(nk), fv_real(0));
    d4_bg.assign(size_t(nk), fv_real(0));
    d_con.assign(size_t(nk), fv_real(0));
  }
};

static dgrid::DivergenceLevels dl_of(Dev &d, const Box &b, const Levels &l,
                                     int32_t mode, int32_t nord_sel) {
  dgrid::DivergenceLevels lv{};
  lv.nord_k = VL(d.lev(l.nord_k));
  lv.d2_bg = VL(d.lev(l.d2_bg));
  lv.d4_bg = VL(d.lev(l.d4_bg));
  lv.d_con = VL(d.lev(l.d_con));
  lv.npz = b.nk;
  lv.mode = mode;
  lv.nord_sel = nord_sel;
  return lv;
}

// ---------------------------------------------------------------------------
// Part A / B -- branch L (nord == 0), seam and interior fixtures
// ---------------------------------------------------------------------------
static bool run_branch_l(const Box &b, const char *tag, bool expect_limiter_spread) {
  Grid g;
  g.init(b);
  Levels lev;
  lev.init(b.nk);
  // d2_bg chosen so all three limiter arms are exercised:
  //   k=0: d2_bg = 0     -> the dddmp term always wins the max()
  //   k=1: d2_bg = 0.5   -> d2_bg always wins (min() can never exceed 0.20)
  //   k=2: d2_bg = 0.05  -> mixed across the box
  const double d2v[3] = {0.0, 0.5, 0.05};
  for (int64_t k = 0; k < b.nk; ++k) lev.d2_bg[size_t(k)] = fv_real(d2v[k % 3]);
  // dddmp large enough that dddmp*|delpc*dt| straddles 0.20 across the box.
  const double dddmp = 1000.0;

  A3 u, v, ua, va, uc, vc, ke0;
  u.init(b); v.init(b); ua.init(b); va.init(b); uc.init(b); vc.init(b); ke0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        u(i, j, k) = f_u(i, j, k);
        v(i, j, k) = f_v(i, j, k);
        ua(i, j, k) = f_ua(i, j, k);
        va(i, j, k) = f_va(i, j, k);
        uc(i, j, k) = f_uc(i, j, k);
        vc(i, j, k) = f_vc(i, j, k);
        // ke arrives with prior content: L5 accumulates, it does not overwrite.
        ke0(i, j, k) = fv_real(0.5 * std::sin(0.11 * double(i) + 0.23 * double(j)) +
                               double(k));
      }

  A3 ptc_r, vort_r, delpc_r, ke_r;
  ptc_r.init(b); vort_r.init(b); delpc_r.init(b);
  ke_r = ke0;
  LimStats ls;
  ref_branch_l(b, g, dddmp, DT, lev.d2_bg, ptc_r, vort_r, delpc_r, ke_r, &ls);

  if (expect_limiter_spread) {
    if (ls.clamp20 == 0 || ls.dddmp_arm == 0 || ls.d2_arm == 0) {
      std::printf("  FAIL %s: limiter arms not all live (clamp=%ld dddmp=%ld d2=%ld)\n",
                  tag, ls.clamp20, ls.dddmp_arm, ls.d2_arm);
      return false;
    }
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::DivergenceGeometry dg = dg_of(d, b, g);
  dgrid::DivergenceDampingOptions opt{};
  opt.grid_type = 0;
  opt.do_skeb = 0;
  opt.dddmp = fv_real(dddmp);
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  dgrid::DivergenceLevels lv = dl_of(d, b, lev, 1, 0);

  fv_real *d_u = d.up(u), *d_v = d.up(v), *d_ua = d.up(ua), *d_va = d.up(va);
  fv_real *d_uc = d.up(uc), *d_vc = d.up(vc), *d_ke = d.up(ke0);
  fv_real *d_ptc = d.scratch(b), *d_vort = d.scratch(b), *d_delpc = d.scratch(b);
  if (!d_u || !d_ke || !d_ptc || !d_vort || !d_delpc) {
    cudaStreamDestroy(s);
    return false;
  }

  const fv3_gpu_status st = dgrid::launch_divdamp_nord0_cuda(
      tb_of(b), opt, lv, fv_real(DT), V3<const fv_real>(d_u, b),
      V3<const fv_real>(d_v, b), V3<const fv_real>(d_ua, b), V3<const fv_real>(d_va, b),
      V3<const fv_real>(d_uc, b), V3<const fv_real>(d_vc, b), dg, V3<fv_real>(d_ptc, b),
      V3<fv_real>(d_vort, b), V3<fv_real>(d_delpc, b), V3<fv_real>(d_ke, b), s);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  FAIL %s: launch status %d\n", tag, int(st));
    cudaStreamDestroy(s);
    return false;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) {
    cudaStreamDestroy(s);
    return false;
  }
  const A3 ptc_c = down3(d_ptc, b), vort_c = down3(d_vort, b);
  const A3 delpc_c = down3(d_delpc, b), ke_c = down3(d_ke, b);
  cudaStreamDestroy(s);

  long bad = 0;
  double peak = 0.0;
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is - 1; i <= b.ie + 1; ++i) {
        if (!close_rel(ptc_r(i, j, k), ptc_c(i, j, k), TOL)) {
          if (bad < 4)
            std::printf("  ptc mismatch %s (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n", tag,
                        (long long)i, (long long)j, (long long)k, double(ptc_r(i, j, k)),
                        double(ptc_c(i, j, k)));
          ++bad;
        }
      }
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        const double dr[4] = {double(delpc_r(i, j, k)), double(vort_r(i, j, k)),
                              double(ke_r(i, j, k)), 0.0};
        const double dc[4] = {double(delpc_c(i, j, k)), double(vort_c(i, j, k)),
                              double(ke_c(i, j, k)), 0.0};
        const char *nm[3] = {"delpc", "vort", "ke"};
        for (int f = 0; f < 3; ++f) {
          if (!close_rel(dr[f], dc[f], TOL)) {
            if (bad < 8)
              std::printf("  %s mismatch %s (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                          nm[f], tag, (long long)i, (long long)j, (long long)k, dr[f],
                          dc[f]);
            ++bad;
          }
          const double s2 = std::fmax(std::fabs(dr[f]), 1.0);
          peak = std::fmax(peak, std::fabs(dr[f] - dc[f]) / s2);
        }
      }
  }
  // Non-degeneracy: a uniformly zero delpc would pass any comparison.
  double amax = 0.0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        amax = std::fmax(amax, std::fabs(double(delpc_r(i, j, k))));
  if (amax < 1e-12) {
    std::printf("  FAIL %s: reference delpc is degenerate (max |delpc| = %g)\n", tag,
                amax);
    return false;
  }
  std::printf("  %s: %ld mismatches, peak rel dev %.3g, |delpc|max %.3g\n", tag, bad,
              peak, amax);
  return bad == 0;
}

static bool part_a() { return run_branch_l(seam_box(), "seam", true); }
static bool part_b() { return run_branch_l(interior_box(), "interior", false); }

// ---------------------------------------------------------------------------
// Part C -- fill_corners BGRID scalar rotation, both directions
// ---------------------------------------------------------------------------
static bool part_c() {
  const Box b = seam_box();
  bool ok = true;
  for (int dir = 1; dir <= 2; ++dir) {
    A3 q;
    q.init(b);
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.loj(); j <= b.hij(); ++j)
        for (int64_t i = b.loi(); i <= b.hii(); ++i)
          // Distinct per (i,j,k) so any index confusion is visible, and never
          // zero so a missed write cannot pass by coincidence.
          q(i, j, k) = fv_real(1000.0 * double(k + 1) + 7.0 * double(i) +
                               0.125 * double(j) + 3.5);
    A3 ref = q;
    for (int64_t k = 0; k < b.nk; ++k) ref_fill_bgrid(b, dir, ref, k);

    Dev d;
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return false;
    Levels lev;
    lev.init(b.nk);
    fv_real *dq = d.up(q);
    const fv3_gpu_status st = dgrid::launch_fill_corners_bgrid_cuda(
        tb_of(b), int32_t(dir), dl_of(d, b, lev, 0, 0), V3<fv_real>(dq, b), s);
    if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
      std::printf("  FAIL bgrid dir=%d status %d\n", dir, int(st));
      cudaStreamDestroy(s);
      return false;
    }
    const A3 got = down3(dq, b);
    cudaStreamDestroy(s);

    long bad = 0, changed = 0;
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.loj(); j <= b.hij(); ++j)
        for (int64_t i = b.loi(); i <= b.hii(); ++i) {
          if (double(ref(i, j, k)) != double(q(i, j, k))) ++changed;
          if (!close_rel(ref(i, j, k), got(i, j, k), TOL)) {
            if (bad < 6)
              std::printf("  bgrid dir=%d mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                          dir, (long long)i, (long long)j, (long long)k,
                          double(ref(i, j, k)), double(got(i, j, k)));
            ++bad;
          }
        }
    // 4 corners * ng*ng * nk = 4*9*3 = 108 cells must actually move.
    const long expect = 4 * b.ng * b.ng * b.nk;
    if (changed != expect) {
      std::printf("  FAIL bgrid dir=%d: %ld cells changed, expected %ld\n", dir, changed,
                  expect);
      ok = false;
    }
    std::printf("  bgrid dir=%d: %ld mismatches, %ld cells rotated\n", dir, bad, changed);
    if (bad != 0) ok = false;
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Part D -- fill_corners DGRID vector pair (mySign = -1 on SW/NE only)
// ---------------------------------------------------------------------------
static bool part_d() {
  const Box b = seam_box();
  A3 x, y;
  x.init(b); y.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        x(i, j, k) = fv_real(500.0 * double(k + 1) + 11.0 * double(i) + 0.25 * double(j));
        y(i, j, k) = fv_real(-700.0 * double(k + 1) - 13.0 * double(i) - 0.5 * double(j));
      }
  A3 rx = x, ry = y;
  for (int64_t k = 0; k < b.nk; ++k) ref_fill_dgrid(b, rx, ry, k);

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  Levels lev;
  lev.init(b.nk);
  fv_real *dx = d.up(x), *dy = d.up(y);
  const fv3_gpu_status st = dgrid::launch_fill_corners_dgrid_vector_cuda(
      tb_of(b), dl_of(d, b, lev, 0, 0), V3<fv_real>(dx, b), V3<fv_real>(dy, b), s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL dgrid status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 gx = down3(dx, b), gy = down3(dy, b);
  cudaStreamDestroy(s);

  long bad = 0, changed = 0, negatives = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        if (double(rx(i, j, k)) != double(x(i, j, k))) ++changed;
        if (double(ry(i, j, k)) != double(y(i, j, k))) ++changed;
        if (!close_rel(rx(i, j, k), gx(i, j, k), TOL)) {
          if (bad < 6)
            std::printf("  dgrid x mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k, double(rx(i, j, k)),
                        double(gx(i, j, k)));
          ++bad;
        }
        if (!close_rel(ry(i, j, k), gy(i, j, k), TOL)) {
          if (bad < 6)
            std::printf("  dgrid y mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k, double(ry(i, j, k)),
                        double(gy(i, j, k)));
          ++bad;
        }
      }
  // mySign liveness: the SW and NE blocks must produce sign-flipped copies.
  // With x > 0 and y < 0 everywhere in the seeded field, an SW x-write of
  // -1 * y is POSITIVE while an unsigned NW x-write of y is NEGATIVE. If the
  // sign were applied to the wrong corners this count would change.
  for (int64_t j = 1; j <= b.ng; ++j)
    for (int64_t i = 1; i <= b.ng; ++i)
      if (double(rx(1 - i, 1 - j, 0)) > 0.0) ++negatives;
  if (negatives != b.ng * b.ng) {
    std::printf("  FAIL dgrid: mySign not applied on SW (%ld of %lld)\n", negatives,
                (long long)(b.ng * b.ng));
    return false;
  }
  const long expect = 2 * 4 * b.ng * b.ng * b.nk;
  std::printf("  dgrid: %ld mismatches, %ld cells rotated (expect %ld)\n", bad, changed,
              expect);
  return bad == 0 && changed == expect;
}

// ---------------------------------------------------------------------------
// Part E -- branch H iterate, rows H1..H2g (sw_core.F90:1360-1405)
// ---------------------------------------------------------------------------
// Mixed column: nord_k = {0, 1, 2, 3}. Level 0 takes branch L and must be left
// completely untouched; levels 1..3 each run a different number of sweeps in
// the SAME launcher call, which is the whole reason DivergenceLevels exists.
//
// nord = 3 exactly fills the ng = 3 halo: at nt = 2, H2b reads
// divg_d(ie+2+nt) = divg_d(16) = divg_d(ied+1).
static void ref_branch_h(const Box &b, const Grid &g, int nord_cap,
                         const std::vector<fv_real> &nord_k, A3 &divg_d, A3 &uc, A3 &vc,
                         A3 &delpc, long *sweeps, long *fills) {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, npx = b.npx, npy = b.npy;
  const bool any_corner = (b.west && b.south) || (b.east && b.south) ||
                          (b.east && b.north) || (b.west && b.north);
  for (int nord = 1; nord <= nord_cap; ++nord) {
    for (int64_t k = 0; k < b.nk; ++k) {
      if (int(std::lround(double(nord_k[size_t(k)]))) != nord) continue;

      for (int64_t j = js; j <= je + 1; ++j)  // H1, :1364-1368
        for (int64_t i = is; i <= ie + 1; ++i) delpc(i, j, k) = divg_d(i, j, k);

      for (int n = 1; n <= nord; ++n) {
        const int64_t nt = nord - n;
        const bool fill_c = (nt != 0) && any_corner;
        if (sweeps) ++*sweeps;
        if (fill_c && fills) ++*fills;

        if (fill_c) ref_fill_bgrid(b, 1, divg_d, k);  // H2a
        for (int64_t j = js - nt; j <= je + 1 + nt; ++j)  // H2b
          for (int64_t i = is - 1 - nt; i <= ie + 1 + nt; ++i)
            vc(i, j, k) = (divg_d(i + 1, j, k) - divg_d(i, j, k)) * g.divg_u(i, j);
        if (fill_c) ref_fill_bgrid(b, 2, divg_d, k);  // H2c
        for (int64_t j = js - 1 - nt; j <= je + 1 + nt; ++j)  // H2d
          for (int64_t i = is - nt; i <= ie + 1 + nt; ++i)
            uc(i, j, k) = (divg_d(i, j + 1, k) - divg_d(i, j, k)) * g.divg_v(i, j);
        if (fill_c) ref_fill_dgrid(b, vc, uc, k);  // H2e -- x = vc, y = uc
        for (int64_t j = js - nt; j <= je + 1 + nt; ++j)  // H2f
          for (int64_t i = is - nt; i <= ie + 1 + nt; ++i)
            divg_d(i, j, k) =
                uc(i, j - 1, k) - uc(i, j, k) + vc(i - 1, j, k) - vc(i, j, k);
        // H2g, :1399-1403 -- runs on EVERY sweep, not only when fill_c
        if (b.west && b.south) divg_d(1, 1, k) = divg_d(1, 1, k) - uc(1, 0, k);
        if (b.east && b.south) divg_d(npx, 1, k) = divg_d(npx, 1, k) - uc(npx, 0, k);
        if (b.east && b.north)
          divg_d(npx, npy, k) = divg_d(npx, npy, k) + uc(npx, npy, k);
        if (b.west && b.north) divg_d(1, npy, k) = divg_d(1, npy, k) + uc(1, npy, k);
      }
    }
  }
}

static bool part_e() {
  Box b = seam_box();
  b.nk = 4;
  Grid g;
  g.init(b);
  Levels lev;
  lev.init(b.nk);
  const double nk_nord[4] = {0.0, 1.0, 2.0, 3.0};
  for (int64_t k = 0; k < b.nk; ++k) lev.nord_k[size_t(k)] = fv_real(nk_nord[k]);

  A3 divg0, uc0, vc0;
  divg0.init(b); uc0.init(b); vc0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        divg0(i, j, k) = fv_real(1.0e-5 * std::sin(0.29 * double(i) + 0.19 * double(j) +
                                                   0.4 * double(k)) +
                                 3.0e-6 * double(k + 1));
        uc0(i, j, k) = f_uc(i, j, k);
        vc0(i, j, k) = f_vc(i, j, k);
      }
  A3 divg_r = divg0, uc_r = uc0, vc_r = vc0, delpc_r;
  delpc_r.init(b, nan_poison());
  long sweeps = 0, fills = 0;
  ref_branch_h(b, g, 3, lev.nord_k, divg_r, uc_r, vc_r, delpc_r, &sweeps, &fills);
  // 1 + 2 + 3 = 6 sweeps; fill_c is true whenever nt != 0, i.e. 0 + 1 + 2 = 3.
  if (sweeps != 6 || fills != 3) {
    std::printf("  FAIL branch H: sweeps=%ld fills=%ld, expected 6 and 3\n", sweeps,
                fills);
    return false;
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::DivergenceGeometry dg = dg_of(d, b, g);
  dgrid::DivergenceDampingOptions opt{};
  opt.dddmp = fv_real(0.5);
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  // mode/nord_sel are deliberately set to values the launcher must OVERRIDE:
  // it builds its own per-nord group filter. If it honoured these instead,
  // levels 2 and 3 would be skipped and the comparison would fail.
  dgrid::DivergenceLevels lv = dl_of(d, b, lev, 1, 0);
  fv_real *d_divg = d.up(divg0), *d_uc = d.up(uc0), *d_vc = d.up(vc0);
  fv_real *d_delpc = d.scratch(b);
  if (!d_divg || !d_uc || !d_vc || !d_delpc) {
    cudaStreamDestroy(s);
    return false;
  }
  const fv3_gpu_status st = dgrid::launch_divdamp_high_order_cuda(
      tb_of(b), opt, lv, 3, dg, V3<fv_real>(d_divg, b), V3<fv_real>(d_uc, b),
      V3<fv_real>(d_vc, b), V3<fv_real>(d_delpc, b), s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL branch H: launch status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 divg_c = down3(d_divg, b), uc_c = down3(d_uc, b), vc_c = down3(d_vc, b);
  const A3 delpc_c = down3(d_delpc, b);
  cudaStreamDestroy(s);

  long bad = 0;
  double peak = 0.0;
  for (int64_t k = 0; k < b.nk; ++k) {
    // divg_d, uc and vc are compared over the FULL storage rectangle: the
    // fill_corners rotations write halo corner blocks and a missing rotation
    // must not hide in the halo.
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const double r[3] = {double(divg_r(i, j, k)), double(uc_r(i, j, k)),
                             double(vc_r(i, j, k))};
        const double c[3] = {double(divg_c(i, j, k)), double(uc_c(i, j, k)),
                             double(vc_c(i, j, k))};
        const char *nm[3] = {"divg_d", "uc", "vc"};
        for (int f = 0; f < 3; ++f) {
          if (!close_rel(r[f], c[f], TOL)) {
            if (bad < 8)
              std::printf("  %s mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n", nm[f],
                          (long long)i, (long long)j, (long long)k, r[f], c[f]);
            ++bad;
          }
          peak = std::fmax(peak, std::fabs(r[f] - c[f]) / std::fmax(std::fabs(r[f]), 1.0));
        }
      }
    if (k == 0) continue;  // delpc is undefined on the branch-L level
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        if (!close_rel(delpc_r(i, j, k), delpc_c(i, j, k), TOL)) {
          if (bad < 12)
            std::printf("  delpc mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k,
                        double(delpc_r(i, j, k)), double(delpc_c(i, j, k)));
          ++bad;
        }
  }
  // Level filter: nord_k = 0 must be bitwise untouched in all four arrays.
  long touched = 0;
  for (int64_t j = b.loj(); j <= b.hij(); ++j)
    for (int64_t i = b.loi(); i <= b.hii(); ++i) {
      if (double(divg_c(i, j, 0)) != double(divg0(i, j, 0))) ++touched;
      if (double(uc_c(i, j, 0)) != double(uc0(i, j, 0))) ++touched;
      if (double(vc_c(i, j, 0)) != double(vc0(i, j, 0))) ++touched;
      if (double(delpc_c(i, j, 0)) != 0.0) ++touched;
    }
  if (touched != 0) {
    std::printf("  FAIL branch H: %ld cells written on the nord_k = 0 level\n", touched);
    return false;
  }
  // Non-degeneracy: the iterate must actually change divg_d on the H levels.
  long moved = 0;
  for (int64_t k = 1; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        if (double(divg_c(i, j, k)) != double(divg0(i, j, k))) ++moved;
  const long total = 3 * (b.ie + 2 - b.is) * (b.je + 2 - b.js);
  if (moved < total / 2) {
    std::printf("  FAIL branch H: only %ld of %ld compute cells changed\n", moved, total);
    return false;
  }
  std::printf("  branch H: %ld mismatches, peak rel dev %.3g, %ld/%ld cells moved\n", bad,
              peak, moved, total);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part F -- H3 vort = 0 arm and H4 finalize (sw_core.F90:1408, :1413-1421)
// ---------------------------------------------------------------------------
static bool part_f() {
  Box b = seam_box();
  b.nk = 4;
  Grid g;
  g.init(b);
  Levels lev;
  lev.init(b.nk);
  const double nk_nord[4] = {0.0, 1.0, 2.0, 2.0};
  // d4_bg differs per level so dd8 = da_min*d4_bg**(nord_k+1) exercises three
  // distinct integer powers (2, 3, 3) with three distinct bases.
  const double d4v[4] = {0.11, 0.12, 0.13, 0.14};
  const double d2v[4] = {0.0, 0.0, 0.5, 0.05};
  for (int64_t k = 0; k < b.nk; ++k) {
    lev.nord_k[size_t(k)] = fv_real(nk_nord[k]);
    lev.d4_bg[size_t(k)] = fv_real(d4v[k]);
    lev.d2_bg[size_t(k)] = fv_real(d2v[k]);
  }
  const double dddmp = 3.0;

  A3 delpc, divg_d, vort0, ke0;
  delpc.init(b); divg_d.init(b); vort0.init(b); ke0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        delpc(i, j, k) = fv_real(2.0e-5 * std::sin(0.23 * double(i) - 0.31 * double(j)) +
                                 1.0e-6 * double(k));
        divg_d(i, j, k) = fv_real(1.4e-5 * std::cos(0.17 * double(i) + 0.27 * double(j)) -
                                  2.0e-6 * double(k));
        // vort arrives as smag_c, which is dt*sqrt(...) >= 0, so seed a
        // non-negative field. dddmp*vort straddles 0.20 across the box.
        vort0(i, j, k) = fv_real(std::fabs(0.09 * std::sin(0.37 * double(i) +
                                                           0.13 * double(j)) +
                                           0.02 * double(k)));
        ke0(i, j, k) = fv_real(0.3 * std::cos(0.19 * double(i) - 0.11 * double(j)));
      }

  // --- H3 reference: vort(:,:) = 0 over the DECLARED array, which is
  // vort(isd:ied, jsd:jed) (sw_core.F90:574), filtered to nord_k == 2.
  A3 vort_r = vort0, ke_r = ke0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (int(std::lround(double(lev.nord_k[size_t(k)]))) != 2) continue;
    for (int64_t j = b.js - b.ng; j <= b.je + b.ng; ++j)
      for (int64_t i = b.is - b.ng; i <= b.ie + b.ng; ++i) vort_r(i, j, k) = fv_real(0);
  }
  // --- H4 reference, per nord group in {1, 2}
  LimStats ls;
  for (int nord = 1; nord <= 2; ++nord)
    for (int64_t k = 0; k < b.nk; ++k) {
      if (int(std::lround(double(lev.nord_k[size_t(k)]))) != nord) continue;
      const int64_t n2 = nord + 1;
      fv_real p = lev.d4_bg[size_t(k)];
      for (int64_t e = 1; e < n2; ++e) p *= lev.d4_bg[size_t(k)];
      const fv_real dd8 = fv_real(DA_MIN) * p;
      for (int64_t j = b.js; j <= b.je + 1; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i) {
          const fv_real raw = fv_real(dddmp) * vort_r(i, j, k);  // NO abs() here
          const fv_real lim = raw < fv_real(0.20) ? raw : fv_real(0.20);
          const fv_real d2 = lev.d2_bg[size_t(k)];
          const fv_real m = d2 > lim ? d2 : lim;
          if (raw >= fv_real(0.20)) ++ls.clamp20;
          if (d2 > lim) ++ls.d2_arm; else ++ls.dddmp_arm;
          const fv_real damp2 = fv_real(DA_MIN_C) * m;
          const fv_real inc = damp2 * delpc(i, j, k) + dd8 * divg_d(i, j, k);
          vort_r(i, j, k) = inc;
          ke_r(i, j, k) = ke_r(i, j, k) + inc;
        }
    }
  if (ls.dddmp_arm == 0 || ls.d2_arm == 0) {
    std::printf("  FAIL H4: limiter arms not both live (dddmp=%ld d2=%ld)\n",
                ls.dddmp_arm, ls.d2_arm);
    return false;
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::DivergenceDampingOptions opt{};
  opt.dddmp = fv_real(dddmp);
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  fv_real *d_delpc = d.up(delpc), *d_divg = d.up(divg_d);
  fv_real *d_vort = d.up(vort0), *d_ke = d.up(ke0);
  if (!d_delpc || !d_divg || !d_vort || !d_ke) {
    cudaStreamDestroy(s);
    return false;
  }
  fv3_gpu_status st = dgrid::launch_divdamp_vort_zero_cuda(
      tb_of(b), dl_of(d, b, lev, 2, 2), V3<fv_real>(d_vort, b), s);
  if (st != FV3_GPU_SUCCESS) {
    std::printf("  FAIL vort_zero status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  for (int nord = 1; nord <= 2; ++nord) {
    st = dgrid::launch_divdamp_high_order_finalize_cuda(
        tb_of(b), opt, dl_of(d, b, lev, 2, int32_t(nord)),
        V3<const fv_real>(d_delpc, b), V3<const fv_real>(d_divg, b),
        V3<fv_real>(d_vort, b), V3<fv_real>(d_ke, b), s);
    if (st != FV3_GPU_SUCCESS) {
      std::printf("  FAIL finalize nord=%d status %d\n", nord, int(st));
      cudaStreamDestroy(s);
      return false;
    }
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) {
    cudaStreamDestroy(s);
    return false;
  }
  const A3 vort_c = down3(d_vort, b), ke_c = down3(d_ke, b);
  cudaStreamDestroy(s);

  long bad = 0;
  double peak = 0.0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        // vort is compared over the whole rectangle so the H3 zeroing of the
        // FULL isd..ied x jsd..jed array is checked, not just the compute box.
        const bool in_ke_box = (i >= b.is && i <= b.ie + 1 && j >= b.js && j <= b.je + 1);
        if (!close_rel(vort_r(i, j, k), vort_c(i, j, k), TOL)) {
          if (bad < 8)
            std::printf("  vort mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k,
                        double(vort_r(i, j, k)), double(vort_c(i, j, k)));
          ++bad;
        }
        if (in_ke_box && !close_rel(ke_r(i, j, k), ke_c(i, j, k), TOL)) {
          if (bad < 12)
            std::printf("  ke mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k, double(ke_r(i, j, k)),
                        double(ke_c(i, j, k)));
          ++bad;
        }
        peak = std::fmax(peak, std::fabs(double(vort_r(i, j, k) - vort_c(i, j, k))) /
                                   std::fmax(std::fabs(double(vort_r(i, j, k))), 1.0));
      }
  // Level 0 (nord_k = 0) must keep its seeded vort: neither H3 nor H4 owns it.
  long touched = 0;
  for (int64_t j = b.loj(); j <= b.hij(); ++j)
    for (int64_t i = b.loi(); i <= b.hii(); ++i)
      if (double(vort_c(i, j, 0)) != double(vort0(i, j, 0))) ++touched;
  if (touched != 0) {
    std::printf("  FAIL H3/H4: %ld cells written on the nord_k = 0 level\n", touched);
    return false;
  }
  std::printf("  H3/H4: %ld mismatches, peak rel dev %.3g (clamp=%ld dddmp=%ld d2=%ld)\n",
              bad, peak, ls.clamp20, ls.dddmp_arm, ls.d2_arm);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part G -- smag_corner (sw_core.F90:1857-1947)
// ---------------------------------------------------------------------------
// S1..S7 and S9 are transcribed here; the reference `sh` comes from a DIRECT
// call to WP-D's launch_a2b_ord4_global_cuda on the host-computed wk. See the
// file header: re-transcribing a2b_ord4 would duplicate numerical
// functionality that WP-D already owns and tests.
struct A2B {
  A2 dxa, dya;
  std::vector<fv_real> grid, agrid;
  std::vector<accumulator_real> ew;
  void init(const Box &b) {
    dxa.init(b); dya.init(b);
    grid.assign(size_t(2 * b.n2()), fv_real(0));
    agrid.assign(size_t(2 * b.n2()), fv_real(0));
    ew.assign(size_t(b.ni()), accumulator_real(0.25));
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const size_t o = size_t((i - b.loi()) + (j - b.loj()) * b.ni());
        dxa(i, j) = fv_real(1);
        dya(i, j) = fv_real(1);
        grid[o] = fv_real(0.1 * (double(i) - 0.5));
        grid[o + size_t(b.n2())] = fv_real(0.05 * (double(j) - 0.5));
        agrid[o] = fv_real(0.1 * double(i));
        agrid[o + size_t(b.n2())] = fv_real(0.05 * double(j));
      }
  }
};

static ops::A2BGeometry a2b_of(Dev &d, const Box &b, const A2B &a) {
  void *pg = d.raw(a.grid.size() * sizeof(fv_real));
  void *pa = d.raw(a.agrid.size() * sizeof(fv_real));
  void *pe = d.raw(a.ew.size() * sizeof(accumulator_real));
  cudaMemcpy(pg, a.grid.data(), a.grid.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
  cudaMemcpy(pa, a.agrid.data(), a.agrid.size() * sizeof(fv_real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pe, a.ew.data(), a.ew.size() * sizeof(accumulator_real),
             cudaMemcpyHostToDevice);
  View1<const accumulator_real> w{static_cast<const accumulator_real *>(pe), b.loi(), 1};
  ops::A2BGeometry o{V2<const fv_real>(d.up(a.dxa), b), V2<const fv_real>(d.up(a.dya), b),
                     VSG(pg, b), VSG(pa, b), w, w, w, w};
  return o;
}

static bool part_g() {
  Box b = seam_box();
  Grid g;
  g.init(b);
  A2B ab;
  ab.init(b);
  Levels lev;
  lev.init(b.nk);
  // nord_k = {0, 1, 2}, selector mode 2 / nord_sel 1 -> only k = 1 is active.
  // S1/S2/S5/S6/S7 and the a2b call are UNFILTERED by design (a2b has no level
  // filter and must not read a partially initialised wk); only S3/S4/S9 are.
  const double nkv[3] = {0.0, 1.0, 2.0};
  for (int64_t k = 0; k < b.nk; ++k) lev.nord_k[size_t(k)] = fv_real(nkv[k]);

  A3 u, v, smag0;
  u.init(b); v.init(b); smag0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        u(i, j, k) = f_u(i, j, k);
        v(i, j, k) = f_v(i, j, k);
        // smag_c is intent(out) in Fortran; seed it so the level filter's
        // "left alone" behaviour is observable.
        smag0(i, j, k) = fv_real(-7.25 - double(k));
      }

  // --- host S1..S7. ut/vt/wk start at 0 to match Dev's zeroed scratch, so the
  // rings no kernel writes compare equal instead of hiding a difference.
  A3 ut, vt, wk;
  ut.init(b); vt.init(b); wk.init(b);
  A3 tension;
  tension.init(b, nan_poison());
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je, ng = b.ng;
  const int64_t isd = is - ng, ied = ie + ng, jsd = js - ng, jed = je + ng;
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = js; j <= je + 1; ++j)          // S1
      for (int64_t i = is - 1; i <= ie + 1; ++i) ut(i, j, k) = u(i, j, k) * g.dyc(i, j);
    for (int64_t j = js - 1; j <= je + 1; ++j)      // S2
      for (int64_t i = is; i <= ie + 1; ++i) vt(i, j, k) = v(i, j, k) * g.dxc(i, j);
    for (int64_t j = js; j <= je + 1; ++j)          // S3 -- rarea_c INSIDE the sum
      for (int64_t i = is; i <= ie + 1; ++i)
        tension(i, j, k) = g.rarea_c(i, j) * (vt(i, j - 1, k) - vt(i, j, k) -
                                              ut(i - 1, j, k) + ut(i, j, k));
    // S4 -- added UNSCALED, after rarea_c. Authoritative asymmetry.
    const int64_t npx = b.npx, npy = b.npy;
    if (b.west && b.south) tension(1, 1, k) = tension(1, 1, k) - vt(1, 0, k);
    if (b.east && b.south) tension(npx, 1, k) = tension(npx, 1, k) - vt(npx, 0, k);
    if (b.east && b.north)
      tension(npx, npy, k) = tension(npx, npy, k) + vt(npx, npy, k);
    if (b.west && b.north) tension(1, npy, k) = tension(1, npy, k) + vt(1, npy, k);

    for (int64_t j = jsd; j <= jed + 1; ++j)        // S5 -- OVERWRITES vt
      for (int64_t i = isd; i <= ied; ++i) vt(i, j, k) = u(i, j, k) * g.dx(i, j);
    for (int64_t j = jsd; j <= jed; ++j)            // S6 -- OVERWRITES ut
      for (int64_t i = isd; i <= ied + 1; ++i) ut(i, j, k) = v(i, j, k) * g.dy(i, j);
    for (int64_t j = jsd; j <= jed; ++j)            // S7
      for (int64_t i = isd; i <= ied; ++i)
        wk(i, j, k) = g.rarea(i, j) * (vt(i, j, k) - vt(i, j + 1, k) + ut(i, j, k) -
                                       ut(i + 1, j, k));
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  const ops::HorizontalBounds hb{is, ie, js, je, b.nk};
  const ops::PhysicalEdges pe{b.west != 0, b.east != 0, b.south != 0, b.north != 0};
  const ops::PhysicalCorners pc{pe.west && pe.south, pe.east && pe.south,
                                pe.east && pe.north, pe.west && pe.north};
  ops::A2BGeometry ag = a2b_of(d, b, ab);

  // --- reference sh: a2b_ord4 on the HOST-computed wk, independent buffers
  fv_real *r_wk = d.up(wk), *r_sh = d.scratch(b);
  fv_real *r_qx = d.scratch(b), *r_qy = d.scratch(b);
  fv_real *r_qxx = d.scratch(b), *r_qyy = d.scratch(b);
  if (!r_wk || !r_sh || !r_qyy) {
    cudaStreamDestroy(s);
    return false;
  }
  fv3_gpu_status st = ops::launch_a2b_ord4_global_cuda(
      hb, pe, pc, b.nk, V3<fv_real>(r_wk, b), V3<fv_real>(r_sh, b), ag,
      V3<fv_real>(r_qx, b), V3<fv_real>(r_qy, b), V3<fv_real>(r_qxx, b),
      V3<fv_real>(r_qyy, b), false, s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL smag: reference a2b status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 sh_r = down3(r_sh, b);

  // --- host S9 on the active levels only
  A3 smag_r = smag0;
  long combined = 0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (int(std::lround(double(lev.nord_k[size_t(k)]))) != 1) continue;
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        const fv_real sv = sh_r(i, j, k), tv = tension(i, j, k);
        smag_r(i, j, k) = fv_real(DT) * fv_real(std::sqrt(double(sv) * double(sv) +
                                                          double(tv) * double(tv)));
        ++combined;
      }
  }
  if (combined == 0) {
    std::printf("  FAIL smag: level filter selected nothing\n");
    cudaStreamDestroy(s);
    return false;
  }

  // --- CUDA
  dgrid::DivergenceDampingOptions opt{};
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  dgrid::DivergenceGeometry dg = dg_of(d, b, g);
  dgrid::SmagCornerWorkspace ws{};
  fv_real *w_ut = d.scratch(b), *w_vt = d.scratch(b), *w_wk = d.scratch(b);
  fv_real *w_sh = d.scratch(b), *w_qx = d.scratch(b), *w_qy = d.scratch(b);
  fv_real *w_qxx = d.scratch(b), *w_qyy = d.scratch(b);
  fv_real *d_smag = d.up(smag0), *d_u = d.up(u), *d_v = d.up(v);
  if (!w_ut || !w_qyy || !d_smag) {
    cudaStreamDestroy(s);
    return false;
  }
  ws.ut = V3<fv_real>(w_ut, b); ws.vt = V3<fv_real>(w_vt, b);
  ws.wk = V3<fv_real>(w_wk, b); ws.sh = V3<fv_real>(w_sh, b);
  ws.qx = V3<fv_real>(w_qx, b); ws.qy = V3<fv_real>(w_qy, b);
  ws.qxx = V3<fv_real>(w_qxx, b); ws.qyy = V3<fv_real>(w_qyy, b);

  // A null stream must be rejected: WP-D's a2b launcher rejects one, and
  // failing only at that inner call would leave ut/vt/smag_c half-written.
  if (dgrid::launch_smag_corner_cuda(tb_of(b), opt, dl_of(d, b, lev, 2, 1), fv_real(DT),
                                     V3<const fv_real>(d_u, b), V3<const fv_real>(d_v, b),
                                     dg, ag, ws, V3<fv_real>(d_smag, b),
                                     nullptr) == FV3_GPU_SUCCESS) {
    std::printf("  FAIL smag: null stream accepted\n");
    cudaStreamDestroy(s);
    return false;
  }

  st = dgrid::launch_smag_corner_cuda(tb_of(b), opt, dl_of(d, b, lev, 2, 1), fv_real(DT),
                                      V3<const fv_real>(d_u, b),
                                      V3<const fv_real>(d_v, b), dg, ag, ws,
                                      V3<fv_real>(d_smag, b), s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL smag: launch status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 smag_c = down3(d_smag, b), wk_c = down3(w_wk, b);
  const A3 ut_c = down3(w_ut, b), vt_c = down3(w_vt, b);
  cudaStreamDestroy(s);

  long bad = 0;
  double peak = 0.0;
  for (int64_t k = 0; k < b.nk; ++k) {
    // wk / ut / vt carry the SHEAR values (S5..S7 overwrote the tension ones),
    // so the whole S5..S7 chain is checked, not only its effect through a2b.
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        if (!close_rel(wk(i, j, k), wk_c(i, j, k), TOL)) {
          if (bad < 4)
            std::printf("  wk mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k, double(wk(i, j, k)),
                        double(wk_c(i, j, k)));
          ++bad;
        }
    for (int64_t j = jsd; j <= jed + 1; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        if (!close_rel(vt(i, j, k), vt_c(i, j, k), TOL)) { ++bad; }
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = isd; i <= ied + 1; ++i)
        if (!close_rel(ut(i, j, k), ut_c(i, j, k), TOL)) { ++bad; }
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (!close_rel(smag_r(i, j, k), smag_c(i, j, k), TOL)) {
          if (bad < 8)
            std::printf("  smag_c mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k,
                        double(smag_r(i, j, k)), double(smag_c(i, j, k)));
          ++bad;
        }
        peak = std::fmax(peak, std::fabs(double(smag_r(i, j, k) - smag_c(i, j, k))) /
                                   std::fmax(std::fabs(double(smag_r(i, j, k))), 1.0));
      }
  }
  // Filtered-out levels keep their seed bitwise.
  long touched = 0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (int(std::lround(double(lev.nord_k[size_t(k)]))) == 1) continue;
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i)
        if (double(smag_c(i, j, k)) != double(smag0(i, j, k))) ++touched;
  }
  if (touched != 0) {
    std::printf("  FAIL smag: %ld cells written on filtered-out levels\n", touched);
    return false;
  }
  // Non-degeneracy: sh must not be identically zero, or S9 would collapse to
  // dt*|tension| and the a2b path would be untested.
  double sh_max = 0.0;
  for (int64_t j = js; j <= je + 1; ++j)
    for (int64_t i = is; i <= ie + 1; ++i)
      sh_max = std::fmax(sh_max, std::fabs(double(sh_r(i, j, 1))));
  if (!(sh_max > 1.0e-12)) {
    std::printf("  FAIL smag: reference sh is identically zero (%.3g)\n", sh_max);
    return false;
  }
  std::printf("  smag_corner: %ld mismatches, peak rel dev %.3g, max|sh| = %.3g\n", bad,
              peak, sh_max);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part H -- the d_con chain D1/D2/D3 (sw_core.F90:1425-1520)
// ---------------------------------------------------------------------------
static bool part_h() {
  Box b = seam_box();
  Grid g;
  g.init(b);
  Levels lev;
  lev.init(b.nk);
  // mode 3 selects levels whose d_con exceeds 1e-5, which is d_sw's own gate.
  const double dcv[3] = {0.0, 0.02, 0.05};
  for (int64_t k = 0; k < b.nk; ++k) lev.d_con[size_t(k)] = fv_real(dcv[k]);

  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  A3 vort, u, v, ut, vt, delp, heat0;
  vort.init(b); u.init(b); v.init(b); ut.init(b); vt.init(b);
  delp.init(b); heat0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const double x = double(i), y = double(j);
        vort(i, j, k) = fv_real(4.0e-5 * std::sin(0.31 * x - 0.21 * y) + 1e-6 * double(k));
        u(i, j, k) = f_u(i, j, k);
        v(i, j, k) = f_v(i, j, k);
        // ut/vt here are the del6 vorticity-damping fluxes, not winds.
        ut(i, j, k) = fv_real(0.7 * std::sin(0.19 * x + 0.29 * y) + 0.1 * double(k));
        vt(i, j, k) = fv_real(-0.6 * std::cos(0.23 * x - 0.13 * y) + 0.1 * double(k));
        delp(i, j, k) = fv_real(1500.0 + 40.0 * std::sin(0.1 * x + 0.2 * y) +
                                10.0 * double(k));
        // heat_source arrives holding the w-damping contribution: D3 is an
        // accumulate-then-scale, not an overwrite.
        heat0(i, j, k) = fv_real(0.35 * std::cos(0.17 * x + 0.13 * y) + 0.05 * double(k));
      }

  A3 ub, vb, fx, fy, gx, gy, heat_r;
  ub.init(b, nan_poison()); vb.init(b, nan_poison());
  fx.init(b, nan_poison()); fy.init(b, nan_poison());
  gx.init(b, nan_poison()); gy.init(b, nan_poison());
  heat_r = heat0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (!(double(lev.d_con[size_t(k)]) > 1.0e-5)) continue;
    for (int64_t j = js; j <= je + 1; ++j)  // D1, :1425-1436
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (i <= ie) ub(i, j, k) = vort(i, j, k) - vort(i + 1, j, k);
        if (j <= je) vb(i, j, k) = vort(i, j, k) - vort(i, j + 1, k);
      }
    for (int64_t j = js; j <= je + 1; ++j)  // D2, :1482-1495
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (i <= ie) {
          const fv_real rdx = g.rdx(i, j);
          const fv_real ubv = (ub(i, j, k) + vt(i, j, k)) * rdx;
          const fv_real fyv = u(i, j, k) * rdx;
          ub(i, j, k) = ubv;
          fy(i, j, k) = fyv;
          gy(i, j, k) = fyv * ubv;
        }
        if (j <= je) {
          const fv_real rdy = g.rdy(i, j);
          const fv_real vbv = (vb(i, j, k) - ut(i, j, k)) * rdy;
          const fv_real fxv = v(i, j, k) * rdy;
          vb(i, j, k) = vbv;
          fx(i, j, k) = fxv;
          gx(i, j, k) = fxv * vbv;
        }
      }
    const fv_real damp = fv_real(0.25) * lev.d_con[size_t(k)];
    for (int64_t j = js; j <= je; ++j)  // D3, :1497-1520
      for (int64_t i = is; i <= ie; ++i) {
        const fv_real u2 = fy(i, j, k) + fy(i, j + 1, k);
        const fv_real du2 = ub(i, j, k) + ub(i, j + 1, k);
        const fv_real v2 = fx(i, j, k) + fx(i + 1, j, k);
        const fv_real dv2 = vb(i, j, k) + vb(i + 1, j, k);
        const fv_real ubc = ub(i, j, k), ubn = ub(i, j + 1, k);
        const fv_real vbc = vb(i, j, k), vbe = vb(i + 1, j, k);
        heat_r(i, j, k) =
            delp(i, j, k) *
            (heat_r(i, j, k) -
             damp * g.rsin2(i, j) *
                 ((ubc * ubc + ubn * ubn + vbc * vbc + vbe * vbe) +
                  fv_real(2) * (gy(i, j, k) + gy(i, j + 1, k) + gx(i, j, k) +
                                gx(i + 1, j, k)) -
                  g.cosa_s(i, j) * (u2 * dv2 + v2 * du2 + du2 * dv2)));
      }
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::DivergenceGeometry dg = dg_of(d, b, g);
  dgrid::DivergenceDampingOptions opt{};
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  dgrid::DivergenceLevels lv = dl_of(d, b, lev, 3, 0);
  fv_real *d_vort = d.up(vort), *d_u = d.up(u), *d_v = d.up(v);
  fv_real *d_ut = d.up(ut), *d_vt = d.up(vt), *d_delp = d.up(delp);
  fv_real *d_heat = d.up(heat0);
  fv_real *d_ub = d.scratch(b), *d_vb = d.scratch(b);
  fv_real *d_fx = d.scratch(b), *d_fy = d.scratch(b);
  fv_real *d_gx = d.scratch(b), *d_gy = d.scratch(b);
  if (!d_vort || !d_heat || !d_gy) {
    cudaStreamDestroy(s);
    return false;
  }
  const dgrid::TransportBounds tb = tb_of(b);
  fv3_gpu_status st = dgrid::launch_dcon_vort_diff_cuda(
      tb, lv, V3<const fv_real>(d_vort, b), V3<fv_real>(d_ub, b), V3<fv_real>(d_vb, b), s);
  if (st == FV3_GPU_SUCCESS)
    st = dgrid::launch_dcon_scale_cuda(
        tb, lv, dg, V3<const fv_real>(d_u, b), V3<const fv_real>(d_v, b),
        V3<const fv_real>(d_ut, b), V3<const fv_real>(d_vt, b), V3<fv_real>(d_ub, b),
        V3<fv_real>(d_vb, b), V3<fv_real>(d_fx, b), V3<fv_real>(d_fy, b),
        V3<fv_real>(d_gx, b), V3<fv_real>(d_gy, b), s);
  if (st == FV3_GPU_SUCCESS)
    st = dgrid::launch_dcon_heat_source_cuda(
        tb, opt, lv, dg, V3<const fv_real>(d_delp, b), V3<const fv_real>(d_ub, b),
        V3<const fv_real>(d_vb, b), V3<const fv_real>(d_fx, b),
        V3<const fv_real>(d_fy, b), V3<const fv_real>(d_gx, b),
        V3<const fv_real>(d_gy, b), V3<fv_real>(d_heat, b), s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL d_con: status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 ub_c = down3(d_ub, b), vb_c = down3(d_vb, b);
  const A3 gx_c = down3(d_gx, b), gy_c = down3(d_gy, b);
  const A3 heat_c = down3(d_heat, b);
  cudaStreamDestroy(s);

  long bad = 0;
  double peak = 0.0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (!(double(lev.d_con[size_t(k)]) > 1.0e-5)) continue;
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (i <= ie) {
          if (!close_rel(ub(i, j, k), ub_c(i, j, k), TOL)) ++bad;
          if (!close_rel(gy(i, j, k), gy_c(i, j, k), TOL)) ++bad;
        }
        if (j <= je) {
          if (!close_rel(vb(i, j, k), vb_c(i, j, k), TOL)) ++bad;
          if (!close_rel(gx(i, j, k), gx_c(i, j, k), TOL)) ++bad;
        }
      }
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i) {
        if (!close_rel(heat_r(i, j, k), heat_c(i, j, k), TOL)) {
          if (bad < 8)
            std::printf("  heat_source mismatch (%lld,%lld,%lld) ref=%.17g cuda=%.17g\n",
                        (long long)i, (long long)j, (long long)k,
                        double(heat_r(i, j, k)), double(heat_c(i, j, k)));
          ++bad;
        }
        peak = std::fmax(peak, std::fabs(double(heat_r(i, j, k) - heat_c(i, j, k))) /
                                   std::fmax(std::fabs(double(heat_r(i, j, k))), 1.0));
      }
  }
  long touched = 0;
  for (int64_t j = js; j <= je; ++j)
    for (int64_t i = is; i <= ie; ++i)
      if (double(heat_c(i, j, 0)) != double(heat0(i, j, 0))) ++touched;
  if (touched != 0) {
    std::printf("  FAIL d_con: %ld cells written on the d_con = 0 level\n", touched);
    return false;
  }
  // heat_source must actually move where d_con > 0, otherwise the whole chain
  // could be a no-op and still "agree".
  long moved = 0;
  for (int64_t k = 1; k < b.nk; ++k)
    for (int64_t j = js; j <= je; ++j)
      for (int64_t i = is; i <= ie; ++i)
        if (double(heat_c(i, j, k)) != double(heat0(i, j, k))) ++moved;
  if (moved < 2 * (ie - is + 1) * (je - js + 1)) {
    std::printf("  FAIL d_con: heat_source barely changed (%ld cells)\n", moved);
    return false;
  }
  std::printf("  d_con chain: %ld mismatches, peak rel dev %.3g, %ld cells moved\n", bad,
              peak, moved);
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part I -- V2 damp_v wind update (sw_core.F90:1523-1534)
// ---------------------------------------------------------------------------
// The point of this part is LevelFilter::gate. d_sw gates this update on
// damp_v > 1.E-5, but the coefficient it hands del6_vt_flux is
// damp4 = (damp_v*da_min_c)**(nord_v+1) -- a different number that is large
// even where damp_v is zero. If the filter tested damp instead of gate, level 0
// below would be updated and this part fails.
static bool part_i() {
  Box b = seam_box();
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  A3 ut, vt, u0, v0;
  ut.init(b); vt.init(b); u0.init(b); v0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        ut(i, j, k) = fv_real(0.31 * std::sin(0.2 * double(i) + 0.3 * double(j)) +
                              0.05 * double(k));
        vt(i, j, k) = fv_real(-0.27 * std::cos(0.25 * double(i) - 0.15 * double(j)) +
                              0.04 * double(k));
        u0(i, j, k) = f_u(i, j, k);
        v0(i, j, k) = f_v(i, j, k);
      }
  std::vector<fv_real> gate(size_t(b.nk)), damp(size_t(b.nk)), ndif(size_t(b.nk));
  const double gv[3] = {0.0, 0.5, 0.5};
  for (int64_t k = 0; k < b.nk; ++k) {
    gate[size_t(k)] = fv_real(gv[k]);
    damp[size_t(k)] = fv_real(1.0e9);  // damp4: huge on EVERY level, including 0
    ndif[size_t(k)] = fv_real(2);
  }
  A3 u_r = u0, v_r = v0;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (!(double(gate[size_t(k)]) > 1.0e-5)) continue;
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (i <= ie) u_r(i, j, k) = u_r(i, j, k) + vt(i, j, k);
        if (j <= je) v_r(i, j, k) = v_r(i, j, k) - ut(i, j, k);
      }
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::LevelFilter lf{};
  lf.damp = VL(d.lev(damp));
  lf.ndif = VL(d.lev(ndif));
  lf.gate = VL(d.lev(gate));
  lf.npz = b.nk;
  lf.mode = 1;
  fv_real *d_ut = d.up(ut), *d_vt = d.up(vt), *d_u = d.up(u0), *d_v = d.up(v0);
  if (!d_ut || !d_v) {
    cudaStreamDestroy(s);
    return false;
  }
  const fv3_gpu_status st = dgrid::launch_dwind_damping_add_cuda(
      tb_of(b), lf, V3<const fv_real>(d_ut, b), V3<const fv_real>(d_vt, b),
      V3<fv_real>(d_u, b), V3<fv_real>(d_v, b), s);
  if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL damp_v add: status %d\n", int(st));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 u_c = down3(d_u, b), v_c = down3(d_v, b);
  cudaStreamDestroy(s);

  long bad = 0, touched = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = js; j <= je + 1; ++j)
      for (int64_t i = is; i <= ie + 1; ++i) {
        if (i <= ie && !close_rel(u_r(i, j, k), u_c(i, j, k), TOL)) ++bad;
        if (j <= je && !close_rel(v_r(i, j, k), v_c(i, j, k), TOL)) ++bad;
        if (k == 0) {
          if (i <= ie && double(u_c(i, j, 0)) != double(u0(i, j, 0))) ++touched;
          if (j <= je && double(v_c(i, j, 0)) != double(v0(i, j, 0))) ++touched;
        }
      }
  if (touched != 0) {
    std::printf("  FAIL damp_v add: %ld cells updated where damp_v == 0 -- the filter "
                "is testing damp4, not the gate\n",
                touched);
    return false;
  }
  std::printf("  damp_v add: %ld mismatches, gate honoured on %lld cells\n", bad,
              (long long)((ie - is + 1) * (je - js + 2)));
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part J -- V1 adapter: BITWISE identity with launch_del6_vt_flux_cuda
// ---------------------------------------------------------------------------
static bool part_j() {
  Box b = seam_box();
  Grid g;
  g.init(b);
  A3 wk0, vort0, ut0, vt0;
  wk0.init(b); vort0.init(b); ut0.init(b); vt0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        wk0(i, j, k) = fv_real(3.0e-5 * std::sin(0.27 * double(i) - 0.18 * double(j)) +
                               1e-6 * double(k));
        vort0(i, j, k) = fv_real(0.5);
        ut0(i, j, k) = fv_real(0.25);
        vt0(i, j, k) = fv_real(-0.25);
      }
  std::vector<fv_real> gate(size_t(b.nk)), damp(size_t(b.nk)), ndif(size_t(b.nk));
  for (int64_t k = 0; k < b.nk; ++k) {
    gate[size_t(k)] = fv_real(k == 0 ? 0.0 : 0.5);
    damp[size_t(k)] = fv_real(1.0e6 * double(k + 1));
    ndif[size_t(k)] = fv_real(2);
  }

  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::LevelFilter lf{};
  lf.damp = VL(d.lev(damp));
  lf.ndif = VL(d.lev(ndif));
  lf.gate = VL(d.lev(gate));
  lf.npz = b.nk;
  lf.mode = 3;
  lf.nord_sel = 2;
  View2<const fv_real> d6u = V2<const fv_real>(d.up(g.del6_u), b);
  View2<const fv_real> d6v = V2<const fv_real>(d.up(g.del6_v), b);
  View2<const fv_real> ra = V2<const fv_real>(d.up(g.rarea), b);
  const dgrid::TransportBounds tb = tb_of(b);

  fv_real *a_wk = d.up(wk0), *a_vort = d.up(vort0), *a_ut = d.up(ut0), *a_vt = d.up(vt0);
  fv_real *b_wk = d.up(wk0), *b_vort = d.up(vort0), *b_ut = d.up(ut0), *b_vt = d.up(vt0);
  if (!a_wk || !b_vt) {
    cudaStreamDestroy(s);
    return false;
  }
  fv3_gpu_status sa = dgrid::launch_vorticity_damping_cuda(
      tb, 2, lf, V3<fv_real>(a_wk, b), d6u, d6v, ra, V3<fv_real>(a_vort, b),
      V3<fv_real>(a_ut, b), V3<fv_real>(a_vt, b), s);
  fv3_gpu_status sb = dgrid::launch_del6_vt_flux_cuda(
      tb, 2, lf, V3<fv_real>(b_wk, b), d6u, d6v, ra, V3<fv_real>(b_vort, b),
      V3<fv_real>(b_ut, b), V3<fv_real>(b_vt, b), s);
  if (sa != FV3_GPU_SUCCESS || sb != FV3_GPU_SUCCESS ||
      cudaStreamSynchronize(s) != cudaSuccess) {
    std::printf("  FAIL adapter: status %d / %d\n", int(sa), int(sb));
    cudaStreamDestroy(s);
    return false;
  }
  const A3 aw = down3(a_wk, b), av = down3(a_vort, b);
  const A3 au = down3(a_ut, b), at = down3(a_vt, b);
  const A3 bw = down3(b_wk, b), bv = down3(b_vort, b);
  const A3 bu = down3(b_ut, b), bt = down3(b_vt, b);

  long diff = 0, moved = 0;
  for (size_t n = 0; n < aw.d.size(); ++n) {
    if (double(aw.d[n]) != double(bw.d[n])) ++diff;
    if (double(av.d[n]) != double(bv.d[n])) ++diff;
    if (double(au.d[n]) != double(bu.d[n])) ++diff;
    if (double(at.d[n]) != double(bt.d[n])) ++diff;
    // Non-degeneracy is asserted on the OUTPUTS. wk is del6_vt_flux's `q`
    // argument -- the relative vorticity it reads (sw_core.F90:1555, "rel.
    // vorticity ghosted on input"); the routine's products are d2 = vort and
    // the fluxes fx2 = ut, fy2 = vt.
    if (double(av.d[n]) != double(vort0.d[n])) ++moved;
    if (double(au.d[n]) != double(ut0.d[n])) ++moved;
    if (double(at.d[n]) != double(vt0.d[n])) ++moved;
  }
  if (diff != 0) {
    std::printf("  FAIL adapter: %ld words differ from a direct del6_vt_flux call\n",
                diff);
    cudaStreamDestroy(s);
    return false;
  }
  if (moved == 0) {
    std::printf("  FAIL adapter: del6_vt_flux produced nothing, identity is vacuous\n");
    cudaStreamDestroy(s);
    return false;
  }

  // The argument mistakes the adapter exists to prevent must be rejected.
  struct Bad { const char *why; dgrid::LevelFilter f; int32_t cap; };
  dgrid::LevelFilter no_gate = lf; no_gate.gate = View1<const fv_real>{};
  dgrid::LevelFilter wrong_mode = lf; wrong_mode.mode = 1;
  const Bad cases[] = {{"mode != 3", wrong_mode, 2},
                       {"gate absent", no_gate, 2},
                       {"nord_cap < 0", lf, -1},
                       {"nord_cap > ng-1", lf, 3}};
  for (const Bad &c : cases) {
    const fv3_gpu_status r = dgrid::launch_vorticity_damping_cuda(
        tb, c.cap, c.f, V3<fv_real>(a_wk, b), d6u, d6v, ra, V3<fv_real>(a_vort, b),
        V3<fv_real>(a_ut, b), V3<fv_real>(a_vt, b), s);
    if (r != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  FAIL adapter: %s accepted (status %d)\n", c.why, int(r));
      cudaStreamDestroy(s);
      return false;
    }
  }
  cudaStreamDestroy(s);
  std::printf("  adapter: bitwise identical over %zu words, %ld output words produced, 4 rejections honoured\n",
              aw.d.size(), moved);
  return true;
}

// ---------------------------------------------------------------------------
// Part K -- unsupported configurations must be rejected BEFORE any launch
// ---------------------------------------------------------------------------
// The project rule is that an unsupported configuration fails before mutating
// model state, so each case also checks the output buffer is bit-unchanged.
static bool part_k() {
  Box b = seam_box();
  Grid g;
  g.init(b);
  Levels lev;
  lev.init(b.nk);
  for (int64_t k = 0; k < b.nk; ++k) lev.d2_bg[size_t(k)] = fv_real(0.1);

  A3 zero;
  zero.init(b, fv_real(-3.5));
  Dev d;
  cudaStream_t s = nullptr;
  if (cudaStreamCreate(&s) != cudaSuccess) return false;
  dgrid::DivergenceGeometry dg = dg_of(d, b, g);
  dgrid::DivergenceLevels lv = dl_of(d, b, lev, 1, 0);
  fv_real *p[10];
  for (int n = 0; n < 10; ++n) {
    p[n] = d.up(zero);
    if (!p[n]) {
      cudaStreamDestroy(s);
      return false;
    }
  }
  dgrid::DivergenceDampingOptions ok{};
  ok.grid_type = 0;
  ok.do_skeb = 0;
  ok.dddmp = fv_real(0.5);
  ok.da_min = fv_real(DA_MIN);
  ok.da_min_c = fv_real(DA_MIN_C);

  auto call_l = [&](const dgrid::TransportBounds &t,
                    const dgrid::DivergenceDampingOptions &o,
                    const dgrid::DivergenceLevels &l) {
    return dgrid::launch_divdamp_nord0_cuda(
        t, o, l, fv_real(DT), V3<const fv_real>(p[0], b), V3<const fv_real>(p[1], b),
        V3<const fv_real>(p[2], b), V3<const fv_real>(p[3], b),
        V3<const fv_real>(p[4], b), V3<const fv_real>(p[5], b), dg,
        V3<fv_real>(p[6], b), V3<fv_real>(p[7], b), V3<fv_real>(p[8], b),
        V3<fv_real>(p[9], b), s);
  };

  struct Case { const char *why; dgrid::TransportBounds t;
                dgrid::DivergenceDampingOptions o; dgrid::DivergenceLevels l; };
  std::vector<Case> cases;
  {
    Case c{"grid_type != 0", tb_of(b), ok, lv};
    c.o.grid_type = 4;
    cases.push_back(c);
  }
  {
    Case c{"do_skeb != 0", tb_of(b), ok, lv};
    c.o.do_skeb = 1;
    cases.push_back(c);
  }
  {
    Case c{"nested", tb_of(b), ok, lv};
    c.t.nested = 1;
    cases.push_back(c);
  }
  {
    Case c{"not cubed_sphere", tb_of(b), ok, lv};
    c.t.cubed_sphere = 0;
    cases.push_back(c);
  }
  {
    Case c{"nk <= 0", tb_of(b), ok, lv};
    c.t.nk = 0;
    cases.push_back(c);
  }
  {
    Case c{"ng < 3", tb_of(b), ok, lv};
    c.t.ng = 2;
    cases.push_back(c);
  }
  {
    // west_edge asserted while is != 1: the flags and the index box disagree,
    // which is exactly the bug that silently mis-places a seam formula.
    Case c{"west_edge inconsistent with is", tb_of(b), ok, lv};
    c.t.west_edge = 0;
    cases.push_back(c);
  }
  {
    Case c{"sw_corner without its two edges", tb_of(b), ok, lv};
    c.t.sw_corner = 0;
    cases.push_back(c);
  }
  {
    Case c{"is > ie", tb_of(b), ok, lv};
    c.t.is = c.t.ie + 1;
    cases.push_back(c);
  }
  {
    Case c{"npz <= 0", tb_of(b), ok, lv};
    c.l.npz = 0;
    cases.push_back(c);
  }
  {
    Case c{"mode out of range", tb_of(b), ok, lv};
    c.l.mode = 4;
    cases.push_back(c);
  }
  {
    Case c{"d2_bg absent", tb_of(b), ok, lv};
    c.l.d2_bg = View1<const fv_real>{};
    cases.push_back(c);
  }
  {
    Case c{"nord_k absent with mode 1", tb_of(b), ok, lv};
    c.l.nord_k = View1<const fv_real>{};
    cases.push_back(c);
  }

  long bad = 0;
  for (const Case &c : cases) {
    const fv3_gpu_status r = call_l(c.t, c.o, c.l);
    if (r != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  FAIL reject: %s accepted (status %d)\n", c.why, int(r));
      ++bad;
    }
  }
  // A null output view must also be rejected.
  {
    const fv3_gpu_status r = dgrid::launch_divdamp_nord0_cuda(
        tb_of(b), ok, lv, fv_real(DT), V3<const fv_real>(p[0], b),
        V3<const fv_real>(p[1], b), V3<const fv_real>(p[2], b),
        V3<const fv_real>(p[3], b), V3<const fv_real>(p[4], b),
        V3<const fv_real>(p[5], b), dg, V3<fv_real>(nullptr, b), V3<fv_real>(p[7], b),
        V3<fv_real>(p[8], b), V3<fv_real>(p[9], b), s);
    if (r != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  FAIL reject: null ptc accepted (status %d)\n", int(r));
      ++bad;
    }
  }
  // nord_cap outside [1,3] for the high-order iterate. 0 is branch L's job and
  // 4 would read past the ng = 3 halo.
  for (int32_t cap : {int32_t(0), int32_t(4), int32_t(-1)}) {
    const fv3_gpu_status r = dgrid::launch_divdamp_high_order_cuda(
        tb_of(b), ok, lv, cap, dg, V3<fv_real>(p[0], b), V3<fv_real>(p[1], b),
        V3<fv_real>(p[2], b), V3<fv_real>(p[3], b), s);
    if (r != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  FAIL reject: nord_cap = %d accepted (status %d)\n", int(cap), int(r));
      ++bad;
    }
  }
  // The DGRID vector rotation aliases x and y into a single array if the caller
  // passes the same pointer twice; the two Fortran loops are not symmetric and
  // the result would be silently wrong.
  {
    const fv3_gpu_status r = dgrid::launch_fill_corners_dgrid_vector_cuda(
        tb_of(b), lv, V3<fv_real>(p[0], b), V3<fv_real>(p[0], b), s);
    if (r != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
      std::printf("  FAIL reject: aliased x == y accepted (status %d)\n", int(r));
      ++bad;
    }
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) {
    cudaStreamDestroy(s);
    return false;
  }
  cudaStreamDestroy(s);

  // Nothing may have been written by any rejected call.
  long written = 0;
  for (int n = 0; n < 10; ++n) {
    const A3 got = down3(p[n], b);
    for (size_t w = 0; w < got.d.size(); ++w)
      if (double(got.d[w]) != double(zero.d[w])) ++written;
  }
  if (written != 0) {
    std::printf("  FAIL reject: %ld words mutated by rejected calls\n", written);
    ++bad;
  }
  if (bad == 0)
    std::printf("  rejections: %zu + 5 cases rejected, 0 words mutated\n", cases.size());
  return bad == 0;
}

// ---------------------------------------------------------------------------
// Part L -- DivergenceLevels selector semantics, all four modes + the clamp
// ---------------------------------------------------------------------------
// Every launcher above takes a selector rather than a scalar nord because
// dyn_core.F90:697-760 recomputes nord_k inside the k loop with sponge
// overrides, so one production call really does see several nord values. This
// part pins the selector itself using vort_zero as a cheap observable.
static bool part_l() {
  Box b = seam_box();
  b.nk = 4;
  Levels lev;
  lev.init(b.nk);
  const double nkv[4] = {0.0, 1.0, 2.0, 0.0};
  const double dcv[4] = {0.0, 0.02, 0.0, 0.05};
  for (int64_t k = 0; k < b.nk; ++k) {
    lev.nord_k[size_t(k)] = fv_real(nkv[k]);
    lev.d_con[size_t(k)] = fv_real(dcv[k]);
  }
  struct Expect { const char *why; int32_t mode, sel; bool on[4]; };
  const Expect table[] = {
      {"mode 0 = every level", 0, 0, {true, true, true, true}},
      {"mode 1 = nord_k == 0", 1, 0, {true, false, false, true}},
      {"mode 2 = nord_k == 1", 2, 1, {false, true, false, false}},
      {"mode 2 = nord_k == 2", 2, 2, {false, false, true, false}},
      {"mode 3 = d_con > 1e-5", 3, 0, {false, true, false, true}},
  };

  A3 seed;
  seed.init(b, fv_real(9.75));
  long bad = 0;
  for (const Expect &e : table) {
    Dev d;
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return false;
    fv_real *dv = d.up(seed);
    if (!dv) {
      cudaStreamDestroy(s);
      return false;
    }
    const fv3_gpu_status st = dgrid::launch_divdamp_vort_zero_cuda(
        tb_of(b), dl_of(d, b, lev, e.mode, e.sel), V3<fv_real>(dv, b), s);
    if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
      std::printf("  FAIL selector: %s status %d\n", e.why, int(st));
      cudaStreamDestroy(s);
      return false;
    }
    const A3 got = down3(dv, b);
    cudaStreamDestroy(s);
    for (int64_t k = 0; k < b.nk; ++k) {
      const bool zeroed = double(got(b.is, b.js, k)) == 0.0;
      if (zeroed != e.on[k]) {
        std::printf("  FAIL selector: %s -- level %lld %s\n", e.why, (long long)k,
                    zeroed ? "active but should not be" : "inactive but should be");
        ++bad;
      }
    }
  }
  // The npz clamp: dyn_core passes damp/nord arrays of length npz but d_sw can
  // be invoked with nk == npz + 1 interface levels, so index npz must clamp to
  // npz - 1 rather than read out of bounds.
  {
    Levels shortl;
    shortl.init(2);
    shortl.nord_k[0] = fv_real(0);
    shortl.nord_k[1] = fv_real(2);
    Box c = b;
    c.nk = 3;
    Dev d;
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return false;
    A3 sd;
    sd.init(c, fv_real(4.25));
    fv_real *dv = d.up(sd);
    dgrid::DivergenceLevels lv{};
    lv.nord_k = VL(d.lev(shortl.nord_k));
    lv.d2_bg = VL(d.lev(shortl.d2_bg));
    lv.d4_bg = VL(d.lev(shortl.d4_bg));
    lv.d_con = VL(d.lev(shortl.d_con));
    lv.npz = 2;
    lv.mode = 2;
    lv.nord_sel = 2;
    const fv3_gpu_status st = dgrid::launch_divdamp_vort_zero_cuda(
        tb_of(c), lv, V3<fv_real>(dv, c), s);
    if (st != FV3_GPU_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
      std::printf("  FAIL clamp: status %d\n", int(st));
      cudaStreamDestroy(s);
      return false;
    }
    const A3 got = down3(dv, c);
    cudaStreamDestroy(s);
    const bool on[3] = {false, true, true};  // level 2 clamps to nord_k(1) == 2
    for (int64_t k = 0; k < 3; ++k)
      if ((double(got(c.is, c.js, k)) == 0.0) != on[k]) {
        std::printf("  FAIL clamp: level %lld misclassified\n", (long long)k);
        ++bad;
      }
  }
  if (bad == 0) std::printf("  selector: 5 modes + the npz clamp all correct\n");
  return bad == 0;
}

// ---------------------------------------------------------------------------
int main() {
  int devices = 0;
  const cudaError_t probe = cudaGetDeviceCount(&devices);
  if (probe != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA device available (%s)\n", cudaGetErrorString(probe));
    return 77;
  }
  std::printf("WP-B.4 divergence damping / corner slice, fv_real = %zu bytes, tol %.3g\n",
              sizeof(fv_real), TOL);

  struct Part { const char *name; bool (*fn)(); };
  const Part parts[] = {
      {"A branch L, seam fixture", part_a},
      {"B branch L, interior fixture", part_b},
      {"C fill_corners BGRID scalar", part_c},
      {"D fill_corners DGRID vector", part_d},
      {"E branch H iterate", part_e},
      {"F vort zero + high-order finalize", part_f},
      {"G smag_corner via WP-D a2b_ord4", part_g},
      {"H d_con heat-source chain", part_h},
      {"I damp_v wind update", part_i},
      {"J vorticity-damping adapter", part_j},
      {"K unsupported-configuration rejection", part_k},
      {"L DivergenceLevels selector", part_l},
  };
  int failures = 0;
  for (const Part &p : parts) {
    std::printf("[%s]\n", p.name);
    if (!p.fn()) {
      std::printf("  -> FAIL\n");
      ++failures;
    }
  }
  const cudaError_t residual = cudaGetLastError();
  if (residual != cudaSuccess) {
    std::printf("residual CUDA error: %s\n", cudaGetErrorString(residual));
    ++failures;
  }
  if (failures != 0) {
    std::printf("WP-B.4: %d part(s) FAILED\n", failures);
    return 1;
  }
  std::printf("WP-B.4: all %zu parts PASS\n", sizeof(parts) / sizeof(parts[0]));
  return 0;
}
