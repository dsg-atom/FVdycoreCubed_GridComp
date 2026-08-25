// ===========================================================================
// Milestone 3A.8 / WP-B.5 -- authoritative d_sw driver behavior test.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45, model/sw_core.F90 subroutine
// d_sw (:600-1540).
//
// STRUCTURE
//
//   Parts A-H test the DRIVER-OWNED kernels (K1..K12 and the level-coefficient
//   kernel) in isolation against host references transcribed directly from the
//   Fortran, in the Fortran's own pass structure. Isolation is deliberate: a
//   driver-owned arithmetic error must show up as its own failure rather than as
//   an amplified difference thirty stages downstream.
//
//   Parts I-M test the COMPLETE launch_d_sw_cuda. The numerical services it
//   orchestrates (fv_tp_2d, xtp_u/ytp_v, deln_flux, del6_vt_flux,
//   divergence_corner, divg_d, smag_corner, the KE and vorticity helpers) each
//   already have an independent host transcription in the Milestone 3A.4/3A.5/
//   3A.6/3A.7 tests, so re-transcribing the whole 900-line routine here would
//   duplicate numerical functionality the project rules forbid. What these parts
//   establish instead is what only the driver can get wrong: ORDER, per-level
//   control flow, and anti-dependence.
//
//   The main lever is the per-level independence invariant. dyn_core calls d_sw
//   once per k with SCALAR damping controls; the CUDA driver batches all levels
//   into one call with per-level control arrays. Therefore a mixed per-level run
//   MUST agree, level by level, with the corresponding uniform-parameter runs --
//   and it can only do so if every level really took its own branch. Part I is
//   that comparison, which is also the §4 requirement that different k levels
//   take different damping branches inside one invocation. It is an independent
//   invariant, not a restatement of the driver's stage list.
//
//   Ordering is pinned three further ways, all of them falsifiable:
//     - Parts C and E evaluate deliberate MIS-orderings of the pt/delp/w updates
//       alongside the authoritative one and require them to DIFFER, so those
//       references have teeth rather than being satisfied by any ordering.
//     - Part K ties stages together end to end through quantities that SURVIVE
//       to the end of the call (delp vs dpx, dw vs the fx2/fy2 that nothing
//       after S10 touches, heat_source on a d_con-free level), so moving a stage
//       across another breaks an equality.
//     - Part L requires every unsupported-configuration fence to reject BEFORE
//       any state mutation.
//
//   There is deliberately NO explicit host replay of S1..S38 here. Such a replay
//   would have to be written from the same stage list the driver uses, so it
//   would be a tautology dressed as evidence, and it would duplicate ~200 lines
//   of orchestration in the test.
//
// The NaN poison in the Part A-H references is load-bearing: Fortran automatic
// locals are undefined on entry, so if a CUDA kernel writes a cell the Fortran
// pass structure leaves undefined the reference carries a NaN into the
// comparison and the part fails rather than silently agreeing.
//
// Fixtures:
//   seam     npx = npy = 13, is/js = 1, ie/je = 12 -- owns all four tile edges
//            and all four corners. Storage i,j in [-2, 16].
//   interior npx = npy = 25, is/js = 7, ie/je = 18 -- owns no edge, so every
//            formula takes the unclipped contravariant branch.
//
// Exit codes: 0 pass, 1 fail, 77 no CUDA device (CTest SKIP_RETURN_CODE).
// ===========================================================================

#include "dgrid/wp_b_dsw_cuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// upper bounds and the corner blocks' reads at npx+1 / npy+1.
struct Box {
  int64_t is = 1, ie = 12, js = 1, je = 12, ng = 3, nk = 4;
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
  int64_t isd() const { return is - ng; }
  int64_t ied() const { return ie + ng; }
  int64_t jsd() const { return js - ng; }
  int64_t jed() const { return je + ng; }
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

struct D3 {
  const Box *b = nullptr;
  std::vector<accumulator_real> d;
  void init(const Box &bx, accumulator_real fill = 0.0) {
    b = &bx;
    d.assign(static_cast<size_t>(bx.n3()), fill);
  }
  size_t off(int64_t i, int64_t j, int64_t k) const {
    return static_cast<size_t>((i - b->loi()) + (j - b->loj()) * b->ni() +
                               k * b->ni() * b->nj());
  }
  accumulator_real &operator()(int64_t i, int64_t j, int64_t k) { return d[off(i, j, k)]; }
  accumulator_real operator()(int64_t i, int64_t j, int64_t k) const {
    return d[off(i, j, k)];
  }
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
    // silent about halo cells no kernel writes. This is fixture setup, not an
    // arena wipe inside the code under test.
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
  int32_t *iscratch(const Box &b) {
    return static_cast<int32_t *>(raw(static_cast<size_t>(b.n3()) * sizeof(int32_t)));
  }
  fv_real *lev(const std::vector<fv_real> &v) {
    void *p = raw(v.size() * sizeof(fv_real));
    if (p) cudaMemcpy(p, v.data(), v.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
    return static_cast<fv_real *>(p);
  }
  fv_real *levn(size_t n) {
    return static_cast<fv_real *>(raw(n * sizeof(fv_real)));
  }
  accumulator_real *acc(const D3 &a) {
    void *p = raw(a.d.size() * sizeof(accumulator_real));
    if (p)
      cudaMemcpy(p, a.d.data(), a.d.size() * sizeof(accumulator_real),
                 cudaMemcpyHostToDevice);
    return static_cast<accumulator_real *>(p);
  }
  accumulator_real *accv(const std::vector<accumulator_real> &v) {
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
static View1<fv_real> VLW(void *p) {
  return View1<fv_real>{static_cast<fv_real *>(p), 0, 1};
}

static A3 down3(const void *p, const Box &b) {
  A3 a;
  a.init(b);
  cudaMemcpy(a.d.data(), p, a.d.size() * sizeof(fv_real), cudaMemcpyDeviceToHost);
  return a;
}
static D3 downd(const void *p, const Box &b) {
  D3 a;
  a.init(b);
  cudaMemcpy(a.d.data(), p, a.d.size() * sizeof(accumulator_real),
             cudaMemcpyDeviceToHost);
  return a;
}
static std::vector<fv_real> downl(const void *p, size_t n) {
  std::vector<fv_real> v(n, fv_real(0));
  cudaMemcpy(v.data(), p, n * sizeof(fv_real), cudaMemcpyDeviceToHost);
  return v;
}

// ---------------------------------------------------------------------------
// synthetic fields
// ---------------------------------------------------------------------------
// DX0 != DY0 deliberately: a swapped x/y metric changes the answer by O(20%)
// rather than by round-off.
static const double DX0 = 1.0e5;
static const double DY0 = 1.2e5;
// dt is the ACOUSTIC substep. 30 s rather than 300 s so the delp update stays a
// few percent of delp and delp cannot go non-positive (the pt update divides by
// the new delp).
static const double DT = 30.0;
static const double DA_MIN = 1.15e10;
static const double DA_MIN_C = 1.0e10;

// uc/vc cross zero many times inside both fixtures, so the sin_sg sign-switch
// arms in K2 and the upwind arms in K4b are all live.
static fv_real f_uc(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(8.0 * std::sin(0.37 * x + 0.21 * y + 0.11 * double(k)) +
                 1.5 * std::cos(0.13 * y));
}
static fv_real f_vc(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(7.0 * std::cos(0.29 * y + 0.19 * x + 0.13 * double(k)) -
                 1.2 * std::sin(0.17 * x));
}
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
// delp stays comfortably positive; pt is a plausible potential temperature; w is
// signed. None is constant in any direction, so a dropped flux term cannot hide.
static fv_real f_delp(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(3000.0 + 400.0 * std::sin(0.23 * x + 0.31 * y) + 90.0 * double(k));
}
static fv_real f_pt(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(280.0 + 12.0 * std::cos(0.19 * x - 0.27 * y) + 1.5 * double(k));
}
static fv_real f_w(int64_t i, int64_t j, int64_t k) {
  const double x = double(i), y = double(j);
  return fv_real(0.8 * std::sin(0.33 * x + 0.41 * y) - 0.25 * std::cos(0.21 * y) +
                 0.05 * double(k));
}
// Sentinel used to pre-fill ut/vt so Part A can assert that the rows the Fortran
// pass structure never writes are left UNTOUCHED rather than zeroed.
static fv_real f_sentinel(int64_t i, int64_t j, int64_t k) {
  return fv_real(-777.0 - double(i) - 100.0 * double(j) - 10000.0 * double(k));
}

// Grid metrics. sin_sg components are separated by 0.05 so a component mix-up
// shows up as a several-percent error, far above TOL, not as round-off.
struct Grid {
  A2 rarea_c, rarea, area, dxc, dyc, dx, dy, rdx, rdy, divg_u, divg_v;
  A2 cosa_u, cosa_v, sina_u, sina_v, cosa_s, rsin2, del6_u, del6_v;
  A2 dxa, dya, rdxa, rdya, cosa, rsina, rsin_u, rsin_v, f0;
  SG sin_sg;

  void init(const Box &b) {
    rarea_c.init(b); rarea.init(b); area.init(b); dxc.init(b); dyc.init(b);
    dx.init(b); dy.init(b); rdx.init(b); rdy.init(b);
    divg_u.init(b); divg_v.init(b);
    cosa_u.init(b); cosa_v.init(b); sina_u.init(b); sina_v.init(b);
    cosa_s.init(b); rsin2.init(b); del6_u.init(b); del6_v.init(b);
    dxa.init(b); dya.init(b); rdxa.init(b); rdya.init(b);
    cosa.init(b); rsina.init(b); rsin_u.init(b); rsin_v.init(b); f0.init(b);
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
        dxa(i, j) = fv_real(vdx * 1.02);
        dya(i, j) = fv_real(vdy * 0.98);
        rdxa(i, j) = fv_real(1.0 / (vdx * 1.02));
        rdya(i, j) = fv_real(1.0 / (vdy * 0.98));
        area(i, j) = fv_real(vdx * vdy);
        rarea(i, j) = fv_real(1.0 / (vdx * vdy));
        rarea_c(i, j) = fv_real(1.0 / (vdx * vdy * 1.01));
        divg_u(i, j) = fv_real(0.37 * (1.0 + 0.05 * std::sin(0.28 * x + 0.11 * y)));
        divg_v(i, j) = fv_real(0.41 * (1.0 + 0.04 * std::cos(0.19 * x - 0.23 * y)));
        const double cu = 0.08 * std::sin(0.21 * x + 0.11 * y);
        const double cv = 0.07 * std::cos(0.17 * x - 0.13 * y);
        const double cb = 0.06 * std::sin(0.15 * x - 0.24 * y);
        cosa_u(i, j) = fv_real(cu);
        cosa_v(i, j) = fv_real(cv);
        sina_u(i, j) = fv_real(std::sqrt(1.0 - cu * cu));
        sina_v(i, j) = fv_real(std::sqrt(1.0 - cv * cv));
        rsin_u(i, j) = fv_real(1.0 / (1.0 - cu * cu));
        rsin_v(i, j) = fv_real(1.0 / (1.0 - cv * cv));
        cosa(i, j) = fv_real(cb);
        rsina(i, j) = fv_real(1.0 / (1.0 - cb * cb));
        cosa_s(i, j) = fv_real(cb);
        rsin2(i, j) = fv_real(1.0 / (1.0 - 0.0036 * std::sin(0.15 * x) * std::sin(0.15 * x)));
        del6_u(i, j) = fv_real(vdy / vdx * 0.9);
        del6_v(i, j) = fv_real(vdx / vdy * 1.1);
        f0(i, j) = fv_real(1.0e-4 * (1.0 + 0.1 * std::sin(0.12 * x + 0.08 * y)));
        for (int64_t c = 0; c < 4; ++c)
          sin_sg(i, j, c) = fv_real(0.80 + 0.05 * double(c) +
                                    0.01 * std::sin(0.3 * x + 0.4 * y + double(c)));
      }
    }
  }
};

// a2b_ord4 geometry, needed only by S31's smag_corner arm.
struct A2B {
  std::vector<fv_real> grid, agrid;
  std::vector<accumulator_real> ew;
  int64_t n = 0;
  void init(const Box &b) {
    n = b.n2();
    grid.assign(static_cast<size_t>(n) * 2, fv_real(0));
    agrid.assign(static_cast<size_t>(n) * 2, fv_real(0));
    ew.assign(static_cast<size_t>(b.nj()) * 3, accumulator_real(0));
    for (int64_t j = b.loj(); j <= b.hij(); ++j) {
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const size_t o = static_cast<size_t>((i - b.loi()) + (j - b.loj()) * b.ni());
        const double lon = 0.02 * double(i);
        const double lat = 0.015 * double(j);
        grid[o] = fv_real(lon);
        grid[o + static_cast<size_t>(n)] = fv_real(lat);
        agrid[o] = fv_real(lon + 0.01);
        agrid[o + static_cast<size_t>(n)] = fv_real(lat + 0.0075);
      }
    }
    for (int64_t j = 0; j < b.nj(); ++j) {
      ew[static_cast<size_t>(j)] = 0.25;
      ew[static_cast<size_t>(j + b.nj())] = 0.50;
      ew[static_cast<size_t>(j + 2 * b.nj())] = 0.25;
    }
  }
};

static ops::A2BGeometry a2b_of(Dev &d, const Box &b, const A2B &a, const Grid &g) {
  ops::A2BGeometry o{};
  o.dxa = V2<const fv_real>(d.up(g.dxa), b);
  o.dya = V2<const fv_real>(d.up(g.dya), b);
  void *pg = d.raw(a.grid.size() * sizeof(fv_real));
  cudaMemcpy(pg, a.grid.data(), a.grid.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
  void *pa = d.raw(a.agrid.size() * sizeof(fv_real));
  cudaMemcpy(pa, a.agrid.data(), a.agrid.size() * sizeof(fv_real),
             cudaMemcpyHostToDevice);
  o.grid_lonlat = VSG(pg, b);
  o.agrid_lonlat = VSG(pa, b);
  accumulator_real *pe = d.accv(a.ew);
  const View1<const accumulator_real> e{pe, 0, 1};
  o.edge_w = e; o.edge_e = e; o.edge_s = e; o.edge_n = e;
  return o;
}

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

// ===========================================================================
// Part A -- K1/K2/K3 contravariant, edge and corner winds
// sw_core.F90:672-706 (grid_type < 3, .not. nested), :708-779 (four faces),
// :790-864 (four 2x2 corner solves)
// ===========================================================================
// The host reference reproduces the Fortran's own three-pass structure: the
// clipped interior loops that SKIP j == 0, 1, npy-1, npy, then the eight edge
// blocks in the order West, East, South, North, then the four corner blocks in
// the order sw, se, ne, nw. Every later pass overwrites earlier output in place,
// exactly as the Fortran does.
static void ref_winds(const Box &b, const Grid &g, double dt, A3 &ut, A3 &vt) {
  const int64_t is = b.is, ie = b.ie, js = b.js, je = b.je;
  const int64_t npx = b.npx, npy = b.npy;
  const int64_t isd = b.isd(), ied = b.ied(), jsd = b.jsd(), jed = b.jed();

  for (int64_t k = 0; k < b.nk; ++k) {
    const auto uc = [&](int64_t i, int64_t j) { return double(f_uc(i, j, k)); };
    const auto vc = [&](int64_t i, int64_t j) { return double(f_vc(i, j, k)); };

    // --- :686-698 interior, j == 0, 1, npy-1, npy excluded -----------------
    for (int64_t j = jsd; j <= jed; ++j) {
      if (j == 0 || j == 1 || j == (npy - 1) || j == npy) continue;
      for (int64_t i = is - 1; i <= ie + 2; ++i)
        ut(i, j, k) = fv_real((uc(i, j) - 0.25 * double(g.cosa_u(i, j)) *
                                              (vc(i - 1, j) + vc(i, j) +
                                               vc(i - 1, j + 1) + vc(i, j + 1))) *
                              double(g.rsin_u(i, j)));
    }
    // --- :699-705 interior, j == 1, npy excluded ---------------------------
    for (int64_t j = js - 1; j <= je + 2; ++j) {
      if (j == 1 || j == npy) continue;
      for (int64_t i = isd; i <= ied; ++i)
        vt(i, j, k) = fv_real((vc(i, j) - 0.25 * double(g.cosa_v(i, j)) *
                                              (uc(i, j - 1) + uc(i + 1, j - 1) +
                                               uc(i, j) + uc(i + 1, j))) *
                              double(g.rsin_v(i, j)));
    }

    const auto U = [&](int64_t i, int64_t j) { return double(ut(i, j, k)); };
    const auto V = [&](int64_t i, int64_t j) { return double(vt(i, j, k)); };

    // --- :710-724 West ----------------------------------------------------
    if (is == 1) {
      for (int64_t j = jsd; j <= jed; ++j)
        ut(1, j, k) = fv_real(uc(1, j) * dt > 0.0
                                  ? uc(1, j) / double(g.sin_sg(0, j, 2))
                                  : uc(1, j) / double(g.sin_sg(1, j, 0)));
      const int64_t j0 = js > 3 ? js : 3;
      const int64_t j1 = (je + 1) < (npy - 2) ? (je + 1) : (npy - 2);
      for (int64_t j = j0; j <= j1; ++j) {
        vt(0, j, k) = fv_real(vc(0, j) - 0.25 * double(g.cosa_v(0, j)) *
                                            (U(0, j - 1) + U(1, j - 1) + U(0, j) + U(1, j)));
        vt(1, j, k) = fv_real(vc(1, j) - 0.25 * double(g.cosa_v(1, j)) *
                                            (U(1, j - 1) + U(2, j - 1) + U(1, j) + U(2, j)));
      }
    }
    // --- :727-741 East ----------------------------------------------------
    if ((ie + 1) == npx) {
      for (int64_t j = jsd; j <= jed; ++j)
        ut(npx, j, k) = fv_real(uc(npx, j) * dt > 0.0
                                    ? uc(npx, j) / double(g.sin_sg(npx - 1, j, 2))
                                    : uc(npx, j) / double(g.sin_sg(npx, j, 0)));
      const int64_t j0 = js > 3 ? js : 3;
      const int64_t j1 = (je + 1) < (npy - 2) ? (je + 1) : (npy - 2);
      for (int64_t j = j0; j <= j1; ++j) {
        vt(npx - 1, j, k) =
            fv_real(vc(npx - 1, j) - 0.25 * double(g.cosa_v(npx - 1, j)) *
                                         (U(npx - 1, j - 1) + U(npx, j - 1) +
                                          U(npx - 1, j) + U(npx, j)));
        vt(npx, j, k) =
            fv_real(vc(npx, j) - 0.25 * double(g.cosa_v(npx, j)) *
                                     (U(npx, j - 1) + U(npx + 1, j - 1) + U(npx, j) +
                                      U(npx + 1, j)));
      }
    }
    // --- :744-761 South ---------------------------------------------------
    if (js == 1) {
      for (int64_t i = isd; i <= ied; ++i)
        vt(i, 1, k) = fv_real(vc(i, 1) * dt > 0.0
                                  ? vc(i, 1) / double(g.sin_sg(i, 0, 3))
                                  : vc(i, 1) / double(g.sin_sg(i, 1, 1)));
      const int64_t i0 = is > 3 ? is : 3;
      const int64_t i1 = (ie + 1) < (npx - 2) ? (ie + 1) : (npx - 2);
      for (int64_t i = i0; i <= i1; ++i) {
        ut(i, 0, k) = fv_real(uc(i, 0) - 0.25 * double(g.cosa_u(i, 0)) *
                                            (V(i - 1, 0) + V(i, 0) + V(i - 1, 1) + V(i, 1)));
        ut(i, 1, k) = fv_real(uc(i, 1) - 0.25 * double(g.cosa_u(i, 1)) *
                                            (V(i - 1, 1) + V(i, 1) + V(i - 1, 2) + V(i, 2)));
      }
    }
    // --- :764-778 North ---------------------------------------------------
    if ((je + 1) == npy) {
      for (int64_t i = isd; i <= ied; ++i)
        vt(i, npy, k) = fv_real(vc(i, npy) * dt > 0.0
                                    ? vc(i, npy) / double(g.sin_sg(i, npy - 1, 3))
                                    : vc(i, npy) / double(g.sin_sg(i, npy, 1)));
      const int64_t i0 = is > 3 ? is : 3;
      const int64_t i1 = (ie + 1) < (npx - 2) ? (ie + 1) : (npx - 2);
      for (int64_t i = i0; i <= i1; ++i) {
        ut(i, npy - 1, k) =
            fv_real(uc(i, npy - 1) - 0.25 * double(g.cosa_u(i, npy - 1)) *
                                         (V(i - 1, npy - 1) + V(i, npy - 1) +
                                          V(i - 1, npy) + V(i, npy)));
        ut(i, npy, k) =
            fv_real(uc(i, npy) - 0.25 * double(g.cosa_u(i, npy)) *
                                     (V(i - 1, npy) + V(i, npy) + V(i - 1, npy + 1) +
                                      V(i, npy + 1)));
      }
    }

    // --- :790-803 sw corner ------------------------------------------------
    const auto CU = [&](int64_t i, int64_t j) { return double(g.cosa_u(i, j)); };
    const auto CV = [&](int64_t i, int64_t j) { return double(g.cosa_v(i, j)); };
    double damp;
    if (b.west && b.south) {
      damp = 1.0 / (1.0 - 0.0625 * CU(2, 0) * CV(1, 0));
      ut(2, 0, k) = fv_real((uc(2, 0) - 0.25 * CU(2, 0) *
                                            (V(1, 1) + V(2, 1) + V(2, 0) + vc(1, 0) -
                                             0.25 * CV(1, 0) *
                                                 (U(1, 0) + U(1, -1) + U(2, -1)))) *
                            damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(0, 1) * CV(0, 2));
      vt(0, 2, k) = fv_real((vc(0, 2) - 0.25 * CV(0, 2) *
                                            (U(1, 1) + U(1, 2) + U(0, 2) + uc(0, 1) -
                                             0.25 * CU(0, 1) *
                                                 (V(0, 1) + V(-1, 1) + V(-1, 2)))) *
                            damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(2, 1) * CV(1, 2));
      ut(2, 1, k) = fv_real((uc(2, 1) - 0.25 * CU(2, 1) *
                                            (V(1, 1) + V(2, 1) + V(2, 2) + vc(1, 2) -
                                             0.25 * CV(1, 2) *
                                                 (U(1, 1) + U(1, 2) + U(2, 2)))) *
                            damp);
      vt(1, 2, k) = fv_real((vc(1, 2) - 0.25 * CV(1, 2) *
                                            (U(1, 1) + U(1, 2) + U(2, 2) + uc(2, 1) -
                                             0.25 * CU(2, 1) *
                                                 (V(1, 1) + V(2, 1) + V(2, 2)))) *
                            damp);
    }
    // --- :806-822 se corner ------------------------------------------------
    if (b.east && b.south) {
      damp = 1.0 / (1.0 - 0.0625 * CU(npx - 1, 0) * CV(npx - 1, 0));
      ut(npx - 1, 0, k) =
          fv_real((uc(npx - 1, 0) -
                   0.25 * CU(npx - 1, 0) *
                       (V(npx - 1, 1) + V(npx - 2, 1) + V(npx - 2, 0) + vc(npx - 1, 0) -
                        0.25 * CV(npx - 1, 0) *
                            (U(npx, 0) + U(npx, -1) + U(npx - 1, -1)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(npx + 1, 1) * CV(npx, 2));
      vt(npx, 2, k) =
          fv_real((vc(npx, 2) -
                   0.25 * CV(npx, 2) *
                       (U(npx, 1) + U(npx, 2) + U(npx + 1, 2) + uc(npx + 1, 1) -
                        0.25 * CU(npx + 1, 1) *
                            (V(npx, 1) + V(npx + 1, 1) + V(npx + 1, 2)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(npx - 1, 1) * CV(npx - 1, 2));
      ut(npx - 1, 1, k) =
          fv_real((uc(npx - 1, 1) -
                   0.25 * CU(npx - 1, 1) *
                       (V(npx - 1, 1) + V(npx - 2, 1) + V(npx - 2, 2) + vc(npx - 1, 2) -
                        0.25 * CV(npx - 1, 2) *
                            (U(npx, 1) + U(npx, 2) + U(npx - 1, 2)))) *
                  damp);
      vt(npx - 1, 2, k) =
          fv_real((vc(npx - 1, 2) -
                   0.25 * CV(npx - 1, 2) *
                       (U(npx, 1) + U(npx, 2) + U(npx - 1, 2) + uc(npx - 1, 1) -
                        0.25 * CU(npx - 1, 1) *
                            (V(npx - 1, 1) + V(npx - 2, 1) + V(npx - 2, 2)))) *
                  damp);
    }
    // --- :825-841 ne corner ------------------------------------------------
    if (b.east && b.north) {
      damp = 1.0 / (1.0 - 0.0625 * CU(npx - 1, npy) * CV(npx - 1, npy + 1));
      ut(npx - 1, npy, k) =
          fv_real((uc(npx - 1, npy) -
                   0.25 * CU(npx - 1, npy) *
                       (V(npx - 1, npy) + V(npx - 2, npy) + V(npx - 2, npy + 1) +
                        vc(npx - 1, npy + 1) -
                        0.25 * CV(npx - 1, npy + 1) *
                            (U(npx, npy) + U(npx, npy + 1) + U(npx - 1, npy + 1)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(npx + 1, npy - 1) * CV(npx, npy - 1));
      vt(npx, npy - 1, k) =
          fv_real((vc(npx, npy - 1) -
                   0.25 * CV(npx, npy - 1) *
                       (U(npx, npy - 1) + U(npx, npy - 2) + U(npx + 1, npy - 2) +
                        uc(npx + 1, npy - 1) -
                        0.25 * CU(npx + 1, npy - 1) *
                            (V(npx, npy) + V(npx + 1, npy) + V(npx + 1, npy - 1)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(npx - 1, npy - 1) * CV(npx - 1, npy - 1));
      ut(npx - 1, npy - 1, k) =
          fv_real((uc(npx - 1, npy - 1) -
                   0.25 * CU(npx - 1, npy - 1) *
                       (V(npx - 1, npy) + V(npx - 2, npy) + V(npx - 2, npy - 1) +
                        vc(npx - 1, npy - 1) -
                        0.25 * CV(npx - 1, npy - 1) *
                            (U(npx, npy - 1) + U(npx, npy - 2) + U(npx - 1, npy - 2)))) *
                  damp);
      vt(npx - 1, npy - 1, k) =
          fv_real((vc(npx - 1, npy - 1) -
                   0.25 * CV(npx - 1, npy - 1) *
                       (U(npx, npy - 1) + U(npx, npy - 2) + U(npx - 1, npy - 2) +
                        uc(npx - 1, npy - 1) -
                        0.25 * CU(npx - 1, npy - 1) *
                            (V(npx - 1, npy) + V(npx - 2, npy) + V(npx - 2, npy - 1)))) *
                  damp);
    }
    // --- :844-862 nw corner ------------------------------------------------
    if (b.west && b.north) {
      damp = 1.0 / (1.0 - 0.0625 * CU(2, npy) * CV(1, npy + 1));
      ut(2, npy, k) =
          fv_real((uc(2, npy) -
                   0.25 * CU(2, npy) *
                       (V(1, npy) + V(2, npy) + V(2, npy + 1) + vc(1, npy + 1) -
                        0.25 * CV(1, npy + 1) *
                            (U(1, npy) + U(1, npy + 1) + U(2, npy + 1)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(0, npy - 1) * CV(0, npy - 1));
      vt(0, npy - 1, k) =
          fv_real((vc(0, npy - 1) -
                   0.25 * CV(0, npy - 1) *
                       (U(1, npy - 1) + U(1, npy - 2) + U(0, npy - 2) + uc(0, npy - 1) -
                        0.25 * CU(0, npy - 1) *
                            (V(0, npy) + V(-1, npy) + V(-1, npy - 1)))) *
                  damp);
      damp = 1.0 / (1.0 - 0.0625 * CU(2, npy - 1) * CV(1, npy - 1));
      ut(2, npy - 1, k) =
          fv_real((uc(2, npy - 1) -
                   0.25 * CU(2, npy - 1) *
                       (V(1, npy) + V(2, npy) + V(2, npy - 1) + vc(1, npy - 1) -
                        0.25 * CV(1, npy - 1) *
                            (U(1, npy - 1) + U(1, npy - 2) + U(2, npy - 2)))) *
                  damp);
      vt(1, npy - 1, k) =
          fv_real((vc(1, npy - 1) -
                   0.25 * CV(1, npy - 1) *
                       (U(1, npy - 1) + U(1, npy - 2) + U(2, npy - 2) + uc(2, npy - 1) -
                        0.25 * CU(2, npy - 1) *
                            (V(1, npy) + V(2, npy) + V(2, npy - 1)))) *
                  damp);
    }
  }
}

static bool part_a_one(const Box &b, const char *tag) {
  Dev d;
  Grid g;
  g.init(b);
  const dgrid::TransportBounds tb = tb_of(b);

  A3 uc, vc, ut0, vt0;
  uc.init(b); vc.init(b); ut0.init(b); vt0.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        uc(i, j, k) = f_uc(i, j, k);
        vc(i, j, k) = f_vc(i, j, k);
        // Pre-fill with a sentinel, NOT zero: the rows the Fortran pass
        // structure never touches must survive unchanged.
        ut0(i, j, k) = f_sentinel(i, j, k);
        vt0(i, j, k) = f_sentinel(i, j, k) - fv_real(0.5);
      }

  A3 ut_ref = ut0, vt_ref = vt0;
  ut_ref.b = &b; vt_ref.b = &b;
  ref_winds(b, g, DT, ut_ref, vt_ref);

  fv_real *d_uc = d.up(uc), *d_vc = d.up(vc);
  fv_real *d_ut = d.up(ut0), *d_vt = d.up(vt0);
  fv_real *d_cu = d.up(g.cosa_u), *d_cv = d.up(g.cosa_v);
  fv_real *d_ru = d.up(g.rsin_u), *d_rv = d.up(g.rsin_v);
  fv_real *d_sg = d.up(g.sin_sg);
  if (!d_uc || !d_vc || !d_ut || !d_vt || !d_sg) {
    std::printf("  [A/%s] device allocation failed\n", tag);
    return false;
  }

  const auto UC = V3<const fv_real>(d_uc, b);
  const auto VC = V3<const fv_real>(d_vc, b);
  auto UT = V3<fv_real>(d_ut, b);
  auto VT = V3<fv_real>(d_vt, b);

  fv3_gpu_status rc = dgrid::launch_dsw_contravariant_cuda(
      tb, UC, VC, V2<const fv_real>(d_cu, b), V2<const fv_real>(d_cv, b),
      V2<const fv_real>(d_ru, b), V2<const fv_real>(d_rv, b), UT, VT, nullptr);
  if (rc != FV3_GPU_SUCCESS) {
    std::printf("  [A/%s] K1 returned %d\n", tag, int(rc));
    return false;
  }
  rc = dgrid::launch_dsw_edge_winds_cuda(tb, fv_real(DT), UC, VC,
                                         V2<const fv_real>(d_cu, b),
                                         V2<const fv_real>(d_cv, b), VSG(d_sg, b), UT,
                                         VT, nullptr);
  if (rc != FV3_GPU_SUCCESS) {
    std::printf("  [A/%s] K2 returned %d\n", tag, int(rc));
    return false;
  }
  rc = dgrid::launch_dsw_corner_winds_cuda(tb, UC, VC, V2<const fv_real>(d_cu, b),
                                           V2<const fv_real>(d_cv, b), UT, VT, nullptr);
  if (rc != FV3_GPU_SUCCESS) {
    std::printf("  [A/%s] K3 returned %d\n", tag, int(rc));
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [A/%s] sync failed\n", tag);
    return false;
  }

  const A3 ut_got = down3(d_ut, b), vt_got = down3(d_vt, b);
  long bad = 0, untouched = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        if (!close_rel(ut_got(i, j, k), ut_ref(i, j, k), TOL) ||
            !close_rel(vt_got(i, j, k), vt_ref(i, j, k), TOL)) {
          if (bad < 6)
            std::printf("  [A/%s] i=%lld j=%lld k=%lld ut %.12g vs %.12g | vt %.12g vs %.12g\n",
                        tag, (long long)i, (long long)j, (long long)k,
                        double(ut_got(i, j, k)), double(ut_ref(i, j, k)),
                        double(vt_got(i, j, k)), double(vt_ref(i, j, k)));
          ++bad;
        }
        if (ut_ref(i, j, k) == f_sentinel(i, j, k)) ++untouched;
      }
  if (bad) {
    std::printf("  [A/%s] %ld mismatches\n", tag, bad);
    return false;
  }
  // The excluded-row assertion is only meaningful if there ARE untouched cells.
  if (untouched == 0) {
    std::printf("  [A/%s] fixture degenerate: no untouched ut cells\n", tag);
    return false;
  }
  std::printf("  [A/%s] ok (%ld ut cells correctly left untouched)\n", tag, untouched);
  return true;
}

static bool part_a() {
  const Box s = seam_box(), n = interior_box();
  return part_a_one(s, "seam") && part_a_one(n, "interior");
}

// ===========================================================================
// Part B -- K4a/K4b advective fluxes and Courant numbers, plus ra_x/ra_y
// sw_core.F90:882-921 (xfx_adv/yfx_adv, crx_adv/cry_adv) and :926-937 (ra)
// ===========================================================================
// The in-place rescale is the point: crx_adv consumes the PRE-rescale xfx_adv,
// so a kernel that rescaled first would produce crx_adv larger by a factor
// dy*sin_sg. All four arrays are sentinel-filled first, so a write outside the
// Fortran loop bounds also fails.
static bool part_b_one(const Box &b, const char *tag) {
  Dev d;
  Grid g;
  g.init(b);
  const dgrid::TransportBounds tb = tb_of(b);
  const int64_t isd = b.isd(), ied = b.ied(), jsd = b.jsd(), jed = b.jed();

  A3 ut, vt, crx, cry, xfx, yfx, rax, ray;
  ut.init(b); vt.init(b);
  crx.init(b); cry.init(b); xfx.init(b); yfx.init(b); rax.init(b); ray.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        ut(i, j, k) = f_u(i, j, k);
        vt(i, j, k) = f_v(i, j, k);
        crx(i, j, k) = f_sentinel(i, j, k);
        cry(i, j, k) = f_sentinel(i, j, k) - fv_real(1);
        xfx(i, j, k) = f_sentinel(i, j, k) - fv_real(2);
        yfx(i, j, k) = f_sentinel(i, j, k) - fv_real(3);
        rax(i, j, k) = f_sentinel(i, j, k) - fv_real(4);
        ray(i, j, k) = f_sentinel(i, j, k) - fv_real(5);
      }

  // --- host reference, Fortran pass order ---------------------------------
  A3 crx_r = crx, cry_r = cry, xfx_r = xfx, yfx_r = yfx, rax_r = rax, ray_r = ray;
  long pos_x = 0, neg_x = 0, pos_y = 0, neg_y = 0;
  for (int64_t k = 0; k < b.nk; ++k) {
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i)
        xfx_r(i, j, k) = fv_real(DT * double(ut(i, j, k)));
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        yfx_r(i, j, k) = fv_real(DT * double(vt(i, j, k)));
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        const fv_real x = xfx_r(i, j, k);
        if (x > fv_real(0)) {
          crx_r(i, j, k) = fv_real(double(x) * double(g.rdxa(i - 1, j)));
          xfx_r(i, j, k) =
              fv_real(double(g.dy(i, j)) * double(x) * double(g.sin_sg(i - 1, j, 2)));
          ++pos_x;
        } else {
          crx_r(i, j, k) = fv_real(double(x) * double(g.rdxa(i, j)));
          xfx_r(i, j, k) =
              fv_real(double(g.dy(i, j)) * double(x) * double(g.sin_sg(i, j, 0)));
          ++neg_x;
        }
      }
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = isd; i <= ied; ++i) {
        const fv_real y = yfx_r(i, j, k);
        if (y > fv_real(0)) {
          cry_r(i, j, k) = fv_real(double(y) * double(g.rdya(i, j - 1)));
          yfx_r(i, j, k) =
              fv_real(double(g.dx(i, j)) * double(y) * double(g.sin_sg(i, j - 1, 3)));
          ++pos_y;
        } else {
          cry_r(i, j, k) = fv_real(double(y) * double(g.rdya(i, j)));
          yfx_r(i, j, k) =
              fv_real(double(g.dx(i, j)) * double(y) * double(g.sin_sg(i, j, 1)));
          ++neg_y;
        }
      }
    for (int64_t j = jsd; j <= jed; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i)
        rax_r(i, j, k) = fv_real(double(g.area(i, j)) + double(xfx_r(i, j, k)) -
                                 double(xfx_r(i + 1, j, k)));
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = isd; i <= ied; ++i)
        ray_r(i, j, k) = fv_real(double(g.area(i, j)) + double(yfx_r(i, j, k)) -
                                 double(yfx_r(i, j + 1, k)));
  }
  if (!pos_x || !neg_x || !pos_y || !neg_y) {
    std::printf("  [B/%s] fixture degenerate: upwind arms %ld/%ld %ld/%ld\n", tag,
                pos_x, neg_x, pos_y, neg_y);
    return false;
  }

  fv_real *d_ut = d.up(ut), *d_vt = d.up(vt);
  fv_real *d_crx = d.up(crx), *d_cry = d.up(cry);
  fv_real *d_xfx = d.up(xfx), *d_yfx = d.up(yfx);
  fv_real *d_rax = d.up(rax), *d_ray = d.up(ray);
  fv_real *d_dx = d.up(g.dx), *d_dy = d.up(g.dy);
  fv_real *d_rdxa = d.up(g.rdxa), *d_rdya = d.up(g.rdya);
  fv_real *d_sg = d.up(g.sin_sg), *d_area = d.up(g.area);
  if (!d_ray || !d_area) {
    std::printf("  [B/%s] device allocation failed\n", tag);
    return false;
  }

  fv3_gpu_status rc = dgrid::launch_dsw_advective_flux_cuda(
      tb, fv_real(DT), V3<const fv_real>(d_ut, b), V3<const fv_real>(d_vt, b),
      V3<fv_real>(d_xfx, b), V3<fv_real>(d_yfx, b), nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_dsw_courant_finalize_cuda(
        tb, V2<const fv_real>(d_dx, b), V2<const fv_real>(d_dy, b),
        V2<const fv_real>(d_rdxa, b), V2<const fv_real>(d_rdya, b), VSG(d_sg, b),
        V3<fv_real>(d_crx, b), V3<fv_real>(d_cry, b), V3<fv_real>(d_xfx, b),
        V3<fv_real>(d_yfx, b), nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_ra_x_cuda(tb, V3<const fv_real>(d_xfx, b),
                                 V2<const fv_real>(d_area, b), V3<fv_real>(d_rax, b),
                                 nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_ra_y_cuda(tb, V3<const fv_real>(d_yfx, b),
                                 V2<const fv_real>(d_area, b), V3<fv_real>(d_ray, b),
                                 nullptr);
  if (rc != FV3_GPU_SUCCESS) {
    std::printf("  [B/%s] launch returned %d\n", tag, int(rc));
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [B/%s] sync failed\n", tag);
    return false;
  }

  struct Cmp { const char *n; const void *dev; const A3 *ref; };
  const A3 gx = down3(d_crx, b), gy = down3(d_cry, b);
  const A3 gxf = down3(d_xfx, b), gyf = down3(d_yfx, b);
  const A3 grx = down3(d_rax, b), gry = down3(d_ray, b);
  const A3 *refs[6] = {&crx_r, &cry_r, &xfx_r, &yfx_r, &rax_r, &ray_r};
  const A3 *gots[6] = {&gx, &gy, &gxf, &gyf, &grx, &gry};
  const char *names[6] = {"crx", "cry", "xfx", "yfx", "ra_x", "ra_y"};
  long bad = 0;
  for (int n = 0; n < 6; ++n)
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.loj(); j <= b.hij(); ++j)
        for (int64_t i = b.loi(); i <= b.hii(); ++i)
          if (!close_rel((*gots[n])(i, j, k), (*refs[n])(i, j, k), TOL)) {
            if (bad < 8)
              std::printf("  [B/%s] %s i=%lld j=%lld k=%lld %.12g vs %.12g\n", tag,
                          names[n], (long long)i, (long long)j, (long long)k,
                          double((*gots[n])(i, j, k)), double((*refs[n])(i, j, k)));
            ++bad;
          }
  if (bad) {
    std::printf("  [B/%s] %ld mismatches\n", tag, bad);
    return false;
  }
  std::printf("  [B/%s] ok (upwind arms x %ld/%ld y %ld/%ld)\n", tag, pos_x, neg_x,
              pos_y, neg_y);
  return true;
}

static bool part_b() {
  const Box s = seam_box(), n = interior_box();
  return part_b_one(s, "seam") && part_b_one(n, "interior");
}

// ===========================================================================
// Part F -- K5 flux capacitor, sw_core.F90:942-960
// ===========================================================================
// Four accumulators with four DIFFERENT loop rectangles:
//   cx     jsd..jed x is..ie+1
//   xflux  js..je   x is..ie+1
//   cy     js..je+1 x isd..ied
//   yflux  js..je+1 x is..ie
// The pre-seed is nonzero and the launcher is called TWICE, so a kernel that
// assigned instead of accumulating, or that shared one rectangle across all
// four arrays, fails. §9's "not only a zero-initialized one-shot case".
static bool part_f() {
  const Box b = seam_box();
  Dev d;
  const dgrid::TransportBounds tb = tb_of(b);
  const int64_t isd = b.isd(), ied = b.ied(), jsd = b.jsd(), jed = b.jed();

  A3 crx, cry, fx, fy;
  crx.init(b); cry.init(b); fx.init(b); fy.init(b);
  D3 cx, cy, xflux, yflux;
  cx.init(b); cy.init(b); xflux.init(b); yflux.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        crx(i, j, k) = f_u(i, j, k) * fv_real(0.001);
        cry(i, j, k) = f_v(i, j, k) * fv_real(0.001);
        fx(i, j, k) = f_ua(i, j, k) * fv_real(1000.0);
        fy(i, j, k) = f_va(i, j, k) * fv_real(1000.0);
        // Deliberately nonzero and distinct per array.
        const double s = double(i) + 10.0 * double(j) + 100.0 * double(k);
        cx(i, j, k) = 1.0 + 0.001 * s;
        cy(i, j, k) = 2.0 + 0.002 * s;
        xflux(i, j, k) = 3.0e5 + s;
        yflux(i, j, k) = 4.0e5 + 2.0 * s;
      }

  D3 cx_r = cx, cy_r = cy, xf_r = xflux, yf_r = yflux;
  for (int pass = 0; pass < 2; ++pass)
    for (int64_t k = 0; k < b.nk; ++k) {
      for (int64_t j = jsd; j <= jed; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i)
          cx_r(i, j, k) += accumulator_real(crx(i, j, k));
      for (int64_t j = b.js; j <= b.je; ++j)
        for (int64_t i = b.is; i <= b.ie + 1; ++i)
          xf_r(i, j, k) += accumulator_real(fx(i, j, k));
      for (int64_t j = b.js; j <= b.je + 1; ++j) {
        for (int64_t i = isd; i <= ied; ++i)
          cy_r(i, j, k) += accumulator_real(cry(i, j, k));
        for (int64_t i = b.is; i <= b.ie; ++i)
          yf_r(i, j, k) += accumulator_real(fy(i, j, k));
      }
    }

  fv_real *d_crx = d.up(crx), *d_cry = d.up(cry), *d_fx = d.up(fx), *d_fy = d.up(fy);
  accumulator_real *d_cx = d.acc(cx), *d_cy = d.acc(cy);
  accumulator_real *d_xf = d.acc(xflux), *d_yf = d.acc(yflux);
  if (!d_yf) {
    std::printf("  [F] device allocation failed\n");
    return false;
  }
  for (int pass = 0; pass < 2; ++pass) {
    const fv3_gpu_status rc = dgrid::launch_dsw_flux_capacitor_cuda(
        tb, V3<const fv_real>(d_crx, b), V3<const fv_real>(d_cry, b),
        V3<const fv_real>(d_fx, b), V3<const fv_real>(d_fy, b),
        V3<accumulator_real>(d_cx, b), V3<accumulator_real>(d_cy, b),
        V3<accumulator_real>(d_xf, b), V3<accumulator_real>(d_yf, b), nullptr);
    if (rc != FV3_GPU_SUCCESS) {
      std::printf("  [F] pass %d returned %d\n", pass, int(rc));
      return false;
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [F] sync failed\n");
    return false;
  }

  const D3 gcx = downd(d_cx, b), gcy = downd(d_cy, b);
  const D3 gxf = downd(d_xf, b), gyf = downd(d_yf, b);
  const D3 *refs[4] = {&cx_r, &cy_r, &xf_r, &yf_r};
  const D3 *gots[4] = {&gcx, &gcy, &gxf, &gyf};
  const char *names[4] = {"cx", "cy", "xflux", "yflux"};
  long bad = 0, advanced = 0;
  for (int n = 0; n < 4; ++n)
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.loj(); j <= b.hij(); ++j)
        for (int64_t i = b.loi(); i <= b.hii(); ++i) {
          if (!close_rel((*gots[n])(i, j, k), (*refs[n])(i, j, k), 1e-13)) {
            if (bad < 8)
              std::printf("  [F] %s i=%lld j=%lld k=%lld %.17g vs %.17g\n", names[n],
                          (long long)i, (long long)j, (long long)k,
                          (*gots[n])(i, j, k), (*refs[n])(i, j, k));
            ++bad;
          }
          if ((*gots[n])(i, j, k) != (n == 0   ? cx(i, j, k)
                                      : n == 1 ? cy(i, j, k)
                                      : n == 2 ? xflux(i, j, k)
                                               : yflux(i, j, k)))
            ++advanced;
        }
  if (bad) {
    std::printf("  [F] %ld mismatches\n", bad);
    return false;
  }
  if (!advanced) {
    std::printf("  [F] no accumulator advanced -- fixture degenerate\n");
    return false;
  }
  std::printf("  [F] ok (2 accumulation passes, %ld cells advanced)\n", advanced);
  return true;
}

// ===========================================================================
// Part G -- K6 heat_source/diss_est reset, sw_core.F90:963-968
// ===========================================================================
// Both are intent(OUT) per d_sw call, so the reset is required. Its SCOPE is
// is..ie x js..je only: the halo must keep whatever the caller left there, so a
// whole-array memset fails this part.
static bool part_g() {
  const Box b = seam_box();
  Dev d;
  const dgrid::TransportBounds tb = tb_of(b);
  A3 hs, de;
  hs.init(b); de.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        hs(i, j, k) = f_sentinel(i, j, k);
        de(i, j, k) = f_sentinel(i, j, k) - fv_real(9);
      }
  fv_real *d_hs = d.up(hs), *d_de = d.up(de);
  if (!d_de) {
    std::printf("  [G] device allocation failed\n");
    return false;
  }
  const fv3_gpu_status rc = dgrid::launch_dsw_heat_reset_cuda(
      tb, V3<fv_real>(d_hs, b), V3<fv_real>(d_de, b), nullptr);
  if (rc != FV3_GPU_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [G] launch/sync failed (%d)\n", int(rc));
    return false;
  }
  const A3 ghs = down3(d_hs, b), gde = down3(d_de, b);
  long bad = 0, zeroed = 0, kept = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        const bool inside = i >= b.is && i <= b.ie && j >= b.js && j <= b.je;
        const fv_real eh = inside ? fv_real(0) : hs(i, j, k);
        const fv_real ed = inside ? fv_real(0) : de(i, j, k);
        if (ghs(i, j, k) != eh || gde(i, j, k) != ed) {
          if (bad < 6)
            std::printf("  [G] i=%lld j=%lld k=%lld hs %.8g/%.8g de %.8g/%.8g\n",
                        (long long)i, (long long)j, (long long)k, double(ghs(i, j, k)),
                        double(eh), double(gde(i, j, k)), double(ed));
          ++bad;
        }
        if (inside) ++zeroed; else ++kept;
      }
  if (bad) {
    std::printf("  [G] %ld mismatches\n", bad);
    return false;
  }
  std::printf("  [G] ok (%ld cells reset, %ld halo cells preserved)\n", zeroed, kept);
  return true;
}

// ===========================================================================
// Part C -- K9 joint pt/delp update, sw_core.F90:1034-1047 (.not. inline_q arm)
// ===========================================================================
//   pt   = pt*delp + (gx(i)-gx(i+1)+gy(j)-gy(j+1))*rarea     <- OLD delp
//   delp = delp    + (fx(i)-fx(i+1)+fy(j)-fy(j+1))*rarea
//   pt   = pt/delp                                           <- NEW delp
// Both the OLD-delp mass weighting and the NEW-delp division are load-bearing,
// so the part also builds two WRONG-ORDER references (delp updated first, and
// the divide taken against the old delp) and requires the CUDA result to DIFFER
// from each. Without that, a kernel that used one delp everywhere could pass.
static bool part_c() {
  const Box b = seam_box();
  Dev d;
  Grid g;
  g.init(b);
  const dgrid::TransportBounds tb = tb_of(b);

  A3 pt, delp, fx, fy, gx, gy;
  pt.init(b); delp.init(b); fx.init(b); fy.init(b); gx.init(b); gy.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        pt(i, j, k) = f_pt(i, j, k);
        delp(i, j, k) = f_delp(i, j, k);
        // Flux scales chosen so BOTH increments are ~1% of the field they
        // update: rarea is ~8.3e-11 here and the neighbour differences of f_u /
        // f_ua are O(1), so the flux divergence is ~1e-10 times the scale.
        // Getting this wrong makes the part degenerate rather than wrong --
        // 1e-6 relative increments are invisible at the R4 tolerance, and the
        // two wrong-order references below would then agree with the
        // authoritative one. The discrimination guard at the bottom catches
        // exactly that, so do not retune these without re-reading it.
        fx(i, j, k) = fv_real(double(f_u(i, j, k)) * 3.0e10);   // delp: ~30 Pa of 3000
        fy(i, j, k) = fv_real(double(f_v(i, j, k)) * 3.0e10);
        gx(i, j, k) = fv_real(double(f_ua(i, j, k)) * 3.0e13);  // pt*delp: ~1% of 8.4e5
        gy(i, j, k) = fv_real(double(f_va(i, j, k)) * 3.0e13);
      }

  A3 pt_r = pt, dp_r = delp, pt_w1 = pt, dp_w1 = delp, pt_w2 = pt;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const double ra = double(g.rarea(i, j));
        const double dg = (double(gx(i, j, k)) - double(gx(i + 1, j, k)) +
                           double(gy(i, j, k)) - double(gy(i, j + 1, k))) * ra;
        const double df = (double(fx(i, j, k)) - double(fx(i + 1, j, k)) +
                           double(fy(i, j, k)) - double(fy(i, j + 1, k))) * ra;
        const fv_real dp_old = delp(i, j, k);
        // authoritative
        const fv_real p1 = fv_real(double(pt(i, j, k)) * double(dp_old) + dg);
        const fv_real dp_new = fv_real(double(dp_old) + df);
        dp_r(i, j, k) = dp_new;
        pt_r(i, j, k) = fv_real(double(p1) / double(dp_new));
        // wrong order 1: delp advanced before the pt mass weighting
        const fv_real w1 = fv_real(double(pt(i, j, k)) * double(dp_new) + dg);
        dp_w1(i, j, k) = dp_new;
        pt_w1(i, j, k) = fv_real(double(w1) / double(dp_new));
        // wrong order 2: divide by the OLD delp
        pt_w2(i, j, k) = fv_real(double(p1) / double(dp_old));
      }

  fv_real *d_pt = d.up(pt), *d_dp = d.up(delp);
  fv_real *d_fx = d.up(fx), *d_fy = d.up(fy), *d_gx = d.up(gx), *d_gy = d.up(gy);
  fv_real *d_ra = d.up(g.rarea);
  if (!d_ra) {
    std::printf("  [C] device allocation failed\n");
    return false;
  }
  const fv3_gpu_status rc = dgrid::launch_dsw_pt_delp_update_cuda(
      tb, V3<const fv_real>(d_fx, b), V3<const fv_real>(d_fy, b),
      V3<const fv_real>(d_gx, b), V3<const fv_real>(d_gy, b),
      V2<const fv_real>(d_ra, b), V3<fv_real>(d_pt, b), V3<fv_real>(d_dp, b), nullptr);
  if (rc != FV3_GPU_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [C] launch/sync failed (%d)\n", int(rc));
    return false;
  }
  const A3 gpt = down3(d_pt, b), gdp = down3(d_dp, b);
  long bad = 0, differs_w1 = 0, differs_w2 = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        if (!close_rel(gpt(i, j, k), pt_r(i, j, k), TOL) ||
            !close_rel(gdp(i, j, k), dp_r(i, j, k), TOL)) {
          if (bad < 6)
            std::printf("  [C] i=%lld j=%lld k=%lld pt %.12g/%.12g delp %.12g/%.12g\n",
                        (long long)i, (long long)j, (long long)k, double(gpt(i, j, k)),
                        double(pt_r(i, j, k)), double(gdp(i, j, k)),
                        double(dp_r(i, j, k)));
          ++bad;
        }
        const bool inside = i >= b.is && i <= b.ie && j >= b.js && j <= b.je;
        if (inside && !close_rel(gpt(i, j, k), pt_w1(i, j, k), TOL)) ++differs_w1;
        if (inside && !close_rel(gpt(i, j, k), pt_w2(i, j, k), TOL)) ++differs_w2;
      }
  if (bad) {
    std::printf("  [C] %ld mismatches\n", bad);
    return false;
  }
  const long inside_cells = (b.ie - b.is + 1) * (b.je - b.js + 1) * b.nk;
  if (differs_w1 < inside_cells / 2 || differs_w2 < inside_cells / 2) {
    std::printf("  [C] ordering not discriminated (w1 %ld, w2 %ld of %ld)\n", differs_w1,
                differs_w2, inside_cells);
    return false;
  }
  std::printf("  [C] ok (rejects both wrong orders: %ld/%ld cells)\n", differs_w1,
              differs_w2);
  return true;
}

// ===========================================================================
// Part D -- K10 dpx accumulation, sw_core.F90:1050-1054
// ===========================================================================
//   dpx = dpx + ( (fx(i)-fx(i+1)) + (fy(j)-fy(j+1)) )*rarea
// dpx is kind=8 while fx/fy are working precision, so the Fortran's
// parenthesisation is observable in R4 and is preserved. Seeded nonzero and run
// twice, like Part F.
static bool part_d() {
  const Box b = seam_box();
  Dev d;
  Grid g;
  g.init(b);
  const dgrid::TransportBounds tb = tb_of(b);

  A3 fx, fy;
  fx.init(b); fy.init(b);
  D3 dpx;
  dpx.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        fx(i, j, k) = fv_real(double(f_u(i, j, k)) * 2.0e7);
        fy(i, j, k) = fv_real(double(f_v(i, j, k)) * 2.0e7);
        dpx(i, j, k) = 17.5 + 0.25 * double(i) + 3.0 * double(j) + 11.0 * double(k);
      }
  D3 ref = dpx;
  for (int pass = 0; pass < 2; ++pass)
    for (int64_t k = 0; k < b.nk; ++k)
      for (int64_t j = b.js; j <= b.je; ++j)
        for (int64_t i = b.is; i <= b.ie; ++i)
          ref(i, j, k) += accumulator_real(
              (double(fv_real(fx(i, j, k) - fx(i + 1, j, k))) +
               double(fv_real(fy(i, j, k) - fy(i, j + 1, k)))) * double(g.rarea(i, j)));

  fv_real *d_fx = d.up(fx), *d_fy = d.up(fy), *d_ra = d.up(g.rarea);
  accumulator_real *d_dpx = d.acc(dpx);
  if (!d_dpx) {
    std::printf("  [D] device allocation failed\n");
    return false;
  }
  for (int pass = 0; pass < 2; ++pass) {
    const fv3_gpu_status rc = dgrid::launch_dsw_dpx_accumulate_cuda(
        tb, V3<const fv_real>(d_fx, b), V3<const fv_real>(d_fy, b),
        V2<const fv_real>(d_ra, b), V3<accumulator_real>(d_dpx, b), nullptr);
    if (rc != FV3_GPU_SUCCESS) {
      std::printf("  [D] pass %d returned %d\n", pass, int(rc));
      return false;
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [D] sync failed\n");
    return false;
  }
  const D3 got = downd(d_dpx, b);
  long bad = 0, advanced = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        if (!close_rel(got(i, j, k), ref(i, j, k), TOL)) {
          if (bad < 6)
            std::printf("  [D] i=%lld j=%lld k=%lld %.17g vs %.17g\n", (long long)i,
                        (long long)j, (long long)k, got(i, j, k), ref(i, j, k));
          ++bad;
        }
        if (got(i, j, k) != dpx(i, j, k)) ++advanced;
      }
  if (bad) {
    std::printf("  [D] %ld mismatches\n", bad);
    return false;
  }
  if (!advanced) {
    std::printf("  [D] dpx never advanced -- fixture degenerate\n");
    return false;
  }
  std::printf("  [D] ok (2 passes, %ld cells advanced)\n", advanced);
  return true;
}

// ===========================================================================
// Part E -- the w chain: K7, K8, K11, K12
// sw_core.F90:972-984 (dw + heat_source), :988-992 (mass update),
// :1250-1254 (normalize), :1256-1262 (add dw)
// ===========================================================================
// The anti-dependence under test is that K7 reads the PRE-transport w and K8
// reads the OLD delp while K11 divides by the NEW delp. The part therefore also
// builds a wrong-order reference in which K7 sees the post-mass-update w, and
// requires a difference. Levels 0 and 2 have damp_w below 1.E-5, so K7 and K12
// must skip them entirely -- which is also §4's mixed-branch requirement applied
// to the w path.
static bool part_e() {
  const Box b = seam_box();
  Dev d;
  Grid g;
  g.init(b);
  const dgrid::TransportBounds tb = tb_of(b);

  const std::vector<fv_real> damp_w = {fv_real(0), fv_real(0.06), fv_real(0),
                                       fv_real(0.09)};
  const std::vector<fv_real> kgb = {fv_real(0), fv_real(2.0e-4), fv_real(1.0e-4),
                                    fv_real(0)};
  const std::vector<fv_real> d_con = {fv_real(0), fv_real(1.0), fv_real(0.5),
                                      fv_real(0)};
  if (int64_t(damp_w.size()) != b.nk) {
    std::printf("  [E] level table does not match nk\n");
    return false;
  }

  A3 w, delp, fx2, fy2, gx, gy, dw, hs;
  w.init(b); delp.init(b); fx2.init(b); fy2.init(b); gx.init(b); gy.init(b);
  dw.init(b); hs.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        w(i, j, k) = f_w(i, j, k);
        delp(i, j, k) = f_delp(i, j, k);
        fx2(i, j, k) = fv_real(double(f_ua(i, j, k)) * 3.0e7);
        fy2(i, j, k) = fv_real(double(f_va(i, j, k)) * 3.0e7);
        gx(i, j, k) = fv_real(double(f_u(i, j, k)) * 1.0e7);
        gy(i, j, k) = fv_real(double(f_v(i, j, k)) * 1.0e7);
        dw(i, j, k) = f_sentinel(i, j, k);
        hs(i, j, k) = fv_real(0);  // K6 has already run in the real sequence
      }

  // --- host reference in Fortran order ------------------------------------
  A3 dw_r = dw, hs_r = hs, w_r = w, w_wrong = w;
  long active_levels = 0, skipped_levels = 0;
  for (int64_t k = 0; k < b.nk; ++k) {
    const bool damped = double(damp_w[size_t(k)]) > 1.0e-5;
    if (damped) ++active_levels; else ++skipped_levels;
    // K7
    if (damped) {
      const double dd8 = double(kgb[size_t(k)]) * std::fabs(DT);
      for (int64_t j = b.js; j <= b.je; ++j)
        for (int64_t i = b.is; i <= b.ie; ++i) {
          const double v = (double(fx2(i, j, k)) - double(fx2(i + 1, j, k)) +
                            double(fy2(i, j, k)) - double(fy2(i, j + 1, k))) *
                           double(g.rarea(i, j));
          dw_r(i, j, k) = fv_real(v);
          hs_r(i, j, k) = fv_real(dd8 - double(d_con[size_t(k)]) * v *
                                            (double(w(i, j, k)) + 0.5 * v));
        }
    }
    // K8 (OLD delp), then K11 (NEW delp is the same delp here because Part E
    // does not run K9; the point is which w feeds K7), then K12
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const double m = double(delp(i, j, k)) * double(w(i, j, k)) +
                         (double(gx(i, j, k)) - double(gx(i + 1, j, k)) +
                          double(gy(i, j, k)) - double(gy(i, j + 1, k))) *
                             double(g.rarea(i, j));
        double v = m / double(delp(i, j, k));
        if (damped) v += double(dw_r(i, j, k));
        w_r(i, j, k) = fv_real(v);
      }
  }
  // wrong order: K7's heat source evaluated against the POST-mass-update w
  A3 hs_wrong = hs;
  for (int64_t k = 0; k < b.nk; ++k) {
    if (!(double(damp_w[size_t(k)]) > 1.0e-5)) continue;
    const double dd8 = double(kgb[size_t(k)]) * std::fabs(DT);
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const double v = double(dw_r(i, j, k));
        const double wpost = double(delp(i, j, k)) * double(w(i, j, k)) +
                             (double(gx(i, j, k)) - double(gx(i + 1, j, k)) +
                              double(gy(i, j, k)) - double(gy(i, j + 1, k))) *
                                 double(g.rarea(i, j));
        hs_wrong(i, j, k) =
            fv_real(dd8 - double(d_con[size_t(k)]) * v * (wpost + 0.5 * v));
      }
  }
  (void)w_wrong;
  if (!active_levels || !skipped_levels) {
    std::printf("  [E] fixture degenerate: %ld damped / %ld undamped levels\n",
                active_levels, skipped_levels);
    return false;
  }

  fv_real *d_w = d.up(w), *d_dp = d.up(delp);
  fv_real *d_fx2 = d.up(fx2), *d_fy2 = d.up(fy2);
  fv_real *d_gx = d.up(gx), *d_gy = d.up(gy);
  fv_real *d_dw = d.up(dw), *d_hs = d.up(hs), *d_ra = d.up(g.rarea);
  fv_real *d_dampw = d.lev(damp_w), *d_kgb = d.lev(kgb), *d_dcon = d.lev(d_con);
  if (!d_dcon) {
    std::printf("  [E] device allocation failed\n");
    return false;
  }
  dgrid::LevelFilter lf{};
  lf.npz = b.nk;
  lf.mode = 1;
  lf.damp = VL(d_dampw);
  lf.gate = VL(d_dampw);

  fv3_gpu_status rc = dgrid::launch_dsw_w_damping_heat_cuda(
      tb, lf, fv_real(DT), VL(d_kgb), VL(d_dcon), V3<const fv_real>(d_w, b),
      V3<const fv_real>(d_fx2, b), V3<const fv_real>(d_fy2, b),
      V2<const fv_real>(d_ra, b), V3<fv_real>(d_dw, b), V3<fv_real>(d_hs, b), nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_dsw_w_mass_update_cuda(tb, V3<const fv_real>(d_dp, b),
                                              V3<const fv_real>(d_gx, b),
                                              V3<const fv_real>(d_gy, b),
                                              V2<const fv_real>(d_ra, b),
                                              V3<fv_real>(d_w, b), nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_dsw_w_normalize_cuda(tb, V3<const fv_real>(d_dp, b),
                                            V3<fv_real>(d_w, b), nullptr);
  if (rc == FV3_GPU_SUCCESS)
    rc = dgrid::launch_dsw_w_damping_add_cuda(tb, lf, V3<const fv_real>(d_dw, b),
                                              V3<fv_real>(d_w, b), nullptr);
  if (rc != FV3_GPU_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [E] launch/sync failed (%d)\n", int(rc));
    return false;
  }

  const A3 gw = down3(d_w, b), gdw = down3(d_dw, b), ghs = down3(d_hs, b);
  long bad = 0, differs = 0, inside_damped = 0;
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        if (!close_rel(gw(i, j, k), w_r(i, j, k), TOL) ||
            !close_rel(gdw(i, j, k), dw_r(i, j, k), TOL) ||
            !close_rel(ghs(i, j, k), hs_r(i, j, k), TOL)) {
          if (bad < 8)
            std::printf("  [E] i=%lld j=%lld k=%lld w %.12g/%.12g dw %.12g/%.12g hs %.12g/%.12g\n",
                        (long long)i, (long long)j, (long long)k, double(gw(i, j, k)),
                        double(w_r(i, j, k)), double(gdw(i, j, k)),
                        double(dw_r(i, j, k)), double(ghs(i, j, k)),
                        double(hs_r(i, j, k)));
          ++bad;
        }
        const bool inside = i >= b.is && i <= b.ie && j >= b.js && j <= b.je;
        if (inside && double(damp_w[size_t(k)]) > 1.0e-5) {
          ++inside_damped;
          if (!close_rel(ghs(i, j, k), hs_wrong(i, j, k), TOL)) ++differs;
        }
      }
  if (bad) {
    std::printf("  [E] %ld mismatches\n", bad);
    return false;
  }
  if (differs < inside_damped / 2) {
    std::printf("  [E] heat source does not discriminate w ordering (%ld/%ld)\n",
                differs, inside_damped);
    return false;
  }
  std::printf("  [E] ok (%ld damped / %ld undamped levels, rejects post-transport w "
              "in %ld/%ld cells)\n",
              active_levels, skipped_levels, differs, inside_damped);
  return true;
}

// ===========================================================================
// Part H -- derived per-level coefficient arrays
// ===========================================================================
// Checks the two DIFFERENT damping laws that d_sw uses side by side and that
// Milestone 3A.4/3A.5 warned must never be conflated:
//   fv_tp_2d's deln_flux : (damp*da_min  )**(nord+1), gate at 1.e-4
//   del6_vt_flux         : (damp*da_min_c)**(nord+1), gate at 1.E-5
// The damp table straddles BOTH thresholds (5e-6 below both, 5e-5 between them,
// 0.10 above both) so a swapped gate constant fails.
static bool part_h() {
  Box b = seam_box();
  b.nk = 5;
  Dev d;
  const dgrid::TransportBounds tb = tb_of(b);

  const std::vector<fv_real> damp_v = {fv_real(0.0), fv_real(5.0e-6), fv_real(5.0e-5),
                                       fv_real(0.10), fv_real(0.20)};
  const std::vector<fv_real> damp_w = {fv_real(0.06), fv_real(0.0), fv_real(5.0e-5),
                                       fv_real(0.09), fv_real(5.0e-6)};
  const std::vector<fv_real> damp_t = {fv_real(0.08), fv_real(5.0e-5), fv_real(0.0),
                                       fv_real(5.0e-6), fv_real(0.03)};
  const std::vector<fv_real> nord_v = {fv_real(0), fv_real(1), fv_real(2), fv_real(2),
                                       fv_real(1)};
  const std::vector<fv_real> nord_w = {fv_real(0), fv_real(2), fv_real(1), fv_real(2),
                                       fv_real(0)};
  const int32_t nord_t = 2;

  dgrid::DSwOptions opt{};
  opt.nord_t = nord_t;
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);

  dgrid::DSwLevels lv{};
  lv.damp_v = VL(d.lev(damp_v));
  lv.damp_w = VL(d.lev(damp_w));
  lv.damp_t = VL(d.lev(damp_t));
  lv.nord_v = VL(d.lev(nord_v));
  lv.nord_w = VL(d.lev(nord_w));

  dgrid::WpBDSwArena ws{};
  fv_real *p[7];
  for (int n = 0; n < 7; ++n) p[n] = d.levn(size_t(b.nk));
  if (!p[6]) {
    std::printf("  [H] device allocation failed\n");
    return false;
  }
  ws.deln_damp_delp = VLW(p[0]);
  ws.deln_gate_delp = VLW(p[1]);
  ws.deln_damp_pt = VLW(p[2]);
  ws.deln_gate_pt = VLW(p[3]);
  ws.nord_t_level = VLW(p[4]);
  ws.del6_damp_w = VLW(p[5]);
  ws.del6_damp_v = VLW(p[6]);

  const fv3_gpu_status rc =
      dgrid::launch_dsw_level_coefficients_cuda(tb, opt, lv, ws, nullptr);
  if (rc != FV3_GPU_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("  [H] launch/sync failed (%d)\n", int(rc));
    return false;
  }

  const auto ipow_h = [](fv_real x, int32_t n) {
    fv_real r = fv_real(1);
    for (int32_t i = 0; i < n; ++i) r = fv_real(r * x);
    return r;
  };
  const std::vector<fv_real> got[7] = {
      downl(p[0], size_t(b.nk)), downl(p[1], size_t(b.nk)), downl(p[2], size_t(b.nk)),
      downl(p[3], size_t(b.nk)), downl(p[4], size_t(b.nk)), downl(p[5], size_t(b.nk)),
      downl(p[6], size_t(b.nk))};
  const char *names[7] = {"deln_damp_delp", "deln_gate_delp", "deln_damp_pt",
                          "deln_gate_pt",   "nord_t_level",   "del6_damp_w",
                          "del6_damp_v"};
  long bad = 0, gates_on = 0, gates_off = 0;
  for (int64_t k = 0; k < b.nk; ++k) {
    const size_t s = size_t(k);
    const int32_t nv = int32_t(std::lround(double(nord_v[s])));
    const int32_t nw = int32_t(std::lround(double(nord_w[s])));
    fv_real want[7];
    want[0] = ipow_h(fv_real(damp_v[s] * fv_real(DA_MIN)), nv + 1);
    want[1] = damp_v[s] > fv_real(1.0e-4) ? fv_real(1) : fv_real(0);
    want[2] = ipow_h(fv_real(damp_t[s] * fv_real(DA_MIN)), nord_t + 1);
    want[3] = damp_t[s] > fv_real(1.0e-4) ? fv_real(1) : fv_real(0);
    want[4] = fv_real(nord_t);
    want[5] = ipow_h(fv_real(damp_w[s] * fv_real(DA_MIN_C)), nw + 1);
    want[6] = ipow_h(fv_real(damp_v[s] * fv_real(DA_MIN_C)), nv + 1);
    for (int n = 0; n < 7; ++n) {
      if (!close_rel(got[n][s], want[n], TOL)) {
        std::printf("  [H] k=%lld %s %.12g vs %.12g\n", (long long)k, names[n],
                    double(got[n][s]), double(want[n]));
        ++bad;
      }
    }
    if (want[1] != fv_real(0)) ++gates_on; else ++gates_off;
  }
  if (bad) {
    std::printf("  [H] %ld mismatches\n", bad);
    return false;
  }
  if (!gates_on || !gates_off) {
    std::printf("  [H] deln gate degenerate (%ld on, %ld off)\n", gates_on, gates_off);
    return false;
  }
  std::printf("  [H] ok (deln gate %ld on / %ld off across %lld levels)\n", gates_on,
              gates_off, (long long)b.nk);
  return true;
}

// ===========================================================================
// Full-driver fixture (Parts I..M)
// ===========================================================================
// dt for the full-driver parts is larger than Parts A-H's 30 s so that the
// prognostic increments sit several orders of magnitude above the R4 comparison
// tolerance while delp still moves by only ~0.01%, keeping it strictly
// positive for the pt divide.
static const double DT_FULL = 120.0;

// Per-level control tables. Every entry that dyn_core.F90:699-765 recomputes
// inside its k loop appears here; §4 of wp_b_dsw_cuda.cuh says which ones
// genuinely vary.
struct Levels {
  std::vector<fv_real> nord_k, nord_v, nord_w, d2_bg, d4_bg;
  std::vector<fv_real> damp_v, damp_w, damp_t, d_con, kgb;
};

// The production shape this milestone must support: a sponge top where nord_k
// is forced to 0 sitting above an interior that takes the higher-order arm, in
// ONE d_sw invocation. k=0 and k=2 take the nord == 0 arm, k=1 takes nord 2 and
// k=3 takes nord 1, so all three high-order groups plus the nord0 group are
// live simultaneously. damp_w, damp_v, damp_t and d_con are each on for some
// levels and off for others, so every LevelFilter gate has both arms live.
static Levels mixed_levels() {
  Levels L;
  L.nord_k = {fv_real(0), fv_real(2), fv_real(0), fv_real(1)};
  L.nord_v = {fv_real(0), fv_real(2), fv_real(1), fv_real(2)};
  L.nord_w = {fv_real(0), fv_real(1), fv_real(0), fv_real(2)};
  L.d2_bg = {fv_real(0.02), fv_real(0.0), fv_real(0.03), fv_real(0.0)};
  L.d4_bg = {fv_real(0.12), fv_real(0.16), fv_real(0.14), fv_real(0.10)};
  L.damp_v = {fv_real(0.0), fv_real(0.10), fv_real(0.05), fv_real(0.20)};
  L.damp_w = {fv_real(0.0), fv_real(0.06), fv_real(0.0), fv_real(0.09)};
  L.damp_t = {fv_real(0.0), fv_real(0.08), fv_real(0.0), fv_real(0.03)};
  L.d_con = {fv_real(0.0), fv_real(1.0), fv_real(0.5), fv_real(0.0)};
  // kgb is nonzero on k=2 and k=3 for a reason: k=3 has damp_w > 1.E-5 with
  // d_con == 0, so its heat_source must end the call EXACTLY equal to
  // kgb*abs(dt) (S10's d_con term vanishes and S37 skips the level). That is
  // the sharpest available check on the S9/S10/S37 heat-source ordering.
  L.kgb = {fv_real(0.0), fv_real(0.0), fv_real(1.0e-4), fv_real(5.0e-5)};
  return L;
}

// Replicate level `k` of `m` to every level: the uniform-parameter run whose
// level k must reproduce the mixed run's level k exactly.
static Levels uniform_from(const Levels &m, size_t k, size_t nk) {
  const auto rep = [&](const std::vector<fv_real> &s) {
    return std::vector<fv_real>(nk, s[k]);
  };
  Levels u;
  u.nord_k = rep(m.nord_k); u.nord_v = rep(m.nord_v); u.nord_w = rep(m.nord_w);
  u.d2_bg = rep(m.d2_bg);   u.d4_bg = rep(m.d4_bg);   u.damp_v = rep(m.damp_v);
  u.damp_w = rep(m.damp_w); u.damp_t = rep(m.damp_t); u.d_con = rep(m.d_con);
  u.kgb = rep(m.kgb);
  return u;
}

struct Full {
  Box b;
  Grid g;
  A2B ab;
  Dev d;
  int32_t force_nested = 0;
  // The driver requires an explicit stream: smag_corner reuses WP-D's
  // launch_a2b_ord4_global_cuda, which rejects a null stream. WP-E always
  // supplies one (validate_execution_context rejects a null submission stream),
  // so the fixture behaves like production rather than relying on the default
  // stream. `use_stream` lets Part L exercise the null-stream fence.
  cudaStream_t stream = nullptr;
  bool use_stream = true;

  // Host initial state, kept so every run can start from an identical state.
  A3 h_u, h_v, h_w, h_delp, h_pt, h_uc, h_vc, h_ua, h_va, h_zrat, h_zero;
  D3 h_acc;

  // Device state pointers the parts read back.
  fv_real *p_u = nullptr, *p_v = nullptr, *p_w = nullptr, *p_delp = nullptr;
  fv_real *p_pt = nullptr, *p_uc = nullptr, *p_vc = nullptr;
  fv_real *p_hs = nullptr, *p_de = nullptr;
  fv_real *p_dw = nullptr, *p_fx2 = nullptr, *p_fy2 = nullptr;
  accumulator_real *p_mfx = nullptr, *p_mfy = nullptr, *p_cx = nullptr;
  accumulator_real *p_cy = nullptr, *p_dpx = nullptr;
  fv_real *p_lv[10] = {};

  dgrid::DSwOptions opt{};
  dgrid::DSwLevels lv{};
  dgrid::DSwGeometry geo{};
  dgrid::DSwState st{};
  dgrid::WpBDSwArena ws{};

  dgrid::TransportBounds tb() const {
    dgrid::TransportBounds t = tb_of(b);
    t.nested = force_nested;
    return t;
  }

  bool alloc(const Box &bx);
  bool seed_state();
  bool set_levels(const Levels &L);
  bool seed_accumulators(accumulator_real bias);
  ~Full() {
    if (stream) cudaStreamDestroy(stream);
  }
  fv3_gpu_status launch() {
    return dgrid::launch_d_sw_cuda(tb(), opt, lv, geo, st, ws,
                                   use_stream ? static_cast<void *>(stream) : nullptr);
  }
  bool run() {
    const fv3_gpu_status rc = launch();
    if (rc != FV3_GPU_SUCCESS) {
      std::printf("      launch_d_sw_cuda -> %d\n", int(rc));
      return false;
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      std::printf("      sync -> %s\n", cudaGetErrorString(e));
      return false;
    }
    return true;
  }
};

bool Full::alloc(const Box &bx) {
  b = bx;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::printf("      cudaStreamCreate failed\n");
    return false;
  }
  g.init(b);
  ab.init(b);

  h_u.init(b); h_v.init(b); h_w.init(b); h_delp.init(b); h_pt.init(b);
  h_uc.init(b); h_vc.init(b); h_ua.init(b); h_va.init(b); h_zrat.init(b);
  h_zero.init(b);
  h_acc.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) {
        h_u(i, j, k) = f_u(i, j, k);
        h_v(i, j, k) = f_v(i, j, k);
        h_w(i, j, k) = f_w(i, j, k);
        h_delp(i, j, k) = f_delp(i, j, k);
        h_pt(i, j, k) = f_pt(i, j, k);
        h_uc(i, j, k) = f_uc(i, j, k);
        h_vc(i, j, k) = f_vc(i, j, k);
        h_ua(i, j, k) = f_ua(i, j, k);
        h_va(i, j, k) = f_va(i, j, k);
        h_zrat(i, j, k) = fv_real(1);
        // Distinct nonzero accumulator seeds: d_sw must never zero these.
        h_acc(i, j, k) = 100.0 + 0.5 * double(i) + 7.0 * double(j) + 313.0 * double(k);
      }

  // Geometry.
  geo.div = dg_of(d, b, g);
  geo.a2b = a2b_of(d, b, ab, g);
  geo.area = V2<const fv_real>(d.up(g.area), b);
  geo.dxa = V2<const fv_real>(d.up(g.dxa), b);
  geo.dya = V2<const fv_real>(d.up(g.dya), b);
  geo.rdxa = V2<const fv_real>(d.up(g.rdxa), b);
  geo.rdya = V2<const fv_real>(d.up(g.rdya), b);
  geo.del6_u = V2<const fv_real>(d.up(g.del6_u), b);
  geo.del6_v = V2<const fv_real>(d.up(g.del6_v), b);
  geo.cosa = V2<const fv_real>(d.up(g.cosa), b);
  geo.rsina = V2<const fv_real>(d.up(g.rsina), b);
  geo.rsin_u = V2<const fv_real>(d.up(g.rsin_u), b);
  geo.rsin_v = V2<const fv_real>(d.up(g.rsin_v), b);
  geo.f0 = V2<const fv_real>(d.up(g.f0), b);

  // State.
  p_u = d.scratch(b); p_v = d.scratch(b); p_w = d.scratch(b);
  p_delp = d.scratch(b); p_pt = d.scratch(b);
  p_uc = d.scratch(b); p_vc = d.scratch(b);
  fv_real *p_ua = d.scratch(b), *p_va = d.scratch(b), *p_zrat = d.scratch(b);
  p_hs = d.scratch(b); p_de = d.scratch(b);
  fv_real *p_divgd = d.scratch(b), *p_delpc = d.scratch(b), *p_ptc = d.scratch(b);
  fv_real *p_ut = d.scratch(b), *p_vt = d.scratch(b);
  fv_real *p_crx = d.scratch(b), *p_cry = d.scratch(b);
  fv_real *p_xfx = d.scratch(b), *p_yfx = d.scratch(b);
  const size_t abytes = static_cast<size_t>(b.n3()) * sizeof(accumulator_real);
  p_mfx = static_cast<accumulator_real *>(d.raw(abytes));
  p_mfy = static_cast<accumulator_real *>(d.raw(abytes));
  p_cx = static_cast<accumulator_real *>(d.raw(abytes));
  p_cy = static_cast<accumulator_real *>(d.raw(abytes));
  p_dpx = static_cast<accumulator_real *>(d.raw(abytes));
  if (!p_dpx) return false;

  st.u = V3<fv_real>(p_u, b);         st.v = V3<fv_real>(p_v, b);
  st.w = V3<fv_real>(p_w, b);         st.delp = V3<fv_real>(p_delp, b);
  st.pt = V3<fv_real>(p_pt, b);       st.uc = V3<fv_real>(p_uc, b);
  st.vc = V3<fv_real>(p_vc, b);
  st.ua = V3<const fv_real>(p_ua, b); st.va = V3<const fv_real>(p_va, b);
  st.divgd = V3<fv_real>(p_divgd, b); st.delpc = V3<fv_real>(p_delpc, b);
  st.ptc = V3<fv_real>(p_ptc, b);
  st.ut = V3<fv_real>(p_ut, b);       st.vt = V3<fv_real>(p_vt, b);
  st.crx = V3<fv_real>(p_crx, b);     st.cry = V3<fv_real>(p_cry, b);
  st.xfx = V3<fv_real>(p_xfx, b);     st.yfx = V3<fv_real>(p_yfx, b);
  st.heat_source = V3<fv_real>(p_hs, b);
  st.diss_est = V3<fv_real>(p_de, b);
  st.z_rat = V3<const fv_real>(p_zrat, b);
  st.mfx = V3<accumulator_real>(p_mfx, b);
  st.mfy = V3<accumulator_real>(p_mfy, b);
  st.cx = V3<accumulator_real>(p_cx, b);
  st.cy = V3<accumulator_real>(p_cy, b);
  st.dpx = V3<accumulator_real>(p_dpx, b);

  // ua/va/z_rat never change, so they are uploaded once here rather than in
  // seed_state.
  const size_t nb = static_cast<size_t>(b.n3()) * sizeof(fv_real);
  cudaMemcpy(p_ua, h_ua.d.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(p_va, h_va.d.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(p_zrat, h_zrat.d.data(), nb, cudaMemcpyHostToDevice);

  // Arena: 14 d_sw locals + 7 transport scratch + 3 PPM real + 1 PPM logical +
  // 8 smag privates + 7 level arrays (WpBDSwArena::array_count == 34).
  ws.ub = V3<fv_real>(d.scratch(b), b);   ws.vb = V3<fv_real>(d.scratch(b), b);
  ws.wk = V3<fv_real>(d.scratch(b), b);   ws.ke = V3<fv_real>(d.scratch(b), b);
  ws.vort = V3<fv_real>(d.scratch(b), b);
  ws.fx = V3<fv_real>(d.scratch(b), b);   ws.fy = V3<fv_real>(d.scratch(b), b);
  ws.gx = V3<fv_real>(d.scratch(b), b);   ws.gy = V3<fv_real>(d.scratch(b), b);
  ws.ra_x = V3<fv_real>(d.scratch(b), b); ws.ra_y = V3<fv_real>(d.scratch(b), b);
  p_fx2 = d.scratch(b); p_fy2 = d.scratch(b); p_dw = d.scratch(b);
  ws.fx2 = V3<fv_real>(p_fx2, b); ws.fy2 = V3<fv_real>(p_fy2, b);
  ws.dw = V3<fv_real>(p_dw, b);
  ws.q_i = V3<fv_real>(d.scratch(b), b);    ws.q_j = V3<fv_real>(d.scratch(b), b);
  ws.fx2_tp = V3<fv_real>(d.scratch(b), b); ws.fy2_tp = V3<fv_real>(d.scratch(b), b);
  ws.d2 = V3<fv_real>(d.scratch(b), b);
  ws.fx2_dn = V3<fv_real>(d.scratch(b), b); ws.fy2_dn = V3<fv_real>(d.scratch(b), b);
  ws.bl = V3<fv_real>(d.scratch(b), b);     ws.br = V3<fv_real>(d.scratch(b), b);
  ws.b0 = V3<fv_real>(d.scratch(b), b);
  ws.smt = V3<int32_t>(d.iscratch(b), b);
  ws.smag.ut = V3<fv_real>(d.scratch(b), b);  ws.smag.vt = V3<fv_real>(d.scratch(b), b);
  ws.smag.wk = V3<fv_real>(d.scratch(b), b);  ws.smag.sh = V3<fv_real>(d.scratch(b), b);
  ws.smag.qx = V3<fv_real>(d.scratch(b), b);  ws.smag.qy = V3<fv_real>(d.scratch(b), b);
  ws.smag.qxx = V3<fv_real>(d.scratch(b), b); ws.smag.qyy = V3<fv_real>(d.scratch(b), b);
  fv_real *wl[7];
  for (int n = 0; n < 7; ++n) wl[n] = d.levn(static_cast<size_t>(b.nk));
  if (!wl[6]) return false;
  ws.deln_damp_delp = VLW(wl[0]); ws.deln_gate_delp = VLW(wl[1]);
  ws.deln_damp_pt = VLW(wl[2]);   ws.deln_gate_pt = VLW(wl[3]);
  ws.nord_t_level = VLW(wl[4]);
  ws.del6_damp_w = VLW(wl[5]);    ws.del6_damp_v = VLW(wl[6]);

  // Per-level control buffers.
  for (int n = 0; n < 10; ++n) p_lv[n] = d.levn(static_cast<size_t>(b.nk));
  if (!p_lv[9]) return false;
  lv.nord_k = VL(p_lv[0]); lv.nord_v = VL(p_lv[1]); lv.nord_w = VL(p_lv[2]);
  lv.d2_bg = VL(p_lv[3]);  lv.d4_bg = VL(p_lv[4]);  lv.damp_v = VL(p_lv[5]);
  lv.damp_w = VL(p_lv[6]); lv.damp_t = VL(p_lv[7]); lv.d_con = VL(p_lv[8]);
  lv.kgb = VL(p_lv[9]);

  // Production first-run configuration (project_architecture.md).
  opt = dgrid::DSwOptions{};
  opt.dt = fv_real(DT_FULL);
  opt.hord_mt = 6; opt.hord_vt = 6; opt.hord_tm = 6; opt.hord_dp = -6;
  opt.lim_fac = fv_real(1);
  opt.nord = 2;
  opt.nord_t = 2;
  opt.dddmp = fv_real(0.5);
  opt.da_min = fv_real(DA_MIN);
  opt.da_min_c = fv_real(DA_MIN_C);
  opt.grid_type = 0;
  opt.hydrostatic = 0;
  opt.a_imp = 1.0;
  return true;
}

bool Full::seed_state() {
  const size_t nb = static_cast<size_t>(b.n3()) * sizeof(fv_real);
  struct Pair { fv_real *dst; const A3 *src; };
  const Pair pairs[] = {
      {p_u, &h_u}, {p_v, &h_v}, {p_w, &h_w}, {p_delp, &h_delp}, {p_pt, &h_pt},
      {p_uc, &h_uc}, {p_vc, &h_vc}, {p_hs, &h_zero}, {p_de, &h_zero}};
  for (const Pair &p : pairs)
    if (cudaMemcpy(p.dst, p.src->d.data(), nb, cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  // Sentinel-fill dw so an undamped level that wrongly reaches K12 shows up as
  // a huge w rather than as round-off.
  A3 sent;
  sent.init(b);
  for (int64_t k = 0; k < b.nk; ++k)
    for (int64_t j = b.loj(); j <= b.hij(); ++j)
      for (int64_t i = b.loi(); i <= b.hii(); ++i) sent(i, j, k) = f_sentinel(i, j, k);
  return cudaMemcpy(p_dw, sent.d.data(), nb, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool Full::seed_accumulators(accumulator_real bias) {
  D3 s = h_acc;
  for (size_t n = 0; n < s.d.size(); ++n) s.d[n] += bias;
  const size_t nb = s.d.size() * sizeof(accumulator_real);
  accumulator_real *const dst[5] = {p_mfx, p_mfy, p_cx, p_cy, p_dpx};
  for (accumulator_real *p : dst)
    if (cudaMemcpy(p, s.d.data(), nb, cudaMemcpyHostToDevice) != cudaSuccess) return false;
  return true;
}

bool Full::set_levels(const Levels &L) {
  const std::vector<fv_real> *src[10] = {&L.nord_k, &L.nord_v, &L.nord_w, &L.d2_bg,
                                         &L.d4_bg,  &L.damp_v, &L.damp_w, &L.damp_t,
                                         &L.d_con,  &L.kgb};
  for (int n = 0; n < 10; ++n) {
    if (int64_t(src[n]->size()) != b.nk) return false;
    if (cudaMemcpy(p_lv[n], src[n]->data(), src[n]->size() * sizeof(fv_real),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// slab helpers for the full-driver parts
// ---------------------------------------------------------------------------
static long slab_mismatch(const A3 &x, const A3 &y, const Box &b, int64_t k, double tol,
                          const char *what, int limit) {
  long bad = 0;
  for (int64_t j = b.loj(); j <= b.hij(); ++j)
    for (int64_t i = b.loi(); i <= b.hii(); ++i)
      if (!close_rel(x(i, j, k), y(i, j, k), tol)) {
        if (bad < limit)
          std::printf("      %s k=%lld i=%lld j=%lld %.12g vs %.12g\n", what,
                      (long long)k, (long long)i, (long long)j, double(x(i, j, k)),
                      double(y(i, j, k)));
        ++bad;
      }
  return bad;
}

static long slab_differs(const A3 &x, const A3 &y, const Box &b, int64_t k, double tol) {
  long n = 0;
  for (int64_t j = b.loj(); j <= b.hij(); ++j)
    for (int64_t i = b.loi(); i <= b.hii(); ++i)
      if (!close_rel(x(i, j, k), y(i, j, k), tol)) ++n;
  return n;
}

// ===========================================================================
// Part I -- mixed per-level damping branches in ONE launch_d_sw_cuda call
// ===========================================================================
// d_sw is invoked per k by dyn_core's Fortran loop, so every CUDA level must be
// computable independently. That gives an exact, INDEPENDENT invariant for the
// per-level control flow §4 requires:
//
//   the mixed run's level k  ==  the run with every level set to level k's
//                                parameters, at level k
//
// The mixed table puts the nord_k == 0 arm on k=0 and k=2 and the higher-order
// arm on k=1 (nord 2) and k=3 (nord 1), so all four divergence-damping groups
// plus both arms of every damp_v / damp_w / damp_t / d_con gate are live in the
// SAME invocation. A driver that collapsed the per-level selectors to a scalar,
// or that leaked one level's coefficient into another, fails.
//
// The part also proves the invariant is not vacuous: for every k there must
// exist a k' whose uniform run gives a DIFFERENT answer at level k. Without
// that check, a driver that ignored the level tables entirely would pass.
static bool part_i() {
  Full f;
  if (!f.alloc(seam_box())) {
    std::printf("  [I] fixture allocation failed\n");
    return false;
  }
  const int64_t nk = f.b.nk;
  const Levels M = mixed_levels();

  if (!f.seed_state() || !f.set_levels(M) || !f.seed_accumulators(0.0)) {
    std::printf("  [I] seeding failed\n");
    return false;
  }
  if (!f.run()) return false;
  const A3 mu = down3(f.p_u, f.b), mv = down3(f.p_v, f.b), mw = down3(f.p_w, f.b);
  const A3 mdp = down3(f.p_delp, f.b), mpt = down3(f.p_pt, f.b);
  const A3 mhs = down3(f.p_hs, f.b);

  std::vector<A3> uu, uv, uw, udp, upt, uhs;
  for (int64_t k = 0; k < nk; ++k) {
    const Levels U = uniform_from(M, static_cast<size_t>(k), static_cast<size_t>(nk));
    if (!f.seed_state() || !f.set_levels(U) || !f.seed_accumulators(0.0)) {
      std::printf("  [I] reseeding failed at k=%lld\n", (long long)k);
      return false;
    }
    if (!f.run()) return false;
    uu.push_back(down3(f.p_u, f.b));
    uv.push_back(down3(f.p_v, f.b));
    uw.push_back(down3(f.p_w, f.b));
    udp.push_back(down3(f.p_delp, f.b));
    upt.push_back(down3(f.p_pt, f.b));
    uhs.push_back(down3(f.p_hs, f.b));
  }

  long bad = 0;
  for (int64_t k = 0; k < nk; ++k) {
    const size_t s = static_cast<size_t>(k);
    bad += slab_mismatch(mu, uu[s], f.b, k, TOL, "u", 3);
    bad += slab_mismatch(mv, uv[s], f.b, k, TOL, "v", 3);
    bad += slab_mismatch(mw, uw[s], f.b, k, TOL, "w", 3);
    bad += slab_mismatch(mdp, udp[s], f.b, k, TOL, "delp", 3);
    bad += slab_mismatch(mpt, upt[s], f.b, k, TOL, "pt", 3);
    bad += slab_mismatch(mhs, uhs[s], f.b, k, TOL, "heat_source", 3);
  }
  if (bad) {
    std::printf("  [I] %ld per-level equivalence mismatches\n", bad);
    return false;
  }

  for (int64_t k = 0; k < nk; ++k) {
    long seen = 0;
    for (int64_t k2 = 0; k2 < nk; ++k2) {
      if (k2 == k) continue;
      const size_t a = static_cast<size_t>(k), c = static_cast<size_t>(k2);
      seen += slab_differs(uu[a], uu[c], f.b, k, TOL) +
              slab_differs(uw[a], uw[c], f.b, k, TOL) +
              slab_differs(uhs[a], uhs[c], f.b, k, TOL);
    }
    if (!seen) {
      std::printf("  [I] level %lld is insensitive to its own damping table -- the "
                  "equivalence check is vacuous there\n",
                  (long long)k);
      return false;
    }
  }
  std::printf("  [I] ok (mixed nord_k {0,2,0,1} reproduces 4 uniform runs level by "
              "level; all levels parameter-sensitive)\n");
  return true;
}

// ===========================================================================
// Part J -- flux capacitors across two invocations
// ===========================================================================
// cx, cy, xflux, yflux and dpx are intent(INOUT) kind=8 accumulators that carry
// state across acoustic substeps; dyn_core zeroes them once per k_split and
// d_sw must never touch that. The test runs the driver twice from the SAME
// initial prognostic state but with DIFFERENT accumulator seeds and requires the
// increments to be identical -- which holds only for a pure `+=` with no
// initialization, no coefficient and no sign flip. It also requires the
// increment to be exactly zero outside each accumulator's own Fortran
// rectangle (sw_core.F90:944-960, :1050-1054).
static bool part_j() {
  Full f;
  if (!f.alloc(seam_box())) {
    std::printf("  [J] fixture allocation failed\n");
    return false;
  }
  const Levels M = mixed_levels();
  const accumulator_real BIAS = 4096.0;

  D3 c0[5], c1[5], c2[5];
  accumulator_real *const src[5] = {f.p_cx, f.p_cy, f.p_mfx, f.p_mfy, f.p_dpx};
  const char *names[5] = {"cx", "cy", "xflux", "yflux", "dpx"};

  for (int pass = 0; pass < 2; ++pass) {
    const accumulator_real bias = pass ? BIAS : accumulator_real(0);
    if (!f.seed_state() || !f.set_levels(M) || !f.seed_accumulators(bias)) {
      std::printf("  [J] seeding failed\n");
      return false;
    }
    for (int n = 0; n < 5; ++n) c0[n] = downd(src[n], f.b);
    if (!f.run()) return false;
    for (int n = 0; n < 5; ++n) {
      D3 got = downd(src[n], f.b);
      D3 &dst = pass ? c2[n] : c1[n];
      dst = got;
      for (size_t m = 0; m < dst.d.size(); ++m) dst.d[m] -= c0[n].d[m];
    }
  }

  long bad = 0;
  for (int n = 0; n < 5; ++n) {
    long nonzero = 0, outside = 0;
    for (int64_t k = 0; k < f.b.nk; ++k)
      for (int64_t j = f.b.loj(); j <= f.b.hij(); ++j)
        for (int64_t i = f.b.loi(); i <= f.b.hii(); ++i) {
          const double a = c1[n](i, j, k), c = c2[n](i, j, k);
          if (!close_rel(a, c, 1.0e-12)) {
            if (bad < 6)
              std::printf("      %s increment seed-dependent at i=%lld j=%lld k=%lld: "
                          "%.17g vs %.17g\n",
                          names[n], (long long)i, (long long)j, (long long)k, a, c);
            ++bad;
          }
          if (a != 0.0) ++nonzero;
          bool in_rect = false;
          switch (n) {
            case 0: in_rect = i >= f.b.is && i <= f.b.ie + 1 && j >= f.b.jsd() &&
                              j <= f.b.jed(); break;
            case 1: in_rect = i >= f.b.isd() && i <= f.b.ied() && j >= f.b.js &&
                              j <= f.b.je + 1; break;
            case 2: in_rect = i >= f.b.is && i <= f.b.ie + 1 && j >= f.b.js &&
                              j <= f.b.je; break;
            case 3: in_rect = i >= f.b.is && i <= f.b.ie && j >= f.b.js &&
                              j <= f.b.je + 1; break;
            default: in_rect = i >= f.b.is && i <= f.b.ie && j >= f.b.js &&
                               j <= f.b.je; break;
          }
          if (!in_rect && a != 0.0) {
            if (outside < 4)
              std::printf("      %s advanced OUTSIDE its rectangle at i=%lld j=%lld "
                          "k=%lld by %.17g\n",
                          names[n], (long long)i, (long long)j, (long long)k, a);
            ++outside;
          }
        }
    if (outside) {
      std::printf("  [J] %s wrote outside its Fortran rectangle in %ld cells\n", names[n],
                  outside);
      ++bad;
    }
    if (!nonzero) {
      std::printf("  [J] %s never advanced -- fixture degenerate\n", names[n]);
      ++bad;
    }
  }
  if (bad) {
    std::printf("  [J] %ld capacitor failures\n", bad);
    return false;
  }
  std::printf("  [J] ok (5 accumulators, seed-independent increments, rectangles "
              "respected)\n");
  return true;
}

// ===========================================================================
// Part K -- end-to-end anti-dependence and ordering invariants
// ===========================================================================
// Three invariants that survive to the END of the driver and that only a
// correctly ordered d_sw satisfies:
//
//  K1. delp_final - delp_initial == dpx_final - dpx_initial
//      Both equal div(fx,fy)*rarea from the S7 delp transport (sw_core.F90:1041
//      and :1050). S34 later overwrites fx/fy with the vorticity fluxes, so this
//      holds only if S15 and S16 both ran between S7 and S34.
//      Tolerance is loose because the left side is a catastrophic cancellation
//      of two ~3000 Pa values differing by ~0.3 Pa -- in R4 that costs ~4
//      digits. It still discriminates: a wrong-flux delp update is O(1) off.
//
//  K2. dw_final == div(fx2_final, fy2_final)*rarea on the damp_w > 1.E-5 levels
//      (nothing after S10 writes fx2/fy2), and dw_final is still the SENTINEL on
//      the undamped levels -- with w finite everywhere, which proves K12 never
//      added the sentinel.
//
//  K3. heat_source == kgb*abs(dt) EXACTLY on a level with damp_w > 1.E-5 and
//      d_con == 0 (S10's d_con term vanishes, S37 skips the level), and exactly
//      0 on a level with damp_w == 0 and d_con == 0 (only S9 touches it).
static bool part_k() {
  Full f;
  if (!f.alloc(seam_box())) {
    std::printf("  [K] fixture allocation failed\n");
    return false;
  }
  const Levels M = mixed_levels();
  if (!f.seed_state() || !f.set_levels(M) || !f.seed_accumulators(0.0)) {
    std::printf("  [K] seeding failed\n");
    return false;
  }
  const D3 dpx0 = downd(f.p_dpx, f.b);
  if (!f.run()) return false;

  const A3 dp1 = down3(f.p_delp, f.b), w1 = down3(f.p_w, f.b);
  const A3 dw1 = down3(f.p_dw, f.b), hs1 = down3(f.p_hs, f.b);
  const A3 fx2 = down3(f.p_fx2, f.b), fy2 = down3(f.p_fy2, f.b);
  const D3 dpx1 = downd(f.p_dpx, f.b);

  // K1
  const double tol_k1 = sizeof(fv_real) == 4 ? 5.0e-3 : 1.0e-9;
  long bad = 0, checked = 0;
  for (int64_t k = 0; k < f.b.nk; ++k)
    for (int64_t j = f.b.js; j <= f.b.je; ++j)
      for (int64_t i = f.b.is; i <= f.b.ie; ++i) {
        const double a = double(dp1(i, j, k)) - double(f.h_delp(i, j, k));
        const double c = dpx1(i, j, k) - dpx0(i, j, k);
        ++checked;
        if (!close_rel(a, c, tol_k1)) {
          if (bad < 6)
            std::printf("      K1 i=%lld j=%lld k=%lld ddelp %.12g vs ddpx %.12g\n",
                        (long long)i, (long long)j, (long long)k, a, c);
          ++bad;
        }
      }
  if (bad) {
    std::printf("  [K] %ld/%ld delp-vs-dpx mismatches\n", bad, checked);
    return false;
  }

  // K2
  long damped_checked = 0, sentinel_checked = 0;
  for (int64_t k = 0; k < f.b.nk; ++k) {
    const bool damped = double(M.damp_w[static_cast<size_t>(k)]) > 1.0e-5;
    for (int64_t j = f.b.js; j <= f.b.je; ++j)
      for (int64_t i = f.b.is; i <= f.b.ie; ++i) {
        if (damped) {
          const double want = (double(fx2(i, j, k)) - double(fx2(i + 1, j, k)) +
                               double(fy2(i, j, k)) - double(fy2(i, j + 1, k))) *
                              double(f.g.rarea(i, j));
          ++damped_checked;
          if (!close_rel(dw1(i, j, k), want, TOL)) {
            if (bad < 6)
              std::printf("      K2 dw i=%lld j=%lld k=%lld %.12g vs %.12g\n",
                          (long long)i, (long long)j, (long long)k, double(dw1(i, j, k)),
                          want);
            ++bad;
          }
        } else {
          ++sentinel_checked;
          if (dw1(i, j, k) != f_sentinel(i, j, k)) {
            if (bad < 6)
              std::printf("      K2 dw sentinel overwritten at i=%lld j=%lld k=%lld "
                          "(%.12g)\n",
                          (long long)i, (long long)j, (long long)k, double(dw1(i, j, k)));
            ++bad;
          }
        }
        if (!std::isfinite(double(w1(i, j, k)))) {
          if (bad < 6)
            std::printf("      K2 w non-finite at i=%lld j=%lld k=%lld\n", (long long)i,
                        (long long)j, (long long)k);
          ++bad;
        }
        if (double(w1(i, j, k)) < -1.0e3) {
          if (bad < 6)
            std::printf("      K2 w looks like the dw sentinel at i=%lld j=%lld k=%lld "
                        "(%.12g)\n",
                        (long long)i, (long long)j, (long long)k, double(w1(i, j, k)));
          ++bad;
        }
      }
  }
  if (!damped_checked || !sentinel_checked) {
    std::printf("  [K] fixture degenerate: %ld damped / %ld undamped w cells\n",
                damped_checked, sentinel_checked);
    return false;
  }
  if (bad) {
    std::printf("  [K] %ld dw/w failures\n", bad);
    return false;
  }

  // K3
  long k3 = 0;
  for (int64_t k = 0; k < f.b.nk; ++k) {
    const size_t s = static_cast<size_t>(k);
    const bool damped = double(M.damp_w[s]) > 1.0e-5;
    const bool dcon = double(M.d_con[s]) > 1.0e-5;
    if (dcon) continue;
    const double want = damped ? double(M.kgb[s]) * std::fabs(DT_FULL) : 0.0;
    ++k3;
    for (int64_t j = f.b.js; j <= f.b.je; ++j)
      for (int64_t i = f.b.is; i <= f.b.ie; ++i)
        if (!close_rel(hs1(i, j, k), want, TOL)) {
          if (bad < 6)
            std::printf("      K3 heat_source i=%lld j=%lld k=%lld %.12g vs %.12g\n",
                        (long long)i, (long long)j, (long long)k, double(hs1(i, j, k)),
                        want);
          ++bad;
        }
  }
  if (k3 < 2) {
    std::printf("  [K] only %ld d_con-free levels -- K3 needs both a damped and an "
                "undamped one\n",
                k3);
    return false;
  }
  if (bad) {
    std::printf("  [K] %ld heat_source failures\n", bad);
    return false;
  }
  std::printf("  [K] ok (delp==dpx increments over %ld cells; dw from surviving fx2/fy2; "
              "heat_source exact on %ld d_con-free levels)\n",
              checked, k3);
  return true;
}

// ===========================================================================
// Part L -- unsupported-configuration rejection, before any state mutation
// ===========================================================================
// Every fence in wp_b_dsw_cuda.cuh §6 plus the missing-view case. The part also
// verifies that a rejected call leaves the prognostic state BITWISE unchanged,
// which is the whole reason the fences precede the first launch.
static bool part_l() {
  Full f;
  if (!f.alloc(seam_box())) {
    std::printf("  [L] fixture allocation failed\n");
    return false;
  }
  const Levels M = mixed_levels();
  if (!f.seed_state() || !f.set_levels(M) || !f.seed_accumulators(0.0)) {
    std::printf("  [L] seeding failed\n");
    return false;
  }
  const dgrid::DSwOptions base = f.opt;
  const dgrid::DSwState base_st = f.st;

  struct Case {
    const char *name;
    fv3_gpu_status want;
  };
  long bad = 0;
  const auto expect = [&](const char *name, fv3_gpu_status want) {
    const fv3_gpu_status rc = f.launch();
    if (rc != want) {
      std::printf("      %s -> %d, expected %d\n", name, int(rc), int(want));
      ++bad;
      return;
    }
    // Nothing may have been enqueued, so nothing can be in flight either.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::printf("      %s left a CUDA error behind\n", name);
      ++bad;
      return;
    }
    const A3 u = down3(f.p_u, f.b), dp = down3(f.p_delp, f.b);
    for (size_t n = 0; n < u.d.size(); ++n)
      if (u.d[n] != f.h_u.d[n] || dp.d[n] != f.h_delp.d[n]) {
        std::printf("      %s mutated state before rejecting\n", name);
        ++bad;
        return;
      }
  };

  f.opt = base; f.opt.grid_type = 1;
  expect("grid_type=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.force_nested = 1;
  expect("nested=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.force_nested = 0;
  f.opt = base; f.opt.hydrostatic = 1;
  expect("hydrostatic=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.inline_q = 1;
  expect("inline_q=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.use_cond = 1;
  expect("use_cond=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.do_f3d = 1;
  expect("do_f3d=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.do_skeb = 1;
  expect("do_skeb=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.rot3 = 1;
  expect("rot3=1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.a_imp = 0.5;
  expect("a_imp=0.5", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.hord_mt = 8;
  expect("hord_mt=8", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.hord_dp = -8;
  expect("hord_dp=-8", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.hord_vt = 5;
  expect("hord_vt=5", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.nord = 4;
  expect("nord=4", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.nord = -1;
  expect("nord=-1", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.opt = base; f.opt.nord_t = 3;
  expect("nord_t=3", FV3_GPU_UNSUPPORTED_CONFIGURATION);

  f.opt = base; f.use_stream = false;
  expect("stream=nullptr", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.use_stream = true;

  f.opt = base;
  const int64_t npx = f.b.npx;
  f.b.npx = 0;
  expect("npx=0", FV3_GPU_UNSUPPORTED_CONFIGURATION);
  f.b.npx = npx;

  f.st.u = View3<fv_real>{};
  expect("u view missing", FV3_GPU_INVALID_ARGUMENT);
  f.st = base_st;
  f.ws.dw = View3<fv_real>{};
  expect("arena dw missing", FV3_GPU_INVALID_ARGUMENT);
  f.ws.dw = V3<fv_real>(f.p_dw, f.b);
  f.geo.f0 = View2<const fv_real>{};
  expect("f0 view missing", FV3_GPU_INVALID_ARGUMENT);

  if (bad) {
    std::printf("  [L] %ld fence failures\n", bad);
    return false;
  }
  std::printf("  [L] ok (19 rejections, none mutated state)\n");
  return true;
}

// ===========================================================================
// Part M -- direct launch_d_sw_cuda GPU execution, seam and interior
// ===========================================================================
// GPU EXECUTED evidence for the complete driver: every prognostic field finite,
// delp strictly positive so the pt divide is well posed, every prognostic field
// actually moved, and a second consecutive call advancing the state further.
// This is an execution and sanity part, NOT a numerical-equivalence part.
static bool part_m_one(const Box &box, const char *tag) {
  Full f;
  if (!f.alloc(box)) {
    std::printf("  [M/%s] fixture allocation failed\n", tag);
    return false;
  }
  const Levels M = mixed_levels();
  if (!f.seed_state() || !f.set_levels(M) || !f.seed_accumulators(0.0)) {
    std::printf("  [M/%s] seeding failed\n", tag);
    return false;
  }
  if (!f.run()) return false;

  struct Fld { const char *name; A3 now; const A3 *was; };
  Fld flds[5] = {{"u", down3(f.p_u, f.b), &f.h_u},
                 {"v", down3(f.p_v, f.b), &f.h_v},
                 {"w", down3(f.p_w, f.b), &f.h_w},
                 {"delp", down3(f.p_delp, f.b), &f.h_delp},
                 {"pt", down3(f.p_pt, f.b), &f.h_pt}};
  long bad = 0;
  for (Fld &q : flds) {
    long moved = 0, cells = 0;
    for (int64_t k = 0; k < f.b.nk; ++k)
      for (int64_t j = f.b.js; j <= f.b.je; ++j)
        for (int64_t i = f.b.is; i <= f.b.ie; ++i) {
          const double v = double(q.now(i, j, k));
          ++cells;
          if (!std::isfinite(v)) {
            if (bad < 6)
              std::printf("      %s non-finite at i=%lld j=%lld k=%lld\n", q.name,
                          (long long)i, (long long)j, (long long)k);
            ++bad;
          }
          if (!close_rel(v, double((*q.was)(i, j, k)), 1.0e-9)) ++moved;
        }
    if (moved * 2 < cells) {
      std::printf("  [M/%s] %s changed in only %ld of %ld cells -- update is "
                  "degenerate\n",
                  tag, q.name, moved, cells);
      ++bad;
    }
    if (std::strcmp(q.name, "delp") == 0)
      for (int64_t k = 0; k < f.b.nk; ++k)
        for (int64_t j = f.b.js; j <= f.b.je; ++j)
          for (int64_t i = f.b.is; i <= f.b.ie; ++i)
            if (!(double(q.now(i, j, k)) > 0.0)) {
              if (bad < 6)
                std::printf("      delp non-positive at i=%lld j=%lld k=%lld (%.12g)\n",
                            (long long)i, (long long)j, (long long)k,
                            double(q.now(i, j, k)));
              ++bad;
            }
  }
  if (bad) {
    std::printf("  [M/%s] %ld failures\n", tag, bad);
    return false;
  }

  // A second consecutive call on the state the first one left behind: this is
  // the acoustic-substep pattern, and it must neither fail nor freeze the state.
  const A3 after1 = down3(f.p_pt, f.b);
  if (!f.run()) return false;
  const A3 after2 = down3(f.p_pt, f.b);
  long advanced = 0;
  for (int64_t k = 0; k < f.b.nk; ++k)
    for (int64_t j = f.b.js; j <= f.b.je; ++j)
      for (int64_t i = f.b.is; i <= f.b.ie; ++i) {
        if (!std::isfinite(double(after2(i, j, k)))) {
          std::printf("  [M/%s] pt non-finite after the second call\n", tag);
          return false;
        }
        if (!close_rel(after2(i, j, k), after1(i, j, k), 1.0e-9)) ++advanced;
      }
  if (!advanced) {
    std::printf("  [M/%s] the second call did not advance pt\n", tag);
    return false;
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("  [M/%s] residual CUDA error: %s\n", tag, cudaGetErrorString(e));
    return false;
  }
  std::printf("  [M/%s] ok (2 consecutive driver calls, all fields finite, delp > 0)\n",
              tag);
  return true;
}

static bool part_m() {
  bool ok = part_m_one(seam_box(), "seam");
  ok = part_m_one(interior_box(), "interior") && ok;
  return ok;
}

// ===========================================================================
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::printf("no CUDA device available -- skipping\n");
    return 77;
  }
  std::printf("WP-B.5 d_sw driver CUDA test, fv_real = %zu bytes\n", sizeof(fv_real));

  struct Part {
    const char *name;
    bool (*fn)();
  };
  const Part parts[] = {
      {"A  K1/K2/K3 contravariant, edge and corner winds", &part_a},
      {"B  K4a/K4b advective flux, Courant finalize, ra_x/ra_y", &part_b},
      {"C  K9 joint pt/delp update", &part_c},
      {"D  K10 dpx accumulation", &part_d},
      {"E  K7/K8/K11/K12 w chain", &part_e},
      {"F  K5 flux capacitors in isolation", &part_f},
      {"G  K6 heat_source/diss_est reset", &part_g},
      {"H  derived per-level coefficient arrays", &part_h},
      {"I  mixed per-level damping branches in one call", &part_i},
      {"J  flux capacitors across two invocations", &part_j},
      {"K  end-to-end anti-dependence invariants", &part_k},
      {"L  unsupported-configuration rejection", &part_l},
      {"M  direct launch_d_sw_cuda GPU execution", &part_m},
  };
  int failed = 0;
  for (const Part &p : parts) {
    std::printf("%s\n", p.name);
    if (!p.fn()) {
      std::printf("  FAILED: %s\n", p.name);
      ++failed;
    }
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("residual CUDA error at exit: %s\n", cudaGetErrorString(e));
    ++failed;
  }
  std::printf("%s (%d/%zu parts failed)\n", failed ? "FAIL" : "PASS", failed,
              sizeof(parts) / sizeof(parts[0]));
  return failed ? 1 : 0;
}
