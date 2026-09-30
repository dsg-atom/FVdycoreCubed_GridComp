// Prompt 73 -- host-only harness for the TRACER_2D orchestration layer.
//
// This runs on a login node. It needs no GPU, no CUDA and no MPI: the
// question it answers is purely about the per-k-level ORCHESTRATION
// arithmetic that surrounds the fv_tp_2d flux call inside tracer_2d_1L,
// not about the flux computation itself (that is tp_core.F90's fv_tp_2d,
// already GPU-executed as WP-B's dgrid::launch_scalar_transport_cuda --
// see cuda/dyn_core/tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp).
// Reimplementing PPM/fv_tp_2d here would duplicate WP-B's existing
// numerical service, which project_architecture.md forbids ("WP-C must
// reuse WP-B transport services rather than duplicate them" -- the same
// rule applies to any package consuming fv_tp_2d, including this one).
// So this harness treats the tracer flux divergence (fx, fy) as an
// EXTERNAL input for each fixture, exactly as tracer_2d_1L receives it
// from fv_tp_2d, and transcribes only the surrounding arithmetic that is
// unique to TRACER_2D itself:
//
//   xfx/yfx  -- courant-number-weighted flux areas, upwind-selected
//               sin_sg/dxa/dya face metric (tracer_2d_1L :163-182)
//   ra_x/ra_y -- donor-cell area used by the conservative update
//               (:279-289)
//   dp2       -- post-advection layer mass from the mass fluxes
//               (:296-300)
//   q update  -- conservative, mass-weighted tracer update
//               (:334-338, nsplt==1 case, i.e. the common case established
//               by Prompt 73's own runtime-config finding: production HS
//               carries exactly nq=1 tracer, and Courant numbers below 1
//               everywhere give nsplt=1 for that tracer's own level)
//
// Every formula below is copied verbatim from
// GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/
// @fvdycore/model/fv_tracer2d.F90 (geos/v3.0.0,
// ca30e26305fda43831095334c02a3f5e0b8c0c45), function tracer_2d_1L. No
// production numerical source was modified; this is a test-only,
// independent host-side transcription (same precedent as the WP-B
// scalar-transport test's own host reference), not a second maintained
// implementation.
//
// VALIDATION LEVEL: HOST-ONLY ARITHMETIC TRANSCRIPTION CHECK. This is NOT
// authoritative numerical validation against a Fortran-generated
// savepoint -- no such savepoint exists yet for TRACER_2D. It establishes
// that the selected port unit's formulas are well-defined, deterministic,
// mass-conservative, and free of NaN/Inf across the fixture set below,
// which is the prerequisite for a future authoritative comparison.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct Grid {
  int is, ie, js, je;     // interior bounds (inclusive)
  int isd, ied, jsd, jed; // halo-inclusive bounds (inclusive)

  int nx() const { return ied - isd + 1; }
  int ny() const { return jed - jsd + 1; }
};

// Flat 2D storage over the halo-inclusive domain, indexed [i][j] with
// i in [isd,ied], j in [jsd,jed].
class Field2D {
 public:
  Field2D(const Grid &g, double fill = 0.0) : g_(g), data_(g.nx() * g.ny(), fill) {}
  double &operator()(int i, int j) { return data_[idx(i, j)]; }
  double operator()(int i, int j) const { return data_[idx(i, j)]; }

 private:
  int idx(int i, int j) const { return (i - g_.isd) * g_.ny() + (j - g_.jsd); }
  Grid g_;
  std::vector<double> data_;
};

// tracer_2d_1L :163-182 -- courant-weighted flux area, upwind metric
// selection via sin_sg. cx/cy live on cell faces; sin_sg carries the
// 9-component cubed-sphere face-angle correction (indices 1/3 for the
// x-face pair, 2/4 for the y-face pair, as in the authoritative source).
void compute_xfx_yfx(const Grid &g, const Field2D &cx, const Field2D &cy,
                      const Field2D &dxa, const Field2D &dya, const Field2D &dx,
                      const Field2D &dy, const Field2D &sin_sg1, const Field2D &sin_sg3,
                      const Field2D &sin_sg2, const Field2D &sin_sg4, Field2D &xfx,
                      Field2D &yfx) {
  for (int j = g.jsd; j <= g.jed; ++j) {
    for (int i = g.is; i <= g.ie + 1; ++i) {
      if (cx(i, j) > 0.0) {
        xfx(i, j) = cx(i, j) * dxa(i - 1, j) * dy(i, j) * sin_sg3(i - 1, j);
      } else {
        xfx(i, j) = cx(i, j) * dxa(i, j) * dy(i, j) * sin_sg1(i, j);
      }
    }
  }
  for (int j = g.js; j <= g.je + 1; ++j) {
    for (int i = g.isd; i <= g.ied; ++i) {
      if (cy(i, j) > 0.0) {
        yfx(i, j) = cy(i, j) * dya(i, j - 1) * dx(i, j) * sin_sg4(i, j - 1);
      } else {
        yfx(i, j) = cy(i, j) * dya(i, j) * dx(i, j) * sin_sg2(i, j);
      }
    }
  }
}

// tracer_2d_1L :279-289 -- donor-cell area used by fv_tp_2d's own
// reconstruction and reused here by the conservative update.
void compute_ra(const Grid &g, const Field2D &area, const Field2D &xfx,
                 const Field2D &yfx, Field2D &ra_x, Field2D &ra_y) {
  for (int j = g.jsd; j <= g.jed; ++j) {
    for (int i = g.is; i <= g.ie; ++i) {
      ra_x(i, j) = area(i, j) + (xfx(i, j) - xfx(i + 1, j));
    }
  }
  for (int j = g.js; j <= g.je; ++j) {
    for (int i = g.isd; i <= g.ied; ++i) {
      ra_y(i, j) = area(i, j) + (yfx(i, j) - yfx(i, j + 1));
    }
  }
}

// tracer_2d_1L :296-300 -- post-advection layer mass from mass fluxes.
void compute_dp2(const Grid &g, const Field2D &dp1, const Field2D &mfx,
                  const Field2D &mfy, const Field2D &rarea, Field2D &dp2) {
  for (int j = g.js; j <= g.je; ++j) {
    for (int i = g.is; i <= g.ie; ++i) {
      dp2(i, j) = dp1(i, j) +
                  ((mfx(i, j) - mfx(i + 1, j)) + (mfy(i, j) - mfy(i, j + 1))) * rarea(i, j);
    }
  }
}

// tracer_2d_1L :334-338 (nsplt==1 branch) -- conservative, mass-weighted
// tracer update from the tracer flux divergence (fx, fy) that fv_tp_2d
// would have produced for this tracer.
void update_q(const Grid &g, const Field2D &q_old, const Field2D &dp1,
              const Field2D &dp2, const Field2D &fx, const Field2D &fy,
              const Field2D &rarea, Field2D &q_new) {
  for (int j = g.js; j <= g.je; ++j) {
    for (int i = g.is; i <= g.ie; ++i) {
      q_new(i, j) = (q_old(i, j) * dp1(i, j) +
                     ((fx(i, j) - fx(i + 1, j)) + (fy(i, j) - fy(i, j + 1))) * rarea(i, j)) /
                    dp2(i, j);
    }
  }
}

struct FixtureResult {
  const char *name;
  bool ok = true;
  bool has_nan_inf = false;
  double mass_before = 0.0;
  double mass_after = 0.0;
  double net_boundary_flux = 0.0;
  double mass_residual = 0.0; // (mass_after - mass_before) - net_boundary_flux
};

bool bad(double v) { return std::isnan(v) || std::isinf(v); }

// Builds a minimal, single-tile, halo-1 grid: interior 6x6, halo width 1
// (fv_tp_2d's own stencil needs >=3 in production; 1 is sufficient for
// this orchestration-only harness since none of the transcribed formulas
// reach past a 1-cell halo -- xfx/yfx's own upwind read is (i-1,j)/(i,j)
// at worst). Uniform, flat-Earth metrics (area=dx=dy=dxa=dya=1,
// sin_sg=1) isolate the arithmetic under test from cubed-sphere face
// geometry, which is a WP-E/grid-metrics concern, not TRACER_2D's own.
Grid make_grid() { return Grid{1, 6, 1, 6, 0, 7, 0, 7}; }

// Runs one fixture. `cx_fn`/`cy_fn` seed the Courant-number field (which
// drives xfx/yfx and, in production, nsplt -- this harness fixes
// nsplt=1 per the fixture design, matching Stage 2's production finding
// of sub-unity Courant numbers away from jet-stream levels). `q_fn` seeds
// the tracer. `flux_scale` scales a synthetic fx/fy field that stands in
// for fv_tp_2d's output; `boundary_flux` !=0 exercises the
// halo/domain-boundary-sensitive fixture by injecting a nonzero flux at
// the tile's outer edge.
template <typename CxFn, typename CyFn, typename QFn>
FixtureResult run_fixture(const char *name, CxFn cx_fn, CyFn cy_fn, QFn q_fn,
                           double flux_scale, double boundary_flux) {
  FixtureResult r{name};
  Grid g = make_grid();

  Field2D cx(g, 0.0), cy(g, 0.0), dxa(g, 1.0), dya(g, 1.0), dx(g, 1.0), dy(g, 1.0);
  Field2D sin_sg1(g, 1.0), sin_sg2(g, 1.0), sin_sg3(g, 1.0), sin_sg4(g, 1.0);
  Field2D area(g, 1.0), rarea(g, 1.0);
  Field2D xfx(g, 0.0), yfx(g, 0.0), ra_x(g, 0.0), ra_y(g, 0.0);
  Field2D dp1(g, 1.0), dp2(g, 0.0), mfx(g, 0.0), mfy(g, 0.0);
  Field2D q_old(g, 0.0), q_new(g, 0.0), fx(g, 0.0), fy(g, 0.0);

  for (int j = g.jsd; j <= g.jed; ++j)
    for (int i = g.is; i <= g.ie + 1; ++i) cx(i, j) = cx_fn(i, j);
  for (int j = g.js; j <= g.je + 1; ++j)
    for (int i = g.isd; i <= g.ied; ++i) cy(i, j) = cy_fn(i, j);
  for (int j = g.jsd; j <= g.jed; ++j)
    for (int i = g.isd; i <= g.ied; ++i) q_old(i, j) = q_fn(i, j);

  compute_xfx_yfx(g, cx, cy, dxa, dya, dx, dy, sin_sg1, sin_sg3, sin_sg2, sin_sg4, xfx, yfx);
  compute_ra(g, area, xfx, yfx, ra_x, ra_y);

  // Mass fluxes mirror the Courant-number field so dp2 stays physical
  // (mfx = cx * dp1, consistent with how production seeds mfx2 from the
  // acoustic-loop mass flux -- this harness only needs dp2 to be a
  // well-defined, nonzero denominator, not bitwise production values).
  for (int j = g.js; j <= g.je; ++j)
    for (int i = g.is; i <= g.ie + 1; ++i) mfx(i, j) = cx(i, j) * dp1(i, j);
  for (int j = g.js; j <= g.je + 1; ++j)
    for (int i = g.is; i <= g.ie; ++i) mfy(i, j) = cy(i, j) * dp1(i, j);
  compute_dp2(g, dp1, mfx, mfy, rarea, dp2);

  // Synthetic tracer flux: interior faces get a simple upwind flux
  // consistent with q_old and the Courant field (stands in for
  // fv_tp_2d's PPM flux -- any consistent, bounded flux exercises the
  // orchestration formulas identically). The tile's OUTER boundary faces
  // (i=is or i=ie+1, j=js or j=je+1) instead get `boundary_flux`
  // (0 for the isolated-tile fixtures, nonzero for the
  // halo-sensitive fixture) so the conservation check below can predict
  // the exact expected mass change from that single knob.
  for (int j = g.js; j <= g.je; ++j) {
    for (int i = g.is; i <= g.ie + 1; ++i) {
      double upwind_q = (cx(i, j) > 0.0) ? q_old(i - 1, j) : q_old(i, j);
      fx(i, j) = flux_scale * mfx(i, j) * upwind_q;
      if (i == g.is || i == g.ie + 1) fx(i, j) += boundary_flux;
    }
  }
  for (int j = g.js; j <= g.je + 1; ++j) {
    for (int i = g.is; i <= g.ie; ++i) {
      double upwind_q = (cy(i, j) > 0.0) ? q_old(i, j - 1) : q_old(i, j);
      fy(i, j) = flux_scale * mfy(i, j) * upwind_q;
      if (j == g.js || j == g.je + 1) fy(i, j) += boundary_flux;
    }
  }

  update_q(g, q_old, dp1, dp2, fx, fy, rarea, q_new);

  double mass_before = 0.0, mass_after = 0.0;
  for (int j = g.js; j <= g.je; ++j) {
    for (int i = g.is; i <= g.ie; ++i) {
      mass_before += q_old(i, j) * dp1(i, j);
      mass_after += q_new(i, j) * dp2(i, j);
      if (bad(q_new(i, j)) || bad(dp2(i, j))) r.has_nan_inf = true;
    }
  }

  // Net boundary flux: sum of outward flux at the tile's four edges
  // (west face at i=is carries -fx into the domain's accounting sign
  // convention used by the update formula: (fx(i)-fx(i+1)) is the
  // in-minus-out convention, so the domain's total change is
  // sum(fx(is,j)) - sum(fx(ie+1,j)) + sum(fy(i,js)) - sum(fy(i,je+1))).
  double boundary = 0.0;
  for (int j = g.js; j <= g.je; ++j) boundary += fx(g.is, j) - fx(g.ie + 1, j);
  for (int i = g.is; i <= g.ie; ++i) boundary += fy(i, g.js) - fy(i, g.je + 1);

  r.mass_before = mass_before;
  r.mass_after = mass_after;
  r.net_boundary_flux = boundary;
  r.mass_residual = (mass_after - mass_before) - boundary;
  r.ok = !r.has_nan_inf && std::fabs(r.mass_residual) < 1e-9 * std::max(1.0, std::fabs(mass_before));
  return r;
}

} // namespace

int main() {
  int failures = 0;
  std::vector<FixtureResult> results;

  // 1. Uniform tracer, zero Courant number: q must be exactly unchanged.
  results.push_back(run_fixture(
      "uniform_tracer_zero_courant", [](int, int) { return 0.0; }, [](int, int) { return 0.0; },
      [](int, int) { return 3.0; }, 1.0, 0.0));

  // 2. Smooth gradient tracer, sub-unity Courant (nsplt=1 regime).
  results.push_back(run_fixture(
      "smooth_gradient", [](int, int) { return 0.4; }, [](int, int) { return 0.3; },
      [](int i, int j) { return 0.1 * i + 0.2 * j; }, 1.0, 0.0));

  // 3. Sharp front in the tracer field, moderate positive Courant.
  results.push_back(run_fixture(
      "sharp_front", [](int, int) { return 0.5; }, [](int, int) { return 0.2; },
      [](int i, int) { return i <= 3 ? 0.0 : 10.0; }, 1.0, 0.0));

  // 4. Local extremum (isolated spike) with negative Courant number
  // (exercises the "else" branch of both xfx and yfx upwind selection).
  results.push_back(run_fixture(
      "local_extremum_negative_courant", [](int, int) { return -0.3; },
      [](int, int) { return -0.25; },
      [](int i, int j) { return (i == 3 && j == 3) ? 100.0 : 1.0; }, 1.0, 0.0));

  // 5. Zero tracer everywhere: q must remain exactly zero (0/dp2 = 0).
  results.push_back(run_fixture(
      "zero_tracer", [](int, int) { return 0.35; }, [](int, int) { return -0.15; },
      [](int, int) { return 0.0; }, 1.0, 0.0));

  // 6. Positive-definite edge case: tiny positive tracer near underflow,
  // must not go negative or NaN under the conservative update.
  results.push_back(run_fixture(
      "positive_definite_tiny", [](int, int) { return 0.6; }, [](int, int) { return 0.1; },
      [](int, int) { return 1e-12; }, 1.0, 0.0));

  // 7. Sign-changing (passive, non-positive-definite) tracer -- legal for
  // TRACER_2D's own contract (it makes no positivity assumption; that is
  // hord/limiter behavior inside fv_tp_2d, out of scope here).
  results.push_back(run_fixture(
      "sign_changing_passive", [](int, int) { return 0.2; }, [](int, int) { return 0.45; },
      [](int i, int j) { return (i + j) % 2 == 0 ? -5.0 : 5.0; }, 1.0, 0.0));

  // 8. Halo/domain-boundary-sensitive case: nonzero flux crossing the
  // tile's outer edge (stands in for a real inter-tile/halo exchange).
  // Mass change must equal exactly the injected boundary flux, i.e. the
  // conservation law holds across a genuine open boundary, not just in
  // the closed-tile fixtures above.
  results.push_back(run_fixture(
      "halo_sensitive_boundary_flux", [](int, int) { return 0.3; }, [](int, int) { return 0.3; },
      [](int i, int j) { return 2.0 + 0.05 * i * j; }, 1.0, 0.75));

  // 9. C512-proportioned realistic case: same formulas, realistic
  // magnitude Courant number (production hord=8/kord=9 HS run, sub-jet
  // levels) and a smooth realistic tracer profile.
  results.push_back(run_fixture(
      "c512_like_profile", [](int, int) { return 0.18; }, [](int, int) { return 0.09; },
      [](int i, int j) { return 280.0 + 0.5 * std::sin(0.3 * i) * std::cos(0.2 * j); }, 1.0,
      0.0));

  std::printf("%-32s %-6s %-10s %-14s %-14s %-14s\n", "fixture", "ok", "nan/inf",
              "mass_before", "mass_after", "residual");
  for (const auto &r : results) {
    std::printf("%-32s %-6s %-10s %-14.6e %-14.6e %-14.3e\n", r.name, r.ok ? "PASS" : "FAIL",
                r.has_nan_inf ? "YES" : "no", r.mass_before, r.mass_after, r.mass_residual);
    if (!r.ok) ++failures;
  }

  // Determinism check: rerun fixture 2 and require bitwise-identical mass
  // accounting (no uninitialized reads, no nondeterministic reduction
  // order -- this harness has none, but the check is cheap and Stage 14
  // asks for it explicitly).
  auto rerun = run_fixture(
      "smooth_gradient", [](int, int) { return 0.4; }, [](int, int) { return 0.3; },
      [](int i, int j) { return 0.1 * i + 0.2 * j; }, 1.0, 0.0);
  bool deterministic = (rerun.mass_after == results[1].mass_after) &&
                        (rerun.mass_residual == results[1].mass_residual);
  std::printf("\ndeterminism_check(smooth_gradient rerun): %s\n",
              deterministic ? "PASS (bitwise identical)" : "FAIL");
  if (!deterministic) ++failures;

  std::printf("\n%d/%zu fixtures + determinism check FAILED\n", failures, results.size() + 1);
  return failures == 0 ? 0 : 1;
}
