// ===========================================================================
// Milestone 3A.8 / WP-B.5 -- FieldRegistry binder and dispatch entry point for
// the authoritative d_sw driver.
//
// This file contains NO numerics. It resolves registered views, enforces the
// storage and configuration contract, carves WorkspaceId::transport into
// WpBDSwArena, and calls launch_d_sw_cuda once. Every formula lives in
// wp_b_dsw_cuda.cu and the WP-B services it orchestrates.
//
// Structure deliberately mirrors bind_wp_b_height_transport /
// launch_dgrid_height_transport_cuda in wp_b_contract_cuda.cpp, so the two
// WP-B dispatch slots can be diffed against each other.
// ===========================================================================

#include "dgrid/wp_b_contract_cuda.hpp"
#include "dgrid/wp_b_contract_views.hpp"
#include "dgrid/wp_b_dsw_cuda.cuh"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_types.hpp"

#include <cstdint>
#include <string>

namespace fv3::gpu::dgrid {

using namespace contract_views;

namespace {

// A registered field is missing or mis-shaped -- a wiring defect, not a
// configuration choice, so INVALID_ARGUMENT rather than
// UNSUPPORTED_CONFIGURATION. Matches the split launch_d_sw_cuda itself uses.
int missing(fv3_gpu_error *e, const char *fn, const std::string &what) {
  return fail(e, FV3_GPU_INVALID_ARGUMENT, fn, what);
}

// d_sw binds ~70 views; naming the offending one in the error beats making the
// caller bisect a registration list by hand.
struct Named {
  const DeviceViewBase *v;
  const char *name;
};

} // namespace

int bind_wp_b_d_sw(const ExecutionContext &x, WpBDSwViews &out, WpBDSwArena &arena,
                   fv3_gpu_error *error) noexcept {
  int rc = validate_execution_context(x, error);
  if (rc) return rc;
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "dgrid_dynamics requires the borrowed compute submission");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  // Storage prerequisites only. The NUMERICAL configuration fences (grid_type,
  // nested, hydrostatic, hord_*, nord, a_imp, inline_q, use_cond, do_f3d,
  // do_skeb, rot3, null stream) are §6 of wp_b_dsw_cuda.cuh and are enforced by
  // launch_d_sw_cuda before its first launch. Duplicating them here would give
  // two places to keep in sync; the one below is a shape requirement that the
  // driver cannot see because it never receives the DomainBounds halo.
  //
  // fv_tp_2d's stencil reaches +/-3 (ng = 3), xtp_u/ytp_v likewise, and
  // deln_flux / del6_vt_flux reach +/-(1 + nord).
  if (b.halo_width < 3)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_dynamics requires halo_width >= 3");
  if (b.npz < 1)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_dynamics requires npz >= 1");
  if (!x.immutable->topology)
    return missing(error, __func__, "dgrid_dynamics requires a bound TopologyView");
  const auto &topo = *x.immutable->topology;
  // npx/npy gate every panel-edge formula (xppm/yppm, xtp_u/ytp_v, the edge
  // wind corrections) and the corner fills. A wrong value silently applies an
  // interior stencil at a seam, so refuse to guess.
  if (topo.npx == 0 || topo.npy == 0)
    return missing(error, __func__,
                   "TopologyView npx/npy must be supplied for panel-edge semantics");
  // Same silent-interior-stencil hazard on the corner flags: Fortran only sets
  // them when grid_type < 3 .and. .not. nested (fv_grid_utils.F90:288-298), so a
  // caller asserting a panel corner while denying the cubed sphere is
  // self-contradictory.
  const bool any_corner =
      topo.sw_corner || topo.se_corner || topo.ne_corner || topo.nw_corner;
  if (any_corner && !(topo.global_cubed_sphere && cfg.grid_type < 3 && !topo.nested))
    return missing(error, __func__,
                   "TopologyView asserts a panel corner but the configuration is "
                   "not a non-nested cubed sphere with grid_type < 3");

  const auto &r = *x.fields;
  out = WpBDSwViews{};
  out.u = r.find(FieldId::u);
  out.v = r.find(FieldId::v);
  out.w = r.find(FieldId::w);
  out.delp = r.find(FieldId::delp);
  out.pt = r.find(FieldId::pt);
  out.uc = r.find(FieldId::uc);
  out.vc = r.find(FieldId::vc);
  out.ua = r.find(FieldId::ua);
  out.va = r.find(FieldId::va);
  out.divgd = r.find(FieldId::divgd);
  out.delpc = r.find(FieldId::delpc);
  out.ptc = r.find(FieldId::ptc);
  out.ut = r.find(FieldId::ut);
  out.vt = r.find(FieldId::vt);
  out.crx = r.find(FieldId::crx);
  out.cry = r.find(FieldId::cry);
  out.xfx = r.find(FieldId::xfx);
  out.yfx = r.find(FieldId::yfx);
  out.heat_source = r.find(FieldId::heat_source);
  out.diss_est = r.find(FieldId::diss_est);
  out.z_rat = r.find(FieldId::z_rat);
  out.mfx = r.find(FieldId::mfx);
  out.mfy = r.find(FieldId::mfy);
  out.cx = r.find(FieldId::cx);
  out.cy = r.find(FieldId::cy);
  out.dpx = r.find(FieldId::dpx);

  if (!x.immutable->grid)
    return missing(error, __func__, "dgrid_dynamics requires a bound grid registry");
  const auto &g = *x.immutable->grid;
  out.rarea_c = g.find(FieldId::rarea_c);
  out.rarea = g.find(FieldId::rarea);
  out.dxc = g.find(FieldId::dxc);
  out.dyc = g.find(FieldId::dyc);
  out.dx = g.find(FieldId::dx);
  out.dy = g.find(FieldId::dy);
  out.rdx = g.find(FieldId::rdx);
  out.rdy = g.find(FieldId::rdy);
  out.divg_u = g.find(FieldId::divg_u);
  out.divg_v = g.find(FieldId::divg_v);
  out.cosa_u = g.find(FieldId::cosa_u);
  out.cosa_v = g.find(FieldId::cosa_v);
  out.sina_u = g.find(FieldId::sina_u);
  out.sina_v = g.find(FieldId::sina_v);
  out.cosa_s = g.find(FieldId::cosa_s);
  out.rsin2 = g.find(FieldId::rsin2);
  out.sin_sg = g.find(FieldId::sin_sg);
  out.grid_lonlat = g.find(FieldId::grid_lonlat);
  out.agrid_lonlat = g.find(FieldId::agrid_lonlat);
  out.edge_w = g.find(FieldId::edge_w);
  out.edge_e = g.find(FieldId::edge_e);
  out.edge_s = g.find(FieldId::edge_s);
  out.edge_n = g.find(FieldId::edge_n);
  out.area = g.find(FieldId::area);
  out.dxa = g.find(FieldId::dxa);
  out.dya = g.find(FieldId::dya);
  out.rdxa = g.find(FieldId::rdxa);
  out.rdya = g.find(FieldId::rdya);
  out.del6_u = g.find(FieldId::del6_u);
  out.del6_v = g.find(FieldId::del6_v);
  out.cosa = g.find(FieldId::cosa);
  out.rsina = g.find(FieldId::rsina);
  out.rsin_u = g.find(FieldId::rsin_u);
  out.rsin_v = g.find(FieldId::rsin_v);
  out.f0 = g.find(FieldId::f0);
  out.nord_k = g.find(FieldId::damping_nord_k);
  out.nord_v = g.find(FieldId::damping_ndif);
  out.nord_w = g.find(FieldId::damping_nord_w);
  out.d2_bg = g.find(FieldId::damping_d2_bg);
  out.d4_bg = g.find(FieldId::damping_d4_bg);
  out.damp_v = g.find(FieldId::damping_v);
  out.damp_w = g.find(FieldId::damping_w);
  out.damp_t = g.find(FieldId::damping_t);
  out.d_con = g.find(FieldId::damping_d_con);
  out.kgb = g.find(FieldId::damping_kgb);

  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;
  const int64_t kt = b.npz - 1;

  // --- 3-D state -----------------------------------------------------------
  // The staggered fields are NOT on the same box as the A-grid fields, and
  // this is the one place where getting it wrong is silent. sw_core.F90:548-564
  // declares, verbatim:
  //
  //   delp, pt, ua, va, z_rat, delpc, ptc  (isd:ied,   jsd:jed)
  //   u,  vc                               (isd:ied,   jsd:jed+1)
  //   v,  uc                               (isd:ied+1, jsd:jed)
  //   divg_d                               (isd:ied+1, jsd:jed+1)
  //   ut  (local)                          (isd:ied+1, jsd:jed)
  //   vt  (local)                          (isd:ied,   jsd:jed+1)
  //
  // and those +1 rows/columns are genuinely READ. The ut interior loop
  // (sw_core.F90:690-696) evaluates
  //     ut(i,j) = uc(i,j) - 0.25*cosa_u(i,j)*(vc(i-1,j) + vc(i,j)
  //                                           + vc(i-1,j+1) + vc(i,j+1))
  // over j = jsd..jed, so at j == jed it reads vc(i, jed+1); the vt loop
  // (:702-706) reads uc(i+1, j) over i = isd..ied, so at i == ied it reads
  // uc(ied+1, j).
  //
  // An earlier version of this binder required one uniform
  // [isd,ied] x [jsd,jed] box for all seventeen fields, with a comment
  // claiming that "with halo >= 3 the extended box already contains the +1
  // stagger, so one uniform requirement is strictly stronger". That reasoning
  // was simply wrong: isd/ied ARE is-halo/ie+halo, so jed+1 lies one row
  // outside the box no matter how large the halo is. The requirement was
  // therefore too WEAK, and it accepted a registration the kernels over-read.
  // compute-sanitizer memcheck caught it on the integrated
  // fv3_gpu_dyn_core_step path -- "Invalid __global__ read of size 8 bytes" in
  // dsw_ut_interior_kernel at (i, j) = (isd, jed+1) -- which the focused d_sw
  // test had missed because its hand-built fixture pads its arrays.
  //
  // Requiring MORE than Fortran declares is safe (it rejects a sloppy
  // registration); requiring less reads out of bounds. So each group below is
  // the exact Fortran box, and heat_source/diss_est are the only deliberate
  // over-requirement: Fortran declares them (is:ie, js:je) but the S9 reset
  // kernel covers the full extended box.
  struct Group {
    const Named *fields;
    size_t count;
    int64_t di, dj; // +1 stagger in i and/or j
    const char *box;
  };
  const Named agrid[] = {{out.w, "w"},         {out.delp, "delp"},
                         {out.pt, "pt"},       {out.ua, "ua"},
                         {out.va, "va"},       {out.delpc, "delpc"},
                         {out.ptc, "ptc"},     {out.heat_source, "heat_source"},
                         {out.diss_est, "diss_est"}, {out.z_rat, "z_rat"}};
  const Named yface[] = {{out.u, "u"}, {out.vc, "vc"}, {out.vt, "vt"}};
  const Named xface[] = {{out.v, "v"}, {out.uc, "uc"}, {out.ut, "ut"}};
  const Named bgrid[] = {{out.divgd, "divgd"}};
  const Group groups[] = {
      {agrid, sizeof(agrid) / sizeof(Named), 0, 0, "[isd,ied] x [jsd,jed]"},
      {yface, sizeof(yface) / sizeof(Named), 0, 1, "[isd,ied] x [jsd,jed+1]"},
      {xface, sizeof(xface) / sizeof(Named), 1, 0, "[isd,ied+1] x [jsd,jed]"},
      {bgrid, sizeof(bgrid) / sizeof(Named), 1, 1, "[isd,ied+1] x [jsd,jed+1]"}};
  for (const Group &g : groups) {
    for (size_t n = 0; n < g.count; ++n) {
      const Named &f = g.fields[n];
      if (!f.v || !f.v->data)
        return missing(error, __func__,
                       std::string("dgrid_dynamics is missing required state field ") + f.name);
      if (!covers3(f.v, lo_i, hi_i + g.di, lo_j, hi_j + g.dj, 0, kt))
        return missing(error, __func__,
                       std::string("dgrid_dynamics state field ") + f.name +
                           " does not cover " + g.box + " x [0,npz-1]");
    }
  }

  // crx/xfx are read on [is, ie+1] x [jsd, jed] and cry/yfx on
  // [isd, ied] x [js, je+1] -- the same rectangles the height-transport binder
  // requires, because the consumer (fv_tp_2d) is the same routine.
  if (!covers3(out.crx, b.is, b.ie + 1, lo_j, hi_j, 0, kt) ||
      !covers3(out.xfx, b.is, b.ie + 1, lo_j, hi_j, 0, kt))
    return missing(error, __func__, "crx/xfx do not cover [is,ie+1] x [jsd,jed] x [0,npz-1]");
  if (!covers3(out.cry, lo_i, hi_i, b.js, b.je + 1, 0, kt) ||
      !covers3(out.yfx, lo_i, hi_i, b.js, b.je + 1, 0, kt))
    return missing(error, __func__, "cry/yfx do not cover [isd,ied] x [js,je+1] x [0,npz-1]");

  // --- flux capacitors ------------------------------------------------------
  // Fortran rectangles (sw_core.F90 d_sw, and the Part J test):
  //   xflux/mfx  is..ie+1 x js..je      cx  is..ie+1 x jsd..jed
  //   yflux/mfy  is..ie   x js..je+1    cy  isd..ied x js..je+1
  //   dpx        is..ie   x js..je
  // These are accumulators the driver ADDS to and never initializes, so a view
  // that is too small would silently drop part of the tracer-transport budget.
  if (!covers3(out.mfx, b.is, b.ie + 1, b.js, b.je, 0, kt))
    return missing(error, __func__, "mfx does not cover [is,ie+1] x [js,je] x [0,npz-1]");
  if (!covers3(out.mfy, b.is, b.ie, b.js, b.je + 1, 0, kt))
    return missing(error, __func__, "mfy does not cover [is,ie] x [js,je+1] x [0,npz-1]");
  if (!covers3(out.cx, b.is, b.ie + 1, lo_j, hi_j, 0, kt))
    return missing(error, __func__, "cx does not cover [is,ie+1] x [jsd,jed] x [0,npz-1]");
  if (!covers3(out.cy, lo_i, hi_i, b.js, b.je + 1, 0, kt))
    return missing(error, __func__, "cy does not cover [isd,ied] x [js,je+1] x [0,npz-1]");
  if (!covers3(out.dpx, b.is, b.ie, b.js, b.je, 0, kt))
    return missing(error, __func__, "dpx does not cover [is,ie] x [js,je] x [0,npz-1]");
  // The capacitors are kind=8 in Fortran regardless of FV precision
  // (accumulator_real). make_device_view cannot tell an R4 caller apart from a
  // mis-registered one here, and reinterpreting 4-byte storage as doubles would
  // corrupt the tracer budget silently, so pin the width.
  const Named accs[] = {{out.mfx, "mfx"}, {out.mfy, "mfy"}, {out.cx, "cx"},
                        {out.cy, "cy"},   {out.dpx, "dpx"}};
  for (const Named &f : accs)
    if (!view_element_width_is(f.v, sizeof(accumulator_real)))
      return missing(error, __func__,
                     std::string("flux capacitor ") + f.name +
                         " must be registered with accumulator_real (kind=8) "
                         "element width");

  // --- 2-D geometry ---------------------------------------------------------
  const Named geo2[] = {
      {out.rarea_c, "rarea_c"}, {out.rarea, "rarea"},   {out.dxc, "dxc"},
      {out.dyc, "dyc"},         {out.dx, "dx"},         {out.dy, "dy"},
      {out.rdx, "rdx"},         {out.rdy, "rdy"},       {out.divg_u, "divg_u"},
      {out.divg_v, "divg_v"},   {out.cosa_u, "cosa_u"}, {out.cosa_v, "cosa_v"},
      {out.sina_u, "sina_u"},   {out.sina_v, "sina_v"}, {out.cosa_s, "cosa_s"},
      {out.rsin2, "rsin2"},     {out.area, "area"},     {out.dxa, "dxa"},
      {out.dya, "dya"},         {out.rdxa, "rdxa"},     {out.rdya, "rdya"},
      {out.del6_u, "del6_u"},   {out.del6_v, "del6_v"}, {out.cosa, "cosa"},
      {out.rsina, "rsina"},     {out.rsin_u, "rsin_u"}, {out.rsin_v, "rsin_v"},
      {out.f0, "f0"}};
  for (const Named &f : geo2) {
    if (!f.v || !f.v->data)
      return missing(error, __func__,
                     std::string("dgrid_dynamics is missing required grid metric ") + f.name);
    if (!covers2(f.v, lo_i, hi_i, lo_j, hi_j))
      return missing(error, __func__,
                     std::string("grid metric ") + f.name +
                         " does not cover [isd,ied] x [jsd,jed]");
  }
  // sin_sg carries the four cell-edge components; the CUDA views index them
  // 0-based (Fortran component 1 -> 0).
  if (!covers3(out.sin_sg, lo_i, hi_i, lo_j, hi_j, 0, 3))
    return missing(error, __func__, "sin_sg does not cover [isd,ied] x [jsd,jed] x [0,3]");
  // a2b_ord4 (reached through smag_corner) reads the corner grid and the A-grid
  // centres two cells out, and its lon/lat component axis must exist. Same
  // requirement WP-D's binder imposes on these two fields.
  if (!covers3(out.grid_lonlat, b.is - 2, b.ie + 2, b.js - 2, b.je + 2, 0, 1) ||
      !covers3(out.agrid_lonlat, b.is - 2, b.ie + 2, b.js - 2, b.je + 2, 0, 1))
    return missing(error, __func__,
                   "grid_lonlat/agrid_lonlat do not cover [is-2,ie+2] x [js-2,je+2] "
                   "with a 2-component lon/lat axis");
  // The four a2b edge-weight arrays are indexed by a2b_ord4's own edge logic
  // and are only read on panel edges. Presence and element width are checked;
  // their extents are WP-D's contract (wp_d_contract.cpp does not shape-check
  // them either) and duplicating a guess here would just invent a second,
  // possibly different, rule.
  const Named edges[] = {{out.edge_w, "edge_w"}, {out.edge_e, "edge_e"},
                         {out.edge_s, "edge_s"}, {out.edge_n, "edge_n"}};
  for (const Named &f : edges) {
    if (!f.v || !f.v->data)
      return missing(error, __func__,
                     std::string("dgrid_dynamics is missing a2b edge-weight array ") + f.name);
    if (!view_element_width_is(f.v, sizeof(accumulator_real)))
      return missing(error, __func__,
                     std::string(f.name) +
                         " must be registered with accumulator_real (kind=8) "
                         "element width");
  }

  // --- per-level controls ---------------------------------------------------
  // Indexed at min(k, npz-1) by DivergenceLevels/LevelFilter, so [0, npz-1] is
  // exactly the required extent.
  const Named levels[] = {
      {out.nord_k, "damping_nord_k"}, {out.nord_v, "damping_ndif"},
      {out.nord_w, "damping_nord_w"}, {out.d2_bg, "damping_d2_bg"},
      {out.d4_bg, "damping_d4_bg"},   {out.damp_v, "damping_v"},
      {out.damp_w, "damping_w"},      {out.damp_t, "damping_t"},
      {out.d_con, "damping_d_con"},   {out.kgb, "damping_kgb"}};
  for (const Named &f : levels) {
    if (!covers1(f.v, 0, b.npz - 1))
      return missing(error, __func__,
                     std::string("per-level damping control ") + f.name +
                         " is missing or does not cover [0, npz-1]");
    // damping_nord_k / damping_ndif / damping_nord_w are LogicalType::integer
    // and the public ABI has no integer element type, so make_device_view's
    // precision check is a tautology for them (fv3_gpu_contract.cpp compares
    // element_bytes against itself). Without this guard an R8 caller could
    // register 4-byte storage and the View1<const fv_real> selectors would
    // reinterpret it as doubles -- arbitrary nord values, and therefore
    // arbitrary damping branches per level. The fv_real-typed controls get the
    // same assertion so a future logical-type change cannot reintroduce it.
    if (!view_element_width_is(f.v, sizeof(fv_real)))
      return missing(error, __func__,
                     std::string("per-level damping control ") + f.name +
                         " element width does not match the active fv_real "
                         "precision; integral orders must be registered as "
                         "exactly-representable fv_real");
  }

  // --- arena ---------------------------------------------------------------
  const auto *slice = workspace_slice(*x.workspace, WorkspaceId::transport);
  if (!slice || !slice->data)
    return missing(error, __func__, "WorkspaceId::transport slice is absent");

  uint64_t ni = static_cast<uint64_t>(hi_i - lo_i + 1);
  uint64_t nj = static_cast<uint64_t>(hi_j - lo_j + 1);
  uint64_t nlev = static_cast<uint64_t>(b.npz + 1);
  uint64_t values = 1;
  if (mul_into(values, ni) || mul_into(values, nj) || mul_into(values, nlev))
    return missing(error, __func__, "d_sw arena size overflowed");
  uint64_t bytes = values;
  if (mul_into(bytes, sizeof(fv_real)))
    return missing(error, __func__, "d_sw arena size overflowed");
  uint64_t total = bytes;
  if (mul_into(total, static_cast<uint64_t>(WpBDSwArena::array_count)))
    return missing(error, __func__, "d_sw arena size overflowed");
  if (slice->bytes < total)
    return missing(error, __func__,
                   "WorkspaceId::transport slice is smaller than "
                   "WpBDSwArena::array_count requires");
  // The last slot is subdivided into level_array_count npz-length columns, so
  // it must hold them. ni*nj >= 7 for any halo >= 3 domain, but assert rather
  // than reason about it.
  if (values < static_cast<uint64_t>(WpBDSwArena::level_array_count) *
                   static_cast<uint64_t>(b.npz))
    return missing(error, __func__,
                   "d_sw arena slot cannot hold the per-level coefficient arrays");

  char *base = static_cast<char *>(slice->data);
  auto slot = [&](int64_t n) { return static_cast<void *>(base + static_cast<uint64_t>(n) * bytes); };

  arena = WpBDSwArena{};
  // 14 d_sw locals with real lifetimes.
  arena.ub = to_arena_view3(slot(0), b);
  arena.vb = to_arena_view3(slot(1), b);
  arena.wk = to_arena_view3(slot(2), b);
  arena.ke = to_arena_view3(slot(3), b);
  arena.vort = to_arena_view3(slot(4), b);
  arena.fx = to_arena_view3(slot(5), b);
  arena.fy = to_arena_view3(slot(6), b);
  arena.gx = to_arena_view3(slot(7), b);
  arena.gy = to_arena_view3(slot(8), b);
  arena.ra_x = to_arena_view3(slot(9), b);
  arena.ra_y = to_arena_view3(slot(10), b);
  arena.fx2 = to_arena_view3(slot(11), b);
  arena.fy2 = to_arena_view3(slot(12), b);
  arena.dw = to_arena_view3(slot(13), b);
  // 7 fv_tp_2d / deln_flux scratch arrays.
  arena.q_i = to_arena_view3(slot(14), b);
  arena.q_j = to_arena_view3(slot(15), b);
  arena.fx2_tp = to_arena_view3(slot(16), b);
  arena.fy2_tp = to_arena_view3(slot(17), b);
  arena.d2 = to_arena_view3(slot(18), b);
  arena.fx2_dn = to_arena_view3(slot(19), b);
  arena.fy2_dn = to_arena_view3(slot(20), b);
  // 4 PPM momentum workspace arrays. smt is int32 in a slot sized for fv_real.
  arena.bl = to_arena_view3(slot(21), b);
  arena.br = to_arena_view3(slot(22), b);
  arena.b0 = to_arena_view3(slot(23), b);
  arena.smt = to_arena_view3_i(slot(24), b);
  // 8 smag_corner privates. Distinct from ut/vt/wk on purpose: S31 runs while
  // wk still holds the S27 relative vorticity that S33 and S36 read.
  arena.smag.ut = to_arena_view3(slot(25), b);
  arena.smag.vt = to_arena_view3(slot(26), b);
  arena.smag.wk = to_arena_view3(slot(27), b);
  arena.smag.sh = to_arena_view3(slot(28), b);
  arena.smag.qx = to_arena_view3(slot(29), b);
  arena.smag.qy = to_arena_view3(slot(30), b);
  arena.smag.qxx = to_arena_view3(slot(31), b);
  arena.smag.qyy = to_arena_view3(slot(32), b);
  // 1 slot subdivided into the 7 derived per-level coefficient columns.
  {
    auto *lv = static_cast<fv_real *>(slot(33));
    const int64_t n = b.npz;
    arena.deln_damp_delp = View1<fv_real>{lv + 0 * n, 0, 1};
    arena.deln_gate_delp = View1<fv_real>{lv + 1 * n, 0, 1};
    arena.deln_damp_pt = View1<fv_real>{lv + 2 * n, 0, 1};
    arena.deln_gate_pt = View1<fv_real>{lv + 3 * n, 0, 1};
    arena.nord_t_level = View1<fv_real>{lv + 4 * n, 0, 1};
    arena.del6_damp_w = View1<fv_real>{lv + 5 * n, 0, 1};
    arena.del6_damp_v = View1<fv_real>{lv + 6 * n, 0, 1};
  }

  return FV3_GPU_SUCCESS;
}

// ---------------------------------------------------------------------------
// NumericalDispatch::dgrid_dynamics
// ---------------------------------------------------------------------------
// One bind, one launch. launch_d_sw_cuda executes S1..S38 in authoritative
// order on the borrowed compute stream and never synchronizes, so persistent
// device residency is preserved: nothing is staged through the host and no
// field leaves the device between stages.
int launch_dgrid_dynamics_cuda(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  WpBDSwViews v;
  WpBDSwArena ws;
  int rc = bind_wp_b_d_sw(x, v, ws, error);
  if (rc) return rc;

  const auto &db = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &topo = *x.immutable->topology;
  const auto &dp = *x.dispatch_params;

  TransportBounds b{};
  b.is = db.is;
  b.ie = db.ie;
  b.js = db.js;
  b.je = db.je;
  b.npz = db.npz;
  // d_sw operates on the npz CELL levels (update_dz_d's npz+1 interface count
  // is the other WP-B slot).
  b.nk = db.npz;
  b.halo_width = static_cast<int64_t>(db.halo_width);
  // Fortran ng is a parameter, not the registered halo (tools/fv_mp_mod.F90:108);
  // halo_width >= 3 is validated in the binder so all ng-wide stencils are in
  // bounds.
  b.ng = 3;
  b.npx = topo.npx;
  b.npy = topo.npy;
  b.nested = topo.nested ? 1 : 0;
  b.cubed_sphere = (topo.global_cubed_sphere && cfg.grid_type < 3) ? 1 : 0;
  b.west_edge = (db.is == 1) ? 1 : 0;
  b.east_edge = (db.ie + 1 == topo.npx) ? 1 : 0;
  b.south_edge = (db.js == 1) ? 1 : 0;
  b.north_edge = (db.je + 1 == topo.npy) ? 1 : 0;
  b.sw_corner = topo.sw_corner ? 1 : 0;
  b.se_corner = topo.se_corner ? 1 : 0;
  b.ne_corner = topo.ne_corner ? 1 : 0;
  b.nw_corner = topo.nw_corner ? 1 : 0;

  DSwOptions opt{};
  // d_sw's dt is the ACOUSTIC SUBSTEP dt, not the big timestep:
  //   fv_dynamics -> dyn_core(bdt, n_split)
  //   dyn_core.F90:297   dt = bdt / real(n_split)
  //   dyn_core.F90:798   call d_sw(... dt ...)
  // DispatchParams::dt carries fv3_gpu_step_desc::bdt verbatim
  // (runtime/fv3_gpu_runtime.cpp), and n_split arrives beside it, so the
  // division belongs here -- the same place Fortran puts it.
  //
  // NOTE for integration: fv3_gpu_dyn_core_step currently walks its stage list
  // ONCE per call, so one call advances one acoustic substep. The n_split
  // acoustic loop itself is not yet implemented in the runtime; that is
  // recorded as a Gate C1 item, not papered over here by pretending dt is
  // something it is not.
  const int32_t n_split = dp.n_split > 0 ? dp.n_split : 1;
  opt.dt = static_cast<fv_real>(dp.dt / static_cast<double>(n_split));
  opt.hord_mt = cfg.hord_mt;
  opt.hord_vt = cfg.hord_vt;
  opt.hord_tm = cfg.hord_tm;
  opt.hord_dp = cfg.hord_dp;
  opt.lim_fac = static_cast<fv_real>(cfg.lim_fac);
  opt.nord = cfg.nord;
  opt.nord_t = cfg.nord_t;
  opt.dddmp = static_cast<fv_real>(cfg.dddmp);
  opt.da_min = static_cast<fv_real>(cfg.da_min);
  opt.da_min_c = static_cast<fv_real>(cfg.da_min_c);
  opt.grid_type = cfg.grid_type;
  opt.hydrostatic = cfg.hydrostatic ? 1 : 0;
  opt.inline_q = cfg.inline_q ? 1 : 0;
  opt.use_cond = cfg.use_cond ? 1 : 0;
  opt.do_f3d = cfg.do_f3d ? 1 : 0;
  opt.do_skeb = cfg.do_skeb ? 1 : 0;
  opt.rot3 = cfg.rot3 ? 1 : 0;
  opt.a_imp = cfg.a_imp;

  DSwLevels lv{};
  lv.nord_k = to_view1<const fv_real>(v.nord_k);
  lv.nord_v = to_view1<const fv_real>(v.nord_v);
  lv.nord_w = to_view1<const fv_real>(v.nord_w);
  lv.d2_bg = to_view1<const fv_real>(v.d2_bg);
  lv.d4_bg = to_view1<const fv_real>(v.d4_bg);
  lv.damp_v = to_view1<const fv_real>(v.damp_v);
  lv.damp_w = to_view1<const fv_real>(v.damp_w);
  lv.damp_t = to_view1<const fv_real>(v.damp_t);
  lv.d_con = to_view1<const fv_real>(v.d_con);
  lv.kgb = to_view1<const fv_real>(v.kgb);

  DSwGeometry geo{};
  geo.div.rarea_c = to_view2<const fv_real>(v.rarea_c);
  geo.div.rarea = to_view2<const fv_real>(v.rarea);
  geo.div.dxc = to_view2<const fv_real>(v.dxc);
  geo.div.dyc = to_view2<const fv_real>(v.dyc);
  geo.div.dx = to_view2<const fv_real>(v.dx);
  geo.div.dy = to_view2<const fv_real>(v.dy);
  geo.div.rdx = to_view2<const fv_real>(v.rdx);
  geo.div.rdy = to_view2<const fv_real>(v.rdy);
  geo.div.divg_u = to_view2<const fv_real>(v.divg_u);
  geo.div.divg_v = to_view2<const fv_real>(v.divg_v);
  geo.div.cosa_u = to_view2<const fv_real>(v.cosa_u);
  geo.div.cosa_v = to_view2<const fv_real>(v.cosa_v);
  geo.div.sina_u = to_view2<const fv_real>(v.sina_u);
  geo.div.sina_v = to_view2<const fv_real>(v.sina_v);
  geo.div.cosa_s = to_view2<const fv_real>(v.cosa_s);
  geo.div.rsin2 = to_view2<const fv_real>(v.rsin2);
  geo.div.sin_sg = to_view3<const fv_real>(v.sin_sg);

  geo.a2b.dxa = to_view2<const fv_real>(v.dxa);
  geo.a2b.dya = to_view2<const fv_real>(v.dya);
  geo.a2b.grid_lonlat = to_view3<const fv_real>(v.grid_lonlat);
  geo.a2b.agrid_lonlat = to_view3<const fv_real>(v.agrid_lonlat);
  geo.a2b.edge_w = to_view1<const accumulator_real>(v.edge_w);
  geo.a2b.edge_e = to_view1<const accumulator_real>(v.edge_e);
  geo.a2b.edge_s = to_view1<const accumulator_real>(v.edge_s);
  geo.a2b.edge_n = to_view1<const accumulator_real>(v.edge_n);

  geo.area = to_view2<const fv_real>(v.area);
  geo.dxa = to_view2<const fv_real>(v.dxa);
  geo.dya = to_view2<const fv_real>(v.dya);
  geo.rdxa = to_view2<const fv_real>(v.rdxa);
  geo.rdya = to_view2<const fv_real>(v.rdya);
  geo.del6_u = to_view2<const fv_real>(v.del6_u);
  geo.del6_v = to_view2<const fv_real>(v.del6_v);
  geo.cosa = to_view2<const fv_real>(v.cosa);
  geo.rsina = to_view2<const fv_real>(v.rsina);
  geo.rsin_u = to_view2<const fv_real>(v.rsin_u);
  geo.rsin_v = to_view2<const fv_real>(v.rsin_v);
  geo.f0 = to_view2<const fv_real>(v.f0);

  DSwState st{};
  st.u = to_view3<fv_real>(v.u);
  st.v = to_view3<fv_real>(v.v);
  st.w = to_view3<fv_real>(v.w);
  st.delp = to_view3<fv_real>(v.delp);
  st.pt = to_view3<fv_real>(v.pt);
  st.uc = to_view3<fv_real>(v.uc);
  st.vc = to_view3<fv_real>(v.vc);
  st.ua = to_view3<const fv_real>(v.ua);
  st.va = to_view3<const fv_real>(v.va);
  st.divgd = to_view3<fv_real>(v.divgd);
  st.delpc = to_view3<fv_real>(v.delpc);
  st.ptc = to_view3<fv_real>(v.ptc);
  st.ut = to_view3<fv_real>(v.ut);
  st.vt = to_view3<fv_real>(v.vt);
  st.crx = to_view3<fv_real>(v.crx);
  st.cry = to_view3<fv_real>(v.cry);
  st.xfx = to_view3<fv_real>(v.xfx);
  st.yfx = to_view3<fv_real>(v.yfx);
  st.heat_source = to_view3<fv_real>(v.heat_source);
  st.diss_est = to_view3<fv_real>(v.diss_est);
  st.z_rat = to_view3<const fv_real>(v.z_rat);
  st.mfx = to_view3<accumulator_real>(v.mfx);
  st.mfy = to_view3<accumulator_real>(v.mfy);
  st.cx = to_view3<accumulator_real>(v.cx);
  st.cy = to_view3<accumulator_real>(v.cy);
  st.dpx = to_view3<accumulator_real>(v.dpx);

  const fv3_gpu_status status =
      launch_d_sw_cuda(b, opt, lv, geo, st, ws, x.submission.stream);
  if (status != FV3_GPU_SUCCESS)
    return fail(error, status, __func__, "launch_d_sw_cuda failed");

  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::dgrid
