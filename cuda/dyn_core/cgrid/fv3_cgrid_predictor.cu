// ===========================================================================
// WP-A -- NumericalDispatch::cgrid_predictor.
//
// This slot is dyn_core.F90's per-acoustic-substep C-grid predictor group. It
// contains NO numerics of its own: it resolves registered views, enforces the
// configuration/storage fence, carves WorkspaceId::cgrid, and then replays the
// authoritative c_sw call sequence one level at a time through the existing
// WP-A leaf launchers.
//
// Authoritative sources (GFDL_atmos_cubed_sphere tag geos/v3.0.0, ca30e26):
//
//   dyn_core.F90:514-524   do k=1,npz ; call c_sw(...) ; enddo
//                          -- dord4 = .true. (:521) and dt2 (:521), and the
//                             actual argument for c_sw's `wc` dummy is `omga`
//                             (:518). There is therefore no separate `wc`
//                             FieldId: FieldId::omga IS wc.
//   dyn_core.F90:296-297   dt = bdt/real(n_split) ; dt2 = 0.5*dt
//                          -- carried as DispatchParams::dt2. This slot never
//                             divides by n_split itself.
//   sw_core.F90:103-512    subroutine c_sw -- the phase order implemented below.
//   sw_core.F90:172-173    d2a2c_vect, called from INSIDE c_sw (not dyn_core).
//   sw_core.F90:175-181    divergence_corner, also from inside c_sw, nord > 0.
//
// NOT in this slot: update_dz_c and Riem_Solver_c (dyn_core.F90:551-590) belong
// to NumericalDispatch::cgrid_vertical_solver. fv3_cgrid_dz.hpp is deliberately
// NOT included here any more -- its presence in the previous stub implied a
// coupling that the authoritative call graph does not have.
//
// Numerical semantics changed: NO. Every formula executed here lives in
// fv3_cgrid_d2a2c.cu, fv3_cgrid_divergence.cu, fv3_cgrid_corners.cu and
// fv3_cgrid_csw.cu; this file only orders them and supplies their storage.
// ===========================================================================

#include "fv3_cgrid_predictor.hpp"
#include "fv3_cgrid_corners.hpp"
#include "fv3_cgrid_csw.hpp"
#include "fv3_cgrid_d2a2c.hpp"
#include "fv3_cgrid_divergence.hpp"
#include "fv3_gpu_errors.hpp"

#include <cstdint>
#include <string>

namespace fv3::gpu::cgrid {
namespace {

// ---------------------------------------------------------------------------
// Workspace arena
// ---------------------------------------------------------------------------
// WorkspaceId::cgrid is sized by the runtime (runtime/fv3_gpu_runtime.cpp:203)
// in whole (ni x nj x (npz+1)) fv_real arrays over the full halo box, with
// ni = ie-is+1+2*halo_width and nj = je-js+1+2*halo_width. This arena follows
// that existing WP-B precedent (dgrid/wp_b_dsw_cuda.cuh WpBDSwArena and
// dgrid/wp_b_contract_views.hpp:55 to_arena_view3) rather than inventing a
// second convention.
//
// Slot map. Every entry is one Fortran local that is not a registered field:
//
//   0  utmp   d2a2c_vect, declared (isd:ied,jsd:jed)   sw_core.F90:2813
//   1  vtmp   d2a2c_vect, declared (isd:ied,jsd:jed)   sw_core.F90:2813
//   2  uf     divergence_corner                        sw_core.F90:1668
//   3  vf     divergence_corner                        sw_core.F90:1669
//   4  ke     c_sw (is-1:ie+1,js-1:je+1)               sw_core.F90:123
//   5  vort   c_sw (is-1:ie+1,js-1:je+1)               sw_core.F90:123
//   6  fx     c_sw (is-1:ie+2,js-1:je+1)               sw_core.F90:124
//   7  fx1    c_sw (is-1:ie+2,js-1:je+1)               sw_core.F90:124
//   8  fx2    c_sw (is-1:ie+2,js-1:je+1)               sw_core.F90:124
//   9  fy     c_sw (is-1:ie+1,js-1:je+2)               sw_core.F90:125
//  10  fy1    c_sw (is-1:ie+1,js-1:je+2)               sw_core.F90:125
//  11  fy2    c_sw (is-1:ie+1,js-1:je+2)               sw_core.F90:125
//
// No slot is shared between two live values. utmp/vtmp and uf/vf could in
// principle be recycled as fx2/fy2 (their lifetimes are disjoint within one
// level), but the runtime already provides exactly twelve arrays, so nothing is
// gained by aliasing and a great deal of reviewability is lost.
//
// Every array spans the FULL (isd:ied, jsd:jed) box. That is mandatory, not
// generous: fv3_cgrid_d2a2c.hpp:99-104 documents that d2a2c_vect prefills all
// of utmp/vtmp with big_number (sw_core.F90:2854-2856) and that the corner
// reflections read and write its outermost halo rings.
struct CGridPredictorArena {
  FieldView<fv_real, 3> utmp{}, vtmp{};
  FieldView<fv_real, 3> uf{}, vf{};
  FieldView<fv_real, 3> ke{}, vort{};
  FieldView<fv_real, 3> fx{}, fx1{}, fx2{};
  FieldView<fv_real, 3> fy{}, fy1{}, fy2{};

  // 2 (d2a2c) + 2 (divergence_corner) + 8 (c_sw)
  static constexpr int64_t array_count = 12;
};

// Shape one arena slot exactly as the runtime sized it. `k_lower` is the shared
// vertical lower bound of the registered 3-D state (see the k0 check below), so
// one `k` value indexes state and scratch identically -- the leaf launchers take
// a single per-level `k` and apply it to both.
FieldView<fv_real, 3> arena_view3(void *p, const DomainBounds &b,
                                  int64_t k_lower) {
  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t ni = (b.ie + h) - (b.is - h) + 1;
  const int64_t nj = (b.je + h) - (b.js - h) + 1;
  FieldView<fv_real, 3> v{};
  v.data = static_cast<fv_real *>(p);
  v.lower[0] = b.is - h;
  v.lower[1] = b.js - h;
  v.lower[2] = k_lower;
  v.stride[0] = 1;
  v.stride[1] = ni;
  v.stride[2] = ni * nj;
  return v;
}

// ---------------------------------------------------------------------------
// Local coverage predicates
// ---------------------------------------------------------------------------
// Deliberately local rather than reaching into dgrid/wp_b_contract_views.hpp:
// WP-A must not depend on a WP-B header. Same semantics as covers2/covers3
// there (that file, :83-96).
bool covers2(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
             int64_t hi_j) {
  if (!v || !v->data || v->rank < 2) return false;
  if (v->lower[0] > lo_i ||
      v->lower[0] + static_cast<int64_t>(v->extent[0]) - 1 < hi_i)
    return false;
  if (v->lower[1] > lo_j ||
      v->lower[1] + static_cast<int64_t>(v->extent[1]) - 1 < hi_j)
    return false;
  return true;
}

bool covers3(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
             int64_t hi_j, int64_t lo_k, int64_t hi_k) {
  if (!covers2(v, lo_i, hi_i, lo_j, hi_j)) return false;
  if (v->rank < 3) return false;
  if (v->lower[2] > lo_k ||
      v->lower[2] + static_cast<int64_t>(v->extent[2]) - 1 < hi_k)
    return false;
  return true;
}

struct Named {
  const DeviceViewBase *v;
  const char *name;
};

int missing(fv3_gpu_error *e, const char *fn, const std::string &what) {
  return fail(e, FV3_GPU_INVALID_ARGUMENT, fn, what);
}

int unsupported(fv3_gpu_error *e, const char *fn, const std::string &what) {
  return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, fn, what);
}

} // namespace

int launch_cgrid_predictor_v3(const ExecutionContext &context,
                              fv3_gpu_error *error) {
  int rc = validate_execution_context(context, error);
  if (rc != FV3_GPU_SUCCESS) return rc;

  if (context.submission.stream_id != StreamId::compute)
    return missing(error, __func__,
                   "C-grid predictor requires the WP-E compute stream");
  void *stream = context.submission.stream;
  if (!stream)
    return missing(error, __func__,
                   "C-grid predictor requires a borrowed non-null stream");
  if (!context.dependencies)
    return missing(error, __func__,
                   "C-grid predictor requires a bound DependencyState");
  if (!context.workspace)
    return missing(error, __func__,
                   "C-grid predictor requires a bound WorkspaceLayout");
  if (!context.immutable->grid)
    return missing(error, __func__,
                   "C-grid predictor requires a bound immutable grid registry");
  if (!context.immutable->topology)
    return missing(error, __func__,
                   "C-grid predictor requires a bound TopologyView");

  const auto &bounds = context.immutable->bounds;
  const auto &config = context.immutable->config;
  const auto &topo = *context.immutable->topology;
  const auto &params = *context.dispatch_params;

  // =========================================================================
  // Configuration fence. Everything here runs BEFORE the first kernel launch,
  // so a rejected configuration leaves model state untouched.
  // =========================================================================

  // hydrostatic. The hydrostatic arm of c_sw (sw_core.F90:208-234 and :258-282)
  // is not translated at all -- launch_csw_x_flux_level and
  // launch_csw_y_flux_scalars_level implement only the nonhydrostatic forms and
  // reject a hydrostatic config themselves. Refusing here as well keeps the
  // rejection ahead of the first launch, and ahead of the ut/vt scaling, which
  // has no hydrostatic guard and would otherwise run before the refusal.
  if (config.hydrostatic)
    return unsupported(error, __func__,
                       "hydrostatic c_sw (sw_core.F90:208-234, :258-282) is not "
                       "translated; the production first-run configuration is "
                       "hydrostatic = false");

  // nested. Three independent reasons, any one of which is disqualifying:
  //   * d2a2c_vect's nested branch (sw_core.F90:2858-2887) is a different
  //     algorithm and launch_d2a2c_vect_level rejects it
  //     (fv3_cgrid_d2a2c.cu:632);
  //   * c_sw calls divergence_corner_nest instead of divergence_corner
  //     (sw_core.F90:176-177), and divergence_corner_nest is not translated;
  //   * DivergenceCornerArgs carries no `nested` flag, so the nested
  //     is2/ie1 selection at sw_core.F90:1694-1698 cannot be expressed.
  //     fv3_cgrid_divergence.cu:6-12 explicitly instructs the first caller of
  //     launch_divergence_corner_level -- this file -- to reject nested here
  //     rather than let the global-cubed-sphere formula run on a nested grid.
  if (topo.nested)
    return unsupported(error, __func__,
                       "nested C-grid predictor is not translated "
                       "(divergence_corner_nest, sw_core.F90:1766-, and the "
                       "d2a2c_vect nested branch, :2858-2887)");

  // grid_type. c_sw branches on grid_type at :206, :236, :257, :284 (the corner
  // fills, `grid_type < 3`), :321 and :438 (`grid_type >= 3`), and d2a2c_vect
  // and divergence_corner branch on it again. Only the production value 0 is
  // exercised; anything else is refused rather than partially honoured.
  if (config.grid_type != 0)
    return unsupported(error, __func__,
                       "C-grid predictor supports grid_type = 0 only "
                       "(production first-run configuration)");
  if (!topo.global_cubed_sphere)
    return unsupported(error, __func__,
                       "grid_type = 0 requires a global cubed-sphere topology");

  // Halo width. launch_d2a2c_vect_level rejects ng < 3 (fv3_cgrid_d2a2c.cu:626)
  // because a2c_interpolate_uc_interior reads utmp(i-2,j) at i = is-1. c_sw's
  // own widest reach is delp(i-1,j) at i = is-1 and fx1(i+1,j) at i = ie+1,
  // i.e. is-2..ie+2, which also needs at least 2. ng is DomainBounds::halo_width
  // (fv3_gpu_internal.hpp:131) and is passed through as D2A2CArgs::ng.
  if (bounds.halo_width < 3)
    return unsupported(error, __func__,
                       "C-grid predictor requires halo_width >= 3");
  if (bounds.npz < 1)
    return unsupported(error, __func__, "C-grid predictor requires npz >= 1");
  if (bounds.ie < bounds.is || bounds.je < bounds.js)
    return missing(error, __func__, "C-grid predictor received empty bounds");

  // npx/npy. Every panel-edge and panel-corner formula in this group keys off
  // them: the c_sw KE edge cases (sw_core.F90:344-355, :365-376), the vorticity
  // flux edge cases (:463, :476), the corner vorticity removal (:415-418), the
  // fill_4corners/fill2_4corners index expressions (:3249-3282, :3310-3343) and
  // d2a2c_vect's npt banding. A wrong value applies an interior stencil at a
  // seam, so refuse to guess. Same policy as the WP-B binder
  // (dgrid/wp_b_dsw_contract_cuda.cpp:78-80).
  if (topo.npx <= 0 || topo.npy <= 0)
    return missing(error, __func__,
                   "TopologyView npx/npy must be supplied for the C-grid "
                   "predictor's panel-edge and corner semantics");
  if (topo.npx > INT32_MAX || topo.npy > INT32_MAX)
    return missing(error, __func__, "TopologyView npx/npy exceed int32");

  // -------------------------------------------------------------------------
  // RESOLVED DEFECT (Milestone 3A.9). Kept as a record because the constraint
  // it describes is what the per-level loop below is shaped around.
  //
  // c_sw interleaves the two corner-fill directions with the two flux phases:
  //
  //   :206  fill2_4corners(delp, pt, 1)
  //   :237  fill_4corners (w,       1)
  //   :238-253  x fluxes  fx1 / fx / fx2   <-- must see the dir = 1 values
  //   :257  fill2_4corners(delp, pt, 2)
  //   :284  fill_4corners (w,       2)
  //   :285-300  y fluxes  fy1 / fy / fy2   <-- must see the dir = 2 values
  //
  // The two directions are NOT independent: both write the same four cells with
  // DIFFERENT values. At the SW corner, fill_4corners gives
  //   dir 1 (:3312):  q(0,0) = q(0,1)
  //   dir 2 (:3329):  q(0,0) = q(1,0)
  // and likewise q(npx,0), q(0,npy), q(npx,npy) at the other three corners
  // (:3316/:3333, :3319/:3337, :3323/:3341; fill2_4corners :3250/:3268 etc.).
  //
  // The former launch_csw_fluxes_level enqueued the ut/vt scaling, BOTH flux
  // directions AND the delpc/ptc/wc update as one indivisible sequence, so there
  // was no point at which a caller could substitute the dir = 2 corner values
  // between the x and the y phase, and whichever direction was left in place was
  // wrong for one of them.
  //
  // That was observable, not cosmetic. With dir 1 then dir 2 applied up front,
  // the x fluxes read q(0,0) = q(1,0) instead of q(0,1), so fx1(0,0) and
  // fx1(1,0) are wrong, hence delpc/ptc/wc at (is-1,js-1) AND at (is,js-1);
  // delpc(is,js-1) feeds Riem_Solver_C and then p_grad_c's vc update, so the
  // error reaches the prognostic C winds. The reverse fill order moves the same
  // error onto the y fluxes and into the uc update. Invoking the launcher twice
  // was not a workaround either: the ut/vt scaling is in-place and not
  // idempotent (:186-198).
  //
  // fv3_cgrid_csw.{hpp,cu} now expose the two seams the Fortran has --
  // launch_csw_scale_level / launch_csw_x_flux_level /
  // launch_csw_y_flux_scalars_level -- so the authoritative sequence is emitted
  // for every rank, including one owning panel corners (which every rank does in
  // a single-rank global cubed-sphere run, where all four corner flags are set).
  // The earlier any_corner rejection is therefore gone.
  // -------------------------------------------------------------------------

  // =========================================================================
  // Field resolution
  // =========================================================================
  const auto state = [&](FieldId id) { return context.fields->find(id); };
  const auto grid = [&](FieldId id) { return context.immutable->grid->find(id); };

  // c_sw's own dummy arguments (sw_core.F90:103-113) plus the metrics its
  // callees read. `omga` is c_sw's `wc` (dyn_core.F90:518).
  const auto *v_u = state(FieldId::u);
  const auto *v_v = state(FieldId::v);
  const auto *v_w = state(FieldId::w);
  const auto *v_delp = state(FieldId::delp);
  const auto *v_pt = state(FieldId::pt);
  const auto *v_ua = state(FieldId::ua);
  const auto *v_va = state(FieldId::va);
  const auto *v_uc = state(FieldId::uc);
  const auto *v_vc = state(FieldId::vc);
  const auto *v_ut = state(FieldId::ut);
  const auto *v_vt = state(FieldId::vt);
  const auto *v_delpc = state(FieldId::delpc);
  const auto *v_ptc = state(FieldId::ptc);
  const auto *v_omga = state(FieldId::omga);
  const auto *v_divgd = state(FieldId::divgd);

  const auto *g_dx = grid(FieldId::dx);
  const auto *g_dy = grid(FieldId::dy);
  const auto *g_dxc = grid(FieldId::dxc);
  const auto *g_dyc = grid(FieldId::dyc);
  const auto *g_rdxc = grid(FieldId::rdxc);
  const auto *g_rdyc = grid(FieldId::rdyc);
  const auto *g_rarea = grid(FieldId::rarea);
  const auto *g_rarea_c = grid(FieldId::rarea_c);
  const auto *g_cosa_u = grid(FieldId::cosa_u);
  const auto *g_cosa_v = grid(FieldId::cosa_v);
  const auto *g_cosa_s = grid(FieldId::cosa_s);
  const auto *g_sina_u = grid(FieldId::sina_u);
  const auto *g_sina_v = grid(FieldId::sina_v);
  const auto *g_rsin_u = grid(FieldId::rsin_u);
  const auto *g_rsin_v = grid(FieldId::rsin_v);
  const auto *g_rsin2 = grid(FieldId::rsin2);
  const auto *g_dxa = grid(FieldId::dxa);
  const auto *g_dya = grid(FieldId::dya);
  const auto *g_f_c = grid(FieldId::f_c);
  const auto *g_sin_sg = grid(FieldId::sin_sg);
  const auto *g_cos_sg = grid(FieldId::cos_sg);

  const Named all[] = {
      {v_u, "u"},             {v_v, "v"},
      {v_w, "w"},             {v_delp, "delp"},
      {v_pt, "pt"},           {v_ua, "ua"},
      {v_va, "va"},           {v_uc, "uc"},
      {v_vc, "vc"},           {v_ut, "ut"},
      {v_vt, "vt"},           {v_delpc, "delpc"},
      {v_ptc, "ptc"},         {v_omga, "omga (c_sw wc)"},
      {v_divgd, "divgd"},     {g_dx, "dx"},
      {g_dy, "dy"},           {g_dxc, "dxc"},
      {g_dyc, "dyc"},         {g_rdxc, "rdxc"},
      {g_rdyc, "rdyc"},       {g_rarea, "rarea"},
      {g_rarea_c, "rarea_c"}, {g_cosa_u, "cosa_u"},
      {g_cosa_v, "cosa_v"},   {g_cosa_s, "cosa_s"},
      {g_sina_u, "sina_u"},   {g_sina_v, "sina_v"},
      {g_rsin_u, "rsin_u"},   {g_rsin_v, "rsin_v"},
      {g_rsin2, "rsin2"},     {g_dxa, "dxa"},
      {g_dya, "dya"},         {g_f_c, "f_c"},
      {g_sin_sg, "sin_sg"},   {g_cos_sg, "cos_sg"}};
  for (const Named &f : all) {
    if (!f.v || !f.v->data)
      return missing(error, __func__,
                     std::string("C-grid predictor field ") + f.name +
                         " is not registered");
    // Every field this slot touches is LogicalType::fv_real
    // (runtime/fv3_gpu_contract.cpp:11-77), so a width mismatch is a
    // registration defect, never a precision choice. Asserted explicitly
    // because `stride` is in elements and nothing else in DeviceViewBase
    // records the storage width (fv3_gpu_internal.hpp:70-90).
    if (!view_element_width_is(f.v, sizeof(fv_real)))
      return missing(error, __func__,
                     std::string("C-grid predictor field ") + f.name +
                         " element width does not match the active fv_real "
                         "precision");
  }

  // =========================================================================
  // Storage-shape fence
  // =========================================================================
  const int64_t h = static_cast<int64_t>(bounds.halo_width);
  const int64_t isd = bounds.is - h, ied = bounds.ie + h;
  const int64_t jsd = bounds.js - h, jed = bounds.je + h;

  // All 3-D state must share one vertical lower bound: the leaf launchers take a
  // single per-level `k` and apply it to every 3-D view they hold, including the
  // arena. fv3_cgrid.cu:12 states the same rule ("kernels never assume Fortran k
  // starts at one"), so the value is taken from the registration rather than
  // assumed to be 0 or 1.
  const int64_t k0 = v_delp->lower[2];
  const int64_t k1 = k0 + bounds.npz - 1;
  const Named levels3[] = {{v_u, "u"},         {v_v, "v"},
                           {v_w, "w"},         {v_delp, "delp"},
                           {v_pt, "pt"},       {v_ua, "ua"},
                           {v_va, "va"},       {v_uc, "uc"},
                           {v_vc, "vc"},       {v_ut, "ut"},
                           {v_vt, "vt"},       {v_delpc, "delpc"},
                           {v_ptc, "ptc"},     {v_omga, "omga (c_sw wc)"},
                           {v_divgd, "divgd"}};
  for (const Named &f : levels3) {
    if (f.v->rank < 3)
      return missing(error, __func__,
                     std::string("C-grid predictor field ") + f.name +
                         " must be rank 3");
    if (f.v->lower[2] != k0)
      return missing(error, __func__,
                     std::string("C-grid predictor field ") + f.name +
                         " has a different vertical lower bound than delp; the "
                         "per-level launchers index all 3-D views with one k");
  }

  // Fortran declared extents of c_sw's dummies (sw_core.F90:108-113). These are
  // the contract, so they are required in full rather than reduced to the
  // subset the kernels happen to touch.
  const struct {
    const DeviceViewBase *v;
    const char *name;
    int64_t hi_i, hi_j;
  } boxes3[] = {
      {v_u, "u", ied, jed + 1},          {v_vc, "vc", ied, jed + 1},
      {v_v, "v", ied + 1, jed},          {v_uc, "uc", ied + 1, jed},
      {v_divgd, "divgd", ied + 1, jed + 1},
      {v_delp, "delp", ied, jed},        {v_pt, "pt", ied, jed},
      {v_ua, "ua", ied, jed},            {v_va, "va", ied, jed},
      {v_ut, "ut", ied, jed},            {v_vt, "vt", ied, jed},
      {v_w, "w", ied, jed},              {v_delpc, "delpc", ied, jed},
      {v_ptc, "ptc", ied, jed},          {v_omga, "omga (c_sw wc)", ied, jed}};
  for (const auto &b3 : boxes3)
    if (!covers3(b3.v, isd, b3.hi_i, jsd, b3.hi_j, k0, k1))
      return missing(error, __func__,
                     std::string("C-grid predictor field ") + b3.name +
                         " does not cover its Fortran declared extent "
                         "(sw_core.F90:108-113) over [k0, k0+npz-1]");

  // 2-D metrics. The A box (isd:ied, jsd:jed) is a subset of every stagger's
  // declared box (C-u adds an i ring, C-v a j ring, B both), so it is the
  // correct common minimum to assert; the extra rings are recorded by the
  // FieldSpec stagger and were validated by make_device_view.
  const Named metrics2[] = {
      {g_dx, "dx"},           {g_dy, "dy"},         {g_dxc, "dxc"},
      {g_dyc, "dyc"},         {g_rdxc, "rdxc"},     {g_rdyc, "rdyc"},
      {g_rarea, "rarea"},     {g_rarea_c, "rarea_c"}, {g_cosa_u, "cosa_u"},
      {g_cosa_v, "cosa_v"},   {g_cosa_s, "cosa_s"}, {g_sina_u, "sina_u"},
      {g_sina_v, "sina_v"},   {g_rsin_u, "rsin_u"}, {g_rsin_v, "rsin_v"},
      {g_rsin2, "rsin2"},     {g_dxa, "dxa"},       {g_dya, "dya"},
      {g_f_c, "f_c"}};
  for (const Named &f : metrics2)
    if (!covers2(f.v, isd, ied, jsd, jed))
      return missing(error, __func__,
                     std::string("C-grid predictor metric ") + f.name +
                         " does not cover (isd:ied, jsd:jed)");

  // sin_sg / cos_sg carry the four Fortran components 1..4 on a 0-BASED third
  // axis -- the kernels use literal 0..3 for Fortran 1..4 (for example
  // fv3_cgrid_csw.cu:30 uses component 2 for sin_sg(i-1,j,3)). That convention
  // is fixed by the existing tests (tests/dgrid/test_wp_b5_dsw_cuda.cpp:194-195),
  // so a non-zero lower bound would silently shift every component.
  const Named sg[] = {{g_sin_sg, "sin_sg"}, {g_cos_sg, "cos_sg"}};
  for (const Named &f : sg) {
    if (f.v->rank < 3 || f.v->lower[2] != 0)
      return missing(error, __func__,
                     std::string("C-grid predictor metric ") + f.name +
                         " must be rank 3 with a 0-based component axis");
    if (!covers3(f.v, isd, ied, jsd, jed, 0, 3))
      return missing(error, __func__,
                     std::string("C-grid predictor metric ") + f.name +
                         " must supply components 0..3 over (isd:ied, jsd:jed)");
  }

  // =========================================================================
  // Arena carve
  // =========================================================================
  const WorkspaceSlice *ws = workspace_slice(*context.workspace, WorkspaceId::cgrid);
  if (!ws || !ws->data)
    return missing(error, __func__, "WorkspaceId::cgrid slice is absent");

  const uint64_t ni = static_cast<uint64_t>(ied - isd + 1);
  const uint64_t nj = static_cast<uint64_t>(jed - jsd + 1);
  const uint64_t nlev = static_cast<uint64_t>(bounds.npz) + 1u;
  // Overflow-safe running product. The runtime performs the identical
  // computation when it sizes the slice (runtime/fv3_gpu_runtime.cpp:188-203),
  // so this both validates and reproduces it.
  const auto mul_into = [](uint64_t &acc, uint64_t factor) {
    if (factor != 0 && acc > UINT64_MAX / factor) return true;
    acc *= factor;
    return false;
  };
  uint64_t values = 1;
  if (mul_into(values, ni) || mul_into(values, nj) || mul_into(values, nlev))
    return unsupported(error, __func__, "C-grid arena size overflowed");
  uint64_t bytes_per_array = values;
  if (mul_into(bytes_per_array, sizeof(fv_real)))
    return unsupported(error, __func__, "C-grid arena size overflowed");
  uint64_t total = bytes_per_array;
  if (mul_into(total, static_cast<uint64_t>(CGridPredictorArena::array_count)))
    return unsupported(error, __func__, "C-grid arena size overflowed");
  // Loud, before any launch. Never a truncation and never an out-of-bounds
  // carve: WorkspaceId::cgrid must supply at least array_count whole
  // (ni x nj x (npz+1)) fv_real arrays.
  if (ws->bytes < total)
    return unsupported(error, __func__,
                       "WorkspaceId::cgrid slice is smaller than "
                       "CGridPredictorArena::array_count (12) whole "
                       "(ni x nj x (npz+1)) fv_real arrays");

  char *base = static_cast<char *>(ws->data);
  const auto slot = [&](int64_t n) {
    return static_cast<void *>(base + static_cast<uint64_t>(n) * bytes_per_array);
  };

  CGridPredictorArena arena{};
  arena.utmp = arena_view3(slot(0), bounds, k0);   // sw_core.F90:2813
  arena.vtmp = arena_view3(slot(1), bounds, k0);   // sw_core.F90:2813
  arena.uf = arena_view3(slot(2), bounds, k0);     // sw_core.F90:1668
  arena.vf = arena_view3(slot(3), bounds, k0);     // sw_core.F90:1669
  arena.ke = arena_view3(slot(4), bounds, k0);     // sw_core.F90:123
  arena.vort = arena_view3(slot(5), bounds, k0);   // sw_core.F90:123
  arena.fx = arena_view3(slot(6), bounds, k0);     // sw_core.F90:124
  arena.fx1 = arena_view3(slot(7), bounds, k0);    // sw_core.F90:124
  arena.fx2 = arena_view3(slot(8), bounds, k0);    // sw_core.F90:124
  arena.fy = arena_view3(slot(9), bounds, k0);     // sw_core.F90:125
  arena.fy1 = arena_view3(slot(10), bounds, k0);   // sw_core.F90:125
  arena.fy2 = arena_view3(slot(11), bounds, k0);   // sw_core.F90:125

  // =========================================================================
  // Halo prerequisites
  // =========================================================================
  // u/v D-grid halo. dyn_core.F90:504 completes i_pack(8) -- the DGRID_NE u,v
  // exchange started at :402 (ROT3) or, for substeps after the first, at :1109
  // -- immediately before the c_sw loop at :514. d2a2c_vect reads u and v out to
  // isd/ied/jsd/jed (sw_core.F90:2893-2902), so the halo must genuinely be
  // valid, not merely present.
  fv3_gpu_event dwind_event = 0;
  rc = context.dependencies->require(FV3_GPU_H_DWIND_IN, &dwind_event);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "H_DWIND_IN (u/v halo, dyn_core.F90:504) is not ready");

  // w halo. dyn_core.F90:506-507 completes i_pack(7) (started :427) under
  // `.not. hydrostatic`, again immediately before the c_sw loop. c_sw's
  // nonhydrostatic x/y flux phases read w(i-1,j) and w(i,j-1) into the halo
  // (sw_core.F90:243, :290).
  fv3_gpu_event w_event = 0;
  rc = context.dependencies->require(FV3_GPU_H_W_IN, &w_event);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "H_W_IN (w halo, dyn_core.F90:507) is not ready");

  // H_GZ_FIRST is deliberately NOT required here: gz is consumed by update_dz_c
  // (dyn_core.F90:551-566), which is the cgrid_vertical_solver slot, not this
  // one. c_sw never reads gz.
  //
  // Single compute stream, so there is nothing to wait on; the events are the
  // readiness receipts the runtime hands back and cross-stream ordering remains
  // WP-E's business.
  (void)dwind_event;
  (void)w_event;

  // =========================================================================
  // Argument packs
  // =========================================================================
  const Bounds b{bounds.is, bounds.ie, bounds.js, bounds.je,
                 static_cast<int64_t>(bounds.npz)};
  const auto npx = static_cast<int32_t>(topo.npx);
  const auto npy = static_cast<int32_t>(topo.npy);
  // Corner flags reach the leaves unchanged. They are all zero here (any_corner
  // is fenced above), but they are still propagated so that lifting the fence
  // needs no change on this side.
  const auto swc = static_cast<uint8_t>(topo.sw_corner ? 1 : 0);
  const auto sec = static_cast<uint8_t>(topo.se_corner ? 1 : 0);
  const auto nec = static_cast<uint8_t>(topo.ne_corner ? 1 : 0);
  const auto nwc = static_cast<uint8_t>(topo.nw_corner ? 1 : 0);

  // delp / pt / w are INTENT(INOUT) in c_sw (sw_core.F90:110-111) precisely
  // because fill2_4corners / fill_4corners mutate their corner halo in place.
  // A writable 3-D view is needed for level_slice; the read-only views handed
  // to the flux kernels are separate and const.
  FieldView<fv_real, 3> delp_rw = make_view<fv_real, 3>(*v_delp);
  FieldView<fv_real, 3> pt_rw = make_view<fv_real, 3>(*v_pt);
  FieldView<fv_real, 3> w_rw = make_view<fv_real, 3>(*v_w);

  // --- d2a2c_vect (sw_core.F90:172-173, 2800-3139) ---
  D2A2CArgs d2a{};
  d2a.bounds = b;
  d2a.ng = static_cast<int32_t>(bounds.halo_width);  // fv3_cgrid_d2a2c.hpp:28-40
  d2a.npx = npx;
  d2a.npy = npy;
  d2a.grid_type = config.grid_type;
  d2a.nested = 0;  // fenced above
  d2a.dord4 = 1;   // dyn_core.F90:521 passes `.true.` for c_sw's dord4, which
                   // c_sw forwards to d2a2c_vect at :172
  d2a.sw_corner = swc;
  d2a.se_corner = sec;
  d2a.ne_corner = nec;
  d2a.nw_corner = nwc;
  d2a.u = make_view<const fv_real, 3>(*v_u);
  d2a.v = make_view<const fv_real, 3>(*v_v);
  d2a.ua = make_view<fv_real, 3>(*v_ua);
  d2a.va = make_view<fv_real, 3>(*v_va);
  d2a.uc = make_view<fv_real, 3>(*v_uc);
  d2a.vc = make_view<fv_real, 3>(*v_vc);
  d2a.ut = make_view<fv_real, 3>(*v_ut);
  d2a.vt = make_view<fv_real, 3>(*v_vt);
  d2a.cosa_u = make_view<const fv_real, 2>(*g_cosa_u);
  d2a.cosa_v = make_view<const fv_real, 2>(*g_cosa_v);
  d2a.rsin_u = make_view<const fv_real, 2>(*g_rsin_u);
  d2a.rsin_v = make_view<const fv_real, 2>(*g_rsin_v);
  d2a.cosa_s = make_view<const fv_real, 2>(*g_cosa_s);
  d2a.rsin2 = make_view<const fv_real, 2>(*g_rsin2);
  d2a.dxa = make_view<const fv_real, 2>(*g_dxa);
  d2a.dya = make_view<const fv_real, 2>(*g_dya);
  d2a.sin_sg = make_view<const fv_real, 3>(*g_sin_sg);
  d2a.utmp = arena.utmp;
  d2a.vtmp = arena.vtmp;

  // --- divergence_corner (sw_core.F90:175-181, 1659-1764) ---
  DivergenceCornerArgs dvg{};
  dvg.bounds = b;
  dvg.npx = npx;
  dvg.npy = npy;
  dvg.grid_type = config.grid_type;
  dvg.sw_corner = swc;
  dvg.se_corner = sec;
  dvg.ne_corner = nec;
  dvg.nw_corner = nwc;
  dvg.u = make_view<const fv_real, 3>(*v_u);
  dvg.v = make_view<const fv_real, 3>(*v_v);
  dvg.ua = make_view<const fv_real, 3>(*v_ua);
  dvg.va = make_view<const fv_real, 3>(*v_va);
  dvg.divg_d = make_view<fv_real, 3>(*v_divgd);
  dvg.dxc = make_view<const fv_real, 2>(*g_dxc);
  dvg.dyc = make_view<const fv_real, 2>(*g_dyc);
  dvg.rarea_c = make_view<const fv_real, 2>(*g_rarea_c);
  dvg.sin_sg = make_view<const fv_real, 3>(*g_sin_sg);
  dvg.cos_sg = make_view<const fv_real, 3>(*g_cos_sg);
  dvg.uf = arena.uf;
  dvg.vf = arena.vf;

  // --- corner fills (sw_core.F90:3227-3347) ---
  CornerFillArgs cf{};
  cf.npx = npx;
  cf.npy = npy;
  cf.sw_corner = swc;
  cf.se_corner = sec;
  cf.ne_corner = nec;
  cf.nw_corner = nwc;

  // --- c_sw body (sw_core.F90:183-510) ---
  CSwArgs csw{};
  csw.bounds = b;
  csw.npx = npx;
  csw.npy = npy;
  csw.nord = config.nord;
  // dyn_core.F90:521 passes dt2, not dt. DispatchParams::dt2 already holds
  // 0.5*bdt/n_split (dyn_core.F90:296-297); nothing is re-derived here.
  csw.dt2 = static_cast<fv_real>(params.dt2);
  csw.grid_type = config.grid_type;
  csw.nested = 0;       // fenced above
  csw.hydrostatic = 0;  // fenced above
  csw.sw_corner = swc;
  csw.se_corner = sec;
  csw.ne_corner = nec;
  csw.nw_corner = nwc;
  csw.u = make_view<fv_real, 3>(*v_u);
  csw.v = make_view<fv_real, 3>(*v_v);
  csw.ua = make_view<fv_real, 3>(*v_ua);
  csw.va = make_view<fv_real, 3>(*v_va);
  csw.uc = make_view<fv_real, 3>(*v_uc);
  csw.vc = make_view<fv_real, 3>(*v_vc);
  csw.ut = make_view<fv_real, 3>(*v_ut);
  csw.vt = make_view<fv_real, 3>(*v_vt);
  csw.delp = make_view<const fv_real, 3>(*v_delp);
  csw.pt = make_view<const fv_real, 3>(*v_pt);
  csw.w = w_rw;
  csw.delpc = make_view<fv_real, 3>(*v_delpc);
  csw.ptc = make_view<fv_real, 3>(*v_ptc);
  csw.wc = make_view<fv_real, 3>(*v_omga);  // dyn_core.F90:518: wc <- omga
  csw.divg_d = make_view<fv_real, 3>(*v_divgd);
  csw.dx = make_view<const fv_real, 2>(*g_dx);
  csw.dy = make_view<const fv_real, 2>(*g_dy);
  csw.dxc = make_view<const fv_real, 2>(*g_dxc);
  csw.dyc = make_view<const fv_real, 2>(*g_dyc);
  csw.rdxc = make_view<const fv_real, 2>(*g_rdxc);
  csw.rdyc = make_view<const fv_real, 2>(*g_rdyc);
  csw.rarea = make_view<const fv_real, 2>(*g_rarea);
  csw.cosa_u = make_view<const fv_real, 2>(*g_cosa_u);
  csw.cosa_v = make_view<const fv_real, 2>(*g_cosa_v);
  csw.sina_u = make_view<const fv_real, 2>(*g_sina_u);
  csw.sina_v = make_view<const fv_real, 2>(*g_sina_v);
  csw.f_c = make_view<const fv_real, 2>(*g_f_c);  // gridstruct%fC, :425
  csw.rarea_c = make_view<const fv_real, 2>(*g_rarea_c);
  csw.sin_sg = make_view<const fv_real, 3>(*g_sin_sg);
  csw.cos_sg = make_view<const fv_real, 3>(*g_cos_sg);
  csw.vort = arena.vort;
  csw.ke = arena.ke;
  csw.fx = arena.fx;
  csw.fy = arena.fy;
  csw.fx1 = arena.fx1;
  csw.fy1 = arena.fy1;
  csw.fx2 = arena.fx2;
  csw.fy2 = arena.fy2;

  // =========================================================================
  // Execution. dyn_core.F90:514-524 is `do k=1,npz ; call c_sw(...,k) ; enddo`,
  // and the levels are independent (the OMP loop at :512 declares no reduction
  // and no cross-k shared temporary). The loop is reproduced k-outer, phase-
  // inner so that this file reads in the same order as subroutine c_sw and every
  // level's scratch is written and consumed inside one iteration. Everything is
  // enqueued on the borrowed compute stream: no host staging, no per-phase
  // device synchronization, so persistent device residency is preserved.
  // =========================================================================
  const bool do_divergence_corner = config.nord > 0;  // sw_core.F90:175

  for (int64_t level = 0; level < bounds.npz; ++level) {
    const int64_t k = k0 + level;

    // P1 -- d2a2c_vect (sw_core.F90:172-173). Produces ua/va/uc/vc/ut/vt.
    // Called from inside c_sw, before anything else in the body.
    rc = launch_d2a2c_vect_level(d2a, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue d2a2c_vect (sw_core.F90:172)");

    // P2 -- divergence_corner (sw_core.F90:175-181), only when nord > 0. Reads
    // u/v and the ua/va that P1 just wrote; writes divg_d. The nested variant
    // divergence_corner_nest (:177) is fenced above.
    if (do_divergence_corner) {
      rc = launch_divergence_corner_level(dvg, k, stream);
      if (rc != FV3_GPU_SUCCESS)
        return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                    "failed to enqueue divergence_corner (sw_core.F90:179)");
    }

    // P5 -- ut scaling by dy and sin_sg   sw_core.F90:183-191
    // P6 -- vt scaling by dx and sin_sg   sw_core.F90:192-200
    // Emitted before the corner fills, exactly as the Fortran orders them.
    rc = launch_csw_scale_level(csw, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue c_sw ut/vt scaling (sw_core.F90:183-200)");

    // P3 -- fill2_4corners(delp, pt, 1) (sw_core.F90:206) and
    // P4 -- fill_4corners(w, 1)        (sw_core.F90:236-237).
    // Both are inside `grid_type < 3 .and. .not. nested`, which holds (grid_type
    // is fenced to 0). They must precede the x flux phase, which reads
    // delp(i-1,j), pt(i-1,j) and w(i-1,j) into the corner halo (:241-243).
    // These are WP-A-local corner algebra on this rank's own halo, so they use
    // the plain WP-A launchers and are NOT routed through
    // context.corner_service (fv3_cgrid_corners.hpp:8-13).
    rc = launch_fill2_4corners(cf, level_slice(delp_rw, k), level_slice(pt_rw, k),
                               kCornerFillDirX, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue fill2_4corners dir=1 (sw_core.F90:206)");
    rc = launch_fill_4corners(cf, level_slice(w_rw, k), kCornerFillDirX, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue fill_4corners dir=1 (sw_core.F90:237)");

    // P7 -- x fluxes fx1 / fx / fx2 (sw_core.F90:238-253). Reads the dir = 1
    // corner values that P3/P4 just installed.
    rc = launch_csw_x_flux_level(csw, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue c_sw x flux phase (sw_core.F90:238-253)");

    // P8 -- fill2_4corners(delp, pt, 2) (sw_core.F90:257) and
    // P9 -- fill_4corners(w, 2)         (sw_core.F90:284).
    // Their authoritative position, between the two flux phases. They overwrite
    // the four corner cells the dir = 1 fills wrote, with different values
    // (:3250 vs :3268, :3312 vs :3329, ...), which is why the flux launcher had
    // to be split. Also the state in which delp/pt/w LEAVE this stage: c_sw
    // mutates them in place and nothing in dyn_core restores or re-exchanges
    // them, so d_sw and everything after observe the dir = 2 values.
    rc = launch_fill2_4corners(cf, level_slice(delp_rw, k), level_slice(pt_rw, k),
                               kCornerFillDirY, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue fill2_4corners dir=2 (sw_core.F90:257)");
    rc = launch_fill_4corners(cf, level_slice(w_rw, k), kCornerFillDirY, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue fill_4corners dir=2 (sw_core.F90:284)");

    // P10 -- y fluxes fy1 / fy / fy2 (sw_core.F90:285-300) and
    // P11 -- delpc / ptc / wc        (sw_core.F90:301-309).
    // Kept in one launch: no corner fill separates them in the Fortran and
    // :301-309 reads the fy* just written.
    rc = launch_csw_y_flux_scalars_level(csw, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue c_sw y flux / scalar phase "
                  "(sw_core.F90:285-309)");

    // P12..P17 -- one launcher call covering, in Fortran order:
    //   P12  KE, using the cubed-sphere edge forms   sw_core.F90:321-390
    //   P13  circulation fx = uc*dxc, fy = vc*dyc    sw_core.F90:396-406
    //   P14  vort from the circulation               sw_core.F90:408-412
    //   P15  corner extra-term removal               sw_core.F90:415-418
    //   P16  absolute vorticity fC + rarea_c*vort    sw_core.F90:423-427
    //   P17  vorticity transport fy1/fy and fx1/fx   sw_core.F90:438-498
    // P13 legitimately overwrites the fx/fy that P7/P8 used: by :396 the
    // delpc/ptc/wc update at :301-309 has consumed them.
    rc = launch_csw_ke_vort_level(csw, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue c_sw KE/vorticity phase "
                  "(sw_core.F90:321-498)");

    // P18 -- time-centered C-wind update (sw_core.F90:500-510). Reads the fy1/fy
    // and fx1/fx written by P17 and the ke written by P12.
    rc = launch_csw_cwind_update_level(csw, k, stream);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to enqueue c_sw C-wind update (sw_core.F90:500-510)");
  }

  // =========================================================================
  // Halo invalidation
  // =========================================================================
  // divg_d is written over the compute box only (sw_core.F90:1700-1764), so its
  // halo is stale. dyn_core.F90:525-531 exchanges it -- i_pack(3), position =
  // CORNER -- immediately after the c_sw loop and only when nord > 0, which is
  // exactly when this slot wrote it.
  if (do_divergence_corner) {
    rc = context.dependencies->mark_stale(FV3_GPU_H_DIVGD);
    if (rc != FV3_GPU_SUCCESS)
      return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                  "failed to mark divg_d halo stale");
  }

  // uc/vc are updated over (is:ie+1, js:je) and (is:ie, js:je+1) only
  // (sw_core.F90:501-509), so their halos are stale from here on. The
  // authoritative exchange is i_pack(9), CGRID_NE, started at dyn_core.F90:618
  // after p_grad_c and completed at :640 before d_sw, so the group stays stale
  // across the intervening cgrid_vertical_solver and cgrid_pressure slots --
  // p_grad_c (dyn_core.F90 p_grad_c) also writes uc/vc and marks it stale again
  // (fv3_cgrid.cu:122).
  rc = context.dependencies->mark_stale(FV3_GPU_H_CWIND);
  if (rc != FV3_GPU_SUCCESS)
    return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                "failed to mark C-wind halo stale");

  // Nothing else is marked stale, on purpose:
  //   * delp/pt/w corner halos are mutated by the fills, but the Fortran does
  //     exactly the same and never re-exchanges them, so the CUDA state matches
  //     the authoritative state and there is no exchange to demand.
  //   * ua/va/ut/vt/delpc/ptc/omga are shared temporaries with no halo group.
  //   * u/v are not written by c_sw, so H_DWIND_NEXT is untouched here.
  // No group is marked ready anywhere in this file: readiness is the runtime's
  // to assert.

  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::cgrid
