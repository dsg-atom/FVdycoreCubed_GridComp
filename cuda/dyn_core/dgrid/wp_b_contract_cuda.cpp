#include "dgrid/wp_b_contract_cuda.hpp"
#include "dgrid/wp_b_cuda.cuh"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_types.hpp"

#include <cstdint>
#include <cstring>

namespace fv3::gpu::dgrid {
namespace {

template <class T>
View3<T> to_view3(const DeviceViewBase *v) {
  return View3<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->lower[2],
                  v->stride[0], v->stride[1], v->stride[2]};
}

template <class T>
View2<T> to_view2(const DeviceViewBase *v) {
  return View2<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->stride[0],
                  v->stride[1]};
}

template <class T>
View1<T> to_view1(const DeviceViewBase *v) {
  return View1<T>{static_cast<T *>(v->data), v->lower[0], v->stride[0]};
}

View3<const fv_real> to_const3(const View3<fv_real> &v) {
  return View3<const fv_real>{v.data, v.lo_i, v.lo_j, v.lo_k, v.si, v.sj, v.sk};
}

// Arena arrays are all sized identically ([is-h,ie+h] x [js-h,je+h] x
// [0,npz]) by bind_wp_b_height_transport; wrap each raw pointer as a batched
// View3 over that box (matches WpBHeightTransportArena's documented layout).
View3<fv_real> to_arena_view3(void *p, const DomainBounds &b) {
  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t ni = (b.ie + h) - (b.is - h) + 1;
  const int64_t nj = (b.je + h) - (b.js - h) + 1;
  return View3<fv_real>{static_cast<fv_real *>(p), b.is - h, b.js - h, 0, 1, ni, ni * nj};
}

// gam is a length-npz column living in the first values of one arena slot.
View1<fv_real> to_arena_view1(void *p) {
  return View1<fv_real>{static_cast<fv_real *>(p), 0, 1};
}

View1<const fv_real> to_const1(const View1<fv_real> &v) {
  return View1<const fv_real>{v.data, v.lo_i, v.si};
}

bool covers1(const DeviceViewBase *v, int64_t lo, int64_t hi) {
  if (!v || v->rank < 1 || !v->data) return false;
  if (v->lower[0] > lo) return false;
  if (v->lower[0] + static_cast<int64_t>(v->extent[0]) - 1 < hi) return false;
  return true;
}

bool covers2(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
             int64_t hi_j) {
  if (!v || v->rank < 2) return false;
  if (v->lower[0] > lo_i || v->lower[0] + static_cast<int64_t>(v->extent[0]) - 1 < hi_i)
    return false;
  if (v->lower[1] > lo_j || v->lower[1] + static_cast<int64_t>(v->extent[1]) - 1 < hi_j)
    return false;
  return true;
}

bool covers3(const DeviceViewBase *v, int64_t lo_i, int64_t hi_i, int64_t lo_j,
             int64_t hi_j, int64_t lo_k, int64_t hi_k) {
  if (!covers2(v, lo_i, hi_i, lo_j, hi_j)) return false;
  if (v->rank < 3) return false;
  if (v->lower[2] > lo_k || v->lower[2] + static_cast<int64_t>(v->extent[2]) - 1 < hi_k)
    return false;
  return true;
}

// Overflow-safe uint64_t multiply-accumulate, mirroring the arena-sizing
// rigor used by wp_d_contract.cpp's bind_wp_d.
bool mul_into(uint64_t &acc, uint64_t factor) {
  if (factor != 0 && acc > (UINT64_MAX / factor)) return true;
  acc *= factor;
  return false;
}

} // namespace

int bind_wp_b_height_transport(const ExecutionContext &x, WpBHeightTransportViews &out,
                               WpBHeightTransportArena &arena, fv3_gpu_error *error) noexcept {
  int rc = validate_execution_context(x, error);
  if (rc) return rc;
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "dgrid_height_transport requires the borrowed compute submission");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const int32_t mord = cfg.hord_tm > 0 ? cfg.hord_tm : -cfg.hord_tm;
  if (mord != 6)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport implements the mord == 6 PPM path only");
  // fv_tp_2d's stencil reaches +/-3 relative to is/ie/js/je (isd = is - ng,
  // ied = ie + ng with ng = 3), and del6_vt_flux reaches +/-(1 + nord).
  if (b.halo_width < 3)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport requires halo_width >= 3");
  // edge_profile's tridiagonal closure reads dp0(1) and q(...,npz-2)
  // (nh_utils.F90:1596, 1625), so a single-level column is not representable.
  if (b.npz < 2)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport requires npz >= 2 for edge_profile");
  // del6_vt_flux reaches 1 + nord cells into the halo, and copy_corners only
  // rotates ng == 3 rings, so nord > 2 would read un-rotated corner data no
  // matter how much halo the caller registered. Fortran cannot produce that:
  // dyn_core.F90:713 sets nord_v(k) = min(2, nord). Bound nord_v by the same
  // 2 rather than by halo_width - 1.
  //
  // NOTE: cfg.nord_v is NOT the loop bound for the nord groups -- the
  // authoritative per-level order is the registered damping_ndif array, and
  // launch_del6_vt_flux_cuda is passed ng - 1 so that no damped level can fall
  // outside every group. See the nord_cap contract in wp_b_cuda.cuh.
  if (cfg.nord_v < 0 || cfg.nord_v > 2)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport requires 0 <= nord_v <= 2 "
                "(dyn_core.F90 nord_v(k) = min(2, nord))");

  // npx/npy gate the xppm/yppm panel-edge formulas and copy_corners. A wrong
  // value silently applies interior stencils at a seam, so refuse to guess.
  if (!x.immutable->topology)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "dgrid_height_transport requires a bound TopologyView");
  const auto &topo = *x.immutable->topology;
  if (topo.npx == 0 || topo.npy == 0)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "TopologyView npx/npy must be supplied for panel-edge PPM semantics");
  // The same silent-interior-stencil hazard applies to the cubed_sphere flag,
  // which gates BOTH the xppm/yppm edge formulas and copy_corners. In Fortran
  // the four corner flags are only set when grid_type < 3 .and. .not. nested
  // (fv_grid_utils.F90:288-298), so a caller asserting a panel corner while
  // clearing global_cubed_sphere (or asking for grid_type >= 3, or nested) is
  // self-contradictory. Reject rather than quietly apply interior stencils at
  // a seam the caller just said exists.
  const bool any_corner = topo.sw_corner || topo.se_corner || topo.ne_corner || topo.nw_corner;
  if (any_corner && !(topo.global_cubed_sphere && cfg.grid_type < 3 && !topo.nested))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "TopologyView asserts a panel corner but the configuration is "
                "not a non-nested cubed sphere with grid_type < 3");

  const auto &r = *x.fields;
  out = WpBHeightTransportViews{};
  out.zh = r.find(FieldId::zh);
  out.crx = r.find(FieldId::crx);
  out.cry = r.find(FieldId::cry);
  out.xfx = r.find(FieldId::xfx);
  out.yfx = r.find(FieldId::yfx);
  if (x.immutable->grid) {
    const auto &g = *x.immutable->grid;
    out.area = g.find(FieldId::area);
    out.rarea = g.find(FieldId::rarea);
    out.dxa = g.find(FieldId::dxa);
    out.dya = g.find(FieldId::dya);
    out.del6_u = g.find(FieldId::del6_u);
    out.del6_v = g.find(FieldId::del6_v);
    out.dp_ref = g.find(FieldId::dp_ref);
    out.damping_v = g.find(FieldId::damping_v);
    out.damping_ndif = g.find(FieldId::damping_ndif);
  }

  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;

  if (!covers3(out.zh, lo_i, hi_i, lo_j, hi_j, 0, b.npz))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "zh does not cover required bounds for height transport");
  // crx/xfx are cell-centered in k with npz levels and are read only on
  // [is, ie+1] x [jsd, jed] by edge_profile; cry/yfx symmetrically. Milestone
  // 3A.3 demanded npz+1 levels and the full extended i box, which no
  // authoritative registration satisfies.
  if (!covers3(out.crx, b.is, b.ie + 1, lo_j, hi_j, 0, b.npz - 1) ||
      !covers3(out.xfx, b.is, b.ie + 1, lo_j, hi_j, 0, b.npz - 1))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "crx/xfx do not cover [is,ie+1] x [jsd,jed] x [0,npz-1]");
  if (!covers3(out.cry, lo_i, hi_i, b.js, b.je + 1, 0, b.npz - 1) ||
      !covers3(out.yfx, lo_i, hi_i, b.js, b.je + 1, 0, b.npz - 1))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "cry/yfx do not cover [isd,ied] x [js,je+1] x [0,npz-1]");
  if (!covers2(out.area, lo_i, hi_i, lo_j, hi_j) ||
      !covers2(out.rarea, lo_i, hi_i, lo_j, hi_j))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "area/rarea do not cover required bounds for height transport");
  // The panel-edge reconstruction reads dxa(-1..2, j) / dxa(npx-2..npx+1, j),
  // both of which lie inside [is-3, ie+3] for an edge block.
  if (!covers2(out.dxa, lo_i, hi_i, lo_j, hi_j) ||
      !covers2(out.dya, lo_i, hi_i, lo_j, hi_j))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "dxa/dya do not cover required bounds for panel-edge PPM");
  if (!covers2(out.del6_u, lo_i, hi_i, lo_j, hi_j) ||
      !covers2(out.del6_v, lo_i, hi_i, lo_j, hi_j))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "del6_u/del6_v do not cover required bounds for height transport");
  if (!covers1(out.dp_ref, 0, b.npz - 1))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "dp_ref must cover [0, npz-1] for edge_profile");
  // damp/ndif are indexed at min(k, npz-1), reproducing update_dz_d's
  // damp(km+1) = damp(km) prologue without an npz+1 registration.
  if (!covers1(out.damping_v, 0, b.npz - 1) || !covers1(out.damping_ndif, 0, b.npz - 1))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "damping_v/damping_ndif must cover [0, npz-1]");
  // damping_ndif is LogicalType::integer, and the public ABI has no integer
  // element type, so make_device_view's precision check is a tautology for it
  // (fv3_gpu_contract.cpp compares d.element_bytes against itself). Without
  // this guard an R8 caller could register 4-byte storage and LevelFilter's
  // View1<const fv_real> would reinterpret it as doubles, yielding arbitrary
  // nord values and thus arbitrary del6_vt_flux stencils. Refuse instead.
  if (!view_element_width_is(out.damping_ndif, sizeof(fv_real)))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "damping_ndif element width does not match the active fv_real "
                "precision; integral nord values must be registered as "
                "exactly-representable fv_real");
  // damping_v/dp_ref are LogicalType::fv_real, so make_device_view already
  // pinned their width; assert it here too so a future logical-type change
  // cannot silently reintroduce the same reinterpretation hazard.
  if (!view_element_width_is(out.damping_v, sizeof(fv_real)) ||
      !view_element_width_is(out.dp_ref, sizeof(fv_real)))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "damping_v/dp_ref element width does not match fv_real");

  const auto *slice = workspace_slice(*x.workspace, WorkspaceId::transport);
  if (!slice || !slice->data)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WorkspaceId::transport slice is absent");

  uint64_t ni = static_cast<uint64_t>(hi_i - lo_i + 1);
  uint64_t nj = static_cast<uint64_t>(hi_j - lo_j + 1);
  uint64_t nlev = static_cast<uint64_t>(b.npz + 1);
  uint64_t values = 1;
  if (mul_into(values, ni) || mul_into(values, nj) || mul_into(values, nlev))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "height transport arena size overflowed");
  uint64_t bytes = values;
  if (mul_into(bytes, sizeof(fv_real)))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "height transport arena size overflowed");
  uint64_t total = bytes;
  if (mul_into(total, WpBHeightTransportArena::array_count))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "height transport arena size overflowed");
  if (slice->bytes < total)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WorkspaceId::transport slice is smaller than required");

  char *base = static_cast<char *>(slice->data);
  for (size_t n = 0; n < WpBHeightTransportArena::array_count; ++n)
    arena.arrays[n] = base + n * bytes;

  return FV3_GPU_SUCCESS;
}

// LaunchFn entry point wired into NumericalDispatch::dgrid_height_transport.
//
// Authoritative operation sequence (nh_utils.F90:255-305):
//   edge_profile(crx, xfx, ...)                       -> crx_adv, xfx_adv
//   edge_profile(cry, yfx, ...)                       -> cry_adv, yfx_adv
//   ra_x = area + (xfx_adv(i) - xfx_adv(i+1))         on is..ie x jsd..jed
//   ra_y = area + (yfx_adv(j) - yfx_adv(j+1))         on isd..ied x js..je
//   damped levels   (damp(k) > 1.E-5):
//     z2 = zh ; fv_tp_2d(z2) -> fx, fy
//     del6_vt_flux(ndif(k), damp(k), z2) -> fx2_d6, fy2_d6
//   undamped levels:
//     fv_tp_2d(zh in place) -> fx, fy
//   zh = (source*area + dfx + dfy) / ((ra_x + ra_y) - area) [+ del6 * rarea]
//
// The damped and undamped level sets are disjoint, so their fv_tp_2d passes
// write disjoint k slabs of the shared fx/fy arena and both survive into the
// single zh update launch. No host synchronization and no D2H occurs: the
// per-level branch is evaluated on the device from the resident damping_v /
// damping_ndif arrays via LevelFilter.
int launch_dgrid_height_transport_cuda(const ExecutionContext &x,
                                       fv3_gpu_error *error) noexcept {
  WpBHeightTransportViews v;
  WpBHeightTransportArena a;
  int rc = bind_wp_b_height_transport(x, v, a, error);
  if (rc) return rc;

  const auto &db = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &topo = *x.immutable->topology;

  HeightTransportBounds b{};
  b.is = db.is;
  b.ie = db.ie;
  b.js = db.js;
  b.je = db.je;
  b.npz = db.npz;
  b.halo_width = static_cast<int64_t>(db.halo_width);
  // Fortran ng is a parameter, not the registered halo (tools/fv_mp_mod.F90:108);
  // halo_width >= 3 is validated above so all ng-wide stencils stay in bounds.
  b.ng = 3;
  b.npx = topo.npx;
  b.npy = topo.npy;
  // Fortran guard: .not. nested .and. grid_type < 3 (tp_core.F90:349, 373, 532; corner flags fv_grid_utils.F90:288-298).
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

  void *stream = x.submission.stream;

  View3<const fv_real> crx = to_view3<const fv_real>(v.crx);
  View3<const fv_real> cry = to_view3<const fv_real>(v.cry);
  View3<const fv_real> xfx = to_view3<const fv_real>(v.xfx);
  View3<const fv_real> yfx = to_view3<const fv_real>(v.yfx);
  View2<const fv_real> area = to_view2<const fv_real>(v.area);
  View2<const fv_real> rarea = to_view2<const fv_real>(v.rarea);
  View2<const fv_real> dxa = to_view2<const fv_real>(v.dxa);
  View2<const fv_real> dya = to_view2<const fv_real>(v.dya);
  View2<const fv_real> del6_u = to_view2<const fv_real>(v.del6_u);
  View2<const fv_real> del6_v = to_view2<const fv_real>(v.del6_v);
  View1<const fv_real> dp_ref = to_view1<const fv_real>(v.dp_ref);
  View3<fv_real> zh = to_view3<fv_real>(v.zh);

  View3<fv_real> fx = to_arena_view3(a.arrays[0], db);
  View3<fv_real> fy = to_arena_view3(a.arrays[1], db);
  View3<fv_real> q_i = to_arena_view3(a.arrays[2], db);
  View3<fv_real> q_j = to_arena_view3(a.arrays[3], db);
  View3<fv_real> fx2_tp = to_arena_view3(a.arrays[4], db);
  View3<fv_real> fy2_tp = to_arena_view3(a.arrays[5], db);
  View3<fv_real> ra_x = to_arena_view3(a.arrays[6], db);
  View3<fv_real> ra_y = to_arena_view3(a.arrays[7], db);
  View3<fv_real> z2 = to_arena_view3(a.arrays[8], db);
  View3<fv_real> d2 = to_arena_view3(a.arrays[9], db);
  View3<fv_real> fx2_d6 = to_arena_view3(a.arrays[10], db);
  View3<fv_real> fy2_d6 = to_arena_view3(a.arrays[11], db);
  View3<fv_real> crx_adv = to_arena_view3(a.arrays[12], db);
  View3<fv_real> cry_adv = to_arena_view3(a.arrays[13], db);
  View3<fv_real> xfx_adv = to_arena_view3(a.arrays[14], db);
  View3<fv_real> yfx_adv = to_arena_view3(a.arrays[15], db);
  View1<fv_real> gam = to_arena_view1(a.arrays[16]);

  LevelFilter lf_all{};
  lf_all.damp = to_view1<const fv_real>(v.damping_v);
  lf_all.ndif = to_view1<const fv_real>(v.damping_ndif);
  lf_all.npz = db.npz;
  lf_all.mode = 0;
  LevelFilter lf_damped = lf_all;
  lf_damped.mode = 1;
  LevelFilter lf_undamped = lf_all;
  lf_undamped.mode = 2;

  fv3_gpu_status st = launch_edge_profile_gam_cuda(b, dp_ref, gam, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_edge_profile_gam_cuda failed");

  st = launch_edge_profile_x_cuda(b, crx, xfx, dp_ref, to_const1(gam), crx_adv, xfx_adv,
                                 stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_edge_profile_x_cuda failed");
  st = launch_edge_profile_y_cuda(b, cry, yfx, dp_ref, to_const1(gam), cry_adv, yfx_adv,
                                 stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_edge_profile_y_cuda failed");

  st = launch_ra_x_cuda(b, to_const3(xfx_adv), area, ra_x, stream);
  if (st != FV3_GPU_SUCCESS) return fail(error, st, __func__, "launch_ra_x_cuda failed");
  st = launch_ra_y_cuda(b, to_const3(yfx_adv), area, ra_y, stream);
  if (st != FV3_GPU_SUCCESS) return fail(error, st, __func__, "launch_ra_y_cuda failed");

  // --- damped levels: transport a scratch copy, then add del6 damping ---
  st = launch_copy_z2_cuda(b, lf_damped, to_const3(zh), z2, stream);
  if (st != FV3_GPU_SUCCESS) return fail(error, st, __func__, "launch_copy_z2_cuda failed");

  st = launch_fv_tp_2d_cuda(b, cfg.hord_tm, lf_damped, z2, to_const3(crx_adv),
                            to_const3(cry_adv), to_const3(xfx_adv), to_const3(yfx_adv),
                            area, dxa, dya, to_const3(ra_x), to_const3(ra_y), q_i, q_j, fx,
                            fy, fx2_tp, fy2_tp, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_fv_tp_2d_cuda (damped levels) failed");

  // del6_vt_flux reads z2 AFTER fv_tp_2d's copy_corners calls have updated its
  // corner halos, exactly as in nh_utils.F90:283-289.
  // nord_cap = ng - 1, NOT cfg.nord_v: damping_ndif is authoritative for the
  // per-level order (see wp_b_cuda.cuh's nord_cap contract).
  st = launch_del6_vt_flux_cuda(b, static_cast<int32_t>(b.ng) - 1, lf_damped, z2, del6_u, del6_v, rarea, d2,
                                fx2_d6, fy2_d6, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_del6_vt_flux_cuda failed");

  // --- undamped levels: transport zh in place, no damping ---
  st = launch_fv_tp_2d_cuda(b, cfg.hord_tm, lf_undamped, zh, to_const3(crx_adv),
                            to_const3(cry_adv), to_const3(xfx_adv), to_const3(yfx_adv),
                            area, dxa, dya, to_const3(ra_x), to_const3(ra_y), q_i, q_j, fx,
                            fy, fx2_tp, fy2_tp, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_fv_tp_2d_cuda (undamped levels) failed");

  st = launch_zh_update_cuda(b, lf_all, to_const3(z2), to_const3(fx), to_const3(fy),
                             to_const3(fx2_d6), to_const3(fy2_d6), to_const3(ra_x),
                             to_const3(ra_y), area, rarea, zh, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_zh_update_cuda failed");

  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::dgrid
