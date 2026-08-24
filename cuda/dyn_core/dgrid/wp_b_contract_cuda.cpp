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
  if (mord != 1 && mord != 6 && mord != 8)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport supports hord_tm in {1,6,8} only");
  // fv_tp_2d_hord6's own stencil reaches +/-3 relative to is/ie/js/je
  // (isd = is - 3, ied = ie + 3, and symmetrically in j); see
  // wp_b_kernels.hpp:976-977. The registered halo must cover that.
  if (b.halo_width < 3)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "dgrid_height_transport requires halo_width >= 3");

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
    out.del6_u = g.find(FieldId::del6_u);
    out.del6_v = g.find(FieldId::del6_v);
  }

  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t lo_i = b.is - h, hi_i = b.ie + h;
  const int64_t lo_j = b.js - h, hi_j = b.je + h;

  if (!covers3(out.zh, lo_i, hi_i, lo_j, hi_j, 0, b.npz))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "zh does not cover required bounds for height transport");
  if (!covers3(out.crx, lo_i, hi_i + 1, lo_j, hi_j, 0, b.npz) ||
      !covers3(out.cry, lo_i, hi_i, lo_j, hi_j + 1, 0, b.npz) ||
      !covers3(out.xfx, lo_i, hi_i + 1, lo_j, hi_j, 0, b.npz) ||
      !covers3(out.yfx, lo_i, hi_i, lo_j, hi_j + 1, 0, b.npz))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "crx/cry/xfx/yfx do not cover required bounds for height transport");
  if (!covers2(out.area, lo_i, hi_i, lo_j, hi_j) ||
      !covers2(out.rarea, lo_i, hi_i, lo_j, hi_j))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "area/rarea do not cover required bounds for height transport");
  if (!covers2(out.del6_u, lo_i, hi_i, lo_j, hi_j) ||
      !covers2(out.del6_v, lo_i, hi_i, lo_j, hi_j))
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "del6_u/del6_v do not cover required bounds for height transport");

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
// Executes fv_tp_2d_hord6 (order 6, per the production first-run config's
// hord_tm = 6) on zh for all npz+1 interface levels, then applies the
// resulting flux divergence to zh in place. Does NOT invoke
// deln_flux_transport -- see wp_b_cuda.cuh's launch_deln_flux_transport_cuda
// doc comment and milestone_3a3_wp_b_cuda.md for the damp_c mapping
// ambiguity this defers.
int launch_dgrid_height_transport_cuda(const ExecutionContext &x,
                                       fv3_gpu_error *error) noexcept {
  WpBHeightTransportViews v;
  WpBHeightTransportArena a;
  int rc = bind_wp_b_height_transport(x, v, a, error);
  if (rc) return rc;

  const auto &db = x.immutable->bounds;
  HeightTransportBounds b{db.is, db.ie, db.js, db.je, db.npz,
                          static_cast<int64_t>(db.halo_width)};
  void *stream = x.submission.stream;

  View3<const fv_real> zh = to_view3<const fv_real>(v.zh);
  View3<const fv_real> crx = to_view3<const fv_real>(v.crx);
  View3<const fv_real> cry = to_view3<const fv_real>(v.cry);
  View3<const fv_real> xfx = to_view3<const fv_real>(v.xfx);
  View3<const fv_real> yfx = to_view3<const fv_real>(v.yfx);
  View2<const fv_real> area = to_view2<const fv_real>(v.area);
  View2<const fv_real> rarea = to_view2<const fv_real>(v.rarea);

  View3<fv_real> fx = to_arena_view3(a.arrays[0], db);
  View3<fv_real> fy = to_arena_view3(a.arrays[1], db);
  View3<fv_real> q_work = to_arena_view3(a.arrays[2], db);
  View3<fv_real> fx2 = to_arena_view3(a.arrays[3], db);
  View3<fv_real> fy2 = to_arena_view3(a.arrays[4], db);
  View3<fv_real> ra_x = to_arena_view3(a.arrays[6], db);
  View3<fv_real> ra_y = to_arena_view3(a.arrays[7], db);

  fv3_gpu_status st = launch_ra_xy_cuda(b, xfx, yfx, area, ra_x, ra_y, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_ra_xy_cuda failed");

  const int32_t hord = x.immutable->config.hord_tm;
  st = launch_fv_tp_2d_hord6_cuda(b, hord, zh, crx, cry, xfx, yfx, area, to_const3(ra_x),
                                  to_const3(ra_y), fx, fy, q_work, fx2, fy2, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_fv_tp_2d_hord6_cuda failed");

  // zh is bound read-only above (fv_tp_2d_hord6 only reads q_in); mutate
  // through a fresh non-const view for the in-place divergence update.
  View3<fv_real> zh_mut = to_view3<fv_real>(v.zh);
  st = launch_apply_height_flux_divergence_cuda(b, to_const3(fx), to_const3(fy), rarea,
                                                zh_mut, stream);
  if (st != FV3_GPU_SUCCESS)
    return fail(error, st, __func__, "launch_apply_height_flux_divergence_cuda failed");

  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::dgrid
