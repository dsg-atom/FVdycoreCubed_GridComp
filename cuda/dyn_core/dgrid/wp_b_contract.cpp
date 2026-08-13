#include "dgrid/wp_b_contract.hpp"
#include "fv3_gpu_errors.hpp"

namespace fv3::gpu::dgrid {
namespace {
const DeviceViewBase *need(const FieldRegistry &r, FieldId id) {
  return r.find(id);
}
}

int resolve_contract_v2(const ExecutionContext &x, ContractViews &out,
                        fv3_gpu_error *error) noexcept {
  int rc = validate_execution_context(x, error);
  if (rc) return rc;
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WP-B requires the borrowed compute submission");

  const auto &r = *x.fields;
  out = {need(r, FieldId::u), need(r, FieldId::v), need(r, FieldId::w),
         need(r, FieldId::pt), need(r, FieldId::delp), need(r, FieldId::uc),
         need(r, FieldId::vc), need(r, FieldId::mfx), need(r, FieldId::mfy),
         need(r, FieldId::cx), need(r, FieldId::cy), need(r, FieldId::dpx),
         need(r, FieldId::divgd), need(r, FieldId::crx), need(r, FieldId::cry),
         need(r, FieldId::xfx), need(r, FieldId::yfx),
         need(r, FieldId::heat_source), nullptr, nullptr, nullptr, nullptr,
         workspace_slice(*x.workspace, WorkspaceId::transport), 0, 0};
  if (x.immutable->grid) {
    out.area = need(*x.immutable->grid, FieldId::area);
    out.rarea = need(*x.immutable->grid, FieldId::rarea);
    out.dx = need(*x.immutable->grid, FieldId::dx);
    out.dy = need(*x.immutable->grid, FieldId::dy);
  }
  const void *required[] = {out.u, out.v, out.w, out.pt, out.delp, out.uc,
      out.vc, out.mfx, out.mfy, out.cx, out.cy, out.dpx, out.divgd, out.crx,
      out.cry, out.xfx, out.yfx, out.heat_source, out.area, out.rarea, out.dx,
      out.dy, out.transport};
  for (const void *p : required)
    if (!p) return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                        "WP-B required field or transport workspace is absent");
  rc = x.dependencies->require(FV3_GPU_H_CWIND, &out.cwind_event);
  if (rc) return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                      "H_CWIND is not ready");
  rc = x.dependencies->require(FV3_GPU_H_DIVGD, &out.divgd_event);
  if (rc) return fail(error, static_cast<fv3_gpu_status>(rc), __func__,
                      "H_DIVGD is not ready");
  return FV3_GPU_SUCCESS;
}
} // namespace fv3::gpu::dgrid
