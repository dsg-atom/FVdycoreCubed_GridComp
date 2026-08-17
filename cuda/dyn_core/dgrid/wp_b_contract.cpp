#include "dgrid/wp_b_contract.hpp"
#include "fv3_gpu_errors.hpp"

namespace fv3::gpu::dgrid {
namespace {
const DeviceViewBase *need(const FieldRegistry &r, FieldId id) {
  return r.find(id);
}
}

int resolve_contract_v3(const ExecutionContext &x, ContractViews &out,
                        fv3_gpu_error *error) noexcept {
  int rc = validate_execution_context(x, error);
  if (rc) return rc;
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WP-B requires the borrowed compute submission");

  const auto &r = *x.fields;
  out = ContractViews{};
  out.u = need(r, FieldId::u); out.v = need(r, FieldId::v);
  out.w = need(r, FieldId::w); out.pt = need(r, FieldId::pt);
  out.delp = need(r, FieldId::delp); out.uc = need(r, FieldId::uc);
  out.vc = need(r, FieldId::vc); out.mfx = need(r, FieldId::mfx);
  out.mfy = need(r, FieldId::mfy); out.cx = need(r, FieldId::cx);
  out.cy = need(r, FieldId::cy); out.dpx = need(r, FieldId::dpx);
  out.divgd = need(r, FieldId::divgd); out.crx = need(r, FieldId::crx);
  out.cry = need(r, FieldId::cry); out.xfx = need(r, FieldId::xfx);
  out.yfx = need(r, FieldId::yfx); out.heat_source = need(r, FieldId::heat_source);
  if (x.immutable->grid) {
    const auto &g = *x.immutable->grid;
#define GRID(NAME) out.NAME = need(g, FieldId::NAME)
    GRID(area); GRID(rarea); GRID(dx); GRID(dy); GRID(rdx); GRID(rdy);
    GRID(rdxa); GRID(rdya); GRID(cosa_u); GRID(cosa_v); GRID(cosa_s);
    GRID(sina_u); GRID(sina_v); GRID(rsin_u); GRID(rsin_v); GRID(rsin2);
    GRID(cosa); GRID(rsina); GRID(sin_sg); GRID(f0); GRID(divg_u);
    GRID(divg_v); GRID(del6_u); GRID(del6_v);
    out.d2_bg = need(g, FieldId::damping_d2_bg);
    out.d4_bg = need(g, FieldId::damping_d4_bg);
    out.damp_v = need(g, FieldId::damping_v);
    out.damp_w = need(g, FieldId::damping_w);
    out.damp_t = need(g, FieldId::damping_t);
    out.kgb = need(g, FieldId::damping_kgb);
    out.ndif = need(g, FieldId::damping_ndif);
#undef GRID
  }
  out.transport = workspace_slice(*x.workspace, WorkspaceId::transport);
  out.config = &x.immutable->config;
  out.params = x.dispatch_params;
  out.topology = x.immutable->topology;
  const void *required[] = {out.u, out.v, out.w, out.pt, out.delp, out.uc,
      out.vc, out.mfx, out.mfy, out.cx, out.cy, out.dpx, out.divgd, out.crx,
      out.cry, out.xfx, out.yfx, out.heat_source, out.area, out.rarea, out.dx,
      out.dy, out.rdx, out.rdy, out.rdxa, out.rdya, out.cosa_u, out.cosa_v,
      out.cosa_s, out.sina_u, out.sina_v, out.rsin_u, out.rsin_v, out.rsin2,
      out.cosa, out.rsina, out.sin_sg, out.f0, out.divg_u, out.divg_v,
      out.del6_u, out.del6_v, out.d2_bg, out.d4_bg, out.damp_v, out.damp_w,
      out.damp_t, out.kgb, out.ndif, out.transport, out.config, out.params,
      out.topology};
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
