#include "wp_d_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace fv3::gpu::operators {
namespace {
int fail(fv3_gpu_error *e, fv3_gpu_status code, const char *api,
         const char *message) noexcept {
  if (e) {
    e->code = code;
    std::snprintf(e->api, sizeof(e->api), "%s", api);
    std::snprintf(e->message, sizeof(e->message), "%s", message);
  }
  return code;
}

template<class T>
bool get(const FieldRegistry &r, FieldId id, DeviceView<T> &out) noexcept {
  const DeviceViewBase *base = r.find(id);
  if (!base) return false;
  static_cast<DeviceViewBase &>(out) = *base;
  return true;
}

bool multiply(uint64_t a, uint64_t b, uint64_t &out) noexcept {
  if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
  out = a * b;
  return true;
}

bool covers(const DeviceViewBase &v, uint32_t axis, int64_t lower,
            int64_t upper) noexcept {
  if (axis >= v.rank || !v.extent[axis]) return false;
  const int64_t view_upper =
      v.lower[axis] + static_cast<int64_t>(v.extent[axis]) - 1;
  return v.lower[axis] <= lower && view_upper >= upper;
}

bool covers_2d(const DeviceViewBase &v, int64_t i0, int64_t i1,
               int64_t j0, int64_t j1) noexcept {
  return covers(v, 0, i0, i1) && covers(v, 1, j0, j1);
}

bool covers_3d(const DeviceViewBase &v, int64_t i0, int64_t i1,
               int64_t j0, int64_t j1, uint64_t levels) noexcept {
  return covers_2d(v, i0, i1, j0, j1) && v.rank == 3 &&
         v.extent[2] >= levels;
}
} // namespace

int bind_wp_d(const ExecutionContext &x, WpDViews &v, OperatorArena &arena,
              fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  const auto &state = *x.fields;
  const auto &grid = *x.immutable->grid;
#define REQUIRED_STATE(ID, MEMBER, TYPE) \
  if (!get<TYPE>(state, FieldId::ID, v.MEMBER)) \
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "missing required WP-D field " #ID)
#define REQUIRED_GRID(ID, MEMBER, TYPE) \
  if (!get<TYPE>(grid, FieldId::ID, v.MEMBER)) \
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "missing required WP-D grid field " #ID)
  REQUIRED_STATE(u,u,fv_real); REQUIRED_STATE(v,v,fv_real);
  REQUIRED_STATE(delp,delp,fv_real); REQUIRED_STATE(pt,pt,fv_real);
  REQUIRED_STATE(delz,delz,fv_real); REQUIRED_STATE(cappa,cappa,fv_real);
  REQUIRED_STATE(pe,pe,fv_real); REQUIRED_STATE(peln,peln,fv_real);
  REQUIRED_STATE(pkz,pkz,fv_real); REQUIRED_STATE(gz,gz,fv_real);
  REQUIRED_STATE(zh,zh,fv_real); REQUIRED_STATE(pkc,pkc,fv_real);
  REQUIRED_STATE(pk3,pk3,fv_real); REQUIRED_STATE(du,du,fv_real);
  REQUIRED_STATE(dv,dv,fv_real); REQUIRED_STATE(heat_source,heat_source,fv_real);
  REQUIRED_STATE(divg2,divg2,fv_real);
  REQUIRED_GRID(rdx,rdx,const fv_real); REQUIRED_GRID(rdy,rdy,const fv_real);
  REQUIRED_GRID(rarea,rarea,const fv_real);
  REQUIRED_GRID(del6_u,del6_u,const fv_real); REQUIRED_GRID(del6_v,del6_v,const fv_real);
  REQUIRED_GRID(dx,dx,const fv_real); REQUIRED_GRID(dy,dy,const fv_real);
  REQUIRED_GRID(rdxc,rdxc,const fv_real); REQUIRED_GRID(rdyc,rdyc,const fv_real);
  REQUIRED_GRID(sina_u,sina_u,const fv_real); REQUIRED_GRID(sina_v,sina_v,const fv_real);
  REQUIRED_GRID(dxa,dxa,const fv_real); REQUIRED_GRID(dya,dya,const fv_real);
  REQUIRED_GRID(grid_lonlat,grid_lonlat,const fv_real);
  REQUIRED_GRID(agrid_lonlat,agrid_lonlat,const fv_real);
  REQUIRED_GRID(edge_w,edge_w,const accumulator_real);
  REQUIRED_GRID(edge_e,edge_e,const accumulator_real);
  REQUIRED_GRID(edge_s,edge_s,const accumulator_real);
  REQUIRED_GRID(edge_n,edge_n,const accumulator_real);
#undef REQUIRED_STATE
#undef REQUIRED_GRID
  const auto &b=x.immutable->bounds;
  const int64_t p=2;
  const int64_t passes =
      (x.immutable->config.n_con > 0 && x.immutable->config.d_con > 1e-5 &&
       x.immutable->config.nf_ke > 0)
          ? std::min<int64_t>(3, x.immutable->config.nf_ke)
          : 0;
  if (b.halo_width <
      static_cast<uint32_t>(std::max<int64_t>(p, passes)))
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "configured halo width is too small for WP-D");
  const bool state_shape =
      covers_3d(v.u, b.is, b.ie, b.js, b.je + 1, b.npz) &&
      covers_3d(v.v, b.is, b.ie + 1, b.js, b.je, b.npz) &&
      covers_3d(v.delp, b.is - p, b.ie + p, b.js - p, b.je + p,
                b.npz) &&
      covers_3d(v.pt, b.is, b.ie, b.js, b.je, b.npz) &&
      covers_3d(v.delz, b.is, b.ie, b.js, b.je, b.npz) &&
      covers_3d(v.cappa, b.is, b.ie, b.js, b.je, b.npz) &&
      covers(v.pe, 0, b.is - 1, b.ie + 1) &&
      covers(v.pe, 2, b.js - 1, b.je + 1) && v.pe.rank == 3 &&
      v.pe.extent[1] >= static_cast<uint64_t>(b.npz) + 1 &&
      /* peln is required over its AUTHORITATIVE declared box only --
       * fv_arrays.F90:1474 allocates peln(is:ie, npz+1, js:je), with no halo
       * ring at all, and dyn_core.F90 declares the dummy the same way. The
       * previous is-p..ie+p / js-p..je+p requirement was inherited from the
       * benchmark harness, which allocates peln over the halo-extended box for
       * its own convenience; it is not a WP-D need. No WP-D operator reads
       * views.peln (it appears in WpDViews and in this bind only -- grep for
       * peln under cuda/dyn_core/operators returns nothing else), so narrowing
       * this cannot unmask an out-of-bounds access. peln's only writer anywhere
       * in the port is Riem_Solver3's last_call writeback, which writes
       * is..ie / js..je (fv3_vertical_kernels.cu riem_solver3_kernel bounds).
       * Prompt 17 standalone integration: real Atm%peln has the Fortran box.
       * Numerical semantics changed: NO. */
      covers(v.peln, 0, b.is, b.ie) &&
      covers(v.peln, 2, b.js, b.je) && v.peln.rank == 3 &&
      v.peln.extent[1] >= static_cast<uint64_t>(b.npz) + 1 &&
      covers_3d(v.pkz, b.is, b.ie, b.js, b.je, b.npz) &&
      covers_3d(v.gz, b.is - p, b.ie + p, b.js - p, b.je + p,
                static_cast<uint64_t>(b.npz) + 1) &&
      covers_3d(v.zh, b.is - p, b.ie + p, b.js - p, b.je + p,
                static_cast<uint64_t>(b.npz) + 1) &&
      covers_3d(v.pkc, b.is - p, b.ie + p, b.js - p, b.je + p,
                static_cast<uint64_t>(b.npz) + 1) &&
      covers_3d(v.pk3, b.is - p, b.ie + p, b.js - p, b.je + p,
                static_cast<uint64_t>(b.npz) + 1) &&
      covers_3d(v.du, b.is, b.ie, b.js, b.je + 1, b.npz) &&
      covers_3d(v.dv, b.is, b.ie + 1, b.js, b.je, b.npz) &&
      covers_3d(v.heat_source, b.is - passes, b.ie + passes,
                b.js - passes, b.je + passes, b.npz) &&
      covers_2d(v.divg2, b.is, b.ie + 1, b.js, b.je + 1);
  const int64_t nt = passes > 0 ? passes - 1 : 0;
  const bool grid_shape =
      covers_2d(v.rdx, b.is, b.ie, b.js, b.je + 1) &&
      covers_2d(v.rdy, b.is, b.ie + 1, b.js, b.je) &&
      covers_2d(v.rarea, b.is - nt, b.ie + nt, b.js - nt, b.je + nt) &&
      covers_2d(v.del6_v, b.is - nt, b.ie + 1 + nt, b.js - nt,
                b.je + nt) &&
      covers_2d(v.del6_u, b.is - nt, b.ie + nt, b.js - nt,
                b.je + 1 + nt) &&
      covers_2d(v.dx, b.is - nt, b.ie + nt, b.js - nt, b.je + 1 + nt) &&
      covers_2d(v.dy, b.is - nt, b.ie + 1 + nt, b.js - nt, b.je + nt) &&
      covers_2d(v.rdxc, b.is - nt, b.ie + 1 + nt, b.js - nt,
                b.je + nt) &&
      covers_2d(v.rdyc, b.is - nt, b.ie + nt, b.js - nt,
                b.je + 1 + nt) &&
      covers_2d(v.sina_u, b.is - nt, b.ie + 1 + nt, b.js - nt,
                b.je + nt) &&
      covers_2d(v.sina_v, b.is - nt, b.ie + nt, b.js - nt,
                b.je + 1 + nt) &&
      covers_2d(v.dxa, b.is - p, b.ie + p, b.js - p, b.je + p) &&
      covers_2d(v.dya, b.is - p, b.ie + p, b.js - p, b.je + p) &&
      covers(v.grid_lonlat, 0, b.is - p, b.ie + p) &&
      covers(v.grid_lonlat, 1, b.js - p, b.je + p) &&
      v.grid_lonlat.rank == 3 && v.grid_lonlat.extent[2] >= 2 &&
      covers(v.agrid_lonlat, 0, b.is - p, b.ie + p) &&
      covers(v.agrid_lonlat, 1, b.js - p, b.je + p) &&
      v.agrid_lonlat.rank == 3 && v.agrid_lonlat.extent[2] >= 2;
  if (!state_shape || !grid_shape)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WP-D descriptor does not cover required bounds");
  const WorkspaceSlice *slice = workspace_slice(*x.workspace, WorkspaceId::operators);
  if (!slice || !slice->data)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"operator workspace is unavailable");
  uint64_t ni=uint64_t(b.ie-b.is+1)+2*b.halo_width;
  uint64_t nj=uint64_t(b.je-b.js+1)+2*b.halo_width;
  uint64_t values=0, bytes=0;
  if (!multiply(ni,nj,values) || !multiply(values,uint64_t(b.npz)+1,values) ||
      !multiply(values,sizeof(fv_real),bytes) || bytes>slice->bytes/arena.array_count)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"operator workspace is too small");
  arena.values_per_array=values;
  auto *base=static_cast<unsigned char *>(slice->data);
  for(size_t n=0;n<arena.array_count;++n) arena.arrays[n]=base+n*bytes;
  return FV3_GPU_SUCCESS;
}

int launch_nh_pressure(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  if (!x.corner_service)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "corner_service is required for WP-D pressure operations");
  if (x.submission.stream_id != StreamId::compute)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WP-D requires the borrowed compute stream");
  if (x.immutable->config.hydrostatic ||
     !x.immutable->topology->global_cubed_sphere ||
     x.immutable->topology->nested || x.immutable->topology->regional)
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "WP-D baseline requires nonhydrostatic global configuration");
  if (x.immutable->config.beta != 0 || x.immutable->config.use_logp ||
      x.immutable->config.a2b_order != 4)
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "only milestone-one beta=0, use_logp=false, A2B4 pressure is enabled");
  fv3_gpu_event zh=0,pkc=0;
  if(x.dependencies->require(FV3_GPU_H_ZH,&zh) ||
     x.dependencies->require(FV3_GPU_H_PKC2,&pkc))
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"H_ZH and H_PKC2 must be ready");
  WpDViews views; OperatorArena arena;
  rc = bind_wp_d(x, views, arena, e);
  if (rc) return rc;
  const auto &b = x.immutable->bounds;
  const auto topology = *x.immutable->topology;
  const auto edges = physical_edges(topology);
  const auto corners = physical_corners(topology);
  const auto &cfg = x.immutable->config;
  const fv_real grav = static_cast<fv_real>(cfg.grav);
  const fv_real dt = static_cast<fv_real>(x.dispatch_params->dt);
  const fv_real ptop = static_cast<fv_real>(cfg.ptop);
  /* pk3_halo (dyn_core.F90:1017 -> :1401) and ptk are EXNER consumers of the
   * scalar akap, not closure consumers, so they take akap_exner when the caller
   * supplies it. See NumericalConfig::akap_exner. Prompt 6B. */
  const double akap_exner = cfg.akap_exner > 0.0 ? cfg.akap_exner : cfg.akap;
  const fv_real akap = static_cast<fv_real>(akap_exner);
  // top_value = ptk = ptop**akap, dyn_core.F90:296 / :1623.
  const fv_real ptk = static_cast<fv_real>(std::pow(cfg.ptop, akap_exner));
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};
  geopotential_from_height(hb, 2, grav, view3_const(views.zh), view3<fv_real>(views.gz));
  // dyn_core.F90:946 -- every substep, unconditionally. See the CUDA twin in
  // wp_d_contract_cuda.cpp for why this is required rather than optional.
  reconstruct_pk3_halo(hb, ptop, akap, view3_const(views.delp),
                       view3<fv_real>(views.pk3));
  A2BGeometry geom{view2<const fv_real>(views.dxa), view2<const fv_real>(views.dya),
                   view3<const fv_real>(views.grid_lonlat), view3<const fv_real>(views.agrid_lonlat),
                   view1<const accumulator_real>(views.edge_w), view1<const accumulator_real>(views.edge_e),
                   view1<const accumulator_real>(views.edge_s), view1<const accumulator_real>(views.edge_n)};
  auto *qx = static_cast<fv_real*>(arena.arrays[0]);
  auto *qy = static_cast<fv_real*>(arena.arrays[1]);
  auto *qxx = static_cast<fv_real*>(arena.arrays[2]);
  auto *qyy = static_cast<fv_real*>(arena.arrays[3]);
  auto *pk3_b = static_cast<fv_real*>(arena.arrays[4]);
  auto *pk_b = static_cast<fv_real*>(arena.arrays[5]);
  auto *gz_b = static_cast<fv_real*>(arena.arrays[6]);
  auto *delp_b = static_cast<fv_real*>(arena.arrays[7]);
  const uint64_t ni = static_cast<uint64_t>(b.ie - b.is + 1) + 2 * b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je - b.js + 1) + 2 * b.halo_width;
  const int64_t lo_i = b.is - static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js - static_cast<int64_t>(b.halo_width);
  auto slice2 = [&](void *p) -> View2<fv_real> {
    return {static_cast<fv_real*>(p), lo_i, lo_j, 1, static_cast<int64_t>(ni)};
  };
  auto slice3 = [&](void *p) -> View3<fv_real> {
    return {static_cast<fv_real*>(p), lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};
  };
  View2<fv_real> qx_v = slice2(qx), qy_v = slice2(qy);
  View2<fv_real> qxx_v = slice2(qxx), qyy_v = slice2(qyy);
  View3<fv_real> pk_v = slice3(pk_b);
  View3<fv_real> pk3_v = slice3(pk3_b);
  View3<fv_real> gz_v = slice3(gz_b);
  View3<fv_real> delp_v = slice3(delp_b);
  // pp (=pkc) and pk (=pk3) are interpolated independently, dyn_core.F90:1637-1638.
  for (int64_t k = 0; k <= b.npz; ++k) {
    View2<fv_real> pkc_slice{views.pkc.base() + views.pkc.offset(lo_i, lo_j, k),
                             lo_i, lo_j, views.pkc.stride[0], views.pkc.stride[1]};
    View2<fv_real> pk3_slice{views.pk3.base() + views.pk3.offset(lo_i, lo_j, k),
                             lo_i, lo_j, views.pk3.stride[0], views.pk3.stride[1]};
    View2<fv_real> gz_slice{views.gz.base() + views.gz.offset(lo_i, lo_j, k),
                            lo_i, lo_j, views.gz.stride[0], views.gz.stride[1]};
    View2<fv_real> pk_out{pk_v.data + k * static_cast<int64_t>(ni * nj), lo_i, lo_j, 1, static_cast<int64_t>(ni)};
    View2<fv_real> pk3_out{pk3_v.data + k * static_cast<int64_t>(ni * nj), lo_i, lo_j, 1, static_cast<int64_t>(ni)};
    View2<fv_real> gz_out{gz_v.data + k * static_cast<int64_t>(ni * nj), lo_i, lo_j, 1, static_cast<int64_t>(ni)};
    a2b_ord4_global(hb, edges, corners, pkc_slice, pk_out, geom, qx_v, qy_v, qxx_v, qyy_v, true);
    a2b_ord4_global(hb, edges, corners, pk3_slice, pk3_out, geom, qx_v, qy_v, qxx_v, qyy_v, true);
    a2b_ord4_global(hb, edges, corners, gz_slice, gz_out, geom, qx_v, qy_v, qxx_v, qyy_v, true);
  }
  for (int64_t k = 0; k < b.npz; ++k) {
    View2<fv_real> delp_slice{views.delp.base() + views.delp.offset(lo_i, lo_j, k),
                              lo_i, lo_j, views.delp.stride[0], views.delp.stride[1]};
    View2<fv_real> delp_out{delp_v.data + k * static_cast<int64_t>(ni * nj), lo_i, lo_j, 1, static_cast<int64_t>(ni)};
    a2b_ord4_global(hb, edges, corners, delp_slice, delp_out, geom, qx_v, qy_v, qxx_v, qyy_v, false);
  }
  // dyn_core.F90:1628-1635 -- the k==1 level is not interpolated.
  set_pressure_top(hb, ptk, pk_v, pk3_v);
  PressureInputs pin{pk_v, gz_v, delp_v, pk3_v, view2<const fv_real>(views.rdx), view2<const fv_real>(views.rdy)};
  PressureOutputs pout{view3<fv_real>(views.u), view3<fv_real>(views.v),
                       view3<fv_real>(views.du), view3<fv_real>(views.dv)};
  PressureConfig pcfg{PressureVariant::normal_nh, dt, 0};
  pressure_gradient(hb, pin, pout, pcfg);
  return FV3_GPU_SUCCESS;
}

int launch_post_loop_heating(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  if (x.immutable->config.n_con <= 0 || x.immutable->config.d_con <= 1e-5)
    return FV3_GPU_SUCCESS;
  if (!x.corner_service)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "corner_service is required for WP-D diffusion operations");
  if (x.submission.stream_id != StreamId::compute)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "WP-D requires the borrowed compute stream");
  if (x.immutable->config.nf_ke < 1 || x.immutable->config.nf_ke > 3)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "nf_ke must be in 1..3 when heating is enabled");
  if (x.dispatch_params->diffusion_passes != x.immutable->config.nf_ke ||
      x.dispatch_params->diffusion_pass < 0 ||
      x.dispatch_params->diffusion_pass >=
          x.dispatch_params->diffusion_passes)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "diffusion pass controls do not match nf_ke");
  if (x.dispatch_params->diffusion_pass == 0) {
    fv3_gpu_event heat=0;
    if(x.dependencies->require(FV3_GPU_H_HEAT,&heat))
      return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"H_HEAT must be ready");
  }
  WpDViews views; OperatorArena arena;
  rc = bind_wp_d(x, views, arena, e);
  if (rc) return rc;
  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto corners = physical_corners(*x.immutable->topology);
  const int32_t pass = x.dispatch_params->diffusion_pass;
  const int32_t ntimes = x.dispatch_params->diffusion_passes;
  const int32_t nt = ntimes - 1 - pass;
  const uint32_t halo_band = static_cast<uint32_t>(nt);
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};
  /* Milestone 3A.9 semantics repair -- identical to the CUDA twin in
   * wp_d_contract_cuda.cpp; see the full rationale there.
   *
   * del2_cubed's `cd` is "K * da_min, 0 < K < 0.25" (dyn_core.F90:2253) and the
   * heat_source call site passes cnst_0p20*da_min (dyn_core.F90:1214, with
   * cnst_0p20 = 0.20d0 at :147). It is a timestep-independent hyperdiffusion
   * coefficient with units of area; the previous d_con*dt was dimensionally
   * wrong and belongs to heat_source's FORMATION inside d_sw, not to this filter.
   *
   * Numerical semantics changed: NO -- restores dyn_core.F90:1214/:2253.
   */
  constexpr accumulator_real cnst_0p20 = accumulator_real(0.20);  // dyn_core.F90:147
  const accumulator_real cd = cnst_0p20 * static_cast<accumulator_real>(cfg.da_min);
  auto *fx = static_cast<fv_real*>(arena.arrays[8]);
  auto *fy = static_cast<fv_real*>(arena.arrays[9]);
  const uint64_t ni = static_cast<uint64_t>(b.ie - b.is + 1) + 2 * b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je - b.js + 1) + 2 * b.halo_width;
  const int64_t lo_i = b.is - static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js - static_cast<int64_t>(b.halo_width);
  View3<fv_real> fx_v{fx, lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};
  View3<fv_real> fy_v{fy, lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};
  average_del2_corners(hb, corners, view3<fv_real>(views.heat_source));
  if (nt > 0) {
    CornerServiceRequest req_x = diffusion_corner_request(Axis::x, halo_band, pass,
                                                          DiffusionPhase::x_corner_copy, 0);
    rc = x.corner_service(x, req_x, e);
    if (rc) return rc;
  }
  /* Prompt 6B: the pass is issued phase by phase because del2_cubed rotates the
   * corners TWICE -- dir=1 at dyn_core.F90:2419 (above) and dir=2 at :2437,
   * between the fx and fy loops. Only the dir=1 rotation was issued before; the
   * fy differences at the four physical panel corners therefore consumed the
   * x-rotated halo. Numerical algorithm changed: NO -- restores :2437. */
  const bool use_sg = cfg.use_sg != 0;
  Del2SgMetrics sg{view2<const fv_real>(views.dx), view2<const fv_real>(views.dy),
                   view2<const fv_real>(views.sina_u), view2<const fv_real>(views.sina_v),
                   view2<const fv_real>(views.rdxc), view2<const fv_real>(views.rdyc),
                   view2<const fv_real>(views.rarea)};
  Del2Metrics m{view2<const fv_real>(views.del6_u), view2<const fv_real>(views.del6_v),
                view2<const fv_real>(views.rarea)};
  del2_zonal_flux(hb, static_cast<int>(nt), use_sg,
                  view3<fv_real>(views.heat_source), m, sg, fx_v);
  if (nt > 0) {
    CornerServiceRequest req_y = diffusion_corner_request(Axis::y, halo_band, pass,
                                                          DiffusionPhase::y_corner_copy, 0);
    rc = x.corner_service(x, req_y, e);
    if (rc) return rc;
  }
  del2_meridional_flux(hb, static_cast<int>(nt), use_sg,
                       view3<fv_real>(views.heat_source), m, sg, fy_v);
  del2_update(hb, static_cast<int>(nt), cd, view3<fv_real>(views.heat_source),
              view2<const fv_real>(views.rarea), fx_v, fy_v);
  if (pass == ntimes - 1) {
    /* Milestone 3A.9: the BIG timestep, per dyn_core.F90:1231/:1240
     * (abs(bdt*flagstruct%delt_max)). dispatch_params->dt is now authoritatively
     * the acoustic SUBSTEP dt, so read the explicit bdt member.
     * Numerical semantics changed: NO. */
    const fv_real bdt = static_cast<fv_real>(x.dispatch_params->bdt);
    const fv_real delt_max = static_cast<fv_real>(cfg.delt_max);
    /* Prompt 6B: dyn_core.F90:323 is `cv_air = cp_air - rdgas`, with
     * cp_air = rdgas/akap. This read the specific heat at constant PRESSURE and
     * called it cv_air -- 1004.7 where 717.6 is required, so every dtmp at
     * :1349 was a factor cp_air/cv_air = 1/(1-akap) = 1.4 too small. The
     * hydrostatic branch's `cp_air = cv_air + rdgas` below was written against
     * the correct definition and becomes right again once cv_air is.
     * Numerical algorithm changed: NO -- restores dyn_core.F90:323. */
    const fv_real cv_air =
        static_cast<fv_real>(cfg.rdgas / cfg.akap - cfg.rdgas);
    const fv_real rdg = static_cast<fv_real>(-cfg.rdgas / cfg.grav);
    const fv_real akap = static_cast<fv_real>(cfg.akap);
    const int64_t n_con = cfg.n_con;
    if (cfg.hydrostatic) {
      const fv_real cp_air = cv_air + static_cast<fv_real>(cfg.rdgas);
      apply_diffusive_heating_hydrostatic(hb, bdt, delt_max, cp_air, n_con,
                                          view3_const(views.heat_source),
                                          view3_const(views.delp),
                                          view3_const(views.pkz),
                                          view3<fv_real>(views.pt));
    } else {
      apply_diffusive_heating_nh(hb, bdt, delt_max, cv_air, rdg, akap,
                                 cfg.moist_cappa != 0, n_con,
                                 view3<fv_real>(views.heat_source),
                                 view3_const(views.delp),
                                 view3_const(views.delz),
                                 view3_const(views.cappa),
                                 view3<fv_real>(views.pkz),
                                 view3<fv_real>(views.pt));
    }
  }
  return FV3_GPU_SUCCESS;
}

int bind_omega(const ExecutionContext &x, OmegaViews &v, OperatorArena &arena,
               fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  const auto &state = *x.fields;
  const auto &grid = *x.immutable->grid;
#define REQUIRED_STATE(ID, MEMBER, TYPE) \
  if (!get<TYPE>(state, FieldId::ID, v.MEMBER)) \
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "missing required omega field " #ID)
#define REQUIRED_GRID(ID, MEMBER, TYPE) \
  if (!get<TYPE>(grid, FieldId::ID, v.MEMBER)) \
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "missing required omega grid field " #ID)
  REQUIRED_STATE(omga,omga,fv_real); REQUIRED_STATE(pe,pe,fv_real);
  REQUIRED_STATE(delp,delp,fv_real);
  REQUIRED_STATE(ua,ua,const fv_real); REQUIRED_STATE(va,va,const fv_real);
  REQUIRED_GRID(dx,dx,const fv_real); REQUIRED_GRID(dy,dy,const fv_real);
  REQUIRED_GRID(rarea,rarea,const fv_real);
  REQUIRED_GRID(grid_lonlat,grid_lonlat,const fv_real);
  REQUIRED_GRID(edge_w,edge_w,const accumulator_real);
  REQUIRED_GRID(edge_e,edge_e,const accumulator_real);
  REQUIRED_GRID(edge_s,edge_s,const accumulator_real);
  REQUIRED_GRID(edge_n,edge_n,const accumulator_real);
#undef REQUIRED_STATE
#undef REQUIRED_GRID
  const auto &b = x.immutable->bounds;
  if (b.halo_width < 1)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "omega finalization needs at least one halo ring for pem");
  /* pem is filled over is-1..ie+1 / js-1..je+1 (dyn_core.F90:505-513) and its
   * a2b_ord2 transform is read at is..ie+1 / js..je+1 by adv_pe. */
  const bool shape =
      covers_3d(v.omga, b.is, b.ie, b.js, b.je, b.npz) &&
      covers_3d(v.delp, b.is - 1, b.ie + 1, b.js - 1, b.je + 1, b.npz) &&
      covers_3d(v.ua, b.is, b.ie, b.js, b.je, b.npz) &&
      covers_3d(v.va, b.is, b.ie, b.js, b.je, b.npz) &&
      covers(v.pe, 0, b.is - 1, b.ie + 1) &&
      covers(v.pe, 2, b.js - 1, b.je + 1) && v.pe.rank == 3 &&
      v.pe.extent[1] >= static_cast<uint64_t>(b.npz) + 1 &&
      covers_2d(v.dx, b.is, b.ie, b.js, b.je + 1) &&
      covers_2d(v.dy, b.is, b.ie + 1, b.js, b.je) &&
      covers_2d(v.rarea, b.is, b.ie, b.js, b.je) &&
      covers(v.grid_lonlat, 0, b.is, b.ie + 1) &&
      covers(v.grid_lonlat, 1, b.js, b.je + 1) &&
      v.grid_lonlat.rank == 3 && v.grid_lonlat.extent[2] >= 2;
  if (!shape)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "omega descriptor does not cover required bounds");
  const WorkspaceSlice *slice =
      workspace_slice(*x.workspace, WorkspaceId::operators);
  if (!slice || !slice->data)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"operator workspace is unavailable");
  uint64_t ni=uint64_t(b.ie-b.is+1)+2*b.halo_width;
  uint64_t nj=uint64_t(b.je-b.js+1)+2*b.halo_width;
  uint64_t values=0, bytes=0;
  if (!multiply(ni,nj,values) || !multiply(values,uint64_t(b.npz)+1,values) ||
      !multiply(values,sizeof(fv_real),bytes) || bytes>slice->bytes/arena.array_count)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"operator workspace is too small");
  arena.values_per_array=values;
  auto *base=static_cast<unsigned char *>(slice->data);
  for(size_t n=0;n<arena.array_count;++n) arena.arrays[n]=base+n*bytes;
  return FV3_GPU_SUCCESS;
}

/* Common preconditions for both omega entries.  The implemented branch is the
 * authoritative one for this configuration: use_old_omega = .true.
 * (fv_arrays.F90:465, not overridden by the fixture namelist).  The
 * `.not. use_old_omega` om2d form (dyn_core.F90:797-801, :838-842, :1219-1237)
 * is deliberately not implemented; there is no knob selecting it, so nothing can
 * silently take it. */
int omega_preconditions(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  if (x.submission.stream_id != StreamId::compute)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "omega finalization requires the borrowed compute stream");
  if (!x.immutable->topology->global_cubed_sphere ||
      x.immutable->topology->nested || x.immutable->topology->regional)
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "omega finalization requires a global cubed-sphere configuration");
  if (!x.dispatch_params->end_step)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "omega finalization is only defined on the last_step substep");
  return FV3_GPU_SUCCESS;
}

namespace {
/* Level slice of an arena array laid out (i, j, k) with the standard carve. */
View2<fv_real> arena_level(void *p, int64_t lo_i, int64_t lo_j, uint64_t ni,
                           uint64_t nj, int64_t level) noexcept {
  return {static_cast<fv_real*>(p) + static_cast<int64_t>(ni*nj)*level,
          lo_i, lo_j, 1, static_cast<int64_t>(ni)};
}
} // namespace

int launch_omega_pem_snapshot(const ExecutionContext &x,
                              fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  rc = omega_preconditions(x, e);
  if (rc) return rc;
  OmegaViews v; OperatorArena arena;
  rc = bind_omega(x, v, arena, e);
  if (rc) return rc;
  const auto &b = x.immutable->bounds;
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};
  const uint64_t ni = static_cast<uint64_t>(b.ie-b.is+1)+2*b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je-b.js+1)+2*b.halo_width;
  const int64_t lo_i = b.is-static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js-static_cast<int64_t>(b.halo_width);
  View3<fv_real> pem{static_cast<fv_real*>(arena.arrays[10]), lo_i, lo_j, 0, 1,
                     static_cast<int64_t>(ni), static_cast<int64_t>(ni*nj)};
  omega_pem_snapshot(hb, static_cast<fv_real>(x.immutable->config.ptop),
                     view3_const(v.delp), pem);
  return FV3_GPU_SUCCESS;
}

int launch_omega_finalize(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  rc = omega_preconditions(x, e);
  if (rc) return rc;
  if (x.dispatch_params->rdt == 0)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "rdt is required by the omega finalization");
  OmegaViews v; OperatorArena arena;
  rc = bind_omega(x, v, arena, e);
  if (rc) return rc;
  const auto &b = x.immutable->bounds;
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};
  const uint64_t ni = static_cast<uint64_t>(b.ie-b.is+1)+2*b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je-b.js+1)+2*b.halo_width;
  const int64_t lo_i = b.is-static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js-static_cast<int64_t>(b.halo_width);
  View3<fv_real> pem{static_cast<fv_real*>(arena.arrays[10]), lo_i, lo_j, 0, 1,
                     static_cast<int64_t>(ni), static_cast<int64_t>(ni*nj)};
  View3<fv_real> pb{static_cast<fv_real*>(arena.arrays[11]), lo_i, lo_j, 0, 1,
                    static_cast<int64_t>(ni), static_cast<int64_t>(ni*nj)};
  // dyn_core.F90:1208-1212
  omega_from_pe(hb, static_cast<fv_real>(x.dispatch_params->rdt),
                view3_const(v.pe), pem, view3<fv_real>(v.omga));
  // dyn_core.F90:1218 -> :1602 call a2b_ord2(pin, pb, ...) per level
  A2BGeometry geom{{}, {}, view3<const fv_real>(v.grid_lonlat), {},
                   view1<const accumulator_real>(v.edge_w),
                   view1<const accumulator_real>(v.edge_e),
                   view1<const accumulator_real>(v.edge_s),
                   view1<const accumulator_real>(v.edge_n)};
  const auto edges = physical_edges(*x.immutable->topology);
  const auto corners = physical_corners(*x.immutable->topology);
  for (int64_t level = 0; level < b.npz; ++level)
    a2b_ord2_global(hb, edges, corners,
                    arena_level(arena.arrays[10], lo_i, lo_j, ni, nj, level),
                    arena_level(arena.arrays[11], lo_i, lo_j, ni, nj, level),
                    geom, false);
  // dyn_core.F90:1583-1636
  AdvPeMetrics m{view3<const fv_real>(v.grid_lonlat), view2<const fv_real>(v.dx),
                 view2<const fv_real>(v.dy), view2<const fv_real>(v.rarea)};
  adv_pe(hb, view3<const fv_real>(v.ua), view3<const fv_real>(v.va), pb, m,
         view3<fv_real>(v.omga));
  return FV3_GPU_SUCCESS;
}

CornerServiceRequest diffusion_corner_request(Axis axis, uint32_t width,
                                               int32_t pass, DiffusionPhase phase,
                                               fv3_gpu_event dependency) noexcept {
  CornerServiceRequest r{};
  if((axis!=Axis::x && axis!=Axis::y) || !width || pass<0 ||
     phase==DiffusionPhase::none) {
    r.first=FieldId::invalid;
    return r;
  }
  r.operation=CornerOperation::copy;r.first=FieldId::heat_source;
  r.stagger=FV3_GPU_STAGGER_A;r.axis=axis;
  r.orientation=ComponentOrientation::scalar;
  r.diffusion_phase=phase;r.halo_width=width;
  r.pass_index=pass;r.replace=1;r.dependency=dependency;
  return r;
}

CornerServiceRequest a2b_edge_request(FieldId source, FieldId destination,
                                      int32_t order, bool replace,
                                      fv3_gpu_event dependency) noexcept {
  CornerServiceRequest r{};
  if(order!=2 && order!=4) {
    r.first=FieldId::invalid;
    return r;
  }
  r.operation=CornerOperation::a_to_b_edge;r.first=source;
  r.destination=destination;
  r.stagger=FV3_GPU_STAGGER_A;r.axis=Axis::both;
  r.orientation=ComponentOrientation::scalar;
  r.halo_width=order==2?1u:2u;
  r.interpolation_order=order;
  r.replace=replace?1:0;r.dependency=dependency;
  return r;
}

PhysicalEdges physical_edges(const TopologyView &topology) noexcept {
  auto crosses_tile=[&](TileEdge edge) {
    const auto &neighbor=topology.neighbors[static_cast<size_t>(edge)];
    return neighbor.rank>=0 && neighbor.tile>=0 && neighbor.tile!=topology.tile;
  };
  return {crosses_tile(TileEdge::west),crosses_tile(TileEdge::east),
          crosses_tile(TileEdge::south),crosses_tile(TileEdge::north)};
}

PhysicalCorners physical_corners(const TopologyView &topology) noexcept {
  return {topology.sw_corner!=0,topology.se_corner!=0,
          topology.ne_corner!=0,topology.nw_corner!=0};
}

} // namespace fv3::gpu::operators
