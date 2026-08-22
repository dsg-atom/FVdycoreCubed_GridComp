#include "wp_d_contract_cuda.hpp"
#include "wp_d_cuda.hpp"
#include "fv3_gpu_errors.hpp"

namespace fv3::gpu::operators {

int launch_nh_pressure_cuda(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
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

  void *stream = x.submission.stream;
  const auto &b = x.immutable->bounds;
  const auto topology = *x.immutable->topology;
  const auto edges = physical_edges(topology);
  const auto corners = physical_corners(topology);
  const fv_real grav = static_cast<fv_real>(x.immutable->config.grav);
  const fv_real dt = static_cast<fv_real>(x.dispatch_params->dt);
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};

  fv3_gpu_status s = launch_geopotential_from_height_cuda(
      hb, 2, grav, view3_const(views.zh), view3<fv_real>(views.gz), stream);
  if (s != FV3_GPU_SUCCESS)
    return fail(e, s, __func__, "geopotential_from_height CUDA launch failed");

  A2BGeometry geom{view2<const fv_real>(views.dxa), view2<const fv_real>(views.dya),
                   view3<const fv_real>(views.grid_lonlat), view3<const fv_real>(views.agrid_lonlat),
                   view1<const accumulator_real>(views.edge_w), view1<const accumulator_real>(views.edge_e),
                   view1<const accumulator_real>(views.edge_s), view1<const accumulator_real>(views.edge_n)};

  auto *qx = static_cast<fv_real*>(arena.arrays[0]);
  auto *qy = static_cast<fv_real*>(arena.arrays[1]);
  auto *qxx = static_cast<fv_real*>(arena.arrays[2]);
  auto *qyy = static_cast<fv_real*>(arena.arrays[3]);
  auto *pk_b = static_cast<fv_real*>(arena.arrays[5]);
  auto *gz_b = static_cast<fv_real*>(arena.arrays[6]);
  auto *delp_b = static_cast<fv_real*>(arena.arrays[7]);
  const uint64_t ni = static_cast<uint64_t>(b.ie - b.is + 1) + 2 * b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je - b.js + 1) + 2 * b.halo_width;
  const int64_t lo_i = b.is - static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js - static_cast<int64_t>(b.halo_width);
  // arena scratch is owned by this call, not a registered field, so it is
  // reinterpreted here as a batched View3 (all levels at once) to match the
  // CUDA a2b launcher's signature -- the host version instead reuses a
  // per-level View2 inside a k-loop over the same underlying storage.
  auto slice3 = [&](void *p) -> View3<fv_real> {
    return {static_cast<fv_real*>(p), lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};
  };
  View3<fv_real> qx_v = slice3(qx), qy_v = slice3(qy), qxx_v = slice3(qxx), qyy_v = slice3(qyy);
  View3<fv_real> pk_v = slice3(pk_b), gz_v = slice3(gz_b), delp_v = slice3(delp_b);

  s = launch_a2b_ord4_global_cuda(hb, edges, corners, b.npz + 1, view3<fv_real>(views.pkc), pk_v,
                                   geom, qx_v, qy_v, qxx_v, qyy_v, true, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "A2B (pkc) CUDA launch failed");
  s = launch_a2b_ord4_global_cuda(hb, edges, corners, b.npz + 1, view3<fv_real>(views.gz), gz_v,
                                   geom, qx_v, qy_v, qxx_v, qyy_v, true, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "A2B (gz) CUDA launch failed");
  s = launch_a2b_ord4_global_cuda(hb, edges, corners, b.npz, view3<fv_real>(views.delp), delp_v,
                                   geom, qx_v, qy_v, qxx_v, qyy_v, false, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "A2B (delp) CUDA launch failed");

  PressureInputs pin{pk_v, gz_v, delp_v, pk_v, view2<const fv_real>(views.rdx), view2<const fv_real>(views.rdy)};
  PressureOutputs pout{view3<fv_real>(views.u), view3<fv_real>(views.v),
                       view3<fv_real>(views.du), view3<fv_real>(views.dv)};
  PressureConfig pcfg{PressureVariant::normal_nh, dt, 0};
  s = launch_pressure_gradient_cuda(hb, pin, pout, pcfg, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "pressure_gradient CUDA launch failed");
  return FV3_GPU_SUCCESS;
}

int launch_post_loop_heating_cuda(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
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
      x.dispatch_params->diffusion_pass >= x.dispatch_params->diffusion_passes)
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

  void *stream = x.submission.stream;
  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto corners = physical_corners(*x.immutable->topology);
  const int32_t pass = x.dispatch_params->diffusion_pass;
  const int32_t ntimes = x.dispatch_params->diffusion_passes;
  const int32_t nt = ntimes - 1 - pass;
  const int halo_band = static_cast<int>(nt);
  HorizontalBounds hb{b.is, b.ie, b.js, b.je, b.npz};
  const fv_real dt = static_cast<fv_real>(x.dispatch_params->dt);
  const accumulator_real cd = static_cast<accumulator_real>(cfg.d_con) * dt;
  auto *fx = static_cast<fv_real*>(arena.arrays[8]);
  auto *fy = static_cast<fv_real*>(arena.arrays[9]);
  const uint64_t ni = static_cast<uint64_t>(b.ie - b.is + 1) + 2 * b.halo_width;
  const uint64_t nj = static_cast<uint64_t>(b.je - b.js + 1) + 2 * b.halo_width;
  const int64_t lo_i = b.is - static_cast<int64_t>(b.halo_width);
  const int64_t lo_j = b.js - static_cast<int64_t>(b.halo_width);
  View3<fv_real> fx_v{fx, lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};
  View3<fv_real> fy_v{fy, lo_i, lo_j, 0, 1, static_cast<int64_t>(ni), static_cast<int64_t>(ni * nj)};

  fv3_gpu_status s = launch_average_del2_corners_cuda(hb, corners, view3<fv_real>(views.heat_source), stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "average_del2_corners CUDA launch failed");

  if (nt > 0) {
    CornerServiceRequest req_x = diffusion_corner_request(Axis::x, halo_band, pass,
                                                          DiffusionPhase::x_corner_copy, 0);
    rc = x.corner_service(x, req_x, e);
    if (rc) return rc;
  }

  // Both metrics structs are cheap header views built from the same grid
  // fields the host uses; the CUDA launcher itself branches on use_sg to
  // pick which one it reads, matching del2_pass / del2_pass_sg's split.
  Del2Metrics m{view2<const fv_real>(views.del6_u), view2<const fv_real>(views.del6_v),
                view2<const fv_real>(views.rarea)};
  Del2SgMetrics sg{view2<const fv_real>(views.dx), view2<const fv_real>(views.dy),
                   view2<const fv_real>(views.sina_u), view2<const fv_real>(views.sina_v),
                   view2<const fv_real>(views.rdxc), view2<const fv_real>(views.rdyc),
                   view2<const fv_real>(views.rarea)};
  s = launch_del2_zonal_flux_cuda(hb, halo_band, cfg.use_sg != 0, view3<fv_real>(views.heat_source), m, sg, fx_v, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "del2 zonal flux CUDA launch failed");
  s = launch_del2_meridional_flux_cuda(hb, halo_band, cfg.use_sg != 0, view3<fv_real>(views.heat_source), m, sg, fy_v, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "del2 meridional flux CUDA launch failed");
  s = launch_del2_update_cuda(hb, halo_band, cd, view3<fv_real>(views.heat_source), view2<const fv_real>(views.rarea), fx_v, fy_v, stream);
  if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "del2 update CUDA launch failed");

  if (pass == ntimes - 1) {
    const fv_real bdt = static_cast<fv_real>(x.dispatch_params->dt);
    const fv_real delt_max = static_cast<fv_real>(cfg.delt_max);
    const fv_real cv_air = static_cast<fv_real>(cfg.rdgas / cfg.akap);
    const fv_real rdg = static_cast<fv_real>(-cfg.rdgas / cfg.grav);
    const fv_real akap = static_cast<fv_real>(cfg.akap);
    const int64_t n_con = cfg.n_con;
    if (cfg.hydrostatic) {
      const fv_real cp_air = cv_air + static_cast<fv_real>(cfg.rdgas);
      s = launch_diffusive_heating_hydrostatic_cuda(hb, bdt, delt_max, cp_air, n_con,
                                          view3_const(views.heat_source),
                                          view3_const(views.delp),
                                          view3_const(views.pkz),
                                          view3<fv_real>(views.pt), stream);
      if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "diffusive heating (hydrostatic) CUDA launch failed");
    } else {
      s = launch_diffusive_heating_nh_cuda(hb, bdt, delt_max, cv_air, rdg, akap,
                                 cfg.moist_cappa != 0, n_con,
                                 view3<fv_real>(views.heat_source),
                                 view3_const(views.delp),
                                 view3_const(views.delz),
                                 view3_const(views.cappa),
                                 view3<fv_real>(views.pkz),
                                 view3<fv_real>(views.pt), stream);
      if (s != FV3_GPU_SUCCESS) return fail(e, s, __func__, "diffusive heating (nh) CUDA launch failed");
    }
  }
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::operators
