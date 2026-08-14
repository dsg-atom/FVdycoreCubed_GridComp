#include "wp_d_contract.hpp"

#include <algorithm>
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
      covers(v.peln, 0, b.is - p, b.ie + p) &&
      covers(v.peln, 2, b.js - p, b.je + p) && v.peln.rank == 3 &&
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
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
              "Gate A.2 cannot bind authoritative global A-to-B edge geometry");
}

int launch_post_loop_heating(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  if (x.immutable->config.n_con <= 0 || x.immutable->config.d_con <= 1e-5)
    return FV3_GPU_SUCCESS;
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
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
              "Gate A.2 cannot express intra-pass x/y corner-service phases");
}

CornerServiceRequest diffusion_corner_request(Axis axis, uint32_t width,
                                               int32_t pass,
                                               fv3_gpu_event dependency) noexcept {
  CornerServiceRequest r{};
  if((axis!=Axis::x && axis!=Axis::y) || !width || pass<0) {
    r.first=FieldId::invalid;
    return r;
  }
  r.operation=CornerOperation::copy;r.first=FieldId::heat_source;
  r.stagger=FV3_GPU_STAGGER_A;r.axis=axis;
  r.orientation=ComponentOrientation::scalar;r.halo_width=width;
  r.pass_index=pass;r.dependency=dependency;
  return r;
}

CornerServiceRequest a2b_edge_request(FieldId field, int32_t order,
                                      fv3_gpu_event dependency) noexcept {
  CornerServiceRequest r{};
  if(order!=2 && order!=4) {
    r.first=FieldId::invalid;
    return r;
  }
  r.operation=CornerOperation::a_to_b_edge;r.first=field;
  r.stagger=FV3_GPU_STAGGER_A;r.axis=Axis::both;
  r.orientation=ComponentOrientation::scalar;
  r.halo_width=order==2?1u:2u;
  r.interpolation_order=order;r.dependency=dependency;
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
