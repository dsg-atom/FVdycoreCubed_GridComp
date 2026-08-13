#include "wp_d_contract.hpp"

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
} // namespace

int bind_wp_d(const ExecutionContext &x, WpDViews &v, OperatorArena &arena,
              fv3_gpu_error *e) noexcept {
  int rc = validate_execution_context(x, e);
  if (rc) return rc;
  const auto &r = *x.fields;
#define REQUIRED(ID, MEMBER, TYPE) \
  if (!get<TYPE>(r, FieldId::ID, v.MEMBER)) \
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "missing required WP-D field " #ID)
  REQUIRED(u,u,fv_real); REQUIRED(v,v,fv_real); REQUIRED(delp,delp,fv_real);
  REQUIRED(pt,pt,fv_real); REQUIRED(delz,delz,fv_real); REQUIRED(cappa,cappa,fv_real);
  REQUIRED(pkz,pkz,fv_real); REQUIRED(gz,gz,fv_real); REQUIRED(zh,zh,fv_real);
  REQUIRED(pkc,pkc,fv_real); REQUIRED(pk3,pk3,fv_real); REQUIRED(du,du,fv_real);
  REQUIRED(dv,dv,fv_real); REQUIRED(heat_source,heat_source,fv_real);
  REQUIRED(rdx,rdx,const fv_real); REQUIRED(rdy,rdy,const fv_real);
  REQUIRED(rarea,rarea,const fv_real); REQUIRED(del6_u,del6_u,const fv_real);
  REQUIRED(del6_v,del6_v,const fv_real);
#undef REQUIRED
  const WorkspaceSlice *slice = workspace_slice(*x.workspace, WorkspaceId::operators);
  if (!slice || !slice->data)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"operator workspace is unavailable");
  const auto &b=x.immutable->bounds;
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
  WpDViews views; OperatorArena arena;
  int rc=bind_wp_d(x,views,arena,e); if(rc)return rc;
  fv3_gpu_event zh=0,pkc=0;
  if(x.dependencies->require(FV3_GPU_H_ZH,&zh) ||
     x.dependencies->require(FV3_GPU_H_PKC2,&pkc))
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"H_ZH and H_PKC2 must be ready");
  if(x.immutable->config.hydrostatic || x.immutable->config.nested)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"WP-D baseline requires nonhydrostatic global configuration");
  if(x.immutable->config.beta_nonzero || x.immutable->config.use_logp)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"v2 lacks scalar configuration for the requested pressure branch");
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"CUDA WP-D pressure launcher is not compiled in this package build");
}

int launch_post_loop_heating(const ExecutionContext &x, fv3_gpu_error *e) noexcept {
  WpDViews views; OperatorArena arena;
  int rc=bind_wp_d(x,views,arena,e); if(rc)return rc;
  fv3_gpu_event heat=0;
  if(x.dependencies->require(FV3_GPU_H_HEAT,&heat))
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"H_HEAT must be ready");
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
              "v2 lacks n_con, nf_ke, d_con, and delt_max required by post-loop heating");
}

} // namespace fv3::gpu::operators
