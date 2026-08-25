#include "fv3_gpu_api.h"
#include "fv3_gpu_context.hpp"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_types.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <string>

#ifdef FV3_GPU_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

struct fv3_gpu_context { fv3::gpu::ContextData data; };
namespace {
using namespace fv3::gpu;
bool header_ok(uint32_t version, uint32_t bytes, size_t need) {
  return version == FV3_GPU_ABI_VERSION && bytes >= need;
}
int validate_mask(const fv3_gpu_field_mask *m, fv3_gpu_error *e, const char *api) {
  if (!m || !header_ok(m->abi_version, m->struct_bytes, sizeof(*m)))
    return fail(e, FV3_GPU_INVALID_ARGUMENT, api, "invalid field-mask ABI");
  return FV3_GPU_SUCCESS;
}
[[maybe_unused]] uint64_t issue_event(ContextData &d) { d.last_event = d.next_event++; return d.last_event; }
[[maybe_unused]] bool mul_overflow(uint64_t a,uint64_t b,uint64_t &out){if(a&&b>std::numeric_limits<uint64_t>::max()/a)return true;out=a*b;return false;}
#ifdef FV3_GPU_ENABLE_CUDA
int cuda_fail(cudaError_t rc, fv3_gpu_error *e, const char *api, ContextData *d = nullptr) {
  if (rc == cudaSuccess) return FV3_GPU_SUCCESS;
  if (d) d->phase = ContextPhase::failed;
  return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(rc), (int)rc,
              d ? d->create.mpi_rank : -1, d ? d->create.device_ordinal : -1);
}
#endif
}

namespace fv3::gpu {
void clear_error(fv3_gpu_error *e) noexcept {
  if (!e) return;
  std::memset(e, 0, sizeof(*e)); e->abi_version = FV3_GPU_ABI_VERSION; e->struct_bytes = sizeof(*e);
}
int fail(fv3_gpu_error *e, fv3_gpu_status code, const char *api, const std::string &msg,
         int native, int rank, int device, uint64_t seq) noexcept {
  if (e) { clear_error(e); e->code=code; e->native_code=native; e->rank=rank; e->device=device; e->sequence=seq;
    std::strncpy(e->api, api ? api : "", sizeof(e->api)-1); std::strncpy(e->message, msg.c_str(), sizeof(e->message)-1); }
  return code;
}

int set_numerical_dispatch(fv3_gpu_context *c, const NumericalDispatch &d, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if (d.contract_version!=internal_contract_version) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"wrong dispatch version");
  c->data.dispatch=d;
  return FV3_GPU_SUCCESS;
}
int set_numerical_config(fv3_gpu_context *c, const NumericalConfig &cfg, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  c->data.immutable.config=cfg;
  return FV3_GPU_SUCCESS;
}
int set_topology(fv3_gpu_context *c, const TopologyView &t, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if (t.contract_version!=internal_contract_version) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"wrong topology version");
  c->data.topology=t;
  return FV3_GPU_SUCCESS;
}
int mark_halo_ready(fv3_gpu_context *c, fv3_gpu_halo_group g, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  fv3_gpu_event ev=issue_event(c->data);
  int rc=c->data.dependencies.mark_ready(g,ev);
  if (rc!=0) return fail(e,static_cast<fv3_gpu_status>(rc),__func__,"invalid halo group or event");
  return FV3_GPU_SUCCESS;
}
}

extern "C" int fv3_gpu_validate_field(const fv3_gpu_field_desc *f, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e);
  if (!f || !header_ok(f->abi_version, f->struct_bytes, sizeof(*f))) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid field descriptor ABI");
  if (!f->base || f->rank == 0 || f->rank > FV3_GPU_MAX_RANK) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"base is null or rank is outside 1..4");
  const uint32_t expected = f->element_type == FV3_GPU_ELEMENT_F32 ? 4u : f->element_type == FV3_GPU_ELEMENT_F64 ? 8u : 0u;
  if (!expected || f->element_bytes != expected) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"element type and byte size disagree");
  if (f->memory_location != FV3_GPU_MEMORY_HOST && f->memory_location != FV3_GPU_MEMORY_DEVICE) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid memory location");
  uint64_t max_offset=0;
  for (uint32_t n=0;n<f->rank;++n) {
    if (!f->extent[n] || f->stride_elements[n] <= 0) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"extent must be nonzero and stride positive");
    const int64_t upper=f->lower[n]+(int64_t)f->extent[n]-1;
    if (f->compute_lower[n]<f->lower[n] || f->compute_upper[n]>upper || f->compute_lower[n]>f->compute_upper[n]) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"compute bounds lie outside allocation");
    if (f->extent[n]-1 > (std::numeric_limits<uint64_t>::max()-max_offset)/(uint64_t)f->stride_elements[n]) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"descriptor size overflow");
    max_offset += (f->extent[n]-1)*(uint64_t)f->stride_elements[n];
  }
  if (max_offset > (std::numeric_limits<uint64_t>::max()/f->element_bytes)-1 || (max_offset+1)*f->element_bytes > f->allocation_bytes) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"allocation_bytes is smaller than addressed view");
  if (f->alignment_bytes && ((uintptr_t)f->base % f->alignment_bytes)) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"base does not meet declared alignment");
  return FV3_GPU_SUCCESS;
}

extern "C" int fv3_gpu_create(const fv3_gpu_create_desc *d, fv3_gpu_context **out, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e);
  if (!out) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"context output is null");
  *out=nullptr;
  if (!d || !header_ok(d->abi_version,d->struct_bytes,sizeof(*d)) || d->mpi_size<1 || d->mpi_rank<0 || d->mpi_rank>=d->mpi_size || d->is>d->ie || d->js>d->je || d->npz<1)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid create descriptor");
  if ((uint32_t)active_precision != d->precision) return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"requested precision differs from compiled FV precision");
#ifndef FV3_GPU_ENABLE_CUDA
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#else
  int count=0; cudaError_t rc=cudaGetDeviceCount(&count); if(rc!=cudaSuccess) return cuda_fail(rc,e,__func__);
  if(count==0) return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"no CUDA devices are visible");
  int device=d->device_ordinal; if(device<0) device=d->mpi_rank%count;
  if(device<0 || device>=count) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"device ordinal is unavailable");
  rc=cudaSetDevice(device); if(rc!=cudaSuccess) return cuda_fail(rc,e,__func__);
  auto *ctx=new(std::nothrow) fv3_gpu_context; if(!ctx) return fail(e,FV3_GPU_OUT_OF_MEMORY,__func__,"context allocation failed");
  ctx->data.create=*d; ctx->data.create.device_ordinal=device;
  ctx->data.immutable.bounds={d->is,d->ie,d->js,d->je,d->npz,d->halo_width};
  ctx->data.immutable.topology=&ctx->data.topology;
  cudaStream_t compute{}, transfer{};
  rc=cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking); if(rc==cudaSuccess) rc=cudaStreamCreateWithFlags(&transfer,cudaStreamNonBlocking);
  if(rc!=cudaSuccess){ if(compute) cudaStreamDestroy(compute); delete ctx; return cuda_fail(rc,e,__func__); }
  ctx->data.compute_stream=compute; ctx->data.transfer_stream=transfer; *out=ctx; return FV3_GPU_SUCCESS;
#endif
}

extern "C" int fv3_gpu_initialize_state(fv3_gpu_context *c, const fv3_gpu_state_desc *s, fv3_gpu_event *ev, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0;
  if(!c || !s || !ev || !header_ok(s->abi_version,s->struct_bytes,sizeof(*s)) || !s->fields || !s->field_count) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid context, state descriptor, or event output");
  if(c->data.phase!=ContextPhase::created) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not in created state");
#ifdef FV3_GPU_ENABLE_CUDA
  std::vector<DeviceField> staged; staged.reserve(s->field_count); FieldRegistry registry; auto stream=(cudaStream_t)c->data.transfer_stream;
  for(uint32_t i=0;i<s->field_count;++i){ if(fv3_gpu_validate_field(&s->fields[i],e)!=0) goto rollback; for(const auto &prior:staged) if(prior.host.field_id==s->fields[i].field_id){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"duplicate field id");goto rollback;} DeviceField x; x.host=s->fields[i];
    if(cudaMalloc(&x.device,x.host.allocation_bytes)!=cudaSuccess) goto rollback; if(x.host.memory_location==FV3_GPU_MEMORY_HOST && cudaMemcpyAsync(x.device,x.host.base,x.host.allocation_bytes,cudaMemcpyHostToDevice,stream)!=cudaSuccess){cudaFree(x.device);goto rollback;} if(registry.insert(x.host,x.device,e)!=0){cudaFree(x.device);goto rollback;} x.host_epoch=x.device_epoch=1; staged.push_back(x); }
  { uint64_t ni=(uint64_t)(c->data.create.ie-c->data.create.is+1+2*c->data.create.halo_width),nj=(uint64_t)(c->data.create.je-c->data.create.js+1+2*c->data.create.halo_width),cells=0,values=0;
    if(mul_overflow(ni,nj,cells)||mul_overflow(cells,(uint64_t)c->data.create.npz+1,values)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace dimension overflow");goto rollback;}
    uint64_t value_bytes=0,cgrid_bytes=0,transport_bytes=0,vertical_bytes=0,operator_bytes=0;
    /* WorkspaceId::transport is sized in whole (ni x nj x npz+1) fv_real
     * arrays. It is shared, sequentially, by every WP-B service:
     *   Milestone 3A.4 update_dz_d height transport -- 17 arrays
     *   Milestone 3A.8 d_sw driver                  -- 34 arrays
     * The d_sw figure is WpBDSwArena::array_count (dgrid/wp_b_dsw_cuda.cuh),
     * which enumerates every sw_core.F90 local that is not a registered field:
     * ub/vb/wk/ke/vort/fx/fy/gx/gy/ra_x/ra_y/fx2/fy2/dw (14), the fv_tp_2d +
     * deln_flux scratch reused across the delp/w/pt/vort transports (7), the
     * PPM momentum workspace (4), smag_corner's private locals plus a2b_ord4
     * scratch (8) and one slot subdivided into the per-level derived
     * coefficient arrays (1). The binder re-checks the slice against that
     * constant, so a stale literal here fails loudly instead of corrupting
     * neighbouring scratch. */
    if(mul_overflow(values,sizeof(fv_real),value_bytes)||mul_overflow(value_bytes,12,cgrid_bytes)||mul_overflow(value_bytes,34,transport_bytes)||mul_overflow(value_bytes,16,vertical_bytes)||mul_overflow(value_bytes,12,operator_bytes)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace byte-size overflow");goto rollback;}
    WorkspaceRequest requests[]={{WorkspaceId::cgrid,cgrid_bytes,256},{WorkspaceId::transport,transport_bytes,256},{WorkspaceId::vertical,vertical_bytes,256},{WorkspaceId::operators,operator_bytes,256}};
    if(build_workspace_layout(requests,4,c->data.workspace.layout,e)!=0)goto rollback;if(cudaMalloc(&c->data.workspace.allocation,c->data.workspace.layout.total_bytes)!=cudaSuccess)goto rollback;
    for(auto &slice:c->data.workspace.layout.slices)slice.data=(char*)c->data.workspace.allocation+slice.offset; }
  c->data.state.fields=std::move(staged);c->data.state.registry=registry;c->data.immutable.grid=&c->data.state.registry;c->data.phase=ContextPhase::initialized; *ev=issue_event(c->data); return FV3_GPU_SUCCESS;
rollback: for(auto &x:staged) cudaFree(x.device);if(c->data.workspace.allocation){cudaFree(c->data.workspace.allocation);c->data.workspace={};} c->data.phase=ContextPhase::failed; if(e && e->code!=FV3_GPU_SUCCESS)return e->code; return fail(e,FV3_GPU_CUDA_ERROR,__func__,"state or workspace allocation/upload failed");
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}

extern "C" int fv3_gpu_sync_from_host(fv3_gpu_context *c,const fv3_gpu_field_mask *m,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0; if(!c||!ev)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context or event"); int rc=validate_mask(m,e,__func__);if(rc)return rc; if(c->data.phase!=ContextPhase::initialized)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
#ifdef FV3_GPU_ENABLE_CUDA
  auto stream=(cudaStream_t)c->data.transfer_stream; for(auto &x:c->data.state.fields) { if(x.host.field_id>=256)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"field id exceeds mask"); if((m->words[x.host.field_id/64]>>(x.host.field_id%64))&1u){ if(x.host.memory_location==FV3_GPU_MEMORY_HOST){auto z=cudaMemcpyAsync(x.device,x.host.base,x.host.allocation_bytes,cudaMemcpyHostToDevice,stream);if(z!=cudaSuccess)return cuda_fail(z,e,__func__,&c->data);} ++x.host_epoch;x.device_epoch=x.host_epoch;}} *ev=issue_event(c->data);return 0;
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
extern "C" int fv3_gpu_sync_to_host(fv3_gpu_context *c,const fv3_gpu_field_mask *m,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0; if(!c||!ev)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context or event"); int rc=validate_mask(m,e,__func__);if(rc)return rc; if(c->data.phase!=ContextPhase::initialized)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
#ifdef FV3_GPU_ENABLE_CUDA
  auto stream=(cudaStream_t)c->data.transfer_stream; for(auto &x:c->data.state.fields){if(x.host.field_id>=256)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"field id exceeds mask");if((m->words[x.host.field_id/64]>>(x.host.field_id%64))&1u && x.host.memory_location==FV3_GPU_MEMORY_HOST){auto z=cudaMemcpyAsync(x.host.base,x.device,x.host.allocation_bytes,cudaMemcpyDeviceToHost,stream);if(z!=cudaSuccess)return cuda_fail(z,e,__func__,&c->data);x.host_epoch=x.device_epoch;}} *ev=issue_event(c->data);return 0;
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
extern "C" int fv3_gpu_dyn_core_step(fv3_gpu_context *c,const fv3_gpu_step_desc *s,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0;
  if(!c||!s||!ev||!header_ok(s->abi_version,s->struct_bytes,sizeof(*s)))
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid context, step descriptor, or event output");
  if(c->data.phase!=ContextPhase::initialized) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
  if(s->dependency && s->dependency>c->data.last_event) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"step dependency references an unknown event");

  const NumericalDispatch &dispatch=c->data.dispatch;
  if(dispatch.contract_version!=internal_contract_version)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical dispatch table is not installed");
  const bool any_stage = dispatch.cgrid_predictor||dispatch.cgrid_vertical_solver||dispatch.cgrid_pressure||
                          dispatch.dgrid_dynamics||dispatch.dgrid_height_transport||dispatch.dgrid_height||
                          dispatch.vertical_solver||dispatch.nh_pressure||dispatch.post_loop_heating;
  if(!any_stage||!dispatch.corner_edge_service)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical dispatch table has no usable stages wired");

  // Milestone 3A.1: dispatch_params carries only what fv3_gpu_step_desc
  // actually provides. Multi-pass hyperdiffusion (diffusion_pass/passes) is
  // not exposed by the current ABI, so a single pass is assumed here; any
  // NumericalConfig requesting more passes will be rejected by the stage
  // itself (launch_post_loop_heating requires diffusion_passes==config.nf_ke).
  DispatchParams &p=c->data.dispatch_params;
  p.contract_version=internal_contract_version; p.dt=s->bdt; p.dt2=s->bdt*0.5;
  p.rdt=(s->bdt!=0.0)?1.0/s->bdt:0.0; p.model_step=s->model_step; p.acoustic_sequence=s->acoustic_sequence;
  p.acoustic_index=0; p.n_split=s->n_split>0?s->n_split:1; p.k_split=s->k_split>0?s->k_split:1; p.vertical_ms=1;
  p.diffusion_pass=0; p.diffusion_passes=1;
  p.init_step=s->init_step; p.end_step=s->end_step; p.last_call=s->end_step; p.fp_out=0; p.remap_step=0;

  ExecutionContext x{};
  x.contract_version=internal_contract_version;
  x.fields=&c->data.state.registry;
  x.immutable=&c->data.immutable;
  x.dispatch_params=&c->data.dispatch_params;
  x.workspace=&c->data.workspace.layout;
  x.submission=Submission{internal_contract_version,c->data.compute_stream,StreamId::compute,s->acoustic_sequence};
  x.dependencies=&c->data.dependencies;
  x.corner_service=dispatch.corner_edge_service;

  int rc=validate_execution_context(x,e);
  if(rc){ c->data.phase=ContextPhase::failed; return rc; }

  // Stage order follows dyn_core.F90's acoustic body: the C-grid predictor
  // group, then the D-grid dynamics (d_sw), then the vertical/nonhydrostatic
  // solve, then the pressure and post-loop operators. Only stages with a real,
  // wired CUDA implementation are invoked; a null slot is skipped rather than
  // treated as an orchestration error. Any stage that IS wired must fully
  // succeed or the step fails -- there is no silent host fallback inside this
  // CUDA path.
  //
  // Milestone 3A.8 adds dispatch.dgrid_dynamics (WP-B.5's complete d_sw
  // driver). Still intentionally null in the production table:
  //   cgrid_predictor, cgrid_vertical_solver -- no CUDA adapter integrated yet
  //   dgrid_height                           -- ditto
  // dgrid_height_transport IS wired, but it is not a free-standing acoustic
  // stage: it is the horizontal-transport body of WP-C's update_dz_d, and the
  // only caller is launch_update_dz_d (vertical/fv3_vertical_kernels.cu:172),
  // which takes the dispatch table as an argument. Listing it here would
  // transport zh a second time per substep, so it stays out.
  //
  // Be precise about what that means today: launch_vertical_solver does NOT
  // call launch_update_dz_d, so nothing on this list reaches
  // dgrid_height_transport. The slot is IMPLEMENTED and separately GPU
  // EXECUTED (Milestone 3A.4's focused test), but it is not yet reached
  // through fv3_gpu_dyn_core_step. That is a Gate C1 item, not a claim to
  // make here.
  //
  // NOTE: this list is walked ONCE per fv3_gpu_dyn_core_step call, so one call
  // advances one acoustic substep. The n_split acoustic loop is not implemented
  // in the runtime yet (the d_sw binder divides dispatch_params->dt by n_split
  // exactly as dyn_core.F90:297 does). That remains a Gate C1 item.
  const LaunchFn stages[]={dispatch.cgrid_predictor,dispatch.cgrid_vertical_solver,dispatch.cgrid_pressure,
                           dispatch.dgrid_dynamics,dispatch.vertical_solver,dispatch.nh_pressure,
                           dispatch.post_loop_heating};
  for(LaunchFn stage:stages){
    if(!stage) continue;
    rc=stage(x,e);
    if(rc!=FV3_GPU_SUCCESS){ c->data.phase=ContextPhase::failed; return rc; }
  }

  *ev=issue_event(c->data);
  return FV3_GPU_SUCCESS;
}
extern "C" int fv3_gpu_event_wait(fv3_gpu_context *c,fv3_gpu_event ev,fv3_gpu_error *e){using namespace fv3::gpu;clear_error(e);if(!c||!ev||ev>c->data.last_event)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"unknown event");
#ifdef FV3_GPU_ENABLE_CUDA
  return cuda_fail(cudaStreamSynchronize((cudaStream_t)c->data.transfer_stream),e,__func__,&c->data);
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
extern "C" int fv3_gpu_get_capabilities(fv3_gpu_context*,fv3_gpu_capabilities *p){if(!p||!header_ok(p->abi_version,p->struct_bytes,sizeof(*p)))return FV3_GPU_INVALID_ARGUMENT;
#ifdef FV3_GPU_ENABLE_CUDA
  p->compiled_with_cuda=1;int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess)n=0;p->device_count=(uint32_t)n;
#else
  p->compiled_with_cuda=0;p->device_count=0;
#endif
  p->precision_mask=(1u<<fv3::gpu::active_precision);p->communication_stage=1;p->feature_mask=0;return 0;}
extern "C" int fv3_gpu_finalize(fv3_gpu_context **pc,fv3_gpu_error *e){using namespace fv3::gpu;clear_error(e);if(!pc||!*pc)return 0;auto*c=*pc;
#ifdef FV3_GPU_ENABLE_CUDA
  cudaSetDevice(c->data.create.device_ordinal);cudaStreamSynchronize((cudaStream_t)c->data.compute_stream);cudaStreamSynchronize((cudaStream_t)c->data.transfer_stream);for(auto &x:c->data.state.fields)cudaFree(x.device);if(c->data.workspace.allocation)cudaFree(c->data.workspace.allocation);cudaStreamDestroy((cudaStream_t)c->data.compute_stream);cudaStreamDestroy((cudaStream_t)c->data.transfer_stream);
#endif
  delete c;*pc=nullptr;return 0;}
