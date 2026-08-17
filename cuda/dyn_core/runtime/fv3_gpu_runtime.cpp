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
    if(mul_overflow(values,sizeof(fv_real),value_bytes)||mul_overflow(value_bytes,12,cgrid_bytes)||mul_overflow(value_bytes,24,transport_bytes)||mul_overflow(value_bytes,16,vertical_bytes)||mul_overflow(value_bytes,12,operator_bytes)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace byte-size overflow");goto rollback;}
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
extern "C" int fv3_gpu_dyn_core_step(fv3_gpu_context*,const fv3_gpu_step_desc*,fv3_gpu_event *ev,fv3_gpu_error *e){if(ev)*ev=0;return fv3::gpu::fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical work packages are not linked into this skeleton");}
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
