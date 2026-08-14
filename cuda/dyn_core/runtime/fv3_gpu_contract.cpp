#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_errors.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace fv3::gpu {
namespace {
#define S(ID,CLS,TYPE,RANK,STAG,ACCESS) {FieldId::ID,#ID,FieldClass::CLS,LogicalType::TYPE,RANK,FV3_GPU_STAGGER_##STAG,FV3_GPU_ACCESS_##ACCESS}
constexpr FieldSpec specs[] = {
 S(u,mutable_state,fv_real,3,D_U,INOUT), S(v,mutable_state,fv_real,3,D_V,INOUT),
 S(w,mutable_state,fv_real,3,A,INOUT), S(delz,mutable_state,fv_real,3,A,INOUT),
 S(pt,mutable_state,fv_real,3,A,INOUT), S(delp,mutable_state,fv_real,3,A,INOUT),
 S(q,mutable_state,fv_real,4,A,INOUT), S(q_con,mutable_state,fv_real,3,A,INOUT),
 S(cappa,mutable_state,fv_real,3,A,INOUT), S(pe,mutable_state,fv_real,3,INTERFACE,INOUT),
 S(pk,mutable_state,fv_real,3,INTERFACE,INOUT), S(peln,mutable_state,fv_real,3,INTERFACE,INOUT),
 S(pkz,mutable_state,fv_real,3,A,INOUT), S(phis,immutable_grid,fv_real,2,A,INPUT),
 S(omga,mutable_state,fv_real,3,A,INOUT), S(ua,mutable_state,fv_real,3,A,INOUT),
 S(va,mutable_state,fv_real,3,A,INOUT), S(uc,mutable_state,fv_real,3,C_U,INOUT),
 S(vc,mutable_state,fv_real,3,C_V,INOUT), S(mfx,mutable_state,accumulator_real,3,X_FACE,INOUT),
 S(mfy,mutable_state,accumulator_real,3,Y_FACE,INOUT), S(cx,mutable_state,accumulator_real,3,X_FACE,INOUT),
 S(cy,mutable_state,accumulator_real,3,Y_FACE,INOUT), S(dpx,mutable_state,accumulator_real,3,A,INOUT),
 S(diss_est,mutable_state,fv_real,3,A,INOUT), S(ws,mutable_state,fv_real,2,A,OUTPUT),
 S(delpc,shared_temporary,fv_real,3,A,INOUT), S(ptc,shared_temporary,fv_real,3,A,INOUT),
 S(ut,shared_temporary,fv_real,3,A,INOUT), S(vt,shared_temporary,fv_real,3,A,INOUT),
 S(divgd,shared_temporary,fv_real,3,B,INOUT), S(crx,shared_temporary,fv_real,3,X_FACE,INOUT),
 S(xfx,shared_temporary,fv_real,3,X_FACE,INOUT), S(cry,shared_temporary,fv_real,3,Y_FACE,INOUT),
 S(yfx,shared_temporary,fv_real,3,Y_FACE,INOUT), S(gz,shared_temporary,fv_real,3,INTERFACE,INOUT),
 S(zh,shared_temporary,fv_real,3,INTERFACE,INOUT), S(pkc,shared_temporary,fv_real,3,INTERFACE,INOUT),
 S(pk3,shared_temporary,fv_real,3,INTERFACE,INOUT), S(du,shared_temporary,fv_real,3,D_U,INOUT),
 S(dv,shared_temporary,fv_real,3,D_V,INOUT), S(heat_source,shared_temporary,fv_real,3,A,INOUT),
 S(ws3,shared_temporary,fv_real,2,A,INOUT), S(z_rat,shared_temporary,fv_real,3,A,INOUT),
 S(divg2,shared_temporary,fv_real,2,B,INOUT),
 S(ak,immutable_grid,fv_real,1,INTERFACE,INPUT), S(bk,immutable_grid,fv_real,1,INTERFACE,INPUT),
 S(pfull,immutable_grid,fv_real,1,A,INPUT), S(dp_ref,immutable_grid,fv_real,1,A,INPUT),
 S(zs,immutable_grid,fv_real,2,A,INPUT), S(area,immutable_grid,fv_real,2,A,INPUT),
 S(rarea,immutable_grid,fv_real,2,A,INPUT), S(dx,immutable_grid,fv_real,2,X_FACE,INPUT),
 S(dy,immutable_grid,fv_real,2,Y_FACE,INPUT), S(dxc,immutable_grid,fv_real,2,C_U,INPUT),
 S(dyc,immutable_grid,fv_real,2,C_V,INPUT), S(rdx,immutable_grid,fv_real,2,X_FACE,INPUT),
 S(rdy,immutable_grid,fv_real,2,Y_FACE,INPUT), S(rdxc,immutable_grid,fv_real,2,C_U,INPUT),
 S(rdyc,immutable_grid,fv_real,2,C_V,INPUT), S(cosa_u,immutable_grid,fv_real,2,D_U,INPUT),
 S(cosa_v,immutable_grid,fv_real,2,D_V,INPUT), S(sina_u,immutable_grid,fv_real,2,D_U,INPUT),
 S(sina_v,immutable_grid,fv_real,2,D_V,INPUT), S(rsin_u,immutable_grid,fv_real,2,D_U,INPUT),
 S(rsin_v,immutable_grid,fv_real,2,D_V,INPUT), S(sin_sg,immutable_grid,fv_real,3,B,INPUT),
 S(cos_sg,immutable_grid,fv_real,3,B,INPUT), S(f0,immutable_grid,fv_real,2,B,INPUT),
 S(del6_u,immutable_grid,fv_real,2,D_U,INPUT), S(del6_v,immutable_grid,fv_real,2,D_V,INPUT),
 S(cosa_s,immutable_grid,fv_real,2,A,INPUT), S(rsin2,immutable_grid,fv_real,2,A,INPUT),
 S(dxa,immutable_grid,fv_real,2,A,INPUT), S(dya,immutable_grid,fv_real,2,A,INPUT),
 S(rarea_c,immutable_grid,fv_real,2,B,INPUT), S(area_c,immutable_grid,fv_real,2,B,INPUT), S(f_c,immutable_grid,fv_real,2,B,INPUT),
 S(rdxa,immutable_grid,fv_real,2,A,INPUT), S(rdya,immutable_grid,fv_real,2,A,INPUT),
 S(cosa,immutable_grid,fv_real,2,A,INPUT), S(rsina,immutable_grid,fv_real,2,A,INPUT),
 S(divg_u,immutable_grid,fv_real,2,D_U,INPUT), S(divg_v,immutable_grid,fv_real,2,D_V,INPUT),
 S(grid_lonlat,immutable_grid,fv_real,3,A,INPUT), S(agrid_lonlat,immutable_grid,fv_real,3,A,INPUT),
 S(edge_w,immutable_grid,accumulator_real,1,Y_FACE,INPUT), S(edge_e,immutable_grid,accumulator_real,1,Y_FACE,INPUT),
 S(edge_s,immutable_grid,accumulator_real,1,X_FACE,INPUT), S(edge_n,immutable_grid,accumulator_real,1,X_FACE,INPUT),
 S(damping_d2_bg,immutable_grid,fv_real,1,A,INPUT), S(damping_d4_bg,immutable_grid,fv_real,1,A,INPUT),
 S(damping_v,immutable_grid,fv_real,1,A,INPUT), S(damping_w,immutable_grid,fv_real,1,A,INPUT),
 S(damping_t,immutable_grid,fv_real,1,A,INPUT), S(damping_kgb,immutable_grid,fv_real,1,A,INPUT),
 S(damping_ndif,immutable_grid,integer,1,A,INPUT)
};
#undef S
uint64_t align_up(uint64_t value,uint32_t alignment) { return (value+alignment-1)&~uint64_t(alignment-1); }
bool valid_group(fv3_gpu_halo_group g){return g>=FV3_GPU_H_DWIND_IN&&g<=FV3_GPU_H_DWIND_NEXT;}
}

const FieldSpec *field_spec(FieldId id) noexcept { for(const auto &s:specs)if(s.id==id)return &s;return nullptr; }
FieldId field_id_from_abi(uint32_t id) noexcept { auto f=static_cast<FieldId>(id);return field_spec(f)?f:FieldId::invalid; }

int make_device_view(const fv3_gpu_field_desc &d,const FieldSpec &s,DeviceViewBase &v,fv3_gpu_error *e) noexcept {
  if(d.field_id!=static_cast<uint32_t>(s.id)||d.rank!=s.rank||d.stagger!=s.stagger||d.access!=s.access)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"descriptor does not match stable field specification");
  uint32_t bytes=s.logical_type==LogicalType::accumulator_real?8u:s.logical_type==LogicalType::fv_real?(uint32_t)sizeof(fv_real):d.element_bytes;
  if(s.logical_type!=LogicalType::bytes&&d.element_bytes!=bytes)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"field precision differs from contract");
  v.data=d.base;v.rank=d.rank;v.stagger=(fv3_gpu_stagger)d.stagger;v.access=(fv3_gpu_access)d.access;
  std::copy_n(d.lower,FV3_GPU_MAX_RANK,v.lower.begin());std::copy_n(d.extent,FV3_GPU_MAX_RANK,v.extent.begin());std::copy_n(d.stride_elements,FV3_GPU_MAX_RANK,v.stride.begin());return 0;
}
int FieldRegistry::insert(const fv3_gpu_field_desc &d,void *device,fv3_gpu_error *e) noexcept {
  auto id=field_id_from_abi(d.field_id);auto *s=field_spec(id);if(!s)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"unknown stable field id");
  auto n=static_cast<size_t>(id);if(present[n])return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"duplicate stable field id");fv3_gpu_field_desc copy=d;copy.base=device;
  int rc=make_device_view(copy,*s,views[n],e);if(!rc)present[n]=1;return rc;
}
const DeviceViewBase *FieldRegistry::find(FieldId id) const noexcept {auto n=static_cast<size_t>(id);return n<present.size()&&present[n]?&views[n]:nullptr;}

int build_workspace_layout(const WorkspaceRequest *r,size_t count,WorkspaceLayout &out,fv3_gpu_error *e) noexcept {
  out={};if((count&&!r)||count>out.slices.size())return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid workspace request list");
  std::array<uint8_t,static_cast<size_t>(WorkspaceId::count)> seen{};uint64_t cursor=0;
  for(size_t i=0;i<count;++i){auto n=static_cast<size_t>(r[i].id);uint32_t a=r[i].alignment;if(n>=seen.size()||seen[n]||!a||(a&(a-1))||a>4096)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace id duplicated or alignment invalid");
    seen[n]=1;uint64_t start=align_up(cursor,a);if(r[i].bytes>std::numeric_limits<uint64_t>::max()-start)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace size overflow");out.slices[n]={nullptr,start,r[i].bytes,a};cursor=start+r[i].bytes;}
  out.total_bytes=cursor;return 0;
}
const WorkspaceSlice *workspace_slice(const WorkspaceLayout &w,WorkspaceId id) noexcept {auto n=static_cast<size_t>(id);if(n>=w.slices.size()||!w.slices[n].bytes)return nullptr;return &w.slices[n];}

int DependencyState::require(fv3_gpu_halo_group g,fv3_gpu_event *ev) const noexcept {if(!valid_group(g)||!ev)return FV3_GPU_INVALID_ARGUMENT;auto n=(size_t)g;if(halo[n]!=HaloState::ready)return FV3_GPU_STATE_CONFLICT;*ev=ready_event[n];return 0;}
int DependencyState::mark_stale(fv3_gpu_halo_group g) noexcept {if(!valid_group(g))return FV3_GPU_INVALID_ARGUMENT;halo[(size_t)g]=HaloState::stale;ready_event[(size_t)g]=0;return 0;}
int DependencyState::mark_pending(fv3_gpu_halo_group g,fv3_gpu_event ev) noexcept {if(!valid_group(g)||!ev)return FV3_GPU_INVALID_ARGUMENT;halo[(size_t)g]=HaloState::exchange_pending;ready_event[(size_t)g]=ev;return 0;}
int DependencyState::mark_ready(fv3_gpu_halo_group g,fv3_gpu_event ev) noexcept {if(!valid_group(g)||!ev)return FV3_GPU_INVALID_ARGUMENT;halo[(size_t)g]=HaloState::ready;ready_event[(size_t)g]=ev;return 0;}
int validate_execution_context(const ExecutionContext &x,fv3_gpu_error *e) noexcept {if(x.contract_version!=internal_contract_version||!x.fields||!x.immutable||!x.immutable->grid||!x.immutable->topology||!x.dispatch_params||x.dispatch_params->contract_version!=internal_contract_version||!x.workspace||!x.submission.stream||!x.dependencies||!x.corner_service)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"incomplete or wrong-version execution context");return 0;}
int validate_corner_request(const ExecutionContext &x,const CornerServiceRequest &r,fv3_gpu_error *e) noexcept {
  int rc=validate_execution_context(x,e);if(rc)return rc;
  if(r.contract_version!=internal_contract_version||r.first==FieldId::invalid||!x.fields->find(r.first)||!r.halo_width||r.halo_width>x.immutable->bounds.halo_width)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid corner request field, version, or width");
  if((r.operation==CornerOperation::fill_pair||r.operation==CornerOperation::vector_transform)&&(r.second==FieldId::invalid||!x.fields->find(r.second)))return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"paired corner operation requires second field");
  if(r.operation==CornerOperation::a_to_b_edge&&!r.replace&&(r.destination==FieldId::invalid||!x.fields->find(r.destination)))return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"A-to-B non-replace operation requires valid destination field");
  if(r.interpolation_order<0||r.interpolation_order>4||(r.sign_x!=1&&r.sign_x!=-1)||(r.sign_y!=1&&r.sign_y!=-1))return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid corner interpolation/sign parameters");
  if(r.replace!=0&&r.replace!=1)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid replace flag");
  return 0;
}
int validate_dispatch(const NumericalDispatch &d,fv3_gpu_error *e) noexcept {if(d.contract_version!=internal_contract_version)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"wrong dispatch version");if(!d.cgrid_predictor||!d.cgrid_vertical_solver||!d.cgrid_pressure||!d.dgrid_dynamics||!d.dgrid_height_transport||!d.dgrid_height||!d.vertical_solver||!d.nh_pressure||!d.post_loop_heating||!d.corner_edge_service)return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical dispatch table is incomplete");return 0;}
}
