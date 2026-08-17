#include "fv3_gpu_internal.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
using namespace fv3::gpu;
static int launch(const ExecutionContext&,fv3_gpu_error*){return 0;}
static int corner(const ExecutionContext&,const CornerServiceRequest&,fv3_gpu_error*){return 0;}
int main(){
  fv_real data[24]{};fv3_gpu_field_desc d{};d.abi_version=FV3_GPU_ABI_VERSION;d.struct_bytes=sizeof(d);d.field_id=(uint32_t)FieldId::pt;std::strcpy(d.name,"pt");d.base=data;d.allocation_bytes=sizeof(data);d.element_type=sizeof(fv_real)==4?FV3_GPU_ELEMENT_F32:FV3_GPU_ELEMENT_F64;d.element_bytes=sizeof(fv_real);d.rank=3;d.memory_location=FV3_GPU_MEMORY_DEVICE;d.extent[0]=4;d.extent[1]=3;d.extent[2]=2;d.stride_elements[0]=1;d.stride_elements[1]=4;d.stride_elements[2]=12;for(int n=0;n<3;++n)d.compute_upper[n]=d.extent[n]-1;d.stagger=FV3_GPU_STAGGER_A;d.access=FV3_GPU_ACCESS_INOUT;
  FieldRegistry registry;fv3_gpu_error e{};assert(registry.insert(d,data,&e)==0);auto*v=registry.find(FieldId::pt);assert(v&&v->stride[2]==12);assert(registry.find(FieldId::u)==nullptr);assert(registry.insert(d,data,&e)==FV3_GPU_INVALID_ARGUMENT);
  fv_real metric[12]{};fv3_gpu_field_desc g=d;g.base=metric;g.allocation_bytes=sizeof(metric);g.rank=2;g.extent[0]=4;g.extent[1]=3;g.extent[2]=0;g.stride_elements[0]=1;g.stride_elements[1]=4;g.stride_elements[2]=0;g.compute_upper[0]=3;g.compute_upper[1]=2;g.compute_upper[2]=0;g.access=FV3_GPU_ACCESS_INPUT;g.stagger=FV3_GPU_STAGGER_A;g.field_id=(uint32_t)FieldId::cosa_s;assert(registry.insert(g,metric,&e)==0);g.field_id=(uint32_t)FieldId::rsin2;assert(registry.insert(g,metric,&e)==0);assert(registry.find(FieldId::cosa_s)!=registry.find(FieldId::rsin2));
  fv_real profile[4]{};fv3_gpu_field_desc p=g;p.base=profile;p.allocation_bytes=sizeof(profile);p.rank=1;p.extent[0]=4;p.extent[1]=0;p.stride_elements[0]=1;p.stride_elements[1]=0;p.compute_upper[0]=3;p.compute_upper[1]=0;p.field_id=(uint32_t)FieldId::damping_v;assert(registry.insert(p,profile,&e)==0);assert(registry.find(FieldId::damping_v)->extent[0]==4);
  WorkspaceRequest r[]={{WorkspaceId::vertical,100,64},{WorkspaceId::cgrid,17,16},{WorkspaceId::operators,33,32},{WorkspaceId::transport,9,8}};WorkspaceLayout w;assert(build_workspace_layout(r,4,w,&e)==0);for(size_t i=0;i<4;++i)for(size_t j=i+1;j<4;++j){auto&a=w.slices[i];auto&b=w.slices[j];assert(a.offset+a.bytes<=b.offset||b.offset+b.bytes<=a.offset);}assert(workspace_slice(w,WorkspaceId::vertical)->offset%64==0);
  TopologyView topology;topology.sw_corner=1;topology.neighbors[0].orientation=EdgeOrientation::swap_xy;ImmutableViews immutable;immutable.bounds={-2,5,-2,5,4,2};immutable.grid=&registry;immutable.topology=&topology;immutable.config.hord_tr=8;immutable.config.hord_dp=6;immutable.config.nord_v=2;immutable.config.beta=0.25;assert(immutable.config.hord_tr!=immutable.config.hord_dp&&immutable.config.beta==0.25);assert(immutable.topology->sw_corner==1);DependencyState deps;assert(deps.mark_ready(FV3_GPU_H_CWIND,7)==0);fv3_gpu_event ev=0;assert(deps.require(FV3_GPU_H_CWIND,&ev)==0&&ev==7);assert(deps.mark_stale(FV3_GPU_H_CWIND)==0);assert(deps.require(FV3_GPU_H_CWIND,&ev)==FV3_GPU_STATE_CONFLICT);
  DispatchParams params;params.dt=30;params.vertical_ms=4;params.last_call=1;params.fp_out=1;ExecutionContext x;x.fields=&registry;x.immutable=&immutable;x.dispatch_params=&params;x.workspace=&w;x.submission.stream=(void*)1;x.dependencies=&deps;x.corner_service=corner;assert(validate_execution_context(x,&e)==0);assert(x.dispatch_params->vertical_ms==4&&x.dispatch_params->last_call&&x.dispatch_params->fp_out);
  CornerServiceRequest request;request.first=FieldId::pt;request.halo_width=2;request.axis=Axis::y;assert(validate_corner_request(x,request,&e)==0);request.halo_width=3;assert(validate_corner_request(x,request,&e)==FV3_GPU_INVALID_ARGUMENT);request.halo_width=1;
  NumericalDispatch table;table.cgrid_predictor=table.cgrid_vertical_solver=table.cgrid_pressure=table.dgrid_dynamics=table.dgrid_height_transport=table.dgrid_height=table.vertical_solver=table.nh_pressure=table.post_loop_heating=launch;table.corner_edge_service=corner;assert(validate_dispatch(table,&e)==0);x.contract_version=1;assert(validate_execution_context(x,&e)==FV3_GPU_INVALID_ARGUMENT);x.contract_version=internal_contract_version;
  // V3.1 corner_service access
  assert(x.corner_service!=nullptr);assert(x.corner_service(x,request,&e)==0);
  // V3.1 A-to-B replace semantics
  CornerServiceRequest a2b_replace;a2b_replace.operation=CornerOperation::a_to_b_edge;a2b_replace.first=FieldId::pt;a2b_replace.replace=1;a2b_replace.halo_width=1;assert(validate_corner_request(x,a2b_replace,&e)==0);
  CornerServiceRequest a2b_no_replace;a2b_no_replace.operation=CornerOperation::a_to_b_edge;a2b_no_replace.first=FieldId::pt;a2b_no_replace.replace=0;a2b_no_replace.destination=FieldId::cosa_s;a2b_no_replace.halo_width=1;assert(validate_corner_request(x,a2b_no_replace,&e)==0);
  CornerServiceRequest a2b_bad;a2b_bad.operation=CornerOperation::a_to_b_edge;a2b_bad.first=FieldId::pt;a2b_bad.replace=0;a2b_bad.destination=FieldId::invalid;a2b_bad.halo_width=1;assert(validate_corner_request(x,a2b_bad,&e)==FV3_GPU_INVALID_ARGUMENT);
  // V3.1 diffusion phases
  assert(DiffusionPhase::none==DiffusionPhase(0));assert(DiffusionPhase::x_corner_copy!=DiffusionPhase::x_flux);assert(DiffusionPhase::y_corner_copy!=DiffusionPhase::y_flux);assert(DiffusionPhase::update!=DiffusionPhase::none);
  CornerServiceRequest diff_req;diff_req.first=FieldId::pt;diff_req.halo_width=1;diff_req.diffusion_phase=DiffusionPhase::x_corner_copy;assert(validate_corner_request(x,diff_req,&e)==0);diff_req.diffusion_phase=DiffusionPhase::y_flux;assert(validate_corner_request(x,diff_req,&e)==0);
  // V3.1 context requires corner_service
  ExecutionContext x_no_cs=x;x_no_cs.corner_service=nullptr;assert(validate_execution_context(x_no_cs,&e)==FV3_GPU_INVALID_ARGUMENT);
  // V3.1 edge descriptor type/rank correction (authoritative Fortran: R_GRID=real64, rank 1)
  auto*edge_spec_w=field_spec(FieldId::edge_w);auto*edge_spec_e=field_spec(FieldId::edge_e);auto*edge_spec_s=field_spec(FieldId::edge_s);auto*edge_spec_n=field_spec(FieldId::edge_n);
  assert(edge_spec_w&&edge_spec_w->rank==1&&edge_spec_w->logical_type==LogicalType::accumulator_real);
  assert(edge_spec_e&&edge_spec_e->rank==1&&edge_spec_e->logical_type==LogicalType::accumulator_real);
  assert(edge_spec_s&&edge_spec_s->rank==1&&edge_spec_s->logical_type==LogicalType::accumulator_real);
  assert(edge_spec_n&&edge_spec_n->rank==1&&edge_spec_n->logical_type==LogicalType::accumulator_real);
  // edge_w/e indexed by j (Y_FACE), edge_s/n indexed by i (X_FACE)
  assert(edge_spec_w->stagger==FV3_GPU_STAGGER_Y_FACE);assert(edge_spec_e->stagger==FV3_GPU_STAGGER_Y_FACE);
  assert(edge_spec_s->stagger==FV3_GPU_STAGGER_X_FACE);assert(edge_spec_n->stagger==FV3_GPU_STAGGER_X_FACE);
  // Insert edge descriptor with correct rank-1 real64 specification
  FieldRegistry edge_reg;double edge_arr[8]{};fv3_gpu_field_desc edge_d{};edge_d.abi_version=FV3_GPU_ABI_VERSION;edge_d.struct_bytes=sizeof(edge_d);
  edge_d.field_id=(uint32_t)FieldId::edge_w;std::strcpy(edge_d.name,"edge_w");edge_d.base=edge_arr;edge_d.allocation_bytes=sizeof(edge_arr);
  edge_d.element_type=FV3_GPU_ELEMENT_F64;edge_d.element_bytes=8;edge_d.rank=1;edge_d.memory_location=FV3_GPU_MEMORY_DEVICE;
  edge_d.extent[0]=8;edge_d.stride_elements[0]=1;edge_d.compute_upper[0]=7;edge_d.stagger=FV3_GPU_STAGGER_Y_FACE;edge_d.access=FV3_GPU_ACCESS_INPUT;
  assert(edge_reg.insert(edge_d,edge_arr,&e)==0);assert(edge_reg.find(FieldId::edge_w)->rank==1);
  // Reject wrong rank (rank 2)
  fv3_gpu_field_desc edge_bad=edge_d;edge_bad.rank=2;edge_bad.extent[1]=1;edge_bad.stride_elements[1]=8;edge_bad.compute_upper[1]=0;
  assert(edge_reg.insert(edge_bad,edge_arr,&e)==FV3_GPU_INVALID_ARGUMENT);
  // Reject wrong precision (fv_real instead of real64)
  fv3_gpu_field_desc edge_wrong_type=edge_d;edge_wrong_type.field_id=(uint32_t)FieldId::edge_e;edge_wrong_type.element_bytes=sizeof(fv_real);edge_wrong_type.element_type=sizeof(fv_real)==4?FV3_GPU_ELEMENT_F32:FV3_GPU_ELEMENT_F64;
  if(sizeof(fv_real)==4)assert(edge_reg.insert(edge_wrong_type,edge_arr,&e)==FV3_GPU_INVALID_ARGUMENT);
}
