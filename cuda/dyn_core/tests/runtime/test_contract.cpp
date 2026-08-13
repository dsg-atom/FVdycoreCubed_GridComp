#include "fv3_gpu_internal.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
using namespace fv3::gpu;
static int launch(const ExecutionContext&,fv3_gpu_error*){return 0;}
int main(){
  fv_real data[24]{};fv3_gpu_field_desc d{};d.abi_version=FV3_GPU_ABI_VERSION;d.struct_bytes=sizeof(d);d.field_id=(uint32_t)FieldId::pt;std::strcpy(d.name,"pt");d.base=data;d.allocation_bytes=sizeof(data);d.element_type=sizeof(fv_real)==4?FV3_GPU_ELEMENT_F32:FV3_GPU_ELEMENT_F64;d.element_bytes=sizeof(fv_real);d.rank=3;d.memory_location=FV3_GPU_MEMORY_DEVICE;d.extent[0]=4;d.extent[1]=3;d.extent[2]=2;d.stride_elements[0]=1;d.stride_elements[1]=4;d.stride_elements[2]=12;for(int n=0;n<3;++n)d.compute_upper[n]=d.extent[n]-1;d.stagger=FV3_GPU_STAGGER_A;d.access=FV3_GPU_ACCESS_INOUT;
  FieldRegistry registry;fv3_gpu_error e{};assert(registry.insert(d,data,&e)==0);auto*v=registry.find(FieldId::pt);assert(v&&v->stride[2]==12);assert(registry.find(FieldId::u)==nullptr);assert(registry.insert(d,data,&e)==FV3_GPU_INVALID_ARGUMENT);
  WorkspaceRequest r[]={{WorkspaceId::vertical,100,64},{WorkspaceId::cgrid,17,16},{WorkspaceId::operators,33,32},{WorkspaceId::transport,9,8}};WorkspaceLayout w;assert(build_workspace_layout(r,4,w,&e)==0);for(size_t i=0;i<4;++i)for(size_t j=i+1;j<4;++j){auto&a=w.slices[i];auto&b=w.slices[j];assert(a.offset+a.bytes<=b.offset||b.offset+b.bytes<=a.offset);}assert(workspace_slice(w,WorkspaceId::vertical)->offset%64==0);
  ImmutableViews immutable;immutable.bounds={-2,5,-2,5,4,2};immutable.grid=&registry;DependencyState deps;assert(deps.mark_ready(FV3_GPU_H_CWIND,7)==0);fv3_gpu_event ev=0;assert(deps.require(FV3_GPU_H_CWIND,&ev)==0&&ev==7);assert(deps.mark_stale(FV3_GPU_H_CWIND)==0);assert(deps.require(FV3_GPU_H_CWIND,&ev)==FV3_GPU_STATE_CONFLICT);
  ExecutionContext x;x.fields=&registry;x.immutable=&immutable;x.workspace=&w;x.submission.stream=(void*)1;x.dependencies=&deps;assert(validate_execution_context(x,&e)==0);x.contract_version=1;assert(validate_execution_context(x,&e)==FV3_GPU_INVALID_ARGUMENT);
  NumericalDispatch table;table.cgrid_predictor=table.cgrid_vertical_solver=table.cgrid_pressure=table.dgrid_dynamics=table.dgrid_height_transport=table.dgrid_height=table.vertical_solver=table.nh_pressure=table.post_loop_heating=table.local_corner_fill=launch;assert(validate_dispatch(table,&e)==0);
}
