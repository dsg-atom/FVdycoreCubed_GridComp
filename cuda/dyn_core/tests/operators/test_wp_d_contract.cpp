#include "wp_d_contract.hpp"

#include <cassert>
#include <cstring>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::operators;

int main() {
  fv3_gpu_error error{};
  FieldRegistry state,grid;
  std::vector<unsigned char> storage(32768);
  for (uint32_t raw=1; raw<static_cast<uint32_t>(FieldId::count); ++raw) {
    const FieldSpec *spec=field_spec(static_cast<FieldId>(raw));
    assert(spec);
    fv3_gpu_field_desc d{};
    d.field_id=raw; d.rank=spec->rank; d.stagger=spec->stagger; d.access=spec->access;
    d.element_bytes=spec->logical_type==LogicalType::accumulator_real ? 8u :
                    spec->logical_type==LogicalType::bytes ? 1u : sizeof(fv_real);
    d.base=storage.data();
    uint64_t stride=1;
    for(uint32_t n=0;n<d.rank;++n) {
      d.lower[n]=-3;d.extent[n]=12;d.stride_elements[n]=stride;stride*=12;
    }
    const auto id=static_cast<FieldId>(raw);
    if(id==FieldId::pe || id==FieldId::peln) {
      d.lower[0]=-3;d.extent[0]=12;d.stride_elements[0]=1;
      d.lower[1]=1;d.extent[1]=4;d.stride_elements[1]=12;
      d.lower[2]=-3;d.extent[2]=12;d.stride_elements[2]=48;
    }
    FieldRegistry &owner=spec->field_class==FieldClass::immutable_grid?grid:state;
    assert(owner.insert(d,storage.data(),&error)==FV3_GPU_SUCCESS);
  }
  std::vector<unsigned char> workspace(12*10*10*4*sizeof(fv_real));
  WorkspaceLayout layout{};
  layout.slices[static_cast<size_t>(WorkspaceId::operators)]={workspace.data(),0,workspace.size(),64};
  TopologyView topology{};
  topology.tile=2;
  topology.neighbors[static_cast<size_t>(TileEdge::west)]={0,1};
  topology.neighbors[static_cast<size_t>(TileEdge::east)]={0,2};
  topology.sw_corner=1;
  ImmutableViews immutable{}; immutable.bounds={0,3,0,3,3,3};
  immutable.grid=&grid;immutable.topology=&topology;
  immutable.config.n_con=3;immutable.config.nf_ke=3;
  immutable.config.d_con=1;immutable.config.a2b_order=4;
  DispatchParams params{};params.dt=1;params.diffusion_pass=0;
  params.diffusion_passes=3;
  DependencyState dependencies;
  dependencies.mark_ready(FV3_GPU_H_ZH,1);
  dependencies.mark_ready(FV3_GPU_H_PKC2,2);
  dependencies.mark_ready(FV3_GPU_H_HEAT,3);
  auto dummy_corner_service=[](const ExecutionContext&,const CornerServiceRequest&,fv3_gpu_error*) noexcept -> int { return FV3_GPU_SUCCESS; };
  ExecutionContext x{}; x.fields=&state;x.immutable=&immutable;
  x.dispatch_params=&params;x.workspace=&layout;
  x.submission.stream=reinterpret_cast<void *>(1);x.dependencies=&dependencies;
  x.corner_service=dummy_corner_service;
  WpDViews views;OperatorArena arena;
  assert(bind_wp_d(x,views,arena,&error)==FV3_GPU_SUCCESS);
  assert(arena.values_per_array==10u*10u*4u);
  assert(arena.arrays[1]==static_cast<unsigned char *>(arena.arrays[0])+10u*10u*4u*sizeof(fv_real));
  const auto physical=physical_edges(topology);
  assert(physical.west&&!physical.east&&!physical.south&&!physical.north);
  const auto corners=physical_corners(topology);
  assert(corners.sw&&!corners.se&&!corners.ne&&!corners.nw);
  immutable.config.grav=9.8;immutable.config.rdgas=287;immutable.config.akap=0.286;
  assert(launch_nh_pressure(x,&error)==FV3_GPU_SUCCESS);
  auto edge_replace=a2b_edge_request(FieldId::pkc,FieldId::pkc,4,true,2);
  assert(validate_corner_request(x,edge_replace,&error)==FV3_GPU_SUCCESS);
  auto edge_nonreplace=a2b_edge_request(FieldId::delp,FieldId::heat_source,4,false,2);
  assert(validate_corner_request(x,edge_nonreplace,&error)==FV3_GPU_SUCCESS);
  auto invalid_edge=a2b_edge_request(FieldId::pkc,FieldId::pkc,3,true,2);
  assert(validate_corner_request(x,invalid_edge,&error)==FV3_GPU_INVALID_ARGUMENT);
  auto copy=diffusion_corner_request(Axis::x,2,0,DiffusionPhase::x_corner_copy,3);
  assert(validate_corner_request(x,copy,&error)==FV3_GPU_SUCCESS);
  auto invalid_copy=diffusion_corner_request(Axis::both,2,-1,DiffusionPhase::x_corner_copy,3);
  assert(validate_corner_request(x,invalid_copy,&error)==FV3_GPU_INVALID_ARGUMENT);
  auto no_phase=diffusion_corner_request(Axis::x,2,0,DiffusionPhase::none,3);
  assert(no_phase.first==FieldId::invalid);

  FieldRegistry undersized=state;
  auto &delp=undersized.views[static_cast<size_t>(FieldId::delp)];
  delp.lower[0]=0;delp.extent[0]=4;
  x.fields=&undersized;
  assert(bind_wp_d(x,views,arena,&error)==FV3_GPU_INVALID_ARGUMENT);
  x.fields=&state;

  dependencies.mark_stale(FV3_GPU_H_ZH);
  assert(launch_nh_pressure(x,&error)==FV3_GPU_STATE_CONFLICT);
  dependencies.mark_ready(FV3_GPU_H_ZH,1);
  assert(launch_post_loop_heating(x,&error)==FV3_GPU_SUCCESS);

  immutable.config.n_con=0;
  assert(launch_post_loop_heating(x,&error)==FV3_GPU_SUCCESS);
  immutable.config.n_con=3;

  ExecutionContext no_service=x;no_service.corner_service=nullptr;
  assert(launch_nh_pressure(no_service,&error)==FV3_GPU_INVALID_ARGUMENT);
  assert(launch_post_loop_heating(no_service,&error)==FV3_GPU_INVALID_ARGUMENT);
}
