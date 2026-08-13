#include "wp_d_contract.hpp"

#include <cassert>
#include <cstring>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::operators;

int main() {
  fv3_gpu_error error{};
  FieldRegistry registry;
  std::vector<unsigned char> storage(4096);
  for (uint32_t raw=1; raw<static_cast<uint32_t>(FieldId::count); ++raw) {
    const FieldSpec *spec=field_spec(static_cast<FieldId>(raw));
    assert(spec);
    fv3_gpu_field_desc d{};
    d.field_id=raw; d.rank=spec->rank; d.stagger=spec->stagger; d.access=spec->access;
    d.element_bytes=spec->logical_type==LogicalType::accumulator_real ? 8u :
                    spec->logical_type==LogicalType::bytes ? 1u : sizeof(fv_real);
    d.base=storage.data();
    for(uint32_t n=0;n<d.rank;++n){d.extent[n]=8;d.stride_elements[n]=n?8:1;}
    assert(registry.insert(d,storage.data(),&error)==FV3_GPU_SUCCESS);
  }
  std::vector<unsigned char> workspace(12*8*8*4*sizeof(fv_real));
  WorkspaceLayout layout{};
  layout.slices[static_cast<size_t>(WorkspaceId::operators)]={workspace.data(),0,workspace.size(),64};
  ImmutableViews immutable{}; immutable.bounds={0,3,0,3,3,2}; immutable.grid=&registry;
  DependencyState dependencies;
  dependencies.mark_ready(FV3_GPU_H_ZH,1);
  dependencies.mark_ready(FV3_GPU_H_PKC2,2);
  dependencies.mark_ready(FV3_GPU_H_HEAT,3);
  ExecutionContext x{}; x.fields=&registry;x.immutable=&immutable;x.workspace=&layout;
  x.submission.stream=reinterpret_cast<void *>(1);x.dependencies=&dependencies;
  WpDViews views;OperatorArena arena;
  assert(bind_wp_d(x,views,arena,&error)==FV3_GPU_SUCCESS);
  assert(arena.values_per_array==8u*8u*4u);
  assert(arena.arrays[1]==static_cast<unsigned char *>(arena.arrays[0])+8u*8u*4u*sizeof(fv_real));
  assert(launch_nh_pressure(x,&error)==FV3_GPU_UNSUPPORTED_CONFIGURATION);
  dependencies.mark_stale(FV3_GPU_H_ZH);
  assert(launch_nh_pressure(x,&error)==FV3_GPU_STATE_CONFLICT);
  assert(launch_post_loop_heating(x,&error)==FV3_GPU_UNSUPPORTED_CONFIGURATION);
}
