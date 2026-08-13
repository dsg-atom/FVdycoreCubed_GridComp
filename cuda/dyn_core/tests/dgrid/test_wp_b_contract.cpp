#include "dgrid/wp_b_contract.hpp"

#include <cassert>

int main() {
  fv3::gpu::ExecutionContext x{};
  fv3::gpu::dgrid::ContractViews views{};
  fv3_gpu_error error{};
  // An incomplete context must be rejected before any field lookup.
  assert(fv3::gpu::dgrid::resolve_contract_v3(x, views, &error) ==
         FV3_GPU_INVALID_ARGUMENT);
  return 0;
}
