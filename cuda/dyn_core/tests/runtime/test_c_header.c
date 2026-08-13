#include "fv3_gpu_api.h"
#include "fv3_gpu_communication.h"
int main(void) {
  fv3_gpu_field_desc field = {0};
  fv3_gpu_halo_group group = FV3_GPU_H_DWIND_IN;
  field.abi_version = FV3_GPU_ABI_VERSION;
  return (field.abi_version == 1u && group == 1) ? 0 : 1;
}
