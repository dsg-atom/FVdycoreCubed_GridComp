#include "fv3_gpu_api.h"
#include <cassert>
#include <cstring>

int main() {
  double a[4*3*2]{};
  fv3_gpu_field_desc f{}; f.abi_version=FV3_GPU_ABI_VERSION; f.struct_bytes=sizeof(f);
  std::strcpy(f.name,"pt"); f.base=a; f.allocation_bytes=sizeof(a);
  f.element_type=FV3_GPU_ELEMENT_F64; f.element_bytes=8; f.rank=3; f.memory_location=FV3_GPU_MEMORY_HOST;
  f.extent[0]=4;f.extent[1]=3;f.extent[2]=2; f.stride_elements[0]=1;f.stride_elements[1]=4;f.stride_elements[2]=12;
  for(int n=0;n<3;++n){f.compute_lower[n]=0;f.compute_upper[n]=(int64_t)f.extent[n]-1;}
  f.stagger=FV3_GPU_STAGGER_A;f.access=FV3_GPU_ACCESS_INOUT;
  fv3_gpu_error e{}; assert(fv3_gpu_validate_field(&f,&e)==FV3_GPU_SUCCESS);
  f.stride_elements[0]=0; assert(fv3_gpu_validate_field(&f,&e)==FV3_GPU_INVALID_ARGUMENT);
  f.stride_elements[0]=1;f.abi_version=99;assert(fv3_gpu_validate_field(&f,&e)==FV3_GPU_INVALID_ARGUMENT);
}
