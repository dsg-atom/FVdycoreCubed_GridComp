#include "fv3_gpu_api.h"
#include <cassert>
int main(){
  fv3_gpu_error e{}; fv3_gpu_context *c=reinterpret_cast<fv3_gpu_context*>(1);
  fv3_gpu_create_desc d{};d.abi_version=FV3_GPU_ABI_VERSION;d.struct_bytes=sizeof(d);d.device_ordinal=-1;d.mpi_rank=0;d.mpi_size=1;
#if defined(FV3_GPU_FV_REAL_IS_FLOAT)
  d.precision=FV3_GPU_PRECISION_R4;
#else
  d.precision=FV3_GPU_PRECISION_R8;
#endif
  d.ie=1;d.je=1;d.npz=1;
  int rc=fv3_gpu_create(&d,&c,&e);
#ifndef FV3_GPU_ENABLE_CUDA
  assert(rc==FV3_GPU_UNSUPPORTED_CONFIGURATION);assert(c==nullptr);
#endif
  assert(fv3_gpu_finalize(&c,&e)==FV3_GPU_SUCCESS);assert(c==nullptr);
  fv3_gpu_capabilities p{};p.abi_version=FV3_GPU_ABI_VERSION;p.struct_bytes=sizeof(p);assert(fv3_gpu_get_capabilities(nullptr,&p)==0);
}
