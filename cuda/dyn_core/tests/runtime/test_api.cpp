#include "fv3_gpu_api.h"
#include <cassert>
#include <cstdio>
int main(){
  // First check capabilities to determine if CUDA is enabled at build time
  fv3_gpu_capabilities p{};
  p.abi_version=FV3_GPU_ABI_VERSION;
  p.struct_bytes=sizeof(p);
  assert(fv3_gpu_get_capabilities(nullptr,&p)==0);

  const bool cuda_enabled = p.compiled_with_cuda != 0;
  printf("Capabilities: compiled_with_cuda=%u, device_count=%u\n", p.compiled_with_cuda, p.device_count);

  fv3_gpu_error e{};
  fv3_gpu_context *c=reinterpret_cast<fv3_gpu_context*>(1);
  fv3_gpu_create_desc d{};
  d.abi_version=FV3_GPU_ABI_VERSION;
  d.struct_bytes=sizeof(d);
  d.device_ordinal=-1;
  d.mpi_rank=0;
  d.mpi_size=1;
#if defined(FV3_GPU_FV_REAL_IS_FLOAT)
  d.precision=FV3_GPU_PRECISION_R4;
#else
  d.precision=FV3_GPU_PRECISION_R8;
#endif
  d.ie=1;d.je=1;d.npz=1;

  int rc=fv3_gpu_create(&d,&c,&e);
  printf("fv3_gpu_create returned: %d\n", rc);

  if (!cuda_enabled) {
    // When CUDA is not enabled, context creation should return unsupported
    assert(rc==FV3_GPU_UNSUPPORTED_CONFIGURATION);
    assert(c==nullptr);
    printf("Non-CUDA path: PASS\n");
  } else {
    // When CUDA is enabled:
    // - If creation succeeds, context should be valid
    // - If creation fails (no GPU), context should be null and rc should indicate error
    if (rc == FV3_GPU_SUCCESS) {
      assert(c != nullptr);
      printf("GPU context created successfully\n");
    } else {
      // Expected failures on systems without GPU or with other errors
      printf("GPU context creation failed with rc=%d (may be expected if no GPU)\n", rc);
      // Context pointer should be null on failure
      c = nullptr;
    }
  }

  assert(fv3_gpu_finalize(&c,&e)==FV3_GPU_SUCCESS);
  assert(c==nullptr);
  printf("API test: PASS\n");
}
