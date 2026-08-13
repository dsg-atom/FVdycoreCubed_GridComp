#ifndef FV3_GPU_COMMUNICATION_H
#define FV3_GPU_COMMUNICATION_H
#include "fv3_gpu_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum fv3_gpu_halo_group {
  FV3_GPU_H_DWIND_IN=1, FV3_GPU_H_W_IN, FV3_GPU_H_GZ_FIRST,
  FV3_GPU_H_Q_INLINE, FV3_GPU_H_DIVGD, FV3_GPU_H_CWIND,
  FV3_GPU_H_DSCALAR, FV3_GPU_H_ZH, FV3_GPU_H_PKC2,
  FV3_GPU_H_HEAT, FV3_GPU_H_DWIND_NEXT
} fv3_gpu_halo_group;
typedef int (*fv3_gpu_halo_begin_fn)(void *token, uint32_t group,
                                    void *send_buffer, uint64_t send_bytes,
                                    void *recv_buffer, uint64_t recv_bytes);
typedef int (*fv3_gpu_halo_complete_fn)(void *token, uint32_t group);
typedef struct fv3_gpu_halo_callbacks {
  uint32_t abi_version;
  uint32_t struct_bytes;
  void *token;
  fv3_gpu_halo_begin_fn begin;
  fv3_gpu_halo_complete_fn complete;
} fv3_gpu_halo_callbacks;
#ifdef __cplusplus
}
#endif
#endif
