#ifndef BN_GPU_ROCM_H
#define BN_GPU_ROCM_H
#ifdef BN_ENABLE_ROCM
#include "gpu_backend.h"
#ifdef __cplusplus
extern "C" {
#endif
BnGPUBackend *bn_gpu_rocm_create(void);
void          bn_gpu_rocm_destroy(BnGPUBackend *gpu);
#ifdef __cplusplus
}
#endif
#endif /* BN_ENABLE_ROCM */
#endif /* BN_GPU_ROCM_H */
