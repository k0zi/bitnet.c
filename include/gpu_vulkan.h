#ifndef BN_GPU_VULKAN_H
#define BN_GPU_VULKAN_H

#ifdef BN_ENABLE_VULKAN

#include "gpu_backend.h"

BnGPUBackend *bn_gpu_vulkan_create(const char *shader_dir);
void bn_gpu_vulkan_destroy(BnGPUBackend *gpu);

#endif // BN_ENABLE_VULKAN

#endif // BN_GPU_VULKAN_H
