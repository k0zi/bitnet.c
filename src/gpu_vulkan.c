#include "gpu_vulkan.h"

#ifdef BN_ENABLE_VULKAN

#include "gguf.h"
#include "model_config.h"
#include "gpu_shader.h"
#include "gpu_shader_ir_internal.h"

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#define _POSIX_C_SOURCE 200809L
#endif

#define BN_VULKAN_MAX_TYPES   40
#define BN_VULKAN_MAX_SETS  4096
#define BN_VULKAN_MAX_BINDINGS 8
#define BN_VULKAN_STAGING_MB   64

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    size_t size;
    int type;
    int rows;
    int cols;
} BnVulkanBuffer;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice phys_dev;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkDescriptorPool desc_pool;
    VkDescriptorSetLayout set_layouts[BN_VULKAN_MAX_BINDINGS + 1]; /* [n] = layout for n bindings */
    VkPipelineLayout pipeline_layout;
    VkPipeline matvec_pipelines[BN_VULKAN_MAX_TYPES];
    VkPipeline matvec_split_pipelines[BN_VULKAN_MAX_TYPES];
    VkPipeline fwd_pipelines[BN_GPU_SHADER_COUNT];
    VkBuffer    act_bufs[BN_GPU_VALUE_COUNT];
    VkDeviceMemory act_mems[BN_GPU_VALUE_COUNT];
    size_t act_sizes[BN_GPU_VALUE_COUNT];
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    size_t staging_cap;
    void *staging_mapped;
    VkBuffer dummy_buf;
    VkDeviceMemory dummy_mem;
    char *shader_dir;
    int kv_f16;
} BnVulkanCtx;

/* ---- Memory helpers ---- */

static uint32_t vulkan_find_memory(VkPhysicalDevice phys, uint32_t type_bits,
                                   VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static int vulkan_alloc(BnVulkanCtx *ctx, VkDeviceSize size,
                        VkBufferUsageFlags usage, VkMemoryPropertyFlags mprops,
                        VkBuffer *out_buf, VkDeviceMemory *out_mem) {
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(ctx->device, &bci, NULL, out_buf) != VK_SUCCESS)
        return -1;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, *out_buf, &req);
    uint32_t mi = vulkan_find_memory(ctx->phys_dev, req.memoryTypeBits, mprops);
    if (mi == UINT32_MAX) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return -1;
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = mi,
    };
    if (vkAllocateMemory(ctx->device, &mai, NULL, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return -1;
    }
    if (vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0) != VK_SUCCESS) {
        vkFreeMemory(ctx->device, *out_mem, NULL);
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return -1;
    }
    return 0;
}

static int vulkan_ensure_staging(BnVulkanCtx *ctx, size_t size) {
    if (size <= ctx->staging_cap) return 0;
    size_t new_cap = size + (4u * 1024u * 1024u);
    if (ctx->staging_mapped) vkUnmapMemory(ctx->device, ctx->staging_mem);
    if (ctx->staging_buf) vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
    if (ctx->staging_mem) vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
    ctx->staging_buf = VK_NULL_HANDLE;
    ctx->staging_mem = VK_NULL_HANDLE;
    ctx->staging_mapped = NULL;
    ctx->staging_cap = 0;
    if (vulkan_alloc(ctx, (VkDeviceSize)new_cap,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &ctx->staging_buf, &ctx->staging_mem) != 0)
        return -1;
    if (vkMapMemory(ctx->device, ctx->staging_mem, 0, VK_WHOLE_SIZE, 0,
                    &ctx->staging_mapped) != VK_SUCCESS)
        return -1;
    ctx->staging_cap = new_cap;
    return 0;
}

static int vulkan_upload(BnVulkanCtx *ctx, VkBuffer dst, const void *data,
                         size_t size) {
    if (vulkan_ensure_staging(ctx, size) != 0) return -1;
    memcpy(ctx->staging_mapped, data, size);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
    VkBufferCopy region = {.srcOffset=0, .dstOffset=0, .size=(VkDeviceSize)size};
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, dst, 1, &region);
    vkEndCommandBuffer(ctx->cmd_buf);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx->cmd_buf,
    };
    vkQueueSubmit(ctx->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
    VkCommandBufferResetFlags rf = 0;
    vkResetCommandBuffer(ctx->cmd_buf, rf);
    return 0;
}

/* ---- SPIR-V pipeline loading ---- */

static uint32_t *vulkan_load_spv(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz & 3)) { fclose(f); return NULL; }
    uint32_t *data = (uint32_t *)malloc((size_t)sz);
    if (!data) { fclose(f); return NULL; }
    if ((long)fread(data, 1, (size_t)sz, f) != sz) {
        fclose(f); free(data); return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return data;
}

static VkPipeline vulkan_create_pipeline(BnVulkanCtx *ctx, const char *name) {
    if (!ctx->shader_dir || !name) return VK_NULL_HANDLE;
    char path[1024];
    snprintf(path, sizeof(path), "%s/vulkan/%s.spv", ctx->shader_dir, name);
    size_t spv_size = 0;
    uint32_t *spv = vulkan_load_spv(path, &spv_size);
    if (!spv) {
        fprintf(stderr, "[bn:gpu:vulkan] missing shader: %s\n", path);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size, .pCode = spv,
    };
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult rv = vkCreateShaderModule(ctx->device, &smci, NULL, &mod);
    free(spv);
    if (rv != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = mod,
            .pName = "main",
        },
        .layout = ctx->pipeline_layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline);
    vkDestroyShaderModule(ctx->device, mod, NULL);
    return pipeline;
}

/* ---- Descriptor set helpers ---- */

/* Always allocate from the max (8-binding) layout so it matches the pipeline layout. */
static VkDescriptorSet vulkan_alloc_desc_set(BnVulkanCtx *ctx) {
    VkDescriptorSetLayout layout = ctx->set_layouts[BN_VULKAN_MAX_BINDINGS];
    if (!layout) return VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(ctx->device, &ai, &ds) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return ds;
}

/* Write n_bufs real bindings + fill remaining slots up to 8 with dummy_buf. */
static void vulkan_write_desc_bufs(BnVulkanCtx *ctx, VkDescriptorSet ds,
                                   VkBuffer *bufs, size_t *sizes, int n) {
    VkDescriptorBufferInfo info[BN_VULKAN_MAX_BINDINGS];
    VkWriteDescriptorSet writes[BN_VULKAN_MAX_BINDINGS];
    for (int i = 0; i < BN_VULKAN_MAX_BINDINGS; i++) {
        VkBuffer buf = (i < n) ? bufs[i] : ctx->dummy_buf;
        size_t sz    = (i < n) ? sizes[i] : 16;
        info[i].buffer = buf;
        info[i].offset = 0;
        info[i].range = sz ? (VkDeviceSize)sz : VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = NULL;
        writes[i].dstSet = ds;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pImageInfo = NULL;
        writes[i].pBufferInfo = &info[i];
        writes[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets(ctx->device, BN_VULKAN_MAX_BINDINGS, writes, 0, NULL);
}

/* ---- Buffer create/destroy ---- */

static void *vulkan_buffer_create(void *vctx, const void *data, size_t size,
                                  int type, int rows, int cols) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    if (!ctx || !data || size == 0) return NULL;
    BnVulkanBuffer *buf = (BnVulkanBuffer *)calloc(1, sizeof(BnVulkanBuffer));
    if (!buf) return NULL;
    if (vulkan_alloc(ctx, (VkDeviceSize)size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     &buf->buf, &buf->mem) != 0) {
        free(buf);
        return NULL;
    }
    if (vulkan_upload(ctx, buf->buf, data, size) != 0) {
        vkDestroyBuffer(ctx->device, buf->buf, NULL);
        vkFreeMemory(ctx->device, buf->mem, NULL);
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    return buf;
}

static void *vulkan_buffer_create_stacked2(void *vctx,
                                           const void *data0, size_t size0,
                                           const void *data1, size_t size1,
                                           int type, int rows, int cols) {
    if (!data0 || !data1 || size0 == 0 || size1 == 0) return NULL;
    size_t total = size0 + size1;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    void *ret = vulkan_buffer_create(vctx, combined, total, type, rows, cols);
    free(combined);
    return ret;
}

static void *vulkan_buffer_create_stacked3(void *vctx,
                                           const void *data0, size_t size0,
                                           const void *data1, size_t size1,
                                           const void *data2, size_t size2,
                                           int type, int rows, int cols) {
    if (!data0 || !data1 || !data2 ||
        size0 == 0 || size1 == 0 || size2 == 0) return NULL;
    size_t total = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);
    void *ret = vulkan_buffer_create(vctx, combined, total, type, rows, cols);
    free(combined);
    return ret;
}

static void vulkan_buffer_destroy(void *vctx, void *buffer) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    BnVulkanBuffer *buf = (BnVulkanBuffer *)buffer;
    if (!buf || !ctx) return;
    vkDestroyBuffer(ctx->device, buf->buf, NULL);
    vkFreeMemory(ctx->device, buf->mem, NULL);
    free(buf);
}

/* ---- Activation management ---- */

static void vulkan_free_activations(void *vctx) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    if (!ctx) return;
    for (int i = 0; i < BN_GPU_VALUE_COUNT; i++) {
        if (ctx->act_bufs[i]) {
            vkDestroyBuffer(ctx->device, ctx->act_bufs[i], NULL);
            ctx->act_bufs[i] = VK_NULL_HANDLE;
        }
        if (ctx->act_mems[i]) {
            vkFreeMemory(ctx->device, ctx->act_mems[i], NULL);
            ctx->act_mems[i] = VK_NULL_HANDLE;
        }
        ctx->act_sizes[i] = 0;
    }
}

static int vulkan_alloc_activation(BnVulkanCtx *ctx, int idx, size_t bytes) {
    if (!ctx || idx < 0 || idx >= BN_GPU_VALUE_COUNT) return -1;
    if (bytes == 0) return 0;
    size_t aligned = (bytes + 255u) & ~(size_t)255u;
    if (vulkan_alloc(ctx, (VkDeviceSize)aligned,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     &ctx->act_bufs[idx], &ctx->act_mems[idx]) != 0)
        return -1;
    ctx->act_sizes[idx] = aligned;
    return 0;
}

static int vulkan_init_activations(void *vctx, const void *config_ptr) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    const BnConfig *c = (const BnConfig *)config_ptr;
    if (!ctx || !c) return -1;

    vulkan_free_activations(ctx);
    ctx->kv_f16 = c->kv_f16;

    int n_attn = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval
        : c->n_layers;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;
    int hb_dim = c->hidden_dim;
    if (c->moe_intermediate_size > hb_dim) hb_dim = c->moe_intermediate_size;

    size_t sizes[BN_GPU_VALUE_COUNT] = {0};
    sizes[BN_GPU_VALUE_X]    = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_VALUE_XB]   = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_VALUE_XB2]  = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_VALUE_Q]    = (size_t)q_dim * sizeof(float);
    sizes[BN_GPU_VALUE_HB]   = (size_t)hb_dim * sizeof(float);
    sizes[BN_GPU_VALUE_HB2]  = (size_t)hb_dim * sizeof(float);
    size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    sizes[BN_GPU_VALUE_KEY_CACHE]   = (size_t)n_attn * c->seq_len * c->kv_dim * kv_elem;
    sizes[BN_GPU_VALUE_VALUE_CACHE] = (size_t)n_attn * c->seq_len * c->kv_dim * kv_elem;
    sizes[BN_GPU_VALUE_ATT]    = (size_t)c->n_heads * c->seq_len * sizeof(float);
    sizes[BN_GPU_VALUE_LOGITS] = (size_t)c->vocab_size * sizeof(float);
    sizes[BN_GPU_VALUE_ROPE_FREQ] = (size_t)(c->head_size / 2) * sizeof(float);
    sizes[BN_GPU_VALUE_SCRATCH] = (size_t)xb_size * sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * c->kv_dim) * sizeof(float);
        size_t gated_q = (size_t)(2 * q_dim) * sizeof(float);
        if (c->full_attn_interval > 0 && c->n_experts > 0) {
            size_t hq = (size_t)(4 * q_dim) * sizeof(float);
            if (hq > gated_q) gated_q = hq;
        }
        sizes[BN_GPU_VALUE_QKV] = qkv_size > gated_q ? qkv_size : gated_q;
    }
    if (c->moe_intermediate_size > 0) {
        int ms = c->moe_intermediate_size;
        if (c->n_experts > ms) ms = c->n_experts;
        if (2 * c->n_experts_active > ms) ms = 2 * c->n_experts_active;
        if (c->n_experts_active > 0 &&
            c->moe_intermediate_size <= INT_MAX / c->n_experts_active &&
            c->moe_intermediate_size * c->n_experts_active > ms)
            ms = c->moe_intermediate_size * c->n_experts_active;
        sizes[BN_GPU_VALUE_MOE_HB]  = (size_t)ms * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_HB2] = (size_t)ms * sizeof(float);
        sizes[BN_GPU_VALUE_MOE_OUT] = (size_t)c->dim * sizeof(float);
    }
    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int n_ssm = c->n_layers - n_attn;
        int nv = c->ssm_time_step_rank;
        int hk = c->ssm_state_size;
        int hv = c->ssm_inner_size / (nv > 0 ? nv : 1);
        int key_dim = c->ssm_group_count * hk;
        int val_dim = c->ssm_inner_size;
        int qkv_dim = key_dim * 2 + val_dim;
        int kern = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;
        sizes[BN_GPU_VALUE_SSM_STATE]      = (size_t)n_ssm * nv * hk * hv * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_CONV_STATE] = (size_t)n_ssm * (kern - 1) * qkv_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_QKV]        = (size_t)qkv_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_Z]          = (size_t)val_dim * sizeof(float);
        if (val_dim > c->dim) sizes[BN_GPU_VALUE_XB2] = (size_t)val_dim * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_ALPHA] = (size_t)nv * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_BETA]  = (size_t)nv * sizeof(float);
        sizes[BN_GPU_VALUE_SSM_V]     = (size_t)val_dim * sizeof(float);
    }

    for (int i = 0; i < BN_GPU_VALUE_COUNT; i++) {
        if (vulkan_alloc_activation(ctx, i, sizes[i]) != 0) {
            vulkan_free_activations(ctx);
            return -1;
        }
    }

    /* Upload RoPE frequencies */
    int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
    int half = rope_dims / 2;
    if (half > 0 && ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ]) {
        float *freq = (float *)malloc((size_t)half * sizeof(float));
        if (!freq) { vulkan_free_activations(ctx); return -1; }
        for (int i = 0; i < half; i++)
            freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
        int rv = vulkan_upload(ctx, ctx->act_bufs[BN_GPU_VALUE_ROPE_FREQ],
                               freq, (size_t)half * sizeof(float));
        free(freq);
        if (rv != 0) { vulkan_free_activations(ctx); return -1; }
    }
    return 0;
}

static int vulkan_write_activation(void *vctx, int idx, const void *data,
                                   size_t size, size_t offset) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    if (!ctx || !data || idx < 0 || idx >= BN_GPU_VALUE_COUNT) return -1;
    if (!ctx->act_bufs[idx] || offset + size > ctx->act_sizes[idx]) return -1;
    if (vulkan_ensure_staging(ctx, size) != 0) return -1;
    memcpy(ctx->staging_mapped, data, size);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
    VkBufferCopy region = {.srcOffset=0, .dstOffset=(VkDeviceSize)offset,
                           .size=(VkDeviceSize)size};
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->staging_buf, ctx->act_bufs[idx], 1, &region);
    vkEndCommandBuffer(ctx->cmd_buf);
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount=1, .pCommandBuffers=&ctx->cmd_buf};
    vkQueueSubmit(ctx->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
    VkCommandBufferResetFlags rf = 0;
    vkResetCommandBuffer(ctx->cmd_buf, rf);
    return 0;
}

static int vulkan_read_activation(void *vctx, int idx, void *out,
                                  size_t size, size_t offset) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    if (!ctx || !out || idx < 0 || idx >= BN_GPU_VALUE_COUNT) return -1;
    if (!ctx->act_bufs[idx] || offset + size > ctx->act_sizes[idx]) return -1;
    /* Need a host-visible readback buffer */
    VkBuffer rb_buf = VK_NULL_HANDLE;
    VkDeviceMemory rb_mem = VK_NULL_HANDLE;
    if (vulkan_alloc(ctx, (VkDeviceSize)size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &rb_buf, &rb_mem) != 0)
        return -1;
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
    VkBufferCopy region = {.srcOffset=(VkDeviceSize)offset, .dstOffset=0,
                           .size=(VkDeviceSize)size};
    vkCmdCopyBuffer(ctx->cmd_buf, ctx->act_bufs[idx], rb_buf, 1, &region);
    vkEndCommandBuffer(ctx->cmd_buf);
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount=1, .pCommandBuffers=&ctx->cmd_buf};
    vkQueueSubmit(ctx->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
    VkCommandBufferResetFlags rf = 0;
    vkResetCommandBuffer(ctx->cmd_buf, rf);
    void *mapped = NULL;
    vkMapMemory(ctx->device, rb_mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (mapped) memcpy(out, mapped, size);
    vkUnmapMemory(ctx->device, rb_mem);
    vkDestroyBuffer(ctx->device, rb_buf, NULL);
    vkFreeMemory(ctx->device, rb_mem, NULL);
    return mapped ? 0 : -1;
}

static int vulkan_memory_info(void *vctx, size_t *free_bytes, size_t *total_bytes) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    if (!ctx || !free_bytes || !total_bytes) return -1;
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys_dev, &props);
    size_t total = 0;
    for (uint32_t i = 0; i < props.memoryHeapCount; i++) {
        if (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += props.memoryHeaps[i].size;
    }
    *total_bytes = total;
    *free_bytes = total / 2; /* estimate — VK_EXT_memory_budget would be accurate */
    return 0;
}

/* ---- Execute ---- */

#define VULKAN_ACT(idx) (ctx->act_bufs[(idx)])
#define VULKAN_ACTSZ(idx) (ctx->act_sizes[(idx)])

static VkPipeline vulkan_pick_matvec(BnVulkanCtx *ctx, int type) {
    if (type >= 0 && type < BN_VULKAN_MAX_TYPES)
        return ctx->matvec_pipelines[type];
    return VK_NULL_HANDLE;
}

static VkPipeline vulkan_pick_matvec_split(BnVulkanCtx *ctx, int type) {
    if (type >= 0 && type < BN_VULKAN_MAX_TYPES)
        return ctx->matvec_split_pipelines[type];
    return VK_NULL_HANDLE;
}

static int vulkan_execute(void *vctx, const void *ops_raw, int n_ops,
                          int readback_buf, float *out_host, int out_len) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)vctx;
    const BnGPUOp *ops = (const BnGPUOp *)ops_raw;
    if (!ctx || !ops || n_ops <= 0) return -1;

    /* Reset descriptor pool for this execute call */
    vkResetDescriptorPool(ctx->device, ctx->desc_pool, 0);

    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);

    for (int i = 0; i < n_ops; i++) {
        const BnGPUOp *op = &ops[i];
        int shader = bn_gpu_shader_from_op_code(op->op_code);
        if (shader < 0) continue;

        VkPipeline pipeline = VK_NULL_HANDLE;
        VkBuffer bufs[BN_VULKAN_MAX_BINDINGS];
        size_t bsizes[BN_VULKAN_MAX_BINDINGS];
        int n_bufs = 0;
        uint32_t wg_x = 1, wg_y = 1;

        BnVulkanBuffer *wbuf  = (BnVulkanBuffer *)op->W_buf;
        BnVulkanBuffer *wbuf2 = (BnVulkanBuffer *)op->W_buf2;
        BnVulkanBuffer *wbuf3 = (BnVulkanBuffer *)op->W_buf3;
        (void)wbuf2; (void)wbuf3;

#define PUSH_BUF(b, s) do { bufs[n_bufs]=(b); bsizes[n_bufs]=(s); n_bufs++; } while(0)
#define PUSH_ACT(idx)  PUSH_BUF(VULKAN_ACT(idx), VULKAN_ACTSZ(idx))
#define PUSH_W()       do { if(!wbuf) goto skip_op; PUSH_BUF(wbuf->buf, wbuf->size); } while(0)

        switch (shader) {
        case BN_GPU_SHADER_MATVEC: {
            pipeline = vulkan_pick_matvec(ctx, op->type);
            if (!pipeline) goto skip_op;
            PUSH_W(); PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_out);
            uint32_t tiled = ((uint32_t)op->rows + 31) / 32;
            if (op->p[3] > 0) {
                wg_x = op->p[3];
                wg_y = (tiled + op->p[3] - 1) / op->p[3];
            } else {
                wg_x = tiled;
                wg_y = op->p[2] ? op->p[2] : 1;
            }
            break;
        }
        case BN_GPU_SHADER_RMSNORM: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in);
            if (wbuf) PUSH_BUF(wbuf->buf, wbuf->size);
            else PUSH_ACT(op->buf_in);
            PUSH_ACT(op->buf_out);
            wg_x = 1;
            break;
        }
        case BN_GPU_SHADER_ROPE: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(BN_GPU_VALUE_ROPE_FREQ);
            wg_x = op->p[0];
            break;
        }
        case BN_GPU_SHADER_ROPE_QK: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            PUSH_ACT(BN_GPU_VALUE_ROPE_FREQ);
            wg_x = op->p[0] + op->p[4];
            break;
        }
        case BN_GPU_SHADER_GQA_SCORES: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(BN_GPU_VALUE_KEY_CACHE);
            PUSH_ACT(BN_GPU_VALUE_ATT);
            wg_x = op->p[0];
            break;
        }
        case BN_GPU_SHADER_SOFTMAX: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(BN_GPU_VALUE_ATT);
            wg_x = op->p[0];
            break;
        }
        case BN_GPU_SHADER_GQA_COMBINE: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(BN_GPU_VALUE_ATT); PUSH_ACT(BN_GPU_VALUE_VALUE_CACHE);
            PUSH_ACT(op->buf_out);
            wg_x = op->p[0];
            break;
        }
        case BN_GPU_SHADER_SILU_GATE:
        case BN_GPU_SHADER_RELU2_GATE:
        case BN_GPU_SHADER_SIGMOID_GATE: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_SILU_ACT:
        case BN_GPU_SHADER_RELU2_ACT: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_RESIDUAL_ADD:
        case BN_GPU_SHADER_WEIGHTED_ADD: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_BIAS_ADD: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_BUF(wbuf->buf, wbuf->size);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_RESIDUAL_RMSNORM: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_ACT(op->buf_out);
            wg_x = 1;
            break;
        }
        case BN_GPU_SHADER_COPY: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_out);
            wg_x = (op->p[2] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_PER_HEAD_RMSNORM: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_BUF(wbuf->buf, wbuf->size);
            wg_x = (uint32_t)op->rows;
            break;
        }
        case BN_GPU_SHADER_DEINTERLEAVE_Q: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_out);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_SSM_CONV_SILU: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(BN_GPU_VALUE_SSM_CONV_STATE);
            PUSH_BUF(wbuf->buf, wbuf->size);
            wg_x = (op->p[0] + 255u) / 256u;
            break;
        }
        case BN_GPU_SHADER_SSM_L2NORM: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            wg_x = (uint32_t)op->rows;
            break;
        }
        case BN_GPU_SHADER_SSM_ALPHA_BETA: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
            BnVulkanBuffer *a_wbuf = (BnVulkanBuffer *)a_ptr;
            if (!a_wbuf) goto skip_op;
            PUSH_ACT(BN_GPU_VALUE_SSM_ALPHA); PUSH_ACT(BN_GPU_VALUE_SSM_BETA);
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_BUF(a_wbuf->buf, a_wbuf->size);
            wg_x = 1;
            break;
        }
        case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
            BnVulkanBuffer *a_wbuf = (BnVulkanBuffer *)a_ptr;
            if (!a_wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(BN_GPU_VALUE_SSM_ALPHA);
            PUSH_ACT(BN_GPU_VALUE_SSM_BETA); PUSH_BUF(wbuf->buf, wbuf->size);
            PUSH_BUF(a_wbuf->buf, a_wbuf->size);
            wg_x = 1;
            break;
        }
        case BN_GPU_SHADER_SSM_DELTA: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline) goto skip_op;
            int v_buf = op->p[7] ? op->buf_in : BN_GPU_VALUE_SSM_V;
            PUSH_ACT(BN_GPU_VALUE_SSM_STATE); PUSH_ACT(op->buf_out);
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            PUSH_ACT(v_buf); PUSH_ACT(BN_GPU_VALUE_SSM_ALPHA);
            PUSH_ACT(BN_GPU_VALUE_SSM_BETA);
            wg_x = (uint32_t)op->rows;
            break;
        }
        case BN_GPU_SHADER_SSM_GATE: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_ACT(op->buf_in); PUSH_ACT(op->buf_aux);
            PUSH_BUF(wbuf->buf, wbuf->size);
            wg_x = (uint32_t)op->rows;
            break;
        }
        case BN_GPU_SHADER_FUSED_GATEUP_SILU: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_ACT(op->buf_in);
            PUSH_ACT(op->buf_out);
            wg_x = (op->p[2] + 31u) / 32u;
            break;
        }
        case BN_GPU_SHADER_MATVEC_SPLIT: {
            pipeline = vulkan_pick_matvec_split(ctx, op->type);
            if (!pipeline || !wbuf) goto skip_op;
            int out2_idx = (op->rows >= 0 && op->rows < BN_GPU_BUF_COUNT)
                         ? op->rows : op->buf_aux;
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_ACT(op->buf_in);
            PUSH_ACT(op->buf_out); PUSH_ACT(op->buf_aux); PUSH_ACT(out2_idx);
            wg_x = (op->p[0] + 31u) / 32u;
            break;
        }
        case BN_GPU_SHADER_Q4K_MATVEC_SPLIT: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_ACT(op->buf_in);
            PUSH_ACT(op->buf_out); PUSH_ACT(op->buf_aux);
            uint32_t split_rows = op->p[3] != 0 ? op->p[2] : op->p[0];
            wg_x = (split_rows + 31u) / 32u;
            break;
        }
        case BN_GPU_SHADER_Q8_MATVEC_SPLIT:
        case BN_GPU_SHADER_Q5K_MATVEC_SPLIT: {
            pipeline = ctx->fwd_pipelines[shader];
            if (!pipeline || !wbuf) goto skip_op;
            int out2_idx = (op->rows >= 0 && op->rows < BN_GPU_BUF_COUNT)
                         ? op->rows : op->buf_aux;
            PUSH_BUF(wbuf->buf, wbuf->size); PUSH_ACT(op->buf_in);
            PUSH_ACT(op->buf_out); PUSH_ACT(op->buf_aux); PUSH_ACT(out2_idx);
            wg_x = (op->p[0] + 31u) / 32u;
            break;
        }
        default:
            goto skip_op;
        }

#undef PUSH_BUF
#undef PUSH_ACT
#undef PUSH_W

        if (!pipeline || n_bufs == 0) goto skip_op;

        /* Validate: no NULL buffers */
        {
            int valid = 1;
            for (int b = 0; b < n_bufs; b++)
                if (!bufs[b]) { valid = 0; break; }
            if (!valid) goto skip_op;
        }
        if (wg_x == 0) goto skip_op;

        /* Allocate + write descriptor set */
        VkDescriptorSet ds = vulkan_alloc_desc_set(ctx);
        if (!ds) goto skip_op;
        vulkan_write_desc_bufs(ctx, ds, bufs, bsizes, n_bufs);

        /* Record commands */
        vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pipeline_layout, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(op->p), op->p);
        vkCmdDispatch(ctx->cmd_buf, wg_x, wg_y, 1);

        /* Memory barrier between dispatches */
        VkMemoryBarrier mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(ctx->cmd_buf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
        continue;
skip_op:;
    }

    /* Readback: copy output activation to staging */
    if (readback_buf >= 0 && readback_buf < BN_GPU_BUF_COUNT &&
        ctx->act_bufs[readback_buf] && out_host && out_len > 0) {
        size_t rb_size = (size_t)out_len * sizeof(float);
        if (rb_size <= ctx->act_sizes[readback_buf]) {
            /* Transition for transfer */
            VkMemoryBarrier mb2 = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            };
            vkCmdPipelineBarrier(ctx->cmd_buf,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 1, &mb2, 0, NULL, 0, NULL);
        }
    }

    vkEndCommandBuffer(ctx->cmd_buf);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx->cmd_buf,
    };
    vkQueueSubmit(ctx->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
    VkCommandBufferResetFlags rf = 0;
    vkResetCommandBuffer(ctx->cmd_buf, rf);

    /* Readback */
    if (readback_buf >= 0 && readback_buf < BN_GPU_BUF_COUNT &&
        ctx->act_bufs[readback_buf] && out_host && out_len > 0) {
        size_t rb_size = (size_t)out_len * sizeof(float);
        if (rb_size <= ctx->act_sizes[readback_buf]) {
            vulkan_read_activation(vctx, readback_buf, out_host, rb_size, 0);
        }
    }
    return 0;
}

/* ---- Create/Destroy ---- */

static void vulkan_ctx_destroy(BnVulkanCtx *ctx) {
    if (!ctx) return;
    vkDeviceWaitIdle(ctx->device);
    vulkan_free_activations(ctx);
    for (int i = 0; i < BN_GPU_SHADER_COUNT; i++) {
        if (ctx->fwd_pipelines[i])
            vkDestroyPipeline(ctx->device, ctx->fwd_pipelines[i], NULL);
    }
    for (int i = 0; i < BN_VULKAN_MAX_TYPES; i++) {
        if (ctx->matvec_pipelines[i])
            vkDestroyPipeline(ctx->device, ctx->matvec_pipelines[i], NULL);
        if (ctx->matvec_split_pipelines[i])
            vkDestroyPipeline(ctx->device, ctx->matvec_split_pipelines[i], NULL);
    }
    if (ctx->pipeline_layout)
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    for (int i = 1; i <= BN_VULKAN_MAX_BINDINGS; i++) {
        if (ctx->set_layouts[i])
            vkDestroyDescriptorSetLayout(ctx->device, ctx->set_layouts[i], NULL);
    }
    if (ctx->desc_pool)
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    if (ctx->cmd_pool)
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    if (ctx->dummy_buf) vkDestroyBuffer(ctx->device, ctx->dummy_buf, NULL);
    if (ctx->dummy_mem) vkFreeMemory(ctx->device, ctx->dummy_mem, NULL);
    if (ctx->staging_mapped) vkUnmapMemory(ctx->device, ctx->staging_mem);
    if (ctx->staging_buf) vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
    if (ctx->staging_mem) vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
    if (ctx->device) vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
    free(ctx->shader_dir);
    free(ctx);
}

BnGPUBackend *bn_gpu_vulkan_create(const char *shader_dir) {
    BnVulkanCtx *ctx = (BnVulkanCtx *)calloc(1, sizeof(BnVulkanCtx));
    BnGPUBackend *gpu = (BnGPUBackend *)calloc(1, sizeof(BnGPUBackend));
    if (!ctx || !gpu) { free(ctx); free(gpu); return NULL; }
    if (shader_dir) ctx->shader_dir = strdup(shader_dir);

    /* Instance */
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "bitnet.c",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &ctx->instance) != VK_SUCCESS) {
        fprintf(stderr, "[bn:gpu:vulkan] vkCreateInstance failed\n");
        goto fail;
    }

    /* Physical device */
    uint32_t n_phys = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &n_phys, NULL);
    if (n_phys == 0) {
        fprintf(stderr, "[bn:gpu:vulkan] no physical device\n");
        goto fail;
    }
    VkPhysicalDevice *phys_devs = (VkPhysicalDevice *)malloc(n_phys * sizeof(VkPhysicalDevice));
    if (!phys_devs) goto fail;
    vkEnumeratePhysicalDevices(ctx->instance, &n_phys, phys_devs);

    /* Pick device with most VRAM */
    ctx->phys_dev = phys_devs[0];
    size_t best_vram = 0;
    for (uint32_t d = 0; d < n_phys; d++) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_devs[d], &mp);
        size_t vram = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram += mp.memoryHeaps[h].size;
        if (vram > best_vram) { best_vram = vram; ctx->phys_dev = phys_devs[d]; }
    }
    free(phys_devs);

    VkPhysicalDeviceProperties phys_props;
    vkGetPhysicalDeviceProperties(ctx->phys_dev, &phys_props);
    fprintf(stderr, "[bn:gpu:vulkan] device: %s\n", phys_props.deviceName);

    /* Find compute queue family */
    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &n_qf, NULL);
    VkQueueFamilyProperties *qfp = (VkQueueFamilyProperties *)malloc(n_qf * sizeof(VkQueueFamilyProperties));
    if (!qfp) goto fail;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &n_qf, qfp);
    ctx->queue_family = UINT32_MAX;
    for (uint32_t q = 0; q < n_qf; q++) {
        if (qfp[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->queue_family = q; break;
        }
    }
    free(qfp);
    if (ctx->queue_family == UINT32_MAX) {
        fprintf(stderr, "[bn:gpu:vulkan] no compute queue\n");
        goto fail;
    }

    /* Logical device */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = 1, .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &dqci,
    };
    if (vkCreateDevice(ctx->phys_dev, &dci, NULL, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "[bn:gpu:vulkan] vkCreateDevice failed\n");
        goto fail;
    }
    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

    /* Command pool + buffer */
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool) != VK_SUCCESS)
        goto fail;
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &ctx->cmd_buf) != VK_SUCCESS)
        goto fail;

    /* Staging buffer */
    size_t staging_bytes = (size_t)BN_VULKAN_STAGING_MB * 1024u * 1024u;
    if (vulkan_alloc(ctx, (VkDeviceSize)staging_bytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &ctx->staging_buf, &ctx->staging_mem) != 0)
        goto fail;
    vkMapMemory(ctx->device, ctx->staging_mem, 0, VK_WHOLE_SIZE, 0, &ctx->staging_mapped);
    ctx->staging_cap = staging_bytes;

    /* Dummy buffer (16 bytes) for unused descriptor slots */
    if (vulkan_alloc(ctx, 16,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     &ctx->dummy_buf, &ctx->dummy_mem) != 0)
        goto fail;

    /* Descriptor set layouts: set_layouts[n] = layout with n storage buffer bindings */
    for (int n = 1; n <= BN_VULKAN_MAX_BINDINGS; n++) {
        VkDescriptorSetLayoutBinding bindings[BN_VULKAN_MAX_BINDINGS];
        for (int b = 0; b < n; b++) {
            bindings[b].binding = (uint32_t)b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[b].pImmutableSamplers = NULL;
        }
        VkDescriptorSetLayoutCreateInfo dlci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = (uint32_t)n, .pBindings = bindings,
        };
        if (vkCreateDescriptorSetLayout(ctx->device, &dlci, NULL,
                                        &ctx->set_layouts[n]) != VK_SUCCESS)
            goto fail;
    }

    /* Descriptor pool — use layout[8] as the largest, allocate max sets */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = (uint32_t)(BN_VULKAN_MAX_SETS * BN_VULKAN_MAX_BINDINGS),
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = BN_VULKAN_MAX_SETS,
        .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool) != VK_SUCCESS)
        goto fail;

    /* Pipeline layout: one descriptor set + push constants (32 bytes) */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = 32,
    };
    /* We need separate pipeline layouts per set layout — but since all ops use
     * different binding counts and different descriptor set layouts, we must pick
     * one layout per pipeline. We use the max (8-binding) set layout for the
     * pipeline layout; the actual sets allocated use the exact count layout. */
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &ctx->set_layouts[BN_VULKAN_MAX_BINDINGS],
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &ctx->pipeline_layout) != VK_SUCCESS)
        goto fail;

    /* Load shaders and create pipelines */
    static const struct { int type; const char *name; } matvec_shaders[] = {
        { BN_GGUF_TENSOR_F32,  "f32_matvec" },
        { BN_GGUF_TENSOR_Q4_0, "q4_matvec"  },
        { BN_GGUF_TENSOR_Q8_0, "q8_matvec"  },
        { BN_GGUF_TENSOR_Q4_K, "q4k_matvec" },
        { BN_GGUF_TENSOR_Q5_K, "q5k_matvec" },
        { BN_GGUF_TENSOR_Q6_K, "q6k_matvec" },
    };
    for (size_t s = 0; s < sizeof(matvec_shaders)/sizeof(matvec_shaders[0]); s++) {
        int t = matvec_shaders[s].type;
        if (t >= 0 && t < BN_VULKAN_MAX_TYPES)
            ctx->matvec_pipelines[t] = vulkan_create_pipeline(ctx, matvec_shaders[s].name);
    }

    static const struct { int type; const char *name; } split_shaders[] = {
        { BN_GGUF_TENSOR_Q4_0, "q4_matvec_split"  },
        { BN_GGUF_TENSOR_Q8_0, "q8_matvec_split"  },
        { BN_GGUF_TENSOR_Q4_K, "q4k_matvec_split" },
        { BN_GGUF_TENSOR_Q5_K, "q5k_matvec_split" },
    };
    for (size_t s = 0; s < sizeof(split_shaders)/sizeof(split_shaders[0]); s++) {
        int t = split_shaders[s].type;
        if (t >= 0 && t < BN_VULKAN_MAX_TYPES)
            ctx->matvec_split_pipelines[t] = vulkan_create_pipeline(ctx, split_shaders[s].name);
    }

    static const struct { int shader; const char *name; } fwd_shaders[] = {
        { BN_GPU_SHADER_RMSNORM,           "rmsnorm"           },
        { BN_GPU_SHADER_ROPE,              "rope"              },
        { BN_GPU_SHADER_ROPE_QK,           "rope_qk"           },
        { BN_GPU_SHADER_GQA_SCORES,        "gqa_scores"        },
        { BN_GPU_SHADER_SOFTMAX,           "softmax"           },
        { BN_GPU_SHADER_GQA_COMBINE,       "gqa_combine"       },
        { BN_GPU_SHADER_SILU_GATE,         "silu_gate"         },
        { BN_GPU_SHADER_RELU2_GATE,        "relu2_gate"        },
        { BN_GPU_SHADER_SIGMOID_GATE,      "sigmoid_gate"      },
        { BN_GPU_SHADER_SILU_ACT,          "silu_act"          },
        { BN_GPU_SHADER_RELU2_ACT,         "relu2_act"         },
        { BN_GPU_SHADER_RESIDUAL_ADD,      "residual_add"      },
        { BN_GPU_SHADER_WEIGHTED_ADD,      "weighted_add"      },
        { BN_GPU_SHADER_BIAS_ADD,          "bias_add"          },
        { BN_GPU_SHADER_RESIDUAL_RMSNORM,  "residual_rmsnorm"  },
        { BN_GPU_SHADER_COPY,              "buf_copy"          },
        { BN_GPU_SHADER_PER_HEAD_RMSNORM,  "per_head_rmsnorm"  },
        { BN_GPU_SHADER_DEINTERLEAVE_Q,    "deinterleave_q"    },
        { BN_GPU_SHADER_SSM_CONV_SILU,     "ssm_conv_silu"     },
        { BN_GPU_SHADER_SSM_L2NORM,        "ssm_l2norm"        },
        { BN_GPU_SHADER_SSM_ALPHA_BETA,    "ssm_alpha_beta"    },
        { BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT, "ssm_alpha_beta_split" },
        { BN_GPU_SHADER_SSM_DELTA,         "ssm_delta"         },
        { BN_GPU_SHADER_SSM_GATE,          "ssm_gate"          },
        { BN_GPU_SHADER_FUSED_GATEUP_SILU, "q4_fused_gateup_silu" },
        { BN_GPU_SHADER_Q4K_MATVEC_SPLIT,  "q4k_matvec_split"  },
        { BN_GPU_SHADER_Q8_MATVEC_SPLIT,   "q8_matvec_split"   },
        { BN_GPU_SHADER_Q5K_MATVEC_SPLIT,  "q5k_matvec_split"  },
    };
    for (size_t s = 0; s < sizeof(fwd_shaders)/sizeof(fwd_shaders[0]); s++) {
        int sh = fwd_shaders[s].shader;
        if (sh >= 0 && sh < BN_GPU_SHADER_COUNT)
            ctx->fwd_pipelines[sh] = vulkan_create_pipeline(ctx, fwd_shaders[s].name);
    }

    /* Populate vtable */
    gpu->buffer_create         = vulkan_buffer_create;
    gpu->buffer_create_stacked2 = vulkan_buffer_create_stacked2;
    gpu->buffer_create_stacked3 = vulkan_buffer_create_stacked3;
    gpu->buffer_destroy        = vulkan_buffer_destroy;
    gpu->init_activations      = vulkan_init_activations;
    gpu->free_activations      = vulkan_free_activations;
    gpu->write_activation      = vulkan_write_activation;
    gpu->read_activation       = vulkan_read_activation;
    gpu->memory_info           = vulkan_memory_info;
    gpu->execute               = vulkan_execute;
    gpu->ctx                   = ctx;
    gpu->kind                  = BN_GPU_BACKEND_VULKAN;
    gpu->max_storage_binding_size = (size_t)-1;
    gpu->caps                  = BN_GPU_CAP_Q4_MATVEC_SPLIT   |
                                 BN_GPU_CAP_Q4K_MATVEC_SPLIT   |
                                 BN_GPU_CAP_Q8_MATVEC_SPLIT    |
                                 BN_GPU_CAP_Q5K_MATVEC_SPLIT   |
                                 BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
    return gpu;

fail:
    vulkan_ctx_destroy(ctx);
    free(gpu);
    return NULL;
}

void bn_gpu_vulkan_destroy(BnGPUBackend *gpu) {
    if (!gpu) return;
    vulkan_ctx_destroy((BnVulkanCtx *)gpu->ctx);
    free(gpu);
}

#endif /* BN_ENABLE_VULKAN */
