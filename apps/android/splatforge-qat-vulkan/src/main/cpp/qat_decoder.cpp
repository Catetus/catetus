//
// qat_decoder.cpp — Vulkan compute dispatch for the QAT-PLY v1 wire
// format. Owns a singleton VkInstance / VkDevice / VkQueue, compiles
// the SPIR-V module emitted at build time (see CMakeLists.txt), creates
// two pipelines (one specialization per DTYPE), and provides a
// blocking decode() API exposed via JNI.
//
// The Java/Kotlin side passes opaque arrays; this file copies them
// into staging buffers, dispatches the kernel, and reads results back.
// All allocations use host-visible coherent memory because each call
// is one-shot — production renderers should keep a persistent
// allocator (e.g. VMA) instead.
//
// SPDX-License-Identifier: MIT
//

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <string>

#include "qat_decoder.h"

namespace splatforge {

namespace {

// Compiled SPIR-V is emitted as a header by CMake via add_custom_command.
#include "qat_dequant_spv.h"  // declares: const uint32_t kQatDequantSPV[]; const size_t kQatDequantSPVSize;

#define VK_CHECK(expr) do {                                              \
    VkResult _r = (expr);                                                \
    if (_r != VK_SUCCESS) {                                              \
        throw std::runtime_error(std::string(#expr " failed: ") +        \
                                 std::to_string((int)_r));               \
    }                                                                    \
} while (0)

struct VkContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipe_int8 = VK_NULL_HANDLE;
    VkPipeline pipe_int4 = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
};

static std::once_flag g_init_flag;
static VkContext g_ctx;
static std::string g_init_error;

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & wanted) == wanted) {
            return i;
        }
    }
    throw std::runtime_error("no compatible memory type");
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceSize size = 0;

    void destroy(VkDevice d) {
        if (mapped) vkUnmapMemory(d, mem);
        if (buf) vkDestroyBuffer(d, buf, nullptr);
        if (mem) vkFreeMemory(d, mem, nullptr);
        *this = {};
    }
};

Buffer make_buffer(VkContext &c, VkDeviceSize size) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device, &bi, nullptr, &b.buf));

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(c.device, b.buf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory_type(c.phys, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(c.device, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(c.device, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(c.device, b.mem, 0, size, 0, &b.mapped));
    return b;
}

void init_context() {
    VkContext &c = g_ctx;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "splatforge_qat";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.instance));

    uint32_t n_phys = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &n_phys, nullptr));
    if (!n_phys) throw std::runtime_error("no vulkan physical device");
    std::vector<VkPhysicalDevice> devs(n_phys);
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &n_phys, devs.data()));
    c.phys = devs[0];

    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n_qf, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(n_qf);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n_qf, qfp.data());
    c.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n_qf; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.queue_family = i; break; }
    }
    if (c.queue_family == UINT32_MAX) throw std::runtime_error("no compute queue family");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(c.phys, &dci, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.queue_family, 0, &c.queue);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = c.queue_family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(c.device, &pci, nullptr, &c.pool));

    // Descriptor set layout: three storage buffers (Q, scale, out).
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{};
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = 3;
    dsli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsli, nullptr, &c.dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 2 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.pipe_layout));

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = kQatDequantSPVSize;
    smi.pCode = kQatDequantSPV;
    VK_CHECK(vkCreateShaderModule(c.device, &smi, nullptr, &c.shader));

    auto make_pipe = [&](uint32_t dtype) -> VkPipeline {
        VkSpecializationMapEntry me{};
        me.constantID = 0;
        me.offset = 0;
        me.size = sizeof(uint32_t);
        VkSpecializationInfo si{};
        si.mapEntryCount = 1;
        si.pMapEntries = &me;
        si.dataSize = sizeof(uint32_t);
        si.pData = &dtype;
        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = c.shader;
        ssci.pName = "main";
        ssci.pSpecializationInfo = &si;
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = ssci;
        cpci.layout = c.pipe_layout;
        VkPipeline p = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
        return p;
    };
    c.pipe_int8 = make_pipe(0);
    c.pipe_int4 = make_pipe(1);

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 64;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 16;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &c.desc_pool));
}

VkContext &ctx() {
    std::call_once(g_init_flag, []{
        try { init_context(); }
        catch (const std::exception &e) { g_init_error = e.what(); }
    });
    if (!g_init_error.empty()) throw std::runtime_error(g_init_error);
    return g_ctx;
}

void run_dispatch(VkPipeline pipe,
                  const void *q_bytes, size_t q_size,
                  const float *scale, size_t n_scale,
                  uint32_t n_rows, uint32_t n_channels,
                  float *out_floats) {
    auto &c = ctx();

    // Pad q buffer to 4-byte alignment as the shader reads uints.
    size_t q_padded = (q_size + 3u) & ~size_t(3u);

    Buffer qbuf = make_buffer(c, q_padded);
    std::memset(qbuf.mapped, 0, q_padded);
    std::memcpy(qbuf.mapped, q_bytes, q_size);

    Buffer sbuf = make_buffer(c, n_scale * sizeof(float));
    std::memcpy(sbuf.mapped, scale, n_scale * sizeof(float));

    size_t out_size = (size_t)n_rows * n_channels * sizeof(float);
    Buffer obuf = make_buffer(c, out_size);
    std::memset(obuf.mapped, 0, out_size);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = c.desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &c.dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(c.device, &dsai, &dset));

    VkDescriptorBufferInfo dbi[3]{};
    dbi[0].buffer = qbuf.buf; dbi[0].offset = 0; dbi[0].range = qbuf.size;
    dbi[1].buffer = sbuf.buf; dbi[1].offset = 0; dbi[1].range = sbuf.size;
    dbi[2].buffer = obuf.buf; dbi[2].offset = 0; dbi[2].range = obuf.size;
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c.device, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = c.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbai, &cmd));

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            c.pipe_layout, 0, 1, &dset, 0, nullptr);
    uint32_t pc_data[2] = { n_rows, n_channels };
    vkCmdPushConstants(cmd, c.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc_data), pc_data);
    uint32_t gx = (n_rows + 15u) / 16u;
    uint32_t gy = (n_channels + 15u) / 16u;
    vkCmdDispatch(cmd, gx, gy, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));

    std::memcpy(out_floats, obuf.mapped, out_size);

    vkFreeDescriptorSets(c.device, c.desc_pool, 1, &dset);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    qbuf.destroy(c.device);
    sbuf.destroy(c.device);
    obuf.destroy(c.device);
}

}  // namespace

void decode_int8(const int8_t *q, const float *scale,
                 uint32_t n_rows, uint32_t n_channels, float *out) {
    run_dispatch(ctx().pipe_int8,
                 (const void *)q, (size_t)n_rows * n_channels,
                 scale, n_channels, n_rows, n_channels, out);
}

void decode_int4_packed(const uint8_t *packed, const float *scale,
                        uint32_t n_rows, uint32_t n_channels, float *out) {
    uint32_t B = (n_channels + 1u) >> 1u;
    run_dispatch(ctx().pipe_int4,
                 (const void *)packed, (size_t)n_rows * B,
                 scale, n_rows, n_rows, n_channels, out);
}

}  // namespace splatforge
