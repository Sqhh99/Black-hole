// ---------------------------------------------------------------------------
// vulkan_context.cpp
// ---------------------------------------------------------------------------
#include "vulkan_context.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
#define VK_CHECK(call)                                                        \
    do {                                                                      \
        VkResult _r = (call);                                                 \
        if (_r != VK_SUCCESS) {                                               \
            LOG_ERROR("Vulkan error %d at %s:%d -> %s", (int)_r, __FILE__,    \
                      __LINE__, #call);                                       \
            throw std::runtime_error("Vulkan call failed: " #call);           \
        }                                                                     \
    } while (0)

namespace
{
const char* kDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
};

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        LOG_WARN("[VK-VALIDATION] %s", data->pMessage);
    return VK_FALSE;
}
} // namespace

// ---------------------------------------------------------------------------
void VulkanContext::init(uint32_t width, uint32_t height, const char* title)
{
    m_width = width;
    m_height = height;

#ifdef _DEBUG
    m_validation = true;
#endif
    if (const char* env = std::getenv("BLACKHOLE_VALIDATION"))
        m_validation = (std::strcmp(env, "1") == 0);

    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");
    if (!glfwVulkanSupported())
        throw std::runtime_error("GLFW reports no Vulkan support (is the Vulkan runtime installed?)");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // fixed 1280x720 interop buffer
    m_window = glfwCreateWindow((int)width, (int)height, title, nullptr, nullptr);
    if (!m_window)
        throw std::runtime_error("glfwCreateWindow failed");

    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createInteropBuffer();
    createInteropSemaphores();
    createCommandBuffers();
    createSyncObjects();

    LOG_INFO("Vulkan context initialized (%ux%u, format %d, %s)",
             width, height, (int)m_swapFormat, m_swapRB ? "BGRA" : "RGBA");
}

// ---------------------------------------------------------------------------
void VulkanContext::createInstance()
{
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName   = "BlackHoleCUDAVulkan";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName        = "none";
    app.apiVersion         = VK_API_VERSION_1_1; // external memory/semaphore are core

    uint32_t glfwCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> exts(glfwExts, glfwExts + glfwCount);

    std::vector<const char*> layers;
    if (m_validation)
    {
        uint32_t n = 0;
        vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> avail(n);
        vkEnumerateInstanceLayerProperties(&n, avail.data());
        for (auto& l : avail)
            if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                break;
            }
        if (layers.empty())
            LOG_WARN("Validation requested but VK_LAYER_KHRONOS_validation not found");
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    ci.enabledLayerCount       = (uint32_t)layers.size();
    ci.ppEnabledLayerNames     = layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));

    if (!layers.empty())
    {
        auto pfn = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
        if (pfn)
        {
            VkDebugUtilsMessengerCreateInfoEXT di{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            di.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            di.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            di.pfnUserCallback = debugCallback;
            pfn(m_instance, &di, nullptr, &m_debugMessenger);
            LOG_INFO("Vulkan validation layer enabled");
        }
    }
}

// ---------------------------------------------------------------------------
void VulkanContext::createSurface()
{
    VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
}

// ---------------------------------------------------------------------------
void VulkanContext::pickPhysicalDevice()
{
    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &n, nullptr));
    if (n == 0) throw std::runtime_error("No Vulkan physical devices found");
    std::vector<VkPhysicalDevice> devices(n);
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &n, devices.data()));

    auto supportsExtensions = [](VkPhysicalDevice d) {
        uint32_t cnt = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &cnt, nullptr);
        std::vector<VkExtensionProperties> props(cnt);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &cnt, props.data());
        std::set<std::string> required(std::begin(kDeviceExtensions), std::end(kDeviceExtensions));
        for (auto& p : props) required.erase(p.extensionName);
        return required.empty();
    };

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = -1;
    uint32_t bestFamily = 0;

    for (auto d : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) continue;
        if (!supportsExtensions(d)) continue;

        // graphics + present queue family
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qprops.data());
        int family = -1;
        for (uint32_t i = 0; i < qn; ++i)
        {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, m_surface, &present);
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present)
            {
                family = (int)i;
                break;
            }
        }
        if (family < 0) continue;

        int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1000 : 10;
        if (score > bestScore)
        {
            bestScore = score;
            best = d;
            bestFamily = (uint32_t)family;
        }
    }

    if (best == VK_NULL_HANDLE)
        throw std::runtime_error(
            "No Vulkan device with swapchain + Win32 external memory/semaphore support "
            "and a graphics+present queue was found");

    m_physicalDevice = best;
    m_queueFamily = bestFamily;

    VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &idProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);
    std::memcpy(m_deviceUUID, idProps.deviceUUID, 16);

    LOG_INFO("Selected Vulkan device: %s", props2.properties.deviceName);
}

// ---------------------------------------------------------------------------
void VulkanContext::createLogicalDevice()
{
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.pEnabledFeatures        = &features;
    ci.enabledExtensionCount   = (uint32_t)(sizeof(kDeviceExtensions) / sizeof(kDeviceExtensions[0]));
    ci.ppEnabledExtensionNames = kDeviceExtensions;
    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));

    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    m_pfnGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)
        vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR");
    m_pfnGetSemaphoreWin32Handle = (PFN_vkGetSemaphoreWin32HandleKHR)
        vkGetDeviceProcAddr(m_device, "vkGetSemaphoreWin32HandleKHR");
    if (!m_pfnGetMemoryWin32Handle || !m_pfnGetSemaphoreWin32Handle)
        throw std::runtime_error("Failed to load Win32 external handle entry points");
}

// ---------------------------------------------------------------------------
void VulkanContext::createSwapchain()
{
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &caps));

    // Format: prefer B8G8R8A8_UNORM, else R8G8B8A8_UNORM, else first available
    uint32_t fn = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fn, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fn);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fn, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { chosen = f; break; }
    if (chosen.format != VK_FORMAT_B8G8R8A8_UNORM)
        for (auto& f : formats)
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM) { chosen = f; break; }
    m_swapFormat = chosen.format;
    m_swapRB = (m_swapFormat == VK_FORMAT_B8G8R8A8_UNORM ||
                m_swapFormat == VK_FORMAT_B8G8R8A8_SRGB);

    // Present mode: MAILBOX (low latency, uncapped) if available, else FIFO
    uint32_t pn = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pn, nullptr);
    std::vector<VkPresentModeKHR> modes(pn);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pn, modes.data());
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = m; break; }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu)
    {
        extent.width  = std::clamp(m_width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent.height = std::clamp(m_height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    m_extent = extent;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        throw std::runtime_error("Swapchain does not support TRANSFER_DST usage");

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface          = m_surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = chosen.format;
    ci.imageColorSpace  = chosen.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = presentMode;
    ci.clipped          = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain));

    uint32_t cnt = 0;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &cnt, nullptr);
    m_swapImages.resize(cnt);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &cnt, m_swapImages.data());
    m_imagesInFlight.assign(cnt, VK_NULL_HANDLE);

    LOG_INFO("Swapchain: %u images, %ux%u, present mode %s",
             cnt, extent.width, extent.height,
             presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO");
}

// ---------------------------------------------------------------------------
void VulkanContext::destroySwapchain()
{
    if (!m_cmdBuffers.empty())
    {
        vkFreeCommandBuffers(m_device, m_cmdPool,
                             (uint32_t)m_cmdBuffers.size(), m_cmdBuffers.data());
        m_cmdBuffers.clear();
    }
    if (m_swapchain)
    {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void VulkanContext::recreateSwapchain()
{
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while (w == 0 || h == 0) // minimized: wait
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(m_window, &w, &h);
    }
    vkDeviceWaitIdle(m_device);
    destroySwapchain();
    createSwapchain();
    createCommandBuffers();
    LOG_WARN("Swapchain recreated");
}

// ---------------------------------------------------------------------------
// Device-local buffer whose memory is exported as an opaque Win32 handle.
// CUDA imports this memory and the render kernel writes RGBA8 pixels into it.
// Every frame Vulkan copies buffer -> swapchain image entirely on the GPU.
// ---------------------------------------------------------------------------
void VulkanContext::createInteropBuffer()
{
    m_interopBufferSize = (size_t)m_width * m_height * 4;

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext       = &ext;
    bci.size        = m_interopBufferSize;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bci, nullptr, &m_interopBuffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, m_interopBuffer, &req);
    m_interopAllocSize = (size_t)req.size;

    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext           = &exp;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &m_interopMemory));
    VK_CHECK(vkBindBufferMemory(m_device, m_interopBuffer, m_interopMemory, 0));

    VkMemoryGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    gi.memory     = m_interopMemory;
    gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VK_CHECK(m_pfnGetMemoryWin32Handle(m_device, &gi, &m_interopMemHandle));

    LOG_INFO("Interop buffer created: %zu bytes (allocation %zu)",
             m_interopBufferSize, m_interopAllocSize);
}

// ---------------------------------------------------------------------------
void VulkanContext::createInteropSemaphores()
{
    VkExportSemaphoreCreateInfo exp{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exp.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &exp;
    VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_semCudaToVk));
    VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_semVkToCuda));

    VkSemaphoreGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    gi.semaphore = m_semCudaToVk;
    VK_CHECK(m_pfnGetSemaphoreWin32Handle(m_device, &gi, &m_semCudaToVkHandle));
    gi.semaphore = m_semVkToCuda;
    VK_CHECK(m_pfnGetSemaphoreWin32Handle(m_device, &gi, &m_semVkToCudaHandle));

    LOG_INFO("Interop semaphores exported");
}

// ---------------------------------------------------------------------------
// Pre-record one command buffer per swapchain image:
//   UNDEFINED -> TRANSFER_DST barrier, copy buffer->image, -> PRESENT_SRC
// ---------------------------------------------------------------------------
void VulkanContext::createCommandBuffers()
{
    if (m_cmdPool == VK_NULL_HANDLE)
    {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = m_queueFamily;
        VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_cmdPool));
    }

    m_cmdBuffers.resize(m_swapImages.size());
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = m_cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)m_cmdBuffers.size();
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, m_cmdBuffers.data()));

    for (size_t i = 0; i < m_cmdBuffers.size(); ++i)
    {
        VkCommandBuffer cb = m_cmdBuffers[i];
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));

        VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image               = m_swapImages[i];
        toDst.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toDst.srcAccessMask       = 0;
        toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toDst);

        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0; // tightly packed
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {m_extent.width, m_extent.height, 1};
        vkCmdCopyBufferToImage(cb, m_interopBuffer, m_swapImages[i],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier toPresent = toDst;
        toPresent.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toPresent.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toPresent.dstAccessMask = 0;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toPresent);

        VK_CHECK(vkEndCommandBuffer(cb));
    }
}

// ---------------------------------------------------------------------------
void VulkanContext::createSyncObjects()
{
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_imgAvailable[i]));
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_renderDone[i]));
        VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_inFlight[i]));
    }
}

// ---------------------------------------------------------------------------
uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable Vulkan memory type found");
}

// ---------------------------------------------------------------------------
VulkanFrameStats VulkanContext::drawFrame()
{
    VulkanFrameStats stats;

    VK_CHECK(vkWaitForFences(m_device, 1, &m_inFlight[m_frame], VK_TRUE, UINT64_MAX));

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                         m_imgAvailable[m_frame], VK_NULL_HANDLE,
                                         &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        acq = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                    m_imgAvailable[m_frame], VK_NULL_HANDLE,
                                    &imageIndex);
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
    {
        LOG_ERROR("vkAcquireNextImageKHR failed: %d", (int)acq);
        return stats;
    }

    if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        VK_CHECK(vkWaitForFences(m_device, 1, &m_imagesInFlight[imageIndex],
                                 VK_TRUE, UINT64_MAX));
    m_imagesInFlight[imageIndex] = m_inFlight[m_frame];
    VK_CHECK(vkResetFences(m_device, 1, &m_inFlight[m_frame]));

    auto t0 = std::chrono::steady_clock::now();

    // Wait: image available (transfer stage) AND CUDA render finished
    VkSemaphore waitSems[2]           = {m_imgAvailable[m_frame], m_semCudaToVk};
    VkPipelineStageFlags waitStages[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT,
                                          VK_PIPELINE_STAGE_TRANSFER_BIT};
    // Signal: presentation semaphore AND "buffer free again" semaphore for CUDA
    VkSemaphore signalSems[2] = {m_renderDone[m_frame], m_semVkToCuda};

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 2;
    si.pWaitSemaphores      = waitSems;
    si.pWaitDstStageMask    = waitStages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &m_cmdBuffers[imageIndex];
    si.signalSemaphoreCount = 2;
    si.pSignalSemaphores    = signalSems;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &si, m_inFlight[m_frame]));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &m_renderDone[m_frame];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &imageIndex;
    VkResult pres = vkQueuePresentKHR(m_queue, &pi);

    stats.presentMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR)
        recreateSwapchain();
    else if (pres != VK_SUCCESS)
    {
        LOG_ERROR("vkQueuePresentKHR failed: %d", (int)pres);
        return stats;
    }

    m_frame = (m_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    stats.success = true;
    return stats;
}

// ---------------------------------------------------------------------------
void VulkanContext::waitIdle()
{
    if (m_device) vkDeviceWaitIdle(m_device);
}

void VulkanContext::cleanup()
{
    waitIdle();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (m_imgAvailable[i]) vkDestroySemaphore(m_device, m_imgAvailable[i], nullptr);
        if (m_renderDone[i])   vkDestroySemaphore(m_device, m_renderDone[i], nullptr);
        if (m_inFlight[i])     vkDestroyFence(m_device, m_inFlight[i], nullptr);
    }
    if (m_semCudaToVk) vkDestroySemaphore(m_device, m_semCudaToVk, nullptr);
    if (m_semVkToCuda) vkDestroySemaphore(m_device, m_semVkToCuda, nullptr);
    if (m_semCudaToVkHandle) CloseHandle(m_semCudaToVkHandle);
    if (m_semVkToCudaHandle) CloseHandle(m_semVkToCudaHandle);

    destroySwapchain();
    if (m_cmdPool) vkDestroyCommandPool(m_device, m_cmdPool, nullptr);

    if (m_interopBuffer) vkDestroyBuffer(m_device, m_interopBuffer, nullptr);
    if (m_interopMemory) vkFreeMemory(m_device, m_interopMemory, nullptr);
    if (m_interopMemHandle) CloseHandle(m_interopMemHandle);

    if (m_device)  vkDestroyDevice(m_device, nullptr);
    if (m_debugMessenger)
    {
        auto pfn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (pfn) pfn(m_instance, m_debugMessenger, nullptr);
    }
    if (m_surface)  vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);

    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
    LOG_INFO("Vulkan context destroyed");
}
