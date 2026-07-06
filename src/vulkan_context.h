#pragma once
// ---------------------------------------------------------------------------
// vulkan_context.h
//
// Owns: GLFW window, VkInstance/Device, swapchain, an exportable
// (VK_KHR_external_memory_win32) device-local staging buffer that CUDA
// renders into, two exportable binary semaphores for GPU-GPU CUDA<->Vulkan
// synchronization, and pre-recorded copy/present command buffers.
// ---------------------------------------------------------------------------
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <vector>

struct VulkanFrameStats
{
    bool   success   = false;
    double presentMs = 0.0;   // CPU time spent in submit + present
};

class VulkanContext
{
public:
    void init(uint32_t width, uint32_t height, const char* title);
    void cleanup();

    GLFWwindow* window() const { return m_window; }
    bool        swapRB() const { return m_swapRB; }
    uint32_t    width()  const { return m_width; }
    uint32_t    height() const { return m_height; }
    bool        fullscreen() const { return m_fullscreen; }

    void toggleFullscreen();
    void recreateDisplayResources();

    // --- CUDA interop exports ---
    HANDLE         interopMemoryHandle() const { return m_interopMemHandle; }
    size_t         interopAllocSize()    const { return m_interopAllocSize; }
    size_t         interopBufferSize()   const { return m_interopBufferSize; }
    HANDLE         semCudaToVkHandle()   const { return m_semCudaToVkHandle; }
    HANDLE         semVkToCudaHandle()   const { return m_semVkToCudaHandle; }
    const uint8_t* deviceUUID()          const { return m_deviceUUID; }

    // Submits the buffer->swapchain copy (waiting on the CUDA-signaled
    // semaphore) and presents. Returns CPU-side timing.
    VulkanFrameStats drawFrame();

    void waitIdle();

private:
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void destroySwapchain();
    void recreateSwapchain();
    void createInteropBuffer();
    void destroyInteropBuffer();
    void createInteropSemaphores();
    void createCommandBuffers();
    void createSyncObjects();
    void updateFramebufferExtent();

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;

    // --- core ---
    GLFWwindow*              m_window        = nullptr;
    VkInstance               m_instance      = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             m_surface       = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device        = VK_NULL_HANDLE;
    VkQueue                  m_queue         = VK_NULL_HANDLE;
    uint32_t                 m_queueFamily   = 0;
    uint8_t                  m_deviceUUID[16] = {};
    bool                     m_validation    = false;
    bool                     m_fullscreen    = false;
    int                      m_windowedX     = 100;
    int                      m_windowedY     = 100;
    int                      m_windowedWidth = 1280;
    int                      m_windowedHeight = 720;

    // --- swapchain ---
    VkSwapchainKHR           m_swapchain     = VK_NULL_HANDLE;
    std::vector<VkImage>     m_swapImages;
    VkFormat                 m_swapFormat    = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D               m_extent        = {};
    uint32_t                 m_width = 0, m_height = 0;
    bool                     m_swapRB        = false;

    // --- commands / sync ---
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    VkCommandPool                m_cmdPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_cmdBuffers;            // one per swap image
    VkSemaphore m_imgAvailable[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore m_renderDone[MAX_FRAMES_IN_FLIGHT]   = {};
    VkFence     m_inFlight[MAX_FRAMES_IN_FLIGHT]     = {};
    std::vector<VkFence> m_imagesInFlight;
    uint32_t    m_frame = 0;

    // --- CUDA interop objects ---
    VkBuffer       m_interopBuffer     = VK_NULL_HANDLE;
    VkDeviceMemory m_interopMemory     = VK_NULL_HANDLE;
    size_t         m_interopBufferSize = 0;
    size_t         m_interopAllocSize  = 0;
    HANDLE         m_interopMemHandle  = nullptr;
    VkSemaphore    m_semCudaToVk       = VK_NULL_HANDLE; // CUDA signals, Vulkan waits
    VkSemaphore    m_semVkToCuda       = VK_NULL_HANDLE; // Vulkan signals, CUDA waits
    HANDLE         m_semCudaToVkHandle = nullptr;
    HANDLE         m_semVkToCudaHandle = nullptr;

    PFN_vkGetMemoryWin32HandleKHR    m_pfnGetMemoryWin32Handle    = nullptr;
    PFN_vkGetSemaphoreWin32HandleKHR m_pfnGetSemaphoreWin32Handle = nullptr;
};
