#pragma once
// ---------------------------------------------------------------------------
// cuda_interop.h
//
// Imports the Vulkan-exported device memory and semaphores into CUDA
// (cudaImportExternalMemory / cudaImportExternalSemaphore) and drives the
// render kernel with pure GPU-side synchronization:
//
//   CUDA:   [wait semVkToCuda] -> kernel -> signal semCudaToVk
//   Vulkan:  wait semCudaToVk  -> copy buffer->swapchain -> signal semVkToCuda
//
// The rendered image never leaves the GPU.
// ---------------------------------------------------------------------------
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "render_params.h"

class CudaInterop
{
public:
    // Returns the CUDA device index whose UUID matches the Vulkan device
    // UUID, or -1 if none matches.
    static int findCudaDeviceByUUID(const uint8_t uuid[16]);

    void init(int cudaDevice,
              HANDLE vkMemoryHandle, size_t allocSize, size_t bufferSize,
              HANDLE semVkToCudaHandle,   // CUDA waits on this
              HANDLE semCudaToVkHandle);  // CUDA signals this
    void cleanup();

    // Enqueues (optional) wait, the render kernel, timing events and the
    // signal on the CUDA stream. Non-blocking apart from collecting the
    // previous frame's kernel time. Returns last measured kernel time in ms.
    float render(const RenderParams& params, bool waitForVulkan);

    void sync(); // full stream sync (used at shutdown)

private:
    cudaStream_t            m_stream      = nullptr;
    cudaExternalMemory_t    m_extMemory   = nullptr;
    void*                   m_devPtr      = nullptr;
    cudaExternalSemaphore_t m_semVkToCuda = nullptr;
    cudaExternalSemaphore_t m_semCudaToVk = nullptr;
    cudaEvent_t             m_evStart     = nullptr;
    cudaEvent_t             m_evStop      = nullptr;
    uint64_t                m_frames      = 0;
    float                   m_lastKernelMs = 0.f;
};
