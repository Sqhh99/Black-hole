// ---------------------------------------------------------------------------
// cuda_interop.cpp
// ---------------------------------------------------------------------------
#include "cuda_interop.h"
#include "logger.h"

#include <cstring>
#include <stdexcept>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            LOG_ERROR("CUDA error %s (%d) at %s:%d -> %s",                    \
                      cudaGetErrorString(_e), (int)_e, __FILE__, __LINE__,    \
                      #call);                                                 \
            throw std::runtime_error("CUDA call failed: " #call);             \
        }                                                                     \
    } while (0)

// Implemented in black_hole_kernel.cu
extern "C" cudaError_t launchRenderPipeline(uchar4* out, float4* accum,
                                            float4* bloomA, float4* bloomB,
                                            const RenderParams& p,
                                            cudaStream_t stream);

// ---------------------------------------------------------------------------
int CudaInterop::findCudaDeviceByUUID(const uint8_t uuid[16])
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        if (std::memcmp(prop.uuid.bytes, uuid, 16) == 0)
        {
            LOG_INFO("CUDA device %d (%s) matches Vulkan device UUID", i, prop.name);
            return i;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
void CudaInterop::init(int cudaDevice, int width, int height,
                       HANDLE vkMemoryHandle, size_t allocSize, size_t bufferSize,
                       HANDLE semVkToCudaHandle, HANDLE semCudaToVkHandle)
{
    CUDA_CHECK(cudaSetDevice(cudaDevice));

    // Auxiliary buffers for the display-quality pipeline (device-local,
    // CUDA-private; never copied to the CPU).
    const int bw = (width + 1) / 2, bh = (height + 1) / 2;
    CUDA_CHECK(cudaMalloc(&m_accum,  sizeof(float4) * (size_t)width * height));
    CUDA_CHECK(cudaMalloc(&m_bloomA, sizeof(float4) * (size_t)bw * bh));
    CUDA_CHECK(cudaMalloc(&m_bloomB, sizeof(float4) * (size_t)bw * bh));
    CUDA_CHECK(cudaMemset(m_accum,  0, sizeof(float4) * (size_t)width * height));
    CUDA_CHECK(cudaMemset(m_bloomA, 0, sizeof(float4) * (size_t)bw * bh));
    CUDA_CHECK(cudaMemset(m_bloomB, 0, sizeof(float4) * (size_t)bw * bh));
    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&m_evStart));
    CUDA_CHECK(cudaEventCreate(&m_evStop));

    // ---- Import the Vulkan device memory ----
    cudaExternalMemoryHandleDesc memDesc{};
    memDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    memDesc.handle.win32.handle = vkMemoryHandle;
    memDesc.size                = allocSize;
    CUDA_CHECK(cudaImportExternalMemory(&m_extMemory, &memDesc));

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = bufferSize;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&m_devPtr, m_extMemory, &bufDesc));

    // ---- Import the Vulkan semaphores ----
    cudaExternalSemaphoreHandleDesc semDesc{};
    semDesc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    semDesc.handle.win32.handle = semVkToCudaHandle;
    CUDA_CHECK(cudaImportExternalSemaphore(&m_semVkToCuda, &semDesc));
    semDesc.handle.win32.handle = semCudaToVkHandle;
    CUDA_CHECK(cudaImportExternalSemaphore(&m_semCudaToVk, &semDesc));

    LOG_INFO("CUDA interop initialized: mapped %zu bytes of Vulkan memory at %p",
             bufferSize, m_devPtr);
}

// ---------------------------------------------------------------------------
float CudaInterop::render(const RenderParams& params, bool waitForVulkan)
{
    // Collect the previous frame's kernel time (the kernel has long since
    // finished, because Vulkan already consumed and presented that frame).
    if (m_frames > 0)
    {
        CUDA_CHECK(cudaEventSynchronize(m_evStop));
        CUDA_CHECK(cudaEventElapsedTime(&m_lastKernelMs, m_evStart, m_evStop));
    }

    if (waitForVulkan)
    {
        cudaExternalSemaphoreWaitParams wp{};
        CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&m_semVkToCuda, &wp, 1, m_stream));
    }

    CUDA_CHECK(cudaEventRecord(m_evStart, m_stream));
    CUDA_CHECK(launchRenderPipeline((uchar4*)m_devPtr, m_accum,
                                    m_bloomA, m_bloomB, params, m_stream));
    CUDA_CHECK(cudaEventRecord(m_evStop, m_stream));

    cudaExternalSemaphoreSignalParams sp{};
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_semCudaToVk, &sp, 1, m_stream));

    ++m_frames;
    return m_lastKernelMs;
}

// ---------------------------------------------------------------------------
void CudaInterop::sync()
{
    if (m_stream) cudaStreamSynchronize(m_stream);
}

void CudaInterop::cleanup()
{
    sync();
    if (m_devPtr)      cudaFree(m_devPtr);
    if (m_accum)       cudaFree(m_accum);
    if (m_bloomA)      cudaFree(m_bloomA);
    if (m_bloomB)      cudaFree(m_bloomB);
    if (m_extMemory)   cudaDestroyExternalMemory(m_extMemory);
    if (m_semVkToCuda) cudaDestroyExternalSemaphore(m_semVkToCuda);
    if (m_semCudaToVk) cudaDestroyExternalSemaphore(m_semCudaToVk);
    if (m_evStart)     cudaEventDestroy(m_evStart);
    if (m_evStop)      cudaEventDestroy(m_evStop);
    if (m_stream)      cudaStreamDestroy(m_stream);
    m_devPtr = nullptr;
    m_accum = m_bloomA = m_bloomB = nullptr;
    m_extMemory = nullptr;
    m_semVkToCuda = m_semCudaToVk = nullptr;
    m_stream = nullptr;
    LOG_INFO("CUDA interop destroyed");
}
