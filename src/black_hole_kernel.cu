// ---------------------------------------------------------------------------
// black_hole_kernel.cu
//
// Render pipeline for the four-model black hole renderer:
//   1. renderAccumKernel   one jittered geodesic sample per pixel, blended
//                          into the linear-HDR accumulation buffer
//                          (progressive temporal anti-aliasing)
//   2. bloomDownsample     half-res bright pass on the exposed color
//   3. bloomBlurH/V        separable 9-tap Gaussian
//   4. compositeKernel     exposure + bloom + ACES + gamma + dither -> uchar4
//
// All physics lives in trace.cuh, all post-processing in post_process.cuh
// (both shared with the test executable and the CPU verification harness).
// ---------------------------------------------------------------------------
#include "render_params.h"
#include "metric.cuh"
#include "trace.cuh"
#include "post_process.cuh"
#include "kernel_launch.h"
#include <cuda_runtime.h>

__global__ void renderAccumKernel(float4* accum, RenderParams P)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= P.width || py >= P.height) return;

    float2 j   = subpixelJitter(P.accumMode, P.sampleIndex);
    float3 cur = renderSampleHDR(P, px, py, j.x, j.y);

    int    idx  = py * P.width + px;
    float4 pv   = accum[idx];
    float3 prev = make_float3(pv.x, pv.y, pv.z);
    if (!isfinite(prev.x) || !isfinite(prev.y) || !isfinite(prev.z))
        prev = cur;

    float3 c = accumBlend(prev, cur, P.accumMode, P.sampleIndex);
    accum[idx] = make_float4(c.x, c.y, c.z, 1.f);
}

__global__ void bloomDownsampleKernel(const float4* accum, float4* dst,
                                      RenderParams P, int bw, int bh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;

    int x0 = x * 2, y0 = y * 2;
    int x1 = (x0 + 1 < P.width)  ? x0 + 1 : x0;
    int y1 = (y0 + 1 < P.height) ? y0 + 1 : y0;

    float4 a = accum[y0 * P.width + x0];
    float4 b = accum[y0 * P.width + x1];
    float4 c = accum[y1 * P.width + x0];
    float4 d = accum[y1 * P.width + x1];
    float3 avg = make_float3(a.x + b.x + c.x + d.x,
                             a.y + b.y + c.y + d.y,
                             a.z + b.z + c.z + d.z) * 0.25f;

    float3 bp = bloomBrightPass(avg * P.exposure);
    dst[y * bw + x] = make_float4(bp.x, bp.y, bp.z, 1.f);
}

__global__ void bloomBlurHKernel(const float4* src, float4* dst, int bw, int bh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    float3 s = make_float3(0.f, 0.f, 0.f);
    for (int k = -4; k <= 4; ++k)
        s += fetchHalf(src, x + k, y, bw, bh) * bloomGaussW(k);
    dst[y * bw + x] = make_float4(s.x, s.y, s.z, 1.f);
}

__global__ void bloomBlurVKernel(const float4* src, float4* dst, int bw, int bh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    float3 s = make_float3(0.f, 0.f, 0.f);
    for (int k = -4; k <= 4; ++k)
        s += fetchHalf(src, x, y + k, bw, bh) * bloomGaussW(k);
    dst[y * bw + x] = make_float4(s.x, s.y, s.z, 1.f);
}

__global__ void compositeKernel(uchar4* out, const float4* accum,
                                const float4* bloomTex,
                                RenderParams P, int bw, int bh)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= P.width || py >= P.height) return;

    float4 av  = accum[py * P.width + px];
    float3 hdr = make_float3(av.x, av.y, av.z);

    float3 bloom = make_float3(0.f, 0.f, 0.f);
    if (P.bloomEnabled)
        bloom = sampleHalfBilinear(bloomTex, (px + 0.5f) * 0.5f,
                                             (py + 0.5f) * 0.5f, bw, bh);

    out[py * P.width + px] = finalizePixel(hdr, bloom, P, px, py);
}

// ---------------------------------------------------------------------------
// Full-quality pipeline entry (sanitizes all inputs).
// ---------------------------------------------------------------------------
extern "C" cudaError_t launchRenderPipeline(uchar4* out, float4* accum,
                                            float4* bloomA, float4* bloomB,
                                            const RenderParams& p,
                                            cudaStream_t stream)
{
    RenderParams P = p;
    sanitizeRenderParams(P);

    const int bw = (P.width + 1) / 2, bh = (P.height + 1) / 2;
    // Trace kernel is register-heavy: 128-thread blocks give it better
    // occupancy. The memory-bound post-process kernels keep 256.
    dim3 blockT(16, 8);
    dim3 gridT((P.width + 15) / 16, (P.height + 7) / 8);
    dim3 block(16, 16);
    dim3 grid((P.width + 15) / 16, (P.height + 15) / 16);
    dim3 hgrid((bw + 15) / 16, (bh + 15) / 16);

    KLAUNCH(renderAccumKernel, gridT, blockT, stream, accum, P);
    if (P.bloomEnabled)
    {
        KLAUNCH(bloomDownsampleKernel, hgrid, block, stream, accum, bloomA, P, bw, bh);
        KLAUNCH(bloomBlurHKernel, hgrid, block, stream, bloomA, bloomB, bw, bh);
        KLAUNCH(bloomBlurVKernel, hgrid, block, stream, bloomB, bloomA, bw, bh);
    }
    KLAUNCH(compositeKernel, grid, block, stream, out, accum, bloomA, P, bw, bh);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Single-pass legacy entry (kept for the verification suite's smoke tests):
// one centered sample, no accumulation, no bloom.
// ---------------------------------------------------------------------------
__global__ void renderKernel(uchar4* out, RenderParams P)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= P.width || py >= P.height) return;

    float3 hdr = renderSampleHDR(P, px, py, 0.5f, 0.5f);
    out[py * P.width + px] =
        finalizePixel(hdr, make_float3(0.f, 0.f, 0.f), P, px, py);
}

extern "C" cudaError_t launchRenderKernel(uchar4* out, const RenderParams& p,
                                          cudaStream_t stream)
{
    RenderParams P = p;
    sanitizeRenderParams(P);
    P.bloomEnabled = 0;

    dim3 block(16, 8);
    dim3 grid((P.width + block.x - 1) / block.x,
              (P.height + block.y - 1) / block.y);
    KLAUNCH(renderKernel, grid, block, stream, out, P);
    return cudaGetLastError();
}
