#pragma once
// ---------------------------------------------------------------------------
// post_process.cuh -- display-quality pipeline shared by the render kernels
// and the CPU verification harness.
//
//   1. Progressive temporal anti-aliasing. Each frame traces one jittered
//      sub-pixel sample (quasi-random R2 sequence). While the view is
//      static the samples are averaged in a linear-HDR float4 accumulation
//      buffer, converging to supersampled quality within a few frames:
//        accumMode 0  view changed   -> overwrite (centered sample, exactly
//                                       the previous single-sample look)
//        accumMode 1  static + disk paused/off -> progressive average
//        accumMode 2  static + disk animating  -> exponential moving average
//                                       (temporal AA + mild motion blur of
//                                       the orbiting gas; no ghosting since
//                                       the camera is static)
//      Exposure and bloom are applied AFTER accumulation, so adjusting them
//      does not reset convergence.
//
//   2. HDR bloom: bright-pass at half resolution, separable 9-tap Gaussian,
//      bilinear upsample, additive composite. Negligible cost next to the
//      geodesic integration.
//
//   3. Dithered quantization: triangular-pdf spatial dither decorrelates the
//      8-bit rounding error, removing banding in the dark background.
// ---------------------------------------------------------------------------
#include "render_params.h"
#include "vec_math.cuh"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// ACES filmic tone map (Narkowicz fit)
// ---------------------------------------------------------------------------
__device__ inline float3 acesToneMap(float3 x)
{
    const float A = 2.51f, B = 0.03f, C = 2.43f, D = 0.59f, E = 0.14f;
    float3 num = x * (x * A + make_float3(B, B, B));
    float3 den = x * (x * C + make_float3(D, D, D)) + make_float3(E, E, E);
    return clamp3(make_float3(num.x / den.x, num.y / den.y, num.z / den.z), 0.f, 1.f);
}

// ---------------------------------------------------------------------------
// Sub-pixel jitter: R2 low-discrepancy sequence. Sample 0 / overwrite mode
// uses the pixel center so a moving camera looks exactly like the
// single-sample renderer.
// ---------------------------------------------------------------------------
__device__ inline float2 subpixelJitter(int accumMode, int sampleIndex)
{
    if (accumMode == 0 || sampleIndex <= 0) return make_float2(0.5f, 0.5f);
    float n = (float)(sampleIndex);
    return make_float2(fract(0.7548776662f * n + 0.5f),
                       fract(0.5698402909f * n + 0.5f));
}

// Blend a new HDR sample into the accumulation buffer value.
__device__ inline float3 accumBlend(float3 prev, float3 cur,
                                    int accumMode, int sampleIndex)
{
    if (accumMode == 0 || sampleIndex <= 0) return cur;
    if (accumMode == 1)
    {
        float n = (float)sampleIndex;
        return prev + (cur - prev) * (1.0f / (n + 1.0f));
    }
    // Higher α keeps orbiting hot spots / photon-ring arcs readable while
    // still damping single-sample noise. (Was 0.25; 0.42 ≈ 2-frame memory.)
    return lerp3(prev, cur, 0.42f);
}

// ---------------------------------------------------------------------------
// Bloom
// ---------------------------------------------------------------------------
// Bright-pass on the *exposed* color: smooth threshold around the tone-map
// shoulder so only genuinely hot regions (inner disk, photon ring, beamed
// side) bloom, not the starfield.
__device__ inline float3 bloomBrightPass(float3 c)
{
    float l = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    // Threshold just under the tone-map shoulder: the photon ring, beamed
    // inner disk and the brightest stars bloom; the outer annulus and the
    // general starfield do not (avoids a soft yellow pancake glow).
    float w = smoothstepf(0.95f, 2.2f, l);
    return c * w;
}

// Normalized 9-tap Gaussian (sigma ~ 3 at half resolution, i.e. an
// effective ~6-pixel radius at full resolution).
__device__ inline float bloomGaussW(int i)   // i in [-4, 4]
{
    const float w[5] = {0.20236f, 0.17925f, 0.12458f, 0.06794f, 0.02907f};
    int k = i < 0 ? -i : i;
    return w[k > 4 ? 4 : k];
}

// Clamped fetch + bilinear sample of a half-resolution float4 buffer.
__device__ inline float3 fetchHalf(const float4* buf, int x, int y, int bw, int bh)
{
    if (x < 0) x = 0;
    if (x >= bw) x = bw - 1;
    if (y < 0) y = 0;
    if (y >= bh) y = bh - 1;
    float4 v = buf[y * bw + x];
    return make_float3(v.x, v.y, v.z);
}

__device__ inline float3 sampleHalfBilinear(const float4* buf,
                                            float x, float y, int bw, int bh)
{
    x -= 0.5f; y -= 0.5f;
    int   x0 = (int)floorf(x), y0 = (int)floorf(y);
    float tx = x - (float)x0,  ty = y - (float)y0;
    float3 c00 = fetchHalf(buf, x0,     y0,     bw, bh);
    float3 c10 = fetchHalf(buf, x0 + 1, y0,     bw, bh);
    float3 c01 = fetchHalf(buf, x0,     y0 + 1, bw, bh);
    float3 c11 = fetchHalf(buf, x0 + 1, y0 + 1, bw, bh);
    return lerp3(lerp3(c00, c10, tx), lerp3(c01, c11, tx), ty);
}

// ---------------------------------------------------------------------------
// Final pixel: exposure -> bloom composite -> ACES -> gamma -> dither -> 8bit
// ---------------------------------------------------------------------------
__device__ inline uchar4 finalizePixel(float3 hdr, float3 bloom,
                                       const RenderParams& P, int px, int py)
{
    float3 c = hdr * P.exposure;
    if (P.bloomEnabled) c += bloom * P.bloomStrength;

    c = acesToneMap(c);
    // Mild vibrance: ACES pulls saturated oranges toward gray; restore some
    // chroma so the disk keeps its EHT palette (applied pre-gamma).
    float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    float3 lv = make_float3(lum, lum, lum);
    c = clamp3(lv + (c - lv) * 1.12f, 0.f, 1.f);
    c = pow3(c, 1.0f / 2.2f);

    // Triangular-pdf spatial dither (+-1 LSB) decorrelates quantization
    // error; static per pixel so a converged image stays perfectly still.
    float2 fp = make_float2((float)px, (float)py);
    float  d  = (hash12(fp) + hash12(fp + make_float2(41.3f, 17.7f))) - 1.0f;
    float  dd = d * (1.0f / 255.0f);

    unsigned char r8 = (unsigned char)(clampf(c.x + dd, 0.f, 1.f) * 255.f + 0.5f);
    unsigned char g8 = (unsigned char)(clampf(c.y + dd, 0.f, 1.f) * 255.f + 0.5f);
    unsigned char b8 = (unsigned char)(clampf(c.z + dd, 0.f, 1.f) * 255.f + 0.5f);

    return P.swapRB ? make_uchar4(b8, g8, r8, 255)
                    : make_uchar4(r8, g8, b8, 255);
}
