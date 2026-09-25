#pragma once
// ---------------------------------------------------------------------------
// vec_math.cuh - minimal float2/float3 math for device code.
// Included only from .cu translation units.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <math.h>

#define VM_FUNC __device__ __forceinline__

// ---------------- float3 ----------------
VM_FUNC float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
VM_FUNC float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
VM_FUNC float3 operator-(float3 a)           { return make_float3(-a.x, -a.y, -a.z); }
VM_FUNC float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
VM_FUNC float3 operator*(float3 a, float s)  { return make_float3(a.x * s, a.y * s, a.z * s); }
VM_FUNC float3 operator*(float s, float3 a)  { return a * s; }
VM_FUNC float3 operator/(float3 a, float s)  { return a * (1.0f / s); }
VM_FUNC float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
VM_FUNC float3& operator*=(float3& a, float s)  { a.x *= s; a.y *= s; a.z *= s; return a; }

VM_FUNC float  dot(float3 a, float3 b)   { return a.x * b.x + a.y * b.y + a.z * b.z; }
VM_FUNC float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
VM_FUNC float  length(float3 a)    { return sqrtf(dot(a, a)); }
VM_FUNC float3 normalize(float3 a) { return a * rsqrtf(fmaxf(dot(a, a), 1e-20f)); }
VM_FUNC float3 lerp3(float3 a, float3 b, float t) { return a + (b - a) * t; }

// ---------------- float2 ----------------
VM_FUNC float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
VM_FUNC float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
VM_FUNC float2 operator*(float2 a, float s)  { return make_float2(a.x * s, a.y * s); }

// ---------------- scalar helpers ----------------
VM_FUNC float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
VM_FUNC float fract(float x)                      { return x - floorf(x); }
VM_FUNC float lerpf(float a, float b, float t)    { return a + (b - a) * t; }
VM_FUNC float smoothstepf(float e0, float e1, float x)
{
    float t = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
VM_FUNC float3 clamp3(float3 v, float lo, float hi)
{
    return make_float3(clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi));
}
VM_FUNC float3 pow3(float3 v, float e)
{
    return make_float3(powf(v.x, e), powf(v.y, e), powf(v.z, e));
}

// ---------------- hashing / noise ----------------
VM_FUNC float hash12(float2 p)
{
    float3 p3 = make_float3(fract(p.x * 0.1031f), fract(p.y * 0.1031f), fract(p.x * 0.1031f));
    float  d  = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    p3 = make_float3(p3.x + d, p3.y + d, p3.z + d);
    return fract((p3.x + p3.y) * p3.z);
}

VM_FUNC float3 hash33(float3 p)
{
    p = make_float3(fract(p.x * 0.1031f), fract(p.y * 0.1030f), fract(p.z * 0.0973f));
    float d = p.x * (p.y + 33.33f) + p.y * (p.x + 33.33f) + p.z * (p.z + 33.33f);
    p = make_float3(p.x + d, p.y + d, p.z + d);
    return make_float3(fract((p.x + p.y) * p.z),
                       fract((p.x + p.z) * p.y),
                       fract((p.y + p.z) * p.x));
}

// 2D value noise
VM_FUNC float vnoise(float2 p)
{
    float2 i = make_float2(floorf(p.x), floorf(p.y));
    float2 f = make_float2(p.x - i.x, p.y - i.y);
    float2 u = make_float2(f.x * f.x * (3.f - 2.f * f.x),
                           f.y * f.y * (3.f - 2.f * f.y));
    float a = hash12(i);
    float b = hash12(i + make_float2(1.f, 0.f));
    float c = hash12(i + make_float2(0.f, 1.f));
    float d = hash12(i + make_float2(1.f, 1.f));
    return lerpf(lerpf(a, b, u.x), lerpf(c, d, u.x), u.y);
}

VM_FUNC float fbm(float2 p)
{
    float v = 0.f, amp = 0.5f;
    for (int i = 0; i < 3; ++i)
    {
        v   += amp * vnoise(p);
        p    = make_float2(p.x * 2.03f + 11.7f, p.y * 2.01f - 5.2f);
        amp *= 0.5f;
    }
    return v;
}

// 3D hash -> [0,1) (Hoskins "hash13")
VM_FUNC float hash13(float3 p)
{
    p = make_float3(fract(p.x * 0.1031f), fract(p.y * 0.1031f), fract(p.z * 0.1031f));
    float d = p.x * (p.z + 31.32f) + p.y * (p.y + 31.32f) + p.z * (p.x + 31.32f);
    p = make_float3(p.x + d, p.y + d, p.z + d);
    return fract((p.x + p.y) * p.z);
}

// 3D value noise. Used wherever a 2D parameterization would have a seam
// (sky directions, disk azimuth), since it can be sampled on closed curves.
VM_FUNC float vnoise3(float3 p)
{
    float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f = p - i;
    float3 u = make_float3(f.x * f.x * (3.f - 2.f * f.x),
                           f.y * f.y * (3.f - 2.f * f.y),
                           f.z * f.z * (3.f - 2.f * f.z));
    float n000 = hash13(i);
    float n100 = hash13(i + make_float3(1.f, 0.f, 0.f));
    float n010 = hash13(i + make_float3(0.f, 1.f, 0.f));
    float n110 = hash13(i + make_float3(1.f, 1.f, 0.f));
    float n001 = hash13(i + make_float3(0.f, 0.f, 1.f));
    float n101 = hash13(i + make_float3(1.f, 0.f, 1.f));
    float n011 = hash13(i + make_float3(0.f, 1.f, 1.f));
    float n111 = hash13(i + make_float3(1.f, 1.f, 1.f));
    float x00 = lerpf(n000, n100, u.x), x10 = lerpf(n010, n110, u.x);
    float x01 = lerpf(n001, n101, u.x), x11 = lerpf(n011, n111, u.x);
    return lerpf(lerpf(x00, x10, u.y), lerpf(x01, x11, u.y), u.z);
}

// Anisotropic 3D fBm: `octaves` octaves, frequency doubling per octave.
VM_FUNC float fbm3(float3 p, int octaves)
{
    float v = 0.f, amp = 0.5f, norm = 0.f;
    for (int i = 0; i < octaves; ++i)
    {
        v    += amp * vnoise3(p);
        norm += amp;
        p     = make_float3(p.x * 2.03f + 17.1f, p.y * 2.01f - 7.3f, p.z * 1.99f + 3.9f);
        amp  *= 0.5f;
    }
    return v / norm;
}
