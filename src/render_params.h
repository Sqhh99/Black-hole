#pragma once
// ---------------------------------------------------------------------------
// render_params.h
// Plain-old-data parameter block shared between the host application (MSVC)
// and the CUDA kernel (NVCC). Uses CUDA's vector_types.h so both compilers
// agree on the layout of float3 / uchar4.
// ---------------------------------------------------------------------------
#include <vector_types.h>
#include <cstdint>

struct RenderParams
{
    // Framebuffer
    int      width  = 1280;
    int      height = 720;
    int      swapRB = 0;          // 1 if swapchain format is BGRA

    // Camera (right-handed, looking at the black hole at the origin)
    float3   camPos     = {0.f, 0.f, 0.f};
    float3   camForward = {0.f, 0.f, -1.f};
    float3   camRight   = {1.f, 0.f, 0.f};
    float3   camUp      = {0.f, 1.f, 0.f};
    float    tanHalfFov = 0.5773503f;   // tan(60deg / 2)
    float    aspect     = 1280.f / 720.f;

    // Simulation / integration (geometric units, Schwarzschild radius rs = 1)
    float    dPhi     = 0.012f;   // RK4 step in the orbital angle phi
    int      maxSteps = 2000;     // hard cap on integration steps per ray

    // Accretion disk
    int      diskEnabled = 1;
    float    diskInner   = 3.0f;  // ISCO for rs = 1 (r = 6M, M = 0.5)
    float    diskOuter   = 12.0f;
    float    diskTime    = 0.f;   // animation time (seconds, pausable)

    // Post processing
    float    exposure = 1.0f;
};
