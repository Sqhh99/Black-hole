#pragma once
// ---------------------------------------------------------------------------
// render_params.h
// Plain-old-data parameter block shared between the host application (MSVC)
// and the CUDA kernel (NVCC). Uses CUDA's vector_types.h so both compilers
// agree on the layout of float3 / uchar4.
//
// Geometric units G = c = 1. Unified mass parameter M; the default M = 0.5
// makes the Schwarzschild radius rs = 2M = 1 code unit, preserving the
// original renderer's length scale (camera distances, disk radii, escape
// radius) and therefore its exact Schwarzschild visuals.
//
// Spin a = a* M may be negative (retrograde). Charge Q = q M stays >= 0.
// When allowNaked != 0, a*^2 + q^2 may exceed 1 (experimental naked
// singularity mode); otherwise parameters are clamped inside the extremal
// bound a*^2 + q^2 <= 0.995.
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

    // Black hole model & parameters (all in code units; host derives and
    // validates them via bhValidateAndDerive in metric.cuh)
    int      model   = 0;      // 0 Schwarzschild, 1 RN, 2 Kerr, 3 Kerr-Newman
    float    M       = 0.5f;   // mass (mass scale of the scene; rs = 2M)
    float    aSpin   = 0.f;    // a  = a* M   (spin, signed; code units)
    float    Qc      = 0.f;    // Q  = q  M   (charge, code units, >= 0)
    float    rPlus   = 1.0f;   // outer event horizon r+ (0 if naked)
    float    rPhoton = 1.5f;   // photon sphere / co-rotating photon orbit
    float    rErgo   = 1.0f;   // equatorial ergosphere radius (info)
    int      allowNaked = 0;   // 1 = permit a*^2+q^2 > 1 (experimental)

    // Integration
    //   Spherical models: dPhi is the RK4 step in orbital angle phi.
    //   Kerr models: the affine-parameter step is adaptive with budgets
    //   proportional to dPhi (angular advance ~2.2*dPhi rad per step,
    //   fractional (r - r+) shrink ~8*dPhi, refined inside the photon
    //   region), so the 1/2/3 quality presets control every model
    //   uniformly and step-halving convergence tests remain meaningful.
    float    dPhi     = 0.012f;
    int      maxSteps = 2000;     // hard cap on integration steps per ray

    // Accretion disk (inner edge = ISCO of the current model, host-computed)
    int      diskEnabled = 1;
    float    diskInner   = 3.0f;  // ISCO: 6M for Schwarzschild with M = 0.5
    float    diskOuter   = 8.0f;  // ~16M — mid size (24M too large, 12M a bit tight)
    float    diskTime    = 0.f;   // animation time (seconds, pausable)

    // Phenomenological RT scales (emission / absorption along the disk).
    // Intensity uses band-integrated I_obs ∝ g^4 I_emit; the Novikov–Thorne
    // flux weight is normalized to 1 at its peak (see BH_NT_PEAK_INV).
    float    diskEmisScale = 1.5f;   // multiplies blackbody * dens * ds * g^4
    float    diskAbsScale  = 2.2f;   // optical-depth scale (keeps shadow crisp)

    // Orbiting hot spots near the ISCO (EHT-style flares). Lensed into the
    // photon ring by the geodesic integrator — not a screen-space overlay.
    int      hotSpotsEnabled  = 1;   // 0 = off
    float    hotSpotStrength  = 0.65f; // moderate — accent ring, not wash disk

    // Post processing
    float    exposure = 1.0f;

    // Progressive temporal anti-aliasing (host-managed; see post_process.cuh)
    //   accumMode 0: view changed -> overwrite with a centered sample
    //             1: static view, disk paused/off -> progressive average
    //             2: static view, disk animating  -> exponential moving avg
    int      sampleIndex = 0;    // samples already accumulated
    int      accumMode   = 0;

    // HDR bloom (applied after accumulation, before tone mapping)
    int      bloomEnabled  = 1;
    float    bloomStrength = 0.35f;
};
