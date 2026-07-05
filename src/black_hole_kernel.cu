// ---------------------------------------------------------------------------
// black_hole_kernel.cu
//
// Real-time Schwarzschild black hole renderer.
//
// Physics model (geometric units, Schwarzschild radius rs = 1, so M = 0.5):
//   * In Schwarzschild spacetime every null geodesic is confined to a plane
//     through the origin. For each camera ray we build an orthonormal basis
//     (e1, e2) of that plane and integrate the Binet equation
//
//         d^2 u / d phi^2 = (3/2) * rs * u^2 - u ,   u = 1/r
//
//     with classic RK4 in the orbital angle phi. This is an exact null
//     geodesic equation (not a screen-space distortion hack).
//   * Termination: horizon capture (u >= 1/rs), far-field escape
//     (r > ESCAPE_R while moving outward), NaN/Inf blow-up, or step budget
//     exhaustion.
//   * A finite-thickness accretion disk in the equatorial plane is sampled
//     volumetrically along each geodesic segment: Gaussian vertical density,
//     Shakura-Sunyaev-like temperature profile T ~ r^(-3/4), Keplerian gas
//     velocity, special-relativistic Doppler factor + gravitational redshift
//     combined into g, with observed intensity scaled by g^4 and observed
//     color taken from a blackbody at T_obs = g * T_emit. This produces the
//     characteristic bright approaching side / dim receding side.
//   * Escaped rays are shaded with a procedural starfield, which therefore
//     shows continuous gravitational lensing; the photon ring emerges
//     naturally from rays that wind near r = 1.5 rs.
//   * HDR accumulation in linear float3, ACES filmic tone mapping, gamma 2.2.
// ---------------------------------------------------------------------------

#include "render_params.h"
#include "vec_math.cuh"
#include <cuda_runtime.h>

namespace
{
constexpr float RS        = 1.0f;   // Schwarzschild radius
constexpr float ESCAPE_R  = 60.0f;  // far-field radius
constexpr float U_ESCAPE  = 1.0f / ESCAPE_R;
constexpr float T_INNER   = 9500.f; // disk temperature at the inner edge [K-ish]
}

// ---------------------------------------------------------------------------
// Blackbody color (Tanner Helland style fit), returned in linear RGB, max ~1.
// ---------------------------------------------------------------------------
__device__ float3 blackbodyRGB(float kelvin)
{
    float t = clampf(kelvin, 1200.f, 40000.f) / 100.f;
    float r, g, b;

    if (t <= 66.f) { r = 255.f; }
    else           { r = 329.698727446f * powf(t - 60.f, -0.1332047592f); }

    if (t <= 66.f) { g = 99.4708025861f * logf(t) - 161.1195681661f; }
    else           { g = 288.1221695283f * powf(t - 60.f, -0.0755148492f); }

    if (t >= 66.f)      { b = 255.f; }
    else if (t <= 19.f) { b = 0.f;   }
    else                { b = 138.5177312231f * logf(t - 10.f) - 305.0447927307f; }

    float3 srgb = make_float3(clampf(r, 0.f, 255.f) / 255.f,
                              clampf(g, 0.f, 255.f) / 255.f,
                              clampf(b, 0.f, 255.f) / 255.f);
    return pow3(srgb, 2.2f); // approximate linearization
}

// ---------------------------------------------------------------------------
// Procedural starfield + faint nebula, evaluated on the *deflected* escape
// direction => the background exhibits the gravitational lensing field.
// ---------------------------------------------------------------------------
__device__ float3 starLayer(float3 d, float cellScale, float intensity)
{
    float3 p  = d * cellScale;
    float3 id = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f  = make_float3(p.x - id.x, p.y - id.y, p.z - id.z);

    float3 h  = hash33(id);
    // Star position inside the cell
    float3 sp = make_float3(0.15f + 0.7f * h.x, 0.15f + 0.7f * h.y, 0.15f + 0.7f * h.z);
    float3 dv = f - sp;
    float  d2 = dot(dv, dv);

    // Power-law magnitude distribution: many dim stars, few bright ones
    float mag    = hash12(make_float2(id.x + 91.7f * id.z, id.y - 33.1f * id.x));
    float bright = 0.02f + 8.0f * powf(mag, 24.f);

    float core = expf(-d2 * 260.f);

    // Slight color temperature variation per star
    float  tint = hash12(make_float2(id.y + 7.3f, id.z - 3.1f));
    float3 col  = lerp3(make_float3(0.72f, 0.82f, 1.0f),   // blue-white
                        make_float3(1.0f, 0.82f, 0.62f),   // warm
                        tint);

    return col * (bright * core * intensity);
}

__device__ float3 backgroundColor(float3 dir)
{
    float3 d = normalize(dir);
    float3 c = make_float3(0.f, 0.f, 0.f);

    c += starLayer(d, 90.f,  1.0f);
    c += starLayer(d, 210.f, 0.45f);
    c += starLayer(d, 460.f, 0.18f);

    // Faint band of "galactic" nebulosity
    float az  = atan2f(d.z, d.x);
    float neb = fbm(make_float2(az * 2.2f, d.y * 5.0f));
    float band = expf(-d.y * d.y * 14.f);
    c += make_float3(0.020f, 0.024f, 0.045f) * (neb * band);
    c += make_float3(0.004f, 0.005f, 0.009f); // deep-space floor

    return c;
}

// ---------------------------------------------------------------------------
// Accretion disk volumetric sampling along one geodesic segment [a, b].
// `rayDir` is the (approximate) photon propagation direction of the backward
// ray at this segment, pointing AWAY from the camera; the physical photon
// travels toward the camera along -rayDir.
// Accumulates emission into `accum` and attenuates `trans` (transmittance).
// ---------------------------------------------------------------------------
__device__ void sampleDiskSegment(const RenderParams& P,
                                  float3 a, float3 b, float3 rayDir,
                                  float3& accum, float& trans)
{
    float3 seg    = b - a;
    float  segLen = length(seg);
    if (segLen < 1e-7f) return;

    // Quick reject: segment entirely far above/below the disk slab
    float maxH = 0.20f * sqrtf(P.diskOuter) * 3.0f;
    if (a.y > maxH && b.y > maxH) return;
    if (a.y < -maxH && b.y < -maxH) return;

    int nSub = 1 + (int)(segLen / 0.15f);
    if (nSub > 12) nSub = 12;
    float ds = segLen / (float)nSub;

    for (int j = 0; j < nSub; ++j)
    {
        float  t  = ((float)j + 0.5f) / (float)nSub;
        float3 p  = lerp3(a, b, t);
        float  rc = sqrtf(p.x * p.x + p.z * p.z);      // cylindrical radius
        if (rc < P.diskInner || rc > P.diskOuter) continue;

        // Finite thickness: flared Gaussian slab, scale height H(rc)
        float H  = 0.20f * sqrtf(rc);
        float dz = p.y / H;
        if (fabsf(dz) > 3.0f) continue;
        float dens = expf(-dz * dz);

        // Soft radial edges
        dens *= smoothstepf(P.diskInner, P.diskInner * 1.18f, rc);
        dens *= 1.0f - smoothstepf(P.diskOuter * 0.72f, P.diskOuter, rc);

        // Differentially rotating turbulence (azimuthal streaks)
        float azim  = atan2f(p.z, p.x);
        float omega = sqrtf(0.5f) * powf(rc, -1.5f);   // Keplerian, M = 0.5
        float azr   = azim - omega * P.diskTime;
        float n     = fbm(make_float2(rc * 3.1f, azr * 1.6f + rc * 0.7f));
        dens       *= (0.45f + 1.1f * n);
        if (dens < 1e-4f) continue;

        // --- Relativistic factors -------------------------------------
        float rr = length(p);
        // Locally measured Keplerian orbital speed (clamped for stability)
        float beta = sqrtf(0.5f / rc) * rsqrtf(fmaxf(1.0f - RS / rc, 0.05f));
        beta = fminf(beta, 0.95f);
        float gamma = rsqrtf(1.0f - beta * beta);

        // Gas velocity direction (prograde around +Y)
        float3 vhat = normalize(make_float3(p.z, 0.f, -p.x));
        // Photon direction toward the observer
        float3 nph  = -rayDir;

        float dopp  = 1.0f / (gamma * (1.0f - beta * dot(vhat, nph)));
        float ggrav = sqrtf(fmaxf(1.0f - RS / rr, 0.02f));
        float g     = dopp * ggrav;

        // --- Emission --------------------------------------------------
        // Shakura-Sunyaev-like temperature profile T ~ r^{-3/4}
        float Temit = T_INNER * powf(P.diskInner / rc, 0.75f);
        float Tobs  = Temit * g;
        float3 col  = blackbodyRGB(Tobs);

        // Fluid-frame emissivity ~ T^4  =>  (r_in / r)^3 ; observed ~ g^4
        float emis = powf(P.diskInner / rc, 3.0f) * (g * g) * (g * g);

        float w = dens * ds;
        accum += col * (emis * 7.0f * w) * trans;
        trans *= expf(-1.6f * w);            // self-absorption
        if (trans < 0.01f) return;
    }
}

// ---------------------------------------------------------------------------
// Straight-line fallback for (nearly) exactly radial rays, where the plane
// basis is degenerate. Physically correct: radial null rays are straight in
// these coordinates for our purposes; capture iff pointing inward.
// ---------------------------------------------------------------------------
__device__ float3 radialRay(const RenderParams& P, float3 o, float3 d,
                            float3& accum, float& trans, bool& horizon)
{
    float step = 0.15f;
    float3 p = o;
    for (int i = 0; i < 800; ++i)
    {
        float3 q = p + d * step;
        if (P.diskEnabled) sampleDiskSegment(P, p, q, d, accum, trans);
        p = q;
        float r = length(p);
        if (r < RS)      { horizon = true; return d; }
        if (r > ESCAPE_R) return d;
    }
    return d;
}

// ---------------------------------------------------------------------------
// ACES filmic tone map (Narkowicz fit)
// ---------------------------------------------------------------------------
__device__ float3 acesToneMap(float3 x)
{
    const float A = 2.51f, B = 0.03f, C = 2.43f, D = 0.59f, E = 0.14f;
    float3 num = x * (x * A + make_float3(B, B, B));
    float3 den = x * (x * C + make_float3(D, D, D)) + make_float3(E, E, E);
    return clamp3(make_float3(num.x / den.x, num.y / den.y, num.z / den.z), 0.f, 1.f);
}

// ---------------------------------------------------------------------------
// Main per-pixel kernel
// ---------------------------------------------------------------------------
__global__ void renderKernel(uchar4* out, RenderParams P)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= P.width || py >= P.height) return;

    // Primary ray
    float nx = ((px + 0.5f) / (float)P.width) * 2.f - 1.f;
    float ny = 1.f - ((py + 0.5f) / (float)P.height) * 2.f;
    float3 dir = normalize(P.camForward
                         + P.camRight * (nx * P.tanHalfFov * P.aspect)
                         + P.camUp    * (ny * P.tanHalfFov));

    float3 accum = make_float3(0.f, 0.f, 0.f);
    float  trans = 1.0f;
    bool   horizon = false;
    bool   escaped = false;
    float3 escDir  = dir;

    // Plane-of-motion basis
    float3 c   = P.camPos;
    float  r0  = length(c);
    float3 e1  = c / r0;
    float  ddr = dot(dir, e1);                 // radial component of ray dir
    float3 perp = dir - e1 * ddr;
    float  pl   = length(perp);

    if (pl < 1e-4f)
    {
        // Degenerate (radial) ray
        escDir = radialRay(P, c, dir, accum, trans, horizon);
        escaped = !horizon;
    }
    else
    {
        float3 e2 = perp / pl;

        // Binet initial conditions: u(0) = 1/r0, u'(0) = -d_r / (r0 * d_perp)
        float u  = 1.0f / r0;
        float du = -ddr / (r0 * pl);
        float phi = 0.f;

        float3 prevPos = c;
        float3 dir3    = dir;
        float  h       = P.dPhi;

        for (int i = 0; i < P.maxSteps; ++i)
        {
            // ---- RK4 on (u, du): u' = du ; du' = 1.5*rs*u^2 - u ----
            float k1u = du;
            float k1v = 1.5f * RS * u * u - u;

            float u2 = u + 0.5f * h * k1u;
            float v2 = du + 0.5f * h * k1v;
            float k2u = v2;
            float k2v = 1.5f * RS * u2 * u2 - u2;

            float u3 = u + 0.5f * h * k2u;
            float v3 = du + 0.5f * h * k2v;
            float k3u = v3;
            float k3v = 1.5f * RS * u3 * u3 - u3;

            float u4 = u + h * k3u;
            float v4 = du + h * k3v;
            float k4u = v4;
            float k4v = 1.5f * RS * u4 * u4 - u4;

            u   += (h / 6.f) * (k1u + 2.f * k2u + 2.f * k3u + k4u);
            du  += (h / 6.f) * (k1v + 2.f * k2v + 2.f * k3v + k4v);
            phi += h;

            // ---- Robust termination ----
            if (!isfinite(u) || !isfinite(du)) { horizon = true; break; }
            if (u >= 1.0f / RS)                { horizon = true; break; }

            // Reconstruct 3D state
            float  r  = 1.0f / u;
            float  cp = cosf(phi), sp = sinf(phi);
            float3 m  = e1 * cp + e2 * sp;
            float3 mp = e1 * (-sp) + e2 * cp;
            float  drdphi = -du / (u * u);
            float3 pos = m * r;
            dir3 = normalize(m * drdphi + mp * r);

            // Accretion disk sampling along this segment
            if (P.diskEnabled && trans > 0.01f)
                sampleDiskSegment(P, prevPos, pos, dir3, accum, trans);
            prevPos = pos;

            if (u < U_ESCAPE && du < 0.f) { escaped = true; escDir = dir3; break; }
        }

        if (!horizon && !escaped)
        {
            // Step budget exhausted. Rays still trapped deep in the strong
            // field are treated as captured; others continue straight.
            if (u > 1.0f / 3.0f) horizon = true;
            else { escaped = true; escDir = dir3; }
        }
    }

    float3 bg  = horizon ? make_float3(0.f, 0.f, 0.f) : backgroundColor(escDir);
    float3 hdr = accum + bg * trans;

    // ---- HDR -> LDR: exposure, ACES, gamma 2.2 ----
    float3 ldr = acesToneMap(hdr * P.exposure);
    ldr = pow3(ldr, 1.0f / 2.2f);

    unsigned char r8 = (unsigned char)(clampf(ldr.x, 0.f, 1.f) * 255.f + 0.5f);
    unsigned char g8 = (unsigned char)(clampf(ldr.y, 0.f, 1.f) * 255.f + 0.5f);
    unsigned char b8 = (unsigned char)(clampf(ldr.z, 0.f, 1.f) * 255.f + 0.5f);

    uchar4 pix = P.swapRB ? make_uchar4(b8, g8, r8, 255)
                          : make_uchar4(r8, g8, b8, 255);
    out[py * P.width + px] = pix;
}

// ---------------------------------------------------------------------------
// C-linkage launcher used by the host code
// ---------------------------------------------------------------------------
extern "C" cudaError_t launchRenderKernel(uchar4* out, const RenderParams& p,
                                          cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((p.width + block.x - 1) / block.x,
              (p.height + block.y - 1) / block.y);
    renderKernel<<<grid, block, 0, stream>>>(out, p);
    return cudaGetLastError();
}
