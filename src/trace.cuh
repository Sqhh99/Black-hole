#pragma once
// ---------------------------------------------------------------------------
// trace.cuh  -- device-side ray tracing shared by the renderer and the tests
//
// Two integrators, one interface (traceRay):
//
//  * Spherically symmetric models (Schwarzschild, Reissner-Nordstrom):
//    every null geodesic lies in a plane through the origin; we integrate
//    the generalized Binet equation with classic RK4 in the orbital angle
//
//        d^2u/dphi^2 = 3 M u^2 - 2 Q^2 u^3 - u ,        u = 1/r .
//
//    For Q = 0 this is byte-for-byte the original Schwarzschild path, so the
//    baseline rendering quality is untouched.
//
//  * Rotating models (Kerr, Kerr-Newman): full 3D null geodesics in
//    Boyer-Lindquist coordinates. NOT a 2D equation plus a cosmetic twist:
//    we integrate the first-order Hamiltonian system for the state
//    (r, theta, phi, p_r, p_theta) with conserved E = -p_t, Lz = p_phi,
//
//        H = 1/(2 Sigma) [ Delta p_r^2 + p_theta^2 + W(r,theta) ] = 0,
//        W = (Lz - a E sin^2th)^2 / sin^2th - P(r)^2 / Delta,
//        P = E (r^2 + a^2) - a Lz,   Delta = r^2 - 2Mr + a^2 + Q^2,
//
//    using RK4 with an adaptive affine-parameter step: per-step budgets
//    proportional to the quality setting dPhi bound the angular advance and
//    the fractional approach to the horizon, extra refinement applies
//    inside the photon region, fine spatial caps apply only where the ray
//    can actually cross the accretion disk, and steps grow linearly with r
//    in the weak field. dp_r/dlambda and
//    dp_theta/dlambda use the FULL partial derivatives of H (including the
//    d(1/Sigma) terms proportional to the constraint K = 2 Sigma H), which
//    keeps the null constraint tightly conserved; the tests monitor
//    |K| / (E^2 (r^2+a^2)) along every ray.
//
//    Camera rays are mapped from the observer's local orthonormal frame
//    into curved spacetime through a *static-observer tetrad* (valid
//    outside the ergosphere, which the camera-distance clamp guarantees):
//    Euclidean pixel directions are never used directly as coordinate
//    derivatives.
//
// The accretion disk is sampled volumetrically along the geodesic in both
// paths. For the rotating models the combined gravitational + Doppler shift
// uses the exact relativistic factor  g = 1 / [ u^t (E - Omega Lz) ]  of a
// circular equatorial emitter, which encodes frame dragging, so redshift
// asymmetry responds to both spin and charge.
// ---------------------------------------------------------------------------
#include "render_params.h"
#include "metric.cuh"
#include "vec_math.cuh"
#include <cuda_runtime.h>

// Disk temperature at the inner edge [K-ish]
constexpr float BH_T_INNER = 9500.f;
constexpr float BH_PI      = 3.14159265358979f;

// ---------------------------------------------------------------------------
// Blackbody color (Tanner Helland style fit), returned in linear RGB.
// ---------------------------------------------------------------------------
__device__ inline float3 blackbodyRGB(float kelvin)
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
__device__ inline float3 starLayer(float3 d, float cellScale, float intensity)
{
    float3 p  = d * cellScale;
    float3 id = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f  = make_float3(p.x - id.x, p.y - id.y, p.z - id.z);

    float3 h  = hash33(id);
    float3 sp = make_float3(0.15f + 0.7f * h.x, 0.15f + 0.7f * h.y, 0.15f + 0.7f * h.z);
    float3 dv = f - sp;
    float  d2 = dot(dv, dv);

    float mag    = hash12(make_float2(id.x + 91.7f * id.z, id.y - 33.1f * id.x));
    float bright = 0.02f + 8.0f * powf(mag, 24.f);
    float core   = expf(-d2 * 260.f);

    float  tint = hash12(make_float2(id.y + 7.3f, id.z - 3.1f));
    float3 col  = lerp3(make_float3(0.72f, 0.82f, 1.0f),
                        make_float3(1.0f, 0.82f, 0.62f), tint);

    return col * (bright * core * intensity);
}

__device__ inline float3 backgroundColor(float3 dir)
{
    float3 d = normalize(dir);
    float3 c = make_float3(0.f, 0.f, 0.f);

    c += starLayer(d, 90.f,  1.0f);
    c += starLayer(d, 210.f, 0.45f);
    c += starLayer(d, 460.f, 0.18f);

    float az   = atan2f(d.z, d.x);
    float neb  = fbm(make_float2(az * 2.2f, d.y * 5.0f));
    float band = expf(-d.y * d.y * 14.f);
    c += make_float3(0.020f, 0.024f, 0.045f) * (neb * band);
    c += make_float3(0.004f, 0.005f, 0.009f);

    return c;
}

// ---------------------------------------------------------------------------
// Accretion disk sampling context.
//   gr = 0 : spherically symmetric shading (original special-relativistic
//            Doppler x gravitational redshift; identical to the baseline
//            for Schwarzschild).
//   gr = 1 : rotating models; exact g-factor from the photon's conserved
//            (E, Lz) and a circular equatorial emitter in Kerr-Newman.
// ---------------------------------------------------------------------------
struct DiskCtx
{
    int   gr;
    float E;    // photon conserved energy (camera-frame energy = 1)
    float Lz;   // photon conserved angular momentum
};

// ---------------------------------------------------------------------------
// Volumetric disk sampling along one geodesic segment [a, b] (Cartesian).
// `rayDir` points away from the camera along the backward ray.
// ---------------------------------------------------------------------------
__device__ inline void sampleDiskSegment(const RenderParams& P, const DiskCtx& ctx,
                                         float3 a, float3 b, float3 rayDir,
                                         float3& accum, float& trans)
{
    float3 seg    = b - a;
    float  segLen = length(seg);
    if (segLen < 1e-7f) return;

    float maxH = 0.20f * sqrtf(P.diskOuter) * 3.0f;
    if (a.y > maxH && b.y > maxH) return;
    if (a.y < -maxH && b.y < -maxH) return;

    // Radial culling: a segment cannot reach the disk annulus if both
    // endpoints are further than segLen outside it (cylindrical metric).
    float rcA = sqrtf(a.x * a.x + a.z * a.z);
    float rcB = sqrtf(b.x * b.x + b.z * b.z);
    float rcMin = fminf(rcA, rcB), rcMax = fmaxf(rcA, rcB);
    if (rcMin - segLen > P.diskOuter) return;
    if (rcMax + segLen < P.diskInner) return;

    int nSub = 1 + (int)(segLen / 0.15f);
    if (nSub > 12) nSub = 12;
    float ds = segLen / (float)nSub;

    for (int j = 0; j < nSub; ++j)
    {
        float  t  = ((float)j + 0.5f) / (float)nSub;
        float3 p  = lerp3(a, b, t);
        float  rc = sqrtf(p.x * p.x + p.z * p.z);      // cylindrical radius
        if (rc < P.diskInner || rc > P.diskOuter) continue;

        float H  = 0.20f * sqrtf(rc);
        float dz = p.y / H;
        if (fabsf(dz) > 3.0f) continue;
        float dens = expf(-dz * dz);

        dens *= smoothstepf(P.diskInner, P.diskInner * 1.18f, rc);
        dens *= 1.0f - smoothstepf(P.diskOuter * 0.72f, P.diskOuter, rc);

        // --- Relativistic g-factor + turbulence phase -------------------
        float g;
        float omega; // coordinate angular velocity, drives streak advection
        if (ctx.gr)
        {
            omega = bhOmegaCircular(rc, P.M, P.aSpin, P.Qc);
            if (omega <= 0.f) continue;
            float gtt, gtp, gpp;
            bhEquatorialMetric(rc, P.M, P.aSpin, P.Qc, gtt, gtp, gpp);
            float den = -(gtt + 2.f * omega * gtp + omega * omega * gpp);
            if (den < 1e-4f) continue;               // orbit not timelike
            float ut  = rsqrtf(den);
            float Eem = ut * (ctx.E - omega * ctx.Lz);
            if (Eem < 0.05f) Eem = 0.05f;
            g = clampf(1.0f / Eem, 0.05f, 4.0f);
        }
        else
        {
            // Original baseline shading (exact for Q = 0), generalized to
            // f(r) = 1 - 2M/r + Q^2/r^2 for Reissner-Nordstrom.
            float rr   = length(p);
            float w2   = P.M / rc - (P.Qc * P.Qc) / (rc * rc);
            float beta = sqrtf(fmaxf(w2, 0.f))
                       * rsqrtf(fmaxf(bhSphF(rc, P.M, P.Qc), 0.05f));
            beta = fminf(beta, 0.95f);
            float gamma = rsqrtf(1.0f - beta * beta);

            float3 vhat = normalize(make_float3(p.z, 0.f, -p.x));
            float3 nph  = -rayDir;

            float dopp  = 1.0f / (gamma * (1.0f - beta * dot(vhat, nph)));
            float ggrav = sqrtf(fmaxf(bhSphF(rr, P.M, P.Qc), 0.02f));
            g = dopp * ggrav;
            omega = sqrtf(fmaxf(P.M / (rc * rc * rc)
                              - (P.Qc * P.Qc) / (rc * rc * rc * rc), 0.f));
        }

        // Differentially rotating turbulence (azimuthal streaks). The two
        // paths use their own internally consistent azimuth convention.
        float azim = ctx.gr ? atan2f(-p.z, p.x) : atan2f(p.z, p.x);
        float azr  = azim - omega * P.diskTime;
        float n    = fbm(make_float2(rc * 3.1f, azr * 1.6f + rc * 0.7f));
        dens      *= (0.45f + 1.1f * n);
        if (dens < 1e-4f) continue;

        // --- Emission ----------------------------------------------------
        float Temit = BH_T_INNER * powf(P.diskInner / rc, 0.75f);
        float Tobs  = Temit * g;
        float3 col  = blackbodyRGB(Tobs);
        float emis  = powf(P.diskInner / rc, 3.0f) * (g * g) * (g * g);

        float w = dens * ds;
        accum += col * (emis * 7.0f * w) * trans;
        trans *= expf(-1.6f * w);
        if (trans < 0.01f) return;
    }
}

// Fused sin+cos (fast device intrinsic; libm on the host)
__device__ inline void bhSinCos(float x, float& s, float& c)
{
#ifdef __CUDA_ARCH__
    __sincosf(x, &s, &c);
#else
    s = sinf(x); c = cosf(x);
#endif
}

// ---------------------------------------------------------------------------
// Result of tracing one ray
// ---------------------------------------------------------------------------
struct TraceResult
{
    bool   horizon;
    bool   escaped;
    float3 escDir;    // deflected direction for background sampling
    float3 accum;     // disk radiance
    float  trans;     // remaining transmittance
    float  maxHviol;  // max normalized Hamiltonian-constraint violation
                      // (rotating path; 0 for the spherical path)
    int    steps;     // integration steps taken (perf diagnostics)
};

// Camera ray direction for a pixel center at fractional coords fx, fy in [0,1)
__device__ inline float3 cameraRayDir(const RenderParams& P, float fx, float fy)
{
    float nx = fx * 2.f - 1.f;
    float ny = 1.f - fy * 2.f;
    return normalize(P.camForward
                   + P.camRight * (nx * P.tanHalfFov * P.aspect)
                   + P.camUp    * (ny * P.tanHalfFov));
}

// ---------------------------------------------------------------------------
// Straight-line fallback for (nearly) exactly radial rays in the spherical
// path, where the orbital-plane basis is degenerate.
// ---------------------------------------------------------------------------
__device__ inline float3 radialRay(const RenderParams& P, const DiskCtx& ctx,
                                   float3 o, float3 d,
                                   float3& accum, float& trans, bool& horizon)
{
    float escR = 120.f * P.M;
    float step = 0.15f;
    float3 p = o;
    for (int i = 0; i < 800; ++i)
    {
        float3 q = p + d * step;
        if (P.diskEnabled) sampleDiskSegment(P, ctx, p, q, d, accum, trans);
        p = q;
        float r = length(p);
        if (r < P.rPlus) { horizon = true; return d; }
        if (r > escR)     return d;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Spherically symmetric integrator (Schwarzschild / Reissner-Nordstrom).
// Generalized Binet equation  u'' = 3 M u^2 - 2 Q^2 u^3 - u ; horizon at
// u >= 1/r+. For Q = 0 identical to the original Schwarzschild renderer.
// ---------------------------------------------------------------------------
__device__ inline TraceResult traceSpherical(const RenderParams& P,
                                             float3 origin, float3 dir)
{
    TraceResult R;
    R.horizon = false; R.escaped = false;
    R.escDir = dir;
    R.accum = make_float3(0.f, 0.f, 0.f);
    R.trans = 1.0f;
    R.maxHviol = 0.f;
    R.steps = 0;

    const DiskCtx ctx = {0, 0.f, 0.f};
    const float M  = P.M;
    const float Q2 = P.Qc * P.Qc;
    const float uHor    = 1.0f / P.rPlus;
    const float escR    = 120.f * M;
    const float uEscape = 1.0f / escR;

    float3 c  = origin;
    float  r0 = length(c);
    float3 e1 = c / r0;
    float  ddr = dot(dir, e1);
    float3 perp = dir - e1 * ddr;
    float  pl   = length(perp);

    if (pl < 1e-4f)
    {
        R.escDir  = radialRay(P, ctx, c, dir, R.accum, R.trans, R.horizon);
        R.escaped = !R.horizon;
        return R;
    }

    float3 e2 = perp / pl;

    // Static-observer initial conditions. The pixel direction is measured
    // in the camera's local orthonormal frame, where proper radial length
    // is dr/sqrt(f); hence dr/dphi = sqrt(f0) r0 d_r/d_perp and
    //   u(0)  = 1/r0,
    //   u'(0) = -sqrt(f(r0)) d_r / (r0 d_perp).
    // (Same physical convention as the tetrad used by the Kerr integrator;
    // Euclidean directions are never used as coordinate derivatives.)
    float f0  = fmaxf(bhSphF(r0, M, P.Qc), 1e-6f);
    float u   = 1.0f / r0;
    float du  = -sqrtf(f0) * ddr / (r0 * pl);
    float phi = 0.f;

    float3 prevPos = c;
    float3 dir3    = dir;
    float  h       = P.dPhi;

    // RHS of the generalized Binet equation
    auto acc = [M, Q2](float uu)
    {
        return 3.f * M * uu * uu - 2.f * Q2 * uu * uu * uu - uu;
    };

    int step = 0;
    for (; step < P.maxSteps; ++step)
    {
        float k1u = du;
        float k1v = acc(u);

        float u2 = u + 0.5f * h * k1u;
        float v2 = du + 0.5f * h * k1v;
        float k2u = v2;
        float k2v = acc(u2);

        float u3 = u + 0.5f * h * k2u;
        float v3 = du + 0.5f * h * k2v;
        float k3u = v3;
        float k3v = acc(u3);

        float u4 = u + h * k3u;
        float v4 = du + h * k3v;
        float k4u = v4;
        float k4v = acc(u4);

        u   += (h / 6.f) * (k1u + 2.f * k2u + 2.f * k3u + k4u);
        du  += (h / 6.f) * (k1v + 2.f * k2v + 2.f * k3v + k4v);
        phi += h;

        if (!isfinite(u) || !isfinite(du)) { R.horizon = true; break; }
        if (u >= uHor)                     { R.horizon = true; break; }

        float  r  = 1.0f / u;
        float  cp = cosf(phi), sp = sinf(phi);
        float3 m  = e1 * cp + e2 * sp;
        float3 mp = e1 * (-sp) + e2 * cp;
        float  drdphi = -du / (u * u);
        float3 pos = m * r;
        dir3 = normalize(m * drdphi + mp * r);

        if (P.diskEnabled && R.trans > 0.01f)
            sampleDiskSegment(P, ctx, prevPos, pos, dir3, R.accum, R.trans);
        prevPos = pos;

        if (u < uEscape && du < 0.f) { R.escaped = true; R.escDir = dir3; break; }
    }
    R.steps = step;

    if (!R.horizon && !R.escaped)
    {
        // Step budget exhausted: deep strong-field rays count as captured
        if (u > 1.0f / (6.f * M)) R.horizon = true;
        else { R.escaped = true; R.escDir = dir3; }
    }
    return R;
}

// ---------------------------------------------------------------------------
// Kerr / Kerr-Newman: Boyer-Lindquist Hamiltonian integrator
// ---------------------------------------------------------------------------
struct KState { float r, th, ph, pr, pth; };

// Full right-hand side of the first-order geodesic system, plus the
// constraint value K = 2 Sigma H (== 0 analytically for null rays).
__device__ inline KState kerrRHS(float M, float a, float Q, float E, float Lz,
                                 const KState& y, float& Kout)
{
    float s, cth;
    bhSinCos(y.th, s, cth);
    if (s < 1e-4f) s = 1e-4f;                 // pole guard (Lz != 0 barrier)
    float s2 = s * s;

    float r  = y.r;
    float r2 = r * r, a2 = a * a;
    float Sig = r2 + a2 * cth * cth;
    float Del = r2 - 2.f * M * r + a2 + Q * Q;
    if (Del < 1e-6f) Del = 1e-6f;             // only reachable at termination

    float Pp = E * (r2 + a2) - a * Lz;
    float B  = Lz / s - a * E * s;            // so B*B = (Lz - aE s^2)^2/s^2
    float W  = B * B - Pp * Pp / Del;

    float K = Del * y.pr * y.pr + y.pth * y.pth + W;
    Kout = K;

    float dDel = 2.f * r - 2.f * M;
    float dP   = 2.f * r * E;
    float Wr   = -(2.f * Pp * dP * Del - Pp * Pp * dDel) / (Del * Del);
    float dB   = -Lz * cth / s2 - a * E * cth;
    float Wth  = 2.f * B * dB;

    float invS  = 1.f / Sig;
    float invS2 = invS * invS;

    KState d;
    d.r   = Del * y.pr * invS;
    d.th  = y.pth * invS;
    d.ph  = ((Lz / s2 - a * E) + a * Pp / Del) * invS;
    // dp_r/dl  = -dH/dr,  dH/dr = (dDel pr^2 + Wr)/(2 Sig) - (r/Sig^2) K
    d.pr  = -0.5f * invS * (dDel * y.pr * y.pr + Wr) + r * invS2 * K;
    // dp_th/dl = -dH/dth, Sig_th = -2 a^2 s cth
    d.pth = -0.5f * invS * Wth - a2 * s * cth * invS2 * K;
    return d;
}

// Boyer-Lindquist (r, th, ph) -> pseudo-Cartesian embedding, chosen so that
// increasing ph matches the prograde disk-rotation direction of the
// baseline renderer (spin axis = +Y):
//   x = r sin th cos ph,  y = r cos th,  z = -r sin th sin ph.
__device__ inline float3 kerrToCart(float r, float th, float ph)
{
    float s, c, sp, cp;
    bhSinCos(th, s, c);
    bhSinCos(ph, sp, cp);
    return make_float3(r * s * cp, r * c, -r * s * sp);
}

__device__ inline void kerrBasis(float th, float ph,
                                 float3& er, float3& eth, float3& eph)
{
    float s, c, sp, cp;
    bhSinCos(th, s, c);
    bhSinCos(ph, sp, cp);
    er  = make_float3(s * cp,  c, -s * sp);
    eth = make_float3(c * cp, -s, -c * sp);
    eph = make_float3(-sp, 0.f, -cp);
}

__device__ inline TraceResult traceKerr(const RenderParams& P,
                                        float3 origin, float3 dir,
                                        bool wantStats)
{
    TraceResult R;
    R.horizon = false; R.escaped = false;
    R.escDir = dir;
    R.accum = make_float3(0.f, 0.f, 0.f);
    R.trans = 1.0f;
    R.maxHviol = 0.f;
    R.steps = 0;

    const float M = P.M, a = P.aSpin, Q = P.Qc;
    const float rp   = P.rPlus;
    const float escR = 120.f * M;

    // ---- Boyer-Lindquist coordinates of the camera -----------------------
    float r0 = length(origin);
    float cth0 = clampf(origin.y / r0, -1.f, 1.f);
    float th0  = acosf(cth0);
    float ph0  = atan2f(-origin.z, origin.x);

    float3 er, eth, eph;
    kerrBasis(th0, ph0, er, eth, eph);
    float drc = dot(dir, er);
    float dtc = dot(dir, eth);
    float dpc = dot(dir, eph);

    // ---- Static-observer tetrad -> photon 4-momentum ---------------------
    float s0  = sinf(th0); if (s0 < 1e-4f) s0 = 1e-4f;
    float s02 = s0 * s0;
    float Sig0 = bhSigma(r0, cth0, a);
    float Del0 = bhDelta(r0, M, a, Q);
    float m2r0 = 2.f * M * r0 - Q * Q;

    float gtt = -(1.f - m2r0 / Sig0);
    if (gtt > -1e-5f) gtt = -1e-5f;        // camera outside ergosphere anyway
    float gtp = -a * m2r0 * s02 / Sig0;
    float gpp = (r0 * r0 + a * a + m2r0 * a * a * s02 / Sig0) * s02;

    float A    = rsqrtf(fmaxf(gpp - gtp * gtp / gtt, 1e-8f));
    float ptUp = rsqrtf(-gtt) - dpc * A * gtp / gtt; // p^t
    float pfUp = dpc * A;                            // p^phi

    float E  = -(gtt * ptUp + gtp * pfUp);
    float Lz = gtp * ptUp + gpp * pfUp;

    KState y;
    y.r   = r0;
    y.th  = th0;
    y.ph  = ph0;
    y.pr  = drc * sqrtf(Sig0 / fmaxf(Del0, 1e-6f)); // p_r  = (Sig/Del) p^r
    y.pth = dtc * sqrtf(Sig0);                      // p_th = Sig p^theta

    const DiskCtx ctx = {1, E, Lz};
    const float E2 = E * E;
    const bool  trackPos = (P.diskEnabled != 0);

    float3 prevPos = origin;
    float3 dir3    = dir;
    KState dLast{};
    bool   haveDeriv = false;
    int    step = 0;

    for (; step < P.maxSteps; ++step)
    {
        float Kv;
        KState d1 = kerrRHS(M, a, Q, E, Lz, y, Kv);

        if (!isfinite(y.r) || !isfinite(y.pr) || !isfinite(y.pth) ||
            !isfinite(d1.r) || !isfinite(d1.pr))
        { R.horizon = true; break; }

        if (y.r <= rp * 1.002f + 1e-3f) { R.horizon = true; break; }

        // Constraint monitor. Skipped in the immediate vicinity of the
        // horizon, where Delta -> 0 makes P^2/Delta a catastrophic float
        // cancellation: those rays are captured within a step or two and
        // the spike is a property of the diagnostic, not of the orbit.
        if (wantStats && y.r > rp * 1.05f)
        {
            float viol = fabsf(Kv) / (E2 * (y.r * y.r + a * a) + 1e-12f);
            if (viol > R.maxHviol) R.maxHviol = viol;
        }

        if (y.r >= escR && d1.r > 0.f)
        {
            float3 e1b, e2b, e3b;
            kerrBasis(y.th, y.ph, e1b, e2b, e3b);
            float s = sinf(y.th);
            R.escDir = normalize(e1b * d1.r + e2b * (y.r * d1.th)
                                            + e3b * (y.r * s * d1.ph));
            R.escaped = true;
            break;
        }

        // ---- Adaptive affine step --------------------------------------
        // Per-step budgets, both scaling with the quality preset: the
        // fractional shrink of (r - r+) is bounded by ~8*dPhi and the
        // angular advance by ~1.6*dPhi rad (comparable to the spherical
        // integrator's fixed step, RK4 in both). Extra refinement inside
        // the photon region. The spatial advance is capped tightly only
        // inside the radial band the accretion disk occupies; in the weak
        // field it grows linearly with r, so far-field flight costs only
        // a handful of steps instead of dozens.
        float h = 1.0f / (fabsf(d1.r) / (8.0f * P.dPhi * fmaxf(y.r - rp, 0.02f))
                        + (fabsf(d1.th) + fabsf(d1.ph)) / (2.2f * P.dPhi)
                        + 1e-5f);
        if (y.r < 1.5f * P.rPhoton) h *= 0.55f;
        float spd = fabsf(d1.r) + y.r * (fabsf(d1.th) + fabsf(d1.ph)) + 1e-6f;
        float capLen;
        if (trackPos && y.r < P.diskOuter * 1.3f)
        {
            // Fine sampling only where the ray can actually be inside the
            // disk slab (|y| < ~3H_max); segments approaching from above or
            // below use a medium cap so the slab entry is still subsampled
            // finely by sampleDiskSegment.
            float yAbs = y.r * fabsf(cosf(y.th));
            capLen = (yAbs < 3.5f) ? 1.2f : 2.0f;
        }
        else capLen = fmaxf(1.2f, 0.22f * y.r);
        h = fminf(h, capLen / spd);

        // ---- Classic RK4 -----------------------------------------------
        float kv2, kv3, kv4;
        KState y2 = { y.r  + 0.5f * h * d1.r,  y.th  + 0.5f * h * d1.th,
                      y.ph + 0.5f * h * d1.ph, y.pr  + 0.5f * h * d1.pr,
                      y.pth + 0.5f * h * d1.pth };
        KState d2 = kerrRHS(M, a, Q, E, Lz, y2, kv2);
        KState y3 = { y.r  + 0.5f * h * d2.r,  y.th  + 0.5f * h * d2.th,
                      y.ph + 0.5f * h * d2.ph, y.pr  + 0.5f * h * d2.pr,
                      y.pth + 0.5f * h * d2.pth };
        KState d3 = kerrRHS(M, a, Q, E, Lz, y3, kv3);
        KState y4 = { y.r  + h * d3.r,  y.th  + h * d3.th,
                      y.ph + h * d3.ph, y.pr  + h * d3.pr,
                      y.pth + h * d3.pth };
        KState d4 = kerrRHS(M, a, Q, E, Lz, y4, kv4);

        float h6 = h / 6.f;
        y.r   += h6 * (d1.r   + 2.f * d2.r   + 2.f * d3.r   + d4.r);
        y.th  += h6 * (d1.th  + 2.f * d2.th  + 2.f * d3.th  + d4.th);
        y.ph  += h6 * (d1.ph  + 2.f * d2.ph  + 2.f * d3.ph  + d4.ph);
        y.pr  += h6 * (d1.pr  + 2.f * d2.pr  + 2.f * d3.pr  + d4.pr);
        y.pth += h6 * (d1.pth + 2.f * d2.pth + 2.f * d3.pth + d4.pth);

        // Pole reflection guard (only reachable for |Lz| ~ 0 rays)
        if (y.th < 1e-3f)            { y.th = 1e-3f;          y.pth =  fabsf(y.pth); }
        else if (y.th > BH_PI - 1e-3f) { y.th = BH_PI - 1e-3f; y.pth = -fabsf(y.pth); }

        dLast = d4;
        haveDeriv = true;

        // ---- Disk sampling along the Cartesian chord --------------------
        // (skipped entirely when the disk is off: no per-step Cartesian
        // conversion is needed then; escape directions are reconstructed
        // from the coordinate derivatives instead.)
        if (trackPos)
        {
            float3 pos = kerrToCart(y.r, y.th, y.ph);
            float3 stepv = pos - prevPos;
            float  sl = length(stepv);
            if (sl > 1e-6f) dir3 = stepv * (1.f / sl);
            if (R.trans > 0.01f)
                sampleDiskSegment(P, ctx, prevPos, pos, dir3, R.accum, R.trans);
            prevPos = pos;
        }
    }
    R.steps = step;

    if (!R.horizon && !R.escaped)
    {
        // Step budget exhausted
        if (y.r < 6.f * M) R.horizon = true;
        else
        {
            R.escaped = true;
            if (haveDeriv)
            {
                float3 e1b, e2b, e3b;
                kerrBasis(y.th, y.ph, e1b, e2b, e3b);
                float s = sinf(y.th);
                R.escDir = normalize(e1b * dLast.r + e2b * (y.r * dLast.th)
                                                   + e3b * (y.r * s * dLast.ph));
            }
            else R.escDir = dir3;
        }
    }
    return R;
}

// ---------------------------------------------------------------------------
// Unified entry point
// ---------------------------------------------------------------------------
__device__ inline TraceResult traceRay(const RenderParams& P,
                                       float3 origin, float3 dir,
                                       bool wantStats)
{
    if (P.model == BH_KERR || P.model == BH_KERR_NEWMAN)
        return traceKerr(P, origin, dir, wantStats);
    return traceSpherical(P, origin, dir);
}

// ---------------------------------------------------------------------------
// One linear-HDR sample for pixel (px, py) with sub-pixel offset (jx, jy).
// Pre-exposure: exposure/bloom/tone mapping happen after accumulation.
// ---------------------------------------------------------------------------
__device__ inline float3 renderSampleHDR(const RenderParams& P,
                                         int px, int py, float jx, float jy)
{
    float fx = ((float)px + jx) / (float)P.width;
    float fy = ((float)py + jy) / (float)P.height;
    float3 dir = cameraRayDir(P, fx, fy);

    TraceResult t = traceRay(P, P.camPos, dir, false);

    float3 bg  = t.horizon ? make_float3(0.f, 0.f, 0.f)
                           : backgroundColor(t.escDir);
    float3 hdr = t.accum + bg * t.trans;

    // Never let a stray non-finite sample poison the accumulation buffer.
    if (!isfinite(hdr.x) || !isfinite(hdr.y) || !isfinite(hdr.z))
        hdr = make_float3(0.f, 0.f, 0.f);
    return hdr;
}
