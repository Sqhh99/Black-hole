#pragma once
// ---------------------------------------------------------------------------
// metric.cuh
//
// Black hole family: Kerr-Newman metric in Boyer-Lindquist coordinates and
// its special cases. Geometric units G = c = 1, unified mass parameter M
// (default M = 0.5 so that the Schwarzschild radius rs = 2M = 1 code unit,
// preserving the original renderer's length scale and visuals exactly).
//
//   model 0  Schwarzschild   a* = 0, q  = 0
//   model 1  Reissner-Nordstrom a* = 0, q != 0
//   model 2  Kerr            a* != 0, q  = 0
//   model 3  Kerr-Newman     a* != 0, q != 0
//
// Dimensionless spin a* = a/M, dimensionless charge q = Q/M.
// Outer horizon r+ = M + sqrt(M^2 - a^2 - Q^2)  (each special case reduces
// to the standard formula). Naked-singularity parameters (a*^2 + q^2 > 1)
// are rejected/clamped host-side; the integrators additionally guard against
// NaN/Inf so no invalid parameter can produce flicker or a crash.
//
// This header is compiled both by NVCC (device+host) and by MSVC (host only,
// via main.cpp and the test driver).
// ---------------------------------------------------------------------------
#include <cmath>

#ifdef __CUDACC__
#define BH_HD __host__ __device__ inline
#else
#define BH_HD inline
#endif

enum BlackHoleModel
{
    BH_SCHWARZSCHILD = 0,
    BH_REISSNER_NORDSTROM = 1,
    BH_KERR = 2,
    BH_KERR_NEWMAN = 3
};

inline const char* bhModelName(int m)
{
    switch (m)
    {
    case BH_SCHWARZSCHILD:      return "Schwarzschild";
    case BH_REISSNER_NORDSTROM: return "Reissner-Nordstrom";
    case BH_KERR:               return "Kerr";
    case BH_KERR_NEWMAN:        return "Kerr-Newman";
    default:                    return "unknown";
    }
}

// Maximum allowed a*^2 + q^2 (margin below the extremal limit 1).
constexpr float BH_EXTREMAL_LIMIT = 0.995f;

// ---------------------------------------------------------------------------
// Elementary metric quantities (float; used on both host and device)
// ---------------------------------------------------------------------------
BH_HD float bhSigma(float r, float cth, float a)
{
    return r * r + a * a * cth * cth;
}

BH_HD float bhDelta(float r, float M, float a, float Q)
{
    return r * r - 2.f * M * r + a * a + Q * Q;
}

// Outer horizon r+ = M + sqrt(M^2 - a^2 - Q^2); caller guarantees validity.
BH_HD float bhHorizonOuter(float M, float a, float Q)
{
    float d = M * M - a * a - Q * Q;
    if (d < 0.f) d = 0.f;
    return M + sqrtf(d);
}

BH_HD float bhHorizonInner(float M, float a, float Q)
{
    float d = M * M - a * a - Q * Q;
    if (d < 0.f) d = 0.f;
    return M - sqrtf(d);
}

// Spherical (a = 0) metric function f(r) = 1 - 2M/r + Q^2/r^2
BH_HD float bhSphF(float r, float M, float Q)
{
    return 1.f - 2.f * M / r + Q * Q / (r * r);
}

// Photon sphere for the spherically symmetric case:
//   r_ph = (3M + sqrt(9M^2 - 8Q^2)) / 2
BH_HD float bhPhotonSphereSpherical(float M, float Q)
{
    float d = 9.f * M * M - 8.f * Q * Q;
    if (d < 0.f) d = 0.f;
    return 0.5f * (3.f * M + sqrtf(d));
}

// Prograde equatorial circular photon orbit of Kerr (Bardeen):
//   r_ph = 2M { 1 + cos[ (2/3) arccos(-a*) ] }
// Used as a strong-field marker for step refinement; for Kerr-Newman it is
// approximate (charge correction ignored) and clamped outside the horizon.
BH_HD float bhPhotonOrbitKerrPrograde(float M, float aStar)
{
    float x = aStar;
    if (x >  1.f) x =  1.f;
    if (x < -1.f) x = -1.f;
    return 2.f * M * (1.f + cosf((2.f / 3.f) * acosf(-x)));
}

// Equatorial ergosphere radius r_E = M + sqrt(M^2 - Q^2)
BH_HD float bhErgoEquatorial(float M, float Q)
{
    float d = M * M - Q * Q;
    if (d < 0.f) d = 0.f;
    return M + sqrtf(d);
}

// Coordinate angular velocity of a prograde equatorial circular orbit in
// Kerr-Newman:  Omega = sqrt(M r - Q^2) / ( r^2 + a sqrt(M r - Q^2) ).
// Reduces to sqrt(M/r^3) (Schwarzschild/Kerr, Q=0) and to
// sqrt(M/r^3 - Q^2/r^4) (Reissner-Nordstrom, a=0).
// Returns 0 if no circular orbit exists at r (M r <= Q^2).
BH_HD float bhOmegaCircular(float r, float M, float a, float Q)
{
    float w2 = M * r - Q * Q;
    if (w2 <= 0.f) return 0.f;
    float sw = sqrtf(w2);
    return sw / (r * r + a * sw);
}

// Equatorial metric components (theta = pi/2, sin = 1) of Kerr-Newman.
BH_HD void bhEquatorialMetric(float r, float M, float a, float Q,
                              float& gtt, float& gtp, float& gpp)
{
    float r2  = r * r;
    float m2r = 2.f * M * r - Q * Q;   // Sigma = r^2 in the equator
    gtt = -(1.f - m2r / r2);
    gtp = -a * m2r / r2;
    gpp = r2 + a * a + m2r * a * a / r2;
}

// ---------------------------------------------------------------------------
// Host-side derived parameters, validation and ISCO
// ---------------------------------------------------------------------------
struct BHDerived
{
    float a       = 0.f;   // spin,  code units (a = a* M)
    float Q       = 0.f;   // charge, code units (Q = q M)
    float rPlus   = 1.f;   // outer horizon
    float rMinus  = 0.f;   // inner horizon
    float rPhoton = 1.5f;  // photon sphere / prograde photon orbit
    float rErgo   = 1.f;   // equatorial ergosphere radius
    float rIsco   = 3.f;   // innermost stable circular orbit (numeric)
    bool  clamped = false; // true if inputs had to be modified
};

// Numeric ISCO for equatorial prograde circular orbits of Kerr-Newman.
// Scans the specific orbital energy
//     E~(r) = -(g_tt + Omega g_tphi) / sqrt(-(g_tt + 2 Omega g_tphi
//                                             + Omega^2 g_phiphi))
// whose minimum over r marks marginal stability. Verified limits:
// Schwarzschild -> 6M; Kerr a*->1 (prograde) -> M.
inline float bhIscoNumeric(float M, float a, float Q)
{
    float rp    = bhHorizonOuter(M, a, Q);
    float best  = 6.f * M;
    float bestE = 1e30f;
    bool  found = false;
    const float r0   = rp * 1.02f;
    const float r1   = 12.f * M;
    const float step = 0.001f * M;
    for (float r = r0; r <= r1; r += step)
    {
        float Om = bhOmegaCircular(r, M, a, Q);
        if (Om <= 0.f) continue;
        float gtt, gtp, gpp;
        bhEquatorialMetric(r, M, a, Q, gtt, gtp, gpp);
        float den = -(gtt + 2.f * Om * gtp + Om * Om * gpp);
        if (den <= 1e-6f) continue;                 // orbit not timelike
        float E = -(gtt + Om * gtp) / std::sqrt(den);
        if (E < bestE) { bestE = E; best = r; found = true; }
    }
    if (!found) best = (6.f * M > rp * 1.5f) ? 6.f * M : rp * 1.5f;
    float lo = rp * 1.15f;
    return best > lo ? best : lo;
}

// Validate and derive everything for a model. aStar/q are passed by
// reference and written back when clamping was necessary, so the UI always
// reflects the values actually used. Guarantees a*^2 + q^2 <= 0.995 for the
// *effective* parameters, hence M^2 - a^2 - Q^2 > 0 (no naked singularity,
// no NaN anywhere downstream).
inline BHDerived bhValidateAndDerive(int model, float& M, float& aStar, float& q)
{
    BHDerived d;

    if (!(M > 0.f) || !std::isfinite(M)) { M = 0.5f; d.clamped = true; }

    if (!std::isfinite(aStar)) { aStar = 0.f; d.clamped = true; }
    if (!std::isfinite(q))     { q     = 0.f; d.clamped = true; }
    if (aStar < 0.f)    { aStar = 0.f;    d.clamped = true; }
    if (aStar > 0.995f) { aStar = 0.995f; d.clamped = true; }
    if (q < 0.f)        { q = 0.f;        d.clamped = true; }
    if (q > 0.995f)     { q = 0.995f;     d.clamped = true; }

    // Effective parameters per model
    float aEff = (model == BH_KERR || model == BH_KERR_NEWMAN) ? aStar : 0.f;
    float qEff = (model == BH_REISSNER_NORDSTROM || model == BH_KERR_NEWMAN) ? q : 0.f;

    // Combined extremal bound a*^2 + q^2 <= BH_EXTREMAL_LIMIT
    float s2 = aEff * aEff + qEff * qEff;
    if (s2 > BH_EXTREMAL_LIMIT)
    {
        float scale = std::sqrt(BH_EXTREMAL_LIMIT / s2);
        aEff *= scale;
        qEff *= scale;
        // write the clamped values back into the user-visible knobs
        if (model == BH_KERR || model == BH_KERR_NEWMAN)               aStar = aEff;
        if (model == BH_REISSNER_NORDSTROM || model == BH_KERR_NEWMAN) q = qEff;
        d.clamped = true;
    }

    d.a = aEff * M;
    d.Q = qEff * M;
    d.rPlus  = bhHorizonOuter(M, d.a, d.Q);
    d.rMinus = bhHorizonInner(M, d.a, d.Q);
    d.rErgo  = bhErgoEquatorial(M, d.Q);
    if (model == BH_KERR || model == BH_KERR_NEWMAN)
    {
        d.rPhoton = bhPhotonOrbitKerrPrograde(M, aEff);
        float lo = d.rPlus * 1.02f;
        if (d.rPhoton < lo) d.rPhoton = lo;
    }
    else
    {
        d.rPhoton = bhPhotonSphereSpherical(M, d.Q);
    }
    d.rIsco = bhIscoNumeric(M, d.a, d.Q);
    return d;
}

// ---------------------------------------------------------------------------
// Render-parameter sanitation (host). Guarantees the kernel can never be
// launched with NaN camera vectors, zero step sizes or a zero step budget.
// Forward declaration here; definition lives below RenderParams.
// ---------------------------------------------------------------------------
#include "render_params.h"

inline bool bhFinite3(const float3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Returns true if anything had to be fixed.
inline bool sanitizeRenderParams(RenderParams& P)
{
    bool fixed = false;
    auto fixf = [&](float& v, float lo, float hi, float def)
    {
        if (!std::isfinite(v)) { v = def; fixed = true; }
        if (v < lo) { v = lo; fixed = true; }
        if (v > hi) { v = hi; fixed = true; }
    };

    if (P.width  < 1) { P.width  = 1; fixed = true; }
    if (P.height < 1) { P.height = 1; fixed = true; }
    fixf(P.dPhi, 1e-4f, 0.1f, 0.012f);
    if (P.maxSteps < 16)     { P.maxSteps = 16;     fixed = true; }
    if (P.maxSteps > 200000) { P.maxSteps = 200000; fixed = true; }
    fixf(P.exposure, 0.01f, 100.f, 1.f);
    fixf(P.M, 1e-3f, 100.f, 0.5f);
    fixf(P.tanHalfFov, 0.05f, 5.f, 0.5773503f);
    fixf(P.aspect, 0.05f, 20.f, 16.f / 9.f);
    if (P.model < 0 || P.model > 3) { P.model = 0; fixed = true; }

    // Spin/charge consistency (a, Q in code units)
    float aStar = P.aSpin / P.M, q = P.Qc / P.M, M = P.M;
    if (!std::isfinite(aStar) || !std::isfinite(q) ||
        aStar * aStar + q * q > BH_EXTREMAL_LIMIT + 1e-4f ||
        aStar < 0.f || q < 0.f ||
        !std::isfinite(P.rPlus) || P.rPlus <= 0.f)
    {
        BHDerived d = bhValidateAndDerive(P.model, M, aStar, q);
        P.M = M; P.aSpin = d.a; P.Qc = d.Q;
        P.rPlus = d.rPlus; P.rPhoton = d.rPhoton; P.rErgo = d.rErgo;
        fixed = true;
    }

    if (!bhFinite3(P.camPos) || !bhFinite3(P.camForward) ||
        !bhFinite3(P.camRight) || !bhFinite3(P.camUp))
    {
        P.camPos     = {13.f, 3.2f, 8.f};
        P.camForward = {-0.81f, -0.20f, -0.55f};
        P.camRight   = {0.56f, 0.f, -0.83f};
        P.camUp      = {-0.17f, 0.98f, -0.11f};
        fixed = true;
    }
    // Camera must be outside the horizon with margin
    float r0 = std::sqrt(P.camPos.x * P.camPos.x + P.camPos.y * P.camPos.y +
                         P.camPos.z * P.camPos.z);
    if (r0 < P.rPlus * 2.0f)
    {
        float s = (P.rPlus * 2.0f) / (r0 > 1e-6f ? r0 : 1e-6f);
        P.camPos.x *= s; P.camPos.y *= s; P.camPos.z *= s;
        fixed = true;
    }

    fixf(P.diskInner, P.rPlus * 1.05f, 1000.f, P.rPlus * 3.f);
    fixf(P.diskOuter, P.diskInner * 1.2f, 2000.f, P.diskInner * 4.f);

    if (P.sampleIndex < 0)      { P.sampleIndex = 0; fixed = true; }
    if (P.sampleIndex > 65536)  { P.sampleIndex = 65536; fixed = true; }
    if (P.accumMode < 0 || P.accumMode > 2) { P.accumMode = 0; fixed = true; }
    fixf(P.bloomStrength, 0.f, 4.f, 0.35f);
    return fixed;
}
