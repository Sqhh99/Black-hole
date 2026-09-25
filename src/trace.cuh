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
// paths as an optically thick LTE absorber/emitter. All models share the
// exact relativistic factor
//   g = 1 / [ u^t (E - Omega Lz) ]
// of a circular equatorial Keplerian emitter (Schwarzschild / RN with a=0,
// Kerr / KN with spin). Photon (E, Lz) come from a static-observer tetrad
// at the camera. The observed source function is the Planck spectrum at
// g T_emit integrated against CIE 1931 (I_nu/nu^3 invariance), which
// carries both the colour shift and the visible-band beaming.
//
// Slow light: coordinate flight time is accumulated along the backward ray
// so turbulent advection uses t_emit = diskTime - t_flight (emission earlier
// than observation for long-wound photon-ring paths).
//
// Background starfield is scaled by the static-observer camera energy factor
// (infinity -> camera blueshift, bolometric ~ g^4 for the sky).
// ---------------------------------------------------------------------------
#include "render_params.h"
#include "metric.cuh"
#include "vec_math.cuh"
#include "blackbody_lut.h"
#include <cuda_runtime.h>

constexpr float BH_PI       = 3.14159265358979f;
// 1 / peak value of bhThinDiskFluxWeight (peak sits near r ≈ 1.25 r_in).
// Normalizes the flux profile to [0,1] so P.diskTemp is actually reached.
constexpr float BH_NT_PEAK_INV = 1.f / 0.075f;

// Disk geometry: Gaussian vertical profile rho ∝ exp(-(y/H)^2) with an
// aspect ratio H/R growing gently outward (mild flaring).
constexpr float BH_DISK_HR_IN  = 0.012f;
constexpr float BH_DISK_HR_OUT = 0.022f;
// Turbulence crossfade period (code time units): the sheared pattern is
// rebuilt from two staggered layers so differential rotation can never
// wind it into ever finer (aliasing) rings however long the app runs.
constexpr float BH_SHEAR_PERIOD = 36.f;

// Peak rest-frame disk temperature. P.diskTemp is the Schwarzschild
// (r_in = 6M) value; at a fixed accretion rate the Novikov–Thorne peak flux
// scales as r_in^-3, i.e. T_peak ∝ r_in^-3/4: prograde spin / charge
// (smaller ISCO, higher efficiency) run hotter, retrograde spin cooler.
BH_HD float bhDiskPeakTemp(const RenderParams& P)
{
    float rin = fmaxf(P.diskInner, 1e-4f);
    float s = powf(6.f * P.M / rin, 0.75f);
    return P.diskTemp * fminf(fmaxf(s, 0.4f), 3.f);
}

// Camera metering for the disk: a hotter disk is far brighter in the
// visible band, so -- like any real camera or eye -- exposure adapts,
// compensating 60% (in log) of the change of the peak luminance relative
// to the Schwarzschild reference. Visible luminance of a Planck source is
// approximated by its value at 555 nm. Returns 1 for Schwarzschild.
BH_HD float bhDiskMeterFactor(const RenderParams& P)
{
    const float c2 = 1.4387769e-2f / 555e-9f;   // h c / (k λ) [K]
    float T0 = fmaxf(P.diskTemp, 100.f);
    float T1 = fmaxf(bhDiskPeakTemp(P), 100.f);
    float y0 = 1.f / (expf(fminf(c2 / T0, 80.f)) - 1.f);
    float y1 = 1.f / (expf(fminf(c2 / T1, 80.f)) - 1.f);
    float m  = powf(y0 / fmaxf(y1, 1e-30f), 0.6f);
    return fminf(fmaxf(m, 0.05f), 20.f);
}

// Capture surface radius: event horizon when present, else singularity cut.
__device__ inline float bhCaptureRadius(const RenderParams& P)
{
    if (P.rPlus > 1e-6f) return P.rPlus;
    return BH_SING_CUT_OVER_M * P.M;
}

// Static-observer camera energy factor g = E_cam / E_infty (>= 1: blueshift
// of the sky as light falls in). Uses alpha = sqrt(-g_tt) so g = 1/alpha.
__device__ inline float cameraEnergyFactor(const RenderParams& P)
{
    float r = length(P.camPos);
    if (r < 1e-4f) return 1.f;
    if (P.model == BH_KERR || P.model == BH_KERR_NEWMAN)
    {
        float cth = clampf(P.camPos.y / r, -1.f, 1.f);
        float Sig = bhSigma(r, cth, P.aSpin);
        float m2r = 2.f * P.M * r - P.Qc * P.Qc;
        float gtt = -(1.f - m2r / fmaxf(Sig, 1e-8f));
        float alpha = sqrtf(fmaxf(-gtt, 1e-6f));
        return clampf(1.f / alpha, 0.5f, 4.f);
    }
    float f = bhSphF(r, P.M, P.Qc);
    return clampf(1.f / sqrtf(fmaxf(f, 1e-6f)), 0.5f, 4.f);
}

// ---------------------------------------------------------------------------
// Planck radiance in linear sRGB from the CIE 1931 lookup table
// (blackbody_lut.h, generated by tools/gen_blackbody_lut.py). Absolute
// photometric scale: luminance 1 at 6500 K. Since I_nu/nu^3 is invariant,
// a Planck emitter at T seen with redshift factor g is exactly a Planck
// spectrum at gT, so planckRGB(g*T) carries both the Doppler/gravitational
// colour shift AND the visible-band beaming -- no ad-hoc g^n factor.
// ---------------------------------------------------------------------------
__device__ inline float3 planckRGB(float kelvin)
{
    const float lo = logf(BB_LUT_TMIN);
    const float k  = (float)(BB_LUT_N - 1) / (logf(BB_LUT_TMAX) - lo);
    float x = (logf(clampf(kelvin, BB_LUT_TMIN, BB_LUT_TMAX)) - lo) * k;
    int   i = (int)x;
    if (i > BB_LUT_N - 2) i = BB_LUT_N - 2;
    float f = x - (float)i;
    const float* a = c_bbLut + 3 * i;
    // Interpolate in log space: the table spans many decades at low T.
    float3 la = make_float3(logf(a[0] + 1e-30f), logf(a[1] + 1e-30f), logf(a[2] + 1e-30f));
    float3 lb = make_float3(logf(a[3] + 1e-30f), logf(a[4] + 1e-30f), logf(a[5] + 1e-30f));
    float3 l  = lerp3(la, lb, f);
    return make_float3(expf(l.x), expf(l.y), expf(l.z));
}

// Blackbody chromaticity only (luminance normalized to 1).
__device__ inline float3 planckChroma(float kelvin)
{
    float3 c = planckRGB(kelvin);
    float  y = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return c * (1.f / fmaxf(y, 1e-20f));
}

// ---------------------------------------------------------------------------
// Procedural starfield + Milky-Way band, evaluated on the *deflected* escape
// direction => the background exhibits the gravitational lensing field.
//
// Stars are point sources: each carries a fixed integrated flux spread over
// a small angular Gaussian (~half a pixel at the default field of view), so
// a converged image shows crisp points whose brightness does not depend on
// the kernel width. Halos are NOT drawn in the sky (they would be lensed
// into streaks); glare comes from the screen-space bloom like a real lens.
// Fluxes follow a steep power law (many faint stars, very few bright ones)
// and colours are real blackbody chromaticities, partially desaturated as
// a photographic sensor records them.
// ---------------------------------------------------------------------------
// Default pixel solid angle (60° vertical FOV over 720 rows): star fluxes
// are specified in units of "HDR value if the star filled one pixel".
constexpr float BH_PIX_ANGLE = 1.6e-3f;

// Lensed point sources: `tanDir` is the sky-space tangential direction of
// the lens map at d and `muT` the tangential magnification. The kernel is
// shrunk by muT along tanDir, so a strongly lensed star still images as a
// point (brightened by muT, as a magnified point source should be) instead
// of being smeared into an arc by the finite kernel width.
__device__ inline float3 starLayer(float3 d, float cellScale, float fluxBase,
                                   float fluxExp, float fluxMax, float sigma,
                                   float seed, float3 tanDir, float muT)
{
    float3 p  = d * cellScale;
    float3 id = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));

    float3 h  = hash33(id + make_float3(seed, 2.f * seed, -seed));
    float3 sp = id + make_float3(0.25f + 0.5f * h.x, 0.25f + 0.5f * h.y,
                                 0.25f + 0.5f * h.z);
    float3 sd = normalize(sp);
    // The star belongs to this cell only if its sky point lies in it;
    // otherwise it would be clipped by the neighbouring cell.
    float3 q = sd * cellScale;
    if (floorf(q.x) != id.x || floorf(q.y) != id.y || floorf(q.z) != id.z)
        return make_float3(0.f, 0.f, 0.f);

    float3 dv  = d - sd;
    float  dt  = dot(dv, tanDir);
    float  th2 = dot(dv, dv) + dt * dt * (muT * muT - 1.f);  // anisotropic
    float  inv2s2 = 1.f / (2.f * sigma * sigma);
    if (th2 * inv2s2 > 9.f) return make_float3(0.f, 0.f, 0.f);

    float u    = fmaxf(hash13(id * 1.37f + make_float3(seed, 5.1f, 9.7f)), 1e-4f);
    float flux = fminf(fluxBase * powf(u, -fluxExp), fluxMax);

    float tsel   = hash13(id + make_float3(7.3f, -3.1f, seed));
    float kelvin = 3000.f + 9500.f * tsel * tsel * tsel;   // mostly K/G,
                                                          // a few A/B stars
    float3 col = planckChroma(kelvin);
    col = lerp3(make_float3(1.f, 1.f, 1.f), col, 0.55f);  // sensor-like chroma

    const float pixSA = BH_PIX_ANGLE * BH_PIX_ANGLE;
    float peak = flux * pixSA * inv2s2 * (muT / BH_PI);    // flux / (2π σ_t σ_r)
    return col * (peak * __expf(-th2 * inv2s2));
}

// Milky-Way frame: galactic pole and centre directions, tilted ~31° to the
// disk plane so the band crosses the default view diagonally instead of
// wrapping the hole in a uniform lensed fog ring.
__device__ inline float3 galacticPole()   { return make_float3(0.1439f, -0.8593f, -0.4910f); }
__device__ inline float3 galacticCenter() { return make_float3(-0.9364f, 0.0437f, -0.3508f); }

// gCam = E_cam / E_infty (static-observer blueshift of the sky); the sky is
// scaled bolometrically by g^4. `tanDir`/`muT` describe the local lens map
// for point sources (see starLayer); the defaults mean "unlensed".
__device__ inline float3 backgroundColor(float3 dir, float gCam = 1.f,
                                         float3 tanDir = make_float3(0.f, 0.f, 0.f),
                                         float muT = 1.f)
{
    float3 d = normalize(dir);

    float Tscale = clampf(gCam, 0.5f, 3.f);
    float Iscale = Tscale * Tscale * Tscale * Tscale;

    // Galactic latitude / band profile
    float sb   = dot(d, galacticPole());
    float band = __expf(-sb * sb * (1.f / (2.f * 0.13f * 0.13f)));
    float cgc  = dot(d, galacticCenter());
    float bulge = __expf((cgc - 1.f) * 2.2f) * __expf(-sb * sb * (1.f / (2.f * 0.22f * 0.22f)));

    const float sig0 = 0.55f * BH_PIX_ANGLE;
    float3 c = make_float3(0.f, 0.f, 0.f);
    c += starLayer(d,  55.f, 0.010f, 1.00f, 10.f, sig0, 0.f, tanDir, muT);   // bright, sparse
    c += starLayer(d, 120.f, 0.004f, 0.90f, 1.2f, sig0, 13.f, tanDir, muT);
    // Dense faint population, concentrated toward the galactic plane.
    float dense = 0.35f + 1.8f * band + 1.5f * bulge;
    c += starLayer(d, 260.f, 0.0012f * dense, 0.70f, 0.2f,
                       0.40f * BH_PIX_ANGLE, 29.f, tanDir, muT);

    // Diffuse unresolved starlight with dust lanes (seam-free 3D noise).
    float mw = band + 1.6f * bulge;
    if (mw > 1e-3f)
    {
        float n    = fbm3(d * 5.0f, 4);
        float dust = fbm3(d * 11.0f + make_float3(3.1f, -7.4f, 1.9f), 3);
        float lanes = smoothstepf(0.42f, 0.62f, dust) * smoothstepf(0.0f, 0.5f, band);
        float glow  = mw * (0.25f + 1.1f * n * n) * (1.f - 0.85f * lanes);
        // Old-population warm white near the bulge, neutral in the disk.
        float3 tint = lerp3(make_float3(0.88f, 0.92f, 1.0f),
                            make_float3(1.0f, 0.93f, 0.84f),
                            clampf(bulge * 2.f, 0.f, 1.f));
        c += tint * (0.0040f * glow);
    }

    // Faint neutral floor keeps the deep sky just off pure black (the
    // dither decorrelates the 8-bit quantization there).
    c += make_float3(0.0008f, 0.0009f, 0.0011f);

    return c * Iscale;
}

// ---------------------------------------------------------------------------
// Accretion disk sampling context — unified GR emitter formula for all models.
// Photon conserved (E, Lz) from the static-observer tetrad at the camera;
// g = 1/[u^t (E − Ω Lz)] for a co-rotating equatorial Keplerian emitter
// (Schwarzschild / RN / Kerr / KN via bhEmitterGFactor).
// ---------------------------------------------------------------------------
struct DiskCtx
{
    float E;    // photon conserved energy (−p_t)
    float Lz;   // photon conserved angular momentum (p_φ)
    float a;    // spin parameter (code units) used for Ω and metric
    float Q;    // charge (code units)
    float jit;  // [0,1) per-sample offset of the volumetric sub-steps
};

// Static-observer tetrad → photon conserved E, Lz (same construction as the
// Kerr integrator; valid for a = 0 Schwarzschild / RN as well).
__device__ inline void photonConservedEL(float3 origin, float3 dir,
                                         float M, float a, float Q,
                                         float& E, float& Lz)
{
    float r0 = length(origin);
    if (r0 < 1e-4f) { E = 1.f; Lz = 0.f; return; }
    float cth0 = clampf(origin.y / r0, -1.f, 1.f);
    float th0  = acosf(cth0);
    float ph0  = atan2f(-origin.z, origin.x);

    float s  = sinf(th0), c = cosf(th0);
    float sp = sinf(ph0), cp = cosf(ph0);
    // Orthonormal polar basis (matches kerrBasis / kerrToCart)
    float3 er  = make_float3(s * cp,  c, -s * sp);
    float3 eph = make_float3(-sp, 0.f, -cp);
    float dpc = dot(dir, eph);
    (void)er;

    float s0 = s; if (s0 < 1e-4f) s0 = 1e-4f;
    float s02 = s0 * s0;
    float Sig0 = bhSigma(r0, cth0, a);
    float m2r0 = 2.f * M * r0 - Q * Q;

    float gtt = -(1.f - m2r0 / fmaxf(Sig0, 1e-8f));
    if (gtt > -1e-5f) gtt = -1e-5f;
    float gtp = -a * m2r0 * s02 / Sig0;
    float gpp = (r0 * r0 + a * a + m2r0 * a * a * s02 / Sig0) * s02;

    float A    = rsqrtf(fmaxf(gpp - gtp * gtp / gtt, 1e-8f));
    float ptUp = rsqrtf(-gtt) - dpc * A * gtp / gtt;
    float pfUp = dpc * A;

    E  = -(gtt * ptUp + gtp * pfUp);
    Lz =   gtp * ptUp + gpp * pfUp;
    if (!(E > 1e-6f) || !isfinite(E)) { E = 1.f; Lz = 0.f; }
}

// Wrap angle difference into (−π, π].
__device__ inline float wrapDeltaPhi(float d)
{
    d = fmodf(d + BH_PI, 2.f * BH_PI);
    if (d < 0.f) d += 2.f * BH_PI;
    return d - BH_PI;
}

// One layer of sheared MRI-like turbulence in the co-rotating frame:
// 3D noise on (log r, cos φ', sin φ') -- seam-free in azimuth, features
// elongated along the flow (radial frequency >> azimuthal).
__device__ inline float diskNoiseLayer(float lr, float phiRot, float seed)
{
    float s, c;
#ifdef __CUDA_ARCH__
    __sincosf(phiRot, &s, &c);
#else
    s = sinf(phiRot); c = cosf(phiRot);
#endif
    // Billowy large scales, ridged (filamentary) small scales.
    float3 q = make_float3(lr * 30.f + seed, c * 4.5f, s * 4.5f);
    float v = 0.f, amp = 0.5f, norm = 0.f;
    for (int i = 0; i < 5; ++i)
    {
        float n = vnoise3(q);
        if (i >= 2) n = 1.f - fabsf(2.f * n - 1.f);
        v += amp * n; norm += amp;
        q = make_float3(q.x * 2.03f + 17.1f, q.y * 2.01f - 7.3f, q.z * 1.99f + 3.9f);
        amp *= 0.62f;
    }
    return v / norm;
}

// Turbulence field in [0,1] (mean ~0.5) advected with the Keplerian flow
// Ω(r). Each of two layers is born unsheared and winds into trailing
// spirals as it ages (inner gas laps outer gas), exactly like a real
// sheared eddy; layers staggered by half a period are crossfaded with a
// variance-preserving blend and reset while invisible, which bounds the
// accumulated shear (no endless winding into aliasing rings).
__device__ inline float diskTurbulence(float rc, float azim, float omega, float t)
{
    float ph = t * (1.f / BH_SHEAR_PERIOD);
    float f0 = fract(ph), f1 = fract(ph + 0.5f);
    float w0 = 1.f - fabsf(2.f * f0 - 1.f);
    float w1 = 1.f - w0;
    float lr = logf(fmaxf(rc, 1e-3f));
    // A layer carrying < 3% of the weight is skipped (saves half the noise
    // work near each crossfade end; the blend below stays normalized).
    if (w0 < 0.03f) w0 = 0.f;
    if (w1 < 0.03f) w1 = 0.f;
    float n0 = 0.5f, n1 = 0.5f;
    if (w0 > 0.f)
        n0 = diskNoiseLayer(lr, azim - omega * f0 * BH_SHEAR_PERIOD, 37.f * floorf(ph));
    if (w1 > 0.f)
        n1 = diskNoiseLayer(lr, azim - omega * f1 * BH_SHEAR_PERIOD,
                            37.f * floorf(ph + 0.5f) + 101.f);
    float n = (w0 * (n0 - 0.5f) + w1 * (n1 - 0.5f)) * rsqrtf(w0 * w0 + w1 * w1);
    return clampf(0.5f + 2.3f * n, 0.f, 1.f);   // fBm is narrow: restore contrast
}

// ---------------------------------------------------------------------------
// Volumetric disk sampling along one geodesic segment [a, b] (Cartesian).
// `tEmit` is the emission coordinate time (slow-light: diskTime - t_flight).
//
// Optically thick thin disk in LTE (Novikov–Thorne is a blackbody disk):
//   • Keplerian orbital velocity field Ω(r) from the metric
//   • Exact g = 1/[u^t (E − Ω Lz)]  (grav. redshift + Doppler beaming)
//   • Novikov–Thorne flux (peak-normalized) → T(r) ∝ F^{1/4}
//   • Source function S = B(g T) (Planck in the observer frame)
//   • Formal solution per sub-step: I += T_r S (1 − e^{−α ds}),
//     T_r *= e^{−α ds}. With the default opacity the inner disk is very
//     optically thick (a solid photosphere, independent of path length),
//     thinning out in the outer annulus where it turns wispy.
// Sub-steps are clipped to the disk slab, sized to resolve the vertical
// profile, and offset by a per-sample jitter so progressive accumulation
// converges instead of freezing the sampling pattern into moiré.
// ---------------------------------------------------------------------------
__device__ inline void sampleDiskSegment(const RenderParams& P, const DiskCtx& ctx,
                                         float3 a, float3 b, float tEmit,
                                         float3& accum, float& trans)
{
    const float rin  = fmaxf(P.diskInner, 1e-4f);
    const float rout = fmaxf(P.diskOuter, rin * 1.01f);

    // Slab |y| <= 3 H(r_out): cull, then clip the segment to it.
    const float yMax = 3.f * BH_DISK_HR_OUT * rout;
    if (a.y > yMax && b.y > yMax) return;
    if (a.y < -yMax && b.y < -yMax) return;

    float t0 = 0.f, t1 = 1.f;
    float dy = b.y - a.y;
    if (fabsf(dy) > 1e-9f)
    {
        float ta = (yMax - a.y) / dy, tb = (-yMax - a.y) / dy;
        t0 = fmaxf(t0, fminf(ta, tb));
        t1 = fminf(t1, fmaxf(ta, tb));
        if (t0 >= t1) return;
    }
    float3 A = lerp3(a, b, t0), B = lerp3(a, b, t1);
    float3 seg = B - A;
    float  segLen = length(seg);
    if (segLen < 1e-7f) return;

    // Radial cull: cylindrical radius range of the (straight) sub-segment.
    float rcA = sqrtf(A.x * A.x + A.z * A.z);
    float rcB = sqrtf(B.x * B.x + B.z * B.z);
    float sxz = seg.x * seg.x + seg.z * seg.z;
    float tc  = (sxz > 1e-12f) ? clampf(-(A.x * seg.x + A.z * seg.z) / sxz, 0.f, 1.f) : 0.f;
    float3 C  = lerp3(A, B, tc);
    float rcMin = sqrtf(C.x * C.x + C.z * C.z);
    float rcMax = fmaxf(rcA, rcB);
    if (rcMin > rout || rcMax < rin) return;

    // Sub-steps: ≤ 0.1 code units along the ray and ≤ 0.4 H vertically.
    float Hmin = BH_DISK_HR_IN * fmaxf(rcMin, rin);
    float nf   = fmaxf(segLen * 10.f, fabsf(seg.y) / (0.4f * Hmin));
    int   nSub = 1 + (int)fminf(nf, 31.f);
    float ds   = segLen / (float)nSub;

    const float M = P.M, aSpin = ctx.a, Qc = ctx.Q;
    const float invOut = 1.f / rout;
    const float Tpeak  = bhDiskPeakTemp(P);
    const bool  wantSpots = P.hotSpotsEnabled && P.hotSpotStrength > 1e-4f;

    for (int j = 0; j < nSub; ++j)
    {
        float  t  = ((float)j + ctx.jit) / (float)nSub;
        float3 p  = lerp3(A, B, t);
        float  rc = sqrtf(p.x * p.x + p.z * p.z);
        if (rc < rin || rc > rout) continue;

        float HR = BH_DISK_HR_IN + (BH_DISK_HR_OUT - BH_DISK_HR_IN) * (rc * invOut);
        float dz = p.y / (HR * rc);
        if (fabsf(dz) > 3.f) continue;

        // Surface-density envelope: sharp (but resolved) ISCO wall, soft
        // outer taper.
        float rho = __expf(-dz * dz)
                  * smoothstepf(rin, rin * 1.05f, rc)
                  * (1.f - smoothstepf(rout * 0.68f, rout, rc));
        if (rho < 1e-4f) continue;

        float omega = bhOmegaCircular(rc, M, aSpin, Qc);
        if (omega <= 0.f) continue;
        float g = bhEmitterGFactor(rc, M, aSpin, Qc, ctx.E, ctx.Lz);
        if (g <= 0.f) continue;

        // Azimuth matches kerrToCart (spin +Y): φ = atan2(-z, x)
        float azim = atan2f(-p.z, p.x);
        float turb = diskTurbulence(rc, azim, omega, tEmit);

        // Temperature: Novikov–Thorne T ∝ F^{1/4}, modulated by the
        // turbulence (±~7% in T is ±~30% in visible brightness).
        float Fr = bhThinDiskFluxWeight(rc, rin) * BH_NT_PEAK_INV;
        float T  = Tpeak * powf(clampf(Fr, 1e-6f, 1.f), 0.25f)
                 * (0.90f + 0.20f * turb);

        // Clumpy opacity: dense filaments, thinner gaps (visible where the
        // disk is marginally thin, i.e. the outer annulus and the edges).
        float dens = rho * (0.25f + 1.5f * turb * turb);

        // --- ISCO hot spots (photon-ring dynamics) ---------------------
        if (wantSpots && rc < rin * 1.6f)
        {
            const float rHs0 = rin * 1.07f;
            const float rHs1 = rin * 1.20f;
            float om0 = bhOmegaCircular(rHs0, M, aSpin, Qc);
            float om1 = bhOmegaCircular(rHs1, M, aSpin, Qc);
            if (om0 <= 0.f) om0 = omega;
            if (om1 <= 0.f) om1 = omega;

            const float inv2sPhi = 1.f / (2.f * 0.22f * 0.22f);
            const float inv2sR0  = 1.f / (2.f * (0.05f * rHs0) * (0.05f * rHs0));
            const float inv2sR1  = 1.f / (2.f * (0.06f * rHs1) * (0.06f * rHs1));

            float dph0 = wrapDeltaPhi(azim - (om0 * tEmit + 0.4f));
            float dph1 = wrapDeltaPhi(azim - (om1 * tEmit + 0.4f + BH_PI));
            float dr0  = rc - rHs0;
            float dr1  = rc - rHs1;
            float flick0 = 0.75f + 0.25f * sinf(1.7f * tEmit + 0.9f);
            float flick1 = 0.70f + 0.30f * sinf(1.1f * tEmit + 2.3f);
            float g0 = __expf(-(dph0 * dph0) * inv2sPhi - dr0 * dr0 * inv2sR0);
            float g1 = __expf(-(dph1 * dph1) * inv2sPhi - dr1 * dr1 * inv2sR1);
            float S  = P.hotSpotStrength;
            T    *= 1.f + S * (0.22f * flick0 * g0 + 0.15f * flick1 * g1);
            dens *= 1.f + S * (g0 + g1);
        }

        // Observed Planck radiance at T_obs = g T (colour + beaming).
        float3 Sfn = planckRGB(g * T) * P.diskEmisScale;

        float dtau = P.diskAbsScale * dens * ds;
        float ab   = 1.f - __expf(-dtau);
        accum += Sfn * (trans * ab);
        trans *= 1.f - ab;
        if (trans < 0.003f) { trans = 0.f; return; }
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
// Fallback for exactly radial rays (orbital-plane basis is degenerate).
// True radial inbound null geodesics fall into the horizon — no Euclidean
// disk sampling (that would invent a midplane column). This only covers
// a measure-zero set of directions; the shadow itself comes from the Binet
// integrator via u >= 1/r+ (or r <= r+), never from a drawn sphere.
// ---------------------------------------------------------------------------
__device__ inline float3 radialRay(const RenderParams& P, const DiskCtx& ctx,
                                   float3 o, float3 d,
                                   float3& accum, float& trans, bool& horizon)
{
    (void)ctx; (void)accum; (void)trans; (void)P;
    if (dot(d, o) < 0.f)
    {
        horizon = true;
        return d;
    }
    // Outbound radial: escapes to infinity without disk samples.
    return d;
}

// ---------------------------------------------------------------------------
// Spherically symmetric integrator (Schwarzschild / Reissner-Nordstrom).
// Generalized Binet equation  u'' = 3 M u^2 - 2 Q^2 u^3 - u ; horizon at
// u >= 1/r+. For Q = 0 identical to the original Schwarzschild renderer.
// ---------------------------------------------------------------------------
__device__ inline TraceResult traceSpherical(const RenderParams& P,
                                             float3 origin, float3 dir,
                                             float jitter)
{
    TraceResult R;
    R.horizon = false; R.escaped = false;
    R.escDir = dir;
    R.accum = make_float3(0.f, 0.f, 0.f);
    R.trans = 1.0f;
    R.maxHviol = 0.f;
    R.steps = 0;

    const float M  = P.M;
    const float Q2 = P.Qc * P.Qc;
    const float rCap    = bhCaptureRadius(P);
    const float uHor    = 1.0f / fmaxf(rCap, 1e-6f);
    const float escR    = 120.f * M;
    const float uEscape = 1.0f / escR;

    // Conserved E, Lz from static-observer tetrad (a=0); enables exact
    // Keplerian g-factor for Schwarzschild / RN disk emission.
    float Ephot = 1.f, Lzphot = 0.f;
    photonConservedEL(origin, dir, M, 0.f, P.Qc, Ephot, Lzphot);
    const DiskCtx ctx = {Ephot, Lzphot, 0.f, P.Qc, jitter};

    float3 c  = origin;
    float  r0 = length(c);
    float3 e1 = c / r0;
    float  ddr = dot(dir, e1);
    float3 perp = dir - e1 * ddr;
    float  pl   = length(perp);

    // Only exactly-degenerate orbital plane (pl ~ 0). Do NOT invent a smaller
    // capture cone (e.g. b < ε): that painted a pure-black "mini sphere" in
    // the middle of the real shadow while neighbours still integrated.
    if (pl < 1e-6f)
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
    // Impact parameter b = L/E ≈ r0 * pl / sqrt(f0) for the static observer map.
    float bImpact = fmaxf(r0 * pl / sqrtf(f0), 1e-6f);

    float u   = 1.0f / r0;
    float du  = -sqrtf(f0) * ddr / (r0 * pl);
    float phi = 0.f;
    float tFlight = 0.f;

    float3 prevPos = c;
    float3 dir3    = dir;
    // Adaptive angular step, recomputed every iteration:
    //   * weak field (r >> r_photon): the Binet ODE is nearly linear
    //     (u'' ≈ -u), so RK4 stays accurate with steps up to 6x dPhi,
    //     growing linearly with r — far-field flight costs few steps;
    //   * strong field (r <~ 9M): full dPhi resolution, unchanged from the
    //     fixed-step integrator, so shadow-boundary classification and the
    //     step-halving convergence order are preserved;
    //   * near-radial rays (large |u'|): step shrinks so |Δu| per step is
    //     bounded without clamping u' (which would corrupt capture/escape).
    const float invNineM = 1.f / (9.f * M);
    float h = P.dPhi;

    // RHS of the generalized Binet equation
    auto acc = [M, Q2](float uu)
    {
        return 3.f * M * uu * uu - 2.f * Q2 * uu * uu * uu - uu;
    };

    int step = 0;
    for (; step < P.maxSteps; ++step)
    {
        float hMax = P.dPhi * clampf(invNineM / u, 1.f, 6.f);  // ∝ r far out
        h = fminf(hMax, 0.10f / fmaxf(fabsf(du), 1e-6f));
        if (h < P.dPhi * 0.03f) h = P.dPhi * 0.03f;

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

        // Capture ONLY by horizon crossing or non-finite state — never by a
        // hand-drawn radius or a reduced impact-parameter ball.
        if (!isfinite(u) || !isfinite(du)) { R.horizon = true; break; }
        if (u >= uHor)                     { R.horizon = true; break; }
        // Deep far-field overshoot (near-radial escape): done.
        if (u < 1e-4f) { R.escaped = true; R.escDir = dir3; break; }

        float  r  = 1.0f / u;
        float  cp = cosf(phi), sp = sinf(phi);
        float3 m  = e1 * cp + e2 * sp;
        float3 mp = e1 * (-sp) + e2 * cp;
        float  drdphi = -du / (u * u);
        float3 pos = m * r;
        dir3 = normalize(m * drdphi + mp * r);

        // Slow light: dt/dφ = r² / (b f(r)) for the spherical null geodesic.
        float fR = fmaxf(bhSphF(r, M, P.Qc), 1e-4f);
        tFlight += h * (r * r) / (bImpact * fR);

        if (P.diskEnabled && R.trans > 0.01f)
            sampleDiskSegment(P, ctx, prevPos, pos,
                              P.diskTime - tFlight, R.accum, R.trans);
        prevPos = pos;

        // Escape: past large radius with outward motion
        if (u < uEscape && du < 0.f) { R.escaped = true; R.escDir = dir3; break; }
    }
    R.steps = step;

    if (!R.horizon && !R.escaped)
    {
        // Step budget exhausted: deep strong-field rays count as captured
        // only if they are clearly inside the photon region (not a mini-sphere).
        if (u > 1.0f / fmaxf(1.5f * P.rPhoton, 3.f * M)) R.horizon = true;
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
                                        bool wantStats, float jitter)
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
    const float rCap = bhCaptureRadius(P);
    // Floor used by the adaptive step when there is no horizon.
    const float rFloor = fmaxf(rp, rCap);
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

    const DiskCtx ctx = {E, Lz, a, Q, jitter};
    const float E2 = E * E;
    const bool  trackPos = (P.diskEnabled != 0);
    float tFlight = 0.f;

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

        if (y.r <= rCap * 1.002f + 1e-3f) { R.horizon = true; break; }

        // Constraint monitor. Skipped in the immediate vicinity of the
        // horizon, where Delta -> 0 makes P^2/Delta a catastrophic float
        // cancellation: those rays are captured within a step or two and
        // the spike is a property of the diagnostic, not of the orbit.
        if (wantStats && y.r > rFloor * 1.05f)
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
        float h = 1.0f / (fabsf(d1.r) / (8.0f * P.dPhi * fmaxf(y.r - rFloor, 0.02f))
                        + (fabsf(d1.th) + fabsf(d1.ph)) / (2.2f * P.dPhi)
                        + 1e-5f);
        if (y.r < 1.5f * P.rPhoton) h *= 0.55f;
        float spd = fabsf(d1.r) + y.r * (fabsf(d1.th) + fabsf(d1.ph)) + 1e-6f;
        float capLen;
        if (trackPos && y.r < P.diskOuter * 1.3f)
        {
            // Fine sampling only where the ray can actually be inside the
            // disk slab (|y| < ~2.6 sigma of H_max); segments approaching
            // from above or below use a medium cap so the slab entry is
            // still subsampled finely by sampleDiskSegment.
            float yAbs = y.r * fabsf(cosf(y.th));
            capLen = (yAbs < 1.8f) ? 1.2f : 2.2f;
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

        // Slow light: Σ dt/dλ = a(L − a E sin²θ) + (r²+a²) P / Δ
        {
            float s, cth;
            bhSinCos(y.th, s, cth);
            if (s < 1e-4f) s = 1e-4f;
            float Sig = bhSigma(y.r, cth, a);
            float Del = fmaxf(bhDelta(y.r, M, a, Q), 1e-5f);
            float Pp  = E * (y.r * y.r + a * a) - a * Lz;
            float dtdl = (a * (Lz - a * E * s * s) + (y.r * y.r + a * a) * Pp / Del)
                       / fmaxf(Sig, 1e-8f);
            tFlight += h * fabsf(dtdl);
        }

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
                sampleDiskSegment(P, ctx, prevPos, pos,
                                  P.diskTime - tFlight, R.accum, R.trans);
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
// `jitter` in [0,1) offsets the disk's volumetric sub-steps (0.5 = centered).
__device__ inline TraceResult traceRay(const RenderParams& P,
                                       float3 origin, float3 dir,
                                       bool wantStats, float jitter = 0.5f)
{
    if (P.model == BH_KERR || P.model == BH_KERR_NEWMAN)
        return traceKerr(P, origin, dir, wantStats, jitter);
    return traceSpherical(P, origin, dir, jitter);
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

    // Disk sub-step offset: per-pixel random start advanced by the golden
    // ratio each sample (low-discrepancy in time, decorrelated in space).
    float jit = fract(hash12(make_float2((float)px * 1.618f + 0.37f, (float)py * 1.618f + 0.37f))
                      + 0.61803399f * (float)P.sampleIndex);
    TraceResult t = traceRay(P, P.camPos, dir, false, jit);

    float  gCam = cameraEnergyFactor(P);
    float3 bg  = make_float3(0.f, 0.f, 0.f);
    if (!t.horizon)
    {
        // Point-source lens map, exact for the spherical models and a good
        // approximation for Kerr: about the camera–hole axis the map is
        // axisymmetric, so a ring of sources at polar angle β images to a
        // ring at α and the tangential magnification is sin α / sin β.
        float3 axis = normalize(make_float3(-P.camPos.x, -P.camPos.y, -P.camPos.z));
        float3 e    = normalize(t.escDir);
        float3 ce   = cross(axis, e);
        float  sinB = length(ce);
        float  sinA = length(cross(axis, dir));
        float  muT  = clampf(sinA / fmaxf(sinB, 1e-4f), 0.05f, 12.f);
        float3 tan  = (sinB > 1e-5f) ? ce * (1.f / sinB) : make_float3(0.f, 0.f, 0.f);
        if (sinB <= 1e-5f) muT = 1.f;
        bg = backgroundColor(t.escDir, gCam, tan, muT);
    }
    float3 hdr = t.accum + bg * t.trans;

    // Never let a stray non-finite sample poison the accumulation buffer.
    if (!isfinite(hdr.x) || !isfinite(hdr.y) || !isfinite(hdr.z))
        hdr = make_float3(0.f, 0.f, 0.f);
    return hdr;
}
