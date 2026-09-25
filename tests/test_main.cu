// ---------------------------------------------------------------------------
// tests/test_main.cu
//
// Executable verification suite for the four-model black hole renderer.
// Runs the *same* device integrators used by the renderer (trace.cuh) on a
// grid of camera rays and checks, with explicit tolerances:
//
//   T1  Event-horizon radii match the closed-form theory for all models.
//   T2  Parameter validation rejects/clamps naked-singularity (default)
//       and non-finite inputs; retrograde a* is accepted; allowNaked path.
//   T3  Model-consistency limits:
//         RN(q=0)        == Schwarzschild   (same integrator, tight tol)
//         KN(a*,q=0)     == Kerr(a*)        (same integrator, tight tol)
//         Kerr(a*=0)     ~= Schwarzschild   (cross-integrator tol)
//         KN(a*=0,q)     ~= RN(q)           (cross-integrator tol)
//         KN(a*=0,q=0)   ~= Schwarzschild   (cross-integrator tol)
//         Kerr(-a*) shadow asymmetry opposite Kerr(+a*)
//   T4  Null-geodesic Hamiltonian constraint |K|/(E^2 (r^2+a^2)) stays
//       small along Kerr/KN rays (max + median monitored).
//   T5  Convergence: halving the step-quality (and doubling the step
//       budget) changes escape directions by less and less; Schwarzschild
//       shadow angular radius matches theory b_crit = 3*sqrt(3)*M to <1%.
//   T6  Stability: zero step size, zero max steps, NaN camera, out-of-range
//       a*/q are sanitized; a full render completes with no CUDA error.
//
// Exit code 0 iff every test passes.
// ---------------------------------------------------------------------------
#include "../src/render_params.h"
#include "../src/metric.cuh"
#include "../src/trace.cuh"
#include "../src/kernel_launch.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::printf("[FATAL] CUDA error %s at %s:%d\n",                  \
                        cudaGetErrorString(_e), __FILE__, __LINE__);         \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

static int g_failures = 0;

static void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

// ---------------------------------------------------------------------------
// Probe kernels
// ---------------------------------------------------------------------------
struct Probe
{
    int    outcome;   // 0 = horizon, 1 = escaped
    float3 escDir;
    float  maxHviol;
    float  lum;       // accumulated disk luminance (rough)
};

__global__ void probeGridKernel(Probe* out, RenderParams P, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    float fx = (i + 0.5f) / (float)nx;
    float fy = (j + 0.5f) / (float)ny;
    float3 dir = cameraRayDir(P, fx, fy);

    TraceResult t = traceRay(P, P.camPos, dir, true);

    Probe pr;
    pr.outcome  = t.escaped ? 1 : 0;
    pr.escDir   = t.escDir;
    pr.maxHviol = t.maxHviol;
    pr.lum      = t.accum.x + t.accum.y + t.accum.z;
    out[j * nx + i] = pr;
}

__global__ void probeOneKernel(Probe* out, RenderParams P, float3 dir)
{
    TraceResult t = traceRay(P, P.camPos, dir, true);
    out->outcome  = t.escaped ? 1 : 0;
    out->escDir   = t.escDir;
    out->maxHviol = t.maxHviol;
    out->lum      = t.accum.x + t.accum.y + t.accum.z;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void hcross(const float f[3], const float u[3], float r[3])
{
    r[0] = f[1]*u[2] - f[2]*u[1];
    r[1] = f[2]*u[0] - f[0]*u[2];
    r[2] = f[0]*u[1] - f[1]*u[0];
}

// Same orbit-camera construction as the application (main.cpp::fillCamera)
static void setCamera(RenderParams& P, float az, float el, float dist)
{
    float ca = std::cos(az), sa = std::sin(az);
    float ce = std::cos(el), se = std::sin(el);
    float p[3] = { dist*ce*ca, dist*se, dist*ce*sa };
    float fl = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    float f[3] = { -p[0]/fl, -p[1]/fl, -p[2]/fl };
    float r[3] = { -f[2], 0.f, f[0] };                // cross(f, +Y)
    float rl = std::sqrt(r[0]*r[0] + r[2]*r[2]);
    if (rl < 1e-6f) { r[0] = 1.f; r[1] = 0.f; r[2] = 0.f; rl = 1.f; }
    r[0] /= rl; r[2] /= rl;
    float u[3]; hcross(r, f, u);

    P.camPos     = { p[0], p[1], p[2] };
    P.camForward = { f[0], f[1], f[2] };
    P.camRight   = { r[0], r[1], r[2] };
    P.camUp      = { u[0], u[1], u[2] };
    P.tanHalfFov = std::tan(60.f * 3.14159265f / 180.f * 0.5f);
    P.aspect     = 16.f / 9.f;
}

static RenderParams makeParams(int model, float aStar, float q,
                               float dPhi = 0.012f, int maxSteps = 4000,
                               bool allowNaked = false)
{
    RenderParams P;
    float M = 0.5f;
    BHDerived d = bhValidateAndDerive(model, M, aStar, q, allowNaked);
    P.model   = model;
    P.M       = M;
    P.aSpin   = d.a;
    P.Qc      = d.Q;
    P.rPlus   = d.rPlus;
    P.rPhoton = d.rPhoton;
    P.rErgo   = d.rErgo;
    P.allowNaked = allowNaked ? 1 : 0;
    P.diskInner = d.rIsco;
    P.diskOuter = 8.f;
    P.diskEnabled = 0;      // pure geometry for the comparison tests
    P.dPhi = dPhi;
    P.maxSteps = maxSteps;
    setCamera(P, 0.6f, 0.25f, 16.f);
    return P;
}

static std::vector<Probe> runGrid(const RenderParams& P, int nx, int ny)
{
    Probe* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, sizeof(Probe) * nx * ny));
    dim3 block(16, 8);
    dim3 grid((nx + 15) / 16, (ny + 7) / 8);
    KLAUNCH(probeGridKernel, grid, block, 0, dev, P, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Probe> host(nx * ny);
    CUDA_CHECK(cudaMemcpy(host.data(), dev, sizeof(Probe) * nx * ny,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev));
    return host;
}

static Probe runOne(const RenderParams& P, float dx, float dy, float dz)
{
    Probe* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, sizeof(Probe)));
    float l = std::sqrt(dx*dx + dy*dy + dz*dz);
    float3 dir = { dx/l, dy/l, dz/l };
    KLAUNCH(probeOneKernel, dim3(1), dim3(1), 0, dev, P, dir);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Probe h;
    CUDA_CHECK(cudaMemcpy(&h, dev, sizeof(Probe), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev));
    return h;
}

struct GridDiff
{
    int    n = 0;
    int    mismatch = 0;       // classification differs
    float  medianAngle = 0.f;  // median escape-direction difference [rad]
    float  maxAngle = 0.f;
};

static GridDiff compareGrids(const std::vector<Probe>& A,
                             const std::vector<Probe>& B)
{
    GridDiff d;
    d.n = (int)A.size();
    std::vector<float> ang;
    for (size_t k = 0; k < A.size(); ++k)
    {
        if (A[k].outcome != B[k].outcome) { ++d.mismatch; continue; }
        if (A[k].outcome == 1)
        {
            // |a - b| = 2 sin(theta/2) ~ theta; exactly 0 for identical
            // vectors (acos(dot) has a ~1e-3 rad float resolution floor).
            float dx = A[k].escDir.x - B[k].escDir.x;
            float dy = A[k].escDir.y - B[k].escDir.y;
            float dz = A[k].escDir.z - B[k].escDir.z;
            ang.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }
    }
    if (!ang.empty())
    {
        std::sort(ang.begin(), ang.end());
        d.medianAngle = ang[ang.size() / 2];
        d.maxAngle    = ang.back();
    }
    return d;
}

// ---------------------------------------------------------------------------
// T1: horizon radii vs closed-form theory (independent double math)
// ---------------------------------------------------------------------------
static void testHorizons()
{
    std::printf("T1  Event-horizon radii vs theory\n");
    struct Case { int model; double aStar, q; };
    const Case cases[] = {
        { BH_SCHWARZSCHILD,      0.0, 0.0 },
        { BH_REISSNER_NORDSTROM, 0.0, 0.6 },
        { BH_KERR,               0.8, 0.0 },
        { BH_KERR_NEWMAN,        0.6, 0.5 },
        { BH_KERR_NEWMAN,        0.9, 0.3 },
    };
    for (const Case& c : cases)
    {
        float M = 0.5f, aS = (float)c.aStar, qq = (float)c.q;
        BHDerived d = bhValidateAndDerive(c.model, M, aS, qq);
        double a = (c.model == BH_KERR || c.model == BH_KERR_NEWMAN)
                 ? c.aStar * 0.5 : 0.0;
        double Q = (c.model == BH_REISSNER_NORDSTROM || c.model == BH_KERR_NEWMAN)
                 ? c.q * 0.5 : 0.0;
        double rp = 0.5 + std::sqrt(0.25 - a*a - Q*Q);   // r+ = M + sqrt(M^2-a^2-Q^2)
        double err = std::fabs((double)d.rPlus - rp);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "%-20s a*=%.2f q=%.2f  r+ = %.6f (theory %.6f, err %.1e)",
                      bhModelName(c.model), c.aStar, c.q, d.rPlus, rp, err);
        check(err < 2e-6, buf);
    }
    // Special-case identities
    {
        float M = 0.5f, a0 = 0.f, q0 = 0.f;
        BHDerived s = bhValidateAndDerive(BH_SCHWARZSCHILD, M, a0, q0);
        check(std::fabs(s.rPlus - 2.f * M) < 1e-6f, "Schwarzschild r+ = 2M");
        check(std::fabs(s.rIsco - 6.f * M) < 0.02f * M,
              "Schwarzschild ISCO = 6M (numeric)");
        check(std::fabs(s.rPhoton - 3.f * M) < 1e-4f,
              "Schwarzschild photon sphere = 3M");
    }
    // Retrograde Kerr: same |a*| => same r+, larger ISCO than prograde
    {
        float M = 0.5f, ap = 0.9f, am = -0.9f, q0 = 0.f;
        float a1 = ap, a2 = am, qq = q0;
        BHDerived dP = bhValidateAndDerive(BH_KERR, M, a1, qq);
        qq = 0.f;
        BHDerived dM = bhValidateAndDerive(BH_KERR, M, a2, qq);
        check(std::fabs(dP.rPlus - dM.rPlus) < 1e-5f,
              "Kerr +/-a* share the same r+");
        check(dM.rIsco > dP.rIsco + 0.1f * M,
              "Kerr retrograde ISCO outside prograde ISCO");
        double rp = 0.5 + std::sqrt(0.25 - (0.9 * 0.5) * (0.9 * 0.5));
        check(std::fabs((double)dM.rPlus - rp) < 2e-6,
              "Kerr a*=-0.9 r+ matches theory");
    }
}

// ---------------------------------------------------------------------------
// T2: parameter validation / clamping
// ---------------------------------------------------------------------------
static void testValidation()
{
    std::printf("T2  Parameter validation\n");
    {
        float M = 0.5f, a = 0.f, q = 2.0f;   // q^2 > 1
        BHDerived d = bhValidateAndDerive(BH_REISSNER_NORDSTROM, M, a, q);
        check(d.clamped && q <= 0.9951f && std::isfinite(d.rPlus) && d.rPlus > 0.f,
              "q = 2 rejected/clamped, horizon finite");
    }
    {
        float M = 0.5f, a = 5.0f, q = 0.f;   // a*^2 > 1
        BHDerived d = bhValidateAndDerive(BH_KERR, M, a, q);
        check(d.clamped && a <= 0.9951f && d.rPlus > d.rMinus - 1e-6f,
              "a* = 5 rejected/clamped, horizon ordered");
    }
    {
        float M = 0.5f, a = -5.0f, q = 0.f;  // |a*| > 1, retrograde
        BHDerived d = bhValidateAndDerive(BH_KERR, M, a, q);
        check(d.clamped && a >= -0.9951f && d.rPlus > 0.f,
              "a* = -5 clamped to retrograde extremal bound");
    }
    {
        float M = 0.5f, a = 0.9f, q = 0.9f;  // a*^2 + q^2 > 1
        BHDerived d = bhValidateAndDerive(BH_KERR_NEWMAN, M, a, q);
        check(d.clamped && a*a + q*q <= BH_EXTREMAL_LIMIT + 1e-4f
              && std::isfinite(d.rPlus),
              "a*^2 + q^2 > 1 scaled back inside the extremal bound");
    }
    {
        float M = 0.5f, a = 1.2f, q = 0.3f;  // naked allowed
        BHDerived d = bhValidateAndDerive(BH_KERR, M, a, q, true);
        check(d.naked && d.rPlus == 0.f && std::isfinite(d.rIsco),
              "allowNaked: a*=1.2 is a naked singularity with r+=0");
    }
    {
        float M = std::nanf(""), a = std::nanf(""), q = -3.f;
        BHDerived d = bhValidateAndDerive(BH_KERR_NEWMAN, M, a, q);
        check(std::isfinite(M) && std::isfinite(a) && q >= 0.f
              && std::isfinite(d.rIsco),
              "NaN mass/spin and negative charge sanitized");
    }
    {
        RenderParams P;                       // deliberately poisoned
        P.dPhi = 0.f; P.maxSteps = 0;
        P.camPos = { std::nanf(""), 0.f, 0.f };
        P.exposure = -5.f; P.model = 99;
        bool fixed = sanitizeRenderParams(P);
        check(fixed && P.dPhi >= 1e-4f && P.maxSteps >= 16
              && bhFinite3(P.camPos) && P.exposure > 0.f
              && P.model >= 0 && P.model <= 3,
              "sanitizeRenderParams fixes zero step / zero budget / NaN camera");
    }
}

// ---------------------------------------------------------------------------
// T3: model-consistency limits
// ---------------------------------------------------------------------------
static void testConsistency()
{
    std::printf("T3  Model-consistency limits (grid %d x %d rays)\n", 96, 54);
    const int NX = 96, NY = 54;

    auto S    = runGrid(makeParams(BH_SCHWARZSCHILD,      0.0f, 0.0f), NX, NY);
    auto RN0  = runGrid(makeParams(BH_REISSNER_NORDSTROM, 0.0f, 0.0f), NX, NY);
    auto RN5  = runGrid(makeParams(BH_REISSNER_NORDSTROM, 0.0f, 0.5f), NX, NY);
    auto K0   = runGrid(makeParams(BH_KERR,               0.0f, 0.0f), NX, NY);
    auto K6   = runGrid(makeParams(BH_KERR,               0.6f, 0.0f), NX, NY);
    auto KN60 = runGrid(makeParams(BH_KERR_NEWMAN,        0.6f, 0.0f), NX, NY);
    auto KN05 = runGrid(makeParams(BH_KERR_NEWMAN,        0.0f, 0.5f), NX, NY);
    auto KN00 = runGrid(makeParams(BH_KERR_NEWMAN,        0.0f, 0.0f), NX, NY);

    char buf[200];

    // Same integrator, parameters reduce identically -> tight tolerance
    GridDiff d = compareGrids(S, RN0);
    std::snprintf(buf, sizeof(buf),
        "RN(q=0) == Schwarzschild        mismatch %d/%d, max dAngle %.2e rad",
        d.mismatch, d.n, d.maxAngle);
    check(d.mismatch == 0 && d.maxAngle < 1e-4f, buf);

    d = compareGrids(K6, KN60);
    std::snprintf(buf, sizeof(buf),
        "KN(a*=.6,q=0) == Kerr(a*=.6)    mismatch %d/%d, max dAngle %.2e rad",
        d.mismatch, d.n, d.maxAngle);
    check(d.mismatch == 0 && d.maxAngle < 1e-4f, buf);

    // Cross-integrator limits (2D Binet vs 3D Boyer-Lindquist Hamiltonian):
    // agreement within discretization tolerance; only rays hugging the
    // shadow boundary may flip classification.
    const float crossMismatchTol = 0.025f;   // <= 2.5% of rays
    const float crossMedianTol   = 0.02f;    // median escape-dir diff [rad]

    d = compareGrids(S, K0);
    std::snprintf(buf, sizeof(buf),
        "Kerr(a*=0) ~= Schwarzschild     mismatch %d/%d, median dAngle %.2e rad",
        d.mismatch, d.n, d.medianAngle);
    check(d.mismatch <= (int)(crossMismatchTol * d.n)
          && d.medianAngle < crossMedianTol, buf);

    d = compareGrids(RN5, KN05);
    std::snprintf(buf, sizeof(buf),
        "KN(a*=0,q=.5) ~= RN(q=.5)       mismatch %d/%d, median dAngle %.2e rad",
        d.mismatch, d.n, d.medianAngle);
    check(d.mismatch <= (int)(crossMismatchTol * d.n)
          && d.medianAngle < crossMedianTol, buf);

    d = compareGrids(S, KN00);
    std::snprintf(buf, sizeof(buf),
        "KN(a*=0,q=0) ~= Schwarzschild   mismatch %d/%d, median dAngle %.2e rad",
        d.mismatch, d.n, d.medianAngle);
    check(d.mismatch <= (int)(crossMismatchTol * d.n)
          && d.medianAngle < crossMedianTol, buf);

    // Physical sanity: charge shrinks the shadow, spin makes it asymmetric.
    auto countCaptured = [](const std::vector<Probe>& v)
    {
        int c = 0;
        for (const Probe& p : v) c += (p.outcome == 0);
        return c;
    };
    int capS = countCaptured(S), capRN = countCaptured(RN5);
    std::snprintf(buf, sizeof(buf),
        "RN(q=.5) shadow smaller than Schwarzschild (%d vs %d captured rays)",
        capRN, capS);
    check(capRN < capS, buf);

    // Retrograde vs prograde: same |a*| must produce different images
    // (frame-dragging asymmetry flips with spin sign).
    auto Kp = runGrid(makeParams(BH_KERR,  0.7f, 0.0f), NX, NY);
    auto Km = runGrid(makeParams(BH_KERR, -0.7f, 0.0f), NX, NY);
    GridDiff dSpin = compareGrids(Kp, Km);
    auto leftRightBias = [&](const std::vector<Probe>& v) -> int
    {
        int L = 0, R = 0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                if (v[j * NX + i].outcome == 0)
                    (i < NX / 2 ? L : R)++;
        return L - R;
    };
    int biasP = leftRightBias(Kp);
    int biasM = leftRightBias(Km);
    std::snprintf(buf, sizeof(buf),
        "Kerr a*=+/-0.7 images differ (mismatch %d, bias %+d vs %+d)",
        dSpin.mismatch, biasP, biasM);
    check(dSpin.mismatch > 0 || dSpin.medianAngle > 1e-3f, buf);
    // Prefer a clean left/right bias flip when the camera is off-axis enough.
    if (biasP != 0 || biasM != 0)
        check(biasP * biasM <= 0,
              "Kerr a*=+/-0.7 left-right capture bias flips or zeros");

    // Naked singularity smoke: rays terminate without CUDA errors.
    {
        auto naked = runGrid(makeParams(BH_KERR, 1.2f, 0.0f, 0.012f, 4000, true),
                             48, 27);
        bool any = !naked.empty();
        for (const Probe& p : naked)
            any &= (p.outcome == 0 || p.outcome == 1);
        check(any, "naked Kerr a*=1.2 grid completes with valid outcomes");
    }
}

// ---------------------------------------------------------------------------
// T4: Hamiltonian-constraint conservation along Kerr/KN rays
// ---------------------------------------------------------------------------
static void testConstraint()
{
    std::printf("T4  Null Hamiltonian constraint (Kerr a*=0.9, KN a*=0.7 q=0.5)\n");
    const int NX = 96, NY = 54;
    const RenderParams cfgs[2] = {
        makeParams(BH_KERR,        0.9f, 0.0f),
        makeParams(BH_KERR_NEWMAN, 0.7f, 0.5f),
    };
    for (int c = 0; c < 2; ++c)
    {
        auto g = runGrid(cfgs[c], NX, NY);
        std::vector<float> v;
        v.reserve(g.size());
        for (const Probe& p : g) v.push_back(p.maxHviol);
        std::sort(v.begin(), v.end());
        float med = v[v.size() / 2], mx = v.back();
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "%-12s |K|/(E^2(r^2+a^2)) median %.2e, max %.2e",
            bhModelName(cfgs[c].model), med, mx);
        check(med < 5e-3f && mx < 5e-2f, buf);
    }
}

// ---------------------------------------------------------------------------
// T5: convergence under step refinement + shadow radius vs theory
// ---------------------------------------------------------------------------
static void testConvergence()
{
    std::printf("T5  Convergence & shadow radius\n");
    const int NX = 96, NY = 54;

    // (a) Kerr a*=0.9: step-quality h, h/2, h/4 -> differences must shrink
    auto A = runGrid(makeParams(BH_KERR, 0.9f, 0.0f, 0.012f,  4000), NX, NY);
    auto B = runGrid(makeParams(BH_KERR, 0.9f, 0.0f, 0.006f,  8000), NX, NY);
    auto C = runGrid(makeParams(BH_KERR, 0.9f, 0.0f, 0.003f, 16000), NX, NY);
    GridDiff dAB = compareGrids(A, B);
    GridDiff dBC = compareGrids(B, C);
    char buf[200];
    std::snprintf(buf, sizeof(buf),
        "Kerr a*=0.9 refinement: median dAngle h->h/2 %.2e, h/2->h/4 %.2e rad",
        dAB.medianAngle, dBC.medianAngle);
    check(dBC.medianAngle <= dAB.medianAngle * 1.2f + 1e-5f
          && dBC.medianAngle < 1e-2f, buf);

    // (b) Schwarzschild shadow: critical impact parameter b_c = 3*sqrt(3)*M.
    // Camera on the equator at r0; a ray tilted by alpha off the radial
    // inward direction has b = r0 sin(alpha) / sqrt(f(r0)); the capture
    // boundary must sit at alpha_c = asin(b_c sqrt(f)/r0).
    RenderParams P = makeParams(BH_SCHWARZSCHILD, 0.f, 0.f, 0.006f, 8000);
    const float r0 = 30.f;
    P.camPos     = { r0, 0.f, 0.f };
    P.camForward = { -1.f, 0.f, 0.f };
    P.camRight   = { 0.f, 0.f, -1.f };
    P.camUp      = { 0.f, 1.f, 0.f };

    const double M  = 0.5;
    const double bc = 3.0 * std::sqrt(3.0) * M;
    const double f0 = 1.0 - 2.0 * M / r0;
    const double alphaTheory = std::asin(bc * std::sqrt(f0) / r0);

    double lo = 0.0, hi = 0.2;               // capture at lo, escape at hi
    for (int it = 0; it < 30; ++it)
    {
        double mid = 0.5 * (lo + hi);
        Probe pr = runOne(P, (float)-std::cos(mid), (float)std::sin(mid), 0.f);
        if (pr.outcome == 0) lo = mid; else hi = mid;
    }
    double alphaNum = 0.5 * (lo + hi);
    double relErr = std::fabs(alphaNum - alphaTheory) / alphaTheory;
    std::snprintf(buf, sizeof(buf),
        "Schwarzschild shadow: alpha = %.6f vs theory %.6f (rel err %.3f%%)",
        alphaNum, alphaTheory, 100.0 * relErr);
    check(relErr < 0.01, buf);
}

// ---------------------------------------------------------------------------
// T6: stability against invalid inputs, full-render smoke test
// ---------------------------------------------------------------------------
extern "C" cudaError_t launchRenderKernel(uchar4* out, const RenderParams& p,
                                          cudaStream_t stream);
extern "C" cudaError_t launchRenderPipeline(uchar4* out, float4* accum,
                                            float4* bloomA, float4* bloomB,
                                            const RenderParams& p,
                                            cudaStream_t stream);

static void testStability()
{
    std::printf("T6  Invalid-input stability (full render smoke tests)\n");
    const int W = 160, H = 90;
    uchar4* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, sizeof(uchar4) * W * H));

    RenderParams bad[4];
    for (RenderParams& P : bad) { P = makeParams(BH_KERR_NEWMAN, 0.7f, 0.5f);
                                  P.width = W; P.height = H; P.diskEnabled = 1; }
    bad[0].dPhi = 0.f;                                  // zero step size
    bad[1].maxSteps = 0;                                // zero step budget
    bad[2].camPos = { std::nanf(""), std::nanf(""), 0.f }; // NaN camera
    bad[3].aSpin = 7.f; bad[3].Qc = 9.f; bad[3].rPlus = -1.f; // naked sing.

    const char* names[4] = {
        "zero step size", "zero max steps", "NaN camera", "invalid a*/q" };

    for (int c = 0; c < 4; ++c)
    {
        CUDA_CHECK(cudaMemset(dev, 0, sizeof(uchar4) * W * H));
        cudaError_t e = launchRenderKernel(dev, bad[c], 0);
        cudaError_t s = cudaDeviceSynchronize();
        std::vector<uchar4> img(W * H);
        cudaError_t m = cudaMemcpy(img.data(), dev, sizeof(uchar4) * W * H,
                                   cudaMemcpyDeviceToHost);
        bool alphaOk = true;
        for (const uchar4& px : img) alphaOk &= (px.w == 255);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "render survives %s (launch %d, sync %d, copy %d, output written %d)",
            names[c], (int)e, (int)s, (int)m, (int)alphaOk);
        check(e == cudaSuccess && s == cudaSuccess && m == cudaSuccess && alphaOk,
              buf);
    }
    CUDA_CHECK(cudaFree(dev));

    // ---- Display-quality pipeline (accumulation + bloom) ----------------
    const int bw = (W + 1) / 2, bh = (H + 1) / 2;
    uchar4* out = nullptr; float4 *acc = nullptr, *bA = nullptr, *bB = nullptr;
    CUDA_CHECK(cudaMalloc(&out, sizeof(uchar4) * W * H));
    CUDA_CHECK(cudaMalloc(&acc, sizeof(float4) * W * H));
    CUDA_CHECK(cudaMalloc(&bA,  sizeof(float4) * bw * bh));
    CUDA_CHECK(cudaMalloc(&bB,  sizeof(float4) * bw * bh));

    RenderParams P = makeParams(BH_KERR, 0.9f, 0.f);
    P.width = W; P.height = H; P.diskEnabled = 1; P.bloomEnabled = 1;

    // Determinism: two fresh accumulations (mode 0) must be bit-identical.
    std::vector<uchar4> imgA(W * H), imgB(W * H);
    P.sampleIndex = 0; P.accumMode = 0;
    CUDA_CHECK(launchRenderPipeline(out, acc, bA, bB, P, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(imgA.data(), out, sizeof(uchar4) * W * H,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(launchRenderPipeline(out, acc, bA, bB, P, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(imgB.data(), out, sizeof(uchar4) * W * H,
                          cudaMemcpyDeviceToHost));
    check(std::memcmp(imgA.data(), imgB.data(), sizeof(uchar4) * W * H) == 0,
          "quality pipeline is deterministic (two fresh frames identical)");

    // Progressive accumulation over 8 jittered samples stays finite and
    // alpha-complete, including with poisoned accumulation state.
    CUDA_CHECK(cudaMemset(acc, 0xFF, sizeof(float4) * W * H)); // NaN garbage
    bool ok = true;
    for (int s = 0; s < 8; ++s)
    {
        P.sampleIndex = s; P.accumMode = (s == 0) ? 0 : 1;
        ok &= (launchRenderPipeline(out, acc, bA, bB, P, 0) == cudaSuccess);
    }
    ok &= (cudaDeviceSynchronize() == cudaSuccess);
    CUDA_CHECK(cudaMemcpy(imgA.data(), out, sizeof(uchar4) * W * H,
                          cudaMemcpyDeviceToHost));
    for (const uchar4& px : imgA) ok &= (px.w == 255);
    check(ok, "8-sample progressive accumulation completes (poisoned "
              "accumulation buffer recovered)");

    // Invalid accumulation fields are sanitized.
    P.sampleIndex = -7; P.accumMode = 99; P.bloomStrength = std::nanf("");
    P.diskTemp = std::nanf(""); P.diskAbsScale = -3.f; P.diskEmisScale = 1e9f;
    ok = (launchRenderPipeline(out, acc, bA, bB, P, 0) == cudaSuccess)
       && (cudaDeviceSynchronize() == cudaSuccess);
    {
        RenderParams S = P;
        sanitizeRenderParams(S);
        ok &= std::isfinite(S.diskTemp) && S.diskTemp >= 1500.f
           && S.diskAbsScale >= 0.f && S.diskEmisScale <= 50.f;
    }
    check(ok, "invalid sampleIndex/accumMode/bloom/disk RT params sanitized");

    // Hot spots (photon-ring dynamics): fixed diskTime, two frames match.
    P = makeParams(BH_KERR, 0.9f, 0.f);
    P.width = W; P.height = H; P.diskEnabled = 1; P.bloomEnabled = 1;
    P.hotSpotsEnabled = 1; P.hotSpotStrength = 1.f;
    P.diskTime = 1.25f; P.sampleIndex = 0; P.accumMode = 0;
    CUDA_CHECK(launchRenderPipeline(out, acc, bA, bB, P, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(imgA.data(), out, sizeof(uchar4) * W * H,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(launchRenderPipeline(out, acc, bA, bB, P, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(imgB.data(), out, sizeof(uchar4) * W * H,
                          cudaMemcpyDeviceToHost));
    check(std::memcmp(imgA.data(), imgB.data(), sizeof(uchar4) * W * H) == 0,
          "hot-spot pipeline deterministic at fixed diskTime");

    CUDA_CHECK(cudaFree(out)); CUDA_CHECK(cudaFree(acc));
    CUDA_CHECK(cudaFree(bA));  CUDA_CHECK(cudaFree(bB));
}

// ---------------------------------------------------------------------------
int main()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
    {
        std::printf("[FATAL] No CUDA device available; the verification "
                    "suite requires the same GPU the renderer uses.\n");
        return 2;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("Black hole renderer verification suite\n"
                "CUDA device 0: %s\n\n", prop.name);

    testHorizons();
    testValidation();
    testConsistency();
    testConstraint();
    testConvergence();
    testStability();

    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
