// ---------------------------------------------------------------------------
// main.cpp
//
// Application entry point: creates the Vulkan context, wires the CUDA
// interop, runs the interactive orbit camera and prints per-frame statistics
// (FPS, CUDA kernel time, Vulkan present time).
//
// Controls:
//   Left mouse drag ......... orbit (azimuth / elevation)
//   Mouse wheel / W,S ....... camera distance
//   A,D ..................... azimuth
//   Q,E ..................... elevation
//   -,= ..................... exposure down / up
//   1,2,3 ................... quality preset (fast / balanced / high)
//   F2,F3,F4,F5 ............. model: Schwarzschild / Reissner-Nordstrom /
//                             Kerr / Kerr-Newman
//   [ , ] ................... spin a* down / up      (Kerr, Kerr-Newman;
//                             a* may be negative = retrograde)
//   , . ..................... charge q down / up     (RN, Kerr-Newman)
//   N ....................... toggle naked-singularity mode (experimental)
//   H ....................... toggle orbiting hot spots (photon-ring dynamics)
//   X / Shift+X ............. rotate black hole about world X (+ / -)
//   Y / Shift+Y ............. rotate black hole about world Y (+ / -)
//   Z / Shift+Z ............. rotate black hole about world Z (+ / -)
//   R ....................... reset orientation + camera to defaults
//   F1 ...................... toggle accretion disk
//   F11 ..................... toggle native fullscreen
//   SPACE ................... pause / resume disk animation
//   ESC ..................... quit
// ---------------------------------------------------------------------------
#include "vulkan_context.h"
#include "cuda_interop.h"
#include "render_params.h"
#include "metric.cuh"
#include "logger.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace
{
constexpr uint32_t kWidth  = 1280;
constexpr uint32_t kHeight = 720;
constexpr float    kPi     = 3.14159265358979f;

// Default camera / body orientation (used by reset key R).
constexpr float kDefaultAzimuth   = 0.6f;
constexpr float kDefaultElevation = 0.20f;
constexpr float kDefaultDistance  = 16.0f;

struct OrbitCamera
{
    float azimuth   = kDefaultAzimuth;    // radians
    float elevation = kDefaultElevation;  // radians above disk plane
    float distance  = kDefaultDistance;   // in Schwarzschild radii
};

struct AppState
{
    OrbitCamera cam;
    float  exposure   = 0.85f;  // slightly lower default: protect orange midtones
    bool   diskOn     = true;
    bool   animPaused = false;
    float  dPhi       = 0.012f;
    int    maxSteps   = 2000;
    double lastX = 0.0, lastY = 0.0;
    bool   dragging = false;

    // Black hole body orientation (world-fixed Euler XYZ, radians).
    // Camera is built in body frame (disk in XZ, spin +Y) then rotated by
    // R = Rz(oriZ) * Ry(oriY) * Rx(oriX) so the hole appears to tumble.
    float  oriX = 0.f;
    float  oriY = 0.f;
    float  oriZ = 0.f;

    // Black hole model knobs. aStar/q are the user's stored settings; each
    // model applies only the parameters that pertain to it (Schwarzschild
    // ignores both, RN ignores spin, Kerr ignores charge). aStar may be
    // negative (retrograde spin).
    int    model   = BH_SCHWARZSCHILD;
    float  aStar   = 0.70f;
    float  q       = 0.40f;
    bool   allowNaked = false; // experimental: permit a*^2 + q^2 > 1
    bool   bhDirty = true;

    bool   bloomOn = true;   // HDR bloom of the disk / photon ring
    bool   hotSpotsOn = true; // orbiting ISCO flares → photon-ring dynamics
    bool   fullscreenToggleRequested = false;
};

AppState g;

// Validate the current model/spin/charge, derive horizon / photon-region /
// ISCO, and push everything into the shared parameter block.
void applyBlackHole(RenderParams& p)
{
    const float knobMax = g.allowNaked ? std::sqrt(BH_NAKED_LIMIT) : BH_EXTREMAL_LIMIT;
    if (g.aStar < -knobMax) g.aStar = -knobMax;
    if (g.aStar >  knobMax) g.aStar =  knobMax;
    if (g.q < 0.f) g.q = 0.f;
    if (g.q > knobMax) g.q = knobMax;

    float M = 0.5f;                 // mass scale: rs = 2M = 1 code unit
    float aS = g.aStar, qq = g.q;
    BHDerived d = bhValidateAndDerive(g.model, M, aS, qq, g.allowNaked);
    if (d.clamped) { g.aStar = aS; g.q = qq; }

    p.model      = g.model;
    p.M          = M;
    p.aSpin      = d.a;
    p.Qc         = d.Q;
    p.rPlus      = d.rPlus;
    p.rPhoton    = d.rPhoton;
    p.rErgo      = d.rErgo;
    p.allowNaked = g.allowNaked ? 1 : 0;
    p.diskInner  = d.rIsco;          // disk inner edge tracks the ISCO
    p.diskOuter  = 8.0f;             // ~16M (between old 24M pancake and 12M tight)

    bool useA = (g.model == BH_KERR || g.model == BH_KERR_NEWMAN);
    bool useQ = (g.model == BH_REISSNER_NORDSTROM || g.model == BH_KERR_NEWMAN);
    char aStr[16], qStr[16];
    if (useA) std::snprintf(aStr, sizeof(aStr), "%.3f", g.aStar);
    else      std::snprintf(aStr, sizeof(aStr), "-");
    if (useQ) std::snprintf(qStr, sizeof(qStr), "%.3f", g.q);
    else      std::snprintf(qStr, sizeof(qStr), "-");
    LOG_INFO("%s | M=%.2f  a*=%s  q=%s | r+=%.4f  r_photon=%.4f  "
             "r_ergo=%.4f  ISCO=%.4f%s%s",
             bhModelName(g.model), M, aStr, qStr,
             d.rPlus, d.rPhoton, d.rErgo, d.rIsco,
             d.naked ? "  [NAKED singularity]" : "",
             d.clamped ? "  [parameters clamped]" : "");
}

void clampCamera()
{
    const float elMax = 1.45f; // avoid pole singularity of the up vector
    if (g.cam.elevation >  elMax) g.cam.elevation =  elMax;
    if (g.cam.elevation < -elMax) g.cam.elevation = -elMax;
    // Outside ~2.2 rs for normal BH; a little further is fine for naked mode.
    float rMin = g.allowNaked ? 1.5f : 2.2f;
    if (g.cam.distance < rMin)  g.cam.distance = rMin;
    if (g.cam.distance > 55.f)  g.cam.distance = 55.f;
}

// Apply R = Rz * Ry * Rx (fixed world axes) to a 3-vector.
void applyBodyOrientation(float& x, float& y, float& z)
{
    const float cx = std::cos(g.oriX), sx = std::sin(g.oriX);
    const float cy = std::cos(g.oriY), sy = std::sin(g.oriY);
    const float cz = std::cos(g.oriZ), sz = std::sin(g.oriZ);

    float x0 = x, y0 = y, z0 = z;
    // Rx
    float x1 = x0;
    float y1 = cx * y0 - sx * z0;
    float z1 = sx * y0 + cx * z0;
    // Ry
    float x2 =  cy * x1 + sy * z1;
    float y2 =  y1;
    float z2 = -sy * x1 + cy * z1;
    // Rz
    x = cz * x2 - sz * y2;
    y = sz * x2 + cz * y2;
    z = z2;
}

void normalize3(float& x, float& y, float& z)
{
    float l = std::sqrt(x * x + y * y + z * z);
    if (l < 1e-8f) { x = 0.f; y = 1.f; z = 0.f; return; }
    x /= l; y /= l; z /= l;
}

void resetView()
{
    g.cam.azimuth   = kDefaultAzimuth;
    g.cam.elevation = kDefaultElevation;
    g.cam.distance  = kDefaultDistance;
    g.oriX = g.oriY = g.oriZ = 0.f;
    clampCamera();
    LOG_INFO("View reset (orientation + camera defaults)");
}

void fillCamera(RenderParams& p)
{
    float ca = std::cos(g.cam.azimuth), sa = std::sin(g.cam.azimuth);
    float ce = std::cos(g.cam.elevation), se = std::sin(g.cam.elevation);
    float px = g.cam.distance * ce * ca;
    float py = g.cam.distance * se;
    float pz = g.cam.distance * ce * sa;

    // forward = look at origin (body frame)
    float fl = std::sqrt(px * px + py * py + pz * pz);
    float fx = -px / fl, fy = -py / fl, fz = -pz / fl;

    // right = normalize(cross(forward, bodyUp)), bodyUp = (0,1,0)
    float rxx = -fz, rxy = 0.f, rxz = fx;
    float rl = std::sqrt(rxx * rxx + rxy * rxy + rxz * rxz);
    if (rl < 1e-6f) { rxx = 1.f; rxy = 0.f; rxz = 0.f; rl = 1.f; }
    rxx /= rl; rxz /= rl;

    // up = cross(right, forward)
    float ux = rxy * fz - rxz * fy;
    float uy = rxz * fx - rxx * fz;
    float uz = rxx * fy - rxy * fx;

    // Map body frame → world via Euler XYZ orientation
    applyBodyOrientation(px, py, pz);
    applyBodyOrientation(fx, fy, fz);
    applyBodyOrientation(rxx, rxy, rxz);
    applyBodyOrientation(ux, uy, uz);
    normalize3(fx, fy, fz);
    normalize3(rxx, rxy, rxz);
    // Re-orthogonalize up from right × forward for numerical stability
    ux = rxy * fz - rxz * fy;
    uy = rxz * fx - rxx * fz;
    uz = rxx * fy - rxy * fx;
    normalize3(ux, uy, uz);

    p.camPos     = {px, py, pz};
    p.camForward = {fx, fy, fz};
    p.camRight   = {rxx, rxy, rxz};
    p.camUp      = {ux, uy, uz};
    p.tanHalfFov = std::tan(60.f * kPi / 180.f * 0.5f);
    p.aspect     = (p.height > 0) ? (float)p.width / (float)p.height : 16.f / 9.f;
}

// ---------------- GLFW callbacks ----------------
void onMouseButton(GLFWwindow* w, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        g.dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g.lastX, &g.lastY);
    }
}

void onCursorPos(GLFWwindow*, double x, double y)
{
    if (g.dragging)
    {
        g.cam.azimuth   += (float)(x - g.lastX) * 0.005f;
        g.cam.elevation += (float)(y - g.lastY) * 0.005f;
        clampCamera();
    }
    g.lastX = x;
    g.lastY = y;
}

void onScroll(GLFWwindow*, double, double yoff)
{
    g.cam.distance *= std::pow(0.92f, (float)yoff);
    clampCamera();
}

void onKey(GLFWwindow* w, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    // Repeat-friendly parameter adjustment
    switch (key)
    {
    case GLFW_KEY_LEFT_BRACKET:  g.aStar -= 0.05f; g.bhDirty = true; return;
    case GLFW_KEY_RIGHT_BRACKET: g.aStar += 0.05f; g.bhDirty = true; return;
    case GLFW_KEY_COMMA:         g.q     -= 0.05f; g.bhDirty = true; return;
    case GLFW_KEY_PERIOD:        g.q     += 0.05f; g.bhDirty = true; return;
    default: break;
    }

    if (action != GLFW_PRESS) return;
    switch (key)
    {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
    case GLFW_KEY_F11: g.fullscreenToggleRequested = true; break;
    case GLFW_KEY_F2: g.model = BH_SCHWARZSCHILD;      g.bhDirty = true; break;
    case GLFW_KEY_F3: g.model = BH_REISSNER_NORDSTROM; g.bhDirty = true; break;
    case GLFW_KEY_F4: g.model = BH_KERR;               g.bhDirty = true; break;
    case GLFW_KEY_F5: g.model = BH_KERR_NEWMAN;        g.bhDirty = true; break;
    case GLFW_KEY_F1:
        g.diskOn = !g.diskOn;
        LOG_INFO("Accretion disk %s", g.diskOn ? "ON" : "OFF");
        break;
    case GLFW_KEY_B:
        g.bloomOn = !g.bloomOn;
        LOG_INFO("Bloom %s", g.bloomOn ? "ON" : "OFF");
        break;
    case GLFW_KEY_N:
        g.allowNaked = !g.allowNaked;
        g.bhDirty = true;
        LOG_INFO("Naked-singularity mode %s (experimental)",
                 g.allowNaked ? "ON" : "OFF");
        break;
    case GLFW_KEY_H:
        g.hotSpotsOn = !g.hotSpotsOn;
        LOG_INFO("Hot spots (photon-ring dynamics) %s",
                 g.hotSpotsOn ? "ON" : "OFF");
        break;
    case GLFW_KEY_R:
        resetView();
        break;
    case GLFW_KEY_SPACE:
        g.animPaused = !g.animPaused;
        LOG_INFO("Disk animation %s", g.animPaused ? "paused" : "running");
        break;
    case GLFW_KEY_1: g.dPhi = 0.020f; g.maxSteps = 1200; LOG_INFO("Quality: FAST");     break;
    case GLFW_KEY_2: g.dPhi = 0.012f; g.maxSteps = 2000; LOG_INFO("Quality: BALANCED"); break;
    case GLFW_KEY_3: g.dPhi = 0.007f; g.maxSteps = 3500; LOG_INFO("Quality: HIGH");     break;
    default: break;
    }
}

void handleHeldKeys(GLFWwindow* w, float dt)
{
    auto down = [w](int k) { return glfwGetKey(w, k) == GLFW_PRESS; };
    if (down(GLFW_KEY_W)) g.cam.distance  *= std::pow(0.5f, dt);
    if (down(GLFW_KEY_S)) g.cam.distance  *= std::pow(2.0f, dt);
    if (down(GLFW_KEY_A)) g.cam.azimuth   -= 1.2f * dt;
    if (down(GLFW_KEY_D)) g.cam.azimuth   += 1.2f * dt;
    if (down(GLFW_KEY_Q)) g.cam.elevation -= 0.9f * dt;
    if (down(GLFW_KEY_E)) g.cam.elevation += 0.9f * dt;
    if (down(GLFW_KEY_MINUS)) g.exposure  *= std::pow(0.4f, dt);
    if (down(GLFW_KEY_EQUAL)) g.exposure  *= std::pow(2.5f, dt);
    if (g.exposure < 0.05f) g.exposure = 0.05f;
    if (g.exposure > 20.f)  g.exposure = 20.f;

    // Black-hole body orientation: X/Y/Z axes, Shift reverses direction.
    const float rate = 1.1f * dt;
    const bool sh = down(GLFW_KEY_LEFT_SHIFT) || down(GLFW_KEY_RIGHT_SHIFT);
    const float s = sh ? -1.f : 1.f;
    if (down(GLFW_KEY_X)) g.oriX += s * rate;
    if (down(GLFW_KEY_Y)) g.oriY += s * rate;
    if (down(GLFW_KEY_Z)) g.oriZ += s * rate;

    clampCamera();
}
} // namespace

// ---------------------------------------------------------------------------
int main()
{
    LOG_INFO("BlackHoleCUDAVulkan starting (Schwarzschild geodesic ray tracer)");
    VulkanContext vk;
    CudaInterop   cuda;

    try
    {
        vk.init(kWidth, kHeight, "Schwarzschild Black Hole - CUDA + Vulkan");

        int cudaDev = CudaInterop::findCudaDeviceByUUID(vk.deviceUUID());
        if (cudaDev < 0)
            throw std::runtime_error(
                "No CUDA device matches the selected Vulkan device UUID. "
                "CUDA-Vulkan interop requires rendering and presenting on the "
                "same NVIDIA GPU.");

        cuda.init(cudaDev, (int)vk.width(), (int)vk.height(),
                  vk.interopMemoryHandle(), vk.interopAllocSize(), vk.interopBufferSize(),
                  vk.semVkToCudaHandle(), vk.semCudaToVkHandle());

        GLFWwindow* win = vk.window();
        glfwSetMouseButtonCallback(win, onMouseButton);
        glfwSetCursorPosCallback(win, onCursorPos);
        glfwSetScrollCallback(win, onScroll);
        glfwSetKeyCallback(win, onKey);

        LOG_INFO("Controls: LMB drag=orbit  wheel/W/S=zoom  A/D/Q/E=orbit  "
                 "-/= exposure  1/2/3 quality  F1 disk  F11 fullscreen  "
                 "SPACE pause  ESC quit");
        LOG_INFO("Models:   F2 Schwarzschild  F3 Reissner-Nordstrom  F4 Kerr  "
                 "F5 Kerr-Newman  |  [/] spin a* (+/-)  ,/. charge q  |  B bloom");
        LOG_INFO("Orient:   X/Y/Z rotate BH axes (Shift=reverse)  R=reset view  |  "
                 "N naked  H hot spots  |  hold still for progressive AA");

        RenderParams params;
        params.width  = (int)vk.width();
        params.height = (int)vk.height();
        params.swapRB = vk.swapRB() ? 1 : 0;
        applyBlackHole(params);
        g.bhDirty = false;

        using clock = std::chrono::steady_clock;
        auto  prevT      = clock::now();
        auto  statT      = prevT;
        int   statFrames = 0;
        double statKernel = 0.0, statPresent = 0.0;
        float  diskTime   = 0.f;
        uint64_t frame    = 0;

        while (!glfwWindowShouldClose(win))
        {
            glfwPollEvents();

            // Skip rendering entirely while minimized
            int fbw = 0, fbh = 0;
            glfwGetFramebufferSize(win, &fbw, &fbh);
            if (fbw == 0 || fbh == 0) { glfwWaitEvents(); continue; }

            if (g.fullscreenToggleRequested)
            {
                g.fullscreenToggleRequested = false;
                cuda.sync();
                vk.waitIdle();
                cuda.releaseFrameResources();
                vk.toggleFullscreen();
                vk.recreateDisplayResources();
                cuda.resize((int)vk.width(), (int)vk.height(),
                            vk.interopMemoryHandle(), vk.interopAllocSize(),
                            vk.interopBufferSize());
                params.width  = (int)vk.width();
                params.height = (int)vk.height();
                params.swapRB = vk.swapRB() ? 1 : 0;
                LOG_INFO("Render size: %dx%d%s", params.width, params.height,
                         vk.fullscreen() ? " fullscreen" : " windowed");
            }

            auto  nowT = clock::now();
            float dt   = std::chrono::duration<float>(nowT - prevT).count();
            prevT = nowT;
            if (dt > 0.1f) dt = 0.1f;
            handleHeldKeys(win, dt);
            if (!g.animPaused) diskTime += dt;

            params.width  = (int)vk.width();
            params.height = (int)vk.height();
            params.swapRB = vk.swapRB() ? 1 : 0;
            if (g.bhDirty) { applyBlackHole(params); g.bhDirty = false; }
            fillCamera(params);
            params.exposure    = g.exposure;
            params.diskEnabled = g.diskOn ? 1 : 0;
            params.diskTime    = diskTime;
            params.dPhi        = g.dPhi;
            params.maxSteps    = g.maxSteps;
            params.bloomEnabled    = g.bloomOn ? 1 : 0;
            params.hotSpotsEnabled = (g.hotSpotsOn && g.diskOn) ? 1 : 0;
            params.hotSpotStrength = 0.65f;

            // --- Progressive temporal anti-aliasing bookkeeping ---------
            // Anything that changes the rendered geometry resets the
            // accumulation; exposure and bloom are applied after
            // accumulation and therefore do NOT reset it. Hot-spot toggle
            // resets so the ring structure updates immediately.
            static bool  viewInit = false;
            static float3 pPos{}, pFwd{};
            static int   pModel = -1, pDisk = -1, pSteps = -1;
            static int   pWidth = -1, pHeight = -1;
            static float pA = -1.f, pQ = -1.f, pPhi = -1.f;
            static int   pNaked = -1, pHot = -1;
            static float pOriX = 1e9f, pOriY = 1e9f, pOriZ = 1e9f;
            bool viewChanged = !viewInit
                || pWidth != params.width || pHeight != params.height
                || pPos.x != params.camPos.x || pPos.y != params.camPos.y
                || pPos.z != params.camPos.z
                || pFwd.x != params.camForward.x || pFwd.y != params.camForward.y
                || pFwd.z != params.camForward.z
                || pModel != params.model || pA != params.aSpin
                || pQ != params.Qc || pPhi != params.dPhi
                || pSteps != params.maxSteps || pDisk != params.diskEnabled
                || pNaked != params.allowNaked
                || pHot != params.hotSpotsEnabled
                || pOriX != g.oriX || pOriY != g.oriY || pOriZ != g.oriZ;
            pPos = params.camPos; pFwd = params.camForward;
            pModel = params.model; pA = params.aSpin; pQ = params.Qc;
            pPhi = params.dPhi; pSteps = params.maxSteps;
            pDisk = params.diskEnabled;
            pNaked = params.allowNaked;
            pHot = params.hotSpotsEnabled;
            pOriX = g.oriX; pOriY = g.oriY; pOriZ = g.oriZ;
            pWidth = params.width; pHeight = params.height;
            viewInit = true;

            static int sampleIndex = 0;
            if (viewChanged) sampleIndex = 0;
            params.sampleIndex = sampleIndex;
            params.accumMode = viewChanged ? 0
                             : ((g.animPaused || !g.diskOn) ? 1 : 2);
            if (sampleIndex < 4096) ++sampleIndex;

            // 1) CUDA renders into the shared Vulkan buffer (GPU-side sync)
            float kernelMs = cuda.render(params, frame > 0);

            // 2) Vulkan copies the shared buffer to the swapchain + presents
            VulkanFrameStats fs = vk.drawFrame();
            if (!fs.success)
            {
                LOG_ERROR("Presentation failed; aborting main loop");
                break;
            }

            // ---- statistics ----
            ++frame;
            ++statFrames;
            statKernel  += kernelMs;
            statPresent += fs.presentMs;
            double statDt = std::chrono::duration<double>(nowT - statT).count();
            if (statDt >= 0.5 && statFrames > 0)
            {
                double fps = statFrames / statDt;
                double avgK = statKernel / statFrames;
                double avgP = statPresent / statFrames;
                char title[256];
                std::snprintf(title, sizeof(title),
                              "%s Black Hole | a*=%.2f q=%.2f | %.1f FPS | "
                              "CUDA %.2f ms | VK %.2f ms | spp %d | r=%.1f | exp %.2f",
                              bhModelName(g.model),
                              params.aSpin / params.M, params.Qc / params.M,
                              fps, avgK, avgP,
                              params.sampleIndex + 1, g.cam.distance, g.exposure);
                glfwSetWindowTitle(win, title);

                static int consoleDiv = 0;
                if (++consoleDiv >= 4) // console log every ~2 s
                {
                    LOG_INFO("FPS %.1f | kernel %.2f ms | present %.2f ms | "
                             "cam(az %.2f, el %.2f, r %.1f)",
                             fps, avgK, avgP,
                             g.cam.azimuth, g.cam.elevation, g.cam.distance);
                    consoleDiv = 0;
                }
                statT = nowT;
                statFrames = 0;
                statKernel = statPresent = 0.0;
            }
        }

        LOG_INFO("Shutting down after %llu frames", (unsigned long long)frame);
        cuda.sync();
        vk.waitIdle();
        cuda.cleanup();
        vk.cleanup();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Fatal: %s", e.what());
        cuda.cleanup();
        vk.cleanup();
        return 1;
    }
    return 0;
}
