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
//   [ , ] ................... spin a* down / up      (Kerr, Kerr-Newman)
//   , . ................... charge q down / up     (RN, Kerr-Newman)
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

struct OrbitCamera
{
    float azimuth   = 0.6f;    // radians
    float elevation = 0.20f;   // radians above disk plane
    float distance  = 16.0f;   // in Schwarzschild radii
};

struct AppState
{
    OrbitCamera cam;
    float  exposure   = 1.0f;
    bool   diskOn     = true;
    bool   animPaused = false;
    float  dPhi       = 0.012f;
    int    maxSteps   = 2000;
    double lastX = 0.0, lastY = 0.0;
    bool   dragging = false;

    // Black hole model knobs. aStar/q are the user's stored settings; each
    // model applies only the parameters that pertain to it (Schwarzschild
    // ignores both, RN ignores spin, Kerr ignores charge).
    int    model   = BH_SCHWARZSCHILD;
    float  aStar   = 0.70f;
    float  q       = 0.40f;
    bool   bhDirty = true;

    bool   bloomOn = true;   // HDR bloom of the disk / photon ring
    bool   fullscreenToggleRequested = false;
};

AppState g;

// Validate the current model/spin/charge, derive horizon / photon-region /
// ISCO, and push everything into the shared parameter block.
void applyBlackHole(RenderParams& p)
{
    if (g.aStar < 0.f) g.aStar = 0.f;
    if (g.aStar > 0.995f) g.aStar = 0.995f;
    if (g.q < 0.f) g.q = 0.f;
    if (g.q > 0.995f) g.q = 0.995f;

    float M = 0.5f;                 // mass scale: rs = 2M = 1 code unit
    float aS = g.aStar, qq = g.q;
    BHDerived d = bhValidateAndDerive(g.model, M, aS, qq);
    if (d.clamped) { g.aStar = aS; g.q = qq; }

    p.model     = g.model;
    p.M         = M;
    p.aSpin     = d.a;
    p.Qc        = d.Q;
    p.rPlus     = d.rPlus;
    p.rPhoton   = d.rPhoton;
    p.rErgo     = d.rErgo;
    p.diskInner = d.rIsco;          // disk inner edge tracks the ISCO
    p.diskOuter = 12.0f;            // 24M

    bool useA = (g.model == BH_KERR || g.model == BH_KERR_NEWMAN);
    bool useQ = (g.model == BH_REISSNER_NORDSTROM || g.model == BH_KERR_NEWMAN);
    char aStr[16], qStr[16];
    if (useA) std::snprintf(aStr, sizeof(aStr), "%.3f", g.aStar);
    else      std::snprintf(aStr, sizeof(aStr), "-");
    if (useQ) std::snprintf(qStr, sizeof(qStr), "%.3f", g.q);
    else      std::snprintf(qStr, sizeof(qStr), "-");
    LOG_INFO("%s | M=%.2f  a*=%s  q=%s | r+=%.4f  r_photon=%.4f  "
             "r_ergo=%.4f  ISCO=%.4f%s",
             bhModelName(g.model), M, aStr, qStr,
             d.rPlus, d.rPhoton, d.rErgo, d.rIsco,
             d.clamped ? "  [parameters clamped to the extremal bound]" : "");
}

void clampCamera()
{
    const float elMax = 1.45f; // avoid pole singularity of the up vector
    if (g.cam.elevation >  elMax) g.cam.elevation =  elMax;
    if (g.cam.elevation < -elMax) g.cam.elevation = -elMax;
    if (g.cam.distance < 2.2f)  g.cam.distance = 2.2f;   // stay outside r = 2.2 rs
    if (g.cam.distance > 55.f)  g.cam.distance = 55.f;
}

void fillCamera(RenderParams& p)
{
    float ca = std::cos(g.cam.azimuth), sa = std::sin(g.cam.azimuth);
    float ce = std::cos(g.cam.elevation), se = std::sin(g.cam.elevation);
    float px = g.cam.distance * ce * ca;
    float py = g.cam.distance * se;
    float pz = g.cam.distance * ce * sa;

    // forward = look at origin
    float fl = std::sqrt(px * px + py * py + pz * pz);
    float fx = -px / fl, fy = -py / fl, fz = -pz / fl;

    // right = normalize(cross(forward, worldUp)), worldUp = (0,1,0)
    // cross(f, up) = (f.y*0 - f.z*1, f.z*0 - f.x*0, f.x*1 - f.y*0) = (-f.z, 0, f.x)
    float rxx = -fz, rxy = 0.f, rxz = fx;
    float rl = std::sqrt(rxx * rxx + rxy * rxy + rxz * rxz);
    if (rl < 1e-6f) { rxx = 1.f; rxy = 0.f; rxz = 0.f; rl = 1.f; }
    rxx /= rl; rxz /= rl;

    // up = cross(right, forward)
    float ux = rxy * fz - rxz * fy;
    float uy = rxz * fx - rxx * fz;
    float uz = rxx * fy - rxy * fx;

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

        LOG_INFO("Controls: LMB drag=orbit  wheel/W/S=zoom  A/D/Q/E=rotate  "
                 "-/= exposure  1/2/3 quality  F1 disk  F11 fullscreen  "
                 "SPACE pause  ESC quit");
        LOG_INFO("Models:   F2 Schwarzschild  F3 Reissner-Nordstrom  F4 Kerr  "
                 "F5 Kerr-Newman  |  [/] spin a*  ,/. charge q  |  B bloom");
        LOG_INFO("Quality:  hold the camera still and the image refines "
                 "itself (progressive anti-aliasing)");

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
            params.bloomEnabled = g.bloomOn ? 1 : 0;

            // --- Progressive temporal anti-aliasing bookkeeping ---------
            // Anything that changes the rendered geometry resets the
            // accumulation; exposure and bloom are applied after
            // accumulation and therefore do NOT reset it.
            static bool  viewInit = false;
            static float3 pPos{}, pFwd{};
            static int   pModel = -1, pDisk = -1, pSteps = -1;
            static int   pWidth = -1, pHeight = -1;
            static float pA = -1.f, pQ = -1.f, pPhi = -1.f;
            bool viewChanged = !viewInit
                || pWidth != params.width || pHeight != params.height
                || pPos.x != params.camPos.x || pPos.y != params.camPos.y
                || pPos.z != params.camPos.z
                || pFwd.x != params.camForward.x || pFwd.y != params.camForward.y
                || pFwd.z != params.camForward.z
                || pModel != params.model || pA != params.aSpin
                || pQ != params.Qc || pPhi != params.dPhi
                || pSteps != params.maxSteps || pDisk != params.diskEnabled;
            pPos = params.camPos; pFwd = params.camForward;
            pModel = params.model; pA = params.aSpin; pQ = params.Qc;
            pPhi = params.dPhi; pSteps = params.maxSteps;
            pDisk = params.diskEnabled;
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
