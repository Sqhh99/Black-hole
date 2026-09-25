// ---------------------------------------------------------------------------
// snapshot.cu -- headless still renderer (no Vulkan / window).
//
// Runs the exact interactive pipeline (launchRenderPipeline: jittered
// geodesic samples -> progressive accumulation -> bloom -> tone mapping)
// for N samples and writes the result as a PNG. Useful for offline stills,
// look development and before/after comparisons.
//
//   blackhole_snapshot out.png [key=value ...]
//
// Keys (defaults = interactive app defaults):
//   model=0..3   (0 Schwarzschild, 1 RN, 2 Kerr, 3 Kerr-Newman)
//   a=0.7 q=0.4  (spin a*, charge q; applied per model like the app)
//   az=0.6 el=0.2 dist=16   (orbit camera, radians / code units)
//   fov=60       (vertical field of view, degrees)
//   spp=64 w=1280 h=720 exp=0.85 bloom=1 disk=1 spots=1 t=0 quality=2
//   temp=, emis=, abs=, glare=   (override disk T [K], brightness, opacity,
//                                 glare strength; defaults from RenderParams)
// ---------------------------------------------------------------------------
#include "render_params.h"
#include "metric.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" cudaError_t launchRenderPipeline(uchar4* out, float4* accum,
                                            float4* bloomA, float4* bloomB,
                                            const RenderParams& p,
                                            cudaStream_t stream);

namespace
{
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {            \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                 __FILE__, __LINE__); std::exit(2); } } while (0)

// ---- minimal PNG writer (zlib "stored" blocks, no compression) ----------
uint32_t crcTable[256];
void initCrc()
{
    for (uint32_t n = 0; n < 256; ++n)
    {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crcTable[n] = c;
    }
}
uint32_t crc(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu)
{
    for (size_t i = 0; i < n; ++i) c = crcTable[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}
void put32(std::vector<uint8_t>& v, uint32_t x)
{
    v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));  v.push_back((uint8_t)x);
}
void chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data)
{
    put32(out, (uint32_t)data.size());
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    put32(out, crc(td.data(), td.size()) ^ 0xFFFFFFFFu);
}
bool writePng(const char* path, const std::vector<uint8_t>& rgb, int w, int h)
{
    initCrc();
    std::vector<uint8_t> raw;
    raw.reserve((size_t)(w * 3 + 1) * h);
    for (int y = 0; y < h; ++y)
    {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3,
                   rgb.begin() + (size_t)(y + 1) * w * 3);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw.size(); off += 65535)
    {
        size_t n = raw.size() - off < 65535 ? raw.size() - off : 65535;
        uint16_t len = (uint16_t)n, nlen = (uint16_t)~len;
        z.push_back(off + n >= raw.size() ? 1 : 0);
        z.push_back((uint8_t)(len & 0xFF)); z.push_back((uint8_t)(len >> 8));
        z.push_back((uint8_t)(nlen & 0xFF)); z.push_back((uint8_t)(nlen >> 8));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    }
    put32(z, (b << 16) | a);

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put32(ihdr, w); put32(ihdr, h);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

// Same orbit-camera construction as main.cpp (body orientation = identity).
void fillCamera(RenderParams& p, float az, float el, float dist, float fovDeg)
{
    float ca = std::cos(az), sa = std::sin(az);
    float ce = std::cos(el), se = std::sin(el);
    float px = dist * ce * ca, py = dist * se, pz = dist * ce * sa;
    float fl = std::sqrt(px * px + py * py + pz * pz);
    float fx = -px / fl, fy = -py / fl, fz = -pz / fl;
    float rx = -fz, rz = fx;
    float rl = std::sqrt(rx * rx + rz * rz);
    rx /= rl; rz /= rl;
    float ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
    float ul = std::sqrt(ux * ux + uy * uy + uz * uz);
    p.camPos     = {px, py, pz};
    p.camForward = {fx, fy, fz};
    p.camRight   = {rx, 0.f, rz};
    p.camUp      = {ux / ul, uy / ul, uz / ul};
    p.tanHalfFov = std::tan(fovDeg * 3.14159265f / 180.f * 0.5f);
    p.aspect     = (float)p.width / (float)p.height;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: blackhole_snapshot out.png [key=value ...]\n");
        return 1;
    }
    const char* outPath = argv[1];
    int   model = 0, spp = 64, w = 1280, h = 720, bloom = 1, disk = 1, spots = 1, quality = 2;
    float aStar = 0.7f, q = 0.4f, az = 0.6f, el = 0.2f, dist = 16.f, fov = 60.f;
    float exposure = 0.85f, t = 0.f;
    float temp = -1.f, emis = -1.f, absK = -1.f, bloomK = -1.f;
    for (int i = 2; i < argc; ++i)
    {
        const char* eq = std::strchr(argv[i], '=');
        if (!eq) continue;
        std::string k(argv[i], eq - argv[i]);
        const char* v = eq + 1;
        if      (k == "model")   model = std::atoi(v);
        else if (k == "spp")     spp = std::atoi(v);
        else if (k == "w")       w = std::atoi(v);
        else if (k == "h")       h = std::atoi(v);
        else if (k == "bloom")   bloom = std::atoi(v);
        else if (k == "disk")    disk = std::atoi(v);
        else if (k == "spots")   spots = std::atoi(v);
        else if (k == "quality") quality = std::atoi(v);
        else if (k == "a")       aStar = (float)std::atof(v);
        else if (k == "q")       q = (float)std::atof(v);
        else if (k == "az")      az = (float)std::atof(v);
        else if (k == "el")      el = (float)std::atof(v);
        else if (k == "dist")    dist = (float)std::atof(v);
        else if (k == "fov")     fov = (float)std::atof(v);
        else if (k == "exp")     exposure = (float)std::atof(v);
        else if (k == "t")       t = (float)std::atof(v);
        else if (k == "temp")    temp = (float)std::atof(v);
        else if (k == "emis")    emis = (float)std::atof(v);
        else if (k == "abs")     absK = (float)std::atof(v);
        else if (k == "glare")   bloomK = (float)std::atof(v);
    }
    if (spp < 1) spp = 1;

    RenderParams P;
    P.width = w; P.height = h; P.swapRB = 0;
    float M = 0.5f;
    float aS = aStar, qq = q;
    BHDerived d = bhValidateAndDerive(model, M, aS, qq, false);
    P.model = model; P.M = M; P.aSpin = d.a; P.Qc = d.Q;
    P.rPlus = d.rPlus; P.rPhoton = d.rPhoton; P.rErgo = d.rErgo;
    P.diskInner = d.rIsco; P.diskOuter = 8.0f;
    fillCamera(P, az, el, dist, fov);
    P.exposure = exposure;
    P.diskEnabled = disk;
    P.diskTime = t;
    P.hotSpotsEnabled = (spots && disk) ? 1 : 0;
    P.hotSpotStrength = 0.65f;
    P.bloomEnabled = bloom;
    if (temp   > 0.f)  P.diskTemp = temp;
    if (emis   >= 0.f) P.diskEmisScale = emis;
    if (absK   >= 0.f) P.diskAbsScale = absK;
    if (bloomK >= 0.f) P.bloomStrength = bloomK;
    if (quality == 1)      { P.dPhi = 0.020f; P.maxSteps = 1200; }
    else if (quality == 3) { P.dPhi = 0.007f; P.maxSteps = 3500; }
    else                   { P.dPhi = 0.012f; P.maxSteps = 2000; }

    size_t n = (size_t)w * h;
    int bw = (w + 1) / 2, bh = (h + 1) / 2;
    uchar4* dOut; float4 *dAcc, *dBA, *dBB;
    CK(cudaMalloc(&dOut, n * sizeof(uchar4)));
    CK(cudaMalloc(&dAcc, n * sizeof(float4)));
    CK(cudaMalloc(&dBA, (size_t)bw * bh * sizeof(float4)));
    CK(cudaMalloc(&dBB, (size_t)bw * bh * sizeof(float4)));

    // Warm-up / first sample (mode 0) is timed separately from accumulation.
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < spp; ++s)
    {
        P.sampleIndex = s;
        P.accumMode = (s == 0) ? 0 : 1;
        CK(launchRenderPipeline(dOut, dAcc, dBA, dBB, P, 0));
        if (s == 0) CK(cudaDeviceSynchronize());
        if (s == 0) t0 = std::chrono::steady_clock::now();
    }
    CK(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

    std::vector<uchar4> px(n);
    CK(cudaMemcpy(px.data(), dOut, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> rgb(n * 3);
    for (size_t i = 0; i < n; ++i)
    {
        rgb[i * 3 + 0] = px[i].x; rgb[i * 3 + 1] = px[i].y; rgb[i * 3 + 2] = px[i].z;
    }
    if (!writePng(outPath, rgb, w, h))
    {
        std::fprintf(stderr, "cannot write %s\n", outPath);
        return 1;
    }
    if (spp > 1)
        std::printf("%s  %dx%d  spp=%d  %.2f ms/frame\n", outPath, w, h, spp,
                    ms / (spp - 1));
    else
        std::printf("%s  %dx%d  spp=1\n", outPath, w, h);
    cudaFree(dOut); cudaFree(dAcc); cudaFree(dBA); cudaFree(dBB);
    return 0;
}
