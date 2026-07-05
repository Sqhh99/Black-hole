# Schwarzschild Black Hole — Real-Time CUDA + Vulkan Renderer

A real-time, physically-motivated renderer of a non-rotating (Schwarzschild)
black hole for **Windows x64**. Per-pixel **null geodesics are numerically
integrated (RK4)** on the GPU with **CUDA**; the resulting HDR image is
tone-mapped in the kernel and handed to **Vulkan** for presentation via
**GPU-side external-memory interop** — the frame never touches the CPU.

No OpenGL, no Qt, no Python at runtime.

---

## 1. Requirements

| Component | Notes |
|---|---|
| Windows 10/11 x64 | NVIDIA GPU required (interop is single-GPU NVIDIA) |
| Visual Studio 2026 | "Desktop development with C++" workload |
| CMake ≥ 3.24 | bundled with VS or standalone |
| CUDA Toolkit 12.x | installed with Visual Studio integration (`nvcc` / `CUDA_PATH`) |
| Vulkan SDK (LunarG) | installer sets `VULKAN_SDK` |
| vcpkg | set `VCPKG_ROOT` (or install at `C:\vcpkg`) |

The only third-party library is **glfw3**, pulled automatically by the
vcpkg manifest (`vcpkg.json`) during CMake configure.

## 2. Build & Run

```bat
build.cmd configure   :: CMake configure (VS 2026 generator, x64, vcpkg toolchain)
build.cmd build       :: configure (if needed) + compile, Release
build.cmd run         :: build (if needed) + launch blackhole.exe
build.cmd clean       :: delete the build directory
build.cmd rebuild     :: clean + configure + build
build.cmd run debug   :: same actions with the Debug configuration
```

Manual equivalent:

```bat
cmake -S . -B build -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release --parallel
build\Release\blackhole.exe
```

Tip: for a much faster first compile targeting only your GPU, add
`-DCMAKE_CUDA_ARCHITECTURES=native` to the configure step (default builds
for SM 61/75/86/89).

## 3. Controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit camera (azimuth / elevation) |
| Mouse wheel, `W`/`S` | Camera distance (clamped to stay outside 2.2 rs) |
| `A`/`D`, `Q`/`E` | Azimuth / elevation via keyboard |
| `-` / `=` | Exposure down / up |
| `1` / `2` / `3` | Quality preset: fast / balanced (default) / high |
| `F1` | Toggle accretion disk |
| `SPACE` | Pause / resume disk animation |
| `ESC` | Quit |

FPS, average CUDA kernel time and Vulkan present time are shown in the
window title (updated every 0.5 s) and logged to the console every ~2 s.

## 4. How the CUDA ↔ Vulkan interop works (no CPU copies)

1. **Vulkan owns the memory.** `VulkanContext::createInteropBuffer` creates a
   `VkBuffer` (width × height × 4 bytes) whose device memory is allocated with
   `VkExportMemoryAllocateInfo` (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`)
   and exported as a Win32 handle via `vkGetMemoryWin32HandleKHR`.
2. **CUDA imports it.** `CudaInterop::init` calls `cudaImportExternalMemory`
   + `cudaExternalMemoryGetMappedBuffer`, yielding a device pointer that
   aliases the very same VRAM. The render kernel writes `uchar4` pixels
   directly into it.
3. **Same physical GPU is enforced.** Vulkan reports its
   `VkPhysicalDeviceIDProperties::deviceUUID`; the CUDA device is selected by
   matching `cudaDeviceProp::uuid`. If no match exists the app aborts with a
   clear error (interop across different GPUs is not supported).
4. **GPU-side synchronization with two exported binary semaphores**
   (`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT`, imported with
   `cudaImportExternalSemaphore`):
   - `semCudaToVk`: CUDA signals after the kernel; Vulkan's submit waits on it.
   - `semVkToCuda`: Vulkan signals when the copy finished; CUDA's next frame
     waits on it (skipped on frame 0).
5. **Presentation.** Each frame Vulkan submits a pre-recorded command buffer:
   layout barrier → `vkCmdCopyBufferToImage` into the acquired swapchain image
   → barrier to `PRESENT_SRC` → `vkQueuePresentKHR`. The copy is a pure
   GPU-to-GPU transfer; the image never leaves the GPU and is never mapped,
   read back, or re-uploaded by the CPU.

## 5. Physics & rendering effects implemented

- **True geodesic integration, not screen-space warping.** For every pixel a
  null geodesic is integrated in its orbital plane using the Binet equation
  `u''(φ) = (3/2) rs u² − u` (u = 1/r, geometric units rs = 1) with a
  **classic 4th-order Runge–Kutta** scheme and adjustable step `dφ`.
- **Robust termination:** capture at the horizon (u ≥ 1), escape beyond
  r = 60 rs with outward motion, NaN/Inf guards (treated as captured), a hard
  step budget, and a straight-line fallback for degenerate purely-radial rays.
- **Pitch-black event horizon** — captured rays contribute no light.
- **Gravitational lensing** of a procedural background starfield: the escape
  direction of the bent geodesic samples the sky, producing continuous
  distortion and Einstein-ring behaviour around the shadow.
- **Photon ring:** rays passing near the r = 1.5 rs photon sphere wind around
  the hole multiple times and sample the disk/sky repeatedly, so the bright
  thin ring at the shadow edge **emerges from the integration itself**.
- **Finite-thickness accretion disk** (3 rs → 12 rs, Gaussian vertical
  profile with H ∝ √r) sampled volumetrically along the geodesic with
  sub-stepping and self-absorption (transmittance), so the disk correctly
  appears in front of, behind (lensed over/under), and inside the photon ring.
- **Temperature profile** T ∝ r^(−3/4) (thin-disk scaling) mapped through a
  black-body colour fit; inner edge hotter/brighter, outer edge cooler/darker.
- **Relativistic Doppler beaming + gravitational redshift:** Keplerian orbital
  velocity gives the special-relativistic Doppler factor, combined with
  √(1 − rs/r); observed intensity scales as g⁴ and the black-body colour is
  shifted by g — the approaching side is visibly brighter and bluer.
- **HDR pipeline:** linear float3 accumulation → exposure → ACES tone map →
  gamma 2.2 → 8-bit output (with R/B swap for BGRA swapchains).
- Turbulent disk detail via differentially-rotating fBm noise, animated in
  real time (SPACE to pause).

## 6. Testing status — please read

This project was authored in a **Linux container without an NVIDIA GPU,
without the CUDA toolkit, without the Vulkan SDK, and without MSVC**, so the
Windows build and the live application **could not be compiled or executed
by the author environment**. What *was* verified:

- `black_hole_kernel.cu`, `cuda_interop.cpp`, `main.cpp` and all project
  headers pass strict host-compiler syntax/type checking (`g++ -std=c++17
  -Wall -Wextra -fsyntax-only`) against thin API shims.
- `vulkan_context.cpp` was reviewed manually and passes structural checks;
  it could not be shim-compiled economically.

Consequently, **first-build issues on real MSVC/NVCC/Vulkan cannot be ruled
out**, and all performance figures are **estimates, not measurements**:
on a desktop RTX-class GPU, 1280×720 at the default "balanced" preset
(dφ = 0.012, ≤ 2000 RK4 steps) is *expected* to run at interactive rates
(tens of FPS), with the "fast" preset comfortably real-time and "high"
substantially heavier. Please verify with the on-screen counters.

## 7. Known limitations

- **Schwarzschild only** — no Kerr metric, so no frame dragging, no
  spin-asymmetric shadow.
- **Approximate radiative transfer:** simple emission/absorption with
  heuristic scalings; no full frequency-dependent radiative transport, no
  polarization, no photon redshift applied to the background starfield.
- The disk is a phenomenological model (thin-disk temperature law + noise),
  not a GRMHD simulation.
- Fixed, non-resizable 1280×720 window (change `kWidth`/`kHeight` in
  `src/main.cpp` and rebuild).
- Binary-semaphore ping-pong serializes CUDA and Vulkan per frame (no
  multi-frame kernel pipelining).
- Single-GPU NVIDIA only; hybrid laptops must run the app on the NVIDIA GPU
  (Windows Graphics settings → High performance) or device matching fails.

## 8. Project layout

```
CMakeLists.txt              CMake project (CXX + CUDA, VS 2026 / x64)
build.cmd                   configure | build | run | clean | rebuild
vcpkg.json                  manifest (glfw3)
src/
  main.cpp                  window loop, orbit camera, input, stats
  vulkan_context.{h,cpp}    instance/device/swapchain, exportable buffer
                            + semaphores, copy-and-present command buffers
  cuda_interop.{h,cpp}      external memory/semaphore import, kernel dispatch,
                            CUDA event timing, device-UUID matching
  black_hole_kernel.cu      RK4 geodesic integration, disk shading, Doppler,
                            starfield, ACES tone mapping
  render_params.h           POD shared between host and device
  vec_math.cuh              device float2/3 math, hashing, fBm noise
  logger.h                  timestamped console logging
```
