cmake_minimum_required(VERSION 3.24)

project(BlackHoleCUDAVulkan
        VERSION 1.0.0
        DESCRIPTION "Real-time Schwarzschild black hole renderer (CUDA ray tracing + Vulkan presentation)"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Reasonable default GPU targets (Pascal .. Ada). Override with
#   -DCMAKE_CUDA_ARCHITECTURES=native   for a local-GPU-only fast build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)
endif()

# ---------------------------------------------------------------------------
# Dependencies
#   - Vulkan SDK   : from the LunarG SDK installation (VULKAN_SDK env var)
#   - CUDA Toolkit : from the NVIDIA CUDA installation
#   - glfw3        : from vcpkg (see vcpkg.json manifest)
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(blackhole
    src/main.cpp
    src/vulkan_context.cpp
    src/cuda_interop.cpp
    src/black_hole_kernel.cu
    src/vulkan_context.h
    src/cuda_interop.h
    src/render_params.h
    src/metric.cuh
    src/trace.cuh
    src/kernel_launch.h
    src/vec_math.cuh
    src/logger.h
)

target_compile_definitions(blackhole PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    VK_USE_PLATFORM_WIN32_KHR
)

if(MSVC)
    target_compile_options(blackhole PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive->)
endif()

# Pass host-compiler conformance flags through nvcc as well
target_compile_options(blackhole PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3 --use_fast_math>)

target_link_libraries(blackhole PRIVATE
    Vulkan::Vulkan
    glfw
    CUDA::cudart
)

set_target_properties(blackhole PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# ---------------------------------------------------------------------------
# Verification suite (CUDA console app; no Vulkan / GLFW dependency).
# Shares the render kernel TU so it exercises exactly the shipped integrators
# and the sanitizing launcher. Run via `build.cmd test`.
# ---------------------------------------------------------------------------
add_executable(blackhole_tests
    tests/test_main.cu
    src/black_hole_kernel.cu
)
target_include_directories(blackhole_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(blackhole_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(blackhole_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3 --use_fast_math>)
target_link_libraries(blackhole_tests PRIVATE CUDA::cudart)
set_target_properties(blackhole_tests PROPERTIES CUDA_RUNTIME_LIBRARY Static)

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Vulkan include     : ${Vulkan_INCLUDE_DIRS}")
