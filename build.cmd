@echo off
rem ===========================================================================
rem build.cmd - build entry point for BlackHoleCUDAVulkan (Windows x64)
rem
rem Usage:
rem   build.cmd                 configure + build (Release)
rem   build.cmd configure       configure only
rem   build.cmd build           configure (if needed) + build
rem   build.cmd run             build + run
rem   build.cmd test            build + run the GPU verification suite
rem   build.cmd clean           remove the build directory
rem   build.cmd rebuild         clean + configure + build
rem   build.cmd <action> debug  use the Debug configuration
rem
rem Requirements:
rem   - Visual Studio 2026 with the "Desktop development with C++" workload
rem   - CMake 3.24+ (bundled with VS or on PATH)
rem   - NVIDIA CUDA Toolkit (12.x recommended) with VS integration
rem   - Vulkan SDK (VULKAN_SDK environment variable set by its installer)
rem   - vcpkg (VCPKG_ROOT environment variable, or installed at C:\vcpkg)
rem ===========================================================================
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=build"

set "CONFIG=Release"
if /I "%~2"=="debug" set "CONFIG=Debug"

rem ---- locate vcpkg -------------------------------------------------------
if not defined VCPKG_ROOT (
    if exist "C:\vcpkg\scripts\buildsystems\vcpkg.cmake" (
        set "VCPKG_ROOT=C:\vcpkg"
    ) else (
        echo [ERROR] VCPKG_ROOT is not set and C:\vcpkg was not found.
        echo         Install vcpkg ^(https://github.com/microsoft/vcpkg^) and set VCPKG_ROOT.
        exit /b 1
    )
)
if not exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo [ERROR] vcpkg toolchain not found under "%VCPKG_ROOT%".
    exit /b 1
)

rem ---- sanity checks -------------------------------------------------------
if not defined VULKAN_SDK (
    echo [ERROR] VULKAN_SDK is not set. Install the LunarG Vulkan SDK.
    exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
    if not defined CUDA_PATH (
        echo [ERROR] CUDA Toolkit not found ^(nvcc not on PATH, CUDA_PATH not set^).
        exit /b 1
    )
)

if /I "%ACTION%"=="clean"   goto :clean
if /I "%ACTION%"=="rebuild" goto :rebuild
if /I "%ACTION%"=="configure" goto :configure
if /I "%ACTION%"=="build"   goto :build
if /I "%ACTION%"=="run"     goto :run
if /I "%ACTION%"=="test"    goto :test
echo [ERROR] Unknown action "%ACTION%". Use configure ^| build ^| run ^| test ^| clean ^| rebuild.
exit /b 1

rem ---------------------------------------------------------------------------
:configure
echo [INFO] Configuring (%CONFIG%) ...
rem Let CMake pick the newest installed Visual Studio generator (VS 2026).
rem To pin it explicitly, add:  -G "Visual Studio 18 2026"
cmake -S "%PROJECT_DIR%." -B "%BUILD_DIR%" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
      -DVCPKG_TARGET_TRIPLET=x64-windows
if errorlevel 1 (
    echo [ERROR] CMake configure failed.
    exit /b 1
)
if /I "%ACTION%"=="configure" exit /b 0
goto :dobuild

rem ---------------------------------------------------------------------------
:build
if not exist "%BUILD_DIR%\CMakeCache.txt" goto :configure
:dobuild
echo [INFO] Building (%CONFIG%) ...
cmake --build "%BUILD_DIR%" --config %CONFIG% --parallel
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)
echo [INFO] Build succeeded: "%BUILD_DIR%\%CONFIG%\blackhole.exe"
if /I "%ACTION%"=="build" exit /b 0
if /I "%ACTION%"=="rebuild" exit /b 0
if /I "%ACTION%"=="test" goto :dotest
goto :dorun

rem ---------------------------------------------------------------------------
:run
if not exist "%BUILD_DIR%\%CONFIG%\blackhole.exe" goto :build
:dorun
echo [INFO] Running blackhole.exe ...
"%BUILD_DIR%\%CONFIG%\blackhole.exe"
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
:test
if not exist "%BUILD_DIR%\CMakeCache.txt" goto :configure
echo [INFO] Building verification suite (%CONFIG%) ...
cmake --build "%BUILD_DIR%" --config %CONFIG% --target blackhole_tests --parallel
if errorlevel 1 (
    echo [ERROR] Test build failed.
    exit /b 1
)
:dotest
echo [INFO] Running blackhole_tests.exe ...
"%BUILD_DIR%\%CONFIG%\blackhole_tests.exe"
exit /b %errorlevel%

rem ---------------------------------------------------------------------------
:clean
echo [INFO] Removing "%BUILD_DIR%" ...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
echo [INFO] Clean done.
exit /b 0

:rebuild
call :clean
goto :configure
