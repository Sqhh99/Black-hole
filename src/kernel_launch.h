#pragma once
// Thin wrapper around the CUDA triple-chevron launch. Under NVCC this is a
// real kernel launch; when the same sources are parsed by a plain host
// compiler for static analysis, it degrades to a direct call so the file
// remains parseable. Production builds always go through NVCC.
#ifdef __CUDACC__
#define KLAUNCH(kernel, grid, block, stream, ...) \
    kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#else
#define KLAUNCH(kernel, grid, block, stream, ...) kernel(__VA_ARGS__)
#endif
