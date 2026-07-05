#pragma once
// ---------------------------------------------------------------------------
// logger.h - tiny timestamped console logger.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdarg>
#include <chrono>

enum class LogLevel { Info, Warn, Error };

inline void logMessage(LogLevel level, const char* fmt, ...)
{
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    double sec = duration<double>(steady_clock::now() - t0).count();

    const char* tag = (level == LogLevel::Info)  ? "INFO "
                    : (level == LogLevel::Warn)  ? "WARN "
                    :                              "ERROR";

    std::fprintf(stderr, "[%9.3f] [%s] ", sec, tag);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

#define LOG_INFO(...)  logMessage(LogLevel::Info,  __VA_ARGS__)
#define LOG_WARN(...)  logMessage(LogLevel::Warn,  __VA_ARGS__)
#define LOG_ERROR(...) logMessage(LogLevel::Error, __VA_ARGS__)
