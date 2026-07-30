/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @file dlp_example_utils.h
 * @brief Helpers shared by the AOCL-DLP classic example programs.
 *
 * This header is intentionally scoped to the examples and is NOT part of the
 * installed AOCL-DLP public API. Utilities used only for demonstration and
 * timing (and therefore not needed by the library itself) live here so that
 * the library headers stay free of definitions that would otherwise be unused
 * in every library translation unit.
 */

#ifndef DLP_EXAMPLE_UTILS_H
#define DLP_EXAMPLE_UTILS_H

/* ----------------------------------------------------------------------
 * High-resolution monotonic timer (elapsed seconds since an unspecified
 * epoch). Intended for measuring durations in the examples, not for
 * obtaining wall-clock / calendar time.
 *
 * Marked `static inline` so that translation units which include this header
 * without calling the helper do not trigger -Wunused-function.
 *
 * On failure the helper returns 0.0; callers use it only for relative
 * (end - start) timing in demonstration code.
 * -------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)

/* NOMINMAX keeps <windows.h> from defining min()/max() macros, matching the
 * guard used in classic/dlp_compat.h. */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static inline double
dlp_get_time_sec(void)
{
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0) {
        /* If the performance counter is unavailable, avoid a divide-by-zero. */
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
            return 0.0;
        }
    }
    LARGE_INTEGER counter;
    if (!QueryPerformanceCounter(&counter)) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

#else /* POSIX */

/* clock_gettime requires _POSIX_C_SOURCE >= 199309L. This define only has an
 * effect if it precedes the first inclusion of any system header in the
 * translation unit, so include this header before other libc headers (or set
 * the macro via the build). */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>

static inline double
dlp_get_time_sec(void)
{
    struct timespec ts = { 0, 0 };
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

#endif /* _WIN32 */

#endif /* DLP_EXAMPLE_UTILS_H */
