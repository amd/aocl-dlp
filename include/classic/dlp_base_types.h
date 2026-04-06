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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#ifndef DLP_BASE_TYPES_H
#define DLP_BASE_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#include "classic/dlp_defines.h"
#include "classic/dlp_errors.h"

#ifndef TRUE
#define TRUE true
#endif

#ifndef FALSE
#define FALSE false
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef DLP_IS_BUILDING_LIBRARY
#define DLP_CLASSIC_EXPORT __declspec(dllexport)
#else
#define DLP_CLASSIC_EXPORT
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define DLP_CLASSIC_EXPORT __attribute__((visibility("default")))
#else
#define DLP_CLASSIC_EXPORT
#endif

// Determine if we are on a 64-bit or 32-bit architecture.
#if defined(_M_X64) || defined(__x86_64) || defined(__aarch64__)               \
    || defined(_ARCH_PPC64) || defined(__s390x__) || defined(_LP64)
#define DLP_CLASSIC_ARCH_64
#else
#define DLP_CLASSIC_ARCH_32
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__ICC)                  \
    || defined(__IBMC__)
#define DLP_CLASSIC_THREAD_LOCAL __thread
#elif defined(_MSC_VER)
#define DLP_CLASSIC_THREAD_LOCAL __declspec(thread)
#else
#define DLP_CLASSIC_THREAD_LOCAL
#error "Thread-local storage not supported on this compiler"
#endif

#ifdef DLP_CLASSIC_ARCH_64
/** @brief Matrix buffer size type (64-bit on 64-bit platforms). */
typedef uint64_t msz_t;
#else
/** @brief Matrix buffer size type (32-bit on 32-bit platforms). */
typedef uint32_t msz_t;
#endif

#ifdef __cplusplus
#define DLP_INLINE inline
#else
#define DLP_INLINE static
#endif

/** @brief Matrix dimension type. Used for m, n, k, leading dimensions. */
typedef int64_t md_t;

/**
 * @brief Loop iteration type, compatible with md_t.
 *
 * Using int64_t avoids sign-compare warnings when comparing
 * with dimension values.
 */
typedef int64_t iter_t;

/**
 * @brief Opaque pointer-sized handle type.
 *
 * This is not itself a function-pointer type; it is an opaque handle that
 * may be used internally to store callable addresses or other implementation
 * specific pointers.
 */
typedef void* opaq_fp_t;

/** @brief Transpose options for matrix operations. */
typedef enum
{
    DLP_NO_TRANSPOSE = 0,  /**< No transpose */
    DLP_TRANSPOSE,         /**< Transpose */
    DLP_CONJ_NO_TRANSPOSE, /**< Conjugate, no transpose */
    DLP_CONJ_TRANSPOSE,    /**< Conjugate transpose */
    DLP_PACKED,            /**< Packed format */
} dlp_trans_t;

#define dlp_min(a, b)  ((a) < (b) ? (a) : (b))
#define dlp_max(a, b)  ((a) > (b) ? (a) : (b))
#define dlp_abs(a)     ((a) <= 0 ? -(a) : (a))
#define dlp_fmin(a, b) dlp_min(a, b)

#endif // DLP_BASE_TYPES_H
