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
 * @file dlp_compat.h
 * @brief Single centralized platform/compiler portability layer.
 *
 * ALL compiler-specific and platform-specific abstractions live in this one
 * header. No other file in the project should contain #ifdef _MSC_VER,
 * #ifdef _WIN32, __declspec, or any other compiler-conditional logic.
 *
 * On MSVC builds this header is force-included via /FI in
 * cmake/dlp_compiler_flags_windows.cmake, so its definitions are available
 * in every translation unit without an explicit #include.
 *
 * Sections:
 *   1.  NOMINMAX & lean Windows headers
 *   2.  Platform system headers
 *   3.  Force-inline
 *   4.  Alignment helpers
 *   5.  DLL export / visibility
 *   6.  Thread-local storage
 *   7.  __builtin_clz
 *   8.  Floating-point infinity
 *   9.  Aligned memory allocation
 *   10. Atomic operations
 *   11. High-resolution timer
 *   12. Environment variables
 *   13. OpenMP nested parallelism
 *   14. SIMD reinterpret-cast macros
 *   15. Post-ops dispatch (computed goto vs switch)
 *   16. Portable pthread-like primitives
 */

#ifndef DLP_COMPAT_H
#define DLP_COMPAT_H

/* ======================================================================
 * 1. NOMINMAX — must come before any Windows header
 * ====================================================================== */
#if defined(_MSC_VER) && !defined(NOMINMAX)
#define NOMINMAX
#endif

/* ======================================================================
 * 2. Platform system headers
 * ====================================================================== */
#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#include <errno.h>
#endif

#include <stdint.h>

/* ======================================================================
 * 3. Force-inline
 * ====================================================================== */
#if defined(_MSC_VER)
#define DLP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define DLP_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define DLP_ALWAYS_INLINE inline
#endif

/* ======================================================================
 * 4. Alignment helpers
 *
 * Variables: DLP_ALIGN_PREFIX(N) type name DLP_ALIGN_SUFFIX(N);
 * Structs:   typedef struct DLP_ALIGNED_STRUCT(N) { ... } t;
 * ====================================================================== */
#if defined(__clang__) || defined(__GNUC__)
#define DLP_ALIGN_PREFIX(N)
#define DLP_ALIGN_SUFFIX(N)    __attribute__((aligned(N)))
#define DLP_ALIGNED_STRUCT(N)  __attribute__((aligned(N)))
#elif defined(_MSC_VER)
#define DLP_ALIGN_PREFIX(N)    __declspec(align(N))
#define DLP_ALIGN_SUFFIX(N)
#define DLP_ALIGNED_STRUCT(N)  __declspec(align(N))
#else
#define DLP_ALIGN_PREFIX(N)
#define DLP_ALIGN_SUFFIX(N)    __attribute__((aligned(N)))
#define DLP_ALIGNED_STRUCT(N)  __attribute__((aligned(N)))
#endif

/* ======================================================================
 * 5. DLL export / visibility
 * ====================================================================== */
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

/* ======================================================================
 * 6. Thread-local storage
 * ====================================================================== */
#if defined(__GNUC__) || defined(__clang__) || defined(__ICC) \
    || defined(__IBMC__)
#define DLP_CLASSIC_THREAD_LOCAL __thread
#elif defined(_MSC_VER)
#define DLP_CLASSIC_THREAD_LOCAL __declspec(thread)
#else
#define DLP_CLASSIC_THREAD_LOCAL
#error "Thread-local storage not supported on this compiler"
#endif

/* ======================================================================
 * 7. __builtin_clz — count leading zeros
 * ====================================================================== */
#if defined(_MSC_VER) && !defined(DLP_BUILTIN_CLZ_DEFINED)
#define DLP_BUILTIN_CLZ_DEFINED
static __forceinline int
__builtin_clz(unsigned int x)
{
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - (int)idx;
}
#endif

/* ======================================================================
 * 8. Floating-point infinity
 *
 * MSVC rejects (float)(1.0/0.0) as C2124. math.h's HUGE_VAL works for
 * C code; C++ code can use std::numeric_limits.
 * ====================================================================== */
#if defined(_MSC_VER)
#include <math.h>
#define DLP_INF  HUGE_VAL
#define DLP_INFF HUGE_VALF
#else
#define DLP_INF  (1.0 / 0.0)
#define DLP_INFF ((float)(1.0 / 0.0))
#endif

#ifdef __cplusplus
#include <limits>
#define DLP_FLOAT_INF (std::numeric_limits<float>::infinity())
#else
#define DLP_FLOAT_INF ((float)DLP_INF)
#endif

/* ======================================================================
 * 9. Aligned memory allocation
 * ====================================================================== */
#if defined(_MSC_VER)
#define dlp_aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define dlp_aligned_free(ptr)              _aligned_free(ptr)
#else
#include <stdlib.h>
#define dlp_aligned_alloc(alignment, size) aligned_alloc((alignment), (size))
#define dlp_aligned_free(ptr)              free(ptr)
#endif

/* ======================================================================
 * 10. Atomic operations
 * ====================================================================== */
#if defined(_MSC_VER)

#define DLP_ATOMIC_LOAD_I64(ptr) \
    ((int64_t)_InterlockedOr64((volatile __int64*)(ptr), 0))
#define DLP_ATOMIC_STORE_I64(ptr, val) \
    ((void)_InterlockedExchange64((volatile __int64*)(ptr), (__int64)(val)))
#define DLP_ATOMIC_ADD_FETCH_I64(ptr, val) \
    ((int64_t)_InterlockedExchangeAdd64((volatile __int64*)(ptr), (__int64)(val)) + (val))
#define DLP_ATOMIC_FETCH_XOR_I64(ptr, val) \
    ((int64_t)_InterlockedXor64((volatile __int64*)(ptr), (__int64)(val)))
#define DLP_ATOMIC_INCREMENT_I64(ptr) \
    ((int64_t)_InterlockedIncrement64((volatile __int64*)(ptr)))

#define DLP_ATOMIC_LOAD_I32(ptr) \
    ((int32_t)_InterlockedOr((volatile long*)(ptr), 0))
#define DLP_ATOMIC_STORE_I32(ptr, val) \
    ((void)_InterlockedExchange((volatile long*)(ptr), (long)(val)))
#define DLP_ATOMIC_LOAD_BOOL(ptr) \
    (_InterlockedOr((volatile long*)(ptr), 0) != 0)
#define DLP_ATOMIC_STORE_BOOL(ptr, val) \
    ((void)_InterlockedExchange((volatile long*)(ptr), (long)((val) ? 1 : 0)))

#else /* GCC/Clang */

#define DLP_ATOMIC_LOAD_I64(ptr) \
    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define DLP_ATOMIC_STORE_I64(ptr, val) \
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define DLP_ATOMIC_ADD_FETCH_I64(ptr, val) \
    __atomic_add_fetch(ptr, val, __ATOMIC_ACQ_REL)
#define DLP_ATOMIC_FETCH_XOR_I64(ptr, val) \
    __atomic_fetch_xor(ptr, val, __ATOMIC_RELEASE)
#define DLP_ATOMIC_INCREMENT_I64(ptr) \
    __atomic_add_fetch(ptr, 1, __ATOMIC_ACQ_REL)

#define DLP_ATOMIC_LOAD_I32(ptr)       __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define DLP_ATOMIC_STORE_I32(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define DLP_ATOMIC_LOAD_BOOL(ptr)      __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define DLP_ATOMIC_STORE_BOOL(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)

#endif

/* ======================================================================
 * 11. High-resolution timer
 * ====================================================================== */
#ifndef DLP_COMPAT_TIMER_DEFINED
#define DLP_COMPAT_TIMER_DEFINED

#if defined(_WIN32) || defined(__CYGWIN__)

static double
dlp_get_time_sec(void)
{
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

#else /* POSIX */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>

static double
dlp_get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

#endif /* _WIN32 */
#endif /* DLP_COMPAT_TIMER_DEFINED */

/* ======================================================================
 * 12. Environment variables
 * ====================================================================== */
#if defined(_WIN32) || defined(__CYGWIN__)
#include <stdlib.h>
#define dlp_setenv(name, value, overwrite) _putenv_s((name), (value))
#else
#define dlp_setenv(name, value, overwrite) setenv((name), (value), (overwrite))
#endif

/* ======================================================================
 * 13. OpenMP nested parallelism
 *
 * omp_set_max_active_levels() requires OpenMP 5.0; MSVC only has 2.0.
 * ====================================================================== */
#ifdef _OPENMP
/* <omp.h> declares C++ templates; if this header is pulled in from within an
 * extern "C" block (common in C++ TUs that include the C API), those templates
 * would inherit C linkage and fail to compile. Force C++ linkage explicitly. */
#ifdef __cplusplus
extern "C++" {
#endif
#include <omp.h>
#ifdef __cplusplus
}
#endif
#endif

#if defined(_MSC_VER)
#define dlp_omp_set_nesting(levels) omp_set_nested(1)
#else
#define dlp_omp_set_nesting(levels) omp_set_max_active_levels(levels)
#endif

/* ======================================================================
 * 14. Inline assembly alignment hint
 *
 * GCC/Clang: __asm__(".p2align N") emits padding to align to 2^N boundary.
 * MSVC: no equivalent; expands to nothing.
 * ====================================================================== */
#if defined(_MSC_VER)
#define DLP_ASM_ALIGN(N)
#else
#define DLP_ASM_ALIGN(N)  __asm__(".p2align " #N "\n")
#endif

/* ======================================================================
 * 15. Platform-specific clock selector
 *
 * dlp_gemm_sys.c uses this to choose between QPC and clock_gettime
 * without an #ifdef _WIN32 in the source file.
 * ====================================================================== */
#if defined(_WIN32) || defined(__CYGWIN__)
#define DLP_CLOCK_USE_QPC 1
#else
#define DLP_CLOCK_USE_QPC 0
#endif

/* ======================================================================
 * 15. SIMD reinterpret-cast macros
 *
 * Defined below, OUTSIDE the main include guard, so that
 * dlp_simd_casts.h can define DLP_COMPAT_INCLUDE_SIMD and
 * re-#include this header to activate the SIMD section.
 * (See the block after `#endif DLP_COMPAT_H` at the bottom.)
 * ====================================================================== */

/* ======================================================================
 * 16. Post-ops dispatch (computed goto vs switch)
 *
 * GCC/Clang support computed gotos (labels-as-values). MSVC does not.
 * On MSVC we emulate the dispatch with a while/switch loop.
 * ====================================================================== */
#if defined(_MSC_VER)

#define POST_OP_LABEL_LASTK_SAFE_JUMP                                          \
    {                                                                          \
        uint64_t _po_target;                                                   \
        if ((post_ops_attr.is_last_k == TRUE) &&                               \
            (post_ops_list_temp != NULL)) {                                    \
            _po_target = post_ops_list_temp->op_code;                          \
        } else {                                                               \
            _po_target = 0;                                                    \
        }                                                                      \
        int _po_done = 0;                                                      \
        while (!_po_done) { switch (_po_target) {

#define POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR                            \
    post_ops_list_temp = post_ops_list_temp->next;                             \
    if (post_ops_list_temp != NULL) {                                          \
        _po_target = post_ops_list_temp->op_code;                              \
    } else {                                                                   \
        _po_target = 0;                                                        \
    }                                                                          \
    break;

#define POST_OPS_DISABLE_LABEL  case 0: default: _po_done = 1; break; }} }

/*
 * Per-case dispatch helpers — replace #ifdef _MSC_VER blocks in kernel files.
 *
 * DLP_POST_OP_CASE(N, LABEL): emits `case N: {` on MSVC, `LABEL: {` on GCC.
 * DLP_POST_OPS_DISABLE(LABEL): emits the disable case on MSVC, `LABEL:` on GCC.
 * DLP_POST_OPS_LABELS_DECL(...): nothing on MSVC (switch doesn't need label array).
 */
#define DLP_POST_OP_CASE(N, LABEL) case N: {
#define DLP_POST_OPS_DISABLE(LABEL) POST_OPS_DISABLE_LABEL
#define DLP_POST_OPS_LABELS_DECL(...)

#else /* GCC/Clang: computed gotos */

#define POST_OP_LABEL_LASTK_SAFE_JUMP                                          \
    if ((post_ops_attr.is_last_k == TRUE) && (post_ops_list_temp != NULL)) {   \
        goto* post_ops_labels[post_ops_list_temp->op_code];                    \
    } else {                                                                   \
        goto* post_ops_labels[0];                                              \
    }

#define POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR                            \
    post_ops_list_temp = post_ops_list_temp->next;                             \
    if (post_ops_list_temp != NULL) {                                          \
        goto* post_ops_labels[post_ops_list_temp->op_code];                    \
    } else {                                                                   \
        goto* post_ops_labels[0];                                              \
    }

#define POST_OPS_DISABLE_LABEL

#define DLP_POST_OP_CASE(N, LABEL) LABEL: {
#define DLP_POST_OPS_DISABLE(LABEL) LABEL:
#define DLP_POST_OPS_LABELS_DECL(...) static void* post_ops_labels[] = { __VA_ARGS__ };

#endif /* _MSC_VER (post-ops) */

/* ======================================================================
 * 16. Portable pthread-like primitives
 *
 * On Windows: INIT_ONCE → pthread_once, SRWLOCK → pthread_mutex.
 * On POSIX:   Thin wrappers around real pthreads.
 * ====================================================================== */
#ifdef __cplusplus
#define DLP_COMPAT_INLINE inline
#else
#define DLP_COMPAT_INLINE static
#endif

#if defined(_MSC_VER)

typedef INIT_ONCE dlp_pthread_once_t;

static BOOL
dlp_init_once_wrapper(dlp_pthread_once_t* once, void* param, void** context)
{
    (void)once;
    (void)context;
    typedef void (*callback)(void);
    ((callback)param)();
    return TRUE;
}

DLP_COMPAT_INLINE void
dlp_pthread_once(dlp_pthread_once_t* once, void (*init)(void))
{
    void* ctx = NULL;
#pragma warning(push)
#pragma warning(disable: 4152)
    InitOnceExecuteOnce(once, dlp_init_once_wrapper, (void*)init, &ctx);
#pragma warning(pop)
}

#define DLP_PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

typedef SRWLOCK dlp_pthread_mutex_t;
#define DLP_PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

DLP_COMPAT_INLINE int
dlp_pthread_mutex_lock(dlp_pthread_mutex_t* mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

DLP_COMPAT_INLINE int
dlp_pthread_mutex_trylock(dlp_pthread_mutex_t* mutex)
{
    return TryAcquireSRWLockExclusive(mutex) ? 0 : EBUSY;
}

DLP_COMPAT_INLINE int
dlp_pthread_mutex_unlock(dlp_pthread_mutex_t* mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

#else /* POSIX */

#include <pthread.h>

typedef pthread_once_t dlp_pthread_once_t;

DLP_COMPAT_INLINE void
dlp_pthread_once(dlp_pthread_once_t* once, void (*init)(void))
{
    pthread_once(once, init);
}

#define DLP_PTHREAD_ONCE_INIT         PTHREAD_ONCE_INIT

typedef pthread_mutex_t dlp_pthread_mutex_t;
#define DLP_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

DLP_COMPAT_INLINE int
dlp_pthread_mutex_lock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_lock(mutex);
}

DLP_COMPAT_INLINE int
dlp_pthread_mutex_trylock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_trylock(mutex);
}

DLP_COMPAT_INLINE int
dlp_pthread_mutex_unlock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_unlock(mutex);
}

#endif /* _MSC_VER (pthreads) */

#undef DLP_COMPAT_INLINE

#endif /* DLP_COMPAT_H */

/* ======================================================================
 * 14. SIMD reinterpret-cast macros  (outside main include guard)
 *
 * This section is deliberately placed AFTER the DLP_COMPAT_H guard so that
 * dlp_simd_casts.h can #define DLP_COMPAT_INCLUDE_SIMD and then re-include
 * this file to activate these definitions.
 *
 * GCC/Clang allow C-style casts between SIMD vector types.
 * MSVC requires the _mm*_cast* intrinsic functions.
 * BF16 reinterprets use union-based helpers (no standard intrinsic).
 * ====================================================================== */
#if defined(DLP_COMPAT_INCLUDE_SIMD) && !defined(DLP_COMPAT_SIMD_DEFINED)
#define DLP_COMPAT_SIMD_DEFINED

#include <immintrin.h>

#if defined(_MSC_VER)

/* 512-bit */
#define DLP_CAST_SI512_PS(v)  _mm512_castsi512_ps(v)
#define DLP_CAST_PS_SI512(v)  _mm512_castps_si512(v)
#define DLP_CAST_SI512_PD(v)  _mm512_castsi512_pd(v)
#define DLP_CAST_PD_SI512(v)  _mm512_castpd_si512(v)
#define DLP_CAST_PS_PD512(v)  _mm512_castps_pd(v)
#define DLP_CAST_PD_PS512(v)  _mm512_castpd_ps(v)

/* 256-bit */
#define DLP_CAST_SI256_PS(v)  _mm256_castsi256_ps(v)
#define DLP_CAST_PS_SI256(v)  _mm256_castps_si256(v)
#define DLP_CAST_SI256_PD(v)  _mm256_castsi256_pd(v)
#define DLP_CAST_PD_SI256(v)  _mm256_castpd_si256(v)
#define DLP_CAST_PS_PD256(v)  _mm256_castps_pd(v)
#define DLP_CAST_PD_PS256(v)  _mm256_castpd_ps(v)

/* 128-bit */
#define DLP_CAST_SI128_PS(v)  _mm_castsi128_ps(v)
#define DLP_CAST_PS_SI128(v)  _mm_castps_si128(v)
#define DLP_CAST_SI128_PD(v)  _mm_castsi128_pd(v)
#define DLP_CAST_PD_SI128(v)  _mm_castpd_si128(v)
#define DLP_CAST_PS_PD128(v)  _mm_castps_pd(v)
#define DLP_CAST_PD_PS128(v)  _mm_castpd_ps(v)

/* BF16 reinterprets via union (no standard intrinsic exists) */
static __forceinline __m512bh dlp_cast_si512_bh(__m512i v)  { union { __m512i i; __m512bh b; } u; u.i = v; return u.b; }
static __forceinline __m512i  dlp_cast_bh_si512(__m512bh v) { union { __m512bh b; __m512i i; } u; u.b = v; return u.i; }
static __forceinline __m256bh dlp_cast_si256_bh(__m256i v)  { union { __m256i i; __m256bh b; } u; u.i = v; return u.b; }
static __forceinline __m256i  dlp_cast_bh_si256(__m256bh v) { union { __m256bh b; __m256i i; } u; u.b = v; return u.i; }

#define DLP_CAST_SI512_BH(v)  dlp_cast_si512_bh(v)
#define DLP_CAST_BH_SI512(v)  dlp_cast_bh_si512(v)
#define DLP_CAST_SI256_BH(v)  dlp_cast_si256_bh(v)
#define DLP_CAST_BH_SI256(v)  dlp_cast_bh_si256(v)

/* Cross-width casts (extract lower half) */
#define DLP_CAST_SI512_SI256(v)  _mm512_castsi512_si256(v)
#define DLP_CAST_SI256_SI128(v)  _mm256_castsi256_si128(v)
#define DLP_CAST_PS512_PS256(v)  _mm512_castps512_ps256(v)
#define DLP_CAST_PS256_PS128(v)  _mm256_castps256_ps128(v)
#define DLP_CAST_PD512_PD256(v)  _mm512_castpd512_pd256(v)
#define DLP_CAST_PD256_PD128(v)  _mm256_castpd256_pd128(v)

#else /* GCC/Clang: C-style casts work fine */

#define DLP_CAST_SI512_PS(v)  ((__m512)(v))
#define DLP_CAST_PS_SI512(v)  ((__m512i)(v))
#define DLP_CAST_SI512_PD(v)  ((__m512d)(v))
#define DLP_CAST_PD_SI512(v)  ((__m512i)(v))
#define DLP_CAST_PS_PD512(v)  ((__m512d)(v))
#define DLP_CAST_PD_PS512(v)  ((__m512)(v))

#define DLP_CAST_SI256_PS(v)  ((__m256)(v))
#define DLP_CAST_PS_SI256(v)  ((__m256i)(v))
#define DLP_CAST_SI256_PD(v)  ((__m256d)(v))
#define DLP_CAST_PD_SI256(v)  ((__m256i)(v))
#define DLP_CAST_PS_PD256(v)  ((__m256d)(v))
#define DLP_CAST_PD_PS256(v)  ((__m256)(v))

#define DLP_CAST_SI128_PS(v)  ((__m128)(v))
#define DLP_CAST_PS_SI128(v)  ((__m128i)(v))
#define DLP_CAST_SI128_PD(v)  ((__m128d)(v))
#define DLP_CAST_PD_SI128(v)  ((__m128i)(v))
#define DLP_CAST_PS_PD128(v)  ((__m128d)(v))
#define DLP_CAST_PD_PS128(v)  ((__m128)(v))

#define DLP_CAST_SI512_BH(v)  ((__m512bh)(v))
#define DLP_CAST_BH_SI512(v)  ((__m512i)(v))
#define DLP_CAST_SI256_BH(v)  ((__m256bh)(v))
#define DLP_CAST_BH_SI256(v)  ((__m256i)(v))

/* Cross-width casts (extract lower half) */
#define DLP_CAST_SI512_SI256(v)  ((__m256i)(v))
#define DLP_CAST_SI256_SI128(v)  ((__m128i)(v))
#define DLP_CAST_PS512_PS256(v)  ((__m256)(v))
#define DLP_CAST_PS256_PS128(v)  ((__m128)(v))
#define DLP_CAST_PD512_PD256(v)  ((__m256d)(v))
#define DLP_CAST_PD256_PD128(v)  ((__m128d)(v))

#endif /* _MSC_VER (SIMD) */
#endif /* DLP_COMPAT_INCLUDE_SIMD && !DLP_COMPAT_SIMD_DEFINED */
