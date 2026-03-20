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

#ifndef DLP_GEMM_SYS_UTILS_H
#define DLP_GEMM_SYS_UTILS_H

#include "classic/dlp_base_types.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#endif

#if defined(_MSC_VER)

#include <errno.h>

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

DLP_INLINE void
dlp_pthread_once(dlp_pthread_once_t* once, void (*init)(void))
{
    InitOnceExecuteOnce(once, dlp_init_once_wrapper, init, NULL);
}

#define DLP_PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

typedef SRWLOCK dlp_pthread_mutex_t;
#define DLP_PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

DLP_INLINE int
dlp_pthread_mutex_lock(dlp_pthread_mutex_t* mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

DLP_INLINE int
dlp_pthread_mutex_trylock(dlp_pthread_mutex_t* mutex)
{
    return TryAcquireSRWLockExclusive(mutex) ? 0 : EBUSY;
}

DLP_INLINE int
dlp_pthread_mutex_unlock(dlp_pthread_mutex_t* mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

#else

#include <pthread.h>

typedef pthread_once_t dlp_pthread_once_t;

DLP_INLINE void
dlp_pthread_once(dlp_pthread_once_t* once, void (*init)(void))
{
    pthread_once(once, init);
}

#define DLP_PTHREAD_ONCE_INIT         PTHREAD_ONCE_INIT

typedef pthread_mutex_t dlp_pthread_mutex_t;
#define DLP_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

DLP_INLINE int
dlp_pthread_mutex_lock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_lock(mutex);
}

DLP_INLINE int
dlp_pthread_mutex_trylock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_trylock(mutex);
}

DLP_INLINE int
dlp_pthread_mutex_unlock(dlp_pthread_mutex_t* mutex)
{
    return pthread_mutex_unlock(mutex);
}

#endif

DLP_CLASSIC_EXPORT size_t
dlp_get_page_size(void);

DLP_CLASSIC_EXPORT void*
dlp_malloc_page_aligned(size_t sz, dlp_clsc_err_t* ret_err);
DLP_CLASSIC_EXPORT void
dlp_free_page_aligned(void* p);

DLP_CLASSIC_EXPORT uint64_t
dlp_gemm_gettid(void);
DLP_CLASSIC_EXPORT uint64_t
dlp_gemm_getpid(void);

DLP_CLASSIC_EXPORT double
dlp_clock(void);
DLP_CLASSIC_EXPORT double
dlp_clock_min_diff(double time_min, double time_start);

#endif // DLP_GEMM_SYS_UTILS_H
