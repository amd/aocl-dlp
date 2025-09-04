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

#define _POSIX_C_SOURCE 199309L /* For clock_gettime and CLOCK_MONOTONIC */

#include <stdlib.h>

#include "gemm_utils/lpgemm_utils.h"
#include "sys_utils/lpgemm_sys.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <unistd.h>
#endif

#include <time.h>

uint64_t
lpgemm_gettid(void)
{
#ifdef DLP_ENABLE_OPENMP
    return (uint64_t)omp_get_thread_num();
#else
#ifdef DLP_ENABLE_PTHREADS
#ifndef _WIN32
    return (uint64_t)pthread_self();
#else
    return 0;
#endif
#else
    return 0;
#endif
#endif
}

uint64_t
lpgemm_getpid(void)
{
#if defined(__linux__)
    return (uint64_t)getpid();
#else
    return 0;
#endif
}

size_t
dlp_get_page_size()
{
    return 4096;
}

void*
dlp_malloc_page_aligned(size_t size, dlp_clsc_err_t* ret_err)
{
    const size_t align_size   = dlp_get_page_size();
    const size_t ptr_size     = sizeof(void*);
    size_t       align_offset = 0;
    void*        p_orig;
    int8_t*      p_byte;
    void**       p_addr;

    if (size == 0) {
        return NULL;
    }

    // Add the alignment size and the size of a pointer to the number
    // of bytes to allocate.
    size += align_size + ptr_size;

    // Call the allocation function.
    p_orig = malloc(size);

    if (p_orig == NULL) {
        dlp_print_msg("Cannot allocate memory. Aborting.", __FILE__, __LINE__);
        abort();
    }

    // The pseudo-return value isn't used yet.
    *ret_err = DLP_CLSC_SUCCESS;

    // Advance the pointer by one pointer element.
    p_byte = p_orig;
    p_byte += ptr_size;

    // Compute the offset to the desired alignment.
    msz_t mod_align_size = ((msz_t)p_byte % (msz_t)align_size);
    if (mod_align_size != 0) {
        align_offset = align_size - mod_align_size;
    }

    p_byte += align_offset;

    // Compute the address of the pointer element just before the start
    // of the aligned address, and store the original address there.
    p_addr  = (void**)(p_byte - ptr_size);
    *p_addr = p_orig;

    return p_byte;
}

void
dlp_free_page_aligned(void* p)
{
    const size_t ptr_size = sizeof(void*);
    void*        p_orig;
    int8_t*      p_byte;
    void**       p_addr;

    if (p == NULL) {
        return;
    }

    // Compute the address of the pointer element just before the start
    // of the aligned address, and recover the original address.
    p_byte = p;
    p_addr = (void**)(p_byte - ptr_size);
    p_orig = *p_addr;

    // Free the original pointer.
    free(p_orig);
}

static double gtod_ref_time_sec = 0.0;

double
dlp_clock(void)
{
#if defined(_WIN32) || defined(__CYGWIN__)

    LARGE_INTEGER clock_freq = { 0 };
    LARGE_INTEGER clock_val;
    BOOL          r_val;

    r_val = QueryPerformanceFrequency(&clock_freq);

    if (r_val == 0) {
        dlp_print_msg("QueryPerformanceFrequency() failed", __FILE__, __LINE__);
        abort();
    }

    r_val = QueryPerformanceCounter(&clock_val);

    if (r_val == 0) {
        dlp_print_msg("QueryPerformanceCounter() failed", __FILE__, __LINE__);
        abort();
    }

    return ((double)clock_val.QuadPart / (double)clock_freq.QuadPart);

#else

    double          the_time, norm_sec;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    if (gtod_ref_time_sec == 0.0) {
        gtod_ref_time_sec = (double)ts.tv_sec;
    }

    norm_sec = (double)ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;

#endif
}

double
dlp_clock_min_diff(double time_min, double time_start)
{
    double time_min_prev;
    double time_diff;

    // Save the old value.
    time_min_prev = time_min;

    time_diff = dlp_clock() - time_start;

    time_min = dlp_fmin(time_min, time_diff);

    // Assume that anything:
    // - under or equal to zero,
    // - under a nanosecond
    // is actually garbled due to the clocks being taken too closely together.
    if (time_min <= 0.0) {
        time_min = time_min_prev;
    } else if (time_min < 1.0e-9) {
        time_min = time_min_prev;
    }

    return time_min;
}
