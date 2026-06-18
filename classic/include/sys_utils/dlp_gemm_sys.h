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

#ifndef DLP_GEMM_SYS_UTILS_H
#define DLP_GEMM_SYS_UTILS_H

#include "classic/dlp_base_types.h"

/*
 * Portable pthread-like primitives (dlp_pthread_once_t, dlp_pthread_mutex_t,
 * DLP_PTHREAD_ONCE_INIT, DLP_PTHREAD_MUTEX_INITIALIZER, and their functions)
 * are defined in dlp_compat.h (Section 16), which is pulled in transitively
 * through dlp_base_types.h.
 */

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
