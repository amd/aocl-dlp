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

#ifndef LPGEMM_LOGGER_H
#define LPGEMM_LOGGER_H

#include <float.h>

#include "classic/aocl_gemm_post_ops.h"
#include "lpgemm_types.h"
#include "sys_utils/lpgemm_sys.h"

#ifdef AOCL_LPGEMM_LOGGER_SUPPORT

#define AOCL_LPGEMM_LOG_FILE_PRFX "aocl_gemm_log"
#define AOCL_LPGEMM_LOG_FILE_EXT  ".txt"

FILE*
lpgemm_start_logger_fn(double* aocl_lpgemm_logger_start_time);
void
lpgemm_stop_logger_fn(FILE* fd, double* aocl_lpgemm_logger_start_time);
void
lpgemm_get_post_ops_str(dlp_metadata_t* metadata, char* ops_str);
void
lpgemm_get_pre_ops_str(dlp_metadata_t* metadata, char* ops_str);
bool
is_logger_enabled();
void
lpgemm_write_logger_gemm_fn(FILE*           fd,
                            char*           op_type,
                            const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            const float     alpha,
                            const md_t      lda,
                            const char      mem_format_a,
                            const md_t      ldb,
                            const char      mem_format_b,
                            const float     beta,
                            const md_t      ldc,
                            dlp_metadata_t* metadata);
void
batch_lpgemm_write_logger_gemm_fn(FILE*            fd,
                                  char*            op_type,
                                  const char*      order,
                                  const char*      transa,
                                  const char*      transb,
                                  const md_t       group_count,
                                  const md_t*      group_size,
                                  const md_t*      m,
                                  const md_t*      n,
                                  const md_t*      k,
                                  const float*     alpha,
                                  const md_t*      lda,
                                  const char*      mem_format_a,
                                  const md_t*      ldb,
                                  const char*      mem_format_b,
                                  const float*     beta,
                                  const md_t*      ldc,
                                  dlp_metadata_t** metadata);

#define LPGEMM_START_LOGGER()                                                  \
    double aocl_lpgemm_logger_start_time = 0;                                  \
    FILE*  fd = lpgemm_start_logger_fn(&aocl_lpgemm_logger_start_time);

#define LPGEMM_STOP_LOGGER()                                                   \
    lpgemm_stop_logger_fn(fd, &aocl_lpgemm_logger_start_time);

#define LPGEMM_WRITE_LOGGER(...) lpgemm_write_logger_gemm_fn(fd, __VA_ARGS__);

#define BATCH_LPGEMM_WRITE_LOGGER(                                             \
    op_type, order, transa, transb, group_count, group_size, m, n, k, alpha,   \
    lda, mem_format_a, ldb, mem_format_b, beta, ldc, metadata)                 \
    {                                                                          \
        if ((is_logger_enabled()) && (fd != NULL)) {                           \
            char pre_ops_str[1024] = { 0 };                                    \
                                                                               \
            char post_ops_str[2048] = { 0 };                                   \
                                                                               \
            fprintf(fd, "%s:group_count=%ld\n", op_type, group_count);         \
            for (md_t i = 0; i < group_count; i++) {                           \
                lpgemm_get_pre_ops_str(metadata[i], pre_ops_str);              \
                lpgemm_get_post_ops_str(metadata[i], post_ops_str);            \
                fprintf(fd,                                                    \
                        "%ld %c %c %c %c %c %ld %ld %ld %ld %ld %ld "          \
                        ":pre_ops=[%s]:metadata=[%s] %f %f\n",                 \
                        group_size[i], order[i], transa[i], transb[i],         \
                        mem_format_a[i], mem_format_b[i], m[i], n[i], k[i],    \
                        lda[i], ldb[i], ldc[i], pre_ops_str, post_ops_str,     \
                        (float)(alpha[i]), (float)(beta[i]));                  \
            }                                                                  \
        }                                                                      \
    }

#else

#define LPGEMM_START_LOGGER(...)

#define LPGEMM_STOP_LOGGER(...)

#define LPGEMM_WRITE_LOGGER(...)

#define BATCH_LPGEMM_WRITE_LOGGER(...)

#endif

#endif // LPGEMM_LOGGER_H
