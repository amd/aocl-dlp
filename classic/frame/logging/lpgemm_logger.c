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

#include <string.h>

#include "bindings/c_wrappers/capi_env_config.h"
#include "logging/lpgemm_logger.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "sys_utils/lpgemm_sys.h"

#ifdef AOCL_LPGEMM_LOGGER_SUPPORT

bool
is_logger_enabled()
{
    return dlp_env_is_logger_enabled();
}

FILE*
lpgemm_start_logger_fn(double* aocl_lpgemm_logger_start_time)
{
    FILE* fd = NULL;

    if (dlp_env_is_logger_enabled() == TRUE) {
        char log_file[255] = { 0 };
        sprintf(log_file, "%s_P%lu_T%lu%s", AOCL_LPGEMM_LOG_FILE_PRFX,
                lpgemm_getpid(), lpgemm_gettid(), AOCL_LPGEMM_LOG_FILE_EXT);

        fd = fopen(log_file, "a");

        (*aocl_lpgemm_logger_start_time) = dlp_clock();
    }

    return fd;
}

void
lpgemm_stop_logger_fn(FILE* fd, double* aocl_lpgemm_logger_start_time)
{
    if ((dlp_env_is_logger_enabled() == TRUE) && (fd != NULL)) {
        double aocl_lpgemm_logger_stop_time = DBL_MAX;
        aocl_lpgemm_logger_stop_time        = dlp_clock_min_diff(
            aocl_lpgemm_logger_stop_time, *aocl_lpgemm_logger_start_time);
        fprintf(fd, "time:%f \n", aocl_lpgemm_logger_stop_time);
        fflush(fd);
        fclose(fd);
    }
}

#define LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, p_str)                  \
    do {                                                                       \
        char*  c_ops_str     = p_str;                                          \
        size_t c_ops_str_len = strlen(c_ops_str);                              \
        strcpy(ops_str + ops_str_len, c_ops_str);                              \
        ops_str_len += c_ops_str_len;                                          \
    } while (0);

void
lpgemm_get_pre_ops_str(dlp_metadata_t* metadata, char* ops_str)
{
    if (metadata == NULL) {
        strcpy(ops_str, "none");
        return;
    }

    dlp_pre_op* pre_ops = metadata->pre_ops;
    if ((pre_ops == NULL) || (pre_ops->seq_length <= 0)) {
        strcpy(ops_str, "none");
        return;
    }
    if ((pre_ops->seq_length > AOCL_MAX_POST_OPS)) {
        strcpy(ops_str, "ops over-limit");
        return;
    }

    size_t ops_str_len   = 0;
    char*  delim_str     = "#";
    size_t delim_str_len = strlen(delim_str);

    LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "group_sz=");
    int written = sprintf((ops_str + ops_str_len), "%ld", pre_ops->group_size);
    if (written > 0) {
        ops_str += written;
    }
    strcpy(ops_str + ops_str_len, delim_str);
    ops_str_len += delim_str_len;

    for (md_t i = 0; i < pre_ops->seq_length; ++i) {
        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "scale=");
        if ((pre_ops->b_scl) != NULL) {
            if ((pre_ops->b_scl + i)->scale_factor_len == 1) {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                         "scalar_scale_factor,");
            } else {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                         "vector_scale_factor,");
            }
        }

        if ((pre_ops->b_zp) != NULL) {
            if ((pre_ops->b_zp + i)->zero_point_len == 1) {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                         "scalar_zero_point,");
            } else {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                         "vector_zero_point,");
            }
        }

        strcpy(ops_str + ops_str_len, delim_str);
        ops_str_len += delim_str_len;
    }
}

void
lpgemm_get_post_ops_str(dlp_metadata_t* metadata, char* ops_str)
{
    if ((metadata == NULL) || (metadata->seq_length <= 0)) {
        strcpy(ops_str, "none");
        return;
    }
    if ((metadata->seq_length > AOCL_MAX_POST_OPS)) {
        strcpy(ops_str, "ops over-limit");
        return;
    }

    size_t ops_str_len   = 0;
    md_t   e_i           = 0; // Multiple eltwise supported.
    md_t   s_i           = 0; // Multiple sum/scale supported.
    char*  delim_str     = "#";
    size_t delim_str_len = strlen(delim_str);
    for (md_t i = 0; i < metadata->seq_length; ++i) {
        // Dispatcher code
        switch (*(metadata->seq_vector + i)) {
            case ELTWISE: {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "eltwise=");
                // Eltwise algo dispatcher.
                switch ((metadata->eltwise + e_i)->algo.algo_type) {
                    case RELU: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "relu");
                    } break;
                    case PRELU: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "prelu");
                    } break;
                    case GELU_TANH: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                 "gelu_tanh");
                    } break;
                    case GELU_ERF: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                 "gelu_erf");
                    } break;
                    case CLIP: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "clip");
                    } break;
                    case SWISH: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "swish");
                    } break;
                    case TANH: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "tanh");
                    } break;
                    case SIGMOID: {
                        LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                 "sigmoid");
                    } break;
                    default:
                        break;
                }
                e_i += 1;
            } break;
            case BIAS: {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "bias");
            } break;
            case SCALE: {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "scale=");
                if ((metadata->scale + s_i)->sf
                    && (metadata->scale + s_i)->sf->scale_factor_len == 1) {
                    LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                             "scalar_scale_factor,");
                } else {
                    LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                             "vector_scale_factor,");
                }

                if ((metadata->scale + s_i)->zp
                    && (metadata->scale + s_i)->zp->zero_point_len == 1) {
                    LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                             "scalar_zero_point,");
                } else {
                    LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                             "vector_zero_point,");
                }

                s_i += 1;
            } break;
            case MATRIX_ADD: {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "mat_add");
            } break;
            case MATRIX_MUL: {
                LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "mat_mul");
            } break;
            default:
                break;
        }

        strcpy(ops_str + ops_str_len, delim_str);
        ops_str_len += delim_str_len;
    }
}

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
                            dlp_metadata_t* metadata)
{
    if ((dlp_env_is_logger_enabled() == TRUE) && (fd != NULL)) {
        char pre_ops_str[1024] = { 0 };
        lpgemm_get_pre_ops_str(metadata, pre_ops_str);

        char post_ops_str[2048] = { 0 };
        lpgemm_get_post_ops_str(metadata, post_ops_str);

        fprintf(fd,
                "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "
                "%s:pre_ops=[%s]:metadata=[%s] %f %f ",
                order, transa, transb, mem_format_a, mem_format_b, m, n, k, lda,
                ldb, ldc, op_type, pre_ops_str, post_ops_str, alpha, beta);
    }
}

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
                                  dlp_metadata_t** metadata)
{
    if ((dlp_env_is_logger_enabled() == TRUE) && (fd != NULL)) {
        char pre_ops_str[1024] = { 0 };

        char post_ops_str[2048] = { 0 };

        fprintf(fd, "%s:group_count=%ld\n", op_type, group_count);
        for (md_t i = 0; i < group_count; i++) {
            lpgemm_get_pre_ops_str(metadata[i], pre_ops_str);
            lpgemm_get_post_ops_str(metadata[i], post_ops_str);
            fprintf(fd,
                    "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "
                    ":pre_ops=[%s]:metadata=[%s] %f %f %ld\n",
                    order[i], transa[i], transb[i], mem_format_a[i],
                    mem_format_b[i], m[i], n[i], k[i], lda[i], ldb[i], ldc[i],
                    pre_ops_str, post_ops_str, (float)(alpha[i]),
                    (float)(beta[i]), group_size[i]);
        }
    }
}

#else

#endif
