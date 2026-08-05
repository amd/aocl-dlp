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

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>

#include "bindings/c_wrappers/capi_env_config.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "logging/dlp_gemm_logger.h"
#include "sys_utils/dlp_gemm_sys.h"

#ifdef AOCL_DLP_LOGGER_SUPPORT

bool
dlp_gemm_is_logger_enabled()
{
    return dlp_env_is_logger_enabled();
}

FILE*
dlp_gemm_start_logger_fn(double* dlp_gemm_logger_start_time)
{
    FILE* fd = NULL;

    if (dlp_env_is_logger_enabled() == TRUE) {
        char log_file[255] = { 0 };
        snprintf(log_file, sizeof(log_file), "%s_P%lu_T%lu%s",
                 AOCL_DLP_GEMM_LOG_FILE_PRFX, dlp_gemm_getpid(),
                 dlp_gemm_gettid(), AOCL_DLP_GEMM_LOG_FILE_EXT);

        fd = fopen(log_file, "a");

        (*dlp_gemm_logger_start_time) = dlp_clock();
    }

    return fd;
}

void
dlp_gemm_stop_logger_fn(FILE* fd, double* dlp_gemm_logger_start_time)
{
    if ((dlp_env_is_logger_enabled() == TRUE) && (fd != NULL)) {
        double dlp_gemm_logger_stop_time = DBL_MAX;
        dlp_gemm_logger_stop_time        = dlp_clock_min_diff(
            dlp_gemm_logger_stop_time, *dlp_gemm_logger_start_time);
        fprintf(fd, "time:%f \n", dlp_gemm_logger_stop_time);
        fflush(fd);
        fclose(fd);
    }
}

static void
dlp_gemm_logger_vstr_append(char*       ops_str,
                            size_t*     ops_str_len,
                            size_t      ops_str_max_len,
                            const char* format,
                            va_list     args)
{
    if ((ops_str == NULL) || (ops_str_len == NULL) || (format == NULL)
        || (ops_str_max_len == 0) || (*ops_str_len >= (ops_str_max_len - 1))) {
        return;
    }

    size_t remaining = ops_str_max_len - *ops_str_len;
    int    written = vsnprintf(ops_str + *ops_str_len, remaining, format, args);

    if (written < 0) {
        return;
    }

    if ((size_t)written >= remaining) {
        *ops_str_len = ops_str_max_len - 1;
    } else {
        *ops_str_len += (size_t)written;
    }
}

static void
dlp_gemm_logger_str_append(char*       ops_str,
                           size_t*     ops_str_len,
                           size_t      ops_str_max_len,
                           const char* format,
                           ...)
{
    va_list args;
    va_start(args, format);
    dlp_gemm_logger_vstr_append(ops_str, ops_str_len, ops_str_max_len, format,
                                args);
    va_end(args);
}

#define DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, p_str)                \
    dlp_gemm_logger_str_append((ops_str), &(ops_str_len),                      \
                               DLP_GEMM_POST_OPS_STR_MAX_LEN, "%s", (p_str))

static void
dlp_gemm_quant_ops_str_append(char*       ops_str,
                              size_t*     ops_str_len,
                              const char* format,
                              ...)
{
    va_list args;
    va_start(args, format);
    dlp_gemm_logger_vstr_append(ops_str, ops_str_len,
                                DLP_GEMM_QUANT_OPS_STR_MAX_LEN, format, args);
    va_end(args);
}

static void
dlp_gemm_append_qparam_str(char*               ops_str,
                           size_t*             ops_str_len,
                           const char*         name,
                           const dlp_qparam_t* qparam)
{
    dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, "%s=", name);
    if (qparam == NULL) {
        dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, "none");
    } else if (qparam->len == 1) {
        dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, "scalar");
    } else {
        dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, "vector");
    }
}

static void
dlp_gemm_append_quant_op_str(char*                 ops_str,
                             size_t*               ops_str_len,
                             const char*           name,
                             const dlp_quant_op_t* quant_op)
{
    if (quant_op == NULL) {
        return;
    }

    dlp_gemm_quant_ops_str_append(
        ops_str, ops_str_len, "%s={kind=%d,src=%d,dst=%d,group_sz=%" PRId64 ",",
        name, (int)quant_op->quant_op_kind, (int)quant_op->src_type,
        (int)quant_op->dst_type, (int64_t)quant_op->group_size);
    dlp_gemm_append_qparam_str(ops_str, ops_str_len, "quant",
                               quant_op->quant_scale_factors);
    dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, ",");
    dlp_gemm_append_qparam_str(ops_str, ops_str_len, "dequant",
                               quant_op->dequant_scale_factors);
    dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, ",");
    dlp_gemm_append_qparam_str(ops_str, ops_str_len, "zero_point",
                               quant_op->zero_point);
    dlp_gemm_quant_ops_str_append(ops_str, ops_str_len, "}");
}

void
dlp_gemm_get_quant_ops_str(dlp_metadata_t* metadata, char* ops_str)
{
    if (metadata == NULL) {
        size_t ops_str_len = 0;
        dlp_gemm_quant_ops_str_append(ops_str, &ops_str_len, "none");
        return;
    }

    if ((metadata->a_quant_op == NULL) && (metadata->b_quant_op == NULL)) {
        size_t ops_str_len = 0;
        dlp_gemm_quant_ops_str_append(ops_str, &ops_str_len, "none");
        return;
    }

    size_t ops_str_len = 0;
    char*  delim_str   = "#";

    if (metadata->a_quant_op != NULL) {
        dlp_gemm_append_quant_op_str(ops_str, &ops_str_len, "a_quant",
                                     metadata->a_quant_op);
        dlp_gemm_quant_ops_str_append(ops_str, &ops_str_len, "%s", delim_str);
    }

    if (metadata->b_quant_op != NULL) {
        dlp_gemm_append_quant_op_str(ops_str, &ops_str_len, "b_quant",
                                     metadata->b_quant_op);
        dlp_gemm_quant_ops_str_append(ops_str, &ops_str_len, "%s", delim_str);
    }
}

void
dlp_gemm_get_post_ops_str(dlp_metadata_t* metadata, char* ops_str)
{
    if ((metadata == NULL) || (metadata->seq_length <= 0)) {
        size_t ops_str_len = 0;
        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "none");
        return;
    }
    if ((metadata->seq_length > AOCL_DLP_MAX_POST_OPS)) {
        size_t ops_str_len = 0;
        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "ops over-limit");
        return;
    }

    size_t ops_str_len = 0;
    iter_t e_i         = 0; // Multiple eltwise supported.
    iter_t s_i         = 0; // Multiple sum/scale supported.
    char*  delim_str   = "#";
    for (iter_t i = 0; i < metadata->seq_length; ++i) {
        // Dispatcher code
        switch (*(metadata->seq_vector + i)) {
            case ELTWISE: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "eltwise=");
                // Eltwise algo dispatcher.
                switch ((metadata->eltwise + e_i)->algo.algo_type) {
                    case RELU: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "relu");
                    } break;
                    case PRELU: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "prelu");
                    } break;
                    case GELU_TANH: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "gelu_tanh");
                    } break;
                    case GELU_ERF: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "gelu_erf");
                    } break;
                    case CLIP: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "clip");
                    } break;
                    case SWISH: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "swish");
                    } break;
                    case TANH: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "tanh");
                    } break;
                    case SIGMOID: {
                        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                                   "sigmoid");
                    } break;
                    default:
                        break;
                }
                e_i += 1;
            } break;
            case BIAS: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "bias");
            } break;
            case SCALE: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "scale=");
                if ((metadata->scale + s_i)->sf
                    && (metadata->scale + s_i)->sf->scale_factor_len == 1) {
                    DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                               "scalar_scale_factor,");
                } else {
                    DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                               "vector_scale_factor,");
                }

                if ((metadata->scale + s_i)->zp
                    && (metadata->scale + s_i)->zp->zero_point_len == 1) {
                    DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                               "scalar_zero_point,");
                } else {
                    DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                               "vector_zero_point,");
                }

                s_i += 1;
            } break;
            case MATRIX_ADD: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "mat_add");
            } break;
            case MATRIX_MUL: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "mat_mul");
            } break;
            default:
                break;
        }

        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, delim_str);
    }

    // GLU is a terminal op and at most one per op chain.
    if (metadata->glu != NULL) {
        DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, "glu=");
        // GLU algo dispatcher.
        switch (metadata->glu->algo_type) {
            case GATED_SWIGLU: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                           "gated_swiglu");
            } break;
            case GATED_SWIGLU_AND_MUL: {
                DLP_GEMM_POST_OPS_STR_COPY(ops_str, ops_str_len,
                                           "gated_swiglu_and_mul");
            } break;
            default:
                break;
        }
    }
}

void
dlp_gemm_get_tuning_str(dlp_metadata_t* metadata, char* tuning_str)
{
    if (metadata == NULL) {
        size_t tuning_str_len = 0;
        dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                                   DLP_GEMM_GEN_STR_MAX_LEN, "none");
        return;
    }

    size_t tuning_str_len = 0;
    char*  delim_str      = "#";

    if (metadata->block_params != NULL) {
        dlp_gemm_logger_str_append(
            tuning_str, &tuning_str_len, DLP_GEMM_GEN_STR_MAX_LEN,
            "block_params={MR=%" PRId64 ",NR=%" PRId64 ",MC=%" PRId64
            ",NC=%" PRId64 ",KC=%" PRId64 "}",
            (metadata->block_params)->MR, (metadata->block_params)->NR,
            (metadata->block_params)->MC, (metadata->block_params)->NC,
            (metadata->block_params)->KC);
    } else {
        dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                                   DLP_GEMM_GEN_STR_MAX_LEN, "none");
    }
    dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                               DLP_GEMM_GEN_STR_MAX_LEN, delim_str);

    if (metadata->sup_thresholds != NULL) {
        dlp_gemm_logger_str_append(
            tuning_str, &tuning_str_len, DLP_GEMM_GEN_STR_MAX_LEN,
            "sup_thresholds={MT=%" PRId64 ",NT=%" PRId64 ",KT=%" PRId64 "}",
            (metadata->sup_thresholds)->MT, (metadata->sup_thresholds)->NT,
            (metadata->sup_thresholds)->KT);
    } else {
        dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                                   DLP_GEMM_GEN_STR_MAX_LEN, "none");
    }
    dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                               DLP_GEMM_GEN_STR_MAX_LEN, delim_str);

    if (metadata->gemm_hints != NULL) {
        dlp_gemm_logger_str_append(
            tuning_str, &tuning_str_len, DLP_GEMM_GEN_STR_MAX_LEN,
            "gemm_hints={m_hint=%" PRId64 ", nt_hint=%" PRId64 "}",
            (metadata->gemm_hints)->m_hint, (metadata->gemm_hints)->nt_hint);
    } else {
        dlp_gemm_logger_str_append(tuning_str, &tuning_str_len,
                                   DLP_GEMM_GEN_STR_MAX_LEN, "none");
    }
}

void
dlp_gemm_write_logger_gemm_fn(FILE*           fd,
                              const char*     op_type,
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
        char quant_ops_str[DLP_GEMM_QUANT_OPS_STR_MAX_LEN] = { 0 };
        dlp_gemm_get_quant_ops_str(metadata, quant_ops_str);

        char post_ops_str[DLP_GEMM_POST_OPS_STR_MAX_LEN] = { 0 };
        dlp_gemm_get_post_ops_str(metadata, post_ops_str);

        char tuning_str[DLP_GEMM_GEN_STR_MAX_LEN] = { 0 };
        dlp_gemm_get_tuning_str(metadata, tuning_str);

        fprintf(fd,
                "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "
                "%s:quant_ops=[%s]:metadata=[%s] %f %f tuning=[%s]\n",
                order, transa, transb, mem_format_a, mem_format_b, m, n, k, lda,
                ldb, ldc, op_type, quant_ops_str, post_ops_str, alpha, beta,
                tuning_str);
    }
}

void
batch_dlp_gemm_write_logger_gemm_fn(FILE*            fd,
                                    const char*      op_type,
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
        char quant_ops_str[DLP_GEMM_QUANT_OPS_STR_MAX_LEN] = { 0 };

        char post_ops_str[DLP_GEMM_POST_OPS_STR_MAX_LEN] = { 0 };

        char tuning_str[DLP_GEMM_GEN_STR_MAX_LEN] = { 0 };

        fprintf(fd, "%s:group_count=%ld\n", op_type, group_count);
        for (iter_t i = 0; i < group_count; i++) {
            dlp_gemm_get_quant_ops_str(metadata[i], quant_ops_str);
            dlp_gemm_get_post_ops_str(metadata[i], post_ops_str);
            dlp_gemm_get_tuning_str(metadata[i], tuning_str);
            fprintf(fd,
                    "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "
                    ":quant_ops=[%s]:metadata=[%s] %f %f %ld tuning=[%s]\n",
                    order[i], transa[i], transb[i], mem_format_a[i],
                    mem_format_b[i], m[i], n[i], k[i], lda[i], ldb[i], ldc[i],
                    quant_ops_str, post_ops_str, (float)(alpha[i]),
                    (float)(beta[i]), group_size[i], tuning_str);
        }
    }
}

#else

#endif
