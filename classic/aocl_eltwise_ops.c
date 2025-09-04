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

#include "aocl_gemm_check.h"
#include "classic/aocl_eltwise_ops_interface_apis.h"
#include "classic/dlp_errors.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "threading/lpgemm_thread_decor_openmp.h"

DLP_INLINE void
aocl_eltwise_ops_bf16of32_base(const char      order,
                               const char      transa,
                               const char      transb,
                               const md_t      m,
                               const md_t      n,
                               const bfloat16* a,
                               const md_t      lda,
                               float*          b,
                               const md_t      ldb,
                               dlp_metadata_t* metadata,
                               DLP_TYPE        c_downscale)
{
    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_column_major = ((order == 'c') || (order == 'C'));

    // Column major support disabled for int API's till micro-kernel
    // post-ops are updated to account for column major.
    if ((is_column_major == TRUE) || (dlp_is_trans(dlp_transa))
        || (dlp_is_trans(dlp_transb))) {
        dlp_print_msg("Column major and transpose not supported.", __FILE__,
                      __LINE__);
        return;
    }

    // The strides are set assuming a row major kernel.
    md_t rs_a = lda;
    md_t cs_a = 1;
    md_t rs_b = ldb;
    md_t cs_b = 1;

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
        metadata, post_op_list, NULL, (void*)(&order), m, n);
    if (err != DLP_CLSC_SUCCESS)
        return;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_eltwise_ops_cntx_t* lcntx_g =
        lpgemm_eltwise_ops_get_global_cntx_obj(BF16OF32);

#ifdef DLP_ENABLE_OPENMP

    lpgemm_eltwise_ops_bf16of32_openmp_thread_decorator(
        m, n, a, rs_a, cs_a, b, rs_b, cs_b, &rntm_g, lcntx_g, post_op_list,
        c_downscale);
#else
    lpgemm_eltwise_ops_bf16of32_thread_decorator(m, n, a, rs_a, cs_a, b, rs_b,
                                                 cs_b, &rntm_g, lcntx_g,
                                                 post_op_list, c_downscale);
#endif
}

void
aocl_gemm_eltwise_ops_bf16of32(const char      order,
                               const char      transa,
                               const char      transb,
                               const md_t      m,
                               const md_t      n,
                               const bfloat16* a,
                               const md_t      lda,
                               float*          b,
                               const md_t      ldb,
                               dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("bf16of32", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

    aocl_eltwise_ops_bf16of32_base(order, transa, transb, m, n, a, lda, b, ldb,
                                   metadata, DLP_F32);
}

void
aocl_gemm_eltwise_ops_bf16obf16(const char      order,
                                const char      transa,
                                const char      transb,
                                const md_t      m,
                                const md_t      n,
                                const bfloat16* a,
                                const md_t      lda,
                                bfloat16*       b,
                                const md_t      ldb,
                                dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("bf16obf16", order, transa, transb, m, n, a,
                                lda, b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

#ifdef LPGEMM_BF16_JIT
    dlp_print_msg("cannot perform the operation with gcc < 11.2", __FILE__,
                  __LINE__);
    return;
#endif

    // Even though b matrix is typecasted to float*, actual load/store
    // and matrix traversal will happen as bfloat16* type. This typecast
    // is only to ensure code is reused.
    aocl_eltwise_ops_bf16of32_base(order, transa, transb, m, n, a, lda,
                                   (float*)b, ldb, metadata, DLP_BF16);
}

DLP_INLINE void
aocl_eltwise_ops_f32of32_base(const char      order,
                              const char      transa,
                              const char      transb,
                              const md_t      m,
                              const md_t      n,
                              const float*    a,
                              const md_t      lda,
                              float*          b,
                              const md_t      ldb,
                              dlp_metadata_t* metadata,
                              DLP_TYPE        c_downscale)
{
    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_column_major = ((order == 'c') || (order == 'C'));

    // Column major support disabled for int API's till micro-kernel
    // post-ops are updated to account for column major.
    if ((is_column_major == TRUE) || (dlp_is_trans(dlp_transa))
        || (dlp_is_trans(dlp_transb))) {
        dlp_print_msg("Column major and transpose not supported.", __FILE__,
                      __LINE__);
        return;
    }

    // The strides are set assuming a row major kernel.
    md_t rs_a = lda;
    md_t cs_a = 1;
    md_t rs_b = ldb;
    md_t cs_b = 1;

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
        metadata, post_op_list, NULL, (void*)(&order), m, n);
    if (err != DLP_CLSC_SUCCESS)
        return;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_eltwise_ops_cntx_t* lcntx_g =
        lpgemm_eltwise_ops_get_global_cntx_obj(F32OF32);

#ifdef DLP_ENABLE_OPENMP

    lpgemm_eltwise_ops_f32of32_openmp_thread_decorator(
        m, n, a, rs_a, cs_a, b, rs_b, cs_b, &rntm_g, lcntx_g, post_op_list,
        c_downscale);
#else
    lpgemm_eltwise_ops_f32of32_thread_decorator(m, n, a, rs_a, cs_a, b, rs_b,
                                                cs_b, &rntm_g, lcntx_g,
                                                post_op_list, c_downscale);
#endif
}

void
aocl_gemm_eltwise_ops_f32of32(const char      order,
                              const char      transa,
                              const char      transb,
                              const md_t      m,
                              const md_t      n,
                              const float*    a,
                              const md_t      lda,
                              float*          b,
                              const md_t      ldb,
                              dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("f32of32", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

    aocl_eltwise_ops_f32of32_base(order, transa, transb, m, n, a, lda, b, ldb,
                                  metadata, DLP_F32);
}

void
aocl_gemm_eltwise_ops_f32obf16(const char      order,
                               const char      transa,
                               const char      transb,
                               const md_t      m,
                               const md_t      n,
                               const float*    a,
                               const md_t      lda,
                               bfloat16*       b,
                               const md_t      ldb,
                               dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("f32obf16", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

#ifdef LPGEMM_BF16_JIT
    dlp_print_msg("cannot perform the operation with gcc < 11.2", __FILE__,
                  __LINE__);
    return;
#endif

    aocl_eltwise_ops_f32of32_base(order, transa, transb, m, n, a, lda,
                                  (float*)b, ldb, metadata, DLP_BF16);
}

void
aocl_gemm_eltwise_ops_f32os32(const char      order,
                              const char      transa,
                              const char      transb,
                              const md_t      m,
                              const md_t      n,
                              const float*    a,
                              const md_t      lda,
                              int32_t*        b,
                              const md_t      ldb,
                              dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("f32os32", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

    aocl_eltwise_ops_f32of32_base(order, transa, transb, m, n, a, lda,
                                  (float*)b, ldb, metadata, DLP_S32);
}

void
aocl_gemm_eltwise_ops_f32os8(const char      order,
                             const char      transa,
                             const char      transb,
                             const md_t      m,
                             const md_t      n,
                             const float*    a,
                             const md_t      lda,
                             int8_t*         b,
                             const md_t      ldb,
                             dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("f32os8", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

    aocl_eltwise_ops_f32of32_base(order, transa, transb, m, n, a, lda,
                                  (float*)b, ldb, metadata, DLP_S8);
}

void
aocl_gemm_eltwise_ops_f32ou8(const char      order,
                             const char      transa,
                             const char      transb,
                             const md_t      m,
                             const md_t      n,
                             const float*    a,
                             const md_t      lda,
                             uint8_t*        b,
                             const md_t      ldb,
                             dlp_metadata_t* metadata)
{
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_UTIL_ELTWISE_OPS_CHECK("f32ou8", order, transa, transb, m, n, a, lda,
                                b, ldb, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return;
    }

    aocl_eltwise_ops_f32of32_base(order, transa, transb, m, n, a, lda,
                                  (float*)b, ldb, metadata, DLP_U8);
}
