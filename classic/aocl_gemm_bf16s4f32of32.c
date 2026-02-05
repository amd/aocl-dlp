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
#include "classic/aocl_gemm_interface_apis.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "logging/lpgemm_logger.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_ops_bundle.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "runtime/dlp_runtime.h"
#include "threading/lpgemm_thread_decor_openmp.h"

void
aocl_gemm_bf16s4f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const int8_t*   b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    LPGEMM_START_LOGGER();
    LPGEMM_WRITE_LOGGER("bf16s4f32of32", order, transa, transb, m, n, k,
                        ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                        ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
#ifdef LPGEMM_BF16_JIT
    dlp_print_msg(" WOQ is not supported by JIT kernels.", __FILE__, __LINE__);
    goto err_hndl;
#endif

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_GEMM_CHECK("bf16s4f32of32", order, transa, transb, m, n, k, a, lda,
                    mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    if (is_column_major == TRUE) {
        // Swapping inputs not possible in case of mixed precision.
        dlp_print_msg(
            " column major not supported yet in bf16s4f32o<f32/bf16>.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // The strides are set assuming a row major kernel.
    md_t rs_a = lda;
    md_t cs_a = 1;

    if (dlp_is_trans(dlp_transa) == TRUE) {
        rs_a = 1;
        cs_a = lda;
    }
    md_t rs_b = ldb;
    md_t cs_b = 1;

    if (dlp_is_trans(dlp_transb) == TRUE) {
        rs_b = 1;
        cs_b = ldb;
    }
    const md_t rs_c = ldc;
    const md_t cs_c = 1;

    AOCL_MEMORY_TAG mtag_a;
    AOCL_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Reorder is not supported for A matrix
    if (mtag_a == REORDERED) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if (mtag_b != REORDERED) {
        dlp_print_msg(" Reordering of B matrix is mandatory.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if (dlp_is_trans(dlp_transa) == TRUE) {
        mtag_a = PACK;
    }

    // Convert pre op struct to pre op linked list format.
    lpgemm_pre_op  pre_op_list[AOCL_MAX_PRE_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_pre_ops_list(metadata->pre_ops,
                                                          pre_op_list, m, n, k);
    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to translate pre ops list. Invalid pre ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    err = lpgemm_translate_to_post_ops_list(metadata, post_op_list, (void*)c,
                                            (void*)(&order), m, n);
    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to translate post ops list. Invalid post ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(BF16S4F32OF32);

    lpgemm_ops_bundle_t ops =
        LPGEMM_OPS_BUNDLE_INIT_MP(pre_op_list, post_op_list);

#ifdef DLP_ENABLE_OPENMP

    lpgemm_bf16s4f32of32_openmp_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c, cs_c,
        alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
#else
    lpgemm_bf16s4f32of32_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c, cs_c,
        alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}

void
aocl_gemm_bf16s4f32obf16(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         const float     alpha,
                         const bfloat16* a,
                         const md_t      lda,
                         const char      mem_format_a,
                         const int8_t*   b,
                         const md_t      ldb,
                         const char      mem_format_b,
                         const float     beta,
                         bfloat16*       c,
                         const md_t      ldc,
                         dlp_metadata_t* metadata)
{
    LPGEMM_START_LOGGER();
    LPGEMM_WRITE_LOGGER("bf16s4f32obf16", order, transa, transb, m, n, k,
                        ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                        ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
#ifdef LPGEMM_BF16_JIT
    dlp_print_msg(" WOQ is not supported by JIT kernels.", __FILE__, __LINE__);
    goto err_hndl;
#endif

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_GEMM_CHECK("bf16s4f32obf16", order, transa, transb, m, n, k, a, lda,
                    mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    if (is_column_major == TRUE) {
        // Swapping inputs not possible in case of mixed precision.
        dlp_print_msg(
            " column major not supported yet in bf16s4f32o<f32/bf16>.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // The strides are set assuming a row major kernel.
    md_t rs_a = lda;
    md_t cs_a = 1;

    if (dlp_is_trans(dlp_transa) == TRUE) {
        rs_a = 1;
        cs_a = lda;
    }

    md_t rs_b = ldb;
    md_t cs_b = 1;

    if (dlp_is_trans(dlp_transb) == TRUE) {
        rs_b = 1;
        cs_b = ldb;
    }
    const md_t rs_c = ldc;
    const md_t cs_c = 1;

    AOCL_MEMORY_TAG mtag_a;
    AOCL_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Reorder is not supported for A matrix
    if (mtag_a == REORDERED) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if (mtag_b != REORDERED) {
        dlp_print_msg(" Reordering of B matrix is mandatory.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if (dlp_is_trans(dlp_transa) == TRUE) {
        mtag_a = PACK;
    }

    // Convert post op struct to post op linked list format.
    lpgemm_pre_op  pre_op_list[AOCL_MAX_PRE_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_pre_ops_list(metadata->pre_ops,
                                                          pre_op_list, m, n, k);

    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to translate pre ops list. Invalid pre ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    err = lpgemm_translate_to_post_ops_list(metadata, post_op_list, (void*)c,
                                            (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to translate post ops list. Invalid post ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(BF16S4F32OF32);

    lpgemm_ops_bundle_t ops =
        LPGEMM_OPS_BUNDLE_INIT_MP(pre_op_list, post_op_list);

#ifdef DLP_ENABLE_OPENMP
    lpgemm_bf16s4f32of32_openmp_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c, rs_c,
        cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_BF16);
#else
    lpgemm_bf16s4f32of32_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c, rs_c,
        cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_BF16);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}
