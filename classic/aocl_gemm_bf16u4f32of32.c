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

/*
 * GEMM for asymmetric weight-only quantization (WOQ): C = alpha * A * B + beta
 * * C
 * - A: bfloat16, B: unsigned 4-bit (u4) quantized, C: float.
 * - B must be provided in reordered format.
 * - U4 to BF16 dequantization is done via pre-ops in the framework before the
 *   micro-kernel.
 * - Both scale factor and zero point are required for asymmetric WOQ.
 * - Two output variants: aocl_gemm_bf16u4f32of32 (float C),
 *   aocl_gemm_bf16u4f32obf16 (bfloat16 C).
 *
 */

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "runtime/dlp_runtime.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

static void
aocl_gemm_bf16u4f32_impl(const char        order,
                         const char        transa,
                         const char        transb,
                         const md_t        m,
                         const md_t        n,
                         const md_t        k,
                         const float       alpha,
                         const bfloat16*   a,
                         const md_t        lda,
                         const char        mem_format_a,
                         const uint8_t*    b,
                         const md_t        ldb,
                         const char        mem_format_b,
                         const float       beta,
                         void*             c,
                         const md_t        ldc,
                         dlp_metadata_t*   metadata,
                         const char*       func_name,
                         kernel_datatype_t krnl_dtype,
                         DLP_TYPE          c_dtype)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER(func_name, order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("bf16u4f32of32", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    // Add early returns for NULL or invalid pointers.
    if (metadata == NULL || metadata->pre_ops == NULL
        || metadata->pre_ops->b_scl == NULL
        || metadata->pre_ops->b_zp == NULL) {
        dlp_print_msg("One or more required parameters (metadata, pre_ops, "
                      "pre_ops->b_zp, "
                      "pre_ops->b_scl) are NULL or invalid. Exiting..",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NULL_POINTER);
        goto err_hndl;
    }

    // Check if zero-point type is supported.
    if (metadata->pre_ops->b_zp->zero_point_type != DLP_S8
        && metadata->pre_ops->b_zp->zero_point_type != DLP_BF16) {
        dlp_print_msg(" zero-point type is not supported. Exiting..", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
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
            " column major not supported yet in bf16u4f32o<f32/bf16>.",
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

    AOCL_DLP_MEMORY_TAG mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Reorder is not supported for A matrix
    if (mtag_a == REORDERED) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if (dlp_is_trans(dlp_transa) == TRUE) {
        mtag_a = PACK;
    }

    if (mtag_b != REORDERED) {
        dlp_print_msg(" Reordering of B matrix is mandatory.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Convert pre op struct to pre op linked list format.
    dlp_gemm_pre_op pre_op_list[AOCL_DLP_MAX_PRE_OPS];
    dlp_clsc_err_t  err = dlp_gemm_translate_to_pre_ops_list(
        metadata->pre_ops, pre_op_list, m, n, k);
    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to translate pre ops list. Invalid pre ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Convert post op struct to post op linked list format.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    err = dlp_gemm_translate_to_post_ops_list(metadata, post_op_list, (void*)c,
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

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16U4F32OF32);
    dlp_gemm_cntx_t  lcntx_l = *lcntx_g;

    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
    // Use BF16BF16F32OF32 kernel : U4->BF16 dequantization is done
    // in the framework (dlp_gemm_bf16u4.c) via pre-ops before the micro-kernel.
    dlp_init_and_get_kernel_hndl(
        krnl_dtype, order, mtag_a, mtag_b, m, n, k, rs_a, cs_a, rs_b, cs_b,
        rs_c, cs_c, (void*)&alpha, (void*)&beta, post_op_list, lcntx_l.blksz.MR,
        lcntx_l.blksz.NR, lcntx_l.blksz.KC, c_dtype, &lcntx_l.dlp_kernel_hndl);

    // Defense-in-depth: if JIT init failed AND the post-op list contains an
    // op_code unsupported by the classic kernel, the 5-loop's classic
    // fallback would crash on goto *post_ops_labels[op_code]. Reject cleanly
    // in that combination; classic-supported post-ops can still proceed via
    // the existing fallback.
    if ((lcntx_l.dlp_kernel_hndl.kernel_base == NULL)
        && (dlp_gemm_post_op_list_has_jit_only_op(post_op_list) == true)) {
        dlp_print_msg(" Requested post-op is not supported in the "
                      "classic kernel.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops =
        DLP_GEMM_OPS_BUNDLE_INIT_MP(pre_op_list, post_op_list);

#ifdef DLP_ENABLE_OPENMP

    dlp_gemm_bf16u4f32of32_openmp_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c, cs_c,
        alpha, beta, &rntm_g, &lcntx_l, &ops, c_dtype);
#else
    dlp_gemm_bf16u4f32of32_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c, cs_c,
        alpha, beta, &rntm_g, &lcntx_l, &ops, c_dtype);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}

// =========================================================================
// API Implementations for bf16u4f32 variants
// =========================================================================

void
aocl_gemm_bf16u4f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const uint8_t*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    aocl_gemm_bf16u4f32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                             mem_format_a, b, ldb, mem_format_b, beta, c, ldc,
                             metadata, "bf16u4f32of32",
                             DLP_KERNEL_BF16BF16F32OF32, DLP_F32);
}

void
aocl_gemm_bf16u4f32obf16(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         const float     alpha,
                         const bfloat16* a,
                         const md_t      lda,
                         const char      mem_format_a,
                         const uint8_t*  b,
                         const md_t      ldb,
                         const char      mem_format_b,
                         const float     beta,
                         bfloat16*       c,
                         const md_t      ldc,
                         dlp_metadata_t* metadata)
{
    aocl_gemm_bf16u4f32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                             mem_format_a, b, ldb, mem_format_b, beta, c, ldc,
                             metadata, "bf16u4f32obf16",
                             DLP_KERNEL_BF16BF16F32OBF16, DLP_BF16);
}
