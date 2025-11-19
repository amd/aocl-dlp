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
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "runtime/dlp_runtime.h"
#include "threading/lpgemm_thread_decor_openmp.h"

static inline bool
is_tiny_input_bf16obf16(md_t m, md_t n, md_t k, lpgemm_cntx_t* lcntx)
{
    const md_t NC = lcntx->blksz.NC;
    const md_t MC = lcntx->blksz.MC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MR = lcntx->blksz.MR;
    const md_t NR = lcntx->blksz.NR;

    md_t       mnk           = m * n * k;
    const md_t mnk_magic_num = 36 * 128 * 256;
    const md_t m_thresh      = 6 * MR;
    const md_t n_thresh      = 6 * NR;
    const md_t k_thresh      = 1024;

    // Need to explicitly check for MC, NC boundaries for safety.
    if ((m <= MC) && (n < NC) && (k < KC)
        && ((m <= m_thresh) && (n <= n_thresh) && (k <= k_thresh)
            && (mnk < mnk_magic_num))) {
        return TRUE;
    }

    return FALSE;
}

void
aocl_gemm_bf16bf16f32obf16(const char      order,
                           const char      transa,
                           const char      transb,
                           const md_t      m,
                           const md_t      n,
                           const md_t      k,
                           const float     alpha,
                           const bfloat16* a,
                           const md_t      lda,
                           const char      mem_format_a,
                           const bfloat16* b,
                           const md_t      ldb,
                           const char      mem_format_b,
                           const float     beta,
                           bfloat16*       c,
                           const md_t      ldc,
                           dlp_metadata_t* metadata)
{
    LPGEMM_START_LOGGER();
    LPGEMM_WRITE_LOGGER("bf16bf16f32obf16", order, transa, transb, m, n, k,
                        ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                        ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_GEMM_CHECK("bf16bf16f32obf16", order, transa, transb, m, n, k, a, lda,
                    mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

#ifdef LPGEMM_BF16_JIT
    if (get_jit_kernels_generated() == FALSE) {
        dlp_print_msg(" Could not generate bf16bf16f32obf16 "
                      " kernels using JIT.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
#endif

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    md_t rs_a = lda;
    md_t cs_a = 1;

    if (dlp_is_trans(dlp_transa)) {
        rs_a = 1;
        cs_a = lda;
    }

    md_t rs_b = ldb;
    md_t cs_b = 1;

    if (dlp_is_trans(dlp_transb)) {
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
    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(
            " Reordering of A matrix is not supported in row major case.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // Reorder is not supported for column major matrices.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // From 5-loop function point of view,
    // B matrix needs to be packed in a certain format in order to be loaded
    // and used in bf16 instrution. As such the mtag_b always needs to be either
    // packed or reordered. B matrix as it is (unpacked) cannot be used, and
    // the mtag_b is set to packed to enable runtime packing.
    if ((is_row_major == TRUE) && (mtag_b == UNPACKED)) {
        mtag_b = PACK;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    else if ((is_column_major == TRUE) && (mtag_a == UNPACKED)) {
        mtag_a = PACK;
    }

    // From 5-loop function point of view,
    // A matrix when in column major storage needs to be packed to row-major
    // storage as kernel expects A matrix to be in row-major format.
    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transa))) {
        mtag_a = PACK;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    else if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transb))) {
        mtag_b = PACK;
    }

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(BF16BF16F32OF32);
    lpgemm_cntx_t  lcntx_l;
    // Create local copy, since each thread in a multi-instance setup
    // modified the context object.
    lcntx_l = *lcntx_g;

    // Initialize DLP Plus kernel path.
    // For machines without AVX512BF16, BF16 operations use F32 JIT kernels.
    // Use F32 block sizes (MR=6, NR=16 for AVX2) instead of BF16 block sizes
    // (MR=6, NR=64 for AVX512BF16) to ensure correct kernel generation.
    md_t mr_hint = lcntx_l.blksz.MR;
    md_t nr_hint = lcntx_l.blksz.NR;
    md_t kc_hint = lcntx_l.blksz.KC;

    AOCL_MEMORY_TAG jit_mtag_b = mtag_b;

    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        // No native BF16 support - will use F32 kernels
        // Get F32 context for proper block sizes
        lpgemm_cntx_t* lcntx_f32 = lpgemm_get_global_cntx_obj(F32F32F32OF32);
        mr_hint                  = lcntx_f32->blksz.MR;
        nr_hint                  = lcntx_f32->blksz.NR;
        kc_hint                  = lcntx_f32->blksz.KC;

        // For m=1 case B matrix is unpacked inside the framework before
        // calling f32 kernel, same should be provided for generating JIT
        // kernels
        if (m == 1) {
            jit_mtag_b = UNPACKED;
        }
    }

    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_BF16BF16F32OBF16, order, mtag_a, jit_mtag_b, m, n, k, rs_a,
        cs_a, rs_b, cs_b, rs_c, cs_c, (void*)&alpha, (void*)&beta, post_op_list,
        mr_hint, nr_hint, kc_hint, DLP_BF16, &lcntx_l.dlp_kernel_hndl);

#if (defined(DLP_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
    /* While AOCL_ENABLE_INSTRUCTIONS=AVX2 is enabled in machines that supports
     * DLP_BF16/VNNI with only the ISA check the exeution could enter tiny path
     * and result in seg fault as the tiny path for DLP_BF16->FP32 is not
     * available. Hence the arch_id also has to be verified here.
     */
    dlp_arch_t arch_id = dlp_get_arch();
    if (((arch_id == DLP_ARCH_ZEN4) || (arch_id == DLP_ARCH_ZEN5))
        && (dlp_cpuid_is_avx512bf16_supported() == TRUE)
        && (is_single_thread(&rntm_g) == TRUE)) {
        if ((is_row_major == TRUE)
            && (is_tiny_input_bf16obf16(m, n, k, &lcntx_l) == TRUE)) {
            lpgemm_rowvar_tiny_bf16bf16f32of32(
                m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b,
                (float*)c, rs_c, cs_c, alpha, beta, &lcntx_l, post_op_list,
                DLP_BF16);
            goto err_hndl;
        } else if ((is_column_major == TRUE)
                   && (is_tiny_input_bf16obf16(n, m, k, &lcntx_l) == TRUE)) {
            lpgemm_rowvar_tiny_bf16bf16f32of32(
                n, m, k, b, rs_b, cs_b, mtag_b, a, rs_a, cs_a, mtag_a,
                (float*)c, rs_c, cs_c, alpha, beta, &lcntx_l, post_op_list,
                DLP_BF16);
            goto err_hndl;
        }
    }
#endif
#ifdef DLP_ENABLE_OPENMP
    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        lpgemm_bf16bf16f32of32_openmp_thread_decorator(
            n, m, k, b, rs_b, cs_b, mtag_b, a, rs_a, cs_a, mtag_a, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
    } else {
        lpgemm_bf16bf16f32of32_openmp_thread_decorator(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
    }
#else
    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        lpgemm_bf16bf16f32of32_thread_decorator(
            n, m, k, b, rs_b, cs_b, mtag_b, a, rs_a, cs_a, mtag_a, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
    } else {
        lpgemm_bf16bf16f32of32_thread_decorator(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
    }
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}
