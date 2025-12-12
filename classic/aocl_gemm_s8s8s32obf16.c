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

void
aocl_gemm_s8s8s32obf16(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const int8_t*   a,
                       const md_t      lda,
                       const char      mem_format_a,
                       const int8_t*   b,
                       const md_t      ldb,
                       const char      mem_format_b,
                       const int32_t   beta,
                       bfloat16*       c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata)
{
    LPGEMM_START_LOGGER();
    LPGEMM_WRITE_LOGGER("s8s8s32obf16", order, transa, transb, m, n, k,
                        ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                        ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32obf16 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

#ifdef LPGEMM_BF16_JIT
    dlp_print_msg("cannot perform s8s8s32obf16 gemm with gcc < 11.2", __FILE__,
                  __LINE__);
    goto err_hndl;
#endif

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_GEMM_CHECK("s8s8s32obf16", order, transa, transb, m, n, k, a, lda,
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

    // Column major support disabled for int API's till micro-kernel
    // post-ops are updated to account for column major.
    if ((is_column_major == TRUE) && (metadata != NULL)
        && (metadata->seq_length > 0)) {
        dlp_print_msg("Column major inputs not supported with Post-ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // The strides are set assuming a row major kernel.
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
        dlp_print_msg(" Reordering of A matrix is not supported in "
                      " row major case.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // Reorder is not supported for column major matrices.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is "
                      " not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // A matrix packing is only done in column major case, or when
    // A matrix is transposed in row major. PackA kernels for row-maj
    // is not supported, hence we set it to unpacked and proceed with GEMM.
    if ((is_row_major == TRUE) && (mtag_a == PACK)) {
        mtag_a = UNPACKED;
    } else if (is_column_major == TRUE && mtag_b == PACK) {
        mtag_b = UNPACKED;
    }
    // From 5-loop function point of view
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

    // Temporary variables to transform the inputs for col-major cases.
    md_t            m_use      = m;
    md_t            n_use      = n;
    md_t            k_use      = k;
    const int8_t*   a_use      = a;
    const int8_t*   b_use      = b;
    bfloat16*       c_use      = c;
    md_t            rs_a_use   = rs_a;
    md_t            cs_a_use   = cs_a;
    md_t            rs_b_use   = rs_b;
    md_t            cs_b_use   = cs_b;
    md_t            rs_c_use   = rs_c;
    md_t            cs_c_use   = cs_c;
    AOCL_MEMORY_TAG mtag_a_use = mtag_a;
    AOCL_MEMORY_TAG mtag_b_use = mtag_b;

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    if (is_column_major) {
        // Swap A and B for column major case.
        m_use      = n;
        n_use      = m;
        a_use      = b;
        b_use      = a;
        rs_a_use   = rs_b;
        cs_a_use   = cs_b;
        rs_b_use   = rs_a;
        cs_b_use   = cs_a;
        mtag_a_use = mtag_b;
        mtag_b_use = mtag_a;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(S8S8S32OS32);
    lpgemm_cntx_t  lcntx_l;
    // Create local copy, since each thread in a multi-instance setup
    // modifies the context object.
    lcntx_l = *lcntx_g;

    // Initialize DLP Plus kernel path.
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_S8S8S32OBF16, order, mtag_a_use, mtag_b_use, m_use, n_use,
        k_use, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c_use, cs_c_use,
        (void*)&alpha, (void*)&beta, post_op_list, lcntx_l.blksz.MR,
        lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_BF16, &lcntx_l.dlp_kernel_hndl);

#ifdef DLP_ENABLE_OPENMP
    lpgemm_s8s8s32o32_openmp_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, (int32_t*)c, rs_c_use, cs_c_use, alpha,
        beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
#else
    lpgemm_s8s8s32o32_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, (int32_t*)c, rs_c_use, cs_c_use, alpha,
        beta, &rntm_g, &lcntx_l, post_op_list, DLP_BF16);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}
