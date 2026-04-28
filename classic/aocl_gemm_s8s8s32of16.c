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

void
aocl_gemm_s8s8s32of16(const char      order,
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
                      float16*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("s8s8s32of16", order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("s8s8s32of16", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    if ((is_column_major == TRUE) && (metadata != NULL)
        && (metadata->seq_length > 0)) {
        dlp_print_msg("Column major inputs not supported with Post-ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

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

    AOCL_DLP_MEMORY_TAG mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(" Reordering of A matrix is not supported in "
                      " row major case.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    } else if ((is_column_major == TRUE)
               && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is "
                      " not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if ((is_row_major == TRUE) && (mtag_a == PACK)) {
        mtag_a = UNPACKED;
    } else if (is_column_major == TRUE && mtag_b == PACK) {
        mtag_b = UNPACKED;
    }

    if ((is_row_major == TRUE) && (mtag_b == UNPACKED)) {
        mtag_b = PACK;
    } else if ((is_column_major == TRUE) && (mtag_a == UNPACKED)) {
        mtag_a = PACK;
    }

    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transa))) {
        mtag_a = PACK;
    } else if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transb))) {
        mtag_b = PACK;
    }

    md_t                m_use      = m;
    md_t                n_use      = n;
    md_t                k_use      = k;
    const int8_t*       a_use      = a;
    const int8_t*       b_use      = b;
    float16*            c_use      = c;
    md_t                rs_a_use   = rs_a;
    md_t                cs_a_use   = cs_a;
    md_t                rs_b_use   = rs_b;
    md_t                cs_b_use   = cs_b;
    md_t                rs_c_use   = rs_c;
    md_t                cs_c_use   = cs_c;
    AOCL_DLP_MEMORY_TAG mtag_a_use = mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b_use = mtag_b;

    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    if (is_column_major) {
        m_use      = n;
        n_use      = m;
        a_use      = b;
        b_use      = a;
        c_use      = c;
        rs_a_use   = rs_b;
        cs_a_use   = cs_b;
        rs_b_use   = rs_a;
        cs_b_use   = cs_a;
        mtag_a_use = mtag_b;
        mtag_b_use = mtag_a;
    }

    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);
    dlp_gemm_cntx_t  lcntx_l;
    lcntx_l = *lcntx_g;

    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_S8S8S32OF16, order, mtag_a_use, mtag_b_use, m_use, n_use,
        k_use, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c_use, cs_c_use,
        (void*)&alpha, (void*)&beta, post_op_list, lcntx_l.blksz.MR,
        lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_F16, &lcntx_l.dlp_kernel_hndl);

    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_JIT_KERNEL);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
    dlp_gemm_s8s8s32o32_openmp_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, (int32_t*)c_use, rs_c_use, cs_c_use,
        alpha, beta, &rntm_g, &lcntx_l, &ops, DLP_F16);
#else
    dlp_gemm_s8s8s32o32_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, (int32_t*)c_use, rs_c_use, cs_c_use,
        alpha, beta, &rntm_g, &lcntx_l, &ops, DLP_F16);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
